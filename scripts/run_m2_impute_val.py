"""WP4: load M2 checkpoint → impute fixed val batch → plots + metrics.

Prerequisites
-------------
- Processed windows from ``scripts/prepare_m2_data.py``.
- WP3 artifacts: ``artifacts/m2/checkpoints/best_score_net.eqx`` and
  sibling ``feature_scaler.npz`` (or pass ``--ckpt`` to another ``.eqx``
  that sits next to a matching scaler).

Example
-------
python scripts/run_m2_impute_val.py --config configs/real_m2.yaml \\
    --ckpt artifacts/m2/checkpoints/best_score_net.eqx
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.normalize import FeatureScaler, resolve_scaler_path
from src.data.real_config import RealM2Config, load_real_m2_config
from src.eval.metrics import EvalTensors, ImputationEvaluator, MetricsTable
from src.eval.m2_val_impute import (
    M2ValImputePaths,
    MetricsArtifactWriter,
    OriginalScaleRestorer,
    load_val_pool_with_scaler,
    metric_observation_mask,
)
from src.eval.plots import ImputationPlotData, save_imputation_bands
from src.models.score_net import (
    ConditionalScoreNet,
    default_score_net_config,
)
from src.sample.reverse_sde import (
    ImputeQuery,
    ReverseSDEImputer,
    context_from_real_config,
    grid_timestamps,
)
from src.sde.vp_sde import VPSDE
from src.train.artifacts import load_checkpoint, resolve_artifact_paths
from src.train.real_loss import (
    RealConditionalDSMLoss,
    RealDSMBatch,
    dsm_settings_from_real_config,
)

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "real_m2.yaml"
_PLOT_TITLE = "M2 imputation mean ± bands (original units)"
_TIME_AXIS = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for the M2 val impute runner.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed flags.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to real_m2 YAML config",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Score-net checkpoint (.eqx); default: artifacts/m2/.../best",
    )
    parser.add_argument(
        "--k-samples",
        type=int,
        default=None,
        help="Override sample.K_samples",
    )
    parser.add_argument(
        "--solver-steps",
        type=int,
        default=None,
        help="Override sample.solver_steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override data.batch_size for the fixed val batch",
    )
    return parser.parse_args(argv)


@dataclass(frozen=True)
class LoadedEvalAssets:
    """Checkpoint, scaler, and config ready for val imputation.

    Public methods
    --------------
    imputer
        Build a ``ReverseSDEImputer`` from the loaded model.
    scaler_path
        Path of the loaded feature scaler NPZ.

    Instance variables
    ------------------
    config : RealM2Config
        Validated (possibly CLI-overridden) config.
    model : ConditionalScoreNet
        Deserialized score network.
    scaler : FeatureScaler
        Train scaler from disk.
    ckpt_path : Path
        Resolved checkpoint path.
    """

    config: RealM2Config
    model: ConditionalScoreNet
    scaler: FeatureScaler
    ckpt_path: Path

    def imputer(self) -> ReverseSDEImputer:
        """Return a reverse-SDE imputer wired to this checkpoint.

        Returns
        -------
        ReverseSDEImputer
            Ready for ``impute``.
        """
        return ReverseSDEImputer(
            context_from_real_config(self.model, self.config)
        )

    def scaler_path(self) -> Path:
        """Return the scaler NPZ path beside the checkpoint.

        Returns
        -------
        pathlib.Path
            ``<ckpt_dir>/feature_scaler.npz``.
        """
        return resolve_scaler_path(self.ckpt_path.parent)


@dataclass(frozen=True)
class NormImputeResult:
    """Normalized-space impute outputs plus masks for WP4 scoring.

    Public methods
    --------------
    metric_mask
        Observation-style mask for artificial targets only.
    conditioning_mask
        Return ``m_train`` used for reverse-SDE conditioning.

    Instance variables
    ------------------
    batch : RealDSMBatch
        Normalized truth + native masks.
    x_obs : jax.Array
        Sentinel-filled observations under ``m_train``.
    m_train : jax.Array
        Composed conditioning mask.
    m_artificial : jax.Array
        Restricted artificial mask.
    samples : jax.Array
        Normalized posterior samples.
    mean : jax.Array
        Normalized posterior mean.
    quantiles : jax.Array
        Normalized quantile bands.
    """

    batch: RealDSMBatch
    x_obs: jnp.ndarray
    m_train: jnp.ndarray
    m_artificial: jnp.ndarray
    samples: jnp.ndarray
    mean: jnp.ndarray
    quantiles: jnp.ndarray

    def metric_mask(self) -> jnp.ndarray:
        """Return EvalTensors mask (0 = artificial target).

        Returns
        -------
        jax.Array
            Float32 observation-style metric mask.
        """
        return metric_observation_mask(
            self.batch.m_native,
            self.m_artificial,
        )

    def conditioning_mask(self) -> jnp.ndarray:
        """Return the reverse-SDE conditioning mask.

        Returns
        -------
        jax.Array
            ``m_train``.
        """
        return self.m_train


class M2ValImputeRunner:
    """Orchestrate load → mask → impute → inverse → plot → metrics.

    Public methods
    --------------
    run
        Execute the WP4 pipeline and return metrics as a dict.
    format_metrics
        Render a metrics mapping as a tiny text table.

    Instance variables
    ------------------
    assets : LoadedEvalAssets
        Checkpoint / scaler / config bundle.
    paths : M2ValImputePaths
        Output paths under ``artifacts/m2/``.
    """

    def __init__(self, assets: LoadedEvalAssets) -> None:
        """Attach loaded assets and resolve output paths.

        Parameters
        ----------
        assets : LoadedEvalAssets
            Validated eval assets.
        """
        self.assets = assets
        artifact_dir = assets.config.train.protocol.artifact_dir
        self.paths = M2ValImputePaths(
            resolve_artifact_paths(artifact_dir).root
        )

    def run(self, key: jax.Array) -> dict[str, float]:
        """Impute the fixed val batch and write WP4 artifacts.

        Parameters
        ----------
        key : jax.Array
            Root PRNG key.

        Returns
        -------
        dict of str to float
            Masked MAE / RMSE / CRPS / coverage in original units.
        """
        self.paths.ensure()
        key_mask, key_impute = jax.random.split(key)
        batch = self._fixed_val_batch()
        x_obs, m_train, m_art = self._compose_masks(key_mask, batch)
        samples_n, mean_n, quantiles_n = self._impute(
            x_obs, m_train, key_impute,
        )
        pack = NormImputeResult(
            batch=batch,
            x_obs=x_obs,
            m_train=m_train,
            m_artificial=m_art,
            samples=samples_n,
            mean=mean_n,
            quantiles=quantiles_n,
        )
        table = self._score_and_plot(pack)
        print(self.format_metrics(table.as_dict()))
        print(f"Checkpoint {self.assets.ckpt_path}")
        print(f"Scaler {self.assets.scaler_path()}")
        print(f"Figure {self.paths.imputation_bands()}")
        print(f"Metrics {self.paths.metrics_json()}")
        return table.as_dict()

    def format_metrics(self, metrics: Mapping[str, float]) -> str:
        """Format a metrics mapping as an aligned text table.

        Parameters
        ----------
        metrics : Mapping[str, float]
            Metric name to scalar.

        Returns
        -------
        str
            Multi-line table for stdout.
        """
        lines = ["metric          value", "--------------  ----------"]
        for name, value in metrics.items():
            lines.append(f"{name:<14}  {float(value):10.6f}")
        return "\n".join(lines)

    def _fixed_val_batch(self) -> RealDSMBatch:
        """Load the deterministic leading val batch (normalized).

        Returns
        -------
        RealDSMBatch
            Fixed val windows and native masks.
        """
        pool = load_val_pool_with_scaler(
            self.assets.config,
            self.assets.scaler,
        )
        return pool.fixed_val_batch(self.assets.config.data.batch_size)

    def _compose_masks(
            self,
            key: jax.Array,
            batch: RealDSMBatch,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Apply config mask_mode on natively observed cells.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        batch : RealDSMBatch
            Normalized val batch.

        Returns
        -------
        x_obs : jax.Array
            Sentinel-filled observations.
        m_train : jax.Array
            Composed conditioning mask.
        m_artificial : jax.Array
            Restricted artificial mask.
        """
        loss_fn = RealConditionalDSMLoss(
            VPSDE(self.assets.config.sde),
            dsm_settings_from_real_config(self.assets.config),
        )
        return loss_fn.sample_mask(key, batch)

    def _impute(
            self,
            x_obs: jnp.ndarray,
            m_train: jnp.ndarray,
            key: jax.Array,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Run reverse-SDE imputation in normalized space.

        Parameters
        ----------
        x_obs : jax.Array
            Observations ``(B, T, F)``.
        m_train : jax.Array
            Conditioning mask ``(B, T, F)``.
        key : jax.Array
            Imputation PRNG key.

        Returns
        -------
        samples : jax.Array
            Normalized samples ``(B, K, T, F)``.
        mean : jax.Array
            Normalized mean ``(B, T, F)``.
        quantiles : jax.Array
            Normalized bands ``(B, T, F, 2)``.
        """
        seq_len = int(x_obs.shape[_TIME_AXIS])
        return self.assets.imputer().impute(
            ImputeQuery(
                observations=x_obs,
                mask=m_train,
                timestamps=grid_timestamps(seq_len),
                key=key,
                k_samples=self.assets.config.sample.k_samples,
            )
        )

    def _score_and_plot(self, pack: NormImputeResult) -> MetricsTable:
        """Inverse-transform, plot bands, dump metrics, return table.

        Parameters
        ----------
        pack : NormImputeResult
            Normalized impute outputs and masks.

        Returns
        -------
        MetricsTable
            Original-unit metrics on artificial targets.
        """
        restorer = OriginalScaleRestorer(self.assets.scaler)
        m_native = pack.batch.m_native
        truth = restorer.restore_windows(pack.batch.x0, m_native)
        obs = restorer.restore_windows(pack.x_obs, pack.m_train)
        mean = restorer.restore_windows(pack.mean, m_native)
        samples = restorer.restore_samples(pack.samples, m_native)
        quantiles = restorer.restore_quantiles(pack.quantiles, m_native)
        metric_mask = pack.metric_mask()
        save_imputation_bands(
            ImputationPlotData(
                truth=truth,
                observations=obs,
                mean=mean,
                quantiles=quantiles,
                mask=metric_mask,
            ),
            self.paths.imputation_bands(),
            title=_PLOT_TITLE,
        )
        table = ImputationEvaluator().compute(
            EvalTensors(
                truth=truth,
                mean=mean,
                samples=samples,
                mask=metric_mask,
            )
        )
        writer = MetricsArtifactWriter()
        writer.write_json(self.paths.metrics_json(), table)
        writer.write_csv(self.paths.metrics_csv(), table)
        return table


def apply_cli_overrides(
        config: RealM2Config,
        args: argparse.Namespace,
) -> RealM2Config:
    """Apply optional CLI overrides onto a loaded config in-place.

    Parameters
    ----------
    config : RealM2Config
        Config to mutate.
    args : argparse.Namespace
        Parsed CLI flags.

    Returns
    -------
    RealM2Config
        The same config instance after overrides + re-validation.
    """
    if args.k_samples is not None:
        config.sample.k_samples = int(args.k_samples)
    if args.solver_steps is not None:
        config.sample.solver_steps = int(args.solver_steps)
    if args.batch_size is not None:
        config.data.batch_size = int(args.batch_size)
    config.validate()
    return config


def load_eval_assets(
        config: RealM2Config,
        ckpt_arg: Path | None,
) -> LoadedEvalAssets:
    """Resolve checkpoint + scaler and deserialize the score net.

    Parameters
    ----------
    config : RealM2Config
        Validated config.
    ckpt_arg : pathlib.Path or None
        Optional CLI checkpoint; default is best under artifact_dir.

    Returns
    -------
    LoadedEvalAssets
        Loaded model and scaler.
    """
    paths = resolve_artifact_paths(config.train.protocol.artifact_dir)
    ckpt_path = _resolve_ckpt(ckpt_arg, paths.best_checkpoint())
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            "checkpoint not found: "
            f"{ckpt_path}; run scripts/train_m2_real.py first"
        )
    scaler_path = resolve_scaler_path(ckpt_path.parent)
    if not scaler_path.is_file():
        raise FileNotFoundError(
            "feature scaler not found: "
            f"{scaler_path}; expected beside the checkpoint"
        )
    scaler = FeatureScaler.load(scaler_path)
    key = jax.random.PRNGKey(config.train.protocol.seed)
    template = ConditionalScoreNet(
        key,
        default_score_net_config(config.data.n_features),
    )
    model = load_checkpoint(ckpt_path, template)
    return LoadedEvalAssets(
        config=config,
        model=model,
        scaler=scaler,
        ckpt_path=ckpt_path,
    )


def _resolve_ckpt(ckpt_arg: Path | None, default_ckpt: Path) -> Path:
    """Resolve a checkpoint path against the repo root when relative.

    Parameters
    ----------
    ckpt_arg : pathlib.Path or None
        CLI path or ``None`` for the default best checkpoint.
    default_ckpt : pathlib.Path
        Fallback absolute best-checkpoint path.

    Returns
    -------
    pathlib.Path
        Absolute checkpoint path.
    """
    if ckpt_arg is None:
        return default_ckpt
    path = Path(ckpt_arg)
    if path.is_absolute():
        return path
    return _REPO_ROOT / path


def main(argv: list[str] | None = None) -> dict[str, float]:
    """CLI entry: run WP4 val impute and return metrics.

    Parameters
    ----------
    argv : list of str or None, optional
        Optional CLI args for tests.

    Returns
    -------
    dict of str to float
        Printed metrics table as a dict.
    """
    args = parse_args(argv)
    config = apply_cli_overrides(load_real_m2_config(args.config), args)
    assets = load_eval_assets(config, args.ckpt)
    key = jax.random.PRNGKey(config.train.protocol.seed)
    return M2ValImputeRunner(assets).run(key)


if __name__ == "__main__":
    main()
