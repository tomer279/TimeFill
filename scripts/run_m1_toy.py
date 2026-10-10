"""End-to-end M1 toy path: data → train → impute → figs → metrics (WP6).

Writes figures and the best checkpoint under ``artifacts/m1/``, then prints
a tiny masked metrics table (MAE / RMSE / CRPS / coverage).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.config import ToyM1Config, load_config
from src.data.masking import block_mask
from src.data.synthetic import generate_sine_windows
from src.eval.metrics import EvalTensors, ImputationEvaluator, MetricsTable
from src.eval.plots import (
    EvalPlotPaths,
    ImputationPlotData,
    save_forward_noising_plot,
    save_imputation_bands,
    save_mask_patterns_plot,
)
from src.models.score_net import (
    ConditionalScoreNet,
    default_score_net_config,
)
from src.sample.reverse_sde import (
    ImputeQuery,
    ReverseSDEImputer,
    context_from_toy_config,
    grid_timestamps,
)
from src.sde.vp_sde import VPSDE
from src.train.loss import ConditionalDSMLoss, dsm_settings_from_config
from src.train.artifacts import (
    resolve_artifact_paths,
    stub_val_imputation_hook,
)
from src.train.loop import (
    M1Trainer,
    TrainerBundle,
    build_optimizer,
)

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "toy_m1.yaml"
_EVAL_BATCH_SIZE = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for the M1 end-to-end toy runner.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed ``config`` path and optional overrides.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to toy_m1 YAML config",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override train.epochs for a shorter CPU smoke run",
    )
    parser.add_argument(
        "--k-samples",
        type=int,
        default=None,
        help="Override sample.K_samples for faster imputation",
    )
    parser.add_argument(
        "--solver-steps",
        type=int,
        default=None,
        help="Override sample.solver_steps for faster reverse SDE",
    )
    return parser.parse_args(argv)


class M1ToyRunner:
    """Orchestrate generate → train → impute → plots → metrics.

    Public methods
    --------------
    run
        Execute the full WP6 pipeline and return metrics as a dict.
    format_metrics
        Render a metrics mapping as a tiny text table.

    Instance variables
    ------------------
    config : ToyM1Config
        Validated (possibly CLI-overridden) toy config.
    paths : ArtifactPaths
        Training artifact helpers (checkpoint / loss curves).
    plot_paths : EvalPlotPaths
        WP6 figure paths under the same artifact root.
    """

    def __init__(self, config: ToyM1Config) -> None:
        """Attach config and resolve artifact / plot paths.

        Parameters
        ----------
        config : ToyM1Config
            Full M1 config; must already be validated.
        """
        config.validate()
        self.config = config
        self.paths = resolve_artifact_paths(config.train.protocol.artifact_dir)
        self.plot_paths = EvalPlotPaths(self.paths.root)

    def run(self, key: jax.Array) -> dict[str, float]:
        """Run data generation, training, imputation, and eval.

        Parameters
        ----------
        key : jax.Array
            Root PRNG key.

        Returns
        -------
        dict of str to float
            Masked MAE / RMSE / CRPS / coverage metrics.
        """
        self.paths.ensure()
        self.plot_paths.ensure()
        key_plot, key_train, key_eval = jax.random.split(key, 3)
        clean = self._make_clean_batch(key_plot, self.config.data.batch_size)
        self._save_diagnostic_plots(clean, key_plot)
        model = self._train(key_train)
        metrics = self._evaluate(model, key_eval)
        print(self.format_metrics(metrics.as_dict()))
        print(f"Checkpoint {self.paths.best_checkpoint()}")
        print(f"Figures under {self.plot_paths.root}")
        return metrics.as_dict()

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

    def _train(self, key: jax.Array) -> ConditionalScoreNet:
        """Train the score net and return the best-val checkpoint model.

        Parameters
        ----------
        key : jax.Array
            Training PRNG key.

        Returns
        -------
        ConditionalScoreNet
            Best validation model (also written to disk).
        """
        key_model, key_loop = jax.random.split(key)
        model = ConditionalScoreNet(
            key_model,
            default_score_net_config(self.config.data.n_features),
        )
        loss_fn = ConditionalDSMLoss(
            VPSDE(self.config.sde),
            dsm_settings_from_config(self.config),
        )
        bundle = TrainerBundle(
            config=self.config,
            model=model,
            loss_fn=loss_fn,
            optimizer=build_optimizer(self.config.train),
            paths=self.paths,
            val_hook=stub_val_imputation_hook,
        )
        best_model, history = M1Trainer(bundle.with_paths()).run(key_loop)
        if history.rows:
            first_loss = float(history.rows[0]["train_loss"])
            last_loss = float(history.rows[-1]["train_loss"])
            print(
                f"train_loss first={first_loss:.6f} last={last_loss:.6f}"
            )
        return best_model

    def _evaluate(
            self,
            model: ConditionalScoreNet,
            key: jax.Array,
    ) -> MetricsTable:
        """Mask held-out sine windows, impute, plot, and score.

        Parameters
        ----------
        model : ConditionalScoreNet
            Trained score network.
        key : jax.Array
            Eval PRNG key.

        Returns
        -------
        MetricsTable
            Masked imputation metrics.
        """
        key_data, key_mask, key_impute = jax.random.split(key, 3)
        truth = self._make_clean_batch(key_data, _EVAL_BATCH_SIZE)
        observations, mask = block_mask(
            truth,
            self.config.data.block_len,
            key_mask,
        )
        imputer = ReverseSDEImputer(
            context_from_toy_config(model, self.config)
        )
        samples, mean, quantiles = imputer.impute(
            ImputeQuery(
                observations=observations,
                mask=mask,
                timestamps=grid_timestamps(self.config.data.seq_len),
                key=key_impute,
                k_samples=self.config.sample.k_samples,
            )
        )
        save_imputation_bands(
            ImputationPlotData(
                truth=truth,
                observations=observations,
                mean=mean,
                quantiles=quantiles,
                mask=mask,
            ),
            self.plot_paths.imputation_bands(),
        )
        evaluator = ImputationEvaluator()
        return evaluator.compute(
            EvalTensors(
                truth=truth,
                mean=mean,
                samples=samples,
                mask=mask,
            )
        )

    def _save_diagnostic_plots(
            self,
            clean: jnp.ndarray,
            key: jax.Array,
    ) -> None:
        """Write forward-noising and mask-pattern figures.

        Parameters
        ----------
        clean : jax.Array
            Clean batch used for diagnostics.
        key : jax.Array
            PRNG key (split internally).
        """
        key_fwd, key_mask = jax.random.split(key)
        save_forward_noising_plot(
            clean,
            self.plot_paths.forward_noising(),
            VPSDE(self.config.sde),
            key_fwd,
        )
        save_mask_patterns_plot(
            clean,
            self.plot_paths.mask_patterns(),
            key_mask,
            self.config.data.mask_rate,
            self.config.data.block_len,
        )

    def _make_clean_batch(
            self,
            key: jax.Array,
            batch_size: int,
    ) -> jnp.ndarray:
        """Draw a clean sine batch with config geometry.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        batch_size : int
            Batch size ``B``.

        Returns
        -------
        jax.Array
            Clean ``x0`` shaped ``(B, T, F)``.
        """
        shape = (
            batch_size,
            self.config.data.seq_len,
            self.config.data.n_features,
        )
        return generate_sine_windows(key, shape)


def apply_cli_overrides(
        config: ToyM1Config,
        args: argparse.Namespace,
) -> ToyM1Config:
    """Apply optional CLI overrides onto a loaded config in-place.

    Parameters
    ----------
    config : ToyM1Config
        Config to mutate.
    args : argparse.Namespace
        Parsed CLI flags.

    Returns
    -------
    ToyM1Config
        The same config instance after overrides + re-validation.
    """
    if args.epochs is not None:
        config.train.schedule.epochs = int(args.epochs)
    if args.k_samples is not None:
        config.sample.k_samples = int(args.k_samples)
    if args.solver_steps is not None:
        config.sample.solver_steps = int(args.solver_steps)
    config.validate()
    return config


def main(argv: list[str] | None = None) -> dict[str, float]:
    """CLI entry: run the M1 toy pipeline and return metrics.

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
    config = apply_cli_overrides(load_config(args.config), args)
    key = jax.random.PRNGKey(config.train.protocol.seed)
    return M1ToyRunner(config).run(key)


if __name__ == "__main__":
    main()
