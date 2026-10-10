"""Jitted Optax training loop, checkpoints, and logging for M1 (WP4).

Classes
-------
ArtifactPaths
    Resolve checkpoint / curve / val-snapshot paths under artifacts.
TrainHistory
    Collect epoch metrics and export CSV / Matplotlib curves.
ValSnapshotPayload
    Packed inputs for the periodic val snapshot hook.
TrainerBundle
    Dependencies for ``M1Trainer`` (keeps constructor arity small).
M1Trainer
    AdamW loop with NaN checks, grad-norm logs, best-val checkpoints.

Functions
---------
build_optimizer
    Build Optax AdamW (+ optional grad clip) from ``TrainConfig``.
save_checkpoint
    Serialize an Equinox model with ``eqx.tree_serialise_leaves``.
load_checkpoint
    Deserialize into a like-shaped Equinox template.
resolve_artifact_paths
    Build ``ArtifactPaths`` from a config ``artifact_dir`` string.
stub_val_imputation_hook
    WP5 stub: write a val snapshot marker (no reverse SDE).
run_m1_training
    Load config, train on sine windows, write artifacts.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from src.data.config import ToyM1Config, TrainConfig, load_config
from src.data.synthetic import generate_sine_windows
from src.models.score_net import (
    ConditionalScoreNet,
    default_score_net_config,
)
from src.plot_io import save_figure
from src.sde.vp_sde import VPSDE
from src.train.loss import (
    ConditionalDSMLoss,
    dsm_settings_from_config,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BEST_CKPT_NAME = "best_score_net.eqx"
_LOSS_CSV_NAME = "loss_curves.csv"
_LOSS_PNG_NAME = "loss_curves.png"
_VAL_SUBDIR = "val_snapshots"
_CKPT_SUBDIR = "checkpoints"
_NAN_MSG = "non-finite train loss detected; aborting"
_GRAD_NORM_EPS = 0.0
_VAL_MC_DRAWS = 4


@dataclass(frozen=True)
class ArtifactPaths:
    """Filesystem layout for M1 training artifacts.

    Public methods
    --------------
    ensure
        Create checkpoint and val-snapshot directories.
    checkpoint_dir
        Return ``<root>/checkpoints``.
    best_checkpoint
        Return path to the best-val score-net file.
    loss_csv
        Return path to the loss-history CSV.
    loss_png
        Return path to the loss-curve PNG.
    val_dir
        Return ``<root>/val_snapshots``.

    Instance variables
    ------------------
    root : pathlib.Path
        Artifact root (typically ``artifacts/m1``).
    """

    root: Path

    def ensure(self) -> None:
        """Create checkpoint and validation snapshot directories."""
        self.checkpoint_dir().mkdir(parents=True, exist_ok=True)
        self.val_dir().mkdir(parents=True, exist_ok=True)

    def checkpoint_dir(self) -> Path:
        """Return the checkpoint directory path.

        Returns
        -------
        pathlib.Path
            ``root / checkpoints``.
        """
        return self.root / _CKPT_SUBDIR

    def best_checkpoint(self) -> Path:
        """Return the best-validation checkpoint path.

        Returns
        -------
        pathlib.Path
            ``.../checkpoints/best_score_net.eqx``.
        """
        return self.checkpoint_dir() / _BEST_CKPT_NAME

    def loss_csv(self) -> Path:
        """Return the loss-history CSV path.

        Returns
        -------
        pathlib.Path
            ``root / loss_curves.csv``.
        """
        return self.root / _LOSS_CSV_NAME

    def loss_png(self) -> Path:
        """Return the loss-curve plot path.

        Returns
        -------
        pathlib.Path
            ``root / loss_curves.png``.
        """
        return self.root / _LOSS_PNG_NAME

    def val_dir(self) -> Path:
        """Return the validation snapshot directory.

        Returns
        -------
        pathlib.Path
            ``root / val_snapshots``.
        """
        return self.root / _VAL_SUBDIR


class TrainHistory:
    """In-memory table of per-epoch training metrics.

    Public methods
    --------------
    append_row
        Append one metric mapping.
    save_csv
        Write all rows to a CSV file.
    save_plot
        Save train/val loss curves with Matplotlib.

    Instance variables
    ------------------
    rows : list of dict
        Metric rows with keys such as ``epoch``, ``train_loss``.
    """

    def __init__(self) -> None:
        """Initialize an empty metric table."""
        self.rows: list[dict[str, Any]] = []

    def append_row(self, row: Mapping[str, Any]) -> None:
        """Append a metric row.

        Parameters
        ----------
        row : Mapping[str, Any]
            Metric name to scalar / string value.
        """
        self.rows.append(dict(row))

    def save_csv(self, path: Path) -> None:
        """Write the history to CSV.

        Parameters
        ----------
        path : pathlib.Path
            Destination CSV path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames = list(self.rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

    def save_plot(self, path: Path) -> None:
        """Plot train/val loss versus epoch.

        Parameters
        ----------
        path : pathlib.Path
            Destination PNG path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            return
        epochs = [int(row["epoch"]) for row in self.rows]
        train_y = [float(row["train_loss"]) for row in self.rows]
        val_y = [float(row["val_loss"]) for row in self.rows]
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.plot(epochs, train_y, label="train")
        axis.plot(epochs, val_y, label="val")
        axis.set_xlabel("epoch")
        axis.set_ylabel("DSM loss")
        axis.set_title("M1 conditional DSM")
        axis.legend()
        fig.tight_layout()
        save_figure(fig, path)
        plt.close(fig)


@dataclass(frozen=True)
class ValSnapshotPayload:
    """Inputs for the periodic validation snapshot hook.

    Public methods
    --------------
    marker_path
        Path for the WP5 stub marker at this step.
    summary_line
        One-line human-readable summary for the stub file.

    Instance variables
    ------------------
    model : ConditionalScoreNet
        Current score network (unused by the WP5 stub).
    x0 : jax.Array
        Clean validation windows.
    mask : jax.Array
        Validation observation mask.
    x_obs : jax.Array
        Sentinel-filled observations.
    step : int
        Epoch index used as the snapshot label.
    out_dir : pathlib.Path
        Directory for snapshot artifacts.
    """

    model: ConditionalScoreNet
    x0: jnp.ndarray
    mask: jnp.ndarray
    x_obs: jnp.ndarray
    step: int
    out_dir: Path

    def marker_path(self) -> Path:
        """Return the stub marker path for this step.

        Returns
        -------
        pathlib.Path
            ``out_dir / val_stub_step_{step}.txt``.
        """
        return self.out_dir / f"val_stub_step_{self.step}.txt"

    def summary_line(self) -> str:
        """Build a short stub summary string.

        Returns
        -------
        str
            Shape summary noting that WP5 impute is not wired yet.
        """
        return (
            f"step={self.step} x0_shape={tuple(self.x0.shape)} "
            f"mask_shape={tuple(self.mask.shape)} "
            "WP5 reverse impute not implemented; stub only"
        )


def stub_val_imputation_hook(payload: ValSnapshotPayload) -> Path:
    """Write a WP5-pending validation snapshot marker.

    Does **not** run Diffrax reverse sampling. WP5 should replace this
    hook with a real imputation mean ± band figure.

    Parameters
    ----------
    payload : ValSnapshotPayload
        Model, val tensors, step, and output directory.

    Returns
    -------
    pathlib.Path
        Path to the written stub marker file.
    """
    payload.out_dir.mkdir(parents=True, exist_ok=True)
    path = payload.marker_path()
    path.write_text(payload.summary_line() + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True)
class TrainerBundle:
    """Bundled dependencies for ``M1Trainer`` (keeps ctor arity small).

    Public methods
    --------------
    validate
        Check that artifact root is non-empty and config validates.
    with_paths
        Ensure artifact directories exist; return self.

    Instance variables
    ------------------
    config : ToyM1Config
        Full config.
    model : ConditionalScoreNet
        Initial score network.
    loss_fn : ConditionalDSMLoss
        DSM loss.
    optimizer : optax.GradientTransformation
        Optax optimizer.
    paths : ArtifactPaths
        Artifact paths.
    val_hook : callable
        Validation snapshot hook.
    """

    config: ToyM1Config
    model: ConditionalScoreNet
    loss_fn: ConditionalDSMLoss
    optimizer: optax.GradientTransformation
    paths: ArtifactPaths
    val_hook: Callable[[ValSnapshotPayload], Path]

    def validate(self) -> None:
        """Raise ``ValueError`` if the bundle is inconsistent."""
        if not str(self.paths.root):
            raise ValueError("ArtifactPaths.root must be non-empty")
        self.config.validate()

    def with_paths(self) -> "TrainerBundle":
        """Ensure artifact directories exist and return this bundle.

        Returns
        -------
        TrainerBundle
            The same bundle instance.
        """
        self.paths.ensure()
        return self


@dataclass(frozen=True)
class _EpochKeyPack:
    """PRNG keys for one training epoch.

    Public methods
    --------------
    n_train_steps
        Number of mini-batch updates in this pack.
    data_key_at
        Data PRNG key for a given step index.

    Instance variables
    ------------------
    key_val : jax.Array
        Key for validation MC loss.
    step_keys : jax.Array
        Stacked per-step train keys.
    data_keys : jax.Array
        Stacked per-batch data keys.
    """

    key_val: jax.Array
    step_keys: jax.Array
    data_keys: jax.Array

    def n_train_steps(self) -> int:
        """Return the number of train steps in this pack.

        Returns
        -------
        int
            ``step_keys.shape[0]``.
        """
        return int(self.step_keys.shape[0])

    def data_key_at(self, step_idx: int) -> jax.Array:
        """Return the data key at ``step_idx``.

        Parameters
        ----------
        step_idx : int
            Mini-batch index.

        Returns
        -------
        jax.Array
            PRNG key for sine batch generation.
        """
        return self.data_keys[step_idx]


@dataclass(frozen=True)
class _EpochInputs:
    """Packed mutable-train inputs for one epoch (arity helper).

    Public methods
    --------------
    split_keys
        Split the epoch key into a ``_EpochKeyPack``.
    n_steps
        Steps per epoch from train config.

    Instance variables
    ------------------
    model : ConditionalScoreNet
        Parameters at epoch start.
    opt_state : optax.OptState
        Optimizer state at epoch start.
    key : jax.Array
        Epoch PRNG key.
    val_x0 : jax.Array
        Fixed validation clean batch.
    epoch : int
        1-based epoch index.
    """

    model: ConditionalScoreNet
    opt_state: optax.OptState
    key: jax.Array
    val_x0: jnp.ndarray
    epoch: int

    def n_steps(self, train_cfg: TrainConfig) -> int:
        """Return mini-batches per epoch.

        Parameters
        ----------
        train_cfg : TrainConfig
            Train hyperparameters.

        Returns
        -------
        int
            ``steps_per_epoch``.
        """
        return int(train_cfg.steps_per_epoch)

    def split_keys(self, n_steps: int) -> _EpochKeyPack:
        """Split into val / step / data keys.

        Parameters
        ----------
        n_steps : int
            Number of train steps this epoch.

        Returns
        -------
        _EpochKeyPack
            Packed PRNG keys for the epoch.
        """
        keys = jax.random.split(self.key, n_steps + 2)
        return _EpochKeyPack(
            key_val=keys[0],
            step_keys=keys[1:n_steps + 1],
            data_keys=jax.random.split(keys[n_steps + 1], n_steps),
        )


class M1Trainer:
    """Train a conditional score net with jitted AdamW updates.

    Public methods
    --------------
    run
        Full train/val loop with checkpointing and curve export.
    evaluate
        Mean DSM loss on a fixed clean batch (multiple MC draws).

    Instance variables
    ------------------
    config : ToyM1Config
        Full M1 config (data / sde / train).
    model : ConditionalScoreNet
        Score network being optimized.
    loss_fn : ConditionalDSMLoss
        Conditional DSM loss module.
    optimizer : optax.GradientTransformation
        Optax optimizer (AdamW ± grad clip).
    paths : ArtifactPaths
        Artifact directory helpers.
    val_hook : callable
        ``ValSnapshotPayload -> Path`` snapshot hook.
    """

    def __init__(self, bundle: TrainerBundle) -> None:
        """Attach a pre-built trainer bundle.

        Parameters
        ----------
        bundle : TrainerBundle
            Model, loss, optimizer, config, paths, and val hook.
        """
        self.config = bundle.config
        self.model = bundle.model
        self.loss_fn = bundle.loss_fn
        self.optimizer = bundle.optimizer
        self.paths = bundle.paths
        self.val_hook = bundle.val_hook

    def run(self, key: jax.Array) -> tuple[ConditionalScoreNet, TrainHistory]:
        """Run the M1 toy training loop on synthetic sine windows.

        Parameters
        ----------
        key : jax.Array
            Root PRNG key.

        Returns
        -------
        model : ConditionalScoreNet
            Best-val score network.
        history : TrainHistory
            Logged train/val metrics.
        """
        self.paths.ensure()
        train_cfg = self.config.train
        key, key_val, key_loop = jax.random.split(key, 3)
        val_x0 = _make_sine_batch(key_val, self.config.data)
        opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
        step_fn = _build_train_step(self.optimizer, self.loss_fn)
        history = TrainHistory()
        best_val = float("inf")
        best_model = self.model
        model = self.model

        for epoch in range(1, train_cfg.epochs + 1):
            key_loop, key_epoch = jax.random.split(key_loop)
            epoch_in = _EpochInputs(
                model,
                opt_state,
                key_epoch,
                val_x0,
                epoch,
            )
            model, opt_state, row = self._run_epoch(step_fn, epoch_in)
            history.append_row(row)
            val_loss = float(row["val_loss"])
            if val_loss < best_val:
                best_val = val_loss
                best_model = model
                save_checkpoint(self.paths.best_checkpoint(), best_model)
            if epoch % train_cfg.val_every == 0:
                self._fire_val_hook(model, val_x0, epoch)

        history.save_csv(self.paths.loss_csv())
        history.save_plot(self.paths.loss_png())
        self.model = best_model
        return best_model, history

    def evaluate(
            self,
            model: ConditionalScoreNet,
            x0: jnp.ndarray,
            key: jax.Array,
    ) -> jnp.ndarray:
        """Estimate mean DSM loss with a few Monte Carlo draws.

        Parameters
        ----------
        model : ConditionalScoreNet
            Score network to evaluate.
        x0 : jax.Array
            Clean evaluation batch.
        key : jax.Array
            PRNG key.

        Returns
        -------
        jax.Array
            Scalar mean loss over internal MC draws.
        """
        keys = jax.random.split(key, _VAL_MC_DRAWS)
        losses = [
            self.loss_fn(model, x0, keys[i]) for i in range(_VAL_MC_DRAWS)
        ]
        return jnp.mean(jnp.stack(losses))

    def _run_epoch(
            self,
            step_fn: Callable,
            inputs: _EpochInputs,
    ) -> tuple[ConditionalScoreNet, optax.OptState, dict[str, Any]]:
        """Execute one epoch of train steps and one val evaluation.

        Parameters
        ----------
        step_fn : callable
            Jitted train step.
        inputs : _EpochInputs
            Model, opt state, keys, val batch, epoch index.

        Returns
        -------
        model : ConditionalScoreNet
            Updated model.
        opt_state : optax.OptState
            Updated optimizer state.
        row : dict of str to Any
            Epoch metrics for logging.
        """
        train_cfg = self.config.train
        n_steps = inputs.n_steps(train_cfg)
        key_pack = inputs.split_keys(n_steps)
        model, opt_state, train_loss, grad_mean = self._accumulate_steps(
            step_fn,
            inputs,
            key_pack,
        )
        val_loss = float(
            self.evaluate(model, inputs.val_x0, key_pack.key_val)
        )
        row = {
            "epoch": inputs.epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "grad_norm": grad_mean,
        }
        if inputs.epoch % train_cfg.log_every == 0:
            print(
                f"epoch={inputs.epoch} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} grad_norm={grad_mean:.6f}"
            )
        return model, opt_state, row

    def _accumulate_steps(
            self,
            step_fn: Callable,
            inputs: _EpochInputs,
            key_pack: _EpochKeyPack,
    ) -> tuple[ConditionalScoreNet, optax.OptState, float, float]:
        """Run all mini-batch updates for one epoch.

        Parameters
        ----------
        step_fn : callable
            Jitted train step.
        inputs : _EpochInputs
            Epoch start model / optimizer state.
        key_pack : _EpochKeyPack
            Per-step train and data PRNG keys.

        Returns
        -------
        model : ConditionalScoreNet
            Updated model.
        opt_state : optax.OptState
            Updated optimizer state.
        train_loss : float
            Mean train loss over steps.
        grad_mean : float
            Mean grad norm over steps.
        """
        n_steps = key_pack.n_train_steps()
        model = inputs.model
        opt_state = inputs.opt_state
        loss_sum = 0.0
        grad_sum = 0.0
        for step_idx in range(n_steps):
            x0 = _make_sine_batch(
                key_pack.data_key_at(step_idx),
                self.config.data,
            )
            model, opt_state, loss, grad_norm = step_fn(
                model,
                opt_state,
                x0,
                key_pack.step_keys[step_idx],
            )
            loss_host = float(loss)
            if not jnp.isfinite(loss):
                raise RuntimeError(_NAN_MSG)
            loss_sum += loss_host
            grad_sum += float(grad_norm)
        inv_steps = 1.0 / float(n_steps)
        return model, opt_state, loss_sum * inv_steps, grad_sum * inv_steps

    def _fire_val_hook(
            self,
            model: ConditionalScoreNet,
            val_x0: jnp.ndarray,
            epoch: int,
    ) -> None:
        """Sample a val mask and invoke the snapshot hook.

        Parameters
        ----------
        model : ConditionalScoreNet
            Current score network.
        val_x0 : jax.Array
            Clean validation batch.
        epoch : int
            Epoch index used as the snapshot step label.
        """
        key = jax.random.PRNGKey(epoch)
        x_obs, mask = self.loss_fn.sample_mask(key, val_x0)
        payload = ValSnapshotPayload(
            model=model,
            x0=val_x0,
            mask=mask,
            x_obs=x_obs,
            step=epoch,
            out_dir=self.paths.val_dir(),
        )
        self.val_hook(payload)


def build_optimizer(train_cfg: TrainConfig) -> optax.GradientTransformation:
    """Build AdamW with optional global gradient clipping.

    Parameters
    ----------
    train_cfg : TrainConfig
        Provides ``lr``, ``weight_decay``, and ``grad_clip_norm``.

    Returns
    -------
    optax.GradientTransformation
        Optax optimizer chain.
    """
    adam = optax.adamw(
        train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    if train_cfg.grad_clip_norm > _GRAD_NORM_EPS:
        return optax.chain(
            optax.clip_by_global_norm(train_cfg.grad_clip_norm),
            adam,
        )
    return adam


def save_checkpoint(path: Path, model: ConditionalScoreNet) -> None:
    """Serialize an Equinox score network to ``path``.

    Parameters
    ----------
    path : pathlib.Path
        Destination file (``.eqx``).
    model : ConditionalScoreNet
        Model to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, model)


def load_checkpoint(
        path: Path,
        like: ConditionalScoreNet,
) -> ConditionalScoreNet:
    """Load a score network checkpoint into a template module.

    Parameters
    ----------
    path : pathlib.Path
        Checkpoint file written by ``save_checkpoint``.
    like : ConditionalScoreNet
        Template with the same architecture / static fields.

    Returns
    -------
    ConditionalScoreNet
        Restored module.
    """
    return eqx.tree_deserialise_leaves(path, like)


def resolve_artifact_paths(artifact_dir: str) -> ArtifactPaths:
    """Build ``ArtifactPaths`` from a config ``artifact_dir`` string.

    Parameters
    ----------
    artifact_dir : str
        Absolute or repo-relative artifact root.

    Returns
    -------
    ArtifactPaths
        Resolved paths helper.
    """
    return ArtifactPaths(_resolve_artifact_root(artifact_dir))


def run_m1_training(config_path: Path | str) -> TrainHistory:
    """Train on sine windows using a YAML config path.

    Parameters
    ----------
    config_path : path-like
        Path to ``configs/toy_m1.yaml`` (or compatible).

    Returns
    -------
    TrainHistory
        Logged metrics (CSV/PNG already written under artifact_dir).
    """
    config = load_config(config_path)
    paths = resolve_artifact_paths(config.train.artifact_dir)
    key = jax.random.PRNGKey(config.train.seed)
    key_model, key_train = jax.random.split(key)
    model = ConditionalScoreNet(
        key_model,
        default_score_net_config(config.data.n_features),
    )
    loss_fn = ConditionalDSMLoss(
        VPSDE(config.sde),
        dsm_settings_from_config(config),
    )
    bundle = TrainerBundle(
        config=config,
        model=model,
        loss_fn=loss_fn,
        optimizer=build_optimizer(config.train),
        paths=paths,
        val_hook=stub_val_imputation_hook,
    )
    bundle.validate()
    trainer = M1Trainer(bundle.with_paths())
    _, history = trainer.run(key_train)
    return history


def _build_train_step(
        optimizer: optax.GradientTransformation,
        loss_fn: ConditionalDSMLoss,
) -> Callable:
    """Create a jitted train step closure.

    Parameters
    ----------
    optimizer : optax.GradientTransformation
        Optax optimizer.
    loss_fn : ConditionalDSMLoss
        DSM loss module closed over by the step.

    Returns
    -------
    callable
        ``(model, opt_state, x0, key) -> (model, opt_state, loss, grad_norm)``.
    """

    @eqx.filter_jit
    def _step(
            model: ConditionalScoreNet,
            opt_state: optax.OptState,
            x0: jnp.ndarray,
            key: jax.Array,
    ) -> tuple[
        ConditionalScoreNet,
        optax.OptState,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        def _loss_fn(score_model: ConditionalScoreNet) -> jnp.ndarray:
            return loss_fn(score_model, x0, key)

        loss, grads = eqx.filter_value_and_grad(_loss_fn)(model)
        grad_norm = _tree_grad_norm(grads)
        updates, new_state = optimizer.update(
            grads,
            opt_state,
            eqx.filter(model, eqx.is_array),
        )
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_state, loss, grad_norm

    return _step


def _tree_grad_norm(grads: Any) -> jnp.ndarray:
    """Compute the global L2 norm of a pytree of gradients.

    Parameters
    ----------
    grads : Any
        Gradient pytree (Equinox-filtered).

    Returns
    -------
    jax.Array
        Scalar grad norm.
    """
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    total = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in leaves:
        total = total + jnp.sum(leaf.astype(jnp.float32) ** 2)
    return jnp.sqrt(total)


def _make_sine_batch(key: jax.Array, data_cfg: Any) -> jnp.ndarray:
    """Draw one clean sine batch from data-config geometry.

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    data_cfg : Any
        Object with ``batch_size``, ``seq_len``, ``n_features``.

    Returns
    -------
    jax.Array
        Clean ``x0`` shaped ``(B, T, F)``.
    """
    shape = (
        data_cfg.batch_size,
        data_cfg.seq_len,
        data_cfg.n_features,
    )
    return generate_sine_windows(key, shape)


def _resolve_artifact_root(artifact_dir: str) -> Path:
    """Resolve artifact_dir relative to the repo root when needed.

    Parameters
    ----------
    artifact_dir : str
        Absolute path or repo-relative path string.

    Returns
    -------
    pathlib.Path
        Absolute artifact root.
    """
    path = Path(artifact_dir)
    if path.is_absolute():
        return path
    return _REPO_ROOT / path
