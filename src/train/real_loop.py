"""Jitted Optax training loop for M2 real windows (WP3).

Classes
-------
M2TrainerBundle
    Dependencies for ``M2Trainer`` (keeps constructor arity small).
M2Trainer
    AdamW loop with NaN checks, grad-norm logs, best-val checkpoints.
RealBatchPool
    Re-exported from ``src.train.real_pool``.
M2ValHookBundle
    Re-exported from ``src.train.real_val``.
FixedValImputeSet
    Re-exported from ``src.train.real_val_select``.

Functions
---------
run_m2_training
    Load processed windows, fit scaler, train, write artifacts.
run_mask_rate_sensitivity
    Optional DSM val-loss smoke at several artificial mask rates.
build_real_train_step
    Re-exported from ``src.train.real_setup``.
build_fixed_val_impute_set
    Re-exported from ``src.train.real_val_select``.
init_ema
    Re-exported from ``src.train.real_val_select``.
update_ema
    Re-exported from ``src.train.real_val_select``.
load_normalized_pool
    Re-exported from ``src.train.real_pool``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.data.real_config import RealM2Config, load_real_m2_config
from src.data.schema import TIME_AXIS
from src.models.score_net import (
    ConditionalScoreNet,
    default_score_net_config,
    receptive_field,
)
from src.sample.reverse_sde import ImputeSolverSettings, resolve_sample_t_eps
from src.sde.vp_sde import VPSDE
from src.train.artifacts import (
    ArtifactPaths,
    TrainHistory,
    ValSnapshotPayload,
    save_checkpoint,
    write_train_meta,
)
from src.train.loop import build_optimizer
from src.train.real_loss import (
    RealConditionalDSMLoss,
    RealDSMBatch,
    dsm_settings_from_real_config,
)
from src.train.real_pool import RealBatchPool, load_normalized_pool
from src.train.real_setup import (
    assert_not_frozen_artifact_dir as _assert_not_frozen_artifact_dir,
    assert_wp0_consumable as _assert_wp0_consumable,
    build_real_train_step,
    prepare_artifacts_and_pool as _prepare_artifacts_and_pool,
)
from src.train.real_val import M2ValHookBundle
from src.train.real_val_select import (
    SELECT_CRPS,
    SELECT_DSM,
    SELECT_MAE,
    FixedValImputeSet,
    ValImputeBundle,
    build_fixed_val_impute_set,
    init_ema,
    make_val_impute_runner,
    score_val_impute,
    update_ema,
)

_NAN_MSG = "non-finite train loss detected; aborting"
_VAL_MC_DRAWS = 4
_LOSS_PLOT_TITLE = "M2 conditional DSM (real windows)"
_DEFAULT_VAL_IMPUTE = {
    "n_windows": 64,
    "K_samples": 8,
    "solver_steps": 50,
    "mask_seed": 0,
    "sample_seed": 1,
    "protocols": ("mcar_0.3", "block_8"),
}
_GRAD_NORM_EPS = 1.0e-12

__all__ = [
    "FixedValImputeSet",
    "M2Trainer",
    "M2TrainerBundle",
    "M2ValHookBundle",
    "RealBatchPool",
    "build_fixed_val_impute_set",
    "build_real_train_step",
    "init_ema",
    "load_normalized_pool",
    "run_m2_training",
    "run_m2_training_from_config",
    "run_mask_rate_sensitivity",
    "update_ema",
]

@dataclass(frozen=True)
class M2TrainerBundle:
    """Bundled dependencies for ``M2Trainer``.

    Public methods
    --------------
    validate
        Check artifact root and config.
    with_paths
        Ensure artifact directories exist; return self.

    Instance variables
    ------------------
    config : RealM2Config
        Full real-m2 config.
    model : ConditionalScoreNet
        Initial score network.
    loss_fn : RealConditionalDSMLoss
        Real DSM loss.
    optimizer : optax.GradientTransformation
        Optax optimizer.
    paths : ArtifactPaths
        Artifact paths under ``artifacts/m2``.
    pool : RealBatchPool
        Normalized train/val pools.
    val_hook : callable
        Validation snapshot hook.
    """

    config: RealM2Config
    model: ConditionalScoreNet
    loss_fn: RealConditionalDSMLoss
    optimizer: optax.GradientTransformation
    paths: ArtifactPaths
    pool: RealBatchPool
    val_hook: Callable[[ValSnapshotPayload], Path]

    def validate(self) -> None:
        """Raise ``ValueError`` if the bundle is inconsistent."""
        if not str(self.paths.root):
            raise ValueError("ArtifactPaths.root must be non-empty")
        self.config.validate()

    def with_paths(self) -> "M2TrainerBundle":
        """Ensure artifact directories exist and return this bundle.

        Returns
        -------
        M2TrainerBundle
            The same bundle instance.
        """
        self.paths.ensure()
        return self


@dataclass
class _StepTotals:
    """Running sums of train loss / grad norm / clip hits within an epoch.

    Public methods
    --------------
    add
        Accumulate one step's loss, grad norm, and clip flag.
    means
        Return mean loss and mean grad norm over ``n_steps``.

    Instance variables
    ------------------
    loss_sum : float
        Sum of per-step losses.
    grad_sum : float
        Sum of per-step grad norms.
    n_clipped : int
        Steps whose pre-clip grad norm exceeded the clip threshold.
    grad_norms : list of float
        Per-step grad norms (for median across the run).
    """

    loss_sum: float = 0.0
    grad_sum: float = 0.0
    n_clipped: int = 0
    grad_norms: list[float] = field(default_factory=list)

    def add(
            self,
            loss: float,
            grad_norm: float,
            clipped: bool,
    ) -> None:
        """Accumulate one step's metrics.

        Parameters
        ----------
        loss : float
            Scalar train loss for the step.
        grad_norm : float
            Global grad norm for the step.
        clipped : bool
            Whether the step exceeded the clip threshold.
        """
        self.loss_sum += loss
        self.grad_sum += grad_norm
        self.grad_norms.append(grad_norm)
        if clipped:
            self.n_clipped += 1

    def means(self, n_steps: int) -> tuple[float, float]:
        """Return mean loss and mean grad norm.

        Parameters
        ----------
        n_steps : int
            Positive number of steps in the epoch.

        Returns
        -------
        train_loss : float
            Mean train loss.
        grad_mean : float
            Mean grad norm.
        """
        inv_steps = 1.0 / float(n_steps)
        return self.loss_sum * inv_steps, self.grad_sum * inv_steps


@dataclass
class _M2BestTracker:
    """Best-val checkpoint tracker and epoch history.

    Public methods
    --------------
    append_row
        Append one epoch metrics row to ``history``.
    consider
        Update best model when validation loss improves.

    Instance variables
    ------------------
    history : TrainHistory
        Epoch metric log.
    best_val : float
        Best validation loss so far.
    best_model : ConditionalScoreNet
        Model weights at ``best_val``.
    """

    history: TrainHistory
    best_val: float
    best_model: ConditionalScoreNet

    def append_row(self, row: Mapping[str, Any]) -> None:
        """Append one epoch metrics row.

        Parameters
        ----------
        row : mapping of str to Any
            Epoch metrics for CSV / plotting.
        """
        self.history.append_row(row)

    def consider(self, model: ConditionalScoreNet, val_loss: float) -> bool:
        """Keep ``model`` if ``val_loss`` improves on ``best_val``.

        Parameters
        ----------
        model : ConditionalScoreNet
            Candidate score network.
        val_loss : float
            Validation DSM loss for this epoch.

        Returns
        -------
        bool
            ``True`` if this candidate became the new best.
        """
        improved = val_loss < self.best_val
        if improved:
            self.best_val = val_loss
            self.best_model = model
        return improved


@dataclass(frozen=True)
class _M2EpochKeyPack:
    """PRNG keys for one M2 training epoch.

    Public methods
    --------------
    n_train_steps
        Number of mini-batch updates in this pack.
    split_from
        Build a pack by splitting an epoch key.

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
    step_keys: jnp.ndarray
    data_keys: jnp.ndarray

    def n_train_steps(self) -> int:
        """Return the number of train steps in this pack.

        Returns
        -------
        int
            ``step_keys.shape[0]``.
        """
        return int(self.step_keys.shape[0])

    @staticmethod
    def split_from(key: jax.Array, n_steps: int) -> "_M2EpochKeyPack":
        """Split an epoch key into val / step / data streams.

        Parameters
        ----------
        key : jax.Array
            Epoch PRNG key.
        n_steps : int
            Mini-batches this epoch.

        Returns
        -------
        _M2EpochKeyPack
            Packed PRNG keys.
        """
        keys = jax.random.split(key, n_steps + 2)
        return _M2EpochKeyPack(
            key_val=keys[0],
            step_keys=keys[1:n_steps + 1],
            data_keys=jax.random.split(keys[n_steps + 1], n_steps),
        )


@dataclass
class _M2LoopState:
    """Mutable state across M2 training epochs.

    Public methods
    --------------
    take_epoch_key
        Split and return the next epoch PRNG key.
    store_epoch
        Store model/opt/EMA and append history (no selection).

    Instance variables
    ------------------
    model : ConditionalScoreNet
        Current score network.
    opt_state : optax.OptState
        Current Optax state.
    key_loop : jax.Array
        PRNG stream for epoch splits.
    val_batch : RealDSMBatch
        Fixed validation batch.
    tracker : _M2BestTracker
        History and best-val bookkeeping.
    step_fn : callable
        Jitted train step.
    ema_model : ConditionalScoreNet or None
        EMA weights when ``ema_decay`` is set.
    """

    model: ConditionalScoreNet
    opt_state: optax.OptState
    key_loop: jax.Array
    val_batch: RealDSMBatch
    tracker: _M2BestTracker
    step_fn: Callable
    ema_model: ConditionalScoreNet | None = None

    def take_epoch_key(self) -> jax.Array:
        """Split ``key_loop`` and return the next epoch key.

        Returns
        -------
        jax.Array
            PRNG key for the upcoming epoch.
        """
        self.key_loop, key_epoch = jax.random.split(self.key_loop)
        return key_epoch

    def store_epoch(
            self,
            model: ConditionalScoreNet,
            opt_state: optax.OptState,
            row: Mapping[str, Any],
            ema_model: ConditionalScoreNet | None,
    ) -> None:
        """Store updated weights and append the epoch metrics row.

        Parameters
        ----------
        model : ConditionalScoreNet
            Updated score network.
        opt_state : optax.OptState
            Updated optimizer state.
        row : mapping of str to Any
            Epoch metrics including ``val_loss``.
        ema_model : ConditionalScoreNet or None
            Updated EMA weights (or ``None``).
        """
        self.model = model
        self.opt_state = opt_state
        self.ema_model = ema_model
        self.tracker.append_row(row)

    def eval_model(self) -> ConditionalScoreNet:
        """Return EMA weights when present, else the raw model."""
        if self.ema_model is None:
            return self.model
        return self.ema_model


class M2Trainer:
    """Train a conditional score net on normalized real windows.

    Public methods
    --------------
    run
        Full train/val loop with checkpointing and curve export.
    evaluate
        Mean DSM loss on a fixed ``RealDSMBatch`` (multiple MC draws).

    Instance variables
    ------------------
    config : RealM2Config
        Full M2 config.
    model : ConditionalScoreNet
        Score network being optimized.
    loss_fn : RealConditionalDSMLoss
        Real DSM loss module.
    optimizer : optax.GradientTransformation
        Optax optimizer.
    paths : ArtifactPaths
        Artifact directory helpers.
    pool : RealBatchPool
        Train/val pools.
    val_hook : callable
        Snapshot hook.
    """

    def __init__(self, bundle: M2TrainerBundle) -> None:
        """Attach a pre-built trainer bundle.

        Parameters
        ----------
        bundle : M2TrainerBundle
            Model, loss, optimizer, config, pool, paths, and val hook.
        """
        self.config = bundle.config
        self.model = bundle.model
        self.loss_fn = bundle.loss_fn
        self.optimizer = bundle.optimizer
        self.paths = bundle.paths
        self.pool = bundle.pool
        self.val_hook = bundle.val_hook

    def run(self, key: jax.Array) -> tuple[ConditionalScoreNet, TrainHistory]:
        """Run the M2 training loop on real windows.

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
        state = self._init_loop_state(key)
        hygiene = _GradHygiene()
        # None = no selection epoch wrote a best ckpt (not "epoch 0").
        selected_epoch: int | None = None
        impute_bundle = self._maybe_build_impute_bundle()
        for epoch in range(1, self.config.train.schedule.epochs + 1):
            model, opt_state, ema_model, row, step_totals = self._run_epoch(
                state,
                state.take_epoch_key(),
                epoch,
            )
            hygiene.absorb(step_totals)
            state.store_epoch(model, opt_state, row, ema_model)
            if self._maybe_select(
                    state,
                    epoch,
                    self.config.train.protocol.select_by,
                    impute_bundle,
            ):
                selected_epoch = epoch
                save_checkpoint(
                    self.paths.best_checkpoint(),
                    state.tracker.best_model,
                )
            # shortcut: skip PNG when impute select already reverse-SDEs;
            # upgrade: cheap PNG path or ASSUMPTIONS note (see WP4 hygiene).
            if (
                    epoch % self.config.train.schedule.val_every == 0
                    and impute_bundle is None
            ):
                self._fire_val_hook(state.eval_model(), state.val_batch, epoch)

        if selected_epoch is None:
            # No consider() ran: overwrite any stale ckpt with final eval weights.
            state.tracker.best_model = state.eval_model()
            save_checkpoint(
                self.paths.best_checkpoint(),
                state.tracker.best_model,
            )
        state.tracker.history.save_csv(self.paths.loss_csv())
        state.tracker.history.save_plot(
            self.paths.loss_png(),
            title=_LOSS_PLOT_TITLE,
        )
        write_train_meta(
            self.paths.root,
            _train_meta_payload(
                self.config,
                state.tracker.best_model,
                hygiene,
                selected_epoch,
            ),
        )
        self.model = state.tracker.best_model
        return state.tracker.best_model, state.tracker.history

    def evaluate(
            self,
            model: ConditionalScoreNet,
            batch: RealDSMBatch,
            key: jax.Array,
    ) -> jnp.ndarray:
        """Estimate mean DSM loss with a few Monte Carlo draws.

        Parameters
        ----------
        model : ConditionalScoreNet
            Score network to evaluate.
        batch : RealDSMBatch
            Evaluation batch.
        key : jax.Array
            PRNG key.

        Returns
        -------
        jax.Array
            Scalar mean loss over internal MC draws.
        """
        keys = jax.random.split(key, _VAL_MC_DRAWS)
        losses = [
            self.loss_fn(model, batch, keys[i]) for i in range(_VAL_MC_DRAWS)
        ]
        return jnp.mean(jnp.stack(losses))

    def _init_loop_state(self, key: jax.Array) -> _M2LoopState:
        """Build initial mutable training-loop state.

        Parameters
        ----------
        key : jax.Array
            Root PRNG key.

        Returns
        -------
        _M2LoopState
            Model, optimizer, val batch, history, and jitted step.
        """
        key_loop, _key_unused = jax.random.split(key)
        model = self.model
        decay = self.config.train.protocol.ema_decay
        val_batch = self.pool.fixed_val_batch(self.config.data.batch_size)
        tracker = _M2BestTracker(
            history=TrainHistory(),
            best_val=float("inf"),
            best_model=init_ema(model) if decay is not None else model,
        )
        return _M2LoopState(
            model=model,
            opt_state=self.optimizer.init(eqx.filter(model, eqx.is_array)),
            key_loop=key_loop,
            val_batch=val_batch,
            tracker=tracker,
            step_fn=build_real_train_step(self.optimizer, self.loss_fn),
            ema_model=init_ema(model) if decay is not None else None,
        )

    def _run_epoch(
            self,
            state: _M2LoopState,
            key_epoch: jax.Array,
            epoch: int,
    ) -> tuple[
        ConditionalScoreNet,
        optax.OptState,
        ConditionalScoreNet | None,
        dict[str, Any],
        _StepTotals,
    ]:
        """Execute one epoch of train steps and one val DSM evaluation."""
        n_steps = int(self.config.train.schedule.steps_per_epoch)
        key_pack = _M2EpochKeyPack.split_from(key_epoch, n_steps)
        model, opt_state, ema_model, totals = self._accumulate_steps(
            state,
            key_pack,
        )
        train_loss, grad_mean = totals.means(n_steps)
        eval_model = ema_model if ema_model is not None else model
        val_loss = float(
            self.evaluate(eval_model, state.val_batch, key_pack.key_val)
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "grad_norm": grad_mean,
        }
        if epoch % self.config.train.schedule.log_every == 0:
            print(
                f"epoch={epoch} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} grad_norm={grad_mean:.6f}"
            )
        return model, opt_state, ema_model, row, totals

    def _accumulate_steps(
            self,
            state: _M2LoopState,
            key_pack: _M2EpochKeyPack,
    ) -> tuple[
        ConditionalScoreNet,
        optax.OptState,
        ConditionalScoreNet | None,
        _StepTotals,
    ]:
        """Run all mini-batch updates for one epoch."""
        model = state.model
        opt_state = state.opt_state
        ema_model = state.ema_model
        decay = self.config.train.protocol.ema_decay
        clip_norm = float(self.config.train.optimizer.grad_clip_norm)
        totals = _StepTotals()
        for step_idx in range(key_pack.n_train_steps()):
            batch = self.pool.sample_train(
                key_pack.data_keys[step_idx],
                self.config.data.batch_size,
            )
            model, opt_state, loss, grad_norm = state.step_fn(
                model,
                opt_state,
                batch,
                key_pack.step_keys[step_idx],
            )
            if not jnp.isfinite(loss):
                raise RuntimeError(_NAN_MSG)
            grad_value = float(grad_norm)
            totals.add(
                float(loss),
                grad_value,
                grad_value > clip_norm > _GRAD_NORM_EPS,
            )
            if decay is not None and ema_model is not None:
                ema_model = update_ema(ema_model, model, decay)
        return model, opt_state, ema_model, totals

    def _maybe_build_impute_bundle(self) -> ValImputeBundle | None:
        """Build fixed val-impute set + jitted runner when selecting by impute."""
        protocol = self.config.train.protocol
        if protocol.select_by not in (SELECT_CRPS, SELECT_MAE):
            return None
        spec = dict(protocol.val_impute or _DEFAULT_VAL_IMPUTE)
        if "protocols" in spec:
            spec["protocols"] = tuple(spec["protocols"])
        fixed = build_fixed_val_impute_set(
            self.pool.val_values,
            self.pool.val_masks,
            spec,
        )
        values = self.pool.val_values[np.asarray(fixed.window_indices)]
        native = self.pool.val_masks[np.asarray(fixed.window_indices)]
        sample_t_eps = resolve_sample_t_eps(
            self.config.sample.t_eps,
            protocol.t_eps,
        )
        settings = ImputeSolverSettings(
            solver_steps=int(spec["solver_steps"]),
            t_eps=sample_t_eps,
            diffusion_impl=str(self.config.sample.diffusion_impl),
        )
        runner = make_val_impute_runner(
            self.loss_fn.sde,
            settings,
            int(values.shape[TIME_AXIS]),
            int(spec["K_samples"]),
            protocol.parameterization,
        )
        return ValImputeBundle(
            fixed=fixed,
            values=values,
            m_native=native,
            runner=runner,
            select_by=protocol.select_by,
        )

    def _maybe_select(
            self,
            state: _M2LoopState,
            epoch: int,
            select_by: str,
            impute_bundle: ValImputeBundle | None,
    ) -> bool:
        """Update best checkpoint from DSM or val-impute score."""
        eval_model = state.eval_model()
        if select_by == SELECT_DSM:
            return state.tracker.consider(
                eval_model,
                float(state.tracker.history.rows[-1]["val_loss"]),
            )
        schedule = self.config.train.schedule
        if epoch % schedule.val_every != 0 or impute_bundle is None:
            return False
        score = score_val_impute(eval_model, impute_bundle)
        if not np.isfinite(score):
            raise RuntimeError(
                "non-finite val-impute score; aborting selection"
            )
        return state.tracker.consider(eval_model, score)

    def _fire_val_hook(
            self,
            model: ConditionalScoreNet,
            val_batch: RealDSMBatch,
            epoch: int,
    ) -> None:
        """Sample a composed val mask and invoke the snapshot hook.

        Parameters
        ----------
        model : ConditionalScoreNet
            Current score network.
        val_batch : RealDSMBatch
            Fixed validation batch.
        epoch : int
            Epoch index used as the snapshot step label.
        """
        key = jax.random.PRNGKey(epoch)
        x_obs, m_train, _m_art = self.loss_fn.sample_mask(key, val_batch)
        payload = ValSnapshotPayload(
            model=model,
            x0=val_batch.x0,
            mask=m_train,
            x_obs=x_obs,
            step=epoch,
            out_dir=self.paths.val_dir(),
        )
        self.val_hook(payload)


def run_m2_training(config_path: Path | str) -> TrainHistory:
    """Train on processed real windows using a YAML config path.

    Parameters
    ----------
    config_path : path-like
        Path to ``configs/real_m2.yaml`` (or compatible).

    Returns
    -------
    TrainHistory
        Logged metrics (CSV/PNG written under ``artifact_dir``).
    """
    return run_m2_training_from_config(load_real_m2_config(config_path))


def run_m2_training_from_config(config: RealM2Config) -> TrainHistory:
    """Train from an in-memory ``RealM2Config`` (CLI overrides share this).

    Parameters
    ----------
    config : RealM2Config
        Validated config (possibly mutated by CLI overrides).

    Returns
    -------
    TrainHistory
        Logged metrics (CSV/PNG written under ``artifact_dir``).
    """
    _assert_wp0_consumable(config)
    paths, pool = _prepare_artifacts_and_pool(config)
    trainer, key_train = _build_m2_trainer(config, paths, pool)
    _model, history = trainer.run(key_train)
    _maybe_run_sensitivity(trainer, config, pool)
    return history


def _build_m2_trainer(
        config: RealM2Config,
        paths: ArtifactPaths,
        pool: RealBatchPool,
) -> tuple[M2Trainer, jax.Array]:
    """Construct an ``M2Trainer`` and the training PRNG key.

    Parameters
    ----------
    config : RealM2Config
        Validated real-m2 config.
    paths : ArtifactPaths
        Artifact directories (already ensured).
    pool : RealBatchPool
        Normalized train/val pools.

    Returns
    -------
    trainer : M2Trainer
        Ready trainer (paths ensured).
    key_train : jax.Array
        PRNG key for ``trainer.run``.
    """
    key_model, key_train = jax.random.split(
        jax.random.PRNGKey(config.train.protocol.seed)
    )
    sde = VPSDE(config.sde)
    protocol = config.train.protocol
    sample_t_eps = resolve_sample_t_eps(config.sample.t_eps, protocol.t_eps)
    bundle = M2TrainerBundle(
        config=config,
        model=ConditionalScoreNet(
            key_model,
            default_score_net_config(config.data.n_features),
        ),
        loss_fn=RealConditionalDSMLoss(
            sde,
            dsm_settings_from_real_config(config),
        ),
        optimizer=build_optimizer(config.train),
        paths=paths,
        pool=pool,
        val_hook=M2ValHookBundle(
            sde=sde,
            t_eps=sample_t_eps,
            k_samples=config.extras.snapshot_k_samples,
            solver_steps=config.extras.snapshot_solver_steps,
            diffusion_impl=str(config.sample.diffusion_impl),
            parameterization=protocol.parameterization,
        ).as_hook(),
    )
    bundle.validate()
    return M2Trainer(bundle.with_paths()), key_train


def _maybe_run_sensitivity(
        trainer: M2Trainer,
        config: RealM2Config,
        pool: RealBatchPool,
) -> None:
    """Run optional mask-rate sensitivity smoke when configured.

    Parameters
    ----------
    trainer : M2Trainer
        Trained trainer (best model attached).
    config : RealM2Config
        Config with optional ``extras.sensitivity_mask_rates``.
    pool : RealBatchPool
        Normalized pools for a fixed val batch.
    """
    rates = config.extras.sensitivity_mask_rates
    if not rates:
        return
    run_mask_rate_sensitivity(
        trainer,
        pool.fixed_val_batch(config.data.batch_size),
        rates,
        jax.random.PRNGKey(config.train.protocol.seed + 1),
    )


def run_mask_rate_sensitivity(
        trainer: M2Trainer,
        val_batch: RealDSMBatch,
        rates: tuple[float, ...],
        key: jax.Array,
) -> list[dict[str, float]]:
    """Evaluate val DSM loss at several artificial mask rates.

    Parameters
    ----------
    trainer : M2Trainer
        Trainer holding the best model (after ``run``).
    val_batch : RealDSMBatch
        Fixed validation batch.
    rates : tuple of float
        Artificial MCAR rates to probe.
    key : jax.Array
        Root PRNG key.

    Returns
    -------
    list of dict
        One ``{mask_rate, val_loss}`` row per rate.
    """
    rows: list[dict[str, float]] = []
    keys = jax.random.split(key, len(rates))
    base_loss = trainer.loss_fn
    for rate_idx, rate in enumerate(rates):
        probe = base_loss.with_mask_rate(rate)
        losses = [
            probe(trainer.model, val_batch, sub_key)
            for sub_key in jax.random.split(keys[rate_idx], _VAL_MC_DRAWS)
        ]
        val_loss = float(jnp.mean(jnp.stack(losses)))
        row = {"mask_rate": float(rate), "val_loss": val_loss}
        rows.append(row)
        print(
            f"sensitivity mask_rate={rate:.2f} val_loss={val_loss:.6f}"
        )
    return rows


@dataclass
class _GradHygiene:
    """Accumulate pre-clip grad norms across training."""

    n_steps: int = 0
    n_clipped: int = 0
    grad_norms: list[float] = field(default_factory=list)

    def absorb(self, totals: _StepTotals) -> None:
        """Merge one epoch's step totals into the run totals."""
        self.n_steps += len(totals.grad_norms)
        self.n_clipped += totals.n_clipped
        self.grad_norms.extend(totals.grad_norms)


def _parameter_count(model: ConditionalScoreNet) -> int:
    """Count array parameters in an Equinox model."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    return int(sum(int(np.asarray(leaf).size) for leaf in leaves))


def _train_meta_payload(
        config: RealM2Config,
        best_model: ConditionalScoreNet,
        hygiene: _GradHygiene,
        selected_epoch: int | None,
) -> dict[str, Any]:
    """Build the ``train_meta.json`` payload for WP4."""
    protocol = config.train.protocol
    n_steps = max(hygiene.n_steps, 1)
    median_grad = (
        float(np.median(np.asarray(hygiene.grad_norms, dtype=np.float64)))
        if hygiene.grad_norms
        else 0.0
    )
    return {
        "clipped_step_fraction": hygiene.n_clipped / float(n_steps),
        "median_grad_norm": median_grad,
        "ema_decay": protocol.ema_decay,
        "parameter_count": _parameter_count(best_model),
        "receptive_field": receptive_field(config.model),
        # null when no selection epoch improved (fallback ckpt only)
        "selected_epoch": selected_epoch,
        "selection_fallback": selected_epoch is None,
        "select_by": protocol.select_by,
        "observed_input": protocol.observed_input,
        "observed_mode": config.sample.observed_mode,
        "lr_schedule": protocol.lr_schedule,
        "warmup_steps": protocol.warmup_steps,
    }
