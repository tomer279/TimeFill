"""Capped DSM training loop for Milestone 7 upload-adapt.

Classes
-------
AdaptLoopDeps
    Dependencies for one capped adapt epoch loop.

Functions
---------
run_capped_adapt_dsm
    Smoke-capped DSM; return best-val weights (memory only).
dsm_settings_from_adapt
    Map adapt caps into ``DSMLossSettings``.

Notes
-----
Does **not** write checkpoints. Callers promote artifacts only after a
clean return (no NaN / timeout abort).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.data.adapt_config import AdaptM7Config
from src.models.score_net import ConditionalScoreNet
from src.sde.vp_sde import VPSDE
from src.train.loop import build_optimizer
from src.train.loss import DSMLossSettings
from src.train.real_loop import build_real_train_step
from src.train.real_loss import RealConditionalDSMLoss, RealDSMBatch
from src.train.real_pool import RealBatchPool

_VAL_MC_DRAWS = 2
_NAN_MSG = (
    "non-finite train loss detected; aborting adapt without promoting best"
)
_TIMEOUT_MSG = (
    "adapt timeout_seconds exceeded; aborting without promoting best"
)


@dataclass(frozen=True)
class AdaptLoopDeps:
    """Dependencies for one capped adapt epoch loop.

    Public methods
    --------------
    assert_not_timed_out
        Raise if the wall-clock deadline has passed.
    fixed_val_batch
        Return the fixed smoke validation batch.

    Instance variables
    ------------------
    step_fn : callable
        Jitted AdamW train step.
    loss_fn : RealConditionalDSMLoss
        Real DSM loss.
    pool : RealBatchPool
        Normalized train / smoke pools.
    batch_size : int
        Mini-batch size.
    n_steps : int
        Steps per epoch.
    deadline : float
        ``time.monotonic`` deadline.
    """

    step_fn: Callable
    loss_fn: RealConditionalDSMLoss
    pool: RealBatchPool
    batch_size: int
    n_steps: int
    deadline: float

    def assert_not_timed_out(self) -> None:
        """Raise if the wall-clock deadline has passed."""
        if time.monotonic() > self.deadline:
            raise RuntimeError(_TIMEOUT_MSG)

    def fixed_val_batch(self) -> RealDSMBatch:
        """Return the fixed smoke validation batch.

        Returns
        -------
        RealDSMBatch
            Leading smoke windows (padded by repeat if needed).
        """
        return self.pool.fixed_val_batch(self.batch_size)


def run_capped_adapt_dsm(
        model: ConditionalScoreNet,
        pool: RealBatchPool,
        config: AdaptM7Config,
        key: jax.Array,
) -> ConditionalScoreNet:
    """Run smoke-capped DSM; return best-val weights (memory only).

    Parameters
    ----------
    model : ConditionalScoreNet
        Initial score network.
    pool : RealBatchPool
        Normalized train / smoke pools.
    config : AdaptM7Config
        Caps, SDE, and DSM knobs.
    key : jax.Array
        Training PRNG key.

    Returns
    -------
    ConditionalScoreNet
        Best validation model from a completed run.

    Raises
    ------
    RuntimeError
        Non-finite loss or wall-clock timeout.
    """
    loss_fn = RealConditionalDSMLoss(
        VPSDE(config.sde),
        dsm_settings_from_adapt(config),
    )
    optimizer = build_optimizer(config.to_train_config())
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    deps = AdaptLoopDeps(
        step_fn=build_real_train_step(optimizer, loss_fn),
        loss_fn=loss_fn,
        pool=pool,
        batch_size=int(config.geometry.batch_size),
        n_steps=int(config.train.schedule.steps_per_epoch),
        deadline=time.monotonic() + float(config.caps.timeout_seconds),
    )
    best_model = model
    best_val = float("inf")
    key_loop = key
    for _epoch in range(int(config.train.schedule.epochs)):
        deps.assert_not_timed_out()
        key_loop, key_epoch = jax.random.split(key_loop)
        model, opt_state, val_loss = _run_adapt_epoch(
            model, opt_state, deps, key_epoch
        )
        if val_loss < best_val:
            best_val = val_loss
            best_model = model
    return best_model


def dsm_settings_from_adapt(config: AdaptM7Config) -> DSMLossSettings:
    """Map adapt caps into ``DSMLossSettings``.

    Parameters
    ----------
    config : AdaptM7Config
        Validated adapt config.

    Returns
    -------
    DSMLossSettings
        Validated DSM hyperparameters.
    """
    settings = DSMLossSettings(
        mask_rate=float(config.caps.smoke_mcar_rate),
        block_len=int(config.geometry.block_len),
        t_eps=float(config.train.protocol.t_eps),
        obs_loss_weight=float(config.train.protocol.obs_loss_weight),
        mask_mode=str(config.train.protocol.mask_mode),
    )
    settings.validate()
    return settings


def _run_adapt_epoch(
        model: ConditionalScoreNet,
        opt_state: optax.OptState,
        deps: AdaptLoopDeps,
        key_epoch: jax.Array,
) -> tuple[ConditionalScoreNet, optax.OptState, float]:
    """One epoch of train steps plus a smoke-val DSM estimate.

    Parameters
    ----------
    model : ConditionalScoreNet
        Score network at epoch start.
    opt_state : optax.OptState
        Optimizer state at epoch start.
    deps : AdaptLoopDeps
        Step function, pool, caps, and deadline.
    key_epoch : jax.Array
        Epoch PRNG key.

    Returns
    -------
    model : ConditionalScoreNet
        Updated score network.
    opt_state : optax.OptState
        Updated optimizer state.
    val_loss : float
        Mean smoke-val DSM loss.
    """
    keys = jax.random.split(key_epoch, deps.n_steps + 2)
    key_val = keys[0]
    step_keys = keys[1:deps.n_steps + 1]
    data_keys = jax.random.split(keys[deps.n_steps + 1], deps.n_steps)
    for step_idx in range(deps.n_steps):
        deps.assert_not_timed_out()
        batch = deps.pool.sample_train(data_keys[step_idx], deps.batch_size)
        model, opt_state, loss, _grad = deps.step_fn(
            model, opt_state, batch, step_keys[step_idx]
        )
        if not jnp.isfinite(loss):
            raise RuntimeError(_NAN_MSG)
    val_loss = _mean_val_loss(
        deps.loss_fn, model, deps.fixed_val_batch(), key_val
    )
    if not np.isfinite(val_loss):
        raise RuntimeError(_NAN_MSG)
    return model, opt_state, val_loss


def _mean_val_loss(
        loss_fn: RealConditionalDSMLoss,
        model: ConditionalScoreNet,
        batch: RealDSMBatch,
        key: jax.Array,
) -> float:
    """Mean DSM loss over a few MC draws on the smoke batch.

    Parameters
    ----------
    loss_fn : RealConditionalDSMLoss
        Real DSM loss.
    model : ConditionalScoreNet
        Score network to evaluate.
    batch : RealDSMBatch
        Smoke validation batch.
    key : jax.Array
        PRNG key.

    Returns
    -------
    float
        Mean loss over internal MC draws.
    """
    keys = jax.random.split(key, _VAL_MC_DRAWS)
    losses = [loss_fn(model, batch, keys[i]) for i in range(_VAL_MC_DRAWS)]
    return float(jnp.mean(jnp.stack(losses)))
