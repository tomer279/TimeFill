"""Fixed val-impute selection and EMA helpers for M2 / WP4 training.

Classes
-------
FixedValImputeSet
    Fixed val windows, artificial masks, and sampler key for selection.
ValImputeBundle
    Cached fixed set plus jitted runner for selection scoring.

Functions
---------
build_fixed_val_impute_set
    Deterministic val-impute mask set from a ``val_impute`` spec.
init_ema
    Copy model arrays for an EMA tree.
update_ema
    EMA update ``decay * ema + (1 - decay) * model``.
make_val_impute_runner
    One filter-jitted impute runner; model is an argument.
score_val_impute
    Mean CRPS or MAE over fixed protocols in normalized space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from src.data.real.masks import MaskComposer, compose_train_mask
from src.data.schema import apply_missing_sentinel
from src.eval.m2_val_impute import metric_observation_mask
from src.eval.metrics import EvalTensors, empirical_crps, masked_mae
from src.models.score_from_eps import as_score_model
from src.models.score_net import ConditionalScoreNet
from src.sample.reverse_sde import (
    ImputeContext,
    ImputeQuery,
    ImputeSolverSettings,
    ReverseSDEImputer,
    grid_timestamps,
)
from src.sde.vp_sde import VPSDE

SELECT_DSM = "val_dsm_loss"
SELECT_CRPS = "val_impute_crps"
SELECT_MAE = "val_impute_mae"
_VAL_INDEX_FOLD = 7919

__all__ = [
    "FixedValImputeSet",
    "SELECT_CRPS",
    "SELECT_DSM",
    "SELECT_MAE",
    "ValImputeBundle",
    "build_fixed_val_impute_set",
    "init_ema",
    "make_val_impute_runner",
    "score_val_impute",
    "update_ema",
]


@dataclass(frozen=True)
class FixedValImputeSet:
    """Fixed val windows, artificial masks, and sampler key for selection.

    Instance variables
    ------------------
    masks : dict of str to jax.Array
        Protocol name → artificial mask ``(n_windows, T, F)``.
    sample_key : jax.Array
        Frozen reverse-SDE PRNG key.
    window_indices : jax.Array
        Indices into the val pool ``(n_windows,)``.
    """

    masks: dict[str, jnp.ndarray]
    sample_key: jax.Array
    window_indices: jnp.ndarray


@dataclass(frozen=True)
class ValImputeBundle:
    """Cached fixed set + runner for val-impute selection.

    Instance variables
    ------------------
    fixed : FixedValImputeSet
        Deterministic windows and masks.
    values : jax.Array
        Normalized val windows for those indices.
    m_native : jax.Array
        Native masks matching ``values``.
    runner : callable
        Jitted ``(model, observations, mask, key) -> (samples, mean)``.
    select_by : str
        ``val_impute_crps`` or ``val_impute_mae``.
    """

    fixed: FixedValImputeSet
    values: jnp.ndarray
    m_native: jnp.ndarray
    runner: Callable
    select_by: str


def init_ema(model: ConditionalScoreNet) -> ConditionalScoreNet:
    """Return an EMA copy of ``model`` array leaves.

    Parameters
    ----------
    model : ConditionalScoreNet
        Source network.

    Returns
    -------
    ConditionalScoreNet
        Tree with the same array values (independent leaves under JAX).
    """
    arrays, static = eqx.partition(model, eqx.is_array)
    return eqx.combine(jax.tree_util.tree_map(lambda x: x, arrays), static)


def update_ema(
        ema: ConditionalScoreNet,
        model: ConditionalScoreNet,
        decay: float,
) -> ConditionalScoreNet:
    """EMA update ``decay * ema + (1 - decay) * model``.

    Parameters
    ----------
    ema : ConditionalScoreNet
        Current EMA weights.
    model : ConditionalScoreNet
        Latest raw weights.
    decay : float
        EMA decay in ``[0, 1)``; ``0`` copies ``model``.

    Returns
    -------
    ConditionalScoreNet
        Updated EMA tree.
    """
    ema_arrays, ema_static = eqx.partition(ema, eqx.is_array)
    model_arrays, _model_static = eqx.partition(model, eqx.is_array)
    new_arrays = jax.tree_util.tree_map(
        lambda ema_leaf, model_leaf: (
            decay * ema_leaf + (1.0 - decay) * model_leaf
        ),
        ema_arrays,
        model_arrays,
    )
    return eqx.combine(new_arrays, ema_static)


def build_fixed_val_impute_set(
        values: jnp.ndarray,
        m_native: jnp.ndarray,
        spec: Mapping[str, Any],
) -> FixedValImputeSet:
    """Build a deterministic val-impute mask set from ``spec``.

    Parameters
    ----------
    values : jax.Array
        Val windows ``(N, T, F)``.
    m_native : jax.Array
        Native masks matching ``values``.
    spec : mapping
        ``n_windows``, ``mask_seed``, ``sample_seed``, ``protocols``.

    Returns
    -------
    FixedValImputeSet
        Fixed indices, per-protocol artificial masks, and sample key.
    """
    n_pool = int(values.shape[0])
    n_take = min(int(spec["n_windows"]), n_pool)
    key_root = jax.random.PRNGKey(int(spec["mask_seed"]))
    # Seeded subset (not prefix) so entity-split val is not one station/season.
    perm = jax.random.permutation(
        jax.random.fold_in(key_root, _VAL_INDEX_FOLD),
        n_pool,
    )
    indices = jnp.sort(perm[:n_take]).astype(jnp.int32)
    vals = values[indices]
    native = m_native[indices]
    masks: dict[str, jnp.ndarray] = {}
    for proto_i, protocol in enumerate(spec["protocols"]):
        key_i = jax.random.fold_in(key_root, int(proto_i))
        masks[str(protocol)] = _artificial_mask_for_protocol(
            key_i,
            vals,
            native,
            str(protocol),
        )
    return FixedValImputeSet(
        masks=masks,
        sample_key=jax.random.PRNGKey(int(spec["sample_seed"])),
        window_indices=indices,
    )


def make_val_impute_runner(
        sde: VPSDE,
        settings: ImputeSolverSettings,
        seq_len: int,
        k_samples: int,
        parameterization: str,
) -> Callable:
    """Build one filter-jitted impute runner; model is an argument.

    Parameters
    ----------
    sde : VPSDE
        VP-SDE matching training.
    settings : ImputeSolverSettings
        Solver steps, ``t_eps``, diffusion impl.
    seq_len : int
        Window length ``T``.
    k_samples : int
        Posterior draws per window.
    parameterization : str
        ``score`` or ``eps``.

    Returns
    -------
    callable
        ``(model, observations, mask, key) -> (samples, mean)``.
    """
    timestamps = grid_timestamps(seq_len)

    @eqx.filter_jit
    def run(
            model: ConditionalScoreNet,
            observations: jnp.ndarray,
            mask: jnp.ndarray,
            key: jax.Array,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        score_model = as_score_model(model, sde, parameterization)
        imputer = ReverseSDEImputer(
            ImputeContext(model=score_model, sde=sde, settings=settings)
        )
        samples, mean, _quantiles = imputer.impute(
            ImputeQuery(
                observations=observations,
                mask=mask,
                timestamps=timestamps,
                key=key,
                k_samples=k_samples,
            )
        )
        return samples, mean

    return run


def score_val_impute(
        model: ConditionalScoreNet,
        bundle: ValImputeBundle,
) -> float:
    """Mean CRPS or MAE over fixed protocols in normalized space.

    Parameters
    ----------
    model : ConditionalScoreNet
        Candidate weights (usually EMA).
    bundle : ValImputeBundle
        Fixed masks, values, and jitted runner.

    Returns
    -------
    float
        Mean metric over protocols (lower is better).
    """
    scores: list[float] = []
    for proto_i, protocol in enumerate(bundle.fixed.masks):
        m_art = bundle.fixed.masks[protocol]
        m_train = compose_train_mask(bundle.m_native, m_art)
        x_obs = apply_missing_sentinel(bundle.values, m_train)
        key = jax.random.fold_in(bundle.fixed.sample_key, proto_i)
        samples, mean = bundle.runner(model, x_obs, m_train, key)
        metric_mask = metric_observation_mask(bundle.m_native, m_art)
        tensors = EvalTensors(
            truth=bundle.values,
            mean=mean,
            samples=samples,
            mask=metric_mask,
        )
        if bundle.select_by == SELECT_MAE:
            scores.append(float(masked_mae(tensors)))
        else:
            scores.append(float(empirical_crps(tensors)))
    return float(np.mean(np.asarray(scores, dtype=np.float64)))


def _artificial_mask_for_protocol(
        key: jax.Array,
        values: jnp.ndarray,
        m_native: jnp.ndarray,
        protocol: str,
) -> jnp.ndarray:
    """Return restricted artificial mask for ``mcar_*`` or ``block_*``."""
    if protocol.startswith("mcar_"):
        rate = float(protocol.split("_", 1)[1])
        _x_obs, _m_train, m_art = MaskComposer(rate, block_len=8).compose_random(
            key, values, m_native,
        )
        return m_art
    if protocol.startswith("block_"):
        block_len = int(protocol.split("_", 1)[1])
        _x_obs, _m_train, m_art = MaskComposer(
            rate=0.3,
            block_len=block_len,
        ).compose_block(key, values, m_native)
        return m_art
    raise ValueError(f"unsupported val_impute protocol {protocol!r}")
