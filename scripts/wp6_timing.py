"""WP6 CPU timing: dense vs diagonal reverse SDE (one window).

Builds one ``eqx.filter_jit`` runner per mode, warms it, then times a second
``K=50`` / ``steps=100`` solve at ``F=6`` (real Beijing width). Writes raw
``mean.csv`` / ``median.csv`` under ``artifacts/m7_5/wp6/`` from an untrained
net (smoke only, not results).

Usage
-----
python scripts/wp6_timing.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.config import load_config
from src.data.schema import MASK_MISSING
from src.eval.metrics import write_mean_median_csvs
from src.models.score_net import (
    ConditionalScoreNet,
    default_score_net_config,
)
from src.sample.reverse_sde import (
    ImputeContext,
    ImputeQuery,
    ImputeSolverSettings,
    ReverseSDEImputer,
    grid_timestamps,
)
from src.sde.vp_sde import VPSDE

_TOY_CONFIG = _REPO_ROOT / "configs" / "toy_m1.yaml"
_OUT_DIR = _REPO_ROOT / "artifacts" / "m7_5" / "wp6"
_SEED = 0
_SEQ = 64
# Real Beijing width (sine toy generator only allows F∈{1,2,3}).
_N_FEATURES = 6
_K_SAMPLES = 50
_SOLVER_STEPS = 100
_T_EPS = 1.0e-3
_SAMPLE_AXIS = 0


def main() -> None:
    """Time warm dense/diagonal solves and write point-estimate CSVs."""
    cfg = load_config(_TOY_CONFIG)
    key = jax.random.PRNGKey(_SEED)
    key_data, key_model, key_impute = jax.random.split(key, 3)
    window = _window_with_gap(key_data)
    model = ConditionalScoreNet(
        key_model,
        default_score_net_config(_N_FEATURES),
    )
    sde = VPSDE(cfg.sde)
    dense_run = _jitted_impute(model, sde, "dense")
    diag_run = _jitted_impute(model, sde, "diagonal")
    dense_s = _warm_seconds(dense_run, window, key_impute)
    diag_s = _warm_seconds(diag_run, window, key_impute)
    samples, mean = diag_run(window[0], window[1], key_impute)
    samples.block_until_ready()
    median = jnp.median(samples, axis=_SAMPLE_AXIS)
    write_mean_median_csvs(_OUT_DIR, mean, median)
    print(
        f"warm CPU T={_SEQ} F={_N_FEATURES} K={_K_SAMPLES} "
        f"steps={_SOLVER_STEPS}: dense={dense_s:.3f}s "
        f"diagonal={diag_s:.3f}s out={_OUT_DIR}"
    )


def _window_with_gap(key: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return one ``(T, F)`` Gaussian window with a mid-gap."""
    values = jax.random.normal(key, (_SEQ, _N_FEATURES), dtype=jnp.float32)
    mask = jnp.ones((_SEQ, _N_FEATURES), dtype=jnp.float32)
    mask = mask.at[_SEQ // 2, :].set(MASK_MISSING)
    return values * mask, mask


def _jitted_impute(
        model: ConditionalScoreNet,
        sde: VPSDE,
        diffusion_impl: str,
) -> Callable:
    """Return one filter-jitted impute runner for ``diffusion_impl``."""
    settings = ImputeSolverSettings(
        solver_steps=_SOLVER_STEPS,
        t_eps=_T_EPS,
        diffusion_impl=diffusion_impl,
    )
    imputer = ReverseSDEImputer(
        ImputeContext(model=model, sde=sde, settings=settings)
    )
    timestamps = grid_timestamps(_SEQ)

    @eqx.filter_jit
    def run(
            observations: jnp.ndarray,
            mask: jnp.ndarray,
            key: jax.Array,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        samples, mean, _quantiles = imputer.impute(
            ImputeQuery(
                observations=observations,
                mask=mask,
                timestamps=timestamps,
                key=key,
                k_samples=_K_SAMPLES,
            )
        )
        return samples, mean

    return run


def _warm_seconds(
        run: Callable,
        window: tuple[jnp.ndarray, jnp.ndarray],
        key: jax.Array,
) -> float:
    """Warm the jitted runner once, then time a second call."""
    observations, mask = window
    run(observations, mask, key)[0].block_until_ready()
    started = time.perf_counter()
    samples, _mean = run(observations, mask, key)
    samples.block_until_ready()
    return time.perf_counter() - started


if __name__ == "__main__":
    main()
