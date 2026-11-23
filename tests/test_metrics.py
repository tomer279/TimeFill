"""WP6/WP5 tests: masked MAE/RMSE/CRPS, coverage, empty-target NaN (A9)."""

import math

import jax.numpy as jnp

from src.data.schema import MASK_MISSING, MASK_OBSERVED
from src.eval.metrics import (
    EvalTensors,
    ImputationEvaluator,
    empirical_crps,
    interval_coverage,
    masked_mae,
    masked_rmse,
)

_BATCH = 1
_K = 4
_SEQ = 3
_FEAT = 1
_LEVEL_90 = 0.9


def _perfect_tensors() -> EvalTensors:
    """Build tensors where samples equal truth on the missing cell."""
    truth = jnp.array([[[0.0], [2.0], [0.0]]], dtype=jnp.float32)
    mask = jnp.array(
        [[[MASK_OBSERVED], [MASK_MISSING], [MASK_OBSERVED]]],
        dtype=jnp.float32,
    )
    mean = truth
    samples = jnp.broadcast_to(
        truth[:, None, :, :],
        (_BATCH, _K, _SEQ, _FEAT),
    )
    return EvalTensors(truth=truth, mean=mean, samples=samples, mask=mask)


def test_masked_mae_rmse_zero_when_perfect() -> None:
    """Perfect mean fill yields zero masked MAE/RMSE."""
    tensors = _perfect_tensors()
    assert float(masked_mae(tensors)) == 0.0
    assert float(masked_rmse(tensors)) == 0.0


def test_empirical_crps_zero_when_degenerate() -> None:
    """Identical samples at the truth → CRPS ≈ 0 on the gap."""
    tensors = _perfect_tensors()
    assert float(empirical_crps(tensors)) < 1e-6


def test_interval_coverage_and_table() -> None:
    """Coverage is 1 for a perfect ensemble; table has expected keys."""
    tensors = _perfect_tensors()
    assert float(interval_coverage(tensors, _LEVEL_90)) == 1.0
    table = ImputationEvaluator().compute(tensors)
    mapping = table.as_dict()
    assert set(mapping) == {
        "mae",
        "rmse",
        "crps",
        "coverage_50",
        "coverage_90",
    }
    assert mapping["coverage_50"] == 1.0
    assert mapping["coverage_90"] == 1.0


def _all_observed_tensors() -> EvalTensors:
    """Build tensors with no missing (target) cells."""
    truth = jnp.array([[[1.0], [2.0], [3.0]]], dtype=jnp.float32)
    mask = jnp.ones_like(truth, dtype=jnp.float32) * MASK_OBSERVED
    mean = truth
    samples = jnp.broadcast_to(
        truth[:, None, :, :],
        (_BATCH, _K, _SEQ, _FEAT),
    )
    return EvalTensors(truth=truth, mean=mean, samples=samples, mask=mask)


def test_all_observed_mask_metrics_are_nan() -> None:
    """No target cells → MAE/RMSE/CRPS/coverage are NaN, not 0.0 (A9)."""
    tensors = _all_observed_tensors()
    assert math.isnan(float(masked_mae(tensors)))
    assert math.isnan(float(masked_rmse(tensors)))
    assert math.isnan(float(empirical_crps(tensors)))
    assert math.isnan(float(interval_coverage(tensors, _LEVEL_90)))
    table = ImputationEvaluator().compute(tensors)
    for name, value in table.as_dict().items():
        assert math.isnan(value), f"{name} should be NaN, got {value}"
