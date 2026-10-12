"""Hermetic LOCF and linear tests for M3 WP1 (no UCI download).

The feature means below are synthetic fixtures. They are the caller-supplied
edge fallback, and they are independent of the Beijing scaler.

Functions
---------
test_locf_hand_computed_interior_and_edges
    Single-gap, multi-gap, leading, trailing, and all-missing LOCF fills.
test_linear_hand_computed_interior_and_edges
    The same windows with bracket interpolation and mean edges.
test_single_observation_edge_policy
    A lone observation: LOCF carries forward; linear uses the mean elsewhere.
test_observed_zero_is_a_real_neighbor
    An observed ``0.0`` is context; the sentinel ``0.0`` is a gap.
test_impute_follows_evaluation_mask
    Fills follow impute-time ``m_eval``, including withheld ground truth.
test_shape_dtype_batch_and_window
    ``(B, T, F)`` and ``(T, F)`` stay float32 with matching shapes.
test_feature_means_length_must_match
    A mean vector whose length differs from ``F`` is rejected.
test_edge_policy_name_and_fallback_means
    Both imputers expose the locked policy name and the stored means.
"""

import jax.numpy as jnp
import pytest

from src.baselines import (
    EDGE_POLICY_TRAIN_FEATURE_MEAN,
    LinearImputer,
    LocfImputer,
    linear_impute,
    locf_impute,
)
from src.data.schema import MASK_MISSING, MASK_OBSERVED, MISSING_VALUE_SENTINEL

_ATOL = 1e-5
_RTOL = 1e-5
_FIRST_BATCH = 0
_SENTINEL = MISSING_VALUE_SENTINEL
_MEAN_0 = 100.0
_MEAN_1 = -3.0
_LONE_MEAN = 99.0
_LONE_VALUE = 5.0
_HELD_OUT_TRUTH = 100.0

_FEATURE_MEANS = jnp.asarray([_MEAN_0, _MEAN_1], dtype=jnp.float32)

# Batch 0 feature 0: observed 2, 4, gap, 8, 10, 12 (single interior gap).
# Batch 0 feature 1: leading, obs 10, interior, obs 16, trailing.
# Batch 1 feature 0: all missing.
# Batch 1 feature 1: obs 1, two-step gap, obs 7, one-step gap, obs 13.
_VALUES = jnp.asarray(
    [
        [
            [2.0, _SENTINEL],
            [4.0, _SENTINEL],
            [_SENTINEL, 10.0],
            [8.0, _SENTINEL],
            [10.0, 16.0],
            [12.0, _SENTINEL],
        ],
        [
            [_SENTINEL, 1.0],
            [_SENTINEL, _SENTINEL],
            [_SENTINEL, _SENTINEL],
            [_SENTINEL, 7.0],
            [_SENTINEL, _SENTINEL],
            [_SENTINEL, 13.0],
        ],
    ],
    dtype=jnp.float32,
)

_MASK = jnp.asarray(
    [
        [
            [MASK_OBSERVED, MASK_MISSING],
            [MASK_OBSERVED, MASK_MISSING],
            [MASK_MISSING, MASK_OBSERVED],
            [MASK_OBSERVED, MASK_MISSING],
            [MASK_OBSERVED, MASK_OBSERVED],
            [MASK_OBSERVED, MASK_MISSING],
        ],
        [
            [MASK_MISSING, MASK_OBSERVED],
            [MASK_MISSING, MASK_MISSING],
            [MASK_MISSING, MASK_MISSING],
            [MASK_MISSING, MASK_OBSERVED],
            [MASK_MISSING, MASK_MISSING],
            [MASK_MISSING, MASK_OBSERVED],
        ],
    ],
    dtype=jnp.float32,
)

_LOCF_EXPECTED = jnp.asarray(
    [
        [
            [2.0, _MEAN_1],
            [4.0, _MEAN_1],
            [4.0, 10.0],
            [8.0, 10.0],
            [10.0, 16.0],
            [12.0, 16.0],
        ],
        [
            [_MEAN_0, 1.0],
            [_MEAN_0, 1.0],
            [_MEAN_0, 1.0],
            [_MEAN_0, 7.0],
            [_MEAN_0, 7.0],
            [_MEAN_0, 13.0],
        ],
    ],
    dtype=jnp.float32,
)

# Feature 0 t=2 is halfway from 4 to 8. Feature 1 t=3 is halfway from 10 to 16.
# Feature 1 of batch 1: t=1 is 3, t=2 is 5, t=4 is 10.
_LINEAR_EXPECTED = jnp.asarray(
    [
        [
            [2.0, _MEAN_1],
            [4.0, _MEAN_1],
            [6.0, 10.0],
            [8.0, 13.0],
            [10.0, 16.0],
            [12.0, _MEAN_1],
        ],
        [
            [_MEAN_0, 1.0],
            [_MEAN_0, 3.0],
            [_MEAN_0, 5.0],
            [_MEAN_0, 7.0],
            [_MEAN_0, 10.0],
            [_MEAN_0, 13.0],
        ],
    ],
    dtype=jnp.float32,
)


def _assert_close(actual: jnp.ndarray, expected: jnp.ndarray) -> None:
    """Assert matching shape, float32 dtype, and numeric agreement.

    Parameters
    ----------
    actual : jax.Array
        Imputer output.
    expected : jax.Array
        Hand-computed fill.
    """
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert bool(jnp.allclose(actual, expected, rtol=_RTOL, atol=_ATOL))


def test_locf_hand_computed_interior_and_edges() -> None:
    """LOCF matches hand fills and leaves the context tensor unchanged."""
    original = jnp.array(_VALUES)
    filled = locf_impute(_VALUES, _MASK, _FEATURE_MEANS)
    _assert_close(filled, _LOCF_EXPECTED)
    assert bool(jnp.array_equal(_VALUES, original))


def test_linear_hand_computed_interior_and_edges() -> None:
    """Linear matches hand fills, including mean-valued leading and trailing gaps."""
    original = jnp.array(_VALUES)
    filled = linear_impute(_VALUES, _MASK, _FEATURE_MEANS)
    _assert_close(filled, _LINEAR_EXPECTED)
    assert bool(jnp.array_equal(_VALUES, original))


def test_single_observation_edge_policy() -> None:
    """One interior observation: LOCF carries it forward; linear uses the mean."""
    means = jnp.asarray([_LONE_MEAN], dtype=jnp.float32)
    values = jnp.asarray(
        [[[_SENTINEL], [_LONE_VALUE], [_SENTINEL]]],
        dtype=jnp.float32,
    )
    mask = jnp.asarray(
        [[[MASK_MISSING], [MASK_OBSERVED], [MASK_MISSING]]],
        dtype=jnp.float32,
    )
    locf_expected = jnp.asarray(
        [[[_LONE_MEAN], [_LONE_VALUE], [_LONE_VALUE]]],
        dtype=jnp.float32,
    )
    linear_expected = jnp.asarray(
        [[[_LONE_MEAN], [_LONE_VALUE], [_LONE_MEAN]]],
        dtype=jnp.float32,
    )
    _assert_close(locf_impute(values, mask, means), locf_expected)
    _assert_close(linear_impute(values, mask, means), linear_expected)


def test_observed_zero_is_a_real_neighbor() -> None:
    """Mask ``1`` at value ``0.0`` is context; mask ``0`` at ``0.0`` is a gap."""
    means = jnp.asarray([_LONE_MEAN], dtype=jnp.float32)
    values = jnp.asarray(
        [[[0.0], [_SENTINEL], [4.0]]],
        dtype=jnp.float32,
    )
    mask = jnp.asarray(
        [[[MASK_OBSERVED], [MASK_MISSING], [MASK_OBSERVED]]],
        dtype=jnp.float32,
    )
    locf_expected = jnp.asarray(
        [[[0.0], [0.0], [4.0]]],
        dtype=jnp.float32,
    )
    linear_expected = jnp.asarray(
        [[[0.0], [2.0], [4.0]]],
        dtype=jnp.float32,
    )
    _assert_close(locf_impute(values, mask, means), locf_expected)
    _assert_close(linear_impute(values, mask, means), linear_expected)


def test_impute_follows_evaluation_mask() -> None:
    """Hand fills follow ``m_eval``. Showing held-out truth under ``m_native`` changes them.

    Context is ``[10, 0, 0, 40]`` with ``m_eval = [1, 0, 0, 1]``, from
    ``m_native = [1, 1, 0, 1]`` and ``m_artificial = [1, 0, 1, 1]``.
    Ground truth ``100`` at the artificial target is withheld from context.
    Linear at t=1 is ``10 + (40 - 10) * (1 / 3) = 20``; t=2 is ``30``.
    """
    means = jnp.asarray([50.0], dtype=jnp.float32)
    context = jnp.asarray(
        [[[10.0], [_SENTINEL], [_SENTINEL], [40.0]]],
        dtype=jnp.float32,
    )
    m_eval = jnp.asarray(
        [[[MASK_OBSERVED], [MASK_MISSING], [MASK_MISSING], [MASK_OBSERVED]]],
        dtype=jnp.float32,
    )
    leaked = jnp.asarray(
        [[[10.0], [_HELD_OUT_TRUTH], [_SENTINEL], [40.0]]],
        dtype=jnp.float32,
    )
    m_native = jnp.asarray(
        [[[MASK_OBSERVED], [MASK_OBSERVED], [MASK_MISSING], [MASK_OBSERVED]]],
        dtype=jnp.float32,
    )
    locf_expected = jnp.asarray(
        [[[10.0], [10.0], [10.0], [40.0]]],
        dtype=jnp.float32,
    )
    linear_expected = jnp.asarray(
        [[[10.0], [20.0], [30.0], [40.0]]],
        dtype=jnp.float32,
    )
    locf_eval = locf_impute(context, m_eval, means)
    linear_eval = linear_impute(context, m_eval, means)
    _assert_close(locf_eval, locf_expected)
    _assert_close(linear_eval, linear_expected)
    locf_native = locf_impute(leaked, m_native, means)
    linear_native = linear_impute(leaked, m_native, means)
    assert bool(jnp.allclose(locf_native, locf_expected, atol=_ATOL)) is False
    assert bool(
        jnp.allclose(linear_native, linear_expected, atol=_ATOL)
    ) is False


def test_shape_dtype_batch_and_window() -> None:
    """Batch output matches ``(B, T, F)``; a single window round-trips rank 2."""
    locf_batch = locf_impute(_VALUES, _MASK, _FEATURE_MEANS)
    linear_batch = linear_impute(_VALUES, _MASK, _FEATURE_MEANS)
    assert locf_batch.shape == _VALUES.shape
    assert linear_batch.shape == _VALUES.shape
    assert locf_batch.dtype == jnp.float32
    assert linear_batch.dtype == jnp.float32
    window_values = _VALUES[_FIRST_BATCH]
    window_mask = _MASK[_FIRST_BATCH]
    locf_window = locf_impute(window_values, window_mask, _FEATURE_MEANS)
    linear_window = linear_impute(
        window_values,
        window_mask,
        _FEATURE_MEANS,
    )
    assert locf_window.shape == window_values.shape
    assert linear_window.shape == window_values.shape
    assert locf_window.dtype == jnp.float32
    assert linear_window.dtype == jnp.float32
    _assert_close(locf_window, locf_batch[_FIRST_BATCH])
    _assert_close(linear_window, linear_batch[_FIRST_BATCH])


def test_feature_means_length_must_match() -> None:
    """A mean vector shorter than ``F`` is rejected by both imputers."""
    wrong = jnp.asarray([_MEAN_0], dtype=jnp.float32)
    with pytest.raises(ValueError, match="n_features"):
        locf_impute(_VALUES, _MASK, wrong)
    with pytest.raises(ValueError, match="n_features"):
        linear_impute(_VALUES, _MASK, wrong)


def test_edge_policy_name_and_fallback_means() -> None:
    """Both classes report ``train_feature_mean`` and echo the stored vector."""
    locf = LocfImputer(_FEATURE_MEANS)
    linear = LinearImputer(_FEATURE_MEANS)
    assert locf.edge_policy() == EDGE_POLICY_TRAIN_FEATURE_MEAN
    assert linear.edge_policy() == EDGE_POLICY_TRAIN_FEATURE_MEAN
    assert bool(jnp.allclose(locf.fallback_means(), _FEATURE_MEANS))
    assert bool(jnp.allclose(linear.fallback_means(), _FEATURE_MEANS))
