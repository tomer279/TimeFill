"""Shared ``(x, m)`` preparation and train-feature-mean edge helpers.

Classes
-------
PreparedImpute
    Float32 ``(B, T, F)`` values and mask, plus an unbatch flag.

Functions
---------
coerce_feature_means
    Validate a finite ``(F,)`` mean vector.
align_feature_means
    Require that vector's length to equal ``F``.
as_observed_mask
    Boolean NumPy mask of cells with ``m == 1``.
prepare_impute_arrays
    Accept ``(B, T, F)`` or ``(T, F)`` and enforce the schema contract.
apply_point_fill
    Run a batched filler, keep observed cells, restore rank.

Constants
---------
EDGE_POLICY_TRAIN_FEATURE_MEAN
    Name of the locked fallback (``configs/bench_m3.yaml``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from src.data.schema import (
    BATCH_AXIS,
    FEATURE_AXIS,
    MASK_MISSING,
    MASK_OBSERVED,
    MISSING_VALUE_SENTINEL,
    N_AXES,
)

EDGE_POLICY_TRAIN_FEATURE_MEAN = "train_feature_mean"

_WINDOW_RANK = 2
_MEAN_RANK = 1
_MEAN_AXIS = 0
_ACCEPTED_RANKS = (_WINDOW_RANK, N_AXES)
_MASK_TOL = 1e-6
_BROADCAST_BATCH = 1
_BROADCAST_TIME = 1

_FillFn = Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray],
    jnp.ndarray,
]


@dataclass(frozen=True)
class PreparedImpute:
    """Batched imputer inputs coerced to float32 ``(B, T, F)``.

    Public methods
    --------------
    n_features
        Return feature count ``F``.
    restore
        Drop the leading batch axis when the caller passed ``(T, F)``.

    Instance variables
    ------------------
    values : jax.Array
        Float32 values shaped ``(B, T, F)``. Missing cells are the sentinel.
    mask : jax.Array
        Float32 evaluation mask shaped ``(B, T, F)``.
    unbatched : bool
        True when the caller passed rank-2 arrays.
    """

    values: jnp.ndarray
    mask: jnp.ndarray
    unbatched: bool

    def n_features(self) -> int:
        """Return the feature count ``F``.

        Returns
        -------
        int
            Size of the feature axis.
        """
        return int(self.values.shape[FEATURE_AXIS])

    def restore(self, filled: jnp.ndarray) -> jnp.ndarray:
        """Match ``filled`` to the rank the caller passed in.

        Parameters
        ----------
        filled : jax.Array
            Batch-shaped fill, same shape as ``values``.

        Returns
        -------
        jax.Array
            ``filled`` with the batch axis removed when ``unbatched`` is set.

        Raises
        ------
        ValueError
            If ``filled`` does not match the prepared values shape.
        """
        if filled.shape != self.values.shape:
            raise ValueError(
                "filled shape must match prepared values; "
                f"got filled={tuple(filled.shape)}, "
                f"values={tuple(self.values.shape)}"
            )
        if self.unbatched:
            return jnp.squeeze(filled, axis=BATCH_AXIS)
        return filled


def coerce_feature_means(feature_means: jnp.ndarray) -> jnp.ndarray:
    """Validate a finite per-feature mean vector.

    Parameters
    ----------
    feature_means : array-like
        One mean per feature, shaped ``(F,)``. Units must match ``values``.

    Returns
    -------
    jax.Array
        Float32 copy shaped ``(F,)``.

    Raises
    ------
    ValueError
        If the vector is not 1-D, empty, or non-finite.
    """
    means = jnp.asarray(feature_means, dtype=jnp.float32)
    if means.ndim != _MEAN_RANK:
        raise ValueError(
            "feature_means must be a 1-D vector shaped (F,); "
            f"got ndim={int(means.ndim)} shape={tuple(means.shape)}"
        )
    if int(means.shape[_MEAN_AXIS]) == 0:
        raise ValueError("feature_means must contain one mean per feature")
    if not bool(jnp.all(jnp.isfinite(means))):
        raise ValueError("feature_means must be finite")
    return means


def align_feature_means(
        feature_means: jnp.ndarray,
        n_features: int,
) -> jnp.ndarray:
    """Require ``feature_means`` to have one entry per feature.

    Parameters
    ----------
    feature_means : array-like
        Candidate mean vector.
    n_features : int
        Feature count ``F`` of the window being filled.

    Returns
    -------
    jax.Array
        Float32 means shaped ``(F,)``.

    Raises
    ------
    ValueError
        If coercion fails or the length differs from ``n_features``.
    """
    means = coerce_feature_means(feature_means)
    mean_count = int(means.shape[_MEAN_AXIS])
    if mean_count != int(n_features):
        raise ValueError(
            "feature_means length must equal n_features; "
            f"got length={mean_count}, n_features={int(n_features)}"
        )
    return means


def as_observed_mask(mask: jnp.ndarray) -> np.ndarray:
    """Return True where the evaluation mask marks an observation.

    Parameters
    ----------
    mask : array-like
        Float mask using ``1`` observed and ``0`` missing.

    Returns
    -------
    numpy.ndarray
        Boolean array, same shape as ``mask``.
    """
    mask_np = np.asarray(mask, dtype=np.float32)
    return np.abs(mask_np - np.float32(MASK_OBSERVED)) < _MASK_TOL


def broadcast_feature_means(feature_means: jnp.ndarray) -> np.ndarray:
    """Reshape validated means to ``(1, 1, F)`` for ``(B, T, F)`` fills.

    Parameters
    ----------
    feature_means : array-like
        Finite means shaped ``(F,)``.

    Returns
    -------
    numpy.ndarray
        Float32 array shaped ``(1, 1, F)``.
    """
    means = np.asarray(coerce_feature_means(feature_means), dtype=np.float32)
    feature_count = int(means.shape[_MEAN_AXIS])
    return means.reshape(_BROADCAST_BATCH, _BROADCAST_TIME, feature_count)


def prepare_impute_arrays(values: jnp.ndarray, mask: jnp.ndarray) -> PreparedImpute:
    """Validate ``(values, mask)`` and return a float32 batch.

    Rank-2 ``(T, F)`` inputs gain a leading batch axis. Missing cells must
    equal ``MISSING_VALUE_SENTINEL``. The mask is the evaluation observation
    mask available at impute time.

    Parameters
    ----------
    values : array-like
        Context values shaped ``(B, T, F)`` or ``(T, F)``.
    mask : array-like
        Evaluation mask, same shape as ``values``.

    Returns
    -------
    PreparedImpute
        Batched float32 arrays and whether the batch axis was added.

    Raises
    ------
    ValueError
        If rank, shape, the mask alphabet, or the missing sentinel fails.
    """
    values_arr, mask_arr = _coerce_pair(values, mask)
    values_arr, mask_arr, unbatched = _ensure_batch(values_arr, mask_arr)
    prepared = _as_float_batch(values_arr, mask_arr, unbatched)
    _validate_mask_alphabet(prepared.mask)
    _validate_missing_sentinel(prepared.values, prepared.mask)
    return prepared


def apply_point_fill(
        feature_means: jnp.ndarray,
        values: jnp.ndarray,
        mask: jnp.ndarray,
        fill_batched: _FillFn,
) -> jnp.ndarray:
    """Fill a window and return values with the caller's original rank.

    Observed cells are copied from ``values`` after the filler runs. The
    result is values only: this helper does not return a second mask.

    Parameters
    ----------
    feature_means : array-like
        Train-feature means shaped ``(F,)``, same units as ``values``.
    values : array-like
        Context values shaped ``(B, T, F)`` or ``(T, F)``.
    mask : array-like
        Evaluation observation mask, same shape as ``values``.
    fill_batched : callable
        Maps a ``(B, T, F)`` batch ``(values, mask, means)`` to filled values.

    Returns
    -------
    jax.Array
        Float32 fill, same shape as ``values``.
    """
    prepared = prepare_impute_arrays(values, mask)
    means = align_feature_means(feature_means, prepared.n_features())
    filled = fill_batched(prepared.values, prepared.mask, means)
    kept = _keep_observed(prepared.mask, prepared.values, filled)
    return prepared.restore(kept)


def _as_float_batch(
        values: jnp.ndarray,
        mask: jnp.ndarray,
        unbatched: bool,
) -> PreparedImpute:
    """Cast a batched pair to float32.

    Parameters
    ----------
    values : jax.Array
        Values shaped ``(B, T, F)``.
    mask : jax.Array
        Mask shaped ``(B, T, F)``.
    unbatched : bool
        Whether the caller passed rank 2.

    Returns
    -------
    PreparedImpute
        Float32 batch.
    """
    return PreparedImpute(
        values=values.astype(jnp.float32),
        mask=mask.astype(jnp.float32),
        unbatched=unbatched,
    )


def _coerce_pair(
        values: jnp.ndarray,
        mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Check that ``values`` and ``mask`` share a supported shape.

    Parameters
    ----------
    values : array-like
        Candidate value tensor.
    mask : array-like
        Candidate mask tensor.

    Returns
    -------
    tuple of jax.Array
        ``(values, mask)`` as JAX arrays.

    Raises
    ------
    ValueError
        If shapes differ or the rank is neither 2 nor 3.
    """
    values_arr = jnp.asarray(values)
    mask_arr = jnp.asarray(mask)
    if values_arr.shape != mask_arr.shape:
        raise ValueError(
            "values and mask shapes must match; "
            f"got values={tuple(values_arr.shape)}, "
            f"mask={tuple(mask_arr.shape)}"
        )
    if values_arr.ndim not in _ACCEPTED_RANKS:
        raise ValueError(
            "values must have shape (T, F) or (B, T, F); "
            f"got ndim={int(values_arr.ndim)}"
        )
    return values_arr, mask_arr


def _ensure_batch(
        values: jnp.ndarray,
        mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, bool]:
    """Add a leading batch axis to rank-2 inputs.

    Parameters
    ----------
    values : jax.Array
        Values shaped ``(B, T, F)`` or ``(T, F)``.
    mask : jax.Array
        Mask with the same shape as ``values``.

    Returns
    -------
    values_b : jax.Array
        Values shaped ``(B, T, F)``.
    mask_b : jax.Array
        Mask shaped ``(B, T, F)``.
    unbatched : bool
        True when a batch axis was inserted.
    """
    if values.ndim == N_AXES:
        return values, mask, False
    batched_values = values[jnp.newaxis, ...]
    batched_mask = mask[jnp.newaxis, ...]
    return batched_values, batched_mask, True


def _validate_mask_alphabet(mask: jnp.ndarray) -> None:
    """Require every mask entry to be observed or missing.

    Parameters
    ----------
    mask : jax.Array
        Float32 mask shaped ``(B, T, F)``.

    Raises
    ------
    ValueError
        If any entry lies outside ``{0, 1}``.
    """
    is_observed = jnp.abs(mask - MASK_OBSERVED) < _MASK_TOL
    is_missing = jnp.abs(mask - MASK_MISSING) < _MASK_TOL
    if not bool(jnp.all(is_observed | is_missing)):
        raise ValueError(
            "mask entries must be 0.0 (missing) or 1.0 (observed)"
        )


def _validate_missing_sentinel(values: jnp.ndarray, mask: jnp.ndarray) -> None:
    """Require missing cells to hold the schema sentinel.

    Parameters
    ----------
    values : jax.Array
        Float32 values shaped ``(B, T, F)``.
    mask : jax.Array
        Float32 mask shaped ``(B, T, F)``.

    Raises
    ------
    ValueError
        If a missing cell differs from ``MISSING_VALUE_SENTINEL``.
    """
    missing = jnp.abs(mask - MASK_MISSING) < _MASK_TOL
    missing_vals = values[missing]
    if int(missing_vals.size) == 0:
        return
    ok = jnp.all(
        jnp.abs(missing_vals - MISSING_VALUE_SENTINEL) < _MASK_TOL
    )
    if not bool(ok):
        raise ValueError(
            "missing cells in values must equal MISSING_VALUE_SENTINEL "
            f"({MISSING_VALUE_SENTINEL})"
        )


def _keep_observed(
        mask: jnp.ndarray,
        values: jnp.ndarray,
        filled: jnp.ndarray,
) -> jnp.ndarray:
    """Copy observed context cells onto the fill.

    Parameters
    ----------
    mask : jax.Array
        Evaluation mask shaped ``(B, T, F)``.
    values : jax.Array
        Original context values.
    filled : jax.Array
        Filler output, same shape as ``values``.

    Returns
    -------
    jax.Array
        Fill with observed positions taken from ``values``.
    """
    observed = jnp.abs(mask - MASK_OBSERVED) < _MASK_TOL
    return jnp.where(observed, values, filled)


__all__ = [
    "EDGE_POLICY_TRAIN_FEATURE_MEAN",
    "PreparedImpute",
    "align_feature_means",
    "apply_point_fill",
    "as_observed_mask",
    "broadcast_feature_means",
    "coerce_feature_means",
    "prepare_impute_arrays",
]
