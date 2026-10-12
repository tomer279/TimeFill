"""Last-observation-carried-forward baseline for M3.

Classes
-------
LocfImputer
    Per-feature LOCF with a caller-supplied train-feature mean.

Functions
---------
locf_impute
    Functional wrapper around ``LocfImputer.impute``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from src.baselines.protocol import (
    EDGE_POLICY_TRAIN_FEATURE_MEAN,
    apply_point_fill,
    as_observed_mask,
    broadcast_feature_means,
    coerce_feature_means,
)
from src.data.schema import TIME_AXIS

_FIRST_TIME = 0
_NO_PRIOR_INDEX = -1


class LocfImputer:
    """Carry the latest earlier observation forward along time.

    Each feature of each window is filled independently. A missing cell
    copies the latest earlier cell with ``m == 1``. When no earlier
    observation exists, the cell takes ``feature_means`` for that feature.
    Cells after the last observation keep that observation. Observed cells
    stay equal to the input.

    Pass means in the same units as ``values``. Time is the integer index
    along the window; irregular wall-clock is not an input.

    Public methods
    --------------
    impute
        Fill ``(values, mask)`` and return the same shape.
    fallback_means
        Return a copy of the stored mean vector.
    edge_policy
        Return the locked edge-policy name.

    Instance variables
    ------------------
    _feature_means : jax.Array
        Finite float32 means shaped ``(F,)``.
    """

    def __init__(self, feature_means: jnp.ndarray) -> None:
        """Store the train-feature mean used when LOCF has no prior observation.

        Parameters
        ----------
        feature_means : array-like
            Finite vector shaped ``(F,)``, same units as the values to fill.
        """
        self._feature_means = coerce_feature_means(feature_means)

    def impute(self, values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """Fill missing cells by carrying the last observation forward.

        Parameters
        ----------
        values : array-like
            Context values shaped ``(B, T, F)`` or ``(T, F)``. Missing cells
            must equal the schema sentinel ``0.0``.
        mask : array-like
            Evaluation observation mask, same shape as ``values``.
            ``1`` is available context and ``0`` is filled.

        Returns
        -------
        jax.Array
            Float32 values, same shape as ``values``.
        """
        return apply_point_fill(
            self._feature_means,
            values,
            mask,
            _locf_fill,
        )

    def fallback_means(self) -> jnp.ndarray:
        """Return a copy of the train-feature mean vector.

        Returns
        -------
        jax.Array
            Float32 vector shaped ``(F,)``.
        """
        return jnp.array(self._feature_means, dtype=jnp.float32)

    def edge_policy(self) -> str:
        """Return the locked edge-policy name.

        Returns
        -------
        str
            ``train_feature_mean``.
        """
        return EDGE_POLICY_TRAIN_FEATURE_MEAN


def locf_impute(
        values: jnp.ndarray,
        mask: jnp.ndarray,
        feature_means: jnp.ndarray,
) -> jnp.ndarray:
    """Fill with LOCF using ``feature_means`` when no prior observation exists.

    Parameters
    ----------
    values : array-like
        Context values shaped ``(B, T, F)`` or ``(T, F)``.
    mask : array-like
        Evaluation observation mask, same shape as ``values``.
    feature_means : array-like
        Finite means shaped ``(F,)``, same units as ``values``.

    Returns
    -------
    jax.Array
        Float32 fill, same shape as ``values``.
    """
    return LocfImputer(feature_means).impute(values, mask)


def _locf_fill(
        values: jnp.ndarray,
        mask: jnp.ndarray,
        feature_means: jnp.ndarray,
) -> jnp.ndarray:
    """Apply LOCF to one prepared ``(B, T, F)`` batch.

    Parameters
    ----------
    values : jax.Array
        Float32 context shaped ``(B, T, F)``.
    mask : jax.Array
        Float32 evaluation mask shaped ``(B, T, F)``.
    feature_means : jax.Array
        Float32 means shaped ``(F,)``.

    Returns
    -------
    jax.Array
        Float32 fill shaped ``(B, T, F)``.
    """
    series = np.asarray(values, dtype=np.float32)
    last_index = _last_observed_index(mask)
    gathered = _gather_along_time(series, last_index)
    return _apply_feature_means(gathered, last_index, feature_means)


def _last_observed_index(mask: jnp.ndarray) -> np.ndarray:
    """Index of the latest earlier observation, or ``-1`` when none exists.

    Parameters
    ----------
    mask : jax.Array
        Evaluation mask shaped ``(B, T, F)``.

    Returns
    -------
    numpy.ndarray
        Int32 indexes shaped ``(B, T, F)``.
    """
    observed = as_observed_mask(mask)
    time_index = np.arange(observed.shape[TIME_AXIS], dtype=np.int32)
    stamped = np.where(
        observed,
        time_index[np.newaxis, :, np.newaxis],
        np.int32(_NO_PRIOR_INDEX),
    )
    return np.maximum.accumulate(stamped, axis=TIME_AXIS)


def _gather_along_time(series: np.ndarray, last_index: np.ndarray) -> np.ndarray:
    """Gather series values at ``last_index``, clipping the empty-history marker.

    Parameters
    ----------
    series : numpy.ndarray
        Float32 values shaped ``(B, T, F)``.
    last_index : numpy.ndarray
        Observation indexes shaped ``(B, T, F)``.

    Returns
    -------
    numpy.ndarray
        Gathered values shaped ``(B, T, F)``.
    """
    safe_index = np.maximum(last_index, _FIRST_TIME)
    return np.take_along_axis(series, safe_index, axis=TIME_AXIS)


def _apply_feature_means(
        gathered: np.ndarray,
        last_index: np.ndarray,
        feature_means: jnp.ndarray,
) -> jnp.ndarray:
    """Use gathered history, and the feature mean where history is absent.

    Parameters
    ----------
    gathered : numpy.ndarray
        LOCF candidates shaped ``(B, T, F)``.
    last_index : numpy.ndarray
        ``-1`` marks cells with no earlier observation.
    feature_means : jax.Array
        Means shaped ``(F,)``.

    Returns
    -------
    jax.Array
        Float32 fill shaped ``(B, T, F)``.
    """
    has_history = last_index >= _FIRST_TIME
    broadcast = broadcast_feature_means(feature_means)
    filled = np.where(has_history, gathered, broadcast)
    return jnp.asarray(filled, dtype=jnp.float32)


__all__ = [
    "LocfImputer",
    "locf_impute",
]
