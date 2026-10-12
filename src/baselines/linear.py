"""Per-feature linear interpolation baseline for M3.

Classes
-------
LinearImputer
    Interpolate interior gaps; fall back to a train-feature mean on edges.

Functions
---------
linear_impute
    Functional wrapper around ``LinearImputer.impute``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from src.baselines.protocol import (
    EDGE_POLICY_TRAIN_FEATURE_MEAN,
    apply_point_fill,
    as_observed_mask,
    coerce_feature_means,
)
from src.data.schema import BATCH_AXIS

_SERIES_AXIS = 0
_WINDOW_FEATURE_AXIS = 1
_NO_OBSERVATIONS = 0
_ONE_OBSERVATION = 1
_ONLY_KNOT = 0


class LinearImputer:
    """Linearly interpolate each feature along integer time indexes.

    An interior gap is the open interval between the nearest earlier and
    nearest later observed cells. Those cells are linearly interpolated.
    A leading gap, a trailing gap, times around a lone observation, and an
    all-missing feature take ``feature_means`` for that feature. Observed
    cells stay equal to the input.

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
        """Store the train-feature mean used when a gap has no bracketing pair.

        Parameters
        ----------
        feature_means : array-like
            Finite vector shaped ``(F,)``, same units as the values to fill.
        """
        self._feature_means = coerce_feature_means(feature_means)

    def impute(self, values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """Fill missing cells by linear interpolation on observed indexes.

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
            _linear_fill,
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


def linear_impute(
        values: jnp.ndarray,
        mask: jnp.ndarray,
        feature_means: jnp.ndarray,
) -> jnp.ndarray:
    """Fill interior gaps by linear interpolation and edges by ``feature_means``.

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
    return LinearImputer(feature_means).impute(values, mask)


def _linear_fill(
        values: jnp.ndarray,
        mask: jnp.ndarray,
        feature_means: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate one prepared ``(B, T, F)`` batch.

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
    observed = as_observed_mask(mask)
    means = np.asarray(feature_means, dtype=np.float32)
    filled = np.empty_like(series)
    _fill_all_windows(filled, series, observed, means)
    return jnp.asarray(filled, dtype=jnp.float32)


def _fill_all_windows(
        filled: np.ndarray,
        series: np.ndarray,
        observed: np.ndarray,
        means: np.ndarray,
) -> None:
    """Fill every window of a ``(B, T, F)`` batch in place.

    Parameters
    ----------
    filled : numpy.ndarray
        Output buffer shaped ``(B, T, F)``.
    series : numpy.ndarray
        Context values shaped ``(B, T, F)``.
    observed : numpy.ndarray
        Boolean evaluation mask shaped ``(B, T, F)``.
    means : numpy.ndarray
        Float32 means shaped ``(F,)``.
    """
    batch_count = int(series.shape[BATCH_AXIS])
    for batch_index in range(batch_count):
        _fill_linear_batch(
            filled[batch_index],
            series[batch_index],
            observed[batch_index],
            means,
        )


def _fill_linear_batch(
        filled_tf: np.ndarray,
        series_tf: np.ndarray,
        observed_tf: np.ndarray,
        means: np.ndarray,
) -> None:
    """Fill one window shaped ``(T, F)`` in place.

    Parameters
    ----------
    filled_tf : numpy.ndarray
        Output buffer shaped ``(T, F)``.
    series_tf : numpy.ndarray
        Context values shaped ``(T, F)``.
    observed_tf : numpy.ndarray
        Boolean mask shaped ``(T, F)``.
    means : numpy.ndarray
        Float32 means shaped ``(F,)``.
    """
    feature_count = int(series_tf.shape[_WINDOW_FEATURE_AXIS])
    for feature_index in range(feature_count):
        filled_tf[:, feature_index] = _linear_series(
            series_tf[:, feature_index],
            observed_tf[:, feature_index],
            float(means[feature_index]),
        )


def _linear_series(
        series: np.ndarray,
        observed: np.ndarray,
        fallback: float,
) -> np.ndarray:
    """Interpolate one feature series of length ``T``.

    Parameters
    ----------
    series : numpy.ndarray
        Float values shaped ``(T,)``.
    observed : numpy.ndarray
        Boolean mask shaped ``(T,)``.
    fallback : float
        Train-feature mean for this feature.

    Returns
    -------
    numpy.ndarray
        Float32 series shaped ``(T,)``.
    """
    observed_index = np.flatnonzero(observed)
    n_observed = int(observed_index.size)
    if n_observed == _NO_OBSERVATIONS:
        return _constant_series(series, fallback)
    if n_observed == _ONE_OBSERVATION:
        return _linear_one_observation(series, observed_index, fallback)
    return _interp_interior(series, observed_index, fallback)


def _constant_series(series: np.ndarray, fallback: float) -> np.ndarray:
    """Fill a series with the feature mean.

    Parameters
    ----------
    series : numpy.ndarray
        Template series whose shape is copied.
    fallback : float
        Train-feature mean.

    Returns
    -------
    numpy.ndarray
        Float32 series of the same shape as ``series``.
    """
    return np.full(series.shape, np.float32(fallback), dtype=np.float32)


def _linear_one_observation(
        series: np.ndarray,
        observed_index: np.ndarray,
        fallback: float,
) -> np.ndarray:
    """Keep the only observation and fill every other time with the mean.

    Parameters
    ----------
    series : numpy.ndarray
        Feature series shaped ``(T,)``.
    observed_index : numpy.ndarray
        The single observed time index.
    fallback : float
        Train-feature mean.

    Returns
    -------
    numpy.ndarray
        Float32 series shaped ``(T,)``.
    """
    filled = _constant_series(series, fallback)
    knot = int(observed_index[_ONLY_KNOT])
    filled[knot] = series[knot]
    return filled


def _interp_interior(
        series: np.ndarray,
        observed_index: np.ndarray,
        fallback: float,
) -> np.ndarray:
    """Interpolate between observed knots; edges receive ``fallback``.

    Parameters
    ----------
    series : numpy.ndarray
        Feature series shaped ``(T,)``.
    observed_index : numpy.ndarray
        Strictly increasing observed time indexes, at least two.
    fallback : float
        Value written before the first knot and after the last knot.

    Returns
    -------
    numpy.ndarray
        Float32 series shaped ``(T,)``.
    """
    time_index = np.arange(series.shape[_SERIES_AXIS], dtype=np.float64)
    knots = observed_index.astype(np.float64)
    knot_values = series[observed_index].astype(np.float64)
    interpolated = np.interp(
        time_index,
        knots,
        knot_values,
        left=float(fallback),
        right=float(fallback),
    )
    return interpolated.astype(np.float32)


__all__ = [
    "LinearImputer",
    "linear_impute",
]
