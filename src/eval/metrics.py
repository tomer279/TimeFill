"""Point and probabilistic metrics on masked imputation targets (WP6).

Classes
-------
EvalTensors
    Packed ``truth``, ``mean``, ``samples``, and ``mask`` for metrics.
MetricsTable
    Scalar MAE / RMSE / CRPS / coverage with export helpers.
ImputationEvaluator
    Compute the M1 metrics table and format it for CLI printing.

Functions
---------
masked_mae
    Mean absolute error on cells where ``m == 0``.
masked_mae_per_feature
    Per-feature masked MAE shaped ``(F,)``.
masked_rmse
    Root mean squared error on masked cells.
masked_rmse_per_feature
    Per-feature masked RMSE shaped ``(F,)``.
empirical_crps
    Ensemble CRPS averaged over masked cells.
empirical_crps_per_feature
    Per-feature empirical CRPS shaped ``(F,)``.
interval_coverage
    Fraction of truth inside a central predictive interval.
interval_coverage_per_feature
    Per-feature interval coverage shaped ``(F,)``.
write_mean_median_csvs
    Write raw ``mean.csv`` / ``median.csv`` array dumps (not app export).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from src.data.schema import (
    BATCH_AXIS,
    FEATURE_AXIS,
    MASK_MISSING,
    N_AXES,
    TIME_AXIS,
)

_SAMPLE_AXIS = 1
_PAIR_AXIS_A = 1
_PAIR_AXIS_B = 2
_MASK_COMPARE_TOL = 0.5
_HALF = 2.0
_LEVEL_50 = 0.5
_LEVEL_90 = 0.9
_COVERAGE_LEVELS = (_LEVEL_50, _LEVEL_90)
_EXPECTED_SAMPLE_RANK = 4
_POINT_ESTIMATE_MEAN = "mean"
_POINT_ESTIMATE_MEDIAN = "median"
_ALLOWED_POINT_ESTIMATE = frozenset(
    {_POINT_ESTIMATE_MEAN, _POINT_ESTIMATE_MEDIAN}
)
_MEAN_CSV_NAME = "mean.csv"
_MEDIAN_CSV_NAME = "median.csv"


@dataclass(frozen=True)
class EvalTensors:
    """Tensors needed for masked point and probabilistic metrics.

    Public methods
    --------------
    validate_shapes
        Check ``(B, T, F)`` / ``(B, K, T, F)`` layout consistency.
    missing_weights
        Return ``1`` on missing cells and ``0`` on observed cells.

    Instance variables
    ------------------
    truth : jax.Array
        Clean ground truth shaped ``(B, T, F)``.
    mean : jax.Array
        Posterior mean fill shaped ``(B, T, F)``.
    samples : jax.Array
        Posterior samples shaped ``(B, K, T, F)``.
    mask : jax.Array
        Observation mask ``(B, T, F)``; ``1`` observed, ``0`` missing.
    """

    truth: jnp.ndarray
    mean: jnp.ndarray
    samples: jnp.ndarray
    mask: jnp.ndarray

    def validate_shapes(self) -> None:
        """Raise ``ValueError`` if tensor shapes are inconsistent."""
        if self.truth.ndim != N_AXES:
            raise ValueError(
                "truth must have shape (B, T, F); "
                f"got {self.truth.shape}"
            )
        if self.mean.shape != self.truth.shape:
            raise ValueError(
                "mean shape must match truth; "
                f"got {self.mean.shape} vs {self.truth.shape}"
            )
        if self.mask.shape != self.truth.shape:
            raise ValueError(
                "mask shape must match truth; "
                f"got {self.mask.shape} vs {self.truth.shape}"
            )
        if self.samples.ndim != _EXPECTED_SAMPLE_RANK:
            raise ValueError(
                "samples must have shape (B, K, T, F); "
                f"got {self.samples.shape}"
            )
        expected = (
            self.truth.shape[BATCH_AXIS],
            self.samples.shape[_SAMPLE_AXIS],
            self.truth.shape[TIME_AXIS],
            self.truth.shape[FEATURE_AXIS],
        )
        if tuple(self.samples.shape) != expected:
            raise ValueError(
                "samples must align with truth as (B, K, T, F); "
                f"got {self.samples.shape}, expected {expected}"
            )

    def missing_weights(self) -> jnp.ndarray:
        """Return float weights that select missing (target) cells.

        Returns
        -------
        jax.Array
            ``(B, T, F)`` weights; ``1`` where ``m == 0``.
        """
        is_missing = jnp.abs(self.mask - MASK_MISSING) < _MASK_COMPARE_TOL
        return is_missing.astype(jnp.float32)


@dataclass(frozen=True)
class MetricsTable:
    """Tiny M1 metrics table for masked imputation quality.

    Public methods
    --------------
    as_dict
        Export metrics as a name → float mapping.
    rows
        Export ordered ``(name, value)`` rows for printing.

    Instance variables
    ------------------
    mae : float
        Masked mean absolute error of the posterior mean.
    rmse : float
        Masked root mean squared error of the posterior mean.
    crps : float
        Empirical CRPS from ``K`` samples on masked cells.
    coverage_50 : float
        Empirical coverage of the central 50% interval.
    coverage_90 : float
        Empirical coverage of the central 90% interval.
    """

    mae: float
    rmse: float
    crps: float
    coverage_50: float
    coverage_90: float

    def as_dict(self) -> dict[str, float]:
        """Return metrics as a plain dict.

        Returns
        -------
        dict of str to float
            Metric name to scalar value.
        """
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "crps": self.crps,
            "coverage_50": self.coverage_50,
            "coverage_90": self.coverage_90,
        }

    def rows(self) -> list[tuple[str, float]]:
        """Return ordered rows for a tiny printed table.

        Returns
        -------
        list of tuple of str and float
            ``(metric_name, value)`` pairs.
        """
        mapping = self.as_dict()
        return list(mapping.items())


class ImputationEvaluator:
    """Compute MAE / RMSE / CRPS / coverage on masked cells.

    Public methods
    --------------
    compute
        Build a ``MetricsTable`` from ``EvalTensors``.
    format_table
        Render a metrics table as an aligned text block.

    Instance variables
    ------------------
    coverage_levels : tuple of float
        Central predictive interval levels (default 0.5 and 0.9).
    point_estimate : str
        ``mean`` (legacy) or ``median`` for MAE only.
    """

    def __init__(
            self,
            coverage_levels: tuple[float, ...] = _COVERAGE_LEVELS,
            point_estimate: str = _POINT_ESTIMATE_MEAN,
    ) -> None:
        """Store coverage levels and MAE point-estimate mode.

        Parameters
        ----------
        coverage_levels : tuple of float, optional
            Central interval levels in ``(0, 1]``. Defaults to
            ``(0.5, 0.9)``.
        point_estimate : str, optional
            ``mean`` (legacy) or ``median`` for MAE only.
        """
        if len(coverage_levels) != len(_COVERAGE_LEVELS):
            raise ValueError(
                "M1 evaluator expects exactly two coverage levels "
                f"(50% and 90%); got {coverage_levels!r}"
            )
        for level in coverage_levels:
            if not 0.0 < level <= 1.0:
                raise ValueError(
                    f"coverage level must be in (0, 1], got {level}"
                )
        if point_estimate not in _ALLOWED_POINT_ESTIMATE:
            raise ValueError(
                "point_estimate must be one of "
                f"{sorted(_ALLOWED_POINT_ESTIMATE)}, "
                f"got {point_estimate!r}"
            )
        self.coverage_levels = coverage_levels
        self.point_estimate = point_estimate

    def compute(self, tensors: EvalTensors) -> MetricsTable:
        """Compute the M1 masked metrics table.

        Parameters
        ----------
        tensors : EvalTensors
            Truth, posterior mean, samples, and mask.

        Returns
        -------
        MetricsTable
            MAE, RMSE, CRPS, and 50%/90% coverage.
        """
        tensors.validate_shapes()
        level_50, level_90 = self.coverage_levels
        return MetricsTable(
            mae=float(
                masked_mae(tensors, point_estimate=self.point_estimate)
            ),
            rmse=float(masked_rmse(tensors)),
            crps=float(empirical_crps(tensors)),
            coverage_50=float(interval_coverage(tensors, level_50)),
            coverage_90=float(interval_coverage(tensors, level_90)),
        )

    def format_table(self, table: MetricsTable) -> str:
        """Format metrics as a small aligned text table.

        Parameters
        ----------
        table : MetricsTable
            Metrics produced by ``compute``.

        Returns
        -------
        str
            Multi-line table suitable for stdout.
        """
        lines = ["metric          value", "--------------  ----------"]
        for name, value in table.rows():
            lines.append(f"{name:<14}  {value:10.6f}")
        return "\n".join(lines)


def masked_mae(
        tensors: EvalTensors,
        point_estimate: str = _POINT_ESTIMATE_MEAN,
) -> jnp.ndarray:
    """Mean absolute error on missing cells.

    Parameters
    ----------
    tensors : EvalTensors
        Must include ``truth``, ``mean``, ``samples``, and ``mask``.
    point_estimate : str, optional
        ``mean`` (legacy) or ``median`` over the sample axis ``K``.

    Returns
    -------
    jax.Array
        Scalar MAE over masked entries.
    """
    if point_estimate not in _ALLOWED_POINT_ESTIMATE:
        raise ValueError(
            "point_estimate must be one of "
            f"{sorted(_ALLOWED_POINT_ESTIMATE)}, "
            f"got {point_estimate!r}"
        )
    weights = tensors.missing_weights()
    fill = _point_fill(tensors, point_estimate)
    abs_err = jnp.abs(fill - tensors.truth)
    return _weighted_mean(abs_err, weights)


def masked_mae_per_feature(
        tensors: EvalTensors,
        point_estimate: str = _POINT_ESTIMATE_MEAN,
) -> jnp.ndarray:
    """Per-feature mean absolute error on missing cells.

    Parameters
    ----------
    tensors : EvalTensors
        Must include ``truth``, ``mean``, ``samples``, and ``mask``.
    point_estimate : str, optional
        ``mean`` (legacy) or ``median`` over the sample axis ``K``.

    Returns
    -------
    jax.Array
        MAE per feature shaped ``(F,)``; ``NaN`` when a feature has no
        target cells.
    """
    if point_estimate not in _ALLOWED_POINT_ESTIMATE:
        raise ValueError(
            "point_estimate must be one of "
            f"{sorted(_ALLOWED_POINT_ESTIMATE)}, "
            f"got {point_estimate!r}"
        )
    weights = tensors.missing_weights()
    fill = _point_fill(tensors, point_estimate)
    abs_err = jnp.abs(fill - tensors.truth)
    return _weighted_mean_per_feature(abs_err, weights)


def masked_rmse(tensors: EvalTensors) -> jnp.ndarray:
    """Root mean squared error on missing cells using the posterior mean.

    Parameters
    ----------
    tensors : EvalTensors
        Must include ``truth``, ``mean``, and ``mask``.

    Returns
    -------
    jax.Array
        Scalar RMSE over masked entries.
    """
    weights = tensors.missing_weights()
    sq_err = (tensors.mean - tensors.truth) ** 2
    return jnp.sqrt(_weighted_mean(sq_err, weights))


def masked_rmse_per_feature(tensors: EvalTensors) -> jnp.ndarray:
    """Per-feature RMSE on missing cells using the posterior mean.

    Parameters
    ----------
    tensors : EvalTensors
        Must include ``truth``, ``mean``, and ``mask``.

    Returns
    -------
    jax.Array
        RMSE per feature shaped ``(F,)``.
    """
    weights = tensors.missing_weights()
    sq_err = (tensors.mean - tensors.truth) ** 2
    return jnp.sqrt(_weighted_mean_per_feature(sq_err, weights))


def empirical_crps(tensors: EvalTensors) -> jnp.ndarray:
    """Empirical ensemble CRPS averaged over missing cells.

    Uses
    ``mean_k |x_k - y| - 0.5 * mean_{k,j} |x_k - x_j|``.

    Parameters
    ----------
    tensors : EvalTensors
        Must include ``truth``, ``samples``, and ``mask``.

    Returns
    -------
    jax.Array
        Scalar CRPS over masked entries.
    """
    return _weighted_mean(
        _cell_crps(tensors),
        tensors.missing_weights(),
    )


def empirical_crps_per_feature(tensors: EvalTensors) -> jnp.ndarray:
    """Per-feature empirical ensemble CRPS on missing cells.

    Parameters
    ----------
    tensors : EvalTensors
        Must include ``truth``, ``samples``, and ``mask``.

    Returns
    -------
    jax.Array
        CRPS per feature shaped ``(F,)``.
    """
    return _weighted_mean_per_feature(
        _cell_crps(tensors),
        tensors.missing_weights(),
    )


def interval_coverage(
        tensors: EvalTensors,
        level: float,
) -> jnp.ndarray:
    """Empirical coverage of a central predictive interval.

    Parameters
    ----------
    tensors : EvalTensors
        Must include ``truth``, ``samples``, and ``mask``.
    level : float
        Central interval level in ``(0, 1]`` (e.g. ``0.9`` → 5–95%).

    Returns
    -------
    jax.Array
        Fraction of masked truth cells inside the interval.
    """
    return _weighted_mean(
        _interval_inside(tensors, level).astype(jnp.float32),
        tensors.missing_weights(),
    )


def interval_coverage_per_feature(
        tensors: EvalTensors,
        level: float,
) -> jnp.ndarray:
    """Per-feature empirical coverage of a central predictive interval.

    Parameters
    ----------
    tensors : EvalTensors
        Must include ``truth``, ``samples``, and ``mask``.
    level : float
        Central interval level in ``(0, 1]``.

    Returns
    -------
    jax.Array
        Coverage per feature shaped ``(F,)``.
    """
    return _weighted_mean_per_feature(
        _interval_inside(tensors, level).astype(jnp.float32),
        tensors.missing_weights(),
    )


def write_mean_median_csvs(
        out_dir: Path,
        mean: jnp.ndarray,
        median: jnp.ndarray,
) -> tuple[Path, Path]:
    """Write ``mean.csv`` and ``median.csv`` under ``out_dir``.

    Raw array dumps (last axis = features), not the app download format.
    Prefer ``artifacts/m7_5/…``; do not write into frozen claim dirs.

    Parameters
    ----------
    out_dir : pathlib.Path
        Destination directory (created if needed).
    mean : jax.Array
        Posterior mean fill.
    median : jax.Array
        Posterior median fill.

    Returns
    -------
    mean_path, median_path : pathlib.Path
        Written file paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mean_path = out_dir / _MEAN_CSV_NAME
    median_path = out_dir / _MEDIAN_CSV_NAME
    _write_array_csv(mean_path, mean)
    _write_array_csv(median_path, median)
    return mean_path, median_path


def _point_fill(
        tensors: EvalTensors,
        point_estimate: str,
) -> jnp.ndarray:
    """Return the configured point fill ``(B, T, F)``."""
    if point_estimate == _POINT_ESTIMATE_MEDIAN:
        return jnp.median(tensors.samples, axis=_SAMPLE_AXIS)
    return tensors.mean


def _write_array_csv(path: Path, array: jnp.ndarray) -> None:
    """Write an array as CSV rows shaped ``(-1, n_features)``."""
    flat = np.asarray(array, dtype=np.float64)
    if flat.ndim == 1:
        rows = flat.reshape(1, -1)
    else:
        rows = flat.reshape(-1, flat.shape[-1])
    np.savetxt(path, rows, delimiter=",")


def _cell_crps(tensors: EvalTensors) -> jnp.ndarray:
    """Return per-cell empirical CRPS ``(B, T, F)``."""
    tensors.validate_shapes()
    truth_exp = jnp.expand_dims(tensors.truth, axis=_SAMPLE_AXIS)
    abs_err = jnp.mean(
        jnp.abs(tensors.samples - truth_exp),
        axis=_SAMPLE_AXIS,
    )
    delta = (
        jnp.expand_dims(tensors.samples, axis=_PAIR_AXIS_B)
        - jnp.expand_dims(tensors.samples, axis=_PAIR_AXIS_A)
    )
    spread = (_HALF ** -1) * jnp.mean(
        jnp.abs(delta),
        axis=(_PAIR_AXIS_A, _PAIR_AXIS_B),
    )
    return abs_err - spread


def _interval_inside(
        tensors: EvalTensors,
        level: float,
) -> jnp.ndarray:
    """Return a boolean mask of truth inside the central interval."""
    if not 0.0 < level <= 1.0:
        raise ValueError(f"level must be in (0, 1], got {level}")
    tensors.validate_shapes()
    tail = (1.0 - level) / _HALF
    low = jnp.quantile(tensors.samples, tail, axis=_SAMPLE_AXIS)
    high = jnp.quantile(tensors.samples, 1.0 - tail, axis=_SAMPLE_AXIS)
    return (tensors.truth >= low) & (tensors.truth <= high)


def _weighted_mean(
        values: jnp.ndarray,
        weights: jnp.ndarray,
) -> jnp.ndarray:
    """Weighted mean; ``NaN`` when the weight sum is zero.

    Parameters
    ----------
    values : jax.Array
        Values shaped like ``weights``.
    weights : jax.Array
        Non-negative weights.

    Returns
    -------
    jax.Array
        Scalar weighted mean, or ``NaN`` if there are no weights.
    """
    total = jnp.sum(weights)
    return jnp.where(
        total > 0.0,
        jnp.sum(values * weights) / total,
        jnp.nan,
    )


def _weighted_mean_per_feature(
        values: jnp.ndarray,
        weights: jnp.ndarray,
) -> jnp.ndarray:
    """Per-feature weighted mean; ``NaN`` where a feature has no weight."""
    counts = jnp.sum(weights, axis=(BATCH_AXIS, TIME_AXIS))
    sums = jnp.sum(values * weights, axis=(BATCH_AXIS, TIME_AXIS))
    return jnp.where(counts > 0.0, sums / counts, jnp.nan)
