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
masked_rmse
    Root mean squared error on masked cells.
empirical_crps
    Ensemble CRPS averaged over masked cells.
interval_coverage
    Fraction of truth inside a central predictive interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

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
_WEIGHT_FLOOR = 1.0
_MASK_COMPARE_TOL = 0.5
_HALF = 2.0
_LEVEL_50 = 0.5
_LEVEL_90 = 0.9
_COVERAGE_LEVELS = (_LEVEL_50, _LEVEL_90)
_EXPECTED_SAMPLE_RANK = 4


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
    """

    def __init__(
            self,
            coverage_levels: tuple[float, ...] = _COVERAGE_LEVELS,
    ) -> None:
        """Store coverage levels used by ``compute``.

        Parameters
        ----------
        coverage_levels : tuple of float, optional
            Central interval levels in ``(0, 1]``. Defaults to
            ``(0.5, 0.9)``.
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
        self.coverage_levels = coverage_levels

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
            mae=float(masked_mae(tensors)),
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


def masked_mae(tensors: EvalTensors) -> jnp.ndarray:
    """Mean absolute error on missing cells using the posterior mean.

    Parameters
    ----------
    tensors : EvalTensors
        Must include ``truth``, ``mean``, and ``mask``.

    Returns
    -------
    jax.Array
        Scalar MAE over masked entries.
    """
    weights = tensors.missing_weights()
    abs_err = jnp.abs(tensors.mean - tensors.truth)
    return _weighted_mean(abs_err, weights)


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
    cell_crps = abs_err - spread
    return _weighted_mean(cell_crps, tensors.missing_weights())


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
    if not 0.0 < level <= 1.0:
        raise ValueError(f"level must be in (0, 1], got {level}")
    tensors.validate_shapes()
    tail = (1.0 - level) / _HALF
    low = jnp.quantile(tensors.samples, tail, axis=_SAMPLE_AXIS)
    high = jnp.quantile(tensors.samples, 1.0 - tail, axis=_SAMPLE_AXIS)
    inside = (tensors.truth >= low) & (tensors.truth <= high)
    return _weighted_mean(inside.astype(jnp.float32), tensors.missing_weights())


def _weighted_mean(
        values: jnp.ndarray,
        weights: jnp.ndarray,
) -> jnp.ndarray:
    """Weighted mean with a floor on the weight sum.

    Parameters
    ----------
    values : jax.Array
        Values shaped like ``weights``.
    weights : jax.Array
        Non-negative weights.

    Returns
    -------
    jax.Array
        Scalar weighted mean.
    """
    denom = jnp.maximum(jnp.sum(weights), _WEIGHT_FLOOR)
    return jnp.sum(values * weights) / denom
