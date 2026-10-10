"""Tensor conventions and validation for M1 windowed time series.

Classes
-------
TimeSeriesBatch
    Container for values `x` and mask `m` with shape helpers.
TensorSchema
    Validates `(B, T, F)` layout and mask/sentinel contracts.

Functions
---------
apply_missing_sentinel
    Replace masked-out cells in `x` with the missing-value sentinel.
build_stub_batch
    Build a deterministic fake batch for WP0 tests (not sine/AR).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from src.data.config import DataConfig

BATCH_AXIS = 0
TIME_AXIS = 1
FEATURE_AXIS = 2
N_AXES = 3

MASK_OBSERVED = 1.0
MASK_MISSING = 0.0
MISSING_VALUE_SENTINEL = 0.0

_MASK_COMPARE_TOL = 1e-6


@dataclass(frozen=True)
class TimeSeriesBatch:
    """Batch of fixed-length multivariate windows with observation masks.

    Public methods
    --------------
    shape
        Return `(B, T, F)` for `x`.
    n_observed
        Count entries with `m == MASK_OBSERVED`.

    Instance variables
    ------------------
    x : jax.Array
        Values shaped `(B, T, F)`; missing cells equal the sentinel.
    m : jax.Array
        Mask shaped `(B, T, F)`; `1` observed, `0` missing.
    """

    x: jnp.ndarray
    m: jnp.ndarray

    def shape(self) -> tuple[int, int, int]:
        """Return the `(B, T, F)` shape of `x`.

        Returns
        -------
        tuple of int
            Batch, time, and feature dimensions.
        """
        batch_size = int(self.x.shape[BATCH_AXIS])
        seq_len = int(self.x.shape[TIME_AXIS])
        n_features = int(self.x.shape[FEATURE_AXIS])
        return batch_size, seq_len, n_features

    def n_observed(self) -> int:
        """Count observed entries under the mask contract.

        Returns
        -------
        int
            Number of cells with mask equal to `MASK_OBSERVED`.
        """
        observed = jnp.abs(self.m - MASK_OBSERVED) < _MASK_COMPARE_TOL
        return int(jnp.sum(observed))


class TensorSchema:
    """Validate M1 tensor contracts for values and masks.

    Public methods
    --------------
    validate_pair
        Check a raw `(x, m)` pair against layout and value rules.
    validate_batch
        Check a `TimeSeriesBatch` against this schema.

    Instance variables
    ------------------
    seq_len : int
        Expected window length `T`.
    n_features : int
        Expected feature count `F`.
    """

    def __init__(self, data_config: DataConfig) -> None:
        """Store expected `T` and `F` from the data config section.

        Parameters
        ----------
        data_config : DataConfig
            Config providing `seq_len` and `n_features`.
        """
        self.seq_len = int(data_config.seq_len)
        self.n_features = int(data_config.n_features)

    def validate_pair(self, values: jnp.ndarray, mask: jnp.ndarray) -> None:
        """Validate shapes, axes, mask alphabet, and missing sentinel.

        Parameters
        ----------
        values : jax.Array
            Array expected to have shape `(B, T, F)`.
        mask : jax.Array
            Array expected to match `values` with entries in `{0, 1}`.

        Raises
        ------
        ValueError
            If any contract check fails.
        """
        self._validate_rank(values, mask)
        self._validate_dims(values, mask)
        self._validate_mask_alphabet(mask)
        self._validate_missing_sentinel(values, mask)

    def validate_batch(self, batch: TimeSeriesBatch) -> None:
        """Validate a batch object against this schema.

        Parameters
        ----------
        batch : TimeSeriesBatch
            Batch whose `x` and `m` must satisfy the M1 contract.

        Raises
        ------
        ValueError
            If any contract check fails.
        """
        self.validate_pair(batch.x, batch.m)

    def _validate_rank(self, values: jnp.ndarray, mask: jnp.ndarray) -> None:
        """Check that values and mask are rank-3 with matching shapes.

        Parameters
        ----------
        values : jax.Array
            Value tensor expected to have shape `(B, T, F)`.
        mask : jax.Array
            Mask tensor expected to match `values` exactly in shape.

        Raises
        ------
        ValueError
            If either array has the wrong rank or the shapes differ.
        """
        if values.ndim != N_AXES:
            raise ValueError(
                "values must have rank 3 (B, T, F); "
                f"got ndim={values.ndim}"
            )
        if mask.ndim != N_AXES:
            raise ValueError(
                "mask must have rank 3 (B, T, F); "
                f"got ndim={mask.ndim}"
            )
        if values.shape != mask.shape:
            raise ValueError(
                "values and mask shapes must match; "
                f"got values={values.shape}, mask={mask.shape}"
            )

    def _validate_dims(self, values: jnp.ndarray, mask: jnp.ndarray) -> None:
        """Check that `T` and `F` match this schema's expected sizes.

        Parameters
        ----------
        values : jax.Array
            Value tensor with shape `(B, T, F)`.
        mask : jax.Array
            Mask tensor; unused after rank checks (kept for call symmetry).

        Raises
        ------
        ValueError
            If `T` or `F` differ from `seq_len` / `n_features`.
        """
        del mask  # shape equality already enforced
        seq_len = int(values.shape[TIME_AXIS])
        n_features = int(values.shape[FEATURE_AXIS])
        if seq_len != self.seq_len:
            raise ValueError(
                f"expected seq_len={self.seq_len}, got {seq_len}"
            )
        if n_features != self.n_features:
            raise ValueError(
                f"expected n_features={self.n_features}, got {n_features}"
            )

    def _validate_mask_alphabet(self, mask: jnp.ndarray) -> None:
        """Check that every mask entry is observed (`1`) or missing (`0`).

        Parameters
        ----------
        mask : jax.Array
            Mask tensor shaped `(B, T, F)`.

        Raises
        ------
        ValueError
            If any entry is outside `{MASK_MISSING, MASK_OBSERVED}`.
        """
        is_observed = jnp.abs(mask - MASK_OBSERVED) < _MASK_COMPARE_TOL
        is_missing = jnp.abs(mask - MASK_MISSING) < _MASK_COMPARE_TOL
        if not bool(jnp.all(is_observed | is_missing)):
            raise ValueError(
                "mask entries must be 0.0 (missing) or 1.0 (observed)"
            )

    def _validate_missing_sentinel(
            self,
            values: jnp.ndarray,
            mask: jnp.ndarray,
    ) -> None:
        """Check that missing cells in `values` equal the locked sentinel.

        Parameters
        ----------
        values : jax.Array
            Value tensor shaped `(B, T, F)`.
        mask : jax.Array
            Mask tensor shaped `(B, T, F)`; `0` marks missing cells.

        Raises
        ------
        ValueError
            If any missing cell differs from `MISSING_VALUE_SENTINEL`.
        """
        is_missing = jnp.abs(mask - MASK_MISSING) < _MASK_COMPARE_TOL
        missing_vals = values[is_missing]
        if missing_vals.size == 0:
            return
        ok = jnp.all(
            jnp.abs(missing_vals - MISSING_VALUE_SENTINEL) < _MASK_COMPARE_TOL
        )
        if not bool(ok):
            raise ValueError(
                "missing cells in values must equal MISSING_VALUE_SENTINEL "
                f"({MISSING_VALUE_SENTINEL})"
            )


def apply_missing_sentinel(
        values: jnp.ndarray,
        mask: jnp.ndarray,
) -> jnp.ndarray:
    """Set missing positions to the locked sentinel value.

    Parameters
    ----------
    values : jax.Array
        Clean or partially filled values shaped `(B, T, F)`.
    mask : jax.Array
        Observation mask shaped `(B, T, F)`.

    Returns
    -------
    jax.Array
        Values with `m == 0` positions set to `MISSING_VALUE_SENTINEL`.
    """
    observed = jnp.abs(mask - MASK_OBSERVED) < _MASK_COMPARE_TOL
    return jnp.where(observed, values, MISSING_VALUE_SENTINEL)


def build_stub_batch(data_config: DataConfig) -> TimeSeriesBatch:
    """Build a deterministic stub batch for WP0 (not sine/AR generators).

    A contiguous time block of length `block_len` at the end of each window
    is marked missing. Remaining cells are observed. This only exercises the
    schema contract until WP1 lands real generators.

    Parameters
    ----------
    data_config : DataConfig
        Provides `batch_size`, `seq_len`, `n_features`, and `block_len`.

    Returns
    -------
    TimeSeriesBatch
        Batch with documented `(B, T, F)` shapes and mask convention.
    """
    batch_size = int(data_config.batch_size)
    seq_len = int(data_config.seq_len)
    n_features = int(data_config.n_features)
    block_len = int(data_config.block_len)
    _validate_stub_dims(seq_len, block_len, n_features)

    time_idx = jnp.arange(seq_len, dtype=jnp.float32)
    feature_idx = jnp.arange(n_features, dtype=jnp.float32)
    scale = 0.1
    pattern = time_idx[None, :, None] * scale + feature_idx[None, None, :]
    x_clean = jnp.broadcast_to(pattern, (batch_size, seq_len, n_features))

    mask = jnp.ones((batch_size, seq_len, n_features), dtype=jnp.float32)
    block_start = seq_len - block_len
    mask = mask.at[:, block_start:, :].set(MASK_MISSING)
    values = apply_missing_sentinel(x_clean, mask)
    return TimeSeriesBatch(x=values, m=mask)


def _validate_stub_dims(
        seq_len: int,
        block_len: int,
        n_features: int,
) -> None:
    """Validate stub-batch dimensions against M1 constraints.

    Parameters
    ----------
    seq_len : int
        Window length `T`; must be positive.
    block_len : int
        Contiguous missing-block length; must lie in `[1, seq_len]`.
    n_features : int
        Feature count `F`; must be in `{1, 2, 3}` for M1.

    Raises
    ------
    ValueError
        If any dimension violates the M1 stub-batch contract.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if block_len <= 0 or block_len > seq_len:
        raise ValueError(
            f"block_len must be in [1, seq_len], got {block_len}"
        )
    if n_features not in (1, 2, 3):
        raise ValueError(
            f"n_features must be in {{1, 2, 3}} for M1, got {n_features}"
        )
