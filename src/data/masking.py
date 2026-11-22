"""Observation masks for M1 synthetic windows.

Classes
-------
RandomMasker
    Independent MCAR Bernoulli masks at a fixed missing rate.
BlockMasker
    Contiguous time-block masks of fixed length per feature.

Functions
---------
random_mask
    Apply an MCAR mask; return `(x_obs, m)`.
block_mask
    Apply a contiguous block mask; return `(x_obs, m)`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from src.data.schema import (
    BATCH_AXIS,
    FEATURE_AXIS,
    MASK_MISSING,
    MASK_OBSERVED,
    TIME_AXIS,
    apply_missing_sentinel,
)


class RandomMasker:
    """MCAR masker: each cell is missing independently with probability `rate`.

    Public methods
    --------------
    apply
        Sample a mask and return `(x_obs, m)`.
    expected_missing_fraction
        Return the configured missing probability.

    Instance variables
    ------------------
    rate : float
        Probability that a cell is missing (`m = 0`).
    """

    def __init__(self, rate: float) -> None:
        """Store the MCAR missing rate.

        Parameters
        ----------
        rate : float
            Missing probability in `[0, 1]`.
        """
        _validate_rate(rate)
        self.rate = float(rate)

    def apply(
            self,
            key: jax.Array,
            values: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply a random mask to a clean `(B, T, F)` tensor.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        values : jax.Array
            Clean values shaped `(B, T, F)`.

        Returns
        -------
        x_obs : jax.Array
            Values with missing cells set to the schema sentinel `0.0`.
        m : jax.Array
            Mask with `1` observed and `0` missing.
        """
        _validate_values_rank(values)
        draw = jax.random.uniform(key, shape=values.shape, dtype=jnp.float32)
        is_missing = draw < self.rate
        mask = jnp.where(is_missing, MASK_MISSING, MASK_OBSERVED)
        x_obs = apply_missing_sentinel(values, mask)
        return x_obs, mask

    def expected_missing_fraction(self) -> float:
        """Return the configured MCAR missing probability.

        Returns
        -------
        float
            Same as `rate`.
        """
        return self.rate


class BlockMasker:
    """Contiguous time-block masker (fixed or ranged length).

    Legacy: one gap per `(b, f)` of length ``block_len``.
    With ``block_len_range`` / ``max_blocks_per_feature > 1``: sample
    up to that many gaps per feature and union them (overlaps allowed).

    Public methods
    --------------
    apply
        Sample block masks and return `(x_obs, m)`.
    missing_length
        Return the configured contiguous block length (legacy fixed).

    Instance variables
    ------------------
    block_len : int
        Fixed length when ``block_len_range`` is ``None``.
    block_len_range : tuple of int or None
        Inclusive ``(lo, hi)`` length bounds; ``None`` = fixed ``block_len``.
    max_blocks_per_feature : int
        Number of gaps drawn per feature (union).
    """

    def __init__(
            self,
            block_len: int,
            block_len_range: tuple[int, int] | None = None,
            max_blocks_per_feature: int = 1,
    ) -> None:
        """Store block length and optional range / multi-block knobs.

        Parameters
        ----------
        block_len : int
            Positive fixed length when no range is set.
        block_len_range : tuple of int or None, optional
            Inclusive ``(lo, hi)`` with ``0 < lo <= hi``.
        max_blocks_per_feature : int, optional
            Positive gap count per feature (default ``1``).
        """
        if block_len <= 0:
            raise ValueError(f"block_len must be positive, got {block_len}")
        if max_blocks_per_feature <= 0:
            raise ValueError(
                "max_blocks_per_feature must be positive, "
                f"got {max_blocks_per_feature}"
            )
        if block_len_range is not None:
            lo_bound, hi_bound = block_len_range
            if lo_bound <= 0 or hi_bound < lo_bound:
                raise ValueError(
                    "block_len_range must satisfy 0 < lo <= hi, "
                    f"got {block_len_range!r}"
                )
        self.block_len = int(block_len)
        self.block_len_range = (
            None
            if block_len_range is None
            else (int(block_len_range[0]), int(block_len_range[1]))
        )
        self.max_blocks_per_feature = int(max_blocks_per_feature)

    def apply(
            self,
            key: jax.Array,
            values: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply a contiguous block mask to a clean `(B, T, F)` tensor.

        Parameters
        ----------
        key : jax.Array
            PRNG key for per-feature block starts.
        values : jax.Array
            Clean values shaped `(B, T, F)`.

        Returns
        -------
        x_obs : jax.Array
            Values with missing cells set to the schema sentinel `0.0`.
        m : jax.Array
            Mask with `1` observed and `0` missing.
        """
        _validate_values_rank(values)
        batch_size = int(values.shape[BATCH_AXIS])
        seq_len = int(values.shape[TIME_AXIS])
        n_features = int(values.shape[FEATURE_AXIS])
        dims = (batch_size, seq_len, n_features)
        if self.block_len_range is None and self.max_blocks_per_feature == 1:
            return _apply_fixed_block(key, values, self.block_len, dims)
        return _apply_ranged_blocks(
            key,
            values,
            self._resolved_range(seq_len),
            self.max_blocks_per_feature,
            dims,
        )

    def missing_length(self) -> int:
        """Return the configured contiguous block length.

        Returns
        -------
        int
            Fixed ``block_len``, or the range lower bound when ranged.
        """
        if self.block_len_range is None:
            return self.block_len
        return self.block_len_range[0]

    def _resolved_range(self, seq_len: int) -> tuple[int, int]:
        """Clip configured length bounds to ``seq_len``."""
        if self.block_len_range is None:
            lo_bound = hi_bound = self.block_len
        else:
            lo_bound, hi_bound = self.block_len_range
        hi_eff = min(hi_bound, seq_len)
        lo_eff = min(lo_bound, hi_eff)
        if lo_eff <= 0:
            raise ValueError(
                "effective block_len_range is empty for "
                f"seq_len={seq_len}, range={self.block_len_range!r}"
            )
        if self.max_blocks_per_feature * lo_eff > seq_len:
            raise ValueError(
                "max_blocks_per_feature * lo exceeds seq_len: "
                f"{self.max_blocks_per_feature} * {lo_eff} > {seq_len}"
            )
        return lo_eff, hi_eff


def _apply_fixed_block(
        key: jax.Array,
        values: jnp.ndarray,
        block_len: int,
        dims: tuple[int, int, int],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Legacy single fixed-length block per ``(b, f)``."""
    batch_size, seq_len, n_features = dims
    if block_len > seq_len:
        raise ValueError(
            "block_len must be <= seq_len, "
            f"got block_len={block_len}, seq_len={seq_len}"
        )
    max_start_exclusive = seq_len - block_len + 1
    starts = jax.random.randint(
        key,
        shape=(batch_size, n_features),
        minval=0,
        maxval=max_start_exclusive,
    )
    is_missing = _span_mask(starts, block_len, seq_len)
    mask = jnp.where(is_missing, MASK_MISSING, MASK_OBSERVED).astype(
        jnp.float32
    )
    return apply_missing_sentinel(values, mask), mask


def _apply_ranged_blocks(
        key: jax.Array,
        values: jnp.ndarray,
        length_range: tuple[int, int],
        max_blocks: int,
        dims: tuple[int, int, int],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Union of up to ``max_blocks`` variable-length gaps per feature."""
    seq_len = dims[TIME_AXIS]
    bf_shape = (dims[BATCH_AXIS], dims[FEATURE_AXIS])
    lo_bound, hi_bound = length_range
    is_missing = jnp.zeros(dims, dtype=bool)
    # shortcut: python over static max_blocks; upgrade to fori_loop if needed
    for _ in range(max_blocks):
        key, key_len, key_start = jax.random.split(key, 3)
        lengths = jax.random.randint(
            key_len,
            shape=bf_shape,
            minval=lo_bound,
            maxval=hi_bound + 1,
        )
        # start = floor(U * (T - L + 1)) keeps maxval data-dependent / JIT-safe
        starts = jnp.floor(
            jax.random.uniform(key_start, shape=bf_shape)
            * (seq_len - lengths + 1).astype(jnp.float32)
        ).astype(jnp.int32)
        is_missing = is_missing | _span_mask(starts, lengths, seq_len)
    mask = jnp.where(is_missing, MASK_MISSING, MASK_OBSERVED).astype(
        jnp.float32
    )
    return apply_missing_sentinel(values, mask), mask


def _span_mask(
        starts: jnp.ndarray,
        lengths: jnp.ndarray | int,
        seq_len: int,
) -> jnp.ndarray:
    """Boolean ``(B, T, F)`` mask for ``[start, start + length)`` spans."""
    time = jnp.arange(seq_len, dtype=jnp.int32)
    starts_exp = starts[:, None, :]
    if isinstance(lengths, int):
        lengths_exp = lengths
    else:
        lengths_exp = lengths[:, None, :]
    time_exp = time[None, :, None]
    return (time_exp >= starts_exp) & (time_exp < starts_exp + lengths_exp)


def random_mask(
        values: jnp.ndarray,
        rate: float,
        key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply an MCAR random mask.

    Observed entries are unchanged; missing entries become `0.0`.

    Parameters
    ----------
    values : jax.Array
        Clean values shaped `(B, T, F)`.
    rate : float
        Probability a cell is missing.
    key : jax.Array
        PRNG key.

    Returns
    -------
    x_obs : jax.Array
        Partially observed values (sentinel-filled).
    m : jax.Array
        Observation mask (`1` observed, `0` missing).
    """
    return RandomMasker(rate).apply(key, values)


def block_mask(
        values: jnp.ndarray,
        block_len: int,
        key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply a contiguous time-block mask per feature.

    Observed entries are unchanged; missing entries become `0.0`.

    Parameters
    ----------
    values : jax.Array
        Clean values shaped `(B, T, F)`.
    block_len : int
        Contiguous missing length along time.
    key : jax.Array
        PRNG key.

    Returns
    -------
    x_obs : jax.Array
        Partially observed values (sentinel-filled).
    m : jax.Array
        Observation mask (`1` observed, `0` missing).
    """
    return BlockMasker(block_len).apply(key, values)


def _validate_rate(rate: float) -> None:
    """Ensure a missing rate lies in ``[0, 1]``.

    Parameters
    ----------
    rate : float
        Candidate MCAR rate.

    Raises
    ------
    ValueError
        If `rate` is outside `[0, 1]`.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")


def _validate_values_rank(values: jnp.ndarray) -> None:
    """Ensure values are rank-3 `(B, T, F)`.

    Parameters
    ----------
    values : jax.Array
        Candidate value tensor.

    Raises
    ------
    ValueError
        If `values.ndim != 3`.
    """
    if values.ndim != 3:
        raise ValueError(
            "values must have shape (B, T, F); "
            f"got ndim={values.ndim}"
        )
