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
    """Contiguous time-block masker (one gap per batch item and feature).

    For each `(b, f)`, a start index is drawn uniformly in
    ``[0, T - block_len]`` and times
    ``[start, start + block_len)`` are marked missing.

    Public methods
    --------------
    apply
        Sample block masks and return `(x_obs, m)`.
    missing_length
        Return the configured contiguous block length.

    Instance variables
    ------------------
    block_len : int
        Contiguous missing span along the time axis.
    """

    def __init__(self, block_len: int) -> None:
        """Store the contiguous missing-block length.

        Parameters
        ----------
        block_len : int
            Positive block length; must be ``<= T`` at apply time.
        """
        if block_len <= 0:
            raise ValueError(f"block_len must be positive, got {block_len}")
        self.block_len = int(block_len)

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
        if self.block_len > seq_len:
            raise ValueError(
                "block_len must be <= seq_len, "
                f"got block_len={self.block_len}, seq_len={seq_len}"
            )

        max_start_exclusive = seq_len - self.block_len + 1
        starts = jax.random.randint(
            key,
            shape=(batch_size, n_features),
            minval=0,
            maxval=max_start_exclusive,
        )
        time = jnp.arange(seq_len, dtype=jnp.int32)
        starts_exp = starts[:, None, :]
        time_exp = time[None, :, None]
        is_missing = (time_exp >= starts_exp) & (
            time_exp < starts_exp + self.block_len
        )
        mask = jnp.where(is_missing, MASK_MISSING, MASK_OBSERVED).astype(
            jnp.float32
        )
        x_obs = apply_missing_sentinel(values, mask)
        return x_obs, mask

    def missing_length(self) -> int:
        """Return the configured contiguous block length.

        Returns
        -------
        int
            Same as `block_len`.
        """
        return self.block_len


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
