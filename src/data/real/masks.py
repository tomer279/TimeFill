"""Native + artificial mask composition for M2 training.

Classes
-------
MaskComposer
    Compose M1 artificial masks with native masks (`m_native * m_art`).

Functions
---------
compose_train_mask
    Elementwise `m_train = m_native * m_artificial`.
restrict_artificial_mask
    Force `m_artificial = 0` on natively missing cells.
artificial_target_mask
    Cells with ground truth held out by the artificial mask.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from src.data.masking import block_mask, random_mask
from src.data.schema import (
    MASK_MISSING,
    MASK_OBSERVED,
    apply_missing_sentinel,
)

_MASK_TOL = 1e-6


class MaskComposer:
    """Compose M1 artificial masks with native observation masks.

    Artificial MCAR / block masks are sampled with M1 helpers, then
    restricted so natively missing cells are never marked observed.
    Training mask: ``m_train = m_native * m_artificial``.

    Public methods
    --------------
    compose_random
        MCAR artificial mask composed with ``m_native``.
    compose_block
        Contiguous-block artificial mask composed with ``m_native``.

    Instance variables
    ------------------
    rate : float
        MCAR drop probability on eligible cells.
    block_len : int
        Contiguous block length for block mode.
    """

    def __init__(self, rate: float, block_len: int) -> None:
        """Store artificial-mask hyperparameters.

        Parameters
        ----------
        rate : float
            MCAR rate in ``[0, 1]``.
        block_len : int
            Positive block length (``<= T`` at apply time).
        """
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {rate}")
        if block_len <= 0:
            raise ValueError(f"block_len must be positive, got {block_len}")
        self.rate = float(rate)
        self.block_len = int(block_len)

    def compose_random(
            self,
            key: jax.Array,
            values: jnp.ndarray,
            m_native: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Apply MCAR masking on natively observed cells only.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        values : jax.Array
            Ground-truth windows ``(B, T, F)``.
        m_native : jax.Array
            Native masks ``(B, T, F)``.

        Returns
        -------
        x_obs : jax.Array
            Values with ``m_train`` missing cells set to the sentinel.
        m_train : jax.Array
            ``m_native * m_artificial``.
        m_artificial : jax.Array
            Restricted artificial mask (never observes native gaps).
        """
        _x_ignored, m_art_raw = random_mask(values, self.rate, key)
        return _finalize_composition(values, m_native, m_art_raw)

    def compose_block(
            self,
            key: jax.Array,
            values: jnp.ndarray,
            m_native: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Apply block masking on natively observed cells only.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        values : jax.Array
            Ground-truth windows ``(B, T, F)``.
        m_native : jax.Array
            Native masks ``(B, T, F)``.

        Returns
        -------
        x_obs : jax.Array
            Values with ``m_train`` missing cells set to the sentinel.
        m_train : jax.Array
            ``m_native * m_artificial``.
        m_artificial : jax.Array
            Restricted artificial mask (never observes native gaps).
        """
        _x_ignored, m_art_raw = block_mask(values, self.block_len, key)
        return _finalize_composition(values, m_native, m_art_raw)


def compose_train_mask(
        m_native: jnp.ndarray,
        m_artificial: jnp.ndarray,
) -> jnp.ndarray:
    """Return ``m_train = m_native * m_artificial`` (elementwise).

    Parameters
    ----------
    m_native : jax.Array
        Native observation mask.
    m_artificial : jax.Array
        Artificial mask (already restricted if required).

    Returns
    -------
    jax.Array
        Training / conditioning mask.
    """
    return m_native * m_artificial


def restrict_artificial_mask(
        m_native: jnp.ndarray,
        m_artificial: jnp.ndarray,
) -> jnp.ndarray:
    """Clear artificial 'observed' flags on natively missing cells.

    Parameters
    ----------
    m_native : jax.Array
        Native mask (``1`` observed).
    m_artificial : jax.Array
        Raw artificial mask from M1 APIs.

    Returns
    -------
    jax.Array
        Artificial mask with native gaps forced to ``0``.
    """
    native_observed = jnp.abs(m_native - MASK_OBSERVED) < _MASK_TOL
    return jnp.where(native_observed, m_artificial, MASK_MISSING)


def artificial_target_mask(
        m_native: jnp.ndarray,
        m_artificial: jnp.ndarray,
) -> jnp.ndarray:
    """Mask of loss / metric targets with known ground truth.

    A target cell is natively observed and artificially missing.

    Parameters
    ----------
    m_native : jax.Array
        Native mask.
    m_artificial : jax.Array
        Restricted artificial mask.

    Returns
    -------
    jax.Array
        Float mask with ``1`` on target cells and ``0`` elsewhere.
    """
    native_observed = jnp.abs(m_native - MASK_OBSERVED) < _MASK_TOL
    art_missing = jnp.abs(m_artificial - MASK_MISSING) < _MASK_TOL
    return jnp.where(
        native_observed & art_missing,
        MASK_OBSERVED,
        MASK_MISSING,
    )


def _finalize_composition(
        values: jnp.ndarray,
        m_native: jnp.ndarray,
        m_art_raw: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Restrict artificial mask, compose train mask, apply sentinel.

    Parameters
    ----------
    values : jax.Array
        Ground-truth values ``(B, T, F)``.
    m_native : jax.Array
        Native masks.
    m_art_raw : jax.Array
        Unrestricted artificial masks from M1 helpers.

    Returns
    -------
    x_obs : jax.Array
        Sentinel-filled observations under ``m_train``.
    m_train : jax.Array
        Composed training mask.
    m_artificial : jax.Array
        Restricted artificial mask.
    """
    m_artificial = restrict_artificial_mask(m_native, m_art_raw)
    m_train = compose_train_mask(m_native, m_artificial)
    x_obs = apply_missing_sentinel(values, m_train)
    return x_obs, m_train, m_artificial
