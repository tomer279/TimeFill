"""Conditional DSM for M2 real windows with native mask composition.

Classes
-------
RealDSMBatch
    Clean values plus native masks for one mini-batch.
RealConditionalDSMLoss
    VP-SDE DSM with ``MaskComposer`` and artificial-target weighting.

Functions
---------
dsm_settings_from_real_config
    Map ``RealM2Config`` fields into ``DSMLossSettings``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from src.data.real.masks import MaskComposer, artificial_target_mask
from src.data.real_config import RealM2Config
from src.data.schema import BATCH_AXIS, MASK_OBSERVED
from src.models.score_net import ConditionalScoreNet
from src.sde.vp_sde import VPSDE
from src.train.loss import (
    DSMLossSettings,
    _DsmKeyPack,
    _weighted_mean_square,
)

_MASK_MODE_RANDOM = "random"
_MASK_MODE_BLOCK = "block"
_MIXED_BLOCK_THRESHOLD = 0.5
_MASK_COMPARE_TOL = 0.5


class RealDSMBatch(eqx.Module):
    """One mini-batch of real windows for M2 DSM.

    Public methods
    --------------
    validate
        Check rank-3 shape agreement between ``x0`` and ``m_native``.
    batch_size
        Return ``B`` from axis 0.

    Instance variables
    ------------------
    x0 : jax.Array
        Ground-truth windows ``(B, T, F)`` (sentinel on native gaps).
    m_native : jax.Array
        Native masks ``(B, T, F)``.
    """

    x0: jnp.ndarray
    m_native: jnp.ndarray

    def validate(self) -> None:
        """Raise ``ValueError`` if batch tensors are inconsistent."""
        if self.x0.ndim != 3 or self.m_native.ndim != 3:
            raise ValueError(
                "x0 and m_native must have rank 3 (B, T, F); "
                f"got {self.x0.ndim}, {self.m_native.ndim}"
            )
        if self.x0.shape != self.m_native.shape:
            raise ValueError(
                "x0 and m_native shapes must match; "
                f"got {self.x0.shape} vs {self.m_native.shape}"
            )

    def batch_size(self) -> int:
        """Return the batch size ``B``.

        Returns
        -------
        int
            ``x0.shape[0]``.
        """
        return int(self.x0.shape[BATCH_AXIS])


class RealConditionalDSMLoss:
    """M2 conditional DSM with native + artificial mask composition.

    Training view per batch:

    1. Start from (normalized) ``x0`` with ``m_native``.
    2. Sample ``m_artificial`` via ``MaskComposer`` (random / block / mixed).
    3. Set ``m_train = m_native * m_artificial``.
    4. Run VP-SDE DSM; reduce on **artificial targets** with known GT
       (``artificial_target_mask``), not on natively missing cells.

    Public methods
    --------------
    __call__
        Scalar DSM loss for one ``RealDSMBatch``.
    sample_mask
        Draw ``(x_obs, m_train, m_artificial)`` under the mask mode.
    reduction_weights
        Per-cell weights from native / artificial masks.
    with_mask_rate
        Return a copy with a different artificial MCAR rate.

    Instance variables
    ------------------
    sde : VPSDE
        Forward VP-SDE.
    settings : DSMLossSettings
        Mask / time / reduction hyperparameters.
    composer : MaskComposer
        Native-aware artificial mask composer.
    """

    def __init__(self, sde: VPSDE, settings: DSMLossSettings) -> None:
        """Store SDE, settings, and a matching ``MaskComposer``.

        Parameters
        ----------
        sde : VPSDE
            VP-SDE used for marginal mean/std and forward samples.
        settings : DSMLossSettings
            Mask rate/mode, ``t_eps``, and observed-cell weight.
        """
        settings.validate()
        if settings.t_eps >= sde.t_diff:
            raise ValueError(
                "t_eps must be < T_diff, "
                f"got t_eps={settings.t_eps}, T_diff={sde.t_diff}"
            )
        self.sde = sde
        self.settings = settings
        self.composer = MaskComposer(settings.mask_rate, settings.block_len)

    def __call__(
            self,
            model: ConditionalScoreNet,
            batch: RealDSMBatch,
            key: jax.Array,
    ) -> jnp.ndarray:
        """Compute conditional DSM loss on a real batch.

        Parameters
        ----------
        model : ConditionalScoreNet
            Score network ``s_theta``.
        batch : RealDSMBatch
            Normalized windows and native masks.
        key : jax.Array
            PRNG key (mask, time, noise).

        Returns
        -------
        jax.Array
            Scalar mean weighted squared score error.
        """
        batch.validate()
        keys = _DsmKeyPack.split_from(key)
        x_obs, m_train, m_art = self.sample_mask(keys.mask, batch)
        time = self._sample_time(keys.time, batch.batch_size())
        eps = jax.random.normal(keys.eps, batch.x0.shape, dtype=jnp.float32)
        y_t = self.sde.forward_sample(batch.x0, time, eps)
        pred = model(y_t, time, m_train, x_obs)
        target = self._score_target(eps, batch.x0, time)
        weights = self.reduction_weights(batch.m_native, m_art, m_train)
        return _weighted_mean_square(pred, target, weights)

    def sample_mask(
            self,
            key: jax.Array,
            batch: RealDSMBatch,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample composed training masks for a real batch.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        batch : RealDSMBatch
            Values and native masks.

        Returns
        -------
        x_obs : jax.Array
            Sentinel-filled observations under ``m_train``.
        m_train : jax.Array
            ``m_native * m_artificial``.
        m_artificial : jax.Array
            Restricted artificial mask.
        """
        key_rand, key_block, key_pick = jax.random.split(key, 3)
        mode = self.settings.mask_mode
        if mode == _MASK_MODE_RANDOM:
            return self.composer.compose_random(
                key_rand, batch.x0, batch.m_native,
            )
        if mode == _MASK_MODE_BLOCK:
            return self.composer.compose_block(
                key_block, batch.x0, batch.m_native,
            )
        use_block = jax.random.uniform(key_pick) < _MIXED_BLOCK_THRESHOLD
        return _blend_compose_results(
            self.composer.compose_random(key_rand, batch.x0, batch.m_native),
            self.composer.compose_block(key_block, batch.x0, batch.m_native),
            use_block,
        )


    def reduction_weights(
            self,
            m_native: jnp.ndarray,
            m_artificial: jnp.ndarray,
            m_train: jnp.ndarray,
    ) -> jnp.ndarray:
        """Build per-cell DSM weights for M2.

        Artificial targets with known ground truth get weight ``1``.
        Observed conditioning cells get ``obs_loss_weight`` (default ``0``).
        Natively missing cells get weight ``0``.

        Parameters
        ----------
        m_native : jax.Array
            Native masks.
        m_artificial : jax.Array
            Restricted artificial masks.
        m_train : jax.Array
            Composed training masks.

        Returns
        -------
        jax.Array
            Non-negative weights, same shape as ``m_train``.
        """
        targets = artificial_target_mask(m_native, m_artificial)
        observed = jnp.abs(m_train - MASK_OBSERVED) < _MASK_COMPARE_TOL
        return (
            targets.astype(jnp.float32)
            + observed.astype(jnp.float32) * self.settings.obs_loss_weight
        )

    def with_mask_rate(self, mask_rate: float) -> "RealConditionalDSMLoss":
        """Return a loss instance with a different artificial MCAR rate.

        Parameters
        ----------
        mask_rate : float
            New MCAR rate in ``[0, 1]``.

        Returns
        -------
        RealConditionalDSMLoss
            Fresh instance sharing the same SDE object.
        """
        settings = DSMLossSettings(
            mask_rate=float(mask_rate),
            block_len=self.settings.block_len,
            t_eps=self.settings.t_eps,
            obs_loss_weight=self.settings.obs_loss_weight,
            mask_mode=self.settings.mask_mode,
            std_floor=self.settings.std_floor,
        )
        return RealConditionalDSMLoss(self.sde, settings)

    def _score_target(
            self,
            eps: jnp.ndarray,
            x0: jnp.ndarray,
            time: jnp.ndarray,
    ) -> jnp.ndarray:
        """Gaussian transition score target ``-eps / std(t)``.

        Parameters
        ----------
        eps : jax.Array
            Standard normal noise used in ``forward_sample``.
        x0 : jax.Array
            Clean state (broadcast shape for ``std``).
        time : jax.Array
            Diffusion times shaped ``(B,)``.

        Returns
        -------
        jax.Array
            Target score, same shape as ``eps``.
        """
        _, std = self.sde.marginal_prob(x0, time)
        std_safe = jnp.maximum(std, self.settings.std_floor)
        return -eps / std_safe

    def _sample_time(
            self,
            key: jax.Array,
            batch_size: int,
    ) -> jnp.ndarray:
        """Sample per-example diffusion times in ``[t_eps, T_diff]``.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        batch_size : int
            Batch size ``B``.

        Returns
        -------
        jax.Array
            Times shaped ``(B,)``.
        """
        u_rand = jax.random.uniform(key, shape=(batch_size,))
        span = self.sde.t_diff - self.settings.t_eps
        return self.settings.t_eps + span * u_rand


def dsm_settings_from_real_config(config: RealM2Config) -> DSMLossSettings:
    """Build DSM settings from a validated real M2 config.

    Parameters
    ----------
    config : RealM2Config
        Nested config with ``data`` and ``train`` sections.

    Returns
    -------
    DSMLossSettings
        Validated DSM hyperparameters.
    """
    settings = DSMLossSettings(
        mask_rate=config.data.mask_rate,
        block_len=config.data.block_len,
        t_eps=config.train.protocol.t_eps,
        obs_loss_weight=config.train.protocol.obs_loss_weight,
        mask_mode=config.train.protocol.mask_mode,
    )
    settings.validate()
    return settings

def _blend_compose_results(
        rand_out: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        block_out: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        use_block: jnp.ndarray | bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Elementwise select block vs random compose triples.

    Parameters
    ----------
    rand_out : tuple of jax.Array
        ``(x_obs, m_train, m_artificial)`` from ``compose_random``.
    block_out : tuple of jax.Array
        ``(x_obs, m_train, m_artificial)`` from ``compose_block``.
    use_block : jax.Array or bool
        Selector; true chooses the block compose result.

    Returns
    -------
    x_obs : jax.Array
        Selected observations.
    m_train : jax.Array
        Selected training mask.
    m_artificial : jax.Array
        Selected artificial mask.
    """
    x_rand, m_train_rand, m_art_rand = rand_out
    x_block, m_train_block, m_art_block = block_out
    return (
        jnp.where(use_block, x_block, x_rand),
        jnp.where(use_block, m_train_block, m_train_rand),
        jnp.where(use_block, m_art_block, m_art_rand),
    )
