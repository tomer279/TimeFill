"""Conditional continuous denoising score matching for M1 (WP4).

Classes
-------
DSMLossSettings
    Mask / time / reduction hyperparameters for DSM.
ConditionalDSMLoss
    Self-supervised VP-SDE DSM with loss on masked target cells.

Functions
---------
dsm_settings_from_config
    Map ``ToyM1Config`` data/train fields into ``DSMLossSettings``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from src.data.config import ToyM1Config
from src.data.masking import block_mask, random_mask
from src.data.schema import BATCH_AXIS, MASK_MISSING, MASK_OBSERVED
from src.models.score_net import ConditionalScoreNet
from src.sde.vp_sde import VPSDE

_STD_FLOOR = 1e-5
_WEIGHT_FLOOR = 1.0
_MASK_MODE_RANDOM = "random"
_MASK_MODE_BLOCK = "block"
_MASK_MODE_MIXED = "mixed"
_MIXED_BLOCK_THRESHOLD = 0.5
_MASK_COMPARE_TOL = 0.5


@dataclass(frozen=True)
class DSMLossSettings:
    """Settings for conditional continuous DSM.

    Public methods
    --------------
    validate
        Check positivity and allowed mask modes.
    to_dict
        Serialize fields to a plain dict.

    Instance variables
    ------------------
    mask_rate : float
        MCAR missing rate for random masks.
    block_len : int
        Contiguous block length for block masks.
    t_eps : float
        Lower bound of Uniform diffusion-time sampling.
    obs_loss_weight : float
        Weight on observed cells; M1 default is ``0.0`` (zero aux loss).
    mask_mode : str
        ``random``, ``block``, or ``mixed``.
    std_floor : float
        Floor for ``std(t)`` when forming the score target.
    """

    mask_rate: float
    block_len: int
    t_eps: float
    obs_loss_weight: float
    mask_mode: str
    std_floor: float = _STD_FLOOR

    def validate(self) -> None:
        """Raise ``ValueError`` if DSM settings are invalid."""
        if not 0.0 <= self.mask_rate <= 1.0:
            raise ValueError(
                f"mask_rate must be in [0, 1], got {self.mask_rate}"
            )
        if self.block_len <= 0:
            raise ValueError(
                f"block_len must be positive, got {self.block_len}"
            )
        if self.t_eps <= 0.0:
            raise ValueError(f"t_eps must be positive, got {self.t_eps}")
        if self.obs_loss_weight < 0.0:
            raise ValueError(
                "obs_loss_weight must be >= 0, "
                f"got {self.obs_loss_weight}"
            )
        if self.mask_mode not in (
            _MASK_MODE_RANDOM,
            _MASK_MODE_BLOCK,
            _MASK_MODE_MIXED,
        ):
            raise ValueError(f"unsupported mask_mode={self.mask_mode!r}")
        if self.std_floor <= 0.0:
            raise ValueError(
                f"std_floor must be positive, got {self.std_floor}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable dict of fields.

        Returns
        -------
        dict of str to Any
            Field name to value.
        """
        return {
            "mask_rate": self.mask_rate,
            "block_len": self.block_len,
            "t_eps": self.t_eps,
            "obs_loss_weight": self.obs_loss_weight,
            "mask_mode": self.mask_mode,
            "std_floor": self.std_floor,
        }


class ConditionalDSMLoss:
    """CSDI-adapted continuous DSM under a VP-SDE forward process.

    Training view per batch:

    1. Start from clean ``x0``.
    2. Sample self-supervised mask ``m`` (random / block / mixed).
    3. Sample ``t ~ Uniform(t_eps, T_diff)`` per example.
    4. Perturb ``y_t = mean(x0, t) + std(t) * eps``.
    5. Predict ``s_theta(y_t, t, m, x_obs)``.
    6. Target score ``-eps / std(t)`` (floored).
    7. Reduce squared error with weights: target cells ``(1 - m)`` get
       weight ``1``; observed cells get ``obs_loss_weight`` (M1: ``0``).

    Public methods
    --------------
    __call__
        Scalar DSM loss for one clean batch.
    sample_mask
        Draw ``(x_obs, m)`` under the configured mask mode.
    reduction_weights
        Per-cell loss weights from ``m`` and ``obs_loss_weight``.
    score_target
        Closed-form Gaussian score target from ``eps`` and ``std(t)``.

    Instance variables
    ------------------
    sde : VPSDE
        Forward VP-SDE providing marginal mean/std.
    settings : DSMLossSettings
        Mask, time, and reduction hyperparameters.
    """

    def __init__(self, sde: VPSDE, settings: DSMLossSettings) -> None:
        """Store SDE and validated DSM settings.

        Parameters
        ----------
        sde : VPSDE
            VP-SDE used for ``marginal_prob`` / ``forward_sample``.
        settings : DSMLossSettings
            Mask rate/mode, ``t_eps``, and observed-cell weight.
        """
        settings.validate()
        if settings.t_eps >= sde.T_diff:
            raise ValueError(
                "t_eps must be < T_diff, "
                f"got t_eps={settings.t_eps}, T_diff={sde.T_diff}"
            )
        self.sde = sde
        self.settings = settings

    def __call__(
            self,
            model: ConditionalScoreNet,
            x0: jnp.ndarray,
            key: jax.Array,
    ) -> jnp.ndarray:
        """Compute conditional DSM loss on a clean ``(B, T, F)`` batch.

        Parameters
        ----------
        model : ConditionalScoreNet
            Score network ``s_theta``.
        x0 : jax.Array
            Clean windows shaped ``(B, T, F)``.
        key : jax.Array
            PRNG key (mask, time, and Gaussian noise).

        Returns
        -------
        jax.Array
            Scalar mean weighted squared score error.
        """
        key_mask, key_time, key_eps = jax.random.split(key, 3)
        x_obs, mask = self.sample_mask(key_mask, x0)
        batch_size = int(x0.shape[BATCH_AXIS])
        time = self._sample_time(key_time, batch_size)
        eps = jax.random.normal(key_eps, x0.shape, dtype=jnp.float32)
        y_t = self.sde.forward_sample(x0, time, eps)
        pred = model(y_t, time, mask, x_obs)
        target = self.score_target(eps, x0, time)
        weights = self.reduction_weights(mask)
        return _weighted_mean_square(pred, target, weights)

    def sample_mask(
            self,
            key: jax.Array,
            x0: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample a training mask and sentinel-filled observations.

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        x0 : jax.Array
            Clean values shaped ``(B, T, F)``.

        Returns
        -------
        x_obs : jax.Array
            Values with missing cells set to sentinel ``0.0``.
        mask : jax.Array
            ``1`` observed, ``0`` missing (imputation targets).
        """
        key_rand, key_block, key_pick = jax.random.split(key, 3)
        x_rand, m_rand = random_mask(
            x0,
            self.settings.mask_rate,
            key_rand,
        )
        x_block, m_block = block_mask(
            x0,
            self.settings.block_len,
            key_block,
        )
        mode = self.settings.mask_mode
        if mode == _MASK_MODE_RANDOM:
            return x_rand, m_rand
        if mode == _MASK_MODE_BLOCK:
            return x_block, m_block
        use_block = (
            jax.random.uniform(key_pick) < _MIXED_BLOCK_THRESHOLD
        )
        x_obs = jnp.where(use_block, x_block, x_rand)
        mask = jnp.where(use_block, m_block, m_rand)
        return x_obs, mask

    def reduction_weights(self, mask: jnp.ndarray) -> jnp.ndarray:
        """Build per-cell DSM weights from the observation mask.

        Target (missing) cells always have weight ``1``. Observed cells
        have weight ``obs_loss_weight`` (``0`` in the M1 default).

        Parameters
        ----------
        mask : jax.Array
            Observation mask shaped ``(B, T, F)``.

        Returns
        -------
        jax.Array
            Non-negative weights, same shape as ``mask``.
        """
        target = jnp.abs(mask - MASK_MISSING) < _MASK_COMPARE_TOL
        observed = jnp.abs(mask - MASK_OBSERVED) < _MASK_COMPARE_TOL
        return (
            target.astype(jnp.float32)
            + observed.astype(jnp.float32) * self.settings.obs_loss_weight
        )

    def score_target(
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
            Clean state (defines broadcast shape for ``std``).
        time : jax.Array
            Diffusion times, scalar or shaped ``(B,)``.

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
        span = self.sde.T_diff - self.settings.t_eps
        return self.settings.t_eps + span * u_rand


def dsm_settings_from_config(config: ToyM1Config) -> DSMLossSettings:
    """Build DSM settings from a validated toy M1 config.

    Parameters
    ----------
    config : ToyM1Config
        Nested config with ``data`` and ``train`` sections.

    Returns
    -------
    DSMLossSettings
        Validated DSM hyperparameters.
    """
    settings = DSMLossSettings(
        mask_rate=config.data.mask_rate,
        block_len=config.data.block_len,
        t_eps=config.train.t_eps,
        obs_loss_weight=config.train.obs_loss_weight,
        mask_mode=config.train.mask_mode,
    )
    settings.validate()
    return settings


def _weighted_mean_square(
        pred: jnp.ndarray,
        target: jnp.ndarray,
        weights: jnp.ndarray,
) -> jnp.ndarray:
    """Weighted mean of squared errors.

    Parameters
    ----------
    pred : jax.Array
        Predicted score.
    target : jax.Array
        Target score.
    weights : jax.Array
        Non-negative per-cell weights.

    Returns
    -------
    jax.Array
        Scalar ``sum(w * (pred - target)^2) / max(sum(w), 1)``.
    """
    sq_err = (pred - target) ** 2
    numer = jnp.sum(weights * sq_err)
    denom = jnp.maximum(jnp.sum(weights), _WEIGHT_FLOOR)
    return numer / denom
