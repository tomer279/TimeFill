"""Variance-preserving SDE forward process (Song et al.).

Classes
-------
VPSDE
    Linear-β VP-SDE with closed-form marginals and coefficient helpers.
"""

from __future__ import annotations

import jax.numpy as jnp

from src.data.config import SdeConfig

_HALF = 0.5
_VARIANCE_FLOOR = 0.0
_TIME_EXPAND_AXIS = -1


class VPSDE:
    """Variance-preserving diffusion SDE with a linear beta schedule.

    Forward Itô SDE (Song et al.)::

        dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dW

    with ``beta(t)`` linear from ``beta_min`` to ``beta_max`` on
    ``[0, T_diff]``. The Gaussian transition admits a closed form used by
    ``marginal_prob`` / ``forward_sample``.

    Public methods
    --------------
    beta
        Evaluate the linear noise schedule ``β(t)``.
    marginal_prob
        Return closed-form ``(mean, std)`` of ``p(x_t | x_0)``.
    forward_sample
        Draw ``x_t = mean + std * eps`` from the marginal.
    sde_coefficients
        Return ``(drift_coeff, diffusion)`` so drift is ``drift_coeff * x``.
    reverse_drift
        Anderson reverse-time drift given a score (WP5 stub helper).

    Instance variables
    ------------------
    beta_min : float
        Lower end of the linear ``β(t)`` schedule.
    beta_max : float
        Upper end of the linear ``β(t)`` schedule.
    T_diff : float
        Diffusion time horizon; ``t`` is expected in ``[0, T_diff]``.
    """

    def __init__(self, sde_config: SdeConfig) -> None:
        """Store validated VP-SDE schedule parameters.

        Parameters
        ----------
        sde_config : SdeConfig
            Config section with ``beta_min``, ``beta_max``, and ``T_diff``.
        """
        sde_config.validate()
        self.beta_min = float(sde_config.beta_min)
        self.beta_max = float(sde_config.beta_max)
        self.T_diff = float(sde_config.T_diff)

    def beta(self, time: jnp.ndarray | float) -> jnp.ndarray:
        """Evaluate linear ``β(t)`` on ``[0, T_diff]``.

        Parameters
        ----------
        time : float or jax.Array
            Diffusion time(s); broadcastable shapes are allowed.

        Returns
        -------
        jax.Array
            ``β(t) = β_min + (t / T_diff) * (β_max - β_min)``.
        """
        time_arr = jnp.asarray(time, dtype=jnp.float32)
        span = self.beta_max - self.beta_min
        return self.beta_min + (time_arr / self.T_diff) * span

    def marginal_prob(
            self,
            x0: jnp.ndarray,
            time: jnp.ndarray | float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Closed-form Gaussian marginal parameters for ``x_t | x_0``.

        Parameters
        ----------
        x0 : jax.Array
            Clean state; typically shaped ``(B, T, F)``.
        time : float or jax.Array
            Diffusion time(s), broadcastable to ``x0``.

        Returns
        -------
        mean : jax.Array
            ``exp(-0.5 * ∫β) * x0``, same shape as ``x0``.
        std : jax.Array
            ``sqrt(1 - exp(-∫β))``, broadcastable to ``x0``.
        """
        time_arr = _expand_time(time, x0.ndim)
        int_beta = self._integrated_beta(time_arr)
        mean_scale = jnp.exp(-_HALF * int_beta)
        mean = mean_scale * x0
        variance = 1.0 - jnp.exp(-int_beta)
        std = jnp.sqrt(jnp.maximum(variance, _VARIANCE_FLOOR))
        # Scalar / low-rank ``t`` expands to rank-3 ones; broadcast std to x0.
        std = jnp.broadcast_to(std, x0.shape)
        return mean, std

    def forward_sample(
            self,
            x0: jnp.ndarray,
            time: jnp.ndarray | float,
            eps: jnp.ndarray,
    ) -> jnp.ndarray:
        """Sample the VP marginal ``x_t = mean + std * eps``.

        Parameters
        ----------
        x0 : jax.Array
            Clean state shaped like the data windows.
        time : float or jax.Array
            Diffusion time(s), broadcastable to ``x0``.
        eps : jax.Array
            Standard normal noise with the same shape as ``x0``.

        Returns
        -------
        jax.Array
            Noised state ``x_t`` with the same shape as ``x0``.
        """
        mean, std = self.marginal_prob(x0, time)
        return mean + std * eps

    def sde_coefficients(
            self,
            time: jnp.ndarray | float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return multiplicative drift coefficient and diffusion scale.

        Parameters
        ----------
        time : float or jax.Array
            Diffusion time(s).

        Returns
        -------
        drift_coeff : jax.Array
            Factor such that the Itô drift is ``drift_coeff * x``
            (equals ``-0.5 * β(t)``).
        diffusion : jax.Array
            Diffusion coefficient ``sqrt(β(t))``.
        """
        beta_t = self.beta(time)
        drift_coeff = -_HALF * beta_t
        diffusion = jnp.sqrt(beta_t)
        return drift_coeff, diffusion

    def reverse_drift(
            self,
            state: jnp.ndarray,
            time: jnp.ndarray | float,
            score: jnp.ndarray,
    ) -> jnp.ndarray:
        """Anderson reverse-time drift given a score estimate (WP5 stub).

        Uses ``f(x,t) - g(t)^2 score`` with
        ``f = -0.5 β x`` and ``g = sqrt(β)``, i.e.
        ``-0.5 β x - β score``. This does **not** run a Diffrax solver;
        reverse sampling belongs to WP5.

        Parameters
        ----------
        state : jax.Array
            Current reverse-process state ``x``.
        time : float or jax.Array
            Diffusion time(s), broadcastable to ``state``.
        score : jax.Array
            Score estimate ``∇_x log p_t(x)`` (or conditional score),
            same shape as ``state``.

        Returns
        -------
        jax.Array
            Reverse drift evaluated at ``(state, time)``.
        """
        beta_t = self.beta(_expand_time(time, state.ndim))
        return -_HALF * beta_t * state - beta_t * score

    def _integrated_beta(self, time: jnp.ndarray) -> jnp.ndarray:
        """Integrate linear ``β`` from ``0`` to ``time``.

        Parameters
        ----------
        time : jax.Array
            Diffusion time(s).

        Returns
        -------
        jax.Array
            ``∫_0^t β(s) ds``.
        """
        span = self.beta_max - self.beta_min
        return self.beta_min * time + span * (time ** 2) / (
            2.0 * self.T_diff
        )


def _expand_time(
        time: jnp.ndarray | float,
        n_axes: int,
) -> jnp.ndarray:
    """Expand trailing axes so ``time`` can broadcast to rank ``n_axes``.

    Parameters
    ----------
    time : float or jax.Array
        Scalar or low-rank time tensor.
    n_axes : int
        Target rank (usually ``x0.ndim``).

    Returns
    -------
    jax.Array
        Time array with ``ndim == n_axes`` via trailing ``None`` axes.
    """
    time_arr = jnp.asarray(time, dtype=jnp.float32)
    while time_arr.ndim < n_axes:
        time_arr = jnp.expand_dims(time_arr, axis=_TIME_EXPAND_AXIS)
    return time_arr
