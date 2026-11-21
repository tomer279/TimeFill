"""Eps-prediction → score wrapper for reverse sampling (M7.5 WP2).

Classes
-------
ScoreFromEps
    Calls an eps-net and returns ``-eps_hat / std(t)``.

Functions
---------
as_score_model
    Return ``model`` unchanged for ``score``, else wrap with ``ScoreFromEps``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from src.sde.vp_sde import VPSDE

_STD_FLOOR = 1.0e-5
_PARAM_SCORE = "score"
_PARAM_EPS = "eps"


class ScoreFromEps:
    """Treat network output as ``eps_hat`` and convert to a VP score.

    Public methods
    --------------
    __call__
        ``(y_t, t, mask, x_obs) -> -eps_hat / std(t)``.
    eps_to_score
        Convert a raw ``eps_hat`` tensor at diffusion time ``t``.

    Instance variables
    ------------------
    model : Any
        Eps-predicting callable with the score-net signature.
    sde : VPSDE
        Provides ``marginal_prob`` for ``std(t)``.
    std_floor : float
        Floor applied to ``std(t)`` before division.
    """

    def __init__(
            self,
            model: Any,
            sde: VPSDE,
            std_floor: float = _STD_FLOOR,
    ) -> None:
        """Store the eps model and SDE used for the conversion.

        Parameters
        ----------
        model : Any
            Eps-predicting callable.
        sde : VPSDE
            Forward VP-SDE.
        std_floor : float
            Positive floor for ``std(t)``.
        """
        self.model = model
        self.sde = sde
        self.std_floor = float(std_floor)

    def __call__(
            self,
            y_t: jnp.ndarray,
            time: jnp.ndarray | float,
            mask: jnp.ndarray,
            x_obs: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return score ``-eps_hat / std(t)`` from the wrapped eps model."""
        eps_hat = self.model(y_t, time, mask, x_obs)
        return self.eps_to_score(eps_hat, y_t, time)

    def eps_to_score(
            self,
            eps_hat: jnp.ndarray,
            y_like: jnp.ndarray,
            time: jnp.ndarray | float,
    ) -> jnp.ndarray:
        """Convert ``eps_hat`` to score using ``std(t)`` from ``y_like`` shape.

        Parameters
        ----------
        eps_hat : jax.Array
            Predicted noise.
        y_like : jax.Array
            Tensor used only for broadcast shape of ``std``.
        time : float or jax.Array
            Diffusion time(s).

        Returns
        -------
        jax.Array
            ``-eps_hat / max(std(t), std_floor)``.
        """
        _, std = self.sde.marginal_prob(y_like, time)
        std_safe = jnp.maximum(std, self.std_floor)
        return -eps_hat / std_safe


def as_score_model(
        model: Any,
        sde: VPSDE,
        parameterization: str,
        std_floor: float = _STD_FLOOR,
) -> Any:
    """Wrap ``model`` when ``parameterization == 'eps'``.

    Parameters
    ----------
    model : Any
        Raw network (score or eps).
    sde : VPSDE
        Forward SDE for ``std(t)``.
    parameterization : str
        ``score`` or ``eps``.
    std_floor : float
        Floor for the eps→score conversion.

    Returns
    -------
    Any
        Score-valued callable for ``ReverseSDEImputer``.
    """
    if parameterization == _PARAM_SCORE:
        return model
    if parameterization != _PARAM_EPS:
        raise ValueError(
            "parameterization must be 'score' or 'eps', "
            f"got {parameterization!r}"
        )
    return ScoreFromEps(model=model, sde=sde, std_floor=float(std_floor))
