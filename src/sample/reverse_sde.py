"""Conditional reverse-time VP-SDE imputation via Diffrax (WP5).

Classes
-------
ImputeSolverSettings
    Solver steps, terminal ``t_eps``, quantile levels, noise-init mode.
ImputeContext
    Score model + ``VPSDE`` + solver settings for reverse imputation.
ImputeQuery
    Observations, mask, timestamps, PRNG key, and sample count for one call.
ReverseSDEImputer
    Multi-sample reverse SDE with observation freezing / replacement.

Functions
---------
context_from_toy_config
    Map ``ToyM1Config`` (+ model) into an ``ImputeContext``.
context_from_real_config
    Map ``RealM2Config`` (+ model) into an ``ImputeContext``.
grid_timestamps
    Integer grid ``0 … T-1`` used when wall-clock times are absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import (
    ControlTerm,
    Euler,
    MultiTerm,
    ODETerm,
    SaveAt,
    VirtualBrownianTree,
    diffeqsolve,
)

from src.data.config import ToyM1Config
from src.data.real_config import RealM2Config
from src.data.schema import (
    BATCH_AXIS,
    MASK_OBSERVED,
    N_AXES,
    TIME_AXIS,
)
from src.sde.vp_sde import VPSDE

_SAMPLE_AXIS = 1
_QUANTILE_AXIS = -1
_DEFAULT_QUANTILE_LOW = 0.1
_DEFAULT_QUANTILE_HIGH = 0.9
_DEFAULT_NOISE_INIT = "missing_only"
_NOISE_INIT_MISSING_ONLY = "missing_only"
_NOISE_INIT_FULL = "full"
_ALLOWED_NOISE_INIT = frozenset(
    {_NOISE_INIT_MISSING_ONLY, _NOISE_INIT_FULL}
)
_BM_TOL_FRACTION = 0.5
_MAX_STEP_SLACK = 2
_MASK_COMPARE_TOL = 0.5
_UNBATCHED_RANK = 2
_TIME_AXIS_TF = 0
_FEATURE_AXIS_TF = 1


@dataclass(frozen=True)
class ImputeSolverSettings:
    """Diffrax reverse-SDE knobs for M1 imputation.

    Public methods
    --------------
    validate
        Check positivity and allowed categorical fields.
    to_dict
        Serialize fields to a plain dict.

    Instance variables
    ------------------
    solver_steps : int
        Number of Euler–Maruyama steps from ``t=T_diff`` to ``t_eps``.
    t_eps : float
        Terminal diffusion time (avoid exactly ``0``).
    quantile_low : float
        Lower predictive quantile in ``[0, 1]`` (default 0.1).
    quantile_high : float
        Upper predictive quantile in ``[0, 1]`` (default 0.9).
    noise_init : str
        ``missing_only`` (default) or ``full`` — see ASSUMPTIONS.md.
    """

    solver_steps: int
    t_eps: float
    quantile_low: float = _DEFAULT_QUANTILE_LOW
    quantile_high: float = _DEFAULT_QUANTILE_HIGH
    noise_init: str = _DEFAULT_NOISE_INIT

    def validate(self) -> None:
        """Raise ``ValueError`` if solver settings are invalid."""
        if self.solver_steps <= 0:
            raise ValueError(
                f"solver_steps must be positive, got {self.solver_steps}"
            )
        if self.t_eps <= 0.0:
            raise ValueError(f"t_eps must be positive, got {self.t_eps}")
        if not 0.0 <= self.quantile_low < self.quantile_high <= 1.0:
            raise ValueError(
                "require 0 <= quantile_low < quantile_high <= 1, "
                f"got {self.quantile_low}, {self.quantile_high}"
            )
        if self.noise_init not in _ALLOWED_NOISE_INIT:
            raise ValueError(
                "noise_init must be one of "
                f"{sorted(_ALLOWED_NOISE_INIT)}, got {self.noise_init!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable dict of fields.

        Returns
        -------
        dict of str to Any
            Field name to value.
        """
        return {
            "solver_steps": self.solver_steps,
            "t_eps": self.t_eps,
            "quantile_low": self.quantile_low,
            "quantile_high": self.quantile_high,
            "noise_init": self.noise_init,
        }


@dataclass(frozen=True)
class ImputeContext:
    """Bundled model / SDE / solver settings for reverse imputation.

    Public methods
    --------------
    validate
        Validate nested solver settings and SDE horizon vs ``t_eps``.
    diffusion_horizon
        Return ``sde.t_diff``.

    Instance variables
    ------------------
    model : Any
        Callable score model ``(y_t, t, mask, x_obs) -> score`` with
        batched ``(B, T, F)`` tensors (e.g. ``ConditionalScoreNet``).
    sde : VPSDE
        Forward VP-SDE providing ``reverse_drift`` / coefficients.
    settings : ImputeSolverSettings
        Diffrax and summary hyperparameters.
    """

    model: Any
    sde: VPSDE
    settings: ImputeSolverSettings

    def validate(self) -> None:
        """Validate solver settings against the SDE horizon."""
        self.settings.validate()
        if self.settings.t_eps >= self.sde.t_diff:
            raise ValueError(
                "t_eps must be < T_diff, "
                f"got t_eps={self.settings.t_eps}, "
                f"T_diff={self.sde.t_diff}"
            )

    def diffusion_horizon(self) -> float:
        """Return the forward diffusion horizon ``T_diff``.

        Returns
        -------
        float
            ``self.sde.t_diff``.
        """
        return float(self.sde.t_diff)


@dataclass(frozen=True)
class ImputeQuery:
    """Inputs for one ``ReverseSDEImputer.impute`` call.

    Public methods
    --------------
    validate
        Check that ``k_samples`` is positive.
    batched_pair
        Batch ``observations`` / ``mask`` if the input was unbatched.

    Instance variables
    ------------------
    observations : jax.Array
        Values shaped ``(B, T, F)`` or ``(T, F)``; missing cells
        should be the sentinel ``0.0``.
    mask : jax.Array
        Observation mask, same shape as ``observations``
        (``1`` observed, ``0`` missing).
    timestamps : jax.Array
        Grid times shaped ``(T,)`` or ``(B, T)``. Unused by the M1
        dynamics (regular grid); validated for length only.
    key : jax.Array
        PRNG key for init noise and Brownian paths.
    k_samples : int
        Number of posterior samples ``K`` per series.
    """

    observations: jnp.ndarray
    mask: jnp.ndarray
    timestamps: jnp.ndarray
    key: jax.Array
    k_samples: int

    def validate(self) -> None:
        """Raise ``ValueError`` if ``k_samples`` is not positive."""
        if self.k_samples <= 0:
            raise ValueError(
                f"k_samples must be positive, got {self.k_samples}"
            )

    def batched_pair(self) -> tuple[jnp.ndarray, jnp.ndarray, bool]:
        """Return batched observations/mask and an unbatched flag.

        Returns
        -------
        observations : jax.Array
            Batched values shaped ``(B, T, F)``.
        mask : jax.Array
            Batched mask shaped ``(B, T, F)``.
        unbatched : bool
            ``True`` if the original inputs were rank-2 ``(T, F)``.
        """
        return _ensure_batch_pair(self.observations, self.mask)


class _ReversePack(eqx.Module):
    """Packed tensors for a single reverse solve (keeps arities small).

    Implemented as an ``eqx.Module`` so arrays remain dynamic under
    ``filter_jit`` / ``vmap`` (a plain dataclass is treated as static).

    Public methods
    --------------
    flat_state
        Flatten ``state_0`` to shape ``(T * F,)``.
    window_shape
        Return ``(T, F)``.

    Instance variables
    ------------------
    state_0 : jax.Array
        Initial reverse state ``(T, F)``.
    observations : jax.Array
        Observed context ``(T, F)``.
    mask : jax.Array
        Observation mask ``(T, F)``.
    """

    state_0: jnp.ndarray
    observations: jnp.ndarray
    mask: jnp.ndarray

    def flat_state(self) -> jnp.ndarray:
        """Return the flattened initial state.

        Returns
        -------
        jax.Array
            Vector of length ``T * F``.
        """
        return self.state_0.reshape((-1,))

    def window_shape(self) -> tuple[int, int]:
        """Return the ``(T, F)`` window shape.

        Returns
        -------
        tuple of int
            Time and feature dimensions.
        """
        return (
            int(self.state_0.shape[_TIME_AXIS_TF]),
            int(self.state_0.shape[_FEATURE_AXIS_TF]),
        )


class ReverseSDEImputer:
    """Conditional reverse-time SDE imputation with observation replacement.

    Integrates the Anderson reverse VP-SDE in reverse-time coordinate
    ``τ = T_diff - t`` with Diffrax ``Euler`` (Euler–Maruyama) and a
    ``VirtualBrownianTree``. Observed coordinates are frozen during the
    solve (zero drift / diffusion on ``m == 1``) and hard-replaced with
    ``observations`` at the end.

    Public methods
    --------------
    impute
        Draw ``K`` posterior fills from an ``ImputeQuery``.
    summarize_samples
        Reduce an existing ``(B, K, T, F)`` stack to mean and quantiles.

    Instance variables
    ------------------
    context : ImputeContext
        Model, SDE, and solver settings.
    """

    def __init__(self, context: ImputeContext) -> None:
        """Store a validated imputation context.

        Parameters
        ----------
        context : ImputeContext
            Bundled score model, ``VPSDE``, and solver settings.
        """
        context.validate()
        self.context = context

    def impute(
            self,
            query: ImputeQuery,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Impute missing cells via conditional reverse-SDE sampling.

        Parameters
        ----------
        query : ImputeQuery
            Observations, mask, timestamps, PRNG key, and ``K``.

        Returns
        -------
        samples : jax.Array
            ``(B, K, T, F)`` or ``(K, T, F)`` if the input was unbatched.
        mean : jax.Array
            Posterior mean fill, ``(B, T, F)`` or ``(T, F)``.
        quantiles : jax.Array
            Low/high quantiles on the last axis, shaped
            ``(B, T, F, 2)`` or ``(T, F, 2)``.
        """
        query.validate()
        obs_b, mask_b, unbatched = query.batched_pair()
        _validate_timestamps(
            query.timestamps,
            int(obs_b.shape[TIME_AXIS]),
        )
        samples_b = self._impute_batched(
            obs_b,
            mask_b,
            query.key,
            query.k_samples,
        )
        mean_b, quantiles_b = self.summarize_samples(samples_b)
        if unbatched:
            return samples_b[0], mean_b[0], quantiles_b[0]
        return samples_b, mean_b, quantiles_b

    def summarize_samples(
            self,
            samples: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute posterior mean and configured quantile bands.

        Parameters
        ----------
        samples : jax.Array
            Stack shaped ``(B, K, T, F)``.

        Returns
        -------
        mean : jax.Array
            Mean over the sample axis, shaped ``(B, T, F)``.
        quantiles : jax.Array
            ``(..., 2)`` with low then high quantile along the last axis.
        """
        settings = self.context.settings
        mean = jnp.mean(samples, axis=_SAMPLE_AXIS)
        low = jnp.quantile(
            samples,
            settings.quantile_low,
            axis=_SAMPLE_AXIS,
        )
        high = jnp.quantile(
            samples,
            settings.quantile_high,
            axis=_SAMPLE_AXIS,
        )
        quantiles = jnp.stack([low, high], axis=_QUANTILE_AXIS)
        return mean, quantiles

    def _impute_batched(
            self,
            observations: jnp.ndarray,
            mask: jnp.ndarray,
            key: jax.Array,
            k_samples: int,
    ) -> jnp.ndarray:
        """Impute a batched ``(B, T, F)`` pair; return ``(B, K, T, F)``.

        Parameters
        ----------
        observations : jax.Array
            Batched observations.
        mask : jax.Array
            Batched mask.
        key : jax.Array
            PRNG key.
        k_samples : int
            Posterior sample count ``K``.

        Returns
        -------
        jax.Array
            Samples shaped ``(B, K, T, F)``.
        """
        batch_size = int(observations.shape[BATCH_AXIS])
        keys = jax.random.split(key, batch_size)
        return jax.vmap(
            lambda obs_i, mask_i, key_i: self._impute_window(
                obs_i,
                mask_i,
                key_i,
                k_samples,
            )
        )(observations, mask, keys)

    def _impute_window(
            self,
            observations: jnp.ndarray,
            mask: jnp.ndarray,
            key: jax.Array,
            k_samples: int,
    ) -> jnp.ndarray:
        """Draw ``K`` imputations for one window ``(T, F)``.

        Parameters
        ----------
        observations : jax.Array
            Single window values ``(T, F)``.
        mask : jax.Array
            Single window mask ``(T, F)``.
        key : jax.Array
            PRNG key.
        k_samples : int
            Number of samples ``K``.

        Returns
        -------
        jax.Array
            Samples shaped ``(K, T, F)``.
        """
        keys = jax.random.split(key, k_samples)
        return jax.vmap(
            lambda sample_key: self._sample_single(
                observations,
                mask,
                sample_key,
            )
        )(keys)

    def _sample_single(
            self,
            observations: jnp.ndarray,
            mask: jnp.ndarray,
            key: jax.Array,
    ) -> jnp.ndarray:
        """One reverse-SDE path for a single ``(T, F)`` window.

        Parameters
        ----------
        observations : jax.Array
            Context values ``(T, F)``.
        mask : jax.Array
            Observation mask ``(T, F)``.
        key : jax.Array
            PRNG key.

        Returns
        -------
        jax.Array
            Completed window ``(T, F)``.
        """
        key_init, key_bm = jax.random.split(key)
        state_0 = self._initial_state(key_init, observations, mask)
        pack = _ReversePack(state_0, observations, mask)
        state_1 = _integrate_reverse(
            self.context.model,
            self.context.sde,
            self.context.settings,
            key_bm,
            pack,
        )
        return _replace_observed(state_1, observations, mask)

    def _initial_state(
            self,
            key: jax.Array,
            observations: jnp.ndarray,
            mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Build the reverse-process state at ``t = T_diff``.

        Parameters
        ----------
        key : jax.Array
            PRNG key for Gaussian noise.
        observations : jax.Array
            Observed context ``(T, F)``.
        mask : jax.Array
            Observation mask ``(T, F)``.

        Returns
        -------
        jax.Array
            Initial state ``(T, F)`` under the configured noise strategy.
        """
        noise = jax.random.normal(key, observations.shape, dtype=jnp.float32)
        if self.context.settings.noise_init == _NOISE_INIT_FULL:
            return noise
        observed = jnp.abs(mask - MASK_OBSERVED) < _MASK_COMPARE_TOL
        return jnp.where(observed, observations, noise)


def context_from_toy_config(
        model: Any,
        config: ToyM1Config,
) -> ImputeContext:
    """Build an ``ImputeContext`` from a toy M1 config and score model.

    Parameters
    ----------
    model : Any
        Trained (or randomly initialized) score callable / Equinox module.
    config : ToyM1Config
        Validated config; uses ``sde``, ``sample``, and ``train.protocol.t_eps``.

    Returns
    -------
    ImputeContext
        Ready for ``ReverseSDEImputer``.
    """
    settings = ImputeSolverSettings(
        solver_steps=int(config.sample.solver_steps),
        t_eps=float(config.train.protocol.t_eps),
    )
    context = ImputeContext(
        model=model,
        sde=VPSDE(config.sde),
        settings=settings,
    )
    context.validate()
    return context


def context_from_real_config(
        model: Any,
        config: RealM2Config,
) -> ImputeContext:
    """Build an ``ImputeContext`` from a real M2 config and score model.

    Parameters
    ----------
    model : Any
        Trained (or randomly initialized) score callable / Equinox module.
    config : RealM2Config
        Validated config; uses ``sde``, ``sample``, and ``train.protocol.t_eps``.

    Returns
    -------
    ImputeContext
        Ready for ``ReverseSDEImputer``.
    """
    settings = ImputeSolverSettings(
        solver_steps=int(config.sample.solver_steps),
        t_eps=float(config.train.protocol.t_eps),
    )
    context = ImputeContext(
        model=model,
        sde=VPSDE(config.sde),
        settings=settings,
    )
    context.validate()
    return context


def grid_timestamps(seq_len: int) -> jnp.ndarray:
    """Return default M1 regular-grid timestamps ``0 … T-1``.

    Parameters
    ----------
    seq_len : int
        Window length ``T``.

    Returns
    -------
    jax.Array
        Float32 vector shaped ``(T,)``.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    return jnp.arange(seq_len, dtype=jnp.float32)


def _integrate_reverse(
        model: Any,
        sde: VPSDE,
        settings: ImputeSolverSettings,
        key: jax.Array,
        pack: _ReversePack,
) -> jnp.ndarray:
    """Integrate the reverse SDE from ``t=T_diff`` to ``t=t_eps``.

    Parameters
    ----------
    model : Any
        Score callable with batched ``(B, T, F)`` signature.
    sde : VPSDE
        VP-SDE providing reverse drift and diffusion scale.
    settings : ImputeSolverSettings
        Step count and terminal ``t_eps``.
    key : jax.Array
        Brownian-tree key.
    pack : _ReversePack
        Initial state, observations, and mask for one window.

    Returns
    -------
    jax.Array
        State at ``t ≈ t_eps``, shape ``(T, F)``.
    """
    t0 = 0.0
    t1 = sde.t_diff - settings.t_eps
    step_dt = t1 / float(settings.solver_steps)
    y0_flat = pack.flat_state()
    terms = _reverse_multiterm(
        model,
        sde,
        pack,
        key,
        (t0, t1, step_dt),
    )
    solution = diffeqsolve(
        terms,
        Euler(),
        t0=t0,
        t1=t1,
        dt0=step_dt,
        y0=y0_flat,
        saveat=SaveAt(t1=True),
        max_steps=int(settings.solver_steps) + _MAX_STEP_SLACK,
    )
    return _saved_endpoint(solution, pack.state_0.shape)


def _reverse_multiterm(
        model: Any,
        sde: VPSDE,
        pack: _ReversePack,
        key: jax.Array,
        time_span: tuple[float, float, float],
) -> MultiTerm:
    """Build Diffrax ``MultiTerm`` for reverse integration.

    Parameters
    ----------
    model : Any
        Score callable.
    sde : VPSDE
        VP-SDE coefficient provider.
    pack : _ReversePack
        Window state / observations / mask.
    key : jax.Array
        Brownian-tree key.
    time_span : tuple of float
        ``(t0, t1, step_dt)`` in reverse-time coordinate.

    Returns
    -------
    MultiTerm
        Drift ODE term plus diffusion control term.
    """
    t0, t1, step_dt = time_span
    n_flat = int(pack.state_0.size)
    brownian = VirtualBrownianTree(
        t0,
        t1,
        tol=step_dt * _BM_TOL_FRACTION,
        shape=(n_flat,),
        key=key,
    )
    return MultiTerm(
        ODETerm(_make_drift_fn(model, sde, pack)),
        ControlTerm(_make_diffusion_fn(sde, pack.mask), brownian),
    )


def _saved_endpoint(
        solution: Any,
        state_shape: tuple[int, ...],
) -> jnp.ndarray:
    """Return the last saved Diffrax state, reshaped.

    Parameters
    ----------
    solution : Any
        Diffrax ``diffeqsolve`` result.
    state_shape : tuple of int
        Target ``(T, F)`` shape.

    Returns
    -------
    jax.Array
        Endpoint state reshaped to ``state_shape``.

    Raises
    ------
    RuntimeError
        If Diffrax saved no states.
    """
    ys = solution.ys
    if ys is None:
        raise RuntimeError(
            "diffeqsolve returned no saved states "
            "(expected SaveAt(t1=True))"
        )
    return ys[-1].reshape(state_shape)


def _make_drift_fn(
        model: Any,
        sde: VPSDE,
        pack: _ReversePack,
):
    """Build ``dY/dτ = -reverse_drift`` with observed coords frozen.

    Parameters
    ----------
    model : Any
        Score callable.
    sde : VPSDE
        Forward / reverse coefficient provider.
    pack : _ReversePack
        Holds ``mask`` and ``observations`` for conditioning.

    Returns
    -------
    callable
        Diffrax drift ``(tau, y_flat, args) -> dydtau_flat``.
    """
    horizon = sde.t_diff
    mask = pack.mask
    x_obs = pack.observations
    missing = 1.0 - mask

    def drift(tau: Any, y_flat: Any, _args: Any) -> jnp.ndarray:
        state = y_flat.reshape(mask.shape)
        time = horizon - tau
        score = model(
            state[None, ...],
            time,
            mask[None, ...],
            x_obs[None, ...],
        )[0]
        d_tau = -sde.reverse_drift(state, time, score)
        return (missing * d_tau).reshape((-1,))

    return drift


def _make_diffusion_fn(sde: VPSDE, mask: jnp.ndarray):
    """Build diagonal diffusion ``g(t)`` with zeros on observed cells.

    Returns a dense diagonal matrix so ``ControlTerm`` applies independent
    noise per coordinate without requiring the optional ``lineax`` package.

    Parameters
    ----------
    sde : VPSDE
        Provides ``sde_coefficients(t)``.
    mask : jax.Array
        Observation mask ``(T, F)``.

    Returns
    -------
    callable
        Diffrax diffusion ``(tau, y_flat, args) -> (n, n)`` matrix.
    """
    horizon = sde.t_diff
    missing_flat = (1.0 - mask).reshape((-1,))

    def diffusion(tau: Any, y_flat: Any, _args: Any) -> jnp.ndarray:
        del y_flat
        time = horizon - tau
        _drift_coeff, g_t = sde.sde_coefficients(time)
        diagonal = missing_flat * g_t
        return jnp.diag(diagonal)

    return diffusion


def _replace_observed(
        state: jnp.ndarray,
        observations: jnp.ndarray,
        mask: jnp.ndarray,
) -> jnp.ndarray:
    """Hard-replace observed coordinates with data values.

    Parameters
    ----------
    state : jax.Array
        Reverse-SDE endpoint ``(T, F)``.
    observations : jax.Array
        Observed context ``(T, F)``.
    mask : jax.Array
        Observation mask ``(T, F)``.

    Returns
    -------
    jax.Array
        State with ``m == 1`` cells set to ``observations``.
    """
    observed = jnp.abs(mask - MASK_OBSERVED) < _MASK_COMPARE_TOL
    return jnp.where(observed, observations, state)


def _ensure_batch_pair(
        observations: jnp.ndarray,
        mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, bool]:
    """Add a batch axis when inputs are unbatched ``(T, F)``.

    Parameters
    ----------
    observations : jax.Array
        Values ``(B, T, F)`` or ``(T, F)``.
    mask : jax.Array
        Mask with the same shape as ``observations``.

    Returns
    -------
    observations_b : jax.Array
        Batched values ``(B, T, F)``.
    mask_b : jax.Array
        Batched mask ``(B, T, F)``.
    unbatched : bool
        ``True`` if the original inputs lacked a batch axis.
    """
    if observations.shape != mask.shape:
        raise ValueError(
            "observations and mask shapes must match; "
            f"got {observations.shape} vs {mask.shape}"
        )
    if observations.ndim == N_AXES:
        return observations, mask, False
    if observations.ndim == _UNBATCHED_RANK:
        return observations[None, ...], mask[None, ...], True
    raise ValueError(
        "observations must have shape (B, T, F) or (T, F); "
        f"got {observations.shape}"
    )


def _validate_timestamps(timestamps: jnp.ndarray, seq_len: int) -> None:
    """Check that timestamps match the window length ``T``.

    Parameters
    ----------
    timestamps : jax.Array
        Grid times shaped ``(T,)`` or ``(B, T)``.
    seq_len : int
        Expected window length ``T``.

    Raises
    ------
    ValueError
        If the last axis length is not ``seq_len``.
    """
    time_arr = jnp.asarray(timestamps)
    if time_arr.ndim == 1 and int(time_arr.shape[0]) == seq_len:
        return
    if (
        time_arr.ndim == _UNBATCHED_RANK
        and int(time_arr.shape[-1]) == seq_len
    ):
        return
    raise ValueError(
        "timestamps must have shape (T,) or (B, T) with "
        f"T={seq_len}; got {time_arr.shape}"
    )
