"""WP5 tests: impute API shapes, observed freeze, sine-gap smoke."""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from src.data.config import load_config
from src.data.masking import block_mask
from src.data.schema import MASK_OBSERVED
from src.data.synthetic import generate_sine_windows
from src.models.score_net import (
    ConditionalScoreNet,
    default_score_net_config,
)
from src.sample.reverse_sde import (
    ImputeContext,
    ImputeQuery,
    ImputeSolverSettings,
    ReverseSDEImputer,
    context_from_toy_config,
    grid_timestamps,
)
from src.sde.vp_sde import VPSDE

REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_CONFIG_PATH = REPO_ROOT / "configs" / "toy_m1.yaml"

_SEED = 0
_K_SAMPLES = 4
_SOLVER_STEPS = 12
_BATCH_SIZE = 2
_SEQ_LEN = 32
_N_FEATURES = 1
_BLOCK_LEN = 8
_STD_FLOOR = 1e-5
_OBS_ATOL = 1e-5
_Q_LOW_INDEX = 0
_Q_HIGH_INDEX = 1
_MASK_MISSING_THRESH = 0.5


@pytest.fixture
def toy_cfg():
    """Load the locked toy M1 config."""
    return load_config(TOY_CONFIG_PATH)


@pytest.fixture
def fast_settings(toy_cfg) -> ImputeSolverSettings:
    """CPU-friendly solver settings for unit tests."""
    settings = ImputeSolverSettings(
        solver_steps=_SOLVER_STEPS,
        t_eps=float(toy_cfg.train.protocol.t_eps),
    )
    settings.validate()
    return settings


def test_impute_shapes_and_observed_freeze(toy_cfg, fast_settings) -> None:
    """Shapes match contract; observed cells equal input observations."""
    key = jax.random.PRNGKey(_SEED)
    key_data, key_mask, key_model, key_impute = jax.random.split(key, 4)
    shape = (_BATCH_SIZE, _SEQ_LEN, _N_FEATURES)
    x0 = generate_sine_windows(key_data, shape, noise_std=0.0)
    x_obs, mask = block_mask(x0, _BLOCK_LEN, key_mask)
    model = ConditionalScoreNet(
        key_model,
        default_score_net_config(_N_FEATURES),
    )
    context = ImputeContext(
        model=model,
        sde=VPSDE(toy_cfg.sde),
        settings=fast_settings,
    )
    imputer = ReverseSDEImputer(context)
    samples, mean, quantiles = imputer.impute(
        ImputeQuery(
            observations=x_obs,
            mask=mask,
            timestamps=grid_timestamps(_SEQ_LEN),
            key=key_impute,
            k_samples=_K_SAMPLES,
        )
    )
    assert samples.shape == (_BATCH_SIZE, _K_SAMPLES, _SEQ_LEN, _N_FEATURES)
    assert mean.shape == (_BATCH_SIZE, _SEQ_LEN, _N_FEATURES)
    assert quantiles.shape == (
        _BATCH_SIZE,
        _SEQ_LEN,
        _N_FEATURES,
        2,
    )
    assert bool(
        jnp.all(
            quantiles[..., _Q_LOW_INDEX] <= quantiles[..., _Q_HIGH_INDEX]
        )
    )
    observed = jnp.abs(mask - MASK_OBSERVED) < _MASK_MISSING_THRESH
    assert bool(jnp.allclose(mean[observed], x_obs[observed], atol=_OBS_ATOL))
    assert bool(
        jnp.allclose(
            samples[:, 0][observed],
            x_obs[observed],
            atol=_OBS_ATOL,
        )
    )


def test_impute_unbatched_shapes(toy_cfg, fast_settings) -> None:
    """Unbatched ``(T, F)`` input yields ``(K, T, F)`` samples."""
    key = jax.random.PRNGKey(_SEED + 1)
    key_data, key_mask, key_model, key_impute = jax.random.split(key, 4)
    shape = (1, _SEQ_LEN, _N_FEATURES)
    x0 = generate_sine_windows(key_data, shape, noise_std=0.0)[0]
    x_obs_b, mask_b = block_mask(x0[None], _BLOCK_LEN, key_mask)
    x_obs, mask = x_obs_b[0], mask_b[0]
    model = ConditionalScoreNet(
        key_model,
        default_score_net_config(_N_FEATURES),
    )
    imputer = ReverseSDEImputer(
        ImputeContext(model, VPSDE(toy_cfg.sde), fast_settings)
    )
    samples, mean, quantiles = imputer.impute(
        ImputeQuery(
            observations=x_obs,
            mask=mask,
            timestamps=grid_timestamps(_SEQ_LEN),
            key=key_impute,
            k_samples=_K_SAMPLES,
        )
    )
    assert samples.shape == (_K_SAMPLES, _SEQ_LEN, _N_FEATURES)
    assert mean.shape == (_SEQ_LEN, _N_FEATURES)
    assert quantiles.shape == (_SEQ_LEN, _N_FEATURES, 2)


def test_mean_beats_zero_on_sine_gap_with_oracle(
        toy_cfg,
        fast_settings,
) -> None:
    """Oracle marginal score: imputed mean beats always-zero on a gap."""
    key = jax.random.PRNGKey(_SEED + 2)
    key_data, key_mask, key_impute = jax.random.split(key, 3)
    shape = (1, _SEQ_LEN, _N_FEATURES)
    x0 = generate_sine_windows(key_data, shape, noise_std=0.0)
    x_obs, mask = block_mask(x0, _BLOCK_LEN, key_mask)
    sde = VPSDE(toy_cfg.sde)
    oracle = _OracleScore(sde, x0)
    imputer = ReverseSDEImputer(
        ImputeContext(oracle, sde, fast_settings)
    )
    _samples, mean, _quantiles = imputer.impute(
        ImputeQuery(
            observations=x_obs,
            mask=mask,
            timestamps=grid_timestamps(_SEQ_LEN),
            key=key_impute,
            k_samples=_K_SAMPLES,
        )
    )
    missing = mask < _MASK_MISSING_THRESH
    weight = missing.astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(weight), 1.0)
    mae_impute = jnp.sum(jnp.abs(mean - x0) * weight) / denom
    mae_zero = jnp.sum(jnp.abs(x0) * weight) / denom
    assert float(mae_impute) < float(mae_zero)


def test_context_from_toy_config_smoke(toy_cfg) -> None:
    """Factory wires sample.solver_steps and train.protocol.t_eps."""
    model = ConditionalScoreNet(
        jax.random.PRNGKey(_SEED),
        default_score_net_config(toy_cfg.data.n_features),
    )
    context = context_from_toy_config(model, toy_cfg)
    assert context.settings.solver_steps == toy_cfg.sample.solver_steps
    assert context.settings.t_eps == toy_cfg.train.protocol.t_eps


class _OracleScore:
    """Closed-form VP marginal score toward a known clean window.

    Public methods
    --------------
    __call__
        Batched score ``-(y - mean) / std^2``.
    clean_shape
        Return the stored clean ``x0`` shape.

    Instance variables
    ------------------
    sde : VPSDE
        Forward process for ``marginal_prob``.
    x0_true : jax.Array
        Ground-truth clean windows ``(B, T, F)``.
    """

    def __init__(self, sde: VPSDE, x0_true: jnp.ndarray) -> None:
        """Store SDE and clean target windows.

        Parameters
        ----------
        sde : VPSDE
            VP-SDE used for closed-form marginals.
        x0_true : jax.Array
            Clean ground truth shaped ``(B, T, F)``.
        """
        self.sde = sde
        self.x0_true = x0_true

    def __call__(
            self,
            y_t: jnp.ndarray,
            time: jnp.ndarray | float,
            mask: jnp.ndarray,
            x_obs: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return the Gaussian transition score toward ``x0_true``.

        Parameters
        ----------
        y_t : jax.Array
            Noisy state ``(B, T, F)``.
        time : float or jax.Array
            Diffusion time.
        mask : jax.Array
            Unused; accepted for score-net call compatibility.
        x_obs : jax.Array
            Unused; accepted for score-net call compatibility.

        Returns
        -------
        jax.Array
            Score with the same shape as ``y_t``.
        """
        del mask, x_obs
        x0 = jnp.broadcast_to(self.x0_true, y_t.shape)
        mean, std = self.sde.marginal_prob(x0, time)
        std_safe = jnp.maximum(std, _STD_FLOOR)
        return -(y_t - mean) / (std_safe ** 2)

    def clean_shape(self) -> tuple[int, ...]:
        """Return the shape of the stored clean windows.

        Returns
        -------
        tuple of int
            ``x0_true.shape``.
        """
        return tuple(int(size) for size in self.x0_true.shape)
