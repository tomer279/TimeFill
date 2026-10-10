"""WP3 tests: conditional score-net shapes, finiteness, parameter count."""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from src.data.config import load_config
from src.data.masking import random_mask
from src.data.schema import BATCH_AXIS, FEATURE_AXIS, TIME_AXIS
from src.data.synthetic import generate_sine_windows
from src.models.score_net import (
    ConditionalScoreNet,
    default_score_net_config,
)
from src.sde.vp_sde import VPSDE

REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_CONFIG_PATH = REPO_ROOT / "configs" / "toy_m1.yaml"

_SEED = 0
_DIFFUSION_TIME_FRAC = 0.5


@pytest.fixture
def toy_cfg():
    """Load the locked toy M1 config."""
    return load_config(TOY_CONFIG_PATH)


@pytest.fixture
def score_batch(toy_cfg):
    """Build realistic WP1/WP2 smoke inputs: sine → mask → VP noise."""
    data_cfg = toy_cfg.data
    shape = (
        data_cfg.batch_size,
        data_cfg.seq_len,
        data_cfg.n_features,
    )
    key_data, key_mask, key_eps = jax.random.split(jax.random.PRNGKey(_SEED), 3)
    x0 = generate_sine_windows(key_data, shape)
    x_obs, mask = random_mask(x0, data_cfg.mask_rate, key_mask)
    vp_sde = VPSDE(toy_cfg.sde)
    time = _DIFFUSION_TIME_FRAC * vp_sde.T_diff
    eps = jax.random.normal(key_eps, shape=x0.shape, dtype=jnp.float32)
    y_t = vp_sde.forward_sample(x0, time, eps)
    return y_t, time, mask, x_obs


@pytest.fixture
def score_net(toy_cfg) -> ConditionalScoreNet:
    """Initialize a default conditional score net for toy ``F``."""
    config = default_score_net_config(toy_cfg.data.n_features)
    return ConditionalScoreNet(jax.random.PRNGKey(_SEED), config)


def test_forward_shape_finite_and_param_count(
        score_net: ConditionalScoreNet,
        score_batch,
        toy_cfg,
        capsys,
) -> None:
    """Forward returns ``(B, T, F)``, no NaNs; parameter count is logged."""
    y_t, time, mask, x_obs = score_batch
    score = score_net(y_t, time, mask, x_obs)

    expected = (
        toy_cfg.data.batch_size,
        toy_cfg.data.seq_len,
        toy_cfg.data.n_features,
    )
    assert score.shape == expected
    assert score.shape[BATCH_AXIS] == toy_cfg.data.batch_size
    assert score.shape[TIME_AXIS] == toy_cfg.data.seq_len
    assert score.shape[FEATURE_AXIS] == toy_cfg.data.n_features
    assert bool(jnp.all(jnp.isfinite(score)))

    n_params = score_net.count_parameters()
    print(f"ConditionalScoreNet parameter count: {n_params}")
    captured = capsys.readouterr()
    assert "parameter count" in captured.out
    assert n_params > 0


def test_per_example_time_vector(
        score_net: ConditionalScoreNet,
        score_batch,
) -> None:
    """Per-example ``t`` of shape ``(B,)`` is accepted and stays finite."""
    y_t, time, mask, x_obs = score_batch
    batch_size = int(y_t.shape[BATCH_AXIS])
    time_b = jnp.full((batch_size,), float(time), dtype=jnp.float32)
    score = score_net(y_t, time_b, mask, x_obs)
    assert score.shape == y_t.shape
    assert bool(jnp.all(jnp.isfinite(score)))
