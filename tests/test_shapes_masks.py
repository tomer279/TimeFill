"""WP1 tests: synthetic generators, mask rates, contiguity, and shapes."""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from src.data.config import load_config
from src.data.masking import block_mask, random_mask
from src.data.schema import (
    BATCH_AXIS,
    FEATURE_AXIS,
    MASK_MISSING,
    MASK_OBSERVED,
    MISSING_VALUE_SENTINEL,
    TIME_AXIS,
    TensorSchema,
)
from src.data.synthetic import generate_ar_windows, generate_sine_windows

REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_CONFIG_PATH = REPO_ROOT / "configs" / "toy_m1.yaml"

_RATE_TOL = 0.02
_LARGE_BATCH = 256
_SEED = 0
_FIRST = 0


@pytest.fixture
def toy_data_cfg():
    """Load the locked toy `data` section."""
    return load_config(TOY_CONFIG_PATH).data


def test_sine_and_ar_shapes(toy_data_cfg) -> None:
    """Generators return `(B, T, F)` float windows."""
    shape = (
        toy_data_cfg.batch_size,
        toy_data_cfg.seq_len,
        toy_data_cfg.n_features,
    )
    key_sine, key_ar = jax.random.split(jax.random.PRNGKey(_SEED))
    sine = generate_sine_windows(key_sine, shape)
    ar = generate_ar_windows(key_ar, shape)

    assert sine.shape == shape
    assert ar.shape == shape
    assert sine.dtype == jnp.float32
    assert ar.dtype == jnp.float32


def test_random_mask_rate_and_shapes(toy_data_cfg) -> None:
    """MCAR empirical missing rate is near the requested rate."""
    shape = (
        _LARGE_BATCH,
        toy_data_cfg.seq_len,
        toy_data_cfg.n_features,
    )
    key_data, key_mask = jax.random.split(jax.random.PRNGKey(_SEED))
    clean = generate_sine_windows(key_data, shape)
    rate = toy_data_cfg.mask_rate
    x_obs, mask = random_mask(clean, rate, key_mask)

    assert x_obs.shape == clean.shape
    assert mask.shape == clean.shape
    unique = set(jnp.unique(mask).tolist())
    assert unique <= {MASK_MISSING, MASK_OBSERVED}

    empirical = float(jnp.mean(mask == MASK_MISSING))
    assert empirical == pytest.approx(rate, abs=_RATE_TOL)


def test_block_mask_contiguous_and_length(toy_data_cfg) -> None:
    """Each (batch, feature) missing span is contiguous of length block_len."""
    shape = (
        toy_data_cfg.batch_size,
        toy_data_cfg.seq_len,
        toy_data_cfg.n_features,
    )
    key_data, key_mask = jax.random.split(jax.random.PRNGKey(_SEED))
    clean = generate_ar_windows(key_data, shape)
    block_len = toy_data_cfg.block_len
    _x_obs, mask = block_mask(clean, block_len, key_mask)

    batch_size = mask.shape[BATCH_AXIS]
    n_features = mask.shape[FEATURE_AXIS]
    for batch_idx in range(batch_size):
        for feature_idx in range(n_features):
            missing_times = jnp.where(
                mask[batch_idx, :, feature_idx] == MASK_MISSING
            )[0]
            assert int(missing_times.size) == block_len
            start = int(missing_times[_FIRST])
            stop = int(missing_times[-1])
            assert stop - start + 1 == block_len
            assert bool(
                jnp.all(
                    missing_times
                    == jnp.arange(start, start + block_len)
                )
            )


def test_masks_preserve_observed_and_sentinel(toy_data_cfg) -> None:
    """Observed cells match clean x; missing cells equal sentinel 0.0."""
    shape = (
        toy_data_cfg.batch_size,
        toy_data_cfg.seq_len,
        toy_data_cfg.n_features,
    )
    keys = jax.random.split(jax.random.PRNGKey(_SEED), 4)
    clean = generate_sine_windows(keys[0], shape)

    x_rand, m_rand = random_mask(clean, toy_data_cfg.mask_rate, keys[1])
    x_block, m_block = block_mask(clean, toy_data_cfg.block_len, keys[2])

    for x_obs, mask in ((x_rand, m_rand), (x_block, m_block)):
        observed = mask == MASK_OBSERVED
        missing = mask == MASK_MISSING
        assert bool(jnp.allclose(x_obs[observed], clean[observed]))
        assert bool(
            jnp.allclose(x_obs[missing], MISSING_VALUE_SENTINEL)
        )


def test_masked_batch_passes_schema(toy_data_cfg) -> None:
    """Masked outputs satisfy TensorSchema contracts."""
    shape = (
        toy_data_cfg.batch_size,
        toy_data_cfg.seq_len,
        toy_data_cfg.n_features,
    )
    key_data, key_mask = jax.random.split(jax.random.PRNGKey(_SEED))
    clean = generate_sine_windows(key_data, shape)
    x_obs, mask = block_mask(clean, toy_data_cfg.block_len, key_mask)
    schema = TensorSchema(toy_data_cfg)
    schema.validate_pair(x_obs, mask)
    assert x_obs.shape[TIME_AXIS] == toy_data_cfg.seq_len
    assert x_obs.shape[FEATURE_AXIS] == toy_data_cfg.n_features
