"""WP0 tests: YAML config loading and `(B, T, F)` schema contracts."""

from pathlib import Path

import jax.numpy as jnp
import pytest

from src.data.config import load_config, load_toy_m1
from src.data.schema import (
    BATCH_AXIS,
    FEATURE_AXIS,
    MASK_MISSING,
    MASK_OBSERVED,
    MISSING_VALUE_SENTINEL,
    TIME_AXIS,
    TensorSchema,
    apply_missing_sentinel,
    build_stub_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_CONFIG_PATH = REPO_ROOT / "configs" / "toy_m1.yaml"


def test_load_config_has_required_keys() -> None:
    """Loader exposes the locked M1 hyperparameter keys."""
    config = load_config(TOY_CONFIG_PATH)

    assert config.data.seq_len == 64
    assert config.data.n_features == 2
    assert config.data.mask_rate == pytest.approx(0.3)
    assert config.data.block_len == 8

    assert config.sde.beta_min == pytest.approx(0.1)
    assert config.sde.beta_max == pytest.approx(20.0)
    assert config.sde.t_diff == pytest.approx(1.0)

    assert config.train.optimizer.lr == pytest.approx(0.001)
    assert config.train.schedule.epochs == 40
    assert config.train.protocol.obs_loss_weight == pytest.approx(0.0)
    assert config.train.protocol.mask_mode == "mixed"
    assert config.train.protocol.artifact_dir == "artifacts/m1"

    assert config.sample.k_samples == 50
    assert config.sample.solver_steps == 100


def test_stub_batch_shapes_and_mask_convention() -> None:
    """Stub loader returns `(B, T, F)` with 1=obs / 0=miss and sentinel."""
    config, batch = load_toy_m1(TOY_CONFIG_PATH)
    schema = TensorSchema(config.data)
    schema.validate_batch(batch)

    batch_size, seq_len, n_features = batch.shape()
    assert batch_size == config.data.batch_size
    assert seq_len == config.data.seq_len
    assert n_features == config.data.n_features
    assert batch.x.shape[BATCH_AXIS] == batch_size
    assert batch.x.shape[TIME_AXIS] == seq_len
    assert batch.x.shape[FEATURE_AXIS] == n_features
    assert batch.m.shape == batch.x.shape

    unique_mask = set(jnp.unique(batch.m).tolist())
    assert unique_mask <= {MASK_MISSING, MASK_OBSERVED}

    is_missing = batch.m == MASK_MISSING
    assert bool(jnp.all(batch.x[is_missing] == MISSING_VALUE_SENTINEL))

    block_start = seq_len - config.data.block_len
    assert bool(jnp.all(batch.m[:, block_start:, :] == MASK_MISSING))
    assert bool(jnp.all(batch.m[:, :block_start, :] == MASK_OBSERVED))


def test_apply_missing_sentinel_preserves_observed() -> None:
    """Sentinel helper clears only masked cells."""
    config = load_config(TOY_CONFIG_PATH)
    clean = jnp.ones(
        (
            config.data.batch_size,
            config.data.seq_len,
            config.data.n_features,
        ),
        dtype=jnp.float32,
    )
    mask = jnp.ones_like(clean)
    mask = mask.at[:, -1, :].set(MASK_MISSING)

    filled = apply_missing_sentinel(clean, mask)
    assert bool(jnp.all(filled[:, :-1, :] == 1.0))
    assert bool(jnp.all(filled[:, -1, :] == MISSING_VALUE_SENTINEL))


def test_schema_rejects_bad_mask_alphabet() -> None:
    """Validator rejects masks outside `{0, 1}`."""
    config = load_config(TOY_CONFIG_PATH)
    batch = build_stub_batch(config.data)
    bad_mask = batch.m.at[0, 0, 0].set(2.0)
    schema = TensorSchema(config.data)
    with pytest.raises(ValueError, match="mask entries"):
        schema.validate_pair(batch.x, bad_mask)
