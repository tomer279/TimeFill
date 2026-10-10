"""WP3 hermetic tests: real DSM mask weights and one jitted train step."""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.data.real_config import load_real_m2_config
from src.data.schema import MASK_MISSING
from src.models.score_net import (
    ConditionalScoreNet,
    default_score_net_config,
)
from src.sde.vp_sde import VPSDE
from src.train.loop import build_optimizer
from src.train.real_loop import build_real_train_step
from src.train.real_pool import RealBatchPool
from src.train.real_loss import (
    RealConditionalDSMLoss,
    RealDSMBatch,
    dsm_settings_from_real_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG_PATH = REPO_ROOT / "configs" / "real_m2.yaml"

_SEED = 0
_BATCH = 2
_SEQ = 8
_FEAT = 6
_NATIVE_GAP_T = 3
_WEIGHT_TOL = 1e-6


@pytest.fixture
def real_cfg():
    """Load real_m2 YAML (does not require processed NPZ)."""
    return load_real_m2_config(REAL_CONFIG_PATH)


def _tiny_batch() -> RealDSMBatch:
    """Build a tiny synthetic real batch with one native gap column-time."""
    rng = np.random.default_rng(_SEED)
    values = rng.normal(size=(_BATCH, _SEQ, _FEAT)).astype(np.float32)
    masks = np.ones((_BATCH, _SEQ, _FEAT), dtype=np.float32)
    masks[:, _NATIVE_GAP_T, :] = MASK_MISSING
    values[:, _NATIVE_GAP_T, :] = 0.0
    return RealDSMBatch(
        x0=jnp.asarray(values),
        m_native=jnp.asarray(masks),
    )


def test_load_real_m2_config_allows_six_features(real_cfg) -> None:
    """real_m2.yaml loads with F=6 (M1 loader would reject this)."""
    assert real_cfg.data.n_features == _FEAT
    assert len(real_cfg.data.feature_names) == _FEAT


def test_real_dsm_weights_skip_native_missing(real_cfg) -> None:
    """Loss weights are zero on natively missing cells."""
    loss_fn = RealConditionalDSMLoss(
        VPSDE(real_cfg.sde),
        dsm_settings_from_real_config(real_cfg),
    )
    batch = _tiny_batch()
    _x_obs, m_train, m_art = loss_fn.sample_mask(
        jax.random.PRNGKey(1),
        batch,
    )
    weights = loss_fn.reduction_weights(batch.m_native, m_art, m_train)
    native_missing = jnp.abs(batch.m_native - MASK_MISSING) < _WEIGHT_TOL
    assert bool(jnp.all(weights[native_missing] == 0.0))
    assert bool(jnp.all(m_train[native_missing] == MASK_MISSING))


def test_real_dsm_loss_scalar_finite(real_cfg) -> None:
    """One DSM evaluation on a tiny batch is a finite scalar."""
    loss_fn = RealConditionalDSMLoss(
        VPSDE(real_cfg.sde),
        dsm_settings_from_real_config(real_cfg),
    )
    model = ConditionalScoreNet(
        jax.random.PRNGKey(_SEED),
        default_score_net_config(_FEAT),
    )
    loss = loss_fn(model, _tiny_batch(), jax.random.PRNGKey(2))
    assert loss.shape == ()
    assert bool(jnp.isfinite(loss))


def test_real_train_step_finite(real_cfg) -> None:
    """One jitted AdamW step on a tiny real batch stays finite."""
    real_cfg.data.batch_size = _BATCH
    loss_fn = RealConditionalDSMLoss(
        VPSDE(real_cfg.sde),
        dsm_settings_from_real_config(real_cfg),
    )
    model = ConditionalScoreNet(
        jax.random.PRNGKey(_SEED),
        default_score_net_config(_FEAT),
    )
    optimizer = build_optimizer(real_cfg.train)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    step_fn = build_real_train_step(optimizer, loss_fn)
    batch = _tiny_batch()
    new_model, _new_state, loss, grad_norm = step_fn(
        model,
        opt_state,
        batch,
        jax.random.PRNGKey(3),
    )
    assert bool(jnp.isfinite(loss))
    assert bool(jnp.isfinite(grad_norm))
    assert new_model is not None


def test_real_batch_pool_sample_shapes() -> None:
    """RealBatchPool train/val sampling honors (B, T, F)."""
    values = jnp.ones((5, _SEQ, _FEAT), dtype=jnp.float32)
    masks = jnp.ones((5, _SEQ, _FEAT), dtype=jnp.float32)
    pool = RealBatchPool(values, masks, values[:3], masks[:3])
    train = pool.sample_train(jax.random.PRNGKey(0), _BATCH)
    val = pool.fixed_val_batch(_BATCH)
    assert train.x0.shape == (_BATCH, _SEQ, _FEAT)
    assert val.m_native.shape == (_BATCH, _SEQ, _FEAT)
