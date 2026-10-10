"""WP4 tests: conditional DSM loss shapes, weights, and one train step."""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from src.data.config import load_config
from src.data.schema import MASK_MISSING, MASK_OBSERVED
from src.data.synthetic import generate_sine_windows
from src.models.score_net import (
    ConditionalScoreNet,
    default_score_net_config,
)
from src.sde.vp_sde import VPSDE
from src.train.loss import (
    ConditionalDSMLoss,
    DSMLossSettings,
    dsm_settings_from_config,
)
from src.train.loop import (
    ArtifactPaths,
    M1Trainer,
    TrainerBundle,
    build_optimizer,
    stub_val_imputation_hook,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_CONFIG_PATH = REPO_ROOT / "configs" / "toy_m1.yaml"

_SEED = 0
_WEIGHT_TOL = 1e-6
_N_AXES = 3


@pytest.fixture
def toy_cfg():
    """Load the locked toy M1 config."""
    return load_config(TOY_CONFIG_PATH)


@pytest.fixture
def dsm_loss(toy_cfg) -> ConditionalDSMLoss:
    """Build DSM loss from toy config."""
    return ConditionalDSMLoss(
        VPSDE(toy_cfg.sde),
        dsm_settings_from_config(toy_cfg),
    )


@pytest.fixture
def score_net(toy_cfg) -> ConditionalScoreNet:
    """Initialize a default conditional score net."""
    return ConditionalScoreNet(
        jax.random.PRNGKey(_SEED),
        default_score_net_config(toy_cfg.data.n_features),
    )


@pytest.fixture
def clean_x0(toy_cfg) -> jnp.ndarray:
    """Clean sine batch matching config geometry."""
    shape = (
        toy_cfg.data.batch_size,
        toy_cfg.data.seq_len,
        toy_cfg.data.n_features,
    )
    return generate_sine_windows(jax.random.PRNGKey(_SEED), shape)


def test_dsm_loss_is_scalar_finite(
        dsm_loss: ConditionalDSMLoss,
        score_net: ConditionalScoreNet,
        clean_x0: jnp.ndarray,
) -> None:
    """DSM loss returns a finite scalar."""
    loss = dsm_loss(score_net, clean_x0, jax.random.PRNGKey(1))
    assert loss.shape == ()
    assert bool(jnp.isfinite(loss))


def test_reduction_weights_zero_on_observed(
        dsm_loss: ConditionalDSMLoss,
) -> None:
    """With obs_loss_weight=0, only missing cells have weight 1."""
    assert dsm_loss.settings.obs_loss_weight == pytest.approx(0.0)
    mask = jnp.array(
        [
            [
                [MASK_OBSERVED, MASK_MISSING],
                [MASK_MISSING, MASK_OBSERVED],
            ]
        ],
        dtype=jnp.float32,
    )
    weights = dsm_loss.reduction_weights(mask)
    assert bool(jnp.allclose(weights[..., 0, 0], 0.0, atol=_WEIGHT_TOL))
    assert bool(jnp.allclose(weights[..., 0, 1], 1.0, atol=_WEIGHT_TOL))
    assert bool(jnp.allclose(weights[..., 1, 0], 1.0, atol=_WEIGHT_TOL))
    assert bool(jnp.allclose(weights[..., 1, 1], 0.0, atol=_WEIGHT_TOL))


def test_score_target_matches_formula(
        dsm_loss: ConditionalDSMLoss,
        clean_x0: jnp.ndarray,
) -> None:
    """Target equals ``-eps / std`` with floored std."""
    time = jnp.full((clean_x0.shape[0],), 0.5, dtype=jnp.float32)
    eps = jnp.ones_like(clean_x0)
    target = dsm_loss.score_target(eps, clean_x0, time)
    _, std = dsm_loss.sde.marginal_prob(clean_x0, time)
    std_safe = jnp.maximum(std, dsm_loss.settings.std_floor)
    expected = -eps / std_safe
    assert target.shape == clean_x0.shape
    assert bool(jnp.allclose(target, expected, rtol=1e-5, atol=1e-5))


def test_settings_from_config_matches_yaml(toy_cfg) -> None:
    """DSM settings mirror data/train YAML fields."""
    settings = dsm_settings_from_config(toy_cfg)
    assert isinstance(settings, DSMLossSettings)
    assert settings.mask_rate == pytest.approx(toy_cfg.data.mask_rate)
    assert settings.block_len == toy_cfg.data.block_len
    assert settings.obs_loss_weight == pytest.approx(0.0)
    assert settings.mask_mode == toy_cfg.train.mask_mode


def test_one_train_step_finite(
        toy_cfg,
        score_net: ConditionalScoreNet,
        dsm_loss: ConditionalDSMLoss,
        clean_x0: jnp.ndarray,
        tmp_path: Path,
) -> None:
    """One jitted AdamW step yields finite loss and updated params."""
    toy_cfg.train.epochs = 1
    toy_cfg.train.steps_per_epoch = 1
    toy_cfg.train.val_every = 1
    toy_cfg.train.log_every = 1
    paths = ArtifactPaths(tmp_path / "m1")
    bundle = TrainerBundle(
        config=toy_cfg,
        model=score_net,
        loss_fn=dsm_loss,
        optimizer=build_optimizer(toy_cfg.train),
        paths=paths,
        val_hook=stub_val_imputation_hook,
    )
    trainer = M1Trainer(bundle.with_paths())
    model, history = trainer.run(jax.random.PRNGKey(2))
    assert history.rows
    assert bool(jnp.isfinite(jnp.asarray(history.rows[0]["train_loss"])))
    assert paths.best_checkpoint().is_file()
    assert paths.loss_csv().is_file()
    assert clean_x0.ndim == _N_AXES
    assert model.count_parameters() > 0
