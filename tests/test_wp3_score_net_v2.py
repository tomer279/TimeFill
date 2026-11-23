"""WP3 tests: tcn_v2 score net shapes, RF, time embed, capacity, toy train.

Functions
---------
test_tcn_v2_output_shapes
    Forward shapes for ``F in {1,3,6}`` and ``T in {16,64}``.
test_receptive_field_tcn_v2_is_one_two_nine
    Stem-included M7.5 RF is exactly 129.
test_receptive_field_legacy_v1_is_seventeen
    Legacy stem + ``[1,2,4]`` body matches the finding (RF 17).
test_scaled_time_embedding_resolves_close_times
    ``scale=1000`` separates close times; ``scale=1`` does not.
test_m7_5_v2_parameter_count_in_target_band
    M7.5 v2 config lands in ~150k–400k params (logged).
test_existing_v1_checkpoint_still_loads
    Done-when: legacy v1 ``.eqx`` still deserializes and runs.
test_v2_sidecar_round_trip_loads
    Save v2 + sidecar; ``load_checkpoint_with_config`` restores it.
test_build_score_net_returns_v1_for_tcn_v1
    Factory returns ``ConditionalScoreNet`` for legacy arch.
test_non_divisible_norm_groups_raises
    ``norm=group`` with indivisible channels fails validate.
test_v1_rejects_non_legacy_arch_knobs
    ``tcn_v1`` refuses custom dilations / scale / cond / norm.
test_v2_trains_on_toy_path
    Done-when: a few M1 steps succeed with ``arch=tcn_v2``.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from src.data.config import load_config
from src.models.score_net import (
    ConditionalScoreNet,
    ScoreNetConfig,
    default_score_net_config,
)
from src.sde.vp_sde import VPSDE
from src.train.artifacts import (
    ArtifactPaths,
    load_checkpoint_with_config,
    save_checkpoint,
    save_checkpoint_with_sidecar,
    stub_val_imputation_hook,
)
from src.train.loop import M1Trainer, TrainerBundle, build_optimizer
from src.train.loss import ConditionalDSMLoss, dsm_settings_from_config

REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_CONFIG_PATH = REPO_ROOT / "configs" / "toy_m1.yaml"
_M2_CKPT = (
    REPO_ROOT / "artifacts" / "m2" / "checkpoints" / "best_score_net.eqx"
)

_SEED = 0
_BATCH = 2
_FEATURE_COUNTS = (1, 3, 6)
_SEQ_LENS = (16, 64)
_M75_DILATIONS = (1, 2, 4, 8, 16, 32)
_M75_HIDDEN = 64
_M75_TIME_EMBED = 64
_M75_TIME_SCALE = 1000.0
_M75_RF = 129
_PARAM_COUNT_LO = 150_000
_PARAM_COUNT_HI = 400_000
_T_NEAR = 1.0e-3
_T_FAR = 1.0e-2
_COSINE_DISTANCE_MIN = 0.05
_TOY_V2_EPOCHS = 2
_TOY_V2_STEPS = 2


def _wp3_api():
    """Import WP3 symbols."""
    from src.models.score_net import (
        ConditionalScoreNetV2,
        build_score_net,
        receptive_field,
        sinusoidal_time_embedding,
    )

    return (
        ConditionalScoreNetV2,
        build_score_net,
        receptive_field,
        sinusoidal_time_embedding,
    )


def _m7_5_v2_config(n_features: int = 6) -> ScoreNetConfig:
    """M7.5 ``real_m7_5.yaml`` model knobs (capacity target)."""
    return ScoreNetConfig(
        n_features=n_features,
        arch="tcn_v2",
        hidden_channels=_M75_HIDDEN,
        n_layers=len(_M75_DILATIONS),
        dilations=_M75_DILATIONS,
        kernel_size=3,
        time_embed_dim=_M75_TIME_EMBED,
        time_embed_scale=_M75_TIME_SCALE,
        time_cond="per_block",
        norm="group",
        norm_groups=8,
    )


def _toy_v2_config(n_features: int) -> ScoreNetConfig:
    """Smaller v2 net for CPU toy train smoke."""
    dilations = (1, 2, 4, 8)
    return ScoreNetConfig(
        n_features=n_features,
        arch="tcn_v2",
        hidden_channels=32,
        n_layers=len(dilations),
        dilations=dilations,
        kernel_size=3,
        time_embed_dim=32,
        time_embed_scale=_M75_TIME_SCALE,
        time_cond="per_block",
        norm="group",
        norm_groups=8,
    )


def _cosine_distance(left: jnp.ndarray, right: jnp.ndarray) -> float:
    """Return ``1 - cos_sim`` for two vectors."""
    left_n = left / jnp.linalg.norm(left)
    right_n = right / jnp.linalg.norm(right)
    return float(1.0 - jnp.dot(left_n, right_n))


@pytest.mark.parametrize("n_features", _FEATURE_COUNTS)
@pytest.mark.parametrize("seq_len", _SEQ_LENS)
def test_tcn_v2_output_shapes(n_features: int, seq_len: int) -> None:
    """``tcn_v2`` forward returns finite ``(B, T, F)``."""
    score_v2, build_score_net, _rf, _embed = _wp3_api()
    config = _m7_5_v2_config(n_features=n_features)
    model = build_score_net(jax.random.PRNGKey(_SEED), config)
    assert isinstance(model, score_v2)
    shape = (_BATCH, seq_len, n_features)
    y_t = jnp.zeros(shape, dtype=jnp.float32)
    mask = jnp.ones(shape, dtype=jnp.float32)
    out = model(y_t, 0.5, mask, y_t)
    assert out.shape == shape
    assert bool(jnp.all(jnp.isfinite(out)))


def test_receptive_field_tcn_v2_is_one_two_nine() -> None:
    """Stem-included M7.5 dilations → RF exactly 129."""
    _v2, _build, receptive_field, _embed = _wp3_api()
    assert receptive_field(_m7_5_v2_config()) == _M75_RF


def test_receptive_field_legacy_v1_is_seventeen() -> None:
    """Legacy defaults match the review finding (RF 17 with stem)."""
    _v2, _build, receptive_field, _embed = _wp3_api()
    assert receptive_field(default_score_net_config(n_features=6)) == 17


def test_scaled_time_embedding_resolves_close_times() -> None:
    """``scale=1000`` separates close times; unscaled does not (finding 6)."""
    _v2, _build, _rf, sinusoidal_time_embedding = _wp3_api()
    near_s = sinusoidal_time_embedding(
        jnp.asarray(_T_NEAR),
        _M75_TIME_EMBED,
        scale=_M75_TIME_SCALE,
    )
    far_s = sinusoidal_time_embedding(
        jnp.asarray(_T_FAR),
        _M75_TIME_EMBED,
        scale=_M75_TIME_SCALE,
    )
    near_u = sinusoidal_time_embedding(
        jnp.asarray(_T_NEAR),
        _M75_TIME_EMBED,
        scale=1.0,
    )
    far_u = sinusoidal_time_embedding(
        jnp.asarray(_T_FAR),
        _M75_TIME_EMBED,
        scale=1.0,
    )
    assert _cosine_distance(near_s, far_s) > _COSINE_DISTANCE_MIN
    assert _cosine_distance(near_u, far_u) < _COSINE_DISTANCE_MIN


def test_m7_5_v2_parameter_count_in_target_band(capsys) -> None:
    """M7.5 v2 config param count is logged and in ~150k–400k."""
    _v2, build_score_net, _rf, _embed = _wp3_api()
    model = build_score_net(jax.random.PRNGKey(_SEED), _m7_5_v2_config())
    n_params = model.count_parameters()
    print(f"ConditionalScoreNetV2 parameter count: {n_params}")
    captured = capsys.readouterr()
    assert "parameter count" in captured.out
    assert _PARAM_COUNT_LO <= n_params <= _PARAM_COUNT_HI


def test_existing_v1_checkpoint_still_loads(tmp_path: Path) -> None:
    """Done-when: a v1 checkpoint still loads (claim M2 if present)."""
    key = jax.random.PRNGKey(_SEED)
    if _M2_CKPT.is_file():
        ckpt = _M2_CKPT
        n_features = 6
    else:
        n_features = 2
        ckpt = tmp_path / "best_score_net.eqx"
        save_checkpoint(
            ckpt,
            ConditionalScoreNet(key, default_score_net_config(n_features)),
        )
    restored = load_checkpoint_with_config(
        ckpt,
        n_features=n_features,
        key=jax.random.PRNGKey(_SEED + 1),
    )
    assert isinstance(restored, ConditionalScoreNet)
    y_t = jnp.zeros((_BATCH, 8, n_features), dtype=jnp.float32)
    mask = jnp.ones_like(y_t)
    out = restored(y_t, 0.1, mask, y_t)
    assert out.shape == y_t.shape
    assert bool(jnp.all(jnp.isfinite(out)))


def test_v2_sidecar_round_trip_loads(tmp_path: Path) -> None:
    """v2 ``.eqx`` + sidecar reloads via ``load_checkpoint_with_config``."""
    score_v2, build_score_net, _rf, _embed = _wp3_api()
    config = _toy_v2_config(n_features=2)
    model = build_score_net(jax.random.PRNGKey(_SEED), config)
    ckpt = tmp_path / "best_score_net.eqx"
    save_checkpoint_with_sidecar(
        ckpt,
        model,
        {
            "score_net": config.to_dict(),
            "parameterization": "eps",
            "arch_version": 2,
        },
    )
    restored = load_checkpoint_with_config(
        ckpt,
        n_features=2,
        key=jax.random.PRNGKey(_SEED + 1),
    )
    assert isinstance(restored, score_v2)
    y_t = jnp.zeros((_BATCH, 8, 2), dtype=jnp.float32)
    mask = jnp.ones_like(y_t)
    out = restored(y_t, 0.3, mask, y_t)
    assert out.shape == y_t.shape
    assert bool(jnp.all(jnp.isfinite(out)))


def test_build_score_net_returns_v1_for_tcn_v1() -> None:
    """Factory returns legacy class for ``arch=tcn_v1``."""
    _v2, build_score_net, _rf, _embed = _wp3_api()
    model = build_score_net(
        jax.random.PRNGKey(_SEED),
        default_score_net_config(n_features=2),
    )
    assert isinstance(model, ConditionalScoreNet)


def test_non_divisible_norm_groups_raises() -> None:
    """``norm=group`` requires ``hidden_channels % norm_groups == 0``."""
    config = ScoreNetConfig(
        n_features=2,
        arch="tcn_v2",
        hidden_channels=32,
        n_layers=2,
        dilations=(1, 2),
        time_cond="per_block",
        norm="group",
        norm_groups=7,
    )
    with pytest.raises(ValueError, match="divisible by norm_groups"):
        config.validate()


def test_v1_rejects_non_legacy_arch_knobs() -> None:
    """``tcn_v1`` refuses knobs it does not implement."""
    with pytest.raises(ValueError, match="dilations"):
        ScoreNetConfig(
            n_features=2,
            arch="tcn_v1",
            n_layers=3,
            dilations=(1, 2, 4),
        ).validate()
    with pytest.raises(ValueError, match="time_embed_scale"):
        ScoreNetConfig(
            n_features=2,
            arch="tcn_v1",
            time_embed_scale=1000.0,
        ).validate()
    with pytest.raises(ValueError, match="time_cond"):
        ScoreNetConfig(
            n_features=2,
            arch="tcn_v2",
            n_layers=2,
            dilations=(1, 2),
            time_cond="stem",
            norm="group",
        ).validate()


def test_v2_trains_on_toy_path(tmp_path: Path) -> None:
    """Done-when: toy M1 trainer runs a few steps with ``arch=tcn_v2``."""
    score_v2, build_score_net, _rf, _embed = _wp3_api()
    config = load_config(TOY_CONFIG_PATH)
    config.train.protocol.artifact_dir = str(tmp_path)
    config.train.schedule.epochs = _TOY_V2_EPOCHS
    config.train.schedule.steps_per_epoch = _TOY_V2_STEPS
    key_model, key_train = jax.random.split(jax.random.PRNGKey(_SEED))
    model = build_score_net(
        key_model,
        _toy_v2_config(config.data.n_features),
    )
    assert isinstance(model, score_v2)
    bundle = TrainerBundle(
        config=config,
        model=model,
        loss_fn=ConditionalDSMLoss(
            VPSDE(config.sde),
            dsm_settings_from_config(config),
        ),
        optimizer=build_optimizer(config.train),
        paths=ArtifactPaths(tmp_path),
        val_hook=stub_val_imputation_hook,
    )
    bundle.validate()
    _trained, history = M1Trainer(bundle.with_paths()).run(key_train)
    assert len(history.rows) == _TOY_V2_EPOCHS
    assert all(
        jnp.isfinite(jnp.asarray(row["train_loss"]))
        for row in history.rows
    )
