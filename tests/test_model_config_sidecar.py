"""WP0 tests: model_config.json sidecar write/read and checkpoint load fallback."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from src.models.score_net import (
    ConditionalScoreNet,
    ScoreNetConfig,
    default_score_net_config,
)
from src.train.artifacts import load_checkpoint, save_checkpoint

_SEED = 0
_N_FEATURES = 2
_SEQ_LEN = 8
_BATCH = 2
_REPO = Path(__file__).resolve().parents[1]
_M2_CKPT = _REPO / "artifacts" / "m2" / "checkpoints" / "best_score_net.eqx"


def _tiny_model() -> ConditionalScoreNet:
    """Build a tiny score net for hermetic checkpoint tests."""
    return ConditionalScoreNet(
        jax.random.PRNGKey(_SEED),
        default_score_net_config(_N_FEATURES),
    )


def _sidecar_api():
    """Import WP0 sidecar helpers (missing until implementation lands)."""
    from src.train.artifacts import (
        load_checkpoint_with_config,
        load_model_config,
        save_checkpoint_with_sidecar,
        save_model_config,
    )

    return (
        save_model_config,
        load_model_config,
        save_checkpoint_with_sidecar,
        load_checkpoint_with_config,
    )


def test_model_config_json_round_trip(tmp_path: Path) -> None:
    """Sidecar JSON round-trips ScoreNetConfig + parameterization."""
    save_model_config, load_model_config, _, _ = _sidecar_api()
    config = default_score_net_config(_N_FEATURES)
    payload = {
        "score_net": config.to_dict(),
        "parameterization": "score",
        "arch_version": 1,
    }
    path = tmp_path / "model_config.json"
    save_model_config(path, payload)
    loaded = load_model_config(path)
    assert loaded == payload
    assert loaded["score_net"]["n_features"] == _N_FEATURES
    assert loaded["parameterization"] == "score"
    assert loaded["arch_version"] == 1


def test_save_checkpoint_with_sidecar_writes_both_files(tmp_path: Path) -> None:
    """Saving with sidecar writes ``.eqx`` and ``model_config.json`` beside it."""
    _, load_model_config, save_checkpoint_with_sidecar, _ = _sidecar_api()
    model = _tiny_model()
    ckpt = tmp_path / "checkpoints" / "best_score_net.eqx"
    save_checkpoint_with_sidecar(
        ckpt,
        model,
        {
            "score_net": default_score_net_config(_N_FEATURES).to_dict(),
            "parameterization": "score",
            "arch_version": 1,
        },
    )
    assert ckpt.is_file()
    assert (ckpt.parent / "model_config.json").is_file()
    sidecar = load_model_config(ckpt.parent / "model_config.json")
    assert sidecar["parameterization"] == "score"


def test_load_checkpoint_with_config_uses_sidecar_template(
        tmp_path: Path,
) -> None:
    """Loader reads a non-default sidecar; legacy template would mismatch."""
    _, _, save_checkpoint_with_sidecar, load_checkpoint_with_config = (
        _sidecar_api()
    )
    score_cfg = ScoreNetConfig(
        n_features=_N_FEATURES,
        hidden_channels=16,
    )
    model = ConditionalScoreNet(jax.random.PRNGKey(_SEED), score_cfg)
    ckpt = tmp_path / "best_score_net.eqx"
    save_checkpoint_with_sidecar(
        ckpt,
        model,
        {
            "score_net": score_cfg.to_dict(),
            "parameterization": "score",
            "arch_version": 1,
        },
    )
    restored = load_checkpoint_with_config(
        ckpt,
        n_features=_N_FEATURES,
        key=jax.random.PRNGKey(_SEED + 1),
    )
    y_t = jnp.zeros((_BATCH, _SEQ_LEN, _N_FEATURES), dtype=jnp.float32)
    mask = jnp.ones_like(y_t)
    out = restored(y_t, 0.5, mask, y_t)
    assert out.shape == y_t.shape
    assert bool(jnp.all(jnp.isfinite(out)))

    legacy_like = ConditionalScoreNet(
        jax.random.PRNGKey(_SEED),
        default_score_net_config(_N_FEATURES),
    )
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        load_checkpoint(ckpt, legacy_like)


def test_load_checkpoint_without_sidecar_uses_legacy_defaults(
        tmp_path: Path,
) -> None:
    """Missing sidecar falls back to legacy ``default_score_net_config``."""
    _, _, _, load_checkpoint_with_config = _sidecar_api()
    model = _tiny_model()
    ckpt = tmp_path / "best_score_net.eqx"
    save_checkpoint(ckpt, model)
    assert not (ckpt.parent / "model_config.json").is_file()

    restored = load_checkpoint_with_config(
        ckpt,
        n_features=_N_FEATURES,
        key=jax.random.PRNGKey(_SEED + 2),
    )
    like = ConditionalScoreNet(
        jax.random.PRNGKey(_SEED),
        default_score_net_config(_N_FEATURES),
    )
    via_legacy = load_checkpoint(ckpt, like)
    y_t = jnp.ones((_BATCH, _SEQ_LEN, _N_FEATURES), dtype=jnp.float32)
    mask = jnp.ones_like(y_t)
    assert restored(y_t, 0.2, mask, y_t).shape == via_legacy(
        y_t, 0.2, mask, y_t
    ).shape


def test_existing_m2_checkpoint_still_loads() -> None:
    """Claim M2 ``.eqx`` loads through the sidecar-aware helper (no sidecar)."""
    if not _M2_CKPT.is_file():
        pytest.skip("artifacts/m2 checkpoint not present in this workspace")
    _, _, _, load_checkpoint_with_config = _sidecar_api()
    restored = load_checkpoint_with_config(
        _M2_CKPT,
        n_features=6,
        key=jax.random.PRNGKey(_SEED),
    )
    y_t = jnp.zeros((_BATCH, _SEQ_LEN, 6), dtype=jnp.float32)
    mask = jnp.ones_like(y_t)
    out = restored(y_t, 0.1, mask, y_t)
    assert out.shape == y_t.shape


def test_score_net_config_to_dict_includes_arch_fields() -> None:
    """Legacy ScoreNetConfig.to_dict includes M7.5 arch keys at v1 defaults."""
    config = ScoreNetConfig(n_features=_N_FEATURES)
    payload = config.to_dict()
    assert payload["arch"] == "tcn_v1"
    assert payload["time_embed_scale"] == pytest.approx(1.0)
    assert payload["time_cond"] == "stem"
    assert payload["norm"] == "none"
