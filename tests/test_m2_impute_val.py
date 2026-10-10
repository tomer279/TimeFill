"""WP4 hermetic tests: metric masking, scaler restore, impute wiring."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.data.normalize import FeatureScaler
from src.data.real.masks import MaskComposer, artificial_target_mask
from src.data.real_config import load_real_m2_config
from src.data.schema import MASK_MISSING, MASK_OBSERVED
from src.eval.metrics import EvalTensors, ImputationEvaluator
from src.eval.m2_val_impute import (
    MetricsArtifactWriter,
    OriginalScaleRestorer,
    metric_observation_mask,
)
from src.models.score_net import (
    ConditionalScoreNet,
    default_score_net_config,
)
from src.sample.reverse_sde import (
    ImputeQuery,
    ReverseSDEImputer,
    context_from_real_config,
    grid_timestamps,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG_PATH = REPO_ROOT / "configs" / "real_m2.yaml"

_SEED = 0
_BATCH = 1
_SEQ = 8
_FEAT = 6
_K = 2
_SOLVER_STEPS = 4
_NATIVE_GAP_T = 2
_ART_GAP_T = 5
_BLOCK_LEN = 2
_MASK_RATE = 0.0
_ATOL = 1e-5
_WEIGHT_TOL = 1e-6


@pytest.fixture
def real_cfg():
    """Load real_m2 YAML (no processed NPZ required)."""
    return load_real_m2_config(REAL_CONFIG_PATH)


def _windows_with_native_gap() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Tiny synthetic windows: one native gap time, rest observed."""
    rng = np.random.default_rng(_SEED)
    values = rng.normal(size=(_BATCH, _SEQ, _FEAT)).astype(np.float32)
    masks = np.ones((_BATCH, _SEQ, _FEAT), dtype=np.float32)
    masks[:, _NATIVE_GAP_T, :] = MASK_MISSING
    values[:, _NATIVE_GAP_T, :] = 0.0
    return jnp.asarray(values), jnp.asarray(masks)


def test_metric_observation_mask_selects_artificial_targets() -> None:
    """Metric mask is 0 only on natively observed ∧ artificially missing."""
    values, m_native = _windows_with_native_gap()
    composer = MaskComposer(rate=_MASK_RATE, block_len=_BLOCK_LEN)
    _x_obs, _m_train, m_art = composer.compose_block(
        jax.random.PRNGKey(1),
        values,
        m_native,
    )
    metric_mask = metric_observation_mask(m_native, m_art)
    targets = artificial_target_mask(m_native, m_art)
    assert bool(
        jnp.all(
            jnp.abs(metric_mask - (1.0 - targets)) < _WEIGHT_TOL
        )
    )
    native_missing = jnp.abs(m_native - MASK_MISSING) < _WEIGHT_TOL
    assert bool(
        jnp.all(metric_mask[native_missing] == MASK_OBSERVED)
    )


def test_evaluator_skips_native_gaps() -> None:
    """Perfect fill on artificial targets → zero metrics despite native junk."""
    truth = jnp.zeros((_BATCH, _SEQ, _FEAT), dtype=jnp.float32)
    truth = truth.at[:, _ART_GAP_T, :].set(3.0)
    m_native = jnp.ones((_BATCH, _SEQ, _FEAT), dtype=jnp.float32)
    m_native = m_native.at[:, _NATIVE_GAP_T, :].set(MASK_MISSING)
    m_art = jnp.ones((_BATCH, _SEQ, _FEAT), dtype=jnp.float32)
    m_art = m_art.at[:, _ART_GAP_T, :].set(MASK_MISSING)
    mean = truth
    mean = mean.at[:, _NATIVE_GAP_T, :].set(999.0)
    samples = jnp.broadcast_to(
        mean[:, None, :, :],
        (_BATCH, _K, _SEQ, _FEAT),
    )
    metric_mask = metric_observation_mask(m_native, m_art)
    table = ImputationEvaluator().compute(
        EvalTensors(
            truth=truth,
            mean=mean,
            samples=samples,
            mask=metric_mask,
        )
    )
    assert table.mae == 0.0
    assert table.rmse == 0.0
    assert table.crps < 1e-6


def test_original_scale_restorer_roundtrip() -> None:
    """Inverse-transform restores observed cells after standardize."""
    values, m_native = _windows_with_native_gap()
    scaler = FeatureScaler().fit(values, m_native)
    normalized = scaler.transform(values, m_native)
    restorer = OriginalScaleRestorer(scaler)
    restored = restorer.restore_windows(normalized, m_native)
    observed = jnp.abs(m_native - MASK_OBSERVED) < _WEIGHT_TOL
    assert bool(
        jnp.allclose(restored[observed], values[observed], atol=_ATOL)
    )
    samples_n = jnp.broadcast_to(
        normalized[:, None, :, :],
        (_BATCH, _K, _SEQ, _FEAT),
    )
    samples = restorer.restore_samples(samples_n, m_native)
    assert samples.shape == (_BATCH, _K, _SEQ, _FEAT)
    assert bool(
        jnp.allclose(samples[:, 0][observed], values[observed], atol=_ATOL)
    )


def test_context_from_real_config_smoke(real_cfg) -> None:
    """Factory wires sample.solver_steps and train.protocol.t_eps."""
    real_cfg.sample.k_samples = _K
    real_cfg.sample.solver_steps = _SOLVER_STEPS
    model = ConditionalScoreNet(
        jax.random.PRNGKey(_SEED),
        default_score_net_config(real_cfg.data.n_features),
    )
    context = context_from_real_config(model, real_cfg)
    assert context.settings.solver_steps == _SOLVER_STEPS
    assert context.settings.t_eps == real_cfg.train.protocol.t_eps


def test_tiny_impute_shapes_with_real_context(real_cfg) -> None:
    """Tiny K/steps impute honors (B, K, T, F); observed cells frozen."""
    real_cfg.sample.k_samples = _K
    real_cfg.sample.solver_steps = _SOLVER_STEPS
    values, m_native = _windows_with_native_gap()
    composer = MaskComposer(rate=0.3, block_len=_BLOCK_LEN)
    x_obs, m_train, _m_art = composer.compose_block(
        jax.random.PRNGKey(2),
        values,
        m_native,
    )
    model = ConditionalScoreNet(
        jax.random.PRNGKey(_SEED),
        default_score_net_config(_FEAT),
    )
    imputer = ReverseSDEImputer(context_from_real_config(model, real_cfg))
    samples, mean, quantiles = imputer.impute(
        ImputeQuery(
            observations=x_obs,
            mask=m_train,
            timestamps=grid_timestamps(_SEQ),
            key=jax.random.PRNGKey(3),
            k_samples=_K,
        )
    )
    assert samples.shape == (_BATCH, _K, _SEQ, _FEAT)
    assert mean.shape == (_BATCH, _SEQ, _FEAT)
    assert quantiles.shape == (_BATCH, _SEQ, _FEAT, 2)
    observed = jnp.abs(m_train - MASK_OBSERVED) < _WEIGHT_TOL
    assert bool(jnp.allclose(mean[observed], x_obs[observed], atol=_ATOL))


def test_metrics_artifact_writer_roundtrip(tmp_path: Path) -> None:
    """JSON/CSV dumps contain the expected metric keys."""
    table = ImputationEvaluator().compute(
        EvalTensors(
            truth=jnp.zeros((_BATCH, 2, 1), dtype=jnp.float32),
            mean=jnp.zeros((_BATCH, 2, 1), dtype=jnp.float32),
            samples=jnp.zeros((_BATCH, _K, 2, 1), dtype=jnp.float32),
            mask=jnp.array(
                [[[MASK_MISSING], [MASK_OBSERVED]]],
                dtype=jnp.float32,
            ),
        )
    )
    writer = MetricsArtifactWriter()
    json_path = writer.write_json(tmp_path / "m.json", table)
    csv_path = writer.write_csv(tmp_path / "m.csv", table)
    assert "mae" in json_path.read_text(encoding="utf-8")
    assert "mae" in csv_path.read_text(encoding="utf-8")
