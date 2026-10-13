"""Config + optional e2e smoke for M4 session impute (skip without ckpt)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.m4_config import load_app_m4_config
from app.session_impute import impute_bundled_sample
from src.data.schema import MASK_OBSERVED

_REPO = Path(__file__).resolve().parents[1]
_APP_CFG = _REPO / "configs" / "app_m4.yaml"
_CKPT = _REPO / "artifacts" / "m2" / "checkpoints" / "best_score_net.eqx"
_MASK_TOL = 1e-6
_ATOL = 1e-5


def test_load_app_m4_config_paths() -> None:
    """``app_m4.yaml`` resolves checkpoint / scaler / sample under the repo."""
    config = load_app_m4_config(_APP_CFG)
    assert config.schema.n_features == 6
    assert config.sample.k_samples == 8
    assert config.paths.sample_csv.is_file()
    assert config.paths.real_config.is_file()


@pytest.mark.skipif(not _CKPT.is_file(), reason="M2 checkpoint absent")
def test_impute_bundled_sample_preserves_observed() -> None:
    """When artifacts exist, observed cells survive mean-fill merge."""
    result = impute_bundled_sample(_APP_CFG)
    observed = np.abs(result.context.m_eval - MASK_OBSERVED) < _MASK_TOL
    missing = ~observed
    assert result.n_windows() >= 1
    assert result.context.mean_fill.shape == result.tensors.mean.shape
    assert np.allclose(
        result.context.mean_fill[observed],
        result.context.observations[observed],
        rtol=0.0,
        atol=_ATOL,
    )
    assert np.allclose(
        result.context.mean_fill[missing],
        result.tensors.mean[missing],
        rtol=0.0,
        atol=_ATOL,
    )
