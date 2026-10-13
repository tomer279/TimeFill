"""Hermetic tests for M4 mean-fill merge (no SDE / GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from app.session_impute import merge_mean_fill
from src.data.schema import MASK_MISSING, MASK_OBSERVED, MISSING_VALUE_SENTINEL

_BATCH = 1
_SEQ = 2
_FEAT = 2
_OBS_VALUE = 3.5
_MEAN_FILL_VALUE = 9.0
_OTHER_MEAN = 1.25


def test_merge_preserves_observed_cells() -> None:
    """``m == 1`` cells keep the observed value, not the posterior mean."""
    observed = np.full((_BATCH, _SEQ, _FEAT), _OBS_VALUE, dtype=np.float32)
    mean = np.full((_BATCH, _SEQ, _FEAT), _MEAN_FILL_VALUE, dtype=np.float32)
    mask = np.ones((_BATCH, _SEQ, _FEAT), dtype=np.float32)
    mask[0, 0, 0] = MASK_MISSING
    observed[0, 0, 0] = MISSING_VALUE_SENTINEL
    filled = merge_mean_fill(observed, mean, mask)
    assert filled[0, 0, 0] == pytest.approx(_MEAN_FILL_VALUE)
    assert filled[0, 0, 1] == pytest.approx(_OBS_VALUE)
    assert filled[0, 1, 0] == pytest.approx(_OBS_VALUE)


def test_merge_fills_only_missing_targets() -> None:
    """Only ``m == 0`` cells are replaced."""
    observed = np.array(
        [[[1.0, 2.0], [3.0, 4.0]]],
        dtype=np.float32,
    )
    mean = np.full((_BATCH, _SEQ, _FEAT), _OTHER_MEAN, dtype=np.float32)
    mask = np.array(
        [[[MASK_OBSERVED, MASK_MISSING], [MASK_MISSING, MASK_OBSERVED]]],
        dtype=np.float32,
    )
    filled = merge_mean_fill(observed, mean, mask)
    assert filled[0, 0, 0] == pytest.approx(1.0)
    assert filled[0, 0, 1] == pytest.approx(_OTHER_MEAN)
    assert filled[0, 1, 0] == pytest.approx(_OTHER_MEAN)
    assert filled[0, 1, 1] == pytest.approx(4.0)
