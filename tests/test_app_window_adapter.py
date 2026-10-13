"""Hermetic tests for M4 CSV → ``(x, m)`` window adaptation."""

from __future__ import annotations

import numpy as np

from app.schema_validate import ValidatedFrame, default_app_schema
from app.window_adapter import WindowAdaptRequest, adapt_validated_frame
from src.data.schema import (
    FEATURE_AXIS,
    MASK_MISSING,
    MASK_OBSERVED,
    MISSING_VALUE_SENTINEL,
    TIME_AXIS,
)

_FEATURES = default_app_schema().feature_names
_N_FEATURES = len(_FEATURES)
_SEQ_LEN = 4
_GAP_TIME = 1
_GAP_FEAT = 0
_REMAINDER_REAL_ROWS = 2
_N_FULL_WINDOWS_UNCAPPED = 3


def _frame_from_matrix(
        values: np.ndarray,
        native_mask: np.ndarray,
) -> ValidatedFrame:
    """Build a ``ValidatedFrame`` around pre-made arrays.

    Parameters
    ----------
    values : numpy.ndarray
        Float32 ``(L, F)`` values.
    native_mask : numpy.ndarray
        Float32 ``(L, F)`` masks.

    Returns
    -------
    ValidatedFrame
        Frame with frozen Beijing feature names.
    """
    return ValidatedFrame(
        values=values.astype(np.float32),
        native_mask=native_mask.astype(np.float32),
        feature_names=_FEATURES,
        timestamps=None,
        entity_id="DemoSite",
    )


def test_nan_contract_preserved_in_windows() -> None:
    """Native gaps stay sentinel / ``m == 0``; short series is right-padded."""
    n_rows = 3
    values = np.ones((n_rows, _N_FEATURES), dtype=np.float32)
    mask = np.ones((n_rows, _N_FEATURES), dtype=np.float32)
    values[_GAP_TIME, _GAP_FEAT] = MISSING_VALUE_SENTINEL
    mask[_GAP_TIME, _GAP_FEAT] = MASK_MISSING
    adapted = adapt_validated_frame(
        _frame_from_matrix(values, mask),
        WindowAdaptRequest(seq_len=_SEQ_LEN, max_windows=None),
    )
    assert adapted.n_windows() == 1
    assert adapted.values.shape == (1, _SEQ_LEN, _N_FEATURES)
    assert adapted.values[0, _GAP_TIME, _GAP_FEAT] == MISSING_VALUE_SENTINEL
    assert adapted.native_mask[0, _GAP_TIME, _GAP_FEAT] == MASK_MISSING
    assert np.all(adapted.native_mask[0, n_rows:, :] == MASK_MISSING)
    assert np.all(adapted.values[0, n_rows:, :] == MISSING_VALUE_SENTINEL)


def test_long_series_emits_padded_remainder() -> None:
    """Longer-than-``T`` series yields non-overlapping windows + padded tail."""
    n_rows = _SEQ_LEN + _REMAINDER_REAL_ROWS
    values = np.arange(n_rows * _N_FEATURES, dtype=np.float32).reshape(
        n_rows,
        _N_FEATURES,
    )
    mask = np.ones((n_rows, _N_FEATURES), dtype=np.float32)
    adapted = adapt_validated_frame(
        _frame_from_matrix(values, mask),
        WindowAdaptRequest(seq_len=_SEQ_LEN, max_windows=None),
    )
    assert adapted.n_windows() == 2
    assert adapted.values.shape[TIME_AXIS] == _SEQ_LEN
    assert adapted.values.shape[FEATURE_AXIS] == _N_FEATURES
    assert np.all(
        adapted.native_mask[1, :_REMAINDER_REAL_ROWS, :] == MASK_OBSERVED
    )
    assert np.all(
        adapted.native_mask[1, _REMAINDER_REAL_ROWS:, :] == MASK_MISSING
    )


def test_max_windows_cap() -> None:
    """``max_windows`` keeps only the leading windows."""
    n_rows = _SEQ_LEN * _N_FULL_WINDOWS_UNCAPPED
    values = np.ones((n_rows, _N_FEATURES), dtype=np.float32)
    mask = np.ones((n_rows, _N_FEATURES), dtype=np.float32)
    adapted = adapt_validated_frame(
        _frame_from_matrix(values, mask),
        WindowAdaptRequest(seq_len=_SEQ_LEN, max_windows=1),
    )
    assert adapted.n_windows() == 1
    batch = adapted.as_batch()
    assert batch.shape() == (1, _SEQ_LEN, _N_FEATURES)
