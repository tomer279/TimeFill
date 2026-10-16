"""Hermetic tests for M6 preview summary helpers and impute gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.flexible_ingest import ingest_flexible_csv
from app.preview_ingest import (
    MODE_EXACT_BEIJING,
    MODE_FLEXIBLE,
    FlexiblePreviewKnobs,
    PreviewBuildInputs,
    build_ingest_preview,
    evaluate_impute_gate,
    features_match_checkpoint,
    read_csv_headers,
    select_ingest_mode,
    try_adapt_flexible_windows,
)
from app.schema_validate import ValidatedFrame, default_app_schema
from app.window_adapter import WindowAdaptRequest, adapt_validated_frame
from src.data.real.specs import default_beijing_features

_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "toy_non_beijing.csv"
)
_BEIJING = default_beijing_features()
_FLEX = ("temp_c", "humidity_pct", "pressure_hpa")
_SEQ_LEN = 64
_MIN_ROWS = 16
_ACCEPT_ROWS = 20
_SEQ_LEN_SMALL = 8
_MIN_ROWS_SMALL = 2
_TEMP_MISSING_PERCENT = 25.0
_FIRST_FEATURE_IDX = 0
_SECOND_FEATURE_IDX = 1
_REMAINING_FEATURES_START = 2


def _dense_frame(n_rows: int, names: tuple[str, ...]) -> ValidatedFrame:
    """Build a fully observed hermetic frame.

    Parameters
    ----------
    n_rows : int
        Grid length ``L``.
    names : tuple of str
        Feature names.

    Returns
    -------
    ValidatedFrame
        Dense float32 values / masks.
    """
    n_features = len(names)
    values = np.ones((n_rows, n_features), dtype=np.float32)
    mask = np.ones((n_rows, n_features), dtype=np.float32)
    return ValidatedFrame(
        values=values,
        native_mask=mask,
        feature_names=names,
        timestamps=None,
        entity_id="lab-A",
    )


def test_select_mode_exact_when_beijing_headers() -> None:
    """Headers containing Beijing names select Exact mode."""
    headers = ("timestamp",) + _BEIJING + ("site",)
    assert select_ingest_mode(headers, default_app_schema()) == (
        MODE_EXACT_BEIJING
    )


def test_select_mode_flexible_for_non_beijing_fixture() -> None:
    """Non-Beijing fixture headers select Flexible mode."""
    headers = read_csv_headers(_FIXTURE)
    assert select_ingest_mode(headers, default_app_schema()) == MODE_FLEXIBLE
    assert "PM2.5" not in headers


def test_features_match_checkpoint_predicate() -> None:
    """Exact tuple equality is required (order and count)."""
    assert features_match_checkpoint(_BEIJING, _BEIJING)
    assert not features_match_checkpoint(_FLEX, _BEIJING)
    swapped = (
        (_BEIJING[_SECOND_FEATURE_IDX], _BEIJING[_FIRST_FEATURE_IDX])
        + _BEIJING[_REMAINING_FEATURES_START:]
    )
    assert not features_match_checkpoint(swapped, _BEIJING)


def test_impute_gate_allows_exact_mode() -> None:
    """Exact mode is allowed even before comparing flexible maps."""
    gate = evaluate_impute_gate(_BEIJING, _BEIJING, MODE_EXACT_BEIJING)
    assert gate.is_allowed()
    assert gate.block_message() == ""


def test_impute_gate_blocks_flexible_mismatch() -> None:
    """Flexible non-Beijing features are blocked with M7 / checkpoint copy."""
    gate = evaluate_impute_gate(_FLEX, _BEIJING, MODE_FLEXIBLE)
    assert not gate.is_allowed()
    message = gate.block_message().lower()
    assert "milestone 7" in message or "m7" in message
    assert "compatible checkpoint" in message
    assert "arbitrary csvs cannot be imputed" in message
    assert "any csv works" not in message


def test_impute_gate_allows_flexible_exact_match() -> None:
    """Flexible path may impute only when names+F match the checkpoint."""
    gate = evaluate_impute_gate(_BEIJING, _BEIJING, MODE_FLEXIBLE)
    assert gate.is_allowed()


def test_preview_summary_for_non_beijing_fixture() -> None:
    """Fixture preview exposes schema + missingness; adapt may reject."""
    frame = ingest_flexible_csv(_FIXTURE)
    knobs = FlexiblePreviewKnobs(
        seq_len=_SEQ_LEN,
        max_windows=4,
    )
    windows, adapt_error = try_adapt_flexible_windows(frame, knobs)
    assert windows is None
    assert adapt_error is not None
    assert "too short" in adapt_error
    preview = build_ingest_preview(
        PreviewBuildInputs(
            frame=frame,
            ingest_mode=MODE_FLEXIBLE,
            windows=windows,
            adapt_error=adapt_error,
        )
    )
    assert preview.feature_names == _FLEX
    assert preview.feature_count() == len(_FLEX)
    assert preview.n_rows == frame.n_rows()
    assert preview.adapt.n_windows is None
    assert preview.per_feature_missing_percent["temp_c"] == pytest.approx(
        _TEMP_MISSING_PERCENT
    )
    text = "\n".join(preview.as_text_lines())
    assert "temp_c" in text
    assert "PM2.5" not in text


def test_preview_summary_with_successful_windows() -> None:
    """Accepted flexible series reports window count after adapt."""
    frame = _dense_frame(_ACCEPT_ROWS, _FLEX)
    knobs = FlexiblePreviewKnobs(
        seq_len=_SEQ_LEN_SMALL,
        max_windows=None,
    )
    assert knobs.reject_policy().min_rows == _MIN_ROWS_SMALL
    windows, adapt_error = try_adapt_flexible_windows(frame, knobs)
    assert adapt_error is None
    assert windows is not None
    preview = build_ingest_preview(
        PreviewBuildInputs(
            frame=frame,
            ingest_mode=MODE_FLEXIBLE,
            windows=windows,
            adapt_error=None,
        )
    )
    expected_b = _ACCEPT_ROWS // _SEQ_LEN_SMALL + (
        1 if _ACCEPT_ROWS % _SEQ_LEN_SMALL else 0
    )
    assert preview.adapt.n_windows == expected_b
    assert "adapted" in (preview.adapt.adapt_message or "")


def test_exact_path_preview_still_adapts() -> None:
    """Exact Beijing-shaped frame still adapts via shared pad path."""
    frame = _dense_frame(_MIN_ROWS, _BEIJING)
    windows = adapt_validated_frame(
        frame,
        WindowAdaptRequest(seq_len=_SEQ_LEN, max_windows=1),
    )
    preview = build_ingest_preview(
        PreviewBuildInputs(
            frame=frame,
            ingest_mode=MODE_EXACT_BEIJING,
            windows=windows,
            adapt_error=None,
        )
    )
    assert preview.adapt.n_windows == 1
    gate = evaluate_impute_gate(
        frame.feature_names,
        _BEIJING,
        MODE_EXACT_BEIJING,
    )
    assert gate.is_allowed()
