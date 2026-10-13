"""Hermetic tests for M4 export frame helpers (no Streamlit / SDE)."""

from __future__ import annotations

import numpy as np
import pytest

from app.export_frames import (
    MeanFillExportRequest,
    SampleExportRequest,
    dataframe_to_csv_bytes,
    mean_fill_dataframe,
    samples_long_dataframe,
)

_BATCH = 1
_SEQ = 2
_FEAT = 2
_K = 2
_FEATURE_NAMES = ("PM2.5", "PM10")
_N_LONG_ROWS = _BATCH * _K * _SEQ * _FEAT


def test_mean_fill_dataframe_shape_and_values() -> None:
    """Wide mean-fill has one row per (window, t) and preserves cells."""
    mean_fill = np.arange(_BATCH * _SEQ * _FEAT, dtype=np.float32).reshape(
        _BATCH,
        _SEQ,
        _FEAT,
    )
    frame = mean_fill_dataframe(
        MeanFillExportRequest(
            mean_fill=mean_fill,
            feature_names=_FEATURE_NAMES,
        )
    )
    assert list(frame.columns) == [
        "window_id",
        "t_index",
        "PM2.5",
        "PM10",
    ]
    assert len(frame) == _BATCH * _SEQ
    assert frame.loc[0, "PM2.5"] == pytest.approx(0.0)
    assert frame.loc[1, "PM10"] == pytest.approx(3.0)


def test_samples_long_dataframe_row_count_and_columns() -> None:
    """Long sample export has ``B * K * T * F`` rows in locked column order."""
    samples = np.arange(
        _BATCH * _K * _SEQ * _FEAT,
        dtype=np.float32,
    ).reshape(_BATCH, _K, _SEQ, _FEAT)
    frame = samples_long_dataframe(
        SampleExportRequest(
            samples=samples,
            feature_names=_FEATURE_NAMES,
        )
    )
    assert list(frame.columns) == [
        "sample_id",
        "window_id",
        "t_index",
        "feature",
        "value",
    ]
    assert len(frame) == _N_LONG_ROWS
    assert set(frame["feature"]) == set(_FEATURE_NAMES)


def test_dataframe_to_csv_bytes_contains_header() -> None:
    """CSV bytes include the mean-fill header row."""
    mean_fill = np.zeros((_BATCH, _SEQ, _FEAT), dtype=np.float32)
    frame = mean_fill_dataframe(
        MeanFillExportRequest(
            mean_fill=mean_fill,
            feature_names=_FEATURE_NAMES,
        )
    )
    payload = dataframe_to_csv_bytes(frame).decode("utf-8")
    assert "window_id,t_index,PM2.5,PM10" in payload.splitlines()[0]


def test_mean_fill_rejects_feature_name_mismatch() -> None:
    """Validate rejects ``feature_names`` length ≠ ``F``."""
    mean_fill = np.zeros((_BATCH, _SEQ, _FEAT), dtype=np.float32)
    with pytest.raises(ValueError, match="feature_names"):
        mean_fill_dataframe(
            MeanFillExportRequest(
                mean_fill=mean_fill,
                feature_names=("PM2.5",),
            )
        )
