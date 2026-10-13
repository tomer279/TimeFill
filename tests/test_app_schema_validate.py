"""Hermetic tests for M4 CSV schema validation (no Streamlit / network)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.schema_validate import (
    default_app_schema,
    summarize_missingness,
    validate_upload_csv,
    validate_upload_rows,
)
from src.data.schema import MASK_MISSING, MASK_OBSERVED, MISSING_VALUE_SENTINEL

_FEATURES = (
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
)
_REPO_SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "sample_data"
    / "beijing_demo_slice.csv"
)
_MASK_TOL = 1e-6
_GAP_ROW = 1
_GAP_FEATURE = "PM10"
_GAP_FEATURE_IDX = 1
_N_DEMO_ROWS = 2


def _write_demo_csv(path: Path, include_no2: bool = True) -> None:
    """Write a tiny valid (or column-dropped) Beijing-schema CSV.

    Parameters
    ----------
    path : pathlib.Path
        Output CSV path.
    include_no2 : bool, optional
        If ``False``, omit the ``NO2`` column to trigger schema failure.
    """
    fieldnames = ["timestamp", "entity", *_FEATURES]
    rows = [
        {
            "timestamp": "2013-03-01T00:00:00",
            "entity": "DemoSite",
            "PM2.5": "10.0",
            "PM10": "20.0",
            "SO2": "3.0",
            "NO2": "15.0",
            "CO": "300.0",
            "O3": "70.0",
        },
        {
            "timestamp": "2013-03-01T01:00:00",
            "entity": "DemoSite",
            "PM2.5": "11.0",
            "PM10": "",
            "SO2": "4.0",
            "NO2": "16.0",
            "CO": "310.0",
            "O3": "72.0",
        },
    ]
    if not include_no2:
        fieldnames = [name for name in fieldnames if name != "NO2"]
        for row in rows:
            row.pop("NO2", None)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_good_csv_passes(tmp_path: Path) -> None:
    """A well-formed Beijing-schema CSV validates successfully."""
    path = tmp_path / "good.csv"
    _write_demo_csv(path)
    frame = validate_upload_csv(path, default_app_schema())
    assert frame.n_rows() == _N_DEMO_ROWS
    assert frame.n_features() == len(_FEATURES)
    assert frame.feature_names == _FEATURES
    assert frame.entity_id == "DemoSite"


def test_bundled_sample_validates() -> None:
    """The repo-bundled demo CSV matches the locked M4 schema."""
    assert _REPO_SAMPLE.is_file(), f"missing bundled sample: {_REPO_SAMPLE}"
    frame = validate_upload_csv(_REPO_SAMPLE, default_app_schema())
    assert frame.n_features() == len(_FEATURES)
    assert frame.n_rows() > 0
    assert frame.feature_names == _FEATURES


def test_missing_column_fails_with_actionable_message(tmp_path: Path) -> None:
    """Missing required feature columns raise a clear error (no remaps)."""
    path = tmp_path / "missing_col.csv"
    _write_demo_csv(path, include_no2=False)
    with pytest.raises(ValueError, match="NO2"):
        validate_upload_csv(path, default_app_schema())


def test_non_numeric_fails_with_column_and_row(tmp_path: Path) -> None:
    """Non-numeric feature cells name the column and data row."""
    path = tmp_path / "bad_numeric.csv"
    _write_demo_csv(path)
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["CO"] = "not-a-number"
    with pytest.raises(ValueError, match="CO"):
        validate_upload_rows(rows, default_app_schema())


def test_nan_becomes_mask_and_sentinel(tmp_path: Path) -> None:
    """Empty cells become ``m == 0`` and sentinel ``0.0`` in ``x``."""
    path = tmp_path / "nan.csv"
    _write_demo_csv(path)
    frame = validate_upload_csv(path, default_app_schema())
    assert frame.native_mask[_GAP_ROW, _GAP_FEATURE_IDX] == MASK_MISSING
    assert frame.values[_GAP_ROW, _GAP_FEATURE_IDX] == MISSING_VALUE_SENTINEL
    assert frame.native_mask[0, 0] == MASK_OBSERVED
    assert abs(
        frame.native_mask[_GAP_ROW, _GAP_FEATURE_IDX] - MASK_MISSING
    ) < _MASK_TOL
    summary = summarize_missingness(frame)
    fractions = summary.per_feature_fraction()
    assert fractions[_GAP_FEATURE] == pytest.approx(0.5)
    assert fractions["PM2.5"] == pytest.approx(0.0)
