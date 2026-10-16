"""Hermetic tests for M6 flexible CSV ingest (map / infer / masks / window)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from app.flexible_ingest import (
    ColumnRoleMap,
    IngestInferConfig,
    column_role_map_from_mapping,
    infer_column_roles,
    ingest_flexible_csv,
    ingest_flexible_rows,
)
from app.schema_validate import ValidatedFrame
from app.window_adapter import (
    FlexibleAdaptRequest,
    IngestRejectPolicy,
    WindowAdaptRequest,
    adapt_flexible_frame,
    adapt_validated_frame,
    bin_validated_frame_hourly,
    default_ingest_reject_policy,
)
from src.data.real.specs import default_beijing_features
from src.data.schema import (
    BATCH_AXIS,
    FEATURE_AXIS,
    MASK_MISSING,
    MASK_OBSERVED,
    MISSING_VALUE_SENTINEL,
    TIME_AXIS,
)

_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "toy_non_beijing.csv"
)
_FEATURE_NAMES = ("temp_c", "humidity_pct", "pressure_hpa")
_N_FIXTURE_ROWS = 4
_TEMP_IDX = 0
_HUMIDITY_IDX = 1
_PRESSURE_IDX = 2
_GAP_TEMP_ROW = 1
_GAP_HUMIDITY_ROW = 2
_GAP_PRESSURE_ROW = 3
_BEIJING_NAMES = frozenset(default_beijing_features())
_TINY_MAX_FEATURES = 2
_FLEX_FEATURES = ("temp_c", "humidity_pct")
_TEMP_F = 0
_HUM_F = 1
_SEQ_LEN_SMALL = 8
_MIN_ROWS_SMALL = max(1, _SEQ_LEN_SMALL // 4)
_ACCEPT_ROWS = 20
_PAD_ROWS = 5
_SHORT_ROWS = 1
_SPARSE_ROWS = 20
_DEFAULT_SEQ_LEN = 64
_DEFAULT_MIN_ROWS = 16
_SPARSE_OBSERVED_ROWS = 1


def _hermetic_toy_rows() -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """Return headers and rows matching the non-Beijing fixture schema.

    Returns
    -------
    headers : tuple of str
        Column names.
    rows : list of dict
        String cells.
    """
    headers = (
        "time",
        "sensor_id",
        "temp_c",
        "humidity_pct",
        "pressure_hpa",
    )
    rows = [
        {
            "time": "2020-01-01T00:00:00",
            "sensor_id": "lab-A",
            "temp_c": "21.5",
            "humidity_pct": "45.0",
            "pressure_hpa": "1013.2",
        },
        {
            "time": "2020-01-01T01:00:00",
            "sensor_id": "lab-A",
            "temp_c": "",
            "humidity_pct": "46.0",
            "pressure_hpa": "1012.8",
        },
        {
            "time": "2020-01-01T02:00:00",
            "sensor_id": "lab-A",
            "temp_c": "22.0",
            "humidity_pct": "nan",
            "pressure_hpa": "1012.5",
        },
        {
            "time": "2020-01-01T03:00:00",
            "sensor_id": "lab-A",
            "temp_c": "22.5",
            "humidity_pct": "47.0",
            "pressure_hpa": "",
        },
    ]
    return headers, rows


def test_fixture_infers_non_beijing_frame() -> None:
    """Non-Beijing fixture validates with native masks and local names."""
    assert _FIXTURE.is_file(), f"missing fixture: {_FIXTURE}"
    frame = ingest_flexible_csv(_FIXTURE)
    assert frame.n_rows() == _N_FIXTURE_ROWS
    assert frame.feature_names == _FEATURE_NAMES
    assert _BEIJING_NAMES.isdisjoint(frame.feature_names)
    assert frame.entity_id == "lab-A"
    assert frame.timestamps is not None
    assert frame.native_mask[_GAP_TEMP_ROW, _TEMP_IDX] == MASK_MISSING
    assert frame.values[_GAP_TEMP_ROW, _TEMP_IDX] == MISSING_VALUE_SENTINEL
    assert frame.native_mask[_GAP_HUMIDITY_ROW, _HUMIDITY_IDX] == MASK_MISSING
    assert (
            frame.values[_GAP_HUMIDITY_ROW, _HUMIDITY_IDX]
            == MISSING_VALUE_SENTINEL
    )
    assert frame.native_mask[_GAP_PRESSURE_ROW, _PRESSURE_IDX] == MASK_MISSING
    assert frame.native_mask[0, _TEMP_IDX] == MASK_OBSERVED
    assert frame.values[0, _TEMP_IDX] == pytest.approx(21.5)


def test_explicit_mapping_yaml_ready() -> None:
    """Explicit map dict selects roles without renaming to Beijing."""
    headers, rows = _hermetic_toy_rows()
    role_map = column_role_map_from_mapping(
        {
            "features": ["temp_c", "pressure_hpa"],
            "timestamp": "time",
            "entity": "sensor_id",
        }
    )
    frame = ingest_flexible_rows(rows, headers, role_map=role_map)
    assert frame.feature_names == ("temp_c", "pressure_hpa")
    assert _BEIJING_NAMES.isdisjoint(frame.feature_names)
    assert frame.entity_id == "lab-A"


def test_infer_roles_hermetic() -> None:
    """Hermetic rows infer time / sensor_id / numeric features."""
    headers, rows = _hermetic_toy_rows()
    role_map = infer_column_roles(headers, rows, IngestInferConfig())
    assert role_map.timestamp == "time"
    assert role_map.entity == "sensor_id"
    assert role_map.features == _FEATURE_NAMES


def test_empty_file_errors(tmp_path: Path) -> None:
    """Empty CSV files raise a clear user-facing error."""
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        ingest_flexible_csv(path)


def test_zero_features_errors() -> None:
    """All-text columns yield a zero-feature reject."""
    headers = ("label", "note")
    rows = [
        {"label": "a", "note": "hello"},
        {"label": "b", "note": "world"},
    ]
    with pytest.raises(ValueError, match="zero feature"):
        ingest_flexible_rows(rows, headers)


def test_duplicate_feature_map_errors() -> None:
    """Duplicate feature names in the map are rejected."""
    with pytest.raises(ValueError, match="duplicate feature"):
        ColumnRoleMap(features=("temp_c", "temp_c")).validate()


def test_overlapping_role_map_errors() -> None:
    """A column cannot be both a feature and timestamp."""
    with pytest.raises(ValueError, match="duplicate map"):
        ColumnRoleMap(
            features=("temp_c",),
            timestamp="temp_c",
        ).validate()


def test_non_numeric_feature_cell_errors() -> None:
    """Non-numeric cells in mapped feature columns name column and row."""
    headers, rows = _hermetic_toy_rows()
    rows[0]["temp_c"] = "not-a-number"
    role_map = ColumnRoleMap(
        features=_FEATURE_NAMES,
        timestamp="time",
        entity="sensor_id",
    )
    with pytest.raises(ValueError, match="temp_c"):
        ingest_flexible_rows(rows, headers, role_map=role_map)


def test_max_features_cap_errors() -> None:
    """F above max_features fails clearly (no silent truncate)."""
    headers, rows = _hermetic_toy_rows()
    config = IngestInferConfig(max_features=_TINY_MAX_FEATURES)
    with pytest.raises(ValueError, match="max_features"):
        ingest_flexible_rows(rows, headers, config=config)


def test_missing_mapped_column_errors() -> None:
    """Explicit maps that reference absent headers are rejected."""
    headers, rows = _hermetic_toy_rows()
    role_map = ColumnRoleMap(features=("temp_c", "missing_col"))
    with pytest.raises(ValueError, match="missing_col"):
        ingest_flexible_rows(rows, headers, role_map=role_map)


def test_no_silent_beijing_remap_on_unknown_headers(tmp_path: Path) -> None:
    """Unknown numeric headers keep their names (never PM2.5…O3)."""
    path = tmp_path / "custom.csv"
    fieldnames = ["ts", "x1", "x2"]
    rows = [
        {"ts": "2021-01-01T00:00:00", "x1": "1.0", "x2": "2.0"},
        {"ts": "2021-01-01T01:00:00", "x1": "", "x2": "3.0"},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    frame = ingest_flexible_csv(path)
    assert frame.feature_names == ("x1", "x2")
    assert _BEIJING_NAMES.isdisjoint(frame.feature_names)
    assert "PM2.5" not in frame.feature_names


def _non_beijing_frame(n_rows: int) -> ValidatedFrame:
    """Build a hermetic non-Beijing ``ValidatedFrame`` (row-order grid).

    Parameters
    ----------
    n_rows : int
        Grid length ``L``.

    Returns
    -------
    ValidatedFrame
        Float32 values / masks with local feature names (fully observed).
    """
    n_features = len(_FLEX_FEATURES)
    values = np.ones((n_rows, n_features), dtype=np.float32)
    mask = np.ones((n_rows, n_features), dtype=np.float32)
    return ValidatedFrame(
        values=values,
        native_mask=mask,
        feature_names=_FLEX_FEATURES,
        timestamps=None,
        entity_id="lab-A",
    )


def _flexible_request(
        seq_len: int,
        min_rows: int,
        max_windows: int | None = None,
        apply_hourly_bin: bool = False,
) -> FlexibleAdaptRequest:
    """Build a flexible adapt request with explicit reject floors.

    Parameters
    ----------
    seq_len : int
        Window length ``T``.
    min_rows : int
        Reject floor on ``L``.
    max_windows : int or None, optional
        Optional cap.
    apply_hourly_bin : bool, optional
        Whether to hourly-bin when timestamps exist.

    Returns
    -------
    FlexibleAdaptRequest
        Test knobs.
    """
    return FlexibleAdaptRequest(
        window=WindowAdaptRequest(
            seq_len=seq_len,
            max_windows=max_windows,
        ),
        reject=IngestRejectPolicy(min_rows=min_rows),
        apply_hourly_bin=apply_hourly_bin,
    )


def test_flexible_accept_windows_shape() -> None:
    """Valid non-Beijing frame → ``(B, T, F)`` windows + masks."""
    frame = _non_beijing_frame(_ACCEPT_ROWS)
    adapted = adapt_flexible_frame(
        frame,
        _flexible_request(_SEQ_LEN_SMALL, _MIN_ROWS_SMALL),
    )
    n_full = _ACCEPT_ROWS // _SEQ_LEN_SMALL
    n_rem = _ACCEPT_ROWS % _SEQ_LEN_SMALL
    expected_b = n_full + (1 if n_rem else 0)
    assert adapted.n_windows() == expected_b
    assert adapted.values.shape == (
        expected_b,
        _SEQ_LEN_SMALL,
        len(_FLEX_FEATURES),
    )
    assert adapted.feature_names == _FLEX_FEATURES
    assert "PM2.5" not in adapted.feature_names
    assert adapted.native_mask.shape[BATCH_AXIS] == expected_b
    assert adapted.native_mask.shape[TIME_AXIS] == _SEQ_LEN_SMALL
    assert adapted.native_mask.shape[FEATURE_AXIS] == len(_FLEX_FEATURES)


def test_flexible_reject_too_short() -> None:
    """``L < min_rows`` fails with a too-short user-facing error."""
    frame = _non_beijing_frame(_SHORT_ROWS)
    policy = default_ingest_reject_policy(_DEFAULT_SEQ_LEN)
    assert policy.min_rows == _DEFAULT_MIN_ROWS
    request = FlexibleAdaptRequest(
        window=WindowAdaptRequest(
            seq_len=_DEFAULT_SEQ_LEN,
            max_windows=None,
        ),
        reject=policy,
        apply_hourly_bin=False,
    )
    with pytest.raises(ValueError, match="too short"):
        adapt_flexible_frame(frame, request)


def test_flexible_reject_too_sparse_global() -> None:
    """Global observed rate below floor is rejected."""
    n_features = len(_FLEX_FEATURES)
    values = np.full(
        (_SPARSE_ROWS, n_features),
        MISSING_VALUE_SENTINEL,
        dtype=np.float32,
    )
    mask = np.full(
        (_SPARSE_ROWS, n_features),
        MASK_MISSING,
        dtype=np.float32,
    )
    values[:_SPARSE_OBSERVED_ROWS, :] = 1.0
    mask[:_SPARSE_OBSERVED_ROWS, :] = MASK_OBSERVED
    frame = ValidatedFrame(
        values=values,
        native_mask=mask,
        feature_names=_FLEX_FEATURES,
        timestamps=None,
        entity_id="lab-A",
    )
    with pytest.raises(ValueError, match="too sparse"):
        adapt_flexible_frame(
            frame,
            _flexible_request(_SEQ_LEN_SMALL, _MIN_ROWS_SMALL),
        )


def test_flexible_reject_too_sparse_per_feature() -> None:
    """A single all-missing feature is rejected even if others are dense."""
    values = np.ones((_SPARSE_ROWS, 2), dtype=np.float32)
    mask = np.ones((_SPARSE_ROWS, 2), dtype=np.float32)
    values[:, _HUM_F] = MISSING_VALUE_SENTINEL
    mask[:, _HUM_F] = MASK_MISSING
    frame = ValidatedFrame(
        values=values,
        native_mask=mask,
        feature_names=_FLEX_FEATURES,
        timestamps=None,
        entity_id=None,
    )
    with pytest.raises(ValueError, match="humidity_pct"):
        adapt_flexible_frame(
            frame,
            _flexible_request(_SEQ_LEN_SMALL, _MIN_ROWS_SMALL),
        )


def test_flexible_pad_short_but_allowed() -> None:
    """``min_rows <= L < T`` right-pads with ``m = 0`` on pad cells."""
    frame = _non_beijing_frame(_PAD_ROWS)
    assert _MIN_ROWS_SMALL <= _PAD_ROWS < _SEQ_LEN_SMALL
    adapted = adapt_flexible_frame(
        frame,
        _flexible_request(_SEQ_LEN_SMALL, _MIN_ROWS_SMALL),
    )
    assert adapted.n_windows() == 1
    assert np.all(
        adapted.native_mask[0, :_PAD_ROWS, :] == MASK_OBSERVED
    )
    assert np.all(
        adapted.native_mask[0, _PAD_ROWS:, :] == MASK_MISSING
    )
    assert np.all(
        adapted.values[0, _PAD_ROWS:, :] == MISSING_VALUE_SENTINEL
    )


def test_exact_path_still_pads_tiny_without_reject() -> None:
    """Exact ``adapt_validated_frame`` does not apply M6 min_rows reject."""
    frame = _non_beijing_frame(_SHORT_ROWS)
    adapted = adapt_validated_frame(
        frame,
        WindowAdaptRequest(seq_len=_SEQ_LEN_SMALL, max_windows=None),
    )
    assert adapted.n_windows() == 1
    assert np.all(
        adapted.native_mask[0, _SHORT_ROWS:, :] == MASK_MISSING
    )


def test_hourly_bin_inserts_gap_hours() -> None:
    """Timestamps with a missing hour become native-missing grid steps."""
    stamps = np.asarray(
        [
            "2020-01-01T00:00:00",
            "2020-01-01T02:00:00",
        ],
        dtype="datetime64[s]",
    )
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mask = np.ones((2, 2), dtype=np.float32)
    frame = ValidatedFrame(
        values=values,
        native_mask=mask,
        feature_names=_FLEX_FEATURES,
        timestamps=stamps,
        entity_id="lab-A",
    )
    binned = bin_validated_frame_hourly(frame)
    assert binned.n_rows() == 3
    assert binned.native_mask[1, _TEMP_F] == MASK_MISSING
    assert binned.values[1, _TEMP_F] == MISSING_VALUE_SENTINEL
    assert binned.native_mask[0, _TEMP_F] == MASK_OBSERVED
    assert binned.values[2, _TEMP_F] == pytest.approx(3.0)
