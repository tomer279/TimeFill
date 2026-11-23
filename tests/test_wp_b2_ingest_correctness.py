"""WP-B2 ingest correctness tests (B3, B6, B7).

These encode the Milestone 7.5 Done-when scenarios. They must fail on the
pre-fix code (first-row entity only, Exact row-order windows, finite-unchecked
parses, silent duplicate-hour bins) and pass after WP-B2 lands.

test_two_sites_sequential_24h_rejected
    A-then-B 24h multi-entity Exact CSV is rejected.
test_two_sites_feature_mix_hours_rejected
    Overlapping hours where site B lacks a feature site A has are rejected.
test_two_sites_disjoint_hours_rejected
    Two sites at different hours are rejected (no silent interleave).
test_exact_descending_timestamps_sorted
    Exact path with timestamps yields time-ordered windows.
test_exact_half_hour_rows_binned
    Exact 30-minute rows bin and surface a multi-row-hour preview warning.
test_exact_without_timestamps_keeps_row_order
    Exact CSV without timestamps keeps upload row order.
test_soft_exact_duplicate_bins_with_preview_warning
    Below-threshold exact dups bin with a multi-row preview warning.
test_station_column_multi_site_rejected
    Raw PRSA ``station`` multi-site files are rejected.
test_inf_raises_with_row_and_column
    ``inf`` is rejected with column and data-row context.
test_neg_inf_raises_with_row_and_column
    ``-inf`` is rejected with column and data-row context.
test_overflow_float_raises_with_row_and_column
    ``1e999`` is rejected with column and data-row context.
test_float32_overflow_raises_with_row_and_column
    Values above float32 range are rejected (no silent ``inf``).
test_duplicate_hour_share_above_threshold_rejected
    Hourly bins with too many exact-duplicate hours are rejected.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from app.m4_config import (
    AppM4ArtifactPaths,
    AppM4Config,
    AppM4SampleKnobs,
    AppM4Schema,
    AppM4SmokeCaps,
)
from app.preview_ingest import (
    MODE_EXACT_BEIJING,
    PreviewBuildInputs,
    build_ingest_preview,
)
from app.schema_validate import (
    ValidatedFrame,
    default_app_schema,
    validate_upload_csv,
    validate_upload_rows,
)
from app.ui_session import validate_and_adapt_upload
from app.window_adapter import bin_validated_frame_hourly
from src.data.real.specs import default_beijing_features
from src.data.schema import MASK_OBSERVED

_FEATURES = default_beijing_features()
_PM25 = "PM2.5"
_PM10 = "PM10"
_PM25_IDX = 0
_ENTITY_A = "SiteA"
_ENTITY_B = "SiteB"
_HOURS_PER_SITE = 24
_SEQ_LEN = 8
_BASE_DAY = "2013-03-01"
_NEXT_DAY = "2013-03-02"
_DUP_HOUR_SHARE_THRESHOLD = 0.01
_N_DUP_GRID_HOURS = 2
_N_DUP_RAW_ROWS = 3
_FLOAT32_OVERFLOW = "1e39"
_SOFT_DUP_GRID_HOURS = 200
_MULTI_ROW_WARNING_SUFFIX = "hours had multiple rows; kept last value"


def _feature_cells(fill: str) -> dict[str, str]:
    """Return Beijing feature cells filled with one string value.

    Parameters
    ----------
    fill : str
        Cell text for every pollutant column.

    Returns
    -------
    dict of str to str
        Feature name → cell text.
    """
    return {name: fill for name in _FEATURES}


def _beijing_row(timestamp: str, entity: str, fill: str) -> dict[str, str]:
    """Build one Exact Beijing CSV row.

    Parameters
    ----------
    timestamp : str
        ISO wall-clock cell.
    entity : str
        Site label.
    fill : str
        Value written into every feature column.

    Returns
    -------
    dict of str to str
        Header-keyed row.
    """
    row = {"timestamp": timestamp, "entity": entity}
    row.update(_feature_cells(fill))
    return row


def _write_beijing_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write an Exact Beijing CSV.

    Parameters
    ----------
    path : pathlib.Path
        Output path.
    rows : list of dict
        Data rows (header-keyed).
    """
    fieldnames = ["timestamp", "entity", *_FEATURES]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _hour_stamp(day: str, hour: int) -> str:
    """Format an hourly ISO timestamp.

    Parameters
    ----------
    day : str
        ``YYYY-MM-DD`` prefix.
    hour : int
        Hour of day in ``[0, 23]``.

    Returns
    -------
    str
        ``YYYY-MM-DDTHH:00:00``.
    """
    return f"{day}T{hour:02d}:00:00"


def _tiny_app_config(tmp_path: Path) -> AppM4Config:
    """Build a hermetic Exact adapt config (paths unused for validate/adapt).

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temp directory for dummy artifact paths.

    Returns
    -------
    AppM4Config
        Config with small ``seq_len`` and no window cap.
    """
    return AppM4Config(
        paths=AppM4ArtifactPaths(
            checkpoint=tmp_path / "unused.eqx",
            scaler=tmp_path / "unused.npz",
            sample_csv=tmp_path / "unused.csv",
            real_config=tmp_path / "unused.yaml",
        ),
        schema=AppM4Schema(
            n_features=len(_FEATURES),
            feature_names=_FEATURES,
            seq_len=_SEQ_LEN,
        ),
        sample=AppM4SampleKnobs(
            k_samples=2,
            solver_steps=2,
            seed_mask=0,
            seed_sample=1,
        ),
        smoke=AppM4SmokeCaps(
            max_windows=None,
            k_samples=2,
            solver_steps=2,
            mcar_rates=(0.3,),
            block_len=8,
        ),
        artifact_dir=tmp_path / "artifacts",
    )


def _exact_adapt(path: Path, tmp_path: Path):
    """Run Exact Beijing validate + adapt (production entry point).

    Parameters
    ----------
    path : pathlib.Path
        CSV path.
    tmp_path : pathlib.Path
        Temp dir for hermetic config paths.

    Returns
    -------
    tuple
        ``(ValidatedFrame, AdaptOutcome)`` from ``validate_and_adapt_upload``.
    """
    return validate_and_adapt_upload(
        path,
        MODE_EXACT_BEIJING,
        _tiny_app_config(tmp_path),
    )


def test_two_sites_sequential_24h_rejected(tmp_path: Path) -> None:
    """A-then-B 24h each must not validate into a single-entity frame."""
    rows: list[dict[str, str]] = []
    for hour in range(_HOURS_PER_SITE):
        rows.append(
            _beijing_row(_hour_stamp(_BASE_DAY, hour), _ENTITY_A, "10.0")
        )
    for hour in range(_HOURS_PER_SITE):
        rows.append(
            _beijing_row(_hour_stamp(_NEXT_DAY, hour), _ENTITY_B, "20.0")
        )
    path = tmp_path / "two_sites_24h.csv"
    _write_beijing_csv(path, rows)
    with pytest.raises(ValueError) as exc_info:
        validate_upload_csv(path, default_app_schema())
    message = str(exc_info.value)
    assert _ENTITY_A in message
    assert _ENTITY_B in message


def test_two_sites_feature_mix_hours_rejected() -> None:
    """Overlapping hours that would mix features across sites are rejected."""
    rows = [
        _beijing_row(f"{_BASE_DAY}T00:00:00", _ENTITY_A, "10.0"),
        _beijing_row(f"{_BASE_DAY}T00:00:00", _ENTITY_B, "20.0"),
    ]
    rows[1][_PM10] = ""
    with pytest.raises(ValueError) as exc_info:
        validate_upload_rows(rows, default_app_schema())
    message = str(exc_info.value)
    assert _ENTITY_A in message
    assert _ENTITY_B in message


def test_two_sites_disjoint_hours_rejected() -> None:
    """Sites reporting at different hours must not interleave silently."""
    rows = [
        _beijing_row(f"{_BASE_DAY}T00:00:00", _ENTITY_A, "10.0"),
        _beijing_row(f"{_BASE_DAY}T01:00:00", _ENTITY_B, "20.0"),
    ]
    with pytest.raises(ValueError) as exc_info:
        validate_upload_rows(rows, default_app_schema())
    message = str(exc_info.value)
    assert _ENTITY_A in message
    assert _ENTITY_B in message


def test_station_column_multi_site_rejected() -> None:
    """Raw PRSA ``station`` column must trigger the multi-entity reject."""
    rows = [
        {
            "timestamp": f"{_BASE_DAY}T00:00:00",
            "station": _ENTITY_A,
            **_feature_cells("10.0"),
        },
        {
            "timestamp": f"{_BASE_DAY}T01:00:00",
            "station": _ENTITY_B,
            **_feature_cells("20.0"),
        },
    ]
    with pytest.raises(ValueError) as exc_info:
        validate_upload_rows(rows, default_app_schema())
    message = str(exc_info.value)
    assert _ENTITY_A in message
    assert _ENTITY_B in message


def test_exact_descending_timestamps_sorted(tmp_path: Path) -> None:
    """Exact adapt with timestamps must expose increasing wall-clock order."""
    rows = [
        _beijing_row(f"{_BASE_DAY}T02:00:00", _ENTITY_A, "30.0"),
        _beijing_row(f"{_BASE_DAY}T01:00:00", _ENTITY_A, "20.0"),
        _beijing_row(f"{_BASE_DAY}T00:00:00", _ENTITY_A, "10.0"),
    ]
    path = tmp_path / "descending.csv"
    _write_beijing_csv(path, rows)
    _frame, outcome = _exact_adapt(path, tmp_path)
    assert outcome.adapt_error is None
    assert outcome.windows is not None
    pm25 = outcome.windows.values[0, :3, _PM25_IDX]
    assert pm25 == pytest.approx([10.0, 20.0, 30.0])


def test_exact_half_hour_rows_binned(tmp_path: Path) -> None:
    """Exact 30-minute rows bin to hourly last values and warn in preview."""
    rows = [
        _beijing_row(f"{_BASE_DAY}T00:00:00", _ENTITY_A, "1.0"),
        _beijing_row(f"{_BASE_DAY}T00:30:00", _ENTITY_A, "2.0"),
        _beijing_row(f"{_BASE_DAY}T01:00:00", _ENTITY_A, "3.0"),
        _beijing_row(f"{_BASE_DAY}T01:30:00", _ENTITY_A, "4.0"),
    ]
    path = tmp_path / "half_hour.csv"
    _write_beijing_csv(path, rows)
    frame, outcome = _exact_adapt(path, tmp_path)
    assert outcome.adapt_error is None
    assert outcome.windows is not None
    pm25 = outcome.windows.values[0, :2, _PM25_IDX]
    assert pm25 == pytest.approx([2.0, 4.0])
    preview = build_ingest_preview(
        PreviewBuildInputs(
            frame=frame,
            ingest_mode=MODE_EXACT_BEIJING,
            windows=outcome.windows,
            adapt_error=None,
        )
    )
    assert len(preview.adapt.warnings) == 1
    assert _MULTI_ROW_WARNING_SUFFIX in preview.adapt.warnings[0]
    assert "2 hours" in preview.adapt.warnings[0]


def _assert_non_finite_error(exc: BaseException) -> None:
    """Require column name, data-row context, and non-finite wording.

    Parameters
    ----------
    exc : BaseException
        Raised validation error.
    """
    message = str(exc)
    assert _PM25 in message
    assert "data row" in message
    assert "non-finite" in message


def test_inf_raises_with_row_and_column() -> None:
    """``inf`` must raise naming the feature column and data row."""
    rows = [_beijing_row(f"{_BASE_DAY}T00:00:00", _ENTITY_A, "10.0")]
    rows[0][_PM25] = "inf"
    with pytest.raises(ValueError) as exc_info:
        validate_upload_rows(rows, default_app_schema())
    _assert_non_finite_error(exc_info.value)


def test_neg_inf_raises_with_row_and_column() -> None:
    """``-inf`` must raise naming the feature column and data row."""
    rows = [_beijing_row(f"{_BASE_DAY}T00:00:00", _ENTITY_A, "10.0")]
    rows[0][_PM25] = "-inf"
    with pytest.raises(ValueError) as exc_info:
        validate_upload_rows(rows, default_app_schema())
    _assert_non_finite_error(exc_info.value)


def test_overflow_float_raises_with_row_and_column() -> None:
    """``1e999`` must raise naming the feature column and data row."""
    rows = [_beijing_row(f"{_BASE_DAY}T00:00:00", _ENTITY_A, "10.0")]
    rows[0][_PM25] = "1e999"
    with pytest.raises(ValueError) as exc_info:
        validate_upload_rows(rows, default_app_schema())
    _assert_non_finite_error(exc_info.value)


def test_float32_overflow_raises_with_row_and_column() -> None:
    """Values above float32 range must not land as observed ``inf``."""
    rows = [_beijing_row(f"{_BASE_DAY}T00:00:00", _ENTITY_A, "10.0")]
    rows[0][_PM25] = _FLOAT32_OVERFLOW
    with pytest.raises(ValueError) as exc_info:
        validate_upload_rows(rows, default_app_schema())
    _assert_non_finite_error(exc_info.value)


def _beijing_frame(
        stamps: np.ndarray,
        fills: list[float],
) -> ValidatedFrame:
    """Build an Exact-shaped frame with constant fill per row.

    Parameters
    ----------
    stamps : numpy.ndarray
        Length-``L`` timestamps.
    fills : list of float
        Per-row pollutant fill value.

    Returns
    -------
    ValidatedFrame
        Observed float32 table.
    """
    values = np.asarray(
        [[fill] * len(_FEATURES) for fill in fills],
        dtype=np.float32,
    )
    masks = np.ones_like(values, dtype=np.float32) * MASK_OBSERVED
    return ValidatedFrame(
        values=values,
        native_mask=masks,
        feature_names=_FEATURES,
        timestamps=stamps,
        entity_id=_ENTITY_A,
    )


def test_duplicate_hour_share_above_threshold_rejected() -> None:
    """Duplicate-hour share above the locked threshold must reject."""
    # 2 grid hours, 1 with exact same-second dup → share 0.5 > 0.01.
    stamps = np.asarray(
        [
            f"{_BASE_DAY}T00:00:00",
            f"{_BASE_DAY}T00:00:00",
            f"{_BASE_DAY}T01:00:00",
        ],
        dtype="datetime64[s]",
    )
    assert _N_DUP_RAW_ROWS == stamps.shape[0]
    assert _N_DUP_GRID_HOURS == 2
    frame = _beijing_frame(stamps, [1.0, 2.0, 3.0])
    assert (
        1.0 / float(_N_DUP_GRID_HOURS)
    ) > _DUP_HOUR_SHARE_THRESHOLD
    with pytest.raises(ValueError, match="duplicate"):
        bin_validated_frame_hourly(frame)


def test_soft_exact_duplicate_bins_with_preview_warning() -> None:
    """Below-threshold exact dups bin with a multi-row-hour preview warning."""
    stamps = [
        np.datetime64(f"{_BASE_DAY}T00:00:00") + np.timedelta64(hour, "h")
        for hour in range(_SOFT_DUP_GRID_HOURS)
    ]
    # Exact dup at hour 0 keeps share = 1/200 = 0.005 < 0.01.
    stamps.insert(1, stamps[0])
    fills = [float(idx) for idx in range(len(stamps))]
    frame = _beijing_frame(np.asarray(stamps), fills)
    binned = bin_validated_frame_hourly(frame)
    assert binned.n_rows() == _SOFT_DUP_GRID_HOURS
    preview = build_ingest_preview(
        PreviewBuildInputs(
            frame=frame,
            ingest_mode=MODE_EXACT_BEIJING,
            windows=None,
            adapt_error=None,
        )
    )
    assert len(preview.adapt.warnings) == 1
    assert _MULTI_ROW_WARNING_SUFFIX in preview.adapt.warnings[0]


def test_exact_without_timestamps_keeps_row_order(tmp_path: Path) -> None:
    """Exact CSV without a timestamp column keeps upload row order."""
    fieldnames = ["entity", *_FEATURES]
    rows = [
        {"entity": _ENTITY_A, **_feature_cells("30.0")},
        {"entity": _ENTITY_A, **_feature_cells("20.0")},
        {"entity": _ENTITY_A, **_feature_cells("10.0")},
    ]
    path = tmp_path / "no_time.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    frame, outcome = _exact_adapt(path, tmp_path)
    assert frame.timestamps is None
    assert outcome.windows is not None
    pm25 = outcome.windows.values[0, :3, _PM25_IDX]
    assert pm25 == pytest.approx([30.0, 20.0, 10.0])
    preview = build_ingest_preview(
        PreviewBuildInputs(
            frame=frame,
            ingest_mode=MODE_EXACT_BEIJING,
            windows=outcome.windows,
            adapt_error=None,
        )
    )
    assert preview.adapt.warnings == ()

