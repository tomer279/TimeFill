"""CSV schema validation and missingness summaries for the M4 app.

Classes
-------
AppSchemaSpec
    Frozen column contract (features, optional timestamp / entity).
ValidatedFrame
    Parsed table ready for window adaptation.
MissingnessSummary
    Per-feature native missing rates.

Functions
---------
default_app_schema
    Schema matching the frozen Beijing ``F = 6`` feature list.
validate_upload_csv
    Load a CSV path and validate against an ``AppSchemaSpec``.
validate_upload_rows
    Validate already-parsed string rows (hermetic tests).
summarize_missingness
    Build a ``MissingnessSummary`` from a ``ValidatedFrame``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.real.specs import default_beijing_features
from src.data.schema import (
    MASK_MISSING,
    MASK_OBSERVED,
    MISSING_VALUE_SENTINEL,
)

_MISSING_TOKENS = frozenset({"", "na", "nan", "null", "none"})
_TIME_AXIS = 0
_FEATURE_AXIS = 1
_PERCENT = 100.0
_MASK_COMPARE_TOL = 1e-6
_DEFAULT_TIMESTAMP = "timestamp"
_DEFAULT_ENTITY_COLUMNS = ("entity", "site")


@dataclass(frozen=True)
class AppSchemaSpec:
    """Column contract for M4 upload / bundled sample CSVs.

    Public methods
    --------------
    n_features
        Return ``F = len(feature_names)``.
    validate
        Check that feature names are non-empty and unique.

    Instance variables
    ------------------
    feature_names : tuple of str
        Required pollutant columns in frozen EVAL order.
    timestamp_column : str
        Optional wall-clock column name when present.
    entity_columns : tuple of str
        Optional entity / site column aliases (first match wins).
    require_timestamp : bool
        If ``True``, reject CSVs missing ``timestamp_column``.
    """

    feature_names: tuple[str, ...]
    timestamp_column: str = _DEFAULT_TIMESTAMP
    entity_columns: tuple[str, ...] = _DEFAULT_ENTITY_COLUMNS
    require_timestamp: bool = False

    def n_features(self) -> int:
        """Return the feature count ``F``.

        Returns
        -------
        int
            ``len(feature_names)``.
        """
        return len(self.feature_names)

    def validate(self) -> None:
        """Raise ``ValueError`` if the schema spec is inconsistent."""
        if not self.feature_names:
            raise ValueError("feature_names must be non-empty")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError(
                f"feature_names must be unique; got {self.feature_names!r}"
            )
        if not self.timestamp_column:
            raise ValueError("timestamp_column must be non-empty")


@dataclass(frozen=True)
class ValidatedFrame:
    """Parsed upload / sample table in original feature units.

    Native empty / NaN cells are already mapped to ``m == 0`` and value
    sentinel ``0.0``. Missingness is never inferred from nonzero ``x``.

    Public methods
    --------------
    n_rows
        Number of time steps ``L``.
    n_features
        Feature count ``F``.

    Instance variables
    ------------------
    values : numpy.ndarray
        Float32 ``(L, F)``; native missing cells use sentinel ``0.0``.
    native_mask : numpy.ndarray
        Float32 ``(L, F)``; ``1`` observed, ``0`` missing.
    feature_names : tuple of str
        Column order matching the feature axis.
    timestamps : numpy.ndarray or None
        Length-``L`` wall-clock labels (``datetime64`` or object), if any.
    entity_id : str or None
        Optional site / entity label for the whole frame.
    """

    values: np.ndarray
    native_mask: np.ndarray
    feature_names: tuple[str, ...]
    timestamps: np.ndarray | None
    entity_id: str | None

    def n_rows(self) -> int:
        """Return the number of rows ``L``.

        Returns
        -------
        int
            ``values.shape[0]``.
        """
        return int(self.values.shape[_TIME_AXIS])

    def n_features(self) -> int:
        """Return the feature count ``F``.

        Returns
        -------
        int
            ``values.shape[1]``.
        """
        return int(self.values.shape[_FEATURE_AXIS])


@dataclass(frozen=True)
class MissingnessSummary:
    """Per-feature native missingness for a validated frame.

    Public methods
    --------------
    per_feature_fraction
        Mapping feature name → fraction missing in ``[0, 1]``.
    per_feature_percent
        Mapping feature name → percent missing in ``[0, 100]``.

    Instance variables
    ------------------
    feature_names : tuple of str
        Feature axis labels.
    missing_counts : numpy.ndarray
        Int64 count of ``m == 0`` per feature, shape ``(F,)``.
    n_rows : int
        Number of time steps used as the denominator.
    """

    feature_names: tuple[str, ...]
    missing_counts: np.ndarray
    n_rows: int

    def per_feature_fraction(self) -> dict[str, float]:
        """Return missing fractions keyed by feature name.

        Returns
        -------
        dict of str to float
            Fraction of rows with ``m == 0`` per feature.
        """
        if self.n_rows <= 0:
            return {name: 0.0 for name in self.feature_names}
        fractions = self.missing_counts.astype(np.float64) / float(self.n_rows)
        return {
            name: float(fractions[idx])
            for idx, name in enumerate(self.feature_names)
        }

    def per_feature_percent(self) -> dict[str, float]:
        """Return missing percentages keyed by feature name.

        Returns
        -------
        dict of str to float
            Percent of rows with ``m == 0`` per feature.
        """
        fractions = self.per_feature_fraction()
        return {
            name: fraction * _PERCENT for name, fraction in fractions.items()
        }


@dataclass(frozen=True)
class _RowParseTarget:
    """Mutable output buffers for one feature row parse.

    Instance variables
    ------------------
    row_idx : int
        Zero-based data-row index (for error messages).
    values_out : numpy.ndarray
        Length-``F`` value buffer (mutated).
    masks_out : numpy.ndarray
        Length-``F`` mask buffer (mutated).
    """

    row_idx: int
    values_out: np.ndarray
    masks_out: np.ndarray


def default_app_schema() -> AppSchemaSpec:
    """Return the locked M4 Beijing feature schema.

    Returns
    -------
    AppSchemaSpec
        ``F = 6`` pollutants; timestamp recommended but not required.
    """
    return AppSchemaSpec(feature_names=default_beijing_features())


def validate_upload_csv(path: Path, schema: AppSchemaSpec) -> ValidatedFrame:
    """Validate a CSV file against the M4 upload schema.

    Parameters
    ----------
    path : pathlib.Path
        CSV path (bundled sample or user upload).
    schema : AppSchemaSpec
        Required feature columns and optional timestamp / entity names.

    Returns
    -------
    ValidatedFrame
        Parsed values, native masks, and optional metadata.

    Raises
    ------
    ValueError
        If columns are missing, non-numeric, or empty after the header.
    """
    schema.validate()
    rows = _read_csv_rows(Path(path))
    return validate_upload_rows(rows, schema)


def validate_upload_rows(
        rows: list[dict[str, str]],
        schema: AppSchemaSpec,
) -> ValidatedFrame:
    """Validate string-dict CSV rows against the M4 upload schema.

    Parameters
    ----------
    rows : list of dict
        Header-keyed cell strings (as from ``csv.DictReader``).
    schema : AppSchemaSpec
        Required feature columns and optional timestamp / entity names.

    Returns
    -------
    ValidatedFrame
        Parsed values, native masks, and optional metadata.

    Raises
    ------
    ValueError
        If columns are missing, non-numeric, or there are no data rows.
    """
    schema.validate()
    if not rows:
        raise ValueError("CSV has no data rows")
    header = set(rows[0].keys())
    _assert_required_columns(header, schema)
    n_rows = len(rows)
    n_features = schema.n_features()
    values = np.zeros((n_rows, n_features), dtype=np.float32)
    masks = np.zeros((n_rows, n_features), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        target = _RowParseTarget(
            row_idx=row_idx,
            values_out=values[row_idx],
            masks_out=masks[row_idx],
        )
        _fill_feature_row(row, schema, target)
    timestamps = _parse_timestamps(rows, schema, header)
    entity_id = _parse_entity_id(rows[0], schema, header)
    return ValidatedFrame(
        values=values,
        native_mask=masks,
        feature_names=schema.feature_names,
        timestamps=timestamps,
        entity_id=entity_id,
    )


def summarize_missingness(frame: ValidatedFrame) -> MissingnessSummary:
    """Compute per-feature native missingness for a validated frame.

    Parameters
    ----------
    frame : ValidatedFrame
        Parsed table with native masks.

    Returns
    -------
    MissingnessSummary
        Counts and helpers for UI / tests.
    """
    missing = np.abs(frame.native_mask - MASK_MISSING) < _MASK_COMPARE_TOL
    counts = np.sum(missing, axis=_TIME_AXIS).astype(np.int64)
    return MissingnessSummary(
        feature_names=frame.feature_names,
        missing_counts=counts,
        n_rows=frame.n_rows(),
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV into a list of string dictionaries.

    Parameters
    ----------
    path : pathlib.Path
        CSV path.

    Returns
    -------
    list of dict
        Header-keyed rows.

    Raises
    ------
    ValueError
        If the file has no header row.
    """
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {path}")
        return list(reader)


def _assert_required_columns(header: set[str], schema: AppSchemaSpec) -> None:
    """Reject missing required feature columns (no silent remaps).

    Parameters
    ----------
    header : set of str
        CSV header field names.
    schema : AppSchemaSpec
        Required feature / timestamp contract.

    Raises
    ------
    ValueError
        If required columns are absent.
    """
    missing = [name for name in schema.feature_names if name not in header]
    if missing:
        required = ", ".join(schema.feature_names)
        raise ValueError(
            "CSV is missing required feature column(s): "
            f"{missing!r}. Required exactly: {required}. "
            "Column aliases are not remapped."
        )
    if schema.require_timestamp and schema.timestamp_column not in header:
        raise ValueError(
            "CSV is missing required timestamp column "
            f"{schema.timestamp_column!r}"
        )


def _fill_feature_row(
        row: dict[str, str],
        schema: AppSchemaSpec,
        target: _RowParseTarget,
) -> None:
    """Parse one row's feature cells into value / mask buffers.

    Parameters
    ----------
    row : dict of str to str
        One CSV row.
    schema : AppSchemaSpec
        Feature column names.
    target : _RowParseTarget
        Row index and output buffers.
    """
    for feat_idx, name in enumerate(schema.feature_names):
        raw = row.get(name, "")
        try:
            value, observed = _parse_numeric_cell(raw)
        except ValueError as exc:
            raise ValueError(
                f"non-numeric value in column {name!r} at data row "
                f"{target.row_idx + 1}: {raw!r}"
            ) from exc
        target.values_out[feat_idx] = value
        target.masks_out[feat_idx] = observed


def _parse_numeric_cell(raw: str) -> tuple[float, float]:
    """Parse one CSV cell into ``(value, mask_flag)``.

    Empty / NA tokens and NaN become native missing with sentinel ``0.0``.

    Parameters
    ----------
    raw : str
        Cell text.

    Returns
    -------
    value : float
        Parsed float or ``MISSING_VALUE_SENTINEL``.
    observed : float
        ``MASK_OBSERVED`` or ``MASK_MISSING``.

    Raises
    ------
    ValueError
        If the cell is non-empty and not a finite float / NA token.
    """
    text = str(raw).strip()
    if text.lower() in _MISSING_TOKENS:
        return float(MISSING_VALUE_SENTINEL), float(MASK_MISSING)
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"non-numeric cell: {raw!r}") from exc
    if np.isnan(number):
        return float(MISSING_VALUE_SENTINEL), float(MASK_MISSING)
    return float(number), float(MASK_OBSERVED)


def _parse_timestamps(
        rows: list[dict[str, str]],
        schema: AppSchemaSpec,
        header: set[str],
) -> np.ndarray | None:
    """Parse optional timestamp column; ``None`` if the column is absent.

    Parameters
    ----------
    rows : list of dict
        CSV rows.
    schema : AppSchemaSpec
        Timestamp column name.
    header : set of str
        CSV header field names.

    Returns
    -------
    numpy.ndarray or None
        Length-``L`` timestamps, or ``None`` when the column is absent.

    Raises
    ------
    ValueError
        If the timestamp column is present but contains empty cells.
    """
    if schema.timestamp_column not in header:
        return None
    texts = [row.get(schema.timestamp_column, "").strip() for row in rows]
    if any(not text for text in texts):
        raise ValueError(
            f"timestamp column {schema.timestamp_column!r} has empty cells"
        )
    try:
        return np.asarray(texts, dtype="datetime64[s]")
    except ValueError:
        return np.asarray(texts, dtype=object)


def _parse_entity_id(
        first_row: dict[str, str],
        schema: AppSchemaSpec,
        header: set[str],
) -> str | None:
    """Return the first present entity / site column value, if any.

    Parameters
    ----------
    first_row : dict of str to str
        First data row (entity is treated as frame-level in M4).
    schema : AppSchemaSpec
        Entity column aliases.
    header : set of str
        CSV header field names.

    Returns
    -------
    str or None
        Entity label, or ``None`` if absent / blank.
    """
    for column in schema.entity_columns:
        if column in header:
            text = str(first_row.get(column, "")).strip()
            if text:
                return text
            return None
    return None
