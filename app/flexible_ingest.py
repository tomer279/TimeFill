"""Infer / map arbitrary CSV columns into a ``ValidatedFrame`` (M6 WP1).

Flexible ingest never silently renames columns to Beijing ``PM2.5``…``O3``.
Exact Beijing uploads stay on ``app.schema_validate.validate_upload_*``.

Classes
-------
ColumnRoleMap
    Explicit or inferred feature / timestamp / entity column roles.
IngestInferConfig
    Heuristic hints and ``max_features`` cap (ASSUMPTIONS M6).

Functions
---------
column_role_map_from_mapping
    Build a ``ColumnRoleMap`` from a YAML-ready mapping dict.
infer_column_roles
    Suggest roles from headers and sample rows.
ingest_flexible_rows
    Validate hermetic string rows under a role map (infer if omitted).
ingest_flexible_csv
    Load a CSV path and return a ``ValidatedFrame``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.schema_validate import (
    AppSchemaSpec,
    ValidatedFrame,
    validate_upload_rows,
)

_MISSING_TOKENS = frozenset({"", "na", "nan", "null", "none"})
_ABSENT_TIMESTAMP = "__timefill_no_timestamp__"
_DEFAULT_MAX_FEATURES = 32
_DEFAULT_TIMESTAMP_HINTS = (
    "timestamp",
    "time",
    "datetime",
    "date",
)
_DEFAULT_ENTITY_HINTS = (
    "entity",
    "id",
    "site",
    "station",
    "sensor",
    "sensor_id",
)


@dataclass(frozen=True)
class ColumnRoleMap:
    """Column roles for flexible-schema ingest (no Beijing rename).

    Public methods
    --------------
    n_features
        Return ``F = len(features)``.
    validate
        Reject empty / duplicate / overlapping role assignments.
    to_app_schema
        Build an ``AppSchemaSpec`` for ``validate_upload_rows``.

    Instance variables
    ------------------
    features : tuple of str
        Source CSV column names used as the feature axis (order kept).
    timestamp : str or None
        Optional wall-clock column name.
    entity : str or None
        Optional entity / id column name.
    """

    features: tuple[str, ...]
    timestamp: str | None = None
    entity: str | None = None

    def n_features(self) -> int:
        """Return the feature count ``F``.

        Returns
        -------
        int
            ``len(features)``.
        """
        return len(self.features)

    def validate(self) -> None:
        """Raise ``ValueError`` if the role map is inconsistent."""
        if not self.features:
            raise ValueError(
                "features must be non-empty (zero feature columns "
                "after map/infer)"
            )
        if len(set(self.features)) != len(self.features):
            raise ValueError(
                "duplicate feature column mapping: "
                f"{self.features!r}"
            )
        roles = [
            ("timestamp", self.timestamp),
            ("entity", self.entity),
        ]
        for role_name, column in roles:
            if column is None:
                continue
            if not column:
                raise ValueError(f"{role_name} must be non-empty when set")
            if column in self.features:
                raise ValueError(
                    f"duplicate map: {role_name} column {column!r} "
                    "is also listed under features"
                )
        if (
                self.timestamp is not None
                and self.entity is not None
                and self.timestamp == self.entity
        ):
            raise ValueError(
                "duplicate map: timestamp and entity both use "
                f"{self.timestamp!r}"
            )

    def to_app_schema(self) -> AppSchemaSpec:
        """Return an ``AppSchemaSpec`` compatible with Exact-path parsers.

        Returns
        -------
        AppSchemaSpec
            Feature names preserved; absent timestamp uses an internal
            sentinel name that is never invented as a Beijing column.
        """
        self.validate()
        entity_columns: tuple[str, ...] = (
            (self.entity,) if self.entity is not None else ()
        )
        timestamp_column = (
            self.timestamp
            if self.timestamp is not None
            else _ABSENT_TIMESTAMP
        )
        return AppSchemaSpec(
            feature_names=self.features,
            timestamp_column=timestamp_column,
            entity_columns=entity_columns,
            require_timestamp=False,
        )


@dataclass(frozen=True)
class IngestInferConfig:
    """Heuristics and caps for flexible column inference.

    Public methods
    --------------
    validate
        Check ``max_features`` and non-empty hint tuples.
    feature_cap
        Return the locked ``max_features`` soft cap.

    Instance variables
    ------------------
    max_features : int
        Reject when ``F > max_features`` (ASSUMPTIONS default 32).
    timestamp_name_hints : tuple of str
        Case-insensitive header names preferred as timestamp.
    entity_name_hints : tuple of str
        Case-insensitive header names preferred as entity / id.
    """

    max_features: int = _DEFAULT_MAX_FEATURES
    timestamp_name_hints: tuple[str, ...] = _DEFAULT_TIMESTAMP_HINTS
    entity_name_hints: tuple[str, ...] = _DEFAULT_ENTITY_HINTS

    def validate(self) -> None:
        """Raise ``ValueError`` if inference knobs are invalid."""
        if self.max_features < 1:
            raise ValueError(
                f"max_features must be >= 1, got {self.max_features}"
            )
        if not self.timestamp_name_hints:
            raise ValueError("timestamp_name_hints must be non-empty")
        if not self.entity_name_hints:
            raise ValueError("entity_name_hints must be non-empty")

    def feature_cap(self) -> int:
        """Return the maximum allowed feature count.

        Returns
        -------
        int
            ``max_features``.
        """
        return self.max_features


def column_role_map_from_mapping(mapping: dict) -> ColumnRoleMap:
    """Build a ``ColumnRoleMap`` from a YAML-ready dict.

    Expected keys: ``features`` (sequence of str), optional ``timestamp``,
    optional ``entity``. Feature names are source headers only — never
    rewritten to Beijing pollutant names.

    Parameters
    ----------
    mapping : dict
        Role mapping (e.g. loaded from YAML).

    Returns
    -------
    ColumnRoleMap
        Validated role map.

    Raises
    ------
    ValueError
        If ``features`` is missing / empty / not a string sequence, or
        roles overlap.
    """
    raw_features = mapping.get("features")
    if raw_features is None:
        raise ValueError("mapping must include 'features'")
    if isinstance(raw_features, (str, bytes)) or not hasattr(
            raw_features, "__iter__"
    ):
        raise ValueError("mapping 'features' must be a sequence of strings")
    features = tuple(str(name) for name in raw_features)
    timestamp = _optional_str_field(mapping, "timestamp")
    entity = _optional_str_field(mapping, "entity")
    role_map = ColumnRoleMap(
        features=features,
        timestamp=timestamp,
        entity=entity,
    )
    role_map.validate()
    return role_map


def infer_column_roles(
        headers: tuple[str, ...],
        rows: list[dict[str, str]],
        config: IngestInferConfig,
) -> ColumnRoleMap:
    """Infer feature / timestamp / entity roles from CSV headers and rows.

    Numeric columns become features (header order). Datetime-like or
    name-hinted columns become timestamp. Id-like name hints become entity.
    No column is renamed to Beijing ``PM2.5``…``O3``.

    Parameters
    ----------
    headers : tuple of str
        CSV header names in file order.
    rows : list of dict
        Header-keyed cell strings.
    config : IngestInferConfig
        Hints and ``max_features`` cap.

    Returns
    -------
    ColumnRoleMap
        Inferred roles.

    Raises
    ------
    ValueError
        If headers / rows are empty, no numeric features remain, or
        ``F > max_features``.
    """
    config.validate()
    if not headers:
        raise ValueError("CSV has no header columns")
    if not rows:
        raise ValueError("CSV has no data rows")
    timestamp = _pick_timestamp_column(headers, rows, config)
    entity = _pick_entity_column(headers, config, timestamp)
    reserved = {
        name for name in (timestamp, entity) if name is not None
    }
    features = _pick_feature_columns(headers, rows, reserved)
    role_map = ColumnRoleMap(
        features=features,
        timestamp=timestamp,
        entity=entity,
    )
    role_map.validate()
    _assert_within_feature_cap(role_map, config)
    return role_map


def ingest_flexible_rows(
        rows: list[dict[str, str]],
        headers: tuple[str, ...],
        role_map: ColumnRoleMap | None = None,
        config: IngestInferConfig | None = None,
) -> ValidatedFrame:
    """Validate flexible-schema string rows into a ``ValidatedFrame``.

    Parameters
    ----------
    rows : list of dict
        Header-keyed cell strings.
    headers : tuple of str
        CSV header names in file order.
    role_map : ColumnRoleMap or None, optional
        Explicit roles; inferred when ``None``.
    config : IngestInferConfig or None, optional
        Inference / cap settings (defaults when ``None``).

    Returns
    -------
    ValidatedFrame
        Values ``(L, F)``, native masks, names, optional timestamps /
        entity.

    Raises
    ------
    ValueError
        On empty input, invalid maps, missing columns, non-numeric
        feature cells, or ``F`` above cap.
    """
    resolved = config if config is not None else IngestInferConfig()
    resolved.validate()
    if not headers:
        raise ValueError("CSV has no header columns")
    if not rows:
        raise ValueError("CSV has no data rows")
    active_map = role_map
    if active_map is None:
        active_map = infer_column_roles(headers, rows, resolved)
    else:
        active_map.validate()
        _assert_map_columns_present(active_map, set(headers))
        _assert_within_feature_cap(active_map, resolved)
    schema = active_map.to_app_schema()
    return validate_upload_rows(rows, schema)


def ingest_flexible_csv(
        path: Path,
        role_map: ColumnRoleMap | None = None,
        config: IngestInferConfig | None = None,
) -> ValidatedFrame:
    """Load a CSV path and validate under flexible-schema rules.

    Parameters
    ----------
    path : pathlib.Path
        CSV path (e.g. ``tests/fixtures/toy_non_beijing.csv``).
    role_map : ColumnRoleMap or None, optional
        Explicit roles; inferred when ``None``.
    config : IngestInferConfig or None, optional
        Inference / cap settings (defaults when ``None``).

    Returns
    -------
    ValidatedFrame
        Parsed flexible-schema frame (not Exact Beijing unless headers
        happen to match).

    Raises
    ------
    ValueError
        If the file is empty, has no header, or fails flexible validation.
    """
    headers, rows = _read_csv_table(Path(path))
    return ingest_flexible_rows(
        rows,
        headers,
        role_map=role_map,
        config=config,
    )


def _optional_str_field(mapping: dict, key: str) -> str | None:
    """Return an optional string role from a mapping dict.

    Parameters
    ----------
    mapping : dict
        Role mapping.
    key : str
        Field name.

    Returns
    -------
    str or None
        Stripped string, or ``None`` when absent / null.

    Raises
    ------
    ValueError
        If the value is present but not a non-empty string.
    """
    if key not in mapping or mapping[key] is None:
        return None
    value = mapping[key]
    if not isinstance(value, str):
        raise ValueError(f"mapping {key!r} must be a string or null")
    text = value.strip()
    if not text:
        raise ValueError(f"mapping {key!r} must be non-empty when set")
    return text


def _read_csv_table(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """Read CSV headers and string rows.

    Parameters
    ----------
    path : pathlib.Path
        CSV path.

    Returns
    -------
    headers : tuple of str
        Field names in file order.
    rows : list of dict
        Header-keyed rows.

    Raises
    ------
    ValueError
        If the file is empty or has no header row.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise ValueError(f"CSV path does not exist: {csv_path}")
    if csv_path.stat().st_size == 0:
        raise ValueError(f"CSV file is empty: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")
        headers = tuple(
            name for name in reader.fieldnames
            if name is not None and str(name).strip() != ""
        )
        if not headers:
            raise ValueError(f"CSV has no header columns: {csv_path}")
        rows = list(reader)
    return headers, rows


def _assert_map_columns_present(
        role_map: ColumnRoleMap,
        header: set[str],
) -> None:
    """Reject role maps that reference missing headers.

    Parameters
    ----------
    role_map : ColumnRoleMap
        Explicit column roles.
    header : set of str
        CSV header field names.

    Raises
    ------
    ValueError
        If any mapped column is absent.
    """
    missing = [name for name in role_map.features if name not in header]
    if role_map.timestamp is not None and role_map.timestamp not in header:
        missing.append(role_map.timestamp)
    if role_map.entity is not None and role_map.entity not in header:
        missing.append(role_map.entity)
    if missing:
        raise ValueError(
            "column map references missing header(s): "
            f"{missing!r}. Columns are not remapped to Beijing names."
        )


def _assert_within_feature_cap(
        role_map: ColumnRoleMap,
        config: IngestInferConfig,
) -> None:
    """Reject frames with ``F`` above the locked soft cap.

    Parameters
    ----------
    role_map : ColumnRoleMap
        Feature roles.
    config : IngestInferConfig
        Cap source.

    Raises
    ------
    ValueError
        If ``n_features() > max_features``.
    """
    n_features = role_map.n_features()
    cap = config.feature_cap()
    if n_features > cap:
        raise ValueError(
            f"feature count F={n_features} exceeds max_features={cap}"
        )


def _hint_match(column: str, hints: tuple[str, ...]) -> bool:
    """Return whether ``column`` matches any hint (case-insensitive).

    Parameters
    ----------
    column : str
        Header name.
    hints : tuple of str
        Allowed aliases.

    Returns
    -------
    bool
        ``True`` on casefold equality with any hint.
    """
    folded = column.casefold()
    return any(folded == hint.casefold() for hint in hints)


def _pick_timestamp_column(
        headers: tuple[str, ...],
        rows: list[dict[str, str]],
        config: IngestInferConfig,
) -> str | None:
    """Choose a timestamp column by name hint, else datetime-like values.

    Parameters
    ----------
    headers : tuple of str
        Header names.
    rows : list of dict
        Sample rows.
    config : IngestInferConfig
        Timestamp name hints.

    Returns
    -------
    str or None
        Selected timestamp column, if any.
    """
    for name in headers:
        if _hint_match(name, config.timestamp_name_hints):
            return name
    for name in headers:
        if _column_is_numeric(rows, name):
            continue
        if _column_is_datetime_like(rows, name):
            return name
    return None


def _pick_entity_column(
        headers: tuple[str, ...],
        config: IngestInferConfig,
        timestamp: str | None,
) -> str | None:
    """Choose an entity column by name hint among non-timestamp headers.

    Parameters
    ----------
    headers : tuple of str
        Header names.
    config : IngestInferConfig
        Entity name hints.
    timestamp : str or None
        Already claimed timestamp column.

    Returns
    -------
    str or None
        Selected entity column, if any.
    """
    for name in headers:
        if timestamp is not None and name == timestamp:
            continue
        if _hint_match(name, config.entity_name_hints):
            return name
    return None


def _pick_feature_columns(
        headers: tuple[str, ...],
        rows: list[dict[str, str]],
        reserved: set[str],
) -> tuple[str, ...]:
    """Select numeric columns not reserved for timestamp / entity.

    Parameters
    ----------
    headers : tuple of str
        Header names in file order.
    rows : list of dict
        Sample rows.
    reserved : set of str
        Columns already assigned other roles.

    Returns
    -------
    tuple of str
        Feature column names (may be empty; caller validates).
    """
    selected: list[str] = []
    for name in headers:
        if name in reserved:
            continue
        if _column_is_numeric(rows, name):
            selected.append(name)
    return tuple(selected)


def _column_is_numeric(
        rows: list[dict[str, str]],
        column: str,
) -> bool:
    """Return whether every cell is numeric or a documented NA token.

    Parameters
    ----------
    rows : list of dict
        CSV rows.
    column : str
        Column name.

    Returns
    -------
    bool
        ``True`` when the column can serve as a feature.
    """
    if not rows:
        return False
    for row in rows:
        if not _cell_is_numeric_or_missing(row.get(column, "")):
            return False
    return True


def _column_is_datetime_like(
        rows: list[dict[str, str]],
        column: str,
) -> bool:
    """Return whether non-missing cells parse as datetimes.

    Parameters
    ----------
    rows : list of dict
        CSV rows.
    column : str
        Column name.

    Returns
    -------
    bool
        ``True`` when at least one non-missing datetime cell exists and
        every non-missing cell is datetime-like.
    """
    saw_value = False
    for row in rows:
        text = str(row.get(column, "")).strip()
        if text.lower() in _MISSING_TOKENS:
            continue
        if not _looks_datetime(text):
            return False
        saw_value = True
    return saw_value


def _cell_is_numeric_or_missing(raw: str) -> bool:
    """Return whether a cell is a float or a documented missing token.

    Parameters
    ----------
    raw : str
        Cell text.

    Returns
    -------
    bool
        ``True`` for floats / NA tokens (including NaN).
    """
    text = str(raw).strip()
    if text.lower() in _MISSING_TOKENS:
        return True
    try:
        number = float(text)
    except ValueError:
        return False
    return bool(np.isfinite(number) or np.isnan(number))


def _looks_datetime(text: str) -> bool:
    """Return whether ``text`` parses as ``datetime64``.

    Parameters
    ----------
    text : str
        Non-empty cell text.

    Returns
    -------
    bool
        ``True`` when NumPy accepts the string as datetime.
    """
    try:
        parsed = np.datetime64(text)
    except ValueError:
        return False
    return not np.isnat(parsed)
