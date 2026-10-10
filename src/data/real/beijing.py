"""Beijing Multi-Site CSV loading onto regular hourly grids.

Classes
-------
HourlySiteSeries
    One monitoring site on a regular hourly grid with native masks.
BeijingCsvLoader
    Parse site CSVs into hourly grids (last-value-in-bin; gaps → missing).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from src.data.real.specs import BeijingLoadSpec, RealDataPaths
from src.data.schema import (
    MASK_MISSING,
    MASK_OBSERVED,
    MISSING_VALUE_SENTINEL,
)

_MASK_TOL = 1e-6
_TIME_YEAR = 0
_TIME_MONTH = 1
_TIME_DAY = 2
_TIME_HOUR = 3
_HOUR_DELTA = np.timedelta64(1, "h")
_MISSING_TOKENS = frozenset({"", "na", "nan", "null", "none"})


@dataclass(frozen=True)
class HourlySiteSeries:
    """One entity on a regular hourly grid after resample / binning.

    Public methods
    --------------
    n_steps
        Length of the hourly grid ``L``.
    n_features
        Feature count ``F``.

    Instance variables
    ------------------
    entity_id : str
        Monitoring site id.
    timestamps : numpy.ndarray
        ``datetime64[h]`` grid of shape ``(L,)``.
    values : numpy.ndarray
        Float32 values ``(L, F)``; missing cells use sentinel ``0.0``.
    native_mask : numpy.ndarray
        Float32 mask ``(L, F)``; ``1`` observed, ``0`` missing.
    """

    entity_id: str
    timestamps: np.ndarray
    values: np.ndarray
    native_mask: np.ndarray

    def n_steps(self) -> int:
        """Return the number of hourly steps ``L``.

        Returns
        -------
        int
            ``timestamps.shape[0]``.
        """
        return int(self.timestamps.shape[0])

    def n_features(self) -> int:
        """Return the feature count ``F``.

        Returns
        -------
        int
            Second axis of ``values`` ``(L, F)``.
        """
        return int(self.values.shape[1])


@dataclass
class _GridBuffers:
    """Mutable hourly-grid buffers used while scattering CSV rows.

    Instance variables
    ------------------
    grid : numpy.ndarray
        Dense ``datetime64[h]`` axis.
    values : numpy.ndarray
        Float32 values ``(L, F)``.
    masks : numpy.ndarray
        Float32 native masks ``(L, F)``.
    """

    grid: np.ndarray
    values: np.ndarray
    masks: np.ndarray


class BeijingCsvLoader:
    """Load Beijing Multi-Site CSVs onto regular hourly grids.

    Missing source cells and empty hourly bins become native missing
    (``m = 0``, value sentinel ``0.0``). Duplicate hours keep the last
    **observed** value per feature (last-value-in-bin).

    Public methods
    --------------
    load_csv
        Parse one site CSV into an ``HourlySiteSeries``.
    load_directory
        Load every CSV matched by ``RealDataPaths``.

    Instance variables
    ------------------
    load_spec : BeijingLoadSpec
        Feature / time / station column contract.
    """

    def __init__(self, load_spec: BeijingLoadSpec | None = None) -> None:
        """Store the Beijing column contract.

        Parameters
        ----------
        load_spec : BeijingLoadSpec, optional
            Defaults to frozen M2 pollutant features.
        """
        spec = load_spec if load_spec is not None else BeijingLoadSpec()
        spec.validate()
        self.load_spec = spec

    def load_csv(self, path: Path) -> HourlySiteSeries:
        """Load one CSV file into a regular hourly site series.

        Parameters
        ----------
        path : pathlib.Path
            Path to a PRSA-style site CSV.

        Returns
        -------
        HourlySiteSeries
            Hourly grid with native masks.
        """
        rows = _read_csv_rows(Path(path))
        if not rows:
            raise ValueError(f"CSV has no data rows: {path}")
        entity_id = _infer_entity_id(rows, self.load_spec, Path(path))
        stamps, values, masks = _rows_to_arrays(rows, self.load_spec)
        return _resample_hourly(entity_id, stamps, values, masks)

    def load_directory(self, paths: RealDataPaths) -> list[HourlySiteSeries]:
        """Load all site CSVs under ``paths.raw_dir``.

        Parameters
        ----------
        paths : RealDataPaths
            Raw directory and glob.

        Returns
        -------
        list of HourlySiteSeries
            One series per CSV (sorted by path name).
        """
        return [
            self.load_csv(csv_path) for csv_path in paths.resolve_raw_files()
        ]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV into a list of string dictionaries.

    Parameters
    ----------
    path : pathlib.Path
        CSV path.

    Returns
    -------
    list of dict
        Row mappings from ``csv.DictReader``.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _infer_entity_id(
        rows: list[dict[str, str]],
        load_spec: BeijingLoadSpec,
        path: Path,
) -> str:
    """Infer the site id from the station column or file stem.

    Parameters
    ----------
    rows : list of dict
        CSV rows.
    load_spec : BeijingLoadSpec
        Provides ``station_column``.
    path : pathlib.Path
        Fallback stem if the station column is absent.

    Returns
    -------
    str
        Entity id.
    """
    column = load_spec.station_column
    if column in rows[0] and rows[0][column] not in (None, ""):
        return str(rows[0][column])
    return path.stem


def _rows_to_arrays(
        rows: list[dict[str, str]],
        load_spec: BeijingLoadSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert CSV rows to timestamp / value / native-mask arrays.

    Parameters
    ----------
    rows : list of dict
        CSV rows.
    load_spec : BeijingLoadSpec
        Feature and time columns.

    Returns
    -------
    stamps : numpy.ndarray
        ``datetime64[h]`` of shape ``(n_rows,)``.
    values : numpy.ndarray
        Float32 ``(n_rows, F)`` with sentinel on missing.
    masks : numpy.ndarray
        Float32 ``(n_rows, F)`` native masks.
    """
    n_rows = len(rows)
    n_features = load_spec.n_features()
    stamps = np.empty(n_rows, dtype="datetime64[h]")
    values = np.zeros((n_rows, n_features), dtype=np.float32)
    masks = np.zeros((n_rows, n_features), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        stamps[row_idx] = _row_timestamp(row, load_spec)
        _fill_feature_row(row, load_spec, values[row_idx], masks[row_idx])
    return stamps, values, masks


def _row_timestamp(
        row: dict[str, str],
        load_spec: BeijingLoadSpec,
) -> np.datetime64:
    """Parse ``year/month/day/hour`` columns into ``datetime64[h]``.

    Parameters
    ----------
    row : dict of str to str
        One CSV row.
    load_spec : BeijingLoadSpec
        Provides time column names.

    Returns
    -------
    numpy.datetime64
        Hour-resolution timestamp.
    """
    keys = load_spec.time_columns
    stamp = datetime(
        int(row[keys[_TIME_YEAR]]),
        int(row[keys[_TIME_MONTH]]),
        int(row[keys[_TIME_DAY]]),
        int(row[keys[_TIME_HOUR]]),
    )
    return np.datetime64(stamp, "h")


def _fill_feature_row(
        row: dict[str, str],
        load_spec: BeijingLoadSpec,
        values_out: np.ndarray,
        masks_out: np.ndarray,
) -> None:
    """Write one CSV row's features into preallocated vectors.

    Parameters
    ----------
    row : dict of str to str
        One CSV row.
    load_spec : BeijingLoadSpec
        Feature column names.
    values_out : numpy.ndarray
        Length-``F`` output values (mutated).
    masks_out : numpy.ndarray
        Length-``F`` output masks (mutated).
    """
    for feat_idx, name in enumerate(load_spec.feature_names):
        value, observed = _parse_numeric_cell(row.get(name, ""))
        values_out[feat_idx] = value
        masks_out[feat_idx] = observed


def _parse_numeric_cell(raw: str) -> tuple[float, float]:
    """Parse one CSV cell into ``(value, mask_flag)``.

    Parameters
    ----------
    raw : str
        Cell text; empty / NA tokens are native missing.

    Returns
    -------
    value : float
        Parsed float or ``MISSING_VALUE_SENTINEL``.
    observed : float
        ``MASK_OBSERVED`` or ``MASK_MISSING``.
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


def _resample_hourly(
        entity_id: str,
        stamps: np.ndarray,
        values: np.ndarray,
        masks: np.ndarray,
) -> HourlySiteSeries:
    """Bin rows onto a contiguous hourly grid (last observed wins).

    Parameters
    ----------
    entity_id : str
        Site id.
    stamps : numpy.ndarray
        Source timestamps.
    values : numpy.ndarray
        Source values ``(n_rows, F)``.
    masks : numpy.ndarray
        Source native masks ``(n_rows, F)``.

    Returns
    -------
    HourlySiteSeries
        Dense hourly grid with empty bins marked missing.
    """
    order = np.argsort(stamps, kind="mergesort")
    stamps_sorted = stamps[order]
    values_sorted = values[order]
    masks_sorted = masks[order]
    buffers = _empty_hourly_grid(
        stamps_sorted[0],
        stamps_sorted[-1],
        int(values.shape[1]),
    )
    _scatter_last_observed(
        stamps_sorted,
        values_sorted,
        masks_sorted,
        buffers,
    )
    return HourlySiteSeries(
        entity_id=entity_id,
        timestamps=buffers.grid,
        values=buffers.values,
        native_mask=buffers.masks,
    )


def _empty_hourly_grid(
        start: np.datetime64,
        stop: np.datetime64,
        n_features: int,
) -> _GridBuffers:
    """Allocate a dense hourly grid filled with native missing.

    Parameters
    ----------
    start : numpy.datetime64
        Inclusive start hour.
    stop : numpy.datetime64
        Inclusive stop hour.
    n_features : int
        Feature count ``F``.

    Returns
    -------
    _GridBuffers
        Grid axis plus zeroed / missing value-mask buffers.
    """
    grid = np.arange(start, stop + _HOUR_DELTA, _HOUR_DELTA)
    out_values = np.full(
        (grid.size, n_features),
        MISSING_VALUE_SENTINEL,
        dtype=np.float32,
    )
    out_masks = np.full(
        (grid.size, n_features),
        MASK_MISSING,
        dtype=np.float32,
    )
    return _GridBuffers(grid=grid, values=out_values, masks=out_masks)


def _scatter_last_observed(
        stamps: np.ndarray,
        values: np.ndarray,
        masks: np.ndarray,
        buffers: _GridBuffers,
) -> None:
    """Scatter source rows onto ``buffers``; later observed values overwrite.

    Parameters
    ----------
    stamps : numpy.ndarray
        Sorted source timestamps.
    values : numpy.ndarray
        Sorted source values.
    masks : numpy.ndarray
        Sorted source native masks.
    buffers : _GridBuffers
        Destination grid mutated in place.
    """
    start = buffers.grid[0]
    for row_idx, ts in enumerate(stamps):
        grid_idx = int((ts - start) / _HOUR_DELTA)
        observed = masks[row_idx] >= (MASK_OBSERVED - _MASK_TOL)
        buffers.values[grid_idx, observed] = values[row_idx, observed]
        buffers.masks[grid_idx, observed] = MASK_OBSERVED
