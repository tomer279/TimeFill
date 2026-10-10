"""Shared specs and path helpers for M2 real-data loading.

Classes
-------
BeijingLoadSpec
    Frozen column / feature names for UCI Beijing Multi-Site CSVs.
WindowSpec
    Fixed-length window geometry (`seq_len`, non-overlapping `stride`).
SplitFractions
    Deterministic train / val / test fractions plus RNG seed.
RealDataPaths
    Raw / processed directory layout from `configs/real_m2.yaml`.

Functions
---------
default_beijing_features
    Frozen M2 pollutant feature names (`F = 6`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_N_TIME_PARTS = 4
_DEFAULT_FEATURES = (
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
)
_DEFAULT_TIME_COLUMNS = ("year", "month", "day", "hour")
_DEFAULT_STATION_COLUMN = "station"

PROCESSED_VALUES_NAME = "values"
PROCESSED_MASKS_NAME = "native_masks"
PROCESSED_ENTITY_NAME = "entity_ids"
PROCESSED_TRAIN_NAME = "train_idx"
PROCESSED_VAL_NAME = "val_idx"
PROCESSED_TEST_NAME = "test_idx"
PROCESSED_ARRAYS_NAME = "windows.npz"


def default_beijing_features() -> tuple[str, ...]:
    """Return the frozen M2 pollutant feature list (`F = 6`).

    Returns
    -------
    tuple of str
        ``PM2.5``, ``PM10``, ``SO2``, ``NO2``, ``CO``, ``O3``.
    """
    return _DEFAULT_FEATURES


@dataclass(frozen=True)
class BeijingLoadSpec:
    """Column contract for UCI Beijing Multi-Site air-quality CSVs.

    Public methods
    --------------
    n_features
        Return ``F = len(feature_names)``.
    validate
        Check that feature / time column names are non-empty.

    Instance variables
    ------------------
    feature_names : tuple of str
        Pollutant columns to load (order defines feature axis).
    time_columns : tuple of str
        ``(year, month, day, hour)`` column names.
    station_column : str
        Entity / site id column name.
    """

    feature_names: tuple[str, ...] = _DEFAULT_FEATURES
    time_columns: tuple[str, ...] = _DEFAULT_TIME_COLUMNS
    station_column: str = _DEFAULT_STATION_COLUMN

    def n_features(self) -> int:
        """Return the feature count ``F``.

        Returns
        -------
        int
            ``len(feature_names)``.
        """
        return len(self.feature_names)

    def validate(self) -> None:
        """Raise ``ValueError`` if the load spec is inconsistent."""
        if not self.feature_names:
            raise ValueError("feature_names must be non-empty")
        if len(self.time_columns) != _N_TIME_PARTS:
            raise ValueError(
                "time_columns must be (year, month, day, hour); "
                f"got {self.time_columns!r}"
            )
        if not self.station_column:
            raise ValueError("station_column must be a non-empty string")


@dataclass(frozen=True)
class WindowSpec:
    """Fixed-length windowing on a regular hourly grid.

    Public methods
    --------------
    n_steps_per_window
        Return ``seq_len``.
    validate
        Check positive geometry and ``stride <= seq_len``.

    Instance variables
    ------------------
    seq_len : int
        Window length ``T``.
    stride : int
        Step between window starts; ``stride == seq_len`` ⇒ no overlap.
    """

    seq_len: int
    stride: int

    def n_steps_per_window(self) -> int:
        """Return the window length ``T``.

        Returns
        -------
        int
            Same as ``seq_len``.
        """
        return int(self.seq_len)

    def validate(self) -> None:
        """Raise ``ValueError`` if window geometry is invalid."""
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        if self.stride > self.seq_len:
            raise ValueError(
                "stride must be <= seq_len for M2 non-leakage defaults; "
                f"got stride={self.stride}, seq_len={self.seq_len}"
            )


@dataclass(frozen=True)
class SplitFractions:
    """Train / val / test fractions with a deterministic seed.

    Public methods
    --------------
    total
        Sum of the three fractions.
    validate
        Check positivity and that fractions sum to one.

    Instance variables
    ------------------
    train : float
        Train fraction in ``(0, 1]``.
    val : float
        Val fraction in ``[0, 1)``.
    test : float
        Test fraction in ``[0, 1)``.
    seed : int
        RNG seed for shuffling entities / indices.
    """

    train: float
    val: float
    test: float
    seed: int

    def total(self) -> float:
        """Return ``train + val + test``.

        Returns
        -------
        float
            Sum of split fractions.
        """
        return float(self.train + self.val + self.test)

    def validate(self) -> None:
        """Raise ``ValueError`` if fractions are outside the M2 contract."""
        if self.train <= 0.0:
            raise ValueError(f"train fraction must be > 0, got {self.train}")
        if self.val < 0.0 or self.test < 0.0:
            raise ValueError("val and test fractions must be >= 0")
        if abs(self.total() - 1.0) > 1e-6:
            raise ValueError(
                "train + val + test must equal 1.0, "
                f"got {self.total()}"
            )


@dataclass(frozen=True)
class RealDataPaths:
    """Filesystem layout for raw Beijing CSVs and processed windows.

    Public methods
    --------------
    resolve_raw_files
        Expand ``raw_glob`` under ``raw_dir``.
    ensure_processed_dir
        Create ``processed_dir`` if needed and return it.

    Instance variables
    ------------------
    raw_dir : pathlib.Path
        Directory of per-site CSVs.
    processed_dir : pathlib.Path
        Output directory for NPZ bundles.
    raw_glob : str
        Glob for site CSV files (default ``*.csv``).
    """

    raw_dir: Path
    processed_dir: Path
    raw_glob: str = "*.csv"

    def resolve_raw_files(self) -> list[Path]:
        """List raw CSV paths in sorted order.

        Returns
        -------
        list of pathlib.Path
            Matching files under ``raw_dir``.

        Raises
        ------
        FileNotFoundError
            If ``raw_dir`` is missing or the glob matches nothing.
        """
        raw_dir = Path(self.raw_dir)
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"raw_dir not found: {raw_dir}")
        paths = sorted(raw_dir.glob(self.raw_glob))
        if not paths:
            raise FileNotFoundError(
                f"no files matching {self.raw_glob!r} under {raw_dir}"
            )
        return paths

    def ensure_processed_dir(self) -> Path:
        """Create the processed directory and return it.

        Returns
        -------
        pathlib.Path
            ``processed_dir`` (created if absent).
        """
        out_dir = Path(self.processed_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
