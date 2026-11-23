"""YAML config loading for the M2 real-data training path.

Classes
-------
RealDataPaths
    Raw / processed / scaler path knobs from the ``data`` section.
RealDataWindow
    Window geometry and feature-list knobs.
RealDataTransform
    Optional log1p / clip knobs (parse-only until WP5).
RealDataBatchMask
    Batch size and artificial-mask knobs.
RealDataSplit
    Split mode / seed / fractions and empty-feature fallback.
RealDataConfig
    Nested ``data`` section facade (flat property accessors).
RealTrainExtras
    WP3 snapshot / sensitivity knobs from the flat ``train`` section.
RealM2Config
    Nested root config matching ``configs/real_m2.yaml``.

Functions
---------
load_real_m2_config
    Parse and validate a real-m2 YAML file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.data.config import (
    SampleConfig,
    SdeConfig,
    TrainConfig,
    _require_keys,
)
from src.data.real.specs import default_beijing_features
from src.models.score_net import ScoreNetConfig

_REQUIRED_ROOT_KEYS = ("data", "sde", "train", "sample")
_REQUIRED_DATA_KEYS = (
    "seq_len",
    "n_features",
    "mask_rate",
    "block_len",
    "batch_size",
    "processed_dir",
)
_DEFAULT_RAW_DIR = "data/raw/beijing_multisite"
_DEFAULT_RAW_GLOB = "*.csv"
_DEFAULT_STRIDE = 64
_DEFAULT_SCALER_NAME = "feature_scaler.npz"
_DEFAULT_SNAPSHOT_K = 4
_DEFAULT_SNAPSHOT_STEPS = 20
_DEFAULT_SPLIT_MODE = "entity"
_DEFAULT_SPLIT_SEED = 0
_DEFAULT_TRAIN_FRACTION = 0.70
_DEFAULT_VAL_FRACTION = 0.15
_DEFAULT_TEST_FRACTION = 0.15
_ALLOWED_MASK_MODES = frozenset({"random", "block", "mixed"})
_ALLOWED_SPLIT_MODES = frozenset({"entity", "index"})


class RealDataPaths:
    """Filesystem knobs inside the YAML ``data`` section.

    Public methods
    --------------
    to_dict
        Serialize fields to a plain dict.
    validate
        Check that path strings are non-empty.
    resolve_processed_dir
        Absolute processed NPZ directory relative to a repo root.

    Instance variables
    ------------------
    processed_dir : str
        Directory containing ``windows.npz``.
    raw_dir : str
        Raw CSV cache (documentation / prepare parity).
    raw_glob : str
        Glob for site CSVs.
    scaler_filename : str
        Scaler NPZ basename beside checkpoints.
    """

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """Initialize path fields from a ``data`` mapping.

        Parameters
        ----------
        mapping : Mapping[str, Any]
            Must contain ``processed_dir``; other keys optional.
        """
        self.processed_dir = str(mapping["processed_dir"])
        self.raw_dir = str(mapping.get("raw_dir", _DEFAULT_RAW_DIR))
        self.raw_glob = str(mapping.get("raw_glob", _DEFAULT_RAW_GLOB))
        self.scaler_filename = str(
            mapping.get("scaler_filename", _DEFAULT_SCALER_NAME)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable dict of path fields.

        Returns
        -------
        dict of str to Any
            Field name to value.
        """
        return {
            "processed_dir": self.processed_dir,
            "raw_dir": self.raw_dir,
            "raw_glob": self.raw_glob,
            "scaler_filename": self.scaler_filename,
        }

    def validate(self) -> None:
        """Raise ``ValueError`` if path knobs are invalid."""
        if not self.processed_dir:
            raise ValueError("processed_dir must be a non-empty path string")
        if not self.raw_dir:
            raise ValueError("raw_dir must be a non-empty path string")
        if not self.raw_glob:
            raise ValueError("raw_glob must be a non-empty glob string")
        if not self.scaler_filename:
            raise ValueError("scaler_filename must be a non-empty string")

    def resolve_processed_dir(self, repo_root: Path) -> Path:
        """Resolve ``processed_dir`` against ``repo_root`` when relative.

        Parameters
        ----------
        repo_root : pathlib.Path
            Repository root used for relative paths.

        Returns
        -------
        pathlib.Path
            Absolute processed directory.
        """
        path = Path(self.processed_dir)
        if path.is_absolute():
            return path
        return Path(repo_root) / path


class RealDataWindow:
    """Window geometry and feature list inside the YAML ``data`` section.

    Public methods
    --------------
    to_dict
        Serialize fields to a plain dict.
    validate
        Check positive geometry and feature-list consistency.

    Instance variables
    ------------------
    seq_len : int
        Window length ``T``.
    stride : int
        Window stride used at prepare time.
    n_features : int
        Feature count ``F`` (M2 primary: 6).
    feature_names : tuple of str
        Ordered pollutant channel names.
    """

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """Initialize window fields from a ``data`` mapping.

        Parameters
        ----------
        mapping : Mapping[str, Any]
            Must contain ``seq_len`` and ``n_features``.
        """
        self.seq_len = int(mapping["seq_len"])
        self.stride = int(mapping.get("stride", _DEFAULT_STRIDE))
        self.n_features = int(mapping["n_features"])
        names = mapping.get("feature_names", list(default_beijing_features()))
        self.feature_names = tuple(str(name) for name in names)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable dict of window fields.

        Returns
        -------
        dict of str to Any
            Field name to value.
        """
        return {
            "seq_len": self.seq_len,
            "stride": self.stride,
            "n_features": self.n_features,
            "feature_names": list(self.feature_names),
        }

    def validate(self) -> None:
        """Raise ``ValueError`` if window knobs are invalid."""
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        if self.n_features <= 0:
            raise ValueError(
                f"n_features must be positive, got {self.n_features}"
            )
        if len(self.feature_names) != self.n_features:
            raise ValueError(
                "len(feature_names) must equal n_features, "
                f"got {len(self.feature_names)} vs {self.n_features}"
            )


class RealDataTransform:
    """Optional per-feature log1p / clip knobs for ``FeatureScaler``.

    Public methods
    --------------
    to_dict
        Serialize fields to a plain dict.
    validate
        No-op placeholder for API parity.

    Instance variables
    ------------------
    log1p_features : tuple of str
        Feature names to ``log1p`` before z-score (legacy empty).
    clip_nonnegative : bool
        Clip restored values at 0 (legacy ``False``).
    """

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """Initialize from a ``data.transform`` mapping (or empty)."""
        names = mapping.get("log1p_features", []) or []
        self.log1p_features = tuple(str(name) for name in names)
        self.clip_nonnegative = bool(mapping.get("clip_nonnegative", False))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable dict of transform fields."""
        return {
            "log1p_features": list(self.log1p_features),
            "clip_nonnegative": self.clip_nonnegative,
        }

    def validate(self) -> None:
        """Validate transform knobs (names may be empty)."""
        if any(not name for name in self.log1p_features):
            raise ValueError("log1p_features entries must be non-empty strings")


class RealDataSplit:
    """Train/val/test split knobs and empty-feature fallback.

    Public methods
    --------------
    to_dict
        Serialize fields for a flat YAML ``data`` section.
    validate
        Check mode allowlist and fraction sum.

    Instance variables
    ------------------
    split_mode : str
        ``entity`` or ``index``.
    split_seed : int
        Split RNG seed.
    train_fraction : float
        Train split fraction.
    val_fraction : float
        Val split fraction.
    test_fraction : float
        Test split fraction.
    allow_empty_feature_fallback : bool
        Legacy silent identity scaler when a feature has no train obs.
    """

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """Parse split knobs from a flat ``data`` mapping."""
        self.split_mode = str(
            mapping.get("split_mode", _DEFAULT_SPLIT_MODE)
        ).lower()
        self.split_seed = int(mapping.get("split_seed", _DEFAULT_SPLIT_SEED))
        self.train_fraction = float(
            mapping.get("train_fraction", _DEFAULT_TRAIN_FRACTION)
        )
        self.val_fraction = float(
            mapping.get("val_fraction", _DEFAULT_VAL_FRACTION)
        )
        self.test_fraction = float(
            mapping.get("test_fraction", _DEFAULT_TEST_FRACTION)
        )
        self.allow_empty_feature_fallback = bool(
            mapping.get("allow_empty_feature_fallback", True)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable dict of split fields."""
        return {
            "split_mode": self.split_mode,
            "split_seed": self.split_seed,
            "train_fraction": self.train_fraction,
            "val_fraction": self.val_fraction,
            "test_fraction": self.test_fraction,
            "allow_empty_feature_fallback": (
                self.allow_empty_feature_fallback
            ),
        }

    def validate(self) -> None:
        """Raise ``ValueError`` if split knobs are invalid."""
        if self.split_mode not in _ALLOWED_SPLIT_MODES:
            raise ValueError(
                "split_mode must be one of "
                f"{sorted(_ALLOWED_SPLIT_MODES)}, got {self.split_mode!r}"
            )
        frac_sum = (
            self.train_fraction + self.val_fraction + self.test_fraction
        )
        if abs(frac_sum - 1.0) > 1e-6:
            raise ValueError(
                "train/val/test fractions must sum to 1.0, "
                f"got {frac_sum}"
            )


class RealDataBatchMask:
    """Batch size and artificial-mask knobs inside the YAML ``data`` section.

    Public methods
    --------------
    to_dict
        Serialize fields to a plain dict.
    validate
        Check rates / lengths against a window length ``T``.

    Instance variables
    ------------------
    mask_rate : float
        Artificial MCAR rate on natively observed cells.
    block_len : int
        Contiguous artificial block length.
    batch_size : int
        Mini-batch size ``B``.
    """

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """Initialize batch/mask fields from a ``data`` mapping.

        Parameters
        ----------
        mapping : Mapping[str, Any]
            Must contain ``mask_rate``, ``block_len``, ``batch_size``.
        """
        self.mask_rate = float(mapping["mask_rate"])
        self.block_len = int(mapping["block_len"])
        self.batch_size = int(mapping["batch_size"])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable dict of batch/mask fields.

        Returns
        -------
        dict of str to Any
            Field name to value.
        """
        return {
            "mask_rate": self.mask_rate,
            "block_len": self.block_len,
            "batch_size": self.batch_size,
        }

    def validate(self, seq_len: int) -> None:
        """Raise ``ValueError`` if batch/mask knobs are invalid.

        Parameters
        ----------
        seq_len : int
            Window length ``T`` used to bound ``block_len``.
        """
        if not 0.0 <= self.mask_rate <= 1.0:
            raise ValueError(
                f"mask_rate must be in [0, 1], got {self.mask_rate}"
            )
        if self.block_len <= 0 or self.block_len > seq_len:
            raise ValueError(
                "block_len must be in [1, seq_len], "
                f"got {self.block_len}"
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {self.batch_size}"
            )


class RealDataConfig:
    """M2 ``data`` section: nested paths / window / batch-mask groups.

    Flat YAML keys remain available as properties for call-site parity
    with the M1 ``DataConfig`` style (``config.data.batch_size``, etc.).

    Public methods
    --------------
    to_dict
        Flatten nested groups back to a YAML-shaped dict.
    validate
        Validate all nested groups.
    resolve_processed_dir
        Delegate to ``paths.resolve_processed_dir``.

    Instance variables
    ------------------
    paths : RealDataPaths
        Raw / processed / scaler paths.
    window : RealDataWindow
        ``T``, stride, ``F``, feature names.
    batch : RealDataBatchMask
        ``batch_size``, ``mask_rate``, ``block_len``.
    transform : RealDataTransform
        Optional log1p / clip knobs (WP5).
    split : RealDataSplit
        Split mode / seed / fractions and empty-feature fallback.
    """

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """Initialize from the YAML ``data`` mapping.

        Parameters
        ----------
        mapping : Mapping[str, Any]
            Must contain ``_REQUIRED_DATA_KEYS``.
        """
        _require_keys(mapping, _REQUIRED_DATA_KEYS, section="data")
        self.paths = RealDataPaths(mapping)
        self.window = RealDataWindow(mapping)
        self.batch = RealDataBatchMask(mapping)
        transform_raw = mapping.get("transform") or {}
        if not isinstance(transform_raw, Mapping):
            raise ValueError("data.transform must be a mapping when set")
        self.transform = RealDataTransform(transform_raw)
        self.split = RealDataSplit(mapping)

    @property
    def split_mode(self) -> str:
        """Forward ``split.split_mode`` for call-site parity."""
        return self.split.split_mode

    @property
    def split_seed(self) -> int:
        """Forward ``split.split_seed`` for call-site parity."""
        return self.split.split_seed

    @property
    def train_fraction(self) -> float:
        """Forward ``split.train_fraction`` for call-site parity."""
        return self.split.train_fraction

    @property
    def val_fraction(self) -> float:
        """Forward ``split.val_fraction`` for call-site parity."""
        return self.split.val_fraction

    @property
    def test_fraction(self) -> float:
        """Forward ``split.test_fraction`` for call-site parity."""
        return self.split.test_fraction

    @property
    def allow_empty_feature_fallback(self) -> bool:
        """Forward ``split.allow_empty_feature_fallback`` for call-site parity."""
        return self.split.allow_empty_feature_fallback

    @property
    def seq_len(self) -> int:
        """Window length ``T``."""
        return self.window.seq_len

    @property
    def stride(self) -> int:
        """Window stride."""
        return self.window.stride

    @property
    def n_features(self) -> int:
        """Feature count ``F``."""
        return self.window.n_features

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Ordered pollutant channel names."""
        return self.window.feature_names

    @property
    def mask_rate(self) -> float:
        """Artificial MCAR rate."""
        return self.batch.mask_rate

    @property
    def block_len(self) -> int:
        """Contiguous artificial block length."""
        return self.batch.block_len

    @property
    def batch_size(self) -> int:
        """Mini-batch size ``B``."""
        return self.batch.batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        """Override mini-batch size (CLI / smoke tests).

        Parameters
        ----------
        value : int
            New batch size.
        """
        self.batch.batch_size = int(value)

    @property
    def processed_dir(self) -> str:
        """Processed NPZ directory string."""
        return self.paths.processed_dir

    @property
    def raw_dir(self) -> str:
        """Raw CSV directory string."""
        return self.paths.raw_dir

    @property
    def raw_glob(self) -> str:
        """Raw CSV glob pattern."""
        return self.paths.raw_glob

    @property
    def scaler_filename(self) -> str:
        """Scaler NPZ basename."""
        return self.paths.scaler_filename

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable dict of fields.

        Returns
        -------
        dict of str to Any
            Flattened field name to value.
        """
        merged: dict[str, Any] = {}
        merged.update(self.window.to_dict())
        merged.update(self.batch.to_dict())
        merged.update(self.paths.to_dict())
        merged["transform"] = self.transform.to_dict()
        merged.update(self.split.to_dict())
        return merged

    def validate(self) -> None:
        """Raise ``ValueError`` if M2 data constraints are violated."""
        self.paths.validate()
        self.window.validate()
        self.batch.validate(self.window.seq_len)
        self.transform.validate()
        unknown = set(self.transform.log1p_features) - set(self.feature_names)
        if unknown:
            raise ValueError(
                "data.transform.log1p_features not in feature_names: "
                f"{sorted(unknown)}"
            )
        self.split.validate()

    def resolve_processed_dir(self, repo_root: Path) -> Path:
        """Resolve ``processed_dir`` against ``repo_root`` when relative.

        Parameters
        ----------
        repo_root : pathlib.Path
            Repository root used for relative paths.

        Returns
        -------
        pathlib.Path
            Absolute processed directory.
        """
        return self.paths.resolve_processed_dir(repo_root)


class RealTrainExtras:
    """Optional WP3 train knobs (snapshot impute + sensitivity smoke).

    Public methods
    --------------
    to_dict
        Serialize fields to a plain dict.
    validate
        Check positivity / rate list contents.

    Instance variables
    ------------------
    snapshot_k_samples : int
        ``K`` for periodic val reverse-impute snapshots.
    snapshot_solver_steps : int
        Diffrax steps for those lightweight snapshots.
    sensitivity_mask_rates : tuple of float
        Optional MCAR rates for a post-train smoke (empty = disabled).
    """

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """Initialize from the flat YAML ``train`` mapping.

        Parameters
        ----------
        mapping : Mapping[str, Any]
            Optional WP3-only keys with defaults.
        """
        self.snapshot_k_samples = int(
            mapping.get("snapshot_k_samples", _DEFAULT_SNAPSHOT_K)
        )
        self.snapshot_solver_steps = int(
            mapping.get("snapshot_solver_steps", _DEFAULT_SNAPSHOT_STEPS)
        )
        rates = mapping.get("sensitivity_mask_rates", None)
        self.sensitivity_mask_rates = _parse_rate_list(rates)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable dict of fields.

        Returns
        -------
        dict of str to Any
            Field name to value.
        """
        return {
            "snapshot_k_samples": self.snapshot_k_samples,
            "snapshot_solver_steps": self.snapshot_solver_steps,
            "sensitivity_mask_rates": list(self.sensitivity_mask_rates),
        }

    def validate(self) -> None:
        """Raise ``ValueError`` if snapshot / sensitivity knobs are invalid."""
        if self.snapshot_k_samples <= 0:
            raise ValueError(
                "snapshot_k_samples must be positive, "
                f"got {self.snapshot_k_samples}"
            )
        if self.snapshot_solver_steps <= 0:
            raise ValueError(
                "snapshot_solver_steps must be positive, "
                f"got {self.snapshot_solver_steps}"
            )
        for rate in self.sensitivity_mask_rates:
            if not 0.0 <= rate <= 1.0:
                raise ValueError(
                    f"sensitivity mask rates must be in [0, 1], got {rate}"
                )


class RealM2Config:
    """Root config object for ``configs/real_m2.yaml``.

    Public methods
    --------------
    to_dict
        Serialize the nested config to a plain dict.
    validate
        Validate all nested sections.

    Instance variables
    ------------------
    data : RealDataConfig
        Real-data paths and geometry.
    sde : SdeConfig
        VP-SDE section (shared with M1).
    train : TrainConfig
        Nested optimizer / schedule / protocol (shared with M1).
    sample : SampleConfig
        Full reverse-sample knobs (WP4 primary; WP3 uses extras).
    extras : RealTrainExtras
        Snapshot / sensitivity knobs.
    model : ScoreNetConfig
        Optional ``model:`` section (legacy ``tcn_v1`` defaults).
    """

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """Initialize from a full YAML root mapping.

        Parameters
        ----------
        mapping : Mapping[str, Any]
            Must contain ``data``, ``sde``, ``train``, and ``sample``.
        """
        _require_keys(mapping, _REQUIRED_ROOT_KEYS, section="root")
        self.data = RealDataConfig(mapping["data"])
        self.sde = SdeConfig(mapping["sde"])
        self.train = TrainConfig(mapping["train"])
        self.sample = SampleConfig(mapping["sample"])
        self.extras = RealTrainExtras(mapping["train"])
        model_raw = dict(mapping.get("model") or {})
        model_raw.setdefault("n_features", self.data.n_features)
        self.model = ScoreNetConfig.from_dict(model_raw)

    def to_dict(self) -> dict[str, Any]:
        """Return a nested dict mirroring the YAML layout.

        Returns
        -------
        dict of str to Any
            Sections ``data``, ``sde``, ``train``, ``sample`` (+ extras
            merged into ``train``).
        """
        train_dict = self.train.to_dict()
        train_dict.update(self.extras.to_dict())
        return {
            "data": self.data.to_dict(),
            "sde": self.sde.to_dict(),
            "train": train_dict,
            "sample": self.sample.to_dict(),
            "model": self.model.to_dict(),
        }

    def validate(self) -> None:
        """Validate all nested config sections in place."""
        self.data.validate()
        self.sde.validate()
        self.train.validate()
        self.sample.validate()
        self.extras.validate()
        self.model.validate()
        if self.model.n_features != self.data.n_features:
            raise ValueError(
                "model.n_features must match data.n_features, "
                f"got {self.model.n_features} vs {self.data.n_features}"
            )
        if self.train.protocol.mask_mode not in _ALLOWED_MASK_MODES:
            raise ValueError(
                f"unsupported mask_mode={self.train.protocol.mask_mode!r}"
            )


def load_real_m2_config(path: Path | str) -> RealM2Config:
    """Load and validate a real-m2 YAML config file.

    Parameters
    ----------
    path : path-like
        Path to ``configs/real_m2.yaml`` (or compatible).

    Returns
    -------
    RealM2Config
        Typed nested configuration.

    Notes
    -----
    Do **not** use ``src.data.config.load_config`` for this file: M1
    ``DataConfig.validate`` rejects ``n_features=6``.
    """
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"config root must be a mapping, got {type(raw)!r}"
        )
    config = RealM2Config(raw)
    config.validate()
    return config


def _parse_rate_list(rates: Any) -> tuple[float, ...]:
    """Parse an optional YAML list of mask rates.

    Parameters
    ----------
    rates : Any
        ``None``, empty, or a sequence of floats.

    Returns
    -------
    tuple of float
        Parsed rates (possibly empty).
    """
    if rates is None:
        return ()
    if not isinstance(rates, Sequence) or isinstance(rates, (str, bytes)):
        raise ValueError(
            "sensitivity_mask_rates must be a list of floats or null"
        )
    return tuple(float(rate) for rate in rates)
