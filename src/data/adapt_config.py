"""YAML config loading for Milestone 7 upload-adapt (smoke caps).

Classes
-------
AdaptCaps
    Runtime / leakage caps from the YAML ``adapt`` section.
AdaptModeSection
    Fine-tune vs scratch policy defaults from YAML.
AdaptGeometry
    Window / batch geometry for adapt DSM.
AdaptSampleSection
    Reverse-SDE ``K`` / steps / seeds for post-adapt impute.
AdaptWindowCliOverrides
    Optional CLI overrides for window / batch geometry.
AdaptLoopCliOverrides
    Optional CLI overrides for train schedule / sample / timeout.
AdaptCliOverrides
    Bundled CLI smoke-cap overrides (``window`` + ``loop``).
AdaptM7Config
    Root smoke-cap config (nested; ≤7 instance variables).

Functions
---------
load_adapt_m7_config
    Parse and validate ``configs/adapt_m7.yaml``.
apply_adapt_cli_overrides
    Return a copy of ``AdaptM7Config`` with CLI overrides applied.
allowed_adapt_modes
    Frozen set of legal adapt mode strings.
default_adapt_mode
    Product default mode string (``scratch``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.data.config import SdeConfig, TrainConfig, _require_keys

_MODE_SCRATCH = "scratch"
_MODE_FINE_TUNE = "fine_tune"
_ALLOWED_MODES = frozenset({_MODE_SCRATCH, _MODE_FINE_TUNE})
_DEFAULT_BLOCK_LEN = 8
_DEFAULT_SEQ_LEN = 64
_DEFAULT_BATCH_SIZE = 4
_DEFAULT_MAX_WINDOWS = 4
_DEFAULT_K_SAMPLES = 8
_DEFAULT_SOLVER_STEPS = 50
_DEFAULT_SEED_MASK = 0
_DEFAULT_SEED_SAMPLE = 1
_REQUIRED_ROOT_KEYS = ("adapt", "train", "sde", "sample")


def allowed_adapt_modes() -> frozenset[str]:
    """Return the frozen set of legal adapt mode strings.

    Returns
    -------
    frozenset of str
        ``scratch`` and ``fine_tune``.
    """
    return _ALLOWED_MODES


def default_adapt_mode() -> str:
    """Return the product default adapt mode.

    Returns
    -------
    str
        ``scratch``.
    """
    return _MODE_SCRATCH


@dataclass(frozen=True)
class AdaptCaps:
    """Runtime / leakage caps from the YAML ``adapt`` section.

    Public methods
    --------------
    validate
        Check timeout, min windows, and fractions.
    to_dict
        Serialize fields for ``adapt_meta.json``.

    Instance variables
    ------------------
    artifact_dir : str
        Root under which ``<run_id>/`` is created.
    timeout_seconds : float
        Wall-clock abort budget.
    min_windows : int
        Minimum ``B`` after M6 window adapt.
    train_fraction : float
        Train share of windows (smoke = remainder).
    smoke_mcar_rate : float
        Artificial MCAR rate for DSM during adapt.
    """

    artifact_dir: str
    timeout_seconds: float
    min_windows: int
    train_fraction: float
    smoke_mcar_rate: float

    def validate(self) -> None:
        """Raise ``ValueError`` if runtime caps are invalid."""
        if self.timeout_seconds <= 0.0:
            raise ValueError(
                f"timeout_seconds must be positive, got {self.timeout_seconds}"
            )
        if self.min_windows < 2:
            raise ValueError(
                f"min_windows must be >= 2, got {self.min_windows}"
            )
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError(
                f"train_fraction must be in (0, 1), got {self.train_fraction}"
            )
        if not 0.0 <= self.smoke_mcar_rate <= 1.0:
            raise ValueError(
                "smoke_mcar_rate must be in [0, 1], "
                f"got {self.smoke_mcar_rate}"
            )
        if not self.artifact_dir:
            raise ValueError("artifact_dir must be a non-empty path string")

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-serializable dict of cap fields.

        Returns
        -------
        dict of str to float or int or str
            Cap field name to value.
        """
        return {
            "artifact_dir": self.artifact_dir,
            "timeout_seconds": self.timeout_seconds,
            "min_windows": self.min_windows,
            "train_fraction": self.train_fraction,
            "smoke_mcar_rate": self.smoke_mcar_rate,
        }


@dataclass(frozen=True)
class AdaptModeSection:
    """Fine-tune vs scratch policy knobs from YAML ``adapt``.

    Public methods
    --------------
    validate
        Check default mode string and base path.
    to_dict
        Serialize fields for metadata.

    Instance variables
    ------------------
    default_mode : str
        Product default when request omits mode.
    allow_fine_tune_when_f_matches : bool
        Gate for fine-tune path.
    require_feature_name_match_for_fine_tune : bool
        Name-equality policy from YAML.
    base_checkpoint : str
        Default base checkpoint path string.
    """

    default_mode: str
    allow_fine_tune_when_f_matches: bool
    require_feature_name_match_for_fine_tune: bool
    base_checkpoint: str

    def validate(self) -> None:
        """Raise ``ValueError`` if mode knobs are invalid."""
        mode = str(self.default_mode).strip().lower()
        if mode not in _ALLOWED_MODES:
            raise ValueError(
                f"default_mode must be one of {sorted(_ALLOWED_MODES)}, "
                f"got {self.default_mode!r}"
            )
        if not self.base_checkpoint:
            raise ValueError("base_checkpoint must be a non-empty path string")

    def to_dict(self) -> dict[str, bool | str]:
        """Return a JSON-serializable dict of mode fields.

        Returns
        -------
        dict of str to bool or str
            Mode field name to value.
        """
        return {
            "default_mode": self.default_mode,
            "allow_fine_tune_when_f_matches": (
                self.allow_fine_tune_when_f_matches
            ),
            "require_feature_name_match_for_fine_tune": (
                self.require_feature_name_match_for_fine_tune
            ),
            "base_checkpoint": self.base_checkpoint,
        }


@dataclass(frozen=True)
class AdaptGeometry:
    """Window / batch geometry for adapt DSM.

    Public methods
    --------------
    validate
        Check ``seq_len``, ``batch_size``, and ``block_len``.
    to_dict
        Serialize fields for metadata.

    Instance variables
    ------------------
    seq_len : int
        Expected window length ``T``.
    batch_size : int
        Mini-batch size ``B`` for DSM steps.
    block_len : int
        Block mask length (DSM); must be in ``[1, seq_len]``.
    max_windows : int or None
        Optional ingest window cap (``window.max_windows``).
    """

    seq_len: int
    batch_size: int
    block_len: int
    max_windows: int | None = None

    def validate(self) -> None:
        """Raise ``ValueError`` if geometry knobs are invalid."""
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {self.batch_size}"
            )
        if self.block_len <= 0 or self.block_len > self.seq_len:
            raise ValueError(
                "block_len must be in [1, seq_len], "
                f"got {self.block_len} (seq_len={self.seq_len})"
            )
        if self.max_windows is not None and self.max_windows <= 0:
            raise ValueError(
                "max_windows must be positive when set, "
                f"got {self.max_windows}"
            )

    def to_dict(self) -> dict[str, int | None]:
        """Return a JSON-serializable dict of geometry fields.

        Returns
        -------
        dict of str to int or None
            Geometry field name to value.
        """
        return {
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "block_len": self.block_len,
            "max_windows": self.max_windows,
        }


@dataclass(frozen=True)
class AdaptSampleSection:
    """Post-adapt reverse-SDE sample knobs from YAML.

    Public methods
    --------------
    validate
        Check positivity of ``K`` / steps.
    to_dict
        Serialize fields for metadata.

    Instance variables
    ------------------
    k_samples : int
        Posterior sample count ``K``.
    solver_steps : int
        Euler–Maruyama steps.
    seed_mask : int
        PRNG seed for optional smoke corruption.
    seed_sample : int
        PRNG seed for reverse-SDE sampling.
    """

    k_samples: int
    solver_steps: int
    seed_mask: int
    seed_sample: int

    def validate(self) -> None:
        """Raise ``ValueError`` if sample knobs are invalid."""
        if self.k_samples <= 0:
            raise ValueError(
                f"K_samples must be positive, got {self.k_samples}"
            )
        if self.solver_steps <= 0:
            raise ValueError(
                f"solver_steps must be positive, got {self.solver_steps}"
            )

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serializable dict of sample fields.

        Returns
        -------
        dict of str to int
            Sample field name to value.
        """
        return {
            "k_samples": self.k_samples,
            "solver_steps": self.solver_steps,
            "seed_mask": self.seed_mask,
            "seed_sample": self.seed_sample,
        }


@dataclass(frozen=True)
class AdaptWindowCliOverrides:
    """Optional CLI overrides for window / batch geometry.

    Public methods
    --------------
    has_any
        Whether at least one override is set.
    as_set_fields
        Names of override fields that are set.

    Instance variables
    ------------------
    seq_len : int or None
        Override ``window.seq_len``.
    max_windows : int or None
        Override ``window.max_windows``.
    batch_size : int or None
        Override ``train.batch_size``.
    """

    seq_len: int | None = None
    max_windows: int | None = None
    batch_size: int | None = None

    def has_any(self) -> bool:
        """Return whether any override field is set.

        Returns
        -------
        bool
            ``True`` if at least one field is not ``None``.
        """
        return len(self.as_set_fields()) > 0

    def as_set_fields(self) -> tuple[str, ...]:
        """Return names of override fields that are set.

        Returns
        -------
        tuple of str
            Field names with non-``None`` values.
        """
        mapping = {
            "seq_len": self.seq_len,
            "max_windows": self.max_windows,
            "batch_size": self.batch_size,
        }
        return tuple(
            name for name, value in mapping.items() if value is not None
        )


@dataclass(frozen=True)
class AdaptLoopCliOverrides:
    """Optional CLI overrides for train schedule / sample / timeout.

    Public methods
    --------------
    has_any
        Whether at least one override is set.
    as_set_fields
        Names of override fields that are set.

    Instance variables
    ------------------
    epochs : int or None
        Override ``train.epochs``.
    steps_per_epoch : int or None
        Override ``train.steps_per_epoch``.
    k_samples : int or None
        Override ``sample.K_samples``.
    solver_steps : int or None
        Override ``sample.solver_steps``.
    timeout_seconds : float or None
        Override ``adapt.timeout_seconds``.
    """

    epochs: int | None = None
    steps_per_epoch: int | None = None
    k_samples: int | None = None
    solver_steps: int | None = None
    timeout_seconds: float | None = None

    def has_any(self) -> bool:
        """Return whether any override field is set.

        Returns
        -------
        bool
            ``True`` if at least one field is not ``None``.
        """
        return len(self.as_set_fields()) > 0

    def as_set_fields(self) -> tuple[str, ...]:
        """Return names of override fields that are set.

        Returns
        -------
        tuple of str
            Field names with non-``None`` values.
        """
        mapping = {
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "k_samples": self.k_samples,
            "solver_steps": self.solver_steps,
            "timeout_seconds": self.timeout_seconds,
        }
        return tuple(
            name for name, value in mapping.items() if value is not None
        )


@dataclass(frozen=True)
class AdaptCliOverrides:
    """Bundled CLI smoke-cap overrides (WP2).

    Public methods
    --------------
    has_any
        Whether any nested override is set.
    as_set_fields
        Flattened names of set override fields.

    Instance variables
    ------------------
    window : AdaptWindowCliOverrides
        Geometry / batch overrides.
    loop : AdaptLoopCliOverrides
        Epoch / sample / timeout overrides.
    """

    window: AdaptWindowCliOverrides = AdaptWindowCliOverrides()
    loop: AdaptLoopCliOverrides = AdaptLoopCliOverrides()

    def has_any(self) -> bool:
        """Return whether any nested override is set.

        Returns
        -------
        bool
            ``True`` if window or loop has a set field.
        """
        return self.window.has_any() or self.loop.has_any()

    def as_set_fields(self) -> tuple[str, ...]:
        """Return flattened names of set override fields.

        Returns
        -------
        tuple of str
            Field names with non-``None`` values.
        """
        return self.window.as_set_fields() + self.loop.as_set_fields()


@dataclass(frozen=True)
class AdaptM7Config:
    """Root M7 adapt config (≤7 instance variables).

    Public methods
    --------------
    validate
        Validate all nested sections.
    to_train_config
        Return ``TrainConfig`` for ``build_optimizer``.

    Instance variables
    ------------------
    caps : AdaptCaps
        Timeout, min windows, train fraction, artifact root, MCAR rate.
    mode : AdaptModeSection
        Scratch / fine-tune policy defaults.
    geometry : AdaptGeometry
        ``T``, batch size, block length, optional max windows.
    train : TrainConfig
        Optimizer / schedule / protocol (shared M1/M2 type).
    sde : SdeConfig
        VP-SDE schedule.
    sample : AdaptSampleSection
        Post-adapt ``K`` / solver steps / seeds.
    """

    caps: AdaptCaps
    mode: AdaptModeSection
    geometry: AdaptGeometry
    train: TrainConfig
    sde: SdeConfig
    sample: AdaptSampleSection

    def validate(self) -> None:
        """Validate all nested adapt config sections."""
        self.caps.validate()
        self.mode.validate()
        self.geometry.validate()
        self.train.validate()
        self.sde.validate()
        self.sample.validate()

    def to_train_config(self) -> TrainConfig:
        """Return the train section used for Optax.

        Returns
        -------
        TrainConfig
            Optimizer / schedule / protocol bundle.
        """
        return self.train


def load_adapt_m7_config(path: Path | str) -> AdaptM7Config:
    """Load and validate an adapt-m7 YAML config file.

    Parameters
    ----------
    path : path-like
        Path to ``configs/adapt_m7.yaml`` (or compatible).

    Returns
    -------
    AdaptM7Config
        Typed smoke-cap configuration.
    """
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"config root must be a mapping, got {type(raw)!r}"
        )
    return _parse_adapt_m7_mapping(raw)


def apply_adapt_cli_overrides(
        config: AdaptM7Config,
        overrides: AdaptCliOverrides,
) -> AdaptM7Config:
    """Return ``config`` with optional CLI smoke-cap overrides.

    Parameters
    ----------
    config : AdaptM7Config
        Loaded YAML config.
    overrides : AdaptCliOverrides
        Fields set to non-``None`` replace YAML values.

    Returns
    -------
    AdaptM7Config
        Validated config (same object if no overrides).
    """
    if not overrides.has_any():
        return config
    window = overrides.window
    loop = overrides.loop
    seq_len = (
        int(config.geometry.seq_len)
        if window.seq_len is None
        else int(window.seq_len)
    )
    batch_size = (
        int(config.geometry.batch_size)
        if window.batch_size is None
        else int(window.batch_size)
    )
    max_windows = (
        config.geometry.max_windows
        if window.max_windows is None
        else int(window.max_windows)
    )
    block_len = min(int(config.geometry.block_len), seq_len)
    train_map = dict(config.train.to_dict())
    if loop.epochs is not None:
        train_map["epochs"] = int(loop.epochs)
    if loop.steps_per_epoch is not None:
        train_map["steps_per_epoch"] = int(loop.steps_per_epoch)
    timeout = (
        float(config.caps.timeout_seconds)
        if loop.timeout_seconds is None
        else float(loop.timeout_seconds)
    )
    k_samples = (
        int(config.sample.k_samples)
        if loop.k_samples is None
        else int(loop.k_samples)
    )
    solver_steps = (
        int(config.sample.solver_steps)
        if loop.solver_steps is None
        else int(loop.solver_steps)
    )
    updated = AdaptM7Config(
        caps=AdaptCaps(
            artifact_dir=config.caps.artifact_dir,
            timeout_seconds=timeout,
            min_windows=config.caps.min_windows,
            train_fraction=config.caps.train_fraction,
            smoke_mcar_rate=config.caps.smoke_mcar_rate,
        ),
        mode=config.mode,
        geometry=AdaptGeometry(
            seq_len=seq_len,
            batch_size=batch_size,
            block_len=block_len,
            max_windows=max_windows,
        ),
        train=TrainConfig(train_map),
        sde=config.sde,
        sample=AdaptSampleSection(
            k_samples=k_samples,
            solver_steps=solver_steps,
            seed_mask=config.sample.seed_mask,
            seed_sample=config.sample.seed_sample,
        ),
    )
    updated.validate()
    return updated


def _parse_adapt_m7_mapping(raw: Mapping[str, Any]) -> AdaptM7Config:
    """Build ``AdaptM7Config`` from a YAML root mapping.

    Parameters
    ----------
    raw : Mapping[str, Any]
        Parsed YAML root.

    Returns
    -------
    AdaptM7Config
        Validated config.
    """
    _require_keys(raw, _REQUIRED_ROOT_KEYS, section="root")
    adapt = raw["adapt"]
    window = raw.get("window", {})
    train_map = dict(raw["train"])
    batch_size = int(train_map.pop("batch_size", _DEFAULT_BATCH_SIZE))
    seq_len = int(window.get("seq_len", _DEFAULT_SEQ_LEN))
    max_windows_raw = window.get("max_windows", _DEFAULT_MAX_WINDOWS)
    max_windows = None if max_windows_raw is None else int(max_windows_raw)
    block_len = min(
        int(train_map.pop("block_len", _DEFAULT_BLOCK_LEN)),
        seq_len,
    )
    sample_raw = raw["sample"]
    seeds_raw = raw.get("seeds", {})
    if "artifact_dir" not in train_map:
        train_map["artifact_dir"] = str(
            adapt.get("artifact_dir", "artifacts/m7")
        )
    config = AdaptM7Config(
        caps=AdaptCaps(
            artifact_dir=str(adapt.get("artifact_dir", "artifacts/m7")),
            timeout_seconds=float(adapt.get("timeout_seconds", 600)),
            min_windows=int(adapt.get("min_windows", 2)),
            train_fraction=float(adapt.get("train_fraction", 0.80)),
            smoke_mcar_rate=float(adapt.get("smoke_mcar_rate", 0.30)),
        ),
        mode=AdaptModeSection(
            default_mode=str(adapt.get("default_mode", _MODE_SCRATCH)),
            allow_fine_tune_when_f_matches=bool(
                adapt.get("allow_fine_tune_when_f_matches", True)
            ),
            require_feature_name_match_for_fine_tune=bool(
                adapt.get("require_feature_name_match_for_fine_tune", True)
            ),
            base_checkpoint=str(
                adapt.get(
                    "base_checkpoint",
                    "artifacts/m2/checkpoints/best_score_net.eqx",
                )
            ),
        ),
        geometry=AdaptGeometry(
            seq_len=seq_len,
            batch_size=batch_size,
            block_len=block_len,
            max_windows=max_windows,
        ),
        train=TrainConfig(train_map),
        sde=SdeConfig(raw["sde"]),
        sample=AdaptSampleSection(
            k_samples=int(sample_raw.get("K_samples", _DEFAULT_K_SAMPLES)),
            solver_steps=int(
                sample_raw.get("solver_steps", _DEFAULT_SOLVER_STEPS)
            ),
            seed_mask=int(seeds_raw.get("mask", _DEFAULT_SEED_MASK)),
            seed_sample=int(seeds_raw.get("sample", _DEFAULT_SEED_SAMPLE)),
        ),
    )
    config.validate()
    return config
