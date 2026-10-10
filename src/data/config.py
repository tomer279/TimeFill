"""YAML config loading for the M1 toy path.

Classes
-------
DataConfig
    Window, feature, and masking hyperparameters.
SdeConfig
    VP-SDE beta schedule and diffusion horizon.
TrainConfig
    Optimizer / loop hyperparameters.
SampleConfig
    Reverse-SDE sampling hyperparameters.
ToyM1Config
    Nested root config matching `configs/toy_m1.yaml`.

Functions
---------
load_config
    Parse a YAML file into `ToyM1Config`.
load_toy_m1
    Load config and a schema-valid stub batch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from src.data.schema import TimeSeriesBatch, build_stub_batch

_REQUIRED_ROOT_KEYS = ("data", "sde", "train", "sample")
_REQUIRED_DATA_KEYS = (
    "seq_len",
    "n_features",
    "mask_rate",
    "block_len",
    "batch_size",
)
_REQUIRED_SDE_KEYS = ("beta_min", "beta_max", "T_diff")
_REQUIRED_TRAIN_KEYS = ("lr", "epochs")
_DEFAULT_WEIGHT_DECAY = 1e-4
_DEFAULT_SEED = 0
_DEFAULT_T_EPS = 1e-5
_DEFAULT_OBS_LOSS_WEIGHT = 0.0
_DEFAULT_STEPS_PER_EPOCH = 20
_DEFAULT_VAL_EVERY = 5
_DEFAULT_LOG_EVERY = 1
_DEFAULT_GRAD_CLIP_NORM = 1.0
_DEFAULT_MASK_MODE = "mixed"
_DEFAULT_ARTIFACT_DIR = "artifacts/m1"
_ALLOWED_MASK_MODES = frozenset({"random", "block", "mixed"})
_REQUIRED_SAMPLE_KEYS = ("K_samples", "solver_steps")


class DataConfig:
    """Data section: windowing, features, and mask protocol knobs.

    Public methods
    --------------
    to_dict
        Serialize fields to a plain dict.
    validate
        Check M1 ranges for `F`, rates, and block length.

    Instance variables
    ------------------
    seq_len : int
        Window length `T`.
    n_features : int
        Feature count `F`.
    mask_rate : float
        MCAR drop rate in `[0, 1]`.
    block_len : int
        Contiguous missing block length.
    batch_size : int
        Default batch size `B` for loaders / stubs.
    """

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """Initialize from a mapping (typically the YAML `data` section).

        Parameters
        ----------
        mapping : Mapping[str, Any]
            Must contain `_REQUIRED_DATA_KEYS`.
        """
        _require_keys(mapping, _REQUIRED_DATA_KEYS, section="data")
        self.seq_len = int(mapping["seq_len"])
        self.n_features = int(mapping["n_features"])
        self.mask_rate = float(mapping["mask_rate"])
        self.block_len = int(mapping["block_len"])
        self.batch_size = int(mapping["batch_size"])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable dict of fields.

        Returns
        -------
        dict of str to Any
            Field name to value.
        """
        return {
            "seq_len": self.seq_len,
            "n_features": self.n_features,
            "mask_rate": self.mask_rate,
            "block_len": self.block_len,
            "batch_size": self.batch_size,
        }

    def validate(self) -> None:
        """Raise `ValueError` if M1 data constraints are violated."""
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.n_features not in (1, 2, 3):
            raise ValueError(
                "n_features must be in {1, 2, 3}, "
                f"got {self.n_features}"
            )
        if not 0.0 <= self.mask_rate <= 1.0:
            raise ValueError(
                f"mask_rate must be in [0, 1], got {self.mask_rate}"
            )
        if self.block_len <= 0 or self.block_len > self.seq_len:
            raise ValueError(
                "block_len must be in [1, seq_len], "
                f"got {self.block_len}"
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {self.batch_size}"
            )


class SdeConfig:
    """SDE section: VP noise schedule and diffusion horizon.

    Public methods
    --------------
    to_dict
        Serialize fields to a plain dict.
    validate
        Check beta ordering and positive horizon.

    Instance variables
    ------------------
    beta_min : float
        Lower end of linear `β(t)`.
    beta_max : float
        Upper end of linear `β(t)`.
    T_diff : float
        Diffusion time horizon.
    """

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """Initialize from a mapping (typically the YAML `sde` section).

        Parameters
        ----------
        mapping : Mapping[str, Any]
            Must contain `_REQUIRED_SDE_KEYS`.
        """
        _require_keys(mapping, _REQUIRED_SDE_KEYS, section="sde")
        self.beta_min = float(mapping["beta_min"])
        self.beta_max = float(mapping["beta_max"])
        self.T_diff = float(mapping["T_diff"])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable dict of fields.

        Returns
        -------
        dict of str to Any
            Field name to value.
        """
        return {
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
            "T_diff": self.T_diff,
        }

    def validate(self) -> None:
        """Raise `ValueError` if SDE constraints are violated."""
        if self.beta_min <= 0.0 or self.beta_max <= 0.0:
            raise ValueError("beta_min and beta_max must be positive")
        if self.beta_min >= self.beta_max:
            raise ValueError(
                "beta_min must be < beta_max, "
                f"got {self.beta_min} >= {self.beta_max}"
            )
        if self.T_diff <= 0.0:
            raise ValueError(f"T_diff must be positive, got {self.T_diff}")


class TrainConfig:
    """Train section: optimizer, DSM, and loop hyperparameters.

    Public methods
    --------------
    to_dict
        Serialize fields to a plain dict.
    validate
        Check positivity / ranges for all train knobs.

    Instance variables
    ------------------
    lr : float
        Optimizer learning rate.
    epochs : int
        Number of training epochs.
    weight_decay : float
        AdamW weight decay.
    seed : int
        PRNG seed for data / masks / training noise.
    t_eps : float
        Lower bound for Uniform diffusion-time sampling.
    obs_loss_weight : float
        Weight on observed cells in DSM (M1 default ``0.0`` = zero).
    steps_per_epoch : int
        Mini-batches drawn per epoch.
    val_every : int
        Validate / checkpoint every this many epochs.
    log_every : int
        Print train metrics every this many epochs.
    grad_clip_norm : float
        Global grad-norm clip; ``<= 0`` disables clipping.
    mask_mode : str
        ``random``, ``block``, or ``mixed`` self-supervised masks.
    artifact_dir : str
        Root directory for checkpoints and loss curves.
    """

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """Initialize from a mapping (typically the YAML `train` section).

        Parameters
        ----------
        mapping : Mapping[str, Any]
            Must contain `_REQUIRED_TRAIN_KEYS`.
        """
        _require_keys(mapping, _REQUIRED_TRAIN_KEYS, section="train")
        self.lr = float(mapping["lr"])
        self.epochs = int(mapping["epochs"])
        self.weight_decay = float(
            mapping.get("weight_decay", _DEFAULT_WEIGHT_DECAY)
        )
        self.seed = int(mapping.get("seed", _DEFAULT_SEED))
        self.t_eps = float(mapping.get("t_eps", _DEFAULT_T_EPS))
        self.obs_loss_weight = float(
            mapping.get("obs_loss_weight", _DEFAULT_OBS_LOSS_WEIGHT)
        )
        self.steps_per_epoch = int(
            mapping.get("steps_per_epoch", _DEFAULT_STEPS_PER_EPOCH)
        )
        self.val_every = int(mapping.get("val_every", _DEFAULT_VAL_EVERY))
        self.log_every = int(mapping.get("log_every", _DEFAULT_LOG_EVERY))
        self.grad_clip_norm = float(
            mapping.get("grad_clip_norm", _DEFAULT_GRAD_CLIP_NORM)
        )
        self.mask_mode = str(
            mapping.get("mask_mode", _DEFAULT_MASK_MODE)
        ).lower()
        self.artifact_dir = str(
            mapping.get("artifact_dir", _DEFAULT_ARTIFACT_DIR)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable dict of fields.

        Returns
        -------
        dict of str to Any
            Field name to value.
        """
        return {
            "lr": self.lr,
            "epochs": self.epochs,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "t_eps": self.t_eps,
            "obs_loss_weight": self.obs_loss_weight,
            "steps_per_epoch": self.steps_per_epoch,
            "val_every": self.val_every,
            "log_every": self.log_every,
            "grad_clip_norm": self.grad_clip_norm,
            "mask_mode": self.mask_mode,
            "artifact_dir": self.artifact_dir,
        }

    def validate(self) -> None:
        """Raise `ValueError` if train constraints are violated."""
        if self.lr <= 0.0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be >= 0, got {self.weight_decay}"
            )
        if self.t_eps <= 0.0:
            raise ValueError(f"t_eps must be positive, got {self.t_eps}")
        if self.obs_loss_weight < 0.0:
            raise ValueError(
                "obs_loss_weight must be >= 0, "
                f"got {self.obs_loss_weight}"
            )
        if self.steps_per_epoch <= 0:
            raise ValueError(
                "steps_per_epoch must be positive, "
                f"got {self.steps_per_epoch}"
            )
        if self.val_every <= 0:
            raise ValueError(
                f"val_every must be positive, got {self.val_every}"
            )
        if self.log_every <= 0:
            raise ValueError(
                f"log_every must be positive, got {self.log_every}"
            )
        if self.mask_mode not in _ALLOWED_MASK_MODES:
            raise ValueError(
                "mask_mode must be one of "
                f"{sorted(_ALLOWED_MASK_MODES)}, got {self.mask_mode!r}"
            )
        if not self.artifact_dir:
            raise ValueError("artifact_dir must be a non-empty path string")


class SampleConfig:
    """Sample section: reverse-SDE imputation hyperparameters.

    Public methods
    --------------
    to_dict
        Serialize fields to a plain dict.
    validate
        Check positivity of sample count and solver steps.

    Instance variables
    ------------------
    K_samples : int
        Number of posterior imputation samples.
    solver_steps : int
        Diffrax / discrete reverse solver steps.
    """

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """Initialize from a mapping (typically the YAML `sample` section).

        Parameters
        ----------
        mapping : Mapping[str, Any]
            Must contain `_REQUIRED_SAMPLE_KEYS`.
        """
        _require_keys(mapping, _REQUIRED_SAMPLE_KEYS, section="sample")
        self.K_samples = int(mapping["K_samples"])
        self.solver_steps = int(mapping["solver_steps"])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable dict of fields.

        Returns
        -------
        dict of str to Any
            Field name to value.
        """
        return {
            "K_samples": self.K_samples,
            "solver_steps": self.solver_steps,
        }

    def validate(self) -> None:
        """Raise `ValueError` if sample constraints are violated."""
        if self.K_samples <= 0:
            raise ValueError(
                f"K_samples must be positive, got {self.K_samples}"
            )
        if self.solver_steps <= 0:
            raise ValueError(
                f"solver_steps must be positive, got {self.solver_steps}"
            )


class ToyM1Config:
    """Root config object for `configs/toy_m1.yaml`.

    Public methods
    --------------
    to_dict
        Serialize the nested config to a plain dict.
    validate
        Validate all nested sections.

    Instance variables
    ------------------
    data : DataConfig
        Data / masking section.
    sde : SdeConfig
        Forward SDE section.
    train : TrainConfig
        Training loop section.
    sample : SampleConfig
        Reverse sampling section.
    """

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """Initialize from a full YAML root mapping.

        Parameters
        ----------
        mapping : Mapping[str, Any]
            Must contain `data`, `sde`, `train`, and `sample` sections.
        """
        _require_keys(mapping, _REQUIRED_ROOT_KEYS, section="root")
        self.data = DataConfig(mapping["data"])
        self.sde = SdeConfig(mapping["sde"])
        self.train = TrainConfig(mapping["train"])
        self.sample = SampleConfig(mapping["sample"])

    def to_dict(self) -> dict[str, Any]:
        """Return a nested dict mirroring the YAML layout.

        Returns
        -------
        dict of str to Any
            Sections `data`, `sde`, `train`, `sample`.
        """
        return {
            "data": self.data.to_dict(),
            "sde": self.sde.to_dict(),
            "train": self.train.to_dict(),
            "sample": self.sample.to_dict(),
        }

    def validate(self) -> None:
        """Validate all nested config sections in place."""
        self.data.validate()
        self.sde.validate()
        self.train.validate()
        self.sample.validate()


def load_config(path: Path | str) -> ToyM1Config:
    """Load and validate a toy M1 YAML config file.

    Parameters
    ----------
    path : path-like
        Path to a YAML file with `data` / `sde` / `train` / `sample`.

    Returns
    -------
    ToyM1Config
        Typed nested configuration.
    """
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"config root must be a mapping, got {type(raw)!r}"
        )
    config = ToyM1Config(raw)
    config.validate()
    return config


def load_toy_m1(path: Path | str) -> tuple[ToyM1Config, TimeSeriesBatch]:
    """Load config and a deterministic stub batch with contracted shapes.

    Parameters
    ----------
    path : path-like
        Path to `configs/toy_m1.yaml` (or compatible).

    Returns
    -------
    config : ToyM1Config
        Validated configuration.
    batch : TimeSeriesBatch
        Stub `(x, m)` batch with shapes `(B, T, F)`.
    """
    config = load_config(path)
    batch = build_stub_batch(config.data)
    return config, batch


def _require_keys(
    mapping: Mapping[str, Any],
    keys: tuple[str, ...],
    section: str,
) -> None:
    """Ensure a config mapping contains every required key.

    Parameters
    ----------
    mapping : Mapping[str, Any]
        Section (or root) mapping loaded from YAML.
    keys : tuple of str
        Keys that must be present in ``mapping``.
    section : str
        Section name used only in the error message (e.g. ``"data"``).

    Raises
    ------
    KeyError
        If one or more of ``keys`` are absent from ``mapping``.
    """
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(
            f"config section '{section}' missing keys: {missing}"
        )
