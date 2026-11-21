"""YAML config loading for the M1 toy path.

Classes
-------
DataConfig
    Window, feature, and masking hyperparameters.
SdeConfig
    VP-SDE beta schedule and diffusion horizon.
TrainOptimizerConfig
    Learning rate, weight decay, and grad clipping.
TrainScheduleConfig
    Epoch / step / logging schedule knobs.
TrainProtocolLossExtras
    M7.5 loss / schedule / selection flags.
TrainProtocolMaskExtras
    M7.5 artificial-mask protocol flags.
TrainProtocolExtras
    Bundles loss + mask extras under ``TrainProtocolConfig``.
TrainProtocolConfig
    Seed, DSM protocol, mask mode, and artifact root.
TrainConfig
    Nested optimizer / schedule / protocol train section.
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
from typing import TYPE_CHECKING, Any, Mapping

import yaml

if TYPE_CHECKING:
    from src.data.schema import TimeSeriesBatch

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
_DEFAULT_PARAMETERIZATION = "score"
_DEFAULT_OBSERVED_INPUT = "noised"
_DEFAULT_MIXED_SELECT = "per_batch"
_DEFAULT_MAX_BLOCKS_PER_FEATURE = 1
_DEFAULT_LR_SCHEDULE = "constant"
_DEFAULT_WARMUP_STEPS = 0
_DEFAULT_SELECT_BY = "val_dsm_loss"
_DEFAULT_OBSERVED_MODE = "frozen_clean"
_DEFAULT_DIFFUSION_IMPL = "dense"
_DEFAULT_POINT_ESTIMATE = "mean"
_ALLOWED_PARAMETERIZATIONS = frozenset({"score", "eps"})
_ALLOWED_OBSERVED_INPUT = frozenset({"noised", "clean"})
_ALLOWED_MIXED_SELECT = frozenset({"per_batch", "per_window"})
_ALLOWED_LR_SCHEDULE = frozenset({"constant", "warmup_cosine"})
_ALLOWED_SELECT_BY = frozenset({
    "val_dsm_loss",
    "val_impute_crps",
    "val_impute_mae",
})
_ALLOWED_OBSERVED_MODE = frozenset({"frozen_clean", "replace_noised"})
_ALLOWED_DIFFUSION_IMPL = frozenset({"dense", "diagonal"})
_ALLOWED_POINT_ESTIMATE = frozenset({"mean", "median"})


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
    t_diff : float
        Diffusion time horizon (YAML key ``T_diff``).
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
        self.t_diff = float(mapping["T_diff"])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable dict of fields.

        Returns
        -------
        dict of str to Any
            Field name to value (YAML key ``T_diff``).
        """
        return {
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
            "T_diff": self.t_diff,
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
        if self.t_diff <= 0.0:
            raise ValueError(f"t_diff must be positive, got {self.t_diff}")


class TrainOptimizerConfig:
    """Optimizer knobs inside the flat YAML ``train`` section.

    Public methods
    --------------
    to_dict
        Serialize fields to a plain dict.
    validate
        Check positivity / ranges for optimizer knobs.

    Instance variables
    ------------------
    lr : float
        Optimizer learning rate.
    weight_decay : float
        AdamW weight decay.
    grad_clip_norm : float
        Global grad-norm clip; ``<= 0`` disables clipping.
    """

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """Initialize optimizer fields from a flat ``train`` mapping.

        Parameters
        ----------
        mapping : Mapping[str, Any]
            Must contain ``lr``; other keys are optional with defaults.
        """
        self.lr = float(mapping["lr"])
        self.weight_decay = float(
            mapping.get("weight_decay", _DEFAULT_WEIGHT_DECAY)
        )
        self.grad_clip_norm = float(
            mapping.get("grad_clip_norm", _DEFAULT_GRAD_CLIP_NORM)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable dict of optimizer fields.

        Returns
        -------
        dict of str to Any
            Field name to value.
        """
        return {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "grad_clip_norm": self.grad_clip_norm,
        }

    def validate(self) -> None:
        """Raise `ValueError` if optimizer constraints are violated."""
        if self.lr <= 0.0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be >= 0, got {self.weight_decay}"
            )


class TrainScheduleConfig:
    """Epoch / logging schedule knobs inside the flat YAML ``train`` section.

    Public methods
    --------------
    to_dict
        Serialize fields to a plain dict.
    validate
        Check positivity of schedule knobs.

    Instance variables
    ------------------
    epochs : int
        Number of training epochs.
    steps_per_epoch : int
        Mini-batches drawn per epoch.
    val_every : int
        Validate / checkpoint every this many epochs.
    log_every : int
        Print train metrics every this many epochs.
    """

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """Initialize schedule fields from a flat ``train`` mapping.

        Parameters
        ----------
        mapping : Mapping[str, Any]
            Must contain ``epochs``; other keys are optional with defaults.
        """
        self.epochs = int(mapping["epochs"])
        self.steps_per_epoch = int(
            mapping.get("steps_per_epoch", _DEFAULT_STEPS_PER_EPOCH)
        )
        self.val_every = int(mapping.get("val_every", _DEFAULT_VAL_EVERY))
        self.log_every = int(mapping.get("log_every", _DEFAULT_LOG_EVERY))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable dict of schedule fields.

        Returns
        -------
        dict of str to Any
            Field name to value.
        """
        return {
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "val_every": self.val_every,
            "log_every": self.log_every,
        }

    def validate(self) -> None:
        """Raise `ValueError` if schedule constraints are violated."""
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
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


class TrainProtocolLossExtras:
    """M7.5 loss / schedule / selection flags.

    Public methods
    --------------
    to_dict
        Serialize fields for a flat YAML ``train`` section.
    validate
        Check allowlists and ranges.
    """

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """Parse loss-related M7.5 keys with legacy defaults."""
        self.parameterization = str(
            mapping.get("parameterization", _DEFAULT_PARAMETERIZATION)
        ).lower()
        self.observed_input = str(
            mapping.get("observed_input", _DEFAULT_OBSERVED_INPUT)
        ).lower()
        self.ema_decay = _optional_float(mapping.get("ema_decay", None))
        self.lr_schedule = str(
            mapping.get("lr_schedule", _DEFAULT_LR_SCHEDULE)
        ).lower()
        self.warmup_steps = int(
            mapping.get("warmup_steps", _DEFAULT_WARMUP_STEPS)
        )
        self.select_by = str(
            mapping.get("select_by", _DEFAULT_SELECT_BY)
        ).lower()

    def to_dict(self) -> dict[str, Any]:
        """Return JSON/YAML-serializable loss-extra fields."""
        return {
            "parameterization": self.parameterization,
            "observed_input": self.observed_input,
            "ema_decay": self.ema_decay,
            "lr_schedule": self.lr_schedule,
            "warmup_steps": self.warmup_steps,
            "select_by": self.select_by,
        }

    def validate(self) -> None:
        """Raise ``ValueError`` if loss-extra knobs are invalid."""
        if self.parameterization not in _ALLOWED_PARAMETERIZATIONS:
            raise ValueError(
                "parameterization must be one of "
                f"{sorted(_ALLOWED_PARAMETERIZATIONS)}, "
                f"got {self.parameterization!r}"
            )
        if self.observed_input not in _ALLOWED_OBSERVED_INPUT:
            raise ValueError(
                "observed_input must be one of "
                f"{sorted(_ALLOWED_OBSERVED_INPUT)}, "
                f"got {self.observed_input!r}"
            )
        if self.ema_decay is not None and not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(
                "ema_decay must be in [0, 1), "
                f"got {self.ema_decay}"
            )
        if self.lr_schedule not in _ALLOWED_LR_SCHEDULE:
            raise ValueError(
                "lr_schedule must be one of "
                f"{sorted(_ALLOWED_LR_SCHEDULE)}, "
                f"got {self.lr_schedule!r}"
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be >= 0, got {self.warmup_steps}"
            )
        if self.select_by not in _ALLOWED_SELECT_BY:
            raise ValueError(
                "select_by must be one of "
                f"{sorted(_ALLOWED_SELECT_BY)}, got {self.select_by!r}"
            )


class TrainProtocolMaskExtras:
    """M7.5 artificial-mask protocol flags.

    Public methods
    --------------
    to_dict
        Serialize fields for a flat YAML ``train`` section.
    validate
        Check allowlists and ranges.
    """

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """Parse mask-related M7.5 keys with legacy defaults."""
        self.mixed_select = str(
            mapping.get("mixed_select", _DEFAULT_MIXED_SELECT)
        ).lower()
        self.block_len_range = _optional_int_pair(
            mapping.get("block_len_range", None)
        )
        self.max_blocks_per_feature = int(
            mapping.get(
                "max_blocks_per_feature",
                _DEFAULT_MAX_BLOCKS_PER_FEATURE,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON/YAML-serializable mask-extra fields."""
        return {
            "mixed_select": self.mixed_select,
            "block_len_range": (
                None if self.block_len_range is None
                else list(self.block_len_range)
            ),
            "max_blocks_per_feature": self.max_blocks_per_feature,
        }

    def validate(self) -> None:
        """Raise ``ValueError`` if mask-extra knobs are invalid."""
        if self.mixed_select not in _ALLOWED_MIXED_SELECT:
            raise ValueError(
                "mixed_select must be one of "
                f"{sorted(_ALLOWED_MIXED_SELECT)}, "
                f"got {self.mixed_select!r}"
            )
        if self.block_len_range is not None:
            lo_bound, hi_bound = self.block_len_range
            if lo_bound <= 0 or hi_bound < lo_bound:
                raise ValueError(
                    "block_len_range must satisfy 0 < lo <= hi, "
                    f"got {self.block_len_range!r}"
                )
        if self.max_blocks_per_feature <= 0:
            raise ValueError(
                "max_blocks_per_feature must be positive, "
                f"got {self.max_blocks_per_feature}"
            )


class TrainProtocolExtras:
    """Milestone 7.5 protocol flags nested under ``TrainProtocolConfig``.

    Public methods
    --------------
    to_dict
        Serialize fields for a flat YAML ``train`` section.
    validate
        Validate nested loss and mask extras.
    """

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """Parse M7.5 protocol keys with legacy defaults.

        Parameters
        ----------
        mapping : Mapping[str, Any]
            Flat YAML ``train`` mapping.
        """
        self.loss = TrainProtocolLossExtras(mapping)
        self.mask = TrainProtocolMaskExtras(mapping)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable dict of M7.5 protocol fields."""
        merged: dict[str, Any] = {}
        merged.update(self.loss.to_dict())
        merged.update(self.mask.to_dict())
        return merged

    def validate(self) -> None:
        """Raise ``ValueError`` if M7.5 protocol knobs are invalid."""
        self.loss.validate()
        self.mask.validate()


class TrainProtocolConfig:
    """DSM protocol and artifact knobs inside the flat YAML ``train`` section.

    Public methods
    --------------
    to_dict
        Serialize fields to a plain dict.
    validate
        Check positivity / ranges for protocol knobs.

    Instance variables
    ------------------
    seed : int
        PRNG seed for data / masks / training noise.
    t_eps : float
        Lower bound for Uniform diffusion-time sampling.
    obs_loss_weight : float
        Weight on observed cells in DSM (M1 default ``0.0`` = zero).
    mask_mode : str
        ``random``, ``block``, or ``mixed`` self-supervised masks.
    artifact_dir : str
        Root directory for checkpoints and loss curves.
    extras : TrainProtocolExtras
        Milestone 7.5 flags (also exposed via properties below).
    """

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """Initialize protocol fields from a flat ``train`` mapping.

        Parameters
        ----------
        mapping : Mapping[str, Any]
            Optional keys with module-level defaults.
        """
        self.seed = int(mapping.get("seed", _DEFAULT_SEED))
        self.t_eps = float(mapping.get("t_eps", _DEFAULT_T_EPS))
        self.obs_loss_weight = float(
            mapping.get("obs_loss_weight", _DEFAULT_OBS_LOSS_WEIGHT)
        )
        self.mask_mode = str(
            mapping.get("mask_mode", _DEFAULT_MASK_MODE)
        ).lower()
        self.artifact_dir = str(
            mapping.get("artifact_dir", _DEFAULT_ARTIFACT_DIR)
        )
        self.extras = TrainProtocolExtras(mapping)

    @property
    def parameterization(self) -> str:
        """Forward ``extras.loss.parameterization`` for call-site parity."""
        return self.extras.loss.parameterization

    @property
    def observed_input(self) -> str:
        """Forward ``extras.loss.observed_input`` for call-site parity."""
        return self.extras.loss.observed_input

    @property
    def mixed_select(self) -> str:
        """Forward ``extras.mask.mixed_select`` for call-site parity."""
        return self.extras.mask.mixed_select

    @property
    def block_len_range(self) -> tuple[int, int] | None:
        """Forward ``extras.mask.block_len_range`` for call-site parity."""
        return self.extras.mask.block_len_range

    @property
    def max_blocks_per_feature(self) -> int:
        """Forward ``extras.mask.max_blocks_per_feature`` for call-site parity."""
        return self.extras.mask.max_blocks_per_feature

    @property
    def ema_decay(self) -> float | None:
        """Forward ``extras.loss.ema_decay`` for call-site parity."""
        return self.extras.loss.ema_decay

    @property
    def lr_schedule(self) -> str:
        """Forward ``extras.loss.lr_schedule`` for call-site parity."""
        return self.extras.loss.lr_schedule

    @property
    def warmup_steps(self) -> int:
        """Forward ``extras.loss.warmup_steps`` for call-site parity."""
        return self.extras.loss.warmup_steps

    @property
    def select_by(self) -> str:
        """Forward ``extras.loss.select_by`` for call-site parity."""
        return self.extras.loss.select_by

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable dict of protocol fields.

        Returns
        -------
        dict of str to Any
            Field name to value.
        """
        merged = {
            "seed": self.seed,
            "t_eps": self.t_eps,
            "obs_loss_weight": self.obs_loss_weight,
            "mask_mode": self.mask_mode,
            "artifact_dir": self.artifact_dir,
        }
        merged.update(self.extras.to_dict())
        return merged

    def validate(self) -> None:
        """Raise `ValueError` if protocol constraints are violated."""
        if self.t_eps <= 0.0:
            raise ValueError(f"t_eps must be positive, got {self.t_eps}")
        if self.obs_loss_weight < 0.0:
            raise ValueError(
                "obs_loss_weight must be >= 0, "
                f"got {self.obs_loss_weight}"
            )
        if self.mask_mode not in _ALLOWED_MASK_MODES:
            raise ValueError(
                "mask_mode must be one of "
                f"{sorted(_ALLOWED_MASK_MODES)}, got {self.mask_mode!r}"
            )
        if not self.artifact_dir:
            raise ValueError("artifact_dir must be a non-empty path string")
        self.extras.validate()


class TrainConfig:
    """Train section: nested optimizer, schedule, and protocol knobs.

    Public methods
    --------------
    to_dict
        Serialize nested fields to a flat dict matching YAML.
    validate
        Validate all nested train subsections.

    Instance variables
    ------------------
    optimizer : TrainOptimizerConfig
        Learning rate, weight decay, grad clip.
    schedule : TrainScheduleConfig
        Epochs, steps, val/log cadence.
    protocol : TrainProtocolConfig
        Seed, DSM weights, mask mode, artifacts.
    """

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """Initialize from a mapping (typically the YAML `train` section).

        Parameters
        ----------
        mapping : Mapping[str, Any]
            Must contain `_REQUIRED_TRAIN_KEYS` (flat layout).
        """
        _require_keys(mapping, _REQUIRED_TRAIN_KEYS, section="train")
        self.optimizer = TrainOptimizerConfig(mapping)
        self.schedule = TrainScheduleConfig(mapping)
        self.protocol = TrainProtocolConfig(mapping)

    def to_dict(self) -> dict[str, Any]:
        """Return a flat dict mirroring the YAML ``train`` layout.

        Returns
        -------
        dict of str to Any
            Merged optimizer / schedule / protocol fields.
        """
        merged: dict[str, Any] = {}
        merged.update(self.optimizer.to_dict())
        merged.update(self.schedule.to_dict())
        merged.update(self.protocol.to_dict())
        return merged

    def validate(self) -> None:
        """Raise `ValueError` if any nested train subsection is invalid."""
        self.optimizer.validate()
        self.schedule.validate()
        self.protocol.validate()


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
    k_samples : int
        Number of posterior imputation samples (YAML key ``K_samples``).
    solver_steps : int
        Diffrax / discrete reverse solver steps.
    t_eps : float or None
        Sampler endpoint; ``None`` means fall back to ``train.t_eps``.
    observed_mode : str
        ``frozen_clean`` (legacy) or ``replace_noised``.
    diffusion_impl : str
        ``dense`` (legacy) or ``diagonal``.
    point_estimate : str
        ``mean`` (legacy) or ``median`` for MAE reporting.
    """

    def __init__(self, mapping: Mapping[str, Any]) -> None:
        """Initialize from a mapping (typically the YAML `sample` section).

        Parameters
        ----------
        mapping : Mapping[str, Any]
            Must contain `_REQUIRED_SAMPLE_KEYS`.
        """
        _require_keys(mapping, _REQUIRED_SAMPLE_KEYS, section="sample")
        self.k_samples = int(mapping["K_samples"])
        self.solver_steps = int(mapping["solver_steps"])
        self.t_eps = _optional_float(mapping.get("t_eps", None))
        self.observed_mode = str(
            mapping.get("observed_mode", _DEFAULT_OBSERVED_MODE)
        ).lower()
        self.diffusion_impl = str(
            mapping.get("diffusion_impl", _DEFAULT_DIFFUSION_IMPL)
        ).lower()
        self.point_estimate = str(
            mapping.get("point_estimate", _DEFAULT_POINT_ESTIMATE)
        ).lower()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable dict of fields.

        Returns
        -------
        dict of str to Any
            Field name to value (YAML key ``K_samples``).
        """
        return {
            "K_samples": self.k_samples,
            "solver_steps": self.solver_steps,
            "t_eps": self.t_eps,
            "observed_mode": self.observed_mode,
            "diffusion_impl": self.diffusion_impl,
            "point_estimate": self.point_estimate,
        }

    def validate(self) -> None:
        """Raise `ValueError` if sample constraints are violated."""
        if self.k_samples <= 0:
            raise ValueError(
                f"k_samples must be positive, got {self.k_samples}"
            )
        if self.solver_steps <= 0:
            raise ValueError(
                f"solver_steps must be positive, got {self.solver_steps}"
            )
        if self.t_eps is not None and self.t_eps <= 0.0:
            raise ValueError(f"sample t_eps must be positive, got {self.t_eps}")
        if self.observed_mode not in _ALLOWED_OBSERVED_MODE:
            raise ValueError(
                "observed_mode must be one of "
                f"{sorted(_ALLOWED_OBSERVED_MODE)}, "
                f"got {self.observed_mode!r}"
            )
        if self.diffusion_impl not in _ALLOWED_DIFFUSION_IMPL:
            raise ValueError(
                "diffusion_impl must be one of "
                f"{sorted(_ALLOWED_DIFFUSION_IMPL)}, "
                f"got {self.diffusion_impl!r}"
            )
        if self.point_estimate not in _ALLOWED_POINT_ESTIMATE:
            raise ValueError(
                "point_estimate must be one of "
                f"{sorted(_ALLOWED_POINT_ESTIMATE)}, "
                f"got {self.point_estimate!r}"
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
    # Lazy import breaks the config ↔ schema cycle that confuses Pylance.
    from src.data.schema import build_stub_batch

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


def _optional_float(value: Any) -> float | None:
    """Parse an optional YAML float (``None`` stays ``None``)."""
    if value is None:
        return None
    return float(value)


def _optional_int_pair(value: Any) -> tuple[int, int] | None:
    """Parse an optional ``[lo, hi]`` integer pair from YAML."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            f"block_len_range must be a length-2 list, got {value!r}"
        )
    return int(value[0]), int(value[1])
