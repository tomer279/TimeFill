"""Training package for M1 conditional denoising score matching (WP4).

Classes
-------
DSMLossSettings
    Hyperparameters for conditional continuous DSM.
ConditionalDSMLoss
    CSDI-style masked DSM under the VP-SDE marginal.
ArtifactPaths
    Checkpoint / curve paths under ``artifacts/m1/``.
TrainHistory
    In-memory train/val loss rows with CSV / plot export.
ValSnapshotPayload
    Packed inputs for the periodic val snapshot hook.
TrainerBundle
    Dependencies for ``M1Trainer``.
M1Trainer
    Jitted Optax AdamW loop with checkpointing and val stub hook.

Functions
---------
dsm_settings_from_config
    Build ``DSMLossSettings`` from a validated ``ToyM1Config``.
build_optimizer
    AdamW (+ optional global grad clip) from ``TrainConfig``.
save_checkpoint
    Serialize an Equinox score net to disk.
load_checkpoint
    Restore a score net into a like-shaped template.
resolve_artifact_paths
    Build ``ArtifactPaths`` from a config ``artifact_dir`` string.
stub_val_imputation_hook
    WP5-pending val snapshot writer (no reverse SDE).
run_m1_training
    Convenience entrypoint: config path → trained model + artifacts.
"""

from src.train.loss import (
    ConditionalDSMLoss,
    DSMLossSettings,
    dsm_settings_from_config,
)
from src.train.loop import (
    ArtifactPaths,
    M1Trainer,
    TrainHistory,
    TrainerBundle,
    ValSnapshotPayload,
    build_optimizer,
    load_checkpoint,
    resolve_artifact_paths,
    run_m1_training,
    save_checkpoint,
    stub_val_imputation_hook,
)

__all__ = [
    "ArtifactPaths",
    "ConditionalDSMLoss",
    "DSMLossSettings",
    "M1Trainer",
    "TrainHistory",
    "TrainerBundle",
    "ValSnapshotPayload",
    "build_optimizer",
    "dsm_settings_from_config",
    "load_checkpoint",
    "resolve_artifact_paths",
    "run_m1_training",
    "save_checkpoint",
    "stub_val_imputation_hook",
]
