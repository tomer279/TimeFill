"""Training package for M1 conditional denoising score matching (WP4).

Modules
-------
artifacts
    Artifact paths, train history, checkpoints, and val snapshot stubs.
loss
    Conditional continuous DSM loss and settings.
loop
    Jitted Optax trainer, optimizer builder, and train entrypoint.
real_loss
    M2 DSM with native mask composition and artificial-target weights.
real_pool
    Normalized train/val window pools and processed-NPZ loading.
real_val
    Lightweight reverse-impute validation snapshot hooks.
real_loop
    M2 real-window trainer and train entrypoint.

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
RealDSMBatch
    Real-window DSM batch (``x0``, ``m_native``).
RealConditionalDSMLoss
    M2 conditional DSM with ``MaskComposer``.
M2TrainerBundle
    Dependencies for ``M2Trainer``.
M2Trainer
    Real-window AdamW loop with checkpoints and val impute snapshots.

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
dsm_settings_from_real_config
    Build ``DSMLossSettings`` from ``RealM2Config``.
run_m2_training
    Convenience entrypoint for processed real windows → ``artifacts/m2/``.
"""

from src.train.artifacts import (
    ArtifactPaths,
    TrainHistory,
    ValSnapshotPayload,
    load_checkpoint,
    resolve_artifact_paths,
    save_checkpoint,
    stub_val_imputation_hook,
)
from src.train.loss import (
    ConditionalDSMLoss,
    DSMLossSettings,
    dsm_settings_from_config,
)
from src.train.loop import (
    M1Trainer,
    TrainerBundle,
    build_optimizer,
    run_m1_training,
)
from src.train.real_loss import (
    RealConditionalDSMLoss,
    RealDSMBatch,
    dsm_settings_from_real_config,
)
from src.train.real_loop import (
    M2Trainer,
    M2TrainerBundle,
    run_m2_training,
)

__all__ = [
    "ArtifactPaths",
    "ConditionalDSMLoss",
    "DSMLossSettings",
    "M1Trainer",
    "M2Trainer",
    "M2TrainerBundle",
    "RealConditionalDSMLoss",
    "RealDSMBatch",
    "TrainHistory",
    "TrainerBundle",
    "ValSnapshotPayload",
    "build_optimizer",
    "dsm_settings_from_config",
    "dsm_settings_from_real_config",
    "load_checkpoint",
    "resolve_artifact_paths",
    "run_m1_training",
    "run_m2_training",
    "save_checkpoint",
    "stub_val_imputation_hook",
]
