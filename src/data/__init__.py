"""Data contracts, config loading, synthetic generators, and masking for M1.

Classes
-------
DataConfig
    Typed `data` section of `configs/toy_m1.yaml`.
SdeConfig
    Typed `sde` section of the toy config.
TrainOptimizerConfig
    Learning rate, weight decay, and grad clipping.
TrainScheduleConfig
    Epoch / step / logging schedule knobs.
TrainProtocolConfig
    Seed, DSM protocol, mask mode, and artifact root.
TrainConfig
    Nested optimizer / schedule / protocol train section.
SampleConfig
    Typed `sample` section of the toy config.
ToyM1Config
    Full nested toy config object.
TimeSeriesBatch
    Batch container for `(x, m)` with shape helpers.
TensorSchema
    Validator for `(B, T, F)` value/mask contracts.
SineWindowGenerator
    Multi-frequency sine window factory.
ArWindowGenerator
    AR(1) / diagonal VAR(1) window factory.
RandomMasker
    MCAR Bernoulli masker.
BlockMasker
    Contiguous time-block masker.
BeijingCsvLoader
    Load Beijing Multi-Site CSVs onto hourly grids (M2).
RealWindowBuilder
    Cut fixed-length real windows `(N, T, F)`.
EntityAwareSplitter
    Deterministic entity / index train-val-test splits.
MaskComposer
    Compose native and artificial masks for M2 training.
RealWindowDataset
    Stacked real windows with native masks and entity ids.
SplitIndices
    Disjoint train / val / test index arrays.
FeatureScaler
    Per-feature mean/std fit on train observed cells (M2).
RealDataPaths
    Path knobs nested under ``RealDataConfig``.
RealDataWindow
    Window / feature knobs nested under ``RealDataConfig``.
RealDataBatchMask
    Batch / mask knobs nested under ``RealDataConfig``.
RealDataConfig
    Typed ``data`` section of ``configs/real_m2.yaml``.
RealM2Config
    Full nested real-m2 config object.

Functions
---------
load_config
    Load `toy_m1.yaml` (or compatible YAML) into `ToyM1Config`.
load_real_m2_config
    Load `real_m2.yaml` into `RealM2Config` (allows ``F=6``).
build_stub_batch
    Deterministic fake batch honoring schema contracts (pre-WP1).
load_toy_m1
    Load config plus a stub batch with documented shapes.
apply_missing_sentinel
    Zero-fill missing cells according to the mask.
generate_sine_windows
    Clean sine `x0` shaped `(B, T, F)`.
generate_ar_windows
    Clean AR/VAR `x0` shaped `(B, T, F)`.
random_mask
    MCAR mask → `(x_obs, m)`.
block_mask
    Contiguous block mask → `(x_obs, m)`.
compose_train_mask
    `m_train = m_native * m_artificial`.
restrict_artificial_mask
    Never mark natively missing cells as artificially observed.
artificial_target_mask
    Targets with ground truth held out by artificial masks.
default_beijing_features
    Frozen M2 pollutant feature names (`F = 6`).
save_processed_dataset
    Write real windows + split indices to processed NPZ.
load_processed_dataset
    Load a previously prepared M2 NPZ bundle.
save_mask_comparison
    Side-by-side random vs block mask PNG helper.
resolve_scaler_path
    ``artifacts/m2/checkpoints/feature_scaler.npz`` helper.
fit_scaler_on_train_split
    Fit ``FeatureScaler`` on train indices only.
default_scaler_filename
    Locked scaler NPZ basename.
"""

from src.data.config import (
    DataConfig,
    SampleConfig,
    SdeConfig,
    ToyM1Config,
    TrainConfig,
    TrainOptimizerConfig,
    TrainProtocolConfig,
    TrainScheduleConfig,
    load_config,
    load_toy_m1,
)
from src.data.real_config import (
    RealDataBatchMask,
    RealDataConfig,
    RealDataPaths,
    RealDataWindow,
    RealM2Config,
    load_real_m2_config,
)
from src.data.mask_plots import save_mask_comparison
from src.data.masking import (
    BlockMasker,
    RandomMasker,
    block_mask,
    random_mask,
)
from src.data.normalize import (
    FeatureScaler,
    default_scaler_filename,
    fit_scaler_on_train_split,
    resolve_scaler_path,
)
from src.data.real import (
    BeijingCsvLoader,
    EntityAwareSplitter,
    MaskComposer,
    RealWindowBuilder,
    RealWindowDataset,
    SplitIndices,
    artificial_target_mask,
    compose_train_mask,
    default_beijing_features,
    load_processed_dataset,
    restrict_artificial_mask,
    save_processed_dataset,
)
from src.data.schema import (
    MASK_MISSING,
    MASK_OBSERVED,
    MISSING_VALUE_SENTINEL,
    TensorSchema,
    TimeSeriesBatch,
    apply_missing_sentinel,
    build_stub_batch,
)
from src.data.synthetic import (
    ArWindowGenerator,
    SineWindowGenerator,
    generate_ar_windows,
    generate_sine_windows,
)

__all__ = [
    "ArWindowGenerator",
    "BeijingCsvLoader",
    "BlockMasker",
    "DataConfig",
    "EntityAwareSplitter",
    "FeatureScaler",
    "MASK_MISSING",
    "MASK_OBSERVED",
    "MISSING_VALUE_SENTINEL",
    "MaskComposer",
    "RandomMasker",
    "RealDataBatchMask",
    "RealDataConfig",
    "RealDataPaths",
    "RealDataWindow",
    "RealM2Config",
    "RealWindowBuilder",
    "RealWindowDataset",
    "SampleConfig",
    "SdeConfig",
    "SineWindowGenerator",
    "SplitIndices",
    "TensorSchema",
    "TimeSeriesBatch",
    "ToyM1Config",
    "TrainConfig",
    "TrainOptimizerConfig",
    "TrainProtocolConfig",
    "TrainScheduleConfig",
    "apply_missing_sentinel",
    "artificial_target_mask",
    "block_mask",
    "build_stub_batch",
    "compose_train_mask",
    "default_beijing_features",
    "default_scaler_filename",
    "fit_scaler_on_train_split",
    "generate_ar_windows",
    "generate_sine_windows",
    "load_config",
    "load_processed_dataset",
    "load_real_m2_config",
    "load_toy_m1",
    "random_mask",
    "resolve_scaler_path",
    "restrict_artificial_mask",
    "save_mask_comparison",
    "save_processed_dataset",
]
