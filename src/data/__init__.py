"""Data contracts, config loading, synthetic generators, and masking for M1.

Classes
-------
DataConfig
    Typed `data` section of `configs/toy_m1.yaml`.
SdeConfig
    Typed `sde` section of the toy config.
TrainConfig
    Typed `train` section of the toy config.
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

Functions
---------
load_config
    Load `toy_m1.yaml` (or compatible YAML) into `ToyM1Config`.
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
save_mask_comparison
    Side-by-side random vs block mask PNG helper.
"""

from src.data.config import (
    DataConfig,
    SampleConfig,
    SdeConfig,
    ToyM1Config,
    TrainConfig,
    load_config,
    load_toy_m1,
)
from src.data.mask_plots import save_mask_comparison
from src.data.masking import (
    BlockMasker,
    RandomMasker,
    block_mask,
    random_mask,
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
    "BlockMasker",
    "DataConfig",
    "MASK_MISSING",
    "MASK_OBSERVED",
    "MISSING_VALUE_SENTINEL",
    "RandomMasker",
    "SampleConfig",
    "SdeConfig",
    "SineWindowGenerator",
    "TensorSchema",
    "TimeSeriesBatch",
    "ToyM1Config",
    "TrainConfig",
    "apply_missing_sentinel",
    "block_mask",
    "build_stub_batch",
    "generate_ar_windows",
    "generate_sine_windows",
    "load_config",
    "load_toy_m1",
    "random_mask",
    "save_mask_comparison",
]
