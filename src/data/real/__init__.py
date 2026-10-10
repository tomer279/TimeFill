"""Real multivariate windows for Milestone 2 (Beijing air-quality primary).

Modules
-------
specs
    Load specs, window geometry, split fractions, and path helpers.
beijing
    Beijing CSV → hourly grid loader.
windows
    Fixed-length window stacking into `(N, T, F)`.
splits
    Deterministic entity / index train-val-test splits.
masks
    Native + artificial mask composition.
io
    Processed NPZ save / load.

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
HourlySiteSeries
    One monitoring site on a regular hourly grid with native masks.
RealWindowDataset
    Stacked windows `(N, T, F)` plus entity ids and native masks.
SplitIndices
    Disjoint train / val / test index arrays.
BeijingCsvLoader
    Parse site CSVs into hourly grids (last-value-in-bin; gaps → missing).
RealWindowBuilder
    Cut non-overlapping windows from hourly site series.
EntityAwareSplitter
    Entity- or index-based deterministic splits without window leakage.
MaskComposer
    Compose M1 artificial masks with native masks (`m_native * m_art`).

Functions
---------
default_beijing_features
    Frozen M2 pollutant feature names (`F = 6`).
compose_train_mask
    Elementwise `m_train = m_native * m_artificial`.
restrict_artificial_mask
    Force `m_artificial = 0` on natively missing cells.
artificial_target_mask
    Cells with ground truth held out by the artificial mask.
save_processed_dataset
    Write windows + split indices under `data/processed/m2/`.
load_processed_dataset
    Load a previously prepared NPZ bundle.
"""

from src.data.real.beijing import BeijingCsvLoader, HourlySiteSeries
from src.data.real.io import load_processed_dataset, save_processed_dataset
from src.data.real.masks import (
    MaskComposer,
    artificial_target_mask,
    compose_train_mask,
    restrict_artificial_mask,
)
from src.data.real.specs import (
    BeijingLoadSpec,
    RealDataPaths,
    SplitFractions,
    WindowSpec,
    default_beijing_features,
)
from src.data.real.splits import EntityAwareSplitter, SplitIndices
from src.data.real.windows import RealWindowBuilder, RealWindowDataset

__all__ = [
    "BeijingCsvLoader",
    "BeijingLoadSpec",
    "EntityAwareSplitter",
    "HourlySiteSeries",
    "MaskComposer",
    "RealDataPaths",
    "RealWindowBuilder",
    "RealWindowDataset",
    "SplitFractions",
    "SplitIndices",
    "WindowSpec",
    "artificial_target_mask",
    "compose_train_mask",
    "default_beijing_features",
    "load_processed_dataset",
    "restrict_artificial_mask",
    "save_processed_dataset",
]
