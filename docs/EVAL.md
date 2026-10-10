# Evaluation Contract (Phase 0 / Milestone 2)

This document freezes datasets, schema, splits, missingness protocols, metrics,
and the baseline list for TimeFill. M2 implements the real-data training path
against this contract; full baseline tables are deferred to **M3**.

## Primary dataset (M2 exit criterion)

**Choice:** [Beijing Multi-Site Air-Quality Data Set](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)
(UCI Machine Learning Repository; Zhang et al.).

**Rationale**
- Publicly downloadable CSV with no credential wall (PhysioNet ICU access is a
  friction risk for reproducible M2).
- Multivariate continuous features with **native missingness**, suitable for
  mask composition (`m_native * m_artificial`).
- Near-hourly sampling supports the M1/M2 contract: resample / bin to a
  **regular grid**, then fixed-length windows `(B, T, F)`.
- Aligns with plan.md "air-quality" option for M2 and with CSDI-style
  air-quality imputation setups in the literature.

**Secondary / deferred:** PhysioNet ICU-style series remain the healthcare story
dataset; not required for the M2 exit criterion.

**Local layout (WP1+; do not commit large dumps)**
- Raw cache: `data/raw/beijing_multisite/`
- Processed windows: `data/processed/m2/` (e.g. arrays + split indices)

### Fetch and prepare (WP1)

Do **not** commit the zip or full CSV dumps (see `.gitignore` for
`data/raw/` and `data/processed/`).

1. Download **PRSA2017_Data_20130301-20170228.zip** from the UCI page above
   (dataset id 501).
2. Unzip the per-site `PRSA_Data_*.csv` files into
   `data/raw/beijing_multisite/`.
3. From the repo root (after `pip install -e .`):

```bash
python scripts/prepare_m2_data.py --config configs/real_m2.yaml
```

This writes `data/processed/m2/windows.npz` with values / native masks /
entity ids and deterministic train/val/test indices
(`split_seed` / fractions from `configs/real_m2.yaml`). Unit tests use
tiny hermetic CSV fixtures and do **not** require network or the full
UCI download.

Loader contract: `src/data/real/` (hourly last-value-in-bin resample,
non-overlapping windows, `m_train = m_native * m_artificial`).

### Train and val impute (WP3–WP4)

After prepare (from the repo root):

```bash
python scripts/train_m2_real.py --config configs/real_m2.yaml
python scripts/run_m2_impute_val.py --config configs/real_m2.yaml \
    --ckpt artifacts/m2/checkpoints/best_score_net.eqx
```

- Checkpoint / scaler: `artifacts/m2/checkpoints/best_score_net.eqx`,
  `artifacts/m2/checkpoints/feature_scaler.npz`
- Val deliverables: `artifacts/m2/imputation_mean_bands.png`,
  `artifacts/m2/val_metrics.json` (and `.csv`)
- Full clone runbook + CPU smoke overrides: `README.md` (How to run (M2)).

## Schema

| Field | Required | Notes |
|---|---|---|
| `timestamp` | yes | Wall-clock datetime; after prep, windows live on a regular grid |
| `entity` / `id` | optional | Monitoring site id (e.g. station name); used for entity-aware splits |
| feature columns | yes | Frozen list below; continuous numeric |
| `mask` | yes | Same shape as values; `1` = observed, `0` = missing (`float32`) |

**Tensor contract (carried from M1):** values and masks `(B, T, F)`; missing cells
in `x` use sentinel `0.0`; never infer missingness from `nonzero(x)`.

### Frozen feature list (`F = 6`)

| Index | Name | Role |
|---|---|---|
| 0 | `PM2.5` | pollutant |
| 1 | `PM10` | pollutant |
| 2 | `SO2` | pollutant |
| 3 | `NO2` | pollutant |
| 4 | `CO` | pollutant |
| 5 | `O3` | pollutant |

Meteorology (`TEMP`, `PRES`, `DEWP`, `RAIN`, `WSPM`, wind direction) is out of
the M2 primary feature set unless a later assumption explicitly expands `F`.

## Splits

- **Entity-aware preferred:** hold out entire station(s) and/or contiguous calendar
  blocks so the same physical window cannot appear in more than one split.
- Default M2 ratios (deterministic seed in config): train / val / test ≈
  **70% / 15% / 15%** of eligible windows after filtering.
- **No window leakage:** overlapping sliding windows that share timesteps must
  not cross the train–val or train–test cut; either non-overlapping windows or
  split **before** windowing on time/entity partitions.
- Artificial masks for metrics are drawn only on cells that are observed in the
  ground-truth window (`m_native == 1`).

## Missingness protocols

1. **Native missingness:** cells already missing in the source CSV
   (`m_native == 0`). Preserved through resampling (empty bins → missing).
2. **MCAR (artificial):** independently drop originally observed cells at rates
   **10% / 30% / 50%** (protocol rates for sensitivity / M3). Config default in
   `configs/real_m2.yaml` is `mask_rate: 0.2` for the main train / impute path.
3. **Contiguous block (artificial):** per `(window, feature)`, one contiguous
   gap of length `block_len` (M2 default `8`, same spirit as M1).
4. **Composition:** `m_train = m_native * m_artificial` (elementwise). Only
   originally observed cells can become imputation targets.
5. **Irregular wall-clock as SDE dynamics:** deferred; irregularity is handled by
   binning / resampling onto a regular grid before windowing (see `ASSUMPTIONS.md`).

## Metrics

Computed **only on artificially masked cells that had ground truth**
(`m_native == 1` and `m_artificial == 0`), unless noted.

| Metric | Role | M2 |
|---|---|---|
| MAE / RMSE | Point error of posterior mean | yes (lightweight on val) |
| CRPS | Probabilistic (ensemble from `K` samples) | yes (lightweight on val) |
| Coverage 50% / 90% | Calibration of central predictive intervals | yes (lightweight on val) |
| ACF / cross-corr error | Structure vs completed truth | optional; not required for M2 exit |

Reported MAE/RMSE / CRPS / coverage use **original (inverse-transformed) units**
via `FeatureScaler.inverse_transform` (`scripts/run_m2_impute_val.py`).

## Baselines (list only — run in M3)

Do **not** require baseline tables for M2 exit.

1. **LOCF** — last observation carried forward (per feature)
2. **Linear interpolation** — per feature along time
3. **Stronger baseline (time permitting)** — e.g. BRITS, a simple Transformer
   imputer, or a reimplementation-light CSDI-style discrete conditional diffusion

## Out of scope here

- Running LOCF / linear / CSDI tables (**M3**)
- Streamlit product (**M4**)
- Full portfolio README rewrite (**M5**)
