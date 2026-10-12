# Evaluation Contract (Phase 0 / Milestone 2)

This document freezes datasets, schema, splits, missingness protocols, metrics,
and the baseline list for TimeFill. M2 implements the real-data training path
against this contract; full baseline tables are deferred to **M3**.

The **M3 run contract** below freezes the report split, protocol grid, seeds,
and point-baseline metric policy. The feature list and mask rules in this
file stay as written for M2.

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
- **M3 report split:** **test**. Validation is for M2 model selection and for
  M3 smoke/debug only (see M3 run contract).

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

M3 required vs preferred status, and the point-baseline CRPS / coverage
policy, are locked under **M3 run contract**.

## M3 run contract

M3 reuses the M2 processed windows, checkpoint, and scaler. Training is not an
M3 exit criterion. Config: `configs/bench_m3.yaml`. Artifact root:
`artifacts/m3/`.

Tensor and mask conventions are unchanged: `x` and `m` shaped `(B, T, F)` with
`F = 6`; `m`: `1` = observed, `0` = missing; missing `x` cells are sentinel
`0.0`. Composition for scoring is `m_eval = m_native * m_artificial`. Metrics
run only on cells with `m_native == 1` and `m_artificial == 0`. DSM and the
reverse SDE stay in normalized space. Reported MAE / RMSE / CRPS / coverage
are in **original units** via `FeatureScaler.inverse_transform`.

### Report split

- **Final tables use the test partition** from M2 prepare
  (`data/processed/m2/`, `split_seed` and fractions from `configs/real_m2.yaml`).
- **Val is smoke / debug only** (CPU subset, checkpoint sanity). Val numbers
  are not the M3 comparison.
- Load `artifacts/m2/checkpoints/best_score_net.eqx` and
  `artifacts/m2/checkpoints/feature_scaler.npz`.

### Protocol grid

| Protocol | Settings | Required for M3 tables |
|---|---|---|
| MCAR | rates **0.10 / 0.30 / 0.50** on cells with `m_native == 1` | yes |
| Contiguous block | one gap per `(window, feature)`, `block_len` default **8** | optional |

The M2 train / impute default `mask_rate: 0.2` stays on the training path.
M3 sensitivity tables use the grid above. The same artificial masks are shared
across methods in one comparison.

### Seeds

Fixed in `configs/bench_m3.yaml` and logged on every artifact row:

| Field | Role |
|---|---|
| `seeds.mask` | Artificial-mask draw (MCAR and block) |
| `seeds.sample` | TimeFill reverse-SDE sampler |

### Point-baseline CRPS / coverage

LOCF and linear each emit one fill. **Locked policy:** treat that fill as a
degenerate ensemble with **`K = 1`**, so CRPS and coverage columns stay
populated next to TimeFill. Tables note that those methods are
non-probabilistic. With `K = 1`, CRPS reduces to absolute error of the point
fill, and a central interval has width 0 (coverage is the hit rate of that
point).

**Alternative (not used):** mark CRPS / coverage **N/A** for point methods and
compare those metrics for TimeFill only. Switching to N/A requires an edit to
this section and to the Milestone 3 section of `ASSUMPTIONS.md`.

TimeFill uses `sample.K_samples` (default 50) and `sample.solver_steps`
(default 100) from `configs/bench_m3.yaml`.

### Baselines (M3)

| Method | Status |
|---|---|
| LOCF | **required** |
| Linear interpolation | **required** |
| Stronger baseline (CSDI-style, BRITS, or a simple Transformer) | **deferred for v1** (absent from `methods`; compare vs LOCF / linear only) |

**Edge policy (LOCF / linear).** Fallback value is the **train-feature mean**
in original units: `FeatureScaler.mean[f]` from the saved M2 scaler (fit on
train cells with `m_native == 1`). The normalized mean is 0 and is not the
fallback. Baselines read the evaluation observation mask `m_eval`
(`1` = available context, `0` = to fill).

A **usable neighbor** depends on the method:

- **LOCF:** the latest earlier cell in the same window and feature with
  `m_eval == 1`. Carry that value forward. A leading run, and every cell of a
  feature with no observed cell in the window, has no usable neighbor and
  takes the train-feature mean. A trailing run after the last observation is
  LOCF (that observation is a usable neighbor).
- **Linear:** the nearest earlier and nearest later observed cells that
  bracket the gap. Interior gaps are linear interpolation on observed time
  indices. A leading gap, a trailing gap, and an all-missing feature have no
  bracketing pair and take the train-feature mean.

WP1 unit-tests this rule. Which array space the rest of the fill is computed
in (original units, or normalized then inverse-transformed) is chosen in the
benchmark runner (WP2). The reported edge value equals `FeatureScaler.mean[f]`.

### Still deferred

Irregular wall-clock time as first-class SDE dynamics stays deferred. M3 uses
the M2 regular grid plus masks. Streamlit is M4. Portfolio polish beyond an
M3 runbook is M5. Stronger baseline (CSDI / BRITS / Transformer) is deferred
for v1.

## Out of scope here

- Implementing the benchmark runner and baseline code (later M3 work packages;
  the report contract is **M3 run contract** above)
- Streamlit product (**M4**)
- Full portfolio README rewrite (**M5**)
