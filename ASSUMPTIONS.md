# Milestone 1 Assumptions

This document locks the data and modeling contracts for M1 (toy path).
Later milestones may relax these assumptions; they must not be changed silently
inside M1.

## Windowing

- Series are represented as **fixed-length windows** on a **regular time grid**.
- Default window length: `T = seq_len = 64`.
- Grid spacing is uniform (`Δt` constant). Irregular wall-clock timestamps are
  out of scope for M1; time may be treated as integer indices `0 … T-1`.
- Batch tensor layout is always `(B, T, F)` for values and masks.

## Features

- Number of channels `F = n_features` is small: `F ∈ {1, 2, 3}` for M1.
- Toy generators live in `src/data/synthetic.py`:
  - **Sine:** feature `f` has `f+1` cycles per window, phase `2π f / F`,
    plus small i.i.d. Gaussian noise (default std `0.05`).
  - **AR:** diagonal VAR(1) / AR(1) with `A = a I` (default `a = 0.8`) and
    Gaussian innovations (default std `0.1`).

## Mask convention

- Mask tensor `m` has the same shape as values `x`: `(B, T, F)`.
- `m[..., ] == 1` means **observed** (conditioning context).
- `m[..., ] == 0` means **missing** (imputation target).
- Mask dtype is floating (`float32`) so it can be concatenated with model inputs.

## Missing value sentinel

- Observed entries keep their true values inside `x`.
- Missing entries in `x` are stored as the sentinel **`0.0`**
  (`MISSING_VALUE_SENTINEL` in `src/data/schema.py`).
- Downstream code must use `m`, not “nonzero in `x`”, to detect missingness,
  because a true observed zero is allowed.

## Missingness protocols (M1)

- **MCAR random:** drop entries independently at rate `mask_rate`.
- **Contiguous block:** for each `(batch, feature)`, drop one contiguous
  time span of length `block_len` with a uniformly random start index
  (`src/data/masking.py`).
- Mask APIs return `(x_obs, m)` where observed cells are unchanged and
  missing cells are the sentinel `0.0`.
- WP1 mask comparison figure path:
  `artifacts/m1/mask_random_vs_block.png`
  (via `scripts/plot_m1_masks.py` / `save_mask_comparison`).
- Training / evaluation treat masked cells as targets; the score network still
  **sees observed context** via `(x * m, m)`-style conditioning (CSDI-style).

## Loss focus

- Denoising score matching (DSM) loss is computed **primarily on masked /
  target cells** (`m == 0`).
- **M1 / WP4 choice:** observed cells (`m == 1`) receive **zero** loss weight.
  Config key `train.obs_loss_weight` defaults to `0.0`. A positive value would
  add a light auxiliary DSM term on observed cells; that is disabled for M1.
- Training artifacts (WP4):
  - checkpoints: `artifacts/m1/checkpoints/` (best val loss)
  - loss CSV / plot: `artifacts/m1/loss_curves.csv`, `artifacts/m1/loss_curves.png`
  - val snapshot stub (until WP5 reverse impute): `artifacts/m1/val_snapshots/`

## Diffusion / training (config keys)

- VP-SDE uses linear `β(t)` between `beta_min` and `beta_max` over diffusion
  horizon `T_diff` (implemented in WP2: `src/sde/vp_sde.py`).
- Closed-form marginal: `x_t = mean(x0, t) + std(t) * eps` with
  `std^2 → 1` as `t → T_diff` (variance-preserving).
- WP2 forward-noising figure path:
  `artifacts/m1/forward_noising.png`
  (via `scripts/plot_m1_forward.py` / `save_forward_noising`).
- Training uses `lr`, `epochs`, AdamW `weight_decay`, `t_eps`, mask mode,
  logging / checkpoint knobs (WP4: `src/train/`).
- Sampling uses `K_samples` posterior draws and `solver_steps` reverse steps (WP5).

## Reverse SDE imputation (WP5)

- Implementation: `src/sample/reverse_sde.py` via Diffrax `diffeqsolve` +
  `Euler` (Euler–Maruyama) and `VirtualBrownianTree`.
- Time coordinate: integrate forward in `τ = T_diff - t` from `τ=0` to
  `τ = T_diff - t_eps`, with drift `dY/dτ = -VPSDE.reverse_drift(...)`.
- **Noise init (default `missing_only`):** at `t = T_diff`, missing cells
  (`m == 0`) are i.i.d. `N(0, 1)`; observed cells (`m == 1`) are set to the
  data values in `observations`. Alternative `full`: entire window is
  `N(0, 1)` (still ends with hard observation replacement).
- **Observation replacement (M1):** during the solve, drift and diffusion are
  multiplied by `(1 - m)` so observed coordinates stay fixed; after the solve,
  observed cells are hard-replaced with `observations` again. Continuous
  conditioning also uses the score net inputs `(mask, x_obs)`.
- Public API: `ReverseSDEImputer.impute(observations, mask, timestamps, key,
  k_samples) -> samples, mean, quantiles` with model/SDE/solver bundled in
  `ImputeContext`. `timestamps` may be grid indices `0 … T-1` and are not used
  in the M1 dynamics.
- Outputs: `samples` is `(B, K, T, F)` (or `(K, T, F)` if unbatched);
  `mean` matches the data layout; `quantiles` has trailing axis size 2
  (default 10% / 90%).

## Lightweight evaluation (WP6)

- Metrics are computed **only on masked / target cells** (`m == 0`) against
  known synthetic ground truth:
  - point: MAE / RMSE of the posterior mean
  - probabilistic: empirical ensemble CRPS from `K` samples
  - calibration (optional): empirical coverage of central 50% / 90% intervals
- End-to-end runner: `scripts/run_m1_toy.py` (config `configs/toy_m1.yaml`).
- WP6 figures under `artifacts/m1/`:
  - `forward_noising.png`
  - `mask_random_vs_block.png`
  - `imputation_mean_bands.png`
- LOCF / linear baseline tables remain **out of scope** until M3.

## Out of scope for M1

- PhysioNet / real CSV loaders
- Streamlit app
- LOCF / linear baseline comparison tables

---

# Milestone 2 Assumptions

This section extends M1 for real-data training. M1 contracts above remain in
force unless an item below explicitly overrides them for the M2 path.

## Primary dataset and features

- **Primary dataset:** UCI Beijing Multi-Site Air-Quality (see `docs/EVAL.md`).
- Feature count **`F = 6`**: `PM2.5`, `PM10`, `SO2`, `NO2`, `CO`, `O3`.
- Optional entity field: monitoring **site id** for entity-aware splits.
- PhysioNet ICU-style data is **not** required for the M2 exit criterion.

## Windowing after resample / binning

- Raw rows may be irregular or gappy in wall-clock time.
- **Resample / bin** onto a regular hourly (or config-chosen) grid before
  windowing: last-value-in-bin when present; empty bin → native missing
  (`m_native = 0`, value sentinel `0.0`).
- After resampling, series are **fixed-length windows** on that regular grid
  with default **`T = seq_len = 64`** (same default as M1 unless
  `configs/real_m2.yaml` overrides).
- Batch layout remains **`(B, T, F)`** for values and masks.
- Grid indices `0 … T-1` may be passed as `timestamps` to the imputer; they are
  not used as first-class irregular SDE dynamics in M2.

## Mask composition

- **`m_native`:** observed vs missing from the source after resampling.
- **`m_artificial`:** MCAR or contiguous-block masks from M1 APIs, applied only
  where `m_native == 1`.
- **Training / eval mask:** `m_train = m_native * m_artificial` (elementwise).
  Only originally observed cells can be held out as targets.
- Mask convention unchanged: `1` = observed, `0` = missing; `float32`.
- Missing cells in `x` remain sentinel **`0.0`**; never infer missingness from
  nonzero values in `x`.

## Normalization

- Per-feature standardization (or robust scaling) is fit on **train-split
  observed cells only** (`m_native == 1` within train windows).
- Val / test use train statistics; scaler stats are stored with checkpoints under
  `artifacts/m2/`.
- **WP2 path:** `artifacts/m2/checkpoints/feature_scaler.npz`
  (same directory as `best_score_net.eqx`; see `src/data/normalize.py`).
- DSM / SDE training runs in normalized space; reported MAE/RMSE prefer
  original units after inverse transform (`docs/EVAL.md`).
- Missing cells stay sentinel `0.0` under transform / inverse_transform and
  never enter fit statistics.

## Loss and model reuse

- DSM remains primarily on target cells (`m == 0`); `train.obs_loss_weight`
  default **`0.0`** unless deliberately enabled and documented.
- **M2 weighting detail:** DSM reduction uses
  `artificial_target_mask` (natively observed ∧ artificially missing).
  Natively missing cells are **not** scored even though `m_train == 0`
  there (no ground truth).
- Reuse M1 **VP-SDE** and Diffrax **reverse imputer** APIs; adapt `F` and
  normalization only.

## Artifacts

- M2 artifact root: **`artifacts/m2/`**
  - `checkpoints/`
  - `loss_curves.csv` / `loss_curves.png`
  - `val_snapshots/`
- WP4 val impute (from checkpoint alone):
  - `imputation_mean_bands.png` — mean ± bands in **original units**
  - `val_metrics.json` / `val_metrics.csv` — MAE / RMSE / CRPS /
    coverage_50 / coverage_90 on **artificial targets with GT** only
- Large raw / processed data live under `data/raw/` and `data/processed/` and
  are gitignored.

## Irregular time (status)

- **Deferred:** irregular wall-clock as first-class SDE dynamics is **not**
  promoted in M2.
- Irregularity is handled only via resample / bin → regular windows + masks.
- Do not advertise irregular-time dynamics as a completed M2 differentiator.

## Out of scope for M2

- LOCF / linear / CSDI baseline tables (**M3**)
- Streamlit app (**M4**)
- Portfolio polish / full README rewrite beyond an M2 runbook (**M5**)
- Breaking or regressing the M1 toy path (`configs/toy_m1.yaml`,
  `scripts/run_m1_toy.py`)

---

# Milestone 3 Assumptions

This section extends M1 and M2 for the benchmark. M1 and M2 contracts above
remain in force. M3 does not redefine training, tensor layout, or mask
conventions.

## Carry-forward (unchanged)

- **Dataset:** UCI Beijing Multi-Site Air-Quality. **`F = 6`:** `PM2.5`,
  `PM10`, `SO2`, `NO2`, `CO`, `O3` (same order as `docs/EVAL.md`).
- Values and masks `(B, T, F)`; `1` = observed, `0` = missing; missing `x`
  cells are sentinel `0.0`.
- `m_eval = m_native * m_artificial`. Metrics only where `m_native == 1` and
  `m_artificial == 0`.
- DSM / reverse SDE in normalized space. Reported MAE / RMSE / CRPS / coverage
  in original units via `FeatureScaler.inverse_transform`.
- Reuse M2 processed windows (`data/processed/m2/`), checkpoint, and scaler.
  Training is not an M3 exit criterion.

## Report split, protocol grid, seeds

- **Report split: test.** Val is smoke / debug only and is not the comparison
  table.
- **Protocol grid:** MCAR rates **0.10 / 0.30 / 0.50** (required). Optional
  contiguous block with **`block_len` default 8**.
- **Seeds** in `configs/bench_m3.yaml`, logged in artifacts:
  - `seeds.mask` — artificial mask (default `0`)
  - `seeds.sample` — TimeFill sampler (default `1`)
- One comparison shares the same artificial masks across methods.

## Point-baseline probabilistic metrics

- LOCF and linear are single fills. **Locked policy:** degenerate ensemble
  **`K = 1`** so CRPS and coverage stay in the table. Record in the table that
  those methods are non-probabilistic (`K = 1` CRPS is absolute error; interval
  width is 0).
- Alternative not selected: CRPS / coverage **N/A** for point methods.
- TimeFill keeps multi-sample `K` from `sample.K_samples`.

## LOCF / linear edge policy

- Fallback is the **train-feature mean in original units**,
  `FeatureScaler.mean[f]` on the saved M2 scaler. The normalized mean (0) is
  not the fallback.
- Context mask is `m_eval`.
- **LOCF usable neighbor:** latest earlier observed cell in that window and
  feature. Otherwise (leading gap, or feature all missing) use the
  train-feature mean. Trailing cells after the last observation are carried
  forward.
- **Linear usable neighbors:** an earlier observed cell and a later observed
  cell that bracket the gap. Interior gaps are interpolated. Leading gaps,
  trailing gaps, and all-missing features use the train-feature mean.
- The reported edge value equals that original-unit mean.
- **WP2 fill space (locked):** LOCF / linear run in **original units** on
  inverse-transformed context windows under `m_eval`. TimeFill remains
  normalized for reverse SDE; all reported metrics stay in original units.

## Artifacts

- M3 artifact root: **`artifacts/m3/`**
- Config skeleton: `configs/bench_m3.yaml` (paths to M2 `processed_dir`,
  checkpoint, and scaler).

## Stronger baseline

- CSDI-style, BRITS, or a simple Transformer imputer is **deferred for v1**
  (not in `methods` / `metrics_by_rate.*`). M3 compares TimeFill to LOCF and
  linear only; VP-SDE + Diffrax remains the primary method.
- Required methods: **TimeFill, LOCF, linear**.

## Irregular time (status)

- **Deferred:** irregular wall-clock as first-class SDE dynamics is not
  promoted in M3. Regular grid + masks only, same as M2.

## Out of scope for M3

- Streamlit app (**M4**)
- Portfolio polish beyond an M3 runbook (**M5**)
- Redefining M2 training as an exit criterion
- Changing `F`, the sentinel, or metric-cell composition
