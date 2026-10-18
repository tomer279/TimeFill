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
- Portfolio polish / full README rewrite (**M5**) — **shipped**; see
  Milestone 5 below
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
- Portfolio polish beyond an M3 runbook (**M5**) — **shipped**; see
  Milestone 5 below
- Redefining M2 training as an exit criterion
- Changing `F`, the sentinel, or metric-cell composition

---

# Milestone 4 Assumptions

This section extends M1–M3 for the Streamlit product layer (Phase 5). M1–M3
contracts above remain in force. M4 does not redefine `F`, the sentinel, the
mask alphabet, or training.

## Carry-forward (unchanged)

- **Dataset / pretrained path:** UCI Beijing Multi-Site Air-Quality. **`F = 6`**
  in order `PM2.5`, `PM10`, `SO2`, `NO2`, `CO`, `O3` (same as `docs/EVAL.md`).
- Values and masks `(B, T, F)`; default **`T = seq_len = 64`**; `1` = observed,
  `0` = missing; missing `x` cells are sentinel `0.0`. Never infer missingness
  from nonzero values in `x`.
- DSM / reverse SDE in **normalized** space via saved `FeatureScaler`.
  Displayed and downloaded values are in **original units** after
  `inverse_transform`.
- Reuse M2 checkpoint + scaler:
  `artifacts/m2/checkpoints/best_score_net.eqx`,
  `artifacts/m2/checkpoints/feature_scaler.npz`.
- TimeFill imputer: existing `src/sample` reverse-SDE impute API (multi-sample
  mean ± bands). Observed cells stay conditioning context.

## Happy path (pretrained only)

- Default UX loads the M2 checkpoint + scaler and completes on the bundled
  sample without training.
- **On-the-fly fine-tune is deferred for v1** (not the happy path). Shipping it
  later requires an explicit ASSUMPTIONS unlock and an opt-in UI toggle; it
  must not block the pretrained demo.

## Bundled sample and upload schema

- Bundled demo CSV lives under **`app/sample_data/`** (config key
  `data.sample_csv`). The file may be a stub path until WP1 ships the slice.
- **Required columns:** frozen feature names in EVAL order
  (`PM2.5`, `PM10`, `SO2`, `NO2`, `CO`, `O3`). Exact names; **reject** missing
  required columns and **do not silently remap** aliases.
- **`timestamp`:** recommended (wall-clock for resample / display). If absent,
  treat rows as already ordered on a regular grid index `0 … N-1`.
- **`entity` / `site`:** optional; used for grouping / display only in M4.
- Native NaNs / empty cells → `m_native = 0`, value sentinel `0.0`.

## Window / length policy

- After optional hourly last-value-in-bin resample (same spirit as M2 prep),
  build **non-overlapping** windows of length `T = seq_len` (default 64).
- **Shorter than `T`:** single window, **right-pad** with sentinel `0.0` and
  `m = 0` on padded steps (padded cells are never imputation targets with GT).
- **Longer than `T`:** emit consecutive non-overlapping windows; a trailing
  remainder shorter than `T` is right-padded as above.
- Cap processed windows with `smoke.max_windows` (UI latency guard). Document
  the effective count in session / download metadata when practical.

## Demo corruption

- Sidebar may apply optional artificial **MCAR** or **contiguous block** masks
  on otherwise observed demo cells.
- Compose **`m = m_native * m_artificial`** (elementwise), same as M2/M3 eval.
  Only natively observed cells can become demo targets.

## UI defaults (latency-first)

Locked in `configs/app_m4.yaml` (smaller than the M3 report settings):

| Knob | M4 default | M3 report (`bench_m3.yaml`) |
|---|---|---|
| `sample.K_samples` | **8** | 50 |
| `sample.solver_steps` | **20** | 100 |
| `seeds.mask` | **0** | 0 |
| `seeds.sample` | **1** | 1 |

Raising `K` / steps toward M3 values is allowed in the sidebar for quality;
defaults stay latency-first so the pretrained demo finishes in a few minutes.

## Downloads

- **Required:** mean-fill CSV in **original units**. Observed cells
  (`m == 1` after composition used for the run) are preserved; targets filled
  with the posterior mean.
- **Optional:** multi-sample long format
  (`sample_id`, time, feature, value) in original units.

## Demo-eval button

- **Shipped (smoke only):** the Streamlit “Run evaluation on demo dataset”
  control runs a tiny MCAR slice via `app/demo_eval_smoke.py`, reusing
  `ImputationEvaluator` / `metric_observation_mask`. Caps live under
  `smoke.*` in `configs/app_m4.yaml`.
- The full M3 protocol grid stays in `scripts/run_m3_benchmark.py`. UI metrics
  must be labeled as a smoke / demo slice, not the report table.

## Artifacts

- App config: **`configs/app_m4.yaml`** (M2 checkpoint / scaler, sample CSV,
  UI defaults, seeds, smoke caps).
- Optional runtime outputs: **`artifacts/m4/`** (e.g. persisted demo-eval
  slice). Not required for the happy path.
- Launch (WP3+): `streamlit run app/streamlit_app.py`

## Irregular time (status)

- **Deferred:** irregular wall-clock as first-class SDE dynamics is **not**
  promoted in M4. Timestamps may be used for resample / display only; dynamics
  remain regular grid + masks.

## Out of scope for M4

- Portfolio polish / full README rewrite (**M5**) — **shipped**; see
  Milestone 5 below
- Full M3 benchmark grid inside the app
- Retraining / fine-tune as the happy path
- Redefining `F`, the sentinel, or the mask alphabet
- Breaking or regressing M1–M3 contracts
- Replacing the project with a CSDI port
- Advertising irregular-time SDE dynamics as completed

---

# Milestone 5 Assumptions (v1 portfolio)

This section locks the **v1 portfolio / README surface** for Milestone 5.
M1–M4 contracts above remain in force. M5 does **not** redefine `F`, the
sentinel, the mask alphabet, training, the M3 report protocol, or the M4
Streamlit happy path.

## Portfolio surface

- **README.md** is the portfolio front door: narrative problem → method →
  evidence → demo → reproduce → assumptions / limits → citations.
- **Contracts stay here and in `docs/EVAL.md`** (and walkthroughs). Do not
  move tensor / mask / metric-cell rules into README prose as the source of
  truth.
- Product name: **TimeFill**. License: **MIT** (`LICENSE.txt`).
- **North star vs v1:** README may state a universal train-on-upload +
  gap-regime routing product goal as **post-v1**. v1 claim scope stays
  Beijing `F=6` pretrained path + LOCF/linear evidence only — do not advertise
  train-on-arbitrary-CSV or gap specialists as shipped.

## Metrics source of truth

- Portfolio numbers must come from:
  - `docs/m3_failure_notes.md` (preferred narrative + claim checklist), and/or
  - `artifacts/m3/metrics_by_rate.csv` (and `.json`) produced by
    `scripts/run_m3_benchmark.py` on the **test** split.
- Do **not** invent metrics in README. If local `artifacts/m3/` is missing,
  point to `docs/m3_failure_notes.md` only.
- Large binary / panel dumps under `artifacts/m3/panels/` are **local only**
  (do not commit large artifacts).

## Claim discipline (v1)

- Compare and claim wins **vs LOCF and linear only** (M3 required methods).
- **No CSDI / BRITS / Transformer win claim** — stronger baseline remains
  **deferred for v1** (same as M3 / `docs/EVAL.md`).
- **Irregular-time as first-class SDE dynamics is deferred for v1.** State
  this next to the method wedge in README, not only in fine print. v1 uses
  resample / bin → regular grid + masks (M2–M4).
- PhysioNet ICU-style data is **motivation / future only** for v1 — do **not**
  claim PhysioNet was run for the shipped portfolio.
- Prefer the locked summary from `docs/m3_failure_notes.md`: TimeFill is the
  better **probabilistic** imputer vs LOCF / linear (CRPS + coverage); point
  MAE is competitive and wins on block gaps; it does not dominate linear on
  easy short-gap MCAR.

## Target README TOC (M5)

Hero → problem → method (short) → evidence → demo → reproduce →
assumptions / limits → citations.

Evidence presentation (later WPs; not a full rewrite in WP0):

- Inline: **summary claim** + a **compact highlight** (recommend block
  `block_len=8` and/or MCAR 50% CRPS / coverage rows) **or** a small table
  subset.
- Full grid + shortfalls: link to `docs/m3_failure_notes.md` (and optionally
  `artifacts/m3/metrics_by_rate.csv` for local repro).

## Asset inventory (portfolio)

| Asset | Role |
|---|---|
| `docs/m3_failure_notes.md` | Claim checklist, full results table, win/lose notes |
| `artifacts/m3/metrics_by_rate.csv` | Machine-readable metrics (local / gitignored) |
| `artifacts/m3/panels/` | Qualitative panels (local; do not commit large dumps) |
| `streamlit run app/streamlit_app.py` | M4 demo entrypoint |
| `docs/EVAL.md` | Frozen eval / schema / M3 run contract |
| `ASSUMPTIONS.md` | Tensor / mask / milestone contracts (this file) |

## Required Limitations themes (eventual README)

README Limitations (later WP) must cover at least:

1. Regular-grid windows after resample; irregular wall-clock SDE dynamics deferred.
2. Comparison scope: LOCF / linear only; no stronger baseline in v1.
3. MAE shortfalls vs linear on easy short-gap MCAR (honest M3 notes).
4. Compute / latency: sampler `K` and solver steps; M4 UI caps vs M3 report settings.
5. Domain: Beijing air-quality `F = 6` pretrained path; schema not silently remapped.
6. PhysioNet / healthcare irregular story not part of the v1 evidence run.

## Banned phrases / overclaims (v1 README and portfolio copy)

Do **not** use language that asserts or implies:

- “irregular-time SDE complete” / “first-class irregular-time dynamics shipped”
- “beats CSDI” / “outperforms CSDI” / “better than CSDI / BRITS / Transformer”
- “SOTA on PhysioNet” / “PhysioNet results” / “evaluated on PhysioNet” (v1)
- “first ever score-based / diffusion imputer”
- “calibrated uncertainty” without tying to reported coverage (and M3 notes)
- “production-ready” / “deployed at scale” (v1 is a research + demo portfolio)

Allowed framing examples: “CSDI-style conditional imputation objective”;
“continuous-time VP-SDE + Diffrax wedge”; “irregular-time **deferred for v1**”;
“vs LOCF / linear on the M3 test protocols.”

## Out of scope for M5

- Stretch goals from `plan.md` (true irregular-time Neural SDE encoder–decoder,
  block-gap curriculum, constrained imputation, privacy / memorization checks).
- Redefining `F`, the missing-value sentinel, or mask composition.
- New training runs or changing M2/M3 report defaults as an M5 exit criterion.
- Regressing the M4 Streamlit app or the M3 benchmark path.
- M5 closure is docs-only (citations, hygiene, exit checklist); no new `src/`.

---

# Milestone 6 Assumptions (schema-flexible ingest)

This section extends M1–M5 for **post-v1 schema-flexible ingest**. M1–M5
contracts above remain in force. M6 does **not** redefine the tensor / mask
alphabet, the missing-value sentinel, Beijing `F = 6` claim tables, M2/M3
training or report protocols, or the M4 exact-schema Streamlit happy path.

## Carry-forward (unchanged)

- Values and masks `(B, T, F)`; default **`T = seq_len = 64`**; `1` = observed,
  `0` = missing; missing `x` cells are sentinel **`0.0`**. Never infer
  missingness from nonzero values in `x`.
- **v1 pretrained / evidence path:** Beijing `F = 6` exact names
  (`PM2.5`, `PM10`, `SO2`, `NO2`, `CO`, `O3`); M2 checkpoint + scaler; M3 vs
  LOCF / linear only (`docs/EVAL.md`).
- **Regular grid after optional binning:** resample / last-value-in-bin → fixed
  `T` (same spirit as M2–M4). **Irregular wall-clock as first-class SDE
  dynamics remains deferred.** Timestamps may be used for bin / display only.
- **M4 pad / multi-window spirit:** non-overlapping windows of length `T`;
  right-pad short series and trailing remainders with sentinel `0.0` and
  `m = 0`; optional `max_windows` cap (UI / preview latency).

## Two ingest modes

| Mode | Schema | Allowed actions in M6 |
|---|---|---|
| **Exact Beijing (fallback)** | Required feature names match EVAL order exactly | Ingest + window + **pretrained impute** (current M4 path) |
| **Flexible schema** | User / inferred `features[]` with `F ≥ 1` (see max-`F` cap) | **Ingest + schema / missingness preview only** until M7 or a **matching** checkpoint / scaler |

- Exact mode is the default pretrained demo; flexible mode must not silently
  degrade into Exact by renaming columns.
- Mode selection (later WPs): auto-detect Exact when headers match
  `default_app_schema()`; otherwise Flexible (with optional user map override).

## Column roles

- **`features[]`:** continuous numeric columns used as the feature axis
  (order is user-confirmed or map-YAML / CLI map; no silent Beijing rename).
- **`timestamp` (optional):** wall-clock for optional hourly last-value-in-bin
  and display. If absent, row order is the grid index `0 … L-1`.
- **`entity` / `id` (optional):** grouping for multi-series when present;
  single series when absent. Display / grouping only in M6 (no new entity
  model).
- **Hard ban:** silent remap of unknown headers to `PM2.5`…`O3`.

## Native missingness

- Empty / NaN / documented NA tokens (`""`, `na`, `nan`, `null`, `none`,
  case-insensitive) → **`m_native = 0`**, value sentinel **`0.0`**.
- Same mask alphabet as M1–M5; missingness is never inferred from nonzero `x`.

## Pad / truncate / window defaults

- After optional regular-grid binning (or row-order if no timestamps), cut
  **non-overlapping** windows of length **`T = seq_len`** (default **64**).
- Short series / trailing remainder: **right-pad** with sentinel `0.0` and
  `m = 0`. Padded cells are never GT imputation targets.
- Optional **`max_windows`** cap (preview / UI latency); document the effective
  count when practical.
- **Locked default:** multi-window + pad remainder (M4 spirit). A
  “truncate to first `T` only” toggle is **not** required for M6 exit and is
  not the default.

## Reject thresholds (locked defaults)

Reject with **user-facing errors** (not stack traces only). Never invent
Beijing columns to “fix” a reject. Keys below match
`configs/ingest_m6.yaml` when that skeleton is present.

| Rule | Locked default | Definition |
|---|---|---|
| **`min_rows`** | **`max(1, seq_len // 4)`** → **16** when `seq_len = 64` | After optional binning, reject if usable grid length `L < min_rows` |
| **`min_observed_rate_global`** | **0.10** | Reject if `mean(m_native) < 0.10` over all cells |
| **`min_observed_rate_per_feature`** | **0.05** | Reject if **any** feature has `mean(m_native[:, f]) < 0.05` |
| **`max_features`** | **32** | Soft cap on `F`; reject clearly if `F > 32` |

- **Schema-invalid** (zero features after map, duplicate maps, non-numeric
  feature cells after parse, empty file) → reject.
- Thresholds apply to the **native** frame / grid **before** artificial demo
  masks (same spirit as M4: reject on source quality, not on demo corruption).

## Impute gate

- **Exact Beijing:** pretrained impute allowed when features **exactly** match
  the loaded M2 scaler / checkpoint names and count (current M4 behavior).
- **Flexible schema:** may show schema + missingness + window preview, then
  **block impute** unless mapped `features[]` **exactly** match the loaded
  scaler / checkpoint feature names and `F`.
- Block message must point to **M7 train-on-upload / adapt** or “load a
  compatible checkpoint / scaler,” not imply silent success with Beijing
  weights on a mismatched schema.

## Fixture location plan

- Small non-Beijing CSV (git-friendly): **`tests/fixtures/toy_non_beijing.csv`**
  (shipped in a later WP; path locked here).
- Must **not** use the frozen Beijing pollutant name set as its schema.
- Optional preview sample path may also be referenced from
  `configs/ingest_m6.yaml` (`data.fixture_csv`).

## M4 extension points (inventory)

| Module | Current exact-schema role |
|---|---|
| `app/schema_validate.py` | `AppSchemaSpec`, `validate_upload_csv` / `validate_upload_rows`, `ValidatedFrame`, `summarize_missingness` |
| `app/window_adapter.py` | `WindowAdaptRequest`, `adapt_validated_frame` (pad / multi-window / `max_windows`) |
| `app/streamlit_app.py` | `_load_payload`, `_render_upload_guidance`, `_validate_uploaded_bytes` (upload → exact validate → adapt) |
| `configs/app_m4.yaml` | Beijing demo defaults; **keep** as Exact-mode config |

## Banned behaviors

- Silent remap of unknown columns to Beijing names (`PM2.5`…`O3`).
- Claiming arbitrary CSVs can be imputed with the M2 Beijing checkpoint.
- Merging flexible-ingest demos into M3 Beijing claim tables or rewriting
  `docs/EVAL.md` protocols as if flexible ingest expands v1 metrics.
- Advertising a universal imputer from M6 alone.
- Rewriting M1–M5 tensor / mask / sentinel contracts.

## Claim discipline (flexible ingest ≠ Beijing evidence)

- Flexible ingest validates windows and previews missingness; it does **not**
  expand the v1 Beijing M3 claim table or EVAL report protocols (see
  `docs/EVAL.md` § **M6 flexible ingest**).
- Portfolio / README numbers remain Beijing-only until a later milestone
  explicitly unlocks a new domain with measured evidence.
- M6 ships ingest flexibility, **not** a universal imputer. Impute on a
  Flexible schema requires a matching checkpoint / scaler or **M7**.

## Artifacts

- Optional config skeleton: **`configs/ingest_m6.yaml`** (thresholds, `max_F`,
  fixture path). Do **not** overwrite M2/M3 claim artifacts or
  `configs/app_m4.yaml` Beijing defaults.
- Do not wire Streamlit to `ingest_m6.yaml` in WP0.

## Irregular time (status)

- **Deferred:** irregular wall-clock as first-class SDE dynamics is **not**
  promoted in M6. Timestamps for bin / display only; dynamics remain regular
  grid + masks.

## Out of scope for M6

- **M7** train-on-upload / adapt
- **M8** gap-regime routing
- **M9** evidence expansion (hosted checkpoint, second domain, stronger
  baseline)
- Stretch irregular-time Neural SDE
- Claiming a universal imputer
- Rewriting M1–M5 tensor contracts or silently changing Beijing `EVAL.md`
  claim protocols

---

# Milestone 7 Assumptions (train-on-upload / adapt)

This section extends M1–M6 for **post-v1 train-on-upload / adapt**. M1–M6
contracts above remain in force. M7 does **not** redefine the tensor / mask
alphabet, the missing-value sentinel, Beijing `F = 6` claim tables, M2/M3
training or report protocols, or the Exact Beijing pretrained Streamlit path.

## Carry-forward (unchanged)

- Values and masks `(B, T, F)`; default **`T = seq_len = 64`**; `1` = observed,
  `0` = missing; missing `x` cells are sentinel **`0.0`**. Never infer
  missingness from nonzero values in `x`.
- **M6 ingest:** Exact Beijing vs Flexible; reject thresholds; pad /
  multi-window; no silent Beijing remaps.
- **Normalization:** DSM / reverse SDE in scaled space; display / download in
  original units via `FeatureScaler.inverse_transform`.
- **Train-only scaler (M2 spirit):** fit on **native-observed cells in the
  declared train split only**; never on held-out artificial-gap targets used
  for smoke quality check.
- **v1 evidence path:** Beijing `F = 6` M2 checkpoint + scaler; M3 vs LOCF /
  linear only — frozen (`docs/EVAL.md`).
- **Irregular wall-clock as first-class SDE dynamics remains deferred.**
  Timestamps for bin / display only.
- **Impute API:** VP-SDE + Diffrax reverse sampling, `K` samples, quantile
  bands, observed-context conditioning (same spirit as M1 / M4).
- **`ConditionalScoreNet.n_features` is static** (`eqx.field(static=True)`).
  Fine-tune from a Beijing (`F = 6`) base is allowed **only when** upload
  `F` matches base `F`.

## Two product paths after M7

| Path | Ingest | Model | Impute |
|---|---|---|---|
| **Exact Beijing pretrained (unchanged)** | Exact (or Flexible with exact name/`F` match) | Load M2 claim weights + scaler | M4 session impute; no adapt required |
| **Upload-adapt** | M6 Flexible (or Exact) → windows | Short adapt → **session** checkpoint + scaler | Reverse-SDE impute → download mean ± bands |

- Exact Beijing pretrained demo stays intact and independent of adapt.
- Upload-adapt **never** overwrites `artifacts/m2/checkpoints/*` as a
  “universal” model.

## Adapt modes

| Mode | When | Behavior |
|---|---|---|
| **Train-from-scratch** | Default when `F ≠` base `F`; always available | Init `ConditionalScoreNet(n_features=F)` for the upload’s `F`; train under smoke caps |
| **Fine-tune-from-base** | Opt-in only when `F` equals base `F` | Load base checkpoint (default: M2 Beijing) and continue DSM under smoke caps |

- **Refuse fine-tune** when `F ≠` base `F`. Surface a clear message and offer
  scratch. Do not pad / truncate the feature axis to force a load.
- **Fine-tune name policy (locked):** base `feature_names` must equal the
  upload’s mapped `features[]` in order, **or** the user must explicitly
  confirm same semantics. **Silent remap of unknown columns to `PM2.5`…`O3`
  remains banned** (same as M6).
- Default product choice when both are legal: **scratch** unless the user
  opts into fine-tune.

## Train / smoke split and scaler leakage

- After M6 window adapt, declare a deterministic split of windows:
  - **Train fraction:** **`0.80`**
  - **Smoke fraction:** **`0.20`**
  - Split by **window index** in adapt order (no random permutation of
    windows for the scaler fit path; optional seeded shuffle of *train*
    window order for SGD only does **not** move smoke cells into fit).
- Fit `FeatureScaler` on **train-split native-observed** cells only
  (`m_native == 1` within train windows). Apply the same stats to smoke /
  impute windows.
- Artificial gaps for smoke eval (MCAR / block) are applied **after** scaler
  fit, only where `m_native == 1`, composing
  `m = m_native * m_artificial` (M2/M3 spirit).
- **Never** fold held-out artificial-gap smoke targets into scaler fit or into
  DSM labels incorrectly.
- DSM / reverse SDE stay in normalized space; downloads use
  `inverse_transform`.

## Locked smoke caps (Streamlit / CLI demo)

Suitable for a short CPU demo, **not** a full M2 train. Keys match
`configs/adapt_m7.yaml` when that skeleton is present.

| Knob | Locked default |
|---|---|
| `train.epochs` | **5** |
| `train.steps_per_epoch` | **20** |
| `train.batch_size` | **4** |
| `sample.K_samples` | **8** |
| `sample.solver_steps` | **50** |
| `adapt.timeout_seconds` | **600** (10 minutes wall-clock) |
| `adapt.min_windows` | **2** |
| `adapt.train_fraction` | **0.80** |
| `train.seed` | **0** |
| `seeds.mask` | **0** |
| `seeds.sample` | **1** |
| `train.obs_loss_weight` | **0.0** |
| `train.lr` | **0.0003** (same order as `configs/real_m2.yaml`) |

- Raising caps for longer CLI runs is allowed only via explicit config / CLI
  overrides; **defaults stay smoke-capped**.
- UI / CLI must surface the effective caps (epochs, timeout, `K`, steps).

## Min data and M6 rejects

- Upload must pass **M6 reject thresholds** first (`min_rows`, observed-rate
  floors, `max_features`, schema-invalid) — see Milestone 6.
- Additionally for adapt: require **`n_windows ≥ 2`** after M6 window adapt
  (and global observed rate still ≥ **0.10**).
- Too little data → **user-facing reject**; do not start training.

## Failure modes (abort without promoting best)

| Failure | Behavior |
|---|---|
| Too little data / M6 reject / `n_windows < 2` | Do not start adapt; clear message |
| Non-finite (NaN / Inf) train or val loss | **Abort**; do **not** save or promote a “best” checkpoint |
| Wall-clock **`timeout_seconds` exceeded** | **Abort**; do not promote partial best as the session default |
| OOM / resource error | **Abort**; clear message; no half-written global default |

- On abort: leave any incomplete run dir unmarked as successful; do not bind
  session impute to a failed run.
- Never silently fall back to writing into `artifacts/m2/checkpoints/`.

## Artifact layout

- M7 run root: **`artifacts/m7/<run_id>/`**
  - `best_score_net.eqx` — session score net (only after a **successful**
    capped run that completed without abort)
  - `feature_scaler.npz` — train-only scaler for this upload
  - `adapt_meta.json` — `feature_names`, `F`, `T`, seeds, mode
    (`scratch` / `fine_tune`), caps, run status
- Optional: `loss_curves.csv` / smoke metric side files under the **same**
  `run_id` dir.
- **`run_id`:** opaque unique id (e.g. UTC stamp + short random suffix);
  one adapt invocation → one directory.
- **Never** default-write adapt outputs into `artifacts/m2/` or overwrite
  M2 claim checkpoints / scaler as the sole universal model.
- Optional config skeleton: **`configs/adapt_m7.yaml`** (caps, timeout,
  `artifact_dir`, seeds). Do **not** wire Streamlit or the training loop to
  it in WP0.

## Smoke eval (file-local only)

- Optional MAE / RMSE (and light CRPS if cheap) on artificial gaps on the
  **held-out smoke** windows of **this upload**.
- Default smoke artificial protocol (when enabled): MCAR rate **0.30**
  (aligned with M4 demo-eval smoke), seed `seeds.mask = 0`.
- Documented and labeled as **file-local smoke**, **not** a new EVAL / M3
  claim and **not** cross-dataset SOTA.
- **Banned:** merge upload-adapt smoke metrics into `artifacts/m3/`; rewrite
  `docs/EVAL.md` Beijing protocols; advertise universal / cross-dataset SOTA
  from one-file adapt.

### Smoke protocol (locked defaults)

| Knob | Locked default | Notes |
|---|---|---|
| Train / smoke **window** split | **80% / 20%** by window index (adapt order) | Scaler fit uses **train** windows only |
| Artificial gaps | Applied **after** scaler fit on smoke windows | Only where `m_native == 1` |
| Default mask | MCAR **`smoke_mcar_rate = 0.30`** | `seeds.mask = 0`; optional block uses `block_len` default **8** |
| Metrics | MAE / RMSE / CRPS / coverage_50 / coverage_90 | Same metric cells as M3 spirit: artificial targets with GT only |
| Sampler for smoke | `sample.K_samples = 8`, `sample.solver_steps = 50` | Smoke caps, not M3 report (`50` / `100`) |
| Label | **“file-local smoke (this upload only)”** | CLI: `--smoke-eval`; UI: demo-check caption |
| Artifact home | Under `artifacts/m7/<run_id>/` only (optional side files) | **Never** `artifacts/m3/` or `artifacts/m2/` |

See also `docs/EVAL.md` § **M7 upload-adapt smoke** (does not expand the
Beijing claim table).

## Impute gate update (post-adapt)

- **Exact Beijing pretrained:** unchanged — load M2 weights → impute (M4).
- **Flexible (and Exact) after successful adapt:** session checkpoint +
  scaler for **this** schema may **unlock impute** for the current session /
  run (lift the M6 Flexible block that pointed at M7).
- Gate must bind to the **session** bundle under `artifacts/m7/<run_id>/`,
  not silently substitute Beijing weights on a mismatched schema.
- Failed / aborted adapt does **not** unlock Flexible impute.

## Downloads

- Same as M4: **required** mean-fill CSV in original units (observed cells
  preserved; targets = posterior mean); **optional** multi-sample long format.

## Extension points (reuse inventory)

| Module | Role for M7 (wrap / extend; do not fork blindly) |
|---|---|
| `src/train/real_loop.py` | `M2Trainer` / `RealConditionalDSMLoss` / train-step primitives — wrap with smoke caps + timeout; do not default `artifact_dir` to `artifacts/m2` |
| `src/train/artifacts.py` | `ArtifactPaths`, `save_checkpoint` — point root at `artifacts/m7/<run_id>/` |
| `src/data/normalize.py` | `FeatureScaler`, `fit_scaler_on_train_split` — train-native-observed only |
| `src/models/score_net.py` | `ConditionalScoreNet(n_features=F)`; refuse fine-tune on `F` mismatch |
| `app/preview_ingest.py` | `evaluate_impute_gate` — extend to allow session-bundle unlock after successful adapt |
| `app/session_impute.py` | Load session ckpt/scaler vs M2 paths |
| `app/flexible_ingest.py` / `app/window_adapter.py` | M6 ingest → `AdaptedWindows` input to adapt |
| `app/ui_session.py` / `app/streamlit_app.py` | Product wiring: Streamlit adapt panel + CLI (WP2–WP3) |
| `configs/real_m2.yaml` | Naming style for train / sample knobs; M2 claim paths stay read-only for fine-tune base |

## Banned behaviors

- Overwrite `artifacts/m2/checkpoints/*` as the “universal” model.
- Fold held-out artificial-gap smoke targets into scaler fit or claim tables.
- Merge upload-adapt smoke metrics into `artifacts/m3/` or rewrite EVAL
  Beijing protocols.
- Advertise universal / cross-dataset SOTA from one-file adapt.
- Silent remap of unknown columns to `PM2.5`…`O3`.
- Scope creep into **M8** gap routing or **M9** hosted checkpoint.
- Fine-tune when `F ≠` base `F`.
- Promote a checkpoint after NaN loss, timeout, or OOM abort.

## Claim discipline (adapt ≠ Beijing evidence)

- M7 is **file-local adapt**, not a new Beijing / M3 claim expansion.
- Portfolio / README numbers remain Beijing-only until a later milestone
  explicitly unlocks a new domain with measured evidence.
- One successful non-Beijing smoke run does **not** prove a universal
  imputer.

## Irregular time (status)

- **Deferred:** irregular wall-clock as first-class SDE dynamics is **not**
  promoted in M7. Timestamps for bin / display only; dynamics remain regular
  grid + masks.

## Out of scope for M7

- **M8** gap-regime routing (after adapt, single policy: always TimeFill)
- **M9** hosted checkpoint, second-domain SOTA claims, stronger baselines as
  product backends
- True irregular-time Neural SDE
- Full multi-hour UI train (smoke caps + timeout are the product default)
- Rewriting M1–M6 tensor / mask / sentinel contracts
- Silently changing Beijing `docs/EVAL.md` claim protocols
