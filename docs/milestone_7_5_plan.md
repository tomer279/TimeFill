# Milestone 7.5 — Model-correctness fixes (pre-M8)

**Status:** planned
**Depends on:** M2 (train path), M3 (benchmark contract), M7 (shared adapt path)
**Blocks:** M8 (gap-regime thresholds must come from M7.5 evidence, not M3)

This milestone fixes problems found in two code reviews (November 2026):

- **Track A — model correctness** (WP0–WP8): training and sampling of the
  TimeFill score model, plus the evaluation plumbing the re-benchmark relies on.
- **Track B — app and data correctness** (WP-B1–WP-B5): the upload → ingest →
  adapt → fill → download path (Streamlit + CLI) — see §9.

It is a **correctness and quality** milestone, not a feature milestone: no new
product surface, no new datasets, no new baselines. The two tracks touch
mostly different files and can run in parallel (separate Cursor sessions /
PRs); the cross-track dependencies are listed in §9.4.

The fixes come first because M8 sets its routing thresholds ("short gaps →
linear, long gaps → TimeFill") from the M3 numbers. If those numbers are
held back by fixable bugs, M8 would lock the bugs in.

---

## 1. Why: findings → work packages

| # | Finding (code review) | Where | Severity | WP |
|---|---|---|---|---|
| 1 | Train/sample mismatch: training gives the model **noised** observed cells; the sampler gives it **clean** observed cells at every `t` | `src/train/real_loss.py`, `src/train/loss.py`, `src/sample/reverse_sde.py` | Critical | WP1 |
| 2 | Unweighted score target `-eps/std(t)` with `t_eps=1e-5` → targets up to ~1000; loss swings 43–169; grad norm ~30–47 against a clip of 1.0 | `loss.py`, `real_loss.py`, `configs/real_m2.yaml` | Critical | WP2 |
| 3 | Network outputs the raw score with no scaling by noise level | `src/models/score_net.py` | High | WP2 |
| 4 | Best checkpoint picked from a noisy DSM loss on 8 val windows × 4 MC draws, fresh key each epoch; no EMA | `src/train/real_loop.py`, `src/train/real_pool.py` | High | WP4 |
| 5 | Tiny model (~17k params); receptive field 17 steps; time added only at the stem; no normalization | `score_net.py` | High | WP3 |
| 6 | Sinusoidal time embedding on raw `t∈[0,1]` → max angle ≤ 1 rad, poor resolution | `score_net.py::_sinusoidal_time_embedding` | Medium | WP3 |
| 7 | Plain z-score on skewed pollutants; CO dominates the pooled original-unit MAE/RMSE; samples can go negative | `src/data/normalize.py`, `src/eval/*` | Medium | WP5 |
| 8 | `obs_loss_weight: 0.1` trains the score on cells the sampler never uses | `configs/real_m2.yaml` | Medium | WP2 |
| 9 | Mixed mask mode picks random-vs-block once per batch; one fixed length-8 block per feature (same as the test protocol); batch size 8 | `real_loss.py`, `src/data/masking.py` | Medium | WP4 |
| 10 | Dense `(T·F)²` diffusion matrix via `jnp.diag` at every step | `reverse_sde.py::_make_diffusion_fn` | Medium (perf) | WP6 |
| 11 | MAE scored on the posterior mean (the median minimizes MAE) | `src/eval/m3_benchmark.py`, `metrics.py` | Low | WP6 |
| 12 | Timestamps validated but unused ("continuous-time" covers diffusion time only) | `reverse_sde.py::ImputeQuery` | Scope | Out of scope (see §8) |
| 13 | App windows imputed independently, no overlap → jumps at boundaries | `app/window_adapter.py` | Low | WP8 (optional) |
| 14 | M7 scratch adapt saved after 1 step on 1 window; outputs look like real imputations | `artifacts/m7/...`, `src/train/adapt_m7.py` | Low (UX honesty) | WP8 |
| 15 | Single mask seed / sample seed → no error bars on ~2-point MAE gaps | `configs/bench_m3.yaml` | Medium (evidence) | WP7 |
| — | Stray scratch file `src/data/Untitled-1.py` inside the package | `src/data/` | Housekeeping | WP8 |
| A5 | M3 test metrics not pinned to the trained split: re-running `prepare_m2_data.py` with another split mode/seed silently lets train data into "test" | `src/eval/m3_data.py`, prepare/train write path | High (evidence) | WP0 |
| A8 | Scaler "train-only / missing ignored" tests cannot fail: val tensors are deleted before fit; masked cells are overwritten with the sentinel before fit | `tests/test_normalize_fit_train_only.py` | High (evidence) | WP5 |
| A9 | Metrics return 0.0 when there are no target cells (weight sum floored at 1.0) → a degenerate run looks perfect | `src/eval/metrics.py::_weighted_mean` | Medium | WP5 |
| A11 | Features with no train observations silently get mean 0 / std 1 (identity scaler) → unscaled values into the net | `src/data/normalize.py::_feature_moments`, `src/train/adapt_m7.py` | Medium | WP5 |

Rows **A5–A16** use the numbering of the second (app/data) audit. A5, A8, A9
and A11 are in Track A because the WP7 re-benchmark is not trustworthy without
them; the rest are in Track B (§9).

---

## 2. Ground rules (read before every WP)

1. **Frozen claim artifacts.** Never overwrite `artifacts/m2/`, `artifacts/m3/`,
   `artifacts/m7/`, or the v1 tables in `docs/EVAL.md`. All M7.5 outputs go to
   `artifacts/m7_5/` (train, benchmark, and ablation subfolders).
2. **Legacy-preserving flags.** Every behavior change sits behind a config flag
   whose **default reproduces the current (legacy) behavior** in the existing
   configs (`toy_m1.yaml`, `real_m2.yaml`, `bench_m3.yaml`, `app_m4.yaml`,
   `adapt_m7.yaml`). New behavior is turned on in the **new** configs
   `configs/real_m7_5.yaml` and `configs/bench_m7_5.yaml`. The existing test
   suite must keep passing unchanged.
3. **Checkpoint compatibility.** The old M2 `.eqx` checkpoint must still load
   with the legacy architecture. New checkpoints save a sidecar
   `model_config.json` (`ScoreNetConfig.to_dict()` + `parameterization` +
   architecture version). Loaders read the sidecar when present and fall back
   to the legacy defaults when absent.
4. **Tensor contract unchanged:** `(B, T, F)`, `m: 1=observed, 0=missing`,
   sentinel `0.0` on missing cells, never infer missingness from values.
5. **One WP per Cursor session / PR.** Run `pytest` after each WP. Add tests
   in the same WP as the change.
6. **No new claims in README** until WP7 numbers exist; then report wins
   *and* losses.

---

## 3. Config surface (new keys)

Add to the config dataclasses (`src/data/config.py`, `src/data/real_config.py`,
`src/data/adapt_config.py`) with legacy defaults:

```yaml
# configs/real_m7_5.yaml (new values shown; legacy default in comments)
data:
  transform:                       # legacy: none
    log1p_features: [PM2.5, PM10, SO2, NO2, CO]   # O3 stays linear
    clip_nonnegative: true         # legacy: false (clip in original units on restore)
  batch_size: 32                   # legacy: 8

model:                             # legacy: ScoreNetConfig defaults
  arch: tcn_v2                     # legacy: tcn_v1
  hidden_channels: 64              # legacy: 32
  dilations: [1, 2, 4, 8, 16, 32]  # legacy: n_layers=3 → [1, 2, 4]
  kernel_size: 3
  time_embed_dim: 64               # legacy: 32
  time_embed_scale: 1000.0         # legacy: 1.0
  time_cond: per_block             # legacy: stem
  norm: group                      # legacy: none
  norm_groups: 8

train:
  parameterization: eps            # legacy: score
  t_eps: 1.0e-3                    # train time floor; legacy: 1.0e-5
  obs_loss_weight: 0.0             # legacy: 0.1 (real_m2.yaml)
  observed_input: clean            # legacy: noised  (WP1)
  mask_mode: mixed
  mixed_select: per_window         # legacy: per_batch
  block_len_range: [4, 16]         # legacy: fixed block_len
  max_blocks_per_feature: 2        # legacy: 1
  ema_decay: 0.999                 # legacy: null (no EMA)
  lr_schedule: warmup_cosine       # legacy: constant
  warmup_steps: 1000
  select_by: val_impute_crps       # legacy: val_dsm_loss
  val_impute:
    n_windows: 64
    K_samples: 8
    solver_steps: 50
    mask_seed: 0
    sample_seed: 1
    protocols: [mcar_0.3, block_8]

sample:
  t_eps: 1.0e-3                    # sampler endpoint; legacy: train.t_eps
  observed_mode: frozen_clean      # legacy: frozen_clean  (alt: replace_noised)
  diffusion_impl: diagonal         # legacy: dense
  point_estimate: median           # legacy: mean (MAE only; RMSE keeps mean)
  K_samples: 50
  solver_steps: 100
```

Every new key gets a `validate()` check and a row in `ASSUMPTIONS.md`
(new **Milestone 7.5** section).

---

## 4. Work packages

### WP0 — Contract, configs, scaffolding
- Add the config keys from §3 with legacy defaults; add `configs/real_m7_5.yaml`
  and `configs/bench_m7_5.yaml` (same protocols as `bench_m3.yaml`, plus
  `seeds.mask: [0, 1, 2]`, `seeds.sample: [1, 2, 3]`, checkpoint →
  `artifacts/m7_5/train/checkpoints/best_score_net.eqx`).
- Add the `ASSUMPTIONS.md` **Milestone 7.5** section: list each flag, its
  legacy default, its M7.5 value, and why.
- Add the `model_config.json` sidecar writer/reader (`src/train/artifacts.py`)
  and loader fallback.
- **Split pinning (A5):** at train time, write `split_manifest.json` next to
  the checkpoint: split mode, split seed, fractions, per-split entity lists,
  and a SHA-256 of the sorted train/val/test window indices plus the
  `windows.npz` content hash. `m3_data.py` (and the M7.5 benchmark) recompute
  this from the current `windows.npz` and **refuse to run** on mismatch, with
  a message naming the differing field. Legacy checkpoints without a manifest:
  warn loudly and record `split_pinned: false` in the benchmark output
  (don't break the v1 reproduce path).
- **Tests:** configs parse; every legacy config resolves to legacy values;
  loading the existing M2 checkpoint still works; split manifest round-trip;
  a benchmark load with a re-prepared (different seed or `index` mode)
  dataset raises; a legacy checkpoint without a manifest warns but runs.
- **Done when:** `pytest` is green with zero behavior change.
- **Done (2026-11-21):** Scaffolding only (no train/sample behavior change).
  Added `real_m7_5`/`bench_m7_5` YAMLs, ASSUMPTIONS §M7.5, `model_config.json`
  sidecar helper, A5 `split_manifest` write+verify, train refuse for unused
  M7.5 flags and for `artifacts/m2|m3|m7` (+ subdirs). Decisions: pin authority
  = two hashes; clear stale best before new pin; `split_pinned` in JSON not CSV;
  sidecar loaders switch in WP3; impute/bench frozen writes deferred to WP8.
  Commands: `pytest -q` → **154 passed** in ~65s.

### WP1 — Make conditioning consistent between training and sampling (finding 1)
- In `ConditionalDSMLoss.__call__` and `RealConditionalDSMLoss.__call__`, after
  `y_t = forward_sample(...)`, apply:
  ```python
  if observed_input == "clean":
      y_t = m_train * x_obs + (1.0 - m_train) * y_t
  ```
  This matches what `ReverseSDEImputer` does today (`missing_only` init +
  freezing + replacement).
- Add `sample.observed_mode = replace_noised` as an alternative: at each step,
  set observed coordinates to `mean(t)·x_obs + std(t)·z` before calling the
  model (Song et al. replacement). This pairs with `observed_input: noised`
  and is used only in the WP7 ablation.
- Make `ImputeContext.validate()` (or a new helper) **warn** when the pairing
  is inconsistent (`clean` + `replace_noised`, or `noised` + `frozen_clean`),
  and log the pairing to `adapt_meta.json` and the train metadata.
- **Tests:**
  - With `observed_input=clean`, the model's `y_t` input equals `x_obs` on
    `m_train==1` cells (use a spy/stub model).
  - Natively missing cells are not affected (`m_train==0`).
  - `replace_noised` sampler: observed cells at step `t` have empirical std
    close to `std(t)` over many draws (statistical test with loose tolerance).
- **Done when:** both pairings run end-to-end on the toy config.
- **Done (2026-11-21):** Train `observed_input=clean` overwrite in toy+real DSM;
  sampler `replace_noised` (model-input only; freeze+final clean replace kept);
  pairing warn helper (legacy `noised`+`frozen_clean` silent); `clean` requires
  `obs_loss_weight=0`; DSM validate runs in `_assert_wp0_consumable` before
  artifact wipe; `adapt_meta` logs `observed_input` only. Decisions: re-refuse
  `sample.observed_mode != frozen_clean` until WP7/WP-B4 wire val/bench/app;
  train_meta pairing deferred to WP4. Commands:
  `pytest tests/test_wp1_observed_conditioning.py -q` → **10 passed**;
  `pytest -q` → **164 passed** in ~70s.

### WP2 — Noise-prediction parameterization and loss weighting (findings 2, 3, 8)
- Add `parameterization: score | eps`.
  - `eps`: the network output is `ε̂`; loss is
    `weighted_mean((ε̂ − ε)², weights)`; the score passed to the sampler is
    `−ε̂ / std(t)`.
  - Do the conversion in a small wrapper (e.g. `ScoreFromEps(model, sde)` in
    `src/models/`) so `reverse_sde.py` keeps calling a score model and stays
    agnostic.
- Split the time floors: `train.t_eps` (sampling of `t` during training,
  M7.5 value `1e-3`) and `sample.t_eps` (solver endpoint). Keep the existing
  `t_eps < T_diff` validation for both.
- Set `obs_loss_weight: 0.0` in `real_m7_5.yaml`.
- **Tests:**
  - At `t = t_eps`, the `eps` loss of a zero-output model is ≈ 1 (unit
    Gaussian target), not ≈ 1/std².
  - `ScoreFromEps` matches `−eps/std` exactly on random inputs.
  - Short training (e.g. 200 steps, toy config): the `eps` run's loss
    coefficient of variation over the last 100 steps is lower than the
    `score` run's (regression guard, loose threshold).
- **Done when:** the toy M1 path trains and imputes with `parameterization: eps`.
- **Done (2026-11-21):** Toy+real DSM `parameterization: eps` (target `ε`); plain
  `ScoreFromEps` + `context_from_*` wrap; `resolve_sample_t_eps`; `obs_loss_weight: 0`
  already in `real_m7_5.yaml`. Decisions: keep real refuse for `eps`/`sample.t_eps`
  until val hook wraps; adapt refuses non-`score` until WP-B4; drop unused
  `std_floor` field. Commands:
  `pytest tests/test_wp2_eps_parameterization.py -q` → **6 passed** (~34s);
  `pytest -q` → **170 passed** in ~90s.

### WP3 — Score network v2 (findings 5, 6)
- Add `arch: tcn_v2` next to the current network (keep `tcn_v1` untouched
  for old checkpoints). Proposed block:
  ```
  h → GroupNorm → SiLU → DilatedConv(k, d) → + FiLM(time_emb) →
      GroupNorm → SiLU → Conv1x1 → (+ residual) * 1/√2
  ```
  Dilations from config (default `[1, 2, 4, 8, 16, 32]`, receptive field
  127 ≥ T=64). Time conditioning in **every** block (FiLM scale/shift, or at
  least an additive bias).
- Time embedding: multiply `t` by `time_embed_scale` (1000) before the
  sinusoids. Optionally embed `log std(t)` instead; pick one and document it.
- Optional (add only if the WP7 ablation shows a gain): a light
  cross-feature mixing layer per block (a 1×1 conv already mixes features;
  a small attention over `F=6` is cheap).
- Add `receptive_field(config) -> int` and expose it in the train log.
- **Tests:**
  - Output shapes for `F ∈ {1, 3, 6}`, `T ∈ {16, 64}`.
  - `receptive_field(tcn_v2 default) ≥ 64`.
  - The time embedding varies noticeably between `t=0.001` and `t=0.01`
    (cosine distance above a threshold).
  - Parameter count is logged; target ~150–400k for the M7.5 config.
- **Done when:** the v2 net trains on the toy path and old v1 checkpoints still load.
- **Done (2026-11-23):** `tcn_v2` beside frozen `tcn_v1`; FiLM `(1+γ)`;
  scaled-`t` embed; stem-included RF; loader uses `build_score_net`;
  `train_meta.receptive_field` filled. Decisions: v1 rejects non-legacy
  knobs; v2 requires `per_block`; toy done-when injects the model;
  `model.*` refuse stays until pre-WP7 wires `config.model` + sidecar write.
  Numbers: M7.5 config **172550** params, RF **129**. Commands:
  `pytest tests/test_wp3_score_net_v2.py -q` → **16 passed**;
  `pytest -q` → **211 passed** in ~125s.

### WP4 — Training hygiene: EMA, selection by imputation, masks, schedule (findings 4, 9)
- **EMA:** keep an EMA copy of the model weights (`optax.ema` or a manual
  tree-map); save and evaluate the **EMA** weights.
- **JIT reuse (WP6 follow-up):** each `impute()` rebuilds drift/diffusion
  closures, so Diffrax recompiles every call. Val imputation every
  `val_every` epochs will pay full compile unless you wrap the imputer once
  (`eqx.filter_jit` on a stable runner, or inside `ReverseSDEImputer`) and
  reuse it across epochs. Do this before relying on WP6 diagonal for cheap
  val selection.
- **Checkpoint selection:** replace "min val DSM loss" with
  `select_by: val_impute_crps` (or `val_impute_mae`):
  - Build a **fixed** val imputation set once (`n_windows` val windows,
    fixed artificial masks from `mask_seed`, fixed sampler key).
  - Every `val_every` epochs, run the reverse sampler (`K=8`,
    `solver_steps=50`) on the EMA model and compute CRPS/MAE in normalized
    space (cheap) on artificial targets.
  - Keep logging DSM loss for curves, but don't use it for selection.
- **Masks:**
  - `mixed_select: per_window` — draw the random-vs-block choice per window
    (shape `(B,1,1)`), not per batch.
  - `block_len_range: [lo, hi]` and `max_blocks_per_feature` in
    `BlockMasker` (legacy = fixed length, 1 block).
- **Schedule:** `warmup_cosine` LR schedule in Optax; `batch_size: 32`.
- Write `artifacts/m7_5/train/train_meta.json` with the clipped-step
  fraction, median grad norm, EMA decay, parameter count, receptive field,
  and the selected epoch.
- **Tests:**
  - The val imputation set is identical across two calls (same masks/keys).
  - EMA weights differ from raw weights after N steps and equal them when
    `decay=0`.
  - `per_window` mixing yields both mask types inside one batch (with a
    fixed key).
  - Block masks with a range produce lengths within `[lo, hi]`.
- **Done when:** a full hygiene train finishes and its train meta reports a
  clipped-step fraction < 0.5 (target; report actual). Full `real_m7_5.yaml`
  still waits on WP3 model.* / WP5 transform.
- **Done (2026-11-22):** EMA, val-impute select (model-arg JIT), masks,
  warmup_cosine, `train_meta`, fallback ckpt, `configs/real_m7_5_hygiene.yaml`.
  Decisions: skip PNG under impute-select; `selected_epoch=null` on fallback;
  seeded val subset; multi-block = union. Hygiene smoke (5 ep × 40 steps,
  tiny val_impute): `clipped_step_fraction=0.0`, `median_grad_norm≈0.44`,
  `selected_epoch=5`, `parameter_count=17286`. Commands:
  `pytest tests/test_wp4_train_hygiene.py -q` → **14 passed**;
  shortened `run_m2_training_from_config(hygiene)` → `artifacts/m7_5/hygiene/`.

### WP5 — Feature transform and per-feature evaluation (finding 7)
- Extend `FeatureScaler` with an optional per-feature `log1p` before the
  z-score; inverse = z-inverse then `expm1`; optional nonnegative clip on
  restore. Persist the transform spec in `feature_scaler.npz` (legacy files
  without the key load as "no transform").
- Fit stays **train-only on natively observed cells** (unchanged rule).
- **No silent identity scaler (A11):** when a feature has zero observed cells
  in the fit slice, raise a `ValueError` naming the feature(s) and the fit
  slice (train split / adapt train fraction). Apply the per-feature
  observed-rate floor (`min_observed_rate_per_feature`) to the **train slice**
  in the M7 adapt path, not only globally. Legacy behavior is behind
  `allow_empty_feature_fallback: true` (legacy default) so v1 configs are
  unchanged; `real_m7_5.yaml` and the M7.5 adapt opt-in set it to `false`.
- **Empty-target metrics (A9):** `_weighted_mean` returns `NaN` when the weight
  sum is 0 (no floor). Callers that compare or aggregate must handle NaN
  explicitly; `metrics_by_rate.*` writes `n_target_cells` on every row; the
  Streamlit quick-check shows "no target cells" instead of a number.
- **Make the leakage tests real (A8):** rewrite
  `tests/test_normalize_fit_train_only.py` so it can fail:
  - Fit on train, then **perturb val/test windows** (e.g. ×1000) and refit;
    assert the stats are identical.
  - Put huge values and `NaN` under `m=0` cells **before** fit (no sentinel
    overwrite); assert the stats equal a fit with those cells removed.
  - Mutation check (one-off, document in the walkthrough): temporarily fit
    on all windows and confirm the test fails.
- Eval: add per-feature MAE/RMSE/CRPS/coverage and a **normalized-unit**
  aggregate next to the pooled original-unit metrics in `metrics_by_rate.*`
  (new columns/rows; old columns keep their meaning).
- **Tests:**
  - transform → inverse round-trip on observed cells (tolerance).
  - Negative restored values are clipped to 0 when enabled.
  - Legacy scaler files load unchanged.
  - Per-feature metrics averaged with cell-count weights equal the pooled metric.
  - Zero-observation feature in the fit slice raises (new flag) / falls back
    (legacy flag).
  - All-observed mask → metrics are NaN, not 0.0.
- **Done when:** the benchmark writes per-feature tables.
- **Done (2026-11-23):** `FeatureScaler` log1p/clip/empty-fallback (A11);
  A8 leakage tests via `fit_scaler_on_train_split`; A9 NaN empty weights;
  mean-after-restore; true per-feature MAE/RMSE/CRPS/coverage +
  `per_feature.csv` / `*_norm`; NPZ names + bench-config fallback.
  Decisions: clip all; raise on log1p `x<0`; adapt floor when fallback
  false; no adapt log1p; legacy NPZ → empty names. Commands:
  `pytest tests/test_wp5_feature_transform.py tests/test_normalize_fit_train_only.py -q`
  → **19 passed**; `pytest -q` → **229 passed** in ~135s.

### WP6 — Sampler efficiency and point estimate (findings 10, 11)
- `diffusion_impl: diagonal`: replace `jnp.diag(...)` with a diagonal
  operator. Diffrax ≥ 0.5 depends on `lineax`; check the installed version
  and use `lineax.DiagonalLinearOperator` in the `ControlTerm`. If that is
  unavailable, fall back to a manual Euler–Maruyama `jax.lax.scan` with
  per-coordinate noise (keep Diffrax as the default path for positioning).
- `point_estimate: median` for MAE; RMSE keeps the mean. Store both
  `mean.csv` and `median.csv` (app downloads stay mean by default unless
  the config says otherwise).
- **Tests:**
  - Dense vs diagonal give identical samples (same key, small window,
    `atol` ~1e-5).
  - Median path is used for MAE when configured.
- **Done when:** a timing note (CPU, `K=50`, `steps=100`, one window) is
  recorded in the walkthrough for dense vs diagonal.
- **Done (2026-11-22):** `diffusion_impl` (lineax diagonal) wired through
  sampler, val snapshots, and M3 bench; `masked_mae(point_estimate=)`;
  `write_mean_median_csvs` + `scripts/wp6_timing.py` → `artifacts/m7_5/wp6/`.
  Decisions: no EM fallback (`lineax` required); app stays dense until WP-B4;
  config→bench MAE deferred to WP7; WP4 must reuse one jitted imputer (bare
  `impute()` recompiles). Warm CPU T=64 F=6 K=50 steps=100: dense **0.726s**,
  diagonal **0.723s**. Commands: `pytest tests/test_wp6_sampler_efficiency.py -q`
  → **11 passed**; `python scripts/wp6_timing.py`.

### WP7 — Re-benchmark, ablation, and claim update (finding 15)
- Train with `real_m7_5.yaml` → `artifacts/m7_5/train/`.
- Run `bench_m7_5.yaml` on **test**: all M3 protocols (MCAR 10/30/50%,
  block 8) × 3 mask seeds × 3 sample seeds; report mean ± std. Add a block-16
  protocol as an **extension** row (clearly labelled, not part of the frozen
  M3 grid).
- **Point estimate (from WP6):** bench / `ImputationEvaluator` must read
  `sample.point_estimate` and pass it into `masked_mae` so M7.5 tables use
  median MAE when the config says `median` (RMSE stays on the mean).
- **Ablation** (fewer seeds and a shorter train budget are fine; document it):
  1. M2 legacy checkpoint (reference)
  2. + WP1 consistency only
  3. + WP2 eps / t_eps
  4. + WP3 net v2
  5. + WP4 EMA / selection / masks
  6. + WP5 log1p (full M7.5)
  7. Alternative pairing: `noised` + `replace_noised`
- Outputs: `artifacts/m7_5/bench/metrics_by_rate.{csv,json}`,
  `artifacts/m7_5/bench/per_feature.csv`, `artifacts/m7_5/ablation.csv`,
  a comparison panel PNG.
- Docs: new **M7.5** section in `docs/EVAL.md` (v1 tables untouched);
  `docs/m7_5_results.md` with a claim checklist in the same style as
  `docs/m3_failure_notes.md`; README "Results" gains an M7.5 row only after
  the numbers exist.
- **M8 hand-off:** add a short table "gap regime → best method by seed-averaged
  MAE and CRPS" (short MCAR, high MCAR, block 8, block 16). M8 thresholds are
  derived from this table.
- **Done when:** the tables exist, each claim is marked pass/fail honestly,
  and the M8 hand-off table is written.

### WP8 — Housekeeping and honesty fixes (findings 13, 14, cleanup)
- Delete `src/data/Untitled-1.py` (move it to `notebooks/scratch/` if it is
  worth keeping).
- M7 adapt: if `epochs × steps_per_epoch < min_adapt_steps` (new key, e.g.
  200), mark the run `status: smoke_only` in `adapt_meta.json` and show a
  Streamlit warning saying the imputations are not quality results.
- Make sure the M7 adapt path picks up the new loss/model flags through
  `adapt_m7.yaml` (legacy default stays legacy; document how to opt in).
- **Optional:** overlapping windows plus stitching in `app/window_adapter.py`
  (stride `T/2`, average or crossfade samples in overlaps). Can move to M8
  if time is short.
- **Tests:** smoke-only flag logic; adapt run with the M7.5 flags on the
  toy fixture.

---

## 5. Suggested order and effort

| Order | WP | Rough size | Notes |
|---|---|---|---|
| 1 | WP0 | S | No behavior change |
| 2 | WP1 | S | One-line core fix + tests; biggest conceptual bug |
| 3 | WP2 | M | Biggest stability win |
| 4 | WP6 | S | Makes later val-imputation selection affordable |
| 5 | WP4 | M | Needs WP6 for cheap val imputation |
| 6 | WP3 | M | Easier to judge once training is stable |
| 7 | WP5 | S–M | |
| 8 | WP8 | S | |
| 9 | WP7 | L (compute) | Needs everything above; GPU recommended for the seed grid |

Track B (§9) runs in parallel; its suggested order is in §9.3. WP0's split
pinning and WP5's scaler/metric fixes must be merged **before** WP7.

If compute is tight: run WP7 with 1 seed for the ablation and 3 seeds only
for the final M7.5 config.

---

## 6. Exit criteria (Definition of Done)

1. All WP0–WP6 and WP8 changes are merged behind flags; legacy configs
   reproduce legacy behavior; `pytest` green (old and new tests).
2. A full `real_m7_5.yaml` training run completes with `train_meta.json`
   reporting: clipped-step fraction, median grad norm, selected epoch, EMA
   used, parameter count, receptive field.
3. `bench_m7_5.yaml` results on test with ≥ 3 mask seeds, mean ± std,
   pooled + per-feature + normalized-unit metrics.
4. Ablation table covering at least rows 1, 2, 3, and 6 from WP7.
5. `docs/m7_5_results.md` with an honest claim checklist and the M8 hand-off table.
6. v1/M3 artifacts and tables are byte-identical to before M7.5.
7. The M7.5 benchmark runs only against a split-pinned checkpoint (A5), and
   the rewritten scaler tests fail under the documented mutation check (A8).
8. Track B exit criteria (§9.5) are met.

**Target results (to aim for, not exit gates; report either way):**
- TimeFill MAE ≤ linear on MCAR 30% and block 8 (seed-averaged).
- CRPS better than M3 TimeFill on every protocol.
- coverage_90 within 0.85–0.95 on every protocol (M3 block was 0.81).
- Training: clipped-step fraction < 0.5; no val-loss swings > 10× between
  consecutive epochs.

---

## 7. Cursor workflow tips

- Open one chat per WP. Paste the WP section plus §2 (ground rules) and §3
  (config keys) as context, and attach the files named in §1 (Track A) or
  §9.1 (Track B) for that WP. For Track B, also paste §9.4.
- Prompt pattern:
  > Implement WP\<n> from `docs/milestone_7_5_plan.md`. Follow the ground
  > rules in §2: legacy defaults, no writes to `artifacts/m2|m3|m7`, add the
  > listed tests. Show me the diff plan before editing.
- After each WP: `pytest -q`, then a toy smoke run
  (`python scripts/run_m1_toy.py ...`) before moving on.
- Keep a running `docs/milestone_7_5_walkthrough.md` (same style as the
  M1–M7 walkthroughs) with what changed, the commands run, and the numbers.

---

## 8. Explicitly out of scope for M7.5

- Using wall-clock timestamps / irregular Δt in the model (finding 12).
  It stays a documented limitation; it is a candidate for M9 or a stretch
  goal. README must not advertise irregular-time support.
- CSDI / BRITS / Transformer baselines (still M9 track 3).
- Gap-regime routing (M8).
- New datasets or a hosted checkpoint (M9).
- Multi-user / hosted security hardening. Track B assumes the current load
  model: one person on a laptop running the CLI and Streamlit.

---

## 9. Track B — App and data correctness

The second audit looked at the product path: Streamlit/CLI upload → M6 ingest
→ optional M7 adapt → fill → CSV download. It found ways the app can return
fills for the **wrong data** or a download that **mixes invented and
overwritten values**, with no error. These do not depend on the model fixes,
but they must land before M8 adds routing on top of the same path.

### 9.1 Findings → work packages

| # | Finding | Where | Severity | WP |
|---|---|---|---|---|
| B1 | Adapt-session fingerprint is filename + ingest mode + feature names only → a different file with the same name and columns keeps the **old** windows and session model for Fill | `app/adapt_session_ui.py` (~L412-434), `app/streamlit_app.py` (~L864-877) | Must fix | WP-B1 |
| B2 | A failed re-adapt (both the "too little data" and the adapt-failure branch) returns without clearing `_SESSION_ADAPT_KEY`; `_resolve_bound_session` then returns the old bind because the fingerprint still matches. The error says "No session checkpoint was promoted" while Fill keeps working | `app/streamlit_app.py` (L669-696) | Should fix (stale, not wrong, on its own; wrong when combined with B1) | WP-B1 |
| B3 | Multi-entity CSV silently collapsed: entity taken from the first row only; no distinct-entity check anywhere; hourly binning keeps the last **observed** value per hour **per feature**, so one output row can mix values from different sites, and sites reporting at different hours are interleaved into one series. All labelled as the first site | `app/schema_validate.py` (L492-519), `app/window_adapter.py` (L396-455) | Must fix | WP-B2 |
| B4a | "Completed CSV" exports right-pad rows as real data (e.g. 100 rows → 128). Pad cells can't be told apart from native gaps after windowing: `AdaptedWindows` has no real-length / pad marker | `app/window_adapter.py` (`_cut_padded_windows`, `AdaptedWindows`), `app/export_frames.py` (L135-157) | Must fix | WP-B3 |
| B4b | With "Simulate extra gaps" on (off by default), `merge_mean_fill(..., m_eval)` replaces the hidden **true** values with model means in the download, while the help text says "Observed values kept" | `app/session_impute.py` (L750-754, L879-915), `app/streamlit_app.py` (L528-541) | Should fix | WP-B3 |
| B4c | Download has only `window_id` / `t_index`: no timestamps or entity. The binned hourly grid is not kept in `AdaptedWindows`, so the export has nothing to attach | `app/window_adapter.py`, `app/export_frames.py` | Should fix | WP-B3 |
| B17 | Stale results without any adapt: `_SESSION_RESULT_KEY` / `_SESSION_FEATURES_KEY` are cleared only on bind mismatch, gate failure or a successful adapt. On the pretrained Exact path (no bind), Fill the sample, then upload another Exact file, and the old charts and **download buttons** stay on screen under the new preview until Fill is pressed again | `app/streamlit_app.py` (L167-195, L698, L800, L810-825) | Must fix | WP-B1 |
| B18 | Silent truncation: `max_windows` (4 in `app_m4.yaml`) keeps only the first windows (`window_adapter.py` L539-542), so a long upload's "completed CSV" covers only the first 4 × 64 = 256 hours. Mentioned in the config and README, not in the UI | `app/window_adapter.py`, `configs/app_m4.yaml`, `app/streamlit_app.py` | Must fix (at minimum a visible warning) | WP-B3 |
| B6 | Exact Beijing path cuts windows by row order; no time sort/bin, although the UI says "hourly resampling" | `app/ui_session.py` (~L624-630), `app/streamlit_app.py` (~L618) | Should fix | WP-B2 |
| B7 | `inf` / `1e999` parsed as observed values | `app/schema_validate.py::_parse_numeric_cell` (~L441-450) | Should fix | WP-B2 |
| B10 | No test that NaN loss / timeout really skip writing `best_score_net.eqx` | `src/train/adapt_loop.py`, `src/train/adapt_m7.py` (~L446-455) | Should fix | WP-B4 |
| B12 | CLI and Streamlit build the session imputer with duplicated `SessionAssetRequest` wiring → sample / `t_eps` / SDE knobs can drift | `app/adapt_session_ui.py` (~L501-533), `app/adapt_upload_m7.py` (~L511-543) | Should fix | WP-B4 |
| B13 | Flexible inference: an odd NA token can drop a column; `year`/`month`/`day`/`hour` can become features; a step-index `time` column can explode the hourly grid | `app/flexible_ingest.py` | Nice to have | WP-B5 |
| B14 | Dead adapt knobs: `adapt.base_scaler`, `confirm_same_semantics` never set | `configs/adapt_m7.yaml`, adapt config | Nice to have | WP-B5 |
| B15 | Nested caption types that exist mainly to satisfy a 7-field dataclass limit test | `tests/test_adapt_session_ui.py` (~L86-93), `app/adapt_session_ui.py` | Nice to have | WP-B5 |
| B16 | `pytest` is a core install dependency | `pyproject.toml` (~L42) | Nice to have | WP-B5 |

B1–B4c and B17–B18 were traced in the code (Nov 2026); line numbers were
correct at that time but may drift, so confirm before editing. B17 and B18
were not in the audit; they came from that trace. Other line numbers are from
the audit. Audit items A5, A8, A9 and A11 are in Track A (WP0 / WP5).

### 9.2 Work packages

#### WP-B1 — Session binding integrity (B1, B2, B17)
- **Fingerprint by content:** add a content digest to the adapt/impute
  session fingerprint: SHA-256 of the uploaded bytes (preferred), or of
  `shape + values + masks` after ingest. Keep filename / ingest mode /
  feature names as extra fields for display only.
- **Invalidate on change:** when the fingerprint changes, clear the adapt
  bind, the impute results and any cached windows **before** rendering
  preview or Fill. Put this in one helper (e.g. `reset_session_binding()`)
  used by both the upload handler and Run adapt.
- **Results are bound too (B17):** store the data fingerprint (and the
  bound run id, or `pretrained`) with every Fill result in session state.
  Before rendering charts or download buttons, compare it with the current
  upload's fingerprint and drop the result on mismatch. This covers the
  pretrained Exact path, which has no adapt bind, and switching between the
  bundled sample and an upload.
- **Failed adapt clears state:** on any Run adapt failure (too few windows,
  NaN loss, timeout, exception), call the same reset helper, then show the
  error. The error text should say that no model is bound.
- Show the bound run id and data fingerprint (short hash) next to the Fill
  button so the user can see what Fill will use.
- **Tests:**
  - Same filename + same columns + different values → the bind is cleared and
    Fill can't run until adapt is re-run.
  - Success then failure → no bind remains; Fill is disabled.
  - Identical re-upload keeps the bind (no needless re-adapt).
  - Pretrained path: Fill on the sample, then load a different Exact file →
    no charts or download buttons until Fill is pressed for the new file.
  - Switching sample ↔ upload drops results from the other source.
- **Done when:** both scenarios from the audit are covered by tests that fail
  on the current code.
- **Done (notes):** Content-hash bind (SHA-256 of raw CSV bytes) in
  `app/adapt_session_ui.py` + Streamlit wiring: digest change clears
  bind/results before preview; Fill results store the digest; Run adapt
  starts with `reset_session_binding` so exceptions cannot leave a stale
  model; caption shows `run_id`/`pretrained` + short hash. Decisions: bind
  key = digest only (label/mode/features display-only); shared `SESSION_*`
  keys; skip AppTest / `model_ref`-on-results (results still cleared when
  the model or digest changes). Command:
  `py -m pytest tests/test_session_binding.py tests/test_adapt_session_ui.py -q`
  → **13 passed**.

#### WP-B2 — Ingest correctness (B3, B6, B7)
- **Multi-entity:** read the entity column over **all** rows, not just the
  first (`_parse_entity_id` today). If it has more than one distinct value,
  **reject** with a message listing the entities and telling the user to
  filter to one (or, if cheap, offer a picker in the preview). Don't
  auto-merge. Apply the check before hourly binning, so per-feature mixing
  across sites (B3) can't happen.
- **Duplicate timestamps:** after hourly binning, count hours with more than
  one raw row. Above a config threshold (e.g. `max_duplicate_hour_share:
  0.01`), reject; below it, warn and record the count in the preview. Prefer
  the **mean** of observed values in a bin over "last value" and document it
  in ASSUMPTIONS (legacy "last" behind a flag if tests depend on it).
- **Exact Beijing path:** if a timestamp column exists, sort by time and
  apply the same hourly bin/reject rules as Flexible (multi-entity reject,
  non-hourly cadence warn/reject). If no timestamps exist, keep row order but
  **remove the "hourly resampling" wording** from the Exact UI copy.
- **Non-finite values:** `_parse_numeric_cell` rejects `inf`, `-inf` and
  overflow (`1e999`) with row/column context in the error. Also check
  `np.isfinite` once more after the numeric pass.
- **Tests:** the audit's A-then-B two-sensor fixture (24h each) is rejected;
  a fixture where site B is missing one feature in hours where site A has it
  is rejected (it would otherwise mix sites within a row); two sites at
  different hours are rejected (they would otherwise interleave);
  descending timestamps in an Exact file come out sorted; a 30-minute Exact
  file is binned or rejected per the rule; `inf` and `1e999` raise with
  row/column.
- **Done when:** none of the audit's confirmed repros produce windows silently.

#### WP-B3 — Honest export (B4a, B4b, B4c, B18)
- **Prerequisite: carry provenance through windowing.** Extend
  `AdaptedWindows` with:
  - `valid_len` `(B,)` (or a `pad_mask` `(B, T)`), set by
    `_cut_padded_windows`, so pad cells can be told apart from native gaps;
  - `row_index` `(B, T)`, mapping each cell to its source row (or binned
    hour), with `-1` on pad cells;
  - `timestamps` `(B, T)` when the frame had timestamps (after binning:
    the hourly grid), `NaT` on pad cells.

  Legacy callers keep working (new fields have defaults when built from
  arrays without provenance). The model and sampler are unchanged: pad cells
  stay `m=0` for imputation and are dropped only at export.
- **Separate the two outputs:**
  - **Best-estimate CSV (download):** start from the user's **native** data.
    Keep every `m_native == 1` value exactly; fill only `m_native == 0` cells
    with the point estimate: merge with `m_native`, never `m_eval` (B4b).
    Drop pad cells using `valid_len` / `row_index` (B4a). Carry `timestamp`
    and `entity` columns when the input had them; otherwise keep `row_index`
    from the original file (B4c). Add a per-cell
    `is_imputed` flag (wide `<feature>_imputed` boolean columns or a long
    export; pick one and document it).
  - **Demo-evaluation view:** the artificial demo mask (`m_eval`) is used
    **only** for the chart and the quick-check metrics, never for the
    download.
- Multi-sample (long) export follows the same rules: no pad rows; observed
  cells either omitted or flagged.
- **Window cap (B18):** when `max_windows` drops windows, the UI must say so
  next to the Fill button and the downloads ("Filled the first N of M hours;
  raise `smoke.max_windows` or use the CLI for the full series"), and the
  download file name or a header column must show it is partial (e.g.
  `timefill_mean_fill_partial_rows_0-255.csv`). Optional: process the full
  series in batches of `max_windows` with a progress bar instead of
  truncating; if so, keep the warning for timeouts.
- Update the help text in `app/streamlit_app.py` to match what the file
  contains.
- **Tests:**
  - 100-row input with `seq_len=64` → exactly 100 output rows.
  - A native gap in the last real row is still filled, and pad rows after it
    are absent (pad vs native gap is distinguished).
  - With the demo mask on, every natively observed value in the download is
    byte-equal to the input.
  - Timestamps/entity round-trip when present.
  - Pad cells never appear in the long export.
  - Upload longer than `max_windows × seq_len` → the truncation warning is
    shown and the file is marked partial (or, with batching, all rows are
    present).
- **Done when:** the download for any input has the same rows as the input
  (or is clearly marked partial under the window cap) and differs only in
  previously missing cells.
- **Done (2026-11-21):** `WindowPadInfo` provenance; download merge on
  `m_native`; pads dropped; wide `*_imputed` + long `row_index`/`is_imputed`
  (native draws frozen); UI/CLI partial filenames + knob-named warnings;
  Fill stores windows for export. Decisions: truncate+warn (no batching);
  shared `mean_fill_download_filename(stem=…)`; Flexible `row_index` =
  post-bin grid. Commands: `pytest tests/test_wp_b3_honest_export.py -q`
  → **24 passed**; `pytest -q` → **181 passed** (~91s).
- **Coordination:** WP8's optional overlapping-window stitching changes how
  windows map back to rows. Whichever of WP-B3 / WP8-stitching lands second
  must rerun these tests.

#### WP-B4 — Adapt-path guarantees and single imputer factory (B10, B12)
- **Abort-without-promote test:** a hermetic test that forces (a) a
  non-finite loss (stub loss returns `NaN`) and (b) a timeout
  (`timeout_seconds` tiny, or an injected clock), then asserts no
  `best_score_net.eqx` / `feature_scaler.npz` exists under the run dir and
  `adapt_meta.json` (if written) says `status != success`.
- **One imputer factory:** move the duplicated `SessionAssetRequest` →
  imputer wiring into one function (e.g. `app/session_assets.py::
  build_session_imputer(request, sample_cfg)`) and call it from both the CLI
  and Streamlit. It must read the M7.5 flags (`parameterization`,
  `sample.t_eps`, `observed_mode`, `diffusion_impl`) from the one config, so
  Track A changes reach both paths at once.
- **Tests:** a parity test that builds the imputer via the CLI and the UI
  entry points from the same config and asserts identical settings
  (`ImputeSolverSettings.to_dict()`, SDE params, model config).
- **Done when:** both tests are in CI; the duplicate code is deleted.

#### WP-B5 — Ingest heuristics and cleanup (B13–B16)
- **Flexible ingest:** widen missing tokens (`n/a`, `N/A`, `-`, `--`, `?`,
  `missing`, `null`, case-insensitive, trimmed); exclude calendar/index-like
  names (`year`, `month`, `day`, `hour`, `minute`, `index`, `no`, `id`,
  `unnamed: 0`) from feature inference unless the user picks them; accept a
  timestamp column only when most values parse as plausible datetimes (a
  monotone integer step column is not a timestamp); list every dropped or
  excluded column, with the reason, in the preview.
- **Dead knobs:** remove `adapt.base_scaler` and `confirm_same_semantics`
  (config + dataclass + docs) or wire them; default is remove.
- **Flatten captions:** replace the nested caption types with a few plain
  caption strings; drop the attribute-count test.
- **Dependencies:** move `pytest` to `[project.optional-dependencies] dev`;
  CI installs `.[dev]`; README install lines updated.
- **Done when:** the preview explains every drop, and a plain
  `pip install -e .` doesn't install pytest.

### 9.3 Suggested order (Track B)

| Order | WP | Size | Notes |
|---|---|---|---|
| 1 | WP-B1 | S | Smallest change with the largest "wrong output" risk removed (incl. stale results on the pretrained path) |
| 2 | WP-B3 | M | Start with the `AdaptedWindows` provenance fields; coordinate with WP8 stitching |
| 3 | WP-B2 | M | |
| 4 | WP-B4 | S | Best done after Track A WP0–WP2, so the factory reads the new flags |
| 5 | WP-B5 | S | Any time |

### 9.4 Cross-track dependencies

- **Track A WP0 / WP5 → WP7:** split pinning (A5) and the scaler/metric fixes
  (A8, A9, A11) must merge before the re-benchmark.
- **Track A WP0–WP2 → WP-B4:** the shared imputer factory should read the
  new flags; if WP-B4 lands first, add the flags to the factory when WP2 lands.
- **WP-B3 ↔ WP8 (stitching):** whichever lands second reruns the export tests.
- **WP-B2 ↔ WP5:** WP5's train-slice feature floor (A11) and WP-B2's
  ingest rejects both produce user-facing errors; keep the wording consistent
  and list both in the ASSUMPTIONS Milestone 7.5 section.

### 9.5 Exit criteria (Track B)

1. Every repro for B1, B2, B3, B4a, B4b, B17 and B18 has a test that fails
   on the pre-M7.5 code and passes after.
2. Fill can only run against a bind whose content fingerprint matches the
   current upload; a failed adapt leaves no bind; charts and downloads are
   shown only for results whose fingerprint matches the current upload.
3. The best-estimate download has exactly the input's rows (or is visibly
   marked partial under the window cap), keeps every natively observed value
   unchanged, flags imputed cells, and carries timestamp/entity when present.
4. Multi-entity, duplicate-hour, and non-finite inputs are rejected or warned
   with row/column/entity context; none are silently collapsed.
5. CLI and Streamlit build the imputer through one function (parity test in CI).
6. ASSUMPTIONS Milestone 7.5 section documents the new ingest rules and export
   format; README / UI help text matches.
