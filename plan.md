# Project Plan: Continuous-Time Score-SDE Imputation for Irregular Time Series

## Objective
Build a continuous-time, score-based diffusion model (SDE-based) that performs
**probabilistic imputation** on irregular / missing multivariate time series, and
optionally exports synthetic trajectories.

**Product north star (post-v1):** a **universal** data-science workflow —
upload an arbitrary messy multivariate CSV → **train / adapt a model on that
data** → fill gaps with dependence-aware uncertainty bands → download a
completed dataset. Optionally **route by gap regime** (small / medium / large
contiguous gaps): light policies for short holes; heavier score-SDE (or
specialist) imputers for long blocks.

**v1 scope (shipped DoD):** prove the SDE stack on one real path (Beijing
air-quality, `F=6`), pretrained demo + LOCF/linear evidence. Train-on-upload,
schema-agnostic CSV, and gap-size routing are **not** v1 exit criteria.

**Tech stack:** Python, JAX, Equinox, Diffrax, Optax, Streamlit, Matplotlib / Plotly.

**Core claim:** Unlike mean / LOCF / simple interpolators, a conditional reverse
SDE can exploit temporal and cross-feature structure and return a predictive
distribution over missing values—not only point estimates.

**Non-goals (v1):** Differentially private training, production-scale distributed
training, beating every SOTA paper on every dataset, causal treatment-effect
estimation, universal train-on-upload as a shipped product claim.

**Positioning:** CSDI-style conditional imputation objective, implemented and
extended as an explicit continuous-time score-SDE system with irregular-time
emphasis and a practical uncertainty UX — with CSDI treated as prior art and
an M3 baseline, not as unrelated work.

---

## Problem Framing (Why this project)
- Real datasets (ICU sensors, air quality, energy, product telemetry) are often
  irregularly sampled and partially observed.
- Point imputation hides uncertainty and can break autocorrelation / cross-correlation.
- Industry and research both treat this as a first-class problem (e.g. PhysioNet
  ICU data; CSDI-style score-based imputation; Neural SDEs for irregular series).

**Primary user:** Data scientist / ML engineer with a multivariate time-series CSV
containing timestamps, missing entries, and uneven sampling.

---

## Phase 0: Problem Spec, Data, and Evaluation Contract
- **Goal:** Lock the scientific claim and metrics before modeling.
- **Tasks:**
  1. Define data schema: `timestamp`, entity/id (optional), feature columns,
     explicit missingness mask.
  2. Select datasets:
     - **Healthcare / irregular:** PhysioNet ICU-style series (primary story).
     - **General DS:** one of air quality, electricity, or traffic (secondary).
  3. Define missingness protocols for experiments:
     - Random missing (MCAR-style) at rates e.g. 10% / 30% / 50%.
     - Block missing (contiguous gaps).
     - Timestamp irregularity (uneven Δt), if available in the raw data.
  4. Freeze metrics:
     - Point: MAE / RMSE on held-out observed entries.
     - Probabilistic: CRPS (and/or NLL) on imputed values.
     - Calibration: empirical coverage of 50% / 90% predictive intervals.
     - Structure (lightweight): autocorrelation / cross-correlation error vs ground truth
       on completed series.
  5. Freeze baselines for v1:
     - Last Observation Carried Forward (LOCF)
     - Linear interpolation (per feature)
     - One stronger baseline if time permits (e.g. BRITS, simple Transformer imputer,
       or a reimplementation-light CSDI-style discrete diffusion)

**Exit criteria:** A short `EVAL.md` (or README section) stating datasets, splits,
masks, metrics, and baselines.

---

## Phase 1: Forward Process, Masking, and Score Network
- **Goal:** Implement the continuous-time noising process and a conditional score
  network that sees observations + masks + time.
- **Tasks:**
  1. Implement a Variance Preserving SDE (VP-SDE) forward process on the series
     representation (start with fixed-length windows; document the assumption).
  2. Define the conditioning interface:
     - Noisy state `y_t`
     - Diffusion time scalar `t`
     - Observation mask `m`
     - Observed values (conditioned channels / values)
     - Optional wall-clock / irregular time encoding for timestamps
  3. Implement an Equinox score network with **temporal inductive bias**
     (prefer TCN / temporal Transformer / U-Net-along-time over a flat MLP).
  4. Self-supervised training view: randomly mask observed entries and train the
     model to denoise / impute them (CSDI-style masking), adapted to continuous time.
  5. Visualize:
     - Forward noising of a clean window into Gaussian noise.
     - Mask patterns (random vs block).

**Exit criteria:** Forward process + conditional score net run on a toy batch;
plots saved; shapes/masks unit-tested.

---

## Phase 2: Training (Conditional Denoising Score Matching)
- **Goal:** Train the score network with an efficient JAX loop.
- **Tasks:**
  1. Implement continuous-time denoising score matching loss conditioned on
     observed context (loss computed on masked / target positions as appropriate).
  2. Training loop with `jax.grad`, `jax.jit`, Optax (AdamW or similar).
  3. Logging: train/val loss curves; periodic imputation snapshots on a fixed val batch.
  4. Basic robustness checks: NaNs, exploding norms, sensitivity to mask rate.
  5. Checkpointing (best val CRPS or val loss).

**Exit criteria:** Stable loss curves on at least one real dataset; checkpoint that
can impute a validation batch.

---

## Phase 3: Reverse SDE Sampling for Imputation
- **Goal:** Generate posterior samples of missing values given observations.
- **Tasks:**
  1. Formulate the conditional reverse-time SDE (Anderson reverse + conditioning
     via the trained score / replacement of observed coordinates as needed).
  2. Implement sampling with `diffrax.diffeqsolve` (Euler–Maruyama or Heun).
  3. Draw `K` imputation samples per series (e.g. K = 20–100) to form:
     - Posterior mean imputation
     - Quantile bands (e.g. 10–90%)
  4. Optional v1.1: unconditional / partially conditional **synthetic trajectory**
     export using the same backbone (secondary feature, not the main claim).

**Exit criteria:** End-to-end impute function:
`observations, mask, timestamps -> samples, mean, quantiles`.

---

## Phase 4: Benchmarking Against Baselines (The Evidence Layer)
- **Goal:** Prove usefulness the way a data scientist would evaluate it.
- **Tasks:**
  1. Run the frozen missingness protocols from Phase 0.
  2. Compare against LOCF, linear interpolation, and the stronger baseline.
  3. Report tables for MAE/RMSE, CRPS, and interval coverage by missing rate.
  4. Qualitative panels: real vs imputed mean ± bands on held-out gaps.
  5. Failure analysis: long block gaps, sparse features, distribution shift between
     datasets.
  6. (Optional) Lightweight downstream check: train a simple forecaster on imputed
     data vs on ground truth / baseline imputations (utility test).

**Exit criteria:** A results figure + table you could put in a README or portfolio
post; honest limitations section.

---

## Phase 5: Streamlit Application (Product Layer)
- **Goal:** Make the method usable without reading the training code.
- **Tasks:**
  1. Sidebar controls: mask rate (for demo corruption), number of reverse samples `K`,
     learning rate / epochs (if on-the-fly fine-tune is supported), solver step settings.
  2. CSV uploader with schema validation and missingness summary.
  3. Interactive charts (Plotly or Streamlit charts):
     - Original series with gaps
     - Imputed mean + uncertainty bands
     - Optional overlay of multiple posterior samples
  4. Download buttons:
     - Imputed CSV (mean fill)
     - Optional multi-sample export (long format)
  5. “Run evaluation on demo dataset” button that reproduces a small benchmark slice.

**Exit criteria:** A demo path that works on a bundled sample CSV in under a few
minutes (pretrained weights) or with a short fine-tune on a small subset.

---

## Suggested Repository Layout
```text
plan.md
README.md
src/
  data/          # loaders, masking, splits
  sde/           # VP-SDE forward / reverse helpers
  models/        # Equinox score network
  train/         # loss, loop, checkpoints
  sample/        # Diffrax reverse sampling / imputation
  eval/          # MAE, CRPS, coverage, plots
app/
  streamlit_app.py
configs/
  default.yaml
tests/
notebooks/       # optional exploration only
```

---

## Milestones and Definition of Done

### v1 (shipped)
1. **M1 — Toy path:** forward SDE + conditional score net + reverse samples on synthetic sine/AR data with masks.
2. **M2 — Real data train:** stable training on PhysioNet-style or air-quality windows.
3. **M3 — Benchmark:** beat LOCF / linear interpolation on CRPS and coverage; competitive MAE.
4. **M4 — Product:** Streamlit upload → impute → download.
5. **M5 — Portfolio polish:** README with problem statement, citations, metrics, limitations.

**Definition of done (v1):**
- Reproducible training + imputation scripts
- Reported metrics vs at least two baselines
- Working Streamlit demo on a sample dataset
- Clear statement of assumptions (windowing, missingness types, compute limits)

### Post-v1 (scheduled)
Roadmap toward the **product north star** (universal CSV → train/adapt →
bands → download; optional gap-regime routing). Each milestone has its own
DoD; v1 Beijing pretrained claims stay frozen unless a milestone explicitly
extends the eval contract.

6. **M6 — Schema-flexible ingest** (done)
7. **M7 — Train-on-upload / adapt** (done)
8. **M7.5 — Correctness fixes**, in two tracks — **must land before M8**:
   - **A, model:** train/sample consistency, loss weighting, score net v2,
     EMA + selection by imputation, transforms, split pinning, multi-seed
     re-benchmark
   - **B, app/data:** session binding, ingest rejects, honest export, a single
     imputer factory
9. **M8 — Gap-regime routing** (thresholds from M7.5 evidence)
10. **M9 — Evidence expansion** (pick one primary track; others optional)

Detail, tasks, and exit criteria: **Post-v1 milestones (M6–M9, incl. M7.5)** below.
Stretch items (irregular-time Neural SDE, constrained sampling, privacy)
remain unscheduled until promoted.

---

## Post-v1 milestones (M6–M9, incl. M7.5)

Working plan for the next slices. README **Future work** stays a short
product-facing summary and points here for DoD.

### M6 — Schema-flexible ingest

- **Goal:** Accept messy multivariate CSVs beyond the frozen Beijing `F=6`
  column names, without training or claiming a universal imputer yet.
- **Tasks:**
  1. Infer or map feature columns, optional entity/id, and timestamps.
  2. Build native missingness masks; pad / truncate / window with documented
     rules (regular grid after binning remains the v1-compatible path).
  3. Fail clearly when the series is too short, too sparse, or schema-invalid
     (no silent remaps of unknown columns to Beijing names).
  4. Unit tests + a small fixture CSV that is *not* the Beijing schema.
  5. Wire Streamlit (or a CLI) to preview schema / missingness summary before
     impute; pretrained Beijing path remains available as the exact-schema
     fallback.
- **Exit criteria:** Upload (or load) a non-Beijing multivariate CSV → validated
  windows + masks → clear error messages on reject; docs note that imputation
  still requires a compatible checkpoint or M7 train-on-upload.
- **Walkthrough:** `docs/milestone_6_walkthrough.md` when work starts.

### M7 — Train-on-upload / adapt

- **Goal:** On a user CSV (via M6 ingest), **train or fine-tune** a score model
  on that file, then impute and download mean ± bands.
- **Tasks:**
  1. Short adaptation loop (fine-tune from a base checkpoint and/or train-from-
     scratch on small data) with smoke caps suitable for Streamlit / CPU.
  2. Fit scaler and masks **train-only** on the uploaded series (or its
     declared train split); no leakage from held-out artificial gaps used for
     a quick quality check.
  3. UI / CLI: run adapt → impute → download; surface compute limits and
     failure modes (too little data, NaN loss, timeout).
  4. Keep the v1 pretrained Beijing demo path intact; do not overwrite M2
     claim artifacts as the sole “universal” model.
  5. Minimal eval slice: smoke metrics on artificial gaps for the uploaded
     file (not a new cross-dataset SOTA claim).
- **Exit criteria:** End-to-end path on at least one non-Beijing fixture CSV:
  ingest → adapt → `K` posterior samples → downloadable mean-fill (+ optional
  multi-sample export). Documented assumptions and compute caps.
- **Walkthrough:** `docs/milestone_7_walkthrough.md` when work starts.
- **Depends on:** M6.

### M7.5 — Correctness fixes (pre-M8)

- **Goal:** Fix problems found in two code reviews (Nov 2026) before M8 sets
  routing thresholds from benchmark numbers and adds routing on top of the
  upload path. **Track A** covers model training/sampling and the evaluation
  plumbing behind the re-benchmark; **Track B** covers the app/data path
  (upload → ingest → adapt → fill → download). This is a correctness /
  quality milestone: no new product surface, datasets, or baselines. Full
  spec: **`docs/milestone_7_5_plan.md`**.
- **Why now:** M3 shows TimeFill behind linear on MAE/RMSE at MCAR 10% / 30%.
  Part of that gap is likely caused by fixable issues, and M8 would otherwise
  route around bugs instead of real method limits.
- **Track A tasks (work packages):**
  0. **WP0 Contract:** new config keys with **legacy defaults** (old configs
     reproduce old behavior); `configs/real_m7_5.yaml`,
     `configs/bench_m7_5.yaml`; `model_config.json` checkpoint sidecar;
     **split manifest** next to the checkpoint, and the benchmark refuses a
     re-prepared / mismatched split; ASSUMPTIONS Milestone 7.5 section.
  1. **WP1 Conditioning consistency:** training currently noises observed
     cells, while the sampler keeps them clean. Default fix: train with clean
     observed cells in `y_t` (`observed_input: clean`), matching the frozen
     sampler. Ablation alternative: noised training + per-step
     replacement sampling (`observed_mode: replace_noised`).
  2. **WP2 Loss parameterization:** ε-prediction (`‖ε̂−ε‖²`, score
     `= −ε̂/σ(t)`), training `t_eps = 1e-3` (separate from the sampler
     endpoint), `obs_loss_weight = 0`.
  3. **WP3 Score net v2:** residual blocks with GroupNorm and per-block time
     conditioning (FiLM), dilations up to 32 (receptive field ≥ T),
     hidden 64, time embedding scaled (`1000·t`); v1 kept for old checkpoints.
  4. **WP4 Training hygiene:** EMA weights; checkpoint selection by fixed-set
     val **imputation** CRPS/MAE (not DSM loss); per-window mixed masks;
     variable block lengths / multiple blocks; batch 32; warmup + cosine LR.
  5. **WP5 Transforms & eval:** optional per-feature `log1p` before z-score,
     nonnegative clip on restore; per-feature and normalized-unit metrics
     next to pooled original-unit metrics; error (not an identity scaler)
     for features with no train observations; metrics are NaN when there are
     no target cells; rewrite the scaler leakage tests so they can fail.
  6. **WP6 Sampler:** diagonal diffusion (no dense `(T·F)²` matrix); median
     point estimate for MAE (mean kept for RMSE).
  7. **WP7 Re-benchmark:** M3 protocols on test × ≥3 mask seeds (mean ± std),
     block-16 extension row, ablation table, `docs/m7_5_results.md`, and an
     **M8 hand-off table** (gap regime → best method).
  8. **WP8 Housekeeping:** remove `src/data/Untitled-1.py`; mark
     under-trained M7 adapt runs `smoke_only` with a UI warning; optional
     overlapping-window stitching in the app.
- **Track B tasks (work packages):**
  - **WP-B1 Session binding:** content-hash fingerprint for adapt/impute
    binds; clear the bind on upload change and on failed adapt; bind Fill
    results to the upload too, so stale charts and downloads never show
    under a new file (including the pretrained path).
  - **WP-B2 Ingest correctness:** reject multi-entity files and excess
    duplicate hours; sort/bin the Exact path by time (or drop the hourly
    claim); reject non-finite values.
  - **WP-B3 Honest export:** first make windows carry real length, source
    row and timestamps; then the download keeps the input's rows (no pad
    rows), never overwrites native observations (demo mask only for the
    chart), flags imputed cells, carries timestamp/entity, and is visibly
    marked partial when the window cap truncates a long upload.
  - **WP-B4 Adapt guarantees:** hermetic NaN/timeout no-promote test; one
    imputer factory shared by CLI and Streamlit (parity test).
  - **WP-B5 Cleanup:** wider NA tokens, exclude calendar/index columns, drop
    dead adapt knobs, flatten captions, move `pytest` to `dev` extras.
- **Ground rules:** write only under `artifacts/m7_5/`; v1/M3/M7 artifacts and
  `docs/EVAL.md` v1 tables stay byte-identical; one WP per PR with tests.
- **Exit criteria:** All Track A WPs merged behind flags with `pytest` green;
  the benchmark runs only against a split-pinned checkpoint; one full
  M7.5 training run with `train_meta.json` (clipped-step fraction, grad norm,
  selected epoch, EMA, params, receptive field); multi-seed M7.5 benchmark +
  ablation; honest claim checklist and M8 hand-off table. *Targets (reported
  either way, not gates):* MAE ≤ linear on MCAR 30% and block 8; CRPS better
  than M3 TimeFill everywhere; coverage_90 in 0.85–0.95. **Track B:** each
  must-fix audit repro has a test that fails before and passes after; Fill
  only runs against a content-matched bind, and charts/downloads only show
  for the current upload; the download keeps the input's rows and native
  observations (or is marked partial under the window cap).
- **Out of scope:** wall-clock / irregular-Δt modelling (stays a documented
  limitation), CSDI/BRITS baselines (M9), routing (M8), multi-user / hosted
  security hardening (load model stays one person on a laptop).
- **Walkthrough:** `docs/milestone_7_5_walkthrough.md` when work starts.
- **Depends on:** M2, M3, M7 (shared adapt path picks up the new flags opt-in).

### M8 — Gap-regime routing

- **Goal:** Detect contiguous gap length and choose a policy — light / point
  methods for short holes; full TimeFill (or a specialist) for medium / large
  blocks — matching the evidence (M3, re-measured in M7.5) that linear is
  strong on easy short MCAR while TimeFill wins on blocks and probabilistic
  scores. If M7.5 closes the short-gap MAE gap, routing may reduce to a
  latency / cost policy; document whichever is true.
- **Tasks:**
  1. Define small / medium / large thresholds (config-driven; document defaults).
  2. Implement a router that assigns each gap (or window) a backend policy.
  3. Baselines for short gaps (e.g. linear / LOCF); TimeFill for medium/large;
     optional long-gap specialist or block-gap curriculum if time permits.
  4. Benchmark vs single-policy TimeFill and vs LOCF/linear on the frozen M3
     protocols (or a documented extension); report wins/losses honestly.
  5. Optional Streamlit toggle: “auto route” vs “always TimeFill.”
- **Exit criteria:** Documented routing contract + table or notes showing when
  routing helps or hurts; runnable script or app path that applies the router.
- **Walkthrough:** `docs/milestone_8_walkthrough.md` when work starts.
- **Depends on:** **M7.5** (thresholds and "who wins where" come from the M7.5
  multi-seed hand-off table, not the single-seed M3 table); M3 protocol
  contract; product wiring can use M4/M7 paths.

### M9 — Evidence expansion

- **Goal:** Strengthen credibility or demo friction **without** expanding the
  v1 Beijing claim by accident. Choose **one primary** track for the milestone
  DoD; treat the others as optional follow-ons in the same milestone only if
  capacity allows.
- **Primary tracks (pick one):**
  1. **Hosted checkpoint:** downloadable M2 weights + scaler so clone → demo
     does not require a local full train (still Beijing-scoped unless a
     generalist from M7 ships).
  2. **Second domain:** PhysioNet-style or other multivariate series with its
     own prepare/train/eval notes; claims kept separate from Beijing v1.
  3. **Stronger baseline:** CSDI / BRITS / Transformer (or similar) as a
     measuring stick or optional backend — comparison only; no win claimed
     until measured.
- **Tasks (per chosen track):** freeze protocol updates in `docs/EVAL.md` /
  ASSUMPTIONS as needed; scripts + artifacts path; README honesty (scope,
  non-claims); tests or smoke where applicable.
- **Exit criteria:** Primary track shipped with docs and a reproduce path;
  README / failure notes updated so v1 Beijing numbers remain clearly scoped.
- **Walkthrough:** `docs/milestone_9_walkthrough.md` when work starts.
- **Suggested order after M6–M8:** hosted checkpoint if demo UX is the
  bottleneck; second domain or stronger baseline if portfolio evidence is.

---

## Key References (anchor the problem and method)
1. Tashiro et al., *CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation*, NeurIPS 2021.
2. Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*, ICLR 2021.
3. Oh et al., *Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data*, ICLR 2024.
4. Silva et al., PhysioNet ICU multiparameter time-series resources (irregular clinical data).
5. Kidger et al., Neural CDEs / Diffrax ecosystem (continuous-time numerics in JAX).

---

## Risks and Mitigations
| Risk | Mitigation |
|------|------------|
| Flat MLP ignores temporal structure | Use temporal score architecture from Phase 1 |
| Irregular time is hard | Start with windowed regular grids + masks; add time encodings next (still deferred after M7.5: timestamps are validated but unused by the model) |
| Sampling too slow for Streamlit | Pretrained checkpoints; fewer steps / smaller `K` in the UI |
| Overclaiming vs CSDI | Position as continuous-time / irregular-time JAX implementation with uncertainty UX, not “first ever imputer” |
| Metric gaming | Pre-register protocols in Phase 0; report failures |
| Project collapses into “CSDI in JAX” | Keep SDE+Diffrax primary; ship irregular-time or label it deferred; benchmark vs CSDI in M3 |
| Universal path overclaims v1 Beijing results | Keep M6–M9 claims scoped; do not silently remap schemas or merge eval tables |
| Train-on-upload too slow / unstable on CPU | Smoke caps, fine-tune-from-base, clear timeouts; hosted checkpoint (M9) for pretrained demo |
| Train/sample conditioning mismatch or high-variance DSM loss degrades imputations | M7.5 WP1–WP2: consistent observed-cell handling, ε-prediction, `t_eps=1e-3` |
| Noisy checkpoint selection picks a lucky epoch | M7.5 WP4: EMA + selection by fixed-set val imputation metrics |
| Single-seed benchmark overstates small MAE gaps | M7.5 WP7: ≥3 mask seeds, mean ± std |
| App returns fills for the wrong upload, or the download mixes invented / overwritten cells | M7.5 Track B: content-hash binds, clear on failure, native-preserving export without pad rows |
| Re-preparing data silently leaks train windows into "test" | M7.5 WP0: split manifest next to the checkpoint; benchmark refuses a mismatch |
| M7.5 refactor silently changes v1 numbers | Legacy-default flags, checkpoint sidecar + fallback, frozen-artifact rule |

---

## Stretch Goals (unscheduled; promote before implementing)
- True irregular-time latent Neural SDE encoder–decoder
- Block-gap curriculum and longer horizons (partially overlapping M8; full
  curriculum stays stretch unless pulled into M8 DoD)
- Constrained imputation (business / physical bounds) during reverse sampling
- Simple privacy risk discussion (memorization checks) if synthetic export is
  emphasized
- Synthetic trajectory export as a first-class product feature (optional v1.1
  idea from Phase 3; still secondary to imputation)

## Relation to Prior Art (CSDI)

This project is **not** a from-scratch invention of diffusion imputation.
It is intentionally positioned relative to CSDI (Tashiro et al., NeurIPS 2021).

**Shared with CSDI**
- Probabilistic multivariate time-series imputation
- Self-supervised masking of observed values during training
- Conditioning the generative process on observations + masks
- Multi-sample inference for means and uncertainty bands

**Deliberate differences (project wedge)**
1. **Continuous-time formulation:** VP-SDE forward process and reverse-time SDE
   (Song et al.), not a discrete DDPM-style denoising chain as the primary object.
2. **Numerics:** sampling via Diffrax SDE solvers (e.g. Euler–Maruyama / Heun),
   with an explicit continuous-time score network \(s_\theta\).
3. **Irregular time (emphasized goal):** treat timestamps / uneven \(\Delta t\) as
   first-class inputs (time encodings; avoid silently assuming a dense regular grid
   without documenting it).
4. **Product layer:** Streamlit workflow for data scientists (CSV → calibrated
   bands → download), with a frozen evaluation contract (`docs/EVAL.md`).
5. **Baselines:** LOCF and linear interpolation required; **CSDI (or a faithful
   discrete conditional diffusion baseline) strongly preferred in M3** so gains
   are measured, not assumed.

**Non-claims**
- We do not claim to be the first conditional score-based imputer.
- A JAX port of CSDI alone is **out of scope** as the project definition of done.
- If irregular-time support is deferred, the README/plan must say so and must not
  advertise it as a completed differentiator.
