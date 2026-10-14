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
| Irregular time is hard | Start with windowed regular grids + masks; add time encodings next |
| Sampling too slow for Streamlit | Pretrained checkpoints; fewer steps / smaller `K` in the UI |
| Overclaiming vs CSDI | Position as continuous-time / irregular-time JAX implementation with uncertainty UX, not “first ever imputer” |
| Metric gaming | Pre-register protocols in Phase 0; report failures |
| Project collapses into “CSDI in JAX” | Keep SDE+Diffrax primary; ship irregular-time or label it deferred; benchmark vs CSDI in M3 |

---

## Stretch Goals (only after M4)
- True irregular-time latent Neural SDE encoder–decoder
- Block-gap curriculum and longer horizons
- Constrained imputation (business / physical bounds) during reverse sampling
- Simple privacy risk discussion (memorization checks) if synthetic export is emphasized

## Post-v1 roadmap (not v1 DoD)

Ideas only; see README **Product north star** / **Future work**. Not
milestones unless scheduled later:

- **Train-on-upload:** schema-flexible CSV → train / fine-tune on the user’s
  file → impute → download (beyond frozen Beijing `F=6` pretrained path)
- **Gap-regime routing:** small / medium / large gap policies or specialists
  (plus block-gap curriculum / longer horizons)
- Stronger baselines or optional imputers (CSDI / BRITS / Transformer) for
  comparison — no win claimed until measured
- Hosted M2 checkpoint + scaler for clone → demo without local full train
- Second-domain evidence (e.g. PhysioNet-style), still separate from v1 Beijing
  claim scope

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
