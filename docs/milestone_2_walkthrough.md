# Milestone 2 Walkthrough — Real Data Train

## What M2 is

From `plan.md`:

> **M2 — Real data train:** stable training on PhysioNet-style or air-quality windows.

M2 takes the M1 loop (VP-SDE → conditional score net → DSM train → Diffrax reverse impute) off synthetic sine/AR and onto **at least one real multivariate time-series dataset**. The product of M2 is a reproducible real-data training path with stable loss curves and a checkpoint that can impute a validation batch.

**Success look:**  
Load / window real CSV → apply M1 mask protocols (plus native missingness) → train → reverse-sample on a fixed val batch → save loss curves + imputation snapshots under `artifacts/m2/`.

**Explicitly deferred:** LOCF / linear / CSDI baseline tables (**M3**), Streamlit (**M4**), portfolio polish (**M5**).

---

## Scope map (plan → M2)

| Plan phase | In M2? | What you actually build |
|---|---|---|
| Phase 0 | Yes (partial) | Real schema, one primary dataset, splits, missingness protocols; start `docs/EVAL.md` (datasets / splits / masks / metrics; baselines listed but not fully run) |
| Phase 1 | Reuse | Keep VP-SDE + conditional score net; adapt input `F`, normalization, optional time encoding stubs |
| Phase 2 | Yes (real) | Stable DSM training on real windows; logging, NaN/grad guards, checkpoints |
| Phase 3 | Yes (reuse) | Same `impute(...)` API on real val batches; snapshot plots |
| Phase 4 | No (prep only) | Freeze protocols/metrics for M3; do **not** require full baseline bake-off |
| Phase 5 | No | Streamlit stays out |

---

## Assumptions: carry forward vs relax

`ASSUMPTIONS.md` currently locks **M1**. For M2, treat it as the base contract and **extend** (do not silently rewrite M1 behavior).

### Carry forward (unless you document a deliberate change)

1. **Fixed-length windows** on a **regular grid** after resampling / binning (`T = seq_len`, default still e.g. 64 unless config says otherwise).
2. **Batch layout** `(B, T, F)` for values and masks.
3. **Mask convention:** `1 = observed`, `0 = missing`; float32 masks.
4. **Missing sentinel:** missing cells in `x` are `0.0`; never infer missingness from “nonzero in `x`”.
5. **Loss focus:** DSM primarily on target cells (`m == 0`); `train.obs_loss_weight` default `0.0` unless you deliberately enable auxiliary observed loss and record it.
6. **VP-SDE + Diffrax reverse imputer** from M1 (replacement conditioning, `K` samples, quantile bands).
7. **Irregular wall-clock as first-class dynamics** remains deferred unless you explicitly promote it in M2; if still deferred, say so in `ASSUMPTIONS.md` / README (plan: do not advertise irregular-time as done).

### New / relaxed for M2 (document in `ASSUMPTIONS.md` under an “M2” section)

1. **Dataset:** primary = PhysioNet ICU-style **or** air quality (plan secondary: electricity / traffic). Pick one primary for the exit criterion; optional second later.
2. **Feature count `F`:** larger than M1’s `{1,2,3}` — freeze a concrete channel list (e.g. subset of vital signs / pollutants) and document it.
3. **Native missingness:** real series already have gaps; training masks are applied on top of (or composed with) native `m_native`. Document composition rule, e.g.  
   `m_train = m_native * m_artificial` (only originally observed cells can be held out as targets).
4. **Normalization:** per-feature standardization (or robust scaling) fit on **train windows only**; store stats with the checkpoint.
5. **Splits:** entity- or time-aware train/val(/test) so the same window does not leak across splits.
6. **Resampling:** irregular timestamps → regular grid via binning / last-value-in-bin / NaN if empty — document the rule; this is the plan’s “windowed regular grids + masks first” mitigation.
7. **Artifacts root:** `artifacts/m2/` (checkpoints, loss curves, val imputation panels).

---

## Suggested repo layout additions for M2

Keep M1 code; add real-data surface:

```text
ASSUMPTIONS.md                     # add Milestone 2 section (do not erase M1)
docs/
  EVAL.md                          # Phase 0 contract start (datasets, splits, masks, metrics, baseline list)
  milestone_2_walkthrough.md       # this file
configs/
  toy_m1.yaml                      # unchanged M1
  real_m2.yaml                     # dataset path, F, T, mask rates, SDE, train, sample knobs
src/
  data/
    schema.py                      # reuse conventions
    masking.py                     # reuse random + block
    synthetic.py                   # keep for regression / smoke
    real.py                        # NEW: download/load helpers, windowing, native masks
    normalize.py                   # NEW: fit/transform feature scalers
  sde/ models/ train/ sample/ eval/  # reuse M1
scripts/
  prepare_m2_data.py               # optional: raw → windowed arrays + split indices
  train_m2_real.py                 # train on real config
  run_m2_impute_val.py             # load ckpt → impute fixed val batch → plots
tests/
  test_real_loader.py              # shapes, native mask, no leakage in splits
  test_normalize_fit_train_only.py
artifacts/
  m2/
    checkpoints/
    loss_curves.csv / .png
    val_snapshots/
data/                              # gitignore large raw dumps; document how to fetch
  raw/                             # optional local cache
  processed/                       # windowed npz/zarr if you materialize
```

---

## Work packages (do in this order)

### WP0 — Evaluation contract + M2 assumptions (½–1 day)

Align with Phase 0 exit criteria (partial):

- Choose **primary dataset** (PhysioNet-style vs air quality) and write why.
- Freeze in `docs/EVAL.md`:
  - schema: `timestamp`, optional entity/id, feature columns, mask
  - splits
  - missingness protocols: MCAR 10/30/50%, block gaps, note native irregularity / missingness
  - metrics: MAE / RMSE, CRPS, 50%/90% coverage (structure metrics optional for M2)
  - baselines **listed** for M3: LOCF, linear, stronger baseline if time permits
- Extend `ASSUMPTIONS.md` with an **M2** section (windowing after resample, `F`, normalization, mask composition).

**Done when:** `docs/EVAL.md` + M2 assumptions exist; config skeleton `configs/real_m2.yaml` lists the frozen knobs.

---

### WP1 — Real data loader + windowing (Phase 0 / data)

**`src/data/real/` (package; split modules)**

- Load CSV / PhysioNet-style tables into a typed schema.
- Build regular-grid windows of length `T` with values `(N, T, F)` and native masks `(N, T, F)`.
- Entity/id handling if present (ICU stay, station id, etc.).
- Deterministic train/val split API.

**Mask composition**

- Reuse `random_mask` / `block_mask` from M1 on cells that are already observed.
- Targets for loss/metrics: artificially masked cells (and optionally evaluate only where ground truth existed).

**Tests**

- Output shapes `(B, T, F)`.
- Native missing cells are sentinel + `m==0`.
- Artificial masks never “observe” a natively missing cell.
- Split indices disjoint.

**Done when:** a script can emit one train batch and one frozen val batch from real data.

---

### WP2 — Normalization + config wiring

- Fit mean/std (or robust stats) on train observed cells only.
- Apply transform before SDE/score net; inverse-transform for plots/metrics in original units (document which space metrics use — prefer original units for reported MAE/RMSE).
- Wire `configs/real_m2.yaml`: paths, `seq_len`, feature list, mask rates, SDE betas, `lr`, `epochs`, `K_samples`, `solver_steps`, checkpoint dir under `artifacts/m2/`.

**Done when:** train/val tensors are finite; scaler stats saved next to checkpoints.

---

### WP3 — Train on real windows (Phase 2)

Reuse M1 training stack (`src/train/loss.py`, `loop.py`) with real dataloading:

1. Sample window batch → compose masks → DSM on target cells.
2. `jax.jit` + Optax AdamW; log train/val loss.
3. Periodic val imputation snapshots (mean ± bands) into `artifacts/m2/val_snapshots/`.
4. Robustness: NaN checks, grad-norm logging, optional sensitivity smoke at 10%/30%/50% mask rate.
5. Checkpoint on best val loss (or val CRPS if cheap enough to compute periodically).

**Done when:** loss curves are stable (no NaN divergence) on the primary real dataset for a full intended run; best checkpoint written.

---

### WP4 — Val-batch imputation + lightweight metrics (Phase 3 reuse)

- Load best checkpoint + scaler + fixed val batch.
- Call existing reverse SDE imputer:  
  `observations, mask, timestamps → samples, mean, quantiles`.
- Save qualitative panel(s): real vs imputed mean ± bands on held-out gaps.
- Print MAE / RMSE / CRPS / coverage **on artificially masked cells with known ground truth** (same metric code as M1 WP6).

Not required: full LOCF/linear tables (M3).

**Done when:** `scripts/run_m2_impute_val.py` (or equivalent) reproduces plots + a tiny metrics dump from the checkpoint alone.

---

### WP5 — Docs / regression hygiene

- README: how to fetch data, prepare windows, train, impute.
- Keep M1 `scripts/run_m1_toy.py` + tests green (real-data work must not break the toy path).
- Note compute limits (CPU vs GPU, expected wall time, recommended `K` / solver steps).

**Done when:** a clean clone + documented data fetch can reach “stable train + val impute” without reading source.

---

## Exit criteria checklist (from plan M2 + Phase 2)

- [ ] Primary real dataset chosen and documented (`docs/EVAL.md`)
- [ ] M2 assumptions written (windowing/resample, `F`, normalization, mask composition)
- [ ] Real loader + windowing + native masks, unit-tested
- [ ] Train/val split without leakage; scaler fit on train only
- [ ] Stable DSM loss curves on real windows (`artifacts/m2/loss_curves.*`)
- [ ] Checkpoint that imputes a fixed validation batch
- [ ] Val snapshot: imputation mean ± uncertainty bands
- [ ] Lightweight MAE/RMSE/CRPS (and optional coverage) on held-out real cells
- [ ] M1 toy path still passes
- [ ] Irregular-time status explicitly stated (supported vs deferred)

---

## Suggested implementation order (calendar)

| Day | Focus |
|---|---|
| 1 | WP0: dataset choice, `EVAL.md`, M2 assumptions, `real_m2.yaml` |
| 2 | WP1: loader, windowing, native masks, split tests |
| 3 | WP2: normalization + end-to-end batch smoke into existing score net |
| 4–5 | WP3: full real train, checkpoints, loss curves, robustness logs |
| 6 | WP4–WP5: val impute script, bands plot, metrics dump, README runbook |

---

## Risks specific to M2 (from plan + ASSUMPTIONS)

| Risk | M2 mitigation |
|---|---|
| Irregular time is hard | Resample to regular windows + masks; document; defer true irregular SDE |
| Overclaiming vs CSDI | M2 is “real-data train in continuous-time JAX,” not SOTA claim; baselines in M3 |
| Training unstable on real scale/noise | Normalize; start with fewer features / shorter `T`; grad clip; sine regression smoke |
| Native + artificial missingness bugs | Unit-test mask composition; metrics only where ground truth exists |
| Data / licensing friction | Prefer a clearly licensed public set; script the download; don’t commit huge raw files |
| Silent assumption drift | Extend `ASSUMPTIONS.md` instead of editing M1 contracts in place |

---

## What “done” looks like demowise

```bash
# after data prep per README
python scripts/train_m2_real.py --config configs/real_m2.yaml
python scripts/run_m2_impute_val.py --config configs/real_m2.yaml --ckpt artifacts/m2/checkpoints/best_score_net.eqx
pytest tests/ -q
```

You should get:

1. Stable train/val loss on the real dataset  
2. `artifacts/m2/` loss curves + val imputation panels  
3. A checkpoint that fills held-out real gaps with mean ± bands  

That is the handoff into **M3 — Benchmark** (LOCF / linear / stronger baseline on the frozen `EVAL.md` protocols).

---

## References to keep open while coding

1. Silva et al., PhysioNet ICU multiparameter time-series resources  
2. Tashiro et al., CSDI (conditional masking on real clinical / air-quality style setups)  
3. Song et al., Score-Based Generative Modeling through SDEs  
4. Project `plan.md` Phase 0 + Phase 2 exit criteria; `ASSUMPTIONS.md` M1 contracts to extend
