# Milestone 3 Walkthrough — Benchmark

## What M3 is

From `plan.md`:

> **M3 — Benchmark:** beat LOCF / linear interpolation on CRPS and coverage; competitive MAE.

M3 is Phase 4 — the **evidence layer**. It reuses the M2 real-data path (Beijing Multi-Site Air-Quality windows, frozen `docs/EVAL.md` protocols, trained checkpoint + scaler) and proves usefulness the way a data scientist would: **same held-out protocols, same metrics, side-by-side with simple baselines**.

**Success look:**  
Load processed windows + M2 checkpoint → run MCAR **10% / 30% / 50%** (and optional block) on a frozen held-out split → impute with TimeFill, LOCF, and linear → write comparison tables + qualitative panels under `artifacts/m3/`.

**Explicitly deferred:** Streamlit (**M4**), portfolio polish beyond an M3 runbook (**M5**). Irregular wall-clock as first-class SDE dynamics stays deferred (same as M2) unless you explicitly promote it.

---

## Scope map (plan → M3)

| Plan phase | In M3? | What you actually build |
|---|---|---|
| Phase 0 | Reuse / extend | Keep `docs/EVAL.md`; document M3 report split, seeds, and point-baseline metric policy |
| Phase 1 | Reuse | VP-SDE + conditional score net unchanged |
| Phase 2 | Reuse | Use the M2 checkpoint; do **not** redefine training as an M3 exit criterion |
| Phase 3 | Reuse | Same `impute(...)` API; multi-sample mean ± bands for TimeFill |
| Phase 4 | **Yes** | LOCF + linear (+ optional stronger baseline); tables by missing rate; panels; failure notes |
| Phase 5 | No | Streamlit stays out |

---

## Assumptions: carry forward vs extend

`ASSUMPTIONS.md` locks **M1** and **M2**. For M3, **extend** (do not silently rewrite M1/M2 contracts).

### Carry forward

1. **Dataset / schema:** UCI Beijing Multi-Site Air-Quality; `F = 6` pollutant features; `(B, T, F)` windows after hourly last-value-in-bin resample (`T = 64` default).
2. **Mask convention:** `1 = observed`, `0 = missing`; missing cells in `x` use sentinel `0.0`.
3. **Mask composition:** `m_eval = m_native * m_artificial`; metrics only on cells with `m_native == 1` and `m_artificial == 0`.
4. **Normalization:** DSM / reverse SDE in normalized space; **reported** MAE / RMSE / CRPS / coverage in **original units** via `FeatureScaler.inverse_transform` (same as M2 WP4).
5. **TimeFill imputer:** VP-SDE + Diffrax reverse sampling, `K` samples, quantile bands.
6. **Irregular wall-clock SDE:** still deferred; regular grid + masks only.

### New / lock for M3 (document under an “M3” section in `ASSUMPTIONS.md`)

1. **Report split:** prefer the frozen **test** partition for final tables (val was for M2 model selection / snapshots). Document if you temporarily use val for CPU smoke.
2. **Protocol grid:** MCAR rates **0.10 / 0.30 / 0.50**; optional contiguous block (`block_len` from config, default `8`).
3. **Seeds:** artificial-mask seed and TimeFill sampler seed fixed in `configs/bench_m3.yaml` and logged in artifacts.
4. **Point-baseline probabilistic metrics:** LOCF / linear produce a single fill. **Default policy:** treat the point fill as a degenerate ensemble (`K = 1`) so CRPS / coverage columns stay populated and comparable; note in tables that baselines are non-probabilistic. Alternative (must document if chosen): mark CRPS / coverage **N/A** for point methods and compare those metrics for TimeFill only.
5. **Edge policy for LOCF / linear:** if no prior observation exists (leading gap) or a feature is entirely missing in the window, fall back to a documented rule (recommended: global train-feature mean from the saved scaler, or leave sentinel and exclude that cell from metrics if still missing after fill). Unit-test the chosen rule.
6. **Artifacts root:** `artifacts/m3/`.
7. **Stronger baseline:** CSDI-style / BRITS / Transformer is **strongly preferred** but **not required** for M3 exit; if deferred, say so explicitly in README / `EVAL.md`.

---

## Decisions to lock before WP1

Resolve these in WP0 and write them into `docs/EVAL.md` + `ASSUMPTIONS.md`:

| Decision | Recommended default |
|---|---|
| Report split | **test** (val only for smoke / debug) |
| Point-baseline CRPS / coverage | Degenerate ensemble **`K = 1`** |
| Stronger baseline | Timebox (e.g. 1–2 days); else defer with a one-line note |
| LOCF / linear edge policy | Train-feature mean fill when no usable neighbor |

---

## Suggested repo layout additions for M3

Keep M1 / M2 code; add the benchmark surface:

```text
ASSUMPTIONS.md                     # add Milestone 3 section (do not erase M1/M2)
docs/
  EVAL.md                          # EDIT: M3 run contract (split, seeds, point-baseline policy)
  milestone_3_walkthrough.md       # this file
configs/
  real_m2.yaml                     # unchanged train path; M3 reads ckpt/scaler paths
  bench_m3.yaml                    # NEW: rates, seeds, K, baselines, artifact_dir
src/
  baselines/                       # NEW
    __init__.py
    locf.py
    linear.py
    protocol.py                    # optional: shared (x, m) → filled interface helpers
  eval/
    metrics.py                     # reuse
    m2_val_impute.py               # reuse inverse-transform / metric dump patterns
    m3_benchmark.py                # NEW: protocol loop, table assembly (optional module)
  sde/ models/ train/ sample/ data/  # reuse M1/M2
scripts/
  run_m3_benchmark.py              # NEW: end-to-end comparison runner
tests/
  test_baselines_locf_linear.py    # hermetic tiny windows; no UCI download
  test_m3_protocol_masks.py        # optional: metric-cell composition on fixtures
artifacts/
  m3/
    metrics_by_rate.csv / .json
    comparison_table.md            # optional rendered summary
    panels/                        # truth vs TimeFill vs baselines
```

---

## Work packages (do in this order)

### WP0 — M3 eval surface + assumptions (½ day)

Align with Phase 0 / Phase 4 prep already started in M2:

- Confirm primary dataset and feature list still match `docs/EVAL.md` (Beijing, `F = 6`).
- Extend `docs/EVAL.md`:
  - M3 **report split** (test)
  - protocol grid (MCAR 10/30/50%, optional block)
  - seed fields
  - point-baseline CRPS / coverage policy
  - baseline list: LOCF + linear **required**; stronger baseline **preferred**
- Extend `ASSUMPTIONS.md` with an **M3** section (report split, edge policy, artifact root, stronger-baseline status).
- Add config skeleton `configs/bench_m3.yaml`: ckpt / scaler paths, rates, `K_samples`, `solver_steps`, seeds, `artifact_dir`.

**Done when:** `EVAL.md` + M3 assumptions + `bench_m3.yaml` exist; no code required yet beyond stubs if useful.

---

### WP1 — LOCF + linear imputers

**`src/baselines/`**

- Shared contract with the rest of the repo:
  - inputs: values `x` and observation mask `m`, shapes `(B, T, F)` (or `(T, F)` with a thin batch wrapper)
  - `m`: `1` observed, `0` missing; missing `x` cells are sentinel `0.0`
  - output: filled values same shape; **do not** invent a second mask convention
- **LOCF:** per feature along time; carry last observed value forward.
- **Linear interpolation:** per feature along time on observed indices; interpolate interior gaps.
- Implement and unit-test the **edge policy** from WP0 (leading gaps / all-missing feature).

**Tests (hermetic; no network / full UCI dump)**

- Interior single-gap and multi-gap cases match hand-computed fills.
- Leading / trailing gaps follow the documented edge policy.
- Native-missing vs artificial-missing: baselines see the **evaluation observation mask** (what is available at impute time), not the ground-truth native mask alone.
- Shape and dtype sanity (`float32` preferred for consistency with M1/M2).

**Done when:** `pytest tests/test_baselines_locf_linear.py` passes on tiny fixtures.

---

### WP2 — Benchmark runner (Phase 4 core)

**`scripts/run_m3_benchmark.py`** (+ optional `src/eval/m3_benchmark.py`)

For each protocol in the grid (and each method):

1. Load processed windows (`data/processed/m2/`) and **test** indices; load scaler + TimeFill checkpoint from M2 artifacts.
2. Draw artificial masks with a fixed seed; compose `m_eval = m_native * m_artificial`.
3. Build imputer inputs: observed context only (targets zeroed / masked per M1/M2 contract).
4. Run methods:
   - **TimeFill:** existing reverse-SDE impute → `samples`, `mean`, quantiles; inverse-transform to original units.
   - **LOCF / linear:** point fill in the space you document (recommended: **original units** on inverse-transformed windows, or fill in normalized space then inverse — pick one and stick to it).
5. Score with `src/eval/metrics.py` on artificial targets with ground truth only.
6. Append rows to a long-form table: `method`, `protocol`, `mask_rate` / `block_len`, `mae`, `rmse`, `crps`, `coverage_50`, `coverage_90`, seeds, `n_target_cells`.

**Artifacts**

- `artifacts/m3/metrics_by_rate.csv` and `.json`
- Optional markdown/HTML summary table for README paste
- Qualitative panels under `artifacts/m3/panels/`: truth, TimeFill mean ± bands, LOCF, linear on the same held-out gaps

**Done when:** one command reproduces tables + at least one comparison panel from the M2 checkpoint alone (plus processed data).

---

### WP3 — Stronger baseline (timeboxed)

Priority from `plan.md` / `EVAL.md`:

1. Faithful **CSDI-style** discrete conditional diffusion (or a thin wrapper around an existing implementation), **or**
2. **BRITS** / simple Transformer imputer, **or**
3. **Defer** with an explicit note in README / `EVAL.md` / M3 assumptions.

Rules of engagement:

- Do **not** redefine the project as “CSDI in JAX.”
- Keep SDE + Diffrax as the primary method; the stronger baseline is a measuring stick.
- If shipped: same protocols, same metric cells, same original-unit reporting.

**Done when:** either the stronger baseline appears in `metrics_by_rate.*`, or docs state it is deferred for v1.

---

### WP4 — Failure analysis + docs / regression hygiene

- Short failure notes (README or `artifacts/m3/failure_notes.md`): long block gaps, high MCAR rate, sparse features / stations — where TimeFill wins or loses vs LOCF / linear.
- Optional lightweight structure metrics (ACF / cross-corr error) — nice-to-have, not required for exit.
- Optional downstream utility smoke (forecaster on imputed vs truth) — stretch only.
- README: “How to run (M3)” — prepare data (M2) → train (M2) → `run_m3_benchmark.py`.
- Keep M1 toy path and M2 prepare/train/impute green.

**Done when:** a clean clone with data + M2 checkpoint can reproduce the benchmark tables without reading source; M1/M2 tests still pass.

---

## Exit criteria checklist (from plan M3 + Phase 4)

- [x] M3 assumptions + `EVAL.md` run contract (split, seeds, point-baseline policy)
- [x] LOCF + linear implemented and unit-tested (edge policy documented)
- [x] Benchmark runner over frozen protocols (MCAR 10/30/50%; optional block)
- [x] Metrics tables by missing rate: MAE / RMSE / CRPS / coverage (per locked policy)
- [x] Qualitative comparison panels under `artifacts/m3/`
- [x] Claim discipline: beat LOCF / linear on CRPS and coverage where applicable; competitive MAE — or document shortfalls honestly
- [x] Stronger baseline shipped **or** explicitly deferred
- [x] README M3 runbook; M1 + M2 paths still pass
- [x] Irregular-time status still stated as deferred (unless deliberately promoted)

---

## Suggested implementation order (calendar)

| Day | Focus |
|---|---|
| 1 | WP0: `EVAL.md` / M3 assumptions / `bench_m3.yaml`; lock the four decisions |
| 2 | WP1: LOCF + linear + hermetic tests |
| 3–4 | WP2: benchmark script, metrics tables, comparison panels |
| 5 | WP3: stronger baseline **or** deferral note |
| 6 | WP4: failure notes, README runbook, full pytest green |

---

## Risks specific to M3 (from plan + ASSUMPTIONS)

| Risk | M3 mitigation |
|---|---|
| Metric gaming / silent protocol drift | Freeze rates, split, seeds, and metric cells in `EVAL.md`; log them in artifacts |
| Unfair baseline comparison | Same masks, same target cells, same original units; document edge policy |
| Overclaiming vs CSDI | Prefer shipping a stronger baseline; otherwise state “vs LOCF/linear only” clearly |
| Project collapses into “CSDI in JAX” | Keep SDE+Diffrax primary; timebox any discrete diffusion baseline |
| Point baselines break CRPS / coverage | Lock degenerate `K=1` or N/A policy in WP0 before coding the runner |
| Val leakage into reported SOTA-style tables | Report on **test**; use val only for debug smoke |
| Slow full-grid reverse sampling | Config knobs for `K` / solver steps; document CPU vs GPU wall time; allow a smoke subset of windows |
| M2 checkpoint / scaler mismatch | Resolve paths via `bench_m3.yaml`; fail fast if scaler feature count ≠ `F` |

---

## What “done” looks like demowise

```bash
# M2 data + checkpoint (once)
python scripts/prepare_m2_data.py --config configs/real_m2.yaml
python scripts/train_m2_real.py --config configs/real_m2.yaml

# M3 benchmark
python scripts/run_m3_benchmark.py --config configs/bench_m3.yaml

pytest tests/ -q
```

You should get:

1. `artifacts/m3/metrics_by_rate.csv` (and `.json`) with TimeFill vs LOCF vs linear by protocol  
2. At least one comparison panel showing held-out gaps vs mean ± bands vs point baselines  
3. A documented claim (or honest shortfall) on CRPS / coverage / MAE vs LOCF and linear  

That is the handoff into **M4 — Product** (Streamlit upload → impute → download), which can reuse a small benchmark slice as a demo “run evaluation” affordance.

---

## References to keep open while coding

1. Tashiro et al., *CSDI* (NeurIPS 2021) — conditional masking protocols; stronger baseline target  
2. Song et al., *Score-Based Generative Modeling through SDEs* (ICLR 2021) — TimeFill primary method  
3. Project `plan.md` Phase 4 exit criteria; `docs/EVAL.md`; `ASSUMPTIONS.md` M1/M2 contracts to extend  
4. M2 runbook in `README.md` and `docs/milestone_2_walkthrough.md` (data prep, ckpt, original-unit metrics)
