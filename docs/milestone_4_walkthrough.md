# Milestone 4 Walkthrough — Product (Streamlit)

## What M4 is

From `plan.md`:

> **M4 — Product:** Streamlit upload → impute → download.

M4 is Phase 5 — the **product layer**. It reuses the M2 checkpoint + scaler and
the M1/M2/M3 imputation contract so a data scientist can use TimeFill **without
reading the training code**: upload a messy CSV → dependence-aware fills with
uncertainty bands → download a completed dataset.

**Success look:**  
Bundled sample CSV (or user upload matching the frozen schema) → schema
validation + missingness summary → reverse-SDE impute with pretrained weights →
interactive mean ± bands (optional sample overlay) → download imputed CSV
(mean fill; optional multi-sample long format). A “run evaluation on demo
dataset” control may reproduce a **small** M3-style benchmark slice.

**Explicitly deferred:** Portfolio polish / full README rewrite (**M5**).
True irregular-time SDE dynamics stay deferred (same as M2/M3) unless you
explicitly promote them. On-the-fly fine-tune in the UI is optional and must
not block the pretrained demo path.

---

## Scope map (plan → M4)

| Plan phase | In M4? | What you actually build |
|---|---|---|
| Phase 0 | Reuse / thin extend | Keep `docs/EVAL.md`; document demo schema, bundled sample, and UI seed / `K` defaults |
| Phase 1 | Reuse | VP-SDE + conditional score net unchanged |
| Phase 2 | Reuse | Use the M2 checkpoint + scaler; training is not an M4 exit criterion |
| Phase 3 | Reuse | Same `impute(...)` API; multi-sample mean ± bands |
| Phase 4 | Reuse (optional slice) | Optional “run evaluation” button over a **small** frozen demo slice; full M3 grid stays in `scripts/run_m3_benchmark.py` |
| Phase 5 | **Yes** | Streamlit: upload → validate → impute → charts → download |

---

## Assumptions: carry forward vs extend

`ASSUMPTIONS.md` locks **M1**, **M2**, and **M3**. For M4, **extend** (do not
silently rewrite earlier contracts).

### Carry forward

1. **Dataset / schema (pretrained path):** UCI Beijing Multi-Site Air-Quality;
   `F = 6` pollutants in frozen order (`PM2.5`, `PM10`, `SO2`, `NO2`, `CO`, `O3`);
   windows `(B, T, F)` after hourly last-value-in-bin resample (`T = 64` default).
2. **Mask convention:** `1` = observed, `0` = missing; missing cells in `x` use
   sentinel `0.0`. Never infer missingness from “nonzero in `x`”.
3. **Normalization:** reverse SDE in normalized space via saved
   `FeatureScaler`; **displayed / downloaded** values in **original units**
   after `inverse_transform` (same as M2 WP4 / M3).
4. **TimeFill imputer:** VP-SDE + Diffrax reverse sampling, `K` samples,
   quantile bands; observed context replacement conditioning as in M1/M2.
5. **Irregular wall-clock SDE:** still deferred; UI may accept timestamps for
   display / resample-to-grid, but dynamics remain regular grid + masks.
6. **Baselines:** LOCF / linear exist for optional demo eval; stronger baseline
   remains deferred for v1 unless already shipped.

### New / lock for M4 (document under an “M4” section in `ASSUMPTIONS.md`)

1. **Demo path:** pretrained M2 checkpoint + scaler required; default UX must
   complete on the bundled sample in **under a few minutes** (plan exit).
2. **Bundled sample CSV:** small, git-friendly slice under e.g.
   `app/sample_data/` (or `data/demo/`) matching the frozen feature list;
   document column names and how native NaNs become `m`.
3. **Upload schema:** required columns = frozen features (and `timestamp`
   recommended); optional `entity` / site. Reject or clearly warn on unknown /
   missing required columns; do not silently remap names.
4. **Windowing in the UI:** uploaded series → same resample / window rules as
   M2 prep (or a documented thin adapter). If the upload is shorter/longer than
   `T`, document pad / truncate / multi-window behavior.
5. **Demo corruption (sidebar):** optional MCAR / block mask rate to *simulate*
   gaps on otherwise complete demo cells; compose with native missingness the
   same way as eval (`m = m_native * m_artificial` when demo-masking).
6. **UI knobs (defaults for speed):** `K` (samples), solver steps, optional
   quantile levels for bands; keep defaults smaller than full M3 report if
   needed for Streamlit latency. Log effective knobs in the session / download
   metadata when practical.
7. **Downloads:**
   - Required: imputed CSV **mean fill** in original units (observed cells
     preserved; targets filled).
   - Optional: multi-sample export in long format
     (`sample_id`, time, feature, value).
8. **On-the-fly fine-tune:** plan mentions LR / epochs in the sidebar **if**
   supported. **Recommended default:** defer fine-tune for v1 M4; pretrained
   only. If shipped, isolate behind an explicit toggle and do not make it the
   happy path.
9. **Artifacts / app config:** e.g. `configs/app_m4.yaml` pointing at M2
   checkpoint, scaler, sample CSV, default `K` / steps; runtime outputs may live
   under `artifacts/m4/` only if you persist demo eval slices (optional).
10. **“Run evaluation” button:** reproduces a **small** benchmark slice (few
    windows, one or two rates), not the full M3 grid; reuse M3 metrics helpers
    where possible.

---

## Decisions to lock before WP1

Resolve these in WP0 and write them into `ASSUMPTIONS.md` (+ a short M4 note in
`docs/EVAL.md` or README runbook):

| Decision | Recommended default |
|---|---|
| Happy path | **Pretrained only** (no fine-tune in v1 UI) |
| Bundled sample | Thin Beijing-schema CSV under `app/sample_data/` |
| Upload → model input | Resample / window like M2; reject schema mismatches |
| Default `K` / solver steps | Latency-first (document vs M3 report settings) |
| Demo eval button | Small frozen slice; optional; full tables stay M3 CLI |
| Chart library | Plotly (interactive bands) or Streamlit native charts |

---

## Suggested repo layout additions for M4

Keep M1–M3 code; add the product surface:

```text
ASSUMPTIONS.md                     # add Milestone 4 section (do not erase M1–M3)
docs/
  EVAL.md                          # EDIT: short M4 demo schema / sample note
  milestone_4_walkthrough.md       # this file
configs/
  real_m2.yaml                     # unchanged train path
  bench_m3.yaml                    # unchanged full benchmark
  app_m4.yaml                      # NEW: ckpt, scaler, sample path, UI defaults
app/
  streamlit_app.py                 # NEW: main entry (plan layout)
  sample_data/                     # NEW: bundled demo CSV
  # optional helpers (prefer thin wrappers over duplicating train/eval logic)
  #   io_validate.py
  #   session_impute.py
src/
  sample/ models/ sde/ data/ eval/ baselines/  # reuse
scripts/
  # optional: export_demo_csv.py from processed windows
tests/
  test_app_schema_validate.py      # hermetic CSV fixtures; no Streamlit server required
  test_app_window_adapter.py       # optional: upload → (x, m) contract
artifacts/
  m4/                              # optional: persisted demo-eval outputs
```

Launch (illustrative):

```bash
streamlit run app/streamlit_app.py
```

---

## Work packages (do in this order)

### WP0 — M4 product contract + assumptions (½ day)

- Extend `ASSUMPTIONS.md` with an **M4** section (demo path, schema, knobs,
  download contract, fine-tune status, irregular-time status).
- Add `configs/app_m4.yaml`: paths to M2 `best_score_net.eqx` +
  `feature_scaler.npz`, bundled sample path, default `K`, solver steps, seeds.
- Decide and document: pretrained-only vs fine-tune; pad/truncate policy;
  demo-eval slice size.
- Optional one-paragraph “How to run (M4)” stub in README (full polish is M5).

**Done when:** M4 assumptions + `app_m4.yaml` exist; no Streamlit UI required yet.

---

### WP1 — Schema validation + bundled sample

- Ship a **small** demo CSV with the frozen `F = 6` columns (+ `timestamp`).
- Implement validation:
  - required columns present and numeric (after parse)
  - missingness summary (per-feature % NaN / empty)
  - clear user-facing errors (no stack-trace-only failures for bad uploads)
- Adapter: CSV → `(x, m)` windows in the M1/M2 tensor contract (sentinel `0.0`,
  float masks). Reuse `FeatureScaler` transform before impute.
- Hermetic unit tests on tiny fixtures (no UCI download, no GPU).

**Done when:** `pytest` covers validate + adapt; bundled sample passes validation.

---

### WP2 — Impute path behind the UI (core product)

Wire Streamlit (or a thin service module called by Streamlit) to:

1. Load checkpoint + scaler from `app_m4.yaml`.
2. Build observed context from upload / sample (optional demo corruption masks).
3. Call existing reverse-SDE `impute(...)` with UI `K` / steps.
4. Inverse-transform mean and quantiles to original units.
5. Preserve observed cells in the exported mean fill.

Keep business logic out of pure Streamlit widgets where practical so tests can
call the same functions.

**Done when:** one non-UI (or minimal) script/function call imputes the bundled
sample from the M2 checkpoint alone.

---

### WP3 — Streamlit UX (Phase 5 tasks)

From `plan.md` Phase 5:

1. **Sidebar:** demo mask rate; `K`; solver step settings; (fine-tune controls
   only if WP0 unlocked them).
2. **Uploader** + missingness summary.
3. **Charts:**
   - original series with gaps
   - imputed mean + uncertainty bands
   - optional overlay of posterior samples
4. **Downloads:** mean-fill CSV; optional multi-sample long format.
5. **“Run evaluation on demo dataset”** (optional but plan-listed): small
   benchmark slice; show a compact metrics table (reuse `src/eval/metrics.py`
   / M3 helpers). Do not require full M3 runtime in the app.

**Done when:** `streamlit run app/streamlit_app.py` completes the happy path on
the bundled sample with pretrained weights in a few minutes.

---

### WP4 — Latency, docs, regression hygiene

- Tune default `K` / steps for Streamlit; document CPU vs GPU expectations
  (plan risk: sampling too slow → pretrained + smaller `K` / fewer steps).
- Fail fast on missing checkpoint / scaler / feature-count mismatch.
- README: “How to run (M4)” — need M2 artifacts → launch Streamlit → upload /
  sample → download.
- Keep M1 toy, M2 prepare/train/impute, and M3 benchmark paths green.
- Do not advertise irregular-time dynamics as done.

**Done when:** clean clone with M2 checkpoint can demo without reading source;
existing pytest suite still passes.

---

## Exit criteria checklist (from plan M4 + Phase 5)

- [x] M4 assumptions + `app_m4.yaml` (ckpt, scaler, sample, UI defaults)
- [x] Bundled sample CSV + schema validation + missingness summary
- [x] Streamlit happy path: sample or upload → impute → charts → download mean CSV
- [x] Uncertainty UX: mean ± bands (optional sample overlay)
- [x] Optional multi-sample download and/or demo-eval button (plan tasks; timebox)
- [x] Demo completes in under a few minutes with pretrained weights
  (latency-first defaults: `K=8`, `solver_steps=20`, `max_windows=4`; wall
  time is machine-dependent)
- [x] On-the-fly fine-tune shipped **or** explicitly deferred
  (**deferred for v1**; pretrained-only happy path)
- [x] README M4 runbook; M1 + M2 + M3 paths still documented
- [x] Irregular-time status still stated as deferred (unless deliberately promoted)

---

## Suggested implementation order (calendar)

| Day | Focus |
|---|---|
| 1 | WP0: M4 assumptions / `app_m4.yaml` / lock UI decisions |
| 2 | WP1: bundled sample + validate/adapt + hermetic tests |
| 3–4 | WP2–WP3: impute wiring + Streamlit charts + downloads |
| 5 | WP3 polish: demo-eval button (or deferral note); latency knobs |
| 6 | WP4: README runbook, fail-fast paths, full pytest green |

---

## Risks specific to M4 (from plan + ASSUMPTIONS)

| Risk | M4 mitigation |
|---|---|
| Sampling too slow for Streamlit | Pretrained checkpoint; smaller default `K` / solver steps; document wall time |
| Schema drift vs M2 `F = 6` | Strict column validation; fail if scaler feature count ≠ upload |
| Silent unit mismatch | Impute in normalized space; download / plot in original units only |
| Reimplementing train/eval in the app | Thin adapters over `src/sample`, `src/eval`, checkpoint loaders |
| Demo eval ≠ M3 claims | Label UI metrics as a smoke slice; full tables remain `run_m3_benchmark.py` |
| Overclaiming irregular time | Timestamps for resample/display only; ASSUMPTIONS keep dynamics deferred |
| Fine-tune blowing the “few minutes” exit | Default pretrained-only; fine-tune behind toggle or defer |
| Large CSV / many windows | Cap windows processed in the UI; document batching |

---

## What “done” looks like demowise

```bash
# Prerequisites (once): M2 data + checkpoint
python scripts/prepare_m2_data.py --config configs/real_m2.yaml
python scripts/train_m2_real.py --config configs/real_m2.yaml

# M4 product
streamlit run app/streamlit_app.py

pytest tests/ -q
```

You should get:

1. A Streamlit demo that loads the bundled sample (or a valid upload)  
2. Interactive charts of gaps vs imputed mean ± bands  
3. A downloadable mean-fill CSV in original units  
4. (Optional) multi-sample export and/or a small demo evaluation table  

That is the handoff into **M5 — Portfolio polish** (README problem statement,
citations, metrics, limitations).

---

## References to keep open while coding

1. Project `plan.md` Phase 5 tasks + M4 definition of done  
2. `ASSUMPTIONS.md` M1–M3 contracts (mask, sentinel, scaler, Beijing `F = 6`)  
3. `docs/EVAL.md` frozen feature list and mask composition  
4. M2/M3 runbooks in `README.md`, `docs/milestone_2_walkthrough.md`,
   `docs/milestone_3_walkthrough.md` (ckpt paths, original-unit metrics)  
5. Tashiro et al., *CSDI* — product positioning relative to prior art (do not
   redefine the project as a CSDI port)
