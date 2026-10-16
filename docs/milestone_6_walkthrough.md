# Milestone 6 Walkthrough — Schema-flexible ingest

## What M6 is

From `plan.md`:

> **M6 — Schema-flexible ingest:** Accept messy multivariate CSVs beyond the
> frozen Beijing `F=6` column names, without training or claiming a universal
> imputer yet.

M6 is the first **post-v1** milestone. It generalizes the **ingest / validate /
window** path that M4 hard-coded to Beijing names, so a data scientist can
upload a non-Beijing multivariate CSV and get **validated windows + native
masks + a clear schema / missingness preview** — or a **clear reject** when the
series is too short, too sparse, or schema-invalid.

**Success look:**  
Upload (or load) a non-Beijing fixture CSV → infer or map feature columns,
optional entity/id, timestamps → build native missingness masks → pad /
truncate / window with documented rules → Streamlit (or CLI) **preview** schema
+ missingness **before** impute. Pretrained Beijing path remains the
**exact-schema fallback**. Docs state that **imputation still needs a
compatible checkpoint or M7 train-on-upload**.

**Explicitly out of scope for M6:**

| Deferred to | What |
|---|---|
| **M7** | Train / fine-tune on the uploaded CSV, then impute |
| **M8** | Gap-regime routing (short vs block policies) |
| **M9** | Hosted checkpoint, second-domain claims, stronger baselines |
| Stretch | True irregular-time Neural SDE dynamics |

M6 must **not** silently remap unknown columns to Beijing names, and must
**not** claim that arbitrary CSVs can be imputed with the M2 Beijing
checkpoint.

---

## Scope map (plan → M6)

| Plan phase / milestone | In M6? | What you actually build |
|---|---|---|
| Phase 0 | **Extend** | Document flexible schema contract; keep Beijing `EVAL.md` claim table separate |
| Phase 1–3 | Reuse | No new SDE / score / impute API required for exit |
| Phase 4 / M3 | Reuse | Do not merge non-Beijing ingest into M3 claim tables |
| Phase 5 / M4 | **Extend** | Generalize validate / adapt; preview UI; keep exact-schema Beijing demo |
| M5 | Reuse | Update README Future work / honesty when M6 ships |
| **M6** | **Yes** | Infer/map columns → masks → pad/truncate/window → tests + fixture → preview |
| M7+ | No | Adapt / impute-on-arbitrary-schema stays blocked or clearly gated |

---

## Assumptions: carry forward vs extend

`ASSUMPTIONS.md` locks **M1–M5**. For M6, **extend** with a Milestone 6
section; do **not** silently rewrite Beijing v1 contracts.

### Carry forward

1. **Tensor / mask contract:** `(B, T, F)` windows; `1` = observed, `0` =
   missing; missing `x` = sentinel `0.0`. Never infer missingness from nonzero
   `x`.
2. **v1 evidence / pretrained path:** Beijing `F = 6` exact names; M2
   checkpoint + scaler; M3 vs LOCF/linear only.
3. **Normalization for Beijing impute:** reverse SDE in scaled space;
   display/download in original units.
4. **Regular grid after binning:** v1-compatible path remains resample /
   last-value-in-bin → fixed `T` (default 64). Irregular wall-clock as
   first-class SDE dynamics stays **deferred**.
5. **M4 pad / multi-window spirit:** non-overlapping windows; right-pad short
   tails with sentinel + `m = 0`; optional `max_windows` cap.
6. **Claim discipline:** no silent schema remaps; no universal-imputer claim
   from M6 alone.

### New / lock for M6 (document under an “M6” section in `ASSUMPTIONS.md`)

1. **Two ingest modes:**
   - **Exact Beijing (fallback):** current M4 path — required names match EVAL
     order; pretrained impute allowed.
   - **Flexible schema:** user/inferred feature set `F ≥ 1` (cap documented);
     **ingest + preview only** until M7 (or a matching checkpoint exists).
2. **Column roles:** map or infer `features[]`, optional `timestamp`, optional
   `entity` / `id`. No silent rename of unknown columns to `PM2.5`, etc.
3. **Native missingness:** empty / NaN / documented tokens → `m_native = 0`,
   sentinel in `x`.
4. **Pad / truncate / window rules** (document defaults):
   - After optional regular-grid binning, cut non-overlapping windows of
     length `T`.
   - Short series / remainder: right-pad with sentinel + `m = 0`.
   - Reject (or hard-warn) when usable length / observed density fails
     thresholds (see WP0).
5. **Reject contract:** schema-invalid, too short, or too sparse →
   **user-facing errors**, not stack traces only; never invent Beijing columns.
6. **Impute gate:** flexible-schema path may show preview and then **block
   impute** with a message pointing to M7 / “need compatible checkpoint,”
   unless features exactly match a loaded scaler/checkpoint.
7. **Fixture:** small non-Beijing CSV under e.g. `tests/fixtures/` or
   `app/sample_data/` (git-friendly).
8. **Artifacts:** optional `configs/ingest_m6.yaml` or extend app config; do
   not overwrite M2/M3 claim artifacts.

---

## Decisions to lock before WP1

| Decision | Recommended default |
|---|---|
| Inference vs mapping | Auto-suggest numeric feature columns; user confirms / edits map in UI (or YAML/CLI map) |
| Timestamp | Parse if present; else row order = grid index. Optional hourly bin like M2 when wall-clock present |
| Entity | Optional; group-by for multi-series if present; single series if absent |
| Min length | Reject if usable rows `< T` **and** pad-only would be mostly missing (or: reject if `L < min_rows`, e.g. `T // 4`) — **lock one rule in WP0** |
| Max sparsity | Reject if per-feature or global observed rate `<` threshold (e.g. 5–10%); document |
| Max `F` | Soft cap for UI (e.g. 32); fail clearly above cap |
| Beijing impute | Only when mapped features **exactly** match checkpoint/scaler names + count |
| Preview surface | Streamlit section first; thin CLI optional (`scripts/preview_ingest_m6.py`) |
| Truncate policy | Prefer multi-window + pad remainder (M4); document if you also allow “truncate to first `T` only” toggle |

---

## Suggested repo layout additions for M6

Keep M1–M5 code; generalize ingest without breaking the Beijing demo:

```text
ASSUMPTIONS.md                     # EDIT: Milestone 6 section
docs/
  EVAL.md                          # EDIT: note flexible ingest ≠ Beijing claim table
  milestone_6_walkthrough.md       # this file
  # optional: ingest_m6.md short user-facing rules
README.md                          # EDIT: Future work — M6 status when shipped
app/
  schema_validate.py               # EDIT or split: exact vs flexible specs
  window_adapter.py                # EDIT: documented reject thresholds if needed
  streamlit_app.py                 # EDIT: schema map + missingness preview; impute gate
  # optional:
  #   flexible_ingest.py           # NEW: infer/map → ValidatedFrame-like
configs/
  app_m4.yaml                      # keep Beijing demo defaults
  # optional ingest_m6.yaml        # thresholds, max_F, sample flexible fixture
tests/
  fixtures/
    toy_non_beijing.csv            # NEW: not PM2.5…O3 schema
  test_flexible_ingest.py          # NEW: map, masks, pad, rejects
  test_app_schema_validate.py      # keep / extend exact-schema tests
scripts/
  # optional: preview_ingest_m6.py
```

Do **not** regress `configs/app_m4.yaml` Beijing happy path or M3 bench claims.

---

## Work packages (do in this order)

### WP0 — M6 ingest contract (½–1 day)

- Add **Milestone 6** to `ASSUMPTIONS.md` (two modes, reject rules,
  pad/truncate, impute gate, irregular-time still deferred).
- Lock thresholds: min length, max sparsity, max `F`, timestamp/entity
  behavior.
- Inventory current M4 APIs: `app/schema_validate.py`,
  `app/window_adapter.py`, Streamlit upload path.
- List banned behaviors: silent Beijing remap; claiming universal impute;
  mixing flexible CSV into M3 tables.

**Done when:** M6 assumptions drafted; reject thresholds agreed; extension
points listed.

---

### WP1 — Infer / map columns + native masks

- Accept CSV with arbitrary headers.
- Infer candidates: numeric columns → features; datetime-like → timestamp;
  id-like → entity (heuristics + overrides).
- Build a typed frame compatible with the existing tensor contract (`values`,
  `native_mask`, names, optional timestamps/entity).
- Clear errors: zero features, duplicate maps, non-numeric feature cells after
  parse, empty file.
- Unit tests on the non-Beijing fixture + hermetic string-row fixtures.

**Done when:** non-Beijing CSV → validated values/masks without Beijing column
names.

---

### WP2 — Pad / truncate / window + reject policy

- Apply documented regular-grid path (bin if timestamps; else ordered rows).
- Non-overlapping windows length `T`; right-pad short tails (`m = 0`); optional
  `max_windows`.
- Implement **fail clearly** for too short / too sparse / schema-invalid
  (user-facing messages).
- Tests: short series reject or pad behavior per ASSUMPTIONS; sparse reject;
  valid short-but-allowed pad.

**Done when:** pytest covers accept + reject paths; behavior matches
ASSUMPTIONS.

---

### WP3 — Preview UX (Streamlit and/or CLI)

- Wire preview **before** impute: detected/mapped schema, `F`, length,
  per-feature missingness %, window count after adapt.
- Keep **bundled Beijing sample + exact-schema upload** as pretrained impute
  path.
- Flexible upload: allow preview; **gate impute** unless schema matches loaded
  checkpoint/scaler (message: need M7 or compatible weights).
- Optional CLI: load CSV + map YAML → print summary / write tiny NPZ of
  windows (no training).

**Done when:** Streamlit (or CLI) shows schema/missingness for non-Beijing
fixture; Beijing demo still imputes.

---

### WP4 — Docs, honesty, regression hygiene

- README Future work: mark M6 shipped when exit criteria met; state ingest ≠
  universal impute.
- Short note in `docs/EVAL.md` or ASSUMPTIONS: flexible ingest does not expand
  v1 Beijing metrics.
- Grep for overclaims (“any CSV”, “universal imputer”).
- `pytest tests/ -q` green; M4 exact-schema tests still pass.

**Done when:** exit checklist below is checkable; v1 claims unchanged.

---

## Exit criteria checklist (from plan M6)

- [x] `ASSUMPTIONS.md` has an M6 section (flexible vs exact modes, pad/truncate,
  rejects, impute gate)
- [x] Infer or map feature columns, optional entity/id, timestamps
- [x] Native missingness masks; pad / truncate / window with documented rules
- [x] Clear failures: too short, too sparse, schema-invalid (no silent Beijing
  remaps)
- [x] Unit tests + small **non-Beijing** fixture CSV
- [x] Streamlit (or CLI) schema / missingness **preview before impute**
- [x] Pretrained Beijing exact-schema path still works as fallback
- [x] Docs: imputation on flexible schema still needs compatible checkpoint
  **or M7**
- [x] No merge of flexible-ingest demos into M3 Beijing claim tables
- [x] pytest suite green; M1–M5 contracts not silently broken

---

## Suggested implementation order (calendar)

| Day | Focus |
|---|---|
| 1 | WP0: M6 assumptions + thresholds + banned behaviors |
| 2 | WP1: infer/map + masks + non-Beijing fixture + tests |
| 3 | WP2: window/pad + reject policy + tests |
| 4 | WP3: Streamlit/CLI preview + impute gate; Beijing path intact |
| 5 | WP4: README / EVAL honesty, overclaim grep, full pytest |

---

## Risks specific to M6

| Risk | M6 mitigation |
|---|---|
| Users think any CSV can be imputed with M2 weights | Impute gate + explicit “needs M7 / matching checkpoint” copy |
| Silent remap to Beijing names | Reject unknown→Beijing; tests assert no alias magic |
| Flexible path pollutes v1 metrics | Keep M3/EVAL Beijing-only; README claim discipline |
| Pad-only “success” on tiny series | Hard reject thresholds in ASSUMPTIONS + tests |
| Scope creep into M7 training | Preview-only flexible path; no adapt loop in M6 |
| Irregular-time overclaim | Timestamps for bin/display only; dynamics still grid + masks |

---

## What “done” looks like demowise

```bash
# Flexible ingest preview (illustrative once WP3 exists)
# streamlit: upload tests/fixtures/toy_non_beijing.csv → schema + missingness
# impute button disabled or clearly blocked unless schema matches checkpoint

# Exact-schema fallback (unchanged M4)
pip install -e ".[app]"
streamlit run app/streamlit_app.py
# bundled Beijing sample → impute → download

# Tests
pytest tests/test_flexible_ingest.py tests/test_app_schema_validate.py -q
pytest tests/ -q
```

You should get:

1. Non-Beijing CSV → validated windows + masks (or a clear reject)
2. Schema / missingness preview before any impute
3. Beijing pretrained demo still works on exact names
4. Docs that M6 is ingest flexibility, not a universal imputer (M7 next)

---

## References to keep open while implementing

1. `plan.md` — Post-v1 **M6** tasks / exit criteria; M7 dependency note; risk
   “Universal path overclaims v1”
2. `ASSUMPTIONS.md` — M4 upload / pad policy (extend, don’t erase)
3. `app/schema_validate.py`, `app/window_adapter.py`, `app/streamlit_app.py` —
   current exact-schema path
4. `docs/EVAL.md` — Beijing claim contract stays frozen
5. `docs/milestone_4_walkthrough.md` — product UX patterns to extend
6. `docs/milestone_5_walkthrough.md` — claim discipline / README honesty
7. `README.md` Future work — update when M6 ships
