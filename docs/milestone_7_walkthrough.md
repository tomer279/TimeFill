# Milestone 7 Walkthrough — Train-on-upload / adapt

## What M7 is

From `plan.md`:

> **M7 — Train-on-upload / adapt:** On a user CSV (via M6 ingest), **train or
> fine-tune** a score model on that file, then impute and download mean ± bands.

M7 is the second **post-v1** milestone. M6 unlocked schema-flexible **ingest +
preview** and gated impute. M7 opens that gate for a **user-owned model**: short
adapt (fine-tune and/or train-from-scratch) on the uploaded series → fit a
**train-only** scaler → reverse-SDE impute → download mean ± bands.

**Success look:**  
Non-Beijing fixture CSV → M6 ingest / windows → **adapt with smoke caps** →
`K` posterior samples → downloadable mean-fill (+ optional multi-sample
export). Documented assumptions, compute caps, and failure modes (too little
data, NaN loss, timeout). **Exact Beijing pretrained demo stays intact**; M2
claim artifacts are **not** overwritten as a “universal” model.

**Explicitly out of scope for M7:**

| Deferred to | What |
|---|---|
| **M8** | Gap-regime routing (short vs block policies) |
| **M9** | Hosted checkpoint, second-domain SOTA claims, stronger baselines as product backends |
| Stretch | True irregular-time Neural SDE; full multi-hour UI train |

M7 must **not** merge upload-adapt smoke metrics into M3 Beijing claim tables,
and must **not** claim cross-dataset SOTA from a single-file smoke eval.

---

## Scope map (plan → M7)

| Plan phase / milestone | In M7? | What you actually build |
|---|---|---|
| Phase 0 | **Extend** | Document adapt contract, compute caps, smoke-eval ≠ Beijing claims |
| Phase 1–2 | **Reuse / thin wrap** | VP-SDE + `ConditionalScoreNet` + DSM; `n_features = F` for the upload |
| Phase 3 | Reuse | Same `impute(...)` / mean ± bands / download contract as M4 |
| Phase 4 / M3 | Reuse only | Optional **smoke** metrics on artificial gaps for *this* file — not a new EVAL protocol |
| Phase 5 / M4 | **Extend** | Adapt controls + progress / failure UX; keep pretrained Exact path |
| M5 | Reuse | Claim discipline: adapt demo ≠ universal SOTA |
| M6 | **Depends** | Flexible ingest → windows / masks / preview; lift impute gate after adapt |
| **M7** | **Yes** | Adapt loop + train-only scaler + UI/CLI path + smoke eval + docs |
| M8+ | No | Routing / hosted weights stay later |

---

## Assumptions: carry forward vs extend

`ASSUMPTIONS.md` locks **M1–M6**. For M7, **extend** with a Milestone 7
section; do **not** silently rewrite Beijing v1 or M6 ingest contracts.

### Carry forward

1. **Tensor / mask contract:** `(B, T, F)`; `1` = observed, `0` = missing;
   missing `x` = sentinel `0.0`. Never infer missingness from nonzero `x`.
2. **M6 ingest:** Exact Beijing vs Flexible; reject thresholds; pad /
   multi-window; no silent Beijing remaps.
3. **Normalization:** DSM / reverse SDE in scaled space; display/download in
   original units via `FeatureScaler.inverse_transform`.
4. **Train-only scaler (M2 spirit):** fit on **observed cells in the declared
   train split only**; never on held-out artificial-gap targets used for smoke
   quality check.
5. **v1 evidence path:** Beijing `F = 6` M2 checkpoint + scaler; M3 vs
   LOCF/linear only — frozen unless a later milestone explicitly extends EVAL.
6. **Irregular wall-clock SDE:** still deferred; timestamps for bin/display
   only.
7. **Impute API:** VP-SDE + Diffrax reverse sampling, `K` samples, quantile
   bands, observed-context conditioning.

### New / lock for M7 (document under an “M7” section in `ASSUMPTIONS.md`)

1. **Two product paths after M7:**
   - **Exact Beijing pretrained (unchanged):** load M2 weights → impute (M4).
   - **Upload-adapt:** M6 Flexible (or Exact) ingest → **adapt** → session
     checkpoint + scaler → impute → download.
2. **Adapt modes (pick defaults in WP0):**
   - **Train-from-scratch:** init `ConditionalScoreNet(n_features=F)` for the
     upload’s `F`. **Required** when `F ≠` base checkpoint `F` (Beijing base
     is `F = 6`; Equinox `n_features` is static).
   - **Fine-tune-from-base:** load a base checkpoint **only when** feature
     count (and preferably names / order) match; otherwise refuse fine-tune
     and offer scratch.
3. **Scaler / leakage:** fit `FeatureScaler` on train-split **native-observed**
   cells only. Artificial gaps for smoke eval are applied **after** fit; never
   fold smoke targets into scaler or DSM labels incorrectly.
4. **Smoke caps (Streamlit / CPU):** locked defaults for epochs, steps/epoch,
   batch size, `K`, solver steps, wall-clock timeout — suitable for a demo,
   not a full M2 train.
5. **Failure modes (user-facing):** too few windows / too sparse after M6
   rejects; NaN loss; timeout; OOM — stop cleanly; do not leave a half-written
   “best” that silently becomes the global default.
6. **Artifacts:** session / run dir under e.g. `artifacts/m7/` (or temp session
   dir). **Never** overwrite `artifacts/m2/checkpoints/*` claim artifacts as
   the sole universal model.
7. **Smoke eval:** optional MAE/RMSE (and light CRPS if cheap) on artificial
   gaps on a **held-out** slice of the upload — documented as **file-local
   smoke**, not a new cross-dataset claim.
8. **Impute gate update:** after a successful adapt for this schema, allow
   impute with the **session** checkpoint/scaler; Exact Beijing pretrained
   path remains available independently.

---

## Decisions to lock before WP1

| Decision | Recommended default |
|---|---|
| Default adapt mode | **Scratch** when `F` ≠ base; **fine-tune** only if `F` matches base + user opts in |
| Base checkpoint | Optional: M2 Beijing weights when `F = 6` **and** names match (or user confirms same semantics); else scratch |
| Train / smoke split | Deterministic split of windows (e.g. 80/20 by window index or time); document; no random reshuffle of cells into scaler |
| Smoke caps (UI) | e.g. `epochs ≤ 5`, `steps_per_epoch ≤ 20`, `batch_size ≤ 4`, `K ≤ 8`, `solver_steps ≤ 50`, timeout ~5–15 min — **lock numbers in WP0** |
| CLI vs Streamlit | CLI smoke path first (easier to pytest); Streamlit adapt panel second |
| Session artifacts | `artifacts/m7/<run_id>/` with `best_score_net.eqx`, `feature_scaler.npz`, `adapt_meta.json` |
| Min data for adapt | Reuse M6 rejects; optionally require `n_windows ≥ N` (e.g. 2+) and global observed rate above M6 floor |
| NaN / non-finite loss | Abort adapt; surface message; do not save “best” |
| Eval honesty | Label UI metrics “smoke on this file”; never write into `artifacts/m3/` claim CSVs |
| Download | Same as M4: mean-fill CSV; optional multi-sample long format |

---

## Suggested repo layout additions for M7

Keep M1–M6; add an adapt wrapper without breaking the Beijing demo:

```text
ASSUMPTIONS.md                     # EDIT: Milestone 7 section
docs/
  EVAL.md                          # EDIT: note M7 smoke ≠ Beijing claim table
  milestone_7_walkthrough.md       # this file
README.md                          # EDIT: Future work — M7 status when shipped
app/
  streamlit_app.py                 # EDIT: adapt controls; lift gate after session ckpt
  session_impute.py                # EDIT or thin: load session bundle vs M2
  # optional:
  #   adapt_session.py             # NEW: orchestrate fit scaler + train + save
configs/
  app_m4.yaml                      # keep Beijing pretrained defaults
  # NEW: adapt_m7.yaml             # smoke caps, timeout, artifact_dir, seeds
src/train/
  real_loop.py / artifacts.py      # REUSE; wrap rather than fork if possible
scripts/
  # NEW: adapt_upload_m7.py        # CLI: CSV → adapt → optional smoke impute
tests/
  fixtures/
    toy_non_beijing.csv            # REUSE from M6
  test_adapt_m7.py                 # NEW: hermetic tiny train + scaler leakage tests
artifacts/
  m7/                              # NEW run outputs (gitignored like m2)
```

Do **not** regress Exact Beijing Streamlit impute or M3 claim reproduce paths.

---

## Work packages (do in this order)

### WP0 — M7 adapt contract (½–1 day)

- Add **Milestone 7** to `ASSUMPTIONS.md` (modes, leakage rules, smoke caps,
  artifacts, failure modes, claim discipline).
- Lock: fine-tune vs scratch policy when `F` mismatches; timeout; min
  windows; artifact layout.
- Inventory reuse: `src/train/real_loop.py`, `FeatureScaler`,
  `app/preview_ingest.py` impute gate, `app/session_impute.py`.
- List banned behaviors: overwrite M2 claim ckpt; fold smoke gaps into
  scaler; advertise universal SOTA; silent Beijing remap.

**Done when:** M7 assumptions drafted; caps agreed; extension points listed.

---

### WP1 — Train-only scaler + adapt loop (library)

- From M6 `AdaptedWindows` (or equivalent tensors): declare train/smoke
  split.
- Fit `FeatureScaler` on **train native-observed** cells only; transform
  train/smoke with those stats.
- Build `ConditionalScoreNet(n_features=F)`; scratch init, or load base when
  allowed.
- Short DSM loop reusing `RealConditionalDSMLoss` / trainer primitives with
  **smoke caps** from config.
- Save session checkpoint + scaler + metadata (`feature_names`, `F`, `T`,
  seeds, mode).
- Tests: scaler ignores smoke-held-out cells; `F` mismatch refuses
  fine-tune; tiny synthetic windows train ≥1 step without NaN.

**Done when:** library API: windows → session artifacts under
`artifacts/m7/…`.

---

### WP2 — CLI path: ingest → adapt → impute → export

- Script e.g. `scripts/adapt_upload_m7.py`: load non-Beijing fixture via M6
  ingest → adapt → impute `K` → write mean-fill CSV (and optional samples).
- Surface compute caps and failures (timeout, NaN, too little data) as clear
  exit codes / messages.
- Optional: print smoke MAE/RMSE on artificial gaps (file-local only).
- Keep `scripts/train_m2_real.py` / M2 artifact paths untouched.

**Done when:** one command on `tests/fixtures/toy_non_beijing.csv` completes
adapt + downloadable fill (with aggressive caps).

---

### WP3 — Streamlit: adapt → impute → download

- Flexible (and optionally Exact) upload: after preview, **Run adapt** with
  visible caps / ETA / cancel-or-timeout messaging.
- On success: bind session checkpoint/scaler; **enable impute** (update
  `evaluate_impute_gate` / equivalent).
- Keep bundled Beijing sample + Exact pretrained path as the zero-train happy
  path.
- Failure UX: too little data, NaN loss, timeout — no stack-trace-only dead
  ends.
- Download mean ± bands as in M4.

**Done when:** Streamlit E2E on non-Beijing fixture with caps; Beijing
pretrained still works without adapt.

---

### WP4 — Smoke eval, docs, honesty, regression hygiene

- Document smoke protocol (mask rate, split, metrics) in ASSUMPTIONS / short
  EVAL note: **not** merged into M3 Beijing tables.
- README Future work: mark M7 shipped when exit criteria met; north star
  still honest (one-file adapt ≠ proven universal).
- Grep for overclaims (“universal imputer”, “any CSV SOTA”).
- `pytest tests/ -q` green; M4 Exact + M6 preview tests still pass; M2
  artifacts unchanged by adapt runs.

**Done when:** exit checklist below is checkable; v1 claims unchanged.

---

## Exit criteria checklist (from plan M7)

- [x] `ASSUMPTIONS.md` Milestone 7 (modes, train-only scaler, caps, failures,
  artifacts) + smoke protocol note; `docs/EVAL.md` § M7 smoke honesty
- [x] Capped adapt loop: `src/train/adapt_loop.py` / `adapt_m7.py`;
  `configs/adapt_m7.yaml` smoke caps
- [x] Train-only scaler + leakage tests: `tests/test_adapt_m7.py`
- [x] CLI `scripts/adapt_upload_m7.py` + Streamlit adapt panel; failure UX
- [x] Exact Beijing path intact; promote refuses `artifacts/m2`
- [x] File-local smoke (`--smoke-eval` / demo-eval helpers); not merged to
  `artifacts/m3/`
- [x] E2E on `tests/fixtures/toy_non_beijing.csv` (CLI + UI helpers)
- [x] Caps documented in ASSUMPTIONS + README Future work (M7 shipped)
- [x] `pytest tests/ -q` (see WP4 command list); M1–M6 contracts unchanged

---

## Suggested implementation order (calendar)

| Day | Focus |
|---|---|
| 1 | WP0: M7 assumptions + caps + banned behaviors |
| 2 | WP1: scaler leakage rules + capped adapt loop + unit tests |
| 3 | WP2: CLI E2E on toy non-Beijing fixture |
| 4 | WP3: Streamlit adapt panel + impute gate lift; Beijing path intact |
| 5 | WP4: smoke-eval honesty, README / EVAL notes, full pytest |

---

## Risks specific to M7

| Risk | M7 mitigation |
|---|---|
| UI train too slow / unstable on CPU | Hard smoke caps + timeout; CLI for longer runs; fine-tune only when `F` matches |
| `F` ≠ 6 fine-tune from Beijing ckpt | Architecture is static in `n_features` — refuse fine-tune; scratch only |
| Leakage via scaler or shared masks | Train-only fit; held-out artificial gaps after fit; unit tests |
| Overwriting M2 claim weights | Separate `artifacts/m7/`; never default-save into `artifacts/m2/checkpoints/` |
| Users treat smoke MAE as SOTA | UI + docs label “smoke on this file”; no M3 table merge |
| Scope creep into M8 routing | Single policy (always TimeFill) after adapt; routing deferred |
| NaN loss leaves corrupt “best” | Abort without promoting checkpoint; clear message |

---

## What “done” looks like demowise

```bash
# CLI adapt + impute (illustrative once WP2 exists)
python scripts/adapt_upload_m7.py \
  --csv tests/fixtures/toy_non_beijing.csv \
  --config configs/adapt_m7.yaml

# Streamlit: upload non-Beijing CSV → preview (M6) → Run adapt → Impute → Download
pip install -e ".[app]"
streamlit run app/streamlit_app.py

# Exact-schema fallback (unchanged)
# bundled Beijing sample → pretrained impute → download (no adapt required)

# Tests
pytest tests/test_adapt_m7.py tests/test_flexible_ingest.py -q
pytest tests/ -q
```

You should get:

1. Non-Beijing CSV → adapt under caps → session checkpoint + scaler
2. Impute with `K` samples → downloadable mean-fill (± bands / optional
   samples)
3. Beijing pretrained demo still works without training
4. Docs that M7 is **file-local adapt**, not a new Beijing/M3 claim expansion

---

## References to keep open while implementing

1. `plan.md` — Post-v1 **M7** tasks / exit criteria; risk “Train-on-upload too
   slow / unstable on CPU”
2. `ASSUMPTIONS.md` — M2 train-only scaler; M4 demo path; M6 ingest + impute
   gate
3. `docs/milestone_6_walkthrough.md` — ingest / preview / gate to lift
4. `scripts/train_m2_real.py`, `src/train/real_loop.py`,
   `src/data/normalize.py` — reuse for capped adapt
5. `app/preview_ingest.py`, `app/session_impute.py`, `app/streamlit_app.py` —
   product wiring
6. `docs/EVAL.md` — Beijing claim contract stays frozen
7. `README.md` Future work — update when M7 ships
