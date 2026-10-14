# Milestone 5 Walkthrough — Portfolio polish

## What M5 is

From `plan.md`:

> **M5 — Portfolio polish:** README with problem statement, citations, metrics, limitations.

M5 is the **portfolio / v1 closure layer**. It does **not** add a new modeling
phase. It turns the already-shipped stack (M1 toy → M2 Beijing train → M3
LOCF/linear evidence → M4 Streamlit) into something a hiring manager or
collaborator can understand in one pass: **what problem, what method, what
evidence, what limits, how to run**.

**Success look:**  
A top-level `README.md` that leads with the problem and product story, cites
prior art honestly (especially CSDI / Song SDE / Diffrax), surfaces **real M3
numbers** (or a clear pointer + compact table), states assumptions and deferred
work without overclaim, and keeps a short “reproduce v1” path (install → M2
artifacts → Streamlit and/or M3 table). `ASSUMPTIONS.md` + `docs/EVAL.md`
remain the contract; the README becomes the front door.

**Explicitly deferred (unless you deliberately promote them):** stretch goals
from `plan.md` (true irregular-time Neural SDE, block-gap curriculum,
constrained imputation, privacy/memorization checks). M5 documents those as
future work — it does not implement them.

Also closes the **v1 definition of done** from `plan.md`:

- Reproducible training + imputation scripts
- Reported metrics vs at least two baselines (LOCF + linear)
- Working Streamlit demo on a sample dataset
- Clear statement of assumptions (windowing, missingness, compute limits)

---

## Scope map (plan → M5)

| Plan phase | In M5? | What you actually do |
|---|---|---|
| Phase 0 | **Polish / surface** | Keep `docs/EVAL.md`; README links metrics, splits, masks, baselines |
| Phase 1–3 | Reuse | No new SDE / score / impute API; describe method at portfolio depth |
| Phase 4 | **Surface evidence** | Lift M3 table / panel into README (or compact summary + link to `docs/m3_failure_notes.md`) |
| Phase 5 | Reuse | Point to Streamlit happy path; do not rebuild the app |
| M5 (milestone) | **Yes** | Problem statement, citations, metrics, limitations, claim discipline |

---

## Assumptions: carry forward vs extend

`ASSUMPTIONS.md` locks **M1–M4**. For M5, **extend** with a short “M5 / v1
portfolio” section; do **not** silently rewrite earlier contracts.

### Carry forward

1. **Dataset / schema:** UCI Beijing Multi-Site Air-Quality; `F = 6` pollutants;
   windows `(B, T, F)` after hourly last-value-in-bin (`T = 64` default).
2. **Mask convention:** `1` = observed, `0` = missing; sentinel `0.0` in `x`.
3. **Normalization:** train/impute in normalized space; report / download in
   **original units**.
4. **Evidence claim scope:** TimeFill vs **LOCF and linear only**; stronger
   baseline deferred for v1.
5. **Product path:** pretrained M2 checkpoint + scaler; M4 Streamlit defaults
   for latency.
6. **Irregular wall-clock SDE:** still **deferred**; must not be advertised as
   a completed differentiator (`plan.md` Relation to Prior Art).

### New / lock for M5 (document under an “M5” section in `ASSUMPTIONS.md`)

1. **README is the portfolio surface:** one narrative README, not five stacked
   “status” essays as the only story (runbooks can stay as subsections or
   collapse into “Reproduce”).
2. **Metrics source of truth:** numbers in the README must match a frozen
   artifact (`artifacts/m3/metrics_by_rate.csv` and/or filled
   `docs/m3_failure_notes.md`). No hand-wavy “we beat baselines” without cells.
3. **Claim discipline (non-negotiable):**
   - Probabilistic win vs LOCF / linear where evidence supports it
     (CRPS / coverage).
   - MAE: competitive; document shortfalls (e.g. easy MCAR vs linear).
   - No win claimed vs CSDI / BRITS / Transformer.
   - No claim of completed irregular-time SDE dynamics.
4. **Citations:** anchor Song (score-SDE), Tashiro (CSDI), Diffrax / Kidger
   ecosystem; dataset citation (Beijing / Zhang et al. as in `EVAL.md`);
   optional Oh et al. / PhysioNet as motivation, not as “we ran PhysioNet.”
5. **Limitations section required:** windowed regular grid; Beijing-only
   evidence; point-baseline comparison; CPU latency / `K` tradeoff; native
   missingness composition; deferred items list.
6. **License / naming:** confirm MIT (`LICENSE.txt`) and product name
   **TimeFill** in the hero blurb.
7. **Artifacts:** no new training required; optional `docs/` portfolio assets
   (one README figure path, e.g. a copied panel under `docs/figures/` if you
   want git-friendly images — do not commit large `artifacts/`).

---

## Decisions to lock before WP1

| Decision | Recommended default |
|---|---|
| README structure | Hero → problem → method (short) → evidence → demo → reproduce → assumptions/limits → citations |
| Metrics in README | Compact table (or 1 protocol highlight + link to full `m3_failure_notes.md`) |
| Figure | One qualitative panel (mean ± bands vs gaps) + optional small metrics screenshot |
| PhysioNet | Mention as **motivation / future**, not as a completed v1 dataset |
| Stretch goals | “Future work” bullets only |
| Milestone status sections | Compress M1–M4 into a short “What’s in the box” + keep detailed How-to-run |

---

## Suggested repo layout additions for M5

Mostly docs; almost no new `src/`:

```text
ASSUMPTIONS.md                     # EDIT: add Milestone 5 / v1 portfolio section
README.md                          # EDIT: full portfolio rewrite (main deliverable)
docs/
  EVAL.md                          # EDIT: optional one-line “see README for portfolio summary”
  milestone_5_walkthrough.md       # this file
  m3_failure_notes.md              # reuse as metrics appendix (already filled)
  figures/                         # OPTIONAL: 1–2 git-friendly PNGs for README
plan.md                            # unchanged; M5 closes its DoD
LICENSE.txt                        # confirm README License section matches
```

No new training configs. Do not regress `configs/app_m4.yaml` / M3 bench paths.

---

## Work packages (do in this order)

### WP0 — M5 portfolio contract (½ day)

- Add **Milestone 5** section to `ASSUMPTIONS.md` (claim discipline, metrics
  source, deferred list, irregular-time wording).
- Outline the target README TOC; decide which M3 rows appear inline.
- Inventory existing assets: `docs/m3_failure_notes.md`, M3 panels under
  `artifacts/m3/panels/` (local), Streamlit demo path.
- List phrases that must **not** appear (e.g. “irregular-time SDE complete”,
  “beats CSDI”).

**Done when:** M5 assumptions drafted; README outline agreed; banned claims
listed.

---

### WP1 — Problem statement + method framing

Rewrite the top of `README.md` so a stranger gets:

1. **Product one-liner** (messy multivariate CSV → probabilistic fills + bands).
2. **Why it matters** (point imputers hide uncertainty; break structure) —
   aligned with `plan.md` Problem Framing.
3. **Method sketch (short):** conditional VP-SDE score model; Diffrax reverse
   sampling; `K` samples → mean ± quantiles; observed-context conditioning.
4. **Positioning vs CSDI:** shared goal; wedge = continuous-time SDE + Diffrax
   + product UX; irregular-time **goal deferred** for v1 — say so next to the
   wedge, not in fine print only.

**Done when:** first screen of README could stand alone as a portfolio blurb
without opening `src/`.

---

### WP2 — Evidence: metrics + limitations

- Pull headline results from `docs/m3_failure_notes.md` /
  `artifacts/m3/metrics_by_rate.csv` into README (full table or honest summary
  + link).
- State protocols: test split, MCAR 10/30/50, block_len 8, original units,
  metric cells = native observed ∩ artificial missing.
- Add **Limitations** (required by Phase 4 exit + M5):
  - Regular-grid windows + masks (not first-class irregular Δt dynamics)
  - Evidence on Beijing pollutants only (PhysioNet not run for v1)
  - Baselines = LOCF / linear; stronger baseline deferred
  - MAE does not dominate linear on easy short-gap MCAR (if that remains true)
  - Streamlit uses smaller `K` / steps than the M3 report
  - Compute: CPU vs GPU expectations
- Optional: one qualitative figure (imputation bands) with caption “held-out
  gaps, original units”.

**Done when:** a reader can see **numbers + limits** without hunting through
milestone status walls.

---

### WP3 — Reproduce path + demo pointer

Keep runbooks accurate, but make the **happy portfolio path** obvious:

1. Install (`pip install -e .` / `.[app]`)
2. Data + M2 train (or note if you later ship a downloadable checkpoint — v1
   may still require local train)
3. `streamlit run app/streamlit_app.py`
4. Optional: `python scripts/run_m3_benchmark.py ...` for full tables

Cross-link: `ASSUMPTIONS.md`, `docs/EVAL.md`, M1–M4 walkthroughs (as appendix,
not the hero). Confirm License section matches `LICENSE.txt`.

**Done when:** “clone → demo” and “clone → metrics” are each a short,
copy-pasteable section.

---

### WP4 — Citations, hygiene, v1 checklist

- **References** section from `plan.md` Key References (+ dataset citation).
  Use consistent academic style.
- Remove or demote “Portfolio polish (**M5**)” deferred lines from older README
  milestone blocks once M5 ships.
- Grep README / docs for overclaims (irregular-time done; SOTA; beats CSDI).
- Smoke: `pytest tests/ -q` still green; no code contract changes required for
  M5.
- Optional polish: architecture one-liner diagram (Mermaid in README), GIF of
  Streamlit — timebox; not a blocker if docs are solid.

**Done when:** v1 DoD checklist below is checkable from README alone + existing
scripts.

---

## Exit criteria checklist (from plan M5 + v1 DoD)

- [x] `ASSUMPTIONS.md` has an M5 / v1 portfolio section (claim discipline +
  deferred list)
- [x] README leads with **problem statement** (not only milestone changelogs)
- [x] README includes **citations** (Song SDE, CSDI, Diffrax/Kidger, dataset)
- [x] README includes **metrics** tied to M3 artifacts / failure notes
- [x] README includes honest **limitations** (incl. irregular-time deferred;
  stronger baseline deferred)
- [x] Positioning vs CSDI is explicit; no “first ever imputer” claim
- [x] Working Streamlit path documented (`pip install -e ".[app]"` → M2
  artifacts → `streamlit run ...`)
- [x] Reproducible train / impute / benchmark scripts still documented
- [x] Assumptions (windowing, missingness types, compute limits) stated clearly
- [x] Stretch goals listed as future work only (unless deliberately promoted)
- [x] pytest suite still green; no silent contract breakage

---

## Suggested implementation order (calendar)

| Day | Focus |
|---|---|
| 1 | WP0: M5 assumptions + README outline + banned claims |
| 2 | WP1: hero + problem + method + CSDI positioning |
| 3 | WP2: metrics table / figure + limitations |
| 4 | WP3–WP4: reproduce path, citations, grep for overclaims, final pass |

---

## Risks specific to M5

| Risk | M5 mitigation |
|---|---|
| Overclaiming irregular time | Put deferred status next to method wedge; match `ASSUMPTIONS.md` |
| Overclaiming vs CSDI | Cite as prior art; baselines = LOCF/linear only |
| Metrics drift / cherry-picking | Freeze source file + seeds; link full table |
| README becomes a second ASSUMPTIONS | Keep contracts in `ASSUMPTIONS.md` / `EVAL.md`; README summarizes |
| Rewriting code “while polishing” | M5 is docs-first; code only for broken links / tiny doc bugs |
| Hiding MAE shortfalls | Keep failure-notes honesty in the Limitations / Results narrative |

---

## What “done” looks like demowise

```bash
# Docs / claims: open README.md — problem, metrics, limits, citations

# Still green
pytest tests/ -q

# Demo path (unchanged from M4)
pip install -e ".[app]"
streamlit run app/streamlit_app.py

# Evidence path (unchanged from M3)
python scripts/run_m3_benchmark.py --config configs/bench_m3.yaml
```

You should get:

1. A portfolio-grade README a stranger can skim in ~5 minutes
2. Evidence and limitations that match M3 / ASSUMPTIONS
3. Clear “how to run the demo” without reading training code
4. v1 definition of done satisfied on paper and in the repo

That closes the milestone ladder in `plan.md` (M1 → M5). Stretch goals remain
optional post-v1 work.

---

## References to keep open while writing

1. `plan.md` — Objective, Problem Framing, M5 DoD, Key References, Relation to
   Prior Art, Stretch Goals
2. `docs/m3_failure_notes.md` — frozen claim checklist + results table
3. `ASSUMPTIONS.md` (M1–M4) — contracts the README must not contradict
4. `docs/EVAL.md` — protocols, feature list, mask composition
5. `docs/milestone_4_walkthrough.md` — product path to link, not rebuild
6. Tashiro et al. (CSDI); Song et al. (score-SDE); Kidger / Diffrax
