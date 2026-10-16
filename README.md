# TimeFill

**Messy multivariate CSV → probabilistic fills with uncertainty bands**
(mean ± quantiles), not point-only interpolants.

Point imputers hide uncertainty and can break autocorrelation /
cross-correlation structure. TimeFill targets data scientists who need
dependence-aware fills and bands they can inspect and download.

## Product north star

**Destination (not claimed as shipped in v1):** a **universal** imputation
workflow — upload an arbitrary messy multivariate CSV → **train / adapt a
model on that data** → fill gaps → download completed series with uncertainty
bands. Optionally **route by gap regime** (small / medium / large contiguous
gaps) so short holes can use a light policy and long blocks use a heavier
score-SDE (or specialist) imputer.

**v1 (what this repo proves today):** that stack works end-to-end on one real
path — UCI Beijing air-quality (`F=6`), pretrained M2 checkpoint, Streamlit
demo with **exact schema** (no silent remaps), and honest LOCF / linear
evidence. **Post-v1 M6** adds schema-flexible **ingest / preview** for
non-Beijing CSVs; that is **not** a universal imputer — Flexible-schema
**impute** still needs a compatible checkpoint/scaler or **M7**.
Train-on-upload, gap-size routing, and evidence expansion remain later
post-v1 (see Future work).

## Method (short)

Conditional **VP-SDE** score model (observed-context conditioning via values +
mask). Training: continuous-time denoising score matching. Inference: **Diffrax**
reverse-SDE sampling; draw **K** trajectories → posterior mean and quantile
bands. Stack: **JAX / Equinox / Diffrax / Optax**.

## Positioning vs CSDI

**Shared goal** with CSDI (Tashiro et al.): probabilistic multivariate
imputation, self-supervised masking, observation+mask conditioning, multi-sample
means and uncertainty bands.

**Wedge for this repo:** continuous-time VP-SDE + Diffrax reverse sampling + a
Streamlit product path (upload → bands → download), with a frozen eval contract
in `docs/EVAL.md`.

**Deferred for v1 (stated with the wedge):** irregular wall-clock time as
**first-class SDE dynamics**. Shipped path is resample / bin → regular-grid
windows + masks. Do not read “continuous-time” as “irregular-time complete.”

PhysioNet-style ICU series motivate the broader problem; **v1 evidence is not
a PhysioNet run** (Beijing air-quality path; see `docs/EVAL.md`).

## What’s in the box

- **Toy + real train path** — synthetic M1; UCI Beijing M2 (`F=6`, `T=64`)
- **Evidence (M3)** — TimeFill vs LOCF / linear; see **Evidence** below and
  `docs/m3_failure_notes.md`
- **Demo (M4)** — see **Clone → demo** (local M2 checkpoint + Streamlit)
- **Metrics reproduce** — see **Clone → metrics** (optional full M3 table)
- **Contracts** — `ASSUMPTIONS.md` (incl. Milestone 5 portfolio rules),
  `docs/EVAL.md`
- **Flexible ingest (M6)** — non-Beijing CSV → validate / window / schema +
  missingness **preview**; pretrained impute stays Exact Beijing (or a
  matching checkpoint). Walkthrough: `docs/milestone_6_walkthrough.md`

Milestone walkthroughs and contracts are linked in **Docs appendix**.

## Evidence

**Source of truth:** `docs/m3_failure_notes.md` (filled from
`artifacts/m3/metrics_by_rate.csv` on a **test**-split M3 run). Contracts:
`docs/EVAL.md`, `ASSUMPTIONS.md`. Comparison scope: TimeFill vs **LOCF and
linear only** — no claim vs CSDI / BRITS / Transformer.

### Protocol

- Split: `test`
- Missingness: MCAR rates **10% / 30% / 50%**; contiguous **block_len = 8**
- Units: metrics in **original** (unnormalized) units
- Score cells: `m_native == 1` and `m_artificial == 0` (held-out artificial
  gaps with ground truth)
- Seeds: `mask=0`, `sample=1`
- TimeFill: `K_samples=50`; LOCF / linear: degenerate `K=1`

### Headline results

| method | protocol | rate / block | mae | rmse | crps | coverage_50 | coverage_90 |
|---|---|---|---:|---:|---:|---:|---:|
| timefill | mcar | 0.10 | 35.01 | 159.68 | 26.20 | 0.596 | 0.901 |
| locf | mcar | 0.10 | 42.71 | 168.70 | 42.71 | 0.177 | 0.177 |
| linear | mcar | 0.10 | 33.02 | 141.35 | 33.02 | 0.120 | 0.120 |
| timefill | mcar | 0.30 | 39.01 | 165.34 | 29.19 | 0.578 | 0.901 |
| locf | mcar | 0.30 | 49.18 | 196.10 | 49.18 | 0.162 | 0.162 |
| linear | mcar | 0.30 | 37.45 | 161.23 | 37.45 | 0.103 | 0.103 |
| timefill | mcar | 0.50 | 46.03 | 174.57 | 34.36 | 0.540 | 0.892 |
| locf | mcar | 0.50 | 58.61 | 228.00 | 58.61 | 0.146 | 0.146 |
| linear | mcar | 0.50 | 45.76 | 188.03 | 45.76 | 0.086 | 0.086 |
| timefill | block | 8 | 64.87 | 246.20 | 49.79 | 0.430 | 0.811 |
| locf | block | 8 | 93.57 | 328.29 | 93.57 | 0.104 | 0.104 |
| linear | block | 8 | 68.46 | 249.40 | 68.46 | 0.045 | 0.045 |

Full table + claim checklist: [`docs/m3_failure_notes.md`](docs/m3_failure_notes.md).

**Summary:** TimeFill is the better **probabilistic** imputer vs LOCF / linear
(CRPS and coverage on all four protocols). Point MAE is competitive and wins
on long/block gaps; it does **not** dominate linear on easy short-gap MCAR
(MAE trails linear at MCAR 10%: 35.01 vs 33.02; MCAR 30%: 39.01 vs 37.45).

**Qualitative figure:** optional git-friendly copy under `docs/figures/`
(not committed in v1). After a local M3 run, the panel is at
`artifacts/m3/panels/comparison_timefill_locf_linear.png` (held-out gaps,
original units; see Panels in `docs/m3_failure_notes.md`). Copy that file to
`docs/figures/m3_comparison_panel.png` and embed it here with caption
“held-out gaps, original units.” Do not commit large `artifacts/` dumps.

## Limitations

- **Regular grid, not irregular Δt:** windows + masks after hourly binning;
  irregular wall-clock as first-class SDE dynamics is **deferred** for v1.
- **Beijing-only evidence:** UCI Multi-Site Air-Quality (`F=6`); PhysioNet
  not run for v1.
- **Baselines = LOCF / linear:** stronger baseline (e.g. CSDI) deferred; do
  not read this table as a win vs those methods.
- **MAE shortfall on easy MCAR:** still ~2 points behind linear at MCAR 10%
  and 30%; CRPS / coverage remain the intended differentiators.
- **Streamlit ≠ M3 report settings:** app defaults `K_samples=8`,
  `solver_steps=20` (and ≤4 windows); M3 report uses `50` / `100`
  (`configs/bench_m3.yaml` vs `configs/app_m4.yaml`).
- **CPU vs GPU:** full M3 / M2 defaults are slow on CPU; GPU optional via a
  matching JAX CUDA build. Demo targets a few minutes on CPU with smoke caps.
- **Flexible ingest ≠ universal impute:** M6 accepts messy non-Beijing
  schemas for preview (and clear rejects). The M2 Beijing checkpoint does
  **not** impute arbitrary CSVs; use Exact names, a matching checkpoint, or
  M7 train-on-upload.
- **Point-baseline CRPS / coverage are degenerate `K=1`:** CRPS ≈ absolute
  error; interval width 0 — not calibrated probabilistic scores.

## Clone → demo

Local M2 checkpoint + scaler required (no hosted weights for v1). Latency
defaults are locked in `configs/app_m4.yaml` (`K_samples=8`,
`solver_steps=20`, `smoke.max_windows=4`). Fine-tune-in-UI is not shipped.

Fetch Beijing CSVs first (see How to run (M2)), then:

```bash
pip install -e .
pip install -e ".[app]"

python scripts/prepare_m2_data.py --config configs/real_m2.yaml
python scripts/train_m2_real.py --config configs/real_m2.yaml
# Expect:
#   artifacts/m2/checkpoints/best_score_net.eqx
#   artifacts/m2/checkpoints/feature_scaler.npz

streamlit run app/streamlit_app.py
```

In the app: bundled sample or upload → **Fill missing values** → mean ± bands
→ download. CPU/GPU and window-batching notes: How to run (M4).

## Clone → metrics (optional)

Reproduce the frozen LOCF / linear comparison (test split; slow on CPU at full
defaults). Claim table:
[`docs/m3_failure_notes.md`](docs/m3_failure_notes.md).

```bash
# After M2 artifacts exist (same prepare + train as Clone → demo):
python scripts/run_m3_benchmark.py --config configs/bench_m3.yaml
```

CPU smoke (debug only — not the report table):

```bash
python scripts/run_m3_benchmark.py --config configs/bench_m3.yaml --smoke
```

## Install

Requires Python 3.10+. From the repo root:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
```

For the Streamlit demo, also install the app extra:

```bash
pip install -e ".[app]"
```

This installs the project in editable mode and pulls CPU-friendly defaults for
JAX (via the `jax` package on PyPI). After install, `import src` and
`pytest` should resolve from any working directory.

### JAX: CPU vs CUDA

- **CPU (default):** `pip install -e .` is enough for the M1 toy path.
- **CUDA:** install a JAX CUDA build that matches your CUDA toolkit / driver
  (see [JAX installation](https://jax.readthedocs.io/en/latest/installation.html)).
  Example pattern only — choose the extra that matches your machine:

```bash
# Example (CUDA 12); verify against current JAX docs before running:
pip install -U "jax[cuda12]"
```

GPU is optional for M1; smoke runs and tests are intended to work on CPU.

## How to run (M1)

From the repo root (after `pip install -e .`, or via the script `sys.path`
bootstrap):

```bash
python scripts/run_m1_toy.py --config configs/toy_m1.yaml
pytest tests/ -q
```

Shorter CPU smoke (optional overrides):

```bash
python scripts/run_m1_toy.py --config configs/toy_m1.yaml --epochs 5 --k-samples 8 --solver-steps 20
```

### Expected outputs under `artifacts/m1/`

| Artifact | Meaning |
|---|---|
| `forward_noising.png` | Clean window noised at several `t` |
| `mask_random_vs_block.png` | MCAR vs block mask patterns |
| `imputation_mean_bands.png` | Truth, observations, mean ± bands |
| `loss_curves.csv` / `loss_curves.png` | Train / val DSM loss |
| `checkpoints/best_score_net.eqx` | Best validation score net |

The script also prints a tiny metrics table (`mae`, `rmse`, `crps`,
`coverage_50`, `coverage_90`) on held-out masked sine cells.

### Related scripts

- `scripts/train_m1_dsm.py` — training-only smoke
- `scripts/plot_m1_forward.py` / `scripts/plot_m1_masks.py` — individual figures


## How to run (M2)

Prerequisites: `pip install -e .` (see Install). Do **not** commit raw CSVs,
processed arrays, or checkpoints (`data/raw/`, `data/processed/`, and
`artifacts/` are gitignored).

### 1) Fetch data

1. Download **PRSA2017_Data_20130301-20170228.zip** from the
   [UCI Beijing Multi-Site Air-Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)
   page (dataset id 501).
2. Unzip the per-site `PRSA_Data_*.csv` files into
   `data/raw/beijing_multisite/`.

Details and schema: `docs/EVAL.md`.

### 2) Prepare windows

```bash
python scripts/prepare_m2_data.py --config configs/real_m2.yaml
```

Writes `data/processed/m2/windows.npz` (values, native masks, entity ids,
train/val/test indices). Unit tests use hermetic CSV fixtures and do **not**
need the UCI download.

### 3) Train

```bash
python scripts/train_m2_real.py --config configs/real_m2.yaml
```

Writes under `artifacts/m2/`:

| Artifact | Meaning |
|---|---|
| `checkpoints/best_score_net.eqx` | Best validation score net |
| `checkpoints/feature_scaler.npz` | Train-only feature scaler |
| `loss_curves.csv` / `loss_curves.png` | Train / val DSM loss |
| `val_snapshots/` | Lightweight reverse-impute panels during training |

### 4) Impute validation batch

```bash
python scripts/run_m2_impute_val.py --config configs/real_m2.yaml --ckpt artifacts/m2/checkpoints/best_score_net.eqx
```

| Artifact | Meaning |
|---|---|
| `imputation_mean_bands.png` | Truth / obs / mean ± bands (original units) |
| `val_metrics.json` / `val_metrics.csv` | MAE / RMSE / CRPS / coverage on artificial targets with GT |

### Compute limits (CPU vs GPU)

- **CPU:** full defaults in `configs/real_m2.yaml`
  (`epochs: 100`, `steps_per_epoch: 200`, `sample.K_samples: 50`,
  `sample.solver_steps: 100`) are slow; prefer the smoke overrides below for
  a first end-to-end check.
- **GPU:** optional. After `pip install -e .`, install a matching JAX CUDA
  build (see Install → JAX: CPU vs CUDA). Same scripts; no config change
  required beyond shorter wall time.
- **During train**, val snapshots use
  `train.snapshot_k_samples` / `train.snapshot_solver_steps` (defaults 4 / 20),
  not the full `sample.*` knobs. Full `K` / solver steps apply in
  `run_m2_impute_val.py`.

Suggested CPU smoke overrides:

```bash
# prepare (once, after fetch)
python scripts/prepare_m2_data.py --config configs/real_m2.yaml

# train smoke
python scripts/train_m2_real.py --config configs/real_m2.yaml --epochs 2 --steps-per-epoch 2 --batch-size 2 --snapshot-k 2 --snapshot-steps 10

# impute smoke
python scripts/run_m2_impute_val.py --config configs/real_m2.yaml --ckpt artifacts/m2/checkpoints/best_score_net.eqx --k-samples 8 --solver-steps 20 --batch-size 2
```

### Related M2 scripts / tests

- `scripts/prepare_m2_data.py` — raw CSVs → `windows.npz`
- `scripts/train_m2_real.py` — real-window DSM trainer
- `scripts/run_m2_impute_val.py` — checkpoint → val metrics + bands plot
- `pytest tests/test_real_loader.py tests/test_normalize_fit_train_only.py tests/test_real_train_smoke.py tests/test_m2_impute_val.py -q`

## How to run (M3)

Prerequisites: M2 data + checkpoint (prepare → train). Do **not** redefine
training as an M3 exit criterion; reuse
`artifacts/m2/checkpoints/best_score_net.eqx` and
`artifacts/m2/checkpoints/feature_scaler.npz`.

### 1) Prepare + train (M2, once)

```bash
python scripts/prepare_m2_data.py --config configs/real_m2.yaml
python scripts/train_m2_real.py --config configs/real_m2.yaml
```

### 2) Benchmark

```bash
python scripts/run_m3_benchmark.py --config configs/bench_m3.yaml
```

Writes under `artifacts/m3/`:

| Artifact | Meaning |
|---|---|
| `metrics_by_rate.csv` / `.json` | TimeFill vs LOCF vs linear by protocol |
| `panels/` | Truth vs TimeFill mean ± bands vs point baselines |

Config: `configs/bench_m3.yaml` (rates, seeds, `K`, paths, `artifact_dir`).
Point baselines use degenerate `K = 1` CRPS / coverage (non-probabilistic;
see `docs/EVAL.md`).

### Compute limits (CPU smoke)

- **Full defaults** (`sample.K_samples: 50`, `sample.solver_steps: 100`,
  all test windows) are slow on CPU.
- **`--smoke`:** uses `smoke.*` from the config — default `K_samples: 2`,
  `solver_steps: 10`, and `split: val` (debug only; **not** the report
  table). Set `smoke.max_windows` to a small int (e.g. `4`) for a true
  subset; `null` means all windows in that split.

```bash
# CPU smoke (val + short sampler; set smoke.max_windows in YAML for a cap)
python scripts/run_m3_benchmark.py --config configs/bench_m3.yaml --smoke
```

### Related M3 scripts / tests

- `scripts/run_m3_benchmark.py` — end-to-end comparison runner
- `pytest tests/test_baselines_locf_linear.py tests/test_m3_benchmark_helpers.py -q`
- Filled claim table + win/lose notes: `docs/m3_failure_notes.md`

### Failure notes (vs LOCF / linear — filled)

Frozen from a **test**-split M3 run in
[`docs/m3_failure_notes.md`](docs/m3_failure_notes.md) (source:
`artifacts/m3/metrics_by_rate.csv`). Re-paste from that CSV if you re-run
the full benchmark and numbers change.

| Regime | Observed (TimeFill) | Notes |
|---|---|---|
| Long contiguous blocks (`block_len=8`) | Win CRPS / coverage; MAE beats LOCF and linear | Uncertainty bands; point methods extrapolate poorly across long gaps |
| High MCAR (50%) | Win CRPS / coverage; MAE near-tie with linear | Cross-feature context; RMSE better than both point methods |
| Short-gap MCAR (10% / 30%) | Win CRPS / coverage; MAE trails linear ~2 pts | Linear already strong on short interior gaps |
| Sparse / heavily native-missing features | Risk of lose/tie on MAE | Edge policy falls back to train-feature mean (see ASSUMPTIONS / EVAL) |

**Claim discipline:** report vs **LOCF and linear only**. No overclaim vs
CSDI (stronger baseline deferred for v1). Keep shortfalls in
`docs/m3_failure_notes.md` — do not paper over them.

## How to run (M4)

Prerequisites: `pip install -e ".[app]"` and M2 artifacts (see **Clone →
demo**). Paths are locked in `configs/app_m4.yaml`.

### 1) M2 artifacts (once)

```bash
python scripts/prepare_m2_data.py --config configs/real_m2.yaml
python scripts/train_m2_real.py --config configs/real_m2.yaml
```

Expect:

- `artifacts/m2/checkpoints/best_score_net.eqx`
- `artifacts/m2/checkpoints/feature_scaler.npz`

Missing checkpoint / scaler / feature-count mismatch fails fast with a clear
message.

### 2) Launch the demo

```bash
streamlit run app/streamlit_app.py
```

In the app: choose the bundled sample (or upload a CSV with the frozen
columns) → **Fill missing values** → inspect mean ± bands → download the
mean-fill CSV. Optional: multi-sample long CSV; **Run a quick quality check**
(smoke only).

### 3) Compute limits (CPU vs GPU)

- **CPU:** defaults (`K=8`, `solver_steps=20`, ≤4 windows via
  `smoke.max_windows`) target a few minutes. Raising `K` / steps toward M3
  report settings (`50` / `100` in `configs/bench_m3.yaml`) is much slower.
- **GPU:** optional. Install a matching JAX CUDA build (see Install →
  JAX: CPU vs CUDA); same config, shorter wall time.
- **Window batching:** the UI processes at most `smoke.max_windows`
  (default **4**). Longer uploads need another run or a deliberate cap
  increase in `configs/app_m4.yaml`.

Config: `configs/app_m4.yaml`. Full benchmark tables: **Clone → metrics**.

## Docs appendix

- [`ASSUMPTIONS.md`](ASSUMPTIONS.md) — M1–M6 contracts
- [`docs/EVAL.md`](docs/EVAL.md) — eval / mask / split protocol
- [`docs/milestone_1_walkthorugh.md`](docs/milestone_1_walkthorugh.md)
- [`docs/milestone_2_walkthrough.md`](docs/milestone_2_walkthrough.md)
- [`docs/milestone_3_walkthrough.md`](docs/milestone_3_walkthrough.md)
- [`docs/milestone_4_walkthrough.md`](docs/milestone_4_walkthrough.md)
- [`docs/milestone_5_walkthrough.md`](docs/milestone_5_walkthrough.md)
- [`docs/milestone_6_walkthrough.md`](docs/milestone_6_walkthrough.md)

## Future work

Roadmap toward the **product north star** above. Scheduled milestones and
exit criteria live in [`plan.md`](plan.md) (**Post-v1 milestones (M6–M9)**).
M6 detail: [`docs/milestone_6_walkthrough.md`](docs/milestone_6_walkthrough.md).

**Shipped (post-v1):** **M6 — Schema-flexible ingest** — infer / map columns,
native masks, pad / window / reject thresholds, Streamlit + CLI schema /
missingness **preview before impute**. Schema-flexible ingest is **not** a
universal imputer: Flexible-schema **impute** still needs a **compatible
checkpoint / scaler** or **M7** train-on-upload. v1 Beijing M3 claim tables
are unchanged (`docs/EVAL.md`).

**Next scheduled:** **M7 — Train-on-upload / adapt** → **M8 — Gap-regime
routing** → **M9 — Evidence expansion** (hosted checkpoint, second domain,
or stronger baseline — one primary track).

- **M6 Schema-flexible ingest (shipped):** column mapping, timestamps,
  pad/truncate, clear rejects for too short / sparse / invalid schema;
  Exact Beijing remains the pretrained fallback. See
  [`docs/milestone_6_walkthrough.md`](docs/milestone_6_walkthrough.md).
- **M7 Train-on-upload:** train or fine-tune on the user’s CSV, then impute
  and download. Pretrained impute stays Exact Beijing (or a matching
  checkpoint) until M7.
- **M8 Gap-regime routing:** detect gap length → light / point methods for
  short holes, full TimeFill (or specialist) for medium / large blocks.
- **M9 Evidence expansion:** hosted M2 checkpoint, second-domain evidence
  (e.g. PhysioNet-style), and/or stronger baselines (CSDI / BRITS /
  Transformer) — comparison only until measured; no win claimed by default.

**Unscheduled stretch** (see `plan.md`): true irregular-time latent Neural SDE
encoder–decoder; constrained imputation (business / physical bounds); privacy /
memorization checks if synthetic export is emphasized.
## References

1. Tashiro, Y., Song, J., Song, Y., & Ermon, S. (2021). CSDI: Conditional
   Score-based Diffusion Models for Probabilistic Time Series Imputation.
   *NeurIPS 2021*.
2. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., &
   Poole, B. (2021). Score-Based Generative Modeling through Stochastic
   Differential Equations. *ICLR 2021*.
3. Kidger, P. (2021–). Diffrax / Neural CDE ecosystem (continuous-time
   numerics in JAX). https://github.com/patrick-kidger/diffrax
4. Zhang, S., Guo, B., Dong, A., He, J., Xu, Z., & Chen, S. X. (2017).
   Cautionary tales on air-quality improvement in Beijing. *Proc. R. Soc. A*,
   473, 20170457. Dataset: Beijing Multi-Site Air-Quality (UCI ML Repository,
   id 501; https://doi.org/10.24432/C5RK5G).
5. Oh, Y., Lim, D., & Kim, S. (2024). Stable Neural Stochastic Differential
   Equations in Analyzing Irregular Time Series Data. *ICLR 2024*.
   (Motivation for irregular-time modeling; not a v1 evaluation dataset.)
6. Silva, I., Moody, G., Scott, D. J., Celi, L. A., & Mark, R. G. PhysioNet
   ICU multiparameter time-series resources. (Motivation / future evidence
   path only; **not** run for v1.)

## License

Unless otherwise noted, this project is released under the MIT License — see
`LICENSE.txt`.
