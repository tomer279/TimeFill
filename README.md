# TimeFill

Probabilistic missing-value imputation for multivariate time series —
with uncertainty bands, not just point estimates.

Under the hood: a continuous-time (VP-SDE) score-based model in
JAX / Equinox / Diffrax / Optax.

## Milestone 1 status

**M1 — Toy path (Phases 0–3 on synthetic data)** is complete and kept green:

- Fixed-length regular-grid windows (`T=64`, `F ∈ {1,2,3}`)
- MCAR random + contiguous block masks
- VP-SDE forward process + conditional temporal score net
- Conditional DSM training with checkpoints / loss curves
- Diffrax reverse-SDE multi-sample imputation
- Lightweight masked MAE / RMSE / CRPS / coverage + artifact plots

LOCF / linear baseline tables are under Milestone 3; the Streamlit product
is Milestone 4 below. Stronger baseline is deferred for v1.

See `ASSUMPTIONS.md` for the locked tensor / mask / sentinel contract and
`docs/milestone_1_walkthorugh.md` for the work-package walkthrough.

## Milestone 2 status

**M2 — Real data train** on **UCI Beijing Multi-Site Air-Quality**
(Zhang et al.; see `docs/EVAL.md`).

**What works**
- Prepare → train → val-impute path on hourly regular-grid windows
  (`T=64`, `F=6`: PM2.5, PM10, SO2, NO2, CO, O3)
- Native + artificial mask composition: `m_train = m_native * m_artificial`
- Train-only `FeatureScaler`; DSM in normalized space; metrics / plots in
  **original units**
- Stable DSM training with checkpoints, loss curves, and val snapshots under
  `artifacts/m2/`
- Checkpoint reverse-SDE imputation + lightweight MAE / RMSE / CRPS /
  coverage_50 / coverage_90 on artificial targets with ground truth

**Deferred (do not advertise as done)**
- Irregular wall-clock as **first-class SDE dynamics** (resample / bin →
  regular windows + masks only; see `ASSUMPTIONS.md`)
- Stronger baseline (CSDI / BRITS / Transformer) **deferred for v1**
- Portfolio polish beyond this runbook (**M5**)

**Later milestones (shipped below):** LOCF / linear tables → Milestone 3;
Streamlit product → Milestone 4.

Contracts: `ASSUMPTIONS.md` (M1 + M2), evaluation freeze: `docs/EVAL.md`,
walkthrough: `docs/milestone_2_walkthrough.md`.

## Milestone 3 status

**M3 — Benchmark** (Phase 4 evidence layer) on the frozen M2 Beijing path.

**What works**
- Same held-out protocols and metric cells as `docs/EVAL.md` /
  `ASSUMPTIONS.md` (M3): MCAR **10% / 30% / 50%**, optional contiguous
  block (`block_len` 8)
- Methods: **TimeFill** (VP-SDE + Diffrax, multi-sample) vs **LOCF** and
  **linear** (point fills; degenerate `K = 1` for CRPS / coverage columns)
- Report split: **test** (val only via `--smoke`); artifacts under
  `artifacts/m3/` (`metrics_by_rate.csv` / `.json`, `panels/`)
- Metrics in **original units** via `FeatureScaler.inverse_transform`

**Claims (vs LOCF / linear only)**
- Target: beat LOCF / linear on **CRPS** and **coverage** where the
  comparison is meaningful; stay **competitive on MAE**.
- Fill real numbers into `docs/m3_failure_notes.md` after a full test run
  (see Failure notes under How to run (M3)). Do **not** claim a win vs
  CSDI / BRITS / Transformer — that stronger baseline is **deferred for v1**.

**Deferred (do not advertise as done)**
- Irregular wall-clock as **first-class SDE dynamics** (same as M2;
  regular grid + masks only)
- Stronger baseline (CSDI-style / BRITS / Transformer) **deferred for v1**
- Portfolio polish beyond this M3 runbook (**M5**)

Streamlit product → **Milestone 4** below.

Contracts: `ASSUMPTIONS.md` (M1–M3), `docs/EVAL.md` (M3 run contract),
walkthrough: `docs/milestone_3_walkthrough.md`.

## Milestone 4 status

**M4 — Product** (Phase 5 Streamlit layer): bundled sample or upload →
reverse-SDE impute → mean ± bands → download mean-fill CSV (original units).

**What works**
- Pretrained M2 checkpoint + scaler only (no training in the happy path)
- Schema: frozen `F = 6` Beijing columns (`docs/EVAL.md`); exact names; no
  silent remaps
- Latency defaults in `configs/app_m4.yaml`: `K=8`, `solver_steps=20`,
  `smoke.max_windows=4` (smaller than M3 report settings)
- UI: missingness summary, Plotly mean ± bands, optional sample overlay,
  mean-fill + optional multi-sample long CSV
- Optional “Quick quality check” = **smoke slice only** (not the M3 report)
- Non-UI smoke: `python scripts/run_m4_impute_sample.py --config configs/app_m4.yaml`

**Explicitly deferred**
- On-the-fly fine-tune in the UI (**deferred for v1**; pretrained only)
- Irregular wall-clock as first-class SDE dynamics (timestamps for
  resample / display only; regular grid + masks)
- Full M3 benchmark grid in the app → use `scripts/run_m3_benchmark.py`
- Portfolio polish / full README rewrite (**M5**)

Contracts: `ASSUMPTIONS.md` (Milestone 4), `docs/EVAL.md` (M4 demo note),
walkthrough: `docs/milestone_4_walkthrough.md`.

### How to run (M4)

Prerequisites: core install plus the Streamlit extra, and M2 artifacts.

```bash
pip install -e ".[app]"
```

#### 1) M2 artifacts (once)

```bash
python scripts/prepare_m2_data.py --config configs/real_m2.yaml
python scripts/train_m2_real.py --config configs/real_m2.yaml
```

Expect:

- `artifacts/m2/checkpoints/best_score_net.eqx`
- `artifacts/m2/checkpoints/feature_scaler.npz`

Missing checkpoint / scaler / feature-count mismatch fails fast with a clear
message (paths are locked in `configs/app_m4.yaml`).

#### 2) Launch the demo

```bash
streamlit run app/streamlit_app.py
```

In the app: choose the bundled sample (or upload a CSV with the frozen
columns) → **Fill missing values** → inspect mean ± bands → download the
mean-fill CSV. Optional: multi-sample long CSV; **Run a quick quality check**
(smoke only).

#### 3) Compute limits (CPU vs GPU)

- **CPU:** defaults (`K=8`, `solver_steps=20`, ≤4 windows via
  `smoke.max_windows`) target a few minutes. Raising `K` / steps toward M3
  report settings (`50` / `100` in `configs/bench_m3.yaml`) is much slower.
- **GPU:** optional. Install a matching JAX CUDA build (see Install →
  JAX: CPU vs CUDA); same config, shorter wall time.
- **Window batching:** the UI processes at most `smoke.max_windows`
  (default **4**). Longer uploads need another run or a deliberate cap
  increase in `configs/app_m4.yaml`.

Config: `configs/app_m4.yaml`. Full benchmark tables remain:

```bash
python scripts/run_m3_benchmark.py --config configs/bench_m3.yaml
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
- Failure-note template: `docs/m3_failure_notes.md`

### Failure notes (expected win / lose vs LOCF / linear)

Paste numbers from `artifacts/m3/metrics_by_rate.csv` after a **test** run
into `docs/m3_failure_notes.md`. Hypotheses until then:

| Regime | Expect TimeFill | Why |
|---|---|---|
| Long contiguous blocks | Win on CRPS / coverage; MAE mixed | Uncertainty bands; LOCF/linear extrapolate poorly across long gaps |
| High MCAR (50%) | Win on CRPS / coverage; MAE competitive | Multi-feature context; point methods ignore cross-channel structure |
| Sparse / heavily native-missing features | Lose or tie on MAE | Edge policy falls back to train-feature mean; little context for the score net |
| Short interior gaps / low MCAR (10%) | Competitive MAE; CRPS edge smaller | Linear / LOCF already strong on short gaps |

**Claim discipline:** report vs **LOCF and linear only**. No overclaim vs
CSDI (stronger baseline deferred for v1). If CRPS / coverage or MAE fall
short of the M3 target, document the shortfall in the failure notes — do
not paper over it.

## License

Unless otherwise noted, this project is released under the MIT License — see
`LICENSE`. Replace this section if you choose a different SPDX license before
publishing.
