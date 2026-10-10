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

Out of scope until later milestones: Streamlit (M4) and LOCF / linear /
CSDI baseline tables (M3).

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
- LOCF / linear / CSDI baseline tables (**M3**)
- Streamlit app (**M4**)
- Portfolio polish beyond this runbook (**M5**)

Contracts: `ASSUMPTIONS.md` (M1 + M2), evaluation freeze: `docs/EVAL.md`,
walkthrough: `docs/milestone_2_walkthrough.md`.

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

## License

Unless otherwise noted, this project is released under the MIT License — see
`LICENSE`. Replace this section if you choose a different SPDX license before
publishing.
