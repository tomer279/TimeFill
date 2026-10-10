# Continuous-Time Diffusion Model for Synthetic Time-Series

Continuous-time (VP-SDE) score-based imputation for multivariate time series,
implemented in JAX / Equinox / Diffrax / Optax.

## Milestone 1 status

**M1 — Toy path (Phases 0–3 on synthetic data)** is the current focus:

- Fixed-length regular-grid windows (`T=64`, `F ∈ {1,2,3}`)
- MCAR random + contiguous block masks
- VP-SDE forward process + conditional temporal score net
- Conditional DSM training with checkpoints / loss curves
- Diffrax reverse-SDE multi-sample imputation
- Lightweight masked MAE / RMSE / CRPS / coverage + artifact plots

Out of scope until later milestones: PhysioNet / real CSV loaders, Streamlit,
and LOCF / linear baseline tables (M3).

See `ASSUMPTIONS.md` for the locked tensor / mask / sentinel contract and
`docs/milestone_1_walkthorugh.md` for the work-package walkthrough.

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


## License

Unless otherwise noted, this project is released under the MIT License — see
`LICENSE`. Replace this section if you choose a different SPDX license before
publishing.