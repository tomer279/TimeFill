# Milestone 1 Assumptions

This document locks the data and modeling contracts for M1 (toy path).
Later milestones may relax these assumptions; they must not be changed silently
inside M1.

## Windowing

- Series are represented as **fixed-length windows** on a **regular time grid**.
- Default window length: `T = seq_len = 64`.
- Grid spacing is uniform (`Δt` constant). Irregular wall-clock timestamps are
  out of scope for M1; time may be treated as integer indices `0 … T-1`.
- Batch tensor layout is always `(B, T, F)` for values and masks.

## Features

- Number of channels `F = n_features` is small: `F ∈ {1, 2, 3}` for M1.
- Toy generators live in `src/data/synthetic.py`:
  - **Sine:** feature `f` has `f+1` cycles per window, phase `2π f / F`,
    plus small i.i.d. Gaussian noise (default std `0.05`).
  - **AR:** diagonal VAR(1) / AR(1) with `A = a I` (default `a = 0.8`) and
    Gaussian innovations (default std `0.1`).

## Mask convention

- Mask tensor `m` has the same shape as values `x`: `(B, T, F)`.
- `m[..., ] == 1` means **observed** (conditioning context).
- `m[..., ] == 0` means **missing** (imputation target).
- Mask dtype is floating (`float32`) so it can be concatenated with model inputs.

## Missing value sentinel

- Observed entries keep their true values inside `x`.
- Missing entries in `x` are stored as the sentinel **`0.0`**
  (`MISSING_VALUE_SENTINEL` in `src/data/schema.py`).
- Downstream code must use `m`, not “nonzero in `x`”, to detect missingness,
  because a true observed zero is allowed.

## Missingness protocols (M1)

- **MCAR random:** drop entries independently at rate `mask_rate`.
- **Contiguous block:** for each `(batch, feature)`, drop one contiguous
  time span of length `block_len` with a uniformly random start index
  (`src/data/masking.py`).
- Mask APIs return `(x_obs, m)` where observed cells are unchanged and
  missing cells are the sentinel `0.0`.
- WP1 mask comparison figure path:
  `artifacts/m1/mask_random_vs_block.png`
  (via `scripts/plot_m1_masks.py` / `save_mask_comparison`).
- Training / evaluation treat masked cells as targets; the score network still
  **sees observed context** via `(x * m, m)`-style conditioning (CSDI-style).

## Loss focus

- Denoising score matching (DSM) loss is computed **primarily on masked /
  target cells** (`m == 0`).
- **M1 / WP4 choice:** observed cells (`m == 1`) receive **zero** loss weight.
  Config key `train.obs_loss_weight` defaults to `0.0`. A positive value would
  add a light auxiliary DSM term on observed cells; that is disabled for M1.
- Training artifacts (WP4):
  - checkpoints: `artifacts/m1/checkpoints/` (best val loss)
  - loss CSV / plot: `artifacts/m1/loss_curves.csv`, `artifacts/m1/loss_curves.png`
  - val snapshot stub (until WP5 reverse impute): `artifacts/m1/val_snapshots/`

## Diffusion / training (config keys)

- VP-SDE uses linear `β(t)` between `beta_min` and `beta_max` over diffusion
  horizon `T_diff` (implemented in WP2: `src/sde/vp_sde.py`).
- Closed-form marginal: `x_t = mean(x0, t) + std(t) * eps` with
  `std^2 → 1` as `t → T_diff` (variance-preserving).
- WP2 forward-noising figure path:
  `artifacts/m1/forward_noising.png`
  (via `scripts/plot_m1_forward.py` / `save_forward_noising`).
- Training uses `lr`, `epochs`, AdamW `weight_decay`, `t_eps`, mask mode,
  logging / checkpoint knobs (WP4: `src/train/`).
- Sampling uses `K_samples` posterior draws and `solver_steps` reverse steps (WP5).

## Reverse SDE imputation (WP5)

- Implementation: `src/sample/reverse_sde.py` via Diffrax `diffeqsolve` +
  `Euler` (Euler–Maruyama) and `VirtualBrownianTree`.
- Time coordinate: integrate forward in `τ = T_diff - t` from `τ=0` to
  `τ = T_diff - t_eps`, with drift `dY/dτ = -VPSDE.reverse_drift(...)`.
- **Noise init (default `missing_only`):** at `t = T_diff`, missing cells
  (`m == 0`) are i.i.d. `N(0, 1)`; observed cells (`m == 1`) are set to the
  data values in `observations`. Alternative `full`: entire window is
  `N(0, 1)` (still ends with hard observation replacement).
- **Observation replacement (M1):** during the solve, drift and diffusion are
  multiplied by `(1 - m)` so observed coordinates stay fixed; after the solve,
  observed cells are hard-replaced with `observations` again. Continuous
  conditioning also uses the score net inputs `(mask, x_obs)`.
- Public API: `ReverseSDEImputer.impute(observations, mask, timestamps, key,
  k_samples) -> samples, mean, quantiles` with model/SDE/solver bundled in
  `ImputeContext`. `timestamps` may be grid indices `0 … T-1` and are not used
  in the M1 dynamics.
- Outputs: `samples` is `(B, K, T, F)` (or `(K, T, F)` if unbatched);
  `mean` matches the data layout; `quantiles` has trailing axis size 2
  (default 10% / 90%).

## Lightweight evaluation (WP6)

- Metrics are computed **only on masked / target cells** (`m == 0`) against
  known synthetic ground truth:
  - point: MAE / RMSE of the posterior mean
  - probabilistic: empirical ensemble CRPS from `K` samples
  - calibration (optional): empirical coverage of central 50% / 90% intervals
- End-to-end runner: `scripts/run_m1_toy.py` (config `configs/toy_m1.yaml`).
- WP6 figures under `artifacts/m1/`:
  - `forward_noising.png`
  - `mask_random_vs_block.png`
  - `imputation_mean_bands.png`
- LOCF / linear baseline tables remain **out of scope** until M3.

## Out of scope for M1

- PhysioNet / real CSV loaders
- Streamlit app
- LOCF / linear baseline comparison tables
