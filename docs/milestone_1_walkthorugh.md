# Milestone 1 Walkthrough — Toy Path

## What M1 is

From `plan.md`:

> **M1 — Toy path:** forward SDE + conditional score net + reverse samples on synthetic sine/AR data with masks.

M1 is the first working loop of Phases 1–3 on synthetic data only: generate toy series → mask → forward-noise → train a conditional score net → reverse-sample imputations → inspect plots/metrics. Real datasets (PhysioNet, air quality), Streamlit, and baseline tables are **out of scope** until M2–M4.

**Success look:**  
`observations, mask, timestamps → K posterior samples, mean, quantiles` on sine / AR windows, with unit tests and a few saved figures.

---

## Scope map (plan → M1)

| Plan phase | In M1? | What you actually build |
|---|---|---|
| Phase 0 | Partial | Toy schema + mask protocols + lightweight MAE/RMSE/CRPS on held-out cells |
| Phase 1 | Yes | VP-SDE forward, masks, Equinox score net, forward/mask plots |
| Phase 2 | Yes (toy) | Conditional DSM loss, JAX train loop, checkpoints, loss curves |
| Phase 3 | Yes (toy) | Diffrax reverse SDE → multi-sample imputation API |
| Phase 4–5 | No | Defer baselines / Streamlit |

---

## Assumptions to lock early

Document these (README or short `ASSUMPTIONS.md`):

1. **Fixed-length windows** on a regular grid (e.g. `T=64` or `128` steps). Irregular wall-clock can be encoded later; M1 may use uniform Δt.
2. **Multivariate channels** `F` small (e.g. 1–3): sine phases / AR(1) coupled features.
3. **Missingness:** random (MCAR) and contiguous block masks on observed entries.
4. **Conditioning:** observed values stay visible to the score net; loss focuses on masked / target positions (CSDI-style, continuous-time).
5. **Tech:** Python, JAX, Equinox, Diffrax, Optax; Matplotlib for plots.

---

## Suggested repo layout for M1

Align with the plan’s layout; only build what M1 needs:

```text
plan.md
README.md                          # short M1 status + how to run
configs/
  toy_m1.yaml                      # T, F, mask rates, SDE betas, train hyperparams
src/
  data/
    synthetic.py                   # sine / AR generators
    masking.py                     # random + block masks
    schema.py                      # window tensors + mask conventions
  sde/
    vp_sde.py                      # forward marginals, drift/diffusion, reverse drift helper
  models/
    score_net.py                   # Equinox TCN or temporal Transformer
  train/
    loss.py                        # conditional continuous DSM
    loop.py                        # jit train step, logging, checkpoint
  sample/
    reverse_sde.py                 # Diffrax reverse sampling / imputation
  eval/
    metrics.py                     # MAE, RMSE, CRPS, simple coverage
    plots.py                       # forward noise, masks, imputation bands
tests/
  test_shapes_masks.py
  test_vp_sde_marginals.py
  test_impute_api.py
scripts/
  run_m1_toy.py                    # end-to-end: data → train → impute → save figs
```

---

## Work packages (do in this order)

### WP0 — Contract & config (½ day)

- Define tensor convention, e.g.  
  `x: (B, T, F)`, `m: (B, T, F)` with `1 = observed`, `0 = missing`.  
  Keep observed values in `x`; missing cells can be 0 or a sentinel—pick one and stick to it.
- Freeze toy generators:
  - **Sine:** multi-frequency / phase-shifted channels + small Gaussian noise.
  - **AR:** discrete AR(1)/VAR(1) on a grid.
- Config keys: `seq_len`, `n_features`, `mask_rate`, `block_len`, `beta_min/max`, `T_diff` (diffusion horizon), `lr`, `epochs`, `K_samples`, `solver_steps`.

**Done when:** `configs/toy_m1.yaml` exists and a loader returns a batch with known shapes.

---

### WP1 — Synthetic data + masking (Phase 0/1)

**`src/data/synthetic.py`**
- `generate_sine_windows(...)` → clean `x0`
- `generate_ar_windows(...)` → clean `x0`

**`src/data/masking.py`**
- `random_mask(x, rate)` → `(x_obs, m)`
- `block_mask(x, block_len)` → `(x_obs, m)`

**Tests:** mask rates ≈ requested; block masks are contiguous; shapes match; observed entries unchanged.

**Plots:** side-by-side random vs block mask on one window.

---

### WP2 — VP-SDE forward process (Phase 1)

Implement Variance Preserving SDE (Song et al.):

\[
\mathrm{d}x = -\tfrac{1}{2}\beta(t)\,x\,\mathrm{d}t + \sqrt{\beta(t)}\,\mathrm{d}W
\]

with linear \(\beta(t)\) between `beta_min` and `beta_max`.

Public helpers (at least two methods per class for your pylint rule):

- `marginal_prob(x0, t) → (mean, std)` for closed-form perturbation  
  \(x_t = \alpha(t)\,x_0 + \sigma(t)\,\varepsilon\)
- `forward_sample(x0, t, eps)` / `sde_coefficients(t) → (drift_coeff, diffusion)`

**Tests:** at `t→0`, `mean ≈ x0`, `std ≈ 0`; at large `t`, variance ≈ 1 (VP); broadcasting over `(B,T,F)`.

**Plot:** one clean window → several `t` levels → near-Gaussian at `t≈T`.

---

### WP3 — Conditional score network (Phase 1)

Conditioning interface (from the plan):

| Input | Role |
|---|---|
| `y_t` | Noisy state |
| `t` | Diffusion time (sinusoidal embedding) |
| `m` | Observation mask |
| observed values | Context channels (e.g. `x_obs * m`) |
| optional time encoding | Wall-clock / index encoding along `T` |

**Architecture preference:** temporal inductive bias — TCN or small temporal Transformer / U-Net-along-time, not a flat MLP.

Minimal Equinox module:

- Embed `t`
- Concatenate / fuse `[y_t, m, x_obs]` along feature axis
- Temporal backbone → score tensor same shape as `y_t`

**Done when:** forward pass on a toy batch returns `(B,T,F)` with no NaNs; parameter count logged.

---

### WP4 — Conditional DSM loss + train loop (Phase 2)

Self-supervised view (CSDI-adapted):

1. Start from clean `x0`.
2. Sample training mask `m` (random and/or block); treat masked cells as imputation targets.
3. Sample diffusion time `t ~ Uniform(ε, T)`.
4. Perturb: `y_t = mean(x0,t) + std(t) * eps`.
5. Predict score `s_θ(y_t, t, m, x_obs)`.
6. Target score from Gaussian transition, e.g. \(-\varepsilon / \sigma(t)\).
7. Reduce loss on **target (masked) positions** (optionally also lightly on observed for stability—document the choice).

Training loop:

- `jax.jit` + `jax.grad` + Optax AdamW
- Log train loss; periodic val snapshot imputation plot
- Checkpoint on best val loss
- Guardrails: NaN check, grad-norm log

**Done when:** loss decreases stably on sine (and ideally AR) for a short run (e.g. minutes on CPU/GPU).

---

### WP5 — Reverse SDE imputation (Phase 3)

Reverse-time SDE (Anderson) using learned score; for imputation:

- Start from noise at `t=T` (or noise only on missing coords; document strategy).
- Integrate with `diffrax.diffeqsolve` (Euler–Maruyama or Heun).
- **Observation replacement / conditioning:** after each step (or continuously via the conditional score), keep observed coordinates consistent with data (replacement method is fine for M1).

API target:

```python
samples, mean, quantiles = impute(
    observations, mask, timestamps,  # timestamps may be grid indices in M1
    key=..., k_samples=50,
)
```

- `samples`: `(K, T, F)` or `(B, K, T, F)`
- `mean`: posterior mean fill
- `quantiles`: e.g. 10% / 90% bands

**Done when:** held-out sine gaps look plausible visually; mean tracks the true curve better than “always zero.”

---

### WP6 — Lightweight eval + scripts (Phase 0/4 lite)

On synthetic held-out windows with known ground truth under the mask:

- MAE / RMSE on masked entries
- CRPS from `K` samples (empirical)
- Optional: 50%/90% interval coverage

`scripts/run_m1_toy.py` should:

1. Generate data  
2. Train briefly  
3. Impute  
4. Save figs under `artifacts/m1/`  
5. Print a tiny metrics table  

Not required for M1: LOCF/linear tables (those are M3).

---

## Exit criteria checklist (from plan + M1)

- [ ] Forward VP-SDE + closed-form marginals, unit-tested  
- [ ] Random + block masking, unit-tested  
- [ ] Conditional Equinox score net runs on a toy batch  
- [ ] DSM training loop jitted; loss curves saved  
- [ ] Reverse sampling via Diffrax produces `K` imputations  
- [ ] End-to-end `impute(...)` on sine/AR with masks  
- [ ] Plots: forward noising, mask patterns, imputation mean ± bands  
- [ ] Assumptions (windowing, mask types) written down  

---

## Suggested implementation order (calendar)

| Day | Focus |
|---|---|
| 1 | WP0–WP1: schema, synthetic data, masks, tests, mask plots |
| 2 | WP2: VP-SDE + forward noising plot + marginal tests |
| 3 | WP3: score net + shape smoke test |
| 4 | WP4: loss + train loop + checkpoint on sine |
| 5 | WP5–WP6: reverse impute API, bands plot, MAE/CRPS smoke metrics, `run_m1_toy.py` |

---

## Risks specific to M1 (from plan)

| Risk | M1 mitigation |
|---|---|
| Flat MLP ignores time | Start with TCN / temporal Transformer |
| Irregular time hard | Regular grid windows first; optional index encoding only |
| Training unstable | Small model, sine first, clip grads, log norms |
| Conditioning bugs | Unit-test that observed cells are not overwritten incorrectly at the end |

---

## What “done” looks like demowise

Run something like:

```bash
python scripts/run_m1_toy.py --config configs/toy_m1.yaml
pytest tests/ -q
```

You should get:

1. Decreasing train loss  
2. Artifact plots of noise / masks / imputations  
3. A checkpoint that can fill sine gaps with uncertainty bands  

That is the handoff into **M2 — Real data train**.

---

## References to keep open while coding

1. Song et al., Score-Based Generative Modeling through SDEs (VP-SDE + DSM)  
2. Tashiro et al., CSDI (conditional masking / imputation view)  
3. Diffrax docs (SDE solve / Euler–Maruyama)

---