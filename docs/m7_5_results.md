# M7.5 results — claims and M8 hand-off

Filled from the **thin** test-split claim bench
(`configs/bench_m7_5_thin.yaml` → `artifacts/m7_5/bench/`) and the
documented ablation (`artifacts/m7_5/ablation.csv`). Style matches
`docs/m3_failure_notes.md`. **v1 / M3 tables are untouched.**

**Comparison scope:** TimeFill vs LOCF / linear only. No claim vs CSDI /
BRITS / Transformer.

**Seed honesty:** claim table is **one** seed pair (`mask=0`, `sample=1`).
The planned multi-seed grid was not completed; do **not** read these
numbers as seed-averaged mean ± std. **Milestone exit criterion 3
(≥3 mask seeds, mean ± std) is not met** — WP7 is closed as thin /
single-seed claims only.

**Train:** `configs/real_m7_5.yaml` → `artifacts/m7_5/train/`
(`selected_epoch=180`, `clipped_step_fraction=0.0`, EMA on,
`select_by=val_impute_crps`, params=172550, RF=129).

## Claim checklist

| Claim | Status | Notes |
|---|---|---|
| TF MAE ≤ linear on MCAR 30% | pass (1 seed) | 26.04 vs 36.80; plan wording was seed-averaged |
| TF MAE ≤ linear on block 8 | pass (1 seed) | 41.79 vs 68.39 |
| coverage_90 in 0.85–0.95 on every protocol | pass (1 seed) | 0.874–0.912 (incl. block 16) |
| clipped-step fraction < 0.5 | pass | 0.0 (full M7.5 train) |
| CRPS better than M3 TimeFill on shared protocols | pass (1 seed) | like-for-like (both K=50 / 100 steps) |
| Seed-averaged (± std) claim table / exit #3 | fail | thin single-seed only; M7.5 not closed on this gate |
| Ablation MAE incremental WP1 / WP2 effect sizes | fail | rows 2–3 are 40×100 undertrain vs full M2 |

**Summary claim:** On the thin single-seed M7.5 bench, TimeFill beats LOCF
and linear on MAE, RMSE, and CRPS for MCAR 10/30/50%, block 8, and block 16
(extension). Coverage_90 sits in the target band. vs M3 TimeFill, CRPS and
RMSE improve on all shared protocols (MAE also drops, but see mean/median
note below). Do not advertise seed-averaged robustness or per-WP MAE
ablation deltas from rows 2–3.

## Results table (thin claim bench)

Source: `artifacts/m7_5/bench/metrics_by_rate.csv`.
`K_samples=50`, `solver_steps=100`, `point_estimate=median`, split=`test`.

| method | protocol | rate / block | mae | rmse | crps | coverage_50 | coverage_90 | n_target_cells |
|---|---|---|---:|---:|---:|---:|---:|---:|
| timefill | mcar | 0.10 | 22.96 | 97.03 | 17.12 | 0.564 | 0.885 | 40737 |
| locf | mcar | 0.10 | 42.45 | 169.50 | 42.45 | 0.177 | 0.177 | 40737 |
| linear | mcar | 0.10 | 32.46 | 142.99 | 32.46 | 0.102 | 0.102 | 40737 |
| timefill | mcar | 0.30 | 26.04 | 110.89 | 19.42 | 0.572 | 0.899 | 122172 |
| locf | mcar | 0.30 | 48.87 | 197.11 | 48.87 | 0.162 | 0.162 | 122172 |
| linear | mcar | 0.30 | 36.80 | 163.90 | 36.80 | 0.089 | 0.089 | 122172 |
| timefill | mcar | 0.50 | 32.41 | 131.93 | 24.12 | 0.572 | 0.912 | 204062 |
| locf | mcar | 0.50 | 58.07 | 228.90 | 58.07 | 0.146 | 0.146 | 204062 |
| linear | mcar | 0.50 | 44.67 | 190.71 | 44.67 | 0.075 | 0.075 | 204062 |
| timefill | block | 8 | 41.79 | 164.31 | 30.62 | 0.526 | 0.878 | 50909 |
| locf | block | 8 | 93.19 | 328.10 | 93.19 | 0.104 | 0.104 | 50909 |
| linear | block | 8 | 68.39 | 253.83 | 68.39 | 0.042 | 0.042 | 50909 |
| timefill | block | 16 | 51.81 | 193.85 | 38.18 | 0.516 | 0.874 | 101911 |
| locf | block | 16 | 117.64 | 407.03 | 117.64 | 0.085 | 0.085 | 101911 |
| linear | block | 16 | 89.02 | 320.07 | 89.02 | 0.034 | 0.034 | 101911 |

Block 16 is an **extension** row (not part of the frozen M3 protocol grid).

LOCF / linear native-unit numbers can differ slightly from M3 because baseline
edge fills use `scaler.original_unit_means()`: under log1p that is
`expm1(mean of logs)` (geometric-style), not the legacy arithmetic mean. That
makes baselines a bit stronger, so it does **not** inflate TimeFill’s win.

## vs M3 TimeFill (same seed / shared protocols)

| protocol | M3 TF mae / rmse / crps | M7.5 TF mae / rmse / crps |
|---|---:|---:|
| MCAR 10% | 35.01 / 159.68 / 26.20 | 22.96 / 97.03 / 17.12 |
| MCAR 30% | 39.01 / 165.34 / 29.19 | 26.04 / 110.89 / 19.42 |
| MCAR 50% | 46.03 / 174.57 / 34.36 | 32.41 / 131.93 / 24.12 |
| block 8 | 64.87 / 246.20 / 49.79 | 41.79 / 164.31 / 30.62 |

**Note:** M3 TimeFill MAE used posterior **mean**; M7.5 thin used
`point_estimate=median`. MAE is therefore not a pure like-for-like cell.
**RMSE** (always on the mean) and **CRPS** are the clean M3→M7.5 compares;
both improve on every shared protocol. The MAE drop is large enough that it
almost certainly holds under a mean rescore, but that rescore was not run.

## Ablation (TimeFill MAE only)

Budget: short trains `40×100` for rows 2–3; bench `K=20` / `steps=50`;
protocols MCAR 30% + block 8 only. Rows 4, 5, 7 skipped.
Source: `artifacts/m7_5/ablation.csv`.

| row | config | MCAR 30% MAE | block 8 MAE |
|---|---|---:|---:|
| 1 | M2 legacy (full M2 ckpt) | 40.05 | 66.68 |
| 2 | WP1 clean only (short) | 68.57 | 104.32 |
| 3 | WP1 + WP2 eps (short) | 53.99 | 66.42 |
| 6 | full M7.5 (full train, cheap knobs) | 27.14 | 42.78 |

Linear under the same cheap knobs: MCAR ≈37 / block ≈68. Row 6 beats
linear; rows 2–3 MAE must not be read as WP effect sizes vs full M2
(different train budgets).

**Clean WP2 finding (same 40×100 budget, rows 2→3):** score → eps cuts
`clipped_step_fraction` 0.96 → 0.26 and `median_grad_norm` 6.33 → 0.79
(`artifacts/m7_5/ablation/wp1|wp1_wp2/train_meta.json`). That matches the
WP2 intent and is not confounded by budget. The high clip rate likely
contributes to why row 2 MAE is worse than legacy M2; row 2 also had far
fewer steps and a different batch size than the full M2 train.

## M8 hand-off — gap regime → best method

Single-seed thin M7.5 table. “Best” = lowest MAE and lowest CRPS among
TimeFill / LOCF / linear.

| Gap regime | Best MAE | Best CRPS | Notes for M8 routing |
|---|---|---|---|
| Short MCAR (10%) | TimeFill | TimeFill | Point linear still strong historically (M3); M7.5 TF now wins MAE too on this seed |
| Mid MCAR (30%) | TimeFill | TimeFill | Primary M7.5 MAE target — pass (1 seed) |
| High MCAR (50%) | TimeFill | TimeFill | Prefer probabilistic fill |
| Block 8 | TimeFill | TimeFill | Clear block win vs linear / LOCF |
| Block 16 (extension) | TimeFill | TimeFill | Longer contiguous gaps — TF preferred |

**Suggested M8 default:** route short interior holes and long/block gaps to
TimeFill when an M7.5 (or better) checkpoint is available; keep LOCF /
linear only as cheap fallbacks when diffusion is unavailable.

## Shortfalls (honest)

- Exit criterion 3 unmet: no ≥3 mask-seed mean ± std table.
  Thin `metrics_summary.csv` has `n_seeds=1`; `*_std=0.0` means not
  measured (not zero variance).
- Ablation rows 2–3 cannot support per-WP **MAE** claims vs full M2.
- Still no CSDI / BRITS / Transformer baseline.
- Thin claim `K=50` / `steps=100`; ablation used cheaper knobs — do not
  mix the two tables when quoting absolute MAE.
- MAE vs M3 mixes mean (M3) and median (M7.5); prefer RMSE / CRPS.

## Reproduce

```text
python scripts/train_m2_real.py --config configs/real_m7_5.yaml
python -u scripts/run_m3_benchmark.py --config configs/bench_m7_5_thin.yaml --real-config configs/real_m7_5.yaml

python scripts/train_m2_real.py --config configs/ablation/real_wp1.yaml
python scripts/train_m2_real.py --config configs/ablation/real_wp1_wp2.yaml
python -u scripts/run_m3_benchmark.py --config configs/ablation/bench_row1_m2.yaml --real-config configs/real_m2.yaml
python -u scripts/run_m3_benchmark.py --config configs/ablation/bench_row2_wp1.yaml --real-config configs/ablation/real_wp1.yaml
python -u scripts/run_m3_benchmark.py --config configs/ablation/bench_row3_wp1_wp2.yaml --real-config configs/ablation/real_wp1_wp2.yaml
python -u scripts/run_m3_benchmark.py --config configs/ablation/bench_row6_m75.yaml --real-config configs/real_m7_5.yaml
python scripts/assemble_m7_5_ablation.py
```

Wrong `--real-config` can still run without error — pairings above are required.

Panel: `artifacts/m7_5/bench/panels/comparison_timefill_locf_linear.png`.
