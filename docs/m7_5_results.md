# M7.5 results — claims and M8 hand-off

Filled from the **3×3 seed-grid** claim bench
(`configs/bench_m7_5.yaml` → `artifacts/m7_5/bench_grid/`) and the
documented ablation (`artifacts/m7_5/ablation.csv`). Style matches
`docs/m3_failure_notes.md`. **v1 / M3 tables are untouched.**

**Comparison scope:** TimeFill vs LOCF / linear only. No claim vs CSDI /
BRITS / Transformer.

**Seed honesty:** claim table is seed-averaged over **9** pairs
(`mask ∈ {0,1,2}` × `sample ∈ {1,2,3}`), reported as mean ± std.
**Milestone exit criterion 3 is met.** An earlier thin single-seed run
remains under `artifacts/m7_5/bench/` (not the claim table).

**Train:** `configs/real_m7_5.yaml` → `artifacts/m7_5/train/`
(`selected_epoch=180`, `clipped_step_fraction=0.0`, EMA on,
`select_by=val_impute_crps`, params=172550, RF=129).

## Claim checklist

| Claim | Status | Notes |
|---|---|---|
| TF MAE ≤ linear on MCAR 30% | pass | 26.58±0.41 vs 37.15±0.43 (n=9) |
| TF MAE ≤ linear on block 8 | pass | 41.22±0.98 vs 66.54±1.45 |
| coverage_90 in 0.85–0.95 on every protocol | pass | 0.875–0.911 (incl. block 16) |
| clipped-step fraction < 0.5 | pass | 0.0 (full M7.5 train) |
| CRPS better than M3 TimeFill on shared protocols | pass | seed-mean CRPS; like-for-like K=50 / 100 steps |
| Seed-averaged (± std) claim table / exit #3 | pass | `bench_grid/`, n_seeds=9 |
| Ablation MAE incremental WP1 / WP2 effect sizes | fail | rows 2–3 are 40×100 undertrain vs full M2 |

**Summary claim:** On the seed-averaged M7.5 grid, TimeFill beats LOCF and
linear on MAE, RMSE, and CRPS for MCAR 10/30/50%, block 8, and block 16
(extension). Coverage_90 sits in the target band. vs M3 TimeFill, CRPS and
RMSE improve on all shared protocols (MAE also drops, but see mean/median
note below). Do not advertise per-WP MAE ablation deltas from rows 2–3.

## Results table (seed-averaged claim bench)

Source: `artifacts/m7_5/bench_grid/metrics_summary.csv`.
`K_samples=50`, `solver_steps=100`, `point_estimate=median`, split=`test`,
`n_seeds=9`. Cells are mean ± std.

| method | protocol | rate / block | mae | rmse | crps | coverage_90 |
|---|---|---|---:|---:|---:|---:|
| timefill | mcar | 0.10 | 22.86±0.14 | 98.71±2.43 | 17.08±0.06 | 0.884±0.001 |
| locf | mcar | 0.10 | 42.47±0.76 | 174.90±4.86 | 42.47±0.76 | 0.175±0.002 |
| linear | mcar | 0.10 | 32.03±0.41 | 150.53±9.02 | 32.03±0.41 | 0.099±0.002 |
| timefill | mcar | 0.30 | 26.58±0.41 | 116.03±3.71 | 19.83±0.29 | 0.898±0.001 |
| locf | mcar | 0.30 | 49.32±0.58 | 203.21±4.84 | 49.32±0.58 | 0.162±0.000 |
| linear | mcar | 0.30 | 37.15±0.43 | 168.88±3.79 | 37.15±0.43 | 0.089±0.001 |
| timefill | mcar | 0.50 | 33.03±0.47 | 139.05±6.76 | 24.59±0.39 | 0.911±0.001 |
| locf | mcar | 0.50 | 58.58±0.44 | 235.46±4.98 | 58.58±0.44 | 0.146±0.000 |
| linear | mcar | 0.50 | 45.12±0.37 | 197.55±6.76 | 45.12±0.37 | 0.075±0.000 |
| timefill | block | 8 | 41.22±0.98 | 167.27±7.86 | 30.34±0.64 | 0.876±0.002 |
| locf | block | 8 | 92.35±1.45 | 339.03±10.10 | 92.35±1.45 | 0.103±0.001 |
| linear | block | 8 | 66.54±1.45 | 245.77±6.10 | 66.54±1.45 | 0.043±0.001 |
| timefill | block | 16 | 50.85±0.89 | 185.71±6.81 | 37.25±0.81 | 0.875±0.001 |
| locf | block | 16 | 115.16±1.86 | 393.99±10.32 | 115.16±1.86 | 0.086±0.002 |
| linear | block | 16 | 86.83±1.65 | 304.35±13.07 | 86.83±1.65 | 0.035±0.001 |

Block 16 is an **extension** row (not part of the frozen M3 protocol grid).

LOCF / linear native-unit numbers can differ slightly from M3 because baseline
edge fills use `scaler.original_unit_means()`: under log1p that is
`expm1(mean of logs)` (geometric-style), not the legacy arithmetic mean. That
makes baselines a bit stronger, so it does **not** inflate TimeFill’s win.

## vs M3 TimeFill (shared protocols)

M3 column is the frozen single-seed M3 table. M7.5 column is seed-mean
from `bench_grid/` (n=9).

| protocol | M3 TF mae / rmse / crps | M7.5 TF mae / rmse / crps (mean) |
|---|---:|---:|
| MCAR 10% | 35.01 / 159.68 / 26.20 | 22.86 / 98.71 / 17.08 |
| MCAR 30% | 39.01 / 165.34 / 29.19 | 26.58 / 116.03 / 19.83 |
| MCAR 50% | 46.03 / 174.57 / 34.36 | 33.03 / 139.05 / 24.59 |
| block 8 | 64.87 / 246.20 / 49.79 | 41.22 / 167.27 / 30.34 |

**Note:** M3 TimeFill MAE used posterior **mean**; M7.5 used
`point_estimate=median`. MAE is therefore not a pure like-for-like cell.
**RMSE** (always on the mean) and **CRPS** are the clean M3→M7.5 compares;
both improve on every shared protocol.

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

Seed-averaged M7.5 grid. “Best” = lowest MAE and lowest CRPS among
TimeFill / LOCF / linear. Margin = TimeFill MAE mean ÷ linear MAE mean.

| Gap regime | Best MAE | Best CRPS | TF/linear MAE | Notes for M8 routing |
|---|---|---|---:|---|
| Short MCAR (10%) | TimeFill | TimeFill | 0.71 | TF wins seed-averaged MAE too |
| Mid MCAR (30%) | TimeFill | TimeFill | 0.72 | Primary MAE target — pass |
| High MCAR (50%) | TimeFill | TimeFill | 0.73 | Prefer probabilistic fill |
| Block 8 | TimeFill | TimeFill | 0.62 | Clear block win |
| Block 16 (extension) | TimeFill | TimeFill | 0.59 | Longer contiguous gaps — TF preferred |

**Suggested M8 default:** route short interior holes and long/block gaps to
TimeFill when an M7.5 (or better) checkpoint is available; keep LOCF /
linear only as cheap fallbacks when diffusion is unavailable.

## Shortfalls (honest)

- Ablation rows 2–3 cannot support per-WP **MAE** claims vs full M2.
- Still no CSDI / BRITS / Transformer baseline.
- Ablation used cheaper knobs (`K=20` / `steps=50`) — do not mix with the
  claim grid when quoting absolute MAE.
- MAE vs M3 mixes mean (M3) and median (M7.5); prefer RMSE / CRPS.
- Thin single-seed folder (`artifacts/m7_5/bench/`) is historical only.

## Reproduce

```text
python scripts/train_m2_real.py --config configs/real_m7_5.yaml
python -u scripts/run_m3_benchmark.py --config configs/bench_m7_5.yaml --real-config configs/real_m7_5.yaml
# → artifacts/m7_5/bench_grid/  (~33 h on the CPU used here)

# optional thin smoke (does not overwrite bench_grid/):
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

Panel: `artifacts/m7_5/bench_grid/panels/comparison_timefill_locf_linear.png`.
