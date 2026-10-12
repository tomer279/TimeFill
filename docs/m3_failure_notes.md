# M3 failure notes — TimeFill vs LOCF / linear

Filled from a **test**-split run of
`python scripts/run_m3_benchmark.py --config configs/bench_m3.yaml`.
Source numbers: `artifacts/m3/metrics_by_rate.csv` (and `.json`).
Optional local copy: `artifacts/m3/failure_notes.md`.

**Comparison scope:** TimeFill vs LOCF vs linear only. Stronger baseline
(CSDI / BRITS / Transformer) is **deferred for v1** — do not claim a win
against those methods.

**Metric cells:** `m_native == 1` and `m_artificial == 0`, original units.
Point baselines: degenerate `K = 1` (CRPS ≈ absolute error; interval width 0).

## Claim checklist

| Claim | Status (`pass` / `fail` / `pending`) | Notes |
|---|---|---|
| CRPS better than LOCF and linear (per protocol where applicable) | pass | TimeFill best CRPS on all four protocols |
| Coverage_50 / coverage_90 better or better-calibrated than point baselines | pass | TimeFill ~0.43–0.60 @50%, ~0.81–0.90 @90%; point baselines are degenerate (`K=1`, width 0) |
| MAE competitive with LOCF / linear | pass (with shortfall) | Beats LOCF everywhere; beats linear on block; near-tie at MCAR 50%; slightly behind linear at MCAR 10% / 30% |

**Summary claim:** TimeFill is the better probabilistic imputer vs LOCF / linear
(CRPS + coverage). Point MAE is competitive and wins on long/block gaps; it
does not dominate linear on easy short-gap MCAR.

## Results table (from metrics_by_rate)

| method | protocol | mask_rate / block_len | mae | rmse | crps | coverage_50 | coverage_90 | n_target_cells |
|---|---|---|---:|---:|---:|---:|---:|---:|
| timefill | mcar | 0.10 | 35.01 | 159.68 | 26.20 | 0.596 | 0.901 | 40737 |
| locf | mcar | 0.10 | 42.71 | 168.70 | 42.71 | 0.177 | 0.177 | 40737 |
| linear | mcar | 0.10 | 33.02 | 141.35 | 33.02 | 0.120 | 0.120 | 40737 |
| timefill | mcar | 0.30 | 39.01 | 165.34 | 29.19 | 0.578 | 0.901 | 122172 |
| locf | mcar | 0.30 | 49.18 | 196.10 | 49.18 | 0.162 | 0.162 | 122172 |
| linear | mcar | 0.30 | 37.45 | 161.23 | 37.45 | 0.103 | 0.103 | 122172 |
| timefill | mcar | 0.50 | 46.03 | 174.57 | 34.36 | 0.540 | 0.892 | 204062 |
| locf | mcar | 0.50 | 58.61 | 228.00 | 58.61 | 0.146 | 0.146 | 204062 |
| linear | mcar | 0.50 | 45.76 | 188.03 | 45.76 | 0.086 | 0.086 | 204062 |
| timefill | block | 8 | 64.87 | 246.20 | 49.79 | 0.430 | 0.811 | 50909 |
| locf | block | 8 | 93.57 | 328.29 | 93.57 | 0.104 | 0.104 | 50909 |
| linear | block | 8 | 68.46 | 249.40 | 68.46 | 0.045 | 0.045 | 50909 |

Seeds (from config / artifact rows): `mask=0`, `sample=1`. Split: `test`.
TimeFill `K_samples=50`; LOCF / linear `K=1` (`degenerate_k1`).

## Where TimeFill won

1. **Long contiguous blocks** — MAE 64.87 vs linear 68.46 vs LOCF 93.57;
   CRPS 49.79 vs 68.46 / 93.57; coverage_90 0.811 vs ~0.04–0.10.
2. **High MCAR (50%)** — CRPS 34.36 vs linear 45.76 / LOCF 58.61; MAE
   nearly tied with linear (46.03 vs 45.76); RMSE better than both.
3. **Calibration** — coverage_50 / coverage_90 stay near nominal bands;
   point baselines cannot produce intervals, so their coverage collapses
   to a hit-rate of a zero-width fill.
4. **All protocols vs LOCF** — lower MAE, RMSE, and CRPS everywhere.

## Where TimeFill lost or tied

1. **MCAR 10% MAE / RMSE** — linear MAE 33.02 vs TimeFill 35.01
   (RMSE 141.35 vs 159.68). Short gaps favor linear; TimeFill still wins
   CRPS (26.20 vs 33.02).
2. **MCAR 30% MAE / RMSE** — linear MAE 37.45 vs TimeFill 39.01
   (RMSE 161.23 vs 165.34). Again CRPS favors TimeFill (29.19 vs 37.45).
3. **MAE-only reading** — an MAE-only table would understate the M3 win;
   CRPS / coverage are the intended differentiators.

## Shortfalls (honest)

- MAE still ~2 points behind linear at MCAR 10% and MCAR 30% (easy short gaps).
- TimeFill RMSE remains higher than linear at MCAR 10% / 30% (heavier
  tails / occasional large errors); gap closes or reverses at MCAR 50% and
  block.
- No comparison vs CSDI / BRITS / Transformer (deferred for v1).

## Panels

Qualitative panel: `artifacts/m3/panels/comparison_timefill_locf_linear.png`
(MCAR rate=0.1 example window). TimeFill mean tracks peaks where LOCF /
linear flatten; held-out targets generally fall inside the TimeFill band.
Supports the CRPS / coverage story even when aggregate MAE still slightly
trails linear on easy MCAR.
