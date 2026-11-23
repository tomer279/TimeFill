# Milestone 7.5 walkthrough

## WP3 — Score network v2

- `tcn_v2` lives beside unchanged `tcn_v1` (`build_score_net`, FiLM `(1+γ)`, stem-included RF).
- Done-when toy smoke **injects** a v2 model into `M1Trainer`; `run_m1_training` still builds v1 defaults (no toy `model:` section).
- `load_checkpoint_with_config` uses `build_score_net` so a v2 sidecar reloads.
- `train_meta.receptive_field` is filled from `receptive_field(config.model)`.
- `assert_wp0_consumable` still refuses non-legacy `model.*` / WP5 transform — `real_m7_5.yaml` cannot train v2 until pre-WP7 wiring.

```text
pytest tests/test_wp3_score_net_v2.py -q
```

## WP6 — Sampler efficiency and point estimate

- Diagonal diffusion via `lineax.DiagonalLinearOperator` (`sample.diffusion_impl`).
- MAE can use sample median through `masked_mae(..., point_estimate="median")`; RMSE stays on the mean.
- `write_mean_median_csvs` writes raw `mean.csv` / `median.csv` (not app export) via `scripts/wp6_timing.py` → `artifacts/m7_5/wp6/`.
- Decision: no Euler–Maruyama fallback — `lineax` is a hard dependency.
- Decision: `run_m1_toy` code unchanged; it follows `sample.diffusion_impl` from YAML when set.
- Decision: app `session_impute` stays on default dense until WP-B4; val/bench now read the flag.
- Decision: `SampleConfig.point_estimate` → bench/evaluator MAE wire deferred to WP7.
- CSVs come from an **untrained** score net (smoke only; values are not quality results).
- WP4 prerequisite: bare `impute()` recompiles every call; val loops must reuse one jitted runner (see plan WP4).

### Timing (warm CPU, one window, T=64, F=6, K=50, steps=100)

One `eqx.filter_jit` runner per mode; warm once; time the second solve with `block_until_ready()`.
Untrained net; Windows CPU. At this size the dense matvec is cheap vs the net forward, so times match.

| impl | seconds |
|---|---|
| dense | 0.726 |
| diagonal | 0.723 |

Commands:

```text
python scripts/wp6_timing.py
pytest tests/test_wp6_sampler_efficiency.py -q
pytest -q
```
