# Milestone 7.5 walkthrough

## WP7 Phase 0–1 — Train/bench wiring (pre-rebenchmark)

- Lifted `model.*` refuse; trainer builds `build_score_net(config.model)` and
  writes `model_config.json` beside best `.eqx`.
- Bench loader: seed lists → `seed_pairs` (3×3), `point_estimate` → MAE,
  sidecar load + `ScoreFromEps` / `sample.t_eps` / `observed_mode`,
  `block.extra_lens` for block-16 extension.
- Decision: `main()` still runs the first seed pair only; Cartesian loop is
  Phase 3. `replace_noised` train still refused.
- Commands:
  `pytest tests/test_wp7_wiring.py tests/test_wp4_train_hygiene.py::test_assert_wp0_allows_real_m7_5 -q`

## WP8 — Adapt honesty + M7.5 flag opt-in

- `adapt.min_adapt_steps` (dataclass default `null`); product
  `configs/adapt_m7.yaml` sets `200` so `5×20=100` → `adapt_meta.status:
  smoke_only` (still promotes). Streamlit warns once in `_show_bound_adapt_info`;
  CLI `_print_success` prints `status=` + the same warning.
- Opt-in: `configs/adapt_m7_5.yaml` (eps/clean/diagonal/strict empty-feature;
  same under-budget smoke_only unless you raise epochs/steps).
- Abort tests require no `adapt_meta.json` (not merely `status != success`).
- Deferred: window stitching, impute/bench frozen-root refuse, adapt sidecar /
  `tcn_v2`; status ignores best-val epoch (configured budget only).
- `src/data/Untitled-1.py` already absent (no-op).
- Pre-WP8 session dirs (e.g. `artifacts/m7/20261018T124132Z_775488/`) may
  still say `status: success` for a 1×1 run — leave them; they predate WP8.

```text
pytest tests/test_wp8_adapt_honesty.py tests/test_adapt_m7.py tests/test_wp_b4_adapt_guarantees.py -q
# → 25 passed
pytest -q
# → 255 passed in ~146s
```

## WP5 — Feature transform and per-feature evaluation

- `FeatureScaler`: optional `log1p_features`, `clip_nonnegative`, empty-feature
  raise vs `allow_empty_feature_fallback`, NPZ transform keys.
- A9: empty-target metrics are NaN; quick-check shows `no target cells`.
- Bench writes `per_feature.csv` and `*_norm` columns on `metrics_by_rate.*`.
- Adapt caps gain `allow_empty_feature_fallback` (legacy `true`); strict mode
  also enforces train-slice `min_observed_rate_per_feature=0.05`.

### A8 mutation check (manual)

`test_scaler_fit_ignores_perturbed_held_out` goes through
`fit_scaler_on_train_split`. To confirm it can fail: temporarily change that
helper to `FeatureScaler().fit(dataset.values, dataset.native_masks)`, re-run
the test, and expect a failure; then revert.

```text
pytest tests/test_normalize_fit_train_only.py tests/test_wp5_feature_transform.py tests/test_metrics.py tests/test_adapt_m7.py -q
```

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
