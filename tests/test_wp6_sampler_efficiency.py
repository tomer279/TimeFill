"""WP6 tests: diagonal diffusion and median point estimate for MAE.

Functions
---------
test_impute_settings_default_diffusion_impl_is_dense
    Legacy default keeps ``diffusion_impl="dense"``.
test_dense_vs_diagonal_samples_match
    Same key / window: ``dense`` and ``diagonal`` samples agree (atol).
test_mae_uses_median_when_configured
    With ``point_estimate=median``, MAE uses the sample median fill.
test_rmse_keeps_mean_when_mae_uses_median
    RMSE still uses the posterior mean when MAE uses the median.
test_context_wires_diffusion_impl_from_sample_config
    ``context_from_toy_config`` copies ``sample.diffusion_impl``.
test_assert_wp0_allows_diagonal_and_median_sample_flags
    Real train guard accepts only those two sample flags when else legacy.
test_write_mean_median_csvs_round_trip
    ``write_mean_median_csvs`` writes both files and reloads matching values.
test_write_mean_median_csvs_rank3_round_trip
    Rank-3 ``(B, T, F)`` dumps reload as ``(B·T, F)``.
test_impute_settings_rejects_unknown_diffusion_impl
    Invalid ``diffusion_impl`` raises in ``validate``.
test_masked_mae_rejects_unknown_point_estimate
    Invalid ``point_estimate`` raises.
test_diagonal_diffusion_returns_lineax_operator
    ``diagonal`` path returns ``lineax.DiagonalLinearOperator``.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import lineax
import numpy as np
import pytest
import yaml

from src.data.config import load_config
from src.data.real_config import load_real_m2_config
from src.data.schema import MASK_MISSING
from src.data.synthetic import generate_sine_windows
from src.eval.metrics import (
    EvalTensors,
    masked_mae,
    masked_rmse,
    write_mean_median_csvs,
)
from src.models.score_net import (
    ConditionalScoreNet,
    default_score_net_config,
)
from src.sample.reverse_sde import (
    ImputeContext,
    ImputeQuery,
    ImputeSolverSettings,
    ReverseSDEImputer,
    _make_diffusion_fn,
    context_from_toy_config,
    grid_timestamps,
)
from src.sde.vp_sde import VPSDE
from src.train.real_loop import _assert_wp0_consumable

REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_CONFIG_PATH = REPO_ROOT / "configs" / "toy_m1.yaml"
REAL_CONFIG_PATH = REPO_ROOT / "configs" / "real_m2.yaml"

_SEED = 0
_SEQ = 8
_FEAT = 2
_SOLVER_STEPS = 4
_K_SAMPLES = 2
_SAMPLE_ATOL = 1.0e-5
_T_EPS = 1.0e-3
_BATCH = 1
_K_METRIC = 4
_METRIC_SEQ = 1
_METRIC_FEAT = 1
_GAP_TRUTH = 0.0
_SAMPLE_VALUES = (0.0, 0.0, 0.0, 10.0)
_MEAN_FILL = 2.5
_MEDIAN_FILL = 0.0


def test_impute_settings_default_diffusion_impl_is_dense() -> None:
    """Unset ``diffusion_impl`` resolves to the legacy dense path."""
    settings = ImputeSolverSettings(
        solver_steps=_SOLVER_STEPS,
        t_eps=_T_EPS,
    )
    assert settings.diffusion_impl == "dense"


def test_dense_vs_diagonal_samples_match() -> None:
    """Dense ``jnp.diag`` and diagonal operator paths match on one window."""
    toy_cfg = load_config(TOY_CONFIG_PATH)
    key = jax.random.PRNGKey(_SEED)
    key_data, key_model, key_impute = jax.random.split(key, 3)
    observations, mask = _one_window_with_gap(key_data)
    model = ConditionalScoreNet(
        key_model,
        default_score_net_config(_FEAT),
    )
    sde = VPSDE(toy_cfg.sde)
    dense = _impute_samples(
        model,
        sde,
        observations,
        mask,
        key_impute,
        "dense",
    )
    diagonal = _impute_samples(
        model,
        sde,
        observations,
        mask,
        key_impute,
        "diagonal",
    )
    assert dense.shape == diagonal.shape
    assert jnp.allclose(dense, diagonal, atol=_SAMPLE_ATOL)


def test_mae_uses_median_when_configured() -> None:
    """Configured median MAE matches |median − truth|, not |mean − truth|."""
    tensors = _mean_median_differ_tensors()
    mae = float(masked_mae(tensors, point_estimate="median"))
    expected = abs(_MEDIAN_FILL - _GAP_TRUTH)
    assert mae == pytest.approx(expected)
    mean_mae = float(masked_mae(tensors, point_estimate="mean"))
    assert mean_mae == pytest.approx(abs(_MEAN_FILL - _GAP_TRUTH))
    assert mae != mean_mae


def test_rmse_keeps_mean_when_mae_uses_median() -> None:
    """RMSE stays on the posterior mean even when MAE uses the median."""
    tensors = _mean_median_differ_tensors()
    rmse = float(masked_rmse(tensors))
    expected = abs(_MEAN_FILL - _GAP_TRUTH)
    assert rmse == pytest.approx(expected)


def test_context_wires_diffusion_impl_from_sample_config(
        tmp_path: Path,
) -> None:
    """Toy context settings carry ``sample.diffusion_impl`` from YAML."""
    cfg_path = _toy_yaml_with_diffusion(tmp_path, "diagonal")
    config = load_config(cfg_path)
    key = jax.random.PRNGKey(_SEED)
    model = ConditionalScoreNet(
        key,
        default_score_net_config(config.data.n_features),
    )
    context = context_from_toy_config(model, config)
    assert context.settings.diffusion_impl == "diagonal"


def test_assert_wp0_allows_diagonal_and_median_sample_flags() -> None:
    """Train guard no longer refuses WP6 sample flags alone."""
    config = load_real_m2_config(REAL_CONFIG_PATH)
    config.sample.diffusion_impl = "diagonal"
    config.sample.point_estimate = "median"
    config.validate()
    _assert_wp0_consumable(config)


def test_write_mean_median_csvs_round_trip(tmp_path: Path) -> None:
    """Both CSVs exist and reload to the written arrays."""
    mean = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    median = jnp.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=jnp.float32)
    mean_path, median_path = write_mean_median_csvs(tmp_path, mean, median)
    assert mean_path.is_file()
    assert median_path.is_file()
    assert np.allclose(np.loadtxt(mean_path, delimiter=","), np.asarray(mean))
    assert np.allclose(
        np.loadtxt(median_path, delimiter=","),
        np.asarray(median),
    )


def test_write_mean_median_csvs_rank3_round_trip(tmp_path: Path) -> None:
    """Rank-3 arrays flatten to ``(B·T, F)`` rows."""
    mean = jnp.arange(8, dtype=jnp.float32).reshape((2, 2, 2))
    median = mean + 1.0
    mean_path, _median_path = write_mean_median_csvs(tmp_path, mean, median)
    loaded = np.loadtxt(mean_path, delimiter=",")
    assert loaded.shape == (4, 2)
    assert np.allclose(loaded, np.asarray(mean).reshape(4, 2))


def test_impute_settings_rejects_unknown_diffusion_impl() -> None:
    """Unknown ``diffusion_impl`` fails validation."""
    settings = ImputeSolverSettings(
        solver_steps=_SOLVER_STEPS,
        t_eps=_T_EPS,
        diffusion_impl="foo",
    )
    with pytest.raises(ValueError, match="diffusion_impl"):
        settings.validate()


def test_masked_mae_rejects_unknown_point_estimate() -> None:
    """Unknown ``point_estimate`` raises."""
    tensors = _mean_median_differ_tensors()
    with pytest.raises(ValueError, match="point_estimate"):
        masked_mae(tensors, point_estimate="foo")


def test_diagonal_diffusion_returns_lineax_operator() -> None:
    """Diagonal impl returns a lineax diagonal operator, not a dense matrix."""
    toy_cfg = load_config(TOY_CONFIG_PATH)
    mask = jnp.ones((_SEQ, _FEAT), dtype=jnp.float32)
    mask = mask.at[_SEQ // 2, :].set(MASK_MISSING)
    diffusion = _make_diffusion_fn(VPSDE(toy_cfg.sde), mask, "diagonal")
    y_flat = jnp.zeros((_SEQ * _FEAT,), dtype=jnp.float32)
    vf = diffusion(0.0, y_flat, None)
    assert isinstance(vf, lineax.DiagonalLinearOperator)


def _one_window_with_gap(key: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a single ``(T, F)`` observation / mask with a mid-gap.

    Parameters
    ----------
    key : jax.Array
        PRNG key for the sine window.

    Returns
    -------
    observations : jax.Array
        Values ``(T, F)`` with the gap zeroed.
    mask : jax.Array
        ``1`` observed / ``0`` missing, shape ``(T, F)``.
    """
    windows = generate_sine_windows(
        key,
        (_BATCH, _SEQ, _FEAT),
        noise_std=0.0,
    )[0]
    mask = jnp.ones((_SEQ, _FEAT), dtype=jnp.float32)
    gap_t = _SEQ // 2
    mask = mask.at[gap_t, :].set(MASK_MISSING)
    observations = windows * mask
    return observations, mask


def _impute_samples(
        model: ConditionalScoreNet,
        sde: VPSDE,
        observations: jnp.ndarray,
        mask: jnp.ndarray,
        key: jax.Array,
        diffusion_impl: str,
) -> jnp.ndarray:
    """Draw ``K`` imputations under one diffusion implementation.

    Parameters
    ----------
    model : ConditionalScoreNet
        Score network (shared across dense / diagonal).
    sde : VPSDE
        Shared SDE coefficients.
    observations : jax.Array
        Window values ``(T, F)``.
    mask : jax.Array
        Observation mask ``(T, F)``.
    key : jax.Array
        Sampler PRNG key (identical for both paths).
    diffusion_impl : str
        ``dense`` or ``diagonal``.

    Returns
    -------
    jax.Array
        Samples ``(K, T, F)``.
    """
    settings = ImputeSolverSettings(
        solver_steps=_SOLVER_STEPS,
        t_eps=_T_EPS,
        diffusion_impl=diffusion_impl,
    )
    imputer = ReverseSDEImputer(
        ImputeContext(model=model, sde=sde, settings=settings)
    )
    samples, _mean, _quantiles = imputer.impute(
        ImputeQuery(
            observations=observations,
            mask=mask,
            timestamps=grid_timestamps(_SEQ),
            key=key,
            k_samples=_K_SAMPLES,
        )
    )
    return samples


def _mean_median_differ_tensors() -> EvalTensors:
    """Build tensors where posterior mean and median disagree on the gap.

    Returns
    -------
    EvalTensors
        One missing cell; sample stack mean ``2.5``, median ``0``.
    """
    truth = jnp.full(
        (_BATCH, _METRIC_SEQ, _METRIC_FEAT),
        _GAP_TRUTH,
        dtype=jnp.float32,
    )
    mask = jnp.full(
        (_BATCH, _METRIC_SEQ, _METRIC_FEAT),
        MASK_MISSING,
        dtype=jnp.float32,
    )
    mean = jnp.full(
        (_BATCH, _METRIC_SEQ, _METRIC_FEAT),
        _MEAN_FILL,
        dtype=jnp.float32,
    )
    sample_vec = jnp.asarray(_SAMPLE_VALUES, dtype=jnp.float32)
    samples = jnp.broadcast_to(
        sample_vec.reshape((_BATCH, _K_METRIC, _METRIC_SEQ, _METRIC_FEAT)),
        (_BATCH, _K_METRIC, _METRIC_SEQ, _METRIC_FEAT),
    )
    return EvalTensors(truth=truth, mean=mean, samples=samples, mask=mask)


def _toy_yaml_with_diffusion(tmp_path: Path, diffusion_impl: str) -> Path:
    """Write a toy config copy with an overridden sample diffusion impl.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temp directory.
    diffusion_impl : str
        Value for ``sample.diffusion_impl``.

    Returns
    -------
    pathlib.Path
        Path to the written YAML file.
    """
    raw = yaml.safe_load(TOY_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["sample"]["diffusion_impl"] = diffusion_impl
    out = tmp_path / "toy_wp6.yaml"
    out.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return out
