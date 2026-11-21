"""WP2 tests: eps parameterization, ScoreFromEps, and loss stability.

Functions
---------
test_eps_loss_zero_model_near_one_at_t_eps
    Zero-output model at ``t=t_eps``: ``eps`` loss ≈ 1, ``score`` ≫ 1.
test_score_from_eps_matches_neg_eps_over_std
    ``ScoreFromEps`` equals ``-eps_hat / std(t)`` on random inputs.
test_context_from_toy_wraps_eps_and_resolves_sample_t_eps
    ``parameterization=eps`` wraps the model; ``sample.t_eps`` wins.
test_eps_wrapped_impute_smoke_finite_and_freezes_observed
    Done-when: wrapped eps model through Diffrax; observed cells unchanged.
test_adapt_refuses_non_score_parameterization
    Adapt settings factory blocks ``eps`` until WP-B4 wraps Fill.
test_eps_train_loss_cv_lower_than_score
    Short toy train: last-100-step loss CV is lower for ``eps`` than ``score``.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from src.data.adapt_config import load_adapt_m7_config
from src.data.config import load_config
from src.data.schema import MASK_OBSERVED
from src.data.synthetic import generate_sine_windows
from src.models.score_from_eps import ScoreFromEps
from src.models.score_net import (
    ConditionalScoreNet,
    default_score_net_config,
)
from src.sample.reverse_sde import (
    ImputeQuery,
    ReverseSDEImputer,
    context_from_toy_config,
    grid_timestamps,
)
from src.sde.vp_sde import VPSDE
from src.train.adapt_loop import dsm_settings_from_adapt
from src.train.artifacts import ArtifactPaths, stub_val_imputation_hook
from src.train.loop import M1Trainer, TrainerBundle, build_optimizer
from src.train.loss import ConditionalDSMLoss, dsm_settings_from_config

REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_CONFIG_PATH = REPO_ROOT / "configs" / "toy_m1.yaml"
ADAPT_CONFIG_PATH = REPO_ROOT / "configs" / "adapt_m7.yaml"

_SEED = 0
_T_EPS = 1.0e-3
_EPS_LOSS_ATOL = 0.15
_SCORE_LOSS_MIN = 10.0
_MASK_RATE_ALL = 1.0
_CV_STEPS = 200
_CV_TAIL = 100
# History rows are per-epoch means; 1 step/epoch → one loss sample per step.
_CV_EPOCHS = 200
_CV_STEPS_PER_EPOCH = 1
_BATCH = 4
_SEQ = 32
_FEAT = 2
_STD_FLOOR = 1.0e-5
_SMOKE_STEPS = 2
_SMOKE_K = 1
_MASK_TOL = 0.5


class _ZeroScore:
    """Score / eps stub that always returns zeros.

    Public methods
    --------------
    __call__
        Return a zero tensor matching ``y_t``.
    clear
        No-op for spy API parity with WP1 stubs.
    """

    def __call__(
            self,
            y_t: jnp.ndarray,
            time: jnp.ndarray | float,
            mask: jnp.ndarray,
            x_obs: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return zeros shaped like ``y_t``."""
        return jnp.zeros_like(y_t)

    def clear(self) -> None:
        """No-op placeholder."""
        return None


class _FixedEpsModel:
    """Stub whose output is a fixed ``eps_hat`` tensor.

    Public methods
    --------------
    __call__
        Ignore inputs and return the stored ``eps_hat``.
    set_eps_hat
        Replace the stored output tensor.
    """

    def __init__(self, eps_hat: jnp.ndarray) -> None:
        """Store the fixed network output.

        Parameters
        ----------
        eps_hat : jax.Array
            Constant model output.
        """
        self.eps_hat = eps_hat

    def __call__(
            self,
            y_t: jnp.ndarray,
            time: jnp.ndarray | float,
            mask: jnp.ndarray,
            x_obs: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return the stored ``eps_hat``."""
        return self.eps_hat

    def set_eps_hat(self, eps_hat: jnp.ndarray) -> None:
        """Replace the fixed output.

        Parameters
        ----------
        eps_hat : jax.Array
            New constant model output.
        """
        self.eps_hat = eps_hat


def _force_time(loss: ConditionalDSMLoss, time_value: float) -> None:
    """Pin diffusion-time sampling to a constant.

    Parameters
    ----------
    loss : ConditionalDSMLoss
        Loss instance whose ``_sample_time`` is replaced.
    time_value : float
        Constant diffusion time for every batch element.
    """

    def _fixed_time(
            key: jax.Array,
            batch_size: int,
    ) -> jnp.ndarray:
        del key
        return jnp.full((batch_size,), time_value, dtype=jnp.float32)

    loss._sample_time = _fixed_time  # type: ignore[method-assign]


def _loss_with_parameterization(
        parameterization: str,
        t_eps: float = _T_EPS,
) -> ConditionalDSMLoss:
    """Build a toy DSM loss with an overridden parameterization.

    Parameters
    ----------
    parameterization : str
        ``score`` or ``eps``.
    t_eps : float
        Train-time floor.

    Returns
    -------
    ConditionalDSMLoss
        Loss using the toy SDE and forced ``mask_rate=1``.
    """
    config = load_config(TOY_CONFIG_PATH)
    config.train.protocol.extras.loss.parameterization = parameterization
    config.train.protocol.t_eps = t_eps
    config.data.mask_rate = _MASK_RATE_ALL
    settings = dsm_settings_from_config(config)
    assert settings.parameterization == parameterization
    loss = ConditionalDSMLoss(VPSDE(config.sde), settings)
    _force_time(loss, t_eps)
    return loss


def test_eps_loss_zero_model_near_one_at_t_eps() -> None:
    """At ``t=t_eps``, zero-model ``eps`` loss ≈ 1; ``score`` loss ≫ 1."""
    shape = (_BATCH, _SEQ, _FEAT)
    x0 = generate_sine_windows(jax.random.PRNGKey(_SEED), shape)
    zero_model = _ZeroScore()
    keys = jax.random.split(jax.random.PRNGKey(_SEED + 1), 8)

    eps_loss = _loss_with_parameterization("eps")
    eps_vals = [
        float(eps_loss(zero_model, x0, key)) for key in keys
    ]
    eps_mean = float(np.mean(eps_vals))
    assert eps_mean == pytest.approx(1.0, abs=_EPS_LOSS_ATOL)

    score_loss = _loss_with_parameterization("score")
    score_vals = [
        float(score_loss(zero_model, x0, key)) for key in keys
    ]
    score_mean = float(np.mean(score_vals))
    assert score_mean > _SCORE_LOSS_MIN
    assert score_mean > eps_mean * _SCORE_LOSS_MIN


def test_score_from_eps_matches_neg_eps_over_std() -> None:
    """``ScoreFromEps`` output matches ``-eps_hat / std(t)`` exactly."""
    config = load_config(TOY_CONFIG_PATH)
    sde = VPSDE(config.sde)
    shape = (_BATCH, _SEQ, _FEAT)
    key = jax.random.PRNGKey(_SEED)
    key_eps, key_y, key_m, key_x = jax.random.split(key, 4)
    eps_hat = jax.random.normal(key_eps, shape, dtype=jnp.float32)
    y_t = jax.random.normal(key_y, shape, dtype=jnp.float32)
    mask = (jax.random.uniform(key_m, shape) > 0.3).astype(jnp.float32)
    x_obs = jax.random.normal(key_x, shape, dtype=jnp.float32) * mask
    time = jnp.full((_BATCH,), 0.37, dtype=jnp.float32)

    wrapped = ScoreFromEps(model=_FixedEpsModel(eps_hat), sde=sde)
    got = wrapped(y_t, time, mask, x_obs)
    _, std = sde.marginal_prob(y_t, time)
    std_safe = jnp.maximum(std, _STD_FLOOR)
    expected = -eps_hat / std_safe
    assert got.shape == expected.shape
    assert bool(jnp.allclose(got, expected, rtol=0.0, atol=1e-6))


def _train_loss_tail(
        parameterization: str,
        tmp_path: Path,
        seed: int,
) -> np.ndarray:
    """Run a short toy train and return the last ``_CV_TAIL`` train losses.

    Parameters
    ----------
    parameterization : str
        ``score`` or ``eps``.
    tmp_path : pathlib.Path
        Artifact root for this run.
    seed : int
        Train PRNG seed.

    Returns
    -------
    numpy.ndarray
        1-D array of the last ``_CV_TAIL`` logged ``train_loss`` values.
    """
    with TOY_CONFIG_PATH.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["train"]["parameterization"] = parameterization
    raw["train"]["t_eps"] = _T_EPS
    raw["train"]["epochs"] = _CV_EPOCHS
    raw["train"]["steps_per_epoch"] = _CV_STEPS_PER_EPOCH
    raw["train"]["val_every"] = _CV_EPOCHS + 1
    raw["train"]["log_every"] = 1
    raw["train"]["seed"] = seed
    raw["train"]["artifact_dir"] = str(tmp_path / parameterization)
    raw["data"]["batch_size"] = _BATCH
    raw["data"]["seq_len"] = _SEQ
    cfg_path = tmp_path / f"toy_{parameterization}.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(cfg_path)
    key_model, key_train = jax.random.split(jax.random.PRNGKey(seed))
    model = ConditionalScoreNet(
        key_model,
        default_score_net_config(config.data.n_features),
    )
    loss_fn = ConditionalDSMLoss(
        VPSDE(config.sde),
        dsm_settings_from_config(config),
    )
    assert loss_fn.settings.parameterization == parameterization
    bundle = TrainerBundle(
        config=config,
        model=model,
        loss_fn=loss_fn,
        optimizer=build_optimizer(config.train),
        paths=ArtifactPaths(tmp_path / parameterization),
        val_hook=stub_val_imputation_hook,
    )
    _, history = M1Trainer(bundle.with_paths()).run(key_train)
    losses = np.asarray(
        [float(row["train_loss"]) for row in history.rows],
        dtype=np.float64,
    )
    assert losses.size >= _CV_TAIL
    return losses[-_CV_TAIL:]


def _coeff_of_variation(values: np.ndarray) -> float:
    """Return ``std / |mean|`` for a 1-D loss series.

    Parameters
    ----------
    values : numpy.ndarray
        Loss samples.

    Returns
    -------
    float
        Coefficient of variation.
    """
    mean = float(np.mean(values))
    std = float(np.std(values))
    return std / max(abs(mean), 1e-8)


def test_context_from_toy_wraps_eps_and_resolves_sample_t_eps() -> None:
    """Toy context wraps eps nets and prefers ``sample.t_eps`` when set."""
    config = load_config(TOY_CONFIG_PATH)
    model = ConditionalScoreNet(
        jax.random.PRNGKey(_SEED),
        default_score_net_config(config.data.n_features),
    )
    legacy = context_from_toy_config(model, config)
    assert legacy.model is model
    assert legacy.settings.t_eps == pytest.approx(config.train.protocol.t_eps)

    config.train.protocol.extras.loss.parameterization = "eps"
    config.sample.t_eps = _T_EPS
    wrapped = context_from_toy_config(model, config)
    assert isinstance(wrapped.model, ScoreFromEps)
    assert wrapped.model.model is model
    assert wrapped.settings.t_eps == pytest.approx(_T_EPS)


def test_eps_wrapped_impute_smoke_finite_and_freezes_observed() -> None:
    """Wrapped eps model imputes; mean is finite; observed cells stay put."""
    config = load_config(TOY_CONFIG_PATH)
    config.train.protocol.extras.loss.parameterization = "eps"
    config.sample.t_eps = _T_EPS
    config.sample.solver_steps = _SMOKE_STEPS
    config.sample.k_samples = _SMOKE_K
    model = ConditionalScoreNet(
        jax.random.PRNGKey(_SEED),
        default_score_net_config(config.data.n_features),
    )
    context = context_from_toy_config(model, config)
    assert isinstance(context.model, ScoreFromEps)

    seq_len = int(config.data.seq_len)
    gap_lo = seq_len // 4
    gap_hi = seq_len // 2
    shape = (
        config.data.batch_size,
        seq_len,
        config.data.n_features,
    )
    x0 = generate_sine_windows(jax.random.PRNGKey(_SEED + 1), shape)
    x_obs = x0.at[:, gap_lo:gap_hi, :].set(0.0)
    mask = jnp.ones_like(x0)
    mask = mask.at[:, gap_lo:gap_hi, :].set(0.0)
    _samples, mean, _quantiles = ReverseSDEImputer(context).impute(
        ImputeQuery(
            observations=x_obs,
            mask=mask,
            timestamps=grid_timestamps(seq_len),
            key=jax.random.PRNGKey(_SEED + 2),
            k_samples=_SMOKE_K,
        )
    )
    assert bool(jnp.isfinite(mean).all())
    observed = jnp.abs(mask - MASK_OBSERVED) < _MASK_TOL
    assert bool(jnp.allclose(mean[observed], x_obs[observed], atol=1e-5))


def test_adapt_refuses_non_score_parameterization() -> None:
    """Adapt factory refuses ``parameterization=eps`` until WP-B4."""
    config = load_adapt_m7_config(ADAPT_CONFIG_PATH)
    config.train.protocol.extras.loss.parameterization = "eps"
    with pytest.raises(NotImplementedError, match="parameterization"):
        dsm_settings_from_adapt(config)


def test_eps_train_loss_cv_lower_than_score(tmp_path: Path) -> None:
    """``eps`` short-train loss CV (last 100 steps) is lower than ``score``."""
    assert _CV_EPOCHS * _CV_STEPS_PER_EPOCH == _CV_STEPS
    eps_tail = _train_loss_tail("eps", tmp_path, seed=_SEED)
    score_tail = _train_loss_tail("score", tmp_path, seed=_SEED)
    eps_cv = _coeff_of_variation(eps_tail)
    score_cv = _coeff_of_variation(score_tail)
    # Loose guard: eps should not be noisier than score (allow tiny tie slack).
    assert eps_cv <= score_cv * 1.05
