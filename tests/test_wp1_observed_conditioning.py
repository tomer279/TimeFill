"""WP1 tests: train/sample observed-cell conditioning consistency.

Functions
---------
test_clean_observed_input_y_t_equals_x_obs_on_observed
    Spy: with ``observed_input=clean``, model ``y_t`` matches ``x_obs`` on
    ``m==1`` (toy DSM).
test_clean_observed_input_leaves_missing_cells_noised
    Spy: ``m==0`` cells keep the forward-sample (noised) values (toy DSM).
test_clean_observed_input_real_loss_matches_x_obs_on_observed
    Same clean-input spy check for ``RealConditionalDSMLoss``.
test_replace_noised_observed_cells_empirical_std
    ``observed_mode=replace_noised``: model-input observed cells have
    empirical std close to ``std(t)`` over many draws.
test_pairing_clean_frozen_toy_step_and_impute
    Done-when: ``clean`` + ``frozen_clean`` one DSM step + impute on toy.
test_pairing_noised_replace_noised_toy_step_and_impute
    Done-when: ``noised`` + ``replace_noised`` one DSM step + impute on toy.
test_clean_requires_zero_obs_loss_weight
    ``observed_input=clean`` rejects nonzero ``obs_loss_weight``.
test_settings_factories_preserve_observed_input
    Factories pass through ``observed_input=clean`` (not only the default).
test_assert_wp0_consumable_rejects_clean_with_obs_weight
    Bad clean+weight fails in ``_assert_wp0_consumable`` before prepare.
test_clean_observed_input_real_native_gap_stays_noised
    Real DSM: natively missing cells keep forward-noised ``y_t``.
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
from src.data.real_config import load_real_m2_config
from src.data.schema import MASK_OBSERVED
from src.data.synthetic import generate_sine_windows
from src.models.score_net import (
    ConditionalScoreNet,
    default_score_net_config,
)
from src.sample.reverse_sde import (
    ImputeContext,
    ImputeQuery,
    ImputeSolverSettings,
    ReverseSDEImputer,
    grid_timestamps,
)
from src.sde.vp_sde import VPSDE
from src.train.adapt_loop import dsm_settings_from_adapt
from src.train.loss import (
    ConditionalDSMLoss,
    DSMLossSettings,
    DsmLossExtras,
    dsm_settings_from_config,
)
from src.train.real_loop import _assert_wp0_consumable
from src.train.real_loss import (
    RealConditionalDSMLoss,
    RealDSMBatch,
    dsm_settings_from_real_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_CONFIG_PATH = REPO_ROOT / "configs" / "toy_m1.yaml"
REAL_CONFIG_PATH = REPO_ROOT / "configs" / "real_m2.yaml"
REAL_M75_CONFIG_PATH = REPO_ROOT / "configs" / "real_m7_5.yaml"
ADAPT_CONFIG_PATH = REPO_ROOT / "configs" / "adapt_m7.yaml"

_SEED = 0
_BATCH = 2
_SEQ = 16
_FEAT = 2
_MASK_RATE = 0.3
_BLOCK_LEN = 4
_T_EPS = 1.0e-3
_ATOL = 1e-5
_MASK_TOL = 0.5
_REPLACE_DRAWS = 256
_REPLACE_TIME = 0.5
_STD_RTOL = 0.25
_MEAN_ATOL = 0.15
_SOLVER_STEPS = 8
_K_SAMPLES = 2
_NATIVE_GAP_T = _SEQ // 2


class _SpyScore:
    """Record the ``y_t`` tensor passed into the score call.

    Public methods
    --------------
    __call__
        Store ``y_t`` / ``mask`` / ``x_obs`` and return zeros.
    clear
        Drop the last recorded tensors.
    """

    def __init__(self) -> None:
        """Initialize with empty capture slots."""
        self.y_t: jnp.ndarray | None = None
        self.mask: jnp.ndarray | None = None
        self.x_obs: jnp.ndarray | None = None

    def __call__(
            self,
            y_t: jnp.ndarray,
            time: jnp.ndarray | float,
            mask: jnp.ndarray,
            x_obs: jnp.ndarray,
    ) -> jnp.ndarray:
        """Capture inputs and return a zero score.

        Parameters
        ----------
        y_t : jax.Array
            Noisy (or clean-conditioned) state.
        time : float or jax.Array
            Diffusion time (unused).
        mask : jax.Array
            Observation mask.
        x_obs : jax.Array
            Sentinel-filled observations.

        Returns
        -------
        jax.Array
            Zeros matching ``y_t``.
        """
        del time
        self.y_t = y_t
        self.mask = mask
        self.x_obs = x_obs
        return jnp.zeros_like(y_t)

    def clear(self) -> None:
        """Reset captured tensors to ``None``."""
        self.y_t = None
        self.mask = None
        self.x_obs = None


def _toy_sde() -> VPSDE:
    """Build a VP-SDE from the locked toy config."""
    return VPSDE(load_config(TOY_CONFIG_PATH).sde)


def _clean_settings(observed_input: str) -> DSMLossSettings:
    """Build DSM settings with an explicit ``observed_input`` mode.

    Parameters
    ----------
    observed_input : str
        ``clean`` or ``noised``.

    Returns
    -------
    DSMLossSettings
        Toy-scale settings including ``observed_input``.
    """
    return DSMLossSettings(
        mask_rate=_MASK_RATE,
        block_len=_BLOCK_LEN,
        t_eps=_T_EPS,
        obs_loss_weight=0.0,
        mask_mode="random",
        extras=DsmLossExtras(observed_input=observed_input),
    )


def _sine_batch() -> jnp.ndarray:
    """Return a small clean sine batch ``(B, T, F)``."""
    shape = (_BATCH, _SEQ, _FEAT)
    return generate_sine_windows(jax.random.PRNGKey(_SEED), shape)


def test_clean_observed_input_y_t_equals_x_obs_on_observed() -> None:
    """With ``observed_input=clean``, model ``y_t`` equals ``x_obs`` on ``m==1``."""
    loss_fn = ConditionalDSMLoss(_toy_sde(), _clean_settings("clean"))
    spy = _SpyScore()
    loss_fn(spy, _sine_batch(), jax.random.PRNGKey(1))
    assert spy.y_t is not None and spy.mask is not None and spy.x_obs is not None
    observed = jnp.abs(spy.mask - MASK_OBSERVED) < _MASK_TOL
    assert bool(jnp.any(observed))
    assert bool(
        jnp.allclose(
            spy.y_t[observed],
            spy.x_obs[observed],
            atol=_ATOL,
        )
    )


def test_clean_observed_input_leaves_missing_cells_noised() -> None:
    """With ``observed_input=clean``, ``m==0`` cells stay forward-noised."""
    sde = _toy_sde()
    settings = _clean_settings("clean")
    loss_fn = ConditionalDSMLoss(sde, settings)
    x0 = _sine_batch()
    key = jax.random.PRNGKey(2)
    spy = _SpyScore()
    loss_fn(spy, x0, key)

    assert spy.y_t is not None and spy.mask is not None
    # Replay the same key stream the loss uses (mask, time, eps).
    key_mask, key_time, key_eps = jax.random.split(key, 3)
    _x_obs, mask = loss_fn.sample_mask(key_mask, x0)
    time = loss_fn._sample_time(key_time, int(x0.shape[0]))
    eps = jax.random.normal(key_eps, x0.shape, dtype=jnp.float32)
    y_noised = sde.forward_sample(x0, time, eps)
    missing = jnp.abs(mask - MASK_OBSERVED) >= _MASK_TOL
    assert bool(jnp.any(missing))
    assert bool(
        jnp.allclose(
            spy.y_t[missing],
            y_noised[missing],
            atol=_ATOL,
        )
    )


def test_clean_observed_input_real_loss_matches_x_obs_on_observed() -> None:
    """Real DSM with ``observed_input=clean`` matches ``x_obs`` on ``m_train==1``."""
    rng = np.random.default_rng(_SEED)
    values = rng.normal(size=(_BATCH, _SEQ, _FEAT)).astype(np.float32)
    m_native = np.ones((_BATCH, _SEQ, _FEAT), dtype=np.float32)
    m_native[:, _SEQ // 2, :] = 0.0
    values[:, _SEQ // 2, :] = 0.0
    batch = RealDSMBatch(
        x0=jnp.asarray(values),
        m_native=jnp.asarray(m_native),
    )
    loss_fn = RealConditionalDSMLoss(_toy_sde(), _clean_settings("clean"))
    spy = _SpyScore()
    loss_fn(spy, batch, jax.random.PRNGKey(3))
    assert spy.y_t is not None and spy.mask is not None and spy.x_obs is not None
    observed = jnp.abs(spy.mask - MASK_OBSERVED) < _MASK_TOL
    assert bool(jnp.any(observed))
    assert bool(
        jnp.allclose(
            spy.y_t[observed],
            spy.x_obs[observed],
            atol=_ATOL,
        )
    )


def test_replace_noised_observed_cells_empirical_std() -> None:
    """``replace_noised`` model inputs on observed cells match ``std(t)``.

    Draws many independent replacements at a fixed diffusion time via the
    sampler's public ``observed_mode`` path (stub model records ``y_t``).
    """
    sde = _toy_sde()
    settings = ImputeSolverSettings(
        solver_steps=1,
        t_eps=_T_EPS,
        observed_mode="replace_noised",
    )
    settings.validate()
    x0 = generate_sine_windows(
        jax.random.PRNGKey(_SEED),
        (1, _SEQ, _FEAT),
        noise_std=0.0,
    )[0]
    # Half observed so the std estimate is over a stable mask.
    mask = jnp.ones((_SEQ, _FEAT), dtype=jnp.float32)
    mask = mask.at[_SEQ // 2 :, :].set(0.0)
    x_obs = jnp.where(mask > _MASK_TOL, x0, 0.0)
    observed = jnp.abs(mask - MASK_OBSERVED) < _MASK_TOL
    assert bool(jnp.any(observed))

    recorded: list[np.ndarray] = []

    class _RecordScore:
        """Host-side capture of model ``y_t`` (single-window, no vmap)."""

        def __call__(self, y_t, time, mask_b, x_obs_b):
            del time, mask_b, x_obs_b
            recorded.append(np.asarray(y_t[0]))
            return jnp.zeros_like(y_t)

    # Probe one reverse step's model input many times with fresh keys.
    # Implementation must feed replace_noised inputs into the score model.
    from src.sample import reverse_sde as rev

    pack_state = x_obs  # frozen_clean init; replace_noised rewrites before score
    for draw in range(_REPLACE_DRAWS):
        key = jax.random.PRNGKey(_SEED + 100 + draw)
        pack = rev._ReversePack(pack_state, x_obs, mask)
        drift = rev._make_drift_fn(
            _RecordScore(),
            sde,
            pack,
            settings,
            key,
        )
        # tau = T_diff - t  =>  t = _REPLACE_TIME  when tau = T_diff - t
        tau = float(sde.t_diff - _REPLACE_TIME)
        y_flat = pack_state.reshape((-1,))
        drift(tau, y_flat, None)

    assert recorded, "expected score calls that capture y_t"
    stack = np.stack(recorded, axis=0)
    emp_std = stack[:, observed].std(axis=0).mean()
    emp_mean = stack[:, observed].mean(axis=0)
    mean, std = sde.marginal_prob(x_obs[None], jnp.asarray(_REPLACE_TIME))
    target_std = float(np.asarray(std)[0][observed].mean())
    target_mean = np.asarray(mean)[0][observed]
    assert emp_std == pytest.approx(target_std, rel=_STD_RTOL)
    assert bool(np.allclose(emp_mean, target_mean, atol=_MEAN_ATOL))


def test_pairing_clean_frozen_toy_step_and_impute() -> None:
    """Done-when: ``clean`` + ``frozen_clean`` DSM step and impute stay finite."""
    sde = _toy_sde()
    loss_fn = ConditionalDSMLoss(sde, _clean_settings("clean"))
    x0 = _sine_batch()
    model = ConditionalScoreNet(
        jax.random.PRNGKey(_SEED),
        default_score_net_config(_FEAT),
    )
    loss = loss_fn(model, x0, jax.random.PRNGKey(4))
    assert loss.shape == ()
    assert bool(jnp.isfinite(loss))

    settings = ImputeSolverSettings(
        solver_steps=_SOLVER_STEPS,
        t_eps=_T_EPS,
        observed_mode="frozen_clean",
    )
    imputer = ReverseSDEImputer(ImputeContext(model, sde, settings))
    x_obs = x0.at[:, _SEQ // 4: _SEQ // 2, :].set(0.0)
    mask = jnp.ones_like(x0)
    mask = mask.at[:, _SEQ // 4: _SEQ // 2, :].set(0.0)
    samples, mean, _quantiles = imputer.impute(
        ImputeQuery(
            observations=x_obs,
            mask=mask,
            timestamps=grid_timestamps(_SEQ),
            key=jax.random.PRNGKey(5),
            k_samples=_K_SAMPLES,
        )
    )
    assert samples.shape[0] == _BATCH
    assert bool(jnp.isfinite(mean).all())


def test_pairing_noised_replace_noised_toy_step_and_impute() -> None:
    """Done-when: ``noised`` + ``replace_noised`` DSM step and impute stay finite."""
    sde = _toy_sde()
    loss_fn = ConditionalDSMLoss(sde, _clean_settings("noised"))
    x0 = _sine_batch()
    model = ConditionalScoreNet(
        jax.random.PRNGKey(_SEED + 1),
        default_score_net_config(_FEAT),
    )
    loss = loss_fn(model, x0, jax.random.PRNGKey(6))
    assert loss.shape == ()
    assert bool(jnp.isfinite(loss))

    settings = ImputeSolverSettings(
        solver_steps=_SOLVER_STEPS,
        t_eps=_T_EPS,
        observed_mode="replace_noised",
    )
    imputer = ReverseSDEImputer(ImputeContext(model, sde, settings))
    x_obs = x0.at[:, _SEQ // 4: _SEQ // 2, :].set(0.0)
    mask = jnp.ones_like(x0)
    mask = mask.at[:, _SEQ // 4: _SEQ // 2, :].set(0.0)
    samples, mean, _quantiles = imputer.impute(
        ImputeQuery(
            observations=x_obs,
            mask=mask,
            timestamps=grid_timestamps(_SEQ),
            key=jax.random.PRNGKey(7),
            k_samples=_K_SAMPLES,
        )
    )
    assert samples.shape[0] == _BATCH
    assert bool(jnp.isfinite(mean).all())


def test_clean_requires_zero_obs_loss_weight() -> None:
    """``observed_input=clean`` rejects nonzero observed-cell loss weight."""
    with pytest.raises(ValueError, match="obs_loss_weight"):
        DSMLossSettings(
            mask_rate=_MASK_RATE,
            block_len=_BLOCK_LEN,
            t_eps=_T_EPS,
            obs_loss_weight=0.1,
            mask_mode="random",
            extras=DsmLossExtras(observed_input="clean"),
        ).validate()


def _yaml_with_observed_clean(src: Path, dest: Path) -> Path:
    """Copy a YAML config with ``train.observed_input: clean``.

    Parameters
    ----------
    src : pathlib.Path
        Source YAML path.
    dest : pathlib.Path
        Destination path for the rewritten YAML.

    Returns
    -------
    pathlib.Path
        ``dest``.
    """
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    raw["train"]["observed_input"] = "clean"
    dest.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return dest


def test_settings_factories_preserve_observed_input(
        tmp_path: Path,
) -> None:
    """Factories pass through ``observed_input=clean`` (not just the default)."""
    toy_path = _yaml_with_observed_clean(
        TOY_CONFIG_PATH,
        tmp_path / "toy_clean.yaml",
    )
    assert dsm_settings_from_config(load_config(toy_path)).observed_input == (
        "clean"
    )

    real = load_real_m2_config(REAL_M75_CONFIG_PATH)
    assert dsm_settings_from_real_config(real).observed_input == "clean"

    adapt_path = _yaml_with_observed_clean(
        ADAPT_CONFIG_PATH,
        tmp_path / "adapt_clean.yaml",
    )
    assert dsm_settings_from_adapt(
        load_adapt_m7_config(adapt_path)
    ).observed_input == "clean"

    clean = _clean_settings("clean")
    loss_fn = RealConditionalDSMLoss(_toy_sde(), clean)
    assert loss_fn.with_mask_rate(0.5).settings.observed_input == "clean"


def test_assert_wp0_consumable_rejects_clean_with_obs_weight() -> None:
    """Bad clean+weight fails in ``_assert_wp0_consumable`` before prepare."""
    config = load_real_m2_config(REAL_CONFIG_PATH)
    config.train.protocol.extras.loss.observed_input = "clean"
    with pytest.raises(ValueError, match="obs_loss_weight"):
        _assert_wp0_consumable(config)


def test_clean_observed_input_real_native_gap_stays_noised() -> None:
    """Under ``clean``, natively missing cells keep forward-sample noise."""
    sde = _toy_sde()
    loss_fn = RealConditionalDSMLoss(sde, _clean_settings("clean"))
    rng = np.random.default_rng(_SEED)
    values = rng.normal(size=(_BATCH, _SEQ, _FEAT)).astype(np.float32)
    m_native = np.ones((_BATCH, _SEQ, _FEAT), dtype=np.float32)
    m_native[:, _NATIVE_GAP_T, :] = 0.0
    values[:, _NATIVE_GAP_T, :] = 0.0
    batch = RealDSMBatch(
        x0=jnp.asarray(values),
        m_native=jnp.asarray(m_native),
    )
    key = jax.random.PRNGKey(8)
    spy = _SpyScore()
    loss_fn(spy, batch, key)

    assert spy.y_t is not None
    key_mask, key_time, key_eps = jax.random.split(key, 3)
    _x_obs, m_train, _m_art = loss_fn.sample_mask(key_mask, batch)
    time = loss_fn._sample_time(key_time, batch.batch_size())
    eps = jax.random.normal(key_eps, batch.x0.shape, dtype=jnp.float32)
    y_noised = sde.forward_sample(batch.x0, time, eps)
    native_missing = jnp.abs(batch.m_native - MASK_OBSERVED) >= _MASK_TOL
    assert bool(jnp.all(m_train[native_missing] < _MASK_TOL))
    assert bool(
        jnp.allclose(
            spy.y_t[native_missing],
            y_noised[native_missing],
            atol=_ATOL,
        )
    )
