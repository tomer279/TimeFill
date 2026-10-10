"""WP2 tests: VP-SDE marginals, variance floor, and broadcasting."""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from src.data.config import load_config
from src.sde.vp_sde import VPSDE

REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_CONFIG_PATH = REPO_ROOT / "configs" / "toy_m1.yaml"

_SEED = 0
_T_NEAR_ZERO = 1e-5
_MEAN_TOL = 1e-4
_STD_NEAR_ZERO_TOL = 2e-3
_VAR_AT_T_TOL = 1e-2
_MEAN_AT_T_REL_TOL = 0.02
_ABS_FLOOR = 1e-6
_BATCH_SIZE = 4
_SEQ_LEN = 64
_N_FEATURES = 2


@pytest.fixture
def vp_sde() -> VPSDE:
    """Build a VPSDE from the locked toy config."""
    return VPSDE(load_config(TOY_CONFIG_PATH).sde)


@pytest.fixture
def clean_batch() -> jnp.ndarray:
    """Deterministic clean ``(B, T, F)`` tensor for marginal checks."""
    key = jax.random.PRNGKey(_SEED)
    return jax.random.normal(
        key,
        shape=(_BATCH_SIZE, _SEQ_LEN, _N_FEATURES),
        dtype=jnp.float32,
    )


def test_marginal_near_zero_time(vp_sde: VPSDE, clean_batch: jnp.ndarray) -> None:
    """As ``t → 0``, mean ≈ x0 and std ≈ 0."""
    mean, std = vp_sde.marginal_prob(clean_batch, _T_NEAR_ZERO)
    assert mean.shape == clean_batch.shape
    assert std.shape == clean_batch.shape
    assert bool(jnp.allclose(mean, clean_batch, atol=_MEAN_TOL))
    assert bool(jnp.all(std < _STD_NEAR_ZERO_TOL))


def test_marginal_variance_near_one_at_horizon(
        vp_sde: VPSDE,
        clean_batch: jnp.ndarray,
) -> None:
    """At ``t = T_diff``, VP marginal variance is ≈ 1."""
    mean, std = vp_sde.marginal_prob(clean_batch, vp_sde.T_diff)
    variance = std ** 2
    assert std.shape == clean_batch.shape
    assert bool(jnp.allclose(variance, 1.0, atol=_VAR_AT_T_TOL))
    # Mean scale is exp(-0.5 * ∫β) ≈ exp(-5) for toy betas; not exactly 0.
    rel = jnp.abs(mean) / jnp.maximum(jnp.abs(clean_batch), _ABS_FLOOR)
    assert bool(jnp.all(rel < _MEAN_AT_T_REL_TOL))


def test_marginal_broadcasts_over_btf(
        vp_sde: VPSDE,
        clean_batch: jnp.ndarray,
) -> None:
    """Scalar ``t`` broadcasts over ``(B, T, F)`` outputs."""
    mean, std = vp_sde.marginal_prob(clean_batch, 0.5 * vp_sde.T_diff)
    assert mean.shape == (_BATCH_SIZE, _SEQ_LEN, _N_FEATURES)
    assert std.shape == (_BATCH_SIZE, _SEQ_LEN, _N_FEATURES)


def test_forward_sample_matches_marginal(
        vp_sde: VPSDE,
        clean_batch: jnp.ndarray,
) -> None:
    """``forward_sample`` equals ``mean + std * eps``."""
    time = 0.3 * vp_sde.T_diff
    eps = jax.random.normal(
        jax.random.PRNGKey(_SEED + 1),
        shape=clean_batch.shape,
        dtype=jnp.float32,
    )
    mean, std = vp_sde.marginal_prob(clean_batch, time)
    sample = vp_sde.forward_sample(clean_batch, time, eps)
    assert bool(jnp.allclose(sample, mean + std * eps))


def test_sde_coefficients_and_reverse_drift_shapes(
        vp_sde: VPSDE,
        clean_batch: jnp.ndarray,
) -> None:
    """Coefficient helpers and reverse-drift stub are shape-safe."""
    time = 0.7 * vp_sde.T_diff
    drift_coeff, diffusion = vp_sde.sde_coefficients(time)
    assert drift_coeff.shape == ()
    assert diffusion.shape == ()
    assert float(drift_coeff) == pytest.approx(-0.5 * float(vp_sde.beta(time)))
    assert float(diffusion) == pytest.approx(float(jnp.sqrt(vp_sde.beta(time))))

    score = jnp.zeros_like(clean_batch)
    rev = vp_sde.reverse_drift(clean_batch, time, score)
    assert rev.shape == clean_batch.shape
