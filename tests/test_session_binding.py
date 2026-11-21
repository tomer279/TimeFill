"""WP-B1 session binding tests (content digest, reset, stale results).

These encode the Milestone 7.5 Done-when scenarios. They must fail on the
pre-fix code (no content digest in the fingerprint, no reset helper, no
result-digest check) and pass after WP-B1 lands.
"""

from __future__ import annotations

from app.adapt_session_ui import (
    SESSION_ADAPT_KEY,
    SESSION_FEATURES_KEY,
    SESSION_RESULT_DIGEST_KEY,
    SESSION_RESULT_KEY,
)
from app.preview_ingest import MODE_EXACT_BEIJING, MODE_FLEXIBLE, evaluate_impute_gate
from src.data.real.specs import default_beijing_features

_FLEX = ("temp_c", "humidity_pct", "pressure_hpa")
_BYTES_A = b"temp_c,humidity_pct,pressure_hpa\n1,2,3\n"
_BYTES_B = b"temp_c,humidity_pct,pressure_hpa\n9,8,7\n"
_ADAPT_KEY = SESSION_ADAPT_KEY
_RESULT_KEY = SESSION_RESULT_KEY
_FEATURES_KEY = SESSION_FEATURES_KEY
_RESULT_DIGEST_KEY = SESSION_RESULT_DIGEST_KEY


def _binding_api():
    """Import WP-B1 binding helpers (fail until implemented).

    Returns
    -------
    tuple
        ``content_digest_from_bytes``, ``make_upload_fingerprint``,
        ``reset_session_binding``, ``impute_result_is_current``.
    """
    from app.adapt_session_ui import (
        content_digest_from_bytes,
        impute_result_is_current,
        make_upload_fingerprint,
        reset_session_binding,
    )

    return (
        content_digest_from_bytes,
        make_upload_fingerprint,
        reset_session_binding,
        impute_result_is_current,
    )


def _fingerprint(label: str, mode: str, raw: bytes) -> str:
    """Build a fingerprint the way the UI will after WP-B1.

    Parameters
    ----------
    label : str
        Source label / filename.
    mode : str
        Ingest mode.
    raw : bytes
        Uploaded or sample CSV bytes.

    Returns
    -------
    str
        Binding fingerprint.
    """
    digest_fn, make_fp, _, _ = _binding_api()
    return make_fp(label, mode, _FLEX, digest_fn(raw))


def _bound_state(digest: str) -> dict:
    """Session-state mapping with a fake adapt bind and Fill result.

    Parameters
    ----------
    digest : str
        Content digest stored with the Fill result.

    Returns
    -------
    dict
        Mutable stand-in for ``st.session_state``.
    """
    return {
        _ADAPT_KEY: object(),
        _RESULT_KEY: object(),
        _FEATURES_KEY: _FLEX,
        _RESULT_DIGEST_KEY: digest,
    }


def test_same_name_different_bytes_changes_fingerprint() -> None:
    """B1: same filename + columns + different values → different bind key."""
    first = _fingerprint("u.csv", MODE_FLEXIBLE, _BYTES_A)
    second = _fingerprint("u.csv", MODE_FLEXIBLE, _BYTES_B)
    assert first != second


def test_different_bytes_clears_bind_and_blocks_flexible_fill() -> None:
    """B1 Done-when: bind cleared; Flexible Fill blocked until re-adapt."""
    digest_fn, make_fp, reset_fn, _ = _binding_api()
    digest_a = digest_fn(_BYTES_A)
    digest_b = digest_fn(_BYTES_B)
    state = _bound_state(digest_a)
    fp_a = make_fp("u.csv", MODE_FLEXIBLE, _FLEX, digest_a)
    fp_b = make_fp("u.csv", MODE_FLEXIBLE, _FLEX, digest_b)
    assert fp_a != fp_b
    reset_fn(state)
    assert _ADAPT_KEY not in state
    assert _RESULT_KEY not in state
    assert _FEATURES_KEY not in state
    assert _RESULT_DIGEST_KEY not in state
    gate = evaluate_impute_gate(
        _FLEX,
        default_beijing_features(),
        MODE_FLEXIBLE,
        session_features=None,
    )
    assert not gate.is_allowed()


def test_success_then_failure_clears_bind() -> None:
    """B2: failed re-adapt clears the prior bind; Flexible Fill disabled."""
    digest_fn, _, reset_fn, _ = _binding_api()
    state = _bound_state(digest_fn(_BYTES_A))
    # Mirrors _run_adapt_button: clear before attempt so crashes can't stale-bind.
    reset_fn(state)
    assert _ADAPT_KEY not in state
    assert _RESULT_KEY not in state
    gate = evaluate_impute_gate(
        _FLEX,
        default_beijing_features(),
        MODE_FLEXIBLE,
        session_features=None,
    )
    assert not gate.is_allowed()


def test_reset_uses_shared_session_key_constants() -> None:
    """Session keys cleared by reset match the exported constants."""
    _, _, reset_fn, _ = _binding_api()
    state = {
        SESSION_ADAPT_KEY: object(),
        SESSION_RESULT_KEY: object(),
        SESSION_FEATURES_KEY: _FLEX,
        SESSION_RESULT_DIGEST_KEY: "abc",
    }
    reset_fn(state)
    assert state == {}


def test_identical_reupload_keeps_fingerprint() -> None:
    """Identical bytes → same fingerprint (no needless re-adapt)."""
    first = _fingerprint("u.csv", MODE_FLEXIBLE, _BYTES_A)
    second = _fingerprint("u.csv", MODE_FLEXIBLE, _BYTES_A)
    assert first == second


def test_pretrained_result_dropped_on_new_exact_file() -> None:
    """B17: Fill on sample, then another Exact file → hide charts/downloads."""
    digest_fn, _, reset_fn, result_ok = _binding_api()
    sample_digest = digest_fn(_BYTES_A)
    upload_digest = digest_fn(_BYTES_B)
    state = {
        _RESULT_KEY: object(),
        _FEATURES_KEY: default_beijing_features(),
        _RESULT_DIGEST_KEY: sample_digest,
    }
    assert not result_ok(state.get(_RESULT_DIGEST_KEY), upload_digest)
    reset_fn(state)
    assert _RESULT_KEY not in state
    assert _FEATURES_KEY not in state
    assert _RESULT_DIGEST_KEY not in state
    assert not result_ok(None, upload_digest)


def test_sample_vs_upload_drops_other_results() -> None:
    """Switching sample ↔ upload drops results bound to the other source."""
    digest_fn, _, reset_fn, result_ok = _binding_api()
    sample_fp = _fingerprint(
        "Try the sample dataset",
        MODE_EXACT_BEIJING,
        _BYTES_A,
    )
    upload_fp = _fingerprint("site.csv", MODE_EXACT_BEIJING, _BYTES_B)
    assert sample_fp != upload_fp
    sample_digest = digest_fn(_BYTES_A)
    upload_digest = digest_fn(_BYTES_B)
    state = {
        _RESULT_KEY: object(),
        _FEATURES_KEY: default_beijing_features(),
        _RESULT_DIGEST_KEY: sample_digest,
    }
    assert not result_ok(state.get(_RESULT_DIGEST_KEY), upload_digest)
    reset_fn(state)
    assert _RESULT_KEY not in state
