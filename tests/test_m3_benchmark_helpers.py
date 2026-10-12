"""Hermetic tests for M3 WP2 table assembly and mask helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from src.data.schema import MASK_MISSING
from src.eval.m3_benchmark import (
    MetricRecord,
    MetricsTableAssembler,
    ProtocolComposeRequest,
    ProtocolKey,
    RunMeta,
    ScoreBlock,
    SeedPair,
    SplitWindowBundle,
    build_protocol_grid,
    compose_protocol_masks,
    count_target_cells,
    point_fill_as_ensemble,
)

_SEED = 0
_BATCH = 2
_SEQ = 8
_FEAT = 3
_RATE = 0.5
_BLOCK = 2
_NATIVE_GAP_T = 1


def _bundle() -> SplitWindowBundle:
    """Tiny normalized windows with one native-gap time."""
    values = np.ones((_BATCH, _SEQ, _FEAT), dtype=np.float32)
    masks = np.ones((_BATCH, _SEQ, _FEAT), dtype=np.float32)
    masks[:, _NATIVE_GAP_T, :] = MASK_MISSING
    values[:, _NATIVE_GAP_T, :] = 0.0
    return SplitWindowBundle(
        values_n=jnp.asarray(values),
        m_native=jnp.asarray(masks),
        split_name="test",
    )


def test_point_fill_as_ensemble_rank() -> None:
    """Point fills become ``(B, 1, T, F)``."""
    filled = jnp.zeros((_BATCH, _SEQ, _FEAT), dtype=jnp.float32)
    samples = point_fill_as_ensemble(filled)
    assert samples.shape == (_BATCH, 1, _SEQ, _FEAT)


def test_build_protocol_grid_mcar_and_block() -> None:
    """Protocol grid expands rates then optional block."""
    protocols = build_protocol_grid(
        {
            "mcar_rates": [0.1, 0.3, 0.5],
            "block": {"enabled": True, "block_len": 8},
        }
    )
    assert len(protocols) == 4
    assert protocols[0].name == "mcar"
    assert protocols[0].mask_rate == 0.1
    assert protocols[-1].is_block()
    assert protocols[-1].block_len == 8


def test_count_target_cells_matches_composer() -> None:
    """Target cells are natively observed and artificially missing."""
    bundle = _bundle()
    protocol = ProtocolKey(name="mcar", mask_rate=_RATE, block_len=None)
    masks = compose_protocol_masks(
        ProtocolComposeRequest(
            protocol=protocol,
            windows=bundle,
            key=jax.random.PRNGKey(_SEED),
        )
    )
    n_cells = count_target_cells(masks.m_native, masks.m_artificial)
    assert n_cells == masks.n_target_cells()
    assert n_cells > 0
    native_gap = bundle.m_native[:, _NATIVE_GAP_T, :]
    assert bool(jnp.all(jnp.abs(native_gap - MASK_MISSING) < 1e-6))


def test_m_eval_is_native_times_artificial() -> None:
    """``m_eval`` equals elementwise ``m_native * m_artificial``."""
    bundle = _bundle()
    protocol = ProtocolKey(name="block", mask_rate=None, block_len=_BLOCK)
    masks = compose_protocol_masks(
        ProtocolComposeRequest(
            protocol=protocol,
            windows=bundle,
            key=jax.random.PRNGKey(_SEED),
        )
    )
    composed = bundle.m_native * masks.m_artificial
    assert bool(jnp.allclose(masks.m_eval, composed))


def test_metrics_table_assembler_csv_json(tmp_path) -> None:
    """Assembler writes long-form CSV and JSON with expected columns."""
    record = MetricRecord(
        method="locf",
        protocol=ProtocolKey(name="mcar", mask_rate=0.1, block_len=None),
        scores=ScoreBlock(
            mae=1.0,
            rmse=2.0,
            crps=1.0,
            coverage_50=0.0,
            coverage_90=0.0,
        ),
        meta=RunMeta(
            seeds=SeedPair(mask=0, sample=1),
            split="test",
            k_samples=50,
            point_metrics="degenerate_k1",
        ),
        n_target_cells=10,
    )
    assembler = MetricsTableAssembler()
    assembler.append(record)
    csv_path = assembler.write_csv(tmp_path / "metrics_by_rate.csv")
    json_path = assembler.write_json(tmp_path / "metrics_by_rate.json")
    text = csv_path.read_text(encoding="utf-8")
    assert "method,protocol,mask_rate" in text
    assert "locf" in text
    assert "degenerate_k1" in text
    assert json_path.is_file()
    flattened = record.as_dict()
    assert flattened["k_samples"] == 1
    assert flattened["point_metrics"] == "degenerate_k1"
    assert len(assembler.rows()) == 1


def test_truncate_max_windows() -> None:
    """Smoke ``max_windows`` keeps a leading subset."""
    bundle = _bundle().truncate(1)
    assert bundle.n_windows() == 1


def test_metric_record_flattens_nested_fields() -> None:
    """Composed ``MetricRecord`` flattens to the long-form schema."""
    record = MetricRecord(
        method="timefill",
        protocol=ProtocolKey(name="block", mask_rate=None, block_len=8),
        scores=ScoreBlock(
            mae=0.5,
            rmse=0.7,
            crps=0.4,
            coverage_50=0.5,
            coverage_90=0.9,
        ),
        meta=RunMeta(
            seeds=SeedPair(mask=0, sample=1),
            split="test",
            k_samples=50,
            point_metrics="ensemble",
        ),
        n_target_cells=42,
    )
    payload = record.as_dict()
    assert payload["protocol"] == "block"
    assert payload["block_len"] == 8
    assert payload["mask_rate"] is None
    assert payload["mask_seed"] == 0
    assert payload["sample_seed"] == 1
    assert payload["k_samples"] == 50
    assert payload["n_target_cells"] == 42
