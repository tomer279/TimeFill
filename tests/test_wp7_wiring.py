"""WP7 Phase 0–1 wiring tests (fail on pre-wiring code).

Functions
---------
test_build_m2_trainer_uses_config_model_arch
    ``_build_m2_trainer`` builds ``tcn_v2`` from ``config.model``.
test_m2_trainer_writes_model_config_sidecar
    Best-checkpoint save writes ``model_config.json`` beside the ``.eqx``.
test_bench_loader_scalar_seeds_still_parse
    Legacy scalar ``seeds.mask`` / ``seeds.sample`` still resolve.
test_bench_loader_expands_mask_sample_seed_lists
    List seeds expand to the Cartesian product of pairs.
test_aggregate_seed_metric_rows_mean_std
    Seed-grid summary reports mean ± std over pairs.
test_bench_loader_reads_point_estimate
    Bench YAML ``sample.point_estimate`` lands on the plan.
test_imputation_evaluator_honors_point_estimate
    ``ImputationEvaluator(point_estimate=...)`` drives MAE.
test_load_impute_context_wraps_eps_parameterization
    Bench load uses sidecar + ``ScoreFromEps`` + sample ``t_eps``.
test_protocol_grid_includes_extra_block_lens
    ``block.extra_lens`` adds extension block protocols after ``block_len``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest
import yaml

from src.data.real_config import load_real_m2_config
from src.data.schema import MASK_MISSING, MASK_OBSERVED
from src.eval.m3_benchmark import SeedPair, build_protocol_grid
from src.eval.metrics import EvalTensors, ImputationEvaluator, masked_mae
from src.models.score_from_eps import ScoreFromEps
from src.models.score_net import (
    ConditionalScoreNetV2,
    ScoreNetConfig,
    build_score_net,
)
from src.train.artifacts import (
    ArtifactPaths,
    save_checkpoint_with_sidecar,
)
from src.train.real_loop import _build_m2_trainer
from src.train.real_pool import RealBatchPool

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_M75_PATH = REPO_ROOT / "configs" / "real_m7_5.yaml"
REAL_M2_PATH = REPO_ROOT / "configs" / "real_m2.yaml"
BENCH_SCRIPT = REPO_ROOT / "scripts" / "run_m3_benchmark.py"

_SEED = 0
_LOOP_FEAT = 6
_LOOP_SEQ = 8
_LOOP_POOL = 4
_LOOP_BATCH = 2
_RUNNER_K = 2
_GAP_TRUTH = 0.0
_MEAN_FILL = 2.0
_MEDIAN_FILL = 10.0
_SAMPLE_T_EPS = 1.0e-3
_TRAIN_T_EPS = 1.0e-5


def test_build_m2_trainer_uses_config_model_arch(tmp_path: Path) -> None:
    """Trainer model matches ``config.model.arch`` (``tcn_v2``)."""
    config = load_real_m2_config(REAL_M75_PATH)
    config.data.batch_size = _LOOP_BATCH
    config.train.schedule.epochs = 1
    config.train.schedule.steps_per_epoch = 1
    # Keep optimizer build valid without changing model.* under test.
    config.train.protocol.extras.loss.warmup_steps = 0
    config.train.protocol.artifact_dir = str(tmp_path / "artifacts")
    paths = ArtifactPaths(tmp_path / "artifacts")
    paths.ensure()
    trainer, _key = _build_m2_trainer(config, paths, _tiny_pool())
    assert isinstance(trainer.model, ConditionalScoreNetV2)


def test_m2_trainer_writes_model_config_sidecar(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Best checkpoint directory includes ``model_config.json``."""
    monkeypatch.setattr(
        "src.train.real_loop.make_val_impute_runner",
        _stub_val_impute_runner_factory,
    )
    config = load_real_m2_config(REAL_M2_PATH)
    config.data.batch_size = _LOOP_BATCH
    config.train.schedule.epochs = 1
    config.train.schedule.steps_per_epoch = 1
    config.train.schedule.val_every = 1
    config.train.protocol.extras.loss.select_by = "val_impute_crps"
    config.train.protocol.val_impute = {
        "n_windows": 2,
        "K_samples": _RUNNER_K,
        "solver_steps": 2,
        "mask_seed": 0,
        "sample_seed": 1,
        "protocols": ["mcar_0.3"],
    }
    config.train.protocol.artifact_dir = str(tmp_path / "artifacts")
    paths = ArtifactPaths(tmp_path / "artifacts")
    paths.ensure()
    trainer, key = _build_m2_trainer(config, paths, _tiny_pool())
    trainer.run(key)
    ckpt = trainer.paths.best_checkpoint()
    assert ckpt.is_file()
    assert (ckpt.parent / "model_config.json").is_file()


def test_bench_loader_scalar_seeds_still_parse() -> None:
    """Scalar mask/sample seeds still resolve (M3 legacy)."""
    loader = _bench_module().BenchM3ConfigLoader(REPO_ROOT)
    plan = loader.resolve(_minimal_bench_raw(seeds={"mask": 0, "sample": 1}), False)
    assert plan.seeds == SeedPair(mask=0, sample=1)


def test_bench_loader_expands_mask_sample_seed_lists() -> None:
    """List seeds expand to nine ``SeedPair`` values (3×3)."""
    loader = _bench_module().BenchM3ConfigLoader(REPO_ROOT)
    plan = loader.resolve(
        _minimal_bench_raw(seeds={"mask": [0, 1, 2], "sample": [1, 2, 3]}),
        False,
    )
    pairs = plan.seed_pairs
    assert len(pairs) == 9
    assert pairs[0] == SeedPair(mask=0, sample=1)
    assert pairs[-1] == SeedPair(mask=2, sample=3)


def test_aggregate_seed_metric_rows_mean_std() -> None:
    """Summary rows average MAE over seed pairs and expose std."""
    from src.eval.m3_table import (
        MetricRecord,
        ProtocolKey,
        RunMeta,
        ScoreBlock,
        aggregate_seed_metric_rows,
    )

    protocol = ProtocolKey(name="mcar", mask_rate=0.3, block_len=None)
    rows = [
        MetricRecord(
            method="timefill",
            protocol=protocol,
            scores=ScoreBlock(
                mae=float(mae),
                rmse=1.0,
                crps=1.0,
                coverage_50=0.5,
                coverage_90=0.9,
            ),
            meta=RunMeta(
                seeds=SeedPair(mask=mask, sample=1),
                split="test",
                k_samples=2,
                point_metrics="ensemble",
            ),
            n_target_cells=10,
        )
        for mask, mae in ((0, 2.0), (1, 4.0), (2, 6.0))
    ]
    summary = aggregate_seed_metric_rows(rows)
    assert len(summary) == 1
    assert summary[0]["n_seeds"] == 3
    assert summary[0]["mae_mean"] == pytest.approx(4.0)
    assert summary[0]["mae_std"] == pytest.approx(2.0)


def test_bench_loader_reads_point_estimate() -> None:
    """``sample.point_estimate`` from YAML is exposed on the plan."""
    loader = _bench_module().BenchM3ConfigLoader(REPO_ROOT)
    raw = _minimal_bench_raw(seeds={"mask": 0, "sample": 1})
    raw["sample"]["point_estimate"] = "median"
    plan = loader.resolve(raw, False)
    assert plan.sample.point_estimate == "median"


def test_imputation_evaluator_honors_point_estimate() -> None:
    """Evaluator MAE follows ``point_estimate`` (median ≠ mean)."""
    tensors = _mean_median_differ_tensors()
    mean_mae = float(
        ImputationEvaluator(point_estimate="mean").compute(tensors).mae
    )
    median_mae = float(
        ImputationEvaluator(point_estimate="median").compute(tensors).mae
    )
    assert mean_mae == pytest.approx(
        float(masked_mae(tensors, point_estimate="mean"))
    )
    assert median_mae == pytest.approx(
        float(masked_mae(tensors, point_estimate="median"))
    )
    assert median_mae != mean_mae


def test_load_impute_context_wraps_eps_parameterization(
        tmp_path: Path,
) -> None:
    """Bench context wraps eps nets and prefers ``sample.t_eps``."""
    bench = _bench_module()
    ckpt = tmp_path / "checkpoints" / "best_score_net.eqx"
    score_cfg = ScoreNetConfig(
        n_features=_LOOP_FEAT,
        arch="tcn_v2",
        hidden_channels=8,
        n_layers=2,
        dilations=(1, 2),
        time_embed_dim=8,
        time_embed_scale=1000.0,
        time_cond="per_block",
        norm="group",
        norm_groups=2,
    )
    model = build_score_net(jax.random.PRNGKey(_SEED), score_cfg)
    save_checkpoint_with_sidecar(
        ckpt,
        model,
        {
            "score_net": score_cfg.to_dict(),
            "parameterization": "eps",
            "arch_version": 2,
        },
    )
    real_cfg_path = tmp_path / "real_eps.yaml"
    real_cfg_path.write_text(
        _real_yaml_eps_split_t_eps(),
        encoding="utf-8",
    )
    raw = _minimal_bench_raw(seeds={"mask": 0, "sample": 1})
    raw["data"]["checkpoint"] = str(ckpt)
    raw["data"]["scaler"] = str(tmp_path / "scaler.npz")
    raw["data"]["processed_dir"] = str(tmp_path / "processed")
    raw["sample"]["diffusion_impl"] = "diagonal"
    raw["sample"]["point_estimate"] = "median"
    plan = bench.BenchM3ConfigLoader(REPO_ROOT).resolve(raw, False)
    context = bench.load_impute_context(plan, real_cfg_path)
    assert isinstance(context.model, ScoreFromEps)
    assert context.settings.t_eps == pytest.approx(_SAMPLE_T_EPS)
    assert context.settings.observed_mode == "frozen_clean"
    assert context.settings.diffusion_impl == "diagonal"


def test_protocol_grid_includes_extra_block_lens() -> None:
    """``block.extra_lens`` appends extension block protocols."""
    protocols = build_protocol_grid(
        {
            "mcar_rates": [0.1],
            "block": {
                "enabled": True,
                "block_len": 8,
                "extra_lens": [16],
            },
        }
    )
    block_lens = [p.block_len for p in protocols if p.is_block()]
    assert block_lens == [8, 16]


def _bench_module() -> Any:
    """Load ``scripts/run_m3_benchmark.py`` as a module."""
    name = "run_m3_benchmark_wp7"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BENCH_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BENCH_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tiny_pool() -> RealBatchPool:
    """Return a tiny fully-observed pool for trainer construction."""
    values = jnp.zeros(
        (_LOOP_POOL, _LOOP_SEQ, _LOOP_FEAT),
        dtype=jnp.float32,
    )
    masks = jnp.ones_like(values)
    return RealBatchPool(
        train_values=values,
        train_masks=masks,
        val_values=values,
        val_masks=masks,
    )


def _stub_val_impute_runner_factory(
        *_args: Any,
        **_kwargs: Any,
) -> Any:
    """Cheap runner so the sidecar save test skips Diffrax."""

    def run(
            _model: Any,
            observations: jnp.ndarray,
            _mask: jnp.ndarray,
            _key: jax.Array,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        batch_size, seq_len, n_features = observations.shape
        mean = jnp.zeros(
            (batch_size, seq_len, n_features),
            dtype=jnp.float32,
        )
        samples = jnp.zeros(
            (batch_size, _RUNNER_K, seq_len, n_features),
            dtype=jnp.float32,
        )
        return samples, mean

    return run


def _minimal_bench_raw(*, seeds: dict[str, Any]) -> dict[str, Any]:
    """Return a minimal bench YAML mapping with the given seeds."""
    return {
        "data": {
            "processed_dir": "data/processed/m2",
            "checkpoint": "artifacts/m2/checkpoints/best_score_net.eqx",
            "scaler": "artifacts/m2/checkpoints/feature_scaler.npz",
            "split": "test",
            "n_features": 6,
            "feature_names": [
                "PM2.5",
                "PM10",
                "SO2",
                "NO2",
                "CO",
                "O3",
            ],
            "seq_len": 64,
        },
        "protocols": {
            "mcar_rates": [0.1],
            "block": {"enabled": False},
        },
        "sample": {
            "K_samples": 2,
            "solver_steps": 4,
            "diffusion_impl": "dense",
        },
        "seeds": seeds,
        "methods": ["timefill", "locf", "linear"],
        "baselines": {
            "edge_policy": "train_feature_mean",
            "point_metrics": "degenerate_k1",
            "point_k": 1,
        },
        "artifact_dir": "artifacts/m7_5/bench",
        "smoke": {
            "max_windows": 1,
            "K_samples": 1,
            "solver_steps": 2,
            "split": "val",
        },
    }


def _mean_median_differ_tensors() -> EvalTensors:
    """One missing cell where sample mean and median disagree."""
    truth = jnp.asarray(
        [[[_GAP_TRUTH, 0.0]]],
        dtype=jnp.float32,
    )
    mean = jnp.asarray(
        [[[_MEAN_FILL, 0.0]]],
        dtype=jnp.float32,
    )
    samples = jnp.asarray(
        [
            [
                [[0.0, 0.0]],
                [[_MEDIAN_FILL, 0.0]],
                [[_MEDIAN_FILL, 0.0]],
            ]
        ],
        dtype=jnp.float32,
    )
    mask = jnp.asarray(
        [[[MASK_MISSING, MASK_OBSERVED]]],
        dtype=jnp.float32,
    )
    return EvalTensors(truth=truth, mean=mean, samples=samples, mask=mask)


def _real_yaml_eps_split_t_eps() -> str:
    """Minimal real YAML: ``eps``, sample ``t_eps`` ≠ train ``t_eps``."""
    payload = {
        "data": {
            "dataset_name": "beijing_multisite_air_quality",
            "raw_dir": "data/raw/beijing_multisite",
            "processed_dir": "data/processed/m2",
            "raw_glob": "*.csv",
            "seq_len": 64,
            "stride": 64,
            "n_features": _LOOP_FEAT,
            "feature_names": [
                "PM2.5",
                "PM10",
                "SO2",
                "NO2",
                "CO",
                "O3",
            ],
            "mask_rate": 0.3,
            "block_len": 8,
            "batch_size": 2,
            "split_mode": "entity",
            "split_seed": 0,
            "train_fraction": 0.70,
            "val_fraction": 0.15,
            "test_fraction": 0.15,
        },
        "sde": {"beta_min": 0.1, "beta_max": 20.0, "T_diff": 1.0},
        "train": {
            "lr": 0.0003,
            "epochs": 1,
            "weight_decay": 0.0001,
            "seed": 0,
            "t_eps": _TRAIN_T_EPS,
            "obs_loss_weight": 0.0,
            "steps_per_epoch": 1,
            "val_every": 1,
            "log_every": 1,
            "grad_clip_norm": 1.0,
            "mask_mode": "random",
            "parameterization": "eps",
            "observed_input": "clean",
            "artifact_dir": "artifacts/m7_5/train",
        },
        "sample": {
            "K_samples": 2,
            "solver_steps": 4,
            "t_eps": _SAMPLE_T_EPS,
            "observed_mode": "frozen_clean",
            "diffusion_impl": "diagonal",
            "point_estimate": "median",
        },
    }
    return yaml.safe_dump(payload)
