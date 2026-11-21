"""WP0 tests: M7.5 config keys parse with legacy defaults and M7.5 YAML values."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.data.adapt_config import load_adapt_m7_config
from src.data.config import load_config
from src.data.real_config import load_real_m2_config

REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_PATH = REPO_ROOT / "configs" / "toy_m1.yaml"
REAL_PATH = REPO_ROOT / "configs" / "real_m2.yaml"
ADAPT_PATH = REPO_ROOT / "configs" / "adapt_m7.yaml"
BENCH_M3_PATH = REPO_ROOT / "configs" / "bench_m3.yaml"
REAL_M75_PATH = REPO_ROOT / "configs" / "real_m7_5.yaml"
BENCH_M75_PATH = REPO_ROOT / "configs" / "bench_m7_5.yaml"


def test_toy_m1_resolves_m7_5_keys_to_legacy_defaults() -> None:
    """Existing toy config exposes §3 keys at legacy defaults."""
    config = load_config(TOY_PATH)
    protocol = config.train.protocol
    sample = config.sample

    assert protocol.parameterization == "score"
    assert protocol.observed_input == "noised"
    assert protocol.mixed_select == "per_batch"
    assert protocol.block_len_range is None
    assert protocol.max_blocks_per_feature == 1
    assert protocol.ema_decay is None
    assert protocol.lr_schedule == "constant"
    assert protocol.warmup_steps == 0
    assert protocol.select_by == "val_dsm_loss"

    assert sample.t_eps is None
    assert sample.observed_mode == "frozen_clean"
    assert sample.diffusion_impl == "dense"
    assert sample.point_estimate == "mean"


def test_real_m2_resolves_m7_5_keys_to_legacy_defaults() -> None:
    """Existing real_m2 config keeps legacy train/sample/model/data flags."""
    config = load_real_m2_config(REAL_PATH)
    protocol = config.train.protocol
    sample = config.sample
    model = config.model

    assert protocol.parameterization == "score"
    assert protocol.observed_input == "noised"
    assert protocol.obs_loss_weight == pytest.approx(0.1)
    assert protocol.t_eps == pytest.approx(1.0e-5)
    assert protocol.mixed_select == "per_batch"
    assert protocol.ema_decay is None
    assert protocol.select_by == "val_dsm_loss"

    assert sample.t_eps is None
    assert sample.observed_mode == "frozen_clean"
    assert sample.diffusion_impl == "dense"
    assert sample.point_estimate == "mean"

    assert model.arch == "tcn_v1"
    assert model.hidden_channels == 32
    assert model.time_embed_scale == pytest.approx(1.0)
    assert model.time_cond == "stem"
    assert model.norm == "none"

    assert config.data.split_mode == "entity"
    assert config.data.split_seed == 0
    assert config.data.transform.log1p_features == ()
    assert config.data.transform.clip_nonnegative is False
    assert config.data.allow_empty_feature_fallback is True
    assert config.data.batch_size == 8


def test_adapt_m7_resolves_m7_5_train_keys_to_legacy_defaults() -> None:
    """adapt_m7.yaml keeps legacy train protocol flags via shared TrainConfig."""
    config = load_adapt_m7_config(ADAPT_PATH)
    protocol = config.train.protocol

    assert protocol.parameterization == "score"
    assert protocol.observed_input == "noised"
    assert protocol.ema_decay is None
    assert protocol.select_by == "val_dsm_loss"


def test_bench_m3_seeds_remain_scalars() -> None:
    """Legacy bench_m3 seeds stay single ints (not seed lists)."""
    with BENCH_M3_PATH.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    assert raw["seeds"]["mask"] == 0
    assert raw["seeds"]["sample"] == 1
    assert isinstance(raw["seeds"]["mask"], int)
    assert isinstance(raw["seeds"]["sample"], int)


def test_real_m7_5_yaml_parses_with_m7_5_values() -> None:
    """New real_m7_5.yaml turns on the Milestone 7.5 flag set."""
    assert REAL_M75_PATH.is_file()
    config = load_real_m2_config(REAL_M75_PATH)
    protocol = config.train.protocol
    sample = config.sample
    model = config.model

    assert protocol.parameterization == "eps"
    assert protocol.observed_input == "clean"
    assert protocol.t_eps == pytest.approx(1.0e-3)
    assert protocol.obs_loss_weight == pytest.approx(0.0)
    assert protocol.mixed_select == "per_window"
    assert protocol.block_len_range == (4, 16)
    assert protocol.max_blocks_per_feature == 2
    assert protocol.ema_decay == pytest.approx(0.999)
    assert protocol.lr_schedule == "warmup_cosine"
    assert protocol.warmup_steps == 1000
    assert protocol.select_by == "val_impute_crps"
    assert config.data.batch_size == 32
    assert protocol.artifact_dir == "artifacts/m7_5/train"

    assert sample.t_eps == pytest.approx(1.0e-3)
    assert sample.observed_mode == "frozen_clean"
    assert sample.diffusion_impl == "diagonal"
    assert sample.point_estimate == "median"

    assert model.arch == "tcn_v2"
    assert model.hidden_channels == 64
    assert list(model.dilations) == [1, 2, 4, 8, 16, 32]
    assert model.time_embed_dim == 64
    assert model.time_embed_scale == pytest.approx(1000.0)
    assert model.time_cond == "per_block"
    assert model.norm == "group"

    assert list(config.data.transform.log1p_features) == [
        "PM2.5",
        "PM10",
        "SO2",
        "NO2",
        "CO",
    ]
    assert config.data.transform.clip_nonnegative is True
    assert config.data.allow_empty_feature_fallback is False


def test_bench_m7_5_yaml_points_at_m7_5_artifacts_and_seed_lists() -> None:
    """New bench_m7_5.yaml pins M7.5 checkpoint paths and multi-seed lists."""
    assert BENCH_M75_PATH.is_file()
    with BENCH_M75_PATH.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    assert raw["data"]["checkpoint"].endswith(
        "artifacts/m7_5/train/checkpoints/best_score_net.eqx"
    ) or raw["data"]["checkpoint"] == (
        "artifacts/m7_5/train/checkpoints/best_score_net.eqx"
    )
    assert raw["artifact_dir"] == "artifacts/m7_5/bench"
    assert raw["seeds"]["mask"] == [0, 1, 2]
    assert raw["seeds"]["sample"] == [1, 2, 3]


def test_invalid_parameterization_raises(tmp_path: Path) -> None:
    """Config validate rejects an unknown parameterization."""
    with TOY_PATH.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["train"]["parameterization"] = "not_a_param"
    bad_path = tmp_path / "bad_toy.yaml"
    bad_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="parameterization"):
        load_config(bad_path)
