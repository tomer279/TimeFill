"""Prepare M2 Beijing windows: raw CSVs → ``data/processed/m2/windows.npz``.

Reads ``configs/real_m2.yaml`` ``data`` knobs, loads site CSVs, builds
non-overlapping regular-grid windows, writes values / native masks /
entity ids and deterministic train/val/test indices.

Does not download data (see ``docs/EVAL.md`` fetch steps).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.real import (
    BeijingCsvLoader,
    BeijingLoadSpec,
    EntityAwareSplitter,
    RealDataPaths,
    RealWindowBuilder,
    SplitFractions,
    WindowSpec,
    default_beijing_features,
    save_processed_dataset,
)

_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "real_m2.yaml"
_MIN_ENTITIES_FOR_ENTITY_SPLIT = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for the M2 data-preparation script.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed flags.
    """
    parser = argparse.ArgumentParser(
        description="Prepare M2 Beijing windows under data/processed/m2/",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to configs/real_m2.yaml",
    )
    parser.add_argument(
        "--split-mode",
        choices=("entity", "index"),
        default="entity",
        help="entity hold-out (default) or shuffled window indices",
    )
    return parser.parse_args(argv)


def load_data_section(config_path: Path) -> Mapping[str, Any]:
    """Load the YAML ``data`` mapping from an M2 config file.

    Parameters
    ----------
    config_path : pathlib.Path
        Path to ``real_m2.yaml`` (or compatible).

    Returns
    -------
    Mapping of str to Any
        The ``data`` section.
    """
    with Path(config_path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping) or "data" not in raw:
        raise ValueError("config must be a mapping with a 'data' section")
    data = raw["data"]
    if not isinstance(data, Mapping):
        raise ValueError("config 'data' section must be a mapping")
    return data


def build_and_save(data: Mapping[str, Any], split_mode: str) -> Path:
    """Load raw CSVs, window, split, and write the processed NPZ.

    Parameters
    ----------
    data : Mapping of str to Any
        ``data`` section from ``real_m2.yaml``.
    split_mode : str
        ``entity`` or ``index``.

    Returns
    -------
    pathlib.Path
        Processed output directory.
    """
    feature_names = tuple(
        data.get("feature_names", list(default_beijing_features()))
    )
    seq_len = int(data["seq_len"])
    paths = RealDataPaths(
        raw_dir=_REPO_ROOT / str(data["raw_dir"]),
        processed_dir=_REPO_ROOT / str(data["processed_dir"]),
        raw_glob=str(data.get("raw_glob", "*.csv")),
    )
    loader = BeijingCsvLoader(BeijingLoadSpec(feature_names=feature_names))
    sites = loader.load_directory(paths)
    dataset = RealWindowBuilder(
        WindowSpec(seq_len=seq_len, stride=seq_len)
    ).build(sites)
    fractions = SplitFractions(
        train=float(data["train_fraction"]),
        val=float(data["val_fraction"]),
        test=float(data["test_fraction"]),
        seed=int(data["split_seed"]),
    )
    splitter = EntityAwareSplitter(fractions)
    n_entities = int(np.unique(np.asarray(dataset.entity_ids)).size)
    use_entity = (
        split_mode == "entity"
        and n_entities >= _MIN_ENTITIES_FOR_ENTITY_SPLIT
    )
    if use_entity:
        split = splitter.split_by_entity(dataset.entity_ids)
    else:
        split = splitter.split_by_index(dataset.shape()[0])
    out_dir = save_processed_dataset(dataset, split, paths)
    print(f"Wrote processed M2 windows to {out_dir}")
    print(f"shape={dataset.shape()} split_mode={split_mode}")
    print(
        "n_train/n_val/n_test="
        f"{split.train_idx.size}/{split.val_idx.size}/{split.test_idx.size}"
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector.

    Returns
    -------
    int
        Process exit code.
    """
    args = parse_args(argv)
    data = load_data_section(args.config)
    build_and_save(data, args.split_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
