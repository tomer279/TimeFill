"""Long-form M3 metrics row types and CSV / JSON assembly.

Classes
-------
ProtocolKey
    MCAR-rate or block protocol identity for table rows.
SeedPair
    Artificial-mask and TimeFill sampler seeds.
ScoreBlock
    MAE / RMSE / CRPS / coverage scalars.
RunMeta
    Split, ``K``, point-metric policy, and seeds for one row.
MetricRecord
    Composed long-form row (flatten only at CSV / JSON write).
MetricsTableAssembler
    Collect records and write ``metrics_by_rate`` artifacts.

Functions
---------
format_csv_cell
    Format one CSV cell (``None`` → empty string).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from src.eval.metrics import MetricsTable

_POINT_K = 1
_POINT_METHODS = frozenset({"locf", "linear"})
_CSV_FIELDS = (
    "method",
    "protocol",
    "mask_rate",
    "block_len",
    "mae",
    "rmse",
    "crps",
    "coverage_50",
    "coverage_90",
    "mask_seed",
    "sample_seed",
    "n_target_cells",
    "point_metrics",
    "split",
    "k_samples",
)


@dataclass(frozen=True)
class ProtocolKey:
    """One frozen M3 protocol (MCAR rate or contiguous block).

    Public methods
    --------------
    is_block
        Whether this protocol is contiguous-block masking.
    table_fields
        Dict fragment for long-form ``mask_rate`` / ``block_len``.

    Instance variables
    ------------------
    name : str
        ``\"mcar\"`` or ``\"block\"``.
    mask_rate : float or None
        MCAR rate; ``None`` for block.
    block_len : int or None
        Contiguous block length; ``None`` for MCAR.
    """

    name: str
    mask_rate: float | None
    block_len: int | None

    def is_block(self) -> bool:
        """Return True for the contiguous-block protocol."""
        return self.name == "block"

    def table_fields(self) -> dict[str, float | int | None]:
        """Return ``mask_rate`` / ``block_len`` for a table row."""
        return {
            "mask_rate": self.mask_rate,
            "block_len": self.block_len,
        }


@dataclass(frozen=True)
class SeedPair:
    """Artificial-mask and TimeFill sampler seeds.

    Public methods
    --------------
    as_dict
        Serialize both seeds for JSON / CSV flatten.
    describe
        Short log string.

    Instance variables
    ------------------
    mask : int
        Seed for MCAR / block artificial masks.
    sample : int
        Seed for TimeFill reverse-SDE sampling.
    """

    mask: int
    sample: int

    def as_dict(self) -> dict[str, int]:
        """Return ``mask_seed`` / ``sample_seed`` keys."""
        return {
            "mask_seed": int(self.mask),
            "sample_seed": int(self.sample),
        }

    def describe(self) -> str:
        """Return a short seed summary."""
        return f"mask_seed={self.mask}, sample_seed={self.sample}"


@dataclass(frozen=True)
class ScoreBlock:
    """Point and probabilistic metrics for one method × protocol.

    Public methods
    --------------
    as_dict
        Export metric name → float.
    from_table
        Classmethod: build from a ``MetricsTable``.

    Instance variables
    ------------------
    mae : float
        Masked MAE.
    rmse : float
        Masked RMSE.
    crps : float
        Empirical CRPS.
    coverage_50 : float
        Central 50% coverage.
    coverage_90 : float
        Central 90% coverage.
    """

    mae: float
    rmse: float
    crps: float
    coverage_50: float
    coverage_90: float

    def as_dict(self) -> dict[str, float]:
        """Return metric scalars as a plain dict."""
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "crps": self.crps,
            "coverage_50": self.coverage_50,
            "coverage_90": self.coverage_90,
        }

    @classmethod
    def from_table(cls, table: MetricsTable) -> "ScoreBlock":
        """Build a score block from an ``ImputationEvaluator`` table.

        Parameters
        ----------
        table : MetricsTable
            Computed metrics.

        Returns
        -------
        ScoreBlock
            Copied scalars.
        """
        return cls(
            mae=table.mae,
            rmse=table.rmse,
            crps=table.crps,
            coverage_50=table.coverage_50,
            coverage_90=table.coverage_90,
        )


@dataclass(frozen=True)
class RunMeta:
    """Non-score columns shared by a metrics row.

    Public methods
    --------------
    as_dict
        Flatten seeds and meta fields for export.
    k_for_method
        Effective ``K`` (``1`` for point baselines).

    Instance variables
    ------------------
    seeds : SeedPair
        Mask and sampler seeds.
    split : str
        Report split name (``test`` or smoke ``val``).
    k_samples : int
        TimeFill ensemble size from the bench config.
    point_metrics : str
        ``degenerate_k1`` or ``ensemble``.
    """

    seeds: SeedPair
    split: str
    k_samples: int
    point_metrics: str

    def as_dict(self) -> dict[str, int | str]:
        """Return meta fields including flattened seed columns."""
        payload: dict[str, int | str] = {
            "split": self.split,
            "k_samples": int(self.k_samples),
            "point_metrics": self.point_metrics,
        }
        payload.update(self.seeds.as_dict())
        return payload

    def k_for_method(self, method: str) -> int:
        """Return ``1`` for point baselines, else TimeFill ``K``.

        Parameters
        ----------
        method : str
            Method name (``timefill``, ``locf``, ``linear``).

        Returns
        -------
        int
            Ensemble size recorded in the table.
        """
        if method in _POINT_METHODS:
            return _POINT_K
        return int(self.k_samples)


@dataclass(frozen=True)
class MetricRecord:
    """One composed long-form M3 metrics row.

    Public methods
    --------------
    as_dict
        Flatten nested fields for JSON.
    csv_cells
        Ordered cell values matching the long-form schema.

    Instance variables
    ------------------
    method : str
        ``timefill``, ``locf``, or ``linear``.
    protocol : ProtocolKey
        Protocol identity.
    scores : ScoreBlock
        Metric scalars.
    meta : RunMeta
        Seeds, split, ``K``, point-metric policy.
    n_target_cells : int
        Count of scored artificial targets.
    """

    method: str
    protocol: ProtocolKey
    scores: ScoreBlock
    meta: RunMeta
    n_target_cells: int

    def as_dict(self) -> dict[str, float | int | str | None]:
        """Flatten to the long-form artifact schema."""
        payload: dict[str, float | int | str | None] = {
            "method": self.method,
            "protocol": self.protocol.name,
            "n_target_cells": int(self.n_target_cells),
        }
        payload.update(self.protocol.table_fields())
        payload.update(self.scores.as_dict())
        meta = self.meta.as_dict()
        meta["k_samples"] = self.meta.k_for_method(self.method)
        if self.method in _POINT_METHODS:
            meta["point_metrics"] = "degenerate_k1"
        else:
            meta["point_metrics"] = "ensemble"
        payload.update(meta)
        return payload

    def csv_cells(self) -> list[str]:
        """Return CSV cells in long-form column order."""
        mapping = self.as_dict()
        return [format_csv_cell(mapping[name]) for name in _CSV_FIELDS]


class MetricsTableAssembler:
    """Collect composed metric records and write CSV / JSON artifacts.

    Public methods
    --------------
    append
        Add one ``MetricRecord``.
    write_csv
        Write ``metrics_by_rate.csv``.
    write_json
        Write ``metrics_by_rate.json``.
    rows
        Return a copy of collected records.

    Instance variables
    ------------------
    _rows : list of MetricRecord
        Buffer of completed rows.
    """

    def __init__(self) -> None:
        """Create an empty row buffer."""
        self._rows: list[MetricRecord] = []

    def append(self, row: MetricRecord) -> None:
        """Append one metrics record.

        Parameters
        ----------
        row : MetricRecord
            Completed protocol × method row.
        """
        self._rows.append(row)

    def rows(self) -> list[MetricRecord]:
        """Return a shallow copy of collected records."""
        return list(self._rows)

    def write_csv(self, path: Path) -> Path:
        """Write long-form CSV under ``path``.

        Parameters
        ----------
        path : pathlib.Path
            Destination CSV path.

        Returns
        -------
        pathlib.Path
            Path written.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(list(_CSV_FIELDS))
            for row in self._rows:
                writer.writerow(row.csv_cells())
        return out

    def write_json(self, path: Path) -> Path:
        """Write long-form JSON list under ``path``.

        Parameters
        ----------
        path : pathlib.Path
            Destination JSON path.

        Returns
        -------
        pathlib.Path
            Path written.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = [row.as_dict() for row in self._rows]
        out.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        return out


def format_csv_cell(value: float | int | str | None) -> str:
    """Format one CSV cell, using empty string for ``None``.

    Parameters
    ----------
    value : float or int or str or None
        Cell value.

    Returns
    -------
    str
        Serialized cell.
    """
    if value is None:
        return ""
    return str(value)


def point_method_names() -> frozenset[str]:
    """Return the locked point-baseline method name set.

    Returns
    -------
    frozenset of str
        ``{\"locf\", \"linear\"}``.
    """
    return _POINT_METHODS


__all__ = [
    "MetricRecord",
    "MetricsTableAssembler",
    "ProtocolKey",
    "RunMeta",
    "ScoreBlock",
    "SeedPair",
    "format_csv_cell",
    "point_method_names",
]
