"""Stitch per-row ablation bench CSVs into ``artifacts/m7_5/ablation.csv``.

Example
-------
python scripts/assemble_m7_5_ablation.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_OUT = _REPO_ROOT / "artifacts" / "m7_5" / "ablation.csv"
_ROWS = (
    ("1_m2_legacy", "artifacts/m7_5/ablation/bench_row1_m2/metrics_by_rate.csv"),
    ("2_wp1_clean", "artifacts/m7_5/ablation/bench_row2_wp1/metrics_by_rate.csv"),
    ("3_wp1_wp2_eps", "artifacts/m7_5/ablation/bench_row3_wp1_wp2/metrics_by_rate.csv"),
    ("6_full_m7_5", "artifacts/m7_5/ablation/bench_row6_m75/metrics_by_rate.csv"),
)


def main() -> Path:
    """Write the combined ablation table and return its path."""
    combined: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for label, rel_path in _ROWS:
        path = _REPO_ROOT / rel_path
        if not path.is_file():
            raise FileNotFoundError(f"missing ablation bench CSV: {path}")
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = ["ablation", *reader.fieldnames]
            for row in reader:
                payload = {"ablation": label}
                payload.update(row)
                combined.append(payload)
    if fieldnames is None:
        raise RuntimeError("no ablation rows to write")
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)
    print(f"Wrote {_OUT} ({len(combined)} rows)")
    return _OUT


if __name__ == "__main__":
    main()
