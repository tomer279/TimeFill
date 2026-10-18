"""M7 CLI: Flexible ingest → upload-adapt → impute → downloadable fill.

File-local adapt smoke only. Does **not** write ``artifacts/m2/`` or
``artifacts/m3/`` claim tables. Does not overwrite M2 checkpoints.

Exit codes
----------
0
    Success (adapt + mean-fill export).
1
    Unexpected / unclassified failure (message on stderr).
2
    Ingest / too-little-data / policy reject (no adapt start).
3
    Adapt abort: non-finite loss (no promote).
4
    Adapt abort: wall-clock timeout (no promote).

Examples
--------
# Toy non-Beijing fixture (4 rows): use small T so B >= min_windows
python scripts/adapt_upload_m7.py \\
  --csv tests/fixtures/toy_non_beijing.csv \\
  --config configs/adapt_m7.yaml \\
  --seq-len 2 \\
  --epochs 1 \\
  --steps-per-epoch 1 \\
  --batch-size 2 \\
  --k-samples 2 \\
  --solver-steps 4 \\
  --timeout-seconds 120 \\
  --smoke-eval \\
  --export-samples
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.adapt_upload_m7 import adapt_upload_main  # pylint: disable=wrong-import-position


def main(argv: list[str] | None = None) -> int:
    """Delegate to ``app.adapt_upload_m7.adapt_upload_main``.

    Parameters
    ----------
    argv : list of str or None, optional
        Optional CLI args.

    Returns
    -------
    int
        Process exit code.
    """
    return adapt_upload_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
