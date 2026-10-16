"""M6 CLI: flexible / Exact ingest preview (no training).

Load a CSV (+ optional column-map YAML), print schema / missingness /
window summary, and optionally write a tiny windows NPZ for debugging.
Does not run reverse-SDE impute or fine-tune.

Examples
--------
python scripts/preview_ingest_m6.py --csv tests/fixtures/toy_non_beijing.csv
python scripts/preview_ingest_m6.py --csv path.csv --map map.yaml --dump-npz out.npz
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.flexible_ingest import (  # pylint: disable=wrong-import-position
    ColumnRoleMap,
    column_role_map_from_mapping,
    ingest_flexible_csv,
)
from app.preview_ingest import (  # pylint: disable=wrong-import-position
    MODE_EXACT_BEIJING,
    MODE_FLEXIBLE,
    FlexiblePreviewKnobs,
    PreviewBuildInputs,
    build_ingest_preview,
    evaluate_impute_gate,
    read_csv_headers,
    select_ingest_mode,
    try_adapt_flexible_windows,
)
from app.schema_validate import (  # pylint: disable=wrong-import-position
    ValidatedFrame,
    default_app_schema,
    validate_upload_csv,
)
from app.window_adapter import (  # pylint: disable=wrong-import-position
    AdaptedWindows,
    WindowAdaptRequest,
    adapt_validated_frame,
)
from src.data.real.specs import (  # pylint: disable=wrong-import-position
    default_beijing_features,
)

_DEFAULT_SEQ_LEN = 64
_DEFAULT_MAX_WINDOWS = 4


@dataclass(frozen=True)
class PreviewCliLoadRequest:
    """CSV load knobs for the M6 preview CLI.

    Public methods
    --------------
    as_flexible_knobs
        Build ``FlexiblePreviewKnobs`` from ``seq_len`` / cap.
    has_role_map
        Whether an explicit column map was supplied.

    Instance variables
    ------------------
    csv_path : pathlib.Path
        Input CSV path.
    role_map : ColumnRoleMap or None
        Optional Flexible map override.
    seq_len : int
        Window length ``T``.
    max_windows : int
        Optional window cap.
    """

    csv_path: Path
    role_map: ColumnRoleMap | None
    seq_len: int
    max_windows: int

    def as_flexible_knobs(self) -> FlexiblePreviewKnobs:
        """Return flexible adapt knobs for this CLI run.

        Returns
        -------
        FlexiblePreviewKnobs
            ``T`` and ``max_windows`` for preview adapt.
        """
        return FlexiblePreviewKnobs(
            seq_len=self.seq_len,
            max_windows=self.max_windows,
        )

    def has_role_map(self) -> bool:
        """Return whether an explicit role map was provided.

        Returns
        -------
        bool
            ``True`` when ``role_map`` is not ``None``.
        """
        return self.role_map is not None


@dataclass(frozen=True)
class PreviewCliLoadResult:
    """Validated frame plus optional adapted windows.

    Public methods
    --------------
    has_windows
        Whether adaptation produced windows.
    window_batch
        Return windows or raise if absent.

    Instance variables
    ------------------
    frame : ValidatedFrame
        Parsed table.
    mode : str
        Ingest mode string.
    windows : AdaptedWindows or None
        Adapted batch when successful.
    adapt_error : str or None
        Reject message when adapt fails.
    """

    frame: ValidatedFrame
    mode: str
    windows: AdaptedWindows | None
    adapt_error: str | None

    def has_windows(self) -> bool:
        """Return whether adapted windows are available.

        Returns
        -------
        bool
            ``True`` when ``windows`` is not ``None``.
        """
        return self.windows is not None

    def window_batch(self) -> AdaptedWindows:
        """Return adapted windows.

        Returns
        -------
        AdaptedWindows
            Non-null window batch.

        Raises
        ------
        ValueError
            If adaptation did not produce windows.
        """
        if self.windows is None:
            raise ValueError("no adapted windows available")
        return self.windows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for the M6 ingest preview.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed flags.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to a multivariate CSV",
    )
    parser.add_argument(
        "--map",
        type=Path,
        default=None,
        help="Optional YAML with features / timestamp / entity",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=_DEFAULT_SEQ_LEN,
        help="Window length T (default 64)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=_DEFAULT_MAX_WINDOWS,
        help="Optional window cap (default 4)",
    )
    parser.add_argument(
        "--dump-npz",
        type=Path,
        default=None,
        help="Optional path to write values/mask windows NPZ",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run ingest preview and print a short summary.

    Parameters
    ----------
    argv : list of str or None, optional
        Optional CLI args.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    args = parse_args(argv)
    role_map = None
    if args.map is not None:
        role_map = _load_role_map(Path(args.map))
    request = PreviewCliLoadRequest(
        csv_path=Path(args.csv),
        role_map=role_map,
        seq_len=int(args.seq_len),
        max_windows=int(args.max_windows),
    )
    loaded = _load_frame_and_windows(request)
    preview = build_ingest_preview(
        PreviewBuildInputs(
            frame=loaded.frame,
            ingest_mode=loaded.mode,
            windows=loaded.windows,
            adapt_error=loaded.adapt_error,
        )
    )
    gate = evaluate_impute_gate(
        loaded.frame.feature_names,
        default_beijing_features(),
        loaded.mode,
    )
    for line in preview.as_text_lines():
        print(line)
    print(f"impute_gate_allowed={gate.is_allowed()}")
    print(f"impute_gate: {gate.reason}")
    if args.dump_npz is not None:
        _maybe_dump_windows(Path(args.dump_npz), loaded)
    return 0


def _load_role_map(path: Path) -> ColumnRoleMap:
    """Load a YAML column role map.

    Parameters
    ----------
    path : pathlib.Path
        Map YAML path.

    Returns
    -------
    ColumnRoleMap
        Validated roles.

    Raises
    ------
    ValueError
        If YAML is empty or not a mapping.
    """
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"map YAML must be a mapping: {path}")
    return column_role_map_from_mapping(raw)


def _load_frame_and_windows(
        request: PreviewCliLoadRequest,
) -> PreviewCliLoadResult:
    """Validate CSV and adapt windows for preview.

    Parameters
    ----------
    request : PreviewCliLoadRequest
        CSV path, optional map, and window knobs.

    Returns
    -------
    PreviewCliLoadResult
        Frame, mode, and optional windows / reject text.
    """
    if request.has_role_map():
        # Explicit map forces Flexible (never Exact remap via YAML).
        return _load_flexible_result(request, MODE_FLEXIBLE)
    headers = read_csv_headers(request.csv_path)
    mode = select_ingest_mode(headers, default_app_schema())
    if mode == MODE_EXACT_BEIJING:
        frame = validate_upload_csv(request.csv_path, default_app_schema())
        windows = adapt_validated_frame(
            frame,
            WindowAdaptRequest(
                seq_len=request.seq_len,
                max_windows=request.max_windows,
            ),
        )
        return PreviewCliLoadResult(
            frame=frame,
            mode=mode,
            windows=windows,
            adapt_error=None,
        )
    return _load_flexible_result(request, mode)


def _load_flexible_result(
        request: PreviewCliLoadRequest,
        mode: str,
) -> PreviewCliLoadResult:
    """Ingest under Flexible rules and adapt with reject policy.

    Parameters
    ----------
    request : PreviewCliLoadRequest
        CSV path, optional map, and window knobs.
    mode : str
        Ingest mode label (typically ``flexible``).

    Returns
    -------
    PreviewCliLoadResult
        Frame plus windows or adapt reject text.
    """
    frame = ingest_flexible_csv(
        request.csv_path,
        role_map=request.role_map,
    )
    windows, adapt_error = try_adapt_flexible_windows(
        frame,
        request.as_flexible_knobs(),
    )
    return PreviewCliLoadResult(
        frame=frame,
        mode=mode,
        windows=windows,
        adapt_error=adapt_error,
    )


def _maybe_dump_windows(
        path: Path,
        loaded: PreviewCliLoadResult,
) -> None:
    """Write windows NPZ when adaptation succeeded; else print a skip note.

    Parameters
    ----------
    path : pathlib.Path
        Output ``.npz`` path.
    loaded : PreviewCliLoadResult
        Load / adapt outcome.
    """
    if not loaded.has_windows():
        print("dump-npz skipped: no adapted windows")
        return
    _write_windows_npz(path, loaded.window_batch())
    print(f"wrote {path}")


def _write_windows_npz(path: Path, windows: AdaptedWindows) -> None:
    """Write a tiny debug NPZ of adapted windows (no training artifacts).

    Parameters
    ----------
    path : pathlib.Path
        Output ``.npz`` path.
    windows : AdaptedWindows
        Adapted batch.
    """
    np.savez_compressed(
        path,
        values=windows.values,
        native_mask=windows.native_mask,
        feature_names=np.asarray(windows.feature_names),
    )


if __name__ == "__main__":
    raise SystemExit(main())
