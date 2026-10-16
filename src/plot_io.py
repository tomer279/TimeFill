"""Filesystem helpers for Matplotlib figure export.

Functions
---------
save_figure
    Write a figure via a temporary file, then atomically replace the target.
"""

from __future__ import annotations

import time
from pathlib import Path

from matplotlib.figure import Figure

_TMP_SUFFIX = ".tmp"
_DEFAULT_DPI = 120
_MAX_ATTEMPTS = 3
_RETRY_SLEEP_S = 0.05


def save_figure(
        fig: Figure,
        path: Path | str,
        dpi: int = _DEFAULT_DPI,
) -> Path:
    """Save ``fig`` to ``path`` using a same-directory temp file.

    Writing through a temporary file then ``replace`` avoids intermittent
    Windows ``OSError: [Errno 22] Invalid argument`` when overwriting an
    existing PNG that is briefly locked (Explorer thumbnails, viewers).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to persist.
    path : path-like
        Destination image path (parent dirs are created).
    dpi : int, optional
        Resolution passed to ``Figure.savefig``.

    Returns
    -------
    pathlib.Path
        Destination path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    last_error: OSError | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return _save_via_temp(fig, out, dpi)
        except OSError as exc:
            last_error = exc
            if attempt + 1 >= _MAX_ATTEMPTS:
                break
            time.sleep(_RETRY_SLEEP_S)
    assert last_error is not None
    raise last_error


def _save_via_temp(
        fig: Figure,
        out: Path,
        dpi: int,
) -> Path:
    """Write ``fig`` to a temp path beside ``out``, then replace.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to persist.
    out : pathlib.Path
        Final destination path.
    dpi : int
        Save resolution.

    Returns
    -------
    pathlib.Path
        ``out`` after a successful replace.
    """
    # Keep the real suffix last so Matplotlib infers format (e.g. .png).
    tmp_path = out.with_name(out.stem + _TMP_SUFFIX + out.suffix)
    replaced = False
    try:
        fig.savefig(str(tmp_path), dpi=dpi)
        _replace_destination(tmp_path, out)
        replaced = True
    finally:
        if not replaced:
            _remove_if_exists(tmp_path)
    return out


def _replace_destination(tmp_path: Path, out: Path) -> None:
    """Atomically move ``tmp_path`` onto ``out``, with unlink fallback.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary file written by ``savefig``.
    out : pathlib.Path
        Final destination path.
    """
    try:
        tmp_path.replace(out)
        return
    except OSError:
        if out.exists():
            out.unlink()
        tmp_path.replace(out)


def _remove_if_exists(path: Path) -> None:
    """Delete ``path`` when present; ignore missing-file races.

    Parameters
    ----------
    path : pathlib.Path
        Candidate temporary file.
    """
    if not path.exists():
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return
