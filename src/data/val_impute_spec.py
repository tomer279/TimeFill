"""Validate ``train.val_impute`` mappings for WP4 selection.

Functions
---------
validate_val_impute_mapping
    Raise ``ValueError`` if a ``train.val_impute`` mapping is invalid.
validate_val_impute_protocol
    Raise ``ValueError`` unless a protocol is ``mcar_*`` or ``block_*``.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "validate_val_impute_mapping",
    "validate_val_impute_protocol",
]


def validate_val_impute_mapping(mapping: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` if a ``train.val_impute`` mapping is invalid.

    Parameters
    ----------
    mapping : mapping
        ``n_windows``, ``K_samples``, ``solver_steps``, seeds, ``protocols``.
    """
    required = (
        "n_windows",
        "K_samples",
        "solver_steps",
        "mask_seed",
        "sample_seed",
        "protocols",
    )
    missing = [key for key in required if key not in mapping]
    if missing:
        raise ValueError(f"val_impute missing keys: {missing}")
    for key in ("n_windows", "K_samples", "solver_steps"):
        if int(mapping[key]) <= 0:
            raise ValueError(f"val_impute.{key} must be positive")
    protocols = mapping["protocols"]
    if not isinstance(protocols, (list, tuple)) or not protocols:
        raise ValueError("val_impute.protocols must be a non-empty list")
    for protocol in protocols:
        validate_val_impute_protocol(str(protocol))


def validate_val_impute_protocol(protocol: str) -> None:
    """Raise ``ValueError`` unless ``protocol`` is ``mcar_*`` or ``block_*``.

    Parameters
    ----------
    protocol : str
        Protocol name from ``val_impute.protocols``.
    """
    if protocol.startswith("mcar_"):
        try:
            rate = float(protocol.split("_", 1)[1])
        except ValueError as exc:
            raise ValueError(
                f"unsupported val_impute protocol {protocol!r}"
            ) from exc
        if not 0.0 <= rate <= 1.0:
            raise ValueError(
                f"val_impute mcar rate must be in [0, 1], got {protocol!r}"
            )
        return
    if protocol.startswith("block_"):
        try:
            block_len = int(protocol.split("_", 1)[1])
        except ValueError as exc:
            raise ValueError(
                f"unsupported val_impute protocol {protocol!r}"
            ) from exc
        if block_len <= 0:
            raise ValueError(
                f"val_impute block length must be positive, got {protocol!r}"
            )
        return
    raise ValueError(f"unsupported val_impute protocol {protocol!r}")
