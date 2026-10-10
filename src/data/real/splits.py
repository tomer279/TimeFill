"""Deterministic train / val / test splits for M2 real windows.

Classes
-------
SplitIndices
    Disjoint train / val / test index arrays.
EntityAwareSplitter
    Entity- or index-based splits without window leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.real.specs import SplitFractions

_SPLIT_TRAIN = 0
_SPLIT_VAL = 1
_N_SPLITS = 3


@dataclass(frozen=True)
class SplitIndices:
    """Disjoint train / val / test index arrays into a window dataset.

    Public methods
    --------------
    assert_disjoint
        Raise if any index appears in more than one split.
    n_total
        Total number of assigned indices.

    Instance variables
    ------------------
    train_idx : numpy.ndarray
        Train window indices.
    val_idx : numpy.ndarray
        Validation window indices.
    test_idx : numpy.ndarray
        Test window indices.
    """

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    def assert_disjoint(self) -> None:
        """Raise ``ValueError`` if train / val / test indices overlap."""
        train_set = set(np.asarray(self.train_idx, dtype=np.int64).tolist())
        val_set = set(np.asarray(self.val_idx, dtype=np.int64).tolist())
        test_set = set(np.asarray(self.test_idx, dtype=np.int64).tolist())
        if train_set & val_set:
            raise ValueError("train and val indices overlap")
        if train_set & test_set:
            raise ValueError("train and test indices overlap")
        if val_set & test_set:
            raise ValueError("val and test indices overlap")

    def n_total(self) -> int:
        """Return the number of indices across all splits.

        Returns
        -------
        int
            ``len(train) + len(val) + len(test)``.
        """
        return (
            int(self.train_idx.size)
            + int(self.val_idx.size)
            + int(self.test_idx.size)
        )


class EntityAwareSplitter:
    """Deterministic train / val / test splits without index overlap.

    Prefer holding out entire entities when enough sites exist; otherwise
    use ``split_by_index`` over non-overlapping windows.

    Public methods
    --------------
    split_by_entity
        Assign every window of an entity to one split.
    split_by_index
        Shuffle window indices and cut by largest-remainder sizes.

    Instance variables
    ------------------
    fractions : SplitFractions
        Fractions and seed.
    """

    def __init__(self, fractions: SplitFractions) -> None:
        """Store validated split fractions.

        Parameters
        ----------
        fractions : SplitFractions
            Train / val / test fractions and seed.
        """
        fractions.validate()
        self.fractions = fractions

    def split_by_entity(self, entity_ids: np.ndarray) -> SplitIndices:
        """Split windows by unique entity id (no entity in two splits).

        Parameters
        ----------
        entity_ids : numpy.ndarray
            Per-window entity labels, shape ``(N,)``.

        Returns
        -------
        SplitIndices
            Disjoint index arrays covering all windows.
        """
        entities = np.asarray(entity_ids)
        unique = np.unique(entities)
        n_entities = int(unique.size)
        if n_entities == 0:
            raise ValueError("entity_ids must be non-empty")
        order = _seeded_permutation(n_entities, self.fractions.seed)
        shuffled = unique[order]
        n_train, n_val, n_test = _split_sizes(n_entities, self.fractions)
        train_ents = shuffled[:n_train]
        val_ents = shuffled[n_train:n_train + n_val]
        test_ents = shuffled[n_train + n_val:n_train + n_val + n_test]
        split = SplitIndices(
            train_idx=_indices_for_entities(entities, train_ents),
            val_idx=_indices_for_entities(entities, val_ents),
            test_idx=_indices_for_entities(entities, test_ents),
        )
        split.assert_disjoint()
        return split

    def split_by_index(self, n_windows: int) -> SplitIndices:
        """Split by shuffled window indices.

        Parameters
        ----------
        n_windows : int
            Total number of windows ``N``.

        Returns
        -------
        SplitIndices
            Disjoint index arrays covering ``0 … N-1``.
        """
        if n_windows <= 0:
            raise ValueError(f"n_windows must be positive, got {n_windows}")
        order = _seeded_permutation(n_windows, self.fractions.seed)
        n_train, n_val, n_test = _split_sizes(n_windows, self.fractions)
        split = SplitIndices(
            train_idx=np.asarray(order[:n_train], dtype=np.int64),
            val_idx=np.asarray(
                order[n_train:n_train + n_val],
                dtype=np.int64,
            ),
            test_idx=np.asarray(
                order[n_train + n_val:n_train + n_val + n_test],
                dtype=np.int64,
            ),
        )
        split.assert_disjoint()
        return split


def _indices_for_entities(
        entities: np.ndarray,
        selected: np.ndarray,
) -> np.ndarray:
    """Return window indices whose entity is in ``selected``.

    Parameters
    ----------
    entities : numpy.ndarray
        Per-window entity labels.
    selected : numpy.ndarray
        Entity ids assigned to one split.

    Returns
    -------
    numpy.ndarray
        Integer indices, possibly empty.
    """
    if selected.size == 0:
        return np.asarray([], dtype=np.int64)
    return np.flatnonzero(np.isin(entities, selected)).astype(np.int64)


def _seeded_permutation(n_items: int, seed: int) -> np.ndarray:
    """Return a deterministic permutation of ``0 … n_items-1``.

    Parameters
    ----------
    n_items : int
        Number of items.
    seed : int
        RNG seed.

    Returns
    -------
    numpy.ndarray
        Permuted indices.
    """
    rng = np.random.default_rng(int(seed))
    return rng.permutation(int(n_items))


def _split_sizes(
        n_items: int,
        fractions: SplitFractions,
) -> tuple[int, int, int]:
    """Largest-remainder allocation of ``n_items`` into three splits.

    Parameters
    ----------
    n_items : int
        Number of entities or windows.
    fractions : SplitFractions
        Target fractions.

    Returns
    -------
    n_train : int
        Train count.
    n_val : int
        Val count.
    n_test : int
        Test count (absorbs rounding remainder).
    """
    weights = np.asarray(
        [fractions.train, fractions.val, fractions.test],
        dtype=np.float64,
    )
    raw = weights * float(n_items)
    floors = np.floor(raw).astype(np.int64)
    remainder = int(n_items - int(floors.sum()))
    frac_parts = raw - floors
    order = np.argsort(-frac_parts, kind="mergesort")
    for rank in range(remainder):
        floors[order[rank % _N_SPLITS]] += 1
    n_train = int(floors[_SPLIT_TRAIN])
    n_val = int(floors[_SPLIT_VAL])
    n_test = int(n_items - n_train - n_val)
    return n_train, n_val, n_test
