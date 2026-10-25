"""Seeded, stratified splitters that return index arrays.

Splits are defined on labels (and optional groups) only, never on features, so the same
split can be applied to any representation of the same samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.model_selection import train_test_split as _sk_train_test_split

from tranpy_core.errors import ConfigurationError
from tranpy_core.random import Seed, spawn_seed

__all__ = ["Split", "kfold_splits", "train_test_split"]

IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class Split:
    """Disjoint train and test indices."""

    train: IntArray
    test: IntArray

    def __post_init__(self) -> None:
        train = np.array(self.train, dtype=np.int64)
        test = np.array(self.test, dtype=np.int64)
        if np.intersect1d(train, test).size:
            raise ConfigurationError("train and test indices overlap")
        train.setflags(write=False)
        test.setflags(write=False)
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "test", test)


def train_test_split(
    labels: npt.ArrayLike,
    *,
    test_size: float,
    seed: Seed,
    stratify: bool = True,
    groups: npt.ArrayLike | None = None,
) -> Split:
    """Split sample indices into train and test sets.

    Args:
        labels: class label per sample.
        test_size: fraction of samples in the test set, in (0, 1).
        seed: random seed or generator.
        stratify: keep class proportions equal in both sets.
        groups: optional group id per sample; a group never spans both sets. With groups the
            split is the first fold of a stratified group k-fold with
            ``round(1 / test_size)`` folds, so ``test_size`` is approximate.
    """
    y = np.asarray(labels)
    if not 0.0 < test_size < 1.0:
        raise ConfigurationError(f"test_size must lie in (0, 1), got {test_size}")
    index = np.arange(y.shape[0])
    if groups is not None:
        n_splits = max(2, round(1.0 / test_size))
        return kfold_splits(y, n_splits=n_splits, seed=seed, groups=groups, stratify=stratify)[0]
    train, test = _sk_train_test_split(
        index, test_size=test_size, random_state=spawn_seed(seed), stratify=y if stratify else None
    )
    return Split(train=train, test=test)


def kfold_splits(
    labels: npt.ArrayLike,
    *,
    n_splits: int,
    seed: Seed,
    groups: npt.ArrayLike | None = None,
    stratify: bool = True,
) -> list[Split]:
    """Shuffled (stratified, optionally grouped) k-fold splits."""
    y = np.asarray(labels)
    if n_splits < 2:
        raise ConfigurationError(f"n_splits must be at least 2, got {n_splits}")
    random_state = spawn_seed(seed)
    placeholder = np.zeros((y.shape[0], 1))
    if groups is not None:
        g = np.asarray(groups)
        strat = y if stratify else np.zeros_like(y)
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = splitter.split(placeholder, strat, g)
    elif stratify:
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(
            placeholder, y
        )
    else:
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(placeholder)
    return [Split(train=tr, test=te) for tr, te in folds]
