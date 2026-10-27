from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tranpy_core.data import Split, kfold_splits, train_test_split
from tranpy_core.errors import ConfigurationError

labels_strategy = st.lists(st.integers(0, 1), min_size=30, max_size=200).filter(
    lambda y: min(y.count(0), y.count(1)) >= 6
)


@settings(max_examples=40, deadline=None)
@given(labels_strategy, st.floats(0.15, 0.5), st.integers(0, 10_000))
def test_train_test_split_is_disjoint_complete_and_stratified(
    y: list[int], test_size: float, seed: int
) -> None:
    labels = np.array(y)
    split = train_test_split(labels, test_size=test_size, seed=seed)
    assert np.intersect1d(split.train, split.test).size == 0
    assert np.array_equal(np.sort(np.concatenate([split.train, split.test])), np.arange(len(y)))
    overall = labels.mean()
    # stratification keeps each part within one sample of the overall proportion
    assert abs(labels[split.test].mean() - overall) <= 1.0 / split.test.size + 1e-9
    assert split == split  # identity
    again = train_test_split(labels, test_size=test_size, seed=seed)
    assert np.array_equal(again.test, split.test)


@settings(max_examples=25, deadline=None)
@given(labels_strategy, st.integers(2, 5), st.integers(0, 1000))
def test_kfold_partitions_samples(y: list[int], k: int, seed: int) -> None:
    labels = np.array(y)
    folds = kfold_splits(labels, n_splits=k, seed=seed)
    assert len(folds) == k
    tests = np.concatenate([f.test for f in folds])
    assert np.array_equal(np.sort(tests), np.arange(len(y)))
    for f in folds:
        assert np.intersect1d(f.train, f.test).size == 0


def test_groups_never_span_both_sides() -> None:
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 40, size=300)
    labels = (groups % 3 == 0).astype(int)
    split = train_test_split(labels, test_size=0.25, seed=1, groups=groups)
    assert not set(groups[split.train]) & set(groups[split.test])
    for fold in kfold_splits(labels, n_splits=4, seed=1, groups=groups, stratify=False):
        assert not set(groups[fold.train]) & set(groups[fold.test])


def test_unstratified_and_invalid_arguments() -> None:
    labels = np.array([0] * 50 + [1] * 50)
    plain = train_test_split(labels, test_size=0.2, seed=0, stratify=False)
    assert plain.test.size == 20
    assert len(kfold_splits(labels, n_splits=4, seed=0, stratify=False)) == 4
    with pytest.raises(ConfigurationError):
        train_test_split(labels, test_size=1.5, seed=0)
    with pytest.raises(ConfigurationError):
        kfold_splits(labels, n_splits=1, seed=0)
    with pytest.raises(ConfigurationError, match="overlap"):
        Split(train=np.array([1, 2]), test=np.array([2, 3]))
    s = Split(train=np.array([0]), test=np.array([1]))
    with pytest.raises(ValueError, match="read-only"):
        s.train[0] = 5


def test_matches_sklearn_reference_split() -> None:
    """Same indices as sklearn's stratified train_test_split with random_state=seed."""
    from sklearn.model_selection import train_test_split as sk

    labels = np.array([0, 1] * 60 + [1] * 30)
    tr, te = sk(np.arange(labels.size), test_size=0.2, random_state=42, stratify=labels)
    split = train_test_split(labels, test_size=0.2, seed=42)
    assert np.array_equal(split.train, tr) and np.array_equal(split.test, te)
