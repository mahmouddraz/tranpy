from __future__ import annotations

import numpy as np
import pytest

from tranpy_core._optional import is_available, require
from tranpy_core.errors import MissingDependencyError
from tranpy_core.random import make_rng, spawn_seed


def test_make_rng_is_reproducible_and_passes_generators_through() -> None:
    assert make_rng(3).integers(0, 1000) == make_rng(3).integers(0, 1000)
    gen = np.random.default_rng(1)
    assert make_rng(gen) is gen
    assert make_rng(np.int64(5)).integers(0, 10) == make_rng(5).integers(0, 10)


@pytest.mark.parametrize("bad", [None, 1.5, "1", True])
def test_make_rng_rejects_non_seeds(bad: object) -> None:
    with pytest.raises(TypeError):
        make_rng(bad)  # type: ignore[arg-type]


def test_spawn_seed() -> None:
    assert spawn_seed(7) == 7
    gen = np.random.default_rng(0)
    a, b = spawn_seed(gen), spawn_seed(gen)
    assert a != b
    assert spawn_seed(np.random.default_rng(0)) == a


def test_no_global_rng_use(toy) -> None:  # type: ignore[no-untyped-def]
    from tranpy_core.data import kfold_splits, train_test_split

    state = np.random.get_state()[1].copy()  # noqa: NPY002
    train_test_split(toy.labels, test_size=0.3, seed=0)
    kfold_splits(toy.labels, n_splits=3, seed=0)
    assert np.array_equal(np.random.get_state()[1], state)  # noqa: NPY002


def test_require() -> None:
    assert require("json", "x").dumps([1]) == "[1]"
    with pytest.raises(MissingDependencyError, match=r"tranpy-ai\[extra\]"):
        require("definitely_not_a_module_xyz", "extra")
    assert is_available("json")
    assert not is_available("definitely_not_a_module_xyz")
