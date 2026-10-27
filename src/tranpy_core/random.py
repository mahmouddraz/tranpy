"""Explicit randomness: every random operation receives a seed or a Generator.

No function in this package touches NumPy's global random state.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Seed", "make_rng", "spawn_seed"]

Seed = int | np.random.Generator
"""An integer seed or an existing generator (shared, so draws advance it)."""

_MAX_SEED = 2**31 - 1


def make_rng(seed: Seed) -> np.random.Generator:
    """Return a generator for ``seed``; a Generator is passed through unchanged."""
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
        raise TypeError(f"seed must be an int or numpy.random.Generator, got {type(seed).__name__}")
    return np.random.default_rng(int(seed))


def spawn_seed(seed: Seed) -> int:
    """Derive an int seed, e.g. for libraries that take ``random_state: int``.

    For an int the value itself is returned, so ``random_state=seed`` is reproduced
    exactly; for a Generator a fresh integer is drawn from it.
    """
    if isinstance(seed, np.random.Generator):
        return int(seed.integers(0, _MAX_SEED))
    make_rng(seed)  # type validation
    return int(seed)
