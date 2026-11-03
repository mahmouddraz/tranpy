from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tranpy.conventional.network import (
    Branches,
    admittance_matrix,
    internal_node_admittance,
    kron_reduce,
    with_shunt,
)
from tranpy_core.errors import ConfigurationError


def _branches(f, t, r, x, b, tap=None) -> Branches:
    n = len(f)
    return Branches(
        np.asarray(f, dtype=np.int64),
        np.asarray(t, dtype=np.int64),
        np.asarray(r, dtype=float),
        np.asarray(x, dtype=float),
        np.asarray(b, dtype=float),
        np.ones(n) if tap is None else np.asarray(tap, dtype=float),
    )


def test_two_machines_through_one_line_reduce_to_the_series_reactance() -> None:
    # hand calculation: x'd 0.1 + line 0.3 + x'd 0.1 = 0.5 pu, so Y'12 = -1/(j0.5) = +j2
    y_bus = admittance_matrix(2, _branches([0], [1], [0.0], [0.3], [0.0]))
    y = internal_node_admittance(y_bus, [0, 1], [0.1, 0.1])
    np.testing.assert_allclose(y, [[-2j, 2j], [2j, -2j]], atol=1e-12)


def test_a_bolted_fault_at_one_terminal_decouples_the_machines() -> None:
    y_bus = with_shunt(admittance_matrix(2, _branches([0], [1], [0.0], [0.3], [0.0])), 0, 1e9)
    y = internal_node_admittance(y_bus, [0, 1], [0.1, 0.1])
    assert abs(y[0, 1]) < 1e-6


def test_kron_reduction_keeping_every_node_changes_nothing() -> None:
    y = admittance_matrix(3, _branches([0, 1], [1, 2], [0.01, 0.02], [0.1, 0.2], [0.0, 0.0]))
    np.testing.assert_allclose(kron_reduce(y, [0, 1, 2]), y)


@settings(max_examples=40, deadline=None)
@given(
    st.integers(min_value=3, max_value=8),
    st.data(),
)
def test_reduction_of_a_passive_network_is_symmetric(n: int, data: st.DataObject) -> None:
    # ring plus random chords, no off-nominal taps: a reciprocal network
    f = list(range(n))
    t = [(i + 1) % n for i in range(n)]
    r = data.draw(st.lists(st.floats(0.001, 0.05), min_size=n, max_size=n))
    x = data.draw(st.lists(st.floats(0.01, 0.5), min_size=n, max_size=n))
    b = data.draw(st.lists(st.floats(0.0, 0.3), min_size=n, max_size=n))
    y_bus = admittance_matrix(n, _branches(f, t, r, x, b))
    loads = np.array(data.draw(st.lists(st.floats(0.1, 2.0), min_size=n, max_size=n))) * (1 - 0.3j)
    y = internal_node_admittance(y_bus, [0, n // 2], [0.1, 0.2], loads)
    np.testing.assert_allclose(y, y.T, atol=1e-9)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ConfigurationError):
        _branches([0], [1], [0.0], [0.0], [0.0])
    y = admittance_matrix(2, _branches([0], [1], [0.0], [0.3], [0.0]))
    with pytest.raises(ConfigurationError):
        internal_node_admittance(y, [0, 1], [0.1, -0.1])
