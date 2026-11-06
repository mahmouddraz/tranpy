"""Kinetic-energy terms of the transient energy function (COI and corrected)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tranpy.conventional.energy import (
    coi_kinetic_energy,
    corrected_kinetic_energy,
    critical_group_by_largest_gap,
    kinetic_energy_features,
)
from tranpy_core.errors import ConfigurationError, DataError


def test_two_machine_coi_energy_equals_the_corrected_energy() -> None:
    # two machines: the COI KE is the KE of the relative motion, 1/2 M_eq (w1 - w2)^2
    w = np.array([3.0, 1.0])  # H * Sn
    dw = np.array([[0.01, -0.02]])
    expected = 3.0 * 1.0 / 4.0 * 0.03**2
    assert coi_kinetic_energy(dw, w)[0] == pytest.approx(expected)
    crit = np.array([[True, False]])
    assert corrected_kinetic_energy(dw, w, crit)[0] == pytest.approx(expected)


def test_common_speed_change_carries_no_energy_in_the_coi_frame() -> None:
    w = np.array([[2.0, 5.0, 1.0]])
    assert coi_kinetic_energy(np.full((1, 3), 0.004), w)[0] == pytest.approx(0.0)


@settings(max_examples=50, deadline=None)
@given(
    st.lists(st.floats(0.1, 10.0), min_size=3, max_size=8),
    st.lists(st.floats(-0.05, 0.05), min_size=8, max_size=8),
    st.integers(1, 7),
)
def test_corrected_energy_never_exceeds_the_coi_energy(w, dw, k) -> None:
    n = len(w)
    weights = np.array(w)
    speeds = np.array(dw[:n])[None, :]
    crit = np.zeros((1, n), dtype=bool)
    crit[0, : min(k, n - 1)] = True
    # the inter-group part is a component of the total relative KE (Fouad & Vittal)
    assert (
        corrected_kinetic_energy(speeds, weights, crit)[0]
        <= coi_kinetic_energy(speeds, weights)[0] * (1 + 1e-9) + 1e-15
    )


def test_largest_gap_picks_the_advanced_machines_and_skips_offline_ones() -> None:
    x = np.array([[30.0, 28.0, 5.0, 3.0, 100.0]])
    online = np.array([[True, True, True, True, False]])
    crit = critical_group_by_largest_gap(x, online)
    assert crit.tolist() == [[True, True, False, False, False]]
    with pytest.raises(DataError):
        critical_group_by_largest_gap(np.array([[1.0, 2.0]]), np.array([[True, False]]))


def test_empty_group_gives_nan_and_shapes_are_checked() -> None:
    w = np.array([1.0, 1.0])
    assert np.isnan(corrected_kinetic_energy([[0.1, 0.0]], w, [[True, True]])[0])
    with pytest.raises(ConfigurationError):
        coi_kinetic_energy([[0.1, 0.0, 0.0]], w)
    with pytest.raises(ConfigurationError):
        coi_kinetic_energy([[0.1, 0.0]], [0.0, 0.0])


def test_features_from_aligned_columns_per_case() -> None:
    aligned = pd.DataFrame(
        {
            "pre|A|dangle": [0.0, 0.0, 0.0],
            "pre|B|dangle": [0.0, 0.0, 0.0],
            "pre|C|dangle": [0.0, 0.0, 0.0],
            "clear|A|dangle": [20.0, 1.0, 0.0],
            "clear|B|dangle": [-5.0, -1.0, 0.0],
            "clear|C|dangle": [-6.0, 0.0, 0.0],
            "clear|A|dw": [0.02, 0.001, 0.0],
            "clear|B|dw": [-0.01, -0.001, 0.0],
            "clear|C|dw": [-0.01, 0.0, 0.0],
        }
    )
    cases = pd.Series(["h1", "h1", "h2"])
    weights = {"h1": {"A": 1.0, "B": 2.0, "C": 2.0}, "h2": {"A": 1.0, "C": 1.0}}
    out = kinetic_energy_features(aligned, cases, weights)
    assert out.loc[0, "n_critical"] == 1
    assert out.loc[0, "ke_corr_mj"] == pytest.approx(1.0 * 4.0 / 5.0 * 0.03**2)
    assert out.loc[0, "ke_coi_mj"] == pytest.approx(
        coi_kinetic_energy([[0.02, -0.01, -0.01]], [1.0, 2.0, 2.0])[0]
    )
    assert out.loc[2, "ke_coi_mj"] == 0.0
    with pytest.raises(DataError):
        kinetic_energy_features(aligned, pd.Series(["h3"] * 3), weights)
