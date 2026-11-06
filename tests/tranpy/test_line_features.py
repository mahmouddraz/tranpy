"""Faulted-line descriptors follow the engine conventions and the pre-fault state only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tranpy.line_features import CaseGrid, line_features, line_power
from tranpy.ml_events import check_no_leakage
from tranpy_core.errors import DataError

ANGLES = np.array([0.0, -0.1, -0.2, -0.25, -0.2])  # rad, buses 1..5
MVA = 100.0


def _sheets() -> dict[str, pd.DataFrame]:
    """Five buses: 1-2-3-4 at 380 kV (3-4 single, 2-3 double circuit), a step-up
    transformer 4-5 to a 21 kV machine terminal, and an equivalent machine at bus 3.
    The transformer is the second ``Line`` row, so the engine must skip it."""
    bus = pd.DataFrame({"idx": [1, 2, 3, 4, 5], "Vn": [380, 380, 380, 380, 21]})
    line = pd.DataFrame(
        {
            "idx": ["L0", "T0", "L1", "L2", "L3"],
            "u": 1,
            "bus1": [1, 4, 2, 2, 3],
            "bus2": [2, 5, 3, 3, 4],
            "Vn1": [380, 380, 380, 380, 380],
            "Vn2": [380, 21, 380, 380, 380],
            "r": [0.01, 0.0, 0.01, 0.01, 0.02],
            "x": [0.1, 0.05, 0.1, 0.1, 0.2],
            "b": [0.2, 0.0, 0.1, 0.1, 0.3],
            "g": 0.0,
            "b1": 0.0,
            "g1": 0.0,
            "b2": 0.0,
            "g2": 0.0,
            "trans": [0, 1, 0, 0, 0],
            "tap": [1.0, 1.02, 1.0, 1.0, 1.0],
            "phi": 0.0,
        }
    )
    genrou = pd.DataFrame({"idx": ["M0"], "u": 1, "bus": [5], "gen": ["G0"], "Sn": 200.0, "M": 8.0})
    gencls = pd.DataFrame({"idx": ["E0"], "u": 1, "bus": [3], "gen": ["G1"], "Sn": 100.0, "M": 4.0})
    pv = pd.DataFrame({"idx": ["G0", "G1"], "bus": [5, 3], "p0": [1.0, 0.3]})
    return {"Bus": bus, "Line": line, "GENROU": genrou, "GENCLS": gencls, "PV": pv}


@pytest.fixture
def grid() -> CaseGrid:
    return CaseGrid.from_sheets(_sheets(), np.ones(5), ANGLES, mva=MVA)


def _events(fault_line: list[int], fault_position: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case": "tiny_t10",
            "fault_line": [float(i) for i in fault_line],
            "fault_position": fault_position,
            "clearing_time_s": 0.1,
            "label": 0,
        },
        index=[10 * (k + 1) for k in range(len(fault_line))],
    )


def test_engine_order_skips_transformers(grid: CaseGrid) -> None:
    assert grid.fault_lines.tolist() == [0, 2, 3, 4]
    assert grid.n_fault_lines == 4
    assert grid.branches.loc[grid.fault_lines, "idx"].tolist() == ["L0", "L1", "L2", "L3"]


def test_faulted_bus_follows_fault_position(grid: CaseGrid) -> None:
    X = line_features(_events([0, 0, 0], [0.0, 0.49, 0.5]), {"tiny_t10": grid})
    assert X.index.tolist() == [10, 20, 30]
    # bus 1 has one line, bus 2 has three (L0, L1, L2)
    assert X["line|bus_degree"].tolist() == [1.0, 1.0, 3.0]
    assert X["line|bus_degree_lines"].tolist() == [1.0, 1.0, 3.0]
    assert np.allclose(X["line|bus_sc_strength"], [10.0, 10.0, 30.0])


def test_line_power_matches_hand_calculation(grid: CaseGrid) -> None:
    X = line_features(_events([0, 0], [0.0, 1.0]), {"tiny_t10": grid})
    g, b = np.real(1 / (0.01 + 0.1j)), np.imag(1 / (0.01 + 0.1j))
    th = ANGLES[0] - ANGLES[1]
    p1 = g - (g * np.cos(th) + b * np.sin(th))  # V = 1 pu at both ends
    q1 = -(b + 0.2 / 2) - (g * np.sin(th) - b * np.cos(th))
    p2 = g - (g * np.cos(-th) + b * np.sin(-th))
    assert np.isclose(X.loc[10, "line|p_mw"], p1 * MVA)
    assert np.isclose(X.loc[10, "line|q_mvar"], q1 * MVA)
    assert np.isclose(X.loc[20, "line|p_mw"], p2 * MVA)
    assert X.loc[10, "line|p_mw"] > 0 > X.loc[20, "line|p_mw"]  # flows from bus 1 to bus 2
    assert np.isclose(X.loc[10, "line|delta_deg"], np.degrees(0.1))
    assert np.isclose(X.loc[10, "line|y_series"], 1 / np.hypot(0.01, 0.1))
    assert np.isclose(X.loc[10, "line|r_x"], 0.1)


def test_line_power_both_ends_balance_losses() -> None:
    lines = pd.DataFrame(
        {
            "r": [0.01],
            "x": [0.1],
            "b": [0.0],
            "g": [0.0],
            "b1": 0.0,
            "g1": 0.0,
            "b2": 0.0,
            "g2": 0.0,
        }
    )
    p1, q1, p2, q2 = line_power(np.r_[1.0], np.r_[0.0], np.r_[0.98], np.r_[-0.2], lines)
    i2 = ((p1 + p2) / 0.01) ** 0.5  # the loss is r * |I|^2 and the var loss x * |I|^2
    assert np.isclose(q1 + q2, 0.1 * i2**2)


def test_hops_inertia_and_nearest_machine(grid: CaseGrid) -> None:
    # fault on L0 at bus 1 and at bus 2; machines at bus 3 (H*Sn = 200) and 5 (800)
    X = line_features(_events([0, 0], [0.0, 1.0]), {"tiny_t10": grid})
    assert X["line|machine_hops"].tolist() == [2.0, 1.0]
    assert np.allclose(X["line|machine_x_dist"], [0.2, 0.1])  # parallel 2-3 counts once
    assert X["line|kinetic_mws_1hop"].tolist() == [0.0, 200.0]
    assert X["line|kinetic_mws_2hop"].tolist() == [200.0, 200.0]
    assert X["line|kinetic_mws_3hop"].tolist() == [200.0, 1000.0]
    assert np.allclose(X["line|gen_mw_3hop"], [30.0, 130.0])


def test_parallel_circuits_and_other_lines(grid: CaseGrid) -> None:
    X = line_features(_events([0, 1, 2, 3], [1.0, 0.0, 0.0, 1.0]), {"tiny_t10": grid})
    assert X["line|n_circuits"].tolist() == [1.0, 2.0, 2.0, 1.0]
    # L0 faulted at bus 2: the other lines at bus 2 are L1/L2 (0.1 rad); nothing at bus 1
    assert np.isclose(X.loc[10, "line|other_delta_max_deg"], np.degrees(0.1))
    # L3 (0.05 rad) at bus 4: the other line at bus 3 is L1/L2 (0.1 rad); bus 4 has T0 only
    assert np.isclose(X.loc[40, "line|other_delta_max_deg"], np.degrees(0.1))
    assert (X["line|vn_kv"] == 380).all()
    assert (X["line|bus_vn_kv"] == 380).all()


def test_loading_is_relative_to_the_voltage_level(grid: CaseGrid) -> None:
    X = line_features(_events([0, 1, 2, 3], [0.0, 0.0, 0.0, 0.0]), {"tiny_t10": grid})
    assert (X["line|loading"] > 0).all()
    assert X["line|loading"].max() >= 1.0  # the most loaded line exceeds the 95th percentile


def test_no_missing_values_and_categorical(grid: CaseGrid) -> None:
    X = line_features(_events([3, 1, 0], [0.0, 1.0, 0.3]), {"tiny_t10": grid})
    assert not X.isna().any().any()
    assert all(c.startswith("line|") for c in X.columns)
    assert isinstance(X["line|fault_line_cat"].dtype, pd.CategoricalDtype)
    assert X["line|fault_line_cat"].tolist() == [3, 1, 0]
    assert "label" not in {c.removeprefix("line|") for c in X.columns}


def test_leakage_guard_covers_line_prefix() -> None:
    with pytest.raises(DataError, match="outcome"):
        check_no_leakage(["line|x", "line|label"])
    check_no_leakage(["line|x", "line|p_mw"])


def test_bad_inputs_are_errors(grid: CaseGrid) -> None:
    with pytest.raises(DataError, match="no case grid"):
        line_features(_events([0], [0.0]), {})
    with pytest.raises(DataError, match="out of range"):
        line_features(_events([4], [0.0]), {"tiny_t10": grid})
    with pytest.raises(DataError, match="voltages"):
        CaseGrid.from_sheets(_sheets(), np.ones(4), ANGLES)
