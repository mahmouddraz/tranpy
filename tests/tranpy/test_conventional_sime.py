"""SIME: OMIB equivalent, stability margin and critical machines on synthetic trajectories."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tranpy.conventional.energy import cct_normalised_energy
from tranpy.conventional.sime import (
    critical_cluster,
    inertia_coefficient,
    omib,
    sime_margin,
)
from tranpy_core.errors import ConfigurationError, DataError

# Two-machine lossless classical system: machine 1 against a heavy machine 2 through a
# reactance; P_e1 = P_max sin(delta_1 - delta_2), P_e2 = -P_e1, P_m2 = -P_m1. Its OMIB is the
# SMIB  M d^2 delta/dt^2 = P_m1 - P_max sin delta  with M = M_1 M_2 / (M_1 + M_2).
M1, M2 = 0.04, 0.4  # power s^2/rad (per unit)
PM, PMAX = 0.8, 1.6


def _two_machine_trajectory(
    delta0: float, omega0: float, t_end: float = 2.0, dt: float = 0.002
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Post-clearing trajectory (RK4) from OMIB state ``(delta0, omega0)``; machine 2 at rest."""
    m_eq = M1 * M2 / (M1 + M2)

    def f(x: np.ndarray) -> np.ndarray:
        d, w = x
        return np.array([w, (PM - PMAX * np.sin(d)) / m_eq])

    n = round(t_end / dt) + 1
    xs = np.empty((n, 2))
    x = np.array([delta0, omega0])
    for k in range(n):
        xs[k] = x
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    t = np.arange(n) * dt
    # split the relative motion between the machines in inverse proportion to inertia
    delta = np.column_stack([xs[:, 0] * M2 / (M1 + M2), -xs[:, 0] * M1 / (M1 + M2)])
    omega = np.column_stack([xs[:, 1] * M2 / (M1 + M2), -xs[:, 1] * M1 / (M1 + M2)])
    pe1 = PMAX * np.sin(xs[:, 0])
    pe = np.column_stack([pe1, -pe1])
    pm = np.array([PM, -PM])
    return t, delta, omega, pm, pe


def _analytic_margin(delta0: float, omega0: float) -> float:
    """Equal-area margin: decelerating area to delta_u minus the kinetic energy at clearing."""
    m_eq = M1 * M2 / (M1 + M2)
    delta_u = np.pi - np.arcsin(PM / PMAX)
    a_dec = PMAX * (np.cos(delta0) - np.cos(delta_u)) - PM * (delta_u - delta0)
    return float(a_dec - 0.5 * m_eq * omega0**2)


def test_omib_of_two_machines_is_the_analytic_smib() -> None:
    t, delta, omega, pm, pe = _two_machine_trajectory(0.9, 3.0, t_end=0.2)
    o = omib(delta, omega, pm, pe, [M1, M2], [True, False], time=t)
    assert o.m_eq == pytest.approx(M1 * M2 / (M1 + M2))
    np.testing.assert_allclose(o.delta, delta[:, 0] - delta[:, 1])
    np.testing.assert_allclose(o.pm, PM)
    np.testing.assert_allclose(o.pe, PMAX * np.sin(o.delta))
    np.testing.assert_allclose(o.pa, PM - PMAX * np.sin(o.delta))


def test_unstable_margin_matches_equal_area_criterion() -> None:
    delta0, omega0 = 1.0, 9.0
    expected = _analytic_margin(delta0, omega0)
    assert expected < 0
    t, delta, omega, pm, pe = _two_machine_trajectory(delta0, omega0)
    r = sime_margin(delta, omega, pm, pe, [M1, M2], time=t)
    assert r.kind == "unstable" and r.unstable
    assert r.variant == "recorded_pm"
    assert r.eta == pytest.approx(expected, rel=1e-3)
    assert r.delta_u == pytest.approx(np.pi - np.arcsin(PM / PMAX), abs=2e-3)
    assert r.t_u > 0 and np.isnan(r.t_r)
    assert r.critical.tolist() == [True, False]


def test_stable_margin_is_positive_with_the_return_point_and_extrapolated_delta_u() -> None:
    delta0, omega0 = 1.0, 6.5
    assert _analytic_margin(delta0, omega0) > 0
    t, delta, omega, pm, pe = _two_machine_trajectory(delta0, omega0)
    r = sime_margin(delta, omega, pm, pe, [M1, M2], time=t)
    assert r.kind == "stable" and not r.unstable
    assert r.eta > 0 and np.isnan(r.t_u)
    # return point: omega = 0 where the energy is exhausted, P_a < 0 there
    assert r.pa_r < 0
    delta_r = (delta[:, 0] - delta[:, 1])[np.argmax(delta[:, 0] - delta[:, 1])]
    assert r.delta_r == pytest.approx(delta_r, abs=1e-3)
    # the linear extrapolation of P_a(delta) to zero lands near the unstable equilibrium
    # (P_a(delta) is convex there, so the tangent overshoots the unstable equilibrium)
    assert r.delta_u == pytest.approx(np.pi - np.arcsin(PM / PMAX), abs=0.3)
    # the triangle (2.20) is an approximation of the remaining area, of the same order
    assert r.eta == pytest.approx(_analytic_margin(delta0, omega0), rel=0.5)


def test_margin_sign_tracks_the_equal_area_criterion_across_clearing_speeds() -> None:
    for omega0 in np.linspace(3.0, 10.0, 15):
        t, delta, omega, pm, pe = _two_machine_trajectory(1.0, float(omega0))
        r = sime_margin(delta, omega, pm, pe, [M1, M2], time=t)
        assert r.unstable == (_analytic_margin(1.0, float(omega0)) < 0)
        if r.unstable:
            assert r.eta < 0
        else:
            # a positive margin, or undefined (NaN) when the return point is so far from the
            # unstable equilibrium that P_a(delta) still falls there (no extrapolation)
            assert np.isnan(r.eta) or r.eta > 0


def test_trajectory_variant_agrees_with_the_recorded_powers() -> None:
    t, delta, omega, pm, pe = _two_machine_trajectory(1.0, 9.0)
    with_pm = sime_margin(delta, omega, pm, pe, [M1, M2], time=t)
    from_traj = sime_margin(delta, omega, None, None, [M1, M2], time=t)
    assert from_traj.variant == "trajectory"
    assert from_traj.kind == "unstable"
    assert from_traj.eta == pytest.approx(with_pm.eta, rel=0.02)
    no_speed = sime_margin(delta, None, None, None, [M1, M2], time=t)
    assert no_speed.kind == "unstable"
    assert no_speed.eta == pytest.approx(with_pm.eta, rel=0.05)


def test_stable_trajectory_cut_before_the_return_is_extrapolated() -> None:
    t, delta, omega, pm, pe = _two_machine_trajectory(1.0, 6.5)
    rel = omega[:, 0] - omega[:, 1]
    peak = int(np.flatnonzero((rel[:-1] > 0) & (rel[1:] <= 0))[0])  # first return
    cut = slice(0, peak - 20)
    r = sime_margin(delta[cut], omega[cut], pm, pe[cut], [M1, M2], time=t[cut])
    assert r.kind == "stable_extrapolated"
    assert r.eta > 0
    full = sime_margin(delta, omega, pm, pe, [M1, M2], time=t)
    assert r.eta == pytest.approx(full.eta, rel=0.5)


def test_critical_cluster_recovers_the_separating_group() -> None:
    # machines 0 and 2 run away together (decelerating first, then accelerating: P_a crosses
    # zero upwards while omega > 0); the others stay near their equilibrium with small swings
    t = np.linspace(0.0, 1.5, 151)
    m = np.array([0.02, 0.05, 0.03, 0.08, 0.06])
    omega_c = 6.0 - 12.0 * t + 10.0 * t**2
    delta_c = np.cumsum(np.r_[0.0, 0.5 * (omega_c[1:] + omega_c[:-1]) * np.diff(t)])
    rng = np.random.default_rng(1)
    delta = np.zeros((t.size, 5))
    omega = np.zeros((t.size, 5))
    for j in (1, 3, 4):
        phase = rng.uniform(0, np.pi)
        delta[:, j] = 0.01 * np.sin(2 * np.pi * 1.2 * t + phase) + 0.05 * j
        omega[:, j] = 0.01 * 2 * np.pi * 1.2 * np.cos(2 * np.pi * 1.2 * t + phase)
    for j in (0, 2):
        delta[:, j] = 1.0 + delta_c + 0.02 * j
        omega[:, j] = omega_c
    cluster = critical_cluster(delta, m, time=t, omega_t=omega, n_gaps=2)
    assert cluster.critical.tolist() == [True, False, True, False, False]
    assert cluster.kind == "unstable"
    r = sime_margin(delta, omega, None, None, m, time=t, n_gaps=2)
    assert r.kind == "unstable" and r.eta < 0
    assert r.t_u == pytest.approx(0.6, abs=0.02)  # d omega/dt = -12 + 20 t = 0
    assert r.cluster.n_candidates >= 1


def test_critical_cluster_of_a_stable_swing_is_the_widest_gap_at_the_return() -> None:
    # the separating pair decelerates until it returns at t = 0.4 s with P_a still negative
    # but rising towards zero (a near-critical first swing); the others swing back slightly
    t = np.linspace(0.0, 1.0, 201)
    m = np.array([0.02, 0.05, 0.03, 0.08])
    delta = np.zeros((t.size, 4))
    omega = np.zeros((t.size, 4))
    w = 10.0 * (0.4 - t) * (0.9 - t)
    swing = np.cumsum(np.r_[0.0, 0.5 * (w[1:] + w[:-1]) * np.diff(t)])
    delta[:, 0] = 1.5 + swing
    delta[:, 1] = 1.4 + 0.9 * swing
    omega[:, 0] = w
    omega[:, 1] = 0.9 * w
    delta[:, 2] = -0.5 - 0.3 * swing
    omega[:, 2] = -0.3 * w
    delta[:, 3] = -0.4 - 0.2 * swing
    omega[:, 3] = -0.2 * w
    cluster = critical_cluster(delta, m, time=t, omega_t=omega)
    assert cluster.critical.tolist() == [True, True, False, False]
    assert cluster.kind == "stable"
    r = sime_margin(delta, omega, None, None, m, time=t)
    assert r.kind == "stable" and r.eta > 0
    assert r.t_r == pytest.approx(0.4, abs=0.01)


def test_given_critical_set_and_argument_checks() -> None:
    t, delta, omega, pm, pe = _two_machine_trajectory(1.0, 9.0, t_end=0.5)
    r = sime_margin(delta, omega, pm, pe, [M1, M2], time=t, critical=[True, False])
    assert r.cluster.kind == "given"
    with pytest.raises(DataError):
        omib(delta, omega, pm, pe, [M1, M2], [True, True], time=t)
    with pytest.raises(ConfigurationError):
        omib(delta, omega, pm, None, [M1, M2], [True, False], time=t)
    with pytest.raises(ConfigurationError):
        omib(delta, omega, pm, pe, [M1], [True, False], time=t)
    with pytest.raises(ConfigurationError):
        sime_margin(delta, None, None, None, [M1, M2], time=np.zeros_like(t))


def test_inertia_coefficient_units() -> None:
    # H Sn = 2364 MW s at 50 Hz: M = 2 * 2364 / (2 pi 50) MW s^2/rad
    assert inertia_coefficient(2364.0, 50.0) == pytest.approx(2 * 2364.0 / (2 * np.pi * 50))


def test_cct_normalised_energy_interpolates_the_sweep_at_the_bracket_midpoint() -> None:
    # sweep of line 7 in case A: KE grows linearly with clearing time, CCT bracketed in
    # [0.2, 0.3] so KE_cct = KE(0.25) = 2.5; line 9 is stable at every time (no bracket)
    sweep = pd.DataFrame(
        {
            "case": ["A"] * 8,
            "fault_line": [7] * 4 + [9] * 4,
            "clearing_time_s": [0.1, 0.2, 0.3, 0.4] * 2,
            "ke": [1.0, 2.0, 3.0, 4.0, 0.5, 1.0, 1.5, 2.0],
            "label": [0, 0, 1, 1, 0, 0, 0, 0],
        }
    )
    random = pd.DataFrame(
        {
            "case": ["A", "A", "B"],
            "fault_line": [7, 9, 7],
            "clearing_time_s": [0.26, 0.15, 0.26],
            "ke": [5.0, 0.7, 5.0],
            "label": [1, 0, 1],
        }
    )
    events = pd.concat([sweep, random], ignore_index=True)
    brackets = pd.DataFrame(
        {
            "case": ["A", "A"],
            "fault_line": [7, 9],
            "t_stable_max": [0.2, 0.4],
            "t_unstable_min": [0.3, np.nan],
            "status": ["bracketed", "above_sweep"],
        }
    )
    ratio = cct_normalised_energy(
        events["ke"],
        brackets,
        events["case"],
        events["fault_line"],
        events["clearing_time_s"],
        events.index < len(sweep),
    )
    np.testing.assert_allclose(ratio[:4], np.array([1.0, 2.0, 3.0, 4.0]) / 2.5)
    assert np.isnan(ratio[4:8]).all()  # no bracket on line 9
    assert ratio[8] == pytest.approx(2.0)  # random event on the bracketed line
    assert np.isnan(ratio[9]) and np.isnan(ratio[10])  # unbracketed line, other case
    with pytest.raises(DataError):
        cct_normalised_energy(
            events["ke"],
            brackets.drop(columns="status"),
            events["case"],
            events["fault_line"],
            events["clearing_time_s"],
            events.index < len(sweep),
        )
