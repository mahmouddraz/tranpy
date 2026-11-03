from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tranpy.conventional.eac import (
    EqualAreaCriterion,
    SmibCurve,
    SmibEvent,
    critical_clearing_angle,
    critical_clearing_time,
    smib_curve,
)

NO_POWER = SmibCurve(0.0, 0.0, 0.0)


def test_smib_curve_of_two_machines_is_the_textbook_power_angle_curve() -> None:
    # Y'12 = +j2 (x = 0.5 pu): Pe = E1 E2 / x sin(d1 - d2) = 2.2 sin(d1) with E1 = 1.1, d2 = 0
    curve = smib_curve([[-2j, 2j], [2j, -2j]], [1.1, 1.0], [0.0, 0.0], 0)
    assert math.isclose(curve.pc, 0.0, abs_tol=1e-12)
    assert math.isclose(curve.pmax, 2.2)
    assert math.isclose(curve.power(math.pi / 2), 2.2)


def test_bolted_terminal_fault_matches_the_closed_form_critical_angle() -> None:
    # Classic closed form for a fault that removes all transfer (Pe = 0 during the fault)
    # with the pre-fault network restored on clearing: cos(dc) = (pi - 2 d0) sin(d0) - cos(d0),
    # e.g. Grainger and Stevenson, Power System Analysis (1994), Ch. 16; Kundur (1994), Sec. 13.1.
    pmax, pm = 2.0, 1.0
    d0 = math.asin(pm / pmax)
    post = SmibCurve(0.0, pmax, 0.0)
    dc = critical_clearing_angle(pm, NO_POWER, post, d0)
    expected = math.acos((math.pi - 2 * d0) * math.sin(d0) - math.cos(d0))
    assert math.isclose(dc, expected, rel_tol=1e-8)
    # with Pe = 0 the accelerating power is exactly constant, so Eq. (5) is exact here
    tc = critical_clearing_time(d0, dc, pm, inertia_s=5.0, frequency_hz=50.0)
    assert math.isclose(tc, math.sqrt(4 * 5.0 * (dc - d0) / (2 * math.pi * 50.0 * pm)))


def test_no_acceleration_means_always_stable_and_no_post_fault_equilibrium_means_never() -> None:
    post = SmibCurve(0.0, 2.0, 0.0)
    d0 = math.asin(0.5)
    assert critical_clearing_angle(1.0, SmibCurve(0.0, 3.0, 0.0), post, d0) == math.inf
    assert critical_clearing_angle(1.0, NO_POWER, SmibCurve(0.0, 0.5, 0.0), d0) == d0


@settings(max_examples=60, deadline=None)
@given(
    pmax=st.floats(1.0, 3.0),
    share=st.floats(0.0, 0.6),
    pm1=st.floats(0.05, 0.85),
    step=st.floats(0.01, 0.1),
)
def test_cct_shrinks_as_mechanical_loading_rises(
    pmax: float, share: float, pm1: float, step: float
) -> None:
    eac = EqualAreaCriterion(50.0)
    during = SmibCurve(0.0, share * pmax, 0.0)
    post = SmibCurve(0.0, pmax, 0.0)

    def cct(fraction: float) -> float:
        pm = fraction * pmax
        d0 = math.asin(fraction)
        return eac.cct(SmibEvent(0, pm, d0, 5.0, during, post, 0.1))

    lo, hi = cct(pm1), cct(min(pm1 + step, 0.95))
    assert hi <= lo + 1e-9


def test_predict_labels_unstable_when_clearing_exceeds_cct() -> None:
    post = SmibCurve(0.0, 2.0, 0.0)
    d0 = math.asin(0.5)
    events = [
        SmibEvent(i, 1.0, d0, 5.0, NO_POWER, post, clearing)
        for i, clearing in enumerate([0.05, 1.0])
    ]
    out = EqualAreaCriterion(50.0).predict(events)
    np.testing.assert_array_equal(out.label, [0, 1])
    assert np.all(out.compute_s >= 0)
