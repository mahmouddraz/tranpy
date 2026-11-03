"""M1: equal-area criterion and critical clearing time (paper Section I, Eqs. 1-5).

Single-machine-infinite-bus (SMIB) reduction: the faulted machine ``k`` swings against the
rest of the system, whose internal EMFs are held at their pre-fault angles. Its electrical
power on the reduced network ``Y'`` (Eq. 1) is then ``Pc + Pmax sin(delta + phi)`` for each
network state (during fault, after clearing).

The critical clearing angle equates the accelerating and decelerating areas up to the
unstable equilibrium of the post-fault curve. The critical clearing time follows Eq. (5)
with the accelerating power held at its value at ``delta_0`` (the paper's simplification):
``t_c = sqrt(4 H (delta_c - delta_0) / (omega_s P_a))``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq

from tranpy_core.errors import ConfigurationError

__all__ = [
    "ConventionalPrediction",
    "EqualAreaCriterion",
    "SmibCurve",
    "SmibEvent",
    "critical_clearing_angle",
    "critical_clearing_time",
    "smib_curve",
]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SmibCurve:
    """Electrical power of the SMIB equivalent: ``pc + pmax * sin(delta + phi)`` (per unit)."""

    pc: float
    pmax: float
    phi: float

    def power(self, delta: float) -> float:
        return self.pc + self.pmax * math.sin(delta + self.phi)

    def integral(self, a: float, b: float) -> float:
        """Integral of :meth:`power` from ``a`` to ``b``."""
        return self.pc * (b - a) + self.pmax * (math.cos(a + self.phi) - math.cos(b + self.phi))


def smib_curve(
    y_reduced: npt.ArrayLike, emf_pu: npt.ArrayLike, delta_rad: npt.ArrayLike, machine: int
) -> SmibCurve:
    """SMIB curve of ``machine`` on ``y_reduced`` with the other machines at ``delta_rad``.

    ``Pe_k = E_k^2 G_kk + sum_j E_k E_j (G_kj cos(d_k - d_j) + B_kj sin(d_k - d_j))``.
    """
    y = np.asarray(y_reduced, dtype=np.complex128)
    e = np.asarray(emf_pu, dtype=np.float64)
    d = np.asarray(delta_rad, dtype=np.float64)
    k = machine
    a = b = 0.0
    for j in range(e.size):
        if j == k:
            continue
        g_kj, b_kj = y[k, j].real, y[k, j].imag
        a += e[k] * e[j] * (g_kj * math.cos(d[j]) - b_kj * math.sin(d[j]))
        b += e[k] * e[j] * (g_kj * math.sin(d[j]) + b_kj * math.cos(d[j]))
    return SmibCurve(pc=e[k] ** 2 * y[k, k].real, pmax=math.hypot(a, b), phi=math.atan2(a, b))


def critical_clearing_angle(
    pm_pu: float, during: SmibCurve, post: SmibCurve, delta0_rad: float
) -> float:
    """Clearing angle at which the accelerating and decelerating areas are equal.

    Returns ``delta0_rad`` when no clearing can save the machine (the post-fault curve
    never reaches ``pm_pu``, or its whole decelerating area is too small), and ``inf`` when
    the machine does not accelerate during the fault.
    """
    if post.pmax <= 0 or abs(pm_pu - post.pc) > post.pmax:
        return delta0_rad
    delta_u = math.pi - math.asin((pm_pu - post.pc) / post.pmax) - post.phi
    if delta_u <= delta0_rad:
        return delta0_rad

    def balance(delta_c: float) -> float:
        accel = pm_pu * (delta_c - delta0_rad) - during.integral(delta0_rad, delta_c)
        decel = post.integral(delta_c, delta_u) - pm_pu * (delta_u - delta_c)
        return accel - decel

    low, high = balance(delta0_rad), balance(delta_u)
    if low >= 0:
        return delta0_rad
    if high <= 0:
        return math.inf
    return float(brentq(balance, delta0_rad, delta_u, xtol=1e-10))


def critical_clearing_time(
    delta0_rad: float,
    delta_c_rad: float,
    accelerating_power_pu: float,
    inertia_s: float,
    frequency_hz: float,
) -> float:
    """Eq. (5): ``sqrt(4 H (delta_c - delta_0) / (omega_s P_a))``, ``P_a`` held constant."""
    if inertia_s <= 0 or frequency_hz <= 0:
        raise ConfigurationError("inertia and frequency must be positive")
    if math.isinf(delta_c_rad) or accelerating_power_pu <= 0:
        return math.inf
    omega_s = 2 * math.pi * frequency_hz
    return math.sqrt(
        4 * inertia_s * max(delta_c_rad - delta0_rad, 0.0) / (omega_s * accelerating_power_pu)
    )


@dataclass(frozen=True)
class SmibEvent:
    """One fault event reduced to the faulted machine's SMIB equivalent.

    Attributes:
        event_id: identifier of the event in its dataset.
        pm_pu: mechanical power of the machine on the system base.
        delta0_rad: pre-fault rotor angle.
        inertia_s: inertia constant ``H`` on the system base.
        during, post: electrical power curves during the fault and after clearing.
        clearing_s: actual fault duration of the event.
    """

    event_id: int
    pm_pu: float
    delta0_rad: float
    inertia_s: float
    during: SmibCurve
    post: SmibCurve
    clearing_s: float


@dataclass(frozen=True)
class ConventionalPrediction:
    """Predictions of a conventional method: label 1 is unstable, as in the ML pipeline.

    ``margin_s = cct_s - clearing_s`` (positive: predicted stable); ``compute_s`` is the
    wall-clock time per event.
    """

    event_id: npt.NDArray[np.int64]
    label: npt.NDArray[np.int64]
    margin_s: FloatArray
    cct_s: FloatArray
    compute_s: FloatArray


@dataclass(frozen=True)
class EqualAreaCriterion:
    """Equal-area criterion / critical clearing time (EAC/CCT), paper Eqs. (1)-(5).

    Attributes:
        nominal_frequency_hz: system frequency, for ``omega_s``.
    """

    nominal_frequency_hz: float = 50.0

    def __post_init__(self) -> None:
        if self.nominal_frequency_hz <= 0:
            raise ConfigurationError("nominal_frequency_hz must be positive")

    def cct(self, event: SmibEvent) -> float:
        delta_c = critical_clearing_angle(event.pm_pu, event.during, event.post, event.delta0_rad)
        pa = event.pm_pu - event.during.power(event.delta0_rad)
        return critical_clearing_time(
            event.delta0_rad, delta_c, pa, event.inertia_s, self.nominal_frequency_hz
        )

    def predict(self, events: Sequence[SmibEvent]) -> ConventionalPrediction:
        """Per-event loop: each SMIB problem is a scalar root search (no time integration)."""
        cct = np.empty(len(events))
        spent = np.empty(len(events))
        for i, event in enumerate(events):
            start = time.perf_counter()
            cct[i] = self.cct(event)
            spent[i] = time.perf_counter() - start
        clearing = np.array([e.clearing_s for e in events], dtype=np.float64)
        margin = cct - clearing
        return ConventionalPrediction(
            event_id=np.array([e.event_id for e in events], dtype=np.int64),
            label=(margin <= 0).astype(np.int64),
            margin_s=margin,
            cct_s=cct,
            compute_s=spent,
        )
