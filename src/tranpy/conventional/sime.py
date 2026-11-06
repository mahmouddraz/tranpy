"""SIME: the single-machine equivalent of a multi-machine trajectory and its stability margin.

SIME (Pavella, Ernst and Ruiz-Vega (2000), *Transient Stability of Power Systems: A Unified
Approach to Assessment and Control*, Kluwer, Ch. 2; Zhang, Wehenkel, Rousseaux and Pavella
(1997), "SIME: A hybrid approach to fast transient stability assessment and contingency
selection", Int. J. Electr. Power Energy Syst. 19(3), 195-208) replaces the multi-machine
post-fault trajectory by a one-machine infinite-bus (OMIB) equivalent built from the
critical machines (CMs, the group that separates) and the non-critical machines (NMs).
Equation numbers below are those of the book, Ch. 2 (§2.2, "Foundations").

OMIB equivalent (:func:`omib`), with ``M_k`` the inertia coefficients:

* ``M_C = sum_C M_k``, ``M_N = sum_N M_j``, ``M = M_C M_N / (M_C + M_N)``  (2.7)
* ``delta_C = (1/M_C) sum_C M_k delta_k``, ``delta_N = (1/M_N) sum_N M_j delta_j``  (2.3-2.4)
* ``delta = delta_C - delta_N``, ``omega = omega_C - omega_N``  (2.5-2.6)
* ``P_m = M [(1/M_C) sum_C P_mk - (1/M_N) sum_N P_mj]``  (2.8)
* ``P_e = M [(1/M_C) sum_C P_ek - (1/M_N) sum_N P_ej]``  (2.9)
* ``P_a = P_m - P_e``, so that ``M d^2 delta / dt^2 = P_a``  (2.10-2.11)

Stability conditions and margin (:func:`sime_margin`):

* unstable: the OMIB reaches ``P_a(t_u) = 0`` with ``dP_a/dt > 0`` (2.15-2.16) while still
  accelerating (``omega > 0``); the margin is ``eta = A_dec - A_acc = -1/2 M omega_u^2``
  (2.12-2.14), negative.
* stable: the OMIB returns, ``omega(t_r) = 0`` with ``P_a(t_r) < 0`` (2.17-2.18); the
  margin is the decelerating area not used, ``eta = int_{delta_r}^{delta_u} |P_a| d delta``
  (2.19), which SIME approximates by the triangle ``1/2 |P_a(t_r)| (delta_u - delta_r)``
  (2.20) with ``delta_u`` from a linear extrapolation of ``P_a(delta)`` to zero (§2.2.2).
  If the stored trajectory stops before the return point while still decelerating
  (``omega > 0``, ``P_a < 0``), the same extrapolation gives the remaining decelerating area
  and ``eta`` is that area minus the kinetic energy left, ``1/2 M omega_e^2``; the result
  flags this as ``"stable_extrapolated"``.

Critical machines (:func:`critical_cluster`, book §2.2.3): along the post-clearing
trajectory the machines are sorted by rotor angle at every step; each of the largest angular
gaps between adjacent machines is a candidate decomposition (the machines above the gap
are the CMs). The candidate whose OMIB meets the unstable conditions first is retained. If
no candidate meets them (stable case), the decomposition with the widest gap at its own
OMIB's return point is retained.

Units: angles in rad, speeds in rad/s (deviation from synchronous, or any common reference,
which cancels in the differences), powers in one power unit (e.g. MW) and ``M`` in
``power s^2/rad``, so ``M = 2 H S_n / omega_s`` (:func:`inertia_coefficient`). The margin
is then an area of the ``P_a``-``delta`` plane in ``power * rad`` (MW rad with MW), as in the
book, where everything is in per unit on the system base.

When the mechanical powers are not recorded (``pm=None``) the accelerating power is taken
from the trajectory itself, ``P_a = M d omega / dt`` by finite differences of the OMIB speed
(2.11), which includes governor and damping effects; this is the "trajectory-based" variant
and :class:`SimeResult.variant` says which one was used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from tranpy_core.errors import ConfigurationError, DataError

__all__ = [
    "CriticalCluster",
    "OmibTrajectory",
    "SimeResult",
    "critical_cluster",
    "inertia_coefficient",
    "omib",
    "sime_margin",
]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
Kind = Literal["unstable", "stable", "stable_extrapolated", "undecided"]
Margin = tuple[Kind, float, float, float, float, float, float, float]
"""``(kind, eta, t_u, delta_u, omega_u, t_r, delta_r, pa_r)``."""
Arrays = tuple[
    FloatArray, FloatArray | None, FloatArray | None, FloatArray | None, FloatArray, FloatArray
]


def inertia_coefficient(kinetic: npt.ArrayLike, f0: float = 50.0) -> FloatArray:
    """``M = 2 H S_n / omega_s`` (power s^2/rad) from ``H S_n`` (power s) and frequency ``f0``."""
    return np.asarray(2.0 * np.asarray(kinetic, dtype=np.float64) / (2.0 * np.pi * f0))


@dataclass(frozen=True)
class OmibTrajectory:
    """OMIB equivalent along a trajectory (book eqs 2.3-2.10); one entry per time step."""

    delta: FloatArray
    omega: FloatArray
    pm: FloatArray
    pe: FloatArray
    pa: FloatArray
    m_eq: float
    m_c: float
    m_n: float
    critical: BoolArray


@dataclass(frozen=True)
class CriticalCluster:
    """Critical machines retained by :func:`critical_cluster` and why."""

    critical: BoolArray
    kind: str
    """``"unstable"`` (first candidate meeting 2.15-2.16), ``"stable"`` (widest gap at the return),
    ``"undecided"`` (no candidate decided; widest gap overall) or ``"given"``."""
    t_index: int
    """Index of the time step at which the retained decomposition was formed."""
    n_candidates: int
    gap_rad: float
    """Angular gap between the last CM and the first NM at ``t_index``."""


@dataclass(frozen=True)
class SimeResult:
    """Stability margin of one event (book eqs 2.14 and 2.20)."""

    eta: float
    kind: Kind
    variant: str
    """``"recorded_pm"`` (P_a = P_m - P_e) or ``"trajectory"`` (P_a = M d omega/dt)."""
    t_u: float
    delta_u: float
    omega_u: float
    t_r: float
    delta_r: float
    pa_r: float
    m_eq: float
    critical: BoolArray
    cluster: CriticalCluster
    omib: OmibTrajectory

    @property
    def unstable(self) -> bool:
        return self.kind == "unstable"


def _arrays(
    delta_t: npt.ArrayLike,
    omega_t: npt.ArrayLike | None,
    pm: npt.ArrayLike | None,
    pe_t: npt.ArrayLike | None,
    m: npt.ArrayLike,
    time: npt.ArrayLike | None,
) -> Arrays:
    d = np.asarray(delta_t, dtype=np.float64)
    if d.ndim != 2:
        raise ConfigurationError("delta_t must be (time steps, machines)")
    n_t, n = d.shape
    mm = np.asarray(m, dtype=np.float64)
    if mm.shape != (n,):
        raise ConfigurationError(f"M must have one entry per machine ({n}), got {mm.shape}")
    if np.any(mm <= 0):
        raise ConfigurationError("inertia coefficients must be positive")
    t = np.arange(n_t, dtype=np.float64) if time is None else np.asarray(time, dtype=np.float64)
    if t.shape != (n_t,):
        raise ConfigurationError("time must have one entry per time step")
    w = None if omega_t is None else np.asarray(omega_t, dtype=np.float64)
    if w is not None and w.shape != d.shape:
        raise ConfigurationError("omega_t and delta_t differ in shape")
    pe = None if pe_t is None else np.asarray(pe_t, dtype=np.float64)
    if pe is not None and pe.shape != d.shape:
        raise ConfigurationError("pe_t and delta_t differ in shape")
    p = None
    if pm is not None:
        p = np.asarray(pm, dtype=np.float64)
        if p.shape == (n,):
            p = np.broadcast_to(p, d.shape)
        elif p.shape != d.shape:
            raise ConfigurationError("pm must be (machines,) or (time steps, machines)")
    if (p is None) != (pe is None):
        raise ConfigurationError("pm and pe_t go together: give both or neither")
    return d, w, p, pe, mm, t


def _omib_batch(
    d: FloatArray,
    w: FloatArray | None,
    p: FloatArray | None,
    pe: FloatArray | None,
    m: FloatArray,
    t: FloatArray,
    crit: BoolArray,
) -> list[OmibTrajectory]:
    """OMIB of every candidate decomposition (rows of ``crit``), eqs 2.3-2.11, in one pass."""
    c = crit.astype(np.float64)
    wc = c * m  # (K, n)
    wn = (1.0 - c) * m
    m_c = wc.sum(axis=1)
    m_n = wn.sum(axis=1)
    if np.any(m_c <= 0) or np.any(m_n <= 0):
        raise DataError("a decomposition leaves one group empty")
    m_eq = m_c * m_n / (m_c + m_n)  # (2.7)
    delta = d @ wc.T / m_c - d @ wn.T / m_n  # (2.3)-(2.5), (T, K)
    if (w is None or p is None) and np.any(np.diff(t) <= 0):
        raise ConfigurationError("time must be strictly increasing for the finite differences")
    # (2.6), or the time derivative of (2.5) when the speeds are not recorded
    omega = np.gradient(delta, t, axis=0) if w is None else w @ wc.T / m_c - w @ wn.T / m_n
    if p is None or pe is None:
        pa = m_eq[None, :] * np.gradient(omega, t, axis=0)  # (2.11), trajectory-based
        pm_eq = np.full_like(pa, np.nan)
        pe_eq = np.full_like(pa, np.nan)
    else:
        cn = 1.0 - c
        pm_eq = m_eq[None, :] * (p @ c.T / m_c - p @ cn.T / m_n)  # (2.8), plain group sums
        pe_eq = m_eq[None, :] * (pe @ c.T / m_c - pe @ cn.T / m_n)  # (2.9)
        pa = pm_eq - pe_eq  # (2.10)
    return [
        OmibTrajectory(
            delta=delta[:, k],
            omega=omega[:, k],
            pm=pm_eq[:, k],
            pe=pe_eq[:, k],
            pa=pa[:, k],
            m_eq=float(m_eq[k]),
            m_c=float(m_c[k]),
            m_n=float(m_n[k]),
            critical=crit[k].copy(),
        )
        for k in range(crit.shape[0])
    ]


def omib(
    delta_t: npt.ArrayLike,
    omega_t: npt.ArrayLike | None,
    pm: npt.ArrayLike | None,
    pe_t: npt.ArrayLike | None,
    M: npt.ArrayLike,
    critical: npt.ArrayLike,
    *,
    time: npt.ArrayLike | None = None,
) -> OmibTrajectory:
    """OMIB equivalent of a trajectory for the decomposition ``critical`` (book eqs 2.3-2.11).

    Args:
        delta_t: rotor angles (time steps x machines), rad.
        omega_t: rotor speeds (same shape), rad/s; ``None`` differentiates ``delta_t``.
        pm: mechanical powers, (machines,) if constant or (time steps x machines); ``None``
            with ``pe_t=None`` selects the trajectory-based ``P_a = M d omega/dt``.
        pe_t: electrical powers (time steps x machines), same unit as ``pm``.
        M: inertia coefficients (machines,), power s^2/rad.
        critical: boolean mask of the critical machines.
        time: time stamps of the steps (needed for the finite differences).
    """
    d, w, p, pe, m, t = _arrays(delta_t, omega_t, pm, pe_t, M, time)
    crit = np.asarray(critical, dtype=bool)
    if crit.shape != (d.shape[1],):
        raise ConfigurationError("critical must have one entry per machine")
    return _omib_batch(d, w, p, pe, m, t, crit[None, :])[0]


def _first_sign_change(x: FloatArray, upward: bool, start: int = 0) -> int | None:
    """First ``k > start`` with ``x[k-1] < 0 <= x[k]`` (upward) or ``x[k-1] > 0 >= x[k]``."""
    s = x[start:] if upward else -x[start:]
    hit = np.flatnonzero((s[:-1] < 0) & (s[1:] >= 0))
    return int(hit[0]) + 1 + start if hit.size else None


def _interp_zero(x: FloatArray, k: int, *ys: FloatArray) -> tuple[float, ...]:
    """Linear interpolation of ``ys`` at the zero of ``x`` between steps ``k-1`` and ``k``."""
    x0, x1 = x[k - 1], x[k]
    f = 0.0 if x1 == x0 else float(x0 / (x0 - x1))
    return tuple(float(y[k - 1] + f * (y[k] - y[k - 1])) for y in ys)


def _unstable_step(o: OmibTrajectory, start: int = 0) -> int | None:
    """First step after ``start`` meeting the unstable conditions (2.15)-(2.16), accelerating."""
    k = _first_sign_change(o.pa, upward=True, start=start)
    while k is not None:
        if o.omega[k] > 0:
            return k
        k = _first_sign_change(o.pa, upward=True, start=k)
    return None


def _return_step(o: OmibTrajectory, start: int = 0) -> int | None:
    """First step after ``start`` where ``omega`` crosses zero downwards with ``P_a < 0``."""
    k = _first_sign_change(o.omega, upward=False, start=start)
    return k if k is not None and o.pa[k] < 0 else None


def _extrapolated_delta_u(o: OmibTrajectory, stop: int, n_fit: int) -> float:
    """``delta_u`` from a linear fit of ``P_a(delta)`` over the ``n_fit`` steps before ``stop``.

    Returns NaN unless the fitted line rises towards zero (positive slope in ``delta``).
    """
    lo = max(0, stop - n_fit)
    x, y = o.delta[lo : stop + 1], o.pa[lo : stop + 1]
    if x.size < 2 or np.ptp(x) <= 0:
        return float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    if slope <= 0:
        return float("nan")
    return float(-intercept / slope)


def _margin(o: OmibTrajectory, t: FloatArray, n_fit: int, start: int = 0) -> Margin:
    """Stability margin and the instants that define it, of one OMIB trajectory.

    ``start`` is the step the test begins at (0 = clearing: the first swing).
    """
    nan = float("nan")
    ku = _unstable_step(o, start)
    kr = _return_step(o, start)
    if ku is not None and (kr is None or ku <= kr):
        t_u, d_u, w_u = _interp_zero(o.pa, ku, t, o.delta, o.omega)
        return "unstable", -0.5 * o.m_eq * w_u**2, t_u, d_u, w_u, nan, nan, nan  # (2.14)
    if kr is not None:
        t_r, d_r, pa_r = _interp_zero(o.omega, kr, t, o.delta, o.pa)
        d_u = _extrapolated_delta_u(o, kr - 1, n_fit)
        eta = 0.5 * abs(pa_r) * (d_u - d_r) if np.isfinite(d_u) and d_u > d_r else nan  # (2.20)
        return "stable", eta, nan, d_u, nan, t_r, d_r, pa_r
    last = o.pa.size - 1
    if o.omega[last] > 0 and o.pa[last] < 0:
        d_u = _extrapolated_delta_u(o, last, n_fit)
        if np.isfinite(d_u) and d_u > o.delta[last]:
            eta = 0.5 * abs(o.pa[last]) * (d_u - o.delta[last]) - 0.5 * o.m_eq * o.omega[last] ** 2
            if eta >= 0:
                return "stable_extrapolated", eta, nan, d_u, nan, nan, nan, nan
            # the kinetic energy left exceeds the extrapolated decelerating area
            w_e = float(o.omega[last])
            return "unstable", eta, float(t[last]), d_u, w_e, nan, nan, nan
    if o.omega[last] > 0 and np.all(o.pa >= 0):
        # never decelerates after clearing: no decelerating area at all (A_dec = 0 in 2.12)
        w_e = float(o.omega[last])
        eta = -0.5 * o.m_eq * w_e**2
        return "unstable", eta, float(t[last]), float(o.delta[last]), w_e, nan, nan, nan
    return "undecided", nan, nan, nan, nan, nan, nan, nan


def _candidates(
    d: FloatArray, n_gaps: int
) -> tuple[BoolArray, npt.NDArray[np.int64], FloatArray, BoolArray]:
    """Unique decompositions from the ``n_gaps`` largest gaps of the sorted angles at every step.

    Returns the masks (K x machines), the step each was first formed at, its gap there and
    ``active`` (steps x K): whether the decomposition is one of the top gaps at that step.
    """
    n_t, n = d.shape
    order = np.argsort(-d, axis=1, kind="stable")
    ranked = np.take_along_axis(d, order, axis=1)
    gaps = ranked[:, :-1] - ranked[:, 1:]  # gap below the i-th most advanced machine
    k_top = min(n_gaps, n - 1)
    top = np.argsort(-gaps, axis=1, kind="stable")[:, :k_top]
    seen: dict[bytes, int] = {}
    masks: list[BoolArray] = []
    steps: list[int] = []
    widths: list[float] = []
    active: list[list[int]] = []
    for it in range(n_t):
        for cut in top[it]:
            mask = np.zeros(n, dtype=bool)
            mask[order[it, : cut + 1]] = True
            key = np.packbits(mask).tobytes()
            if key not in seen:
                seen[key] = len(masks)
                masks.append(mask)
                steps.append(it)
                widths.append(float(gaps[it, cut]))
                active.append([])
            active[seen[key]].append(it)
    act = np.zeros((n_t, len(masks)), dtype=bool)
    for k, its in enumerate(active):
        act[its, k] = True
    return np.asarray(masks), np.asarray(steps, dtype=np.int64), np.asarray(widths), act


def _gap_at(d: FloatArray, mask: BoolArray, step: int) -> float:
    return float(d[step, mask].min() - d[step, ~mask].max())


def _select(
    d: FloatArray,
    w: FloatArray | None,
    p: FloatArray | None,
    pe: FloatArray | None,
    m: FloatArray,
    t: FloatArray,
    n_gaps: int,
    n_fit: int,
) -> tuple[CriticalCluster, OmibTrajectory, Margin]:
    if d.shape[1] < 2:
        raise DataError("SIME needs at least two machines")
    masks, steps, widths, active = _candidates(d, n_gaps)
    omibs = _omib_batch(d, w, p, pe, m, t, masks)
    # every candidate is tested on its first swing from clearing (step 0): a candidate examined
    # only from the step its gap appears at meets (2.15)-(2.16) trivially when that step falls
    # in an arbitrary phase of a damped later swing (535 of 1,000 stable grid events, with
    # omega_u ~ 0.02 rad/s and t_u 1-6 s after clearing, before this rule)
    margins = [_margin(o, t, n_fit) for o in omibs]
    # ... and its unstable crossing counts only at a step where it is one of the top-gap
    # decompositions (book §2.2.3 examines the candidates of the current instant): a cluster
    # that is not separating at t_u crosses P_a = 0 with omega ~ 0 by numerical wobble
    unstable = []
    for k, mg in enumerate(margins):
        if mg[0] != "unstable":
            continue
        step_u = int(np.searchsorted(t, mg[2], side="right") - 1)
        if active[min(step_u + 1, active.shape[0] - 1), k] or active[step_u, k]:
            unstable.append((mg[2], k))
    if unstable:
        # first candidate to meet the unstable conditions; candidates reaching them within
        # one time step of each other are told apart by the gap width at that instant
        t_first = min(unstable)[0]
        tol = float(np.median(np.diff(t))) if t.size > 1 else 0.0
        tied = [k for t_u, k in unstable if t_u <= t_first + tol]
        step_u = int(np.searchsorted(t, t_first, side="right") - 1)
        k = max(tied, key=lambda k: _gap_at(d, masks[k], step_u))
        gap = _gap_at(d, masks[k], step_u)
        cluster = CriticalCluster(masks[k], "unstable", step_u, len(masks), gap)
        return cluster, omibs[k], margins[k]
    best: tuple[float, int] | None = None
    for k, mg in enumerate(margins):
        if mg[0] not in ("stable", "stable_extrapolated"):
            continue
        step = int(np.argmax(omibs[k].delta)) if mg[0] == "stable" else d.shape[0] - 1
        gap = _gap_at(d, masks[k], step)
        if best is None or gap > best[0]:
            best = (gap, k)
    if best is None:
        k = int(np.argmax(widths))
        cluster = CriticalCluster(masks[k], "undecided", int(steps[k]), len(masks), widths[k])
        return cluster, omibs[k], margins[k]
    gap, k = best
    step = int(np.argmax(omibs[k].delta)) if margins[k][0] == "stable" else d.shape[0] - 1
    cluster = CriticalCluster(masks[k], "stable", step, len(masks), gap)
    return cluster, omibs[k], margins[k]


def critical_cluster(
    delta_t: npt.ArrayLike,
    M: npt.ArrayLike,
    *,
    time: npt.ArrayLike | None = None,
    omega_t: npt.ArrayLike | None = None,
    pm: npt.ArrayLike | None = None,
    pe_t: npt.ArrayLike | None = None,
    n_gaps: int = 5,
    n_fit: int = 3,
) -> CriticalCluster:
    """Critical machines of a post-clearing trajectory (book §2.2.3).

    At every step the machines are sorted by angle and the ``n_gaps`` largest gaps give the
    candidate decompositions; the candidate whose OMIB first meets the unstable conditions
    (2.15)-(2.16) is retained, else the one with the widest gap at its OMIB's return point.
    ``P_a`` is ``P_m - P_e`` when ``pm`` and ``pe_t`` are given and ``M d omega/dt`` from the
    trajectory otherwise. Arguments as in :func:`omib`.
    """
    d, w, p, pe, m, t = _arrays(delta_t, omega_t, pm, pe_t, M, time)
    cluster, _, _ = _select(d, w, p, pe, m, t, n_gaps, n_fit)
    return cluster


def sime_margin(
    delta_t: npt.ArrayLike,
    omega_t: npt.ArrayLike | None,
    pm: npt.ArrayLike | None,
    pe_t: npt.ArrayLike | None,
    M: npt.ArrayLike,
    *,
    time: npt.ArrayLike | None = None,
    critical: npt.ArrayLike | None = None,
    n_gaps: int = 5,
    n_fit: int = 3,
) -> SimeResult:
    """SIME stability margin of one post-clearing trajectory.

    ``eta = A_dec - A_acc``: ``-1/2 M omega_u^2`` at the unstable crossing (2.14, negative,
    ``kind="unstable"``) or the triangle approximation of the unused decelerating area
    (2.20, positive, ``kind="stable"``); ``"stable_extrapolated"`` when the trajectory ends
    before the return point and the area had to be extrapolated from its end; ``"undecided"``
    (NaN) when neither condition is met and no extrapolation is possible. The critical
    machines come from :func:`critical_cluster` unless ``critical`` is given. Arguments as in
    :func:`omib`; ``delta_t`` must start at fault clearing (post-fault trajectory only).
    """
    d, w, p, pe, m, t = _arrays(delta_t, omega_t, pm, pe_t, M, time)
    if critical is None:
        cluster, o, mg = _select(d, w, p, pe, m, t, n_gaps, n_fit)
    else:
        crit = np.asarray(critical, dtype=bool)
        if crit.shape != (d.shape[1],):
            raise ConfigurationError("critical must have one entry per machine")
        o = _omib_batch(d, w, p, pe, m, t, crit[None, :])[0]
        mg = _margin(o, t, n_fit)
        cluster = CriticalCluster(crit, "given", 0, 1, _gap_at(d, crit, 0))
    kind, eta, t_u, d_u, w_u, t_r, d_r, pa_r = mg
    return SimeResult(
        eta=eta,
        kind=kind,
        variant="trajectory" if p is None else "recorded_pm",
        t_u=t_u,
        delta_u=d_u,
        omega_u=w_u,
        t_r=t_r,
        delta_r=d_r,
        pa_r=pa_r,
        m_eq=o.m_eq,
        critical=cluster.critical,
        cluster=cluster,
        omib=o,
    )
