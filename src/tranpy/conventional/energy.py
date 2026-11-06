"""M3: kinetic-energy terms of the transient energy function (TEF) at fault clearing.

The TEF of the classical multi-machine model in the centre-of-inertia (COI) frame is
``V = V_KE + V_PE`` (Athay, Podmore and Virmani (1979), "A practical method for the direct
analysis of transient stability", IEEE Trans. PAS-98(2), 573-584; Pai (1989), *Energy
Function Analysis for Power System Stability*, Kluwer, Ch. 3; Fouad and Vittal (1992),
*Power System Transient Stability Analysis Using the Transient Energy Function Method*,
Prentice Hall, Ch. 4). This module implements the parts that need only machine speeds,
angles and inertias:

* :func:`coi_kinetic_energy` - ``V_KE = 1/2 sum_i M_i (omega_i - omega_COI)^2``, the
  kinetic energy relative to the COI (paper Eq. 7 in the COI frame).
* :func:`corrected_kinetic_energy` - the kinetic energy responsible for the separation of a
  critical machine group C from the rest N (Fouad and Stanton (1981), "Transient stability
  of a multi-machine power system, Part II: Critical transient energy", IEEE Trans.
  PAS-100(7); Fouad, Vittal and Oh (1984), "Critical energy for direct transient stability
  assessment of a multimachine power system", IEEE Trans. PAS-103(8), 2199-2206; Fouad and
  Vittal (1992)): ``V_KEcorr = 1/2 M_eq (omega_C - omega_N)^2`` with
  ``M_eq = M_C M_N / (M_C + M_N)`` and ``omega_C``, ``omega_N`` the inertia-weighted group
  speeds. It is also the kinetic energy of the one-machine infinite-bus (OMIB) equivalent
  of the SIME/EEAC methods (Pavella, Ernst and Ruiz-Vega (2000), *Transient Stability of
  Power Systems: A Unified Approach to Assessment and Control*, Kluwer, Ch. 2).
* :func:`cct_normalised_energy` - the kinetic energy at clearing divided by the kinetic
  energy at the critical clearing time of the same fault, the direct-method decision
  ``V(t_cl) >= V_cr`` with the critical energy taken from the clearing-time sweep instead of
  a potential-energy surface (Fouad and Vittal (1992), Ch. 6: the critical energy equals the
  transient energy at the critical clearing time, which is how direct-method CCTs are
  validated against time-domain simulation). Ratios above 1 mean the event's clearing
  energy exceeds that of the bracketed CCT.
* :func:`critical_group_by_largest_gap` - the critical group as the machines above the
  largest gap of the sorted rotor-angle excursions. SIME (Pavella et al. (2000)) forms its
  candidate decompositions from the largest gaps of the sorted angles along the post-fault
  trajectory and keeps the first candidate to lose synchronism; with one snapshot at
  clearing only the largest gap can be used. This is the documented substitute for the
  mode-of-disturbance test of Fouad and Vittal, which needs the post-fault network.

Units: with ``w_i = H_i S_n,i`` (MW s) and speeds in per unit of nominal, the physical
kinetic energy of a deviation is ``w_i dw_i^2`` (MJ), i.e. ``M_i = 2 w_i`` in the formulas
above. This is independent of the system power base, so values of different operating
hours are comparable.

The potential-energy part (``V_PE``: post-fault stable equilibrium, transfer conductances of
the reduced network) is not implemented here; see ``network.py`` for the reduction and the
feasibility note of the benchmark write-up for the data it would need.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd

from tranpy_core.errors import ConfigurationError, DataError

__all__ = [
    "cct_normalised_energy",
    "coi_kinetic_energy",
    "corrected_kinetic_energy",
    "critical_group_by_largest_gap",
    "kinetic_energy_features",
]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def _check(dw: npt.ArrayLike, weights: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    d = np.atleast_2d(np.asarray(dw, dtype=np.float64))
    w = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    if w.shape[0] == 1 and d.shape[0] > 1:
        w = np.broadcast_to(w, d.shape)
    if d.shape != w.shape:
        raise ConfigurationError(f"speeds {d.shape} and weights {w.shape} differ in shape")
    if np.any(w < 0) or np.any(w.sum(axis=1) <= 0):
        raise ConfigurationError("inertia weights must be non-negative with a positive sum")
    return d, w


def coi_kinetic_energy(dw: npt.ArrayLike, weights: npt.ArrayLike) -> FloatArray:
    """``sum_i w_i (dw_i - dw_COI)^2`` per row (events x machines), in MJ for MW s weights.

    ``weights`` are ``H_i S_n,i`` (0 for machines not committed), either one row for all
    events or one row per event.
    """
    d, w = _check(dw, weights)
    coi = (d * w).sum(axis=1) / w.sum(axis=1)
    return np.asarray(((d - coi[:, None]) ** 2 * w).sum(axis=1), dtype=np.float64)


def critical_group_by_largest_gap(
    excursion: npt.ArrayLike, online: npt.ArrayLike | None = None
) -> BoolArray:
    """Machines above the largest gap of the sorted excursions (events x machines).

    Committed machines (``online``) are sorted by decreasing excursion; the largest
    difference between consecutive machines splits them, and the advanced side is the
    critical group. Machines not committed are never critical.
    """
    x = np.atleast_2d(np.asarray(excursion, dtype=np.float64))
    on = np.ones_like(x, dtype=bool) if online is None else np.atleast_2d(online).astype(bool)
    if on.shape != x.shape:
        raise ConfigurationError("online mask and excursions differ in shape")
    critical = np.zeros_like(x, dtype=bool)
    for r in range(x.shape[0]):
        idx = np.flatnonzero(on[r])
        if idx.size < 2:
            raise DataError(f"row {r}: fewer than two committed machines")
        order = idx[np.argsort(-x[r, idx], kind="stable")]
        gaps = x[r, order[:-1]] - x[r, order[1:]]
        cut = int(np.argmax(gaps))
        critical[r, order[: cut + 1]] = True
    return critical


def corrected_kinetic_energy(
    dw: npt.ArrayLike, weights: npt.ArrayLike, critical: npt.ArrayLike
) -> FloatArray:
    """``w_C w_N / (w_C + w_N) (dw_C - dw_N)^2`` per row (``= 1/2 M_eq omega_eq^2``).

    ``critical`` marks the critical group of each row; the rest of the committed machines
    (weight > 0) form the other group. Rows with an empty group give NaN.
    """
    d, w = _check(dw, weights)
    c = np.atleast_2d(np.asarray(critical, dtype=bool))
    if c.shape != d.shape:
        raise ConfigurationError("critical mask and speeds differ in shape")
    wc = (w * c).sum(axis=1)
    wn = (w * ~c).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        omega_c = (w * c * d).sum(axis=1) / wc
        omega_n = (w * ~c * d).sum(axis=1) / wn
        m_eq = wc * wn / (wc + wn)
    out = m_eq * (omega_c - omega_n) ** 2
    out[(wc <= 0) | (wn <= 0)] = np.nan
    return np.asarray(out, dtype=np.float64)


def _machine_block(
    aligned: pd.DataFrame,
    rows: npt.NDArray[np.int64],
    instant: str,
    ids: list[str],
    var: str,
    case: str,
) -> FloatArray:
    names = [f"{instant}|{m}|{var}" for m in ids]
    missing = [n for n in names if n not in aligned.columns]
    if missing:
        raise DataError(f"case {case!r}: aligned columns missing, e.g. {missing[0]}")
    return np.asarray(aligned.iloc[rows][names].to_numpy(np.float64), dtype=np.float64)


def kinetic_energy_features(
    aligned: pd.DataFrame,
    cases: pd.Series,
    weights_by_case: Mapping[str, Mapping[str, float]],
    *,
    instant: str = "clear",
    reference: str = "pre",
) -> pd.DataFrame:
    """Kinetic-energy indices of every event from the aligned per-machine features.

    Args:
        aligned: :func:`tranpy.ml_events.aligned_features` with ``instant`` and
            ``reference`` (columns ``<inst>|<machine id>|dw`` / ``dangle``).
        cases: case of every row of ``aligned``.
        weights_by_case: ``H * Sn`` (MW s) per machine id of every case (the
            ``kinetic_mws`` column of :func:`tranpy.ml_events.case_machines`).
        instant: snapshot at which the energy is evaluated (fault clearing).
        reference: snapshot the angle excursion is measured from (pre-fault).

    Returns:
        ``ke_coi_mj`` (:func:`coi_kinetic_energy`), ``ke_corr_mj``
        (:func:`corrected_kinetic_energy` with the group of
        :func:`critical_group_by_largest_gap` on the COI-angle excursion
        ``dangle(instant) - dangle(reference)``) and ``n_critical``; index of ``aligned``.
    """
    if len(cases) != len(aligned):
        raise ConfigurationError("one case per row")
    case_arr = np.asarray(cases, dtype=object)
    out = pd.DataFrame(
        np.nan, index=aligned.index, columns=["ke_coi_mj", "ke_corr_mj", "n_critical"]
    )
    for case in sorted(set(case_arr.tolist()), key=str):
        if case not in weights_by_case:
            raise DataError(f"no machine weights for case {case!r}")
        rows = np.flatnonzero(case_arr == case)
        ids = list(weights_by_case[case])
        w = np.array([weights_by_case[case][m] for m in ids], dtype=np.float64)
        dw = _machine_block(aligned, rows, instant, ids, "dw", str(case))
        excursion = _machine_block(aligned, rows, instant, ids, "dangle", str(case))
        excursion = excursion - _machine_block(aligned, rows, reference, ids, "dangle", str(case))
        critical = critical_group_by_largest_gap(excursion)
        out.iloc[rows, 0] = coi_kinetic_energy(dw, w)
        out.iloc[rows, 1] = corrected_kinetic_energy(dw, w, critical)
        out.iloc[rows, 2] = critical.sum(axis=1)
    return out


def cct_normalised_energy(
    ke_at_clearing: npt.ArrayLike,
    cct_brackets: pd.DataFrame,
    case: npt.ArrayLike,
    fault_line: npt.ArrayLike,
    clearing_time: npt.ArrayLike,
    sweep: npt.ArrayLike,
) -> FloatArray:
    """Kinetic energy at clearing over the kinetic energy at the bracketed CCT, per event.

    For every bracketed ``(case, fault_line)`` of ``cct_brackets``
    (:func:`tranpy.conventional.cct.cct_brackets`: ``status == "bracketed"`` with
    ``t_stable_max`` and ``t_unstable_min``), the critical energy is the kinetic energy
    interpolated linearly across the sweep's clearing times at the bracket midpoint
    ``(t_stable_max + t_unstable_min) / 2``. Every event of that case and line (sweep or
    random) gets ``ke / ke_cct``; events on lines without a bracket get NaN.

    Args:
        ke_at_clearing: kinetic energy of every event at fault clearing (any index of
            :func:`kinetic_energy_features`).
        cct_brackets: one row per swept ``(case, fault_line)``.
        case, fault_line, clearing_time: per event.
        sweep: per event, whether it belongs to the clearing-time sweep (the ``grid`` kind);
            only sweep events define the energy-versus-clearing-time curve.
    """
    ke = np.asarray(ke_at_clearing, dtype=np.float64)
    cases = np.asarray(case, dtype=object)
    lines = np.asarray(fault_line, dtype=np.float64)
    t_cl = np.asarray(clearing_time, dtype=np.float64)
    is_sweep = np.asarray(sweep, dtype=bool)
    if not ke.shape == cases.shape == lines.shape == t_cl.shape == is_sweep.shape:
        raise ConfigurationError("one value of every argument per event")
    needed = {"case", "fault_line", "t_stable_max", "t_unstable_min", "status"}
    if missing := sorted(needed - set(cct_brackets.columns)):
        raise DataError(f"cct_brackets lacks columns {missing}")
    out = np.full(ke.shape, np.nan)
    bracketed = cct_brackets[cct_brackets["status"] == "bracketed"]
    b_case = bracketed["case"].to_numpy(dtype=object)
    b_line = bracketed["fault_line"].to_numpy(dtype=np.float64)
    b_cct = 0.5 * (
        bracketed["t_stable_max"].to_numpy(dtype=np.float64)
        + bracketed["t_unstable_min"].to_numpy(dtype=np.float64)
    )
    for c, line, t_cct in zip(b_case, b_line, b_cct, strict=True):
        on_line = (cases == c) & (lines == line)
        pts = np.flatnonzero(on_line & is_sweep & np.isfinite(ke))
        if pts.size < 2:
            continue
        order = pts[np.argsort(t_cl[pts], kind="stable")]
        ke_cct = float(np.interp(t_cct, t_cl[order], ke[order]))
        if ke_cct > 0:
            out[on_line] = ke[on_line] / ke_cct
    return out
