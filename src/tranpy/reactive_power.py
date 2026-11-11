"""Reactive power of the grid model: embedded wind and solar, load and TSO compensation.

Embedded units (class C of :mod:`tranpy.grid_connections`, rows ``embedded/...``) get the
reactive range that the German connection rules require at the connection point, as a share of
the installed active power ``P_b,inst`` (``max_p_mw``):

* VDE-AR-N 4110:2018-11 (medium voltage), clause 10.2.2.2: Q between cos phi 0.95 underexcited
  and 0.95 overexcited at ``P_b,inst``, Q/P_b,inst from -0.33 to +0.33;
* VDE-AR-N 4120:2018-11 (high voltage), clause 10.2.2.2, three variants of the same span:
  variant 1 cos phi 0.975 underexcited to 0.90 overexcited (-0.23 to +0.48), variant 2 0.95
  underexcited to 0.925 overexcited (-0.33 to +0.41), variant 3 0.925 underexcited to 0.95
  overexcited (-0.41 to +0.33). Variant 2 is the one the BMWi study "Zukunftige Beschaffung von
  Blindleistung II" (INA, 25 June 2021, p. 85) uses for the HV fleet;
* clause 10.2.2.3 of both: the range applies below ``P_b,inst`` down to a minimum active power
  (``min_active_share``, 0.1 here); below it no range is required, so the unit gets none;
* clause 10.2.2.4 of both: the operator sets the method, among them Q(U), cos phi(P), Q(P) and a
  fixed cos phi. ``control="q_u"`` lets the unit hold the voltage of its lumped bus within the
  range, a Q(U) characteristic with zero droop; ``"cos_phi"`` sets a fixed overexcited share.

The clause numbers follow the 2018-11 editions; the standards are sold, not published, so the
numbers are to be checked against the text. Assumptions, labelled: the whole embedded fleet
meets these rules (plants before 2018 followed the BDEW MV guideline 2008 and the SDLWindV 2009,
both cos phi 0.95, so the share is close), and the 110 kV network between the units and the
lumped bus is ignored, as in the connection model.

Load: :func:`load_tan_phi` ties the load's tan phi to its level with the values of the BMWi study
(p. 56/57): households cos phi 0.99, industry and commerce cos phi(P) from 0.925 at minimum to
0.975 at maximum demand, industry and commerce 75 % of demand (p. 113). These are loads only:
the charging of 110 kV cables, which makes the distribution network capacitive at light load,
is not in the lumped model, so the values are an upper bound of the inductive demand.

Compensation: :func:`read_compensation` reads ``compensation_de.csv`` (TSO press releases, NEP
project sheets, OSM ``power=compensator``) and returns the in-service units for the builders of
:mod:`tranpy.powsybl_model` and :mod:`tranpy.pandapower_model`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "COMPENSATION_COLUMNS",
    "RULES",
    "ReactiveRange",
    "embedded_reactive_capability",
    "load_tan_phi",
    "read_compensation",
    "with_load_tan_phi",
]


@dataclass(frozen=True)
class ReactiveRange:
    """Reactive range as Q / P_b,inst: ``under`` (absorbing, negative) to ``over`` (injecting)."""

    under: float
    over: float
    source: str


def _tan(cos_phi: float) -> float:
    return float(np.tan(np.arccos(cos_phi)))


RULES: Mapping[str, ReactiveRange] = {
    "4110": ReactiveRange(-_tan(0.95), _tan(0.95), "VDE-AR-N 4110:2018-11 cl. 10.2.2.2"),
    "4120-1": ReactiveRange(-_tan(0.975), _tan(0.90), "VDE-AR-N 4120:2018-11 cl. 10.2.2.2 var. 1"),
    "4120-2": ReactiveRange(-_tan(0.95), _tan(0.925), "VDE-AR-N 4120:2018-11 cl. 10.2.2.2 var. 2"),
    "4120-3": ReactiveRange(-_tan(0.925), _tan(0.95), "VDE-AR-N 4120:2018-11 cl. 10.2.2.2 var. 3"),
}
_RENEWABLE = ("Windenergie (Onshore", "Solare Strahlungsenergie")
_OWN_LOADS_EXCLUDED = ("hvdc/", "pump/")


def embedded_reactive_capability(
    machines: pd.DataFrame,
    *,
    rule: str = "4110",
    control: Literal["q_u", "cos_phi"] = "q_u",
    cos_phi_share: float = 1.0,
    min_active_share: float = 0.1,
    prefix: str = "embedded/",
) -> pd.DataFrame:
    """Machine table with the connection-rule range for embedded wind and solar.

    Args:
        machines: ``synchronous_machine`` table (``max_p_mw`` is the installed power).
        rule: key of :data:`RULES`.
        control: ``"q_u"`` holds the lumped-bus voltage within the range; ``"cos_phi"`` injects
            ``cos_phi_share`` of the overexcited limit at fixed Q (column ``target_q_mvar``).
        cos_phi_share: share of the overexcited limit for ``"cos_phi"`` (negative absorbs).
        min_active_share: least P / P_b,inst with a required range (clause 10.2.2.3).
        prefix: mrid prefix of the embedded rows.
    """
    limits = RULES[rule]
    table = machines.copy()
    chosen = table["mrid"].astype(str).str.startswith(prefix) & table["fuel"].astype(
        str
    ).str.startswith(_RENEWABLE)
    installed = table["max_p_mw"].astype(float)
    active = table["p_mw"].astype(float)
    eligible = chosen & table["in_service"].astype(bool) & active.ge(min_active_share * installed)
    eligible &= installed.gt(0)
    q_max = np.where(eligible, limits.over * installed, 0.0)
    q_min = np.where(eligible, limits.under * installed, 0.0)
    if "target_q_mvar" not in table:
        table["target_q_mvar"] = 0.0
    table.loc[chosen, "q_max_mvar"] = q_max[chosen.to_numpy()]
    table.loc[chosen, "q_min_mvar"] = q_min[chosen.to_numpy()]
    if control == "q_u":
        table.loc[chosen, "voltage_control"] = eligible[chosen]
        table.loc[chosen, "target_q_mvar"] = 0.0
    else:
        share = float(cos_phi_share)
        fixed = np.where(share >= 0, share * q_max, -share * q_min)
        table.loc[chosen, "voltage_control"] = False
        table.loc[chosen, "target_q_mvar"] = fixed[chosen.to_numpy()]
    return table


def load_tan_phi(load_share: float, *, industry_share: float = 0.75) -> float:
    """tan phi of the load at ``load_share`` = load / annual peak (0..1), see the module notes.

    Households cos phi 0.99; industry and commerce cos phi(P) linear from 0.925 at the least to
    0.975 at the greatest demand (BMWi "Zukunftige Beschaffung von Blindleistung II", 2021,
    scenario kennlV, p. 56); industry and commerce 75 % of demand (p. 113).
    """
    share = float(np.clip(load_share, 0.0, 1.0))
    industry = _tan(0.925 + (0.975 - 0.925) * share)
    return industry_share * industry + (1.0 - industry_share) * _tan(0.99)


def with_load_tan_phi(loads: pd.DataFrame, tan_phi: float) -> pd.DataFrame:
    """``energy_consumer`` with q = p * tan phi on own loads (HVDC and pump rows kept)."""
    table = loads.copy()
    own = ~table["mrid"].astype(str).str.startswith(_OWN_LOADS_EXCLUDED)
    table.loc[own, "q_mvar"] = table.loc[own, "p_mw"].astype(float) * tan_phi
    return table


COMPENSATION_COLUMNS = ("mrid", "bus_mrid", "kind", "q_max_mvar", "q_min_mvar", "target_v_pu")
"""Builder input: ``kind`` is ``"controlled"`` (STATCOM, SVC, synchronous condenser: holds
``target_v_pu`` within the limits) or ``"shunt"`` (switched capacitor or reactor, fixed Q at
1 pu in ``q_max_mvar``). Positive Q injects."""
_CONTROLLED = {"STATCOM", "SVC", "synchronous_condenser"}


def read_compensation(
    path: str | Path,
    buses: pd.DataFrame,
    *,
    with_assumptions: bool = False,
    switching: Literal["raise", "lower", "all"] = "raise",
    target_v_pu: float = 1.02,
) -> pd.DataFrame:
    """In-service compensation of ``compensation_de.csv`` on the buses of the model.

    Args:
        path: the CSV (columns ``node``, ``type``, ``q_max_mvar``, ``q_min_mvar``, ``in_model``,
            ``in_model_with_assumptions``).
        buses: ``busbar_section`` with ``name`` = node (``site|kv``) and ``mrid``.
        with_assumptions: also units with an assumed (NEP standard) rating.
        switching: switched shunts that are on: capacitors (``"raise"``, a low-voltage case),
            reactors (``"lower"``) or both.
        target_v_pu: voltage the controlled units hold.
    """
    units = pd.read_csv(path)
    keep = units["in_model_with_assumptions" if with_assumptions else "in_model"].astype(bool)
    units = units[keep].reset_index(drop=True)
    bus_of = dict(zip(buses["name"], buses["mrid"], strict=False))
    units["bus_mrid"] = units["node"].map(bus_of)
    units = units.dropna(subset=["bus_mrid"])
    controlled = units["type"].isin(_CONTROLLED)
    capacitive = ~controlled & units["q_max_mvar"].astype(float).gt(0)
    on = controlled | (capacitive if switching == "raise" else ~capacitive)
    if switching == "all":
        on = pd.Series(True, index=units.index)
    units = units[on]
    return pd.DataFrame(
        {
            "mrid": [
                f"comp/{i}/{n}/{t}"
                for i, n, t in zip(units.index, units["node"], units["type"], strict=True)
            ],
            "bus_mrid": units["bus_mrid"].to_numpy(),
            "kind": np.where(units["type"].isin(_CONTROLLED), "controlled", "shunt"),
            "q_max_mvar": units["q_max_mvar"].astype(float).to_numpy(),
            "q_min_mvar": units["q_min_mvar"].astype(float).to_numpy(),
            "target_v_pu": target_v_pu,
        }
    )


def boundary_q_limits(
    lines: pd.DataFrame,
    injections: pd.DataFrame,
    *,
    share: float = 0.33,
    floor_mvar: float = 50.0,
) -> dict[str, float]:
    """Reactive limit (Mvar, symmetric) of each boundary injection from its in-operation lines.

    The limit is ``share`` of the summed thermal rating (sqrt(3) * kV * A) of the lines that end
    on the injection's bus, but at least ``floor_mvar``. Lines whose ``status`` is not
    ``in_operation`` are left out (a table without the column counts every line).
    """
    if "status" in lines:
        lines = lines[lines["status"].fillna("in_operation").eq("in_operation")]
    mva = (
        (lines["rated_current_a"].astype(float) * lines["nominal_voltage_kv"].astype(float))
        * np.sqrt(3)
        / 1000.0
    )
    limits = {}
    for mrid, bus in zip(injections["mrid"], injections["bus_mrid"], strict=True):
        at = lines["bus1_mrid"].eq(bus) | lines["bus2_mrid"].eq(bus)
        limits[mrid] = max(share * float(np.nansum(mva[at])), floor_mvar)
    return limits
