"""ANDES case of the German grid model built from the gold tables (optional ``[andes]`` extra).

Dynamic data per plant is not published, so every dynamic value here is an estimate:

* the plants of one bus are aggregated into one synchronous machine (GENCLS) and one inverter;
* inertia and transient reactance per fuel come from the ranges in Kundur, *Power System
  Stability and Control* (1994), Tables 3.2 and 4.3; damping is assumed;
* committed machine rating is ``dispatched power / machine_loading``, since the model has no
  unit commitment;
* inverters (solar, wind, storage) sit on a stub bus as REGCA1 converters without inertia and
  without plant controls, or as negative loads with ``inverter_model="pq"``;
* neighbouring networks are one machine per border node: either the earlier assumed machine or
  a Continental Europe equivalent with sourced inertia, FCR and self-regulation;
* phase shifter angles, transformer magnetising branches and HVDC links are left out: ANDES has
  no two-terminal HVDC model.

:data:`DYNAMIC_VALUES` is saved beside the case by :func:`build_grid_case`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from tranpy.simulation.andes import _load_openmp_libraries_first
from tranpy_core._optional import require
from tranpy_core.errors import DataError, SimulationError

__all__ = [
    "BORDER_SHORT_CIRCUIT",
    "CONTROLS",
    "DYNAMIC_VALUES",
    "MACHINE_VALUES",
    "NeighbourEquivalent",
    "border_shares",
    "border_short_circuit_gva",
    "build_connected_case",
    "build_grid_case",
    "continental_europe_equivalent",
    "entsoe_kinetic_energy_mws",
    "fuel_class",
    "line_parameters",
    "neighbour_kinetic_energy",
]

_BASE_MVA = 100.0
_SOURCE = "estimate within the ranges of Kundur (1994), Tables 3.2 and 4.3"
DYNAMIC_VALUES = pd.DataFrame(
    [
        ("thermal", 4.5, 0.30, 2.0, _SOURCE),
        ("hydro", 3.0, 0.35, 2.0, _SOURCE),
        ("external", 5.0, 0.30, 2.0, "assumed equivalent of a neighbouring network"),
    ],
    columns=["machine_class", "h_s", "xd1_pu", "damping_pu", "source"],
).set_index("machine_class")
"""Estimated machine values (H and x'd on the machine base; damping assumed)."""

_DAMPING_PU = 2.0
_INVERTER = ("Solare Strahlungsenergie", "Windenergie", "Speicher")
_HYDRO = ("Wasser", "Pumpspeicher")
_EXTERNAL_MVA = 3000.0


MACHINE_VALUES = pd.DataFrame(
    [
        ("thermal", 4.5, 1.8, 1.7, 0.30, 0.55, 0.25, 0.15, 6.0, 0.8, 0.035, 0.05, 0.003),
        ("hydro", 3.0, 1.0, 0.65, 0.30, 0.65, 0.25, 0.15, 6.0, 0.5, 0.035, 0.05, 0.003),
    ],
    columns=[
        "machine_class", "h_s", "xd", "xq", "xd1", "xq1", "xd2", "xl",
        "td10", "tq10", "td20", "tq20", "ra",
    ],
).set_index("machine_class")  # fmt: skip
"""GENROU values on the machine base: assumed within typical ranges, not plant data.

ANDES requires ``xq2 = xd2``. The textbook page references are still to be confirmed.
"""

CONTROLS: Mapping[str, Mapping[str, float]] = {
    "SEXS": {"TATB": 0.1, "TB": 10.0, "K": 100.0, "TE": 0.05, "EMIN": 0.0, "EMAX": 5.0},
    "TGOV1": {"R": 0.05, "T1": 0.5, "T2": 3.0, "T3": 10.0, "VMAX": 1.2, "VMIN": 0.0},
    "HYGOV": {
        "R": 0.05, "r": 1.0, "Tr": 1.0, "Tf": 0.05, "Tg": 0.05, "VELM": 0.3,
        "GMAX": 1.0, "GMIN": 0.0, "Tw": 1.0, "At": 1.0, "Dt": 0.0, "qNL": 0.1,
    },
}  # fmt: skip
"""Assumed control values on the machine base, not plant data.

HYGOV values are the parameter defaults of the ANDES 2.0 ``HYGOV`` model
(``andes/models/governor/hygov.py``), which follow the PSS/E HYGOV defaults.
"""


def fuel_class(fuel: str) -> str:
    """``inverter``, ``hydro`` or ``thermal`` for a fuel of the plant register."""
    if str(fuel).startswith(_INVERTER):
        return "inverter"
    return "hydro" if str(fuel) in _HYDRO else "thermal"


def line_parameters(
    r_ohm: float, x_ohm: float, vn_kv: float, base_mva: float = _BASE_MVA
) -> tuple[float, float]:
    """Series impedance in per unit of the system base."""
    z_base = vn_kv**2 / base_mva
    return r_ohm / z_base, x_ohm / z_base


def _main_component(buses: list[str], pairs: list[tuple[str, str]]) -> set[str]:
    parent = {b: b for b in buses}

    def root(b: str) -> str:
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        return b

    for a, b in pairs:
        parent[root(a)] = root(b)
    groups: dict[str, set[str]] = {}
    for b in buses:
        groups.setdefault(root(b), set()).add(b)
    return max(groups.values(), key=len)


def build_grid_case(
    tables: Any,
    out_path: str | Path,
    *,
    dispatch: Mapping[str, float] | None = None,
    machine_loading: float = 0.8,
    inverter_model: Literal["regca1", "pq"] = "regca1",
    min_machine_mva: float = 0.0,
    frequency_hz: float = 50.0,
) -> Path:
    """Write the gold tables as an ANDES ``.xlsx`` case with its value table beside it.

    Args:
        tables: the model (``CimTables``).
        out_path: case file to write.
        dispatch: MW per machine ``mrid`` of a snapshot; the model's ``p_mw`` if None.
        machine_loading: dispatched power as a share of the committed machine rating.
        inverter_model: ``regca1`` or ``pq``.
        min_machine_mva: a bus's machines below this rating become a negative load.
        frequency_hz: nominal frequency.

    Raises:
        SimulationError: if the AC power flow does not converge.
    """
    _load_openmp_libraries_first()
    andes = require("andes", "andes")
    vl_kv = dict(
        zip(tables.voltage_level["mrid"], tables.voltage_level["nominal_voltage_kv"], strict=True)
    )
    bus = tables.busbar_section
    lines = tables.ac_line_segment
    lines = lines[lines["status"].fillna("in_operation").eq("in_operation")]
    lines = lines[lines["bus1_mrid"] != lines["bus2_mrid"]]
    transformers = tables.power_transformer
    transformers = transformers[transformers["bus1_mrid"] != transformers["bus2_mrid"]]
    pairs = list(zip(lines["bus1_mrid"], lines["bus2_mrid"], strict=True))
    pairs += list(zip(transformers["bus1_mrid"], transformers["bus2_mrid"], strict=True))
    keep = _main_component(list(bus["mrid"]), pairs)
    bus = bus[bus["mrid"].isin(keep)].reset_index(drop=True)
    lines = lines[lines["bus1_mrid"].isin(keep)]
    transformers = transformers[transformers["bus1_mrid"].isin(keep)]
    ids = {m: k + 1 for k, m in enumerate(bus["mrid"])}
    kv = {m: float(vl_kv[v]) for m, v in zip(bus["mrid"], bus["voltage_level_mrid"], strict=True)}

    machines = tables.synchronous_machine
    machines = machines[machines["bus_mrid"].isin(keep)].copy()
    if dispatch is not None:
        machines["p_mw"] = machines["mrid"].map(dispatch).fillna(0.0)
    machines["group"] = machines["fuel"].map(fuel_class)
    power = machines.pivot_table(
        index="bus_mrid", columns="group", values="p_mw", aggfunc="sum", fill_value=0.0
    )
    for name in ("thermal", "hydro", "inverter"):
        if name not in power:
            power[name] = 0.0
    external = tables.external_network_injection
    external = external[external["bus_mrid"].isin(keep)]
    consumers = tables.energy_consumer
    consumers = consumers[consumers["bus_mrid"].isin(keep)]
    if power[["thermal", "hydro"]].sum(axis=1).max() <= 0:
        raise DataError("no synchronous generation to hold the system")

    ss = andes.System()
    ss.config.freq = frequency_hz
    next_bus = len(ids) + 1
    for row in bus.itertuples():
        ss.add(
            "Bus",
            {
                "idx": ids[row.mrid],
                "name": str(row.name),
                "Vn": kv[row.mrid],
                "xcoord": float(row.lon),
                "ycoord": float(row.lat),
            },
        )
    for k, row in enumerate(consumers.itertuples()):
        ss.add(
            "PQ",
            {
                "idx": f"L{k}",
                "bus": ids[row.bus_mrid],
                "Vn": kv[row.bus_mrid],
                "p0": float(row.p_mw) / _BASE_MVA,
                "q0": float(row.q_mvar) / _BASE_MVA,
            },
        )
    for k, row in enumerate(lines.itertuples()):
        vn = kv[row.bus1_mrid]
        r, x = line_parameters(float(row.r_ohm), float(row.x_ohm), vn)
        ss.add(
            "Line",
            {
                "idx": f"Line_{k}",
                "bus1": ids[row.bus1_mrid],
                "bus2": ids[row.bus2_mrid],
                "r": r,
                "x": x,
                "b": float(row.bch_s) * vn**2 / _BASE_MVA,
                "Vn1": vn,
                "Vn2": kv[row.bus2_mrid],
                "trans": 0,
            },
        )
    for k, row in enumerate(transformers.itertuples()):
        vn1, vn2 = kv[row.bus1_mrid], kv[row.bus2_mrid]
        u1, u2 = float(row.rated_u1_kv), float(row.rated_u2_kv)
        # as in network_from_cim_tables: table values on side 1, referred to side 2
        ratio2 = (u2 / u1) ** 2
        r, x = line_parameters(float(row.r_ohm) * ratio2, float(row.x_ohm) * ratio2, vn2)
        ss.add(
            "Line",
            {
                "idx": f"Tr_{k}",
                "bus1": ids[row.bus1_mrid],
                "bus2": ids[row.bus2_mrid],
                "r": r,
                "x": x,
                "Vn1": vn1,
                "Vn2": vn2,
                "tap": (u1 / vn1) / (u2 / vn2),
                "trans": 1,
            },
        )
    v_set = 1.02
    sources: list[dict[str, Any]] = []
    for mrid, row in power.iterrows():
        for name in ("thermal", "hydro"):
            p = float(row[name])
            if p > 0:
                sources.append({"bus": mrid, "class": name, "p": p, "sn": p / machine_loading})
    for row in external.itertuples():
        sources.append(
            {"bus": row.bus_mrid, "class": "external", "p": float(row.p_mw), "sn": _EXTERNAL_MVA}
        )
    # one voltage-controlled device per bus: machines of one bus merge, inertia by rating
    merged: dict[str, dict[str, Any]] = {}
    for item in sources:
        values = DYNAMIC_VALUES.loc[item["class"]]
        entry = merged.setdefault(item["bus"], {"p": 0.0, "sn": 0.0, "hs": 0.0, "xs": 0.0})
        entry["p"] += item["p"]
        entry["sn"] += item["sn"]
        entry["hs"] += float(values["h_s"]) * item["sn"]
        entry["xs"] += float(values["xd1_pu"]) * item["sn"]
    slack = max(merged, key=lambda m: merged[m]["sn"])
    small = [m for m, item in merged.items() if item["sn"] < min_machine_mva and m != slack]
    for k, mrid in enumerate(small):
        ss.add(
            "PQ",
            {
                "idx": f"S{k}",
                "bus": ids[mrid],
                "Vn": kv[mrid],
                "p0": -merged.pop(mrid)["p"] / _BASE_MVA,
                "q0": 0.0,
            },
        )
    for k, (mrid, item) in enumerate(merged.items()):
        vn, sn = kv[mrid], item["sn"]
        ss.add(
            "Slack" if mrid == slack else "PV",
            {
                "idx": f"G{k}",
                "bus": ids[mrid],
                "Vn": vn,
                "Sn": sn,
                "p0": item["p"] / _BASE_MVA,
                "v0": v_set,
                "pmax": 10 * sn / _BASE_MVA,
                "pmin": -10 * sn / _BASE_MVA,
                "qmax": 10 * sn / _BASE_MVA,
                "qmin": -10 * sn / _BASE_MVA,
            },
        )
        ss.add(
            "GENCLS",
            {
                "idx": f"M{k}",
                "bus": ids[mrid],
                "gen": f"G{k}",
                "Sn": sn,
                "Vn": vn,
                "M": 2 * item["hs"] / sn,
                "D": _DAMPING_PU,
                "xd1": item["xs"] / sn,
                "ra": 0.0,
            },
        )
    inverters = power.loc[power["inverter"] > 0, "inverter"]
    for k, (mrid, p_mw) in enumerate(inverters.items()):
        vn = kv[str(mrid)]
        if inverter_model == "pq":
            ss.add(
                "PQ",
                {
                    "idx": f"I{k}",
                    "bus": ids[str(mrid)],
                    "Vn": vn,
                    "p0": -float(p_mw) / _BASE_MVA,
                    "q0": 0.0,
                },
            )
            continue
        sn = float(p_mw) / machine_loading
        stub = next_bus
        next_bus += 1
        ss.add("Bus", {"idx": stub, "name": f"inverter {ids[str(mrid)]}", "Vn": vn})
        ss.add(
            "Line",
            {
                "idx": f"Stub_{k}",
                "bus1": ids[str(mrid)],
                "bus2": stub,
                "r": 0.0,
                "x": 0.1 * _BASE_MVA / sn,
                "Vn1": vn,
                "Vn2": vn,
                "trans": 1,
            },
        )
        ss.add(
            "PV",
            {
                "idx": f"IG{k}",
                "bus": stub,
                "Vn": vn,
                "Sn": sn,
                "p0": float(p_mw) / _BASE_MVA,
                "v0": v_set,
                "pmax": 10 * sn / _BASE_MVA,
                "pmin": 0.0,
                "qmax": 10 * sn / _BASE_MVA,
                "qmin": -10 * sn / _BASE_MVA,
            },
        )
        ss.add("REGCA1", {"idx": f"R{k}", "bus": stub, "gen": f"IG{k}", "Sn": sn})
    _set_nominal_frequency(ss, frequency_hz)
    ss.setup()
    if not ss.PFlow.run():
        raise SimulationError("the AC power flow of the grid case does not converge")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    andes.io.xlsx.write(ss, str(out), overwrite=True)
    DYNAMIC_VALUES.to_csv(out.with_suffix(".dynamic_values.csv"))
    pd.DataFrame(
        {"bus": list(ids), "andes_bus": [ids[m] for m in ids]},
    ).to_csv(out.with_suffix(".buses.csv"), index=False)
    return out


def _set_nominal_frequency(ss: Any, frequency_hz: float) -> None:
    """Set ``fn`` on every device that has it; ANDES defaults it to 60 Hz per model."""
    for model in ss.models.values():
        if model.n and "fn" in model.params:
            model.params["fn"].v[:] = [frequency_hz] * model.n


# Continental Europe (CE) equivalent of the neighbouring networks. Sources:
# [FSC] ENTSO-E RG CE SPD, "Frequency Stability Evaluation Criteria for the Synchronous Zone of
#       Continental Europe" (2016): network acceleration time constant T_N = 2 * E_k / P_load,
#       "typical values of T_N for today's power system of CE are larger than 10 seconds" (ch. 6),
#       T_N = 2.3-10 s and load self-regulation 2 %/Hz in Fig. 3; 5-10 mHz/s after 1 GW
#       outages (ch. 2).
# [SOGL] Commission Regulation (EU) 2017/1485, Annex III/V and Art. 153-154: CE reference incident
#       and FCR 3000 MW, full activation at 200 mHz within 30 s, maximum instantaneous deviation
#       800 mHz, maximum steady-state deviation 200 mHz.
# [P1]  ENTSO-E (UCTE) Operation Handbook Policy 1, A-D3: load self-regulation 1 %/Hz.
CE_FCR_MW = 3000.0
"""CE reference incident and FCR [SOGL]."""
CE_FCR_FULL_HZ = 0.2
"""Frequency deviation of full FCR activation [SOGL]."""
CE_FCR_TIME_S = 30.0
"""Full FCR activation time [SOGL]."""
CE_SELF_REGULATION_PER_HZ = 0.01
"""Load self-regulation 1 %/Hz [P1]; [FSC] Fig. 3 uses 2 %/Hz."""
CE_ACCELERATION_TIME_S: Mapping[str, float] = {"low": 5.0, "central": 10.0, "high": 17.0}
"""T_N on the CE load: central 10 s [FSC ch. 6, "larger than 10 s"]; low 5 s ([FSC] Fig. 3 range);
high 17 s, i.e. E_k of about 2500 GWs at 300 GW, which gives the 10 mHz/s per GW observed
[FSC ch. 2]."""
DE_LOAD_SHARE = 0.19
"""Estimate: Germany's share of CE electricity consumption (about 460 of 2400 TWh, ENTSO-E
Statistical Factsheet 2023, rounded)."""
DE_FCR_SHARE = 0.19
"""Estimate: Germany's share of the CE FCR (about 570 of 3000 MW in the 2024 FCR cooperation)."""

# [RoCoF] ENTSO-E SPD Inertia TF, "Inertia and Rate of Change of Frequency (RoCoF)", v17,
#       16 Dec 2020: Table 1 (p. 24) mean H and loading factor LF per production type; eq. 5.5-5.7
#       w_i = S_i / S_tot, H_sys = sum(H_i * w_i / LF_i), E_sys = H_sys * S_tot, with S_i the
#       hourly generation per type from the Transparency Platform, so E_sys = sum(H_i * S_i / LF_i)
#       (S_i / LF_i is the online rating); reference model (p. 29): FCR as a 10 s first-order lag,
#       +-3000 MW, load self-regulation 2 %/Hz (2020 scenarios) and 1.5 %/Hz (2030).
ENTSOE_INERTIA: Mapping[str, tuple[float, float]] = {
    "B01": (3.3, 0.70),  # Biomass
    "B02": (3.8, 0.81),  # Fossil Brown coal/Lignite
    "B03": (4.2, 0.54),  # Fossil Coal-derived gas
    "B04": (4.2, 0.60),  # Fossil Gas
    "B05": (4.2, 0.70),  # Fossil Hard coal
    "B06": (4.3, 0.40),  # Fossil Oil
    "B07": (4.3, 0.40),  # Fossil Oil shale
    "B08": (3.8, 0.59),  # Fossil Peat
    "B09": (3.5, 0.83),  # Geothermal
    "B10": (3.5, 0.46),  # Hydro Pumped Storage (generation)
    "B11": (2.7, 0.61),  # Hydro Run-of-river and poundage
    "B12": (3.7, 0.56),  # Hydro Water Reservoir
    "B13": (3.8, 0.50),  # Marine
    "B14": (5.9, 0.96),  # Nuclear
    "B15": (3.5, 0.50),  # Other renewable
    "B16": (0.0, 1.0),  # Solar
    "B17": (3.8, 0.28),  # Waste
    "B18": (0.0, 1.0),  # Wind Offshore
    "B19": (0.0, 1.0),  # Wind Onshore
    "B20": (3.8, 0.56),  # Other
    "B25": (0.0, 1.0),  # Energy storage: not in Table 1; inverter-based, no inertia (assumption)
}
"""(H in s, loading factor) per ENTSO-E production type code, [RoCoF] Table 1."""
CE_SELF_REGULATION_ENTSOE_PER_HZ = 0.02
"""Load self-regulation 2 %/Hz of the [RoCoF] reference model for 2020 scenarios (p. 29)."""
NEIGHBOUR_INERTIA = Path(__file__).with_name("neighbour_inertia_entsoe.csv")
"""Kinetic energy per CE country (Germany excluded) and case hour by [RoCoF] eq. 5.5-5.7 from
ENTSO-E Transparency actual generation per production type (A75); columns ``hour``,
``time_utc``, ``country``, ``generation_mw``, ``kinetic_energy_mws``, ``source``."""
BORDER_ROUTE: Mapping[str, Mapping[str, float]] = {
    "BE": {"NL": 0.5, "FR": 0.5},
    "ES": {"FR": 1.0},
    "PT": {"FR": 1.0},
    "IT": {"FR": 1 / 3, "CH": 1 / 3, "AT": 1 / 3},
    "SK": {"CZ": 1.0},
    "LT": {"PL": 1.0},
    "LV": {"PL": 1.0},
    "EE": {"PL": 1.0},
    **{c: {"AT": 0.5, "CZ": 0.5} for c in ("HU", "SI", "HR", "BA", "RS", "ME", "MK", "AL", "XK",
                                          "GR", "BG", "RO", "MD")},
}  # fmt: skip
"""Assumption: share of a country without an AC border node that reaches Germany through each
neighbour, by the first AC neighbours on the way (Belgium's link to Germany is HVDC)."""


def entsoe_kinetic_energy_mws(generation_mw: Mapping[str, float]) -> float:
    """Kinetic energy ``sum(H_i * P_i / LF_i)`` of [RoCoF] eq. 5.5-5.7 for generation per type."""
    unknown = set(generation_mw) - set(ENTSOE_INERTIA)
    if unknown:
        raise DataError(f"no ENTSO-E inertia constant for production types {sorted(unknown)}")
    return sum(
        ENTSOE_INERTIA[k][0] * max(float(p), 0.0) / ENTSOE_INERTIA[k][1]
        for k, p in generation_mw.items()
    )


def neighbour_kinetic_energy(hour: str, path: str | Path = NEIGHBOUR_INERTIA) -> dict[str, float]:
    """Kinetic energy in MWs per CE country, Germany excluded, at a case hour (``high_wind``)."""
    table = pd.read_csv(path)
    rows = table[table["hour"].eq(hour)]
    if rows.empty:
        raise DataError(f"no neighbour kinetic energy for hour {hour!r} in {path}")
    return dict(zip(rows["country"], rows["kinetic_energy_mws"].astype(float), strict=True))


def neighbour_generation_mw(hour: str, path: str | Path = NEIGHBOUR_INERTIA) -> float:
    """Total generation of the CE countries other than Germany at a case hour."""
    table = pd.read_csv(path)
    return float(table.loc[table["hour"].eq(hour), "generation_mw"].sum())


BORDER_SHORT_CIRCUIT = Path(__file__).with_name("border_short_circuit.csv")
"""Short-circuit power Sk'' per border node (``node`` is the bus name), with source and label."""
BORDER_SK_GVA_BY_KV: Mapping[int, float] = {380: 30.0, 220: 10.0}
"""Estimate for a border node without its own row: typical Sk'' per voltage level, within the
63 kA breaker bound (45.8 GVA at 420 kV, 26.7 GVA at 245 kV)."""


def border_short_circuit_gva(name: str, nominal_kv: float) -> float:
    """Sk'' of a border node: its own row in :data:`BORDER_SHORT_CIRCUIT`, else the level value."""
    table = pd.read_csv(BORDER_SHORT_CIRCUIT).set_index("node")["sk_gva"]
    if name in table.index:
        return float(table[name])
    level = min(BORDER_SK_GVA_BY_KV, key=lambda kv: abs(kv - nominal_kv))
    return BORDER_SK_GVA_BY_KV[level]


@dataclass(frozen=True)
class NeighbourEquivalent:
    """Machines that stand for the rest of the synchronous area at the border nodes.

    Attributes:
        kinetic_energy_mws: kinetic energy of the whole synchronous area; the German machines
            of the case are subtracted and the rest goes to the neighbours.
        h_s: inertia constant of each equivalent machine on its rating.
        damping_mw_per_hz: load self-regulation outside the model, as machine damping.
        fcr_mw_per_hz: FCR gain of the neighbours.
        fcr_max_mw: FCR of the neighbours at full activation.
        t_valve_s, t_lead_s, t_lag_s: TGOV1 T1, T2, T3.
        short_circuit_scale: factor on every border Sk'' (e.g. 0.7 and 1.3 to bound results).
        includes_germany: whether ``kinetic_energy_mws`` is the whole synchronous area (the
            German machines are subtracted) or only the neighbours.
        country_energy_mws: kinetic energy per country; when given, each border node gets the
            energy of its country plus the routed share of countries without an AC border node
            (:data:`BORDER_ROUTE`), split over that country's nodes by Vn**2. FCR and
            self-regulation follow the same shares. Otherwise all is split by Vn**2.
    """

    kinetic_energy_mws: float
    h_s: float = 5.0
    damping_mw_per_hz: float = 0.0
    fcr_mw_per_hz: float = 0.0
    fcr_max_mw: float = 0.0
    t_valve_s: float = 0.5
    t_lead_s: float = 0.0
    t_lag_s: float = 8.0
    short_circuit_scale: float = 1.0
    includes_germany: bool = True
    country_energy_mws: Mapping[str, float] | None = None


def continental_europe_equivalent(
    german_load_mw: float,
    inertia: Literal["low", "central", "high", "entsoe_hourly"] = "central",
    *,
    hour: str | None = None,
    inertia_scale: float = 1.0,
    self_regulation_per_hz: float | None = None,
    short_circuit_scale: float = 1.0,
    split: Literal["voltage", "country"] = "voltage",
) -> NeighbourEquivalent:
    """CE equivalent for an hour with the given German load.

    ``low``/``central``/``high``: the CE load is ``german_load_mw / DE_LOAD_SHARE`` and the
    kinetic energy of the whole area ``T_N * load / 2``; the German machines are subtracted later.
    ``entsoe_hourly``: the neighbours' kinetic energy of the case ``hour`` by [RoCoF] eq. 5.5-5.7
    from the actual generation mix (:data:`NEIGHBOUR_INERTIA`), per country, times
    ``inertia_scale``; the CE load is the German load plus the neighbours' generation, and the
    self-regulation defaults to 2 %/Hz ([RoCoF] p. 29) instead of 1 %/Hz. ``split="voltage"``
    (default) spreads the total over the border nodes by Vn**2 as for the other options;
    ``split="country"`` gives each node its country's energy (:func:`border_shares`). The country
    split leaves tiny machines (26 MVA at the Luxembourg nodes) behind a 10-30 GVA Sk'' and roughly
    halved the CCT in two of five checked cases, so it is not the default.

    The neighbours carry the non-German FCR (``CE_FCR_MW * (1 - DE_FCR_SHARE)``, linear up to
    200 mHz) and the self-regulation of the whole CE load, since the German loads of the case are
    constant power. H = 5 s is the [FSC] example value
    (T_A = 10 s per synchronous generator). TGOV1 T1 = 0.5 s and T3 = 8 s are assumed so that
    the FCR is 85 % active after 15 s and 95 % after 24 s, within the 30 s of [SOGL]; its step
    response is close to the 10 s first-order lag of [RoCoF] p. 29 (63 % after 8.5 s, not 10 s).
    """
    fcr = CE_FCR_MW * (1.0 - DE_FCR_SHARE)
    common = {
        "h_s": 5.0,
        "fcr_mw_per_hz": fcr / CE_FCR_FULL_HZ,
        "fcr_max_mw": fcr,
        "short_circuit_scale": short_circuit_scale,
    }
    if inertia == "entsoe_hourly":
        if hour is None:
            raise DataError("inertia='entsoe_hourly' needs the case hour")
        countries = {c: e * inertia_scale for c, e in neighbour_kinetic_energy(hour).items()}
        ce_load = german_load_mw + neighbour_generation_mw(hour)
        regulation = (
            CE_SELF_REGULATION_ENTSOE_PER_HZ
            if self_regulation_per_hz is None
            else self_regulation_per_hz
        )
        return NeighbourEquivalent(
            kinetic_energy_mws=sum(countries.values()),
            damping_mw_per_hz=regulation * ce_load,
            includes_germany=False,
            country_energy_mws=countries if split == "country" else None,
            **common,
        )
    ce_load = german_load_mw / DE_LOAD_SHARE
    regulation = (
        CE_SELF_REGULATION_PER_HZ if self_regulation_per_hz is None else self_regulation_per_hz
    )
    return NeighbourEquivalent(
        kinetic_energy_mws=CE_ACCELERATION_TIME_S[inertia] * ce_load / 2.0 * inertia_scale,
        # all CE self-regulation, the German share included: the German loads are constant power
        damping_mw_per_hz=regulation * ce_load,
        **common,  # type: ignore[arg-type]
    )


def border_shares(
    names: Mapping[str, str],
    nominal_kv: Mapping[str, float],
    country_energy_mws: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Share of the neighbour equivalent per border node (``names``: node id to bus name).

    Without ``country_energy_mws`` the shares are proportional to Vn**2. With it, each country's
    energy goes to its own AC border nodes, or through :data:`BORDER_ROUTE` to the neighbours on
    the way, and within a country by Vn**2; energy of a country with neither goes by Vn**2 to all.
    """
    weight = {m: nominal_kv[m] ** 2 for m in names}
    if not country_energy_mws:
        total = sum(weight.values())
        return {m: w / total for m, w in weight.items()}
    country = pd.read_csv(BORDER_SHORT_CIRCUIT).set_index("node")["country"].to_dict()
    nodes: dict[str, list[str]] = {}
    for m, name in names.items():
        if name not in country:
            raise DataError(f"border node {name!r} has no country in {BORDER_SHORT_CIRCUIT.name}")
        nodes.setdefault(str(country[name]), []).append(m)
    energy = dict.fromkeys(names, 0.0)

    def spread(e: float, members: list[str]) -> None:
        w = sum(weight[m] for m in members)
        for m in members:
            energy[m] += e * weight[m] / w

    for c, e in country_energy_mws.items():
        if c in nodes:
            spread(e, nodes[c])
            continue
        route = {a: r for a, r in BORDER_ROUTE.get(c, {}).items() if a in nodes}
        if not route:
            spread(e, list(names))
            continue
        norm = sum(route.values())
        for a, r in route.items():
            spread(e * r / norm, nodes[a])
    total = sum(energy.values())
    return {m: e / total for m, e in energy.items()}


def build_connected_case(
    tables: Any,
    voltage_pu: Mapping[str, float],
    machine_power: Mapping[str, tuple[float, float]],
    static_power: Mapping[str, tuple[float, float]],
    neighbour_power: Mapping[str, float],
    out_path: str | Path,
    *,
    machine_model: Literal["genrou", "gencls"] = "genrou",
    controls: bool = True,
    neighbour_mva: float = _EXTERNAL_MVA,
    min_machine_mva: float = 0.0,
    frequency_hz: float = 50.0,
    hydro_governor: Literal["hygov", "tgov1"] = "hygov",
    neighbour_equivalent: NeighbourEquivalent | None = None,
) -> Path:
    """Write connected tables with a solved operating point as an ANDES ``.xlsx`` case.

    The operating point comes from an AC power flow with reactive limits: voltage-holding machines
    and neighbours keep the solved voltage, so a limit reached there stays respected. Units without
    a dynamic model are constant injections at their solved power.

    Args:
        tables: connected model, with its terminal buses and lumped 110 kV buses.
        voltage_pu: solved voltage per bus ``mrid``.
        machine_power: solved (MW, Mvar) of each terminal bus with a dynamic machine.
        static_power: solved (MW, Mvar) of the other generation per bus.
        neighbour_power: solved MW per foreign bus.
        out_path: case file to write.
        machine_model: ``genrou`` or ``gencls``.
        controls: add SEXS and a governor to GENROU machines: TGOV1 for thermal units.
        hydro_governor: governor of hydro and pumped-storage machines, ``hygov`` or ``tgov1``.
        neighbour_mva: rating of the machine equivalent at each foreign bus; used only without
            ``neighbour_equivalent`` (the earlier assumed equivalent: H 5 s, D 2 pu, no governor).
        neighbour_equivalent: Continental Europe equivalent (see
            :func:`continental_europe_equivalent`). Its kinetic energy minus that of the German
            machines of this case is split over the border nodes in proportion to ``Vn**2`` (a
            proxy for the surge-impedance loading of the border circuits); each node gets the same
            share of the self-regulation (as damping D) and of the FCR (as a TGOV1 governor whose
            valve limits are the solved power plus or minus its FCR share).
        min_machine_mva: a terminal bus whose machines are rated below this becomes a constant
            injection at its solved power, so very small units do not drive the stability label.
        frequency_hz: nominal frequency.

    Raises:
        SimulationError: if the AC power flow does not converge.
    """
    _load_openmp_libraries_first()
    andes = require("andes", "andes")
    vl_kv = dict(
        zip(tables.voltage_level["mrid"], tables.voltage_level["nominal_voltage_kv"], strict=True)
    )
    lines = tables.ac_line_segment
    lines = lines[lines["status"].fillna("in_operation").eq("in_operation")]
    lines = lines[lines["bus1_mrid"] != lines["bus2_mrid"]]
    transformers = tables.power_transformer
    transformers = transformers[transformers["bus1_mrid"] != transformers["bus2_mrid"]]
    bus = tables.busbar_section
    pairs = list(zip(lines["bus1_mrid"], lines["bus2_mrid"], strict=True))
    pairs += list(zip(transformers["bus1_mrid"], transformers["bus2_mrid"], strict=True))
    keep = _main_component(list(bus["mrid"]), pairs)
    bus = bus[bus["mrid"].isin(keep)].reset_index(drop=True)
    lines = lines[lines["bus1_mrid"].isin(keep)]
    transformers = transformers[transformers["bus1_mrid"].isin(keep)]
    ids = {m: k + 1 for k, m in enumerate(bus["mrid"])}
    kv = {m: float(vl_kv[v]) for m, v in zip(bus["mrid"], bus["voltage_level_mrid"], strict=True)}

    ss = andes.System()
    ss.config.freq = frequency_hz
    for row in bus.itertuples():
        ss.add("Bus", {"idx": ids[row.mrid], "name": str(row.name), "Vn": kv[row.mrid]})
    for k, row in enumerate(tables.energy_consumer.itertuples()):
        if row.bus_mrid in keep:
            ss.add(
                "PQ",
                {
                    "idx": f"L{k}",
                    "bus": ids[row.bus_mrid],
                    "Vn": kv[row.bus_mrid],
                    "p0": float(row.p_mw) / _BASE_MVA,
                    "q0": float(row.q_mvar) / _BASE_MVA,
                },
            )
    for k, (mrid, (p, q)) in enumerate(static_power.items()):
        if mrid in keep and (p or q):
            ss.add(
                "PQ",
                {
                    "idx": f"I{k}",
                    "bus": ids[mrid],
                    "Vn": kv[mrid],
                    "p0": -p / _BASE_MVA,
                    "q0": -q / _BASE_MVA,
                },
            )
    for k, row in enumerate(lines.itertuples()):
        vn = kv[row.bus1_mrid]
        r, x = line_parameters(float(row.r_ohm), float(row.x_ohm), vn)
        ss.add(
            "Line",
            {
                "idx": f"Line_{k}",
                "bus1": ids[row.bus1_mrid],
                "bus2": ids[row.bus2_mrid],
                "r": r,
                "x": x,
                "b": float(row.bch_s) * vn**2 / _BASE_MVA,
                "Vn1": vn,
                "Vn2": kv[row.bus2_mrid],
                "trans": 0,
            },
        )
    for k, row in enumerate(transformers.itertuples()):
        vn1, vn2 = kv[row.bus1_mrid], kv[row.bus2_mrid]
        u1, u2 = float(row.rated_u1_kv), float(row.rated_u2_kv)
        ratio2 = (u2 / u1) ** 2
        r, x = line_parameters(float(row.r_ohm) * ratio2, float(row.x_ohm) * ratio2, vn2)
        # magnetising admittance as in the power-flow models (given on the rated_u2 side, placed at
        # the winding-1 terminal behind the ratio); ANDES Line would split it over both ends, so it
        # is added as a bus-1 shunt scaled by 1/tap**2
        g_s, b_s = (float(getattr(row, c, 0.0) or 0.0) for c in ("g_s", "b_s"))
        tap = (u1 / vn1) / (u2 / vn2)
        shunt = vn2**2 / _BASE_MVA / ratio2 / tap**2
        g_s, b_s = (g_s if g_s == g_s else 0.0) * shunt, (b_s if b_s == b_s else 0.0) * shunt
        if g_s or b_s:
            ss.add(
                "Shunt",
                {
                    "idx": f"TrSh_{k}",
                    "bus": ids[row.bus1_mrid],
                    "Vn": vn1,
                    "g": g_s,
                    "b": b_s,
                    "fn": frequency_hz,
                },
            )
        ss.add(
            "Line",
            {
                "idx": f"Tr_{k}",
                "bus1": ids[row.bus1_mrid],
                "bus2": ids[row.bus2_mrid],
                "r": r,
                "x": x,
                "Vn1": vn1,
                "Vn2": vn2,
                "tap": (u1 / vn1) / (u2 / vn2),
                "trans": 1,
            },
        )

    machines = tables.synchronous_machine
    machines = machines[machines["bus_mrid"].isin(machine_power)].copy()
    machines["group"] = machines["fuel"].map(fuel_class)
    rows: list[dict[str, Any]] = []
    neighbour_rows: list[dict[str, Any]] = []
    machine_kinetic = 0.0
    for k, (mrid, (p_mw, _)) in enumerate(machine_power.items()):
        if mrid not in keep:
            continue
        here = machines[machines["bus_mrid"].eq(mrid)]
        rated = pd.to_numeric(here["rating_mva"], errors="coerce")
        sn = float(rated.sum()) if rated.notna().any() else p_mw / 0.8
        if sn < min_machine_mva:
            q_mvar = machine_power[mrid][1]
            ss.add(
                "PQ",
                {
                    "idx": f"S{k}",
                    "bus": ids[mrid],
                    "Vn": kv[mrid],
                    "p0": -p_mw / _BASE_MVA,
                    "q0": -q_mvar / _BASE_MVA,
                },
            )
            rows.append(
                {"machine": f"S{k}", "bus": mrid, "sn_mva": sn, "p_mw": p_mw, "small": True}
            )
            continue
        group = str(here["group"].mode().iloc[0]) if len(here) else "thermal"
        group = group if group in MACHINE_VALUES.index else "thermal"
        vn = kv[mrid]
        ss.add(
            "PV",
            {
                "idx": f"G{k}",
                "bus": ids[mrid],
                "Vn": vn,
                "Sn": sn,
                "p0": p_mw / _BASE_MVA,
                "v0": float(voltage_pu[mrid]),
                "pmax": 10 * sn / _BASE_MVA,
                "pmin": -10 * sn / _BASE_MVA,
                "qmax": 10 * sn / _BASE_MVA,
                "qmin": -10 * sn / _BASE_MVA,
            },
        )
        v: dict[Any, Any] = MACHINE_VALUES.loc[group].to_dict()
        common = {
            "idx": f"M{k}",
            "bus": ids[mrid],
            "gen": f"G{k}",
            "Sn": sn,
            "Vn": vn,
            "M": 2 * float(v["h_s"]),
            "D": _DAMPING_PU,
            "xd1": float(v["xd1"]),
            "ra": float(v["ra"]),
        }
        machine_kinetic += float(v["h_s"]) * sn
        if machine_model == "gencls":
            ss.add("GENCLS", common)
            rows.append(
                {"machine": f"M{k}", "bus": mrid, "sn_mva": sn, "p_mw": p_mw, "small": False}
            )
            continue
        ss.add(
            "GENROU",
            common
            | {
                "xd": float(v["xd"]),
                "xq": float(v["xq"]),
                "xq1": float(v["xq1"]),
                "xd2": float(v["xd2"]),
                "xq2": float(v["xd2"]),
                "xl": float(v["xl"]),
                "Td10": float(v["td10"]),
                "Tq10": float(v["tq10"]),
                "Td20": float(v["td20"]),
                "Tq20": float(v["tq20"]),
            },
        )
        if controls:
            ss.add("SEXS", {"idx": f"E{k}", "syn": f"M{k}"} | dict(CONTROLS["SEXS"]))
            gov = "HYGOV" if group == "hydro" and hydro_governor == "hygov" else "TGOV1"
            ss.add(gov, {"idx": f"T{k}", "syn": f"M{k}", "Tn": sn} | dict(CONTROLS[gov]))
        rows.append(
            {
                "machine": f"M{k}", "bus": mrid, "sn_mva": sn, "p_mw": p_mw, "small": False,
                "group": group,
            }
        )  # fmt: skip
    foreign = [m for m in neighbour_power if m in keep]
    largest = max(foreign, key=lambda m: abs(neighbour_power[m]), default=None)
    if largest is None:
        raise DataError("no neighbouring network to hold the reference")
    eq = neighbour_equivalent
    external: dict[Any, Any] = DYNAMIC_VALUES.loc["external"].to_dict()
    names = dict(zip(bus["mrid"], bus["name"].astype(str), strict=True))
    shares = border_shares(
        {m: names[m] for m in foreign}, kv, None if eq is None else eq.country_energy_mws
    )
    neighbour_kinetic = 0.0
    if eq is not None:
        neighbour_kinetic = eq.kinetic_energy_mws - (
            machine_kinetic if eq.includes_germany else 0.0
        )
        if neighbour_kinetic <= 0:
            raise DataError("the German machines alone exceed the target kinetic energy")
    for k, (mrid, p_mw) in enumerate(neighbour_power.items()):
        if mrid not in keep:
            continue
        vn = kv[mrid]
        share = shares[mrid]
        if eq is None:
            h_s, damping = float(external["h_s"]), _DAMPING_PU
            sn = neighbour_mva
            xd1 = float(external["xd1_pu"])
        else:
            h_s = eq.h_s
            sn = neighbour_kinetic * share / h_s
            # D on the machine base: D * (omega - 1) * Sn equals the self-regulation in MW/Hz * df
            damping = eq.damping_mw_per_hz * share * frequency_hz / sn
            # the reactance stands for the neighbouring grid's short-circuit strength, not for the
            # inertia: xd1 = Sn / Sk'' on the machine base, so the ohmic value is Vn**2 / Sk''
            xd1 = sn / (1000.0 * border_short_circuit_gva(names[mrid], vn) * eq.short_circuit_scale)
        ss.add(
            "Slack" if mrid == largest else "PV",
            {
                "idx": f"X{k}",
                "bus": ids[mrid],
                "Vn": vn,
                "Sn": sn,
                "p0": p_mw / _BASE_MVA,
                "v0": float(voltage_pu[mrid]),
                "pmax": 999.0,
                "pmin": -999.0,
                "qmax": 999.0,
                "qmin": -999.0,
            },
        )
        ss.add(
            "GENCLS",
            {
                "idx": f"XM{k}",
                "bus": ids[mrid],
                "gen": f"X{k}",
                "Sn": sn,
                "Vn": vn,
                "M": 2 * h_s,
                "D": damping,
                "xd1": xd1,
                "ra": 0.0,
            },
        )
        if eq is not None and eq.fcr_mw_per_hz > 0:
            gain = eq.fcr_mw_per_hz * share  # MW/Hz
            reserve = eq.fcr_max_mw * share  # MW at the full-activation deviation
            ss.add(
                "TGOV1",
                {
                    "idx": f"XT{k}",
                    "syn": f"XM{k}",
                    "Tn": sn,
                    "R": sn / (frequency_hz * gain),
                    "T1": eq.t_valve_s,
                    "T2": eq.t_lead_s,
                    "T3": eq.t_lag_s,
                    "VMAX": (p_mw + reserve) / sn,
                    "VMIN": (p_mw - reserve) / sn,
                    "Dt": 0.0,
                },
            )
        neighbour_rows.append(
            {
                "machine": f"XM{k}",
                "bus": mrid,
                "sn_mva": sn,
                "h_s": h_s,
                "damping_pu": damping,
                "xd1_pu": xd1,
            }
        )
    _set_nominal_frequency(ss, frequency_hz)
    ss.setup()
    if not ss.PFlow.run():
        raise SimulationError("the AC power flow of the grid case does not converge")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    andes.io.xlsx.write(ss, str(out), overwrite=True)
    pd.DataFrame(rows).to_csv(out.with_suffix(".machines.csv"), index=False)
    pd.DataFrame(neighbour_rows).to_csv(out.with_suffix(".neighbours.csv"), index=False)
    pd.DataFrame({"bus": list(ids), "andes_bus": [ids[m] for m in ids]}).to_csv(
        out.with_suffix(".buses.csv"), index=False
    )
    return out
