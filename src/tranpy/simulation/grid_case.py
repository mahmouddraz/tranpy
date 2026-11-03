"""ANDES case of the German grid model built from the gold tables (optional ``[andes]`` extra).

Dynamic data per plant is not published, so every dynamic value here is an estimate:

* the plants of one bus are aggregated into one synchronous machine (GENCLS) and one inverter;
* inertia and transient reactance per fuel come from the ranges in Kundur, *Power System
  Stability and Control* (1994), Tables 3.2 and 4.3; damping is assumed;
* committed machine rating is ``dispatched power / machine_loading``, since the model has no
  unit commitment;
* inverters (solar, wind, storage) sit on a stub bus as REGCA1 converters without inertia and
  without plant controls, or as negative loads with ``inverter_model="pq"``;
* neighbouring networks are one assumed machine per border node;
* phase shifter angles, transformer magnetising branches and HVDC links are left out: ANDES has
  no two-terminal HVDC model.

:data:`DYNAMIC_VALUES` is saved beside the case by :func:`build_grid_case`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from tranpy.simulation.andes import _load_openmp_libraries_first
from tranpy_core._optional import require
from tranpy_core.errors import DataError, SimulationError

__all__ = [
    "CONTROLS",
    "DYNAMIC_VALUES",
    "MACHINE_VALUES",
    "build_connected_case",
    "build_grid_case",
    "fuel_class",
    "line_parameters",
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
        neighbour_mva: rating of the machine equivalent at each foreign bus.
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
    foreign = [m for m in neighbour_power if m in keep]
    largest = max(foreign, key=lambda m: abs(neighbour_power[m]), default=None)
    if largest is None:
        raise DataError("no neighbouring network to hold the reference")
    for k, (mrid, p_mw) in enumerate(neighbour_power.items()):
        if mrid not in keep:
            continue
        vn = kv[mrid]
        ss.add(
            "Slack" if mrid == largest else "PV",
            {
                "idx": f"X{k}",
                "bus": ids[mrid],
                "Vn": vn,
                "Sn": neighbour_mva,
                "p0": p_mw / _BASE_MVA,
                "v0": float(voltage_pu[mrid]),
                "pmax": 999.0,
                "pmin": -999.0,
                "qmax": 999.0,
                "qmin": -999.0,
            },
        )
        values: dict[Any, Any] = DYNAMIC_VALUES.loc["external"].to_dict()
        ss.add(
            "GENCLS",
            {
                "idx": f"XM{k}",
                "bus": ids[mrid],
                "gen": f"X{k}",
                "Sn": neighbour_mva,
                "Vn": vn,
                "M": 2 * float(values["h_s"]),
                "D": _DAMPING_PU,
                "xd1": float(values["xd1_pu"]),
                "ra": 0.0,
            },
        )
    rows: list[dict[str, Any]] = []
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
    _set_nominal_frequency(ss, frequency_hz)
    ss.setup()
    if not ss.PFlow.run():
        raise SimulationError("the AC power flow of the grid case does not converge")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    andes.io.xlsx.write(ss, str(out), overwrite=True)
    pd.DataFrame(rows).to_csv(out.with_suffix(".machines.csv"), index=False)
    pd.DataFrame({"bus": list(ids), "andes_bus": [ids[m] for m in ids]}).to_csv(
        out.with_suffix(".buses.csv"), index=False
    )
    return out
