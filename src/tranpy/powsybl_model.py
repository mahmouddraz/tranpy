"""The gold grid model in pypowsybl (optional ``[powsybl]`` extra).

* :func:`network_from_cim_tables` - a pypowsybl network (bus-breaker topology) from the
  tables of :func:`tranpy.grid_layers.cim_tables`;
* :func:`run_load_flow` - AC power flow (OpenLoadFlow, distributed slack on generation);
* :func:`export_cgmes` - the network as a CGMES (CIM) archive.

pypowsybl expects transformer impedances referred to the second (low-voltage) end; the
published parameters refer to the high-voltage end and are converted with the square of
the rated voltage ratio.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from tranpy_core._optional import require

if TYPE_CHECKING:
    from tranpy.grid_layers import CimTables

__all__ = [
    "LoadFlowSummary",
    "export_cgmes",
    "line_ids",
    "line_loading",
    "network_from_cim_tables",
    "run_load_flow",
]


def _ids(prefix: str, mrids: pd.Series) -> list[str]:
    return [f"{prefix}_{m.replace('-', '')}" for m in mrids]


def network_from_cim_tables(tables: CimTables, name: str = "germany_hv") -> Any:
    """pypowsybl network of the gold tables (circuits with both ends at one bus skipped)."""
    pp = require("pypowsybl", "powsybl")
    network = pp.network.create_empty(name)
    sub = tables.substation
    german = sub["country"] == "DE"
    network.create_substations(
        id=_ids("S", sub.loc[german, "mrid"]),
        name=sub.loc[german, "site"].tolist(),
        country=["DE"] * int(german.sum()),
    )
    network.create_substations(
        id=_ids("S", sub.loc[~german, "mrid"]), name=sub.loc[~german, "site"].tolist()
    )
    vl = tables.voltage_level
    vl_id = dict(zip(vl["mrid"], _ids("VL", vl["mrid"]), strict=True))
    sub_id = dict(zip(sub["mrid"], _ids("S", sub["mrid"]), strict=True))
    network.create_voltage_levels(
        id=[vl_id[m] for m in vl["mrid"]],
        substation_id=[sub_id[m] for m in vl["substation_mrid"]],
        topology_kind=["BUS_BREAKER"] * len(vl),
        nominal_v=vl["nominal_voltage_kv"].tolist(),
        low_voltage_limit=(vl["nominal_voltage_kv"] * 0.9).tolist(),
        high_voltage_limit=(vl["nominal_voltage_kv"] * 1.1).tolist(),
    )
    bus = tables.busbar_section
    bus_id = dict(zip(bus["mrid"], _ids("B", bus["mrid"]), strict=True))
    bus_vl = dict(zip(bus["mrid"], [vl_id[m] for m in bus["voltage_level_mrid"]], strict=True))
    network.create_buses(
        id=[bus_id[m] for m in bus["mrid"]], voltage_level_id=[bus_vl[m] for m in bus["mrid"]]
    )

    line = operating_lines(tables)
    line = line[line["bus1_mrid"] != line["bus2_mrid"]]
    half_b = line["bch_s"].astype(float) / 2
    network.create_lines(
        id=_ids("L", line["mrid"]),
        voltage_level1_id=[bus_vl[m] for m in line["bus1_mrid"]],
        bus1_id=[bus_id[m] for m in line["bus1_mrid"]],
        voltage_level2_id=[bus_vl[m] for m in line["bus2_mrid"]],
        bus2_id=[bus_id[m] for m in line["bus2_mrid"]],
        r=line["r_ohm"].tolist(),
        x=line["x_ohm"].tolist(),
        g1=[0.0] * len(line),
        b1=half_b.tolist(),
        g2=[0.0] * len(line),
        b2=half_b.tolist(),
    )

    tr = tables.power_transformer
    tr = tr[tr["bus1_mrid"] != tr["bus2_mrid"]]
    ratio2 = (tr["rated_u2_kv"] / tr["rated_u1_kv"]) ** 2
    network.create_2_windings_transformers(
        id=_ids("T", tr["mrid"]),
        voltage_level1_id=[bus_vl[m] for m in tr["bus1_mrid"]],
        bus1_id=[bus_id[m] for m in tr["bus1_mrid"]],
        voltage_level2_id=[bus_vl[m] for m in tr["bus2_mrid"]],
        bus2_id=[bus_id[m] for m in tr["bus2_mrid"]],
        rated_u1=tr["rated_u1_kv"].tolist(),
        rated_u2=tr["rated_u2_kv"].tolist(),
        r=(tr["r_ohm"] * ratio2).tolist(),
        x=(tr["x_ohm"] * ratio2).tolist(),
        g=(tr["g_s"] / ratio2).tolist(),
        b=(tr["b_s"] / ratio2).tolist(),
    )

    nominal = dict(
        zip(
            bus["mrid"],
            [vl.set_index("mrid").at[m, "nominal_voltage_kv"] for m in bus["voltage_level_mrid"]],
            strict=True,
        )
    )
    gen = tables.synchronous_machine
    ext = tables.external_network_injection
    network.create_generators(
        id=_ids("G", gen["mrid"]) + _ids("X", ext["mrid"]),
        voltage_level_id=[bus_vl[m] for m in gen["bus_mrid"]]
        + [bus_vl[m] for m in ext["bus_mrid"]],
        bus_id=[bus_id[m] for m in gen["bus_mrid"]] + [bus_id[m] for m in ext["bus_mrid"]],
        min_p=gen["min_p_mw"].tolist() + [-10_000.0] * len(ext),
        max_p=gen["max_p_mw"].tolist() + [10_000.0] * len(ext),
        target_p=gen["p_mw"].tolist() + ext["p_mw"].tolist(),
        target_v=[nominal[m] * v for m, v in zip(gen["bus_mrid"], gen["target_v_pu"], strict=True)]
        + [nominal[m] * v for m, v in zip(ext["bus_mrid"], ext["target_v_pu"], strict=True)],
        target_q=[0.0] * (len(gen) + len(ext)),
        voltage_regulator_on=gen["voltage_control"].tolist() + [True] * len(ext),
    )
    load = tables.energy_consumer
    network.create_loads(
        id=_ids("C", load["mrid"]),
        voltage_level_id=[bus_vl[m] for m in load["bus_mrid"]],
        bus_id=[bus_id[m] for m in load["bus_mrid"]],
        p0=load["p_mw"].tolist(),
        q0=load["q_mvar"].tolist(),
    )
    out_of_service = _ids("G", gen.loc[~gen["in_service"].astype(bool), "mrid"])
    if out_of_service:
        network.update_generators(id=out_of_service, connected=[False] * len(out_of_service))
    return network


@dataclass(frozen=True)
class LoadFlowSummary:
    """Result of :func:`run_load_flow` for the main synchronous component."""

    status: str
    iterations: int
    slack_mismatch_mw: float
    v_min_pu: float
    v_max_pu: float
    max_line_loading: float
    load_mw: float
    generation_mw: float
    buses_solved: int

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def operating_lines(
    tables: CimTables, statuses: tuple[str, ...] = ("in_operation",)
) -> pd.DataFrame:
    """Lines with one of ``statuses`` (all lines when the table has no status column).

    Planning studies pass more statuses, for example ``("in_operation", "under_construction")``.
    """
    lines = tables.ac_line_segment
    if "status" not in lines:
        return lines
    return lines[lines["status"].fillna("in_operation").isin(statuses)]


def line_ids(tables: CimTables) -> dict[str, str]:
    """pypowsybl line id -> mrid of the ``ac_line_segment`` table."""
    return dict(
        zip(_ids("L", tables.ac_line_segment["mrid"]), tables.ac_line_segment["mrid"], strict=True)
    )


def line_loading(network: Any, tables: CimTables) -> pd.Series:
    """Largest end current / rated current per line (lines with a rating), by mrid."""
    lines = network.get_lines()
    rated = dict(
        zip(
            _ids("L", tables.ac_line_segment["mrid"]),
            tables.ac_line_segment["rated_current_a"].astype(float),
            strict=True,
        )
    )
    current = lines[["i1", "i2"]].abs().max(axis=1)
    loading = (current / lines.index.map(rated)).dropna()
    ids = line_ids(tables)
    return pd.Series(loading.to_numpy(), index=[ids[i] for i in loading.index], name="loading")


def run_load_flow(network: Any, tables: CimTables | None = None) -> LoadFlowSummary:
    """AC power flow with slack distributed over generation (proportional to max P).

    With ``tables`` the largest line loading against the rated currents is reported.
    """
    pp = require("pypowsybl", "powsybl")
    parameters = pp.loadflow.Parameters(
        distributed_slack=True,
        balance_type=pp.loadflow.BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX,
    )
    results = pp.loadflow.run_ac(network, parameters)
    main = results[0]
    buses = network.get_buses()
    vls = network.get_voltage_levels()
    solved = buses.dropna(subset=["v_mag"])
    v_pu = solved["v_mag"] / solved["voltage_level_id"].map(vls["nominal_v"])
    loading = np.nan
    if tables is not None:
        values = line_loading(network, tables)
        loading = float(values.max()) if len(values) else np.nan
    loads = network.get_loads()
    gens = network.get_generators()
    return LoadFlowSummary(
        status=str(main.status.name),
        iterations=int(main.iteration_count),
        slack_mismatch_mw=float(getattr(main, "slack_bus_active_power_mismatch", np.nan)),
        v_min_pu=float(v_pu.min()) if len(v_pu) else np.nan,
        v_max_pu=float(v_pu.max()) if len(v_pu) else np.nan,
        max_line_loading=loading,
        load_mw=float(loads["p"].sum()),
        generation_mw=float(-gens["p"].sum()),
        buses_solved=len(solved),
    )


def export_cgmes(network: Any, path: str | Path) -> Path:
    """Write the network as a CGMES archive: one ``.zip`` with the EQ, SSH, TP and SV
    profiles (CIM XML)."""
    out = Path(path).with_suffix(".zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / out.stem
        network.save(str(base), format="CGMES")
        profiles = sorted(Path(tmp).glob(f"{out.stem}_*.xml"))
        # built locally and copied: Unity Catalog volumes accept only sequential writes
        local = Path(tmp) / out.name
        with zipfile.ZipFile(local, "w", zipfile.ZIP_DEFLATED) as archive:
            for profile in profiles:
                archive.write(profile, profile.name)
        shutil.copyfile(local, out)
    return out
