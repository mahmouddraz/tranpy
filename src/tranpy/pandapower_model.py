"""The gold grid model in pandapower, to cross-check pypowsybl (pandapower needs pandas < 3).

Every plant of one bus is merged into at most two elements, because pandapower allows one voltage
setpoint per bus:

* ``voltage_control="table"`` follows ``voltage_control`` of the machine table, as pypowsybl does:
  a machine without it injects at unity power factor;
* ``voltage_control="fuel"`` makes inverters (solar, wind, storage) static generators at
  ``inverter_cos_phi`` and lets synchronous machines hold voltage where the bus dispatches at
  least ``min_pv_mw``; smaller synchronous generation is static as well.

Neighbouring networks are voltage-controlling generators at their imposed power. The slack is an
``ext_grid`` at the largest voltage-controlling bus of the main component.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from tranpy.powsybl_model import operating_lines
from tranpy.simulation.grid_case import _main_component, fuel_class
from tranpy_core._optional import require
from tranpy_core.errors import DataError

if TYPE_CHECKING:
    from tranpy.grid_layers import CimTables

__all__ = ["line_loading", "network_from_cim_tables"]

_TRANSFORMER_MVA = 100.0
_FREQUENCY_HZ = 50.0
_UNLIMITED_Q_RATIO = 1e3
_NEIGHBOUR_Q_MVAR = 1e6


def _shift_degree(row: Any, first_high: bool) -> float:
    """pandapower shift of the low side behind the high side, for a table angle as in pypowsybl."""
    angle = getattr(row, "phase_shift_deg", 0.0)
    if angle is None or np.isnan(angle):
        return 0.0
    return -float(angle) if first_high else float(angle)


def _plants(
    machines: pd.DataFrame,
    voltage_control: str,
    inverter_cos_phi: float,
    min_pv_mw: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(voltage-controlling, static) power per bus of the machines in service."""
    table = machines[machines["in_service"].astype(bool)].copy()
    table["inverter"] = table["fuel"].map(fuel_class).eq("inverter")
    if voltage_control == "table":
        table["pv"] = table["voltage_control"].astype(bool)
    else:
        synchronous = table[~table["inverter"]].groupby("bus_mrid")["p_mw"].sum()
        table["pv"] = ~table["inverter"] & (
            table["bus_mrid"].map(synchronous).fillna(0.0).ge(min_pv_mw)
        )
    pv = (
        table[table["pv"]]
        .groupby("bus_mrid", as_index=False)
        .agg(
            p_mw=("p_mw", "sum"), max_p_mw=("max_p_mw", "sum"), target_v_pu=("target_v_pu", "first")
        )
    )
    rest = table[~table["pv"]].copy()
    tan_phi = float(np.tan(np.arccos(inverter_cos_phi)))
    rest["q_mvar"] = rest["p_mw"] * tan_phi * rest["inverter"].astype(float)
    if "target_q_mvar" in rest:
        rest["q_mvar"] += rest["target_q_mvar"].astype(float).fillna(0.0)
    static = rest.groupby("bus_mrid", as_index=False).agg(
        p_mw=("p_mw", "sum"), q_mvar=("q_mvar", "sum")
    )
    return pv, static


def network_from_cim_tables(
    tables: CimTables,
    *,
    voltage_control: Literal["table", "fuel"] = "table",
    inverter_cos_phi: float = 1.0,
    min_pv_mw: float = 0.0,
    q_limit_ratio: float | None = None,
    compensation: pd.DataFrame | None = None,
) -> Any:
    """pandapower network of the gold tables.

    Args:
        tables: the model (``CimTables``).
        voltage_control: which machines hold voltage, see the module description.
        inverter_cos_phi: power factor of inverters (producing reactive power when below 1).
        min_pv_mw: least dispatched synchronous power for a bus to hold voltage (``"fuel"``).
        q_limit_ratio: reactive limit of a German plant as a share of its ``max_p_mw``, for
            ``runpp(enforce_q_lims=True)``; neighbours stay unlimited.
        compensation: units of :func:`tranpy.reactive_power.read_compensation`: controlled ones
            become generators without active power (at the set point of a bus that already
            holds voltage), shunts pandapower shunts. A machine column ``target_q_mvar`` adds
            fixed reactive power to the static injections.

    Raises:
        DataError: if no bus of the main component holds voltage.
    """
    pp = require("pandapower", "pandapower")
    net = pp.create_empty_network(f_hz=_FREQUENCY_HZ, sn_mva=100.0)
    kv_of = dict(
        zip(tables.voltage_level["mrid"], tables.voltage_level["nominal_voltage_kv"], strict=False)
    )
    index = {}
    for row in tables.busbar_section.itertuples():
        index[row.mrid] = pp.create_bus(
            net, vn_kv=float(kv_of[row.voltage_level_mrid]), name=row.mrid
        )

    lines = operating_lines(tables)
    lines = lines[lines["bus1_mrid"] != lines["bus2_mrid"]]
    for row in lines.itertuples():
        # the tables hold totals; a line without a usable length keeps them on one nominal km
        km = float(row.length_km) if float(row.length_km) > 0 else 1.0
        pp.create_line_from_parameters(
            net,
            index[row.bus1_mrid],
            index[row.bus2_mrid],
            length_km=km,
            r_ohm_per_km=float(row.r_ohm) / km,
            x_ohm_per_km=float(row.x_ohm) / km,
            c_nf_per_km=float(row.bch_s) / (2 * np.pi * _FREQUENCY_HZ) * 1e9 / km,
            max_i_ka=float(row.rated_current_a) / 1000.0,
            name=row.mrid,
        )
    transformers = tables.power_transformer
    transformers = transformers[transformers["bus1_mrid"] != transformers["bus2_mrid"]]
    for row in transformers.itertuples():
        # table values refer to side 1, which is not always the high-voltage side
        u1, u2 = float(row.rated_u1_kv), float(row.rated_u2_kv)
        first_high = u1 >= u2
        z_base = u1**2 / _TRANSFORMER_MVA
        vkr = 100.0 * float(row.r_ohm) / z_base
        vk = 100.0 * float(np.hypot(row.r_ohm, row.x_ohm)) / z_base
        y_pu = float(np.hypot(row.g_s, row.b_s)) * z_base
        pp.create_transformer_from_parameters(
            net,
            hv_bus=index[row.bus1_mrid if first_high else row.bus2_mrid],
            lv_bus=index[row.bus2_mrid if first_high else row.bus1_mrid],
            sn_mva=_TRANSFORMER_MVA,
            vn_hv_kv=max(u1, u2),
            vn_lv_kv=min(u1, u2),
            vkr_percent=vkr,
            vk_percent=vk,
            pfe_kw=float(row.g_s) * u1**2 * 1000.0,
            i0_percent=100.0 * y_pu,
            shift_degree=_shift_degree(row, first_high),
            name=row.mrid,
        )

    pairs = list(zip(lines["bus1_mrid"], lines["bus2_mrid"], strict=False))
    pairs += list(zip(transformers["bus1_mrid"], transformers["bus2_mrid"], strict=False))
    main = _main_component(list(index), pairs)

    pv, static = _plants(tables.synchronous_machine, voltage_control, inverter_cos_phi, min_pv_mw)
    candidates = pv[pv["bus_mrid"].isin(main)]
    if candidates.empty:
        raise DataError("no bus of the main component holds voltage")
    slack = candidates.loc[candidates["max_p_mw"].idxmax(), "bus_mrid"]
    limit = _UNLIMITED_Q_RATIO if q_limit_ratio is None else q_limit_ratio
    for row in pv.itertuples():
        if row.bus_mrid == slack:
            pp.create_ext_grid(
                net,
                index[row.bus_mrid],
                vm_pu=float(row.target_v_pu),
                slack_weight=float(row.max_p_mw),
                name="slack",
            )
            # distributed slack ignores any set point on ext_grid, so its dispatch is an injection
            pp.create_sgen(net, index[row.bus_mrid], p_mw=float(row.p_mw), name="slack dispatch")
            continue
        q = float(row.max_p_mw) * limit
        pp.create_gen(
            net,
            index[row.bus_mrid],
            p_mw=float(row.p_mw),
            vm_pu=float(row.target_v_pu),
            slack_weight=float(row.max_p_mw),
            min_q_mvar=-q,
            max_q_mvar=q,
            name=row.bus_mrid,
        )
    for row in static.itertuples():
        pp.create_sgen(
            net, index[row.bus_mrid], p_mw=float(row.p_mw), q_mvar=float(row.q_mvar),
            name=row.bus_mrid,
        )  # fmt: skip
    held = dict(zip(pv["bus_mrid"], pv["target_v_pu"].astype(float), strict=True))
    for row in tables.external_network_injection.itertuples():
        if row.bus_mrid in held:
            pp.create_sgen(net, index[row.bus_mrid], p_mw=float(row.p_mw), name=row.mrid)
            continue
        held[row.bus_mrid] = float(row.target_v_pu)
        pp.create_gen(
            net,
            index[row.bus_mrid],
            p_mw=float(row.p_mw),
            vm_pu=float(row.target_v_pu),
            slack_weight=0.0,
            min_q_mvar=-_NEIGHBOUR_Q_MVAR,
            max_q_mvar=_NEIGHBOUR_Q_MVAR,
            name=row.mrid,
        )
    for row in tables.energy_consumer.itertuples():
        pp.create_load(net, index[row.bus_mrid], p_mw=float(row.p_mw), q_mvar=float(row.q_mvar))
    if compensation is not None:
        _add_compensation(pp, net, index, compensation, held)
    return net


def _add_compensation(
    pp: Any, net: Any, index: dict[str, int], units: pd.DataFrame, held: dict[str, float]
) -> None:
    columns = ("mrid", "bus_mrid", "kind", "q_max_mvar", "q_min_mvar", "target_v_pu")
    for mrid, bus_mrid, kind, q_max, q_min, target in units[list(columns)].to_numpy().tolist():
        bus = index[str(bus_mrid)]
        if kind == "shunt":
            # pandapower counts shunt reactive power as consumption
            pp.create_shunt(net, bus, q_mvar=-float(q_max), p_mw=0.0, name=mrid)
            continue
        vm = held.setdefault(str(bus_mrid), float(target))
        pp.create_gen(
            net, bus, p_mw=0.0, vm_pu=vm, slack_weight=0.0,
            min_q_mvar=float(q_min), max_q_mvar=float(q_max), name=mrid,
        )  # fmt: skip


def line_loading(net: Any) -> pd.Series:
    """Largest end current / rated current per line of a solved network, by mrid."""
    current = net.res_line[["i_from_ka", "i_to_ka"]].max(axis=1)
    loading = (current / net.line["max_i_ka"]).dropna()
    return pd.Series(loading.to_numpy(), index=net.line.loc[loading.index, "name"], name="loading")
