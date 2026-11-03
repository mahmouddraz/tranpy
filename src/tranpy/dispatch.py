"""Dispatch of real hours: measured generation per type, load and cross-border flows.

Generation per control area and production type and physical flows come from ENTSO-E
Transparency, load per control area from SMARD. Within a control area and type, generation is
shared in proportion to installed capacity: the plant-level split is an assumption. Foreign nodes
in the model are stubs, so a border flow is imposed here and cannot be used to validate the model.
A zone reached only by HVDC has no AC path, so its flow is imposed at the German converter instead.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from tranpy.grid_layers import CimTables

FUEL_TYPES: Mapping[str, tuple[str, ...]] = {
    "Biomasse": ("B01",),
    "Klärschlamm": ("B01",),
    "Braunkohle": ("B02",),
    "Grubengas": ("B03", "B04"),
    "Erdgas": ("B03", "B04"),
    "andere Gase": ("B03", "B04"),
    "Wasserstoff": ("B03", "B04"),
    "Steinkohle": ("B05",),
    "Mineralölprodukte": ("B06",),
    "Geothermie": ("B09",),
    "Pumpspeicher": ("B10",),
    "Wasser": ("B11", "B12"),
    "Solarthermie": ("B15",),
    "Solare Strahlungsenergie": ("B16",),
    "nicht biogener Abfall": ("B17",),
    "Windenergie (Offshore-Anlage)": ("B18",),
    "Windenergie (Onshore-Anlage)": ("B19",),
    "Wärme": ("B20",),
    "Druck aus Gasleitungen": ("B20",),
    "Druck aus Wasserleitungen": ("B20",),
    "Speicher": ("B20",),
}
"""Register fuels of the model per ENTSO-E production type."""

COUNTRY_ZONES: Mapping[str, str] = {
    "AUT": "AT",
    "BEL": "BE",
    "CHE": "CH",
    "CZE": "CZ",
    "DNK": "DK1",
    "FRA": "FR",
    "LUX": "LU",
    "NLD": "NL",
    "NOR": "NO2",
    "POL": "PL",
}
"""Countries of the model's foreign nodes per neighbouring bidding zone."""

HVDC_CONVERTERS: Mapping[str, str] = {
    "BE": "oberzier|380",  # ALEGrO
    "DK2": "bentwisch|380",  # Kontek, Kriegers Flak Combined Grid Solution
    "NO2": "wilster/west|380",  # NordLink
    # Baltic Cable lands at Herrenwyk; the model has one 950 A route out of Luebeck, while published
    # flows exceed 600 MW for about 2,900 hours of 2025. Placed at Luebeck 220 kV as a stand-in for
    # the parallel routes: an assumption, not confirmed topology.
    "SE4": "lubeck|220",
}
"""German converter node per zone reached only by HVDC: its flow is a fixed injection there."""

_RENEWABLE_WIND = ("B18", "B19")
_ADDED_CONSUMERS = ("hvdc/", "pump/")
FEED_IN_LABEL = "estimated: published balance gap"
INCOMPLETE_COVERAGE: Mapping[str, tuple[str, ...]] = {
    "Erdgas": ("Amprion",),
    "Steinkohle": ("Amprion", "TenneT"),
    "andere Gase": ("50Hertz", "Amprion", "TenneT"),
    "Biomasse": ("50Hertz",),
}
"""Fuel and area with incomplete generation data, SMARD user handbook 04/2026, pp. 39, 71."""
DISPERSED_FUELS = ("Erdgas", "andere Gase", "Biomasse")


@dataclass(frozen=True)
class HourData:
    """Published values of one hour (all MW)."""

    time: pd.Timestamp
    generation: pd.Series  # (area, psr_type)
    load: pd.Series  # area
    exports: pd.Series  # zone, positive out of Germany
    pumping: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))  # area


@dataclass
class Dispatch:
    """Dispatched model of one hour and how far it reached the published values."""

    tables: CimTables
    allocated_mw: pd.Series
    unallocated_mw: pd.Series
    unplaced_exports_mw: pd.Series
    info: dict[str, Any] = field(default_factory=dict)


PUMPED_STORAGE = "B10"


def hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """Hourly means of a table with ``psr_type``, ``time``, ``mw``, one column per type.

    Publication intervals differ (15 to 60 minutes), so each interval's mean is taken first and
    a type missing in an hour stays NaN. Consumption series (``consumption`` true, published for
    pumped storage) are left out: :func:`hourly_pumping` reads them.
    """
    if "consumption" in frame:
        frame = frame[~frame["consumption"].astype(bool)]
    table = frame.assign(hour=pd.to_datetime(frame["time"], utc=True).dt.floor("h"))
    return table.pivot_table(index="hour", columns="psr_type", values="mw", aggfunc="mean")


def hourly_pumping(frame: pd.DataFrame) -> pd.Series:
    """Hourly mean MW of the pumped storage consumption series; hours without one are 0."""
    pumps = frame[frame["consumption"].astype(bool) & frame["psr_type"].eq(PUMPED_STORAGE)]
    hour = pd.to_datetime(pumps["time"], utc=True).dt.floor("h")
    return pumps.groupby(hour)["mw"].mean().rename_axis("hour")


def select_hours(generation: pd.DataFrame, load: pd.Series) -> dict[str, Any]:
    """Hours of highest wind, solar and load, lowest load, and median load.

    ``generation`` holds national hourly MW per type, ``load`` national hourly MW.
    """
    common = generation.index.intersection(load.index)
    frame = generation.loc[common].fillna(0.0)
    demand = load.loc[common]
    wind = frame.reindex(columns=_RENEWABLE_WIND, fill_value=0.0).sum(axis=1)
    return {
        "high_wind": wind.idxmax(),
        "high_solar": frame["B16"].idxmax(),
        "low_load": demand.idxmin(),
        "high_load": demand.idxmax(),
        "median_load": (demand - demand.median()).abs().idxmin(),
    }


def _capped_shares(total: float, capacity: pd.Series) -> pd.Series:
    """``total`` shared in proportion to capacity, never above it."""
    if capacity.sum() <= 0 or total <= 0:
        return capacity * 0.0
    return capacity * min(total / float(capacity.sum()), 1.0)


def dispatch_hour(
    tables: CimTables, nodes: pd.DataFrame, areas: pd.Series, hour: HourData
) -> Dispatch:
    """Model with the published values of ``hour``; ``areas`` maps node to control area.

    Machines without output are switched off. What a control area cannot place because the type
    has too little capacity there goes to the same type elsewhere, and is reported if it still
    does not fit. Load keeps each node's share of its control area and its power factor.
    """
    gen = tables.synchronous_machine.copy()
    capacity = gen["max_p_mw"].astype(float)
    zone = gen["node"].map(areas)
    p = pd.Series(0.0, index=gen.index)
    placed: dict[tuple[str, ...], float] = {}
    unallocated: dict[tuple[str, ...], float] = {}
    groups: dict[tuple[str, ...], pd.Index] = {}
    for fuel, types in FUEL_TYPES.items():
        members = gen.index[gen["fuel"].eq(fuel)]
        groups[types] = groups[types].union(members) if types in groups else members
    published = hour.generation.fillna(0.0)
    for types, index in groups.items():
        rest = 0.0
        for area in published.index.get_level_values(0).unique():
            total = float(sum(published.get((area, t), 0.0) for t in types))
            members = index[zone.loc[index].eq(area)]
            share = _capped_shares(total, capacity.loc[members])
            p.loc[members] = share
            rest += total - float(share.sum())
        if rest > 0:
            room = capacity.loc[index] - p.loc[index]
            extra = _capped_shares(rest, room)
            p.loc[index] += extra
            rest -= float(extra.sum())
        placed[types] = float(p.loc[index].sum())
        unallocated[types] = rest
    gen["p_mw"] = p
    gen["in_service"] = p > 0

    load = tables.energy_consumer.copy()
    load_area = load["node"].map(areas)
    base = load["p_mw"].astype(float)
    scale = load_area.map(hour.load) / base.groupby(load_area).transform("sum")
    load["p_mw"] = base * scale
    load["q_mvar"] = load["q_mvar"].astype(float) * scale
    load = pd.concat([load, _pump_loads(gen, zone, capacity, hour.pumping)], ignore_index=True)

    ext = tables.external_network_injection.copy()
    ext["p_mw"] = 0.0
    country = ext["node"].map(nodes.set_index("node")["country"])
    weights = _border_weights(tables, ext["node"])
    unplaced = {}
    border = country.map(COUNTRY_ZONES)
    converters = []
    for name, value in hour.exports.items():
        at = load[load["node"].eq(HVDC_CONVERTERS.get(str(name), ""))].head(1)
        if len(at):
            row = at[[c for c in ("node", "bus_mrid") if c in at]].assign(
                mrid=f"hvdc/{name}", p_mw=float(value), q_mvar=0.0
            )
            converters.append(row)
            continue
        chosen = border[border == name].index
        if len(chosen) == 0:
            unplaced[str(name)] = float(value)
            continue
        ext.loc[chosen, "p_mw"] = -float(value) * weights.loc[chosen] / weights.loc[chosen].sum()
    load = pd.concat([load, *converters], ignore_index=True)
    return Dispatch(
        replace(
            tables, synchronous_machine=gen, energy_consumer=load, external_network_injection=ext
        ),
        pd.Series(placed),
        pd.Series(unallocated),
        pd.Series(unplaced, dtype=float),
    )


def balance_gap_mw(tables: CimTables) -> float:
    """Load plus net exports minus generation of a dispatched model (positive: generation short)."""
    exports = -float(tables.external_network_injection["p_mw"].sum())
    return float(
        tables.energy_consumer["p_mw"].sum() + exports - tables.synchronous_machine["p_mw"].sum()
    )


def domestic_load_mw(tables: CimTables, losses_mw: float = 0.0) -> float:
    """Load of a dispatched model comparable with published Netzlast (ENTSO-E total load).

    Pumping (``pump/``) and HVDC converter flows (``hvdc/``) are consumers of the model but not
    part of published load; ``losses_mw`` taken off by :func:`without_losses` is added back.
    """
    load = tables.energy_consumer
    own = ~load["mrid"].astype(str).str.startswith(_ADDED_CONSUMERS)
    return float(load.loc[own, "p_mw"].sum()) + losses_mw


def with_unreported_feed_in(
    tables: CimTables, areas: pd.Series, gap_mw: float
) -> tuple[CimTables, float]:
    """Add ``gap_mw`` as separate machine rows; returns the model and the MW placed.

    Published generation misses some feed-in, so reported values stay as they are. The new rows
    carry the ``feed_in/`` id and :data:`FEED_IN_LABEL`. They use the unused capacity of
    :data:`INCOMPLETE_COVERAGE` first, then of other dispersed gas and biomass plants. This
    placement is an assumption: no source publishes the gap per area or fuel. A surplus
    (``gap_mw`` <= 0) is not corrected.
    """
    gen = tables.synchronous_machine
    if gap_mw <= 0:
        return tables, 0.0
    room = (gen["max_p_mw"].astype(float) - gen["p_mw"].astype(float)).clip(lower=0.0)
    zone = gen["node"].map(areas)
    covered = pd.Series(
        [area in INCOMPLETE_COVERAGE.get(f, ()) for f, area in zip(gen["fuel"], zone, strict=True)],
        index=gen.index,
    )
    first = _capped_shares(gap_mw, room.where(covered, 0.0))
    second = _capped_shares(
        gap_mw - float(first.sum()),
        room.where(gen["fuel"].isin(DISPERSED_FUELS) & ~covered, 0.0),
    )
    p = first + second
    rows = gen[p > 0].copy()
    rows["p_mw"] = p[p > 0]
    rows["max_p_mw"] = p[p > 0]
    rows["in_service"] = True
    rows["data_quality"] = FEED_IN_LABEL
    if "voltage_control" in rows:
        rows["voltage_control"] = False
    if "mrid" in rows:
        rows["mrid"] = "feed_in/" + rows["mrid"].astype(str)
    placed = float(p.sum())
    return replace(tables, synchronous_machine=pd.concat([gen, rows], ignore_index=True)), placed


def without_losses(tables: CimTables, losses_mw: float, published: pd.DataFrame) -> CimTables:
    """Take ``losses_mw`` off the ``published`` load: Netzlast already includes grid losses.

    Consumers keep their power factor. Pumping and HVDC flows are not part of published load
    and stay as they are.
    """
    table = published.copy()
    own = ~table["mrid"].astype(str).str.startswith(_ADDED_CONSUMERS)
    scale = 1.0 - losses_mw / float(table.loc[own, "p_mw"].sum())
    table.loc[own, "p_mw"] = table.loc[own, "p_mw"] * scale
    table.loc[own, "q_mvar"] = table.loc[own, "q_mvar"] * scale
    return replace(tables, energy_consumer=table)


def corrected_balance(
    tables: CimTables,
    areas: pd.Series,
    solved_losses_mw: Callable[[CimTables], float],
    iterations: int = 3,
) -> tuple[CimTables, float, float]:
    """Dispatched model with the published balance gap and double-counted losses corrected.

    ``solved_losses_mw`` solves a model and returns its network losses. Returns the model, the
    feed-in placed and the losses removed.
    """
    corrected, placed = with_unreported_feed_in(tables, areas, balance_gap_mw(tables))
    published = corrected.energy_consumer
    losses = 0.0
    for _ in range(iterations):
        corrected = without_losses(corrected, losses, published)
        losses = solved_losses_mw(corrected)
    return without_losses(corrected, losses, published), placed, losses


def _pump_loads(
    gen: pd.DataFrame, zone: pd.Series, capacity: pd.Series, pumping: pd.Series
) -> pd.DataFrame:
    """Consumers at the pumped storage plants of each area, shared by capacity.

    Published load excludes storage pumping, so these come on top of it. Pumping of an area
    without such a plant goes to the plants of the other areas.
    """
    plants = gen.index[gen["fuel"].eq("Pumpspeicher")]
    power = pd.Series(0.0, index=plants)
    rest = 0.0
    for area, total in pumping.fillna(0.0).items():
        members = plants[zone.loc[plants].eq(str(area))]
        if capacity.loc[members].sum() > 0:
            power.loc[members] = total * capacity.loc[members] / capacity.loc[members].sum()
        else:
            rest += float(total)
    if rest and capacity.loc[plants].sum() > 0:
        power += rest * capacity.loc[plants] / capacity.loc[plants].sum()
    active = power.index[power > 0]
    rows = gen.loc[active, [c for c in ("mrid", "node", "bus_mrid") if c in gen]].copy()
    if "mrid" in rows:
        rows["mrid"] = "pump/" + rows["mrid"].astype(str)
    return rows.assign(p_mw=power.loc[active], q_mvar=0.0)


def _border_weights(tables: CimTables, border_nodes: pd.Series) -> pd.Series:
    """Rated capacity of the lines at each foreign node: a border's flow is split by it."""
    line = tables.ac_line_segment
    bus = tables.external_network_injection.set_index("node")["bus_mrid"]
    mva = line["rated_current_a"].astype(float) * line["nominal_voltage_kv"].astype(float)
    weights = []
    for node in border_nodes:
        at = line["bus1_mrid"].eq(bus[node]) | line["bus2_mrid"].eq(bus[node])
        weights.append(float(np.nansum(mva[at])))
    return pd.Series(weights, index=border_nodes.index).clip(lower=1.0)


WINTER_MONTHS = (11, 12, 1, 2, 3)
"""Months rated at the JAO ``dlr_max``: cold air keeps conductors cool."""
SUMMER_MONTHS = (6, 7, 8)
"""Months whose daytime hours are rated at the JAO ``dlr_min``."""
DAY_HOURS = (8, 20)
"""Local hours [start, end) counted as daytime; other hours are night, rated at ``dlr_max``."""
_SHOULDER = {4: 1 / 3, 5: 2 / 3, 9: 2 / 3, 10: 1 / 3}


def rating_weight(when: pd.Timestamp) -> float:
    """Share of the way from ``dlr_max`` (0) to ``dlr_min`` (1) at a time.

    Winter months and night hours take 0, summer daytime 1, spring and autumn daytime a linear
    step in between. This is a rule of thumb for line monitoring, not a weather model.
    """
    local = when.tz_convert("Europe/Berlin") if when.tzinfo is not None else when
    if local.month in WINTER_MONTHS or not DAY_HOURS[0] <= local.hour < DAY_HOURS[1]:
        return 0.0
    return 1.0 if local.month in SUMMER_MONTHS else _SHOULDER[local.month]


def hourly_line_ratings(
    lines: pd.DataFrame, circuits: pd.DataFrame, when: pd.Timestamp, basis: str = "seasonal"
) -> pd.DataFrame:
    """Lines with ``rated_current_a`` set from the JAO dynamic band for one hour.

    Only lines whose published rating is the band (``rating_basis`` ``dlr_min``, found by
    ``eic`` in ``circuits``) change; fixed, seasonal and upper-band ratings stay. ``basis``
    ``"seasonal"`` interpolates with :func:`rating_weight`, ``"dlr_min"`` keeps the lower band
    as a sensitivity. A line without its own EIC (an added parallel circuit) can name the
    circuit it copies in an optional ``rating_eic`` column.
    """
    if basis not in ("seasonal", "dlr_min"):
        raise ValueError(f"unknown rating basis {basis!r}")
    band = circuits[circuits["rating_basis"].eq("dlr_min")].drop_duplicates("eic")
    band = band.set_index("eic")[["dlr_min_a", "dlr_max_a"]].dropna()
    out = lines.copy()
    key = out["eic"]
    if "rating_eic" in out:
        key = out["rating_eic"].where(out["rating_eic"].notna(), key)
    low = key.map(band["dlr_min_a"])
    high = key.map(band["dlr_max_a"])
    weight = 1.0 if basis == "dlr_min" else rating_weight(when)
    value = high - weight * (high - low)
    out["rated_current_a"] = value.where(value.notna(), out["rated_current_a"])
    return out
