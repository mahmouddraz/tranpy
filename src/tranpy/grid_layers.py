"""Silver and gold tables of the German transmission grid model.

* **Silver** (cleaned, conformed, with evidence): the network from
  :func:`tranpy.german_grid.build_german_grid`, the power plant list attached to the
  network (:func:`attach_plants`) and the population share of every transmission
  substation (:func:`load_shares`).
* **Gold** (simulation ready): :func:`cim_tables` - the network, generators and loads as
  tables named after CIM (IEC 61970 / CGMES) classes, SI units, one stable ``mrid`` per
  object. :mod:`tranpy.powsybl_model` turns them into a pypowsybl network (AC power flow,
  CGMES export).

Every estimated value is marked: column ``data_quality`` is ``"published"`` (taken from an
official source), ``"verified"`` (also checked, e.g. on orthophotos) or ``"estimated"``
(derived by a documented rule).
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from tranpy.german_grid import GermanGrid
from tranpy.osm_export import IN_OPERATION, OUT_OF_SERVICE, PLANNED, UNDER_CONSTRUCTION
from tranpy_core.errors import DataError

__all__ = [
    "LOAD_REFERENCE",
    "CimTables",
    "attach_plants",
    "cim_tables",
    "load_shares",
    "mrid",
    "node_states",
    "plants_from_kraftwerksliste",
]

TOTAL_LOAD_MW = 466_100_000 / 8760
"""Default national load of the base case (MW): see :data:`LOAD_REFERENCE`."""

LOAD_REFERENCE: dict[str, Any] = {
    "total_load_mw": TOTAL_LOAD_MW,
    "source": "Final electricity consumption (Letztverbrauch) Germany 2024: 466.1 TWh (BDEW); "
    "average over 8760 h. Gross consumption 2025: 517.2 TWh (BDEW, preliminary).",
}
"""Default national load (MW) of the base case and its source."""

_NAMESPACE = uuid.UUID("6f1d3c3e-6a0e-4c1b-9e57-1d0c7f3b2a90")
_EARTH_RADIUS_M = 6378137.0
_POWER_FACTOR = 0.95

_RESERVE = ("netzreserve", "kapazitätsreserve", "besonderes netztechnisches", "vorläufig")
_RENEWABLE = ("solar", "wind", "biomasse", "wasser", "geothermie", "deponie", "klär", "gruben")
_NON_DISPATCHABLE = ("solar", "wind")

OFFSHORE_LANDINGS: dict[str, tuple[str, ...]] = {
    "Nordsee": ("diele|380", "dorpen/west|380", "emden/ost|380", "buttel|380", "cloppenburg|380"),
    "Ostsee": ("lubmin|380", "bentwisch|380"),
}
"""Onshore substations where the offshore connections of the North Sea and Baltic Sea land
(converter and AC landing points; used to place offshore wind without coordinates)."""


def mrid(*parts: Any) -> str:
    """Stable CIM identifier (UUID 5) of an object from its natural key."""
    return str(uuid.uuid5(_NAMESPACE, "|".join(str(p) for p in parts)))


def _web_mercator_to_lonlat(
    x: npt.ArrayLike, y: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    xa, ya = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    lon = np.degrees(xa / _EARTH_RADIUS_M)
    lat = np.degrees(np.arctan(np.sinh(ya / _EARTH_RADIUS_M)))
    return lon, lat


def _km(lon1: Any, lat1: Any, lon2: Any, lat2: Any) -> Any:
    return np.hypot((lon1 - lon2) * np.cos(np.radians((lat1 + lat2) / 2)), lat1 - lat2) * 111.2


def _plant_lifecycle(status: pd.Series) -> pd.Series:
    """Lifecycle of plant list entries; grid and capacity reserve plants are in operation."""
    return pd.Series(
        np.select(
            [
                status.str.contains("stillgelegt"),
                status.str.contains("planung"),
                status.str.contains("im bau|in bau"),
            ],
            [OUT_OF_SERVICE, PLANNED, UNDER_CONSTRUCTION],
            IN_OPERATION,
        ),
        index=status.index,
    )


def plants_from_kraftwerksliste(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalised power plant list (Bundesnetzagentur Kraftwerksliste).

    Columns: ``mastr_id``, ``name``, ``operator``, ``state``, ``fuel``, ``capacity_mw`` (net
    nominal), ``status``, ``in_operation`` (in operation), ``reserve`` (grid or capacity
    reserve, not in the market), ``decommissioned``, ``lifecycle_status``, ``voltage_level``
    (first of the listed levels), ``grid_operator``, ``lon``, ``lat`` (WGS84, from EPSG:3857),
    ``aggregated`` (one row per state instead of single units).
    """
    status = raw["Kraftwerksstatus_der_Einheit"].fillna("").astype(str)
    low = status.str.lower()
    lon, lat = _web_mercator_to_lonlat(
        pd.to_numeric(raw["x"], errors="coerce"), pd.to_numeric(raw["y"], errors="coerce")
    )
    level = raw["Spannungsebene"].fillna("").astype(str).str.split(";").str[0].str.strip()
    mastr = raw["MaStR_Nr_der_Stromerzeugungseinheit"].fillna("").astype(str)
    return pd.DataFrame(
        {
            "mastr_id": mastr,
            "name": raw["Anzeige_Name_der_Stromerzeugungseinheit"],
            "operator": raw["Anlagenbetreiber"],
            "state": raw["Bundesland_der_Einheit"],
            "fuel": raw["Energietr_ger"],
            "capacity_mw": pd.to_numeric(
                raw["Nettonennleistung_elektrische_Wirkleistung_in_MW"], errors="coerce"
            ),
            "status": status,
            "in_operation": low.str.startswith("in betrieb"),
            "reserve": low.str.startswith(_RESERVE),
            "decommissioned": low.str.contains("endgültig stillgelegt"),
            "lifecycle_status": _plant_lifecycle(low),
            "voltage_level": level.replace("", None),
            "grid_operator": raw["Anschlussnetzbetreiber"],
            "lon": lon,
            "lat": lat,
            "aggregated": mastr.str.contains("dummy", case=False),
        }
    )


def node_states(nodes: pd.DataFrame, plants: pd.DataFrame, k: int = 7) -> pd.Series:
    """Federal state of each German node: majority of the ``k`` nearest located plants."""
    located = plants.dropna(subset=["lon", "lat", "state"])
    located = located[~located["aggregated"]]
    if located.empty:
        raise DataError("no located plants with a state")
    px, py = located["lon"].to_numpy(), located["lat"].to_numpy()
    states = located["state"].to_numpy()
    out = {}
    for row in nodes.itertuples():
        if not np.isfinite(row.lon):
            continue
        d = _km(px, py, row.lon, row.lat)
        nearest = states[np.argsort(d)[:k]]
        values, counts = np.unique(nearest, return_counts=True)
        out[row.node] = values[np.argmax(counts)]
    return pd.Series(out, name="state")


def _transmission_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    """German substations (and busbars) with a location: candidates for plants and loads."""
    return nodes[
        nodes["kind"].isin(("substation", "busbar")) & np.isfinite(nodes["lon"].astype(float))
    ]


def load_shares(population: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Population assigned to the nearest German transmission substation (estimate).

    ``population`` has columns ``lon``, ``lat``, ``population`` (grid cells). Returns
    ``node``, ``population``, ``share``, ``cells``.
    """
    candidates = _transmission_nodes(nodes)
    candidates = candidates.sort_values("kv", ascending=False).drop_duplicates("site")
    nx, ny = candidates["lon"].to_numpy(float), candidates["lat"].to_numpy(float)
    keys = candidates["node"].to_numpy()
    pop = population.dropna(subset=["lon", "lat", "population"])
    assigned = np.empty(len(pop), dtype=object)
    for i, (x, y) in enumerate(
        zip(pop["lon"].astype(float), pop["lat"].astype(float), strict=True)
    ):
        assigned[i] = keys[int(np.argmin(_km(nx, ny, x, y)))]
    frame = pd.DataFrame({"node": assigned, "population": pop["population"].astype(float)})
    out = frame.groupby("node").agg(population=("population", "sum"), cells=("population", "size"))
    out["share"] = out["population"] / out["population"].sum()
    return out.reset_index()


@dataclass(frozen=True)
class _Attachment:
    node: str
    share: float
    method: str
    distance_km: float = float("nan")


_FOREIGN_STATES = {"Österreich": "AUT", "Luxemburg": "LUX", "Schweiz": "CHE"}
"""States of the plant list that are neighbouring countries -> ISO 3166 alpha-3."""


def node_countries(nodes: pd.DataFrame, borders: dict[str, Any]) -> pd.Series:
    """ISO alpha-3 country of every located node (point in country polygon).

    ``borders`` is a GeoJSON feature collection with property ``ADM0_A3`` (Natural Earth).
    """
    from matplotlib.path import Path as Polygon

    shapes: list[tuple[str, Any]] = []
    for feature in borders["features"]:
        geometry = feature["geometry"]
        polygons = (
            geometry["coordinates"]
            if geometry["type"] == "MultiPolygon"
            else [geometry["coordinates"]]
        )
        for polygon in polygons:
            shapes.append((feature["properties"]["ADM0_A3"], Polygon(np.asarray(polygon[0]))))
    out = {}
    for row in nodes.itertuples():
        if np.isfinite(row.lon):
            out[row.node] = next(
                (code for code, shape in shapes if shape.contains_point((row.lon, row.lat))), ""
            )
    return pd.Series(out, name="country")


def attach_plants(
    plants: pd.DataFrame,
    nodes: pd.DataFrame,
    shares: pd.DataFrame,
    countries: pd.Series | None = None,
) -> pd.DataFrame:
    """Connect every plant to transmission substations; one row per (plant, node).

    Rules (column ``method``):

    * located plant at "Höchstspannung": nearest transmission substation
      (``nearest substation``);
    * located plant at a lower voltage level: aggregated to the nearest transmission
      substation (``aggregated to nearest substation``);
    * offshore wind without coordinates: spread equally over the landing substations of its
      sea (``offshore landing``, estimate);
    * any other plant without coordinates (incl. the per-state rows of wind and solar):
      spread over the substations of its state by population share (``state population
      share``, estimate); without a state ("ohne Zuordnung"): over all substations by
      population share (``national population share``, estimate);
    * plants in a neighbouring country feeding the German grid (Austria, Luxembourg,
      Switzerland): spread equally over the model's border nodes in that country
      (``border node``, estimate; needs ``countries`` from :func:`node_countries`).

    Decommissioned plants are left out.
    """
    candidates = _transmission_nodes(nodes)
    candidates = candidates.sort_values("kv", ascending=False).drop_duplicates("site")
    nx, ny = candidates["lon"].to_numpy(float), candidates["lat"].to_numpy(float)
    keys = candidates["node"].to_numpy()
    states = node_states(candidates, plants)
    by_state = (
        shares.assign(state=shares["node"].map(states)).dropna(subset=["state"]).groupby("state")
    )
    rows: list[dict[str, Any]] = []
    for plant in plants[~plants["decommissioned"]].itertuples():
        attachments: list[_Attachment] = []
        fuel = str(plant.fuel).lower()
        if np.isfinite(plant.lon) and np.isfinite(plant.lat):
            d = _km(nx, ny, plant.lon, plant.lat)
            i = int(np.argmin(d))
            method = (
                "nearest substation"
                if str(plant.voltage_level) == "Höchstspannung"
                else "aggregated to nearest substation"
            )
            attachments = [_Attachment(str(keys[i]), 1.0, method, float(d[i]))]
        elif "offshore" in fuel:
            sea = "Ostsee" if "ostsee" in str(plant.state).lower() else "Nordsee"
            landing = [n for n in OFFSHORE_LANDINGS[sea] if n in set(keys)]
            attachments = [_Attachment(n, 1 / len(landing), "offshore landing") for n in landing]
        elif plant.state in _FOREIGN_STATES and countries is not None:
            border = [
                n
                for n, c in countries.items()
                if c == _FOREIGN_STATES[str(plant.state)] and n in set(nodes["node"])
            ]
            attachments = [_Attachment(str(n), 1 / len(border), "border node") for n in border]
        elif plant.state in by_state.groups:
            group = by_state.get_group(plant.state)
            attachments = [
                _Attachment(str(n), float(s), "state population share")
                for n, s in zip(group["node"], group["share"] / group["share"].sum(), strict=True)
            ]
        elif not isinstance(plant.state, str) or plant.state == "ohne Zuordnung":
            attachments = [
                _Attachment(str(n), float(s), "national population share")
                for n, s in zip(shares["node"], shares["share"], strict=True)
            ]
        for a in attachments:
            rows.append(
                {
                    "mastr_id": plant.mastr_id,
                    "name": plant.name,
                    "fuel": plant.fuel,
                    "state": plant.state,
                    "status": plant.status,
                    "in_operation": plant.in_operation,
                    "reserve": plant.reserve,
                    "node": a.node,
                    "share": a.share,
                    "capacity_mw": float(np.nan_to_num(plant.capacity_mw)) * a.share,
                    "method": a.method,
                    "distance_km": a.distance_km,
                    "data_quality": "published"
                    if a.method == "nearest substation"
                    else "estimated",
                    "renewable": any(f in fuel for f in _RENEWABLE),
                }
            )
    return pd.DataFrame(rows)


@dataclass
class CimTables:
    """Gold tables, named after CIM classes (see :func:`cim_tables`)."""

    substation: pd.DataFrame
    voltage_level: pd.DataFrame
    busbar_section: pd.DataFrame
    ac_line_segment: pd.DataFrame
    power_transformer: pd.DataFrame
    synchronous_machine: pd.DataFrame
    energy_consumer: pd.DataFrame
    external_network_injection: pd.DataFrame
    model_info: dict[str, Any] = field(default_factory=dict)

    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            "cim_substation": self.substation,
            "cim_voltage_level": self.voltage_level,
            "cim_busbar_section": self.busbar_section,
            "cim_ac_line_segment": self.ac_line_segment,
            "cim_power_transformer": self.power_transformer,
            "cim_synchronous_machine": self.synchronous_machine,
            "cim_energy_consumer": self.energy_consumer,
            "cim_external_network_injection": self.external_network_injection,
        }


def _route(points: Any) -> list[dict[str, float]]:
    return [{"lon": float(x), "lat": float(y)} for x, y in np.asarray(points).reshape(-1, 2)]


def cim_tables(
    grid: GermanGrid,
    plants: pd.DataFrame,
    shares: pd.DataFrame,
    *,
    total_load_mw: float = TOTAL_LOAD_MW,
    load_source: str = LOAD_REFERENCE["source"],
    model_version: str = "",
    countries: pd.Series | None = None,
) -> CimTables:
    """Gold model of the German transmission grid (base case).

    * ``substation`` / ``voltage_level`` / ``busbar_section``: one busbar per network node;
    * ``ac_line_segment``: AC circuits (HVDC links are left out and listed in
      ``model_info``); impedances in ohm, susceptance ``bch_s`` in siemens per circuit;
    * ``power_transformer``: two-winding transformers and phase shifters, parameters
      referred to the high-voltage end (as published), ratings in A at that end;
    * ``synchronous_machine``: generation aggregated per (node, fuel, in operation), with
      ``max_p_mw`` and a base-case setpoint ``p_mw`` (all plants in operation scaled to
      cover ``total_load_mw``; reserve plants out of service);
    * ``energy_consumer``: ``total_load_mw`` distributed by population share, power factor
      0.95 (estimate);
    * ``external_network_injection``: neighbouring grids at tie-line ends outside Germany
      (voltage sources, zero active power setpoint); ``countries`` (from
      :func:`node_countries`) keeps tie-line ends located in Germany out.
    """
    nodes = grid.nodes.copy()
    bus_mrid = {n: mrid("bus", n) for n in nodes["node"]}
    in_country = nodes["node"].map(countries) if countries is not None else None
    site_country = {
        r.site: (
            "DE"
            if r.kind != "foreign" or (in_country is not None and in_country.iloc[i] == "DEU")
            else ""
        )
        for i, r in enumerate(nodes.itertuples())
    }
    substation = (
        nodes.groupby("site")
        .agg(
            lon=("lon", "mean"),
            lat=("lat", "mean"),
            tso=("tso", "first"),
            names=("names", lambda v: sorted({n for names in v for n in names})),
        )
        .reset_index()
    )
    substation.insert(0, "mrid", [mrid("substation", s) for s in substation["site"]])
    substation["country"] = substation["site"].map(site_country)
    voltage_level = nodes[["site", "kv"]].drop_duplicates().reset_index(drop=True)
    voltage_level.insert(
        0,
        "mrid",
        [
            mrid("vl", s, kv)
            for s, kv in zip(voltage_level["site"], voltage_level["kv"], strict=True)
        ],
    )
    voltage_level["substation_mrid"] = voltage_level["site"].map(
        dict(zip(substation["site"], substation["mrid"], strict=True))
    )
    voltage_level["nominal_voltage_kv"] = voltage_level["kv"].astype(float)
    busbar = pd.DataFrame(
        {
            "mrid": [bus_mrid[n] for n in nodes["node"]],
            "name": nodes["node"],
            "voltage_level_mrid": [
                mrid("vl", s, kv) for s, kv in zip(nodes["site"], nodes["kv"], strict=True)
            ],
            "kind": nodes["kind"],
            "tso": nodes["tso"],
            "lon": nodes["lon"],
            "lat": nodes["lat"],
            "location_source": nodes["location_source"],
            "data_quality": np.where(
                nodes["location_source"].eq("verified"), "verified", "published"
            ),
        }
    )

    lines = grid.lines[~grid.lines["hvdc"]]
    ac_line = pd.DataFrame(
        {
            "mrid": [
                mrid("line", e, a, b)
                for e, a, b in zip(lines["eic"], lines["node0"], lines["node1"], strict=True)
            ],
            "name": lines["name"],
            "eic": lines["eic"],
            "tso": lines["tso"],
            "bus1_mrid": lines["node0"].map(bus_mrid),
            "bus2_mrid": lines["node1"].map(bus_mrid),
            "nominal_voltage_kv": lines["kv"].astype(float),
            "length_km": lines["length_km"],
            "r_ohm": lines["r_ohm"].astype(float),
            "x_ohm": lines["x_ohm"].astype(float),
            "bch_s": lines["b_us"].astype(float).fillna(0.0) * 1e-6,
            "gch_s": 0.0,
            "rated_current_a": lines["imax_a"],
            "tie_line": lines["tie_line"],
            "route": [_route(r) for r in lines["route"]],
            "route_source": lines["route_source"],
            "route_evidence": lines.get("route_evidence"),
            "status": lines.get("status", "in_operation"),
            "status_date": lines.get("status_date"),
            "status_evidence": lines.get("status_evidence"),
            "review": lines.get("review"),
            "data_quality": lines.get("data_quality", "published"),
        }
    )
    tr = grid.transformers
    transformer = pd.DataFrame(
        {
            "mrid": [mrid("trafo", e, n) for e, n in zip(tr["eic"], tr["name"], strict=True)],
            "name": tr["name"],
            "eic": tr["eic"],
            "tso": tr["tso"],
            "bus1_mrid": tr["node_hv"].map(bus_mrid),
            "bus2_mrid": tr["node_lv"].map(bus_mrid),
            "rated_u1_kv": tr["kv_hv"].astype(float),
            "rated_u2_kv": tr["kv_lv"].astype(float),
            "r_ohm": tr["r_ohm"].astype(float),
            "x_ohm": tr["x_ohm"].astype(float),
            "b_s": tr["b_us"].astype(float).fillna(0.0) * 1e-6,
            "g_s": tr["g_us"].astype(float).fillna(0.0) * 1e-6,
            "rated_current_a": tr["imax_a"],
            "phase_shifter": tr["phase_shifter"],
            "data_quality": "published",
        }
    )

    gen = plants[plants["node"].isin(bus_mrid)].copy()
    storage = gen["fuel"].str.contains("speicher", case=False, na=False)
    gen["active"] = gen["in_operation"] & ~gen["reserve"] & ~storage
    if "unit_count" not in gen:
        gen["unit_count"] = np.nan
    grouped = (
        gen.groupby(["node", "fuel", "active"])
        .agg(
            max_p_mw=("capacity_mw", "sum"),
            units=("unit_count", lambda v: v.fillna(1.0).sum()),
            estimated=("data_quality", lambda v: bool((v == "estimated").any())),
            renewable=("renewable", "first"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["max_p_mw"] > 0]
    active = grouped["active"]
    scale = total_load_mw / float(grouped.loc[active, "max_p_mw"].sum())
    fuel_lower = grouped["fuel"].str.lower()
    machine = pd.DataFrame(
        {
            "mrid": [
                mrid("gen", n, f, a)
                for n, f, a in zip(grouped["node"], grouped["fuel"], grouped["active"], strict=True)
            ],
            "bus_mrid": grouped["node"].map(bus_mrid),
            "node": grouped["node"],
            "fuel": grouped["fuel"],
            "units": grouped["units"],
            "max_p_mw": grouped["max_p_mw"],
            "min_p_mw": 0.0,
            "p_mw": np.where(active, grouped["max_p_mw"] * scale, 0.0),
            "in_service": active,
            "voltage_control": active & ~fuel_lower.str.contains("|".join(_NON_DISPATCHABLE)),
            "target_v_pu": 1.02,
            "renewable": grouped["renewable"],
            "data_quality": np.where(grouped["estimated"], "estimated", "published"),
        }
    )

    tan_phi = math.tan(math.acos(_POWER_FACTOR))
    load = shares[shares["node"].isin(bus_mrid)].copy()
    consumer = pd.DataFrame(
        {
            "mrid": [mrid("load", n) for n in load["node"]],
            "bus_mrid": load["node"].map(bus_mrid),
            "node": load["node"],
            "p_mw": load["share"] * total_load_mw,
            "q_mvar": load["share"] * total_load_mw * tan_phi,
            "population": load["population"],
            "data_quality": "estimated",
        }
    )

    # neighbouring grids: nodes outside Germany (by country polygon where known)
    located_in = (
        nodes["node"].map(countries) if countries is not None else pd.Series("", index=nodes.index)
    )
    abroad = (nodes["kind"] == "foreign") & located_in.fillna("").ne("DEU")
    foreign = nodes[abroad]
    injection = pd.DataFrame(
        {
            "mrid": [mrid("extgrid", n) for n in foreign["node"]],
            "bus_mrid": foreign["node"].map(bus_mrid),
            "node": foreign["node"],
            "p_mw": 0.0,
            "target_v_pu": 1.0,
            "data_quality": "estimated",
        }
    )
    hvdc = grid.lines[grid.lines["hvdc"]]
    info = {
        "model_version": model_version,
        "total_load_mw": total_load_mw,
        "load_source": load_source,
        "power_factor": _POWER_FACTOR,
        "dispatch": f"all plants in operation (excl. reserve) at {scale:.3f} x net capacity",
        "hvdc_left_out": ", ".join(hvdc["name"].astype(str)),
        "sources": "; ".join(f"{k}: {v}" for k, v in grid.sources.items()),
    }
    return CimTables(
        substation=substation,
        voltage_level=voltage_level,
        busbar_section=busbar,
        ac_line_segment=ac_line,
        power_transformer=transformer,
        synchronous_machine=machine,
        energy_consumer=consumer,
        external_network_injection=injection,
        model_info=info,
    )
