"""GIS export of the German grid model (GeoPackage and GeoJSON, ``[grid]`` extra).

:func:`build_layers` turns a frozen model ``(silver, cim)`` into GIS layers in EPSG:4326;
:func:`export_grid_gis` writes them to ``grid_model.gpkg`` (one layer each) and, for the
web viewer, to compact GeoJSON files. The model is only read.

Layers (see ``docs/grid_map.md`` for the attributes):

* ``lines`` - AC circuits (routes from the model; a straight bus-to-bus segment where the
  model has no route, ``route_quality = "straight"``) and the HVDC links the electrical
  model leaves out (``hvdc = True``);
* ``substations``, ``buses``, ``transformers`` (at the high-voltage bus), ``plants``
  (Marktstammdatenregister units with their connection bus), ``neighbours`` (equivalent
  networks at tie-line ends abroad);
* ``control_areas`` - dissolved Voronoi cells of each TSO's buses clipped to the German
  border (approximate), ``bidding_zones`` - DE-LU and neighbouring zones from Natural Earth
  country shapes (approximate where a country holds several zones);
* ``metadata`` - model version, source hash, export time (table without geometry).

A geometry overlay (:func:`read_geometry_overlay`, built by
``scripts/build_geometry_overlay.py``) corrects display positions of sites and routes of
lines where the model's are schematic; every feature records ``geometry_source``.
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import math
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tranpy.grid_register import _CONTROL_AREA, node_control_areas
from tranpy.grid_stations import (
    METRIC_CRS,
    empty_station_layers,
    layout_stations,
    read_osm_footprints,
    station_footprints,
    write_station_details,
    write_station_diagrams,
)
from tranpy.grid_towers import empty_tower_lines, read_tower_ways, split_by_towers
from tranpy_core._optional import require

__all__ = [
    "BIDDING_ZONES",
    "FUEL_CATEGORY",
    "LAYERS",
    "TEE_AT_STATION_KM",
    "TSO_COLOURS",
    "build_layers",
    "export_grid_gis",
    "load_model",
    "location_flag",
    "place_tee",
    "polyline_km",
    "read_geometry_overlay",
    "section_of",
    "section_route",
]

CRS = "EPSG:4326"
BORDER_TOLERANCE_KM = 3.0
"""A node of kind ``substation`` this close to the German border counts as German."""
_KM_PER_DEGREE_LAT = 111.2
SNAP_MAX_M = 3000.0
"""A route end this close to its substation is extended to the substation point (display)."""
FOLLOW_M = 200.0
"""An export-section end this close to a drawn model route continues along it (display)."""
END_SEARCH_M = 10000.0
"""Route ends within this distance count for placing a schematic substation (display)."""
CORRIDOR_TOLERANCE_M = 150.0
"""Circuits whose display routes are within this (Hausdorff) distance share a corridor."""
TEE_AT_STATION_KM = 0.5
"""A tee whose published leg to a station is shorter than this sits at that station."""
OVERLAY_COLUMNS = ("element", "id", "name", "field", "old", "new", "method", "evidence")
"""Columns of the geometry overlay: ``element`` ``site`` (``field`` ``point``, values
``[lon, lat]`` as JSON) or ``line`` (``id`` the segment mrid, ``field`` ``route``, values
``[[lon, lat], ...]`` as JSON)."""
PLANT_INDIVIDUAL_MW = 50.0
"""Plants of at least this capacity are drawn individually; smaller ones are aggregated."""
LAYERS = (
    "lines",
    "osm_lines",
    "tower_lines",
    "osm_substations",
    "station_footprints",
    "station_levels",
    "station_transformers",
    "substations",
    "buses",
    "transformers",
    "plant_connections",
    "plant_aggregates",
    "plants",
    "offshore_links",
    "renewable_inverters",
    "offshore_farms",
    "neighbours",
    "control_areas",
    "bidding_zones",
    "metadata",
)
"""Layer names of the GeoPackage, in drawing order of the QGIS project (bottom last)."""

TSO_COLOURS = {
    "50Hertz": "#2a78d6",
    "Amprion": "#eb6834",
    "TenneT": "#1baf7a",
    "TransnetBW": "#eda100",
}
"""Control area -> colour, as in :func:`tranpy.maps.plot_operator_map`."""

BIDDING_ZONES: dict[str, tuple[str, str, bool]] = {
    "DEU": ("DE-LU", "Germany/Luxembourg", False),
    "LUX": ("DE-LU", "Germany/Luxembourg", False),
    "AUT": ("AT", "Austria", False),
    "CHE": ("CH", "Switzerland", False),
    "FRA": ("FR", "France", False),
    "BEL": ("BE", "Belgium", False),
    "NLD": ("NL", "Netherlands", False),
    "DNK": ("DK1", "Denmark West", True),
    "POL": ("PL", "Poland", False),
    "CZE": ("CZ", "Czechia", False),
    "NOR": ("NO2", "Norway South", True),
}
"""ISO alpha-3 -> (bidding zone, name, approximate). Denmark and Norway hold several zones;
the model's tie-lines end in DK1 (Kassø, Endrup) and NO2 (Ertsmyra), so the whole country
shape stands for that zone (``approximate``)."""

FUEL_CATEGORY = {
    "Kernenergie": "nuclear",
    "Braunkohle": "lignite",
    "Steinkohle": "hard_coal",
    "Erdgas": "gas",
    "andere Gase": "gas",
    "Grubengas": "gas",
    "Mineralölprodukte": "oil",
    "Wasser": "hydro",
    "Pumpspeicher": "pumped_storage",
    "Speicher": "storage",
    "Biomasse": "biomass",
    "Klärschlamm": "biomass",
    "nicht biogener Abfall": "waste",
    "Wärme": "other",
    "Druck aus Gasleitungen": "other",
    "Druck aus Wasserleitungen": "other",
}
"""Marktstammdatenregister fuel -> map category (symbol and colour of the styles)."""


def load_model(path: str | Path) -> tuple[dict[str, pd.DataFrame], Any]:
    """The frozen model ``(silver tables, CIM tables)`` from its pickle."""
    silver, cim = pd.read_pickle(path)
    return silver, cim


def polyline_km(coords: Any) -> float:
    """Length in km of a lon/lat polyline (haversine per vertex pair)."""
    pts = np.radians(np.asarray(coords, dtype=float).reshape(-1, 2))
    if len(pts) < 2:
        return 0.0
    dlon, dlat = np.diff(pts[:, 0]), np.diff(pts[:, 1])
    lat0, lat1 = pts[:-1, 1], pts[1:, 1]
    h = np.sin(dlat / 2) ** 2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlon / 2) ** 2
    return float((2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))).sum())


def _point_at(coords: Any, km: float) -> tuple[float, float]:
    """Point at arc length ``km`` along a lon/lat polyline (clamped to its ends)."""
    pts = np.asarray(coords, dtype=float).reshape(-1, 2)
    done = 0.0
    for a, b in itertools.pairwise(pts):
        step = polyline_km([a, b])
        if step > 0 and done + step >= km:
            t = max(km - done, 0.0) / step
            return (float(a[0] + t * (b[0] - a[0])), float(a[1] + t * (b[1] - a[1])))
        done += step
    return (float(pts[-1][0]), float(pts[-1][1]))


def place_tee(
    legs: list[tuple[str, tuple[float, float], float, bool]],
    route: tuple[str, str, Any] | None = None,
) -> tuple[tuple[float, float], str, str] | None:
    """Position of a tee from its published leg lengths: ``(point, method, evidence)``.

    ``legs`` are ``(end, end point, leg km, end is a station)``. A leg to a station shorter
    than :data:`TEE_AT_STATION_KM` puts the tee at that station (the shortest such leg
    wins); a leg of 0 km is a missing value, not a zero length, and never places a tee.
    Otherwise, with ``route = (end a, end b, polyline from a to b)`` for two of the
    legs, the tee sits on the route at the arc length ``km_a / (km_a + km_b)`` of the route
    (the published legs scaled to the route length). Returns None when neither applies.
    """
    short = [leg for leg in legs if leg[3] and 0 < leg[2] < TEE_AT_STATION_KM]
    if short:
        end, point, km, _ = min(short, key=lambda leg: leg[2])
        return (point, "tee_at_station", f"leg to {end} {km:g} km < {TEE_AT_STATION_KM:g} km")
    if route is not None:
        a, b, coords = route
        km_of = {leg[0]: leg[2] for leg in legs}
        if a in km_of and b in km_of and km_of[a] + km_of[b] > 0:
            total = polyline_km(coords)
            share = km_of[a] / (km_of[a] + km_of[b])
            return (
                _point_at(coords, share * total),
                "tee_on_route",
                f"legs {a} {km_of[a]:g} km, {b} {km_of[b]:g} km; route {total:.1f} km, "
                f"tee at {share * total:.1f} km from {a}",
            )
    return None


def read_geometry_overlay(path: str | Path) -> pd.DataFrame:
    """Geometry overlay table (parquet or csv) with :data:`OVERLAY_COLUMNS`."""
    path = Path(path)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = set(OVERLAY_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"geometry overlay lacks columns {sorted(missing)}")
    return frame


def _route_coords(route: Any) -> list[tuple[float, float]]:
    if route is None or (isinstance(route, float) and np.isnan(route)):
        return []
    points = list(route)
    if points and isinstance(points[0], Mapping):
        return [(float(p["lon"]), float(p["lat"])) for p in points]
    return [(float(x), float(y)) for x, y in np.asarray(points, dtype=float).reshape(-1, 2)]


def _dist_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Distance in metres between two lon/lat points (equirectangular, fine below ~50 km)."""
    lat = math.radians((a[1] + b[1]) / 2)
    return math.hypot((a[0] - b[0]) * math.cos(lat) * 111_320.0, (a[1] - b[1]) * 110_574.0)


_SECTION_ID = re.compile(r"(?:^|export section )osm-(\d+)")


def section_of(eic: Any, route_source: Any) -> str | None:
    """Export section (``osm-N``) of a circuit taken from the OpenStreetMap export, if any.

    The ``osm-N-k`` circuits carry it in their EIC; curated circuits name it in
    ``route_source`` (``export section osm-N ...``).
    """
    for text in (eic, route_source):
        found = _SECTION_ID.search(str(text)) if isinstance(text, str) else None
        if found:
            return f"osm-{found.group(1)}"
    return None


def section_route(
    paths: Any, p1: tuple[float, float], p2: tuple[float, float]
) -> list[tuple[float, float]]:
    """Path along the ways of an export section between the vertices nearest ``p1`` and ``p2``.

    ``paths`` are the section's way pieces (lists of ``[lon, lat]``); they join at shared
    vertices, so a branching section gives the branch between the two ends. Empty when the
    two ends fall on the same vertex or are not connected.
    """
    graph: dict[tuple[float, float], dict[tuple[float, float], float]] = {}
    for path in paths if paths is not None else []:
        pts = [(float(x), float(y)) for x, y in path]
        for a, b in itertools.pairwise(pts):
            if a != b:
                d = _dist_m(a, b)
                graph.setdefault(a, {})[b] = d
                graph.setdefault(b, {})[a] = d
    if not graph or not all(math.isfinite(v) for v in (*p1, *p2)):
        return []
    start = min(graph, key=lambda v: _dist_m(v, p1))
    goal = min(graph, key=lambda v: _dist_m(v, p2))
    if start == goal:
        return []
    best = {start: 0.0}
    prev: dict[tuple[float, float], tuple[float, float]] = {}
    queue = [(0.0, start)]
    while queue:
        d, v = heapq.heappop(queue)
        if v == goal:
            break
        if d > best[v]:
            continue
        for w, step in graph[v].items():
            if d + step < best.get(w, math.inf):
                best[w] = d + step
                prev[w] = v
                heapq.heappush(queue, (d + step, w))
    if goal not in prev:
        return []
    out = [goal]
    while out[-1] != start:
        out.append(prev[out[-1]])
    return out[::-1]


def _xy(point: Any) -> tuple[float, float]:
    return (float(point.x), float(point.y))


def _starts_at_first(
    coords: list[tuple[float, float]], p1: tuple[float, float], p2: tuple[float, float]
) -> bool:
    """True if ``coords`` runs from ``p1`` to ``p2`` (rather than the other way)."""
    keep = _dist_m(coords[0], p1) + _dist_m(coords[-1], p2)
    flip = _dist_m(coords[0], p2) + _dist_m(coords[-1], p1)
    return keep <= flip


def location_flag(source: Any, lon: float, lat: float) -> str:
    """``"schematic"`` if the model says so, ``"rounded"`` for an unsurveyed point on a 0.01 deg
    grid, else ``""``."""
    text = str(source or "").lower()
    if "schematic" in text:
        return "schematic"
    if any(k in text for k in ("osm", "openstreetmap", "verified")):
        return ""  # surveyed point: a round value is a coincidence
    if all(abs(v * 100 - round(v * 100)) < 1e-6 for v in (lon, lat)):
        return "rounded"
    return ""


def _area(code: Any) -> str | None:
    """Control area of an operator string (first listed German TSO), else None."""
    for part in str(code or "").split(";"):
        part = part.strip()
        if part in _CONTROL_AREA:
            return _CONTROL_AREA[part]
        low = part.lower()
        for key, name in (
            ("50hertz", "50Hertz"),
            ("amprion", "Amprion"),
            ("tennet", "TenneT"),
            ("transnet", "TransnetBW"),
        ):
            if key in low:
                return name
    return None


def _joined(values: Any) -> str:
    if isinstance(values, (list, tuple, np.ndarray)):
        return "; ".join(str(v) for v in values)
    return "" if values is None or (isinstance(values, float) and np.isnan(values)) else str(values)


def _route_key(coords: list[tuple[float, float]]) -> str:
    """Direction-free key of a route rounded to ~1 m ("" for no route)."""
    if len(coords) < 2:
        return ""
    pts = [(round(x, 5), round(y, 5)) for x, y in coords]
    return json.dumps(min(pts, pts[::-1]))


def _corridors(
    geoms: list[Any], hvdc: list[bool], kv: list[float], names: list[str]
) -> tuple[list[int], list[int], list[int], list[Any]]:
    """Group circuits drawn on (nearly) the same path into corridors.

    Two circuits share a corridor when their display routes are within
    :data:`CORRIDOR_TOLERANCE_M` of each other (Hausdorff distance) and both are AC or
    both HVDC. Returns per circuit the corridor id, its index in the corridor (highest
    voltage first, then by name), the corridor size, and the geometry oriented like the
    corridor's first circuit (so a pixel offset puts the circuits side by side).
    """
    shapely = require("shapely", "grid")
    lat0 = math.radians(51.0)
    metric = [
        shapely.transform(g, lambda c: c * [math.cos(lat0) * 111_320.0, 110_574.0]) for g in geoms
    ]
    parent = list(range(len(geoms)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree = shapely.STRtree(metric)
    left, right = tree.query(metric, predicate="dwithin", distance=CORRIDOR_TOLERANCE_M)
    for i, j in zip(left.tolist(), right.tolist(), strict=True):
        if (
            i < j
            and hvdc[i] == hvdc[j]
            and find(i) != find(j)
            and shapely.hausdorff_distance(metric[i], metric[j]) < CORRIDOR_TOLERANCE_M
        ):
            parent[find(i)] = find(j)
    roots = [find(i) for i in range(len(geoms))]
    members: dict[int, list[int]] = {}
    for i, r in enumerate(roots):
        members.setdefault(r, []).append(i)
    corridor = [0] * len(geoms)
    index = [0] * len(geoms)
    size = [1] * len(geoms)
    oriented = list(geoms)
    order = sorted(members.values(), key=lambda m: min(m))
    for cid, group in enumerate(order):
        group = sorted(group, key=lambda i: (-kv[i], names[i]))
        ref = list(geoms[group[0]].coords)
        for k, i in enumerate(group):
            corridor[i], index[i], size[i] = cid, k, len(group)
            coords = list(geoms[i].coords)
            if k and _dist_m(coords[0], ref[-1]) < _dist_m(coords[0], ref[0]):
                oriented[i] = type(geoms[i])(coords[::-1])
    return corridor, index, size, oriented


def _andes_buses(case: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Bus table of an ANDES case with its generators summed per bus."""
    bus = case["Bus"][["idx", "name", "Vn"]].rename(columns={"idx": "andes_idx"})
    gens = []
    for kind in ("PV", "Slack"):
        if kind in case:
            gens.append(case[kind][["idx", "bus", "Sn"]])
    gen = pd.concat(gens) if gens else pd.DataFrame(columns=["idx", "bus", "Sn"])
    machines = []
    for model in ("GENROU", "GENCLS"):
        if model in case:
            frame = case[model][["gen", "M"]].copy()
            frame["model"] = model
            machines.append(frame)
    if machines:
        gen = gen.merge(pd.concat(machines), left_on="idx", right_on="gen", how="left")
    else:
        gen = gen.assign(M=np.nan, model=None)
    gen["hs"] = gen["M"].astype(float) / 2.0 * gen["Sn"].astype(float)
    per_bus = gen.groupby("bus").agg(
        andes_generators=("idx", "size"),
        andes_gen_sn_mva=("Sn", "sum"),
        andes_gen_models=("model", lambda v: "; ".join(sorted({str(x) for x in v if x}))),
        hs=("hs", "sum"),
    )
    per_bus["andes_gen_h_s"] = per_bus["hs"] / per_bus["andes_gen_sn_mva"]
    out = bus.merge(per_bus.drop(columns="hs"), left_on="andes_idx", right_index=True, how="left")
    out["andes_generators"] = out["andes_generators"].fillna(0).astype(int)
    return out


INVERTER_CATEGORY = {
    "Solare Strahlungsenergie": "solar",
    "Windenergie (Onshore-Anlage)": "wind_onshore",
    "Windenergie (Offshore-Anlage)": "wind_offshore",
    "Speicher": "storage",
}
"""Inverter fuel of the model (``grid_case._INVERTER``) -> category of the renewables layer."""

DISPLAY_ONLY_CABLES = {
    "DolWin5": (
        "DolWin5",
        "emden/ost",
        "NEP Strom 2037/2045 (2025), 2nd draft, ch. 5.3.1 p. 143: NOR-1-1 (DolWin5) in operation "
        "2025; offshorewind.biz 2025-10-17: DolWin5 connects Borkum Riffgrund 3 to Emden/Ost",
    ),
}
"""Register connection points the model's curated table (``offshore_connections_de.csv``)
does not name: cable and landing for the map popup only, with the source; not used by the
model."""


def _farm_table() -> pd.DataFrame:
    from importlib import resources

    path = resources.files("tranpy.resources.grid").joinpath("offshore_wind_farms_de.csv")
    with path.open(encoding="utf-8") as fh:
        frame = pd.read_csv(fh, dtype={"register_connection": str})
    frame["register_connection"] = frame["register_connection"].fillna("")
    return frame


def inverter_layers(
    silver: Mapping[str, pd.DataFrame],
    cim: Any,
    site_point: Any,
    *,
    area: Any = None,
) -> dict[str, Any]:
    """Display layers of the model's inverter infeed; the model is not changed.

    * ``renewable_inverters``: one point per bus with an inverter, at its substation; the
      capacity (``max_p_mw``) and units per category exactly as the model holds them, and
      ``in_case`` whether the bus is in the main component the ANDES case is built from
      (one REGCA1 per such bus, :func:`tranpy.simulation.grid_case.build_grid_case`);
    * ``offshore_farms``: the German offshore wind farms (``offshore_wind_farms_de.csv``) at
      their published position and capacity, their register connection point (with that
      point's register MW and units), export cable, landing
      substation, and how the model attaches that capacity;
    * ``offshore_links``: farm -> landing substation of the model (schematic straight line).

    ``site_point(bus)`` gives the display point of a bus; ``area(bus)`` its control area.
    """
    gpd = require("geopandas", "grid")
    geom = require("shapely.geometry", "grid")
    from tranpy.grid_register import OFFSHORE_FUEL, offshore_connections
    from tranpy.simulation.grid_case import _main_component

    machines = cim.synchronous_machine
    inv = machines[machines["fuel"].isin(INVERTER_CATEGORY)].copy()
    inv["category"] = inv["fuel"].map(INVERTER_CATEGORY)
    # main component exactly as build_grid_case selects it
    lines = cim.ac_line_segment
    lines = lines[lines["status"].fillna("in_operation").eq("in_operation")]
    lines = lines[lines["bus1_mrid"] != lines["bus2_mrid"]]
    trafo = cim.power_transformer
    trafo = trafo[trafo["bus1_mrid"] != trafo["bus2_mrid"]]
    pairs = list(zip(lines["bus1_mrid"], lines["bus2_mrid"], strict=True))
    pairs += list(zip(trafo["bus1_mrid"], trafo["bus2_mrid"], strict=True))
    keep = _main_component(list(cim.busbar_section["mrid"]), pairs)

    cats = list(dict.fromkeys(INVERTER_CATEGORY.values()))
    mw = inv.pivot_table(
        index=["bus_mrid", "node"], columns="category", values="max_p_mw", aggfunc="sum"
    ).reindex(columns=cats)
    units = inv.pivot_table(
        index=["bus_mrid", "node"], columns="category", values="units", aggfunc="sum"
    ).reindex(columns=cats)
    p0 = inv.groupby(["bus_mrid", "node"])["p_mw"].sum()
    table = pd.DataFrame(index=mw.index)
    for c in cats:
        table[f"{c}_mw"] = mw[c].fillna(0.0).round(1)
        table[f"{c}_units"] = units[c].fillna(0.0).round(1)
    table["capacity_mw"] = mw.fillna(0.0).sum(axis=1).round(1)
    table["dispatch_mw"] = p0.reindex(table.index).round(1)
    table["dominant"] = mw.fillna(0.0).idxmax(axis=1)
    table = table.reset_index().rename(columns={"node": "bus"})
    table["in_case"] = table["bus_mrid"].isin(keep)
    table["andes_model"] = np.where(table["in_case"] & (table["dispatch_mw"] > 0), "REGCA1", "")
    table["tso"] = table["bus"].map(area) if area is not None else None
    table["site"] = table["bus"].astype(str).str.split("|").str[0]
    table = table.sort_values("capacity_mw", ascending=False).reset_index(drop=True)
    pts = [site_point(b) for b in table["bus"]]
    inverters = gpd.GeoDataFrame(
        table.drop(columns=["bus_mrid"]),
        geometry=gpd.points_from_xy([p[0] for p in pts], [p[1] for p in pts]),
        crs=CRS,
    )

    # offshore: the model's register cells (MaStR units in 0.02 degree cells, with the
    # register connection point), attached as grid_register.attach_register does
    cells = silver["grid_de_register_cells"]
    cells = cells[cells["fuel"] == OFFSHORE_FUEL].copy()
    curated = offshore_connections().set_index("connection")
    model_nodes = set(machines.loc[machines["fuel"] == OFFSHORE_FUEL, "node"])
    farms = _farm_table()
    for row in farms[farms["lon"].isna()].itertuples():
        own = cells[cells["connection"] == row.register_connection]
        w = own["mw"].astype(float)
        farms.loc[row.Index, ["lon", "lat"]] = [
            float(np.average(own["lon"], weights=w)),
            float(np.average(own["lat"], weights=w)),
        ]
    code_mw = cells.groupby("connection")["mw"].sum()
    code_units = cells.groupby("connection")["units"].sum()

    def landing(code: str) -> tuple[str, str, str, str]:
        """(cable, landing, cable source, model attachment) of a register connection point."""
        if code in curated.index:
            rec = curated.loc[code]
            node = str(rec["node"])
            how = (
                f"offshore connection: {node}"
                if node in model_nodes
                else "offshore landing: spread equally over the landing substations of its sea "
                f"({node} is not a model substation)"
            )
            return str(rec["system"]), node.split("|")[0], str(rec["evidence"]), how
        how = "offshore landing: spread equally over the landing substations of its sea"
        if code in DISPLAY_ONLY_CABLES:
            cable, land, src = DISPLAY_ONLY_CABLES[code]
            return cable, land, src, how
        return "", "", "", how

    turbines = farms.groupby("register_connection")["turbines"].sum()
    rows, links = [], []
    for f in farms.itertuples():
        # connection point: given per farm; check: the turbines of the farms on one point sum
        # to its register units (``connection_evidence``)
        code = f.register_connection
        cable, land, src, how = landing(code) if code else ("", "", "", "")
        mw = float(f.capacity_mw) if pd.notna(f.capacity_mw) else float(code_mw.get(code, np.nan))
        flags = []
        if code not in code_mw.index:
            flags.append("connection point not in the register cells")
        if not cable:
            flags.append("export cable not sourced")
        if "spread equally" in how:
            flags.append("model spreads this capacity over the landing substations")
        rows.append(
            {
                "name": f.name,
                "sea": f.sea,
                "capacity_mw": round(mw, 1),
                "capacity_source": "published" if pd.notna(f.capacity_mw) else "register",
                "turbines": f.turbines,
                "commissioned": f.commissioned,
                "status": f.status,
                "connection_point": code,
                "connection_register_mw": round(float(code_mw.get(code, 0.0)), 1),
                "connection_register_units": int(code_units.get(code, 0)),
                "connection_farms_turbines": int(turbines.get(code, 0)),
                "cable": cable,
                "landing": land,
                "model_attachment": how,
                "cable_source": src,
                "farm_source": f.farm_source,
                "check": "; ".join(flags) or "ok",
                "geometry": geom.Point(float(f.lon), float(f.lat)),
            }
        )
        node = str(curated.loc[code, "node"]) if code in curated.index else ""
        end = site_point(node) if node in model_nodes else (np.nan, np.nan)
        if all(math.isfinite(v) for v in end):
            links.append(
                {
                    "name": f.name,
                    "cable": cable,
                    "landing": node,
                    "geometry": geom.LineString([(float(f.lon), float(f.lat)), end]),
                }
            )
    farms_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)
    farms_gdf = farms_gdf.sort_values("capacity_mw", ascending=False).reset_index(drop=True)
    links_gdf = gpd.GeoDataFrame(
        links or None, columns=["name", "cable", "landing", "geometry"], geometry="geometry",
        crs=CRS,
    )
    unassigned = set(code_mw.index) - set(farms_gdf["connection_point"])
    return {
        "renewable_inverters": inverters,
        "offshore_farms": farms_gdf,
        "offshore_links": links_gdf,
        "_unassigned_mw": float(code_mw[list(unassigned)].sum()),
    }


def build_layers(
    silver: Mapping[str, pd.DataFrame],
    cim: Any,
    *,
    borders: Mapping[str, Any],
    andes_case: Mapping[str, pd.DataFrame] | None = None,
    overlay: pd.DataFrame | None = None,
    towers: Any = None,
    stations: Any = None,
) -> dict[str, Any]:
    """GIS layers of the model (GeoDataFrames in EPSG:4326; ``metadata`` is a DataFrame).

    Args:
        silver: silver tables of the frozen model (``grid_de_*``).
        cim: CIM-like tables of the frozen model.
        borders: Natural Earth countries (GeoJSON with ``ADM0_A3``).
        andes_case: sheets of an ANDES case of the model (adds the simulation bus ids and
            the 110/21 kV connection buses to ``buses``); optional.
        overlay: geometry overlay (:func:`read_geometry_overlay`): corrected site points
            and line routes for display; the model is not changed.
        towers: OpenStreetMap tower lines (:func:`tranpy.grid_towers.read_tower_ways`):
            adds the ``tower_lines`` layer and draws circuits side by side per tower line
            (``lines`` then holds one row per piece); without it, circuits are grouped by
            corridor (Hausdorff distance) and ``tower_lines`` is empty.
        stations: station footprints for the station layout
            (:func:`tranpy.grid_stations.layout_stations`): OpenStreetMap substation
            polygons (:func:`tranpy.grid_stations.read_osm_footprints`) or a station table
            with ``site`` and polygon ``geometry``. Circuits then end at the footprint edge
            and the ``station_*`` layers are filled; without it they are empty.
    """
    gpd = require("geopandas", "grid")
    shapely = require("shapely", "grid")
    geom = require("shapely.geometry", "grid")
    shapely_ops = require("shapely.ops", "grid")

    nodes = silver["grid_de_nodes"].copy()
    cells = silver.get("grid_de_register_cells")
    nodes["tso_area"] = node_control_areas(nodes, cells).to_numpy()
    nodes["country"] = nodes["country"].fillna("DEU")
    # Natural Earth 1:50m puts some German border stations (Hagenwerder, Jardelund) just
    # outside Germany: a substation (not a foreign node or tee) within BORDER_TOLERANCE_KM
    # of the border is German.
    countries = gpd.GeoDataFrame.from_features(borders["features"], crs=CRS)
    germany = shapely.union_all(countries[countries["ADM0_A3"] == "DEU"].geometry.to_numpy())
    near = (
        shapely.distance(
            germany, gpd.points_from_xy(nodes["lon"].fillna(0), nodes["lat"].fillna(0))
        )
        * _KM_PER_DEGREE_LAT
    )  # degrees -> km (approximate, conservative in longitude)
    fixed = (
        (nodes["country"] != "DEU") & (nodes["kind"] == "substation") & (near < BORDER_TOLERANCE_KM)
    )
    nodes["country_source"] = np.where(fixed, "border tolerance", "natural earth")
    nodes.loc[fixed, "country"] = "DEU"
    zone_of = {k: v[0] for k, v in BIDDING_ZONES.items()}
    nodes["bidding_zone"] = nodes["country"].map(zone_of).fillna("DE-LU")
    busbar = cim.busbar_section
    node_by_mrid = dict(zip(busbar["mrid"], busbar["name"], strict=True))
    nodes_ix = nodes.set_index("node")
    xy = {n: (float(r.lon), float(r.lat)) for n, r in nodes_ix.iterrows()}

    # lines
    seg = cim.ac_line_segment.copy()
    seg["bus1"] = seg["bus1_mrid"].map(node_by_mrid)
    seg["bus2"] = seg["bus2_mrid"].map(node_by_mrid)
    circuits = silver["grid_de_circuits"]
    hvdc = circuits[circuits["hvdc"].astype(bool)]
    hv = pd.DataFrame(
        {
            "mrid": hvdc["eic"],
            "name": hvdc["name"],
            "eic": hvdc["eic"],
            "tso": hvdc["tso"],
            "bus1": hvdc["node0"],
            "bus2": hvdc["node1"],
            "nominal_voltage_kv": hvdc["kv"].astype(float),
            "length_km": hvdc["length_km"],
            "rated_current_a": hvdc["imax_a"],
            "tie_line": hvdc["tie_line"],
            "route": hvdc["route"],
            "route_source": hvdc["route_source"],
            "route_quality": hvdc["route_quality"],
            "status": hvdc["status"],
            "data_quality": hvdc["data_quality"],
        }
    )
    seg["hvdc"] = False
    hv["hvdc"] = True
    seg = pd.concat([seg, hv], ignore_index=True)

    # display positions: a substation whose model point is schematic (or on a 0.01 deg
    # grid) is drawn at the centroid of the route ends that terminate there; the model
    # itself is not changed (``location_flag``, ``display_relocated_m``)
    site_of = nodes_ix["site"].to_dict()
    source_of = dict(zip(busbar["name"], busbar["location_source"], strict=True))
    display_xy = {
        str(s): (float(x), float(y))
        for s, x, y in zip(
            cim.substation["site"], cim.substation["lon"], cim.substation["lat"], strict=True
        )
    }
    for node, point in xy.items():
        display_xy.setdefault(str(site_of.get(node, node)), point)
    over = overlay if overlay is not None else pd.DataFrame(columns=list(OVERLAY_COLUMNS))
    site_fix = {
        str(r.id): (tuple(json.loads(str(r.new))), f"overlay: {r.method}")
        for r in over[(over["element"] == "site") & (over["field"] == "point")].itertuples()
    }
    line_fix = {
        str(r.id): ([tuple(p) for p in json.loads(str(r.new))], f"overlay: {r.method}")
        for r in over[(over["element"] == "line") & (over["field"] == "route")].itertuples()
    }
    for node in xy:
        site = str(site_of.get(node, node))
        if site in site_fix:
            xy[node] = (float(site_fix[site][0][0]), float(site_fix[site][0][1]))
    for site, (point, _) in site_fix.items():
        display_xy[site] = (float(point[0]), float(point[1]))
    site_flag: dict[str, str] = {}
    for node, (lon, lat) in xy.items():
        flag = location_flag(source_of.get(node), lon, lat)
        if flag and str(site_of.get(node, node)) not in site_fix:
            site_flag.setdefault(str(site_of.get(node, node)), flag)
    routes = [
        line_fix[str(m)][0] if str(m) in line_fix else _route_coords(r)
        for m, r in zip(seg["mrid"], seg["route"], strict=True)
    ]
    route_ends: dict[str, list[tuple[float, float]]] = {}
    for coords, b1, b2 in zip(routes, seg["bus1"], seg["bus2"], strict=True):
        if len(coords) < 2 or b1 not in xy or b2 not in xy:
            continue
        s1, s2 = str(site_of.get(b1, b1)), str(site_of.get(b2, b2))
        if s1 == s2:
            continue
        first = _starts_at_first(coords, xy[b1], xy[b2])
        for site, bus, end in (
            (s1, b1, coords[0] if first else coords[-1]),
            (s2, b2, coords[-1] if first else coords[0]),
        ):
            if site in site_flag and 1.0 < _dist_m(end, xy[bus]) < END_SEARCH_M:
                route_ends.setdefault(site, []).append(end)
    relocated: dict[str, float] = {}
    for site, pts in route_ends.items():
        centre = (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])))
        relocated[site] = _dist_m(centre, display_xy[site])
        display_xy[site] = centre

    def bus_point(bus: Any) -> tuple[float, float]:
        """Model point of a bus, or its substation's display point when relocated."""
        site = site_of.get(bus)
        if site is not None and str(site) in relocated:
            return display_xy[str(site)]
        return xy.get(bus, (np.nan, np.nan))

    def site_point(bus: Any) -> tuple[float, float]:
        site = site_of.get(bus)
        if site is not None and str(site) in display_xy:
            return display_xy[str(site)]
        return xy.get(bus, (np.nan, np.nan))

    # buses
    buses = busbar[["mrid", "name", "kind", "lon", "lat", "location_source", "data_quality"]]
    buses = buses.rename(columns={"mrid": "bus_mrid", "name": "bus"})
    meta = nodes_ix.reindex(buses["bus"])
    buses = buses.assign(
        site=meta["site"].to_numpy(),
        voltage_kv=meta["kv"].astype(float).to_numpy(),
        tso_codes=meta["tso"].to_numpy(),
        tso=meta["tso_area"].to_numpy(),
        country=meta["country"].to_numpy(),
        country_source=meta["country_source"].to_numpy(),
        bidding_zone=meta["bidding_zone"].to_numpy(),
        names=[_joined(v) for v in meta["names"]],
    )
    site_tso = buses.groupby("site")["tso"].agg(lambda v: v.dropna().mode().iat[0])
    site_xy = buses.groupby("site")[["lon", "lat"]].mean()
    if andes_case is not None:
        andes = _andes_buses(andes_case)
        buses = buses.merge(andes.drop(columns="Vn"), left_on="bus", right_on="name", how="left")
        buses = buses.drop(columns="name")
        extra = andes[~andes["name"].isin(buses["bus"])].copy()
        extra["site"] = extra["name"].str.rsplit("|", n=1).str[0]
        extra["kind"] = np.where(extra["name"].str.startswith("term/"), "terminal", "connection")
        # buses whose name is not "<site>|kV" (generator terminals): site of a neighbour
        # bus in the case's branch table, propagated until nothing changes
        known = dict(zip(buses["bus"], buses["site"], strict=True))
        for name, site in zip(extra["name"], extra["site"], strict=True):
            if site in site_xy.index:
                known[name] = site
        name_of = dict(zip(andes["andes_idx"], andes["name"], strict=True))
        branches = andes_case.get("Line", pd.DataFrame(columns=["bus1", "bus2"]))
        ends = [
            (name_of.get(a), name_of.get(b))
            for a, b in zip(branches["bus1"], branches["bus2"], strict=True)
        ]
        changed = True
        while changed:
            changed = False
            for na, nb in ends:
                for u, v in ((na, nb), (nb, na)):
                    if u in known and v is not None and v not in known:
                        known[v] = known[u]
                        changed = True
        extra["site"] = extra["name"].map(known)
        extra = extra[extra["site"].isin(site_xy.index)]
        extra = pd.DataFrame(
            {
                "bus_mrid": None,
                "bus": extra["name"],
                "kind": extra["kind"],
                "lon": extra["site"].map(site_xy["lon"]),
                "lat": extra["site"].map(site_xy["lat"]),
                "location_source": "substation",
                "data_quality": "estimated",
                "site": extra["site"],
                "voltage_kv": extra["Vn"].astype(float),
                "tso_codes": None,
                "tso": extra["site"].map(site_tso),
                "country": "DEU",
                "bidding_zone": "DE-LU",
                "names": "",
                "andes_idx": extra["andes_idx"],
                "andes_generators": extra["andes_generators"],
                "andes_gen_sn_mva": extra["andes_gen_sn_mva"],
                "andes_gen_models": extra["andes_gen_models"],
                "andes_gen_h_s": extra["andes_gen_h_s"],
            }
        )
        buses = pd.concat([buses, extra], ignore_index=True)
        buses["andes_idx"] = buses["andes_idx"].astype("Int64")
    buses["tso"] = buses["tso"].fillna(buses["site"].map(site_tso))
    moved = buses["site"].isin(list(relocated))
    buses["location_flag"] = buses["site"].map(site_flag).astype(object).fillna("")
    buses["display_relocated_m"] = np.where(
        moved,
        [
            _dist_m((x, y), display_xy[s]) if m else 0.0
            for x, y, s, m in zip(buses["lon"], buses["lat"], buses["site"], moved, strict=True)
        ],
        0.0,
    ).round(0)
    buses.loc[moved, "lon"] = [display_xy[s][0] for s in buses.loc[moved, "site"]]
    buses.loc[moved, "lat"] = [display_xy[s][1] for s in buses.loc[moved, "site"]]
    fixed_site = buses["site"].astype(str).isin(list(site_fix))
    buses.loc[fixed_site, "lon"] = [site_fix[str(s)][0][0] for s in buses.loc[fixed_site, "site"]]
    buses.loc[fixed_site, "lat"] = [site_fix[str(s)][0][1] for s in buses.loc[fixed_site, "site"]]
    buses["geometry_source"] = [
        site_fix[str(s)][1] if f else ("display relocated" if m else "model")
        for s, f, m in zip(buses["site"], fixed_site, moved, strict=True)
    ]
    buses = gpd.GeoDataFrame(
        buses, geometry=gpd.points_from_xy(buses["lon"], buses["lat"]), crs=CRS
    )

    fixed_quality = dict.fromkeys(
        ("osm_route", "sibling_model_route", "named_osm_route", "osm_route_length_missing"),
        "mapped",
    )
    route_quality = [
        (
            fixed_quality.get(line_fix[str(m)][1].removeprefix("overlay: "), "straight")
            if str(m) in line_fix
            else q
        )
        for m, q in zip(seg["mrid"], seg["route_quality"], strict=True)
    ]
    # circuits taken from the OpenStreetMap export have no route in the model: draw them on
    # the ways of their export section (display only)
    sections = silver.get("grid_de_osm_sections")
    section_paths = (
        dict(zip(sections["section"], sections["paths"], strict=True))
        if sections is not None and "paths" in sections
        else {}
    )
    geometry_source = [line_fix[str(m)][1] if str(m) in line_fix else "model" for m in seg["mrid"]]
    known_lines = [geom.LineString(c) for c in routes if len(c) >= 2]
    tree = shapely.STRtree(known_lines) if known_lines else None

    def follow(end: tuple[float, float], point: tuple[float, float]) -> list[Any] | None:
        """Stretch of a drawn model route from near ``end`` to near ``point`` (else None)."""
        if tree is None or _dist_m(end, point) <= SNAP_MAX_M:
            return None
        here = geom.Point(end)
        best: tuple[float, list[Any]] | None = None
        for j in tree.query(here.buffer(FOLLOW_M / 70_000.0)):
            line = known_lines[int(j)]
            near = line.interpolate(line.project(geom.Point(point)))
            if _dist_m(_xy(line.interpolate(line.project(here))), end) > FOLLOW_M:
                continue
            if _dist_m(_xy(near), point) > SNAP_MAX_M:
                continue
            a, b = line.project(here), line.project(geom.Point(point))
            piece = shapely_ops.substring(line, min(a, b), max(a, b))
            coords = [(float(x), float(y)) for x, y in piece.coords]
            coords = coords if a <= b else coords[::-1]
            if best is None or polyline_km(coords) < best[0]:
                best = (polyline_km(coords), coords)
        return best[1] if best else None

    for k, (e, src, b1, b2) in enumerate(
        zip(seg["eic"], seg["route_source"], seg["bus1"], seg["bus2"], strict=True)
    ):
        section = section_of(e, src)
        if len(routes[k]) >= 2 or section not in section_paths:
            continue
        p1, p2 = bus_point(b1), bus_point(b2)
        drawn = section_route(section_paths[section], p1, p2)
        if len(drawn) < 2:
            continue
        if not _starts_at_first(drawn, p1, p2):
            drawn = drawn[::-1]
        how = f"export section {section}"
        # an end where the section joins a modelled line (its station further along that
        # line): continue on the drawn route of that line, else straight to the station
        head = follow(drawn[0], p1)
        tail = follow(drawn[-1], p2)
        if head is not None or tail is not None:
            how += " + modelled route to station"
        drawn = [*(head or [])[::-1], *drawn, *(tail or [])]
        for end, point, at in ((drawn[0], p1, 0), (drawn[-1], p2, len(drawn))):
            if _dist_m(end, point) > SNAP_MAX_M:
                drawn.insert(at, point)
                how += " + straight to station"
        routes[k] = drawn
        route_quality[k] = "mapped"
        geometry_source[k] = how
    geoms, quality, gaps, snaps = [], [], [], []
    for coords, rq, b1, b2 in zip(routes, route_quality, seg["bus1"], seg["bus2"], strict=True):
        p1, p2 = bus_point(b1), bus_point(b2)
        gap = snap = 0.0
        if len(coords) < 2:
            coords = [p1, p2]
            rq = "straight"
        elif all(math.isfinite(v) for v in (*p1, *p2)):
            a, b = (p1, p2) if _starts_at_first(coords, p1, p2) else (p2, p1)
            g0, g1 = _dist_m(coords[0], a), _dist_m(coords[-1], b)
            gap = max(g0, g1)
            if 1.0 < g0 < SNAP_MAX_M:
                coords = [a, *coords]
                snap = g0
            if 1.0 < g1 < SNAP_MAX_M:
                coords = [*coords, b]
                snap = max(snap, g1)
        if len(set(coords)) < 2:
            # both ends in one station: a 1 m stub keeps the geometry valid
            coords = [coords[0], (coords[0][0] + 1e-5, coords[0][1])]
            rq = "in_station"
        geoms.append(geom.LineString(coords))
        quality.append(rq)
        gaps.append(round(gap))
        snaps.append(round(snap))
    corridor, circuit_no, circuits_n, geoms = _corridors(
        geoms,
        seg["hvdc"].astype(bool).tolist(),
        seg["nominal_voltage_kv"].astype(float).tolist(),
        seg["name"].astype(str).tolist(),
    )
    route_key = pd.Series([_route_key(c) for c in routes], dtype=object)
    same_route = route_key.map(route_key[route_key != ""].value_counts()).fillna(1)

    country = nodes_ix["country"]
    zone = nodes_ix["bidding_zone"]
    area = nodes_ix["tso_area"]
    line_tso = [
        _area(t) or area.get(a) or area.get(b)
        for t, a, b in zip(seg["tso"], seg["bus1"], seg["bus2"], strict=True)
    ]
    from_c = seg["bus1"].map(country).fillna("DEU")
    to_c = seg["bus2"].map(country).fillna("DEU")
    is_ic = (from_c == "DEU") != (to_c == "DEU")  # exactly one end in Germany
    tie = seg["tie_line"].fillna(False).astype(bool)
    from_z = seg["bus1"].map(zone).fillna("DE-LU")
    to_z = seg["bus2"].map(zone).fillna("DE-LU")
    lines = gpd.GeoDataFrame(
        {
            "mrid": seg["mrid"],
            "name": seg["name"],
            "eic": seg["eic"],
            "voltage_kv": seg["nominal_voltage_kv"].astype(float),
            "corridor": corridor,
            "circuits": circuits_n,
            "circuit_no": circuit_no,
            "same_route": same_route.to_numpy(dtype=int),
            "length_km": seg["length_km"].astype(float),
            "r_ohm": seg.get("r_ohm"),
            "x_ohm": seg.get("x_ohm"),
            "b_us": seg.get("bch_s") * 1e6,
            "imax_a": seg["rated_current_a"].astype(float),
            "rating_mva": np.sqrt(3) * seg["nominal_voltage_kv"] * seg["rated_current_a"] / 1e3,
            "tso": line_tso,
            "tso_code": seg["tso"],
            "bus1": seg["bus1"],
            "bus2": seg["bus2"],
            "bidding_zone": np.where(from_z != "DE-LU", from_z, to_z),
            "from_zone": from_z,
            "to_zone": to_z,
            "is_interconnector": is_ic.to_numpy(),
            "tso_tie": (tie & ~is_ic & (from_c == "DEU") & (to_c == "DEU")).to_numpy(),
            "abroad": ((from_c != "DEU") & (to_c != "DEU")).to_numpy(),
            "neighbour": np.where(from_c != "DEU", from_c, np.where(to_c != "DEU", to_c, "")),
            "hvdc": seg["hvdc"].astype(bool),
            "route_quality": quality,
            "route_length_km": [round(polyline_km(g.coords), 3) for g in geoms],
            "geometry_source": geometry_source,
            "end_gap_m": gaps,
            "display_snapped_m": snaps,
            "route_source": seg["route_source"],
            "status": seg["status"].fillna("in_operation"),
            "data_quality": seg["data_quality"],
        },
        geometry=geoms,
        crs=CRS,
    )
    if towers is not None and len(towers):
        lines, tower_lines = split_by_towers(lines, towers)
    else:
        lines = lines.assign(tower_way="", piece=0, pieces=1, label_piece=True)
        tower_lines = empty_tower_lines()

    # substations
    sub = cim.substation.copy()
    per_site = buses[~buses["kind"].isin(["connection", "terminal"])].groupby("site")
    tr = cim.power_transformer.copy()
    tr["bus1"] = tr["bus1_mrid"].map(node_by_mrid)
    tr["bus2"] = tr["bus2_mrid"].map(node_by_mrid)
    tr["site"] = tr["bus1"].map(nodes_ix["site"])
    kinds = per_site["kind"].agg(lambda v: "substation" if "substation" in set(v) else v.iat[0])
    max_kv = sub["site"].map(per_site["voltage_kv"].max())
    min_kv = sub["site"].map(per_site["voltage_kv"].min())
    kind = sub["site"].map(kinds)
    coupling = (max_kv >= 300) & (min_kv < 300)
    symbol = np.where(
        kind.isin(["tee", "junction", "busbar"]),
        "tee",
        np.where(coupling, "coupling", np.where(max_kv >= 300, "380", "220")),
    )
    site_source = per_site["location_source"].first()
    sub_points = [
        display_xy.get(str(s), (x, y))
        for s, x, y in zip(sub["site"], sub["lon"], sub["lat"], strict=True)
    ]
    substations = gpd.GeoDataFrame(
        {
            "mrid": sub["mrid"],
            "site": sub["site"],
            "name": [_joined(v) for v in sub["names"]],
            "kind": kind,
            "symbol": symbol,
            "coupling": coupling.to_numpy(),
            "max_kv": max_kv,
            "voltages_kv": sub["site"].map(
                per_site["voltage_kv"].agg(lambda v: "/".join(f"{k:g}" for k in sorted(set(v))))
            ),
            "buses": sub["site"].map(per_site["bus"].agg(lambda v: "; ".join(sorted(v)))),
            "transformers": sub["site"].map(tr["site"].value_counts()).fillna(0).astype(int),
            "tso": sub["site"].map(site_tso),
            "country": sub["site"].map(per_site["country"].first()),
            "bidding_zone": sub["site"].map(per_site["bidding_zone"].first()),
            "location_source": sub["site"].map(site_source),
            "location_flag": sub["site"].map(site_flag).astype(object).fillna(""),
            "display_relocated_m": sub["site"].map(relocated).fillna(0.0).round(0),
            "geometry_source": [
                site_fix[str(s)][1]
                if str(s) in site_fix
                else ("display relocated" if str(s) in relocated else "model")
                for s in sub["site"]
            ],
        },
        geometry=gpd.points_from_xy([p[0] for p in sub_points], [p[1] for p in sub_points]),
        crs=CRS,
    )

    # transformers (at their substation; ``slot`` orders them for the symbol offset)
    tr["slot"] = tr.sort_values(["site", "rated_u2_kv", "name"]).groupby("site").cumcount()
    transformers = gpd.GeoDataFrame(
        {
            "mrid": tr["mrid"],
            "name": tr["name"],
            "eic": tr["eic"],
            "site": tr["site"],
            "bus_hv": tr["bus1"],
            "bus_lv": tr["bus2"],
            "kv_hv": tr["rated_u1_kv"].astype(float),
            "kv_lv": tr["rated_u2_kv"].astype(float),
            "sn_mva": np.sqrt(3) * tr["rated_u1_kv"] * tr["rated_current_a"] / 1e3,
            "r_ohm": tr["r_ohm"],
            "x_ohm": tr["x_ohm"],
            "phase_shifter": tr["phase_shifter"].astype(bool),
            "slot": tr["slot"].astype(int),
            "site_transformers": tr["site"].map(tr["site"].value_counts()).astype(int),
            "tso": [_area(t) or area.get(b) for t, b in zip(tr["tso"], tr["bus1"], strict=True)],
            "data_quality": tr["data_quality"],
        },
        geometry=[geom.Point(site_point(b)) for b in tr["bus1"]],
        crs=CRS,
    )

    # plants: >= PLANT_INDIVIDUAL_MW at their register location with a connector to their
    # substation; smaller units in operation aggregated per connection bus and fuel
    plants = silver["grid_de_plants"].copy()
    att = silver["grid_de_plant_attachments"].sort_values("share", ascending=False)
    att = att.drop_duplicates("mastr_id").set_index("mastr_id")
    plants["bus"] = plants["mastr_id"].map(att["node"])
    plants["attach_method"] = plants["mastr_id"].map(att["method"])
    located = plants["lon"].notna() & plants["lat"].notna()
    plants["location"] = np.where(located, "register", "bus")
    at_bus = [site_point(b) for b in plants["bus"]]
    plants["lon"] = plants["lon"].where(located, [p[0] for p in at_bus])
    plants["lat"] = plants["lat"].where(located, [p[1] for p in at_bus])
    plants = plants[plants["lon"].notna() & plants["lat"].notna()]
    bus_info = buses.set_index("bus")
    capacity = plants["capacity_mw"].astype(float)
    in_operation = plants["in_operation"].astype(bool)
    aggregated = in_operation & (capacity < PLANT_INDIVIDUAL_MW) & plants["bus"].notna()
    plants = gpd.GeoDataFrame(
        {
            "mastr_id": plants["mastr_id"],
            "name": plants["name"],
            "operator": plants["operator"],
            "fuel": plants["fuel"],
            "fuel_category": plants["fuel"].map(FUEL_CATEGORY).fillna("other"),
            "capacity_mw": capacity,
            "status": plants["status"],
            "in_operation": in_operation,
            "reserve": plants["reserve"].astype(bool),
            "voltage_level": plants["voltage_level"],
            "bus": plants["bus"],
            "attach_method": plants["attach_method"],
            "location": plants["location"],
            "display": np.where(aggregated, "aggregated", "individual"),
            "tso": plants["bus"].map(area),
            "andes_idx": (
                plants["bus"].map(bus_info["andes_idx"]) if "andes_idx" in bus_info else None
            ),
        },
        geometry=gpd.points_from_xy(plants["lon"], plants["lat"]),
        crs=CRS,
    )
    shown = plants[(plants["display"] == "individual") & plants["bus"].notna()]
    links = []
    for row in shown.itertuples():
        a = (row.geometry.x, row.geometry.y)
        b = site_point(row.bus)
        if all(math.isfinite(v) for v in b) and _dist_m(a, b) > 50.0:
            links.append(
                {
                    "mastr_id": row.mastr_id,
                    "name": row.name,
                    "bus": row.bus,
                    "fuel_category": row.fuel_category,
                    "capacity_mw": row.capacity_mw,
                    "in_operation": row.in_operation,
                    "length_m": round(_dist_m(a, b)),
                    "geometry": geom.LineString([a, b]),
                }
            )
    plant_connections = gpd.GeoDataFrame(
        links or None,
        columns=[
            "mastr_id",
            "name",
            "bus",
            "fuel_category",
            "capacity_mw",
            "in_operation",
            "length_m",
            "geometry",
        ],
        geometry="geometry",
        crs=CRS,
    )
    small = plants[plants["display"] == "aggregated"]
    agg = (
        small.groupby(["bus", "fuel_category"])
        .agg(
            capacity_mw=("capacity_mw", "sum"),
            units=("mastr_id", "size"),
            names=("name", lambda v: "; ".join(sorted(map(str, v))[:12])),
        )
        .reset_index()
        .sort_values(["bus", "capacity_mw"], ascending=[True, False])
    )
    agg["slot"] = agg.groupby("bus").cumcount()
    agg["bus_fuels"] = agg["bus"].map(agg["bus"].value_counts())
    agg["slot_y"] = agg["slot"] - (agg["bus_fuels"] - 1) / 2.0  # column centred on the bus
    agg["site"] = agg["bus"].map(site_of)
    agg["tso"] = agg["bus"].map(area)
    points = [site_point(b) for b in agg["bus"]]
    plant_aggregates = gpd.GeoDataFrame(
        agg.reset_index(drop=True),
        geometry=gpd.points_from_xy([p[0] for p in points], [p[1] for p in points]),
        crs=CRS,
    )

    # neighbours
    ext = cim.external_network_injection.copy()
    ext["bus"] = ext["bus_mrid"].map(node_by_mrid)
    ties = lines[lines["is_interconnector"]]
    tie_count = pd.concat([ties["bus1"], ties["bus2"]]).value_counts()
    neighbours = gpd.GeoDataFrame(
        {
            "mrid": ext["mrid"],
            "bus": ext["bus"],
            "name": ext["bus"].map(nodes_ix["names"]).map(_joined),
            "voltage_kv": ext["bus"].map(nodes_ix["kv"]).astype(float),
            "country": ext["bus"].map(country),
            "bidding_zone": ext["bus"].map(zone),
            "tie_lines": ext["bus"].map(tie_count).fillna(0).astype(int),
            "tso": ext["bus"].map(area),
        },
        geometry=[geom.Point(xy.get(b, (np.nan, np.nan))) for b in ext["bus"]],
        crs=CRS,
    )

    # bidding zones and control areas
    countries = countries[countries["ADM0_A3"].isin(BIDDING_ZONES)]
    countries = countries.assign(
        zone=countries["ADM0_A3"].map(lambda c: BIDDING_ZONES[c][0]),
        zone_name=countries["ADM0_A3"].map(lambda c: BIDDING_ZONES[c][1]),
        approximate=countries["ADM0_A3"].map(lambda c: BIDDING_ZONES[c][2]),
    )
    bidding = countries.dissolve(
        by="zone",
        aggfunc={"zone_name": "first", "approximate": "max", "ADM0_A3": lambda v: ";".join(v)},
    ).reset_index()
    bidding = bidding.rename(columns={"ADM0_A3": "countries"})[
        ["zone", "zone_name", "countries", "approximate", "geometry"]
    ]
    inside = buses[(buses["country"] == "DEU") & buses["tso"].notna()]
    inside = inside[~inside["kind"].isin(["connection", "terminal"])].drop_duplicates(
        subset=["lon", "lat"]
    )
    points = geom.MultiPoint(list(zip(inside["lon"], inside["lat"], strict=True)))
    cells_v = shapely.voronoi_polygons(points, extend_to=germany.buffer(2.0))
    pieces = list(shapely.get_parts(cells_v))
    owner = []
    tree = shapely.STRtree(pieces)
    for pt, tso in zip(inside.geometry, inside["tso"], strict=True):
        hit = tree.query(pt, predicate="intersects")
        owner.append((int(hit[0]), tso) if len(hit) else None)
    by_tso: dict[str, list[Any]] = {}
    for item in owner:
        if item is not None:
            by_tso.setdefault(item[1], []).append(pieces[item[0]])
    control = gpd.GeoDataFrame(
        {
            "tso": list(by_tso),
            "colour": [TSO_COLOURS.get(t, "#888888") for t in by_tso],
            "approximate": True,
            "method": "Voronoi cells of the TSO's buses, clipped to the German border",
        },
        geometry=[shapely.intersection(shapely.union_all(v), germany) for v in by_tso.values()],
        crs=CRS,
    )
    control = control.sort_values("tso").reset_index(drop=True)

    info = dict(getattr(cim, "model_info", {}) or {})
    metadata = pd.DataFrame(
        {"key": list(info), "value": [str(v) for v in info.values()]},
    )
    result = {
        "lines": lines,
        "osm_lines": gpd.GeoDataFrame({"way_id": []}, geometry=[], crs=CRS),
        "osm_substations": gpd.GeoDataFrame({"osm_id": []}, geometry=[], crs=CRS),
        "tower_lines": tower_lines,
        "substations": substations,
        "buses": buses.drop(columns=["lon", "lat"]),
        "transformers": transformers,
        "plant_connections": plant_connections,
        "plant_aggregates": plant_aggregates,
        "plants": plants,
        "neighbours": neighbours,
        **{
            k: v
            for k, v in inverter_layers(silver, cim, site_point, area=area).items()
            if not k.startswith("_")
        },
        "control_areas": control,
        "bidding_zones": bidding,
        "metadata": metadata,
    }
    if stations is not None:
        result.update(layout_stations(result, station_footprints(substations, stations)))
    else:
        result.update(empty_station_layers())
    return result


NETWORK_DROP = (
    "tower_way",
    "piece",
    "pieces",
    "label_piece",
    "station_end",
    "circuit_no",
    "circuits",
)
"""Per-piece columns of ``lines`` that the OSM-graph bundles replace."""


END_TOUCH_M = 1.0
"""A circuit end this close to its box edge or junction dot needs no end connector."""
END_SOLID_MAX_M = 300.0
"""An end connector to a junction dot longer than this is drawn dashed."""
BOX_REACH_M = 300.0
"""A station box grows (convex hull) to the route ends of its own circuits up to this far
outside its footprint; an end further away gets a dashed connector and a station_gaps row."""
BOX_MARGIN_M = 30.0
"""Margin around the grown station box."""
BOX_WAY_END_M = 150.0
"""A station box also grows to the open OSM way ends (portals, dead ends) this close to its
footprint, so the tower lines reach the box; never over another station or junction."""
SHORT_WAY_M = 150.0
"""``osm_lines`` pieces shorter than this get no check halo (``check`` ``short``: the check
of the corridor they belong to applies); without a model circuit and touching a station or
plant footprint they are left out (switchyard bays, plant-yard ways; still in
``tower_lines``)."""
YARD_INSIDE_M = 30.0
"""``osm_lines`` sections without a model circuit lying (90 %) within this distance of a
station or plant footprint are yard wiring and left out (any length)."""
HALO_CLASSES = ("model_fewer", "model_more")
"""``osm_lines.check`` values drawn with a mismatch halo (unexplained differences)."""
_CLIP_STEP_M = 5.0
CLIP_NEAR_M = 1000.0
"""A line end is cut where it last leaves its box only if the part cut away stays this
close to the box (else where it first leaves it)."""
CLIP_KEEP_M = 200.0
"""The box cut must leave at least this length (or the line is not cut)."""


def _route_note(rec: Mapping[str, Any], *, partial: bool = False) -> str:
    """Popup text of a circuit not fully routed on the OSM graph (status, break, override)."""
    status = str(rec.get("status") or "not_in_network")
    text = {
        "no_path_strict": "no connected OSM path between its stations",
        "no_terminal": "an end station or junction has no OSM attachment",
        "not_in_network": "not in the OSM network table",
    }.get(status, status)
    note = f"not fully on the OSM graph ({status}): {text}"
    x, y, gap = rec.get("break_lon"), rec.get("break_lat"), rec.get("break_gap_m")
    if x is not None and y is not None and pd.notna(x) and pd.notna(y):
        note += f"; break at {float(x):.4f}/{float(y):.4f}"
        if gap is not None and pd.notna(gap):
            note += f" (gap {float(gap):.0f} m)"
    reason = str(rec.get("override_reason") or "").strip()
    if reason:
        note += f"; override: {reason}"
    if partial:
        return note + "; the OSM pieces found are solid, the breaks dashed"
    return note + "; no OSM pieces: drawn dashed on the previous geometry"


def _orient(xy: list[np.ndarray]) -> list[bool]:
    """Per ordered piece: True when its own direction is the travel direction."""
    fwd: list[bool] = []
    for i, c in enumerate(xy):
        if i + 1 < len(xy):
            nb = xy[i + 1]
            d_end = min(np.hypot(*(c[-1] - nb[0])), np.hypot(*(c[-1] - nb[-1])))
            d_start = min(np.hypot(*(c[0] - nb[0])), np.hypot(*(c[0] - nb[-1])))
            fwd.append(bool(d_end <= d_start))
        elif i > 0:
            pv = xy[i - 1]
            tail = pv[-1] if fwd[-1] else pv[0]
            fwd.append(bool(np.hypot(*(c[0] - tail)) <= np.hypot(*(c[-1] - tail))))
        else:
            fwd.append(True)
    return fwd


def clip_chain(geoms: list[Any], box1: Any = None, box2: Any = None) -> tuple[list[Any], str]:
    """Pieces of one circuit (metric, in travel order) with the parts inside its two end
    boxes removed. Each end is cut where the line last leaves its box within its near half
    (if the part cut away stays within :data:`CLIP_NEAR_M` of the box, else where it first
    leaves it); a later pass by the station stays. The pieces keep their own direction; a
    piece left with nothing is ``None``. When less than :data:`CLIP_KEEP_M` would be left
    nothing is cut. Returns the pieces and ``box`` or ``none``.
    """
    geo = require("shapely.geometry", "grid")
    ops = require("shapely.ops", "grid")
    shapely = require("shapely", "grid")
    if not geoms:
        return [], "none"
    fwd = _orient([np.asarray(g.coords, float)[:, :2] for g in geoms])
    lengths = np.array([g.length for g in geoms])
    start = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
    total = float(lengths.sum())
    if total <= 0:
        return list(geoms), "none"
    pos, pts = [], []
    for g, f, s0, ln in zip(geoms, fwd, start, lengths, strict=True):
        n = max(2, int(np.ceil(ln / _CLIP_STEP_M)) + 1)
        d = np.linspace(0.0, ln, n)
        pts.extend(g.interpolate(d if f else ln - d))
        pos.extend(s0 + d)
    pos = np.asarray(pos)
    pts = np.asarray(pts, dtype=object)

    def cut(box: Any, from_start: bool) -> float:
        if box is None or box.is_empty:
            return 0.0 if from_start else total
        inside = shapely.contains(box, pts)
        half = pos <= total / 2 if from_start else pos >= total / 2
        hit = np.flatnonzero(inside & half)
        if not len(hit):
            return 0.0 if from_start else total
        near = box.buffer(CLIP_NEAR_M)
        if from_start:
            k = int(hit[-1])
            if shapely.contains(near, pts[: k + 1]).all():
                return float(pos[min(k + 1, len(pos) - 1)])
            if not inside[0]:
                return 0.0
            return float(pos[int(np.argmin(inside))]) if not inside.all() else total
        k = int(hit[0])
        if shapely.contains(near, pts[k:]).all():
            return float(pos[max(k - 1, 0)])
        if not inside[-1]:
            return total
        return float(pos[len(pos) - 1 - int(np.argmin(inside[::-1]))]) if not inside.all() else 0.0

    s1, s2 = cut(box1, True), cut(box2, False)
    if s2 - s1 < min(CLIP_KEEP_M, total - 1.0):
        return list(geoms), "none"
    out: list[Any] = []
    for g, f, s0, ln in zip(geoms, fwd, start, lengths, strict=True):
        a, b = max(s1, s0) - s0, min(s2, s0 + ln) - s0
        if b - a < 0.5:
            out.append(None)
            continue
        if a <= 1e-6 and b >= ln - 1e-6:
            out.append(g)
            continue
        lo, hi = (a, b) if f else (ln - b, ln - a)
        piece = ops.substring(g, lo, hi)
        out.append(piece if piece.geom_type == "LineString" else geo.LineString())
    return out, "box"


def _end_target(point: Any, target: Any) -> Any:
    """Where an end at ``point`` meets its ``target`` (box polygon: the nearest outline
    point, unless the end is inside; junction/station point: the point). None if there."""
    if target is None:
        return None
    ops = require("shapely.ops", "grid")
    if target.geom_type in ("Polygon", "MultiPolygon"):
        if target.contains(point) or target.distance(point) <= END_TOUCH_M:
            return None
        return ops.nearest_points(target.boundary, point)[0]
    return target if target.distance(point) > END_TOUCH_M else None


def network_lines(
    lines: Any,
    footprints: Any,
    network_dir: str | Path,
    *,
    substations: Any = None,
    buses: Any = None,
    yards: Mapping[str, Any] | None = None,
    tower_lines: Any = None,
    plant_shapes: list[Any] | None = None,
    review: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Model AC circuits drawn on the OpenStreetMap graph (display only).

    Station boxes grow (:func:`grow_boxes`) to their circuits' route ends within
    :data:`BOX_REACH_M` and to open OSM way ends within :data:`BOX_WAY_END_M`
    (:func:`way_end_points`), never over another station or junction. ``plant_shapes``
    (metric polygons of OSM ``power=plant``) and ``review`` (explained differences,
    :func:`osm_way_lines`) go to the ``osm_lines`` layer.

    ``yards`` (site -> metric polygon, :func:`station_yards`) adds a station's other OSM
    yards (e.g. a separate 380 kV polygon) to its box. Besides ``lines`` the result holds
    ``osm_lines`` (one feature per OSM 220/380 kV way, exact geometry cut at the station
    shapes, with the model circuits it carries and its check class, :func:`osm_way_lines`)
    and ``parallel_way_findings`` (:func:`parallel_way_findings`).

    ``network_dir`` is the output of :mod:`tranpy.grid_network`. A circuit with
    ``circuit_pieces`` (``routed``, or the partial route of a ``no_path_strict`` one) is
    drawn as its pieces (one OSM way, join or connector each) with the exact OSM geometry;
    :func:`tranpy.grid_bundles.piece_slots` gives each piece ``circuit_no`` (slot) and
    ``circuits`` (n), which the web map turns into a screen offset
    ``(slot - (n - 1) / 2) * gap_px(zoom)``. Nothing is drawn inside the circuit's own end
    boxes (``footprints``; :func:`clip_chain`). Where a route ends off its box edge or
    junction dot, a straight ``end`` connector joins it (``piece_kind`` ``end``, same
    slot; dashed when longer than :data:`END_SOLID_MAX_M`); a break inside a partial route
    is a dashed ``break`` connector between the piece ends. Junctions (``substations``
    kind ``tee``/``junction``) move to the OSM node where most legs attach
    (``junction_snap_m``).

    Circuits without pieces (``no_terminal``, ``no_path_strict`` without a partial route,
    very short in-station links, circuits missing from the network table) keep their rows
    of ``lines`` (the previous geometry chain), ``route_quality`` ``unrouted`` (dashed,
    centred), with dashed end connectors where needed. ``route_note`` gives the status,
    break and override reason. HVDC links stay as they are. Returns ``lines`` and, when
    given, the updated ``substations``.
    """
    gpd = require("geopandas", "grid")
    shapely = require("shapely", "grid")
    geo = require("shapely.geometry", "grid")
    from tranpy import grid_bundles

    network_dir = Path(network_dir)
    graph, routes_all = grid_bundles.read_network(network_dir)
    routes_all = routes_all.assign(mrid=routes_all["mrid"].astype(str))
    # every circuit of the table (read_network drops those without edges)
    rinfo = pd.read_parquet(network_dir / "circuit_routes.parquet")
    rinfo = rinfo.drop(columns=[c for c in ("geometry",) if c in rinfo])
    rinfo = rinfo.assign(mrid=rinfo["mrid"].astype(str)).set_index("mrid")
    pc = pd.read_parquet(network_dir / "circuit_pieces.parquet")
    pc["mrid"] = pc["mrid"].astype(str)
    drawable = set(routes_all.loc[routes_all["status"].isin(["routed", "no_path_strict"]), "mrid"])
    # route_overrides.csv: an override on a circuit that is not routed (e.g. Preilack -
    # Streumen 560, "wrong corridor; draw dashed") means its partial pieces are not drawn
    if "override_reason" in rinfo:
        reason = rinfo["override_reason"].fillna("").astype(str).str.strip()
        overridden = set(rinfo.index[(reason != "") & (rinfo["status"] != "routed")])
        drawable -= overridden
    pc = pc[pc["mrid"].isin(drawable)]
    routes = routes_all[routes_all["mrid"].isin(set(pc["mrid"]))].reset_index(drop=True)
    pieces = gpd.GeoDataFrame(
        pc.drop(columns="geometry"), geometry=shapely.from_wkb(pc["geometry"].to_numpy()), crs=CRS
    )
    pieces = grid_bundles.piece_slots(graph, routes, pieces)
    pieces = pieces.sort_values(["mrid", "seq"]).reset_index(drop=True)

    lines = lines.copy()
    mrid = lines["mrid"].astype(str)
    hvdc = lines["hvdc"].fillna(False).astype(bool)
    drawn = set(pieces["mrid"]) & set(mrid[~hvdc])
    pieces = pieces[pieces["mrid"].isin(drawn)]

    site_of_bus: dict[str, str] = {}
    if buses is not None and len(buses):
        site_of_bus = dict(zip(buses["bus"].astype(str), buses["site"].astype(str), strict=True))

    def site(bus: Any) -> str:
        return site_of_bus.get(str(bus), str(bus).split("|")[0])

    boxes: dict[str, Any] = {}
    if footprints is not None and len(footprints):
        fp = footprints.to_crs(METRIC_CRS)
        boxes = {str(k): g for k, g in fp.dissolve("site").geometry.items()}
        for s_, yard in (yards or {}).items():
            if s_ in boxes and yard is not None and not yard.is_empty:
                boxes[s_] = shapely.union_all([boxes[s_], yard])

    # ---- junction dots on the OSM node most of their legs attach to
    picks: dict[str, list[tuple[float, float]]] = {}
    for k in (1, 2):
        if f"entry{k}_kind" not in routes:
            continue
        hit = routes[routes[f"entry{k}_kind"] == "junction"]
        for bus, x, y in zip(
            hit[f"bus{k}"], hit[f"entry{k}_lon"], hit[f"entry{k}_lat"], strict=True
        ):
            if pd.notna(x) and pd.notna(y):
                picks.setdefault(site(bus), []).append((round(float(x), 7), round(float(y), 7)))
    junction_xy = {s_: max(set(v), key=v.count) for s_, v in picks.items()}
    points: dict[str, Any] = {}
    kinds: dict[str, str] = {}
    if substations is not None:
        substations = substations.copy()
        is_j = substations["kind"].isin(["tee", "junction"]) & substations["site"].astype(str).isin(
            junction_xy
        )
        was = substations.loc[is_j, ["site", "geometry"]]
        to = gpd.GeoSeries(
            [geo.Point(junction_xy[str(x)]) for x in was["site"]], index=was.index, crs=CRS
        )
        moved = to.to_crs(METRIC_CRS).distance(
            gpd.GeoSeries(was.geometry, crs=CRS).to_crs(METRIC_CRS)
        )
        old_xy = {str(x): (g.x, g.y) for x, g in zip(was["site"], was.geometry, strict=True)}
        substations.loc[was.index, "geometry"] = to.to_numpy()
        substations["junction_snap_m"] = 0.0
        substations.loc[was.index, "junction_snap_m"] = moved.round(1).to_numpy()
        sm = substations.to_crs(METRIC_CRS)
        points = dict(zip(sm["site"].astype(str), sm.geometry, strict=True))
        kinds = dict(zip(sm["site"].astype(str), sm["kind"].astype(str), strict=True))
    else:
        old_xy = {}

    def target(bus: Any) -> Any:
        s_ = site(bus)
        if kinds.get(s_) in ("tee", "junction"):
            return points.get(s_)
        return boxes.get(s_, points.get(s_))

    def connector(a: Any, b: Any) -> Any:
        return geo.LineString([(a.x, a.y), (b.x, b.y)])

    # ---- circuits with OSM pieces
    attrs = lines[mrid.isin(drawn) & ~hvdc].drop_duplicates("mrid")
    attrs = attrs.drop(columns=[c for c in (*NETWORK_DROP, "geometry") if c in attrs])
    attrs = attrs.assign(mrid=attrs["mrid"].astype(str)).set_index("mrid")
    pm = pieces.to_crs(METRIC_CRS)
    orig_boxes = dict(boxes)

    def chain_ends(gs: list[Any], b1: Any = None, b2: Any = None) -> tuple[Any, Any]:
        """Start and end of an ordered chain; with ``b1``/``b2`` the pair is swapped when
        the chain runs from bus2 to bus1 (old geometry chains)."""
        fw = _orient([np.asarray(g.coords, float)[:, :2] for g in gs])
        c0, c1 = list(gs[0].coords), list(gs[-1].coords)
        head = geo.Point(c0[0] if fw[0] else c0[-1])
        tail = geo.Point(c1[-1] if fw[-1] else c1[0])
        t1 = orig_boxes.get(site(b1), points.get(site(b1))) if b1 is not None else None
        t2 = orig_boxes.get(site(b2), points.get(site(b2))) if b2 is not None else None
        if (
            t1 is not None
            and t2 is not None
            and t1.distance(head) + t2.distance(tail) > t1.distance(tail) + t2.distance(head)
        ):
            head, tail = tail, head
        return head, tail

    def is_box(bus: Any) -> bool:
        return kinds.get(site(bus)) not in ("tee", "junction") and site(bus) in orig_boxes

    # ---- pass 1: route ends of every circuit; each station box grows to its own ends
    end_rows: list[dict[str, Any]] = []

    def note_ends(m: str, name: Any, gs: list[Any], b1: Any, b2: Any) -> None:
        if not gs:
            return
        head, tail = chain_ends(gs, b1, b2)
        for end, bus, which in ((head, b1, "bus1"), (tail, b2, "bus2")):
            if not is_box(bus):
                continue
            d = float(orig_boxes[site(bus)].distance(end))
            end_rows.append(
                {
                    "station": site(bus),
                    "circuit": str(name),
                    "mrid": m,
                    "end": which,
                    "distance_m": round(d, 1),
                    "within_cap": d <= BOX_REACH_M,
                    "point": end,
                }
            )

    clipped: dict[str, tuple[list[Any], str]] = {}
    for m, grp in pm.groupby("mrid", sort=False):
        b1, b2 = attrs.at[m, "bus1"], attrs.at[m, "bus2"]
        cut, level = clip_chain(
            list(grp.geometry),
            orig_boxes.get(site(b1)) if is_box(b1) else None,
            orig_boxes.get(site(b2)) if is_box(b2) else None,
        )
        clipped[m] = (cut, level)
        note_ends(
            m,
            attrs.at[m, "name"] if "name" in attrs else m,
            [g for g in cut if g is not None and not g.is_empty],
            b1,
            b2,
        )
    old_m = lines[~mrid.isin(set(clipped)) & ~hvdc].assign(mrid=lambda d: d["mrid"].astype(str))
    for m, grp in old_m.sort_values("piece").groupby("mrid", sort=False):
        gs = [g for g in grp.to_crs(METRIC_CRS).geometry if g is not None and not g.is_empty]
        if gs and all(g.geom_type == "LineString" for g in gs):
            note_ends(
                m,
                grp["name"].iloc[0] if "name" in grp else m,
                gs,
                grp["bus1"].iloc[0],
                grp["bus2"].iloc[0],
            )
    grown: dict[str, list[Any]] = {}
    for r in end_rows:
        if r["within_cap"] and r["distance_m"] > 0:
            grown.setdefault(r["station"], []).append(r["point"])
    # open OSM way ends (portals) just outside a box: the box reaches them too
    for s_, pts in way_end_points(graph, orig_boxes, BOX_WAY_END_M).items():
        grown.setdefault(s_, []).extend(pts)
    others = dict(orig_boxes)
    for k, pt in points.items():
        if k not in others:
            others[k] = pt
    boxes.update(grow_boxes(orig_boxes, grown, others, margin=BOX_MARGIN_M))
    gaps = pd.DataFrame(
        end_rows, columns=["station", "circuit", "mrid", "end", "distance_m", "within_cap", "point"]
    ).drop(columns="point")
    gaps = gaps[gaps["distance_m"] > BOX_MARGIN_M].reset_index(drop=True)

    rows, geoms = [], []
    for m, grp in pm.groupby("mrid", sort=False):
        b1, b2 = attrs.at[m, "bus1"], attrs.at[m, "bus2"]
        status = str(rinfo.at[m, "status"]) if m in rinfo.index else "routed"
        partial = status != "routed"
        t1, t2 = target(b1), target(b2)
        cut, level = clipped[m]
        recs = grp.drop(columns="geometry").to_dict("records")
        kept = [(r, g) for r, g in zip(recs, cut, strict=True) if g is not None and not g.is_empty]
        if not kept:
            continue
        fwd = _orient([np.asarray(g.coords, float)[:, :2] for _, g in kept])
        seq: list[tuple[dict[str, Any], Any]] = []

        def add(rec: dict[str, Any], g: Any, kind: str, dashed: bool, seq: list = seq) -> None:
            seq.append(({**rec, "piece_kind": kind, "dashed": dashed}, g))

        def travel_slot(rec: dict[str, Any], forward: bool) -> int:
            return int(rec["slot"]) if forward else int(rec["n"]) - 1 - int(rec["slot"])

        def ends(g: Any, forward: bool) -> tuple[Any, Any]:
            c = list(g.coords)
            a, b = geo.Point(c[0]), geo.Point(c[-1])
            return (a, b) if forward else (b, a)

        r0, g0 = kept[0]
        head, _ = ends(g0, fwd[0])
        to1 = _end_target(head, t1)
        if to1 is not None:
            c = connector(to1, head)
            rec = {**r0, "slot": travel_slot(r0, fwd[0]), "edge_id": -1, "way_id": -1}
            add(rec, c, "end", partial or is_box(b1) or c.length > END_SOLID_MAX_M)
        for i, (r, g) in enumerate(kept):
            if i:
                _, tail = ends(kept[i - 1][1], fwd[i - 1])
                start, _ = ends(g, fwd[i])
                if tail.distance(start) > END_TOUCH_M:
                    rp = kept[i - 1][0]
                    rec = {**rp, "slot": travel_slot(rp, fwd[i - 1]), "edge_id": -1, "way_id": -1}
                    add(rec, connector(tail, start), "break", True)
            add(r, g, str(r["kind"]), False)
        rl, gl = kept[-1]
        _, tail = ends(gl, fwd[-1])
        to2 = _end_target(tail, t2)
        if to2 is not None:
            c = connector(tail, to2)
            rec = {**rl, "slot": travel_slot(rl, fwd[-1]), "edge_id": -1, "way_id": -1}
            add(rec, c, "end", partial or is_box(b2) or c.length > END_SOLID_MAX_M)
        note = _route_note(rinfo.loc[m].to_dict(), partial=True) if partial else ""
        for k, (rec, g) in enumerate(seq):
            way = int(rec["way_id"])
            rows.append(
                {
                    "mrid": m,
                    "piece": k,
                    "tower_way": str(way) if way >= 0 and rec["piece_kind"] == "way" else "",
                    "piece_kind": rec["piece_kind"],
                    "edge_id": int(rec["edge_id"]),
                    "circuit_no": int(rec["slot"]),
                    "circuits": int(rec["n"]),
                    "station_cut": level,
                    "route_quality": "unrouted" if rec["dashed"] else "osm_graph",
                    "route_status": status,
                    "route_note": note,
                }
            )
            geoms.append(g)
    new = pd.DataFrame(rows)
    if len(new):
        new["pieces"] = new.groupby("mrid")["piece"].transform("count").astype(int)
        new["label_piece"] = new["piece"] == new.groupby("mrid")["piece"].transform(
            lambda p: p.iloc[len(p) // 2]
        )
        new = new.merge(attrs.reset_index(), on="mrid", how="left", suffixes=("", "_old"))
        new = new.drop(columns=[c for c in new.columns if c.endswith("_old")])
    new["geometry_source"] = [
        "OSM graph route (grid_network), OSM way geometry"
        if k in ("way", "join", "connector")
        else f"straight {k} connector (display)"
        for k in new.get("piece_kind", pd.Series(dtype=str))
    ]
    new = gpd.GeoDataFrame(new, geometry=geoms, crs=METRIC_CRS)
    drawn = set(new["mrid"]) if len(new) else set()

    # ---- no pieces: previous geometry chain, dashed, centred, joined to its ends
    old = lines[~mrid.isin(drawn) | hvdc].copy()
    old["mrid"] = old["mrid"].astype(str)
    old_hvdc = old["hvdc"].fillna(False).astype(bool)
    old["route_status"] = [
        "hvdc" if h else str(rinfo.at[m, "status"]) if m in rinfo.index else "not_in_network"
        for m, h in zip(old["mrid"], old_hvdc, strict=True)
    ]
    old["route_note"] = [
        "" if h else _route_note(rinfo.loc[m].to_dict() if m in rinfo.index else {"status": None})
        for m, h in zip(old["mrid"], old_hvdc, strict=True)
    ]
    old.loc[~old_hvdc, "route_quality"] = "unrouted"
    old.loc[~old_hvdc, "geometry_source"] = (
        old.loc[~old_hvdc, "geometry_source"].astype(str) + " (not routed on the OSM graph)"
    )
    for col, v in (("circuit_no", 0), ("circuits", 1)):
        if col in old:
            old.loc[~old_hvdc, col] = v
    if junction_xy and old_xy:
        old = _snap_old_ends(old, old_xy, junction_xy, site)
    old = old.to_crs(METRIC_CRS)
    if "piece" not in old:
        old["piece"] = 0
    old["piece_kind"] = ""
    old["station_cut"] = "none"
    extra = []
    for _m, grp in old[~old_hvdc].sort_values("piece").groupby("mrid", sort=False):
        gs = [g for g in grp.geometry if g is not None and not g.is_empty]
        if not gs or any(g.geom_type != "LineString" for g in gs):
            continue
        base = grp.iloc[0].drop(labels="geometry").to_dict()
        last = int(grp["piece"].max())
        b1, b2 = base["bus1"], base["bus2"]
        head, tail = chain_ends(gs, b1, b2)
        first_end = geo.Point(gs[0].coords[0]), geo.Point(gs[0].coords[-1])
        for end, t in ((head, target(b1)), (tail, target(b2))):
            # before the first piece when the end is one of the first piece's ends
            at_first = min(p.distance(end) for p in first_end) < END_TOUCH_M and (
                len(gs) > 1 or end.equals(first_end[0])
            )
            k = -1 if at_first else last + 1
            to = _end_target(end, t)
            if to is not None:
                extra.append(
                    {**base, "piece": k, "piece_kind": "end", "geometry": connector(end, to)}
                )
    if extra:
        old = pd.concat([old, gpd.GeoDataFrame(extra, geometry="geometry", crs=METRIC_CRS)])
        old["piece"] = old.groupby("mrid")["piece"].rank(method="first").astype(int) - 1
        old = old.sort_values(["mrid", "piece"]).reset_index(drop=True)
        n_old = old.groupby("mrid")["piece"].transform("count")
        if "pieces" in old:
            old["pieces"] = n_old.astype(int)
    old = gpd.GeoDataFrame(old, geometry="geometry", crs=METRIC_CRS)
    # ---- final cut: nothing of any circuit inside any drawn station shape (the grown
    # boxes are final here); through-lines keep their two outer parts
    shapes = [g for g in boxes.values() if g is not None and not g.is_empty]
    new, inside_new = _cut_out(new, shapes)
    old_ac, inside_old = _cut_out(old[~old["hvdc"].fillna(False).astype(bool)], shapes)
    old = pd.concat([old_ac, old[old["hvdc"].fillna(False).astype(bool)]], ignore_index=True)
    inside_only = sorted(inside_new | inside_old)
    # ends left off their own target by the cut (a foreign shape next to the station or
    # junction): a dashed end connector
    fix = []
    for frame in (new, old):
        ac = frame[~frame["hvdc"].fillna(False).astype(bool)] if "hvdc" in frame else frame
        for m, grp in ac.groupby("mrid", sort=False):
            gs = [g for g in grp.geometry if g is not None and not g.is_empty]
            if str(m) in inside_only or not gs:
                continue
            b1, b2 = grp["bus1"].iloc[0], grp["bus2"].iloc[0]
            head, tail = chain_ends(gs, b1, b2)
            base = grp.iloc[0].drop(labels="geometry").to_dict()
            for end, bus, k in ((head, b1, -1), (tail, b2, int(grp["piece"].max()) + 1)):
                t_ = target(bus)
                if t_ is None:
                    continue
                if t_.geom_type == "Point":
                    holder = [g for g in shapes if g.distance(t_) <= END_TOUCH_M]
                    if any(g.boundary.distance(end) <= END_TOUCH_M for g in holder):
                        continue
                to = _end_target(end, t_)
                if to is None:
                    continue
                c = connector(end, to) if k > 0 else connector(to, end)
                fix.append(
                    {
                        **base,
                        "piece": k,
                        "piece_kind": "end",
                        "route_quality": "unrouted",
                        "geometry": c,
                    }
                )
                if is_box(bus):
                    gaps.loc[len(gaps)] = {
                        "station": site(bus),
                        "circuit": str(base.get("name", m)),
                        "mrid": str(m),
                        "end": "bus1" if k < 0 else "bus2",
                        "distance_m": round(float(c.length), 1),
                        "within_cap": False,
                    }
    if fix:
        fx = gpd.GeoDataFrame(fix, geometry="geometry", crs=METRIC_CRS)
        is_new = fx["mrid"].isin(set(new["mrid"]))
        new = _renumber(pd.concat([new, fx[is_new]], ignore_index=True), new.crs)
        old = _renumber(pd.concat([old, fx[~is_new]], ignore_index=True), old.crs)
        # a connector crossing a neighbouring shape is cut there as well
        new, _ = _cut_out(new, shapes)
        old_ac, _ = _cut_out(old[~old["hvdc"].fillna(False).astype(bool)], shapes)
        old = gpd.GeoDataFrame(
            pd.concat([old_ac, old[old["hvdc"].fillna(False).astype(bool)]], ignore_index=True),
            geometry="geometry",
            crs=METRIC_CRS,
        )
    old = gpd.GeoDataFrame(old, geometry="geometry", crs=METRIC_CRS).to_crs(CRS)
    new = new.to_crs(CRS)
    new["route_length_km"] = (new.to_crs(METRIC_CRS).geometry.length / 1000).round(3)
    out = pd.concat([new, old], ignore_index=True)
    cols = [c for c in lines.columns if c != "geometry"]
    cols += [c for c in out.columns if c not in cols and c != "geometry"]
    result = {"lines": gpd.GeoDataFrame(out[[*cols, "geometry"]], geometry="geometry", crs=CRS)}
    if substations is not None:
        result["substations"] = substations
    if footprints is not None and len(footprints):
        fp = footprints.copy()
        fm = fp.to_crs(METRIC_CRS)
        fm["geometry_grown"] = [
            boxes.get(str(s_), g) for s_, g in zip(fm["site"], fm.geometry, strict=True)
        ]
        # one grown box per site (first row), the other rows of a site are dropped
        fm = fm.drop_duplicates("site")
        # the footprint itself (nothing of a circuit is drawn inside it), for the checks
        fm["footprint_wkt"] = [
            orig_boxes[str(s_)].wkt if str(s_) in orig_boxes else "" for s_ in fm["site"]
        ]
        # how far the box reaches beyond its footprint plus the margin (route and way ends)
        fm["grown_m"] = [
            round(max(0.0, float(g.hausdorff_distance(orig_boxes[str(s_)])) - BOX_MARGIN_M), 1)
            if str(s_) in orig_boxes
            else 0.0
            for s_, g in zip(fm["site"], fm["geometry_grown"], strict=True)
        ]
        fm = fm.set_geometry(gpd.GeoSeries(fm["geometry_grown"].to_numpy(), crs=METRIC_CRS))
        result["station_footprints"] = gpd.GeoDataFrame(
            fm.drop(columns="geometry_grown"), geometry="geometry", crs=METRIC_CRS
        ).to_crs(fp.crs)
    result["station_gaps"] = gaps.sort_values("distance_m", ascending=False).reset_index(drop=True)
    result["inside_only"] = inside_only
    on_graph = set(new.loc[new["piece_kind"] == "way", "mrid"].astype(str)) if len(new) else set()
    ways = osm_way_lines(
        network_dir,
        list(boxes.values()),
        lines,
        tower_lines,
        drawn=on_graph,
        yard_shapes=[*boxes.values(), *(plant_shapes or [])],
        review=review,
    )
    result["osm_lines"] = ways
    result["parallel_way_findings"] = parallel_way_findings(ways)
    return result


def _osm_yards(path: str | Path) -> tuple[list[Any], Any]:
    """Plant footprints (metric polygons of OSM ``power_plant``) and the OSM substation
    polygons of at least 110 kV (EPSG:4326, ``osm_id``, ``name``, ``operator``,
    ``voltage``) from an OpenStreetMap GeoPackage."""
    gpd = require("geopandas", "grid")
    pyogrio = require("pyogrio", "grid")
    names = set(pyogrio.list_layers(path)[:, 0])
    plants: list[Any] = []
    if "power_plant" in names:
        pl = gpd.read_file(path, layer="power_plant")
        pl = pl[pl.geometry.notna() & pl.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        plants = list(pl.to_crs(METRIC_CRS).geometry)
    subs = gpd.GeoDataFrame({"osm_id": []}, geometry=[], crs=CRS)
    if "power_substation_polygon" in names:
        sp = gpd.read_file(path, layer="power_substation_polygon")
        volts = sp.get("voltage", sp.get("voltages", pd.Series("", index=sp.index)))
        kv = (
            volts.astype(str).str.findall(r"\d+").map(lambda v: max((int(x) for x in v), default=0))
        )
        sp = sp[(kv >= 110_000) & sp.geometry.notna()]
        subs = gpd.GeoDataFrame(
            {
                "osm_id": sp.get("id", sp.get("osm_id", pd.Series("", index=sp.index))).astype(str),
                "name": sp.get("name", pd.Series("", index=sp.index)).fillna("").astype(str),
                "operator": sp.get("operator", pd.Series("", index=sp.index))
                .fillna("")
                .astype(str),
                "voltage": volts.loc[sp.index].fillna("").astype(str),
            },
            geometry=sp.geometry.to_numpy(),
            crs=sp.crs,
        ).to_crs(CRS)
    return plants, subs


def way_end_points(graph: Any, boxes: Mapping[str, Any], reach_m: float) -> dict[str, list[Any]]:
    """Open ends (degree 1) of the real OSM line edges of ``graph`` (``read_network``) lying
    outside a box of ``boxes`` (site -> metric polygon) but within ``reach_m`` of it, per
    site (metric points). An end within reach of two boxes goes to the nearer one."""
    gpd = require("geopandas", "grid")
    shapely = require("shapely", "grid")
    kind = graph["kind"] if "kind" in graph else pd.Series("line", index=graph.index)
    virtual = (
        graph["virtual"].astype(bool) if "virtual" in graph else pd.Series(False, index=graph.index)
    )
    e = graph[(kind == "line") & ~virtual]
    if not len(e) or not boxes:
        return {}
    deg = pd.concat([e["u"], e["v"]]).value_counts()
    geoms = gpd.GeoSeries(e.geometry.to_numpy(), crs=CRS).to_crs(METRIC_CRS).to_numpy()
    pts: dict[int, Any] = {}
    for u, v, g in zip(e["u"], e["v"], geoms, strict=True):
        if deg.get(u, 0) == 1:
            pts[int(u)] = shapely.get_point(g, 0)
        if deg.get(v, 0) == 1:
            pts[int(v)] = shapely.get_point(g, -1)
    sites = list(boxes)
    shapes = [boxes[k] for k in sites]
    tree = shapely.STRtree(shapes)
    out: dict[str, list[Any]] = {}
    for p in pts.values():
        hit = tree.query(p, predicate="dwithin", distance=reach_m)
        if not len(hit):
            continue
        d = [(float(shapes[int(i)].distance(p)), int(i)) for i in hit]
        dist, i = min(d)
        if dist > 0:
            out.setdefault(sites[i], []).append(p)
    return out


def grow_boxes(
    boxes: Mapping[str, Any],
    grown: Mapping[str, list[Any]],
    others: Mapping[str, Any],
    *,
    margin: float = BOX_MARGIN_M,
) -> dict[str, Any]:
    """Grown station boxes (metric): the convex hull of each box and its ``grown`` points,
    plus ``margin``. A grown box never reaches another station: points nearer to another
    site's shape (``others``: site -> original box or point) than to their own box are
    dropped, then the farthest points go until the hull keeps ``margin`` off every other
    shape; with no point left the box is its footprint plus ``margin``."""
    shapely = require("shapely", "grid")
    names = list(others)
    tree = shapely.STRtree([others[k] for k in names])
    out: dict[str, Any] = {}
    for s_, g in boxes.items():
        pts = list(grown.get(s_, []))

        def foreign(shape: Any, s_: str = s_) -> set[str]:
            hit = tree.query(shape, predicate="dwithin", distance=margin)
            return {names[int(i)] for i in hit if names[int(i)] != s_}

        near = {k: others[k] for k in foreign(g.buffer(BOX_REACH_M))}
        pts = [p for p in pts if all(o.distance(p) > g.distance(p) for o in near.values())]
        pts.sort(key=lambda p: g.distance(p))
        already = foreign(g)  # shapes touching the footprint itself (adjacent yards)
        while pts and not foreign(shapely.union_all([g, *pts]).convex_hull) <= already:
            pts.pop()
        out[s_] = (shapely.union_all([g, *pts]).convex_hull if pts else g).buffer(margin)
    return out


def station_yards(stations: Any, osm_polygons: Any, *, max_m: float = 200.0) -> dict[str, Any]:
    """Other OpenStreetMap substation polygons of each station (display only).

    A polygon of ``osm_polygons`` (``power_substation_polygon``: ``name``, ``operator``)
    within ``max_m`` of a station's footprint (``stations``: station table with
    ``station``, ``osm_operator``, ``osm_names``, polygon geometry) belongs to the station
    when its operator is the station's, or it has no operator and its name holds the
    station name or "Erweiterung" (an extension yard); polygons of a distribution operator
    and polygons touching another station's footprint are left out. Returns site -> union
    of the extra polygons (EPSG:3035).
    """
    shapely = require("shapely", "grid")
    st = stations.to_crs(METRIC_CRS)
    st = st[st.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    po = osm_polygons.to_crs(METRIC_CRS)
    po = po[po.geometry.notna() & ~po.geometry.is_empty]
    po = po[(po.geometry.area > 500) & (po.geometry.area < 400_000)]
    tree = shapely.STRtree(list(po.geometry))
    own_tree = shapely.STRtree(list(st.geometry))
    sites = list(st["station"].astype(str))
    out: dict[str, Any] = {}

    def norm(text: Any) -> str:
        return (
            ""
            if text is None or (isinstance(text, float) and np.isnan(text))
            else str(text).lower().strip()
        )

    for k, row in enumerate(st.itertuples(index=False)):
        f = row.geometry
        op = norm(getattr(row, "osm_operator", ""))
        names = {norm(n) for n in str(getattr(row, "osm_names", "") or "").split(";")}
        token = sites[k].split("/")[0]
        extra = []
        for i in tree.query(f.buffer(max_m)):
            p = po.iloc[int(i)]
            g = p.geometry
            if g.distance(f) > max_m or g.difference(f).area < 500:
                continue
            others = [int(j) for j in own_tree.query(g) if int(j) != k]
            if any(st.geometry.iloc[j].intersection(g).area > 100 for j in others):
                continue
            p_op, p_name = norm(p.get("operator")), norm(p.get("name"))
            same_op = bool(op) and p_op == op
            named = not p_op and (token in p_name or p_name in names or "erweiterung" in p_name)
            if same_op or named:
                extra.append(g)
        if extra:
            out[sites[k]] = shapely.union_all(extra)
    return out


WAY_CLASSES = ("match", "model_fewer", "model_more", "no_model", "no_tag", "short", "explained")
"""Check class of an OSM way on the map (see :func:`osm_way_lines`)."""


ROUTING_REASON = "model circuit drawn off this way (routing)"
"""Popup reason of a halo whose model circuit is drawn schematic / dashed / on another way."""


def count_class(model_n: int, tag: Any) -> str | None:
    """Check class from the section's own counts (``None`` when the OSM tag is missing)."""
    if tag is None or pd.isna(tag):
        return None
    if model_n == 0:
        return "no_model"
    if model_n > tag:
        return "model_more"
    return "model_fewer" if model_n < tag else "match"


def _way_class(classes: set[str]) -> str:
    if "model_more" in classes:
        return "model_more"
    if "model_fewer" in classes:
        return "model_fewer"
    return "no_tag" if classes == {"no_tag"} else "match"


def osm_way_lines(
    network_dir: str | Path,
    shapes: list[Any],
    lines: Any,
    tower_lines: Any = None,
    *,
    drawn: set[str] | None = None,
    yard_shapes: list[Any] | None = None,
    review: pd.DataFrame | None = None,
    short_m: float = SHORT_WAY_M,
) -> Any:
    """One feature per OSM 220/380 kV way section of the network graph (display only).

    A way is split where the set of model circuits on its edges changes (e.g. at a junction
    inside the way), so each feature carries exactly the circuits drawn on it. Exact OSM
    geometry, cut at the drawn station ``shapes`` (metric). Attributes: ``way_id``,
    ``name``, ``ref``, ``kv``, ``voltage_kv``, ``circuits_tag`` (OSM), ``operator`` (from
    ``tower_lines`` when known), ``construction`` (way tagged under construction /
    proposed, used only by a model route), ``model_n`` and ``model_circuits`` (JSON list of
    ``mrid``, ``name``, ``eic``, ``kv`` of the model circuits drawn on the section
    (``drawn`` mrids), from ``circuit_pieces``), ``check`` and ``check_reason``:

    - ``no_model`` (no model circuit), else from the section's own counts (``model_n`` vs
      ``circuits_tag``: ``match`` / ``model_fewer`` / ``model_more``, see :func:`count_class`);
      untagged sections from ``edge_check.csv`` (``no_tag`` when all edges are untagged);
    - ``short``: a section drawn shorter than ``short_m`` (no halo; the check of the
      corridor it belongs to applies);
    - ``explained``: a ``model_more`` / ``model_fewer`` section listed in ``review``
      (``way_id``, ``osm_n``, ``model_n``, ``reason``; a row with blank counts matches any
      count) with the same counts; ``check_reason`` holds the reason (no halo). A row with
      ``kind`` ``routing`` keeps the halo and sets ``check_reason`` (a model circuit of the
      way is drawn elsewhere, :data:`ROUTING_REASON`).

    Sections without a model circuit that are drawn shorter than ``short_m`` and touch a
    ``yard_shapes`` polygon (station boxes, plant footprints), or that lie (90 %) within
    :data:`YARD_INSIDE_M` of one, are left out, as are construction ways without a model
    circuit.
    """
    gpd = require("geopandas", "grid")
    shapely = require("shapely", "grid")
    ops = require("shapely.ops", "grid")
    d = Path(network_dir)
    e = pd.read_parquet(d / "edges.parquet")
    e = e[~e["virtual"].astype(bool) & (e["way_id"] >= 0)].copy()
    e["geom"] = shapely.from_wkb(e["geometry"].to_numpy())
    for col, default in (
        ("kv", "380"),
        ("ref", ""),
        ("name", ""),
        ("circuits_tag", np.nan),
        ("construction", False),
    ):
        if col not in e:
            e[col] = default
    if "length_m" not in e:
        e["length_m"] = gpd.GeoSeries(e["geom"].to_numpy(), crs=CRS).to_crs(METRIC_CRS).length
    pc = pd.read_parquet(d / "circuit_pieces.parquet")
    pc = pc[[c for c in ("mrid", "way_id", "edge_id", "kind") if c in pc]].copy()
    pc["mrid"] = pc["mrid"].astype(str)
    if drawn is None:
        drawn = set(pc["mrid"])
    pc = pc[(pc["kind"] == "way") & pc["mrid"].isin(drawn)]
    if "edge_id" in pc:
        per_edge = pc.groupby("edge_id")["mrid"].agg(lambda s: tuple(dict.fromkeys(s)))
        e["set"] = [per_edge.get(int(x), ()) for x in e["edge_id"]]
    else:  # older network output: one set per way
        per_way = pc.groupby("way_id")["mrid"].agg(lambda s: tuple(dict.fromkeys(s)))
        e["set"] = [per_way.get(int(w), ()) for w in e["way_id"]]
    e["key"] = [tuple(sorted(x)) for x in e["set"]]
    meta = lines.drop_duplicates("mrid").set_index(
        lines.drop_duplicates("mrid")["mrid"].astype(str)
    )
    ec = pd.read_csv(d / "edge_check.csv") if (d / "edge_check.csv").exists() else None
    edge_cls = (
        dict(zip(ec["edge_id"].astype(int), ec["class"].astype(str), strict=True))
        if ec is not None and "edge_id" in ec
        else {}
    )
    way_cls = {}
    if ec is not None:
        for w, grp in ec.groupby("way_id"):
            way_cls[int(w)] = _way_class(set(grp["class"].astype(str)))
    reasons: dict[int, list[tuple[float, float, str, str]]] = {}
    if review is not None and len(review):
        for r in review.itertuples(index=False):
            reasons.setdefault(int(r.way_id), []).append(
                (
                    float(getattr(r, "osm_n", np.nan)),
                    float(getattr(r, "model_n", np.nan)),
                    str(r.reason),
                    str(getattr(r, "kind", "") or ""),
                )
            )
    operator = {}
    if tower_lines is not None and len(tower_lines) and "osm_id" in tower_lines:
        operator = dict(
            zip(tower_lines["osm_id"].astype(str), tower_lines.get("operator", ""), strict=True)
        )
    rows, geoms = [], []
    for (w, _key), grp in e.groupby(["way_id", "key"], sort=True):
        parts = list(grp["geom"])
        merged = parts[0] if len(parts) == 1 else ops.linemerge(shapely.union_all(parts))
        mrids = list(dict.fromkeys(m for x in grp["set"] for m in x))
        building = bool(grp["construction"].fillna(False).astype(bool).any())
        if building and not mrids:
            continue
        circuits = [
            {
                "mrid": m,
                "name": str(meta.at[m, "name"]) if m in meta.index else m,
                "eic": str(meta.at[m, "eic"]) if m in meta.index and "eic" in meta else "",
                "kv": int(meta.at[m, "voltage_kv"])
                if m in meta.index and "voltage_kv" in meta and pd.notna(meta.at[m, "voltage_kv"])
                else None,
            }
            for m in mrids
        ]
        kv = str(grp["kv"].iloc[0])
        tag = grp["circuits_tag"].max()
        if edge_cls:
            cls = _way_class({edge_cls.get(int(x), "match") for x in grp["edge_id"]})
        else:
            cls = way_cls.get(int(w), "match")
        cls = count_class(len(circuits), tag) or cls
        rows.append(
            {
                "way_id": int(w),
                "name": str(grp["name"].iloc[0] or ""),
                "ref": str(grp["ref"].iloc[0] or ""),
                "kv": kv,
                "voltage_kv": 380 if "380" in kv else 220,
                "circuits_tag": None if pd.isna(tag) else int(tag),
                "operator": str(operator.get(str(int(w)), "") or ""),
                "construction": building,
                "model_n": len(circuits),
                "model_circuits": json.dumps(circuits, ensure_ascii=False),
                "check": "no_model" if not circuits else cls,
                "check_reason": "",
                "length_km": round(float(grp["length_m"].sum()) / 1000, 3),
            }
        )
        geoms.append(merged)
    ways = gpd.GeoDataFrame(rows, geometry=geoms, crs=CRS).to_crs(METRIC_CRS)
    if not len(ways):
        ways["geometry_full"] = ways.geometry
        return ways
    ways["geometry_full"] = ways.geometry
    if shapes:
        tree = shapely.STRtree(shapes)
        cut = []
        for g in ways.geometry:
            hit = tree.query(g, "intersects")
            cut.append(
                g.difference(shapely.union_all([shapes[int(i)] for i in hit])) if len(hit) else g
            )
        ways = ways.set_geometry(cut, crs=METRIC_CRS)
    ways = ways[~ways.geometry.is_empty].copy()
    drawn_m = ways.geometry.length
    short = drawn_m < short_m
    yards = [g for g in (yard_shapes or []) if g is not None and not g.is_empty]
    if yards:
        ytree = shapely.STRtree(yards)
        near = np.zeros(len(ways), dtype=bool)
        idx, _ = ytree.query(list(ways["geometry_full"]), predicate="dwithin", distance=1.0)
        near[np.unique(idx)] = True
        # without a model circuit and (nearly) all inside a yard (plant switchyard ways of
        # any length) also left out
        inside = np.zeros(len(ways), dtype=bool)
        none = (ways["model_n"] == 0).to_numpy()
        for k, g in enumerate(ways["geometry_full"]):
            if not none[k] or g.length <= 0:
                continue
            hit = ytree.query(g, predicate="dwithin", distance=YARD_INSIDE_M)
            if len(hit):
                zone = shapely.union_all([yards[int(i)].buffer(YARD_INSIDE_M) for i in hit])
                inside[k] = g.intersection(zone).length >= 0.9 * g.length
        ways = ways[~((short.to_numpy() & near & none) | (inside & none))].copy()
        short = ways.geometry.length < short_m
    for k in np.flatnonzero(((ways["check"].isin(HALO_CLASSES)) & ~short).to_numpy()):
        row = ways.iloc[k]
        for osm_n, model_n, reason, kind in reasons.get(int(row["way_id"]), []):
            tag = row["circuits_tag"]
            ok_osm = np.isnan(osm_n) or (tag is not None and pd.notna(tag) and osm_n == tag)
            ok_model = np.isnan(model_n) or model_n == row["model_n"]
            if ok_osm and ok_model:
                if kind != "routing":
                    ways.iloc[k, ways.columns.get_loc("check")] = "explained"
                ways.iloc[k, ways.columns.get_loc("check_reason")] = reason
                break
    in_halo = ways["check"].isin(HALO_CLASSES) & short
    ways.loc[in_halo, "check"] = "short"
    return ways


def parallel_way_findings(ways: Any, *, near_m: float = 150.0, min_km: float = 1.0) -> Any:
    """Circuits routed on one of two parallel OSM ways (routing review list, display only).

    For two ways of the same voltage running within ``near_m`` of each other over at least
    ``min_km``: ``ref_on_parallel`` when circuits on way A carry a number that the ref of
    way B names and A's ref does not; ``count_shift`` when A carries more model circuits
    than its OSM ``circuits`` while B carries none but has an OSM count. ``ways`` is
    :func:`osm_way_lines` (with ``geometry_full``). Returns a DataFrame (one row per finding).
    """
    import re

    shapely = require("shapely", "grid")
    full = list(ways["geometry_full"])
    tree = shapely.STRtree(full)
    recs = ways.drop(columns=["geometry", "geometry_full"]).to_dict("records")
    nums = [set(re.findall(r"\d{2,4}", str(r["ref"]))) for r in recs]
    rows = []
    for a, ga in enumerate(full):
        ra = recs[a]
        if ra["model_n"] == 0:
            continue
        circ = json.loads(ra["model_circuits"])
        for b in tree.query(ga.buffer(near_m)):
            b = int(b)
            rb = recs[b]
            if b == a or rb["voltage_kv"] != ra["voltage_kv"]:
                continue
            gb = full[b]
            steps = np.linspace(0, ga.length, max(2, int(ga.length // 50) + 1))
            near = np.array([gb.distance(ga.interpolate(s)) <= near_m for s in steps])
            km = near.mean() * ga.length / 1000
            if km < min_km:
                continue
            on_b = [
                c["name"]
                for c in circ
                if (set(re.findall(r"\d{2,4}", c["name"])) & nums[b])
                and not (set(re.findall(r"\d{2,4}", c["name"])) & nums[a])
            ]
            osm_a = ra["circuits_tag"]
            shift = (
                rb["model_n"] == 0
                and pd.notna(rb["circuits_tag"])
                and rb["circuits_tag"] > 0
                and pd.notna(osm_a)
                and ra["model_n"] > osm_a
            )
            for kind, names in (
                ("ref_on_parallel", on_b),
                ("count_shift", [c["name"] for c in circ] if shift else []),
            ):
                if not names:
                    continue
                mid = ga.interpolate(0.5, normalized=True)
                rows.append(
                    {
                        "kind": kind,
                        "way_a": ra["way_id"],
                        "ref_a": ra["ref"],
                        "name_a": ra["name"],
                        "osm_n_a": osm_a,
                        "model_n_a": ra["model_n"],
                        "way_b": rb["way_id"],
                        "ref_b": rb["ref"],
                        "name_b": rb["name"],
                        "osm_n_b": rb["circuits_tag"],
                        "model_n_b": rb["model_n"],
                        "kv": ra["voltage_kv"],
                        "parallel_km": round(km, 2),
                        "circuits": "; ".join(names),
                        "x": mid.x,
                        "y": mid.y,
                    }
                )
    out = pd.DataFrame(rows)
    if len(out):
        gpd = require("geopandas", "grid")
        ll = gpd.GeoSeries(gpd.points_from_xy(out["x"], out["y"]), crs=METRIC_CRS).to_crs(CRS)
        out["lon"], out["lat"] = ll.x.round(5), ll.y.round(5)
        out = out.drop(columns=["x", "y"]).sort_values(
            ["kind", "parallel_km"], ascending=[True, False]
        )
    return out.reset_index(drop=True)


def _renumber(frame: Any, crs: Any) -> Any:
    """Rows sorted by ``piece`` per circuit and renumbered 0..n-1 (``pieces`` updated)."""
    gpd = require("geopandas", "grid")
    out = frame.sort_values(["mrid", "piece"], kind="stable").reset_index(drop=True)
    out["piece"] = out.groupby("mrid").cumcount()
    if "pieces" in out:
        out["pieces"] = out.groupby("mrid")["piece"].transform("count")
    return gpd.GeoDataFrame(out, geometry="geometry", crs=crs)


def _cut_out(frame: Any, shapes: list[Any]) -> tuple[Any, set[str]]:
    """``frame`` (metric lines) with every part inside ``shapes`` removed; a line split by a
    shape becomes several rows (``piece`` renumbered per circuit). A circuit left with
    nothing keeps one row with an empty geometry and is returned in the set (inside a
    station only: shown in the station panel, not as a line)."""
    shapely = require("shapely", "grid")
    if not len(frame) or not shapes:
        return frame, set()
    tree = shapely.STRtree(shapes)
    rows, geoms, inside = [], [], set()
    recs = frame.drop(columns="geometry").to_dict("records")
    for rec, g in zip(recs, frame.geometry, strict=True):
        if g is None or g.is_empty:
            continue
        hit = tree.query(g, "intersects")
        if len(hit):
            g = g.difference(shapely.union_all([shapes[int(i)] for i in hit]))
        parts = list(g.geoms) if g.geom_type.startswith("Multi") else [g]
        for k, part in enumerate(x for x in parts if x.geom_type == "LineString"):
            if part.length < 0.5:
                continue
            rows.append({**rec, "_sub": k})
            geoms.append(part)
    out = frame.__class__(pd.DataFrame(rows), geometry=geoms, crs=frame.crs)
    lost = (
        set(frame["mrid"].astype(str)) - set(out["mrid"].astype(str))
        if len(out)
        else set(frame["mrid"].astype(str))
    )
    if lost:
        # nothing outside the shapes: one row without geometry (the station panel lists it)
        geo = require("shapely.geometry", "grid")
        keep = frame[frame["mrid"].astype(str).isin(lost)].drop_duplicates("mrid")
        keep = keep.assign(_sub=0).set_geometry([geo.LineString()] * len(keep), crs=frame.crs)
        out = pd.concat([out, keep], ignore_index=True)
        inside = lost
    out = out.sort_values(["mrid", "piece", "_sub"]).drop(columns="_sub").reset_index(drop=True)
    out["piece"] = out.groupby("mrid").cumcount()
    if "pieces" in out:
        out["pieces"] = out.groupby("mrid")["piece"].transform("count")
    return frame.__class__(out, geometry="geometry", crs=frame.crs), inside


def _snap_old_ends(
    old: Any,
    old_xy: Mapping[str, tuple[float, float]],
    new_xy: Mapping[str, tuple[float, float]],
    site: Any,
) -> Any:
    """Move the end vertices of kept (not routed) circuits from a moved junction's old
    point to its new point (lon/lat, within 1 m of the old point)."""
    geo = require("shapely.geometry", "grid")
    if not len(old):
        return old
    old = old.copy()
    geoms = list(old.geometry)
    for i, (b1, b2) in enumerate(zip(old["bus1"], old["bus2"], strict=True)):
        g = geoms[i]
        if g is None or g.is_empty or g.geom_type != "LineString":
            continue
        xy = [tuple(c) for c in g.coords]
        for bus in (b1, b2):
            k = site(bus)
            if k not in new_xy or k not in old_xy:
                continue
            for at in (0, -1):
                if _dist_m(xy[at], old_xy[k]) < 1.0:
                    xy[at] = new_xy[k]
        geoms[i] = geo.LineString(xy)
    return old.set_geometry(geoms, crs=old.crs)


def _network_note(network_dir: str | Path, lines: Any) -> str:
    """Metadata text of the OSM-graph drawing: folder, its model sha and the counts."""
    meta_f = Path(network_dir) / "meta.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    ac = lines[~lines["hvdc"].fillna(False).astype(bool)].drop_duplicates("mrid")
    counts = ac["route_quality"].value_counts().to_dict()
    return (
        f"{Path(network_dir).name} (model sha {str(meta.get('model_sha', ''))[:12]}; "
        f"{counts.get('osm_graph', 0)} circuits on the OSM graph, "
        f"{counts.get('unrouted', 0) + counts.get('straight', 0)} dashed)"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


RAW_TOWER_DROP = (
    "model_circuits",
    "model_names",
    "model_eics",
    "model_kv",
    "mismatch",
    "other_voltage_circuits",
    "parallel_way",
)
"""``tower_lines`` columns left out of the web map's raw OSM layer (model check is in
``osm_lines``; the old way-level counts there are not reliable)."""


def write_layers(
    layers: Mapping[str, Any], out_dir: str | Path, *, web_dir: str | Path | None = None
) -> Path:
    """Write the layers to ``out_dir/grid_model.gpkg`` (and GeoJSON files to ``web_dir``)."""
    pyogrio = require("pyogrio", "grid")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gpkg = out / "grid_model.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    for name in LAYERS:
        if name not in layers:
            continue
        frame = layers[name]
        if name == "metadata":
            pyogrio.write_dataframe(frame, gpkg, layer=name, driver="GPKG")
        else:
            frame.to_file(gpkg, layer=name, driver="GPKG", engine="pyogrio")
    if web_dir is not None:
        web = Path(web_dir)
        web.mkdir(parents=True, exist_ok=True)
        for name in LAYERS:
            if name not in layers:
                continue
            frame = layers[name]
            if name == "metadata":
                (web / "metadata.json").write_text(
                    json.dumps(dict(zip(frame["key"], frame["value"], strict=True)), indent=1)
                )
                continue
            frame = frame.copy()
            if name == "tower_lines":  # raw OSM layer: OSM tags only, the check is osm_lines
                frame = frame.drop(columns=[c for c in RAW_TOWER_DROP if c in frame])
            # line ends and box edges meet exactly: finer grid for the lines and boxes, so a
            # line grazing a box edge is not pushed into it by the rounding
            fine = name in ("lines", "osm_lines", "station_footprints")
            frame.geometry = frame.geometry.set_precision(1e-6 if fine else 1e-5)
            (web / f"{name}.geojson").write_text(frame.to_json(na="null", drop_id=True))
    return gpkg


def export_grid_gis(
    model_path: str | Path,
    out_dir: str | Path,
    *,
    borders_path: str | Path | None = None,
    andes_case: str | Path | None = None,
    web_dir: str | Path | None = None,
    overlay_path: str | Path | None = None,
    towers_path: str | Path | None = None,
    towers_overpass: tuple[str | Path, ...] = (),
    stations_path: str | Path | None = None,
    diagrams: bool = False,
    network_dir: str | Path | None = None,
    station_table: str | Path | None = None,
    osm_substations_path: str | Path | None = None,
    edge_review: str | Path | None = None,
    way_notes: str | Path | None = None,
) -> Path:
    """Export the frozen model at ``model_path`` to ``out_dir/grid_model.gpkg``.

    ``borders_path`` defaults to Natural Earth 1:50m countries (downloaded once);
    ``andes_case`` is an ANDES ``.xlsx`` case of the model (adds simulation bus ids);
    ``web_dir`` also writes GeoJSON per layer for the web viewer; ``overlay_path`` applies
    a geometry overlay (:func:`read_geometry_overlay`); ``towers_path`` (OpenStreetMap
    line export, plus ``towers_overpass`` JSON downloads) adds the tower lines and writes
    ``tower_circuit_mismatch.csv`` (ways whose OpenStreetMap circuit count differs from the
    model circuits drawn on them) next to the GeoPackage. ``stations_path`` (OpenStreetMap
    GeoPackage with ``power_substation_polygon``, or a station table with ``site``) lays
    out the stations (:func:`tranpy.grid_stations.layout_stations`). ``diagrams`` writes a
    pypowsybl single-line diagram per substation to ``web_dir/sld`` (needs ``web_dir`` and
    the ``[powsybl]`` extra) and links it from ``substations.diagram``.

    ``network_dir`` (output of :mod:`tranpy.grid_network`) draws the model AC circuits on
    the OpenStreetMap graph: exact OSM way pieces with a slot per piece for the web map's
    screen offset, ending on their own station box edge
    (:func:`network_lines`; needs ``stations_path`` for the station boxes).
    ``station_table`` (folder with ``stations.parquet``, ``checks.csv``) adds the OSM names
    and flags to the station popup written to ``web_dir/station_details.json``; with
    ``web_dir`` the popup file is always written. ``osm_substations_path`` (OpenStreetMap
    GeoPackage with ``power_substation_polygon``) adds each station's other yards to its box
    (:func:`station_yards`). With ``network_dir`` the export also writes ``osm_lines`` (one
    feature per OSM way) and ``parallel_way_findings.csv``; ``osm_substations_path`` then
    also gives the plant footprints (``power_plant``) whose short yard ways ``osm_lines``
    leaves out, and the raw ``osm_substations`` layer (OSM substation polygons of at least
    110 kV). ``edge_review`` (CSV ``way_id``, ``osm_n``, ``model_n``, ``reason``) marks
    reviewed, explained model/OSM differences in ``osm_lines`` (:func:`osm_way_lines`).
    ``way_notes`` (CSV ``way_id``, ``note``) adds a reviewed ``note`` (e.g. the expansion
    project of a way under construction, with its source) to the raw ``tower_lines`` layer;
    without it the layer has no ``note`` column.
    """
    model_path = Path(model_path)
    if borders_path is None:
        from tranpy.maps import NATURAL_EARTH_COUNTRIES
        from tranpy_core.data.download import fetch

        borders_path = fetch(NATURAL_EARTH_COUNTRIES)
    borders = json.loads(Path(borders_path).read_text(encoding="utf-8"))
    case = pd.read_excel(andes_case, sheet_name=None) if andes_case is not None else None
    silver, cim = load_model(model_path)
    overlay = read_geometry_overlay(overlay_path) if overlay_path is not None else None
    towers = (
        read_tower_ways(towers_path, overpass=towers_overpass) if towers_path is not None else None
    )
    stations = None
    if stations_path is not None:
        stations = read_osm_footprints(stations_path)
    layers = build_layers(
        silver,
        cim,
        borders=borders,
        andes_case=case,
        overlay=overlay,
        towers=towers,
        stations=stations,
    )
    yards = None
    if network_dir is not None and osm_substations_path is not None and stations_path is not None:
        gpd = require("geopandas", "grid")
        yards = station_yards(
            gpd.read_file(stations_path, layer="stations"),
            gpd.read_file(osm_substations_path, layer="power_substation_polygon"),
        )
    plant_shapes = None
    if network_dir is not None and osm_substations_path is not None:
        plant_shapes, layers["osm_substations"] = _osm_yards(osm_substations_path)
    review = pd.read_csv(edge_review) if edge_review is not None else None
    if network_dir is not None:
        layers.update(
            network_lines(
                layers["lines"],
                layers["station_footprints"] if stations is not None else None,
                network_dir,
                substations=layers["substations"],
                buses=layers["buses"],
                yards=yards,
                tower_lines=layers["tower_lines"],
                plant_shapes=plant_shapes,
                review=review,
            )
        )
    osm_lines = layers.get("osm_lines")
    if osm_lines is None:
        gpd = require("geopandas", "grid")
        osm_lines = gpd.GeoDataFrame({"way_id": []}, geometry=[], crs=CRS)
    elif "geometry_full" in osm_lines:
        osm_lines = osm_lines.drop(columns="geometry_full").to_crs(CRS)
    layers["osm_lines"] = osm_lines
    parallel = layers.pop("parallel_way_findings", None)
    if parallel is not None:
        out_par = Path(out_dir) / "parallel_way_findings.csv"
        out_par.parent.mkdir(parents=True, exist_ok=True)
        parallel.to_csv(out_par, index=False)
    inside_only = layers.pop("inside_only", [])
    gaps = layers.pop("station_gaps", None)
    if gaps is not None:
        # model-OSM deviation for review: route ends off their own station footprint
        out_gaps = Path(out_dir) / "station_gaps.csv"
        out_gaps.parent.mkdir(parents=True, exist_ok=True)
        out_gaps.write_text(
            "# model-OSM deviation for review: circuit route ends outside their own station "
            f"footprint (> {BOX_MARGIN_M:g} m; within_cap: the box grows to the end, else a "
            f"dashed connector, cap {BOX_REACH_M:g} m)\n" + gaps.to_csv(index=False)
        )
    subs = layers["substations"]
    subs["diagram"] = ""
    if diagrams and web_dir is not None:
        written = write_station_diagrams(cim, Path(web_dir) / "sld")
        subs["diagram"] = (
            subs["site"].astype(str).map({k: f"sld/{v}" for k, v in written.items()}).fillna("")
        )
    if web_dir is not None:
        write_station_details(layers, Path(web_dir) / "station_details.json", station_table)
    if way_notes is not None and towers is not None:
        notes = pd.read_csv(way_notes, dtype={"way_id": "int64", "note": str})
        tl = layers["tower_lines"]
        by_way = dict(zip(notes["way_id"], notes["note"], strict=True))
        tl["note"] = tl["osm_id"].astype("int64").map(by_way)
        tl["note"] = tl["note"].fillna("")
    if towers is not None:
        tl = layers["tower_lines"]
        out_csv = Path(out_dir) / "tower_circuit_mismatch.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        tl[tl["mismatch"] != ""].drop(columns="geometry").to_csv(out_csv, index=False)
    now = datetime.now(UTC)
    extra = {
        "build_id": now.strftime("%Y%m%dT%H%M%SZ"),
        "source_file": model_path.name,
        "source_sha256": _sha256(model_path),
        "exported_utc": now.isoformat(timespec="seconds"),
        "andes_case": Path(andes_case).name if andes_case is not None else "",
        "geometry_overlay": (
            f"{Path(overlay_path).name} ({len(overlay)} corrections)"
            if overlay is not None and overlay_path is not None
            else ""
        ),
        "tower_lines": (
            f"{Path(towers_path).name} ({len(towers)} ways)"
            if towers is not None and towers_path is not None
            else ""
        ),
        "stations": (
            f"{Path(stations_path).name} ({(layers['station_footprints']['source'] != 'box').sum()}"
            f" footprints, {(layers['station_footprints']['source'] == 'box').sum()} boxes)"
            if stations_path is not None
            else ""
        ),
        "network": _network_note(network_dir, layers["lines"]) if network_dir is not None else "",
        "inside_station_only": ";".join(inside_only),
        "station_table": Path(station_table).name if station_table is not None else "",
        "edge_review": (
            f"{Path(edge_review).name} ({len(review)} rows)" if review is not None else ""
        ),
        "crs": CRS,
    }
    layers["metadata"] = pd.concat(
        [layers["metadata"], pd.DataFrame({"key": list(extra), "value": list(extra.values())})],
        ignore_index=True,
    )
    return write_layers(layers, out_dir, web_dir=web_dir)
