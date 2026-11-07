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
import itertools
import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tranpy.grid_register import _CONTROL_AREA, node_control_areas
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
]

CRS = "EPSG:4326"
BORDER_TOLERANCE_KM = 3.0
"""A node of kind ``substation`` this close to the German border counts as German."""
_KM_PER_DEGREE_LAT = 111.2
SNAP_MAX_M = 3000.0
"""A route end this close to its substation is extended to the substation point (display)."""
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
    "tower_lines",
    "substations",
    "buses",
    "transformers",
    "plant_connections",
    "plant_aggregates",
    "plants",
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


def build_layers(
    silver: Mapping[str, pd.DataFrame],
    cim: Any,
    *,
    borders: Mapping[str, Any],
    andes_case: Mapping[str, pd.DataFrame] | None = None,
    overlay: pd.DataFrame | None = None,
    towers: Any = None,
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
    """
    gpd = require("geopandas", "grid")
    shapely = require("shapely", "grid")
    geom = require("shapely.geometry", "grid")

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
            "geometry_source": [
                line_fix[str(m)][1] if str(m) in line_fix else "model" for m in seg["mrid"]
            ],
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
    return {
        "lines": lines,
        "tower_lines": tower_lines,
        "substations": substations,
        "buses": buses.drop(columns=["lon", "lat"]),
        "transformers": transformers,
        "plant_connections": plant_connections,
        "plant_aggregates": plant_aggregates,
        "plants": plants,
        "neighbours": neighbours,
        "control_areas": control,
        "bidding_zones": bidding,
        "metadata": metadata,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        frame = layers[name]
        if name == "metadata":
            pyogrio.write_dataframe(frame, gpkg, layer=name, driver="GPKG")
        else:
            frame.to_file(gpkg, layer=name, driver="GPKG", engine="pyogrio")
    if web_dir is not None:
        web = Path(web_dir)
        web.mkdir(parents=True, exist_ok=True)
        for name in LAYERS:
            frame = layers[name]
            if name == "metadata":
                (web / "metadata.json").write_text(
                    json.dumps(dict(zip(frame["key"], frame["value"], strict=True)), indent=1)
                )
                continue
            frame = frame.copy()
            frame.geometry = frame.geometry.set_precision(1e-5)
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
) -> Path:
    """Export the frozen model at ``model_path`` to ``out_dir/grid_model.gpkg``.

    ``borders_path`` defaults to Natural Earth 1:50m countries (downloaded once);
    ``andes_case`` is an ANDES ``.xlsx`` case of the model (adds simulation bus ids);
    ``web_dir`` also writes GeoJSON per layer for the web viewer; ``overlay_path`` applies
    a geometry overlay (:func:`read_geometry_overlay`); ``towers_path`` (OpenStreetMap
    line export, plus ``towers_overpass`` JSON downloads) adds the tower lines and writes
    ``tower_circuit_mismatch.csv`` (ways whose OpenStreetMap circuit count differs from the
    model circuits drawn on them) next to the GeoPackage.
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
    layers = build_layers(
        silver, cim, borders=borders, andes_case=case, overlay=overlay, towers=towers
    )
    if towers is not None:
        tl = layers["tower_lines"]
        out_csv = Path(out_dir) / "tower_circuit_mismatch.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        tl[tl["mismatch"] != ""].drop(columns="geometry").to_csv(out_csv, index=False)
    extra = {
        "source_file": model_path.name,
        "source_sha256": _sha256(model_path),
        "exported_utc": datetime.now(UTC).isoformat(timespec="seconds"),
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
        "crs": CRS,
    }
    layers["metadata"] = pd.concat(
        [layers["metadata"], pd.DataFrame({"key": list(extra), "value": list(extra.values())})],
        ignore_index=True,
    )
    return write_layers(layers, out_dir, web_dir=web_dir)
