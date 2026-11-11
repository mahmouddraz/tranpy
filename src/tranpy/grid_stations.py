"""Station layout for the grid map: one plain box per station (display only).

Without this step every circuit is drawn into one point (the station's model or display
point), so circuits from all directions and both voltage levels tangle inside a station.
:func:`layout_stations` instead

1. gives each substation a footprint: its OpenStreetMap ``power=substation`` polygon(s)
   (:func:`read_osm_footprints`, :func:`station_footprints`) or, without one, a fixed-size
   box around the display point;
   A footprint that no circuit end reaches is replaced by a box centred at the circuit
   ends (``source`` ``box at circuit ends``);
2. clips every circuit at the footprint of its end station: the circuit ends where it
   crosses the outline. The map shows no busbars, links or transformer symbols inside the
   box (``station_levels`` and ``station_transformers`` stay empty);
3. moves junctions (tees) onto the nearest tower line.

The inside of a station (voltage levels, busbars, transformers, circuits with their far
ends, plants, data-quality flags) is in the click popup (:func:`station_details`) and its
pypowsybl single-line diagram (:func:`write_station_diagrams`).

Only geometries change; every circuit keeps at least one row, the model is not touched.
A station table (``site`` + polygon ``geometry``) can replace the OpenStreetMap matching
by passing it as ``footprints`` to :func:`layout_stations`.
"""

from __future__ import annotations

import html
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tranpy_core._optional import require

__all__ = [
    "BOX_M",
    "END_TOUCH_M",
    "FOOTPRINT_MATCH_M",
    "empty_station_layers",
    "layout_stations",
    "read_osm_footprints",
    "station_details",
    "station_footprints",
    "write_station_details",
    "write_station_diagrams",
]

CRS = "EPSG:4326"
METRIC_CRS = "EPSG:3035"
FOOTPRINT_MATCH_M = 300.0
"""OpenStreetMap substation polygons (>= 220 kV) this close to a station form its footprint."""
FOOTPRINT_MAX_M = 2000.0
"""A matched footprint wider than this is rejected (two stations merged); a box is used."""
BOX_M = (240.0, 70.0)
"""Fixed box (width, height per voltage level) for stations without a footprint."""
FALLBACK_SPAN_M = 600.0
"""A box at circuit ends covers ends at most this far apart."""
TAIL_M = 400.0
"""The last piece of a circuit into a station up to this long keeps its neighbour's slot."""
END_TOUCH_M = 15.0
"""A circuit end this close to a footprint ends at that footprint."""
FALLBACK_M = 1500.0
"""Circuit ends this close to a station centre a box when they miss its footprint."""
JUNCTION_SNAP_M = 200.0
"""A junction (tee) is moved onto a tower line this close."""
_TEE_KINDS = ("tee", "junction", "busbar")


def read_osm_footprints(path: str | Path) -> Any:
    """OpenStreetMap substation polygons carrying 220 or 380 kV (EPSG:4326).

    ``path`` is the OpenStreetMap GeoPackage (layer ``power_substation_polygon``), a
    parquet / GeoPackage with ``osm_id``, ``name``, ``voltages`` and polygons, or a station
    table (GeoPackage layer ``stations`` or parquet with ``station``/``site`` and polygon
    footprints; its rows without a polygon are left out).
    """
    gpd = require("geopandas", "grid")
    path = Path(path)
    if path.suffix == ".parquet":
        frame = gpd.read_parquet(path)
    else:
        pyogrio = require("pyogrio", "grid")
        layers = {str(n) for n, _ in pyogrio.list_layers(path)}
        layer = next((n for n in ("stations", "power_substation_polygon") if n in layers), None)
        frame = pyogrio.read_dataframe(path, layer=layer)
    if "station" in frame and "site" not in frame:
        frame = frame.rename(columns={"station": "site"})
    if "site" in frame:  # a station table: its footprint polygons by site
        keep = frame.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        cols = ["site", *[c for c in ("osm_ids",) if c in frame], "geometry"]
        return frame.loc[keep, cols].reset_index(drop=True).to_crs(CRS)
    frame = frame.rename(columns={"id": "osm_id"})
    volts = frame.get("voltages", pd.Series("", index=frame.index)).astype(str)
    top = pd.to_numeric(
        frame.get("max_voltage", pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    hv = (top >= 200_000) | volts.str.contains("220000|380000")
    out = frame.loc[hv, ["osm_id", "name", "voltages", "geometry"]].copy()
    out = out[out.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    return gpd.GeoDataFrame(out.reset_index(drop=True), geometry="geometry", crs=frame.crs).to_crs(
        CRS
    )


def _box(x: float, y: float, levels: int) -> Any:
    geom = require("shapely.geometry", "grid")
    w, h = BOX_M[0] / 2.0, BOX_M[1] * (max(levels, 1) + 1) / 2.0
    return geom.box(x - w, y - h, x + w, y + h)


def station_footprints(substations: Any, osm: Any | None = None) -> Any:
    """Footprint polygon per station (``site``; tees and junctions have none).

    Every OpenStreetMap polygon is given to its nearest station within
    :data:`FOOTPRINT_MATCH_M`; a station's polygons are merged (convex hull when they are
    apart, e.g. separate 380 and 220 kV yards). A station table (``site`` column) gives the
    polygons by site instead (``source`` ``station table``), used when the polygon is within
    :data:`FOOTPRINT_MATCH_M` of the station's display point or around it (convex hull,
    separate yards). Stations without a polygon,
    or whose merged footprint is wider than :data:`FOOTPRINT_MAX_M`, get a box (``source``
    ``box``).
    """
    gpd = require("geopandas", "grid")
    shapely = require("shapely", "grid")
    subs = substations[~substations["kind"].isin(_TEE_KINDS)].copy()
    subs = subs[subs.geometry.notna() & ~subs.geometry.is_empty]
    pts = subs.to_crs(METRIC_CRS).geometry.reset_index(drop=True)
    levels = [
        len(str(v).split("/")) if isinstance(v, str) and v else 1 for v in subs["voltages_kv"]
    ]
    owned: dict[int, list[int]] = {}
    table = osm is not None and "site" in osm
    found = "station table" if table else "osm"
    if table and len(osm):
        osm = osm.reset_index(drop=True)
        polys = osm.to_crs(METRIC_CRS).geometry
        rows_of = osm.groupby(osm["site"].astype(str)).indices
        for i, (site, pt) in enumerate(zip(subs["site"].astype(str), pts, strict=True)):
            # a station of separate yards (e.g. 380 and 220 kV) may have its display point
            # (the polygons' centroid) between them, farther than FOOTPRINT_MATCH_M from
            # each yard: a point inside the convex hull of its own polygons also matches
            near = [
                int(j)
                for j in rows_of.get(site, [])
                if polys.iat[j].distance(pt) <= FOOTPRINT_MATCH_M
                or polys.iat[j].convex_hull.contains(pt)
            ]
            if near:
                owned[i] = near
    elif osm is not None and len(osm):
        polys = osm.to_crs(METRIC_CRS).geometry.reset_index(drop=True)
        tree = shapely.STRtree(pts.to_numpy())
        for j, poly in enumerate(polys):
            hit = tree.query(poly, predicate="dwithin", distance=FOOTPRINT_MATCH_M)
            if len(hit):
                d = shapely.distance(pts.to_numpy()[hit], poly)
                owned.setdefault(int(hit[int(np.argmin(d))]), []).append(j)
    rows, shapes = [], []
    for i, (site, name, pt) in enumerate(zip(subs["site"], subs["name"], pts, strict=True)):
        shape, source, ids = None, "box", ""
        if i in owned:
            parts = [polys.iat[j] for j in owned[i]]
            merged = shapely.union_all(parts)
            if merged.geom_type != "Polygon":
                merged = merged.convex_hull
            minx, miny, maxx, maxy = merged.bounds
            if max(maxx - minx, maxy - miny) <= FOOTPRINT_MAX_M:
                shape, source = merged, found
                col = "osm_ids" if table else "osm_id"
                if col in osm:
                    ids = ";".join(str(osm[col].iat[j]) for j in owned[i])
        if shape is None:
            shape = _box(pt.x, pt.y, levels[i])
        rows.append({"site": site, "name": name, "source": source, "osm_ids": ids})
        shapes.append(shape)
    out = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=shapes, crs=METRIC_CRS)
    return out.to_crs(CRS)


def _levels(value: Any) -> list[float]:
    if not isinstance(value, str) or not value:
        return []
    return sorted({float(v) for v in value.split("/") if v}, reverse=True)


def _coords(shape: Any) -> list[tuple[float, float]]:
    """All vertex coordinates of a (multi-part) geometry."""
    if shape is None or shape.is_empty:
        return []
    if hasattr(shape, "geoms"):
        return [c for part in shape.geoms for c in _coords(part)]
    return [(float(x), float(y)) for x, y, *_ in shape.coords]


def _fallback_boxes(
    lines_m: Any, lines: Any, bus_site: Mapping[str, str], poly_of: dict[str, Any], subs: Any
) -> dict[str, Any]:
    """Boxes for stations whose footprint no circuit end reaches (centred at the ends).

    Per circuit the end nearest to the station point is taken; if none of a station's
    circuit ends lies within :data:`END_TOUCH_M` of its footprint and they are within
    :data:`FALLBACK_M` of the station point, the footprint becomes a box around their mean.
    """
    geom = require("shapely.geometry", "grid")
    pts = subs.to_crs(METRIC_CRS)
    point_of = dict(zip(pts["site"].astype(str), pts.geometry, strict=True))
    levels_of = {
        str(s): len(_levels(v)) for s, v in zip(subs["site"], subs["voltages_kv"], strict=True)
    }
    nearest: dict[tuple[str, str], tuple[float, tuple[float, float]]] = {}
    for g, b1, b2, mrid in zip(
        lines_m.geometry, lines["bus1"], lines["bus2"], lines["mrid"], strict=True
    ):
        if g is None or g.is_empty or g.geom_type != "LineString":
            continue
        for site in {bus_site.get(b1, ""), bus_site.get(b2, "")} & set(poly_of):
            p = point_of.get(site)
            if p is None:
                continue
            for end in (g.coords[0], g.coords[-1]):
                d = p.distance(geom.Point(end))
                key = (str(mrid), site)
                if key not in nearest or d < nearest[key][0]:
                    nearest[key] = (d, (end[0], end[1]))
    by_site: dict[str, list[tuple[float, tuple[float, float]]]] = {}
    for (_, site), v in nearest.items():
        by_site.setdefault(site, []).append(v)
    out = {}
    for site, items in by_site.items():
        poly = poly_of[site]
        if any(poly.distance(geom.Point(e)) <= END_TOUCH_M for _, e in items):
            continue
        near = [e for d, e in items if d <= FALLBACK_M]
        if not near:
            continue
        # ends spread wider than FALLBACK_SPAN_M: the box takes the largest group of them
        groups = [[e for e in near if math.dist(e, f) <= FALLBACK_SPAN_M] for f in near]
        near = max(groups, key=len)
        x = float(np.mean([e[0] for e in near]))
        y = float(np.mean([e[1] for e in near]))
        box = _box(x, y, levels_of.get(site, 1))
        spread = geom.MultiPoint(near).buffer(END_TOUCH_M).envelope
        out[site] = box.union(spread).envelope
    return out


def _continue_offsets(rows: list[dict[str, Any]], shapes: list[Any]) -> None:
    """Keep parallel circuits side by side through short pieces (in place).

    A piece not on a tower line (gap between two OpenStreetMap ways) whose neighbouring
    pieces of the same circuit both sit in the same slot of a multi-circuit tower line,
    and the last piece into a station (shorter than :data:`TAIL_M`), take that slot
    (``circuits``, ``circuit_no``), so the drawing does not jump sideways there.
    """
    if not rows or "circuits" not in rows[0] or "circuit_no" not in rows[0]:
        return
    by_mrid: dict[Any, list[int]] = {}
    for k, rec in enumerate(rows):
        by_mrid.setdefault(rec.get("mrid"), []).append(k)

    def slot(k: int) -> tuple[int, int]:
        return int(rows[k].get("circuits") or 1), int(rows[k].get("circuit_no") or 0)

    for _ in range(2):
        for members in by_mrid.values():
            if len(members) < 2:
                continue
            for k in members:
                g = shapes[k]
                if g is None or g.is_empty or g.geom_type != "LineString":
                    continue
                gap = not str(rows[k].get("tower_way") or "")
                tail = bool(rows[k].get("station_end")) and g.length < TAIL_M
                if not gap and not tail:
                    continue
                if slot(k)[0] >= 2:
                    continue
                found = []
                for end in (g.coords[0], g.coords[-1]):
                    near = [
                        j
                        for j in members
                        if j != k
                        and shapes[j] is not None
                        and shapes[j].geom_type == "LineString"
                        and min(
                            math.dist(end, shapes[j].coords[0]),
                            math.dist(end, shapes[j].coords[-1]),
                        )
                        < 1.0
                        and slot(j)[0] >= 2
                    ]
                    found.append({slot(j) for j in near})
                both = found[0] & found[1]
                pick = None
                if gap and len(both) == 1:
                    pick = next(iter(both))
                elif tail and len(found[0] | found[1]) == 1:
                    pick = next(iter(found[0] | found[1]))
                if pick is not None:
                    rows[k] = {**rows[k], "circuits": pick[0], "circuit_no": pick[1]}


def layout_stations(layers: Mapping[str, Any], footprints: Any) -> dict[str, Any]:
    """Lay out stations in the map layers (display only); returns the changed/new layers.

    Args:
        layers: output of :func:`tranpy.grid_gis.build_layers` (``lines``, ``substations``,
            ``transformers``, ``buses``, ``tower_lines``).
        footprints: :func:`station_footprints` (or a station table with ``site`` and a
            polygon ``geometry``).

    Returns:
        ``lines`` (clipped at footprints, ending on the outline; ``station_end`` added),
        ``substations`` (junctions on their tower line, ``footprint`` added),
        ``station_footprints`` (one box per station, with ``max_kv`` for its outline) and
        ``station_levels`` / ``station_transformers`` (kept for a stable layer list, empty:
        the station inside is in the popup, :func:`station_details`).
    """
    gpd = require("geopandas", "grid")
    shapely = require("shapely", "grid")
    geom = require("shapely.geometry", "grid")
    ops = require("shapely.ops", "grid")

    lines = layers["lines"].reset_index(drop=True)
    subs = layers["substations"].copy()
    buses = layers["buses"]
    bus_site = dict(zip(buses["bus"], buses["site"].astype(str), strict=True))
    fp = footprints.to_crs(METRIC_CRS).reset_index(drop=True)
    fp = fp[fp["site"].astype(str).isin(set(subs["site"].astype(str)))].reset_index(drop=True)
    if "source" not in fp:
        fp["source"] = "station table"
    poly_of = dict(zip(fp["site"].astype(str), fp.geometry, strict=True))
    lines_m = lines.to_crs(METRIC_CRS)

    # footprints no circuit end reaches: a box centred at the circuit ends
    boxes = _fallback_boxes(lines_m, lines, bus_site, poly_of, subs)
    if boxes:
        fp = fp.copy()
        for k, site in enumerate(fp["site"].astype(str)):
            if site in boxes:
                fp.at[k, "geometry"] = boxes[site]
                fp.at[k, "source"] = "box at circuit ends"
        poly_of.update(boxes)

    # lines: clip at the footprints of their end stations
    geoms: dict[int, Any] = {}  # rows whose geometry is replaced (pieces inside a footprint)
    ends: list[dict[str, Any]] = []  # one per (row, side) that ends in a footprint
    cut: dict[int, list[float]] = {}
    inside_rows: dict[int, list[str]] = {}
    for i, (g, b1, b2) in enumerate(
        zip(lines_m.geometry, lines["bus1"], lines["bus2"], strict=True)
    ):
        if g is None or g.is_empty or g.geom_type != "LineString":
            continue
        sites = sorted({bus_site.get(b1, ""), bus_site.get(b2, "")} & set(poly_of))
        if not sites:
            continue
        length = g.length
        lo, hi = 0.0, length
        coords = list(g.coords)
        mine = []
        for side, end in ((0, coords[0]), (1, coords[-1])):
            for site in sites:
                poly = poly_of[site]
                if poly.distance(geom.Point(end)) > END_TOUCH_M:
                    continue
                if poly.buffer(END_TOUCH_M).contains(g):
                    inside_rows[i] = [
                        s for s in sites if poly_of[s].buffer(END_TOUCH_M).contains(g)
                    ]
                    break
                ring = poly.exterior
                ds = sorted(g.project(geom.Point(c)) for c in _coords(g.intersection(ring)))
                # no crossing: the circuit ends just outside the footprint
                d = (ds[0] if side == 0 else ds[-1]) if ds else (0.0 if side == 0 else length)
                if side == 0:
                    lo = max(lo, d)
                else:
                    hi = min(hi, d)
                kv = float(lines["voltage_kv"].iat[i])
                mine.append({"row": i, "side": side, "site": site, "kv": kv})
                break
        if i in inside_rows or hi - lo < 1.0:
            inside_rows.setdefault(i, [mine[0]["site"] if mine else sites[0]])
            continue
        ends.extend(mine)
        cut[i] = [lo, hi]

    # a circuit whose only piece at a station lies inside its footprint (a short spur
    # from a nearby junction, a link between two yards): drawn from its outer end to the
    # nearest point of the footprint edge
    reached = {(str(lines["mrid"].iat[e["row"]]), e["site"]) for e in ends}
    longest: dict[tuple[str, str], int] = {}
    for i, sites_in in inside_rows.items():
        for site in sites_in:
            key = (str(lines["mrid"].iat[i]), site)
            if key in reached:
                continue
            if (
                key not in longest
                or lines_m.geometry.iat[i].length > lines_m.geometry.iat[longest[key]].length
            ):
                longest[key] = i
    needed: dict[int, list[str]] = {}
    for (_, site), i in longest.items():
        needed.setdefault(i, []).append(site)
    for i, sites_in in needed.items():
        g = lines_m.geometry.iat[i]
        kv = float(lines["voltage_kv"].iat[i])
        if len(sites_in) >= 2:  # a link between two (overlapping) footprints
            s1, s2 = sites_in[:2]
            r1, r2 = poly_of[s1].exterior, poly_of[s2].exterior
            p1 = r1.interpolate(r1.project(poly_of[s2].centroid))
            p2 = r2.interpolate(r2.project(poly_of[s1].centroid))
            if p1.distance(p2) >= 0.5:
                geoms[i] = geom.LineString([(p1.x, p1.y), (p2.x, p2.y)])
                cut[i] = [0.0, geoms[i].length]
                ends.append({"row": i, "side": 0, "site": s1, "kv": kv})
                ends.append({"row": i, "side": 1, "site": s2, "kv": kv})
                del inside_rows[i]
                continue
        site = sites_in[0]
        poly = poly_of[site]
        c = poly.centroid
        a, b = geom.Point(g.coords[0]), geom.Point(g.coords[-1])
        far = a if a.distance(c) >= b.distance(c) else b
        port = poly.exterior.interpolate(poly.exterior.project(far))
        if far.distance(port) < 0.5:
            far = b if far is a else a
        if far.distance(port) < 0.5:
            continue
        geoms[i] = geom.LineString([(far.x, far.y), (port.x, port.y)])
        cut[i] = [0.0, geoms[i].length]
        ends.append({"row": i, "side": 1, "site": site, "kv": kv})
        del inside_rows[i]

    def geom_of(i: int) -> Any:
        return geoms.get(i, lines_m.geometry.iat[i])

    # where each end crosses the footprint outline
    for e in ends:
        g = geom_of(e["row"])
        lo, hi = cut[e["row"]]
        ring = poly_of[e["site"]].exterior
        x = g.interpolate(lo if e["side"] == 0 else hi)
        e["cross"] = ring.interpolate(ring.project(x))

    # no ports, bars or stubs: each circuit ends where it crosses the footprint outline;
    # the station's inside (levels, transformers) is in the click popup
    # (:func:`station_details`) and its single-line diagram
    for e in ends:
        e["port"] = e["cross"]
    empty = empty_station_layers()
    station_levels = empty["station_levels"]
    station_transformers = empty["station_transformers"]

    # pieces inside a footprint are dropped, but every circuit keeps at least one row
    drop = set(inside_rows)
    left = set(range(len(lines))) - drop
    left_mrids = set(lines["mrid"].iloc[sorted(left)])
    for i in sorted(drop):
        if lines["mrid"].iat[i] not in left_mrids:
            drop.discard(i)
            left_mrids.add(lines["mrid"].iat[i])

    port_of = {(e["row"], e["side"]): e for e in ends}
    rows: list[dict[str, Any]] = []
    shapes: list[Any] = []
    records = lines.drop(columns="geometry").to_dict("records")
    for i, rec in enumerate(records):
        if i in drop:
            continue
        g = geom_of(i)
        if i not in cut:
            rows.append({**rec, "station_end": ""})
            shapes.append(g)
            continue
        lo, hi = cut[i]
        core = ops.substring(g, lo, hi)
        if core.geom_type != "LineString" or core.length == 0:
            rows.append({**rec, "station_end": ""})
            shapes.append(g)
            continue
        coords = list(core.coords)
        names = []
        for side in (0, 1):
            e = port_of.get((i, side))
            if e is None:
                continue
            names.append(e["site"])
            p = (e["port"].x, e["port"].y)
            # the cut point is replaced by the port
            coords = [p, *coords[1:]] if side == 0 else [*coords[:-1], p]
        coords = [c for k, c in enumerate(coords) if k == 0 or c != coords[k - 1]]
        if len(coords) < 2:
            coords = list(core.coords)
        rows.append({**rec, "station_end": ";".join(names)})
        shapes.append(geom.LineString(coords))
    _continue_offsets(rows, shapes)
    out_lines = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=shapes, crs=METRIC_CRS).to_crs(CRS)
    out_lines = out_lines[
        [*[c for c in lines.columns if c != "geometry"], "station_end", "geometry"]
    ]

    # junctions on their tower line
    towers = layers.get("tower_lines")
    tee = subs["kind"].isin(_TEE_KINDS)
    subs["footprint"] = (
        subs["site"]
        .astype(str)
        .map(dict(zip(fp["site"].astype(str), fp["source"], strict=True)))
        .fillna("")
    )
    subs["junction_snap_m"] = 0.0
    if towers is not None and len(towers) and tee.any():
        ways = towers.to_crs(METRIC_CRS).geometry.to_numpy()
        tree = shapely.STRtree(ways)
        pts = subs.loc[tee].to_crs(METRIC_CRS).geometry
        moved = []
        for idx, p in pts.items():
            near = tree.query(p, predicate="dwithin", distance=JUNCTION_SNAP_M)
            if not len(near):
                moved.append((idx, p, 0.0))
                continue
            w = ways[near[int(np.argmin(shapely.distance(ways[near], p)))]]
            q = w.interpolate(w.project(p))
            moved.append((idx, q, round(p.distance(q))))
        snapped = gpd.GeoSeries([m[1] for m in moved], crs=METRIC_CRS).to_crs(CRS)
        for (idx, _, d), q in zip(moved, snapped, strict=True):
            subs.at[idx, "geometry"] = q
            subs.at[idx, "junction_snap_m"] = d
    footprints_out = fp.to_crs(CRS).merge(
        subs[["site", "kind", "symbol", "max_kv", "voltages_kv", "transformers", "tso"]],
        on="site",
        how="left",
    )
    return {
        "lines": out_lines,
        "substations": subs,
        "station_footprints": gpd.GeoDataFrame(footprints_out, geometry="geometry", crs=CRS),
        "station_levels": station_levels,
        "station_transformers": station_transformers,
    }


def empty_station_layers() -> dict[str, Any]:
    """``station_*`` layers without features (export without station footprints)."""
    gpd = require("geopandas", "grid")

    def empty(cols: list[str]) -> Any:
        return gpd.GeoDataFrame(
            pd.DataFrame({c: pd.Series(dtype=object) for c in cols}), geometry=[], crs=CRS
        )

    return {
        "station_footprints": empty(["site", "name", "source", "osm_ids"]),
        "station_levels": empty(["site", "voltage_kv", "part", "mrid"]),
        "station_transformers": empty(["name", "site", "kv_hv", "kv_lv", "slot"]),
    }


def _num(v: Any, nd: int = 1) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, nd)


def _text(v: Any) -> str:
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)


def _station_name(names: Any, site: str) -> str:
    """Station name from the joined model names, without the transformer names (``TR …``)."""
    parts = [p.strip() for p in _text(names).split(";") if p.strip()]
    plain = [p for p in parts if not p.startswith("TR ")]
    return (plain or parts or [site])[0]


def station_details(
    layers: Mapping[str, Any], stations_dir: str | Path | None = None
) -> dict[str, dict[str, Any]]:
    """Content of the station click popup of the web map, per site (display only).

    Args:
        layers: output of :func:`tranpy.grid_gis.build_layers` (``substations``, ``buses``,
            ``transformers``, ``lines``, optionally ``plants``); ``substations.diagram``
            (single-line diagram file) is passed through when present.
        stations_dir: the station table folder (``stations.parquet``, ``checks.csv``):
            adds the OpenStreetMap operator and names and the data-quality flags.

    Returns:
        ``{site: {name, model_names, kind, tso, operator, osm_names, voltages_kv, footprint,
        diagram, nodes, transformers, circuits, plants, plants_mw, flags}}``. ``nodes`` are the
        model buses (busbars per voltage level), ``circuits`` one row per model circuit
        with its far-end station, ``plants`` the plants in operation at the site's buses
        (largest first), ``flags`` the station-table checks of the site or its nodes.
    """
    subs = layers["substations"]
    buses = layers["buses"]
    bus_site = dict(zip(buses["bus"].astype(str), buses["site"].astype(str), strict=True))
    site_name = {
        str(s): _station_name(n, str(s)) for s, n in zip(subs["site"], subs["name"], strict=True)
    }
    out: dict[str, dict[str, Any]] = {}
    for row in subs.drop(columns="geometry").to_dict("records"):
        site = str(row["site"])
        out[site] = {
            "name": site_name[site],
            "model_names": _text(row.get("name")),
            "kind": _text(row.get("kind")),
            "tso": _text(row.get("tso")),
            "operator": "",
            "osm_names": "",
            "voltages_kv": _text(row.get("voltages_kv")),
            "footprint": _text(row.get("footprint")),
            "diagram": _text(row.get("diagram")),
            "nodes": [],
            "transformers": [],
            "circuits": [],
            "plants": [],
            "plants_mw": 0.0,
            "flags": [],
        }
    for b in buses.drop(columns="geometry").to_dict("records"):
        d = out.get(str(b["site"]))
        if d is not None:
            d["nodes"].append(
                {
                    "node": str(b["bus"]),
                    "kv": _num(b.get("voltage_kv"), 0),
                    "kind": _text(b.get("kind")),
                    "mrid": _text(b.get("bus_mrid")),
                }
            )
    for t in layers["transformers"].drop(columns="geometry").to_dict("records"):
        d = out.get(str(t["site"]))
        if d is not None:
            d["transformers"].append(
                {
                    "name": _text(t.get("name")),
                    "kv_hv": _num(t.get("kv_hv"), 0),
                    "kv_lv": _num(t.get("kv_lv"), 0),
                    "sn_mva": _num(t.get("sn_mva"), 0),
                    "x_ohm": _num(t.get("x_ohm"), 2),
                    "phase_shifter": bool(t.get("phase_shifter")),
                }
            )
    lines = layers["lines"].drop(columns="geometry").drop_duplicates("mrid")
    for c in lines.to_dict("records"):
        ends = [bus_site.get(str(c.get(k)), "") for k in ("bus1", "bus2")]
        for k, site in enumerate(ends):
            d = out.get(site)
            if d is None:
                continue
            far = ends[1 - k]
            d["circuits"].append(
                {
                    "name": _text(c.get("name")),
                    "kv": _num(c.get("voltage_kv"), 0),
                    "far_end": site_name.get(far, far) if far != site else "(inside station)",
                    "far_site": far,
                    "length_km": _num(c.get("length_km"), 1),
                    "rating_mva": _num(c.get("rating_mva"), 0),
                    "status": _text(c.get("status")),
                    "hvdc": bool(c.get("hvdc")),
                }
            )
    plants = layers.get("plants")
    if plants is not None and len(plants) and "bus" in plants:
        live = plants["in_operation"].astype(bool) if "in_operation" in plants else True
        for p in plants[live].drop(columns="geometry").to_dict("records"):
            d = out.get(bus_site.get(str(p.get("bus")), ""))
            if d is None:
                continue
            mw = _num(p.get("capacity_mw"), 1) or 0.0
            d["plants"].append(
                {
                    "name": _text(p.get("name")),
                    "fuel": _text(p.get("fuel_category")) or _text(p.get("fuel")),
                    "mw": mw,
                    "node": _text(p.get("bus")),
                }
            )
            d["plants_mw"] = round(d["plants_mw"] + mw, 1)
    for d in out.values():
        d["nodes"].sort(key=lambda n: -(n["kv"] or 0))
        d["circuits"].sort(key=lambda c: (-(c["kv"] or 0), c["far_end"], c["name"]))
        d["plants"].sort(key=lambda p: -p["mw"])
    if stations_dir is not None:
        folder = Path(stations_dir)
        table = folder / "stations.parquet"
        if table.exists():
            st = pd.read_parquet(table, columns=["station", "osm_operator", "osm_names"])
            for r in st.to_dict("records"):
                d = out.get(str(r["station"]))
                if d is not None:
                    d["operator"] = _text(r["osm_operator"])
                    d["osm_names"] = _text(r["osm_names"])
        checks = folder / "checks.csv"
        if checks.exists():
            for r in pd.read_csv(checks, dtype=str).fillna("").to_dict("records"):
                d = out.get(r["object"].split("|")[0])
                if d is not None:
                    d["flags"].append({k: r[k] for k in ("check", "object", "severity", "detail")})
    return out


def write_station_details(
    layers: Mapping[str, Any], path: str | Path, stations_dir: str | Path | None = None
) -> Path:
    """Write :func:`station_details` as JSON (``station_details.json`` of the web map)."""
    import json

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            station_details(layers, stations_dir), ensure_ascii=False, separators=(",", ":")
        ),
        encoding="utf-8",
    )
    return out


def write_station_diagrams(cim: Any, out_dir: str | Path) -> dict[str, str]:
    """Single-line diagram (SVG) per substation from the pypowsybl network of the model.

    Uses :func:`tranpy.powsybl_model.network_from_cim_tables` (the network of the
    second-solver checks) with readable names (circuit and transformer names, ``site kV``
    for buses, fuel and MW for plants, MW for loads), and pypowsybl's single-line diagram
    (``Convergence`` symbols). Files are ``out_dir/<substation mrid>.svg``.

    Returns:
        ``{site: file name}`` of the diagrams written.
    """
    from tranpy.powsybl_model import _ids, network_from_cim_tables

    pp = require("pypowsybl", "powsybl")
    network = network_from_cim_tables(cim)
    lines = set(network.get_lines(all_attributes=False).index)
    seg = cim.ac_line_segment
    pairs = [
        (i, str(n)) for i, n in zip(_ids("L", seg["mrid"]), seg["name"], strict=True) if i in lines
    ]
    network.update_lines(id=[a for a, _ in pairs], name=[b for _, b in pairs])
    tr = cim.power_transformer
    present = set(network.get_2_windings_transformers(all_attributes=False).index)
    pairs = [
        (i, str(n)) for i, n in zip(_ids("T", tr["mrid"]), tr["name"], strict=True) if i in present
    ]
    network.update_2_windings_transformers(id=[a for a, _ in pairs], name=[b for _, b in pairs])
    vl = cim.voltage_level
    network.update_voltage_levels(
        id=_ids("VL", vl["mrid"]),
        name=[f"{s} {kv:g} kV" for s, kv in zip(vl["site"], vl["nominal_voltage_kv"], strict=True)],
    )
    # buses, plants, loads and external grids have no name in the network: their labels
    # in the SVG are replaced by readable ones
    labels: dict[str, str] = {}
    bb = cim.busbar_section
    for i, n in zip(_ids("B", bb["mrid"]), bb["name"], strict=True):
        node = str(n)
        labels[i] = node.replace("|", " ") + " kV" if "|" in node else node
    sm = cim.synchronous_machine
    for i, fuel, mw in zip(_ids("G", sm["mrid"]), sm["fuel"], sm["max_p_mw"], strict=True):
        labels[i] = f"{fuel} {float(mw):.0f} MW"
    ec = cim.energy_consumer
    for i, mw in zip(_ids("C", ec["mrid"]), ec["p_mw"], strict=True):
        labels[i] = f"load {float(mw):.0f} MW"
    ext = cim.external_network_injection
    for i, node in zip(_ids("X", ext["mrid"]), ext["node"], strict=True):
        labels[i] = f"external grid {node}"
    pattern = re.compile(r">([A-Z]{1,2}_[0-9a-f]{32})<")

    def readable(path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        new = pattern.sub(lambda m: ">" + html.escape(labels.get(m[1], m[1])) + "<", text)
        if new != text:
            path.write_text(new, encoding="utf-8")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params = pp.network.SldParameters(use_name=True, component_library="Convergence")
    subs = cim.substation
    written: dict[str, str] = {}
    for sid, mrid, site in zip(_ids("S", subs["mrid"]), subs["mrid"], subs["site"], strict=True):
        name = f"{mrid}.svg"
        try:
            network.write_single_line_diagram_svg(sid, str(out / name), parameters=params)
        except Exception:  # a station pypowsybl cannot draw gets no link
            continue
        readable(out / name)
        written[str(site)] = name
    return written
