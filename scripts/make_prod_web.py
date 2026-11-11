"""Write a clean production copy of a staging web build of the grid map.

    python scripts/make_prod_web.py STAGING_WEB_DIR PROD_DIR [--app APP_DIR] [--styles QML_DIR]

STAGING_WEB_DIR is an export folder as served on 8094 (``*.geojson``, ``sld/``,
``station_details.json``, ``metadata.json``). PROD_DIR gets

* ``data/``: the same layers with every debugging / data-quality field removed
  (see ``DROP``), the same features and geometries, ``sld/`` unchanged,
  ``station_details.json`` without the data-quality flags and a ``metadata.json``
  restricted to ``META_KEEP`` plus ``mode: production``;
* the app files (``app.js``, ``index.html``, ``style.css``, ``style.js``, ``symbols/``)
  from ``--app`` (default: ``apps/grid_map`` of this checkout); app.js reads ``mode`` and
  hides the halo, check legend, flags and review notes (one codebase, two profiles);
* ``qgis/``: ``grid_model.gpkg`` written from the clean layers, the QGIS styles, the
  README, and ``data/grid_model_qgis.zip`` (the source of the QWC 8088 rebuild).

Production is model only: the ``REMOVED`` layers (raw OSM, OSM way sections, offshore
farm markers and links) are left out, every circuit is one connected line from station to
station (``connect_circuits``: a broken drawn route becomes a straight station-to-station
line in the same style; list in ``build_report/fallback_circuits.csv``), offshore farm names
and MW go to the landing bus, and ``--model`` adds the model's machines to the station panel.

The staging folder is only read. Model data, sources and attribution are kept.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# fields removed per layer (debug, review and check information); everything else is kept
DROP: dict[str, set[str]] = {
    "lines": {
        "route_note",
        "route_status",
        "end_gap_m",
        "display_snapped_m",
        "edge_id",
        "station_cut",
        "geometry_source",
    },
    "osm_lines": {"check", "check_reason", "model_n", "model_circuits"},
    "tower_lines": {"note"},
    "offshore_farms": {"check"},
    "offshore_links": {"sourced_in"},
    "border_terminals": {"port_spread_m"},
    "substations": {"display_relocated_m", "junction_snap_m", "location_flag", "geometry_source"},
    "buses": {"display_relocated_m", "location_flag", "geometry_source"},
    "station_footprints": {"grown_m", "footprint_wkt"},
    "renewable_inverters": {"offshore_spread_note", "offshore_spread_mw"},
}
# layers left out of production: production shows only what the simulation model uses
# (layer -> reason; scripts/prod_acceptance.py fails if one of them is present)
REMOVED: dict[str, str] = {
    "osm_lines": "OSM 220/380 kV way sections with the model check: OSM context and review aid, "
    "not a model element (the circuits are drawn from the model's lines layer)",
    "tower_lines": "raw OpenStreetMap power lines of all voltages ('Assets (OpenStreetMap)')",
    "osm_substations": "raw OpenStreetMap substation polygons ('Assets (OpenStreetMap)')",
}
# offshore farm fields kept in production (no review notes, spread notes or open questions)
FARM_KEEP = {
    "name",
    "sea",
    "capacity_mw",
    "capacity_source",
    "turbines",
    "commissioned",
    "status",
    "cable",
    "landing",
    "cable_source",
    "farm_source",
}
# extra line fields dropped in production (route qualification is a staging matter)
LINE_ROUTE_FIELDS = {
    "route_quality",
    "route_source",
    "route_length_km",
    "piece_kind",
    "station_end",
}
# connected-route test (metric, EPSG:3035): pieces join when ends are this close, or when
# both ends lie on the same drawn station shape (a pass through a station box); a circuit
# end must lie this close to its own station (shape, else point) or its border port
JOIN_M = 20.0
END_TOL_M = 25.0
ON_SHAPE_M = 3.0
DROP["lines"] |= LINE_ROUTE_FIELDS
# metadata keys kept (allow list: diagnostic keys such as inside_station_only,
# border_cut_ties, edge_review, network, geometry_overlay, renewables_c14b are left out)
META_KEEP = [
    "model_version",
    "total_load_mw",
    "load_source",
    "power_factor",
    "dispatch",
    "hvdc_left_out",
    "sources",
    "build_id",
    "source_file",
    "source_sha256",
    "exported_utc",
    "crs",
    "border_terminals",
    "renewables",
]
# farms the model spreads over the landing substations of their sea: neutral wording
SPREAD_NOTE = "connected in the model via the regional landing substations"
# text that must never reach production (checked here and by scripts/prod_acceptance.py)
BANNED = re.compile(
    r"needs owner|not sourced|v6 to fix|(?<!-)\bC1[0-9][ab]?\b|halo|mismatch|override:", re.I
)
APP_FILES = ["app.js", "index.html", "style.css", "style.js", "symbols"]
# QWC / QGIS layers (apps/grid_map/qwc/build_project.py LAYERS)
GPKG_LAYERS = [
    "neighbours",
    "plants",
    "plant_aggregates",
    "station_transformers",
    "transformers",
    "substations",
    "buses",
    "plant_connections",
    "station_levels",
    "lines",
    "station_footprints",
    "control_areas",
    "bidding_zones",
]


def clean_feature(layer: str, p: dict) -> dict:
    p = {k: v for k, v in p.items() if k not in DROP.get(layer, ())}
    if layer == "osm_lines":
        pass  # in_model is set by the caller from model_n before the drop
    elif layer == "offshore_farms":
        # neutral farm popup: name, MW, turbines, cable, landing substation and sources only
        p = {k: v for k, v in p.items() if k in FARM_KEEP}
    elif layer == "offshore_links":
        if p.get("note"):
            p["note"] = SPREAD_NOTE
    elif layer == "renewable_inverters":
        if p.get("offshore_spread_note"):
            mw = p.get("offshore_spread_mw") or 0
            p["offshore_spread_note"] = (
                f"includes {mw:.0f} MW of offshore wind farms {SPREAD_NOTE}"
                if mw
                else f"includes offshore wind farms {SPREAD_NOTE}"
            )
    return p


def clean_layer(layer: str, fc: dict) -> dict:
    out = []
    for f in fc["features"]:
        p = dict(f["properties"])
        if layer == "osm_lines":
            p["in_model"] = bool((p.get("model_n") or 0) > 0)
        out.append({**f, "properties": clean_feature(layer, p)})
    return {**fc, "features": out}


def banned_hits(obj, where: str = "") -> list[str]:
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            hits += banned_hits(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += banned_hits(v, f"{where}[{i}]")
    elif isinstance(obj, str) and BANNED.search(obj):
        hits.append(f"{where}: {obj[:120]}")
    return hits


def write_gpkg(data: Path, gpkg: Path) -> None:
    import geopandas as gpd

    gpkg.unlink(missing_ok=True)
    for name in GPKG_LAYERS:
        src = data / f"{name}.geojson"
        if not src.exists():
            continue
        gdf = gpd.read_file(src)
        if gdf.empty:
            continue
        gdf.to_file(gpkg, layer=name, driver="GPKG")


# ---------------------------------------------------------------- connected circuits


class Stations:
    """Station shapes and points (EPSG:3035) of a web build, for the end and pass tests."""

    def __init__(self, data: Path):
        import shapely
        from pyproj import Transformer

        self.tf = Transformer.from_crs(4326, 3035, always_xy=True)
        self.back = Transformer.from_crs(3035, 4326, always_xy=True)
        ld = lambda n: (  # noqa: E731
            json.loads((data / f"{n}.geojson").read_text())["features"]
            if (data / f"{n}.geojson").exists()
            else []
        )
        self.site_of = {f["properties"]["bus"]: f["properties"]["site"] for f in ld("buses")}
        self.point: dict[str, object] = {}
        for f in ld("substations"):
            self.point[f["properties"]["site"]] = self.geom(f["geometry"])
        for f in ld("neighbours"):  # foreign nodes: keyed by bus and by site
            g = self.geom(f["geometry"])
            self.point.setdefault(f["properties"]["bus"], g)
        self.bus_point = {}
        for f in ld("buses"):
            self.point.setdefault(f["properties"]["site"], self.geom(f["geometry"]))
            self.bus_point[f["properties"]["bus"]] = self.geom(f["geometry"])
        shp = defaultdict(list)
        for f in ld("station_footprints"):
            if f["geometry"]:
                shp[f["properties"]["site"]].append(self.geom(f["geometry"]))
        self.shapes = [shapely.union_all(v) for v in shp.values()]
        self.shape_of = {k: shapely.union_all(v) for k, v in shp.items()}
        self.tree = shapely.STRtree(self.shapes) if self.shapes else None
        # overlapping shapes form one block (as in the staging acceptance)
        self.block_of = {}
        for k, g in self.shape_of.items():
            hit = [self.shapes[int(i)] for i in self.tree.query(g, "intersects")]
            self.block_of[k] = shapely.union_all(hit) if hit else g
        # border ports of the ties cut at the German border: mrid -> port points
        self.ports: dict[str, list] = defaultdict(list)
        for f in ld("border_terminals"):
            p = f["properties"]
            pts = [
                self.geom({"type": "Point", "coordinates": c})
                for c in json.loads(p.get("ports") or "[]")
            ]
            for c in json.loads(p.get("circuits") or "[]"):
                self.ports[c["mrid"]] += pts

    def geom(self, gj):
        from shapely.geometry import shape
        from shapely.ops import transform

        return transform(self.tf.transform, shape(gj))

    def to_wgs(self, g):
        from shapely.ops import transform

        return transform(self.back.transform, g)

    def site(self, bus: str) -> str:
        return self.site_of.get(bus, bus.split("|")[0])

    def end_targets(self, bus: str) -> list:
        """Where a circuit end at ``bus`` may lie: the station block, the station point and
        the bus point (a junction is drawn as a dot)."""
        site = self.site(bus)
        out = [
            self.block_of.get(site),
            self.point.get(site),
            self.point.get(bus),
            self.bus_point.get(bus),
        ]
        out = [g for g in out if g is not None]
        # a junction dot inside (or within 10 m of) a drawn station shape: its legs end on
        # that shape (as in the staging acceptance)
        if site not in self.block_of and self.tree is not None:
            for pt in out[:]:
                out += [self.shapes[int(i)] for i in self.tree.query(pt.buffer(10.0), "intersects")]
        return out


def circuit_problem(st: Stations, pieces: list[dict], graph: dict | None = None) -> str:
    """'' when the circuit's pieces form one simple connected line from station to station.

    The pieces are the edges of a small graph. A piece end on a drawn station shape (within
    ``ON_SHAPE_M``) is that shape's node (the line is cut where it passes a station box);
    other ends within ``JOIN_M`` of each other are one node. The circuit is clean when the
    graph is connected, no node has more than two piece ends (no branch, stub or doubled
    piece) and exactly two nodes are open ends, one on each of its own stations (or the
    tie's border port) within ``END_TOL_M``."""
    from shapely.geometry import Point

    idx, geoms = [], []
    for i, f in enumerate(pieces):
        g = st.geom(f["geometry"]) if f.get("geometry") else None
        if g is not None and g.geom_type == "LineString" and g.length > 0:
            idx.append(i)
            geoms.append(g)
    if not geoms:
        return "no line"
    ends = []
    for g in geoms:
        ends += [Point(g.coords[0]), Point(g.coords[-1])]
    parent = list(range(len(ends)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    shape_node: dict[int, int] = {}
    for k, pt in enumerate(ends):
        if st.tree is None:
            break
        for j in st.tree.query(pt.buffer(ON_SHAPE_M)):
            shp = st.shapes[int(j)]
            if shp.boundary.distance(pt) <= ON_SHAPE_M or shp.contains(pt):
                if int(j) in shape_node:
                    parent[find(k)] = find(shape_node[int(j)])
                else:
                    shape_node[int(j)] = k
                break
    for a in range(len(ends)):
        for b in range(a + 1, len(ends)):
            if ends[a].distance(ends[b]) <= JOIN_M:
                parent[find(a)] = find(b)
    # pieces no longer than the join distance are contracted into one node
    short = [g.length <= JOIN_M for g in geoms]
    if all(short):
        short = [False] * len(geoms)
    for i, sh in enumerate(short):
        if sh:
            parent[find(2 * i)] = find(2 * i + 1)
    node = [find(k) for k in range(len(ends))]
    deg = defaultdict(int)
    for k, n in enumerate(node):
        deg[n] += 0 if short[k // 2] else 1
    # connectivity over the pieces (edge i joins node[2i] and node[2i+1])
    comp = {n: n for n in deg}

    def cf(n):
        while comp[n] != n:
            n = comp[n]
        return n

    for i in range(len(geoms)):
        comp[cf(node[2 * i])] = cf(node[2 * i + 1])
    if len({cf(n) for n in deg}) > 1:
        return "gap or fragment"
    # terminals: nodes at the circuit's own two stations (or the tie's border port); short
    # wiggles at the station edge may meet there, so a terminal may hold several ends
    p0 = pieces[0]["properties"]
    term = []
    for bus in (p0["bus1"], p0["bus2"]):
        targets = st.end_targets(bus) + st.ports.get(p0["mrid"], [])
        if not targets:
            return f"no station position for {bus}"
        term.append(
            {
                node[k]
                for k, pt in enumerate(ends)
                if min(t.distance(pt) for t in targets) <= END_TOL_M
            }
        )
    if graph is not None:
        graph.update(idx=idx, geoms=geoms, node=node, short=short, term=term)
    if not term[0] or not term[1]:
        return "end not on its station"
    at_station = term[0] | term[1]
    inner = {n: d for n, d in deg.items() if n not in at_station}
    if any(d == 1 for d in inner.values()):
        return "loose end"
    if any(d > 2 for d in inner.values()):
        return "branch or doubled piece"
    return ""


def repair_route(st: Stations, pieces: list[dict]) -> list[dict] | None:
    """The drawn pieces on the shortest connected path (by length) between the circuit's two
    station terminals: stray branches and doubled pieces are dropped. None when no connected
    path exists or the result is still not clean."""
    import heapq

    g: dict = {}
    circuit_problem(st, pieces, g)
    if not g.get("term") or not g["term"][0] or not g["term"][1]:
        return None
    adj = defaultdict(list)
    for e, (geom, sh) in enumerate(zip(g["geoms"], g["short"], strict=True)):
        if sh:
            continue
        a, b = g["node"][2 * e], g["node"][2 * e + 1]
        adj[a].append((b, e, geom.length))
        adj[b].append((a, e, geom.length))
    src, dst = g["term"][0], g["term"][1]
    dist = {n: 0.0 for n in src}
    prev: dict = {}
    heap = [(0.0, n) for n in src]
    end = None
    while heap:
        d, n = heapq.heappop(heap)
        if d > dist.get(n, float("inf")):
            continue
        if n in dst:
            end = n
            break
        for m, e, w in adj[n]:
            if d + w < dist.get(m, float("inf")):
                dist[m], prev[m] = d + w, (n, e)
                heapq.heappush(heap, (d + w, m))
    if end is None:
        return None
    keep = set()
    while end in prev:
        end, e = prev[end]
        keep.add(g["idx"][e])
    out = [f for i, f in enumerate(pieces) if i in keep]
    if not out or circuit_problem(st, out):
        return None
    return out


def fallback_line(st: Stations, p: dict):
    """A clean straight line between the circuit's two stations, cut at their shapes."""
    from shapely.geometry import LineString

    pts = []
    for bus in (p["bus1"], p["bus2"]):
        site = st.site(bus)
        port = st.ports.get(p["mrid"])
        shp = st.shape_of.get(site)
        if shp is not None:
            pts.append((shp.centroid, st.block_of.get(site, shp)))
        elif st.point.get(site) is not None or st.point.get(bus) is not None:
            pts.append(
                (st.point.get(site) if st.point.get(site) is not None else st.point.get(bus), None)
            )
        elif port:
            pts.append((port[0], None))
        else:
            return None
    line = LineString([pts[0][0], pts[1][0]])
    for _, shp in pts:
        if shp is not None and line.length > 0:
            cut = line.difference(shp)
            if cut.geom_type == "LineString" and cut.length > 0:
                line = cut
            elif cut.geom_type == "MultiLineString":
                line = max(cut.geoms, key=lambda g: g.length)
    return st.to_wgs(line)


def connect_circuits(data: Path, lines: dict) -> tuple[dict, list[dict]]:
    """Production lines: a circuit keeps its drawn route when it is one connected line from
    station to station; a broken one (gap, loose or unattached end, fragment) is replaced by
    one clean line between its two stations in the same style. Circuits drawn only inside a
    station (no geometry) and HVDC links (drawn to the converter) are kept as they are."""
    from shapely.geometry import mapping

    st = Stations(data)
    by = defaultdict(list)
    for f in lines["features"]:
        by[f["properties"]["mrid"]].append(f)
    redraw, report = {}, []
    for mrid, fs in by.items():
        p = fs[0]["properties"]
        if p.get("hvdc") or all(f.get("geometry") is None for f in fs):
            continue
        why = circuit_problem(st, fs)
        if why:
            redraw[mrid] = why

    def _rep(fs, why, action, n, m):
        p = fs[0]["properties"]
        return {
            "mrid": p["mrid"],
            "name": p["name"],
            "voltage_kv": p.get("voltage_kv"),
            "tso": p.get("tso"),
            "status": p.get("status"),
            "reason": why,
            "action": action,
            "pieces_before": len(fs),
            "pieces_after": n,
            "km_after": round(m / 1000, 2),
        }

    # parallel fallbacks between the same two stations sit side by side (slots)
    pair = defaultdict(list)
    for mrid in redraw:
        p = by[mrid][0]["properties"]
        pair[tuple(sorted((st.site(p["bus1"]), st.site(p["bus2"]))))].append(mrid)
    slot = {}
    for _key, ms in pair.items():
        for k, m in enumerate(sorted(ms, key=lambda m: by[m][0]["properties"]["name"])):
            slot[m] = (k, len(ms))
    repaired = {}
    for mrid in list(redraw):
        r = repair_route(st, by[mrid])
        if r is not None:
            repaired[mrid] = {id(f) for f in r}
            report.append(
                _rep(
                    by[mrid],
                    redraw[mrid],
                    "repaired: shortest connected path",
                    len(r),
                    sum(st.geom(f["geometry"]).length for f in r),
                )
            )
            del redraw[mrid]
    out = []
    done = set()
    for f in lines["features"]:
        mrid = f["properties"]["mrid"]
        if mrid in repaired:
            if id(f) in repaired[mrid]:
                out.append(f)
            continue
        if mrid not in redraw:
            out.append(f)
            continue
        if mrid in done:
            continue
        done.add(mrid)
        p = dict(f["properties"])
        g = fallback_line(st, p)
        k, n = slot[mrid]
        p.update(
            piece=0,
            pieces=1,
            label_piece=True,
            circuit_no=k,
            circuits=n,
            same_route=n,
            tower_way="",
        )
        out.append(
            {"type": "Feature", "properties": p, "geometry": mapping(g) if g is not None else None}
        )
        km = st.geom(mapping(g)).length if g is not None else 0.0
        report.append(_rep(by[mrid], redraw[mrid], "straight fallback (no connected path)", 1, km))
    return {**lines, "features": out}, report


# ---------------------------------------------------------------- model additions


def model_machines(model: Path) -> dict[str, list[dict]]:
    """The model's machines per bus node (synchronous and inverter), from the frozen pickle."""
    import pandas as pd

    _silver, cim = pd.read_pickle(model)
    m = cim.synchronous_machine
    out = defaultdict(list)
    for r in m.sort_values("max_p_mw", ascending=False).itertuples():
        out[r.node].append(
            {
                "mrid": r.mrid,
                "fuel": r.fuel,
                "units": float(r.units),
                "max_mw": round(float(r.max_p_mw), 1),
                "p_mw": round(float(r.p_mw), 1),
                "in_service": bool(r.in_service),
                "type": "inverter" if bool(r.renewable) else "synchronous",
            }
        )
    return dict(out)


def farm_names(src: Path) -> dict[str, float]:
    f = src / "offshore_farms.geojson"
    if not f.exists():
        return {}
    return {
        x["properties"]["name"]: x["properties"].get("capacity_mw")
        for x in json.loads(f.read_text())["features"]
    }


def with_farm_mw(text: str, mw: dict[str, float]) -> str:
    """'Hohe See; Riffgat (via X)' -> 'Hohe See (497 MW); Riffgat (113 MW, via X)'."""
    parts = []
    for t in str(text).split(";"):
        t = t.strip()
        m = re.match(r"(.*?)\s*\((via [^)]*)\)$", t)
        name, via = (m.group(1), m.group(2)) if m else (t, "")
        cap = mw.get(name)
        inner = ", ".join(x for x in [f"{cap:.0f} MW" if cap else "", via] if x)
        parts.append(f"{name} ({inner})" if inner else name)
    return "; ".join(parts)


def prod_qml(text: str) -> str:
    """Production QGIS style: one solid line style (no dotted 'route not mapped' rules)."""
    text = re.sub(r"<rule [^>]*route not mapped[^>]*/>\s*", "", text)
    text = re.sub(r"\s*AND &quot;route_quality&quot; NOT IN \([^)]*\)", "", text)
    text = re.sub(
        r'(name="(?:line|outline)_style" type="QString" value=")(dot|dash)[^"]*"', r'\1solid"', text
    )
    text = re.sub(r'(name="use_custom_dash" type="QString" value=")1"', r'\g<1>0"', text)
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("staging", type=Path, help="staging web build folder (e.g. web_v5_1_c16)")
    ap.add_argument("out", type=Path, help="new production folder (must not exist)")
    ap.add_argument(
        "--app", type=Path, default=REPO / "apps/grid_map", help="app files (one codebase)"
    )
    ap.add_argument(
        "--styles", type=Path, default=REPO / "apps/grid_map/qgis/styles", help="QGIS .qml styles"
    )
    ap.add_argument(
        "--model",
        type=Path,
        help="frozen model pickle: the model's machines per "
        "bus go into the station panel (read only)",
    )
    ap.add_argument("--no-gpkg", action="store_true", help="skip the GeoPackage / QGIS zip")
    a = ap.parse_args(argv)
    src = a.staging.resolve()
    if not (src / "metadata.json").exists():
        raise SystemExit(f"{src}: no metadata.json")
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; a production build is written once into a new folder")
    data = a.out / "data"
    data.mkdir(parents=True)

    meta = json.loads((src / "metadata.json").read_text())
    dump = lambda obj: json.dumps(obj, ensure_ascii=False, separators=(",", ":"))  # noqa: E731
    for f in sorted(src.glob("*.geojson")):
        if f.stem in REMOVED:
            continue
        fc = clean_layer(f.stem, json.loads(f.read_text()))
        if f.stem == "renewable_inverters":
            mw = farm_names(src)
            for ft in fc["features"]:
                q = ft["properties"]
                if q.get("offshore_farms_landing"):
                    q["offshore_farms_landing"] = with_farm_mw(q["offshore_farms_landing"], mw)
        (data / f.name).write_text(dump(fc))
    # farm links: only where the farm's model landing is sourced (no open question)
    farms = {
        f["properties"]["name"]: f["properties"]
        for f in json.loads((src / "offshore_farms.geojson").read_text())["features"]
    }
    links = json.loads((data / "offshore_links.geojson").read_text())
    links["features"] = [
        f
        for f in links["features"]
        if farms.get(f["properties"]["name"], {}).get("cable_source")
        and not str(farms[f["properties"]["name"]].get("upstream") or "").startswith("needs owner")
    ]
    (data / "offshore_links.geojson").write_text(dump(links))
    # circuits: one connected line each, station to station (see connect_circuits)
    lines, fallback = connect_circuits(data, json.loads((data / "lines.geojson").read_text()))
    (data / "lines.geojson").write_text(dump(lines))
    rep = a.out / "build_report"
    rep.mkdir()
    import csv

    with (rep / "fallback_circuits.csv").open("w", newline="") as fh:
        cols = [
            "mrid",
            "name",
            "voltage_kv",
            "tso",
            "status",
            "reason",
            "action",
            "pieces_before",
            "pieces_after",
            "km_after",
        ]
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(fallback)
    print(
        f"{len(fallback)} broken circuits: "
        + ", ".join(
            f"{k} {v}"
            for k, v in __import__("collections").Counter(r["action"] for r in fallback).items()
        )
    )
    shutil.copytree(src / "sld", data / "sld")
    details = json.loads((src / "station_details.json").read_text())
    machines = model_machines(a.model) if a.model else {}
    for d in details.values():
        d["flags"] = []
        if machines:
            nodes = d["nodes"] if isinstance(d["nodes"], list) else []
            d["machines"] = [
                {"node": n["node"], **m} for n in nodes for m in machines.get(n["node"], [])
            ]
    (data / "station_details.json").write_text(
        json.dumps(details, ensure_ascii=False, separators=(",", ":"))
    )
    prod_meta = {k: meta[k] for k in META_KEEP if k in meta}
    prod_meta.update(
        mode="production",
        staging_build_id=meta.get("build_id", ""),
        build_id=f"{meta.get('build_id', '')}-prod",
        prod_built_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        staging_source=str(src),
    )
    (data / "metadata.json").write_text(json.dumps(prod_meta, indent=2, ensure_ascii=False))

    hits = []
    for f in sorted([*data.glob("*.json"), *data.glob("*.geojson")]):
        hits += banned_hits(json.loads(f.read_text()), f.name)
    if hits:
        raise SystemExit("debug text left in the production data:\n" + "\n".join(hits[:40]))

    for name in APP_FILES:
        s = a.app / name
        (shutil.copytree if s.is_dir() else shutil.copy2)(s, a.out / name)
    bid = prod_meta["build_id"]
    idx = a.out / "index.html"
    idx.write_text(re.sub(r"\?v=[^\"']*", f"?v={bid}", idx.read_text()))

    if not a.no_gpkg:
        q = a.out / "qgis"
        (q / "styles").mkdir(parents=True)
        for qml in sorted(a.styles.glob("*.qml")):
            text = prod_qml(qml.read_text())
            if qml.stem.split("_")[0] in {"tower", "osm", "offshore"}:
                continue
            if re.search(r"\bcheck\b|halo|mismatch|route_note", text):
                raise SystemExit(f"{qml}: style uses a debug field")
            (q / "styles" / qml.name).write_text(text)
        readme = a.styles.parent / "README.md"
        if readme.exists():
            shutil.copy2(readme, q / "README.md")
        write_gpkg(data, q / "grid_model.gpkg")
        with zipfile.ZipFile(data / "grid_model_qgis.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(q.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(q))
    print(f"production build {bid} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
