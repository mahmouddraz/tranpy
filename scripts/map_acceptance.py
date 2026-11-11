"""Acceptance checks for a grid map export (``grid_gis`` GeoPackage or web GeoJSON dir).

Every map build must pass these before it is published (see
``docs/map_panel.md`` and the map/topology plan). Hard checks:

- every in-service model circuit is drawn exactly once (no repeated piece or OSM edge),
  and the drawn circuit count equals the model;
- tower_line_is_osm_way: every drawn tower line (``osm_lines``) lies on the geometry of
  its OSM way (``--network`` edges) within ``inside_m``;
- inside_box: no drawn line piece (circuits and ``osm_lines``) runs more than ``inside_m``
  (2 m) inside any drawn station shape (``station_footprints``);
- station_touch: both circuit ends reach their own box edge (bare footprint) or junction
  dot within ``touch_m`` (10 m): no floating ends;
- both circuit ends lie on their own station box (footprint) or junction node, within
  ``--tol-m``; when the export cuts lines ``station_clip_m`` outside the box (metadata or
  ``--station-clip-m``), on the box buffered by it;
- no undashed straight segment (a 2-vertex piece longer than ``--straight-m`` or a
  single span longer than ``--span-m`` that is not drawn dashed, ``route_quality=straight``);
- no gap over ``--gap-m`` between consecutive pieces of a circuit;
- the slot of a circuit changes only where the set of circuits on the edge changes
  (if ``circuit_no`` / ``circuits`` / ``tower_way`` exist);
- the metadata names the model sha256;
- box_no_swallow: no grown station box reaches another station's footprint or point;
- no_halo_on_short_way: no ``osm_lines`` piece under 150 m has a mismatch halo;
- halo_popup_consistent: every ``osm_lines`` feature's ``check`` follows from its own
  ``model_n`` vs ``circuits_tag`` (equal: ``match``; fewer / more: ``model_fewer`` /
  ``model_more``, or ``explained`` / ``short``; no model circuit: ``no_model``) and its popup
  list ``model_circuits`` has ``model_n`` entries;
- border_terminals (export with ``border_terminals``): a tie circuit cut at the German border
  ends on its own border port (the crossing; checked in place of the foreign station by the
  end checks), every port lies within ``terminal_m`` of its terminal symbol and every tie the
  metadata lists as cut (``border_cut_ties``) has a terminal;
- sld_complete (web dir with ``station_details.json``): every station names its single-line
  diagram, the file exists under ``sld/`` and the folder is not empty.

Listed (not hard): route/model length ratio outside ``--ratio``.

Usage::

    python scripts/map_acceptance.py --export ~/data/grid/grid_model.gpkg \
        --model ~/data/grid/v5/model_v5_3e.pkl --out ~/data/grid/acceptance/baseline

Writes ``acceptance.json`` and ``acceptance.md``; exits 1 on a hard failure.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

METRIC_CRS = "EPSG:3035"
LAYERS = (
    "lines",
    "osm_lines",
    "substations",
    "station_footprints",
    "buses",
    "tower_lines",
    "neighbours",
    "border_terminals",
)
LIVE = "in_operation"
DASHED = ("straight", "unrouted")
"""route_quality values drawn dashed by the web map (not on the OSM graph)."""


@dataclass
class Settings:
    tol_m: float = 25.0
    gap_m: float = 20.0
    ratio: tuple[float, float] = (0.7, 1.5)
    straight_m: float = 1000.0
    span_m: float = 3000.0
    junction_radius_m: float = 1000.0
    pieces_per_circuit: float = 2.5
    max_list: int = 200
    # geometry checks
    on_way_m: float = 30.0
    spacing_m: float = 90.0
    sample_m: float = 25.0
    zigzag_m: float = 40.0
    turn_deg: float = 150.0
    turn_window_m: float = 200.0
    spike_m: float = 50.0
    spike_len_m: float = 150.0
    inside_box_m: float = 50.0
    min_piece_m: float = 100.0
    touch_m: float = 10.0
    terminal_m: float = 2000.0
    inside_m: float = 2.0
    station_order_m: float = 200.0
    station_clip_m: float | None = None
    """Lines end this far outside their own station box (None: the export's metadata
    ``station_clip_m``, else 0). Ends and ``station_end`` are then checked against the
    box buffered by it (the buffer edge)."""


# ---------------------------------------------------------------- reading


def read_export(path: str | Path) -> tuple[dict[str, gpd.GeoDataFrame], dict[str, str]]:
    """Layers and metadata of a GeoPackage export or a web dir of GeoJSON files."""
    p = Path(path).expanduser()
    layers: dict[str, gpd.GeoDataFrame] = {}
    if p.is_dir():
        for name in LAYERS:
            f = p / f"{name}.geojson"
            if f.exists():
                layers[name] = gpd.read_file(f)
        meta_f = p / "metadata.json"
        meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    else:
        import pyogrio

        names = {n for n, _ in pyogrio.list_layers(p)}
        for name in LAYERS:
            if name in names:
                layers[name] = gpd.read_file(p, layer=name)
        meta = {}
        if "metadata" in names:
            m = pyogrio.read_dataframe(p, layer="metadata")
            meta = dict(zip(m["key"], m["value"], strict=True))
    return layers, {str(k): str(v) for k, v in meta.items()}


def read_model_lines(path: str | Path) -> pd.DataFrame:
    """AC circuits of a frozen model pickle (``mrid``, ``name``, ``status``, ``length_km``)."""
    _silver, cim = pd.read_pickle(Path(path).expanduser())
    seg = cim.ac_line_segment
    return seg[["mrid", "name", "status", "length_km", "nominal_voltage_kv"]].copy()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------- helpers


def _ends(coords: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(coords[0]), np.asarray(coords[-1])


def _chain(
    geoms: list[Any], pairs: list | None = None
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Start, end and inter-piece gaps (metres) of ordered pieces, whatever their direction
    (``pairs`` collects the two end points of each gap)."""
    pts = [
        _ends(list(g.geoms[0].coords) + list(g.geoms[-1].coords))
        if g.geom_type.startswith("Multi")
        else _ends(list(g.coords))
        for g in geoms
    ]
    if len(pts) == 1:
        return pts[0][0], pts[0][1], []
    gaps: list[float] = []
    # orient the first piece away from the second
    a0, b0 = pts[0]
    nxt = np.vstack(pts[1])
    if np.min(np.linalg.norm(nxt - a0, axis=1)) < np.min(np.linalg.norm(nxt - b0, axis=1)):
        a0, b0 = b0, a0
    start, tail = a0, b0
    for a, b in pts[1:]:
        da, db = np.linalg.norm(a - tail), np.linalg.norm(b - tail)
        if db < da:
            a, b = b, a
            da = db
        gaps.append(float(da))
        if pairs is not None:
            pairs.append((tail, a))
        tail = b
    return start, tail, gaps


def _forward(geoms: list[Any]) -> np.ndarray:
    """Per ordered piece: True when drawn in travel direction (its end nearer the next)."""
    xy = [
        np.asarray((g.geoms[0] if g.geom_type.startswith("Multi") else g).coords)[:, :2]
        for g in geoms
    ]
    out: list[bool] = []
    for i, c in enumerate(xy):
        if i + 1 < len(xy):
            nb = xy[i + 1]
            d_end = min(np.hypot(*(c[-1] - nb[0])), np.hypot(*(c[-1] - nb[-1])))
            d_start = min(np.hypot(*(c[0] - nb[0])), np.hypot(*(c[0] - nb[-1])))
            out.append(bool(d_end <= d_start))
        elif i > 0:
            pv = xy[i - 1]
            tail = pv[-1] if out[-1] else pv[0]
            out.append(bool(np.hypot(*(c[0] - tail)) <= np.hypot(*(c[-1] - tail))))
        else:
            out.append(True)
    return np.asarray(out)


def _site_of(bus: str, buses: pd.DataFrame | None) -> str:
    if buses is not None and "bus" in buses and "site" in buses:
        hit = buses.loc[buses["bus"] == bus, "site"]
        if len(hit) and isinstance(hit.iloc[0], str) and hit.iloc[0]:
            return hit.iloc[0]
    return str(bus).split("|")[0]


# ---------------------------------------------------------------- checks


def read_network(path: str | Path) -> dict[str, Any]:
    """``grid_network`` output for the checks: way geometry per OSM way id (EPSG:3035,
    from ``edges.parquet``) and the routed circuits per graph edge (``circuit_pieces``)."""
    import shapely

    d = Path(path).expanduser()
    e = pd.read_parquet(d / "edges.parquet", columns=["way_id", "virtual", "geometry"])
    e = e[~e["virtual"].astype(bool) & (e["way_id"] >= 0)]
    g = gpd.GeoSeries(shapely.from_wkb(e["geometry"].to_numpy()), crs="EPSG:4326")
    g = g.to_crs(METRIC_CRS)
    ways = (
        gpd.GeoDataFrame({"way": e["way_id"].astype(int).astype(str).to_numpy()}, geometry=g.values)
        .dissolve("way")
        .geometry.to_dict()
    )
    pc = pd.read_parquet(d / "circuit_pieces.parquet", columns=["mrid", "edge_id"])
    rt = pd.read_parquet(d / "circuit_routes.parquet", columns=["mrid", "status"])
    drawn = set(rt.loc[rt["status"].isin(["routed", "no_path_strict"]), "mrid"])
    pc = pc[pc["mrid"].isin(drawn) & (pc["edge_id"] >= 0)]
    sets = pc.groupby("edge_id")["mrid"].agg(frozenset)
    return {"ways": ways, "edge_sets": {f"e{int(k)}": v for k, v in sets.items()}}


def run_checks(
    layers: Mapping[str, gpd.GeoDataFrame],
    model: pd.DataFrame,
    meta: Mapping[str, str],
    model_sha: str,
    settings: Settings | None = None,
    network: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """All checks and metrics; the result dict holds ``hard`` (name -> ok), lists and metrics."""
    s = settings or Settings()
    lines = layers["lines"].to_crs(METRIC_CRS)
    if "hvdc" in lines:
        lines = lines[~lines["hvdc"].fillna(False).astype(bool)]
    for col, default in (("piece", 0), ("pieces", 1), ("tower_way", ""), ("route_quality", "")):
        if col not in lines:
            lines[col] = default
    lines["tower_way"] = lines["tower_way"].fillna("").astype(str)
    live = lines[lines["status"].fillna(LIVE) == LIVE]
    no_geom = sorted(set(live.loc[live.geometry.isna() | live.geometry.is_empty, "mrid"]))
    live = live[~(live.geometry.isna() | live.geometry.is_empty)]

    foot = layers.get("station_footprints")
    foot = foot.to_crs(METRIC_CRS) if foot is not None and len(foot) else None
    foot_by_site = {} if foot is None else foot.dissolve("site").geometry.to_dict()
    clip = s.station_clip_m
    if clip is None:
        try:
            clip = float(meta.get("station_clip_m") or 0.0)
        except ValueError:
            clip = 0.0
    end_by_site = {k: g.buffer(clip) for k, g in foot_by_site.items()} if clip else foot_by_site
    # station_end: the line must not run inside the buffer (5 m slack at the edge)
    # with the export's own footprint (``footprint_wkt``: the box may be grown to the route
    # ends) station_end is checked against the footprint itself
    if foot is not None and "footprint_wkt" in foot:
        import shapely

        raw = foot[foot["footprint_wkt"].fillna("") != ""]
        foot_raw = dict(
            zip(raw["site"], shapely.from_wkt(raw["footprint_wkt"].to_numpy()), strict=True)
        )
    else:
        foot_raw = foot_by_site
    yard_by_site = (
        {k: g.buffer(max(clip - 5.0, 0.0)) for k, g in foot_raw.items()} if clip else foot_raw
    )
    nodes = []
    for name in ("substations", "neighbours"):
        lay = layers.get(name)
        if lay is not None and len(lay):
            lay = lay.to_crs(METRIC_CRS)
            key = "site" if "site" in lay else "bus"
            nodes.append(pd.Series(lay.geometry.to_numpy(), index=lay[key].astype(str)))
    node_by_site = pd.concat(nodes) if nodes else pd.Series(dtype=object)
    node_by_site = node_by_site[~node_by_site.index.duplicated()].to_dict()
    buses = layers.get("buses")

    model_live = model[model["status"].fillna(LIVE) == LIVE]
    want = set(model_live["mrid"])
    drawn = set(live["mrid"])  # circuits without geometry count as missing
    # circuits with nothing outside the drawn station shapes (export metadata): shown in
    # the station panel, not as a line
    inside_only = {m for m in str(meta.get("inside_station_only", "")).split(";") if m}
    missing = sorted(want - drawn - inside_only)
    extra = sorted(drawn - want)
    model_len = model.set_index("mrid")["length_km"].to_dict()

    touch_bad: list[dict[str, Any]] = []
    station_pass = 0
    # overlapping drawn shapes form one block: an end on its outline touches the station
    cluster_by_site = dict(foot_by_site)
    shapes_all = list(foot.geometry) if foot is not None and len(foot) else []
    import shapely as _shp

    shape_tree = _shp.STRtree(shapes_all) if shapes_all else None
    # a junction dot inside (or on) a drawn station shape: its legs end on that shape
    for key, pt in list(node_by_site.items()):
        if key in foot_by_site or shape_tree is None:
            continue
        near = [shapes_all[int(i)] for i in shape_tree.query(pt.buffer(s.touch_m))]
        if near:
            node_by_site[key] = _shp.union_all([pt, *near])
    if shape_tree is not None:
        for key, g in foot_by_site.items():
            hit = [shapes_all[int(i)] for i in shape_tree.query(g, "intersects")]
            if len(hit) > 1:
                cluster_by_site[key] = _shp.union_all(hit)
    port_of, term_bad = border_ports(layers, meta, s)
    dup, ends_bad, straight, gaps_bad, ratio_bad, slot_bad = [], [], [], [], [], []
    straight_dashed = 0
    gap_count = 0
    circuit_km = 0.0
    osm_km = 0.0

    # circuits on each OSM edge (tower way string) for the slot check
    # OSM-graph pieces (``edge_id``): one graph edge each, the set is per edge
    by_edge = "edge_id" in live and live["edge_id"].notna().any()
    live = live.copy()
    live["set_key"] = live["tower_way"]
    if by_edge:
        eid = live["edge_id"].fillna(-1).astype(int).astype(str)
        on_graph = (live["tower_way"] != "") & live["edge_id"].notna()
        live["set_key"] = np.where(on_graph, "e" + eid, live["tower_way"])
    ex = live[["mrid", "set_key"]].assign(way=live["set_key"].str.split(";")).explode("way")
    ex = ex[ex["way"].fillna("") != ""]
    way_sets = ex.groupby("way")["mrid"].agg(frozenset).to_dict()
    # with the network: the routed set per edge (before the station cut), as the slots use
    if network is not None:
        way_sets.update(network.get("edge_sets") or {})

    def edge_set(ways: str) -> frozenset:
        return frozenset().union(*(way_sets.get(w, frozenset()) for w in ways.split(";")))

    has_slot = {"circuit_no", "circuits"} <= set(live.columns)

    for mrid, grp in live.groupby("mrid", sort=False):
        grp = grp.sort_values("piece")
        name = str(grp["name"].iloc[0]) if "name" in grp else mrid
        # drawn once: no repeated piece number and no OSM edge used twice by the circuit
        # (missing piece numbers are allowed: the station layout drops in-station pieces;
        # holes show up in the gap check)
        # one OSM way is split into several graph edges: with ``edge_id`` (OSM-graph
        # pieces) the repeated unit is the edge, else the way
        if "edge_id" in grp and grp["edge_id"].notna().all():
            ways = grp.loc[grp["tower_way"] != "", "edge_id"].astype(int).astype(str)
        else:
            ways = grp.loc[grp["tower_way"] != "", "tower_way"]
        # a piece split by a station shape keeps its edge on consecutive rows
        ways = ways[ways.ne(ways.shift())]
        if grp["piece"].duplicated().any() or ways.duplicated().any():
            dup.append(
                {
                    "mrid": mrid,
                    "name": name,
                    "pieces": list(grp["piece"].astype(int)),
                    "repeated_ways": sorted(set(ways[ways.duplicated()])),
                }
            )
        geoms = list(grp.geometry)
        pairs: list = []
        start, end, gaps = _chain(geoms, pairs)
        # a gap where the line passes a drawn station shape (cut at its entry and exit,
        # both ends on the outline) is by design
        passes = [
            shape_tree is not None
            and any(
                shapes_all[int(i)].boundary.distance(Point(a)) <= s.inside_m + 1
                and shapes_all[int(i)].boundary.distance(Point(b)) <= s.inside_m + 1
                for i in shape_tree.query(Point(a).buffer(s.inside_m + 1))
            )
            for a, b in pairs
        ]
        station_pass += sum(1 for g, p in zip(gaps, passes, strict=True) if p and g > s.gap_m)
        gaps = [g for g, p in zip(gaps, passes, strict=True) if not p]
        big = [round(g, 1) for g in gaps if g > s.gap_m]
        gap_count += len(big)
        if big:
            gaps_bad.append({"mrid": mrid, "name": name, "gaps_m": big})
        lengths = grp.geometry.length.to_numpy()
        km = float(lengths.sum()) / 1000
        circuit_km += km
        osm_km += float(lengths[(grp["tower_way"] != "").to_numpy()].sum()) / 1000
        mk = model_len.get(mrid)
        if mk and mk > 0 and not (s.ratio[0] <= km / mk <= s.ratio[1]):
            ratio_bad.append(
                {
                    "mrid": mrid,
                    "name": name,
                    "route_km": round(km, 2),
                    "model_km": mk,
                    "ratio": round(km / mk, 3),
                }
            )
        # straight pieces
        for (_, row), length in zip(grp.iterrows(), lengths, strict=True):
            xy = np.asarray(row.geometry.coords)
            spans = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            is_straight = (len(xy) == 2 and length > s.straight_m) or spans.max() > s.span_m
            # a real OSM span on the circuit's own way (e.g. a river crossing) is not a
            # straight shortcut
            if not is_straight or str(row.get("piece_kind", "")) == "way":
                continue
            if row["route_quality"] in DASHED:
                straight_dashed += 1
            else:
                straight.append(
                    {
                        "mrid": mrid,
                        "name": name,
                        "piece": int(row["piece"]),
                        "length_m": round(float(length)),
                        "longest_span_m": round(float(spans.max())),
                    }
                )
        # ends
        bus1, bus2 = str(grp["bus1"].iloc[0]), str(grp["bus2"].iloc[0])
        cut = str(grp["station_cut"].iloc[0]) if "station_cut" in grp else "buffer"
        ends_at = end_by_site if cut == "buffer" else cluster_by_site
        nodes_at, touch_at = node_by_site, cluster_by_site
        if mrid in port_of:  # tie cut at the border: the foreign end is its border port
            fsite, port = port_of[mrid]
            nodes_at = {**node_by_site, fsite: port}
            for b in (bus1, bus2):
                if _site_of(b, buses) == fsite:
                    nodes_at[b] = port
            ends_at = {k: v for k, v in ends_at.items() if k != fsite}
            touch_at = {k: v for k, v in cluster_by_site.items() if k != fsite}
        bad = _end_errors(bus1, bus2, start, end, buses, ends_at, nodes_at)
        if bad is not None and bad > s.tol_m:
            ends_bad.append(
                {
                    "mrid": mrid,
                    "name": name,
                    "bus1": bus1,
                    "bus2": bus2,
                    "worst_end_m": round(bad, 1) if np.isfinite(bad) else None,
                }
            )
        # station_touch: both ends on their own box edge (bare footprint) or junction dot
        touch = _end_errors(bus1, bus2, start, end, buses, touch_at, nodes_at)
        if touch is None or not np.isfinite(touch) or touch > s.touch_m:
            touch_bad.append(
                {
                    "mrid": mrid,
                    "name": name,
                    "bus1": bus1,
                    "bus2": bus2,
                    "worst_end_m": round(touch, 1)
                    if touch is not None and np.isfinite(touch)
                    else None,
                }
            )
        # slots
        if has_slot and len(grp) > 1:
            slot = (grp["circuit_no"] - (grp["circuits"] - 1) / 2).to_numpy()
            # a screen offset is to the right of the drawn direction: compare slots in
            # travel direction
            slot = slot * np.where(_forward(geoms), 1.0, -1.0)
            ways = grp["set_key"].to_numpy()
            own = [
                foot_by_site[x]
                for x in (_site_of(bus1, buses), _site_of(bus2, buses))
                if x in foot_by_site
            ]
            own_near = gpd.GeoSeries(own).union_all() if own else None
            # pieces shorter than min_piece_m take their neighbours' slot (grid_bundles):
            # compare consecutive longer pieces
            longer = np.flatnonzero(lengths >= s.min_piece_m) if by_edge else np.arange(len(grp))
            for i, j in itertools.pairwise(longer):
                if slot[i] == slot[j] or not ways[i] or not ways[j]:
                    continue
                # circuits meeting a station are ordered by where they reach its outline
                # (grid_bundles): a change next to the circuit's own box is allowed
                near = own_near is not None and s.station_order_m > 0
                if near and geoms[j].distance(own_near) <= s.station_order_m:
                    continue
                between = [w for w in ways[i : j + 1] if w]
                if len({edge_set(w) for w in between}) == 1:
                    slot_bad.append(
                        {
                            "mrid": mrid,
                            "name": name,
                            "piece": i,
                            "slot": [float(slot[i]), float(slot[j])],
                        }
                    )

    # inside_box: no model line inside any drawn station shape (``inside_m`` tolerance)
    inside_bad: list[dict[str, Any]] = []
    if foot is not None and len(foot):
        import shapely

        shapes = list(foot.geometry)
        tree = shapely.STRtree(shapes)
        for rec, g in zip(live[["mrid", "name"]].to_dict("records"), live.geometry, strict=True):
            hit = tree.query(g, "intersects")
            if not len(hit):
                continue
            inside = g.intersection(shapely.union_all([shapes[int(i)] for i in hit])).length
            if inside > s.inside_m:
                inside_bad.append({**rec, "inside_m": round(float(inside), 1)})
    # osm_lines: each drawn tower line lies on its OSM way (``way_id``) and outside every
    # drawn station shape
    osm_bad: list[dict[str, Any]] = []
    osm_inside: list[dict[str, Any]] = []
    ol = layers.get("osm_lines")
    if ol is not None and len(ol):
        import shapely

        ol = ol.to_crs(METRIC_CRS)
        way_geom = (network or {}).get("ways") or {}
        for rec, g in zip(ol[["way_id", "name"]].to_dict("records"), ol.geometry, strict=True):
            w = way_geom.get(str(int(rec["way_id"])))
            if w is None:
                osm_bad.append({**rec, "off_m": None})
                continue
            off = g.difference(w.buffer(s.inside_m)).length
            if off > s.inside_m:
                osm_bad.append({**rec, "off_m": round(float(off), 1)})
            if shape_tree is not None:
                hit = shape_tree.query(g, "intersects")
                inside = (
                    g.intersection(shapely.union_all([shapes_all[int(i)] for i in hit])).length
                    if len(hit)
                    else 0.0
                )
                if inside > s.inside_m:
                    osm_inside.append({**rec, "inside_m": round(float(inside), 1)})
    geo = geometry_checks(live, layers, yard_by_site, buses, s, bare=foot_raw, network=network)
    frag = _fragmented_junctions(layers, live, s)
    mismatch = {}
    tl = layers.get("tower_lines")
    if tl is not None and "mismatch" in tl:
        vc = tl["mismatch"].fillna("").astype(str).value_counts()
        mismatch = {k: int(v) for k, v in vc.items() if k}

    swallow = box_swallow(layers)
    short_halo = short_halos(layers)
    popup_bad = halo_popup_mismatches(layers)

    meta_sha = meta.get("source_sha256", "")
    hard = {
        "box_no_swallow": not swallow,
        "no_halo_on_short_way": not short_halo,
        "halo_popup_consistent": not popup_bad,
        "drawn_once": not dup,
        "count_equals_model": not missing and not extra,
        "ends_on_own_station": not ends_bad,
        "station_touch": not touch_bad,
        "inside_box": not inside_bad and not osm_inside,
        "tower_line_is_osm_way": not osm_bad,
        "no_undashed_straight": not straight,
        "no_gap_over_limit": not gaps_bad,
        "slot_changes_only_at_set_change": not slot_bad,
        "metadata_names_model_sha": bool(meta_sha) and meta_sha == model_sha,
        "border_terminals": not term_bad,
        **{k: not geo["lists"][k] for k in GEOMETRY_CHECKS},
    }
    straight_routes = int(
        live.groupby("mrid")["route_quality"].apply(lambda q: q.isin(DASHED).any()).sum()
    )
    metrics = {
        "model_circuits_in_service": len(want),
        "drawn_circuits_in_service": len(drawn),
        "missing": len(missing),
        "without_geometry": len(set(no_geom) - inside_only),
        "inside_station_only": len(inside_only),
        "extra": len(extra),
        "drawn_not_once": len(dup),
        "ends_off_station": len(ends_bad),
        "station_touch_failures": len(touch_bad),
        "inside_box_pieces": len(inside_bad),
        "osm_lines": 0 if ol is None else len(ol),
        "osm_lines_off_way": len(osm_bad),
        "osm_lines_inside_box": len(osm_inside),
        "inside_box_m": round(sum(r["inside_m"] for r in inside_bad), 1),
        "straight_undashed_pieces": len(straight),
        "straight_dashed_pieces": straight_dashed,
        "straight_circuits_route_quality": straight_routes,
        "gaps_over_limit": gap_count,
        "station_passes": station_pass,
        "circuits_with_gaps": len(gaps_bad),
        "ratio_outside": len(ratio_bad),
        "slot_changes_without_set_change": len(slot_bad),
        "fragmented_junctions": len(frag),
        "share_length_on_osm_ways": round(osm_km / circuit_km, 4) if circuit_km else 0.0,
        "circuit_km_drawn": round(circuit_km, 1),
        "tower_line_mismatch": mismatch,
        "slot_info": has_slot,
        "box_swallows": len(swallow),
        "border_ports": len(port_of),
        "border_terminal_problems": len(term_bad),
        "halos_on_short_ways": len(short_halo),
        "halo_popup_mismatches": len(popup_bad),
        "osm_lines_halo": int(ol["check"].isin(HALO).sum())
        if ol is not None and "check" in ol
        else 0,
        "osm_lines_explained": int((ol["check"] == "explained").sum())
        if ol is not None and "check" in ol
        else 0,
        **geo["metrics"],
    }
    cap = s.max_list
    return {
        "model_sha256": model_sha,
        "metadata_sha256": meta_sha,
        "settings": s.__dict__ | {"ratio": list(s.ratio), "station_clip_m": clip},
        "passed": all(hard.values()),
        "hard": hard,
        "metrics": metrics,
        "lists": {
            "missing": missing[:cap],
            "extra": extra[:cap],
            "drawn_not_once": dup[:cap],
            "ends_off_station": sorted(ends_bad, key=lambda r: -(r["worst_end_m"] or 1e12))[:cap],
            "inside_box": sorted(inside_bad + osm_inside, key=lambda r: -r["inside_m"])[:cap],
            "tower_line_is_osm_way": osm_bad[:cap],
            "border_terminals": term_bad[:cap],
            "station_touch": sorted(touch_bad, key=lambda r: -(r["worst_end_m"] or 1e12))[:cap],
            "straight_undashed": sorted(straight, key=lambda r: -r["length_m"])[:cap],
            "gaps": sorted(gaps_bad, key=lambda r: -max(r["gaps_m"]))[:cap],
            "ratio_outside": sorted(ratio_bad, key=lambda r: -abs(np.log(r["ratio"])))[:cap],
            "slot_changes": slot_bad[:cap],
            "fragmented_junctions": frag[:cap],
            "box_no_swallow": swallow[:cap],
            "no_halo_on_short_way": short_halo[:cap],
            "halo_popup_consistent": popup_bad[:cap],
            **{k: v[:cap] for k, v in geo["lists"].items()},
        },
        "events": geo["events"],
    }


HALO = ("model_fewer", "model_more")
"""``osm_lines.check`` values the map draws with a mismatch halo."""
SHORT_WAY_M = 150.0


def box_swallow(layers: Mapping[str, Any], margin_m: float = 30.0) -> list[dict[str, Any]]:
    """Grown station boxes that reach another station: a box (``station_footprints``) that
    intersects another station's own footprint (``footprint_wkt``) or contains another
    station's or junction's point (``substations``), where its bare footprint plus
    ``margin_m`` did not already (adjacent yards)."""
    import shapely

    foot = layers.get("station_footprints")
    if foot is None or not len(foot):
        return []
    foot = foot.to_crs(METRIC_CRS)
    wkt = foot["footprint_wkt"] if "footprint_wkt" in foot else pd.Series("", index=foot.index)
    own = [
        shapely.from_wkt(w) if isinstance(w, str) and w else g
        for w, g in zip(wkt, foot.geometry, strict=True)
    ]
    base = [o.buffer(margin_m) for o in own]
    sites = list(foot["site"].astype(str))
    subs = layers.get("substations")
    pts = (
        []
        if subs is None
        else list(zip(subs["site"].astype(str), subs.to_crs(METRIC_CRS).geometry, strict=True))
    )
    tree = shapely.STRtree(own)
    out = []
    for i, g in enumerate(foot.geometry):
        for j in tree.query(g, "intersects"):
            j = int(j)
            if sites[j] != sites[i] and not base[i].intersects(own[j]):
                out.append({"site": sites[i], "reaches": sites[j], "what": "footprint"})
        for site, pt in pts:
            if site != sites[i] and g.contains(pt) and not base[i].contains(pt):
                out.append({"site": sites[i], "reaches": site, "what": "point"})
    return out


def short_halos(layers: Mapping[str, Any], short_m: float = SHORT_WAY_M) -> list[dict[str, Any]]:
    """``osm_lines`` drawn shorter than ``short_m`` with a mismatch halo."""
    ol = layers.get("osm_lines")
    if ol is None or not len(ol) or "check" not in ol:
        return []
    m = ol.to_crs(METRIC_CRS)
    bad = m[m["check"].isin(HALO) & (m.geometry.length < short_m)]
    return [
        {"way_id": int(w), "length_m": round(float(g.length), 1)}
        for w, g in zip(bad["way_id"], bad.geometry, strict=True)
    ]


def halo_popup_mismatches(layers: Mapping[str, Any]) -> list[dict[str, Any]]:
    """``osm_lines`` features whose ``check`` does not follow from ``model_n`` vs
    ``circuits_tag``, or whose popup list ``model_circuits`` does not have ``model_n``
    entries."""
    ol = layers.get("osm_lines")
    if ol is None or not len(ol) or "check" not in ol:
        return []
    out = []
    for r in ol.itertuples(index=False):
        try:
            circ = json.loads(getattr(r, "model_circuits", "") or "[]")
        except (TypeError, ValueError):
            circ = None
        n = int(r.model_n)
        tag = getattr(r, "circuits_tag", None)
        check = str(r.check)
        why = ""
        if circ is None or len(circ) != n:
            why = f"popup lists {'?' if circ is None else len(circ)} circuits, model_n {n}"
        elif n == 0:
            why = "" if check == "no_model" else f"no model circuit but check {check}"
        elif check == "no_model":
            why = f"check no_model with model_n {n}"
        elif tag is not None and pd.notna(tag):
            want = "match" if n == tag else ("model_fewer" if n < tag else "model_more")
            ok = {want} if want == "match" else {want, "explained", "short"}
            if check not in ok:
                why = f"OSM {tag:g} / model {n}: expected {want}, got {check}"
        if why:
            out.append({"way_id": int(r.way_id), "check": check, "problem": why})
    return out


def border_ports(
    layers: Mapping[str, Any], meta: Mapping[str, str], s: Settings
) -> tuple[dict[str, tuple[str, Any]], list[dict[str, Any]]]:
    """Border port per cut tie circuit (mrid -> (foreign site, metric point)) and problems."""
    bt = layers.get("border_terminals")
    cut = {m for m in str(meta.get("border_cut_ties", "")).split(";") if m}
    if bt is None or not len(bt):
        return {}, [{"problem": "no terminal", "mrid": m} for m in sorted(cut)]
    bt = bt.to_crs(METRIC_CRS)
    ports, problems = {}, []
    for _, r in bt.iterrows():
        for c in json.loads(r.get("circuits") or "[]"):
            if not c.get("mrid") or not c.get("port"):
                continue
            q = gpd.GeoSeries([Point(c["port"])], crs="EPSG:4326").to_crs(METRIC_CRS).iloc[0]
            ports[c["mrid"]] = (str(c.get("site", "")), q)
            d = float(q.distance(r.geometry))
            if d > s.terminal_m:
                problems.append({"problem": "port far from terminal", "mrid": c["mrid"],
                                 "name": c.get("name"), "distance_m": round(d)})
    problems += [{"problem": "no terminal", "mrid": m} for m in sorted(cut - set(ports))]
    return ports, problems


def _end_errors(bus1, bus2, start, end, buses, foot_by_site, node_by_site) -> float | None:
    """Worst distance (m) of the two ends to their own station; pairs ends either way round."""

    def dist(bus: str, xy: np.ndarray) -> float:
        site = _site_of(bus, buses)
        pt = Point(xy)
        if site in foot_by_site:
            return float(foot_by_site[site].distance(pt))
        for key in (site, bus):
            if key in node_by_site:
                return float(node_by_site[key].distance(pt))
        return float("inf")

    a = max(dist(bus1, start), dist(bus2, end))
    b = max(dist(bus1, end), dist(bus2, start))
    return min(a, b)


def _fragmented_junctions(layers, live, s: Settings) -> list[dict[str, Any]]:
    sub = layers.get("substations")
    if sub is None or "kind" not in sub:
        return []
    junc = sub[sub["kind"].isin(["tee", "junction"])].to_crs(METRIC_CRS)
    out = []
    sindex = live.sindex
    for _, j in junc.iterrows():
        near = live.iloc[sindex.query(j.geometry.buffer(s.junction_radius_m), "intersects")]
        n_circ = near["mrid"].nunique()
        if n_circ and len(near) / n_circ > s.pieces_per_circuit:
            ll = gpd.GeoSeries([j.geometry], crs=METRIC_CRS).to_crs("EPSG:4326").iloc[0]
            out.append(
                {
                    "site": j["site"],
                    "pieces": len(near),
                    "circuits": int(n_circ),
                    "lon": round(ll.x, 5),
                    "lat": round(ll.y, 5),
                }
            )
    return sorted(out, key=lambda r: -r["pieces"] / r["circuits"])


# ---------------------------------------------------------------- report


def markdown(result: Mapping[str, Any], export: str, model: str) -> str:
    out = [
        "# Map acceptance",
        "",
        f"- export: `{export}`",
        f"- model: `{model}` (sha256 `{result['model_sha256'][:16]}…`)",
        f"- metadata sha256: `{result['metadata_sha256'][:16]}…`",
        f"- result: **{'PASS' if result['passed'] else 'FAIL'}**",
        "",
        "## Hard checks",
        "",
        "| check | result |",
        "|---|---|",
    ]
    out += [f"| {k} | {'ok' if v else 'FAIL'} |" for k, v in result["hard"].items()]
    out += ["", "## Metrics", "", "| metric | value |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in result["metrics"].items()]
    for name, rows in result["lists"].items():
        if not rows:
            continue
        out += ["", f"## {name} ({len(rows)} listed)", ""]
        if isinstance(rows[0], dict):
            cols = list(rows[0])
            out += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows[:50]]
        else:
            out += [f"- {r}" for r in rows[:50]]
    return "\n".join(out) + "\n"


def sld_problems(web_dir: str | Path) -> list[dict[str, Any]]:
    """Stations of ``station_details.json`` without a single-line diagram file, and an empty
    or missing ``sld/`` folder (a rebuild without ``--diagrams`` drops them)."""
    web = Path(web_dir)
    details = json.loads((web / "station_details.json").read_text(encoding="utf-8"))
    sld = web / "sld"
    out: list[dict[str, Any]] = []
    if not sld.is_dir() or not any(sld.iterdir()):
        out.append({"site": "", "problem": "sld/ folder missing or empty"})
    for site, d in details.items():
        name = str(d.get("diagram") or "")
        if not name:
            out.append({"site": site, "problem": "no diagram named"})
        elif not (web / name).is_file():
            out.append({"site": site, "problem": f"missing file {name}"})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--export", required=True, help="grid_model.gpkg or web dir")
    ap.add_argument("--model", required=True, help="model pickle")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--tol-m", type=float, default=25.0)
    ap.add_argument("--gap-m", type=float, default=20.0)
    ap.add_argument("--ratio", type=float, nargs=2, default=(0.7, 1.5))
    ap.add_argument("--straight-m", type=float, default=1000.0)
    ap.add_argument("--span-m", type=float, default=3000.0)
    ap.add_argument("--network", help="grid_network dir: its OSM ways and per-edge circuit sets")
    ap.add_argument(
        "--station-clip-m", type=float, help="end buffer (default: export metadata, else 0)"
    )
    a = ap.parse_args(argv)
    settings = Settings(
        tol_m=a.tol_m,
        gap_m=a.gap_m,
        ratio=tuple(a.ratio),
        straight_m=a.straight_m,
        span_m=a.span_m,
        station_clip_m=a.station_clip_m,
    )
    layers, meta = read_export(a.export)
    network = read_network(a.network) if a.network else None
    result = run_checks(
        layers, read_model_lines(a.model), meta, sha256(a.model), settings, network=network
    )
    export = Path(a.export).expanduser()
    if export.is_dir() and (export / "station_details.json").exists():
        sld_bad = sld_problems(export)
        result["hard"]["sld_complete"] = not sld_bad
        result["lists"]["sld_complete"] = sld_bad[: settings.max_list]
        result["metrics"]["sld_problems"] = len(sld_bad)
        result["passed"] = all(result["hard"].values())
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "acceptance.json").write_text(json.dumps(result, indent=1, default=str))
    (out / "acceptance.md").write_text(markdown(result, a.export, a.model))
    pd.DataFrame(result["events"], columns=["check", "mrid", "name", "value", "lon", "lat"]).to_csv(
        out / "events.csv", index=False
    )
    print(json.dumps({"passed": result["passed"], **result["hard"]}, indent=1))
    return 0 if result["passed"] else 1


# ---------------------------------------------------------------- geometry checks

GEOMETRY_CHECKS = ("on_own_way", "no_zigzag", "no_loop", "no_spike", "station_end")
"""Hard geometry checks; each list holds the offending circuits."""


def _sample(line: Any, step: float) -> np.ndarray:
    """Points every ``step`` metres along a line (both ends included)."""
    n = max(2, int(np.ceil(line.length / step)) + 1)
    return np.asarray([line.interpolate(d).coords[0] for d in np.linspace(0, line.length, n)])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``(start, stop)`` index ranges of True runs."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _turns(xy: np.ndarray, k: int, max_deg: float) -> list[int]:
    """Sample indices where the heading over ``k`` samples before vs after turns > max_deg."""
    if len(xy) < 2 * k + 1:
        return []
    a = xy[k:-k] - xy[: -2 * k]
    b = xy[2 * k :] - xy[k:-k]
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ok = (na > 1e-6) & (nb > 1e-6)
    cos = np.einsum("ij,ij->i", a, b) / np.where(ok, na * nb, 1.0)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return [int(i) + k for i in np.flatnonzero(ok & (ang > max_deg))]


def geometry_checks(
    live: gpd.GeoDataFrame,
    layers: Mapping[str, gpd.GeoDataFrame],
    foot_by_site: Mapping[str, Any],
    buses: pd.DataFrame | None,
    s: Settings,
    bare: Mapping[str, Any] | None = None,
    network: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """on_own_way, no_zigzag, no_loop, no_spike, station_end on the drawn lines (EPSG:3035).

    Each drawn line is sampled every ``sample_m``; ``d`` is the distance of a sample to the
    nearest OSM way of the circuit (its ``tower_way`` ids), on the GeoJSON geometry, i.e.
    before the web map's screen offset (OSM-graph pieces are exact). A baked bundle offset
    (``offset_baked``: circuits side by side, ``spacing_m`` apart) is allowed on top of
    ``on_way_m``; spikes are measured against the local median of ``d``, so a constant
    offset is no spike. ``foot_by_site`` are the station boxes (buffered by the station
    clip when the export cuts lines outside them). Parts inside the circuit's own end
    stations and dashed lines are not checked against the ways. ``events`` holds one point
    per finding (lon/lat).
    """
    import shapely

    tl = layers.get("tower_lines")
    ways: dict[str, Any] = {}
    if tl is not None and len(tl) and "osm_id" in tl:
        tl = tl.to_crs(METRIC_CRS)
        ways = dict(zip(tl["osm_id"].astype(str), tl.geometry, strict=True))
    # the OSM ways of the network graph (grid_network edges), for ways not in tower_lines
    for w, g in ((network or {}).get("ways") or {}).items():
        ways.setdefault(w, g)
    lists: dict[str, list] = {k: [] for k in GEOMETRY_CHECKS}
    events: list[dict[str, Any]] = []
    on_km = all_km = strict_km = virtual_km = 0.0
    k_turn = max(1, round(s.turn_window_m / 2 / s.sample_m))
    k_med = max(2, round(300 / s.sample_m))
    baked = "offset_baked" in live and live["offset_baked"].fillna(False).astype(bool).any()

    def add_event(check, row, xy, value):
        events.append(
            {
                "check": check,
                "mrid": row["mrid"],
                "name": row.get("name", ""),
                "x": float(xy[0]),
                "y": float(xy[1]),
                "value": round(float(value), 1),
            }
        )

    for _, row in live.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        parts = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
        ends = {_site_of(str(row.get(b, "")), buses) for b in ("bus1", "bus2")}
        cut = str(row.get("station_cut", "buffer") or "buffer")
        boxes = foot_by_site if cut == "buffer" or bare is None else bare
        own_boxes = [boxes[x] for x in ends if x in boxes]
        box_union = shapely.union_all(own_boxes) if own_boxes else None
        name = str(row.get("name", row["mrid"]))
        # station_end: line inside own boxes
        if box_union is not None and cut != "none":
            inside = geom.intersection(box_union).length
            if inside > s.inside_box_m * len(own_boxes):
                lists["station_end"].append(
                    {"mrid": row["mrid"], "name": name, "inside_box_m": round(inside)}
                )
                c = box_union.centroid
                add_event("station_end", row, (c.x, c.y), inside)
        # loop: self-intersection
        if not geom.is_simple:
            lists["no_loop"].append({"mrid": row["mrid"], "name": name, "kind": "self-crossing"})
            p = geom.representative_point()
            add_event("no_loop", row, (p.x, p.y), 0)
        dashed = row.get("route_quality", "") in DASHED
        # OSM-graph join/connector pieces are the graph's virtual links (no OSM way): not
        # checked against the ways, their length is reported
        if str(row.get("piece_kind", "") or "") in ("join", "connector"):
            virtual_km += geom.length / 1000
            continue
        way_ids = [w for w in str(row.get("tower_way", "")).split(";") if w in ways]
        own = [ways[w] for w in way_ids]
        allow = s.on_way_m
        if baked and bool(row.get("offset_baked", False)):
            n = row.get("max_n", row.get("circuits", 1))
            n = 1 if pd.isna(n) else float(n)
            allow += max(0.0, (n - 1) / 2) * s.spacing_m
        turned = spiked = zig = False
        off_len = 0.0
        for part in parts:
            xy = _sample(part, s.sample_m)
            pts = shapely.points(xy)
            seg = s.sample_m if len(xy) > 1 else 0.0
            # back-tracking
            for i in _turns(xy, k_turn, s.turn_deg):
                add_event("no_loop", row, xy[i], s.turn_deg)
                turned = True
            if dashed:
                continue
            check = np.ones(len(xy), bool)
            if box_union is not None:
                check &= ~shapely.contains(box_union, pts)
            all_km += check.sum() * seg / 1000
            if not own:
                off_len += check.sum() * seg
                if check.sum() > 2:
                    mid = np.flatnonzero(check)[check.sum() // 2]
                    add_event("on_own_way", row, xy[mid], check.sum() * seg)
                continue
            tree = shapely.STRtree(own)
            idx = tree.query_nearest(pts, return_distance=False, all_matches=False)[1]
            d = shapely.distance(pts, np.asarray(own, dtype=object)[idx])
            ok = d <= allow
            strict_km += (check & (d <= s.on_way_m)).sum() * seg / 1000
            on_km += (check & ok).sum() * seg / 1000
            off_len += (check & ~ok).sum() * seg
            for a0, b0 in _runs(check & ~ok):
                if b0 - a0 > 2:
                    add_event("on_own_way", row, xy[(a0 + b0) // 2], (b0 - a0) * seg)
            # zigzag: a lateral jump > zigzag_m between consecutive vertices that is steeper
            # than 45 deg relative to the nearest way, or a switch between ways that do not
            # touch
            vx = np.asarray(part.coords)[:, :2]
            vp = shapely.points(vx)
            vi = tree.query_nearest(vp, return_distance=False, all_matches=False)[1]
            vd = shapely.distance(vp, np.asarray(own, dtype=object)[vi])
            vin = np.ones(len(vx), bool) if box_union is None else ~shapely.contains(box_union, vp)
            for i in range(len(vx) - 1):
                if not (vin[i] and vin[i + 1]):
                    continue
                w = own[vi[i]]
                lat = abs(vd[i + 1] - vd[i])
                along = abs(w.project(vp[i + 1]) - w.project(vp[i]))
                touch = vi[i] == vi[i + 1] or w.distance(own[vi[i + 1]]) <= 5.0
                far = vd[i] <= allow and vd[i + 1] <= allow
                if (lat > s.zigzag_m and lat > along) or (not touch and far):
                    add_event("no_zigzag", row, vx[i], lat)
                    zig = True
            # spike: excursion from the local median > spike_m, back within spike_len_m
            med = pd.Series(d).rolling(2 * k_med + 1, center=True, min_periods=1).median()
            exc = (d - med.to_numpy()) > s.spike_m
            for a0, b0 in _runs(exc & check):
                if (b0 - a0) * s.sample_m <= s.spike_len_m and a0 > 0 and b0 < len(xy):
                    j = a0 + int(np.argmax(d[a0:b0]))
                    add_event("no_spike", row, xy[j], d[j] - med.iloc[j])
                    spiked = True
        if off_len > s.sample_m * 2:
            lists["on_own_way"].append(
                {"mrid": row["mrid"], "name": name, "off_way_m": round(off_len), "ways": len(own)}
            )
        if turned:
            lists["no_loop"].append({"mrid": row["mrid"], "name": name, "kind": "turn>150"})
        if zig:
            lists["no_zigzag"].append({"mrid": row["mrid"], "name": name})
        if spiked:
            lists["no_spike"].append({"mrid": row["mrid"], "name": name})

    if events:
        ev = gpd.GeoDataFrame(
            events,
            geometry=gpd.points_from_xy([e["x"] for e in events], [e["y"] for e in events]),
            crs=METRIC_CRS,
        ).to_crs(4326)
        for e, g in zip(events, ev.geometry, strict=True):
            e.pop("x"), e.pop("y")
            e["lon"], e["lat"] = round(g.x, 6), round(g.y, 6)
    metrics = {f"{k}_circuits": len({r["mrid"] for r in v}) for k, v in lists.items()}
    metrics["share_on_own_way"] = round(on_km / all_km, 4) if all_km else 0.0
    metrics["share_on_own_way_strict_30m"] = round(strict_km / all_km, 4) if all_km else 0.0
    metrics["virtual_piece_km"] = round(virtual_km, 1)
    return {"lists": lists, "metrics": metrics, "events": events}


if __name__ == "__main__":
    sys.exit(main())
