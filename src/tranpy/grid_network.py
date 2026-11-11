"""OpenStreetMap 220/380 kV network graph, model circuit routing and per-edge circuit check.

The OSM ``power=line`` ways (:func:`tranpy.grid_towers.read_tower_ways`) are split at shared
nodes (rounded-coordinate node ids, as in ``build_geometry_overlay._ways``) into edges. Every
model AC circuit is routed between its two terminals (station footprint or junction) by a
same-voltage shortest path whose cost is length x a penalty that is lower on edges whose
``ref``/``name`` holds the circuit number, that lie on the circuit's model ``route``, or that
belong to an OSM ``route=power`` relation with the circuit number. Per edge, the number of
routed model circuits is compared with the OSM ``circuits`` count.

Footprints are the station-table OSM polygons, else the fixed box of
:func:`tranpy.grid_stations._box` (:func:`fallback_box`), as drawn on the map. A circuit with
no same-voltage path is retried with ``long_gap`` bridges and other-voltage edges (cost x
:data:`PENALTY_LONG`). Circuits without a number match edges by their line code
(:func:`circuit_code`) and end-station names; edges naming only other circuit numbers cost
x :data:`PENALTY_FOREIGN_REF`.

Strict mode (the default of ``main`` since round 5; ``--legacy`` gives the round-4 graph):
the ways are the Overpass DE 220/380 kV ``power=line``/``cable`` ways with their real node
ids (:func:`fetch_ways`, cached in ``cache/overpass_de_ways_nodes.json``), joined only at
shared node ids. There are no gap, link or long-gap bridges, no other-voltage or transit
fallbacks, no end extensions; the only virtual connection is a terminal's access to the way
nodes inside its footprint (+ :data:`STRICT_BUFFER_M`) or a tee's nearest way node
(:data:`STRICT_SNAP_M`), plus controlled ``join`` edges over OSM tagging breaks
(:func:`add_join_edges`, off with ``--no-joins``). A circuit without a path is
``no_path_strict`` with the break location (``break_lon/lat``, ``break_gap_m``,
``break_ways``). Routes end at their first contact with each end footprint.

Outputs (``python -m tranpy.grid_network --out DIR``):

``edges.parquet`` (graph edges; geometry as WKB, EPSG:4326)
    ``edge_id`` int, ``u``/``v`` int node ids, ``kind`` (``line`` real OSM piece, ``gap``
    virtual bridge between way ends under :data:`GAP_M`, ``join`` controlled strict-mode join
    (:func:`add_join_edges`), ``long_gap`` bridge up to
    :data:`LONG_GAP_M` used only by the retry pass, ``link`` from a still-open way end to the
    nearest other way within :data:`LINK_M`, whose edge is split there), ``gap_where``
    (``station`` / ``midline`` for virtual edges, ``""`` for lines), ``virtual`` bool,
    ``way_id`` int, ``join_ways`` (``"way;way"`` for a ``join`` edge, strict mode),
    ``join_station`` (the station footprint or OSM substation polygon ``osm:<id>`` holding
    both join ends + 50 m: any angle allowed),
    ``kv`` str (220/380 kV parts, ``"220;380"`` for mixed), ``circuits_tag`` float (OSM count
    of 220/380 kV circuits, NaN when untagged), ``circuits_max`` float (upper bound from
    ``cables``), ``ref`` str, ``name`` str, ``length_m`` float, ``geometry`` bytes.
``nodes.parquet``
    ``node`` int, ``lon``, ``lat``, ``degree`` int.
``terminals.parquet``
    ``bus`` (model busbar name, e.g. ``"pasewalk|380"``), ``kind`` (substation / tee / ...),
    ``station``, ``kv``, ``lon``, ``lat``, ``attach`` (``footprint`` / ``point`` /
    ``way_end`` / ``far`` / ``snap`` / ``route_end`` / ``sibling`` / ``far_snap`` / ``none``),
    ``n_nodes``
    (OSM nodes the terminal connects to),
    ``min_cost`` (cheapest access, m).
``circuit_routes.parquet``
    ``mrid``, ``name``, ``kv`` int, ``bus1``/``bus2`` (terminal names), ``edges`` list[int]
    (ordered ``edge_id`` from bus1 to bus2, real and gap edges only), ``nodes`` list[int]
    (ordered OSM nodes), ``route_km`` (sum of edge lengths), ``model_km`` (``length_km``),
    ``ratio`` (route/model), ``model_route_km`` (length of the model ``route`` unless
    schematic; a detour is judged against the larger of the two lengths; a detour is retried
    with long gaps and other-voltage edges), ``gap_edges`` int, ``transit_stations`` (stations
    passed
    through, ``;``-joined), ``on_ref_km`` / ``on_model_route_km`` / ``on_relation_km``
    (km of the route that got each penalty), ``source`` (``osm`` / ``osm+gap`` /
    ``osm+transit`` / ``osm+long_gap`` / ``osm+otherkv``, the last two only from the retry
    pass), ``foreign_ref_km`` (km on edges whose ref/name names only other circuit numbers),
    ``midline_gaps`` (virtual edges used outside footprints), and the station-outline clip:
    ``edges_outside`` list[int] (``edges`` without the pieces inside the end footprints),
    ``entry1_lon/lat`` / ``entry2_lon/lat`` (where the route leaves the bus1 footprint and
    first enters the bus2 footprint), ``entry1_kind`` / ``entry2_kind`` (``outline``;
    ``outline_near`` when the route stops short of the footprint: the geometry then ends with
    a connector to the nearest outline point, ``entry1_gap_m`` / ``entry2_gap_m`` long;
    ``overlap`` when the two footprints touch and nothing is clipped; ``junction`` for tee
    ends, the snapped node),
    ``clip_km``, ``geometry`` (WKB LineString lon/lat, the clipped route); ``extend1_m`` /
    ``extend2_m`` (OSM way length added at that end to reach the footprint,
    :data:`EXTEND_M`; included in ``edges`` and ``route_km``), ``near_other_km`` (km within
    :data:`NEAR_STATION_M` of a station that is not an end, not naming the circuit),
    ``shared_km`` (km on edges of an earlier circuit between the same two stations).
    ``status`` (``routed`` / ``detour`` (see :data:`DETOUR_RATIO`, not counted
    in the edge check) / ``no_path`` / ``no_path_strict`` (strict mode; with ``break_*``; a
    strict-mode detour is also ``no_path_strict``, with ``detour_km`` the long way round and
    ``break_*`` where the network splits once that path is removed, and has no pieces) /
    ``no_terminal``); ``override_reason`` (set by ``route_overrides.csv``, see
    :func:`apply_overrides`; status ``accept_detour`` keeps a reviewed strict-mode detour as
    ``routed``, drawn on its OSM ways).
``edge_check.csv``
    per real edge: ``edge_id``, ``u``, ``v``, ``way_id``, ``kv``, ``ref``, ``name``,
    ``length_m``, ``osm_n`` (circuits_tag), ``osm_max``, ``model_n``, ``model_names`` (``;``),
    ``class``: ``match`` / ``model_fewer`` / ``model_more`` / ``no_tag``; ``unrouted_ref``
    (numbers in the edge ref/name of model circuits that were not routed: such a finding may
    be an artefact), ``foreign_circuits`` (routed circuits whose number the edge's ref/name
    does not name while it names others), ``odd_ratio_circuits`` (routed circuits with a
    ratio outside :data:`RATIO_OK`).
``findings_ranked.csv``
    runs of connected edges with the same class, OSM count and circuit set (not ``match``):
    ``rank``, ``kv`` (max), ``class``, ``osm_n``, ``osm_max``, ``model_n``, ``km``,
    ``n_edges``, ``way_ids``, ``refs``, ``names``, ``circuits``, ``unrouted_ref``,
    ``foreign_circuits``, ``odd_ratio_circuits``, ``lon0``, ``lat0``, ``lon1``, ``lat1``;
    sorted by kv descending then km descending.
``circuit_pieces.parquet``
    one row per drawn piece: ``mrid``, ``name``, ``kv``, ``seq`` (order from bus1 to bus2),
    ``edge_id``, ``way_id`` (the OSM way, -1 for a connector or join), ``kind`` (``way`` /
    ``join`` (a join edge) /
    ``connector``: the at most 50 m straight end to the footprint outline), ``geometry`` (WKB
    LineString lon/lat, clipped at the footprints).
``self_check.csv``
    violations of: every piece on a way carrying the circuit's voltage (``kv``); consecutive
    route edges share a node (``jump``); a used ``join`` joins two ways that intersect away
    from the join (``join_cross``). Empty when clean.
``findings_clean.csv``
    :func:`clean_findings` of ``findings_ranked.csv`` (same columns, re-ranked): the model-gap
    candidates for review.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

GAP_M = 50.0
"""Way ends of the same voltage closer than this are bridged by a virtual ``gap`` edge."""
FOOTPRINT_BUFFER_M = 60.0
"""OSM nodes this close to a station footprint connect to the station terminal."""
POINT_RADIUS_M = 600.0
"""A station without footprint connects to the nodes this close to its point."""
END_REACH_M = 1500.0
"""A station also connects to way ends (degree-1 nodes) of its voltage this close (cost 2 x d)."""
SNAP_NODES = 3
"""A junction attaches to its this many nearest nodes of the voltage."""
SIBLING_M = 10000.0
FAR_SNAP_M = 10000.0
"""Last resort for a terminal without nodes (``sibling`` / ``far_snap``)."""
SNAP_M = 1500.0
STRICT_BUFFER_M = 50.0
"""Strict mode: a station connects to the way nodes inside its footprint + this."""
STRICT_SNAP_M = 200.0
"""Strict mode: a junction (tee) takes its nearest way node of the voltage within this."""
"""A junction (tee) attaches to its nearest node of the voltage within this distance."""
ON_ROUTE_M = 60.0
"""An edge lies on the circuit's model route when its sample points are this close to it."""
PENALTY_REF = 0.5
PENALTY_MODEL_ROUTE = 0.6
PENALTY_RELATION = 0.5
PENALTY_GAP = 2.0
PENALTY_FOREIGN_REF = 3.0
"""An edge whose ref/name names other circuit numbers (3-4 digits) but not this circuit's."""
PENALTY_NAME = 0.8
"""An edge whose ref/name mentions an end station of a circuit without a number."""
PENALTY_LONG = 3.0
"""Retry only: ``long_gap`` bridges and edges of the other voltage (``osm+long_gap`` /
``osm+otherkv``)."""
LINK_M = 100.0
"""A way end still open after the gap bridges joins the nearest other way this close
(``link``)."""
PENALTY_LINK = 3.0
LONG_GAP_M = 3000.0
"""Retry only: way ends of a common voltage this close (each to its 6 nearest) are bridged."""
ACCESS_OUTSIDE = 3.0
"""Access cost per metre to a node outside a station's footprint (when it has one)."""
EXTEND_M = 2500.0
"""A route that stops short of its end footprint is extended along OSM ways this far."""
NEAR_STATION_M = 500.0
"""Edges this close to a footprint of a station that is not an end of the circuit cost
x :data:`PENALTY_NEAR_STATION` unless they name the circuit."""
PENALTY_NEAR_STATION = 3.0
PENALTY_SHARED = 0.7
PENALTY_CONSTRUCTION = 10.0
"""Ways tagged under construction or proposed (:data:`OVERPASS_CONSTRUCTION_QUERY`) cost this
much more, so a circuit uses them only where no in-service OSM path exists (e.g. Mehringen)."""
"""Edges already used by a circuit between the same two stations (parallel circuits)."""
RATIO_OK = (0.7, 1.5)
"""Route/model length ratios outside this mark a circuit's edge-check findings as suspect."""
DETOUR_RATIO = 1.5
DETOUR_EXCESS_KM = 10.0
"""A path longer than ``DETOUR_RATIO`` x the model length and ``DETOUR_EXCESS_KM`` beyond it is
a ``detour`` (the mapped line is missing or has another voltage); it is kept but not counted in
the edge check, and in strict mode it gets no pieces (it is not drawn on the foreign ways of
the long way round). 1.5 is the upper end of :data:`RATIO_OK`; it catches Bischofsheim -
Urberach ERZHAU N (92 km via Kriftel for 47 km), Buerstadt - Y-Buerstadt NASSAU (161 km for
106 km) and Rommerskirchen - Sechtem VORGBG O (52 km for 34 km over Paffendorf)."""
TRANSIT_M = 2000.0
"""Passing through a foreign station's footprint costs this extra (``osm+transit``)."""
EXPECTED_SHA = "3393038e8946540ca904f06705ff0b3cd62d620e3a3dfcc8af0c1eb199b6aba9"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_KM = np.array([np.cos(np.radians(51.0)) * 111.2, 111.2])
_HV = {220, 380}


def _xy_m(lonlat: np.ndarray) -> np.ndarray:
    """Equirectangular metres around 51 degrees N (good to <1 % for German distances)."""
    return np.asarray(lonlat, dtype=float).reshape(-1, 2) * _KM * 1000.0


def circuit_number(name: str) -> str:
    """Trailing 3-4 digit circuit number of a name (``"Pasewalk - Güstrow 315"`` -> 315).

    Shorter numbers (``"Lehrte - Wahle 1"``) are colour/system indices, not line numbers, and
    would match unrelated refs, so they give ``""``.
    """
    m = re.search(r"(\d{3,4})\s*$", str(name))
    return m.group(1) if m else ""


def fold(text: Any) -> str:
    """Lower case without accents, ``ae/oe/ue`` folded to ``a/o/u`` (as in model bus names)."""
    t = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    t = t.lower().replace("ae", "a").replace("oe", "o").replace("ue", "u")
    return re.sub(r"\s+", " ", t)


def circuit_code(name: str) -> str:
    """Folded line code of a circuit without a number (``"... BIENWD W"`` -> ``bienwd``)."""
    codes = re.findall(r"\b[A-ZÄÖÜ]{4,}\b", str(name))
    return fold(codes[-1]) if codes else ""


def _tokens(*texts: Any) -> set[str]:
    out: set[str] = set()
    for t in texts:
        if t is None or (isinstance(t, float) and np.isnan(t)):
            continue
        out |= set(re.findall(r"\d+", str(t)))
    return out


def _kv_set(voltages: str) -> set[int]:
    out = set()
    for p in str(voltages).split(";"):
        with contextlib.suppress(ValueError):
            out.add(int(float(p)) // 1000)
    return out & _HV


@dataclass
class Graph:
    """Edges (real and gap), node coordinates and terminal attachments."""

    edges: pd.DataFrame
    nodes: pd.DataFrame
    coords: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    """edge_id -> lon/lat polyline from u to v."""
    terminals: pd.DataFrame = field(default_factory=pd.DataFrame)
    access: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    """terminal bus -> [(node, access cost m)]."""
    transit: dict[str, list[int]] = field(default_factory=dict)
    """station -> OSM nodes inside its footprint (pass-through hubs)."""
    footprints: dict[str, Any] = field(default_factory=dict)
    """station -> footprint polygon (lon/lat): OSM polygon or fallback box."""
    terminal_kind: dict[str, str] = field(default_factory=dict)
    join_zones: dict[str, Any] = field(default_factory=dict)
    """Station zones for any-angle joins: footprints plus all OSM 220/380 kV substation
    polygons (``osm:<id>``; the station table can hold only part of a yard)."""
    strict: bool = False
    """Real OSM topology only: ways join at shared node ids, no virtual bridges or fallbacks."""


def split_ways(ways: Iterable[dict[str, Any]]) -> tuple[pd.DataFrame, dict[int, list]]:
    """Split ways (``id``, ``coords`` lon/lat list, tag fields) at shared nodes into edges."""
    ways = list(ways)
    node_id: dict[tuple[float, float], int] = {}
    keyed = []
    for w in ways:
        pts = [(round(float(x), 7), round(float(y), 7)) for x, y in w["coords"]]
        if w.get("nodes") is not None:  # real OSM node ids
            real = [
                (p, int(n))
                for i, (p, n) in enumerate(zip(pts, w["nodes"], strict=True))
                if i == 0 or int(n) != int(w["nodes"][i - 1])
            ]
            if len(real) < 2:
                continue
            keyed.append((w, [p for p, _ in real], [n for _, n in real]))
            continue
        pts = [p for i, p in enumerate(pts) if i == 0 or p != pts[i - 1]]
        if len(pts) < 2:
            continue
        keyed.append((w, pts, [node_id.setdefault(p, len(node_id)) for p in pts]))
    count: dict[int, int] = {}
    for _, _, ids in keyed:
        for n in set(ids):
            count[n] = count.get(n, 0) + 1
    ends = {n for _, _, ids in keyed for n in (ids[0], ids[-1])}
    rows, geoms = [], {}
    for w, pts, ids in keyed:
        cuts = [0, *[i for i in range(1, len(ids) - 1) if count[ids[i]] > 1 or ids[i] in ends]]
        cuts.append(len(ids) - 1)
        for i, j in itertools.pairwise(cuts):
            piece = pts[i : j + 1]
            length = float(np.hypot(*np.diff(_xy_m(np.array(piece)), axis=0).T).sum())
            eid = len(rows)
            rows.append(
                {
                    "edge_id": eid,
                    "u": ids[i],
                    "v": ids[j],
                    "kind": "line",
                    "virtual": False,
                    "way_id": int(w["id"]),
                    "kv": ";".join(str(k) for k in sorted(w["kvs"])),
                    "circuits_tag": float(w.get("circuits_tag", np.nan)),
                    "circuits_max": float(w.get("circuits_max", np.nan)),
                    "ref": str(w.get("ref", "") or ""),
                    "name": str(w.get("name", "") or ""),
                    "construction": bool(w.get("construction", False)),
                    "hvdc": bool(w.get("hvdc", False)),
                    "length_m": max(length, 0.1),
                }
            )
            geoms[eid] = piece
    edges = pd.DataFrame(rows)
    return edges, geoms


def add_gap_edges(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    geoms: dict,
    gap_m: float = GAP_M,
    *,
    kind: str = "gap",
    min_m: float = 0.0,
    nearest: int = 0,
    way_ends: bool = False,
):
    """Bridge way ends (degree-1 nodes) of a common voltage closer than ``gap_m``.

    With ``min_m`` only pairs at least that far apart, with ``nearest`` only each end's
    ``nearest`` closest other ends; ``kind`` names the new virtual edges. With ``way_ends``
    every end of an OSM way counts, also where it meets other ways (a 380 kV line ending on
    a mixed 220/380 kV spur, e.g. north of Weida).
    """
    real = edges[edges["kind"] == "line"]
    if way_ends:
        inc = pd.concat(
            [
                real[["way_id", "u"]].rename(columns={"u": "n"}),
                real[["way_id", "v"]].rename(columns={"v": "n"}),
            ]
        )
        once = inc.groupby(["way_id", "n"]).size()
        ends = np.unique(once[once == 1].index.get_level_values("n").to_numpy())
    else:
        deg = pd.concat([real["u"], real["v"]]).value_counts()
        ends = deg[deg == 1].index.to_numpy()
    if len(ends) < 2:
        return edges
    kvs: dict[int, set[int]] = {}
    for u, v, kv in zip(edges["u"], edges["v"], edges["kv"], strict=True):
        for n in (u, v):
            kvs.setdefault(n, set()).update(int(k) for k in kv.split(";") if k)
    lonlat = nodes.set_index("node").loc[ends, ["lon", "lat"]].to_numpy()
    tree = cKDTree(_xy_m(lonlat))
    rows = []
    if nearest:
        dist, idx = tree.query(_xy_m(lonlat), k=nearest + 1, distance_upper_bound=gap_m)
        pairs = {
            (min(i, int(j)), max(i, int(j)))
            for i in range(len(ends))
            for d, j in zip(dist[i][1:], idx[i][1:], strict=True)
            if np.isfinite(d) and d >= min_m
        }
    else:
        pairs = tree.query_pairs(gap_m)
    for i, j in sorted(pairs):
        a, b = int(ends[i]), int(ends[j])
        common = kvs[a] & kvs[b]
        if not common:
            continue
        d = float(np.hypot(*(_xy_m(lonlat[i]) - _xy_m(lonlat[j]))[0]))
        if d < min_m:
            continue
        eid = int(edges["edge_id"].max()) + 1 + len(rows)
        rows.append(
            {
                "edge_id": eid,
                "u": a,
                "v": b,
                "kind": kind,
                "virtual": True,
                "way_id": -1,
                "kv": ";".join(str(k) for k in sorted(common)),
                "circuits_tag": np.nan,
                "circuits_max": np.nan,
                "ref": "",
                "name": "",
                "length_m": max(d, 0.1),
            }
        )
        geoms[eid] = [tuple(lonlat[i]), tuple(lonlat[j])]
    return pd.concat([edges, pd.DataFrame(rows)], ignore_index=True) if rows else edges


def add_link_edges(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    geoms: dict,
    link_m: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join each still-open way end to the nearest other way of its voltage within ``link_m``.

    A line that runs onto a shared tower line often ends tens of metres beside it, away from
    its nodes (e.g. 560 onto the 561/562 way near Streumen). The other way's edge is split at
    the foot of the perpendicular (a new node) and a virtual ``link`` edge joins the end to it
    (cost x :data:`PENALTY_LINK`). Returns the new edges and nodes.
    """
    shapely = __import__("shapely")
    link_m = LINK_M if link_m is None else link_m
    usable = edges["kind"] != "long_gap"
    deg = pd.concat([edges.loc[usable, c] for c in ("u", "v")]).value_counts()
    ends = deg[deg == 1].index.to_numpy()
    real = edges[edges["kind"] == "line"]
    if not len(ends) or real.empty:
        return edges, nodes
    node_way: dict[int, set[int]] = {}
    node_kv: dict[int, set[int]] = {}
    for u, v, w, kv in zip(real["u"], real["v"], real["way_id"], real["kv"], strict=True):
        for n in (u, v):
            node_way.setdefault(n, set()).add(int(w))
            node_kv.setdefault(n, set()).update(int(k) for k in kv.split(";") if k)
    lines = [shapely.LineString(_xy_m(np.asarray(geoms[int(i)]))) for i in real["edge_id"]]
    tree = shapely.STRtree(lines)
    xy = nodes.set_index("node")[["lon", "lat"]]
    splits: dict[int, list[tuple[float, int, int, float]]] = {}
    next_node = int(nodes["node"].max()) + 1
    for a in ends:
        a = int(a)
        if a not in node_kv:
            continue
        pt = shapely.Point(_xy_m(xy.loc[a].to_numpy())[0])
        best = None
        for k in tree.query(pt, predicate="dwithin", distance=link_m):
            row = real.iloc[int(k)]
            if int(row["way_id"]) in node_way[a]:
                continue
            if not node_kv[a] & {int(x) for x in row["kv"].split(";") if x}:
                continue
            d = lines[int(k)].distance(pt)
            if best is None or d < best[0]:
                best = (d, int(k))
        if best is None or best[0] <= 0:
            continue
        d, k = best
        at = lines[k].project(pt)
        splits.setdefault(k, []).append((at, a, next_node, d))
        next_node += 1
    if not splits:
        return edges, nodes
    edges = edges.copy()
    new_rows, new_nodes = [], []
    eid = int(edges["edge_id"].max()) + 1
    substring = __import__("shapely.ops", fromlist=["substring"]).substring
    for k, cuts in splits.items():
        row = real.iloc[k]
        line = lines[k]
        total = line.length
        cuts = sorted(cuts)
        marks = [0.0, *[c[0] for c in cuts], total]
        ids = [int(row["u"]), *[c[2] for c in cuts], int(row["v"])]
        for c in cuts:
            q = np.asarray(line.interpolate(c[0]).coords[0]) / (_KM * 1000.0)
            new_nodes.append((c[2], float(q[0]), float(q[1])))
            # the link edge from the open end to the new node
            new_rows.append(
                {**_virtual_row(eid, c[1], c[2], row["kv"], "link"), "length_m": max(c[3], 0.1)}
            )
            geoms[eid] = [tuple(xy.loc[c[1]].to_numpy()), (float(q[0]), float(q[1]))]
            eid += 1
        pieces = []
        for (m0, m1), (n0, n1) in zip(
            itertools.pairwise(marks), itertools.pairwise(ids), strict=True
        ):
            seg = substring(line, m0, m1)
            coords = np.asarray(seg.coords) / (_KM * 1000.0)
            if len(coords) < 2:
                coords = np.vstack([coords, coords])
            pieces.append((n0, n1, [tuple(map(float, c)) for c in coords], max(m1 - m0, 0.1)))
        first = pieces[0]
        pos = edges.index[edges["edge_id"] == int(row["edge_id"])][0]
        edges.loc[pos, ["v", "length_m"]] = [first[1], first[3]]
        geoms[int(row["edge_id"])] = first[2]
        for n0, n1, coords, length in pieces[1:]:
            new_rows.append({**row.to_dict(), "edge_id": eid, "u": n0, "v": n1, "length_m": length})
            geoms[eid] = coords
            eid += 1
    edges = pd.concat([edges, pd.DataFrame(new_rows)], ignore_index=True)
    nodes = pd.concat(
        [nodes, pd.DataFrame(new_nodes, columns=["node", "lon", "lat"])], ignore_index=True
    )
    return edges, nodes


def _virtual_row(eid: int, u: int, v: int, kv: str, kind: str) -> dict[str, Any]:
    return {
        "edge_id": eid,
        "u": u,
        "v": v,
        "kind": kind,
        "virtual": True,
        "way_id": -1,
        "kv": kv,
        "circuits_tag": np.nan,
        "circuits_max": np.nan,
        "ref": "",
        "name": "",
    }


def attach_terminals(
    graph: Graph, terminals: pd.DataFrame, footprints: dict[str, Any] | None = None
) -> None:
    """Connect terminals (``bus``, ``kind``, ``station``, ``kv``, ``lon``, ``lat``) to nodes.

    Substations connect to every node of their voltage within :data:`FOOTPRINT_BUFFER_M` of
    their footprint (``footprints[station]``, a shapely polygon in lon/lat), else within
    :data:`POINT_RADIUS_M` of their point; junctions (any other kind) snap to the nearest node
    of their voltage within :data:`SNAP_M`. Footprint nodes also serve as transit hubs.
    """
    shapely = __import__("shapely")
    footprints = footprints or {}
    e = graph.edges
    node_kv: dict[int, set[int]] = {}
    for u, v, kv in zip(e["u"], e["v"], e["kv"], strict=True):
        ks = {int(k) for k in kv.split(";") if k}
        node_kv.setdefault(u, set()).update(ks)
        node_kv.setdefault(v, set()).update(ks)
    nodes = graph.nodes[graph.nodes["node"].isin(node_kv)]
    ids = nodes["node"].to_numpy()
    xy = _xy_m(nodes[["lon", "lat"]].to_numpy())
    tree = cKDTree(xy)
    deg = graph.nodes.set_index("node")["degree"]
    end = (deg.reindex(ids).fillna(0) == 1).to_numpy()
    buf_m = STRICT_BUFFER_M if graph.strict else FOOTPRINT_BUFFER_M
    out = []
    for t in terminals.itertuples(index=False):
        p = _xy_m(np.array([t.lon, t.lat]))[0]
        kv = int(t.kv)
        poly = footprints.get(t.station) if t.kind in ("substation", "foreign", "busbar") else None
        station_like = t.kind in ("substation", "foreign", "busbar")
        poly = footprints.get(t.station) if station_like else None
        cost: dict[int, float] = {}
        attach = "none"

        def offer(i: int, factor: float, cost=cost, p=p, kv=kv) -> bool:
            n = int(ids[i])
            if kv not in node_kv[n]:
                return False
            c = 1.0 + factor * float(np.hypot(*(xy[i] - p)))
            cost[n] = min(cost.get(n, np.inf), c)
            return True

        if poly is not None:
            minx, miny, maxx, maxy = poly.bounds
            lo = _xy_m(np.array([minx, miny]))[0] - buf_m
            hi = _xy_m(np.array([maxx, maxy]))[0] + buf_m
            cand = np.flatnonzero(np.all((xy >= lo) & (xy <= hi), axis=1))
            scale = _KM * 1000.0  # buffer in metres, not degrees
            region = shapely.transform(poly, lambda q, scale=scale: q * scale).buffer(buf_m)
            pts = shapely.points(xy[cand])
            inside = cand[shapely.contains(region, pts)] if len(cand) else cand
            hub = [int(ids[i]) for i in inside]
            if hub:
                graph.transit[t.station] = sorted(set(graph.transit.get(t.station, [])) | set(hub))
            hit = [offer(i, 0.05) for i in inside]  # every node is offered
            if any(hit):
                attach = "footprint"
        if station_like and graph.strict:
            pass  # strict: only nodes in the footprint (+30 m)
        elif station_like:
            # with a footprint, outside nodes cost more than walking the way into it
            factor = 0.5 if attach == "none" else ACCESS_OUTSIDE
            near = [offer(i, factor) for i in tree.query_ball_point(p, POINT_RADIUS_M)]
            if any(near) and attach == "none":
                attach = "point"
            factor = 2.0 if attach == "none" else ACCESS_OUTSIDE
            ends = [offer(i, factor) for i in tree.query_ball_point(p, END_REACH_M) if end[i]]
            if any(ends) and attach == "none":
                attach = "way_end"
            if attach == "none":  # approximate station point (no polygon): any node nearby
                far = [offer(i, 2.0) for i in tree.query_ball_point(p, END_REACH_M)]
                attach = "far" if any(far) else "none"
        else:
            reach = STRICT_SNAP_M if graph.strict else SNAP_M
            most = 1 if graph.strict else SNAP_NODES
            dist, idx = tree.query(p, k=min(len(ids), 64), distance_upper_bound=reach)
            hits = 0
            for d, i in zip(np.atleast_1d(dist), np.atleast_1d(idx), strict=True):
                if np.isfinite(d) and hits < most and offer(int(i), 2.0):
                    hits += 1
            attach = "snap" if hits else "none"
        chosen = sorted(cost.items())
        graph.access[t.bus] = chosen
        out.append(
            {
                "bus": t.bus,
                "kind": t.kind,
                "station": t.station,
                "kv": kv,
                "lon": t.lon,
                "lat": t.lat,
                "attach": attach,
                "n_nodes": len(chosen),
                "min_cost": min((c for _, c in chosen), default=np.nan),
            }
        )
    if "route_lon" in terminals:
        for o, rl, rt in zip(out, terminals["route_lon"], terminals["route_lat"], strict=True):
            if o["attach"] == "none":
                o["_route"] = np.array([rl, rt], dtype=float)
    if not graph.strict:
        _fallback_terminals(graph, out, ids, xy, node_kv)
    for o in out:
        o.pop("_route", None)
    graph.terminals = pd.DataFrame(out)
    graph.terminal_kind = dict(zip(graph.terminals["bus"], graph.terminals["kind"], strict=True))


JOIN_M = 50.0
"""Strict mode: a dangling way end joins another way's dangling end (or a graph vertex,
i.e. a node shared by ways) this close, when the direction continues (``kind=join``)."""
JOIN_ANGLE = 45.0
BREAK_CORRIDOR_M = 3000.0
"""A break is looked for within this (or 20 % of the length) of the circuit's model route."""
PENALTY_JOIN = 2.0


def _token_set(ref: str, name: str) -> set[str]:
    text = fold(f"{ref} {name}")
    return set(re.findall(r"\d{3,4}", text)) | set(re.findall(r"[a-z]{5,}", text))


def ways_cross_away(
    g1: Any, g2: Any, ends: Sequence[Any], tol_m: float = JOIN_M, zone: Any = None
) -> bool:
    """True when two metric way geometries intersect farther than ``tol_m`` from ``ends``
    (and, with ``zone``, outside that station zone: crossings inside a yard are allowed)."""
    shapely = __import__("shapely")
    inter = g1.intersection(g2)
    if inter.is_empty:
        return False
    pts = shapely.get_coordinates(inter)
    for q in pts:
        if zone is not None and zone.contains(shapely.Point(q)):
            continue
        if all(np.hypot(*(q - np.asarray(e))) > tol_m for e in ends):
            return True
    return False


def station_zone(graph_footprints: dict[str, Any], name: str) -> Any:
    """Metric footprint + :data:`STRICT_BUFFER_M` of a station (``None`` if unknown)."""
    shapely = __import__("shapely")
    poly = graph_footprints.get(name)
    if poly is None:
        return None
    scale = _KM * 1000.0
    return shapely.transform(poly, lambda q, scale=scale: q * scale).buffer(STRICT_BUFFER_M)


def add_join_edges(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    geoms: dict,
    join_m: float | None = None,
    footprints: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Controlled joins over OSM tagging breaks (strict mode).

    A dangling end (degree 1) joins, within ``join_m``, the nearest dangling end of another
    way, or a graph vertex (a node shared by ways, e.g. a station portal or a tower where ways
    meet), when: the voltages are common; the direction continues (the end's outward segment
    and the other way's segment away from the join differ by less than :data:`JOIN_ANGLE`) or
    the two ways share a ref/name token; and the two ways do not intersect away from the
    join. No edge is split and no point inside an edge is used. When both join ends lie in
    the same station footprint (+ :data:`STRICT_BUFFER_M`) any angle is allowed: inside a
    station there is no neighbouring line to zigzag onto.
    """
    shapely = __import__("shapely")
    join_m = JOIN_M if join_m is None else join_m
    real = edges[edges["kind"] == "line"]
    deg = pd.concat([real["u"], real["v"]]).value_counts()
    pos = nodes.set_index("node")[["lon", "lat"]]
    inc: dict[int, list[int]] = {}
    for eid, u, v in zip(real["edge_id"], real["u"], real["v"], strict=True):
        inc.setdefault(int(u), []).append(int(eid))
        inc.setdefault(int(v), []).append(int(eid))
    row = real.set_index("edge_id")
    kvs = {
        int(e): {int(k) for k in kv.split(";") if k}
        for e, kv in zip(real["edge_id"], real["kv"], strict=True)
    }
    tok = {
        int(e): _token_set(r, n)
        for e, r, n in zip(real["edge_id"], real["ref"], real["name"], strict=True)
    }
    metric = {int(e): _xy_m(np.asarray(geoms[int(e)])) for e in real["edge_id"]}

    def away(eid: int, n: int) -> np.ndarray:
        """Unit direction along edge ``eid`` leaving node ``n``."""
        c = metric[eid]
        if int(row.at[eid, "u"]) != n:
            c = c[::-1]
        d = c[min(1, len(c) - 1)] - c[0]
        k = 1
        while np.hypot(*d) < 5 and k + 1 < len(c):  # skip near-duplicate vertices
            k += 1
            d = c[k] - c[0]
        return d / (np.hypot(*d) or 1.0)

    way_edges = real.groupby("way_id")["edge_id"].apply(list).to_dict()
    way_cache: dict[int, Any] = {}

    def way_geom(w: int) -> Any:
        """The whole way (all its edges), metric: crossings are checked way against way."""
        if w not in way_cache:
            way_cache[w] = shapely.MultiLineString([metric[int(i)] for i in way_edges[w]])
        return way_cache[w]

    names = list(footprints or {})
    st_tree = None
    if names:
        scale = _KM * 1000.0
        zones = [
            shapely.transform(footprints[n], lambda q, scale=scale: q * scale).buffer(
                STRICT_BUFFER_M
            )
            for n in names
        ]
        st_tree = shapely.STRtree(zones)

    def stations_at(q: np.ndarray) -> set[int]:
        if st_tree is None:
            return set()
        return {int(k) for k in st_tree.query(shapely.Point(q), predicate="within")}

    verts = np.array(sorted(inc))
    tree = cKDTree(_xy_m(pos.loc[verts].to_numpy()))
    ends = [int(n) for n in deg[deg == 1].index]
    rows, seen = [], set()
    eid_next = int(edges["edge_id"].max()) + 1
    for a in ends:
        ea = inc[a][0]
        pa = _xy_m(pos.loc[a].to_numpy())[0]
        out_a = -away(ea, a)
        st_a = stations_at(pa)
        best = None
        for j in tree.query_ball_point(pa, join_m):
            b = int(verts[j])
            if b == a or (min(a, b), max(a, b)) in seen:
                continue
            if b in {int(row.at[ea, "u"]), int(row.at[ea, "v"])}:
                continue
            dangling = deg.get(b, 0) == 1
            if not dangling and deg.get(b, 0) < 2:
                continue
            cand = [
                eb
                for eb in inc[b]
                if row.at[eb, "way_id"] != row.at[ea, "way_id"] and kvs[eb] & kvs[ea]
            ]
            pb = _xy_m(pos.loc[b].to_numpy())[0]
            d = float(np.hypot(*(pb - pa)))
            shared = st_a & stations_at(pb)
            same_station = bool(shared)
            zone = zones[min(shared)] if shared else None
            zone_name = names[min(shared)] if shared else ""
            for eb in cand:
                cos = float(np.dot(out_a, away(eb, b)))
                aligned = cos > np.cos(np.radians(JOIN_ANGLE)) or bool(tok[ea] & tok[eb])
                if not (aligned or same_station):
                    continue
                g1 = way_geom(int(row.at[ea, "way_id"]))
                g2 = way_geom(int(row.at[eb, "way_id"]))
                if ways_cross_away(g1, g2, [pa, pb], zone=zone):
                    continue
                # a continuing way wins over a nearer any-angle one inside a station
                score = (0 if aligned else 1, d)
                if best is None or score < best[0]:
                    best = (score, b, eb, zone_name)
        if best is None:
            continue
        (_, d), b, eb, zone_name = best
        seen.add((min(a, b), max(a, b)))
        kv = ";".join(str(k) for k in sorted(kvs[ea] & kvs[eb]))
        rows.append(
            {
                "edge_id": eid_next,
                "u": a,
                "v": b,
                "kind": "join",
                "virtual": True,
                "way_id": -1,
                "kv": kv,
                "circuits_tag": np.nan,
                "circuits_max": np.nan,
                "ref": "",
                "name": "",
                "length_m": max(d, 0.1),
                "join_ways": f"{int(row.at[ea, 'way_id'])};{int(row.at[eb, 'way_id'])}",
                "join_station": zone_name,
            }
        )
        geoms[eid_next] = [tuple(pos.loc[a]), tuple(pos.loc[b])]
        eid_next += 1
    if not rows:
        return edges
    edges = edges.copy()
    edges["join_ways"] = ""
    return pd.concat([edges, pd.DataFrame(rows)], ignore_index=True)


def read_join_overrides(path: Path | None) -> pd.DataFrame | None:
    """``join_overrides.csv``: ``node_a``, ``node_b`` (OSM node ids), ``reason``."""
    if path is None or not Path(path).exists():
        return None
    return pd.read_csv(path, dtype={"node_a": "int64", "node_b": "int64"}).fillna("")


def add_reviewed_joins(
    edges: pd.DataFrame, nodes: pd.DataFrame, geoms: dict, table: pd.DataFrame
) -> pd.DataFrame:
    """Join edges for reviewed tagging breaks (e.g. a branch ending at its tee point at an
    angle the join rules reject); appended after all other edges, so edge ids stay."""
    pos = nodes.set_index("node")[["lon", "lat"]]
    real = edges[edges["kind"] == "line"]
    kv_at: dict[int, set[int]] = {}
    way_at: dict[int, int] = {}
    for u, v, kv, w in zip(real["u"], real["v"], real["kv"], real["way_id"], strict=True):
        for n in (int(u), int(v)):
            kv_at.setdefault(n, set()).update(int(k) for k in str(kv).split(";") if k)
            way_at.setdefault(n, int(w))
    rows = []
    eid = int(edges["edge_id"].max()) + 1
    for r in table.itertuples(index=False):
        a, b = int(r.node_a), int(r.node_b)
        if a not in pos.index or b not in pos.index:
            raise ValueError(f"join override {a}-{b}: node not in the graph")
        kv = kv_at.get(a, set()) & kv_at.get(b, set())
        if not kv:
            raise ValueError(f"join override {a}-{b}: no common voltage")
        pa, pb = pos.loc[a].to_numpy(), pos.loc[b].to_numpy()
        d = float(np.hypot(*(_xy_m(np.array([pb]))[0] - _xy_m(np.array([pa]))[0])))
        rows.append(
            {
                "edge_id": eid,
                "u": a,
                "v": b,
                "kind": "join",
                "virtual": True,
                "way_id": -1,
                "kv": ";".join(str(k) for k in sorted(kv)),
                "circuits_tag": np.nan,
                "circuits_max": np.nan,
                "ref": "",
                "name": "",
                "length_m": max(d, 0.1),
                "join_ways": f"{way_at[a]};{way_at[b]}",
                "join_station": f"override: {r.reason}"[:200],
            }
        )
        geoms[eid] = [tuple(pa), tuple(pb)]
        eid += 1
    out = edges.copy()
    if "join_ways" not in out:
        out["join_ways"] = ""
    return pd.concat([out, pd.DataFrame(rows)], ignore_index=True)


def _gap_where(edges: pd.DataFrame, geoms: dict, footprints: dict[str, Any]) -> list[str]:
    """``station`` for a virtual edge whose midpoint lies in a footprint (+60 m), else
    ``midline``; ``""`` for real edges."""
    shapely = __import__("shapely")
    polys = list(footprints.values())
    tree = shapely.STRtree(polys) if polys else None
    buf = FOOTPRINT_BUFFER_M / 1000.0 / 111.2 * 1.3
    out = []
    for eid, kind in zip(edges["edge_id"], edges["kind"], strict=True):
        if kind == "line":
            out.append("")
            continue
        a, b = geoms[int(eid)][0], geoms[int(eid)][-1]
        mid = shapely.Point((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        hit = tree is not None and len(tree.query(mid, predicate="dwithin", distance=buf)) > 0
        out.append("station" if hit else "midline")
    return out


def _fallback_terminals(graph: Graph, out: list[dict], ids, xy, node_kv) -> None:
    """Terminals still without nodes: the mapped model route end, a sibling, a far snap.

    First the nearest node of the voltage within :data:`SNAP_M` of where the circuits'
    mapped (non-schematic) model routes end at the bus (``route_end``).
    A sibling is an attached terminal of the same voltage whose station name starts with
    this one's first word, within :data:`SIBLING_M` (``gkm`` -> ``gkmb``: the schematic GKM
    point lies 6 km off the plant); ``sibling``. Otherwise the nearest node of the voltage
    within :data:`FAR_SNAP_M` (schematic points, e.g. ``kus lungenkamp``); ``far_snap``.
    """
    tree = cKDTree(xy)
    done = [o for o in out if o["attach"] != "none"]
    for o in out:
        if o["attach"] != "none":
            continue
        q = o.pop("_route", None)
        if q is not None and np.all(np.isfinite(q)):
            dist, idx = tree.query(_xy_m(q)[0], k=min(len(ids), 64), distance_upper_bound=SNAP_M)
            for d, i in zip(np.atleast_1d(dist), np.atleast_1d(idx), strict=True):
                if np.isfinite(d) and o["kv"] in node_kv[int(ids[i])]:
                    graph.access[o["bus"]] = [(int(ids[i]), 1.0 + 2.0 * float(d))]
                    o.update(attach="route_end", n_nodes=1, min_cost=1.0 + 2.0 * float(d))
                    break
            if o["attach"] != "none":
                continue
        p = _xy_m(np.array([o["lon"], o["lat"]]))[0]
        word = fold(o["station"]).split(" ")[0]
        sib = [
            d
            for d in done
            if d["kv"] == o["kv"]
            and d["station"] != o["station"]
            and fold(d["station"]).split(" ")[0].startswith(word)
            and np.hypot(*(_xy_m(np.array([d["lon"], d["lat"]]))[0] - p)) < SIBLING_M
        ]
        if sib and len(word) >= 3:
            best = min(sib, key=lambda d: np.hypot(*(_xy_m(np.array([d["lon"], d["lat"]]))[0] - p)))
            graph.access[o["bus"]] = list(graph.access[best["bus"]])
            o.update(attach="sibling", n_nodes=best["n_nodes"], min_cost=best["min_cost"])
            continue
        dist, idx = tree.query(p, k=min(len(ids), 64), distance_upper_bound=FAR_SNAP_M)
        for d, i in zip(np.atleast_1d(dist), np.atleast_1d(idx), strict=True):
            if np.isfinite(d) and o["kv"] in node_kv[int(ids[i])]:
                graph.access[o["bus"]] = [(int(ids[i]), 1.0 + 2.0 * float(d))]
                o.update(attach="far_snap", n_nodes=1, min_cost=1.0 + 2.0 * float(d))
                break


def build_graph(
    ways: Iterable[dict[str, Any]],
    terminals: pd.DataFrame | None = None,
    footprints: dict[str, Any] | None = None,
    gap_m: float = GAP_M,
    strict: bool = False,
    joins: bool = True,
    join_zones: dict[str, Any] | None = None,
    extra_joins: pd.DataFrame | None = None,
) -> Graph:
    """The network graph from ways (dicts: ``id``, ``coords``, ``kvs``, tags) and terminals.

    With ``strict`` (ways carrying real OSM ``nodes`` ids) ways join only at shared node ids:
    no gap, link or long-gap bridges. ``extra_joins`` (``node_a``, ``node_b``, ``reason``,
    see :func:`read_join_overrides`) adds reviewed join edges the join rules do not make."""
    edges, geoms = split_ways(ways)
    ends = {}
    for eid, u, v in zip(edges["edge_id"], edges["u"], edges["v"], strict=True):
        ends[u] = geoms[eid][0]
        ends[v] = geoms[eid][-1]
    nodes = pd.DataFrame(
        [(n, p[0], p[1]) for n, p in ends.items()], columns=["node", "lon", "lat"]
    ).sort_values("node", ignore_index=True)
    if strict and joins:
        zones = {**(footprints or {}), **(join_zones or {})}
        edges = add_join_edges(edges, nodes, geoms, footprints=zones)
    if strict and extra_joins is not None and len(extra_joins):
        edges = add_reviewed_joins(edges, nodes, geoms, extra_joins)
    if not strict:
        edges = add_gap_edges(edges, nodes, geoms, gap_m)
        edges, nodes = add_link_edges(edges, nodes, geoms)
        edges = add_gap_edges(
            edges, nodes, geoms, LONG_GAP_M, kind="long_gap", min_m=gap_m, nearest=6, way_ends=True
        )
    if "join_ways" not in edges:
        edges["join_ways"] = ""
    edges["join_ways"] = edges["join_ways"].fillna("")
    if "join_station" not in edges:
        edges["join_station"] = ""
    edges["join_station"] = edges["join_station"].fillna("")
    edges["gap_where"] = _gap_where(edges, geoms, footprints or {})
    line = edges[edges["kind"] != "long_gap"]
    deg = pd.concat([line["u"], line["v"]]).value_counts()
    nodes["degree"] = nodes["node"].map(deg).fillna(0).astype(int)
    graph = Graph(edges=edges.reset_index(drop=True), nodes=nodes, coords=geoms, strict=strict)
    graph.footprints = dict(footprints or {})
    graph.join_zones = {**graph.footprints, **(join_zones or {})}
    if terminals is not None:
        attach_terminals(graph, terminals, footprints)
    return graph


@dataclass
class Circuit:
    mrid: str
    name: str
    kv: int
    bus1: str
    bus2: str
    model_km: float
    route: Sequence[tuple[float, float]] = ()
    route_source: str = ""

    @property
    def model_route_km(self) -> float:
        """Length of the model ``route`` when it follows mapped lines, else NaN."""
        if len(self.route) < 2 or self.route_source == "schematic":
            return float("nan")
        xy = _xy_m(np.asarray(self.route, dtype=float))
        return float(np.hypot(*np.diff(xy, axis=0).T).sum()) / 1000.0


class Router:
    """Same-voltage constrained shortest paths with per-circuit edge penalties."""

    def __init__(
        self,
        graph: Graph,
        relations: dict[str, set[int]] | None = None,
        accept_detour: Iterable[str] = (),
    ):
        self.g = graph
        self.relations = relations or {}
        self.accept_detour = set(accept_detour)
        e = graph.edges
        if not (e["edge_id"].to_numpy() == np.arange(len(e))).all():
            raise ValueError("edge_id must equal the row position")
        self.node_index = {int(n): i for i, n in enumerate(graph.nodes["node"])}
        self.n_osm = len(self.node_index)
        self.ui = e["u"].map(self.node_index).to_numpy()
        self.vi = e["v"].map(self.node_index).to_numpy()
        self.length = e["length_m"].to_numpy(float)
        self.is_gap = (e["kind"] == "gap").to_numpy()
        self.is_long = (e["kind"] == "long_gap").to_numpy()
        self.is_link = (e["kind"] == "link").to_numpy()
        self.is_join = (e["kind"] == "join").to_numpy()
        self.is_construction = (
            e["construction"].fillna(False).astype(bool).to_numpy()
            if "construction" in e
            else np.zeros(len(e), dtype=bool)
        )
        self.is_hvdc = (
            e["hvdc"].fillna(False).astype(bool).to_numpy()
            if "hvdc" in e
            else np.zeros(len(e), dtype=bool)
        )
        self.midline = ((e["kind"] != "line") & (e["gap_where"] == "midline")).to_numpy()
        self.kvs = [{int(k) for k in s.split(";") if k} for s in e["kv"]]
        self.tokens = [_tokens(r, n) for r, n in zip(e["ref"], e["name"], strict=True)]
        self.numbers = [{t for t in tok if 3 <= len(t) <= 4} for tok in self.tokens]
        self.text = [fold(f"{r} {n}") for r, n in zip(e["ref"], e["name"], strict=True)]
        self.way = e["way_id"].to_numpy()
        # three sample points per edge (start, middle by vertex, end) for the model-route test
        samples = []
        for eid in e["edge_id"]:
            c = np.asarray(graph.coords[int(eid)])
            samples.append([c[0], c[len(c) // 2], c[-1]])
        self.samples = _xy_m(np.array(samples).reshape(-1, 2)).reshape(-1, 3, 2)
        self.sample_tree = cKDTree(self.samples.reshape(-1, 2))
        # transit hubs: station -> virtual hub index
        self.hubs = {} if graph.strict else {s: n for s, n in graph.transit.items() if n}
        # undirected base graph (real, gap and link edges, any voltage) for end extensions
        base = np.flatnonzero(~self.is_long)
        n = self.n_osm
        r = np.concatenate([self.ui[base], self.vi[base]])
        c = np.concatenate([self.vi[base], self.ui[base]])
        frame = pd.DataFrame({"r": r, "c": c, "w": np.tile(self.length[base], 2)})
        frame["e"] = np.tile(base, 2)
        frame = frame.sort_values("w").drop_duplicates(["r", "c"])
        self.base = coo_matrix((frame["w"], (frame["r"], frame["c"])), shape=(n, n)).tocsr()
        self.base_edge = coo_matrix(
            (frame["e"] + 1, (frame["r"], frame["c"])), shape=(n, n)
        ).tocsr()
        # edges near each station footprint
        self.near_count = np.zeros(len(self.length), dtype=int)
        self.near_edges: dict[str, np.ndarray] = {}
        names = list(graph.footprints)
        if names:
            shapely = __import__("shapely")
            scale = _KM * 1000.0
            polys = [shapely.transform(graph.footprints[s], lambda x: x * scale) for s in names]
            tree = shapely.STRtree(polys)
            pts = shapely.points(self.samples.reshape(-1, 2))
            pi, si = tree.query(pts, predicate="dwithin", distance=NEAR_STATION_M)
            pairs = pd.DataFrame({"e": pi // 3, "s": si}).drop_duplicates()
            for k, grp in pairs.groupby("s"):
                self.near_edges[names[int(k)]] = grp["e"].to_numpy()
            np.add.at(self.near_count, pairs["e"].to_numpy(), 1)
        self.pair_edges: dict[frozenset, set[int]] = {}
        self._label_cache: dict[tuple, np.ndarray] = {}

    def _on_route(self, route: Sequence[tuple[float, float]]) -> np.ndarray:
        hit = np.zeros(len(self.length), dtype=bool)
        if route is None or len(route) < 2:
            return hit
        pts = np.asarray(route, dtype=float)
        dense = [pts[0]]
        for a, b in itertools.pairwise(pts):
            d = float(np.hypot(*(_xy_m(b) - _xy_m(a))[0]))
            k = max(int(d // (ON_ROUTE_M / 2)), 1)
            dense += [a + (b - a) * t for t in np.linspace(0, 1, k + 1)[1:]]
        near = self.sample_tree.query_ball_point(_xy_m(np.array(dense)), ON_ROUTE_M)
        flags = np.zeros(len(self.length) * 3, dtype=bool)
        for lst in near:
            flags[lst] = True
        return flags.reshape(-1, 3).all(axis=1)

    def weights(self, c: Circuit, retry: bool = False) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        num = circuit_number(c.name)
        same_kv = np.array([c.kv in k for k in self.kvs])
        stations = [fold(b.split("|")[0]) for b in (c.bus1, c.bus2)]
        stations = [x for x in stations if len(x) >= 4 and not x.startswith(("y-", "osm/"))]
        both = (
            np.array([all(x in t for x in stations) for t in self.text])
            if len(stations) == 2
            else np.zeros(len(self.text), bool)
        )
        if num:
            ref = np.array([num in t for t in self.tokens])
            # a way naming both end stations is this circuit's line even under another
            # number (JAO renumbering, e.g. Röhrsdorf-Weida 573/574 = OSM 575)
            foreign = np.array([bool(t) and num not in t for t in self.numbers]) & ~both
            name = both
            rel_ways = self.relations.get(num, set())
        else:
            code = circuit_code(c.name)
            ref = np.array([bool(code) and code in t for t in self.text])
            foreign = np.zeros(len(ref), bool)
            name = np.array([any(x in t for x in stations) for t in self.text])
            rel_ways = set(self.relations.get(code, set())) if code else set()
            if len(stations) == 2:
                rel_ways |= self.relations.get(stations[0], set()) & self.relations.get(
                    stations[1], set()
                )
        rel = np.isin(self.way, list(rel_ways)) if rel_ways else np.zeros(len(ref), bool)
        on = self._on_route(c.route)
        p = np.ones(len(ref))
        p[ref] *= PENALTY_REF
        p[name & ~ref] *= PENALTY_NAME
        p[foreign] *= PENALTY_FOREIGN_REF
        p[on] *= PENALTY_MODEL_ROUTE
        p[rel] *= PENALTY_RELATION
        p[self.is_gap] *= PENALTY_GAP
        p[self.is_long] *= PENALTY_LONG
        p[self.is_link] *= PENALTY_LINK
        p[self.is_join] *= PENALTY_JOIN
        p[self.is_construction] *= PENALTY_CONSTRUCTION
        own = np.zeros(len(p), dtype=int)
        ends = {b.split("|")[0] for b in (c.bus1, c.bus2)}
        for st in ends:
            if st in self.near_edges:
                own[self.near_edges[st]] += 1
        near_other = (self.near_count - own) > 0
        p[near_other & ~ref] *= PENALTY_NEAR_STATION
        shared = np.zeros(len(p), dtype=bool)
        used = self.pair_edges.get(frozenset(ends))
        if used:
            shared[list(used)] = True
            p[shared] *= PENALTY_SHARED
        other_kv = ~same_kv & ~self.is_long
        p[other_kv] *= PENALTY_LONG
        allowed = (same_kv & ~self.is_long) | (retry & (self.is_long | other_kv))
        allowed &= ~self.is_hvdc  # AC circuits never run in a DC link (A-Nord, Ultranet)
        w = np.where(allowed, self.length * p, np.inf)
        flags = {"ref": ref | name, "on": on, "rel": rel, "foreign": foreign, "other": other_kv}
        flags.update({"near_other": near_other, "shared": shared})
        return w, flags

    def route(self, c: Circuit) -> dict[str, Any]:
        out = {
            "mrid": c.mrid,
            "name": c.name,
            "kv": c.kv,
            "bus1": c.bus1,
            "bus2": c.bus2,
            "model_km": c.model_km,
            "model_route_km": np.nan,
            "edges": [],
            "nodes": [],
            "route_km": np.nan,
            "ratio": np.nan,
            "gap_edges": 0,
            "transit_stations": "",
            "on_ref_km": 0.0,
            "on_model_route_km": 0.0,
            "on_relation_km": 0.0,
            "foreign_ref_km": 0.0,
            "midline_gaps": 0,
            "edges_outside": [],
            "entry1_lon": np.nan,
            "entry1_lat": np.nan,
            "entry1_kind": "",
            "entry2_lon": np.nan,
            "entry2_lat": np.nan,
            "entry2_kind": "",
            "clip_km": np.nan,
            "entry1_gap_m": np.nan,
            "entry2_gap_m": np.nan,
            "extend1_m": 0.0,
            "extend2_m": 0.0,
            "break_lon": np.nan,
            "break_lat": np.nan,
            "break_gap_m": np.nan,
            "break_ways": "",
            "detour_km": np.nan,
            "near_other_km": 0.0,
            "shared_km": 0.0,
            "geometry": None,
            "source": "",
            "status": "no_terminal",
        }
        a = self.g.access.get(c.bus1, [])
        b = self.g.access.get(c.bus2, [])
        if not a or not b:
            return out
        found = self._path(c, a, b, retry=False)
        ref_km = np.nanmax([c.model_km or 0.0, c.model_route_km])

        def too_long(f) -> bool:
            km = float(self.length[np.array(f[0], dtype=int)].sum()) / 1000.0
            return bool(ref_km) and km > DETOUR_RATIO * ref_km and km - ref_km > DETOUR_EXCESS_KM

        if not self.g.strict and (found is None or too_long(found)):
            # no path, or a detour: retry with long gaps and other-voltage edges
            again = self._path(c, a, b, retry=True)
            if again is not None and (found is None or not too_long(again)):
                found = again
        if found is None:
            out["status"] = "no_path_strict" if self.g.strict else "no_path"
            out.update(self.break_point(c, a, b))
            return out
        edges, nodes, transit, flags, _retried = found
        idx = np.array(edges, dtype=int)
        km = float(self.length[idx].sum()) / 1000.0
        gaps = int((self.is_gap | self.is_long | self.is_link | self.is_join)[idx].sum())
        joins_used = int(self.is_join[idx].sum())
        longs = int(self.is_long[idx].sum())
        other = int(flags["other"][idx].sum())
        source = "osm"
        if transit:
            source = "osm+transit"
        elif joins_used:
            source = "osm+join"
        elif gaps:
            source = "osm+gap"
        if other:
            source = "osm+otherkv"
        elif longs:
            source = "osm+long_gap"

        def km_of(mask: np.ndarray) -> float:
            return float(self.length[idx][mask[idx]].sum()) / 1000

        out.update(
            {
                "edges": edges,
                "nodes": nodes,
                "route_km": km,
                "ratio": km / c.model_km if c.model_km else np.nan,
                "gap_edges": gaps,
                "midline_gaps": int(self.midline[idx].sum()),
                "transit_stations": ";".join(transit),
                "on_ref_km": km_of(flags["ref"]),
                "on_model_route_km": km_of(flags["on"]),
                "on_relation_km": km_of(flags["rel"]),
                "foreign_ref_km": km_of(flags["foreign"]),
                "source": source,
                "status": "routed",
            }
        )
        ext = [0.0, 0.0]
        for end, bus in ((0, c.bus1), (1, c.bus2)):
            if self.g.strict:  # no walking beyond the first footprint contact
                break
            more = self._extend(bus.split("|")[0], nodes[0] if end == 0 else nodes[-1])
            if more is None:
                continue
            e_more, n_more, length = more
            ext[end] = length
            if end == 0:  # path runs from the footprint to the route start
                edges = e_more[::-1] + edges
                nodes = n_more[::-1][:-1] + nodes
            else:
                edges = edges + e_more
                nodes = nodes + n_more[1:]
        if any(ext):
            idx = np.array(edges, dtype=int)
            km = float(self.length[idx].sum()) / 1000.0
            out.update(
                {
                    "edges": edges,
                    "nodes": nodes,
                    "route_km": km,
                    "ratio": km / c.model_km if c.model_km else np.nan,
                }
            )
        out.update(
            {
                "extend1_m": ext[0],
                "extend2_m": ext[1],
                "near_other_km": km_of(flags["near_other"]),
                "shared_km": km_of(flags["shared"]),
            }
        )
        out.update(self._clip(c, edges, nodes))
        out["model_route_km"] = c.model_route_km
        if (
            ref_km
            and km > DETOUR_RATIO * ref_km
            and km - ref_km > DETOUR_EXCESS_KM
            and c.mrid not in self.accept_detour  # reviewed: the OSM path is the real route
        ):
            out["status"] = "detour"
            if self.g.strict:  # drawn dashed: where the direct line breaks
                out.update(self.break_point(c, a, b, detour_of=edges))
                out["status"] = "no_path_strict"
                out["detour_km"] = km
                # the long way round runs on other tower lines: no pieces (the map draws
                # the circuit dashed on its previous geometry, not on foreign ways)
                out["_pieces"] = []
        return out

    def break_point(self, c: Circuit, a, b, detour_of: Sequence[int] = ()) -> dict[str, Any]:
        """Where the same-voltage network between the two terminals breaks.

        The closest pair of nodes, one reachable from each terminal; their midpoint, gap and
        the ways they lie on (``break_lon/lat``, ``break_gap_m``, ``break_ways``)."""
        out = {"break_lon": np.nan, "break_lat": np.nan, "break_gap_m": np.nan, "break_ways": ""}
        labels = self._labels(c.kv, without=tuple(detour_of))
        ra = {labels[self.node_index[n]] for n, _ in a}
        rb = {labels[self.node_index[n]] for n, _ in b}
        ia = np.flatnonzero(np.isin(labels, list(ra)))
        ib = np.flatnonzero(np.isin(labels, list(rb)))
        if not len(ia) or not len(ib):
            return out
        xy = _xy_m(self.g.nodes[["lon", "lat"]].to_numpy())
        # look for the break along the circuit's corridor, not anywhere in the network
        shapely = __import__("shapely")
        if len(c.route) >= 2:
            line = shapely.LineString(_xy_m(np.asarray(c.route, dtype=float)))
        else:
            t = self.g.terminals.set_index("bus")
            ends = [t.loc[x, ["lon", "lat"]].to_numpy(float) for x in (c.bus1, c.bus2)]
            line = shapely.LineString(_xy_m(np.asarray(ends)))
        width = max(BREAK_CORRIDOR_M, 0.2 * line.length)
        zone = line.buffer(width)
        fa = ia[shapely.contains(zone, shapely.points(xy[ia]))]
        fb = ib[shapely.contains(zone, shapely.points(xy[ib]))]
        if len(fa) and len(fb):
            ia, ib = fa, fb
        d, j = cKDTree(xy[ib]).query(xy[ia])
        k = int(np.argmin(d))
        na, nb = int(ia[k]), int(ib[int(j[k])])
        mid = self.g.nodes[["lon", "lat"]].to_numpy()[[na, nb]].mean(axis=0)
        e = self.g.edges
        ways = sorted(
            {int(w) for x in (na, nb) for w in e.loc[(self.ui == x) | (self.vi == x), "way_id"]}
        )
        out.update(
            break_lon=float(mid[0]),
            break_lat=float(mid[1]),
            break_gap_m=float(d[k]),
            break_ways=";".join(str(w) for w in ways),
        )
        return out

    def _labels(self, kv: int, without: tuple[int, ...] = ()) -> np.ndarray:
        """Components of the voltage's graph, ``without`` the given edges (a detour's path:
        what remains split shows where the direct line breaks)."""
        key = (kv, without)
        if key not in self._label_cache:
            from scipy.sparse.csgraph import connected_components

            mask = np.array([kv in k for k in self.kvs]) & ~self.is_long
            if without:
                mask[list(without)] = False
            n = self.n_osm
            m = coo_matrix((np.ones(mask.sum()), (self.ui[mask], self.vi[mask])), shape=(n, n))
            self._label_cache[key] = connected_components(m, directed=False)[1]
        return self._label_cache[key]

    def register(self, c: Circuit, row: dict[str, Any]) -> None:
        """Remember a routed circuit's edges for parallel circuits (same two stations)."""
        if row["status"] == "routed":
            key = frozenset(b.split("|")[0] for b in (c.bus1, c.bus2))
            self.pair_edges.setdefault(key, set()).update(int(e) for e in row["edges"])

    def _extend(self, station: str, node: int) -> tuple[list[int], list[int], float] | None:
        """OSM way path (edges, nodes from ``node``, metres) into ``station``'s footprint.

        ``None`` when ``node`` is in the footprint already or none is reached in
        :data:`EXTEND_M`; the route then keeps its straight connector."""
        targets = self.g.transit.get(station)
        if not targets or node in targets or station not in self.g.footprints:
            return None
        start = self.node_index[node]
        dist, pred = dijkstra(self.base, indices=start, return_predecessors=True, limit=EXTEND_M)
        cand = [(dist[self.node_index[t]], self.node_index[t]) for t in targets]
        d, j = min(cand)
        if not np.isfinite(d):
            return None
        path = [j]
        while path[-1] != start:
            path.append(int(pred[path[-1]]))
        path = path[::-1]
        edges = [int(self.base_edge[x, y]) - 1 for x, y in itertools.pairwise(path)]
        inv = self.g.nodes["node"].to_numpy()
        return edges, [int(inv[q]) for q in path], float(d)

    def _path(self, c: Circuit, a, b, retry: bool):
        w, flags = self.weights(c, retry)
        ok = np.isfinite(w)
        rows = list(self.ui[ok]) + list(self.vi[ok])
        cols = list(self.vi[ok]) + list(self.ui[ok])
        cost = list(w[ok]) * 2
        eidx = list(np.flatnonzero(ok)) * 2
        n = self.n_osm
        src, dst = n, n + 1
        for node, cst in a:
            rows.append(src)
            cols.append(self.node_index[node])
            cost.append(cst)
            eidx.append(-1)
        for node, cst in b:
            rows.append(self.node_index[node])
            cols.append(dst)
            cost.append(cst)
            eidx.append(-1)
        # transit through other stations' footprints (hub node, flagged)
        st1, st2 = c.bus1.split("|")[0], c.bus2.split("|")[0]
        hub_of: dict[int, str] = {}
        size = n + 2
        for st, members in self.hubs.items():
            if st in (st1, st2):
                continue
            h = size
            size += 1
            hub_of[h] = st
            for m in members:
                mi = self.node_index[m]
                rows += [mi, h]
                cols += [h, mi]
                cost += [TRANSIT_M / 2] * 2
                eidx += [-2, -2]
        # keep the cheapest of parallel entries: coo sums duplicates, so dedupe first
        frame = pd.DataFrame({"r": rows, "c": cols, "w": cost, "e": eidx})
        frame = frame.sort_values("w").drop_duplicates(["r", "c"])
        mat = coo_matrix((frame["w"], (frame["r"], frame["c"])), shape=(size, size)).tocsr()
        dist, pred = dijkstra(mat, directed=True, indices=src, return_predecessors=True)
        if not np.isfinite(dist[dst]):
            return None
        path = [dst]
        while path[-1] != src:
            path.append(int(pred[path[-1]]))
        path = path[::-1][1:-1]
        lookup = coo_matrix(
            (frame["e"].to_numpy() + 3, (frame["r"], frame["c"])), shape=(size, size)
        ).tocsr()
        edges, nodes, transit = [], [], []
        for x, y in itertools.pairwise(path):
            e = int(lookup[x, y]) - 3
            if e >= 0:
                edges.append(e)
            if y in hub_of:
                transit.append(hub_of[y])
        inv = self.g.nodes["node"].to_numpy()
        nodes = [int(inv[p]) for p in path if p < n]
        return edges, nodes, transit, flags, retry

    def _clip(self, c: Circuit, edges: list[int], nodes: list[int]) -> dict[str, Any]:
        """End the route at its end stations' footprint outlines (see module docstring)."""
        shapely = __import__("shapely")
        substring = __import__("shapely.ops", fromlist=["substring"]).substring
        shapely.ops = __import__("shapely.ops", fromlist=["nearest_points"])
        coords: list[tuple[float, float]] = []
        spans: list[tuple[int, float, float]] = []
        done = 0.0
        e = self.g.edges
        pos = 0
        for eid in edges:
            u, v = int(e["u"].iat[eid]), int(e["v"].iat[eid])
            piece = list(self.g.coords[eid])
            while pos < len(nodes) and nodes[pos] not in (u, v):
                pos += 1
            if pos < len(nodes) and nodes[pos] == v:
                piece = piece[::-1]
            pos += 1
            coords += piece if not coords else piece[1:]
            spans.append((eid, done, done + self.length[eid]))
            done += self.length[eid]
        out = {
            "edges_outside": list(edges),
            "entry1_lon": np.nan,
            "entry1_lat": np.nan,
            "entry1_kind": "",
            "entry2_lon": np.nan,
            "entry2_lat": np.nan,
            "entry2_kind": "",
            "clip_km": np.nan,
            "entry1_gap_m": np.nan,
            "entry2_gap_m": np.nan,
            "geometry": None,
        }
        if len(coords) < 2:
            return out
        scale = _KM * 1000.0
        line = shapely.LineString(np.asarray(coords) * scale)
        total = line.length
        s1, s2 = 0.0, total
        kinds = []
        outline: dict[int, tuple[np.ndarray, float]] = {}
        for bus, end in ((c.bus1, 0), (c.bus2, 1)):
            poly = self.g.footprints.get(bus.split("|")[0])
            kind = self.g.terminal_kind.get(bus, "")
            if kind not in ("substation", "foreign", "busbar") or poly is None:
                kinds.append("junction" if kind and kind not in ("substation",) else "end")
                continue
            pm = shapely.transform(poly, lambda x: x * scale)
            inside = line.intersection(pm)
            parts = [
                g
                for g in getattr(inside, "geoms", [inside])
                if not g.is_empty and g.geom_type == "LineString" and g.length > 0
            ]
            if not parts:  # the route stops short: the outline point nearest its end
                tip = shapely.Point(line.coords[0] if end == 0 else line.coords[-1])
                gap = float(pm.exterior.distance(tip) if hasattr(pm, "exterior") else 0.0)
                near = shapely.ops.nearest_points(pm.boundary, tip)[0]
                outline[end] = (np.asarray(near.coords[0]) / scale, gap)
                kinds.append("outline_near")
                continue
            if end == 0:  # leave the start station: last exit point
                s1 = max(line.project(shapely.Point(g.coords[-1])) for g in parts)
            else:  # first entry into the end station
                s2 = min(line.project(shapely.Point(g.coords[0])) for g in parts)
            kinds.append("outline")
        if s2 <= s1:  # stations touch or overlap: keep the whole route
            s1, s2 = 0.0, total
            kinds = [k if k != "outline" else "overlap" for k in kinds]
        clipped = substring(line, s1, s2) if (s1, s2) != (0.0, total) else line
        lonlat = np.asarray(clipped.coords) / scale
        gaps_m = [0.0, 0.0]
        if 0 in outline:  # route starts outside its footprint: begin on the outline
            lonlat = np.vstack([outline[0][0], lonlat])
            gaps_m[0] = outline[0][1]
        if 1 in outline:
            lonlat = np.vstack([lonlat, outline[1][0]])
            gaps_m[1] = outline[1][1]
        # map clip distances (projected metres) to the edge spans (edge lengths)
        f = done / total if total else 1.0
        keep = [eid for eid, a0, a1 in spans if a1 > s1 * f + 0.5 and a0 < s2 * f - 0.5]
        pieces = []
        if 0 in outline:
            pieces.append((-1, [outline[0][0], lonlat[1]], "connector"))
        for eid, a0, a1 in spans:
            lo, hi = max(a0 / f, s1), min(a1 / f, s2)
            if hi - lo > 0.5:
                seg = np.asarray(substring(line, lo, hi).coords) / scale
                kind = "join" if self.is_join[int(eid)] else "way"
                pieces.append((int(eid), seg, kind))
        if 1 in outline:
            pieces.append((-1, [lonlat[-2], outline[1][0]], "connector"))
        way = self.g.edges["way_id"].to_numpy()
        out["_pieces"] = [
            {
                "seq": k,
                "edge_id": eid,
                "way_id": int(way[eid]) if eid >= 0 else -1,
                "kind": kind,
                "geometry": shapely.to_wkb(shapely.LineString(np.asarray(seg, dtype=float))),
            }
            for k, (eid, seg, kind) in enumerate(pieces)
        ]
        out.update(
            {
                "edges_outside": keep,
                "entry1_lon": float(lonlat[0][0]),
                "entry1_lat": float(lonlat[0][1]),
                "entry1_kind": kinds[0],
                "entry2_lon": float(lonlat[-1][0]),
                "entry2_lat": float(lonlat[-1][1]),
                "entry2_kind": kinds[1],
                "clip_km": clipped.length / 1000.0,
                "entry1_gap_m": gaps_m[0],
                "entry2_gap_m": gaps_m[1],
                "geometry": shapely.to_wkb(shapely.LineString(lonlat)),
            }
        )
        return out


def route_circuits(
    graph: Graph,
    circuits: Iterable[Circuit],
    relations: dict[str, set[int]] | None = None,
    accept_detour: Iterable[str] = (),
) -> pd.DataFrame:
    """Route every circuit; ``accept_detour`` mrids keep a detour path as ``routed``."""
    router = Router(graph, relations, accept_detour)
    rows, pieces = [], []
    for c in circuits:
        row = router.route(c)
        router.register(c, row)
        for q in row.pop("_pieces", []):
            pieces.append({"mrid": c.mrid, "name": c.name, "kv": c.kv, **q})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["pieces"] = pd.DataFrame(
        pieces, columns=["mrid", "name", "kv", "seq", "edge_id", "way_id", "kind", "geometry"]
    )
    return out


def self_check(graph: Graph, routes: pd.DataFrame, pieces: pd.DataFrame) -> pd.DataFrame:
    """Violations: a piece on a way without the circuit's voltage (``kv``), or consecutive
    route edges that share no node (``jump``)."""
    e = graph.edges.set_index("edge_id")
    bad = []
    for r in pieces[pieces["kind"] == "way"].itertuples(index=False):
        if str(r.kv) not in str(e.at[r.edge_id, "kv"]).split(";"):
            bad.append({"mrid": r.mrid, "name": r.name, "check": "kv", "edge_id": r.edge_id})
    for r in routes.itertuples(index=False):
        eds = list(r.edges) if r.edges is not None else []
        for x, y in itertools.pairwise(eds):
            if not {e.at[x, "u"], e.at[x, "v"]} & {e.at[y, "u"], e.at[y, "v"]}:
                bad.append({"mrid": r.mrid, "name": r.name, "check": "jump", "edge_id": y})
    shapely = __import__("shapely")
    used = {
        int(x)
        for r in routes.itertuples(index=False)
        for x in (r.edges if r.edges is not None else [])
    }
    joins = graph.edges[(graph.edges["kind"] == "join") & graph.edges["edge_id"].isin(used)]
    real = graph.edges[graph.edges["kind"] == "line"]
    for j in joins.itertuples(index=False):
        w1, w2 = (int(x) for x in str(j.join_ways).split(";"))
        g = [
            shapely.MultiLineString(
                [
                    _xy_m(np.asarray(graph.coords[int(i)]))
                    for i in real.loc[real["way_id"] == w, "edge_id"]
                ]
            )
            for w in (w1, w2)
        ]
        ends = _xy_m(np.asarray(graph.coords[int(j.edge_id)]))
        zone = station_zone(graph.join_zones, str(j.join_station or ""))
        if ways_cross_away(g[0], g[1], list(ends), zone=zone):
            bad.append(
                {
                    "mrid": "",
                    "name": f"join {j.join_ways}",
                    "check": "join_cross",
                    "edge_id": j.edge_id,
                }
            )
    return pd.DataFrame(bad, columns=["mrid", "name", "check", "edge_id"])


ACCEPT_DETOUR = "accept_detour"
"""``route_overrides.csv`` status: draw a reviewed detour (route longer than
:data:`DETOUR_RATIO` x the model length) on its OSM ways, e.g. a tie line whose model length
covers one TSO's part only."""


def detour_overrides(path: Path | None) -> set[str]:
    """Mrids whose override status is ``accept_detour``."""
    if path is None or not Path(path).exists():
        return set()
    table = pd.read_csv(path, dtype=str).fillna("")
    return set(table.loc[table["status"] == ACCEPT_DETOUR, "mrid"])


def apply_overrides(
    routes: pd.DataFrame, pieces: pd.DataFrame, path: Path | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hand-set statuses from a CSV (``mrid``, ``status``, ``reason``), e.g. a route the
    rules cannot catch (Preilack-Streumen 560 on the Graustein corridor). The reason goes to
    ``override_reason``; a circuit set to a non-routed status loses its pieces. Status
    ``accept_detour`` (see :func:`detour_overrides`, applied in routing) becomes ``routed``."""
    routes = routes.copy()
    routes["override_reason"] = ""
    if path is None or not Path(path).exists():
        return routes, pieces
    table = pd.read_csv(path, dtype=str).fillna("")
    for r in table.itertuples(index=False):
        hit = routes["mrid"] == r.mrid
        if not hit.any():
            raise ValueError(f"override for unknown circuit {r.mrid}")
        if r.status == ACCEPT_DETOUR:
            if not (routes.loc[hit, "status"] == "routed").all():
                raise ValueError(f"accept_detour override for {r.mrid} found no path")
        else:
            routes.loc[hit, "status"] = r.status
        routes.loc[hit, "override_reason"] = r.reason
    dropped = set(routes.loc[routes["status"] != "routed", "mrid"]) & set(table["mrid"])
    return routes, pieces[~pieces["mrid"].isin(dropped)].reset_index(drop=True)


def classify(model_n: int, osm_n: float) -> str:
    if not np.isfinite(osm_n):
        return "no_tag"
    if model_n == osm_n:
        return "match"
    return "model_fewer" if model_n < osm_n else "model_more"


def edge_check(graph: Graph, routes: pd.DataFrame) -> pd.DataFrame:
    """Per real edge: routed model circuits versus the OSM circuit count."""
    on: dict[int, list[str]] = {}
    missing = {circuit_number(n) for n in routes.loc[routes["status"] != "routed", "name"]}
    missing.discard("")
    routes = routes[routes["status"] == "routed"]
    odd = set(routes.loc[~routes["ratio"].between(*RATIO_OK), "name"].astype(str))
    for name, edges in zip(routes["name"], routes["edges"], strict=True):
        for e in edges if edges is not None else []:
            on.setdefault(int(e), []).append(str(name))
    e = graph.edges[graph.edges["kind"] == "line"].copy()
    e["osm_n"] = e["circuits_tag"]
    e["osm_max"] = e["circuits_max"]
    e["model_n"] = e["edge_id"].map(lambda x: len(on.get(int(x), []))).astype(int)
    e["model_names"] = e["edge_id"].map(lambda x: ";".join(sorted(on.get(int(x), []))))
    e["class"] = [classify(m, o) for m, o in zip(e["model_n"], e["osm_n"], strict=True)]
    e["unrouted_ref"] = [
        ";".join(sorted(_tokens(r, n) & missing)) for r, n in zip(e["ref"], e["name"], strict=True)
    ]
    numbers = [
        {t for t in _tokens(r, n) if 3 <= len(t) <= 4}
        for r, n in zip(e["ref"], e["name"], strict=True)
    ]
    e["foreign_circuits"] = [
        ";".join(
            sorted(
                c
                for c in on.get(int(x), [])
                if nums and circuit_number(c) and circuit_number(c) not in nums
            )
        )
        for x, nums in zip(e["edge_id"], numbers, strict=True)
    ]
    e["odd_ratio_circuits"] = [
        ";".join(sorted(c for c in on.get(int(x), []) if c in odd)) for x in e["edge_id"]
    ]
    cols = ["edge_id", "u", "v", "way_id", "kv", "ref", "name", "length_m", "osm_n", "osm_max"]
    extra = ["model_n", "model_names", "class", "unrouted_ref", "foreign_circuits"]
    return e[[*cols, *extra, "odd_ratio_circuits"]].reset_index(drop=True)


def clean_findings(ranked: pd.DataFrame) -> pd.DataFrame:
    """Model-gap candidates: 220/380 kV findings where the model carries a circuit and that
    are not explained by an unrouted circuit (``unrouted_ref``), a circuit routed on a way
    naming only other circuits (``foreign_circuits``) or one whose route length is off
    (``odd_ratio_circuits``, outside :data:`RATIO_OK`); ``no_tag`` is left out."""
    if ranked.empty:
        return ranked
    keep = (
        ranked["kv"].isin([220, 380])
        & (ranked["model_n"] >= 1)
        & ranked["class"].isin(["model_fewer", "model_more"])
    )
    for col in ("unrouted_ref", "foreign_circuits", "odd_ratio_circuits"):
        keep &= ranked[col].fillna("").astype(str) == ""
    out = ranked[keep].sort_values(["kv", "km"], ascending=[False, False]).copy()
    out["rank"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)


def _find(parent: dict[int, int], x: int) -> int:
    """Union-find root of ``x`` (path halving)."""
    while parent.setdefault(x, x) != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def findings(check: pd.DataFrame, graph: Graph) -> pd.DataFrame:
    """Collapse connected edges with equal class, OSM count and circuit set into findings."""
    bad = check[check["class"] != "match"].copy()
    if bad.empty:
        return pd.DataFrame()
    key = bad["class"] + "|" + bad["osm_n"].astype(str) + "|" + bad["model_names"]
    rows = []
    nodes = graph.nodes.set_index("node")
    for _, grp in bad.groupby(key):
        parent: dict[int, int] = {}
        for u, v in zip(grp["u"], grp["v"], strict=True):
            parent[_find(parent, int(u))] = _find(parent, int(v))
        comp = grp["u"].map(lambda x, p=parent: _find(p, int(x)))
        for _, part in grp.groupby(comp):
            ends = pd.concat([part["u"], part["v"]]).value_counts()
            tips = list(ends[ends == 1].index) or list(ends.index[:2])
            p0 = nodes.loc[tips[0]]
            p1 = nodes.loc[tips[-1]]
            first = part.iloc[0]
            rows.append(
                {
                    "kv": max(int(x) for s in part["kv"] for x in s.split(";")),
                    "class": first["class"],
                    "osm_n": first["osm_n"],
                    "osm_max": part["osm_max"].max(),
                    "model_n": int(first["model_n"]),
                    "km": part["length_m"].sum() / 1000.0,
                    "n_edges": len(part),
                    "way_ids": ";".join(str(w) for w in sorted(set(part["way_id"]))),
                    "refs": ";".join(sorted({r for r in part["ref"] if r})),
                    "names": ";".join(sorted({r for r in part["name"] if r}))[:200],
                    "circuits": first["model_names"],
                    **{
                        col: ";".join(
                            sorted({x for v in part[col] for x in str(v).split(";") if x})
                        )
                        for col in ("unrouted_ref", "foreign_circuits", "odd_ratio_circuits")
                    },
                    "lon0": p0["lon"],
                    "lat0": p0["lat"],
                    "lon1": p1["lon"],
                    "lat1": p1["lat"],
                }
            )
    out = pd.DataFrame(rows).sort_values(["kv", "km"], ascending=[False, False])
    out.insert(0, "rank", range(1, len(out) + 1))
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- data loading (cairo-lab)


def osm_ways(lines: Path, overpass: Sequence[Path] = ()) -> list[dict[str, Any]]:
    """220/380 kV in-service ways as dicts from the OSM export (via ``read_tower_ways``)."""
    from tranpy.grid_towers import read_tower_ways

    frame = read_tower_ways(lines, overpass=overpass)
    frame = frame[~frame["disused"]]
    out = []
    for r in frame.itertuples(index=False):
        g = r.geometry
        kvs = _kv_set(r.voltages)
        for part in getattr(g, "geoms", [g]):
            out.append(
                {
                    "id": r.osm_id,
                    "coords": list(part.coords),
                    "kvs": kvs,
                    "circuits_tag": r.circuits_tag,
                    "circuits_max": r.circuits_max,
                    "ref": r.ref,
                    "name": r.name,
                    "construction": r.construction,
                }
            )
    return out


OVERPASS_WAYS_QUERY = (
    '[out:json][timeout:600];area["ISO3166-1"="DE"][admin_level=2]->.a;'
    '(way[power~"^(line|cable)$"][voltage~"220000|380000"](area.a););out body;>;out skel qt;'
)


OVERPASS_CONSTRUCTION_QUERY = (
    '[out:json][timeout:600];area["ISO3166-1"="DE"][admin_level=2]->.a;'
    '(way[power~"^(construction|proposed)$"][voltage~"220000|380000"](area.a);'
    'way["construction:power"~"^(line|cable)$"][voltage~"220000|380000"](area.a);'
    'way["proposed:power"~"^(line|cable)$"][voltage~"220000|380000"](area.a););'
    "out body;>;out skel qt;"
)
"""220/380 kV ways under construction or proposed (``power=construction``/``proposed``,
``construction:power``/``proposed:power`` = line/cable); routing fallback only."""


def fetch_ways(cache: Path, url: str = OVERPASS_URL, query: str | None = None) -> Path:
    """Overpass DE 220/380 kV ``power=line``/``cable`` ways with node ids and nodes (cached);
    ``query`` replaces :data:`OVERPASS_WAYS_QUERY` (e.g. :data:`OVERPASS_CONSTRUCTION_QUERY`)."""
    if not cache.exists():
        req = urllib.request.Request(
            url,
            data=urllib.parse.urlencode({"data": query or OVERPASS_WAYS_QUERY}).encode(),
            headers={"User-Agent": "tranpy-grid/0.1 (research)"},
        )
        with urllib.request.urlopen(req, timeout=900) as resp:
            body = resp.read()
        json.loads(body)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(body)
    return cache


HVDC_NAME = re.compile(
    r"a-?nord|s(?:ü|ue|u)d-?(?:ost-?)?link|ultranet|nord-?link|neu-?connect|hg(?:ü|ue)|hvdc"
    r"|gleichstrom|borwin|dolwin|sylwin|helwin|baltwin",
    re.IGNORECASE,
)
"""Names of German / cross-border HVDC links (A-Nord, SuedLink, SuedOstLink, Ultranet,
NordLink, ...): a way named only after them carries no AC circuit."""


def is_hvdc_way(tags: Mapping[str, Any]) -> bool:
    """A DC-only way: every ``frequency`` value 0, or every ``;`` part of its name an HVDC
    link (not an AC ``Anbindung`` to its converter). Shared AC/DC towers (``Ultranet;380 kV
    ...``, ``frequency=50;0``) keep their AC parts and are not DC-only."""
    freq = [f.strip() for f in str(tags.get("frequency", "") or "").split(";") if f.strip()]
    if freq and all(f == "0" for f in freq):
        return True
    parts = [t.strip() for t in str(tags.get("name", "") or "").split(";") if t.strip()]
    return bool(parts) and all(
        HVDC_NAME.search(t) and not re.search(r"anbindung|\bkv\b|ltg", t, re.IGNORECASE)
        for t in parts
    )


def _power_kind(tags: Mapping[str, Any]) -> tuple[str, bool]:
    """(``line``/``cable``/``""``, under construction or proposed) of an OSM way's tags."""
    power = str(tags.get("power", ""))
    if power in ("line", "cable"):  # a construction=* note on a mapped line is often stale
        return power, False
    for key in ("construction:power", "proposed:power", "construction", "proposed"):
        value = str(tags.get(key, ""))
        if power in ("construction", "proposed", "") and value in ("line", "cable"):
            return value, True
    return "", False


def osm_ways_strict(
    paths: Sequence[Path], construction: Sequence[Path] = ()
) -> list[dict[str, Any]]:
    """220/380 kV ways with real node ids from Overpass JSON (``out body; >; out skel`` or
    ``out geom`` with ``nodes``); a way id seen twice is kept from the first file. Ways from
    ``construction`` files (or tagged under construction / proposed) get
    ``construction=True``; busbar and bay ways of those files are left out. DC-only ways
    (:func:`is_hvdc_way`: A-Nord, SuedLink, Ultranet, ...) get ``hvdc=True``: they stay in
    the graph (edge ids unchanged) but no AC circuit is routed on them."""
    from tranpy.grid_towers import tower_circuits_220_380

    out, seen = [], set()
    building = {str(p) for p in construction}
    for path in [*paths, *construction]:
        elements = json.loads(Path(path).read_text(encoding="utf-8"))["elements"]
        coord = {int(e["id"]): (e["lon"], e["lat"]) for e in elements if e["type"] == "node"}
        for w in elements:
            tags = w.get("tags", {})
            if w.get("type") != "way" or int(w["id"]) in seen:
                continue
            kind, planned = _power_kind(tags)
            if not kind:
                continue
            planned = planned or str(path) in building
            if planned and tags.get("line") in ("busbar", "bay"):
                continue
            kvs = _kv_set(tags.get("voltage", ""))
            if not kvs:
                continue
            nodes = [int(n) for n in w.get("nodes", [])]
            if "geometry" in w:
                pts = [(q["lon"], q["lat"]) for q in w["geometry"]]
            else:
                pts = [coord.get(n) for n in nodes]
            if len(nodes) < 2 or any(p is None for p in pts) or len(pts) != len(nodes):
                continue
            seen.add(int(w["id"]))
            volts, circ, cab = tags.get("voltage", ""), tags.get("circuits"), tags.get("cables")
            out.append(
                {
                    "id": int(w["id"]),
                    "coords": pts,
                    "nodes": nodes,
                    "kvs": kvs,
                    "circuits_tag": tower_circuits_220_380(volts, circ, cab),
                    "circuits_max": tower_circuits_220_380(volts, circ, cab, upper=True),
                    "ref": tags.get("ref", ""),
                    "name": tags.get("name", ""),
                    "construction": planned,
                    "hvdc": is_hvdc_way(tags),
                }
            )
    return out


def fetch_relations(cache: Path, url: str = OVERPASS_URL) -> dict[str, set[int]]:
    """OSM ``route=power`` relations in Germany: token -> member way ids.

    Tokens are the numbers and the folded words of 4+ letters in the relation ref and name
    (circuit numbers, line codes, station names).

    Cached as the raw Overpass JSON in ``cache``; an empty dict when it cannot be fetched.
    """
    if not cache.exists():
        query = (
            '[out:json][timeout:300];area["ISO3166-1"="DE"]->.a;'
            "relation[route=power](area.a);out body;"
        )
        req = urllib.request.Request(
            url,
            data=urllib.parse.urlencode({"data": query}).encode(),
            headers={"User-Agent": "tranpy-grid/0.1 (research)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=330) as resp:
                body = resp.read()
            json.loads(body)
        except Exception:
            return {}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(body)
    out: dict[str, set[int]] = {}
    for rel in json.loads(cache.read_text(encoding="utf-8")).get("elements", []):
        tags = rel.get("tags", {})
        ways = {int(m["ref"]) for m in rel.get("members", []) if m.get("type") == "way"}
        words = set(re.findall(r"[a-z]{4,}", fold(f"{tags.get('ref', '')} {tags.get('name', '')}")))
        for tok in _tokens(tags.get("ref"), tags.get("name")) | words:
            out.setdefault(tok, set()).update(ways)
    return out


def model_inputs(model: Path, stations: Path) -> tuple[list[Circuit], pd.DataFrame, dict]:
    """In-service AC circuits, terminals and footprints from the pinned model and stations."""
    shapely = __import__("shapely")
    from tranpy.grid_gis import load_model

    _, cim = load_model(model)
    bus = cim.busbar_section.set_index("mrid")
    a = cim.ac_line_segment
    a = a[a["status"] == "in_operation"]
    circuits = []
    for r in a.itertuples(index=False):
        route = [(q["lon"], q["lat"]) for q in r.route] if r.route is not None else []
        circuits.append(
            Circuit(
                mrid=r.mrid,
                name=r.name,
                kv=int(r.nominal_voltage_kv),
                bus1=bus.at[r.bus1_mrid, "name"],
                bus2=bus.at[r.bus2_mrid, "name"],
                model_km=float(r.length_km),
                route=route,
                route_source=str(r.route_source or ""),
            )
        )
    used = {c.bus1 for c in circuits} | {c.bus2 for c in circuits}
    bus_xy = dict(
        zip(
            cim.busbar_section["name"],
            zip(cim.busbar_section["lon"], cim.busbar_section["lat"], strict=True),
            strict=True,
        )
    )
    ends: dict[str, list[tuple[float, float]]] = {}
    for c in circuits:  # mapped route ends: where the circuit really meets each bus
        if len(c.route) < 2 or c.route_source == "schematic":
            continue
        p0, p1 = np.asarray(c.route[0]), np.asarray(c.route[-1])
        b1 = np.asarray(bus_xy[c.bus1])
        if np.hypot(*(p0 - b1)) > np.hypot(*(p1 - b1)):
            p0, p1 = p1, p0
        ends.setdefault(c.bus1, []).append(tuple(p0))
        ends.setdefault(c.bus2, []).append(tuple(p1))
    b = cim.busbar_section[cim.busbar_section["name"].isin(used)]
    terminals = pd.DataFrame(
        {
            "bus": b["name"],
            "kind": b["kind"],
            "station": b["name"].str.split("|").str[0],
            "kv": b["name"].str.split("|").str[1].astype(int),
            "lon": b["lon"],
            "lat": b["lat"],
        }
    ).reset_index(drop=True)
    med = {k: np.median(np.asarray(v), axis=0) for k, v in ends.items()}
    terminals["route_lon"] = terminals["bus"].map(lambda x: med[x][0] if x in med else np.nan)
    terminals["route_lat"] = terminals["bus"].map(lambda x: med[x][1] if x in med else np.nan)
    st = pd.read_parquet(stations / "stations.parquet")
    polys = {}
    for s, wkb in zip(st["station"], st["geometry"], strict=True):
        g = shapely.from_wkb(wkb)
        if g is not None and g.geom_type in ("Polygon", "MultiPolygon"):
            polys[s] = g
    for s, lon, lat, levels in zip(
        st["station"], st["lon"], st["lat"], st["n_levels"], strict=True
    ):
        if s not in polys and np.isfinite(lon) and np.isfinite(lat):
            polys[s] = fallback_box(lon, lat, int(levels or 1))
    return circuits, terminals, polys


def fallback_box(lon: float, lat: float, levels: int) -> Any:
    """The fixed station box of :func:`tranpy.grid_stations._box` (``BOX_M``) in lon/lat."""
    from tranpy.grid_stations import BOX_M

    shapely = __import__("shapely")
    w = BOX_M[0] / 2.0 / (_KM[0] * 1000.0)
    h = BOX_M[1] * (max(levels, 1) + 1) / 2.0 / (_KM[1] * 1000.0)
    return shapely.box(lon - w, lat - h, lon + w, lat + h)


def _write(graph: Graph, routes, check, ranked, out: Path) -> None:
    shapely = __import__("shapely")
    out.mkdir(parents=True, exist_ok=True)
    e = graph.edges.copy()
    e["geometry"] = [
        shapely.to_wkb(shapely.linestrings(graph.coords[int(i)])) for i in e["edge_id"]
    ]
    e.to_parquet(out / "edges.parquet", index=False)
    graph.nodes.to_parquet(out / "nodes.parquet", index=False)
    graph.terminals.to_parquet(out / "terminals.parquet", index=False)
    routes.to_parquet(out / "circuit_routes.parquet", index=False)
    check.to_csv(out / "edge_check.csv", index=False)
    ranked.to_csv(out / "findings_ranked.csv", index=False)
    clean_findings(ranked).to_csv(out / "findings_clean.csv", index=False)


def main(argv: Sequence[str] | None = None) -> None:
    home = Path.home() / "data/grid"
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", type=Path, default=home / "v5/frozen/model_v5_frozen.pkl")
    p.add_argument("--stations", type=Path, default=home / "v5/stations")
    p.add_argument("--lines", type=Path, default=home / "osm/hv_lines.parquet")
    p.add_argument("--overpass", type=Path, nargs="*", default=[home / "osm/overpass_at_west.json"])
    p.add_argument("--out", type=Path, default=home / "v5/network")
    p.add_argument("--no-relations", action="store_true")
    p.add_argument("--no-joins", action="store_true", help="strict graph without join edges")
    p.add_argument(
        "--no-construction",
        action="store_true",
        help="leave out the 220/380 kV ways under construction / proposed (routing fallback)",
    )
    p.add_argument(
        "--overrides",
        type=Path,
        default=home / "v5/network/route_overrides.csv",
        help="CSV mrid,status,reason: hand-set route statuses (see apply_overrides)",
    )
    p.add_argument(
        "--join-overrides",
        type=Path,
        default=home / "v5/network/join_overrides.csv",
        help="CSV node_a,node_b,reason: reviewed join edges (see add_reviewed_joins)",
    )
    p.add_argument("--osm-substations", type=Path, default=home / "osm/DEU.gpkg")
    p.add_argument(
        "--legacy",
        action="store_true",
        help="round-4 graph: export ways joined by rounded coordinates plus proximity bridges",
    )
    args = p.parse_args(argv)
    t0 = time.time()
    sha = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA:
        raise SystemExit(f"model sha {sha} is not the pinned {EXPECTED_SHA}")
    circuits, terminals, polys = model_inputs(args.model, args.stations)
    if args.legacy:
        ways = osm_ways(args.lines, args.overpass)
    else:
        cache = fetch_ways(args.out / "cache/overpass_de_ways_nodes.json")
        building = []
        if not args.no_construction:
            building = [
                fetch_ways(
                    args.out / "cache/overpass_de_construction_ways.json",
                    query=OVERPASS_CONSTRUCTION_QUERY,
                )
            ]
        ways = osm_ways_strict([cache, *args.overpass], building)
    zones = {}
    if not args.legacy and args.osm_substations.exists():
        from tranpy.grid_stations import read_osm_footprints

        osm = read_osm_footprints(args.osm_substations)
        zones = {f"osm:{i}": g for i, g in zip(osm["osm_id"], osm.geometry, strict=True)}
    graph = build_graph(
        ways,
        terminals,
        polys,
        strict=not args.legacy,
        joins=not args.no_joins,
        join_zones=zones,
        extra_joins=read_join_overrides(args.join_overrides),
    )
    t1 = time.time()
    rel = {} if args.no_relations else fetch_relations(args.out / "cache/overpass_de_routes.json")
    routes = route_circuits(graph, circuits, rel, detour_overrides(args.overrides))
    t2 = time.time()
    pieces = routes.attrs.pop("pieces")
    routes, pieces = apply_overrides(routes, pieces, args.overrides)
    check = edge_check(graph, routes)
    ranked = findings(check, graph)
    _write(graph, routes, check, ranked, args.out)
    pieces.to_parquet(args.out / "circuit_pieces.parquet", index=False)
    bad = self_check(graph, routes, pieces)
    bad.to_csv(args.out / "self_check.csv", index=False)
    meta = {
        "model_sha": sha,
        "ways": len(ways),
        "edges": int((graph.edges["kind"] == "line").sum()),
        "gap_edges": int((graph.edges["kind"] == "gap").sum()),
        "gap_station": int(
            ((graph.edges["kind"] == "gap") & (graph.edges["gap_where"] == "station")).sum()
        ),
        "gap_midline": int(
            ((graph.edges["kind"] == "gap") & (graph.edges["gap_where"] == "midline")).sum()
        ),
        "long_gap_edges": int((graph.edges["kind"] == "long_gap").sum()),
        "link_edges": int((graph.edges["kind"] == "link").sum()),
        "link_midline": int(
            ((graph.edges["kind"] == "link") & (graph.edges["gap_where"] == "midline")).sum()
        ),
        "relations_tokens": len(rel),
        "strict": not args.legacy,
        "self_check_violations": len(bad),
        "join_edges": int((graph.edges["kind"] == "join").sum()),
        "graph_s": round(t1 - t0, 1),
        "routing_s": round(t2 - t1, 1),
        "total_s": round(time.time() - t0, 1),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))  # noqa: T201


if __name__ == "__main__":
    main()
