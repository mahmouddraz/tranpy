"""Ordered bundles of parallel model circuits along OpenStreetMap edges (display only).

Circuits that share towers are drawn side by side. The order of the circuits on each edge
is chosen once for the whole network so that

- a circuit keeps one slot along a *corridor* (a maximal run of edges with the same set of
  circuits, all passing straight through);
- slots change only where a circuit joins or leaves the bundle;
- unavoidable crossings sit at a junction, never in the middle of a corridor.

Method
------
This is the pairwise "follow both lines until they split" ordering used for the metro-line
crossing minimisation problem (MLCM): for two lines sharing an edge, walk along their common
path in both directions until they part; at the split node, the line that turns further to
the left of the travel direction goes on the left. Lines that never part (same start and end
nodes) fall back to a fixed key, measured in a direction fixed by the two end nodes, so the
fallback does not flip along the run. When the two splits disagree, the crossing is
unavoidable and the split whose node key sorts first wins, which puts the crossing at the
other split node, for every edge of the run alike. The pair verdicts on an edge are ranked by
Copeland score (number of pairs won).

References: Benkert, Nöllenburg, Uno, Wolff, "Minimizing intra-edge crossings in wiring
diagrams and public transportation maps", GD 2006 (the MLCM problem); Asquith, Gudmundsson,
Merrick, "An ILP for the metro-line crossing problem", CATS 2008; Bekos, Kaufmann, Potika,
Symvonis, "Line crossing minimization on metro maps", GD 2007; Nöllenburg, "An improved
algorithm for the metro-line crossing minimization problem", GD 2009 (lines ordered by the
order of their continuations at the split ends, crossings placed at the ends of a common
subpath); Pupyrev, Nachmanson, Bereg, Holroyd, "Edge routing with ordered bundles", GD 2011
(the same pairwise comparison on a routing graph); Bast, Brosi, Storandt, "Efficient
generation of geographically accurate transit maps", SIGSPATIAL 2018 (LOOM, not used here).

Drawing
-------
Each circuit becomes one LineString per contiguous part of its route, with its offset baked
in (metres, positive = left of the travel direction). The offset is constant along each edge
and ramps linearly over :data:`TRANSITION_M` across the node where it changes, so the legs of
a junction meet without gaps.

Inputs (agent A's ``grid_network`` output, or :func:`stub_from_export` until it exists):

- ``graph``: GeoDataFrame of edges, one row per edge, columns ``edge_id``, ``u``, ``v``
  (node ids; the geometry runs from ``u`` to ``v``), ``geometry`` (LineString, any CRS).
  If ``u`` / ``v`` are missing they are derived from the rounded end coordinates.
- ``circuit_routes``: DataFrame with ``mrid`` and the ordered edge list in ``edges`` (or
  ``edge_ids``; a list or a ``;``-separated string), or long format ``mrid``, ``seq``,
  ``edge_id``. Optional ``voltage_kv`` and ``name`` are used as the tie-break key.

The model is not changed.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from tranpy_core._optional import require

__all__ = [
    "FAN_M",
    "SPACING_M",
    "TRANSITION_M",
    "bundle_geometries",
    "edge_slots",
    "piece_slots",
    "read_network",
    "route_slots",
    "station_fans",
    "stub_from_export",
]

METRIC_CRS = "EPSG:3035"
SPACING_M = 90.0
"""Default distance between neighbouring circuits (about 7.6 px at zoom 13)."""
TRANSITION_M = 50.0
"""Length over which a circuit moves to its new slot at a node."""
_MAX_WALK = 100_000
ZONE_M = 500.0
"""Station zone buffer (at least), see :func:`station_fans`."""
ZONE_MAX_M = 1000.0
"""Station zone buffer (at most)."""
GROUP_M = 150.0
"""Cut points this close (and with the same heading) enter a station as one group."""
ENTRY_GAP_M = 25.0
"""Least distance between entry slots on a station outline."""
FREEZE_M = 400.0
"""A station fan starts this far before the zone edge (group order frozen there)."""
MIN_PIECE_M = 100.0
"""Pieces shorter than this take their neighbours' slot (:func:`route_slots`)."""
JOIN_PARTS_M = 200.0
"""Consecutive route parts of one circuit closer than this are joined."""
FAN_M = 300.0
"""Length over which a bundle fans in onto the entry points at a station outline."""
PARALLEL_M = 100.0
"""Bundles on ways closer than this are drawn as one bundle."""
PARALLEL_SHARE = 0.7
"""Share of an edge's sample points that must lie within :data:`PARALLEL_M` of the other."""
JOIN_ROUTE_M = 25.0
"""Stub: consecutive runs of one circuit whose ends are this close share a node."""

Line = tuple[Any, int]
"""A contiguous route part: ``(mrid, part)``."""


# ---------------------------------------------------------------- input normalisation


def _routes_table(circuit_routes: pd.DataFrame) -> dict[Any, list[Any]]:
    df = circuit_routes
    if "edge_id" in df.columns and "edges" not in df.columns and "edge_ids" not in df.columns:
        key = "seq" if "seq" in df.columns else None
        df = df.sort_values(["mrid", key]) if key else df
        return {m: list(g["edge_id"]) for m, g in df.groupby("mrid", sort=False)}
    col = "edges" if "edges" in df.columns else "edge_ids"
    out: dict[Any, list[Any]] = {}
    for m, e in zip(df["mrid"], df[col], strict=True):
        if isinstance(e, str):
            e = [x for x in e.split(";") if x]
        out[m] = list(e) if e is not None else []
    return out


def _route_keys(circuit_routes: pd.DataFrame) -> dict[Any, tuple[float, str]]:
    first = circuit_routes.drop_duplicates("mrid")
    kcol = next((c for c in ("voltage_kv", "kv") if c in first.columns), None)
    kv = first[kcol] if kcol else pd.Series(0.0, first.index)
    name = first["name"] if "name" in first.columns else first["mrid"]
    return {m: (-float(k or 0.0), str(n)) for m, k, n in zip(first["mrid"], kv, name, strict=True)}


def _node_ids(graph: Any) -> tuple[dict[Any, Hashable], dict[Any, Hashable]]:
    eids = list(graph["edge_id"])
    if "u" in graph.columns and "v" in graph.columns:
        return dict(zip(eids, graph["u"], strict=True)), dict(zip(eids, graph["v"], strict=True))
    u, v = {}, {}
    for e, g in zip(eids, graph.geometry, strict=True):
        c = list(g.coords)
        u[e] = (round(c[0][0], 0), round(c[0][1], 0))
        v[e] = (round(c[-1][0], 0), round(c[-1][1], 0))
    return u, v


def _orient(
    route: Sequence[Any], eu: dict[Any, Hashable], ev: dict[Any, Hashable]
) -> list[list[tuple[Any, bool]]]:
    """Split a route into contiguous parts of ``(edge, forward)``."""
    route = [e for e in route if e in eu]
    parts: list[list[tuple[Any, bool]]] = []
    cur: list[tuple[Any, bool]] = []
    for i, e in enumerate(route):
        ends = {eu[e], ev[e]}
        if cur:
            prev_e, prev_f = cur[-1]
            at = ev[prev_e] if prev_f else eu[prev_e]
            if at in ends:
                cur.append((e, eu[e] == at))
                continue
            parts.append(cur)
            cur = []
        nxt = route[i + 1] if i + 1 < len(route) else None
        fwd = True
        if nxt is not None:
            nends = {eu[nxt], ev[nxt]}
            fwd = ev[e] in nends or eu[e] not in nends
        cur.append((e, fwd))
    if cur:
        parts.append(cur)
    return parts


# ---------------------------------------------------------------- geometry helpers


def _heading_in(coords: np.ndarray, fwd: bool) -> np.ndarray:
    """Unit direction of travel arriving at the far end of an edge."""
    c = coords if fwd else coords[::-1]
    return _unit(_far_point(c[::-1]), c[-1])


def _heading_out(coords: np.ndarray, fwd: bool) -> np.ndarray:
    """Unit direction of travel leaving the near end of an edge."""
    c = coords if fwd else coords[::-1]
    return _unit(c[0], _far_point(c))


def _far_point(c: np.ndarray, reach: float = 60.0) -> np.ndarray:
    """The first vertex at least ``reach`` from ``c[0]`` (or the last one)."""
    d = np.hypot(*(c - c[0]).T)
    k = np.nonzero(d >= reach)[0]
    return c[k[0]] if len(k) else c[-1]


def _unit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.asarray(b, float) - np.asarray(a, float)
    n = float(np.hypot(*d))
    return d / n if n > 0 else np.array([1.0, 0.0])


def _turn(h_in: np.ndarray, h_out: np.ndarray) -> float:
    """Signed turn angle (radians, left positive) from ``h_in`` to ``h_out``."""
    return math.atan2(h_in[0] * h_out[1] - h_in[1] * h_out[0], float(h_in @ h_out))


# ---------------------------------------------------------------- ordering


class _Orderer:
    def __init__(
        self,
        parts: dict[Line, list[tuple[Any, bool]]],
        coords: dict[Any, np.ndarray],
        eu: dict[Any, Hashable],
        ev: dict[Any, Hashable],
        keys: dict[Line, tuple],
        short_m: float = 2 * TRANSITION_M,
        ends: dict[tuple[Line, str], np.ndarray] | None = None,
    ) -> None:
        self.parts, self.coords, self.eu, self.ev, self.keys = parts, coords, eu, ev, keys
        self.short_m = short_m
        self.ends: dict[tuple[Line, str], np.ndarray] = ends or {}
        self.length = {e: float(np.hypot(*np.diff(c, axis=0).T).sum()) for e, c in coords.items()}
        self.on_edge: dict[Any, dict[Line, int]] = defaultdict(dict)
        for ln, route in parts.items():
            for i, (e, _) in enumerate(route):
                self.on_edge[e].setdefault(ln, i)

    def _walk(self, e: Any, a: Line, b: Line, toward_v: bool) -> tuple[Hashable, int | None]:
        """Follow ``a`` and ``b`` away from ``e`` until they part.

        Returns the node where they part (or end) and +1 when ``a`` lies left of ``b`` in
        the ``u -> v`` direction of ``e``, -1 for right, ``None`` when undecided.
        """
        ra, rb = self.parts[a], self.parts[b]
        ia, ib = self.on_edge[e][a], self.on_edge[e][b]
        sa = 1 if ra[ia][1] == toward_v else -1
        sb = 1 if rb[ib][1] == toward_v else -1
        cur, cur_travel_fwd = e, toward_v
        node = self.ev[e] if toward_v else self.eu[e]
        for _ in range(_MAX_WALK):
            ka, kb = ia + sa, ib + sb
            end_a, end_b = not (0 <= ka < len(ra)), not (0 <= kb < len(rb))
            if end_a or end_b:
                return node, self._end_split(a, b, cur, cur_travel_fwd, toward_v, (sa, sb), ka, kb)
            (ea, fa), (eb, fb) = ra[ka], rb[kb]
            fa, fb = fa == (sa > 0), fb == (sb > 0)  # travel direction away from e
            if ea == eb and fa == fb:
                ia, ib, cur, cur_travel_fwd = ka, kb, ea, fa
                node = self.ev[ea] if fa else self.eu[ea]
                continue
            ja, jb = ka + sa, kb + sb
            if (
                self.length[ea] < self.short_m
                and self.length[eb] < self.short_m
                and 0 <= ja < len(ra)
                and 0 <= jb < len(rb)
                and ra[ja][0] == rb[jb][0]
            ):
                # both take a short detour and meet again: not a split
                fj = ra[ja][1] == (sa > 0)
                ia, ib, cur, cur_travel_fwd = ja, jb, ra[ja][0], fj
                node = self.ev[cur] if fj else self.eu[cur]
                continue
            h = _heading_in(self.coords[cur], cur_travel_fwd)
            ta = _turn(h, _heading_out(self.coords[ea], fa))
            tb = _turn(h, _heading_out(self.coords[eb], fb))
            if abs(ta - tb) < 1e-9:
                return node, None
            left = 1 if ta > tb else -1
            return node, left if toward_v else -left
        return node, None

    def _end_split(
        self,
        a: Line,
        b: Line,
        cur: Any,
        cur_fwd: bool,
        toward_v: bool,
        steps: tuple[int, int],
        ka: int,
        kb: int,
    ) -> int | None:
        """Order two lines where at least one ends: an ending line heads for its station
        entry point (``ends``); a continuing one for its next edge."""
        c = self.coords[cur] if cur_fwd else self.coords[cur][::-1]
        at = c[-1]
        h = _heading_in(self.coords[cur], cur_fwd)
        turns = []
        for ln, k, st in ((a, ka, steps[0]), (b, kb, steps[1])):
            r = self.parts[ln]
            if 0 <= k < len(r):
                e2, f2 = r[k]
                turns.append(_turn(h, _heading_out(self.coords[e2], f2 == (st > 0))))
                continue
            t = self.ends.get((ln, "end" if st > 0 else "start"))
            if t is None or float(np.hypot(*(t - at))) < 1.0:
                return None
            turns.append(_turn(h, _unit(at, t)))
        if abs(turns[0] - turns[1]) < 1e-9:
            return None
        left = 1 if turns[0] > turns[1] else -1
        return left if toward_v else -left

    def compare(self, e: Any, a: Line, b: Line) -> int:
        nf, vf = self._walk(e, a, b, True)
        nb, vb = self._walk(e, a, b, False)
        if vf is not None and vb is not None and vf != vb:
            return vf if str(nf) <= str(nb) else vb
        if vf is not None:
            return vf
        if vb is not None:
            return vb
        # never part: fixed key, read in the direction from the smaller end-node key
        sign = 1 if str(nb) <= str(nf) else -1
        return sign * (1 if self.keys[a] < self.keys[b] else -1)

    def order(self, e: Any) -> list[Line]:
        """Lines on ``e`` from left to right in its ``u -> v`` direction."""
        lines = sorted(self.on_edge[e], key=lambda ln: self.keys[ln])
        if len(lines) < 2:
            return lines
        wins = dict.fromkeys(lines, 0)
        for i, a in enumerate(lines):
            for b in lines[i + 1 :]:
                if self.compare(e, a, b) > 0:
                    wins[a] += 1
                else:
                    wins[b] += 1
        return sorted(lines, key=lambda ln: (-wins[ln], self.keys[ln]))


def _corridors(o: _Orderer) -> dict[Any, tuple[Any, bool]]:
    """Map each edge to (representative edge, same orientation as it).

    Two edges are in one corridor when they carry the same lines and every one of them
    passes directly from one to the other.
    """
    trans: dict[tuple[Any, Any], int] = defaultdict(int)
    link: dict[tuple[Any, Any], bool] = {}
    for route in o.parts.values():
        for (e1, f1), (e2, f2) in itertools.pairwise(route):
            if e1 == e2:
                continue
            trans[(e1, e2)] += 1
            trans[(e2, e1)] += 1
            link[(e1, e2)] = link[(e2, e1)] = f1 == f2
    adj: dict[Any, list[tuple[Any, bool]]] = defaultdict(list)
    for (e1, e2), n in trans.items():
        s1, s2 = o.on_edge[e1], o.on_edge[e2]
        if n == len(s1) == len(s2) and set(s1) == set(s2):
            adj[e1].append((e2, link[(e1, e2)]))
    rep: dict[Any, tuple[Any, bool]] = {}
    for e in o.on_edge:
        if e in rep:
            continue
        rep[e] = (e, True)
        stack = [e]
        while stack:
            x = stack.pop()
            for y, same in adj[x]:
                if y not in rep:
                    rep[y] = (e, rep[x][1] == same)
                    stack.append(y)
    return rep


def _prepare(
    graph: Any, circuit_routes: pd.DataFrame, short_m: float = 2 * TRANSITION_M
) -> tuple[Any, _Orderer, dict]:
    g = graph.reset_index(drop=True)
    if g.crs is not None and g.crs.is_geographic:
        g = g.to_crs(METRIC_CRS)
    eu, ev = _node_ids(g)
    coords = {
        e: np.asarray(geom.coords, float) for e, geom in zip(g["edge_id"], g.geometry, strict=True)
    }
    keys = _route_keys(circuit_routes)
    parts: dict[Line, list[tuple[Any, bool]]] = {}
    for m, route in _routes_table(circuit_routes).items():
        for p, part in enumerate(_orient(route, eu, ev)):
            parts[(m, p)] = part
    lkeys = {ln: (*keys.get(ln[0], (0.0, str(ln[0]))), str(ln[0]), ln[1]) for ln in parts}
    clipped = _clipped(circuit_routes, g.crs)
    last = {}
    for m, p in parts:
        last[m] = max(last.get(m, 0), p)
    ends = {}
    for m, (cg, *_) in clipped.items():
        if m in last:
            ends[((m, 0), "start")] = np.asarray(cg.coords[0], float)
            ends[((m, last[m]), "end")] = np.asarray(cg.coords[-1], float)
    o = _Orderer(parts, coords, eu, ev, lkeys, short_m, ends)
    o.clipped = clipped
    o.last = last
    order: dict[Any, list[Line]] = {}
    rep = _corridors(o)
    by_rep: dict[Any, list[Line]] = {}
    for e in o.on_edge:
        r, same = rep[e]
        if r not in by_rep:
            by_rep[r] = o.order(r)
        order[e] = by_rep[r] if same else by_rep[r][::-1]
    return g, o, order


def route_slots(
    graph: Any, circuit_routes: pd.DataFrame, min_piece_m: float = MIN_PIECE_M
) -> pd.DataFrame:
    """Slot of every circuit on every edge of its route, for a pixel ``line-offset``.

    The order is the global one of :func:`edge_slots` (one order per corridor, changes
    only where the circuit set changes). Along each route, an edge shorter than
    ``min_piece_m`` takes the slot of its neighbours (the previous one; the next one at
    the route start), so short pieces do not jog.

    Returns:
        DataFrame ``mrid``, ``part``, ``seq`` (position in the route), ``edge_id``,
        ``forward`` (route runs u -> v), ``slot`` (0 = leftmost in the edge's u -> v
        direction), ``n`` (circuits on the edge, after smoothing), ``length_m``.
    """
    _, o, order = _prepare(graph, circuit_routes, 2 * TRANSITION_M)
    slot = {(e, ln): (k, len(lines)) for e, lines in order.items() for k, ln in enumerate(lines)}
    rows = []
    for ln, route in o.parts.items():
        # left offset in travel direction, in slot units, per edge
        left = []
        for e, fwd in route:
            k, n = slot[(e, ln)]
            v = (n - 1) / 2 - k
            left.append((v if fwd else -v, n))
        lengths = [o.length[e] for e, _ in route]
        for i in range(len(route)):
            if lengths[i] >= min_piece_m or len(route) == 1:
                continue
            j = (
                i - 1
                if i > 0
                else next((q for q in range(1, len(route)) if lengths[q] >= min_piece_m), None)
            )
            if j is not None:
                left[i] = left[j]
        for i, ((e, fwd), (v, n)) in enumerate(zip(route, left, strict=True)):
            v_uv = v if fwd else -v
            rows.append(
                {
                    "mrid": ln[0],
                    "part": ln[1],
                    "seq": i,
                    "edge_id": e,
                    "forward": fwd,
                    "slot": round((n - 1) / 2 - v_uv),
                    "n": n,
                    "length_m": lengths[i],
                }
            )
    return pd.DataFrame(rows)


def piece_slots(graph: Any, circuit_routes: pd.DataFrame, pieces: Any) -> Any:
    """Give every drawn piece its ``slot`` and ``n`` for a MapLibre pixel offset.

    ``pieces`` is ``grid_network``'s ``circuit_pieces`` (``mrid``, ``seq``, ``edge_id``,
    ``kind``, ``geometry``; any CRS). The geometry is not changed. For a piece drawn in
    the direction of its own geometry, ``line-offset = (slot - (n - 1) / 2) * spacing_px``
    (MapLibre: positive to the right) puts every circuit in its corridor slot. Connectors
    (``edge_id`` -1, the short end to the station outline) and ``join`` pieces (bridges over
    OSM tagging breaks) keep the slot of the piece next to them, so a line keeps its
    corridor slot up to the box.

    Returns ``pieces`` with ``slot`` and ``n`` (int).
    """
    rs = route_slots(graph, circuit_routes)
    rs = rs.drop_duplicates(["mrid", "edge_id"]).set_index(["mrid", "edge_id"])
    pc = pieces.sort_values(["mrid", "seq"]).reset_index(drop=True)
    xy = [np.asarray(g.coords, float) for g in pc.geometry]
    slots, ns = [0] * len(pc), [1] * len(pc)
    for _m, idx in pc.groupby("mrid", sort=False).groups.items():
        idx = list(idx)
        # travel direction of each piece: its end is nearer the next piece
        along = []
        for q, i in enumerate(idx):
            c = xy[i]
            nb = xy[idx[q + 1]] if q + 1 < len(idx) else None
            pv = xy[idx[q - 1]] if q > 0 else None
            if nb is not None:
                d_end = min(np.hypot(*(c[-1] - nb[0])), np.hypot(*(c[-1] - nb[-1])))
                d_start = min(np.hypot(*(c[0] - nb[0])), np.hypot(*(c[0] - nb[-1])))
                along.append(bool(d_end <= d_start))
            elif pv is not None:
                d_start = min(np.hypot(*(c[0] - pv[0])), np.hypot(*(c[0] - pv[-1])))
                d_end = min(np.hypot(*(c[-1] - pv[0])), np.hypot(*(c[-1] - pv[-1])))
                along.append(bool(d_start <= d_end))
            else:
                along.append(True)
        # travel-left position per piece (slot units)
        vt: list[tuple[float, int] | None] = []
        for i in idx:
            key = (pc.at[i, "mrid"], pc.at[i, "edge_id"])
            kind = str(pc.at[i, "kind"]) if "kind" in pc.columns else "way"
            if kind in ("connector", "join") or key not in rs.index:
                # connectors, joins over tagging breaks and unknown edges
                vt.append(None)
                continue
            r = rs.loc[key]
            v = (r["n"] - 1) / 2 - r["slot"]
            vt.append((v if r["forward"] else -v, int(r["n"])))
        for q in range(len(idx)):
            if vt[q] is None:
                near = [vt[k] for k in (q - 1, q + 1) if 0 <= k < len(idx) and vt[k]]
                vt[q] = near[0] if near else (0.0, 1)
        for q, i in enumerate(idx):
            v, n = vt[q]
            v_left = v if along[q] else -v
            slots[i], ns[i] = round((n - 1) / 2 - v_left), n
    return pc.assign(slot=slots, n=ns)


def _clipped(circuit_routes: pd.DataFrame, crs: Any) -> dict[Any, tuple[Any, float, float]]:
    """mrid -> (clipped route in ``crs``, start connector m, end connector m)."""
    if "geometry" not in circuit_routes.columns:
        return {}
    gpd = require("geopandas", "grid")
    shapely = require("shapely", "grid")
    geo = circuit_routes["geometry"]
    if len(geo) and isinstance(geo.iloc[0], bytes | bytearray):
        geo = shapely.from_wkb(geo.to_numpy())
    src = getattr(circuit_routes, "crs", None) or "EPSG:4326"
    gs = gpd.GeoSeries(list(geo), crs=src)
    if crs is not None:
        gs = gs.to_crs(crs)
    out = {}
    for k, (m, cg) in enumerate(zip(circuit_routes["mrid"], gs, strict=True)):
        if cg is None or cg.is_empty or cg.geom_type != "LineString":
            continue
        gaps = []
        for i in (1, 2):
            kind = circuit_routes.get(f"entry{i}_kind")
            gap = circuit_routes.get(f"entry{i}_gap_m")
            near = kind is not None and kind.iloc[k] == "outline_near"
            gaps.append(float(gap.iloc[k]) if near and gap is not None else 0.0)
        fans = []
        for i in (1, 2):
            kind = circuit_routes.get(f"entry{i}_kind")
            fans.append(kind is not None and kind.iloc[k] in ("outline", "outline_near"))
        out[m] = (cg, gaps[0], gaps[1], fans[0], fans[1])
    return out


def edge_slots(graph: Any, circuit_routes: pd.DataFrame) -> pd.DataFrame:
    """The slot of every circuit on every edge.

    Returns:
        DataFrame ``edge_id``, ``mrid``, ``part``, ``slot`` (0 = leftmost in the edge's
        ``u -> v`` direction), ``n`` (circuits on the edge).
    """
    _, _, order = _prepare(graph, circuit_routes)
    rows = [
        {"edge_id": e, "mrid": ln[0], "part": ln[1], "slot": k, "n": len(lines)}
        for e, lines in order.items()
        for k, ln in enumerate(lines)
    ]
    return pd.DataFrame(rows, columns=["edge_id", "mrid", "part", "slot", "n"])


def _offset_line(
    pieces: list[tuple[np.ndarray, float]], transition_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Join oriented edge coordinates and shift them left by a per-edge offset.

    ``pieces``: (coords in travel order, left offset in metres) per edge. Returns the
    offset coordinates, their arc length along the centreline and their offsets.
    """
    pts: list[np.ndarray] = []
    vals: list[float] = []
    for c, off in pieces:
        if pts and np.allclose(pts[-1][-1], c[0]):
            c = c[1:]
        if not len(c):
            continue
        pts.append(c)
        vals.append(off)
    xy = np.vstack(pts)
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    # edge boundaries in arc length
    ends = []
    n = 0
    for chunk in pts:
        n += len(chunk)
        ends.append(s[min(n, len(s)) - 1])
    # breakpoints of the offset profile: ramps of transition_m centred on changes
    bx = [0.0]
    by = [vals[0]]
    for k in range(1, len(vals)):
        b = ends[k - 1]
        if vals[k] == vals[k - 1]:
            continue
        lo = ends[k - 2] if k >= 2 else 0.0
        hi = ends[k] if k < len(ends) else s[-1]
        half = min(transition_m / 2, (b - lo) / 2, (hi - b) / 2)
        a0, a1 = max(b - half, bx[-1]), b + half
        bx += [a0, a1]
        by += [vals[k - 1], vals[k]]
    bx.append(s[-1])
    by.append(vals[-1])
    # insert ramp breakpoints as vertices
    extra = np.array([x for x in bx if 0 < x < s[-1]])
    if len(extra):
        allx = np.unique(np.concatenate([s, extra]))
        xy = np.column_stack([np.interp(allx, s, xy[:, 0]), np.interp(allx, s, xy[:, 1])])
        s = allx
    off = np.interp(s, bx, by)
    keep = np.concatenate([[True], np.diff(s) > 1e-6])
    xy, s, off = xy[keep], s[keep], off[keep]
    if len(xy) < 2:
        return xy, s, off
    d = np.diff(xy, axis=0)
    d /= np.maximum(np.hypot(*d.T), 1e-12)[:, None]
    nrm = np.column_stack([-d[:, 1], d[:, 0]])  # left normals per segment
    vn = np.vstack([nrm[:1], nrm[:-1] + nrm[1:], nrm[-1:]])
    vn /= np.maximum(np.hypot(*vn.T), 1e-12)[:, None]
    # miter length, limited at sharp bends
    cosh = np.ones(len(xy))
    cosh[1:-1] = np.clip((vn[1:-1] * nrm[:-1]).sum(axis=1), 0.5, 1.0)
    return xy + vn * (off / cosh)[:, None], s, off


def _parallel_offsets(
    o: _Orderer, order: dict[Any, list[Line]], spacing_m: float, parallel_m: float
) -> dict[tuple[Any, Line], float]:
    """Left offset (``u -> v`` frame of the edge) of every line on every edge.

    Bundles on OpenStreetMap ways that run side by side (at least :data:`PARALLEL_SHARE`
    of both edges within ``parallel_m`` of each other, no shared node) are stacked into
    one bundle centred between the ways, each way's lines on its own side.
    """
    shapely = require("shapely", "grid")
    out: dict[tuple[Any, Line], float] = {}
    edges = [e for e in order if o.length[e] >= parallel_m] if parallel_m > 0 else []
    geoms = {e: shapely.LineString(o.coords[e]) for e in edges}
    samples = {
        e: shapely.line_interpolate_point(geoms[e], np.linspace(0.1, 0.9, 9), normalized=True)
        for e in edges
    }

    def share(a: Any, b: Any) -> float:
        return float(np.mean(shapely.distance(geoms[b], samples[a]) <= parallel_m))

    nbrs: dict[Any, list[Any]] = defaultdict(list)
    if edges:
        tree = shapely.STRtree([geoms[e] for e in edges])
        ia, ib = tree.query([geoms[e] for e in edges], predicate="dwithin", distance=parallel_m)
        for i, j in zip(ia, ib, strict=True):
            a, b = edges[i], edges[j]
            if i >= j or {o.eu[a], o.ev[a]} & {o.eu[b], o.ev[b]}:
                continue
            ca, cb = o.coords[a], o.coords[b]
            if abs(float(_unit(ca[0], ca[-1]) @ _unit(cb[0], cb[-1]))) < 0.9:
                continue  # not running the same way (e.g. converging into a station)
            if share(a, b) >= PARALLEL_SHARE and share(b, a) >= PARALLEL_SHARE:
                nbrs[a].append(b)
                nbrs[b].append(a)
    for e, lines in order.items():
        n = len(lines)
        own = [((n - 1) / 2 - k) * spacing_m for k in range(n)]
        if not nbrs.get(e):
            out.update({(e, ln): x for ln, x in zip(lines, own, strict=True)})
            continue
        mid = shapely.line_interpolate_point(geoms[e], 0.5, normalized=True)
        c = o.coords[e]
        h = _unit(c[0], c[-1])
        stack: list[tuple[float, Any, bool]] = [(0.0, e, True)]
        for f in nbrs[e]:
            s_f = shapely.line_locate_point(geoms[f], mid)
            p = shapely.line_interpolate_point(geoms[f], s_f)
            d = np.array([p.x - mid.x, p.y - mid.y])
            lat = float(h[0] * d[1] - h[1] * d[0])  # left of e positive
            cf = o.coords[f]
            stack.append((lat, f, float(h @ _unit(cf[0], cf[-1])) >= 0))
        stack.sort(key=lambda t: t[0])  # right to left
        slots: list[tuple[Any, Line]] = []
        for _, f, aligned in stack:
            fl = order[f] if aligned else order[f][::-1]
            slots += [(f, ln) for ln in reversed(fl)]  # right to left
        total = len(slots)
        centre = (stack[0][0] + stack[-1][0]) / 2
        for j, (f, ln) in enumerate(slots):
            if f == e:
                out[(e, ln)] = centre + (j - (total - 1) / 2) * spacing_m
    return out


def _clip_ends(
    pieces: list[tuple[np.ndarray, float]],
    xy: np.ndarray,
    arc: np.ndarray,
    off: np.ndarray,
    clip: tuple[Any, float, float, bool, bool],
    first: bool,
    last: bool,
    fan_m: float = FAN_M,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Cut an offset line at the station outlines of its clipped route.

    Where the route stops short of a footprint (``outline_near``), the line continues
    straight to the outline point, keeping its offset, so the bundle runs into the box
    edge side by side. At a station end, the last ``fan_m`` then bends smoothly (cubic
    Hermite) onto the circuit's own outline entry point: the bundle fans in, in the
    order it already has, without reordering. Returns the coordinates and the two
    connector lengths (m).
    """
    shapely = require("shapely", "grid")
    cg, gap1, gap2, fan1, fan2 = clip
    chunks = [pieces[0][0]] + [
        c[1:] if np.allclose(pieces[i][0][-1], c[0]) else c for i, (c, _) in enumerate(pieces[1:])
    ]
    centre = shapely.LineString(np.vstack([c for c in chunks if len(c)]))
    s0 = float(centre.project(cg.interpolate(gap1))) if first else 0.0
    s1 = float(centre.project(cg.interpolate(cg.length - gap2))) if last else arc[-1]
    if s1 - s0 < 1.0:
        return xy, (0.0, 0.0)
    inner = (arc > s0) & (arc < s1)
    xs = np.concatenate([[s0], arc[inner], [s1]])
    pts = np.column_stack([np.interp(xs, arc, xy[:, 0]), np.interp(xs, arc, xy[:, 1])])
    o0, o1 = float(np.interp(s0, arc, off)), float(np.interp(s1, arc, off))
    con = [0.0, 0.0]
    if first and gap1 > 0:
        t = np.asarray(cg.coords[0], float)
        q = np.asarray(centre.interpolate(s0).coords[0], float)
        d = _unit(t, q)
        pts = np.vstack([t + np.array([-d[1], d[0]]) * o0, pts])
        con[0] = float(np.hypot(*(q - t)))
    if last and gap2 > 0:
        t = np.asarray(cg.coords[-1], float)
        q = np.asarray(centre.interpolate(s1).coords[0], float)
        d = _unit(q, t)
        pts = np.vstack([pts, t + np.array([-d[1], d[0]]) * o1])
        con[1] = float(np.hypot(*(t - q)))
    if last and fan2 and fan_m > 0:
        pts = _fan(pts, np.asarray(cg.coords[-1], float), _tail_dir(cg, False), fan_m)
    if first and fan1 and fan_m > 0:
        head = _fan(pts[::-1], np.asarray(cg.coords[0], float), _tail_dir(cg, True), fan_m)
        pts = head[::-1]
    return pts, (con[0], con[1])


def _tail_dir(cg: Any, start: bool) -> np.ndarray:
    """Direction of travel into the station at one end of the clipped route."""
    c = np.asarray(cg.coords, float)
    c = c[::-1] if start else c
    return _unit(_far_point(c[::-1], 30.0), c[-1])


def _fan(pts: np.ndarray, target: np.ndarray, d_end: np.ndarray, fan_m: float) -> np.ndarray:
    """Replace the last ``fan_m`` of ``pts`` by a smooth curve ending at ``target``."""
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    span = min(fan_m, s[-1] / 2)
    if span < 1.0:
        return pts
    sa = s[-1] - span
    a = np.array([np.interp(sa, s, pts[:, 0]), np.interp(sa, s, pts[:, 1])])
    k = int(np.searchsorted(s, sa, side="right"))
    d_a = _unit(pts[max(k - 1, 0)], pts[min(k, len(pts) - 1)])
    chord = float(np.hypot(*(target - a)))
    t = np.linspace(0.0, 1.0, 13)[:, None]
    h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
    h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
    curve = h00 * a + h10 * chord * d_a + h01 * target + h11 * chord * d_end
    return np.vstack([pts[:k], curve])


def bundle_geometries(
    graph: Any,
    circuit_routes: pd.DataFrame,
    spacing_m: float = SPACING_M,
    transition_m: float = TRANSITION_M,
    parallel_m: float = PARALLEL_M,
    stations: Any = None,
    zone_m: float = ZONE_M,
    zone_max_m: float = ZONE_MAX_M,
) -> Any:
    """Draw every circuit as one offset LineString per contiguous route part.

    Args:
        graph: edges (``edge_id``, ``u``, ``v``, ``geometry`` from ``u`` to ``v``).
        circuit_routes: ``mrid`` and its ordered ``edges`` (see the module docstring).
        spacing_m: distance between neighbouring circuits on an edge.
        transition_m: length over which a circuit moves to a new slot at a node. Circuit
            set changes on edges shorter than twice this are ignored.
        parallel_m: bundles on ways closer than this are stacked as one (0: off).
        stations: optional GeoDataFrame of station footprints (``site``, polygon). When
            given, each circuit end at a station is drawn as a schematic radial fan inside
            the station zone (see :func:`station_fans`); ``circuit_routes`` then needs
            ``bus1`` / ``bus2`` (``site|kv``).
        zone_m, zone_max_m: station zone buffer, ``max(zone_m, 0.5 x footprint
            diameter)`` capped at ``zone_max_m``.

    Returns:
        GeoDataFrame in the CRS of ``graph`` with columns ``mrid``, ``part``,
        ``edges`` (``;``-joined edge ids in travel order), ``slots`` (``;``-joined
        ``slot/n`` per edge, slot 0 leftmost in the edge's ``u -> v`` direction),
        ``max_n`` (largest bundle on the route), ``slot_changes`` (nodes where the
        offset changes), ``offset_m`` (``;``-joined left offset per edge, travel
        direction), ``connector1_m`` / ``connector2_m`` (straight run from the route end
        to the station outline, 0 when none) and ``geometry`` (offset baked in). When
        ``circuit_routes`` has the clipped route ``geometry`` (and ``entry1/2_kind``,
        ``entry1/2_gap_m``), lines are cut at the station outlines, and circuits ending at
        a station are ordered by where they meet its outline.
    """
    gpd = require("geopandas", "grid")
    shapely = require("shapely", "grid")
    g, o, order = _prepare(graph, circuit_routes, 2 * transition_m)
    slot = {(e, ln): (k, len(lines)) for e, lines in order.items() for k, ln in enumerate(lines)}
    left = _parallel_offsets(o, order, spacing_m, parallel_m)
    rows, shapes = [], []
    for ln, route in o.parts.items():
        pieces, offs, sl = [], [], []
        for e, fwd in route:
            k, n = slot[(e, ln)]
            off = left[(e, ln)] if fwd else -left[(e, ln)]
            c = o.coords[e] if fwd else o.coords[e][::-1]
            pieces.append((c, off))
            offs.append(off)
            sl.append(f"{k}/{n}")
        # a short edge between two edges with the same offset keeps that offset
        for i in range(1, len(offs) - 1):
            e = route[i][0]
            if offs[i - 1] == offs[i + 1] != offs[i] and o.length[e] < 2 * transition_m:
                offs[i] = offs[i - 1]
                pieces[i] = (pieces[i][0], offs[i])
        xy, arc, off = _offset_line(pieces, transition_m)
        if len(xy) < 2:
            continue
        con = (0.0, 0.0)
        clip = o.clipped.get(ln[0])
        if clip is not None:
            xy, con = _clip_ends(pieces, xy, arc, off, clip, ln[1] == 0, ln[1] == o.last[ln[0]])
            if len(xy) < 2:
                continue
        changes = sum(1 for a, b in itertools.pairwise(offs) if a != b)
        rows.append(
            {
                "mrid": ln[0],
                "part": ln[1],
                "edges": ";".join(str(e) for e, _ in route),
                "slots": ";".join(sl),
                "max_n": max(int(x.split("/")[1]) for x in sl),
                "slot_changes": changes,
                "offset_m": ";".join(f"{x:g}" for x in offs),
                "connector1_m": con[0],
                "connector2_m": con[1],
            }
        )
        shapes.append(shapely.LineString(xy))
    out = gpd.GeoDataFrame(rows, geometry=shapes, crs=g.crs)
    out = _join_parts(out)
    if stations is not None and len(out):
        out = station_fans(out, circuit_routes, stations, zone_m, zone_max_m)
    if graph.crs is not None and out.crs is not None and graph.crs != out.crs:
        out = out.to_crs(graph.crs)
    return out


def _join_parts(out: Any, join_m: float = JOIN_PARTS_M) -> Any:
    """Join consecutive parts of one circuit whose ends are closer than ``join_m``.

    ``part_gap_m`` keeps the largest remaining gap (0 when the circuit is one line).
    """
    shapely = require("shapely", "grid")
    out = out.assign(part_gap_m=0.0)
    if not len(out) or (out["part"] == 0).all():
        return out
    keep = []
    for _, grp in out.groupby("mrid", sort=False):
        grp = grp.sort_values("part")
        rows = [grp.iloc[0].copy()]
        for _, r in grp.iloc[1:].iterrows():
            a, b = rows[-1].geometry.coords[-1], r.geometry.coords[0]
            gap = math.dist(a, b)
            if gap < join_m:
                rows[-1]["geometry"] = shapely.LineString(
                    list(rows[-1].geometry.coords) + list(r.geometry.coords)
                )
                for col in ("edges", "slots", "offset_m"):
                    rows[-1][col] = f"{rows[-1][col]};{r[col]}"
                rows[-1]["slot_changes"] += r["slot_changes"] + 1
                rows[-1]["connector2_m"] = r["connector2_m"]
            else:
                rows[-1]["part_gap_m"] = max(rows[-1]["part_gap_m"], gap)
                rows.append(r.copy())
        keep += rows
    return type(out)(keep, geometry="geometry", crs=out.crs).reset_index(drop=True)


def _round_zone(g: Any, zone: Any, margin_m: float = 30.0) -> Any:
    """``g`` with its stretch through ``zone`` replaced by the shorter way round it."""
    shapely = require("shapely", "grid")
    ops = require("shapely.ops", "grid")
    ring = zone.buffer(margin_m).exterior
    x = g.intersection(ring)
    pts = [p for p in shapely.get_parts(x) if p.geom_type == "Point"]
    if len(pts) < 2:
        return None
    sp = sorted(float(g.project(p)) for p in pts)
    s_in, s_out = sp[0], sp[-1]
    if s_out - s_in < 1.0:
        return None
    p_in, p_out = g.interpolate(s_in), g.interpolate(s_out)
    t1, t2 = float(ring.project(p_in)), float(ring.project(p_out))
    total = ring.length
    fwd = t2 - t1 <= total / 2 if t2 >= t1 else t1 - t2 > total / 2
    step = np.linspace(0, 1, 40)
    span = (t2 - t1) % total if fwd else -((t1 - t2) % total)
    arc = [ring.interpolate((t1 + f * span) % total).coords[0] for f in step]
    head = ops.substring(g, 0, s_in)
    tail = ops.substring(g, s_out, g.length)
    coords = list(head.coords) if head.length > 0 else []
    coords += arc
    coords += list(tail.coords)[1:] if tail.length > 0 else []
    return shapely.LineString(coords)


def _root(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _hermite(a: np.ndarray, da: np.ndarray, b: np.ndarray, db: np.ndarray) -> np.ndarray:
    chord = float(np.hypot(*(b - a)))
    t = np.linspace(0.0, 1.0, 17)[:, None]
    h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
    h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
    return h00 * a + h10 * chord * da + h01 * b + h11 * chord * db


def station_fans(
    lines: Any,
    circuit_routes: pd.DataFrame,
    stations: Any,
    zone_m: float = ZONE_M,
    zone_max_m: float = ZONE_MAX_M,
    group_m: float = GROUP_M,
    entry_gap_m: float = ENTRY_GAP_M,
    freeze_m: float = FREEZE_M,
) -> Any:
    """Replace the OSM station approaches by a schematic radial fan (display only).

    The station zone is the footprint buffered by ``max(zone_m, 0.5 x diameter)``,
    capped at ``zone_max_m``. Each circuit ending at a station is cut where it first
    enters that zone coming from its far end. Cut points closer than ``group_m`` with the
    same heading form a group that keeps its bundle order. Groups are ordered by the
    polar angle of their cut point about the station centroid and get entry slots on the
    outline in the same angular order (projection of the cut point, slots at least
    ``entry_gap_m`` apart, members side by side); each circuit then runs on a smooth
    curve from its cut point to its slot. Since slot order equals angle order, fans of
    one station do not cross. The fan starts ``freeze_m`` before the zone edge, so a
    group's order is fixed there. Circuits passing a zone without ending there stay on
    their route, or go round the zone boundary when that crosses fewer fans of the
    station; the result lists them in ``through_zones`` (``;``-joined sites) and
    ``around_zones`` (sites they were drawn round).

    ``lines`` is the output of :func:`bundle_geometries` (any CRS); returns it with the
    station ends redrawn and ``fan1`` / ``fan2`` (site fanned at the start / end).
    """
    gpd = require("geopandas", "grid")
    shapely = require("shapely", "grid")
    polygon = require("shapely.geometry.polygon", "grid")
    crs = lines.crs
    li = lines.to_crs(METRIC_CRS) if crs is not None else lines
    st = stations.to_crs(METRIC_CRS) if stations.crs is not None else stations
    st = st[st.geometry.notna() & st.geom_type.isin(["Polygon", "MultiPolygon"])]
    foot = {}
    for site, geom in zip(st["site"], st.geometry, strict=True):
        if geom.geom_type == "MultiPolygon":
            geom = geom.convex_hull
        foot[str(site)] = polygon.orient(geom, 1.0)
    zones = {}
    for site, poly in foot.items():
        x0, y0, x1, y1 = poly.bounds
        z = min(max(zone_m, 0.5 * math.hypot(x1 - x0, y1 - y0)), zone_max_m)
        zones[site] = poly.buffer(z)
    bus = circuit_routes.drop_duplicates("mrid").set_index("mrid")
    geoms = list(li.geometry)
    first_part = li.groupby("mrid")["part"].transform("min").eq(li["part"]).to_numpy()
    last_part = li.groupby("mrid")["part"].transform("max").eq(li["part"]).to_numpy()
    # cut points per (site, row, end)
    ends: dict[str, list[dict]] = defaultdict(list)
    for i, (m, g) in enumerate(zip(li["mrid"], geoms, strict=True)):
        if m not in bus.index or g is None or g.length < 1:
            continue
        sites = [
            str(bus.at[m, c]).split("|")[0] if c in bus.columns else "" for c in ("bus1", "bus2")
        ]
        c = np.asarray(g.coords, float)
        n = max(2, int(g.length // 10) + 1)
        d = np.linspace(0, g.length, n)
        pts = shapely.line_interpolate_point(g, d)
        for k, site in enumerate(sites):
            if site not in zones or not (first_part[i] if k == 0 else last_part[i]):
                continue
            inside = shapely.contains(zones[site], pts)
            other = sites[1 - k]
            out_other = ~shapely.contains(zones[other], pts) if other in zones else np.ones(n, bool)
            # first entry into this zone, coming from the far end (skipping the far
            # station's own zone)
            seq_in, seq_out = (inside, out_other) if k == 1 else (inside[::-1], out_other[::-1])
            away = np.nonzero(seq_out)[0]
            if not len(away):
                continue
            hit = np.nonzero(seq_in[away[0] :])[0]
            if not len(hit):
                continue
            j = int(away[0] + hit[0])
            if j <= 0 or j >= n - 1:
                continue
            j = j if k == 1 else n - 1 - j
            # freeze: start the fan freeze_m earlier, so the group's order cannot change
            # just before it
            back_n = round(freeze_m / max(d[1] - d[0], 1e-9))
            j = max(1, j - back_n) if k == 1 else min(n - 2, j + back_n)
            cut, s_cut = np.array([pts[j].x, pts[j].y]), d[j]
            back = pts[j - 1] if k == 1 else pts[j + 1]
            head = _unit(np.array([back.x, back.y]), cut)
            ends[site].append({"row": i, "end": k, "cut": cut, "s": s_cut, "head": head})
    new = dict(enumerate(geoms))
    fan = {1: [""] * len(li), 2: [""] * len(li)}
    for site, items in ends.items():
        poly = foot[site]
        ring = poly.exterior
        ctr = np.array(poly.centroid.coords[0])
        # groups: single linkage on distance, same heading
        grp = list(range(len(items)))

        for a, b in itertools.combinations(range(len(items)), 2):
            ia, ib = items[a], items[b]
            if math.dist(ia["cut"], ib["cut"]) <= group_m and float(
                ia["head"] @ ib["head"]
            ) > math.cos(math.radians(25)):
                grp[_root(grp, a)] = _root(grp, b)
        groups: dict[int, list[int]] = defaultdict(list)
        for a in range(len(items)):
            groups[_root(grp, a)].append(a)
        glist = []
        for members in groups.values():
            mc = np.mean([items[a]["cut"] for a in members], axis=0)
            ang = math.atan2(mc[1] - ctr[1], mc[0] - ctr[0])
            # travel heading of the group (its bundle order is relative to it)
            h = _unit(np.zeros(2), np.sum([items[a]["head"] for a in members], axis=0))
            # left of inward travel first (that is increasing outline parameter)
            lat = {
                a: float(h[0] * (items[a]["cut"] - mc)[1] - h[1] * (items[a]["cut"] - mc)[0])
                for a in members
            }
            members.sort(key=lambda a: -lat[a])
            t0 = float(ring.project(shapely.Point(mc)))
            glist.append((ang, t0, members))
        glist.sort(key=lambda x: x[0])
        # slot parameters on the CCW outline, in angular order, at least entry_gap_m apart
        total = ring.length
        gap = min(entry_gap_m, total / max(1, len(items)))
        start = glist[0][1] - (len(glist[0][2]) - 1) / 2 * gap
        slots: list[tuple[int, float]] = []
        prev = None
        for _, t0, members in glist:
            t = t0 - (len(members) - 1) / 2 * gap
            if prev is not None:
                t = max(t, prev + gap) if t0 >= start else max(t + total, prev + gap)
            for a in members:
                slots.append((a, t))
                prev = t
                t += gap
        for a, t in slots:
            it = items[a]
            p = np.array(ring.interpolate(t % total).coords[0])
            tan = _unit(
                np.array(ring.interpolate((t - 1) % total).coords[0]),
                np.array(ring.interpolate((t + 1) % total).coords[0]),
            )
            inward = np.array([-tan[1], tan[0]])  # CCW ring: left of tangent is inside
            g = new[it["row"]]
            c = np.asarray(g.coords, float)
            sc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(c, axis=0).T))])
            if it["end"] == 1:
                keep = c[sc < it["s"]]
                curve = _hermite(it["cut"], it["head"], p, inward)
                new[it["row"]] = shapely.LineString(np.vstack([keep, curve]))
                fan[2][it["row"]] = site
            else:
                keep = c[sc > it["s"]]
                curve = _hermite(it["cut"], it["head"], p, inward)[::-1]
                new[it["row"]] = shapely.LineString(np.vstack([curve, keep]))
                fan[1][it["row"]] = site
    through = []
    zone_tree_sites = list(zones)
    ztree = shapely.STRtree([zones[s] for s in zone_tree_sites])
    for m, g in zip(li["mrid"], geoms, strict=True):
        own = {
            str(bus.at[m, c]).split("|")[0]
            for c in ("bus1", "bus2")
            if m in bus.index and c in bus.columns
        }
        hits = ztree.query(g, predicate="intersects")
        through.append(";".join(sorted({zone_tree_sites[h] for h in hits} - own)))
    around = [""] * len(li)
    fanned: dict[str, list[int]] = defaultdict(list)
    for r, site in enumerate(fan[1]):
        if site:
            fanned[site].append(r)
    for r, site in enumerate(fan[2]):
        if site:
            fanned[site].append(r)
    for r, sites in enumerate(through):
        for site in [x for x in sites.split(";") if x and x in fanned]:
            fans = shapely.union_all([new[q] for q in fanned[site] if q != r])
            g = new[r]
            alt = _round_zone(g, zones[site])
            if alt is None:
                continue

            def n_cross(line: Any, fans: Any = fans) -> int:
                x = line.intersection(fans)
                return 0 if x.is_empty else len(shapely.get_parts(x))

            if n_cross(alt) < n_cross(g):
                new[r] = alt
                around[r] = f"{around[r]};{site}".strip(";")
    out = li.assign(fan1=fan[1], fan2=fan[2], through_zones=through, around_zones=around)
    out = gpd.GeoDataFrame(out, geometry=[new[i] for i in range(len(li))], crs=li.crs)
    return out.to_crs(crs) if crs is not None else out


# ---------------------------------------------------------------- inputs


def read_network(directory: Any) -> tuple[Any, pd.DataFrame]:
    """Read ``grid_network`` output (``edges.parquet``, ``nodes.parquet``,
    ``circuit_routes.parquet``) as ``(graph, circuit_routes)`` for :func:`bundle_geometries`.

    Edge geometries are turned to run from ``u`` to ``v``; circuits without edges are dropped.
    """
    from pathlib import Path

    gpd = require("geopandas", "grid")
    shapely = require("shapely", "grid")
    d = Path(directory)
    edges = pd.read_parquet(d / "edges.parquet")
    nodes = pd.read_parquet(d / "nodes.parquet").set_index("node")
    routes = pd.read_parquet(d / "circuit_routes.parquet")
    geoms = shapely.from_wkb(edges["geometry"].to_numpy())
    first = shapely.get_point(geoms, 0)
    ux = nodes.loc[edges["u"], ["lon", "lat"]].to_numpy()
    vx = nodes.loc[edges["v"], ["lon", "lat"]].to_numpy()
    fx = np.column_stack([shapely.get_x(first), shapely.get_y(first)])
    flip = np.hypot(*(fx - vx).T) < np.hypot(*(fx - ux).T)
    geoms = np.where(flip, shapely.reverse(geoms), geoms)
    graph = gpd.GeoDataFrame(edges.drop(columns="geometry"), geometry=geoms, crs="EPSG:4326")
    if "edges_outside" in routes.columns:
        out = [
            list(o) if o is not None and len(o) else list(e) if e is not None else []
            for o, e in zip(routes["edges_outside"], routes["edges"], strict=True)
        ]
        routes = routes.assign(edges=out)
    routes = routes.assign(edges=[list(e) if e is not None else [] for e in routes["edges"]])
    routes = routes[routes["edges"].map(len) > 0]
    return graph, routes.reset_index(drop=True)


def stub_from_export(
    lines: Any, tower_lines: Any, cluster_m: float = 30.0, join_m: float = 5.0
) -> tuple[Any, pd.DataFrame]:
    """Build a stand-in graph and circuit routes from today's map export.

    Each piece of the ``lines`` layer names the OpenStreetMap ways it runs on
    (``tower_way``, ``;``-joined, in travel order). The ways are cut where any circuit
    joins or leaves them; pieces off the ways (gaps, station tails) become virtual edges
    (``virtual`` True). Nodes closer than ``join_m`` are merged, and so are the ends of
    consecutive runs of one circuit within :data:`JOIN_ROUTE_M`.

    Returns:
        ``(graph, circuit_routes)`` in :data:`METRIC_CRS`, in the form
        :func:`bundle_geometries` takes.
    """
    gpd = require("geopandas", "grid")
    shapely = require("shapely", "grid")
    ops = require("shapely.ops", "grid")
    tw = tower_lines.to_crs(METRIC_CRS)
    way_geom = {str(int(i)): g for i, g in zip(tw["osm_id"], tw.geometry, strict=True)}
    li = lines.to_crs(METRIC_CRS)
    li = li.sort_values(["mrid", "piece"], kind="stable")
    meta = li.drop_duplicates("mrid").set_index("mrid")
    # 1. runs per circuit: ("way", id, s0, s1) or ("gap", coords)
    runs: dict[Any, list[tuple]] = {}
    cuts: dict[str, list[float]] = defaultdict(list)
    for m, grp in li.groupby("mrid", sort=False):
        out = []
        for tw_ids, geom in zip(grp["tower_way"].fillna(""), grp.geometry, strict=True):
            if geom is None or geom.is_empty or geom.geom_type != "LineString":
                continue
            ids = [x for x in str(tw_ids).split(";") if x in way_geom]
            if not ids:
                if geom.length > 1.0:
                    out.append(("gap", np.asarray(geom.coords, float)))
                continue
            # sample the piece, give each sample its nearest listed way, cut into runs
            n = max(2, int(geom.length // 25.0) + 1)
            d = np.linspace(0.0, geom.length, n)
            pts = shapely.line_interpolate_point(geom, d)
            dist = np.array([shapely.distance(way_geom[w], pts) for w in ids])
            best = np.where(dist.min(axis=0) < 60.0, dist.argmin(axis=0), -1)
            k = 0
            last_d = 0.0
            while k < n:
                j = k
                while j + 1 < n and best[j + 1] == best[k]:
                    j += 1
                if best[k] >= 0 and j > k:
                    wid = ids[best[k]]
                    if d[k] - last_d > 1.0:
                        out.append(("gap", np.asarray(ops.substring(geom, last_d, d[k]).coords)))
                    s0 = float(shapely.line_locate_point(way_geom[wid], pts[k]))
                    s1 = float(shapely.line_locate_point(way_geom[wid], pts[j]))
                    if abs(s1 - s0) >= 1.0:
                        out.append(("way", wid, s0, s1))
                        cuts[wid] += [s0, s1]
                    last_d = d[j]
                k = j + 1
            if geom.length - last_d > 1.0:
                out.append(("gap", np.asarray(ops.substring(geom, last_d, geom.length).coords)))
        runs[m] = out
    # 2. cut positions per way, clustered
    clusters: dict[str, list[float]] = {}
    for wid, ss in cuts.items():
        length = way_geom[wid].length
        vals = sorted([0.0, length, *ss])
        cl: list[list[float]] = [[vals[0]]]
        for x in vals[1:]:
            if x - cl[-1][-1] <= cluster_m:
                cl[-1].append(x)
            else:
                cl.append([x])
        pos = [float(np.mean(c)) for c in cl]
        pos[0], pos[-1] = 0.0, length  # way ends stay where they are
        clusters[wid] = pos

    def cidx(wid: str, s: float) -> int:
        return int(np.argmin(np.abs(np.asarray(clusters[wid]) - s)))

    # 3. nodes: (wid, cluster) and gap ends, merged by distance
    node_xy: dict[Hashable, tuple[float, float]] = {}
    for wid, pos in clusters.items():
        for k, s in enumerate(pos):
            p = way_geom[wid].interpolate(s)
            node_xy[(wid, k)] = (p.x, p.y)
    edges: dict[Any, dict] = {}
    routes: dict[Any, list[Any]] = {}
    gap_n = 0
    for m, rs in runs.items():
        seq: list[Any] = []
        prev_node: Hashable | None = None
        for r in rs:
            if r[0] == "way":
                _, wid, s0, s1 = r
                a, b = cidx(wid, s0), cidx(wid, s1)
                if a == b:
                    continue
                step = 1 if b > a else -1
                start = (wid, a)
                if prev_node is not None and prev_node != start:
                    seq.append(("join", prev_node, start))
                for k in range(a, b, step):
                    lo, hi = min(k, k + step), max(k, k + step)
                    eid = f"{wid}:{lo}"
                    if eid not in edges:
                        pos = clusters[wid]
                        sub = ops.substring(way_geom[wid], pos[lo], pos[hi])
                        edges[eid] = {
                            "u": (wid, lo),
                            "v": (wid, hi),
                            "coords": np.asarray(sub.coords, float),
                            "virtual": False,
                        }
                    seq.append(eid)
                prev_node = (wid, b)
            else:
                c = r[1]
                n0, n1 = ("g", gap_n, 0), ("g", gap_n, 1)
                gap_n += 1
                node_xy[n0], node_xy[n1] = tuple(c[0]), tuple(c[-1])
                if prev_node is not None:
                    seq.append(("join", prev_node, n0))
                eid = f"g{gap_n}"
                edges[eid] = {"u": n0, "v": n1, "coords": c, "virtual": True}
                seq.append(eid)
                prev_node = n1
        routes[m] = seq
    # merge nodes: within join_m, and route joins
    parent: dict[Hashable, Hashable] = {n: n for n in node_xy}

    def find(x: Hashable) -> Hashable:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: Hashable, b: Hashable) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb, key=str)] = min(ra, rb, key=str)

    names = list(node_xy)
    xy = np.array([node_xy[n] for n in names])
    tree = shapely.STRtree(shapely.points(xy))
    i, j = tree.query(shapely.points(xy), predicate="dwithin", distance=join_m)
    for a, b in zip(i, j, strict=True):
        if a < b:
            union(names[a], names[b])
    for seq in routes.values():
        for x in seq:
            if isinstance(x, tuple) and math.dist(node_xy[x[1]], node_xy[x[2]]) <= JOIN_ROUTE_M:
                union(x[1], x[2])
    groups: dict[Hashable, list[Hashable]] = defaultdict(list)
    for n in names:
        groups[find(n)].append(n)
    centre = {r: tuple(np.mean([node_xy[n] for n in ns], axis=0)) for r, ns in groups.items()}
    # 4. edges with merged ends; joins that stay apart become straight virtual edges;
    #    virtual edges between the same nodes are shared
    shared: dict[tuple, str] = {}
    final: dict[str, dict] = {}
    for eid, ed in edges.items():
        u, v = find(ed["u"]), find(ed["v"])
        c = ed["coords"].copy()
        c[0], c[-1] = centre[u], centre[v]
        if ed["virtual"]:
            key = (min(u, v, key=str), max(u, v, key=str))
            if key in shared:
                final[eid] = final[shared[key]]
                continue
            shared[key] = eid
        final[eid] = {
            "edge_id": eid,
            "u": str(u),
            "v": str(v),
            "coords": c,
            "virtual": ed["virtual"],
        }
    out_routes = []
    jn = 0
    for m, seq in routes.items():
        ids: list[str] = []
        for x in seq:
            if isinstance(x, tuple):
                u, v = find(x[1]), find(x[2])
                if u == v:
                    continue
                key = (min(u, v, key=str), max(u, v, key=str))
                if key not in shared:
                    jn += 1
                    eid = f"j{jn}"
                    shared[key] = eid
                    final[eid] = {
                        "edge_id": eid,
                        "u": str(u),
                        "v": str(v),
                        "coords": np.array([centre[u], centre[v]]),
                        "virtual": True,
                    }
                ids.append(final[shared[key]]["edge_id"])
            else:
                if final[x]["u"] == final[x]["v"]:
                    continue
                ids.append(final[x]["edge_id"])
        dedup = [e for k, e in enumerate(ids) if k == 0 or e != ids[k - 1]]
        out_routes.append(
            {
                "mrid": m,
                "name": meta.at[m, "name"],
                "voltage_kv": float(meta.at[m, "voltage_kv"]),
                "edges": dedup,
            }
        )
    seen, erows, eshapes = set(), [], []
    for ed in final.values():
        if ed["edge_id"] in seen:
            continue
        seen.add(ed["edge_id"])
        erows.append({k: ed[k] for k in ("edge_id", "u", "v", "virtual")})
        eshapes.append(shapely.LineString(ed["coords"]))
    graph = gpd.GeoDataFrame(erows, geometry=eshapes, crs=METRIC_CRS)
    return graph, pd.DataFrame(out_routes)
