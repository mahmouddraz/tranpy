"""Routes of the model's circuits along the mapped OpenStreetMap lines.

OpenStreetMap carries two things the Core Static Grid Model lacks: the geometry of every
line and, as ``route=power`` relations, the identity of the circuits on it (name, line
number, colour). A circuit's route is found by following the relation that passes both
of its stations at the circuit's voltage; without one, by following named ways, and as a
last resort any in-operation ways of that voltage. Only ways in operation can carry a
route. Every route is then compared with independent evidence (:func:`check_routes`), and
nothing here changes the electrical model.
"""

from __future__ import annotations

import itertools
import json
import re
import unicodedata
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

OVERPASS = "https://overpass-api.de/api/interpreter"
CIRCUITS_QUERY = (
    '[out:json][timeout:170];area["ISO3166-1"="DE"]->.d;'
    'way["power"="line"]["voltage"~"220000|380000"](area.d);'
    'rel(bw)["route"="power"];out body;'
)
"""Power circuit relations on the German 220/380 kV lines."""

_KM = np.array([np.cos(np.radians(51.0)) * 111.2, 111.2])
"""Kilometres per degree of longitude and latitude at the centre of Germany."""

_TYPICAL_X_OHM_PER_KM = {220: 0.32, 380: 0.27}
_COLOURS = {
    "blau": "blue", "rot": "red", "gelb": "yellow", "grun": "green",
    "weis": "white", "schwarz": "black", "braun": "brown",
}  # fmt: skip
_END_REACH_KM = 2.5
"""How far from a station the mapped line may start (large stations, map offsets)."""


def fetch_circuit_relations(target: Path, timeout_s: float = 190.0) -> Path:
    """Download the circuit relations (kept if already there)."""
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            OVERPASS,
            urllib.parse.urlencode({"data": CIRCUITS_QUERY}).encode(),
            {"User-Agent": "tranpy-research"},
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            target.write_bytes(response.read())
    return target


def fold(text: Any) -> str:
    """Lower-case ASCII with umlaut transliterations and doubled letters folded."""
    s = unicodedata.normalize("NFKD", str(text).lower())
    s = "".join(c for c in s if not unicodedata.combining(c)).replace("ß", "ss")
    return re.sub(r"(.)\1", r"\1", re.sub(r"([aou])e", r"\1", s))


def voltage_classes(tags: Mapping[str, Any]) -> set[int]:
    """220 and/or 380 for the voltages tagged on a way."""
    return {
        380 if int(v) >= 300000 else 220
        for v in re.findall(r"\d+", str(tags.get("voltage", "")))
        if int(v) >= 200000
    }


@dataclass
class WayNetwork:
    """In-operation ways of one voltage class, as a graph of way pieces."""

    kv: int
    ways: list[dict[str, Any]]
    points: npt.NDArray[np.float64] = field(repr=False)
    tree: cKDTree = field(repr=False)

    @classmethod
    def build(cls, ways: Iterable[dict[str, Any]], kv: int, excluded: set[int]) -> WayNetwork:
        selected = [
            w for w in ways if w["id"] not in excluded and kv in voltage_classes(w.get("tags", {}))
        ]
        points = np.array(
            [(q["lon"], q["lat"]) for w in selected for q in w["geometry"]] or np.empty((0, 2))
        )
        return cls(kv, selected, points, cKDTree(points * _KM))

    def near(self, centre: npt.NDArray[np.float64], radius_km: float) -> list[dict[str, Any]]:
        """Ways whose middle lies within ``radius_km`` of ``centre`` (km coordinates)."""
        out = []
        for w in self.ways:
            g = w["geometry"]
            mid = np.array([(g[0]["lon"] + g[-1]["lon"]) / 2, (g[0]["lat"] + g[-1]["lat"]) / 2])
            if np.hypot(*(mid * _KM - centre)) < radius_km:
                out.append(w)
        return out


@dataclass
class Walk:
    """A route along way pieces: points (lon, lat), length and the joins that were needed."""

    points: npt.NDArray[np.float64]
    km: float
    joins: list[dict[str, Any]]


def walk(
    ways: Sequence[dict[str, Any]],
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    join_km: float = 0.5,
) -> Walk | None:
    """Shortest route over ``ways`` from near ``a`` to near ``b`` (km coordinates).

    Ways are split at shared nodes; ends closer than ``join_km`` are joined (a join costs
    twice its length, so mapped line is preferred). The route is charged the distance from
    its ends to the stations, so it reaches them instead of stopping at a nearby junction.
    """
    count: dict[int, int] = {}
    coord: dict[int, tuple[float, float]] = {}
    for w in ways:
        for n, q in zip(w["nodes"], w["geometry"], strict=True):
            count[n] = count.get(n, 0) + 1
            coord[n] = (q["lon"], q["lat"])
    if not coord:
        return None
    ends = {n for w in ways for n in (w["nodes"][0], w["nodes"][-1])}
    edges: list[tuple[int, int, npt.NDArray[np.float64], float]] = []
    for w in ways:
        ids = w["nodes"]
        geometry = np.array([(q["lon"], q["lat"]) for q in w["geometry"]])
        cuts = [0, *[i for i in range(1, len(ids) - 1) if count[ids[i]] > 1 or ids[i] in ends]]
        cuts.append(len(ids) - 1)
        for i, j in itertools.pairwise(cuts):
            piece = geometry[i : j + 1]
            km = float(np.hypot(*(np.diff(piece, axis=0) * _KM).T).sum())
            edges.append((ids[i], ids[j], piece, km))
    names = sorted(coord)
    index = {n: i for i, n in enumerate(names)}
    xy = np.array([coord[n] for n in names]) * _KM
    best: dict[tuple[int, int], int] = {}
    for k, e in enumerate(edges):
        u, v = index[e[0]], index[e[1]]
        if (u, v) not in best or e[3] < edges[best[(u, v)]][3]:
            best[(u, v)] = best[(v, u)] = k
    rows = [u for u, _ in best]
    cols = [v for _, v in best]
    cost = [max(edges[k][3], 1e-3) for k in best.values()]
    joins: dict[tuple[int, int], float] = {}
    for i, j in cKDTree(xy).query_pairs(join_km):
        if (i, j) not in best:
            d = float(np.hypot(*(xy[i] - xy[j])))
            joins[(i, j)] = joins[(j, i)] = d
            rows += [i, j]
            cols += [j, i]
            cost += [d * 2 + 0.01] * 2
    graph = coo_matrix((cost, (rows, cols)), shape=(len(names),) * 2).tocsr()
    tree = cKDTree(xy)
    near_a = tree.query_ball_point(a, _END_REACH_KM)
    near_b = tree.query_ball_point(b, _END_REACH_KM)
    if not near_a or not near_b:
        return None
    distance, predecessor = dijkstra(graph, indices=near_a, return_predecessors=True)
    reach_a = np.hypot(*(xy[near_a] - a).T)
    reach_b = np.hypot(*(xy[near_b] - b).T)
    score = distance[:, near_b] + 3 * reach_a[:, None] + 3 * reach_b[None, :]
    i, j = (int(v) for v in np.unravel_index(int(np.argmin(score)), score.shape))
    if not np.isfinite(score[i, j]):
        return None
    node, start = near_b[j], near_a[i]
    pieces: list[npt.NDArray[np.float64]] = []
    km = 0.0
    needed: list[dict[str, Any]] = []
    while node != start:
        prev = predecessor[i, node]
        if (prev, node) in best:
            e = edges[best[(prev, node)]]
            pieces.append(e[2] if index[e[0]] == prev else e[2][::-1])
            km += e[3]
        else:
            d = joins[(prev, node)]
            piece = np.array([xy[prev], xy[node]]) / _KM
            pieces.append(piece)
            km += d
            if d > 0.1:
                needed.append({"a": piece[0].tolist(), "b": piece[1].tolist(), "km": round(d, 2)})
        node = prev
    if not pieces:
        return None
    return Walk(np.vstack(pieces[::-1]), km, needed[::-1])


def plausible(route: Walk) -> bool:
    """A route is not much shorter than the distance between its own ends, nor a detour."""
    straight = float(np.hypot(*((route.points[0] - route.points[-1]) * _KM)))
    return route.km >= 0.3 and 0.9 * straight <= route.km <= 2.5 * max(straight, 1.5)


def _identifiers(name: str) -> tuple[set[str], set[str], list[str]]:
    """Line numbers, colours and operator codes a circuit is known by."""
    folded = fold(name)
    numbers = set(re.findall(r"(?<!\d)\d{3,4}(?!\d)", name))
    colours = {en for de, en in _COLOURS.items() if re.search(rf"\b{de}\b", folded)}
    colours |= {de for de in _COLOURS if re.search(rf"\b{de}\b", folded)}
    codes = [fold(x) for x in re.findall(r"\b[A-ZÄÖÜ]{4,}\b", name)]
    return numbers, colours, codes


def _way_text(way: Mapping[str, Any]) -> str:
    return fold(" ".join(str(way["tags"].get(k, "")) for k in ("name", "ref")))


def _code_in(code: str, text: str) -> bool:
    """``kondlw`` is in ``kondelwald``: same first letter, letters in order."""
    for word in re.findall(r"[a-z]{4,}", text):
        letters = iter(word)
        if word[0] == code[0] and all(c in letters for c in code):
            return True
    return False


def prepare_relations(
    relations: Iterable[dict[str, Any]], ways: Mapping[int, dict[str, Any]], excluded: set[int]
) -> list[dict[str, Any]]:
    """Circuit relations with their usable member ways and a search text."""
    out = []
    for r in relations:
        tags = r.get("tags", {})
        classes = voltage_classes(tags)
        if not classes:
            continue
        kv = 380 if 380 in classes else 220
        members = [
            ways[m["ref"]]
            for m in r.get("members", [])
            if m["type"] == "way"
            and m["ref"] in ways
            and m["ref"] not in excluded
            and kv in voltage_classes(ways[m["ref"]].get("tags", {}))
        ]
        if not members:
            continue
        points = np.array([(q["lon"], q["lat"]) for w in members for q in w["geometry"]]) * _KM
        text = fold(" ".join(str(tags.get(k) or "") for k in ("name", "ref", "colour")))
        out.append(
            {"id": r["id"], "kv": kv, "name": tags.get("name"), "ref": tags.get("ref"),
             "colour": tags.get("colour"), "ways": members, "tree": cKDTree(points), "text": text}
        )  # fmt: skip
    return out


def route_circuit(
    name: str,
    kv: int,
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    relations: Sequence[dict[str, Any]],
    network: WayNetwork,
) -> tuple[Walk, str, str, dict[str, Any] | None] | None:
    """Route one circuit between station positions ``a`` and ``b`` (lon, lat).

    Returns the route, the method (``circuit record``, ``named ways``, ``same-voltage
    ways``), the identity evidence (``named``, ``colour``, ``passes both stations``,
    ``none``) and the relation used.
    """
    a_km, b_km = a * _KM, b * _KM
    numbers, colours, codes = _identifiers(name)
    candidates = []
    for r in relations:
        if r["kv"] != kv:
            continue
        da, _ = r["tree"].query(a_km)
        db, _ = r["tree"].query(b_km)
        if da <= _END_REACH_KM and db <= _END_REACH_KM:
            score = 0
            if any(re.search(rf"(?<!\d){x}(?!\d)", r["text"]) for x in numbers):
                score += 4
            if any(_code_in(c, r["text"]) for c in codes):
                score += 4
            if any(c in r["text"] for c in colours):
                score += 2
            candidates.append((score, -(da + db), r))
    candidates.sort(key=lambda c: (-c[0], -c[1]))
    for score, _, r in candidates[:4]:
        found = walk(r["ways"], a_km, b_km)
        if found and plausible(found):
            identity = "named" if score >= 4 else "colour" if score >= 2 else "passes both stations"
            return found, "circuit record", identity, r
    centre = (a_km + b_km) / 2
    radius = float(np.hypot(*(a_km - b_km))) + 30
    nearby = network.near(centre, radius)
    named = [
        w
        for w in nearby
        if any(
            re.search(rf"(?<!\d){x}(?!\d)", _way_text(w))
            for x in numbers
        )
        or any(_code_in(c, fold(w["tags"].get("name", ""))) for c in codes)
    ]  # fmt: skip
    if named:
        found = walk(named, a_km, b_km, join_km=8.0)
        if found and plausible(found):
            return found, "named ways", "named", None
    for join_km in (0.5, 2.0, 8.0):
        found = walk(nearby, a_km, b_km, join_km=join_km)
        if found and plausible(found):
            return found, "same-voltage ways", "none", None
    return None


def place_tees(
    tees: Iterable[str],
    lines: pd.DataFrame,
    positions: Mapping[str, Any],
    network_of: Mapping[int, WayNetwork],
) -> tuple[dict[str, list[float]], pd.DataFrame]:
    """Locate tee points on the mapped route where the official section lengths meet.

    For a tee with sections to ends A and B of lengths L(A) and L(B), the pair of located
    ends whose route length best matches L(A)+L(B) is taken, and the tee is placed at
    distance L(A) along that route, snapped to a junction of the way graph within 1.5 km.
    """  # fmt: skip
    junctions: dict[int, tuple[npt.NDArray[np.float64], cKDTree | None]] = {}
    for kv, network in network_of.items():
        count: dict[int, int] = {}
        coord: dict[int, tuple[float, float]] = {}
        for w in network.ways:
            for n, q in zip(w["nodes"], w["geometry"], strict=True):
                count[n] = count.get(n, 0) + 1
                coord[n] = (q["lon"], q["lat"])
        pts = np.array([coord[n] for n, c in count.items() if c >= 3] or np.empty((0, 2)))
        junctions[kv] = (pts, cKDTree(pts * _KM) if len(pts) else None)
    placed: dict[str, list[float]] = {}
    report: list[dict[str, Any]] = []
    for tee in tees:
        sections = lines[(lines["node0"] == tee) | (lines["node1"] == tee)]
        kv = 380 if sections["kv"].max() >= 300 else 220
        ends: list[tuple[str, float, npt.NDArray[np.float64]]] = []
        for s in sections.to_dict("records"):
            other = str(s["node1"] if s["node0"] == tee else s["node0"])
            p = positions.get(other)
            length = float(s["length_km"]) if pd.notna(s["length_km"]) else np.nan
            if p is not None and np.isfinite(length) and not other.startswith("y-"):
                ends.append((other, length, np.array(p, dtype=float) * _KM))
        if len(ends) < 2:
            report.append({"tee": tee, "result": "fewer than two located ends"})
            continue
        best = None
        network = network_of[kv]
        for (ea, la, pa), (eb, lb, pb) in itertools.combinations(ends, 2):
            found = walk(network.near((pa + pb) / 2, float(np.hypot(*(pa - pb))) + 30), pa, pb, 2.0)
            if found is None:
                continue
            error = abs(la + lb - found.km) / max(found.km, 1.0)
            if best is None or error < best[0]:
                best = (error, ea, la, eb, lb, found)
        if best is None:
            report.append({"tee": tee, "result": "no route between ends"})
            continue
        error, ea, la, eb, lb, found = best
        pa = next(p for e, _, p in ends if e == ea)
        if np.hypot(*(found.points[0] * _KM - pa)) > np.hypot(*(found.points[-1] * _KM - pa)):
            found = Walk(found.points[::-1], found.km, found.joins)
        if error > 0.2:
            report.append({"tee": tee, "result": f"ends do not straddle the tee ({error:.0%} off)"})
            continue
        along = np.r_[0, np.cumsum(np.hypot(*(np.diff(found.points, axis=0) * _KM).T))]
        i = int(np.argmin(np.abs(along - min(la, along[-1]))))
        point = found.points[i]
        how = "on route at section length"
        pts, tree = junctions[kv]
        if tree is not None:
            d, k = tree.query(point * _KM)
            if d <= 1.5:
                point, how = pts[k], f"junction within {d:.1f} km"
        placed[tee] = [float(point[0]), float(point[1])]
        report.append(
            {"tee": tee, "a": ea, "b": eb, "route_km": found.km, "sections_km": la + lb, "how": how}
        )
    return placed, pd.DataFrame(report)


def _densify(points: npt.NDArray[np.float64], step_km: float = 0.15) -> npt.NDArray[np.float64]:
    p = np.asarray(points, dtype=float) * _KM
    out = []
    for u, v in itertools.pairwise(p):
        n = max(1, int(np.hypot(*(v - u)) / step_km))
        out.append(u + np.linspace(0, 1, n, endpoint=False)[:, None] * (v - u))
    return np.vstack([*out, p[-1:]])


def check_routes(
    routes: pd.DataFrame,
    paths: Mapping[str, npt.NDArray[np.float64]],
    reference: Mapping[str, Sequence[npt.NDArray[np.float64]]],
    end_reached: Mapping[str, tuple[bool, bool]],
) -> pd.DataFrame:
    """Compare each route with independent geometry and the published reactance.

    ``reference`` maps a source name (``dlm250``, ``official``) to line geometries in
    (lon, lat); the result has the share of each route within 250 m of that source, the
    route length over the length implied by the reactance, and a proposed status.
    """
    trees = {k: cKDTree(np.vstack([_densify(g, 0.1) for g in v])) for k, v in reference.items()}
    rows = []
    for r in routes.to_dict("records"):
        key = str(r["key"])
        row: dict[str, Any] = {"key": key}
        p = paths.get(key)
        if p is None:
            row["proposal"] = "no route"
            rows.append(row)
            continue
        q = _densify(p)
        for k, t in trees.items():
            row[k] = float((t.query(q)[0] <= 0.25).mean())
        x_ohm = float(r["x_ohm"]) if pd.notna(r["x_ohm"]) else 0.0
        kv = int(r["kv"])
        ratio = float(r["km"]) / (x_ohm / _TYPICAL_X_OHM_PER_KM[kv]) if x_ohm > 0 else float("nan")
        row["x_ratio"] = ratio
        end_a, end_b = end_reached.get(key, (False, False))
        row["end_a"], row["end_b"] = end_a, end_b
        ok_ends = end_a and end_b
        ok_length = 0.7 <= ratio <= 1.6
        on_map = float(row.get("dlm250", 1.0)) >= 0.7 or bool(r.get("border", False))
        identified = r["identity"] in ("named", "colour")
        if identified and on_map and ok_ends and ok_length and int(r["joins"]) == 0:
            row["proposal"] = "mapped, consistent"
        elif on_map and ok_ends and ok_length:
            row["proposal"] = "mapped, small gaps" if identified else "mapped, identity open"
        elif not ok_ends:
            row["proposal"] = "end not reached"
        elif not ok_length:
            row["proposal"] = "length vs reactance off"
        elif not on_map:
            row["proposal"] = "disagrees with DLM250"
        else:
            row["proposal"] = "check"
        rows.append(row)
    return pd.DataFrame(rows)


def load_ways(path: Path) -> dict[int, dict[str, Any]]:
    """Overpass line download as ``{way id: way}``."""
    return {w["id"]: w for w in json.loads(path.read_text(encoding="utf-8"))["elements"]}


def load_relations(path: Path) -> list[dict[str, Any]]:
    return list(json.loads(path.read_text(encoding="utf-8"))["elements"])
