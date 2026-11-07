"""Physical tower lines (OpenStreetMap ``power=line`` ways) for the grid map (display only).

:func:`read_tower_ways` loads the 220/380 kV ways (also mixed-voltage ways that carry a
220 or 380 kV circuit) from the OpenStreetMap line export (and optional Overpass JSON);
:func:`split_by_towers` finds the ways each drawn model circuit runs on, cuts the circuit
at way boundaries and gives every piece the circuit's slot on that way, so circuits that
share towers are drawn side by side along the whole shared stretch (MapLibre
``line-offset`` / QGIS offset per piece) and branch where a circuit leaves. It also
returns the ``tower_lines`` layer: each way with the model circuits it carries and the
OpenStreetMap circuit count, flagging mismatches. The model is not changed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tranpy_core._optional import require

__all__ = [
    "MATCH_M",
    "MIN_RUN_M",
    "TOWER_COLUMNS",
    "assign_runs",
    "read_tower_ways",
    "split_by_towers",
    "tower_circuits_220_380",
]

CRS = "EPSG:4326"
METRIC_CRS = "EPSG:3035"
TOWER_KV = (220_000, 380_000)
MATCH_M = 60.0
"""A circuit point this close to a way (and nearest to it) runs on that way."""
SAMPLE_M = 25.0
"""Spacing of the sample points along a circuit."""
MIN_RUN_M = 150.0
"""Shorter stretches on a way (crossings, station entries) do not count, unless they cover
half of a way that is itself short."""
TOWER_COLUMNS = (
    "osm_id",
    "name",
    "ref",
    "operator",
    "voltages",
    "max_kv",
    "mixed_voltage",
    "circuits_tag",
    "cables",
    "location",
    "construction",
    "disused",
)


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _number(value: Any) -> float:
    try:
        return float(str(value).split(";")[0])
    except (TypeError, ValueError):
        return float("nan")


def _kv_parts(voltages: str) -> list[int]:
    out = []
    for part in voltages.split(";"):
        try:
            out.append(int(float(part)))
        except ValueError:
            out.append(0)
    return out


def tower_circuits_220_380(voltages: str, circuits: Any, cables: Any) -> float:
    """OpenStreetMap count of 220/380 kV circuits on a way (NaN when the tags do not say).

    ``voltage``, ``cables`` and ``circuits`` are aligned ``;`` lists in OpenStreetMap. A way of
    only 220/380 kV counts ``circuits`` (or ``cables`` / 3); a mixed-voltage way counts the
    cables of its 220/380 kV parts when the lists align, else it is unknown.
    """
    kv = _kv_parts(voltages)
    hv = [k in TOWER_KV for k in kv]
    circ = [_number(c) for c in _text(circuits).split(";")] if _text(circuits) else []
    cab = [_number(c) for c in _text(cables).split(";")] if _text(cables) else []
    if all(hv):
        if circ and len(circ) in (1, len(kv)) and all(np.isfinite(circ)):
            return float(sum(circ))
        if cab and len(cab) in (1, len(kv)) and all(np.isfinite(cab)):
            return float(sum(c // 3 for c in cab))
        return float("nan")
    for parts, per in ((circ, 1.0), (cab, 3.0)):
        if len(parts) == len(kv) and all(np.isfinite(parts)):
            return float(sum(p // per for p, h in zip(parts, hv, strict=True) if h))
    return float("nan")


def read_tower_ways(path: str | Path, *, overpass: Sequence[str | Path] = ()) -> Any:
    """220/380 kV ``power=line`` ways (EPSG:4326 GeoDataFrame, :data:`TOWER_COLUMNS`).

    ``path`` is the OpenStreetMap line export (GeoParquet with ``id``, ``voltages``,
    ``circuits``, ``cables``, ...); ``overpass`` are Overpass JSON downloads (``out geom``)
    adding ways outside it (e.g. western Austria); ways already in ``path`` are kept from it.
    """
    gpd = require("geopandas", "grid")
    geom = require("shapely.geometry", "grid")
    src = gpd.read_parquet(path).to_crs(CRS)
    frame = pd.DataFrame(
        {
            "osm_id": src["id"].astype("int64"),
            "name": src["name"].map(_text),
            "ref": src["ref"].map(_text),
            "operator": src["operator"].map(_text),
            "voltages": src["voltages"].map(_text),
            "circuits_raw": src["circuits"],
            "cables": src["cables"].map(_text),
            "location": src["location"].map(_text),
            "construction": src["construction"].fillna(False).astype(bool),
            "disused": src["disused"].fillna(False).astype(bool),
        }
    )
    frames = [gpd.GeoDataFrame(frame, geometry=src.geometry.to_numpy(), crs=CRS)]
    for extra in overpass:
        rows, shapes = [], []
        for e in json.loads(Path(extra).read_text(encoding="utf-8"))["elements"]:
            tags = e.get("tags", {})
            if e.get("type") != "way" or tags.get("power") != "line" or len(e["geometry"]) < 2:
                continue
            rows.append(
                {
                    "osm_id": int(e["id"]),
                    "name": tags.get("name", ""),
                    "ref": tags.get("ref", ""),
                    "operator": tags.get("operator", ""),
                    "voltages": tags.get("voltage", ""),
                    "circuits_raw": tags.get("circuits"),
                    "cables": tags.get("cables", ""),
                    "location": tags.get("location", "overhead"),
                    "construction": False,
                    "disused": False,
                }
            )
            shapes.append(geom.LineString([(q["lon"], q["lat"]) for q in e["geometry"]]))
        if rows:
            frames.append(gpd.GeoDataFrame(pd.DataFrame(rows), geometry=shapes, crs=CRS))
    ways = pd.concat(frames, ignore_index=True).drop_duplicates("osm_id", keep="first")
    kv = ways["voltages"].map(_kv_parts)
    ways = ways[[any(k in TOWER_KV for k in v) for v in kv]].copy()
    kv = ways["voltages"].map(_kv_parts)
    ways["max_kv"] = [max(k for k in v if k in TOWER_KV) / 1000 for v in kv]
    ways["mixed_voltage"] = [any(k not in TOWER_KV for k in v) for v in kv]
    ways["circuits_tag"] = [
        tower_circuits_220_380(v, c, b)
        for v, c, b in zip(ways["voltages"], ways["circuits_raw"], ways["cables"], strict=True)
    ]
    ways = ways[[*TOWER_COLUMNS, "geometry"]].reset_index(drop=True)
    return gpd.GeoDataFrame(ways, geometry="geometry", crs=CRS)


def assign_runs(
    circuit_m: Any, ways_m: Any, way_length_m: Sequence[float]
) -> list[tuple[float, float, int]]:
    """Stretches ``(start_m, end_m, way index or -1)`` of a metric circuit line on the ways.

    Points every :data:`SAMPLE_M` take the nearest way within :data:`MATCH_M`; stretches
    shorter than :data:`MIN_RUN_M` (unless half of a short way) are dropped and gaps shorter
    than that join the stretch before them. Boundaries lie half-way between sample points.
    """
    shapely = require("shapely", "grid")
    length = float(circuit_m.length)
    if length <= 0:
        return [(0.0, length, -1)]
    dist = np.append(np.arange(0.0, length, SAMPLE_M), length)
    pts = shapely.line_interpolate_point(circuit_m, dist)
    tree = shapely.STRtree(ways_m)
    pi, wi = tree.query_nearest(pts, max_distance=MATCH_M, all_matches=False)
    label = np.full(len(dist), -1, dtype=int)
    label[pi] = wi

    def runs_of(lab: np.ndarray) -> list[list[int]]:
        out: list[list[int]] = []
        for i, v in enumerate(lab.tolist()):
            if out and out[-1][2] == v:
                out[-1][1] = i
            else:
                out.append([i, i, v])
        return out

    def span(r: list[int]) -> float:
        lo = dist[r[0] - 1] if r[0] else dist[0]
        hi = dist[r[1] + 1] if r[1] + 1 < len(dist) else dist[-1]
        return float(hi - lo) / 2.0 + float(dist[r[1]] - dist[r[0]]) / 2.0

    for r in runs_of(label):
        w = r[2]
        if w >= 0 and span(r) < MIN_RUN_M and span(r) < 0.5 * way_length_m[w]:
            label[r[0] : r[1] + 1] = -1
    runs = runs_of(label)
    for k, r in enumerate(runs):
        if r[2] == -1 and span(r) < MIN_RUN_M and len(runs) > 1:
            fill = runs[k - 1][2] if k else runs[k + 1][2]
            label[r[0] : r[1] + 1] = fill
    out = []
    runs = runs_of(label)
    for k, (a, b, w) in enumerate(runs):
        start = 0.0 if k == 0 else float(dist[a - 1] + dist[a]) / 2.0
        end = length if k == len(runs) - 1 else float(dist[b] + dist[b + 1]) / 2.0
        out.append((start, end, w))
    return out


def _canonical_forward(coords: Sequence[tuple[float, float]]) -> bool:
    """A way's reference direction: eastward, or northward when it runs north-south."""
    (x0, y0), (x1, y1) = coords[0], coords[-1]
    dx, dy = x1 - x0, y1 - y0
    return dx > 0 if abs(dx) > 0.3 * abs(dy) else dy > 0


def split_by_towers(lines: Any, towers: Any) -> tuple[Any, Any]:
    """Cut the drawn circuits at tower-line boundaries and slot them per tower line.

    Args:
        lines: ``lines`` layer (one row per circuit, oriented display geometry, with
            ``circuit_no`` / ``circuits`` from the corridor grouping as fallback).
        towers: :func:`read_tower_ways`.

    Returns:
        ``(pieces, tower_lines)``: the lines layer with one row per piece (``circuit_no`` /
        ``circuits`` the slot on its tower line, ``tower_way`` the OpenStreetMap way ids,
        ``piece``, ``pieces``, ``label_piece`` for the longest piece) and the tower lines with
        the circuits they carry (``model_circuits``, ``model_names``, ``model_eics``,
        ``model_kv``) and ``mismatch`` against ``circuits_tag``.
    """
    gpd = require("geopandas", "grid")
    shapely = require("shapely", "grid")
    ops = require("shapely.ops", "grid")
    lines = lines.reset_index(drop=True)
    ways_m = towers.to_crs(METRIC_CRS).geometry.to_numpy()
    way_len = shapely.length(ways_m).tolist()
    lines_m = lines.to_crs(METRIC_CRS).geometry.to_numpy()
    eligible = (
        ~lines["hvdc"].astype(bool) & ~lines["route_quality"].isin(["straight", "in_station"])
    ).to_numpy()
    runs: list[list[tuple[float, float, int]]] = []
    carried: dict[int, set[int]] = {}
    for i, g in enumerate(lines_m):
        r = assign_runs(g, ways_m, way_len) if eligible[i] and len(ways_m) else []
        runs.append(r)
        for _, _, w in r:
            if w >= 0:
                carried.setdefault(w, set()).add(i)
    kv = lines["voltage_kv"].astype(float).tolist()
    names = lines["name"].astype(str).tolist()
    slot: dict[tuple[int, int], tuple[int, int]] = {}
    for w, members in carried.items():
        order = sorted(members, key=lambda i: (-kv[i], names[i], i))
        for k, i in enumerate(order):
            slot[(w, i)] = (k, len(order))
    forward = [_canonical_forward(list(g.coords)) for g in towers.geometry]
    rows: list[dict[str, Any]] = []
    shapes: list[Any] = []
    for i, row in enumerate(lines.drop(columns="geometry").to_dict("records")):
        g = lines.geometry.iloc[i]
        if not runs[i] or all(w < 0 for _, _, w in runs[i]):
            rows.append({**row, "tower_way": "", "piece": 0, "pieces": 1, "label_piece": True})
            shapes.append(g)
            continue
        total = float(lines_m[i].length)
        parts: list[tuple[float, float, tuple[int, int], bool, list[str]]] = []
        for start, end, w in runs[i]:
            if w >= 0:
                k, n = slot[(w, i)]
                way = ways_m[w]
                p0 = shapely.line_interpolate_point(lines_m[i], start)
                p1 = shapely.line_interpolate_point(lines_m[i], end)
                along = shapely.line_locate_point(way, p1) >= shapely.line_locate_point(way, p0)
                rev = along != forward[w]
                ids = [str(int(towers["osm_id"].iat[w]))]
            else:
                k, n, rev, ids = 0, 1, False, []
            last = parts[-1] if parts else None
            if last is not None and last[2] == (k, n) and last[3] == rev:
                parts[-1] = (last[0], end, (k, n), rev, last[4] + ids)
            else:
                parts.append((start, end, (k, n), rev, ids))
        lengths = [b - a for a, b, *_ in parts]
        longest = int(np.argmax(lengths))
        for p, (start, end, (k, n), rev, ids) in enumerate(parts):
            piece = ops.substring(g, start / total, end / total, normalized=True)
            if piece.geom_type != "LineString" or piece.length == 0:
                continue
            if rev:
                piece = type(piece)(list(piece.coords)[::-1])
            rows.append(
                {
                    **row,
                    "circuit_no": k,
                    "circuits": n,
                    "tower_way": ";".join(dict.fromkeys(ids)),
                    "piece": p,
                    "pieces": len(parts),
                    "label_piece": p == longest,
                }
            )
            shapes.append(piece)
    pieces = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=shapes, crs=lines.crs)

    out = towers.copy()
    on_way = [
        sorted(carried.get(w, set()), key=lambda i: (-kv[i], names[i])) for w in range(len(out))
    ]
    out["model_circuits"] = [len(m) for m in on_way]
    out["model_names"] = ["; ".join(names[i] for i in m) for m in on_way]
    out["model_eics"] = ["; ".join(_text(lines["eic"].iat[i]) for i in m) for m in on_way]
    out["model_kv"] = ["; ".join(f"{kv[i]:g}" for i in m) for m in on_way]
    tag = out["circuits_tag"].astype(float)
    model = out["model_circuits"]
    out["length_km"] = (shapely.length(ways_m) / 1000.0).round(3)
    out["mismatch"] = np.select(
        [model == 0, tag.isna(), model < tag, model > tag],
        ["no_model_circuit", "no_tag", "model_fewer", "model_more"],
        "",
    )
    return pieces, out


def empty_tower_lines() -> Any:
    """``tower_lines`` layer without ways (export without ``--towers``)."""
    gpd = require("geopandas", "grid")
    cols = [
        *TOWER_COLUMNS,
        "length_km",
        "model_circuits",
        "model_names",
        "model_eics",
        "model_kv",
        "mismatch",
    ]
    return gpd.GeoDataFrame(
        pd.DataFrame({c: pd.Series(dtype=object) for c in cols}), geometry=[], crs=CRS
    )
