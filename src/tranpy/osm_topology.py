"""Network topology from the OpenStreetMap export, with JAO circuits kept as they are.

The export decides which lines exist. A section is the stretch of ways of one voltage between
stations. Sections no modelled circuit runs on become circuits with typical values, labelled as
estimates; a modelled circuit is never changed. Needs the ``[grid]`` extra.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from itertools import pairwise
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial import cKDTree

from tranpy_core._optional import require

from .osm_export import HV_VOLTS, IN_OPERATION

IN_MODEL = "in_model"
ADD = "add"
TWO_VOLTAGE = "two_voltage_unconfirmed"
OPEN_END = "open_end"
SPECIAL = "offshore_converter_or_cable_end"
NOT_OPERATING = "not_in_operation"
SHORT = "short"
CROSS_BORDER = "cross_border_end"
VOLTAGE_CONFLICT = "voltage_conflict"
"""Tagged one voltage, but a modelled circuit of the other voltage runs there."""

COVERED = 0.8
SCHEMATIC_MATCH = "matches_schematic_circuit"
"""Runs between the stations of a JAO circuit whose route is still schematic."""
MIN_KM = 1.0
OTHER_VOLTAGE = 0.2
MIN_RUN_KM = 0.3
MIN_FOLLOWED_KM = 5.0
"""A shorter stretch joining a modelled line is a route deviation, not a circuit of its own."""
ROUTE_DEVIATION = "route_deviation"
NEW_BUILD = "new_build_unconfirmed"
"""Tagged in operation, but public project documents do not confirm energisation."""
ROUTE_TOLERANCE_KM = 0.08
STATION_BUFFER_DEGREES = 0.004
NODE_DISTANCE_KM = 2.0
FOLLOW_LIMIT_KM = 5.0
FOLLOW_LIMIT_SHARE = 0.3
"""A followed end further than this (or than this share of the section) is a long way round."""
TEE_DISTANCE_KM = 3.0
"""A long followed end joins a model node of its voltage this close to the end of the ways."""
PARAMETER_SOURCE = "estimated: median per km of verified JAO circuits of this voltage"

_KM = np.array([0.63 * 111.2, 111.2])
_SPECIAL_STATION = re.compile(r"konverter|hgü|anlandepunkt|kabelüb|küa|kabelanfang", re.IGNORECASE)
_ABROAD = re.compile(r"liaison|bundesgrenze|grenze", re.IGNORECASE)
_SPECIAL_LINE = re.compile(r"ostwind|kontek|alegro", re.IGNORECASE)
_NEW_BUILD_380 = re.compile(r"garrel|cappeln|liedingen|pfungstadt|dauersberg", re.IGNORECASE)
"""Conneforde-Cloppenburg-Merzen, Liedingen switchgear (NEP P228), Urberach-Pfungstadt-Weinheim
(NEP M60) and EnLAG 19 Dauersberg-Eiserfeld, where JAO keeps the circuit at 220 kV."""


def voltage_levels(voltages: Any) -> set[int]:
    """Model voltage levels (220, 380) in a semicolon-separated list of volts."""
    levels = set()
    for text in re.findall(r"[\d.]+", str(voltages)):
        volts = float(text)
        if HV_VOLTS <= volts <= 420_000:
            levels.add(380 if volts >= 300_000 else 220)
    return levels


def _length(xy: npt.NDArray[np.float64]) -> float:
    return float(np.hypot(*np.diff(xy, axis=0).T).sum())


def _dense(xy: npt.NDArray[np.float64], step_km: float = 0.05) -> npt.NDArray[np.float64]:
    parts = []
    for start, end in pairwise(xy):
        count = max(1, int(np.hypot(*(end - start)) / step_km))
        parts.append(start + np.linspace(0, 1, count, endpoint=False)[:, None] * (end - start))
    return np.vstack([*parts, xy[-1:]])


def _route_xy(route: Any) -> npt.NDArray[np.float64]:
    """Route as km coordinates, from lon/lat pairs or a list of ``{"lon", "lat"}`` points."""
    if isinstance(route[0], Mapping):
        points = np.array([[p["lon"], p["lat"]] for p in route], dtype=float)
    else:
        points = np.asarray(route, dtype=float)[:, :2]
    return np.asarray(points * _KM, dtype=np.float64)


def _route_trees(circuits: pd.DataFrame) -> dict[int, cKDTree]:
    points: dict[int, list[npt.NDArray[np.float64]]] = {220: [], 380: []}
    for kv, route in zip(circuits["kv"], circuits["route"], strict=True):
        if isinstance(route, (list, np.ndarray)) and len(route) >= 2:
            xy = _route_xy(route)
            points[380 if kv >= 300 else 220].append(_dense(xy))
    return {kv: cKDTree(np.vstack(parts)) for kv, parts in points.items() if parts}


def _runs(
    coords: list[tuple[float, float]],
    xy: npt.NDArray[np.float64],
    kv: int,
    trees: Mapping[int, cKDTree],
) -> list[tuple[list[tuple[float, float]], float, float]]:
    """Way split into stretches on and off the modelled routes: (coords, coverage, other)."""
    own = trees.get(kv)
    other = trees.get(600 - kv)
    states: list[bool] = []
    for start, end in pairwise(xy):
        dense = _dense(np.vstack([start, end]))
        states.append(
            own is not None and bool((own.query(dense)[0] <= ROUTE_TOLERANCE_KM).mean() >= 0.5)
        )
    lengths = np.hypot(*np.diff(xy, axis=0).T)
    # stretches shorter than this are matching noise, not a change of state
    index = 0
    while index < len(states):
        stop = index
        while stop < len(states) and states[stop] == states[index]:
            stop += 1
        if lengths[index:stop].sum() < MIN_RUN_KM and (index > 0 or stop < len(states)):
            fill = states[index - 1] if index > 0 else states[stop]
            states[index:stop] = [fill] * (stop - index)
        index = stop
    pieces = []
    index = 0
    while index < len(states):
        stop = index
        while stop < len(states) and states[stop] == states[index]:
            stop += 1
        piece = coords[index : stop + 1]
        share = 0.0
        if other is not None:
            dense = _dense(xy[index : stop + 1])
            share = float((other.query(dense)[0] <= ROUTE_TOLERANCE_KM).mean())
        pieces.append((piece, 1.0 if states[index] else 0.0, share))
        index = stop
    return pieces


def _ways(ways: pd.DataFrame, trees: Mapping[int, cKDTree]) -> list[dict[str, Any]]:
    wkt = require("shapely.wkt", "grid")
    rows = []
    table: Any = ways
    for way in table.itertuples():
        levels = voltage_levels(way.voltages)
        frequency = str(getattr(way, "frequency", ""))
        if not levels or frequency == "0":
            continue
        geometry = wkt.loads(way.geometry_wkt)
        for line in getattr(geometry, "geoms", [geometry]):
            coords = [(float(c[0]), float(c[1])) for c in line.coords]
            xy = np.array(coords) * _KM
            others = {float(v) for v in re.findall(r"[\d.]+", str(way.voltages))}
            for kv in levels:
                for piece, covered, other in _runs(coords, xy, kv, trees):
                    rows.append(
                        {
                            "id": way.id,
                            "kv": kv,
                            "coords": piece,
                            "coverage": covered,
                            "other_coverage": other,
                            "other_voltages": len(others) > 1,
                            "status": way.status,
                            "two_voltage": len(levels) > 1,
                            "circuits": way.circuits,
                            "ref": way.ref,
                            "name": way.name,
                            "operator": way.operator,
                        }
                    )
    return rows


def _find(parent: dict[Any, Any], item: Any) -> Any:
    parent.setdefault(item, item)
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def _join(values: Iterable[Any]) -> str:
    return ";".join(sorted({str(v) for v in values if pd.notna(v) and str(v)}))


def _classify(section: Mapping[str, Any]) -> str:
    if section["km"] < MIN_KM:
        return SHORT
    if section["status"] != IN_OPERATION:
        return NOT_OPERATING
    if section["coverage"] >= COVERED:
        return IN_MODEL
    if section["other_coverage"] >= OTHER_VOLTAGE and not section["two_voltage"]:
        return VOLTAGE_CONFLICT
    names = section["station_names"]
    if _SPECIAL_STATION.search(names) or _SPECIAL_LINE.search(section["name"]):
        return SPECIAL
    if _ABROAD.search(section["name"]):
        return CROSS_BORDER
    if section["station_count"] < 2:
        return OPEN_END
    return TWO_VOLTAGE if section["two_voltage"] else ADD


def build_sections(
    ways: pd.DataFrame, substations: pd.DataFrame, circuits: pd.DataFrame
) -> pd.DataFrame:
    """Sections of the export between stations, each compared with the modelled circuits.

    Ways sharing a vertex outside a station are one section, so a branching section can reach
    more than two stations. ``topology`` holds the result of the comparison.
    """
    shapely = require("shapely", "grid")
    polygons = substations[substations["shape"].eq("polygon")].reset_index(drop=True)
    shapes = shapely.from_wkt(polygons["geometry_wkt"].to_numpy())
    tree = shapely.STRtree(shapes)
    found: dict[tuple[float, float], int | None] = {}

    def station(point: tuple[float, float]) -> int | None:
        if point not in found:
            hits = tree.query(
                shapely.Point(point).buffer(STATION_BUFFER_DEGREES), predicate="intersects"
            )
            found[point] = int(hits[0]) if len(hits) else None
        return found[point]

    rows = _ways(ways, _route_trees(circuits))
    sections: list[dict[str, Any]] = []
    for kv in (220, 380):
        for on_route in (True, False):
            level = [r for r in rows if r["kv"] == kv and (r["coverage"] > 0) == on_route]
            if level:
                sections += _group(level, station, polygons)
    _follow_modelled_towers(sections, polygons)
    table = pd.DataFrame(sections)
    return table.assign(section=[f"osm-{i}" for i in range(len(table))]).drop(
        columns=["vertices", "ends"]
    )


def _follow_modelled_towers(sections: list[dict[str, Any]], polygons: pd.DataFrame) -> None:
    """Close open ends that join a modelled line, at the nearest station along that line.

    A circuit often leaves towers it shares with a modelled circuit: the stretch alone then ends
    on the shared line, and the station it serves lies further along. ``extra_km`` holds the
    straight distance to that station, outside the export length of the section.
    ``follow_ends`` keeps each such end as ``[lon, lat, station, km]``, so a long way round can
    be replaced by the junction at the end of the ways (:func:`added_circuits`).
    """
    centres = {
        str(i): (float(x), float(y))
        for i, x, y in zip(polygons["id"], polygons["lon"], polygons["lat"], strict=True)
    }
    names = dict(zip(polygons["id"].astype(str), polygons["name"], strict=True))
    modelled: dict[tuple[int, Any], list[dict[str, Any]]] = {}
    for section in sections:
        if section["topology"] == IN_MODEL or section["coverage"] > 0:
            for vertex in section["vertices"]:
                modelled.setdefault((section["kv"], vertex), []).append(section)
    for section in sections:
        if section["topology"] != OPEN_END or section["two_voltage"]:
            continue
        found = set(section["station_ids"])
        for vertex in section["ends"]:
            start = np.array(vertex) * _KM
            options = {
                station
                for other in modelled.get((section["kv"], vertex), [])
                for station in other["station_ids"]
                if station not in found and station in centres
            }
            if not options:
                continue
            nearest = min(
                options, key=lambda c: float(np.hypot(*(np.array(centres[c]) * _KM - start)))
            )
            distance = float(np.hypot(*(np.array(centres[nearest]) * _KM - start)))
            section["extra_km"] += distance
            section["follow_ends"].append([vertex[0], vertex[1], nearest, distance])
            found.add(nearest)
            section["followed"] = True
        if len(found) >= 2:
            section["station_ids"] = sorted(found)
            section["station_count"] = len(found)
            section["station_names"] = _join(names.get(i) for i in found)
            section["topology"] = (
                _classify(section) if section["km"] >= MIN_FOLLOWED_KM else ROUTE_DEVIATION
            )


def _group(
    level: list[dict[str, Any]], station: Any, polygons: pd.DataFrame
) -> list[dict[str, Any]]:
    """Ways of one voltage joined at shared vertices outside stations."""
    use = Counter(c for r in level for c in r["coords"])
    parts = []
    for way in level:
        coords = way["coords"]
        cuts = [0, *[k for k in range(1, len(coords) - 1) if use[coords[k]] > 1]]
        cuts.append(len(coords) - 1)
        parts += [(way, coords[a : b + 1]) for a, b in pairwise(cuts)]
    parent: dict[Any, Any] = {}
    for index, (_, coords) in enumerate(parts):
        _find(parent, ("part", index))
        for point in (coords[0], coords[-1]):
            if station(point) is None:
                parent[_find(parent, ("part", index))] = _find(parent, ("vertex", point))
    groups: dict[Any, list[int]] = {}
    for index in range(len(parts)):
        groups.setdefault(_find(parent, ("part", index)), []).append(index)
    sections = []
    for indices in groups.values():
        km = covered = other = 0.0
        stations: set[int] = set()
        members: dict[Any, dict[str, Any]] = {}
        by_status: dict[str, float] = {}
        for index in indices:
            way, coords = parts[index]
            length = _length(np.array(coords) * _KM)
            km += length
            covered += length * way["coverage"]
            other += length * way["other_coverage"]
            members[way["id"]] = way
            by_status[way["status"]] = by_status.get(way["status"], 0.0) + length
            stations |= {s for s in (station(coords[0]), station(coords[-1])) if s is not None}
        tags = [m["circuits"] for m in members.values() if pd.notna(m["circuits"])]
        section: dict[str, Any] = {
            "kv": level[0]["kv"],
            "km": km,
            "coverage": covered / km if km else 0.0,
            "other_coverage": other / km if km else 0.0,
            "status": max(by_status, key=lambda k: by_status[k]),
            "other_voltages": any(m["other_voltages"] for m in members.values()),
            "two_voltage": any(m["two_voltage"] for m in members.values()),
            "station_count": len(stations),
            "station_ids": [str(polygons["id"].iloc[s]) for s in sorted(stations)],
            "station_names": _join(polygons["name"].iloc[s] for s in stations),
            "circuits": Counter(map(str, tags)).most_common(1)[0][0] if tags else None,
            "ref": _join(m["ref"] for m in members.values()),
            "name": _join(m["name"] for m in members.values()),
            "operator": _join(m["operator"] for m in members.values()),
            "way_ids": sorted(str(i) for i in members),
            "paths": [[list(c) for c in parts[i][1]] for i in indices],
            "vertices": {c for i in indices for c in parts[i][1]},
            "ends": {c for i in indices for c in (parts[i][1][0], parts[i][1][-1])},
            "followed": False,
            "extra_km": 0.0,
            "follow_ends": [],
        }
        sections.append({**section, "topology": _classify(section)})
    return sections


def added_transformers(
    export: pd.DataFrame, transformers: pd.DataFrame, nodes: pd.DataFrame
) -> pd.DataFrame:
    """380/220 kV transformers the export shows at sites where the model links neither voltage.

    Each site pair gets one unit with the median values of the modelled units, as estimates.
    """
    primary = pd.to_numeric(export["v_primary"], errors="coerce")
    secondary = pd.to_numeric(export["v_secondary"], errors="coerce")
    units = export[(primary >= 300_000) & secondary.between(HV_VOLTS, 299_999)]
    located = nodes.dropna(subset=["lon", "lat"])
    high, low = located[located["kv"] >= 300], located[located["kv"] < 300]
    points = units[["lon", "lat"]].to_numpy(float) * _KM
    to_high, at_high = cKDTree(high[["lon", "lat"]].to_numpy(float) * _KM).query(points)
    to_low, at_low = cKDTree(low[["lon", "lat"]].to_numpy(float) * _KM).query(points)
    close = (to_high <= NODE_DISTANCE_KM) & (to_low <= NODE_DISTANCE_KM)
    pairs = {
        (str(high["node"].iloc[a]), str(low["node"].iloc[b]))
        for a, b, near in zip(at_high, at_low, close, strict=True)
        if near
    }
    # the network requires both ends in one substation; other pairs need the site split resolved
    sites = dict(zip(located["node"], located.get("site", located["node"]), strict=True))
    if "site" in located:
        pairs = {(hv, lv) for hv, lv in pairs if sites[hv] == sites[lv]}
    linked = set(zip(transformers["node_hv"], transformers["node_lv"], strict=True))
    plain = transformers[~transformers["phase_shifter"].astype(bool)]
    typical = plain[["kv_hv", "kv_lv", "r_ohm", "x_ohm", "b_us", "g_us", "imax_a"]].median()
    rows = [
        {
            **typical.to_dict(),
            "eic": f"osm-tr-{hv}-{lv}",
            "name": f"Transformer {hv} / {lv}",
            "tso": None,
            "node_hv": hv,
            "node_lv": lv,
            "phase_shifter": False,
            "data_quality": "estimated",
            "parameter_source": "estimated: median of modelled JAO transformers; "
            "existence from OpenStreetMap export",
        }
        for hv, lv in sorted(pairs - linked)
    ]
    return pd.DataFrame(rows)


def with_osm_topology(
    lines: pd.DataFrame, nodes: pd.DataFrame, added: pd.DataFrame, new_nodes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lines and nodes of the model extended by the added ones; modelled rows stay untouched."""
    extra = added.rename(columns={"mrid": "eic"}).drop(columns=["section"])
    extra = extra.assign(
        tie_line=False,
        hvdc=False,
        route=pd.Series([np.empty((0, 2)) for _ in range(len(extra))], index=extra.index),
        route_quality="osm_only",
    )
    return (
        pd.concat([lines, extra], ignore_index=True),
        pd.concat([nodes, new_nodes], ignore_index=True),
    )


def export_length_km(ways: pd.DataFrame) -> float:
    """Length of the AC ways of the export per voltage they carry."""
    wkt = require("shapely.wkt", "grid")
    total = 0.0
    for voltages, frequency, text in zip(
        ways["voltages"], ways["frequency"], ways["geometry_wkt"], strict=True
    ):
        count = len(voltage_levels(voltages))
        if count and str(frequency) != "0":
            coords = np.array(wkt.loads(text).coords)[:, :2]
            total += count * _length(coords * _KM)
    return total


def typical_values(circuits: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Median R, X, B per km and median rating of verified circuits, per voltage."""
    values = {}
    frame = circuits[circuits["route_quality"].eq("verified") & (circuits["length_km"] > 2)]
    for kv in (220, 380):
        level = frame[(frame["kv"] >= 300) == (kv == 380)]
        values[kv] = {
            "r": float((level["r_ohm"] / level["length_km"]).median()),
            "x": float((level["x_ohm"] / level["length_km"]).median()),
            "b": float((level["b_us"] / level["length_km"]).median()),
            "i": float(level["imax_a"].median()),
        }
    return values


_CIRCUIT_MARK = re.compile(r"\b\d[a-z]\b")


def _circuit_count(tag: Any, other_voltages: bool = False, name: str = "") -> tuple[int, str]:
    """Circuits of a section. A tag on a way with several voltages may count them all.

    Amprion names list its circuits, such as ``220 kV Niederaußem 1a|2b|3c|4d``.
    """
    marks = set(_CIRCUIT_MARK.findall(name))
    if other_voltages and marks:
        return len(marks), "circuit marks in the OpenStreetMap name"
    if other_voltages:
        return 1, "assumed one: tag covers other voltages"
    count = pd.to_numeric(str(tag).split(";")[0], errors="coerce")
    return (int(count), "OpenStreetMap") if count == count and count > 0 else (1, "assumed one")


def long_way_round(km: float, section_km: float) -> bool:
    """A followed end too far from the end of its ways to be the station it serves."""
    return km > FOLLOW_LIMIT_KM or km > FOLLOW_LIMIT_SHARE * section_km


def added_circuits(
    sections: pd.DataFrame,
    substations: pd.DataFrame,
    circuits: pd.DataFrame,
    nodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Circuits and new nodes of the ``add`` sections, with labelled typical values.

    A station within :data:`NODE_DISTANCE_KM` of a model node of its voltage uses that node. A
    section reaching more than two stations gets a junction node, with the length shared equally.
    An end followed along a modelled line to a distant station (:func:`long_way_round`) joins the
    model node within :data:`TEE_DISTANCE_KM` of the end of its ways instead, such as a tee.
    """
    typical = typical_values(circuits)
    stations = substations[substations["shape"].eq("polygon")].set_index(
        substations.loc[substations["shape"].eq("polygon"), "id"].astype(str)
    )
    trees = {}
    for kv in (220, 380):
        level = nodes[(nodes["kv"] >= 300) == (kv == 380)].dropna(subset=["lon", "lat"])
        trees[kv] = (cKDTree(level[["lon", "lat"]].to_numpy(float) * _KM), level["node"].to_numpy())
    german = nodes
    if "country" in nodes:
        german = nodes[nodes["country"].eq("DEU")]
    with_area = german.dropna(subset=["lon", "lat", "tso"])
    with_area = with_area[with_area["tso"].astype(str).str.len() > 0]
    area_tree = cKDTree(with_area[["lon", "lat"]].to_numpy(float) * _KM)
    areas = with_area["tso"].to_numpy()

    def area_at(lon: float, lat: float) -> str:
        """Control area of the nearest model node (the export has no control area)."""
        return str(areas[area_tree.query(np.array([lon, lat]) * _KM)[1]])

    new_nodes: dict[str, dict[str, Any]] = {}

    def node_of(station_id: str, kv: int) -> str:
        row: Any = stations.loc[station_id]
        distance, index = trees[kv][0].query(np.array([row["lon"], row["lat"]]) * _KM)
        if distance <= NODE_DISTANCE_KM:
            return str(trees[kv][1][index])
        node = f"osm/{station_id}|{kv}"
        new_nodes[node] = {
            "node": node,
            "site": f"osm/{station_id}",
            "kv": kv,
            "names": [str(row["name"])],
            "kind": "substation",
            "tso": area_at(float(row["lon"]), float(row["lat"])),
            "lon": float(row["lon"]),
            "lat": float(row["lat"]),
            "location_source": "OpenStreetMap export",
            "country": "DEU",
        }
        return node

    rows = []

    def add(section: Any, a: str, b: str, km: float, suffix: str) -> None:
        count, count_source = _circuit_count(section.circuits, section.other_voltages, section.name)
        values = typical[section.kv]
        for number in range(count):
            rows.append(
                {
                    "mrid": f"{section.section}{suffix}-{number}",
                    "section": section.section,
                    "name": section.name or section.station_names,
                    "tso": section.operator,
                    "node0": a,
                    "node1": b,
                    "kv": float(section.kv),
                    "length_km": km,
                    "r_ohm": values["r"] * km,
                    "x_ohm": values["x"] * km,
                    "b_us": values["b"] * km,
                    "imax_a": values["i"],
                    "status": section.status,
                    "route_source": "OpenStreetMap export",
                    "data_quality": "estimated",
                    "parameter_source": PARAMETER_SOURCE,
                    "circuit_count_source": count_source,
                }
            )

    schematic = circuits[circuits["route_quality"].eq("estimated")]
    sites = dict(zip(nodes["node"], nodes.get("site", nodes["node"]), strict=True))
    open_pairs = {
        frozenset((sites.get(a, a), sites.get(b, b)))
        for a, b in zip(schematic["node0"], schematic["node1"], strict=True)
    }
    new_build = (
        sections["topology"].eq(ADD)
        & sections["kv"].eq(380)
        & sections["station_names"].str.contains(_NEW_BUILD_380)
    )
    sections.loc[new_build, "topology"] = NEW_BUILD
    chosen: Any = sections[sections["topology"].eq(ADD)]
    neighbours: dict[str, set[str]] = {}
    for a, b in zip(circuits["node0"], circuits["node1"], strict=True):
        neighbours.setdefault(str(a), set()).add(sites.get(b, b))
        neighbours.setdefault(str(b), set()).add(sites.get(a, a))
    joined: Counter[str] = Counter()

    def tee_for(section: Any, lon: float, lat: float, own: set[str]) -> tuple[str, float] | None:
        """Model node near the end of the ways; at one place, the tee whose circuit continues.

        Tees sharing a position are told apart by a circuit to a station the section also
        reaches; lines without one join the tee another line already took there, as lines
        ending at one tower meet (at Oege both ENERVIE ways end at tower 38).
        """
        tree, names = trees[section.kv]
        point = np.array([lon, lat]) * _KM
        found = [
            (str(names[i]), float(np.hypot(*(tree.data[i] - point))))
            for i in tree.query_ball_point(point, TEE_DISTANCE_KM)
        ]
        found = [f for f in found if f[0] not in own]
        if not found:
            return None
        nearest = min(d for _, d in found)
        tied = [f for f in found if f[1] <= nearest + 0.1]
        return min(tied, key=lambda f: (not neighbours.get(f[0], set()) & own, -joined[f[0]], f))

    def matches(section: Any) -> bool:
        own = {sites.get(n, n) for n in (node_of(s, section.kv) for s in section.station_ids)}
        return any(
            neighbours.get(str(name), set()) & own
            for lon, lat, _, km in getattr(section, "follow_ends", None) or []
            if long_way_round(km, section.km)
            for name in [
                trees[section.kv][1][i]
                for i in trees[section.kv][0].query_ball_point(
                    np.array([lon, lat]) * _KM, TEE_DISTANCE_KM
                )
            ]
        )

    for section in sorted(chosen.itertuples(), key=lambda s: not matches(s)):
        by_station = {s: node_of(s, section.kv) for s in section.station_ids}
        extra_km = section.extra_km
        for lon, lat, station_id, km in getattr(section, "follow_ends", None) or []:
            if not long_way_round(km, section.km):
                continue
            own = {sites.get(n, n) for n in by_station.values()} | set(by_station.values())
            tee = tee_for(section, lon, lat, own)
            if tee is not None:
                node, distance = tee
                by_station[str(station_id)] = node
                joined[node] += 1
                extra_km += distance - km
                row = sections["section"].eq(section.section)
                sections.loc[row, "joined_at"] = node
                sections.loc[row, "extra_km"] = extra_km
        ends = list(dict.fromkeys(by_station.values()))
        if len(ends) < 2:
            continue
        if len(ends) == 2 and frozenset(sites.get(e, e) for e in ends) in open_pairs:
            sections.loc[sections["section"].eq(section.section), "topology"] = SCHEMATIC_MATCH
            continue
        if len(ends) == 2:
            add(section, ends[0], ends[1], section.km + extra_km, "")
            continue
        first: Any = stations.loc[section.station_ids[0]]
        junction = f"{section.section}|{section.kv}"
        new_nodes[junction] = {
            "node": junction,
            "site": section.section,
            "kv": section.kv,
            "names": [],
            "kind": "junction",
            "tso": area_at(float(first["lon"]), float(first["lat"])),
            "lon": float(first["lon"]),
            "lat": float(first["lat"]),
            "location_source": "position of first station, junction not located",
            "country": "DEU",
        }
        for index, end in enumerate(ends):
            add(section, junction, end, section.km / len(ends), f"-{index}")
    return pd.DataFrame(rows), pd.DataFrame(list(new_nodes.values()))


def check_topology(
    sections: pd.DataFrame,
    added: pd.DataFrame,
    new_nodes: pd.DataFrame,
    circuits: pd.DataFrame,
    nodes: pd.DataFrame,
    export_km: float | None = None,
) -> list[str]:
    """Problems of the added topology (empty: all checks passed).

    With ``export_km``, every kilometre of the export must be in exactly one section.
    """
    problems = []
    if export_km is not None and abs(sections["km"].sum() - export_km) > 0.01 * export_km:
        problems.append(f"sections hold {sections['km'].sum():.0f} of {export_km:.0f} km")
    if added.empty:
        return ["no circuits added"]
    if added["mrid"].duplicated().any():
        problems.append("duplicate circuit ids")
    known = set(nodes["node"]) | (set(new_nodes["node"]) if len(new_nodes) else set())
    unknown = (~added["node0"].isin(known)).sum() + (~added["node1"].isin(known)).sum()
    if unknown:
        problems.append(f"{unknown} circuit ends on unknown nodes")
    if (added["node0"] == added["node1"]).any():
        problems.append("circuits with both ends on one node")
    if not added["status"].eq(IN_OPERATION).all():
        problems.append("added circuits not in operation")
    if not added["data_quality"].eq("estimated").all():
        problems.append("added circuits not labelled as estimates")
    bad = sections["section"].isin(added["section"]) & ~sections["topology"].eq(ADD)
    if bad.any():
        problems.append(f"{int(bad.sum())} circuits from sections not classed as add")
    values = added[["r_ohm", "x_ohm", "b_us", "imax_a", "length_km"]]
    if not (values.notna().all(axis=None) and (values > 0).all(axis=None)):
        problems.append("missing or non-positive values")
    if set(added["mrid"]) & set(circuits.get("eic", pd.Series(dtype=str))):
        problems.append("ids collide with JAO circuits")
    return problems
