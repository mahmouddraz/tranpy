"""High-voltage layers of the Open Infrastructure Map export (OpenStreetMap data, ODbL).

The export is a GeoPackage snapshot of Germany. It adds what the grid model's own OpenStreetMap
download lacks: pylons, ways under construction, substation outlines, transformers and switches.
Needs the ``[grid]`` extra.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from tranpy_core._optional import require

HV_VOLTS = 200_000
INFEED_VOLTS = 100_000
"""Lowest voltage of the transmission grid in this model."""

SOURCE = "Open Infrastructure Map export of OpenStreetMap data, ODbL"
IN_OPERATION = "in_operation"
UNDER_CONSTRUCTION = "under_construction"
PLANNED = "planned"
OUT_OF_SERVICE = "out_of_service"
STATUSES = (IN_OPERATION, UNDER_CONSTRUCTION, PLANNED, OUT_OF_SERVICE)
"""Lifecycle of every object in the model. All are kept; a study selects the ones it needs."""

_KM_PER_DEGREE = 111.2
_COS_LAT = np.cos(np.radians(51.0))
_TOWER_REF = re.compile(r"(?<!\d)(\d{3,4})\s*M\s*(\d+\w*)", re.IGNORECASE)


def latest_export(directory: Path) -> Path | None:
    """Newest ``<snapshot>/DEU.gpkg`` below ``directory``, or None."""
    found = sorted(directory.glob("*/DEU.gpkg"))
    return found[-1] if found else None


def _read(path: Path, layer: str, **options: Any) -> Any:
    geopandas = require("geopandas", "grid")
    return geopandas.read_file(path, layer=layer, engine="pyogrio", **options)


def _flat(frame: Any) -> pd.DataFrame:
    """Plain table with the geometry as WKT and its centre as ``lon``, ``lat``."""
    centre = frame.geometry.representative_point()
    table = pd.DataFrame(frame.drop(columns="geometry"))
    return table.assign(geometry_wkt=frame.geometry.to_wkt(), lon=centre.x, lat=centre.y)


def status(frame: pd.DataFrame) -> pd.Series:
    """Lifecycle status of each object, one of :data:`STATUSES`.

    The export marks construction and disuse. Construction with a start date after
    ``today`` still counts as ``under construction``: only an official source can say
    ``planned``.
    """
    construction = frame["construction"].fillna(False).astype(bool)
    disused = frame.get("disused", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    return pd.Series(
        np.where(construction, UNDER_CONSTRUCTION, np.where(disused, OUT_OF_SERVICE, IN_OPERATION)),
        index=frame.index,
    )


def _high_voltage(frame: Any, column: str = "max_voltage") -> Any:
    return frame[pd.to_numeric(frame[column], errors="coerce") >= HV_VOLTS]


def tower_line_number(ref: Any) -> str | None:
    """Line number of a pylon reference: ``2409`` from ``2409 M136``, else None."""
    match = _TOWER_REF.search(ref) if isinstance(ref, str) else None
    return match.group(1) if match else None


def _near_lines(towers: Any, lines: Any, distance_km: float) -> Any:
    points: list[Any] = []
    for line in lines.geometry:
        points.append(np.asarray(line.coords)[:, :2])
    xy = np.vstack(points) * [_COS_LAT * _KM_PER_DEGREE, _KM_PER_DEGREE]
    query = np.c_[towers.geometry.x * _COS_LAT, towers.geometry.y] * _KM_PER_DEGREE
    # way vertices are usually pylons, so distance to the nearest vertex is a sound filter
    distance, _ = cKDTree(xy).query(query)
    return towers[distance <= distance_km]


def _vertex(lon: float, lat: float) -> int:
    """Stable id of a vertex: the export has no OpenStreetMap node ids."""
    return round(lon * 1e6) * 10_000_000_000 + round(lat * 1e6)


def _with_vertices(way: dict[str, Any]) -> dict[str, Any]:
    return {**way, "nodes": [_vertex(q["lon"], q["lat"]) for q in way["geometry"]]}


def export_ways(
    lines: pd.DataFrame,
    download: Mapping[Any, dict[str, Any]] | None = None,
    abroad: Callable[[float, float], bool] | None = None,
) -> dict[int, dict[str, Any]]:
    """Export lines as ways for the route builder, ``{way id: way}``.

    The export ends at the German border, so ways of ``download`` it lacks are kept where
    ``abroad`` holds at their midpoint. Vertices are identified by position in both.
    """
    wkt = require("shapely.wkt", "grid")
    ways: dict[int, dict[str, Any]] = {}
    records: list[Any] = lines.to_dict("records")
    for row in records:
        tags = {"power": "line", "voltage": str(row["voltages"])}
        for key in ("name", "ref", "circuits", "cables", "location", "frequency", "operator"):
            if pd.notna(row.get(key)) and str(row.get(key)):
                tags[key] = str(row[key])
        if row["status"] == UNDER_CONSTRUCTION:
            tags["construction"] = "line"
        if row["status"] == OUT_OF_SERVICE:
            tags["disused:power"] = "line"
        line = wkt.loads(row["geometry_wkt"])
        ways[int(row["id"])] = _with_vertices(
            {
                "type": "way",
                "id": int(row["id"]),
                "geometry": [{"lon": float(c[0]), "lat": float(c[1])} for c in line.coords],
                "tags": tags,
            }
        )
    if download and abroad:
        for key, way in download.items():
            middle = way["geometry"][len(way["geometry"]) // 2]
            if int(key) not in ways and abroad(middle["lon"], middle["lat"]):
                ways[int(key)] = _with_vertices(way)
    return ways


def read_hv_layers(path: Path, tower_distance_km: float = 0.3) -> dict[str, pd.DataFrame]:
    """HV lines, pylons near them, HV substations, transformers and switches near them.

    The tables keep every published column and add ``geometry_wkt``, ``lon``, ``lat``; lines and
    substations also get ``status``, pylons ``line_number``.
    """
    all_lines = _read(path, "power_line")
    lines = _high_voltage(all_lines)
    towers = _near_lines(_read(path, "power_tower"), lines, tower_distance_km)
    points = _read(path, "power_substation_point")
    polygons = _read(path, "power_substation_polygon")
    every_transformer = _read(path, "power_transformer")
    primary = pd.to_numeric(every_transformer["v_primary"], errors="coerce")
    transformers = every_transformer[primary >= HV_VOLTS]
    substations = pd.concat(
        [
            _flat(_high_voltage(polygons)).assign(shape="polygon"),
            _flat(_high_voltage(points)).assign(shape="point"),
        ],
        ignore_index=True,
    )
    switches = _near_lines(_read(path, "power_switch"), lines, tower_distance_km)
    tower_table = _flat(towers)
    # the 110 kV yard of an EHV site is often its own polygon, below the HV filter above
    voltage = pd.to_numeric(polygons["max_voltage"], errors="coerce")
    infeed = polygons[
        (voltage >= INFEED_VOLTS)
        & ~polygons.reindex(columns=["frequency"])["frequency"].astype(str).eq("16.7")
    ]
    return {
        "lines": _flat(lines).pipe(lambda t: t.assign(status=status(t))),
        "towers": tower_table.assign(line_number=tower_table["ref"].map(tower_line_number)),
        "substations": substations.assign(status=status(substations)),
        "transformers": _flat(transformers),
        "switches": _flat(switches),
        "infeed_stations": _flat(infeed),
        "infeed_lines": _flat(
            all_lines[pd.to_numeric(all_lines["max_voltage"], errors="coerce") >= INFEED_VOLTS]
        ),
        "infeed_transformers": _flat(every_transformer[primary.isna() | (primary >= HV_VOLTS)]),
    }
