"""High-voltage layers of the Open Infrastructure Map export (OpenStreetMap data, ODbL).

The export is a GeoPackage snapshot of Germany. It adds what the grid model's own OpenStreetMap
download lacks: pylons, ways under construction, substation outlines, transformers and switches.
Needs the ``[grid]`` extra.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from tranpy_core._optional import require

HV_VOLTS = 200_000
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


def read_hv_layers(path: Path, tower_distance_km: float = 0.3) -> dict[str, pd.DataFrame]:
    """HV lines, pylons near them, HV substations, transformers and switches near them.

    The tables keep every published column and add ``geometry_wkt``, ``lon``, ``lat``; lines and
    substations also get ``status``, pylons ``line_number``.
    """
    lines = _high_voltage(_read(path, "power_line"))
    towers = _near_lines(_read(path, "power_tower"), lines, tower_distance_km)
    points = _read(path, "power_substation_point")
    polygons = _read(path, "power_substation_polygon")
    transformers = _read(path, "power_transformer")
    transformers = transformers[
        pd.to_numeric(transformers["v_primary"], errors="coerce") >= HV_VOLTS
    ]
    substations = pd.concat(
        [
            _flat(_high_voltage(polygons)).assign(shape="polygon"),
            _flat(_high_voltage(points)).assign(shape="point"),
        ],
        ignore_index=True,
    )
    switches = _near_lines(_read(path, "power_switch"), lines, tower_distance_km)
    tower_table = _flat(towers)
    return {
        "lines": _flat(lines).pipe(lambda t: t.assign(status=status(t))),
        "towers": tower_table.assign(line_number=tower_table["ref"].map(tower_line_number)),
        "substations": substations.assign(status=status(substations)),
        "transformers": _flat(transformers),
        "switches": _flat(switches),
    }
