"""Fill the CGMES tables from silver: one function per group of classes, no invented values.

Each result maps a class name to rows with columns of that class; :func:`conform` checks
them against the schema and adds the unused columns. Identifiers reuse
:func:`tranpy.grid_layers.mrid`, so an object keeps its mRID across builds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tranpy.cim.schema import Table, tables
from tranpy.grid_layers import mrid
from tranpy_core.errors import DataError

Rows = dict[str, pd.DataFrame]


def _ids(prefix: str, values: Any) -> list[str]:
    return [mrid(prefix, v) for v in values]


def _control_area(nodes: pd.DataFrame) -> pd.Series:
    """A node's region: the first reporting operator in Germany, else its country."""
    tso = nodes["tso"].fillna("").astype(str).str.split(";").str[0]
    return tso.where(nodes["country"] == "DEU", nodes["country"])


def network(nodes: pd.DataFrame, circuits: pd.DataFrame, transformers: pd.DataFrame) -> Rows:
    """Regions, stations, nodes, lines, transformers and their terminals and limits."""
    nodes = nodes.assign(region=_control_area(nodes))
    station = nodes[nodes["kind"] != "tee"]
    tee = nodes[nodes["kind"] == "tee"]
    circuits = circuits[~circuits["hvdc"]]

    countries = sorted(nodes["country"].dropna().unique())
    regions = nodes[["region", "country"]].drop_duplicates("region")
    rows: Rows = {
        "GeographicalRegion": pd.DataFrame({"mRID": _ids("geo", countries), "name": countries}),
        "SubGeographicalRegion": pd.DataFrame(
            {
                "mRID": _ids("subgeo", regions["region"]),
                "name": regions["region"],
                "Region": _ids("geo", regions["country"]),
            }
        ),
    }
    levels = sorted(
        {float(v) for v in nodes["kv"]}
        | {float(v) for v in transformers["kv_hv"]}
        | {float(v) for v in transformers["kv_lv"]}
    )
    rows["BaseVoltage"] = pd.DataFrame(
        {
            "mRID": _ids("basevoltage", levels),
            "name": [f"{v:g} kV" for v in levels],
            "nominalVoltage": levels,
        }
    )
    sites = station.drop_duplicates("site")
    rows["Substation"] = pd.DataFrame(
        {
            "mRID": _ids("substation", sites["site"]),
            "name": sites["site"],
            "Region": _ids("subgeo", sites["region"]),
        }
    )
    level_id = [mrid("vl", s, k) for s, k in zip(station["site"], station["kv"], strict=True)]
    rows["VoltageLevel"] = pd.DataFrame(
        {
            "mRID": level_id,
            "name": [f"{s} {k} kV" for s, k in zip(station["site"], station["kv"], strict=True)],
            "Substation": _ids("substation", station["site"]),
            "BaseVoltage": _ids("basevoltage", station["kv"].astype(float)),
        }
    )
    rows["BusbarSection"] = pd.DataFrame(
        {
            "mRID": _ids("bus", station["node"]),
            "name": station["node"],
            "EquipmentContainer": level_id,
            "BaseVoltage": _ids("basevoltage", station["kv"].astype(float)),
            "inService": True,
        }
    )
    # a tee is a connection point on a line: its node lies in a line container of its own
    tee_line = _ids("teeline", tee["node"])
    lines = pd.DataFrame(
        {
            "mRID": [*_ids("line", circuits["name"].drop_duplicates()), *tee_line],
            "name": [*circuits["name"].drop_duplicates(), *tee["node"]],
        }
    )
    rows["Line"] = lines
    container = dict(zip(station["node"], level_id, strict=True))
    container.update(zip(tee["node"], tee_line, strict=True))
    rows["ConnectivityNode"] = pd.DataFrame(
        {
            "mRID": _ids("cn", nodes["node"]),
            "name": nodes["node"],
            "ConnectivityNodeContainer": nodes["node"].map(container),
        }
    )
    rows["TopologicalNode"] = pd.DataFrame(
        {
            "mRID": _ids("tn", nodes["node"]),
            "name": nodes["node"],
            "BaseVoltage": _ids("basevoltage", nodes["kv"].astype(float)),
            "ConnectivityNodeContainer": nodes["node"].map(container),
        }
    )

    keys = [
        f"{e}~{a}~{b}"
        for e, a, b in zip(circuits["eic"], circuits["node0"], circuits["node1"], strict=True)
    ]
    segment = [
        mrid("line", e, a, b)
        for e, a, b in zip(circuits["eic"], circuits["node0"], circuits["node1"], strict=True)
    ]
    in_service = circuits["status"].fillna("in_operation") == "in_operation"
    rows["ACLineSegment"] = pd.DataFrame(
        {
            "mRID": segment,
            "name": circuits["name"],
            "EquipmentContainer": _ids("line", circuits["name"]),
            "BaseVoltage": _ids("basevoltage", circuits["kv"].astype(float)),
            "length": pd.to_numeric(circuits["length_km"], errors="coerce"),
            "r": circuits["r_ohm"].astype(float),
            "x": circuits["x_ohm"].astype(float),
            "bch": circuits["b_us"].astype(float).fillna(0.0) * 1e-6,
            "gch": 0.0,
            "inService": in_service.to_numpy(),
            "normallyInService": in_service.to_numpy(),
        }
    )

    tr_id = [
        mrid("trafo", e, n) for e, n in zip(transformers["eic"], transformers["name"], strict=True)
    ]
    site_of = dict(zip(nodes["node"], nodes["site"], strict=True))
    rows["PowerTransformer"] = pd.DataFrame(
        {
            "mRID": tr_id,
            "name": transformers["name"],
            "EquipmentContainer": _ids("substation", transformers["node_hv"].map(site_of)),
            "inService": True,
        }
    )
    ends = []
    for number, _node, kv in ((1, "node_hv", "kv_hv"), (2, "node_lv", "kv_lv")):
        high = number == 1  # the source gives the impedance on the high voltage side
        ends.append(
            pd.DataFrame(
                {
                    "mRID": [mrid("trend", t, number) for t in tr_id],
                    "name": [f"{n} end {number}" for n in transformers["name"]],
                    "PowerTransformer": tr_id,
                    "endNumber": number,
                    "BaseVoltage": _ids("basevoltage", transformers[kv].astype(float)),
                    "ratedU": transformers[kv].astype(float).to_numpy(),
                    "Terminal": [mrid("terminal", t, number) for t in tr_id],
                    "r": transformers["r_ohm"].astype(float).to_numpy() if high else 0.0,
                    "x": transformers["x_ohm"].astype(float).to_numpy() if high else 0.0,
                    "b": transformers["b_us"].astype(float).fillna(0.0).to_numpy() * 1e-6
                    if high
                    else 0.0,
                    "g": transformers["g_us"].astype(float).fillna(0.0).to_numpy() * 1e-6
                    if high
                    else 0.0,
                }
            )
        )
    rows["PowerTransformerEnd"] = pd.concat(ends, ignore_index=True)

    terminals = []
    for number in (1, 2):
        for ids, frame, col in (
            (segment, circuits, f"node{number - 1}"),
            (tr_id, transformers, "node_hv" if number == 1 else "node_lv"),
        ):
            terminals.append(
                pd.DataFrame(
                    {
                        "mRID": [mrid("terminal", i, number) for i in ids],
                        "name": [f"{n} T{number}" for n in frame["name"]],
                        "ConductingEquipment": ids,
                        "sequenceNumber": number,
                        "connected": True,
                        "ConnectivityNode": _ids("cn", frame[col]),
                        "TopologicalNode": _ids("tn", frame[col]),
                    }
                )
            )
    terminals.append(
        pd.DataFrame(
            {
                "mRID": [mrid("terminal", i, 1) for i in rows["BusbarSection"]["mRID"]],
                "name": [f"{n} T1" for n in station["node"]],
                "ConductingEquipment": rows["BusbarSection"]["mRID"].to_numpy(),
                "sequenceNumber": 1,
                "connected": True,
                "ConnectivityNode": _ids("cn", station["node"]),
                "TopologicalNode": _ids("tn", station["node"]),
            }
        )
    )
    rows["Terminal"] = pd.concat(terminals, ignore_index=True)

    # limits: the published rating of a line, on its first terminal
    rated = circuits.assign(segment=segment, key=keys)
    rated = rated[pd.to_numeric(rated["imax_a"], errors="coerce") > 0]
    limit_type = mrid("limittype", "patl")
    rows["OperationalLimitType"] = pd.DataFrame(
        {
            "mRID": [limit_type],
            "name": ["PATL"],
            "direction": ["absoluteValue"],
            "isInfiniteDuration": [True],
            "kind": ["patl"],
        }
    )
    limit_set = [mrid("limitset", s) for s in rated["segment"]]
    rows["OperationalLimitSet"] = pd.DataFrame(
        {
            "mRID": limit_set,
            "name": rated["name"],
            "Terminal": [mrid("terminal", s, 1) for s in rated["segment"]],
            "Equipment": rated["segment"],
        }
    )
    rows["CurrentLimit"] = pd.DataFrame(
        {
            "mRID": [mrid("currentlimit", s) for s in rated["segment"]],
            "name": rated["name"],
            "OperationalLimitSet": limit_set,
            "OperationalLimitType": limit_type,
            "normalValue": pd.to_numeric(rated["imax_a"]).astype(float).to_numpy(),
            "value": pd.to_numeric(rated["imax_a"]).astype(float).to_numpy(),
        }
    )
    return rows


def _points(route: Any) -> list[tuple[float, float]]:
    if route is None:
        return []
    points = [p.asDict() if hasattr(p, "asDict") else p for p in route]
    return [
        (float(p["lon"]), float(p["lat"])) if isinstance(p, dict) else (float(p[0]), float(p[1]))
        for p in points
    ]


def geometry(nodes: pd.DataFrame, circuits: pd.DataFrame) -> Rows:
    """WGS84 locations of stations and the routes of lines (schematic routes included)."""
    system = mrid("crs", "wgs84")
    rows: Rows = {
        "CoordinateSystem": pd.DataFrame(
            {"mRID": [system], "name": ["WGS84"], "crsUrn": ["urn:ogc:def:crs:OGC:1.3:CRS84"]}
        )
    }
    locations, points = [], []

    def add(resource: str, name: str, coords: list[tuple[float, float]]) -> None:
        location = mrid("location", resource)
        locations.append(
            {
                "mRID": location,
                "name": name,
                "CoordinateSystem": system,
                "PowerSystemResources": resource,
            }
        )
        for k, (lon, lat) in enumerate(coords, start=1):
            points.append(
                {
                    "Location": location,
                    "sequenceNumber": k,
                    "xPosition": repr(lon),
                    "yPosition": repr(lat),
                }
            )

    for site, group in nodes[
        (nodes["kind"] != "tee") & np.isfinite(nodes["lon"].astype(float))
    ].groupby("site"):
        first = group.iloc[0]
        add(mrid("substation", site), str(site), [(float(first["lon"]), float(first["lat"]))])
    for c in circuits[~circuits["hvdc"]].to_dict("records"):
        coords = _points(c["route"])
        if len(coords) >= 2:
            add(mrid("line", c["eic"], c["node0"], c["node1"]), str(c["name"]), coords)
    rows["Location"] = pd.DataFrame(locations)
    rows["PositionPoint"] = pd.DataFrame(points)
    return rows


def _mandatory_in_equipment(table: Table) -> list[str]:
    return [
        c.name
        for c in table.columns
        if c.multiplicity.startswith("1") and "Equipment" in c.profiles
    ]


def conform(rows: Rows) -> Rows:
    """Rows with exactly the columns of their table; raises on an unknown class or column,
    or on a value missing in a column the Equipment profile requires."""
    schema = {t.name: t for t in tables()}
    out: Rows = {}
    problems = []
    for name, frame in rows.items():
        table = schema.get(name)
        if table is None:
            raise DataError(f"{name} is not a CGMES class")
        columns = [c.name for c in table.columns]
        unknown = set(frame.columns) - set(columns)
        if unknown:
            raise DataError(f"{name}: not in the schema: {sorted(unknown)}")
        for column in _mandatory_in_equipment(table):
            if column not in frame or frame[column].isna().any():
                problems.append(f"{name}.{column}")
        out[name] = frame.reindex(columns=columns).reset_index(drop=True)
    if problems:
        raise DataError("mandatory values missing: " + ", ".join(problems))
    return out
