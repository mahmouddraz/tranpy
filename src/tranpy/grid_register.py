"""Generation and load of the German grid model from the official registers.

Generation comes from the Marktstammdatenregister: units of the Bundesnetzagentur power
plant list are handled by :func:`tranpy.grid_layers.attach_plants`, every other unit is
aggregated into cells (:func:`aggregate_units`) and attached here. Load comes from the
measured load per control area (SMARD): the population shares only divide each area's load
among its substations.
"""

from __future__ import annotations

from importlib import resources
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from tranpy.grid_layers import _RENEWABLE, OFFSHORE_LANDINGS, _transmission_nodes

_CONTROL_AREA = {
    "50HERTZ": "50Hertz",
    "AMPRION GMBH": "Amprion",
    "TENNETGMBH": "TenneT",
    "TRANSNETBW": "TransnetBW",
}
_CELL_DEGREES = 0.02
_KM_PER_DEGREE = 111.2
_COS_LAT = np.cos(np.radians(51.0))
OFFSHORE_FUEL = "Windenergie (Offshore-Anlage)"


def offshore_connections() -> pd.DataFrame:
    """Curated connection point -> onshore substation, each entry with its evidence."""
    path = resources.files("tranpy.resources.grid").joinpath("offshore_connections_de.csv")
    with path.open(encoding="utf-8") as fh:
        return pd.read_csv(fh, dtype=str)


def connection_key(point: Any) -> str | None:
    """Key of a register connection point: the platform code of ``L-155-HOSE-BWGA-AC146``."""
    if not isinstance(point, str) or not point.strip():
        return None
    parts = point.split("-")
    return parts[3] if len(parts) >= 5 and parts[0] == "L" else point.strip()


def node_control_areas(nodes: pd.DataFrame) -> pd.Series:
    """Control area of each node: its first reporting German operator."""
    first = nodes["tso"].fillna("").astype(str).str.split(";").str[0]
    return pd.Series(first.map(_CONTROL_AREA).to_numpy(), index=nodes["node"].to_numpy())


def aggregate_units(units: pd.DataFrame) -> pd.DataFrame:
    """Register units in cells per fuel, control area, location method and connection point.

    ``units`` has ``fuel``, ``mw``, ``lon``, ``lat``, ``zone``, ``how``, ``connection``.
    Returns those columns with the cell centre as ``lon``, ``lat`` and ``units``.
    """
    frame = units.dropna(subset=["lon", "lat"]).copy()
    frame["lon"] = (frame["lon"] / _CELL_DEGREES).round() * _CELL_DEGREES
    frame["lat"] = (frame["lat"] / _CELL_DEGREES).round() * _CELL_DEGREES
    keys = ["lon", "lat", "fuel", "zone", "how", "connection"]
    return (
        frame.groupby(keys, dropna=False, observed=True)
        .agg(mw=("mw", "sum"), units=("mw", "size"))
        .reset_index()
    )


def _nearest(cells: pd.DataFrame, candidates: pd.DataFrame) -> tuple[Any, Any]:
    points = np.c_[candidates["lon"].astype(float) * _COS_LAT, candidates["lat"].astype(float)]
    distance, index = cKDTree(points).query(np.c_[cells["lon"] * _COS_LAT, cells["lat"]])
    return candidates["node"].to_numpy()[index], distance * _KM_PER_DEGREE


def attach_register(cells: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Connect register cells to substations; same columns as ``attach_plants``.

    Rules (column ``method``), all ``estimated`` except a sourced offshore connection:

    * offshore wind with a curated connection point: its onshore substation
      (``offshore connection``, published);
    * other offshore wind: equally over the landing substations of its sea (``offshore
      landing``);
    * any other unit: nearest substation of its control area, or of the whole grid when the
      register names no area (``nearest in control area``, ``nearest substation``).
    """
    candidates = _transmission_nodes(nodes)
    candidates = candidates.sort_values("kv", ascending=False).drop_duplicates("site").copy()
    candidates["zone"] = candidates["node"].map(node_control_areas(nodes))
    known = set(candidates["node"])
    target = dict(zip(*(offshore_connections()[c] for c in ("connection", "node")), strict=True))
    parts = []

    offshore = cells["fuel"] == OFFSHORE_FUEL
    sea = cells[offshore].copy()
    sea["node"] = sea["connection"].map(target)
    sea.loc[~sea["node"].isin(known), "node"] = None
    placed = sea[sea["node"].notna()]
    parts.append(placed.assign(method="offshore connection", distance_km=np.nan, share=1.0))
    rest = sea[sea["node"].isna()]
    for name, group in rest.groupby(np.where(rest["lon"] > 10.0, "Ostsee", "Nordsee")):
        landing = [n for n in OFFSHORE_LANDINGS[str(name)] if n in known]
        spread = group.drop(columns="node").merge(pd.Series(landing, name="node"), how="cross")
        parts.append(
            spread.assign(method="offshore landing", distance_km=np.nan, share=1 / len(landing))
        )

    land = cells[~offshore]
    for zone, group in land.groupby(land["zone"].fillna(""), sort=False):
        pool = candidates[candidates["zone"] == zone]
        method = "nearest in control area"
        if pool.empty:
            pool, method = candidates, "nearest substation"
        node, distance = _nearest(group, pool)
        parts.append(group.assign(node=node, distance_km=distance, method=method, share=1.0))

    out = pd.concat(parts, ignore_index=True)
    out["capacity_mw"] = out["mw"] * out["share"]
    out["unit_count"] = out["units"] * out["share"]
    out["weighted_km"] = out["distance_km"].fillna(0.0) * out["capacity_mw"]
    grouped = (
        out.groupby(["node", "fuel", "method"], observed=True)
        .agg(
            capacity_mw=("capacity_mw", "sum"),
            unit_count=("unit_count", "sum"),
            weighted_km=("weighted_km", "sum"),
        )
        .reset_index()
    )
    fuel = grouped["fuel"].str.lower()
    return pd.DataFrame(
        {
            "mastr_id": "register:" + grouped["node"] + ":" + grouped["fuel"],
            "name": "register units: " + grouped["fuel"],
            "fuel": grouped["fuel"],
            "state": None,
            "status": "In Betrieb",
            "in_operation": True,
            "reserve": False,
            "node": grouped["node"],
            "share": 1.0,
            "capacity_mw": grouped["capacity_mw"],
            "method": grouped["method"],
            "distance_km": grouped["weighted_km"] / grouped["capacity_mw"].where(lambda v: v > 0),
            "data_quality": np.where(
                grouped["method"] == "offshore connection", "published", "estimated"
            ),
            "renewable": [any(f in text for f in _RENEWABLE) for text in fuel],
            "unit_count": grouped["unit_count"],
        }
    )


def area_load_shares(
    shares: pd.DataFrame, nodes: pd.DataFrame, area_mw: dict[str, float]
) -> tuple[pd.DataFrame, float]:
    """Node shares from the measured load per control area, population within each area.

    Returns the table of :func:`tranpy.grid_layers.load_shares` with ``share`` replaced,
    and the total load in MW.
    """
    out = shares.copy()
    zone = out["node"].map(node_control_areas(nodes))
    within = out["population"] / out.groupby(zone)["population"].transform("sum")
    total = float(sum(area_mw.values()))
    out["share"] = within * zone.map(area_mw) / total
    missing = sorted(set(area_mw) - set(zone.dropna()))
    if missing or out["share"].isna().any():
        raise ValueError(f"no load assigned for control areas {missing or 'of some nodes'}")
    return out, total
