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
_AREA_RADIUS_KM = 25.0
_SHARED_FUELS = ("Solare Strahlungsenergie", "Windenergie (Onshore-Anlage)")
_SHARED_NODES = 4
_SHARED_RADIUS_KM = 40.0
_SHARED_MIN_KM = 5.0
_CAP_ROUNDS = 8


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


def node_control_areas(nodes: pd.DataFrame, cells: pd.DataFrame | None = None) -> pd.Series:
    """Control area of each node.

    A node of several operators (a border between areas) belongs to the listed area holding the
    most onshore register capacity within :data:`_AREA_RADIUS_KM`, so the operators' sort order
    does not decide it. Without ``cells`` or nearby capacity it keeps its first listed operator.
    """
    operators = nodes["tso"].fillna("").astype(str).str.split(";")
    areas = pd.Series(
        operators.str[0].map(_CONTROL_AREA).to_numpy(), index=nodes["node"].to_numpy()
    )
    joint = nodes[operators.str.len() > 1]
    if cells is None or joint.empty:
        return areas
    near = cells[cells["fuel"].ne(OFFSHORE_FUEL) & cells["zone"].notna()]
    near_xy = np.c_[near["lon"].astype(float) * _COS_LAT, near["lat"].astype(float)]
    tree = cKDTree(near_xy)
    mw = near["mw"].to_numpy(dtype=float)
    zones = near["zone"].to_numpy()
    for row in joint.itertuples():
        if not np.isfinite(row.lon) or not np.isfinite(row.lat):
            continue
        hits = tree.query_ball_point(
            [float(row.lon) * _COS_LAT, float(row.lat)], _AREA_RADIUS_KM / _KM_PER_DEGREE
        )
        listed = {_CONTROL_AREA.get(name) for name in str(row.tso).split(";")}
        totals = pd.Series(mw[hits]).groupby(zones[hits]).sum() if hits else pd.Series(dtype=float)
        totals = totals[totals.index.isin(listed)]
        if len(totals):
            areas[row.node] = totals.idxmax()
    return areas


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


def connected_nodes(
    nodes: pd.DataFrame, lines: pd.DataFrame, transformers: pd.DataFrame
) -> pd.DataFrame:
    """Nodes of the main network: operating AC circuits and transformers join them."""
    active = lines[lines["status"].fillna("in_operation").eq("in_operation")]
    if "hvdc" in active:
        active = active[~active["hvdc"].fillna(False).astype(bool)]
    parent = {str(n): str(n) for n in nodes["node"]}

    def root(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    pairs = list(zip(active["node0"], active["node1"], strict=True))
    pairs += list(zip(transformers["node_hv"], transformers["node_lv"], strict=True))
    for a, b in pairs:
        if a in parent and b in parent:
            parent[root(a)] = root(b)
    groups = pd.Series({n: root(n) for n in parent})
    return nodes[groups.reindex(nodes["node"]).eq(groups.value_counts().idxmax()).to_numpy()]


def connection_mva(lines: pd.DataFrame) -> pd.Series:
    """Rated MVA of the operating circuits at each node."""
    active = lines[lines["status"].fillna("in_operation").eq("in_operation")]
    mva = np.sqrt(3) * active["kv"].astype(float) * active["imax_a"].astype(float) / 1000.0
    ends = pd.concat(
        [
            pd.DataFrame({"node": active["node0"], "mva": mva}),
            pd.DataFrame({"node": active["node1"], "mva": mva}),
        ]
    )
    return ends.groupby("node")["mva"].sum()


def _shared(cells: pd.DataFrame, pool: pd.DataFrame, limit: pd.Series | None) -> pd.DataFrame:
    """One row per cell and nearby node, shares by inverse distance within the node limits.

    Capacity above a node's limit moves to the cell's other nearby nodes; what fits nowhere
    stays, so no capacity is lost.
    """
    points = np.c_[pool["lon"].astype(float) * _COS_LAT, pool["lat"].astype(float)]
    k = min(_SHARED_NODES, len(pool))
    distance, index = cKDTree(points).query(np.c_[cells["lon"] * _COS_LAT, cells["lat"]], k=k)
    distance = distance.reshape(len(cells), k) * _KM_PER_DEGREE
    index = index.reshape(len(cells), k)
    near = distance <= _SHARED_RADIUS_KM
    near[:, 0] = True
    weight = near / np.maximum(distance, _SHARED_MIN_KM)
    share = weight / weight.sum(axis=1, keepdims=True)
    names = pool["node"].to_numpy()
    mw = cells["mw"].to_numpy(dtype=float)
    if limit is not None:
        caps = limit.reindex(names).fillna(np.inf).to_numpy(dtype=float)
        for _ in range(_CAP_ROUNDS):
            load = np.bincount(index.ravel(), (share * mw[:, None]).ravel(), len(names))
            factor = np.minimum(1.0, caps / np.maximum(load, 1e-9))
            kept = share * factor[index]
            spare = (factor[index] >= 1.0) & near
            excess = share.sum(axis=1) - kept.sum(axis=1)
            room = (weight * spare).sum(axis=1)
            moved = np.divide(
                weight * spare, room[:, None], out=np.zeros_like(share), where=room[:, None] > 0
            )
            share = kept + moved * excess[:, None]
            # cells without a free node keep the rest where they are
            stuck = room <= 0
            share[stuck] += (kept[stuck] / kept[stuck].sum(axis=1, keepdims=True)) * excess[
                stuck, None
            ]
    rows = cells.loc[cells.index.repeat(k)].reset_index(drop=True)
    return rows.assign(
        node=names[index.ravel()], distance_km=distance.ravel(), share=share.ravel()
    ).query("share > 0")


def apply_plant_connections(attachments: pd.DataFrame) -> pd.DataFrame:
    """Move plants to the substation their published connection names."""
    path = resources.files("tranpy.resources.grid").joinpath("plant_connections_de.csv")
    with path.open(encoding="utf-8") as fh:
        table = pd.read_csv(fh, dtype=str)
    out = attachments.copy()
    for row in table.itertuples():
        chosen = out["name"].astype(str).eq(row.plant)
        out.loc[chosen, ["node", "method", "data_quality"]] = [
            row.node,
            "published connection",
            "published",
        ]
        out.loc[chosen, "distance_km"] = np.nan
    return out


def attach_register(
    cells: pd.DataFrame,
    nodes: pd.DataFrame,
    limit_mva: pd.Series | None = None,
) -> pd.DataFrame:
    """Connect register cells to substations; same columns as ``attach_plants``.

    Rules (column ``method``), all ``estimated`` except a sourced offshore connection:

    * offshore wind with a curated connection point: its onshore substation
      (``offshore connection``, published);
    * other offshore wind: equally over the landing substations of its sea (``offshore
      landing``);
    * solar and onshore wind, connected in the distribution grids the model does not hold:
      shared over nearby substations of the area by inverse distance, within ``limit_mva`` per
      node (``shared in control area``);
    * any other unit: nearest substation of its control area, or of the whole grid when the
      register names no area (``nearest in control area``, ``nearest substation``).
    """
    candidates = _transmission_nodes(nodes)
    candidates = candidates.sort_values("kv", ascending=False).drop_duplicates("site").copy()
    candidates["zone"] = candidates["node"].map(node_control_areas(nodes, cells))
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
        spread = group["fuel"].isin(_SHARED_FUELS)
        if spread.any():
            label = "shared in control area" if method != "nearest substation" else method
            parts.append(_shared(group[spread], pool, limit_mva).assign(method=label))
        if (~spread).any():
            node, distance = _nearest(group[~spread], pool)
            parts.append(
                group[~spread].assign(node=node, distance_km=distance, method=method, share=1.0)
            )

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
    shares: pd.DataFrame,
    nodes: pd.DataFrame,
    area_mw: dict[str, float],
    cells: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, float]:
    """Node shares from the measured load per control area, population within each area.

    Returns the table of :func:`tranpy.grid_layers.load_shares` with ``share`` replaced,
    and the total load in MW.
    """
    out = shares.copy()
    zone = out["node"].map(node_control_areas(nodes, cells))
    within = out["population"] / out.groupby(zone)["population"].transform("sum")
    total = float(sum(area_mw.values()))
    out["share"] = within * zone.map(area_mw) / total
    missing = sorted(set(area_mw) - set(zone.dropna()))
    if missing or out["share"].isna().any():
        raise ValueError(f"no load assigned for control areas {missing or 'of some nodes'}")
    return out, total


def rating_suspect(lines: pd.DataFrame, share: float = 0.5, km: float = 1.0) -> pd.Series:
    """Tee stubs under ``share`` of the highest rating at their tee; values stay published."""
    tee = (
        lines["node0"].astype(str).str.startswith("y-"),
        lines["node1"].astype(str).str.startswith("y-"),
    )
    node = lines["node0"].where(tee[0], lines["node1"].where(tee[1]))
    other = lines.groupby(node)["imax_a"].transform("max")
    return (
        (lines["length_km"].astype(float) < km) & node.notna() & (lines["imax_a"] < share * other)
    )
