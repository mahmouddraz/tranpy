"""Tests for the ordered circuit bundles (synthetic graphs in metres)."""

from __future__ import annotations

import itertools

import pandas as pd
import pytest

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")

from tranpy.grid_bundles import (  # noqa: E402
    bundle_geometries,
    edge_slots,
    piece_slots,
    route_slots,
)

CRS = "EPSG:3035"
SP = 20.0


def _graph(edges: dict[str, tuple[str, str, list[tuple[float, float]]]]):
    rows = [{"edge_id": e, "u": u, "v": v} for e, (u, v, _) in edges.items()]
    geoms = [shapely.LineString(c) for _, _, c in edges.values()]
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=CRS)


def _routes(routes: dict[str, list[str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"mrid": m, "name": m, "voltage_kv": 380.0, "edges": e} for m, e in routes.items()]
    )


def _crossings(out) -> int:
    geo = dict(zip(out["mrid"], out.geometry, strict=True))
    return sum(int(geo[a].intersects(geo[b])) for a, b in itertools.combinations(geo, 2))


def _y_at(geom, x: float) -> float:
    cut = geom.intersection(shapely.LineString([(x, -1e5), (x, 1e5)]))
    return float(cut.y) if cut.geom_type == "Point" else float(cut.geoms[0].y)


def test_tee_two_circuits_split_without_crossing():
    g = _graph(
        {
            "e1": ("A", "B", [(0, 0), (1000, 0)]),
            "up": ("B", "C", [(1000, 0), (2000, 600)]),
            "dn": ("B", "D", [(1000, 0), (2000, -600)]),
        }
    )
    # names sort the other way round, so only the split decides
    out = bundle_geometries(g, _routes({"b": ["e1", "up"], "a": ["e1", "dn"]}), spacing_m=SP)
    geo = dict(zip(out["mrid"], out.geometry, strict=True))
    assert _y_at(geo["b"], 500) == pytest.approx(SP / 2)
    assert _y_at(geo["a"], 500) == pytest.approx(-SP / 2)
    assert _crossings(out) == 0
    assert out.set_index("mrid")["slot_changes"].to_dict() == {"a": 1, "b": 1}
    # each circuit is one unbroken line
    assert (out.geom_type == "LineString").all()


def test_three_circuits_one_leaves_mid_corridor():
    g = _graph(
        {
            "e1": ("A", "B", [(0, 0), (1000, 0)]),
            # reversed orientation in the middle of the corridor
            "e2": ("C", "B", [(2000, 0), (1000, 0)]),
            "e3": ("C", "D", [(2000, 0), (3000, 0)]),
            "off": ("C", "E", [(2000, 0), (2600, -800)]),
        }
    )
    routes = {"a": ["e1", "e2", "e3"], "c": ["e1", "e2", "off"], "b": ["e1", "e2", "e3"]}
    out = bundle_geometries(g, _routes(routes), spacing_m=SP)
    geo = dict(zip(out["mrid"], out.geometry, strict=True))
    assert _crossings(out) == 0
    # the leaving circuit is on the right (south) side, and keeps its slot on e1/e2
    assert _y_at(geo["c"], 500) == pytest.approx(-SP)
    assert _y_at(geo["c"], 1500) == pytest.approx(-SP)
    # slots change only at C, where the set changes
    for m in ("a", "b"):
        assert _y_at(geo[m], 500) == pytest.approx(_y_at(geo[m], 1500))
    changes = out.set_index("mrid")["slot_changes"]
    assert changes["a"] == changes["b"] == changes["c"] == 1
    slots = edge_slots(g, _routes(routes))
    assert set(slots.loc[slots["edge_id"] == "e3", "n"]) == {2}


def test_junction_with_three_legs():
    g = _graph(
        {
            "w": ("W", "J", [(-1000, 0), (0, 0)]),
            "ne": ("J", "N", [(0, 0), (700, 700)]),
            "se": ("J", "S", [(0, 0), (700, -700)]),
        }
    )
    routes = {"p": ["w", "ne"], "q": ["w", "se"], "r": ["ne", "se"]}
    out = bundle_geometries(g, _routes(routes), spacing_m=SP)
    assert len(out) == 3
    assert _crossings(out) == 0
    for geom in out.geometry:
        assert geom.is_simple


def test_identical_circuits_keep_order_through_orientation_flips():
    g = _graph(
        {
            "e1": ("A", "B", [(0, 0), (1000, 0)]),
            "e2": ("C", "B", [(2000, 0), (1000, 0)]),
            "e3": ("C", "D", [(2000, 0), (3000, 0)]),
        }
    )
    out = bundle_geometries(g, _routes({"x": ["e1", "e2", "e3"], "y": ["e3", "e2", "e1"]}), SP)
    assert (out["slot_changes"] == 0).all()
    assert _crossings(out) == 0


def test_parallel_ways_form_one_bundle():
    # two tower lines 40 m apart, two circuits each: one bundle of four, no overlap
    g = _graph(
        {
            "n": ("A", "B", [(0, 40), (2000, 40)]),
            "s": ("C", "D", [(2000, 0), (0, 0)]),
        }
    )
    out = bundle_geometries(
        g, _routes({"n1": ["n"], "n2": ["n"], "s1": ["s"], "s2": ["s"]}), spacing_m=SP
    )
    ys = sorted(_y_at(geom, 1000) for geom in out.geometry)
    assert ys == pytest.approx([20 - 1.5 * SP, 20 - 0.5 * SP, 20 + 0.5 * SP, 20 + 1.5 * SP])
    geo = dict(zip(out["mrid"], out.geometry, strict=True))
    assert min(_y_at(geo["n1"], 1000), _y_at(geo["n2"], 1000)) > max(
        _y_at(geo["s1"], 1000), _y_at(geo["s2"], 1000)
    )
    off = bundle_geometries(
        g, _routes({"n1": ["n"], "n2": ["n"], "s1": ["s"], "s2": ["s"]}), SP, parallel_m=0
    )
    assert _crossings(off) == 0


def test_short_detour_is_transparent():
    # b takes a 40 m parallel span between two shared edges; order and offset stay
    g = _graph(
        {
            "e1": ("A", "B", [(0, 0), (1000, 0)]),
            "s1": ("B", "C", [(1000, 0), (1040, 0)]),
            "s2": ("B", "C", [(1000, 0), (1020, 5), (1040, 0)]),
            "e2": ("C", "D", [(1040, 0), (2000, 0)]),
            "up": ("D", "E", [(2000, 0), (2600, 800)]),
            "dn": ("D", "F", [(2000, 0), (2600, -800)]),
        }
    )
    routes = {"a": ["e1", "s1", "e2", "up"], "b": ["e1", "s2", "e2", "dn"]}
    out = bundle_geometries(g, _routes(routes), spacing_m=SP).set_index("mrid")
    assert _y_at(out.geometry["a"], 500) == pytest.approx(SP / 2)
    assert _y_at(out.geometry["b"], 500) == pytest.approx(-SP / 2)
    assert out["slot_changes"].to_dict() == {"a": 1, "b": 1}


def test_station_end_ordered_by_outline_point_with_connector():
    # both circuits end 100 m short of a station; "b" (names say: left) meets its outline
    # south, "a" north, so "a" goes left and both run on as a bundle into the box edge
    g = _graph({"e1": ("A", "B", [(0, 0), (1000, 0)])})
    routes = _routes({"b": ["e1"], "a": ["e1"]})
    clipped = {
        "a": shapely.LineString([(0, 0), (1000, 0), (1100, 30)]),
        "b": shapely.LineString([(0, 0), (1000, 0), (1100, -30)]),
    }
    routes["geometry"] = [clipped[m] for m in routes["mrid"]]
    routes = gpd.GeoDataFrame(routes, geometry="geometry", crs=CRS)
    routes["entry1_kind"] = "junction"
    routes["entry1_gap_m"] = 0.0
    routes["entry2_kind"] = "outline_near"
    routes["entry2_gap_m"] = [clipped[m].length - 1000 for m in routes["mrid"]]
    out = bundle_geometries(g, routes, spacing_m=SP).set_index("mrid")
    assert _y_at(out.geometry["a"], 500) == pytest.approx(SP / 2)
    assert _y_at(out.geometry["b"], 500) == pytest.approx(-SP / 2)
    assert out.geometry["a"].coords[-1][1] > out.geometry["b"].coords[-1][1]
    assert out["connector2_m"].round().tolist() == [104.0, 104.0]
    # the bundle fans in onto each circuit's own entry point
    assert out.geometry["a"].coords[-1] == pytest.approx((1100, 30))
    assert out.geometry["b"].coords[-1] == pytest.approx((1100, -30))
    assert _crossings(out.reset_index()) == 0


def test_station_fan_is_crossing_free():
    # three legs into a 200 m box at the origin; the west leg carries two circuits that
    # weave on short approach ways inside the zone; all four end on the box outline
    box = shapely.box(-100, -100, 100, 100)
    g = _graph(
        {
            "w": ("W", "J", [(-3000, 0), (-400, 0)]),
            "wa": ("J", "S", [(-400, 0), (-200, 300), (0, -50)]),
            "wb": ("J", "S", [(-400, 0), (-200, -300), (0, 50)]),
            "n": ("N", "S", [(0, 3000), (0, 0)]),
            "e": ("E", "S", [(3000, 200), (0, 0)]),
        }
    )
    routes = _routes({"a": ["w", "wa"], "b": ["w", "wb"], "c": ["n"], "d": ["e"]})
    routes["bus1"] = ["far|380"] * 4
    routes["bus2"] = ["st|380"] * 4
    stations = gpd.GeoDataFrame({"site": ["st"]}, geometry=[box], crs=CRS)
    out = bundle_geometries(g, routes, spacing_m=SP, stations=stations)
    assert _crossings(out) == 0
    assert (out["fan2"] == "st").all()
    for geom in out.geometry:
        assert box.exterior.distance(shapely.Point(geom.coords[-1])) < 1e-6
    ends = [shapely.Point(geom.coords[-1]) for geom in out.geometry]
    assert min(p.distance(q) for p, q in itertools.combinations(ends, 2)) >= 25 - 1e-6


def _left(rs, m, e):
    """Travel-left position (slot units) of circuit ``m`` on edge ``e``."""
    r = rs[(rs["mrid"] == m) & (rs["edge_id"] == e)].iloc[0]
    v = (r["n"] - 1) / 2 - r["slot"]
    return v if r["forward"] else -v


def test_route_slots_tee():
    g = _graph(
        {
            "e1": ("A", "B", [(0, 0), (1000, 0)]),
            "up": ("B", "C", [(1000, 0), (2000, 600)]),
            "dn": ("B", "D", [(1000, 0), (2000, -600)]),
        }
    )
    rs = route_slots(g, _routes({"b": ["e1", "up"], "a": ["e1", "dn"]}))
    assert _left(rs, "b", "e1") > _left(rs, "a", "e1")
    assert set(rs.loc[rs["edge_id"] == "e1", "n"]) == {2}
    assert set(rs.loc[rs["edge_id"] != "e1", "n"]) == {1}


def test_route_slots_one_leaves_mid_corridor_and_orientation():
    g = _graph(
        {
            "e1": ("A", "B", [(0, 0), (1000, 0)]),
            "e2": ("C", "B", [(2000, 0), (1000, 0)]),
            "e3": ("C", "D", [(2000, 0), (3000, 0)]),
            "off": ("C", "E", [(2000, 0), (2600, -800)]),
        }
    )
    routes = {"a": ["e1", "e2", "e3"], "c": ["e1", "e2", "off"], "b": ["e1", "e2", "e3"]}
    rs = route_slots(g, _routes(routes))
    for m in "abc":
        assert _left(rs, m, "e1") == _left(rs, m, "e2")  # same corridor, same place
    assert _left(rs, "c", "e1") == -1  # rightmost, it turns right
    assert _left(rs, "a", "e3") != _left(rs, "a", "e2")  # changes only where c leaves


def test_route_slots_junction_three_legs():
    g = _graph(
        {
            "w": ("W", "J", [(-1000, 0), (0, 0)]),
            "ne": ("J", "N", [(0, 0), (700, 700)]),
            "se": ("J", "S", [(0, 0), (700, -700)]),
        }
    )
    rs = route_slots(g, _routes({"p": ["w", "ne"], "q": ["w", "se"], "r": ["ne", "se"]}))
    assert _left(rs, "p", "w") > _left(rs, "q", "w")  # p turns left
    assert sorted(rs.groupby("edge_id")["n"].first()) == [2, 2, 2]
    assert (rs.groupby("edge_id")["slot"].nunique() == 2).all()


def test_route_slots_short_piece_inherits():
    g = _graph(
        {
            "e1": ("A", "B", [(0, 0), (1000, 0)]),
            "s1": ("B", "C", [(1000, 0), (1040, 0)]),
            "s2": ("B", "C", [(1000, 0), (1020, 5), (1040, 0)]),
            "e2": ("C", "D", [(1040, 0), (2000, 0)]),
        }
    )
    rs = route_slots(g, _routes({"a": ["e1", "s1", "e2"], "b": ["e1", "s2", "e2"]}))
    for m in "ab":
        assert _left(rs, m, "e1") == _left(rs, m, "e2") == _left(rs, m, "s1" if m == "a" else "s2")


def test_piece_slots_follow_piece_direction():
    # pieces drawn in either direction get the slot that puts "up" on the north side
    g = _graph(
        {
            "e1": ("A", "B", [(0, 0), (1000, 0)]),
            "up": ("B", "C", [(1000, 0), (2000, 600)]),
            "dn": ("B", "D", [(1000, 0), (2000, -600)]),
        }
    )
    routes = _routes({"b": ["e1", "up"], "a": ["e1", "dn"]})
    rows = [
        ("b", 0, "e1", [(0, 0), (1000, 0)]),
        ("b", 1, "up", [(1000, 0), (2000, 600)]),
        ("a", 0, "e1", [(1000, 0), (0, 0)]),  # drawn against travel
        ("a", 1, "dn", [(1000, 0), (2000, -600)]),
        ("a", 2, -1, [(2000, -600), (2020, -610)]),  # connector to a box
    ]
    pieces = gpd.GeoDataFrame(
        [{"mrid": m, "seq": q, "edge_id": e} for m, q, e, _ in rows],
        geometry=[shapely.LineString(c) for *_, c in rows],
        crs=CRS,
    )
    out = piece_slots(g, routes, pieces).set_index(["mrid", "seq"])
    # MapLibre offset (slot - (n-1)/2), positive = right of the drawn direction
    off_b = out.at[("b", 0), "slot"] - (out.at[("b", 0), "n"] - 1) / 2
    off_a = out.at[("a", 0), "slot"] - (out.at[("a", 0), "n"] - 1) / 2
    assert off_b == -0.5  # b drawn eastward, north = left
    assert off_a == -0.5  # a drawn westward, south = left
    assert out.at[("a", 2), "n"] == out.at[("a", 1), "n"] == 1
