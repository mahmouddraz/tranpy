"""Tests for the OSM network graph, circuit routing and edge check (synthetic graph)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shapely")

from tranpy.grid_network import (
    Circuit,
    apply_overrides,
    build_graph,
    circuit_code,
    circuit_number,
    classify,
    clean_findings,
    detour_overrides,
    edge_check,
    fallback_box,
    findings,
    fold,
    route_circuits,
    self_check,
)

D = 0.01  # degrees, about 0.7-1.1 km


def way(i, coords, kv=380, circuits=2.0, ref="", name=""):
    return {
        "id": i,
        "coords": coords,
        "kvs": {kv},
        "circuits_tag": circuits,
        "circuits_max": circuits,
        "ref": ref,
        "name": name,
    }


def terminal(bus, lon, lat, kind="substation", kv=380):
    return {
        "bus": bus,
        "kind": kind,
        "station": bus.split("|")[0],
        "kv": kv,
        "lon": lon,
        "lat": lat,
    }


@pytest.fixture
def graph():
    """A--T tee to B and C; A--D on two parallel routes (ref 501 north); C has a 30 m gap."""
    ways = [
        way(1, [(10.0, 51.0), (10.0 + D, 51.0), (10.0 + 2 * D, 51.0)]),  # A -> T
        way(2, [(10.0 + 2 * D, 51.0), (10.0 + 3 * D, 51.0)]),  # T -> B
        way(3, [(10.0 + 2 * D, 51.0), (10.0 + 2 * D, 51.0 - D)]),  # T -> gap
        way(4, [(10.0 + 2 * D + 0.0004, 51.0 - D), (10.0 + 2 * D, 51.0 - 2 * D)]),  # gap -> C
        # parallel routes A -> D: south (shorter, no ref) and north (ref 501)
        way(5, [(10.0, 51.0), (10.0 - D, 51.0 - 0.3 * D), (10.0 - 2 * D, 51.0)], ref=""),
        way(6, [(10.0, 51.0), (10.0 - D, 51.0 + 0.5 * D), (10.0 - 2 * D, 51.0)], ref="501"),
        # an island 170 m beyond C's way end: not bridged
        way(7, [(10.0 + 2 * D + 0.0025, 51.0 - 2 * D), (10.0 + 6 * D, 51.0 - 2 * D)]),
    ]
    terms = pd.DataFrame(
        [
            terminal("a|380", 10.0, 51.0),
            terminal("b|380", 10.0 + 3 * D, 51.0),
            terminal("c|380", 10.0 + 2 * D, 51.0 - 2 * D),
            terminal("d|380", 10.0 - 2 * D, 51.0),
            terminal("e|380", 10.0 + 6 * D, 51.0 - 2 * D),
            terminal("t#1|380", 10.0 + 2 * D + 0.0001, 51.0, kind="tee"),
        ]
    )
    return build_graph(ways, terms, footprints={})


def test_split_and_gap(graph):
    e = graph.edges
    assert (e["kind"] == "gap").sum() == 1
    gap = e[e["kind"] == "gap"].iloc[0]
    assert gap["virtual"] and gap["length_m"] < 50
    # way 1 split at the tee node shared with ways 2 and 3? way 1 ends there: 1 edge
    assert (e["way_id"] == 1).sum() == 1


def test_tee_routing_and_gap(graph):
    cs = [
        Circuit("m1", "A - B 400", 380, "a|380", "b|380", 3.5),
        Circuit("m2", "A - T 401", 380, "a|380", "t#1|380", 1.4),
        Circuit("m3", "T - C 402", 380, "t#1|380", "c|380", 2.2),
        Circuit("m4", "A - E 403", 380, "a|380", "e|380", 5.0),
        Circuit("m5", "A - X 404", 220, "a|380", "b|380", 1.0),
    ]
    r = route_circuits(graph, cs).set_index("mrid")
    assert r.at["m1", "status"] == "routed" and r.at["m1", "source"] == "osm"
    assert r.at["m2", "status"] == "routed"
    assert r.at["m3", "status"] == "routed" and r.at["m3", "source"] == "osm+gap"
    assert r.at["m3", "gap_edges"] == 1
    # the 170 m gap is neither a bridge nor a link; only the retry pass crosses it
    assert r.at["m4", "status"] == "routed" and r.at["m4", "source"] == "osm+long_gap"
    assert r.at["m5", "status"] == "routed" or r.at["m5", "status"] == "no_path"
    assert 0.5 < r.at["m1", "ratio"] < 1.5


def test_parallel_route_preference(graph):
    plain = Circuit("p", "A - D 999", 380, "a|380", "d|380", 1.5)
    labelled = Circuit("q", "A - D 501", 380, "a|380", "d|380", 1.5)
    north = [(10.0, 51.0), (10.0 - D, 51.0 + 0.5 * D), (10.0 - 2 * D, 51.0)]
    on_route = Circuit("s", "A - D", 380, "a|380", "d|380", 1.5, route=north)
    r = route_circuits(graph, [plain, labelled, on_route]).set_index("mrid")
    ways = graph.edges.set_index("edge_id")["way_id"]
    assert set(ways[r.at["p", "edges"]]) == {5}
    assert set(ways[r.at["q", "edges"]]) == {6}
    assert r.at["q", "on_ref_km"] > 0
    assert set(ways[r.at["s", "edges"]]) == {6}
    assert r.at["s", "on_model_route_km"] > 0


def test_relation_penalty(graph):
    c = Circuit("r", "A - D RELCODE", 380, "a|380", "d|380", 1.5)
    r = route_circuits(graph, [c], relations={"relcode": {6}}).iloc[0]
    ways = graph.edges.set_index("edge_id")["way_id"]
    assert set(ways[r["edges"]]) == {6} and r["on_relation_km"] > 0


def test_edge_check_and_findings(graph):
    cs = [
        Circuit("m1", "A - B 400", 380, "a|380", "b|380", 3.5),
        Circuit("m6", "A - B 405", 380, "a|380", "b|380", 3.5),
        Circuit("m7", "A - B 406", 380, "a|380", "b|380", 3.5),
        Circuit("m3", "T - C 402", 380, "t#1|380", "c|380", 2.2),
    ]
    r = route_circuits(graph, cs)
    chk = edge_check(graph, r).set_index("way_id")
    assert chk.loc[1, "class"] == "model_more" and chk.loc[1, "model_n"] == 3
    assert chk.loc[3, "class"] == "model_fewer"
    assert chk.loc[5, "class"] == "model_fewer" and chk.loc[5, "model_n"] == 0
    ranked = findings(edge_check(graph, r), graph)
    assert ranked["rank"].tolist() == list(range(1, len(ranked) + 1))
    assert "match" not in set(ranked["class"])
    clean = clean_findings(ranked)
    assert set(clean["class"]) <= {"model_fewer", "model_more"}
    assert (clean["model_n"] >= 1).all()
    # ways 1 and 2 carry the same three circuits and the same class: one finding
    more = ranked[ranked["class"] == "model_more"]
    assert len(more) == 1 and more.iloc[0]["n_edges"] == 2


def test_helpers():
    assert circuit_number("Pasewalk - Güstrow 315") == "315"
    assert circuit_number("Ohne Nummer") == ""
    assert circuit_number("Lehrte - Wahle 1") == ""
    assert classify(2, 2.0) == "match"
    assert classify(1, 2.0) == "model_fewer"
    assert classify(3, 2.0) == "model_more"
    assert classify(1, np.nan) == "no_tag"


OUT = Path(os.environ.get("TRANPY_NETWORK_OUT", Path.home() / "data/grid/v5/network"))


@pytest.mark.skipif(not (OUT / "circuit_routes.parquet").exists(), reason="no network output")
def test_known_cases():
    """Known cases on the strict graph: routed, or reported with a reason (no fallbacks)."""
    r = pd.read_parquet(OUT / "circuit_routes.parquet")
    by = r.set_index("name")
    for n in ("Pasewalk - Güstrow 315", "Pasewalk - Güstrow 316"):
        assert by.at[n, "status"] == "routed"
        assert 0.7 <= by.at[n, "ratio"] <= 1.5
    for n in (
        "Neuenhagen - Hennigsdorf 293",
        "Neuenhagen - Wustermark 294",
        "Hennigsdorf - Wustermark 295",
    ):
        assert by.at[n, "status"] == "routed"
    # breaks of at most 50 m are joined: these must route
    for n in ("Görries - Güstrow 423", "Wessin - Güstrow 424", "Neuenhagen - Malchow 520"):
        assert by.at[n, "status"] == "routed", n
    # real OSM gaps over 50 m (Preilack 243 m, north of Weida 158 m) may stay unrouted
    for n in ("Preilack - Streumen 560", "Röhrsdorf - Weida 573", "Röhrsdorf - Weida 574"):
        assert by.at[n, "status"] in ("routed", "no_path_strict"), n
    if (OUT / "self_check.csv").exists():
        assert pd.read_csv(OUT / "self_check.csv").empty


def test_foreign_ref_penalty():
    """A shorter way that names only other circuits loses to the circuit's own way."""
    ways = [
        way(1, [(10.0, 51.0), (10.0 + D, 51.0 + 0.2 * D), (10.0 + 2 * D, 51.0)], ref="561;562"),
        way(2, [(10.0, 51.0), (10.0 + D, 51.0 - 0.6 * D), (10.0 + 2 * D, 51.0)], name="Ltg 9"),
    ]
    terms = pd.DataFrame([terminal("a|380", 10.0, 51.0), terminal("b|380", 10.0 + 2 * D, 51.0)])
    g = build_graph(ways, terms)
    r = route_circuits(g, [Circuit("x", "A - B 560", 380, "a|380", "b|380", 1.5)]).iloc[0]
    ways_used = set(g.edges.set_index("edge_id")["way_id"][r["edges"]])
    assert ways_used == {2} and r["foreign_ref_km"] == 0


def test_name_code_matching():
    """A circuit without a number follows the way naming its line code."""
    ways = [
        way(1, [(10.0, 51.0), (10.0 + D, 51.0 + 0.2 * D), (10.0 + 2 * D, 51.0)]),
        way(2, [(10.0, 51.0), (10.0 + D, 51.0 - 0.5 * D), (10.0 + 2 * D, 51.0)], name="Bienwald"),
    ]
    terms = pd.DataFrame([terminal("a|380", 10.0, 51.0), terminal("b|380", 10.0 + 2 * D, 51.0)])
    g = build_graph(ways, terms)
    r = route_circuits(g, [Circuit("x", "A - B BIENWALD W", 380, "a|380", "b|380", 1.5)])
    assert set(g.edges.set_index("edge_id")["way_id"][r.iloc[0]["edges"]]) == {2}
    assert circuit_code("Daxlanden - Mutterstadt BIENWD W") == "bienwd"
    assert fold("Güstrow Duennwald") == "gustrow dunnwald"


def test_clip_at_footprint_outline():
    """Routes end where they first enter the end station's footprint; the gap is in it."""
    ways = [
        way(1, [(10.0, 51.0), (10.0 + D, 51.0), (10.0 + 2 * D, 51.0)]),
        way(2, [(10.0 + 2 * D + 0.0003, 51.0), (10.0 + 2 * D + 0.002, 51.0)]),
    ]
    a_box = fallback_box(10.0, 51.0, 1)
    b_box = fallback_box(10.0 + 2 * D + 0.0005, 51.0, 4)
    terms = pd.DataFrame(
        [terminal("a|380", 10.0, 51.0), terminal("b|380", 10.0 + 2 * D + 0.0005, 51.0)]
    )
    g = build_graph(ways, terms, footprints={"a": a_box, "b": b_box})
    gap = g.edges[g.edges["kind"] == "gap"]
    assert len(gap) == 1 and gap["gap_where"].iloc[0] == "station"
    r = route_circuits(g, [Circuit("x", "A - B 500", 380, "a|380", "b|380", 1.5)]).iloc[0]
    assert r["entry1_kind"] == "outline" and r["entry2_kind"] == "outline"
    assert np.isclose(r["entry1_lon"], a_box.bounds[2], atol=1e-6)
    assert np.isclose(r["entry2_lon"], b_box.bounds[0], atol=1e-6)
    assert r["clip_km"] < r["route_km"]
    assert r["midline_gaps"] == 0


def test_near_other_station_penalty():
    """A slightly shorter way past a third station loses to one that keeps clear of it."""
    ways = [
        way(1, [(10.0, 51.0), (10.0 + D, 51.0 + 0.3 * D), (10.0 + 2 * D, 51.0)]),
        way(2, [(10.0, 51.0), (10.0 + D, 51.0 - 0.4 * D), (10.0 + 2 * D, 51.0)]),
    ]
    terms = pd.DataFrame([terminal("a|380", 10.0, 51.0), terminal("b|380", 10.0 + 2 * D, 51.0)])
    third = fallback_box(10.0 + D, 51.0 + 0.3 * D + 0.002, 1)
    g = build_graph(ways, terms, footprints={"x": third})
    r = route_circuits(g, [Circuit("c", "A - B 500", 380, "a|380", "b|380", 1.5)]).iloc[0]
    assert set(g.edges.set_index("edge_id")["way_id"][r["edges"]]) == {2}
    assert r["near_other_km"] == 0


def test_renumbered_way_naming_both_stations():
    """A way naming another number but both end stations is not penalised (JAO renumbering)."""
    ways = [
        way(
            1,
            [(10.0, 51.0), (10.0 + D, 51.0 + 0.2 * D), (10.0 + 2 * D, 51.0)],
            name="Ltg 575 A - B",
        ),
        way(2, [(10.0, 51.0), (10.0 + D, 51.0 - 0.6 * D), (10.0 + 2 * D, 51.0)]),
    ]
    terms = pd.DataFrame(
        [terminal("aaaa|380", 10.0, 51.0), terminal("bbbb|380", 10.0 + 2 * D, 51.0)]
    )
    ways[0]["name"] = "Ltg 575 Aaaa - Bbbb"
    g = build_graph(ways, terms)
    c = Circuit("x", "Aaaa - Bbbb 573", 380, "aaaa|380", "bbbb|380", 1.5)
    r = route_circuits(g, [c]).iloc[0]
    assert set(g.edges.set_index("edge_id")["way_id"][r["edges"]]) == {1}
    assert r["foreign_ref_km"] == 0


def strict_way(i, nodes, coords, kv=380, ref="", circuits=2.0):
    return {**way(i, coords, kv=kv, ref=ref, circuits=circuits), "nodes": nodes}


def test_strict_joins_only_at_shared_node_ids():
    """Coinciding coordinates with different node ids do not join; a shared id does."""
    a, m, b = (10.0, 51.0), (10.0 + D, 51.0), (10.0 + 2 * D, 51.0)
    ways = [
        strict_way(1, [1, 2], [a, m]),
        strict_way(2, [3, 4], [m, b]),  # touches way 1 at m, but node 3 != node 2
        strict_way(3, [2, 5], [m, (10.0 + D, 51.0 - D)]),  # shares node 2 with way 1
    ]
    terms = pd.DataFrame(
        [
            terminal("a|380", *a),
            terminal("b|380", *b),
            terminal("c|380", 10.0 + D, 51.0 - D),
        ]
    )
    boxes = {k: fallback_box(*xy, 1) for k, xy in (("a", a), ("b", b), ("c", (10.0 + D, 51.0 - D)))}
    g = build_graph(ways, terms, boxes, strict=True, joins=False)
    assert set(g.edges["kind"]) == {"line"}
    cs = [
        Circuit("ab", "A - B 500", 380, "a|380", "b|380", 1.4),
        Circuit("ac", "A - C 501", 380, "a|380", "c|380", 1.4),
    ]
    r = route_circuits(g, cs)
    pieces = r.attrs["pieces"]
    r = r.set_index("mrid")
    assert r.at["ab", "status"] == "no_path_strict"
    assert r.at["ab", "break_gap_m"] < 1.0  # the break is where ways 1 and 2 touch
    assert r.at["ac", "status"] == "routed"
    got = pieces[pieces["mrid"] == "ac"]
    assert list(got.loc[got["kind"] == "way", "way_id"]) == [1, 3]
    assert self_check(g, r.reset_index(), pieces).empty


def test_strict_cut_at_first_footprint_contact():
    """The route ends where it first touches the end footprint; yard ways are not walked."""
    a, b = (10.0, 51.0), (10.0 + 2 * D, 51.0)
    yard = [(10.0 + 2 * D - 0.0012, 51.0), (10.0 + 2 * D + 0.001, 51.0005)]
    ways = [
        strict_way(1, [1, 2, 3], [a, (10.0 + D, 51.0), yard[0]]),
        strict_way(2, [3, 4], [yard[0], yard[1]]),  # yard-internal way
    ]
    terms = pd.DataFrame([terminal("a|380", *a), terminal("b|380", *b)])
    box_b = fallback_box(*b, 6)
    g = build_graph(ways, terms, {"a": fallback_box(*a, 1), "b": box_b}, strict=True)
    r = route_circuits(g, [Circuit("x", "A - B 500", 380, "a|380", "b|380", 1.4)])
    row = r.iloc[0]
    assert row["status"] == "routed" and row["entry2_kind"] == "outline"
    assert np.isclose(row["entry2_lon"], box_b.bounds[0], atol=1e-6)
    assert row["extend2_m"] == 0


def test_strict_detour_has_no_pieces():
    """A strict path far longer than the circuit (ratio rule) is a dashed no_path_strict without
    pieces: it is not drawn on the foreign ways of the long way round."""
    a, b = (10.0, 51.0), (10.0 + 2 * D, 51.0)
    far = [a, (10.0, 51.0 + 10 * D), (10.0 + 2 * D, 51.0 + 10 * D), b]
    ways = [strict_way(1, [1, 2, 3, 4], far)]
    terms = pd.DataFrame([terminal("a|380", *a), terminal("b|380", *b)])
    g = build_graph(ways, terms, {"a": fallback_box(*a, 1), "b": fallback_box(*b, 1)}, strict=True)
    r = route_circuits(g, [Circuit("x", "A - B 500", 380, "a|380", "b|380", 1.4)])
    row = r.iloc[0]
    assert row["status"] == "no_path_strict" and row["detour_km"] > 20
    assert r.attrs["pieces"].empty


def test_accept_detour_override(tmp_path):
    """A reviewed detour (status accept_detour in route_overrides.csv) stays routed and is
    drawn on its OSM ways."""
    a, b = (10.0, 51.0), (10.0 + 2 * D, 51.0)
    far = [a, (10.0, 51.0 + 10 * D), (10.0 + 2 * D, 51.0 + 10 * D), b]
    ways = [strict_way(1, [1, 2, 3, 4], far)]
    terms = pd.DataFrame([terminal("a|380", *a), terminal("b|380", *b)])
    g = build_graph(ways, terms, {"a": fallback_box(*a, 1), "b": fallback_box(*b, 1)}, strict=True)
    f = tmp_path / "route_overrides.csv"
    f.write_text("mrid,status,reason\nx,accept_detour,tie: model km is one TSO part\n")
    r = route_circuits(
        g, [Circuit("x", "A - B 500", 380, "a|380", "b|380", 1.4)], None, detour_overrides(f)
    )
    r2, p2 = apply_overrides(r, r.attrs["pieces"], f)
    assert r2.iloc[0]["status"] == "routed" and r2.iloc[0]["ratio"] > 1.5
    assert r2.iloc[0]["override_reason"].startswith("tie")
    assert set(p2["way_id"]) == {1}


def test_join_continues_direction_only():
    """A 30 m break in line continuation is joined; a perpendicular end and a crossing are not."""
    a, b = (10.0, 51.0), (10.0 + 2 * D, 51.0)
    m1 = (10.0 + D, 51.0)
    m2 = (10.0 + D + 0.0004, 51.0)  # ~28 m further east
    ways = [
        strict_way(1, [1, 2], [a, m1]),
        strict_way(2, [3, 4], [m2, b]),
        # a perpendicular spur ending 25 m from way 1's end: not a continuation
        strict_way(3, [5, 6], [(10.0 + D, 51.0 + 0.0002), (10.0 + D, 51.0 + D)]),
    ]
    terms = pd.DataFrame([terminal("a|380", *a), terminal("b|380", *b)])
    boxes = {"a": fallback_box(*a, 1), "b": fallback_box(*b, 1)}
    g = build_graph(ways, terms, boxes, strict=True)
    joins = g.edges[g.edges["kind"] == "join"]
    assert len(joins) == 1 and set(joins.iloc[0]["join_ways"].split(";")) == {"1", "2"}
    assert joins.iloc[0]["length_m"] < 50
    r = route_circuits(g, [Circuit("x", "A - B 500", 380, "a|380", "b|380", 1.4)])
    assert r.iloc[0]["status"] == "routed" and r.iloc[0]["source"] == "osm+join"
    pieces = r.attrs["pieces"]
    assert "join" in set(pieces["kind"])
    assert self_check(g, r, pieces).empty
    # without joins the break stays
    g0 = build_graph(ways, terms, boxes, strict=True, joins=False)
    r0 = route_circuits(g0, [Circuit("x", "A - B 500", 380, "a|380", "b|380", 1.4)])
    assert r0.iloc[0]["status"] == "no_path_strict"


def test_join_never_links_crossing_ways():
    """Ways that cross away from the join are never joined."""
    ways = [
        strict_way(1, [1, 2], [(10.0, 51.0), (10.0 + D, 51.0)]),
        # starts 20 m past way 1's end, then bends back across it
        strict_way(
            2,
            [3, 4, 5],
            [(10.0 + D + 0.0003, 51.0), (10.0 + D + 0.002, 51.0), (10.0 + 0.5 * D, 51.0 - D)],
        ),
        strict_way(6, [7, 8], [(10.0 + 0.5 * D, 51.0 + D), (10.0 + 0.5 * D, 51.0 - 2 * D)]),
    ]
    g = build_graph(ways, None, {}, strict=True)
    for j in g.edges[g.edges["kind"] == "join"].itertuples():
        assert "6" not in j.join_ways.split(";")


def test_route_overrides(tmp_path, graph):
    """An override sets the status and reason and drops the circuit's pieces."""
    cs = [Circuit("m1", "A - B 400", 380, "a|380", "b|380", 3.5)]
    r = route_circuits(graph, cs)
    pieces = r.attrs["pieces"]
    f = tmp_path / "route_overrides.csv"
    f.write_text("mrid,status,reason\nm1,no_path_strict,wrong corridor\n", encoding="utf-8")
    r2, p2 = apply_overrides(r, pieces, f)
    assert r2.iloc[0]["status"] == "no_path_strict"
    assert r2.iloc[0]["override_reason"] == "wrong corridor"
    assert p2.empty


def test_construction_ways_only_where_no_in_service_path(tmp_path):
    """Ways tagged under construction (separate Overpass file or construction:power) join
    the graph at a high cost: a circuit takes them only where no in-service path exists;
    their bay/busbar ways are left out."""
    import json

    from tranpy.grid_network import _power_kind, osm_ways_strict

    assert _power_kind({"power": "line"}) == ("line", False)
    assert _power_kind({"power": "line", "construction": "yes"}) == ("line", False)
    assert _power_kind({"construction:power": "line"}) == ("line", True)
    assert _power_kind({"power": "construction", "construction": "cable"}) == ("cable", True)
    assert _power_kind({"power": "substation"}) == ("", False)
    a, m, b, c = (10.0, 51.0), (10.0 + D, 51.0), (10.0 + 2 * D, 51.0), (10.0 + D, 51.0 + D)

    def doc(ways):
        nodes = {n: p for w in ways for n, p in zip(w["nodes"], w["coords"], strict=True)}
        return {
            "elements": [
                {"type": "node", "id": n, "lon": p[0], "lat": p[1]} for n, p in nodes.items()
            ]
            + [{"type": "way", "id": w["id"], "nodes": w["nodes"], "tags": w["tags"]} for w in ways]
        }

    live = [
        {
            "id": 1,
            "nodes": [1, 2],
            "coords": [a, m],
            "tags": {"power": "line", "voltage": "380000", "circuits": "2"},
        }
    ]
    built = [
        {
            "id": 2,
            "nodes": [2, 3],
            "coords": [m, b],
            "tags": {"construction:power": "line", "voltage": "380000"},
        },
        {
            "id": 3,
            "nodes": [2, 4],
            "coords": [m, c],
            "tags": {"construction:power": "line", "line": "bay", "voltage": "380000"},
        },
    ]
    (tmp_path / "live.json").write_text(json.dumps(doc(live)))
    (tmp_path / "built.json").write_text(json.dumps(doc(built)))
    ways = osm_ways_strict([tmp_path / "live.json"], [tmp_path / "built.json"])
    assert {w["id"]: w["construction"] for w in ways} == {1: False, 2: True}
    terms = pd.DataFrame([terminal("a|380", *a), terminal("b|380", *b)])
    boxes = {k: fallback_box(*xy, 1) for k, xy in (("a", a), ("b", b))}
    g = build_graph(ways, terms, boxes, strict=True, joins=False)
    assert g.edges.set_index("way_id")["construction"].to_dict() == {1: False, 2: True}
    r = route_circuits(g, [Circuit("x", "A - B 500", 380, "a|380", "b|380", 1.4)]).iloc[0]
    assert r["status"] == "routed"
    assert set(g.edges.set_index("edge_id")["way_id"][r["edges"]]) == {1, 2}


def test_hvdc_ways_left_out_of_ac_routing():
    """DC-only ways (A-Nord, Ultranet sections, frequency=0) are not in the AC graph; shared
    AC/DC towers and AC converter connections are."""
    from tranpy.grid_network import is_hvdc_way

    assert is_hvdc_way({"name": "A-Nord", "frequency": "50", "location": "underground"})
    assert is_hvdc_way({"name": "Ultranet Abschnitt Bürstadt-Wallstadt (A1)"})
    assert is_hvdc_way({"name": "", "frequency": "0"})
    assert not is_hvdc_way({"name": "Ultranet;380 kV Bergheim West", "frequency": "50"})
    assert not is_hvdc_way({"name": "Anbindung SuedOstLink", "frequency": "50"})
    assert not is_hvdc_way({"name": "Suedostlink;Ltg 437 und 438", "frequency": "50;0"})
    assert not is_hvdc_way({"name": "380kV Hessen Ost", "frequency": ""})


def test_reviewed_join_override():
    """A branch ending 30 m from its tee point at a right angle is not joined by the rules;
    a reviewed join (node_a, node_b) adds the edge after all others."""
    a, m, b = (10.0, 51.0), (10.0 + D, 51.0), (10.0 + 2 * D, 51.0)
    ways = [
        strict_way(1, [1, 2], [a, m]),
        strict_way(2, [2, 3], [m, b], ref="302"),
        strict_way(3, [4, 5], [(10.0 + D, 51.0 - 0.00027), (10.0 + D, 51.0 - 3 * D)], ref="320"),
    ]
    g = build_graph(ways, None, {}, strict=True)
    assert (g.edges["kind"] == "join").sum() == 0
    extra = pd.DataFrame({"node_a": [4], "node_b": [2], "reason": ["tee"]})
    g2 = build_graph(ways, None, {}, strict=True, extra_joins=extra)
    j = g2.edges[g2.edges["kind"] == "join"]
    assert len(j) == 1 and j.iloc[0]["edge_id"] == len(g.edges)
    assert (g2.edges.iloc[: len(g.edges)]["u"].to_numpy() == g.edges["u"].to_numpy()).all()


def test_hvdc_edges_are_never_routed():
    """A DC-only way stays in the graph but no AC circuit uses it, even when shorter."""
    a, b = (10.0, 51.0), (10.0 + 2 * D, 51.0)
    ways = [
        strict_way(1, [1, 2, 3], [a, (10.0 + D, 51.0), b]),
        strict_way(2, [1, 4, 3], [a, (10.0 + D, 51.0 + 0.3 * D), b]),
    ]
    ways[0]["hvdc"] = True
    terms = pd.DataFrame([terminal("a|380", *a), terminal("b|380", *b)])
    g = build_graph(ways, terms, {"a": fallback_box(*a, 1), "b": fallback_box(*b, 1)}, strict=True)
    r = route_circuits(g, [Circuit("x", "A - B 500", 380, "a|380", "b|380", 1.5)]).iloc[0]
    used = set(g.edges.set_index("edge_id")["way_id"][r["edges"]])
    assert r["status"] == "routed" and used == {2}
