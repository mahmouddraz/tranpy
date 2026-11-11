"""Tests for scripts/map_acceptance.py on a tiny synthetic export."""

import importlib.util
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "map_acceptance.py"
spec = importlib.util.spec_from_file_location("map_acceptance", SCRIPT)
ma = importlib.util.module_from_spec(spec)
sys.modules["map_acceptance"] = ma
spec.loader.exec_module(ma)

CRS = "EPSG:3035"
X0, Y0 = 4_300_000.0, 3_200_000.0  # central Germany in EPSG:3035


def _xy(dx, dy):
    return (X0 + dx, Y0 + dy)


def _layers(lines_rows, geoms):
    lines = gpd.GeoDataFrame(pd.DataFrame(lines_rows), geometry=geoms, crs=CRS).to_crs(4326)
    foot = gpd.GeoDataFrame(
        {"site": ["a", "b"]},
        geometry=[box(*_xy(-100, -100), *_xy(0, 100)), box(*_xy(10_000, -100), *_xy(10_100, 100))],
        crs=CRS,
    ).to_crs(4326)
    subs = gpd.GeoDataFrame(
        {"site": ["a", "b", "j"], "kind": ["substation", "substation", "tee"]},
        geometry=[Point(_xy(-50, 0)), Point(_xy(10_050, 0)), Point(_xy(5_000, 3_000))],
        crs=CRS,
    ).to_crs(4326)
    buses = pd.DataFrame({"bus": ["a|380", "b|380", "j|380"], "site": ["a", "b", "j"]})
    # OSM ways "1".."9": the straight corridor a-b (the wiggles stay within 15 m of it)
    ways = gpd.GeoDataFrame(
        {"osm_id": [str(i) for i in range(1, 10)]},
        geometry=[LineString([_xy(0, 15), _xy(10_000, 15)])] * 9,
        crs=CRS,
    ).to_crs(4326)
    return {
        "lines": lines,
        "station_footprints": foot,
        "substations": subs,
        "buses": buses,
        "tower_lines": ways,
    }


def _row(mrid, piece, pieces, quality="verified", way="", bus2="b|380", cno=0, n=1):
    return {
        "mrid": mrid,
        "name": mrid,
        "status": "in_operation",
        "bus1": "a|380",
        "bus2": bus2,
        "piece": piece,
        "pieces": pieces,
        "route_quality": quality,
        "tower_way": way,
        "circuit_no": cno,
        "circuits": n,
        "hvdc": False,
    }


def _model(*mrids, km=10.0):
    return pd.DataFrame(
        {
            "mrid": list(mrids),
            "name": list(mrids),
            "status": "in_operation",
            "length_km": km,
            "nominal_voltage_kv": 380.0,
        }
    )


def _wiggle(x0, x1, n=40):
    return LineString([_xy(x0 + (x1 - x0) * i / n, 30 * (i % 2)) for i in range(n + 1)])


def test_clean_map_passes():
    layers = _layers(
        [_row("c1", 0, 2, way="1"), _row("c1", 1, 2, way="2")],
        [_wiggle(0, 5_000), _wiggle(5_000, 10_000)],
    )
    res = ma.run_checks(layers, _model("c1"), {"source_sha256": "abc"}, "abc")
    assert res["passed"], res["hard"]
    assert res["metrics"]["share_length_on_osm_ways"] == pytest.approx(1.0)


def test_failures_are_detected():
    rows = [
        _row("ok", 0, 1, way="1"),
        _row("straight", 0, 1),  # undashed 10 km straight line
        _row("dashed", 0, 1, quality="straight"),  # dashed straight is allowed
        _row("gap", 0, 2, way="3"),
        _row("gap", 1, 2, way="4"),
        _row("offend", 0, 1, bus2="j|380", way="5"),
    ]
    geoms = [
        _wiggle(0, 10_000),
        LineString([_xy(0, 0), _xy(10_000, 0)]),
        LineString([_xy(0, 0), _xy(10_000, 0)]),
        _wiggle(0, 4_000),
        _wiggle(4_100, 10_000),  # 100 m gap
        _wiggle(0, 10_000),  # ends at b, but its own end is junction j
    ]
    layers = _layers(rows, geoms)
    model = _model("ok", "straight", "dashed", "gap", "offend", "missing")
    res = ma.run_checks(layers, model, {"source_sha256": "old"}, "new")
    hard = res["hard"]
    assert not res["passed"]
    assert not hard["count_equals_model"]
    assert res["lists"]["missing"] == ["missing"]
    assert [r["mrid"] for r in res["lists"]["straight_undashed"]] == ["straight"]
    assert res["metrics"]["straight_dashed_pieces"] == 1
    assert [r["mrid"] for r in res["lists"]["gaps"]] == ["gap"]
    assert [r["mrid"] for r in res["lists"]["ends_off_station"]] == ["offend"]
    assert not hard["metadata_names_model_sha"]
    assert hard["drawn_once"]


def test_ratio_duplicates_and_slots():
    rows = [
        _row("dup", 0, 3, way="9"),
        _row("dup", 1, 3, way="8"),
        _row("dup", 2, 3, way="9"),  # same OSM edge twice (not a split piece)
        _row("s1", 0, 2, way="1", cno=0, n=2),
        _row("s1", 1, 2, way="2", cno=1, n=2),  # slot changes, set {s1, s2} unchanged
        _row("s2", 0, 2, way="1", cno=1, n=2),
        _row("s2", 1, 2, way="2", cno=0, n=2),
    ]
    geoms = [_wiggle(0, 3_000), _wiggle(3_000, 6_000), _wiggle(6_000, 10_000)]
    geoms += [_wiggle(0, 5_000), _wiggle(5_000, 10_000)] * 2
    layers = _layers(rows, geoms)
    model = _model("dup", "s1", "s2", km=5.0)  # drawn ~10 km -> ratio ~2
    # station_order_m 0: the change at the box would otherwise be the allowed station order
    res = ma.run_checks(
        layers, model, {"source_sha256": "x"}, "x", ma.Settings(station_order_m=0.0)
    )
    assert not res["hard"]["drawn_once"]
    assert {r["mrid"] for r in res["lists"]["slot_changes"]} == {"s1", "s2"}
    assert res["metrics"]["ratio_outside"] == 3


def test_main_writes_reports_and_exit_code(tmp_path, monkeypatch):
    layers = _layers([_row("c1", 0, 1, way="1")], [_wiggle(0, 10_000)])
    monkeypatch.setattr(ma, "read_export", lambda p: (layers, {"source_sha256": "s"}))
    monkeypatch.setattr(ma, "read_model_lines", lambda p: _model("c1", "c2"))
    monkeypatch.setattr(ma, "sha256", lambda p: "s")
    code = ma.main(["--export", "x", "--model", "y", "--out", str(tmp_path)])
    assert code == 1
    assert (tmp_path / "acceptance.json").exists()
    assert "FAIL" in (tmp_path / "acceptance.md").read_text()


def _geo(geom, way="1"):
    layers = _layers([_row("c", 0, 1, way=way)], [geom])
    res = ma.run_checks(layers, _model("c"), {"source_sha256": "x"}, "x")
    return res["hard"], res["events"]


def test_geometry_checks_clean_line():
    hard, events = _geo(_wiggle(0, 10_000))
    assert all(hard[k] for k in ma.GEOMETRY_CHECKS), hard
    assert events == []


def test_spike_zigzag_loop_offway_and_station_end():
    base = [_xy(x, 15) for x in range(0, 10_001, 500)]
    spike = [*base[:10], _xy(4_750, 150), *base[10:]]  # out 135 m and back within 100 m
    hard, events = _geo(LineString(spike))
    assert not hard["no_spike"]
    assert {e["check"] for e in events} >= {"no_spike"}

    zig = [*base[:10], _xy(4600, 15), _xy(4610, 120), _xy(6000, 120), *base[13:]]
    hard, _ = _geo(LineString(zig))
    assert not hard["no_zigzag"]
    assert not hard["on_own_way"]  # 120 m off for 1.4 km

    loop = [*base[:10], _xy(5200, 15), _xy(5200, 300), _xy(4800, 300), _xy(4800, 0), *base[10:]]
    hard, _ = _geo(LineString(loop))
    assert not hard["no_loop"]

    back = [*base[:10], _xy(5000, 15), _xy(4000, 15), _xy(4000, 16), _xy(10000, 16)]
    hard, _ = _geo(LineString(back))
    assert not hard["no_loop"]  # back-tracking 1 km

    deep = [_xy(-100, 0), _xy(-2, 0), _xy(-2, 90), _xy(-98, 90), _xy(-98, 15), *base]
    hard, _ = _geo(LineString(deep))
    assert not hard["station_end"]  # ~280 m drawn inside the end box


def test_baked_offset_is_allowed():
    layers = _layers([_row("c", 0, 1, way="1", n=4)], [LineString([_xy(0, 150), _xy(10_000, 150)])])
    layers["lines"]["offset_baked"] = True
    layers["lines"]["max_n"] = 4  # up to 1.5 x 90 m + 30 m from the way
    res = ma.run_checks(layers, _model("c"), {"source_sha256": "x"}, "x")
    assert res["hard"]["on_own_way"]
    assert res["metrics"]["share_on_own_way_strict_30m"] < 0.1


def test_station_clip_ends_on_buffer_edge_and_edge_pieces():
    # an OSM-graph export cut 150 m outside the boxes: ends at the buffer edge pass, a line
    # still running into the box fails station_end; one way split into two graph edges is
    # not "drawn twice"
    rows = [
        {**_row("c1", 0, 2, way="1"), "edge_id": 10, "piece_kind": "way"},
        {**_row("c1", 1, 2, way="1"), "edge_id": 11, "piece_kind": "way"},
    ]
    geoms = [
        LineString([_xy(150, 15), _xy(5_000, 15)]),
        LineString([_xy(5_000, 15), _xy(9_850, 15)]),
    ]
    meta = {"source_sha256": "abc", "station_clip_m": "150"}
    res = ma.run_checks(_layers(rows, geoms), _model("c1"), meta, "abc")
    assert res["hard"]["ends_on_own_station"] and res["hard"]["station_end"], res["lists"]
    assert res["hard"]["drawn_once"] and res["settings"]["station_clip_m"] == 150.0
    # without the clip the same ends are 150 m off their boxes
    res0 = ma.run_checks(_layers(rows, geoms), _model("c1"), {"source_sha256": "abc"}, "abc")
    assert not res0["hard"]["ends_on_own_station"]
    # into the yard: inside the buffer
    deep = [LineString([_xy(10, 15), _xy(5_000, 15)]), geoms[1]]
    res2 = ma.run_checks(_layers(rows, deep), _model("c1"), meta, "abc")
    assert not res2["hard"]["station_end"]
    # a short line left uncut (station_cut none) is not a station_end failure
    rows_none = [{**r, "station_cut": "none"} for r in rows]
    res3 = ma.run_checks(_layers(rows_none, deep), _model("c1"), meta, "abc")
    assert res3["hard"]["station_end"]


def test_station_touch_flags_floating_ends():
    rows = [_row("c1", 0, 1, way="1")]
    ok = ma.run_checks(
        _layers(rows, [LineString([_xy(0, 15), _xy(10_000, 15)])]), _model("c1"), {}, ""
    )
    assert ok["hard"]["station_touch"]
    short = ma.run_checks(
        _layers(rows, [LineString([_xy(0, 15), _xy(9_970, 15)])]), _model("c1"), {}, ""
    )
    assert not short["hard"]["station_touch"]
    assert short["lists"]["station_touch"][0]["worst_end_m"] == pytest.approx(30, abs=1)


def test_halo_popup_consistency():
    """The check class follows from model_n vs circuits_tag; the popup list has model_n."""
    one = '[{"mrid": "a", "name": "A"}]'
    two = '[{"mrid": "a", "name": "A"}, {"mrid": "b", "name": "B"}]'
    rows = [
        (1, 2.0, 2, two, "match"),  # ok
        (2, 2.0, 1, one, "explained"),  # ok: fewer, explained
        (3, 2.0, 1, one, "model_fewer"),  # ok
        (4, 2.0, 1, two, "model_fewer"),  # popup lists 2, model_n 1
        (5, 2.0, 1, one, "match"),  # class does not follow the counts
        (6, 1.0, 1, one, "explained"),  # equal counts cannot be explained
        (7, None, 0, "[]", "no_model"),  # ok
        (8, None, 1, one, "no_tag"),  # ok
    ]
    ol = gpd.GeoDataFrame(
        pd.DataFrame(
            rows, columns=["way_id", "circuits_tag", "model_n", "model_circuits", "check"]
        ),
        geometry=[LineString([_xy(0, k), _xy(500, k)]) for k in range(len(rows))],
        crs=CRS,
    )
    bad = ma.halo_popup_mismatches({"osm_lines": ol})
    assert [b["way_id"] for b in bad] == [4, 5, 6]


def test_sld_complete(tmp_path):
    """Every station names an existing SLD file; an empty sld/ folder fails."""
    import json

    (tmp_path / "station_details.json").write_text(
        json.dumps({"a": {"diagram": "sld/a.svg"}, "b": {"diagram": "sld/b.svg"}, "c": {}})
    )
    assert {p["problem"] for p in ma.sld_problems(tmp_path)} >= {"sld/ folder missing or empty"}
    (tmp_path / "sld").mkdir()
    (tmp_path / "sld" / "a.svg").write_text("<svg/>")
    bad = ma.sld_problems(tmp_path)
    assert sorted(p["site"] for p in bad) == ["b", "c"]


def _terminal(port_dx, symbol_dx, mrid="c1"):
    """A border terminal of site ``b`` with one port (the tie's border crossing)."""
    import json

    port = gpd.GeoSeries([Point(_xy(port_dx, 15))], crs=CRS).to_crs(4326).iloc[0]
    circ = [{"mrid": mrid, "name": mrid, "site": "b", "port": [port.x, port.y]}]
    return gpd.GeoDataFrame(
        {"circuits": [json.dumps(circ)]}, geometry=[Point(_xy(symbol_dx, 15))], crs=CRS
    ).to_crs(4326)


def test_border_cut_tie_ends_on_its_port():
    """A tie cut at the border (b abroad, its box removed) ends on its port, not at b."""
    rows = [_row("c1", 0, 1, way="1")]
    layers = _layers(rows, [LineString([_xy(0, 15), _xy(6_000, 15)])])
    layers["station_footprints"] = layers["station_footprints"].iloc[[0]]
    meta = {"border_cut_ties": "c1"}
    # without a terminal the cut end floats and the cut tie has no terminal
    bare = ma.run_checks(layers, _model("c1"), meta, "")
    assert not bare["hard"]["station_touch"] and not bare["hard"]["border_terminals"]
    layers["border_terminals"] = _terminal(6_000, 6_500)
    ok = ma.run_checks(layers, _model("c1"), meta, "")
    assert ok["hard"]["station_touch"] and ok["hard"]["ends_on_own_station"]
    assert ok["hard"]["border_terminals"] and ok["metrics"]["border_ports"] == 1
    # a line that stops short of its port floats; a port far from its symbol is flagged
    layers["border_terminals"] = _terminal(6_100, 6_100)
    assert not ma.run_checks(layers, _model("c1"), meta, "")["hard"]["station_touch"]
    layers["border_terminals"] = _terminal(6_000, 9_000)
    far = ma.run_checks(layers, _model("c1"), meta, "")
    assert not far["hard"]["border_terminals"]
    assert far["lists"]["border_terminals"][0]["problem"] == "port far from terminal"
