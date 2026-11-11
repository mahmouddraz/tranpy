"""GIS export of the grid model on a tiny synthetic model."""

from __future__ import annotations

import json
import xml.dom.minidom
from types import SimpleNamespace

import pandas as pd
import pytest

gpd = pytest.importorskip("geopandas")
pytest.importorskip("pyogrio")

from tranpy.grid_gis import (  # noqa: E402
    LAYERS,
    OVERLAY_COLUMNS,
    build_layers,
    place_tee,
    polyline_km,
    section_of,
    section_route,
    write_layers,
)
from tranpy.grid_gis_style import write_all  # noqa: E402


def _square(x0: float, y0: float, x1: float, y1: float) -> list[list[list[float]]]:
    return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]


BORDERS = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"ADM0_A3": "DEU"},
            "geometry": {"type": "Polygon", "coordinates": _square(6, 47.4, 15, 55)},
        },
        {
            "type": "Feature",
            "properties": {"ADM0_A3": "AUT"},
            "geometry": {"type": "Polygon", "coordinates": _square(9, 46, 17, 47.4)},
        },
    ],
}


def _model() -> tuple[dict[str, pd.DataFrame], SimpleNamespace]:
    nodes = pd.DataFrame(
        {
            "node": ["a|380", "b|380", "b|220", "c|220", "x|220"],
            "site": ["a", "b", "b", "c", "x"],
            "kv": [380, 380, 220, 220, 220],
            "names": [["A"], ["B"], ["B"], ["C"], ["X"]],
            "tso": ["50HERTZ", "TENNETGMBH", "TENNETGMBH", "AMPRION GMBH;TENNETGMBH", "TENNETGMBH"],
            "kind": ["substation", "substation", "substation", "substation", "foreign"],
            "lon": [13.0, 10.0, 10.0, 8.0, 12.0],
            "lat": [52.0, 52.0, 52.0, 50.0, 47.0],
            "country": ["DEU", "DEU", "DEU", "DEU", "AUT"],
        }
    )
    mrid = {n: f"m-{n}" for n in nodes["node"]}
    busbar = pd.DataFrame(
        {
            "mrid": [mrid[n] for n in nodes["node"]],
            "name": nodes["node"],
            "kind": nodes["kind"],
            "tso": nodes["tso"],
            "lon": nodes["lon"],
            "lat": nodes["lat"],
            "location_source": ["osm", "osm", "osm", "schematic", "osm"],
            "data_quality": "published",
        }
    )
    route = [{"lon": 13.0, "lat": 52.0}, {"lon": 11.5, "lat": 52.1}, {"lon": 10.0, "lat": 52.0}]
    # stops ~690 m short of B (drawn extended to B) and runs from B to A
    short = [{"lon": 10.01, "lat": 52.0}, {"lon": 11.5, "lat": 52.1}, {"lon": 13.0, "lat": 52.0}]
    # ends ~600 m from C, whose model point is schematic
    to_c = [{"lon": 10.0, "lat": 52.0}, {"lon": 9.0, "lat": 51.0}, {"lon": 8.006, "lat": 50.003}]
    seg = pd.DataFrame(
        {
            "mrid": ["l1", "l2", "l3", "l4"],
            "name": ["A - B 1", "A - B 2", "B - C", "C - X"],
            "eic": ["E1", "E2", "E3", "E4"],
            "tso": ["50HERTZ", "50HERTZ", "TenneT TSO GmbH", "TENNETGMBH"],
            "bus1_mrid": [mrid["a|380"], mrid["a|380"], mrid["b|220"], mrid["c|220"]],
            "bus2_mrid": [mrid["b|380"], mrid["b|380"], mrid["c|220"], mrid["x|220"]],
            "nominal_voltage_kv": [380.0, 380.0, 220.0, 220.0],
            "length_km": [200.0, 200.0, 250.0, 330.0],
            "r_ohm": [5.0, 5.0, 20.0, 30.0],
            "x_ohm": [50.0, 50.0, 100.0, 120.0],
            "bch_s": [1e-4] * 4,
            "rated_current_a": [2500.0, 2500.0, 1000.0, 1000.0],
            "tie_line": [False, False, True, True],
            "route": [route, short, to_c, None],
            "route_source": ["osm", "osm", None, None],
            "route_quality": ["verified", "verified", "estimated", "estimated"],
            "status": ["in_operation"] * 4,
            "data_quality": ["published"] * 4,
        }
    )
    circuits = pd.DataFrame(
        {
            "eic": ["H1"],
            "name": ["DC link"],
            "tso": ["TENNETGMBH"],
            "node0": ["b|380"],
            "node1": ["a|380"],
            "kv": [380.0],
            "length_km": [300.0],
            "imax_a": [1500.0],
            "tie_line": [False],
            "hvdc": [True],
            "route": [None],
            "route_source": [None],
            "route_quality": [None],
            "status": ["in_operation"],
            "data_quality": ["published"],
        }
    )
    transformer = pd.DataFrame(
        {
            "mrid": ["t1"],
            "name": ["TR B"],
            "eic": ["T1"],
            "tso": ["TENNETGMBH"],
            "bus1_mrid": [mrid["b|380"]],
            "bus2_mrid": [mrid["b|220"]],
            "rated_u1_kv": [400.0],
            "rated_u2_kv": [231.0],
            "r_ohm": [0.5],
            "x_ohm": [60.0],
            "rated_current_a": [866.0],
            "phase_shifter": [False],
            "data_quality": ["published"],
        }
    )
    substation = pd.DataFrame(
        {
            "mrid": ["s-a", "s-b", "s-c", "s-x"],
            "site": ["a", "b", "c", "x"],
            "lon": [13.0, 10.0, 8.0, 12.0],
            "lat": [52.0, 52.0, 50.0, 47.0],
            "tso": ["50HERTZ", "TENNETGMBH", "AMPRION GMBH", "TENNETGMBH"],
            "names": [["A"], ["B"], ["C"], ["X"]],
            "country": ["DE", "DE", "DE", ""],
        }
    )
    ext = pd.DataFrame({"mrid": ["e1"], "bus_mrid": [mrid["x|220"]]})
    plants = pd.DataFrame(
        {
            "mastr_id": ["P1", "P2", "P3", "P4"],
            "name": ["Plant 1", "Plant 2", "Plant 3", "Plant 4"],
            "operator": ["o"] * 4,
            "fuel": ["Braunkohle", "Erdgas", "Erdgas", "Erdgas"],
            "capacity_mw": [900.0, 50.0, 20.0, 10.0],
            "status": ["In Betrieb"] * 4,
            "in_operation": [True] * 4,
            "reserve": [False] * 4,
            "voltage_level": ["Höchstspannung", "Hochspannung", "Hochspannung", "Hochspannung"],
            "lon": [12.9, None, 8.1, 8.2],
            "lat": [52.1, None, 50.1, 50.0],
        }
    )
    attachments = pd.DataFrame(
        {
            "mastr_id": ["P1", "P2", "P3", "P4"],
            "node": ["a|380", "c|220", "c|220", "c|220"],
            "share": [1.0] * 4,
            "method": ["nearest"] * 4,
        }
    )
    silver = {
        "grid_de_nodes": nodes,
        "grid_de_circuits": circuits,
        "grid_de_plants": plants,
        "grid_de_plant_attachments": attachments,
    }
    cim = SimpleNamespace(
        busbar_section=busbar,
        ac_line_segment=seg,
        power_transformer=transformer,
        substation=substation,
        external_network_injection=ext,
        model_info={"model_version": "test"},
    )
    return silver, cim


@pytest.fixture(scope="module")
def layers() -> dict:
    silver, cim = _model()
    return build_layers(silver, cim, borders=BORDERS)


def test_layers_crs_and_geometry(layers: dict) -> None:
    assert set(layers) == set(LAYERS)
    for name in LAYERS:
        if name == "metadata":
            continue
        frame = layers[name]
        assert frame.crs.to_epsg() == 4326, name
        # no tower lines or station footprints given
        assert (
            len(frame)
            or name in ("tower_lines", "osm_lines", "osm_substations")
            or name.startswith("station_")
        ), name
        assert frame.geometry.is_valid.all(), name
        assert not frame.geometry.is_empty.any(), name


def test_required_columns(layers: dict) -> None:
    required = {
        "lines": {
            "name",
            "eic",
            "voltage_kv",
            "circuits",
            "tso",
            "is_interconnector",
            "from_zone",
            "to_zone",
            "hvdc",
            "route_quality",
            "status",
        },
        "substations": {"name", "max_kv", "voltages_kv", "tso", "buses", "transformers"},
        "buses": {"bus", "bus_mrid", "voltage_kv", "tso", "bidding_zone"},
        "plants": {"name", "fuel_category", "capacity_mw", "bus", "tso"},
        "neighbours": {"bus", "country", "bidding_zone", "tie_lines"},
    }
    for name, columns in required.items():
        assert columns <= set(layers[name].columns), name


def test_every_bus_has_a_tso_and_border_node_resolved(layers: dict) -> None:
    buses = layers["buses"].set_index("bus")
    assert buses["tso"].notna().all()
    assert buses.loc["a|380", "tso"] == "50Hertz"
    assert buses.loc["x|220", "bidding_zone"] == "AT"


def test_lines(layers: dict) -> None:
    lines = layers["lines"].set_index("eic")
    assert lines.loc["E1", "circuits"] == 2
    assert sorted(lines.loc[["E1", "E2"], "circuit_no"]) == [0, 1]
    assert lines.loc["E4", "route_quality"] == "straight"
    assert lines.loc["E3", "tso"] == "TenneT"  # operator name spelled differently
    assert not lines.loc["E3", "is_interconnector"] and lines.loc["E3", "tso_tie"]
    assert lines.loc["E4", "is_interconnector"] and lines.loc["E4", "neighbour"] == "AUT"
    assert lines.loc["E4", "to_zone"] == "AT"
    assert lines.loc["H1", "hvdc"]
    start = lines.loc["E1"].geometry.coords[0]
    assert start == pytest.approx((13.0, 52.0))
    assert lines.loc["E1", "same_route"] == 1 and lines.loc["E1", "display_snapped_m"] == 0
    assert layers["neighbours"]["tie_lines"].tolist() == [1]


def test_plants_and_areas(layers: dict) -> None:
    plants = layers["plants"].set_index("mastr_id")
    assert plants.loc["P1", "fuel_category"] == "lignite"
    assert plants.loc["P2", "location"] == "bus"  # no register coordinates: at its bus
    areas = layers["control_areas"]
    assert set(areas["tso"]) <= {"50Hertz", "Amprion", "TenneT", "TransnetBW"}
    assert areas["approximate"].all()
    assert "DE-LU" in set(layers["bidding_zones"]["zone"])


def test_write_round_trip(layers: dict, tmp_path) -> None:
    import pyogrio

    gpkg = write_layers(layers, tmp_path, web_dir=tmp_path / "web")
    names = {row[0] for row in pyogrio.list_layers(gpkg)}
    assert names == set(LAYERS)
    back = gpd.read_file(gpkg, layer="lines")
    assert back.crs.to_epsg() == 4326
    assert set(back["eic"]) == set(layers["lines"]["eic"])
    web = json.loads((tmp_path / "web" / "buses.geojson").read_text())
    assert len(web["features"]) == len(layers["buses"])


def test_styles_and_project_are_well_formed(tmp_path) -> None:
    out = write_all(tmp_path)
    for path in [*out["styles"], out["project"], *out["symbols"]]:
        xml.dom.minidom.parse(str(path))
    project = out["project"].read_text(encoding="utf-8")
    assert "./grid_model.gpkg|layername=lines" in project
    assert '<Absolute type="bool">false</Absolute>' in project
    assert "./grid_model.gpkg|layername=plant_connections" in project
    assert "./grid_model.gpkg|layername=plant_aggregates" in project
    names = {p.name for p in out["symbols"]}
    assert {"substation_coupling.svg", "transformer_380_220.svg"} <= names
    lines_qml = (tmp_path / "qgis" / "styles" / "lines_voltage.qml").read_text(encoding="utf-8")
    assert "circuit_no" in lines_qml  # parallel circuits offset side by side


def test_display_fixes(layers: dict) -> None:
    lines = layers["lines"].set_index("eic")
    # E2 stops ~690 m short of B: extended for display, the gap recorded
    assert 600 < lines.loc["E2", "display_snapped_m"] < 800
    # E2 runs B -> A in the model; drawn in the corridor's direction (A -> B) for the offset
    coords = list(lines.loc["E2"].geometry.coords)
    assert coords[0] == pytest.approx((13.0, 52.0)) and coords[-1] == pytest.approx((10.0, 52.0))
    assert lines.loc["E1", "corridor"] == lines.loc["E2", "corridor"]
    assert lines.loc[["E1", "E2"], "circuits"].tolist() == [2, 2]
    # C is schematic: drawn at the end of the route that terminates there
    subs = layers["substations"].set_index("site")
    assert subs.loc["c", "location_flag"] == "schematic"
    assert 400 < subs.loc["c", "display_relocated_m"] < 700
    assert (subs.loc["c"].geometry.x, subs.loc["c"].geometry.y) == pytest.approx((8.006, 50.003))
    assert lines.loc["E4"].geometry.coords[0] == pytest.approx((8.006, 50.003))
    assert lines.loc["E3", "display_snapped_m"] == 0
    assert subs.loc["b", "coupling"] and subs.loc["b", "symbol"] == "coupling"
    assert subs.loc["a", "symbol"] == "380"
    buses = layers["buses"].set_index("bus")
    assert buses.loc["c|220", "display_relocated_m"] > 400


def test_plant_display(layers: dict) -> None:
    plants = layers["plants"].set_index("mastr_id")
    assert plants.loc["P1", "display"] == "individual"
    assert plants.loc["P3", "display"] == "aggregated"
    links = layers["plant_connections"]
    assert links["mastr_id"].tolist() == ["P1"]
    assert links.geometry.iloc[0].coords[-1] == pytest.approx((13.0, 52.0))
    agg = layers["plant_aggregates"]
    assert len(agg) == 1
    row = agg.iloc[0]
    assert (row["bus"], row["fuel_category"], row["units"]) == ("c|220", "gas", 2)
    assert row["capacity_mw"] == pytest.approx(30.0)
    trafo = layers["transformers"].iloc[0]
    assert trafo["slot"] == 0 and trafo["site_transformers"] == 1


def test_place_tee_at_station_from_short_leg() -> None:
    # a 0.28 km leg puts the tee at that station, whatever the other legs say
    legs = [
        ("tiengen", (8.26, 47.63), 114.25, True),
        ("herbertingen", (9.42, 48.06), 0.28, True),
        ("y-werben", (9.7, 47.4), 88.1, False),
    ]
    placed = place_tee(legs)
    assert placed is not None
    point, method, evidence = placed
    assert point == (9.42, 48.06) and method == "tee_at_station" and "herbertingen" in evidence
    # 0 km is a missing length (BLUDNZ W "Y-Werben" leg); a short leg to another tee and a
    # leg of 0.5 km or more do not count either
    assert (
        place_tee([("werben", (9.72, 47.43), 0.0, True), ("burs", (9.81, 47.14), 28.4, True)])
        is None
    )
    assert place_tee([("y-x", (1.0, 1.0), 0.1, False), ("a", (2.0, 2.0), 0.5, True)]) is None


def test_place_tee_on_route_by_leg_ratio() -> None:
    # straight route A (0, 50) -> B (0.2, 50), ~14.3 km; legs 3 and 7 km -> 30 % along
    route = [(0.0, 50.0), (0.1, 50.0), (0.2, 50.0)]
    legs = [("a", route[0], 3.0, True), ("b", route[-1], 7.0, True)]
    placed = place_tee(legs, route=("a", "b", route))
    assert placed is not None
    point, method, _ = placed
    assert method == "tee_on_route"
    assert point[1] == pytest.approx(50.0)
    assert polyline_km([route[0], point]) == pytest.approx(0.3 * polyline_km(route), rel=1e-3)


def test_geometry_overlay_applied() -> None:
    silver, cim = _model()
    rows = [
        ("site", "c", "C", "point", "[8.0, 50.0]", "[8.1, 50.1]", "osm_substation_name", "t"),
        ("line", "l4", "C - X", "route", "[]", "[[8.1, 50.1], [10.0, 48.5], [12.0, 47.0]]",
         "osm_route", "t"),
    ]  # fmt: skip
    overlay = pd.DataFrame(rows, columns=list(OVERLAY_COLUMNS))
    out = build_layers(silver, cim, borders=BORDERS, overlay=overlay)
    subs = out["substations"].set_index("site")
    assert (subs.loc["c"].geometry.x, subs.loc["c"].geometry.y) == pytest.approx((8.1, 50.1))
    assert subs.loc["c", "geometry_source"] == "overlay: osm_substation_name"
    assert subs.loc["c", "display_relocated_m"] == 0
    lines = out["lines"].set_index("eic")
    assert lines.loc["E4", "geometry_source"] == "overlay: osm_route"
    assert lines.loc["E4", "route_quality"] == "mapped"
    assert len(lines.loc["E4"].geometry.coords) == 3
    assert lines.loc["E1", "geometry_source"] == "model"
    assert lines.loc["E4", "route_length_km"] == pytest.approx(
        polyline_km([(8.1, 50.1), (10.0, 48.5), (12.0, 47.0)]), abs=1e-3
    )
    buses = out["buses"].set_index("bus")
    assert buses.loc["c|220", "geometry_source"] == "overlay: osm_substation_name"


def test_section_of_and_route() -> None:
    assert section_of("osm-4706-1-0", "OpenStreetMap export") == "osm-4706"
    assert section_of("curated/Z", "export section osm-14312, lines 577/578") == "osm-14312"
    assert section_of("11TD2L0000001249", "osm") is None
    # a Y-shaped section: the branch between the two ends, not the third arm
    paths = [[[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [2.0, 0.5]], [[1.0, 0.0], [2.0, -0.5]]]
    assert section_route(paths, (2.0, 0.49), (-0.01, 0.0)) == [(2.0, 0.5), (1.0, 0.0), (0.0, 0.0)]
    assert section_route([[[0.0, 0.0], [1.0, 0.0]], [[5.0, 5.0], [6.0, 5.0]]], (0, 0), (6, 5)) == []
    assert section_route(paths, (0.0, 0.0), (0.0, 0.0)) == []


def test_export_section_drawn_on_its_ways() -> None:
    silver, cim = _model()
    seg = cim.ac_line_segment
    seg.loc[seg["mrid"] == "l4", ["eic", "route_source"]] = ["osm-7-0", "OpenStreetMap export"]
    silver["grid_de_osm_sections"] = pd.DataFrame(
        {
            "section": ["osm-7"],
            "paths": [
                [
                    [[8.006, 50.003], [10.0, 48.5]],
                    [[10.0, 48.5], [12.0, 47.0]],
                    [[10.0, 48.5], [11.0, 49.0]],
                ]
            ],
        }
    )
    lines = build_layers(silver, cim, borders=BORDERS)["lines"].set_index("eic")
    assert lines.loc["osm-7-0", "route_quality"] == "mapped"
    assert lines.loc["osm-7-0", "geometry_source"] == "export section osm-7"
    coords = list(lines.loc["osm-7-0"].geometry.coords)
    assert (10.0, 48.5) in coords and (11.0, 49.0) not in coords


def test_station_layout_from_osm_footprints() -> None:
    from shapely.geometry import Point, box

    silver, cim = _model()
    plain = build_layers(silver, cim, borders=BORDERS)
    yard = box(9.997, 51.998, 10.003, 52.002)  # OSM polygon around B (10.0, 52.0)
    osm = gpd.GeoDataFrame(
        {"osm_id": [7], "name": ["UW B"], "voltages": ["380000;220000"]},
        geometry=[yard],
        crs="EPSG:4326",
    )
    out = build_layers(silver, cim, borders=BORDERS, stations=osm)
    fp = out["station_footprints"].set_index("site")
    assert fp.loc["b", "source"] == "osm" and fp.loc["b", "osm_ids"] == "7"
    assert fp.loc["a", "source"] == "box"  # no polygon near A: fixed-size box
    # no circuit lost, every circuit layer keeps its count
    for name in ("tower_lines", "substations", "buses", "transformers"):
        assert len(out[name]) == len(plain[name])
    assert set(out["lines"]["mrid"]) == set(plain["lines"]["mrid"])
    # circuits end on the outline of B (one plain box), not at the station point
    lines = out["lines"].to_crs(3035)
    yard_m = gpd.GeoSeries([yard], crs=4326).to_crs(3035).iat[0]
    ends_b = lines[out["lines"]["station_end"].str.contains("b")]
    assert len(ends_b) >= 3
    for line in ends_b.geometry:
        assert line.intersection(yard_m).length < 1.0
        assert min(yard_m.exterior.distance(Point(c)) for c in line.coords) < 1.0
    # no busbars, links or transformer symbols on the map: the box only
    assert len(out["station_levels"]) == 0
    assert len(out["station_transformers"]) == 0
    assert {"site", "voltage_kv", "part", "mrid"} <= set(out["station_levels"].columns)
    assert fp.loc["b", "max_kv"] == 380.0  # outline colour of the box
    # the two A - B circuits each keep their own end on the outline (no shared port)
    ab = lines[
        out["lines"]["name"].str.startswith("A - B")
        & out["lines"]["station_end"].str.split(";").map(lambda s: "b" in s)
    ]
    assert len(ab) == 2


def test_station_details_popup(tmp_path) -> None:
    import json

    from shapely.geometry import box

    from tranpy.grid_stations import station_details, write_station_details

    pytest.importorskip("pyarrow")  # the station table is parquet
    silver, cim = _model()
    osm = gpd.GeoDataFrame(
        {"osm_id": [7], "name": ["UW B"], "voltages": ["380000;220000"]},
        geometry=[box(9.997, 51.998, 10.003, 52.002)],
        crs="EPSG:4326",
    )
    layers = build_layers(silver, cim, borders=BORDERS, stations=osm)
    folder = tmp_path / "stations"
    folder.mkdir()
    pd.DataFrame({"station": ["b"], "osm_operator": ["TenneT"], "osm_names": ["UW B"]}).to_parquet(
        folder / "stations.parquet"
    )
    pd.DataFrame(
        {
            "check": ["dead_end_node", "dead_end_node"],
            "object": ["b|220", "a|380"],
            "severity": ["warning", "warning"],
            "detail": ["one EHV branch", "x"],
        }
    ).to_csv(folder / "checks.csv", index=False)
    details = station_details(layers, folder)
    b = details["b"]
    assert b["operator"] == "TenneT" and b["osm_names"] == "UW B"
    assert {n["node"] for n in b["nodes"]} == set(
        layers["buses"].loc[layers["buses"]["site"] == "b", "bus"]
    )
    assert len(b["transformers"]) == (layers["transformers"]["site"] == "b").sum() >= 1
    assert all(t["sn_mva"] for t in b["transformers"])
    # every circuit at B once, with the far-end station by name
    at_b = layers["lines"].drop_duplicates("mrid")
    buses_b = set(layers["buses"].loc[layers["buses"]["site"] == "b", "bus"])
    n_b = sum((r.bus1 in buses_b) + (r.bus2 in buses_b) for r in at_b.itertuples())
    assert len(b["circuits"]) == n_b
    assert "A" in {c["far_end"] for c in b["circuits"]}
    assert [f["object"] for f in b["flags"]] == ["b|220"]
    path = write_station_details(layers, tmp_path / "web" / "station_details.json", folder)
    assert json.loads(path.read_text(encoding="utf-8"))["b"]["name"] == b["name"]


def test_station_layout_from_station_table() -> None:
    from shapely.geometry import box

    silver, cim = _model()
    table = gpd.GeoDataFrame(
        {"site": ["b", "c"]},
        geometry=[box(9.997, 51.998, 10.003, 52.002), box(5.0, 45.0, 5.01, 45.01)],
        crs="EPSG:4326",
    )
    out = build_layers(silver, cim, borders=BORDERS, stations=table)
    fp = out["station_footprints"].set_index("site")
    assert fp.loc["b", "source"] == "station table"
    assert fp.loc["c", "source"] == "box"  # table polygon far from the drawn station
    assert out["lines"]["station_end"].str.contains("b").any()


def test_station_table_two_yards_around_the_point() -> None:
    """Point between two yards (> FOOTPRINT_MATCH_M from each): the yards' hull is used."""
    from shapely.geometry import MultiPolygon, Point, box

    from tranpy.grid_stations import station_footprints

    subs = gpd.GeoDataFrame(
        {"site": ["s", "far"], "name": ["S", "F"], "kind": ["substation"] * 2},
        geometry=[Point(10.0, 52.0), Point(11.0, 52.0)],
        crs="EPSG:4326",
    ).assign(voltages_kv=["220/380", "380"])
    yards = MultiPolygon([box(9.998, 51.993, 10.002, 51.995), box(9.998, 52.005, 10.002, 52.007)])
    table = gpd.GeoDataFrame(
        {"site": ["s", "far"]},
        geometry=[yards, box(11.0, 52.006, 11.002, 52.008)],  # far: ~670 m off, not around
        crs="EPSG:4326",
    )
    fp = station_footprints(subs, table).set_index("site")
    assert fp.loc["s", "source"] == "station table"
    _, miny, _, maxy = fp.loc["s"].geometry.bounds
    assert miny < 51.994 and maxy > 52.006  # convex hull over both yards
    assert fp.loc["far", "source"] == "box"


def _network_dir(tmp_path, pts):
    """A two-edge OSM graph (ways 111, 222) with circuits a, b routed s1 -> s2 and c not."""
    import shapely

    edge_geoms = [shapely.LineString([pts[1], pts[2]]), shapely.LineString([pts[2], pts[3]])]
    pd.DataFrame(
        {
            "edge_id": [0, 1],
            "u": [1, 2],
            "v": [2, 3],
            "way_id": [111, 222],
            "virtual": [False, False],
            "geometry": shapely.to_wkb(edge_geoms),
        }
    ).to_parquet(tmp_path / "edges.parquet")
    pd.DataFrame(
        {"node": list(pts), "lon": [p[0] for p in pts.values()], "lat": [52.0] * 3}
    ).to_parquet(tmp_path / "nodes.parquet")
    pd.DataFrame(
        {
            "mrid": ["a", "b", "c"],
            "edges": [[0, 1], [0, 1], []],
            "status": ["routed", "routed", "no_path_strict"],
            "break_lon": [None, None, 10.015],
            "break_lat": [None, None, 52.0],
            "break_gap_m": [None, None, 420.0],
            "override_reason": ["", "", "wrong corridor"],
        }
    ).to_parquet(tmp_path / "circuit_routes.parquet")
    rows = [
        ("a", 0, 0, 111, [pts[1], pts[2]]),
        ("a", 1, 1, 222, [pts[2], pts[3]]),
        ("b", 0, 0, 111, [pts[2], pts[1]]),  # drawn against travel
        ("b", 1, 1, 222, [pts[2], pts[3]]),
    ]
    pd.DataFrame(
        {
            "mrid": [r[0] for r in rows],
            "name": [r[0] for r in rows],
            "kv": 380,
            "seq": [r[1] for r in rows],
            "edge_id": [r[2] for r in rows],
            "way_id": [r[3] for r in rows],
            "kind": "way",
            "geometry": shapely.to_wkb([shapely.LineString(r[4]) for r in rows]),
        }
    ).to_parquet(tmp_path / "circuit_pieces.parquet")
    (tmp_path / "meta.json").write_text('{"model_sha": "abc"}')


def test_network_lines_pieces_touch_boxes_and_dashed_unrouted(tmp_path) -> None:
    """Routed circuits are their exact OSM pieces with an integer slot (no baked offset);
    nothing is drawn inside the end boxes; a route end off its box grows the box (up to
    300 m) or gets a dashed connector and a station_gaps row; circuits without pieces keep
    their old geometry, dashed, centred and with the reason."""
    pytest.importorskip("pyarrow")
    import shapely

    from tranpy.grid_gis import network_lines

    pts = {1: (10.0, 52.0), 2: (10.02, 52.0), 3: (10.04, 52.0)}
    _network_dir(tmp_path, pts)
    old = gpd.GeoDataFrame(
        {
            "mrid": ["a", "b", "c", "c", "h"],
            "name": ["A 1", "A 2", "C", "C", "HVDC"],
            "bus1": ["s1|380"] * 5,
            "bus2": ["s2|380"] * 5,
            "hvdc": [False, False, False, False, True],
            "route_quality": ["mapped", "straight", "mapped", "mapped", "mapped"],
            "geometry_source": ["model"] * 5,
            "status": ["in_operation"] * 5,
            "tower_way": ["111", "111", "9", "8", ""],
            "piece": [0, 0, 0, 1, 0],
            "pieces": [1, 1, 2, 2, 1],
            "label_piece": [True, True, True, False, True],
            "circuit_no": [0, 1, 0, 1, 0],
            "circuits": [2, 2, 2, 2, 1],
        },
        geometry=[
            shapely.LineString([pts[1], pts[3]]),
            shapely.LineString([pts[1], pts[3]]),
            shapely.LineString([pts[1], pts[2]]),
            shapely.LineString([pts[2], pts[3]]),
            shapely.LineString([pts[1], pts[3]]),
        ],
        crs="EPSG:4326",
    )
    # s1 covers the route start; s2 ends 500 m before the route end (beyond the cap)
    s2 = (10.04 + 500 / 68_600, 52.0)
    boxes = gpd.GeoDataFrame(
        {"site": ["s1", "s2"]},
        geometry=[shapely.Point(pts[1]).buffer(0.002), shapely.Point(s2).buffer(0.0003)],
        crs="EPSG:4326",
    )
    res = network_lines(old, boxes, tmp_path)
    out = res["lines"]
    assert set(out["mrid"]) == {"a", "b", "c", "h"}
    assert "offset_baked" not in out
    g = out[out["mrid"].isin(["a", "b"])]
    ways = g[g["piece_kind"] == "way"]
    assert (ways["route_quality"] == "osm_graph").all() and set(ways["tower_way"]) == {"111", "222"}
    assert (ways["circuits"] == 2).all()
    fp = boxes.to_crs("EPSG:3035").set_index("site").geometry
    metric = g.to_crs("EPSG:3035")
    # nothing inside the s1 footprint
    assert metric.geometry.intersection(fp["s1"]).length.max() < 1.0
    # the s2 end is 500 m off: a dashed end connector to the box and a station_gaps row
    ends = g[g["piece_kind"] == "end"]
    assert len(ends) == 2 and (ends["route_quality"] == "unrouted").all()
    gaps = res["station_gaps"]
    assert set(gaps["station"]) == {"s2"} and not gaps["within_cap"].any()
    assert gaps["distance_m"].min() > 300
    c = out[out["mrid"] == "c"]
    assert (c["route_quality"] == "unrouted").all() and (
        c["route_status"] == "no_path_strict"
    ).all()
    assert (c["circuit_no"] == 0).all() and (c["circuits"] == 1).all()
    note = c["route_note"].iloc[0]
    assert "no_path_strict" in note and "10.0150/52.0000" in note and "wrong corridor" in note
    h = out[out["mrid"] == "h"].iloc[0]
    assert h["route_quality"] == "mapped" and h["route_status"] == "hvdc"


def test_clip_chain_removes_the_inside_of_the_boxes() -> None:
    import shapely

    from tranpy.grid_gis import clip_chain

    line = [shapely.LineString([(0, 0), (500, 0)]), shapely.LineString([(1000, 0), (500, 0)])]
    b1 = shapely.Point(0, 0).buffer(100)
    b2 = shapely.Point(1000, 0).buffer(100)
    out, level = clip_chain(line, b1, b2)
    assert level == "box"
    assert out[0].coords[0][0] == pytest.approx(100, abs=6)
    assert out[1].coords[0][0] == pytest.approx(900, abs=6)  # own direction kept
    # leaves the box, comes back past it, leaves again: cut where it last leaves
    hook = [shapely.LineString([(0, 0), (200, 0), (200, 50), (50, 50), (50, 300), (5000, 300)])]
    cut, _ = clip_chain(hook, shapely.Point(0, 0).buffer(100), None)
    assert cut[0].coords[0][0] == pytest.approx(50) and cut[0].coords[0][1] > 50
    # too short to cut: unchanged
    short = [shapely.LineString([(0, 0), (250, 0)])]
    out, level = clip_chain(
        short, shapely.Point(0, 0).buffer(100), shapely.Point(250, 0).buffer(100)
    )
    assert level == "none" and out[0].length == pytest.approx(250)


def test_station_yards_adds_own_operator_and_extension_polygons() -> None:
    import shapely

    from tranpy.grid_gis import station_yards

    def sq(x, y, d):
        return shapely.box(x, y, x + d, y + d)

    m = "EPSG:3035"
    x0, y0 = 4_300_000.0, 3_200_000.0
    stations = gpd.GeoDataFrame(
        {"station": ["a", "b"], "osm_operator": ["TSO", "TSO"], "osm_names": ["Alpha", "Beta"]},
        geometry=[sq(x0, y0, 200), sq(x0 + 5000, y0, 200)],
        crs=m,
    )
    osm = gpd.GeoDataFrame(
        {
            "name": ["Alpha 380", "Alpha", "Erweiterung", "far"],
            "operator": ["TSO", "DSO GmbH", None, "TSO"],
        },
        geometry=[
            sq(x0 + 250, y0, 150),  # own operator, 50 m away: added
            sq(x0 - 200, y0, 150),  # distribution operator: left out
            sq(x0, y0 + 260, 120),  # extension yard without operator: added
            sq(x0 + 2000, y0, 150),  # too far
        ],
        crs=m,
    )
    out = station_yards(stations, osm)
    assert set(out) == {"a"}
    assert out["a"].area == pytest.approx(150 * 150 + 120 * 120, rel=1e-6)


def test_osm_way_lines_and_parallel_findings(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    import shapely

    from tranpy.grid_gis import osm_way_lines, parallel_way_findings

    # two parallel 380 kV ways 100 m apart over 3 km; the model routes 4 circuits on way 1
    # (OSM 2), none on way 2 (OSM 2, ref 305)
    a = [(10.0, 54.0), (10.045, 54.0)]
    b = [(10.0, 54.0009), (10.045, 54.0009)]
    pd.DataFrame(
        {
            "edge_id": [0, 1],
            "u": [1, 3],
            "v": [2, 4],
            "way_id": [1, 2],
            "virtual": [False, False],
            "kind": ["line", "line"],
            "kv": ["380", "380"],
            "ref": ["", "305"],
            "name": ["", "Handewitt"],
            "circuits_tag": [2.0, 2.0],
            "length_m": [3000.0, 3000.0],
            "geometry": shapely.to_wkb([shapely.LineString(a), shapely.LineString(b)]),
        }
    ).to_parquet(tmp_path / "edges.parquet")
    pd.DataFrame(
        {
            "mrid": list("wxyz"),
            "way_id": 1,
            "kind": "way",
            "geometry": shapely.to_wkb([shapely.LineString(a)] * 4),
        }
    ).to_parquet(tmp_path / "circuit_pieces.parquet")
    pd.DataFrame(
        {"edge_id": [0, 1], "way_id": [1, 2], "class": ["model_more", "model_fewer"]}
    ).to_csv(tmp_path / "edge_check.csv", index=False)
    lines = pd.DataFrame(
        {
            "mrid": list("wxyz"),
            "name": ["S - H 1", "S - H 2", "S - H 3", "S - H 4"],
            "eic": ["e1", "e2", "e3", "e4"],
            "voltage_kv": 380,
        }
    )
    box = gpd.GeoSeries([shapely.Point(10.0, 54.0).buffer(0.003)], crs="EPSG:4326").to_crs(3035)
    ways = osm_way_lines(tmp_path, [box.iloc[0]], lines, drawn=set("wxyz"))
    w = ways.set_index("way_id")
    assert w.at[1, "check"] == "model_more" and w.at[1, "model_n"] == 4
    assert w.at[2, "check"] == "no_model" and w.at[2, "ref"] == "305"
    # cut at the box, otherwise the exact OSM geometry
    full = gpd.GeoSeries([shapely.LineString(a)], crs="EPSG:4326").to_crs(3035).iloc[0]
    assert w.at[1, "geometry"].length < full.length
    assert w.at[1, "geometry"].difference(full.buffer(0.5)).length < 0.5
    found = parallel_way_findings(ways)
    shift = found[found["kind"] == "count_shift"]
    assert len(shift) == 1 and shift["way_b"].iloc[0] == 2 and shift["model_n_a"].iloc[0] == 4


def _way_network(tmp_path, ways, pieces, classes) -> None:
    import shapely

    pd.DataFrame(
        {
            "edge_id": list(range(len(ways))),
            "u": [2 * i for i in range(len(ways))],
            "v": [2 * i + 1 for i in range(len(ways))],
            "way_id": [w[0] for w in ways],
            "virtual": False,
            "kind": "line",
            "kv": "380",
            "ref": "",
            "name": "",
            "circuits_tag": [w[2] for w in ways],
            "construction": [w[3] if len(w) > 3 else False for w in ways],
            "length_m": [
                float(
                    gpd.GeoSeries([shapely.LineString(w[1])], crs="EPSG:4326")
                    .to_crs(3035)
                    .length.iloc[0]
                )
                for w in ways
            ],
            "geometry": shapely.to_wkb([shapely.LineString(w[1]) for w in ways]),
        }
    ).to_parquet(tmp_path / "edges.parquet")
    pd.DataFrame(
        {
            "mrid": [p[0] for p in pieces],
            "edge_id": [p[1] for p in pieces],
            "way_id": [ways[p[1]][0] for p in pieces],
            "kind": "way",
        }
    ).to_parquet(tmp_path / "circuit_pieces.parquet")
    pd.DataFrame(
        {"edge_id": list(range(len(ways))), "way_id": [w[0] for w in ways], "class": classes}
    ).to_csv(tmp_path / "edge_check.csv", index=False)


def test_osm_way_lines_split_at_circuit_set_change_short_and_explained(tmp_path) -> None:
    """A way is split where its circuit set changes (a tee inside the way: each section
    carries only its own circuits); a halo section under 150 m becomes ``short``; a reviewed
    difference becomes ``explained`` with its reason; a short way without a model circuit
    touching a yard is left out, as is a construction way without one."""
    pytest.importorskip("pyarrow")
    import shapely

    lon = 10.0
    x = [lon + k * 0.01 for k in range(4)]  # ~0.65 km steps at 54 N
    ways = [
        (7, [(x[0], 54.0), (x[1], 54.0)], 2.0),  # way 7, edge 0: circuits a, b
        (7, [(x[1], 54.0), (x[2], 54.0)], 2.0),  # way 7, edge 1: circuits a, c (tee at x1)
        (8, [(x[2], 54.0), (x[3], 54.0)], 3.0),  # way 8: a only (OSM 3) -> reviewed
        (9, [(x[3], 54.0), (x[3] + 0.0012, 54.0)], 2.0),  # ~80 m, model_fewer -> short
        (10, [(x[0], 54.002), (x[0] + 0.001, 54.002)], 1.0),  # 65 m bay, no model
        (11, [(x[1], 54.01), (x[2], 54.01)], 2.0, True),  # construction, no model
    ]
    pieces = [("a", 0), ("b", 0), ("a", 1), ("c", 1), ("a", 2), ("a", 3)]
    classes = ["match", "match", "model_fewer", "model_fewer", "model_fewer", "model_fewer"]
    _way_network(tmp_path, ways, pieces, classes)
    lines = pd.DataFrame(
        {"mrid": list("abc"), "name": ["A", "B", "C"], "eic": list("123"), "voltage_kv": 380}
    )
    yard = gpd.GeoSeries([shapely.Point(x[0], 54.002).buffer(0.0003)], crs="EPSG:4326")
    yard = yard.to_crs(3035).iloc[0]
    review = pd.DataFrame(
        {"way_id": [8], "osm_n": [3.0], "model_n": [1.0], "reason": ["OSM 3 / model 1: test"]}
    )
    from tranpy.grid_gis import osm_way_lines

    out = osm_way_lines(tmp_path, [], lines, drawn=set("abc"), yard_shapes=[yard], review=review)
    w7 = out[out["way_id"] == 7]
    assert len(w7) == 2 and sorted(w7["model_n"]) == [2, 2]
    assert {tuple(sorted(c["name"] for c in json.loads(m))) for m in w7["model_circuits"]} == {
        ("A", "B"),
        ("A", "C"),
    }
    w = out.set_index("way_id")
    assert w.at[8, "check"] == "explained" and w.at[8, "check_reason"].startswith("OSM 3")
    assert w.at[9, "check"] == "short"
    assert 10 not in w.index and 11 not in w.index


def test_grow_boxes_reach_line_ends_but_never_another_station() -> None:
    import shapely

    from tranpy.grid_gis import grow_boxes

    a = shapely.box(0, 0, 100, 100)
    b = shapely.box(400, 0, 500, 100)
    # a: a way end 120 m east (towards b) and one 100 m north; b's own box is 300 m east
    grown = {"a": [shapely.Point(220, 50), shapely.Point(50, 200)], "b": [shapely.Point(330, 50)]}
    others = {"a": a, "b": b, "tee": shapely.Point(50, 320)}
    out = grow_boxes({"a": a, "b": b}, grown, others, margin=30)
    assert out["a"].contains(shapely.Point(220, 50)) and out["a"].contains(shapely.Point(50, 200))
    assert not out["a"].intersects(b) and not out["a"].contains(others["tee"])
    # b's end point is nearer to a's grown reach but nearer to b's own box: b grows, no overlap
    assert out["b"].contains(shapely.Point(330, 50)) and not out["b"].intersects(a)
    assert out["a"].distance(out["b"]) >= 0
    # a point that would swallow the other station is dropped
    out2 = grow_boxes({"a": a}, {"a": [shapely.Point(480, 50)]}, {"a": a, "b": b}, margin=30)
    assert not out2["a"].intersects(b)
