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
        assert len(frame) or name == "tower_lines", name  # no tower lines given
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
