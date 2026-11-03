from __future__ import annotations

import pandas as pd
import pytest

from tranpy.osm_topology import (
    ADD,
    IN_MODEL,
    NOT_OPERATING,
    OPEN_END,
    TWO_VOLTAGE,
    VOLTAGE_CONFLICT,
    _classify,
    added_circuits,
    added_transformers,
    build_sections,
    check_topology,
    long_way_round,
    voltage_levels,
)

pytest.importorskip("shapely")


def _box(lon: float, lat: float) -> str:
    a, b = lon - 0.001, lon + 0.001
    c, d = lat - 0.001, lat + 0.001
    return f"POLYGON(({a} {c},{b} {c},{b} {d},{a} {d},{a} {c}))"


def _stations() -> pd.DataFrame:
    rows = [("1", "A", 10.0, 50.0), ("2", "B", 10.2, 50.0), ("3", "C", 10.4, 50.0)]
    return pd.DataFrame(
        [
            {"id": i, "name": n, "lon": x, "lat": y, "shape": "polygon", "geometry_wkt": _box(x, y)}
            for i, n, x, y in rows
        ]
    )


def _way(way: int, a: float, b: float, voltages: str = "380000", **extra: object) -> dict:
    base = {
        "id": way,
        "voltages": voltages,
        "frequency": "50",
        "geometry_wkt": f"LINESTRING({a} 50,{(a + b) / 2} 50,{b} 50)",
        "status": "in_operation",
        "circuits": "2",
        "ref": None,
        "name": "Ltg",
        "operator": "TSO",
    }
    return {**base, **extra}


def _circuits(route: list[tuple[float, float]] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "eic": "E",
                "kv": 380.0,
                "length_km": 10.0,
                "r_ohm": 0.3,
                "x_ohm": 2.5,
                "b_us": 40.0,
                "imax_a": 2720.0,
                "route_quality": "verified",
                "node0": "x|380",
                "node1": "y|380",
                "route": [{"lon": x, "lat": y} for x, y in route or [(13.0, 53.0), (13.1, 53.0)]],
            }
        ]
    )


def _nodes() -> pd.DataFrame:
    return pd.DataFrame([{"node": "a|380", "kv": 380, "lon": 10.0, "lat": 50.0, "tso": "T"}])


@pytest.mark.parametrize(
    ("text", "levels"),
    [("380000", {380}), ("380000;220000", {220, 380}), ("110000", set()), ("525000", set())],
)
def test_voltage_levels(text: str, levels: set[int]) -> None:
    assert voltage_levels(text) == levels


def test_uncovered_section_becomes_estimated_circuits() -> None:
    ways = pd.DataFrame([_way(1, 10.0, 10.2)])
    sections = build_sections(ways, _stations(), _circuits())
    assert sections["topology"].tolist() == [ADD]
    added, new = added_circuits(sections, _stations(), _circuits(), _nodes())
    assert len(added) == 2
    assert set(added["node0"]) | set(added["node1"]) == {"a|380", "osm/2|380"}
    assert new["node"].tolist() == ["osm/2|380"]
    assert added["data_quality"].eq("estimated").all()
    assert added["x_ohm"].iloc[0] == pytest.approx(0.25 * added["length_km"].iloc[0])
    assert check_topology(sections, added, new, _circuits(), _nodes()) == []


def test_modelled_route_is_not_added() -> None:
    sections = build_sections(
        pd.DataFrame([_way(1, 10.0, 10.2)]), _stations(), _circuits([(10.0, 50.0), (10.2, 50.0)])
    )
    assert sections["topology"].tolist() == [IN_MODEL]
    assert added_circuits(sections, _stations(), _circuits(), _nodes())[0].empty


def test_two_voltage_tag_stays_unconfirmed() -> None:
    ways = pd.DataFrame([_way(1, 10.0, 10.2, "380000;220000")])
    sections = build_sections(ways, _stations(), _circuits())
    assert set(sections["topology"]) == {TWO_VOLTAGE}


def test_open_end_is_not_added() -> None:
    sections = build_sections(pd.DataFrame([_way(1, 10.0, 10.1)]), _stations(), _circuits())
    assert sections["topology"].tolist() == [OPEN_END]


def test_branching_section_gets_a_junction() -> None:
    ways = pd.DataFrame(
        [
            _way(1, 10.0, 10.2),
            _way(2, 10.2, 10.4),
            {**_way(3, 10.2, 10.2), "geometry_wkt": "LINESTRING(10.1 50,10.1 50.2,10.0 50.3)"},
        ]
    )
    stations = pd.concat(
        [
            _stations(),
            pd.DataFrame(
                [
                    {
                        "id": "4",
                        "name": "D",
                        "lon": 10.0,
                        "lat": 50.3,
                        "shape": "polygon",
                        "geometry_wkt": _box(10.0, 50.3),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    sections = build_sections(ways, stations, _circuits())
    three = sections[sections["station_count"] == 3].iloc[0]
    assert three["topology"] == ADD
    added, new = added_circuits(sections, stations, _circuits(), _nodes())
    assert "junction" in set(new["kind"])
    assert added["length_km"].sum() > 0


def test_other_voltage_on_the_route_is_a_conflict() -> None:
    circuits = _circuits([(10.0, 50.0), (10.2, 50.0)]).assign(kv=220.0)
    sections = build_sections(pd.DataFrame([_way(1, 10.0, 10.2)]), _stations(), circuits)
    assert sections["topology"].tolist() == [VOLTAGE_CONFLICT]


def test_tag_of_several_voltages_counts_one_circuit() -> None:
    ways = pd.DataFrame([_way(1, 10.0, 10.2, "380000;110000", circuits="4")])
    sections = build_sections(ways, _stations(), _circuits())
    assert sections["topology"].tolist() == [ADD]
    added = added_circuits(sections, _stations(), _circuits(), _nodes())[0]
    assert len(added) == 1
    assert added["circuit_count_source"].iloc[0].startswith("assumed one")


def test_way_splits_where_the_modelled_route_ends() -> None:
    way = _way(1, 10.0, 10.4)
    way["geometry_wkt"] = "LINESTRING(10 50,10.1 50,10.2 50,10.3 50,10.4 50)"
    stations = _stations().iloc[[0, 2]].reset_index(drop=True)
    circuits = _circuits([(10.0, 50.0), (10.2, 50.0)])
    sections = build_sections(pd.DataFrame([way]), stations, circuits)
    assert len(sections) == 2
    assert set(sections["coverage"]) == {0.0, 1.0}
    assert sections["km"].sum() == pytest.approx(0.63 * 111.2 * 0.4)


def test_section_between_schematic_circuit_stations_is_not_added() -> None:
    circuits = pd.concat(
        [
            _circuits(),
            pd.DataFrame(
                [
                    {
                        "kv": 380.0,
                        "route_quality": "estimated",
                        "node0": "a|380",
                        "node1": "b|380",
                        "length_km": 1.0,
                        "r_ohm": 1.0,
                        "x_ohm": 1.0,
                        "b_us": 1.0,
                        "imax_a": 1.0,
                        "route": [],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    nodes = pd.concat(
        [
            _nodes(),
            pd.DataFrame([{"node": "b|380", "kv": 380, "lon": 10.2, "lat": 50.0, "tso": "T"}]),
        ],
        ignore_index=True,
    ).assign(site=["a", "b"])
    sections = build_sections(pd.DataFrame([_way(1, 10.0, 10.2)]), _stations(), circuits)
    added = added_circuits(sections, _stations(), circuits, nodes)[0]
    assert added.empty
    assert sections["topology"].tolist() == ["matches_schematic_circuit"]


def test_export_transformer_links_unlinked_site() -> None:
    export = pd.DataFrame([{"v_primary": 380000, "v_secondary": 220000, "lon": 10.0, "lat": 50.0}])
    nodes = pd.DataFrame(
        [
            {"node": "a|380", "kv": 380, "lon": 10.0, "lat": 50.0},
            {"node": "a|220", "kv": 220, "lon": 10.0, "lat": 50.0},
        ]
    )
    known = pd.DataFrame(
        [
            {
                "node_hv": "b|380",
                "node_lv": "b|220",
                "kv_hv": 400.0,
                "kv_lv": 231.0,
                "r_ohm": 0.6,
                "x_ohm": 50.0,
                "b_us": 0.0,
                "g_us": 0.0,
                "imax_a": 866.0,
                "phase_shifter": False,
            }
        ]
    )
    added = added_transformers(export, known, nodes)
    assert added["node_hv"].tolist() == ["a|380"]
    assert added["data_quality"].iloc[0] == "estimated"
    linked = known.assign(node_hv="a|380", node_lv="a|220")
    assert added_transformers(export, linked, nodes).empty


def test_open_end_on_a_modelled_line_reaches_its_next_station() -> None:
    shared = _way(1, 10.0, 10.2)
    alone = _way(2, 10.2, 10.3)
    alone["geometry_wkt"] = "LINESTRING(10.1 50,10.1 50.1,10.2 50.2)"
    shared["geometry_wkt"] = "LINESTRING(10 50,10.1 50,10.2 50)"
    stations = pd.DataFrame(
        [
            {"id": i, "name": n, "lon": x, "lat": y, "shape": "polygon", "geometry_wkt": _box(x, y)}
            for i, n, x, y in [
                ("1", "A", 10.0, 50.0),
                ("2", "B", 10.2, 50.0),
                ("5", "E", 10.2, 50.2),
            ]
        ]
    )
    circuits = _circuits([(10.0, 50.0), (10.1, 50.0), (10.2, 50.0)])
    sections = build_sections(pd.DataFrame([shared, alone]), stations, circuits)
    added = sections[sections["topology"].eq(ADD)]
    assert len(added) == 1
    assert added["followed"].iloc[0]
    assert added["extra_km"].iloc[0] > 0
    assert added["station_count"].iloc[0] == 2


def test_short_stretch_along_a_modelled_line_is_a_deviation() -> None:
    shared = _way(1, 10.0, 10.2)
    shared["geometry_wkt"] = "LINESTRING(10 50,10.1 50,10.2 50)"
    alone = _way(2, 10.1, 10.13)
    alone["geometry_wkt"] = "LINESTRING(10.1 50,10.12 50.01,10.13 50.02)"
    stations = pd.concat(
        [
            _stations().iloc[:2],
            pd.DataFrame(
                [
                    {
                        "id": "9",
                        "name": "F",
                        "lon": 10.13,
                        "lat": 50.02,
                        "shape": "polygon",
                        "geometry_wkt": _box(10.13, 50.02),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    circuits = _circuits([(10.0, 50.0), (10.1, 50.0), (10.2, 50.0)])
    sections = build_sections(pd.DataFrame([shared, alone]), stations, circuits)
    assert "route_deviation" in set(sections["topology"])
    assert ADD not in set(sections["topology"])


def test_a_line_not_in_operation_is_never_counted_as_in_the_model() -> None:
    section = {"km": 5.0, "coverage": 1.0, "other_coverage": 1.0, "two_voltage": False}
    assert _classify({**section, "status": "under_construction"}) == NOT_OPERATING


def _oege_case() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A line leaving a long modelled line midway, like the ENERVIE lines at Oege."""
    shared = _way(1, 10.0, 10.6)
    shared["geometry_wkt"] = "LINESTRING(10 50,10.3 50,10.6 50)"
    alone = _way(2, 10.3, 10.3)
    alone["geometry_wkt"] = "LINESTRING(10.3 50,10.3 50.05,10.3 50.1)"
    stations = pd.DataFrame(
        [
            {"id": i, "name": n, "lon": x, "lat": y, "shape": "polygon", "geometry_wkt": _box(x, y)}
            for i, n, x, y in [
                ("1", "A", 10.0, 50.0),
                ("2", "B", 10.6, 50.0),
                ("5", "E", 10.3, 50.1),
            ]
        ]
    )
    circuits = _circuits([(10.0, 50.0), (10.3, 50.0), (10.6, 50.0)])
    sections = build_sections(pd.DataFrame([shared, alone]), stations, circuits)
    return sections, stations, circuits


def test_long_way_round_joins_the_tee_at_the_end_of_the_ways() -> None:
    sections, stations, circuits = _oege_case()
    added = sections[sections["topology"].eq(ADD)]
    assert added["extra_km"].iloc[0] > 20
    nodes = pd.concat(
        [
            _nodes(),
            pd.DataFrame([{"node": "y-t|380", "kv": 380, "lon": 10.3, "lat": 50.005, "tso": "T"}]),
        ],
        ignore_index=True,
    )
    rows, new_nodes = added_circuits(sections, stations, circuits, nodes)
    assert set(rows[["node0", "node1"]].iloc[0]) == {"y-t|380", "osm/5|380"}
    assert rows["length_km"].iloc[0] == pytest.approx(11.12 + 0.56, abs=0.1)
    joined = sections.loc[sections["topology"].eq(ADD), "joined_at"]
    assert joined.tolist() == ["y-t|380"]
    assert not check_topology(sections, rows, new_nodes, circuits, nodes)


def test_long_way_round_without_a_node_nearby_stays_followed() -> None:
    sections, stations, circuits = _oege_case()
    rows, _ = added_circuits(sections, stations, circuits, _nodes())
    assert rows["length_km"].iloc[0] > 30
    assert "joined_at" not in sections


@pytest.mark.parametrize(
    ("km", "section_km", "long"), [(3.0, 20.0, False), (6.0, 50.0, True), (4.0, 10.0, True)]
)
def test_long_way_round(km: float, section_km: float, long: bool) -> None:
    assert long_way_round(km, section_km) is long


def test_tees_at_one_place_go_by_the_station_the_section_reaches() -> None:
    sections, stations, circuits = _oege_case()
    nodes = pd.DataFrame(
        [
            {"node": n, "kv": 380, "lon": 10.3, "lat": 50.005, "tso": "T"}
            for n in ("y-a|380", "y-b|380")
        ]
        + [{"node": "c|380", "kv": 380, "lon": 10.3, "lat": 50.1, "tso": "T"}]
    )
    circuits = pd.concat(
        [circuits, circuits.assign(eic="F", node0="y-b|380", node1="c|380")], ignore_index=True
    )
    rows, _ = added_circuits(sections, stations, circuits, nodes)
    assert set(rows[["node0", "node1"]].iloc[0]) == {"y-b|380", "c|380"}
