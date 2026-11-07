"""Tower lines and per-tower-line circuit offsets (display only)."""

from __future__ import annotations

import math

import pytest

gpd = pytest.importorskip("geopandas")
geom = pytest.importorskip("shapely.geometry")

from tranpy.grid_towers import (  # noqa: E402
    split_by_towers,
    tower_circuits_220_380,
)

# ~0.0146 deg lon = 1 km at 52 N
TRUNK = [(8.0, 52.0), (8.1, 52.0)]  # shared tower line, ~6.9 km
BRANCH = [(8.1, 52.0), (8.1, 52.05)]  # where circuit B leaves (northwards)
SPUR = [(8.1, 52.0), (8.2, 52.0)]  # where circuit A continues east


def _towers():
    return gpd.GeoDataFrame(
        {
            "osm_id": [1, 2, 3],
            "name": ["trunk", "branch", "spur"],
            "ref": ["4132", "", ""],
            "operator": ["", "", ""],
            "voltages": ["380000;220000", "220000", "380000"],
            "max_kv": [380.0, 220.0, 380.0],
            "mixed_voltage": [False, False, False],
            "circuits_tag": [2.0, 1.0, 2.0],
            "cables": ["6", "3", "6"],
            "location": ["overhead"] * 3,
            "construction": [False] * 3,
            "disused": [False] * 3,
        },
        geometry=[geom.LineString(c) for c in (TRUNK, BRANCH, SPUR)],
        crs="EPSG:4326",
    )


def _lines():
    a = geom.LineString([TRUNK[0], TRUNK[1], SPUR[1]])
    # B drawn in the opposite direction, 20 m north of the trunk
    b = geom.LineString([BRANCH[1], (8.1, 52.00018), (8.0, 52.00018)])
    c = geom.LineString([(9.0, 50.0), (9.1, 50.0)])  # on no tower line
    return gpd.GeoDataFrame(
        {
            "mrid": ["a", "b", "c"],
            "name": ["A 380", "B 220", "C"],
            "eic": ["EA", "EB", "EC"],
            "voltage_kv": [380.0, 220.0, 380.0],
            "hvdc": [False, False, False],
            "route_quality": ["verified", "mapped", "mapped"],
            "circuit_no": [0, 0, 0],
            "circuits": [1, 1, 1],
        },
        geometry=[a, b, c],
        crs="EPSG:4326",
    )


def test_tower_circuit_count_from_tags() -> None:
    assert tower_circuits_220_380("380000", 2, "6") == 2
    assert tower_circuits_220_380("380000", None, "6") == 2
    assert tower_circuits_220_380("380000;110000", None, "6;6") == 2
    assert tower_circuits_220_380("380000;220000;110000", None, "6;3;6") == 3
    assert math.isnan(tower_circuits_220_380("380000;110000", 4, "12"))  # tags do not say


def test_circuits_side_by_side_on_shared_tower_line() -> None:
    pieces, towers = split_by_towers(_lines(), _towers())
    on_trunk = pieces[pieces["tower_way"] == "1"].set_index("mrid")
    assert sorted(on_trunk["circuit_no"]) == [0, 1]
    assert set(on_trunk["circuits"]) == {2}
    assert on_trunk.loc["a", "circuit_no"] == 0  # higher voltage first
    # both pieces run the same way (east), so the offsets put them side by side
    for mrid in "ab":
        xs = [p[0] for p in on_trunk.loc[mrid].geometry.coords]
        assert xs[-1] > xs[0]
    # where they branch, each circuit is alone on its tower line
    a_spur = pieces[(pieces["mrid"] == "a") & (pieces["tower_way"] == "3")]
    b_branch = pieces[(pieces["mrid"] == "b") & (pieces["tower_way"] == "2")]
    assert a_spur["circuits"].tolist() == [1] and b_branch["circuits"].tolist() == [1]
    # pieces of a circuit cover its route; one label piece per circuit
    assert pieces.groupby("mrid")["label_piece"].sum().tolist() == [1, 1, 1]
    a_len = pieces[pieces["mrid"] == "a"].to_crs(3035).length.sum()
    assert a_len == pytest.approx(_lines().to_crs(3035).length.iloc[0], rel=1e-3)
    # a circuit on no tower line keeps one piece and its corridor slot
    c = pieces[pieces["mrid"] == "c"]
    assert len(c) == 1 and c["tower_way"].iat[0] == ""

    t = towers.set_index("osm_id")
    assert t.loc[1, "model_circuits"] == 2 and t.loc[1, "mismatch"] == ""
    assert t.loc[1, "model_names"] == "A 380; B 220"
    assert t.loc[2, "model_eics"] == "EB" and t.loc[2, "mismatch"] == ""
    assert t.loc[3, "model_circuits"] == 1 and t.loc[3, "mismatch"] == "model_fewer"
