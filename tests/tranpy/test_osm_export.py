from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tranpy.osm_export import export_ways, latest_export, read_hv_layers, status, tower_line_number

gpd = pytest.importorskip("geopandas")
pytest.importorskip("pyogrio")
geometry = pytest.importorskip("shapely.geometry")


@pytest.mark.parametrize(
    ("ref", "number"),
    [("2409 M136", "2409"), ("2409 M 143", "2409"), ("6", None), (None, None)],
)
def test_tower_line_number(ref: str | None, number: str | None) -> None:
    assert tower_line_number(ref) == number


def test_status_prefers_construction() -> None:
    frame = pd.DataFrame({"construction": [True, False, False], "disused": [False, True, False]})
    assert status(frame).tolist() == ["under_construction", "out_of_service", "in_operation"]


def test_latest_export_takes_newest_snapshot(tmp_path: Path) -> None:
    assert latest_export(tmp_path) is None
    for day in ("2026-10-31", "2026-12-01"):
        (tmp_path / day).mkdir()
        (tmp_path / day / "DEU.gpkg").touch()
    assert latest_export(tmp_path) == tmp_path / "2026-12-01" / "DEU.gpkg"


def test_read_hv_layers_keeps_only_the_transmission_grid(tmp_path: Path) -> None:
    path = tmp_path / "DEU.gpkg"
    line, poly, point = geometry.LineString, geometry.box, geometry.Point
    tables = {
        "power_line": ([{"max_voltage": 380000.0, "construction": False, "disused": False},
                        {"max_voltage": 110000.0, "construction": False, "disused": False}],
                       [line([(7, 50), (7.01, 50)]), line([(9, 50), (9.01, 50)])]),
        "power_tower": ([{"ref": "2409 M1"}, {"ref": "9"}], [point(7, 50), point(9, 50)]),
        "power_substation_point": ([{"max_voltage": 380000.0, "construction": False}],
                                   [point(7, 50)]),
        "power_substation_polygon": ([{"max_voltage": 110000.0, "construction": False}],
                                     [poly(9, 50, 9.01, 50.01)]),
        "power_transformer": ([{"v_primary": 380000.0}, {"v_primary": 110000.0}],
                              [point(7, 50), point(9, 50)]),
        "power_switch": ([{"ref": "a"}, {"ref": "b"}], [point(7, 50), point(9, 50)]),
    }  # fmt: skip
    for layer, (rows, shapes) in tables.items():
        gpd.GeoDataFrame(rows, geometry=shapes, crs=4326).to_file(path, layer=layer)
    out = read_hv_layers(path)
    assert {name: len(table) for name, table in out.items()} == {
        "lines": 1, "towers": 1, "substations": 1, "transformers": 1, "switches": 1,
        "infeed_stations": 1, "infeed_transformers": 1, "infeed_lines": 2,
    }  # fmt: skip
    assert out["towers"]["line_number"].tolist() == ["2409"]
    assert out["lines"]["status"].tolist() == ["in_operation"]
    assert out["lines"]["geometry_wkt"].iloc[0].startswith("LINESTRING")


def test_export_ways_share_vertices_and_keep_only_foreign_downloads() -> None:
    def row(way: int, wkt: str, state: str = "in_operation") -> dict[str, object]:
        return {"id": way, "voltages": "380000", "status": state, "geometry_wkt": wkt}

    lines = pd.DataFrame(
        [
            row(1, "LINESTRING(10 50,10.1 50)"),
            row(2, "LINESTRING(10.1 50,10.2 50)", "out_of_service"),
        ]
    )
    point = {"lon": 9.0, "lat": 47.0}
    download = {
        1: {"id": 1, "geometry": [point, point], "tags": {}},
        3: {"id": 3, "geometry": [point, point], "tags": {}},
        4: {"id": 4, "geometry": [point, point], "tags": {}},
    }
    ways = export_ways(lines, download, lambda lon, lat: True)
    assert ways[1]["nodes"][-1] == ways[2]["nodes"][0]
    assert ways[1]["geometry"][0]["lon"] == 10.0
    assert "disused:power" in ways[2]["tags"]
    assert set(ways) == {1, 2, 3, 4}
    assert set(export_ways(lines, download, lambda lon, lat: False)) == {1, 2}
