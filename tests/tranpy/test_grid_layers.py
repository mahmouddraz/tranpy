from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tranpy.german_grid import GermanGrid
from tranpy.grid_layers import (
    attach_plants,
    cim_tables,
    load_shares,
    mrid,
    node_countries,
    plants_from_kraftwerksliste,
)

R = 6378137.0


def _merc(lon: float, lat: float) -> tuple[float, float]:
    return R * np.radians(lon), R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def _grid() -> GermanGrid:
    nodes = pd.DataFrame(
        {
            "node": ["a|380", "b|380", "b|220", "x|380"],
            "site": ["a", "b", "b", "x"],
            "kv": [380, 380, 220, 380],
            "names": [["A"], ["B"], ["B"], ["X"]],
            "tso": ["T1", "T1", "T1", "T1"],
            "kind": ["substation", "substation", "substation", "foreign"],
            "lon": [10.0, 11.0, 11.0, 15.5],
            "lat": [50.0, 50.0, 50.0, 50.0],
            "location_source": ["osm", "verified", "osm", "osm"],
            "osm_bus": [None] * 4,
        }
    )
    lines = pd.DataFrame(
        {
            "eic": ["E1", "E2", "E3"],
            "name": ["A - B", "B - X", "A - X DC"],
            "tso": ["T1"] * 3,
            "node0": ["a|380", "b|380", "a|380"],
            "node1": ["b|380", "x|380", "x|380"],
            "kv": [380.0] * 3,
            "length_km": [70.0, 60.0, 300.0],
            "r_ohm": [2.0, 1.5, 0.1],
            "x_ohm": [18.0, 15.0, 0.1],
            "b_us": [300.0, 250.0, 0.0],
            "imax_a": [2500.0, 2500.0, np.nan],
            "tie_line": [False, True, True],
            "hvdc": [False, False, True],
            "route": [np.array([[10, 50], [11, 50]], float)] * 3,
            "route_source": ["osm"] * 3,
        }
    )
    transformers = pd.DataFrame(
        {
            "eic": ["T1"],
            "name": ["TR B 220/400"],
            "tso": ["T1"],
            "node_hv": ["b|380"],
            "node_lv": ["b|220"],
            "kv_hv": [400.0],
            "kv_lv": [231.0],
            "r_ohm": [0.7],
            "x_ohm": [60.0],
            "b_us": [-1.0],
            "g_us": [0.8],
            "imax_a": [866.0],
            "phase_shifter": [False],
        }
    )
    return GermanGrid(nodes, lines, transformers, pd.DataFrame(), {"test": "synthetic"})


def _plant_list() -> pd.DataFrame:
    rows = []
    for name, fuel, state, mw, status, level, lonlat, mastr in [
        (
            "Coal A",
            "Steinkohle",
            "Bayern",
            800.0,
            "In Betrieb",
            "Höchstspannung",
            (10.05, 50.0),
            "SEE1",
        ),
        ("Gas B", "Erdgas", "Bayern", 200.0, "in Betrieb", "Hochspannung", (11.1, 50.05), "SEE2"),
        (
            "Old",
            "Kernenergie",
            "Bayern",
            1400.0,
            "endgültig stillgelegt 2023",
            "Höchstspannung",
            (10.0, 50.0),
            "SEE3",
        ),
        (
            "Reserve",
            "Erdgas",
            "Bayern",
            300.0,
            "Netzreserve aufgrund § 13b EnWG",
            "Höchstspannung",
            (11.0, 50.0),
            "SEE4",
        ),
        (
            None,
            "Solare Strahlungsenergie",
            "Bayern",
            1000.0,
            "In Betrieb",
            None,
            None,
            "SEE9-Dummy-EEG",
        ),
        (None, "Erdgas", "ohne Zuordnung", 100.0, "In Betrieb", None, None, "SEE9-Dummy-KWK"),
    ]:
        x, y = _merc(*lonlat) if lonlat else (np.nan, np.nan)
        rows.append(
            {
                "MaStR_Nr_der_Stromerzeugungseinheit": mastr,
                "Anzeige_Name_der_Stromerzeugungseinheit": name,
                "Anlagenbetreiber": "Op",
                "Bundesland_der_Einheit": state,
                "Energietr_ger": fuel,
                "Nettonennleistung_elektrische_Wirkleistung_in_MW": mw,
                "Kraftwerksstatus_der_Einheit": status,
                "Spannungsebene": level,
                "Anschlussnetzbetreiber": "TSO",
                "x": x,
                "y": y,
            }
        )
    return pd.DataFrame(rows)


def test_mrid_is_stable() -> None:
    assert mrid("bus", "a|380") == mrid("bus", "a|380")
    assert mrid("bus", "a|380") != mrid("bus", "b|380")


def test_plants_are_normalised() -> None:
    p = plants_from_kraftwerksliste(_plant_list()).set_index("mastr_id")
    assert p.loc["SEE1", "lon"] == pytest.approx(10.05) and p.loc["SEE1", "lat"] == pytest.approx(
        50.0
    )
    assert p.loc["SEE2", "in_operation"]  # lower-case status
    assert p.loc["SEE3", "decommissioned"] and p.loc["SEE4", "reserve"]
    assert p.loc["SEE9-Dummy-EEG", "aggregated"]


def test_attachment_and_loads() -> None:
    grid = _grid()
    plants = plants_from_kraftwerksliste(_plant_list())
    population = pd.DataFrame(
        {"lon": [10.0, 10.1, 11.0], "lat": [50.0, 50.0, 50.0], "population": [100.0, 100.0, 200.0]}
    )
    shares = load_shares(population, grid.nodes)
    assert shares.set_index("node")["share"].to_dict() == {"a|380": 0.5, "b|380": 0.5}
    att = attach_plants(plants, grid.nodes, shares)
    assert "SEE3" not in set(att["mastr_id"])  # decommissioned
    by = att.groupby("mastr_id")
    assert by.get_group("SEE1")["node"].tolist() == ["a|380"]
    assert by.get_group("SEE1")["method"].iloc[0] == "nearest substation"
    assert by.get_group("SEE2")["method"].iloc[0] == "aggregated to nearest substation"
    solar = by.get_group("SEE9-Dummy-EEG")
    assert solar["capacity_mw"].sum() == pytest.approx(1000.0)
    assert set(solar["method"]) == {"state population share"}
    assert set(by.get_group("SEE9-Dummy-KWK")["method"]) == {"national population share"}
    assert set(att.loc[att["method"] != "nearest substation", "data_quality"]) == {"estimated"}

    tables = cim_tables(grid, att, shares, total_load_mw=1000.0)
    assert len(tables.ac_line_segment) == 2  # HVDC left out
    assert "A - X DC" in tables.model_info["hvdc_left_out"]
    assert tables.energy_consumer["p_mw"].sum() == pytest.approx(1000.0)
    gen = tables.synchronous_machine
    assert gen.loc[gen["in_service"], "p_mw"].sum() == pytest.approx(1000.0)
    assert not gen.loc[gen["fuel"].eq("Erdgas") & ~gen["in_service"], "p_mw"].any()
    assert set(tables.external_network_injection["node"]) == {"x|380"}
    bus = set(tables.busbar_section["mrid"])
    for t in (tables.ac_line_segment, tables.power_transformer):
        assert set(t["bus1_mrid"]) | set(t["bus2_mrid"]) <= bus
    assert tables.ac_line_segment["bch_s"].iloc[0] == pytest.approx(300e-6)


def test_node_countries() -> None:
    square = {"type": "Polygon", "coordinates": [[[9, 49], [12, 49], [12, 51], [9, 51], [9, 49]]]}
    borders = {"features": [{"geometry": square, "properties": {"ADM0_A3": "DEU"}}]}
    c = node_countries(_grid().nodes, borders)
    assert c["a|380"] == "DEU" and c["x|380"] == ""


@pytest.mark.requires_powsybl
def test_powsybl_load_flow(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("pypowsybl")
    from tranpy.powsybl_model import export_cgmes, network_from_cim_tables, run_load_flow

    grid = _grid()
    plants = plants_from_kraftwerksliste(_plant_list())
    population = pd.DataFrame({"lon": [10.0, 11.0], "lat": [50.0, 50.0], "population": [1.0, 1.0]})
    shares = load_shares(population, grid.nodes)
    tables = cim_tables(
        grid, attach_plants(plants, grid.nodes, shares), shares, total_load_mw=500.0
    )
    network = network_from_cim_tables(tables)
    assert len(network.get_lines()) == 2 and len(network.get_2_windings_transformers()) == 1
    summary = run_load_flow(network, tables)
    assert summary.status == "CONVERGED"
    assert summary.load_mw == pytest.approx(500.0)
    assert 0.9 < summary.v_min_pu <= summary.v_max_pu < 1.1
    assert 0 < summary.max_line_loading < 1
    archive = export_cgmes(network, tmp_path / "model.zip")
    import zipfile

    import pypowsybl as pp

    names = zipfile.ZipFile(archive).namelist()
    assert sorted(n.split("_")[-1] for n in names) == ["EQ.xml", "SSH.xml", "SV.xml", "TP.xml"]
    again = pp.network.load(str(archive))  # round trip through CGMES
    loads = again.get_loads()
    real = loads[loads["type"] != "FICTITIOUS"]  # CGMES state injections become fictitious loads
    assert len(again.get_lines()) == 2 and len(real) == len(network.get_loads())
