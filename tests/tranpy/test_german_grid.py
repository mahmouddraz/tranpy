from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tranpy.german_grid import (
    _clean_lines,
    _Issues,
    _merge_double_reports,
    _OsmNetwork,
    _transformer_site,
    _unify_reported_tees,
    _words,
    site_key,
)
from tranpy.grid_data import CoreGridModel


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Brunsbüttel", "Brunsbuettel"),
        ("Hamburg/N", "Hamburg/Nord"),
        ("Doerpen West", "Doerpen/W"),
        ("Großkrotzenburg", "Grosskrotzenburg"),
        ("Eisenhüttenstadt (Abzweig)", "Eisenhuettenstadt abzweig"),
    ],
)
def test_site_key_unifies_spellings(a: str, b: str) -> None:
    assert site_key(a) == site_key(b)


def test_site_key_keeps_operator_suffix() -> None:
    # TenneT's Hamburg/N and 50Hertz's Hamburg/N 50H are different substations
    assert site_key("Hamburg/N 50H") != site_key("Hamburg/N")


@pytest.mark.parametrize(
    ("name", "site"),
    [
        ("TR Eula 220/400 401", "Eula"),
        ("TR Gronau TR 441 E    ", "Gronau"),
        ("TR / 380-220kV / Daxlanden / Trafo 401", "Daxlanden"),
        ("PST Diele T441", "Diele"),
        ("PST Krempermarsch T441a", "Krempermarsch"),
        ("TR Hamburg/Nord 220/400 405", "Hamburg/Nord"),
    ],
)
def test_transformer_site(name: str, site: str) -> None:
    assert _transformer_site(name) == site


def test_words_ignore_generic_terms() -> None:
    assert _words("Umspannwerk Karlsruhe-Daxlanden 380 kV") == {"karlsruhe", "daxlanden"}


def _lines(rows: list[tuple[str, str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"eic": e, "tso": t, "node0": a, "node1": b, "x_ohm": x, "r_ohm": x / 10,
             "b_us": 100.0, "length_km": 10.0, "name": f"{a}-{b}"}
            for e, t, a, b, x in rows
        ]
    )  # fmt: skip


def test_double_reports_keep_both() -> None:
    issues = _Issues()
    lines = _lines(
        [
            ("E1", "A", "p|380", "q|380", 3.55),
            ("E1", "B", "q|380", "p|380", 3.54),
            ("E3", "A", "p|380", "r|380", 5.0),  # one operator: kept as it is
        ]
    )
    out = _merge_double_reports(lines, issues).set_index("eic")
    assert sorted(out.index) == ["E1", "E3"]
    assert out.at["E1", "x_ohm"] == 3.55
    assert out.at["E1", "other_tso"] == "B"
    assert out.at["E1", "other_x_ohm"] == 3.54
    assert np.isnan(out.at["E1", "length_km"])
    assert list(issues.frame()["check"]) == ["double report"]


def test_differently_named_tees_are_unified() -> None:
    issues = _Issues()
    lines = _lines(
        [
            ("E1", "A", "y-one|380", "s|380", 6.26),
            ("E1", "B", "s|380", "y-two|380", 6.40),
            ("E2", "A", "y-two|380", "t|380", 3.0),
            ("E3", "A", "u|380", "v|380", 2.0),
            ("E3", "B", "v|380", "y-w|380", 2.0),  # substation vs tee: only reported
        ]
    )
    out = _unify_reported_tees(lines, issues)
    assert set(out["node0"]) | set(out["node1"]) >= {"y-one|380"}
    assert "y-two|380" not in set(out["node0"]) | set(out["node1"])
    assert list(issues.frame()["check"]) == ["tee name", "review"]


def test_parallel_osm_lines_are_not_added() -> None:
    buses = pd.DataFrame(
        {"bus_id": ["a", "b", "c"], "x": [10.0, 10.5, 11.0], "y": [50.0, 50.0, 50.0],
         "voltage": [380, 380, 380]}
    )  # fmt: skip
    route = np.array([[10.0, 50.0], [10.5, 50.0]])
    lines = pd.DataFrame(
        {"bus0": ["a", "a", "b"], "bus1": ["b", "b", "c"], "length": [36000.0, 36500.0, 36000.0],
         "route": [route, route, route + np.array([0.5, 0.0])]}
    )  # fmt: skip
    substations = pd.DataFrame(
        {"name": ["Umspannwerk Alpha"], "levels": [{380}], "lon": [10.0], "lat": [50.0]}
    )
    net = _OsmNetwork(buses, lines, substations)
    found = net.route(0, 2, 72.0, 0.15)
    assert found is not None and len(found) == 4  # 36 + 36 km, not 36 + 36.5 + 36
    assert net.route(0, 2, 120.0, 0.15) is None
    assert net.locate("alpha", 380, np.array([10.01, 50.0]), 5.0) == (0, "osm")
    assert net.locate("y-alpha", 380, np.array([10.01, 50.0]), 5.0) is None


def test_reports_of_two_operators_reach_the_merge() -> None:
    columns = ["name", "eic", "tso", "from_substation", "to_substation", "kv", "r_ohm", "x_ohm",
               "b_us", "length_km", "imax_a", "tie_line"]  # fmt: skip
    core = CoreGridModel(
        lines=pd.DataFrame(
            [
                ["A - B", "E1", "50HERTZ", "A", "B", 380.0, 0.4, 7.0, 130.0, 25.0, 3600.0, True],
                ["B - A", "E1", "TENNETGMBH", "B", "A", 380.0, 0.5, 7.1, 130.0, 30.0, 3600.0, True],
                ["A - B", "E1", "50HERTZ", "A", "B", 380.0, 0.4, 7.0, 130.0, 25.0, 3600.0, True],
            ],
            columns=columns,
        ),
        transformers=pd.DataFrame(),
        routes={},
        release="test",
    )
    issues = _Issues()
    out = _clean_lines(core, {}, issues)
    assert len(out) == 1
    assert out.iloc[0]["other_x_ohm"] == 7.1
    assert list(issues.frame()["check"]).count("duplicate") == 1


def test_half_share_report_is_doubled() -> None:
    from tranpy.german_grid import _reconcile_ties

    lines = pd.DataFrame(
        [{"name": "A - B", "eic": "E1", "tso": "TENNETGMBH", "node0": "a|380", "node1": "b|380",
          "kv": 380.0, "r_ohm": 0.4, "x_ohm": 6.5, "b_us": 105.0, "length_km": 47.0,
          "tie_line": True, "hvdc": False, "other_tso": None, "other_r_ohm": np.nan,
          "other_x_ohm": np.nan, "other_b_us": np.nan}]
    )  # fmt: skip
    nodes = pd.DataFrame(
        {"node": ["a|380", "b|380"], "lon": [11.0, 11.64], "lat": [52.2, 52.27],
         "location_source": ["osm", "osm"]}
    )  # fmt: skip
    issues = _Issues()
    _reconcile_ties(lines, lines.iloc[0:0], nodes, issues)
    assert lines.at[0, "x_ohm"] == pytest.approx(13.0)
    assert lines.at[0, "b_us"] == pytest.approx(52.5)
    assert "half-share" in issues.frame()["detail"].iloc[-1]
