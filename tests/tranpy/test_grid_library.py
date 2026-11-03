"""Standard values from the pandapower library and checks against them."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tranpy.grid_library import (
    LINE_TYPES,
    TRANSFORMER_TYPES,
    bundle_rule,
    correct_placeholder_ratings,
    correct_stub_ratings,
    length_ledger,
    line_plausibility,
    line_values,
    match_transformer,
    resolve_flagged,
    route_km,
    transformer_uk,
)


def _line(**values):
    base = {
        "mrid": "a",
        "name": "a",
        "nominal_voltage_kv": 380.0,
        "length_km": 100.0,
        "r_ohm": 3.0,
        "x_ohm": 25.0,
        "bch_s": 4e-6 * 100,
        "rated_current_a": 2700.0,
        "tie_line": False,
        "route_quality": "verified",
        "data_quality": "published",
        "route": [{"lon": 10.0, "lat": 50.0}, {"lon": 10.0, "lat": 50.0 + 100 / 111.2}],
    }
    return pd.DataFrame([base | values])


def test_values_match_the_library() -> None:
    pp = pytest.importorskip("pandapower")
    net = pp.create_empty_network()
    lines = pp.available_std_types(net, "line")
    for line in LINE_TYPES:
        row = lines.loc[line.name]
        assert (row.r_ohm_per_km, row.x_ohm_per_km) == pytest.approx((line.r, line.x), abs=1e-3)
        assert (row.c_nf_per_km, row.max_i_ka * 1000) == pytest.approx((line.c, line.i))
    trafos = pp.available_std_types(net, "trafo")
    for kind in TRANSFORMER_TYPES:
        row = trafos.loc[kind.name]
        assert (row.sn_mva, row.vk_percent / 100) == pytest.approx((kind.mva, kind.uk))


def test_bundle_scales_resistance_and_rating_but_not_reactance() -> None:
    line, bundle = bundle_rule(2880.0, 380.0)
    assert (line.i, bundle) == (960.0, 3)
    values = line_values(line, bundle, 10.0)
    assert values["rated_current_a"] == 2880.0
    assert values["r_ohm"] == pytest.approx(line.r / 3 * 10)
    assert values["x_ohm"] == pytest.approx(line.x * 10)


def test_transformer_count_keeps_the_site_capacity() -> None:
    kind, units = match_transformer(380.0, 700.0)
    assert (kind.mva, units) == (160.0, 5)
    assert match_transformer(220.0, 100.0)[1] == 1


def test_route_length() -> None:
    assert route_km(None) != route_km(None)
    assert route_km([{"lon": 10.0, "lat": 50.0}, {"lon": 10.0, "lat": 51.0}]) == pytest.approx(
        111.2
    )


def test_a_consistent_circuit_is_confirmed() -> None:
    assert length_ledger(_line())["length_class"].tolist() == ["confirmed"]


def test_tie_line_share_gets_a_proposal_only() -> None:
    frame = _line(length_km=33.8, x_ohm=43.6, tie_line=True)
    row = length_ledger(frame).iloc[0]
    assert (row.length_class, row.length_km) == ("flagged", 33.8)
    assert row.proposed_km == pytest.approx(43.6 / 0.2515)
    assert "tie-line" in row.cause


def test_placeholder_reactance_and_stub_are_told_apart() -> None:
    assert length_ledger(_line(length_km=26.0, x_ohm=0.05)).cause[0] == "placeholder reactance"
    assert length_ledger(_line(length_km=0.02, x_ohm=0.05)).length_class[0] == "unconfirmed"


def test_plausibility_flags_values_outside_the_band() -> None:
    assert line_plausibility(_line()).empty
    flags = line_plausibility(_line(rated_current_a=150.0))
    assert flags["quantity"].tolist() == ["rating"]
    assert flags["factor"][0] == pytest.approx(960 / 150)


def test_uk_from_reactance_and_rating() -> None:
    table = pd.DataFrame(
        {
            "mrid": ["t"],
            "name": ["t"],
            "rated_u1_kv": [380.0],
            "rated_u2_kv": [110.0],
            "phase_shifter": [False],
            "x_ohm": [0.12 * 380**2 / 300],
            "rated_current_a": [300e3 / 380 / np.sqrt(3)],
        }
    )
    assert transformer_uk(table)["uk"][0] == pytest.approx(0.12)


@dataclass
class _Replaceable:
    ac_line_segment: pd.DataFrame
    busbar_section: pd.DataFrame


_BUSES = pd.DataFrame({"mrid": ["n", "s"], "lon": [10.0, 10.0], "lat": [50.0, 50.2]})


def test_a_placeholder_between_two_yards_of_one_site_keeps_its_reactance() -> None:
    lines = pd.DataFrame(
        {
            "mrid": ["y"],
            "name": ["yard"],
            "data_quality": ["published"],
            "length_km": [58.0],
            "x_ohm": [0.1],
            "bus1_mrid": ["n"],
            "bus2_mrid": ["n2"],
        }
    )
    ledger = pd.DataFrame(
        {
            "mrid": ["y"],
            "cause": ["placeholder reactance"],
            "typical_x": [0.25],
            "length_km": [58.0],
            "proposed_km": [float("nan")],
        }
    )
    buses = pd.DataFrame({"mrid": ["n", "n2"], "lon": [10.0, 10.0], "lat": [50.0, 50.0]})
    final, overrides = resolve_flagged(_Replaceable(lines, buses), ledger)
    out = final.ac_line_segment.iloc[0]
    assert out["x_ohm"] == 0.1
    assert out["length_km"] == 0.4
    assert out["data_quality"] == "published"
    assert overrides["field"].tolist() == ["length_km"]


def test_flags_resolve_with_the_previous_value_kept() -> None:
    lines = pd.DataFrame(
        {
            "mrid": ["a", "b"],
            "name": ["placeholder", "tie"],
            "data_quality": ["published", "published"],
            "length_km": [20.0, 10.0],
            "x_ohm": [0.05, 15.0],
            "bus1_mrid": ["n", "n"],
            "bus2_mrid": ["s", "s"],
        }
    )
    ledger = pd.DataFrame(
        {
            "mrid": ["a", "b"],
            "cause": ["placeholder reactance", "tie-line share: length covers one operator"],
            "typical_x": [0.25, 0.25],
            "length_km": [20.0, 10.0],
            "proposed_km": [float("nan"), 60.0],
        }
    )
    tables = SimpleNamespace(ac_line_segment=lines)
    final, overrides = resolve_flagged(_Replaceable(lines, _BUSES), ledger)
    out = final.ac_line_segment.set_index("mrid")
    assert out.loc["a", "x_ohm"] == 5.0
    assert out.loc["a", "data_quality"] == "library_match"
    assert out.loc["b", "length_km"] == 60.0
    assert out.loc["b", "x_ohm"] == 15.0
    assert out.loc["b", "data_quality"] == "published"
    assert overrides["previous"].tolist() == [0.05, 10.0]
    assert tables.ac_line_segment["x_ohm"].tolist() == [0.05, 15.0]


def test_a_low_rated_stub_takes_its_circuit_and_keeps_the_published_value() -> None:
    lines = pd.DataFrame(
        {
            "mrid": ["stub", "long1", "long2", "fine"],
            "name": ["stub", "long1", "long2", "fine"],
            "bus1_mrid": ["plant", "tee", "tee", "tee2"],
            "bus2_mrid": ["tee", "a", "b", "x"],
            "nominal_voltage_kv": [380.0] * 4,
            "length_km": [0.02, 10.0, 20.0, 0.1],
            "rated_current_a": [150.0, 2216.0, 2400.0, 150.0],
            "data_quality": ["published"] * 4,
        }
    )
    bars = pd.DataFrame({"mrid": ["tee", "tee2"], "name": ["y-a#1|380", "y-b#1|380"]})

    @dataclass(frozen=True)
    class Tables:
        ac_line_segment: pd.DataFrame
        busbar_section: pd.DataFrame

    final, overrides = correct_stub_ratings(Tables(lines, bars))
    out = final.ac_line_segment.set_index("mrid")
    assert out.loc["stub", "rated_current_a"] == 2216.0
    assert out.loc["stub", "data_quality"] == "sibling_rating"
    assert out.loc["fine", "rated_current_a"] == 150.0  # no sibling: the value stays
    assert overrides["previous"].tolist() == [150.0]


def test_a_placeholder_rating_takes_a_library_bundle() -> None:
    lines = pd.DataFrame(
        {
            "mrid": ["a", "b"],
            "name": ["a", "b"],
            "nominal_voltage_kv": [220.0, 220.0],
            "rated_current_a": [9999.0, 1000.0],
            "data_quality": ["published", "published"],
        }
    )
    fixed, records = correct_placeholder_ratings(_Replaceable(lines, _BUSES))
    assert fixed.ac_line_segment["rated_current_a"].tolist() == [3450.0, 1000.0]
    assert records["previous"].tolist() == [9999.0]
