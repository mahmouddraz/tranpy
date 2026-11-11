from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from tranpy import dispatch
from tranpy.dispatch import (
    FUEL_TYPES,
    HourData,
    dispatch_hour,
    hourly,
    hourly_pumping,
    select_hours,
)
from tranpy.grid_layers import CimTables


def _tables() -> CimTables:
    empty = pd.DataFrame()
    return CimTables(
        substation=empty,
        voltage_level=empty,
        busbar_section=empty,
        ac_line_segment=pd.DataFrame(
            {
                "bus1_mrid": ["bx1", "bx2"],
                "bus2_mrid": ["b1", "b2"],
                "rated_current_a": [1000.0, 3000.0],
                "nominal_voltage_kv": [380.0, 380.0],
            }
        ),
        power_transformer=empty,
        synchronous_machine=pd.DataFrame(
            {
                "node": ["a", "a", "b", "b"],
                "fuel": ["Erdgas", "Erdgas", "Erdgas", "Solare Strahlungsenergie"],
                "max_p_mw": [100.0, 300.0, 100.0, 50.0],
                "p_mw": [1.0, 1.0, 1.0, 1.0],
                "in_service": [True, True, True, True],
            }
        ),
        energy_consumer=pd.DataFrame(
            {"node": ["a", "a", "b"], "p_mw": [10.0, 30.0, 20.0], "q_mvar": [1.0, 3.0, 2.0]}
        ),
        external_network_injection=pd.DataFrame(
            {"node": ["x1", "x2"], "bus_mrid": ["bx1", "bx2"], "p_mw": [0.0, 0.0]}
        ),
    )


_NODES = pd.DataFrame({"node": ["x1", "x2"], "country": ["FRA", "FRA"]})
_AREAS = pd.Series({"a": "A", "b": "B"})


def _hour(gas_a: float = 200.0, solar_b: float = 0.0) -> HourData:
    return HourData(
        pd.Timestamp("2025-06-01", tz="UTC"),
        pd.Series(
            {("A", "B04"): gas_a, ("A", "B16"): 0.0, ("B", "B04"): 50.0, ("B", "B16"): solar_b}
        ),
        pd.Series({"A": 80.0, "B": 40.0}),
        pd.Series({"FR": 400.0, "SE4": 25.0}),
    )


def test_generation_follows_capacity_within_area() -> None:
    gen = dispatch_hour(_tables(), _NODES, _AREAS, _hour()).tables.synchronous_machine
    assert gen["p_mw"].tolist() == pytest.approx([50.0, 150.0, 50.0, 0.0])
    assert gen["in_service"].tolist() == [True, True, True, False]


def test_overflow_moves_to_other_areas_then_is_reported() -> None:
    result = dispatch_hour(_tables(), _NODES, _AREAS, _hour(gas_a=450.0))
    assert result.tables.synchronous_machine["p_mw"].iloc[:3].sum() == pytest.approx(500.0)
    assert result.unallocated_mw.sum() == pytest.approx(0.0)
    over = dispatch_hour(_tables(), _NODES, _AREAS, _hour(gas_a=500.0))
    assert over.unallocated_mw.sum() == pytest.approx(50.0)


def test_load_matches_area_totals_and_keeps_power_factor() -> None:
    load = dispatch_hour(_tables(), _NODES, _AREAS, _hour()).tables.energy_consumer
    assert load["p_mw"].tolist() == pytest.approx([20.0, 60.0, 40.0])
    assert (load["q_mvar"] / load["p_mw"]).tolist() == pytest.approx([0.1, 0.1, 0.1])


def test_exports_split_by_line_capacity_and_unplaced_reported() -> None:
    result = dispatch_hour(_tables(), _NODES, _AREAS, _hour())
    assert result.tables.external_network_injection["p_mw"].tolist() == pytest.approx(
        [-100.0, -300.0]
    )
    assert result.unplaced_exports_mw.to_dict() == {"SE4": 25.0}


def test_hvdc_zone_flow_is_imposed_at_the_german_converter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(dispatch.HVDC_CONVERTERS, "SE4", "b")
    result = dispatch_hour(_tables(), _NODES, _AREAS, _hour())
    load = result.tables.energy_consumer
    row = load[load["mrid"].eq("hvdc/SE4")].iloc[0]
    assert (row["node"], row["p_mw"], row["q_mvar"]) == ("b", 25.0, 0.0)
    assert result.unplaced_exports_mw.empty


def test_source_table_is_not_changed() -> None:
    tables = _tables()
    dispatch_hour(tables, _NODES, _AREAS, _hour())
    assert tables.synchronous_machine["p_mw"].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_hourly_and_selection() -> None:
    frame = pd.DataFrame(
        {
            "psr_type": ["B16", "B16", "B19"],
            "time": ["2025-01-01T10:00Z", "2025-01-01T10:30Z", "2025-01-01T11:00Z"],
            "mw": [10.0, 20.0, 5.0],
        }
    )
    table = hourly(frame)
    assert table.loc[pd.Timestamp("2025-01-01T10:00Z"), "B16"] == pytest.approx(15.0)
    load = pd.Series([1.0, 2.0], index=table.index)
    hours = select_hours(table, load)
    assert hours["high_solar"] == table.index[0]
    assert hours["high_wind"] == table.index[1]
    assert hours["high_load"] == table.index[1]


def test_every_model_fuel_is_mapped() -> None:
    assert "Windenergie (Onshore-Anlage)" in FUEL_TYPES


def test_pumping_is_load_at_the_storage_plants_not_generation() -> None:
    frame = pd.DataFrame(
        {
            "psr_type": ["B10", "B10"],
            "consumption": [False, True],
            "time": ["2025-01-01T00:00Z", "2025-01-01T00:00Z"],
            "mw": [4.0, 1000.0],
        }
    )
    assert hourly(frame)["B10"].iloc[0] == 4.0
    assert hourly_pumping(frame).iloc[0] == 1000.0

    tables = _tables()
    machines = tables.synchronous_machine.assign(
        fuel=["Erdgas", "Pumpspeicher", "Erdgas", "Pumpspeicher"],
        max_p_mw=[100.0, 300.0, 100.0, 100.0],
    )
    hour = HourData(
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Series(dtype=float),
        pd.Series({"A": 100.0, "B": 50.0}),
        pd.Series(dtype=float),
        pd.Series({"A": 300.0, "C": 100.0}),
    )
    result = dispatch_hour(
        replace(tables, synchronous_machine=machines), _NODES, _AREAS, hour
    ).tables.energy_consumer
    assert result["p_mw"].sum() == pytest.approx(150.0 + 400.0)
    assert result.groupby("node")["p_mw"].sum()["a"] == pytest.approx(100.0 + 300.0 + 75.0)


def test_feed_in_is_separate_labelled_and_prefers_incomplete_coverage() -> None:
    gen = pd.DataFrame(
        {
            "mrid": ["a", "b"],
            "node": ["n1", "n2"],
            "fuel": ["Erdgas", "Erdgas"],
            "p_mw": [10.0, 10.0],
            "max_p_mw": [100.0, 100.0],
        }
    )
    tables = replace(_tables(), synchronous_machine=gen)
    out, placed = dispatch.with_unreported_feed_in(
        tables, pd.Series({"n1": "Amprion", "n2": "TenneT"}), 50.0
    )
    rows = out.synchronous_machine.iloc[2:]
    assert placed == 50.0
    assert rows["mrid"].tolist() == ["feed_in/a"]
    assert rows["data_quality"].iloc[0] == dispatch.FEED_IN_LABEL
    assert out.synchronous_machine["p_mw"].iloc[:2].tolist() == [10.0, 10.0]
    assert dispatch.with_unreported_feed_in(tables, pd.Series(dtype=str), -5.0)[1] == 0.0


def test_losses_come_off_published_load_only() -> None:
    load = pd.DataFrame(
        {
            "mrid": ["l", "pump/x", "hvdc/NO2"],
            "p_mw": [100.0, 30.0, 20.0],
            "q_mvar": [10.0, 0.0, 0.0],
        }
    )
    out = dispatch.without_losses(_tables(), 10.0, load).energy_consumer
    assert out["p_mw"].tolist() == [90.0, 30.0, 20.0]
    assert out["q_mvar"].tolist() == [9.0, 0.0, 0.0]


def test_domestic_load_leaves_pumping_and_hvdc_out_and_adds_losses_back() -> None:
    load = pd.DataFrame(
        {
            "mrid": ["l", "pump/x", "hvdc/NO2"],
            "p_mw": [100.0, 30.0, 20.0],
            "q_mvar": [10.0, 0.0, 0.0],
        }
    )
    tables = dispatch.without_losses(_tables(), 10.0, load)
    assert dispatch.domestic_load_mw(tables, 10.0) == 100.0


def test_line_ratings_follow_season_and_hour() -> None:
    from tranpy.dispatch import hourly_line_ratings, rating_weight

    berlin = "Europe/Berlin"
    assert rating_weight(pd.Timestamp("2026-01-22 11:00", tz=berlin)) == 0.0
    assert rating_weight(pd.Timestamp("2026-07-15 13:00", tz=berlin)) == 1.0
    assert rating_weight(pd.Timestamp("2026-07-15 02:00", tz=berlin)) == 0.0
    assert rating_weight(pd.Timestamp("2026-04-15 13:00", tz=berlin)) == pytest.approx(1 / 3)
    lines = pd.DataFrame({"eic": ["a", "b"], "rated_current_a": [800.0, 1000.0]})
    circuits = pd.DataFrame(
        {
            "eic": ["a", "b"],
            "rating_basis": ["dlr_min", "fixed"],
            "dlr_min_a": [800.0, None],
            "dlr_max_a": [1100.0, None],
        }
    )
    may = pd.Timestamp("2026-05-10 12:00", tz=berlin)
    seasonal = hourly_line_ratings(lines, circuits, may)
    assert seasonal["rated_current_a"].tolist() == pytest.approx([900.0, 1000.0])
    low = hourly_line_ratings(lines, circuits, may, basis="dlr_min")
    assert low["rated_current_a"].tolist() == [800.0, 1000.0]
    assert lines["rated_current_a"].tolist() == [800.0, 1000.0]
    copied = lines.assign(eic=[None, "b"], rating_eic=["a", None])
    assert hourly_line_ratings(copied, circuits, may)["rated_current_a"].iloc[0] == 900.0
    with pytest.raises(ValueError, match="unknown rating basis"):
        hourly_line_ratings(lines, circuits, may, basis="max")


def test_border_weights_ignore_lines_not_in_operation() -> None:
    tables = _tables()
    lines = pd.concat(
        [
            tables.ac_line_segment.assign(status="in_operation"),
            pd.DataFrame(
                {
                    "bus1_mrid": ["bx1"],
                    "bus2_mrid": ["b1"],
                    "rated_current_a": [5000.0],
                    "nominal_voltage_kv": [380.0],
                    "status": ["decommissioned"],
                }
            ),
        ],
        ignore_index=True,
    )
    weights = dispatch._border_weights(
        replace(tables, ac_line_segment=lines), pd.Series(["x1", "x2"])
    )
    assert weights.tolist() == pytest.approx([380_000.0, 1_140_000.0])


def test_border_split_follows_measured_tso_segments_and_keeps_the_total() -> None:
    """Per-TSO border split: measured segments (opposite signs allowed) plus capacity rest."""
    from tranpy.dispatch import border_split

    # Hradec: 50Hertz lines 3300 MVA + TenneT 1300 MVA; Prestice: TenneT 1500 MVA
    weights = pd.DataFrame(
        {"50Hertz": [3300.0, 0.0], "TenneT": [1300.0, 1500.0]}, index=["hradec", "prestice"]
    )
    flows = pd.Series({("CZ", "50Hertz"): 1326.0, ("CZ", "TenneT"): -259.0})
    split = border_split(1067.0, weights, flows, "CZ")
    assert split.sum() == pytest.approx(1067.0)
    tenne_t = -259.0
    assert split["prestice"] == pytest.approx(tenne_t * 1500.0 / 2800.0)
    assert split["hradec"] == pytest.approx(1326.0 + tenne_t * 1300.0 / 2800.0)
    # without measured segments: plain capacity split
    plain = border_split(1067.0, weights, pd.Series(dtype=float), "CZ")
    assert plain["hradec"] == pytest.approx(1067.0 * 4600.0 / 6100.0)
    # one measured segment only: capacity split (a single TSO border carries the whole zone)
    single = border_split(1067.0, weights, pd.Series({("CZ", "TenneT"): 500.0}), "CZ")
    pd.testing.assert_series_equal(single, plain)


def test_border_split_missing_tso_value_falls_back_to_capacity_split() -> None:
    """A segment without a published A11 value (NaN) is missing, never 0 MW."""
    from tranpy.dispatch import border_split

    weights = pd.DataFrame(
        {"TenneT": [3000.0, 0.0], "Amprion": [0.0, 2000.0]}, index=["meeden", "maasbracht"]
    )
    plain = border_split(-4000.0, weights, pd.Series(dtype=float), "NL")
    missing = pd.Series({("NL", "TenneT"): float("nan"), ("NL", "Amprion"): -1500.0})
    pd.testing.assert_series_equal(border_split(-4000.0, weights, missing, "NL"), plain)
    zero = pd.Series({("NL", "TenneT"): 0.0, ("NL", "Amprion"): -1500.0})
    # a published 0 MW is a measured value: Amprion -1500 plus its capacity share of the rest
    assert border_split(-4000.0, weights, zero, "NL")["maasbracht"] == pytest.approx(-2500.0)


def test_segment_flow_marks_unpublished_hours_missing() -> None:
    """fetch_tso_border_flows: an hour missing in either direction is NaN, not 0 MW."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "fetch_tso_border_flows.py"
    spec = importlib.util.spec_from_file_location("fetch_tso_border_flows", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hours = pd.Series(pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"], utc=True))
    out = pd.Series([100.0], index=pd.DatetimeIndex(hours[:1]))
    back = pd.Series([30.0], index=pd.DatetimeIndex(hours[:1]))
    flow = module.segment_flow(out, back, hours)
    assert flow.iloc[0] == pytest.approx(70.0)
    assert pd.isna(flow.iloc[1])  # neither direction published: missing, not 0 MW
    one_way = module.segment_flow(pd.Series(dtype=float), back, hours)
    assert one_way.iloc[0] == pytest.approx(-30.0) and pd.isna(one_way.iloc[1])
