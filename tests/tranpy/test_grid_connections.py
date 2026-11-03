"""Connection of generators and load to the 220/380 kV grid."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tranpy.grid_connections import (
    EHV_110,
    ConnectionParameters,
    catchment_shares,
    has_110,
    has_ehv,
    infeed_transformers,
    plant_class,
    rating_mva,
    set_taps,
    soundness_problems,
    transformer_impedance,
    voltage_setpoint_pu,
)
from tranpy_core.errors import DataError


def test_voltage_lists_and_ratings_are_read() -> None:
    assert has_110("380000;110000")
    assert has_ehv("380000;110000")
    assert not has_ehv("110000")
    assert rating_mva("300 MVA") == 300.0
    assert rating_mva("40 MVA") is None
    assert rating_mva(None) is None


@pytest.mark.parametrize(
    ("level", "fuel", "expected"),
    [
        ("Höchstspannung", "Braunkohle", "A"),
        ("Höchstspannung", "Solare Strahlungsenergie", "D"),
        ("Hochspannung", "Erdgas", "B"),
        ("Hochspannung", "Windenergie (Onshore-Anlage)", "C"),
        ("Mittelspannung", "Erdgas", "C"),
        (None, "Erdgas", "C"),
        ("Hochspannung", "Windenergie (Offshore-Anlage)", "D"),
    ],
)
def test_plant_class_follows_the_published_level(level: str, fuel: str, expected: str) -> None:
    assert plant_class(level, fuel) == expected


def test_transformer_impedance_matches_the_per_unit_definition() -> None:
    r, x = transformer_impedance(380.0, 350.0, 0.22, 0.00257)
    assert (r**2 + x**2) ** 0.5 == pytest.approx(0.22 * 380**2 / 350)
    assert r == pytest.approx(0.00257 * 380**2 / 350)
    two = transformer_impedance(380.0, 350.0, 0.22, 0.00257, units=2)
    assert two[1] == pytest.approx(x / 2)
    with pytest.raises(DataError):
        transformer_impedance(380.0, 350.0, 0.0, 0.1)


def test_defaults_are_marked_for_sweeping() -> None:
    assert ConnectionParameters().class_b_min_mw == 25.0
    assert EHV_110[380]["uk"] == 0.22


def _places() -> tuple[pd.DataFrame, pd.DataFrame]:
    points = pd.DataFrame(
        {"lon": [10.0, 10.4], "lat": [50.0, 50.0], "zone": ["z", "z"], "mw": [100.0, 100.0]}
    )
    sites = pd.DataFrame(
        {"site": ["w", "e"], "lon": [10.0, 10.4], "lat": [50.0, 50.0], "zone": "z", "mva": 300.0}
    )
    return points, sites


def test_catchment_goes_to_the_nearest_site_when_capacity_allows() -> None:
    shares = catchment_shares(*_places())
    assert sorted(shares["site"]) == ["e", "w"]
    assert shares["share"].tolist() == pytest.approx([1.0, 1.0])


def test_catchment_respects_transformer_capacity() -> None:
    points, sites = _places()
    points["mw"] = [300.0, 100.0]
    sites["mva"] = [100.0, 300.0]
    shares = catchment_shares(points, sites)
    west = shares[shares["site"].eq("w")]["share"].sum()
    assert west * 300.0 == pytest.approx(100.0 * 1.1 * 400 / 400, rel=1e-3)


def test_catchment_without_a_site_in_the_area_is_an_error() -> None:
    points, sites = _places()
    with pytest.raises(DataError):
        catchment_shares(points, sites.assign(zone="other"))


def test_taps_move_in_steps_inside_their_range() -> None:
    frame = pd.DataFrame(
        {"mrid": ["lumped/tr/a", "lumped/tr/b"], "bus2_mrid": ["x", "y"], "rated_u2_kv": 110.0}
    )

    model = SimpleNamespace(power_transformer=frame)
    out = set_taps(model, {"x": 0.97, "y": 0.5}).power_transformer["rated_u2_kv"].tolist()
    assert out[0] == pytest.approx(110.0 * 1.05)
    assert out[1] == pytest.approx(110.0 * 1.16)


def _sound() -> SimpleNamespace:
    return SimpleNamespace(
        power_transformer=pd.DataFrame(
            {
                "mrid": ["stepup/g", "lumped/tr/h"],
                "bus1_mrid": ["h", "h"],
                "bus2_mrid": ["t", "l"],
                "rated_u1_kv": [380.0, 380.0],
                "rated_u2_kv": [21.0, 110.0],
                "r_ohm": [0.1, 0.1],
                "x_ohm": [10.0, 10.0],
            }
        ),
        busbar_section=pd.DataFrame(
            {"mrid": ["h", "t", "l"], "voltage_level_mrid": ["v380", "v21", "v110"]}
        ),
        voltage_level=pd.DataFrame(
            {"mrid": ["v380", "v21", "v110"], "nominal_voltage_kv": [380.0, 21.0, 110.0]}
        ),
        ac_line_segment=pd.DataFrame(columns=["bus1_mrid", "bus2_mrid"]),
        synchronous_machine=pd.DataFrame(
            {
                "mrid": ["term/g"],
                "bus_mrid": ["t"],
                "p_mw": [80.0],
                "max_p_mw": [100.0],
                "in_service": [True],
                "rating_mva": [100.0],
            }
        ),
        energy_consumer=pd.DataFrame({"bus_mrid": ["l"], "p_mw": [50.0]}),
    )


def test_a_sound_model_has_no_problems() -> None:
    model = _sound()
    assert soundness_problems(model, model, pd.Series([0.1])) == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda m: m.power_transformer.__setitem__("x_ohm", 0.0), "impedance"),
        (lambda m: m.power_transformer.__setitem__("rated_u2_kv", [21.0, 150.0]), "tap range"),
        (lambda m: m.power_transformer.__setitem__("rated_u1_kv", 400.0), "high side"),
        (lambda m: m.synchronous_machine.__setitem__("p_mw", 120.0), "above"),
        (lambda m: m.energy_consumer.__setitem__("bus_mrid", "nowhere"), "unknown"),
    ],
)
def test_faults_are_found(change, message: str) -> None:  # type: ignore[no-untyped-def]
    model = _sound()
    try:
        change(model)
        assert any(message in p for p in soundness_problems(model))
    except KeyError:
        assert message == "unknown"


def test_isolated_power_totals_and_balance_are_found() -> None:
    model = _sound()
    model.power_transformer = model.power_transformer.iloc[:1]
    assert any("off the main" in p for p in soundness_problems(model))
    changed = _sound()
    changed.energy_consumer["p_mw"] = 60.0
    assert any("load changed" in p for p in soundness_problems(changed, _sound()))
    assert any("balance" in p for p in soundness_problems(_sound(), None, pd.Series([5.0])))


def test_connection_tables_label_every_assumed_value() -> None:
    pytest.importorskip("shapely")
    from tranpy.grid_pipeline import connection_tables

    stations = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["Tagged", "Far"],
            "voltages": ["380000;110000", "380000;110000"],
            "geometry_wkt": [
                "POLYGON ((10 50, 10.01 50, 10.01 50.01, 10 50.01, 10 50))",
                "POLYGON ((12 52, 12.01 52, 12.01 52.01, 12 52.01, 12 52))",
            ],
        }
    )
    units = pd.DataFrame(
        {
            "lon": [10.005, 10.006],
            "lat": [50.005, 50.006],
            "v_primary": [380000, 380000],
            "v_secondary": [110000, 110000],
            "rating": ["300 MVA", "300 MVA"],
        }
    )
    buses = pd.DataFrame(
        {"mrid": ["b"], "site": ["s"], "lon": [10.005], "lat": [50.005], "kv": [380.0]}
    )
    tables = connection_tables(stations, units, buses)
    row = tables["grid_de_connection_infeed"].iloc[0]
    assert (row["units"], row["mva"], row["rating_source"]) == (2, 600.0, "export")
    assert tables["grid_de_connection_unmatched"]["name"].tolist() == ["Far"]
    assert (
        tables["grid_de_connection_parameters"]["status"]
        .eq("assumption, uncited, owner to confirm")
        .all()
    )


def test_units_count_from_transformers_even_when_the_station_lacks_the_110_kv_tag() -> None:
    stations = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["both", "assumed"],
            "voltages": ["380000;110000", "380000;110000"],
            "geometry_wkt": [
                "POLYGON((10 50,10.001 50,10.001 50.001,10 50.001,10 50))",
                "POLYGON((12 50,12.001 50,12.001 50.001,12 50.001,12 50))",
            ],
        }
    )
    transformers = pd.DataFrame(
        {
            "v_primary": [380e3, 380e3, 380e3],
            "v_secondary": [110e3, 110e3, 110e3],
            "rating": ["300 MVA", None, None],
            "lon": [10.0005, 10.0005, 11.0],
            "lat": [50.0005, 50.0005, 50.0],
        }
    )
    buses = pd.DataFrame(
        {
            "mrid": ["a", "b", "c"],
            "site": ["a", "b", "c"],
            "lon": [10.0, 11.0, 12.0],
            "lat": [50.0, 50.0, 50.0],
            "kv": [380.0, 380.0, 380.0],
        }
    )
    pytest.importorskip("shapely")
    rows, missed = infeed_transformers(stations, transformers, buses)
    rows = rows.set_index("site")
    assert rows.loc["a", "units"] == 2
    assert rows.loc["a", "mva"] == pytest.approx(600.0)
    # one tagged unit stays one: no n-1 floor
    assert rows.loc["b", "units"] == 1
    assert rows.loc["b", "count_evidence"] == "export"
    assert rows.loc["c", "count_evidence"] == "assumed"
    assert rows.loc["c", "type_units"] == 4
    assert missed.empty


def test_untagged_objects_in_a_station_count_as_units_with_their_rating_tag() -> None:
    square = "POLYGON((10 50,10.001 50,10.001 50.001,10 50.001,10 50))"
    stations = pd.DataFrame(
        {"id": [1], "name": ["site"], "voltages": ["380000;110000"], "geometry_wkt": [square]}
    )
    transformers = pd.DataFrame(
        {
            "v_primary": [380e3, None, None, 110e3],
            "v_secondary": [110e3, None, None, 20e3],
            "rating": [None, "200 MVA", None, None],
            "lon": [10.0005] * 4,
            "lat": [50.0005] * 4,
        }
    )
    buses = pd.DataFrame(
        {"mrid": ["a"], "site": ["a"], "lon": [10.0], "lat": [50.0], "kv": [380.0]}
    )
    pytest.importorskip("shapely")
    rows, _ = infeed_transformers(stations, transformers, buses)
    row = rows.iloc[0]
    # only one untagged unit touches both voltages
    lines = pd.DataFrame(
        {
            "voltages": ["380000", "110000"],
            "geometry_wkt": ["LINESTRING(10.0005 50.0005,10.0008 50.0005)"] * 2,
        }
    )
    touching = transformers.assign(lon=[10.0005, 10.0005, 10.0002, 10.0005])
    joined, _ = infeed_transformers(stations, touching, buses, lines)
    assert joined.iloc[0]["units"] == 2
    assert row["units"] == 3
    assert row["count_evidence"] == "export + untagged, voltage inferred"
    assert row["units_rated"] == 1
    assert row["mva"] == pytest.approx(200.0 + 2 * 350.0)


def test_catchment_does_not_reach_past_the_distance_limit_for_capacity() -> None:
    points = pd.DataFrame({"lon": [10.0], "lat": [50.0], "zone": ["z"], "mw": [500.0]})
    sites = pd.DataFrame(
        {
            "site": ["near", "far"],
            "lon": [10.0, 14.0],
            "lat": [50.0, 50.0],
            "zone": ["z", "z"],
            "mva": [100.0, 1000.0],
        }
    )
    shares = catchment_shares(points, sites)
    assert shares["site"].tolist() == ["near"]


def test_light_hours_hold_a_lower_voltage() -> None:
    assert voltage_setpoint_pu(33000.0) == 1.00
    assert voltage_setpoint_pu(60000.0) == 1.02


def test_researched_site_units_come_first_and_counts_are_never_invented() -> None:
    from tranpy.grid_connections import apply_site_units, calibrate_to_totals

    infeed = pd.DataFrame(
        {
            "site": ["a", "b", "c", "d"],
            "bus_hv": ["A", "B", "C", "D"],
            "kv_hv": [380.0, 380.0, 220.0, 380.0],
            "units": [2, 2, 2, 2],
            "mva": [680.0, 680.0, 400.0, 680.0],
            "count_evidence": ["assumed", "export", "assumed", "export"],
            "rating_source": ["median", "export", "median", "median"],
        }
    )
    research = pd.DataFrame(
        {
            "node": ["A", "B", "C", "D"],
            "units": [3, None, None, 3],
            "unit_mva": [350.0, None, None, 250.0],
            "uk_percent": [19.0, None, None, None],
            "source_type": ["published: a", "osm_export", "planned: tender", "published: d"],
            "source_url": ["u", "e", "t", "v"],
            "notes": ["", "", "", "one unit of 250 MVA confirmed"],
        }
    )
    out = apply_site_units(infeed, research).set_index("site")
    assert out.loc["a", "units"] == 3
    assert out.loc["a", "mva"] == 1050.0
    assert out.loc["a", "uk"] == 0.19
    assert out.loc["b", "mva"] == 680.0
    assert out.loc["c", "units"] == 2
    assert out.loc["c", "mva"] == 400.0
    assert out.loc["d", "mva"] == 250.0 + 2 * 340.0
    scaled = calibrate_to_totals(out.reset_index(), {"A": "x", "B": "x", "C": "x"}, {"x": 2130.0})
    assert scaled["mva"].tolist()[:3] == [1050.0, 680.0, 400.0]
    scaled = calibrate_to_totals(out.reset_index(), {"C": "y"}, {"y": 300.0})
    assert scaled.set_index("site").loc["c", "mva"] == 300.0
    assert scaled.set_index("site").loc["c", "rating_source"].startswith("calibrated")


def _counted() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site": ["a", "b", "c", "d"],
            "bus_hv": ["A", "B", "C", "D"],
            "kv_hv": [380.0, 380.0, 220.0, 220.0],
            "units": [1, 2, 2, 3],
            "units_rated": [1, 1, 0, 0],
            "rated_mva": [300.0, 250.0, 0.0, 0.0],
            "mva": [300.0, 590.0, 600.0, 900.0],
            "count_evidence": ["export", "export", "assumed", "export"],
        }
    )


def test_unit_counts_follow_owner_then_published_then_export_then_default() -> None:
    from tranpy.grid_connections import LIBRARY_TYPES, unit_counts

    owner = pd.DataFrame(
        {
            "site": ["b", "x"],
            "kv_hv": [380, 380],
            "kv_lv": [110, 110],
            "ehv_units": [3, 1],
            "evidence": ["owner OpenInfraMap", "owner"],
            "note": ["3 on boundary", ""],
        }
    )
    published = pd.DataFrame(
        {
            "node": ["B", "D"],
            "units": [5, 4],
            "unit_mva": [None, 200.0],
            "uk_percent": [None, 15.0],
            "source_type": ["published: report", "published: decision"],
            "source_url": ["u", "v"],
        }
    )
    out, missed = unit_counts(_counted(), owner, published, ratings="library")
    out = out.set_index("site")
    # the export count stays as it is, without a floor
    assert out.loc["a", "units"] == 1
    assert out.loc["a", "count_label"] == "export"
    assert out.loc["a", "mva"] == 300.0
    assert out.loc["a", "rating_source"] == "export"
    # owner beats published; one export rating kept, the rest from the library
    assert out.loc["b", "units"] == 3
    assert out.loc["b", "count_label"] == "owner_checked"
    assert out.loc["b", "mva"] == 250.0 + 2 * 160.0
    assert out.loc["b", "rating_source"] == "export + library_match"
    assert out.loc["c", ["units", "count_label", "mva"]].tolist() == [2, "default", 200.0]
    assert out.loc["c", "rating_source"] == "library_match"
    assert out.loc["d", ["units", "count_label", "mva", "uk"]].tolist() == [
        4, "published", 800.0, 0.15]  # fmt: skip
    assert out.loc["c", "uk"] == LIBRARY_TYPES[220]["uk"]
    assert out.loc["c", "g_s"] == pytest.approx(2 * 55e-3 / 220.0**2)
    assert out.loc["c", "b_s"] < 0
    assert missed["site"].tolist() == ["x"]


def test_capped_catchment_spills_load_beyond_the_cap_to_the_next_site() -> None:
    from tranpy.grid_connections import spill_over_cap

    load = np.array([100.0, 50.0])
    share = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    near = np.array([[0, 1, 2], [0, 2, 1]])
    out = spill_over_cap(load, share, near, np.array([90.0, 20.0, 1000.0]))
    held = np.bincount(near.ravel(), (out * load[:, None]).ravel(), minlength=3)
    assert held == pytest.approx([90.0, 20.0, 40.0])
    assert out.sum(axis=1) == pytest.approx([1.0, 1.0])
    # a site at its cap takes nothing more, and load without room stays
    stuck = spill_over_cap(load, share, near, np.array([90.0, 0.0, 0.0]))
    assert stuck == pytest.approx(share)


def test_capped_rule_limits_a_site_to_its_share_of_capacity() -> None:
    points = pd.DataFrame(
        {
            "lon": [10.0, 10.0, 10.3],
            "lat": [50.0, 50.0, 50.0],
            "zone": "z",
            "mw": [200.0, 0.0, 10.0],
            "kind": ["load", "solar", "load"],
        }
    )
    sites = pd.DataFrame(
        {"site": ["w", "e"], "lon": [10.0, 10.3], "lat": [50.0, 50.0], "zone": "z",
         "mva": [100.0, 150.0]}
    )  # fmt: skip

    def west(shares: pd.DataFrame) -> float:
        return float(shares[shares["site"].eq("w") & shares["cell"].eq(shares["cell"].iloc[0])]
                     ["share"].sum() * 200.0)  # fmt: skip

    # the old rule keeps 110% of the area's average utilisation: 92.4 MW on 100 MVA
    assert west(catchment_shares(points, sites)) == pytest.approx(92.4)
    # capped at 80 MVA, the east site has room for 2.4 MW only (120 MVA cap, 117.6 MW held)
    capped = catchment_shares(points, sites, "capped", cap=0.8, load_factor=1.0)
    assert west(capped) == pytest.approx(90.0)


def test_owner_count_at_a_voltage_without_export_units_gets_its_own_row() -> None:
    from tranpy.grid_connections import unit_counts

    owner = pd.DataFrame({"site": ["c"], "kv_hv": [380], "kv_lv": [None], "ehv_units": [1]})
    buses = pd.DataFrame({"mrid": ["C380"], "site": ["c"], "kv": [380.0]})
    out, missed = unit_counts(_counted(), owner, buses=buses, ratings="library")
    row = out.set_index("bus_hv").loc["C380"]
    assert row[["units", "count_label", "mva"]].tolist() == [1, "owner_checked", 160.0]
    # the assumed 220 kV row of the checked site is gone
    assert "C" not in set(out["bus_hv"])
    assert missed.empty


def test_unrated_units_take_the_median_published_rating_unless_library_is_asked() -> None:
    from tranpy.grid_connections import LIBRARY_TYPES, MEDIAN_MATCH, unit_counts

    published = pd.DataFrame(
        {"node": ["D"], "units": [4], "unit_mva": [200.0],
         "source_type": ["published: decision"], "source_url": ["v"]}
    )  # fmt: skip
    out = unit_counts(_counted(), site_units=published)[0].set_index("site")
    lib = unit_counts(_counted(), site_units=published, ratings="library")[0].set_index("site")
    # 220 kV: the published unit of d (200 MVA) fills c; 380 kV: export tags 300 and 250
    assert out.loc["c", ["mva", "rating_source"]].tolist() == [400.0, MEDIAN_MATCH]
    assert out.loc["b", "mva"] == 250.0 + 275.0
    assert lib.loc["c", ["mva", "rating_source"]].tolist() == [200.0, "library_match"]
    # a published rating stays; impedance stays per unit, iron losses scale with the rating
    assert out.loc["d", "mva"] == lib.loc["d", "mva"] == 800.0
    assert out.loc["c", "uk"] == LIBRARY_TYPES[220]["uk"]
    assert out.loc["c", "g_s"] == pytest.approx(2 * lib.loc["c", "g_s"])
    with pytest.raises(ValueError, match="ratings"):
        unit_counts(_counted(), ratings="other")


def test_backfeed_beyond_the_cap_spills_like_load() -> None:
    from tranpy.grid_connections import spill_backfeed

    # cell 0 feeds back 100 MW, cell 1 draws 30 MW at site 0
    net = np.array([100.0, -30.0])
    share = np.array([[1.0, 0.0], [1.0, 0.0]])
    near = np.array([[0, 1], [0, 1]])
    out = spill_backfeed(net, share, near, np.array([40.0, 1000.0]))
    held = np.bincount(near.ravel(), (out * net[:, None]).ravel(), minlength=2)
    # the 30 MW load absorbs backfeed: site 0 keeps 40 MW net, 70 MW gross backfeed
    assert held == pytest.approx([40.0, 30.0])
    assert out[1] == pytest.approx(share[1])
    assert out.sum(axis=1) == pytest.approx([1.0, 1.0])


def test_capped_rule_with_backfeed_moves_surplus_solar() -> None:
    points = pd.DataFrame(
        {
            "lon": [10.0, 10.0, 10.3],
            "lat": [50.0, 50.0, 50.0],
            "zone": "z",
            "mw": [10.0, 400.0, 10.0],
            "kind": ["load", "solar", "load"],
        }
    )
    sites = pd.DataFrame(
        {"site": ["w", "e"], "lon": [10.0, 10.3], "lat": [50.0, 50.0], "zone": "z",
         "mva": [300.0, 200.0]}
    )  # fmt: skip

    def west_net(shares: pd.DataFrame) -> float:
        net = shares["cell"].map({5000250: 390.0, 5200250: -10.0})
        return float((shares["share"] * net)[shares["site"].eq("w")].sum())

    plain = catchment_shares(points, sites, "capped", cap=0.8, load_factor=1.0)
    capped = catchment_shares(points, sites, "capped", cap=0.8, load_factor=1.0, backfeed=True)
    assert west_net(plain) > 240.0
    assert west_net(capped) == pytest.approx(240.0)


def test_median_rating_per_area_with_national_fallback() -> None:
    from tranpy.grid_connections import unit_counts

    rows = [("a", f"A{i}", 1, 400.0) for i in range(5)] + [
        ("c", f"C{i}", 1, 100.0) for i in range(6)
    ]
    rows += [("b", "B0", 1, 250.0), ("a", "A9", 0, 0.0), ("b", "B9", 0, 0.0)]
    infeed = pd.DataFrame(
        {
            "site": [r[1] for r in rows],
            "bus_hv": [r[1] for r in rows],
            "kv_hv": 380.0,
            "units": 1,
            "units_rated": [r[2] for r in rows],
            "rated_mva": [r[3] for r in rows],
            "mva": [r[3] for r in rows],
            "count_evidence": "export",
        }
    )
    areas = {r[1]: r[0] for r in rows}
    out = unit_counts(infeed, areas=areas)[0].set_index("site")
    assert out.loc["A9", "mva"] == pytest.approx(400.0)
    # area b has one sample: the national median of all twelve
    assert out.loc["B9", "mva"] == pytest.approx(175.0)
    national = unit_counts(infeed)[0].set_index("site")
    assert national.loc["A9", "mva"] == pytest.approx(175.0)


def test_backfeed_cap_acts_on_this_hours_embedded_power() -> None:
    """The cap compares this hour's backfeed, not installed capacity, with the station.

    The far site is beyond the catchment distance, so only the spill can reach it.
    """
    from tranpy.grid_connections import connect
    from tranpy.grid_layers import CimTables

    empty = pd.DataFrame()
    tables = CimTables(
        substation=pd.DataFrame({"mrid": ["sw", "se"]}),
        voltage_level=pd.DataFrame(
            {"mrid": ["vw", "ve"], "site": ["w", "e"], "kv": [380, 380],
             "substation_mrid": ["sw", "se"], "nominal_voltage_kv": [380.0, 380.0]}
        ),
        busbar_section=pd.DataFrame(
            {"mrid": ["bw", "be"], "name": ["w|380", "e|380"], "voltage_level_mrid": ["vw", "ve"],
             "lon": [10.0, 13.0], "lat": [50.0, 50.0]}
        ),
        ac_line_segment=empty,
        power_transformer=pd.DataFrame(columns=["mrid"]),
        synchronous_machine=pd.DataFrame(
            {"mrid": ["wind"], "bus_mrid": ["bw"], "node": ["w|380"],
             "fuel": ["Windenergie (Onshore-Anlage)"], "max_p_mw": [3000.0], "min_p_mw": [0.0],
             "p_mw": [600.0], "in_service": [True], "voltage_control": [False],
             "target_v_pu": [1.0], "data_quality": ["x"]}
        ),
        energy_consumer=pd.DataFrame(
            {"mrid": ["l"], "bus_mrid": ["bw"], "node": ["w|380"], "p_mw": [20.0],
             "q_mvar": [0.0]}
        ),
        external_network_injection=empty,
        model_info=empty,
    )  # fmt: skip
    infeed = pd.DataFrame(
        {"site": ["w", "e"], "bus_hv": ["bw", "be"], "kv_hv": [380.0, 380.0],
         "mva": [300.0, 1000.0], "units": [1, 2], "rating_source": ["t", "t"]}
    )  # fmt: skip
    cells = pd.DataFrame(
        {"lon": [10.0], "lat": [50.0], "fuel": ["Windenergie (Onshore-Anlage)"], "mw": [3000.0]}
    )
    people = pd.DataFrame({"lon": [10.0, 13.0], "lat": [50.0, 50.0], "population": [1.0, 1.0]})
    plants = pd.DataFrame(columns=["mastr_id", "voltage_level", "lon", "lat"])
    attachments = pd.DataFrame(columns=["mastr_id", "node", "fuel", "capacity_mw"])
    areas = pd.Series({"w|380": "z", "e|380": "z"})
    par = ConnectionParameters(load_cap=0.8, backfeed_cap=True, load_cap_tan_phi=0.0)
    out, _, _ = connect(tables, infeed, plants, attachments, cells, people, areas, par, "capped")
    machines = out.synchronous_machine
    west = machines.loc[machines["bus_mrid"].eq("lumped/w"), "p_mw"].sum()
    loads = out.energy_consumer
    west_load = loads.loc[loads["bus_mrid"].eq("lumped/w"), "p_mw"].sum()
    assert machines["p_mw"].sum() == pytest.approx(600.0)
    assert west - west_load <= 240.0 + 1e-6


@pytest.mark.parametrize("split", [True, False])
def test_two_level_site_gets_a_110_kv_bus_per_level(split: bool) -> None:
    """Load and embedded generation split by unit MVA; totals stay."""
    from tranpy.grid_connections import connect
    from tranpy.grid_layers import CimTables

    empty = pd.DataFrame()
    tables = CimTables(
        substation=pd.DataFrame({"mrid": ["s"]}),
        voltage_level=pd.DataFrame(
            {"mrid": ["v4", "v2"], "site": ["r", "r"], "kv": [380, 220],
             "substation_mrid": ["s", "s"], "nominal_voltage_kv": [380.0, 220.0]}
        ),
        busbar_section=pd.DataFrame(
            {"mrid": ["b4", "b2"], "name": ["r|380", "r|220"], "voltage_level_mrid": ["v4", "v2"],
             "lon": [10.0, 10.0], "lat": [50.0, 50.0]}
        ),
        ac_line_segment=empty,
        power_transformer=pd.DataFrame(columns=["mrid"]),
        synchronous_machine=pd.DataFrame(
            {"mrid": ["pv"], "bus_mrid": ["b4"], "node": ["r|380"],
             "fuel": ["Solare Strahlungsenergie"], "max_p_mw": [500.0], "min_p_mw": [0.0],
             "p_mw": [250.0], "in_service": [True], "voltage_control": [False],
             "target_v_pu": [1.0], "data_quality": ["x"]}
        ),
        energy_consumer=pd.DataFrame(
            {"mrid": ["l"], "bus_mrid": ["b4"], "node": ["r|380"], "p_mw": [100.0],
             "q_mvar": [20.0]}
        ),
        external_network_injection=empty,
        model_info=empty,
    )  # fmt: skip
    infeed = pd.DataFrame(
        {"site": ["r", "r"], "bus_hv": ["b4", "b2"], "kv_hv": [380.0, 220.0],
         "mva": [300.0, 200.0], "units": [1, 1], "rating_source": ["t", "t"]}
    )  # fmt: skip
    cells = pd.DataFrame(
        {"lon": [10.0], "lat": [50.0], "fuel": ["Solare Strahlungsenergie"], "mw": [500.0]}
    )
    people = pd.DataFrame({"lon": [10.0], "lat": [50.0], "population": [1.0]})
    plants = pd.DataFrame(columns=["mastr_id", "voltage_level", "lon", "lat"])
    attachments = pd.DataFrame(columns=["mastr_id", "node", "fuel", "capacity_mw"])
    areas = pd.Series({"r|380": "z", "r|220": "z"})
    par = ConnectionParameters(split_levels=split)
    out, ext, _ = connect(tables, infeed, plants, attachments, cells, people, areas, par)
    units = out.power_transformer.set_index("bus1_mrid")["bus2_mrid"]
    loads, machines = out.energy_consumer, out.synchronous_machine
    assert loads["p_mw"].sum() == pytest.approx(100.0)
    assert machines["p_mw"].sum() == pytest.approx(250.0)
    if not split:
        assert units["b4"] == units["b2"] == "lumped/r"
        return
    assert units["b4"] == "lumped/r" and units["b2"] == "lumped/r/220"
    assert ext["r/220|110"] == "z"
    at = loads.groupby("bus_mrid")["p_mw"].sum()
    assert at["lumped/r"] == pytest.approx(60.0) and at["lumped/r/220"] == pytest.approx(40.0)
    pv = machines.groupby("bus_mrid")["p_mw"].sum()
    assert pv["lumped/r/220"] == pytest.approx(100.0)
