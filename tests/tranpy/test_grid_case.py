from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from tranpy.simulation.grid_case import (
    BORDER_SHORT_CIRCUIT,
    CE_FCR_FULL_HZ,
    DYNAMIC_VALUES,
    ENTSOE_INERTIA,
    MACHINE_VALUES,
    NEIGHBOUR_INERTIA,
    _main_component,
    border_shares,
    border_short_circuit_gva,
    build_connected_case,
    build_grid_case,
    continental_europe_equivalent,
    entsoe_kinetic_energy_mws,
    fuel_class,
    line_parameters,
)
from tranpy_core.errors import DataError


@pytest.mark.parametrize(
    ("fuel", "group"),
    [
        ("Windenergie (Onshore-Anlage)", "inverter"),
        ("Solare Strahlungsenergie", "inverter"),
        ("Pumpspeicher", "hydro"),
        ("Braunkohle", "thermal"),
    ],
)
def test_fuel_class(fuel: str, group: str) -> None:
    assert fuel_class(fuel) == group


def test_line_parameters_per_unit() -> None:
    r, x = line_parameters(14.44, 144.4, 380.0)
    assert r == pytest.approx(0.01)
    assert x == pytest.approx(0.1)


def test_main_component() -> None:
    assert _main_component(["a", "b", "c"], [("a", "b")]) == {"a", "b"}


def test_every_estimate_has_a_source() -> None:
    assert DYNAMIC_VALUES["source"].str.len().gt(0).all()


def _tables() -> SimpleNamespace:
    bus = pd.DataFrame(
        {
            "mrid": ["a", "b"],
            "name": ["A", "B"],
            "voltage_level_mrid": ["v", "v"],
            "lon": [10.0, 10.5],
            "lat": [50.0, 50.0],
        }
    )
    return SimpleNamespace(
        voltage_level=pd.DataFrame({"mrid": ["v"], "nominal_voltage_kv": [380.0]}),
        busbar_section=bus,
        ac_line_segment=pd.DataFrame(
            {
                "bus1_mrid": ["a"],
                "bus2_mrid": ["b"],
                "r_ohm": [1.0],
                "x_ohm": [10.0],
                "bch_s": [1e-5],
                "status": ["in_operation"],
            }
        ),
        power_transformer=pd.DataFrame(columns=["bus1_mrid", "bus2_mrid"]),
        synchronous_machine=pd.DataFrame(
            {
                "mrid": ["g", "w"],
                "bus_mrid": ["a", "a"],
                "fuel": ["Erdgas", "Windenergie (Onshore-Anlage)"],
                "p_mw": [200.0, 50.0],
            }
        ),
        external_network_injection=pd.DataFrame(columns=["bus_mrid", "p_mw"]),
        energy_consumer=pd.DataFrame({"bus_mrid": ["b"], "p_mw": [240.0], "q_mvar": [20.0]}),
    )


@pytest.mark.filterwarnings("ignore")
def test_case_keeps_inverters_out_of_the_machines(tmp_path) -> None:  # type: ignore[no-untyped-def]
    andes = pytest.importorskip("andes")
    path = build_grid_case(_tables(), tmp_path / "case.xlsx", inverter_model="pq")
    assert path.with_suffix(".dynamic_values.csv").exists()
    system = andes.load(str(path), setup=True, no_output=True, default_config=True)
    assert system.GENCLS.n == 1
    assert system.PFlow.run()
    assert system.GENCLS.Sn.v[0] == pytest.approx(250.0)


def _connected() -> SimpleNamespace:
    t = _tables()
    t.busbar_section = pd.concat(
        [
            t.busbar_section,
            pd.DataFrame({"mrid": ["x"], "name": ["X"], "voltage_level_mrid": ["v"]}),
        ],
    )
    t.ac_line_segment = pd.concat(
        [
            t.ac_line_segment,
            pd.DataFrame(
                {
                    "bus1_mrid": ["b"],
                    "bus2_mrid": ["x"],
                    "r_ohm": [1.0],
                    "x_ohm": [10.0],
                    "bch_s": [0.0],
                    "status": ["in_operation"],
                }
            ),
        ]
    )
    t.synchronous_machine = t.synchronous_machine.assign(rating_mva=[300.0, None])
    return t


def test_machine_values_keep_the_genrou_order() -> None:
    v = MACHINE_VALUES
    assert (v["xd"] > v["xd1"]).all() and (v["xd1"] >= v["xd2"]).all()
    assert (v["td10"] > v["td20"]).all()


@pytest.mark.filterwarnings("ignore")
def test_connected_case_keeps_the_solved_point_flat(tmp_path) -> None:  # type: ignore[no-untyped-def]
    andes = pytest.importorskip("andes")
    path = build_connected_case(
        _connected(),
        {"a": 1.02, "b": 1.0, "x": 1.0},
        {"a": (200.0, 30.0)},
        {"a": (50.0, 0.0)},
        {"x": -10.0},
        tmp_path / "case.xlsx",
    )
    system = andes.load(str(path), setup=True, no_output=True, default_config=True)
    assert system.GENROU.n == 1 and system.SEXS.n == 1 and system.TGOV1.n == 1
    assert system.GENCLS.n == 1  # the neighbour
    assert system.GENROU.Sn.v[0] == pytest.approx(300.0)
    assert system.PFlow.run()
    system.TDS.config.tf = 2.0
    system.TDS.config.no_tqdm = 1
    assert system.TDS.run()
    omega = system.dae.ts.x[:, list(system.GENROU.omega.a)]
    assert abs(omega - 1.0).max() < 1e-6


@pytest.mark.filterwarnings("ignore")
def test_cases_run_every_device_at_50_hz(tmp_path) -> None:  # type: ignore[no-untyped-def]
    andes = pytest.importorskip("andes")
    paths = [
        build_grid_case(_tables(), tmp_path / "grid.xlsx", inverter_model="pq"),
        build_connected_case(
            _connected(),
            {"a": 1.02, "b": 1.0, "x": 1.0},
            {"a": (200.0, 30.0)},
            {"a": (50.0, 0.0)},
            {"x": -10.0},
            tmp_path / "connected.xlsx",
        ),
    ]
    for path in paths:
        system = andes.load(str(path), setup=True, no_output=True, default_config=True)
        for model in system.models.values():
            if model.n and "fn" in model.params:
                assert set(model.params["fn"].v) == {50.0}, model.class_name


# ANDES writes its empty model tables with a pandas call deprecated upstream
@pytest.mark.filterwarnings("ignore:The inplace keyword:DeprecationWarning")
# pandas 3 raises Pandas4Warning (a DeprecationWarning); named by base class so the filter also loads under pandas 2
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_connected_case_turns_small_machines_into_injections(tmp_path) -> None:  # type: ignore[no-untyped-def]
    andes = pytest.importorskip("andes")
    path = build_connected_case(
        _connected(),
        {"a": 1.02, "b": 1.0, "x": 1.0},
        {"a": (200.0, 30.0)},
        {},
        {"x": -10.0},
        tmp_path / "case.xlsx",
        min_machine_mva=500.0,
    )
    system = andes.load(str(path), setup=True, no_output=True, default_config=True)
    assert system.GENROU.n == 0
    assert system.PQ.p0.v[list(system.PQ.idx.v).index("S0")] == pytest.approx(-2.0)
    machines = pd.read_csv(path.with_suffix(".machines.csv"))
    assert machines["small"].tolist() == [True]
    assert system.PFlow.run()


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(("governor", "model"), [("hygov", "HYGOV"), ("tgov1", "TGOV1")])
def test_connected_case_gives_hydro_units_a_hydro_governor(tmp_path, governor, model) -> None:  # type: ignore[no-untyped-def]
    andes = pytest.importorskip("andes")
    t = _connected()
    t.synchronous_machine = t.synchronous_machine.assign(
        fuel=["Wasser", "Windenergie (Onshore-Anlage)"]
    )
    path = build_connected_case(
        t,
        {"a": 1.02, "b": 1.0, "x": 1.0},
        {"a": (200.0, 30.0)},
        {"a": (50.0, 0.0)},
        {"x": -10.0},
        tmp_path / "case.xlsx",
        hydro_governor=governor,
    )
    system = andes.load(str(path), setup=True, no_output=True, default_config=True)
    assert getattr(system, model).n == 1
    assert system.HYGOV.n + system.TGOV1.n == 1
    assert system.PFlow.run()
    system.TDS.config.tf = 2.0
    system.TDS.config.no_tqdm = 1
    assert system.TDS.run()
    omega = system.dae.ts.x[:, list(system.GENROU.omega.a)]
    assert abs(omega - 1.0).max() < 1e-6
    assert pd.read_csv(path.with_suffix(".machines.csv"))["group"].tolist() == ["hydro"]


@pytest.mark.parametrize("inertia", ["low", "central", "high"])
def test_continental_europe_equivalent_follows_the_entsoe_values(inertia) -> None:  # type: ignore[no-untyped-def]
    eq = continental_europe_equivalent(50_000.0, inertia)
    ce_load = 50_000.0 / 0.19
    assert eq.kinetic_energy_mws == pytest.approx(
        {"low": 5, "central": 10, "high": 17}[inertia] * ce_load / 2
    )
    assert 3.0 <= eq.h_s <= 6.0
    assert eq.fcr_max_mw / eq.fcr_mw_per_hz == pytest.approx(CE_FCR_FULL_HZ)
    assert eq.t_valve_s + eq.t_lag_s * 3 <= 30.0  # 95 % of the FCR within 30 s


@pytest.mark.filterwarnings("ignore")
def test_neighbour_equivalent_holds_the_target_inertia_and_stays_flat(tmp_path) -> None:  # type: ignore[no-untyped-def]
    andes = pytest.importorskip("andes")
    eq = continental_europe_equivalent(50_000.0)
    path = build_connected_case(
        _connected(),
        {"a": 1.02, "b": 1.0, "x": 1.0},
        {"a": (200.0, 30.0)},
        {"a": (50.0, 0.0)},
        {"x": -10.0},
        tmp_path / "case.xlsx",
        neighbour_equivalent=eq,
    )
    system = andes.load(str(path), setup=False, no_output=True, default_config=True)
    neighbour = system.GENCLS.as_df()
    german = system.GENROU.as_df()
    kinetic = (neighbour["M"] / 2 * neighbour["Sn"]).sum() + (german["M"] / 2 * german["Sn"]).sum()
    assert kinetic == pytest.approx(eq.kinetic_energy_mws)
    assert ((neighbour["M"] / 2).between(3.0, 6.0)).all()
    assert (neighbour["D"] * neighbour["Sn"] / 50.0).sum() == pytest.approx(eq.damping_mw_per_hz)
    governors = system.TGOV1.as_df()
    mine = governors[governors["syn"].isin(neighbour["idx"])]
    assert len(mine) == len(neighbour)
    assert (mine["Tn"] / (50.0 * mine["R"])).sum() == pytest.approx(eq.fcr_mw_per_hz)
    system.setup()
    for model in system.models.values():
        if model.n and "fn" in model.params:
            assert set(model.params["fn"].v) == {50.0}, model.class_name
    assert system.PFlow.run()
    system.TDS.config.tf = 20.0
    system.TDS.config.no_tqdm = 1
    assert system.TDS.run()
    omega = system.dae.ts.x[:, list(system.GENROU.omega.a) + list(system.GENCLS.omega.a)]
    assert abs(omega - 1.0).max() < 1e-6


def test_border_short_circuit_table_is_sourced_and_within_the_breaker_bound() -> None:
    table = pd.read_csv(BORDER_SHORT_CIRCUIT)
    assert (
        table["source"].str.len().gt(0).all()
        and table["label"].isin({"published", "estimate"}).all()
    )
    bound = 3**0.5 * table["nominal_kv"] * 1.1 * 63 / 1000  # GVA at 63 kA and Um = 1.1 Un
    assert (table["sk_gva"] <= bound).all()
    vigy = table.set_index("node")["sk_gva"].to_dict()["vigy|380"]
    assert border_short_circuit_gva("vigy|380", 380.0) == pytest.approx(vigy)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("german_load_mw", [20_000.0, 80_000.0])
def test_border_reactance_follows_the_short_circuit_power_not_the_rating(  # type: ignore[no-untyped-def]
    tmp_path, german_load_mw
) -> None:
    andes = pytest.importorskip("andes")
    path = build_connected_case(
        _connected(),
        {"a": 1.02, "b": 1.0, "x": 1.0},
        {"a": (200.0, 30.0)},
        {"a": (50.0, 0.0)},
        {"x": -10.0},
        tmp_path / "case.xlsx",
        neighbour_equivalent=continental_europe_equivalent(german_load_mw),
    )
    system = andes.load(str(path), setup=False, no_output=True, default_config=True)
    g = system.GENCLS.as_df().iloc[0]
    x_ohm = g["xd1"] * g["Vn"] ** 2 / g["Sn"]
    assert x_ohm == pytest.approx(g["Vn"] ** 2 / (1000.0 * border_short_circuit_gva("X", g["Vn"])))


@pytest.mark.filterwarnings("ignore")
def test_short_circuit_scale_scales_the_border_reactance(tmp_path) -> None:  # type: ignore[no-untyped-def]
    andes = pytest.importorskip("andes")
    x = []
    for scale in (1.0, 1.3):
        path = build_connected_case(
            _connected(),
            {"a": 1.02, "b": 1.0, "x": 1.0},
            {"a": (200.0, 30.0)},
            {"a": (50.0, 0.0)},
            {"x": -10.0},
            tmp_path / f"case{scale}.xlsx",
            neighbour_equivalent=continental_europe_equivalent(50_000.0, short_circuit_scale=scale),
        )
        system = andes.load(str(path), setup=False, no_output=True, default_config=True)
        x.append(float(system.GENCLS.xd1.v[0]))
    assert x[0] / x[1] == pytest.approx(1.3)


CASE_HOURS = (
    "high_wind", "high_solar", "low_load", "high_load", "median_load",
    "winter_peak", "summer_peak", "very_light_night", "baltic_import_low_load",
)  # fmt: skip


def test_entsoe_kinetic_energy_uses_table_1_on_the_online_rating() -> None:
    # nuclear 960 MW at LF 0.96 is 1000 MVA online at H 5.9 s; wind and solar add nothing
    assert entsoe_kinetic_energy_mws({"B14": 960.0, "B19": 5000.0, "B16": 3000.0}) == (
        pytest.approx(5900.0)
    )
    assert entsoe_kinetic_energy_mws({"B02": 810.0}) == pytest.approx(3800.0)
    assert ENTSOE_INERTIA["B11"] == (2.7, 0.61)
    with pytest.raises(DataError):
        entsoe_kinetic_energy_mws({"B99": 1.0})


def test_neighbour_inertia_table_covers_every_case_hour_in_ce_only() -> None:
    table = pd.read_csv(NEIGHBOUR_INERTIA)
    assert set(table["hour"]) == set(CASE_HOURS)
    assert not set(table["country"]) & {"DE", "GB", "IE", "NO", "SE", "FI", "DK2"}
    assert table["source"].str.contains("A75").all()
    per_hour = table.groupby("hour")["kinetic_energy_mws"].sum()
    assert per_hour.between(400_000.0, 2_500_000.0).all()


@pytest.mark.parametrize("scale", [0.7, 1.0, 1.3])
def test_entsoe_hourly_equivalent_takes_the_hour_and_scale(scale) -> None:  # type: ignore[no-untyped-def]
    table = pd.read_csv(NEIGHBOUR_INERTIA)
    rows = table[table["hour"].eq("high_wind")]
    eq = continental_europe_equivalent(
        50_000.0, "entsoe_hourly", hour="high_wind", inertia_scale=scale
    )
    assert eq.kinetic_energy_mws == pytest.approx(scale * rows["kinetic_energy_mws"].sum())
    assert not eq.includes_germany
    assert eq.country_energy_mws is None
    by_country = continental_europe_equivalent(
        50_000.0, "entsoe_hourly", hour="high_wind", split="country"
    )
    assert by_country.country_energy_mws is not None
    assert set(by_country.country_energy_mws) == set(rows["country"])
    ce_load = 50_000.0 + rows["generation_mw"].sum()
    assert eq.damping_mw_per_hz == pytest.approx(0.02 * ce_load)
    one = continental_europe_equivalent(
        50_000.0, "entsoe_hourly", hour="high_wind", self_regulation_per_hz=0.01
    )
    assert one.damping_mw_per_hz == pytest.approx(0.01 * ce_load)
    with pytest.raises(DataError):
        continental_europe_equivalent(50_000.0, "entsoe_hourly")


def test_border_shares_route_countries_without_an_ac_border_node() -> None:
    names = {"a": "vigy|380", "b": "vogelgrun|220", "c": "kasso|380"}
    kv = {"a": 380.0, "b": 220.0, "c": 380.0}
    plain = border_shares(names, kv)
    assert plain["a"] == pytest.approx(380**2 / (2 * 380**2 + 220**2))
    # Spain goes to France, Italy only to France here (no Swiss or Austrian node), Belgium half to
    # France and half to the Netherlands (no Dutch node: all to France)
    shares = border_shares(
        names, kv, {"FR": 100.0, "ES": 50.0, "IT": 30.0, "BE": 20.0, "DK1": 40.0}
    )
    assert shares["c"] == pytest.approx(40.0 / 240.0)
    assert shares["a"] + shares["b"] == pytest.approx(200.0 / 240.0)
    assert shares["a"] / shares["b"] == pytest.approx(380**2 / 220**2)
    with pytest.raises(DataError):
        border_shares({"x": "nowhere|380"}, {"x": 380.0}, {"FR": 1.0})


def test_border_nodes_have_countries_and_hvdc_terminals_are_marked() -> None:
    table = pd.read_csv(BORDER_SHORT_CIRCUIT).set_index("node")
    assert table["country"].notna().all()
    assert table.loc["asphard|380", "country"] == "CH"
    assert table.loc["blooren|220", "country"] == "LU"
    assert table.loc[["werben|220", "meiningen|220"], "country"].eq("AT").all()
    assert table.loc["ertsmyra|380", "country"] == "NO"
    assert "HVDC" in table.loc["ertsmyra|380", "operator"]
    assert "HVDC" in table.loc["lixhe|380", "operator"]


@pytest.mark.filterwarnings("ignore")
def test_entsoe_hourly_neighbours_hold_their_own_energy_and_stay_flat(tmp_path) -> None:  # type: ignore[no-untyped-def]
    andes = pytest.importorskip("andes")
    tables = _connected()
    bus = tables.busbar_section
    tables.busbar_section = bus.assign(name=bus["name"].where(bus["mrid"].ne("x"), "vigy|380"))
    eq = continental_europe_equivalent(50_000.0, "entsoe_hourly", hour="high_wind")
    path = build_connected_case(
        tables,
        {"a": 1.02, "b": 1.0, "x": 1.0},
        {"a": (200.0, 30.0)},
        {"a": (50.0, 0.0)},
        {"x": -10.0},
        tmp_path / "case.xlsx",
        neighbour_equivalent=eq,
    )
    system = andes.load(str(path), setup=False, no_output=True, default_config=True)
    neighbour = system.GENCLS.as_df()
    # the German machines are not subtracted: the table holds the neighbours only
    assert (neighbour["M"] / 2 * neighbour["Sn"]).sum() == pytest.approx(eq.kinetic_energy_mws)
    assert (neighbour["D"] * neighbour["Sn"] / 50.0).sum() == pytest.approx(eq.damping_mw_per_hz)
    system.setup()
    assert system.PFlow.run()
    system.TDS.config.tf = 10.0
    system.TDS.config.no_tqdm = 1
    assert system.TDS.run()
    omega = system.dae.ts.x[:, list(system.GENROU.omega.a) + list(system.GENCLS.omega.a)]
    assert abs(omega - 1.0).max() < 1e-6


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("hydro_governor", ["hygov", "tgov1"])
def test_german_governors_deliver_the_published_fcr(tmp_path, hydro_governor) -> None:  # type: ignore[no-untyped-def]
    andes = pytest.importorskip("andes")
    from tranpy.simulation.grid_case import CE_FCR_FULL_HZ, DE_FCR_MW

    common = (
        _connected(),
        {"a": 1.02, "b": 1.0, "x": 1.0},
        {"a": (200.0, 30.0)},
        {"a": (50.0, 0.0)},
        {"x": -10.0},
    )
    eq = continental_europe_equivalent(50_000.0)
    path = build_connected_case(
        *common, tmp_path / "fcr.xlsx", neighbour_equivalent=eq, hydro_governor=hydro_governor
    )
    system = andes.load(str(path), setup=False, no_output=True, default_config=True)
    german = system.GENROU.as_df()
    assert (german["D"] == 0).all()
    gains, reserve = 0.0, 0.0
    for name in ("TGOV1", "HYGOV"):
        model = getattr(system, name)
        if not model.n:
            continue
        gov = model.as_df()
        gov = gov[gov["syn"].isin(german["idx"])]
        gains += float((gov["Tn"] / (50.0 * gov["R"])).sum())
        if name == "TGOV1":
            p0 = 200.0  # the solved machine power of the fixture
            reserve += float((gov["VMAX"] * gov["Tn"] - p0).sum())
    assert gains == pytest.approx(DE_FCR_MW / CE_FCR_FULL_HZ)
    if reserve:
        assert reserve == pytest.approx(DE_FCR_MW)
    system.setup()
    assert system.PFlow.run()
    system.TDS.config.tf = 10.0
    system.TDS.config.no_tqdm = 1
    assert system.TDS.run()
    omega = system.dae.ts.x[:, list(system.GENROU.omega.a) + list(system.GENCLS.omega.a)]
    assert abs(omega - 1.0).max() < 1e-6

    legacy = build_connected_case(
        *common, tmp_path / "legacy.xlsx", neighbour_equivalent=eq, german_fcr_mw=None
    )
    old = andes.load(str(legacy), setup=False, no_output=True, default_config=True)
    assert (old.GENROU.as_df()["D"] == 2.0).all()
    assert set(old.TGOV1.as_df().query("syn in @german['idx']")["R"]) <= {0.05}
