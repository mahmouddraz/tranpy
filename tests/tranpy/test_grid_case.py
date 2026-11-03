from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from tranpy.simulation.grid_case import (
    DYNAMIC_VALUES,
    MACHINE_VALUES,
    _main_component,
    build_connected_case,
    build_grid_case,
    fuel_class,
    line_parameters,
)


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
