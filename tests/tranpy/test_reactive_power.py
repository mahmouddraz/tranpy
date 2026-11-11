from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tranpy.reactive_power import (
    RULES,
    boundary_q_limits,
    embedded_reactive_capability,
    load_tan_phi,
    read_compensation,
    with_load_tan_phi,
)


def _machines() -> pd.DataFrame:
    base = {"in_service": True, "min_p_mw": 0.0, "target_v_pu": 1.02, "voltage_control": False}
    return pd.DataFrame(
        [
            {**base, "mrid": "embedded/Z/Windenergie (Onshore-Anlage)/a", "bus_mrid": "b",
             "fuel": "Windenergie (Onshore-Anlage)", "p_mw": 60.0, "max_p_mw": 100.0},
            {**base, "mrid": "embedded/Z/Solare Strahlungsenergie/a", "bus_mrid": "b",
             "fuel": "Solare Strahlungsenergie", "p_mw": 5.0, "max_p_mw": 100.0},
            {**base, "mrid": "embedded/Z/Biomasse/a", "bus_mrid": "b", "fuel": "Biomasse",
             "p_mw": 10.0, "max_p_mw": 20.0, "voltage_control": True, "q_max_mvar": 3.3},
            {**base, "mrid": "g1", "bus_mrid": "a", "fuel": "Erdgas", "p_mw": 60.0,
             "max_p_mw": 500.0, "voltage_control": True},
        ]
    )  # fmt: skip


def test_rules_match_the_power_factors() -> None:
    assert RULES["4110"].over == pytest.approx(0.3287, abs=1e-4)
    assert RULES["4120-1"].under == pytest.approx(-0.2279, abs=1e-4)
    assert RULES["4120-1"].over == pytest.approx(0.4843, abs=1e-4)
    assert RULES["4120-2"].over == pytest.approx(0.4108, abs=1e-4)
    assert RULES["4120-3"].under == pytest.approx(-0.4108, abs=1e-4)


def test_range_follows_installed_power_above_the_minimum() -> None:
    out = embedded_reactive_capability(_machines(), rule="4120-2")
    wind, solar, bio, gas = (out.iloc[i] for i in range(4))
    assert wind["q_max_mvar"] == pytest.approx(41.08, abs=0.01)
    assert wind["q_min_mvar"] == pytest.approx(-32.87, abs=0.01)
    assert bool(wind["voltage_control"])
    # 5 % of the installed power is below the 10 % minimum: no range, no control
    assert solar["q_max_mvar"] == 0.0 and not bool(solar["voltage_control"])
    # other fuels and non-embedded machines are left as they are
    assert bio["q_max_mvar"] == 3.3 and bool(gas["voltage_control"])


def test_fixed_cos_phi_sets_target_q() -> None:
    out = embedded_reactive_capability(_machines(), control="cos_phi", cos_phi_share=1.0)
    assert out.iloc[0]["target_q_mvar"] == pytest.approx(32.87, abs=0.01)
    assert not bool(out.iloc[0]["voltage_control"])
    under = embedded_reactive_capability(_machines(), control="cos_phi", cos_phi_share=-0.5)
    assert under.iloc[0]["target_q_mvar"] == pytest.approx(-16.43, abs=0.01)


def test_load_tan_phi_falls_with_load() -> None:
    peak, light = load_tan_phi(1.0), load_tan_phi(0.0)
    assert peak == pytest.approx(0.75 * np.tan(np.arccos(0.975)) + 0.25 * np.tan(np.arccos(0.99)))
    assert light > peak
    assert load_tan_phi(2.0) == peak


def test_with_load_tan_phi_keeps_hvdc_rows() -> None:
    loads = pd.DataFrame(
        {"mrid": ["lumped/load/a", "hvdc/x"], "p_mw": [100.0, 50.0], "q_mvar": [0.0, 7.0]}
    )
    out = with_load_tan_phi(loads, 0.2)
    assert out["q_mvar"].tolist() == [20.0, 7.0]


def _csv(tmp_path):  # type: ignore[no-untyped-def]
    path = tmp_path / "c.csv"
    pd.DataFrame(
        {
            "node": ["a|380", "a|380", "a|380", "b|380", "z|380"],
            "type": ["STATCOM", "MSCDN", "shunt_reactor", "STATCOM", "STATCOM"],
            "q_max_mvar": [300.0, 300.0, -120.0, 300.0, 300.0],
            "q_min_mvar": [-300.0, 300.0, -120.0, -300.0, -300.0],
            "in_model": [True, True, True, False, True],
            "in_model_with_assumptions": [True, True, True, True, True],
        }
    ).to_csv(path, index=False)
    return path


def test_read_compensation_keeps_published_units_on_known_buses(tmp_path) -> None:  # type: ignore[no-untyped-def]
    buses = pd.DataFrame({"mrid": ["A", "B"], "name": ["a|380", "b|380"]})
    raise_ = read_compensation(_csv(tmp_path), buses)
    assert raise_["kind"].tolist() == ["controlled", "shunt"]
    assert raise_["bus_mrid"].tolist() == ["A", "A"]
    lower = read_compensation(_csv(tmp_path), buses, switching="lower")
    assert lower["q_max_mvar"].tolist() == [300.0, -120.0]
    assumed = read_compensation(_csv(tmp_path), buses, with_assumptions=True, switching="all")
    assert len(assumed) == 4


def _grid(comp: bool) -> SimpleNamespace:
    machines = _machines().iloc[[0, 3]].reset_index(drop=True)
    machines.loc[0, "bus_mrid"] = "b"
    machines["q_max_mvar"] = [0.0, 500.0]
    machines["q_min_mvar"] = [0.0, -500.0]
    machines["target_q_mvar"] = [10.0, 0.0]
    return SimpleNamespace(
        substation=pd.DataFrame({"mrid": ["s"], "site": ["s"], "country": ["DE"]}),
        voltage_level=pd.DataFrame(
            {"mrid": ["v"], "substation_mrid": ["s"], "nominal_voltage_kv": [380.0]}
        ),
        busbar_section=pd.DataFrame({"mrid": ["a", "b"], "voltage_level_mrid": ["v", "v"]}),
        ac_line_segment=pd.DataFrame(
            {
                "mrid": ["l1"],
                "bus1_mrid": ["a"],
                "bus2_mrid": ["b"],
                "length_km": [100.0],
                "r_ohm": [3.0],
                "x_ohm": [30.0],
                "bch_s": [0.0],
                "rated_current_a": [2000.0],
            }
        ),
        power_transformer=pd.DataFrame(
            columns=[
                "mrid",
                "bus1_mrid",
                "bus2_mrid",
                "rated_u1_kv",
                "rated_u2_kv",
                "r_ohm",
                "x_ohm",
                "g_s",
                "b_s",
            ]
        ),
        synchronous_machine=machines,
        external_network_injection=pd.DataFrame(
            columns=["mrid", "bus_mrid", "p_mw", "target_v_pu"]
        ),
        energy_consumer=pd.DataFrame(
            {"mrid": ["c"], "bus_mrid": ["b"], "p_mw": [100.0], "q_mvar": [200.0]}
        ),
    )


def _units() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mrid": ["s1", "c1"],
            "bus_mrid": ["b", "b"],
            "kind": ["controlled", "shunt"],
            "q_max_mvar": [100.0, 50.0],
            "q_min_mvar": [-100.0, 50.0],
            "target_v_pu": [1.0, 1.0],
        }
    )


def test_pandapower_compensation_and_target_q() -> None:
    pp = pytest.importorskip("pandapower")
    from tranpy.pandapower_model import network_from_cim_tables

    tables = _grid(True)
    net = network_from_cim_tables(tables, compensation=_units())
    assert net.shunt["q_mvar"].tolist() == [-50.0]
    assert net.gen.loc[net.gen["name"].eq("s1"), "max_q_mvar"].iloc[0] == 100.0
    assert net.sgen.loc[net.sgen["name"].eq("b"), "q_mvar"].iloc[0] == 10.0
    pp.runpp(net, enforce_q_lims=True)
    held = net.gen["name"].eq("s1")
    assert 0.0 < net.res_gen.loc[held, "q_mvar"].iloc[0] < 100.0
    assert net.res_bus.loc[net.gen.loc[held, "bus"].iloc[0], "vm_pu"] == pytest.approx(1.0)


@pytest.mark.requires_powsybl
def test_powsybl_compensation_and_target_q() -> None:
    pytest.importorskip("pypowsybl")
    import pypowsybl as ps

    from tranpy.powsybl_model import _ids, network_from_cim_tables

    network = network_from_cim_tables(_grid(True), compensation=_units())
    assert len(network.get_shunt_compensators()) == 1
    ps.loadflow.run_ac(network, ps.loadflow.Parameters(use_reactive_limits=True))
    shunt = network.get_shunt_compensators()
    gens = network.get_generators()
    wind = _ids("G", pd.Series(["embedded/Z/Windenergie (Onshore-Anlage)/a"]))[0]
    assert gens.loc[wind, "q"] == pytest.approx(-10.0)
    assert 0.0 < -gens.loc["Q_s1", "q"] <= 100.0 + 1e-6
    # the shunt injects 50 Mvar at 1 pu, scaled with the square of the solved voltage
    v = network.get_buses()["v_mag"].min() / 380.0
    assert -shunt["q"].iloc[0] == pytest.approx(50.0 * v**2, rel=0.02)


def test_boundary_q_limits_ignore_lines_not_in_operation() -> None:
    lines = pd.DataFrame(
        {
            "bus1_mrid": ["x", "x", "y"],
            "bus2_mrid": ["a", "b", "c"],
            "rated_current_a": [1000.0, 4000.0, 10.0],
            "nominal_voltage_kv": [220.0, 220.0, 220.0],
            "status": ["in_operation", "out_of_service", "in_operation"],
        }
    )
    ext = pd.DataFrame({"mrid": ["ex", "ey"], "bus_mrid": ["x", "y"]})
    limits = boundary_q_limits(lines, ext)
    assert limits["ex"] == pytest.approx(0.33 * 220.0 * 1000.0 * np.sqrt(3) / 1000.0)
    assert limits["ey"] == 50.0
