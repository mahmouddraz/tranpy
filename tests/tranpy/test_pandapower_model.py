from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

pp = pytest.importorskip("pandapower")

from tranpy.pandapower_model import line_loading, network_from_cim_tables  # noqa: E402


def _tables(wind_mw: float = 40.0) -> SimpleNamespace:
    machine = {"in_service": True, "min_p_mw": 0.0, "target_v_pu": 1.02}
    return SimpleNamespace(
        voltage_level=pd.DataFrame({"mrid": ["v"], "nominal_voltage_kv": [380.0]}),
        busbar_section=pd.DataFrame({"mrid": ["a", "b"], "voltage_level_mrid": ["v", "v"]}),
        ac_line_segment=pd.DataFrame(
            {
                "mrid": ["l1", "out"],
                "bus1_mrid": ["a", "a"],
                "bus2_mrid": ["b", "b"],
                "length_km": [10.0, 10.0],
                "r_ohm": [1.0, 1.0],
                "x_ohm": [10.0, 10.0],
                "bch_s": [0.0, 0.0],
                "rated_current_a": [1000.0, 1000.0],
                "status": ["in_operation", "under_construction"],
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
        synchronous_machine=pd.DataFrame(
            [
                {
                    **machine,
                    "bus_mrid": "a",
                    "fuel": "Erdgas",
                    "p_mw": 60.0,
                    "max_p_mw": 500.0,
                    "voltage_control": True,
                },
                {
                    **machine,
                    "bus_mrid": "b",
                    "fuel": "Windenergie (Onshore-Anlage)",
                    "p_mw": wind_mw,
                    "max_p_mw": 100.0,
                    "voltage_control": False,
                },
            ]
        ),
        external_network_injection=pd.DataFrame(
            columns=["mrid", "bus_mrid", "p_mw", "target_v_pu"]
        ),
        energy_consumer=pd.DataFrame({"bus_mrid": ["b"], "p_mw": [100.0], "q_mvar": [0.0]}),
    )


def test_only_operating_lines_and_wind_is_static() -> None:
    net = network_from_cim_tables(_tables(), voltage_control="fuel")
    assert net.line["name"].tolist() == ["l1"]
    assert len(net.gen) == 0
    wind = net.sgen[net.sgen["name"].ne("slack dispatch")]
    assert wind["p_mw"].tolist() == [40.0]
    assert wind["q_mvar"].tolist() == [0.0]


def test_inverter_power_factor_gives_reactive_power() -> None:
    net = network_from_cim_tables(_tables(), voltage_control="fuel", inverter_cos_phi=0.8)
    wind = net.sgen[net.sgen["name"].ne("slack dispatch")]
    assert wind["q_mvar"].iloc[0] == pytest.approx(30.0)


def test_flow_and_loading_match_a_hand_calculation() -> None:
    net = network_from_cim_tables(_tables())
    pp.runpp(net)
    slack_dispatch = net.sgen.loc[net.sgen["name"].eq("slack dispatch"), "p_mw"].sum()
    assert net.res_ext_grid["p_mw"].iloc[0] + slack_dispatch == pytest.approx(60.0, abs=0.1)
    # 60 MW at about 388 kV is about 89 A on a 1000 A line
    assert line_loading(net)["l1"] == pytest.approx(0.089, abs=0.003)


def test_distributed_slack_keeps_the_dispatch_of_the_slack_bus() -> None:
    pp = pytest.importorskip("pandapower")
    tables = _tables()
    net = network_from_cim_tables(tables, voltage_control="table")
    pp.runpp(net, distributed_slack=True)
    total = net.res_ext_grid.p_mw.sum() + net.res_gen.p_mw.sum() + net.res_sgen.p_mw.sum()
    assert total == pytest.approx(net.res_load.p_mw.sum() + net.res_line.pl_mw.sum(), abs=1e-3)
    assert net.sgen["name"].eq("slack dispatch").any()
