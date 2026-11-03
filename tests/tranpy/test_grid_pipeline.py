from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tranpy.grid_pipeline import (
    PipelineConfig,
    _safe_columns,
    export_tables,
    split_out_of_service,
    verify_tables,
)
from tranpy_core.errors import ConfigurationError


def test_config_from_environment(tmp_path: Path) -> None:
    env = {
        "TRANPY_FILES": str(tmp_path),
        "TRANPY_CATALOG": "cat",
        "TRANPY_GOLD_SCHEMA": "dev_x_grid_gold",
        "TRANPY_RUN_DATE": "2026-10-29",
    }
    config = PipelineConfig.from_env(env, silver_schema="s")
    assert config.schemas == {"bronze": "grid_bronze", "silver": "s", "gold": "dev_x_grid_gold"}
    assert config.run_dir == tmp_path / "2026-10-29"
    assert config.table("gold", "cim_substation") == "`cat`.`dev_x_grid_gold`.`cim_substation`"
    local = PipelineConfig.from_env({}, files=str(tmp_path))
    assert local.catalog is None
    with pytest.raises(ConfigurationError, match="no catalog"):
        local.table("gold", "x")
    with pytest.raises(ConfigurationError, match="TRANPY_FILES"):
        PipelineConfig.from_env({})


def test_safe_columns() -> None:
    frame = pd.DataFrame([[1, 2, 3]], columns=["Resistance_R(Ω)", "Full_name", "Full name"])
    assert list(_safe_columns(frame).columns) == ["resistance_r", "full_name", "full_name_2"]


def _gold() -> dict[str, pd.DataFrame]:
    bus = pd.DataFrame({"mrid": ["b1", "b2"], "lat": [50.0, 51.0], "lon": [10.0, 11.0]})
    return {
        "cim_busbar_section": bus,
        "cim_ac_line_segment": pd.DataFrame(
            {"mrid": ["l1"], "bus1_mrid": ["b1"], "bus2_mrid": ["b2"]}
        ),
        "cim_power_transformer": pd.DataFrame(
            {"mrid": ["t1"], "bus1_mrid": ["b1"], "bus2_mrid": ["b2"]}
        ),
        "cim_synchronous_machine": pd.DataFrame({"mrid": ["g1"], "bus_mrid": ["b1"]}),
        "cim_energy_consumer": pd.DataFrame({"mrid": ["c1"], "bus_mrid": ["b2"]}),
        "cim_external_network_injection": pd.DataFrame({"mrid": [], "bus_mrid": []}),
        "cim_voltage_level": pd.DataFrame({"nominal_voltage_kv": [220.0, 380.0]}),
        "cim_model_info": pd.DataFrame({"key": ["loadflow_status"], "value": ["CONVERGED"]}),
    }


def test_verify_tables() -> None:
    tables = _gold()
    assert verify_tables(lambda layer, name: tables[name]) == []
    tables["cim_energy_consumer"] = pd.DataFrame({"mrid": ["c1", "c1"], "bus_mrid": ["b2", "b9"]})
    tables["cim_busbar_section"].loc[1, "lat"] = 30.0
    tables["cim_voltage_level"] = pd.DataFrame({"nominal_voltage_kv": [110.0]})
    tables["cim_model_info"] = pd.DataFrame({"key": ["loadflow_status"], "value": ["FAILED"]})
    problems = verify_tables(lambda layer, name: tables[name])
    assert len(problems) == 5
    assert any("unknown buses" in p for p in problems)
    assert any("power flow: FAILED" in p for p in problems)


def test_no_export_gives_no_tables(tmp_path: Path) -> None:
    assert export_tables(tmp_path) == ({}, {})


def test_split_out_of_service_leaves_reserve_and_closed_plants_out() -> None:
    attachments = pd.DataFrame(
        {
            "mastr_id": ["A", "B", "C", "D", "cell"],
            "name": ["a", "b", "c", "d", None],
            "node": ["x|380"] * 5,
            "fuel": ["Erdgas"] * 5,
            "capacity_mw": [100.0, 200.0, 50.0, 10.0, 5.0],
            "status": [
                "In Betrieb",
                "Netzreserve aufgrund § 13b EnWG",
                "vorläufig stillgelegt nach § 13b EnWG",
                "endgültig stillgelegt",
                None,
            ],
            "in_operation": [True, False, False, False, None],
            "reserve": [False, True, True, False, None],
        }
    )
    kept, out = split_out_of_service(attachments)
    assert kept["mastr_id"].tolist() == ["A", "cell"]
    assert out["element"].tolist() == ["B", "C", "D"]
    assert out["group"].tolist() == ["grid reserve", "provisionally closed", "closed"]
    assert out["capacity_mw"].sum() == 260.0
