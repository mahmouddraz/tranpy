from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from tranpy.simulation import FaultScenario, generate_events
from tranpy_core.errors import DataError

pytestmark = [pytest.mark.requires_andes, pytest.mark.slow]
pytest.importorskip("andes")

BUSES = """\
ID, Name,Voltage [KV],Demand [MW],Load Freq Coeff [WMs],Country,Population Prop,Longitude,Latitude
1,North 380,380,0,0,DE,0,10,54
2,Middle 380,380,400,0,DE,0,10,52
3,South 220,220,300,0,DE,0,10,48
4,Island 380,380,50,0,DE,0,13,54
5,Paris 380,380,500,0,FR,0,2,48
"""
GENERATORS = """\
ID,Technology,Max Output [MW], Marginal Cost [ua/MW],Inertia [MWs^2],Primary Control [MWs]
1,NU,1000,16,0.38,0.07
3,CG,500,40,0.10,0.04
5,NU,900,16,0.34,0.06
"""
LINES = """ID1,ID2,Resistance [Ohm],Reactance [Ohm], Thermal limit [MW]
1,2,0.001,0.01,1700
1,2,0.001,0.01,1700
2,3,0.001,0.02,1000
2,5,0.001,0.01,1700
"""


def _archive(path: Path, generators: str = GENERATORS) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in (("buses", BUSES), ("generators", generators), ("lines", LINES)):
            archive.writestr(f"pantagruel_csv/{name}.csv", text)
    return path


def test_build_small_case(tmp_path: Path) -> None:
    from tranpy.simulation.andes import AndesEngine
    from tranpy.simulation.cases import build_germany_case

    case = build_germany_case(tmp_path / "de.xlsx", source=_archive(tmp_path / "src.zip"))
    ss = AndesEngine("germany", case=case)._load(setup=True)
    assert sorted(ss.Bus.idx.v) == [1, 2, 3]  # foreign bus 5 and isolated bus 4 removed
    assert ss.Line.n == 3
    assert list(ss.Line.trans.v).count(1) == 1  # 380/220 kV branch
    assert ss.GENCLS.n == 2
    h = np.asarray(ss.GENCLS.M.vin) / 2  # machine base (.v is on the system base)
    np.testing.assert_allclose(h, [0.38, 0.10] * np.array([2 * np.pi * 50 * 100 / 2]) / [1000, 500])
    assert sum(ss.PV.p0.v) + sum(ss.Slack.p0.v) == pytest.approx(7.0 * 1.02)


def test_build_rejects_bad_sources(tmp_path: Path) -> None:
    from tranpy.simulation.cases import build_germany_case

    small = GENERATORS.splitlines()[0] + "\n1,NU,100,16,0.38,0.07\n"
    with pytest.raises(DataError, match="capacity"):
        build_germany_case(tmp_path / "a.xlsx", source=_archive(tmp_path / "a.zip", small))
    with zipfile.ZipFile(tmp_path / "b.zip", "w") as archive:
        archive.writestr("pantagruel_csv/buses.csv", BUSES)
    with pytest.raises(DataError, match=r"generators\.csv"):
        build_germany_case(tmp_path / "b.xlsx", source=tmp_path / "b.zip")


def test_bundled_germany_case() -> None:
    from tranpy.simulation.andes import AndesEngine

    engine = AndesEngine("germany", simulation_time_s=2.0, name_width=4)
    assert engine.nominal_frequency_hz == 50.0
    assert engine.n_lines == 1224
    assert engine.n_loads == 551
    scenario = FaultScenario(
        scenario_id=0, fault_line=0, fault_start_s=0.5, clearing_duration_s=0.1
    )
    events = generate_events(engine, [scenario], criterion="angle_360", protocol="stress_test")
    assert events.n_samples == 1
    assert events.n_channels == 553 * 3 + 86 * 6
    assert events.channel_names[0].startswith("Bus ")
