from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tranpy.datasets import FAULT_CLEAR, FAULT_START
from tranpy.simulation import (
    FaultScenario,
    MockEngine,
    ScenarioSampler,
    SimulationEngine,
    SimulationResult,
    generate_events,
)
from tranpy_core.errors import ConfigurationError, MissingDependencyError, SimulationError


def test_sampler_is_reproducible_and_in_range() -> None:
    sampler = ScenarioSampler(n_lines=5, n_loads=3, max_load_change_pct=10, trip_adjacent_line=True)
    a = sampler.sample(200, seed=4)
    assert a == sampler.sample(200, seed=4)
    assert a != sampler.sample(200, seed=5)
    changes = {s.load_change_pct for s in a}
    assert min(changes) == -10 and max(changes) == 10  # both ends reachable
    assert all(0 <= s.fault_start_s <= 5.0 for s in a)
    assert all(s.clearing_duration_s == pytest.approx(0.2) for s in a)
    assert all(s.trip_line is not None and s.trip_line != s.fault_line for s in a)
    assert [s.scenario_id for s in sampler.sample(3, seed=0, first_id=10)] == [10, 11, 12]


def test_scenario_and_sampler_validation() -> None:
    with pytest.raises(ConfigurationError) as info:
        FaultScenario(
            scenario_id=0,
            fault_line=-1,
            fault_start_s=-1,
            clearing_duration_s=0,
            fault_resistance_ohm=-1,
            trip_line=-1,
        )
    assert len(info.value.problems) == 5
    with pytest.raises(ConfigurationError, match="trip_line"):
        FaultScenario(
            scenario_id=0, fault_line=1, fault_start_s=0, clearing_duration_s=0.1, trip_line=1
        )
    with pytest.raises(ConfigurationError) as info:
        ScenarioSampler(n_lines=0, n_loads=0, clearing_cycles=0, fault_start_fraction=(0.6, 0.2),
                        fault_resistance_ohm=(2, 1), max_load_change_pct=-1)  # fmt: skip
    assert len(info.value.problems) == 6


def test_mock_engine_dynamics_define_the_label() -> None:
    engine = MockEngine("ieee_9bus", simulation_time_s=4.0)
    assert isinstance(engine, SimulationEngine)
    short = FaultScenario(scenario_id=0, fault_line=0, fault_start_s=0.5, clearing_duration_s=0.02)
    long = FaultScenario(scenario_id=1, fault_line=0, fault_start_s=0.5, clearing_duration_s=1.5)
    events = generate_events(engine, [short, long])
    assert events.labels.tolist() == [0, 1]  # a longer fault destabilises the same contingency
    assert events.annotations[FAULT_START].tolist() == [0.5, 0.5]
    assert events.annotations[FAULT_CLEAR].tolist() == pytest.approx([0.52, 2.0])
    flags = events[1].values[:, events.channel_index("*|s:outofstep")]
    assert np.all(np.diff(flags, axis=0) >= 0)  # latched
    assert np.isnan(events.annotations["trip_line"]).all()


def test_mock_engine_is_deterministic_and_sized_like_the_system() -> None:
    a, b = (
        MockEngine("new_england", simulation_time_s=1.0),
        MockEngine("new_england", simulation_time_s=1.0),
    )
    assert a.network.n_generators == 10 and a.network.n_buses == 39
    np.testing.assert_array_equal(a.network.coupling_pu, b.network.coupling_pu)
    sc = ScenarioSampler(
        n_lines=a.n_lines, n_loads=a.n_loads, simulation_time_s=1.0, trip_adjacent_line=True
    ).sample(3, seed=0)
    ra, rb = a.simulate(sc), b.simulate(sc)
    np.testing.assert_array_equal(ra[2].values, rb[2].values)
    assert ra[0].channel_names[0] == "Bus 01|m:u in p.u."
    assert (
        MockEngine("new_england", network_seed=1).network.coupling_pu.tolist()
        != a.network.coupling_pu.tolist()
    )
    with pytest.raises(ConfigurationError):
        MockEngine(step_s=0)
    with pytest.raises(ConfigurationError, match="outside the network"):
        a.simulate(
            [
                FaultScenario(
                    scenario_id=0, fault_line=99, fault_start_s=0.1, clearing_duration_s=0.1
                )
            ]
        )
    assert MockEngine("ieee_9bus", angle_spread_rad=0.1, coupling_scale=2.0).network.lines


def test_generate_events_checks_engine_output() -> None:
    sc = [FaultScenario(scenario_id=0, fault_line=0, fault_start_s=0.1, clearing_duration_s=0.1)]
    with pytest.raises(ConfigurationError):
        generate_events(MockEngine(), [])

    class Broken(MockEngine):
        def simulate(self, scenarios):  # type: ignore[no-untyped-def,override]
            return []

    with pytest.raises(SimulationError, match="0 results"):
        generate_events(Broken(), sc)

    class Inconsistent(MockEngine):
        def simulate(self, scenarios):  # type: ignore[no-untyped-def,override]
            r = super().simulate(scenarios)[0]
            other = SimulationResult(
                scenario=r.scenario,
                time=r.time,
                values=r.values[:, :1],
                channel_names=r.channel_names[:1],
            )
            return [r, other]

    with pytest.raises(SimulationError, match="inconsistent channels"):
        generate_events(Inconsistent(), sc * 2)
    with pytest.raises(SimulationError, match="do not match"):
        SimulationResult(
            scenario=sc[0], time=np.zeros(3), values=np.zeros((2, 1)), channel_names=("a",)
        )


# ---------------------------------------------------------------- PowerFactory adapter, faked


class _Obj:
    def __init__(self, name: str = "", children: list[Any] | None = None) -> None:
        self.loc_name = name
        self.children = children if children is not None else []
        self.vars: list[tuple[Any, tuple[str, ...]]] = []
        self.deleted = False

    def GetContents(self) -> list[Any]:
        return [c for c in self.children if not c.deleted]

    def ClearVars(self) -> None:
        self.vars.clear()

    def AddVars(self, element: Any, *variables: str) -> None:
        self.vars.append((element, variables))

    def CreateObject(self, cls: str, name: str) -> Any:
        obj = _Obj(name)
        obj.cls = cls  # type: ignore[attr-defined]
        self.children.append(obj)
        return obj

    def Delete(self) -> None:
        self.deleted = True

    def Activate(self) -> int:
        return 0


class _FakeApp:
    def __init__(self, fail: str | None = None) -> None:
        self.fail = fail
        self.events = _Obj("events")
        self.results = _Obj("results", [_Obj("old")])
        self.lines = [_Obj(f"Line {i}") for i in range(3)]
        self.loads = [_Obj("Load A")]
        self.created: list[list[str]] = []
        self.export = _Obj("export")
        self.export.Execute = self._export  # type: ignore[attr-defined]
        self.sim = _Obj("sim")
        self.sim.Execute = lambda: 1 if fail == "sim" else 0  # type: ignore[attr-defined]
        self.init = _Obj("init")
        self.init.Execute = lambda: 0  # type: ignore[attr-defined]
        self.resets = 0

    def _export(self) -> int:
        self.created.append([f"{e.cls}:{e.p_target.loc_name}" for e in self.events.GetContents()])  # type: ignore[attr-defined]
        cols = pd.MultiIndex.from_arrays(
            [
                ["Results", "\\g\\Bus 1.ElmTerm", "\\g\\G 1.ElmSym"],
                ["b:tnow in s", "m:u in p.u.", "s:outofstep"],
            ]
        )
        pd.DataFrame([[0.0, 1.0, 0.0], [0.01, 0.9, 0.0]], columns=cols).to_csv(
            self.export.f_name, index=False
        )  # type: ignore[attr-defined]
        return 0

    def ActivateProject(self, name: str) -> Any:
        return None if self.fail == "project" else _Obj(name)

    def GetProjectFolder(self, kind: str) -> Any:
        return _Obj("study", [] if self.fail == "cases" else [_Obj("case")])

    def GetCalcRelevantObjects(self, pattern: str) -> list[Any]:
        return {"*.ElmLne": self.lines, "*.ElmLod": self.loads}.get(pattern, [_Obj("x")])

    def GetFromStudyCase(self, name: str) -> Any:
        return {
            "Results.ElmRes": self.results,
            "IntEvt": self.events,
            "ComInc": self.init,
            "ComSim": self.sim,
            "ComRes": self.export,
        }[name]

    def ResetCalculation(self) -> None:
        self.resets += 1


def test_powerfactory_engine_with_fake_application() -> None:
    from tranpy.simulation.powerfactory import PowerFactoryEngine

    app = _FakeApp()
    engine = PowerFactoryEngine("NineBusSystem", app=app)
    assert isinstance(engine, SimulationEngine)
    assert engine.n_lines == 3 and engine.n_loads == 1
    assert app.results.vars and app.results.vars[-1][1] == ("b:tnow",)
    scenario = FaultScenario(
        scenario_id=7, fault_line=1, fault_start_s=0.2, clearing_duration_s=0.1, trip_line=2
    )
    events = generate_events(engine, [scenario])
    assert events.channel_names == ("Bus 1|m:u in p.u.", "G 1|s:outofstep")
    assert events.labels.tolist() == [0]
    assert app.created[0] == [
        "EvtShc:Line 1",
        "EvtSwitch:Line 1",
        "EvtSwitch:Line 2",
        "EvtLod:Load A",
    ]
    assert app.events.GetContents() == [] and app.resets == 1  # events removed, calculation reset
    assert app.sim.tstop == 10.0  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("fail", "message"), [("project", "cannot activate"), ("cases", "no study case")]
)
def test_powerfactory_engine_setup_errors(fail: str, message: str) -> None:
    from tranpy.simulation.powerfactory import PowerFactoryEngine

    with pytest.raises(SimulationError, match=message):
        PowerFactoryEngine("ieee_9bus", app=_FakeApp(fail))


def test_powerfactory_simulation_failure_cleans_up() -> None:
    from tranpy.simulation.powerfactory import PowerFactoryEngine

    app = _FakeApp("sim")
    engine = PowerFactoryEngine("ieee_9bus", app=app)
    with pytest.raises(SimulationError, match="code 1"):
        engine.simulate(
            [FaultScenario(scenario_id=0, fault_line=0, fault_start_s=0.1, clearing_duration_s=0.1)]
        )
    assert app.events.GetContents() == [] and app.resets == 1


def test_import_powerfactory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from tranpy.simulation import powerfactory as pfmod

    with pytest.raises(MissingDependencyError, match="python_path"):
        pfmod.import_powerfactory(tmp_path)
    assert str(tmp_path) not in sys.path
    (tmp_path / "powerfactory.py").write_text("def GetApplication():\n    return None\n")
    monkeypatch.delitem(sys.modules, "powerfactory", raising=False)
    module = pfmod.import_powerfactory(tmp_path)
    assert module.GetApplication() is None
    with pytest.raises(SimulationError, match="GetApplication"):
        pfmod.PowerFactoryEngine("ieee_9bus", python_path=tmp_path)
    monkeypatch.delitem(sys.modules, "powerfactory", raising=False)
