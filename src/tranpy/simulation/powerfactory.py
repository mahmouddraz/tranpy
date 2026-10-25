"""DIgSILENT PowerFactory engine (needs a local PowerFactory with its Python API).

Implements :class:`~tranpy.simulation.engine.SimulationEngine`: for every scenario it
creates the fault, clearing, optional line-trip and load events, runs an RMS simulation
and exports *all* monitored variables to a temporary CSV that is parsed like the recorded
datasets and then deleted. Nothing is written elsewhere.

``powerfactory`` is imported only when an engine is created. Pass ``python_path`` (the
``Python/<version>`` folder of the installation) if the module is not on ``sys.path``;
the path is added for the import and removed again.
"""

from __future__ import annotations

import importlib
import logging
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tranpy.datasets import flatten_powerfactory_column
from tranpy.simulation.engine import FaultScenario, SimulationResult
from tranpy.systems import PowerSystem, get_system
from tranpy.vocabulary import Variable
from tranpy_core.errors import ConfigurationError, MissingDependencyError, SimulationError

__all__ = ["MONITORED_VARIABLES", "PowerFactoryEngine", "import_powerfactory"]

logger = logging.getLogger(__name__)

MONITORED_VARIABLES: dict[str, tuple[str, ...]] = {
    "*.ElmTerm": ("m:u", "m:phiu", "m:fehz"),
    "*.ElmSym": (
        "s:ve", "s:pt", "s:ut", "s:ie", "s:xspeed", "s:xme", "s:xmt", "s:cur1", "s:P1", "s:Q1",
        "s:outofstep", "c:firel", "c:firot", "c:dfrotx", "c:fi", "c:dorhz",
    ),
    "*.ElmLne": ("c:loading",),
    "*.ElmLod": ("m:Psum:bus1", "m:I1:bus1", "n:u1:bus1"),
}  # fmt: skip
"""Variables recorded per element class (matches the bundled event datasets)."""


def import_powerfactory(python_path: str | Path | None = None) -> Any:
    """Import the ``powerfactory`` module, temporarily extending ``sys.path`` if given."""
    added = False
    if python_path is not None and str(python_path) not in sys.path:
        sys.path.insert(0, str(python_path))
        added = True
    try:
        return importlib.import_module("powerfactory")
    except ImportError as exc:
        raise MissingDependencyError(
            "the PowerFactory Python API ('powerfactory') is not importable; install PowerFactory "
            "and pass python_path=<installation>/Python/<version>"
        ) from exc
    finally:
        if added:
            sys.path.remove(str(python_path))


def _check(code: int, what: str) -> None:
    if code != 0:
        raise SimulationError(f"PowerFactory {what} failed with code {code}")


@dataclass
class PowerFactoryEngine:
    """Run fault scenarios in a PowerFactory project.

    Attributes:
        system_name: system key; ``project`` defaults to its legacy name (e.g. ``NewEngland``).
        project: PowerFactory project to activate.
        simulation_time_s: RMS simulation end time.
        step_s: integration step.
        python_path: folder containing the ``powerfactory`` module, if not on ``sys.path``.
        app: an existing application object (mainly for tests); created if None.
    """

    system_name: str
    project: str | None = None
    simulation_time_s: float = 10.0
    step_s: float = 0.01
    python_path: str | Path | None = None
    app: Any = None
    _elements: dict[str, list[Any]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.app is None:
            self.app = import_powerfactory(self.python_path).GetApplication()
            if self.app is None:
                raise SimulationError(
                    "PowerFactory.GetApplication() returned None (licence or session problem)"
                )
        project = self.project or self.system.legacy_name
        if self.app.ActivateProject(project) is None:
            raise SimulationError(f"cannot activate PowerFactory project {project!r}")
        cases = self.app.GetProjectFolder("study").GetContents()
        if not cases:
            raise SimulationError(f"project {project!r} has no study case")
        cases[0].Activate()
        self._elements = {
            pattern: list(self.app.GetCalcRelevantObjects(pattern))
            for pattern in MONITORED_VARIABLES
        }
        results = self.app.GetFromStudyCase("Results.ElmRes")
        for element in results.GetContents():
            element.ClearVars()
        for pattern, variables in MONITORED_VARIABLES.items():
            for element in self._elements[pattern]:
                results.AddVars(element, *variables)
        results.AddVars(results, "b:tnow")
        logger.info(
            "PowerFactory project %s: %d lines, %d loads, %d generators",
            project,
            self.n_lines,
            self.n_loads,
            len(self._elements["*.ElmSym"]),
        )

    @property
    def system(self) -> PowerSystem:
        return get_system(self.system_name)

    @property
    def n_lines(self) -> int:
        return len(self._elements["*.ElmLne"])

    @property
    def n_loads(self) -> int:
        return len(self._elements["*.ElmLod"])

    def simulate(self, scenarios: Sequence[FaultScenario]) -> list[SimulationResult]:
        """Simulate the scenarios one after another."""
        return [self._simulate_one(s) for s in scenarios]

    def _simulate_one(self, scenario: FaultScenario) -> SimulationResult:
        if scenario.operating_point != 1.0:
            raise ConfigurationError(
                f"scenario {scenario.scenario_id}: operating_point != 1 is not supported by the "
                "PowerFactory engine yet; set the loading in the study case instead"
            )
        app = self.app
        lines, loads = self._elements["*.ElmLne"], self._elements["*.ElmLod"]
        folder = app.GetFromStudyCase("IntEvt")
        for event in folder.GetContents():
            event.Delete()
        fault = folder.CreateObject("EvtShc", "short circuit")
        fault.p_target = lines[scenario.fault_line]
        fault.time = scenario.fault_start_s
        fault.i_shc = 0
        fault.R_f = scenario.fault_resistance_ohm
        fault.X_f = scenario.fault_reactance_ohm
        clearing = folder.CreateObject("EvtSwitch", "fault clearing")
        clearing.p_target = lines[scenario.fault_line]
        clearing.time = scenario.fault_clear_s
        if scenario.trip_line is not None:
            trip = folder.CreateObject("EvtSwitch", "line tripping")
            trip.p_target = lines[scenario.trip_line]
            trip.time = scenario.fault_clear_s + scenario.trip_delay_s
        load = folder.CreateObject("EvtLod", "load event")
        load.p_target = loads[scenario.load_index]
        load.time = 0.0
        load.iopt_type = 0
        load.dP = scenario.load_change_pct
        try:
            init = app.GetFromStudyCase("ComInc")
            init.iopt_sim = "rms"
            init.iopt_net = "sym"
            init.dtgrd = self.step_s
            _check(init.Execute(), "initial conditions")
            sim = app.GetFromStudyCase("ComSim")
            sim.tstop = self.simulation_time_s
            _check(sim.Execute(), f"simulation of scenario {scenario.scenario_id}")
            frame = self._export()
        finally:
            for event in folder.GetContents():
                event.Delete()
            app.ResetCalculation()
        names = tuple(flatten_powerfactory_column(c) for c in frame.columns)
        data = frame.to_numpy(np.float64)
        if not names[0].endswith(Variable.TIME_S.value):
            raise SimulationError(f"unexpected first export column {names[0]!r}")
        return SimulationResult(
            scenario=scenario, time=data[:, 0], values=data[:, 1:], channel_names=names[1:]
        )

    def _export(self) -> pd.DataFrame:
        export = self.app.GetFromStudyCase("ComRes")
        export.pResult = self.app.GetFromStudyCase("Results.ElmRes")
        export.iopt_exp = 6  # CSV
        export.iopt_csel = 0  # all monitored variables of all elements
        export.iopt_tsel = 0  # all time steps
        export.iopt_locn = 2
        export.iopt_sep = 1
        export.ciopt_head = 1
        with tempfile.TemporaryDirectory(prefix="tranpy-pf-") as tmp:
            path = Path(tmp) / "result.csv"
            export.f_name = str(path)
            _check(export.Execute(), "result export")
            return pd.read_csv(path, header=[0, 1])
