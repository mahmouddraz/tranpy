"""ANDES engine: open-source RMS simulation (optional ``[andes]`` extra).

`ANDES <https://github.com/CURENT/andes>`_ simulates power-system dynamics in Python with
detailed generator, exciter, governor and stabiliser models and reads PSS/E ``.raw/.dyr``
and its own ``.xlsx`` cases, so grids other than the bundled benchmarks can be studied
without a PowerFactory licence.

How a :class:`~tranpy.simulation.engine.FaultScenario` is applied:

* **lines**: only transmission lines are fault candidates; transformers (which ANDES also
  stores as lines) are excluded, as in the PowerFactory campaigns.
* **fault**: ANDES faults buses, not lines. The fault is applied at the *from* bus of the
  scenario's line (impedance converted from ohm to per unit on the bus base, reactance at
  least :data:`MIN_FAULT_REACTANCE_PU`) and cleared
  at ``fault_clear_s``, when the line is opened (as in a protection trip).
* **additional trip**: ``trip_line`` is opened ``trip_delay_s`` after clearing.
  :attr:`AndesEngine.line_neighbours` lets the sampler pick a line that shares a bus with
  the faulted one.
* **islanding**: if the line openings cut off part of the network, that part is switched
  off (its buses, loads and shunts) at the moment of separation, as protection would; its
  bus voltages are zero from then on. A cut-off part that contains a generator is not
  simulated and returned as not completed.
* **load change**: the selected load's demand is scaled by ``1 + load_change_pct/100``
  before the power flow, i.e. as a change of operating point, not as a step at t = 0.
* **operating point**: ``load_factors`` scale each load (generation re-dispatched in
  proportion to the total demand), otherwise ``operating_point`` scales all loads and
  generators; applied before the power flow. Operating points whose power flow does not
  converge or with a bus voltage below :data:`MIN_PREFAULT_VOLTAGE_PU` are rejected.
* **fault type and position**: three-phase faults only; the fault is applied at the line
  terminal nearer to ``fault_position``.

Recorded channels use the same names as the PowerFactory exports (``Bus 01|m:u in p.u.``,
``G 01|c:firot in deg`` ...). Rotor angles are relative to the centre of inertia. A
generator is out of step once its angle relative to the centre of inertia exceeds
180 degrees (latched), the criterion ANDES itself uses.

After loss of synchronism the ANDES solver may stop converging before the end time; the
recording then ends early and the true end time is stored per event
(annotation ``simulation_end_s``). A run that stops early *without* an out-of-step
generator has no decided outcome; it is marked as not completed and left out by
:func:`~tranpy.simulation.engine.generate_events` instead of being labelled stable.

Scenarios run in parallel processes with ``n_jobs > 1``.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from tranpy.simulation.engine import FaultScenario, SimulationResult
from tranpy.systems import PowerSystem, get_system
from tranpy.vocabulary import Variable
from tranpy_core._optional import require
from tranpy_core.errors import ConfigurationError, SimulationError

__all__ = [
    "DEFAULT_CASES",
    "MIN_FAULT_REACTANCE_PU",
    "AndesEngine",
    "clamped_governors",
    "shift_governor_limits",
]

logger = logging.getLogger(__name__)

DEFAULT_CASES: dict[str, str] = {"new_england": "ieee39/ieee39_full.xlsx", "germany": "germany"}
"""Cases used when no case file is given: ANDES built-in cases, and ``germany`` (bundled,
see :func:`tranpy.simulation.cases.build_germany_case`)."""

_CASE_FREQUENCY_HZ: dict[str, float] = {"germany": 50.0}
"""Nominal frequency of bundled cases (ANDES cases do not store it; its default is 60 Hz)."""

_MACHINES = ("GENROU", "GENCLS")
"""Synchronous-machine models whose states are recorded."""

MIN_PREFAULT_VOLTAGE_PU = 0.9
"""Operating points with a lower pre-fault bus voltage are rejected (not simulated)."""

MIN_FAULT_REACTANCE_PU = 1e-4
"""Smallest fault reactance used (the ANDES default). Bolted faults given in ohm (e.g.
0.01 ohm at 345 kV, about 1e-6 pu) make the ANDES solution numerically unstable: rotor
angles diverge even after a 50 ms fault that is stable with 1e-4 pu."""

_FaultArgs = dict[str, Any]


def _load_openmp_libraries_first() -> None:
    """Import installed boosting libraries before ANDES.

    ANDES's sparse solver (kvxopt) bundles its own OpenMP runtime. On macOS, XGBoost
    crashes (segmentation fault) when it is first imported after that runtime is loaded,
    e.g. when models are trained in the process that ran the simulations; importing it
    first avoids that.
    """
    for name in ("xgboost", "lightgbm"):
        if importlib.util.find_spec(name) is not None:
            importlib.import_module(name)


_GOVERNOR_LIMITS = {"TGOV1": ("VMAX", "VMIN"), "HYGOV": ("GMAX", "GMIN")}


def shift_governor_limits(ss: Any, pv_base: npt.ArrayLike, *, slack: bool = False) -> None:
    """Move each governor's limits by its unit's change of PV setpoint (set-up system).

    With ``slack=True`` (call after the power flow) only the governors of slack machines move,
    by the solved slack power minus its case setpoint ``Slack.p0``: the slack takes up the
    mismatch of non-uniform ``load_factors`` and of losses, so its solved power leaves the
    band built around the base dispatch.

    ``pv_base`` is ``PV.p0`` before the scenario scaled it. Limits are on the system base after
    ``setup`` (ANDES ``power=True``), like ``p0``: the TGOV1 valve limits move by the setpoint
    change dp, the HYGOV gate limits by dp / At (steady state P = At * (gate - qNL)). A lower
    limit that was not negative stays at 0 or above; a negative one (a neighbour equivalent that
    imports) is shifted unclipped.
    """
    if slack:
        solved = np.asarray(ss.Slack.p.v, dtype=float) - np.asarray(ss.Slack.p0.v, dtype=float)
        dp = dict(zip(ss.Slack.idx.v, solved, strict=True))
    else:
        base = np.asarray(pv_base, dtype=float)
        dp = dict(zip(ss.PV.idx.v, np.asarray(ss.PV.p0.v, dtype=float) - base, strict=True))
    gen_of: dict[Any, Any] = {}
    for name in _MACHINES:
        model = getattr(ss, name)
        gen_of.update(zip(model.idx.v, model.gen.v, strict=True))
    for name, (upper, lower) in _GOVERNOR_LIMITS.items():
        gov = getattr(ss, name, None)
        if gov is None or not gov.n:
            continue
        hi, lo = gov.params[upper].v, gov.params[lower].v
        at = np.asarray(gov.At.v, dtype=float) if name == "HYGOV" else np.ones(gov.n)
        for k, syn in enumerate(gov.syn.v):
            d = dp.get(gen_of.get(syn), 0.0) / at[k]
            if d:
                hi[k] += d
                lo[k] = lo[k] + d if lo[k] < 0 else max(lo[k] + d, 0.0)


def clamped_governors(ss: Any) -> list[tuple[str, Any, str]]:
    """(model, idx, limit) of each governor variable held at a limit after ``TDS.init``."""
    out: list[tuple[str, Any, str]] = []
    for name in _GOVERNOR_LIMITS:
        gov = getattr(ss, name, None)
        if gov is None or not gov.n:
            continue
        for item in gov.discrete.values():
            for row in item.get_limit_report():
                out.append((name, row["idx"], str(row["limit_name"])))
    return out


def _record_init_clamps(ss: Any) -> list[tuple[str, Any, str]]:
    """List that ``TDS.run`` fills with :func:`clamped_governors` at its initialisation test.

    The check runs inside ANDES' own initialisation test (``TDS.test_init``), so ``TDS.run``
    keeps its usual path (a separate ``TDS.init`` call would make it resume and drop t = 0).
    """
    found: list[tuple[str, Any, str]] = []
    test_init = ss.TDS.test_init

    def checked() -> bool:
        ok = bool(test_init())
        found.extend(clamped_governors(ss))
        return ok

    ss.TDS.test_init = checked
    return found


@dataclass
class AndesEngine:
    """Run fault scenarios with ANDES.

    Attributes:
        system_name: system key (only used for naming and metadata if ``case`` is given).
        case: ANDES case file; defaults to the built-in case of ``system_name``.
        simulation_time_s: simulated duration.
        step_s: output and integration step.
        name_width: zero padding of bus and generator numbers in channel names.
        n_jobs: parallel worker processes.
        record_buses: ``"all"`` or ``"generators"`` (only buses with a generator; for
            large grids, where recording every bus would not fit in memory).
        shift_governor_limits: after a scenario scales the dispatch (``operating_point`` or
            ``load_factors``), move each governor's limits (TGOV1 VMAX/VMIN, HYGOV GMAX/GMIN) with
            its unit's setpoint, so the band the case builder set around the solved power stays
            around the scaled power and no governor starts clamped at a limit. False reproduces
            the v4.1 events (limits fixed at the base dispatch; test e10, hygov check).
    """

    system_name: str = "new_england"
    case: str | Path | None = None
    simulation_time_s: float = 10.0
    step_s: float = 0.01
    name_width: int = 2
    n_jobs: int = 1
    record_buses: Literal["all", "generators"] = "all"
    shift_governor_limits: bool = True
    _lines: list[Any] = field(default_factory=list, init=False, repr=False)
    _loads: list[Any] = field(default_factory=list, init=False, repr=False)
    _frequency_hz: float = field(default=50.0, init=False, repr=False)
    _branches: list[tuple[Any, Any, Any, bool]] = field(
        default_factory=list, init=False, repr=False
    )
    _bus_ids: list[Any] = field(default_factory=list, init=False, repr=False)
    _gen_buses: set[Any] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.step_s <= 0 or self.simulation_time_s <= self.step_s:
            raise ConfigurationError("need 0 < step_s < simulation_time_s")
        if self.record_buses not in ("all", "generators"):
            raise ConfigurationError(f"unknown record_buses {self.record_buses!r}")
        if self.case is None and self.system.key not in DEFAULT_CASES:
            raise ConfigurationError(f"no built-in ANDES case for {self.system.key!r}; pass case=")
        ss = self._load(setup=True)
        self._lines = [
            i for i, trans in zip(ss.Line.idx.v, ss.Line.trans.v, strict=True) if not trans
        ]
        self._loads = list(ss.PQ.idx.v)
        # a case file stores fn per device, not System.freq; take it from the machines
        fn = [f for name in _MACHINES for f in getattr(ss, name).fn.v]
        if self.case is None or not fn or len(set(fn)) > 1:
            self._frequency_hz = float(ss.config.freq)
        else:
            self._frequency_hz = float(fn[0])
        self._bus_ids = list(ss.Bus.idx.v)
        self._gen_buses = set(ss.PV.bus.v) | set(ss.Slack.bus.v)
        for name in _MACHINES:
            self._gen_buses |= set(getattr(ss, name).bus.v)
        self._branches = [
            (idx, b1, b2, bool(u))
            for idx, b1, b2, u in zip(
                ss.Line.idx.v, ss.Line.bus1.v, ss.Line.bus2.v, ss.Line.u.v, strict=True
            )
        ]

    @property
    def system(self) -> PowerSystem:
        return get_system(self.system_name)

    @property
    def case_path(self) -> str:
        andes = require("andes", "andes")
        if self.case is not None:
            return str(self.case)
        if self.system.key == "germany":
            from tranpy.simulation.cases import germany_case_path

            return str(germany_case_path())
        return str(andes.get_case(DEFAULT_CASES[self.system.key]))

    @property
    def n_lines(self) -> int:
        return len(self._lines)

    @property
    def n_loads(self) -> int:
        return len(self._loads)

    @property
    def nominal_frequency_hz(self) -> float:
        return self._frequency_hz

    @property
    def line_neighbours(self) -> tuple[tuple[int, ...], ...]:
        """For each fault-candidate line, the candidate lines sharing a bus with it."""
        ends = {idx: {b1, b2} for idx, b1, b2, _ in self._branches}
        return tuple(
            tuple(j for j, other in enumerate(self._lines) if j != i and ends[line] & ends[other])
            for i, line in enumerate(self._lines)
        )

    def _components(self, removed: set[Any]) -> list[list[Any]]:
        parent = {b: b for b in self._bus_ids}

        def root(b: Any) -> Any:
            while parent[b] != b:
                parent[b] = parent[parent[b]]
                b = parent[b]
            return b

        for idx, b1, b2, in_service in self._branches:
            if in_service and idx not in removed:
                parent[root(b1)] = root(b2)
        groups: dict[Any, list[Any]] = {}
        for b in self._bus_ids:
            groups.setdefault(root(b), []).append(b)
        return list(groups.values())

    def _recorded(self, bus: Any) -> bool:
        return self.record_buses == "all" or bus in self._gen_buses

    def islands(self, scenario: FaultScenario) -> bool:
        """Whether opening the scenario's lines splits the network into separate parts."""
        return self._separation(scenario) is not None

    def _separation(self, scenario: FaultScenario) -> tuple[float, list[Any], bool] | None:
        """Split time, cut-off buses and whether a cut-off part has a generator (None: no split)."""
        removed = {self._lines[scenario.fault_line]}
        t_split = scenario.fault_clear_s
        parts = self._components(removed)
        if len(parts) == 1 and scenario.trip_line is not None:
            removed.add(self._lines[scenario.trip_line])
            t_split = scenario.fault_clear_s + scenario.trip_delay_s
            parts = self._components(removed)
        if len(parts) == 1:
            return None
        main = max(parts, key=lambda part: (len(self._gen_buses & set(part)), len(part)))
        cut = [b for part in parts if part is not main for b in part]
        return t_split, cut, bool(self._gen_buses & set(cut))

    def _load(self, *, setup: bool) -> Any:
        _load_openmp_libraries_first()
        andes = require("andes", "andes")
        andes.config_logger(stream_level=logging.ERROR)
        with warnings.catch_warnings():
            # ANDES itself still uses pandas and NumPy APIs that are deprecated upstream
            warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"andes(\.|$)")
            warnings.filterwarnings("ignore", category=FutureWarning, module=r"andes(\.|$)")
            frequency = (
                _CASE_FREQUENCY_HZ.get(self.system.key)
                if self.case is None
                else getattr(self, "_frequency_hz", None)
            )
            options = [f"System.freq={frequency}"] if frequency else None
            ss = andes.load(
                self.case_path,
                setup=False,
                no_output=True,
                default_config=True,
                config_option=options,
            )
            if frequency:  # ANDES defaults each device's fn to 60 Hz
                for model in ss.models.values():
                    if model.n and "fn" in model.params:
                        model.params["fn"].v[:] = [frequency] * model.n
            if setup:
                ss.setup()
            return ss

    def simulate(self, scenarios: Sequence[FaultScenario]) -> list[SimulationResult]:
        """Simulate the scenarios (several seconds each), in parallel if ``n_jobs > 1``."""
        if self.n_jobs <= 1 or len(scenarios) < 2:
            return [self._simulate_one(s) for s in scenarios]
        with ProcessPoolExecutor(self.n_jobs) as pool:
            return list(pool.map(self._simulate_one, scenarios, chunksize=1))

    def _simulate_one(self, scenario: FaultScenario) -> SimulationResult:
        lines = self._lines
        for index in (scenario.fault_line, scenario.trip_line):
            if index is not None and not 0 <= index < len(lines):
                raise ConfigurationError(
                    f"scenario {scenario.scenario_id}: line {index} outside the network"
                )
        if not 0 <= scenario.load_index < len(self._loads):
            raise ConfigurationError(f"scenario {scenario.scenario_id}: load outside the network")
        separation = self._separation(scenario)
        if separation is not None and separation[2]:
            logger.info(
                "scenario %d: a part with a generator is cut off; not simulated",
                scenario.scenario_id,
            )
            return SimulationResult(
                scenario=scenario,
                time=np.array([0.0, self.step_s]),
                values=np.zeros((2, 1)),
                channel_names=("not simulated|generator island",),
                completed=False,
            )
        ss = self._load(setup=False)
        n_case_toggles = ss.Toggle.n
        if scenario.fault_type != "three_phase":
            raise ConfigurationError(
                f"scenario {scenario.scenario_id}: ANDES models three-phase faults only"
            )
        line = lines[scenario.fault_line]
        terminals = ss.Line.bus2 if scenario.fault_position >= 0.5 else ss.Line.bus1
        bus = terminals.v[ss.Line.idx.v.index(line)]
        vn_kv = ss.Bus.Vn.v[ss.Bus.idx.v.index(bus)]
        z_base = vn_kv**2 / ss.config.mva
        ss.add(
            "Fault",
            {
                "bus": bus,
                "tf": scenario.fault_start_s,
                "tc": scenario.fault_clear_s,
                "rf": scenario.fault_resistance_ohm / z_base,
                "xf": max(scenario.fault_reactance_ohm / z_base, MIN_FAULT_REACTANCE_PU),
            },
        )
        ss.add("Toggle", {"model": "Line", "dev": line, "t": scenario.fault_clear_s})
        if scenario.trip_line is not None:
            ss.add(
                "Toggle",
                {
                    "model": "Line",
                    "dev": lines[scenario.trip_line],
                    "t": scenario.fault_clear_s + scenario.trip_delay_s,
                },
            )
        if separation is not None:
            t_split, cut, _ = separation
            for model in ("PQ", "Shunt"):
                devices = ss.models[model]
                for idx, bus in zip(devices.idx.v, devices.bus.v, strict=True):
                    if bus in cut:
                        ss.add("Toggle", {"model": model, "dev": idx, "t": t_split})
            for bus in cut:
                ss.add("Toggle", {"model": "Bus", "dev": bus, "t": t_split})
        monitored = set(ss.BusFreq.bus.v)
        for b in ss.Bus.idx.v:
            if b not in monitored and self._recorded(b):
                ss.add("BusFreq", {"bus": b, "fn": self._frequency_hz})
        ss.setup()
        for k in range(
            n_case_toggles
        ):  # events stored in the case file are not part of the scenario
            ss.Toggle.u.v[k] = 0
        load_pos = ss.PQ.idx.v.index(self._loads[scenario.load_index])
        pv_base = np.array(ss.PV.p0.v, dtype=float)
        if scenario.load_factors is not None:
            if len(scenario.load_factors) != len(self._loads):
                raise ConfigurationError(
                    f"scenario {scenario.scenario_id}: {len(scenario.load_factors)} load factors "
                    f"for {len(self._loads)} loads"
                )
            base_demand = float(np.sum(ss.PQ.p0.v))
            for idx, factor in zip(self._loads, scenario.load_factors, strict=True):
                k = ss.PQ.idx.v.index(idx)
                ss.PQ.p0.v[k] *= factor
                ss.PQ.q0.v[k] *= factor
            ss.PV.p0.v[:] *= float(np.sum(ss.PQ.p0.v)) / base_demand  # proportional re-dispatch
        else:
            ss.PQ.p0.v[:] *= scenario.operating_point
            ss.PQ.q0.v[:] *= scenario.operating_point
            ss.PV.p0.v[:] *= scenario.operating_point
        ss.PQ.p0.v[load_pos] *= 1.0 + scenario.load_change_pct / 100.0
        ss.PQ.q0.v[load_pos] *= 1.0 + scenario.load_change_pct / 100.0
        if self.shift_governor_limits:
            shift_governor_limits(ss, pv_base)
        if not ss.PFlow.run() or float(np.min(ss.Bus.v.v)) < MIN_PREFAULT_VOLTAGE_PU:
            logger.info("scenario %d: infeasible pre-fault operating point", scenario.scenario_id)
            return SimulationResult(
                scenario=scenario,
                time=np.array([0.0, self.step_s]),
                values=np.zeros((2, 1)),
                channel_names=("not simulated|infeasible operating point",),
                completed=False,
            )
        if self.shift_governor_limits:
            shift_governor_limits(ss, pv_base, slack=True)
        ss.TDS.config.tf = self.simulation_time_s
        ss.TDS.config.tstep = self.step_s
        ss.TDS.config.no_tqdm = 1
        ss.TDS.config.criteria = 0
        clamped = _record_init_clamps(ss)
        ss.TDS.run()
        if clamped:
            logger.warning(
                "scenario %d: %d governor(s) clamped at a limit at initialisation: %s",
                scenario.scenario_id,
                len(clamped),
                ", ".join(f"{m} {i} {lim}" for m, i, lim in clamped[:5]),
            )
        return self._result(ss, scenario)

    def _result(self, ss: Any, scenario: FaultScenario) -> SimulationResult:
        ts = ss.dae.ts
        time = np.asarray(ts.t, dtype=np.float64)
        keep = np.isfinite(ts.y).all(axis=1) & np.isfinite(ts.x).all(axis=1)
        if keep.sum() < 2:
            raise SimulationError(f"scenario {scenario.scenario_id}: no valid simulation output")
        x, y, time = ts.x[keep], ts.y[keep], time[keep]
        f0, sb = self._frequency_hz, float(ss.config.mva)
        bus_order = np.argsort(np.asarray(ss.Bus.idx.v, dtype=float))
        freq_by_bus = {b: i for i, b in enumerate(ss.BusFreq.bus.v)}
        v = y[:, np.asarray(ss.Bus.v.a)][:, bus_order]
        a = np.degrees(y[:, np.asarray(ss.Bus.a.a)][:, bus_order])
        f_all = y[:, np.asarray(ss.BusFreq.f.a)]
        recorded = [k for k, i in enumerate(bus_order) if self._recorded(ss.Bus.idx.v[i])]
        bus_order, v, a = bus_order[recorded], v[:, recorded], a[:, recorded]
        bus_ids = [ss.Bus.idx.v[i] for i in bus_order]
        f = f_all[:, [freq_by_bus[b] for b in bus_ids]] * f0
        machines = [getattr(ss, name) for name in _MACHINES if getattr(ss, name).n]
        delta = np.hstack([x[:, np.asarray(m.delta.a)] for m in machines])
        omega = np.hstack([x[:, np.asarray(m.omega.a)] for m in machines])
        inertia = np.concatenate([np.asarray(m.M.v, dtype=np.float64) for m in machines])
        relative = delta - (delta * inertia).sum(axis=1, keepdims=True) / inertia.sum()
        out_of_step = np.maximum.accumulate(np.abs(relative) > np.pi, axis=0).astype(np.float64)
        pe = np.hstack([y[:, np.asarray(m.Pe.a)] for m in machines]) * sb
        qe = np.hstack([y[:, np.asarray(m.Qe.a)] for m in machines]) * sb
        w = self.name_width
        names: list[str] = []
        columns: list[npt.NDArray[np.float64]] = []
        for k, b in enumerate(bus_ids):
            bus = f"Bus {int(b):0{w}d}" if str(b).isdigit() else f"Bus {k + 1:0{w}d}"
            for var, data in (
                (Variable.BUS_VOLTAGE_PU, v[:, k]),
                (Variable.BUS_ANGLE_DEG, a[:, k]),
                (Variable.BUS_FREQUENCY_HZ, f[:, k]),
            ):
                names.append(f"{bus}|{var.value}")
                columns.append(data)
        for g in range(delta.shape[1]):
            gen = f"G {g + 1:0{w}d}"
            for var, data in (
                (Variable.ROTOR_SPEED_PU, omega[:, g]),
                (Variable.ACTIVE_POWER_MW, pe[:, g]),
                (Variable.REACTIVE_POWER_MVAR, qe[:, g]),
                (Variable.OUT_OF_STEP, out_of_step[:, g]),
                (Variable.ROTOR_ANGLE_DEG, np.degrees(relative[:, g])),
                (Variable.ROTOR_FREQUENCY_DEVIATION_HZ, (omega[:, g] - 1.0) * f0),
            ):
                names.append(f"{gen}|{var.value}")
                columns.append(data)
        values = np.column_stack(columns)
        finished = time[-1] >= self.simulation_time_s - 1.5 * self.step_s
        decided = finished or bool(out_of_step.any())
        if not finished:
            logger.info(
                "scenario %d: solver stopped at %.2f s (outcome decided: %s)",
                scenario.scenario_id,
                time[-1],
                decided,
            )
        return SimulationResult(
            scenario=scenario,
            time=time,
            values=values,
            channel_names=tuple(names),
            completed=decided,
        )
