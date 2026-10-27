"""Simulation interface: fault scenarios in, trajectories out.

Every engine (the always-available :class:`~tranpy.simulation.mock.MockEngine` and the
optional :class:`~tranpy.simulation.powerfactory.PowerFactoryEngine`) implements
:class:`SimulationEngine`. Dataset generation (:func:`generate_events`) depends only on
this interface. Because a simulation knows its own event times exactly, generated
datasets carry the *true* fault start and clearing times as annotations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from tranpy.datasets import FAULT_CLEAR, FAULT_START, stability_labels
from tranpy.systems import PowerSystem
from tranpy.vocabulary import LABEL_NAMES
from tranpy_core.data import SignalDataset
from tranpy_core.errors import ConfigurationError, SimulationError
from tranpy_core.random import Seed, make_rng

__all__ = [
    "FaultScenario",
    "ScenarioSampler",
    "SimulationEngine",
    "SimulationResult",
    "generate_events",
]


@dataclass(frozen=True)
class FaultScenario:
    """One contingency: a short circuit on a line, cleared by opening that line.

    Attributes:
        scenario_id: identifier, becomes the sample id of the generated event.
        fault_line: index of the faulted line in the engine's line list.
        fault_start_s: fault inception time.
        clearing_duration_s: time from inception until the faulted line is opened.
        fault_resistance_ohm: fault resistance.
        fault_reactance_ohm: fault reactance.
        load_index: index of the load whose demand changes at t = 0.
        load_change_pct: step change of that load in percent of its demand.
        trip_line: index of a line tripped ``trip_delay_s`` after clearing, or None.
        trip_delay_s: delay of the additional line trip.
        operating_point: system loading relative to the base case. Supported by the mock
            engine; the PowerFactory engine currently requires 1.0.
    """

    scenario_id: int
    fault_line: int
    fault_start_s: float
    clearing_duration_s: float
    fault_resistance_ohm: float = 0.01
    fault_reactance_ohm: float = 0.01
    load_index: int = 0
    load_change_pct: float = 0.0
    trip_line: int | None = None
    trip_delay_s: float = 0.08
    operating_point: float = 1.0

    def __post_init__(self) -> None:
        problems = []
        if self.operating_point <= 0:
            problems.append(f"operating_point must be > 0, got {self.operating_point}")
        if self.fault_line < 0:
            problems.append(f"fault_line must be >= 0, got {self.fault_line}")
        if self.fault_start_s < 0:
            problems.append(f"fault_start_s must be >= 0, got {self.fault_start_s}")
        if self.clearing_duration_s <= 0:
            problems.append(f"clearing_duration_s must be > 0, got {self.clearing_duration_s}")
        if self.fault_resistance_ohm < 0 or self.fault_reactance_ohm < 0:
            problems.append("fault impedance must be non-negative")
        if self.trip_line is not None and self.trip_line == self.fault_line:
            problems.append("trip_line must differ from fault_line")
        if problems:
            raise ConfigurationError(f"invalid scenario {self.scenario_id}:", tuple(problems))

    @property
    def fault_clear_s(self) -> float:
        return self.fault_start_s + self.clearing_duration_s


@dataclass(frozen=True)
class ScenarioSampler:
    """Draws random scenarios for an engine's network.

    Attributes:
        n_lines: number of lines that can be faulted.
        n_loads: number of loads that can change.
        simulation_time_s: simulated duration; faults start in
            ``fault_start_fraction`` of it.
        clearing_cycles: fault duration in cycles of the nominal frequency.
        nominal_frequency_hz: converts cycles to seconds.
        max_load_change_pct: load changes are integers in [-max, +max] percent.
        fault_start_fraction: interval of the fault start as a fraction of the duration.
        trip_adjacent_line: also trip the next line in the list after clearing.
        fault_resistance_ohm: (low, high) uniform range.
        fault_reactance_ohm: (low, high) uniform range.
        operating_point: (low, high) uniform range of the system loading.
    """

    n_lines: int
    n_loads: int
    simulation_time_s: float = 10.0
    clearing_cycles: float = 10.0
    nominal_frequency_hz: float = 50.0
    max_load_change_pct: int = 60
    fault_start_fraction: tuple[float, float] = (0.0, 0.5)
    trip_adjacent_line: bool = False
    fault_resistance_ohm: tuple[float, float] = (0.01, 0.01)
    fault_reactance_ohm: tuple[float, float] = (0.01, 0.01)
    operating_point: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        problems = []
        if self.n_lines < 1 + int(self.trip_adjacent_line):
            problems.append("not enough lines")
        if self.n_loads < 1:
            problems.append("at least one load is required")
        if (
            self.clearing_cycles <= 0
            or self.nominal_frequency_hz <= 0
            or self.simulation_time_s <= 0
        ):
            problems.append(
                "clearing_cycles, nominal_frequency_hz and simulation_time_s must be > 0"
            )
        lo, hi = self.fault_start_fraction
        if not 0.0 <= lo <= hi < 1.0:
            problems.append(
                "fault_start_fraction must satisfy 0 <= low <= high < 1, "
                f"got {self.fault_start_fraction}"
            )
        if not 0.0 < self.operating_point[0] <= self.operating_point[1]:
            problems.append(
                f"operating_point must satisfy 0 < low <= high, got {self.operating_point}"
            )
        for name in ("fault_resistance_ohm", "fault_reactance_ohm"):
            a, b = getattr(self, name)
            if not 0.0 <= a <= b:
                problems.append(f"{name} must satisfy 0 <= low <= high, got {(a, b)}")
        if self.max_load_change_pct < 0:
            problems.append("max_load_change_pct must be >= 0")
        if problems:
            raise ConfigurationError("invalid scenario sampler:", tuple(problems))

    def sample(self, n: int, *, seed: Seed, first_id: int = 0) -> tuple[FaultScenario, ...]:
        """Draw ``n`` scenarios (vectorised, reproducible for a given seed)."""
        rng = make_rng(seed)
        lo, hi = self.fault_start_fraction
        lines = rng.integers(0, self.n_lines, size=n)
        starts = rng.uniform(lo, hi, size=n) * self.simulation_time_s
        loads = rng.integers(0, self.n_loads, size=n)
        changes = rng.integers(
            -self.max_load_change_pct, self.max_load_change_pct, size=n, endpoint=True
        )
        r_f = rng.uniform(*self.fault_resistance_ohm, size=n)
        x_f = rng.uniform(*self.fault_reactance_ohm, size=n)
        loading = rng.uniform(*self.operating_point, size=n)
        duration = self.clearing_cycles / self.nominal_frequency_hz
        out = []
        for i in range(n):
            trip = None
            if self.trip_adjacent_line:
                trip = int(lines[i] + 1) if lines[i] + 1 < self.n_lines else int(lines[i] - 1)
            out.append(
                FaultScenario(
                    scenario_id=first_id + i,
                    fault_line=int(lines[i]),
                    fault_start_s=float(starts[i]),
                    clearing_duration_s=duration,
                    fault_resistance_ohm=float(r_f[i]),
                    fault_reactance_ohm=float(x_f[i]),
                    load_index=int(loads[i]),
                    load_change_pct=float(changes[i]),
                    trip_line=trip,
                    operating_point=float(loading[i]),
                )
            )
        return tuple(out)


@dataclass(frozen=True, eq=False)
class SimulationResult:
    """Trajectories of one scenario: ``values[k, c]`` is channel ``c`` at ``time[k]``."""

    scenario: FaultScenario
    time: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]
    channel_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.values.shape != (self.time.shape[0], len(self.channel_names)):
            raise SimulationError(
                f"scenario {self.scenario.scenario_id}: values {self.values.shape} do not match "
                f"{self.time.shape[0]} steps x {len(self.channel_names)} channels"
            )


@runtime_checkable
class SimulationEngine(Protocol):
    """What dataset generation needs from a simulator."""

    @property
    def system(self) -> PowerSystem: ...

    @property
    def n_lines(self) -> int: ...

    @property
    def n_loads(self) -> int: ...

    @property
    def simulation_time_s(self) -> float: ...

    def simulate(self, scenarios: Sequence[FaultScenario]) -> list[SimulationResult]: ...


def generate_events(
    engine: SimulationEngine,
    scenarios: Sequence[FaultScenario],
    *,
    name: str | None = None,
) -> SignalDataset:
    """Simulate ``scenarios`` and collect them as a labelled event dataset.

    Labels follow :func:`tranpy.datasets.stability_labels`. The out-of-step channels stay
    in the result (drop them before feature extraction, as :func:`tranpy.load_events`
    does by default). Annotations hold the true event times and scenario parameters.
    """
    if not scenarios:
        raise ConfigurationError("no scenarios to simulate")
    results = engine.simulate(scenarios)
    if len(results) != len(scenarios):
        raise SimulationError(
            f"engine returned {len(results)} results for {len(scenarios)} scenarios"
        )
    channels = results[0].channel_names
    if any(r.channel_names != channels for r in results):
        raise SimulationError("results have inconsistent channels")
    spec = engine.system
    dataset = SignalDataset.from_samples(
        [r.time for r in results],
        [r.values for r in results],
        channel_names=channels,
        labels=np.zeros(len(results), dtype=np.int64),
        label_names=LABEL_NAMES,
        name=name or f"{spec.key}_simulated",
        sample_ids=np.array([s.scenario_id for s in scenarios], dtype=np.int64),
        description=f"{spec.name}: events simulated with {type(engine).__name__}",
        metadata={"system": spec.key, "engine": type(engine).__name__},
    )
    return dataset.replace(labels=stability_labels(dataset)).with_annotations(
        **{
            FAULT_START: [s.fault_start_s for s in scenarios],
            FAULT_CLEAR: [s.fault_clear_s for s in scenarios],
            "fault_line": [s.fault_line for s in scenarios],
            "load_index": [s.load_index for s in scenarios],
            "load_change_pct": [s.load_change_pct for s in scenarios],
            "fault_resistance_ohm": [s.fault_resistance_ohm for s in scenarios],
            "fault_reactance_ohm": [s.fault_reactance_ohm for s in scenarios],
            "trip_line": [np.nan if s.trip_line is None else s.trip_line for s in scenarios],
            "operating_point": [s.operating_point for s in scenarios],
        }
    )
