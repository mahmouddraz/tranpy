"""Scenario protocols: how fault scenarios are sampled for a dataset.

Three protocols are available (see ``docs/scenario_design.md``):

* ``baseline`` - :class:`~tranpy.simulation.engine.ScenarioSampler`, the setup of the
  PowerFactory campaigns behind the bundled datasets (one load step at t = 0, fault at a
  random time, fixed clearing time, optional second trip of the next line in the list,
  label from the out-of-step flag). Unchanged, so the bundled datasets can be reproduced
  and compared with the other protocols.
* ``stress_test`` - :class:`StressTestSampler`: broad coverage for learning the stability
  boundary. Loading and dispatch sampled before the power flow, fault on a settled system
  at a fixed time, worst-case three-phase faults, clearing times over a wide range.
* ``normal_operation`` - :class:`NormalOperationSampler`: operating points from load profiles (e.g.
  historical time series), fault types by observed frequency, locations weighted by line
  length, clearing times from protection timing (mostly primary, sometimes backup).

All samplers return :class:`~tranpy.simulation.engine.FaultScenario` tuples, so every
engine runs every protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from tranpy.simulation.engine import FaultScenario
from tranpy_core.errors import ConfigurationError
from tranpy_core.random import Seed, make_rng

__all__ = ["PROTOCOLS", "NormalOperationSampler", "StressTestSampler"]

PROTOCOLS: dict[str, str] = {
    "baseline": "setup of the bundled PowerFactory datasets (ScenarioSampler, label out_of_step)",
    "stress_test": "broad coverage of loading, fault location and clearing time (label angle_360)",
    "normal_operation": "load profiles, fault statistics and protection timing (label angle_360)",
}


def _check_range(name: str, bounds: tuple[float, float], low: float, problems: list[str]) -> None:
    a, b = bounds
    if not low <= a <= b:
        problems.append(f"{name} must satisfy {low} <= low <= high, got {bounds}")


def _neighbour(
    rng: np.random.Generator, line: int, neighbours: tuple[tuple[int, ...], ...] | None
) -> int | None:
    if neighbours is None or not neighbours[line]:
        return None
    return int(rng.choice(neighbours[line]))


@dataclass(frozen=True)
class StressTestSampler:
    """Broad, boundary-covering scenarios (protocol ``stress_test``).

    Attributes:
        n_lines: fault-candidate lines of the engine.
        n_loads: loads of the engine.
        loading: (low, high) range of the system loading factor.
        load_noise: standard deviation of the independent per-load deviation.
        clearing_s: (low, high) uniform range of the fault duration.
        fault_time_s: fault inception on the settled system.
        post_clearing_s: simulated time after the latest possible clearing.
        fault_position: ``"terminal"`` (0 or 1 at random) or ``"along"`` (uniform 0..1).
        second_trip: also trip a line sharing a bus with the faulted one.
        line_neighbours: from the engine (``engine.line_neighbours``), for ``second_trip``.
        fault_impedance_ohm: (low, high) range of the fault reactance; resistance 0.
    """

    n_lines: int
    n_loads: int
    loading: tuple[float, float] = (0.8, 1.2)
    load_noise: float = 0.05
    clearing_s: tuple[float, float] = (0.05, 0.40)
    fault_time_s: float = 1.0
    post_clearing_s: float = 5.0
    fault_position: Literal["terminal", "along"] = "terminal"
    second_trip: bool = False
    line_neighbours: tuple[tuple[int, ...], ...] | None = None
    fault_impedance_ohm: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        problems: list[str] = []
        if self.n_lines < 1 or self.n_loads < 1:
            problems.append("need at least one line and one load")
        _check_range("loading", self.loading, 1e-6, problems)
        _check_range("clearing_s", self.clearing_s, 1e-6, problems)
        _check_range("fault_impedance_ohm", self.fault_impedance_ohm, 0.0, problems)
        if self.load_noise < 0:
            problems.append("load_noise must be >= 0")
        if self.fault_time_s <= 0 or self.post_clearing_s <= 0:
            problems.append("fault_time_s and post_clearing_s must be > 0")
        if self.fault_position not in ("terminal", "along"):
            problems.append(f"unknown fault_position {self.fault_position!r}")
        if self.second_trip and self.line_neighbours is None:
            problems.append("second_trip needs line_neighbours")
        if problems:
            raise ConfigurationError("invalid stress-test sampler:", tuple(problems))

    @property
    def simulation_time_s(self) -> float:
        """Duration that covers every sampled scenario."""
        return self.fault_time_s + self.clearing_s[1] + self.post_clearing_s

    def sample(self, n: int, *, seed: Seed, first_id: int = 0) -> tuple[FaultScenario, ...]:
        rng = make_rng(seed)
        loading = rng.uniform(*self.loading, size=n)
        noise = np.clip(rng.normal(1.0, self.load_noise, size=(n, self.n_loads)), 0.05, None)
        lines = rng.integers(0, self.n_lines, size=n)
        durations = rng.uniform(*self.clearing_s, size=n)
        positions = (
            rng.integers(0, 2, size=n).astype(float)
            if self.fault_position == "terminal"
            else rng.uniform(0.0, 1.0, size=n)
        )
        reactance = rng.uniform(*self.fault_impedance_ohm, size=n)
        scenarios = []
        for i in range(n):
            trip = (
                _neighbour(rng, int(lines[i]), self.line_neighbours) if self.second_trip else None
            )
            scenarios.append(
                FaultScenario(
                    scenario_id=first_id + i,
                    fault_line=int(lines[i]),
                    fault_start_s=self.fault_time_s,
                    clearing_duration_s=float(durations[i]),
                    fault_resistance_ohm=0.0,
                    fault_reactance_ohm=float(reactance[i]),
                    trip_line=trip,
                    operating_point=float(loading[i]),
                    load_factors=tuple(float(f) for f in loading[i] * noise[i]),
                    fault_position=float(positions[i]),
                )
            )
        return tuple(scenarios)


@dataclass(frozen=True)
class NormalOperationSampler:
    """Scenarios with the frequencies of normal operation (protocol ``normal_operation``).

    Attributes:
        n_lines: fault-candidate lines of the engine.
        load_profiles: ``(n_times, n_loads)`` demand factors relative to the base case, one
            row per time step of a load time series; each scenario uses a random row.
        line_weights: relative fault frequency per line (e.g. line length); uniform if None.
        single_phase_share: share of single-phase-to-ground faults (the rest three-phase).
        primary_clearing_s: (low, high) fault duration when primary protection operates.
        backup_clearing_s: (low, high) fault duration when backup protection operates.
        backup_share: share of faults cleared by backup protection.
        fault_impedance_ohm: (low, high) range of the fault resistance.
        fault_time_s: fault inception on the settled system.
        post_clearing_s: simulated time after the latest possible clearing.
    """

    n_lines: int
    load_profiles: npt.NDArray[np.float64]
    line_weights: Sequence[float] | None = None
    single_phase_share: float = 0.8
    primary_clearing_s: tuple[float, float] = (0.08, 0.15)
    backup_clearing_s: tuple[float, float] = (0.30, 0.50)
    backup_share: float = 0.05
    fault_impedance_ohm: tuple[float, float] = (0.0, 5.0)
    fault_time_s: float = 1.0
    post_clearing_s: float = 5.0
    _weights: npt.NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        problems: list[str] = []
        profiles = np.asarray(self.load_profiles, dtype=np.float64)
        if profiles.ndim != 2 or profiles.shape[0] < 1 or np.any(profiles <= 0):
            problems.append("load_profiles must be a positive (n_times, n_loads) array")
        weights = (
            np.ones(self.n_lines)
            if self.line_weights is None
            else np.asarray(self.line_weights, float)
        )
        if weights.shape != (self.n_lines,) or np.any(weights < 0) or weights.sum() <= 0:
            problems.append("line_weights needs one non-negative weight per line")
        for name in ("single_phase_share", "backup_share"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                problems.append(f"{name} must lie in [0, 1]")
        _check_range("primary_clearing_s", self.primary_clearing_s, 1e-6, problems)
        _check_range("backup_clearing_s", self.backup_clearing_s, 1e-6, problems)
        _check_range("fault_impedance_ohm", self.fault_impedance_ohm, 0.0, problems)
        if problems:
            raise ConfigurationError("invalid normal-operation sampler:", tuple(problems))
        object.__setattr__(self, "load_profiles", profiles)
        object.__setattr__(self, "_weights", weights / weights.sum())

    @property
    def n_loads(self) -> int:
        return int(self.load_profiles.shape[1])

    @property
    def simulation_time_s(self) -> float:
        return (
            self.fault_time_s
            + max(self.primary_clearing_s[1], self.backup_clearing_s[1])
            + self.post_clearing_s
        )

    def sample(self, n: int, *, seed: Seed, first_id: int = 0) -> tuple[FaultScenario, ...]:
        rng = make_rng(seed)
        rows = rng.integers(0, self.load_profiles.shape[0], size=n)
        lines = rng.choice(self.n_lines, size=n, p=self._weights)
        single = rng.random(n) < self.single_phase_share
        backup = rng.random(n) < self.backup_share
        primary = rng.uniform(*self.primary_clearing_s, size=n)
        late = rng.uniform(*self.backup_clearing_s, size=n)
        positions = rng.uniform(0.0, 1.0, size=n)
        resistance = rng.uniform(*self.fault_impedance_ohm, size=n)
        base = self.load_profiles.mean()
        return tuple(
            FaultScenario(
                scenario_id=first_id + i,
                fault_line=int(lines[i]),
                fault_start_s=self.fault_time_s,
                clearing_duration_s=float(late[i] if backup[i] else primary[i]),
                fault_resistance_ohm=float(resistance[i]),
                fault_reactance_ohm=0.0,
                operating_point=float(self.load_profiles[rows[i]].mean() / base),
                load_factors=tuple(float(f) for f in self.load_profiles[rows[i]]),
                fault_type="single_phase" if single[i] else "three_phase",
                fault_position=float(positions[i]),
            )
            for i in range(n)
        )
