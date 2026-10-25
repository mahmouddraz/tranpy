"""Simulation engines and dataset generation.

``MockEngine`` is always available; ``PowerFactoryEngine`` (in
:mod:`tranpy.simulation.powerfactory`) needs a local PowerFactory installation and is not
imported here.
"""

from tranpy.simulation.engine import (
    FaultScenario,
    ScenarioSampler,
    SimulationEngine,
    SimulationResult,
    generate_events,
)
from tranpy.simulation.mock import MockEngine, MockNetwork

__all__ = [
    "FaultScenario",
    "MockEngine",
    "MockNetwork",
    "ScenarioSampler",
    "SimulationEngine",
    "SimulationResult",
    "generate_events",
]
