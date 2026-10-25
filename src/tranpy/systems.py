"""Test systems for which datasets are available."""

from __future__ import annotations

from dataclasses import dataclass

from tranpy_core.errors import ConfigurationError

__all__ = ["IEEE_9BUS", "NEW_ENGLAND", "SYSTEMS", "PowerSystem", "get_system"]


@dataclass(frozen=True)
class PowerSystem:
    """Static description of a test system.

    Attributes:
        key: identifier used in file names and APIs.
        name: human-readable name.
        n_buses: number of buses.
        n_generators: number of synchronous generators.
        nominal_frequency_hz: system frequency of the simulation model.
        legacy_name: name used by tranpy <= 0.1 and by the PowerFactory project.
    """

    key: str
    name: str
    n_buses: int
    n_generators: int
    nominal_frequency_hz: float
    legacy_name: str


NEW_ENGLAND = PowerSystem(
    key="new_england",
    name="IEEE 39-bus New England system",
    n_buses=39,
    n_generators=10,
    nominal_frequency_hz=50.0,
    legacy_name="NewEngland",
)
IEEE_9BUS = PowerSystem(
    key="ieee_9bus",
    name="IEEE 9-bus (WSCC) system",
    n_buses=9,
    n_generators=3,
    nominal_frequency_hz=50.0,
    legacy_name="NineBusSystem",
)
SYSTEMS: dict[str, PowerSystem] = {s.key: s for s in (NEW_ENGLAND, IEEE_9BUS)}


def get_system(system: str | PowerSystem) -> PowerSystem:
    """Look up a system by key (``"new_england"``) or legacy name (``"NewEngland"``)."""
    if isinstance(system, PowerSystem):
        return system
    for s in SYSTEMS.values():
        if system in (s.key, s.legacy_name):
            return s
    raise ConfigurationError(f"unknown system {system!r}; available: {', '.join(SYSTEMS)}")
