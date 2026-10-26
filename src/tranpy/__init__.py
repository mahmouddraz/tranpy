"""Machine learning for power-system transient stability.

The power-system layer on top of :mod:`tranpy_core`: test systems, datasets of simulated
fault events, trajectory features, early-prediction experiments and simulation engines.
"""

from importlib.metadata import PackageNotFoundError, version

from tranpy.datasets import (
    list_datasets,
    load_events,
    load_ieee_9bus,
    load_new_england,
    load_snapshots,
    open_event_shards,
)
from tranpy.features import TrajectoryFeatures
from tranpy.systems import IEEE_9BUS, NEW_ENGLAND, PowerSystem, get_system
from tranpy.vocabulary import LABEL_NAMES, Stability

try:
    __version__ = version("tranpy-ai")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0+unknown"

__all__ = [
    "IEEE_9BUS",
    "LABEL_NAMES",
    "NEW_ENGLAND",
    "PowerSystem",
    "Stability",
    "TrajectoryFeatures",
    "__version__",
    "get_system",
    "list_datasets",
    "load_events",
    "load_ieee_9bus",
    "load_new_england",
    "load_snapshots",
    "open_event_shards",
]
