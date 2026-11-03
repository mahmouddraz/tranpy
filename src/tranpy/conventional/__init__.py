"""Conventional (non-ML) transient stability methods, benchmarked against the ML pipeline.

See ``adrc/`` for the specification: M1 equal-area criterion (:mod:`.eac`), with the network
reduction shared by M1 and M3 in :mod:`.network`.
"""

from __future__ import annotations

from .eac import (
    ConventionalPrediction,
    EqualAreaCriterion,
    SmibCurve,
    SmibEvent,
    critical_clearing_angle,
    critical_clearing_time,
    smib_curve,
)
from .network import Branches, admittance_matrix, internal_node_admittance, kron_reduce, with_shunt

__all__ = [
    "Branches",
    "ConventionalPrediction",
    "EqualAreaCriterion",
    "SmibCurve",
    "SmibEvent",
    "admittance_matrix",
    "critical_clearing_angle",
    "critical_clearing_time",
    "internal_node_admittance",
    "kron_reduce",
    "smib_curve",
    "with_shunt",
]
