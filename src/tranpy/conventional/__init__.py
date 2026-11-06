"""Conventional (non-ML) transient stability methods, benchmarked against the ML pipeline.

See ``adrc/`` for the specification: M1 equal-area criterion (:mod:`.eac`), with the network
reduction shared by M1 and M3 in :mod:`.network`; the kinetic-energy part of the M3
transient energy function (:mod:`.energy`) and its CCT-normalised form; CCT brackets from
clearing-time sweeps for M4 (:mod:`.cct`); the SIME single-machine equivalent and its margin
(:mod:`.sime`); single-score threshold baselines (:mod:`.baselines`).
"""

from __future__ import annotations

from .baselines import RuleCandidate, ThresholdRule, f1_threshold, youden_threshold
from .cct import cct_brackets
from .eac import (
    ConventionalPrediction,
    EqualAreaCriterion,
    SmibCurve,
    SmibEvent,
    critical_clearing_angle,
    critical_clearing_time,
    smib_curve,
)
from .energy import (
    cct_normalised_energy,
    coi_kinetic_energy,
    corrected_kinetic_energy,
    critical_group_by_largest_gap,
    kinetic_energy_features,
)
from .network import Branches, admittance_matrix, internal_node_admittance, kron_reduce, with_shunt
from .sime import (
    CriticalCluster,
    OmibTrajectory,
    SimeResult,
    critical_cluster,
    inertia_coefficient,
    omib,
    sime_margin,
)

__all__ = [
    "Branches",
    "ConventionalPrediction",
    "CriticalCluster",
    "EqualAreaCriterion",
    "OmibTrajectory",
    "RuleCandidate",
    "SimeResult",
    "SmibCurve",
    "SmibEvent",
    "ThresholdRule",
    "admittance_matrix",
    "cct_brackets",
    "cct_normalised_energy",
    "coi_kinetic_energy",
    "corrected_kinetic_energy",
    "critical_clearing_angle",
    "critical_clearing_time",
    "critical_cluster",
    "critical_group_by_largest_gap",
    "f1_threshold",
    "inertia_coefficient",
    "internal_node_admittance",
    "kinetic_energy_features",
    "kron_reduce",
    "omib",
    "sime_margin",
    "smib_curve",
    "with_shunt",
    "youden_threshold",
]
