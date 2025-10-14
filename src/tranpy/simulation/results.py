"""
Data structures for storing PowerFactory simulation results.

This module provides classes for managing simulation results in a format
compatible with the TranPy dataset API.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import pickle
from pathlib import Path


@dataclass
class EventInfo:
    """Information about a single contingency event."""
    event_id: int
    event_types: List[str]  # e.g., ['short circuit', 'fault clearing', 'load event']
    event_locations: List[str]  # Element names where events occur
    event_times: List[float]  # Event occurrence times in seconds
    fault_time: float  # When fault occurs
    clearing_time: float  # When fault is cleared
    fault_impedance: Dict[str, float]  # R_f and X_f
    load_change: Optional[int] = None  # Load change percentage (if applicable)


@dataclass
class BusSnapshot:
    """Bus voltage and angle data at a specific time point."""
    time: float
    voltages: Dict[str, float]  # bus_name -> voltage magnitude (p.u.)
    angles: Dict[str, float]  # bus_name -> phase angle (degrees)

    def to_feature_array(self) -> np.ndarray:
        """
        Convert to flat feature array [v1, a1, v2, a2, ...].

        Returns:
            1D array of alternating voltage and angle values
        """
        features = []
        # Sort bus names to ensure consistent ordering
        bus_names = sorted(self.voltages.keys())
        for bus_name in bus_names:
            features.append(self.voltages[bus_name])
            features.append(self.angles[bus_name])
        return np.array(features)


@dataclass
class GeneratorSnapshot:
    """Generator state at a specific time point."""
    active_powers: Dict[str, float]  # gen_name -> P (MW)
    reactive_powers: Dict[str, float]  # gen_name -> Q (Mvar)
    out_of_step: Dict[str, int]  # gen_name -> 0 (stable) or 1 (unstable)


@dataclass
class EventResult:
    """Results from a single contingency simulation."""
    event_info: EventInfo
    bus_snapshot_at_fault: BusSnapshot
    bus_snapshot_at_clearing: BusSnapshot
    generator_snapshot: GeneratorSnapshot
    time_series_data: Optional[pd.DataFrame] = None  # Full time series (optional)

    @property
    def is_stable(self) -> bool:
        """Check if system is stable (no generators out of step)."""
        return 1 not in self.generator_snapshot.out_of_step.values()

    @property
    def stability_label(self) -> str:
        """Return 'stable' or 'unstable'."""
        return 'stable' if self.is_stable else 'unstable'


@dataclass
class SimulationResults:
    """
    Complete results from a PowerFactory simulation campaign.

    This class stores all data from multiple contingency simulations
    and provides methods to export in various formats.
    """
    grid_name: str
    simulation_time: float
    events: List[EventResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_event(self, event_result: EventResult):
        """Add results from a single event simulation."""
        self.events.append(event_result)

    def to_dataframe(self, snapshot_type: str = 'clearing') -> pd.DataFrame:
        """
        Convert results to DataFrame format.

        Args:
            snapshot_type: 'fault' or 'clearing' - which snapshot to use

        Returns:
            DataFrame with bus features and stability labels
        """
        if not self.events:
            return pd.DataFrame()

        feature_arrays = []
        labels = []

        for event in self.events:
            # Get the appropriate snapshot
            if snapshot_type == 'fault':
                snapshot = event.bus_snapshot_at_fault
            else:
                snapshot = event.bus_snapshot_at_clearing

            # Convert to feature array
            features = snapshot.to_feature_array()
            feature_arrays.append(features)

            # Add label (0 for stable, 1 for unstable)
            labels.append(0 if event.is_stable else 1)

        # Create DataFrame
        n_features = len(feature_arrays[0])
        feature_names = [f'F_{i}' for i in range(n_features)]

        df = pd.DataFrame(feature_arrays, columns=feature_names)
        df['stable-unstable'] = labels

        return df

    def get_events_dataframe(self) -> pd.DataFrame:
        """
        Get DataFrame with event metadata.

        Returns:
            DataFrame with event information
        """
        data = []
        for event in self.events:
            info = event.event_info
            data.append({
                'events': info.event_types,
                'event_locations': info.event_locations,
                'event_clearing_time': info.clearing_time,
                'event_time': info.event_times,
                'stability_each_generator': list(event.generator_snapshot.out_of_step.values()),
                'system_stability': event.stability_label
            })
        return pd.DataFrame(data)

    def save(self, filepath: Path):
        """
        Save results to pickle file.

        Args:
            filepath: Path to save pickle file
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: Path) -> 'SimulationResults':
        """
        Load results from pickle file.

        Args:
            filepath: Path to pickle file

        Returns:
            SimulationResults object
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of simulation results."""
        total_events = len(self.events)
        stable_events = sum(1 for e in self.events if e.is_stable)
        unstable_events = total_events - stable_events

        return {
            'grid_name': self.grid_name,
            'total_events': total_events,
            'stable_events': stable_events,
            'unstable_events': unstable_events,
            'stability_ratio': stable_events / total_events if total_events > 0 else 0
        }

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"SimulationResults(grid='{stats['grid_name']}', "
            f"n_events={stats['total_events']}, "
            f"stable={stats['stable_events']}, "
            f"unstable={stats['unstable_events']})"
        )


class LegacyDataFormat:
    """
    Compatibility layer for legacy pickle format.

    This class provides methods to convert between the new SimulationResults
    format and the legacy Data class format used in the old codebase.
    """

    @staticmethod
    def to_legacy_format(results: SimulationResults) -> Any:
        """
        Convert SimulationResults to legacy Data class format.

        Args:
            results: SimulationResults object

        Returns:
            Object with structure matching legacy Data class
        """
        from types import SimpleNamespace

        # Create legacy Data object structure
        legacy_data = SimpleNamespace()

        # DataFrames for various measurements
        legacy_data.f1_generator_active_powers = pd.DataFrame([
            e.generator_snapshot.active_powers for e in results.events
        ])

        legacy_data.f2_generator_reactive_powers = pd.DataFrame([
            e.generator_snapshot.reactive_powers for e in results.events
        ])

        legacy_data.f3_out_of_step = pd.DataFrame([
            e.generator_snapshot.out_of_step for e in results.events
        ])

        legacy_data.f4_bus_voltage = pd.DataFrame([
            e.bus_snapshot_at_clearing.voltages for e in results.events
        ])

        legacy_data.f5_bus_angles = pd.DataFrame([
            e.bus_snapshot_at_clearing.angles for e in results.events
        ])

        # Events DataFrame
        legacy_data.df_events = results.get_events_dataframe()

        # Bus data post fault [at_fault, at_clearing]
        legacy_data.bus_data_post_fault = [
            [
                {f'bus_data_post_fault_event_{i}': {
                    'b:tnow in s': e.bus_snapshot_at_fault.time,
                    **{f'{bus}:m:u in p.u.': v for bus, v in e.bus_snapshot_at_fault.voltages.items()},
                    **{f'{bus}:m:phiu in deg': a for bus, a in e.bus_snapshot_at_fault.angles.items()}
                }} for i, e in enumerate(results.events)
            ],
            [
                {f'bus_data_post_fault_clearing_event_{i}': {
                    'b:tnow in s': e.bus_snapshot_at_clearing.time,
                    **{f'{bus}:m:u in p.u.': v for bus, v in e.bus_snapshot_at_clearing.voltages.items()},
                    **{f'{bus}:m:phiu in deg': a for bus, a in e.bus_snapshot_at_clearing.angles.items()}
                }} for i, e in enumerate(results.events)
            ]
        ]

        legacy_data.post_fault_results = []  # Optional: full time series

        return legacy_data

    @staticmethod
    def save_legacy_format(results: SimulationResults, filepath: Path):
        """
        Save in legacy pickle format for backward compatibility.

        Args:
            results: SimulationResults object
            filepath: Path to save pickle file
        """
        legacy_data = LegacyDataFormat.to_legacy_format(results)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(legacy_data, f, protocol=pickle.HIGHEST_PROTOCOL)
