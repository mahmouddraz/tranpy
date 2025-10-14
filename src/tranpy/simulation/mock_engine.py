"""
Mock simulation engine for testing without PowerFactory.

This module provides a mock implementation that generates synthetic simulation data
with realistic characteristics, allowing users to test the entire pipeline without
requiring PowerFactory installation.
"""

import numpy as np
from typing import List, Optional
from pathlib import Path

from ..utils.logging import get_logger
from .results import (
    EventInfo, BusSnapshot, GeneratorSnapshot, EventResult, SimulationResults
)
from .events import EventGenerator

logger = get_logger(__name__)


class MockSimulationEngine:
    """
    Mock simulation engine that generates synthetic stability data.

    This engine mimics PowerFactory's behavior but generates data algorithmically,
    allowing for testing and development without PowerFactory installation.

    Args:
        grid: Grid name ('NewEngland' or 'NineBusSystem')
        simulation_time: Simulation duration in seconds
        random_seed: Random seed for reproducibility

    Examples:
        >>> engine = MockSimulationEngine('NewEngland', simulation_time=10.0)
        >>> results = engine.run(num_events=100)
    """

    def __init__(
        self,
        grid: str = 'NewEngland',
        simulation_time: float = 10.0,
        random_seed: Optional[int] = None
    ):
        self.grid = grid
        self.simulation_time = simulation_time
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

        # Grid parameters
        if grid == 'NewEngland':
            self.n_buses = 39
            self.n_generators = 10
            self.n_lines = 46
            self.n_loads = 19
        elif grid == 'NineBusSystem':
            self.n_buses = 9
            self.n_generators = 3
            self.n_lines = 9
            self.n_loads = 3
        else:
            raise ValueError(f"Unknown grid: {grid}")

    def _generate_bus_snapshot(
        self,
        time: float,
        base_stable: bool,
        severity: float = 0.0
    ) -> BusSnapshot:
        """
        Generate bus voltage and angle snapshot.

        Args:
            time: Snapshot time
            base_stable: Whether system is fundamentally stable
            severity: Disturbance severity (0.0 to 1.0)

        Returns:
            BusSnapshot object
        """
        voltages = {}
        angles = {}

        for i in range(1, self.n_buses + 1):
            bus_name = f"Bus_{i}"

            if base_stable:
                # Stable: voltages near 1.0, small angle deviations
                v_mean = 1.0
                v_std = 0.02 + severity * 0.05  # More variation with higher severity
                a_mean = 0.0
                a_std = 5.0 + severity * 10.0
            else:
                # Unstable: larger voltage drops, wider angle swings
                v_mean = 0.85 - severity * 0.2
                v_std = 0.15
                a_mean = 0.0
                a_std = 20.0 + severity * 20.0

            # Add some spatial correlation (nearby buses similar)
            spatial_offset = np.sin(i / self.n_buses * 2 * np.pi) * severity * 0.1

            voltages[bus_name] = np.clip(
                np.random.normal(v_mean + spatial_offset, v_std),
                0.5, 1.2
            )
            angles[bus_name] = np.random.normal(a_mean, a_std)

        return BusSnapshot(time=time, voltages=voltages, angles=angles)

    def _generate_generator_snapshot(
        self,
        base_stable: bool,
        severity: float = 0.0
    ) -> GeneratorSnapshot:
        """
        Generate generator state snapshot.

        Args:
            base_stable: Whether system is fundamentally stable
            severity: Disturbance severity

        Returns:
            GeneratorSnapshot object
        """
        active_powers = {}
        reactive_powers = {}
        out_of_step = {}

        for i in range(1, self.n_generators + 1):
            gen_name = f"Gen_{i}"

            # Power output
            if base_stable:
                p_base = 100 + i * 50
                q_base = 20 + i * 10
            else:
                p_base = (100 + i * 50) * (0.7 - severity * 0.3)
                q_base = (20 + i * 10) * (1.2 + severity * 0.3)

            active_powers[gen_name] = p_base + np.random.normal(0, 10)
            reactive_powers[gen_name] = q_base + np.random.normal(0, 5)

            # Out of step determination
            if base_stable:
                out_of_step[gen_name] = 0
            else:
                # Probability of out-of-step increases with severity
                prob_unstable = severity * 0.5
                out_of_step[gen_name] = 1 if np.random.random() < prob_unstable else 0

        return GeneratorSnapshot(
            active_powers=active_powers,
            reactive_powers=reactive_powers,
            out_of_step=out_of_step
        )

    def _determine_stability(
        self,
        fault_clearing_cycles: int,
        load_change: int,
        fault_location_idx: int
    ) -> tuple:
        """
        Determine if event leads to stability based on parameters.

        Uses heuristics to create realistic stability patterns.

        Returns:
            Tuple of (is_stable, severity)
        """
        # Base stability probability
        stability_prob = 0.7

        # Adjust based on clearing time (faster clearing = more stable)
        if fault_clearing_cycles <= 8:
            stability_prob += 0.15
        elif fault_clearing_cycles >= 14:
            stability_prob -= 0.15

        # Adjust based on load change
        load_severity = abs(load_change) / 100.0
        stability_prob -= load_severity * 0.2

        # Some fault locations are more critical
        critical_lines = [0, self.n_lines // 2, self.n_lines - 1]
        if fault_location_idx in critical_lines:
            stability_prob -= 0.1

        # Determine stability
        is_stable = np.random.random() < stability_prob

        # Calculate severity (how close to instability)
        if is_stable:
            severity = np.random.uniform(0.1, 0.4)
        else:
            severity = np.random.uniform(0.6, 1.0)

        return is_stable, severity

    def run(
        self,
        num_events: int,
        fault_clearing_time: List[int] = None,
        max_load_change: List[int] = None,
        output_dir: Path = None,
        random_seed: Optional[int] = None,
        verbose: bool = True
    ) -> SimulationResults:
        """
        Run mock simulations.

        Args:
            num_events: Number of events to simulate
            fault_clearing_time: Clearing time cycles [NineBus, NewEngland]
            max_load_change: Max load change percentage [NineBus, NewEngland]
            output_dir: Output directory (not used in mock)
            random_seed: Random seed
            verbose: Print progress

        Returns:
            SimulationResults object
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Default parameters
        if fault_clearing_time is None:
            fault_clearing_time = [10, 12]
        if max_load_change is None:
            max_load_change = [60, 60]

        # Get grid-specific parameters
        if self.grid == 'NineBusSystem':
            clearing_cycles = fault_clearing_time[0]
            max_load = max_load_change[0]
        else:
            clearing_cycles = fault_clearing_time[1] if len(fault_clearing_time) > 1 else fault_clearing_time[0]
            max_load = max_load_change[1] if len(max_load_change) > 1 else max_load_change[0]

        if verbose:
            logger.info("="*60)
            logger.info(f"Mock Simulation: {self.grid}")
            logger.info(f"Generating {num_events} synthetic events...")
            logger.info(f"Fault clearing: {clearing_cycles} cycles, Max load: {max_load}%")
            logger.info("="*60)

        # Initialize results
        results = SimulationResults(
            grid_name=self.grid,
            simulation_time=self.simulation_time,
            metadata={
                'num_events': num_events,
                'fault_clearing_cycles': clearing_cycles,
                'max_load_change': max_load,
                'random_seed': random_seed,
                'simulation_engine': 'mock'
            }
        )

        # Event generator
        event_gen = EventGenerator(random_seed)

        # Generate events
        for event_id in range(num_events):
            if verbose and (event_id + 1) % 10 == 0:
                logger.info(f"Event {event_id + 1}/{num_events}...")

            # Generate event parameters
            fault_time = event_gen.generate_fault_time(self.simulation_time)
            clearing_time = event_gen.generate_clearing_time(
                fault_time, clearing_cycles, frequency=50.0
            )
            fault_line_idx = event_gen.generate_fault_location(range(self.n_lines))
            load_idx = event_gen.generate_fault_location(range(self.n_loads))
            load_change = event_gen.generate_load_change(max_load)
            r_f, x_f = event_gen.generate_fault_impedance()

            # Determine stability
            is_stable, severity = self._determine_stability(
                clearing_cycles, load_change, fault_line_idx
            )

            # Create event info
            if self.grid == 'NineBusSystem':
                event_types = ['short circuit', 'fault clearing', 'load event']
                event_locations = [f'Line_{fault_line_idx}', f'Line_{fault_line_idx}', f'Load_{load_idx}']
                event_times = [fault_time, clearing_time, 0.0]
            else:  # NewEngland
                trip_line_idx = (fault_line_idx + 1) % self.n_lines
                trip_time = clearing_time + 0.08
                event_types = ['short circuit', 'fault clearing', 'line tripping', 'load event']
                event_locations = [f'Line_{fault_line_idx}', f'Line_{fault_line_idx}',
                                  f'Line_{trip_line_idx}', f'Load_{load_idx}']
                event_times = [fault_time, clearing_time, trip_time, 0.0]

            event_info = EventInfo(
                event_id=event_id,
                event_types=event_types,
                event_locations=event_locations,
                event_times=event_times,
                fault_time=fault_time,
                clearing_time=clearing_time,
                fault_impedance={'R_f': r_f, 'X_f': x_f},
                load_change=load_change
            )

            # Generate snapshots
            bus_snapshot_fault = self._generate_bus_snapshot(
                fault_time, is_stable, severity * 1.2
            )
            bus_snapshot_clearing = self._generate_bus_snapshot(
                clearing_time, is_stable, severity
            )

            # Generate generator snapshot
            gen_snapshot = self._generate_generator_snapshot(is_stable, severity)

            # Create event result
            event_result = EventResult(
                event_info=event_info,
                bus_snapshot_at_fault=bus_snapshot_fault,
                bus_snapshot_at_clearing=bus_snapshot_clearing,
                generator_snapshot=gen_snapshot
            )

            results.add_event(event_result)

        # Log summary
        if verbose:
            logger.info("="*60)
            stats = results.get_statistics()
            logger.info("Mock Simulation Complete:")
            logger.info(f"  Total events: {stats['total_events']}")
            logger.info(f"  Stable: {stats['stable_events']} ({stats['stability_ratio']*100:.1f}%)")
            logger.info(f"  Unstable: {stats['unstable_events']}")
            logger.info("="*60)

        return results


def create_mock_simulator(
    grid: str = 'NewEngland',
    simulation_time: float = 10.0,
    random_seed: Optional[int] = None
) -> MockSimulationEngine:
    """
    Factory function to create mock simulator.

    Args:
        grid: Grid name
        simulation_time: Simulation time
        random_seed: Random seed

    Returns:
        MockSimulationEngine instance

    Examples:
        >>> simulator = create_mock_simulator('NewEngland')
        >>> results = simulator.run(num_events=100)
    """
    return MockSimulationEngine(grid, simulation_time, random_seed)
