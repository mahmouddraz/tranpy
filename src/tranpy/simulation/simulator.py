"""
PowerFactory simulator for generating stability datasets.

This module provides a complete implementation for running PowerFactory simulations
and generating training datasets.

IMPORTANT: PowerFactory requires special path configuration before import.
See powerfactory_config.py for setup instructions.
"""

import os
import math
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .powerfactory_config import is_powerfactory_available, import_powerfactory

logger = get_logger(__name__)
from .model import PowerFactoryModel
from .events import EventGenerator, PowerFactoryEventConfigurator
from .results import (
    SimulationResults, EventResult, EventInfo,
    BusSnapshot, GeneratorSnapshot, LegacyDataFormat
)
from .mock_engine import MockSimulationEngine


def check_powerfactory():
    """Check if PowerFactory is available."""
    if not is_powerfactory_available():
        raise ImportError(
            "PowerFactory is not available.\n\n"
            "To use PowerFactory simulation:\n"
            "1. Install DIgSILENT PowerFactory\n"
            "2. Configure the Python path:\n\n"
            "   from tranpy.simulation import configure_powerfactory_path\n"
            "   configure_powerfactory_path(\n"
            "       custom_path=r'C:\\Program Files\\DIgSILENT\\PowerFactory 2019\\Python\\3.13'\n"
            "   )\n\n"
            "3. Or use auto-configuration:\n"
            "   configure_powerfactory_path(version='2019')\n\n"
            "4. Alternatively, use pre-generated datasets:\n"
            "   from tranpy.datasets import load_newengland\n"
        )


class PowerSystemSimulator:
    """
    Interface to PowerFactory for power system simulation.

    This class provides a clean API for running transient stability simulations
    and generating training datasets.

    Args:
        grid: Grid model name ('NewEngland' or 'NineBusSystem')
        simulation_time: Simulation duration in seconds
        output_dir: Directory for saving results

    Examples:
        >>> simulator = PowerSystemSimulator(
        ...     grid='NewEngland',
        ...     simulation_time=10
        ... )
        >>> dataset = simulator.run(
        ...     num_events=1000,
        ...     fault_clearing_time=[12],
        ...     max_load_change=[60]
        ... )
    """

    def __init__(
        self,
        grid: str = 'NewEngland',
        simulation_time: float = 10.0,
        output_dir: str = 'simulation_results',
        simulation_engine: str = 'powerfactory'
    ):
        """
        Initialize PowerSystemSimulator.

        Args:
            grid: Grid name ('NewEngland' or 'NineBusSystem')
            simulation_time: Simulation duration in seconds
            output_dir: Directory for saving results
            simulation_engine: 'powerfactory' or 'mock'
        """
        self.grid = grid
        self.simulation_time = simulation_time
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.simulation_engine = simulation_engine.lower()

        if self.simulation_engine not in ['powerfactory', 'mock']:
            raise ValueError(
                f"Unknown simulation_engine: '{simulation_engine}'. "
                "Must be 'powerfactory' or 'mock'"
            )

        # Check PowerFactory only if needed
        if self.simulation_engine == 'powerfactory':
            check_powerfactory()

        # These will be initialized based on engine type
        self.app = None
        self.model = None
        self.mock_engine = None

    def connect(self):
        """Connect to PowerFactory application."""
        try:
            # Import PowerFactory module
            pf = import_powerfactory()

            # Get application instance
            self.app = pf.GetApplication()
            if self.app is None:
                raise RuntimeError(
                    "PowerFactory GetApplication() returned None.\n"
                    "Make sure PowerFactory is running and accessible."
                )

            self.app.ClearOutputWindow()
            logger.info("Connected to PowerFactory successfully")
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to PowerFactory: {e}\n\n"
                "Troubleshooting:\n"
                "1. Ensure PowerFactory is installed\n"
                "2. Configure the Python path correctly\n"
                "3. Check that PowerFactory is running (for some versions)\n"
                "4. Verify Python version matches PowerFactory's Python version"
            ) from e

    def run(
        self,
        num_events: int = 100,
        fault_clearing_time: List[int] = None,
        max_load_change: List[int] = None,
        random_seed: Optional[int] = None,
        save_results: bool = True,
        export_csv: bool = False
    ) -> SimulationResults:
        """
        Run simulations and generate dataset.

        Args:
            num_events: Number of contingency events to simulate
            fault_clearing_time: Fault clearing time in cycles for each grid
            max_load_change: Maximum load change percentage for each grid
            random_seed: Random seed for reproducibility
            save_results: Whether to save results to pickle files
            export_csv: Whether to export CSV files for each event

        Returns:
            SimulationResults object containing all simulation data

        Examples:
            >>> # With mock engine (no PowerFactory needed)
            >>> simulator = PowerSystemSimulator('NewEngland', simulation_engine='mock')
            >>> results = simulator.run(num_events=100, random_seed=42)

            >>> # With PowerFactory
            >>> simulator = PowerSystemSimulator('NewEngland', simulation_engine='powerfactory')
            >>> results = simulator.run(num_events=100, random_seed=42)
        """
        # Route to appropriate engine
        if self.simulation_engine == 'mock':
            return self._run_mock(
                num_events, fault_clearing_time, max_load_change,
                random_seed, save_results
            )
        else:
            return self._run_powerfactory(
                num_events, fault_clearing_time, max_load_change,
                random_seed, save_results, export_csv
            )

    def _run_mock(
        self,
        num_events: int,
        fault_clearing_time: List[int],
        max_load_change: List[int],
        random_seed: Optional[int],
        save_results: bool
    ) -> SimulationResults:
        """Run simulation with mock engine."""
        # Create mock engine
        self.mock_engine = MockSimulationEngine(
            self.grid,
            self.simulation_time,
            random_seed
        )

        # Run mock simulation
        results = self.mock_engine.run(
            num_events=num_events,
            fault_clearing_time=fault_clearing_time,
            max_load_change=max_load_change,
            output_dir=self.output_dir,
            random_seed=random_seed,
            verbose=True
        )

        # Save results if requested
        if save_results:
            self._save_results(results)

        return results

    def _run_powerfactory(
        self,
        num_events: int,
        fault_clearing_time: List[int],
        max_load_change: List[int],
        random_seed: Optional[int],
        save_results: bool,
        export_csv: bool
    ) -> SimulationResults:
        """Run simulation with PowerFactory engine."""
        if random_seed is not None:
            np.random.seed(random_seed)

        # Default parameters
        if fault_clearing_time is None:
            fault_clearing_time = [10, 12]  # for NineBus, NewEngland
        if max_load_change is None:
            max_load_change = [60, 60]

        # Determine grid-specific parameters
        if self.grid == 'NineBusSystem':
            clearing_cycles = fault_clearing_time[0]
            max_load = max_load_change[0]
        else:  # NewEngland
            clearing_cycles = fault_clearing_time[1] if len(fault_clearing_time) > 1 else fault_clearing_time[0]
            max_load = max_load_change[1] if len(max_load_change) > 1 else max_load_change[0]

        # Connect to PowerFactory
        if self.app is None:
            self.connect()

        logger.info("="*60)
        logger.info(f"Running {num_events} simulations for {self.grid}")
        logger.info(f"Fault clearing: {clearing_cycles} cycles, Max load change: {max_load}%")
        logger.info("="*60)

        # Setup model
        model = PowerFactoryModel(self.app, self.grid, self.simulation_time)
        model.setup_grid()
        model.configure_results()
        model.setup_initial_conditions()

        # Initialize event generator and configurator
        event_gen = EventGenerator(random_seed)
        event_config = PowerFactoryEventConfigurator(model.events_folder, event_gen)

        # Initialize results container
        results = SimulationResults(
            grid_name=self.grid,
            simulation_time=self.simulation_time,
            metadata={
                'num_events': num_events,
                'fault_clearing_cycles': clearing_cycles,
                'max_load_change': max_load,
                'random_seed': random_seed
            }
        )

        # Main simulation loop
        for event_id in range(num_events):
            logger.info(f"Event {event_id + 1}/{num_events}...")

            try:
                # Configure events based on grid type
                if self.grid == 'NineBusSystem':
                    events, event_info = event_config.configure_ieee9bus_events(
                        model.lines, model.loads, self.simulation_time,
                        clearing_cycles, max_load
                    )
                else:  # NewEngland
                    events, event_info = event_config.configure_newengland39_events(
                        model.lines, model.loads, self.simulation_time,
                        clearing_cycles, max_load
                    )

                # Execute initial conditions
                if not model.execute_initial_conditions():
                    logger.warning("Initial conditions failed, skipping event")
                    event_config.clear_created_events()
                    model.reset_calculation()
                    continue

                # Run simulation
                if not model.run_simulation():
                    logger.warning("Simulation failed, skipping event")
                    event_config.clear_created_events()
                    model.reset_calculation()
                    continue

                # Export results to CSV (optional)
                csv_path = None
                if export_csv:
                    events_dir = self.output_dir / self.grid / 'events'
                    csv_path = model.export_results(events_dir, event_id)

                # Extract post-fault data
                event_result = self._extract_event_results(
                    model, event_id, event_info, csv_path
                )

                # Add to results
                results.add_event(event_result)

                # Status
                if event_result.is_stable:
                    logger.info("Event stable")
                else:
                    logger.info("Event unstable")

            except Exception as e:
                logger.error(f"Error during simulation: {str(e)}")

            finally:
                # Cleanup for next iteration
                event_config.clear_created_events()
                model.reset_calculation()

        # Log summary
        logger.info("="*60)
        logger.info("Simulation Summary:")
        stats = results.get_statistics()
        logger.info(f"  Total events: {stats['total_events']}")
        logger.info(f"  Stable: {stats['stable_events']} ({stats['stability_ratio']*100:.1f}%)")
        logger.info(f"  Unstable: {stats['unstable_events']}")
        logger.info("="*60)

        # Save results
        if save_results:
            self._save_results(results)

        return results

    def _extract_event_results(
        self,
        model: PowerFactoryModel,
        event_id: int,
        event_info: dict,
        csv_path: Optional[Path]
    ) -> EventResult:
        """
        Extract results from a single event simulation.

        Args:
            model: PowerFactory model instance
            event_id: Event identifier
            event_info: Event metadata dictionary
            csv_path: Path to CSV file (if exported)

        Returns:
            EventResult object with all data
        """
        # Read CSV time series data if available
        time_series_df = None
        if csv_path and csv_path.exists():
            time_series_df = pd.read_csv(csv_path, header=1)

        # Calculate time indices for snapshots
        fault_time = event_info['fault_time']
        clearing_time = event_info['clearing_time']
        step_size = model.initial_conditions.iopt_dtgrd

        # Get snapshots from time series or current model state
        if time_series_df is not None and 'b:tnow in s' in time_series_df.columns:
            # Extract from CSV
            bus_snapshot_fault = self._extract_bus_snapshot_from_csv(
                time_series_df, fault_time, step_size, model.buses
            )
            bus_snapshot_clearing = self._extract_bus_snapshot_from_csv(
                time_series_df, clearing_time, step_size, model.buses
            )
        else:
            # Get from current model state (use clearing time data)
            voltages = model.get_bus_voltages()
            angles = model.get_bus_angles()

            # Create snapshots (approximation)
            bus_snapshot_fault = BusSnapshot(
                time=fault_time,
                voltages=voltages.copy(),
                angles=angles.copy()
            )
            bus_snapshot_clearing = BusSnapshot(
                time=clearing_time,
                voltages=voltages,
                angles=angles
            )

        # Get generator data
        active_powers, reactive_powers = model.get_generator_powers()
        out_of_step = model.get_generator_out_of_step()

        generator_snapshot = GeneratorSnapshot(
            active_powers=active_powers,
            reactive_powers=reactive_powers,
            out_of_step=out_of_step
        )

        # Create EventInfo
        event_info_obj = EventInfo(
            event_id=event_id,
            event_types=event_info['event_types'],
            event_locations=event_info['event_locations'],
            event_times=event_info['event_times'],
            fault_time=event_info['fault_time'],
            clearing_time=event_info['clearing_time'],
            fault_impedance=event_info['fault_impedance'],
            load_change=event_info.get('load_change')
        )

        # Create EventResult
        return EventResult(
            event_info=event_info_obj,
            bus_snapshot_at_fault=bus_snapshot_fault,
            bus_snapshot_at_clearing=bus_snapshot_clearing,
            generator_snapshot=generator_snapshot,
            time_series_data=time_series_df
        )

    def _extract_bus_snapshot_from_csv(
        self,
        df: pd.DataFrame,
        target_time: float,
        step_size: float,
        buses: List
    ) -> BusSnapshot:
        """
        Extract bus voltage and angle snapshot from CSV at specific time.

        Args:
            df: DataFrame with time series data
            target_time: Target time point
            step_size: Simulation step size
            buses: List of bus objects

        Returns:
            BusSnapshot object
        """
        # Find closest time index
        time_idx = math.ceil(target_time / step_size)

        # Filter to positive time indices
        df_filtered = df[df['b:tnow in s'] >= 0].reset_index(drop=True)

        if time_idx >= len(df_filtered):
            time_idx = len(df_filtered) - 1

        row = df_filtered.iloc[time_idx]
        actual_time = row['b:tnow in s']

        # Extract voltages and angles
        voltages = {}
        angles = {}

        for bus in buses:
            bus_name = bus.loc_name
            # Try different column name formats
            v_col = f'{bus_name}:m:u in p.u.'
            a_col = f'{bus_name}:m:phiu in deg'

            if v_col in row:
                voltages[bus_name] = row[v_col]
            if a_col in row:
                angles[bus_name] = row[a_col]

        return BusSnapshot(
            time=actual_time,
            voltages=voltages,
            angles=angles
        )

    def _save_results(self, results: SimulationResults):
        """Save simulation results to pickle files."""
        # Create output directories
        results_dir = self.output_dir / self.grid / 'pickles'
        results_dir.mkdir(parents=True, exist_ok=True)

        data_dir = self.output_dir / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save new format
        results_path = results_dir / f'{self.grid}_results.pickle'
        results.save(results_path)
        logger.info(f"Saved results: {results_path}")

        # Save legacy format for backward compatibility
        legacy_path = data_dir / f'{self.grid}.pickle'
        LegacyDataFormat.save_legacy_format(results, legacy_path)
        logger.info(f"Saved legacy format: {legacy_path}")

