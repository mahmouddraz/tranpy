"""
PowerFactory grid model configuration and setup.

This module handles PowerFactory project activation, grid element access,
and simulation parameter configuration.
"""

from typing import List, Optional, Any
from pathlib import Path

from ..utils.logging import get_logger

logger = get_logger(__name__)


class PowerFactoryModel:
    """
    Interface to PowerFactory grid model configuration.

    This class manages:
    - Project and study case activation
    - Grid element access (buses, lines, generators, loads, etc.)
    - Results monitoring setup
    - Initial conditions configuration
    - Load flow execution

    Args:
        app: PowerFactory application instance
        grid_name: Name of the grid model ('NewEngland' or 'NineBusSystem')
        simulation_time: Simulation duration in seconds

    Examples:
        >>> pf = import_powerfactory()
        >>> app = pf.GetApplication()
        >>> model = PowerFactoryModel(app, 'NewEngland', 10.0)
        >>> model.setup_grid()
        >>> model.configure_results()
    """

    def __init__(self, app: Any, grid_name: str, simulation_time: float = 10.0):
        self.app = app
        self.grid_name = grid_name
        self.simulation_time = simulation_time

        # PowerFactory objects (initialized in setup_grid)
        self.events_folder = None
        self.initial_conditions = None
        self.sim = None
        self.elm_res = None
        self.ldf = None

        # Grid elements (initialized in setup_grid)
        self.buses = []
        self.lines = []
        self.generators = []
        self.loads = []
        self.transformers = []

    def setup_grid(self):
        """
        Activate PowerFactory project and study case.

        This method:
        1. Activates the specified grid project
        2. Loads the study case
        3. Retrieves all grid elements
        4. Gets simulation objects

        Raises:
            RuntimeError: If project or study case cannot be activated
        """
        self.app.ClearOutputWindow()
        self.app.GetCurrentUser()

        # Activate project
        project = self.app.ActivateProject(self.grid_name)
        if project is None:
            raise RuntimeError(
                f"Could not activate project '{self.grid_name}'.\n"
                f"Please ensure the project exists in PowerFactory."
            )

        # Get study case folder
        study_folder = self.app.GetProjectFolder('study')
        if study_folder is None:
            raise RuntimeError("Could not access study case folder")

        all_study_cases = study_folder.GetContents()
        if not all_study_cases:
            raise RuntimeError("No study cases found in project")

        # Activate first study case
        study_case = all_study_cases[0]
        study_case.Deactivate()
        study_case.Activate()

        # Get simulation objects
        self.events_folder = self.app.GetFromStudyCase('IntEvt')
        self.initial_conditions = self.app.GetFromStudyCase('ComInc')
        self.sim = self.app.GetFromStudyCase('ComSim')
        self.elm_res = self.app.GetFromStudyCase('Results.ElmRes')
        self.ldf = self.app.GetFromStudyCase('ComLdf')

        # Get all grid elements
        self.buses = self.app.GetCalcRelevantObjects("*.ElmTerm")
        self.lines = self.app.GetCalcRelevantObjects("*.ElmLne")
        self.generators = self.app.GetCalcRelevantObjects("*.ElmSym")
        self.loads = self.app.GetCalcRelevantObjects("*.ElmLod")
        self.transformers = self.app.GetCalcRelevantObjects("*.ElmTr2")

        logger.info(f"Grid '{self.grid_name}' loaded:")
        logger.info(f"  - Buses: {len(self.buses)}")
        logger.info(f"  - Lines: {len(self.lines)}")
        logger.info(f"  - Generators: {len(self.generators)}")
        logger.info(f"  - Loads: {len(self.loads)}")
        logger.info(f"  - Transformers: {len(self.transformers)}")

    def configure_results(self):
        """
        Configure result variables to monitor during simulation.

        This sets up monitoring for:
        - Bus voltages (m:u) and phase angles (m:phiu)
        - Generator variables (power, voltage, speed, torque, out-of-step, etc.)
        - Time variable (b:tnow)
        """
        if self.elm_res is None:
            raise RuntimeError("Results object not initialized. Call setup_grid() first.")

        # Clear any existing variables
        for element in self.elm_res.GetContents():
            element.ClearVars()

        # Add bus voltage and angle monitoring
        for bus in self.buses:
            self.elm_res.AddVars(
                bus,
                'm:u',      # Voltage magnitude (p.u.)
                'm:phiu',   # Phase angle (degrees)
            )

        # Add generator monitoring variables
        for gen in self.generators:
            self.elm_res.AddVars(
                gen,
                's:ve',         # p.u.   Excitation Voltage
                's:pt',         # p.u.   Turbine Power
                's:ut',         # p.u.   Terminal Voltage
                's:ie',         # p.u.   Excitation Current
                's:xspeed',     # p.u.   Speed
                's:xme',        # p.u.   Electrical Torque
                's:xmt',        # p.u.   Mechanical Torque
                's:cur1',       # p.u.   Positive-Sequence Current, Magnitude
                's:P1',         # MW     Positive-Sequence, Active Power
                's:Q1',         # Mvar   Positive-Sequence, Reactive Power
                's:outofstep',  # 0/1    Generator out of step indicator
                'c:firel',      # deg    Rotor angle with reference to reference machine
                'c:firot',      # deg    Rotor angle with reference to bus voltage angle
                'c:dfrotx',     # deg    Maximum rotor angle difference
                'c:fi',         # deg    Rotor angle
                'c:dorhz',      # Hz     Speed deviation
            )

        # Add time variable
        self.elm_res.AddVars(self.elm_res, 'b:tnow')

        logger.info("Results monitoring configured")

    def setup_initial_conditions(
        self,
        simulation_method: str = 'rms',
        network_type: str = 'sym',
        step_size: float = 0.01
    ):
        """
        Configure initial conditions for transient stability simulation.

        Args:
            simulation_method: 'rms' (RMS simulation) or 'ins' (EMT simulation)
            network_type: 'sym' (balanced) or 'rst' (unbalanced)
            step_size: Simulation step size in seconds

        Raises:
            RuntimeError: If initial conditions object not initialized
        """
        if self.initial_conditions is None:
            raise RuntimeError("Initial conditions not initialized. Call setup_grid() first.")

        self.initial_conditions.iopt_sim = simulation_method
        self.initial_conditions.iopt_net = network_type
        self.initial_conditions.iopt_dtgrd = step_size

        logger.info(f"Initial conditions configured: {simulation_method.upper()}, step={step_size}s")

    def execute_load_flow(self):
        """
        Execute load flow calculation.

        Returns:
            True if load flow converged, False otherwise
        """
        if self.ldf is None:
            raise RuntimeError("Load flow object not initialized. Call setup_grid() first.")

        result = self.ldf.Execute()
        if result == 0:
            logger.info("Load flow converged")
            return True
        else:
            logger.warning(f"Load flow failed with code: {result}")
            return False

    def execute_initial_conditions(self):
        """
        Execute initial conditions calculation.

        Returns:
            True if successful, False otherwise
        """
        if self.initial_conditions is None:
            raise RuntimeError("Initial conditions not initialized. Call setup_grid() first.")

        result = self.initial_conditions.Execute()
        if result == 0:
            logger.info("Initial conditions calculated")
            return True
        else:
            logger.warning(f"Initial conditions failed with code: {result}")
            return False

    def run_simulation(self) -> bool:
        """
        Execute transient stability simulation.

        Returns:
            True if simulation completed successfully, False otherwise
        """
        if self.sim is None:
            raise RuntimeError("Simulation object not initialized. Call setup_grid() first.")

        self.sim.tstop = self.simulation_time
        result = self.sim.Execute()

        if result == 0:
            return True
        else:
            logger.warning(f"Simulation failed with code: {result}")
            return False

    def export_results(self, output_path: Path, event_id: int):
        """
        Export simulation results to CSV file.

        Args:
            output_path: Directory to save results
            event_id: Event identifier for filename

        Returns:
            Path to exported CSV file
        """
        if self.elm_res is None:
            raise RuntimeError("Results object not initialized. Call setup_grid() first.")

        # Create output directory
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Configure export
        com_res = self.app.GetFromStudyCase('ComRes')
        com_res.iopt_exp = 6        # Export as CSV
        com_res.iopt_csel = 0       # Export everything
        com_res.iopt_tsel = 0       # All time steps
        com_res.iopt_honly = 0      # Include data (not just headers)
        com_res.iopt_locn = 2       # Location setting
        com_res.iopt_sep = 1        # Use system separator
        com_res.ciopt_head = 1      # Include headers

        # Set elements to export (generators)
        com_res.element = self.generators
        com_res.variable = []
        com_res.pResult = self.elm_res

        # Set output filename
        csv_path = output_path / f'event_{event_id}.csv'
        com_res.f_name = str(csv_path)

        # Execute export
        result = com_res.Execute()
        if result != 0:
            logger.warning(f"Export failed with code: {result}")
            return None

        return csv_path

    def reset_calculation(self):
        """Reset calculation to prepare for next simulation."""
        self.app.ResetCalculation()

    def get_bus_voltages(self) -> dict:
        """
        Get current bus voltage magnitudes.

        Returns:
            Dictionary mapping bus names to voltage values (p.u.)
        """
        return {
            bus.loc_name: bus.GetAttribute('m:u')
            for bus in self.buses
        }

    def get_bus_angles(self) -> dict:
        """
        Get current bus phase angles.

        Returns:
            Dictionary mapping bus names to angle values (degrees)
        """
        return {
            bus.loc_name: bus.GetAttribute('m:phiu')
            for bus in self.buses
        }

    def get_generator_powers(self) -> tuple:
        """
        Get current generator active and reactive powers.

        Returns:
            Tuple of (active_powers, reactive_powers) dictionaries
        """
        active = {
            gen.loc_name: gen.GetAttribute('s:P1')
            for gen in self.generators
        }
        reactive = {
            gen.loc_name: gen.GetAttribute('s:Q1')
            for gen in self.generators
        }
        return active, reactive

    def get_generator_out_of_step(self) -> dict:
        """
        Get generator out-of-step indicators.

        Returns:
            Dictionary mapping generator names to out-of-step status (0 or 1)
        """
        return {
            gen.loc_name: gen.GetAttribute('s:outofstep')
            for gen in self.generators
        }

    def clear_events(self):
        """Clear all events from the events folder."""
        if self.events_folder is None:
            return

        for event in self.events_folder.GetContents():
            event.Delete()
