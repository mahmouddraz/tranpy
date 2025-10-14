"""
Configuration management for PowerFactory simulations.

This module provides configuration classes and YAML loading utilities
for simulation parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SimulationConfig:
    """
    Configuration for PowerFactory transient stability simulations.

    Attributes:
        grid: Grid model name ('NewEngland' or 'NineBusSystem')
        simulation_time: Simulation duration in seconds
        number_of_events: Number of contingency events to simulate
        fault_clearing_time_cycles: Fault clearing time in cycles [NineBus, NewEngland]
        max_load_change: Maximum load change percentage [NineBus, NewEngland]
        output_path: Directory for saving results
        random_seed: Random seed for reproducibility
        export_csv: Whether to export CSV files for each event
        ml_algorithm: List of ML algorithms to use (for training)
        epochs: Number of training epochs (for deep learning)
        batch_size: Batch size (for deep learning)
        optimizer: Optimizer name (for deep learning)
        learning_rate: Learning rate (for deep learning)
        reduced: Feature reduction flag (0 or 1)

    Examples:
        >>> config = SimulationConfig(
        ...     grid='NewEngland',
        ...     number_of_events=100,
        ...     fault_clearing_time_cycles=[10, 12],
        ...     max_load_change=[60, 60]
        ... )
        >>> config.get_grid_parameters()
    """

    # Core simulation parameters
    grid: str = 'NewEngland'
    simulation_time: float = 10.0
    number_of_events: int = 100
    fault_clearing_time_cycles: List[int] = field(default_factory=lambda: [10, 12])
    max_load_change: List[int] = field(default_factory=lambda: [60, 60])

    # Simulation engine
    simulation_engine: str = 'powerfactory'  # 'powerfactory' or 'mock'

    # Output configuration
    output_path: str = 'simulation_results'
    random_seed: Optional[int] = None
    export_csv: bool = False

    # ML training parameters (optional)
    ml_algorithm: List[str] = field(default_factory=lambda: ['svm'])
    epochs: int = 10
    batch_size: int = 32
    optimizer: str = 'Adam'
    learning_rate: float = 0.01
    n_steps: int = 1
    reduced: int = 0

    # Legacy compatibility
    key_data: int = 1  # 1 for simulation, 0 for loading existing data

    def get_grid_parameters(self) -> Dict[str, Any]:
        """
        Get grid-specific parameters.

        Returns:
            Dictionary with fault clearing cycles and max load change
        """
        if self.grid == 'NineBusSystem':
            idx = 0
        else:  # NewEngland
            idx = 1 if len(self.fault_clearing_time_cycles) > 1 else 0

        return {
            'fault_clearing_cycles': self.fault_clearing_time_cycles[idx],
            'max_load_change': self.max_load_change[idx] if len(self.max_load_change) > 1 else self.max_load_change[0]
        }

    def validate(self):
        """
        Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        valid_grids = ['NewEngland', 'NineBusSystem']
        if self.grid not in valid_grids:
            raise ValueError(
                f"Invalid grid: {self.grid}. Must be one of {valid_grids}"
            )

        valid_engines = ['powerfactory', 'mock']
        if self.simulation_engine.lower() not in valid_engines:
            raise ValueError(
                f"Invalid simulation_engine: {self.simulation_engine}. "
                f"Must be one of {valid_engines}"
            )

        if self.simulation_time <= 0:
            raise ValueError("simulation_time must be positive")

        if self.number_of_events <= 0:
            raise ValueError("number_of_events must be positive")

        if not self.fault_clearing_time_cycles:
            raise ValueError("fault_clearing_time_cycles cannot be empty")

        if not self.max_load_change:
            raise ValueError("max_load_change cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'grid': self.grid,
            'simulation_time': self.simulation_time,
            'number_of_events': self.number_of_events,
            'fault_clearing_time__cycles': self.fault_clearing_time_cycles,
            'max_load_change': self.max_load_change,
            'simulation_engine': self.simulation_engine,
            'path': self.output_path,
            'key_data': self.key_data,
            'ml_algorithm': self.ml_algorithm,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'n_steps': self.n_steps,
            'reduced': self.reduced
        }

    def __repr__(self) -> str:
        params = self.get_grid_parameters()
        return (
            f"SimulationConfig(\n"
            f"  grid='{self.grid}',\n"
            f"  engine='{self.simulation_engine}',\n"
            f"  n_events={self.number_of_events},\n"
            f"  clearing_cycles={params['fault_clearing_cycles']},\n"
            f"  max_load={params['max_load_change']}%\n"
            f")"
        )


def load_config_from_yaml(yaml_path: [str, Path]) -> SimulationConfig:
    """
    Load configuration from YAML file.

    This function loads configuration in the format compatible with
    the legacy config.yaml format.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        SimulationConfig object

    Raises:
        FileNotFoundError: If YAML file not found
        ValueError: If YAML format is invalid

    Examples:
        >>> config = load_config_from_yaml('config.yaml')
        >>> print(config)
    """
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Invalid YAML format: expected dictionary")

    # Map legacy field names to new config
    config = SimulationConfig(
        grid=data.get('grid', 'NewEngland'),
        simulation_time=data.get('simulation_time', 10.0),
        number_of_events=data.get('number_of_events', 100),
        fault_clearing_time_cycles=data.get('fault_clearing_time__cycles', [10, 12]),
        max_load_change=data.get('max_load_change', [60, 60]),
        simulation_engine=data.get('simulation_engine', 'powerfactory'),
        output_path=data.get('path', 'simulation_results'),
        key_data=data.get('key_data', 1),
        ml_algorithm=data.get('ml_algorithm', ['svm']),
        epochs=data.get('epochs', 10),
        batch_size=data.get('batch_size', 32),
        optimizer=data.get('optimizer', 'Adam'),
        learning_rate=data.get('learning_rate', 0.01),
        n_steps=data.get('n_steps', 1),
        reduced=data.get('reduced', 0)
    )

    # Validate configuration
    config.validate()

    return config


def save_config_to_yaml(config: SimulationConfig, yaml_path: [str, Path]):
    """
    Save configuration to YAML file.

    Args:
        config: SimulationConfig object
        yaml_path: Path to save YAML file

    Examples:
        >>> config = SimulationConfig(grid='NewEngland', number_of_events=100)
        >>> save_config_to_yaml(config, 'my_config.yaml')
    """
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_dict()

    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved configuration: {yaml_path}")


def create_default_config(grid: str = 'NewEngland') -> SimulationConfig:
    """
    Create default configuration for a grid.

    Args:
        grid: Grid name ('NewEngland' or 'NineBusSystem')

    Returns:
        SimulationConfig with default parameters

    Examples:
        >>> config = create_default_config('NewEngland')
        >>> config.number_of_events = 1000
    """
    if grid == 'NineBusSystem':
        return SimulationConfig(
            grid=grid,
            simulation_time=10.0,
            number_of_events=100,
            fault_clearing_time_cycles=[10, 12],
            max_load_change=[60, 60],
            output_path='simulation_results'
        )
    else:  # NewEngland (default)
        return SimulationConfig(
            grid=grid,
            simulation_time=10.0,
            number_of_events=100,
            fault_clearing_time_cycles=[10, 12],
            max_load_change=[60, 60],
            output_path='simulation_results'
        )


# Configuration templates
CONFIG_TEMPLATES = {
    'quick_test': {
        'description': 'Quick test with few events',
        'number_of_events': 10,
        'export_csv': False
    },
    'standard': {
        'description': 'Standard dataset generation',
        'number_of_events': 1000,
        'export_csv': False
    },
    'detailed': {
        'description': 'Detailed with CSV export',
        'number_of_events': 1000,
        'export_csv': True
    },
    'large_scale': {
        'description': 'Large scale dataset',
        'number_of_events': 10000,
        'export_csv': False
    }
}


def create_config_from_template(
    template_name: str,
    grid: str = 'NewEngland',
    **overrides
) -> SimulationConfig:
    """
    Create configuration from template.

    Args:
        template_name: Template name ('quick_test', 'standard', 'detailed', 'large_scale')
        grid: Grid name
        **overrides: Additional parameters to override

    Returns:
        SimulationConfig object

    Examples:
        >>> config = create_config_from_template('quick_test', grid='NewEngland')
        >>> config = create_config_from_template('standard', simulation_time=15.0)
    """
    if template_name not in CONFIG_TEMPLATES:
        available = list(CONFIG_TEMPLATES.keys())
        raise ValueError(
            f"Unknown template: {template_name}. "
            f"Available templates: {available}"
        )

    template = CONFIG_TEMPLATES[template_name]

    # Start with default config
    config = create_default_config(grid)

    # Apply template
    for key, value in template.items():
        if key != 'description' and hasattr(config, key):
            setattr(config, key, value)

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    config.validate()
    return config
