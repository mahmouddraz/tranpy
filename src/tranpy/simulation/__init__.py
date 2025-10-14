"""
TranPy Simulation Module

Provides interface to DIgSILENT PowerFactory for generating stability datasets.

Note: PowerFactory must be installed separately and is only required
for dataset generation, not for using pre-generated datasets.

Setup Instructions:
    1. Install DIgSILENT PowerFactory
    2. Configure Python path:
        from tranpy.simulation import configure_powerfactory_path
        configure_powerfactory_path(
            custom_path=r"C:\\Program Files\\DIgSILENT\\PowerFactory 2019\\Python\\3.13"
        )
    3. Use simulator:
        from tranpy.simulation import PowerSystemSimulator
        simulator = PowerSystemSimulator(grid='NewEngland')
        results = simulator.run(num_events=100)

    4. Generate dataset:
        from tranpy.simulation import generate_dataset_from_simulation
        dataset, splits = generate_dataset_from_simulation(results)
"""

from .powerfactory_config import (
    configure_powerfactory_path,
    import_powerfactory,
    is_powerfactory_available,
    get_powerfactory_info
)

try:
    from .simulator import PowerSystemSimulator
    from .events import EventGenerator, PowerFactoryEventConfigurator
    from .model import PowerFactoryModel
    from .mock_engine import MockSimulationEngine
    from .results import (
        SimulationResults,
        EventResult,
        EventInfo,
        BusSnapshot,
        GeneratorSnapshot,
        LegacyDataFormat
    )
    from .dataset_generator import (
        generate_dataset_from_simulation,
        load_and_generate_dataset,
        generate_legacy_dataset,
        DatasetGenerator
    )
    from .config import (
        SimulationConfig,
        load_config_from_yaml,
        save_config_to_yaml,
        create_default_config,
        create_config_from_template
    )
    _SIMULATION_AVAILABLE = True
except ImportError as e:
    _SIMULATION_AVAILABLE = False
    PowerSystemSimulator = None
    EventGenerator = None
    PowerFactoryEventConfigurator = None
    PowerFactoryModel = None
    MockSimulationEngine = None
    SimulationResults = None
    EventResult = None
    EventInfo = None
    BusSnapshot = None
    GeneratorSnapshot = None
    LegacyDataFormat = None
    generate_dataset_from_simulation = None
    load_and_generate_dataset = None
    generate_legacy_dataset = None
    DatasetGenerator = None
    SimulationConfig = None
    load_config_from_yaml = None
    save_config_to_yaml = None
    create_default_config = None
    create_config_from_template = None

__all__ = [
    # PowerFactory configuration
    'configure_powerfactory_path',
    'import_powerfactory',
    'is_powerfactory_available',
    'get_powerfactory_info',

    # Simulation
    'PowerSystemSimulator',
    'PowerFactoryModel',
    'MockSimulationEngine',

    # Events
    'EventGenerator',
    'PowerFactoryEventConfigurator',

    # Results
    'SimulationResults',
    'EventResult',
    'EventInfo',
    'BusSnapshot',
    'GeneratorSnapshot',
    'LegacyDataFormat',

    # Dataset generation
    'generate_dataset_from_simulation',
    'load_and_generate_dataset',
    'generate_legacy_dataset',
    'DatasetGenerator',

    # Configuration
    'SimulationConfig',
    'load_config_from_yaml',
    'save_config_to_yaml',
    'create_default_config',
    'create_config_from_template',
]
