Simulation Module
=================

The simulation module provides PowerFactory integration for generating custom stability datasets.

.. currentmodule:: tranpy.simulation

Power System Simulator
----------------------

.. autoclass:: PowerSystemSimulator
   :members:
   :show-inheritance:

Mock Simulation Engine
----------------------

.. autoclass:: MockSimulationEngine
   :members:
   :show-inheritance:

   **Example Usage:**

   .. code-block:: python

      from tranpy.simulation import MockSimulationEngine

      # Create mock simulator (no PowerFactory required)
      engine = MockSimulationEngine('NewEngland', simulation_time=10.0)
      results = engine.run(num_events=100, random_seed=42)

Results Classes
---------------

.. autoclass:: SimulationResults
   :members:
   :show-inheritance:

.. autoclass:: EventResult
   :members:
   :show-inheritance:

.. autoclass:: EventInfo
   :members:
   :show-inheritance:

.. autoclass:: BusSnapshot
   :members:
   :show-inheritance:

.. autoclass:: GeneratorSnapshot
   :members:
   :show-inheritance:

Dataset Generation
------------------

.. autofunction:: generate_dataset_from_simulation

.. autoclass:: DatasetGenerator
   :members:
   :show-inheritance:

Event Generation
----------------

.. autoclass:: EventGenerator
   :members:
   :show-inheritance:

PowerFactory Configuration
--------------------------

.. autofunction:: configure_powerfactory_path

.. autofunction:: is_powerfactory_available

.. autofunction:: import_powerfactory
