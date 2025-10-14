Installation Guide
==================

Requirements
------------

- Python 3.8 or higher
- pip or uv package manager

Basic Installation
------------------

Install TranPy using pip:

.. code-block:: bash

   pip install tranpy

This installs the core package with basic dependencies.

Installation with Optional Features
------------------------------------

TranPy offers several optional feature sets:

Explainability Features
~~~~~~~~~~~~~~~~~~~~~~~

Install with all explainability tools (SHAP, LIME, DALEX):

.. code-block:: bash

   pip install tranpy[explainability]

Or just LIME (lightweight):

.. code-block:: bash

   pip install tranpy[explainability-lite]

Neural Network Support
~~~~~~~~~~~~~~~~~~~~~~

Install with TensorFlow for neural network models:

.. code-block:: bash

   pip install tranpy[neural]

Boosting Models
~~~~~~~~~~~~~~~

Install XGBoost and LightGBM:

.. code-block:: bash

   pip install tranpy[boosting]

All Features
~~~~~~~~~~~~

Install everything:

.. code-block:: bash

   pip install tranpy[all]

Development Installation
------------------------

For development or contributing:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/mahmouddraz/tranpy.git
   cd tranpy

   # Install in editable mode with dev dependencies
   pip install -e ".[dev,docs,all]"

   # Or using uv (faster)
   uv pip install -e ".[dev,docs,all]"

Verify Installation
-------------------

Test your installation:

.. code-block:: python

   import tranpy
   from tranpy.datasets import load_newengland

   # Load a dataset
   dataset = load_newengland()
   print(f"Loaded {dataset.data.shape[0]} samples")
   print("✓ TranPy installed successfully!")

Troubleshooting
---------------

PowerFactory Issues
~~~~~~~~~~~~~~~~~~~

If you're using the simulation module with PowerFactory:

1. Ensure PowerFactory is installed
2. Configure the Python path before importing:

.. code-block:: python

   from tranpy.simulation import configure_powerfactory_path

   configure_powerfactory_path(
       custom_path=r'C:\Program Files\DIgSILENT\PowerFactory 2019\Python\3.13'
   )

Or use the mock simulation engine (no PowerFactory required):

.. code-block:: python

   from tranpy.simulation import PowerSystemSimulator

   simulator = PowerSystemSimulator('NewEngland', simulation_engine='mock')
   results = simulator.run(num_events=100)

Missing Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you encounter import errors for optional features:

.. code-block:: bash

   # For SHAP
   pip install shap

   # For LIME
   pip install lime

   # For DALEX
   pip install dalex

   # For XGBoost
   pip install xgboost

   # For LightGBM
   pip install lightgbm

   # For TensorFlow
   pip install tensorflow
