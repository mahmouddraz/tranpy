TranPy Documentation
====================

Power system transient stability analysis with machine learning.

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

What's Included
---------------

- **Datasets**: New England 39-bus, IEEE 9-bus systems
- **Models**: Random Forest, SVM, Neural Networks, XGBoost, LightGBM, and 10+ more
- **Explainability**: SHAP, LIME, DALEX integration
- **Simulation**: PowerFactory interface, mock engine for testing
- **Pretrained Models**: Pre-trained classifiers available for download

Installation
------------

.. code-block:: bash

   pip install tranpy                    # Core package
   pip install tranpy[explainability]    # Add SHAP, LIME, DALEX
   pip install tranpy[neural]            # Add TensorFlow
   pip install tranpy[boosting]          # Add XGBoost, LightGBM
   pip install tranpy[all]               # Everything

Basic Usage
-----------

.. code-block:: python

   from tranpy.datasets import load_newengland
   from tranpy.models import RandomForestClassifier

   # Load dataset
   X_train, X_test, y_train, y_test = load_newengland(test_size=0.2, random_state=42)

   # Train and evaluate
   model = RandomForestClassifier(n_estimators=100)
   model.fit(X_train, y_train)
   accuracy = model.evaluate(X_test, y_test)

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Guides

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/datasets
   api/models
   api/explainers
   api/simulation
   api/data
   api/utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
