Quick Start
===========

Load a Dataset
--------------

.. code-block:: python

   from tranpy.datasets import load_newengland

   # Load New England 39-bus system
   X_train, X_test, y_train, y_test = load_newengland(
       test_size=0.2,
       random_state=42
   )

   print(f"Training samples: {X_train.shape[0]}")
   print(f"Features: {X_train.shape[1]}")

Train a Model
-------------

.. code-block:: python

   from tranpy.models import RandomForestClassifier

   # Create and train model
   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)

   # Evaluate
   accuracy = model.evaluate(X_test, y_test)
   print(f"Accuracy: {accuracy:.2%}")

Pretrained Models
-----------------

.. code-block:: python

   from tranpy.models import load_pretrained, list_pretrained_models

   # List available models
   models = list_pretrained_models(grid='NewEngland')
   for m in models:
       print(f"{m['model_id']}: {m['type']}")

   # Load a pretrained model
   model = load_pretrained('rf_ne39')

   # Make predictions
   predictions = model.predict(X_test)

Explainability
--------------

.. code-block:: python

   from tranpy.explainers import SHAPExplainer

   # Create explainer
   explainer = SHAPExplainer(
       model=model,
       X_train=X_train[:100],  # Use subset for speed
       X_test=X_test
   )

   # Compute SHAP values
   shap_values = explainer.explain_global()

   # Plot summary
   explainer.plot_summary(shap_values, save_path='shap_summary.png')

   # Get top features
   top_features = explainer.get_top_features(n_features=10)
   print(top_features)

Simulation
----------

.. code-block:: python

   from tranpy.simulation import PowerSystemSimulator

   # Create simulator with mock engine
   simulator = PowerSystemSimulator(
       grid='NewEngland',
       simulation_time=10.0,
       simulation_engine='mock'
   )

   # Generate dataset
   results = simulator.run(
       num_events=1000,
       random_seed=42
   )

   # Convert to training dataset
   from tranpy.simulation import generate_dataset_from_simulation

   dataset, splits = generate_dataset_from_simulation(results)
   X_train = splits['X_train']
   y_train = splits['y_train']

Logging
-------

.. code-block:: python

   from tranpy.utils import configure_logging, set_level
   import logging

   # Enable debug logging
   set_level(logging.DEBUG)

   # Or configure globally
   configure_logging(
       level=logging.INFO,
       log_file='tranpy.log'
   )

   # Disable logging
   from tranpy.utils import disable_logging
   disable_logging()

Complete Example
----------------

.. code-block:: python

   from tranpy.datasets import load_newengland
   from tranpy.models import RandomForestClassifier, save_model
   from tranpy.explainers import SHAPExplainer
   from tranpy.utils import get_logger

   # Setup logging
   logger = get_logger(__name__)

   # Load data
   logger.info("Loading dataset...")
   X_train, X_test, y_train, y_test = load_newengland(
       test_size=0.2, random_state=42
   )

   # Train model
   logger.info("Training model...")
   model = RandomForestClassifier(n_estimators=100)
   model.fit(X_train, y_train)

   # Evaluate
   logger.info("Evaluating model...")
   accuracy = model.evaluate(X_test, y_test)
   logger.info(f"Test accuracy: {accuracy:.2%}")

   # Save model
   save_model(model, 'my_model.pkl')

   # Explain predictions
   logger.info("Generating explanations...")
   explainer = SHAPExplainer(model, X_train[:100], X_test)
   shap_values = explainer.explain_global()
   explainer.plot_summary(shap_values, save_path='explanations.png')

   logger.info("Complete!")

API Reference
-------------

See :doc:`api/models`, :doc:`api/datasets`, :doc:`api/explainers` for complete API documentation.
