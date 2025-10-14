Utilities Module
================

The utils module provides configuration, I/O, and logging utilities.

.. currentmodule:: tranpy.utils

Logging
-------

.. autofunction:: get_logger

.. autofunction:: configure_logging

.. autofunction:: set_level

.. autofunction:: disable_logging

.. autofunction:: enable_logging

   **Example Usage:**

   .. code-block:: python

      from tranpy.utils import get_logger, configure_logging, set_level
      import logging

      # Get a logger for your module
      logger = get_logger(__name__)
      logger.info("Processing started")
      logger.warning("Configuration not found")

      # Configure logging globally
      configure_logging(level=logging.DEBUG, log_file='tranpy.log')

      # Change logging level at runtime
      set_level(logging.WARNING)  # Only show warnings and errors

Configuration
-------------

.. autofunction:: tranpy.utils.config.load_config

.. autofunction:: tranpy.utils.config.save_config

.. autofunction:: tranpy.utils.config.get_default_config

   **Example Usage:**

   .. code-block:: python

      from tranpy.utils.config import load_config, save_config

      # Load configuration
      config = load_config('experiment_config.yaml')

      # Save configuration
      config = {
          'grid': 'NewEngland',
          'model': {'type': 'random_forest', 'n_estimators': 100}
      }
      save_config(config, 'my_config.yaml')

I/O Utilities
-------------

.. autofunction:: tranpy.utils.io.save_results

.. autofunction:: tranpy.utils.io.load_results

   **Example Usage:**

   .. code-block:: python

      from tranpy.utils.io import save_results, load_results

      # Save results
      results = {'accuracy': 0.95, 'model': model}
      save_results(results, 'results.pkl')

      # Load results
      results = load_results('results.pkl')
