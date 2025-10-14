Data Module
===========

The data module provides utilities for downloading, caching, and managing TranPy data.

.. currentmodule:: tranpy.data

Cache Management
----------------

.. autofunction:: tranpy.data.cache.get_cache_dir

.. autofunction:: tranpy.data.cache.get_models_cache_dir

.. autofunction:: tranpy.data.cache.get_datasets_cache_dir

.. autofunction:: tranpy.data.cache.get_cache_info

.. autofunction:: tranpy.data.cache.clear_cache

   **Example Usage:**

   .. code-block:: python

      from tranpy.data.cache import get_cache_info, clear_cache

      # Get cache information
      info = get_cache_info()
      print(f"Cache size: {info['total_size_mb']:.1f} MB")
      print(f"Cached models: {info['num_models']}")

      # Clear cache
      clear_cache(confirm=False)

Download Utilities
------------------

.. autofunction:: tranpy.data.download.download_from_google_drive

.. autofunction:: tranpy.data.download.get_google_drive_direct_link

.. autofunction:: tranpy.data.download.extract_file_id_from_link

   **Example Usage:**

   .. code-block:: python

      from tranpy.data.download import download_from_google_drive

      # Download a file from Google Drive
      path = download_from_google_drive(
          file_id='1eXtw44VXhYM0jQyJGGY5Eevdrg8yui0w',
          destination='model.pkl',
          filename='my_model.pkl'
      )
