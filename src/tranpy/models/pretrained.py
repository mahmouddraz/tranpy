"""Pretrained model loading and management with Google Drive integration."""

import json
import pickle
from pathlib import Path
from typing import Optional, List, Dict

from ..utils.logging import get_logger
from ..data.cache import get_models_cache_dir, save_cache_metadata
from ..data.download import download_from_google_drive

logger = get_logger(__name__)


def _load_registry() -> dict:
    """Load pretrained models registry from JSON file."""
    registry_path = Path(__file__).parent.parent / 'data' / 'models' / 'registry.json'
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Models registry not found at {registry_path}. "
            "Please ensure the package is installed correctly."
        )
    with open(registry_path, 'r') as f:
        return json.load(f)


def list_pretrained_models(grid: Optional[str] = None) -> List[Dict]:
    """
    List available pretrained models from registry.

    Args:
        grid: If provided, filter models by grid system
              ('NewEngland' or 'NineBusSystem')

    Returns:
        List of model information dictionaries

    Examples:
        >>> from tranpy.models import list_pretrained_models
        >>>
        >>> # List all models
        >>> models = list_pretrained_models()
        >>> for model in models:
        ...     print(f"{model['model_id']}: {model['type']} ({model['size_kb']} KB)")
        >>>
        >>> # Filter by grid
        >>> ne_models = list_pretrained_models(grid='NewEngland')
    """
    registry = _load_registry()

    models = []
    for grid_name, grid_models in registry['models'].items():
        # Filter by grid if specified
        if grid and grid.lower() not in grid_name.lower():
            continue

        for model_key, model_info in grid_models.items():
            # Check if model is cached
            cache_dir = get_models_cache_dir()
            cache_path = cache_dir / f"{model_info['model_id']}.pkl"
            is_cached = cache_path.exists()

            models.append({
                'model_id': model_info['model_id'],
                'type': model_info['model_type'],
                'grid': model_info['grid'],
                'size_kb': model_info['size_kb'],
                'description': model_info['description'],
                'bundled': model_info.get('bundled', False),
                'cached': is_cached,
                'google_drive_folder': model_info['google_drive']['folder_link'],
                'filename': model_info['filename']
            })

    return models


def load_pretrained(model_id: str, force_download: bool = False, local: bool = False):
    """
    Load a pretrained model (from local package or download from Google Drive).

    Args:
        model_id: Model identifier (e.g., 'dt_ne39', 'adaboost_9bus')
        force_download: If True, re-download even if cached (ignored if local=True)
        local: If True, load from package's local pretrained models directory
               If False (default), download from Google Drive and cache

    Returns:
        Loaded pretrained model

    Raises:
        ValueError: If model_id not found in registry
        FileNotFoundError: If model file cannot be found/downloaded

    Examples:
        >>> from tranpy.models import load_pretrained
        >>>
        >>> # Load from local package (fast, offline-capable)
        >>> model = load_pretrained('dt_ne39', local=True)
        >>>
        >>> # Load from Google Drive (downloads on first use)
        >>> model = load_pretrained('dt_ne39')
        >>> # First time: Downloads from Google Drive
        >>> # Next time: Loads from cache (~/.tranpy/models/)
        >>>
        >>> # Make predictions
        >>> predictions = model.predict(X_test)
        >>>
        >>> # Force re-download from Google Drive
        >>> model = load_pretrained('dt_ne39', force_download=True)
    """
    registry = _load_registry()

    # Find model in registry
    model_info = None
    for grid_models in registry['models'].values():
        for model_data in grid_models.values():
            if model_data['model_id'] == model_id:
                model_info = model_data
                break
        if model_info:
            break

    if not model_info:
        available_ids = []
        for grid_models in registry['models'].values():
            for model_data in grid_models.values():
                available_ids.append(model_data['model_id'])

        raise ValueError(
            f"Unknown model ID: '{model_id}'\n"
            f"Available models: {available_ids[:10]}...\n"
            f"Use list_pretrained_models() to see all available models."
        )

    # Determine model path based on local parameter
    if local:
        # Load from local package pretrained models directory
        local_models_dir = Path(__file__).parent.parent / 'data' / 'models' / 'pretrained'
        model_path = local_models_dir / model_info['filename']

        if not model_path.exists():
            raise FileNotFoundError(
                f"Local model not found: {model_path}\n"
                f"Available options:\n"
                f"1. Run: python scripts/train_all_models.py (to train locally)\n"
                f"2. Use: load_pretrained('{model_id}') without local=True (to download from Google Drive)"
            )
        logger.info(f"Loading {model_id} from local package...")

    elif model_info.get('bundled', False):
        # Load from bundled package data (legacy support)
        model_path = Path(__file__).parent.parent / 'data' / 'models' / model_info['filename']
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model {model_id} should be bundled but not found at {model_path}"
            )
    else:
        # Check cache first
        cache_dir = get_models_cache_dir()
        cache_path = cache_dir / f"{model_id}.pkl"

        if cache_path.exists() and not force_download:
            logger.info(f"Loading {model_id} from cache...")
            model_path = cache_path
        else:
            # Need to download from Google Drive
            file_id = model_info['google_drive'].get('file_id')

            if not file_id:
                raise NotImplementedError(
                    f"Model {model_id} requires Google Drive download, "
                    f"but individual file ID is not yet configured in registry.\n\n"
                    f"To download manually:\n"
                    f"1. Visit: {model_info['google_drive']['folder_link']}\n"
                    f"2. Download: {model_info['filename']}\n"
                    f"3. Save to: {cache_path}\n\n"
                    f"Or use: load_pretrained('{model_id}', local=True) to load from local package"
                )

            # Download from Google Drive
            logger.info(f"Downloading {model_info['model_type']} for {model_info['grid']}...")
            model_path = download_from_google_drive(
                file_id=file_id,
                destination=cache_path,
                filename=model_info['filename']
            )

            # Save metadata
            save_cache_metadata(model_id, {
                'model_type': model_info['model_type'],
                'grid': model_info['grid'],
                'size_kb': model_info['size_kb'],
                'filename': model_info['filename']
            })

    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    logger.info(f"Loaded {model_info['model_type']} for {model_info['grid']}")
    return model


def save_model(model, filepath: str):
    """
    Save a trained model to disk.

    Args:
        model: Trained model instance
        filepath: Path to save the model

    Examples:
        >>> from tranpy.models import SVMClassifier, save_model
        >>>
        >>> # Train a model
        >>> model = SVMClassifier()
        >>> model.fit(X_train, y_train)
        >>>
        >>> # Save it
        >>> save_model(model, 'my_svm_model.pkl')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    logger.info(f"Model saved to: {filepath}")


def load_model(filepath: str):
    """
    Load a trained model from disk.

    Args:
        filepath: Path to the saved model

    Returns:
        Loaded model instance

    Examples:
        >>> from tranpy.models import load_model
        >>>
        >>> # Load a saved model
        >>> model = load_model('my_svm_model.pkl')
        >>>
        >>> # Use it
        >>> predictions = model.predict(X_test)
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)

    logger.info(f"Model loaded from: {filepath}")
    return model
