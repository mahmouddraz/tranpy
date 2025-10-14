"""Cache management for TranPy."""

import json
import shutil
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

from ..utils.logging import get_logger

logger = get_logger(__name__)


def get_cache_dir() -> Path:
    """
    Get TranPy cache directory.

    Returns:
        Path to ~/.tranpy/ cache directory

    Examples:
        >>> from tranpy.data import get_cache_dir
        >>> cache_dir = get_cache_dir()
        >>> print(cache_dir)
        /Users/username/.tranpy
    """
    cache_dir = Path.home() / '.tranpy'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_models_cache_dir() -> Path:
    """Get directory for cached pretrained models."""
    models_dir = get_cache_dir() / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_datasets_cache_dir() -> Path:
    """Get directory for cached datasets."""
    datasets_dir = get_cache_dir() / 'datasets'
    datasets_dir.mkdir(parents=True, exist_ok=True)
    return datasets_dir


def get_cache_info() -> Dict:
    """
    Get information about cached files.

    Returns:
        Dictionary with cache statistics

    Examples:
        >>> from tranpy.data import get_cache_info
        >>> info = get_cache_info()
        >>> print(f"Cache size: {info['total_size_mb']:.1f} MB")
    """
    cache_dir = get_cache_dir()

    if not cache_dir.exists():
        return {
            'cache_dir': str(cache_dir),
            'exists': False,
            'total_size_mb': 0,
            'num_models': 0,
            'num_datasets': 0
        }

    # Calculate total size
    total_size = sum(
        f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()
    )

    # Count cached items
    models_dir = cache_dir / 'models'
    datasets_dir = cache_dir / 'datasets'

    num_models = len(list(models_dir.glob('*.pkl'))) if models_dir.exists() else 0
    num_datasets = len(list(datasets_dir.glob('*.pickle'))) if datasets_dir.exists() else 0

    return {
        'cache_dir': str(cache_dir),
        'exists': True,
        'total_size_mb': total_size / (1024 * 1024),
        'num_models': num_models,
        'num_datasets': num_datasets,
        'models_dir': str(models_dir) if models_dir.exists() else None,
        'datasets_dir': str(datasets_dir) if datasets_dir.exists() else None
    }


def clear_cache(confirm: bool = True) -> None:
    """
    Clear TranPy cache directory.

    Args:
        confirm: If True, ask for confirmation before clearing

    Examples:
        >>> from tranpy.data import clear_cache
        >>> clear_cache()  # Will ask for confirmation
        >>> clear_cache(confirm=False)  # Clear without asking
    """
    cache_dir = get_cache_dir()

    if not cache_dir.exists():
        logger.info(f"Cache directory does not exist: {cache_dir}")
        return

    # Get info before clearing
    info = get_cache_info()

    if confirm:
        response = input(
            f"Are you sure you want to clear the cache?\n"
            f"  Location: {cache_dir}\n"
            f"  Size: {info['total_size_mb']:.1f} MB\n"
            f"  Models: {info['num_models']}\n"
            f"  Datasets: {info['num_datasets']}\n"
            f"Type 'yes' to confirm: "
        )
        if response.lower() != 'yes':
            logger.info("Cache clear cancelled.")
            return

    # Clear cache
    shutil.rmtree(cache_dir)
    logger.info(f"Cache cleared: {cache_dir}")


def save_cache_metadata(model_id: str, metadata: Dict) -> None:
    """
    Save metadata for a cached model.

    Args:
        model_id: Model identifier
        metadata: Metadata dictionary
    """
    cache_dir = get_cache_dir()
    metadata_file = cache_dir / 'cache_metadata.json'

    # Load existing metadata
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            all_metadata = json.load(f)
    else:
        all_metadata = {}

    # Add new metadata
    all_metadata[model_id] = {
        **metadata,
        'cached_at': datetime.now().isoformat()
    }

    # Save
    with open(metadata_file, 'w') as f:
        json.dump(all_metadata, f, indent=2)


def get_cache_metadata(model_id: Optional[str] = None) -> Dict:
    """
    Get metadata for cached models.

    Args:
        model_id: If provided, get metadata for specific model.
                  Otherwise, get all metadata.

    Returns:
        Metadata dictionary
    """
    cache_dir = get_cache_dir()
    metadata_file = cache_dir / 'cache_metadata.json'

    if not metadata_file.exists():
        return {} if model_id is None else None

    with open(metadata_file, 'r') as f:
        all_metadata = json.load(f)

    if model_id:
        return all_metadata.get(model_id)
    else:
        return all_metadata
