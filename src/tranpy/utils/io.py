"""Input/Output utilities."""

import pickle
import json
from pathlib import Path
from typing import Any, Dict

from .logging import get_logger

logger = get_logger(__name__)


def save_results(results: Dict[str, Any], filepath: str, format: str = 'pickle'):
    """
    Save results to file.

    Args:
        results: Results dictionary
        filepath: Path to save file
        format: Format ('pickle' or 'json')

    Examples:
        >>> results = {'accuracy': 0.95, 'predictions': predictions}
        >>> save_results(results, 'results.pkl')
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    if format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    elif format == 'json':
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info(f"Results saved to {filepath}")


def load_results(filepath: str, format: str = 'pickle') -> Dict[str, Any]:
    """
    Load results from file.

    Args:
        filepath: Path to results file
        format: Format ('pickle' or 'json')

    Returns:
        Results dictionary

    Examples:
        >>> results = load_results('results.pkl')
        >>> print(results['accuracy'])
    """
    if format == 'pickle':
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
    elif format == 'json':
        with open(filepath, 'r') as f:
            results = json.load(f)
    else:
        raise ValueError(f"Unknown format: {format}")

    return results
