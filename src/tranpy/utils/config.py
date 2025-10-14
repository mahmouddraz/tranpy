"""Configuration management utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        filepath: Path to YAML config file

    Returns:
        Dictionary with configuration parameters

    Examples:
        >>> config = load_config('config.yaml')
        >>> print(config['grid'])
    """
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], filepath: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        filepath: Path to save YAML file

    Examples:
        >>> config = {'grid': 'NewEngland', 'epochs': 10}
        >>> save_config(config, 'my_config.yaml')
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for TranPy.

    Returns:
        Dictionary with default parameters
    """
    return {
        'grid': 'NewEngland',
        'test_size': 0.2,
        'random_state': 42,
        'model': {
            'type': 'svm',
            'kernel': 'rbf',
            'C': 1.0
        },
        'training': {
            'epochs': 10,
            'batch_size': 32,
            'validation_split': 0.2
        },
        'explainability': {
            'methods': ['lime', 'shap'],
            'n_features': 10
        }
    }
