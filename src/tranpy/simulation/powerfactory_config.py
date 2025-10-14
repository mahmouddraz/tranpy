"""PowerFactory configuration and path setup."""

import sys
import os
from pathlib import Path
from typing import Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


# Default PowerFactory installation paths by platform
POWERFACTORY_PATHS = {
    'windows': {
        '2019': r"C:\Program Files\DIgSILENT\PowerFactory 2019\Python\3.13",
        '2020': r"C:\Program Files\DIgSILENT\PowerFactory 2020\Python\3.8",
        '2021': r"C:\Program Files\DIgSILENT\PowerFactory 2021\Python\3.8",
        '2022': r"C:\Program Files\DIgSILENT\PowerFactory 2022\Python\3.10",
        '2023': r"C:\Program Files\DIgSILENT\PowerFactory 2023\Python\3.10",
    },
    'darwin': {  # macOS
        '2019': "/Applications/PowerFactory.app/Contents/Python/3.13",
        '2020': "/Applications/PowerFactory.app/Contents/Python/3.8",
    },
    'linux': {
        '2019': "/opt/digsilent/powerfactory/python/3.13",
        '2020': "/opt/digsilent/powerfactory/python/3.8",
    }
}


def get_platform():
    """Get current platform identifier."""
    if sys.platform.startswith('win'):
        return 'windows'
    elif sys.platform == 'darwin':
        return 'darwin'
    else:
        return 'linux'


def configure_powerfactory_path(
    custom_path: Optional[str] = None,
    version: str = '2019',
    python_version: Optional[str] = None
) -> bool:
    """
    Configure PowerFactory Python path for importing powerfactory module.

    Args:
        custom_path: Custom path to PowerFactory Python directory
        version: PowerFactory version ('2019', '2020', etc.)
        python_version: Specific Python version (e.g., '3.13')

    Returns:
        True if path was added, False if already in sys.path

    Examples:
        >>> # Auto-detect based on version
        >>> configure_powerfactory_path(version='2019')

        >>> # Custom path
        >>> configure_powerfactory_path(
        ...     custom_path=r"C:\\Program Files\\DIgSILENT\\PowerFactory 2019\\Python\\3.13"
        ... )

        >>> # Now you can import
        >>> import powerfactory as pf
        >>> app = pf.GetApplication()
    """
    if custom_path:
        # Use custom path provided by user
        pf_path = Path(custom_path)
    else:
        # Auto-detect based on platform and version
        platform = get_platform()

        if platform not in POWERFACTORY_PATHS:
            raise ValueError(
                f"Unsupported platform: {platform}\n"
                f"Please provide custom_path manually."
            )

        if version not in POWERFACTORY_PATHS[platform]:
            available = list(POWERFACTORY_PATHS[platform].keys())
            raise ValueError(
                f"PowerFactory version {version} not configured for {platform}.\n"
                f"Available versions: {available}\n"
                f"Please provide custom_path manually."
            )

        pf_path = Path(POWERFACTORY_PATHS[platform][version])

        # If specific Python version requested, update path
        if python_version:
            pf_path = pf_path.parent / python_version

    # Check if path exists
    if not pf_path.exists():
        raise FileNotFoundError(
            f"PowerFactory Python directory not found: {pf_path}\n\n"
            f"Please verify PowerFactory is installed or provide correct path:\n"
            f"  configure_powerfactory_path(custom_path='your/path/here')\n\n"
            f"Typical locations:\n"
            f"  Windows: C:\\Program Files\\DIgSILENT\\PowerFactory XXXX\\Python\\X.XX\n"
            f"  macOS: /Applications/PowerFactory.app/Contents/Python/X.XX\n"
            f"  Linux: /opt/digsilent/powerfactory/python/X.XX"
        )

    pf_path_str = str(pf_path)

    # Add to sys.path if not already there
    if pf_path_str not in sys.path:
        sys.path.insert(0, pf_path_str)
        logger.info(f"Added PowerFactory Python path: {pf_path_str}")
        return True
    else:
        logger.info(f"PowerFactory path already in sys.path: {pf_path_str}")
        return False


def is_powerfactory_available() -> bool:
    """
    Check if PowerFactory module can be imported.

    Returns:
        True if powerfactory module is available
    """
    try:
        import powerfactory
        return True
    except ImportError:
        return False


def import_powerfactory(
    custom_path: Optional[str] = None,
    version: str = '2019',
    python_version: Optional[str] = None
):
    """
    Configure path and import PowerFactory module.

    Args:
        custom_path: Custom path to PowerFactory Python directory
        version: PowerFactory version
        python_version: Specific Python version

    Returns:
        powerfactory module

    Raises:
        ImportError: If PowerFactory cannot be imported

    Examples:
        >>> # Method 1: Auto-configuration
        >>> pf = import_powerfactory(version='2019')
        >>> app = pf.GetApplication()

        >>> # Method 2: Custom path
        >>> pf = import_powerfactory(
        ...     custom_path=r"C:\\Program Files\\DIgSILENT\\PowerFactory 2019\\Python\\3.13"
        ... )
    """
    # Check if already available
    if is_powerfactory_available():
        import powerfactory
        return powerfactory

    # Configure path
    try:
        configure_powerfactory_path(custom_path, version, python_version)
    except (ValueError, FileNotFoundError) as e:
        raise ImportError(
            f"Failed to configure PowerFactory path: {e}\n\n"
            f"To use PowerFactory simulation:\n"
            f"1. Install DIgSILENT PowerFactory\n"
            f"2. Configure the path:\n"
            f"   from tranpy.simulation import configure_powerfactory_path\n"
            f"   configure_powerfactory_path(custom_path='your/path/here')\n"
            f"3. Or use pre-generated datasets:\n"
            f"   from tranpy.datasets import load_newengland"
        ) from e

    # Try to import
    try:
        import powerfactory
        logger.info("PowerFactory module imported successfully")
        return powerfactory
    except ImportError as e:
        raise ImportError(
            f"PowerFactory module not found after path configuration.\n"
            f"Path configured: {sys.path[0]}\n"
            f"Error: {e}\n\n"
            f"Please verify:\n"
            f"1. PowerFactory is installed correctly\n"
            f"2. The Python version matches your environment\n"
            f"3. The path contains the powerfactory module"
        ) from e


def get_powerfactory_info() -> dict:
    """
    Get information about PowerFactory installation.

    Returns:
        Dictionary with PowerFactory info or error details
    """
    info = {
        'available': is_powerfactory_available(),
        'platform': get_platform(),
        'sys_path': sys.path,
    }

    if info['available']:
        try:
            import powerfactory as pf
            info['module_path'] = pf.__file__
            # Try to get version
            try:
                app = pf.GetApplication()
                if app:
                    info['connected'] = True
                else:
                    info['connected'] = False
                    info['note'] = "PowerFactory module available but GetApplication() returned None"
            except Exception as e:
                info['connected'] = False
                info['error'] = str(e)
        except Exception as e:
            info['error'] = str(e)
    else:
        info['note'] = "PowerFactory module not available. Use configure_powerfactory_path() to set it up."

    return info
