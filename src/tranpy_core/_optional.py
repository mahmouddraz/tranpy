"""Lazy import of optional dependencies with an actionable error message."""

from __future__ import annotations

import importlib
from types import ModuleType

from tranpy_core.errors import MissingDependencyError

__all__ = ["is_available", "require"]


def require(module: str, extra: str) -> ModuleType:
    """Import ``module`` or raise :class:`MissingDependencyError` naming the extra to install."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependencyError(
            f"'{module}' is required for this feature. "
            f"Install it with: pip install 'tranpy-ai[{extra}]'"
        ) from exc


def is_available(module: str) -> bool:
    """Return whether ``module`` can be imported (imports it on success)."""
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True
