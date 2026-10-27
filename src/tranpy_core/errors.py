"""Exception hierarchy shared by the generic and the domain layer."""

from __future__ import annotations

__all__ = [
    "ChecksumError",
    "ConfigurationError",
    "DataError",
    "MissingDependencyError",
    "ModelError",
    "SimulationError",
    "TranpyError",
]


class TranpyError(Exception):
    """Base class of every error raised on purpose by this package."""


class ConfigurationError(TranpyError, ValueError):
    """Invalid configuration, specification or argument value.

    ``problems`` lists every issue found, so a config file can be fixed in one pass.
    """

    def __init__(self, message: str, problems: tuple[str, ...] = ()) -> None:
        self.problems = problems
        if problems:
            message = message + "\n" + "\n".join(f"  - {p}" for p in problems)
        super().__init__(message)


class DataError(TranpyError):
    """A dataset is missing, malformed or inconsistent."""


class ChecksumError(DataError):
    """A file does not match its recorded SHA-256 checksum."""


class ModelError(TranpyError):
    """A model cannot be built, fitted, loaded or used as requested."""


class SimulationError(TranpyError):
    """A simulation engine failed or returned unusable results."""


class MissingDependencyError(TranpyError, ImportError):
    """An optional dependency needed for the requested feature is not installed."""
