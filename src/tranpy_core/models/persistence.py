"""Save and load fitted estimators with skops (optional ``[persistence]`` extra).

Unlike pickle, skops only instantiates types it has been told to trust. Loading refuses a
file that contains types outside scikit-learn/NumPy and this package unless they are
listed explicitly in ``trusted``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tranpy_core._optional import require
from tranpy_core.data.storage import verify_sha256
from tranpy_core.errors import ModelError

__all__ = ["load_model", "save_model"]

_TRUSTED_PREFIXES = (
    "sklearn.",
    "numpy.",
    "scipy.",
    "tranpy_core.",
    "tranpy.",
    "builtins.",
    "collections.",
)


def save_model(model: Any, path: str | Path) -> Path:
    """Write a fitted estimator to ``path`` (``.skops``)."""
    sio = require("skops.io", "persistence")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(model, path)
    return path


def load_model(path: str | Path, *, sha256: str | None = None, trusted: Sequence[str] = ()) -> Any:
    """Load an estimator saved with :func:`save_model`.

    Args:
        path: ``.skops`` file.
        sha256: expected checksum, verified before the file is parsed.
        trusted: extra fully qualified type names to allow.
    """
    sio = require("skops.io", "persistence")
    if sha256 is not None:
        verify_sha256(path, sha256)
    untrusted = sio.get_untrusted_types(file=path)
    refused = [t for t in untrusted if not t.startswith(_TRUSTED_PREFIXES) and t not in trusted]
    if refused:
        raise ModelError(f"{path} contains untrusted types: {', '.join(refused)}")
    return sio.load(path, trusted=list(untrusted))
