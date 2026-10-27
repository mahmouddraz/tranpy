"""Checksum-verified downloads into a per-user cache directory.

A file is only moved into the cache after its SHA-256 digest matches the expected value,
and a cached file is re-verified before every use. Nothing is written inside the
installed package.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import platformdirs

from tranpy_core.data.storage import sha256_file
from tranpy_core.errors import ChecksumError, ConfigurationError, DataError

__all__ = ["RemoteFile", "cache_dir", "fetch"]

logger = logging.getLogger(__name__)

_CACHE_ENV = "TRANPY_CACHE_DIR"


@dataclass(frozen=True)
class RemoteFile:
    """A downloadable file pinned by its checksum."""

    filename: str
    url: str
    sha256: str
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        problems = []
        if not self.filename or "/" in self.filename or self.filename.startswith("."):
            problems.append(f"invalid filename {self.filename!r}")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256.lower()):
            problems.append(f"sha256 must be 64 hex characters, got {self.sha256!r}")
        if not self.url.startswith(("https://", "file://")):
            problems.append(
                f"url must use https:// (or file:// for local mirrors), got {self.url!r}"
            )
        if problems:
            raise ConfigurationError("invalid remote file:", tuple(problems))


def cache_dir(app_name: str = "tranpy") -> Path:
    """Cache directory; ``$TRANPY_CACHE_DIR`` overrides the platform default. Not created here."""
    override = os.environ.get(_CACHE_ENV)
    return Path(override) if override else Path(platformdirs.user_cache_dir(app_name))


def fetch(remote: RemoteFile, *, directory: Path | None = None, timeout_s: float = 60.0) -> Path:
    """Return a local, verified copy of ``remote``, downloading it if needed."""
    target_dir = directory if directory is not None else cache_dir()
    target = target_dir / remote.filename
    if target.exists():
        if sha256_file(target) == remote.sha256.lower():
            return target
        logger.warning("cached %s fails its checksum; downloading it again", target)
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("downloading %s", remote.url)
    with tempfile.NamedTemporaryFile(dir=target_dir, prefix=".part-", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(remote.url, timeout=timeout_s) as response:
                shutil.copyfileobj(response, tmp)
        except OSError as exc:
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise DataError(f"download of {remote.url} failed: {exc}") from exc
    actual = sha256_file(tmp_path)
    if actual != remote.sha256.lower():
        tmp_path.unlink(missing_ok=True)
        raise ChecksumError(
            f"downloaded {remote.url} has sha256 {actual}, expected {remote.sha256}; file discarded"
        )
    tmp_path.replace(target)
    return target
