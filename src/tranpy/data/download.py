"""Google Drive download utilities for TranPy."""

import requests
from pathlib import Path
from typing import Optional
import sys

from ..utils.logging import get_logger

logger = get_logger(__name__)


def download_from_google_drive(
    file_id: str,
    destination: Path,
    filename: str,
    show_progress: bool = True
) -> Path:
    """
    Download file from Google Drive.

    Args:
        file_id: Google Drive file ID
        destination: Local destination path
        filename: Original filename (for display)
        show_progress: Show download progress

    Returns:
        Path to downloaded file

    Raises:
        requests.RequestException: If download fails

    Examples:
        >>> from tranpy.data import download_from_google_drive
        >>> path = download_from_google_drive(
        ...     file_id='1eXtw44VXhYM0jQyJGGY5Eevdrg8yui0w',
        ...     destination=Path('~/.tranpy/datasets/NewEngland.pickle'),
        ...     filename='NewEngland.pickle'
        ... )
    """
    destination = Path(destination).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Google Drive direct download URL
    URL = "https://drive.google.com/uc?export=download"

    logger.info(f"Downloading {filename} from Google Drive...")

    session = requests.Session()

    try:
        # Initial request
        response = session.get(URL, params={'id': file_id}, stream=True)
        response.raise_for_status()

        # Handle large files (confirmation required)
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)
            response.raise_for_status()

        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))

        # Download file
        downloaded = 0
        chunk_size = 8192

        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if show_progress and total_size > 0:
                        # Simple progress indicator
                        progress = (downloaded / total_size) * 100
                        sys.stdout.write(
                            f"\rProgress: {progress:.1f}% "
                            f"({downloaded / (1024*1024):.1f} MB / "
                            f"{total_size / (1024*1024):.1f} MB)"
                        )
                        sys.stdout.flush()

        if show_progress:
            print()  # New line after progress
        logger.info(f"Downloaded to: {destination}")

        return destination

    except requests.RequestException as e:
        if destination.exists():
            destination.unlink()  # Clean up partial download
        raise RuntimeError(
            f"Failed to download {filename} from Google Drive.\n"
            f"Error: {e}\n"
            f"File ID: {file_id}\n"
            f"Please check your internet connection or download manually."
        ) from e


def get_google_drive_direct_link(file_id: str) -> str:
    """
    Generate Google Drive direct download link.

    Args:
        file_id: Google Drive file ID

    Returns:
        Direct download URL
    """
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def extract_file_id_from_link(link: str) -> Optional[str]:
    """
    Extract file ID from Google Drive link.

    Args:
        link: Google Drive URL

    Returns:
        File ID if found, None otherwise

    Examples:
        >>> link = "https://drive.google.com/file/d/1eXtw44VXhYM0jQyJGGY5Eevdrg8yui0w/view"
        >>> extract_file_id_from_link(link)
        '1eXtw44VXhYM0jQyJGGY5Eevdrg8yui0w'
    """
    # Pattern: /d/{file_id}/ or id={file_id}
    if '/d/' in link:
        try:
            return link.split('/d/')[1].split('/')[0]
        except IndexError:
            pass

    if 'id=' in link:
        try:
            return link.split('id=')[1].split('&')[0]
        except IndexError:
            pass

    return None
