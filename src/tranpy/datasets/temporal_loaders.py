"""Temporal trajectory loaders for time series analysis.

This module provides functions to load raw event trajectory data from CSV files,
extract stability labels from trajectory data, and prepare data for temporal
feature extraction and machine learning.

Supports two data sources:
- **Local files**: Event CSVs in the package data directory (bundled examples)
- **Google Drive**: Stream event CSVs from public Google Drive folders on-demand

The loaders are designed to be scalable:
- n_events parameter controls how many events to load
- Google Drive files are cached locally after first download
- File ID index is built incrementally and persisted
"""

import io
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd


# Google Drive folder IDs for each grid
DRIVE_FOLDERS = {
    'NineBusSystem': '1bSb_TRky0r_lgrCHiQwAezIdAfVWLuMn',
    'NewEngland': '15wKSn-9UDSYLb_bRGtMrRNNqbQt6QRFc',
}


@dataclass
class TemporalEventData:
    """Container for a single event's temporal trajectory data.

    Attributes:
        event_id: Integer identifier for the event
        grid_name: Name of the power system ('NewEngland' or 'NineBusSystem')
        trajectory: DataFrame with time series data (columns are measurements)
        time: Time vector extracted from trajectory
        label: Stability label (0=stable, 1=unstable), None if not yet extracted
        fault_time: Time of fault inception in seconds (detected from voltage dip)
        clearing_time: Time of fault clearing in seconds (detected from voltage recovery)
        metadata: Optional dictionary with additional info
    """
    event_id: int
    grid_name: str
    trajectory: pd.DataFrame
    time: np.ndarray
    label: Optional[int] = None
    fault_time: Optional[float] = None
    clearing_time: Optional[float] = None
    metadata: Optional[Dict] = None

    def __repr__(self) -> str:
        n_timesteps = len(self.trajectory)
        n_features = len(self.trajectory.columns)
        label_str = f", label={self.label}" if self.label is not None else ""
        fault_str = ""
        if self.fault_time is not None:
            fault_str = f", fault={self.fault_time:.3f}s"
        if self.clearing_time is not None:
            fault_str += f", clear={self.clearing_time:.3f}s"
        return (
            f"TemporalEventData(event={self.event_id}, grid='{self.grid_name}', "
            f"timesteps={n_timesteps}, features={n_features}{label_str}{fault_str})"
        )

    @property
    def duration(self) -> float:
        """Return total duration of the trajectory in seconds."""
        return float(self.time[-1] - self.time[0])

    @property
    def sampling_rate(self) -> float:
        """Return average sampling rate in Hz."""
        return len(self.time) / self.duration

    @property
    def fault_duration(self) -> Optional[float]:
        """Return fault duration in seconds (clearing_time - fault_time)."""
        if self.fault_time is not None and self.clearing_time is not None:
            return self.clearing_time - self.fault_time
        return None

    @property
    def post_clearing_duration(self) -> Optional[float]:
        """Return available post-clearing observation time in seconds."""
        if self.clearing_time is not None:
            return float(self.time[-1]) - self.clearing_time
        return None

    def get_window(self, start_time: float = 0.0, end_time: Optional[float] = None) -> pd.DataFrame:
        """Extract a time window from the trajectory.

        Args:
            start_time: Start time in seconds (absolute simulation time)
            end_time: End time in seconds (None = end of trajectory)

        Returns:
            DataFrame with the windowed trajectory
        """
        mask = self.time >= start_time
        if end_time is not None:
            mask &= self.time <= end_time
        return self.trajectory[mask].copy()

    def get_post_clearing_window(self, observation_time: float) -> pd.DataFrame:
        """Extract a window of observation_time seconds after fault clearing.

        This is the physically meaningful window for early prediction:
        how much post-clearing trajectory do we need to observe?

        Args:
            observation_time: Seconds of post-clearing data to include

        Returns:
            DataFrame from fault clearing to clearing + observation_time

        Raises:
            ValueError: If clearing_time is not set
        """
        if self.clearing_time is None:
            raise ValueError("clearing_time not detected for this event")
        end_time = self.clearing_time + observation_time
        # Include pre-fault data for reference
        return self.get_window(end_time=end_time)


@dataclass
class TemporalDataset:
    """Container for multiple event trajectories.

    Attributes:
        events: List of TemporalEventData objects
        grid_name: Name of the power system
        labels: Array of stability labels (0=stable, 1=unstable)
    """
    events: List[TemporalEventData]
    grid_name: str

    def __repr__(self) -> str:
        n_events = len(self.events)
        n_stable = sum(1 for e in self.events if e.label == 0)
        n_unstable = sum(1 for e in self.events if e.label == 1)
        return (
            f"TemporalDataset(grid='{self.grid_name}', n_events={n_events}, "
            f"stable={n_stable}, unstable={n_unstable})"
        )

    @property
    def labels(self) -> np.ndarray:
        """Return array of stability labels."""
        return np.array([e.label for e in self.events])

    def get_stable_events(self) -> List[TemporalEventData]:
        """Return list of stable events."""
        return [e for e in self.events if e.label == 0]

    def get_unstable_events(self) -> List[TemporalEventData]:
        """Return list of unstable events."""
        return [e for e in self.events if e.label == 1]


# ---------------------------------------------------------------------------
# Path and cache management
# ---------------------------------------------------------------------------

def _get_cache_dir(grid: str) -> Path:
    """Get local cache directory for downloaded event files."""
    cache_dir = Path(__file__).parent.parent / 'data' / 'datasets' / 'events' / grid
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _get_index_path(grid: str) -> Path:
    """Get path to the file ID index JSON for a grid."""
    return _get_cache_dir(grid) / '_file_index.json'


def _load_file_index(grid: str) -> Dict[str, str]:
    """Load cached file ID index (event_filename -> google_drive_file_id)."""
    index_path = _get_index_path(grid)
    if index_path.exists():
        with open(index_path, 'r') as f:
            return json.load(f)
    return {}


def _save_file_index(grid: str, index: Dict[str, str]):
    """Save file ID index to cache."""
    index_path = _get_index_path(grid)
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)


def _load_api_key() -> Optional[str]:
    """Load Google API key from .env.local file."""
    env_paths = [
        Path(__file__).parent.parent.parent.parent.parent / '.env.local',  # project root
        Path.cwd() / '.env.local',
    ]
    for env_path in env_paths:
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('GOOGLE_API_KEY='):
                        return line.split('=', 1)[1].strip()
    # Also check environment variable
    return os.environ.get('GOOGLE_API_KEY')


def _build_file_index_api(grid: str, verbose: bool = True) -> Dict[str, str]:
    """Build file ID index using Google Drive API v3 (supports all 10,000+ files).

    Requires a GOOGLE_API_KEY in .env.local or environment variable.
    Paginates through the full folder contents.

    Args:
        grid: Grid name
        verbose: Print progress

    Returns:
        Dictionary mapping filename to Google Drive file ID
    """
    import requests

    api_key = _load_api_key()
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not found. Add it to .env.local or set the "
            "GOOGLE_API_KEY environment variable."
        )

    folder_id = DRIVE_FOLDERS.get(grid)
    if not folder_id:
        raise ValueError(f"No Google Drive folder configured for '{grid}'")

    if verbose:
        print(f"Building file index from Google Drive API for {grid}...")

    index = {}
    page_token = None
    page_count = 0

    while True:
        params = {
            'q': f"'{folder_id}' in parents and trashed = false",
            'key': api_key,
            'fields': 'nextPageToken, files(id, name)',
            'pageSize': 1000,
        }
        if page_token:
            params['pageToken'] = page_token

        resp = requests.get(
            'https://www.googleapis.com/drive/v3/files',
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        for f in data.get('files', []):
            index[f['name']] = f['id']

        page_count += 1
        if verbose:
            print(f"  Page {page_count}: {len(index)} files indexed so far...")

        page_token = data.get('nextPageToken')
        if not page_token:
            break

    if verbose:
        print(f"  Total: {len(index)} files indexed for {grid}")

    _save_file_index(grid, index)
    return index


def _build_file_index_gdown(grid: str, verbose: bool = True) -> Dict[str, str]:
    """Build file ID index from Google Drive using gdown (fallback, max ~50 files).

    Args:
        grid: Grid name
        verbose: Print progress

    Returns:
        Dictionary mapping filename to Google Drive file ID
    """
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required for Google Drive access. Install with: "
            "pip install gdown"
        )

    folder_id = DRIVE_FOLDERS.get(grid)
    if not folder_id:
        raise ValueError(f"No Google Drive folder configured for '{grid}'")

    url = f'https://drive.google.com/drive/folders/{folder_id}'

    if verbose:
        print(f"Building file index from Google Drive for {grid} (gdown fallback)...")
        print(f"  (gdown lists up to 50 files per call)")

    try:
        result = gdown.download_folder(
            url=url,
            skip_download=True,
            quiet=True,
            remaining_ok=True,
        )
    except Exception as e:
        if verbose:
            print(f"  Warning: gdown folder listing returned error: {e}")
        result = None

    if not result:
        if verbose:
            print("  No files found via gdown.")
        return {}

    index = {}
    for item in result:
        fname = item.path
        fid = item.id
        index[fname] = fid

    if verbose:
        print(f"  Indexed {len(index)} files from Google Drive")

    _save_file_index(grid, index)
    return index


def _build_file_index(grid: str, verbose: bool = True) -> Dict[str, str]:
    """Build file ID index from Google Drive.

    Uses Google Drive API v3 if GOOGLE_API_KEY is available (supports all files),
    otherwise falls back to gdown (limited to ~50 files).

    Args:
        grid: Grid name
        verbose: Print progress

    Returns:
        Dictionary mapping filename to Google Drive file ID
    """
    api_key = _load_api_key()
    if api_key:
        return _build_file_index_api(grid, verbose=verbose)
    else:
        if verbose:
            print("  No GOOGLE_API_KEY found, falling back to gdown (max ~50 files)")
        return _build_file_index_gdown(grid, verbose=verbose)


def _download_event_from_drive(
    grid: str,
    event_id: int,
    file_index: Optional[Dict[str, str]] = None,
    verbose: bool = False,
) -> Path:
    """Download a single event CSV from Google Drive to local cache.

    Args:
        grid: Grid name
        event_id: Event number
        file_index: Pre-loaded file index (None = load from cache)
        verbose: Print progress

    Returns:
        Path to the locally cached CSV file

    Raises:
        FileNotFoundError: If event is not in the file index
    """
    import requests

    cache_dir = _get_cache_dir(grid)
    local_path = cache_dir / f'results_event_{event_id}.csv'

    # Already cached locally
    if local_path.exists():
        return local_path

    # Load file index
    if file_index is None:
        file_index = _load_file_index(grid)

    fname = f'results_event_{event_id}.csv'
    file_id = file_index.get(fname)

    if not file_id:
        raise FileNotFoundError(
            f"Event {event_id} not found in Google Drive index for {grid}. "
            f"Index has {len(file_index)} files. "
            f"Run build_drive_index('{grid}') to refresh."
        )

    # Download from Google Drive (public file, no auth needed)
    url = f'https://drive.google.com/uc?export=download&id={file_id}'

    if verbose:
        print(f"  Downloading event {event_id} from Google Drive...")

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    # Save to cache
    local_path.write_bytes(resp.content)

    return local_path


# ---------------------------------------------------------------------------
# Public API: path resolution
# ---------------------------------------------------------------------------

def get_events_path(grid: str) -> Path:
    """Get path to event trajectory files directory.

    Args:
        grid: Grid name ('NewEngland' or 'NineBusSystem')

    Returns:
        Path to directory containing event CSV files

    Raises:
        ValueError: If grid name is not recognized
    """
    valid_grids = ['NewEngland', 'NineBusSystem']
    if grid not in valid_grids:
        raise ValueError(f"Unknown grid '{grid}'. Valid options: {valid_grids}")

    cache_dir = _get_cache_dir(grid)
    return cache_dir


def list_available_events(grid: str, include_drive: bool = True) -> List[int]:
    """List all available event IDs for a grid.

    Checks both local files and the Google Drive index.

    Args:
        grid: Grid name ('NewEngland' or 'NineBusSystem')
        include_drive: Also list events available on Google Drive

    Returns:
        Sorted list of event IDs (integers)
    """
    event_ids = set()

    # Local files
    events_path = _get_cache_dir(grid)
    for f in events_path.glob('results_event_*.csv'):
        try:
            event_id = int(f.stem.split('_')[-1])
            event_ids.add(event_id)
        except ValueError:
            continue

    # Google Drive index
    if include_drive:
        file_index = _load_file_index(grid)
        for fname in file_index:
            try:
                event_id = int(fname.replace('results_event_', '').replace('.csv', ''))
                event_ids.add(event_id)
            except ValueError:
                continue

    return sorted(event_ids)


def build_drive_index(grid: str, verbose: bool = True) -> Dict[str, str]:
    """Build/refresh the Google Drive file index for a grid.

    Call this once before loading events from Google Drive.
    The index is cached locally so subsequent calls are instant.

    Uses Google Drive API v3 (all files) if GOOGLE_API_KEY is set in
    .env.local or environment. Falls back to gdown (~50 files max).

    Args:
        grid: Grid name ('NewEngland' or 'NineBusSystem')
        verbose: Print progress

    Returns:
        Dictionary mapping filename to Google Drive file ID
    """
    return _build_file_index(grid, verbose=verbose)


# ---------------------------------------------------------------------------
# Core loading functions
# ---------------------------------------------------------------------------

def _parse_trajectory_csv(csv_path_or_buffer, source_label: str = '') -> pd.DataFrame:
    """Parse a trajectory CSV with multi-row header into a flat DataFrame.

    Args:
        csv_path_or_buffer: File path or BytesIO buffer
        source_label: Label for error messages

    Returns:
        DataFrame with flattened column names ("Source|Variable")
    """
    df = pd.read_csv(csv_path_or_buffer, header=[0, 1])

    # Flatten multi-level columns to "Source|Variable"
    new_columns = []
    for col in df.columns:
        source = (
            str(col[0])
            .split('\\')[-1]
            .replace('.ElmSym', '')
            .replace('.ElmTerm', '')
            .replace('.ElmLne', '')
            .replace('.ElmLod', '')
            .replace('.ElmTr2', '')
            .replace('.ElmRes', '')
        )
        var = str(col[1])
        new_columns.append(f"{source}|{var}")

    df.columns = new_columns
    return df


def load_event_trajectory(
    grid: str,
    event_id: int,
    extract_label: bool = True,
    source: str = 'auto',
) -> TemporalEventData:
    """Load a single event trajectory from local file or Google Drive.

    Args:
        grid: Grid name ('NewEngland' or 'NineBusSystem')
        event_id: Integer ID of the event to load
        extract_label: If True, extract stability label from trajectory
        source: Data source - 'local', 'drive', or 'auto' (try local first)

    Returns:
        TemporalEventData object with trajectory and metadata

    Raises:
        FileNotFoundError: If event file doesn't exist locally or on Drive
    """
    cache_dir = _get_cache_dir(grid)
    event_file = cache_dir / f'results_event_{event_id}.csv'

    # Try local first
    if source in ('auto', 'local') and event_file.exists():
        df = _parse_trajectory_csv(event_file)
        src_label = str(event_file)
    elif source in ('auto', 'drive'):
        # Try Google Drive
        try:
            downloaded_path = _download_event_from_drive(grid, event_id)
            df = _parse_trajectory_csv(downloaded_path)
            src_label = f'drive:{grid}/results_event_{event_id}.csv'
        except (FileNotFoundError, ImportError) as e:
            if source == 'drive':
                raise
            raise FileNotFoundError(
                f"Event {event_id} not found locally at {event_file} "
                f"and Google Drive download failed: {e}"
            )
    else:
        raise FileNotFoundError(f"Event file not found: {event_file}")

    # Extract time column (first column)
    time_col = df.columns[0]
    time = df[time_col].values

    trajectory = df.copy()

    label = None
    if extract_label:
        label = extract_stability_label(df)

    # Detect fault timing
    fault_time, clearing_time = detect_fault_timing(df, time)

    return TemporalEventData(
        event_id=event_id,
        grid_name=grid,
        trajectory=trajectory,
        time=time,
        label=label,
        fault_time=fault_time,
        clearing_time=clearing_time,
        metadata={
            'source': src_label,
            'n_timesteps': len(time),
            'time_range': (float(time[0]), float(time[-1])),
        }
    )


def detect_fault_timing(
    trajectory_df: pd.DataFrame,
    time: np.ndarray,
    voltage_threshold: float = 0.8,
) -> Tuple[Optional[float], Optional[float]]:
    """Detect fault inception and clearing times from voltage trajectory.

    Fault inception is detected as the first time any bus voltage drops below
    the threshold. Fault clearing is detected as the first time all bus voltages
    recover above the threshold after the dip.

    For unstable events where voltage never fully recovers, clearing_time is
    estimated from the first voltage recovery attempt (local minimum followed
    by partial recovery).

    Args:
        trajectory_df: DataFrame with trajectory data
        time: Time vector
        voltage_threshold: Voltage threshold in p.u. for fault detection

    Returns:
        Tuple of (fault_time, clearing_time) in seconds. Either may be None
        if detection fails.
    """
    v_cols = [c for c in trajectory_df.columns if 'm:u in p.u.' in c.lower()]
    if not v_cols:
        return None, None

    voltages = trajectory_df[v_cols].values  # (n_timesteps, n_buses)
    min_v = np.min(voltages, axis=1)

    # Find indices where min voltage is below threshold
    low_v_mask = min_v < voltage_threshold
    low_v_idx = np.where(low_v_mask)[0]

    if len(low_v_idx) == 0:
        return None, None

    # Fault inception: first voltage dip
    fault_time = float(time[low_v_idx[0]])

    # Fault clearing: find consecutive blocks of low voltage
    # The first block ending = fault clearing
    diffs = np.diff(low_v_idx)
    gaps = np.where(diffs > 1)[0]

    if len(gaps) > 0:
        # First gap = end of fault period
        first_block_end = low_v_idx[gaps[0]]
        clear_idx = first_block_end + 1
        if clear_idx < len(time):
            clearing_time = float(time[clear_idx])
        else:
            clearing_time = float(time[first_block_end])
    else:
        # All low-voltage indices are contiguous
        # If voltage is low until end of simulation -> unstable, use end of low block
        last_low = low_v_idx[-1]
        if last_low < len(time) - 2:
            clearing_time = float(time[last_low + 1])
        else:
            # Voltage never recovers — estimate clearing from voltage derivative
            # Look for the steepest recovery slope after the nadir
            nadir_idx = low_v_idx[0] + np.argmin(min_v[low_v_idx[0]:low_v_idx[-1]+1])
            dt = np.diff(time)
            dt[dt == 0] = 1e-6
            dv = np.diff(min_v) / dt
            # After nadir, find the first time dv becomes positive (recovery starts)
            post_nadir_dv = dv[nadir_idx:]
            recovery_start = np.where(post_nadir_dv > 0.1)[0]
            if len(recovery_start) > 0:
                clearing_time = float(time[nadir_idx + recovery_start[0]])
            else:
                clearing_time = float(time[nadir_idx])

    return fault_time, clearing_time


def extract_stability_label(trajectory_df: pd.DataFrame) -> int:
    """Extract stability label from trajectory data.

    A system is classified as unstable if any generator loses synchronism,
    indicated by the out-of-step flag (s:outofstep) being 1 at any point
    during the simulation.

    Args:
        trajectory_df: DataFrame with trajectory data

    Returns:
        0 = Stable (no generator loses synchronism)
        1 = Unstable (at least one generator out of step)
    """
    # Find all outofstep columns
    outofstep_cols = [c for c in trajectory_df.columns if 'outofstep' in c.lower()]

    if not outofstep_cols:
        # Fallback: check for large angle divergence
        angle_cols = [c for c in trajectory_df.columns if 'firot' in c.lower()]
        if angle_cols:
            angles = trajectory_df[angle_cols].values
            max_angle_diff = np.max(angles) - np.min(angles)
            return 1 if max_angle_diff > 180 else 0
        else:
            return 0

    # Check if any generator goes out of step at any time
    outofstep_values = trajectory_df[outofstep_cols].values
    return 1 if np.any(outofstep_values > 0) else 0


def load_temporal_dataset(
    grid: str,
    n_events: Optional[int] = None,
    event_ids: Optional[List[int]] = None,
    extract_labels: bool = True,
    source: str = 'auto',
    verbose: bool = True
) -> TemporalDataset:
    """Load multiple event trajectories as a TemporalDataset.

    Supports loading from local files, Google Drive, or both (auto mode).
    In auto mode, local files are used when available, and Google Drive
    is used as fallback.

    Args:
        grid: Grid name ('NewEngland' or 'NineBusSystem')
        n_events: Number of events to load (None = all available)
        event_ids: Specific event IDs to load (overrides n_events)
        extract_labels: If True, extract stability labels from trajectories
        source: 'local', 'drive', or 'auto' (try local, fallback to drive)
        verbose: If True, print progress messages

    Returns:
        TemporalDataset with loaded events

    Examples:
        >>> # Load first 5 local events for testing
        >>> dataset = load_temporal_dataset('NewEngland', n_events=5)

        >>> # Load from Google Drive (builds index first)
        >>> build_drive_index('NewEngland')
        >>> dataset = load_temporal_dataset('NewEngland', n_events=50, source='drive')
    """
    # Get list of events to load
    if event_ids is not None:
        ids_to_load = event_ids
    else:
        available_ids = list_available_events(grid, include_drive=(source != 'local'))
        if n_events is not None:
            ids_to_load = available_ids[:n_events]
        else:
            ids_to_load = available_ids

    if verbose:
        print(f"Loading {len(ids_to_load)} events from {grid} (source={source})...")

    events = []
    for i, event_id in enumerate(ids_to_load):
        try:
            event = load_event_trajectory(
                grid, event_id, extract_label=extract_labels, source=source
            )
            events.append(event)

            if verbose and (i + 1) % 50 == 0:
                print(f"  Loaded {i + 1}/{len(ids_to_load)} events...")

        except FileNotFoundError as e:
            if verbose:
                print(f"  Warning: Event {event_id} not found, skipping.")
            continue

    if verbose:
        n_stable = sum(1 for e in events if e.label == 0)
        n_unstable = sum(1 for e in events if e.label == 1)
        print(f"Loaded {len(events)} events: {n_stable} stable, {n_unstable} unstable")

    return TemporalDataset(events=events, grid_name=grid)


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

def get_generator_columns(
    trajectory_df: pd.DataFrame,
    generator_id: Optional[str] = None
) -> Dict[str, List[str]]:
    """Get column names grouped by generator.

    Args:
        trajectory_df: DataFrame with trajectory data
        generator_id: Specific generator ID to filter (e.g., 'G1', 'G 01')

    Returns:
        Dictionary mapping generator IDs to lists of column names
    """
    generators = {}

    for col in trajectory_df.columns:
        parts = col.split('|')
        if len(parts) >= 2:
            source = parts[0].strip()
            if 'G' in source and any(c.isdigit() for c in source):
                gen_id = source
                if generator_id is None or gen_id == generator_id:
                    if gen_id not in generators:
                        generators[gen_id] = []
                    generators[gen_id].append(col)

    return generators


def get_bus_columns(
    trajectory_df: pd.DataFrame,
    bus_id: Optional[str] = None
) -> Dict[str, List[str]]:
    """Get column names grouped by bus.

    Args:
        trajectory_df: DataFrame with trajectory data
        bus_id: Specific bus ID to filter (e.g., 'Bus 1')

    Returns:
        Dictionary mapping bus IDs to lists of column names
    """
    buses = {}

    for col in trajectory_df.columns:
        parts = col.split('|')
        if len(parts) >= 2:
            source = parts[0].strip()
            if 'Bus' in source:
                bus_name = source
                if bus_id is None or bus_name == bus_id:
                    if bus_name not in buses:
                        buses[bus_name] = []
                    buses[bus_name].append(col)

    return buses


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def load_newengland_events(
    n_events: Optional[int] = None,
    **kwargs
) -> TemporalDataset:
    """Load New England 39-bus event trajectories.

    Args:
        n_events: Number of events to load (None = all available)
        **kwargs: Additional arguments passed to load_temporal_dataset

    Returns:
        TemporalDataset with New England events
    """
    return load_temporal_dataset('NewEngland', n_events=n_events, **kwargs)


def load_ieee9bus_events(
    n_events: Optional[int] = None,
    **kwargs
) -> TemporalDataset:
    """Load IEEE 9-bus event trajectories.

    Args:
        n_events: Number of events to load (None = all available)
        **kwargs: Additional arguments passed to load_temporal_dataset

    Returns:
        TemporalDataset with IEEE 9-bus events
    """
    return load_temporal_dataset('NineBusSystem', n_events=n_events, **kwargs)
