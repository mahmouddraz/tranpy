"""Power-system stability datasets.

Two kinds of data are available per test system:

* **snapshots** - one feature vector per fault event (:class:`~tranpy_core.TabularDataset`),
  bundled with the package: :func:`load_new_england`, :func:`load_ieee_9bus`.
* **events** - full simulated trajectories of every recorded variable
  (:class:`~tranpy_core.SignalDataset`), loaded with :func:`load_events` from a converted
  ``.npz`` file, from a folder of PowerFactory result CSVs, or downloaded (checksum
  verified) once a release URL is registered.

Labels: 0 = stable, 1 = unstable (any generator out of step at any time).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from tranpy.systems import PowerSystem, get_system
from tranpy.vocabulary import LABEL_NAMES, ElementKind, Stability, Variable, element_kind
from tranpy_core.data import (
    RemoteFile,
    ShardManifest,
    ShardWriter,
    SignalDataset,
    TabularDataset,
    fetch,
    load_signals,
    load_tabular,
    verify_sha256,
)
from tranpy_core.errors import DataError
from tranpy_core.features import segments as seg

__all__ = [
    "data_registry",
    "detect_fault_timing",
    "event_csv_ids",
    "flatten_powerfactory_column",
    "list_datasets",
    "load_events",
    "load_ieee_9bus",
    "load_new_england",
    "load_snapshots",
    "open_event_shards",
    "read_event_csvs",
    "stability_labels",
    "write_event_shards",
]

logger = logging.getLogger(__name__)

FAULT_START = "fault_start_s"
"""Annotation: detected fault inception time (first bus voltage below the threshold)."""
FAULT_CLEAR = "fault_clear_s"
"""Annotation: detected fault clearing time (first voltage recovery after the dip)."""

LABEL_CHANNELS: tuple[str, ...] = (f"*|{Variable.OUT_OF_STEP.value}",)
"""Channels that define the stability label (never use them as model input)."""
_EVENT_FILE = re.compile(r"^results_event_(\d+)\.csv$")
_ELEMENT_SUFFIXES = (".ElmSym", ".ElmTerm", ".ElmLne", ".ElmLod", ".ElmTr2", ".ElmRes")


@cache
def data_registry() -> dict[str, Any]:
    """Contents of the bundled ``resources/registry.json`` (files, checksums, URLs)."""
    text = resources.files("tranpy").joinpath("resources/registry.json").read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(text)
    return data


def _resource_path(filename: str) -> Path:
    path = Path(str(resources.files("tranpy").joinpath(f"resources/{filename}")))
    if not path.exists():
        raise DataError(f"bundled file {filename} is missing; reinstall tranpy-ai")
    return path


def list_datasets() -> pd.DataFrame:
    """Available datasets with their size and whether they are bundled."""
    rows = []
    for kind, entries in (
        ("snapshots", data_registry()["snapshots"]),
        ("events", data_registry()["events"]),
    ):
        for key, entry in entries.items():
            rows.append(
                {
                    "system": key,
                    "kind": kind,
                    "n_samples": entry.get("n_samples"),
                    "bundled": entry.get("bundled", False),
                    "download_url": entry.get("url"),
                    "filename": entry["filename"],
                }
            )
    return pd.DataFrame(rows)


def load_snapshots(system: str | PowerSystem) -> TabularDataset:
    """Bundled snapshot dataset of ``system`` (checksum verified)."""
    key = get_system(system).key
    entry = data_registry()["snapshots"].get(key)
    if entry is None:
        raise DataError(f"no snapshot dataset for system {key!r}")
    path = _resource_path(entry["filename"])
    verify_sha256(path, entry["sha256"])
    return load_tabular(path)


def load_new_england() -> TabularDataset:
    """Snapshot dataset of the IEEE 39-bus New England system (5000 events, 77 features)."""
    return load_snapshots("new_england")


def load_ieee_9bus() -> TabularDataset:
    """Snapshot dataset of the IEEE 9-bus system (5000 events, 17 features)."""
    return load_snapshots("ieee_9bus")


def load_events(
    system: str | PowerSystem,
    *,
    source: str | Path | None = None,
    channels: Sequence[str] | None = None,
    include_label_channels: bool = False,
    cache_directory: Path | None = None,
) -> SignalDataset:
    """Event trajectories of ``system``.

    Args:
        system: system key or legacy name.
        source: a ``.npz`` file written by :func:`tranpy_core.data.save_signals`, or a folder
            of PowerFactory result CSVs (``results_event_<id>.csv``). If omitted, the
            registered release file is downloaded to the user cache and verified.
        channels: shell-style patterns of channels to keep (default: all).
        include_label_channels: keep the out-of-step flags. They *define* the label, so they
            are dropped by default to keep them out of any feature set.
        cache_directory: download directory (default: user cache, see ``TRANPY_CACHE_DIR``).
    """
    spec = get_system(system)
    if source is None:
        entry = data_registry()["events"][spec.key]
        if not entry.get("url") or not entry.get("sha256"):
            raise DataError(
                f"the {spec.key} event dataset has no registered download yet. Pass "
                "source=<folder of results_event_*.csv> or a converted .npz file."
            )
        path = fetch(
            RemoteFile(filename=entry["filename"], url=entry["url"], sha256=entry["sha256"]),
            directory=cache_directory,
        )
        dataset = load_signals(path)
    else:
        src = Path(source)
        dataset = read_event_csvs(src, spec) if src.is_dir() else load_signals(src)
    if dataset.metadata.get("system") not in (None, spec.key):
        raise DataError(
            f"{source} holds data of system {dataset.metadata.get('system')!r}, not {spec.key!r}"
        )
    if not include_label_channels and dataset.channel_index(LABEL_CHANNELS).size:
        dataset = dataset.drop_channels(LABEL_CHANNELS)
    if channels is not None:
        dataset = dataset.select_channels(channels)
    return dataset


def event_csv_ids(directory: str | Path) -> list[int]:
    """Event ids of the ``results_event_<id>.csv`` files in ``directory``, sorted."""
    return sorted(
        int(m.group(1)) for p in Path(directory).iterdir() if (m := _EVENT_FILE.match(p.name))
    )


def write_event_shards(
    directory: str | Path,
    system: str | PowerSystem,
    out_dir: str | Path,
    *,
    shard_size: int = 250,
    event_ids: Sequence[int] | None = None,
) -> ShardManifest:
    """Convert a folder of PowerFactory CSVs into a sharded dataset, one shard at a time.

    Memory use is bounded by one shard. Resumable: events already in ``out_dir`` are skipped.
    """
    spec = get_system(system)
    writer = ShardWriter(
        out_dir,
        name=f"{spec.key}_events",
        shard_size=shard_size,
        metadata={"system": spec.key, "source": "PowerFactory RMS simulation (CSV export)"},
    )
    done = writer.done_sample_ids()
    todo = [
        i
        for i in (event_ids if event_ids is not None else event_csv_ids(directory))
        if i not in done
    ]
    for start in range(0, len(todo), shard_size):
        writer.add(read_event_csvs(directory, spec, event_ids=todo[start : start + shard_size]))
    return writer.close()


def open_event_shards(
    system: str | PowerSystem,
    *,
    source: str | Path | None = None,
    cache_directory: Path | None = None,
) -> ShardManifest:
    """Manifest of a sharded event dataset (for datasets too large to load at once).

    Args:
        system: system key or legacy name.
        source: a ``manifest.json`` or the folder containing it. If omitted, the manifest
            registered for ``system`` is downloaded and verified; its shards are then
            streamed on demand (see :func:`tranpy_core.data.iter_shards`).
        cache_directory: download directory for the manifest.
    """
    spec = get_system(system)
    if source is None:
        entry = data_registry()["events"][spec.key]
        if not entry.get("manifest_url") or not entry.get("manifest_sha256"):
            raise DataError(
                f"no sharded event dataset is registered for {spec.key!r}; pass source=<manifest>"
            )
        path = fetch(
            RemoteFile(
                filename=f"{spec.key}_events_manifest.json",
                url=entry["manifest_url"],
                sha256=entry["manifest_sha256"],
            ),
            directory=cache_directory,
        )
        manifest = ShardManifest.load(path)
        manifest = ShardManifest.from_dict(manifest.to_dict())  # remote: no local folder
    else:
        src = Path(source)
        manifest = ShardManifest.load(src / "manifest.json" if src.is_dir() else src)
    if manifest.metadata.get("system") not in (None, spec.key):
        raise DataError(
            f"{source} holds data of system {manifest.metadata.get('system')!r}, not {spec.key!r}"
        )
    return manifest


def flatten_powerfactory_column(column: Any) -> str:
    """Channel name ``"<element>|<variable>"`` from a two-row PowerFactory CSV header."""
    element = str(column[0]).split("\\")[-1]
    for suffix in _ELEMENT_SUFFIXES:
        element = element.replace(suffix, "")
    return f"{element}|{column[1]}"


def read_event_csvs(
    directory: str | Path,
    system: str | PowerSystem,
    *,
    event_ids: Sequence[int] | None = None,
    voltage_threshold_pu: float = 0.8,
) -> SignalDataset:
    """Read PowerFactory result CSVs (two header rows: element, variable) into a dataset.

    Events are ordered by event id. Labels come from the out-of-step flags
    (:func:`stability_labels`); fault start and clearing times are detected from the bus
    voltages (:func:`detect_fault_timing`) and stored as annotations.
    """
    spec = get_system(system)
    directory = Path(directory)
    available = sorted(
        (int(m.group(1)), p) for p in directory.iterdir() if (m := _EVENT_FILE.match(p.name))
    )
    if event_ids is not None:
        wanted = set(event_ids)
        missing = wanted - {i for i, _ in available}
        if missing:
            raise DataError(f"events not found in {directory}: {sorted(missing)[:10]}")
        available = [(i, p) for i, p in available if i in wanted]
    if not available:
        raise DataError(f"no results_event_<id>.csv files in {directory}")
    times, values, ids = [], [], []
    channel_names: tuple[str, ...] | None = None
    for event_id, path in available:
        frame = pd.read_csv(path, header=[0, 1])
        names = tuple(flatten_powerfactory_column(c) for c in frame.columns)
        if channel_names is None:
            channel_names = names
        elif names != channel_names:
            raise DataError(f"{path.name} has different columns than the first event file")
        data = frame.to_numpy(np.float64)
        times.append(data[:, 0])
        values.append(data[:, 1:])
        ids.append(event_id)
    assert channel_names is not None
    if not channel_names[0].endswith(Variable.TIME_S.value):
        raise DataError(f"first column must be the simulation time, got {channel_names[0]!r}")
    dataset = SignalDataset.from_samples(
        times,
        values,
        channel_names=channel_names[1:],
        labels=np.zeros(len(ids), dtype=np.int64),
        label_names=LABEL_NAMES,
        name=f"{spec.key}_events",
        sample_ids=np.asarray(ids, dtype=np.int64),
        description=f"{spec.name}: simulated fault-event trajectories (PowerFactory RMS)",
        metadata={
            "system": spec.key,
            "source": str(directory),
            "voltage_threshold_pu": voltage_threshold_pu,
        },
    )
    labels = stability_labels(dataset)
    start, clear = detect_fault_timing(dataset, voltage_threshold_pu=voltage_threshold_pu)
    return dataset.replace(labels=labels).with_annotations(
        **{FAULT_START: start, FAULT_CLEAR: clear}
    )


def _columns_of(
    dataset: SignalDataset, kind: ElementKind, variable: Variable
) -> npt.NDArray[np.int64]:
    suffix = "|" + variable.value
    return np.array(
        [
            i
            for i, c in enumerate(dataset.channel_names)
            if c.endswith(suffix) and element_kind(c.split("|", 1)[0]) is kind
        ],
        dtype=np.int64,
    )


def stability_labels(
    dataset: SignalDataset, *, angle_spread_limit_deg: float = 180.0
) -> npt.NDArray[np.int64]:
    """Label each event: unstable if any generator's out-of-step flag is ever set.

    Without out-of-step channels, an event is unstable if the spread of rotor angles over
    all generators and the whole trajectory exceeds ``angle_spread_limit_deg``.
    """
    flags = _columns_of(dataset, ElementKind.GENERATOR, Variable.OUT_OF_STEP)
    if flags.size:
        unstable = seg.segment_any((dataset.values[:, flags] > 0).any(axis=1), dataset.offsets)
    else:
        angles = _columns_of(dataset, ElementKind.GENERATOR, Variable.ROTOR_ANGLE_DEG)
        if angles.size == 0:
            raise DataError("cannot label events: no out-of-step flags and no rotor angles")
        a = dataset.values[:, angles]
        spread = seg.segment_max(a.max(axis=1), dataset.offsets) - seg.segment_min(
            a.min(axis=1), dataset.offsets
        )
        unstable = spread > angle_spread_limit_deg
    return np.where(unstable, Stability.UNSTABLE, Stability.STABLE).astype(np.int64)


def detect_fault_timing(
    dataset: SignalDataset, *, voltage_threshold_pu: float = 0.8
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Detect fault inception and clearing from the lowest bus voltage of each event.

    * inception: first time the lowest bus voltage drops below ``voltage_threshold_pu``;
    * clearing: first time it is back above the threshold after the first dip. If the
      voltage stays low until the end, the start of the steepest recovery (slope above
      0.1 p.u./s after the nadir) is used.

    Inception only uses data up to that instant. **Clearing is not causal**: for events
    whose voltage collapses it is found late and depends on the post-fault outcome, so
    windows measured from it carry label information (see docs/validation.md). Events
    without a dip get NaN.
    """
    cols = _columns_of(dataset, ElementKind.BUS, Variable.BUS_VOLTAGE_PU)
    n = dataset.n_samples
    start = np.full(n, np.nan)
    clear = np.full(n, np.nan)
    if cols.size == 0:
        return start, clear
    min_v_all = dataset.values[:, cols].min(axis=1)
    for i in range(n):
        lo, hi = int(dataset.offsets[i]), int(dataset.offsets[i + 1])
        time, min_v = dataset.time[lo:hi], min_v_all[lo:hi]
        low = np.flatnonzero(min_v < voltage_threshold_pu)
        if low.size == 0:
            continue
        start[i] = time[low[0]]
        gaps = np.flatnonzero(np.diff(low) > 1)
        if gaps.size:
            block_end = low[gaps[0]]
            clear[i] = time[block_end + 1] if block_end + 1 < time.size else time[block_end]
        elif low[-1] < time.size - 2:
            clear[i] = time[low[-1] + 1]
        else:
            nadir = low[0] + int(np.argmin(min_v[low[0] : low[-1] + 1]))
            dt = np.diff(time)
            dt[dt == 0] = 1e-6
            rising = np.flatnonzero((np.diff(min_v) / dt)[nadir:] > 0.1)
            clear[i] = time[nadir + rising[0]] if rising.size else time[nadir]
    return start, clear
