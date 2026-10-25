"""Convert the legacy pickle / CSV datasets of tranpy <= 0.1.x to the safe .npz format.

Provenance record, run once by the maintainer. Pickles can execute code when loaded, so
only run this on files you created yourself.

    python scripts/convert_legacy_data.py snapshots LEGACY_DIR OUT_DIR
    python scripts/convert_legacy_data.py events CSV_DIR SYSTEM OUT_FILE

``snapshots`` reads ``NewEngland.pickle`` / ``NineBusSystem.pickle`` (a list whose item 7 is
the full DataFrame with a ``stable-unstable`` label column) and writes
``<system>_snapshots.npz``. ``events`` converts a folder of PowerFactory result CSVs.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from tranpy.datasets import read_event_csvs
from tranpy.systems import get_system
from tranpy.vocabulary import LABEL_NAMES
from tranpy_core.data import TabularDataset, save_signals, save_tabular, sha256_file

LEGACY = {"NewEngland": "new_england", "NineBusSystem": "ieee_9bus"}


def _install_pandas_shim() -> None:
    # the pickles reference pandas.core.indexes.numeric, removed in pandas 2.0
    if "pandas.core.indexes.numeric" not in sys.modules:
        shim = ModuleType("pandas.core.indexes.numeric")
        shim.Int64Index = shim.UInt64Index = shim.Float64Index = pd.Index  # type: ignore[attr-defined]
        sys.modules["pandas.core.indexes.numeric"] = shim


def convert_snapshots(legacy_dir: Path, out_dir: Path) -> None:
    _install_pandas_shim()
    for legacy_name, key in LEGACY.items():
        source = legacy_dir / f"{legacy_name}.pickle"
        with source.open("rb") as fh:
            frame = pickle.load(fh)[7]
        labels = frame["stable-unstable"].to_numpy().astype(np.int64)
        features = frame.drop(columns=["stable-unstable"])
        system = get_system(key)
        dataset = TabularDataset(
            features=features.to_numpy(np.float64),
            labels=labels,
            feature_names=tuple(map(str, features.columns)),
            label_names=LABEL_NAMES,
            name=key,
            description=(
                f"{system.name}: one feature vector per simulated fault event "
                f"(PowerFactory RMS simulation), label 0 = stable, 1 = unstable."
            ),
            sample_ids=frame.index.to_numpy(np.int64),
            metadata={
                "system": key,
                "source_file": source.name,
                "source_sha256": sha256_file(source),
                "feature_semantics": "unknown: columns F_0..F_n of the legacy dataset",
            },
        )
        path = save_tabular(dataset, out_dir / f"{key}_snapshots.npz")
        print(f"{path}  sha256={sha256_file(path)}  {dataset}")


def convert_events(csv_dir: Path, system: str, out_file: Path) -> None:
    dataset = read_event_csvs(csv_dir, system)
    path = save_signals(dataset, out_file)
    print(f"{path}  sha256={sha256_file(path)}  {dataset}")


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    if mode == "snapshots":
        convert_snapshots(Path(args[0]), Path(args[1]))
    elif mode == "events":
        convert_events(Path(args[0]), args[1], Path(args[2]))
    else:
        raise SystemExit(__doc__)
