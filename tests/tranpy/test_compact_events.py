"""Compact event format (events_v2 --store compact): reading and feature extraction."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tranpy.datasets import load_compact_events, read_compact_event_table
from tranpy.features import TrajectoryFeatures

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

GENS = ("G 0001", "G 0002")
BUSES = ("Bus 0014", "Bus 0042")
CHANNELS = (
    *(f"{b}|{v}" for b in BUSES for v in ("m:u in p.u.", "m:phiu in deg")),
    *(
        f"{g}|{v}"
        for g in GENS
        for v in ("s:xspeed in p.u.", "s:P1 in MW", "s:Q1 in Mvar", "c:firot in deg")
    ),
)


def _grid(clear: float) -> np.ndarray:
    """Reduced grid as written by the pipeline: 50 ms, 10 ms in [0.9, clear + 0.5],
    duplicate samples at the fault and clearing instants."""
    coarse = np.round(np.arange(0, 6.6001, 0.05), 6)
    fine = np.round(np.arange(0.9, clear + 0.5 + 1e-9, 0.01), 6)
    return np.sort(np.r_[np.union1d(coarse, fine), 1.0, clear], kind="stable")


def _write_shard(folder, case: str, tag: str, ids: list[int], labels: list[int]) -> None:
    clear = [1.1 + 0.05 * k for k in range(len(ids))]
    header = {
        "format": "tranpy-compact-events",
        "version": 1,
        "case": case,
        "tag": tag,
        "label_names": ["stable", "unstable"],
    }
    events = pd.DataFrame(
        {
            "sample_id": np.asarray(ids, np.int64),
            "label": np.asarray(labels, np.int64),
            "fault_start_s": 1.0,
            "fault_clear_s": clear,
            "fault_line": 7.0,
            "tsi": [50.0 if lab == 0 else -90.0 for lab in labels],
            "clearing_time_s": np.round(np.asarray(clear) - 1.0, 6),
            "n_out_of_step": np.asarray(labels, np.int64),
            "traced": True,
        }
    )
    header["snap_columns"] = [f"snap:pre|{g}|delta" for g in GENS]
    tab = pa.Table.from_pandas(events, preserve_index=False)
    snap = np.arange(len(ids) * len(GENS), dtype=np.float32)
    tab = tab.append_column("snap", pa.FixedSizeListArray.from_arrays(pa.array(snap), len(GENS)))
    pq.write_table(
        tab.replace_schema_metadata({b"tranpy": json.dumps(header).encode()}),
        folder / f"{tag}.events.parquet",
        compression="zstd",
    )
    rng = np.random.default_rng(0)
    sid, time, vals = [], [], []
    for i, c in zip(ids, clear, strict=True):
        t = _grid(c)
        sid.append(np.full(t.size, i, np.int64))
        time.append(t)
        x = 1.0 + 0.01 * rng.standard_normal((t.size, len(CHANNELS)))
        x[:, [k for k, n in enumerate(CHANNELS) if n.endswith("deg")]] *= 20.0
        vals.append(x.astype(np.float32))
    v = np.vstack(vals)
    cols = {"sample_id": pa.array(np.concatenate(sid)), "time": pa.array(np.concatenate(time))}
    cols.update({n: pa.array(v[:, k]) for k, n in enumerate(CHANNELS)})
    th = dict(header, format="tranpy-compact-traces", channel_names=list(CHANNELS))
    pq.write_table(
        pa.table(cols).replace_schema_metadata({b"tranpy": json.dumps(th).encode()}),
        folder / f"{tag}.traces.parquet",
        compression="zstd",
    )


def test_compact_events_round_trip_and_features(tmp_path) -> None:
    case = tmp_path / "high_solar_t10"
    case.mkdir()
    _write_shard(case, "high_solar_t10", "r000", [0, 3], [0, 1])
    _write_shard(case, "high_solar_t10", "grid", [100000], [1])

    table = read_compact_event_table(tmp_path)
    assert len(table) == 3 and set(table["case"]) == {"high_solar_t10"}
    row = table[(table.shard == "r000") & (table.sample_id == 3)].iloc[0]
    assert row["label"] == 1
    assert row["snap:pre|G 0002|delta"] == 3.0  # second event, second generator

    ds = load_compact_events(tmp_path)
    assert ds.n_samples == 3
    assert ds.channel_names == CHANNELS
    assert sorted(ds.sample_ids.tolist()) == [0, 3, 100000]
    by_id = dict(zip(ds.sample_ids.tolist(), ds.labels.tolist(), strict=True))
    assert by_id == {0: 0, 3: 1, 100000: 1}
    assert np.allclose(ds.annotations["fault_start_s"], 1.0)
    k = ds.sample_ids.tolist().index(3)
    t = ds.time[ds.offsets[k] : ds.offsets[k + 1]]
    assert np.all(np.diff(t) >= 0) and t[-1] == pytest.approx(6.6)
    assert ds.values.dtype == np.float64

    features = TrajectoryFeatures(window_s=0.2, reference="fault_clear").transform(ds)
    assert features.features.shape[0] == 3
    assert np.isfinite(features.features).any(axis=0).all()
