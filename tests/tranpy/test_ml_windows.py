"""Assessment-time windows on traced events: instants, series, tables and exclusion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tranpy.ml_windows import (
    DEFAULT_INSTANTS,
    FAULT_START_S,
    GRID_DT,
    SERIES,
    CaseTraces,
    Instant,
    curve_frame,
    evaluate_window_file,
    load_merged_traces,
    sequence_table,
    system_series,
    trace_channels,
    window_features,
    window_tables,
    write_window_tables,
)
from tranpy_core.errors import ConfigurationError, DataError

GEN_MAP = [["G 0001", "M0", "Bus 1"], ["G 0002", "M1", "Bus 2"]]


def _merged(path, *, n_events=12, unstable_end=1.6):
    """A merged trace file: 50 ms grid, 10 ms from 0.9 s to clearing + 0.5 s, 6.6 s end;
    unstable events end at ``unstable_end``."""
    names = trace_channels(GEN_MAP)
    rows = []
    rng = np.random.default_rng(0)
    events = []
    for sid in range(n_events):
        tc = 0.1 + 0.05 * (sid % 8)
        label = int(sid % 3 == 0)
        end = unstable_end if label else 6.6
        coarse = np.arange(0.0, end + 1e-9, 0.05)
        fine = np.arange(0.9, min(end, FAULT_START_S + tc + 0.5) + 1e-9, 0.01)
        t = np.unique(np.round(np.r_[coarse, fine], 6))
        for tt in t:
            swing = 30.0 * (tt - FAULT_START_S) if tt >= FAULT_START_S else 0.0
            vals = [1.0, 0.0, 1.0, 10.0]  # bus 1 V, angle; bus 2 V, angle
            for k in range(2):
                vals += [1.0 + 0.001 * k * swing, 100.0 + k, 10.0, (k + 1) * swing + 5.0 * k]
            rows.append(
                {"sample_id": sid, "time": float(tt), "values": np.asarray(vals, dtype=np.float32)}
            )
        events.append(
            {
                "case": "case_t10",
                "sample_id": sid,
                "label": label,
                "clearing_time_s": tc,
                "shard": "r000",
            }
        )
    rng.shuffle(rows)
    pd.DataFrame(rows).to_parquet(path, index=False)
    assert len(names) == 12
    return pd.DataFrame(events)


@pytest.fixture
def case(tmp_path):
    ev = _merged(tmp_path / "case_t10.parquet")
    ds = load_merged_traces(tmp_path / "case_t10.parquet", GEN_MAP, ev)
    ct = CaseTraces("case_t10", ds, ("G 0001", "G 0002"), np.array([100.0, 300.0]))
    return ct, ev


def test_instants_parse_and_time() -> None:
    assert Instant.parse("pre") == Instant("pre")
    assert Instant.parse("clear+0.1s") == Instant("clear", 0.1)
    assert Instant.parse("inception+0.5s").name == "inception+0.5s"
    tc = np.array([1.2, 1.5])
    np.testing.assert_allclose(Instant("clear", 0.1).time(tc), [1.3, 1.6])
    np.testing.assert_allclose(Instant("inception", 0.3).time(tc), [1.3, 1.3])
    assert Instant("pre").start(tc) is None
    assert Instant("clear", 0.3).n_steps == 31
    assert Instant("pre").n_steps == 0
    with pytest.raises(ConfigurationError):
        Instant("pre", 0.1)
    with pytest.raises(ConfigurationError):
        Instant.parse("clear+0.1")
    assert [i.name for i in DEFAULT_INSTANTS][:3] == ["pre", "clear+0s", "clear+0.05s"]


def test_load_merged_traces_sorts_and_annotates(case) -> None:
    ct, ev = case
    ds = ct.data
    assert ds.n_samples == 12
    assert ds.n_channels == 12
    for i in range(ds.n_samples):
        assert np.all(np.diff(ds[i].time) > 0)
    np.testing.assert_allclose(
        ds.annotations["fault_clear_s"], FAULT_START_S + ev["clearing_time_s"]
    )
    assert ds.labels.sum() == 4


def test_system_series_values(case) -> None:
    ct, _ = case
    t, s = system_series(ct, 1)
    assert s.shape == (t.size, len(SERIES))
    k = int(np.flatnonzero(np.isclose(t, 1.5))[0])  # 0.5 s after inception: swing 15 deg
    row = dict(zip(SERIES, s[k], strict=True))
    # angles 15 and 35 (gen 2 has +5 offset), COI with weights 1:3 = 30
    assert row["angle_spread"] == pytest.approx(20.0)
    assert row["angle_max"] == pytest.approx(15.0)
    assert row["dw_coi"] == pytest.approx(0.75 * 0.015)
    assert row["p_total"] == pytest.approx(201.0)
    assert row["v_min"] == pytest.approx(1.0)


def test_window_features_and_exclusion(case) -> None:
    ct, ev = case
    pre = window_features(ct, Instant("pre"))
    assert len(pre) == 12 and set(pre.columns) == {"clearing_time_s", *(f"{s}|at" for s in SERIES)}
    assert pre["angle_spread|at"].eq(5.0).all()  # no swing before the fault
    late = window_features(ct, Instant("clear", 0.3))
    # unstable traces end at 1.6 s: usable only when clearing + 0.3 <= 1.6, i.e. tc <= 0.3
    expect = ev[(ev["label"] == 0) | (ev["clearing_time_s"] <= 0.3 + 1e-9)]
    assert sorted(late.index) == sorted(expect["sample_id"])
    assert f"{SERIES[1]}|slope" in late.columns
    sid = int(late.index[0])
    tc = float(ev.set_index("sample_id").loc[sid, "clearing_time_s"])
    # the spread grows 30 deg/s: slope and change over 0.3 s
    assert late.loc[sid, "angle_spread|slope"] == pytest.approx(30.0, rel=1e-3)
    assert late.loc[sid, "angle_spread|change"] == pytest.approx(9.0, rel=1e-3)
    assert late.loc[sid, "angle_spread|at"] == pytest.approx(5.0 + 30.0 * (tc + 0.3), rel=1e-3)
    zero = window_features(ct, Instant("clear", 0.0))
    assert not any(c.endswith("|slope") for c in zero.columns)


def test_sequence_table_shape(case) -> None:
    ct, _ = case
    seq = sequence_table(ct, Instant("inception", 0.3))
    assert seq.shape[1] == 31 * len(SERIES)
    assert list(seq.columns[:2]) == [f"s0|{SERIES[0]}", f"s0|{SERIES[1]}"]
    # time-major: step k of the spread equals 5 + 30 * k * dt
    cols = [f"s{k}|angle_spread" for k in range(31)]
    np.testing.assert_allclose(seq[cols].iloc[0], 5.0 + 30.0 * GRID_DT * np.arange(31), rtol=1e-6)
    with pytest.raises(ConfigurationError):
        sequence_table(ct, Instant("pre"))


def test_window_tables_and_writing(case, tmp_path) -> None:
    ct, ev = case
    strata = ev.assign(hour="case", fault_line=ev["sample_id"] % 4)
    line = ev[["case", "sample_id", "clearing_time_s"]].assign(**{"line|x": 0.1})
    weights = {("case_t10", int(i)): 2.0 for i in ev["sample_id"]}
    tables = window_tables(
        [ct],
        strata,
        instants=[Instant("pre"), Instant("clear", 0.3)],
        line_features=line,
        weights=weights,
    )
    t = tables["clear+0.3s"]
    assert set(t.inputs) == {"window", "window_line", "sequence"}
    assert t.n_excluded == 12 - len(t.strata)
    assert t.excluded["label"].eq(1).all()
    assert t.strata["weight"].eq(2.0).all()
    assert "line|x" in t.inputs["window_line"] and "clearing_time_s" in t.inputs["window_line"]
    assert len(t.inputs["sequence"]) == len(t.strata)
    assert "sequence" not in tables["pre"].inputs
    manifest = write_window_tables(tables, tmp_path / "w")
    assert manifest["clear+0.3s"]["n_steps"] == 31
    assert (tmp_path / "w" / "clear+0.3s__sequence.parquet").exists()
    with pytest.raises(DataError):
        window_tables([ct], pd.concat([strata, strata]), instants=[Instant("pre")])


def test_evaluate_window_file_weighted(case, tmp_path) -> None:
    from tranpy.ml_benchmark import Candidate

    ct, ev = case
    strata = ev.assign(hour=np.where(ev["sample_id"] % 2 == 0, "a", "b"))
    tables = window_tables([ct], strata, instants=[Instant("clear", 0.1)], sequences=False)
    write_window_tables(tables, tmp_path / "w")
    cand = Candidate("logistic_regression", protocol="hour")
    _inst, _inp, _c, result, error, _ = evaluate_window_file(
        tmp_path / "w", "clear+0.1s", "window", cand, 0, ["hour"], n_boot=20
    )
    assert error is None, error
    assert result is not None and result.protocol == "hour"
    assert "average_precision_lo" in result.metrics()
    assert set(result.by_stratum) == {"hour"}


def test_curve_frame() -> None:
    res = pd.DataFrame(
        {
            "instant": ["pre", "clear+0.1s", "inception+0.3s"],
            "model": ["m"] * 3,
            "average_precision": [0.4, 0.8, 0.9],
        }
    )
    c = curve_frame(res)
    assert list(c["window_s"]) == [-GRID_DT, 0.1, 0.3]
    assert list(c["reference"]) == ["pre", "clear", "inception"]
    assert "average_precision_mean" in c
