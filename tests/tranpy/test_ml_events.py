"""Event features are aligned across hours, outcome-free, and split by hour without overlap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tranpy.ml_events import (
    aligned_features,
    case_machines,
    check_no_leakage,
    event_frame,
    population_counts,
    system_features,
)
from tranpy_core.data.split import leave_one_group_out_splits
from tranpy_core.errors import DataError

VARS = ("delta", "omega", "P", "Q", "V")


def _case(case: str, machines: list[tuple[str, str, float]], n: int, seed: int):
    """Events of one case: machines are (name, stable bus id, H*Sn); G k follows list order."""
    rng = np.random.default_rng(seed)
    gen_map = [[f"G {k + 1:04d}", m, f"Bus {k}"] for k, (m, _, _) in enumerate(machines)]
    ids = pd.DataFrame({"machine": [m for m, _, _ in machines], "bus": [b for _, b, _ in machines]})
    dyn = pd.DataFrame(
        {"idx": [m for m, _, _ in machines], "Sn": 100.0, "M": [2 * w / 100 for *_, w in machines]}
    )
    rows = {
        "case": case,
        "shard": ["grid"] + ["r000"] * (n - 1),
        "sample_id": np.arange(n),
        "label": (np.arange(n) % 4 == 0).astype(int),
        "clearing_time_s": np.linspace(0.05, 0.55, n),
        "fault_line": rng.integers(0, 10, n).astype(float),
        "operating_point": 1.0,
        "tsi": rng.normal(size=n),
        "unstable_angle_360": 0.0,
    }
    for inst in ("pre", "fault", "clear"):
        for g, _, _ in gen_map:
            for var in VARS:
                base = 1.0 if var in ("omega", "V") else 0.0
                rows[f"snap:{inst}|{g}|{var}"] = base + rng.normal(scale=0.01, size=n)
    return pd.DataFrame(rows), case_machines(gen_map, ids, dyn)


@pytest.fixture
def two_hours():
    a, ma = _case("night_t10", [("M1", "busA", 4.0), ("M2", "busB", 2.0)], 6, 0)
    b, mb = _case("peak_t20", [("M9", "busB", 2.0), ("M3", "busC", 1.0), ("M1", "busA", 4.0)], 8, 1)
    events = pd.concat([a, b], ignore_index=True)
    return events, {"night_t10": ma, "peak_t20": mb}


def test_event_frame_adds_grouping_columns(two_hours) -> None:
    events, _ = two_hours
    frame = event_frame(events)
    assert set(frame["hour"]) == {"night", "peak"}
    assert set(frame["tan_phi"]) == {0.1, 0.2}
    assert frame.loc[0, "kind"] == "grid"
    assert frame.loc[1, "kind"] == "random"
    assert set(frame["event_type"]) == {"line_fault_3ph"}
    assert frame["clearing_band"].str.endswith("s").all()


def test_aligned_columns_follow_the_machine_not_the_position(two_hours) -> None:
    events, machines = two_hours
    X = aligned_features(events, machines, instants=("pre",))
    night, peak = events["case"] == "night_t10", events["case"] == "peak_t20"
    # busB is G 0002 in the night hour and G 0001 in the peak hour
    w = np.array([4.0, 2.0])
    delta = events.loc[night, ["snap:pre|G 0001|delta", "snap:pre|G 0002|delta"]].to_numpy()
    coi = delta @ w / w.sum()
    assert np.allclose(X.loc[night, "pre|busB|dangle"], delta[:, 1] - coi)
    assert np.allclose(X.loc[peak, "pre|busB|P"], events.loc[peak, "snap:pre|G 0001|P"])
    # busC is not committed at night: offline flag 0 and zero values
    assert (X.loc[night, "online|busC"] == 0).all()
    assert (X.loc[night, "pre|busC|V"] == 0).all()
    assert (X.loc[peak, "online|busC"] == 1).all()
    assert not X.isna().any().any()


def test_coi_relative_angles_have_zero_weighted_mean(two_hours) -> None:
    events, machines = two_hours
    X = aligned_features(events, machines, instants=("clear",))
    peak = events["case"] == "peak_t20"
    weighted = (
        X.loc[peak, "clear|busB|dangle"] * 2.0
        + X.loc[peak, "clear|busC|dangle"] * 1.0
        + X.loc[peak, "clear|busA|dangle"] * 4.0
    )
    assert np.allclose(weighted, 0.0)


def test_system_features_are_machine_set_independent(two_hours) -> None:
    events, machines = two_hours
    S = system_features(events, machines, instants=("pre", "clear"))
    assert {"pre|angle_max", "clear|v_min", "kinetic_energy_mws", "clearing_time_s"} <= set(S)
    assert set(S.loc[events["case"] == "peak_t20", "n_machines"]) == {3.0}
    assert set(S.loc[events["case"] == "night_t10", "kinetic_energy_mws"]) == {6.0}
    assert not S.isna().any().any()


def test_features_never_contain_outcome_columns(two_hours) -> None:
    events, machines = two_hours
    for X in (aligned_features(events, machines), system_features(events, machines)):
        assert not {"label", "tsi", "unstable_angle_360"} & set(X)
    with pytest.raises(DataError, match="outcome"):
        check_no_leakage(["clearing_time_s", "tsi"])
    with pytest.raises(DataError, match="outcome"):
        check_no_leakage(["unstable_coi_180"])


def test_missing_machine_table_or_instant_is_an_error(two_hours) -> None:
    events, machines = two_hours
    with pytest.raises(DataError, match="no machine table"):
        aligned_features(events, {"night_t10": machines["night_t10"]})
    with pytest.raises(DataError, match="unknown instants"):
        system_features(events, machines, instants=("after",))


def test_case_machines_rejects_unknown_machines() -> None:
    ids = pd.DataFrame({"machine": ["M1"], "bus": ["a"]})
    dyn = pd.DataFrame({"idx": ["M1"], "Sn": [1.0], "M": [2.0]})
    with pytest.raises(DataError, match="without identity"):
        case_machines([["G 0001", "M2", "Bus 1"]], ids, dyn)


def test_leave_one_hour_out_never_shares_an_hour(two_hours) -> None:
    events, _ = two_hours
    frame = event_frame(events)
    splits = leave_one_group_out_splits(frame["hour"])
    assert len(splits) == 2
    for split in splits:
        assert not set(frame["hour"].iloc[split.train]) & set(frame["hour"].iloc[split.test])


def test_population_counts_per_case_and_label(two_hours) -> None:
    events, _ = two_hours
    counts = population_counts(events)
    assert counts[("night_t10", 1)] == 2
    assert sum(counts.values()) == len(events)


def test_missing_snapshot_stays_missing_and_is_excluded(two_hours) -> None:
    from tranpy.ml_events import complete_events

    events, machines = two_hours
    events = events.copy()
    gone = [c for c in events if c.startswith("snap:clear|")]
    events.loc[2, gone] = np.nan  # no sample at the clearing instant for event 2
    A = aligned_features(events, machines, instants=("pre", "clear"))
    S = system_features(events, machines, instants=("pre", "clear"))
    assert A.loc[2, [c for c in A if "|" in c and not c.startswith("online")]].isna().all()
    assert not A.drop(index=2).isna().any().any()  # offline machines are still 0, not missing
    keep = complete_events({"aligned": A, "system": S, "pre": aligned_features(events, machines)})
    assert keep.tolist() == [i != 2 for i in range(len(events))]


def _chain_grid(n_lines: int):
    """A chain of ``n_lines`` 380 kV lines with a machine at the last bus."""
    from tranpy.line_features import CaseGrid

    n = n_lines + 1
    sheets = {
        "Bus": pd.DataFrame({"idx": np.arange(1, n + 1), "Vn": 380.0}),
        "Line": pd.DataFrame(
            {
                "idx": [f"L{k}" for k in range(n_lines)],
                "bus1": np.arange(1, n),
                "bus2": np.arange(2, n + 1),
                "Vn1": 380.0,
                "Vn2": 380.0,
                "r": 0.01,
                "x": 0.1,
                "b": 0.1,
                "g": 0.0,
                "b1": 0.0,
                "g1": 0.0,
                "b2": 0.0,
                "g2": 0.0,
                "trans": 0,
                "tap": 1.0,
                "phi": 0.0,
            }
        ),
        "GENROU": pd.DataFrame({"idx": ["M"], "bus": [n], "gen": ["G"], "Sn": 100.0, "M": 6.0}),
        "PV": pd.DataFrame({"idx": ["G"], "bus": [n], "p0": [0.5]}),
    }
    return CaseGrid.from_sheets(sheets, np.ones(n), -0.02 * np.arange(n))


def test_line_feature_sets_extend_the_old_ones(two_hours) -> None:
    from tranpy.ml_events import FEATURE_SETS, GRID_FEATURE_SETS, feature_sets

    events, machines = two_hours
    events = events.assign(fault_position=[k % 2 for k in range(len(events))])
    old = feature_sets(events, machines)
    assert set(old) == {"system_pre", "system_clear", "aligned_pre", "aligned_clear"}
    grids = {"night_t10": _chain_grid(10), "peak_t20": _chain_grid(10)}
    sets = feature_sets(events, machines, grids=grids)
    assert set(sets) == set(FEATURE_SETS) >= set(GRID_FEATURE_SETS)
    line_cols = {c for c in sets["line_pre"] if c.startswith("line|")}
    assert line_cols and set(sets["line_pre"]) == line_cols | {
        "clearing_time_s",
        "fault_line",
        "fault_position",
        "operating_point",
        "tan_phi",
    }
    assert set(sets["system_line_pre"]) == set(sets["system_pre"]) | line_cols
    assert set(sets["system_line_clear"]) == set(sets["system_clear"]) | line_cols
    pd.testing.assert_frame_equal(
        sets["system_line_pre"][list(sets["system_pre"])], sets["system_pre"]
    )
    for name in GRID_FEATURE_SETS:
        assert sets[name].index.equals(events.index)
        assert not sets[name].isna().any().any()
        assert isinstance(sets[name]["line|fault_line_cat"].dtype, pd.CategoricalDtype)
    with pytest.raises(DataError, match="CaseGrid"):
        feature_sets(events, machines, ["line_pre"])


def test_event_frame_caps_angle_separation_only_in_the_derived_frame(two_hours) -> None:
    from tranpy.ml_events import ANGLE_SEPARATION_CAP_DEG, cap_angle_separation

    events, _ = two_hours
    raw = np.full(len(events), 120.0)
    raw[0], raw[1], raw[2] = 2.1e4, 400.0, np.nan  # pole slip, just past the threshold, missing
    events = events.assign(angle_separation_deg=raw.copy())
    frame = event_frame(events)
    assert ANGLE_SEPARATION_CAP_DEG > 360.0
    assert frame["angle_separation_deg"].max() == ANGLE_SEPARATION_CAP_DEG
    assert frame.loc[0, "angle_separation_deg"] == ANGLE_SEPARATION_CAP_DEG
    assert frame.loc[1, "angle_separation_deg"] == 400.0
    assert np.isnan(frame.loc[2, "angle_separation_deg"])
    assert (frame.loc[3:, "angle_separation_deg"] == 120.0).all()
    # the input (the stored raw events) is untouched
    assert events.loc[0, "angle_separation_deg"] == 2.1e4
    # capped values stay on the unstable side of the 360 degree label threshold
    assert (frame.loc[frame["angle_separation_deg"] >= 360.0].index == [0, 1]).all()
    with pytest.raises(ValueError, match="above the 360"):
        cap_angle_separation(pd.Series([1.0]), cap_deg=360.0)


def test_event_frame_without_separation_column_is_unchanged(two_hours) -> None:
    events, _ = two_hours
    assert "angle_separation_deg" not in event_frame(events)
