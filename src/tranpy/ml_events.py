"""Machine-learning inputs from the compact fault-event store.

One row per simulated event (a three-phase line fault cleared by tripping the line). The
operating hours commit different generator sets, so the per-generator snapshot columns
``snap:<instant>|G <k>|<var>`` of different hours do not refer to the same machines.
Two established ways to get one fixed-length input for all hours are provided:

* :func:`aligned_features` - per machine, keyed by a stable machine id (the terminal bus
  of German units, the border bus of the Continental Europe equivalents). Rotor angles are
  taken relative to the centre of inertia (COI) of the hour, speeds as deviation from
  nominal. A machine that is not committed in an hour has ``online|<id> = 0`` and zeros in
  its other columns.
* :func:`system_features` - system-level statistics of the same snapshots (largest
  COI-relative angle, angle spread, largest speed deviation, totals, lowest voltage,
  committed kinetic energy), independent of which machines are committed.

Both add the scenario inputs of the event (:data:`SCENARIO_COLUMNS`). Instants are
``pre`` (before the fault), ``fault`` and ``clear`` (the faulted state at the clearing
instant): ``("pre",)`` gives a pre-fault assessment, :data:`AT_CLEARING` an assessment at
the moment of clearing. In the v4-dyn store the ``fault`` snapshot equals ``pre`` for every
event (the simulation output has one sample at the fault instant, before the switching),
so it carries no information and is left out of :data:`AT_CLEARING`. Columns derived from
the simulated outcome are refused (:func:`check_no_leakage`).

Grouping columns for splits and per-group reports come from :func:`event_frame`:
``hour`` (the operating hour, both reactive-load variants together), ``tan_phi``,
``clearing_band``, ``event_type`` and ``kind`` (random or grid-sweep event).
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tranpy_core.errors import DataError

__all__ = [
    "AT_CLEARING",
    "FEATURE_SETS",
    "INSTANTS",
    "LEAKY_COLUMNS",
    "SCENARIO_COLUMNS",
    "aligned_features",
    "case_machines",
    "check_no_leakage",
    "complete_events",
    "event_frame",
    "expand_snap_arrays",
    "feature_sets",
    "population_counts",
    "read_case_machines",
    "read_gen_maps",
    "system_features",
]

INSTANTS = ("pre", "fault", "clear")
AT_CLEARING = ("pre", "clear")

SCENARIO_COLUMNS = (
    "clearing_time_s",
    "fault_line",
    "fault_position",
    "fault_resistance_ohm",
    "fault_reactance_ohm",
    "operating_point",
    "mean_load_factor",
    "tan_phi",
)
"""Inputs known before the outcome: what fault, where, how long, at which loading."""

LEAKY_COLUMNS = (
    "label",
    "tsi",
    "angle_separation_deg",
    "n_out_of_step",
    "traced",
    "simulation_end_s",
)
"""Outcome columns (or columns set from the outcome); ``unstable_*`` is refused as well."""

DEFAULT_EVENT_TYPE = "line_fault_3ph"
CLEARING_BANDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, np.inf)

_CASE = re.compile(r"^(?P<hour>.+)_t(?P<tan>\d+)$")
_SNAP = re.compile(r"^snap:(?P<inst>[^|]+)\|(?P<gen>[^|]+)\|(?P<var>[^|]+)$")


def check_no_leakage(columns: Iterable[str]) -> None:
    """Raise :class:`DataError` if a feature column is derived from the simulated outcome."""
    bad = [c for c in columns if c in LEAKY_COLUMNS or str(c).startswith("unstable_")]
    if bad:
        raise DataError(f"outcome-derived columns in the features: {sorted(bad)}")


def event_frame(events: pd.DataFrame) -> pd.DataFrame:
    """``events`` with the grouping columns ``hour``, ``tan_phi``, ``clearing_band``,
    ``event_type`` and ``kind`` added (existing columns of these names are kept)."""
    out = events.copy()
    parsed = out["case"].astype(str).str.extract(_CASE)
    if parsed["hour"].isna().any():
        bad = sorted(out.loc[parsed["hour"].isna(), "case"].astype(str).unique())
        raise DataError(f"case names without a _t<tan> suffix: {bad}")
    out["hour"] = parsed["hour"]
    if "tan_phi" not in out:
        out["tan_phi"] = parsed["tan"].astype(int) / 100.0
    if "event_type" not in out:
        out["event_type"] = DEFAULT_EVENT_TYPE
    if "kind" not in out:
        out["kind"] = np.where(out["shard"].astype(str) == "grid", "grid", "random")
    out["clearing_band"] = pd.cut(
        out["clearing_time_s"], CLEARING_BANDS, right=False, labels=_band_labels()
    ).astype(str)
    return out


def _band_labels() -> list[str]:
    edges = CLEARING_BANDS
    return [
        f"{a:.1f}-{b:.1f}s" if np.isfinite(b) else f">={a:.1f}s"
        for a, b in itertools.pairwise(edges)
    ]


def population_counts(events: pd.DataFrame, by: str = "case") -> dict[tuple[Any, int], int]:
    """Events per ``(group, label)``: the population for inclusion weights of a subsample."""
    counts = events.groupby([by, "label"]).size()
    return {(key[0], int(key[1])): int(n) for key, n in counts.items()}  # type: ignore[index]


def case_machines(
    gen_map: Sequence[Sequence[str]],
    machines: pd.DataFrame,
    dynamics: pd.DataFrame,
) -> pd.DataFrame:
    """Stable identity and inertia of the snapshot generators of one case.

    Args:
        gen_map: ``[snapshot name, machine, bus]`` per snapshot generator, as in the
            compact-events header (``gen_map``).
        machines: ``machine`` and ``bus`` (stable terminal or border bus id) of every
            machine of the case, German units and the neighbour equivalents together.
        dynamics: ``idx``, ``Sn`` (MVA) and ``M`` (2H, s) of every machine model.

    Returns:
        One row per snapshot generator: ``gen``, ``machine``, ``machine_id``, ``sn_mva``,
        ``kinetic_mws`` (H * Sn, the COI weight).
    """
    gm = pd.DataFrame([list(r)[:2] for r in gen_map], columns=["gen", "machine"])
    ids = machines.drop_duplicates("machine").set_index("machine")["bus"].astype(str)
    dyn = dynamics.drop_duplicates("idx").set_index("idx")
    missing = sorted(set(gm["machine"]) - set(ids.index) | set(gm["machine"]) - set(dyn.index))
    if missing:
        raise DataError(f"snapshot machines without identity or dynamics: {missing[:10]}")
    out = gm.assign(
        machine_id=gm["machine"].map(ids),
        sn_mva=gm["machine"].map(dyn["Sn"]).astype(float),
    )
    out["kinetic_mws"] = gm["machine"].map(dyn["M"]).astype(float) / 2.0 * out["sn_mva"]
    if out["machine_id"].duplicated().any():
        dup = out.loc[out["machine_id"].duplicated(), "machine_id"].tolist()
        raise DataError(f"two snapshot machines share a machine id: {dup[:5]}")
    return out


def read_case_machines(
    prep_dir: str | Path, case: str, gen_map: Sequence[Sequence[str]]
) -> pd.DataFrame:
    """:func:`case_machines` from a case folder (``CASE.machines.csv``,
    ``CASE.neighbours.csv`` and the ANDES workbook ``CASE.xlsx``)."""
    folder = Path(prep_dir)
    parts = [pd.read_csv(folder / f"{case}.machines.csv", usecols=["machine", "bus"])]
    neighbours = folder / f"{case}.neighbours.csv"
    if neighbours.exists():
        parts.append(pd.read_csv(neighbours, usecols=["machine", "bus"]))
    sheets = pd.read_excel(folder / f"{case}.xlsx", sheet_name=None)
    dynamics = pd.concat(
        [sheets[s][["idx", "Sn", "M"]] for s in sheets if s in ("GENCLS", "GENROU")],
        ignore_index=True,
    )
    return case_machines(gen_map, pd.concat(parts, ignore_index=True), dynamics)


def read_gen_maps(source: str | Path) -> dict[str, list[list[str]]]:
    """``gen_map`` of every case in a compact event store (from the events-file headers)."""
    from tranpy.datasets import _COMPACT_EVENTS, _compact_files, _compact_header

    maps: dict[str, list[list[str]]] = {}
    for f in _compact_files(source, _COMPACT_EVENTS):
        header = _compact_header(f)
        case = str(header.get("case", f.parent.name))
        gen_map = [list(map(str, r)) for r in header["gen_map"]]
        if maps.setdefault(case, gen_map) != gen_map:
            raise DataError(f"case {case!r}: shards disagree on the generator map")
    return maps


def _snapshots(frame: pd.DataFrame, instants: Sequence[str]) -> dict[tuple[str, str, str], str]:
    found: dict[tuple[str, str, str], str] = {}
    for c in frame.columns:
        m = _SNAP.match(str(c))
        if m and m["inst"] in instants:
            found[(m["inst"], m["gen"], m["var"])] = c
    return found


def _case_blocks(
    events: pd.DataFrame,
    machines_by_case: Mapping[str, pd.DataFrame],
    instants: Sequence[str],
) -> Iterable[tuple[pd.Index, pd.DataFrame, dict[str, dict[str, np.ndarray]]]]:
    """Per case: row index, machine table, and per instant the arrays (events x machines)
    of COI-relative angle ``dangle``, speed deviation ``dw``, ``P``, ``Q`` and ``V``."""
    unknown = set(instants) - set(INSTANTS)
    if unknown:
        raise DataError(f"unknown instants {sorted(unknown)}; use {INSTANTS}")
    for case, rows in events.groupby("case", sort=True):
        if case not in machines_by_case:
            raise DataError(f"no machine table for case {case!r}")
        mt = machines_by_case[str(case)]
        cols = _snapshots(rows.dropna(axis=1, how="all"), instants)
        weights = mt["kinetic_mws"].to_numpy(np.float64)
        per_instant: dict[str, dict[str, np.ndarray]] = {}
        gens = mt["gen"].tolist()
        for inst in instants:
            arrays = {}
            for var in ("delta", "omega", "P", "Q", "V"):
                try:
                    names = [cols[(inst, g, var)] for g in gens]
                except KeyError as exc:
                    raise DataError(f"case {case!r}: snapshot {exc.args[0]} missing") from exc
                arrays[var] = rows[names].to_numpy(np.float64)
            coi = arrays["delta"] @ weights / weights.sum()
            per_instant[inst] = {
                "dangle": arrays["delta"] - coi[:, None],
                "dw": arrays["omega"] - 1.0,
                "P": arrays["P"],
                "Q": arrays["Q"],
                "V": arrays["V"],
            }
        yield rows.index, mt, per_instant


def _scenario(events: pd.DataFrame) -> pd.DataFrame:
    frame = event_frame(events) if "tan_phi" not in events else events
    present = [c for c in SCENARIO_COLUMNS if c in frame]
    return frame[present].astype(float)


def aligned_features(
    events: pd.DataFrame,
    machines_by_case: Mapping[str, pd.DataFrame],
    *,
    instants: Sequence[str] = ("pre",),
) -> pd.DataFrame:
    """Per-machine features aligned by stable machine id across all cases.

    Columns: the scenario inputs, ``online|<id>``, and ``<instant>|<id>|<var>`` for the
    variables ``dangle`` (rotor angle relative to the COI, in the snapshot unit: degrees),
    ``dw`` (pu speed deviation), ``P``, ``Q``, ``V``. Machines not committed in an event's
    hour are 0 in every column. Events with a missing snapshot of a committed machine are
    missing in every machine column.
    """
    blocks = []
    incomplete: list[Any] = []
    for index, mt, per_instant in _case_blocks(events, machines_by_case, instants):
        data: dict[str, np.ndarray] = {}
        for k, mid in enumerate(mt["machine_id"]):
            data[f"online|{mid}"] = np.ones(len(index))
            for inst in instants:
                for var, arr in per_instant[inst].items():
                    data[f"{inst}|{mid}|{var}"] = arr[:, k]
        block = pd.DataFrame(data, index=index)
        incomplete.extend(block.index[block.isna().any(axis=1)])
        blocks.append(block)
    machine_cols = pd.concat(blocks).reindex(events.index)
    # 0 for machines not committed in the hour; a missing snapshot of a committed machine
    # stays missing (whole row), so incomplete events are visible and can be excluded
    machine_cols = machine_cols[sorted(machine_cols.columns)].fillna(0.0)
    machine_cols.loc[incomplete] = np.nan
    out = pd.concat([_scenario(events), machine_cols], axis=1)
    check_no_leakage(out.columns)
    return out


def system_features(
    events: pd.DataFrame,
    machines_by_case: Mapping[str, pd.DataFrame],
    *,
    instants: Sequence[str] = ("pre",),
) -> pd.DataFrame:
    """System-level features of the snapshots, independent of the committed machine set.

    Per instant: ``angle_max`` (largest absolute COI-relative angle), ``angle_spread``
    (max - min angle), ``angle_std`` (inertia-weighted), ``dw_max`` (largest absolute speed
    deviation), ``dw_coi`` (inertia-weighted mean speed deviation), ``p_total``,
    ``q_total``, ``v_min``, ``v_mean``. Per case: ``kinetic_energy_mws`` (sum of H * Sn)
    and ``n_machines``. Plus the scenario inputs.
    """
    blocks = []
    for index, mt, per_instant in _case_blocks(events, machines_by_case, instants):
        w = mt["kinetic_mws"].to_numpy(np.float64)
        wn = w / w.sum()
        data: dict[str, Any] = {
            "kinetic_energy_mws": np.full(len(index), w.sum()),
            "n_machines": np.full(len(index), float(len(mt))),
        }
        for inst in instants:
            a = per_instant[inst]
            data[f"{inst}|angle_max"] = np.abs(a["dangle"]).max(axis=1)
            data[f"{inst}|angle_spread"] = a["dangle"].max(axis=1) - a["dangle"].min(axis=1)
            data[f"{inst}|angle_std"] = np.sqrt((a["dangle"] ** 2) @ wn)
            data[f"{inst}|dw_max"] = np.abs(a["dw"]).max(axis=1)
            data[f"{inst}|dw_coi"] = a["dw"] @ wn
            data[f"{inst}|p_total"] = a["P"].sum(axis=1)
            data[f"{inst}|q_total"] = a["Q"].sum(axis=1)
            data[f"{inst}|v_min"] = a["V"].min(axis=1)
            data[f"{inst}|v_mean"] = a["V"].mean(axis=1)
        blocks.append(pd.DataFrame(data, index=index))
    system = pd.concat(blocks).reindex(events.index)
    out = pd.concat([_scenario(events), system], axis=1)
    check_no_leakage(out.columns)
    return out


FEATURE_SETS = {
    "system_pre": (system_features, ("pre",)),
    "system_clear": (system_features, AT_CLEARING),
    "aligned_pre": (aligned_features, ("pre",)),
    "aligned_clear": (aligned_features, AT_CLEARING),
}
"""Named feature sets: representation (system-level or aligned per machine) x assessment
time (pre-fault, or at clearing)."""


def feature_sets(
    events: pd.DataFrame,
    machines_by_case: Mapping[str, pd.DataFrame],
    names: Sequence[str] = tuple(FEATURE_SETS),
) -> dict[str, pd.DataFrame]:
    """The feature sets ``names`` of ``events`` (rows in the order of ``events``)."""
    unknown = sorted(set(names) - set(FEATURE_SETS))
    if unknown:
        raise DataError(f"unknown feature sets {unknown}; use {sorted(FEATURE_SETS)}")
    out = {}
    for name in names:
        builder, instants = FEATURE_SETS[name]
        out[name] = builder(events, machines_by_case, instants=instants)
    return out


def expand_snap_arrays(
    events: pd.DataFrame, snap_columns_by_case: Mapping[str, Sequence[str]]
) -> pd.DataFrame:
    """Spread the ``snap`` array column of the Delta ``events`` table into named columns.

    The array layout differs per case (its generator set); ``snap_columns_by_case`` is the
    ``snap_columns`` list of each case (``runs.case_headers[case]["events"]``).
    """
    parts = []
    for case, rows in events.groupby("case", sort=False):
        names = list(snap_columns_by_case[str(case)])
        values = np.stack(list(rows["snap"])).astype(np.float64)
        if values.shape[1] != len(names):
            raise DataError(f"case {case!r}: {values.shape[1]} snapshot values, {len(names)} names")
        snaps = pd.DataFrame(values, columns=names, index=rows.index)
        parts.append(pd.concat([rows.drop(columns="snap"), snaps], axis=1))
    return pd.concat(parts).loc[events.index]


def complete_events(sets: Mapping[str, pd.DataFrame]) -> pd.Series:
    """Events (rows) without a missing value in any of the feature sets.

    Complete-case selection: every feature set is then compared on the same events.
    """
    masks = [~X.isna().any(axis=1) for X in sets.values()]
    keep = masks[0]
    for m in masks[1:]:
        keep = keep & m
    return keep
