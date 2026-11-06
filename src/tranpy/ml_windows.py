"""Accuracy versus assessment time on the traced fault events (the re-traced compact store).

At an *assessment instant* ``t`` the model sees the trajectory up to ``t`` only. Instants
(:class:`Instant`) are placed relative to one of three references:

``pre``
    the last sample before the fault (``t = FAULT_START_S - GRID_DT``);
``clear``
    the clearing instant plus ``offset_s`` (the usual "after clearing" window, which uses
    the event's own clearing time; see D9 in ``docs/validation.md``);
``inception``
    the fault inception plus ``offset_s`` (a fixed wall-clock time after the fault, causal
    with respect to the clearing).

The committed generator sets differ between operating hours, so the inputs are *system-level
series* (:data:`SERIES`), which exist in every hour: the largest and the spread of the
COI-relative rotor angles, their inertia-weighted standard deviation, the largest and the
COI speed deviation, total active and reactive output, the lowest and the mean generator-bus
voltage. Two representations are built per instant:

* ``window`` - the value of every series at ``t`` and, over the window from the reference
  (clearing or inception) to ``t``, its maximum, minimum, mean, last-minus-first and slope
  (least-squares, per second) on a :data:`GRID_DT` grid; plus the clearing time (known at
  clearing). ``window_line`` appends the faulted-line descriptors of
  :mod:`tranpy.line_features` (the ``line_pre`` feature set).
* ``sequence`` - the series resampled to the :data:`GRID_DT` grid over the same window,
  time-major (all series of step 0, then step 1, ...) for the Keras sequence networks of
  :mod:`tranpy_core.models.neural` (``n_steps`` steps).

An event whose stored trajectory ends before ``t`` (an unstable event whose simulation
stopped at the separation) cannot be assessed at ``t``: it is left out of that instant and
counted (``n_excluded``), as the window tables of :mod:`tranpy.early_prediction` do.
Evaluation reuses :func:`tranpy.ml_benchmark.evaluate_candidate` (protocols, bootstrap
intervals, nested tuning) with the Horvitz-Thompson inclusion weights of the traced
subsample (:func:`tranpy_core.imbalance.inclusion_weights`), so the metrics describe the
full event population.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from tranpy.ml_benchmark import Candidate, CandidateResult, evaluate_candidate
from tranpy_core.data import SignalDataset
from tranpy_core.errors import ConfigurationError, DataError

__all__ = [
    "DEFAULT_INSTANTS",
    "FAULT_START_S",
    "GRID_DT",
    "INPUTS",
    "SERIES",
    "TRACE_VARS",
    "CaseTraces",
    "Instant",
    "WindowTable",
    "curve_frame",
    "evaluate_window_file",
    "load_merged_traces",
    "sequence_table",
    "system_series",
    "trace_channels",
    "window_features",
    "window_tables",
    "write_window_tables",
]

FAULT_START_S = 1.0
"""Fault inception in the simulated events (events_v2: the fault is applied at 1.0 s)."""

GRID_DT = 0.01
"""Resampling step of the windows (the trace resolution around the fault)."""

TRACE_VARS = (
    "m:u in p.u.",
    "m:phiu in deg",
    "s:xspeed in p.u.",
    "s:P1 in MW",
    "s:Q1 in Mvar",
    "c:firot in deg",
)
"""Variables of the compact traces: bus voltage and angle, then speed, P, Q and rotor angle
per generator (``events_v2/v2lib.py::write_compact``)."""

SERIES = (
    "angle_max",
    "angle_spread",
    "angle_std",
    "dw_max",
    "dw_coi",
    "dw_std",
    "p_total",
    "q_total",
    "v_min",
    "v_mean",
)
"""System-level series derived from the traces (see the module docstring)."""

WINDOW_STATS = ("max", "min", "mean", "change", "slope")

Reference = Literal["pre", "clear", "inception"]
INPUTS = ("window", "window_line", "sequence")


@dataclass(frozen=True)
class Instant:
    """An assessment instant: ``reference`` plus ``offset_s``."""

    reference: Reference
    offset_s: float = 0.0

    def __post_init__(self) -> None:
        if self.reference not in ("pre", "clear", "inception"):
            raise ConfigurationError(f"unknown reference {self.reference!r}")
        if self.offset_s < 0 or (self.reference == "pre" and self.offset_s):
            raise ConfigurationError(f"invalid offset {self.offset_s} for {self.reference!r}")

    @property
    def name(self) -> str:
        if self.reference == "pre":
            return "pre"
        return f"{self.reference}+{self.offset_s:g}s"

    @classmethod
    def parse(cls, text: str) -> Instant:
        """``"pre"``, ``"clear+0.1s"`` or ``"inception+0.5s"``."""
        s = str(text).strip()
        if s == "pre":
            return cls("pre")
        ref, _, off = s.partition("+")
        if not off.endswith("s"):
            raise ConfigurationError(
                f"instant {text!r}: use 'pre', 'clear+<s>s' or 'inception+<s>s'"
            )
        return cls(ref, float(off[:-1]))  # type: ignore[arg-type]

    def time(self, t_clear: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Assessment time per event from the clearing instants ``t_clear``."""
        tc = np.asarray(t_clear, dtype=np.float64)
        if self.reference == "pre":
            return np.full(tc.shape, FAULT_START_S - GRID_DT)
        if self.reference == "clear":
            return tc + self.offset_s
        return np.full(tc.shape, FAULT_START_S + self.offset_s)

    def start(self, t_clear: npt.ArrayLike) -> npt.NDArray[np.float64] | None:
        """Window start per event (None for ``pre``: a single sample)."""
        tc = np.asarray(t_clear, dtype=np.float64)
        if self.reference == "pre":
            return None
        if self.reference == "clear":
            return tc
        return np.full(tc.shape, FAULT_START_S)

    @property
    def n_steps(self) -> int:
        """Grid steps of the window (1 for a zero-length window, 0 for ``pre``)."""
        if self.reference == "pre":
            return 0
        return round(self.offset_s / GRID_DT) + 1


DEFAULT_INSTANTS: tuple[Instant, ...] = (
    Instant("pre"),
    Instant("clear", 0.0),
    Instant("clear", 0.05),
    Instant("clear", 0.1),
    Instant("clear", 0.2),
    Instant("clear", 0.3),
    Instant("inception", 0.3),
    Instant("inception", 0.5),
    Instant("inception", 0.8),
)


# ----------------------------------------------------------------------------------- traces
def trace_channels(gen_map: Sequence[Sequence[str]]) -> list[str]:
    """Channel order of the compact traces: bus V and angle of the generator buses (unique,
    gen-map order), then speed, P, Q and rotor angle per generator."""
    bus = list(dict.fromkeys(f"{b}|{v}" for _, _, b in gen_map for v in TRACE_VARS[:2]))
    return bus + [f"{g}|{v}" for g, _, _ in gen_map for v in TRACE_VARS[2:]]


def load_merged_traces(
    path: str | Path, gen_map: Sequence[Sequence[str]], events: pd.DataFrame
) -> SignalDataset:
    """:class:`SignalDataset` of one merged trace file of a case (``merge_task.py``:
    columns ``sample_id``, ``time``, ``values``).

    ``events`` holds ``sample_id``, ``label`` and ``clearing_time_s`` of the case's events;
    traced events missing from it are dropped. Annotations: ``fault_start_s``,
    ``fault_clear_s``.
    """
    names = trace_channels(gen_map)
    t = pd.read_parquet(path, columns=["sample_id", "time", "values"])
    sid = t["sample_id"].to_numpy()
    order = np.lexsort((t["time"].to_numpy(), sid))
    t = t.iloc[order]
    sid = t["sample_id"].to_numpy()
    values = np.stack(list(t["values"].to_numpy())).astype(np.float64)
    if values.shape[1] != len(names):
        raise DataError(
            f"{path}: {values.shape[1]} channels, {len(names)} expected from the gen map"
        )
    ev = events.drop_duplicates("sample_id").set_index("sample_id")
    keep = np.isin(sid, ev.index.to_numpy())
    t, sid, values = t.iloc[keep], sid[keep], values[keep]
    if sid.size == 0:
        raise DataError(f"{path}: no traced event is in the events table")
    starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
    offsets = np.r_[starts, sid.size]
    ids = sid[starts]
    labels = ev.loc[ids, "label"].to_numpy().astype(int)
    tc = FAULT_START_S + ev.loc[ids, "clearing_time_s"].to_numpy(dtype=np.float64)
    return SignalDataset(
        time=t["time"].to_numpy(dtype=np.float64),
        values=values,
        offsets=offsets,
        channel_names=tuple(names),
        labels=labels,
        label_names=("stable", "unstable"),
        name=Path(path).stem,
        sample_ids=ids,
        annotations={"fault_start_s": np.full(ids.size, FAULT_START_S), "fault_clear_s": tc},
    )


@dataclass(frozen=True)
class CaseTraces:
    """Traces of one case with what the series need: generator names and COI weights."""

    case: str
    data: SignalDataset
    gens: tuple[str, ...]
    kinetic_mws: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if len(self.gens) != self.kinetic_mws.shape[0]:
            raise ConfigurationError("one COI weight per generator is needed")
        if np.any(self.kinetic_mws <= 0):
            raise ConfigurationError("COI weights must be positive")


def _channel_groups(ct: CaseTraces) -> dict[str, npt.NDArray[np.int64]]:
    idx = {n: i for i, n in enumerate(ct.data.channel_names)}
    bus_v = [i for n, i in idx.items() if n.endswith("|" + TRACE_VARS[0])]
    out = {"v": np.asarray(bus_v, dtype=np.int64)}
    for key, var in zip(("w", "p", "q", "delta"), TRACE_VARS[2:], strict=True):
        try:
            out[key] = np.asarray([idx[f"{g}|{var}"] for g in ct.gens], dtype=np.int64)
        except KeyError as exc:
            raise DataError(f"case {ct.case}: channel {exc.args[0]} missing") from exc
    return out


def system_series(
    ct: CaseTraces, i: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """``(time, series)`` of event ``i``: the :data:`SERIES` at every stored sample."""
    groups = _channel_groups(ct)
    sig = ct.data[i]
    v = sig.values
    w = ct.kinetic_mws
    wn = w / w.sum()
    delta = v[:, groups["delta"]]
    coi = delta @ wn
    dangle = delta - coi[:, None]
    dw = v[:, groups["w"]] - 1.0
    volt = v[:, groups["v"]]
    cols = [
        np.abs(dangle).max(axis=1),
        dangle.max(axis=1) - dangle.min(axis=1),
        np.sqrt((dangle**2) @ wn),
        np.abs(dw).max(axis=1),
        dw @ wn,
        np.sqrt(((dw - (dw @ wn)[:, None]) ** 2) @ wn),
        v[:, groups["p"]].sum(axis=1),
        v[:, groups["q"]].sum(axis=1),
        volt.min(axis=1),
        volt.mean(axis=1),
    ]
    return np.asarray(sig.time, dtype=np.float64), np.column_stack(cols)


def _grid(t0: float, t1: float) -> npt.NDArray[np.float64]:
    n = round((t1 - t0) / GRID_DT) + 1
    return np.asarray(t0 + GRID_DT * np.arange(n), dtype=np.float64)


def _resample(
    t: npt.NDArray[np.float64], s: npt.NDArray[np.float64], grid: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return np.column_stack([np.interp(grid, t, s[:, k]) for k in range(s.shape[1])])


def _usable(
    ct: CaseTraces, instant: Instant
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    t_at = instant.time(ct.data.annotations["fault_clear_s"])
    lengths = ct.data.lengths
    ends = ct.data.time[np.cumsum(lengths) - 1]
    starts = ct.data.time[np.cumsum(lengths) - lengths]
    return t_at, (ends + 1e-9 >= t_at) & (starts <= t_at)


def window_features(ct: CaseTraces, instant: Instant) -> pd.DataFrame:
    """The ``window`` representation of a case at ``instant`` (usable events only).

    Index: ``sample_id``. Columns: ``clearing_time_s``, ``<series>|at`` and, for windows of
    more than one step, ``<series>|<stat>`` for :data:`WINDOW_STATS`.
    """
    t_at, ok = _usable(ct, instant)
    tc = ct.data.annotations["fault_clear_s"]
    t0 = instant.start(tc)
    rows = []
    ids = []
    for i in np.flatnonzero(ok).tolist():
        t, s = system_series(ct, i)
        row: dict[str, float] = {"clearing_time_s": float(tc[i] - FAULT_START_S)}
        at = _resample(t, s, np.array([t_at[i]]))[0]
        row.update({f"{name}|at": float(x) for name, x in zip(SERIES, at, strict=True)})
        if t0 is not None and instant.n_steps > 1:
            grid = _grid(float(t0[i]), float(t_at[i]))
            win = _resample(t, s, grid)
            x = grid - grid[0]
            x = x - x.mean()
            slope = (x @ win) / (x @ x)
            for k, name in enumerate(SERIES):
                col = win[:, k]
                row[f"{name}|max"] = float(np.max(col))
                row[f"{name}|min"] = float(np.min(col))
                row[f"{name}|mean"] = float(col.mean())
                row[f"{name}|change"] = float(col[-1] - col[0])
                row[f"{name}|slope"] = float(slope[k])
        rows.append(row)
        ids.append(int(ct.data.sample_ids[i]))  # type: ignore[index]
    return pd.DataFrame(rows, index=pd.Index(ids, name="sample_id"))


def sequence_table(ct: CaseTraces, instant: Instant) -> pd.DataFrame:
    """The ``sequence`` representation: the series on the grid of the window, time-major
    (columns ``s<k>|<series>``). Not defined for ``pre`` (no window)."""
    if instant.reference == "pre":
        raise ConfigurationError("no sequence at the pre-fault instant")
    t_at, ok = _usable(ct, instant)
    t0 = instant.start(ct.data.annotations["fault_clear_s"])
    assert t0 is not None
    n = instant.n_steps
    columns = [f"s{k}|{name}" for k in range(n) for name in SERIES]
    rows = []
    ids = []
    for i in np.flatnonzero(ok).tolist():
        t, s = system_series(ct, i)
        grid = _grid(float(t0[i]), float(t_at[i]))
        if grid.size != n:
            raise DataError(f"window of {grid.size} steps, {n} expected")
        rows.append(_resample(t, s, grid).reshape(-1))
        ids.append(int(ct.data.sample_ids[i]))  # type: ignore[index]
    return pd.DataFrame(
        np.asarray(rows).reshape(len(rows), -1),
        columns=columns,
        index=pd.Index(ids, name="sample_id"),
    )


# ----------------------------------------------------------------------------------- tables
@dataclass
class WindowTable:
    """Representations of one instant over all cases, on the same events.

    ``strata`` has the keys (``case``, ``sample_id``), ``label``, the grouping columns of the
    events and ``weight`` (inclusion weights); ``inputs`` maps an input name of
    :data:`INPUTS` to its feature frame (rows in the order of ``strata``); ``excluded`` lists
    the traced events left out because their trajectory ends before the instant.
    """

    instant: Instant
    strata: pd.DataFrame
    inputs: dict[str, pd.DataFrame]
    excluded: pd.DataFrame

    @property
    def n_excluded(self) -> int:
        return len(self.excluded)


def window_tables(
    cases: Iterable[CaseTraces],
    strata: pd.DataFrame,
    *,
    instants: Sequence[Instant] = DEFAULT_INSTANTS,
    line_features: pd.DataFrame | None = None,
    weights: Mapping[tuple[str, int], float] | None = None,
    sequences: bool = True,
) -> dict[str, WindowTable]:
    """Window tables of every instant, keyed by :attr:`Instant.name`.

    ``strata`` (``case``, ``sample_id``, ``label``, grouping columns) describes the traced
    events; ``line_features`` (same keys) adds the ``window_line`` input; ``weights`` is the
    inclusion weight per ``(case, sample_id)`` (1 when None).
    """
    cases = list(cases)
    keys = strata.set_index(["case", "sample_id"])
    if keys.index.duplicated().any():
        raise DataError("strata has duplicate (case, sample_id) keys")
    line = None if line_features is None else line_features.set_index(["case", "sample_id"])
    out: dict[str, WindowTable] = {}
    for instant in instants:
        feats, seqs, excluded = [], [], []
        for ct in cases:
            f = window_features(ct, instant)
            f.index = pd.MultiIndex.from_arrays(
                [[ct.case] * len(f), f.index], names=["case", "sample_id"]
            )
            feats.append(f)
            if sequences and instant.reference != "pre":
                s = sequence_table(ct, instant)
                s.index = f.index
                seqs.append(s)
            ids = np.asarray(ct.data.sample_ids)
            _, ok = _usable(ct, instant)
            excluded.append(pd.DataFrame({"case": ct.case, "sample_id": ids[~ok]}))
        window = pd.concat(feats)
        window = window[window.index.isin(keys.index)]
        st = keys.loc[window.index].reset_index()
        st["weight"] = (
            1.0
            if weights is None
            else [
                float(weights[(c, int(i))])
                for c, i in zip(st["case"], st["sample_id"], strict=True)
            ]
        )
        inputs = {"window": window.reset_index(drop=True)}
        if line is not None:
            inputs["window_line"] = pd.concat(
                [window, line.loc[window.index].drop(columns=["clearing_time_s"], errors="ignore")],
                axis=1,
            ).reset_index(drop=True)
        if seqs:
            inputs["sequence"] = pd.concat(seqs).loc[window.index].reset_index(drop=True)
        for name, X in inputs.items():
            if X.isna().any().any():
                raise DataError(f"{instant.name} {name}: missing values")
        ex = pd.concat(excluded, ignore_index=True)
        ex = ex.merge(strata[["case", "sample_id", "label"]], on=["case", "sample_id"], how="inner")
        out[instant.name] = WindowTable(instant, st, inputs, ex)
    return out


def write_window_tables(tables: Mapping[str, WindowTable], folder: str | Path) -> dict[str, Any]:
    """Write ``<instant>__<input>.parquet``, ``<instant>__strata.parquet`` and
    ``<instant>__excluded.parquet`` to ``folder``; returns a manifest."""
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {}
    for name, tab in tables.items():
        tab.strata.to_parquet(out / f"{name}__strata.parquet", index=False)
        tab.excluded.to_parquet(out / f"{name}__excluded.parquet", index=False)
        for inp, X in tab.inputs.items():
            X.to_parquet(out / f"{name}__{inp}.parquet", index=False)
        manifest[name] = {
            "reference": tab.instant.reference,
            "offset_s": tab.instant.offset_s,
            "n_steps": tab.instant.n_steps,
            "events": len(tab.strata),
            "unstable": int(tab.strata["label"].sum()),
            "n_excluded": tab.n_excluded,
            "excluded_unstable": int(tab.excluded["label"].sum()) if len(tab.excluded) else 0,
            "inputs": {k: int(v.shape[1]) for k, v in tab.inputs.items()},
        }
    return manifest


# ------------------------------------------------------------------------------- evaluation
def evaluate_window_file(
    folder: str | Path,
    instant: str,
    input_name: str,
    candidate: Candidate,
    seed: int,
    report_by: Sequence[str],
    *,
    n_boot: int | None = None,
) -> tuple[str, str, Candidate, CandidateResult | None, str | None, float]:
    """:func:`evaluate_candidate` on ``<folder>/<instant>__<input>.parquet`` with the
    inclusion weights of ``<instant>__strata.parquet``, for worker processes.

    Returns ``(instant, input, candidate, result or None, error or None, seconds)``.
    """
    for var in ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        os.environ.setdefault(var, "1")
    base = Path(folder)
    t0 = time.perf_counter()
    try:
        X = pd.read_parquet(base / f"{instant}__{input_name}.parquet")
        strata = pd.read_parquet(base / f"{instant}__strata.parquet")
        kwargs: dict[str, Any] = {} if n_boot is None else {"n_boot": n_boot}
        result = evaluate_candidate(
            candidate,
            X,
            strata["label"].to_numpy(),
            strata,
            feature_set=f"{instant}/{input_name}",
            report_by=report_by,
            seed=seed,
            sample_weight=strata["weight"].to_numpy(dtype=np.float64),
            **kwargs,
        )
    except Exception as exc:  # recorded per candidate
        return (
            instant,
            input_name,
            candidate,
            None,
            f"{type(exc).__name__}: {exc}",
            time.perf_counter() - t0,
        )
    return instant, input_name, candidate, result, None, time.perf_counter() - t0


def curve_frame(results: pd.DataFrame, metric: str = "average_precision") -> pd.DataFrame:
    """Rows of ``results`` (columns ``instant``, ``model``, ``<metric>``) as the frame
    :func:`tranpy.early_prediction.find_knee_point` reads: ``window_s`` (the offset from the
    reference), ``model`` and ``<metric>_mean``, one frame row per result row."""
    inst = [Instant.parse(s) for s in results["instant"]]
    return pd.DataFrame(
        {
            "window_s": [i.offset_s if i.reference != "pre" else -GRID_DT for i in inst],
            "reference": [i.reference for i in inst],
            "model": results["model"].to_numpy(),
            f"{metric}_mean": results[metric].to_numpy(dtype=np.float64),
        }
    )
