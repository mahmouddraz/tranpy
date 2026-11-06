"""Physical descriptors of the faulted line, from the pre-fault state of the case.

The scenario inputs of an event say *which* line is faulted (``fault_line``, an index into
the fault-candidate lines of the engine) and at which end (``fault_position``). The model
cannot know what that index means in the grid; this module describes the faulted line and
the faulted bus physically: pre-fault power through the line, the angle across it, its
impedance and voltage level, how strongly the faulted bus is meshed, how much inertia and
generation sits within a few hops, and how far the nearest machine is. Everything comes
from the ANDES case workbook and its solved power flow, nothing from the simulated
outcome.

The engine convention (:mod:`tranpy.simulation.andes`) is reproduced exactly: the fault
candidates are the ``Line`` rows with ``trans == 0`` in sheet order; the fault is applied
at ``bus2`` when ``fault_position >= 0.5`` and at ``bus1`` otherwise; the line is tripped
at clearing.

Line powers use the ANDES ``Line`` pi model (series ``r + jx``, shunts ``g1 + g/2``,
``b1 + b/2`` at the from end, ``g2 + g/2``, ``b2 + b/2`` at the to end, off-nominal
``tap`` and phase shift ``phi`` on the from side) and the sign of ANDES: positive P is
power flowing from the bus into the line. Graph operations use ``scipy.sparse.csgraph``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra, shortest_path

from tranpy_core.errors import DataError

__all__ = [
    "LINE_PREFIX",
    "CaseGrid",
    "faulted_bus",
    "line_features",
    "line_power",
    "read_case_grid",
]

LINE_PREFIX = "line|"
HOPS = (1, 2, 3)
LOADING_QUANTILE = 0.95
_MACHINE_SHEETS = ("GENROU", "GENCLS")


def line_power(
    v1: np.ndarray,
    a1: np.ndarray,
    v2: np.ndarray,
    a2: np.ndarray,
    lines: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Complex powers (pu) at both ends of the ANDES pi-model branches ``lines``.

    Args:
        v1, a1, v2, a2: from/to voltage magnitude (pu) and angle (rad) per branch.
        lines: ``r``, ``x``, ``b``, ``g``, ``b1``, ``g1``, ``b2``, ``g2``, ``tap``, ``phi``.

    Returns:
        ``(P1, Q1, P2, Q2)``, positive from the bus into the branch (ANDES ``a1.e`` etc.).
    """
    col = lines.to_dict("series")
    f = {k: col[k].to_numpy(np.float64) for k in ("r", "x", "b", "g", "b1", "g1", "b2", "g2")}
    tap = col["tap"].to_numpy(np.float64) if "tap" in col else np.ones(len(lines))
    phi = col["phi"].to_numpy(np.float64) if "phi" in col else np.zeros(len(lines))
    y = 1.0 / (f["r"] + 1j * f["x"])
    yh = (f["g1"] + f["g"] / 2) + 1j * (f["b1"] + f["b"] / 2)
    yk = (f["g2"] + f["g"] / 2) + 1j * (f["b2"] + f["b"] / 2)
    t = tap * np.exp(1j * phi)
    u1 = v1 * np.exp(1j * a1)
    u2 = v2 * np.exp(1j * a2)
    i1 = (yh + y) * u1 / np.abs(t) ** 2 - y * u2 / np.conj(t)
    i2 = (yk + y) * u2 - y * u1 / t
    s1 = u1 * np.conj(i1)
    s2 = u2 * np.conj(i2)
    return s1.real, s1.imag, s2.real, s2.imag


def faulted_bus(fault_position: np.ndarray) -> np.ndarray:
    """``True`` where the fault is at ``bus2`` (engine rule: ``fault_position >= 0.5``)."""
    return np.asarray(fault_position, dtype=np.float64) >= 0.5


@dataclass
class CaseGrid:
    """One case: its buses, branches, machines, solved voltages and derived tables.

    Built once per case with :meth:`from_sheets` (frames in memory) or
    :func:`read_case_grid` (files). ``bus_table`` and ``line_table`` hold the per-bus and
    per-fault-candidate descriptors; :func:`line_features` only looks them up.

    Attributes:
        buses: ``idx``, ``Vn`` (kV), ``v`` (pu), ``a`` (rad), in sheet order.
        branches: every in-service ``Line`` row with ``i1``/``i2`` (bus positions) and the
            pre-fault ``p1``, ``q1``, ``p2``, ``q2`` (pu).
        fault_lines: positions in ``branches`` of the fault candidates, engine order.
        machines: ``bus`` (position), ``Sn`` (MVA), ``M`` (2H, s), ``P`` (pu).
        adjacency: symmetric sparse bus graph (unit weights), in-service branches.
        mva: system base (MVA).
    """

    buses: pd.DataFrame
    branches: pd.DataFrame
    fault_lines: np.ndarray
    machines: pd.DataFrame
    adjacency: csr_matrix
    mva: float = 100.0

    @classmethod
    def from_sheets(
        cls,
        sheets: Mapping[str, pd.DataFrame],
        v: np.ndarray,
        a: np.ndarray,
        *,
        machine_p: Mapping[Any, float] | None = None,
        mva: float = 100.0,
    ) -> CaseGrid:
        """Build from the ANDES workbook sheets and the solved bus voltages.

        Args:
            sheets: ``Bus``, ``Line``, ``GENROU``/``GENCLS`` and ``PV``/``Slack`` frames.
            v, a: solved voltage magnitude (pu) and angle (rad), in ``Bus`` sheet order.
            machine_p: active power (pu on ``mva``) per machine ``idx``; default is the
                ``p0`` of the generator the machine replaces (``gen`` column).
            mva: system base.
        """
        bus = sheets["Bus"].reset_index(drop=True)
        v = np.asarray(v, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        if len(v) != len(bus) or len(a) != len(bus):
            raise DataError(f"{len(bus)} buses but {len(v)} voltages and {len(a)} angles")
        buses = pd.DataFrame({"idx": bus["idx"], "Vn": bus["Vn"].astype(float), "v": v, "a": a})
        pos = pd.Series(np.arange(len(buses)), index=buses["idx"])
        line = sheets["Line"].reset_index(drop=True)
        if "u" in line:
            line = line[line["u"].astype(float) != 0].reset_index(drop=True)
        unknown = sorted((set(line["bus1"]) | set(line["bus2"])) - set(pos.index))
        if unknown:
            raise DataError(f"branches on unknown buses: {unknown[:5]}")
        branches = line.copy()
        for c in ("b", "g", "b1", "g1", "b2", "g2"):
            if c not in branches:
                branches[c] = 0.0
        if "tap" not in branches:
            branches["tap"] = 1.0
        if "phi" not in branches:
            branches["phi"] = 0.0
        branches["i1"] = pos.loc[line["bus1"]].to_numpy()
        branches["i2"] = pos.loc[line["bus2"]].to_numpy()
        i1, i2 = branches["i1"].to_numpy(), branches["i2"].to_numpy()
        p1, q1, p2, q2 = line_power(v[i1], a[i1], v[i2], a[i2], branches)
        branches["p1"], branches["q1"], branches["p2"], branches["q2"] = p1, q1, p2, q2
        branches["trans"] = branches["trans"].astype(int)
        branches["delta_deg"] = np.degrees(np.abs(a[i1] - a[i2]))
        fault_lines = np.flatnonzero(branches["trans"].to_numpy() == 0)
        machines = _machines(sheets, pos, machine_p)
        n = len(buses)
        adjacency = coo_matrix((np.ones(len(i1)), (i1, i2)), shape=(n, n)).tocsr()
        adjacency = ((adjacency + adjacency.T) > 0).astype(np.float64)
        return cls(buses, branches, fault_lines, machines, csr_matrix(adjacency), mva)

    @property
    def n_fault_lines(self) -> int:
        return len(self.fault_lines)

    def _bus_table(self) -> pd.DataFrame:
        """Per bus: voltage level, degree, short-circuit proxy, local inertia and
        generation within 1..3 hops, hop and reactance distance to the nearest machine,
        and the top-two angle differences of the incident fault-candidate lines."""
        n = len(self.buses)
        br = self.branches
        i1, i2 = br["i1"].to_numpy(), br["i2"].to_numpy()
        ends = np.concatenate([i1, i2])
        is_line = np.concatenate([br["trans"].to_numpy() == 0] * 2)
        degree = np.bincount(ends, minlength=n).astype(float)
        degree_lines = np.bincount(ends[is_line], minlength=n).astype(float)
        inv_x = np.concatenate([1.0 / np.abs(br["x"].to_numpy(np.float64))] * 2)
        sc = np.bincount(ends, weights=inv_x, minlength=n)
        hops = shortest_path(self.adjacency, directed=False, unweighted=True)
        xgraph = _symmetric_min(
            br["i1"].to_numpy(), br["i2"].to_numpy(), np.abs(br["x"].to_numpy(np.float64)), n
        )
        mb = self.machines["bus"].to_numpy()
        kinetic = np.bincount(
            mb,
            weights=(self.machines["M"] / 2.0 * self.machines["Sn"]).to_numpy(np.float64),
            minlength=n,
        )
        p_mw = np.bincount(
            mb, weights=self.machines["P"].to_numpy(np.float64) * self.mva, minlength=n
        )
        data: dict[str, np.ndarray] = {
            "bus_vn_kv": self.buses["Vn"].to_numpy(np.float64),
            "bus_degree": degree,
            "bus_degree_lines": degree_lines,
            "bus_sc_strength": sc,
        }
        for k in HOPS:
            within = hops <= k
            data[f"kinetic_mws_{k}hop"] = within @ kinetic
            data[f"gen_mw_{k}hop"] = within @ p_mw
        if len(mb):
            data["machine_hops"] = hops[:, mb].min(axis=1)
            xd = dijkstra(xgraph, directed=False, indices=mb)
            data["machine_x_dist"] = xd.min(axis=0)
        else:
            data["machine_hops"] = np.full(n, np.inf)
            data["machine_x_dist"] = np.full(n, np.inf)
        data.update(_top_two_angles(br, n))
        return pd.DataFrame(data)

    def _line_table(self) -> pd.DataFrame:
        """Per fault candidate (engine order): the line descriptors."""
        br = self.branches.iloc[self.fault_lines].reset_index(drop=True)
        v = self.buses["v"].to_numpy()
        i1, i2 = br["i1"].to_numpy(), br["i2"].to_numpy()
        p1 = br["p1"].to_numpy() * self.mva
        q1 = br["q1"].to_numpy() * self.mva
        p2 = br["p2"].to_numpy() * self.mva
        q2 = br["q2"].to_numpy() * self.mva
        s = np.maximum(np.hypot(p1, q1), np.hypot(p2, q2))
        vn = br["Vn1"].to_numpy(np.float64) if "Vn1" in br else self.buses["Vn"].to_numpy()[i1]
        ref = (
            pd.Series(s)
            .groupby(vn)
            .transform(lambda x: x.quantile(LOADING_QUANTILE))
            .to_numpy(np.float64)
        )
        pair = pd.Series(list(zip(np.minimum(i1, i2), np.maximum(i1, i2), strict=True)))
        x = br["x"].to_numpy(np.float64)
        r = br["r"].to_numpy(np.float64)
        return pd.DataFrame(
            {
                "i1": i1,
                "i2": i2,
                "p1_mw": p1,
                "q1_mvar": q1,
                "p2_mw": p2,
                "q2_mvar": q2,
                "s_mva": s,
                "delta_deg": br["delta_deg"].to_numpy(np.float64),
                "v1_pu": v[i1],
                "v2_pu": v[i2],
                "x": x,
                "r_x": r / np.where(x == 0, np.nan, x),
                "y_series": 1.0 / np.hypot(r, x),
                "vn_kv": vn,
                "loading": s / np.where(ref > 0, ref, np.nan),
                "n_circuits": pair.map(pair.value_counts()).to_numpy(np.float64),
            }
        )


def _symmetric_min(i: np.ndarray, j: np.ndarray, w: np.ndarray, n: int) -> csr_matrix:
    """Undirected weighted bus graph; parallel branches keep their smallest weight."""
    edges = pd.DataFrame(
        {"i": np.concatenate([i, j]), "j": np.concatenate([j, i]), "w": np.tile(w, 2)}
    )
    least = edges[edges["i"] != edges["j"]].groupby(["i", "j"], as_index=False)["w"].min()
    return coo_matrix(
        (least["w"].to_numpy(), (least["i"].to_numpy(), least["j"].to_numpy())), shape=(n, n)
    ).tocsr()


def _top_two_angles(br: pd.DataFrame, n: int) -> dict[str, np.ndarray]:
    """Per bus: the largest and second-largest |angle difference| over the incident
    fault-candidate lines, and the branch row giving the largest (-1 if none)."""
    lines = br[br["trans"] == 0]
    rows = lines.index.to_numpy()
    delta = lines["delta_deg"].to_numpy(np.float64)
    table = pd.DataFrame(
        {
            "bus": np.concatenate([lines["i1"].to_numpy(), lines["i2"].to_numpy()]),
            "branch": np.concatenate([rows, rows]),
            "delta": np.concatenate([delta, delta]),
        }
    ).sort_values(["bus", "delta"], ascending=[True, False])
    rank = table.groupby("bus").cumcount()
    first = table[rank == 0].set_index("bus")
    second = table[rank == 1].set_index("bus")
    top1 = np.zeros(n)
    top1_branch = np.full(n, -1)
    top2 = np.zeros(n)
    top1[first.index] = first["delta"]
    top1_branch[first.index] = first["branch"]
    top2[second.index] = second["delta"]
    return {"top1_delta_deg": top1, "top1_branch": top1_branch, "top2_delta_deg": top2}


def _machines(
    sheets: Mapping[str, pd.DataFrame],
    pos: pd.Series,
    machine_p: Mapping[Any, float] | None,
) -> pd.DataFrame:
    parts = [sheets[s] for s in _MACHINE_SHEETS if s in sheets and len(sheets[s])]
    if not parts:
        return pd.DataFrame({"bus": np.zeros(0, int), "Sn": [], "M": [], "P": []})
    m = pd.concat(parts, ignore_index=True)
    if "u" in m:
        m = m[m["u"].astype(float) != 0].reset_index(drop=True)
    p0: dict[Any, float] = {}
    for s in ("PV", "Slack"):
        if s in sheets and len(sheets[s]):
            p0.update(zip(sheets[s]["idx"], sheets[s]["p0"].astype(float), strict=True))
    if machine_p is not None:
        p = m["idx"].map(dict(machine_p))
    else:
        p = m["gen"].map(p0) if "gen" in m else pd.Series(np.nan, index=m.index)
    missing = m.loc[p.isna(), "idx"].tolist()
    if missing:
        raise DataError(f"machines without an active power: {missing[:5]}")
    return pd.DataFrame(
        {
            "bus": pos.loc[m["bus"]].to_numpy(),
            "Sn": m["Sn"].astype(float).to_numpy(),
            "M": m["M"].astype(float).to_numpy(),
            "P": p.astype(float).to_numpy(),
        }
    )


def line_features(events: pd.DataFrame, grids: Mapping[str, CaseGrid]) -> pd.DataFrame:
    """Descriptors of the faulted line and bus for every event, columns ``line|...``.

    Args:
        events: ``case``, ``fault_line`` (engine index), ``fault_position``.
        grids: :class:`CaseGrid` per case.

    Returns:
        Frame indexed like ``events``. Line columns: ``p_mw``, ``q_mvar`` (at the faulted
        end, from the bus into the line), ``s_mva`` (largest apparent power of the two
        ends), ``delta_deg`` (|angle| across the line), ``x``, ``r_x``, ``y_series``
        (|1/(r+jx)|), ``vn_kv``, ``loading`` (``s_mva`` over the 95th percentile of the
        lines at the same voltage level of the case), ``n_circuits`` (circuits between the
        two ends, the line itself included), ``v_bus_pu``, ``other_delta_max_deg`` (largest
        pre-fault angle difference over the other lines at the two ends). Bus columns (the
        faulted bus): ``bus_vn_kv``, ``bus_degree`` (branches), ``bus_degree_lines``,
        ``bus_sc_strength`` (sum of 1/|x| of the branches at the bus), ``kinetic_mws_khop``
        (sum of H * Sn of the machines within k hops, k = 1, 2, 3), ``gen_mw_khop``,
        ``machine_hops`` and ``machine_x_dist`` (hop count and reactance path to the
        nearest machine). ``fault_line_cat`` is ``fault_line`` as a pandas categorical.
    """
    for c in ("case", "fault_line", "fault_position"):
        if c not in events:
            raise DataError(f"events need a {c!r} column")
    blocks = []
    for case, rows in events.groupby("case", sort=True):
        if case not in grids:
            raise DataError(f"no case grid for {case!r}")
        blocks.append(_case_features(rows, grids[str(case)]))
    out = pd.concat(blocks).reindex(events.index)
    out.columns = [f"{LINE_PREFIX}{c}" for c in out.columns]
    cat = pd.Categorical(events["fault_line"].astype(int))
    out[f"{LINE_PREFIX}fault_line_cat"] = cat
    return out


def _case_features(rows: pd.DataFrame, grid: CaseGrid) -> pd.DataFrame:
    fl = rows["fault_line"].to_numpy()
    if np.any(np.isnan(fl.astype(float))) or np.any(fl != np.floor(fl)):
        raise DataError("fault_line must be integer")
    fl = fl.astype(int)
    if fl.min() < 0 or fl.max() >= grid.n_fault_lines:
        raise DataError(f"fault_line out of range 0..{grid.n_fault_lines - 1}")
    lt = grid._line_table()
    bt = grid._bus_table()
    at2 = faulted_bus(rows["fault_position"].to_numpy())
    L = lt.iloc[fl].reset_index(drop=True)
    bus = np.where(at2, L["i2"], L["i1"])
    far = np.where(at2, L["i1"], L["i2"])
    B = bt.iloc[bus].reset_index(drop=True)
    data = {
        "p_mw": np.where(at2, L["p2_mw"], L["p1_mw"]),
        "q_mvar": np.where(at2, L["q2_mvar"], L["q1_mvar"]),
        "s_mva": L["s_mva"].to_numpy(),
        "delta_deg": L["delta_deg"].to_numpy(),
        "x": L["x"].to_numpy(),
        "r_x": L["r_x"].to_numpy(),
        "y_series": L["y_series"].to_numpy(),
        "vn_kv": L["vn_kv"].to_numpy(),
        "loading": L["loading"].to_numpy(),
        "n_circuits": L["n_circuits"].to_numpy(),
        "v_bus_pu": np.where(at2, L["v2_pu"], L["v1_pu"]),
        "other_delta_max_deg": np.maximum(
            _other_max(bt, bus, grid.fault_lines[fl]), _other_max(bt, far, grid.fault_lines[fl])
        ),
    }
    for c in bt.columns:
        if not c.startswith("top"):
            data[c] = B[c].to_numpy()
    return pd.DataFrame(data, index=rows.index)


def _other_max(bt: pd.DataFrame, bus: np.ndarray, branch: np.ndarray) -> np.ndarray:
    """Largest incident-line angle difference at ``bus`` excluding ``branch``."""
    own = bt["top1_branch"].to_numpy()[bus] == branch
    top1 = bt["top1_delta_deg"].to_numpy(np.float64)[bus]
    top2 = bt["top2_delta_deg"].to_numpy(np.float64)[bus]
    return np.asarray(np.where(own, top2, top1), dtype=np.float64)


def read_case_grid(prep_dir: str | Path, case: str) -> CaseGrid:
    """:class:`CaseGrid` of ``CASE.xlsx`` in ``prep_dir``.

    The solved voltages come from ``CASE.op.pkl`` together with ``CASE.buses.csv``
    (the pypowsybl solution keyed by busbar, written by the case builder) when both are
    present and cover every bus; otherwise the ANDES power flow of the workbook is solved.
    """
    folder = Path(prep_dir)
    xlsx = folder / f"{case}.xlsx"
    if not xlsx.exists():
        raise DataError(f"no case workbook {xlsx}")
    sheets = pd.read_excel(xlsx, sheet_name=None)
    op, buses = folder / f"{case}.op.pkl", folder / f"{case}.buses.csv"
    if op.exists() and buses.exists():
        point = pd.read_pickle(op)
        names = pd.read_csv(buses).set_index("andes_bus")["bus"]
        key = sheets["Bus"]["idx"].map(names)
        v = key.map(point["V"])
        a = key.map(point["A"])
        if not v.isna().any() and not a.isna().any():
            return CaseGrid.from_sheets(sheets, v.to_numpy(), np.radians(a.to_numpy()))
    return _solved_grid(xlsx, sheets)


def _solved_grid(xlsx: Path, sheets: Mapping[str, pd.DataFrame]) -> CaseGrid:
    from tranpy_core._optional import require

    andes = require("andes", "andes")
    ss = andes.load(str(xlsx), setup=True, no_output=True, default_config=True)
    if not ss.PFlow.run():
        raise DataError(f"ANDES power flow of {xlsx.name} did not converge")
    if list(ss.Bus.idx.v) != sheets["Bus"]["idx"].tolist():
        raise DataError(f"{xlsx.name}: ANDES bus order differs from the workbook")
    gen_p: dict[Any, float] = dict(zip(ss.PV.idx.v, ss.PV.p0.v, strict=True))
    gen_p.update(zip(ss.Slack.idx.v, ss.Slack.p.v, strict=True))
    machine_p: dict[Any, float] = {}
    for name in _MACHINE_SHEETS:
        model = getattr(ss, name)
        machine_p.update(
            (idx, float(gen_p[g])) for idx, g in zip(model.idx.v, model.gen.v, strict=True)
        )
    return CaseGrid.from_sheets(
        sheets,
        np.asarray(ss.Bus.v.v),
        np.asarray(ss.Bus.a.v),
        machine_p=machine_p,
        mva=float(ss.config.mva),
    )
