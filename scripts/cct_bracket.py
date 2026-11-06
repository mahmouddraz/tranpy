"""Critical clearing time of every fault-candidate line of the v4.1 cases, by bisection.

python scripts/cct_bracket.py --out ~/data/exp/e4_sim [--cases high_load_t10,low_load_t10,...]
       [--jobs 16] [--lo 0.05] [--hi 1.0] [--width 0.01] [--positions 0,1] [--lines 0,1,2]

For each case (ANDES case CASE.xlsx of the Databricks prep, the same frozen v2lib.Engine as the
v4.1 events: three-phase fault of 0.5 ohm at the line terminal at t = 1 s, cleared by opening
the line, step 10 ms, step/4 retry on a dynamic failure) and each fault-candidate line (a line
whose opening cuts off no generator, gen.py's rule: 996 of 1,015) and terminal (fault_position
0 = from bus, 1 = to bus; the v4.1 events draw either), the clearing time is bisected between
--lo and --hi at the base operating point (operating_point 1, no load noise, as the v4.1 grid
sweeps) until the bracket is at most --width wide. Stable at --hi ends early (status
``above``, cct_lo = hi), unstable at --lo too (status ``below``, cct_hi = lo). The label is the
v4.1 rule: any generator out of step (rotor angle more than 180 degrees from the inertia-
weighted centre of inertia, latched); the simulated window is 1 + hi + 5 s (v4.1: 6.6 s), and
the out-of-step onset time of the unstable run at cct_hi is stored, so a label on the 6.6 s
window can be recovered.

Output (appended as cases complete, resumable per (case, line, position)):
  OUT/cct/cct_CASE.parquet     one row per (case, fault_line, fault_position): cct_lo, cct_hi,
                               status (bracketed/above/below/failed), n_sims, t_oos_s, sims
                               (json list of [clearing_time_s, label]), vn_kv, n_circuits,
                               bus1, bus2, line_idx
  OUT/cct_brackets.parquet     all finished cases (also .csv)
  OUT/progress.json            progress of the running case and the finished cases
  OUT/cct_consistency.json     v4.1 label vs bracket agreement per finished case
"""
# ruff: noqa: E501, B023  # long literals; closures are used inside the loop iteration that defines them

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HOME = Path.home()
DEFAULT_INPUTS = HOME / "data/volume/inputs/v41dyn"
DEFAULT_OUT_ROOT = HOME / "data/volume/outputs/v41dyn/738439695169033"
HOURS = [
    "high_wind",
    "high_solar",
    "low_load",
    "high_load",
    "median_load",
    "winter_peak",
    "summer_peak",
    "very_light_night",
    "baltic_import_low_load",
]
CASES = [f"{h}_t{t}" for h in HOURS for t in ("10", "20")]
PRIORITY = ["high_load_t10", "low_load_t10"]
PIN_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
SIM_END_V41 = 6.6
RESULT_COLS = [
    "fault_line",
    "fault_position",
    "cct_lo",
    "cct_hi",
    "status",
    "n_sims",
    "t_oos_s",
    "n_retried",
    "failure",
    "sims",
    "sec",
]

_E = None
_BASE = None
_CFG: dict = {}


def _init(xlsx: str, ev2: str, cfg: dict) -> None:
    """Worker: one engine per process (the case file is reloaded by ANDES per simulation)."""
    global _E, _BASE, _CFG
    for k in PIN_VARS:
        os.environ[k] = "1"
    sys.path.insert(0, ev2)
    from v2lib import Engine

    from tranpy.simulation.engine import FaultScenario

    _CFG = cfg
    _E = Engine(
        system_name="germany",
        case=xlsx,
        simulation_time_s=1.0 + cfg["hi"] + 5.0,
        step_s=0.01,
        record_buses="generators",
        name_width=4,
        n_jobs=1,
    )
    _BASE = FaultScenario(
        scenario_id=0,
        fault_line=0,
        fault_start_s=1.0,
        clearing_duration_s=0.1,
        fault_resistance_ohm=0.0,
        fault_reactance_ohm=0.5,
        load_index=0,
        load_change_pct=0.0,
        trip_line=None,
        operating_point=1.0,
        load_factors=None,
        fault_position=0.0,
    )


def _sim(line: int, pos: float, clr: float) -> tuple[int | None, float, float, str | None, bool]:
    """(label or None if failed, out-of-step onset time or nan, end time, failure, retried)."""
    sc = replace(_BASE, fault_line=line, fault_position=pos, clearing_duration_s=clr)
    r = _E._simulate_one(sc)
    if not r.completed:
        return None, float("nan"), float(r.time[-1]), r.failure, bool(getattr(r, "retried", False))
    oos = [j for j, c in enumerate(r.channel_names) if c.endswith("s:outofstep")]
    flag = (r.values[:, oos] > 0.5).any(axis=1)
    t_oos = float(r.time[int(np.argmax(flag))]) if flag.any() else float("nan")
    return int(flag.any()), t_oos, float(r.time[-1]), None, bool(getattr(r, "retried", False))


def bracket(task: tuple[int, float]) -> dict:
    line, pos = task
    lo, hi, width = _CFG["lo"], _CFG["hi"], _CFG["width"]
    sims: list[list] = []
    t0 = time.time()
    failure = None
    t_oos_hi = float("nan")
    retried = 0

    def run(clr: float):
        nonlocal failure, retried
        lab, t_o, t_end, fail, rt = _sim(line, pos, clr)
        retried += int(rt)
        sims.append([round(clr, 6), lab, round(t_o, 4) if lab else None, round(t_end, 4)])
        if lab is None:
            failure = fail
        return lab, t_o

    lab, t_o = run(lo)
    if lab is None:
        status, cct_lo, cct_hi = "failed", float("nan"), float("nan")
    elif lab == 1:
        status, cct_lo, cct_hi, t_oos_hi = "below", float("nan"), lo, t_o
    else:
        lab, t_o = run(hi)
        if lab is None:
            status, cct_lo, cct_hi = "failed", lo, float("nan")
        elif lab == 0:
            status, cct_lo, cct_hi = "above", hi, float("nan")
        else:
            cct_lo, cct_hi, t_oos_hi, status = lo, hi, t_o, "bracketed"
            while cct_hi - cct_lo > width + 1e-9:
                mid = round(0.5 * (cct_lo + cct_hi), 4)
                lab, t_o = run(mid)
                if lab is None:
                    status = "failed"
                    break
                if lab == 1:
                    cct_hi, t_oos_hi = mid, t_o
                else:
                    cct_lo = mid
    return {
        "fault_line": int(line),
        "fault_position": float(pos),
        "cct_lo": cct_lo,
        "cct_hi": cct_hi,
        "status": status,
        "n_sims": len(sims),
        "t_oos_s": t_oos_hi,
        "n_retried": retried,
        "failure": failure,
        "sims": json.dumps(sims),
        "sec": round(time.time() - t0, 1),
    }


def line_table(xlsx: Path) -> pd.DataFrame:
    """Fault-candidate line metadata: voltage level, parallel circuits, buses, ANDES idx."""
    x = pd.read_excel(xlsx, sheet_name=["Line", "Bus"])
    L = x["Line"]
    lines = L[L["trans"] == 0].reset_index(drop=True)  # the engine's fault-candidate order
    pair = [
        tuple(sorted((int(a), int(b)))) for a, b in zip(lines["bus1"], lines["bus2"], strict=True)
    ]
    counts = pd.Series(pair).value_counts()
    vn = dict(zip(x["Bus"]["idx"], x["Bus"]["Vn"], strict=True))
    return pd.DataFrame(
        {
            "fault_line": np.arange(len(lines)),
            "line_idx": lines["idx"].astype(str),
            "bus1": lines["bus1"].astype(int),
            "bus2": lines["bus2"].astype(int),
            "vn_kv": [float(vn[b]) for b in lines["bus1"]],
            "n_circuits": [int(counts[p]) for p in pair],
            "x_pu": lines["x"].astype(float),
            "r_pu": lines["r"].astype(float),
            "b_pu": lines["b"].astype(float),
        }
    )


def candidates(xlsx: Path, ev2: str) -> list[int]:
    """gen.py's fault-candidate lines (opening the line cuts off no generator)."""
    sys.path.insert(0, ev2)
    from v2lib import Engine

    from tranpy.simulation.engine import FaultScenario

    e = Engine(
        system_name="germany",
        case=xlsx,
        simulation_time_s=6.6,
        step_s=0.01,
        record_buses="generators",
        name_width=4,
        n_jobs=1,
    )
    base = FaultScenario(scenario_id=0, fault_line=0, fault_start_s=1.0, clearing_duration_s=0.1)
    ok = []
    for line in range(e.n_lines):
        s = e._separation(replace(base, fault_line=line, trip_line=None))
        if not (s is not None and s[2]):
            ok.append(line)
    return ok


def consistency(out: Path, cases: list[str]) -> dict:
    """v4.1 events vs the brackets of the same (case, line, terminal): label unstable <=>
    clearing time > cct_hi (loading differs: v4.1 events draw 0.9-1.1 with 5 % load noise)."""
    import glob

    files = sorted(glob.glob(str(DEFAULT_OUT_ROOT / "_delta_stage/events/*.parquet")))
    if not files:
        return {}
    E = pd.concat(
        [
            pd.read_parquet(
                f,
                columns=[
                    "case",
                    "sample_id",
                    "label",
                    "fault_line",
                    "fault_position",
                    "clearing_time_s",
                    "operating_point",
                    "mean_load_factor",
                ],
            )
            for f in files
        ]
    )
    E = E[E["case"].isin(cases)]
    B = pd.concat([pd.read_parquet(out / "cct" / f"cct_{c}.parquet") for c in cases])
    M = E.merge(
        B[["case", "fault_line", "fault_position", "cct_lo", "cct_hi", "status"]],
        on=["case", "fault_line", "fault_position"],
        how="inner",
    )
    M = M[M["status"] != "failed"].copy()
    hi = M["cct_hi"].fillna(np.inf)  # above: never unstable up to hi
    lo = M["cct_lo"].fillna(-np.inf)  # below: unstable already at lo
    M["pred"] = (M["clearing_time_s"] > hi).astype(int)
    M["inside"] = (M["clearing_time_s"] > lo) & (M["clearing_time_s"] <= hi)
    M["agree"] = M["pred"] == M["label"]
    out_rows = []
    for c, g in M.groupby("case"):
        d = g[~g["inside"]]
        out_rows.append(
            {
                "case": c,
                "events": len(g),
                "inside_bracket": int(g["inside"].sum()),
                "decided": len(d),
                "agree": int(d["agree"].sum()),
                "agreement_rate": round(float(d["agree"].mean()), 4) if len(d) else None,
                "v41_unstable_pred_stable": int(((d["label"] == 1) & (d["pred"] == 0)).sum()),
                "v41_stable_pred_unstable": int(((d["label"] == 0) & (d["pred"] == 1)).sum()),
            }
        )
    dis = M[~M["inside"] & ~M["agree"]].sort_values(["case", "fault_line"])
    dis.to_csv(out / "cct_disagreements.csv", index=False)
    d = M[~M["inside"]]
    res = {
        "cases": out_rows,
        "total": {
            "events": len(M),
            "inside_bracket": int(M["inside"].sum()),
            "decided": len(d),
            "agreement_rate": round(float(d["agree"].mean()), 4) if len(d) else None,
        },
    }
    (out / "cct_consistency.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> None:
    a = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    a.add_argument("--out", default=str(HOME / "data/exp/e4_sim"))
    a.add_argument("--inputs", default=str(DEFAULT_INPUTS))
    a.add_argument("--cases", default=",".join(PRIORITY + [c for c in CASES if c not in PRIORITY]))
    a.add_argument("--jobs", type=int, default=16)
    a.add_argument("--lo", type=float, default=0.05)
    a.add_argument("--hi", type=float, default=1.0)
    a.add_argument("--width", type=float, default=0.01)
    a.add_argument("--positions", default="0,1")
    a.add_argument("--lines", default="", help="restrict to these fault_line indices (comma list)")
    a.add_argument("--checkpoint-s", type=float, default=120.0)
    A = a.parse_args()
    out = Path(A.out).expanduser()
    (out / "cct").mkdir(parents=True, exist_ok=True)
    ev2 = str(Path(A.inputs).expanduser() / "work/events_v2")
    cfg = {"lo": A.lo, "hi": A.hi, "width": A.width}
    positions = [float(p) for p in A.positions.split(",")]
    only = {int(x) for x in A.lines.split(",") if x} if A.lines else None
    cases = [c for c in A.cases.split(",") if c]
    prog_path = out / "progress.json"
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else {}
    prog.update(
        pid=os.getpid(),
        started=prog.get("started") or time.strftime("%Y-%m-%dT%H:%M:%S"),
        jobs=A.jobs,
        cfg=cfg,
        cases_planned=cases,
    )
    prog.setdefault("cases_done", [])
    done_cases = []
    for case in cases:
        xlsx = DEFAULT_OUT_ROOT / case / "prep" / f"{case}.xlsx"
        if not xlsx.exists():
            print(case, "no prep xlsx at", xlsx, "- skipped", flush=True)
            continue
        f = out / "cct" / f"cct_{case}.parquet"
        meta = line_table(xlsx)
        ok = candidates(xlsx, ev2)
        tasks = [(line, p) for line in ok for p in positions if only is None or line in only]
        have = pd.read_parquet(f)[RESULT_COLS] if f.exists() else pd.DataFrame(columns=RESULT_COLS)
        seen = set(
            zip(have["fault_line"].astype(int), have["fault_position"].astype(float), strict=True)
        )
        todo = [t for t in tasks if t not in seen]
        rows = have.to_dict("records")
        print(
            f"{case}: {len(ok)} candidate lines, {len(tasks)} tasks, {len(todo)} to do", flush=True
        )
        t0, last, n_sims = time.time(), time.time(), 0
        pr = prog.setdefault("current", {})
        pr.update(
            case=case,
            total=len(tasks),
            done=len(tasks) - len(todo),
            started=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        def flush(final: bool = False) -> None:
            df = pd.DataFrame(rows, columns=RESULT_COLS)
            df.insert(0, "case", case)
            df = df.merge(meta, on="fault_line", how="left")
            df.to_parquet(f, index=False)
            el = time.time() - t0
            n_new = len(rows) - (len(tasks) - len(todo))
            pr.update(
                done=len(rows),
                elapsed_s=round(el),
                sims=n_sims,
                sims_per_hour=round(3600 * n_sims / el) if el else None,
                sec_per_task=round(el / n_new, 1) if n_new else None,
                eta_s=round(el / n_new * (len(tasks) - len(rows))) if n_new else None,
                updated=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            prog_path.write_text(json.dumps(prog, indent=1))

        if todo:
            with Pool(A.jobs, initializer=_init, initargs=(str(xlsx), ev2, cfg)) as pool:
                for r in pool.imap_unordered(bracket, todo, chunksize=1):
                    rows.append(r)
                    n_sims += r["n_sims"]
                    if time.time() - last > A.checkpoint_s:
                        flush()
                        last = time.time()
        flush(final=True)
        done_cases.append(case)
        if case not in prog["cases_done"]:
            prog["cases_done"].append(case)
        df = pd.read_parquet(f)
        print(
            f"{case}: done {len(df)} rows in {time.time() - t0:.0f} s; status {df['status'].value_counts().to_dict()}; sims {n_sims}",
            flush=True,
        )
        # append to the combined table and refresh the consistency check
        allc = [c for c in CASES if (out / "cct" / f"cct_{c}.parquet").exists()]
        T = pd.concat(
            [pd.read_parquet(out / "cct" / f"cct_{c}.parquet") for c in allc], ignore_index=True
        )
        T.to_parquet(out / "cct_brackets.parquet", index=False)
        T.drop(columns=["sims"]).to_csv(out / "cct_brackets.csv", index=False)
        try:
            res = consistency(out, allc)
            print("consistency", json.dumps(res.get("total")), flush=True)
        except Exception as x:
            print("consistency check failed:", repr(x), flush=True)
        prog_path.write_text(json.dumps(prog, indent=1))
    prog["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    prog_path.write_text(json.dumps(prog, indent=1))
    print("all cases done:", done_cases, flush=True)


if __name__ == "__main__":
    main()
