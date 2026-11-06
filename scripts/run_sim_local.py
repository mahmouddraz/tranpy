"""Run the v4.1 fault-event simulation of one case locally (no Spark, no dbutils).

python scripts/run_sim_local.py --case high_load_t10 --out OUTDIR [--inputs VOL/inputs/v41dyn]
       [--prep DIR] [--jobs 16] [--n 600] [--shard 50] [--grid-lines 6] [--grid-clr .05,...]
       [--run-id ID] [--trace-seed 20261104] [--trace-cand 120] [--ids 1,2,3] [--shards r000,grid]

The same code path as the Databricks job (databricks/grid_transient_stability/src/case_task.py
--mode sim): the frozen events_v2 scripts of the input set (sha256-checked against
FROZEN_DYN_v41.md) draw the scenarios of the case from the case seed, simulate them with the
frozen v2lib.Engine (AndesEngine with failure categories and the step/4 retry) and write the
compact storage (TAG.events.parquet, TAG.traces.parquet, TAG.json, info.json) with
v2lib.write_compact, so tranpy.datasets.load_compact_events / read_compact_event_table and
tranpy.ml_events read the output unchanged. The scenario drawing is gen.py's, line for line, so
a shard holds the same events (same sample ids, lines, loadings, clearing times) as the
Databricks run; the output folder is OUTDIR/CASE/ as gen.py's.

--ids restricts the simulation to those sample ids (every one traced; for validation against
the Databricks shards), --shards to those shard tags. The ANDES case is CASE.xlsx from --prep
(default: the case's prep folder of the Databricks run mirrored under ~/data/volume), copied
next to the output as gen.py expects. Worker processes: --jobs (ProcessPoolExecutor inside the
engine, one BLAS thread each).
"""
# ruff: noqa: E501, E741

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import zlib
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

HOME = Path.home()
DEFAULT_INPUTS = HOME / "data/volume/inputs/v41dyn"
DEFAULT_OUT_ROOT = HOME / "data/volume/outputs/v41dyn/738439695169033"
CLR = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6)
PIN_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def frozen_scripts(inputs: Path, table: str = "FROZEN_DYN_v41.md") -> Path:
    """events_v2 folder of the input set, every script checked against the frozen table."""
    work = inputs / "work"
    want = dict(
        re.findall(r"^\| (events_v2/\S+) \| ([0-9a-f]{64}) \|$", (work / table).read_text(), re.M)
    )
    bad = {p: (sha256(work / p), h) for p, h in want.items() if sha256(work / p) != h}
    if not want or bad:
        sys.exit(f"frozen events_v2 scripts differ from {table} (have, want): {bad or 'none'}")
    return work / "events_v2"


def pin_threads(threads: int) -> None:
    for k in PIN_VARS:
        os.environ[k] = str(threads)


def scenarios(engine, name: str, n: int, grid_lines: int, grid_clr: tuple[float, ...], sampler):
    """gen.py's scenario drawing: random shards and the grid sweep, from the case seed."""
    seed = zlib.crc32(name.encode()) % 10**6
    S = sampler(
        n_lines=engine.n_lines,
        n_loads=engine.n_loads,
        loading=(0.9, 1.1),
        clearing_s=(0.05, 0.6),
        fault_time_s=1.0,
        post_clearing_s=5.0,
        fault_impedance_ohm=(0.5, 0.5),
    )
    SC = list(S.sample(n, seed=seed))
    base = SC[0]

    def island(line: int) -> bool:
        s = engine._separation(replace(base, fault_line=line, trip_line=None))
        return s is not None and bool(s[2])

    ok = [line for line in range(engine.n_lines) if not island(line)]
    rng = np.random.default_rng(seed)
    SC = [replace(sc, fault_line=int(rng.choice(ok)), trip_line=None) for sc in SC]
    sel = rng.choice(ok, grid_lines, replace=False)
    G = [
        replace(
            base,
            scenario_id=100000 + k,
            fault_line=int(line),
            fault_start_s=1.0,
            clearing_duration_s=c,
            load_change_pct=0.0,
            operating_point=1.0,
            load_factors=None,
            fault_reactance_ohm=0.5,
        )
        for k, (line, c) in enumerate((line, c) for line in sel for c in grid_clr)
    ]
    return seed, ok, [int(x) for x in sel], SC, G


def main() -> None:
    a = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    a.add_argument("--case", required=True)
    a.add_argument("--out", required=True, help="output folder (OUT/CASE/ holds the shards)")
    a.add_argument("--inputs", default=str(DEFAULT_INPUTS))
    a.add_argument("--prep", help="folder with CASE.xlsx (default: the Databricks prep folder)")
    a.add_argument("--jobs", type=int, default=16)
    a.add_argument("--threads", type=int, default=1)
    a.add_argument("--n", type=int, default=600)
    a.add_argument("--shard", type=int, default=50)
    a.add_argument("--grid-lines", type=int, default=6)
    a.add_argument("--grid-clr", default=",".join(map(str, CLR)))
    a.add_argument("--run-id", default="")
    a.add_argument("--trace-seed", type=int, default=20261104)
    a.add_argument("--trace-cand", type=int, default=120)
    a.add_argument("--ids", default="", help="sample ids to simulate (comma list), all traced")
    a.add_argument("--shards", default="", help="shard tags to simulate (comma list)")
    A = a.parse_args()
    pin_threads(A.threads)
    inputs = Path(A.inputs).expanduser()
    ev2 = frozen_scripts(inputs)
    sys.path.insert(0, str(ev2))
    from v2lib import PINNED, Engine, gen_bus_map, trace_priority, wjson, write_compact
    from v2lib import versions as pkg_versions

    from tranpy.simulation.engine import generate_events
    from tranpy.simulation.protocol import StressTestSampler

    versions = pkg_versions()
    # pandapower is pinned by the frozen scripts but unused by the simulation; the other pins hold
    bad = {k: (versions[k], w) for k, w in PINNED.items() if versions[k] != w and k != "pandapower"}
    if bad:
        sys.exit(f"package versions differ from the pins (have, want): {bad}")
    name = A.case
    prep = Path(A.prep).expanduser() if A.prep else DEFAULT_OUT_ROOT / name / "prep"
    O = Path(A.out).expanduser().resolve()
    O.mkdir(parents=True, exist_ok=True)
    xlsx = O / f"{name}.xlsx"
    if not xlsx.exists():
        shutil.copyfile(prep / f"{name}.xlsx", xlsx)
    D = O / name
    D.mkdir(exist_ok=True)
    run_id = A.run_id or time.strftime("%Y%m%dT%H%M%S") + "-local"
    clr = tuple(float(c) for c in A.grid_clr.split(","))
    t0 = time.time()
    e = Engine(
        system_name="germany",
        case=xlsx,
        simulation_time_s=6.6,
        step_s=0.01,
        record_buses="generators",
        name_width=4,
        n_jobs=A.jobs,
    )
    seed, ok, sel, SC, G = scenarios(e, name, A.n, A.grid_lines, clr, StressTestSampler)
    info = {
        "run_id": run_id,
        "case": name,
        "seed": seed,
        "lines": e.n_lines,
        "candidates": len(ok),
        "n": A.n,
        "shard": A.shard,
        "grid_lines": sel,
        "grid_clr": list(clr),
        "store": "compact",
        "trace_seed": A.trace_seed,
        "trace_cand": A.trace_cand,
    }
    pri = trace_priority(name, [int(x.scenario_id) for x in SC + G], A.trace_seed)
    cand = sorted((i for i, r in pri.items() if r < A.trace_cand), key=pri.get)
    info["trace_candidates"] = cand
    cand = set(cand)
    info.update(
        local={
            "xlsx_sha256": sha256(xlsx),
            "prep": str(prep),
            "versions": versions,
            "python": sys.version.split()[0],
            "jobs": A.jobs,
            "threads": A.threads,
            "ids": A.ids,
            "shards": A.shards,
        }
    )
    wjson(info, D / "info.json")
    print(
        f"{name}: {e.n_lines} lines, {len(ok)} candidates, seed {seed}, setup {time.time() - t0:.1f} s",
        flush=True,
    )
    parts = [
        (f"r{k:03d}", SC[k * A.shard : (k + 1) * A.shard])
        for k in range((A.n + A.shard - 1) // A.shard)
    ]
    parts.append(("grid", G))
    want_ids = {int(i) for i in A.ids.split(",") if i} if A.ids else None
    want_tags = set(A.shards.split(",")) if A.shards else None
    gmap = gen_bus_map(xlsx)
    errors, done = 0, []
    for tag, sc in parts:
        if want_tags and tag not in want_tags:
            continue
        if want_ids is not None:
            sc = [s for s in sc if int(s.scenario_id) in want_ids]
        if not sc:
            continue
        j = D / f"{tag}.json"
        if j.exists() and json.loads(j.read_text()).get("run_id") == run_id:
            print(name, tag, "resumed (already done in this run)", flush=True)
            continue
        rec = _Rec(e)
        t = time.time()
        req = [int(s.scenario_id) for s in sc]
        try:
            d = generate_events(rec, sc, name=f"de_{name}_{tag}", protocol="stress_test")
            kept = [int(i) for i in d.sample_ids]
        except Exception as x:
            if "all simulations failed" in str(x) and len(rec.failed) == len(sc):
                d, kept = None, []
            else:
                errors += 1
                print(name, tag, "ERROR", repr(x)[:300], flush=True)
                continue
        dt = time.time() - t
        lost = sorted(set(req) - set(kept) - {r["id"] for r in rec.failed})
        if lost:
            errors += 1
            print(name, tag, "ERROR unaccounted scenarios", lost, flush=True)
            continue
        traced, ft = [], None
        if d is not None:
            tid = [
                i
                for i, lab in zip(d.sample_ids, d.labels, strict=True)
                if int(lab) == 1 or int(i) in cand or want_ids is not None
            ]
            _, ft, traced = write_compact(
                d,
                D / tag,
                gmap,
                tid,
                {
                    "run_id": run_id,
                    "case": name,
                    "tag": tag,
                    "trace_seed": A.trace_seed,
                    "trace_cand": A.trace_cand,
                },
            )
        by = Counter(r["category"] for r in rec.failed)
        wjson(
            {
                "run_id": run_id,
                "tag": tag,
                "kind": "grid" if tag == "grid" else "random",
                "req": len(req),
                "kept": len(kept),
                "requested_ids": req,
                "kept_ids": kept,
                "failed": rec.failed,
                "failed_by": dict(by),
                "sec": round(dt, 1),
                "unstable": float(np.mean(d.labels)) if d is not None else None,
                "events": f"{tag}.events.parquet" if d is not None else None,
                "traces": ft.name if ft else None,
                "traced_ids": traced,
            },
            j,
        )
        done.append((tag, len(req), len(kept), dt))
        print(
            f"{name} {tag} req {len(req)} kept {len(kept)} failed {dict(by)} unstable "
            f"{np.mean(d.labels) if d is not None else None} sec {dt:.0f} "
            f"({dt / len(req):.1f} s/event with {A.jobs} workers)",
            flush=True,
        )
    n_ev = sum(r for _, r, _, _ in done)
    sec = sum(dt for *_, dt in done)
    wjson(
        {
            "run_id": run_id,
            "case": name,
            "shards": [t for t, *_ in done],
            "events": n_ev,
            "sec": round(sec, 1),
            "events_per_hour": round(3600 * n_ev / sec, 1) if sec else None,
            "jobs": A.jobs,
            "errors": errors,
        },
        D / "local_run.json",
    )
    if errors:
        sys.exit(f"{name}: {errors} shard(s) errored")
    print(name, "all shards written:", n_ev, "events in", round(sec), "s", flush=True)


class _Rec:
    """gen.py's Rec: passes simulate through and keeps each result's failure category."""

    def __init__(self, e):
        self.e, self.failed = e, []

    def __getattr__(self, k):
        return getattr(self.e, k)

    def simulate(self, sc):
        from v2lib import CATEGORIES

        r = self.e.simulate(sc)
        for x in r:
            if not x.completed:
                assert x.failure in CATEGORIES, x.failure
                self.failed.append({"id": int(x.scenario.scenario_id), "category": x.failure})
        return r


if __name__ == "__main__":
    main()
