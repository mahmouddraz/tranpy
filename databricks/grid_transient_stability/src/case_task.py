"""v4-dyn fault-event run on Databricks in three resumable stages, around the frozen events_v2
scripts.

  prep      python case_task.py --mode prep --cases all ...
            build input + ANDES case per case (run.py, stopped before its events step); one
            subprocess per case
  sim       python case_task.py --mode sim --slot K --slots N ...
            the work items (case x fixed range of random shards, or the grid sweep) of slot K, each
            through the frozen gen.py; every finished shard goes to the volume at once
  finalize  python case_task.py --mode finalize --cases all ...
            run.py --resume on the complete case (all shards present): trace selection, counts,
            manifest, report

Common options: --inputs VOL/inputs/v4dyn --out-root VOL/outputs/v4dyn/RUN_NAME --n 600 --shard 50
--grid-lines 6 --grid-clr .05,... --trace-stable 1000 --trace-seed 20261104 --trace-cand 120
--min-feasible 50 --jobs J --threads 1 --shards-per-item 3 --par P.

The simulation is the frozen code path: run.py (the same argv as the local command restricted to one
case, --hours HOUR --tans TAN, with the case's stable-trace quota of the all-case run as
--trace-stable) builds the input and the ANDES case and, at the end, selects the traces and writes
the manifest; gen.py simulates the shards with the exact command run.py would use. A sim item runs
gen.py with placeholder shard jsons (same run id, kept 0) for the shards outside its range, which
gen.py skips as already done; the placeholders never leave the local disk. Random scenarios, fault
lines and the grid sweep are drawn by gen.py from the case seed before any shard is simulated, so a
shard's events do not depend on which item runs it.

Trace selection is per case and exact: gen.py traces every unstable event and the case's first
--trace-cand stable candidates (seeded order); run.py --resume keeps the case's quota of them
(--trace-stable split over the 18 cases in run.py's case order, 56 or 55 for 1000) and prunes the
rest from final/. shards/ keeps the unpruned files. The events parquet column `traced` is gen.py's
flag (trace written at simulation time) and is not updated by the pruning; merge_task.py writes the
selection to the trace_sample table.

Volume layout under OUT-ROOT/CASE:
  prep/ + prep.json     run.py's case folder after build + case (manifest, inputs, CASE.xlsx, ...),
                        sha256 list
  shards/TAG.{events,traces}.parquet, shards/TAG.json, shards/info.json
                        each shard as gen.py wrote it (unpruned stable trace candidates)
  shards_ok/TAG.json    written after the shard files: their sha256; a shard counts as done only
                        if its files match these hashes (resume and finalize skip/collect exactly
                        these)
  logs/                 gen.py logs of the items
  final/ + final.json   run.py's complete case folder after --resume (pruned traces,
                        run_manifest.json, report, dbx_case.json); final.json is written last
Every stage skips work that is already on the volume with matching hashes, so a failed or
interrupted task is resumed by re-running it (task retry, repair run, or a new job run with the same
run_name).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    HERE = Path(__file__).resolve().parent
except NameError:  # pragma: no cover
    HERE = Path(os.getcwd())
sys.path.insert(0, str(HERE))
import common

PIN_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
POLL_S = 15


class StopBeforeEvents(Exception):
    pass


def args():
    a = argparse.ArgumentParser()
    a.add_argument(
        "--mode", required=True, choices=("prep", "sim", "finalize", "prep1", "finalize1")
    )
    a.add_argument("--cases", default="all")
    a.add_argument("--case")
    a.add_argument("--slot", type=int, default=0)
    a.add_argument("--slots", type=int, default=1)
    a.add_argument("--inputs", required=True)
    a.add_argument("--out-root", required=True)
    a.add_argument("--staged", help="(internal) staging root of the parent task")
    a.add_argument("--n", default="600")
    a.add_argument("--shard", default="50")
    a.add_argument("--grid-lines", default="6")
    a.add_argument("--grid-clr", default=".05,.1,.15,.2,.3,.4,.5,.6")
    a.add_argument("--trace-stable", type=int, default=1000)
    a.add_argument("--trace-seed", default="20261104")
    a.add_argument("--trace-cand", default="120")
    a.add_argument("--min-feasible", default="50")
    a.add_argument("--jobs", type=int, default=4, help="gen.py worker processes per item")
    a.add_argument(
        "--threads", type=int, default=1, help="BLAS/OpenMP threads per process (run.py --threads)"
    )
    a.add_argument("--shards-per-item", type=int, default=3)
    a.add_argument(
        "--par", type=int, default=0, help="cases in parallel in prep/finalize; 0 = CPUs"
    )
    a.add_argument(
        "--retries", type=int, default=2, help="gen.py retries per item (as run.py's --retries)"
    )
    return a.parse_args()


# ---------------------------------------------------------------- helpers
def wjson(obj, p):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=str))


def rjson(p):
    return json.loads(Path(p).read_text())


def put(src, dst):
    """Copy a local file to the volume (plain sequential write)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def pins_env(threads, home=None):
    e = dict(os.environ)
    for k in PIN_VARS:
        e.pop(k, None)
        if threads:
            e[k] = str(threads)
    if home:
        e["HOME"] = str(home)
    return e


def case_split(case):
    hour, tan = case.rsplit("_t", 1)
    return hour, f"{int(tan) / 100:g}"


def tags(A):
    n, sh = int(A.n), int(A.shard)
    t = [f"r{k:03d}" for k in range((n + sh - 1) // sh)]
    if int(A.grid_lines) * len(A.grid_clr.split(",")) > 0:
        t.append("grid")
    return t


def items(A):
    """Work items in a fixed order: per case, chunks of --shards-per-item random shards, then the
    grid sweep."""
    out = []
    for c in common.CASES:
        t = tags(A)
        r = [x for x in t if x != "grid"]
        k = max(1, A.shards_per_item)
        out += [(c, r[i : i + k]) for i in range(0, len(r), k)]
        if "grid" in t:
            out.append((c, ["grid"]))
    return out


def run_argv(A, case, work, out):
    """run.py's argv for one case (identical in prep and finalize, so --resume accepts it)."""
    hour, tan = case_split(case)
    return [
        "run.py",
        "--model",
        str(work / "model_v4final.pkl"),
        "--out",
        str(out),
        "--settings",
        "v4g",
        "--neighbours",
        "ce_entsoe",
        "--store",
        "compact",
        "--hours",
        hour,
        "--tans",
        tan,
        "--n",
        A.n,
        "--shard",
        A.shard,
        "--grid-lines",
        A.grid_lines,
        "--grid-clr",
        A.grid_clr,
        "--trace-stable",
        str(common.trace_quota(case, A.trace_stable)),
        "--trace-seed",
        A.trace_seed,
        "--trace-cand",
        A.trace_cand,
        "--min-feasible",
        A.min_feasible,
        "--jobs",
        str(A.jobs),
        "--threads",
        str(A.threads),
    ]


def stage(A):
    root = Path(
        tempfile.mkdtemp(
            prefix="tranpy_", dir="/local_disk0/tmp" if Path("/local_disk0/tmp").is_dir() else None
        )
    )
    home, work, rec = common.stage(A.inputs, root)
    common.check_pickles(work)
    # generate the ANDES code once into the staged HOME, before parallel processes would race for it
    t = time.time()
    p = subprocess.run(
        [sys.executable, "-m", "andes", "prepare", "-q"],
        env=pins_env(A.threads, home),
        capture_output=True,
        text=True,
    )
    common.log("andes prepare", "rc", p.returncode, round(time.time() - t, 1), "s")
    return root, home, work, rec


def tree_hashes(d):
    d = Path(d)
    return {str(p.relative_to(d)): common.sha256(p) for p in sorted(d.rglob("*")) if p.is_file()}


def upload_tree(src, dst):
    """Mirror local folder src to the volume folder dst (dst is replaced).

    Returns {relpath: sha256}."""
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    h = tree_hashes(src)
    for rel in h:
        put(Path(src) / rel, dst / rel)
    return h


def verified(vol_dir, hashes):
    """True if every file of hashes exists under vol_dir with that sha256."""
    try:
        return all(
            (Path(vol_dir) / r).is_file() and common.sha256(Path(vol_dir) / r) == h
            for r, h in hashes.items()
        )
    except OSError:
        return False


def prep_record(C):
    p = C / "prep.json"
    if not p.exists():
        return None
    P = rjson(p)
    return P if verified(C / "prep", P["files"]) else None


def final_ok(C, argv_rest):
    p = C / "final.json"
    if not p.exists():
        return False
    F = rjson(p)
    return (
        F.get("argv_rest") == argv_rest and F.get("exit") == 0 and verified(C / "final", F["files"])
    )


def shard_ok(C, tag, run_id):
    """Sidecar of a shard if its files on the volume match it, else None."""
    p = C / "shards_ok" / f"{tag}.json"
    if not p.exists():
        return None
    S = rjson(p)
    if S.get("run_id") != run_id:
        return None
    return S if verified(C / "shards", S["files"]) else None


def argv_rest(argv):
    """run.py argv without the local paths (for comparing runs)."""
    skip = {"--model", "--out"}
    out, i = [], 1
    while i < len(argv):
        if argv[i] in skip:
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def children(A, mode, cases, staged):
    """Run one subprocess per case (mode prep1/finalize1), --par at a time.

    Returns {case: exit code}."""
    par = A.par or os.cpu_count() or 4
    base = [sys.executable, str(HERE / "case_task.py"), "--mode", mode, "--staged", str(staged)]
    for k in (
        "inputs",
        "out_root",
        "n",
        "shard",
        "grid_lines",
        "grid_clr",
        "trace_stable",
        "trace_seed",
        "trace_cand",
        "min_feasible",
        "jobs",
        "threads",
        "shards_per_item",
    ):
        base += ["--" + k.replace("_", "-"), str(getattr(A, k))]

    def one(c):
        t = time.time()
        p = subprocess.run([*base, "--case", c], capture_output=True, text=True)
        tail = (p.stdout + p.stderr)[-3000:]
        common.log(
            f"{mode} {c}: exit {p.returncode} in {round(time.time() - t)} s",
            "" if p.returncode == 0 else "\n" + tail,
        )
        return p.returncode

    with ThreadPoolExecutor(par) as ex:
        return dict(zip(cases, ex.map(one, cases), strict=False))


def import_run(home, work, repo):
    os.environ["HOME"] = str(home)
    sys.path.insert(0, str(work / "events_v2"))
    import run  # staged events_v2/run.py, sha256 checked by common.stage

    assert work == run.W, (run.W, work)
    run.REPO = repo
    return run


def staged_paths(staged):
    home = Path(staged) / "home"
    return home, home / "Library/Caches/tranpy/work"


# ---------------------------------------------------------------- prep
def prep1(A):
    case, C = A.case, Path(A.out_root) / A.case
    home, work = staged_paths(A.staged)
    root = Path(tempfile.mkdtemp(prefix=f"prep_{case}_", dir=A.staged))
    out = root / "out"
    argv = run_argv(A, case, work, out)
    P = prep_record(C)
    if P:
        if P["argv_rest"] != argv_rest(argv):
            sys.exit(f"{C} holds a prep of other arguments {P['argv_rest']}; use a new run_name")
        common.log("prep", case, "already on the volume (hashes ok), run", P["run_id"])
        return
    (root / "repo").mkdir()
    run = import_run(home, work, root / "repo")
    orig = run.step

    def step(cmd, log, label, out_dir, retries, M):
        if label.endswith(":events"):
            raise StopBeforeEvents
        return orig(cmd, log, label, out_dir, retries, M)

    run.step = step
    sys.argv = argv
    t = time.time()
    try:
        run.main()
        sys.exit(
            f"{case}: run.py finished without reaching the events step (build or case failed?)"
        )
    except StopBeforeEvents:
        pass
    M = rjson(out / "run_manifest.json")
    if not (out / f"{case}.xlsx").exists() or not (out / f"{case}.mk.json").exists():
        sys.exit(f"{case}: no ANDES case after prep; see {out}/logs")
    files = upload_tree(out, C / "prep")
    wjson(
        {
            "case": case,
            "run_id": M["run_id"],
            "argv_rest": argv_rest(argv),
            "files": files,
            "sec": round(time.time() - t, 1),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        C / "prep.json",
    )
    common.log(
        "prep",
        case,
        "run",
        M["run_id"],
        "uploaded",
        len(files),
        "files in",
        round(time.time() - t),
        "s",
    )


# ---------------------------------------------------------------- sim
class Uploader(threading.Thread):
    """Copies every shard gen.py has finished (its TAG.json exists) to the volume, files first,
    sidecar last."""

    def __init__(s, D, C, mine, run_id):
        super().__init__(daemon=True)
        s.D, s.C, s.mine, s.run_id, s.done, s.stop_ev, s.err = (
            D,
            C,
            set(mine),
            run_id,
            set(),
            threading.Event(),
            None,
        )

    def sync(s):
        if (s.D / "info.json").exists() and not (s.C / "shards" / "info.json").exists():
            put(s.D / "info.json", s.C / "shards" / "info.json")
        for tag in sorted(s.mine - s.done):
            j = s.D / f"{tag}.json"
            if not j.exists():
                continue
            J = rjson(j)
            if J.get("placeholder"):
                continue
            files = [x for x in (J.get("events"), J.get("traces")) if x] + [j.name]
            h = {}
            for f in files:
                put(s.D / f, s.C / "shards" / f)
                h[f] = common.sha256(s.D / f)
            wjson(
                {
                    "run_id": s.run_id,
                    "tag": tag,
                    "files": h,
                    "kept": J.get("kept"),
                    "req": J.get("req"),
                    "sec": J.get("sec"),
                    "uploaded": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                s.C / "shards_ok" / f"{tag}.json",
            )
            s.done.add(tag)
            common.log(
                "uploaded",
                s.C.name,
                tag,
                "kept",
                J.get("kept"),
                "of",
                J.get("req"),
                "in",
                J.get("sec"),
                "s",
            )

    def run(s):
        while not s.stop_ev.wait(POLL_S):
            try:
                s.sync()
            except Exception as e:  # retried at the next poll and at the final sync
                s.err = repr(e)
                common.log("upload error (will retry)", s.err)


def sim_item(A, home, work, root, case, mine):
    C = Path(A.out_root) / case
    probe_argv = run_argv(A, case, work, root)
    if final_ok(C, argv_rest(probe_argv)):
        common.log("item", case, mine, "case already final; skipped")
        return 0
    P = prep_record(C)
    if not P:
        common.log("item", case, mine, "ERROR: no verified prep on the volume")
        return 1
    if P["argv_rest"] != argv_rest(probe_argv):
        common.log("item", case, mine, "ERROR: prep was made with other arguments", P["argv_rest"])
        return 1
    rid = P["run_id"]
    todo = [t for t in mine if not shard_ok(C, t, rid)]
    if not todo:
        common.log("item", case, mine, "all shards already on the volume")
        return 0
    tmp_out = Path(tempfile.mkdtemp(prefix=f"sim_{case}_", dir=root))
    D = tmp_out / case
    D.mkdir()
    put(C / "prep" / f"{case}.xlsx", tmp_out / f"{case}.xlsx")
    if common.sha256(tmp_out / f"{case}.xlsx") != P["files"][f"{case}.xlsx"]:
        common.log("item", case, mine, "ERROR: case xlsx hash differs from prep.json")
        return 1
    if (C / "shards" / "info.json").exists():
        put(C / "shards" / "info.json", D / "info.json")
    for t in tags(A):
        # outside this item or already on the volume: placeholder, skipped by gen.py, never uploaded
        if t not in todo:
            wjson({"run_id": rid, "tag": t, "kept": 0, "placeholder": True}, D / f"{t}.json")
    sf = ["store=compact", f"trace_seed={A.trace_seed}", f"trace_cand={A.trace_cand}"]
    cmd = [
        sys.executable,
        "-W",
        "ignore",
        str(work / "events_v2" / "gen.py"),
        str(tmp_out),
        case,
        A.n,
        A.shard,
        rid,
        f"grid_lines={A.grid_lines}",
        f"grid_clr={A.grid_clr}",
        f"jobs={A.jobs}",
        *sf,
    ]
    log = tmp_out / "gen.log"
    up = Uploader(D, C, todo, rid)
    up.start()
    t0, rc, attempts = time.time(), 1, 0
    for attempt in range(A.retries + 1):
        attempts += 1
        with open(log, "a") as f:
            f.write(
                f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} {case} {','.join(todo)} "
                f"attempt {attempt + 1}: {' '.join(cmd)}\n"
            )
            f.flush()
            rc = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(tmp_out),
                env=pins_env(A.threads, home),
            ).returncode
        if rc == 0:
            break
        common.log("item", case, todo, f"gen.py rc {rc} (attempt {attempt + 1}/{A.retries + 1})")
    up.stop_ev.set()
    up.join()
    # final sync; a shard that cannot be uploaded is reported missing (the item fails, a retry
    # resumes)
    for _k in range(3):
        try:
            up.sync()
            break
        except Exception as e:
            common.log("final upload error", repr(e))
            time.sleep(10)
    name = f"{'_'.join(mine) if len(mine) < 4 else mine[0] + '-' + mine[-1]}"
    put(log, C / "logs" / f"gen_{name}_{time.strftime('%Y%m%dT%H%M%S')}.log")
    missing = [t for t in todo if not shard_ok(C, t, rid)]
    common.log(
        "item",
        case,
        mine,
        "rc",
        rc,
        "attempts",
        attempts,
        "sec",
        round(time.time() - t0),
        "missing",
        missing,
    )
    shutil.rmtree(tmp_out, ignore_errors=True)
    return 0 if rc == 0 and not missing else 1


def sim(A):
    env = common.check_environment()
    root, home, work, _ = stage(A)
    allit = [it for it in items(A) if A.cases == "all" or it[0] in A.cases.split(",")]
    mine = allit[A.slot :: A.slots]
    common.log(
        f"slot {A.slot}/{A.slots}: {len(mine)} of {len(allit)} items, jobs {A.jobs}, "
        f"threads {A.threads}, cpus {os.cpu_count()}",
        "python",
        env["python"],
    )
    bad = []
    for case, t in mine:
        if sim_item(A, home, work, root, case, t):
            bad.append(f"{case}:{','.join(t)}")
    shutil.rmtree(root, ignore_errors=True)
    if bad:
        sys.exit(f"slot {A.slot}: {len(bad)} item(s) failed: {bad}")


# ---------------------------------------------------------------- finalize
def finalize1(A):
    case, C = A.case, Path(A.out_root) / A.case
    home, work = staged_paths(A.staged)
    root = Path(tempfile.mkdtemp(prefix=f"fin_{case}_", dir=A.staged))
    out = root / "out"
    argv = run_argv(A, case, work, out)
    if final_ok(C, argv_rest(argv)):
        common.log("finalize", case, "already final")
        return
    P = prep_record(C)
    if not P or P["argv_rest"] != argv_rest(argv):
        sys.exit(f"{case}: no verified prep of these arguments on the volume")
    rid = P["run_id"]
    for rel in P["files"]:
        put(C / "prep" / rel, out / rel)
    D = out / case
    D.mkdir(parents=True, exist_ok=True)
    missing = []
    for t in tags(A):
        S = shard_ok(C, t, rid)
        if not S:
            missing.append(t)
            continue
        for f in S["files"]:
            put(C / "shards" / f, D / f)
    if missing:
        sys.exit(f"{case}: shards missing or not verified on the volume: {missing}")
    put(C / "shards" / "info.json", D / "info.json")
    bad = [r for r, h in P["files"].items() if common.sha256(out / r) != h]
    if bad:
        sys.exit(f"{case}: restored prep files differ from prep.json: {bad}")
    (root / "repo").mkdir()
    run = import_run(home, work, root / "repo")
    sys.argv = [*argv, "--resume", rid]
    code, t = 0, time.time()
    try:
        run.main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        if code:
            common.log("run.py exited", e.code)
    env = common.check_environment()
    wjson(
        dict(
            case=case,
            exit=code,
            quota=common.trace_quota(case, A.trace_stable),
            trace_stable_total=A.trace_stable,
            jobs=A.jobs,
            threads=A.threads,
            argv=sys.argv[1:],
            **env,
            staging=rjson(Path(A.staged) / "staging.json"),
        ),
        out / "dbx_case.json",
    )
    files = upload_tree(out, C / "final")
    wjson(
        {
            "case": case,
            "run_id": rid,
            "exit": code,
            "argv_rest": argv_rest(argv),
            "files": files,
            "sec": round(time.time() - t, 1),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        C / "final.json",
    )
    common.log("finalize", case, "exit", code, "uploaded", len(files), "files")
    if code:
        sys.exit(f"{case}: run.py failed (exit {code}); see {C}/final/run_manifest.json")


def driver(A, mode):
    common.check_environment()
    root, _home, _work, rec = stage(A)
    wjson(rec, root / "staging.json")
    cases = common.CASES if A.cases == "all" else A.cases.split(",")
    res = children(A, mode, cases, root)
    shutil.rmtree(root, ignore_errors=True)
    bad = {c: r for c, r in res.items() if r}
    common.log(mode, "done:", len(res) - len(bad), "ok,", len(bad), "failed", bad)
    if bad:
        sys.exit(f"{mode} failed for {sorted(bad)}")


def main():
    A = args()
    if A.mode == "prep":
        driver(A, "prep1")
    elif A.mode == "finalize":
        driver(A, "finalize1")
    elif A.mode == "prep1":
        prep1(A)
    elif A.mode == "finalize1":
        finalize1(A)
    else:
        sim(A)


if __name__ == "__main__":
    main()
