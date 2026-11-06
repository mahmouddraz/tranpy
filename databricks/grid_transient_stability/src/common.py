"""Shared helpers of the Databricks fault-event tasks: pins, input staging and hash checks.

The frozen pipeline scripts (events_v2/*, pf_v4final/go4.py) are run unchanged. They resolve the
work directory as Path.home()/"Library/Caches/tranpy/work", so a task stages the uploaded copies
under a local HOME with that layout and points HOME there; run.py's REPO (only used as the working
directory of its subprocesses) is replaced by a local directory. Every staged file is checked
against SHA256SUMS on the volume and against the sha256 tables of FROZEN_MODEL_v4final.md and
FROZEN_DYN (FROZEN_DYN_v41.md) before anything runs.
"""

import hashlib
import importlib.metadata as md
import json
import re
import shutil
import sys
from pathlib import Path

# Environment A of retest_v4final/ENVIRONMENTS.md (the packages the simulation imports).
PINS = {
    "andes": "2.0.0",
    "pypowsybl": "1.16.1",
    "pandapower": "3.1.2",
    "numpy": "2.5.4",
    "pandas": "3.1.0",
    "scipy": "1.18.1",
    "pyarrow": "26.0.1",
    "scikit-learn": "1.9.1",
    "kvxopt": "1.3.3.3",
    "sympy": "1.14.0",
    "openpyxl": "3.1.5",
    "tranpy-ai": "0.2.0.dev0",
}
LOCAL_PYTHON = "3.13.6"  # env A (macOS); the Databricks Python is recorded next to it
HOURS_V4G = [
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
TANS = (0.1, 0.2)
CASES = [f"{h}_t{round(t * 100):02d}" for h in HOURS_V4G for t in TANS]  # run.py's case order
FROZEN_DYN = "FROZEN_DYN_v41.md"  # v4.1: German FCR at the published 564 MW (see that file)
WHEEL_FILES = {  # FROZEN_DYN path -> path inside the installed package
    "src/tranpy/simulation/neighbour_inertia_entsoe.csv": (
        "tranpy/simulation/neighbour_inertia_entsoe.csv"
    ),
    "src/tranpy/simulation/border_short_circuit.csv": "tranpy/simulation/border_short_circuit.csv",
    "src/tranpy/simulation/grid_case.py": "tranpy/simulation/grid_case.py",
    "src/tranpy/datasets.py (uncommitted, reader only)": "tranpy/datasets.py",
}


def log(*a):
    print("[tranpy-dbx]", *a, flush=True)  # noqa: T201


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def check_environment():
    """Python version and package pins; exits with a clear message if a pin is not met."""
    have = {}
    for p in PINS:
        try:
            have[p] = md.version(p)
        except md.PackageNotFoundError:
            have[p] = None
    bad = {p: (have[p], w) for p, w in PINS.items() if have[p] != w}
    py = sys.version.split()[0]
    log("python", py, "(local env A:", LOCAL_PYTHON + ")", "versions", json.dumps(have))
    if bad:
        sys.exit(f"PIN NOT MET (have, want): {bad}")
    if sys.version_info < (3, 12):
        sys.exit(
            f"python {py} is older than 3.12; "
            f"numpy {PINS['numpy']} / pandas {PINS['pandas']} need >= 3.12"
        )
    return {"python": py, "versions": have}


def frozen_table(md_file):
    """{path: sha256} of a FROZEN_*.md table."""
    return dict(re.findall(r"^\| (.+?) \| ([0-9a-f]{64}) \|$", Path(md_file).read_text(), re.M))


def stage(inputs, root, frozen_dyn=None):
    """Copy the uploaded work tree to ROOT/home/Library/Caches/tranpy/work and verify all hashes.

    frozen_dyn names the FROZEN_DYN table of the input set (default FROZEN_DYN; the v4.1 trace
    replay set uses FROZEN_DYN_v41_traces.md, whose table carries the fixed trace writer).
    Returns (home, work, record)."""
    inputs = Path(inputs)
    frozen_dyn = frozen_dyn or FROZEN_DYN
    sums = {}
    for line in (inputs / "SHA256SUMS").read_text().splitlines():
        h, p = line.split(None, 1)
        sums[p.strip().removeprefix("./")] = h
    home = Path(root) / "home"
    work = home / "Library/Caches/tranpy/work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    got = {}
    for rel in sorted(sums):
        if not rel.startswith("work/"):
            continue
        dst = work / rel[len("work/") :]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inputs / rel, dst)
        got[rel] = sha256(dst)
    bad = {p: (got[p], sums[p]) for p in got if got[p] != sums[p]}
    if bad:
        sys.exit(f"staged files differ from SHA256SUMS (have, want): {bad}")
    # the frozen tables, checked against the staged copies
    fm = frozen_table(work / "FROZEN_MODEL_v4final.md")
    fd = frozen_table(work / frozen_dyn)
    if len(fm) != 4:
        sys.exit(f"FROZEN_MODEL_v4final.md: expected 4 hashes, found {len(fm)}")
    bad = {}
    for p, want in fm.items():
        have = sha256(work / p)
        if have != want:
            bad[p] = (have, want)
    for p, want in fd.items():
        if p in WHEEL_FILES:
            continue
        have = sha256(work / p)
        if have != want:
            bad[p] = (have, want)
    # the installed wheel's frozen files
    import importlib.util

    site = Path(importlib.util.find_spec("tranpy").origin).parent.parent
    for p, inside in WHEEL_FILES.items():
        have = sha256(site / inside)
        if have != fd[p]:
            bad[p] = (have, fd[p])
    if bad:
        sys.exit(f"FROZEN HASH MISMATCH (have, want): {bad}")
    log(
        f"staged {len(got)} files to {work}; SHA256SUMS ok; "
        f"frozen model ({len(fm)}) and dyn ({frozen_dyn}: {len(fd)}) hashes ok"
    )
    return (
        home,
        work,
        {
            "sha256sums_checked": len(got),
            "frozen_dyn_file": frozen_dyn,
            "frozen_model": fm,
            "frozen_dyn": fd,
            "staged_sha256": got,
        },
    )


def check_pickles(work):
    """The frozen model and one hour input unpickle under the pinned pandas."""
    import pickle

    import pandas as pd

    s, _m = pd.read_pickle(work / "model_v4final.pkl")
    with open(work / "pf_v4final/input_v4g_high_wind.pkl", "rb") as f:
        inp = pickle.load(f)
    log("pickles load: model tables", len(s), "input base tag", inp["base"].get("tag"))


def trace_quota(case, trace_stable):
    """run.py's per-case stable-trace quota for this case's index among the 18 cases."""
    ci = CASES.index(case)
    return trace_stable // len(CASES) + (1 if ci < trace_stable % len(CASES) else 0)


TRACE_SAMPLE_FIELDS = (
    "case",
    "sample_id",
    "shard",
    "kind",
    "label",
    "reason",
    "trace_stored",
    "candidate_rank",
    "quota_rank",
)


def trace_sample_rows(case, case_dir):
    """The trace sample of one finalized case, from run.py's and gen.py's json files.

    case_dir is run.py's case folder (OUT/CASE with info.json, trace_sample.json and the shard
    jsons). One row per unstable event and per event gen.py wrote a trace for, with reason
      unstable      every unstable event (gen.py traces all of them)
      stable_quota  the case's stable quota: run.py's trace_sample.json stable_selected, i.e. the
                    lowest-ranked stable candidates (quota_rank 0, 1, ...)
      extra         a stable candidate gen.py traced beyond the quota; run.py pruned its trace
    trace_stored tells whether the final traces hold the event. candidate_rank is the rank in the
    case's seeded candidate order (info.json trace_candidates; None if not a candidate). The
    events column `traced` is gen.py's flag (unstable or extra or stable_quota) and stays True for
    the pruned extras.
    """
    d = Path(case_dir)
    ts = json.loads((d / "trace_sample.json").read_text())
    info = json.loads((d / "info.json").read_text())
    rank = {int(i): r for r, i in enumerate(info["trace_candidates"])}
    shard, traced = {}, set()
    for f in sorted(d.glob("*.json")):
        if f.name in ("info.json", "trace_sample.json"):
            continue
        j = json.loads(f.read_text())
        for i in j.get("kept_ids", []):
            shard[int(i)] = j["tag"]
        traced |= {int(i) for i in j.get("traced_ids", [])}
    unstable = sorted(int(i) for i in ts["unstable"])
    stored_unstable = {int(i) for i in ts["unstable_traced"]}
    quota = [int(i) for i in ts["stable_selected"]]
    extra = sorted(traced - set(unstable) - set(quota), key=lambda i: (rank.get(i, len(rank)), i))

    def row(i, label, reason, stored, q=None):
        tag = shard.get(i)
        return {
            "case": case,
            "sample_id": i,
            "shard": tag,
            "kind": None if tag is None else ("grid" if tag == "grid" else "random"),
            "label": label,
            "reason": reason,
            "trace_stored": stored,
            "candidate_rank": rank.get(i),
            "quota_rank": q,
        }

    return (
        [row(i, 1, "unstable", i in stored_unstable) for i in unstable]
        + [row(i, 0, "stable_quota", True, q) for q, i in enumerate(quota)]
        + [row(i, 0, "extra", False) for i in extra]
    )
