"""Merge the 18 finalized case outputs of one run into the Delta tables events, traces,
trace_sample and runs.

python merge_task.py --out-root /Volumes/.../outputs/v4dyn/RUN_NAME --catalog C --schema S
    --job-run-id ID --n N --shard S --grid-lines L --grid-clr C --trace-stable T --trace-seed SEED
    --trace-cand K --min-feasible F [--table-prefix ""] [--git-commit SHA] [--max-dyn-pct 5]

Reads OUT-ROOT/CASE/final/ (run.py's case folder after case_task.py --mode finalize; complete only
if OUT-ROOT/CASE/final.json exists). The run parameters are required and checked against every case
manifest (run.py's args; the stable-trace quota against the case's share of --trace-stable), so the
merge never relies on defaults that could differ from the simulated run.

events: one row per kept event (the compact events parquet: annotations, label, TSI, separation,
  clearing time, n_out_of_step, traced, snap as array<float>) plus case, shard, job_run_id.
  `traced` is gen.py's flag (a trace was written at simulation time); it stays True for the stable
  candidates run.py pruned afterwards, so it is not the stored sample: use trace_sample.
traces: one row per (event, time) of the stored reduced traces: values array<float> in the order of
  the case's channel_names (recorded per case in runs.case_headers).
trace_sample: one row per unstable or gen.py-traced event with the selection reason (unstable,
  stable_quota, extra), trace_stored, candidate_rank and quota_rank (common.trace_sample_rows). The
  stored traces must be exactly unstable + stable_quota, and the stable_quota rows plus the cases'
  shortfalls must add up to --trace-stable; otherwise the run fails.
runs: one manifest row per job run: hashes, versions, counts, failure categories, limits, problems.
Replay (--replay PATH, the parquet the sim tasks replayed, see case_task.py): the cases have no
  final/ folder; the verified shards under OUT-ROOT/CASE/shards (shards_ok sidecars) hold exactly
  the replayed events, all traced. events and traces are written as above for them; trace_sample
  carries the replay rows (reason, ranks of the original sample) with the replayed label and
  trace_stored; a replayed id that was not kept or not traced is a problem (status failed).
  traced_events counts the events flag `traced`; trace_stored_events counts the stored sample.
The failure limits are run.py's: per case and kind and over all cases, dynamic or numerical failures
above --max-dyn-pct FAIL with at least --min-feasible feasible events, otherwise FLAG. Rows of the
same job_run_id are replaced, so a rerun of this task is idempotent. Exits non-zero if the run
failed.
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    HERE = Path(__file__).resolve().parent
except NameError:  # pragma: no cover
    HERE = Path(os.getcwd())
sys.path.insert(0, str(HERE))
from common import CASES, TRACE_SAMPLE_FIELDS, log, trace_quota, trace_sample_rows

CATEGORIES = ("infeasible", "island", "dynamic", "numerical_failure")


def limit_problems(h, k, c, max_pct, min_feasible):
    """Copy of events_v2/run.py limit_problems (run.py imports the simulation stack at import)."""
    out = []
    for key, what in (("dyn_fail_pct", "dynamic"), ("num_fail_pct", "numerical")):
        p = c.get(key)
        if p is None or p <= max_pct:
            continue
        tag = "FAIL" if (c.get("feasible") or 0) >= min_feasible else "FLAG"
        out.append(
            f"{tag} {h} {k}: {what} failures {p}% of {c.get('feasible')} feasible > {max_pct}%"
            + ("" if tag == "FAIL" else f" (below {min_feasible} feasible: not judged)")
        )
    return out


def events_table(t, case, shard, headers):
    """The compact events parquet table as an events-table slice (snap as a list column)."""
    hdr = json.loads(t.schema.metadata[b"tranpy"])
    headers.setdefault(case, {}).update(
        events={
            k: hdr[k]
            for k in (
                "format",
                "version",
                "label_names",
                "gen_map",
                "snap_instants",
                "snap_vars",
                "snap_columns",
            )
        }
    )
    snap = t["snap"].combine_chunks()
    t = t.drop_columns(["snap"]).append_column(
        "snap",
        pa.ListArray.from_arrays(
            pa.array(np.arange(len(snap) + 1, dtype=np.int32) * snap.type.list_size),
            snap.values,
        ),
    )
    t = t.append_column("case", pa.array([case] * len(t))).append_column(
        "shard", pa.array([shard] * len(t))
    )
    return t.replace_schema_metadata(None)


def traces_table(t, case, shard, headers):
    """The compact traces parquet table as a traces-table slice (values as a list column)."""
    hdr = json.loads(t.schema.metadata[b"tranpy"])
    ch = hdr["channel_names"]
    headers.setdefault(case, {}).update(traces={"channel_names": ch, "grid": hdr["grid"]})
    X = np.column_stack([t[c].to_numpy() for c in ch]).astype(np.float32)
    vals = pa.FixedSizeListArray.from_arrays(pa.array(X.ravel()), len(ch))
    vals = pa.ListArray.from_arrays(
        pa.array(np.arange(len(t) + 1, dtype=np.int32) * len(ch)), vals.values
    )
    return pa.table(
        {
            "sample_id": t["sample_id"],
            "time": t["time"],
            "values": vals,
            "case": pa.array([case] * len(t)),
            "shard": pa.array([shard] * len(t)),
        }
    )


TS_TYPES = {
    "sample_id": pa.int64(),
    "label": pa.int32(),
    "trace_stored": pa.bool_(),
    "candidate_rank": pa.int32(),
    "quota_rank": pa.int32(),
}


def write_trace_sample(ts_rows, stage, case, job_run_id):
    cols = {k: [x[k] for x in ts_rows] for k in TRACE_SAMPLE_FIELDS}
    tsr = pa.table({k: pa.array(v, TS_TYPES.get(k, pa.string())) for k, v in cols.items()})
    tsr = tsr.append_column("job_run_id", pa.array([job_run_id] * len(tsr)))
    pq.write_table(tsr, stage / "trace_sample" / f"{case}.parquet", compression="zstd")


def replay_cases(A, out_root, stage, problems, tot, cases, headers, shard_sha):
    """Collect the replayed shards (case_task.py sim items with --replay) of every case."""
    import pandas as pd

    R = pd.read_parquet(A.replay)
    R["sample_id"] = R["sample_id"].astype("int64")
    rsha = hashlib.sha256(Path(A.replay).read_bytes()).hexdigest()
    log("replay", A.replay, "sha256", rsha, "rows", len(R))
    tot.update(replay_rows=len(R), replay_missing=0, replay_label_changed=0)
    for case in CASES:
        rc = R[R["case"] == case]
        if rc.empty:
            continue
        C = out_root / case
        ev, tr, kept, labels = [], [], set(), {}
        for shard in sorted(set(rc["shard"].astype(str))):
            ok = C / "shards_ok" / f"{shard}.json"
            if not ok.is_file():
                problems.append(f"{case}/{shard}: replayed shard not on the volume")
                continue
            S = json.loads(ok.read_text())
            for f, want in S["files"].items():
                have = hashlib.sha256((C / "shards" / f).read_bytes()).hexdigest()
                if have != want:
                    problems.append(f"{case}/shards/{f}: sha256 differs from shards_ok")
                shard_sha[f"{case}/shards/{f}"] = have
                if f.endswith(".events.parquet"):
                    t = events_table(pq.read_table(C / "shards" / f), case, shard, headers)
                    ev.append(t)
                    labels.update(
                        zip(t["sample_id"].to_pylist(), t["label"].to_pylist(), strict=True)
                    )
                elif f.endswith(".traces.parquet"):
                    tr.append(traces_table(pq.read_table(C / "shards" / f), case, shard, headers))
            J = json.loads((C / "shards" / f"{shard}.json").read_text())
            kept |= {int(i) for i in J.get("kept_ids", [])}
            for cat, n in (J.get("failed_by") or {}).items():
                tot[cat] += n
            tot["requested"] += J.get("req", 0)
            tot["kept"] += J.get("kept", 0)
            tot["feasible"] += J.get("req", 0) - sum(
                (J.get("failed_by") or {}).get(c, 0) for c in ("infeasible", "island")
            )
        want_ids = {int(i) for i in rc["sample_id"]}
        missing = sorted(want_ids - kept)
        traced = {int(i) for x in tr for i in x["sample_id"].to_pylist()}
        untraced = sorted((want_ids & kept) - traced)
        if missing:
            problems.append(f"{case}: {len(missing)} replayed events not kept: {missing[:10]}")
        if untraced:
            problems.append(
                f"{case}: {len(untraced)} replayed events without trace: {untraced[:10]}"
            )
        extra = sorted(kept - want_ids)
        if extra:
            problems.append(
                f"{case}: {len(extra)} events kept that were not replayed: {extra[:10]}"
            )
        tot["replay_missing"] += len(missing)
        if ev:
            e = pa.concat_tables(ev)
            e = e.append_column("job_run_id", pa.array([A.job_run_id] * len(e)))
            pq.write_table(e, stage / "events" / f"{case}.parquet", compression="zstd")
            tot["events"] += len(e)
            tot["unstable"] += int(np.sum(e["label"].to_numpy()))
            tot["traced_events"] += int(np.sum(e["traced"].to_numpy()))
        if tr:
            r = pa.concat_tables(tr)
            r = r.append_column("job_run_id", pa.array([A.job_run_id] * len(r)))
            pq.write_table(r, stage / "traces" / f"{case}.parquet", compression="zstd")
            tot["trace_rows"] += len(r)
        ts_rows = []
        for x in rc.sort_values("sample_id").itertuples(index=False):
            i = int(x.sample_id)
            if i in labels and int(x.label) != int(labels[i]):
                tot["replay_label_changed"] += 1
            ts_rows.append(
                {
                    "case": case,
                    "sample_id": i,
                    "shard": str(x.shard),
                    "kind": None if pd.isna(x.kind) else str(x.kind),
                    "label": int(labels[i]) if i in labels else int(x.label),
                    "reason": str(x.reason),
                    "trace_stored": i in traced,
                    "candidate_rank": None if pd.isna(x.candidate_rank) else int(x.candidate_rank),
                    "quota_rank": None if pd.isna(x.quota_rank) else int(x.quota_rank),
                }
            )
        write_trace_sample(ts_rows, stage, case, A.job_run_id)
        tot["trace_stored_events"] += len(traced)
        tot["stable_quota"] += sum(r["reason"] == "stable_quota" for r in ts_rows)
        cases[case] = {
            "replay": {"rows": len(rc), "kept": len(want_ids & kept), "missing": len(missing)}
        }
    return rsha


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--out-root", required=True)
    a.add_argument(
        "--replay", default="", help="replay parquet (case_task.py --replay); shards only"
    )
    a.add_argument("--catalog", required=True)
    a.add_argument("--schema", required=True)
    a.add_argument("--job-run-id", required=True)
    a.add_argument("--table-prefix", default="")
    a.add_argument("--git-commit", default="")
    a.add_argument("--max-dyn-pct", type=float, default=5.0)
    a.add_argument("--min-feasible", type=int, required=True)
    a.add_argument("--trace-stable", type=int, required=True)
    a.add_argument("--n", type=int, required=True)
    a.add_argument("--shard", type=int, required=True)
    a.add_argument("--grid-lines", type=int, required=True)
    a.add_argument("--grid-clr", required=True)
    a.add_argument("--trace-seed", type=int, required=True)
    a.add_argument("--trace-cand", type=int, required=True)
    A = a.parse_args()
    want = {
        "n": A.n,
        "shard": A.shard,
        "grid_lines": A.grid_lines,
        "grid_clr": A.grid_clr,
        "min_feasible": A.min_feasible,
        "max_dyn_pct": A.max_dyn_pct,
        "trace_seed": A.trace_seed,
        "trace_cand": A.trace_cand,
        "store": "compact",
        "settings": "v4g",
        "neighbours": "ce_entsoe",
    }
    out_root = Path(A.out_root)
    stage = out_root / "_delta_stage"
    (stage / "events").mkdir(parents=True, exist_ok=True)
    (stage / "traces").mkdir(parents=True, exist_ok=True)
    (stage / "trace_sample").mkdir(parents=True, exist_ok=True)
    problems, flags, cases, headers, shard_sha = [], [], {}, {}, {}
    tot = dict.fromkeys(CATEGORIES, 0)
    tot.update(requested=0, kept=0, feasible=0, events=0, unstable=0, traced_events=0, trace_rows=0)
    tot.update(trace_stored_events=0, stable_quota=0, stable_extra=0, stable_shortfall=0)
    dbx = None
    rsha = None
    if A.replay:
        rsha = replay_cases(A, out_root, stage, problems, tot, cases, headers, shard_sha)
        st = sorted(out_root.glob("staging_slot*.json"))
        if st:
            d = json.loads(st[0].read_text())
            dbx = {"python": d.get("python"), "versions": d.get("versions"), "staging": d}
    for case in CASES if not A.replay else []:
        F = out_root / case / "final"
        mf = F / "run_manifest.json"
        if not (out_root / case / "final.json").exists() or not mf.exists():
            problems.append(f"{case}: not finalized (no {case}/final.json)")
            continue
        M = json.loads(mf.read_text())
        D = json.loads((F / "dbx_case.json").read_text()) if (F / "dbx_case.json").exists() else {}
        diff = {k: (M["args"].get(k), v) for k, v in want.items() if M["args"].get(k) != v}
        if M["args"].get("trace_stable") != trace_quota(case, A.trace_stable):
            diff["trace_stable (case quota)"] = (
                M["args"].get("trace_stable"),
                trace_quota(case, A.trace_stable),
            )
        if diff:
            problems.append(
                f"{case}: run arguments differ from the job parameters (manifest, job): {diff}"
            )
        dbx = dbx or D
        H = M["hours"].get(case, {})
        rec = {
            "run_py_run_id": M["run_id"],
            "status": M.get("status"),
            "run_py_problems": M.get("problems"),
            "exit": D.get("exit"),
            "sec": H.get("sec"),
            "trace_sample": H.get("trace_sample"),
            "quota": D.get("quota"),
        }
        if H.get("status") != "events_done":
            problems.append(f"{case}: {H.get('status', 'not run')}")
        for k in ("random", "grid"):
            c = H.get(k)
            if not c:
                continue
            rec[k] = {
                x: c[x]
                for x in (
                    "planned",
                    "requested",
                    "kept",
                    "failed_by",
                    "not_run",
                    "unaccounted",
                    "shortfall",
                    "feasible",
                    "dyn_fail_pct",
                    "num_fail_pct",
                    "failed_ids",
                )
            }
            for x in CATEGORIES:
                tot[x] += c["failed_by"].get(x, 0)
            tot["requested"] += c["requested"]
            tot["kept"] += c["kept"]
            tot["feasible"] += c["feasible"]
            if c["unaccounted"] or c["not_run"]:
                problems.append(
                    f"{case} {k}: {c['not_run']} not run, {c['unaccounted']} unaccounted"
                )
            for msg in limit_problems(case, k, c, A.max_dyn_pct, A.min_feasible):
                (problems if msg.startswith("FAIL") else flags).append(msg)
        cases[case] = rec
        ev, tr = [], []
        for s in M.get("shards", []):
            # run.py lists shards relative to its --out, which is the case folder
            # OUT-ROOT/CASE/final
            f = F / s
            if not f.is_file():
                problems.append(f"{case}/{s}: shard listed in the manifest is missing")
                continue
            have = hashlib.sha256(f.read_bytes()).hexdigest()
            if have != M["shard_sha256"][s]:
                problems.append(f"{s}: sha256 differs from the run manifest")
            shard_sha[f"{case}/final/{s}"] = have
            t = pq.read_table(f)
            shard = Path(s).name.split(".")[0]
            if s.endswith(".events.parquet"):
                ev.append(events_table(t, case, shard, headers))
            else:
                tr.append(traces_table(t, case, shard, headers))
        if ev:
            e = pa.concat_tables(ev)
            e = e.append_column("job_run_id", pa.array([A.job_run_id] * len(e)))
            pq.write_table(e, stage / "events" / f"{case}.parquet", compression="zstd")
            tot["events"] += len(e)
            tot["unstable"] += int(np.sum(e["label"].to_numpy()))
            tot["traced_events"] += int(np.sum(e["traced"].to_numpy()))
        if tr:
            r = pa.concat_tables(tr)
            r = r.append_column("job_run_id", pa.array([A.job_run_id] * len(r)))
            pq.write_table(r, stage / "traces" / f"{case}.parquet", compression="zstd")
            tot["trace_rows"] += len(r)
        # the stored trace sample: exactly the unstable events and the case's stable quota
        ts_rows = trace_sample_rows(case, F / case)
        stored = {(x["sample_id"]) for x in ts_rows if x["trace_stored"]}
        have = {int(i) for x in tr for i in x["sample_id"].to_pylist()}
        if have != stored:
            problems.append(
                f"{case}: stored traces differ from the trace sample: "
                f"{len(have - stored)} unexpected, {len(stored - have)} missing"
            )
        quota_n = sum(x["reason"] == "stable_quota" for x in ts_rows)
        short = (H.get("trace_sample") or {}).get("shortfall", 0) or 0
        if quota_n + short != trace_quota(case, A.trace_stable):
            problems.append(
                f"{case}: {quota_n} stable traces + shortfall {short} "
                f"!= quota {trace_quota(case, A.trace_stable)}"
            )
        tot["trace_stored_events"] += len(stored)
        tot["stable_quota"] += quota_n
        tot["stable_extra"] += sum(x["reason"] == "extra" for x in ts_rows)
        tot["stable_shortfall"] += short
        write_trace_sample(ts_rows, stage, case, A.job_run_id)
    if cases and not A.replay and tot["stable_quota"] + tot["stable_shortfall"] != A.trace_stable:
        problems.append(
            f"stable traces {tot['stable_quota']} + shortfall {tot['stable_shortfall']} "
            f"!= --trace-stable {A.trace_stable}"
        )
    feas = tot["feasible"]
    totals = {
        "feasible": feas,
        "dyn_fail_pct": round(100 * tot["dynamic"] / feas, 2) if feas else None,
        "num_fail_pct": round(100 * tot["numerical_failure"] / feas, 2) if feas else None,
    }
    for msg in limit_problems("all", "total", totals, A.max_dyn_pct, A.min_feasible):
        (problems if msg.startswith("FAIL") else flags).append(msg)
    h = hashlib.sha256()
    for s in sorted(shard_sha):  # keys are CASE/final/CASE/shard, relative to OUT-ROOT
        h.update((out_root / s).read_bytes())
    stable_sel = sum(
        (c.get("trace_sample") or {}).get("stable_selected", 0) for c in cases.values()
    )
    status = "ok" if not problems else "failed"
    log("status", status, "problems", problems, "flags", flags, "totals", json.dumps(tot))

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    fq = lambda n: f"`{A.catalog}`.`{A.schema}`.`{A.table_prefix}{n}`"  # noqa: E731
    rid = A.job_run_id.replace("'", "")
    for name in ("events", "traces", "trace_sample"):
        files = sorted(str(p) for p in (stage / name).glob("*.parquet"))
        if not files:
            continue
        df = spark.read.parquet(*files)
        if spark.catalog.tableExists(f"{A.catalog}.{A.schema}.{A.table_prefix}{name}"):
            spark.sql(f"DELETE FROM {fq(name)} WHERE job_run_id = '{rid}'")
        df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(fq(name))
        log("wrote", fq(name), df.count(), "rows")
    row = {
        "job_run_id": A.job_run_id,
        "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "table_prefix": A.table_prefix,
        "status": status,
        "git_commit": A.git_commit,
        "python": (dbx or {}).get("python"),
        "versions": json.dumps((dbx or {}).get("versions")),
        "frozen_model_sha256": json.dumps(((dbx or {}).get("staging") or {}).get("frozen_model")),
        "frozen_dyn_sha256": json.dumps(((dbx or {}).get("staging") or {}).get("frozen_dyn")),
        "staged_sha256": json.dumps(((dbx or {}).get("staging") or {}).get("staged_sha256")),
        "args": json.dumps(
            dict(
                argv=(dbx or {}).get("argv"),
                trace_stable_total=A.trace_stable,
                out_root=str(out_root),
                replay=A.replay or None,
                replay_sha256=rsha,
                replay_rows=tot.get("replay_rows"),
                replay_missing=tot.get("replay_missing"),
                replay_label_changed=tot.get("replay_label_changed"),
                **want,
            )
        ),
        "n_cases": len(cases),
        "requested": tot["requested"],
        "kept": tot["kept"],
        "feasible": feas,
        "events": tot["events"],
        "unstable": tot["unstable"],
        "traced_events": tot["traced_events"],
        "trace_rows": tot["trace_rows"],
        "stable_traced": stable_sel,
        "trace_stored_events": tot["trace_stored_events"],
        "stable_trace_extra_pruned": tot["stable_extra"],
        "failed_infeasible": tot["infeasible"],
        "failed_island": tot["island"],
        "failed_dynamic": tot["dynamic"],
        "failed_numerical": tot["numerical_failure"],
        "dyn_fail_pct_total": totals["dyn_fail_pct"],
        "num_fail_pct_total": totals["num_fail_pct"],
        "problems": json.dumps(problems),
        "flags": json.dumps(flags),
        "cases": json.dumps(cases, default=str),
        "case_headers": json.dumps(headers),
        "shard_sha256": json.dumps(shard_sha),
        "dataset_sha256": h.hexdigest(),
    }
    (out_root / "run_manifest_merged.json").write_text(json.dumps(row, indent=1, default=str))
    import pandas as pd

    rdf = spark.createDataFrame(pd.DataFrame([row]))
    if spark.catalog.tableExists(f"{A.catalog}.{A.schema}.runs"):
        spark.sql(f"DELETE FROM `{A.catalog}`.`{A.schema}`.`runs` WHERE job_run_id = '{rid}'")
    rdf.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(
        f"`{A.catalog}`.`{A.schema}`.`runs`"
    )
    log("wrote runs row", A.job_run_id, status)
    if problems:
        print("RUN FAILED:", *problems, sep="\n  ", flush=True)  # noqa: T201
        sys.exit(1)
    print(  # noqa: T201
        f"RUN OK {A.job_run_id}: events {tot['events']}, "
        f"dynamic failures {totals['dyn_fail_pct']}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
