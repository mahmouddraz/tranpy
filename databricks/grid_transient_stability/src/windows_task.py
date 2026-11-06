"""Accuracy versus assessment time on the re-traced fault events: features -> benchmark.

  features   merged traces of one re-trace run (volume: <traces>/_delta_stage/traces/CASE.parquet,
             gen maps from runs.case_headers), per-case machine tables (outputs volume, prep/) and
             the strata + line_pre feature set of one ML run (its features/ folder) -> per instant
             the window, window_line and sequence tables with inclusion weights (tranpy.ml_windows),
             written to WINDOWS_DIR/features with a manifest.
  benchmark  every (instant, input, candidate, protocol) of the config in parallel processes:
             tuned LightGBM on the window inputs, the Keras sequence networks (LSTM, TCN,
             transformer; class weights) on the sequences; one MLflow run each; Delta table
             tsa_windows (pooled metrics with 95 % bootstrap intervals, n_eval, n_excluded) and
             WINDOWS_DIR/windows_results.csv, windows_curve.csv, knees.json.

Run locally (no Spark, no MLflow) with --local and --gen-maps for a smoke test on a few cases.
All results are tagged provisional=true until the real-event validation of the dynamic model.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

KEYS = ["case", "sample_id"]
STRATA = ["hour", "tan_phi", "clearing_band", "kind", "event_type", "fault_line"]


def log(*a):
    print("[tranpy-windows]", *a, flush=True)  # noqa: T201


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def spark_session():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


def table(a, name):
    return f"{a.catalog}.{a.schema}.{a.table_prefix}{name}"


# ------------------------------------------------------------------------------------ config
def load_config(path):
    """(BenchmarkSpec of the window inputs, instants, sequence section) of the YAML."""
    import yaml

    from tranpy.ml_benchmark import BenchmarkSpec
    from tranpy.ml_windows import INPUTS, Instant

    data = yaml.safe_load(Path(path).read_text())
    instants = [Instant.parse(s) for s in data.pop("instants")]
    sequence = data.pop("sequence", None) or {}
    data["feature_sets"] = data.pop("inputs")
    unknown = sorted(set(data["feature_sets"]) - set(INPUTS))
    if unknown:
        raise SystemExit(f"unknown inputs {unknown}; use {INPUTS}")
    return BenchmarkSpec.from_mapping(data), instants, sequence


def _tuple_params(params):
    return {k: tuple(v) if isinstance(v, list) else v for k, v in dict(params).items()}


def sequence_candidates(sequence, instant, seed_spec):
    """(size, Candidate) of every sequence network of the config at ``instant``."""
    from tranpy.ml_benchmark import Candidate
    from tranpy_core.imbalance import ImbalanceSpec

    out = []
    if not sequence or instant.reference == "pre":
        return out
    imbalance = ImbalanceSpec(method=str(sequence.get("imbalance", "class_weight")))
    for protocol in sequence.get("protocols", seed_spec.protocols):
        for model, sizes in sequence["models"].items():
            for size, params in sizes.items():
                p = {**_tuple_params(params), "n_steps": instant.n_steps}
                out.append((str(size), Candidate(model, imbalance, p, str(protocol), None)))
    return out


# ---------------------------------------------------------------------------------- features
def _gen_maps(a):
    if a.gen_maps:
        raw = json.loads(Path(a.gen_maps).read_text())
        return {c: (v["gen_map"] if isinstance(v, dict) else v) for c, v in raw.items()}
    spark = spark_session()
    headers = json.loads(
        spark.table(f"{a.catalog}.{a.schema}.runs")
        .where(f"job_run_id = {int(a.traces_job_run_id)}")
        .select("case_headers")
        .first()[0]
    )
    return {c: h["events"]["gen_map"] for c, h in headers.items()}


def features(a):
    import numpy as np
    import pandas as pd

    from tranpy.ml_events import population_counts, read_case_machines
    from tranpy.ml_windows import (
        CaseTraces,
        load_merged_traces,
        window_tables,
        write_window_tables,
    )
    from tranpy_core.imbalance import inclusion_weights

    ml_features = Path(a.ml_features)
    strata = pd.read_parquet(ml_features / "strata.parquet")
    line = pd.read_parquet(ml_features / "line_pre.parquet")
    if len(line) != len(strata):
        raise SystemExit("strata.parquet and line_pre.parquet differ in length")
    line = pd.concat([strata[KEYS].reset_index(drop=True), line.reset_index(drop=True)], axis=1)
    if "fault_line" not in strata:
        strata = strata.assign(fault_line=line["fault_line"].to_numpy())
    strata = strata[[*KEYS, "label", *[c for c in STRATA if c in strata]]]
    population = population_counts(strata, by="case")
    gen_maps = _gen_maps(a)
    traces = Path(a.traces)
    files = sorted(traces.glob("*.parquet"))
    if a.only_cases:
        keep = set(a.only_cases.split(","))
        files = [f for f in files if f.stem in keep]
    if not files:
        raise SystemExit(f"no merged trace files in {traces}")
    cases = []
    for f in files:
        case = f.stem
        gen_map = gen_maps[case]
        ev = strata[strata["case"] == case]
        ev = ev.merge(line[[*KEYS, "clearing_time_s"]], on=KEYS)
        ds = load_merged_traces(f, gen_map, ev)
        prep = Path(a.prep) if a.prep else Path(a.outputs) / case / "prep"
        machines = read_case_machines(prep, case, gen_map).set_index("gen")
        gens = tuple(g for g, _, _ in gen_map)
        kinetic = machines.loc[list(gens), "kinetic_mws"].to_numpy(dtype=np.float64)
        cases.append(CaseTraces(case, ds, gens, kinetic))
        log(f"{case}: {ds.n_samples} traced events, {len(gens)} machines")
    traced = pd.concat(
        [
            pd.DataFrame({"case": ct.case, "sample_id": np.asarray(ct.data.sample_ids)})
            for ct in cases
        ]
    ).merge(strata[[*KEYS, "label"]], on=KEYS)
    w = inclusion_weights(traced["case"], traced["label"], population)
    weights = {
        (c, int(i)): float(x)
        for c, i, x in zip(traced["case"], traced["sample_id"], w, strict=True)
    }
    _, instants, sequence = load_config(a.config)
    if a.only_instants:
        keep = set(a.only_instants.split(","))
        instants = [i for i in instants if i.name in keep]
    tables = window_tables(
        cases,
        strata,
        instants=instants,
        line_features=line,
        weights=weights,
        sequences=bool(sequence),
    )
    out = Path(a.windows_dir) / "features"
    manifest = write_window_tables(tables, out)
    for name, m in manifest.items():
        log(
            f"{name}: {m['events']} events ({m['unstable']} unstable), {m['n_excluded']} excluded "
            f"({m['excluded_unstable']} unstable), inputs {m['inputs']}"
        )
    manifest = {
        "instants": manifest,
        "traces_job_run_id": a.traces_job_run_id,
        "dataset_job_run_id": a.dataset_job_run_id,
        "ml_features": str(ml_features),
        "cases": [ct.case for ct in cases],
        "traced_events": len(traced),
        "population": {f"{k[0]}|{k[1]}": v for k, v in population.items()},
        "files": {p.name: sha256(p) for p in sorted(out.glob("*.parquet"))},
        "config": Path(a.config).name,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    log("features written")


# --------------------------------------------------------------------------------- benchmark
def _experiment(a):
    import mlflow
    from databricks.sdk import WorkspaceClient

    user = WorkspaceClient().current_user.me().user_name
    folder = f"/Users/{user}/grid_transient_stability"
    WorkspaceClient().workspace.mkdirs(folder)
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(f"{folder}/{a.experiment}")


def _tags(a, manifest):
    return {
        "dataset_job_run_id": str(a.dataset_job_run_id),
        "traces_job_run_id": str(a.traces_job_run_id),
        "features_sha256": hashlib.sha256(
            json.dumps(manifest["files"], sort_keys=True).encode()
        ).hexdigest(),
        "git_commit": a.git_commit,
        "config": Path(a.config).name,
        "provisional": "true",
        "job_run_id": str(a.job_run_id),
        "benchmark": "windows",
    }


def _jobs(a, spec, instants, sequence):
    """(instant, input, size, candidate) in the order they are submitted (slow ones first)."""
    from tranpy.ml_benchmark import candidates

    only_models = set(a.only_models.split(",")) if a.only_models else None
    seq_jobs, tab_jobs = [], []
    for instant in instants:
        for size, cand in sequence_candidates(sequence, instant, spec):
            if only_models is None or cand.model in only_models:
                seq_jobs.append((instant.name, "sequence", size, cand))
        for inp in spec.feature_sets:
            if instant.reference == "pre" and inp == "sequence":
                continue
            for cand in candidates(spec):
                if only_models is None or cand.model in only_models:
                    tab_jobs.append((instant.name, inp, "", cand))
    return seq_jobs + tab_jobs


def benchmark(a):
    import pandas as pd

    from tranpy.early_prediction import find_knee_point
    from tranpy.ml_windows import curve_frame, evaluate_window_file

    spec, instants, sequence = load_config(a.config)
    if a.only_instants:
        keep = set(a.only_instants.split(","))
        instants = [i for i in instants if i.name in keep]
    feats = Path(a.windows_dir) / "features"
    manifest = json.loads((feats / "manifest.json").read_text())
    instants = [i for i in instants if i.name in manifest["instants"]]
    jobs = _jobs(a, spec, instants, sequence)
    log(f"{len(jobs)} candidates on {a.workers} workers")
    tags = _tags(a, manifest)
    mlflow = None
    parent_id = ""
    if not a.local:
        import mlflow

        _experiment(a)
        parent = mlflow.start_run(run_name=f"windows {a.job_run_id}")
        mlflow.set_tags(tags)
        mlflow.log_artifact(a.config)
        mlflow.log_artifact(str(feats / "manifest.json"))
        parent_id = parent.info.run_id
    rows, strata_rows, failed = [], [], []
    t_start = time.perf_counter()
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as pool:
        futures = {
            pool.submit(
                evaluate_window_file,
                feats,
                inst,
                inp,
                cand,
                spec.seed,
                list(spec.report_by),
                n_boot=a.n_boot,
            ): size
            for inst, inp, size, cand in jobs
        }
        for k, fut in enumerate(as_completed(futures), 1):
            size = futures[fut]
            inst, inp, cand, result, error, dt = fut.result()
            label = f"{cand.model}[{size}]" if size else cand.model
            if result is None:
                failed.append(
                    {
                        "instant": inst,
                        "input": inp,
                        "model": label,
                        "protocol": cand.protocol,
                        "error": error,
                    }
                )
                log(f"[{k}/{len(jobs)}] {inst} {inp} {label} [{cand.protocol}] FAILED {error}")
                continue
            run_id = ""
            if mlflow is not None:
                from tranpy.ml_tracking import log_candidate

                run_id = log_candidate(
                    result, tags={**tags, "instant": inst, "input": inp, "size": size}
                )
            m = result.metrics()
            info = manifest["instants"][inst]
            log(
                f"[{k}/{len(jobs)}] {inst} {inp} {label} [{cand.protocol}] "
                f"AP={m['average_precision']:.3f} [{m['average_precision_lo']:.3f}, "
                f"{m['average_precision_hi']:.3f}] miss={m['miss_rate']:.3f} mcc={m['mcc']:.3f} "
                f"({dt:.0f} s{', tuned' if result.tuned else ''})"
            )
            rows.append(
                {
                    "instant": inst,
                    "reference": info["reference"],
                    "offset_s": info["offset_s"],
                    "n_steps": info["n_steps"],
                    "input": inp,
                    "model": label,
                    "size": size,
                    "candidate": cand.name,
                    "protocol": cand.protocol,
                    "imbalance_method": cand.imbalance.method,
                    "tuned": result.tuned,
                    "tuning_time_s": float(result.tuning["time_s"]) if result.tuned else 0.0,
                    "n_eval": info["events"],
                    "n_excluded": info["n_excluded"],
                    "excluded_unstable": info["excluded_unstable"],
                    "mlflow_run_id": run_id,
                    "seconds": dt,
                    **m,
                }
            )
            for col, tab in result.by_stratum.items():
                t = tab.drop(columns="confusion_matrix").reset_index(names="stratum")
                strata_rows.append(
                    t.assign(
                        instant=inst, input=inp, model=label, protocol=cand.protocol, stratum_by=col
                    )
                )
    if mlflow is not None:
        mlflow.log_text(json.dumps(failed, indent=1), "failed_candidates.json")
        mlflow.log_metric("candidates_failed", len(failed))
        mlflow.end_run()
    if not rows:
        raise SystemExit("every candidate failed")
    meta = {
        "benchmark_run_id": parent_id,
        "job_run_id": str(a.job_run_id),
        "dataset_job_run_id": str(a.dataset_job_run_id),
        "traces_job_run_id": str(a.traces_job_run_id),
        "provisional": True,
    }
    res = pd.DataFrame(rows).assign(**meta)
    out = Path(a.windows_dir)
    res.to_csv(out / "windows_results.csv", index=False)
    pd.concat(strata_rows, ignore_index=True).assign(**meta).to_csv(
        out / "windows_strata.csv", index=False
    )
    # the curve per (protocol, input, model) over the clearing-based instants, and its knee
    knees = {}
    curve = []
    for (protocol, inp, model), g in res.groupby(["protocol", "input", "model"]):
        c = curve_frame(g, spec.primary_metric)
        c = c.assign(protocol=protocol, input=inp)
        curve.append(c)
        for ref in ("clear", "inception"):
            sub = c[c["reference"] == ref]
            if len(sub) >= 3:
                knees[f"{protocol}|{inp}|{model}|{ref}"] = find_knee_point(
                    sub, model, spec.primary_metric
                )
    pd.concat(curve, ignore_index=True).to_csv(out / "windows_curve.csv", index=False)
    (out / "knees.json").write_text(json.dumps(knees, indent=1))
    if failed:
        (out / "failed.json").write_text(json.dumps(failed, indent=1))
    log(f"{len(res)} results, {len(failed)} failed, {(time.perf_counter() - t_start) / 3600:.2f} h")
    if not a.local:
        _write(a, "tsa_windows", res)
        _write(a, "tsa_windows_strata", pd.concat(strata_rows, ignore_index=True).assign(**meta))
    for inst in [i.name for i in instants]:
        best = res[(res["instant"] == inst) & (res["protocol"] == spec.protocols[0])]
        if len(best):
            top = best.sort_values(spec.primary_metric, ascending=False).iloc[0]
            log(
                f"best {inst} [{spec.protocols[0]}]: {top['input']} {top['model']} "
                f"AP={top['average_precision']:.3f} [{top['average_precision_lo']:.3f}, "
                f"{top['average_precision_hi']:.3f}] miss={top['miss_rate']:.3f}"
            )


def _write(a, name, frame):
    """Replace this job run's rows of Delta table ``name`` (append for other runs)."""
    spark = spark_session()
    full = table(a, name)
    df = spark.createDataFrame(frame)
    if spark.catalog.tableExists(full):
        spark.sql(f"DELETE FROM {full} WHERE job_run_id = '{a.job_run_id}'")
        df.write.mode("append").option("mergeSchema", "true").saveAsTable(full)
    else:
        df.write.saveAsTable(full)
    log(f"{full}: {len(frame)} rows")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", required=True, choices=["features", "benchmark"])
    p.add_argument("--catalog", default="")
    p.add_argument("--schema", default="")
    p.add_argument("--table-prefix", default="")
    p.add_argument("--traces", required=True, help="merged per-case trace files")
    p.add_argument("--traces-job-run-id", default="", help="re-trace run (gen maps)")
    p.add_argument("--gen-maps", default="", help="gen_maps.json (local)")
    p.add_argument("--dataset-job-run-id", required=True)
    p.add_argument(
        "--outputs", required=True, help="volume folder of the dataset run's cases (prep/)"
    )
    p.add_argument("--prep", default="", help="one flat prep folder for every case (local)")
    p.add_argument("--ml-features", required=True, help="features/ of an ML run")
    p.add_argument("--windows-dir", required=True, help="output folder of tables and results")
    p.add_argument("--config", required=True, help="windows YAML")
    p.add_argument("--experiment", default="tsa")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--only-cases", default="", help="comma list of cases (smoke)")
    p.add_argument("--only-instants", default="", help="comma list of instant names (smoke)")
    p.add_argument("--only-models", default="", help="comma list of model names (smoke)")
    p.add_argument("--n-boot", type=int, default=None, help="bootstrap resamples")
    p.add_argument("--local", action="store_true", help="no Spark, no MLflow (smoke)")
    p.add_argument("--git-commit", default="")
    p.add_argument("--job-run-id", default="local")
    a = p.parse_args()
    log(f"python {sys.version.split()[0]}, mode {a.mode}")
    {"features": features, "benchmark": benchmark}[a.mode](a)


if __name__ == "__main__":
    main()
