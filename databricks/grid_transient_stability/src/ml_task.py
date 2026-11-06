"""Transient-stability ML on the merged fault events: features -> benchmark -> register -> score.

  features   events (Delta, one dataset job run) + per-case machine tables and case grids
             (outputs volume, prep/: CASE.xlsx with CASE.op.pkl + CASE.buses.csv for the solved
             operating point, else the ANDES power flow) -> the feature sets of the config and
             the strata as parquet in ML_DIR, with a manifest (sha256, counts).
  benchmark  every (feature set, model, imbalance option, protocol) of the config in parallel
             processes, with nested Optuna tuning where the config's `tuning` says so; one
             MLflow run each (nested in a benchmark run); Delta tables tsa_benchmark (pooled
             metrics with 95 % bootstrap intervals <m>_lo/<m>_hi, the protocol and `tuned`),
             tsa_benchmark_strata, tsa_oof_predictions; ML_DIR/best.json.
  register   per assessment time (pre-fault, at clearing) the best candidate of the first
             protocol (tranpy.ml_benchmark.select_best), refitted on all events (tuned once
             more on all events when it was a tuned candidate), logged and registered as
             CATALOG.SCHEMA.tsa_<time>; alias challenger, and champion if it passes the gate
             (primary metric better beyond the interval, or tied with a lower miss rate).
  score      the champion of each registered model on all events -> Delta tsa_predictions
             (in-sample for the training events; the out-of-fold predictions are in
             tsa_oof_predictions).

All results are tagged provisional=true until the real-event validation of the dynamic model.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ASSESSMENT = {
    "pre": ("system_pre", "aligned_pre", "line_pre", "system_line_pre"),
    "clear": ("system_clear", "aligned_clear", "system_line_clear"),
}
KEYS = ["case", "sample_id", "shard"]
STRATA = ["hour", "tan_phi", "clearing_band", "kind", "event_type", "fault_line"]
LINE_STRATA = {"vn_kv": "line|vn_kv"}  # strata read off the line descriptors, when computed


def log(*a):
    print("[tranpy-ml]", *a, flush=True)  # noqa: T201


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def table(a, name):
    return f"{a.catalog}.{a.schema}.{a.table_prefix}{name}"


def spark_session():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


# ---------------------------------------------------------------------------------- features
def _feature_set_names(a):
    """Feature sets of the config (every set when no config is given)."""
    import yaml

    from tranpy.ml_benchmark import BenchmarkSpec

    if not a.config:
        return None
    return list(BenchmarkSpec.from_mapping(yaml.safe_load(Path(a.config).read_text())).feature_sets)


def _case_grids(a, cases, names):
    """CaseGrid per case for the sets that need one (tranpy.ml_events.GRID_FEATURE_SETS)."""
    from tranpy.ml_events import GRID_FEATURE_SETS, read_case_grid

    if names is not None and not set(names) & set(GRID_FEATURE_SETS):
        return None
    grids = {}
    for c in cases:
        prep = Path(a.outputs) / c / "prep"
        solved = (prep / f"{c}.op.pkl").exists() and (prep / f"{c}.buses.csv").exists()
        grids[c] = read_case_grid(prep, c)
        log(
            f"grid {c}: {len(grids[c].branches)} branches, {len(grids[c].fault_lines)} fault "
            f"candidates, voltages from {'op.pkl + buses.csv' if solved else 'ANDES power flow'}"
        )
    return grids


def features(a):
    from tranpy.ml_events import (
        complete_events,
        event_frame,
        expand_snap_arrays,
        feature_sets,
        read_case_machines,
    )

    spark = spark_session()
    run = a.dataset_job_run_id
    events = (
        spark.table(f"{a.catalog}.{a.schema}.{a.events_prefix}events")
        .where(f"job_run_id = {int(run)}")
        .toPandas()
    ).sort_values(KEYS, ignore_index=True)
    headers = json.loads(
        spark.table(f"{a.catalog}.{a.schema}.runs")
        .where(f"job_run_id = {int(run)}")
        .select("case_headers")
        .first()[0]
    )
    log(f"{len(events)} events of job run {run}, {events['case'].nunique()} cases")
    events = expand_snap_arrays(
        events, {c: h["events"]["snap_columns"] for c, h in headers.items()}
    )
    machines = {
        c: read_case_machines(Path(a.outputs) / c / "prep", c, headers[c]["events"]["gen_map"])
        for c in sorted(events["case"].unique())
    }
    names = _feature_set_names(a)
    grids = _case_grids(a, sorted(events["case"].unique()), names)
    frame = event_frame(events)
    sets = feature_sets(frame, machines, names=names, grids=grids)
    # complete-case: an event missing a snapshot (no output sample at the clearing instant)
    # is left out of every feature set, so all sets are compared on the same events
    keep = complete_events(sets)
    dropped = frame.loc[~keep, [*KEYS, "label", "clearing_time_s"]]
    log(f"complete-case: {int((~keep).sum())} events without a full snapshot left out")
    frame = frame.loc[keep].reset_index(drop=True)
    sets = {k: X.loc[keep].reset_index(drop=True) for k, X in sets.items()}
    out = Path(a.ml_dir) / "features"
    out.mkdir(parents=True, exist_ok=True)
    strata = frame[[*KEYS, "label", *STRATA]].copy()
    for name, col in LINE_STRATA.items():
        source = next((X for X in sets.values() if col in X), None)
        if source is not None:
            strata[name] = source[col].to_numpy()
    strata.to_parquet(out / "strata.parquet", index=False)
    manifest = {
        "dataset_job_run_id": run,
        "events": len(frame),
        "unstable": int(frame["label"].sum()),
        "excluded_incomplete": dropped.to_dict(orient="records"),
        "files": {"strata.parquet": sha256(out / "strata.parquet")},
        "columns": {},
        "machines": {c: len(m) for c, m in machines.items()},
        "config": Path(a.config).name if a.config else "",
        "grids": sorted(grids) if grids else [],
    }
    for name, X in sets.items():
        X.to_parquet(out / f"{name}.parquet", index=False)
        manifest["files"][f"{name}.parquet"] = sha256(out / f"{name}.parquet")
        manifest["columns"][name] = X.shape[1]
        log(f"{name}: {X.shape[1]} columns, NaN cells {int(X.isna().sum().sum())}")
        if X.isna().any().any():
            raise SystemExit(f"{name} has missing values")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    log("features written", json.dumps({k: manifest[k] for k in ("events", "unstable")}))


# --------------------------------------------------------------------------------- benchmark
def _load(ml_dir, feature_set):
    import pandas as pd

    base = Path(ml_dir) / "features"
    return pd.read_parquet(base / f"{feature_set}.parquet"), pd.read_parquet(
        base / "strata.parquet"
    )


def _experiment(a):
    import mlflow
    from databricks.sdk import WorkspaceClient

    user = WorkspaceClient().current_user.me().user_name
    folder = f"/Users/{user}/grid_transient_stability"
    WorkspaceClient().workspace.mkdirs(folder)
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(f"{folder}/{a.experiment}")


def _tags(a):
    manifest = json.loads((Path(a.ml_dir) / "features" / "manifest.json").read_text())
    return {
        "dataset_job_run_id": str(a.dataset_job_run_id),
        "dataset_table": f"{a.catalog}.{a.schema}.{a.events_prefix}events",
        "features_sha256": hashlib.sha256(
            json.dumps(manifest["files"], sort_keys=True).encode()
        ).hexdigest(),
        "git_commit": a.git_commit,
        "config": Path(a.config).name,
        "provisional": "true",
        "job_run_id": str(a.job_run_id),
    }


def benchmark(a):
    import mlflow
    import pandas as pd
    import yaml

    from tranpy.ml_benchmark import (
        BenchmarkSpec,
        candidates,
        evaluate_feature_file,
        select_best,
    )
    from tranpy.ml_tracking import log_candidate

    spec = BenchmarkSpec.from_mapping(yaml.safe_load(Path(a.config).read_text()))
    if a.only_models:
        keep = set(a.only_models.split(","))
        spec = BenchmarkSpec.from_mapping(
            {
                **yaml.safe_load(Path(a.config).read_text()),
                "models": [m for m in spec.models if m in keep],
                "extra": [{"model": c.model} for c in spec.extra if c.model in keep],
            }
        )
    sets = [s for s in spec.feature_sets if not a.only_sets or s in a.only_sets.split(",")]
    jobs = [(fs, c) for fs in sets for c in candidates(spec)]
    log(f"{len(jobs)} candidates on {a.workers} workers")
    _experiment(a)
    tags = _tags(a)
    strata = pd.read_parquet(Path(a.ml_dir) / "features" / "strata.parquet")
    rows, strata_rows, oof, failed = [], [], [], []
    with mlflow.start_run(run_name=f"benchmark {a.job_run_id}") as parent:
        mlflow.set_tags(tags)
        mlflow.log_artifact(a.config)
        mlflow.log_artifact(str(Path(a.ml_dir) / "features" / "manifest.json"))
        # spawn: fresh workers that import tranpy, instead of forking the driver (JVM, threads)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as pool:
            futures = [
                pool.submit(
                    evaluate_feature_file,
                    Path(a.ml_dir) / "features",
                    fs,
                    c,
                    spec.seed,
                    c.protocol,
                    list(spec.report_by),
                )
                for fs, c in jobs
            ]
            for k, fut in enumerate(as_completed(futures), 1):
                fs, cand, result, error, dt = fut.result()
                if result is None:
                    failed.append(
                        {
                            "feature_set": fs,
                            "candidate": cand.name,
                            "protocol": cand.protocol,
                            "error": error,
                        }
                    )
                    log(f"[{k}/{len(jobs)}] {fs} {cand.name} [{cand.protocol}] FAILED {error}")
                    continue
                run_id = log_candidate(result, tags=tags)
                m = result.metrics()
                log(
                    f"[{k}/{len(jobs)}] {fs} {cand.name} [{cand.protocol}] "
                    f"AP={m['average_precision']:.3f} "
                    f"[{m['average_precision_lo']:.3f}, {m['average_precision_hi']:.3f}] "
                    f"miss={m['miss_rate']:.3f} mcc={m['mcc']:.3f} "
                    f"({dt:.0f} s{', tuned' if result.tuned else ''})"
                )
                rows.append(
                    {
                        "feature_set": fs,
                        "candidate": cand.name,
                        "protocol": cand.protocol,
                        "model": cand.model,
                        "imbalance_method": cand.imbalance.method,
                        "threshold_metric": cand.imbalance.threshold or "fixed_0.5",
                        "tuned": result.tuned,
                        "tuning_time_s": float(result.tuning["time_s"]) if result.tuned else 0.0,
                        "mlflow_run_id": run_id,
                        **m,
                    }
                )
                for col, tab in result.by_stratum.items():
                    t = tab.drop(columns="confusion_matrix").reset_index(names="stratum")
                    strata_rows.append(
                        t.assign(
                            feature_set=fs,
                            candidate=cand.name,
                            protocol=cand.protocol,
                            stratum_by=col,
                        )
                    )
                oof.append(
                    strata[[*KEYS, "label"]].assign(
                        feature_set=fs,
                        candidate=cand.name,
                        protocol=cand.protocol,
                        p_unstable=result.oof_score,
                        predicted_label=result.oof_pred,
                    )
                )
        mlflow.log_text(json.dumps(failed, indent=1), "failed_candidates.json")
        mlflow.log_metric("candidates_failed", len(failed))
        parent_id = parent.info.run_id
    if not rows:
        raise SystemExit("every candidate failed")
    meta = {
        "benchmark_run_id": parent_id,
        "job_run_id": str(a.job_run_id),
        "dataset_job_run_id": str(a.dataset_job_run_id),
        "provisional": True,
    }
    res = pd.DataFrame(rows).assign(**meta)
    _write(a, "tsa_benchmark", res)
    _write(a, "tsa_benchmark_strata", pd.concat(strata_rows, ignore_index=True).assign(**meta))
    _write(a, "tsa_oof_predictions", pd.concat(oof, ignore_index=True).assign(**meta))
    best = {}
    for time_, sets_ in ASSESSMENT.items():
        # selection on the first protocol only (the others are reported, not compared across)
        pool_ = res[res["feature_set"].isin(sets_) & (res["protocol"] == spec.protocols[0])]
        if pool_.empty:
            continue
        rows_ = pool_.to_dict(orient="records")
        top = pool_.iloc[select_best(rows_, primary=spec.primary_metric, tie_metric="miss_rate")]
        best[time_] = {
            "feature_set": top["feature_set"],
            "candidate": top["candidate"],
            "protocol": top["protocol"],
            "tuned": bool(top["tuned"]),
            "mlflow_run_id": top["mlflow_run_id"],
            "metrics": {k: float(top[k]) for k in res.columns if k in m},
        }
        log(
            f"best {time_}: {top['feature_set']} {top['candidate']} [{top['protocol']}] "
            f"{spec.primary_metric}={top[spec.primary_metric]:.4f} "
            f"[{top[spec.primary_metric + '_lo']:.4f}, {top[spec.primary_metric + '_hi']:.4f}] "
            f"miss={top['miss_rate']:.4f}"
        )
    (Path(a.ml_dir) / "best.json").write_text(json.dumps(best, indent=1))
    if failed:
        log(
            f"{len(failed)} candidates failed "
            f"(see failed_candidates.json in MLflow run {parent_id})"
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


# ---------------------------------------------------------------------------------- register
def register(a):
    import mlflow
    import yaml

    from tranpy.ml_benchmark import BenchmarkSpec, candidates, tuning_for
    from tranpy.ml_tracking import register_model_version
    from tranpy.ml_tuning import tune_params

    spec = BenchmarkSpec.from_mapping(yaml.safe_load(Path(a.config).read_text()))
    by_name = {c.name: c for c in candidates(spec)}
    best = json.loads((Path(a.ml_dir) / "best.json").read_text())
    _experiment(a)
    tags = _tags(a)
    requirements = [r for r in a.pip_requirements.split(",") if r]
    out = {}
    for time_, b in best.items():
        X, strata = _load(a.ml_dir, b["feature_set"])
        cand = by_name[b["candidate"]]
        with mlflow.start_run(run_name=f"register tsa_{time_} {a.job_run_id}"):
            mlflow.set_tags(
                {
                    **tags,
                    "assessment": time_,
                    "candidate": cand.name,
                    "feature_set": b["feature_set"],
                    "cv_run_id": b["mlflow_run_id"],
                }
            )
            y = strata["label"].to_numpy()
            params = {}
            tuning = tuning_for(cand, b["feature_set"]) if b.get("tuned") else None
            if tuning is not None:
                # the final model is tuned the same way on all events (inner folds by hour)
                study = tune_params(
                    cand.model,
                    lambda p, c=cand: c.build_unthresholded(spec.seed, **p),
                    X.to_numpy(),
                    y,
                    spec=tuning,
                    seed=spec.seed,
                    groups=strata["hour"].astype(str).to_numpy(),
                )
                params = study.best_params
                mlflow.log_params({f"tuned_{k}": v for k, v in params.items()})
                mlflow.log_metric("tuning_best_value", study.best_value)
                mlflow.log_text(study.trials.to_csv(index=False), "tuning_trials.csv")
                log(f"tsa_{time_}: tuned on all events, {tuning.metric}={study.best_value:.4f}")
            model = cand.build(spec.seed, **params).fit(X, y)
            threshold = float(getattr(model, "best_threshold_", 0.5))
            mlflow.log_param("decision_threshold", threshold)
            mlflow.log_param("tuned", tuning is not None)
            res = register_model_version(
                model,
                X,
                registered_name=table(a, f"tsa_{time_}"),
                metrics=b["metrics"],
                primary_metric=spec.primary_metric,
                tags={
                    **tags,
                    "assessment": time_,
                    "candidate": cand.name,
                    "feature_set": b["feature_set"],
                    "decision_threshold": str(threshold),
                    "tuned": str(tuning is not None).lower(),
                },
                pip_requirements=requirements or None,
            )
        out[time_] = res
        log(f"tsa_{time_}: version {res['version']} promoted={res['promoted']} ({res['reason']})")
    (Path(a.ml_dir) / "registered.json").write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------- score
def score(a):
    import mlflow
    import pandas as pd

    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.MlflowClient()
    frames = []
    for time_ in ASSESSMENT:
        name = table(a, f"tsa_{time_}")
        try:
            version = client.get_model_version_by_alias(name, "champion")
        except Exception as exc:
            log(f"{name}: no champion ({type(exc).__name__}), skipped")
            continue
        fs = version.tags["feature_set"]
        X, strata = _load(a.ml_dir, fs)
        model = mlflow.sklearn.load_model(f"models:/{name}@champion")
        frames.append(
            strata[[*KEYS, "label"]].assign(
                model_name=name,
                model_version=str(version.version),
                feature_set=fs,
                p_unstable=model.predict_proba(X)[:, 1],
                predicted_label=model.predict(X),
                decision_threshold=float(version.tags.get("decision_threshold", "0.5")),
                in_sample=True,
                provisional=True,
                dataset_job_run_id=str(a.dataset_job_run_id),
                job_run_id=str(a.job_run_id),
            )
        )
        log(f"{name} v{version.version}: scored {len(X)} events")
    if frames:
        _write(a, "tsa_predictions", pd.concat(frames, ignore_index=True))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", required=True, choices=["features", "benchmark", "register", "score"])
    p.add_argument("--catalog", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--table-prefix", default="", help="prefix of the tables written")
    p.add_argument("--events-prefix", default="", help="prefix of the events table read")
    p.add_argument("--dataset-job-run-id", required=True)
    p.add_argument("--outputs", required=True, help="volume folder of the dataset run's cases")
    p.add_argument("--ml-dir", required=True, help="volume folder for features and selections")
    p.add_argument(
        "--config", default="", help="benchmark YAML (feature sets, benchmark and register)"
    )
    p.add_argument("--experiment", default="tsa")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--only-models", default="", help="comma list: restrict the config (smoke)")
    p.add_argument("--only-sets", default="", help="comma list: restrict the feature sets (smoke)")
    p.add_argument("--git-commit", default="")
    p.add_argument("--job-run-id", default="local")
    p.add_argument("--pip-requirements", default="")
    a = p.parse_args()
    log(f"python {sys.version.split()[0]}, mode {a.mode}")
    {"features": features, "benchmark": benchmark, "register": register, "score": score}[a.mode](a)


if __name__ == "__main__":
    main()
