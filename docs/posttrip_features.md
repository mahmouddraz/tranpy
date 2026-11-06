# Post-trip descriptors of the faulted line (v4.1, pre-fault task)

Status 2026-11-06: implemented, verified, benchmarked once on Databricks (run 1031368889100466,
1.63 node-hours). The leave-one-hour-out result did not move; the unseen-line and random
protocols improved slightly. The per-stratum comparison and the larger sweeps continue on the
lab machine (`docs/local_runs.md`, experiment `e1_boosting`).

## What was added (`tranpy.line_features.post_trip_table`)

All quantities come from the solved pre-fault operating point and the DC network with the
faulted line removed, so they are available before the fault. `X_CAP = 1e3` marks islanding.

| column | meaning |
|---|---|
| `line\|x_th_ends` | Thevenin reactance between the two line ends, line in service (DC inverse) |
| `line\|post_x_th_ends` | the same after the trip, Sherman–Morrison: X/(1 − X/x_k); capped if islanded |
| `line\|islanded` | 1 − PTDF_kk < 1e-6: the trip islands the far end |
| `line\|post_bus_sc_strength` | Σ 1/x of the remaining lines at the faulted bus |
| `line\|x_th_bus`, `line\|post_x_th_bus` | Thevenin reactance from the faulted bus to the machines (grounded through x'd on system base), before and after the trip |
| `line\|post_loading_max` | highest post-trip loading of the lines at the two ends (LODF re-routing of the pre-fault P), relative to the per-level 95th-percentile S |
| `line\|post_loading_rise_max` | largest relative loading increase among those lines |
| `line\|x_dist_380kv` | Σx along the shortest path from the faulted bus to the nearest 380 kV bus |

Checks on two real cases (1,251 buses, 1,760 branches, 1,015 candidates): a brute-force DC
recomputation with the line physically removed matches `post_x_th_ends` and `post_x_th_bus` to
5 decimals, LODF flows to 1.7e-3 pu; 82 candidates island their far end (all single-circuit),
no double circuit does; the table builds in < 0.5 s per case. `check_no_leakage` passes;
5 hand-value unit tests in `tests/tranpy/test_line_features.py`.

## Benchmark (pre-fault, tuned, 30 trials, seed 20261104, bootstrap 95 % CI)

Before = run 1086973316840813 (`line_pre` without post-trip columns); after = run
1031368889100466 (with them). Best candidate per cell.

| feature set | protocol | PR-AUC before | PR-AUC after | miss rate after |
|---|---|---|---|---|
| line_pre | hour (primary) | 0.781 [0.761, 0.800] | 0.781 [0.761, 0.798] | 0.264 |
| line_pre | line (unseen lines) | 0.849 | 0.854 | — |
| line_pre | random | 0.885 | 0.897 | — |
| system_line_pre | hour | — | 0.759 | — |
| system_line_pre | line | — | 0.874 | — |
| system_line_pre | random | — | 0.913 | — |

The selected hour-protocol model (`xgboost__class_weight`) was registered as `tsa_pre` v5 and
promoted by the MCC tie rule (0.671 vs 0.665; PR-AUC tied within the interval); it is the first
version saved in skops format. The `score` task failed only because this run had no
post-clearing feature set (`system_line_clear.parquet` missing) — expected for a pre-fault-only
config, no action needed.

## Reading

The post-trip physics does not change what the model can learn across unseen operating hours:
the hour-protocol interval is unchanged. It helps a little where the line is unseen (+0.005)
and under random splits (+0.012), i.e. it describes the lines better but does not add
information about the hour. This is consistent with the sampling-coverage diagnosis in
`docs/failure_analysis.md`: with one clearing time per line and nine hours, the remaining
pre-fault error is a coverage limit, which the v5 CCT bracketing (`docs/v5_sampling_plan.md`)
addresses directly.
