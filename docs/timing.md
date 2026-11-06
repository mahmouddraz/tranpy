# Wall-clock cost per event: time-domain simulation, SIME, machine learning

Measured 2026-11-06 on the v4.1 fault events (job run 738439695169033, 11,162 events) and the
re-traced subsample (run 588561814210793, 2,328 events). Raw numbers and scripts:
`~/Library/Caches/tranpy/work/ml_v41dyn/timing/` (`tds_databricks.json`, `tds_mac.json`,
`sime_timing_*.json`, `ml_timing.json`, `time_sime.py`, `time_ml.py`).

| Method | Needs as input | Per event (median / p90) | Hardware | Status |
|---|---|---|---|---|
| TDS, ANDES 2.0.0, full 6.6 s horizon, 10 ms step | case + scenario | 13.2 s (mean of 20 events, single core; no spread recorded per event) | Apple M3, 1 worker | measured |
| TDS, same, production run | case + scenario | 2.0 s / 3.3 s wall per event with 4 workers = 7.8 / 13.3 core-s; 9.3 core-s mean (28.9 core-h for 11,162 events) | Azure Standard_F8as_v7, 4 workers | measured per shard of 50 events, divided by events |
| SIME margin (Pavella OMIB, 5 gap candidates) | post-clearing trajectory of every machine | 5.5 ms / 9.4 ms (62 machines, 145 samples); 8.6 ms / 17.3 ms (146 machines) | Apple M3 | measured, 96 + 168 events |
| SIME total = TDS to the decision instant + margin | case + scenario (the trajectory must be simulated) | ≈ 1.4 s + 0.01 s ≈ 1.4 s (median); ≈ 3.4 s (p90, 1.44 + 2.47 s of simulated time) | Apple M3 scaling | **estimate**: 13.2 s × (fault 1.0 s + clearing 0.44 s + decision 0.22 s) / 6.6 s; the ANDES cost per simulated second is taken as uniform |
| ML, pre-fault (`tsa_pre` v4: XGBoost on `line_pre`, 33 features) | solved operating point + fault line/position | 0.93 ms / 2.0 ms (batch of 1); 33 µs per event in a batch of 1,000 | Apple M3 | measured, 20 repeats |
| ML, at clearing (`tsa_clear` v3: LightGBM on `system_line_clear`, 53 features) | clearing-instant snapshot + line descriptors | 1.3 ms / 2.1 ms (batch of 1); 14 µs per event in a batch of 1,000 | Apple M3 | measured, 20 repeats |
| Feature computation, `line_pre` | case workbook + solved OP | 0.70 s per case grid once (Excel read + π-model flows), then 0.54 ms per event (643 events of one case); 287 ms when one event is computed alone (pandas overhead) | Apple M3 | measured |

## Ratios (same machine, Apple M3, single core)

- TDS 13.2 s vs ML 0.93 ms (pre-fault, batch of 1): **~14,000×**; vs 33 µs in batches: ~400,000×.
  With the line-descriptor computation included (0.54 ms per event amortised over a case): ~9,000×.
- TDS 13.2 s vs SIME ≈ 1.4 s: ~9× — SIME's gain is only the shortened simulation horizon
  (decision after 0.22 s of post-clearing time at the median, 2.5 s at the p90); the margin itself
  costs 5–17 ms. SIME therefore remains a simulation-based method and cannot be used before the
  fault.
- SIME ≈ 1.4 s vs ML at clearing 1.3 ms: ~1,000×, but the ML at clearing also needs the clearing
  snapshot (fault 1.0 s + clearing 0.44 s of simulated time ≈ 2.9 s of ANDES on the M3, or a
  measurement in operation); the comparison that matters is "snapshot + ML" vs "snapshot +
  0.22 s more simulation + SIME".

## Caveats

- Hardware differs between the production TDS (F8as_v7, 4 processes, Python 3.12) and the
  local measurements (M3). The 20-event M3 run is the like-for-like number for the ratios; it
  agrees in order of magnitude with the F8 core-seconds (13.2 vs 7.8–13.3 core-s).
- The production shards hold 50 events each; the per-event TDS time is the shard time divided
  by its events, so per-event spread (stable vs unstable) is not observable. A least-squares
  split of shard time into stable and unstable events gives 5 core-s (stable) vs 41 core-s
  (unstable) with R² = 0.21 — too weak to quote; unstable events do not stop early in general
  (median `simulation_end_s` = 6.6 s for both labels; 6.07 s mean for unstable).
- The re-trace run (replay of 2,328 events in 233 small shards, 17 events each) cost 21 core-s
  per event because the 4 workers were underused and the per-shard setup (power flow,
  initialisation, separation checks) is amortised over fewer events; it is not a per-event TDS
  cost.
- The SIME total is an estimate (uniform ANDES cost per simulated second; the fault-on period
  with its reduced step is probably costlier than average, which would raise SIME's cost).
- ML times exclude model loading (once) and, for the at-clearing model, the simulation or
  measurement that produces the clearing snapshot.
- The `line_pre` single-event time (287 ms) is pandas group/merge overhead for a one-row frame,
  not computation; in a screening setting the descriptors are computed per operating point for
  all candidate lines at once (0.54 ms per line).
