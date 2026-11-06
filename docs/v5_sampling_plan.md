# v5 event set: sampling plan (draft for decision, 2026-11-06)

Status: proposal. Nothing here is built or launched. v4.1 (job run 738439695169033, frozen inputs
`inputs/v41dyn/`, SHA256SUMS checked by every task) stays untouched and remains the dataset of
all current results; v5 is a new input set and new table prefix. To be decided together with the
May 2021 Gridradar check and the queued generator fixes, so that one regeneration covers all three.

## 1. What v5 must solve

Pre-fault prediction on v4.1 (tuned LightGBM/XGBoost, `line_pre` descriptors, run 1086973316840813):

| protocol | PR-AUC [95 % CI] |
|---|---|
| leave-one-hour-out (primary) | 0.786 [0.767, 0.803] |
| unseen faulted line | 0.849 [0.833, 0.863] |
| stratified random | 0.885 [0.872, 0.898] |

The model generalises across lines but not across operating points: with 9 operating hours it
trains on 8 OPs and is asked to extrapolate to a ninth. The gap (0.10 to the random split, non-
overlapping CIs) is an information limit of the sample, not of the model: every family reached
the same level. More operating hours with fewer events each is the lever; more events per hour
is not (the random split already shows what in-sample OPs give).

## 2. Design options and cost

Cost basis (v4.1 on one Standard_F8as_v7, 2 slots x 4 workers): 7.4 s per event per core,
i.e. 0.93 node-seconds per event; +15 % for prep/finalize/merge and retries -> 1.06 node-s per
event. Prep is ~0.4 min per case. Measured check: 11,166 events took 196 min wall on one node
(DATASET_v4dyn_full.md). Price per node-hour: P_F8 (Azure pay-as-you-go eastus Standard_F8as_v7
+ Databricks jobs-compute DBUs; look up at decision time, left as a variable).

| option | cases | events | node-hours (sim + overhead) | wall, one F8 (quota 10 vCPU) | wall, two F8 (quota raised to 16) | cost |
|---|---|---|---|---|---|---|
| (a) 60 hours x ~190 events | 60 (one tan phi) or 120 (tan 0.1/0.2) | 11,400 | 3.4 (+0.4 prep) | ~4 h | ~2 h | 3.8 x P_F8 |
| (b) 100 hours x ~110 events | 100 / 200 | 11,000 | 3.3 (+0.7 prep) | ~4 h | ~2 h | 4.0 x P_F8 |
| (c) = (a) + CCT sweep, 50 lines | 60 | 11,400 + 24,000 (50 lines x 60 cases x 8 clearing times) | 10.5 | ~11 h | ~5.5 h | 10.9 x P_F8 |
| (c') = (a) + CCT by bisection, 50 lines | 60 | 11,400 + ~21,000 (7 runs per (case, line)) | 9.6 | ~10 h | ~5 h | 10.0 x P_F8 |
| tracing all events (any option) | — | — | +0 (same simulation; storage only) | — | — | ~0.4 MB per event compact -> 4.6 GB for 11,400 |

Doubling the cases with both tan phi variants doubles the events of (a)/(b) if the per-case
count is kept; the recommendation is one tan phi per hour (drawn 0.1/0.2 at random per hour) so
that the hour count, not the variant, carries the OP diversity. The 60-hour run is cheaper in
node-hours than v4.1 was. (c) adds CCT labels per (case, line) on 50 lines: an established
regression target (CCT) that uses every event of a line jointly and gives the energy methods a
per-line critical energy on 3,000 (case, line) pairs instead of 108.

## 3. Hour selection

v4.1 hours (ENTSO-E 2021 national hourly generation and load, research-only data; `tranpy.dispatch.select_hours`
plus `events_v2/run.py` extras): max wind, max solar, min load, max load, median load, winter peak on
another day, summer peak, lightest night hour, Baltic import at low load. These are extremes of single
variables; the OP space between them is empty.

v5 rule (extension of the same data, no new sources): stratify the 8,760 hours of 2021 in the space
(load, wind, solar, net import, synchronous share = thermal + hydro + nuclear / load) and draw hours per
cell: load tercile x wind tercile x solar (day/night) x import sign = 36 cells, 1-2 hours per cell plus
the 9 v4.1 extremes -> 60 hours (option a) or 2-3 per cell -> 100 (option b). Draw with a fixed seed;
reject hours whose power flow does not converge or violates the voltage limits; record every rejection.
The neighbour inertia is taken for the same hour from `neighbour_inertia_entsoe.csv` (hourly ENTSO-E
generation of the neighbours), so the hour must exist there (Turkey and Ukraine missing, as in v4.1).

Already supported by the pipeline: `dispatch_hour` and `mk.py` build a case for any hour with published
values; the job takes `--hours` as a list; the merge/feature code keys everything by case name; the
ML `STRATA` column `hour` and the hour protocol generalise to any number of hours.
New code needed: the stratified hour selection (one function with tests, ~100 lines), batch case
building for 60-100 cases with the init and flat-run checks of FROZEN_DYN_v41 applied to every case
(the check exists, only the loop is new), and the frozen-table update. Rules kept: published values
first, verify after each step, no requests to operators.

## 4. Sampling per hour (v4.1 rules, from gen.py / protocol.StressTestSampler, unchanged unless decided)

* 600 random events per case: loading factor ~ U(0.9, 1.1) with per-load noise N(1, 0.05), fault line
  uniform over the feasible non-transformer lines, three-phase fault at 1.0 s at a line terminal (0 or 1
  at random), 0.5 ohm fault reactance, clearing duration ~ U(0.05, 0.6) s, line tripped at clearing,
  5.0 s after the latest clearing (6.6 s horizon), re-dispatch via the case power flow.
* Grid sweep: 6 random lines x 8 clearing times (0.05-0.6 s) per case, nominal loading.
* Kept if feasible (power flow converges, min_feasible 50 per case) and no numerical failure
  (max_dyn_pct 5 %); v4.1 lost 418 infeasible + 80 numerical of 11,664.
* v5 per-case count ~190 (a) or ~110 (b): same rules, smaller `n`; the sweep shrinks to 2 lines
  per case (or moves to the CCT block of option c).
* Generator fixes already queued, to go in at the same time: fault-instant snapshot currently equals
  the pre-fault one (snapshot taken before the fault is applied); 3 events of v4.1 lack the clearing
  snapshot (stable events whose clearing instant fell between stored samples); the nearest-sample trace
  writer (fix of the exact-grid rule, in `inputs/v41dyn_traces/` v2lib.py sha256 e114a980...).
* Trace every event (no stable quota, no inclusion weights): same simulation cost, 4-5 GB compact
  storage; removes the Horvitz-Thompson weighting from the window and sequence results.

## 5. Evaluation protocol for v5

| split | folds | what it supports |
|---|---|---|
| leave-k-hours-out, k = hours/10 (6 or 10 hours per fold, grouped by hour, stratified by label) | 10 | the deployment claim: unseen operating point |
| stratified random 5-fold | 5 | comparison with the literature's usual protocol; the in-sample ceiling |
| unseen faulted line (StratifiedGroupKFold by line) | 5 | fault-location generalisation |
| unseen hour AND line (held-out hours x held-out lines, reported on the intersection) | 10 | the hardest claim; small test sets, CIs will be wide |

All with the Efron bootstrap CIs and the tie rule of `tranpy.ml_benchmark`. With 60-100 hours the
hour fold has ~1,100 events, so the per-fold CIs are usable (v4.1: 9 folds of ~1,240 but each a
single OP).

## 6. Decision points

* Option (a), (b) or (c)/(c'); one tan phi per hour or both.
* Quota: run on one F8 (4-11 h wall) or request the F-family quota raised to 16 vCPUs.
* Order: Gridradar May 2021 check first (may change the dynamic layer), then generator fixes, then v5
  in one run; or v5 now on the v4.1 dynamic layer and a possible v5.1 later.
* Trace all events (recommended) or keep a stable quota.
* Whether the CCT block (c) is worth ~7 node-hours now or later.

Untouched: v4.1 frozen inputs and their SHA256SUMS, the v4.1 tables and results, `papers/`.
