# CCT sweep speed-up (experiment e11)

Goal: cut the cost of `scripts/cct_bracket.py` before the v5 sweep. Ground truth (GT) is the
e4 brackets of the 18 v4.1 cases (`~/data/exp/e4_sim/cct/cct_<case>.parquet`, 35,988
line-terminal tasks, 161,838 simulations). No v5 sweep was run. All runs were on cairo-lab with at
most 8 worker processes (other agents were active, so per-simulation times are noisy at about
±10 %).

Every number is labelled:
- **[measured]**: new simulations in e11.
- **[replay]**: computed from the simulations already stored in GT (exact for the stored points;
  for brackets it assumes stability is monotonic in clearing time, as the bisection itself does).

Artefacts are in `~/data/exp/e11_cct_speed/`:
- scripts: `bench.py` (single simulations with switchable settings), `e2e.py` (full brackets),
  `offline.py` (replays), `compare.py`;
- results: `offline.json`, `compare.json`, `runs/*.json|parquet`.

## Sample and baseline

- Sample: 2 cases (high_load_t10, high_solar_t10) × 150 line-terminals, stratified by voltage.
  Each case has 75 bracketed, 50 above and 25 failed tasks.
- Point tests run 450 simulations at the hardest clearing times:
  - bracketed tasks: at the GT cct_lo and cct_hi;
  - above tasks: at 1.0 s;
  - failed tasks: at the failing clearing time.
- The baseline reproduces GT exactly: 400 of 400 decided labels match, and all 50 failures
  recur **[measured]**.
- Cost profile, serial **[measured]**:
  - about 5.2 s per simulation: 1.1 s parsing the xlsx, about 4.5 s stepping (706 steps, the
    KLU numeric refactorisation at every Newton iteration);
  - at 8 workers, 11.7 s per simulation.
- GT averages 4.5 simulations per task (60 % of tasks are "above" and take 2), not 16 **[replay]**.

## Results per candidate

Speed and label changes are on the 450-point sample (8 workers) unless stated.

| # | Candidate | Speed | Labels vs baseline | Verdict |
|---|---|---|---|---|
| 0 | **Hi-first ordering**: drop the 0.05 s probe, test 1.0 s first | ×1.29 fewer sims **[replay]** | the 0.05 s probe was never unstable in 35,988 tasks (30 failed) | **use** |
| 1a | **Out-of-step stop**: stop 0.05 s after the latched >180° crossing | ×2.19 per sim on this sample (5.35 vs 11.71 s) **[measured]**; unstable runs end at 1.85 s instead of 7 s | 0 / 450 changed **[measured]** | **use** |
| 1b | Settled-stable stop (amplitude and relative-speed window) | fires on at most 45 % of stable runs, at a mean of 5.5 s: little gain **[measured, post hoc]** | no false stops in 400 runs | not worth it; 4 is better |
| 2 | **Warm start** from the twin case (other tan φ), ±0.01 s bracket with widening | 1.87 vs 3.47 sims/line over 18 cases, ×1.85 **[replay]**; e2e 2.4 vs 5.88 GT sims/line **[measured]** | see e2e below | **use for the second tan φ** |
| 2' | Warm start from the nearest *other hour* | ×1.05 **[replay]** | 100 % agreement | not worth it with 9 hours; recheck with 25–30 hours |
| 3 | Skip "above" lines without simulating | saves 1 sim on predicted lines | rule "above in all other hours": precision 97.6 %, **324 bracketed lines missed, CCT down to 0.19 s**; a gradient-boosting classifier on line and case features is no better (96.9 % at p ≥ 0.99) **[replay]** | **unsafe**. One confirmation run at 1.0 s is safe and is exactly item 0 |
| 4 | **Shorter window**, end at 4.5 s instead of 7.0 s | ×1.61 per sim (7.29 s) **[measured]** | 0 / 450 **[measured]**; over all 59,586 unstable GT sims: end at 3.0 s → 35 brackets change, 3.5 s → 6, 4.0 s → 2, **4.5 s → 0** **[replay]**; latest onset 4.06 s | **use 4.5 s** |
| 5a | JSON case instead of xlsx | ×1.03 **[measured]** | 0 / 450 | use (free) |
| 5b | Step 0.02 s | ×1.09 | **14 changed** (8 vs GT; 6 failures resolved, others flipped) | reject |
| 5c | Newton tolerance 1e-3 | ×1.18 | **9 changed** | reject |
| 5d | Line search off | ×1.11 | **56 changed**, 87 failures instead of 50 | reject |

Note on 1a: the first implementation stopped at the crossing itself. One run (a run retried
with a 2.5 ms step) then lost its out-of-step flag when the retry was resampled onto the 10 ms
grid, and was labelled stable. Stopping 0.05 s after the crossing fixed it (`early_oos_v3`: 0
mismatches). Any port into `cct_bracket.py` must keep this margin.

## End-to-end check (real bisection) **[measured]**

`e2e.py`: hi-first + JSON + out-of-step stop + 4.5 s window + warm start from the GT bracket of
the twin case (`_t20`), on the same 300 tasks with 8 workers.

| | GT (e4) | Fast |
|---|---|---|
| sims per task | 5.88 | **2.40** |
| mean sim time | 11.7 s (baseline at 8 workers) | **3.6 s** |
| total sim-seconds | about 20,600 | **2,595** (×7.9) |
| wall time | about 2,580 s (estimated, 8 workers) | **348 s** (×7.4) |

- **Above:** 100 / 100 identical.
- **Bracketed (GT):** 141 / 150 bracketed. On all 141 the CCT error is ≤ 9.9 ms (mean 4.2 ms),
  i.e. within the 10 ms bracket width.
- **The other 9 GT-bracketed tasks became "failed"**, and 2 GT-failed tasks became bracketed.
  These are numerical failures (ANDES fails after a smaller-step retry) at particular clearing
  times. The warm start probes different times, so it hits or misses different failure points.
  At identical clearing times the fast settings change no failure (items 1a, 4, 5a). Net status
  agreement is 96.3 %.
- **Fix before v5:** on a numerical failure, retry the same task at clearing time ±2 ms (and fall
  back to the cold bisection) before giving up. This was not implemented or measured here.

## v5 defaults check (F1b machine data, governor-limit shift) **[measured]**

- Case: the v5-default high_load_t10 (`~/data/exp/e10_machine_params/v5default/high_load_t10`),
  with the current `src/tranpy/simulation` (`shift_governor_limits=True`).
- Tasks: 105 (40 above, 40 bracketed, 25 failed in v4.1), cold hi-first bisection.
- Baseline (xlsx, 7 s window) against fast (JSON, out-of-step stop, 4.5 s window):

| | Baseline | Fast |
|---|---|---|
| status | 40 above / 46 bracketed / 19 failed | **identical** |
| CCT | | identical (0.0 s difference on all 46 brackets) |
| labels at the same clearing times | | **0 mismatches in 480** |
| mean sim time | 12.85 s | **3.31 s (×3.9)** |

- Warm start could not be checked on v5 (no v5 twin bracket exists yet). It rests on the v4.1
  e2e above.

## Parallel scaling **[measured, small batch]**

- Batch: 24 simulations.
- Wall time 247 / 131 / 82 / 50 s at 1 / 2 / 4 / 8 workers, i.e. ×1.9 / ×3.0 / ×5.0.
- The time per simulation stays about 10–11 s from 1 to 8 workers, so there is no contention up
  to 8. The sublinear wall time is batch tail and pool start-up.
- The machine has 24 physical cores (48 threads). Beyond 8 workers this was not measured (limit).
  The e4 sweep at 32 workers took about 8–27 s per simulation against 11 s here, so expect
  contention above about 24. **Assumption: 24–32 workers give about ×2.5 the throughput of 8.**

## Recommended configuration

1. Hi-first: simulate 1.0 s first. Stable → "above" (1 sim). No 0.05 s probe.
2. Out-of-step stop 0.05 s after the latched crossing (same 180° COI criterion).
3. Window end 4.5 s (fault at 1.0 s, clearing ≤ 2.0 s).
4. Case loaded from ANDES JSON, converted once per case.
5. First tan φ of an hour cold. The second tan φ is warm-started from the first: ±0.01 s around
   its CCT, doubling the step outward on a miss; "above" or failed donor → cold.
6. On a numerical failure, retry at ±2 ms before marking failed (to be added).
7. Keep step 0.01 s, tolerance 1e-4, line search on (changing any of them changes labels).

## Projection (2,000 tasks per case, 8 workers)

| | per case | pair (2 tan φ) | 25 h × 2 | 30 h × 2 | 100 h × 2 |
|---|---|---|---|---|---|
| e4 baseline (4.5 sims/task, 11.7 s) | 3.7 h | 7.3 h | 183 h | 219 h | 730 h |
| fast, cold (3.47 sims/task **[replay]**, about 3.5 s **[measured]**) | 0.84 h | | | | |
| fast, warm (1.87 sims/task **[replay]**) | 0.45 h | **1.3 h** | **32 h** | **39 h** | 130 h |
| same at 24–32 workers (×2.5 assumed) | | 0.5 h | **13 h** | **16 h** | 52 h |

Overall speed-up is about ×5.6 per hour-pair against the e4 settings. Simulation counts come from
the v4.1 status mix; v5 cases with more bracketed lines cost proportionally more.

## Risks

- **Late separations:** the 4.5 s window would miss an out-of-step onset after 4.5 s. None
  occurred among 59,586 unstable GT simulations (latest 4.06 s), and the v5-defaults check
  showed none. Separations after 7 s were never observable in GT either. To monitor this in v5,
  re-run about 1 % of stable results at 7 s.
- **Warm start:** validated on v4.1 twins only. CCT difference between tan φ twins: 0 s median,
  0.015 s p90, 0.14 s max **[replay]**. The widening fallback covers large shifts at a cost of
  extra simulations, not errors (100 % status agreement in the replay). Probing different
  clearing times moves numerical failures around (9 / 150 here) until item 6 is added.
- **Hi-first:** a line already unstable at 0.05 s would be bisected down and end as a bracket at
  the lower bound instead of "below". This never happened in GT.
- **Out-of-step stop:** must keep the 0.05 s margin (see the note under item 1a).
- **Skip rules and solver loosening:** shown to change labels; do not use.
