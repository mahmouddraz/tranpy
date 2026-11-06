# Failure analysis: where the ML errs, and whether SIME errs on the same events

Data: v4.1 events (dataset run 738439695169033; 11,159 events, 1,325 unstable). Models: the
tuned winners of benchmark run 1086973316840813 under the leave-one-hour-out protocol, scored
out of fold at the fixed 0.5 threshold — pre-fault `line_pre` / `xgboost__class_weight`
(PR-AUC 0.78) and at-clearing `system_line_clear` / `xgboost__class_weight` (PR-AUC 0.99).
Energy methods: SIME sign of η (first swing, P_m = pre-fault P_e; 2,241 traced events with a
decision), TEF kinetic part only (`ke_corr_mj`, Youden threshold) and the Δω rule at clearing.
Tables and the script: `~/Library/Caches/tranpy/work/ml_v41dyn/failure/` (`tables.md`,
`rates_by_*.csv`, `per_line.csv`, `pr_operating_points.csv`, `analyse.py`). Wilson 95 % CIs
for rates; McNemar (statsmodels, continuity-corrected) for paired comparisons. Provisional
until the May 2021 real-event check is closed.

## 1. The labels are not a continuum, so "near the boundary" means two different things

Unstable events are almost all clear losses of synchronism: 1,193 of 1,325 have a maximum
separation above 360° (TSI < 0, median separation ~2·10⁴° at 6.6 s); 131 are in 210–360°
(out-of-step / COI criteria); one is below 210°. Stable events reach at most 210°. Only 50
events have |TSI| < 10. Consequently:

| stratum | n | unstable | pre miss | pre false alarm | clear miss | clear false alarm |
|---|---|---|---|---|---|---|
| separation 90–150° (stable) | 7,747 | 0 | – | 0.025 [0.02, 0.03] | – | 0.000 |
| separation 150–180° (stable) | 1,694 | 0 | – | 0.099 [0.09, 0.11] | – | 0.007 |
| separation 180–210° (stable) | 364 | 1 | – | 0.256 [0.21, 0.30] | – | 0.168 [0.13, 0.21] |
| separation 210–360° | 161 | 131 | 0.206 [0.15, 0.28] | 0.200 | 0.031 [0.01, 0.08] | 0.467 [0.30, 0.64] |
| separation > 360° | 1,193 | 1,193 | 0.273 [0.25, 0.30] | – | 0.034 [0.02, 0.05] | – |

- **False alarms concentrate at the boundary from the stable side**: the false-alarm rate rises
  monotonically with the maximum swing (2.5 % → 9.9 % → 25.6 % pre-fault; 0 → 0.7 % → 16.8 % at
  clearing). The at-clearing model's remaining errors are almost entirely these large-swing
  stable events and the 210–360° band (47 % false alarm among its 30 stable events).
- **Pre-fault misses are not borderline events.** Missed and caught unstable events have the
  same TSI (median −96.6 vs −96.7) and the same separation. What distinguishes the misses is
  the clearing time and the line: miss rate 95 % for the 22 unstable events cleared in
  0.2–0.3 s, 36 % at 0.3–0.4 s, 26 % at 0.4–0.5 s, 20 % at ≥ 0.5 s; 38 % on 220 kV lines vs
  22 % on 380 kV. The pre-fault model has learnt "long clearing time on a weak line → unstable";
  it misses the lines that go unstable at short clearing times, i.e. the ones with the smallest
  critical clearing time, which is the opposite of what a screening tool should miss.
- Pre-fault false alarms grow with clearing time (0 % below 0.2 s, 13.5 % above 0.5 s): the
  model over-uses the clearing time where the line descriptors do not separate the classes.
- By hour (unseen in training): pre-fault miss 12 % (Baltic import) to 34 % (very light night);
  at-clearing miss 0–7 %. No hour is an outlier beyond the CIs; the high-wind hour has the
  fewest events (876) and the widest CI.
- By the number of events on the faulted line: pre-fault miss 32 % (3–5 events) → 19 % (> 20
  events), overlapping CIs; no unstable event sits on a line with ≤ 2 events. Line identity is
  not what limits the pre-fault model.

## 2. Do the methods fail on the same events? (traced subsample, n = 2,241)

Rates on the traced events, raw and Horvitz–Thompson weighted to the population:

| method | miss | false alarm | miss (weighted) | false alarm (weighted) |
|---|---|---|---|---|
| ML pre-fault | 0.278 | 0.037 | 0.278 | 0.039 |
| ML at clearing | 0.036 | 0.011 | 0.036 | 0.011 |
| SIME (sign of η) | 0.053 | 0.154 | 0.052 | 0.140 |
| TEF kinetic part (Youden) | 0.090 | 0.174 | 0.089 | 0.171 |
| Δω max rule (Youden) | 0.032 | 0.066 | 0.032 | 0.065 |

Paired disagreements (raw counts; McNemar p):

| pair | both right | A right, B wrong | A wrong, B right | both wrong | p |
|---|---|---|---|---|---|
| A = ML pre, B = ML clear | 1,840 | 19 | 345 | 37 | 4.6e-65 |
| A = ML pre, B = SIME | 1,660 | 199 | 361 | 21 | 1.0e-11 |
| A = ML clear, B = SIME | 1,970 | 215 | 51 | 5 | 1.6e-23 |
| A = SIME, B = TEF-KE | 1,766 | 255 | 189 | 31 | 2.0e-3 |
| A = ML pre, B = ML clear (all 11,159 events) | 10,274 | 74 | 750 | 61 | 2.9e-122 |

- ML at clearing and SIME fail on **different** events: of SIME's 220 errors only 5 are shared
  with the at-clearing model; of its 56 errors only 5 are shared with SIME. The two methods are
  complementary rather than one dominating on a subset.
- SIME's errors are false alarms (154, 15 % of stable traced events) rather than misses (66 of
  1,325; 5 %) — it is conservative, as a first-swing method on a damped multi-machine system is
  expected to be. Its misses are first-swing-stable events that lose synchronism later.
- The pre-fault misses are overwhelmingly caught at clearing (345 of 382 pre-fault errors); the
  37 events both ML models get wrong are 1.7 % of the traced set.

## 3. Borderline set (|TSI| < 10, separation 295–440°)

50 events, all unstable (0.4 % of the data). Pre-fault miss 14 %, at-clearing miss 2 %, SIME
(19 traced) 0 misses, TEF-KE and Δω 0 misses. These events are *easier* than the average
unstable event for every method; they carry 0.9 % of the pre-fault errors. The stability
boundary of this dataset is not where the ML errors are.

## 4. Lines the pre-fault model misses systematically (≥ 5 unstable events, miss > 50 %)

6 of 503 lines with unstable events; 31 unstable events (2.3 %). They are more loaded (median
loading 0.45 vs 0.23 of the per-level 95th percentile), more often 220 kV, single-circuit
(median 1 vs 2) and at weaker buses (short-circuit strength 1,860 vs 2,385) than other lines
with ≥ 5 unstable events, and have a lower unstable share (33 % vs 45 %): lines that are
unstable only at some operating points, where the pre-fault descriptors are not enough. Two of
the six are in the CCT sweep. Local inertia within one hop is zero for both groups (no machine
at the line ends in either), so the 1-hop inertia features cannot help here.

## 5. Attainable operating points (hour protocol, stratified bootstrap 95 % CIs)

| model | constraint | achieved | 95 % CI | threshold | miss rate |
|---|---|---|---|---|---|
| pre-fault | precision ≥ 0.8 | recall 0.589 | [0.518, 0.628] | 0.786 | 0.41 |
| pre-fault | precision ≥ 0.9 | recall 0.367 | [0.313, 0.435] | 0.944 | 0.63 |
| pre-fault | recall ≥ 0.9 | precision 0.489 | [0.439, 0.532] | 0.108 | 0.10 |
| pre-fault | recall ≥ 0.95 | precision 0.371 | [0.347, 0.400] | 0.031 | 0.05 |
| at clearing | precision ≥ 0.8 | recall 0.991 | [0.986, 0.996] | 0.019 | 0.009 |
| at clearing | precision ≥ 0.9 | recall 0.981 | [0.973, 0.989] | 0.239 | 0.019 |
| at clearing | recall ≥ 0.9 | precision 0.994 | [0.988, 0.998] | 0.971 | – |
| at clearing | recall ≥ 0.95 | precision 0.964 | [0.940, 0.981] | 0.793 | – |

Pre-fault, a 10 % miss rate costs a precision of 49 % (one false alarm per true unstable event)
and a 5 % miss rate costs 37 %; with 11.9 % unstable events that is 13 % / 19 % of all
contingencies flagged. At clearing, 1–2 % miss is attainable at ≥ 90 % precision.

## 6. What the numbers support

1. The at-clearing model is accurate across all strata (miss 2–7 % per hour, false alarm
   ≤ 1 % except for the largest stable swings); its errors are the 180–360° stable swings, i.e.
   the physical boundary. It and SIME err on different events.
2. The pre-fault model is a **screening** tool: 90 % recall at 49 % precision, 95 % at 37 %.
   Its misses are not near-boundary cases but clear instabilities at short clearing times on
   loaded, weak 220 kV lines — the line descriptors do not yet encode what makes those lines
   fragile. This is the concrete gap the next feature work should target (post-trip path
   impedance, line-end short-circuit level after the trip), not more model capacity.
3. SIME is conservative (5 % miss, 15 % false alarm) and its disagreement with the TDS label is
   first-swing vs multi-swing, not noise; it remains a fair "energy method" row with that
   stated.
4. Claims that are **not** supported: that ML errors concentrate at the stability boundary
   (true for false alarms, false for pre-fault misses); that unseen lines or few events per
   line explain the pre-fault misses; that the borderline events are the hard ones.
