# Results on the v4.1 fault events: machine learning, time-domain simulation and energy methods

Status: **provisional** (2026-11-06). All numbers below are read from the result files of the
runs named in each section; nothing is quoted from memory. The dataset is v4.1 (dataset job
run 738439695169033, frozen inputs `inputs/v41dyn/`), the tuned benchmark is job run
1086973316840813 (`outputs/ml/1086973316840813/best.json`, `registered.json`, Delta table
`tsa_benchmark`), the accuracy-versus-time study is job run 336521735093249
(`outputs/ml/windows_336521735093249/`), and the conventional-method, failure, CCT and timing
results are in `~/Library/Caches/tranpy/work/ml_v41dyn/` (`conventional/`, `failure/`,
`cct_ml/`, `timing/`). Companion pages: [failure analysis](failure_analysis.md),
[timing](timing.md), [v5 sampling plan](v5_sampling_plan.md), [TFM/GNN plan](tfm_gnn_plan.md).
The May 2021 Gridradar real-event check is still open (section 8); every number here may move
if it changes the dynamic layer.

## 1. Dataset and protocols

**Events.** 11,159 complete three-phase line faults on the German 220/380 kV model, 1,325
unstable (11.9 %), in 9 operating hours of 2021 (max wind, max solar, min load, max load,
median load, winter peak, summer peak, lightest night hour, Baltic import at low load) with two
load power-factor variants each (18 cases). Per case: random events (loading ~ U(0.9, 1.1),
uniform fault line, clearing ~ U(0.05, 0.6) s, line tripped at clearing, 6.6 s horizon) plus a
grid sweep of 6 lines x 8 clearing times at nominal loading. Label: maximum rotor-angle
separation after the fault > 360° (TSI <= 0); the 180° centre-of-inertia label is stored too.
Of the unstable events, 1,193 exceed 360°, 131 are in 210–360° (out-of-step / COI criteria),
one is below 210°; stable events reach at most 210°. Only 50 events have |TSI| < 10.

**Traced subsample.** Trajectories were re-simulated and stored for every unstable event plus
1,000 stable ones (re-trace run 588561814210793: 2,328 events; 2,325 in the SIME files, 84
without a usable post-clearing trajectory, 2,241 with a SIME decision). Metrics on the traced
set are inclusion-weighted (Horvitz–Thompson) to the 11,159-event population.

**Feature sets** (`tranpy.ml_events`, `tranpy.line_features`): `system_pre` (19 system-level
statistics of the pre-fault operating point), `aligned_pre` (1,016 per-machine columns aligned
by stable machine id), `line_pre` (33 columns: the scenario inputs — clearing time, fault line and
position, fault impedance, loading, tan phi — plus the faulted-line and fault-bus descriptors:
pre-fault flow and angle across the line, reactance, r/x, series admittance, voltage level,
loading, circuits, bus voltage and degree, short-circuit strength, kinetic energy and
generation within 1–3 hops, hops and reactance distance to the nearest machine), `system_line_pre` (44),
`system_clear` (28: pre-fault plus the system state at the clearing instant),
`system_line_clear` (53). Outcome-derived columns (`label`, `tsi`, `angle_separation_deg`,
`unstable_*`, `simulation_end_s`, …) are refused by `check_no_leakage`.

**Protocols** (`tranpy.ml_benchmark.PROTOCOLS`):

| protocol | folds | question |
|---|---|---|
| `hour` (primary; selection and promotion gate) | 9, leave one operating hour out | unseen operating point |
| `random` | 5, stratified, shuffled | the i.i.d. estimate the literature usually reports |
| `line` | 5, stratified group folds by faulted line | unseen fault location |

Every pooled metric carries a 95 % stratified bootstrap interval (1,000 resamples); the
primary metric is PR-AUC (average precision); candidates are ranked by PR-AUC with ties inside
the leader's interval settled by the lower miss rate (`select_best`). Models: LightGBM,
XGBoost, logistic regression, balanced random forest, each crossed with no balancing, class
weights, and class weights plus an F1-tuned threshold. The three sets with line descriptors
were tuned (Optuna TPE, 30 trials per outer training fold, inner 3-fold `StratifiedGroupKFold`
by hour; the outer test fold never enters a study); the others keep fixed parameters
(300 trees). Miss rate = share of unstable events predicted stable at the 0.5 threshold.

## 2. Pre-fault prediction

### 2.1 Headline (run 1086973316840813, best candidate per set and protocol, PR-AUC [95 % CI])

| feature set | `hour` | `line` | `random` |
|---|---|---|---|
| `system_pre` (untuned) | 0.428 [0.402, 0.451] | 0.441 [0.419, 0.466] | 0.534 [0.509, 0.559] |
| `aligned_pre` (untuned) | 0.428 [0.405, 0.454] | 0.415 [0.392, 0.439] | 0.480 [0.457, 0.505] |
| `line_pre` (tuned) | 0.786 [0.767, 0.803] | 0.849 [0.833, 0.863] | 0.885 [0.872, 0.898] |
| `system_line_pre` (tuned) | 0.763 [0.743, 0.782] | 0.867 [0.852, 0.880] | 0.907 [0.895, 0.919] |

The `hour` row of `line_pre` is the top-PR-AUC candidate (tuned `lightgbm__none`, miss rate
0.396). The tie rule selected and registered `xgboost__class_weight` instead: PR-AUC 0.781
[0.761, 0.800], inside the leader's interval, with miss rate 0.266 [0.242, 0.290] (precision
0.680, recall 0.734, ROC-AUC 0.956, Brier 0.053). The previous champion (v4 run, `aligned_pre`,
EasyEnsemble) had PR-AUC 0.436 and miss rate 0.134; the new version was registered as
`challenger` but **not** promoted ("miss rate higher beyond the interval",
`registered.json`). On the i.i.d. split the same `line_pre` model reaches 0.885 and on unseen
lines 0.849.

### 2.2 Where the pre-fault information comes from

* The pre-fault state alone does not separate the classes: `system_pre` and the 1,016-column
  `aligned_pre` give PR-AUC 0.43 under every protocol, with every model family at the same
  level (the gap to `line_pre` is 0.36, non-overlapping intervals). The lift comes from the
  **faulted-line descriptors** (line impedance and loading, circuits, voltage level, local
  inertia and short-circuit strength at the line ends, electrical distance to the machines)
  combined with the scenario inputs (clearing time, fault position, loading).
* Adding the system-level statistics to the line descriptors (`system_line_pre`) helps on the
  `line` and `random` splits (+0.02) but **hurts** on the `hour` split (0.763 vs 0.786): the
  system statistics identify the operating hour and let the model over-fit it.
* The model generalises across lines (0.849) but not across operating points (0.786): with
  9 hours it trains on 8 and extrapolates to a ninth. The 0.10 gap to the random split is an
  information limit of the sample, not of the model — every family reached the same level
  (details and the sampling consequence in the [v5 plan](v5_sampling_plan.md)).
* A clearing-time-only rule gives PR-AUC 0.286 (hour protocol, `conventional/TABLES.md`): the
  clearing time matters but is not the signal on its own.

### 2.3 Failure analysis (hour protocol, out of fold, threshold 0.5; `failure/`)

Full tables in [failure_analysis.md](failure_analysis.md). What the pre-fault model misses:

* **Misses are not borderline events.** Missed and caught unstable events have the same TSI
  (median −96.6 vs −96.7) and separation. They differ by clearing time and line: miss rate
  95 % for the 22 unstable events cleared in 0.2–0.3 s, 36 % at 0.3–0.4 s, 26 % at 0.4–0.5 s,
  20 % at >= 0.5 s; 38 % on 220 kV lines vs 22 % on 380 kV. The model has learnt "long clearing
  time on a weak line -> unstable" and misses the lines with the smallest critical clearing
  time — the opposite of what a screening tool should miss.
* **False alarms sit at the boundary from the stable side**, rising monotonically with the
  maximum swing: 2.5 % (90–150°), 9.9 % (150–180°), 25.6 % (180–210°).
* Six of 503 lines with unstable events are missed systematically (>= 5 unstable events, miss
  > 50 %; 31 events, 2.3 %): more loaded, more often 220 kV, single-circuit, at weaker buses,
  and unstable only at some operating points. The 1-hop inertia features are zero for them.
* Line identity and the number of events per line do not explain the misses (miss 32 % with
  3–5 events on the line, 19 % with > 20, overlapping CIs). The 50 borderline events
  (|TSI| < 10) are *easier* than average (miss 14 %) and carry 0.9 % of the pre-fault errors.
* Attainable operating points (stratified bootstrap CIs): recall 0.90 at precision 0.489
  [0.439, 0.532]; recall 0.95 at precision 0.371 [0.347, 0.400]; precision 0.80 at recall
  0.589 [0.518, 0.628]. With 11.9 % unstable events, a 10 % miss rate flags 13 % of all
  contingencies, a 5 % miss rate 19 %: the pre-fault model is a **screening** tool.

## 3. Post-clearing assessment: time-domain simulation, energy methods, machine learning

The reference is the time-domain simulation (TDS, ANDES 2.0.0, 6.6 s horizon) that produced
the labels. Two instant-based alternatives are compared against it on the same events.

### 3.1 All 11,159 events, hour protocol, thresholds set on the training hours (`conventional/TABLES.md`)

| method | input | PR-AUC | ROC-AUC | miss (Youden thr.) | false alarms / misses (counts) |
|---|---|---|---|---|---|
| ML, LightGBM class-weighted, `system_clear` | clearing snapshot | 0.9931 | 0.9986 | 0.041 | 59 / 54 |
| ML, logistic regression, `system_clear` | clearing snapshot | 0.9919 | 0.9987 | 0.058 | 58 / 77 |
| rule: max. rotor-speed deviation at clearing | clearing snapshot | 0.9527 | 0.9927 | 0.030 | 586 / 40 |
| rule: max. rotor angle at clearing | clearing snapshot | 0.9505 | 0.9866 | 0.065 | 619 / 86 |
| TEF, kinetic part only (`ke_corr_mj`, COI-corrected) | clearing snapshot | 0.6372 | 0.9271 | 0.085 | 1,915 / 112 |
| TEF, kinetic part only (`ke_coi_mj`) | clearing snapshot | 0.5864 | 0.8907 | 0.163 | 2,446 / 216 |
| rule: clearing time | scenario | 0.2859 | 0.8121 | 0.086 | 3,830 / 114 |

McNemar against the LightGBM at-clearing model (exact): speed-deviation rule 560 vs 47
discordant events, p = 1.6e-112; TEF kinetic part 1,974 vs 60, p reported as 0. The kinetic energy
at clearing without the potential-energy term is therefore not a usable classifier on this
system; normalised by a per-line critical clearing time it is (PR-AUC 0.997, miss 0.025 on the
329 grid-sweep events with a bracketed CCT, `conventional/cct_eval.md`), but only 38 of the
108 (case, line) sweeps bracket the CCT (70 lie above the 0.6 s sweep), so the critical
energy is known for 1.4 % of the events.

### 3.2 SIME on the traced events (`conventional/sime_v41t_*`, `sime_ieee9_v2`)

SIME (Pavella's one-machine-infinite-bus equivalent, first swing, 5 candidate critical-machine
splits) decides after the post-clearing trajectory; the sign of its margin η is the
prediction. Two mechanical-power conventions were run: P_m = pre-fault P_e (`pre`) and P_m from
the trajectory (`traj`).

| system | events decided | sign agreement with TDS | PR-AUC (η) | ROC-AUC | miss (sign) | false unstable (sign) |
|---|---|---|---|---|---|---|
| German grid, P_m pre-fault | 2,241 of 2,325 | 90.2 % | 0.893 | 0.959 | 5.3 % (66 of 1,325) | 154 |
| German grid, P_m trajectory | 2,241 | 90.2 % | 0.897 | 0.961 | 5.1 % (63) | 157 |
| IEEE 9-bus (1,000 events, 450 unstable), P_m pre-fault | 998 | 99.9 % | 0.9999 | 0.9999 | 0 | 1 |

On the 9-bus system SIME reproduces the TDS label; on the grid it is conservative: its errors
are false alarms (15 % of the stable traced events) rather than misses, as a first-swing
method on a damped multi-machine system is expected to be; its misses are first-swing-stable
events that lose synchronism later. Sign agreement by case ranges from 63 % (`high_wind_t20`)
to 94 % (`median_load_t20`); PR-AUC per hour 0.79–0.97 (both conventions), lowest for the
high-wind and low-load hours.

### 3.3 Do the methods fail on the same events? (traced set, n = 2,241; `failure/tables.md`)

| method | miss | false alarm | miss (weighted) | false alarm (weighted) |
|---|---|---|---|---|
| ML pre-fault (`line_pre`) | 0.278 | 0.037 | 0.278 | 0.039 |
| ML at clearing (`system_line_clear`) | 0.036 | 0.011 | 0.036 | 0.011 |
| SIME (sign of η) | 0.053 | 0.154 | 0.052 | 0.140 |
| TEF, kinetic part only (Youden) | 0.090 | 0.174 | 0.089 | 0.171 |
| speed-deviation rule (Youden) | 0.032 | 0.066 | 0.032 | 0.065 |

McNemar (continuity-corrected): ML at clearing vs SIME, 215 events right only for ML, 51 right
only for SIME, 5 both wrong, p = 1.6e-23; of SIME's 220 errors only 5 are shared with the
at-clearing model. ML pre-fault vs ML at clearing on all 11,159 events: 750 vs 74, p = 2.9e-122
— the pre-fault misses are overwhelmingly caught at clearing. The two instant-based methods
are complementary rather than one dominating a subset.[^energy]

### 3.4 At-clearing ML across protocols (run 1086973316840813)

| feature set | `hour` | `line` | `random` |
|---|---|---|---|
| `system_clear` (untuned) | 0.9933 [0.9914, 0.9951], miss 0.051 | 0.9941 [0.9922, 0.9957] | 0.9947 [0.9928, 0.9963] |
| `system_line_clear` (tuned) | 0.9930 [0.9910, 0.9948], miss 0.034 [0.025, 0.044] | 0.9944 [0.9927, 0.9960] | 0.9950 [0.9930, 0.9967] |

Registered as `tsa_clear` version 3 (tuned `xgboost__class_weight`, precision 0.934, recall
0.966, Brier 0.009) and promoted to `champion` (PR-AUC 0.9930 vs 0.9926 tied within the
interval, miss 0.034 vs 0.045). Its remaining errors are the 180–360° stable swings (false
alarm 16.8 % in 180–210°, 46.7 % in 210–360°), i.e. the physical boundary; miss 2–7 % per
hour. At clearing, 1–2 % miss is attainable at >= 90 % precision (recall 0.981 [0.973, 0.989]
at precision 0.9).

### 3.5 CCT regression as an alternative pre-fault target (`cct_ml/results.md`)

Label-derived CCT bounds per (case, line) from the random events: 7,915 pairs, 383 two-sided
(354 monotone), 6,794 lower-only, 738 upper-only; two-sided interval width median 0.24 s. A
`line_pre` regressor of the CCT, thresholded against the event's clearing time (model B),
reaches PR-AUC 0.396 [0.372, 0.421] on all events under the hour protocol versus 0.783
[0.764, 0.802] for the direct classifier (A) — but 0.959 [0.935, 0.981] vs 0.955 on the 329
grid-sweep events whose CCT is bracketed, and 0.956 vs 0.955 with miss 0.077 vs 0.243 on the
1,036 events of two-sided pairs. The CCT target wins where the CCT is known and loses where it
is censored; a monotone constraint on the direct model (C) changes nothing (0.785). This is
the case for the CCT block of the v5 plan.

## 4. Accuracy versus assessment time (run 336521735093249, traced events, hour protocol)

Instants: pre-fault; clearing + 0 / 0.05 / 0.1 / 0.2 / 0.3 s (the event's own clearing time);
fault inception + 0.3 / 0.5 / 0.8 s (fixed, causal with respect to the clearing, cf. D9 in
[validation.md](validation.md)). Inputs: `window` (system-level series at the instant plus
window statistics), `window_line` (plus the `line_pre` descriptors), `sequence` (10 ms series
for the networks). LightGBM tuned (20 trials); networks in two sizes, untuned. Events whose
simulation ended before an instant are excluded there (all unstable: 21 at clear + 0 s, 101 at
+0.05 s, 123 at inception + 0.8 s of 2,325). PR-AUC [95 % CI]:

| instant | LightGBM `window_line` | LightGBM `window` | best sequence network |
|---|---|---|---|
| pre-fault | 0.675 [0.617, 0.747] | 0.306 [0.277, 0.348] | – |
| clearing + 0 s | 0.990 [0.985, 0.995] | 0.978 [0.955, 0.994] | 0.990 [0.985, 0.994] (Transformer L) |
| clearing + 0.05 s | 0.996 [0.991, 0.999] | 0.998 [0.996, 1.000] | 0.996 [0.993, 0.998] (Transformer L) |
| clearing + 0.1 s | 0.997 [0.993, 1.000] | 0.999 [0.998, 1.000] | 0.997 [0.995, 0.998] (Transformer L) |
| clearing + 0.3 s | 0.999 [0.997, 1.000] | 0.997 [0.991, 1.000] | 0.998 [0.997, 0.999] (Transformer S) |
| inception + 0.3 s | 0.966 [0.943, 0.986] | 0.972 [0.957, 0.985] | 0.866 [0.829, 0.906] (Transformer S) |
| inception + 0.5 s | 0.998 [0.997, 0.999] | 0.998 [0.995, 0.999] | 0.972 [0.956, 0.985] (TCN L) |
| inception + 0.8 s | 0.998 [0.995, 1.000] | 0.998 [0.995, 0.999] | 0.992 [0.986, 0.996] (LSTM L) |

* **Knees** (`knees.json`): clearing + 0.05 s for every model and input under the hour and
  line protocols (LSTM large: 0.2 s; random protocol `window`: 0.1 s), and inception + 0.5 s
  for all. 50 ms after clearing the window models are within their intervals of their final
  level. The pre-fault row is not the benchmark of section 2: it is the traced subsample
  (2,325 events, inclusion-weighted) with a 20-trial tuning budget.
* **Sequence networks <= tuned boosting.** At every instant the networks' interval overlaps
  or lies below the LightGBM interval; at the fixed inception instants they are clearly worse
  (0.79–0.87 vs 0.97 at + 0.3 s, 0.95–0.97 vs 0.998 at + 0.5 s), because without the clearing
  instant as a reference the window mixes fault-on and post-clearing samples. Miss rates at
  clearing + 0.05 s: LightGBM 0.004–0.005, networks 0.009–0.019.
* Across protocols (window inputs only): the `line` and `random` splits give the same picture
  (clearing + 0.05 s 0.996–0.999; inception + 0.5 s 0.996–0.998); the pre-fault `window_line`
  row is 0.758 on `line` and 0.734 on `random`.

## 5. Speed

Measured on the v4.1 events ([timing.md](timing.md); Apple M3 single core unless stated).

| method | needs | per event (median / p90) | status |
|---|---|---|---|
| TDS, ANDES, full 6.6 s | case + scenario | 13.2 s (mean of 20 events); production 9.3 core-s mean on Standard_F8as_v7 (28.9 core-h for 11,162 events) | measured |
| SIME total (TDS to the decision instant + margin) | case + scenario (trajectory must be simulated) | ≈ 1.4 s (≈ 3.4 s p90); margin alone 5.5 / 9.4 ms (62 machines), 8.6 / 17.3 ms (146 machines) | **estimate** (uniform ANDES cost per simulated second) |
| ML pre-fault (`tsa_pre`, XGBoost, 33 features) | solved operating point + fault line/position | 0.93 / 2.0 ms single; 33 µs per event in batches of 1,000 | measured |
| ML at clearing (`tsa_clear`, 53 features) | clearing snapshot + line descriptors | 1.3 / 2.1 ms single; 14 µs batched | measured |
| `line_pre` descriptors | case workbook + solved OP | 0.70 s per case once, then 0.54 ms per event | measured |

Ratios: TDS vs ML pre-fault ~14,000x (batch of 1), ~9,000x with the descriptor computation
included; TDS vs SIME ~9x (only the shortened horizon; SIME remains simulation-based and
cannot run before the fault); SIME vs ML at clearing ~1,000x, but the at-clearing ML needs the
clearing snapshot (≈ 2.9 s of simulated time on the M3, or a measurement in operation). The
comparison that matters in operation is "snapshot + ML" vs "snapshot + 0.22 s more simulation
+ SIME". Unstable events do not stop early in general (median `simulation_end_s` 6.6 s for both
labels), so the TDS cost is label-independent.

## 6. Failure modes

| method | dominant error | where | evidence |
|---|---|---|---|
| ML pre-fault | misses (27 %) | short clearing times (0.2–0.4 s) on loaded, weak, single-circuit 220 kV lines; not near the boundary | section 2.3; `failure/rates_by_clearing_band.csv`, `lines_systematic_miss.csv` |
| ML pre-fault | false alarms growing with clearing time (0 % < 0.2 s, 13.5 % >= 0.5 s) | the model over-uses the clearing time where the line descriptors do not separate the classes | `failure/rates_by_clearing_band.csv` |
| ML at clearing | false alarms on large stable swings (16.8 % in 180–210°, 46.7 % in 210–360°) | the physical boundary | `failure/rates_by_sep_band.csv` |
| SIME | false alarms (15 % of stable traced events); 5 % misses that are multi-swing instabilities | worst in the high-wind hour (sign agreement 63–70 %) | `sime_sanity.json`, `sime_eval.md` |
| TEF, kinetic part only | false alarms (17 %) and misses (9 %): no potential-energy term, no per-line critical energy | everywhere except the 38 lines with a bracketed CCT | `conventional/TABLES.md`, `cct_eval.md` |
| sequence networks | lower precision (0.85–0.92 vs 0.92–0.93) and higher miss (0.009–0.019 vs 0.004–0.005) than boosting at clearing + 0.05 s; degrade at fixed inception instants | window without the clearing reference | `windows_results.csv` |
| CCT regression | censored labels: 86 % of (case, line) pairs have a lower bound only | everything outside the grid sweep | `cct_ml/bounds_report.json` |

Claims **not** supported by the data: that ML errors concentrate at the stability boundary (true
for false alarms, false for pre-fault misses); that unseen lines or few events per line explain
the pre-fault misses; that the borderline events are the hard ones.

## 7. Comparison with published work

Compiled from the literature survey of 2026-11-05 (`literature.md`, web-fetched source text;
only rows whose full text was readable are tabulated, numbers verbatim from the papers). No
published pre-fault study with (i) a > 1,000-bus system, (ii) fault location and clearing time
as inputs, (iii) a held-out-operating-point split and (iv) PR-AUC or recall on a ~12 %
unstable set was found; no TSA study reporting PR-AUC was found at all, and none reports
confidence intervals (dispersion, when given, is a ± std over folds or seeds).

| study | test system | events (unstable share) | split protocol | task | reported metric |
|---|---|---|---|---|---|
| Meghdadi, Tack, Liebman 2019 (ICPES) | WECC 9-bus | 455 train / 699 validation (23 % / 28 %) | validation load range extends beyond training (0.25–1.85 vs 0.3–1.7) | pre-fault, fixed contingency per classifier | 0.57 % error (NN, single validation set) |
| Zhao et al. 2024 (Sensors 24:5052) | IEEE 39-bus | 7,000 train / 3,000 test per domain (5:1 to 40:1) | source domain -> changed renewable share / load (operating scenario shift) | pre-fault, steady-state features | CNN acc 97.9 % in-domain; 83.2 % / 79.4 % cross-domain without adaptation |
| Lei et al. 2024 (Front. Energy Res. 12) | New England 39-bus, 256 power flows | 11,776 (not reported) | 8:1:1 random | pre-fault + fault characteristics | TCS-ELM acc 0.978 |
| Han et al. 2021 (arXiv:2112.08836) | IEEE 39-bus | 14,221 (19.1 %) | random 8,533 / 5,688 | at fault clearing | DT unstable recall 0.935, acc 0.969; MLP 0.926 / 0.977 |
| Chen et al. 2022 (arXiv:2205.06576) | IEEE 39-bus; Polish 2383-bus | 10,000 (29.1 %); 3,000 (33 %) | 10-fold random CV, mean ± std | clearing / early post-fault, GNN | acc 99.2 ± 0.2 / F1 99.4 (39-bus); acc 99.2 ± 0.5 (2383-bus) |
| Huang et al. 2024 (Front. Energy Res. 11) | IEEE 39-bus | 11,160 (46.9 %) | shuffled 3:1:1; N-2/N-3 topologies separately | pre-fault + fault + post-fault snapshots, GraphSAGE | acc 96.66 %, F1 0.802; 96.65 % on unseen topologies |
| Wang et al. 2026 (IEEE PESGM) | IEEE 39-bus; IEEE 145-bus | 15,000 (≈ 25 %); 8,000 | not stated; 40 new topologies tested | pre/during/60 ms post-fault voltages, Transformer + GAT | acc 99.62 %; 99.06 % on new topologies |
| Sun et al. 2024 (J. Mod. Power Syst. Clean Energy 12) | IEEE 39-bus; IEEE 118-bus, 2nd-order swing model | 39,000 (9.4 %); 52,038 (2.5 %) | 60/20/20 | post-fault angular-frequency series | acc 99.38, FNR 0.65 %, AUC 0.9994 (39-bus); transfer to 11th-order model acc 62.4 % |
| Shen et al. 2026 (arXiv:2609.14374, LLaTSA) | IEEE 39/68/118, Iceland 189 | not reported | N-3 and 40 % RES level held out; 20 seeds | post-fault trajectories | unstable recall 96.2 ± 1.2 % in range, 95.6 ± 1.3 % unseen N-3 |

How this relates to the numbers above: the at-clearing and early-window results of section 3–4
(PR-AUC 0.993–0.999, miss 0.3–3 %) sit in the same region as the random-split accuracies of
96.7–99.9 % in the table, and the fair comparison for them is the `random` protocol
(0.995), not the `hour` protocol. For the pre-fault task only Meghdadi 2019 and Zhao 2024 ask a
comparable question, on 9- and 39-bus systems; Zhao 2024 is the one study that quantifies the
drop under an operating-scenario shift (97.5 % -> 79–83 % accuracy), the same axis the `hour`
protocol tests here (0.885 -> 0.786 PR-AUC). The Polish 2383-bus case of Chen 2022 (3,000
events, 33 % unstable, random CV) is the only large-system classification result found.
Abstract-only sources (Cremer et al. 2019/2021, Bellizio 2022, Lu & Bu 2025, Dai 2025, Kim
2023, Huang 2021) are listed in `literature.md` without numbers.

## 8. Limitations and provisional status

* **Real-event check open.** The May 2021 Gridradar comparison of the dynamic layer has not
  been closed; if it changes the generator or neighbour models, v4.1 is re-simulated and every
  number above is superseded. All result tables carry `provisional = true`.
* **Nine operating hours.** The hour protocol trains on 8 operating points; the per-fold CIs
  are wide and the 0.786 vs 0.885 gap is a sampling limit. The [v5 plan](v5_sampling_plan.md)
  (60–100 stratified hours, fewer events each, optional CCT block on 50 lines, every event
  traced) is the planned remedy; nothing of it is built.
* **Generator fixes queued for v5:** the stored fault-instant snapshot equals the pre-fault one;
  3 events lack the clearing snapshot; nearest-sample trace writer.
* **Traced subsample.** SIME, the window study and the error-overlap analysis use 2,241–2,325
  traced events (all unstable plus 1,000 stable), inclusion-weighted; the weighting disappears
  once every event is traced.
* **SIME timing is an estimate**; the fault-on period with its reduced step is probably costlier
  than the uniform per-second cost assumed, which would raise SIME's cost.
* **CCT coverage.** Critical clearing times are bracketed for 38 of 108 swept (case, line)
  pairs and two-sided for 383 of 7,915 random pairs; the energy-method rows that need a
  critical energy are evaluated on 329 events only.
* **Tabular foundation models and graph models** have not been run (plan in
  [tfm_gnn_plan.md](tfm_gnn_plan.md)); the "GBDT is sufficient" reading of section 2 holds
  until an in-context or graph model beats the `hour`-protocol interval on `line_pre`.
* Registered models of run 1086973316840813 were saved by MLflow as cloudpickle rather than
  skops (the Databricks default); `tranpy.ml_tracking` now requests skops explicitly, so later
  versions load without executing pickled code.

[^energy]: The energy-method rows are SIME and the kinetic part of the transient energy
    function ("TEF, kinetic part only", `ke_corr_mj` / `ke_coi_mj`); no full TEF with a
    potential-energy term was evaluated on v4.1. Jev methods were not run on v4.1 and appear in
    no table; they are noted here only so that the method inventory is complete.
