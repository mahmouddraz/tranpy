# Scenario design for transient-stability datasets

How simulated events should be generated, labelled and evaluated, compared with how the
existing PowerFactory campaigns did it. Status: **proposal for review**; nothing here is
implemented as a default yet.

## What the literature does

| Aspect | Common practice | Examples |
|---|---|---|
| Operating points | Vary load *and* generation over a range (typically 80-120 % of base, sometimes wider), re-dispatch, solve the power flow, drop non-convergent or low-voltage cases | Mendeley IEEE 39-bus dataset (80-120 % in 5 % steps, PowerFactory); IEEE 9-bus benchmark (scaled load and generation, AC power flow per case); unified grid representation protocol (excludes low-voltage pre-fault cases) |
| Fault type and location | Three-phase (worst case), sometimes also single-phase; at buses or at line terminals *and* intermediate points | 9-bus benchmark (18 fault locations); IEEE 39/118 protocol (line terminals and intermediate points); LLM-driven pipeline (faults at all buses) |
| Clearing | Clearing time *varied* (e.g. 50-500 ms) or the critical clearing time (CCT) computed by bisection, so both stable and unstable outcomes occur for the same case; the faulted line is opened | LLM-driven pipeline (50-500 ms); comparative CCT study (bisection on CCT, 10 contingencies x 150 operating points) |
| Pre-fault state | Fault applied to a settled system (power-flow initialised, fault at a fixed time such as 1 s) | comparative CCT study (fault at t = 1 s) |
| Stability label | Rotor-angle separation: maximum angle difference between any two generators after the fault; unstable if > 360° (equivalently TSI = (360 - δmax)/(360 + δmax) ≤ 0). Some use 180° w.r.t. the centre of inertia (more conservative); thresholds are not standardised | TSI papers; B-stability (180°/360°); Morales, Milanović, Papadopoulos (PowerTech 2019) on threshold choice; 500° in one protocol |
| Targets | Binary label, plus continuous margins (TSI, CCT) for regression and boundary analysis | CCT regression studies |
| Measurements | PMU-like signals (bus voltage magnitude and angle), short windows after clearing (e.g. 0.6 s at 10 ms) | Mendeley dataset |
| Evaluation | Accuracy alone is insufficient; generalisation to unseen operating conditions, contingencies and topologies is the open problem; class imbalance must be handled | ANN review (Shahzad); domain-adaptation and topology-change studies |

## Where our previous campaigns deviate

| # | Previous PowerFactory campaign | Problem | Proposal |
|---|---|---|---|
| S1 | Operating point varied only by **one random load step of up to ±60 % at t = 0**; generation not re-dispatched | Covers few operating conditions; the load step itself excites dynamics | Scale all loads by λ ~ U(0.8, 1.2) with per-load noise, re-dispatch generators proportionally, solve the power flow, reject non-convergent and V < 0.9 p.u. cases |
| S2 | Fault start uniform in **[0, 5] s after the load step** | The system is not in steady state when the fault occurs; pre-fault oscillations carry operating-point information and vary with the random start time | Fault at a fixed time on a settled system (e.g. t = 1.0 s) |
| S3 | **Fixed clearing time** per system (about 0.25 s) | Outcome depends only on operating point and location; few cases near the stability boundary; no CCT information | Clearing time drawn from 0.05-0.40 s (or around the CCT); optionally compute the CCT per case as a regression target |
| S4 | Additional trip of the **"adjacent" line = next line in the list** | Not a physical neighbour; often islands load buses | No second trip by default (N-1); optional trip of a line that shares a bus with the faulted one |
| S5 | Label from PowerFactory's **pole-slip flag**; clearing time later **detected from voltages** | Label criterion differs from the literature; detected clearing is not causal (D9) | Label from rotor-angle separation (360° by default, 180° COI stored too), continuous TSI stored; true event times stored by the simulator |
| S6 | Random k-fold over events | Same fault location and similar operating points in train and test: optimistic | Report three splits: stratified random, **held-out faulted lines**, **held-out loading band** |
| S7 | Snapshot datasets contain duplicated rows (25 / 22) | Duplicates can sit on both sides of a split | Group duplicates (`group_duplicates=True`) |
| S8 | Signals at the simulator step | Real PMUs report 25-60 frames/s with noise | Optional down-sampling to PMU rates and measurement noise for robustness tests |

## Proposed protocol (v2)

1. **Operating point**: λ ~ U(0.8, 1.2); each load × λ × (1 + ε), ε ~ N(0, 0.05);
   generators re-dispatched in proportion to their base dispatch; AC power flow; reject if
   not converged or any bus voltage < 0.9 p.u.
2. **Contingency**: three-phase fault on a random transmission line (transformers
   excluded), at a line terminal chosen at random (engines that support it: random
   position along the line).
3. **Timing**: fault at t = 1.0 s; clearing time ~ U(0.05, 0.40) s by opening the faulted
   line; no second trip unless configured (then a topological neighbour).
4. **Simulation**: until 5 s after clearing, step 10 ms; record bus voltages, angles and
   frequencies, generator angles, speeds and powers.
5. **Labels**: δmax = maximum angle separation between any two generators after the fault;
   unstable if δmax > 360°; store δmax, TSI, the 180° COI flag and the simulator's own flag.
   Runs that fail without a decided outcome are dropped and counted.
6. **Splits and metrics**: stratified random, grouped by faulted line, grouped by loading
   band; accuracy, balanced accuracy, recall on unstable cases, ROC-AUC, with repeated
   cross-validation and corrected intervals.
7. **Provenance**: every scenario parameter, the engine, its version and the seed are
   stored with the dataset.

The same protocol runs on the mock, ANDES, Dynawo and PowerFactory engines, so results
are comparable across simulators. The existing PowerFactory datasets stay available and
documented as "legacy protocol".

## Sources

- Mendeley Data, *Dataset for Transient Stability Assessment of IEEE 39-Bus System*,
  https://data.mendeley.com/datasets/p992nhb8ss/1
- *A Benchmark Graph Dataset for Transient Stability Assessment of the IEEE 9-Bus System*,
  arXiv:2608.18318
- *A unified power-grid representation for reuse across network structures and
  computational tasks*, arXiv:2609.07650
- *LLM-Driven Transient Stability Assessment*, arXiv:2511.20276
- X. Wu et al., *Comparative Analysis of Learning-Based Methods for Transient Stability
  Assessment*, arXiv:2409.02336
- U. Shahzad, *Artificial Neural Network for Transient Stability Assessment: A Review*,
  arXiv:2206.06800
- J. D. Morales, J. V. Milanović, P. N. Papadopoulos, *Analysis of angular threshold
  criteria for transient instability identification in uncertain power systems*, IEEE
  PowerTech 2019, doi:10.1109/PTC.2019.8810465
- *Evaluation of Suitability of Different Transient Stability Indices*, arXiv:2001.03519
- P. Kundur et al., *Definition and classification of power system stability*, IEEE
  Trans. Power Systems 19(3), 2004
