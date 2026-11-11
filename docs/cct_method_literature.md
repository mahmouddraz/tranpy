# CCT labelling method: literature check

Scope: CCT labels from bracketing of the clearing time (lo 50 ms, hi 1.0 s, 10 ms width, 180° rotor-angle deviation from the COI, 7 s window) on the German 220/380 kV grid in ANDES.
Search date: 2026-11-11. "Verified" means I read the bibliographic record or the source page. "Not verified" means only secondary listings were found.

## 1. CCT by repeated TDS with bisection

### Findings
- **Zhao, You, Mandich, Zhu, Li, Su, Liu, Jiang, Yuan, Zhang, Tan**, "A Fast and Accurate Transient Stability Assessment Method Based on Deep Learning: WECC Case Study", IEEE ISGT 2022, DOI 10.1109/ISGT50606.2022.9817531 (OSTI 1959568). Builds the training database by binary search of the clearing time with TDS, at 1 ms resolution. The abstract gives no search bounds and no rule for out-of-range cases.
- **Wu, Wang, Wang, Caines, Liu**, "Comparative Analysis of Learning-Based Methods for Transient Stability Assessment", arXiv:2409.02336, 2024 (preprint). Uses bisection in PSAT with a 180° criterion and notes that some works use 360°. The stopping tolerance is not stated, but the worked example brackets the CCT to about 0.2 ms. **It excludes samples with "zero and infinite CCTs": 1,500 samples become 1,489.** These are the unstable-at-lower-bound and stable-at-upper-bound cases. This is the only explicit statement of out-of-range handling I found, and the cases are dropped.
- **Hamilton, Papadopoulos**, "Using SHAP Values and Machine Learning to Understand Trends in the Transient Stability Limit", arXiv:2302.06274, 2023 (search snippets say it was published in IEEE Trans. Power Systems; I did not verify a journal DOI). Predicts CCT per busbar on IEEE 39-bus. According to the search snippet, it caps the CCT search at 1.40 s, "beyond which the system is considered very far from instability". I could not open the full text (the PDF was too large), so how capped samples are encoded is not verified.
- Textbooks: **Kundur**, *Power System Stability and Control*, McGraw-Hill/EPRI, 1994 (bibliographic record only). **Pai**, *Energy Function Analysis for Power System Stability*, Kluwer, 1989, ISBN 0-7923-9035-0, https://link.springer.com/978-0-7923-9035-0. Chapters 5–6 of Pai cover PEBS and controlling-UEP CCT estimation. I could not access either text to quote a passage on repeated-simulation CCT, so cite them only for definitions (CCT, energy methods), not for the bisection procedure.

### Verdict
- **Established (high confidence):** CCT by TDS plus binary search, used as the ML label (Zhao et al. 2022; Wu et al. 2024). The 180° criterion is common (Wu et al.; it is also ANDES's default `ddelta_limit=180`).
- **Our own choice (state it explicitly):** the 50 ms to 1.0 s bracket, the 10 ms width (coarser than the 1 ms in Zhao et al.; justify it as at most 10 ms of label uncertainty, which is below typical relay and breaker timing granularity), the 7 s window, and the COI reference.
- **Handling of out-of-range cases:** in the literature I found, these cases are either dropped (Wu et al.) or capped (Hamilton & Papadopoulos, per snippet). I found no standard. Keeping them as censored observations would be a departure from that practice (see section 3).

## 2. Numerical failures at switching events

### Findings
- **Stott, B.**, "Power system dynamic response calculations", Proc. IEEE 67(2):219–241, 1979, DOI 10.1109/PROC.1979.11233. The classic survey of simultaneous implicit (trapezoidal) versus partitioned TDS. Verified as a bibliographic record only.
- **Astic, Bihain, Jerosolimski**, "The mixed Adams-BDF variable step size algorithm to simulate transient and long term phenomena in power systems", IEEE Trans. Power Syst. 9(2):929–935, 1994, DOI 10.1109/59.317654. Variable-step DAE solver, as used in EUROSTAG.
- **Milano, F.**, *Power System Modelling and Scripting*, Springer, 2010, DOI 10.1007/978-3-642-13669-6. It has a "Time Domain Analysis" chapter. I did not read its discontinuity treatment, so cite it only generically.
- **Cui, Li, Tomsovic**, "Hybrid Symbolic-Numeric Framework for Power System Modeling and Analysis", IEEE Trans. Power Syst. 36(2):1373–1384, 2021, DOI 10.1109/TPWRS.2020.3017019 (the ANDES paper).
- **ANDES source** (CURENT/andes master, `andes/routines/tds.py` and `daeint.py`, read 2026-11-11):
  - TDS config: `method='trapezoid'`, `tol=1e-4`, `max_iter=15`, `tstep=1/30`, `fixt=1`, `shrinkt=1`, `honest=0` (do not update the Jacobian every iteration), `linesearch=1`, `g_scale=1`, `reset_tiny=1`, `chatter_iter=4`, `criteria=1`, `ddelta_limit=180`, `check_conn=1`, `refresh_event=0`. The adaptive methods `trap_adapt` and `qndf` use LTE control and need `fixt=0` (with `dtmax`, `dtmin_adapt`, `abstol=1e-6`, `reltol=1e-3`).
  - On non-convergence with `fixt=1, shrinkt=0` the run is busted immediately. Otherwise `h *= 0.9` per failed attempt. When `h < deltatmin = min(tcycle/500, deltatmax/20)` (about 40 µs at 50 Hz), the run stops with "Time step reduced to zero. Convergence not likely." and `busted=True`. After a converged step, `h` halves if `niter >= 15`, grows ×1.1 if `niter <= 6`, and is capped at `tstep` when `fixt=1`. Steps are truncated so that they land on switch times.
  - Near events, the Jacobian is forced to refresh within 0.1 s of the last switch. **There is no separate re-initialisation of the algebraic variables after a switch.** The post-event algebraic state comes only from the ordinary Newton iteration started at the pre-event values.
- **Ali, Cui**, "Homotopy-Based Re-Initialization for Switched DAEs in Power System Transient Simulation", arXiv:2606.12803, 2026 (submitted to IEEE PES Letters). Frames post-event non-convergence in simultaneous DAE solvers as a poor post-event initial guess. Argues that standard restoration (step reduction and similar) can fail at discontinuities. Proposes homotopy-continuation re-initialisation. Cui is the ANDES author, so this is direct evidence that our failure mode is known and not yet solved in ANDES.
- **Hao, Hu, Zhong, Wang**, "Event-Structured Physics-Informed Neural Networks for Differentiable Critical Clearing Boundaries", arXiv:2607.27681, 2026. A search-engine summary says it aligns events to the time grid, applies an algebraic correction after switching, and records non-convergent rollouts as solver failures, not stability outcomes. **I did not verify this in the full text** (the abstract does not say it), so check the paper before citing it for this point.
- A ProQuest-indexed paper (ProQuest docview 3120654279; I could not retrieve title or authors) reportedly shows that the Newton failure at clearing grows with clearing time, because the pre-clearing guess is further away, and proposes a zero-length step solve for the post-clearing algebraic state. **Unverified; do not cite until the record is retrieved.**
- **±2 ms shift of the trial clearing time:** nothing found. No source I located uses or discusses perturbing the clearing time to sidestep a non-convergent trial.

### Verdict
- **Established:** implicit trapezoidal simultaneous solution (Stott 1979; Milano 2010; ANDES); step reduction on non-convergence; that post-event algebraic re-initialisation is the root issue (Ali & Cui 2026, preprint).
- **Practice on failed cases in TSA datasets:** rarely reported. Wu et al. drop out-of-range cases, and no paper I found reports its solver-failure rate. Reporting the failure rate and its distribution (by fault location and loading) would be more transparent than common practice. That is a reporting contribution, not a methodological novelty.
- **±2 ms shift:** our own heuristic with no precedent found (medium confidence that none exists). If you use it, present it as a sensitivity check (the CCT bracket is 10 ms, so a 2 ms shift stays within label resolution) and report how many labels depend on it.

## 3. Interval-censored labels

### Findings
- **Turnbull, B. W.**, "The empirical distribution function with arbitrarily grouped, censored and truncated data", JRSS B 38(3):290–295, 1976, DOI 10.1111/j.2517-6161.1976.tb01597.x. The NPMLE for interval-censored data.
- **Barnwal, Cho, Hocking**, "Survival regression with accelerated failure time model in XGBoost", J. Comput. Graph. Stat. 31(4):1292–1302, 2022, DOI 10.1080/10618600.2022.2067548 (arXiv:2006.04920). AFT loss with left, right and interval censoring (`label_lower_bound`, `label_upper_bound`). This maps directly onto bisection brackets: [lo_stable, hi_unstable], plus [1.0 s, +inf) for stable-at-upper-bound and (0, 50 ms] for unstable-at-lower-bound.
- **Power-system or TSA use of censored or interval CCT labels:** searches combining survival analysis, right-, interval- and censored with CCT, transient stability, time to instability and hazard found **no paper**. The closest works:
  - Mešter, "A Time-to-Boundary Margin for Transient Stability: Unifying Critical Clearing Time and Operating-Point Drift", arXiv:2607.02070, 2026. Defines a first-passage-time margin to the "survival boundary". This is survival-style wording for a dynamical margin; it uses no censored statistics. Conceptually close, methodologically unrelated.
  - Wu et al. 2024 and Hamilton & Papadopoulos 2023 (section 1) drop or cap the very observations that censoring would keep.
  - Probabilistic or risk-based TSA treats clearing time as random against a CCT threshold (e.g. arXiv:2205.03629, "A Probabilistic Framework for Power System Large-Disturbance Global Instability Risk Assessment in the Presence of Renewable Wind Generation"). This is not censored regression.

### Verdict
- **Established:** interval-censored estimation and regression in general statistics (Turnbull 1976) and in gradient boosting (Barnwal et al. 2022, available in XGBoost).
- **Appears novel (medium confidence, given my search coverage: web, arXiv, IEEE abstracts via search engines, not IEEE Xplore full-text search):** treating bisection CCT brackets as interval-censored labels, and out-of-range cases as left- or right-censored, in ML-based TSA. Phrase it as "to our knowledge", and re-run an IEEE Xplore and Scopus query before submission.

## 4. Selection bias from dropping failed or censored runs

### Findings
- **Heckman, J. J.**, "Sample selection bias as a specification error", Econometrica 47(1):153–161, 1979, JSTOR 1912352 (DOI 10.2307/1912352 inferred from JSTOR; not confirmed).
- **Zadrozny, B.**, "Learning and evaluating classifiers under sample selection bias", ICML 2004, DOI 10.1145/1015330.1015425.
- **Little & Rubin**, *Statistical Analysis with Missing Data*, 3rd ed., Wiley, 2019, ISBN 9780470526798 (no DOI found). Covers MCAR, MAR and MNAR. Solver failures that depend on clearing time and stress are a form of non-random missingness.
- Power systems: I found no paper that analyses the bias from dropping non-converged or out-of-range TDS cases in TSA datasets. The closest is Wu et al. 2024, which drops them without analysis. A 2025 data-generation paper (arXiv:2512.06369) discusses infeasible operating points (non-converged power flow) in sampling; I did not read how it treats them.

### Verdict
- **Established:** selection bias and missing-not-at-random theory (Heckman; Zadrozny; Little & Rubin).
- **Gap:** applying it to TSA datasets. Failed searches are likely concentrated near the stability boundary or at highly stressed points. Dropping them biases both the CCT distribution and the test metrics. Quantifying this is a defensible small contribution.

## Recommendations

### Method section
1. Define CCT and the stability criterion (180° from the COI, 7 s), citing the ANDES paper and noting that 180° versus 360° both appear (Wu et al. 2024).
2. State the bracket (50 ms to 1.0 s, 10 ms) as our choice, and contrast it with 1 ms in Zhao et al. 2022 and the 1.40 s cap in Hamilton & Papadopoulos (after verifying the cap in the full text).
3. Store every label as an interval [t_stable, t_unstable], with right-censored [1.0, inf) and left-censored (0, 0.05]. Train a point-regression baseline on midpoints with the censored cases dropped (the literature practice) and an XGBoost AFT model with interval censoring (Barnwal et al. 2022), and compare them. Evaluate with censoring-aware metrics (interval hit rate, C-index).
4. Report solver failures as a third outcome, not as unstable. Give their count, location and stress distribution, and a selection-bias analysis (Heckman / Little & Rubin framing): metrics with failed cases dropped, compared with metrics after repair.
5. Describe the ±2 ms retry, if kept, as an explicitly ad-hoc robustness step with no literature precedent, and report the share of labels that used it.

### Solver fix: test matrix (50 failed + 50 control searches)
Rerun each search unchanged first to confirm the failure is deterministic. Then change one factor at a time:
- A. `tstep`: 1/30 → 1/60 → 1/120 (with `fixt=1, shrinkt=1`).
- B. `honest=1` (fresh Jacobian every Newton iteration; most relevant right after the switch).
- C. `max_iter`: 15 → 30 or 50, and `tol` 1e-4 → 1e-3 as a looser check (report the effect on control CCTs).
- D. `method='trap_adapt'` or `'qndf'` with `fixt=0` (LTE control; set `dtmin_adapt` if the steps collapse).
- E. `linesearch` 1 versus 0 and `g_scale` 1 versus 0 (both are in the config; their effect at events is untested in the literature).
- F. Only then, the ±2 ms clearing-time shift.

Acceptance criteria: the fraction of the 50 failed searches recovered, and the maximum |ΔCCT| on the 50 controls, which must stay within the 10 ms bracket (0 bracket flips is ideal). Record runtime per search. If none of these recovers the failures, the root cause is post-event algebraic re-initialisation, which ANDES does not do separately (Ali & Cui 2026). A post-switch algebraic-only Newton solve (or PFlow-style re-solve) is then the principled fix.
