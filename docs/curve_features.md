# e12: too few hours, or missing features? Learning curve and physics features

Analysis only. Data, scripts and logs are on cairo-lab in `~/data/exp/e12_curve_features/`: `phys_features.py`, `lc.py`, `t2_run.py`, `perm.py`, `analyze.py`, `lc_results.csv`, `lc_fits.csv`, `t2_results.csv`, `t2_paired.csv`, `physics_only.csv`, `perm_importance*.csv`, plus the OOF predictions in `lc/` and `t2/oof/`. No grid model, PLAN.md, papers/ or .env.local was touched, and nothing was committed.

Data: v4.1 `line_pre`, 11,159 events (1,325 unstable), 9 operating hours (2 cases each). The seed (20261104), folds, 0.5 threshold and 1,000-resample stratified bootstrap are the same as in e1/e5/e8. The baselines are the e5 LimiX-2 and e1 posttrip30 LightGBM OOF files.

## Verdict: feature-limited, not data-limited
- **Physics features fix most of the problem, even with very few hours.** LightGBM trained on **1** operating hour with the physics features (PR-AUC 0.947) beats LightGBM trained on all 8 hours without them (0.770).
- **More hours without new features help only slowly.** The fitted curve says the current features would need about 20 hours for 0.90 (Kumo) and roughly 90 to 600 for 0.95. These are extrapolations, and probably optimistic (see Test 1).
- **With the physics features, the hour protocol is already above 0.95:**
  - LimiX-2: 0.986 [0.983, 0.989]
  - LightGBM: 0.971
  - the 220 kV miss rate drops from 35.9 % to 5.6 %.
- **Physics alone, with no training, gives PR-AUC 0.962 [0.954, 0.969].** The model is the analytic EEAC margin (clearing time minus estimated CCT), with the rule margin > 0. That is above every pure-ML hour-protocol result from e1 to e8.

## Test 1: learning curve over operating hours (leave-one-hour-out)
Setup:
- For each held-out hour, the model is trained on k of the other 8 hours. Each k uses 10 random subsets (or all subsets when there are fewer), with the same subsets for every model.
- The predictions of the 9 held-out hours are pooled for each replicate.
- The CI comes from a hierarchical bootstrap (replicate × stratified events, 1,000 draws).
- Models:
  - LightGBM: the e1 nested-Optuna best parameters of that outer fold.
  - Kumo Tabular instead of LimiX-2: about 12 s per fit against about 130 s, with hour PR-AUC 0.857 against 0.867.

| k hours | LightGBM PR-AUC | miss 220 / 380 | Kumo PR-AUC | miss 220 / 380 |
|---|---|---|---|---|
| 1 | 0.565 [0.510, 0.616] | 0.74 / 0.56 | – | – |
| 2 | 0.625 [0.568, 0.694] | 0.68 / 0.51 | 0.712 [0.632, 0.766] | 0.54 / 0.36 |
| 4 | 0.689 [0.621, 0.742] | 0.61 / 0.41 | 0.773 [0.716, 0.816] | 0.44 / 0.29 |
| 6 | 0.749 [0.713, 0.778] | 0.56 / 0.37 | 0.836 [0.803, 0.860] | 0.39 / 0.26 |
| 8 | 0.770 [0.748, 0.790] | 0.50 / 0.34 | 0.857 [0.843, 0.871] | 0.37 / 0.25 |

The same curve with the physics features added (`line_pre_phys`):
- LightGBM: 0.947 (k = 1) → 0.960 (2) → 0.965 (4) → 0.968 (6) → 0.972 (8).
- Kumo: 0.978 (2) → 0.982 (4) → 0.984 (6) → 0.986 (8).
- The 220 kV miss rate is 4.5 to 7.7 % at every k.

**Inverse power-law fit.** The curve is PR-AUC(k) = a − b·k^(−c), fitted to the replicate-level points at k ≥ 2 and bounded to a ≤ 1 (Mukherjee et al. 2003; Figueroa et al. 2012, BMC Med. Inform. Decis. Mak. 12:8; Hestness et al. 2017, arXiv:1712.00409). The uncertainty comes from 1,000 bootstrap refits.

| model, features | c | hours for 0.90: point [95 %] | hours for 0.95 |
|---|---|---|---|
| Kumo, line_pre | 0.47 | 20 [14, 31] | 87 [51, 191] |
| LightGBM, line_pre | 0.36 | 83 [51, 243] | 573 [276, ∞); in 2.5 % of the fits it is never reached |

**These are extrapolations far outside the observed range (k ≤ 8), with only 4 to 8 points.** The asymptote a hit its bound of 1.0 in the point fit. That is an assumption that perfect separation is reachable, not a finding. If the true ceiling is below 1, the hour counts grow quickly or the target is never reached. Read them as lower bounds. With the physics features the targets are already reached at k = 1 or 2, so a fit there tells us nothing.

## Test 2: physics features (`phys_features.py`, no new time-domain simulation)
**Network model.** Per case: the ANDES workbook and the ANDES power flow, which is the state TDS starts from. The transient-period network uses the classical model (Kundur 1994, ch. 13):
- loads as constant admittance, as in the case config (PQ p2z = q2z = 1);
- each machine an E' behind xd1;
- the fault as in the engine: at bus2 when position ≥ 0.5, xf = 1e-4 pu;
- the line out after clearing, applied with an exact rank-2 Woodbury update.

**Checks.** The mapping of `fault_line` was asserted against `line|x`. The pre-fault reduced-network P_e matches P_m to within 5e-5 pu in every case.

**Features (41 new):**
- (a) Short-circuit power at both ends before and after the trip, the post/pre ratios, and an islanding flag.
- (b) The top 3 machines by electrical distance |Z_ff + Z_bb − 2Z_fb| + xd1: Sn, H, P/Sn, E' angle relative to the COI, and the distance.
- (c) ΔP during the fault (P_m − P_e at δ0, machine base) × t_c/(2H) for those 3 machines, plus the maximum and the top-5 mean over all machines.
- (d) An EEAC/ZOOMIB critical clearing time (Xue, Van Cutsem & Ribbens-Pavella 1989; Pavella, Ernst & Ruiz-Vega 2000):
  - candidate critical clusters are the top 1 to 5 machines by initial acceleration;
  - the OMIB is built from the during- and post-fault reduced admittance, with intra-cluster angles frozen;
  - the equal-area criterion gives the critical angle, and the during-fault energy integral gives the time;
  - the result is the minimum over candidates, capped at 1.0 s;
  - the features are eeac_cct, eeac_margin = t_c − cct, the cluster size and a flag.

**How good the EEAC CCT is.** Against the e4 bracketed CCTs, Spearman is 0.91 at 220 kV and 0.93 at 380 kV, and the MAE is 0.06 s. The e8 learned CCT regressor reached 0.092 s under the hour protocol.

| model | protocol | baseline PR-AUC / miss 220 / miss 380 | + phys | ΔPR-AUC [95 %] | McNemar fixed / broken, p |
|---|---|---|---|---|---|
| LimiX-2 | hour | 0.867 / 35.9 % / 21.5 % | **0.986** [0.983, 0.989] / 5.6 % / 9.2 % | +0.119 [0.107, 0.131] | 495 / 68, p < 1e-10 |
| | line | 0.934 / 26.9 / 17.8 | 0.988 / 5.9 / 6.6 | +0.054 [0.047, 0.062] | 313 / 44 |
| | random | 0.947 / 23.4 / 15.7 | 0.990 / 6.4 / 6.1 | +0.043 [0.037, 0.049] | 268 / 50 |
| LightGBM (tuned, 30 trials) | hour | 0.775 / 50.0 / 34.4 | 0.971 [0.965, 0.977] / 5.6 / 10.3 | +0.196 [0.178, 0.217] | 627 / 90 |
| | line | 0.845 / 50.5 / 32.7 | 0.976 / 5.6 / 8.9 | +0.131 | 522 / 86 |
| | random | 0.877 / 37.2 / 25.1 | 0.982 / 6.4 / 8.4 | +0.105 | 403 / 88 |

All McNemar p-values are below 1e-10. Under the hour protocol the false-alarm rate also falls, from 2.6 % to 0.6 % for LimiX-2. The 220 kV gap is reversed: 220 kV is now missed less often than 380 kV. LimiX-2 misses 21 of 375 unstable 220 kV events, against 135 before.

**What matters: permutation importance** (LightGBM, hour protocol, drop in PR-AUC on the held-out hour, mean over the 9 folds):
- eeac_margin: 0.564 (sd 0.10)
- accel_max: 0.022
- eeac_cct: 0.007
- clearing time: 0.004
- everything else: ≤ 0.002

Permuting whole groups gives:
- (d) EEAC: 0.64
- (c) acceleration: 0.035
- other line_pre: 0.009
- (b) nearest machines: 0.002
- (a) short-circuit power: ≈ 0

So the gain comes from the system-level energy (OMIB) quantity. The local short-circuit and nearest-machine terms add almost nothing: the electrically nearest machines are mostly large equivalent units with P ≈ 0.

## Caveats
- **The EEAC CCT is a model, not ground truth.** It uses the classical model, so it ignores AVR, governor and subtransient dynamics. The remaining errors are 380 kV misses (9 to 10 %) and events whose e4 bracket failed (674 of them have an EEAC estimate, but there is no truth to check it against).
- **These are still only 9 hours of one grid model.** The physics features should transfer, because they are computed per state. The learning-curve flatness with the physics features (0.947 → 0.972) supports that, but only within this dataset.
- **The LightGBM learning curve reuses per-fold parameters** tuned on all 8 other hours. That is slightly optimistic for small k.
- **Kumo replaced LimiX-2 in Test 1** for speed. LimiX-2 is used in Test 2.

## What v5 needs
- **Compute the EEAC margin and acceleration features per event.** They come from the PF case alone, at milliseconds per event. Make them part of the standard `line_pre` builder and do not rely on more hours to close the gap.
- **Hours: about 20 to 30 well-spread operating hours are enough**, for honest unseen-hour CIs and to cover the operating envelope. The current features would need tens to hundreds, an extrapolation that is probably optimistic. With the physics features the curve is flat from 2 hours, so extra hours mainly help to measure performance, not to raise it.
- **Next feature work should target the remaining 380 kV misses and the failed-bracket events.** Options are AVR/field-voltage terms and a multi-swing check, an extension of SIME. Local short-circuit power should not be the focus.
