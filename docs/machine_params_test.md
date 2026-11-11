# F1: size- and type-dependent GENROU parameters (e10)

Outputs on cairo-lab: `~/data/exp/e10_machine_params/` (scripts `scripts/01_params.py` to `07_analyse.py`; `param_table.csv`, `machine_assignment.csv`, `flat_runs.json`, `sample.parquet`, `results.txt`, `results_events.parquet`, `cct_compare.csv`; modified case copies in `prep/`). Grid models and v4.1 inputs were not modified.

## 1. Parameter table

The v4.1 GENROU values were already split by class: thermal had H 4.5 s, xd 1.8, xq 1.7, x'd 0.30, x'q 0.55, x''d 0.25, T'd0 6 s and T'q0 0.8 s; hydro had H 3.0 s, xd 1.0 and xq 0.65. D was 0 for all classes. They did not depend on size. Each e10 machine is classified by the dominant fuel at its bus, weighted by MVA, from the prep plant register, and by the Sn of its equivalent machine (small means below 100 MVA). Every value comes from Kundur (1994):

| class | n (18 cases) | H [s] | xd / xq | x'd / x'q | x''d = x''q | xl, ra | T'd0 / T'q0 / T''d0 / T''q0 [s] | source |
|---|---|---|---|---|---|---|---|---|
| steam/gas >= 100 MVA (Braunkohle, Steinkohle, Erdgas, other gases) | 606 | 3.5 | 1.81 / 1.76 | 0.30 / 0.65 | 0.23 | 0.15, 0.003 | 8.0 / 1.0 / 0.03 / 0.07 | Kundur Ex. 3.1/3.2 (555 MVA 2-pole unit) |
| steam/gas < 100 MVA | 326 | 2.5 | as above | as above | 0.23 | as above | as above | H: lower end of the 2-pole range 2.5-6 s, Kundur Table 3.2 |
| hydro >= 100 MVA (Wasser, Pumpspeicher) | 88 | 3.0 | 1.0 / 0.65 | 0.30 / 0.65 | 0.25 | 0.15, 0.003 | 6.0 / 0.5 / 0.035 / 0.05 | H: middle of the hydro range 2-4 s, Table 3.2; reactances inside the Table 4.3 hydro ranges (unchanged from v4.1) |
| hydro < 100 MVA | 88 | 2.0 | as above | | | | | H: lower end of the hydro range, Table 3.2 |

The table has no nuclear class because Germany's nuclear units were shut down in 2023. ANDES requires x''q = x''d, so x''q is set to 0.23 rather than Kundur's 0.25.

**Damping:** D = 0 is kept. GENROU models the damper windings explicitly (Kundur Ch. 5; IEEE Std 1110-2019, model 2.2), so an extra D term would count the damping twice.

**What is not cited:** the 100 MVA size threshold, and the choice of the lower end of each range for small units, are my judgement. Kundur gives H by machine type and speed, not by MVA. The other references the task named (Anderson & Fouad, IEEE 1110 tables, the ENTSO-E/IEEE PES typical-data reports) were not looked up in this session, and no numbers are taken from them.

**Net effect:** H drops for every thermal unit (by 22 % for large units and 44 % for small ones) and for small hydro (by 33 %). x'd stays at 0.30. T'd0 rises from 6 s to 8 s. All changes point towards less stable machines.

## 2. Initialisation and flat runs (step 5)

All 18 cases, with both the old and the new parameters, were initialised and run for 10 s with no disturbance (`flat_runs.json`):
- The power flow converges in every run.
- The largest initial residual is 3e-16.
- The largest speed deviation is 0.
- The largest rotor-angle drift is 1e-6 deg.

The runs are clean. During the fault runs ANDES prints "Initialization FAILED" because HYGOV `gtpos` is clamped at GMAX under the loading scaling. The same message appears with the **original** v4.1 case (6 of 8 test events on baltic_import_low_load_t10), so it is not caused by these parameters. It is still a v4.1 finding to look into.

**Control run:** 40 events of high_load_t10 rerun with the original xlsx reproduce the v4.1 labels exactly (0 of 40 differ).

## 3. Sample design (seed 20261108; 2,000 distinct events)

| stratum | rule | n |
|---|---|---|
| A | all unstable 220 kV events that LimiX-2 missed (hour protocol) | 135 |
| B | the same number of caught unstable 220 kV events, drawn at random | 135 |
| C | all missed unstable 380 kV events (contrast group) | 204 |
| D | the same number of caught unstable 380 kV events, drawn at random | 204 |
| E | clearing time within +-50 ms of the CCT bracket for its (case, line, terminal); all remaining events of this kind (540 in total, 162 already in A-D) | 378 |
| F | random stable events, up to 2,000 in total | 944 |

Everything else was kept identical: the same 18 cases, scenario draw, fault, clearing time, 6.6 s window, step size and label rule, through `run_sim_local.py --ids`. 17 events (0.9 %) ended in numerical failure and are excluded, which leaves 1,983.

## 4. Label flips

| group | n | unstable old | unstable new | unstable to stable | stable to unstable (share of stable) |
|---|---|---|---|---|---|
| A missed 220 kV | 133 | 133 | 133 | 0 | - |
| B caught 220 kV | 133 | 133 | 133 | 0 | - |
| C missed 380 kV | 199 | 199 | 199 | 0 | - |
| D caught 380 kV | 203 | 203 | 203 | 0 | - |
| E near CCT | 375 | 74 | 334 | 0 | 260 (86 %) |
| F random stable | 940 | 0 | 31 | 0 | 31 (3.3 %) |
| **all** | 1,983 | 742 | 1,033 | **0** | **291 (23 % of the sample's stable events)** |

By voltage, the stable-to-unstable share is 18 % at 220 kV and 26 % at 380 kV. By clearing band it is 0 % for 0-0.1 s, 0.5 % for 0.1-0.2 s, 20 % for 0.2-0.3 s, 46 % for 0.3-0.4 s, 30 % for 0.4-0.5 s and 29 % for 0.5 s or more.

By separating-unit size, the unstable-to-stable flips are 0 in every band: below 100 MVA, 100-300 MVA and 300 MVA or more. That size is known only for events that were unstable and traced under v4.1. The separating unit of the new stable-to-unstable events was not extracted.

The flips go only one way. Lowering H, with x'd unchanged, never stabilises an event. It turns borderline stable events unstable, and these are almost all events near the CCT with clearing times above 0.2 s.

## 5. Does the 220 kV miss excess shrink?

LimiX-2's v4.1 predictions were scored against both label sets, without retraining:

| | 220 kV missed / unstable | 380 kV missed / unstable | false alarms 220 / 380 |
|---|---|---|---|
| v4.1 labels (sample) | 133 / 276 (48 %) | 199 / 466 (43 %) | 24 / 62 |
| e10 labels (sample) | 196 / 354 (55 %) | 357 / 679 (53 %) | 9 / 7 |

Weighted back to the full v4.1 population (inverse sampling fraction per cell of voltage x label x prediction; approximate, because it ignores the near-CCT stratum):

| | 220 kV miss rate | 380 kV miss rate | odds ratio, 220 vs 380 |
|---|---|---|---|
| v4.1 | 0.36 | 0.21 | 2.06 (the audit reports 2.24) |
| e10 | 0.71 | 0.62 | 1.50 |

The odds ratio falls, but not because the 220 kV misses are fixed. **None of the 133 missed 220 kV events change label.** The ratio falls because about 300 new borderline unstable events appear at both voltage levels, and the frozen model calls most of them stable. The miss rate rises at both levels and the 220 kV gap becomes relatively smaller. This is dilution, not an explanation of the excess. The new parameters do not support the idea that generic parameters on small units at 220 kV cause the misses.

## 6. CCT shift (step 4; 50 lines x 2 terminals, high_load_t10 and low_load_t10)

- 68 brackets stay bracketed under both parameter sets.
- **Every one of them shifts down.**
  - median shift: -96 ms (IQR -141 to -59 ms)
  - median ratio new/old: 0.75
  - range of the shifts: -15 to -282 ms
- Median shift by case and voltage:

  | case | 220 kV | 380 kV |
  |---|---|---|
  | high_load_t10 | -59 ms | -82 ms |
  | low_load_t10 | -141 ms | -96 ms |

- Status changes: 4 brackets go from "above" to bracketed, and 3 bracketed runs fail numerically.

The size of the shift fits the drop in H: with x'd unchanged, the CCT scales roughly with the square root of H, and sqrt(3.5/4.5) = 0.88 while sqrt(2.5/4.5) = 0.75.

## 7. Effect on the e12 conclusion (EEAC-type CCT feature)

- **Do labels near the CCT flip?** Yes, massively: 86 % of the stable events within +-50 ms of the bracket become unstable. Because all CCTs shift down by about 60-140 ms, the set of borderline events changes almost completely. Events that were borderline under v4.1 become clearly unstable, and events that were comfortably stable at 0.3-0.5 s become borderline.
- **Does this overturn e12?** Not necessarily. An EEAC estimate takes H and x'd as inputs. If it were recomputed with the same machine data as the simulator, it should follow the shift, because the shift is mainly a change of H. Two parts of e12 cannot be carried over as they stand:
  - The 0.867 to 0.986 lift and the removal of the 220 kV gap were measured with the v4.1 labels and v4.1 machine data.
  - Under e10 parameters both the feature and the labels change, and the borderline population is different.
- **What to do:** before the lift is reported as robust, recompute the EEAC feature and the e12 evaluation on v5 labels with the adopted machine data. The fact that the 220 kV misses keep their labels here suggests that their cause, which e12 addresses, does not depend on the parameters.

## Verdict and recommendation

- **How many labels flip:** 291 of 1,983 (14.7 %). All 291 go from stable to unstable, and 0 go from unstable to stable. The flips are concentrated near the CCT, where 260 of 301 stable events flip; 3.3 % of random stable events flip.
- **Effect on the misses:** the new parameters do not resolve the 220 kV miss excess.
- **Adopt for v5: yes, but as a correction to the model, not as a fix for the misses.**
  - The v4.1 thermal H of 4.5 s is above Kundur's example for a large 2-pole unit (3.5 s).
  - Small units at the lower end of the range are physically plausible.
  - Both change the stability margins materially: CCTs are about 25 % shorter, and borderline labels shift.
- **Before adopting:**
  1. Find cited size-dependent H values (Anderson & Fouad's inertia-versus-rating data, or the ENTSO-E inertia reports) to replace the 100 MVA step, which is my own choice.
  2. Rerun the label-generation and CCT bracketing on v5 with the adopted table.
  3. Investigate the pre-existing HYGOV GMAX clamp warnings during initialisation.

---

# F1b: ENTSO-E inertia per fuel (e10/f1b)

Outputs are in `~/data/exp/e10_machine_params/f1b/`:
- parameter table and case copies: `scripts/08_params_f1b.py`, `param_table.csv`, `machine_assignment.csv`, `prep/`;
- reruns and their analysis: `sim/`, `results.txt`, `results_events.parquet`;
- CCT check: `cct_*/`, `cct_compare3.csv`;
- EEAC recompute: `eeac/`, `eeac_pr_auc.csv`.

The clamp check is in `~/data/exp/e10_machine_params/hygov/`.

## Parameters

**Inertia.** H comes from ENTSO-E, *Frequency stability in long-term scenarios and relevant requirements* (RoCoF report v17, 2020), Table 1, via `grid_case.ENTSOE_INERTIA`. This is the same source as the CE neighbour equivalents.
- The H of each GENROU is the MVA-weighted mean over the synchronous plants at its bus. In practice every bus has a single fuel.
- I found no cited source for a size split, so there is none.

| plant-register fuel | ENTSO-E type | machines (18 cases) | H [s] (v4.1) |
|---|---|---|---|
| Braunkohle | B02 lignite | 144 | 3.8 (4.5) |
| Steinkohle | B05 hard coal | 222 | 4.2 (4.5) |
| Erdgas | B04 gas | 502 | 4.2 (4.5) |
| andere Gase, Grubengas | B03 coal-derived gas | 56 | 4.2 (4.5) |
| Mineralölprodukte | B06 oil | 4 | 4.3 (4.5) |
| nicht biogener Abfall | B17 waste | 4 | 3.8 (4.5) |
| Pumpspeicher | B10 pumped storage | 120 | 3.5 (3.0) |
| Wasser | B11 run-of-river | 56 | 2.7 (3.0) |

**My mapping choices:**
- German river hydro is treated as run-of-river.
- Sewage sludge is mapped to biomass.
- Heat, hydrogen and gas-pressure expanders are mapped to "Other". None of these is the dominant fuel at any bus.

**Reactances and time constants** are the Kundur values from F1 (steam and gas). Hydro keeps the v4.1 values. D = 0.

**Effect on kinetic energy:** German kinetic energy is 0.91 of v4.1. In F1 it was lower still, because every thermal unit and every small hydro unit lost H.

**Checks:**
- Flat runs and initialisation are clean for all 18 cases: residual ≤ 3e-16, speed deviation 0, angle drift ≤ 1e-6 deg (`f1b/flat_runs.json`).
- 26 of the 2,000 sample events ended in numerical failure; 1,974 remain.

## Label flips: v4.1, F1 and F1b side by side

| group | n (F1b) | F1: unstable→stable | F1: stable→unstable | F1b: unstable→stable | F1b: stable→unstable |
|---|---|---|---|---|---|
| A missed 220 kV | 129 | 0 | - | 2 (1.6 %) | - |
| B caught 220 kV | 131 | 0 | - | 1 | - |
| C missed 380 kV | 197 | 0 | - | 8 (4.1 %) | - |
| D caught 380 kV | 203 | 0 | - | 1 | - |
| E near CCT | 371 | 0 | 260 (86 %) | 0 | 103 (35 %) |
| F random stable | 943 | - | 31 (3.3 %) | - | 2 (0.2 %) |
| **all** | 1,974 | **0** | **291 (23 % of stable)** | **12 (1.6 % of unstable)** | **105 (8.5 % of stable)** |

**Stable-to-unstable share by voltage:**

| | 220 kV | 380 kV |
|---|---|---|
| F1 | 18 % | 26 % |
| F1b | 4.1 % | 10.8 % |

F1b unstable-to-stable is 1.1 % at 220 kV and 1.9 % at 380 kV.

**F1b by clearing band:**

| clearing band | stable→unstable | unstable→stable |
|---|---|---|
| < 0.2 s | 0 | 0 |
| 0.2–0.3 s | 1.9 % | 0 |
| 0.3–0.4 s | 17.6 % | 5.3 % |
| 0.4–0.5 s | 14.4 % | 0 |
| ≥ 0.5 s | 12.5 % | 0.9 % |

**F1b near the CCT and by unit size:**
- Near the CCT: 103 stable events flip to unstable (35 %) and 8 unstable events flip to stable (3.4 %).
- Away from the CCT: 0.2 % flip to unstable and 0.8 % flip to stable.
- By separating-unit size (unstable events only): 1.4 % flip to stable below 100 MVA, 1.9 % at 100–300 MVA and 1.5 % at 300 MVA or more. Size makes no difference.

Under F1b the labels move in both directions, because H rises for pumped storage and falls slightly for thermal units. All flips except 6 are within ±50 ms of the CCT.

## LimiX-2 (v4.1 model, not retrained)

Weighted to the v4.1 population as in section 5:

| labels | 220 kV miss rate | 380 kV miss rate | odds ratio, 220 vs 380 |
|---|---|---|---|
| v4.1 | 0.36 | 0.21 | 2.04 |
| F1 | 0.71 | 0.62 | 1.50 |
| F1b | 0.50 | 0.42 | 1.33 |

On the sample, 127 of the 129 missed 220 kV events are still unstable under F1b, and the model still misses them. As in F1, the odds ratio falls only because new borderline unstable events appear and the model misses them at both voltage levels. The 220 kV miss excess is not explained by machine parameters.

## CCT shift (same 50 lines × 2 terminals)

| | bracketed under both | median shift | IQR | range | median ratio | shifted up / down by > 10 ms |
|---|---|---|---|---|---|---|
| F1 | 68 | −96 ms | −141 to −59 | −282 to −15 | 0.75 | 0 / 68 |
| F1b | 68 | −26 ms | −45 to −15 | −104 to +37 | 0.94 | 4 / 59 |

Median F1b shift by case and voltage:

| case | 220 kV | 380 kV |
|---|---|---|
| high_load_t10 | −22 ms | −15 ms |
| low_load_t10 | −30 ms | −30 ms |

Bracket status changes from v4.1 to F1b:
- 4 brackets go from "above" to bracketed;
- 3 bracketed runs become failed;
- 3 failed runs become bracketed.

## The e12 question: does the EEAC margin still track the labels?

I recomputed the e12 EEAC/ZOOMIB CCT with `scripts/30_eeac.py`, which is `phys_features.py` with only the input and output folders changed. It was run with the F1 and the F1b machine data. The table gives the PR-AUC of the margin (clearing time − EEAC CCT) on the sample:

| subset | EEAC data | v4.1 labels | F1 labels | F1b labels |
|---|---|---|---|---|
| all (1,974) | v4.1 | 0.941 | 0.986 | 0.942 |
| | F1 | 0.929 | **0.992** | 0.937 |
| | F1b | 0.949 | 0.990 | **0.957** |
| near CCT (530) | v4.1 | 0.620 | 0.964 | 0.710 |
| | F1 | 0.581 | **0.986** | 0.710 |
| | F1b | 0.644 | 0.975 | **0.774** |
| 220 kV (705) | v4.1 | 0.969 | 0.996 | 0.970 |
| | F1b | 0.967 | 0.997 | **0.973** |

The F1 columns near the CCT are inflated: 92 % of those events are positive under F1 labels.

**How the EEAC estimate moves:**
- The EEAC CCT shifts by a median −39 ms with F1 data and −11 ms with F1b data. This is smaller than the simulated shifts (−96 and −26 ms), so the EEAC follows the direction of the shift but understates its size.
- Borderline set (|EEAC margin| ≤ 50 ms): 405 events with v4.1 data, 423 with F1b data and 278 with F1 data. 348 events overlap between v4.1 and F1b; only 160 overlap between v4.1 and F1.

**Answer:**
- **Yes, the physics feature still tracks the new labels.** Recomputed with matching machine data, the margin scores as well or better against the matching labels. For F1b on all events the PR-AUC is 0.957, against 0.941 for v4.1 on v4.1 labels. Near the CCT it is 0.774 against 0.620.
- **F1b leaves the e12 conclusion intact.** Its labels and borderline set are close to v4.1, with 86 % of the v4.1 borderline events still borderline.
- **F1 would change the borderline population substantially.** The e12 LimiX-2 lift (0.867 to 0.986) would need to be re-measured.
- **Either way the feature must be computed from the same machine data as the labels.** Using the v4.1 EEAC with F1b labels costs about 0.015 PR-AUC overall and 0.06 near the CCT.
- The e12 retraining itself was not repeated.

## Hydro governor clamp ("Initialization FAILED")

**Which units and cases.** I parsed ANDES' own clamp table (`hygov/clamp_log.csv`) for 18 cases × operating points 0.9, 0.95, 1.0, 1.05 and 1.1:
- **At operating point 1.0:** no clamp and no "Initialization FAILED" in any case.
- **At 1.05–1.1:** HYGOV gates sit at GMAX. This affects up to 14 cases and 61 units, by up to 6.9 % of the rating.
- **At 0.9–0.95:** HYGOV gates sit at GMIN (up to 6.5 %), and TGOV1 valves sit at VMIN. The VMIN clamp affects up to 144 units in 14 cases, by up to 8.3 %.
- **Neighbour-equivalent governors** clamp at VMIN by at most 2 %.
- **low_load** never clamps.
- **Worst cases:** high_load and winter_peak, with 21–26 HYGOV units affected.

**Why.**
- **Ruled out:**
  - Setpoint above Pmax and dispatch above rating are ruled out: every hydro unit runs at about 0.76 of its rating.
  - Wrong default limits are ruled out too.
- **Cause:**
  - `_add_german_governors` (grid_case.py, around lines 676–688) builds each German governor's limits as the base-dispatch setpoint ± that unit's FCR reserve share, a band of about ±1.4 % of rating.
  - The engine (`andes.py`, around line 360) then scales PQ and PV power by `operating_point` (0.9–1.1) without moving the limits.
  - Any event with |operating point − 1| above about 1.4 % therefore starts with the gate or valve pinned at a limit, and mechanical power does not match electrical power.
  - Almost every random event is affected.

**Effect, tested on copies** (`scripts/22_hygov_fix_test.py`; 30 events per test, v4.1 machine data). Each event was run twice:
- once on the unmodified case;
- once on a copy whose German governor limits are shifted by (operating point − 1) × the unit's base setpoint.

| test | init failures, original → fixed | pre-fault bus-angle drift over 1 s, original | same, fixed | label changes |
|---|---|---|---|---|
| winter_peak_t10, op 1.03–1.1 (HYGOV GMAX) | 30/30 → 6/30 | 0.4–4.2 deg | ≤ 0.29 deg | 1/30: id 125, stable → unstable |
| high_load_t10, op 0.9–0.95 (TGOV1 VMIN, HYGOV GMIN) | 22/22 → 6/22 | 0.8–3.3 deg | ≤ 0.36 deg | 1/22: id 82, unstable → stable |

**What the test shows:**
- With the original cases, the system drifts before the fault because generation and load do not balance mechanically. In high_load_t10 the bus angles end 30–120 deg apart in absolute terms between the two runs over 6.6 s, which is a frequency drift; rotor separation changes by a few degrees.
- Labels change only for borderline events: 2 of 52 (4 %), in both directions.
- With limits fixed, the remaining init messages come from other clamps (neighbour TGOV1 VMIN) that my shift did not touch.

**Proposed fix for v5:** in the engine, after scaling by `operating_point`, shift the German governor limits by (operating point − 1) × setpoint. For HYGOV the shift is (operating point − 1) × P/Sn/At on GMAX and GMIN; for TGOV1 it is (operating point − 1) × P/Sn on VMAX and VMIN, with the minimum limit kept at 0 or above. Rebuilding the governors from the scaled dispatch would work equally well. The same applies to the neighbour equivalents. Then assert that there is no clamp at initialisation.

This is a code change in `src/tranpy/simulation/andes.py` or `grid_case.py`. It is not made here.

## Final recommendation for v5

1. **Adopt the F1b table:** ENTSO-E RoCoF v17 Table 1 inertia per fuel, Kundur Example 3.1/3.2 reactances and time constants for steam and gas units, the v4.1 hydro reactances, and D = 0.
   - Every value is sourced, and the inertia matches the CE neighbour equivalents.
   - It is a moderate change: 8.5 % of stable events and 1.6 % of unstable events flip, almost all near the CCT, and the median CCT shift is −26 ms.
   - It keeps the e12 borderline set and the EEAC feature's performance.
2. **Do not adopt F1.** Its size split and lower-end inertia values are my own choices, not cited values, and it shortens all CCTs by about 25 %.
3. **Fix the governor limits before generating v5 labels.** The clamp affects almost every random event and causes pre-fault drift of several degrees. It flips about 4 % of the borderline labels tested, which is comparable to the F1b effect near the CCT.
4. **After adopting:**
   - regenerate the labels, CCT brackets and EEAC features from the same machine data;
   - re-measure the e12 lift on v5 rather than carrying it over.
   Machine parameters do not explain the 220 kV miss excess; the e12 physics feature, which addresses it, does not depend on them.
