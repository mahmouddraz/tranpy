# Audit of pre-fault TSA misses (v4.1): data and modelling errors vs. missing information

Status: analysis only. Nothing was fixed or committed. `model_v4final.pkl` was read, not written.
Scripts, CSVs and logs are in `cairo-lab:~/data/exp/e9_miss_audit/` (`01_lines.py` … `07_dq04.py`, `*.csv`, `t13.log`, `t46.log`, `t_extra.log`). statsmodels 0.15 was installed into `e9_miss_audit/pylib` with `--target`, so the shared `.venv` was not changed.

Scope: v4.1, 11,159 events, 1,325 unstable, 18 cases, leave-one-hour-out OOF. LimiX-2 = `e5_tfm2/oof/line_pre__limix_2__hour`, LightGBM = `e1_boosting … lightgbm__none` (line_pre, hour).
A **miss** is an unstable event that the model predicts as stable. Rates are over unstable events only, unless the text says otherwise.
Statistics:
- logistic regression with cluster-robust SEs by `fault_line`, adjusted for the 220 kV indicator, the clearing band (<0.3 s merged; 0.3–0.4, 0.4–0.5, ≥0.5 s) and the hour;
- line-level permutation of each flag within voltage level (1,000 permutations), using a CMH-type stratified miss difference over kV × band × hour;
- line-cluster bootstrap stratified by voltage, seed 20261104.

Baseline numbers (LimiX-2): 220 kV miss rate 35.9 % (135/376), 380 kV 21.5 % (204/949). Adjusted OR(220) = **2.24** (p < 0.001). LightGBM: OR 2.03. Bootstrap raw gap: +14.0 pp, 95 % CI [7.8, 20.6].

---

## 1. Data-quality flags

Each of the 1,015 fault lines was mapped to its CIM `ac_line_segment` row. The rule is: in-operation rows without Großgartach–Obrigheim osm-4782. The ANDES bus pairs match the CCT files for 996/996 lines that appear there.
The flags were tagged from:
- DQ01 `occurrences.fault_lines`, including the dead-end sub-class;
- DQ03 non-display classes, via `model_eics`;
- DQ04 by EIC, per group A–F;
- DQ05 nodes, both `topology/model` and any;
- `geom_issues.csv` lines and sites;
- `tower_circuit_mismatch` `model_fewer`;
- the model tables `grid_de_plausibility`, `value_overrides`, `library_changes` and `length_ledger` (flagged);
- data_quality ≠ published; route_quality station/estimated; zero length; tee (`y-` node);
- bridge and radial lines.

`MODELSUSPECT` = DQ01 dead-end ∪ DQ05 topology/model ∪ zero length ∪ length-ledger flagged ∪ geom line issue. `near*_model` means 1–2 hops from a MODELSUSPECT element.

| flag | lines 220/380 | unstable ev. 220 (flag) | LX miss 220 flag / not | LX miss 380 flag / not | adj. OR LX miss (p) | adj. OR GB miss (p) | perm p |
|---|---|---|---|---|---|---|---|
| DQ01 future/dead-end | 42/100 | 19 | 0.47 / 0.35 | 0.24 / 0.21 | 1.32 (0.38) | 1.58 (0.10) | 0.35 |
| DQ01 dead-end only | 28/36 | 15 | 0.47 / 0.36 | 0.04 / 0.22 | 0.55 (0.36) | 0.60 (0.18) | 0.30 |
| DQ03 circuit count | 54/60 | 56 | 0.34 / 0.36 | 0.29 / 0.21 | 1.25 (0.42) | 1.88 (0.02) | 0.43 |
| DQ04 any (143 rows) | 70/71 | 95 | 0.31 / 0.38 | 0.15 / 0.23 | 0.65 (0.06) | – | – |
| DQ04-D (OSM median estimates) | 39/20 | 70 | 0.24 / 0.39 | 0.08 / 0.23 | 0.38 (0.001) | – | – |
| DQ04-F (store/CIM out of sync) | 7/22 | 5 | 0.60 / 0.36 | 0.55 / 0.21 | 4.15 (0.05) | – | – |
| DQ05 topology/model | 17/7 | 49 | 0.14 / 0.39 | 0.25 / 0.22 | **0.19 (<0.001)** | 0.22 (<0.001) | 0.01 |
| geom line issue | 20/10 | 22 | 0.27 / 0.36 | 0.50 / 0.21 | 0.71 (0.58) | 1.16 (0.79) | 0.62 |
| geom site issue | 105/106 | 102 | 0.44 / 0.33 | 0.25 / 0.21 | 1.38 (0.15) | 2.07 (0.001) | 0.11 |
| tower model_fewer | 58/65 | 53 | 0.45 / 0.34 | 0.27 / 0.21 | 1.55 (0.09) | 1.83 (0.03) | 0.14 |
| plausibility table | 118/75 | 95 | 0.43 / 0.34 | 0.17 / 0.22 | 1.07 (0.77) | 0.98 (0.93) | 0.59 |
| not published (library/sibling) | 60/53 | 94 | 0.28 / 0.39 | 0.11 / 0.23 | 0.48 (0.002) | 0.37 (<0.001) | 0.04 |
| length-ledger flagged | 79/40 | 76 | 0.30 / 0.37 | 0.23 / 0.21 | 0.86 (0.62) | 1.43 (0.21) | 0.85 |
| tee leg | 117/133 | 132 | 0.30 / 0.39 | 0.22 / 0.21 | 0.83 (0.38) | 1.20 (0.36) | 0.19 |
| MODELSUSPECT | 125/81 | 138 | 0.27 / 0.41 | 0.16 / 0.22 | 0.49 (0.006) | 0.76 (0.22) | 0.02 |
| any flag | 281/324 | 279 | 0.34 / 0.41 | 0.20 / 0.23 | 0.81 (0.24) | 0.89 (0.49) | 0.33 |
| ≤1 hop from MODELSUSPECT | 124/179 | 123 | 0.47 / 0.30 | 0.23 / 0.21 | 1.48 (0.03) | 1.21 (0.27) | 0.03 |
| ≤2 hops from MODELSUSPECT | 189/383 | 185 | 0.43 / 0.29 | 0.23 / 0.20 | **1.59 (0.008)** | 1.35 (0.07) | 0.014 |

(Full table: `t1_flags.csv`, `t1_dq04.csv`.) For the overall error (all events, adjusted also for the label), MODELSUSPECT has OR 0.50 and DQ05-model OR 0.35, so flagged lines have *fewer* errors. DQ03 (OR 1.73, p = 0.006) and tower model_fewer (1.67, p = 0.008) are the only ones with a modestly higher error.

**Does any flag explain the 220 kV gap?**
- Adding 12 flags at once moves the adjusted OR(220) from 2.24 to 2.89 for LimiX and from 2.03 to 2.28 for LightGBM. The gap gets larger, not smaller.
- On lines with no flag at all, 220 kV misses 41.2 % (n = 97) against 23.2 % at 380 kV (n = 508).
- After excluding MODELSUSPECT lines and their 2-hop neighbourhood: 34.0 % (n = 53) vs 20.5 % (n = 298).
- The gap is the same on clean lines.
- The 49 LimiX misses on suspect lines are 14 % of the 339 misses (`suspect_lines_misses.csv`).

**Verdict: no link for the data flags themselves.** Most flags go the other way: flagged lines are short stubs and tees, whose events are easy. The **≤2-hop neighbourhood effect** (OR 1.59, permutation p = 0.014) is a **weak link**. It is regional: Amprion 220 kV is 45 % vs 32 %, TenneT 41 % vs 31 %. It probably reflects the Ruhr/Lower-Saxony 220 kV meshes, where the flagged stubs are, rather than a causal error. DQ04-F has OR 4.15, but it is based on 16 unstable events. The F rows differ only in the store, and the PF uses the CIM, so no physical mechanism exists; this is treated as noise.

## 2. Parameter plausibility per line

Reference values (typical overhead lines, Oeding/Oswald, *Elektrische Kraftwerke und Netze*):
- 380 kV 4-bundle: X′ ≈ 0.26 Ω/km, R′ ≈ 0.03 Ω/km, C′ ≈ 13.7 nF/km (B′ ≈ 4.3 µS/km);
- 220 kV 2-bundle: X′ ≈ 0.30, R′ ≈ 0.06, C′ ≈ 11.5 nF/km (B′ ≈ 3.6 µS/km);
- cross-check: pandapower std type 490-AL1/64-ST1A, X′ 0.253 / 0.285 Ω/km.

A line is an outlier when it is longer than 1 km and X′ is outside ×0.67–1.5, or B′ is outside ×0.5–2, or R′ is outside ×0.3–3.

| | 220 kV (377 lines) | 380 kV (638 lines) |
|---|---|---|
| X′ median [p5, p95] Ω/km | 0.292 [0.215, 0.399] | 0.253 [0.237, 0.296] |
| B′ median [p5, p95] µS/km | 3.69 [2.76, 5.80] | 4.37 [3.46, 5.06] |
| parameter outliers / <2 km / >150 km / 0 km | 34 / 45 / 4 / 4 | 31 / 44 / 2 / 5 |
| bridges / radial | 41 / 35 | 41 / 39 |

Medians are on target. The 220 kV spread is wider, as expected from mixed conductor types. Outlier lines carry 25 LimiX misses in total. Outliers have adjusted OR 0.86 (p = 0.64); short lines 0.77 (0.41); bridges 0.59 (0.37); radial lines 0.55 (0.36). Outliers worth a check are in `t2_param_outliers.csv`:
- Y-St. Peter 233/234 (X′ 0.032 Ω/km over 75 km, published): the X probably belongs to a short section, not to the route length;
- Giessen/N–Asslar 4 (X′ 3.06 Ω/km over 1.13 km): matches DQ05-26;
- Siems–Lübeck blau (B′ 73 µS/km), Mehrum–Mehrum/N (B′ ×39): probably cable or short-section B;
- Lehrte–Wahle 1 (X′ 0.12): 2 misses out of 2 unstable events;
- Friedrichshain–Marzahn and Reuter–Charlottenburg (B′ ×10–13): Berlin 380 kV cable, plausible.

**Verdict: no link.** A few parameter values are implausible, but they do not explain the misses.

## 3. Topology and electrical position

The Thevenin impedance at both ends was computed for every case from the ANDES case of that case: Y-bus of lines and transformers, with machines as x′d on their own base and no loads (`03_pf_thev.py`, `pf_bus.parquet`). S_sc = 1/|Z_th| at the weaker end. The other variables come from the line_pre features (distance to the nearest machine in hops and reactance, kinetic energy within 2 hops) and from the CIM graph (degree, bridge, radial).

| median (unstable events) | 220 caught | 220 missed | 380 caught | 380 missed |
|---|---|---|---|---|
| n | 241 | 135 | 745 | 204 |
| S_sc weaker end (GVA) | 6.2 | 6.3 | 18.5 | 17.8 |
| machine x-distance (pu) | 0.053 | 0.036 | 0.022 | 0.025 |
| kinetic energy ≤2 hops (MWs) | 356 | 173 | 1,884 | 814 |
| min end degree | 2 | 3 | 4 | 4 |
| bridge share | 0.04 | 0.06 | 0.03 | 0.005 |
| length (km) | 10.4 | 19.1 | 24.6 | 31.9 |

220 kV is physically a different regime: 3× weaker short-circuit power (6 vs 18 GVA, in every hour, `t6_hour_pf.csv`) and 5–10× less kinetic energy nearby. Within a voltage level, however, missed and caught events look alike. The exceptions are less nearby inertia and a longer line for the misses.

Adjusting for S_sc, machine distance and hops, nearby kinetic energy, degree, bridge, radial, length and loading does **not** reduce the gap: OR(220) goes from 2.24 to 4.15 (p < 0.001), with bootstrap median 4.2 and 95 % CI [1.9, 9.9]. Adding the flags gives 5.24. LightGBM goes from 2.03 to 4.47.

**Verdict: no modelling error found. The misses are in a physically harder, weaker regime, and the line-level electrical position does not explain the gap.** The variable that does explain much of it is which machine separates (test 4).

## 4. Label and simulation integrity of the missed events

The traces cover all 1,325 unstable events (compact store; the 'c:firot' channel is the angle against the COI). Limitation: the compact store keeps 0–1 s, the fault-on period up to clearing, and then **only the last sample at 6.6 s**. The time of separation after clearing can therefore not be resolved. It is either "at clearing" or "between clearing and 6.6 s".

| (share of unstable) | 220 caught | 220 missed | 380 caught | 380 missed |
|---|---|---|---|---|
| separation after clearing (not yet at clearing) | 0.29 | **0.64** | 0.41 | **0.69** |
| solver stopped early (out-of-step latched) | 0.11 | 0.09 | 0.12 | 0.08 |
| last sample < 180° (latched transient exceedance) | 0.04 | 0.07 | 0.02 | 0.07 |
| label = 1 but `unstable_coi_180` = 0 | 0.03 | 0.06 | 0.05 | 0.04 |
| a single machine separates | 0.85 | 0.76 | 0.65 | 0.67 |
| separating machine < 100 MVA | 0.53 | 0.55 | 0.22 | **0.41** |
| separating machine median MVA | 89 | 74 | 265 | 111 |
| separating machine at 220 kV EHV bus | 1.00* | 1.00* | 0.05 | 0.15 |
| CE/foreign equivalent (GENCLS) separates | 0 | 0 | 0 | 0 |
| CCT bracketed / failed | 0.68/0.32 | 0.77/0.23 | 0.84/0.16 | 0.82/0.18 |
| median clearing − CCT (bracketed) | +0.16 s | **+0.054 s** | +0.14 s | **+0.047 s** |
| within 50 ms above CCT | 0.13 | 0.45 | 0.13 | 0.52 |

\* 95 % of the 220 kV unstable events separate a machine behind a 220 kV node.

- **Barely unstable.** Missed events lie close to the CCT. A model of the miss on bracketed events, adjusted for hour, gives OR 0.25 per +100 ms margin (p < 0.001). In the "within 50 ms" set the miss rate is 57 % (220 kV 69 %) against 19 %. The CCT bracket comes from the base operating point and fault, while the events vary load, operating point and fault impedance. Some events are therefore unstable below the bracket (13–18 % of the missed bracketed events, but 2 % of the caught ones, have a negative margin). These labels are genuine but borderline.
- **Label-definition artefacts.** 57 events have label 1 but `unstable_coi_180` = 0. Most of them (75 %) are early solver stops with a latched out-of-step flag. 46 events have a last sample below 180°; 15 of those ran to 6.6 s, which means a machine crossed 180° from the COI and came back (pairwise TSI > 0); 10 of them were missed. The combined artefact set (early stop, last sample below 180°, flag mismatch) explains at most about 25 misses. "Late separation" is not an artefact; it is the storage limit described above. Without these categories the gap remains: 19.0 % vs 10.5 %.
- **No equivalent machines separate.** No CE equivalent or GENCLS unit ever separates.
- **Small generic units.** Every GENROU has identical generic dynamics: x′d = 0.3, H = 3 or 4.5 s, **D = 0**, T′d0 = 6 s, whatever the unit size. Adding log(MVA of the separating unit) to the miss model lowers OR(220) from 2.24 to **1.47 (p = 0.06)**; unit MVA has OR 0.51 per e-fold (p < 0.001). Most of the 220 kV gap is therefore "a small 220-kV-connected unit loses step". Pre-fault line features cannot see this well, and the generic, undamped parameters of small units make it likelier.
- **Top separating EHV buses with misses** (`t4_sep_ehv_bus.csv`):
  - osm/35833442|220 (DQ05-21 "length too short"): 44 unstable, 16 missed;
  - krun|220 (52 MVA): 36 / 15;
  - fwh-s|220 (62 MVA): 15 / 12;
  - stade/west|380 (85 MVA): 16 / 12;
  - maade|220: 35 / 11;
  - osm/97857512|220 (DQ05-20 "length too short"): 32 / 10;
  - marke|380 (52 MVA): 10 / 10;
  - erzhausen|220 (DQ05-02/03 broken tee): 16 / 8.

  A unit whose EHV bus is a suspect node has no higher miss rate (OR 1.04, p = 0.85).
- **Per-case initialisation is clean.** All 18 cases pass the flat run: drift 0°, max dω ≤ 2·10⁻⁹, PF dV ≤ 5·10⁻⁵, 0 init lines out of tolerance (`t4_case_init.csv`). high_wind t10/t20 lose 174 and 246 of 600 sampled scenarios as infeasible pre-fault points (V < 0.9 or no PF), against 0–12 in the other cases. This is a selection effect on that hour, not a label error.

**Verdict: weak link / borderline labels, not a numerical artefact.** The misses are mostly genuinely unstable events just above the CCT, in which a small machine separates after clearing. The modelling issue that matters is the generic, undamped machine model for small units: it makes single-unit pole slip the main cause of 220 kV instability. This is a model-fidelity question, not a data error.

## 5. Feature integrity

| check | result |
|---|---|
| `fault_line` in the features = OOF `fault_line` = `line|fault_line_cat` | 11,159 / 11,159 |
| `line|vn_kv` = model kV of the mapped CIM circuit = ANDES Vn | 100 % |
| `line|x` = ANDES Line x (per case; up to parallel-circuit merging) | 100 % (median ratio 1.000) |
| CIM ↔ ANDES ↔ CCT bus pairs | 996 / 996 lines present in the CCT files |
| NaN / inf rate, all 25 `line|` features, 220 and 380 kV | 0 / 0 |
| features constant within a case | only `vn_kv` and `bus_vn_kv` (by construction) |

At 220 kV, `kinetic_mws_2hop` and `gen_mw_2hop` are 0 for the median event, as they are at 380 kV. Those features are sparse, but they are not broken.

**Verdict: no link.** The features of 220 kV lines are not degraded, and fault lines map to the correct branches.

## 6. Per-hour anomalies (high_load, 220 kV, 58 % misses)

| hour | unstable 220/380 | LX miss 220 | LX miss 380 |
|---|---|---|---|
| high_load | 72/151 | **0.583** | 0.238 |
| low_load | 24/81 | 0.500 | 0.123 |
| baltic_import_low_load | 29/76 | 0.414 | 0.079 |
| winter_peak | 101/161 | 0.267 | 0.286 |
| very_light_night | 35/135 | 0.229 | 0.422 |

(Full: `t6_hour_miss.csv`, `t6_hour_pf.csv`.) Base PF of high_load:
- the 220 kV median loading (0.16) and p95 (0.54) match winter_peak and high_wind;
- high_load_t20 has 4 EHV buses below 0.95 pu (Bixterheide, Garenfeld, the Y-Garenfeld tee and osm/1463885594, all Amprion 220 kV, vmin 0.944);
- the S_sc of the weaker 220 kV end is normal (6.8 GVA).

Compared with the other hours, high_load 220 kV unstable events have:
- lower bus SC strength (520 vs 1,014);
- a smaller CCT margin (+0.084 vs +0.120 s);
- a larger pre-fault angle across the line.

Misses are concentrated on:
- TSOs: TransnetBW 8/9 and Amprion 11/14;
- lines: Laer–Pöppinghausen (4/4), Landesbergen–Hannover/W 3 (3/3), Wahle–Gleidingen 2 (3/5) and the Walzwerk stub osm/97857512 (2/2, DQ05-20).

Persistent PF overloads in the base cases, independent of the hour:
- Y-Niederlangen EOWS tee → Meppen / Dörpen-W (1.03–1.29 in 8 of 18 cases); the 0.33 km leg 721 is DQ01 + tee, rating from a sibling;
- Großkrotzenburg–Trennfeld 217 at 151 % in high_solar (rating 834 A);
- Lübeck–Hamburg/N 220 at 109 % in baltic_import.

**Verdict: weak link.** high_load is a stressed-voltage hour in the Amprion 220 kV area, and the hold-out of that hour puts it outside the training distribution: weaker buses and smaller margins, which is legitimate. No dispatch artefact was found in high_load itself. The overloads listed above are dispatch or rating issues of other hours.

---

## Summary: are the misses caused by modelling mistakes?

**Mostly no.**
- The known data-quality findings (DQ01, DQ03, DQ04, DQ05, geometry, towers, plausibility) do not concentrate the misses. Flagged lines are missed *less* often, and the 220 kV gap is unchanged on completely clean lines (41 % vs 23 %).
- Features and branch mapping are correct.
- The simulations initialise cleanly.
- No equivalent machine and no numerical late-time artefact drives the misses.

The misses are genuine but borderline unstable events: median 50 ms above the CCT, with separation after clearing. In 220 kV events, a small unit behind a weak 220 kV node loses step. These are the hardest-to-see cases for pre-fault line features. This is *missing information* for the classifier, not a data mistake.

**One modelling issue is real and partly explains the 220 kV gap:**
- all 128 GENROU units carry identical generic dynamics, including D = 0, regardless of size;
- with the unit's MVA in the model, OR(220) falls from 2.24 to 1.47;
- the result is that instability at 220 kV is mostly single small-unit pole slip.

This is a model-fidelity choice that inflates borderline small-unit instability, not a data error. It should be settled before the CCT sweep, because the sweep will inherit it.

### Suspect elements and proposed checks
| element | evidence | proposed check |
|---|---|---|
| Generic GENROU parameters (D = 0, x′d 0.3, H 3/4.5 s for every unit) | 41 % of 380 kV misses and 55 % of 220 kV misses have a separating unit < 100 MVA; unit MVA explains most of the 220 gap | rerun a sample of missed events with size-dependent H/x′d (e.g. IEEE/Kundur typical by MVA) and D > 0; check the label flip rate |
| Label definition (latched \|δ−δ_COI\| > 180° for any machine, incl. units < 100 MVA) | 15 events cross 180° and recover by 6.6 s; 57 label/`unstable_coi_180` disagreements | report label sensitivity with "≥ x MVA separates" or pairwise > 360°; keep full traces (not compact) for the borderline set |
| osm/35833442\|220, osm/97857512\|220 (DQ05-20/21 "length too short") | 76 unstable events with these units separating, 26 missed | fix the lengths per DQ05; rerun those faults |
| erzhausen\|220 tee (DQ05-02/03 broken, dead-end stub) | 16 unstable / 8 missed with the Erzhausen unit separating | merge the tee (class 02); rerun |
| Y-Niederlangen EOWS tee (721/689/949) | PF loading 1.03–1.29 in 8/18 cases; 0.33 km leg, sibling rating | verify the rating and tee topology against JAO |
| Großkrotzenburg–Trennfeld 217 (834 A) | 151 % in high_solar | check the JAO rating |
| Y-St. Peter 233/234, Giessen/N–Asslar 4, Lehrte–Wahle 1 | X′ 0.03 / 3.06 / 0.12 Ω/km | check the JAO R/X vs the route length |

### What could not be tested
- Time of separation after clearing, and solver warnings per event: the compact traces keep only the last sample after clearing, and no per-event solver log is stored. The early stop was inferred from trace end < 6.5 s.
- Which machine separates *first* after clearing is approximated by the machine above 180° at the first stored sample. For events that separate after clearing, that is the 6.6 s sample.
- The CCT margin uses the base-case bracket, not the event's own operating point and fault impedance.
- DQ02 (tee issues) has no occurrence file. Tees were proxied by `y-` nodes and `grid_de_tee_positions` was not used separately.
- DQ04 was read as it existed at 19:08. It was still being written, and it was not edited.
- Thevenin impedances ignore loads and use x′d for all machines. The PF loading uses base PF flows (before the per-event operating-point scaling) against CIM `rated_current_a`.
