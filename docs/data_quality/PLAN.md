# Data, model and map: issue list and fix plan (2026-11-08)

Rule: every modification fixes a named issue (ID below) with evidence; nothing else changes.

## Safety rules
1. One change = one issue ID, with a published source (+ OSM/map check for any status change).
2. Automatic check after every change: 18/18 cases converge in both solvers. Islands, machines left
   disconnected, overloaded lines, lowest voltage, losses and unit tests must not get worse. Any
   line whose loading changes by more than 10 % is listed with a reason.
3. Before/after recorded per change: power-flow difference, map difference, events on affected lines.
4. Versioned models with sha256 + change log; every change revertible on its own.
5. Physical status beats paperwork only with evidence; JAO vs OSM conflicts are resolved first
   (OSM status may be stale).
6. Map-only changes never alter the power-flow model.

## Model and data issues
| ID | Issue | Source |
|---|---|---|
| A1 | Herbertingen–Hoheneck–Metzingen junction broken (single 121.7 km line, three dead-end stubs, misspelled extra node, X 44–61× published) | DQ04, DQ05-01 |
| A2 | Erzhausen junction split across 380/220 kV + dead-end 220 kV leg | DQ05-02/03 |
| A3 | Kühmoos/Laufenburg junctions lead nowhere (shared made-up point) | DQ05-04/05 |
| A4 | Remptendorf junction (osm-4706) unlocated, 6 circuits at estimated 13.35 km | DQ05-19 |
| A5 | Merzen dead-end stub (FREREN N parameters unpublished) | DQ01 |
| B1 | Parchim–Güstrow 328 out of service in v5: confirmed (JAO r9 removal; 50Hertz dismantles the 220 kV line before building 380 kV; plan approval Dec 2023, construction from late 2024) | 50Hertz, GvW, JAO |
| B2 | 321 on the same line: set out of service (same evidence) | DQ04-042 |
| B3 | Grafenrheinfeld 411/416 "temporary" junction in JAO | DQ04 |
| C1 | Bürs lines 421/422/405A/406A: both operators' reports added together (double count); lengths 0 km | DQ04 |
| C2 | Estimated lengths shorter than the straight-line distance: Walzwerk, Schwelgern | DQ05-20/21 |
| C3 | Gießen/Nord–Aßlar 4: published 1.13 km vs 12.5 km between ends | DQ05-26 |
| C4 | Odd reactance per km: Y-St. Peter 233/234, Lehrte–Wahle 1 | e9 audit |
| C5 | 61 lines added from OSM with estimated values (keep, mark as estimates) | DQ04 |
| D1 | Two copies of the line values (the CIM table the power flow reads vs the stored table) disagree on 100 rows | DQ04 |
| D2 | 11 doubled rows labelled "published"; 29 stored rows out of sync | DQ04 |
| D3 | Line ratings: `imax_a` differs between the two tables on 75 rows; the power flow reads the store's `rating_basis`/`dlr_*` (not synced in step 2) | steps_0_2_3 |
| D4 | Station checks: 59 nodes in two stations, 12 in none, 32 voltage levels without transformer, 65 dead ends (14 errors), 9 possible misspellings: review list | steps_0_2_3 |
| C6 | Remptendorf circuit count: OSM shows four 220 kV circuits west (383/384 direct to Hohenwarte II), model two; Stahlwerk leg one circuit in OSM, two in model | step 4 |
| I1 | Code bug: `qlim` (pf_connected/run.py) and `_border_weights` (src/tranpy/dispatch.py) count out-of-service lines at foreign nodes | step 4 |
| I2 | Default 700 MVA two-unit infeed points attached to invented nodes (found at Herbertingen and Kriftel): check the whole model | step 4 |
| I3 | Code bug: `connect()` in src/tranpy/grid_connections.py (~l. 891) sorts infeed sites with an unstable `sort_values("kv_hv")`; the greedy `spill_over_cap` then depends on row order, so removing or reordering one row moves up to 245 MW of load at 105 lumped buses. Affects the gate deltas of A1, 06#2 Kriftel, 06#4 Perleberg: re-gate after the fix | step 4 (I2) |
| I4 | Co-located infeed sites (Großgartach / Großgartach 2 / C2, Eichstetten / C1, Herbertingen / Herbertingen Amprion): capped load flips between twins when any site changes (up to 321 MW), blocking the I2 removals. Check whether the twins are duplicates (data) before any code tie-break | step 4 (I2) |
| D5 | Transformer ratings: 99 of 145 transformers have no rating. No effect on power flow, dynamics or TSA labels (impedances in ohms are converted on a fixed 100 MVA base, pandapower_model.py:134-157), but transformer loading is meaningless and not in the gate. To do after the current data issues: add JAO/TSO ratings (estimates by size class marked as such), add transformer loading to gate.sh, and check the 18 cases for transformer overloads (operating-point realism) | owner note 2026-11-09, deferred |
| E1 | Base overloads: Y-Niederlangen 103–129 % (8/18 cases), Großkrotzenburg–Trennfeld 217 at 151 % (high_solar) | e9 audit |
| E2 | Four Amprion 220 kV buses below 0.95 pu (high_load_t20) | e9 audit |
| F1 | Generic generator dynamics (D = 0, x′d = 0.3, H 3/4.5 s) for all 128 detailed generator models | e9 audit |
| G1 | Label edge cases: 57 labelled unstable without crossing 180°, 15 cross and recover (≤ 25 misses) | e9 audit |
| H1 | Paper says "9 hours of 2021"; the hours are Nov 2025–Oct 2026 | v5_hours |
| H2 | No station structure (station → voltage level → busbar, transformers, junctions) | owner |

## Map issues
| ID | Issue |
|---|---|
| M1 | All circuits meet at one point inside stations (Landesbergen, Vieselbach, Mutterstadt); no station boxes or voltage levels |
| M2 | Station positions to fix: Würgassen, Lüningkamp, GKM, Ried, Lippborg (DQ05-06..11; GKM and Lippborg need confirmation) |
| M3 | 7 stations point to the wrong OSM substation (05-12..18); 4 drawn lines stop 1.7–7 km short (05-22..25) |
| M4 | 15 circuits still drawn straight; 1,190 tower-line mismatch flags |

## Order
| Step | Work | Changes the power flow? | Status |
|---|---|---|---|
| 0 | Automatic check (rule 2) + change log | no | done: gate.sh/gate.py + CHANGELOG.md (steps_0_2_3.md) |
| 1 | B1: recheck 328 | — | done: change confirmed |
| 2 | D1/D2: one copy of the line values (power flow identical, checked by hash) | no | done: model_v5_1.pkl, gate PASS, hash identical |
| 3 | H2: station table + consistency checks (detection only) | no | done: stations/ + checks.csv, A1–A4 detected |
| 4 | A1–A4 topology fixes, one at a time | local | freeze candidate model_v5_5q.pkl (3393038e; CHANGELOG 5–41, not frozen; no model change in the last round). Code reference: I1, I3, I5, I6 (nearest-site ties; BLUDNZ toggle gone) and the per-TSO border split (ENTSO-E A11, data/entsoe_tso_border_flows_v5.csv; gate runs r<sha8>). Hradec–Etzenricht 441 kept (ENTSO-E A78). Open items in ~/data/grid/v5/KNOWN_LIMITS.md |
| 5 | B2, C1–C4 parameters/status | small | waits for owner approval |
| 6 | E1/E2: recheck after step 4 | depends | — |
| 7 | F1: generator-parameter test on a sample of events (label flips) | labels only | waits for owner |
| 8 | M1–M4: station boxes, lines ending at the box edge, pypowsybl station diagrams, position fixes | no | M1 done (interim OSM footprints; switch to station table) |
| 9 | Freeze v5, then the CCT sweep with speed-ups | — | — |

Owner decisions 2026-11-08: gate exceptions accepted for A1 Herbertingen, A4 Remptendorf, D4 Perleberg (voltage/loss shifts caused by the correct fix). Hüffenhardt OSM-only island line: remove line and both nodes.
