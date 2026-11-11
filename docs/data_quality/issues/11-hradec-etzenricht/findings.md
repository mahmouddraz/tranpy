# 11 Hradec–Etzenricht 441 (10T-CZ-DE-00001W): status Nov 2025–Oct 2026 and the removal gate

Analysis only. No model, map or PLAN.md changes were made.

## Summary

* **441 is in operation.** ENTSO-E Transparency (A78) has active, non-cancelled *planned* outages of
  `L-380-ETZ-HRA-441` on 2025-11-06, 2026-04-14 and 2026-06-01..06. Nobody schedules a day of
  maintenance on a decommissioned line. There is no decommissioning entry (no long or open-ended
  unavailability) after 2025-09-15. The JAO r9 TenneT change log ("Out-of-service" 2024-08-30,
  "Decommissioned" 2025-09-01) contradicts this. The CEPS Tielines row in the same file keeps 441
  without a note, which fits the ENTSO-E data.
* **The gate result is not extreme per case.** The +436 MW loss figure is the **sum over 18 runs**
  (9 cases x tan 0.1/0.2). Per run the increase is +2 to +40 MW, with one exception: high_wind at
  tan 0.2 rises by 122 MW, and that run is also the only place where the lowest voltage falls from
  0.918 to 0.896.
* **Why the results move:** the CZ exchange goes into two fixed-P boundary injections
  (`hradec vychod|380` and `prestice|380`). The two are split by summed line capacity, and the
  model has no Czech grid between them. With 441 in, the Hradec node is the only path from
  Röhrsdorf (50Hertz) to Etzenricht (TenneT). This path stands in for the real loop through the
  Czech grid. Removing 441 cuts that path, and it also takes away about 434 Mvar of the Hradec
  reactive limit and the voltage support near Etzenricht.
* **Recommendation: keep 441 in service.** Record the JAO change-log entry as a known source
  conflict that ENTSO-E overrides. Improving the CZ split is a separate, optional task
  (see below). It is not a reason to remove 441.

## 1 ENTSO-E A78 (unavailability of transmission infrastructure)

Query: A78, in_Domain = CZ (10YCZ-CEPS-----N), out_Domain = DE TenneT (10YDE-EON------1) and
DE 50Hertz (10YDE-VE-------2). Monthly windows from 2024-06 to 2026-10, paged with offset;
4602 documents in total. The latest revision of each document was kept. Status A09 = cancelled.
The raw XML stays in the session scratchpad and is not committed (ENTSO-E data is research-only).

### 441 `L-380-ETZ-HRA-441` (TenneT/ČEPS)
| period (UTC) | type | reason | note |
|---|---|---|---|
| 2024-06-17 .. 2024-08-22/23 | A53 planned | B19 / A95 "Other reason" | long planned outage. Its end falls just before TenneT's change-log "out of service 2024-08-30" |
| 2025-01-01 06:47–11:35 | A54 forced | B18 | short trip |
| 2025-06-02 .. 2025-09-15 | A53, with A54 from 06-02 to 06-04 | A95 "Curtailment due to emergency situation" (ČEPS, 500 = ČEPS series) / B19 | about 3.5 months unavailable or limited. Overlaps TenneT's "decommissioned 2025-09-01" |
| **2025-11-06 06:00–16:00** | A53 planned | B19 maintenance | **active, not cancelled** |
| **2026-04-14 06:00–13:00** | A53 planned | B19 | active |
| **2026-06-01 .. 2026-06-06** | A53 planned | B19 | active |

No 441 entry exists from 2026-06-06 to 2026-10-31.

### 442 `L-380-ETZ-PRE-442`
Planned (B19) outages only: 2024-09 (several days), 2025-03-19..28, 2025-11-05, 2025-11-20
("Upgrading"), 2026-04-28..30, 2026-07-01..15. The line is in operation.

### 445/446 `Hradec-Röhrsdorf 1/2` (50Hertz/ČEPS)
Planned (B19) outages of a few days each: 2024-11-12..14, 2025-11-10, 2026-02-09..13,
2026-04-13..16, 2026-05-11..15, 2026-07-10, 2026-08-06..08 and 2026-08-13..15. Forced trips (A54):
2026-06-09 (both circuits, about 2 h) and 2026-07-14 (both circuits, about 3.5 h, then limited until
07-20). The DE-side PSTs `Röhrsdorf Tr. 441/442` have several day-long planned outages. Both lines
are in operation.

## 2 Physical flows (A11) at the v5 hours, MW, positive = DE→CZ

| case | hour (UTC) | TenneT→CZ | 50Hertz→CZ | total | model CZ net (Hradec+Prestice) |
|---|---|---|---|---|---|
| high_wind | 2026-10-23 18 | 183 | 1275 | 1458 | 1458 |
| high_solar | 2026-08-13 11 | 83 | 363 | 447 | 447 |
| low_load | 2026-08-30 01 | 533 | 1435 | 1968 | 1968 |
| high_load | 2026-01-29 10 | −259 | 1326 | 1067 | 1067 |
| median_load | 2026-04-30 16 | 802 | 1293 | 2096 | 2096 |
| winter_peak | 2026-01-22 07 | 324 | 1704 | 2028 | 2028 |
| summer_peak | 2026-06-29 08 | −325 | −602 | −927 | −927 |
| very_light_night | 2026-08-23 01 | −659 | 1248 | 590 | 590 |
| baltic_import_low_load | 2025-11-03 01 | −208 | 1173 | 965 | 965 |

* The model total matches A11 exactly. All totals fit easily within three or four 380 kV
  circuits (445+446 ≈ 3300 MVA, 441 ≈ 1316 MVA, 442 ≈ 1490 MVA).
* The split does not match. On the real border, 50Hertz→CZ carries most of the flow. The TenneT
  border often flows **CZ→DE** while 50Hertz exports (high_load, very_light_night and
  baltic_import): this is the known loop flow from north-east Germany through CZ into Bavaria.
  The model cannot show it, because both CZ nodes get the same sign. Hradec takes 7040/9305 = 76 %
  (441+445+446 against 442).
* The ERÚ final report on the CZ outage of 4 July 2025 gives the real-time loading of V445 and V446
  as 59 % and 53 %. After PST T454 failed, ČEPS controls V445 with PSTs T451/T452 at Hradec and
  50Hertz controls V446 with PST T442 at Röhrsdorf. T453 is held in reserve. The T454 internal
  failure happened on 2024-11-02, and its replacement is expected around 2030. This matches JAO
  "PST Hradec 4 removed 2025-03". Source:
  [ERÚ, závěrečná zpráva výpadek elektřiny, 23.06.2026](https://eru.gov.cz/sites/default/files/obsah/prilohy/zaverecna-zprava-vypadek-elektrinyfinal23062026.pdf).
  The report does not mention V441 or any decommissioning of it.

## 3 The model

* **Split** (`src/tranpy/dispatch.py::_border_weights`, lines 349–364): each border's exchange is
  spread over that country's boundary nodes in proportion to the rated MVA
  (rated_current × kV) of the in-operation lines ending there. The split is by capacity. It does
  not use impedance or PST control, and it does not use the per-TSO A11 values.
* **Boundary nodes:** each one is a PV generator with fixed P (pypowsybl `min_p`/`max_p` = target
  ±0.001; pandapower gen), set to V = 1.02 pu, with Q limit = 0.33 × MVA of its in-operation lines
  (`reactive_power.py::boundary_q_limits`, `scripts/v5_pf_connected.py::qlim`). Removing 441 cuts
  the Hradec limit from about 0.33 × 4630 to about 0.33 × 3315 MVA, a loss of about 434 Mvar.
* **CZ grid:** none. `hradec vychod|380` and `prestice|380` are separate boundary nodes with no
  equivalent branch between them.
* **PSTs:** the Hradec PSTs (T451–T454) are not modelled. On the DE side, `PST Röhrsdorf 441/442`
  exist as transformers flagged `phase_shifter`, but their angles are not used. The grid_case
  docstring says "phase shifter angles … are left out", so they act as plain series impedances.
* **Effect of removing 441, model_v5_5b vs 5a** (`gates/gate_f99a7cda_vs_d6fe836c.md`, results
  CSVs in `cache/work/pf_v5/p{d6fe836c,f99a7cda}`): Röhrsdorf–Hradec 445/446 loading falls by
  0.16–0.22, Etzenricht–Mechlenreuth rises by 0.18–0.25, and Eula–Röhrsdorf and
  Plattling–Schwandorf change too. Losses rise by 2–40 MW per run (122 MW in high_wind tan 0.2).
  The lowest voltage changes only in high_wind tan 0.2 (0.918 → 0.896).
  How to read this: with 441 in, part of the 50Hertz→Bavaria transit runs
  Röhrsdorf→Hradec→Etzenricht, which matches the real loop through CZ. Without 441 that transit
  goes back onto the German 380 kV corridor (Remptendorf/Redwitz–Mechlenreuth–Etzenricht), and
  Etzenricht loses a nearby voltage-controlled source. In the real grid, the loop would still run
  through Hradec–(CZ grid)–Prestice–442. The model cannot show this because it has no CZ grid.
  (Flow directions on single lines were not extracted; the reading above is inferred from the
  loading changes and the fixed-P injections.)

## 4 Web check

* ERÚ report above: covers V445/V446 and the PSTs in detail, and says nothing about a V441
  decommissioning.
* No ČEPS, TenneT or NEP page was found that reports decommissioning, reconstruction or
  conversion of Hradec–Etzenricht. Searches covered "Hradec Etzenricht V441 ČEPS TenneT
  rekonstrukce" and "TenneT Etzenricht Hradec Rückbau Stilllegung". The hits were about other
  projects:
  [Ostbayernring Rückbau, TenneT 2022](https://netztransparenz.tennet.eu/fileadmin/user_upload/Company/News/German/Haffke/2022/20220804_A030_PM_PFB_Abschnitt_A_final.pdf)
  (the existing Redwitz–Schwandorf line, not the CZ tie), and
  [EC project Přeštice–Vítkov](https://ec.europa.eu/regional_policy/cs/projects/Czechia/new-overhead-power-line-under-construction-between-prestice-and-vitkov-western-czechia)
  (crossing adjustments of the Hradec–Etzenricht line). A conference paper lists V445/446
  reconstruction for 2027–2029.
* [Wikipedia GKK Etzenricht](https://en.wikipedia.org/wiki/GKK_Etzenricht): the 441 line was
  completed in 1992 and is limited to 1316 MW by substation equipment (secondary source).

## 5 Recommendation: keep 441 (in_operation), and note the source conflict as a known limit

1. **Keep it in service.** ENTSO-E is the operational record. It shows planned maintenance on 441
   after the JAO "decommissioned" date, through 2026-06, with no permanent unavailability. The
   CEPS JAO row also keeps the line. This agrees with the memory rules "OSM construction tags
   stale – verify with JAO/TSO" and "no overwrite without gain". The TenneT change-log entry
   probably refers to the long outages of 2024 and summer 2025, not a real decommissioning.
2. **KNOWN_LIMITS:** replace "owner decision against JAO r8" in the `10-OOS held` row with
   "ENTSO-E A78 shows 441 in operation (planned outages 2025-11-06, 2026-04-14, 2026-06-01..06)".
   This is for the data agent to apply; this analysis made no edit.
3. **Optional improvement, not required for 441.** The CZ split is the real weakness.
   (a) Split CZ per DE control area using A11 TenneT↔CZ and 50Hertz↔CZ instead of capacity
   weights. This would give Prestice and the 441 end a TenneT share that can carry the opposite
   sign of the 50Hertz share, as in the real loop. (b) Or add a CZ equivalent branch
   Hradec–Prestice (X from the CZ grid, e.g. via Výškov/Chrást), so the loop forms through the
   impedances. In both options, the Hradec PST (T451/T452 + Röhrsdorf T442) can stay at neutral
   taps: ERÚ reports tap 0 and −2 in normal operation. Expected effect: in the cases where
   TenneT↔CZ has the opposite sign (high_load, very_light_night, baltic_import, about −200 to
   −660 MW), Etzenricht–Mechlenreuth loading changes by roughly that amount divided by its rating.
   This is an estimate and has not been run.
