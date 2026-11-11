# 14 Remaining model-check halos on the staging map (8094)

Date: 2026-11-09. Model: `v5/frozen/model_v5_frozen.pkl` (sha 3393038e, **not changed**).
Map: staging 8094 (`staging_osmgraph/app`, data `web_frozen`). Backups:
`staging_osmgraph/app_before_halo` (its `data` link points to `web_frozen_before_halo`) and
`staging_osmgraph/web_frozen_before_halo`. Working files: `staging_osmgraph/halo/`.

Sources: JAO Core SGM r9 (2026-03, Lines/Tielines/Change Log) and r1 (`v5/jao`), OSM tags
(`osm/hv_lines.parquet`, Overpass caches of the network run), model circuit table, earlier
reviews 03, 08 and 09. No operator was asked. Published values come first. No model change.

## Result

| | halo pieces | findings (tower lines) | km |
|---|---:|---:|---:|
| before (web_frozen 2026-11-09 20:03) | 187 (97 fewer, 90 more) | 147 | 1397 |
| after the routing fix | 156 | 120 | 1188 |
| after the review rows (now on 8094) | **71** (50 fewer, 21 more) | **57** | 711 |

A finding is a group of touching halo pieces with the same check, voltage, OSM count and
model circuit set. `occurrences.csv` has one row per finding: `B###` are the 147 findings
before, `A###` the 120 after the routing fix (6 of them, 54 km, are new; they come from
circuits that are now dashed).

Classes of the 147 findings before (the class of the matching after-finding, or b when the
routing fix removed the halo):

| class | findings | km | status |
|---|---:|---:|---|
| (a) OSM artefact | 39 | 469 | all explained (review row, popup gives the reason) |
| (b) map routing artefact | 96 | 818 | 33 fixed by the routing change, 24 short station-approach pieces explained, 39 open |
| (c) model gap/error | 9 | 54 | proposals below, halo kept |
| (d) unclear | 3 | 56 | halo kept |

Classes of the 120 findings after the routing fix (these rows carry the evidence):

| class / subclass | findings | km |
|---|---:|---:|
| a jao_single_circuit (Wolmirstedt - Parchim 332) | 2 | 109 |
| a jao_removed (extra OSM circuit in r1 but not r9, or removed in the r9 Change Log) | 5 | 140 |
| a no_r9_counterpart (no r9/model circuit of the voltage runs along the line) | 12 | 58 |
| a osm_undercount (all circuits are r9, ratio 0.7-1.5, no parallel OSM way) | 20 | 162 |
| b approach (< 0.5 km at a station, issue-09 rule) | 24 | 8 |
| b missing_terminal (a circuit of the line has no terminal) | 18 | 309 |
| b no_path (a circuit of the line has no strict OSM path, drawn dashed) | 14 | 179 |
| b parallel_way (circuits stacked on one of two parallel ways) | 8 | 52 |
| b odd_ratio (BLUDNZ W schematic legs, Ludersheim - Raitersaich 237) | 3 | 9 |
| c model_only | 9 | 54 |
| d | 5 | 109 |

## Wolmirstedt - Parchim Süd 332 (owner's example): class (a), explained

OSM way 259468196 "Ltg 332 Wolmirstedt - Parchim Süd" (220 kV, 71.7 km, `cables=6` = two
systems) and the 37.6 km shared with 436 Perleberg (ways 1243590034…043, `380000;220000`).

- **JAO r9 has one 220 kV circuit:** `Wolmirstedt - Parchim/Süd 332-322`,
  EIC 11TD8L332-322--6, 140.9 km, Imax 900 A, R 6.95 / X 37.59 Ω / B 605.989 µS. X = 0.267 Ω/km
  and B = 4.30 µS/km are the values of a single circuit, the same as the other 50Hertz 220 kV
  rows (0.25–0.30 Ω/km). Two systems in parallel would give about half that X/km. r1 also has
  exactly one row (Parchim/Süd - Wolmirstedt 332-322, 900 A).
- **r9 Change Log:** r4 2023-09-20 "Removed: decommissioned", r5 2024-03-04 "New Topology",
  r7 2025-03-01 "Updated Parameters", r9 2026-03-01 "Updated Parameters/Naming". The circuit is
  in service in r9.
- **No other 220 kV element ends at either station.** At Wolmirstedt the 220/400 transformer
  403 was "Removed: decommissioned" (r4) and every other Wolmirstedt row is 380 kV. At
  Parchim/Süd, 328 to Güstrow was "Removed Element" in r9 (v5 CHANGELOG 2) and 321 is in no
  release.
- **The shared stretch:** the second OSM circuit there is 380 kV Ltg 436 Perleberg - Parchim
  Süd, the Ersatzneubau. It is not in r9 (issue 08, case 6).

**Verdict:** the towers carry two 220 kV systems, but in 2026 the TSO models only one circuit
between Wolmirstedt and Parchim/Süd. In JAO r9 this tower line is 1 circuit at 220 kV, so the
model's 1 is correct. The second system has no JAO element. Either it is not operated as a
separate circuit, or it is spare from the 332/322 renumbering. On 8094 both pieces now carry the
reason "JAO r1+r9 list one 220 kV circuit 332-322 (single-circuit X, 900 A); 2nd system has no
JAO element (issue 14 #3/#12)".

The other "many cases like this" follow the same pattern (class a, explained):
Trennfeld - Aschaffenburg 218 (r1 only), Isar - Ottenhofen 444/446 (r6 "Out-of-service", 444
removed r8), Oberbachern - Oberbrunn 269 (r1 only), Isar - Pleinting 452 (issue 09), Silz
273C (r1 only), and 12 lines whose extra OSM circuit has no r9 row.

## Display fixes applied on staging

1. **Routing (b), `src/tranpy/grid_network.py`:** a strict-mode detour (path > `DETOUR_RATIO`
   × the circuit length and 10 km beyond it) is `no_path_strict` and **gets no pieces**. Before,
   it was drawn solid over the foreign tower lines of the long way round. `DETOUR_RATIO` goes
   from 2.0 to 1.5, the upper end of `RATIO_OK`. The 14 circuits now drawn dashed on their
   previous geometry are: Conneforde/O - Elsfleth/W gelb/rot (830 km for 54), Etzenricht -
   Mechlenreuth 437/479 (600 for 90), Röhrsdorf - Weida 573/574 (443 for 66), Wolmirstedt -
   Förderstedt 437/438 (421 for 49), Ludersheim - Schwandorf 223, Altdöbern - Streumen 563,
   BIGGE O/W, RIED W, WESTW O2, and, through the new ratio, Bischofsheim - Urberach ERZHAU N
   (92 for 47, via Kriftel), Buerstadt - Y-Buerstadt NASSAU (161 for 106) and Rommerskirchen -
   Sechtem VORGBG O (52 for 34). Test: `test_strict_detour_has_no_pieces`.
2. **Review rows (a) and approach pieces:** `review_rows.csv` here (74 way rows) is read by
   `scripts/map_edge_review.py --extra` (new option; source `issue14`). The popup shows the
   reason.

Rebuild (cairo-lab, `staging_osmgraph`, code copy in `halo/code`):
`python -m tranpy.grid_network --stations frozen_build/stations --out halo/network --overrides
halo/network/route_overrides.csv`, then `bash halo/export.sh WEB OUT
halo/14-remaining-halos/review_rows.csv` (edge review + export_grid_gis).

Acceptance (`halo/acceptance` against `halo/acceptance_base`, the same scripts on the old
build): `passed` is false in both, with the same three failing checks (drawn_once,
no_gap_over_limit, no_loop). osm_lines_halo goes from 187 to 71 and ratio_outside from 63 to
51. drawn_not_once goes from 4 to 8: Weida - Remptendorf 575/576, Y-Kusenhorst GRAFSH W and
Conneforde/O - Elsfleth/W, all on the dashed previous geometry. Circuits with gaps go from 7 to
9 (NASSAU, Vieselbach - Altenfeld 468): the dashed fallback geometry is coarse. pytest
test_grid_gis, test_grid_network and test_grid_towers: 46 passed. ruff: clean.

## Model proposals, class (c): for the owner to decide (gate), not applied

All nine involve model circuits that come from the OSM export (`osm-*`) or are curated, so
they have no JAO r9 row. The model has every r9 DE circuit except the two known ones (Kriftel
TAUNUS 2 removed on purpose, Grafenrheinfeld 411/416 temporary tee). No r9 circuit is missing
on any halo here.

| # | where | model on the way | evidence | proposal | expected effect |
|---|---|---|---|---|---|
| A016 | Rommerskirchen - Sechtem, Amprion 4511, 23.6 km (6.81/50.96) | VILLE O, VILLE W (r9) + osm-14135-0/1 "Station Sechtem; Umspannanlage Brauweiler" | OSM 2, model 4; the two osm-14135 circuits run on the same towers as the r9 VILLE pair | check osm-14135-0/1 against r9 Brauweiler/Sechtem rows; mark them `duplicate` if they are the same systems | halo gone; 2 fewer parallel 380 kV branches |
| A031 | Bixterheide/Garenfeld 2319, 9.3 km | BIXTRH N, BIGGE W (r9) + osm-5010-0 "Garenfeld; Station Halfeshof" | OSM 2, model 3 | same check, `duplicate` if it matches an r9 row | halo gone; one 220 kV branch less |
| A033 | Remptendorf "Ltg 386", 8.5 km (11.47/50.60) | osm-4706-1-0/1 (6 osm-4706 circuits in all, Ltg 383/384/386, estimated) | OSM 1, model 2; issue 09 rank 402 already model_error | reduce osm-4706 to the OSM circuit count per section, or drop the estimated rows that are not in r9 | halo gone; fewer estimated 220 kV parallels near Remptendorf |
| A042 | Zwönitz 577/578, 4.5 km | curated Zwoenitz - Roehrsdorf 577/578 + osm-14312-0/1 Markersbach - Zwönitz - Röhrsdorf | OSM 2, model 4; the two chains overlap on the same ways (issue 09 rank 55) | keep one chain: curated 577/578 to Zwönitz plus only the Markersbach - Zwönitz part of osm-14312 | halo gone; no double 380 kV path Röhrsdorf - Zwönitz |
| A059 | Niederaußem 220 kV 4548, 2 km | osm-4291-0..3 (4 circuits) | OSM 2, model 4 on the shared section | split osm-4291 per OSM section (2 + 2) instead of 4 parallel | halo gone; plant connection impedance corrected |
| A072 | Schwelgern, 0.8 km | osm-5127-0..2 (3) | OSM 2, model 3 | set the count to the OSM `circuits=2` | halo gone |
| A060 | Schwelgern/Beeck, 1.8 km | osm-4386-0 (1) | OSM 2, model 1 | add the second circuit only if a second system is confirmed (OSM `circuits=2`, industrial line, no JAO data) | halo gone; small (industrial 220 kV) |
| A061 | 220kV LMG Ost+West 2437, 1.8 km | osm-4306-0 (1) | OSM 2, model 1; the name says Ost+West = two systems | add the second system (OSM name and count agree) | halo gone; small |
| A063 | 220kV Nordleitung+Südleitung, 1.5 km | osm-4329-0 (1) | OSM 2, model 1; the name says two lines | add the second system | halo gone; small |

## Open routing artefacts, class (b): halo kept

- **Missing terminals (18 findings, 309 km).** GRAFSH W / PRHAMSBW at Gronau/Y node
  (A007/A011/A019/A062; 4305/4306). Laufenburg (CH) for Laufenburg - Trossingen rt and Kühmoos -
  Laufenburg br/ge/rt/ws (A008/A018/A025/A047/A051/A053/A078). Asphard - Eichstetten rt/sw
  (A002, Kühmoos - Daxlanden). Bergshausen - Würgassen 1 (A028). Weissenthurm - Y-Koblenz
  SOONWD O (A035). Altbach - Wendlingen bl (A038). Conneforde - Fedderwarden blau/rot (A045).
  Maasbracht - Siersdorf SELFK SW (A048). Y-Pfungstadt RIED O (A074). Each needs a terminal
  (station or Y node) in the station table: a stations-step change, then a gate.
- **No strict path (14, 179 km):** Preilack - Streumen 560 (override, A006). Pirach/Pleinting -
  St. Peter 256/258 and Simbach - St. Peter 233/234 at the AT border (A009/A017/A024). Giessen/N
  - Dillenburg (A021; Asslar 4 and WESTW O2 belong there). Wolmirstedt - Förderstedt 437/438
  (A030). RIED W (A039). VORGBG O around Brauweiler (A040/A049/A073/A077/A080). Röhrsdorf -
  Weida 573/574 (A075). WIESDF (A065). These need an OSM join at the break (`break_lon/lat` in
  `circuit_routes`) or a route override.
- **Parallel ways (8, 52 km):** Audorf/S - Handewitt / Schuby (A015/A020/A050/A066; 4 circuits
  on one of two parallel ways), Landesbergen 3004, Daxlanden - Weier, Neurott,
  Hamaland/Münsterland.
- **Odd ratio (3, 9 km):** BLUDNZ W schematic legs (A041/A055), Ludersheim - Raitersaich 237 (A070).

## Unclear, class (d): halo kept

- A010 Waldlaubersheim - Weissenthurm SOONWD (41 km, OSM 2 / model 1). NASSAU (now dashed)
  and SOONWD O / Koblenz (no terminal) lie within 5 km. One of them is probably the second
  circuit.
- A014 "302" Unterweser - Conneforde (32 km). Conneforde - Fedderwarden blau/rot (no terminal)
  lie nearby.
- A022 OCHSMO O/W + Ohlensehlen - Wehrendorf DS1 on 4196 (16 km, OSM 2 / model 3). All three
  are r9 and no parallel way was found. Issue 09 rank 24 suspected that DS1 runs elsewhere.
- A027 Trebur/Urberach 4591 (12 km, OSM 4 / model 1). ERZHAU N and RIED W (now dashed) lie
  nearby.
- A036 Neurath 4517 (7.6 km). VORGBG O (dashed) lies nearby.

## Method and limits

`occurrences.csv` columns: evidence, jao_numbers (each OSM ref number with its r9 and r1 rows),
r9_changelog (removal/out-of-service rows for those numbers), confidence. The rules
(`halo/rules.py` and `halo/build.py` on cairo-lab):

- (a) no_r9_counterpart needs three things: no model circuit (the model has all r9 DE
  circuits) of the voltage whose straight end-to-end line runs along at least 50 % of the
  finding within 3 km; no unrouted circuit within 5 km; and no parallel way.
- (a) osm_undercount needs: every EIC in r9, route/model ratio 0.7-1.5, and no parallel
  no-model way within 250 m over at least 30 % of the length. Confidence is medium: OSM may
  simply undercount, or a second tower line may be missing in OSM.

Hand overrides: #2, #3, #5, #7, #8, #11, #12, #21, #22, #28, #34 (evidence in the rows). The
24 approach pieces (< 0.5 km) follow the issue-09 coordinator rule; their cause is listed in
their evidence.

## Explained-row audit (explfix, 2026-11-09)

See `map_changes.md`, which records each change with its evidence, the per-change diffs, and the open cases that need the owner.
