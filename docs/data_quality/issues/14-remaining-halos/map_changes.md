# Map change table (issue 14, staging map 8094)

Rule (owner, 2026-11-09): every change needs evidence from a source independent of both OSM and our model, and every change is tested on its own. The tests are acceptance before and after, screenshots before and after, and a per-change layer diff. Where the evidence is not conclusive, nothing is changed and the case is listed as **needs owner**.

The model is not changed: everything below is display or routing of the frozen model `model_v5_frozen.pkl`, sha256 `3393038e…`.

Screenshots are in the session scratchpad `shots/explfix/` as `<name>_before.png` and `<name>_after.png`. They were taken at the same place with the same zoom: "before" is `web_frozen_before_explfix`, "after" is the live 8094 build `20261109T215602Z`.

## Staging and per-change diffs

On cairo-lab, everything lives under `~/data/grid/staging_osmgraph/halo/stage`. Networks are diffed with `rdiff.py`; exports with `ldiff.py`, which lists every circuit whose drawn geometry changed and every OSM way whose check class, circuit set or reason changed.

| Step | Network | Export | Diff against the previous step |
|---|---|---|---|
| E0 | before | `web_frozen_before_explfix` | (baseline) |
| E1: display code | before | E1 | 0 circuits, 0 ways. The raw layer `tower_lines` loses 7 model columns (C4). |
| E2: HVDC block | n0 | E2 | 0 circuits. Two ways swap: 908175880 / 1095719612, see note N1. |
| E3: WESTW O2 | n1 | E3 | 1 circuit (WESTW O2), 16 ways, all on the O2 path or the Westerwald corridor. |
| E4: Elsfleth join | n2 | E4 | 1 circuit (Conneforde/O - Elsfleth/W gelb/rot), 4 ways in the triangle, and the N1 swap back. |
| E6: routing reason | n2 | E6 → `web_frozen` | 0 circuits. 1 way (30534192): only the popup reason changes. |

Over the whole run (E0 → E6), exactly 2 circuits change their drawing: WESTW O2 and Elsfleth gelb/rot. 20 OSM ways change, all on those two routes. Nothing changed outside the intended places.

Acceptance:

| Build | Hard failures | drawn_not_once | gaps > limit | loops | halos | explained | ratio outside | halo_popup_consistent |
|---|---|---|---|---|---|---|---|---|
| Before | drawn_once, no_gap_over_limit, no_loop | 8 | 22 | 7 | 71 | 300 | 51 | (new check) |
| After | same three (pre-existing) | 7 | 19 | 5 | 76 | 296 | 52 (WESTW O2, see C1) | pass, 0 |

## Changes

### C1: Westerwald, WESTW O2 (Dauersberg - Dillenburg, tie 11T-D2-D7-00020D)

**What changed:** override `accept_detour` (in `halo/network/route_overrides.csv`). O2 was a dashed strict detour; it is now drawn on its OSM ways, 75.4 km, along Dauersberg → Westerwald West+Ost (4124/4564) → 138220867 → Asslar junction → 345675089 → Dillenburg.

**Evidence:** JAO Core Static Grid Model, 9th release (202603, `core_static_grid_model_202603.zip`), sheet Tielines:
- row 21: "Dauersberg - Dillenburg WESTW O2", AMPRION, 380 kV, 38.0 km;
- row 276: "Dillenburg - Dauersberg WO2", TENNETGMBH, 380 kV, 40.64193 km.

The tie is 78.6 km in total, so the OSM route is at ratio 0.96 against it. The model's 35.85 km is one TSO's half, which is why the map ratio (2.10) stays on the ratio list.

**Dillenburg (way 345675089):** JAO r9 lists only "Giessen/N - Dillenburg 3" (L sheet, 25.74 km) and the WO2 tie toward Dillenburg. Both are now on the way: match (2/2).

**Diff:** E2 → E3, as above. Screenshots: `01_westw_138220867_popup`, `02_westw_o2_overview`, `02b_westw_dauersberg_side`, `03_dillenburg_345675089_popup`.

**Side effect (needs owner):** five Westerwald West+Ost ways on the Dauersberg side (159705778, 160573446, 352376142, 352376153, 908473140) now carry O2 only, 1 of 2. Way 908473137 now shows 3 of 4. All six show a model_fewer halo.
- In r9, Dauersberg has LIMBG W, WESTW W1 and WESTW O2 for the two parallel lines (Westerwald West+Ost and Limburg Ost+West). That is 3 circuits for 4 OSM slots.
- The router draws the W1 Dauersberg leg on the Limburg line.
- Which line carries W1 is not settled by an independent source, so no reason is attached and the halo stays.

### C2: Conneforde – Unterweser – Elsfleth/W triangle (Conneforde/O - Elsfleth/W gelb/rot, 11T0-0000-2299-H)

**Cause:** the strict join rule needs the direction to continue. Branch 320 (way 30534192) ends at tower node 337349472, 36 m from anchor tower 337349424 of line 302, at a right angle, so the rule refused the join.

**What changed:** no general rule change. A general "tee" rule was tried and reverted, because it also re-routed Grosskrotzenburg - Mecklar 1. Instead, one reviewed join is listed in `halo/network/join_overrides.csv` (new `--join-overrides`; the join edges are appended, so all other edge ids stay the same). Gelb/rot is now routed at 59.9 km, ratio 1.10.

**Evidence:**
- JAO r9, L sheet: "Conneforde/O - Elsfleth/W gelb/rot", 380 kV, 54.24126 km.
- Esri World Imagery z19 tiles (fetched 2026-11-09): the two lattice towers stand next to each other at 8.4715/53.4294. Image: `shots/explfix/elsfleth_gap_esri_z19.png`, red circle = end of 320, yellow = anchor tower of 302.
- OSM API (2026-11-09): node 337349424 is `power=tower`, `line_attachment=anchor`, refs 1/301; node 337349472 is `power=tower` (version 2, 2010).

**Result:** the Conneforde – Unterweser line (157160048) is now a match (2/2). JAO r9 for that line: "Unterweser - Conneforde rot" (220 kV, 31.64 km) plus gelb/rot.

Branch 320 (30534192) is now 1 of 2: r9 "Unterweser - Elsfleth/W weiss" (380 kV, 27.08 km) exists, but unterweser|380 has no OSM terminal, so weiss is drawn schematic. The popup reason says so; the halo stays.

**Diff:** E3 → E4. Screenshots: `08_triangle_clean`, `09_triangle_raw`, `10_triangle_157160048_popup`.

**Other circuits dashed with a gap under 100 m:** only NASSAU (Buerstadt - Y-Buerstadt) and VORGBG O (Rommerskirchen - Sechtem), both with a 0 m gap. They are ratio detours, not join-rule cases: unchanged, needs owner.

### C3: HVDC ways blocked for AC routing

**What changed:** ways that are DC-only (`frequency=0` only, or named only after an HVDC link: A-Nord, SuedLink, SuedOstLink, Ultranet, NordLink, …) stay in the graph but are never used by an AC circuit (`is_hvdc_way`). Shared AC/DC towers and AC converter connections are not affected.

**Evidence:** Amprion, https://www.amprion.net/Netzausbau/Unsere-Projekte/A-Nord/ (fetched 2026-11-09): "eine rund 300 Kilometer lange Gleichstromverbindung von Emden nach Osterath, die als Erdkabeltrasse gebaut wird."

**Result:** no model circuit used any of these ways before or after (10 ways matched in the caches). Drawing diff: 0 circuits.

### C4: raw OSM layer shows OSM tags only; underground cables dashed

**What changed:** the web `tower_lines` layer no longer has `model_circuits`, `model_names`, `model_eics`, `model_kv`, `mismatch`, `other_voltage_circuits` or `parallel_way`. These came from the old way-level count, which was the A-Nord "Doetinchem - Niederrhein DOETM SW" leak and the Elsfleth "201" leak. The popup has no "Carried model circuits" block. It adds construction and underground notes, and `location=underground/underwater` ways are dashed.

**Evidence:** the A-Nord quote in C3. Doetinchem - Niederrhein is an AC tie, and the clean layer never drew it on A-Nord.

**Diff:** E0 → E1: 0 circuits, 0 ways. Screenshot: `11_anord_osterath_raw_popup`; the before shot shows the old "Carried model circuits" block.

### C5: halo/popup consistency check, cache-busting

- `scripts/map_acceptance.py` has a new hard check, `halo_popup_consistent`: the class follows from model_n vs circuits_tag, and the popup list has model_n entries. 0 problems before and after.
- Cache-busting: `metadata.json` is fetched no-store, and every data file is fetched with `?v=<build_id>`. The index.html asset references are stamped with the build id, and the panel shows "Build … · model sha256 …". It has no effect on map data.

### C6: Simbach cluster, Pirach – Y-Simbach – St. Peter (AT) – Pleinting (220 kV)

**Cause:** the routing extracts had no Austrian ways inside the St. Peter yard. The German crossing "256;258" (way 125690616) ended 310 m short of the `st peter apg|220` terminal, so 256 and 258 were no_path_strict, and the Pirach and Pleinting legs showed a halo (only 257 counted).

The bus is matched by bus id: `st peter apg|220`, kind `foreign`. It is not matched by name, so it is not confused with Amprion's St. Peter in the Rhineland.

**Data:** an Overpass fetch on 2026-11-09 (bbox 48.20,12.90,48.32,13.20; OSM base 2026-11-09T21:59:58Z) returned 57 new 220/380 kV ways. Only the 5 ways the two circuits need are added to the routing input (`halo/network/overpass_stpeter_min.json`). They are the APG yard ways 821951778 (258), 821951779 (256), 821951766, 821951767 and 1087935463. With the full set, the clean layer filled up with Austrian no_model ways, so the full set was not used.

**Evidence (JAO Core SGM r9, 202603):**

| Circuit | Lines / Tielines rows | Model km | Routed km | Ratio |
|---|---|---|---|---|
| Pirach - St. Peter 256 (10T-AT-DE-000029) | Tielines 302 (42.38 km); 72 (APG part, 1.9 km) | 42.38 | 41.3 | 0.97 |
| Pleinting - St. Peter 258 (10T-AT-DE-000037) | Tielines 303 (46.45 km); 73 (APG part, 1.9 km) | 46.45 | 48.6 | 1.05 |

The OSM crossing is about 2 km from the split at 13.0588/48.268 to the APG yard, which matches the 1.9 km APG part. The OSM name of the crossing is "St. Peter - Pirach;St. Peter - Pleinting". Esri World Imagery z16 of the Inn crossing: `img_inn_crossing_z16.png`.

**What changed:**
- 256 and 258 are routed into the St. Peter yard.
- The Pirach leg (91084404, 91086218, 115569716) and the Pleinting leg (73306291) are now matches: 4/4 and 2/2.
- The crossing (125690616) is a match (2/2).

**Diff (E6 live → E8 → E9):** 4 circuits change drawing: 256 and 258. The two dashed Y-St. Peter legs also move their station end by about 60 m, because the St. Peter terminal entry moved; their status is unchanged. 14 OSM ways change, all inside the cluster.
- Way 125690615 ("256", OSM 1) goes from explained to model_more 3 (256 + the two Y-Altheim legs). This is because of the model tee placement; see below.
- Cables 1293441399 and 1293441397 (OSM 1 each): 257 and 258 are now both drawn on cable 1293441399, while the parallel single-circuit cable 1293441403 carries none. The old issue-09 "count artefact" explanation would have hidden this, so a routing row keeps the purple halo. Which circuit uses which cable is not documented: needs owner.

**Acceptance (E6 → E9):** halos 76 → 75. All other metrics are unchanged, the same three pre-existing hard failures remain, and `sld_complete` passes.

**Screenshots:**
- `20_simbach_cluster`, `21_simbach_tee`, `22_simbach_tee_raw`, `23_stpeter_inn_crossing_popup`, `24_pleinting_cables`, `25_pirach_leg_91086218_popup` (each with `_before` / `_after`);
- imagery: `img_split_1057547856_z17.png`, `img_split_1057547914_z17.png`.

**Needs owner (not changed):**
- **Y-Simbach tee location (model location proposal):** the model tee (233/230, 234/230) sits at 13.0588/48.268, the OSM 256/258 split. In OSM, 230/233/234 meet at node 1057547856 (12.9982/48.2953), where the 230;234 branch leaves for Simbach. Esri imagery z17 shows that branch tower and the branch line: `img_split_1057547856_z17.png`. The JAO lengths do not settle it:
  - Y-Simbach leg 4.28 km: both candidate points are about 4 km from Simbach.
  - Y-Altheim leg 80.0 km.
  - Y-St. Peter tieline rows 316/317: 75.37 km, which fits neither point. Rows 74/75 give an APG part of 1.9 km.
  - Proposal: check with APG/TenneT whether the tee is at 12.9982/48.2953, and if so move the model's Y-Simbach node there. Until then the display is not moved.
- **Y-St. Peter (-Simbach - Altheim) 233/230 and 234/230:** with the St. Peter data the router drew these 3.1 km from the model tee over the 256;258 crossing. OSM, however, names them on the 230;233 crossing (125690613), and JAO gives 75.37 km. The sources disagree, so both are held dashed by override (`route_overrides.csv`, reason given).
- **Y-Simbach (-St. Peter - Altheim) legs to the Simbach plant/station:** still dashed (break 3 km). This depends on the tee question.
- **"230;234" (1428461555):** a German 220 kV border way. It carries no model circuit until the tee is resolved.

### C7: regression, single-line diagrams lost in rebuilds

**What happened:** the halo-review exports and my E6 build ran `export_grid_gis.py` without `--diagrams`. As a result `sld/` was empty and no station named a diagram. The coordinator restored `sld/` by hand.

The earlier halo build (`web_frozen_before_halo`) had 541 named diagrams. The `web_frozen_before_explfix` baseline already had 0 named diagrams; its files had been copied in separately.

**Fix:**
- `halo/export.sh` and `halo/stage/export_stage.sh` now pass `--diagrams`. The SVGs are regenerated from the frozen model; that build took 37 s.
- New hard acceptance check `sld_complete` (`scripts/map_acceptance.py`, unit test `test_sld_complete`): every station in `station_details.json` names a diagram, the file exists, and `sld/` is not empty.
- Live build `20261109T220733Z`: 541 stations, 541 diagrams, 0 problems.

### C8: Simbach / St. Peter (Inn crossing): project note on way 1293441396; 257/258 cables and the Y-Simbach tee stay "needs owner"

Baseline is the live v5.1 build `web_v5_1` (`20261109T225332Z`, model `model_v5_1_frozen.pkl`, sha256 `cc4589c1…`). The routing network `halo/network` is unchanged.

**(b) Way 1293441396 (380 kV, circuits 4, cables 12, 5.37 km, OSM `power=construction`, v2 2025-11-22): applied, popup note only.**
- Project: TenneT line B153 "Umspannwerk Simbach am Inn – Landesgrenze St. Peter (AT)", part of BBPlG Vorhaben Nr. 32 "Höchstspannungsleitung Altheim – Bundesgrenze (AT) – Pleinting mit Abzweigen … Matzenhof – Simbach" (NEP TTG-P67 + TTG-P112, PCI 2.1.4). On the Austrian side it is APG's "Deutschlandleitung".
- Evidence:
  - APG, https://www.apg.at/en/projekte/germany-line/ (fetched 2026-11-10): four 380 kV systems from St. Peter to the German border; "Commissioning … is planned for 2027"; update 17.08.2026 "cable pulling has begun"; the existing 220 kV lines St. Peter – Simbach/Altheim and – Pleinting are dismantled after commissioning. Strong.
  - NEP 2037/2045 (2025) project sheet TTG-P112, dated 19.05.2026 (https://www.netzentwicklungsplan.de/projektbibliothek/download/58789/…P112…pdf, fetched 2026-11-10): "BBP-Nr: 32"; M201 Pleinting – Bundesgrenze DE/AT, "Neubau einer 380-kV-Doppelleitung … zwischen Pleinting und St. Peter", "Geplante Inbetriebnahme: 2030"; "Die bestehende 220-kV-Infrastruktur in der Region wird nach Inbetriebnahme von TTG-P67 sowie P112 zurückgebaut." Strong.
  - Regierung von Niederbayern, UVP-Vorprüfung 1. Planänderung (uvp-verbund.de, search excerpt 2026-11-10): Planfeststellungsbeschluss of 16.01.2023 for the "Neubau der 380/110-kV-Freileitung Umspannwerk Simbach am Inn – Landesgrenze St. Peter (AT), Ltg. Nr. B153". Medium (excerpt only).
  - JAO Core SGM r9 (202603), Lines/Tielines: the only St. Peter – Pirach/Pleinting/Simbach/Altheim elements are 220 kV (L rows 2363/2364/2419/2420; T rows 72–75, 302, 303, 316, 317). The change log has no 380 kV element there. Strong.
- Result: none of the four circuits is in service in the v5 hours (Nov 2025 – Oct 2026). The way stays out of the model and out of the clean layer; it already was (no `osm_lines` row; raw layer `construction=true`).
- **What changed:** new optional export input `--way-notes` (CSV `way_id,note`, file `halo/network/way_notes.csv`, one row). The raw `tower_lines` popup shows the note under the construction line. Code: `src/tranpy/grid_gis.py` (`export_grid_gis(way_notes=…)`), `scripts/export_grid_gis.py` (`--way-notes`), `apps/grid_map/app.js` (towers popup). Synced to `halo/code` and the staging `app/app.js`.

**(a) Which 220 kV circuit is on which parallel cable (1293441399 / 1293441403): not resolved, needs owner.**
- Both ways are 2.5 km single-circuit overground cables between 13.058/48.274 and 13.064/48.294, tagged `note=Bauzeitliches Provisorium` (OSM v1, 2024-06-18). They are the construction bypass of the old 220 kV line while B153 is built in its route; 1293441396 runs between the same two points.
- No OSM route relation, TenneT/APG map or Planfeststellung document found names the circuit per cable. The Niederbayern Planänderungsbescheid of 12.08.2024 (…/20240812_planaenderungsbescheid1_270225.pdf) describes cable provisoria ("Baueinsatzkabel") on line B104 (Adlkofen – Matzenhof), not this place. No routing change: 257 and 258 stay drawn on 1293441399 with the routing halo (C6).

**(c) Y-Simbach tee: no public source places it; KNOWN_LIMITS proposal.**
- Searched: APG Deutschlandleitung page, NEP TTG-P112 sheet (it describes only the Pirach branch: "Dieser Abzweig wird teilweise auf der bestehenden 220-kV-Leitung Altheim – St. Peter mitgeführt"), Niederbayern Planfeststellung notices for B153, Prienbach – Pleinting and Adlkofen – Matzenhof, and the JAO r9 change log (only renames of the Y-Simbach elements, 2023-03-01, r3). None gives a tee coordinate.
- Proposal for KNOWN_LIMITS (owner's gate; the frozen KNOWN_LIMITS.md is not edited): "**L-C8 Y-Simbach tee.** The 220 kV tee of 233/230 and 234/230 (Altheim – Simbach – St. Peter) sits in the model at 13.0588/48.268, the OSM 256/258 split. OSM puts the 230/233/234 branch at node 1057547856 (12.9982/48.2953), and Esri z17 imagery shows a branch tower there. No TSO, JAO or planning source places the tee; JAO r9 lengths (Y-Simbach 4.28 km, Y-Altheim 80.0 km, Y-St. Peter 75.37 km) fit neither point. Y-St. Peter 233/230 and 234/230 stay dashed and way 1428461555 carries no model circuit. Electrical effect: none for impedances (JAO R/X/B are used); only the drawn tee location. The 220 kV structure is removed after BBPlG Nr. 32 is commissioned (2027–2030)."

**Diff (`web_v5_1` → `web_v5_1_c8`, `halo/c8c9/layerdiff.py`):** all layers identical except `tower_lines`. There, only way 1293441396 gets a note, and the other 10 962 features get an empty `note` property. 0 geometries changed. `lines`, `osm_lines`, `substations`, `station_details.json`: identical. `sld/`: 541 files, byte-identical.

**Acceptance (`halo/c8c9/acc_web_v5_1*`):** the before and after reports are identical apart from the header. Same three pre-existing hard failures (drawn_once, no_gap_over_limit, no_loop). `sld_complete` ok; `halo_popup_consistent` ok (0 mismatches); halos 75; explained 296; drawn_not_once 7; gaps 19; ratio outside 52.

**Screenshots** (`shots/c8c9/`, `_before` = `20261109T225332Z`, `_after` = `20261110T082606Z`): `c8_simbach_inn_380_raw_popup`, `c8_simbach_cables_clean_popup`, `c8_simbach_cables_raw`, `c8_simbach_overview`.

**Live:** `app/data` → `../web_v5_1_c8`. Rollback: `app/app.js.before_c8`, `app/index.html.before_c8`, and `data` → `../web_v5_1`.

### C9: Dauersberg / Westerwald (OSM ref 4564): diagnosis only, no change

**(d) Circuits on 4564 (way 152403657, 380;220 kV, circuits 4).** The map already shows a match (4/4): HELLER O and HELLER W (220 kV, Dauersberg – Setzer Wiese), LIMBG W and WESTW W1 (380 kV).
- JAO r9, L sheet, every Dauersberg 220 kV element: rows 266/267 HELLER O/W (47.51/49.10 km), 305/306 LEUSHD N/S, 412/413 SIEGRL O/W. All six are in the model.
- OSM relations on the way: "Heller Ost" (220 kV, Amprion), "Limburg West" and "Limburg Ost" (380 kV), "Struthütten Süd" (110 kV).
- **No 220 kV circuit is missing, so there is no model proposal.** The 220 kV pair on 4564 is HELLER O/W.

**(e) W1 placement: needs owner.**
- OSM names the 380 kV pair on 4564 "Limburg West" + "Limburg Ost". JAO r9 has no "LIMBG O" element. The routed WESTW W1 (Dauersberg – Y-Dauersberg – Limburg/Asslar, L rows 478/479, T rows 33/311) would fill the "Limburg Ost" slot, which is what the router draws.
- This rests on OSM naming only; no Amprion or JAO source states the tower position. The C1 side effect (O2 alone on the Westerwald West+Ost ways, orange halo) stays.

**(f) Dashed lines NE of the Dauersberg box: "Dauersberg - Eiserfeld SIEGRL W" (15.74 km) and "Dausersberg - Eiserfeld SIEGRL O" (17.28 km), 220 kV, no_path_strict (break at 7.9075/50.8097, gap 10 km).**
- JAO r9 L rows 412/413 list both, so they are in service.
- The JAO change log (r4, 2023-09-20) has "Dausersberg - Eiserfeld SIEGRL O … Reconstruction of networkelement".
- Cause: the OSM 220 kV way 2471 (152403715, relation 2137096 "Siegerland West", from Dauersberg to Eiserfeld) is only 0.27 km. It ends at node 10537279047, where the new ways 1130668498/500/503/496 and 1436420639 (380;110 kV, circuits 4) and 1130835044/1254852017 (380 kV, circuits 2) continue north-east. These are the new 380 kV lines Kruckel – Dauersberg (Bl. 4319) and Punkt Mudersbach – Eiserfeld (Bl. 4219). Evidence: SGD Nord, Bekanntgabe 28.01.2020 (Az. 21a-7.110-011-2019, uvp-verbund.de), "Neubau der rheinland-pfälzischen Abschnitte der 380-kV-Höchstspannungsfreileitung Kruckel – Dauersberg (Bl. 4319) und der 380-kV-Höchstspannungsfreileitung Punkt Mudersbach – Eiserfeld (Bl. 4219)", Planfeststellungsbeschluss 26.06.2015. The strict router keeps 220 kV circuits off 380 kV-tagged ways, so SIEGRL has no path.
- Why no fix: the SIEGRL circuits are probably operated at 220 kV on the new 380 kV towers. A search excerpt of an NRW Planfeststellung text says that a 380 kV section of Kruckel – Dauersberg (Oberschelden – Landesgrenze) is operated "übergangsweise in der 220-kV-Spannungsebene". That source covers another section; for this segment I found no TSO, JAO or planning source naming the circuits on these ways or their operating voltage. amprion.net (Kruckel-Dauersberg page, 2026-11-10) says only "Der südliche Teil von Dauersberg nach Hünfelden ist bereits fertiggestellt" and "Gesamtinbetriebnahme in 2030", and the D/E construction flyer returns 404. Allowing 220 kV circuits on these ways would be a general rule change with no source for this segment. Kept dashed: **needs owner** (proposal: a reviewed `accept_detour`/voltage exception for SIEGRL O/W on Bl. 4219/4319 once Amprion confirms 220 kV operation).

**Diff and acceptance:** none; no change.

**Screenshots** (`shots/c8c9/`, before/after are the same data): `c9_dauersberg_overview`, `c9_dauersberg_4564_popup`, `c9_dauersberg_2471_popup`, `c9_dauersberg_ne_raw`.

### C10: HVDC interconnectors drawn along their real route (display only)

**Scope.** These are the DE HVDC links in service between Nov 2025 and Oct 2026. The electrical model is unchanged: each link stays a fixed injection at `dispatch.HVDC_CONVERTERS`.

| Link | In service | Before | After: geometry source | OSM route vs published |
|---|---|---|---|---|
| NordLink (Wilster – Tonstad/Ertsmyra), 1,400 MW, ±525 kV, 2021 | yes | straight dashed line Wilster → Ertsmyra | OSM ways 667267039 (DE land), 370147304 (sea), 842077313, 641089872, 629694685 (NO overhead) | 616.5 km vs 623 km (Statnett); end gaps 0.13 / 0.33 km |
| ALEGrO (Oberzier – Lixhe), 1,000 MW, 320 kV, 2020 | yes | straight dashed line | OSM relation 8193755, ways 578137667 (DE), 578137663 (BE) | 89.5 km vs ~90 km (Siemens/Amprion); end gaps 0.36 / 0.01 km |
| Baltic Cable (Herrenwyk – Kruseberg), 600 MW, 450 kV, 1994 | yes | **not drawn** | OSM ways 79092545, 217974209, 889061693, 235735184, 79088887 | 259.0 km vs 260 km (balticcable.com); end gaps 0.02 / 0.04 km |
| Kontek (Bentwisch – Bjæverskov), 600 MW, 400 kV, 1995 | yes | **not drawn** | OSM ways 159089622, 159089626, 159089624, 159089627 (sea), 157883495 (DK land) | 166.0 km vs ~170 km (Hitachi Energy / NKT); end gaps 0.10 / 0.20 km |

**Not drawn, with reasons:**
- Kriegers Flak CGS is not an HVDC transmission link. It is an AC offshore connection with a 410 MW back-to-back converter at Bentwisch (Hitachi Energy KF CGS page, "back-to-back", 410 MW/150 kV). The model already includes it in the DK2 injection at bentwisch|380.
- COBRA is NL–DK.
- NeuConnect (DE–GB) and Bornholm Energy Island / BEI (OSM ways 1432087620 and 1573808238, 50Hertz) are under construction or planned, not in service.
- BorWin/DolWin/etc. are offshore wind connections, not interconnectors.

**Verification.** Every route below is labelled *approximate* in the popup.
- **Endpoints:** both ends of each OSM chain lie within 0.4 km of the converter or station that the operator names.
- **Length:** each OSM length is within 1–3 % of the published length.
- **Not done:** no route was traced against an operator route map; the TenneT and Amprion project pages returned 403/404 on 2026-11-10.

**Sources** (all accessed 2026-11-10):
- Statnett, "NordLink in operation" (statnett.no/en/about-statnett/news-and-press-releases/news-archive-2021/nordlink-in-operation2/): "is 623 km long and runs between Sirdal municipality in Norway" and Wilster; "capacity of up to 1400 MW"; "The takeover from the suppliers took place 31.03.2021".
- Amprion press, 1-year ALEGrO (amprion.net/Press/Press-Detail-Page_37505.html): "maximum transmission capacity of 1,000 MW", "commissioned … (18 November 2020)". The 320 kV DC figure is from Amprion's planning document Anlage 1 D1 (search excerpt: "320-kV-Hochspannungs-Gleichstrom-Verbindung").
- Baltic Cable AB (balticcable.com): "It runs 260 kilometres between Lübeck in Germany and Kruseberg in Sweden"; "GERMAN Converter-station in Herrenwyk".
- Hitachi Energy, Baltic Cable: "capacity of 600 MW at 450 kV DC", 1994.
- Hitachi Energy, Kontek case study (search excerpt): 600 MW, 400 kV DC, 1995, 52 km submarine + 119 km land cable.

**Implementation** (staging only).
- Script: `halo/c10c11/hvdc_display.py` (with `routes.json`, built by `halo/c10c11/chain.py` from Overpass `overpass/q2.json`, OSM base 2026-11-10T08:39Z).
- It post-processes a **copy** of `web_v5_1_c8` into `web_v5_1_c10`. Only `lines.geojson` changes.
- NordLink and ALEGrO keep their model properties. They get the OSM geometry plus the popup fields `hvdc_operator`, `capacity_mw`, `dc_kv`, `commissioned`, `hvdc_source`, `hvdc_route` and `published_length_km`, and `route_note` reads "approximate".
- Baltic Cable and Kontek are added as `display:hvdc:*` features (`data_quality=display_only`). Their `route_note` says that they are not model circuits and names the injection node.
- `app.js` FIELDS.lines has the seven popup fields. They are absent on AC lines, so those popups are unchanged. The edit is in staging `app/app.js` (backup `app.js.before_c10`) and in the repo `apps/grid_map/app.js`.

**Diff** (`halo/c10c11/layerdiff_c8_c10.txt`):
- `lines.geojson`: 6153 → 6155 features. Geometry and properties changed only on "Nordlink Cable" and "Oberzier - Lixhe ALEGRO", and 2 features were appended.
- Every other layer and `station_details.json`: **same**.
- sld: 541 = 541 files, content identical.

**Acceptance:** `halo/c10c11/acc_c10/acceptance.json` is **identical** to `halo/c8c9/acc_web_v5_1_c8/acceptance.json` (the export name aside), and `events.csv` is identical. The checks skip HVDC rows, and the same known FAILs (drawn_once, no_gap_over_limit, no_loop) are carried over unchanged. `sld_complete` = true.

**Swap:** `app/data` → `../web_v5_1_c10`. Rollback: `ln -sfn ../web_v5_1_c8 app/data` and restore `app/app.js.before_c10`.

**Screenshots:** `shots/hvdc/` (before = c8, after = c10): `c10_overview_north`, `c10_nordlink_popup`, `c10_baltic_cable_popup`, `c10_kontek_popup`, `c10_alegro_popup`, `c10_herrenwyk_luebeck`.

### C11: Baltic Cable connection (research; no model change)

**(a) Real connection (evidence strength in brackets)**
- **Converter** Herrenwyk: 600 MW, 450 kV DC (Hitachi Energy) [strong].
- **AC side 380 kV** [medium]:
  - OSM way 275466323 is "HVDC Herrenwyk (Baltic Cable)", operator Baltic Cable AB, voltage 440000;380000.
  - Wikipedia (as a pointer only) says the 380 kV line "which begins at the converter station of Lübeck-Herrenwyk ends at the Lübeck-Siems substation", is about 3 km long, was built in 1993, and has "no transformer for coupling the 380 kV- and the 110 kV-grid" at Herrenwyk.
  - No TSO document seen.
  - The model line osm-14551-0 (2.8 km) matches.
- **Herrenwyk – Siems 380 kV is not in JAO r9** (no Herrenwyk row in any sheet) [strong, absence]. It is probably the radial converter connection, outside the Core SGM.
- **Siems 380/220 kV T421:** r9 Transformers row "TR Siems T421" (11TD2T000000172L), 380/220 kV, Imax fixed **631 A**, R 1.07 Ω, X 86.44 Ω. That is ≈415 MVA if the current is counted on the 380 kV side, which r9 does not state. Imax was updated in r4 (2023-09-20) and r6 (2024-08-30) [strong].
- **Lines leaving Siems/Lübeck in r9** (all 220 kV, TenneT):
  - "Siems - Luebeck blau" 950 A fixed, 9.991 km. B = 725.7 µS on 10 km means a cable, which matches the 2004 220 kV underground cable Siems – Lübeck(-Bargerbrück) per Wikipedia.
  - "Hamburg/N - Luebeck rot" and "Hamburg/N - Luebeck gelb": both DLRmin 949 A / DLRmax 1200 A, 48.8 km.
  - **No 380 kV line at Siems, Lübeck, Herrenwyk or Göhl** in r9 Lines, Tie-lines or the change log.
- **Ostküstenleitung 380 kV** (Kreis Segeberg – Raum Lübeck – Siems / Göhl):
  - Land SH, "Dialogverfahren Ostküstenleitung", last updated 04.11.2025: "Seit Juli 2025 sind alle drei Abschnitte … genehmigt, und die Leitung ist im Bau"; "2027 soll die Ostküstenleitung auf ganzer Länge in Betrieb gehen".
  - topagrar, 24.08.2025: "Die 380-kV-Leitung soll Ende 2027 in Betrieb gehen"; "zwischen Stockelsdorf und Lübeck-Siems jetzt die Leitungen eingehängt".
  - So it is **not in service in Nov 2025 – Oct 2026** [strong]. It must not be added.

**(b) Missing elements and the gelb status**
- **r9 lists no 220/380 kV element out of Siems, Lübeck or Herrenwyk that the model lacks.** The model has T421, Siems–Luebeck blau, Hamburg/N–Luebeck rot and gelb, and Herrenwyk–Siems 380.
- **Hamburg/N – Luebeck gelb:**
  - r9 change log, v6 2024-08-30: "Out-of-service."
  - The r9 Lines sheet (March 2026) still lists it with DLR 949/1200 A and an empty comment.
  - Inconclusive: the v6 note may have been a temporary state, and nothing in r7–r9 restores it. It stays in service in the model: **needs owner**.

**(c) Can SE4 move to Herrenwyk or Siems? Gate test** on a copy of the frozen model.
- Setup:
  - Model copy: `~/data/grid/v5/step6/model_v5_1_copy.pkl` (sha cc4589c1, unchanged).
  - Only `HVDC_CONVERTERS["SE4"]` is changed, in patched source copies `step6/src_{0,H,S}` loaded via PYTHONPATH (`step6/run.sh`, j=4, nice 15).
  - Gate: `gate.py … prefix=z? ref_prefix=a`, against `acc4589c1`; reports are in `step6/gates/`.
  - Only three cases have SE4 ≠ 0: baltic_import_low_load −617 MW (import), winter_peak −217 MW, high_wind +203 MW.

| run | SE4 node | pf hash | losses sy / pp (Σ18) | v_ehv_min | overloaded rows | max loading | lines Δ>0.10 |
|---|---|---|---|---|---|---|---|
| z0 (control) | lubeck\|220 | **identical** to acc4589c1 | 23098 / 23097 | 0.918 | 10 | 1.49 | 0 |
| zH | osm/275466323\|380 (Herrenwyk) | different | 23111 / 23109 (+13 / +12) | 0.918 | 10 | **2.10** | 16 rows (8 per solver) |
| zS | siems\|380 | different | 23111 / 23109 (+13 / +12) | 0.918 | 10 | **2.10** | 12 rows (6 per solver) |

- All three runs PASS the hard gates (18/18 converged, islands, NaN buses, machines, store/CIM sync).
- **Siems – Luebeck blau** in the baltic_import case goes from 0.343 to **2.098**, both solvers. In winter_peak it goes from 0.115 to 0.677, and in high_wind (export) from 1.021 to 0.464. For zH, Herrenwyk–Siems 380 goes from 0.003 to 0.329.
- The `overloaded rows` count does not change (10). The jump to 2.10 shows up only in `maxl`.

**Conclusion.**
- The SE4 injection **cannot move** to Herrenwyk or Siems in v5.1. The only 220/380 kV path out of Siems in both JAO r9 and the model is T421 → the 950 A blau cable (≈362 MVA). A 617 MW import then loads that cable to 210 %.
- The missing capacity is not a missing JAO element; r9 has none. It is most likely the 110 kV network (TenneT's OSM station "Umspannwerk Kraftwerk Herrenwyk" is tagged 380;110 kV, which conflicts with Wikipedia's "no 380/110 kV transformer" at Herrenwyk) plus the Siems SVC. That network is below the model's voltage scope, so **no v5.1 model proposal**.
- Side note: the published SE4 flow in baltic_import (617 MW) is above the 600 MW Baltic Cable rating. Worth checking whether SE4–DE flows include losses or other paths.

**Proposed KNOWN_LIMITS entry** (not applied):

| ID | Item | Why it is open | Effect |
|---|---|---|---|
| HVDC-SE4 | Baltic Cable injection at lubeck\|220 (stand-in), not at the Herrenwyk converter | Real path: Herrenwyk 380 kV → 3 km → Siems 380 → T421 (631 A, JAO r9) → 220 kV cable Siems–Lübeck blau (950 A) plus the 110 kV network. JAO r9 has no further 220/380 kV element; the 380 kV Ostküstenleitung Siems/Göhl is under construction (in service end 2027, Land SH 04.11.2025). With the injection at Herrenwyk or Siems, blau loads to 2.10 at the −617 MW case (step6 gate zH/zS vs acc4589c1); the 110 kV relief is out of scope | Siems–Lübeck blau and T421 are under-loaded on SE4 import (blau 0.34 instead of a physically larger value) and over-loaded on export (1.02 in high_wind instead of 0.46). Revisit when the Ostküstenleitung enters JAO |

### C12: Bentwisch station box; station-table match for split yards (display only)

**What the owner saw.** At Bentwisch (50Hertz, 220/380 kV) there was no box at the yard. The 380 kV busbar and bay ways were drawn as a red comb, 543/544 ended outside the yard, a thin "546" line left to the west, and the Kontek HVDC ended about 1 km from the box.

**Cause (rule bug).**
- The station table (`frozen_build/stations`) gives Bentwisch a MultiPolygon of two OSM polygons:
  - 32063539 "Bentwisch HGÜ": 380 kV, 50Hertz, transmission. This is the 380 kV yard where OSM 543/544 end, at 54.098–54.102.
  - 1464801125 "Bentwisch": 220 kV. This is the 220 kV yard where 317/318 and 275 end, at 54.111–54.113.
- The station's display point (12.2153, 54.1062) lies in the field between the two yards. It is 524 m from the nearest part.
- `station_footprints` accepted a station-table polygon only within `FOOTPRINT_MATCH_M` (300 m) of the display point. So the polygon was dropped and a fixed box was used.
- No circuit end reached that fixed box, so `_fallback_boxes` built a "box at circuit ends" at the 220 kV ends. The 380 kV yard ended up outside every footprint. Because of that, the yard drop rule kept its short ways (the comb), and 543/544 got a 1.1 km stub north to the box.

**Fix.** `src/tranpy/grid_stations.py` (`station_footprints`, station-table branch): a station-table polygon of the station's own site now also matches when the display point lies inside the polygon's convex hull. This is the case of separate yards around a centroid. The existing rule then merges the yards into one convex hull, and the `FOOTPRINT_MAX_M` = 2000 m limit still applies; Bentwisch's hull is 1.7 km. No override table entry. New test: `test_station_table_two_yards_around_the_point`. It fails on the old code and passes on the new.

**Evidence (independent).**
- OSM polygons 32063539 (380 kV, operator 50Hertz) and 1464801125 (220 kV) are both in the station table under `bentwisch`. Their bus nodes are bentwisch|380 and bentwisch|220.
- JAO r9 lists Bentwisch – Güstrow 543/544 (380 kV, 11TD8L543/544) and 275 / Lüdershagen – Bentwisch 318 (220 kV). OSM's 543/544 ways end in 32063539, and the 220 kV ways end at 1464801125.
- The converter polygons 940919943 and 1265537498 ("Bentwisch HGÜ", Kontek and the Kriegers Flak CGS back-to-back) sit beside the 380 kV yard. This matches the Kontek landing.
- Imagery was not checked separately. The yard outlines are the OSM polygons.

**Build.**
- `halo/c12/export_c12.sh` is `export_c8.sh` plus `hvdc_display.py`, with the edge review written to `halo/c12`.
- Rebuilding with the old code (`halo/c12/code_before`) reproduces `web_v5_1_c10` exactly: every layer, `station_details` and all 541 SLDs are identical.
- New web dir: `web_v5_1_c12`. Staging `app/data` now points to `../web_v5_1_c12`. Rollback: `ln -sfn ../web_v5_1_c10 app/data`.

**Layer diff c10 → c12** (keyed, `halo/c12/kdiff_c10_c12.txt`). Only Bentwisch changed:
- `station_footprints`: 1 row (bentwisch, "box at circuit ends" → "station table").
- `substations`: 1 row (bentwisch footprint).
- `station_details`: bentwisch only.
- `lines`: 5 circuits, all at Bentwisch:
  - 543 and 544 each lose the north stub (5 → 4 pieces) and end at the south edge of the box.
  - 275, 318 and 317-276 now end at the north edge of the box.
- `osm_lines`: 4411 → 4393. 20 short yard ways (0.03–0.22 km, inside the yard) drop out. Two 220 kV yard-entry ways (39605059, 39605061 "317") are now drawn as matched.
- Every other layer, and all SLDs, are identical.

**Acceptance** (`halo/c12/acc_web_v5_1_c10` vs `acc_web_v5_1_c12`):
- All hard checks give the same result as before, including `sld_complete` ok and `halo_popup_consistent` ok. The 3 failures that were already there (`drawn_once`, `no_gap_over_limit`, `no_loop`) are unchanged.
- `events.csv` is identical.
- Stats: osm_lines 4411 → 4393, straight_dashed_pieces 29 → 27, station_passes 113 → 110.
- Tests: pytest `test_grid_gis` + `test_grid_network` + `test_map_acceptance` 58 passed (`test_grid_stations` does not exist). ruff check and format are clean.

**PNGs** (`halo/c12/shots`, before = 8094 on c10, after = preview on c12):
- Bentwisch: the box covers both yards, the comb is gone, and 543/544 end at the box.
- Controls Lübeck, Philippsburg, Würgassen and Stade/W are pixel-identical before and after.

**546 verdict: no model change.**
- OSM way 59581816 (ref 546, 380 kV, 50Hertz, 7.22 km) runs from the 380 kV yard to 93 m from OSM power_plant 26389159, Kraftwerk Rostock (hard coal, 514 MW, EIC 11WD8ROST1C---P5).
- It is the power plant's 380 kV connection line. JAO r9 has no Bentwisch line other than 275/318/543/544; the changelog shows only the Kontek DC tie, added and then removed.
- The model attaches Kraftwerk Rostock (MaStR SEE943427876363, Höchstspannung) directly at bentwisch|380, with plant_connections at 6.8 km.
- So it is not a missing circuit, not a construction project, and not HVDC.
- Its short 546 piece inside the yard (59581823) dropped out with C12. The 7.22 km way stays as "no model circuit on this tower line". A popup way note ("Kraftwerk Rostock 380 kV connection; model: plant at bentwisch|380") is proposed and not applied.

**Remaining at Bentwisch (needs owner):**
- At the north edge, the 318 and 317 pieces on the two single-circuit entry ways cross ("<" shape at z15). This is cosmetic, and no_zigzag passes.
- The box is the hull of both yards, so it includes the field between them.

**Scan** (`halo/c12/scan_c10.csv`): stations with a fixed or "box at circuit ends" footprint and an OSM polygon of 220 kV or more within 1 km. Footprints in c10: station table 388, box 42, box at circuit ends 20. Matches: 29 in c10, 28 in c12; only Bentwisch is fixed by C12. Each remaining case is needs owner, and none changes in the C12 diff:

| Cause | Stations |
|---|---|
| Station-table polygon 300–460 m from the display point (single polygon, point outside it) → fixed box | fedderwarden (313 m), lubeck (337), philippsburg (300), rdk7 (457), rdk8 (386), wilster/west (322) |
| No station-table polygon (OSM polygon at 0–945 m) | lippborg (945), ried (799), wurgassen (point inside OSM polygon, table has none) |
| Table polygon accepted, but each circuit's end nearest the display point misses it by more than `END_TOUCH_M` (15 m) → box at circuit ends. In most cases another piece end does touch the polygon, which points to stub pieces | alfstedt, aschaffenburg, audorf, bandstahl, brokdorf, conneforde/ost, dorpen/west, eiserfeld, emden/ost, erzhausen, gkn2, hoheneck, krummel, mehringen, merzen, remptendorf - grossschwabhausen, stade/west, vieselbach, vohringen |

### C13: border terminals; no foreign grid drawn (display only)

**What the owner wants.** Lines leaving Germany end at a clear border terminal, and no foreign geometry is drawn. Each terminal reads as import/export.

**Change** (`halo/c13/border_terminals.py`, applied to a copy of `web_v5_1_c12`; build `halo/c13/export_c13.sh`). The model is not read or changed.
- Border: the OSM boundary relation 51477 of Germany (Overpass 2026-11-10, cached in `halo/c13/osm_de_boundary.json`; polygon `halo/c13/de_border.geojson`). Natural Earth 1:50m (the bidding-zone layer) is 1–3 km off at the borders and was not used for the cut.
- AC ties: 57 in-service tie circuits (one end at a foreign site of `neighbours.geojson`). 48 cross the border exactly once and are cut at the crossing (29 pieces abroad dropped, the label piece moved to the drawn part). 9 ties run between the model's Vorarlberg/Tyrol stations (Bürs, Silz, Kühtai and tees) and a foreign node; they never cross the border and keep their geometry.
- Terminals (`border_terminals.geojson`, 25 AC + 4 HVDC): one symbol per crossing. Ties of one zone whose crossings lie within 2 km share one terminal. Label e.g. "→ AT · St. Peter · 4 circuits"; below zoom 8 only the zone code. The icon is a double arrow across a dashed border mark, turned along the line (outward). The bold head shows the typical measured direction: orange export, blue import, dark both directions, grey no data.
- Popup: the tie circuits (links highlight them), the measured flow per TSO segment from `data/entsoe_tso_border_flows_v5.csv` (ENTSO-E A11, + = out of Germany): typical direction, hours of export, median, min and max MW. Then "Model: measured ENTSO-E flow per TSO + equivalent machine (not drawn beyond the border)". The flow is per TSO × zone segment, so it covers all ties of that TSO to that zone, not one terminal alone. DK1, DK2, NO2 and SE4 have no per-TSO data and show grey.
- HVDC (NordLink → NO2, Baltic Cable → SE4, Kontek → DK2, ALEGrO → BE): the C10 sea/foreign routes are removed. The two model links keep their feature, with the converter point as geometry. The two C10 display-only features are dropped. One terminal at the German converter (Baltic Cable at Herrenwyk; the popup notes that the model injects at Lübeck 220 kV) with operator, MW, kV DC, year and source.
- `osm_lines`: 125 ways wholly abroad removed and 32 clipped at the border. 47 ways abroad that carry the Vorarlberg/Tyrol model circuits are kept. No clipped way became a short-halo way.
- `station_footprints`: the boxes of 24 foreign sites are removed. Walgau, Werben, Westtirol and Meiningen keep theirs, because the kept Vorarlberg/Tyrol lines end there. `substations`: property `foreign_node` added; the app hides those symbols and labels. The `neighbours` symbol layer is replaced by the terminals when the file exists.
- App (`apps/grid_map/app.js`, 8 new symbols `border_*.svg`, `hvdc_terminal_*.svg`): terminals layer, popup and legend. Search on an HVDC link (point geometry) fits to the converter.
- `scripts/map_acceptance.py`: new hard check `border_terminals`. A tie cut at the border is checked at its own border port in place of the foreign station (ends, station_touch). Every port must lie within 2 km of its terminal, and every tie listed as cut in the metadata (`border_cut_ties`) needs a terminal. New test `test_border_cut_tie_ends_on_its_port`: 12 passed.

**Layer diff c12 → c13** (`halo/c13/layerdiff_c12_c13.txt`):
- `lines`: 61 circuits (57 ties + 4 HVDC). All other circuits are byte-identical.
- `osm_lines`: 4393 → 4268.
- `station_footprints`: 24 foreign boxes removed.
- `substations`: only `foreign_node` was added.
- New `border_terminals.geojson` (38 KB). `metadata` has a new build_id plus `border_terminals` and `border_cut_ties`.
- Byte-identical: buses, plants, plant_aggregates, plant_connections, transformers, tower_lines (raw OSM), neighbours, bidding_zones, control_areas, station_details and all SLDs.

**Acceptance** (`halo/c13/acc_web_v5_1_c12` vs `acc_web_v5_1_c13`, same script):
- Every hard check gives the same result, and every failure list is identical. The 3 failures that were already there (`drawn_once`, `no_gap_over_limit`, `no_loop`) are unchanged. `sld_complete` and `halo_popup_consistent` are ok, and the new `border_terminals` is ok (48 ports).
- Listed only: `ratio_outside` 52 → 78 (cut ties are shorter than the model length, as expected). `fragmented_junctions` 18 → 17 (y-kuhmoos#bl).

**PNGs**: `halo/c13/shots` (before = 8094 on c12, after = preview 8097 on c13), plus `shots_t` after the larger icon.

**Final review on c13:**
- Panel (`docs/map_panel.md`, 63 places): `halo/c13/panel_before` and `panel_after`. 8 differ visibly, all as intended: Bentwisch P05/P06 (Kontek terminal), Westtirol P18, y-kuhmoos P32 (Laufenburg terminal, foreign box gone), Endrup P42, Simbach P45/P50 (St. Peter terminal) and Brunsbüttel P48 (C10 NordLink route gone). 7 others differ by 1 pixel.
- `scripts/map_visual_check.py` (`halo/c13/visual_c13`): 541 places, 3 flagged (Y-Dahlem ×2 and Fedderwarden, all the existing `no_loop`).
- Staging `app/data` → `../web_v5_1_c13`; app.js, index.html (cache stamp) and the symbols were updated. Rollback is in `halo/c13/ROLLBACK.txt`.

**Needs owner:**
- The bidding-zone outline (Natural Earth 1:50m) lies 1–3 km from the OSM border, so a terminal can sit off the drawn zone line (e.g. Krajnik, Laufenburg). Proposal: draw the DE border from the OSM relation.
- Vorarlberg/Tyrol: the model's stations abroad (Bürs, Silz, Kühtai and tees) and their lines stay drawn; their ties end at terminals at Walgau, Werben, Westtirol and Meiningen.
- Doetinchem: the two ties cross 1.6 km apart and share one terminal; one line ends 1.6 km from the symbol. Kassø has two terminals (Handewitt and Jardelund corridors). Hradec has two (Röhrsdorf, and Etzenricht shared with Prestice).

### C14a: Renewables (inverter) layer and offshore wind farms (display only)

Build `web_v5_1_c14a` (copy of `web_v5_1_c13` plus three layers); scripts and logs in `halo/c14a/` (`build_c14a.py`, `build.log`, `acc_c13`, `acc_c14a`, `layerdiff_c13_c14a.txt`, `shots/`). Staging stays on c13 and the c14a preview runs on 127.0.0.1:8097. The model is not changed.

- **renewable_inverters** (422 buses): for each bus, the capacity (`max_p_mw`) and units of solar, onshore wind, offshore wind and storage, taken from `cim.synchronous_machine` of `model_v5_1_frozen.pkl` (sha `cc4589c1…`). The fuels are the same as `grid_case._INVERTER`, and the main component is the same as `build_grid_case`. All 422 buses are in the case and each has one REGCA1. Totals: solar 115,423 MW, onshore wind 71,144 MW, offshore wind 11,434 MW, storage 20,578 MW, dispatched 37,461 MW. The offshore MW per landing bus matches the model exactly (Büttel 2417.5, Dörpen/West 2823.0, Emden/Ost 2142.2, Diele 1922.8, Lubmin 1802.9, Bentwisch 325.3).
- **offshore_farms** (34) and **offshore_links** (26, schematic dashed lines from a farm to the landing bus the model uses). The farm list, position, MW, turbines and status come from `src/tranpy/resources/grid/offshore_wind_farms_de.csv`. Each farm's register connection point is checked against the model's MaStR register cells: for every connection point, the turbines of its farms add up exactly to the register units (e.g. DWAL 150 = Borkum Riffgrund 1 78 + Trianel 1 40 + Trianel 2 32). Cables and landings come from the model's own `offshore_connections_de.csv` (with its evidence) and, for display only, from DolWin5.
- Style: proportional circles (area ∝ MW) at the substation, coloured by the largest type (ENTSO-E/energy-charts fuel colours). Offshore farms use a wind-turbine symbol (OpenInfraMap) sized by MW. The layer is off by default in "Our model". `app.js` has the layer, popups and legend; `symbols/wind_offshore.svg` is new.

**Sources:** MaStR via the frozen model (register cells, units, connection points); Wikipedia (de) "Liste der deutschen Offshore-Windparks", retrieved 2026-11-10 (farm MW, turbines, coordinates, status); NEP Strom 2037/2045 (2025), 2nd draft, ch. 5.3.1 p. 143 (DolWin5 and BorWin5 in operation in 2025; DolWin6 in operation); offshorewind.biz 2025-10-17 (DolWin5: Borkum Riffgrund 3 → Emden/Ost); 50Hertz Ostwind 3 leaflet 2024 (Windanker → Ostwind 3, new substation at Brünzow, not Lubmin); `offshore_connections_de.csv` evidence (TenneT/50Hertz project pages and Wikipedia).

**Tests:** `map_acceptance` on c13 and c14a gives identical results (same `acceptance.json`, byte-identical `events.csv`; the three existing fails are unchanged: drawn_once 7, gaps 19, no_loop). The layer diff shows only the three new GeoJSON files and two new keys in `metadata.json` (build_id and renewables). With the layer off, the before and after PNGs differ only in the panel (the new checkbox row): the map pixels are identical.

**Needs owner:**
- The model spreads 2,763.5 MW of offshore wind equally over the landing buses of its sea, because the register connection point is not in `offshore_connections_de.csv` or its node is not a model bus. Affected: DolWin5/Borkum Riffgrund 3 (958.7 MW, which should land at Emden/Ost), BWEP/He Dreiht (870 MW, BorWin5 → Garrel/Ost, not a model bus), Windanker (300 MW, Ostwind 3 → Brünzow), Riffgat, Nordergründe, alpha ventus, EnBW Baltic 1 and 2. This is a model attachment question; the map only shows it.
- No independent source was checked for the export cable or landing of Riffgat (EMDB), Nordergründe (INHA), alpha ventus (HAMA), EnBW Baltic 1 and EnBW Baltic 2. They are left blank and marked "not sourced".
- The farm list is from Wikipedia. BSH/MaStR farm records were not checked one by one; only the unit counts agree with MaStR.
- Storage: 20.6 GW across 415 machines, with p_mw = 0 in the case, as the model holds it.

## Reverted / not applied (needs owner)

The first pass reclassified these ways from "explained" to routing halos. That pass is **reverted**, because no independent source shows the named circuit on that particular way. They keep the earlier explained state.

| Way | Candidate circuit (model status) | Why inconclusive |
|---|---|---|
| 127187723 "380kV Hamaland + Münsterland" (4306) | Gronau - Niederrhein PRHAMSBW (no_terminal) | Only OSM's name and issue-09 note it; no JAO/TSO geometry |
| 61973549 (2644) | GRAFSH W tee (no_terminal) | Issue 09 puts it on 4305/4306, not 2644 |
| 160561540 (4502) | Weissenthurm - Y-Koblenz SOONWD O (no_terminal) | Only nearness of the terminal |
| 50306784, 376136848 (Laufenburg) | Kuehmoos - Laufenburg ties to CH (no_terminal) | Only nearness of the terminal |

Simbach: see C6 (256/258 applied) and the needs-owner items there (tee, 233/230 and 234/230).

## Notes

- **N1:** ways 908175880 (380 kV) and 1095719612 (220;380 kV) are two OSM ways on the same two nodes, both 326.5 m. Meckenheim - Weissenthurm NETTE W picks either one depending on the tie order, and the drawn piece is identical. The final build equals the baseline here.
- The export (from C8 on) adds `--way-notes halo/network/way_notes.csv`; script `halo/c8c9/export_c8.sh`.
- The network is rebuilt with `--overrides halo/network/route_overrides.csv --join-overrides halo/network/join_overrides.csv --overpass ../osm/overpass_at_west.json halo/network/overpass_stpeter_min.json`.
- Backups:
  - `web_frozen_before_explfix`: the baseline.
  - `web_frozen_explfix_v1`: the first, unverified version.
  - `web_frozen_explfix_v2` (the build before C6).
  - `halo/network_before_explfix`, `halo/network_explfix_v1` and `halo/network_explfix_v2`.

## C14b: offshore connection points sourced; station panel shows renewables (display only)

Build `web_v5_1_c14b` (copy of c14a; script `halo/c14b/build_c14b.py`, app patch
`halo/c14b/patch_app.py`, test server 127.0.0.1:8098). 8094 not repointed; the model is unchanged.

Scan of all 34 farms: the 8 farms with `spread equally` attachment are the only ones without a
link or with an unsourced cable; the other 26 link to their sourced curated landing, all model
buses. Spread-rule inverter buses: diele, dorpen/west, emden/ost, buttel (528.2 MW each) and
lubmin, bentwisch (325.4 MW each).

- `offshore_farms`: cable, landing, sources, `landing_model_bus`, `upstream`, and the note
  "model v5.1 spreads this capacity over the landing substations of its sea; v6 to fix" for the 8 farms
  (table in `docs/v6_plan.md`, "Offshore connection points (from C14b)").
- `offshore_links`: 8 dashed links added (26 -> 34) to the sourced landing (model bus, or
  the OSM position of Garrel/Ost, Stilow, Hagermarsch, Emden/Borßum).
- `renewable_inverters`: on the 6 spread buses, `offshore_spread_mw` / `offshore_spread_note`;
  `offshore_farms_landing` per bus. MW values are unchanged.
- Station panel: new section "Renewables (inverter)" after the plant list, per model bus
  (solar, onshore, offshore, storage MW / units, total, dispatched), farms landing there and the
  spread note. "Plants in operation" was not renamed because it also lists batteries (72 storage
  plants), so it is not purely synchronous. A note explains the split instead.
- Check: the station sums equal the layer total (218,578.2 MW, 422 buses, each in exactly one
  station). Panels checked at Siedenbrünzow, Lubmin, Bentwisch, Büttel, Dörpen/West + 10 random.
- Needs owner: the upstream model bus for He Dreiht (Garrel/Ost), Windanker (Stilow) and alpha ventus
  (Hagermarsch 110 kV); whether the Riffgat re-feed to Emden/Ost was completed.
- Acceptance: `map_acceptance.py --network ../network` was run with the same command on c14a and c14b
  (`halo/c14b/acc_same_c14a`, `acc_same_c14b`). The results are identical, and also identical to the
  original `acc_c14a`, with the same 3 known FAILs (drawn_once, no_gap_over_limit, no_loop).
  Layer diff (`halo/c14b/layerdiff_c14a_c14b.txt`): only offshore_farms (8 features),
  offshore_links (+8, a `sourced_in` field on the 26 existing), renewable_inverters (7, properties
  only, no MW or geometry change), metadata and app.js/index.html changed.
- Panel test (`halo/c14b/panel_sum_check.txt`): 14 of 15 panels match the expected per-station sum
  within 1 MW (rounding). The click for "halfeshof" opened the co-located station osm/1463882574
  (30 m away, 0 MW, correctly "none"). This is a click target, not a data error.
- PNGs: `halo/c14b/shots/{farm_*,station_*}_{before,after}.png`.

## C16: routing misses from the C15 triage (issue 15, cause 2)

Build `web_v5_1_c16` on cairo-lab: a copy of `web_v5_1_c14b` (C13 terminals and C14a/b layers kept). Scripts and outputs are in `halo/c16/`. The model is unchanged, nothing under `~/data/grid/v5` was touched, 8094 was not repointed, and nothing is committed. Test servers: 127.0.0.1:8115 (c14b, before) and 127.0.0.1:8116 (c16, after).

Method: each fix is a single row in `halo/c16/join_c16_K.csv`, built cumulatively as networks n1 to n4. Baseline n0 is rebuilt from `halo/network` and is byte-identical to it (routes, pieces, edge_check, terminals). For each step:
- A raw export `EK` is made with `halo/c16/export.sh` (export_c12 recipe).
- `halo/c16/patch.py E(K-1) EK` carries only the changed groups (lines by mrid, osm_lines by way_id, station_footprints by site) onto c16. It aborts if any other layer changed, or if a changed group had been touched by C13 or C14.
- The layer diff is `ldiff_K.txt`, the acceptance run is `acc_K` (diff in `accdiff_K.txt`), and the PNGs are `shots/C16-K_*_{z9,z13}_{before,after}.png`.
- The N1 tie-order artefact (NETTE W on 908175880/1095719612, identical drawn piece) is kept as in c14b (`n1_skip.txt`).
- The 3 known acceptance FAILs are unchanged in every step, and events.csv is identical.
- Join pieces appear in the circuit layer and not in the tower-line view, as with the Elsfleth join.

### C16-1 Etzenricht - Mechlenreuth 437/479 (F005, F027)
- Join 1080503905-13379189318 (115 m). The old Ostbayernring way 116649309 "Ltg 437 Etzenricht - Mechlenreuth" ends short of the Mechlenreuth 380 kV yard. Esri imagery shows the line entering the yard on its south side.
- Sources: JAO r9 Lines r2243 (89.8 km), against an OSM path of 87.2 km. ROPf, 3rd Planänderung OBR B-Süd (2025): the new line is still under construction, and the old line is to be dismantled only after commissioning.
- Result: no_path_strict becomes routed. 7 ways go from no_model to model_fewer (OSM also lists the 220 kV system 248, which is not in the model). The Etzenricht box grows.
- Acceptance: osm_lines_halo goes from 75 to 82 (these are the model_fewer ways); straight circuits go from 120 to 119.

### C16-2 Röhrsdorf - Weida 573/574 (F006)
- Join 13188220521-13188220520 (157 m). The span from the branching tower to the Weida branch is missing in OSM. Esri imagery shows the span.
- Source: JAO r9 Lines r105/r106 (65.96 km), against an OSM path of 65.6 km.
- Result: 3 ways go from no_model to match, and 143891601 goes from model_fewer to match.
- Acceptance: straight circuits go from 119 to 117; halo goes from 82 to 81.

### C16-3 Ludersheim - Schwandorf 223 (F007)
- Join 318903646-653922514 (107 m). Way 37762758 (ref 223) ends at the west edge of the Schwandorf yard (imagery).
- Source: JAO r9 Lines r2310 (63.03 km), against an OSM path of 60.9 km.
- Result: 2 ways go from no_model to no_tag.
- Acceptance: straight circuits go from 117 to 116.

### C16-4 Bischofsheim - Urberach ERZHAU N (F036)
- Join 11252330497-10576520264 (156 m). Way 24686637 (Anlage 4134, 2 circuits) stops south of the road at the Bischofsheim yard (imagery).
- Sources: JAO r9 ERZHAU N (47.29 km), against an OSM path of 46.2 km. The second circuit on the way, Y-Bischofsheim RIED W, is routed as well (25.2 km against 25.69 km).
- Buerstadt - Y-Buerstadt NASSAU would have transited the Bischofsheim yard via the Trafo 411 feeder, and no source supports that. It is held at no_path_strict by `route_c16_4.csv`, so only its route_note text changed.
- Side effect: 507761116 (a 0.2 km tee approach at Pfungstadt) goes from explained to model_more, because both RIED W legs meet there.
- Acceptance: straight circuits go from 116 to 114; halo goes from 81 to 80.

### Needs owner (not applied)
| Finding(s) | Circuit | Why |
|---|---|---|
| F008, F089 | Preilack/Altdöbern - Streumen 560/563 | 4.8 km OSM gap between 154787322 and 1376151495 (no ways in the cache; Overpass returned nothing). 560 is also held by the coordinator's override. |
| F009 | FWH-S - Urberach SHWANH O/W | The OSM path is 30.7 km against 66 km in JAO; the corridor is unclear. |
| F051, F058, F065, F087 | Wolmirstedt - Förderstedt 437/438 | Several km-scale gaps along the SuedOstLink co-routing. |
| F002, F004, F012, F028 and the ties F011, F016, F044, F047, F068, F073, F081, F084, F145 | Leupolz/ALLG O, Lehrte-Erzhausen 1/3, Würgassen Y-circuits, Klixbüll-Endrup, Handewitt-Kassø, Diele-Meeden, Ensdorf-Vigy, Hagenwerder-Mikułowa, Doetinchem-Niederrhein, Röhrsdorf/Etzenricht-Hradec/Prestice | `no_terminal`: the model bus or tee has no OSM node (rounded position, a foreign station outside the DE graph, or a tee off the line). The fix needs a terminal-position override mechanism, a code change in grid_network or the station data. Keep the C13 border cut for the ties. |
| F020, F112, F116, F119, F137 | Y-Altenkleusheim BIGGE O/W | OSM gaps of 1.6 to 2.3 km. |
| F019, F132, F074, F111, F136, F077, F141 and the rest | KONDLW, SIEGRL, Simbach, etc. | Gaps of 3 to 10 km, or previously held by override. |

no_model km (osm_lines, 220/380 kV): 2596.6 in c14b, 2364.8 in c16 (-231.8). Fixed: 5 findings (F005, F006, F007, F027, F036; 229.0 km) through 4 joins, covering 6 circuits.
