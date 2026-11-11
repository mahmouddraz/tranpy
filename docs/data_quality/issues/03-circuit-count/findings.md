# 03 — Circuit count per tower line vs model: findings

Status: **Detect / Investigate / Propose done — waiting for the owner's decision.** I changed nothing in `model_v4final` or `grid_model.gpkg`.

Inputs: `cairo-lab:~/data/grid/tower_circuit_mismatch.csv` (= the `tower_lines` layer, 10,963 ways). Also the `lines` layer of `grid_model.gpkg` and `src/tranpy/grid_towers.py` (assignment: sample points every 25 m within 60 m of a way; lines with `route_quality` straight/in_station are skipped). For JAO I used the Core SGM 9th release (the model source, status March 2026) and its Change Log. I also used the Core SGM **1st release** (JAO, published 2022-09, files `Core _Static Grid Model_template_D2/D7/D8.xlsx`; BE file labelled Q4-2021; sha256 of the zip `69f4f598…5b06`). In that release the TransnetBW (D4) file is empty.

Occurrences: `occurrences.csv`, 1,453 rows. That is every `no_model_circuit` way longer than 2 km (230), every `model_fewer` way (178) and every `model_more` way (1,045). Each row has a `class`, an `evidence` text and the coordinates of a representative point.

## Key cross-cutting finding

**The model's line list is the JAO 9th release, which is the 2026 grid, not 2021.** The 1st release (closest to 2021) has 820 German line/tie-line entries. **152** of their EICs are missing from the model. A name/endpoint match explains 58 of them as renamed or EIC-changed. That leaves **94 entries (47 × 220 kV, 47 × 380 kV)** with no equivalent in the model. Many are restructured tees (Emden, Conneforde, Alfstedt, Voerden, Hardegsen). Others are real circuits that were in service in 2021 and were decommissioned or taken out of the SGM later. The JAO change log gives the reasons ("Removed from SGM. Line is out-of-service", "Existing line was removed from the model", "Removed: decommissioned"). The reverse problem also exists: the model holds 2023–2026 circuits, for example Isar–Ottenhofen 443 and Isar–Pleinting 451, which JAO marked out of service in the 6th release (2024-08). This overlaps class 01 (future assets). It is the main root cause of the real `model_fewer` cases below.

## Counts per class

| mismatch | class | ways | km |
|---|---|---:|---:|
| no_model_circuit (>2 km) | display_artefact_straight_model_line | 34 | 373 |
| | display_artefact_route_offset | 33 | 265 |
| | abroad (CH/AT/CZ/NL/DK) | 27 | 604 |
| | offshore_export_cable (Ostwind 1/2) | 9 | 226 |
| | under_construction_tag (OSM) | 37 | 397 |
| | post_2021_or_future (verified via JAO change log / project) | 13 | 154 |
| | topology_change_post_2021 (Bergshausen/Würgassen/Voerden) | 6 | 33 |
| | plant_or_industrial_lead | 24 | 132 |
| | mixed_110kV_untagged | 12 | 81 |
| | cable_section_unmatched | 14 | 46 |
| | missing_2021_circuit_jao_r1 | 2 | 30 |
| | unresolved | 19 | 191 |
| model_fewer | missing_2021_circuit_jao_r1 | 12 | 381 |
| | display_artefact_route_offset | 77 | 116 |
| | display_artefact_straight_model_line | 15 | 22 |
| | mixed_220_380_shared_towers | 35 | 270 |
| | osm_tag_error (incl. `437;___` empty slot) | 5 | 83 |
| | post_2021_or_future (A-Nord HVDC tagged 380 kV) | 1 | 305 |
| | unresolved | 33 | 267 |
| model_more | display_artefact_short_way (<0.5 km, station entries) | 854 | 106 |
| | display_artefact_parallel_way (another 220/380 way within 80 m) | 75 | 663 |
| | display_artefact_other_voltage | 45 | 365 |
| | display_artefact_routes_collapsed | 62 | 1,447 |
| | osm_tag_error (circuits < cables/3) | 9 | 175 |

## (a) no_model_circuit > 2 km — top items

| physical line | ways / km | class | evidence |
|---|---|---|---|
| 50Hertz 321/328 Parchim Süd–Güstrow | 1 / 51.9 | display artefact | in the model as `osm-5006-0/1` (export section, library parameters), drawn as a straight chord, so `split_by_towers` skips it. JAO r1 has 328 (11TD8L328------I, X = 57.559 Ω, 52.3 km); 9th-release log 2026-03: "Parchim/Süd – Güstrow 328 Removed Element". Note for class 04: the model's library X is 14.44 Ω per circuit vs JAO r1 57.56 Ω |
| 50Hertz 231/232 Weida–Herlasgrün | 1 / 32.1 | display artefact | `osm-4828-0/1`, straight |
| 50Hertz 319/324 Wustermark–Brandenburg West | 1 / 28.9 | display artefact | `osm-4311-0/1`, straight |
| 50Hertz 577/578 Röhrsdorf–Markersbach/Zwönitz | 3 / 47.9 | display artefact | `osm-14312-0/1` + `curated/Zwoenitz - Roehrsdorf 577/578`, straight |
| Amprion 4511 (Sechtem–Brauweiler) | 1 / 23.7 | display artefact | `osm-14135-0/1`, straight |
| also 225/226 Röhrsdorf–Niederwiesa, 205/206 Eula–Taucha, 383–386 Remptendorf, OH-FO, Schwarze Pumpe–Graustein, Anilin, Hambach, Röhe, … | 34 / 373 total | display artefact | all 61 `osm-*`/`curated/*` model circuits have `route_quality = straight` |
| TenneT Stade–Landesbergen (LH-14-3111) | 1 / 37.2 | future | not in any JAO release incl. the 9th; TenneT new build under construction — not 2021 |
| Wahle–Mecklar (LH-11-3040) | 6 / 27 | post-2021 | JAO 3rd/4th release (2023) adds Lamspringe–Wahle 3/4 and Hardegsen tees |
| Ganderkesee–St. Hülfe (LH-14-325), Elsfleth/W–Ganderkesee (LH-14-304) | 3 / 37.5 | post-2021 | JAO 4th release 2023-09 "New German internal tieline" / re-added |
| Klixbüll–DK border (LH-13-322) | 1 / 14.2 | post-2021 | JAO 9th release 2026-03 "Klixbuell/S – Endrup 1/2 New line with DK" |
| Ostwind 1 (261/281/282) and Ostwind 2 cables | 9 / 226 | offshore export cable | radial wind-farm connections to Lubmin; not lines of the SGM (Ostwind 2 is 2023) |
| Swissgrid / APG / TIWAG / Illwerke (Westtirol–Zell, Pradella, Kops, Lünersee…) | 27 / 604 | abroad | >50 % of the way outside the DE polygon |
| Isar/Pleinting, Meppen–Hanekenfähr, Mehringen–Dollern, Pulgar–Vieselbach, Ultranet, Ostbayernring, … | 37 / 397 | under construction (OSM tag) | verified against JAO where possible: Hanekenfähr–Meppen BACCUM (7th release 2025-03), Pulgar–Vieselbach new-build route; none is a 2021 asset |
| Jänschwalde–Preilack MA–MF, Boxberg–Bärwalde, Neurath/Frimmersdorf/Niederaußem, Scholven, Bergkamen, Weisweiler, Schkopau–Lauchstädt, Lippendorf–Pulgar | 24 / 132 | plant lead | generator connections; the SGM lists TSO lines between substations; the plants already connect at their substations through `plant_connections` |
| Amprion 220 kV Wengerohr (2409) | 2 / 29.6 | **missing 2021 circuit** | JAO r1 "Niederstedem – Wengerrohr WENGER" 11T0-0000-0394-2, 220 kV, R 2.37 / X 11.86 Ω / B 151.3 µS, 40.45 km, DLR 1290–1773 A; removed in the 7th release (2025-03) |
| unresolved (19 ways, 191 km) | | | e.g. TenneT 2053 (Bechterdissen, 220 kV, 32 km); TenneT 1257345901 (380 kV, 30.8 km near Conneforde/O–Elsfleth/W, likely a 7th-release new line); Amprion Urberach–Weinheim (18.4 km); TransnetBW Anlage 322 NWHEI–HOHCK 220 kV (the r1 D4 file is empty, so no check was possible); Stadtwerke München Oberbachern–Menzing (city grid, 380 kV) |

## (b) model_fewer — real missing parallel circuits (2021) vs tag/artefact

Real circuits that the JAO **1st release** lists and the model lacks. All parameters below are published JAO r1 values:

| tower line (OSM) | missing circuit (JAO r1) | EIC | kV | R / X / B | km | removed in |
|---|---|---|---|---|---|---|
| Isar–Ottenhofen 443;444;446 (61 km) | Isar–Ottenhofen 444, 446 | 11TD2L0000001249, …1257 | 380 | 1.41/17.291/256.67; 1.35/16.677/266.09 | 61.3 | 444 removed 8th release (2025-09); both "out-of-service" in the 6th release (Isar 2 shut down 2023-04) |
| (same corridor) | Isar–Pleinting 452 | 11TD2L0000001273 | 380 | 1.43/15.587/287.14 | 63.5 | 8th release |
| Trennfeld–Aschaffenburg 217;218 (43.8 km) | Grosskrotzenburg–Trennfeld 218 | 11TD2L000000104F | 220 | 6.39/28.892/229.34 | 76.3 | — (absent from 3rd release on) |
| Oberbrunn–Oberbachern 269;270 (21.5 km) | Oberbachern–Oberbrunn 269 | 11TD2L000000193R | 220 | 0.84/9.469/115.30 | 30.9 | — |
| Grafenrheinfeld–Raitersaich + Bergrheinfeld 433;434;278 (75.7 km) | Bergrheinfeld–Raitersaich 278 | 11TD2L000000022H | 220 | 2.46/26.292/409.04 | 90.8 | — |
| Leitung 2001 Lehrte–Hardegsen (61 km) | Lehrte–Godenau 1 | 11TD2L000000157V | 220 | 4.04/20.36/137.29 | 48.6 | 8th release; **the model has no Godenau bus** |
| TenneT 206 Conneforde–Cloppenburg (55 km) | Conneforde–Cloppenburg/O rot | 11TD2L0000000463 | 220 | 3.754/21.434/204.14 | 55.8 | replaced by 380 kV rot/ga in the 8th release |
| Ltg 437/248 Etzenricht | Etzenricht–Schwandorf 248 | 11TD2L000000176R | 220 | 2.26/17.208/121.81 | 44.2 | — |
| Otterbach–Beeden 220 kV (39 km) | Y-Mutterstadt / Y-Uchtelfangen (-Homburg) OTTERB S | 11T0-0000-1436-9 / -0505-P | 220 | 5.791/31.39/395.3; 1.763/9.039/116.9 | 106 / 31 | tee restructured (medium confidence) |

Other 1st-release circuits with no model equivalent that have no OSM mismatch flagged here: Meissen–Veltheim A08, Würgassen–Göttingen 3/4, Mecklar–Dipperz 1, Dipperz–Grosskrotzenburg 1, Klein Ilsede–Grohnde 1, Dollern–Götzdorf grün/blau, Preilack–Streumen 559, Siedenbrünzow–Güstrow 512, Altentreptow/Süd–Gransee 479, Lauchstädt–Wolmirstedt 535, Förderstedt–Ragow 531, Bertikow/Vierraden/Pasewalk 303–306, Sechtem–Weissenthurm NETTE O, Koblenz–Limburg NASSAU, Hattingen–Laer, Niederstedem–Quint/Trier, Buerstadt–Hoheneck REINAU O, and more. Several of these were later replaced by new topology, for example Vierraden–Bertikow 304-303 and Klostermansfeld–Wolmirstedt 535. Each needs the same check before anything is added (class 01 overlap).

Not real: `437;___` (empty second slot on Etzenricht–Mechlenreuth, 5 ways) → OSM tag. 35 mixed 220/380 ways → the extra circuit belongs to another line on shared towers (for example the Perleberg 436 / Wolmirstedt 332 shared towers). 92 ways → display offset or straight chord. 33 ways unresolved, among them Wolmirstedt–Parchim 332 (2 tagged, 1 in the model; JAO: 332-322 "Removed: decommissioned" in the 4th release and "New Topology" in the 5th), TenneT 2024 Lehrte–Wahle, Amprion 4564 Westerwald, Trebur Nord 4591.

## (c) model_more sample

- 854 ways are shorter than 0.5 km: station entries and crossings, i.e. display only.
- 120 ways have a parallel way or a circuit of the other voltage within the 60 m radius, so the assignment grabs the neighbour.
- 62 long ways (1,447 km): several model corridors drawn on one OSM geometry. Examples: Bärwalde–Schmölln + Bärwalde–Streumen on one 2-circuit way (93 km); Helmstedt–Wolmirstedt 491/492 + Klostermansfeld–Wolmirstedt on Ltg 536 (117 km); Ragow–Streumen 561/562 + Altdöbern/Preilack–Streumen (88 km). The routes were copied from a shared OSM relation, so this is a geometry artefact, not a topology error.
- 9 ways are OSM tag errors (`circuits` < `cables`/3): for example Raitersaich–Trennfeld 211 (circuits = 1, cables = 6, 110.7 km; the model correctly has 211 + 212), Amprion 4134/4196/4114/4591 and Ltg 293/294+517/518.

## Proposals

1. **Display fix (geometry only, no model change):** give the 61 `osm-*`/`curated/*` circuits the geometry of their OSM export section. Their `route_source` already names it (`export section osm-NNNN`). Alternatively, let `split_by_towers` assign straight circuits through their `tower_way` / export-section way ids instead of by proximity. That removes 34 `no_model_circuit` ways (373 km) and 15 `model_fewer` ways at once.
2. **Display fix:** report the matched-radius neighbours (parallel ways, other voltage) separately. Change the mismatch rule to compare against `max(circuits, cables/3)` and skip ways under 0.5 km. About 1,000 `model_more` flags disappear. Route-offset items (110 ways) are already covered by route_quality review (class 02/05 tools).
3. **Model (v5, owner decision):** decide whether the frozen model is the 2021 grid or the 2026 grid. If 2021: build the German line set from the JAO **1st release** (2022-09) plus the 9th release only for parameters that r1 lacks, and treat 2023–2026 additions as class 01. If the model stays at 2026, add only these 2021 circuits, with the published r1 parameters listed above: Isar–Ottenhofen 444/446, Isar–Pleinting 452, Grosskrotzenburg–Trennfeld 218, Oberbachern–Oberbrunn 269, Bergrheinfeld–Raitersaich 278, Etzenricht–Schwandorf 248, Niederstedem–Wengerrohr WENGER, Conneforde–Cloppenburg/O rot, Lehrte–Godenau 1. Lehrte–Godenau needs a new Godenau 220 kV bus (no bus in the model), so it adds a node and a path. All others are parallel circuits on existing paths.
4. Do not add: plant leads, offshore export cables, abroad ways, post-2021 or under-construction lines (Stade–Landesbergen, Wahle–Mecklar new route, Ganderkesee–St. Hülfe, Klixbüll–Endrup, A-Nord, Ultranet, Ostbayernring).

## Effect on the model (analytic DC estimate)

Every addition except Lehrte–Godenau is a parallel circuit on an existing path. Its effect is a lower corridor series reactance, which shifts flow onto that corridor:
- Isar–Ottenhofen: one circuit at 15.31 Ω becomes 15.31 ∥ 17.29 ∥ 16.68 = 5.4 Ω (−65 %).
- Isar–Pleinting: 15.80 Ω becomes 15.80 ∥ 15.59 = 7.85 Ω.
- Grosskrotzenburg–Trennfeld: 28.36 Ω becomes 14.3 Ω.
- Oberbachern–Oberbrunn: 9.47 Ω becomes 4.73 Ω.
- Lehrte–Godenau: a new 220 kV branch to a new bus (topology change).
- Bergrheinfeld–Raitersaich 278 and Niederstedem–Wengerrohr add 220 kV paths between existing buses.

These changes matter most for N-1 events on the Isar (Isar 2 ran in 2021, 1,485 MW), Trennfeld and Eifel corridors. A full DC power flow was not run (read-only). It is the first verification step after approval. Recommended: a `dc_power_flow` on the v4final winter/summer snapshots with and without the circuits, reporting the loading change on the affected corridors and the number of v4.1 events on them.
