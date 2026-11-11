# 05 Substation locations: findings (Detect, Investigate, Propose)

Status: findings ready, **waiting for the owner's decision**. Nothing has been changed in the model or the export.
Model: `~/data/grid/v5/model_v5.pkl` (633 nodes). OSM: `~/data/grid/osm/DEU.gpkg` (power_substation polygon and point, 639 with >= 220 kV).
Export: `grid_model.gpkg` (548 substations, 5,826 drawn circuits). Scripts are in the cairo-lab home directory: `~/dq05.py`, `~/dq05_near.py`, `~/dq05_occ.py`; per-node distances are in `~/dq05_nodes.csv`.

## Detect
1. **Distance to the OSM substation.** Each node was measured against (a) its own `osm_bus` polygon, (b) the nearest name-matched HV polygon (umlauts folded, generic words removed) and (c) the nearest HV polygon of any name.
   - 477 nodes have an `osm_bus` polygon that resolves. Of these, 90 % lie at 0 m and 95 % within 471 m.
   - The distribution breaks at about 500 m, so **X = 500 m**.
   - 20 of the 509 substation, busbar and junction nodes are over X or are schematic.
   - Foreign nodes (CH, AT, NL, NO) fall outside the DEU extract and were not checked.
2. **Clusters.** 84 pairs of nodes from different sites lie within 300 m of each other; 41 pairs in the export lie within 50 m.
   - Almost all are legitimate: variants of one station (Eichstetten c1/c2, Großgartach 2, Urberach neu, Brunsbüttel 50H) or tees of parallel circuits on the same tower.
   - Two clusters are real errors: `herbentingen|380` + `y-hoheneck#herbtg w` placed at Hoheneck, and `y-kuhmoos#bl` + `y-laufenburg#rtge` sharing one schematic point.
   - Meitingen is no longer snapped: its model point lies inside OSM "Station Meitingen". `herbertingen` + `y-tiengen#bludnz w` is the Bürs corridor case, already in class 02.
3. **Line ends against locations.**
   - Drawn end gap (`end_gap_m`): median 78 m, 99 % at or below 1.3 km. 28 drawn circuits have a gap over 2 km; they come from 6 lines.
   - Published length against straight-line distance between the end nodes: 50 circuits are shorter than 95 % of the crow distance. Most are tie lines that carry only the German share (expected). The domestic ones are listed below.

## Topology and model effect (priority)
| id | node | finding | proposal |
|---|---|---|---|
| 05-01 | herbentingen\|380, y-hoheneck#herbtg w, y-herbertingen#herbtg w | JAO EIC 11T-D4-D7-00016J "Y-Metzingen (-Herbertingen - Hoheneck)" is published as **1.0 km, X 0.36 Ω**: a stub at Metzingen. The model has two halves of it. The Amprion half ends at the typo bus `herbentingen` and the TransnetBW half at a tee; both are schematic points at Hoheneck. Both halves are stretched to 61.8 km and their X is replaced by 15.7/22.0 Ω. The Y-Hoheneck leg runs hoheneck–herbertingen amprion at 121.7 km (published 62.8 km). All three tee nodes have degree 1, so the tee is broken. | One tee about 1 km from Metzingen on the Herbertingen–Hoheneck route, with legs Herbertingen 61.4 km, Hoheneck 62.8 km and Metzingen 1.0 km (published X). Drop `herbentingen\|380`. Coordinate with class 02. |
| 05-02 | y-erzhausen#4\|220 | The tee is split across voltages. The 380 kV legs (Hardegsen, Lamspringe) end at `#4\|380`; the Erzhausen leg (0.27 km) ends at `#4\|220`, which has degree 1. | Check the leg's voltage in JAO and merge into one tee. |
| 05-03 | y-erzhausen#2\|220 | Dead-end tee with one leg (5.2 km) and a schematic point. | Classes 01/02: find the missing legs. |
| 05-04/05 | y-kuhmoos#bl, y-laufenburg#rtge | Two dead-end tees at the same schematic point. | Classes 02/borders. |
| 05-19 | osm-4706\|220 | Junction not located. All 6 circuits get the same estimated 13.35 km, but the crow distances are 0.17, 14.35 and 19.93 km. | Locate the junction on the OSM route and take leg lengths from the OSM ways (class 04). |
| 05-20/21 | Walzwerk, Schwelgern (OSM export) | Estimated length is below the crow distance (1.59 vs 3.39 km; 1.56 vs 2.08 km). | Use the OSM route length (class 04). |
| 05-26 | Gießen/N – Aßlar 4 | Both ends are located correctly, 12.5 km apart, but the published length is 1.13 km. Not a location error. | Class 04. |

## Display only (published lengths are not affected)
| id | node | evidence | proposed coordinate |
|---|---|---|---|
| 05-06 | kus lungenkamp\|380 | "KUES Lüngenkamp" = OSM "Kabelübergabestation Lüningkamp" way/844075808 (Amprion 380 kV); the model point is 7.3 km off. The node has degree 1: the Gronau–Niederrhein circuit stops here (see classes 01/02). | 6.88826, 51.85061 |
| 05-07/08 | wurgassen 380/220 | Schematic point 9.40/51.64. The only HV station there is TenneT 380 kV way/274630377, 657 m away. | 9.38786, 51.64061 |
| 05-09 | gkm\|220 | Schematic point at Ladenburg, 4.6 km off. GKM Station B is relation 17591073 (= gkmb); TransnetBW "Umspannwerk Mannheim" 380/220 kV is way/197145478. | 8.49318, 49.44606 (owner: confirm which polygon is Station A) |
| 05-10 | ried\|380 | Schematic point. Biblis A way/963454861 (380 kV). The polygon 702 m away is the post-2021 gas-turbine substation. Dead-end stub. | 8.41547, 49.71072 |
| 05-11 | lippborg\|380 | The "verified" point lies inside the 110 kV Station Lippborg, and `osm_bus` points to Uentrop. Unnamed Amprion 380 kV polygon way/741631067 is 945 m away. | 7.96812, 51.67636 (owner: confirm) |
| 05-12..18 | meitingen, gurtweil, st barbara, benteler, bergrheinfeld/west, diefflen, urdingen | The point lies inside the correctly named OSM polygon, but `osm_bus` references a neighbouring polygon 0.5–1.2 km away (Gurtweil→Tiengen; Bergrheinfeld/West→SuedLink converter). | Keep the coordinates; correct the `osm_bus` reference only. |
| 05-22..25 | Kusenhorst–Hanekenfähr, Dauersberg–Eiserfeld, Gütersloh–Hesseln, Altbach–Mühlhausen | Drawn geometry ends 1.7–7.0 km from the station. | Close the gaps in the geometry overlay. |

## Owner decisions
1. 05-01 (Herbertingen–Hoheneck–Metzingen tee): rebuild the tee with the published leg lengths and reactances, and drop the typo bus? The current model carries the 1 km stub as 61.8 km with about 44–61 times the published X. This changes power flow, so it would be a v5 model change.
2. 05-02/03/04/05/19: handle in classes 01/02/04 (dead ends, split tees, estimated lengths), or here?
3. Display moves 05-06..11: approve the coordinates, particularly GKM Station A and Lippborg 380 kV, which are unnamed or ambiguous in OSM.
4. 05-12..18: approve the `osm_bus` reference fixes (metadata only).
5. 05-22..25: approve the overlay gap fixes.
