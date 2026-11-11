# 06 Station checks (PLAN.md D4): classification

Classification only: no change to the model, the map or the station code.
Inputs: `~/data/grid/v5/stations/checks.csv` (309 rows), `model_v5_1.pkl`, `~/data/grid/osm/DEU.gpkg`, JAO Core SGM r1 (D2 template) and r9 (202603 External.xlsx), v4.1 event counts from `~/data/grid/v5/dq04/events_by_mrid.csv`.
Each row of `checks.csv` has a class, evidence, v4.1 events and a proposed fix in `occurrences.csv`.

## Counts per class (rows)

| Class | Rows |
|---|---|
| real error (topology) | 35 (10 distinct issues) |
| real error (location/metadata only) | 15 |
| matching artefact (fix the check) | 90 |
| intended/legitimate | 130 |
| in step 4 (A1–A4: Herbertingen, Erzhausen, Kühmoos/Laufenburg, Remptendorf) | 39 |

## Real topology errors, ranked by v4.1 events on affected lines

| # | Issue | Evidence | Events | Proposed fix (not applied) |
|---|---|---|---|---|
| 1 | Westtirol: 380/220 kV coupler missing | JAO r9 Transformers: `TR Westtirol WTRHU41` 400/220 kV (APG, 14T-38220-WT041O). The model has no transformer, so the 220 kV ties (Vöhringen FUESSN W, Bürs ws, Silz 274E/413/414) and the 380 kV ties (Leupolz, Bürs rt) do not couple. | 75 (7 lines) | add WTRHU41 with the JAO r9 parameters |
| 2 | Kriftel–Frankfurt TAUNUS 2 is dangling | JAO r9 tie-line `Kriftel - Frankfurt TAUNUS 2` (Amprion, 11T0-0000-0030-G) and TenneT's `Y-Kriftel (-Frankfurt/SW - Karben) 2` (11T0-0000-1032-4) describe the same circuit (14.8 km against 9.8 + 5.6 km). The model ends TAUNUS 2 on its own node `frankfurt|380` (kind foreign), inside OSM polygon 62061952 (Frankfurt Südwest). | 11 | drop TAUNUS 2 and `frankfurt|380`; move its events to the Y-Kriftel leg. Do not merge the nodes, because that would double the circuit |
| 3 | Ibbenbüren 220 kV is a dead end with no load | JAO r9 line `NIKE S` 220 kV (11T0-0000-0545-9). JAO has no Ibbenbüren transformer. OSM polygon 29472852 contains a 220/110 kV transformer. | 19 | attach a 220/110 kV lumped load |
| 4 | Perleberg is split in two (medium confidence) | JAO r9 lines 516 `Perleberg` and 596 `Perleberg - Putlitz` end on two different names. OSM 59707532 is a single 380 kV-only site with 4 × 380/110 kV transformers. The model turns the Stendal/West–Putlitz/Süd through-path into two radial stubs. | 17 | merge or bus-tie the two nodes after checking the 50Hertz grid map |
| 5 | Mehrum/N: 380/220 kV couplers missing | JAO r9 lists the 220 kV leads `Mehrum - Mehrum/N T421/T422` but no transformers. Audorf shows the same pattern, and there JAO does list `TR Audorf/S T421/T422`. OSM polygon 1011084707 contains 2 transformers. As modelled, `mehrum/nord|220` is a stub with no injection. | 16 | add two 380/220 kV couplers |
| 6 | Birkenfeld 220 kV is a dead end with no load | JAO r9 line `Birkenfeld - Daxlanden ge` (11TD400000581-LA). TransnetBW lists no Birkenfeld transformer. OSM 32966636 (380;220;110) has 7 transformers with untagged voltages. | 11 | attach a 220/110 kV lumped load |
| 7 | Beznau 220 kV has no injection | JAO r9 tie-line `AARE O` 220 kV (10T-CH-DE-000034). `beznau|380` has an injection; `beznau|220` has none. | 11 | add a boundary injection at `beznau|220` |
| 8 | Wiesloch 220 kV is a dead end with no load | JAO r9 line `Neurott - Wiesloch bl` (11TD400000582-L6). No Wiesloch transformer is published. OSM 46584363 (380;220;110). | 10 | attach a 220/110 kV lumped load |
| 9 | Brunsbüttel 220 kV: coupler missing | JAO r1 lists `Trafo Brunsbuettel T421` 380/220 kV (11TD2T0000002013); it is gone in r9. JAO r9 still lists `Brunsbuettel - Itzehoe/W rot` 220 kV. | 8 | add T421 with the r1 parameters for the event period |
| 10 | Hüffenhardt island from OSM | OSM-only line osm-4782-0 `Anlage 5230; Großgartach - Obrigheim 220 kV` between OSM-export sites osm/143903369 and osm/159326655. There is no JAO row and no injection, so this is a two-bus island. | 0 | drop it, or set it out_of_service |

The current model gives lines 3 and 6–9 no flow, because each line ends on a bus with no transformer, load or generation. Where no transformer is published, the fix should come from the existing 110 kV lumping method, not from a new estimate.

## Location/metadata only (no effect on the power flow)
- Emden/O: JAO r9 lists `TR Emden/O T425/T428` (380/220 kV), but they are missing from the model. The 220 kV side has no line in JAO. `osm_bus way/1280178911` is the **DolWin 6 HVDC converter** polygon, not the Emden/Ost substation.
- Diele 220 kV: T1 is modelled, but no 220 kV line is published, so the transformer feeds an empty bus.
- GKM, Ried, Würgassen 220/380 kV and KÜS Lüngenkamp have schematic coordinates and no OSM polygon within 500 m. Re-locate them in class 05.
- 5 pairs of tees with **different** names sit at one identical point (Bühl/Daxlanden, Schwelgern/Walsum, Elmenhorst/Pöppinghausen, Bixterheide/Garenfeld, Birkenfeld/Bruchsal). In each pair, one tee took the other's coordinate.

## Intended/legitimate
- **Foreign nodes outside DEU.gpkg:** 28 info rows.
- **Lippborg:** a verified switching point with no OSM polygon.
- **Radial feeders with an injection:** 47 dead-end warnings. This includes the 380 kV halves of Ibbenbüren, Wiesloch, Beznau and Hesseln, and the JAO nodes `Grossgartach 2` / `Großgartach C2`.
- **Gütersloh and Hesseln 220 kV:** the only line, HALLE W (11T0-0000-1955-G), has been out_of_service since 2021-12 (class 01).
- **220/380 kV sites with no coupler in JAO r9:** Bentwisch, Garenfeld, Lüstringen, Osterath, St. Barbara, Gütersloh, Hesseln. Both levels are meshed separately. OSM shows only 220/110 or 380/110 kV transformers.
- **Two-leg Y-points:** Bellendorf, Bidingen, Dahlem ×2 and Hanekenfähr GRAFSH W. JAO r9 lists exactly these two legs.
- **Y-tees at the fence of one of their leg stations:** 31 rows.

## Matching artefacts and proposed check fixes
1. `node_in_two_stations` (44 rows): several JAO sites lie inside one OSM substation polygon and are tied there by short JAO lines or transformers. Examples are Krümmel 993/994, Audorf T421/T422, Eichstetten C1/C2, Urberach Neu, Conneforde/O, DCPH1, RDK7/8, and the "50H" and "c1/c2" cases. **Fix:** allow several sites per polygon when they are connected inside it. Separately, Schwanheim is matched by the nearest-polygon fallback to FWH-S. **Fix:** skip that fallback when the polygon name belongs to another model site.
2. `shared_coordinate` (20 rows): sites of one substation share a point, and same-name tees on a double-circuit tower share a point. **Fix:** exempt both cases.
3. Tee nodes with no legs (Wuhlheide Abzweig 302/1 and 302/2, Y-Grafenrheinfeld ge/rt): the lines end on the station itself. **Fix:** drop leg-less tees in the station build.
4. Halfeshof 220 kV (`osm/1463882574`): a duplicate OSM-export site with no branch. JAO has Halfeshof at 380 kV only. **Fix:** drop OSM-export sites that have no branches.
5. Names (11 rows): the normaliser strips digits and suffixes, so 50H, C1/C2, 2 and RDK7/8 collapse together. Meitingen, Metzingen and Meiningen are distinct sites. **Fix:** keep digit and suffix tokens, and keep a whitelist.
6. `node_no_station` warnings for Beznau, Bürs, Kühtai and Silz: these nodes are in CHE/AUT and the check only searches DEU.gpkg. **Fix:** report them as info, as already done for other foreign nodes.

## Caveats
- JAO r9 does not list every transformer; for example, Amprion lists 41 transformer rows. The "no coupler" verdicts rest on that absence plus the OSM transformer voltages. Lüstringen and Osterath are the least certain of these.
- Perleberg (#4) is medium confidence. The fixes for #3, #6 and #8 assume 220/110 kV load and no 380/220 kV coupler.
- The event counts are the sums of `events_by_mrid.csv` over the lines that end on the affected node(s).
