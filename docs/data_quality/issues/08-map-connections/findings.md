# 08 Map connection anomalies: display or model?

Inputs: the live map layers in `~/data/grid/web/` (export of `model_v5_3e.pkl`, sha 8be55949), the station table `~/data/grid/v5/stations/`, OSM `~/data/grid/osm/DEU.gpkg`, JAO Core SGM r9 publication HTML, and the v4.1 events in `dq04/events_by_mrid.csv`.
Scripts (on cairo-lab): `~/scratch/conn/{case.py,scan.py,shots.py}`. Full list: `occurrences.csv` (1,769 rows).
Figures: `scratchpad/shots/conn_1..6_*.png`. These are rendered with matplotlib from the map's own GeoJSON layers plus OSM polygons and transformers. They are not browser screenshots.

No power-flow sensitivity runs were made. A gate run is 18 cases in two solvers, which is too heavy while other agents are running. The effects below are reasoned from the topology.

## Verdict per case

| # | Case | Class | Effect on power flow |
|---|---|---|---|
| 1 | Mehrum / Mehrum-Nord | **Model topology**, plus a model location error and a display error | `mehrum/nord|220` is a dead stub: its only connections are the two leads T421/T422 to `mehrum|220`, and there is no 380/220 kV transformer. As a result, Mehrum's 220 kV area (Lehrte, Hallendorf, Gleidingen) has no 380 kV infeed at Mehrum. Already listed in KNOWN_LIMITS (06 #5). v4.1 events: T421 7 + T422 9 = 16. |
| 2 | Bentwisch box (labels 317, 318, 275;276) | Display only | None. |
| 3 | 544 pair ending in a yard with no footprint | Display only (the same station as case 2) | None. |
| 4 | Siedenbrünzow | Display only | None. |
| 5 | Marke | Display, plus the location of the plant attachments | None on topology. 158 MW of gas units are attached to `marke|380` through 9–13 km straight lines. |
| 6 | Perleberg after the merge | Display only | None. The merge is correct, and no through-circuit was cut. |

### 1 Mehrum / Mehrum-Nord
- **Model:**
  - `mehrum|220` (OSM 274621059, 220/110 kV, TenneT) has degree 6: Lehrte 1/2, Hallendorf 1, the Y-Mehrum leg, and T421/T422.
  - `mehrum/nord|380` has degree 4: Algermissen, Grohnde, Klein Ilsede and Wahle.
  - `mehrum/nord|220` has degree 2 (T421/T422 only) and no transformer.
  - `mehrum/nord|220` is placed at exactly the `mehrum|220` point. Its OSM bus was matched to the 380-only polygon 1011084707 at osm_d_m 288, so the placement fell back to that point.
- **Physical:**
  - Mehrum-Nord consists of polygon 1011084707 (380 kV) plus polygon 1453341874 (380 kV, west part).
  - Three untagged transformers are inside: 13328282477, 10027531799 and 10027531805.
  - Single-circuit 220 kV ways (1094896263–268, 1138383588–591) run from the Mehrum 220 kV yard to them. These are the transformer leads.
  - JAO r9 lists the leads "Mehrum - Mehrum/N T421/T422" but no "TR Mehrum/N" element.
  - The same naming at Audorf ("Audorf - Audorf/S T421/T422" leads plus "TR Audorf/S T421/T422") shows that T421/T422 are 380/220 kV couplers whose transformer elements are missing at Mehrum.
- **Display:**
  - The 220 kV routes of Lehrte–Mehrum 1/2 and Mehrum–Hallendorf 1 were matched onto those lead ways. They end inside Mehrum-Nord, which creates the convergence point, with end_gap 476 m.
  - The 380 kV routes end at the footprint edge, with 380 kV stubs drawn at 0.8 px.

### 2 + 3 Bentwisch
- **Model:**
  - `bentwisch|220` sits in the 220 kV yard (OSM 1464801125, which has a 220/110 kV transformer). Its circuits are 275 and 318.
  - `bentwisch|380` sits in the 380 kV yard 1.3 km south (OSM 32063539, with 380/110 kV and 380/150 kV transformers and the HVDC back-to-back). Its circuits are 543 and 544.
  - There is no transformer between them.
- **Physical:**
  - Neither OSM nor JAO has a 380/220 kV transformer, so these are two separate yards and the model topology is right.
- **Display:**
  - The site has no footprint polygon, so a "box at circuit ends" is built from the 220 kV ends only. Both bars are drawn in that box.
  - The 380 kV bar therefore has no lines, while 543/544 end 1,286 m away in an undrawn yard.
  - 543/544 also have a synthetic piece (piece 3, no tower way) and pieces whose direction alternates.
  - 275 and 318 have a 351–359 m gap at the box.

### 4 Siedenbrünzow
- The model has 509, 510, 512 and 513. This matches JAO, and the site is 380 kV only (OSM has 3 × 380/110 kV).
- The pair that continues past the station is 220 kV Pasewalk–Güstrow 315/316 on the shared way 59032803. They correctly do not end there.
- 512/513 reuse a 150 m piece of way 59032791, which is tagged 509;510. The stubs are hard to see. Display only.

### 5 Marke
- **Model:** `marke|380` has degree 2: 500 (Jessen/Nord) and 504 (Lauchstädt).
- **Physical:** JAO r9 lists only these two circuits at Marke. OSM way 41378147 ("504;500", 2 circuits) carries both out of the station to the south-west.
- **Connectivity:**
  - The node is weakly connected (two circuits on one tower line). This is real, and its main tie is the 110 kV grid, which is not modelled (OSM has 3 × 380/110 kV).
  - No connectivity error.
- **The two thin straight lines:** these are **plant connections**, not circuits: Wolfen GT (50 MW, 9.4 km) and Bitterfeld GuD (108 MW, 12.7 km), attached to the nearest 380 kV substation.
- **Scan for straight drawings:**
  - 26 circuits have no matched route (schematic/none/straight overlay).
  - Every one connects two model nodes that JAO also connects. Most are Y-junction legs, foreign ties (Dötinchem, Endrup, Lixhe) and HVDC.
  - None hides a connectivity error. The 3 zero-length ones (Y-Werben ×2, length 0.0) are already known overlays.
- **Plant connections:** 130 plant connection lines are longer than 5 km. This is a location matter (aggregation to the nearest substation), not a topology matter.

### 6 Perleberg (after 06 #4)
- **Model:** after the merge, `perleberg|380` has degree 2: 516 to Stendal/West and 596 to Putlitz/Süd.
- **OSM:**
  - Both leave east on one corridor: ways 259468201 and 30419417 "596 und 516", then 259468200, which is a 3-circuit way shared with 220 kV 332.
  - They separate about 14 km east at 11.975 E. 516 turns south (way 41036073, shared with 515) and 596 turns north-east. The drawing matches.
  - The gap and offset near the station come from piece boundaries.
- **Green lines:**
  - The green lines are 220 kV Wolmirstedt–Parchim/Süd 332-322. It passes Perleberg on shared towers (259468200 and 1243590043 "436 und 332").
  - Its model ends are Wolmirstedt and Parchim/Süd, which is correct.
  - There is no 220 kV substation at Perleberg in OSM (59707532 is 380 kV only).
- **Line 436:** OSM 380 kV "Ltg 436 Perleberg–Parchim Süd" is the Ersatzneubau. It is not in JAO r9, so it is correctly absent.
- **The merge:** it created no wrong ending. 515 still passes as Putlitz/Süd–Stendal/West.

## B. Systematic scan (counts; lists in occurrences.csv)

| Check | Count | Class | Notes |
|---|---|---|---|
| Distinct-site nodes < 50 m apart | 20 pairs | model location (no flow effect) | Mostly name variants of one site (Eichstetten c1/c2, Großgartach 2/c2, Urberach neu, Herbertingen Amprion, Brunsbüttel 50H, Conneforde/Ost, Remptendorf tee). Real separate yards that were collapsed: **mehrum/nord\|220**, **audorf/sud\|220** (both lead-pair patterns, each 0 m from the other yard), and halfeshof / osm/1463882574 (30 m). |
| Level with no transformer while another level has lines | 27 nodes | 19 same polygon (candidate missing TR, **model topology**); 8 separate yard | Same polygon: Birkenfeld, Brunsbüttel, Emden/Ost 380, Garenfeld, Gütersloh 380, Hesseln 380, Ibbenbüren, Lüstringen, Osterath, St. Barbara, Wiesloch. Several are already in KNOWN_LIMITS (06 #3/#6/#8). Separate yard: Bentwisch, Mehrum/Nord (the scan treats it as separate because of distance, but evidence says missing TR, see case 1), Beznau, Laufenburg (foreign). |
| Bar with zero circuits | 12 | 8 display (bar has transformers only); 4 model (fully isolated: emden/ost\|220, gutersloh\|220, hesseln\|220, osm/1463882574\|220) | The isolated buses carry no flow but count as islands in the gate (islands_max 9). |
| Route end > 50 m from bar | 1,343 | display | 1,326 have a stub (by design: routes stop at the footprint edge). **17 have no stub**: Bentwisch 543/544 (1,286 m), Mehrum/N T421/T422 and Audorf/S T421/T422 leads (215/355 m), Remptendorf ×6 and Conneforde ×5 (151–218 m, stubbed by a different mrid). |
| Route gap > 100 m within a circuit | 13 circuits | display | e.g. Lüdershagen–Bentwisch 318 (351 m), Bentwisch–Güstrow 275 (359 m), Gurtweil–Laufenburg ge (314 m), Lehrte–Mehrum 1/2 (103 m) |
| Split/rejoin (pieces overlap > 100 m) | 4 | display | POLSUM S (565 m), Koblenz SOONWD O, Gurtweil–Laufenburg ge, Mehrum–Hallendorf 1 |
| `end_gap_m` > 300 m | 180 circuits | display | Top: Y-Metzingen HERBTG W 41 km, Y-Kusenhorst GRAFSH W 7.0 km, Güterloh–Hesseln HALLE W 2.6 km, Zukunft–Oberzier 1.3 km, Y-Bergmannsglück 1.2 km |
| OSM way crosses a footprint and carries a circuit that ends there | 6 | display | GKM ×2, Mehringen, Rommerskirchen, Streumen, Werben. The circuit's own drawn piece is cut, but the grey tower way continues through. No model endpoint is wrong. |
| Model node > 1 km from the drawn route end | 0 | n/a | none |
| Circuit drawn straight (no matched route) | 26 | display | none hide a connectivity error (see case 5) |
| Plant connection lines > 5 km | 130 | plant attachment location | drawn straight by design |

No scan check found a new topology error beyond the known ones. The model-affecting items are Mehrum/N (case 1), Audorf/S (correct: it has its transformers), the 19 same-polygon levels without a transformer, and the 4 isolated bars.

## C. Proposed solutions, ranked by impact

### Model changes (for the gate process)
1. **Mehrum/N: add TR Mehrum/N T421 and T422** (380/220 kV, `mehrum/nord|380`–`mehrum/nord|220`).
   - Evidence:
     - the JAO lead names T421/T422;
     - the Audorf/S analogue in r9;
     - three untagged OSM transformers inside the Mehrum-Nord polygons, fed by the 220 kV lead ways from the Mehrum 220 kV yard;
     - TenneT's Mehrum is a 380/220/110 kV node.
   - Values: no published values exist. Take r/x/Sn from the r9 TenneT 380/220 kV units of the same series (e.g. TR Audorf/S T421/T422) and mark them `estimated`.
   - Expected effect: the Lehrte–Mehrum–Hallendorf 220 kV area gets a 380 kV infeed. Flows on Lehrte–Mehrum 1/2, Mehrum–Hallendorf 1 and the Y-Mehrum leg change, and the 380 kV loading on Grohnde/Algermissen–Mehrum/N shifts. The T421/T422 leads (16 v4.1 events) start carrying flow.
   - Gate: one change, then a gate against 5_4a.
2. **Move `mehrum/nord|220` to its own location** (centroid of the T421/T422 lead ends at the 380 kV yard, ~10.0882, 52.3181, OSM 1453341874). This is a location change with no flow effect. Do the same for `audorf/sud|220`, which is 0 m from `audorf|220`.
3. **The 19 same-polygon levels without a transformer.** Most are already handled by 06 #3/#6/#8 or KNOWN_LIMITS. Re-check Garenfeld, Lüstringen, Osterath and St. Barbara, which have lines on both levels, for a missing coupler. That is the same signature as Mehrum.
4. **The 4 isolated bars** (emden/ost|220, gutersloh|220, hesseln|220, osm/1463882574|220): drop them from the solved case, or tag them as out of service. They inflate the island count in the gate and carry no flow.

### Display changes (grid_stations.py / export)
1. **Per-level footprints when the levels sit in different OSM polygons** (Bentwisch, Mehrum/Nord). In `grid_stations.py`, when a station's bus nodes have distinct `osm_id`s more than 300 m apart, emit one footprint per OSM polygon and draw each bar in its own polygon. Fall back to the box only for levels that have no polygon. This fixes cases 2 and 3.
2. **Build the fallback box from all circuit ends of all levels**, not only the ends of the first level. It is a stop-gap if (1) is not done.
3. **Do not match a circuit's route onto ways that join two of the same site's polygons** (the lead ways at Mehrum, way 59032791 reused at Siedenbrünzow). In the route matcher, cut the route at the first entry into the end station's footprint, so the end is the footprint edge and not a point deep in another yard. This removes the Mehrum convergence point and part of the 180 large end gaps.
4. **Stubs:**
   - draw stubs at the circuit width, not 0.8 px;
   - draw a stub for every circuit end that has none (the 17 rows above), including leads and circuits whose stub goes under a different mrid (Remptendorf, Conneforde).
5. **Gaps and overlaps:**
   - Bridge internal gaps up to 500 m with a dashed connector, so gaps are not drawn as missing (13 circuits).
   - Drop overlapping duplicate pieces (4 circuits).
   - Orient pieces consistently from bus1 to bus2.
6. **Plant connections:** style them as thin dashed lines and label them in the popup as "plant attachment (straight)", so they are not read as circuits (Marke).
7. **Tower ways through end stations:** clip the grey OSM ways at footprints, or draw them under the footprint fill (6 sites).

## Cases 7 and 8 (added later)

### 7 A 380 kV meeting point in open field
- **Not uniquely identified.** I had no screenshot coordinates.
- **Scan:** 49 junction nodes at 380 kV. 22 junctions (all voltages) have more than 2.5 drawn pieces per circuit within 1 km. Most fragmented at 380 kV:
  - y-lippborg (7.982, 51.677: 37 pieces / 13 circuits);
  - y-ohligs (7.004, 51.177);
  - y-kriftel (8.490, 50.085);
  - y-vöhringen;
  - y-koblenz SOONWD;
  - y-ried;
  - y-waldlaubersheim.
  
  The full list is in occurrences.csv (`junction_fragmented_drawing`).
- **Model connection:** every one of these is a model tee (`y-…` node, degree 3) built from JAO Y-elements, so the model does connect there. I found no 380 kV place where three circuits meet in the drawing without a model node: the "node far from route end" check found 0.
- **Why there is no symbol:** the map exports tees as `kind=tee` (90) in substations.geojson, but they get no footprint. The edge-port logic therefore does not run, and each leg is offset side by side right up to the point. That gives the short offset pieces and gaps.
- **Class:** display. Confirm the exact site from the owner's screenshot coordinates.

### 8 Güstrow
- **North-west entry:**
  - The route ends of 423, 424, 543 and 544 are at 12.180–12.183, 53.821. From there a straight synthetic leg runs to 12.185, 53.816 (end_gap 192 m). That is the right angle.
  - The model ends of all four are `gustrow|380`, which matches JAO.
  - This is a route-end assignment artefact (display). It does not come from a junction.
- **North-east pair:** 512 and 514 end at 12.1889, 53.8165, about 70 m from the 380 kV bar. A stub is drawn. Display only.
- **Bars:** the 220 kV bar is 140 m long and the 380 kV bar 430 m. Bar lengths come from the OSM bus extent. Cosmetic.
- **Parchim Süd–Güstrow 321:** the 3e map draws it as in operation. CHANGELOG entry 17 (B2) sets it out of service in the later model, so this is map staleness only. The dotted twin is 328, which is already decommissioned (CHANGELOG entry 2).
- **Transformers:**
  - The model has one 380/220 kV transformer: `TR Güstrow 220/400 405`, 400 MVA, x 64.9 Ω. It connects `gustrow|380` to `gustrow|220`, which is correct.
  - JAO r9 lists only this one.
  - **OSM has 4 × 380/220 kV** (2426825594, 2426825599, 1791864336, 2426825611).
  - **Model topology candidate:** the coupling may be under-represented by 4× in impedance and capacity, unless JAO 405 is an equivalent of the whole bank. To check:
    - the 50Hertz unit list;
    - the r9 transformer Sn compared with the four OSM units;
    - in the next gate, the loading of TR 405.
  - It affects flows between the 220 kV and 380 kV networks in Mecklenburg.

### 3 (close-up) The Bentwisch 380 kV yard
- **The yard:** it is OSM way 32063539, "Bentwisch HGÜ", operator 50Hertz, `voltages=380000`, 48,210 m². It is tagged correctly and matched correctly. The station table has `bentwisch` with `osm_ids = 1464801125;32063539`, footprint_m2 64,169, which is the sum of the two polygons. It also contains the converters 940919943 and 1265537498, the 380/110 and 380/150 kV transformers, and a compensator.
- **Why there is no footprint:**
  - `grid_stations.py` (the `_fallback_boxes` call, ~l. 507) replaces the station's footprint with a "box at circuit ends" when circuit ends miss the footprint.
  - The box only covers ends within `FALLBACK_SPAN_M` = 600 m of each other. So it was built around the 220 kV ends.
  - The two-polygon footprint, including the 380 kV yard, was dropped.
  - Then both bars were drawn in the box, and 543/544 kept ending at the `bentwisch|380` node point in the undrawn yard.
  - This is a display bug in the box fallback. I inferred it from the code and the data and did not trace it step by step.
- **One site or two:**
  - **Two yards, 1.3 km apart.** The 380 kV yard (50Hertz) has 380/110 kV, 380/150 kV (Kriegers Flak) and the HVDC back-to-back. The 220 kV yard (1464801125, E.DIS area) has a 220/110 kV transformer (2595989616).
  - No 380/220 kV transformer exists in OSM, and none in JAO r9.
  - The two levels are coupled only through 110 kV, which is not modelled. **The model has no transformer error here.**
- **The route:**
  - Pieces 0–2 of 543/544 follow ways 253707758/757/749, 59581820 and 220225905 inside and near the yard.
  - Piece 3 (12.21377, 54.09889 → 12.21441, 54.09517, ~415 m) has **no tower way**. It is a straight connector that the route builder made between the end of way 220225905 and the start of the long way 59049216. The OSM ways there go north-west to a pole and then north-east, and those ways were not part of the matched chain.
  - The ~150 m gap is where the side-by-side offset of the two circuits meets this unmatched connector. The pieces alternate direction (a/b reversed), so the offset flips side.
  - Class: display.
- **Fix:**
  - In `_fallback_boxes`, keep the station-table footprint for levels whose ends do hit one of its polygons. Build a box only for the level that misses, or emit one footprint per OSM polygon (see C display 1).
  - In the route builder, fill gaps between matched ways with the OSM path (shortest path over hv_lines ways within 1 km) before falling back to a straight connector.
