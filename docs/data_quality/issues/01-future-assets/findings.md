# 01 Future assets and dead-end busbars: findings (Detect, Investigate, Propose)

Status: findings ready, waiting for the owner's decision. Nothing in `model_v4final` or in the v4.1 data was changed.
Data: `~/data/grid/model_v4final.pkl` (CIM tables + silver), ANDES case `high_load_t10.xlsx`, and the v4.1 events (11,162 events, 1,328 unstable, 18 cases).
The full list is in `occurrences.csv` (145 circuits). Its `category` column says which detector flagged each one:
`dead_end_bus_empty`, `dead_end_bus_no_transformer`, `tso_project`, `osm_relation_absent_2021_or_planned`, `high_eic`.

## How fault_line maps to circuits
`fault_line` is the row index among the ANDES Line rows with `trans==0` (1,015 rows). Those rows follow the CIM `ac_line_segment` order with `status==in_operation` (1,016 rows). Exactly one row is skipped: `osm-4782-0` "Anlage 5230; Großgartach - Obrigheim", which connects two empty OSM buses (an island). Matching the bus pairs reproduces all 1,015 rows.
`radial_bridge` = the circuit is a single-circuit bridge in the line + transformer graph. Removing a bridge cuts off a bus; it does not redistribute flow through the mesh.

## Detect: summary
- **(b) Status fields:** 1,016 of the 1,019 circuits are `in_operation` with no `status_date` or `status_evidence`. In practice the field carries no information; it is not a detector.
- **(a) Dead-end busbars:** 68 buses have exactly one line, no transformer and no external injection.
  - 19 of them also have no generation and no load. These are strict dead ends: foreign or tee stubs, 220 kV halves of 220/380 kV sites with the coupling transformer missing (Ibbenbüren, Hesseln, Brunsbüttel, Birkenfeld, Wiesloch, Beznau), and the Hüffenhardt OSM island.
  - The other 49 carry only *estimated* register infeed or load from the attachment step. Merzen belongs here: 605 MW of estimated register generation and 84 MW of population load are attached to a bus that has no transformer in the model.
  - These buses belong to classes 02 and 04 as much as to this class. They are listed so that the owner can decide.
- **(c) and (d) OSM relations:** 13 route relations used as `route_evidence` did not exist in OSM on 2021-01-01 (Overpass date filter), or are tagged `planned=yes` (relation 1141617, 483 Vierraden–Neuenhagen).
- **High EIC numbers:** 53 circuits carry EICs `11T0-0000-2249` or higher. JAO notices show that Amprion and TenneT allocated this range for 2023–2025 commissionings (2249/2250 BACCUM 2024, 2262/2263 2024, 2414/2415 BRAGAU at the Merzen split, 2480 FREREN N 2025). These EICs mark new or reconfigured circuits.

## Investigate and Propose: verified assets
| Asset (EIC) | Evidence | Verdict | v4.1 events (unstable) | Topology | Proposed v5 fix |
|---|---|---|---|---|---|
| Merzen – Westerkappeln BRAGAU O (11T0-0000-2414-H), bus merzen\|380 7.8435 E 52.4585 N | Amprion project page: "Eine Schalt- und Umspannanlage (UA) ist in Merzen geplant", "vollständige Inbetriebnahme ist für 2028 geplant" (BBPlG 6). JAO 23/09/2025: new FREREN N Hanekenfaehr–Merzen from 26/09/2025, which the model does not contain. OSM at 2021-06-01: relation 962122 "Bramgau" runs from=Westerkappeln to=Hanekenfähr (tower lines 4132 and 4583, Merzen passed without a station) | after 2021 (split 2025, station planned 2028) | 11 (0) | radial stub | **Reroute to the 2021 topology**: one Westerkappeln–Hanekenfähr Bramgau circuit, parallel to BRAGAU W (60.7 km, 11T0-0000-2415-E). Remove merzen\|380 and move its estimated register injections and load to the neighbouring stations. This changes the power flow: the stub carries about 520 MW net today and would become a meshed parallel path |
| Ganderkesee – St. Hülfe HUNTE SW (1934-R) | JAO 08/09/2023: commissioned 21/09/2023 | after 2021 (Sep 2023) | 3 (1) | meshed | Remove for 2021. Check what HUNTE WS (1935-O, 10/2) replaced; it may be the 2021 220 kV path |
| St. Hülfe – Wehrendorf OCHSMO O/W (2066-B, 2067-8) | JAO: "expecting the commissioning ... on 31/07/2023" | after 2021 (Jul 2023) | 31 (1) | meshed | Remove, or restore the 2021 220 kV predecessor if JAO/OSM 2021 show one |
| Hanekenfaehr – Meppen BACCUM W (2250-N); BACCUM O (2249-0) | JAO 2024 notice: Baccum West in service in Sep 2024 | W after 2021; O unknown (EIC from the same batch) | 18 (1) | meshed | Remove W. Check whether O existed in 2021 |
| Daxlanden – Maximiliansau GOLDGR O (2263-A), Daxlanden – Mutterstadt BIENWD W (2262-D) | JAO 25/09/2024: 30/09/2024 and 01/10/2024; replaced GOLDGR (11T-D4-D7-00002U) | after 2021 | 32 (10) | meshed | **Reroute to 2021**: restore GOLDGR 11T-D4-D7-00002U |
| Uckermark: Bertikow–Vierraden 481/482, Vierraden–Neuenhagen 483/484 (0109-X, 2030-4, 0110-J, 2031-1) | Enerdata / dpa: 380 kV Uckermarkleitung in operation in Feb 2025. OSM relations 481/482 did not exist in 2021; 483 still `planned=yes` | after 2021 (Feb 2025) | 41 (6) | meshed | Reroute to the 2021 220 kV Vierraden–Neuenhagen lines. Check Pasewalk–Bertikow 407/408 (27/0) as well |
| Pulgar – Vieselbach 589/590 (BBPlG 13) | energate 20/10/2025: "50Hertz puts 380 kV overhead line into operation". It replaced an existing 380 kV line in the same corridor | after 2021 (Oct 2025) as a rebuild; a predecessor existed in 2021 | 12 (5) | meshed | **Keep** the circuits. Restore the 2021 parameters (old conductor, rating) only if they can be found |
| Husum/N – Klixbüll/S (1860-W, 1861-T) | Schleswig-Holstein: section 4 in operation in Nov 2022 | after 2021 (Nov 2022) | 15 (0) | radial cluster towards Endrup | Remove together with Klixbüll – Endrup |
| Klixbüll/S – Endrup 1/2 (10T-DK-DE-00121R/122P) | Schleswig-Holstein: target Mar 2026; commissioning not confirmed | planned / after 2021 | 20 (0) | tie to DK | Remove |
| Perleberg – Stendal/West 516 (380 kV), bus perleberg\|380 is a dead end | 50Hertz / NEP: 380 kV replacement of the 220 kV line Perleberg–Stendal/West–Wolmirstedt, target 2028; EU tender for 485/486 in 09/2026 | planned (in 2021 it was 220 kV) | 5 (0) | radial | **Reroute to 2021**: model it as the 220 kV line. Check Putlitz/Süd – "Perleberg - Putlitz" 596 (12/0) and "Remptendorf - Großschwabhausen" 415 (3/0); both are bridge-type splits with odd bus names |
| Lamspringe – Wahle 3/4 (1694-I, 1695-F), Borken – Mecklar 1/2 | Wahle–Mecklar: northern sections in operation 2022, full line later (ZfK); target was 2024 | after 2021 (likely) | 31 (3) | meshed | Remove after confirming the date. Restore the 2021 220 kV path if JAO/OSM 2021 show one |
| Handewitt – Jardelund blau/grün (0925-Y, 0926-V) | energate 26/08/2020: Dollern–Hamburg–Kassø 380 kV "is already delivering electricity". The OSM relation is newer, but the line is not | before 2021 | 23 (0) | meshed | Keep |
| Hanekenfaehr – Westerkappeln BRAGAU W (2415-E) | OSM 2021: the Bramgau circuit existed; the new EIC comes from the 2025 renumbering | before 2021 | 10 (3) | meshed | Keep |

## Not yet resolved (Investigate step incomplete)
The remaining `high_eic` circuits carry the verdict "unknown". Examples: the Polsum cluster (2479–2484, 2516), Osterath–Rommerskirchen KNESTE O, the FICHTH and UTFORT circuits around Osterath/Edelstahl, the Birkenfeld tee (2488, 2490–2492), the SOONWD and NASSAU tees, TREBUR S, Sottrum–Stade/W, Mehringen–Dollern, Conneforde/O, Etzenricht–Mechlenreuth, and Röhrsdorf–Weida–Remptendorf. Together these 48 circuits carry 522 events (47 unstable).
The next step is to match them against the JAO "Commissioning of new ..." news notices, which are dated and name the EIC (https://www.jao.eu/news/), and against the OSM topology at 2021-06-01 (Overpass `[date:...]`).
The 19 strict empty dead ends are mostly missing 220/380 kV transformers or tee artefacts. The proposal is to send them to class 04 (transformers) and class 02 (tees) rather than remove them. Exception: the Hüffenhardt island `osm-4782-0` (two empty buses, no events) should be removed.

## Effect on the dataset
- **Verified or likely post-2021 assets:** 24 circuits, 237 events (29 unstable), about 2.1 % of the events. Not every row is removed: Pulgar–Vieselbach and Handewitt–Jardelund are kept, so subtract them for the rows actually affected.
- **Unresolved high-EIC circuits:** 48 circuits, 522 events (47 unstable), about 4.7 %.
- **Merzen:** the circuit is a radial stub, so a fault on it only separates Merzen's estimated injections. Its 11 events are all stable, and they look like an artefact. Rerouting it turns the stub into a meshed parallel Westerkappeln–Hanekenfähr path and changes the power flow in the Emsland and Osnabrück area.

## Proposed fix (needs the owner's decision)
1. Build v5 with a "2021 topology" overlay. Use the JAO notices to remove circuits commissioned after 2021-01-01 and to restore the circuits they replaced (GOLDGR, Bramgau Westerkappeln–Hanekenfähr, 220 kV Vierraden–Neuenhagen, 220 kV Perleberg–Stendal/West).
2. Reassign the injections attached to removed buses (Merzen, Klixbüll/S, Bertikow).
3. Run the power flow again, then regenerate the affected events. Events whose fault_line no longer exists cannot be kept.
