# 07 Default lumped infeed sites (PLAN I2): detection only

Detection only: nothing in the model, map or code was changed.
Model: `model_v5_3e.pkl` (8be55949). Inputs: `grid_de_connection_infeed`, `stations/voltage_levels.parquet` (polygon match), `~/data/grid/osm/DEU.gpkg` (`power_transformer` inside the matched polygon), and the gate run `pf_v5/g8be55949` (inputs of the 9 hours).
Script: `cairo-lab:~/data/grid/v5/step4/i2.py`. Full table: `occurrences.csv`.

A **default site** is an infeed row with `count_evidence` "default: 2 units, the station has no counted unit": 2 × 380(220)/110 kV library units, about 700 MVA. The connection model hangs a lumped 110 kV bus with regional load and embedded generation on it.

**Effect** is the MW of load and of embedded generation placed on the site's lumped 110 kV bus, summed over the 18 gate runs. That is 9 hours × 2 tan φ, and the dispatch is identical for both tan φ.

## Summary

The model has 55 default sites; they carry 120,065 MW of load and 74,809 MW of generation over 18 runs.

| Class | Sites | Load MW (18 runs) | Generation MW (18 runs) |
|---|---|---|---|
| invented: junction node | 2 | 4,576 | 1,596 |
| OSM polygon without any transformer (unconfirmed) | 17 | 39,450 | 24,067 |
| real station: OSM transformer with a 110 kV side | 11 | 19,256 | 13,177 |
| real station: OSM transformers without a 110 kV side (unconfirmed) | 4 | 10,323 | 6,226 |
| real station: OSM transformers, voltage untagged | 21 | 46,459 | 29,742 |

No default site uses a JAO 110 kV transformer: JAO r9 publishes only EHV couplers. The "real station" verdicts therefore rest on OSM alone.

Two invented sites were already removed in step 4: `herbentingen\|380` (A1) and `frankfurt\|380` (06 #2). Removing `frankfurt` lowered the total losses by 70 MW over 18 runs.

## Invented nodes: proposed fixes (not applied)

| Site | Node | Why invented | Effect (18 runs) | Proposed fix |
|---|---|---|---|---|
| y-bellendorf#polsum s | `y-bellendorf#polsum s\|220` | A tee point on the route (two legs: Bergmannsglück 1.36 km, Kusenhorst 10.49 km), not a station | load 4,422 MW, generation 1,343 MW | Remove the infeed row. The connection model then reassigns its catchment to the nearest real sites (Bergmannsglück, Kusenhorst), as happened for herbentingen in A1. Gate it as its own change |
| osm-4706 | `osm-4706\|220` | The OSM branch point of the Remptendorf 220 kV lines (A4), not a station | load 154 MW, generation 253 MW | Remove the infeed row (catchment to Remptendorf / Hohenwarte II / Stahlwerk) |

## Needs review (not invented, but no 110 kV evidence)

- **17 polygons without any OSM transformer.** Examples: Emscherbruch, Charlottenburg, Krümmel 50H, Altbach, Heilbronn, Elsfleth/West, Isar, Fedderwarden, Waldeck, Datteln. Several are switching stations or plant/converter switchyards: "Schaltwerk Krümmel (50 Hertz)", "Schaltanlage Elsfleth/West", "Konverter Station Fedderwarden", "Schaltanlage Datteln 4". A switchyard of that kind normally has no 380/110 kV transformation. If OpenInfraMap or the TSO confirms there is none, remove the infeed row. Emscherbruch alone carries 7,669 MW of load over 18 runs.
- **4 polygons whose transformers have no 110 kV side:** Lippendorf Block R (27/380 kV, a generator step-up), KW Weiher (20/220 kV, step-up), Ratsbusch, and Mecklar (380/138 kV?). The step-up transformers are not 110 kV infeed. Lippendorf R and KW Weiher are candidates for removal.
- **21 polygons with untagged transformers:** the transformer count is known but the voltage is not. Check them on OpenInfraMap as done for the "owner_checked" rows, then replace the default 2 units by the counted units.
- Industrial or converter sites among them (Edelstahl Krefeld, Walzwerk, Stahlwerk Thüringen, Philippsburg DC) probably feed an industrial load rather than a regional 110 kV grid. The regional allocation there is an assumption.

## All default sites

Columns `osm_tr` give: all transformers / with a 110 kV side / untagged, inside the matched polygon.

| Site | Node | Kind | OSM polygon | osm_tr | Load MW | Gen MW | Class |
|---|---|---|---|---|---|---|---|
| emscherbruch | `emscherbruch\|380` | substation | Emscherbruch (1076001174.0) | 0 / 0 / 0 | 7,669 | 891 | OSM polygon without transformers (unconfirmed) |
| charlottenburg | `charlottenburg\|380` | substation | Umspannwerk Charlottenburg (46615737.0) | 0 / 0 / 0 | 4,884 | 884 | OSM polygon without transformers (unconfirmed) |
| krummel 50h | `krummel 50h\|380` | substation | Schaltwerk Krümmel (50 Hertz) (1089911133.0) | 0 / 0 / 0 | 4,804 | 1,301 | OSM polygon without transformers (unconfirmed) (switchyard name) |
| altbach | `altbach\|380` | substation | Umspannwerk Altbach 1 (37721677.0) | 0 / 0 / 0 | 3,165 | 1,097 | OSM polygon without transformers (unconfirmed) |
| heilbronn | `heilbronn\|380` | substation | Heilbronn (1104947187.0) | 0 / 0 / 0 | 3,165 | 1,355 | OSM polygon without transformers (unconfirmed) |
| elsfleth/west | `elsfleth/west\|380` | substation | Schaltanlage Elsfleth/West (829116805.0) | 0 / 0 / 0 | 2,691 | 1,920 | OSM polygon without transformers (unconfirmed) (switchyard name) |
| osm/1217609297 | `osm/1217609297\|380` | substation | Umspannwerk Heyden (1217609297.0) | 0 / 0 / 0 | 2,397 | 1,494 | OSM polygon without transformers (unconfirmed) |
| isar | `isar\|380` | substation | Isar (77697810.0) | 0 / 0 / 0 | 1,872 | 3,000 | OSM polygon without transformers (unconfirmed) |
| fedderwarden | `fedderwarden\|380` | substation | Konverter Station Fedderwarden (31024391.0) | 0 / 0 / 0 | 1,854 | 4,641 | OSM polygon without transformers (unconfirmed) (switchyard name) |
| waldeck | `waldeck\|380` | substation | Waldeck (41072584.0) | 0 / 0 / 0 | 1,747 | 1,599 | OSM polygon without transformers (unconfirmed) |
| laichingen | `laichingen\|380` | substation | Umspannwerk Laichingen (26404600.0) | 0 / 0 / 0 | 1,526 | 1,352 | OSM polygon without transformers (unconfirmed) |
| rotensohl | `rotensohl\|380` | substation | Umspannwerk Rotensohl (38325076.0) | 0 / 0 / 0 | 1,457 | 1,301 | OSM polygon without transformers (unconfirmed) |
| osm/58429825 | `osm/58429825\|380` | substation | - (58429825.0) | 0 / 0 / 0 | 680 | 863 | OSM polygon without transformers (unconfirmed) |
| conneforde | `conneforde\|220` | substation | Conneforde 220/110 kV (1280347702.0) | 0 / 0 / 0 | 587 | 925 | OSM polygon without transformers (unconfirmed) |
| datteln | `datteln\|380` | substation | Schaltanlage Datteln 4 (275138524.0) | 0 / 0 / 0 | 429 | 879 | OSM polygon without transformers (unconfirmed) (switchyard name) |
| remptendorf | `remptendorf\|220` | busbar | Umspannwerk Remptendorf (1086825107.0) | 0 / 0 / 0 | 270 | 321 | OSM polygon without transformers (unconfirmed) |
| maade | `maade\|220` | substation | Umspannwerk Maade (23744346.0) | 0 / 0 / 0 | 252 | 244 | OSM polygon without transformers (unconfirmed) |
| y-bellendorf#polsum s | `y-bellendorf#polsum s\|220` | tee | - (-) |  /  /  | 4,422 | 1,343 | invented: junction node |
| osm-4706 | `osm-4706\|220` | junction | - (-) |  /  /  | 154 | 253 | invented: junction node |
| zolling | `zolling\|220` | substation | Umspannwerk Zolling (1568291190.0) | 2 / 2 / 0 | 3,506 | 2,072 | real station: OSM 110 kV transformer |
| wustermark | `wustermark\|380` | substation | Umspannwerk Wustermark (27149550.0) | 6 / 3 / 0 | 3,285 | 3,599 | real station: OSM 110 kV transformer |
| eichstetten c2 | `eichstetten c2\|380` | substation | 380-kV-Umspannwerk Eichstetten (242756113.0) | 6 / 3 / 0 | 3,165 | 1,032 | real station: OSM 110 kV transformer |
| grossgartach | `grossgartach\|380` | substation | Umspannwerk Großgartach TransnetBW 380kV (1327333163.0) | 4 / 4 / 0 | 2,006 | 1,633 | real station: OSM 110 kV transformer |
| schworstadt | `schworstadt\|220` | substation | Schwörstadt (88901626.0) | 2 / 2 / 0 | 1,835 | 480 | real station: OSM 110 kV transformer |
| weier | `weier\|220` | substation | Umspannwerk Weier (933248930.0) | 4 / 4 / 0 | 1,830 | 763 | real station: OSM 110 kV transformer |
| audorf | `audorf\|220` | substation | Audorf Süd (35176751.0) | 4 / 1 / 0 | 1,659 | 1,848 | real station: OSM 110 kV transformer |
| grossgartach 2 | `grossgartach 2\|380` | substation | Umspannwerk Großgartach TransnetBW 380kV (1327333163.0) | 4 / 4 / 0 | 1,269 | 1,144 | real station: OSM 110 kV transformer |
| osm/188276460 | `osm/188276460\|220` | substation | - (188276460.0) | 2 / 2 / 0 | 365 | 511 | real station: OSM 110 kV transformer |
| rdk7 | `rdk7\|380` | substation | Karlsruhe-Daxlanden (152261133.0) | 8 / 2 / 0 | 229 | 86 | real station: OSM 110 kV transformer |
| rdk8 | `rdk8\|380` | substation | Karlsruhe-Daxlanden (152261133.0) | 8 / 2 / 0 | 108 | 10 | real station: OSM 110 kV transformer |
| osm/714098678 | `osm/714098678\|380` | substation | Umspannstation Lippendorf Block R (714098678.0) | 2 / 0 / 0 | 3,739 | 1,953 | real station: OSM transformers without 110 kV side (unconfirmed) (switchyard name) |
| osm/207756597 | `osm/207756597\|220` | substation | KW Weiher (207756597.0) | 1 / 0 / 0 | 2,319 | 660 | real station: OSM transformers without 110 kV side (unconfirmed) (switchyard name) |
| ratsbusch | `ratsbusch\|220` | substation | Station Ratsbusch (24858617.0) | 2 / 0 / 0 | 2,250 | 345 | real station: OSM transformers without 110 kV side (unconfirmed) |
| mecklar | `mecklar\|380` | substation | Umspannwerk Mecklar 380kV (29309786.0) | 2 / 0 / 0 | 2,016 | 3,268 | real station: OSM transformers without 110 kV side (unconfirmed) |
| grafenrheinfeld | `grafenrheinfeld\|380` | substation | Grafenrheinfeld (108797675.0) | 2 / 0 / 2 | 4,801 | 4,909 | real station: OSM transformers, voltage untagged |
| dornigheim | `dornigheim\|220` | substation | Dörnigheim 220 kV (23881272.0) | 2 / 0 / 2 | 4,155 | 505 | real station: OSM transformers, voltage untagged |
| edelstahl | `edelstahl\|220` | substation | Station Edelstahl Krefeld (93278553.0) | 4 / 0 / 4 | 3,662 | 1,842 | real station: OSM transformers, voltage untagged (switchyard name) |
| philippsburg | `philippsburg\|380` | substation | Gleichstrom-Umspannwerk Philippsburg (691662101.0) | 13 / 0 / 1 | 3,165 | 733 | real station: OSM transformers, voltage untagged (switchyard name) |
| endersbach | `endersbach\|380` | substation | 380 kV-Umspannwerk Endersbach (TransnetBW GmbH) (48428875.0) | 3 / 0 / 3 | 3,165 | 1,328 | real station: OSM transformers, voltage untagged |
| unterweser | `unterweser\|380` | substation | Unterweser (32076853.0) | 3 / 0 / 3 | 2,729 | 4,016 | real station: OSM transformers, voltage untagged |
| osm/97857512 | `osm/97857512\|220` | substation | Umspannwerk Walzwerk (UWWW) (97857512.0) | 8 / 0 / 8 | 2,688 | 2,738 | real station: OSM transformers, voltage untagged (switchyard name) |
| spellen | `spellen\|220` | substation | Spellen (24462317.0) | 7 / 0 / 7 | 2,420 | 1,150 | real station: OSM transformers, voltage untagged |
| bandstahl | `bandstahl\|220` | substation | - (1447099504.0) | 3 / 0 / 3 | 2,380 | 1,165 | real station: OSM transformers, voltage untagged |
| bergmannsgluck | `bergmannsgluck\|220` | substation | Station Bergmannsglück (22944791.0) | 3 / 0 / 3 | 2,094 | 794 | real station: OSM transformers, voltage untagged |
| osm/152219167 | `osm/152219167\|220` | substation | Stahlwerk Thüringen (152219167.0) | 2 / 0 / 2 | 2,088 | 1,456 | real station: OSM transformers, voltage untagged (switchyard name) |
| marbach | `marbach\|220` | substation | Umspannwerk Marbach (97825259.0) | 3 / 0 / 3 | 1,835 | 610 | real station: OSM transformers, voltage untagged |
| oberwald | `oberwald\|220` | substation | Umspannwerk Oberwald (32725957.0) | 5 / 0 / 5 | 1,835 | 251 | real station: OSM transformers, voltage untagged |
| krun | `krun\|220` | substation | Umspannwerk Krün (68532663.0) | 2 / 0 / 1 | 1,794 | 624 | real station: OSM transformers, voltage untagged |
| dc phillipsburg | `dc phillipsburg\|380` | substation | Gleichstrom-Umspannwerk Philippsburg (691662101.0) | 13 / 0 / 1 | 1,581 | 510 | real station: OSM transformers, voltage untagged (switchyard name) |
| niederstotzingen | `niederstotzingen\|380` | substation | Umspannwerk Niederstotzingen (31105067.0) | 5 / 0 / 5 | 1,449 | 1,241 | real station: OSM transformers, voltage untagged |
| huffenhardt | `huffenhardt\|380` | substation | Umspannwerk Hüffenhardt (143903369.0) | 3 / 0 / 3 | 1,411 | 1,312 | real station: OSM transformers, voltage untagged |
| farge | `farge\|380` | substation | Farge (266684854.0) | 1 / 0 / 1 | 1,228 | 1,240 | real station: OSM transformers, voltage untagged |
| goldshofe | `goldshofe\|380` | substation | Umspannwerk Goldshöfe (42050342.0) | 2 / 0 / 2 | 984 | 1,153 | real station: OSM transformers, voltage untagged |
| rommersheim | `rommersheim\|380` | substation | Umspannwerk Rommersheim (1024536785.0) | 2 / 0 / 2 | 738 | 1,889 | real station: OSM transformers, voltage untagged |
| mehrum/nord | `mehrum/nord\|220` | substation | Mehrum-Nord (1011084707.0) | 2 / 0 / 2 | 258 | 275 | real station: OSM transformers, voltage untagged |

## Removal runs (owner approved 2026-11-08): both reverted, and an ordering artefact found

The two invented-node removals were gated on top of `model_v5_3e.pkl`. Both FAILED with the same signature:

| Change | Model | Overloaded rows | Lowest voltage | Losses (sy, 18 runs) | Lines > 0.10 |
|---|---|---|---|---|---|
| y-bellendorf#polsum s removed | 1c3eb028 | 6 -> 7 (Y-Neurott (-GKM - Weinheim) rt 1.004) | 0.916 -> 0.914 | +78 MW | BLUDNZ W +0.22, Y-Gelsenberg GLADBK W +0.15, Y-Schwelgern WALSUM S +0.12, Gütersloh–Hesseln +0.11, Y-Bergmannsglück POLSUM S +0.10, Neurott–Wiesloch +0.10 |
| osm-4706 removed | f22a0ac6 | 6 -> 7 | 0.916 -> 0.914 | +72 MW | BLUDNZ W +0.22, Gütersloh–Hesseln +0.10, Neurott–Wiesloch +0.10 |

**Cause: the result depends on the row order of `grid_de_connection_infeed`.**
- Diagnostic `dorder`: model 3e with the osm-4706 row only moved to the end of the table, nothing removed. It gives the same signature (overloaded rows 7, lowest voltage 0.914, losses 23,807 MW).
- In that run, reordering alone moves up to 245 MW of load at 105 lumped buses (high_load).
- Root cause (proposed I3, not fixed, needs approval): in `src/tranpy/grid_connections.py`, `connect` sorts the sites with an unstable `sort_values("kv_hv")` (line ~891). The greedy `spill_over_cap` then processes the sites in that order. Removing one row can permute sites that share a kv value, so the capped-load spill lands elsewhere.
- Proposed fix: `sort_values(["kv_hv", "site"], ascending=[False, True], kind="stable")`, or sort `here` by site before building the KD-tree. Gate that code change on its own first, then redo these two removals.

Compared against the reordered reference (same order, osm-4706 row only removed), the **intended** effect of the osm-4706 removal is small and local. In high_load its lumped load (−11.2 MW) moves to Remptendorf - Großschwabhausen (+11.2 MW). Smaller shifts (≤ 16 MW) appear among 50Hertz sites: Charlottenburg, Krümmel 50H, Stahlwerk Thüringen, Altenfeld, Hamburg/Ost and others.

For y-bellendorf#polsum s, the high_load load moved mostly to Gelsenberg (+154 MW), Bergmannsglück (+91 MW) and Karnap (+59 MW). That run is mixed with the ordering artefact; Herbertingen Amprion, for example, gained +208 MW. Both models are kept in `step4/reverted/`; the y-bellendorf one was overwritten, but `fix.py b7` reproduces it. Neither change is in the chain.

**Consequence for earlier accepted changes.** Three accepted changes removed infeed rows: A1 (herbentingen), 06 #2 (frankfurt) and 06 #4 (perleberg default). Part of their gate deltas may be this artefact. One example: the +/−0.216 BLUDNZ W signature appears in 06 #2 and in both runs above. These should be re-gated after I3.

## Review of the 17 default sites on OSM polygons with no transformer (detection only)

Evidence used (script `step4/i2b.py`, table `step4/i2b_review.csv` on cairo-lab):
- the polygon's `voltages` and `substation_type` tags;
- OSM 110 kV lines touching the polygon (30 m);
- polygons within 1.5 km;
- OSM transformers within 1.5 km;
- the infeed rows of neighbouring model sites.

MaStR was not available offline. The effect is the load / generation MW on the site's lumped bus, summed over 18 runs.

| Site | OSM evidence | Verdict | Proposed fix | Load / Gen MW (18 runs) |
|---|---|---|---|---|
| emscherbruch | 380 kV polygon, untagged type; 5 × 110 kV lines touch it; "Herne" 380/220/110 kV polygon 682 m away | unclear, leaning real (380/110 kV in the Herne/Emscherbruch area) | count units on OpenInfraMap; keep until then | 7,669 / 891 |
| charlottenburg | tagged 380;110 kV, transmission (50Hertz); 110 kV line; 110 kV stations around | real station feeding 110 kV | replace the default 2 units by the counted units | 4,884 / 884 |
| krummel 50h | "Schaltwerk Krümmel (50 Hertz)" 380 kV; the model's `krummel` site (owner_checked) says "50Hertz Schaltwerk is switching only" | switching station: infeed wrong | remove the infeed row (catchment to Krümmel TenneT / Grüner Jäger) | 4,804 / 1,301 |
| altbach | tagged 380;110 kV (Netze BW); Altbach 2 [110] 346 m | real | count units | 3,165 / 1,097 |
| heilbronn | 380 kV, `substation_type=generation` (plant switchyard); "Umspannwerk Heilbronn" [110] 273 m away | unclear (plant switchyard, 110 kV station separate) | check the TransnetBW 380/110 kV coupling at Heilbronn; remove if only the plant | 3,165 / 1,355 |
| elsfleth/west | "Schaltanlage Elsfleth/West" 380 kV only; Bardenfleth [110] 37 m with a 110/20 kV transformer | unclear (switching station next to a 110 kV station) | check OpenInfraMap for 380/110 kV units | 2,691 / 1,920 |
| osm/1217609297 (Heyden) | "Umspannwerk Heyden" 380 kV, type generation, next to "KW Heyden"; one 110 kV line; transformers "/110" nearby | unclear | check | 2,397 / 1,494 |
| isar | 380 kV plant switchyard (Isar NPP), no 110 kV line; converter Isar 958 m | switchyard: infeed wrong (likely) | remove after one check | 1,872 / 3,000 |
| fedderwarden | "Konverter Station Fedderwarden", type converter, 380 kV; Fedderwarden 380/220 kV next to it | converter / switchyard: infeed wrong | remove the infeed row | 1,854 / 4,641 |
| waldeck | 380 kV (TenneT; pumped-storage switchyard); "Waldeck" [110] 26 m; 3 × 110 kV lines | unclear, leaning real | count units | 1,747 / 1,599 |
| laichingen | tagged 380;110 kV, 14 × 110 kV lines | real | count units | 1,526 / 1,352 |
| rotensohl | tagged 380;110 kV, 8 × 110 kV lines | real | count units | 1,457 / 1,301 |
| osm/58429825 | unnamed 380 kV polygon (50Hertz area), 110 kV polygons 300–760 m | unclear | identify the station | 680 / 863 |
| conneforde (220) | "Conneforde 220/110 kV", 26 × 110 kV lines; OSM transformers nearby are 380/110 and 380/220 kV; conneforde/ost 380 already has 3 units | unclear (220/110 kV vs 380/110 kV counted twice) | check whether Conneforde has 220/110 kV units | 587 / 925 |
| datteln | "Schaltanlage Datteln 4" (Uniper), plant switchyard; transformers 380/3.6 kV (station service) | switchyard: infeed wrong | remove the infeed row | 429 / 879 |
| remptendorf (220) | the 220 kV level of Remptendorf; OSM transformers are 380/110 kV ×2 and 380/220 kV ×2; the 380 kV site already has 2 export units | duplicate: infeed wrong on the 220 kV node | remove the infeed row | 270 / 321 |
| maade | "Umspannwerk Maade" 220 kV (TenneT); one 110 kV line; 110 kV stations nearby | real, leaning (220/110 kV) | count units | 252 / 244 |

Summary:
- **Infeed wrong (switching station, plant or converter switchyard, or duplicate):** Krümmel 50H, Fedderwarden, Datteln, Remptendorf 220. Isar is likely wrong but needs one check. Together these carry 9,229 MW of load and 10,142 MW of generation over 18 runs.
- **Real:** Charlottenburg, Altbach, Laichingen, Rotensohl and Maade. Fix them by counting units on OpenInfraMap.
- **Unclear:** Emscherbruch, Heilbronn, Elsfleth/West, Heyden, Waldeck, osm/58429825, Conneforde 220.
- Every removal waits for the I3 ordering fix, so that its gate measures the change itself.
