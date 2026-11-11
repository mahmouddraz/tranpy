# Grid data quality: process and issue queue

Owner decision 2026-11-07. Problems in the German 220/380 kV model (frozen `model_v4final`) are
handled **one problem class at a time, one agent per class**, in four steps. Nothing in the
frozen model changes without the owner's decision; approved fixes go into the v5 model build.

## Process per problem class
1. **Detect** — list every occurrence in the model (`issues/<NN>-<class>/occurrences.csv`).
2. **Investigate** — evidence per occurrence: JAO static grid model, TSO project pages, BBPlG /
   EnLAG lists, BNetzA monitoring, OSM ways/relations (incl. Overpass outside Germany), published
   lengths/reactances. Rules: published values first; OSM construction tags may be stale
   (verify with JAO/TSO); no requests to operators; change only with demonstrable gain.
3. **Propose** — `issues/<NN>-<class>/findings.md`: one row per occurrence with evidence (URL,
   quote, date), effect on the dataset (power flow, number of v4.1 events on affected lines),
   and a proposed fix. **Stop here for the owner's decision.**
4. **Fix** — only approved rows; implemented as a documented change for v5 (or as a display
   overlay when geometry-only), regenerated and verified (power flow, map, tests).

## Queue
| # | Class | Example | Status |
|---|---|---|---|
| 01 | Future assets (commissioned after 2021) and dead-end busbars | Merzen 380 kV (Amprion: planned, full commissioning 2028, BBPlG project 6) | investigating |
| 02 | Tee topology inconsistent with real routes | BLUDNZ W / DELLM O at Bürs (`~/data/grid/bursa_corridor_findings.md`) | findings ready; owner decision pending (option A recommended) |
| 03 | Circuit count per tower line vs model | OSM line 4132: 380/380/220/110/30 kV circuits | findings done; display fixes 1–2 applied 2026-11-08 (61 export circuits drawn on their section, straight circuits 71 → 15, circuit length on tower lines 93.0 → 95.1 %; mismatch rule `circuits_max`, < 0.5 km skipped, `neighbour_pickup`: flags 9,703 → 1,190); model proposals 3–4 wait for the owner (v5) |
| 04 | Missing or estimated parameters | Bürs–Meiningen/Walgau/Westtirol length 0; NSACHS N reactance estimated | findings ready (143 rows: 45 fix proposed, 86 keep, 12 unresolved); owner decision pending (HERBTG W tee, 321 status, Bürs tie convention) |
| 05 | Substation locations | schematic points (Würgassen, …), snapped clusters (Meitingen, …) | findings ready 2026-11-08 (26 occurrences, X = 500 m): 9 model/topology (worst: Herbertingen–Hoheneck–Metzingen tee, 1 km stub modelled as 61.8 km), 17 display only; owner decision pending |

## Visualization track (parallel, display only)
Tower-line layer from OSM ways (voltages, circuits, ref, operator; popup lists carried
circuits) and model circuits drawn side by side along the tower lines they share, assigned via
each circuit's OSM route relation.
