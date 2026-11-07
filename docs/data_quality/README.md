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
| 03 | Circuit count per tower line vs model | OSM line 4132: 380/380/220/110/30 kV circuits | queued (after the tower-line layer) |
| 04 | Missing or estimated parameters | Bürs–Meiningen/Walgau/Westtirol length 0; NSACHS N reactance estimated | queued |
| 05 | Substation locations | schematic points (Würgassen, …), snapped clusters (Meitingen, …) | queued |

## Visualization track (parallel, display only)
Tower-line layer from OSM ways (voltages, circuits, ref, operator; popup lists carried
circuits) and model circuits drawn side by side along the tower lines they share, assigned via
each circuit's OSM route relation.
