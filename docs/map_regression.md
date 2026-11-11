# Staging grid map (8094): why it looks worse than before

Diagnosis only. Staging, the live map (8080) and the model were not changed.
Screenshots are in `/private/tmp/claude-501/-Users-mahmouddraz-code-tranpy/977eaf6e-afac-4d32-b079-8f313c1f1fb9/scratchpad/shots/regression/`
(`<place>_staging.png` = 8094, `<place>_prev.png` = app_prev on 8096, `<place>_lineoff.png` = staging with all three line layers off).
Places: landesbergen, wehrendorf, farge, elsfleth, grohnde.

## What changed between app_prev and staging

`app/app.js` and `app_prev/app.js` are byte-identical (md5 dbf7419e…, the same as the repo). The change is in the **data**:

- `web_prev_round` has **no `osm_lines.geojson`**. The OSM way layer is empty, so `hasWays=false` and the model `lines` layer (one offset stroke per circuit) is the coloured layer. The OSM ways appear only as the grey `towers` casing.
- `web` (staging) has `osm_lines.geojson` with **4915 ways**. Because `hasWays=true`, `lines` is switched off (app.js l.95), and every OSM 220/380 kV way is drawn **in voltage colour**. That includes switchyard, bay, plant-yard and portal ways. Check classes: no_model 2078, match 1521, no_tag 578, model_more 424, model_fewer 314.
- 1580 of those ways are shorter than 100 m: 1030 no_model, 201 model_more, 44 model_fewer. **245 halos sit on ways shorter than 100 m.** These are bay drops of 30–60 m, and the circuit check makes no sense on them.
- `station_footprints` changed too (1.18 → 1.24 MB, grown hulls). At Landesbergen the box is now an irregular hull.

The visible regression is that ways which used to be grey context are now coloured "main" strokes with check halos.

## Layers drawn by default, and the layer panel

Fresh-visitor defaults (app.js l.23): bidding_zones, control_areas, **tower_lines**, **osm_lines**, lines (forced off when osm_lines has data; `?circuits=1` turns it back on), transformers, substations, plants, neighbours. Buses are off.

- **Toggles are not remembered**: there is no localStorage or sessionStorage in app.js. Every reload restores the defaults, so the owner normally sees tower_lines + osm_lines together.
- **What each "on" layer draws**:
  - substations: the station box (fill + outline), the station icon and the tee dot. It draws no busbars; `station_levels` and `station_transformers` are empty (45 B).
  - transformers and buses: symbols and circles only.
  - plants: plant symbols plus `plant-links`, thin grey lines to the substation, shown from zoom 7.
  - neighbours: symbols. bidding_zones and control_areas: dashed borders.
  - `circuit-hl*` has no visibility switch, but it draws nothing until a circuit is selected.
- **Test with all three line layers off** (`landesbergen_lineoff.png`): no tower stubs, busbars or thin lines remain, only the box. So the busbar drawing and the stubs come from `towers` (tower_lines) and `osm-lines`. They do not come from substations, transformers or buses. Plant links are the only other line source, and only near plants (10 of 36 sampled places).
- If the owner saw lines with all three toggles off, the screenshot was most likely taken before the redraw finished or on a different load. On a fresh load all three line toggles are on.

## Screenshot elements

| place | element | layer / feature | regression? |
|---|---|---|---|
| Landesbergen | green/red stubs and offsets in the plant yard NW | `osm-lines`: plant-yard ways 231731580/583/588/589, 1287991727-29, 318805283-288, 27757991/92. Mostly no_model, 30–380 m long | **new** (grey in prev) |
| Landesbergen | thin red diagonal plant→box | `osm-lines` 166750544, 380 kV, no_model, 753 m, faded and thin (0.6 width × 0.6 opacity) | **new** (grey in prev) |
| Landesbergen | 3004 drawn red (380) with purple halo, where prev showed green 220 kV model circuits | `osm-halo` model_more on 231731592 (OSM 380 kV, model 5 circuits) | **new** |
| Landesbergen | orange halo on 3017 / 2008 pieces | `osm-halo` model_fewer (167609849, 318805283) | new |
| Landesbergen | grey busbar comb inside the box | `towers` (all OSM power=line ways inside the station) | old |
| Landesbergen | loose dashed strands at the box (owner's screenshot) | prev behaviour: `lines-straight` dashed model pieces. In the staging render they are gone | old |
| Wehrendorf | two red bundles from N ending in tips | `osm-lines` 4584/4196 ways cut at the box. Round caps on 3 ways that converge at the edge look like arrowheads | new styling |
| Wehrendorf | green 220 kV crossing the red bundle into the box | `osm-lines` 33411474 (220, match) and 33544550 (2432). This is real OSM geometry (the line passes over the 380 kV approach) | not a bug; it looks new because it is coloured |
| Wehrendorf | thin parallel green | `osm-lines` 416584397 (220/110 kV, no_model, faded) and 1335973978 (Diepholz Ost, no_model) | **new** |
| Farge | through bundles pass SE of the box; 3-strand red stub into the box | the junction is a real OSM dead-end/portal about 30–50 m outside the OSM substation polygon (266684854, 13 159 m², a `station table` box, grown 0). The stubs are bay ways 316867863/865/868 (38–73 m; two no_model, one no_tag) | the box is in the right place but too small; growth is not applied to `station table` boxes |
| Farge | loose green end at the bend | `osm-lines` 220 kV 2143/2144 piece ending at the portal; the 220 kV yard is not in the polygon | new |
| Elsfleth/W | grey-brown busbar/bay drawing in the box | `towers` (tower_lines 1066034685-90, 907067145-50) | old |
| Elsfleth/W | 320 drawn thin, "dotted" east then north | `osm-lines` 30534192 (ref 320, 26.8 km, **no_model**, faded and thin). `osm-lines` has no dash. A dotted stroke can only come from `lines-oos` (`?circuits=1`) or from the `towers` casing showing through the faded stroke | new |
| Elsfleth/W | thin red line NE corner → 321 bend | `osm-lines` 907067149/150 (ref 321, 45 m bay ways, no_model and model_more) plus `towers`. It is not a model route and not a plant_connection (no plant links near Elsfleth) | new |
| Elsfleth/W | 322 and upper bundle stop short with purple tips | 43 m bay ways 907067145-150 with model_more halo (purple). The main ways end 72 m from the box | new |
| Y-Huntorf | thin extra green strand parallel SE | a second, faded 220 kV OSM way (178708511 ref 210, model_more, 5.4 km) next to 201 | new |
| Grohnde | orange halos | within 5 km: 1 model_fewer, 0 model_more (3 no_model faded, 3 match). Germany-wide: 314 model_fewer, 424 model_more | — |

**Removed circuits 321/328.** The model 321/328 is **Parchim Süd – Güstrow (220 kV, 50Hertz)**. It is in lines.geojson with status decommissioned/out_of_service and route_status not_in_network, so it is drawn only by `lines-oos`, and that layer is hidden by default. The "321" and "320" near Elsfleth are **OSM refs of TenneT tower lines**, not model circuits. No removed circuit is drawn as a model line.

## Root causes, counted on 36 places

The sample is the 19 owner places and the random places from `docs/map_panel.md`, plus Marzahn, Gransee, Farge, Wehrendorf, Grohnde, Elsfleth and Y-Huntorf. Table: `cairo-lab:~/reg_sample.csv`.

| # | root cause | places affected |
|---|---|---|
| R1 | short no_model OSM ways (switchyard/bay/plant yard) drawn coloured within 1 km of the box | **29 / 36** |
| R2 | way ends 5–400 m outside the box (lines stop short or dangle; box does not reach the portal) | **36 / 36**. Station-table boxes have grown_m = 0, so they never grow to the line ends. Typical gap 30–75 m |
| R3 | check halos on bay ways under 100 m (purple/orange tips) | 245 ways Germany-wide; halos at 24 / 36 places |
| R4 | two line layers on by default (grey `towers` under coloured `osm-lines`; busbar comb) | 36 / 36 at zoom ≥ 9 |
| R5 | 220 kV way crossing a 380 kV way near the station | 11 / 36. Mostly real geometry; drawing order makes it noisy |
| R6 | plant_connections near the box | 10 / 36. Minor; not seen in the owner's screenshots |
| R7 | round caps of converging ways at the cut look like arrows | wherever ≥ 2 ways meet the box edge (most 380 kV stations) |
| R8 | offset artefacts at bends | not reproduced. Staging draws OSM ways unoffset; the triple bundles are separate OSM ways per circuit. Prev's model `lines` had offset bends |

## Fixes, ranked by impact

1. **Do not colour ways that belong to the station.** In `export_grid_gis.py`/`grid_gis.py`, drop from `osm_lines` any way that lies inside the station or plant footprint (buffered 100 m), or that is shorter than about 150 m and touches a footprint, and has no model circuit. Keep those ways in `tower_lines` only. Fixes R1 and R3 and most of what the owner sees at all four places. Risk: low. A real short tie line (for example a plant 220 kV link) would lose its colour, so whitelist ways with model_n > 0.
2. **Turn the halo check off for short ways.** Set `check = "station"` (no halo) for ways under 150 m. Inherit the check from the parent corridor instead. Risk: low.
3. **One line layer on by default.** Turn `tower_lines` (grey) off by default, or show it only above zoom 14 as context. Default set for the approved design: osm_lines + substations (plain box) + plants (without plant-links below zoom 12) + borders. Turn transformers, buses, tower_lines and lines off. Removes R4, the busbar comb and the grey/coloured double strokes. Risk: none in the data. Users lose station-internal context unless they turn it on.
4. **Make lines reach the box.** Grow `station table` boxes too (hull of footprint + the line end points within 150 m + 30 m, 300 m cap, as for the other boxes), or snap the last vertex of each way to the box border. Fixes R2 at Farge and Elsfleth and the dangling green ends. Risk: medium. Boxes can swallow neighbouring stations or plants, so keep the 300 m cap and the distance to other boxes.
5. **Butt caps at the box.** Use `"line-cap": "butt"` on `osm-lines` and `osm-halo`, or cut with butt and keep round joins. Fixes R7 (arrow tips). Risk: none.
6. **Plant yards (Landesbergen).** When a model circuit terminates in the plant switchyard, add that yard to the station box as a second polygon (MultiPolygon), or label it as part of the station. Risk: medium; this needs the plant footprint from OSM `power=plant`.
7. Lower-priority items:
   - Draw 380 kV above 220 kV consistently (`line-sort-key` already does this) so crossings (R5) read cleanly. Risk: none.
   - Raise `plant-links` minzoom to 12 (R6). Risk: none.

Fixes 1, 3 and 5 alone remove almost every artefact in the owner's five screenshots without touching the model.
