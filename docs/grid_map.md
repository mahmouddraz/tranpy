# Grid map: GeoPackage, QGIS project and web viewer

A GIS view of the frozen grid model (v4final, 380/220 kV): one GeoPackage as the single
source, a styled QGIS project (main view) and a light MapLibre web viewer with the same
cartography. The model is only read.

## Export

```bash
# on cairo-lab (needs the [grid] extra: geopandas, pyogrio; and pyarrow for the pickle)
python scripts/export_grid_gis.py ~/data/grid/v5/model_v5_3e.pkl ~/data/grid \
    --web ~/data/grid/web \
    --overlay ~/data/grid/geometry_overlay.parquet \
    --towers ~/data/grid/osm/hv_lines.parquet --towers-overpass ~/data/grid/osm/overpass_at_west.json \
    --stations ~/data/grid/osm/DEU.gpkg --diagrams
# QGIS styles/project, SVG symbols and style.js (no model needed)
python scripts/export_grid_gis.py --app apps/grid_map
```

Since 2026-11-09 (staging on port 8094, not yet published) the circuits are drawn on the
OpenStreetMap graph:

```bash
S=~/data/grid/staging_osmgraph
python -m tranpy.grid_network --out ~/data/grid/v5/network          # graph + routes (agent A)
python scripts/export_grid_gis.py ~/data/grid/v5/model_v5_4a.pkl $S/out --web $S/web \
    --overlay ~/data/grid/geometry_overlay.parquet \
    --towers ~/data/grid/osm/hv_lines.parquet --towers-overpass ~/data/grid/osm/overpass_at_west.json \
    --stations ~/data/grid/v5/stations/stations.gpkg --station-table ~/data/grid/v5/stations \
    --network ~/data/grid/v5/network --diagrams
python scripts/map_acceptance.py --export $S/out/grid_model.gpkg \
    --model ~/data/grid/v5/model_v5_4a.pkl --network ~/data/grid/v5/network --out $S/acceptance
python scripts/map_visual_check.py --url http://localhost:8094/ --export $S/web \
    --events $S/acceptance/events.csv --out ~/data/grid/acceptance/visual_staging_v2
# after the owner's OK: staging -> 8080 + QWC 8088, rollback copies *_before_osmgraph
bash scripts/publish_map.sh            # --dry-run shows the copies, --rollback undoes it
```

`--network` takes the `grid_network` output; `--stations` must then be the station table
(`stations.gpkg`), the footprints the graph was built with. `--station-table` adds the OSM
names and checks to the station panel (`web/station_details.json`). The export also writes
`station_gaps.csv` next to the GeoPackage (see below).

### Geometry on the OSM graph

OpenStreetMap is the geometry backbone, the model the electrical truth. Display only; no
model file is written. `grid_gis.network_lines`, in this order:

1. **Pieces.** A circuit with `circuit_pieces` (status `routed`, or the partial route of a
   `no_path_strict` circuit) is drawn as its pieces (one OSM way, `join` or `connector`
   each) with the exact OSM geometry. `grid_bundles.piece_slots` gives every piece
   `circuit_no` (slot) and `circuits` (n); the web map offsets it on screen by
   `(slot - (n - 1) / 2) * gap_px(zoom)` (QGIS: 0.8 mm per slot). Nothing is baked into the
   geometry; there are no station fans. A break inside a partial route is a dashed `break`
   connector; the found pieces stay solid. A strict detour (path longer than
   `DETOUR_RATIO` = 1.5 x the circuit and 10 km beyond it) gets no pieces since issue 14: it
   is drawn dashed on its previous geometry, not on the foreign ways of the long way round.
2. **Final station shapes first.** The route ends of each station's own circuits (after
   removing what lies inside the OSM footprint) grow its box: convex hull of footprint and
   ends, plus 30 m (`BOX_MARGIN_M`). An end more than 300 m (`BOX_REACH_M`) outside the
   footprint does not grow it: it gets a dashed `end` connector to the box and a row in
   `station_gaps.csv` ("model–OSM deviation for review": station, circuit, end, distance,
   `within_cap`). The grown polygon is the `station_footprints` geometry that is drawn
   (`grown_m`; the OSM footprint in `footprint_wkt`). Junction dots move to the OSM node
   most of their legs attach to (`junction_snap_m`).
3. **Strict cut.** Every circuit is cut at its own station's final shape, and every part of
   any circuit inside any drawn station shape is removed (a through line keeps its two outer
   parts). Nothing is drawn inside a station; the box and its panel (single-line diagram,
   component tables) show the inside. An end that a neighbouring shape cut away from its own
   target gets a dashed `end` connector. Circuits with nothing outside the shapes (28
   in-station links, e.g. Conneforde – Conneforde/O) keep one row without geometry and are
   named in the metadata `inside_station_only`.
4. **No pieces** (`no_terminal`, `no_path_strict` without a partial route, circuits missing
   from the network table): the previous geometry chain, `route_quality = unrouted`,
   dashed and centred, with `route_note` (status, break location and gap, override reason)
   shown on top of the popup.

The metadata names the model file and sha256 (`source_file`, `source_sha256`) and the
network folder with its counts (`network`).

### Main layer: one line per OSM tower line (2026-11-09 redesign)

With `--network` the export writes `osm_lines` (`grid_gis.osm_way_lines`): one feature per
OSM 220/380 kV way of the graph, the exact way geometry cut at the drawn station shapes,
coloured by voltage, no per-circuit offsets. Per way: `way_id`, `name`, `ref`, `kv`,
`circuits_tag` (OSM), `operator`, `model_n` and `model_circuits` (JSON list of `mrid`,
`name`, `eic`, `kv` of the model circuits drawn on it) and `check`:

| check | meaning | drawn |
|---|---|---|
| `match` | model and OSM agree (`edge_check.csv`) | normal |
| `model_fewer` / `model_more` | model carries fewer / more circuits than OSM on an edge of the way | orange / purple halo |
| `no_model` | no model circuit on the way | thin, faded |
| `no_tag` | OSM circuit count not tagged | neutral |

The web map shows `osm_lines` as the main layer; the per-circuit `lines` start hidden
("Circuits (model, all)" in the layer list, or `?circuits=1`). Clicking a circuit name (in a
tower-line popup, the station panel or the search) highlights that circuit station to
station as a thick overlay along its pieces (dashed where it is not on the OSM graph) and
opens its popup (R, X, B, rating).

`parallel_way_findings.csv` (next to the GeoPackage, `grid_gis.parallel_way_findings`) lists
model circuits routed on one of two parallel ways of the same voltage (< 150 m apart over
≥ 1 km): `count_shift` (way A: model > OSM; way B: model 0, OSM > 0) and `ref_on_parallel`
(the circuit's number is in way B's ref, not in A's), e.g. Audorf/S – Schuby/W
schwarz/weiß and Audorf/S – Handewitt on the way next to the Handewitt way 305. It is a list
for the routing fix in `grid_network`; the drawing does not change it.

Station boxes also take in the station's other OSM yards (`--osm-substations
osm/DEU.gpkg`, `grid_gis.station_yards`): `power_substation_polygon`s within 200 m with the
station's operator, or without operator named after the station or "Erweiterung"
(Güstrow, Sottrum: separate 220/380 kV polygons). `route_overrides.csv` is honoured: a
circuit that is not routed and has an override (Preilack – Streumen 560, "wrong corridor;
draw dashed") is drawn dashed on its previous geometry, not on its partial pieces.

Mehringen (380 kV, OSM polygon 1280357913) has no terminal in the graph because no 380 kV
OSM way reaches it: the nearest one, Sottrum – Landesbergen 3003 (way 231735880), passes
373 m from the polygon with no node within the 60 m access buffer, and OSM has no loop-in
of 3003 into Mehringen. Its four circuits (Landesbergen – Mehringen 3, Mehringen – Dollern
grün-4 / schwarz-3, Y-Mehringen 2/3) stay `no_terminal` until the loop-in is mapped in OSM.

### Live and staging

8080 serves the published folder `~/data/grid/live_map` (app files; `data` →
`~/data/grid/web`), not the code checkout: `scripts/serve_live.sh` switches the server once,
and `scripts/publish_map.sh` copies staging (8094, `~/data/grid/staging_osmgraph`) into it.

### Acceptance (staging, model_v5_4a, 2026-11-09)

`scripts/map_acceptance.py --network ...`. Hard checks added for this design:
`tower_line_is_osm_way` (every `osm_lines` feature lies on its OSM way within 2 m),
`inside_box` (no circuit piece or tower line more than 2 m inside any drawn station shape) and
`station_touch` (both ends within 10 m of their own drawn shape, of overlapping shapes'
outline, or of their junction dot or the shape holding it). `on_own_way` is measured on the
GeoJSON geometry (before the screen offset) against the OSM ways of `tower_lines` and of the
graph; join/connector pieces are virtual links and only counted. `drawn_once` and the slot
check work per graph edge (a piece split by a station is one edge; a slot change next to the
circuit's own station is the station order of `grid_bundles`); a gap where a line passes a
station shape is counted as `station_passes`, not as a gap. The baselines were measured with
the earlier version of the script.

| check / metric | live v5_3e | staging v1 (baked bundles) | staging (strict cut) |
|---|---|---|---|
| circuits in service model / drawn | 1010 / 1010 | 1009 / 1008 | 1009 / 981 + 28 in-station |
| inside_box / station_touch failures | – | – | 0 / 0 |
| drawn not once | 43 | 0 | 6 |
| undashed straight pieces | 35 | 0 | 0 |
| gaps > 20 m (circuits) | 19 | 3 | 7 |
| slot changes without set change | 21 | 7 | 0 |
| on_own_way / no_zigzag / no_spike circuits | 880 / 658 / 45 | 750 / 598 / 315 | 0 / 0 / 0 |
| no_loop circuits | 55 | 212 | 12 (dashed old chains) |
| share on own way (30 m) | 0.926 | 0.106 | 1.000 |
| metadata names model sha | ok | ok | ok |

Remaining hard failures are listed in `$S/acceptance/acceptance.md`; they are input
issues, not drawing: dashed old chains without OSM pieces (loops, a repeated way), two
partial routes whose pieces lie far apart (Preilack – Streumen 560, override "wrong
corridor"; Bergshausen – Würgassen 1), and junction legs with short leftover gaps
(Y-Voehringen/Y-Buers DELLM O).

`--towers` adds the physical tower lines (OpenStreetMap `power=line` ways, plus Overpass JSON
ways outside Germany) and writes `tower_circuit_mismatch.csv` next to the GeoPackage. `--stations` lays out the stations (footprints, voltage bars, circuits ending at the footprint
edge; see below): the OpenStreetMap GeoPackage (`power_substation_polygon`), or later a
station table (`~/data/grid/v5/stations/stations.gpkg`, layer `stations`, polygons by
`station`). The live map uses the OpenStreetMap matching: on 2026-11-08 it put 95.6 % of the
circuit ends at a footprint edge against 93.4 % with the station table (after the second
pass, with the fallback boxes and spurs inside footprints: 98.9 %, 1,760 of 1,779
station-circuit ends, every station with footprint reached by at least one), whose footprints miss
yards at e.g. Vieselbach (380 kV), Conneforde, Rohrsdorf and Osterath. Since 2026-11-08 the map shows model_v5_3e (sha256 8be55949…); the web panel and the QWC title
and attribution name the model file and sha from the export metadata. It is exported without
`--andes`: the only ANDES case (high_load_t10.xlsx) is of v4final and would bring back nodes
v5 removed (herbentingen, frankfurt|380, perleberg - putlitz), so `andes_idx` is empty
until a v5 case exists. The v5 station table (`stations.gpkg`) reached 96.7 % of circuit
ends against 98.9 % with the OpenStreetMap matching, so the map keeps the OpenStreetMap
matching. `--diagrams` writes a single-line diagram
per substation (pypowsybl, `[powsybl]` extra) to `WEB_DIR/sld/<substation mrid>.svg`; the
station popup of the web map links it. `--andes` joins an ANDES case of the model so the `buses` layer carries the simulation bus
ids (`andes_idx`) needed for result overlays. Natural Earth borders are downloaded once
(cached) unless `--borders` is given.

## Staging vs production

One codebase, two build profiles. `app.js` reads `mode` from `data/metadata.json`:
`production` hides every review aid; anything else (staging exports have no `mode`) shows them.

| | Staging (cairo-lab 8094) | Production (8080 web map, 8088 QWC/QGIS) |
|---|---|---|
| Data | export folder as written by `export_grid_gis.py` (e.g. `staging_osmgraph/web_v5_1_c16`) | `scripts/make_prod_web.py` copy of one staging folder |
| Lines | model check per OSM way: orange/purple halo, check legend, check class and reason in the way popup, circuit list per way; unrouted circuits dashed | one solid line per model circuit (parallel circuits side by side), voltage/TSO colours, no dashed line anywhere; no OSM way layer |
| Popups | route notes (red), farm check, "needs owner" / "not sourced", spread notes in red | the same model values and sources; spread farms say neutrally "connected in the model via the regional landing substations"; no route-quality text |
| Station panel | data-quality flags, SLD, components | SLD, components and the model's machines per bus (from the frozen pickle), no flags; offshore farm names and MW in the landing bus's renewables lines |
| OSM layers | raw OSM assets, OSM way sections | none (see "Production content") |
| Offshore farms | markers, dashed links, review notes | in the Renewables group (off by default); popup: name, MW, turbines, cable, landing substation, sources; a solid link only where the model landing is sourced (no open question); no spread notes |
| Footer | `STAGING · Build … · model sha256 …` | `Build <staging build id> · model sha256 …` |

### Production content (model only)

Production shows only what the simulation uses: model circuits (`lines`), substations and
buses, transformers, synchronous plants (`plants`, `plant_aggregates`, `plant_connections`:
they become machines), the inverter infeed per bus (`renewable_inverters`), border terminals
(the neighbour equivalents with their flow note), HVDC injection points, `neighbours`, station
footprints, single-line diagrams, and the TSO-area and bidding-zone reference layers.
`station_levels` / `station_transformers` are model-derived but empty since the plain-box
layout (the inside is in the station panel); they are passed through as they are.

Removed (`make_prod_web.REMOVED`, checked by `model_only`):

| Layer | Why |
|---|---|
| `tower_lines` | raw OpenStreetMap power lines of all voltages ("Assets (OpenStreetMap)") |
| `osm_substations` | raw OpenStreetMap substation polygons ("Assets (OpenStreetMap)") |
| `osm_lines` | OSM way sections with the model check: context and review aid, not a model element; production draws the circuits from `lines` |

### Connected circuits in production

Production draws each circuit as one connected line that starts and ends at its own stations,
in the normal style, without comment. `make_prod_web.circuit_problem` (metric, EPSG:3035)
treats the pieces of a circuit as graph edges: a piece end on a drawn station shape (3 m) is
that shape's node (a line cut where it passes a station box), other ends within 20 m are one
node, and pieces of 20 m or less are contracted. The circuit is clean when the graph is
connected, the open ends are on its own two stations (station block, station or junction
point, or the tie's border port; 25 m), and no other node has one piece end (loose end) or
more than two (branch, stub, doubled piece). A clean circuit keeps its drawn staging route
(geometry byte-identical). A broken one is repaired (`repair_route`): the drawn pieces on the
shortest connected path (by length) between its two station terminals are kept and stray
branches or doubled pieces dropped. Only when no connected path exists is it replaced by one
straight line between the two stations, cut at their boxes (parallel fallbacks side by side). Circuits drawn only inside a
station (no geometry, 25 in c16) and the two HVDC links are kept as they are. The list is
`PROD_DIR/build_report/fallback_circuits.csv` (not published). The electrical data never
changes.

### Route qualification (staging only)

`scripts/route_qualification.py STAGING_DIR MAP_ACCEPTANCE_JSON OUT_DIR` classifies every
circuit of a staging build. `verified` when all hold:

1. every piece is routed on the OSM graph (`route_status` routed, `route_quality`
   osm_graph, empty `route_note`) and no piece has a straight `geometry_source`;
2. every `osm_lines` section the circuit is on has the check `match` or `explained` (sourced
   `check_reason`); `short` (< 150 m: the check of its corridor applies) is accepted;
   `no_tag`, `model_fewer` and `model_more` are not;
3. the circuit is on none of the staging map-acceptance lists (`ends_off_station`,
   `station_touch`, `gaps`, `ratio_outside` 0.7-1.5, `drawn_not_once`,
   `straight_undashed`, `on_own_way`, `no_zigzag`, `no_loop`, `no_spike`, `station_end`);
4. or it is in `route_fixes.csv` (a logged C-change with its citation), which waives only the
   way checks named there; rules 1 and 3 still apply.

Anything else is `schematic`. The backlog is
`docs/data_quality/issues/16-route-qualification/backlog.csv` (ranked by kV, then model km),
with `route_quality.csv` and `summary.json`. A circuit leaves the backlog by fixing it on
staging and re-running the rule. The classification is not written into production.

**Promote** (on cairo-lab, after the owner approved staging; one staging build per promotion):

```bash
bash scripts/publish_map.sh --dry-run ~/data/grid/staging_osmgraph/web_v5_1_c16  # build + checks, prints the publish
bash scripts/publish_map.sh           ~/data/grid/staging_osmgraph/web_v5_1_c16  # publish 8080 + 8088
bash scripts/publish_map.sh --rollback                                           # back to the copies of the last publish
```

The script checks the frozen model sha, writes `~/data/grid/prod_build_<staging folder>`,
builds and loads the QWC project in the QGIS container (`qgis/qgs_check.json`), runs
`scripts/prod_acceptance.py --model` (no debug field or text; feature counts and geometries of
every kept layer identical to staging, lines changed only for the connected fallbacks with the
electrical data unchanged; no removed layer (`model_only`); every circuit one connected line
station to station (`connected_circuits`); circuits = the model's AC line segments by id and
every bus, transformer, external injection and machine of the model present
(`model_complete`); all SLDs; app served on a test port with no console error, no halo, no
removed source and no dashed line style; a mouse click on one AC circuit opens its popup with
the circuit name and only the rows Voltage, Circuits, Length, Rating, TSO, Status; QGIS project and all layers valid, no dashed QGIS
style) and stops on any failure. Publishing first copies
the current production to `web_/qgis_/app_before_prod_<build id>` (tag in
`ROLLBACK_LAST_PROD.txt`).

The production profile (`mode: production`) also has its own compact side panel
(`prodPanel` in `app.js`, CSS under `body.prod`): Voltage/TSO and Light/Dark, search, one-word
layer checkboxes, a short legend of the visible layers only, the summary table (collapsed) and
the attribution and build id under "About"; at phone width it collapses to a button.
Staging keeps the full panel of `index.html` unchanged.

## Views

- **QGIS** (main view): see `apps/grid_map/qgis/README.md`. Bundle on the tailnet:
  <http://cairo-lab:8080/data/grid_model_qgis.zip>.
- **Browser**: <http://cairo-lab:8080> (served on cairo-lab with
  `cd ~/code/tranpy/apps/grid_map && python3 -m http.server 8080 --bind 0.0.0.0`;
  `apps/grid_map/data` is a link to `~/data/grid/web`). URL options:
  `?mode=tso`, `?basemap=dark`, `#zoom/lat/lon`.
  Search by name, EIC, bus id or ANDES id; click any asset for its model attributes.

## Layers and attributes

| Layer | Geometry | Main attributes |
|---|---|---|
| `lines` | LineString | `name`, `eic`, `voltage_kv`, `corridor`, `circuits` (circuits in the corridor), `circuit_no` (index, highest voltage first), `same_route` (circuits in the model with exactly this route), `end_gap_m`, `display_snapped_m`, `length_km`, `r_ohm`, `x_ohm`, `b_us`, `imax_a`, `rating_mva`, `tso` (control area), `tso_code` (as in the source), `bus1`, `bus2`, `is_interconnector` (exactly one end in Germany), `neighbour`, `from_zone`, `to_zone`, `tso_tie` (German TSO-TSO tie), `abroad` (both ends abroad), `hvdc`, `route_quality`, `route_source`, `route_length_km` (drawn length, display only), `geometry_source` (`model`, `overlay: <method>`, or `export section osm-N` for circuits taken from the OpenStreetMap export, drawn on the ways of their section, `+ modelled route to station` where the section ends on another line before the station), `status`, `tower_way` (OSM way ids of the piece), `piece`, `pieces`, `label_piece` (longest piece, carries the label). With tower lines, a circuit is one row per piece and `circuits` / `circuit_no` are the count and slot on that tower line |
| `tower_lines` | LineString | OpenStreetMap ways carrying a 220 or 380 kV circuit (also mixed-voltage ways): `osm_id`, `name`, `ref`, `operator`, `voltages`, `max_kv`, `mixed_voltage`, `circuits_tag` (220/380 kV circuits by the OSM tags: `circuits`, or `cables`/3, per voltage when the `;` lists align; empty when the tags do not say), `circuits_max` (the larger of `circuits` and `cables`/3), `cables`, `location`, `construction`, `disused`, `length_km`, `model_circuits`, `model_names`, `model_eics`, `model_kv` (model circuits drawn on the way), `other_voltage_circuits` (drawn circuits of a voltage the way does not carry), `parallel_way` (another way within 60 m along ≥ 30 % of this one), `mismatch` (`model_fewer`: below `circuits_tag`; `model_more`: above `circuits_max`; `neighbour_pickup`: above `circuits_max` only through other-voltage circuits or a parallel way; `no_tag`, `no_model_circuit`; empty when consistent and on every way shorter than 0.5 km) |
| `substations` | Point | `name`, `site`, `kind`, `symbol` (`380`, `220`, `coupling`, `tee`), `coupling` (380 and 220 kV buses), `max_kv`, `voltages_kv`, `buses`, `transformers`, `tso`, `country`, `bidding_zone`, `location_source`, `location_flag`, `display_relocated_m` |
| `station_footprints` | Polygon | `site`, `name`, `source` (`osm`: OpenStreetMap substation polygon(s); `box`: fixed-size box, no footprint found; `box at circuit ends`: the polygon or box was missed by every circuit end, replaced by a box centred at the ends), `osm_ids`, `kind`, `symbol`, `max_kv`, `voltages_kv`, `transformers`, `tso` |
| `station_levels` | LineString | `site`, `voltage_kv`, `part`, `mrid`: `part` `bar` is one busbar per voltage level inside the footprint; `part` `stub` is the link of one circuit end (`mrid`) from its edge port to its own point on its busbar |
| `station_transformers` | Point | as `transformers`, placed between the bars of their two voltages (to 110 kV: below the lowest bar), side by side by `slot` |
| `buses` | Point | `bus` (model node id), `bus_mrid`, `kind`, `voltage_kv`, `tso`, `tso_codes`, `country`, `country_source`, `bidding_zone`, `andes_idx`, `andes_generators`, `andes_gen_sn_mva`, `andes_gen_models`, `andes_gen_h_s`, `location_flag`, `display_relocated_m` |
| `transformers` | Point (substation) | `name`, `eic`, `bus_hv`, `bus_lv`, `kv_hv`, `kv_lv`, `sn_mva`, `r_ohm`, `x_ohm`, `phase_shifter`, `slot` (position in the row next to the substation), `site_transformers`, `tso` |
| `plants` | Point | `mastr_id`, `name`, `operator`, `fuel`, `fuel_category`, `capacity_mw`, `status`, `in_operation`, `reserve`, `bus` (connection), `andes_idx`, `tso`, `location` (`register` or `bus`), `display` (`individual` ≥ 50 MW or not in operation, else `aggregated`) |
| `plant_connections` | LineString | `mastr_id`, `name`, `bus`, `fuel_category`, `capacity_mw`, `in_operation`, `length_m`: individual plant to its connection substation |
| `plant_aggregates` | Point (substation) | `bus`, `site`, `fuel_category`, `capacity_mw` (sum), `units`, `names`, `slot`, `slot_y`, `bus_fuels`, `tso`: plants < 50 MW in operation, per connection bus and fuel |
| `neighbours` | Point | `bus`, `name`, `country`, `bidding_zone`, `tie_lines`, `tso` |
| `control_areas` | MultiPolygon | `tso`, `approximate` (always true), `method` |
| `bidding_zones` | MultiPolygon | `zone`, `zone_name`, `countries`, `approximate` |
| `metadata` | table | model info, `source_sha256`, `exported_utc`, `andes_case`, `geometry_overlay` |

With `--stations`, `lines` also carries `station_end` (site at whose footprint edge the
piece ends), and `substations` carries `footprint` (`osm`,
`box`, empty for tees), `junction_snap_m` (tee moved onto its tower line) and `diagram`
(relative path of the single-line diagram).

With `--network`, `lines` rows of circuits on the graph are OSM pieces: `piece_kind`
(`way`, `join`, `connector`; `end` and `break` for the straight display connectors),
`edge_id`, `tower_way` (the OSM way id), `circuit_no`/`circuits` (slot / n for the screen
offset), `station_cut`, `route_status`, `route_note` (empty when routed), `route_quality`
(`osm_graph`; `unrouted` = dashed).

`substations` and `buses` also carry `geometry_source` (`model`, `display relocated` or
`overlay: <method>`).

All layers are EPSG:4326.

Derived values and their limits:

- `tso` of a bus: control area from `grid_register.node_control_areas` (a node listed for
  several operators goes to the area with the most registered capacity nearby).
  Connection (110 kV) and generator-terminal (21 kV) buses of the ANDES case take the
  control area of their substation and are drawn at the substation.
- `route_quality = straight`: no route in the model, drawn bus to bus (dotted);
  `in_station`: both ends at one location, drawn as a 1 m stub.
- Display corrections (the model is not changed; each is flagged):
  - circuits on shared towers (`tranpy.grid_towers.split_by_towers`): points every 25 m along
    a drawn circuit take the nearest tower line within 60 m; stretches under 150 m (crossings,
    station entries) are dropped. The circuit is cut at tower-line boundaries; on each tower
    line its circuits get a stable slot (highest voltage first, then name) and each piece is
    oriented along the tower line (eastward, or northward on north-south lines), so a
    per-piece offset (pixels on the web, mm in QGIS) draws them side by side along the whole
    shared stretch and they branch where a circuit leaves. Circuits on no tower line keep the
    corridor grouping (routes within 150 m Hausdorff distance);
  - circuits taken from the OpenStreetMap export (`osm-N-k`, and curated circuits whose
    `route_source` names `export section osm-N`) have no route in the model; they are drawn
    along the ways of their export section, on the shortest path between the vertices
    nearest their two buses (`route_quality = mapped`). Where the section ends on another
    line before the station, the drawing follows that line's drawn route (within 200 m),
    else a straight leg;
  - a route end less than 3 km from its bus is extended to the bus (`display_snapped_m`,
    largest of both ends; `end_gap_m` is the gap before snapping, also when ≥ 3 km);
  - a substation whose model point is `schematic` (or an unsurveyed point on a 0.01° grid,
    `location_flag = rounded`) is drawn at the centroid of the route ends that terminate
    there, within 10 km (`display_relocated_m`); its buses, transformers, plant connectors
    and the lines' snapped ends follow.
  - geometry overlay (`--overlay geometry_overlay.parquet`, built by
    `scripts/build_geometry_overlay.py`, inventory in `geom_issues.csv`): a tee with a
    published leg shorter than 0.5 km to a station is drawn at that station
    (`tee_at_station`, `tranpy.grid_gis.place_tee`); a schematic route, or one ending more
    than 3 km from its bus, is replaced by a shortest path over same-voltage OpenStreetMap
    ways (`osm_route`) or another circuit's model route between the same ends
    (`sibling_model_route`) when that path is within 15 % of the published length; legs
    under 0.5 km are drawn in the station (`short_leg_straight`); otherwise a schematic route
    is drawn straight between its buses (`straight_between_corrected_ends`, dotted) and
    flagged. Lengths and parameters are never changed: `route_length_km` is the drawn
    length; a published length of 0 is listed in `geom_issues.csv` for the owner.
- Station layout (`--stations`, `tranpy.grid_stations.layout_stations`; display only, the
  model is not changed and every circuit keeps at least one row):
  - footprint: every OpenStreetMap substation polygon carrying 220 or 380 kV is given to the
    nearest station within 300 m; a station's polygons are merged (convex hull when apart,
    e.g. separate 380 and 220 kV yards). No polygon, or a merged footprint wider than 2 km:
    a box of 240 m × 70 m per voltage level around the display point (dashed outline);
    A footprint that no circuit end reaches (within 15 m) is replaced by a box centred at
    the circuit ends near the station (within 1.5 km; the largest group of ends within
    600 m of each other), `source` `box at circuit ends`;
  - circuits are clipped where they first enter the footprint of their end station (a
    circuit ending up to 15 m outside it ends there) and end at an edge port on the
    outline. Circuits of one voltage on one tower line (or, between tower lines, to the same
    station) crossing within 40 m share a port; ports are kept at least 30 m apart along the
    outline (less on small footprints). A piece lying wholly inside a footprint is dropped,
    unless it is the circuit's only piece there (a short spur from a nearby junction, a
    link between two overlapping yards): it is then drawn from its outer end to the nearest
    point of the outline;
  - parallel circuits stay side by side up to the port (no merged bundle). A piece between
    two tower lines whose neighbouring pieces sit in the same slot of a multi-circuit tower
    line, and the last piece into a station when shorter than 400 m, keep that slot, so the
    drawing does not jump sideways there;
  - busbars: one per voltage level; the layout is chosen among bars along the long axis of
    the footprint's minimum rotated rectangle or across it, in every order of the levels,
    as the one where the fewest links cross the bar of another level (ties: long axis,
    highest voltage on top). Each circuit end gets its own point on its bar (the point
    nearest its port, at least 8 m from the next), and a thin link (`part` `stub`) runs
    from the port to it; transformers sit between the bars of their two voltages;
  - tees and junctions are moved onto the nearest tower line within 200 m
    (`junction_snap_m`) and labelled from zoom 11.
  The web map shows footprints from zoom 11 and bars and transformers inside from zoom 12.5;
  below that the fixed-size station symbol is used. Busbars are dark grey (light grey on
  the dark basemap), thicker than any circuit and labelled with their voltage from zoom
  13.5; links are thin lines in the voltage colour (grey in the TSO view).
- `control_areas`: Voronoi cells of each TSO's buses, dissolved and clipped to the German
  land border (Natural Earth 1:50m); offshore areas are not covered. Approximate.
- `bidding_zones`: Natural Earth country shapes; DK1 and NO2 are drawn with the whole of
  Denmark and Norway (`approximate = true`).
- `country`: from the model; two German substations that Natural Earth 1:50m places just
  outside Germany (Hagenwerder, Jardelund) are counted as German (`country_source =
  border tolerance`, within 3 km of the border).

## Symbols (ENTSO-E grid-map conventions)

| Element | By voltage | By TSO |
|---|---|---|
| Circuit 380 kV / 220 kV | red / green, width per circuit, parallel circuits side by side | TSO colour; cross-border interconnectors purple with a halo |
| Tower line (OSM) | thin grey line under the circuits from zoom 9, ref label | same |
| HVDC | blue dashed | blue dashed |
| Route not mapped / out of service | dotted / grey dotted | same |
| Substation | filled red square (380 kV), green outlined square (220 kV only), half red/half green square (380/220 kV coupling); size grows with the transformer count | square in TSO colour |
| Station footprint (zoom ≥ 12.5) | grey fill, outline in the colour of the highest voltage (dashed: fixed box), one dark busbar per voltage level with its kV label, thin links in the voltage colour from the edge ports to the busbars, transformers between the bars | outline in TSO colour, grey links |
| Tee / switching point | small ring on the tower line, name label from zoom 11 | small ring |
| Transformer / phase shifter | two circles in a row up-right of the substation (HV circle left, LV right, in voltage colours) from zoom 7.5; one with the count (×n) until zoom 11 / purple diamond | TSO-coloured circles |
| Power plant | ≥ 50 MW: fuel shape and letter at its site, size ∝ √MW, thin grey connector to its substation; < 50 MW: summed per bus and fuel in a column left of the substation (MW from zoom 11.5) | circle in the TSO colour of its connection bus |
| Control areas / DE-LU | dashed outlines / thin outline | translucent fills / bold outline |

TSO colours: 50Hertz `#2a78d6`, Amprion `#eb6834`, TenneT `#1baf7a`, TransnetBW `#eda100`
(as in `tranpy.maps.plot_operator_map`). Labels depend on scale: 380 kV substation names
from about 1:2.2M (web zoom 8), 220 kV from 1:1.1M (zoom 9), line names and plants ≥ 100 MW
from 1:550k (zoom 10.5, one label per circuit), circuit counts (2×) per tower line from zoom 7, tower-line refs (e.g. 4132) from zoom 11.5,
plants ≥ 300 MW from zoom 8. Lines have a white casing; substations draw above lines and
plants above substations. The web basemap is CARTO Positron with its green land cover and
pink borders neutralised to grey.

## Sources and licences

- Electrical model: JAO Core Static Grid Model (2026-03 release).
- Line routes and substation locations: © OpenStreetMap contributors (ODbL), partly via the
  PyPSA-Eur prebuilt network.
- Power plants: Marktstammdatenregister, Bundesnetzagentur (dl-de/by-2-0).
- Borders: Natural Earth (public domain).
- Basemaps: CARTO Positron / Dark Matter, © OpenStreetMap contributors, © CARTO.
- Web map library: MapLibre GL JS 4.7.1 (BSD-3-Clause), loaded from unpkg.
