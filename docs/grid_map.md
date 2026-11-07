# Grid map: GeoPackage, QGIS project and web viewer

A GIS view of the frozen grid model (v4final, 380/220 kV): one GeoPackage as the single
source, a styled QGIS project (main view) and a light MapLibre web viewer with the same
cartography. The model is only read.

## Export

```bash
# on cairo-lab (needs the [grid] extra: geopandas, pyogrio; and pyarrow for the pickle)
python scripts/export_grid_gis.py ~/data/grid/model_v4final.pkl ~/data/grid \
    --web ~/data/grid/web --andes ~/data/ml_v41dyn/prep/high_load_t10.xlsx \
    --overlay ~/data/grid/geometry_overlay.parquet \
    --towers ~/data/grid/osm/hv_lines.parquet --towers-overpass ~/data/grid/osm/overpass_at_west.json
# QGIS styles/project, SVG symbols and style.js (no model needed)
python scripts/export_grid_gis.py --app apps/grid_map
```

`--towers` adds the physical tower lines (OpenStreetMap `power=line` ways, plus Overpass JSON
ways outside Germany) and writes `tower_circuit_mismatch.csv` next to the GeoPackage. `--andes` joins an ANDES case of the model so the `buses` layer carries the simulation bus
ids (`andes_idx`) needed for result overlays. Natural Earth borders are downloaded once
(cached) unless `--borders` is given.

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
| `lines` | LineString | `name`, `eic`, `voltage_kv`, `corridor`, `circuits` (circuits in the corridor), `circuit_no` (index, highest voltage first), `same_route` (circuits in the model with exactly this route), `end_gap_m`, `display_snapped_m`, `length_km`, `r_ohm`, `x_ohm`, `b_us`, `imax_a`, `rating_mva`, `tso` (control area), `tso_code` (as in the source), `bus1`, `bus2`, `is_interconnector` (exactly one end in Germany), `neighbour`, `from_zone`, `to_zone`, `tso_tie` (German TSO-TSO tie), `abroad` (both ends abroad), `hvdc`, `route_quality`, `route_source`, `route_length_km` (drawn length, display only), `geometry_source` (`model` or `overlay: <method>`), `status`, `tower_way` (OSM way ids of the piece), `piece`, `pieces`, `label_piece` (longest piece, carries the label). With tower lines, a circuit is one row per piece and `circuits` / `circuit_no` are the count and slot on that tower line |
| `tower_lines` | LineString | OpenStreetMap ways carrying a 220 or 380 kV circuit (also mixed-voltage ways): `osm_id`, `name`, `ref`, `operator`, `voltages`, `max_kv`, `mixed_voltage`, `circuits_tag` (220/380 kV circuits by the OSM tags: `circuits`, or `cables`/3, per voltage when the `;` lists align; empty when the tags do not say), `cables`, `location`, `construction`, `disused`, `length_km`, `model_circuits`, `model_names`, `model_eics`, `model_kv` (model circuits drawn on the way), `mismatch` (`model_fewer`, `model_more`, `no_tag`, `no_model_circuit`, empty when equal) |
| `substations` | Point | `name`, `site`, `kind`, `symbol` (`380`, `220`, `coupling`, `tee`), `coupling` (380 and 220 kV buses), `max_kv`, `voltages_kv`, `buses`, `transformers`, `tso`, `country`, `bidding_zone`, `location_source`, `location_flag`, `display_relocated_m` |
| `buses` | Point | `bus` (model node id), `bus_mrid`, `kind`, `voltage_kv`, `tso`, `tso_codes`, `country`, `country_source`, `bidding_zone`, `andes_idx`, `andes_generators`, `andes_gen_sn_mva`, `andes_gen_models`, `andes_gen_h_s`, `location_flag`, `display_relocated_m` |
| `transformers` | Point (substation) | `name`, `eic`, `bus_hv`, `bus_lv`, `kv_hv`, `kv_lv`, `sn_mva`, `r_ohm`, `x_ohm`, `phase_shifter`, `slot` (position in the row next to the substation), `site_transformers`, `tso` |
| `plants` | Point | `mastr_id`, `name`, `operator`, `fuel`, `fuel_category`, `capacity_mw`, `status`, `in_operation`, `reserve`, `bus` (connection), `andes_idx`, `tso`, `location` (`register` or `bus`), `display` (`individual` ≥ 50 MW or not in operation, else `aggregated`) |
| `plant_connections` | LineString | `mastr_id`, `name`, `bus`, `fuel_category`, `capacity_mw`, `in_operation`, `length_m`: individual plant to its connection substation |
| `plant_aggregates` | Point (substation) | `bus`, `site`, `fuel_category`, `capacity_mw` (sum), `units`, `names`, `slot`, `slot_y`, `bus_fuels`, `tso`: plants < 50 MW in operation, per connection bus and fuel |
| `neighbours` | Point | `bus`, `name`, `country`, `bidding_zone`, `tie_lines`, `tso` |
| `control_areas` | MultiPolygon | `tso`, `approximate` (always true), `method` |
| `bidding_zones` | MultiPolygon | `zone`, `zone_name`, `countries`, `approximate` |
| `metadata` | table | model info, `source_sha256`, `exported_utc`, `andes_case`, `geometry_overlay` |

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
| Tee / switching point | small ring | small ring |
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
