/* German grid model viewer (MapLibre GL JS). Data: ./data/*.geojson written by
   scripts/export_grid_gis.py; colours and symbols: style.js (generated from the same
   specification as the QGIS styles). */
(function () {
  "use strict";
  const S = window.GRID_STYLE;
  const BASEMAPS = {
    light: "https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json",
    dark: "https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json",
  };
  const LAYER_FILES = ["bidding_zones", "control_areas", "plant_connections", "tower_lines", "lines", "buses",
    "transformers", "substations", "plant_aggregates", "plants", "neighbours"];
  // station boxes (footprint or fixed box); station_levels/station_transformers are empty
  // since the plain-box layout (the inside is in the station panel); all optional so an
  // older export without these files still loads
  // osm_lines: one feature per OSM 220/380 kV way section with the model check (the clean
  // grid layer); osm_substations: raw OSM substation polygons (OpenStreetMap group)
  // border_terminals: one symbol per border crossing of the AC tie lines (cut at the German
  // border) and per HVDC converter; the foreign grid is not drawn
  const OPTIONAL_FILES = ["station_footprints", "station_levels", "station_transformers", "osm_lines",
    "osm_substations", "border_terminals", "renewable_inverters", "offshore_farms", "offshore_links"];
  // layers not in a production build (OpenStreetMap context and offshore farm markers)
  const PROD_REMOVED = ["tower_lines", "osm_lines", "osm_substations"];
  const STATION_ZOOM = 12.5; // footprints replace the fixed-size station symbol from here
  // two groups: our model (clean layers of the frozen model) and the raw OpenStreetMap assets
  // to compare against. The per-circuit model lines are not a panel layer: one circuit is
  // highlighted on click (circuit name in a popup or the station panel).
  const PANEL_GROUPS = [
    { title: "Our model", layers: ["grid", "transformers", "buses", "plants", "renewables", "control_areas", "bidding_zones"] },
    { title: "OpenStreetMap", layers: ["osm_assets"] },
  ];
  const panelGroups = () => (PROD() ? PANEL_GROUPS.filter((g) => g.title !== "OpenStreetMap") : PANEL_GROUPS);
  const TSOS = Object.keys(S.tso);
  // renewables layer colours: ENTSO-E Transparency Platform / energy-charts fuel convention
  const RENEW = {
    solar: { label: "Solar", colour: "#f6b400" },
    wind_onshore: { label: "Wind onshore", colour: "#5aa9e6" },
    wind_offshore: { label: "Wind offshore", colour: "#1f4e9c" },
    storage: { label: "Battery storage", colour: "#8e5bb5" },
  };
  // build profile (metadata.json "mode", written by scripts/make_prod_web.py): "production"
  // hides every check / data-quality aid (halo, check legend, flags, review notes); anything
  // else (staging, older exports without the key) shows them all
  const PROD = () => ((state.meta || {}).mode === "production");
  // a note that is a review flag in staging (red) is plain text in production
  const flagCls = () => (PROD() ? "note" : "flag");
  const state = { mode: "voltage", basemap: "light", data: {}, visible: {
    grid: true, substations: true, transformers: true, buses: false, plants: true, renewables: false, control_areas: true, bidding_zones: true,
    osm_assets: false,
  } };
  const TITLES = {
    grid: "Lines (tower lines, model check) and substations", transformers: "Transformers",
    buses: "Buses (model / ANDES ids)", plants: "Power plants", renewables: "Renewables (inverter): model infeed per substation, offshore wind farms",
    control_areas: "TSO control areas (approx.)",
    bidding_zones: "Bidding zones", osm_assets: "Assets (OpenStreetMap, raw: lines all voltages, substations)",
  };

  const PROD_TITLES = { grid: "Lines and substations", renewables: "Renewables (inverter): model infeed per substation, offshore wind farms" };

  const map = new maplibregl.Map({
    container: "map", style: BASEMAPS.light, center: [10.4, 51.2], zoom: 5.6, minZoom: 4,
    maxZoom: 15, hash: true, attributionControl: false,
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: true, visualizePitch: false }), "top-right");
  map.addControl(new maplibregl.ScaleControl({ maxWidth: 140, unit: "metric" }), "bottom-right");
  map.addControl(new maplibregl.AttributionControl({ compact: true, customAttribution: [
    "Routes © OpenStreetMap contributors (ODbL) via PyPSA-Eur",
    "Grid data JAO Core Static Grid Model", "Plants Marktstammdatenregister (dl-de/by-2-0)",
    "Borders Natural Earth", S.basemap_attribution,
  ] }), "bottom-right");

  // ---------------------------------------------------------------- expressions
  const kvColour = ["case", ["get", "hvdc"], S.voltage.hvdc,
    [">=", ["get", "voltage_kv"], 300], S.voltage["380"], S.voltage["220"]];
  const tsoColour = ["match", ["get", "tso"], ...TSOS.flatMap((t) => [t, S.tso[t]]), S.neutral];
  const lineColour = () => state.mode === "voltage" ? kvColour
    : ["case", ["get", "is_interconnector"], S.interconnector, tsoColour];
  const is380 = [">=", ["get", "voltage_kv"], 300];
  // per-circuit width (px) and spacing of the circuits of one corridor, by zoom; zoom must be
  // the outermost expression, so every stop carries the data expression
  const Z = S.lines.zoom;
  const lw = (add = 0, f = 1) => ["interpolate", ["linear"], ["zoom"],
    ...Z.flatMap(([z, w380, w220]) => [z, ["+", add, ["*", f, ["case", is380, w380, w220]]]])];
  const dashed = ["match", ["get", "route_quality"], ["straight", "unrouted"], true, false];
  // selected circuit: electric blue halo with a white core on its tower lines (no offset)
  const SELECTED = "#2962ff";
  const live = ["==", ["get", "status"], "in_operation"];
  const zoomed = (stops, expr) => ["interpolate", ["linear"], ["zoom"],
    ...stops.flatMap(([z, f]) => [z, ["*", f, expr]])];
  const plantSize = ["interpolate", ["linear"], ["sqrt", ["max", ["get", "capacity_mw"], 0]],
    ...S.plants.size_stops.flat()];
  const plantIcon = zoomed(S.plants.zoom_factor, plantSize);
  const kvClass = (field) => ["case", [">=", ["get", field], 300], "380", [">=", ["get", field], 200], "220", "110"];
  const CABLE = ["any", ["in", "underground", ["coalesce", ["get", "location"], ""]],
    ["in", "underwater", ["coalesce", ["get", "location"], ""]]];
  const subBase = ["case", ["==", ["get", "symbol"], "220"], 0.78, 1.0];
  const subGrow = ["+", 1, ["*", 0.07, ["min", ["get", "transformers"], 6]]];
  const subSize = zoomed(S.substations.zoom_size, ["*", subBase, subGrow]);

  // ---------------------------------------------------------------- data and images
  // cache busting: every data file is fetched with ?v=<build id> from metadata.json, which
  // itself is never cached, so a new export is seen without a hard reload
  let buildId = "";
  const dataUrl = (path) => (buildId ? `${path}?v=${encodeURIComponent(buildId)}` : path);
  async function loadData() {
    try {
      const meta = await (await fetch("data/metadata.json", { cache: "no-store" })).json();
      state.meta = meta;
    } catch (e) { state.meta = {}; }
    buildId = state.meta.build_id || state.meta.exported_utc || "";
    // production holds only the model's layers (scripts/make_prod_web.py REMOVED)
    const skip = PROD() ? new Set(PROD_REMOVED) : new Set();
    skip.forEach((name) => { state.data[name] = { type: "FeatureCollection", features: [] }; });
    const entries = await Promise.all(LAYER_FILES.filter((n) => !skip.has(n)).map(async (name) => {
      const r = await fetch(dataUrl(`data/${name}.geojson`));
      if (!r.ok) throw new Error(`data/${name}.geojson: ${r.status}`);
      return [name, await r.json()];
    }));
    entries.forEach(([k, v]) => { state.data[k] = v; });
    state.data.station_labels = stationLabels(); // filled once the footprints are loaded
    await Promise.all(OPTIONAL_FILES.filter((n) => !skip.has(n)).map(async (name) => {
      const r = await fetch(dataUrl(`data/${name}.geojson`)).catch(() => null);
      state.data[name] = r && r.ok ? await r.json() : { type: "FeatureCollection", features: [] };
    }));
    state.data.station_labels = stationLabels();
    state.hasWays = state.data.osm_lines.features.length > 0;
    // circuits with no piece on an OSM way are drawn dashed in the grid layer (on_graph false)
    const onGraph = new Set(state.data.lines.features.filter((f) => f.properties.piece_kind === "way")
      .map((f) => f.properties.mrid));
    state.data.lines.features.forEach((f) => { f.properties.on_graph = onGraph.has(f.properties.mrid); });
  }

  // one label point per station box, just above its top edge, so the name never covers it
  function stationLabels() {
    const fp = state.data.station_footprints;
    const features = ((fp && fp.features) || []).map((f) => {
      const g = f.geometry;
      if (!g) return null;
      const rings = g.type === "Polygon" ? g.coordinates : g.type === "MultiPolygon" ? g.coordinates.flat() : [];
      const pts = rings.flat();
      if (!pts.length) return null;
      const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
      const at = [(Math.min(...xs) + Math.max(...xs)) / 2, Math.max(...ys)];
      // the station's own name, not the transformer names joined to it ("TR …")
      const parts = String(f.properties.name || f.properties.site || "").split(";").map((t) => t.trim()).filter(Boolean);
      const label = parts.find((t) => !t.startsWith("TR ")) || parts[0] || "";
      return { type: "Feature", geometry: { type: "Point", coordinates: at }, properties: { ...f.properties, label } };
    }).filter(Boolean);
    return { type: "FeatureCollection", features };
  }

  function svgImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image(48, 48);
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = url;
    });
  }

  async function addImages() {
    const names = [...Object.keys(S.fuels).map((f) => `plant_${f}`), "substation_380",
      "substation_220", "substation_coupling", ...TSOS.map((t) => `substation_${t.toLowerCase()}`), "tee",
      ...S.transformer_symbols, "phase_shifter", "neighbour", "wind_offshore",
    ...["border", "hvdc_terminal"].flatMap((k) => ["export", "import", "both", "nodata"].map((d) => `${k}_${d}`))];
    await Promise.all(names.map(async (n) => {
      if (map.hasImage(n)) return;
      map.addImage(n, await svgImage(`symbols/${n}.svg`), { pixelRatio: 2 });
    }));
  }

  // neutral grey basemap: CARTO Positron tints land cover and parks green, borders pink
  function neutraliseBasemap() {
    if (state.basemap !== "light") return;
    for (const layer of map.getStyle().layers) {
      if (layer.type === "fill" && /landcover|park|landuse(?!_residential)/.test(layer.id)) {
        map.setPaintProperty(layer.id, "fill-color", "rgba(232,232,232,0.35)");
      }
      if (layer.type === "line" && /^boundary/.test(layer.id)) {
        map.setPaintProperty(layer.id, "line-color", "#d0d0d0");
      }
    }
  }

  // ---------------------------------------------------------------- layers
  function addOverlay() {
    for (const name of [...LAYER_FILES, ...OPTIONAL_FILES, "station_labels"]) {
      if (!map.getSource(name)) map.addSource(name, { type: "geojson", data: state.data[name] });
    }
    neutraliseBasemap();
    const vis = (name) => (state.visible[name] ? "visible" : "none");
    const dark = state.basemap === "dark";
    const halo = dark ? "#111417" : "#ffffff";
    const text = dark ? "#e8eaed" : "#212121";
    // production: one uniform solid style, no dashed lines anywhere
    const add = (layer) => {
      if (PROD() && PROD_REMOVED.includes(layer.source)) return;
      if (PROD() && layer.paint) delete layer.paint["line-dasharray"];
      if (!map.getLayer(layer.id)) map.addLayer(layer);
    };
    const tso = state.mode === "tso";

    add({ id: "zones-fill", type: "fill", source: "bidding_zones", layout: { visibility: vis("bidding_zones") },
      paint: { "fill-color": dark ? "#263238" : "#eeeeee",
        "fill-opacity": ["case", ["==", ["get", "zone"], "DE-LU"], 0, 0.5] } });
    add({ id: "zones-line", type: "line", source: "bidding_zones", filter: ["!=", ["get", "zone"], "DE-LU"],
      layout: { visibility: vis("bidding_zones") },
      paint: { "line-color": dark ? "#b0bec5" : "#757575", "line-width": 0.6, "line-dasharray": [3, 2] } });
    add({ id: "zones-delu", type: "line", source: "bidding_zones", filter: ["==", ["get", "zone"], "DE-LU"],
      layout: { visibility: vis("bidding_zones") },
      paint: { "line-color": dark ? "#cfd8dc" : "#424242", "line-width": tso ? 2.2 : 1.2 } });
    add({ id: "areas-fill", type: "fill", source: "control_areas", layout: { visibility: vis("control_areas") },
      paint: { "fill-color": tsoColour, "fill-opacity": tso ? 0.16 : 0 } });
    add({ id: "areas-line", type: "line", source: "control_areas", layout: { visibility: vis("control_areas") },
      paint: { "line-color": tso ? tsoColour : (dark ? "#78909c" : "#9e9e9e"), "line-width": tso ? 1.6 : 0.8,
        "line-dasharray": [3, 2], "line-opacity": tso ? 0.9 : 0.6 } });

    // plant connectors under the grid: thin grey from the plant to its substation (zoom 12+)
    add({ id: "plant-links", type: "line", source: "plant_connections", minzoom: 12,
      filter: ["get", "in_operation"], layout: { visibility: vis("plants"), "line-cap": "round" },
      paint: { "line-color": dark ? "#90a4ae" : "#607d8b",
        "line-width": ["interpolate", ["linear"], ["zoom"], 12, 0.8, 15, 1.4], "line-opacity": 0.8 } });

    // OpenStreetMap group: raw OSM assets in neutral grey, thin, for comparison with the model
    const raw = vis("osm_assets");
    add({ id: "osm-subs-fill", type: "fill", source: "osm_substations", minzoom: 10,
      layout: { visibility: raw }, paint: { "fill-color": dark ? "#90a4ae" : "#9e9e9e", "fill-opacity": 0.12 } });
    add({ id: "osm-subs-line", type: "line", source: "osm_substations", minzoom: 10,
      layout: { visibility: raw }, paint: { "line-color": dark ? "#b0bec5" : "#757575", "line-width": 0.8,
        "line-dasharray": [2, 1.5] } });
    add({ id: "towers", type: "line", source: "tower_lines", minzoom: 7, filter: ["!", CABLE],
      layout: { visibility: raw, "line-cap": "butt", "line-join": "round" },
      paint: { "line-color": dark ? "#78909c" : "#9e9e9e",
        "line-width": ["interpolate", ["linear"], ["zoom"], 7, 0.4, 12, 0.9, 15, 1.4],
        "line-opacity": 0.85 } });
    // underground / submarine cables of the raw OSM layer: same line, dashed
    add({ id: "towers-cable", type: "line", source: "tower_lines", minzoom: 7, filter: CABLE,
      layout: { visibility: raw, "line-cap": "butt", "line-join": "round" },
      paint: { "line-color": dark ? "#78909c" : "#9e9e9e",
        "line-width": ["interpolate", ["linear"], ["zoom"], 7, 0.5, 12, 1.1, 15, 1.6],
        "line-opacity": 0.9, "line-dasharray": [2, 1.5] } });
    add({ id: "towers-ref", type: "symbol", source: "tower_lines", minzoom: 11.5,
      filter: ["!=", ["get", "ref"], ""],
      layout: { visibility: raw, "symbol-placement": "line", "symbol-spacing": 400,
        "text-field": ["get", "ref"], "text-size": 9.5, "text-font": ["Open Sans Italic"],
        "text-offset": [0, 1.15], "text-max-angle": 35, "text-padding": 4 },
      paint: { "text-color": dark ? "#b0bec5" : "#616161", "text-halo-color": halo, "text-halo-width": 1.2 } });

    // OSM 220/380 kV tower lines (one per way section, exact geometry, cut at the station
    // boxes), coloured by voltage; the model check per section: halo only where the model
    // carries fewer / more circuits than OSM and the difference is not explained (explained:
    // reason in the popup; sections under 150 m: no halo), thin and faded where no model
    // circuit runs
    const ways = vis("grid");
    const chk = (c) => ["==", ["get", "check"], c];
    // no model circuit on the way: check class (staging) or the in_model flag (production)
    const noModel = ["any", chk("no_model"), ["==", ["get", "in_model"], false]];
    // visible at every zoom: thin at low zoom, full width from zoom 10
    const wayW = (lo, mid, hi) => ["interpolate", ["linear"], ["zoom"],
      4, ["case", noModel, lo * 0.6, lo], 10, ["case", noModel, mid * 0.6, mid],
      14, ["case", noModel, hi * 0.5, hi]];
    if (!PROD()) add({ id: "osm-halo", type: "line", source: "osm_lines",
      filter: ["match", ["get", "check"], ["model_fewer", "model_more"], true, false],
      layout: { visibility: ways, "line-cap": "butt", "line-join": "round" },
      paint: { "line-color": ["match", ["get", "check"], "model_fewer", S.check_fewer || "#ff9800", S.check_more || "#8e24aa"],
        "line-width": wayW(2.5, 7, 11), "line-opacity": 0.45 } });
    add({ id: "osm-lines", type: "line", source: "osm_lines",
      layout: { visibility: ways, "line-cap": "butt", "line-join": "round", "line-sort-key": ["get", "voltage_kv"] },
      paint: { "line-color": tso ? (dark ? "#b0bec5" : "#546e7a") : ["case", [">=", ["get", "voltage_kv"], 300], S.voltage["380"], S.voltage["220"]],
        "line-width": wayW(0.9, 2.4, 4),
        "line-opacity": ["case", noModel, 0.6, chk("no_tag"), 0.85, 1] } });
    add({ id: "osm-label", type: "symbol", source: "osm_lines", minzoom: 10,
      filter: ["!=", ["coalesce", ["get", "ref"], ""], ""],
      layout: { visibility: ways, "symbol-placement": "line", "symbol-spacing": 350,
        "text-field": ["get", "ref"], "text-size": 10, "text-font": ["Open Sans Italic"],
        "text-offset": [0, 1.1], "text-max-angle": 35, "text-padding": 4 },
      paint: { "text-color": text, "text-halo-color": halo, "text-halo-width": 1.3 } });

    // station footprints (OpenStreetMap polygon, or a fixed box) under the circuits
    // production: substations have their own toggle; staging: they follow "grid"
    const subVis = PROD() ? vis("substations") : vis("grid");
    const kvLine = (field) => ["case", [">=", ["get", field], 300], S.voltage["380"], [">=", ["get", field], 200], S.voltage["220"], S.voltage["110"] || "#555555"];
    // one plain box per station: light fill and outline in the colour of its highest voltage
    add({ id: "stations-fill", type: "fill", source: "station_footprints", minzoom: 11,
      layout: { visibility: subVis },
      paint: { "fill-color": tso ? tsoColour : kvLine("max_kv"),
        "fill-opacity": ["interpolate", ["linear"], ["zoom"], 11, 0, STATION_ZOOM, dark ? 0.22 : 0.14] } });
    for (const box of [false, true]) {
      add({ id: box ? "stations-outline-box" : "stations-outline", type: "line", source: "station_footprints", minzoom: 11,
        filter: [box ? "==" : "!=", ["get", "source"], "box"], layout: { visibility: subVis },
        paint: { "line-color": tso ? tsoColour : kvLine("max_kv"),
          "line-width": ["interpolate", ["linear"], ["zoom"], 11, 1, 15, 2.2],
          ...(box ? { "line-dasharray": [2, 1.5] } : {}),
          "line-opacity": ["interpolate", ["linear"], ["zoom"], 11, 0, STATION_ZOOM, 0.9] } });
    }

    // model circuits that are not on an OSM way (whole circuit, or the break of a partial
    // route) are part of the grid layer, dashed; HVDC links dashed in their colour. Out of
    // service circuits are not drawn (the station panel lists them).
    const lineVis = vis("grid");
    const offGraph = ["any", ["!", ["get", "on_graph"]],
      ["all", ["!=", ["get", "route_status"], "routed"], ["==", ["get", "piece_kind"], "break"]]];
    if (PROD()) {
      // production: the model circuits themselves, one solid line per circuit, parallel
      // circuits side by side (slot = circuit_no of circuits)
      const slot = ["-", ["get", "circuit_no"], ["/", ["-", ["get", "circuits"], 1], 2]];
      const offset = ["interpolate", ["linear"], ["zoom"], ...Z.flatMap(([z, , , gap]) => [z, ["*", slot, gap]])];
      add({ id: "lines-casing", type: "line", source: "lines", minzoom: 6.5, filter: ["all", live, ["!", ["get", "hvdc"]]],
        layout: { visibility: lineVis, "line-cap": "round", "line-join": "round" },
        paint: { "line-color": halo, "line-width": lw(1.6), "line-offset": offset, "line-opacity": 0.9 } });
      add({ id: "lines-ac", type: "line", source: "lines", filter: ["all", live, ["!", ["get", "hvdc"]]],
        layout: { visibility: lineVis, "line-cap": "round", "line-join": "round", "line-sort-key": ["get", "voltage_kv"] },
        paint: { "line-color": lineColour(), "line-width": lw(), "line-offset": offset } });
    }
    if (!PROD()) add({ id: "lines-straight", type: "line", source: "lines",
      filter: ["all", live, ["!", ["get", "hvdc"]], offGraph],
      layout: { visibility: lineVis, "line-cap": "butt" },
      paint: { "line-color": lineColour(), "line-width": lw(0, 0.8), "line-dasharray": [1.2, 1.2] } });
    add({ id: "lines-hvdc", type: "line", source: "lines", filter: ["all", live, ["get", "hvdc"]],
      layout: { visibility: lineVis },
      paint: { "line-color": S.voltage.hvdc, "line-width": lw(0, 1.1), "line-dasharray": [3, 1.6] } });
    // one selected circuit: thick overlay along its pieces, station to station (dashed where
    // a piece is not on the OSM graph); shown whatever the circuits toggle says
    const sel = ["==", ["get", "mrid"], state.selected || ""];
    add({ id: "circuit-hl", type: "line", source: "lines", filter: ["all", sel, ["!", dashed]],
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": SELECTED, "line-width": ["interpolate", ["linear"], ["zoom"], 4, 6, 10, 10, 14, 14],
        "line-opacity": 0.55 } });
    add({ id: "circuit-hl-core", type: "line", source: "lines", filter: ["all", sel, ["!", dashed]],
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": "#ffffff", "line-width": ["interpolate", ["linear"], ["zoom"], 4, 1.5, 10, 2.5, 14, 3.5],
        "line-opacity": 0.95 } });
    add({ id: "circuit-hl-dash", type: "line", source: "lines", filter: ["all", sel, dashed],
      layout: { "line-cap": "butt" },
      paint: { "line-color": SELECTED, "line-width": ["interpolate", ["linear"], ["zoom"], 4, 4, 10, 7, 14, 10],
        "line-dasharray": [1.5, 1.2], "line-opacity": 0.8 } });
    add({ id: "buses", type: "circle", source: "buses", minzoom: 7,
      layout: { visibility: vis("buses") },
      paint: { "circle-radius": ["interpolate", ["linear"], ["zoom"], 7, 1.5, 12, 4],
        "circle-color": ["case", ["in", ["get", "kind"], ["literal", ["connection", "terminal"]]], "#bdbdbd",
          ...(tso ? [tsoColour]
            : [[">=", ["get", "voltage_kv"], 300], S.voltage["380"], S.voltage["220"]])],
        "circle-stroke-color": halo, "circle-stroke-width": 0.6 } });

    // substations above the lines: size by highest voltage and transformer count
    const ring = ["==", ["get", "symbol"], "tee"];
    const domestic = ["!", ["coalesce", ["get", "foreign_node"], false]];
    const terminals = state.data.border_terminals.features.length > 0;
    const subIcon = !tso ? ["concat", "substation_", ["get", "symbol"]]
      : ["match", ["get", "tso"], ...TSOS.flatMap((t) => [t, `substation_${t.toLowerCase()}`]), "substation_380"];
    // junctions: a small dot on the tower line where the legs meet
    add({ id: "tees", type: "circle", source: "substations", minzoom: 7.5, filter: ring,
      layout: { visibility: vis("grid") },
      paint: { "circle-radius": ["interpolate", ["linear"], ["zoom"], 7.5, 1.5, 12, 3, 15, 4],
        "circle-color": dark ? "#eceff1" : "#263238", "circle-stroke-color": halo, "circle-stroke-width": 1 } });
    add({ id: "substations", type: "symbol", source: "substations", filter: ["all", ["!", ring], domestic],
      layout: { visibility: subVis, "icon-image": subIcon, "icon-allow-overlap": true,
        "icon-ignore-placement": true, "icon-size": subSize, "symbol-sort-key": ["-", 0, ["get", "max_kv"]] },
      paint: { "icon-opacity": ["interpolate", ["linear"], ["zoom"], STATION_ZOOM - 0.5, 1,
        STATION_ZOOM, ["case", ["==", ["coalesce", ["get", "footprint"], ""], ""], 1, 0]] } });
    // junctions (tees) on their tower line, labelled
    add({ id: "tees-label", type: "symbol", source: "substations", minzoom: 11, filter: ring,
      layout: { visibility: vis("grid"), "text-field": ["get", "name"], "text-size": 10,
        "text-font": ["Open Sans Italic"], "text-offset": [0.8, 0], "text-anchor": "left", "text-optional": true },
      paint: { "text-color": text, "text-halo-color": halo, "text-halo-width": 1.4 } });

    // transformers: two-circle symbol next to the substation (HV colour left, LV right);
    // the first one with the count at mid zoom, all side by side when zoomed in
    const trIcon = ["case", ["get", "phase_shifter"], "phase_shifter",
      ["concat", "transformer_", kvClass("kv_hv"), "_", kvClass("kv_lv")]];
    const trSize = ["interpolate", ["linear"], ["zoom"], 7, 0.55, 10, 0.75, 13, 1.0];
    const trOffset = ["literal", [24, -20]];
    add({ id: "transformers-mid", type: "symbol", source: "transformers", minzoom: S.label_zoom.transformer,
      maxzoom: S.label_zoom.transformer_all, filter: ["==", ["get", "slot"], 0],
      layout: { visibility: vis("transformers"), "icon-image": trIcon, "icon-size": trSize, "icon-offset": trOffset,
        "icon-allow-overlap": true, "icon-ignore-placement": true,
        "text-field": ["case", [">", ["get", "site_transformers"], 1], ["concat", "×", ["to-string", ["get", "site_transformers"]]], ""],
        "text-size": 10, "text-font": ["Open Sans Bold"], "text-anchor": "left", "text-offset": [2.4, -1.1],
        "text-allow-overlap": true, "text-ignore-placement": true },
      paint: { "text-color": text, "text-halo-color": halo, "text-halo-width": 1.4 } });
    const hasStations = state.data.station_footprints.features.length > 0;
    add({ id: "transformers", type: "symbol", source: "transformers", minzoom: S.label_zoom.transformer_all,
      maxzoom: hasStations ? STATION_ZOOM : 24,
      layout: { visibility: vis("transformers"), "icon-image": trIcon, "icon-size": trSize,
        "icon-offset": ["case", ...[0, 1, 2, 3, 4, 5, 6, 7].flatMap((k) =>
          [["==", ["get", "slot"], k], ["literal", [24 + 22 * k, -20]]]), ["literal", [24, -20]]],
        "icon-allow-overlap": true, "icon-ignore-placement": true } });

    add({ id: "substations-label", type: "symbol", source: "substations",
      filter: ["all", ["!", ring], domestic, ["any",
        ["all", [">=", ["get", "max_kv"], 300], [">=", ["zoom"], S.label_zoom.substation_380]],
        [">=", ["zoom"], S.label_zoom.substation_220]]],
      layout: { visibility: subVis, "text-field": ["get", "site"], "text-size": 11,
        "text-font": ["Montserrat SemiBold"], "text-offset": [0.7, 0.45], "text-anchor": "top-left",
        "text-transform": "uppercase", "text-letter-spacing": 0.03, "text-optional": true },
      paint: { "text-color": text, "text-halo-color": halo, "text-halo-width": 1.6,
        "text-opacity": ["step", ["zoom"], 1, STATION_ZOOM,
          ["case", ["==", ["coalesce", ["get", "footprint"], ""], ""], 1, 0]] } });
    add({ id: "stations-label", type: "symbol", source: "station_labels", minzoom: STATION_ZOOM,
      layout: { visibility: subVis, "text-field": ["get", "label"], "text-size": 11,
        "text-font": ["Montserrat SemiBold"], "text-anchor": "bottom", "text-offset": [0, -0.35],
        "text-max-width": 14, "text-optional": true },
      paint: { "text-color": text, "text-halo-color": halo, "text-halo-width": 1.6 } });

    // plants above substations: small units aggregated per bus and fuel in a row left of
    // the substation; units >= 50 MW at their location
    add({ id: "plants-off", type: "circle", source: "plants", minzoom: 9, filter: ["!", ["get", "in_operation"]],
      layout: { visibility: vis("plants") },
      paint: { "circle-radius": 3, "circle-color": "rgba(255,255,255,0.4)", "circle-stroke-color": S.out_of_service,
        "circle-stroke-width": 1 } });
    // a column of fuel symbols left of the substation (slot_y = row from the centre)
    const aggSize = ["interpolate", ["linear"], ["zoom"], 8.5, 0.42, 11, 0.65];
    const rows = [...Array(17).keys()].map((k) => (k - 8) / 2);
    const byRow = (f) => ["case", ...rows.flatMap((r) => [["==", ["get", "slot_y"], r], ["literal", f(r)]]), ["literal", f(0)]];
    add({ id: "plant-aggregates", type: "symbol", source: "plant_aggregates", minzoom: S.label_zoom.plant_aggregate,
      layout: { visibility: vis("plants"), "icon-image": ["concat", "plant_", ["get", "fuel_category"]],
        "icon-size": aggSize, "icon-allow-overlap": true, "icon-ignore-placement": true,
        "icon-offset": byRow((r) => [-30, 26 * r]) },
      paint: { "icon-opacity": tso ? 0.4 : 0.95 } });
    add({ id: "plants", type: "symbol", source: "plants",
      filter: ["all", ["get", "in_operation"], ["==", ["get", "display"], "individual"]],
      layout: { visibility: vis("plants"),
        "icon-image": ["concat", "plant_", ["get", "fuel_category"]],
        "icon-size": plantIcon, "icon-allow-overlap": true,
        "symbol-sort-key": ["-", 0, ["get", "capacity_mw"]] },
      paint: { "icon-opacity": tso ? 0.4 : 1 } });
    add({ id: "plants-label", type: "symbol", source: "plants", minzoom: S.label_zoom.plant,
      filter: ["all", ["get", "in_operation"], [">=", ["get", "capacity_mw"], S.plants.label_mw]],
      layout: { visibility: vis("plants"), "text-field": ["concat", ["get", "name"], "\n",
        ["to-string", ["round", ["get", "capacity_mw"]]], " MW"], "text-size": 10,
        "text-font": ["Open Sans Italic"], "text-offset": [0, 1.3], "text-anchor": "top", "text-optional": true,
        "text-max-width": 12 },
      paint: { "text-color": text, "text-halo-color": halo, "text-halo-width": 1.4 } });
    add({ id: "plant-aggregates-label", type: "symbol", source: "plant_aggregates", minzoom: S.label_zoom.plant_aggregate_mw,
      layout: { visibility: vis("plants"), "text-field": ["concat", ["to-string", ["round", ["get", "capacity_mw"]]], " MW"],
        "text-size": 9, "text-font": ["Open Sans Regular"], "text-anchor": "right",
        "text-offset": byRow((r) => [-3.2, 1.88 * r]), "text-allow-overlap": true, "text-ignore-placement": true },
      paint: { "text-color": text, "text-halo-color": halo, "text-halo-width": 1.2 } });

    // renewables (inverter), display only: the model's one REGCA1 inverter per bus as a
    // proportional circle (area ∝ MW) at the substation, coloured by its largest share
    // (ENTSO-E fuel colours); offshore wind farms as turbine symbols (OpenInfraMap) with a
    // schematic dashed line to the landing substation the model attaches them to
    if (state.data.renewable_inverters && state.data.renewable_inverters.features.length) {
      const rv = vis("renewables");
      const renewColour = ["match", ["get", "dominant"], ...Object.entries(RENEW).flatMap(([k, v]) => [k, v.colour]), "#9e9e9e"];
      add({ id: "offshore-links", type: "line", source: "offshore_links", minzoom: 5,
        layout: { visibility: rv, "line-cap": "round" },
        paint: { "line-color": RENEW.wind_offshore.colour, "line-width": ["interpolate", ["linear"], ["zoom"], 5, 0.8, 10, 1.6],
          "line-dasharray": [2, 2], "line-opacity": 0.6 } });
      add({ id: "renew-inverters", type: "circle", source: "renewable_inverters",
        filter: [">", ["get", "capacity_mw"], 0], layout: { visibility: rv, "circle-sort-key": ["-", 0, ["get", "capacity_mw"]] },
        paint: { "circle-radius": ["interpolate", ["linear"], ["zoom"], 5, ["*", 0.17, ["sqrt", ["get", "capacity_mw"]]],
          9, ["*", 0.34, ["sqrt", ["get", "capacity_mw"]]], 12, ["*", 0.45, ["sqrt", ["get", "capacity_mw"]]]],
          "circle-color": renewColour, "circle-opacity": 0.55, "circle-stroke-color": renewColour,
          "circle-stroke-width": 1, "circle-stroke-opacity": 0.95 } });
      add({ id: "offshore-farms", type: "symbol", source: "offshore_farms",
        layout: { visibility: rv, "icon-image": "wind_offshore", "icon-allow-overlap": true,
          "icon-size": ["interpolate", ["linear"], ["zoom"], 5, ["+", 0.3, ["*", 0.012, ["sqrt", ["get", "capacity_mw"]]]],
            10, ["+", 0.55, ["*", 0.02, ["sqrt", ["get", "capacity_mw"]]]]],
          "symbol-sort-key": ["-", 0, ["get", "capacity_mw"]] } });
      add({ id: "offshore-farms-label", type: "symbol", source: "offshore_farms", minzoom: 7.5,
        layout: { visibility: rv, "text-field": ["concat", ["get", "name"], "\n", ["to-string", ["round", ["get", "capacity_mw"]]], " MW"],
          "text-size": 10, "text-font": ["Open Sans Italic"], "text-offset": [0, 1.4], "text-anchor": "top",
          "text-optional": true, "text-max-width": 12 },
        paint: { "text-color": text, "text-halo-color": halo, "text-halo-width": 1.4 } });
    }

    // border terminals: a double arrow across the border, turned along the line (outward to the
    // right); the bold head is the typical measured direction (export / import), grey: no data
    add({ id: "terminals", type: "symbol", source: "border_terminals",
      layout: { visibility: vis("grid"),
        "icon-image": ["concat", ["match", ["get", "kind"], "hvdc", "hvdc_terminal_", "border_"], ["get", "direction"]],
        "icon-size": ["interpolate", ["linear"], ["zoom"], 5, 0.8, 8, 1.0, 12, 1.3],
        "icon-rotate": ["-", ["get", "rotate"], 90], "icon-rotation-alignment": "map",
        "icon-allow-overlap": true, "icon-ignore-placement": true,
        "text-field": ["step", ["zoom"], ["get", "short"], 8, ["get", "label"]],
        "text-size": ["step", ["zoom"], 9.5, 8, 10.5], "text-font": ["Montserrat SemiBold"],
        "text-offset": [0, 1.5], "text-anchor": "top", "text-optional": true, "text-padding": 6,
        "text-max-width": 16, "symbol-sort-key": ["-", 0, ["get", "circuits_n"]] },
      paint: { "text-color": text, "text-halo-color": halo, "text-halo-width": 1.6 } });
    if (!terminals) add({ id: "neighbours", type: "symbol", source: "neighbours",
      layout: { visibility: vis("grid"), "icon-image": "neighbour", "icon-allow-overlap": true,
        "icon-size": ["interpolate", ["linear"], ["zoom"], 5, 0.35, 10, 0.7],
        "text-field": ["get", "bidding_zone"], "text-size": 10.5, "text-font": ["Montserrat SemiBold"],
        "text-offset": [0, 1.1], "text-anchor": "top" },
      paint: { "text-color": text, "text-halo-color": halo, "text-halo-width": 1.4 } });
  }

  const OVERLAY_IDS = ["zones-fill", "zones-line", "zones-delu", "areas-fill", "areas-line", "plant-links",
    "osm-subs-fill", "osm-subs-line", "towers", "towers-cable", "towers-ref", "osm-halo", "osm-lines", "osm-label", "circuit-hl",
    "circuit-hl-core", "circuit-hl-dash", "stations-fill", "stations-outline", "stations-outline-box",
    "lines-straight", "lines-hvdc", "buses", "tees", "substations", "tees-label",
    "transformers-mid", "transformers", "substations-label", "stations-label",
    "plants-off", "plant-aggregates", "plants", "plants-label", "plant-aggregates-label", "offshore-links", "renew-inverters", "offshore-farms", "offshore-farms-label",
    "neighbours", "terminals"];
  function restyle() {
    // simplest robust way: remove the overlay and add it again with the new mode
    OVERLAY_IDS.forEach((id) => { if (map.getLayer(id)) map.removeLayer(id); });
    addOverlay();
    renderLegend();
  }

  // ---------------------------------------------------------------- popups
  const FIELDS = {
    lines: ["name", "eic", "voltage_kv", "circuits", "circuit_no", "same_route", "length_km", "r_ohm", "x_ohm", "b_us", "imax_a",
      "rating_mva", "tso", "tso_code", "bus1", "bus2", "is_interconnector", "tso_tie", "neighbour",
      "from_zone", "to_zone", "hvdc", "hvdc_operator", "capacity_mw", "dc_kv", "commissioned", "hvdc_source", "hvdc_route", "published_length_km", "route_quality", "route_note", "route_status", "route_source", "end_gap_m", "display_snapped_m", "status", "data_quality", "mrid"],
    substations: ["name", "site", "kind", "voltages_kv", "coupling", "tso", "country", "bidding_zone", "transformers", "buses",
      "location_source", "location_flag", "display_relocated_m", "footprint", "mrid"],
    "stations-fill": ["name", "site", "voltages_kv", "transformers", "tso", "source", "osm_ids"],
    tees: ["name", "site", "kind", "voltages_kv", "tso", "buses", "junction_snap_m"],
    buses: ["bus", "kind", "voltage_kv", "tso", "tso_codes", "country", "bidding_zone", "andes_idx",
      "andes_generators", "andes_gen_sn_mva", "andes_gen_models", "andes_gen_h_s", "location_source", "bus_mrid"],
    transformers: ["name", "eic", "site", "bus_hv", "bus_lv", "kv_hv", "kv_lv", "sn_mva", "x_ohm", "phase_shifter",
      "site_transformers", "tso", "data_quality"],
    "transformers-mid": ["name", "site", "kv_hv", "kv_lv", "sn_mva", "phase_shifter", "site_transformers", "tso"],
    "plant-aggregates": ["bus", "site", "fuel_category", "capacity_mw", "units", "names", "tso"],
    "plant-links": ["name", "bus", "capacity_mw", "length_m", "mastr_id"],
    plants: ["name", "operator", "fuel", "capacity_mw", "status", "in_operation", "reserve", "voltage_level",
      "bus", "andes_idx", "attach_method", "tso", "location", "mastr_id"],
    "plants-off": ["name", "fuel", "capacity_mw", "status", "reserve", "bus", "mastr_id"],
    neighbours: ["name", "bus", "voltage_kv", "country", "bidding_zone", "tie_lines", "tso"],
    towers: ["name", "ref", "operator", "voltages", "circuits_tag", "circuits_max", "cables",
      "location", "construction", "length_km", "osm_id"],
  };
  const fmt = (v) => (v === null || v === undefined || v === "" ? "–"
    : typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(Math.abs(v) < 10 ? 3 : 1)) : String(v));
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const isCable = (p) => /underground|underwater|cable/i.test(String(p.location || ""));
  let diagrams = null;
  function diagramOf(site) {
    if (!diagrams) {
      diagrams = {};
      state.data.substations.features.forEach(({ properties: p }) => { if (p.diagram) diagrams[p.site] = p.diagram; });
    }
    return diagrams[site] || "";
  }
  // the model circuits on an OSM tower line, each a link that highlights the circuit
  const CHECK_TEXT = { match: "model = OSM", model_fewer: "model carries fewer circuits than OSM",
    model_more: "model carries more circuits than OSM", no_model: "no model circuit on this way", no_tag: "OSM circuits not tagged",
    short: "short piece (under 150 m): the check of the corridor it belongs to applies",
    explained: "explained difference" };
  function circuitLink(c) {
    return `<a href="#" class="circ" data-mrid="${esc(c.mrid || "")}" data-name="${esc(c.name || "")}">${esc(c.name)}</a>`;
  }
  function osmHtml(p) {
    let circ = [];
    try { circ = JSON.parse(p.model_circuits || "[]"); } catch (e) { circ = []; }
    const list = circ.map((c) => `<li>${circuitLink(c)} <small>${esc(c.eic || "")} · ${esc(fmt(c.kv))} kV</small></li>`).join("");
    const rows = [["name", p.name], ["ref", p.ref], ["OSM circuits", p.circuits_tag], ["voltage", p.kv + " kV"],
      ["operator", p.operator], ["length km", p.length_km], ["OSM way", p.way_id]]
      .map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(fmt(v))}</td></tr>`).join("");
    const cls = p.check === "explained" && p.check_reason ? `<p class="note">${esc(p.check_reason)}</p>`
      : p.check && p.check !== "match" ? `<p class="flag">${esc(CHECK_TEXT[p.check] || p.check)} (OSM ${esc(fmt(p.circuits_tag))}, model ${esc(fmt(p.model_n))})${p.check_reason ? ": " + esc(p.check_reason) : ""}</p>` : "";
    const building = p.construction ? '<p class="note">OSM way tagged under construction / proposed; the model circuit uses it because no in-service OSM path exists.</p>' : "";
    if (PROD()) {  // production: the OSM way and whether the model uses it, no check
      const used = p.in_model ? "used by model circuits (click a station for its circuits)" : "no model circuit on this way";
      return `<div class="popup"><h3>${esc(p.name || "Tower line " + (p.ref || p.way_id))}</h3>${building}
        <p class="note">${esc(used)}</p><table>${rows}</table></div>`;
    }
    return `<div class="popup"><h3>${esc(p.name || "Tower line " + (p.ref || p.way_id))}</h3>${cls}${building}
      <h4>Model circuits on this way (${circ.length})</h4><ul class="carried">${list || "<li>none</li>"}</ul><table>${rows}</table></div>`;
  }
  // border / HVDC terminal: the tie circuits (links highlight them), the measured flow per TSO
  // segment over the v5 hours (positive = out of Germany) and the model note
  const sign = (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toLocaleString("en-GB");
  function terminalHtml(p) {
    const parse = (v) => { try { return JSON.parse(v || "[]"); } catch (e) { return []; } };
    const circ = parse(p.circuits), flows = parse(p.flows);
    const word = { export: "mostly export", import: "mostly import", "both directions": "both directions" };
    const flowRows = flows.map((f) => !f.hours
      ? `<li><b>${esc(f.tso)} – ${esc(f.zone)}</b>: no per-TSO ENTSO-E data for the v5 hours</li>`
      : `<li><b>${esc(f.tso)} – ${esc(f.zone)}</b>: ${esc(word[f.typical] || f.typical)}
          (${f.typical === "import" ? esc(f.zone) + " → DE" : f.typical === "export" ? "DE → " + esc(f.zone) : "DE ↔ " + esc(f.zone)}),
          export in ${Math.round(f.export_share * f.hours)} of ${f.hours} h<br><small>median ${sign(f.median_mw)} MW · range ${sign(f.min_mw)} … ${sign(f.max_mw)} MW</small></li>`).join("");
    const list = circ.map((c) => `<li>${c.mrid ? circuitLink(c) : esc(c.name)} <small>${c.kv ? esc(fmt(c.kv)) + " kV · " : ""}${esc(c.tso || "")}${c.node && p.site && p.site.includes(";") ? " · " + esc(c.node) : ""}</small></li>`).join("");
    const hv = p.kind === "hvdc" ? `<table>${[["operator", p.hvdc_operator], ["capacity MW", p.capacity_mw], ["kV DC", p.dc_kv],
      ["in service since", p.commissioned], ["far end", p.far_end], ["source", p.hvdc_source]]
      .map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(fmt(v))}</td></tr>`).join("")}</table>` : "";
    return `<div class="popup"><h3>${esc(p.label)}</h3>
      <h4>Measured flow (ENTSO-E A11 per TSO, v5 hours; + = out of Germany)</h4><ul class="carried">${flowRows}</ul>
      ${p.kind === "hvdc" ? "" : `<h4>Tie circuits (${circ.length})</h4><ul class="carried">${list}</ul>`}${hv}
      <p class="note">${esc(p.model_note || "")}</p></div>`;
  }
  function renewHtml(p) {
    const rows = Object.entries(RENEW).filter(([k]) => p[`${k}_mw`] > 0).map(([k, v]) =>
      `<tr><td><span style="color:${v.colour}">●</span> ${v.label}</td><td class="n">${fmt(Math.round(p[`${k}_mw`]))}</td><td class="n">${fmt(Math.round(p[`${k}_units`]))}</td></tr>`).join("");
    return `<div class="popup"><h3>${esc(p.bus)}</h3><p class="muted">Inverter-based infeed of the frozen model at this bus:
      one ${esc(p.andes_model || "inverter")} (ANDES)${p.in_case ? "" : ", bus outside the simulated network"}.
      Units are register units allocated to the bus (shares, so not whole numbers).</p>
      <table><tr><th>Type</th><th class="n">MW</th><th class="n">units</th></tr>${rows}
      <tr><td><b>Total</b></td><td class="n"><b>${fmt(Math.round(p.capacity_mw))}</b></td><td></td></tr>
      <tr><td>Dispatched in the model case</td><td class="n">${fmt(Math.round(p.dispatch_mw))}</td><td></td></tr></table>
      ${p.offshore_spread_note ? `<p class="${flagCls()}">${esc(p.offshore_spread_note)}</p>` : ""}
      ${p.offshore_farms_landing ? `<p class="muted">Offshore farms landing here: ${esc(p.offshore_farms_landing)}</p>` : ""}
      <p class="note">TSO ${esc(p.tso || "–")}. Source: Marktstammdatenregister via the model v5.1 (no new allocation).</p></div>`;
  }
  function farmHtml(p) {
    const row = (k, v) => `<tr><td>${esc(k)}</td><td>${esc(fmt(v))}</td></tr>`;
    const flag = p.check && p.check !== "ok" ? `<p class="flag">${esc(p.check)}</p>` : "";
    return `<div class="popup"><h3>${esc(p.name)}</h3>${flag}<table>
      ${row("capacity MW", p.capacity_mw)}${row("capacity from", p.capacity_source)}${row("turbines", p.turbines)}
      ${row("commissioned", p.commissioned)}${row("status", p.status)}
      ${row("export cable", p.cable || (PROD() ? "–" : "not sourced (needs owner)"))}${row("landing substation", p.landing || "–")}
      ${p.landing_model_bus ? row("landing is model bus", p.landing_model_bus) : ""}${p.upstream ? row("upstream model bus", p.upstream) : ""}
      ${row("register connection point", p.connection_point)}
      ${row("register MW / units on that point", `${fmt(p.connection_register_mw)} / ${fmt(p.connection_register_units)}`)}
      ${row("model attachment", p.model_attachment)}</table>
      ${p.model_note ? `<p class="${flagCls()}">${esc(p.model_note)}</p>` : ""}
      <p class="note">Farm: ${esc(p.farm_source)}<br>Cable: ${esc(p.cable_source || "–")}<br>Register: Marktstammdatenregister (dl-de/by-2-0) as held by the model v5.1.</p></div>`;
  }
  // production circuit popup: the public attributes only (no impedances, bus ids or checks)
  const STATUS_TEXT = { in_operation: "in operation", out_of_service: "out of service" };
  function prodLineHtml(p) {
    const n = (v, d) => (v === null || v === undefined || v === "" ? "–" : Number(v).toLocaleString("en-GB", { maximumFractionDigits: d }));
    const rows = [
      ["Voltage", p.hvdc ? `${n(p.dc_kv, 0)} kV DC` : `${n(p.voltage_kv, 0)} kV`],
      ["Circuits", n(p.circuits, 0)],
      ["Length", `${n(p.length_km, 1)} km`],
      p.hvdc ? ["Capacity", `${n(p.capacity_mw, 0)} MW`] : ["Rating", `${n(p.rating_mva, 0)} MVA`],
      ["TSO", p.hvdc_operator || p.tso || "–"],
      ["Status", STATUS_TEXT[p.status] || String(p.status || "–").replace(/_/g, " ")],
    ].map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("");
    return `<div class="popup"><h3>${esc(p.name || "Circuit")}</h3><table>${rows}</table></div>`;
  }

  function popupHtml(layer, props) {
    if (layer === "terminals") return terminalHtml(props);
    if (layer === "renew-inverters") return renewHtml(props);
    if (layer === "offshore-farms") return farmHtml(props);
    if (layer === "osm_lines") return osmHtml(props);
    if (layer === "lines" && PROD()) return prodLineHtml(props);
    if (layer === "towers") {
      const rows = FIELDS.towers.filter((k) => k in props).map((k) => `<tr><td>${esc(k)}</td><td>${esc(fmt(props[k]))}</td></tr>`).join("");
      // raw OSM layer: OSM tags only (the model check is in the grid layer, osm_lines)
      const under = isCable(props) ? '<p class="note">Underground / submarine cable (OSM location tag).</p>' : "";
      const building = props.construction ? '<p class="note">OSM: under construction / proposed (not in operation).</p>' : "";
      // reviewed note (export --way-notes), e.g. the expansion project of a way under construction
      const note = props.note ? `<p class="note">${esc(props.note)}</p>` : "";
      return `<div class="popup"><h3>${esc(props.name || "Tower line " + (props.ref || props.osm_id))}</h3>${building}${note}${under}<table>${rows}</table></div>`;
    }
    const keys = FIELDS[layer] || Object.keys(props);
    const title = props.name || props.bus || props.site || layer;
    const rows = keys.filter((k) => k in props).map((k) => `<tr><td>${esc(k)}</td><td>${esc(fmt(props[k]))}</td></tr>`).join("");
    // single-line diagram of the station (pypowsybl SVG written by the export with --diagrams)
    const station = ["substations", "stations-fill", "transformers", "tees"].includes(layer);
    const sld = station && diagramOf(props.site);
    const link = sld ? `<p><a href="data/${esc(sld)}" target="_blank" rel="noopener">Single-line diagram (pypowsybl) ↗</a></p>` : "";
    // a circuit not on the OSM graph (dashed): why, where the OSM path breaks, any override
    const note = layer === "lines" && props.route_note ? `<p class="${flagCls()}">${esc(props.route_note)}</p>` : "";
    return `<div class="popup"><h3>${esc(title)}</h3>${note}${link}<table>${rows}</table></div>`;
  }

  // ---------------------------------------------------------------- station panel
  // click a station box (or its symbol): name, operator, levels, model nodes, the
  // single-line diagram and the component tables from data/station_details.json
  // (grid_stations.write_station_details); without that file the plain popup is shown
  let details;
  async function stationDetails() {
    if (details === undefined) {
      details = null;
      try {
        const r = await fetch(dataUrl("data/station_details.json"));
        if (r.ok) details = await r.json();
      } catch (e) { details = null; }
    }
    return details;
  }
  const STATION_LAYERS = ["substations", "stations-fill", "tees", "stations-label"];
  function table(head, rows) {
    if (!rows.length) return '<p class="muted">none</p>';
    return `<table class="grid"><thead><tr>${head.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${
      rows.map((r) => `<tr>${r.map((v) => `<td>${v && v.html ? v.html : esc(fmt(v))}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  }
  function stationHtml(site, d, props) {
    const sld = d.diagram || diagramOf(site);
    const levels = {};
    d.nodes.forEach((n) => { (levels[n.kv] = levels[n.kv] || []).push(n.node); });
    const plantsShown = d.plants.slice(0, 25);
    const more = d.plants.length - plantsShown.length;
    const sev = { error: "flag-error", warning: "flag-warning" };
    return `<header><h3>${esc(d.name)}</h3><button type="button" class="close" aria-label="Close">×</button></header>
      <table class="kv"><tr><td>site</td><td>${esc(site)}</td></tr>
      <tr><td>kind</td><td>${esc(d.kind)}</td></tr>
      <tr><td>TSO (model)</td><td>${esc(d.tso || "–")}</td></tr>
      <tr><td>operator (OSM)</td><td>${esc(d.operator || "–")}</td></tr>
      ${d.osm_names ? `<tr><td>OSM name</td><td>${esc(d.osm_names)}</td></tr>` : ""}
      <tr><td>voltage levels</td><td>${esc(d.voltages_kv || "–")} kV</td></tr>
      <tr><td>footprint</td><td>${esc(d.footprint || "–")}</td></tr>
      <tr><td>model nodes</td><td>${d.nodes.map((n) => esc(n.node)).join("<br>") || "–"}</td></tr></table>
      ${d.flags.length && !PROD() ? `<h4>Data-quality flags (${d.flags.length})</h4><ul class="flags">${d.flags.map((f) =>
        `<li class="${sev[f.severity] || ""}"><b>${esc(f.check)}</b> <small>${esc(f.object)} · ${esc(f.severity)}</small><br>${esc(f.detail)}</li>`).join("")}</ul>` : ""}
      <h4>Single-line diagram</h4>
      ${sld ? `<div class="sld"><object type="image/svg+xml" data="data/${esc(sld)}" aria-label="Single-line diagram"></object></div>
        <p><a href="data/${esc(sld)}" target="_blank" rel="noopener">Open full size ↗</a> <small class="muted">pypowsybl, from the model</small></p>`
        : '<p class="muted">no diagram in this export (--diagrams)</p>'}
      <h4>Busbars per voltage level</h4>
      ${table(["kV", "model nodes"], Object.keys(levels).sort((a, b) => b - a).map((k) => [k, levels[k].join(", ")]))}
      <h4>Transformers (${d.transformers.length})</h4>
      ${table(["name", "kV", "MVA", "x Ω"], d.transformers.map((t) =>
        [t.name + (t.phase_shifter ? " (PST)" : ""), `${fmt(t.kv_hv)}/${fmt(t.kv_lv)}`, t.sn_mva, t.x_ohm]))}
      <h4>Connected circuits (${d.circuits.length})</h4>
      ${table(["circuit", "kV", "far end", "km", "MVA", "status"], d.circuits.map((c) =>
        [{ html: circuitLink(c) }, c.hvdc ? "HVDC" : c.kv, c.far_end, c.length_km, c.rating_mva, c.status === "in_operation" ? "" : c.status]))}
      <h4>Plants in operation (${d.plants.length}, ${fmt(Math.round(d.plants_mw))} MW)</h4>
      ${table(["name", "fuel", "MW"], plantsShown.map((p) => [p.name, p.fuel, p.mw]))}
      ${more > 0 ? `<p class="muted">+ ${more} smaller units</p>` : ""}
      <p class="muted">Plant list of the model (conventional, hydro, pumped storage and large batteries); the
      register-aggregated inverter infeed (solar, wind, small storage) is in the next section.</p>
      ${renewSection(d)}${machineSection(d)}`;
  }
  // the model's machines at the station's buses (production builds: from the frozen model)
  function machineSection(d) {
    if (!d.machines) return "";
    const on = d.machines.filter((m) => m.in_service);
    const mw = on.reduce((a, m) => a + (m.max_mw || 0), 0);
    return `<h4>Model machines (${d.machines.length}; in service ${on.length}, ${fmt(Math.round(mw))} MW)</h4>
      ${table(["bus", "fuel", "type", "units", "max MW", "dispatch MW", ""], d.machines.map((m) =>
        [m.node, m.fuel, m.type, fmt(Math.round(m.units)), m.max_mw, m.p_mw, m.in_service ? "" : "out of service"]))}
      <p class="muted">Machines of the frozen model v5.1 at these buses (synchronous machines and the inverter units).</p>`;
  }
  // renewables (inverter) of the station: the renewable_inverters rows of its model buses (C14b)
  function renewSection(d) {
    const fc = state.data.renewable_inverters;
    if (!fc) return '<h4>Renewables (inverter)</h4><p class="muted">renewables layer not in this export</p>';
    const own = new Set(d.nodes.map((n) => n.node));
    const rows = fc.features.map((f) => f.properties).filter((q) => own.has(q.bus));
    const r1 = (v) => fmt(Math.round(v || 0));
    const head = ["bus", ...Object.values(RENEW).map((v) => `${v.label} MW / units`), "total MW", "dispatched MW"];
    const body = rows.map((q) => [q.bus, ...Object.keys(RENEW).map((k) => `${r1(q[`${k}_mw`])} / ${r1(q[`${k}_units`])}`),
      r1(q.capacity_mw), q.dispatch_mw == null ? "–" : r1(q.dispatch_mw)]);
    const total = rows.reduce((a, q) => a + (q.capacity_mw || 0), 0);
    const notes = rows.flatMap((q) => [
      q.offshore_farms_landing ? `<p class="muted">${esc(q.bus)}: offshore farms landing here: ${esc(q.offshore_farms_landing)}</p>` : "",
      q.offshore_spread_note ? `<p class="${flagCls()}">${esc(q.bus)}: ${esc(q.offshore_spread_note)}</p>` : ""]).join("");
    return `<h4>Renewables (inverter) (${rows.length ? fmt(Math.round(total)) + " MW" : "none"})</h4>
      ${rows.length ? table(head, body) + notes + '<p class="muted">Model v5.1 inverters (REGCA1) per bus, Marktstammdatenregister via the model; dispatched = the model case hour.</p>'
        : '<p class="muted">none (no inverter at this bus in the model)</p>'}`;
  }
  function stationPanel() {
    let el = document.getElementById("station-panel");
    if (!el) {
      el = document.createElement("aside");
      el.id = "station-panel";
      el.setAttribute("aria-label", "Station details");
      document.body.appendChild(el);
      el.addEventListener("click", (ev) => { if (ev.target.closest(".close")) el.classList.remove("open"); });
    }
    return el;
  }
  async function openStation(site, props, lngLat) {
    const all = await stationDetails();
    const d = all && all[site];
    if (!d) {  // older export: the plain popup
      new maplibregl.Popup({ maxWidth: "380px" }).setLngLat(lngLat).setHTML(popupHtml("substations", props)).addTo(map);
      return;
    }
    const el = stationPanel();
    el.innerHTML = stationHtml(site, d, props);
    el.classList.add("open");
    el.scrollTop = 0;
  }

  // lines-ac / lines-casing: the production circuit layers (absent in staging)
  const clickable = ["lines-ac", "lines-casing", "terminals", "offshore-farms", "renew-inverters", "neighbours", "plants", "plants-off", "plant-aggregates", "substations", "tees", "transformers",
    "transformers-mid", "stations-label", "stations-fill", "buses", "plant-links", "circuit-hl", "circuit-hl-dash",
    "lines-straight", "lines-hvdc", "osm-lines", "osm-halo", "towers", "towers-cable", "osm-subs-fill"];

  // ---------------------------------------------------------------- one circuit
  // highlight a circuit (by mrid or name) station to station and open its popup
  function showCircuit(mrid, name) {
    const feats = state.data.lines.features.filter((f) => (mrid ? f.properties.mrid === mrid : f.properties.name === name));
    if (!feats.length) return;
    state.selected = feats[0].properties.mrid;
    const sel = ["==", ["get", "mrid"], state.selected];
    setSelection(sel, feats[0].properties.name);
    const pts = feats.flatMap((f) => (!f.geometry ? [] : f.geometry.type === "Point" ? [f.geometry.coordinates]
      : f.geometry.type === "LineString" ? f.geometry.coordinates : f.geometry.coordinates.flat()));
    if (pts.length) {
      const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
      map.fitBounds([[Math.min(...xs), Math.min(...ys)], [Math.max(...xs), Math.max(...ys)]], { padding: 80, maxZoom: 12.5 });
      const props = (feats.find((f) => f.properties.label_piece) || feats[0]).properties;
      new maplibregl.Popup({ maxWidth: "380px" }).setLngLat(pts[Math.floor(pts.length / 2)]).setHTML(popupHtml("lines", props)).addTo(map);
    }
  }
  function setSelection(sel, name) {
    if (map.getLayer("circuit-hl")) {
      map.setFilter("circuit-hl", ["all", sel, ["!", dashed]]);
      map.setFilter("circuit-hl-core", ["all", sel, ["!", dashed]]);
      map.setFilter("circuit-hl-dash", ["all", sel, dashed]);
    }
    let el = document.getElementById("selection");
    if (!el) {
      el = document.createElement("div");
      el.id = "selection";
      el.setAttribute("role", "status");
      document.body.appendChild(el);
      el.addEventListener("click", (ev) => { if (ev.target.closest(".clear")) clearSelection(); });
    }
    el.innerHTML = name ? `<span class="sw"></span>Selected circuit: <b>${esc(name)}</b>
      <button type="button" class="clear" aria-label="Clear selection">Clear ×</button>` : "";
    el.hidden = !name;
  }
  function clearSelection() {
    state.selected = "";
    setSelection(["==", ["get", "mrid"], ""], "");
  }
  document.addEventListener("keydown", (ev) => { if (ev.key === "Escape" && state.selected) clearSelection(); });
  document.addEventListener("click", (ev) => {
    const a = ev.target.closest && ev.target.closest("a.circ");
    if (!a) return;
    ev.preventDefault();
    showCircuit(a.dataset.mrid, a.dataset.name);
  });
  map.on("error", (e) => { console.error("map error", e.error && e.error.message); });
  map.on("click", (e) => {
    const hits = map.queryRenderedFeatures([[e.point.x - 4, e.point.y - 4], [e.point.x + 4, e.point.y + 4]],
      { layers: clickable.filter((id) => map.getLayer(id)) });
    if (!hits.length) { if (state.selected) clearSelection(); return; }
    // a circuit or plant above a box wins; a click on the box itself opens the station panel
    // (an OSM tower line inside the box does not: the box wins over the station's own wiring)
    const boxLayers = ["stations-fill", "stations-label"];
    const box = hits.find((h) => boxLayers.includes(h.layer.id));
    const f = hits.find((h) => !boxLayers.includes(h.layer.id) && !(box && h.layer.id.startsWith("towers"))) || box || hits[0];
    const layer = f.layer.id.startsWith("lines") || f.layer.id.startsWith("circuit-hl") ? "lines"
      : f.layer.id.startsWith("osm-") ? "osm_lines" : f.layer.id === "towers-cable" ? "towers" : f.layer.id;
    if (STATION_LAYERS.includes(layer) && f.properties.site) {
      openStation(String(f.properties.site), f.properties, e.lngLat);
      return;
    }
    new maplibregl.Popup({ maxWidth: "380px" }).setLngLat(e.lngLat).setHTML(popupHtml(layer, f.properties)).addTo(map);
  });
  map.on("mousemove", (e) => {
    const ids = clickable.filter((id) => map.getLayer(id));
    const n = map.queryRenderedFeatures([[e.point.x - 4, e.point.y - 4], [e.point.x + 4, e.point.y + 4]], { layers: ids }).length;
    map.getCanvas().style.cursor = n ? "pointer" : "";
  });

  // ---------------------------------------------------------------- production panel
  // compact panel of the production profile (staging keeps the index.html panel as is):
  // segmented controls, search, collapsible Layers / Legend / Summary, sources under About
  const PROD_LABELS = { grid: "Lines", substations: "Substations", transformers: "Transformers", buses: "Buses", plants: "Plants",
    renewables: "Renewables", control_areas: "TSOs", bidding_zones: "Zones" };
  const FUEL_SHORT = { hard_coal: "Coal", hydro: "Hydro", pumped_storage: "Pumped", storage: "Battery" };
  const RENEW_SHORT = { solar: "Solar", wind_onshore: "Wind", wind_offshore: "Offshore", storage: "Battery" };
  function prodPanel() {
    document.body.classList.add("prod");
    const panel = document.getElementById("panel");
    const startCollapsed = true;  // clean overview: the panel opens on the ☰ button
    panel.innerHTML = `
      <button type="button" id="panel-toggle" aria-expanded="${!startCollapsed}" aria-controls="panel-body" title="Show or hide the panel">
        <span aria-hidden="true">☰</span><span class="pt-label">Layers</span></button>
      <div id="panel-body">
        <div class="seg-row">
          <div class="segmented" role="radiogroup" aria-label="Colouring">
            <button type="button" data-mode="voltage" class="active">Voltage</button>
            <button type="button" data-mode="tso">TSO</button>
          </div>
          <div class="segmented" role="radiogroup" aria-label="Basemap">
            <button type="button" data-basemap="light" class="active">Light</button>
            <button type="button" data-basemap="dark">Dark</button>
          </div>
        </div>
        <label class="search"><span class="sr-only">Search</span>
          <input id="search" type="search" placeholder="Search" autocomplete="off"></label>
        <ul id="results"></ul>
        <details open><summary>Layers</summary><div id="layers"></div></details>
        <details><summary>Legend</summary><div id="legend"></div></details>
        <details><summary>Summary</summary><div id="summary"></div></details>
        <details class="about"><summary>ⓘ About</summary><div id="sources"></div><p id="model-version"></p></details>
      </div>`;
    panel.classList.toggle("collapsed", startCollapsed);
    document.getElementById("panel-toggle").addEventListener("click", (ev) => {
      const c = panel.classList.toggle("collapsed");
      ev.currentTarget.setAttribute("aria-expanded", String(!c));
    });
  }
  function prodLegend() {
    // short groups with a small heading; long groups (fuels) in two columns
    const groups = [];
    let rows = null;
    const group = (title, cols) => { rows = []; groups.push({ title, cols, rows }); };
    const v = state.visible;
    const line = (c, label) => rows.push(`<div class="legend-row"><span class="ln" style="border-top-color:${c}"></span>${label}</div>`);
    const icon = (n, label) => rows.push(`<div class="legend-row"><img src="symbols/${n}.svg" alt="">${label}</div>`);
    const fill = (c, label, round) => rows.push(`<div class="legend-row"><span class="fill" style="border-color:${c};background:${c}${round ? "8c;border-radius:50%" : "33"}"></span>${label}</div>`);
    if (v.grid || v.substations || v.transformers) {
      group("Grid", 1);
      if (v.grid) {
        if (state.mode === "voltage") { line(S.voltage["380"], "380 kV"); line(S.voltage["220"], "220 kV"); }
        line(S.voltage.hvdc, "HVDC");
      }
      if (v.substations) icon("substation_380", "Substation");
      if (v.grid) icon(state.data.border_terminals.features.length ? "border_export" : "neighbour", "Border");
      if (v.transformers) { icon("transformer_380_220", "Transformer"); icon("phase_shifter", "Phase shifter"); }
    }
    if (state.mode === "tso" || v.control_areas || v.bidding_zones) {
      group("Areas", 2);
      if (state.mode === "tso" || v.control_areas) TSOS.forEach((t) => fill(S.tso[t], t));
      if (v.bidding_zones) rows.push('<div class="legend-row"><span class="fill" style="border-color:#212121"></span>Zone</div>');
    }
    if (v.plants) { group("Plants", 2); Object.entries(S.fuels).forEach(([k, f]) => icon(`plant_${k}`, FUEL_SHORT[k] || f.label)); }
    if (v.renewables) { group("Renewables", 2); Object.entries(RENEW).forEach(([k, r]) => fill(r.colour, RENEW_SHORT[k] || r.label, true)); }
    document.getElementById("legend").innerHTML = groups.filter((g) => g.rows.length).map((g) =>
      `<div class="legend-group"><div class="legend-head">${g.title}</div><div class="legend-grid c${g.cols}">${g.rows.join("")}</div></div>`).join("")
      || '<p class="muted">–</p>';
  }

  // ---------------------------------------------------------------- panel
  function renderLayers() {
    const box = document.getElementById("layers");
    if (PROD()) {
      box.innerHTML = Object.entries(PROD_LABELS).map(([n, label]) =>
        `<label class="layer-row"><input type="checkbox" data-layer="${n}" ${state.visible[n] ? "checked" : ""}> ${label}</label>`).join("");
      box.querySelectorAll("input").forEach((el) => el.addEventListener("change", () => {
        state.visible[el.dataset.layer] = el.checked;
        restyle();
      }));
      return;
    }
    box.innerHTML = panelGroups().map((g) => `<h3 class="layer-group">${esc(g.title)}</h3>` + g.layers.map((n) =>
      `<label class="layer-row"><input type="checkbox" data-layer="${n}" ${state.visible[n] ? "checked" : ""}> ${PROD() && PROD_TITLES[n] ? PROD_TITLES[n] : TITLES[n]}</label>`).join("")).join("");
    box.querySelectorAll("input").forEach((el) => el.addEventListener("change", () => {
      state.visible[el.dataset.layer] = el.checked;
      restyle();
    }));
  }

  function renderLegend() {
    if (PROD()) { prodLegend(); return; }
    const rows = [];
    const line = (c, label, dash) => rows.push(`<div class="legend-row"><span class="ln" style="border-top-color:${c};border-top-style:${dash || "solid"}"></span>${label}</div>`);
    const icon = (n, label) => rows.push(`<div class="legend-row"><img src="symbols/${n}.svg" alt="">${label}</div>`);
    if (state.mode === "voltage") {
      line(S.voltage["380"], "380 kV circuit (parallel circuits side by side)");
      line(S.voltage["220"], "220 kV circuit");
      icon("substation_380", "Substation 380 kV");
      icon("substation_220", "Substation 220 kV");
      icon("substation_coupling", "Coupling substation 380/220 kV");
    } else {
      TSOS.forEach((t) => rows.push(`<div class="legend-row"><span class="fill" style="border-color:${S.tso[t]};background:${S.tso[t]}33"></span>${t}</div>`));
      line(S.interconnector, "Cross-border interconnector");
      rows.push(`<div class="legend-row"><span class="fill" style="border-color:#212121"></span>DE-LU bidding zone</div>`);
    }
    rows.push(PROD() ? '<p class="muted">Symbol size grows with the number of transformers.</p>'
      : '<p class="muted">Symbol size grows with the number of transformers. “2×” = circuits on the tower line.</p>');
    if (state.hasWays && PROD()) {
      rows.push('<p class="muted">Lines: one line per OSM tower line (way) the model circuits run on; search a circuit or click a station to highlight a circuit station to station.</p>');
      rows.push(`<div class="legend-row"><span class="ln" style="border-top:1px solid ${S.voltage["380"]};opacity:.45"></span>OSM tower line with no model circuit</div>`);
    } else if (state.hasWays) {
      rows.push('<p class="muted">Lines: one line per OSM tower line (way); click it for the model circuits it carries, and a circuit name to highlight that circuit station to station.</p>');
      rows.push(`<div class="legend-row"><span class="ln" style="border-top:6px solid ${S.check_fewer || "#ff9800"}88"></span>Model carries fewer circuits than OSM (unexplained)</div>`);
      rows.push(`<div class="legend-row"><span class="ln" style="border-top:6px solid ${S.check_more || "#8e24aa"}88"></span>Model carries more circuits than OSM (unexplained)</div>`);
      rows.push(`<div class="legend-row"><span class="ln" style="border-top:1px solid ${S.voltage["380"]};opacity:.45"></span>No model circuit on this tower line</div>`);
    }
    rows.push(`<div class="legend-row"><span class="ln" style="border-top:6px solid ${SELECTED};box-shadow:inset 0 -2px 0 #fff"></span>Selected circuit (click a circuit name; Esc or the Clear button clears it)</div>`);
    if (PROD()) {
      rows.push('<p class="muted">Lines: one line per model circuit, station to station; click a line or a station for its data.</p>');
      line(S.voltage.hvdc, "HVDC link (to the German converter)");
    } else {
      line(S.voltage.hvdc, "HVDC link", "dashed");
      line(S.neutral, "Model circuit with no OSM tower line (dashed)", "dashed");
    }
    if (state.visible.osm_assets) line("#9e9e9e", "OpenStreetMap power line, raw (grey; ref label)");
    rows.push(`<div class="legend-row"><span class="fill" style="border-color:${S.voltage["380"]};background:${S.voltage["380"]}24"></span>Station (${PROD() ? "OSM footprint or fixed box" : "OSM footprint; dashed: fixed box"}), outline by highest voltage; click for the single-line diagram and components</div>`);
    rows.push(`<div class="legend-row"><span class="dot"></span>Junction (tee) where circuit legs meet</div>`);
    icon("transformer_380_220", "Transformer 380/220 kV");
    icon("transformer_380_110", "Transformer to 110 kV");
    icon("phase_shifter", "Phase-shifting transformer");
    if (state.data.border_terminals.features.length) {
      icon("border_export", "Border terminal: tie lines end at the German border (bold head: typical measured direction; orange export, blue import)");
      icon("hvdc_terminal_nodata", "HVDC terminal at the German converter (sea / foreign route not drawn)");
    } else icon("neighbour", "Neighbouring network");
    rows.push(`<h2>Power plants (size ∝ √MW)</h2><p class="muted">Units ≥ ${S.plants.individual_mw} MW at their site, grey line to the connection substation; smaller units summed per bus and fuel in a row left of the substation.</p>`);
    Object.entries(S.fuels).forEach(([k, f]) => icon(`plant_${k}`, f.label));
    if (state.visible.renewables) {
      rows.push('<h2>Renewables (inverter), model v5.1</h2><p class="muted">One circle per bus: the model\'s inverter (REGCA1) capacity, area ∝ MW, colour by the largest type; click for the split.</p>');
      Object.values(RENEW).forEach((v) => rows.push(`<div class="legend-row"><span class="fill" style="border-color:${v.colour};background:${v.colour}8c;border-radius:50%"></span>${v.label}</div>`));
      icon("wind_offshore", PROD() ? "Offshore wind farm (size by MW); line to its sourced landing substation"
        : "Offshore wind farm (size by MW); dashed: to the landing substation of the model");
    }
    document.getElementById("legend").innerHTML = rows.join("");
  }

  function renderSummary() {
    const km = {}, subs = {}, mw = {};
    TSOS.forEach((t) => { km[t] = { 380: 0, 220: 0 }; subs[t] = 0; mw[t] = 0; });
    state.data.lines.features.forEach(({ properties: p }) => {
      if (!km[p.tso] || p.label_piece === false || p.hvdc || p.status !== "in_operation") return;
      km[p.tso][p.voltage_kv >= 300 ? 380 : 220] += p.length_km || 0;
    });
    state.data.substations.features.forEach(({ properties: p }) => {
      if (p.kind === "substation" && p.country === "DEU" && subs[p.tso] !== undefined) subs[p.tso] += 1;
    });
    state.data.plants.features.forEach(({ properties: p }) => {
      if (p.in_operation && mw[p.tso] !== undefined) mw[p.tso] += p.capacity_mw || 0;
    });
    const n = (v) => Math.round(v).toLocaleString("en-GB");
    if (PROD()) {
      document.getElementById("summary").innerHTML = `<table class="compact"><thead><tr><th>TSO</th><th class="n">380 km</th>
        <th class="n">220 km</th><th class="n">Subst.</th><th class="n">MW</th></tr></thead><tbody>${
        TSOS.map((t) => `<tr><td><span style="color:${S.tso[t]}">■</span> ${t}</td><td class="n">${n(km[t][380])}</td>
        <td class="n">${n(km[t][220])}</td><td class="n">${subs[t]}</td><td class="n">${n(mw[t])}</td></tr>`).join("")}</tbody></table>`;
      return;
    }
    document.getElementById("summary").innerHTML = `<table><thead><tr><th>TSO</th><th class="n">380 kV km</th>
      <th class="n">220 kV km</th><th class="n">Subst.</th><th class="n">Plants MW</th></tr></thead><tbody>${
      TSOS.map((t) => `<tr><td><span style="color:${S.tso[t]}">■</span> ${t}</td><td class="n">${n(km[t][380])}</td>
      <td class="n">${n(km[t][220])}</td><td class="n">${subs[t]}</td><td class="n">${n(mw[t])}</td></tr>`).join("")}</tbody></table>
      <p class="muted">Circuit km (each circuit counted). Plants: MaStR units ≥ connection level of the model, in operation, by the TSO of their connection bus.</p>`;
  }

  function buildIndex() {
    const idx = [];
    const push = (layer, f, text, sub) => idx.push({ layer, f, text: text.toLowerCase(), label: text, sub });
    state.data.substations.features.forEach((f) => push("substations", f, `${f.properties.name || f.properties.site}`, f.properties.voltages_kv + " kV"));
    state.data.lines.features.filter((f) => f.properties.label_piece !== false).forEach((f) => push("lines", f, `${f.properties.name} ${f.properties.eic || ""}`, f.properties.eic || ""));
    state.data.buses.features.forEach((f) => push("buses", f, `${f.properties.bus} #${f.properties.andes_idx ?? ""}`, `ANDES ${f.properties.andes_idx ?? "–"}`));
    state.data.plants.features.forEach((f) => push("plants", f, `${f.properties.name} ${f.properties.mastr_id}`, `${Math.round(f.properties.capacity_mw)} MW`));
    return idx;
  }

  function centre(f) {
    const g = f.geometry;
    if (g.type === "Point") return g.coordinates;
    const c = g.coordinates;
    return c[Math.floor(c.length / 2)];
  }

  function wireSearch(index) {
    const input = document.getElementById("search");
    const list = document.getElementById("results");
    input.addEventListener("input", () => {
      const q = input.value.trim().toLowerCase();
      if (q.length < 2) { list.innerHTML = ""; return; }
      const hits = index.filter((r) => r.text.includes(q)).slice(0, 30);
      list.innerHTML = hits.map((r, i) => `<li data-i="${i}">${esc(r.label)} <small>${esc(r.layer)} · ${esc(r.sub)}</small></li>`).join("");
      list.querySelectorAll("li").forEach((li) => li.addEventListener("click", () => {
        const r = hits[Number(li.dataset.i)];
        const at = centre(r.f);
        map.flyTo({ center: at, zoom: Math.max(map.getZoom(), 10.5) });
        if (r.layer === "substations") openStation(String(r.f.properties.site), r.f.properties, at);
        else if (r.layer === "lines") showCircuit(r.f.properties.mrid, r.f.properties.name);
        else new maplibregl.Popup({ maxWidth: "380px" }).setLngLat(at).setHTML(popupHtml(r.layer, r.f.properties)).addTo(map);
      }));
    });
  }

  function wireButtons() {
    document.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => {
      state.mode = b.dataset.mode;
      document.querySelectorAll("[data-mode]").forEach((x) => x.classList.toggle("active", x === b));
      restyle();
    }));
    document.querySelectorAll("[data-basemap]").forEach((b) => b.addEventListener("click", () => {
      state.basemap = b.dataset.basemap;
      document.documentElement.dataset.theme = state.basemap;
      document.querySelectorAll("[data-basemap]").forEach((x) => x.classList.toggle("active", x === b));
      map.setStyle(BASEMAPS[state.basemap], { diff: false });
      map.once("style.load", async () => { await addImages(); addOverlay(); });
    }));
  }

  // URL parameters for reproducible views/screenshots: ?mode=tso&basemap=dark
  const params = new URLSearchParams(location.search);
  if (params.get("mode") === "tso") state.mode = "tso";
  if (params.get("basemap") === "dark") { state.basemap = "dark"; document.documentElement.dataset.theme = "dark"; }

  Promise.all([loadData(), new Promise((r) => map.once("load", r))]).then(async () => {
    if (state.basemap === "dark") {
      map.setStyle(BASEMAPS.dark, { diff: false });
      await new Promise((r) => map.once("style.load", r));
    }
    await addImages();
    addOverlay();
    if (PROD()) prodPanel();
    document.querySelectorAll("[data-mode]").forEach((x) => x.classList.toggle("active", x.dataset.mode === state.mode));
    document.querySelectorAll("[data-basemap]").forEach((x) => x.classList.toggle("active", x.dataset.basemap === state.basemap));
    renderLayers(); renderLegend(); renderSummary(); wireButtons(); wireSearch(buildIndex());
    const m = state.meta || {};
    if (PROD()) {
      // About: attribution (ODbL and the other licences require it) and the build id
      document.getElementById("sources").innerHTML = "Data: © <a href=\"https://www.openstreetmap.org/copyright\" target=\"_blank\" rel=\"noopener\">OpenStreetMap</a> contributors (ODbL), " +
        "JAO Static Grid Model, ENTSO-E, Marktstammdatenregister, Natural Earth.";
      const build = document.getElementById("model-version");
      build.id = "build-id";
      build.className = "muted";
      build.textContent = `Build ${m.staging_build_id || buildId || "–"} · model ${(m.source_sha256 || "–").slice(0, 12)}`;
    } else {
    document.getElementById("model-version").textContent =
      `Model ${(m.source_file || "").replace(/\.pkl$/, "")} · sha256 ${(m.source_sha256 || "").slice(0, 12)} · exported ${(m.exported_utc || "").slice(0, 10)}`;
    const build = document.createElement("p");
    build.id = "build-id";
    build.className = "muted";
    build.textContent = `${PROD() ? "" : "STAGING · "}Build ${m.staging_build_id || buildId || "–"} · model sha256 ${(m.source_sha256 || "–").slice(0, 12)}`;
    document.getElementById("sources").after(build);
    document.getElementById("sources").innerHTML =
      "Control areas are approximate (Voronoi cells of each TSO's buses). DK1/NO2 drawn with the whole country shape. " +
      "Connection/terminal buses (110/21 kV) sit at their substation. Data: JAO Core Static Grid Model, " +
      "OpenStreetMap contributors (ODbL) via PyPSA-Eur, Marktstammdatenregister, Natural Earth.";
    }
    document.querySelectorAll(".maplibregl-ctrl-attrib").forEach((el) => el.classList.remove("maplibregl-compact-show"));
    document.body.dataset.ready = "1";
    window.gridMap = map; // for the browser console and automated screenshots
    window.gridShowCircuit = showCircuit;
    window.gridOpenStation = (site) => openStation(site, {}, map.getCenter());
    window.gridSetLayer = (name, on) => { state.visible[name] = on; restyle(); renderLayers(); };
  }).catch((err) => {
    document.getElementById("summary").textContent = `Failed to load data: ${err.message}`;
  });
})();
