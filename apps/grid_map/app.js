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
  const PANEL_LAYERS = ["bidding_zones", "control_areas", "tower_lines", "lines", "buses", "transformers",
    "substations", "plants", "neighbours"];
  const TSOS = Object.keys(S.tso);
  const state = { mode: "voltage", basemap: "light", data: {}, visible: {
    bidding_zones: true, control_areas: true, tower_lines: true, lines: true, buses: false, transformers: true,
    substations: true, plants: true, neighbours: true,
  } };
  const TITLES = {
    lines: "Lines (AC circuits, HVDC)", tower_lines: "Tower lines (OpenStreetMap)", substations: "Substations", transformers: "Transformers",
    plants: "Power plants", buses: "Buses (model / ANDES ids)", neighbours: "Neighbouring networks",
    control_areas: "TSO control areas (approx.)", bidding_zones: "Bidding zones",
  };

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
  const lineWidth = lw();
  const casingWidth = lw(1.6);
  const slot = ["-", ["get", "circuit_no"], ["/", ["-", ["get", "circuits"], 1], 2]];
  const offset = ["interpolate", ["linear"], ["zoom"], ...Z.flatMap(([z, , , gap]) => [z, ["*", slot, gap]])];
  const live = ["==", ["get", "status"], "in_operation"];
  const zoomed = (stops, expr) => ["interpolate", ["linear"], ["zoom"],
    ...stops.flatMap(([z, f]) => [z, ["*", f, expr]])];
  const plantSize = ["interpolate", ["linear"], ["sqrt", ["max", ["get", "capacity_mw"], 0]],
    ...S.plants.size_stops.flat()];
  const plantIcon = zoomed(S.plants.zoom_factor, plantSize);
  const kvClass = (field) => ["case", [">=", ["get", field], 300], "380", [">=", ["get", field], 200], "220", "110"];
  const subBase = ["case", ["==", ["get", "symbol"], "220"], 0.78, 1.0];
  const subGrow = ["+", 1, ["*", 0.07, ["min", ["get", "transformers"], 6]]];
  const subSize = zoomed(S.substations.zoom_size, ["*", subBase, subGrow]);

  // ---------------------------------------------------------------- data and images
  async function loadData() {
    const entries = await Promise.all(LAYER_FILES.map(async (name) => {
      const r = await fetch(`data/${name}.geojson`);
      if (!r.ok) throw new Error(`data/${name}.geojson: ${r.status}`);
      return [name, await r.json()];
    }));
    entries.forEach(([k, v]) => { state.data[k] = v; });
    try {
      const meta = await (await fetch("data/metadata.json")).json();
      state.meta = meta;
    } catch (e) { state.meta = {}; }
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
      ...S.transformer_symbols, "phase_shifter", "neighbour"];
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
    for (const name of LAYER_FILES) {
      if (!map.getSource(name)) map.addSource(name, { type: "geojson", data: state.data[name] });
    }
    neutraliseBasemap();
    const vis = (name) => (state.visible[name] ? "visible" : "none");
    const dark = state.basemap === "dark";
    const halo = dark ? "#111417" : "#ffffff";
    const text = dark ? "#e8eaed" : "#212121";
    const add = (layer) => { if (!map.getLayer(layer.id)) map.addLayer(layer); };
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

    // plant connectors under the grid: thin grey from the plant to its substation
    add({ id: "plant-links", type: "line", source: "plant_connections", minzoom: 7,
      filter: ["get", "in_operation"], layout: { visibility: vis("plants"), "line-cap": "round" },
      paint: { "line-color": dark ? "#90a4ae" : "#607d8b",
        "line-width": ["interpolate", ["linear"], ["zoom"], 7, 0.5, 12, 1.4],
        "line-opacity": ["interpolate", ["linear"], ["zoom"], 7, 0.45, 10, 0.8] } });

    // physical tower lines (OpenStreetMap ways) as a thin grey casing under the circuits
    add({ id: "towers", type: "line", source: "tower_lines", minzoom: 9,
      layout: { visibility: vis("tower_lines"), "line-cap": "round", "line-join": "round" },
      paint: { "line-color": dark ? "#78909c" : "#9e9e9e",
        "line-width": ["interpolate", ["linear"], ["zoom"], 9, 0.6, 12, 1.4, 15, 2.4],
        "line-opacity": ["interpolate", ["linear"], ["zoom"], 9, 0.4, 11, 0.8] } });
    add({ id: "towers-ref", type: "symbol", source: "tower_lines", minzoom: 11.5,
      filter: ["!=", ["get", "ref"], ""],
      layout: { visibility: vis("tower_lines"), "symbol-placement": "line", "symbol-spacing": 400,
        "text-field": ["get", "ref"], "text-size": 9.5, "text-font": ["Open Sans Italic"],
        "text-offset": [0, 1.15], "text-max-angle": 35, "text-padding": 4 },
      paint: { "text-color": dark ? "#b0bec5" : "#616161", "text-halo-color": halo, "text-halo-width": 1.2 } });

    const lineVis = vis("lines");
    add({ id: "lines-oos", type: "line", source: "lines", filter: ["!", live],
      layout: { visibility: lineVis, "line-cap": "round" },
      paint: { "line-color": S.out_of_service, "line-width": 1.2, "line-dasharray": [1, 2], "line-offset": offset } });
    add({ id: "lines-casing", type: "line", source: "lines", minzoom: 6.5, filter: ["all", live, ["!", ["get", "hvdc"]]],
      layout: { visibility: lineVis, "line-cap": "round", "line-join": "round" },
      paint: { "line-color": halo, "line-width": casingWidth, "line-offset": offset, "line-opacity": 0.9 } });
    add({ id: "lines-ic-halo", type: "line", source: "lines",
      filter: ["all", live, ["get", "is_interconnector"]],
      layout: { visibility: tso ? lineVis : "none", "line-cap": "round" },
      paint: { "line-color": S.interconnector, "line-width": lw(3.2), "line-offset": offset, "line-opacity": 0.35 } });
    add({ id: "lines-straight", type: "line", source: "lines",
      filter: ["all", live, ["!", ["get", "hvdc"]], ["==", ["get", "route_quality"], "straight"]],
      layout: { visibility: lineVis, "line-cap": "butt" },
      paint: { "line-color": lineColour(), "line-width": lineWidth, "line-dasharray": [1.2, 1.2], "line-offset": offset } });
    add({ id: "lines-ac", type: "line", source: "lines",
      filter: ["all", live, ["!", ["get", "hvdc"]], ["!=", ["get", "route_quality"], "straight"]],
      layout: { visibility: lineVis, "line-cap": "round", "line-join": "round",
        "line-sort-key": ["get", "voltage_kv"] },
      paint: { "line-color": lineColour(), "line-width": lineWidth, "line-offset": offset } });
    add({ id: "lines-hvdc", type: "line", source: "lines", filter: ["all", live, ["get", "hvdc"]],
      layout: { visibility: lineVis },
      paint: { "line-color": S.voltage.hvdc, "line-width": lw(0, 1.1), "line-dasharray": [3, 1.6], "line-offset": offset } });
    // circuit count of a corridor (low/mid zoom) and one name per circuit (high zoom)
    add({ id: "lines-count", type: "symbol", source: "lines", minzoom: S.label_zoom.corridor, maxzoom: S.label_zoom.line,
      filter: ["all", live, ["==", ["get", "circuit_no"], 0], [">=", ["get", "circuits"], 2]],
      layout: { visibility: lineVis, "symbol-placement": "line-center", "text-field": ["concat", ["to-string", ["get", "circuits"]], "×"],
        "text-size": 10, "text-font": ["Open Sans Bold"], "text-rotation-alignment": "viewport", "text-padding": 6 },
      paint: { "text-color": text, "text-halo-color": halo, "text-halo-width": 1.6 } });
    add({ id: "lines-label", type: "symbol", source: "lines", minzoom: S.label_zoom.line,
      filter: ["all", live, ["get", "label_piece"]],
      layout: { visibility: lineVis, "symbol-placement": "line-center", "text-field": ["get", "name"],
        "text-size": 10.5, "text-font": ["Open Sans Regular"], "text-max-angle": 30, "text-padding": 8 },
      paint: { "text-color": text, "text-halo-color": halo, "text-halo-width": 1.4 } });

    add({ id: "buses", type: "circle", source: "buses", minzoom: 7,
      layout: { visibility: vis("buses") },
      paint: { "circle-radius": ["interpolate", ["linear"], ["zoom"], 7, 1.5, 12, 4],
        "circle-color": ["case", ["in", ["get", "kind"], ["literal", ["connection", "terminal"]]], "#bdbdbd",
          ...(tso ? [tsoColour]
            : [[">=", ["get", "voltage_kv"], 300], S.voltage["380"], S.voltage["220"]])],
        "circle-stroke-color": halo, "circle-stroke-width": 0.6 } });

    // substations above the lines: size by highest voltage and transformer count
    const ring = ["==", ["get", "symbol"], "tee"];
    const subIcon = !tso ? ["concat", "substation_", ["get", "symbol"]]
      : ["match", ["get", "tso"], ...TSOS.flatMap((t) => [t, `substation_${t.toLowerCase()}`]), "substation_380"];
    add({ id: "tees", type: "symbol", source: "substations", minzoom: 7.5, filter: ring,
      layout: { visibility: vis("substations"), "icon-image": "tee", "icon-allow-overlap": true,
        "icon-size": ["interpolate", ["linear"], ["zoom"], 7.5, 0.3, 12, 0.5] } });
    add({ id: "substations", type: "symbol", source: "substations", filter: ["!", ring],
      layout: { visibility: vis("substations"), "icon-image": subIcon, "icon-allow-overlap": true,
        "icon-ignore-placement": true, "icon-size": subSize, "symbol-sort-key": ["-", 0, ["get", "max_kv"]] } });

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
    add({ id: "transformers", type: "symbol", source: "transformers", minzoom: S.label_zoom.transformer_all,
      layout: { visibility: vis("transformers"), "icon-image": trIcon, "icon-size": trSize,
        "icon-offset": ["case", ...[0, 1, 2, 3, 4, 5, 6, 7].flatMap((k) =>
          [["==", ["get", "slot"], k], ["literal", [24 + 22 * k, -20]]]), ["literal", [24, -20]]],
        "icon-allow-overlap": true, "icon-ignore-placement": true } });

    add({ id: "substations-label", type: "symbol", source: "substations",
      filter: ["all", ["!", ring], ["any",
        ["all", [">=", ["get", "max_kv"], 300], [">=", ["zoom"], S.label_zoom.substation_380]],
        [">=", ["zoom"], S.label_zoom.substation_220]]],
      layout: { visibility: vis("substations"), "text-field": ["get", "site"], "text-size": 11,
        "text-font": ["Montserrat SemiBold"], "text-offset": [0.7, 0.45], "text-anchor": "top-left",
        "text-transform": "uppercase", "text-letter-spacing": 0.03, "text-optional": true },
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

    add({ id: "neighbours", type: "symbol", source: "neighbours",
      layout: { visibility: vis("neighbours"), "icon-image": "neighbour", "icon-allow-overlap": true,
        "icon-size": ["interpolate", ["linear"], ["zoom"], 5, 0.35, 10, 0.7],
        "text-field": ["get", "bidding_zone"], "text-size": 10.5, "text-font": ["Montserrat SemiBold"],
        "text-offset": [0, 1.1], "text-anchor": "top" },
      paint: { "text-color": text, "text-halo-color": halo, "text-halo-width": 1.4 } });
  }

  const OVERLAY_IDS = ["zones-fill", "zones-line", "zones-delu", "areas-fill", "areas-line", "plant-links",
    "towers", "towers-ref", "lines-oos", "lines-casing", "lines-ic-halo", "lines-straight", "lines-ac", "lines-hvdc", "lines-count",
    "lines-label", "buses", "tees", "substations", "transformers-mid", "transformers", "substations-label",
    "plants-off", "plant-aggregates", "plants", "plants-label", "plant-aggregates-label", "neighbours"];
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
      "from_zone", "to_zone", "hvdc", "route_quality", "route_source", "end_gap_m", "display_snapped_m", "status", "data_quality", "mrid"],
    substations: ["name", "site", "kind", "voltages_kv", "coupling", "tso", "country", "bidding_zone", "transformers", "buses",
      "location_source", "location_flag", "display_relocated_m", "mrid"],
    tees: ["name", "site", "kind", "voltages_kv", "tso", "buses"],
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
    towers: ["name", "ref", "operator", "voltages", "circuits_tag", "cables", "model_circuits", "mismatch",
      "location", "construction", "length_km", "osm_id"],
  };
  const fmt = (v) => (v === null || v === undefined || v === "" ? "–"
    : typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(Math.abs(v) < 10 ? 3 : 1)) : String(v));
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  function towerHtml(p) {
    const names = String(p.model_names || "").split("; ").filter(Boolean);
    const eics = String(p.model_eics || "").split("; ");
    const kvs = String(p.model_kv || "").split("; ");
    const list = names.map((n, i) => `<li>${esc(n)} <small>${esc(eics[i] || "")} · ${esc(kvs[i] || "")} kV</small></li>`).join("");
    const flag = p.mismatch ? `<p class="flag">OSM circuits (220/380 kV): ${esc(fmt(p.circuits_tag))}, model circuits drawn: ${p.model_circuits} (${esc(p.mismatch)})</p>` : "";
    return `<h4>Carried model circuits (${p.model_circuits})</h4>${flag}<ul class="carried">${list || "<li>none</li>"}</ul>`;
  }
  function popupHtml(layer, props) {
    if (layer === "towers") {
      const rows = FIELDS.towers.filter((k) => k in props).map((k) => `<tr><td>${esc(k)}</td><td>${esc(fmt(props[k]))}</td></tr>`).join("");
      return `<div class="popup"><h3>${esc(props.name || "Tower line " + (props.ref || props.osm_id))}</h3>${towerHtml(props)}<table>${rows}</table></div>`;
    }
    const keys = FIELDS[layer] || Object.keys(props);
    const title = props.name || props.bus || props.site || layer;
    const rows = keys.filter((k) => k in props).map((k) => `<tr><td>${esc(k)}</td><td>${esc(fmt(props[k]))}</td></tr>`).join("");
    return `<div class="popup"><h3>${esc(title)}</h3><table>${rows}</table></div>`;
  }
  const clickable = ["neighbours", "plants", "plants-off", "plant-aggregates", "substations", "tees", "transformers",
    "transformers-mid", "buses", "plant-links", "lines-ac", "lines-straight", "lines-hvdc", "lines-oos", "towers"];
  map.on("error", (e) => { console.error("map error", e.error && e.error.message); });
  map.on("click", (e) => {
    const hits = map.queryRenderedFeatures([[e.point.x - 4, e.point.y - 4], [e.point.x + 4, e.point.y + 4]],
      { layers: clickable.filter((id) => map.getLayer(id)) });
    if (!hits.length) return;
    const f = hits[0];
    const layer = f.layer.id.startsWith("lines") ? "lines" : f.layer.id;
    new maplibregl.Popup({ maxWidth: "380px" }).setLngLat(e.lngLat).setHTML(popupHtml(layer, f.properties)).addTo(map);
  });
  map.on("mousemove", (e) => {
    const ids = clickable.filter((id) => map.getLayer(id));
    const n = map.queryRenderedFeatures([[e.point.x - 4, e.point.y - 4], [e.point.x + 4, e.point.y + 4]], { layers: ids }).length;
    map.getCanvas().style.cursor = n ? "pointer" : "";
  });

  // ---------------------------------------------------------------- panel
  function renderLayers() {
    const box = document.getElementById("layers");
    box.innerHTML = PANEL_LAYERS.slice().reverse().map((n) =>
      `<label class="layer-row"><input type="checkbox" data-layer="${n}" ${state.visible[n] ? "checked" : ""}> ${TITLES[n]}</label>`).join("");
    box.querySelectorAll("input").forEach((el) => el.addEventListener("change", () => {
      state.visible[el.dataset.layer] = el.checked;
      restyle();
    }));
  }

  function renderLegend() {
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
    rows.push('<p class="muted">Symbol size grows with the number of transformers. “2×” = circuits on the tower line.</p>');
    line("#9e9e9e", "Tower line (OpenStreetMap power=line, ref label)");
    line(S.voltage.hvdc, "HVDC link", "dashed");
    line(S.neutral, "Route not mapped (straight)", "dashed");
    line(S.out_of_service, "Out of service", "dotted");
    icon("tee", "Tee / switching point");
    icon("transformer_380_220", "Transformer 380/220 kV");
    icon("transformer_380_110", "Transformer to 110 kV");
    icon("phase_shifter", "Phase-shifting transformer");
    icon("neighbour", "Neighbouring network");
    rows.push(`<h2>Power plants (size ∝ √MW)</h2><p class="muted">Units ≥ ${S.plants.individual_mw} MW at their site, grey line to the connection substation; smaller units summed per bus and fuel in a row left of the substation.</p>`);
    Object.entries(S.fuels).forEach(([k, f]) => icon(`plant_${k}`, f.label));
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
        new maplibregl.Popup({ maxWidth: "380px" }).setLngLat(at).setHTML(popupHtml(r.layer, r.f.properties)).addTo(map);
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
    document.querySelectorAll("[data-mode]").forEach((x) => x.classList.toggle("active", x.dataset.mode === state.mode));
    document.querySelectorAll("[data-basemap]").forEach((x) => x.classList.toggle("active", x.dataset.basemap === state.basemap));
    renderLayers(); renderLegend(); renderSummary(); wireButtons(); wireSearch(buildIndex());
    const m = state.meta || {};
    document.getElementById("model-version").textContent =
      `Model ${m.source_file || "v4final"} · sha256 ${(m.source_sha256 || "").slice(0, 12)} · exported ${(m.exported_utc || "").slice(0, 10)}`;
    document.getElementById("sources").innerHTML =
      "Control areas are approximate (Voronoi cells of each TSO's buses). DK1/NO2 drawn with the whole country shape. " +
      "Connection/terminal buses (110/21 kV) sit at their substation. Data: JAO Core Static Grid Model, " +
      "OpenStreetMap contributors (ODbL) via PyPSA-Eur, Marktstammdatenregister, Natural Earth.";
    document.querySelectorAll(".maplibregl-ctrl-attrib").forEach((el) => el.classList.remove("maplibregl-compact-show"));
    document.body.dataset.ready = "1";
    window.gridMap = map; // for the browser console and automated screenshots
  }).catch((err) => {
    document.getElementById("summary").textContent = `Failed to load data: ${err.message}`;
  });
})();
