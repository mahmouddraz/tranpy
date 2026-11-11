"""Acceptance of a production map build against its staging source.

    python scripts/prod_acceptance.py STAGING_WEB_DIR PROD_DIR [--url http://127.0.0.1:8099/]

Hard checks (all must pass):

* ``no_debug_fields``: no field of ``make_prod_web.DROP`` in any layer, no metadata key
  outside ``META_KEEP`` (+ the production keys), no data-quality flags in
  station_details.json, no banned text (needs owner, not sourced, halo, ...), and
  ``mode == "production"``;
* ``same_features``: every kept layer has the same features with identical geometries, in
  the same order, as the staging source; ``lines`` keeps every staging circuit (by mrid)
  and changes geometry only for circuits redrawn as one connected station-to-station line;
* ``model_only``: none of the ``make_prod_web.REMOVED`` layers is present;
* ``connected_circuits``: every circuit drawn outside a station is one connected line with
  both ends on its own stations (``make_prod_web.circuit_problem``: no gap, stub or
  fragment);
* ``model_complete`` (``--model``): circuits = the model's AC line segments (+ HVDC links)
  by id, and every bus, transformer, external injection and machine of the model is in
  the production data (buses, transformers, neighbours, station panel machines);
* ``sld_complete``: every substation diagram exists, and the SLD count equals staging;
* ``app_profile``: the served app (``--url``) loads with no console / page errors, reports
  production, has no halo layer, no layer of a removed source, no dashed line style and
  no data-quality text in the legend;
* ``qgis_project``: ``qgis/qgs_check.json`` (written by publish_map.sh after building
  the QWC project in the QGIS container) reports a valid project with every layer valid.

Writes ``PROD_DIR/acceptance/prod_acceptance.json`` and exits 1 on a failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_prod_web import BANNED, DROP, META_KEEP, REMOVED, banned_hits

PROD_META = {"mode", "staging_build_id", "prod_built_utc", "staging_source"}


def load(p: Path):
    return json.loads(p.read_text())


def check_fields(prod: Path) -> list[str]:
    errs = []
    for f in sorted(prod.glob("*.geojson")):
        bad = DROP.get(f.stem, set()) | {
            "check",
            "check_reason",
            "model_n",
            "model_circuits",
            "route_note",
        }
        for i, ft in enumerate(load(f)["features"]):
            left = bad & set(ft["properties"])
            if left:
                errs.append(f"{f.name}[{i}]: {sorted(left)}")
                break
    meta = load(prod / "metadata.json")
    if meta.get("mode") != "production":
        errs.append("metadata mode is not production")
    extra = set(meta) - set(META_KEEP) - PROD_META
    if extra:
        errs.append(f"metadata keys {sorted(extra)}")
    flags = sum(len(d.get("flags", [])) for d in load(prod / "station_details.json").values())
    if flags:
        errs.append(f"station_details: {flags} flags")
    for f in sorted([*prod.glob("*.json"), *prod.glob("*.geojson")]):
        errs += banned_hits(load(f), f.name)[:5]
    return errs


def check_removed(prod: Path) -> list[str]:
    return [
        f"{n}.geojson present ({why[:60]})"
        for n, why in REMOVED.items()
        if (prod / f"{n}.geojson").exists()
    ]


def check_lines(stage: Path, prod: Path) -> tuple[list[str], dict]:
    """Same circuits as staging; a changed circuit is one feature (the connected fallback)."""
    from collections import defaultdict

    errs = []
    a, b = defaultdict(list), defaultdict(list)
    for f in load(stage / "lines.geojson")["features"]:
        a[f["properties"]["mrid"]].append(f)
    for f in load(prod / "lines.geojson")["features"]:
        b[f["properties"]["mrid"]].append(f)
    if set(a) != set(b):
        errs.append(
            f"circuits differ: {len(set(a) - set(b))} missing, {len(set(b) - set(a))} extra"
        )
    changed = [
        m for m in a if m in b and [x["geometry"] for x in a[m]] != [y["geometry"] for y in b[m]]
    ]
    for m in changed:
        old = [x["geometry"] for x in a[m]]
        kept = all(y["geometry"] in old for y in b[m])  # repaired: a subset of the pieces
        if not kept and len(b[m]) != 1:
            errs.append(f"{m}: changed geometry is neither a subset of its pieces nor one line")
    # electrical data unchanged
    elec = [
        "name",
        "voltage_kv",
        "length_km",
        "r_ohm",
        "x_ohm",
        "b_us",
        "imax_a",
        "rating_mva",
        "tso",
        "bus1",
        "bus2",
        "status",
        "hvdc",
    ]
    for m in b:
        if m in a and any(
            a[m][0]["properties"].get(k) != b[m][0]["properties"].get(k) for k in elec
        ):
            errs.append(f"{m}: electrical data changed")
    return errs, {"circuits": len(b), "redrawn": len(changed)}


def check_connected(prod: Path) -> tuple[list[str], dict]:
    from collections import defaultdict

    from make_prod_web import Stations, circuit_problem

    st = Stations(prod)
    by = defaultdict(list)
    for f in load(prod / "lines.geojson")["features"]:
        by[f["properties"]["mrid"]].append(f)
    errs, inside = [], 0
    for m, fs in by.items():
        p = fs[0]["properties"]
        if all(f.get("geometry") is None for f in fs):
            inside += 1  # drawn only inside its station (station panel), nothing to connect
            continue
        if p.get("hvdc"):
            continue
        why = circuit_problem(st, fs)
        if why:
            errs.append(f"{p['name']} ({m}): {why}")
    return errs, {"inside_station_only": inside, "checked": len(by) - inside}


def check_model(prod: Path, model: Path) -> tuple[list[str], dict]:
    import pandas as pd

    _s, cim = pd.read_pickle(model)
    props = lambda n: [f["properties"] for f in load(prod / f"{n}.geojson")["features"]]  # noqa: E731
    errs, info = [], {}
    lines = props("lines")
    ac = {p["mrid"] for p in lines if not p.get("hvdc")}
    want = set(cim.ac_line_segment["mrid"])
    info["circuits_model"], info["circuits_prod_ac"] = len(want), len(ac)
    info["hvdc_links"] = len({p["mrid"] for p in lines if p.get("hvdc")})
    if ac != want:
        errs.append(f"circuits: {len(want - ac)} model circuits missing, {len(ac - want)} extra")
    for name, table, key, col in (
        ("buses", cim.busbar_section, "bus_mrid", "mrid"),
        ("transformers", cim.power_transformer, "mrid", "mrid"),
        ("neighbours", cim.external_network_injection, "mrid", "mrid"),
    ):
        have = {p[key] for p in props(name)}
        miss = set(table[col]) - have
        info[f"{name}_model"], info[f"{name}_prod"] = len(table), len(have)
        if miss:
            errs.append(f"{name}: {len(miss)} of the model missing")
    det = load(prod / "station_details.json")
    have = {m["mrid"] for d in det.values() for m in d.get("machines", [])}
    miss = set(cim.synchronous_machine["mrid"]) - have
    info["machines_model"], info["machines_prod"] = len(cim.synchronous_machine), len(have)
    if miss:
        errs.append(f"machines: {len(miss)} of the model missing from the station panels")
    return errs, info


def check_features(stage: Path, prod: Path) -> tuple[list[str], dict[str, int]]:
    errs, counts = [], {}
    for f in sorted(stage.glob("*.geojson")):
        if f.stem in REMOVED:
            continue
        p = prod / f.name
        if f.stem == "lines":
            e, counts["lines"] = check_lines(stage, prod)
            errs += e
            continue
        if not p.exists():
            errs.append(f"{f.name} missing")
            continue
        a, b = load(f)["features"], load(p)["features"]
        counts[f.stem] = len(b)
        if f.stem == "offshore_links":  # production keeps only the sourced landings
            names = {y["properties"]["name"] for y in b}
            a = [x for x in a if x["properties"]["name"] in names]
        if len(a) != len(b):
            errs.append(f"{f.name}: {len(a)} staging vs {len(b)} prod features")
            continue
        diff = sum(x["geometry"] != y["geometry"] for x, y in zip(a, b))
        if diff:
            errs.append(f"{f.name}: {diff} geometries differ")
        # every kept property keeps its staging value, except the neutral notes
        neutral = {
            "model_note",
            "note",
            "offshore_spread_note",
            "in_model",
            "offshore_farms_landing",
        }
        for x, y in zip(a, b):
            changed = [
                k
                for k in y["properties"]
                if k not in neutral and x["properties"].get(k) != y["properties"][k]
            ]
            if changed:
                errs.append(f"{f.name}: value changed {changed}")
                break
    return errs, counts


def check_sld(stage: Path, prod: Path) -> tuple[list[str], int]:
    errs = []
    n = len(list((prod / "sld").glob("*.svg")))
    if n != len(list((stage / "sld").glob("*.svg"))):
        errs.append(f"sld count {n} differs from staging")
    for ft in load(prod / "substations.geojson")["features"]:
        d = ft["properties"].get("diagram")
        if d and not (prod / d).exists():
            errs.append(f"missing {d}")
    for site, d in load(prod / "station_details.json").items():
        if d.get("diagram") and not (prod / d["diagram"]).exists():
            errs.append(f"station {site}: missing {d['diagram']}")
    return errs, n


LINE_POPUP_ROWS = ["Voltage", "Circuits", "Length", "Rating", "TSO", "Status"]


def click_line(pg) -> dict:
    """Click a single-circuit AC line in the browser and read the popup that opens.

    Zooms to the middle of a long single circuit (no side-by-side offset), clicks its
    pixel with the mouse (the real click path: the clickable layer list and the popup)
    and returns the circuit name, the popup title and its row labels.
    """
    pg.evaluate("""() => {
      document.querySelectorAll('.maplibregl-popup').forEach((p) => p.remove());
      const st = document.getElementById('station-panel'); if (st) st.classList.remove('open');
      const map = window.gridMap;
      const f = map.querySourceFeatures('lines').filter((f) => f.properties.circuits === 1
        && !f.properties.hvdc && f.properties.status === 'in_operation'
        && f.geometry.type === 'LineString' && f.properties.length_km > 30)
        .sort((a, b) => b.properties.length_km - a.properties.length_km)[0];
      const c = f.geometry.coordinates;
      window.__lineTarget = { name: f.properties.name, at: c[Math.floor(c.length / 2)] };
      map.jumpTo({ center: window.__lineTarget.at, zoom: 11 });
    }""")
    pg.wait_for_timeout(2500)
    xy = pg.evaluate("""() => {
      const p = window.gridMap.project(window.__lineTarget.at);
      const r = window.gridMap.getCanvas().getBoundingClientRect();
      return [r.left + p.x, r.top + p.y];
    }""")
    pg.mouse.click(xy[0], xy[1])
    pg.wait_for_timeout(1000)
    return pg.evaluate("""() => {
      const pop = document.querySelector('.maplibregl-popup-content');
      return { name: window.__lineTarget.name,
        title: pop && pop.querySelector('h3') ? pop.querySelector('h3').textContent : '',
        rows: pop ? [...pop.querySelectorAll('tr td:first-child')].map((td) => td.textContent) : [] };
    }""")


def check_app(url: str) -> tuple[list[str], dict]:
    from playwright.sync_api import sync_playwright

    errs, info = [], {}
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1300, "height": 850})
        console = []
        pg.on("console", lambda m: console.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: console.append(str(e)))
        pg.goto(url, wait_until="networkidle")
        pg.wait_for_selector('body[data-ready="1"]', timeout=120000)
        pg.wait_for_timeout(3000)
        info = pg.evaluate("""() => ({
          layers: window.gridMap.getStyle().layers.map((l) => l.id),
          dashed: window.gridMap.getStyle().layers.filter((l) => l.paint
            && l.paint["line-dasharray"]
            && (window.gridMap.getStyle().sources[l.source] || {}).type === "geojson")
            .map((l) => l.id),
          sources: [...new Set(window.gridMap.getStyle().layers.map((l) => l.source)
            .filter(Boolean))],
          legend: document.getElementById('legend').innerText,
          layerPanel: document.getElementById('layers').innerText,
          build: (document.getElementById('build-id') || {}).textContent || ''})""")
        # open a station panel and a farm popup source: the station panel must have no flags
        pg.evaluate("() => window.gridOpenStation('bentwisch')")
        pg.wait_for_timeout(1500)
        info["panel"] = pg.evaluate(
            "() => (document.getElementById('station-panel') || {}).innerText || ''"
        )
        info["line_popup"] = click_line(pg)
        b.close()
    info["console_errors"] = console
    if console:
        errs.append(f"console errors: {console[:5]}")
    if info["dashed"]:
        errs.append(f"dashed line style in production: {info['dashed']}")
    bad_src = sorted(set(info["sources"]) & set(REMOVED))
    if bad_src:
        errs.append(f"layers of removed sources: {bad_src}")
    if "osm-halo" in info["layers"]:
        errs.append("halo layer in production")
    for t in (info["legend"], info["layerPanel"], info["panel"]):
        if BANNED.search(t) or "unexplained" in t or "Data-quality" in t or "model check" in t:
            errs.append(f"debug text in the app: {t[:200]}")
    if "STAGING" in info["build"]:
        errs.append("app reports staging")
    lp = info["line_popup"]
    if not lp.get("name") or lp.get("title") != lp.get("name"):
        errs.append(f"line click: no circuit popup ({lp})")
    else:
        missing = [k for k in LINE_POPUP_ROWS if k not in lp.get("rows", [])]
        extra = [k for k in lp.get("rows", []) if k not in LINE_POPUP_ROWS]
        if missing or extra:
            errs.append(f"line popup rows: missing {missing}, debug/extra {extra}")
    info["layers"] = len(info["layers"])
    info["panel"] = info["panel"][:300]
    return errs, info


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--sub"]:  # helper mode (run with --py)
        kind, data, arg = argv[1], Path(argv[2]), argv[3] if len(argv) > 3 else ""
        r = check_connected(data) if kind == "connected" else check_model(data, Path(arg))
        print(json.dumps(r))
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("staging", type=Path)
    ap.add_argument("prod", type=Path, help="production build folder (with data/)")
    ap.add_argument("--url", help="served production app, e.g. http://127.0.0.1:8099/")
    ap.add_argument("--skip-qgis", action="store_true")
    ap.add_argument("--model", type=Path, help="frozen model pickle (model_complete check)")
    ap.add_argument(
        "--py",
        default=str(Path.home() / "code/tranpy/.venv/bin/python"),
        help="python with geopandas/pandas for the geometry and model checks",
    )
    a = ap.parse_args(argv)
    data = a.prod / "data"
    res: dict = {"staging": str(a.staging.resolve()), "prod": str(a.prod.resolve())}
    checks = {}
    checks["no_debug_fields"] = check_fields(data)
    checks["same_features"], res["feature_counts"] = check_features(a.staging, data)
    checks["sld_complete"], res["sld_count"] = check_sld(a.staging, data)
    checks["model_only"] = check_removed(data)
    # geometry and model checks need shapely / pandas: run them with --py when this python
    # (playwright) lacks them
    sub = {"connected_circuits": ("connected", None)}
    if a.model:
        sub["model_complete"] = ("model", str(a.model))
    for name, (kind, arg) in sub.items():
        try:
            import pandas  # noqa: F401
            import shapely  # noqa: F401

            e, info = check_connected(data) if kind == "connected" else check_model(data, a.model)
        except ImportError:
            import subprocess

            out = subprocess.run(
                [a.py, __file__, "--sub", kind, str(data), arg or ""],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            e, info = json.loads(out.strip().splitlines()[-1])
        checks[name], res[name] = e, info
    if not a.model:
        checks["model_complete"] = ["not run (no --model)"]
    if a.url:
        checks["app_profile"], res["app"] = check_app(a.url)
    else:
        checks["app_profile"] = ["not run (no --url)"]
    if not a.skip_qgis:
        q = a.prod / "qgis" / "qgs_check.json"
        if not q.exists():
            checks["qgis_project"] = ["qgis/qgs_check.json missing (build the QWC project first)"]
        else:
            res["qgis"] = r = load(q)
            checks["qgis_project"] = (
                []
                if r.get("valid") and not r.get("invalid_layers") and r.get("layers", 0) > 0
                else [f"qgis project: {r}"]
            )
    import re

    dashed_qml = [
        q.name
        for q in sorted((a.prod / "qgis" / "styles").glob("*.qml"))
        if re.search(r'_style" type="QString" value="(dot|dash)', q.read_text())
    ]
    checks["qgis_no_dashed"] = [f"dashed style in {dashed_qml}"] if dashed_qml else []
    res["checks"] = {k: {"passed": not v, "errors": v[:20]} for k, v in checks.items()}
    res["passed"] = all(not v for v in checks.values())
    out = a.prod / "acceptance"
    out.mkdir(exist_ok=True)
    (out / "prod_acceptance.json").write_text(json.dumps(res, indent=1))
    print(
        json.dumps(
            {
                "passed": res["passed"],
                **{k: v["passed"] for k, v in res["checks"].items()},
                "sld_count": res["sld_count"],
                "lines": res["feature_counts"].get("lines"),
                "connected": res.get("connected_circuits"),
                "model": res.get("model_complete"),
            },
            indent=1,
        )
    )
    for k, v in checks.items():
        for e in v[:10]:
            print(f"  {k}: {e}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
