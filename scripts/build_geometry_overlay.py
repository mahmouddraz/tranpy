"""Inventory the display geometry of the grid model and build a geometry overlay.

Usage::

    python scripts/build_geometry_overlay.py MODEL.pkl GPKG OSM_LINES.parquet OSM.gpkg \
        OUT_DIR [--overpass WAYS.json ...]

``GPKG`` is an export *without* overlay (end gaps of the model's own geometry);
``OSM_LINES`` the 220/380 kV lines of the OpenStreetMap export (``hv_lines.parquet``),
``OSM.gpkg`` the export itself (substation polygons); ``--overpass`` adds ways fetched
from Overpass (``out body geom``) for areas outside the export (Austria).

Writes ``OUT_DIR/geom_issues.csv`` (every schematic or doubtful item and what was done)
and ``OUT_DIR/geometry_overlay.parquet`` (:data:`tranpy.grid_gis.OVERLAY_COLUMNS`).
Rules, in order (topology first, published values first):

1. tees (``kind == tee``): :func:`tranpy.grid_gis.place_tee` on the published leg lengths;
   a tee is moved only when a leg shorter than 0.5 km puts it at a station.
2. schematic stations: the OpenStreetMap substation polygon of the same name within
   15 km (voltage >= 220 kV); otherwise unchanged and flagged.
3. lines whose route is schematic (or none) or ends more than 3 km from its bus: a
   shortest path over same-voltage OpenStreetMap ways between the (corrected) ends,
   accepted when its length is within 15 % of the published length; a leg shorter than
   0.5 km is drawn straight between its ends; otherwise a straight line between the
   corrected ends (``straight_between_corrected_ends``) and flagged.

OpenStreetMap data (c) OpenStreetMap contributors, ODbL; research use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tranpy.grid_gis import (
    TEE_AT_STATION_KM,
    _dist_m,
    _route_coords,
    load_model,
    place_tee,
    polyline_km,
)
from tranpy.osm_routes import _KM, fold, walk

GAP_M = 3000.0
LENGTH_TOLERANCE = 0.15
NAME_RADIUS_KM = 15.0
CIRCUIT_OSM_NAMES = {"BLUDNZ W": "Bludenz West", "DELLM O": "Dellmensingen Ost"}
"""Circuit code of the model -> ``name`` of its OpenStreetMap ways (identified by hand from
the ways' name tags; refs 403/404 in Vorarlberg, 4509/0304/0339 in Germany)."""
OVERLAY_COLUMNS = ["element", "id", "name", "field", "old", "new", "method", "evidence"]


def _ways(osm_lines: Path, overpass: list[Path]) -> list[dict[str, Any]]:
    """220/380 kV ways as Overpass-like dicts (``id``, ``nodes``, ``geometry``, ``kv``)."""
    shapely = __import__("shapely")
    frame = pd.read_parquet(osm_lines)
    frame = frame[~frame["disused"].fillna(False).astype(bool)]
    node_id: dict[tuple[float, float], int] = {}
    out: list[dict[str, Any]] = []
    for way_id, volts, wkb in zip(frame["id"], frame["voltages"], frame["geometry"], strict=True):
        kvs = {int(v) // 1000 for v in str(volts).split(";") if v.strip().isdigit()}
        kvs &= {220, 380}
        if not kvs:
            continue
        g = shapely.from_wkb(wkb)
        parts = list(g.geoms) if hasattr(g, "geoms") else [g]
        for part in parts:
            coords = [(round(x, 7), round(y, 7)) for x, y in part.coords]
            ids = [node_id.setdefault(c, -1 - len(node_id)) for c in coords]
            out.append(
                {
                    "id": f"export:{way_id}",
                    "nodes": ids,
                    "geometry": [{"lon": x, "lat": y} for x, y in coords],
                    "kv": kvs,
                }
            )
    for path in overpass:
        for w in json.loads(path.read_text(encoding="utf-8"))["elements"]:
            if w.get("type") != "way" or "geometry" not in w:
                continue
            volts = str(w.get("tags", {}).get("voltage", ""))
            kvs = {int(v) // 1000 for v in volts.split(";") if v.strip().isdigit()} & {220, 380}
            if kvs:
                out.append(
                    {
                        "id": f"way/{w['id']}",
                        "nodes": w["nodes"],
                        "geometry": w["geometry"],
                        "kv": kvs,
                        "names": str(w.get("tags", {}).get("name", "")).split(";"),
                        "ref": str(w.get("tags", {}).get("ref", "")),
                    }
                )
    return out


def _route(
    ways: list[dict[str, Any]], kv: int, a: tuple[float, float], b: tuple[float, float]
) -> tuple[list[tuple[float, float]], list[str]] | None:
    """Shortest route over ``kv`` ways from ``a`` to ``b`` and the way ids it uses."""
    span = max(_dist_m(a, b) / 1000.0, 5.0)
    pad = span * 0.4 / 70.0 + 0.05  # degrees
    lo = (min(a[0], b[0]) - pad * 1.5, min(a[1], b[1]) - pad)
    hi = (max(a[0], b[0]) + pad * 1.5, max(a[1], b[1]) + pad)
    sel = [
        w
        for w in ways
        if kv in w["kv"]
        and any(lo[0] <= q["lon"] <= hi[0] and lo[1] <= q["lat"] <= hi[1] for q in w["geometry"])
    ]
    if not sel:
        return None
    found = walk(sel, np.array(a) * _KM, np.array(b) * _KM)
    if found is None:
        return None
    coords = [a, *[(float(x), float(y)) for x, y in found.points], b]
    used = set(map(tuple, np.round(found.points, 6).tolist()))
    ids = [
        w["id"]
        for w in sel
        if sum((round(q["lon"], 6), round(q["lat"], 6)) in used for q in w["geometry"]) >= 2
    ]
    return coords, ids


def _cut(coords: list[tuple[float, float]], km: float) -> tuple[list[Any], list[Any]]:
    """Split a polyline at arc length ``km``: (start .. cut point, cut point .. end)."""
    done = 0.0
    for i in range(len(coords) - 1):
        step = polyline_km(coords[i : i + 2])
        if step > 0 and done + step >= km:
            t = (km - done) / step
            a, b = coords[i], coords[i + 1]
            p = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            return [*coords[: i + 1], p], [p, *coords[i + 1 :]]
        done += step
    return list(coords), [coords[-1]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("gpkg")
    parser.add_argument("osm_lines")
    parser.add_argument("osm_gpkg")
    parser.add_argument("out_dir")
    parser.add_argument("--overpass", nargs="*", default=[])
    args = parser.parse_args()
    gpd = __import__("geopandas")
    out = Path(args.out_dir)

    silver, cim = load_model(args.model)
    nodes = silver["grid_de_nodes"].set_index("node")
    site_of = nodes["site"].to_dict()
    lines = gpd.read_file(args.gpkg, layer="lines")
    subs = gpd.read_file(args.gpkg, layer="substations")
    seg = cim.ac_line_segment.set_index("mrid")
    ledger = silver["grid_de_length_ledger"].set_index("mrid")
    point = {n: (float(r.lon), float(r.lat)) for n, r in nodes.iterrows()}
    issues: list[dict[str, Any]] = []
    overlay: list[dict[str, Any]] = []

    def issue(element: str, ident: str, name: str, kind: str, detail: str, action: str) -> None:
        issues.append(
            {
                "element": element,
                "id": ident,
                "name": name,
                "issue": kind,
                "detail": detail,
                "action": action,
            }
        )

    ways = _ways(Path(args.osm_lines), [Path(p) for p in args.overpass])
    handled: set[str] = set()
    done_tees: set[str] = set()

    def add_line(ln: Any, coords: list[Any], method: str, evidence: str, action: str) -> None:
        old = _route_coords(seg.loc[ln.mrid, "route"]) if ln.mrid in seg.index else []
        overlay.append(
            {
                "element": "line",
                "id": ln.mrid,
                "name": ln["name"],
                "field": "route",
                "old": json.dumps(old),
                "new": json.dumps([tuple(c) for c in coords]),
                "method": method,
                "evidence": evidence,
            }
        )
        issue(
            "line",
            ln.mrid,
            ln["name"],
            "route",
            f"{ln.route_source}; gap {ln.end_gap_m} m; published {ln.length_km:g} km",
            f"{action} ({method}): {evidence}",
        )
        handled.add(ln.mrid)

    # 0. tee'd circuits with a missing (0 km) leg whose own OSM ways are named: the tee
    # sits on that named route at the published length of the leg from the station
    ac0 = lines[~lines["hvdc"].astype(bool)]
    for code, osm_name in CIRCUIT_OSM_NAMES.items():
        mine = ac0[ac0["name"].str.endswith(f" {code}")]
        tee_nodes = {b for b in pd.concat([mine.bus1, mine.bus2]) if nodes.loc[b, "kind"] == "tee"}
        named = [w for w in ways if osm_name in w.get("names", [])]
        for tee in sorted(tee_nodes):
            legs_df = mine[(mine.bus1 == tee) | (mine.bus2 == tee)]
            ends = {(r.bus2 if r.bus1 == tee else r.bus1): r for _, r in legs_df.iterrows()}
            missing = [e for e, r in ends.items() if not r.length_km > 0]
            known = [
                e for e, r in ends.items() if r.length_km > 0.5 and nodes.loc[e, "kind"] != "tee"
            ]
            if not missing or not known or not named:
                continue
            a = min(known, key=lambda e: ends[e].length_km)  # the station nearest the tee
            far = [e for e in ends if e not in missing and e != a]
            target = point[far[0]] if far else point[missing[0]]
            # the named chain from station a towards the far end
            tips = [
                (q["lon"], q["lat"]) for w in named for q in (w["geometry"][0], w["geometry"][-1])
            ]
            tip = min(tips, key=lambda t: _dist_m(t, target))
            chain_walk = walk(named, np.array(point[a]) * _KM, np.array(tip) * _KM, join_km=1.0)
            if chain_walk is None:
                continue
            chain = [point[a], *[(float(x), float(y)) for x, y in chain_walk.points]]
            chain_km = polyline_km(chain)
            leg_a = float(ends[a].length_km)
            first, rest = _cut(chain, leg_a)
            new = (round(first[-1][0], 6), round(first[-1][1], 6))
            site = str(nodes.loc[tee, "site"])
            ids = sorted({w["id"] for w in named})
            refs = sorted({w["ref"] for w in named if w["ref"]})
            moved = _dist_m(point[tee], new)
            ev = (
                f"OSM ways named '{osm_name}' ({', '.join(ids)}; refs {', '.join(refs)}), "
                f"{chain_km:.1f} km from {a} to {tip}; tee at the published {leg_a:g} km "
                f"from {a}; leg(s) to {', '.join(missing)} 0 km = missing"
            )
            overlay.append(
                {
                    "element": "site",
                    "id": site,
                    "name": tee,
                    "field": "point",
                    "old": json.dumps(point[tee]),
                    "new": json.dumps(new),
                    "method": "tee_on_named_route",
                    "evidence": f"{ev}; moved {moved / 1000:.2f} km",
                }
            )
            issue(
                "site",
                site,
                str(tee),
                "tee position",
                ev,
                f"corrected (tee_on_named_route, {moved / 1000:.2f} km)",
            )
            for n in nodes.index[nodes["site"] == nodes.loc[tee, "site"]]:
                point[n] = new
            done_tees.add(tee)
            add_line(
                ends[a],
                first,
                "named_osm_route",
                f"first {leg_a:g} km of the '{osm_name}' route",
                "corrected",
            )
            for e in far:
                r = ends[e]
                coords = rest
                tail = (
                    _route(ways, 220, rest[-1], point[e])
                    if _dist_m(rest[-1], point[e]) > 1000
                    else None
                )
                coords = [*rest, *(tail[0] if tail else [point[e]])]
                rkm = polyline_km(coords)
                add_line(
                    r,
                    coords,
                    "named_osm_route",
                    f"rest of the '{osm_name}' route ({polyline_km(rest):.1f} km)"
                    f"{' + 220 kV OSM ways to ' + e if tail else ' + straight to ' + e}; "
                    f"{rkm:.1f} km vs published {r.length_km:g} km ({rkm / r.length_km - 1:+.0%})",
                    "corrected",
                )
            for e in missing:
                found_m = _route(ways, 220, new, point[e])
                if found_m:
                    add_line(
                        ends[e],
                        found_m[0],
                        "osm_route_length_missing",
                        f"220 kV OSM ways {', '.join(found_m[1][:8])}; "
                        f"{polyline_km(found_m[0]):.1f} km; published length 0 = missing",
                        "corrected",
                    )
                else:
                    add_line(
                        ends[e],
                        [new, point[e]],
                        "straight_between_corrected_ends",
                        "no OSM route; published length 0 = missing",
                        "flagged",
                    )

    # 1. tees
    ac = lines[~lines["hvdc"].astype(bool)]
    tees = nodes[nodes["kind"] == "tee"]
    for tee, row in tees.iterrows():
        legs_df = ac[(ac["bus1"] == tee) | (ac["bus2"] == tee)]
        if tee in done_tees:
            continue
        doubtful = "schematic" in str(row.location_source) or (
            (legs_df["route_source"] == "schematic").any() | (legs_df["end_gap_m"] > GAP_M).any()
        )
        if not doubtful:
            continue
        legs = []
        for _, leg in legs_df.iterrows():
            end = leg.bus2 if leg.bus1 == tee else leg.bus1
            legs.append((end, point[end], float(leg.length_km), nodes.loc[end, "kind"] != "tee"))
        text = "; ".join(f"{e} {k:g} km" for e, _, k, _ in legs)
        placed = place_tee(legs)
        site = str(row.site)
        if placed is None:
            issue(
                "site",
                site,
                "Y",
                "tee position",
                f"{row.location_source}; legs {text}",
                "flagged: no leg < 0.5 km and no common route; unchanged",
            )
            continue
        new, method, evidence = placed
        moved = _dist_m(point[tee], new)
        if moved < 1.0:
            issue("site", site, "Y", "tee position", f"legs {text}", "already at station")
            continue
        # mapped legs (OSM routes ending at their bus, any voltage of the site) must agree
        site_nodes = set(nodes.index[nodes["site"] == row.site])
        site_legs = ac[ac["bus1"].isin(site_nodes) | ac["bus2"].isin(site_nodes)]
        mapped_ends = [
            min(
                (c[0], c[-1]),
                key=lambda e: _dist_m(e, point[tee]),
            )
            for c, src, gap in zip(
                (list(g.coords) for g in site_legs.geometry),
                site_legs["route_source"],
                site_legs["end_gap_m"],
                strict=True,
            )
            if src not in ("schematic", "none", "within substation") and gap <= GAP_M
        ]
        off = max((_dist_m(e, new) for e in mapped_ends), default=0.0)
        if off > 1000 * TEE_AT_STATION_KM + 500:
            issue(
                "site",
                site,
                str(tee),
                "tee position",
                f"{row.location_source}; legs {text}",
                f"flagged: {evidence}, but mapped OSM legs end {off / 1000:.2f} km from that "
                "station; unchanged",
            )
            continue
        overlay.append(
            {
                "element": "site",
                "id": site,
                "name": tee,
                "field": "point",
                "old": json.dumps(point[tee]),
                "new": json.dumps(new),
                "method": method,
                "evidence": f"{evidence}; moved {moved / 1000:.2f} km",
            }
        )
        issue(
            "site",
            site,
            str(tee),
            "tee position",
            f"{row.location_source}; legs {text}",
            f"corrected ({method}, {moved / 1000:.2f} km)",
        )
        for n in nodes.index[nodes["site"] == row.site]:
            point[n] = new

    # 2. schematic stations
    osm_subs = gpd.read_file(args.osm_gpkg, layer="power_substation_polygon", engine="pyogrio")
    osm_subs = osm_subs[osm_subs["max_voltage"].fillna(0) >= 220000]
    cent = osm_subs.geometry.to_crs(3035).centroid.to_crs(4326)
    osm_subs = osm_subs.assign(lon=cent.x, lat=cent.y, fname=osm_subs["name"].map(fold))
    for _, s in subs[subs["location_source"] == "schematic"].iterrows():
        if s.kind == "tee":
            continue
        site_nodes = nodes.index[nodes["site"] == s.site]
        here = point[site_nodes[0]]
        names = {fold(x) for x in str(s["name"]).split("; ")} | {fold(s.site)}
        hits = []
        for _, o in osm_subs.iterrows():
            d_km = _dist_m(here, (o.lon, o.lat)) / 1000
            tokens = set(str(o.fname).split())
            if d_km < NAME_RADIUS_KM and any(n and set(n.split()) <= tokens for n in names):
                # the polygon must not be another site of the model (GKM vs GKM Station B)
                taken = [
                    n
                    for n, p in point.items()
                    if site_of[n] != s.site and _dist_m(p, (o.lon, o.lat)) < 1000
                ]
                if not taken:
                    hits.append((d_km, o))
        if len(hits) == 1:
            d_km, o = hits[0]
            new = (round(float(o.lon), 6), round(float(o.lat), 6))
            overlay.append(
                {
                    "element": "site",
                    "id": str(s.site),
                    "name": str(s["name"]),
                    "field": "point",
                    "old": json.dumps(here),
                    "new": json.dumps(new),
                    "method": "osm_substation_name",
                    "evidence": f"OSM substation {o.id} '{o['name']}' {o.voltages}; "
                    f"moved {d_km:.2f} km",
                }
            )
            for n in site_nodes:
                point[n] = new
            issue(
                "site",
                str(s.site),
                str(s["name"]),
                "schematic location",
                f"display relocated {s.display_relocated_m:g} m",
                f"corrected (OSM substation {o.id}, {d_km:.2f} km)",
            )
        else:
            issue(
                "site",
                str(s.site),
                str(s["name"]),
                "schematic location",
                f"{s.kind}; display relocated {s.display_relocated_m:g} m; "
                f"{len(hits)} OSM name matches >= 220 kV within {NAME_RADIUS_KM:g} km",
                "flagged: unchanged (display uses route-end centroid if relocated)",
            )

    # 3. lines
    good = [
        (r.mrid, r["name"], list(r.geometry.coords))
        for _, r in lines.iterrows()
        if r.route_source not in ("schematic", "none")
        and r.route_quality not in ("straight", "in_station", "station")
        and r.end_gap_m <= GAP_M
    ]

    def sibling(a: tuple[float, float], b: tuple[float, float], km: float) -> Any:
        """A model route of another circuit between the same two points (1 km)."""
        for mrid, name, c in good:
            for u, v in ((c[0], c[-1]), (c[-1], c[0])):
                if _dist_m(u, a) < 1000 and _dist_m(v, b) < 1000:
                    rkm = polyline_km(c)
                    if abs(rkm / km - 1) <= LENGTH_TOLERANCE:
                        coords = c if u == c[0] else c[::-1]
                        return [a, *coords, b], mrid, name, rkm
        return None

    for _, ln in lines.iterrows():
        if ln.mrid in handled:
            continue
        if ln.hvdc:
            if ln.route_source in ("schematic", "none"):
                issue(
                    "line",
                    ln.mrid,
                    ln["name"],
                    "schematic route (HVDC)",
                    f"{ln.length_km:g} km",
                    "flagged: cable route not in the OSM line export",
                )
            continue
        if not (ln.length_km > 0):
            led = ledger.loc[ln.mrid] if ln.mrid in ledger.index else None
            x = f"x {led.x_ohm:g} ohm, implies {led.implied_km:.1f} km" if led is not None else ""
            issue(
                "line",
                ln.mrid,
                ln["name"],
                "length 0 (JAO published)",
                f"route {polyline_km(ln.geometry.coords):.1f} km; {x}",
                "flagged: parameter gap for the owner; route_length_km shown",
            )
        b1, b2 = ln.bus1, ln.bus2
        p1, p2 = point.get(b1), point.get(b2)
        if p1 is None or p2 is None:
            continue
        ends_moved = any(
            _dist_m(point[b], (float(nodes.loc[b, "lon"]), float(nodes.loc[b, "lat"]))) > 1
            for b in (b1, b2)
        )
        schematic = ln.route_source in ("schematic", "none")
        if not (ln.end_gap_m > GAP_M or (schematic and ends_moved)):
            if schematic:
                issue(
                    "line",
                    ln.mrid,
                    ln["name"],
                    "schematic route",
                    f"gap {ln.end_gap_m} m; published {ln.length_km:g} km",
                    "flagged: unchanged (ends within 3 km)",
                )
            continue
        old = _route_coords(seg.loc[ln.mrid, "route"]) if ln.mrid in seg.index else []
        kv = 380 if float(ln.voltage_kv) >= 300 else 220
        km = float(ln.length_km) if ln.length_km == ln.length_km else 0.0
        detail = f"{ln.route_source}; gap {ln.end_gap_m} m; published {km:g} km"
        route_new: list[tuple[float, float]] | None = None
        twin = sibling(p1, p2, km) if km >= 0.5 else None
        found0 = _route(ways, kv, p1, p2) if km <= 0 and _dist_m(p1, p2) > 500 else None
        if km <= 0 and found0 is not None:
            route_new, method = found0[0], "osm_route_length_missing"
            evidence = (
                f"{kv} kV OSM ways {', '.join(found0[1][:12])}; route "
                f"{polyline_km(found0[0]):.1f} km; published length 0 = missing"
            )
        elif km < 0.5:
            route_new, method = [p1, p2], "short_leg_straight"
            evidence = f"published {km:g} km; ends {_dist_m(p1, p2):.0f} m apart"
        elif twin is not None:
            route_new, method = twin[0], "sibling_model_route"
            evidence = (
                f"route of {twin[2]} ({twin[1]}) between the same ends, {twin[3]:.1f} km vs "
                f"published {km:g} km ({twin[3] / km - 1:+.0%})"
            )
        else:
            found = _route(ways, kv, p1, p2)
            route_km = polyline_km(found[0]) if found else float("nan")
            if found and abs(route_km / km - 1) <= LENGTH_TOLERANCE:
                route_new, method = found[0], "osm_route"
                evidence = (
                    f"{kv} kV OSM ways {', '.join(found[1][:12])}"
                    f"{' ...' if len(found[1]) > 12 else ''}; route {route_km:.1f} km vs "
                    f"published {km:g} km ({route_km / km - 1:+.0%})"
                )
            elif schematic:
                route_new, method = [p1, p2], "straight_between_corrected_ends"
                evidence = (
                    f"no OSM route within {LENGTH_TOLERANCE:.0%} "
                    f"(best {route_km:.1f} km vs published {km:g} km)"
                )
            else:
                issue(
                    "line",
                    ln.mrid,
                    ln["name"],
                    "route",
                    detail,
                    f"flagged: unchanged (best OSM route {route_km:.1f} km vs {km:g} km)",
                )
                continue
        overlay.append(
            {
                "element": "line",
                "id": ln.mrid,
                "name": ln["name"],
                "field": "route",
                "old": json.dumps(old),
                "new": json.dumps(route_new),
                "method": method,
                "evidence": evidence,
            }
        )
        action = "corrected" if method != "straight_between_corrected_ends" else "flagged"
        issue("line", ln.mrid, ln["name"], "route", detail, f"{action} ({method}): {evidence}")

    # snapped station clusters (inventory only)
    snapped = lines[lines["display_snapped_m"] >= 700]
    for site, grp in pd.concat(
        [
            snapped.assign(s=snapped["bus1"].map(site_of)),
            snapped.assign(s=snapped["bus2"].map(site_of)),
        ]
    ).groupby("s"):
        issue(
            "site",
            str(site),
            str(site),
            "route ends snapped >= 0.7 km",
            f"{grp['mrid'].nunique()} lines, max {grp['display_snapped_m'].max()} m",
            "inventory only (display snap, not a model error)",
        )

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(issues).to_csv(out / "geom_issues.csv", index=False)
    pd.DataFrame(overlay, columns=OVERLAY_COLUMNS).to_parquet(out / "geometry_overlay.parquet")
    print(pd.DataFrame(issues)["action"].str.split(" ").str[0].value_counts().to_string())
    print(len(overlay), "overlay rows")


if __name__ == "__main__":
    main()
