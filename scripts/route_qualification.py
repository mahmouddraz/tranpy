"""Route qualification of a STAGING map build: which drawn circuit routes are verified.

    python scripts/route_qualification.py STAGING_WEB_DIR MAP_ACCEPTANCE_JSON OUT_DIR

A circuit's route is ``verified`` when all of these hold (docs/grid_map.md, "Route
qualification"):

1. every piece is routed on the OSM graph (``route_status`` routed, ``route_quality``
   osm_graph, no ``route_note``) and no piece is a straight placeholder
   (``geometry_source`` without "straight");
2. every OSM way section the circuit is drawn on (``osm_lines.model_circuits``) has the
   check ``match`` or ``explained`` (explained = a sourced reason in ``check_reason``); a
   ``short`` section (< 150 m, the check of its corridor applies) is accepted; ``no_tag``
   (OSM has no circuit count), ``model_fewer`` and ``model_more`` are not;
3. the circuit is on none of the staging map acceptance lists (scripts/map_acceptance.py:
   ends_off_station, station_touch, gaps, ratio_outside, drawn_not_once, straight_undashed,
   on_own_way, no_zigzag, no_loop, no_spike, station_end);
4. or it is listed in ``route_fixes.csv`` (OUT_DIR; a logged C-change with a citation),
   which waives the way-count reasons named there; rules 1 and 3 still apply.

Everything else is ``schematic`` (the qualification backlog). Writes OUT_DIR/backlog.csv
(schematic circuits, ranked by kV then length), OUT_DIR/route_quality.csv (all circuits)
and OUT_DIR/summary.json (counts and km by kV and TSO). The staging folder is only read;
the classification is a staging matter and is not written into production.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

OK_WAY = {"match", "explained", "short"}
ACC_LISTS = [
    "ends_off_station",
    "station_touch",
    "gaps",
    "ratio_outside",
    "drawn_not_once",
    "straight_undashed",
    "on_own_way",
    "no_zigzag",
    "no_loop",
    "no_spike",
    "station_end",
]


def classify(web: Path, acc: dict, fixes: dict[str, set[str]]) -> list[dict]:
    lines = json.loads((web / "lines.geojson").read_text())["features"]
    ways = json.loads((web / "osm_lines.geojson").read_text())["features"]
    by = defaultdict(list)
    for f in lines:
        by[f["properties"]["mrid"]].append(f)
    way_checks = defaultdict(list)
    for f in ways:
        p = f["properties"]
        for c in json.loads(p.get("model_circuits") or "[]"):
            way_checks[c["mrid"]].append(p["check"])
    fail = defaultdict(set)
    for k in ACC_LISTS:
        for r in acc["lists"].get(k, []):
            fail[r["mrid"]].add(k)
    rows = []
    for m, fs in by.items():
        P = [f["properties"] for f in fs]
        why = []
        if all(f["geometry"] is None for f in fs):
            why.append("inside_station_only")
        elif any(f["geometry"] is None for f in fs):
            why.append("missing_piece")
        why += sorted({p["route_status"] for p in P} - {"routed"})
        if any(p.get("route_quality") != "osm_graph" for p in P):
            why.append("not_on_osm_graph")
        if any(p.get("route_note") for p in P):
            why.append("route_note")
        if any("straight" in str(p.get("geometry_source")) for p in P):
            why.append("straight_piece")
        checks = set(way_checks.get(m, []))
        if not checks:
            why.append("no_osm_way")
        why += [f"way_{c}" for c in sorted(checks - OK_WAY - fixes.get(m, set()))]
        why += sorted(fail.get(m, ()))
        p = P[0]
        km = sum(x.get("length_km") or 0 for x in P[:1])
        rows.append(
            {
                "mrid": m,
                "name": p["name"],
                "voltage_kv": p.get("voltage_kv"),
                "tso": p.get("tso"),
                "length_km": km,
                "status": p.get("status"),
                "hvdc": bool(p.get("hvdc")),
                "route_quality": "schematic" if why else "verified",
                "reasons": ";".join(why),
                "fix": "route_fixes.csv" if m in fixes else "",
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    web, acc_f, out = Path(argv[0]), Path(argv[1]), Path(argv[2])
    out.mkdir(parents=True, exist_ok=True)
    fixes: dict[str, set[str]] = {}
    ff = out / "route_fixes.csv"
    if ff.exists():
        for r in csv.DictReader(ff.open()):
            fixes[r["mrid"]] = {x for x in r["waives"].split(";") if x}
    rows = classify(web, json.loads(acc_f.read_text()), fixes)
    cols = list(rows[0])
    with (out / "route_quality.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["mrid"]))
    back = sorted(
        [r for r in rows if r["route_quality"] == "schematic"],
        key=lambda r: (-(r["voltage_kv"] or 0), -(r["length_km"] or 0)),
    )
    with (out / "backlog.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, ["rank", *cols])
        w.writeheader()
        w.writerows({"rank": i + 1, **r} for i, r in enumerate(back))
    summ: dict = {"circuits": len(rows), "by_quality": Counter(r["route_quality"] for r in rows)}
    for key in ("voltage_kv", "tso"):
        t = defaultdict(
            lambda: {"verified": 0, "schematic": 0, "verified_km": 0.0, "schematic_km": 0.0}
        )
        for r in rows:
            k = t[str(r[key])]
            k[r["route_quality"]] += 1
            k[r["route_quality"] + "_km"] = round(
                k[r["route_quality"] + "_km"] + (r["length_km"] or 0), 1
            )
        summ[f"by_{key}"] = dict(t)
    summ["reasons"] = Counter(x for r in back for x in r["reasons"].split(";")).most_common()
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: summ[k] for k in ("circuits", "by_quality")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
