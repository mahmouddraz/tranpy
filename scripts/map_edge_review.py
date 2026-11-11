"""Explained model/OSM circuit-count differences for the grid map (``--edge-review``).

Writes a CSV ``way_id, osm_n, model_n, reason, source`` that
``export_grid_gis.py --edge-review`` reads: an ``osm_lines`` section whose check is
``model_fewer``/``model_more`` and whose way and counts match a row is drawn without a halo,
and its popup gives the reason. Two sources:

- ``issue09``: the issue-09 review (``docs/data_quality/issues/09-edge-check/occurrences.csv``).
  Rows classified ``osm_artefact``, ``ignored_approach`` or ``out_of_scope`` are explained;
  ``routing_artefact``, ``unclear`` and ``model_error`` stay unexplained (halo). The
  occurrence is joined to the OSM ways of its finding by its start point (``lon0``/``lat0``)
  in the ``findings_*.csv`` files of the network run it was reviewed on (``--findings``).
- ``status``: a model circuit that is out of service or decommissioned (approved
  CHANGELOG entries) still hangs on the OSM towers. Its ways come from an earlier network
  run in which it was routed (``--old-network``), else from the OSM ``ref``/``name``
  holding its circuit number near its end nodes. A ``model_fewer`` edge of the current run
  (``--network``) on such a way whose shortfall is at most the number of these circuits is
  explained.
- ``extra`` (``--extra``, repeatable): reviewed rows ``way_id, osm_n, model_n, reason`` written
  by a later review (e.g. ``docs/data_quality/issues/14-remaining-halos/review_rows.csv``),
  taken as they are; ``source`` is the file's issue folder or name. An optional ``kind``
  column ``routing`` marks a difference that is a routing artefact (a model circuit of the
  way is drawn schematic / dashed / on another way): the map keeps its halo with that reason,
  and every other row of the way is dropped.

Usage::

    python scripts/map_edge_review.py --model frozen/model_v5_frozen.pkl \\
        --network NET --old-network v5/network --findings v5/network \\
        --occurrences docs/data_quality/issues/09-edge-check/occurrences.csv \\
        --changelog v5/CHANGELOG.md \\
        --extra docs/data_quality/issues/14-remaining-halos/review_rows.csv --out edge_review.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

EXPLAINED = {
    "osm_artefact": "OSM count artefact or stale OSM tag",
    "ignored_approach": "station approach piece",
    "out_of_scope": "foreign circuits (outside the German model)",
}
"""issue-09 classes that explain a difference (no halo on the map)."""
CHANGELOG_BY_EIC = {"osm-5006-1": 2}
"""CHANGELOG entries that do not name the element's EIC (328 is entry 2, by name)."""


def _short(text: str, n: int = 140) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _plain(text: str) -> str:
    t = str(text).lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        t = t.replace(a, b)
    return t


def _fold(names: object) -> list[str]:
    """Lower-case spellings of station names (ü -> ue and u), at least 4 letters."""
    items = [names] if isinstance(names, str) else list(names)  # type: ignore[call-overload]
    out = []
    for n in items:
        t = _plain(n).split("/")[0].strip()
        if len(t) >= 4:
            out += [t, t.replace("ae", "a").replace("oe", "o").replace("ue", "u")]
    return out


def issue09_rows(occurrences: Path, findings_dir: Path) -> pd.DataFrame:
    occ = pd.read_csv(occurrences)
    occ = occ[occ["classification"].isin(EXPLAINED)]
    ways: dict[tuple[float, float], str] = {}
    for f in sorted(findings_dir.glob("findings_*.csv"), key=lambda p: ("round4" not in p.name, p)):
        x = pd.read_csv(f)
        for lon, lat, w in zip(x["lon0"], x["lat0"], x["way_ids"], strict=True):
            ways.setdefault((round(lon, 5), round(lat, 5)), str(w))
    rows = []
    for r in occ.itertuples(index=False):
        ids = ways.get((round(r.lon0, 5), round(r.lat0, 5)))
        if not ids or ids == "nan":
            continue
        reason = (
            f"OSM {r.osm_n:g} / model {r.model_n:g}: {EXPLAINED[r.classification]}"
            f" (issue 09 rank {r.rank}: {_short(r.evidence)})"
        )
        for w in str(ids).split(";"):
            if w.strip().lstrip("-").isdigit():
                rows.append(
                    {
                        "way_id": int(w),
                        "osm_n": float(r.osm_n),
                        "model_n": float(r.model_n),
                        "reason": reason,
                        "source": "issue09",
                    }
                )
    return pd.DataFrame(rows)


def changelog_entry(changelog: str, eic: str, name: str) -> str:
    if eic in CHANGELOG_BY_EIC:
        return str(CHANGELOG_BY_EIC[eic])
    for line in changelog.splitlines():
        m = re.match(r"\| (\d+) \|", line)
        if m and (eic in line or name in line):
            return m.group(1)
    return ""


def status_rows(
    model: Path, network: Path, old_network: Path | None, changelog: Path | None
) -> pd.DataFrame:
    silver, _ = pd.read_pickle(model)
    c = silver["grid_de_circuits"]
    nodes = silver["grid_de_nodes"].set_index("node")
    off = c[c["status"].isin(["out_of_service", "decommissioned"])]
    text = changelog.read_text(encoding="utf-8") if changelog is not None else ""
    edges = pd.read_parquet(network / "edges.parquet", columns=["way_id", "ref", "name"])
    edges = edges[edges["way_id"] >= 0].drop_duplicates("way_id")
    old = None
    if old_network is not None and (old_network / "circuit_pieces.parquet").exists():
        old = pd.read_parquet(old_network / "circuit_pieces.parquet", columns=["name", "way_id"])
    pos = None
    if (network / "edges.parquet").exists():
        import shapely

        e2 = pd.read_parquet(network / "edges.parquet", columns=["way_id", "geometry"])
        e2 = e2[e2["way_id"] >= 0].drop_duplicates("way_id")
        mid = shapely.centroid(shapely.from_wkb(e2["geometry"].to_numpy()))
        pos = pd.DataFrame(
            {"lon": shapely.get_x(mid), "lat": shapely.get_y(mid)}, index=e2["way_id"]
        )
    cur = pd.read_parquet(network / "circuit_pieces.parquet", columns=["name", "way_id", "kind"])
    cur = cur[cur["kind"] == "way"].drop_duplicates(["name", "way_id"])
    cur["fold"] = cur["name"].map(_plain)
    per_way: dict[int, list[str]] = {}
    for r in off.itertuples(index=False):
        entry = changelog_entry(text, str(r.eic), str(r.name))
        label = f"{r.name} {str(r.status).replace('_', ' ')} per JAO" + (
            f", CHANGELOG {entry}" if entry else ""
        )
        ways: set[int] = set()
        if old is not None:
            ways = set(old.loc[old["name"] == r.name, "way_id"].astype(int)) - {-1}
        if not ways and pos is not None:
            nums = set(re.findall(r"\b(\d{3,4})\b", str(r.name)))
            ends = [nodes.loc[n] for n in (r.node0, r.node1) if n in nodes.index]
            if nums and len(ends) == 2:
                lo0, lo1 = sorted(float(x["lon"]) for x in ends)
                la0, la1 = sorted(float(x["lat"]) for x in ends)
                tok = edges["ref"].fillna("") + " " + edges["name"].fillna("")
                hit = tok.map(lambda t, nums=nums: bool(nums & set(re.findall(r"\d{3,4}", t))))
                cand = edges.loc[hit, "way_id"].astype(int)
                p = pos.loc[cand]
                box = p["lon"].between(lo0 - 0.05, lo1 + 0.05) & p["lat"].between(
                    la0 - 0.05, la1 + 0.05
                )
                ways = set(p.index[box].astype(int))
        if not ways:  # a sibling in service between the same two stations (tee legs too)
            words = [_fold(nodes.loc[n, "names"]) for n in (r.node0, r.node1) if n in nodes.index]
            if len(words) == 2 and all(words):
                sib = cur[
                    cur["fold"].map(lambda t, w=words: all(any(x in t for x in o) for o in w))
                ]
                ways = set(sib["way_id"].astype(int)) - {-1}
        for w in ways:
            per_way.setdefault(w, []).append(label)
    ec = pd.read_csv(network / "edge_check.csv")
    ec = ec[(ec["class"] == "model_fewer") & ec["way_id"].isin(per_way)]
    rows = []
    for (w, osm_n, model_n), _grp in ec.groupby(["way_id", "osm_n", "model_n"]):
        labels = per_way[int(w)]
        if 0 < osm_n - model_n <= len(labels):
            rows.append(
                {
                    "way_id": int(w),
                    "osm_n": float(osm_n),
                    "model_n": float(model_n),
                    "reason": f"OSM {osm_n:g} / model {model_n:g}: " + "; ".join(labels),
                    "source": "status",
                }
            )
    return pd.DataFrame(rows)


def extra_rows(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    missing = {"way_id", "osm_n", "model_n", "reason"} - set(x.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    src = path.parent.name.split("-")[0] if path.parent.name[:2].isdigit() else path.stem
    if "kind" not in x:
        x["kind"] = ""
    x = x[["way_id", "osm_n", "model_n", "reason", "kind"]].copy()
    x["kind"] = x["kind"].fillna("")
    x["way_id"] = x["way_id"].astype(int)
    x["reason"] = x["reason"].map(lambda t: _short(t, 240))
    x["source"] = f"issue{src}" if src.isdigit() else src
    return x


def drop_routed_ways(rows: pd.DataFrame) -> pd.DataFrame:
    """Keep only the ``routing`` rows of a way that has one (they override explanations)."""
    routing = rows["kind"] == "routing"
    other = rows["way_id"].isin(set(rows.loc[routing, "way_id"])) & ~routing
    return rows[~other].reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--network", type=Path, required=True, help="grid_network run of the map")
    p.add_argument("--old-network", type=Path, help="earlier run (routes of now-off circuits)")
    p.add_argument("--findings", type=Path, help="folder with the reviewed findings_*.csv")
    p.add_argument("--occurrences", type=Path, help="issue-09 occurrences.csv")
    p.add_argument("--changelog", type=Path)
    p.add_argument("--extra", type=Path, action="append", default=[], help="reviewed rows CSV")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    parts = [status_rows(a.model, a.network, a.old_network, a.changelog)]
    if a.occurrences is not None and a.findings is not None:
        parts.append(issue09_rows(a.occurrences, a.findings))
    parts += [extra_rows(x) for x in a.extra]
    out = pd.concat([x for x in parts if len(x)], ignore_index=True)
    if "kind" not in out:
        out["kind"] = ""
    out["kind"] = out["kind"].fillna("")
    out = drop_routed_ways(out).drop_duplicates(["way_id", "osm_n", "model_n"])
    out.to_csv(a.out, index=False)
    print(a.out, len(out), out["source"].value_counts().to_dict())


if __name__ == "__main__":
    main()
