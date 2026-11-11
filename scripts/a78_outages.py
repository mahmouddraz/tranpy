"""Hourly outage set of German 220/380 kV circuits from ENTSO-E A78 (transmission unavailability).

    python scripts/a78_outages.py HOURS_CSV MODEL_PKL OUT_CSV XML_DIR [XML_DIR ...]

``XML_DIR``: folders of A78 ``Unavailability_MarketDocument`` XML files (unzipped API answers,
fetched with the token from the environment; the token is never stored). Each document keeps
only its latest revision; cancelled documents (docStatus A09) are dropped. An asset is out in a
model hour when the document's unavailability interval overlaps that hour. Assets are matched to
model circuits by EIC (``ac_line_segment.eic`` -> ``mrid``); only in-operation circuits of
220/380 kV are kept. ``HOURS_CSV`` has ``time`` (UTC hour start) and ``name``.

Output: one row per (hour, circuit): hour, time, circuit (model mrid), name, eic, a78_mrid
(document mRID), business_type (A53 planned / A54 forced), reason (code), start, end.
ENTSO-E Transparency data are research-only; the CSV holds only the model hours.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

COLUMNS = [
    "hour", "time", "circuit", "name", "eic", "a78_mrid",
    "business_type", "reason", "start", "end",
]  # fmt: skip


def parse_document(xml: str | bytes) -> list[dict]:
    """Assets and interval of one A78 document (one row per asset)."""
    root = ET.fromstring(xml)
    ns = {"n": root.tag.split("}")[0].strip("{")}

    def text(node: ET.Element, path: str) -> str | None:
        hit = node.find(path, ns)
        return hit.text.strip() if hit is not None and hit.text else None

    base = {
        "a78_mrid": text(root, "n:mRID"),
        "revision": int(text(root, "n:revisionNumber") or 0),
        "doc_status": text(root, "n:docStatus/n:value"),
        "start": pd.Timestamp(text(root, "n:unavailability_Time_Period.timeInterval/n:start")),
        "end": pd.Timestamp(text(root, "n:unavailability_Time_Period.timeInterval/n:end")),
        "reason": text(root, "n:Reason/n:code"),
    }
    rows = []
    for ts in root.iterfind("n:TimeSeries", ns):
        for asset in ts.iterfind(".//n:Asset_RegisteredResource", ns):
            rows.append(
                base
                | {
                    "business_type": text(ts, "n:businessType"),
                    "eic": text(asset, "n:mRID"),
                    "asset_name": text(asset, "n:name"),
                }
            )
    return rows


def latest_documents(rows: pd.DataFrame) -> pd.DataFrame:
    """Latest revision of each document, cancelled (A09) ones dropped."""
    if rows.empty:
        return rows
    top = rows.groupby("a78_mrid")["revision"].transform("max")
    rows = rows[rows["revision"].eq(top)]
    return rows[rows["doc_status"].fillna("").ne("A09")].drop_duplicates(
        ["a78_mrid", "eic", "start", "end"]
    )


def hourly_outages(docs: pd.DataFrame, hours: pd.DataFrame, circuits: pd.DataFrame) -> pd.DataFrame:
    """(hour, circuit) rows for every model hour that an outage interval overlaps.

    ``circuits``: mrid, eic, name, kv, status. Circuits not in operation or below 220 kV are
    left out; several model rows sharing an EIC (tee legs) all go out.
    """
    lines = circuits[
        circuits["status"].fillna("in_operation").eq("in_operation")
        & circuits["kv"].astype(float).ge(220)
        & circuits["eic"].notna()
    ]
    hit = docs.merge(lines[["mrid", "eic", "name"]], on="eic")
    out = []
    for name, t in zip(hours["name"], pd.to_datetime(hours["time"], utc=True), strict=True):
        over = hit[(hit["start"] < t + pd.Timedelta(hours=1)) & (hit["end"] > t)]
        for r in over.itertuples():
            out.append(
                {
                    "hour": name,
                    "time": t,
                    "circuit": r.mrid,
                    "name": r.name,
                    "eic": r.eic,
                    "a78_mrid": r.a78_mrid,
                    "business_type": r.business_type,
                    "reason": r.reason,
                    "start": r.start,
                    "end": r.end,
                }
            )
    frame = pd.DataFrame(out, columns=COLUMNS)
    return frame.drop_duplicates(["hour", "circuit"]).sort_values(["time", "circuit"])


def outages_by_hour(path: str | Path) -> dict[str, list[str]]:
    """Circuit mrids out per hour name, from the CSV written by :func:`main`."""
    frame = pd.read_csv(path)
    return {str(h): sorted(g["circuit"].astype(str).unique()) for h, g in frame.groupby("hour")}


def main(hours_csv: str, model_pkl: str, out_csv: str, *xml_dirs: str) -> None:
    rows = []
    for d in xml_dirs:
        for f in sorted(Path(d).rglob("*.xml")):
            if not f.name.startswith("._"):  # skip macOS resource forks
                rows += parse_document(f.read_bytes())
    docs = latest_documents(pd.DataFrame(rows))
    _, tables = pd.read_pickle(model_pkl)
    circuits = tables.ac_line_segment.rename(columns={"nominal_voltage_kv": "kv"})
    out = hourly_outages(docs, pd.read_csv(hours_csv), circuits)
    out.to_csv(out_csv, index=False)
    n_assets, n_circuits = docs["eic"].nunique(), out["circuit"].nunique()
    print(f"{len(docs)} asset-documents, {n_assets} assets, {n_circuits} model circuits")
    print(out.groupby("name").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
