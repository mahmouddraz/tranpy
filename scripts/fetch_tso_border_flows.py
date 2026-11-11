"""Fetch ENTSO-E A11 physical flows per German control-area border (TSO x neighbour).

    ENTSOE_API_TOKEN=... python scripts/fetch_tso_border_flows.py HOURS_CSV OUT_CSV

The token is read from the environment only; it is never printed or written. ``HOURS_CSV`` has
a ``time`` column (UTC hour starts). For every border segment listed in :data:`SEGMENTS`, both
directions are fetched for the window spanned by the hours (in chunks of at most one year) and
the hourly mean is kept for the requested hours. Output: one row per hour, segment, with
``mw`` = flow out of Germany (DE -> neighbour) minus flow into Germany, MW; ``mw`` is empty
(NaN) when neither direction has a published value for that hour, never 0.

ENTSO-E Transparency data are research-only: the CSV holds only the hours used by the model.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET

import pandas as pd

API = "https://web-api.tp.entsoe.eu/api"
TSO = {
    "TenneT": "10YDE-EON------1",
    "50Hertz": "10YDE-VE-------2",
    "Amprion": "10YDE-RWENET---I",
    "TransnetBW": "10YDE-ENBW-----N",
}
NEIGHBOUR = {
    "AT": "10YAT-APG------L",
    "BE": "10YBE----------2",
    "CH": "10YCH-SWISSGRIDZ",
    "CZ": "10YCZ-CEPS-----N",
    "DK1": "10YDK-1--------W",
    "DK2": "10YDK-2--------M",
    "FR": "10YFR-RTE------C",
    "LU": "10YLU-CEGEDEL-NQ",
    "NL": "10YNL----------L",
    "NO2": "10YNO-2--------T",
    "PL": "10YPL-AREA-----S",
    "SE4": "10Y1001A1001A47J",
}
SEGMENTS = [
    ("CZ", "TenneT"),
    ("CZ", "50Hertz"),
    ("AT", "TenneT"),
    ("AT", "Amprion"),
    ("AT", "TransnetBW"),
    ("PL", "50Hertz"),
    ("NL", "TenneT"),
    ("NL", "Amprion"),
    ("DK1", "TenneT"),
    ("DK2", "50Hertz"),
    ("CH", "Amprion"),
    ("CH", "TransnetBW"),
    ("FR", "Amprion"),
    ("FR", "TransnetBW"),
    ("LU", "Amprion"),
    ("BE", "Amprion"),
    ("SE4", "TenneT"),
    ("SE4", "50Hertz"),
    ("NO2", "TenneT"),
]
"""Border segments queried; a segment with no published data is reported as missing."""


def _get(params: dict[str, str]) -> str | None:
    """One API call through curl (system trust store); the URL, which holds the token, is
    passed on stdin, not on the command line."""
    token = os.environ["ENTSOE_API_TOKEN"]
    url = API + "?" + urllib.parse.urlencode({"securityToken": token, **params})
    for attempt in range(4):
        r = subprocess.run(
            ["curl", "-s", "-K", "-", "-w", "\n%{http_code}"],
            input=f'url = "{url}"\n',
            capture_output=True,
            text=True,
            timeout=180,
        )
        body, _, code = r.stdout.rpartition("\n")
        if code == "200":
            return body
        if code in ("400", "404"):
            return None
        time.sleep(5 * (attempt + 1))
    return None


def _series(xml: str | None) -> pd.Series:
    if not xml or "Acknowledgement_MarketDocument" in xml[:300]:
        return pd.Series(dtype=float)
    root = ET.fromstring(xml)
    ns = {"n": root.tag.split("}")[0].strip("{")}
    out = {}
    for period in root.iterfind(".//n:Period", ns):
        start = pd.Timestamp(period.find("n:timeInterval/n:start", ns).text)
        res = period.find("n:resolution", ns).text
        step = pd.Timedelta(minutes=15) if res == "PT15M" else pd.Timedelta(hours=1)
        for p in period.iterfind("n:Point", ns):
            pos = int(p.find("n:position", ns).text)
            out[start + (pos - 1) * step] = float(p.find("n:quantity", ns).text)
    s = pd.Series(out, dtype=float).sort_index()
    return s.groupby(s.index.floor("h")).mean()


def _flow(src: str, dst: str, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.Series:
    parts = []
    a = t0
    while a < t1:
        b = min(a + pd.Timedelta(days=360), t1)
        xml = _get(
            {
                "documentType": "A11",
                "out_Domain": src,
                "in_Domain": dst,
                "periodStart": a.strftime("%Y%m%d%H%M"),
                "periodEnd": b.strftime("%Y%m%d%H%M"),
            }
        )
        parts.append(_series(xml))
        a = b
    parts = [p for p in parts if len(p)]
    return pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)


def segment_flow(out: pd.Series, back: pd.Series, hours: pd.Series) -> pd.Series:
    """Net flow out of Germany per hour: ``out`` minus ``back``.

    A11 publishes each direction only for the hours with flow that way, so one direction absent
    while the other has a value means no flow that way (0 MW). An hour with no value in either
    direction is NaN (missing) and is never read as 0 MW; the dispatch then falls back to the
    capacity split for that border.
    """
    idx = pd.DatetimeIndex(hours)
    a, b = out.reindex(idx), back.reindex(idx)
    net = a.fillna(0.0) - b.fillna(0.0)
    return net.where(a.notna() | b.notna())


def main(hours_csv: str, out_csv: str) -> None:
    hours = pd.to_datetime(pd.read_csv(hours_csv)["time"], utc=True)
    t0 = hours.min().floor("D")
    t1 = hours.max().floor("D") + pd.Timedelta(days=1)
    rows = []
    for country, tso in SEGMENTS:
        out = _flow(TSO[tso], NEIGHBOUR[country], t0, t1)
        back = _flow(NEIGHBOUR[country], TSO[tso], t0, t1)
        mw = segment_flow(out, back, hours)
        for h in hours:
            rows.append({"time": h, "country": country, "tso": tso, "mw": mw[h]})
        print(f"{country}-{tso}: {int(mw.notna().sum())}/{len(hours)} hours published", flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
