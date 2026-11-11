from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "a78_outages.py"
_SPEC = importlib.util.spec_from_file_location("a78_outages", _PATH)
a78 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(a78)

_DOC = """<?xml version="1.0" encoding="utf-8"?>
<Unavailability_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:outagedocument:3:0">
  <mRID>DOC1</mRID><revisionNumber>{rev}</revisionNumber>{status}
  <unavailability_Time_Period.timeInterval><start>2026-01-22T07:00Z</start>
  <end>2026-01-22T09:30Z</end></unavailability_Time_Period.timeInterval>
  <TimeSeries><businessType>A53</businessType>
    <Asset_RegisteredResource><mRID codingScheme="A01">EIC-1</mRID><name>L1</name>
    </Asset_RegisteredResource></TimeSeries>
  <Reason><code>B19</code></Reason>
</Unavailability_MarketDocument>"""


def test_parse_keeps_latest_revision_and_drops_cancelled() -> None:
    rows = a78.parse_document(_DOC.format(rev=1, status="")) + a78.parse_document(
        _DOC.format(rev=2, status="<docStatus><value>A09</value></docStatus>")
    )
    assert rows[0]["eic"] == "EIC-1" and rows[0]["reason"] == "B19"
    assert a78.latest_documents(pd.DataFrame(rows)).empty
    kept = a78.latest_documents(pd.DataFrame(rows[:1]))
    assert len(kept) == 1


def test_hourly_outages_overlap_voltage_and_status(tmp_path) -> None:
    docs = a78.latest_documents(pd.DataFrame(a78.parse_document(_DOC.format(rev=1, status=""))))
    circuits = pd.DataFrame(
        {
            "mrid": ["m1", "m2", "m3"],
            "eic": ["EIC-1", "EIC-1", "EIC-1"],
            "name": ["L1 leg a", "L1 leg b", "L1 110"],
            "kv": [380, 380, 110],
            "status": ["in_operation", "in_operation", "in_operation"],
        }
    )
    hours = pd.DataFrame(
        {
            "name": ["before", "first", "partial", "after"],
            "time": [
                "2026-01-22 06:00:00+00:00",
                "2026-01-22 07:00:00+00:00",
                "2026-01-22 09:00:00+00:00",
                "2026-01-22 10:00:00+00:00",
            ],
        }
    )
    out = a78.hourly_outages(docs, hours, circuits)
    assert set(out["hour"]) == {"first", "partial"}
    assert set(out["circuit"]) == {"m1", "m2"}  # tee legs share the EIC; 110 kV left out
    assert (out["a78_mrid"] == "DOC1").all()
    path = tmp_path / "o.csv"
    out.to_csv(path, index=False)
    assert a78.outages_by_hour(path) == {"first": ["m1", "m2"], "partial": ["m1", "m2"]}
