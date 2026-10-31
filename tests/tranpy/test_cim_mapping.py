from __future__ import annotations

import pandas as pd
import pytest

from tranpy.cim.mapping import conform, geometry, network
from tranpy.grid_layers import mrid
from tranpy_core.errors import DataError

NODES = pd.DataFrame(
    {"node": ["a|380", "b|380", "b|220", "y-t#x|380"], "site": ["a", "b", "b", "y-t#x"],
     "kv": [380, 380, 220, 380], "tso": ["TENNETGMBH", "TENNETGMBH", "TENNETGMBH", "TENNETGMBH"],
     "kind": ["substation", "substation", "substation", "tee"], "lon": [8.0, 8.6, 8.6, 8.3],
     "lat": [52.0, 52.0, 52.0, 52.0], "country": ["DEU"] * 4}
)  # fmt: skip
CIRCUITS = pd.DataFrame(
    [{"eic": "E1", "name": "A - T x", "node0": "a|380", "node1": "y-t#x|380", "kv": 380.0,
      "length_km": 20.0, "r_ohm": 0.5, "x_ohm": 5.0, "b_us": 60.0, "imax_a": 2000.0, "hvdc": False,
      "status": "in_operation", "route": [{"lon": 8.0, "lat": 52.0}, {"lon": 8.3, "lat": 52.0}]},
     {"eic": "E2", "name": "old", "node0": "y-t#x|380", "node1": "b|380", "kv": 380.0,
      "length_km": 20.0, "r_ohm": 0.5, "x_ohm": 5.0, "b_us": None, "imax_a": None, "hvdc": False,
      "status": "out_of_service", "route": []}]
)  # fmt: skip
TRANSFORMERS = pd.DataFrame(
    [{"eic": "T1", "name": "TR b", "node_hv": "b|380", "node_lv": "b|220", "kv_hv": 380.0,
      "kv_lv": 220.0, "r_ohm": 0.3, "x_ohm": 30.0, "b_us": 0.0, "g_us": 0.0}]
)  # fmt: skip


def _rows() -> dict:
    return conform({**network(NODES, CIRCUITS, TRANSFORMERS), **geometry(NODES, CIRCUITS)})


def test_network_conforms_and_links_by_mrid() -> None:
    rows = _rows()
    assert len(rows["Substation"]) == 2
    assert len(rows["VoltageLevel"]) == 3
    assert len(rows["BusbarSection"]) == 3
    assert len(rows["ConnectivityNode"]) == 4
    assert set(rows["ACLineSegment"]["BaseVoltage"]) <= set(rows["BaseVoltage"]["mRID"])
    terminals = rows["Terminal"]
    assert len(terminals) == 2 * 2 + 2 + 3
    equipment = pd.concat(
        [rows[k]["mRID"] for k in ("ACLineSegment", "PowerTransformer", "BusbarSection")]
    )
    assert set(terminals["ConductingEquipment"]) == set(equipment)
    assert set(terminals["ConnectivityNode"]) <= set(rows["ConnectivityNode"]["mRID"])


def test_status_and_values() -> None:
    line = _rows()["ACLineSegment"].set_index("name")
    assert not line.loc["old", "inService"]
    assert line.loc["A - T x", "r"] == 0.5
    assert line.loc["A - T x", "bch"] == pytest.approx(60e-6)
    assert line.loc["old", "bch"] == 0.0


def test_transformer_impedance_on_the_high_voltage_end() -> None:
    ends = _rows()["PowerTransformerEnd"].set_index("endNumber")
    assert ends.loc[1, "x"] == 30.0 and ends.loc[1, "ratedU"] == 380.0
    assert ends.loc[2, "x"] == 0.0 and ends.loc[2, "ratedU"] == 220.0


def test_only_published_ratings_become_limits() -> None:
    rows = _rows()
    assert rows["CurrentLimit"]["normalValue"].tolist() == [2000.0]


def test_geometry_keeps_route_order() -> None:
    rows = _rows()
    location = mrid("location", mrid("line", "E1", "a|380", "y-t#x|380"))
    points = rows["PositionPoint"][rows["PositionPoint"]["Location"] == location]
    assert points["sequenceNumber"].tolist() == [1, 2]
    assert points["xPosition"].tolist() == ["8.0", "8.3"]


def test_conform_rejects_what_is_not_in_the_standard() -> None:
    with pytest.raises(DataError, match="not a CGMES class"):
        conform({"Pylon": pd.DataFrame({"mRID": ["x"]})})
    with pytest.raises(DataError, match="not in the schema"):
        conform({"Substation": pd.DataFrame({"mRID": ["x"], "status": ["y"]})})
    with pytest.raises(DataError, match=r"Substation\.Region"):
        conform({"Substation": pd.DataFrame({"mRID": ["x"], "name": ["y"]})})


def test_records_follow_the_column_types() -> None:
    from tranpy.cim.fill import to_records
    from tranpy.cim.schema import tables

    table = next(t for t in tables() if t.name == "Terminal")
    rows = _rows()["Terminal"]
    records = to_records(rows, table)
    names = [c.name for c in table.columns]
    first = dict(zip(names, records[0], strict=True))
    assert isinstance(first["sequenceNumber"], int)
    assert first["connected"] is True
    assert first["Measurements"] is None
    many = [c for t in tables() for c in t.columns if c.sql_type.startswith("ARRAY")][0]
    frame = pd.DataFrame({many.name: ["x", None]})
    fake = type(table)("Fake", (), (), [many])
    assert to_records(frame, fake) == [(["x"],), (None,)]
