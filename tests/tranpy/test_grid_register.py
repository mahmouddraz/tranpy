from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tranpy.grid_register import (
    OFFSHORE_FUEL,
    aggregate_units,
    area_load_shares,
    attach_register,
    connection_key,
    node_control_areas,
    offshore_connections,
)


def _nodes() -> pd.DataFrame:
    rows = [
        ("near|380", "near", "TENNETGMBH", 8.0, 53.0),
        ("far|380", "far", "AMPRION GMBH", 8.01, 53.01),
        ("emden/ost|380", "emden/ost", "TENNETGMBH", 7.2, 53.4),
        ("lubmin|380", "lubmin", "50HERTZ", 13.6, 54.1),
    ]
    return pd.DataFrame(
        [
            {"node": n, "site": s, "kv": 380, "tso": t, "kind": "substation", "lon": x, "lat": y}
            for n, s, t, x, y in rows
        ]
    )


def _cell(fuel: str, zone: str | None, connection: str | None = None, lon: float = 8.02) -> dict:
    return {"fuel": fuel, "mw": 10.0, "units": 2, "lon": lon, "lat": 53.02, "zone": zone,
            "how": "register coordinates", "connection": connection}  # fmt: skip


def test_connection_key() -> None:
    assert connection_key("L-155-HOSE-BWGA-AC146") == "BWGA"
    assert connection_key("Offshore Plattform Arkona") == "Offshore Plattform Arkona"
    assert connection_key(None) is None


def test_every_offshore_connection_has_evidence() -> None:
    table = offshore_connections()
    assert table["connection"].is_unique
    assert table["evidence"].str.len().gt(10).all()


def test_units_stay_in_their_control_area() -> None:
    cells = pd.DataFrame([_cell("Wind", "TenneT"), _cell("Wind", None)])
    out = attach_register(cells, _nodes()).set_index("method")
    assert out.loc["nearest in control area", "node"] == "near|380"
    assert out.loc["nearest substation", "node"] == "far|380"
    assert out["capacity_mw"].sum() == pytest.approx(20.0)
    assert set(out["data_quality"]) == {"estimated"}


def test_offshore_uses_sourced_connection_else_spreads_over_sea() -> None:
    cells = pd.DataFrame(
        [_cell(OFFSHORE_FUEL, "TenneT", "BWGA"), _cell(OFFSHORE_FUEL, None, None, lon=6.0)]
    )
    out = attach_register(cells, _nodes())
    sourced = out[out["method"] == "offshore connection"].iloc[0]
    assert sourced["node"] == "emden/ost|380"
    assert sourced["data_quality"] == "published"
    assert out["capacity_mw"].sum() == pytest.approx(20.0)


def test_aggregation_keeps_capacity_and_counts() -> None:
    units = pd.DataFrame(
        {"fuel": ["Wind"] * 3, "mw": [1.0, 2.0, 4.0], "lon": [8.001, 8.002, np.nan],
         "lat": [53.0, 53.0, 53.0], "zone": ["TenneT"] * 3, "how": ["x"] * 3,
         "connection": [None] * 3}
    )  # fmt: skip
    out = aggregate_units(units)
    assert len(out) == 1
    assert out.iloc[0]["mw"] == 3.0
    assert out.iloc[0]["units"] == 2


def test_area_load_divides_each_area_by_population() -> None:
    shares = pd.DataFrame(
        {"node": ["near|380", "emden/ost|380", "far|380"], "population": [1.0, 3.0, 8.0],
         "cells": [1, 1, 1], "share": [0.1, 0.2, 0.7]}
    )  # fmt: skip
    out, total = area_load_shares(shares, _nodes(), {"TenneT": 40.0, "Amprion": 60.0})
    assert total == 100.0
    assert out["share"].tolist() == pytest.approx([0.1, 0.3, 0.6])


def test_area_without_nodes_is_an_error() -> None:
    shares = pd.DataFrame({"node": ["near|380"], "population": [1.0], "cells": [1], "share": [1.0]})
    with pytest.raises(ValueError, match="control areas"):
        area_load_shares(shares, _nodes(), {"TenneT": 40.0, "Amprion": 60.0})


def test_a_node_of_two_operators_takes_the_area_with_the_capacity_around_it() -> None:
    nodes = pd.DataFrame(
        {
            "node": ["border|380", "plain|380"],
            "tso": ["AMPRION GMBH;TENNETGMBH", "AMPRION GMBH"],
            "lon": [7.0, 9.0],
            "lat": [53.0, 51.0],
        }
    )
    cells = pd.DataFrame(
        {
            "lon": [7.0, 7.0, 7.0],
            "lat": [53.0, 53.0, 53.0],
            "fuel": ["Windenergie (Onshore-Anlage)", "Biomasse", OFFSHORE_FUEL],
            "zone": ["TenneT", "Amprion", "Amprion"],
            "mw": [100.0, 10.0, 5000.0],
        }
    )
    assert node_control_areas(nodes).to_dict() == {"border|380": "Amprion", "plain|380": "Amprion"}
    assert node_control_areas(nodes, cells)["border|380"] == "TenneT"


def test_shared_cells_respect_a_node_limit_and_keep_all_capacity() -> None:
    from tranpy.grid_register import _shared

    cells = pd.DataFrame({"lon": [10.0], "lat": [51.0], "mw": [100.0]})
    pool = pd.DataFrame({"node": ["a", "b"], "lon": [10.0, 10.1], "lat": [51.0, 51.0]})
    out = _shared(cells, pool, pd.Series({"a": 20.0})).set_index("node")["share"]
    assert out.sum() == pytest.approx(1.0)
    assert out["a"] * 100 == pytest.approx(20.0)
