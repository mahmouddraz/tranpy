from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tranpy.osm_routes import (
    WayNetwork,
    _code_in,
    _identifiers,
    check_routes,
    fold,
    place_tees,
    plausible,
    prepare_relations,
    route_circuit,
    voltage_classes,
    walk,
)


def _way(wid: int, points: list[tuple[float, float]], first_node: int, **tags: str) -> dict:
    return {
        "id": wid,
        "nodes": list(range(first_node, first_node + len(points))),
        "geometry": [{"lon": x, "lat": y} for x, y in points],
        "tags": {"power": "line", "voltage": "380000", **tags},
    }


# a line A (8.0, 52.0) -> J (8.3, 52.0) -> B (8.6, 52.0), with a branch J -> C (8.3, 52.3)
WAYS = [
    _way(1, [(8.0, 52.0), (8.15, 52.0), (8.3, 52.0)], 100, name="Alpha - Beta 481"),
    _way(2, [(8.3, 52.0), (8.45, 52.0), (8.6, 52.0)], 102, name="Alpha - Beta 481"),
    _way(3, [(8.3, 52.0), (8.3, 52.15), (8.3, 52.3)], 105, name="Gamma branch"),
]
WAYS[1]["nodes"][0] = 102  # the junction node is shared by ways 1, 2 and 3
WAYS[2]["nodes"][0] = 102


def _relation(rid: int, ref: str) -> dict:
    return {
        "id": rid,
        "tags": {"route": "power", "voltage": "380000", "name": "Alpha - Beta", "ref": ref},
        "members": [{"type": "way", "ref": 1}, {"type": "way", "ref": 2}],
    }


def test_fold_and_code_matching() -> None:
    assert fold("Kühmoos Weißenthurm") == "kuhmos weisenthurm"
    assert _code_in("kondlw", fold("220 kV Kondelwald"))
    assert not _code_in("bigge", fold("Kondelwald"))
    numbers, colours, codes = _identifiers("Hamburg/N 50H - Dollern blau 316")
    assert numbers == {"316"} and "blue" in colours and codes == []


def test_voltage_classes() -> None:
    assert voltage_classes({"voltage": "380000;110000"}) == {380}
    assert voltage_classes({"voltage": "220000;0"}) == {220}
    assert voltage_classes({"voltage": "110000"}) == set()


def test_walk_follows_shared_nodes_and_reaches_the_ends() -> None:
    a, b = np.array([8.0, 52.0]) * [70.0, 111.2], np.array([8.6, 52.0]) * [70.0, 111.2]
    found = walk(WAYS, a, b)
    assert found is not None
    assert found.joins == []
    assert found.km == pytest.approx(0.6 * 70.0, rel=0.02)
    assert np.allclose(found.points[0], [8.0, 52.0]) or np.allclose(found.points[-1], [8.0, 52.0])
    assert plausible(found)


def test_route_circuit_prefers_the_named_relation() -> None:
    ways = {w["id"]: w for w in WAYS}
    relations = prepare_relations(
        [
            _relation(9, "481"),
            _relation(10, "482"),
        ],
        ways,
        set(),
    )  # fmt: skip
    network = WayNetwork.build(WAYS, 380, set())
    out = route_circuit(
        "Alpha - Beta 482", 380, np.array([8.0, 52.0]), np.array([8.6, 52.0]), relations, network
    )
    assert out is not None
    _found, method, identity, relation = out
    assert (method, identity, relation["id"]) == ("circuit record", "named", 10)
    assert (
        route_circuit(
            "Alpha - Beta 482",
            380,
            np.array([8.0, 52.0]),
            np.array([9.9, 52.0]),
            relations,
            network,
        )
        is None
    )


def test_place_tee_at_the_junction() -> None:
    lines = pd.DataFrame(
        {"node0": ["alpha|380", "y-j|380", "y-j|380"], "node1": ["y-j|380", "beta|380", "gamma|380"],
         "kv": [380.0] * 3, "length_km": [21.0, 21.0, 23.0]}
    )  # fmt: skip
    positions = {"alpha|380": [8.0, 52.0], "beta|380": [8.6, 52.0], "gamma|380": [8.3, 52.3]}
    placed, report = place_tees(
        ["y-j|380"], lines, positions, {380: WayNetwork.build(WAYS, 380, set())}
    )
    assert np.allclose(placed["y-j|380"], [8.3, 52.0], atol=0.01)
    assert report.iloc[0]["how"].startswith("junction")


def test_check_routes_labels() -> None:
    path = np.array([[8.0, 52.0], [8.3, 52.0], [8.6, 52.0]])
    routes = pd.DataFrame(
        [
            {
                "key": "k",
                "km": 42.0,
                "x_ohm": 11.3,
                "kv": 380,
                "identity": "named",
                "joins": 0,
                "border": False,
            }
        ]
    )
    out = check_routes(routes, {"k": path}, {"dlm250": [path]}, {"k": (True, True)})
    assert out.iloc[0]["proposal"] == "mapped, consistent"
    out = check_routes(
        routes.assign(x_ohm=5.0), {"k": path}, {"dlm250": [path]}, {"k": (True, True)}
    )
    assert out.iloc[0]["proposal"] == "length vs reactance off"
