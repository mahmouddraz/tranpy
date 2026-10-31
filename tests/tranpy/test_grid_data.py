from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tranpy.grid_data import load_core_grid_model
from tranpy_core.errors import DataError

LINE_HEADER = [
    "NE_name", "EIC_Code", "TSO", "Full_name", "Full_name", "Voltage_level(kV)",
    *(f"Period {i}" for i in range(1, 7)), "Fixed", "DLRmin(A)", "DLRmax(A)",
    "Resistance_R(Ω)", "Reactance_X(Ω)", "Susceptance_B(μS)", "Length_(km)", "Comment",
]  # fmt: skip
TRAFO_HEADER = [
    "Full Name", "EIC_Code", "TSO", "Min", "Max", "Fixed", "Primary", "Secondary",
    "Resistance_R(Ω)", "Reactance_X(Ω)", "Susceptance_B (µS)", "Conductance_G (µS)",
    "Taps used for RAO", "Theta θ (°)", "Symmetrical/Asymmetrical",
    "Phase Regulation δu (%)", "Angle Regulation δu (%)", "Comment",
]  # fmt: skip


def _line(name: str, eic: str, tso: str, a: str, b: str, fixed: float | None) -> list[object]:
    periods = [None, 1800.0, 2000.0, None, None, None]
    return [name, eic, tso, a, b, 380, *periods, fixed, None, None, 1.0, 10.0, 100.0, 30.0, None]


def _sheet(header: list[str], rows: list[list[object]]) -> pd.DataFrame:
    return pd.DataFrame([[None] * len(header), header, *rows])


def _release(path: Path, *, with_map: bool = True) -> Path:
    tables = {
        "Lines": _sheet(
            LINE_HEADER,
            [
                _line("A - B 1", "L1", "50HERTZ", "A", "B", 2500.0),
                _line("B - C 2", "L2", "AMPRION GMBH", "B", "C", None),
                _line("X - Y 3", "L3", "RTE", "X", "Y", 3000.0),
            ],
        ),
        "Tielines": _sheet(LINE_HEADER, [_line("C - P 4", "T1", "AMPRION GMBH", "C", "P", 2000.0)]),
        "Transformers": _sheet(
            TRAFO_HEADER,
            [["TR B 220/400", "TR1", "50HERTZ", 600, 600, None, 400, 231, 0.7, 60.0, -1.0, 0.8,
              None, None, None, None, None, None]],
        ),
    }  # fmt: skip
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        for name, frame in tables.items():
            frame.to_excel(writer, sheet_name=name, header=False, index=False)

    def label(eic: str) -> str:
        return f"<b>NE name: x</b><br><b>EIC Code:<b> {eic}<br><table></table>"

    routes = {"L1": ([10.0, 11.0], [52.0, 52.5]), "L2": ([11.0, 11.5, 12.0], [52.5, 52.0, 51.0]),
              "T1": ([12.0, 13.0], [51.0, 51.0])}  # fmt: skip
    widget = {
        "x": {
            "calls": [
                {"method": "addTiles", "args": ["", None, "", {}]},
                {
                    "method": "addPolylines",
                    "args": [
                        [[[{"lng": lng, "lat": lat}]] for lng, lat in routes.values()],
                        None, "Lines", {}, [], None,
                        [label(e) for e in routes], {}, None,
                    ],
                },
                {
                    "method": "addCircleMarkers",
                    "args": [[52.5], [11.0], [5], None, "Transformers", {}, None, None,
                             [], None, [label("TR1")], {}],
                },
            ]
        }
    }  # fmt: skip
    html = (
        '<html><script type="application/json" data-for="htmlwidget-1">'
        + json.dumps(widget)
        + "</script></html>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("release/2026_Core Static Grid Model_External.xlsx", buffer.getvalue())
        if with_map:
            archive.writestr("release/2026_publication.html", html)
    return path


def test_load_german_part(tmp_path: Path) -> None:
    model = load_core_grid_model(_release(tmp_path / "sgm.zip"))
    assert model.release == "2026_Core Static Grid Model"
    assert list(model.lines["eic"]) == ["L1", "L2", "T1"]  # French line dropped
    assert list(model.lines["tie_line"]) == [False, False, True]
    assert list(model.lines["imax_a"]) == [2500.0, 2000.0, 2000.0]  # fixed, else largest period
    assert model.lines["x_ohm"].tolist() == [10.0, 10.0, 10.0]
    np.testing.assert_allclose(model.routes["L2"][-1], [12.0, 51.0])
    assert (model.transformers["lon"].iloc[0], model.transformers["lat"].iloc[0]) == (11.0, 52.5)
    stations = model.substations().set_index("substation")
    assert set(stations.index) == {"A", "B", "C", "P"}
    assert tuple(stations.loc["B", ["lon", "lat"]]) == (11.0, 52.5)
    everyone = load_core_grid_model(tmp_path / "sgm.zip", tsos=None)
    assert "L3" in set(everyone.lines["eic"])


def test_bad_release(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="publication map"):
        load_core_grid_model(_release(tmp_path / "sgm.zip", with_map=False))


@pytest.mark.requires_plot
def test_operator_map(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from tranpy.maps import plot_operator_map

    model = load_core_grid_model(_release(tmp_path / "sgm.zip"))
    fig = plot_operator_map(model, borders=False, title="test")
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels[:2] == ["50Hertz (1)", "Amprion (2)"]


def test_numeric_column_is_missing_for_older_releases() -> None:
    from tranpy.grid_data import _numeric_column

    frame = pd.DataFrame({"DLRmin(A)": ["1000", "-"]})
    assert _numeric_column(frame, "DLRmin(A)").iloc[0] == 1000.0
    assert _numeric_column(frame, "DLRmin(A)").isna().iloc[1]
    assert _numeric_column(frame, "DLRmax(A)").isna().all()
