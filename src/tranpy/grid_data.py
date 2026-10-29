"""Published grid data of the transmission system operators.

* :func:`load_core_grid_model` - the Core Static Grid Model published by JAO for the TSOs
  of the Core capacity calculation region (all four German TSOs, France, Benelux, Poland,
  Czechia, Austria, ...): every relevant line, tie-line and transformer with its operator,
  substations, voltage, resistance, reactance, susceptance, length and current rating,
  plus the geographic route of each line. Updated every six months.

Not included in the source: generators, loads, spur and generator connection lines, and
dynamic data. Parameters are given in ohm and microsiemens per circuit.
"""

from __future__ import annotations

import io
import json
import re
import unicodedata
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tranpy_core.data.download import RemoteFile, fetch
from tranpy_core.errors import DataError

__all__ = ["CORE_SGM", "GERMAN_TSOS", "CoreGridModel", "load_core_grid_model"]

CORE_SGM = RemoteFile(
    filename="core_static_grid_model_202603.zip",
    url=(
        "https://www.jao.eu/sites/default/files/2026-06/"
        "202603_Core%20Static%20Grid%20Model_9th%20release.zip"
    ),
    sha256="e3becdb02f51450e3ef2ab7f5a1fe7b35341873b3e156bd580ac67c0eeba4f39",
)
"""Core Static Grid Model, 9th release (status March 2026)."""

GERMAN_TSOS: dict[str, str] = {
    "50HERTZ": "50Hertz",
    "AMPRION GMBH": "Amprion",
    "TENNETGMBH": "TenneT DE",
    "TRANSNETBW": "TransnetBW",
}
"""Operator codes used in the source -> short names."""

_LINE_COLUMNS = {
    "NE_name": "name",
    "EIC_Code": "eic",
    "TSO": "tso",
    "Full_name": "from_substation",
    "Full_name.1": "to_substation",
    "Voltage_level(kV)": "kv",
    "Fixed": "imax_fixed_a",
    "Resistance_R(Ω)": "r_ohm",
    "Reactance_X(Ω)": "x_ohm",
    "Susceptance_B(μS)": "b_us",
    "Length_(km)": "length_km",
}
_TRANSFORMER_COLUMNS = {
    "Full Name": "name",
    "EIC_Code": "eic",
    "TSO": "tso",
    "Max": "imax_a",
    "Primary": "kv_primary",
    "Secondary": "kv_secondary",
    "Resistance_R(Ω)": "r_ohm",
    "Reactance_X(Ω)": "x_ohm",
    "Susceptance_B (μS)": "b_us",
    "Conductance_G (μS)": "g_us",
    "Theta θ (°)": "phase_shift_deg",
}


@dataclass(frozen=True)
class CoreGridModel:
    """Lines and transformers of the Core Static Grid Model.

    Attributes:
        lines: one row per circuit (``tie_line`` marks interconnectors). Current rating
            ``imax_a`` is the fixed rating, else the largest seasonal rating.
        transformers: one row per transformer with its location (``lon``, ``lat``).
        routes: EIC code -> ``(n_points, 2)`` route of the line (longitude, latitude).
        release: release label of the source.
    """

    lines: pd.DataFrame
    transformers: pd.DataFrame
    routes: dict[str, np.ndarray]
    release: str

    def substations(self) -> pd.DataFrame:
        """Substations with a location, taken from the ends of the line routes."""
        rows = []
        for line in self.lines.itertuples():
            route = self.routes.get(str(line.eic))
            if route is None:
                continue
            for name, point in ((line.from_substation, route[0]), (line.to_substation, route[-1])):
                rows.append((name, line.tso, float(point[0]), float(point[1])))
        frame = pd.DataFrame(rows, columns=["substation", "tso", "lon", "lat"])
        return frame.groupby("substation", as_index=False).agg(
            tso=("tso", "first"), lon=("lon", "median"), lat=("lat", "median")
        )


def _normalised(frame: pd.DataFrame) -> pd.DataFrame:
    # the source mixes the ohm sign (U+2126) with omega and micro (U+00B5) with mu
    return frame.rename(columns=lambda c: unicodedata.normalize("NFKC", str(c)).strip())


def _read_sheet(raw: bytes, sheet: str, columns: dict[str, str]) -> pd.DataFrame:
    frame = _normalised(pd.read_excel(io.BytesIO(raw), sheet_name=sheet, header=1))
    wanted = {unicodedata.normalize("NFKC", k): v for k, v in columns.items()}
    missing = sorted(set(wanted) - set(frame.columns))
    if missing:
        raise DataError(f"sheet {sheet!r} lacks columns {missing}")
    text = {"NE_name", "EIC_Code", "TSO", "Full_name", "Full_name.1", "Full Name"}
    numeric = [c for c in frame.columns if c in wanted and c not in text]
    numeric += [c for c in frame.columns if c.startswith("Period ")]
    # ratings and parameters are sometimes given as text placeholders ("-", "n/a")
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if sheet in ("Lines", "Tielines"):
        periods = frame[[c for c in frame.columns if c.startswith("Period ")]]
        frame["imax_a"] = frame["Fixed"].fillna(periods.max(axis=1))
    out = frame.rename(columns=wanted)
    keep = [
        *wanted.values(),
        *(["imax_a"] if "imax_a" in out and "imax_a" not in wanted.values() else []),
    ]
    return out[keep].dropna(subset=["eic"]).reset_index(drop=True)


def _map_layers(html: str) -> tuple[dict[str, np.ndarray], dict[str, tuple[float, float]]]:
    blobs = re.findall(
        r'<script type="application/json" data-for="[^"]+">(.*?)</script>', html, re.S
    )
    if not blobs:
        raise DataError("no map data found in the publication HTML")
    calls: list[dict[str, Any]] = json.loads(max(blobs, key=len))["x"]["calls"]
    eic = re.compile(r"EIC Code:<b>\s*([^<\s]+)")
    routes: dict[str, np.ndarray] = {}
    for call in calls:
        if call["method"] != "addPolylines":
            continue
        shapes, labels = call["args"][0], call["args"][6]
        for shape, label in zip(shapes, labels, strict=True):
            found = eic.search(label or "")
            if found is None:
                continue
            part = shape[0][0]
            routes.setdefault(found.group(1), np.column_stack([part["lng"], part["lat"]]))
    transformers: dict[str, tuple[float, float]] = {}
    for call in calls:
        if call["method"] == "addCircleMarkers" and call["args"][4] == "Transformers":
            lat, lon, labels = call["args"][0], call["args"][1], call["args"][10]
            for y, x, label in zip(lat, lon, labels, strict=True):
                found = eic.search(label or "")
                if found is not None:
                    transformers[found.group(1)] = (float(x), float(y))
    return routes, transformers


def load_core_grid_model(
    source: str | Path | None = None, *, tsos: Sequence[str] | None = tuple(GERMAN_TSOS)
) -> CoreGridModel:
    """Read the Core Static Grid Model (downloaded and checksum-verified if ``source`` is None).

    Args:
        source: the release zip archive.
        tsos: operator codes to keep (default: the four German TSOs; None keeps all).
    """
    path = Path(source) if source is not None else fetch(CORE_SGM)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        table = next((n for n in names if n.endswith("_External.xlsx")), None)
        page = next((n for n in names if n.endswith(".html")), None)
        if table is None or page is None:
            raise DataError("the archive lacks the model table or the publication map")
        raw = archive.read(table)
        html = archive.read(page).decode("utf-8")
    lines = pd.concat(
        [
            _read_sheet(raw, "Lines", _LINE_COLUMNS).assign(tie_line=False),
            _read_sheet(raw, "Tielines", _LINE_COLUMNS).assign(tie_line=True),
        ],
        ignore_index=True,
    )
    transformers = _read_sheet(raw, "Transformers", _TRANSFORMER_COLUMNS)
    if tsos is not None:
        lines = lines[lines["tso"].isin(tsos)].reset_index(drop=True)
        transformers = transformers[transformers["tso"].isin(tsos)].reset_index(drop=True)
    routes, located = _map_layers(html)
    transformers["lon"] = [located.get(e, (np.nan, np.nan))[0] for e in transformers["eic"]]
    transformers["lat"] = [located.get(e, (np.nan, np.nan))[1] for e in transformers["eic"]]
    return CoreGridModel(
        lines=lines,
        transformers=transformers,
        routes={e: routes[e] for e in lines["eic"] if e in routes},
        release=Path(table).stem.replace("_External", ""),
    )
