"""Simulation cases built from open grid data (optional ``[andes]`` extra).

* :func:`build_germany_case` - the German transmission grid (220 and 380 kV) from
  `PanTaGruEl <https://doi.org/10.5281/zenodo.2642175>`_ (Pagnier & Jacquod, PLOS ONE
  14(3): e0213550, 2019; CC BY 4.0) as an ANDES case. The bundled case
  (``AndesEngine(system_name="germany")``) was written by this function with its
  defaults.

How the German case is derived (see ``docs/germany_model.md``):

* buses with country ``DE``; lines to neighbouring countries removed, so Germany is
  simulated as an island; the largest connected part is kept;
* line resistance and reactance as given (per unit on 100 MVA; no line charging);
  lines between 220 and 380 kV buses are marked as transformers;
* loads from the bus demand, power factor ``power_factor`` (the source has no reactive
  demand);
* every generator dispatched at the same share of its capacity, so generation covers the
  demand plus ``loss_margin`` (concentrated merit-order dispatch makes the AC power flow
  fail without the neighbours' support);
* generators as classical machines (GENCLS): the source gives inertia m and primary
  control d per generator in MW s^2 and MW s on 100 MVA; on the machine base
  H = m * 2 pi f0 * Sb / (2 Sn) and D = d * 2 pi f0 * Sb / Sn. Transient reactance
  ``xd1`` (machine base) is not in the source.
"""

from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from tranpy_core._optional import require
from tranpy_core.data.download import RemoteFile, fetch
from tranpy_core.errors import DataError, SimulationError

__all__ = ["PANTAGRUEL", "build_germany_case", "germany_case_path"]

PANTAGRUEL = RemoteFile(
    filename="pantagruel.zip",
    url="https://zenodo.org/records/2642175/files/pantagruel.zip?download=1",
    sha256="e0b38e1e62fea6fe9a2d962ab66a47146d69b6651d8d3c3033a92b423db04ce5",
)
"""PanTaGruEl v0.1 (Zenodo record 2642175)."""

_BASE_MVA = 100.0


def germany_case_path() -> Path:
    """Path of the bundled German ANDES case."""
    return Path(__file__).resolve().parent.parent / "resources" / "cases" / "germany.xlsx"


def _read_csv(archive: zipfile.ZipFile, name: str) -> pd.DataFrame:
    member = next((m for m in archive.namelist() if m.endswith(f"pantagruel_csv/{name}.csv")), None)
    if member is None:
        raise DataError(f"{name}.csv not found in the PanTaGruEl archive")
    frame = pd.read_csv(io.BytesIO(archive.read(member)), skipinitialspace=True)
    frame.columns = [c.strip() for c in frame.columns]
    return frame


def _largest_component(buses: list[int], lines: pd.DataFrame) -> set[int]:
    parent = {b: b for b in buses}

    def root(b: int) -> int:
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        return b

    for a, b in zip(lines["ID1"], lines["ID2"], strict=True):
        parent[root(int(a))] = root(int(b))
    groups: dict[int, set[int]] = {}
    for b in buses:
        groups.setdefault(root(b), set()).add(b)
    return max(groups.values(), key=len)


def build_germany_case(
    out_path: str | Path,
    *,
    source: str | Path | None = None,
    frequency_hz: float = 50.0,
    power_factor: float = 1.0,
    loss_margin: float = 0.02,
    transient_reactance_pu: float = 0.3,
    generator_voltage_pu: float = 1.02,
) -> Path:
    """Write the German transmission grid as an ANDES ``.xlsx`` case and return its path.

    Args:
        out_path: case file to write.
        source: PanTaGruEl zip archive; downloaded (checksum verified) if None.
        frequency_hz: nominal frequency used to convert inertia and damping.
        power_factor: load power factor (lagging).
        loss_margin: generation in excess of the demand, as a share of the demand.
        transient_reactance_pu: generator transient reactance on the machine base.
        generator_voltage_pu: voltage set point of the generator buses.

    Raises:
        SimulationError: if the AC power flow of the result does not converge.
    """
    andes = require("andes", "andes")
    archive_path = Path(source) if source is not None else fetch(PANTAGRUEL)
    with zipfile.ZipFile(archive_path) as archive:
        buses = _read_csv(archive, "buses")
        generators = _read_csv(archive, "generators")
        lines = _read_csv(archive, "lines")

    de = buses[buses["Country"] == "DE"].set_index("ID")
    lines = lines[lines["ID1"].isin(de.index) & lines["ID2"].isin(de.index)]
    keep = _largest_component([int(i) for i in de.index], lines)
    de = de.loc[sorted(keep)]
    lines = lines[lines["ID1"].isin(keep)]
    generators = generators[generators["ID"].isin(keep)]
    capacity = generators["Max Output [MW]"].to_numpy(np.float64)
    demand = float(de["Demand [MW]"].sum())
    if capacity.sum() < demand * (1 + loss_margin):
        raise DataError("generation capacity does not cover the demand")
    dispatch = capacity * demand * (1 + loss_margin) / capacity.sum()

    w0 = 2 * math.pi * frequency_hz
    tan_phi = math.tan(math.acos(power_factor))
    bus_ids = [int(i) for i in de.index]
    voltage = dict(zip(bus_ids, de["Voltage [KV]"].astype(float), strict=True))
    names = dict(zip(bus_ids, de["Name"].astype(str), strict=True))
    demand_mw = dict(zip(bus_ids, de["Demand [MW]"].astype(float), strict=True))
    ss = andes.System()
    ss.config.freq = frequency_hz
    for bus_id in bus_ids:
        ss.add("Bus", {"idx": bus_id, "name": names[bus_id], "Vn": voltage[bus_id]})
    for bus_id in bus_ids:
        if demand_mw[bus_id] <= 0:
            continue
        p = demand_mw[bus_id] / _BASE_MVA
        ss.add(
            "PQ",
            {
                "idx": f"L{bus_id}",
                "bus": bus_id,
                "Vn": voltage[bus_id],
                "p0": p,
                "q0": p * tan_phi,
            },
        )
    for k, (a, b, r, x) in enumerate(
        zip(
            lines["ID1"].astype(int),
            lines["ID2"].astype(int),
            lines["Resistance [Ohm]"].astype(float),
            lines["Reactance [Ohm]"].astype(float),
            strict=True,
        )
    ):
        ss.add(
            "Line",
            {
                "idx": f"Line_{k}",
                "bus1": a,
                "bus2": b,
                "r": r,
                "x": x,
                "Vn1": voltage[a],
                "Vn2": voltage[b],
                "trans": int(voltage[a] != voltage[b]),
            },
        )
    slack = int(np.argmax(capacity))
    for k, (bus_id, sn, p, m, d) in enumerate(
        zip(
            generators["ID"].astype(int),
            capacity,
            dispatch,
            generators["Inertia [MWs^2]"].astype(float),
            generators["Primary Control [MWs]"].astype(float),
            strict=True,
        )
    ):
        vn = voltage[bus_id]
        static = {
            "idx": f"G{k}",
            "bus": bus_id,
            "Vn": vn,
            "Sn": float(sn),
            "p0": float(p) / _BASE_MVA,
            "v0": generator_voltage_pu,
            "pmax": float(sn) / _BASE_MVA,
            "pmin": 0.0,
            "qmax": 0.6 * float(sn) / _BASE_MVA,
            "qmin": -0.4 * float(sn) / _BASE_MVA,
        }
        ss.add("Slack" if k == slack else "PV", static)
        h = float(m) * w0 * _BASE_MVA / (2 * float(sn))
        ss.add(
            "GENCLS",
            {
                "idx": f"M{k}",
                "bus": int(bus_id),
                "gen": f"G{k}",
                "Sn": float(sn),
                "Vn": vn,
                "M": 2 * h,
                "D": float(d) * w0 * _BASE_MVA / float(sn),
                "xd1": transient_reactance_pu,
                "ra": 0.0,
            },
        )
    ss.setup()
    if not ss.PFlow.run():
        raise SimulationError("the AC power flow of the German case does not converge")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    andes.io.xlsx.write(ss, str(out), overwrite=True)
    return out
