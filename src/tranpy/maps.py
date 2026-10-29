"""Results on a geographic map of the grid (optional ``[andes]`` and ``[plot]`` extras).

* :func:`grid_geometry` - bus coordinates (longitude, latitude) of an ANDES case and the
  end points of its fault-candidate lines and generators, in the engine's order.
* :func:`per_line` - aggregate an event-level quantity by faulted line (e.g. share of
  unstable events, or share of misclassified events).
* :func:`plot_grid_map` - the network on a map, faulted lines and generators coloured by
  a value; country borders from Natural Earth (public domain, downloaded once).

Only cases with coordinates can be drawn (the bundled German case has them).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from tranpy_core._optional import require
from tranpy_core.data import SignalDataset
from tranpy_core.data.download import RemoteFile, fetch
from tranpy_core.errors import DataError
from tranpy_core.plotting import CATEGORICAL, INK, MUTED, SECONDARY

if TYPE_CHECKING:
    from tranpy.grid_data import CoreGridModel
    from tranpy.simulation.andes import AndesEngine

__all__ = [
    "NATURAL_EARTH_COUNTRIES",
    "GridGeometry",
    "grid_geometry",
    "per_line",
    "plot_grid_map",
    "plot_operator_map",
]

NATURAL_EARTH_COUNTRIES = RemoteFile(
    filename="ne_50m_admin_0_countries.geojson",
    url=(
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/"
        "geojson/ne_50m_admin_0_countries.geojson"
    ),
    sha256="3e458fc036ad0a66411f2c1e6cac49c5d7bfb81cb1123bc513b22511a2b7fdeb",
)
"""Natural Earth 1:50m country borders, v5.1.2 (public domain)."""

_RAMP = ("#fbe3d6", "#f2a27e", "#e0602f", "#a8380f", "#5c1a04")
"""Sequential ramp (one hue, light to dark) for line and generator values."""


@dataclass(frozen=True)
class GridGeometry:
    """Coordinates of a case, in degrees (longitude, latitude).

    Attributes:
        buses: ``(n_buses, 2)`` coordinates in the case's bus order.
        bus_ids: bus identifiers in the same order.
        branches: ``(n_branches, 2, 2)`` end points of every in-service branch.
        branch_kv: nominal voltage of each branch (from-bus side).
        lines: ``(n_lines, 2, 2)`` end points of the engine's fault-candidate lines.
        generators: ``(n_generators, 2)`` coordinates in the engine's generator order
            (``G 1``, ``G 2`` ... in the recorded channels).
        generator_mw: rated power of each generator.
    """

    buses: npt.NDArray[np.float64]
    bus_ids: tuple[Any, ...]
    branches: npt.NDArray[np.float64]
    branch_kv: npt.NDArray[np.float64]
    lines: npt.NDArray[np.float64]
    generators: npt.NDArray[np.float64]
    generator_mw: npt.NDArray[np.float64]


def grid_geometry(engine: AndesEngine) -> GridGeometry:
    """Coordinates from the engine's case (``Bus.xcoord``: longitude, ``ycoord``: latitude)."""
    from tranpy.simulation.andes import _MACHINES

    ss = engine._load(setup=True)
    lon = np.asarray(ss.Bus.xcoord.v, dtype=np.float64)
    lat = np.asarray(ss.Bus.ycoord.v, dtype=np.float64)
    if not np.any(lon) and not np.any(lat):
        raise DataError("the case has no bus coordinates")
    where = {b: np.array([x, y]) for b, x, y in zip(ss.Bus.idx.v, lon, lat, strict=True)}
    kv = dict(zip(ss.Bus.idx.v, np.asarray(ss.Bus.Vn.v, dtype=np.float64), strict=True))
    ends = {
        idx: (b1, b2)
        for idx, b1, b2, u in zip(
            ss.Line.idx.v, ss.Line.bus1.v, ss.Line.bus2.v, ss.Line.u.v, strict=True
        )
        if u
    }
    branches = np.array([[where[a], where[b]] for a, b in ends.values()])
    machines = [getattr(ss, name) for name in _MACHINES if getattr(ss, name).n]
    gen_buses = [b for m in machines for b in m.bus.v]
    return GridGeometry(
        buses=np.column_stack([lon, lat]),
        bus_ids=tuple(ss.Bus.idx.v),
        branches=branches,
        branch_kv=np.array([kv[a] for a, _ in ends.values()]),
        lines=np.array([[where[a], where[b]] for a, b in (ends[i] for i in engine._lines)]),
        generators=np.array([where[b] for b in gen_buses]),
        generator_mw=np.concatenate([np.asarray(m.Sn.v, dtype=np.float64) for m in machines]),
    )


def per_line(
    events: SignalDataset, values: npt.ArrayLike, n_lines: int, *, min_events: int = 1
) -> npt.NDArray[np.float64]:
    """Mean of ``values`` (one per event) for each faulted line; NaN below ``min_events``."""
    lines = np.asarray(events.annotations["fault_line"], dtype=np.int64)
    frame = pd.DataFrame({"line": lines, "value": np.asarray(values, dtype=np.float64)})
    stats = frame.groupby("line")["value"].agg(["mean", "size"])
    out = np.full(n_lines, np.nan)
    keep = stats[stats["size"] >= min_events]
    out[keep.index.to_numpy()] = keep["mean"].to_numpy()
    return out


def _projected(lonlat: npt.NDArray[np.float64], lat0: float) -> npt.NDArray[np.float64]:
    out = np.array(lonlat, dtype=np.float64)
    out[..., 0] *= np.cos(np.radians(lat0))
    return out


def _draw_borders(ax: Any, lat0: float, highlight: Sequence[str]) -> None:
    with open(fetch(NATURAL_EARTH_COUNTRIES), encoding="utf-8") as fh:
        features = json.load(fh)["features"]
    for feature in features:
        geometry = feature["geometry"]
        inside = feature["properties"]["ADM0_A3"] in highlight
        polygons = (
            geometry["coordinates"]
            if geometry["type"] == "MultiPolygon"
            else [geometry["coordinates"]]
        )
        for polygon in polygons:
            ring = _projected(np.asarray(polygon[0], dtype=np.float64), lat0)
            ax.fill(
                ring[:, 0],
                ring[:, 1],
                color="#f4f3f0" if inside else "#fbfbfa",
                ec=MUTED if inside else "#d8d7d3",
                lw=0.8 if inside else 0.4,
                zorder=0,
            )


def plot_grid_map(
    geometry: GridGeometry,
    *,
    line_values: npt.ArrayLike | None = None,
    generator_values: npt.ArrayLike | None = None,
    label: str = "",
    vmin: float = 0.0,
    vmax: float | None = None,
    borders: bool = True,
    highlight: Sequence[str] = ("DEU",),
    title: str | None = None,
    ax: Any = None,
) -> Any:
    """Draw the network; colour faulted lines and/or generators by a value.

    Args:
        geometry: from :func:`grid_geometry`.
        line_values: one value per fault-candidate line (NaN: not shown in colour).
        generator_values: one value per generator (marker colour; size by rating).
        label: colour-bar label.
        vmin, vmax: colour range (``vmax`` defaults to the largest value).
        borders: draw country borders (downloads Natural Earth once).
        highlight: ISO 3166 alpha-3 codes of the countries filled as the study area.
        title: axes title.
        ax: axes to draw on; a new figure if None.
    """
    plt = require("matplotlib.pyplot", "plot")
    colors = require("matplotlib.colors", "plot")
    collections = require("matplotlib.collections", "plot")
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 7.2))
    lat0 = float(np.nanmean(geometry.buses[:, 1]))
    if borders:
        _draw_borders(ax, lat0, highlight)
    top = max(geometry.branch_kv) if geometry.branch_kv.size else 0.0
    for is_top, color, width in ((False, "#c4c3be", 0.4), (True, "#8a8985", 0.6)):
        segments = geometry.branches[(geometry.branch_kv >= top) == is_top]
        ax.add_collection(
            collections.LineCollection(
                _projected(segments, lat0), colors=color, linewidths=width, zorder=1
            )
        )
    cmap = colors.LinearSegmentedColormap.from_list("tranpy_seq", _RAMP)
    given = [
        np.asarray(v, dtype=np.float64) for v in (line_values, generator_values) if v is not None
    ]
    top_value = max((float(np.nanmax(v)) for v in given if np.isfinite(v).any()), default=1.0)
    norm = colors.Normalize(vmin=vmin, vmax=vmax if vmax is not None else top_value)
    if line_values is not None:
        values = np.asarray(line_values, dtype=np.float64)
        if values.shape != (len(geometry.lines),):
            raise DataError(f"need one value per line ({len(geometry.lines)}), got {values.shape}")
        shown = np.isfinite(values)
        order = np.argsort(values[shown])  # largest values drawn on top
        ax.add_collection(
            collections.LineCollection(
                _projected(geometry.lines[shown][order], lat0),
                colors=cmap(norm(values[shown][order])),
                linewidths=1.8,
                capstyle="round",
                zorder=2,
            )
        )
    gens = _projected(geometry.generators, lat0)
    sizes = geometry.generator_mw / 12.0
    if generator_values is not None:
        values = np.asarray(generator_values, dtype=np.float64)
        if values.shape != (len(geometry.generators),):
            raise DataError(
                f"need one value per generator ({len(geometry.generators)}), got {values.shape}"
            )
        ax.scatter(gens[:, 0], gens[:, 1], s=sizes, c=cmap(norm(values)), ec=INK, lw=0.5, zorder=3)
    else:
        ax.scatter(
            gens[:, 0], gens[:, 1], s=sizes, color=CATEGORICAL[0], ec="white", lw=0.6, zorder=3
        )
    if given:
        bar = ax.figure.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.45, pad=0.01
        )
        bar.set_label(label, color=SECONDARY)
        bar.outline.set_visible(False)
    pad = 0.3
    xy = _projected(geometry.buses, lat0)
    low, high = np.min(xy, axis=0), np.max(xy, axis=0)
    ax.set_xlim(low[0] - pad, high[0] + pad)
    ax.set_ylim(low[1] - pad, high[1] + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax.figure


def plot_operator_map(
    model: CoreGridModel,
    *,
    names: Mapping[str, str] | None = None,
    borders: bool = True,
    highlight: Sequence[str] = ("DEU",),
    title: str | None = None,
    ax: Any = None,
) -> Any:
    """Line routes coloured by operator (380 kV wider than 220 kV; tie-lines dashed).

    The source draws most lines as straight segments between substations, so the map
    shows the topology rather than the exact geographic routes.

    Args:
        model: from :func:`tranpy.grid_data.load_core_grid_model`.
        names: operator code -> legend name (default: the German TSO short names).
        borders: draw country borders (downloads Natural Earth once).
        highlight: ISO 3166 alpha-3 codes of the countries filled as the study area.
        title: axes title.
        ax: axes to draw on; a new figure if None.
    """
    from tranpy.grid_data import GERMAN_TSOS

    plt = require("matplotlib.pyplot", "plot")
    collections = require("matplotlib.collections", "plot")
    lines_mod = require("matplotlib.lines", "plot")
    names = dict(GERMAN_TSOS if names is None else names)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 7.2))
    frame = model.lines[model.lines["eic"].isin(model.routes)]
    points = np.vstack([model.routes[e] for e in frame["eic"]])
    lat0 = float(np.mean(points[:, 1]))
    if borders:
        _draw_borders(ax, lat0, highlight)
    handles = []
    for i, tso in enumerate(sorted(frame["tso"].unique(), key=lambda t: names.get(t, t))):
        color = CATEGORICAL[i % len(CATEGORICAL)]
        part = frame[frame["tso"] == tso]
        for tie, style in ((False, "solid"), (True, (0, (2, 1.5)))):
            sub = part[part["tie_line"] == tie]
            if sub.empty:
                continue
            widths = np.where(sub["kv"] >= 300, 1.1, 0.6)
            ax.add_collection(
                collections.LineCollection(
                    [_projected(model.routes[e], lat0) for e in sub["eic"]],
                    colors=color,
                    linewidths=widths,
                    linestyles=style,
                    capstyle="round",
                    zorder=2,
                )
            )
        handles.append(
            lines_mod.Line2D(
                [], [], color=color, lw=1.5, label=f"{names.get(tso, tso)} ({len(part)})"
            )
        )
    stations = model.substations()
    xy = _projected(stations[["lon", "lat"]].to_numpy(np.float64), lat0)
    ax.scatter(xy[:, 0], xy[:, 1], s=3, color=INK, lw=0, zorder=3)
    handles += [
        lines_mod.Line2D([], [], color=SECONDARY, lw=1.1, label="380 kV"),
        lines_mod.Line2D([], [], color=SECONDARY, lw=0.6, label="220 kV"),
        lines_mod.Line2D([], [], color=SECONDARY, lw=0.8, ls=(0, (2, 1.5)), label="tie-line"),
        lines_mod.Line2D([], [], ls="", marker="o", ms=2, color=INK, label="substation"),
    ]
    ax.legend(handles=handles, loc="upper left", borderaxespad=0)
    domestic = frame[~frame["tie_line"]]["eic"]
    inside = _projected(np.vstack([model.routes[e] for e in domestic]), lat0)
    low, high = np.min(inside, axis=0), np.max(inside, axis=0)
    ax.set_xlim(low[0] - 0.2, high[0] + 0.2)
    ax.set_ylim(low[1] - 0.2, high[1] + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax.figure
