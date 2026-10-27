"""Publication figures (optional ``[plot]`` extra; matplotlib is imported on first use).

* :func:`publication_style` - rc settings: vector PDF with embedded TrueType fonts,
  thin marks, recessive axes and grid.
* :data:`CATEGORICAL` - colour-blind-safe categorical palette in fixed order (validated
  for CVD separation; slot 3 is below 3:1 contrast, so plots label series directly).
* :func:`save_figure` - writes only where you tell it to.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from tranpy_core._optional import require

__all__ = [
    "CATEGORICAL",
    "INK",
    "MUTED",
    "plot_importance",
    "plot_metric_curves",
    "publication_style",
    "save_figure",
]

CATEGORICAL = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
"""Categorical hues in fixed order; series beyond three should be faceted or grouped."""
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"

_RC: dict[str, Any] = {
    "figure.figsize": (3.5, 2.6),
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": SECONDARY,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.prop_cycle": None,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "xtick.color": SECONDARY,
    "ytick.color": SECONDARY,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "text.color": INK,
}


@contextmanager
def publication_style() -> Iterator[None]:
    """Apply the shared figure style inside a ``with`` block."""
    mpl = require("matplotlib", "plot")
    rc = dict(_RC)
    rc["axes.prop_cycle"] = mpl.cycler(color=list(CATEGORICAL))
    with mpl.rc_context(rc):
        yield


def save_figure(fig: Any, path: str | Path) -> Path:
    """Save ``fig`` (format from the suffix; use ``.pdf`` for print) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path


def _axes(ax: Any) -> tuple[Any, Any]:
    if ax is not None:
        return ax.figure, ax
    plt = require("matplotlib.pyplot", "plot")
    fig, new_ax = plt.subplots()
    return fig, new_ax


def plot_importance(
    names: Sequence[str],
    importance: npt.ArrayLike,
    *,
    top: int = 10,
    ax: Any = None,
    xlabel: str = "mean |attribution|",
    title: str | None = None,
) -> Any:
    """Horizontal bars of the ``top`` most important features, largest on top, values labelled."""
    values = np.asarray(importance, dtype=np.float64)
    order = np.argsort(-values, kind="stable")[:top][::-1]
    fig, ax = _axes(ax)
    y = np.arange(order.size)
    ax.barh(y, values[order], height=0.7, color=CATEGORICAL[0], edgecolor="white", linewidth=0.8)
    ax.set_yticks(y, [names[i] for i in order])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    span = float(np.max(values[order])) if order.size else 1.0
    for yi, v in zip(y, values[order], strict=True):
        ax.text(v + 0.01 * span, yi, f"{v:.3g}", va="center", fontsize=6, color=SECONDARY)
    ax.set_xlim(0, span * 1.15)
    if title:
        ax.set_title(title)
    return fig


def plot_metric_curves(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str,
    band: str | None = None,
    ax: Any = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    logx: bool = False,
) -> Any:
    """One line per ``group`` (fixed colour order) with a legend, optional ±band.

    With up to four groups every line is also labelled at its last point; labels that
    would overlap are moved apart vertically. With ``logx`` the measured x values are used
    as tick labels.
    """
    fig, ax = _axes(ax)
    groups = list(dict.fromkeys(frame[group]))
    ends: list[tuple[float, float, str]] = []
    for i, name in enumerate(groups):
        part = frame[frame[group] == name].sort_values(x)
        color = CATEGORICAL[i % len(CATEGORICAL)]
        ax.plot(part[x], part[y], color=color, marker="o", markersize=3, label=str(name))
        if band is not None:
            ax.fill_between(
                part[x], part[y] - part[band], part[y] + part[band], color=color, alpha=0.15, lw=0
            )
        ends.append((float(part[x].iloc[-1]), float(part[y].iloc[-1]), str(name)))
    if logx:
        ax.set_xscale("log")
        ticks = sorted(set(frame[x]))
        ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
        ax.minorticks_off()
    if len(groups) <= 4:
        _label_line_ends(ax, ends)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if len(groups) > 1:
        ax.legend(loc="lower right")
    if title:
        ax.set_title(title)
    return fig


def _label_line_ends(
    ax: Any,
    ends: list[tuple[float, float, str]],
    min_gap_pt: float = 9.0,
    max_shift_pt: float = 6.0,
) -> None:
    """Annotate line ends; skip labels that would have to move far from their line.

    A label is drawn only if it fits within ``max_shift_pt`` of its line end without
    colliding with another label; otherwise the legend identifies the line.
    """
    fig = ax.figure
    points_per_pixel = 72.0 / fig.dpi
    placed: list[float] = []
    for x_end, y_end, name in sorted(ends, key=lambda e: e[1]):
        y_pt = ax.transData.transform((x_end, y_end))[1] * points_per_pixel
        target = max([y_pt, *(p + min_gap_pt for p in placed)])
        if target - y_pt > max_shift_pt:
            continue
        placed.append(target)
        ax.annotate(
            name,
            (x_end, y_end),
            xytext=(4, target - y_pt),
            textcoords="offset points",
            va="center",
            fontsize=7,
            color=SECONDARY,
        )
