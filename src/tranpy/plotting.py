"""Power-system figures on top of :mod:`tranpy_core.plotting` (optional ``[plot]`` extra)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tranpy.datasets import FAULT_CLEAR, FAULT_START
from tranpy.vocabulary import Stability
from tranpy_core._optional import require
from tranpy_core.data import SignalDataset
from tranpy_core.errors import DataError
from tranpy_core.plotting import CATEGORICAL, MUTED, SECONDARY, plot_metric_curves
from tranpy_core.random import Seed, make_rng

__all__ = ["STABILITY_COLORS", "plot_early_prediction", "plot_trajectories"]

STABILITY_COLORS = {Stability.STABLE: CATEGORICAL[0], Stability.UNSTABLE: CATEGORICAL[1]}


def plot_trajectories(
    events: SignalDataset,
    channel: str,
    *,
    per_class: int = 5,
    seed: Seed = 0,
    align: str | None = FAULT_START,
    ax: Any = None,
    ylabel: str | None = None,
    title: str | None = None,
) -> Any:
    """Overlay ``per_class`` randomly chosen stable and unstable events of one channel.

    With ``align`` set to an annotation (default: fault start) time is shown relative to
    it, and the median fault-clearing instant is marked.
    """
    plt = require("matplotlib.pyplot", "plot")
    col = events.channel_index(channel)
    if col.size != 1:
        raise DataError(
            f"channel pattern {channel!r} must match exactly one channel, matched {col.size}"
        )
    rng = make_rng(seed)
    if ax is None:
        _, ax = plt.subplots()
    clear_offsets = []
    for cls in Stability:
        idx = np.flatnonzero(events.labels == cls)
        chosen = rng.choice(idx, size=min(per_class, idx.size), replace=False) if idx.size else idx
        for n, i in enumerate(chosen):
            sample = events[int(i)]
            t0 = sample.annotations.get(align, 0.0) if align else 0.0
            ax.plot(
                sample.time - t0,
                sample.values[:, col[0]],
                color=STABILITY_COLORS[cls],
                lw=0.8,
                alpha=0.8,
                label=cls.name.lower() if n == 0 else None,
            )
            if align and FAULT_CLEAR in sample.annotations:
                clear_offsets.append(sample.annotations[FAULT_CLEAR] - t0)
    if align:
        ax.axvline(0.0, color=MUTED, lw=0.6, ls="--")
        ax.annotate(
            "fault",
            (0.0, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(2, -8),
            textcoords="offset points",
            fontsize=6,
            color=SECONDARY,
        )
        if clear_offsets:
            tc = float(np.nanmedian(clear_offsets))
            ax.axvline(tc, color=MUTED, lw=0.6, ls=":")
            ax.annotate(
                "clearing (median)",
                (tc, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(2, -16),
                textcoords="offset points",
                fontsize=6,
                color=SECONDARY,
            )
    ax.set_xlabel(
        f"time{' after ' + align.removesuffix('_s').replace('_', ' ') if align else ''} (s)"
    )
    ax.set_ylabel(ylabel or channel)
    ax.legend(loc="best")
    if title:
        ax.set_title(title)
    return ax.figure


def plot_early_prediction(
    results: pd.DataFrame, *, metric: str = "accuracy", ax: Any = None, title: str | None = None
) -> Any:
    """Cross-validated ``metric`` (mean ± std) versus observation window, one line per model."""
    return plot_metric_curves(
        results,
        x="window_s",
        y=f"{metric}_mean",
        group="model",
        band=f"{metric}_std",
        ax=ax,
        xlabel="observation window (s)",
        ylabel=metric.replace("_", " "),
        title=title,
        logx=True,
    )
