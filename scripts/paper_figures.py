"""Generate the figures of ``papers/Access-2026-DE-TSA`` from the result files.

Runs on cairo-lab, where the experiment results live::

    .venv/bin/python scripts/paper_figures.py --exp ~/data/exp --grid ~/data/grid/grid_model.gpkg \
        --paper-src ~/data/paper_src --out ~/paper_figures

One function per figure (``fig_<label>``); every function reads only the result files named in
``papers/specs/00-spec.md`` section 5 and writes ``fig_<label>.pdf`` (vector, page exactly the IEEE
column width, 8 pt). ``--only instants,grid_map`` restricts the run. A missing source file is
reported and the figure is skipped, never faked.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects

# --------------------------------------------------------------------------------------------
# Style: IEEE column widths, 8 pt fonts, colour-blind-safe categorical palette (validated order:
# blue, orange, aqua, yellow, magenta, green, violet, red), one sequential blue ramp.
# --------------------------------------------------------------------------------------------
SINGLE = 3.5
DOUBLE = 7.16
C = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#6250d6",
    "red": "#e34948",
}
CAT = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"], C["green"], C["violet"],
       C["red"]]
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
            "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
# ENTSO-E grid-map convention (fixed by the spec): 380 kV red, 220 kV green.
KV_COLOUR = {380: "#d7191c", 220: "#1a9641"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "grid.color": GRID, "grid.linewidth": 0.5, "axes.grid": False,
    "legend.frameon": False, "legend.handlelength": 1.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.constrained_layout.use": True, "figure.constrained_layout.h_pad": 0.02,
    "figure.constrained_layout.w_pad": 0.02, "lines.linewidth": 1.0,
})


def _save(fig: Any, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"fig_{name}.pdf"
    fig.savefig(p)
    plt.close(fig)
    print(f"wrote {p}")
    return p


def _read_csv(p: Path) -> list[dict[str, str]]:
    with p.open() as f:
        return list(csv.DictReader(f))


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _ci_cell(s: str) -> tuple[float, float, float]:
    """``0.786 [0.767, 0.803]`` -> (0.786, 0.767, 0.803)."""
    m = re.match(r"\s*([0-9.]+)\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]", s)
    if not m:
        raise ValueError(s)
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def _md_tables(text: str) -> dict[str, list[dict[str, str]]]:
    """Parse every ``## heading`` + pipe table of a Markdown file into rows keyed by heading."""
    out: dict[str, list[dict[str, str]]] = {}
    heading = ""
    header: list[str] | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            heading = line[3:].strip()
            header = None
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if header is None:
                header = cells
                continue
            if all(set(c) <= set("-: ") for c in cells):
                continue
            out.setdefault(heading, []).append(dict(zip(header, cells, strict=False)))
        else:
            header = None
    return out


# --------------------------------------------------------------------------------------------
# fig:instants - programmatic timeline
# --------------------------------------------------------------------------------------------
def fig_instants(exp: Path, grid: Path, src: Path, out: Path) -> None:
    fig = plt.figure(figsize=(SINGLE, 2.2), layout="none")
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(-1.4, 2.75)
    ax.set_ylim(-1.75, 1.3)
    ax.axis("off")
    y0 = 0.0
    xf, xc = 0.0, 0.6  # fault and clearing instants on the drawing axis
    ax.annotate("", xy=(2.5, y0), xytext=(-1.3, y0),
                arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": INK})
    ax.text(2.48, y0 - 0.14, "time", ha="right", va="top", fontsize=7, color=INK2)
    ax.fill_between([-1.3, xf], y0 - 0.055, y0 + 0.055, color=C["blue"], alpha=0.18, lw=0)
    ax.text(-0.65, y0 + 0.1, "operating state", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.fill_between([xf, xc], y0 - 0.055, y0 + 0.055, color=C["red"], alpha=0.5, lw=0)
    ax.text((xf + xc) / 2, y0 + 0.1, "fault on", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.fill_between([xc, 2.3], y0 - 0.055, y0 + 0.055, color=C["orange"], alpha=0.22, lw=0)
    ax.text(xc + 0.25, y0 + 0.1, "post-clearing swing", ha="left", va="bottom", fontsize=7,
            color=INK2)
    for x in (xf, xc):
        ax.plot([x, x], [y0 - 0.1, y0 + 0.1], color=INK, lw=0.8)
    ax.text(xf - 0.03, y0 - 0.15, "fault\n$t=0$", ha="right", va="top", fontsize=7, color=INK)
    ax.text(xc + 0.03, y0 - 0.15, "clearing\n$t_c\\sim U(0.05, 0.6)$ s", ha="left", va="top",
            fontsize=7, color=INK)
    # decision instants (arrows down onto the axis; labels kept apart left/right)
    ax.annotate("", xy=(-0.65, y0 + 0.07), xytext=(-0.65, y0 + 0.62),
                arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": C["blue"]})
    ax.text(-1.3, y0 + 0.66, "decision 1:\npre-fault screening ($t<0$)", ha="left",
            va="bottom", fontsize=7, color=C["blue"], fontweight="bold")
    ax.annotate("", xy=(xc + 0.1, y0 + 0.07), xytext=(xc + 0.1, y0 + 0.62),
                arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": C["orange"]})
    ax.text(2.72, y0 + 0.66, "decision 2: post-clearing control\n($t_c$ + 50…150 ms)",
            ha="right", va="bottom", fontsize=7, color=C["orange"], fontweight="bold")
    # cost table (below)
    ya = y0 - 0.82
    ax.text(-1.3, ya, "cost per event", ha="left", va="center", fontsize=7, color=INK,
            fontweight="bold")
    ax.text(1.05, ya, "screening budget", ha="left", va="center", fontsize=7, color=C["blue"],
            fontweight="bold")
    ax.plot([-1.3, 2.5], [ya - 0.13, ya - 0.13], color=GRID, lw=0.6)
    rows = [("time-domain simulation", "13.2 s"), ("energy method (SIME)", "≈1.4 s"),
            ("boosting (CPU)", "0.12 ms")]
    for i, (name, val) in enumerate(rows):
        yy = ya - 0.27 * (i + 1)
        ax.text(-1.3, yy, name, ha="left", va="center", fontsize=7, color=INK2)
        ax.text(0.9, yy, val, ha="right", va="center", fontsize=7, color=INK)
    ax.text(1.05, ya - 0.27, "≈2,000 contingencies\nper minute per state\n(≈30 ms each)",
            ha="left", va="top", fontsize=7, color=C["blue"])
    _save(fig, out, "instants")


# --------------------------------------------------------------------------------------------
# fig:grid_map - German 220/380 kV model from the GeoPackage
# --------------------------------------------------------------------------------------------
def fig_grid_map(exp: Path, grid: Path, src: Path, out: Path) -> None:
    import geopandas as gpd

    lines = gpd.read_file(grid, layer="lines")
    subs = gpd.read_file(grid, layer="substations")
    crs = "EPSG:3035"  # LAEA Europe, equal-area, straight German borders
    lines = lines.to_crs(crs)
    subs = subs.to_crs(crs)
    # Border: Natural Earth 1:50m (tranpy.maps cache) or the gpkg bidding_zones fallback.
    border = None
    ne = Path.home() / ".cache" / "tranpy" / "ne_50m_admin_0_countries.geojson"
    if not ne.exists():
        try:
            from tranpy.maps import NATURAL_EARTH_COUNTRIES

            ne = Path(NATURAL_EARTH_COUNTRIES.path())  # type: ignore[attr-defined]
        except Exception:
            ne = Path("/nonexistent")
    if ne.exists():
        world = gpd.read_file(ne)
        iso = next(c for c in ("ADM0_A3", "ISO_A3", "adm0_a3") if c in world.columns)
        keep = ["DEU", "AUT", "CHE", "FRA", "LUX", "BEL", "NLD", "DNK", "POL", "CZE"]
        border = world[world[iso].isin(keep)].to_crs(crs)
        de = world[world[iso] == "DEU"].to_crs(crs)
        border_src = "Natural Earth 1:50m"
    else:
        zones = gpd.read_file(grid, layer="bidding_zones").to_crs(crs)
        border = zones
        de = zones[zones["zone"] == "DE-LU"]
        border_src = "gpkg bidding_zones"
    print(f"grid_map border source: {border_src}")

    fig, ax = plt.subplots(figsize=(SINGLE, 4.35))
    border.plot(ax=ax, facecolor="#f6f5f2", edgecolor="#b9b7b0", linewidth=0.4, zorder=1)
    de.plot(ax=ax, facecolor="white", edgecolor="#6f6d66", linewidth=0.6, zorder=2)
    for kv, lw, z in ((220, 0.55, 3), (380, 0.8, 4)):
        sel = lines[lines["voltage_kv"].round() == kv]
        sel.plot(ax=ax, color=KV_COLOUR[kv], linewidth=lw, zorder=z)
    st = subs[subs["kind"] == "substation"]
    for kv, z in ((220, 5), (380, 6)):
        s = st[st["max_kv"].round() == kv]
        s.plot(ax=ax, marker="s", markersize=3.2, color=KV_COLOUR[kv], edgecolor="white",
               linewidth=0.25, zorder=z)
    # extent: Germany with a thin margin
    minx, miny, maxx, maxy = de.total_bounds
    pad = 0.04 * (maxy - miny)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    n380 = int((lines["voltage_kv"].round() == 380).sum())
    n220 = int((lines["voltage_kv"].round() == 220).sum())
    print(f"grid_map: {n380} 380 kV and {n220} 220 kV line records, {len(st)} substations")
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=KV_COLOUR[380], lw=1.2, marker="s", ms=3.5, mec="white", mew=0.3,
               label="380 kV"),
        Line2D([], [], color=KV_COLOUR[220], lw=0.9, marker="s", ms=3.5, mec="white", mew=0.3,
               label="220 kV"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2,
              handletextpad=0.6)
    ax.text(0.0, 0.0, f"border: {border_src}; EPSG:3035", transform=ax.transAxes, fontsize=5.5,
            color=INK2, ha="left", va="bottom")
    _save(fig, out, "grid_map")


# --------------------------------------------------------------------------------------------
# fig:event_jan2021 - TransnetBW 1-s measurement vs v4.1 simulation
# --------------------------------------------------------------------------------------------
def fig_event_jan2021(exp: Path, grid: Path, src: Path, out: Path) -> None:
    import pandas as pd

    meas_p = src / "real_events" / "derived" / "transnetbw_1s_2021-01-08_140430-140700.csv"
    sim_dir = src / "real_events" / "sim"
    M = pd.read_csv(meas_p, parse_dates=["timestamp_local_CET"])
    # Alignment as in work/real_events/sim/compare_jan.py: event instant 14:04:56 CET (last
    # pre-event sample before the decline); deviations from the pre-event mean 14:04:30-14:04:56.
    t0 = pd.Timestamp("2021-01-08 14:04:56")
    M["t"] = (M["timestamp_local_CET"] - t0).dt.total_seconds()
    f_pre = float(M.loc[M["t"] <= 0, "frequency_Hz"].mean())
    M["df"] = (M["frequency_Hz"] - f_pre) * 1e3
    nadir_meas = float(M["df"].min())
    qss_meas = float(M.loc[(M["t"] >= 20) & (M["t"] <= 52), "df"].mean())

    fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
    ax.grid(True, axis="y")
    ax.step(M["t"], M["df"], where="post", color=INK, lw=1.0, zorder=5,
            label="TransnetBW 1-s measurement")

    runs = {}
    for tag in ("jan_P5730", "jan_P5800", "jan_P6000"):
        p = sim_dir / f"{tag}.csv"
        if p.exists():
            d = pd.read_csv(p)
            t_ref = json.load((sim_dir / f"{tag}.json").open())["metrics"].get("t_ref_s", 1.0)
            runs[tag] = (d["t_s"].to_numpy() - t_ref, (d["f_coi_hz"].to_numpy() - 50.0) * 1e3)
    sim_nadir = sim_qss = None
    if runs:
        if "jan_P5730" in runs and "jan_P6000" in runs:
            t = runs["jan_P5800"][0] if "jan_P5800" in runs else runs["jan_P5730"][0]
            lo = np.interp(t, *runs["jan_P6000"])
            hi = np.interp(t, *runs["jan_P5730"])
            ax.fill_between(t, lo, hi, color=C["orange"], alpha=0.25, lw=0, zorder=3,
                            label="v4.1 step, 5,730–6,000 MW")
        if "jan_P5800" in runs:
            ax.plot(*runs["jan_P5800"], color=C["orange"], lw=1.0, zorder=4,
                    label="v4.1 step, 5,800 MW")
            t, f = runs["jan_P5800"]
            sim_nadir = float(f.min())
            sim_qss = float(f[(t >= 20) & (t <= 52)].mean())
        sim_src = "sim/results/events_v41/jan_P5800.csv"
    else:
        # No trace: mark the simulated nadir and QSS bands from README section 10.2.
        ax.axhspan(-308, -289, color=C["orange"], alpha=0.3, lw=0,
                   label="v4.1 nadir −308…−289 mHz")
        ax.axhspan(-229, -196, color=C["orange"], alpha=0.15, lw=0,
                   label="v4.1 QSS −229…−196 mHz")
        sim_src = "README §10.2 bands (no trace file)"
    # published ENTSO-E values (deviation from 50.027 Hz)
    ax.axhline(-287, color=INK2, ls=":", lw=0.7, zorder=2)
    ax.text(64, -287 - 6, "ENTSO-E nadir 49.74 Hz", ha="right", va="top", fontsize=6.5,
            color=INK2)
    ax.axhline(-187, color=INK2, ls=":", lw=0.7, zorder=2)
    ax.text(64, -187 + 10, "ENTSO-E QSS 49.84 Hz", ha="right", va="bottom", fontsize=6.5,
            color=INK2)
    ax.set_xlim(-5, 65)
    ax.set_ylim(-340, 25)
    ax.set_xlabel("time after separation (s)")
    ax.set_ylabel("Δf from pre-event 50.027 Hz (mHz)")
    ax.legend(loc="upper right", ncol=1, handlelength=1.4)
    print(f"event_jan2021: measured nadir {nadir_meas:.0f} mHz, QSS {qss_meas:.0f} mHz; "
          f"sim nadir {sim_nadir}, QSS {sim_qss}; sim source {sim_src}")
    _save(fig, out, "event_jan2021")


# --------------------------------------------------------------------------------------------
# fig:instant_curve - PR-AUC vs assessment instant
# --------------------------------------------------------------------------------------------
def fig_instant_curve(exp: Path, grid: Path, src: Path, out: Path) -> None:
    rows = _read_csv(exp / "e3_deep" / "tables" / "curves.csv")
    knees = json.load((exp / "e3_deep" / "tables" / "knees.json").open())
    hour = [r for r in rows if r["protocol"] == "hour"]

    def series(model: str, inp: str, ref: str) -> tuple[np.ndarray, ...]:
        sel = sorted((r for r in hour if r["model"] == model and r["input"] == inp
                      and r["reference"] == ref), key=lambda r: _f(r["window_s"]))
        x = np.array([_f(r["window_s"]) for r in sel])
        return (x, np.array([_f(r["average_precision_mean"]) for r in sel]),
                np.array([_f(r["average_precision_lo"]) for r in sel]),
                np.array([_f(r["average_precision_hi"]) for r in sel]))

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.45), sharey=True,
                             gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.08})
    # (a) clearing-based
    ax = axes[0]
    ax.grid(True, axis="y")
    panels_a = [
        ("lightgbm_tuned", "window_line", "LightGBM, window + line", C["blue"], "o"),
        ("tcn_L", "seq_sys", "TCN-L", C["orange"], "s"),
        ("lstm_L", "seq_sys", "LSTM-L", C["aqua"], "^"),
        ("transformer_L", "seq_sys", "Transformer-L", C["yellow"], "D"),
    ]
    for model, inp, lab, col, mk in panels_a:
        x, m, lo, hi = series(model, inp, "clear")
        if len(x) == 0:
            print(f"instant_curve: no clear-based rows for {model}/{inp}")
            continue
        ax.fill_between(x * 1e3, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(x * 1e3, m, color=col, marker=mk, ms=3, mec="white", mew=0.4, label=lab)
        k = knees.get(f"hour|{inp}|{model}|clear")
        if k is not None:
            ax.plot([k * 1e3], [np.interp(k, x, m)], marker=mk, ms=6, mfc="none", mec=col, mew=0.9,
                    ls="none", zorder=6)
    ax.set_xlim(-12, 515)
    ax.set_xlabel("assessment instant after clearing (ms)")
    ax.set_ylabel("PR-AUC (hour protocol, 95 % CI)")
    ax.set_title("(a) clearing-based", loc="left", fontsize=8)
    ax.legend(loc="lower right", ncol=2, columnspacing=1.0)
    ax.text(0.98, 0.38, "hollow marker: knee (first instant\nwithin the final CI)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=INK2)
    # (b) inception-based
    ax = axes[1]
    ax.grid(True, axis="y")
    for model, inp, lab, col, mk in (
        ("lightgbm_tuned", "window_line", "LightGBM, window + line", C["blue"], "o"),
        ("lightgbm_tuned", "window", "LightGBM, window only", C["violet"], "v"),
    ):
        x, m, lo, hi = series(model, inp, "inception")
        if len(x) == 0:
            print(f"instant_curve: no inception rows for {model}/{inp}")
            continue
        ax.fill_between(x * 1e3, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(x * 1e3, m, color=col, marker=mk, ms=3, mec="white", mew=0.4, label=lab)
        k = knees.get(f"hour|{inp}|{model}|inception")
        if k is not None:
            ax.plot([k * 1e3], [np.interp(k, x, m)], marker=mk, ms=6, mfc="none", mec=col, mew=0.9,
                    ls="none", zorder=6)
    # pre-fault reference of the window model (clearing-based x = -10 ms row)
    xp, mp, _, _ = series("lightgbm_tuned", "window_line", "pre")
    if len(xp):
        axes[0].text(0.98, 0.3, f"pre-fault, same features: {mp[0]:.3f} (off scale)",
                     transform=axes[0].transAxes, ha="right", va="bottom", fontsize=6.5,
                     color=INK2)
    ax.axvspan(50, 600, color=GRID, alpha=0.5, lw=0, zorder=0)
    ax.text(325, 0.935, "clearing times fall\nin U(0.05, 0.6) s", ha="center", va="top",
            fontsize=6.5, color=INK2)
    ax.set_xlim(150, 830)
    ax.set_xlabel("assessment instant after fault inception (ms)")
    ax.set_title("(b) inception-based", loc="left", fontsize=8)
    ax.legend(loc="lower right")
    axes[0].set_ylim(0.9, 1.003)
    _save(fig, out, "instant_curve")


# --------------------------------------------------------------------------------------------
# fig:prefault_rank - hour-protocol PR-AUC with CI, every pre-fault model
# --------------------------------------------------------------------------------------------
FAMILY_COLOUR = {
    "boosting": C["blue"],
    "GNN": C["orange"],
    "TFM zero-shot": C["aqua"],
    "TFM adapted": C["yellow"],
}


def _prefault_models(exp: Path) -> list[dict[str, Any]]:
    """Collect (label, family, ap, lo, hi, source) for the hour protocol, line_pre features."""
    items: list[dict[str, Any]] = []

    def add(label: str, fam: str, ap: float, lo: float, hi: float, source: str) -> None:
        items.append({"label": label, "family": fam, "ap": ap, "lo": lo, "hi": hi,
                      "source": source})

    # e1 boosting
    md = _md_tables((exp / "e1_boosting" / "tables.md").read_text())
    best = md.get("Best candidate per run, feature set and protocol (PR-AUC [95 % CI])", [])
    for r in best:
        if r["protocol"] != "hour" or r["tag"] != "posttrip30":
            continue
        if r["feature_set"] == "line_pre_nopt":
            ap, lo, hi = _ci_cell(r["pr_auc_ci"])
            algo = r["candidate"].split("__")[0]
            add(f"{'LightGBM' if algo == 'lightgbm' else 'XGBoost'} (boosting, tuned)", "boosting",
                ap, lo, hi, "e1_boosting/tables.md")
        elif r["feature_set"] == "line_pre":
            ap, lo, hi = _ci_cell(r["pr_auc_ci"])
            add("Boosting + post-trip features", "boosting", ap, lo, hi, "e1_boosting/tables.md")
    cb = exp / "e1_boosting" / "results" / "catboost100" / "tsa_benchmark.csv"
    if cb.exists():
        rows = [r for r in _read_csv(cb)
                if r["protocol"] == "hour" and r["feature_set"] == "line_pre"]
        if rows:
            r = max(rows, key=lambda r: _f(r["average_precision"]))
            add("CatBoost (tuned)", "boosting", _f(r["average_precision"]),
                _f(r["average_precision_lo"]), _f(r["average_precision_hi"]),
                "e1_boosting/results/catboost100/tsa_benchmark.csv")
    # e3 GNN
    for r in _read_csv(exp / "e3_deep" / "tables" / "prefault_results.csv"):
        if r["protocol"] != "hour" or r["family"] != "gnn":
            continue
        name = {"gnn_gat": "GAT", "gnn_gcn": "GCN", "gnn_sage": "GraphSAGE"}[r["model"]]
        add(name, "GNN", _f(r["average_precision"]), _f(r["average_precision_lo"]),
            _f(r["average_precision_hi"]), "e3_deep/tables/prefault_results.csv")
    # e2 TFM
    e2names = {"kumo_tabular": "Kumo", "mitra_v2": "Mitra", "tabicl_v2": "TabICLv2",
               "tabpfn_v2": "TabPFN v2", "tabdpt": "TabDPT"}
    for r in _read_csv(exp / "e2_tfm" / "summary.csv"):
        if r["protocol"] == "hour" and r["feature_set"] == "line_pre" and r["model"] in e2names:
            add(e2names[r["model"]], "TFM zero-shot", _f(r["pr_auc"]), _f(r["pr_auc_lo"]),
                _f(r["pr_auc_hi"]), "e2_tfm/summary.csv")
    # e5 TFM
    e5names = {"limix_2": "LimiX-2", "limix_16m": "LimiX-16M", "limix_2m": "LimiX-2M",
               "tabfm": "TabFM 1.0", "tabfm_v1_1": "TabFM 1.1",
               "tabfm_ctx100": "TabFM 1.0, ctx 100",
               "causilo": "Causilo", "tabpfn_3": "TabPFN 3", "tabpfn_3_5": "TabPFN 3.5"}
    d5 = json.load((exp / "e5_tfm2" / "done.json").open())
    for key, name in e5names.items():
        v = d5.get(f"line_pre__{key}__hour")
        if v and "average_precision" in v:
            add(name, "TFM zero-shot", v["average_precision"], v["ci"][0], v["ci"][1],
                "e5_tfm2/done.json")
    # e6 adaptation
    e6names = {"kumo_adj512": "Kumo, retrieval k=512", "kumo_adj2048": "Kumo, retrieval k=2048",
               "kumo_knn8192": "Kumo, retrieval k=8192",
               "tabicl_adj2048": "TabICLv2, retrieval k=2048",
               "tabpfn_adj2048": "TabPFN v2, retrieval k=2048",
               "tabpfn_ft": "TabPFN v2, fine-tuned",
               "mitra_ft": "Mitra, fine-tuned"}
    for p in sorted((exp / "e6_finetune" / "lanes").glob("*/done.json")):
        d6 = json.load(p.open())
        for key, name in e6names.items():
            v = d6.get(f"line_pre__{key}__hour")
            if v and "average_precision" in v:
                add(name, "TFM adapted", v["average_precision"], v["ci"][0], v["ci"][1],
                    str(p.relative_to(exp)))
    return items


def fig_prefault_rank(exp: Path, grid: Path, src: Path, out: Path) -> None:
    items = sorted(_prefault_models(exp), key=lambda d: d["ap"])
    n = len(items)
    fig, ax = plt.subplots(figsize=(SINGLE, 0.148 * n + 0.95))
    ax.grid(True, axis="x")
    y = np.arange(n)
    for i, d in enumerate(items):
        col = FAMILY_COLOUR[d["family"]]
        ax.barh(i, d["ap"], height=0.62, color=col, lw=0, zorder=3)
        ax.errorbar(d["ap"], i, xerr=[[d["ap"] - d["lo"]], [d["hi"] - d["ap"]]], fmt="none",
                    ecolor=INK, elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=4)
        ax.text(d["hi"] + 0.006, i, f"{d['ap']:.3f}", va="center", ha="left", fontsize=6.5,
                color=INK, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([d["label"] for d in items], fontsize=6.5)
    ax.set_xlim(0.4, 1.06)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("PR-AUC, hour protocol (95 % CI)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    from matplotlib.patches import Patch

    handles = [Patch(color=c, label=f) for f, c in FAMILY_COLOUR.items()]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2,
              handlelength=1.0, handleheight=0.8, fontsize=6.5, borderaxespad=0.0,
              columnspacing=1.0, handletextpad=0.4)
    top = items[-1]
    print(f"prefault_rank: {n} models; best {top['label']} {top['ap']:.3f} "
          f"[{top['lo']:.3f}, {top['hi']:.3f}]")
    _save(fig, out, "prefault_rank")


# --------------------------------------------------------------------------------------------
# fig:strata - heat-map of miss rate by family x stratum
# --------------------------------------------------------------------------------------------
STRATA = [("vn_kv", "220", "220"), ("vn_kv", "380", "380"),
          ("clearing_band", "0.2-0.3s", "0.2–0.3"),
          ("clearing_band", "0.3-0.4s", "0.3–0.4"),
          ("clearing_band", "0.4-0.5s", "0.4–0.5"), ("clearing_band", ">=0.5s", "≥0.5")]


def fig_strata(exp: Path, grid: Path, src: Path, out: Path) -> None:
    md = _md_tables((exp / "e1_boosting" / "tables.md").read_text())
    key = next(k for k in md if k.startswith("Per-stratum picture of the best hour-protocol"))
    boost = {(r["stratum_by"], r["stratum"]): (_f(r["miss_rate"]), int(r["n_positive"]))
             for r in md[key]}

    def from_csv(folder: Path, prefix: str) -> dict[tuple[str, str], tuple[float, int]]:
        d: dict[tuple[str, str], tuple[float, int]] = {}
        for by in ("vn_kv", "clearing_band"):
            p = folder / f"{prefix}__hour__by_{by}.csv"
            for r in _read_csv(p):
                d[(by, r[by])] = (_f(r["miss_rate"]), int(_f(r["n_positive"]) or 0))
        return d

    fams = [("Boosting (LightGBM)", boost),
            ("Kumo", from_csv(exp / "e2_tfm" / "strata", "line_pre__kumo_tabular")),
            ("LimiX-2", from_csv(exp / "e5_tfm2" / "strata", "line_pre__limix_2"))]
    Z = np.full((len(fams), len(STRATA)), np.nan)
    N = np.zeros(len(STRATA), dtype=int)
    for i, (_, d) in enumerate(fams):
        for j, (by, s, _) in enumerate(STRATA):
            if (by, s) in d:
                Z[i, j], N[j] = d[(by, s)]
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(SINGLE, 1.7))
    im = ax.imshow(Z, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            v = Z[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.55 else INK)
    ax.set_xticks(range(len(STRATA)))
    ax.set_xticklabels([f"{lab}\n({N[j]})" for j, (_, _, lab) in enumerate(STRATA)],
                       fontsize=6)
    ax.set_xlabel("stratum (unstable events in parentheses)", fontsize=7)
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([f for f, _ in fams])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axvline(1.5, color="white", lw=2)
    ax.text(0.5, -0.5 - 0.22, "voltage (kV)", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.text(3.5, -0.5 - 0.22, "clearing time (s)", ha="center", va="bottom", fontsize=7,
            color=INK2)
    # white 2 px gaps between cells
    ax.set_xticks(np.arange(-0.5, len(STRATA), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(fams), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("miss rate (hour protocol)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_visible(False)
    _save(fig, out, "strata")


# --------------------------------------------------------------------------------------------
# fig:cct_lever - direct vs CCT two-stage, all vs bracketed events
# --------------------------------------------------------------------------------------------
def fig_cct_lever(exp: Path, grid: Path, src: Path, out: Path) -> None:
    rows = _read_csv(exp / "e1_boosting" / "results" / "cct_two_stage_pt" / "results.csv")
    rows = [r for r in rows if r["protocol"] == "hour"]
    models = [("A_direct", "A: direct classifier", C["blue"]),
              ("B_cct_regression", "B: CCT regression", C["orange"]),
              ("B2_cct_censored", "B2: censored CCT", C["aqua"])]
    subsets = [("all", "all events"), ("two_sided_pairs", "two-sided\nCCT pair"),
               ("swept_bracketed_lines", "swept,\nbracketed lines")]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
    ax.grid(True, axis="y")
    w = 0.26
    for j, (mkey, mlab, col) in enumerate(models):
        for i, (skey, _) in enumerate(subsets):
            r = next(r for r in rows if r["model"] == mkey and r["subset"] == skey)
            ap, lo, hi = (_f(r["average_precision"]), _f(r["average_precision_lo"]),
                          _f(r["average_precision_hi"]))
            x = i + (j - 1) * w
            ax.bar(x, ap, width=w - 0.03, color=col, lw=0, zorder=3, label=mlab if i == 0 else None)
            ax.errorbar(x, ap, yerr=[[ap - lo], [hi - ap]], fmt="none", ecolor=INK, elinewidth=0.6,
                        capsize=1.5, capthick=0.6, zorder=4)
            ax.text(x, hi + 0.012, f"{ap:.2f}", ha="center", va="bottom", fontsize=6, color=INK)
    ns = {s: next(r for r in rows if r["subset"] == s) for s, _ in subsets}
    ax.set_xticks(range(len(subsets)))
    ax.set_xticklabels([f"{lab}\n$n$ = {int(ns[s]['n']):,} / {int(ns[s]['n_unstable']):,}"
                        for s, lab in subsets], fontsize=6.5)
    ax.set_xlabel("event subset ($n$ events / unstable)")
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("PR-AUC, hour protocol (95 % CI)")
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), columnspacing=1.0,
              handlelength=1.0, handleheight=0.8, borderaxespad=0.0)
    _save(fig, out, "cct_lever")


# --------------------------------------------------------------------------------------------
# fig:latency - latency vs hour PR-AUC
# --------------------------------------------------------------------------------------------
def fig_latency(exp: Path, grid: Path, src: Path, out: Path) -> None:
    lat = [json.loads(line) for line in (exp / "e5_tfm2" / "latency.jsonl").open() if line.strip()]
    pr = {d["label"]: d for d in _prefault_models(exp)}
    # latency model key -> (label in the PR-AUC list, display label)
    key_map = {
        "xgboost": ("XGBoost (boosting, tuned)", "XGBoost"),
        "lightgbm": ("LightGBM (boosting, tuned)", "LightGBM"),
        "tabdpt": ("TabDPT", "TabDPT"), "tabicl_v2": ("TabICLv2", "TabICLv2"),
        "tabpfn_v2": ("TabPFN v2", "TabPFN v2"), "causilo": ("Causilo", "Causilo"),
        "mitra_v2": ("Mitra", "Mitra"), "limix_2m": ("LimiX-2M", "LimiX-2M"),
        "limix_16m": ("LimiX-16M", "LimiX-16M"), "tabfm_v1_1": ("TabFM 1.1", "TabFM 1.1"),
        "kumo_tabular": ("Kumo", "Kumo"), "tabfm": ("TabFM 1.0", "TabFM 1.0"),
        "limix_2": ("LimiX-2", "LimiX-2"),
    }
    # the boosting hour PR-AUC: tables.md names the best tuned candidate (one algorithm) only;
    # map both CPU rows to the best boosting row when the other algorithm is absent.
    boost = next((d for d in pr.values() if d["family"] == "boosting"
                  and "post-trip" not in d["label"] and "CatBoost" not in d["label"]), None)
    pts = []
    for d in lat:
        if d["cache"]:
            continue
        lab, disp = key_map.get(d["model"], (None, d["model"]))
        row = pr.get(lab) if lab else None
        if row is None and d["model"] in ("xgboost", "lightgbm") and boost is not None:
            row = boost
        if row is None:
            print(f"latency: no hour PR-AUC for {d['model']}, skipped")
            continue
        pts.append({"x": d["median_ms"], "y": row["ap"], "lo": row["lo"], "hi": row["hi"],
                    "label": disp, "device": d["device"]})
    fig, ax = plt.subplots(figsize=(SINGLE, 2.8))
    ax.grid(True, axis="both")
    ax.set_xscale("log")
    for dev, col, mk, lab in (("cpu", C["blue"], "o", "ML, CPU"),
                              ("cuda", C["orange"], "s", "ML, GPU")):
        sel = [p for p in pts if p["device"] == dev]
        if not sel:
            continue
        ax.errorbar([p["x"] for p in sel], [p["y"] for p in sel],
                    yerr=[[p["y"] - p["lo"] for p in sel], [p["hi"] - p["y"] for p in sel]],
                    fmt=mk, ms=4, color=col, mec="white", mew=0.4, ecolor=col, elinewidth=0.6,
                    capsize=1.2, label=lab, zorder=4)
    # reference methods
    ax.plot([13_200], [1.0], marker="D", ms=4.5, color=INK, ls="none",
            label="time-domain simulation", zorder=5)
    ax.plot([1_400], [0.893], marker="^", ms=5, color=C["violet"], ls="none",
            label="SIME (post-clearing)", zorder=5)
    offs = {  # label offsets (points) and alignment chosen to avoid overlaps
        "XGBoost": (0, -9, "center"), "LightGBM": (0, 8, "center"), "TabDPT": (6, 0, "left"),
        "LimiX-2M": (-6, -5, "right"), "TabPFN v2": (6, -6, "left"),
        "TabFM 1.1": (-2, -10, "center"), "TabFM 1.0": (6, -7, "left"),
        "LimiX-2": (0, 8, "center"),
    }
    leaders = {  # labels placed in free space (data coordinates) with a leader line
        "Causilo": (120, 0.88, "right"), "TabICLv2": (120, 0.845, "right"),
        "Mitra": (700, 0.915, "center"), "LimiX-16M": (2600, 0.812, "left"),
        "Kumo": (9000, 0.79, "left"),
    }
    for p in pts:
        if p["label"] in leaders:
            lx, ly, ha = leaders[p["label"]]
            ax.annotate(p["label"], (p["x"], p["y"]), xytext=(lx, ly), textcoords="data",
                        fontsize=6, color=INK, ha=ha, va="center",
                        arrowprops={"arrowstyle": "-", "lw": 0.5, "color": INK2, "shrinkA": 0,
                                    "shrinkB": 3},
                        path_effects=[patheffects.withStroke(linewidth=1.8, foreground="white")])
            continue
        dx, dy, ha = offs.get(p["label"], (5, 4, "left"))
        ax.annotate(p["label"], (p["x"], p["y"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6, color=INK, ha=ha, va="center",
                    path_effects=[patheffects.withStroke(linewidth=1.8, foreground="white")])
    ax.annotate("TDS 13.2 s, PR-AUC 1.0", (13_200, 1.0), xytext=(-6, -1),
                textcoords="offset points", fontsize=6, ha="right", va="center", color=INK)
    ax.annotate("SIME ≈1.4 s, 0.893", (1_400, 0.893), xytext=(6, 3), textcoords="offset points",
                fontsize=6, ha="left", va="bottom", color=C["violet"])
    ax.set_xlim(0.03, 150_000)
    ax.set_ylim(0.68, 1.025)
    ax.set_xlabel("single-event latency, median (ms, log)")
    ax.set_ylabel("PR-AUC, hour protocol (95 % CI)")
    ax.legend(loc="upper left", ncol=1, handletextpad=0.4, borderaxespad=0.2)
    _save(fig, out, "latency")


# ============================================================================================
# T15: maps, grid dynamics and real events, at the IEEE Access widths (3.36 in / 6.99 in)
# ============================================================================================
ACC_COL = 3.36
ACC_TXT = 6.99
HOME = Path.home()
PREP = HOME / "data" / "ml_v41dyn" / "prep"
EVENTS = HOME / "data" / "volume" / "outputs" / "v41dyn" / "738439695169033" / "_delta_stage" / "events"
MAP_CASE = "high_load_t10"
FACTS: dict[str, Any] = {}  # numbers stated in the captions, written to figures/figure_numbers.json


def _fact(key: str, value: Any, source: str) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        value = None  # strict JSON
    FACTS[key] = {"value": value, "source": source}
    return value


def _case_sheets(case: str, sheets: list[str]) -> dict[str, Any]:
    import pandas as pd

    return pd.read_excel(PREP / f"{case}.xlsx", sheet_name=sheets)


def _borders(grid: Path) -> tuple[Any, Any]:
    import geopandas as gpd

    ne = Path.home() / ".cache" / "tranpy" / "ne_50m_admin_0_countries.geojson"
    if ne.exists():
        world = gpd.read_file(ne)
        iso = next(c for c in ("ADM0_A3", "ISO_A3", "adm0_a3") if c in world.columns)
        keep = ["DEU", "AUT", "CHE", "FRA", "LUX", "BEL", "NLD", "DNK", "POL", "CZE"]
        return (world[world[iso].isin(keep)].to_crs("EPSG:3035"),
                world[world[iso] == "DEU"].to_crs("EPSG:3035"))
    zones = gpd.read_file(grid, layer="bidding_zones").to_crs("EPSG:3035")
    return zones, zones[zones["zone"] == "DE-LU"]


def _basemap(ax: Any, grid: Path, pad: float = 0.03) -> None:
    border, de = _borders(grid)
    border.plot(ax=ax, facecolor="#f6f5f2", edgecolor="#b9b7b0", linewidth=0.4, zorder=1)
    de.plot(ax=ax, facecolor="white", edgecolor="#6f6d66", linewidth=0.6, zorder=2)
    minx, miny, maxx, maxy = de.total_bounds
    p = pad * (maxy - miny)
    ax.set_xlim(minx - p, maxx + p)
    ax.set_ylim(miny - p, maxy + p)
    ax.set_aspect("equal")
    ax.axis("off")


def _corridors(grid: Path, case: str = MAP_CASE) -> tuple[Any, dict[int, tuple[str, str]]]:
    """GeoPackage lines dissolved per unordered substation-bus pair (one geometry per corridor)
    and the ANDES line index -> pair map (Line sheet bus1/bus2 -> Bus sheet names, which are
    the gpkg bus names). Parallel circuits share a pair and cannot be told apart, so every
    line-level value is aggregated per corridor."""
    import geopandas as gpd

    lines = gpd.read_file(grid, layer="lines").to_crs("EPSG:3035")
    lines["key"] = ["~".join(sorted((a, b))) for a, b in zip(lines["bus1"], lines["bus2"], strict=True)]
    cor = lines.dissolve(by="key", aggfunc={"voltage_kv": "max", "mrid": "count"}).rename(
        columns={"mrid": "n_records"})
    x = _case_sheets(case, ["Line", "Bus"])
    names = dict(zip(x["Bus"]["idx"], x["Bus"]["name"], strict=True))
    L = x["Line"]
    # fault_line = row position in the Line sheet (Line_0..Line_1014, then the Tr_* transformers)
    amap = {k: "~".join(sorted((names[a], names[b])))
            for k, (a, b) in enumerate(zip(L["bus1"], L["bus2"], strict=True))}
    return cor, amap


def _map_lines(ax: Any, cor: Any, values: dict[Any, Any], bins: list[float], colours: list[str],
               base_colour: str = "#d9d8d3") -> None:
    """Corridors without a value thin and grey; others coloured by bin, larger values on top."""
    cor.plot(ax=ax, color=base_colour, linewidth=0.35, zorder=3)
    for k in range(len(colours)):
        keys = [key for key, v in values.items() if key in cor.index
                and bins[k] <= v < bins[k + 1]]
        if keys:
            cor.loc[keys].plot(ax=ax, color=colours[k], linewidth=0.6 + 0.25 * k / len(colours),
                               zorder=4 + k)


def _bin_legend(ax: Any, labels: list[str], colours: list[str], title: str, loc: str = "upper left",
                extra: list[tuple[str, str]] | None = None) -> None:
    from matplotlib.lines import Line2D

    h = [Line2D([], [], color=c, lw=1.6, label=lab) for lab, c in zip(labels, colours, strict=True)]
    for lab, c in extra or []:
        h.append(Line2D([], [], color=c, lw=0.8, label=lab))
    ax.legend(handles=h, title=title, loc=loc, title_fontsize=6.5, fontsize=6, borderaxespad=0.1,
              handlelength=1.4, labelspacing=0.25, alignment="left")


# --------------------------------------------------------------------------------------------
# fig_map_cct - bracketed CCT per corridor, one case
# --------------------------------------------------------------------------------------------
def fig_map_cct(exp: Path, grid: Path, src: Path, out: Path) -> None:
    import pandas as pd

    p = exp / "e4_sim" / "cct" / f"cct_{MAP_CASE}.parquet"
    c = pd.read_parquet(p)
    cor, amap = _corridors(grid)
    c["key"] = c["fault_line"].astype(int).map(amap)
    # per corridor: min over terminals and parallel circuits; 'above' = stable at 1.0 s
    val: dict[Any, float] = {}
    for key, g in c.groupby("key"):
        br = g[g["status"] == "bracketed"]
        if (g["status"] == "below").any():
            val[key] = 0.0
        elif len(br):
            val[key] = float(br["cct_lo"].min())
        elif (g["status"] == "above").all():
            val[key] = 9.0
    bins = [0.0, 0.15, 0.25, 0.4, 0.6, 1.0001, 99]
    labs = ["< 0.15 s", "0.15–0.25 s", "0.25–0.4 s", "0.4–0.6 s", "0.6–1.0 s", "> 1.0 s (stable)"]
    cols = [SEQ_BLUE[12], SEQ_BLUE[9], SEQ_BLUE[7], SEQ_BLUE[5], SEQ_BLUE[3], SEQ_BLUE[0]]
    order = [5, 4, 3, 2, 1, 0]  # draw the critical (short-CCT) corridors last
    fig, ax = plt.subplots(figsize=(ACC_COL, 4.2))
    _basemap(ax, grid)
    cor.plot(ax=ax, color="#d9d8d3", linewidth=0.35, zorder=3)
    for z, i in enumerate(order):
        keys = [k for k, v in val.items() if k in cor.index and bins[i] <= v < bins[i + 1]]
        if keys:
            cor.loc[keys].plot(ax=ax, color=cols[i], linewidth=0.5 + 0.18 * (5 - i), zorder=4 + z)
    _bin_legend(ax, labs, cols, "lowest CCT lower bound", extra=[("no bracket / not swept", "#d9d8d3")])
    n_cor = len(val)
    n_short = sum(v < 0.25 for v in val.values())
    kv = {k: float(cor.loc[k, "voltage_kv"]) for k in val if k in cor.index}
    s220 = [v for k, v in val.items() if kv.get(k) == 220 and v < 9]
    s380 = [v for k, v in val.items() if kv.get(k) == 380 and v < 9]
    src_s = f"e4_sim/cct/cct_{MAP_CASE}.parquet"
    _fact("map_cct.case", MAP_CASE, src_s)
    _fact("map_cct.rows", int(len(c)), src_s)
    _fact("map_cct.status_counts", c["status"].value_counts().to_dict(), src_s)
    _fact("map_cct.corridors", n_cor, src_s)
    _fact("map_cct.corridors_unmatched", int(sum(k not in cor.index for k in val)), src_s)
    _fact("map_cct.corridors_lt_0_25s", int(n_short), src_s)
    _fact("map_cct.corridors_stable_1s", int(sum(v == 9 for v in val.values())), src_s)
    _fact("map_cct.median_bracketed_220", float(np.median(s220)) if s220 else None, src_s)
    _fact("map_cct.median_bracketed_380", float(np.median(s380)) if s380 else None, src_s)
    _fact("map_cct.min_cct_lo", float(c.loc[c.status == "bracketed", "cct_lo"].min()), src_s)
    print("map_cct:", {k: v["value"] for k, v in FACTS.items() if k.startswith("map_cct")})
    _save(fig, out, "map_cct")


# --------------------------------------------------------------------------------------------
# fig_map_unstable - share of unstable events per corridor over the 18 cases
# --------------------------------------------------------------------------------------------
def _events() -> Any:
    import pandas as pd

    fs = sorted(EVENTS.glob("*.parquet"))
    e = pd.concat([pd.read_parquet(f, columns=["case", "sample_id", "label", "fault_line"]) for f in fs])
    e = e.dropna(subset=["label"]).drop_duplicates(["case", "sample_id"])
    return e


def fig_map_unstable(exp: Path, grid: Path, src: Path, out: Path) -> None:
    e = _events()
    cor, amap = _corridors(grid)
    e["key"] = e["fault_line"].astype(int).map(amap)
    g = e.groupby("key")["label"].agg(["mean", "size", "sum"])
    bins = [0.0, 1e-9, 0.1, 0.25, 0.5, 1.01]
    labs = ["0 (all stable)", "0–10 %", "10–25 %", "25–50 %", "> 50 %"]
    cols = [SEQ_BLUE[0], SEQ_BLUE[4], SEQ_BLUE[7], SEQ_BLUE[10], SEQ_BLUE[12]]
    fig, ax = plt.subplots(figsize=(ACC_COL, 4.2))
    _basemap(ax, grid)
    cor.plot(ax=ax, color="#d9d8d3", linewidth=0.35, zorder=3)
    val = g["mean"].to_dict()
    for i in range(len(cols)):
        keys = [k for k, v in val.items() if k in cor.index and bins[i] <= v < bins[i + 1]]
        if keys:
            cor.loc[keys].plot(ax=ax, color=cols[i], linewidth=0.5 + 0.2 * i, zorder=4 + i)
    _bin_legend(ax, labs, cols, "unstable share of faults", extra=[("no fault event", "#d9d8d3")])
    s = "v41dyn/_delta_stage/events/*.parquet"
    kv = cor["voltage_kv"].to_dict()
    unst = e[e.label == 1]
    _fact("map_unstable.events", int(len(e)), s)
    _fact("map_unstable.cases", int(e["case"].nunique()), s)
    _fact("map_unstable.unstable", int(e["label"].sum()), s)
    _fact("map_unstable.corridors_with_events", int(len(g)), s)
    _fact("map_unstable.corridors_any_unstable", int((g["sum"] > 0).sum()), s)
    top = g.sort_values("sum", ascending=False)
    n10 = max(1, int(round(0.1 * len(g))))
    _fact("map_unstable.top10pct_corridors_share_of_unstable",
          round(float(top["sum"].iloc[:n10].sum() / g["sum"].sum()), 3), s)
    _fact("map_unstable.share_unstable_on_220kV",
          round(float(np.mean([kv.get(k) == 220 for k in unst["key"]])), 3), s)
    _fact("map_unstable.share_events_on_220kV",
          round(float(np.mean([kv.get(k) == 220 for k in e["key"]])), 3), s)
    print("map_unstable:", {k: v["value"] for k, v in FACTS.items() if k.startswith("map_unstable")})
    _save(fig, out, "map_unstable")


# --------------------------------------------------------------------------------------------
# fig_map_errors - pre-fault misses per corridor: boosting vs LimiX-2 (hour protocol)
# --------------------------------------------------------------------------------------------
def fig_map_errors(exp: Path, grid: Path, src: Path, out: Path) -> None:
    import pandas as pd

    lim_p = exp / "e5_tfm2" / "oof" / "line_pre__limix_2__hour.parquet"
    lim = pd.read_parquet(lim_p)
    md = _md_tables((exp / "e1_boosting" / "tables.md").read_text())
    best = md["Best candidate per run, feature set and protocol (PR-AUC [95 % CI])"]
    cand = next(r["candidate"] for r in best if r["tag"] == "posttrip30"
                and r["feature_set"] == "line_pre_nopt" and r["protocol"] == "hour")
    b_p = exp / "e1_boosting" / "results" / "posttrip30" / "tsa_oof_predictions.parquet"
    b = pd.read_parquet(b_p, filters=[("feature_set", "==", "line_pre_nopt"), ("protocol", "==", "hour"),
                                      ("candidate", "==", cand)])
    m = lim.merge(b[["case", "sample_id", "predicted_label", "label"]], on=["case", "sample_id"],
                  suffixes=("", "_b"), validate="one_to_one")
    assert (m["label"] == m["label_b"]).all()
    pos = m[m["label"] == 1].copy()
    pos["miss_l"] = pos["pred"] == 0
    pos["miss_b"] = pos["predicted_label"] == 0
    cor, amap = _corridors(grid)
    pos["key"] = pos["fault_line"].astype(int).map(amap)
    bins = [1, 2, 4, 8, 16, 10_000]
    labs = ["1", "2–3", "4–7", "8–15", "≥ 16"]
    cols = [SEQ_BLUE[3], SEQ_BLUE[6], SEQ_BLUE[8], SEQ_BLUE[10], SEQ_BLUE[12]]
    fig, axes = plt.subplots(1, 2, figsize=(ACC_TXT, 4.3))
    name_b = {"lightgbm": "LightGBM", "xgboost": "XGBoost"}.get(cand.split("__")[0], cand)
    s_l, s_b = "e5_tfm2/oof/line_pre__limix_2__hour.parquet", "e1_boosting/results/posttrip30/tsa_oof_predictions.parquet"
    for ax, col, title in ((axes[0], "miss_b", f"(a) boosting ({name_b}, tuned)"),
                           (axes[1], "miss_l", "(b) LimiX-2 (zero-shot)")):
        _basemap(ax, grid)
        cor.plot(ax=ax, color="#d9d8d3", linewidth=0.35, zorder=3)
        cnt = pos.groupby("key")[col].sum()
        val = cnt[cnt > 0].to_dict()
        for i in range(len(cols)):
            keys = [k for k, v in val.items() if k in cor.index and bins[i] <= v < bins[i + 1]]
            if keys:
                cor.loc[keys].plot(ax=ax, color=cols[i], linewidth=0.6 + 0.25 * i, zorder=4 + i)
        ax.set_title(f"{title}: {int(pos[col].sum())} missed", loc="left", fontsize=8)
    _bin_legend(axes[0], labs, cols, "missed unstable events", extra=[("no miss", "#d9d8d3")])
    kv = cor["voltage_kv"].to_dict()
    pos["kv"] = pos["key"].map(kv)
    both = pos["miss_l"] & pos["miss_b"]
    either = pos["miss_l"] | pos["miss_b"]
    srcs = f"{s_l}; {s_b} ({cand}, line_pre_nopt, hour)"
    _fact("map_errors.boosting_candidate", cand, s_b)
    _fact("map_errors.unstable_events", int(len(pos)), srcs)
    _fact("map_errors.miss_boosting", int(pos["miss_b"].sum()), srcs)
    _fact("map_errors.miss_limix2", int(pos["miss_l"].sum()), srcs)
    _fact("map_errors.miss_both", int(both.sum()), srcs)
    _fact("map_errors.limix_misses_also_missed_by_boosting",
          round(float(both.sum() / pos["miss_l"].sum()), 3), srcs)
    _fact("map_errors.jaccard_missed_events", round(float(both.sum() / either.sum()), 3), srcs)
    _fact("map_errors.share_220kV_of_misses_boosting",
          round(float((pos.loc[pos.miss_b, "kv"] == 220).mean()), 3), srcs)
    _fact("map_errors.share_220kV_of_misses_limix2",
          round(float((pos.loc[pos.miss_l, "kv"] == 220).mean()), 3), srcs)
    _fact("map_errors.share_220kV_of_unstable", round(float((pos["kv"] == 220).mean()), 3), srcs)
    cb = pos.groupby("key")["miss_b"].sum()
    cl = pos.groupby("key")["miss_l"].sum()
    _fact("map_errors.corridor_count_spearman",
          round(float(pd.concat([cb, cl], axis=1).corr(method="spearman").iloc[0, 1]), 3), srcs)
    print("map_errors:", {k: v["value"] for k, v in FACTS.items() if k.startswith("map_errors")})
    _save(fig, out, "map_errors")


# --------------------------------------------------------------------------------------------
# fig_map_inertia - synchronous machines by kinetic energy and fuel, TSO control areas
# --------------------------------------------------------------------------------------------
FUEL_GROUP = {"lignite": "lignite", "hard_coal": "hard coal", "gas": "gas/oil", "oil": "gas/oil",
              "nuclear": "nuclear", "hydro": "hydro", "pumped_storage": "hydro"}
FUEL_COLOUR = {"lignite": C["orange"], "hard coal": C["violet"], "gas/oil": C["blue"], "nuclear": C["yellow"], "hydro": C["aqua"],
               "other": C["magenta"]}


def _machines(grid: Path, case: str = MAP_CASE) -> Any:
    """One row per ANDES machine (GENROU, GENCLS): connection bus, Sn, H, E = H Sn, fuel, point."""
    import geopandas as gpd
    import pandas as pd

    x = _case_sheets(case, ["GENROU", "GENCLS", "Line", "Bus"])
    names = dict(zip(x["Bus"]["idx"], x["Bus"]["name"], strict=True))
    rows = []
    for model, g in (("GENROU", x["GENROU"]), ("GENCLS", x["GENCLS"])):
        g = g[g["u"] == 1]
        for _, r in g.iterrows():
            rows.append({"machine": r["idx"], "model": model, "bus": int(r["bus"]),
                         "bus_name": names[int(r["bus"])], "Sn": float(r["Sn"]),
                         "H": float(r["M"]) / 2.0})
    m = pd.DataFrame(rows)
    m["E_mws"] = m["H"] * m["Sn"]
    # the gpkg buses layer holds the generator terminals too (kind 'terminal', with their site)
    buses = gpd.read_file(grid, layer="buses").to_crs("EPSG:3035").dropna(subset=["andes_idx"])
    buses = buses.drop_duplicates("andes_idx").set_index(buses.drop_duplicates("andes_idx")["andes_idx"].astype(int))
    m["geometry"] = m["bus"].map(buses["geometry"])
    m["site"] = m["bus"].map(buses["site"])
    # fuel: the machine group of the case (prep machines.csv: thermal/hydro); thermal units take
    # the largest installed fuel category of their site (gpkg plant_aggregates)
    grp = pd.read_csv(PREP / f"{case}.machines.csv").set_index("machine")["group"]
    pa = gpd.read_file(grid, layer="plant_aggregates", ignore_geometry=True)
    pa = pa[~pa["fuel_category"].isin(["hydro", "pumped_storage", "storage"])]
    fuel = (pa.groupby(["site", "fuel_category"])["capacity_mw"].sum().reset_index()
            .sort_values("capacity_mw").drop_duplicates("site", keep="last")
            .set_index("site")["fuel_category"])

    def _fuel(mod: str, mach: str, site: Any) -> str:
        if mod == "GENCLS":
            return "equivalent"
        if grp.get(mach) == "hydro":
            return "hydro"
        return FUEL_GROUP.get(fuel.get(site, ""), "other")

    m["fuel"] = [_fuel(a, b, c) for a, b, c in zip(m["model"], m["machine"], m["site"], strict=True)]
    # thermal units: prefer the fuel of the nearest MaStR unit of >= 50 MW within 3 km of the
    # terminal (gpkg plants layer, point of the plant itself)
    pl = gpd.read_file(grid, layer="plants").to_crs("EPSG:3035")
    pl = pl[(pl["capacity_mw"] >= 50) & ~pl["fuel_category"].isin(["hydro", "pumped_storage", "storage"])]
    gm = gpd.GeoDataFrame(m[["machine"]], geometry=m["geometry"], crs="EPSG:3035")
    near = gpd.sjoin_nearest(gm, pl[["fuel_category", "capacity_mw", "geometry"]], how="left",
                             max_distance=3000, distance_col="dist")
    near = near.sort_values("capacity_mw").drop_duplicates("machine", keep="last").set_index("machine")
    th = (m["model"] == "GENROU") & (m["fuel"] != "hydro")
    nf = m.loc[th, "machine"].map(near["fuel_category"])
    m.loc[th & nf.notna().reindex(m.index, fill_value=False), "fuel"] = nf.dropna().map(
        lambda f: FUEL_GROUP.get(f, "other"))
    return gpd.GeoDataFrame(m, geometry="geometry", crs="EPSG:3035")


def fig_map_inertia(exp: Path, grid: Path, src: Path, out: Path) -> None:
    import geopandas as gpd

    m = _machines(grid)
    fig, ax = plt.subplots(figsize=(ACC_COL, 5.0))
    _basemap(ax, grid, pad=0.06)
    ca = gpd.read_file(grid, layer="control_areas").to_crs("EPSG:3035")
    ca.boundary.plot(ax=ax, color=INK2, linewidth=0.5, linestyle=(0, (3, 2)), zorder=3)
    # TSO labels at hand-picked free spots inside each area (lon, lat)
    spots = {"50Hertz": (12.9, 53.25), "TenneT": (11.3, 49.5), "Amprion": (7.75, 50.05),
             "TransnetBW": (9.0, 48.2)}
    lab = gpd.GeoSeries(gpd.points_from_xy([v[0] for v in spots.values()],
                                           [v[1] for v in spots.values()]), crs="EPSG:4326").to_crs("EPSG:3035")
    for name, p in zip(spots, lab, strict=True):
        ax.text(p.x, p.y, name, fontsize=6, color=INK2, ha="center", va="center", zorder=4,
                style="italic", path_effects=[patheffects.withStroke(linewidth=1.6, foreground="white")])
    de = m[(m.model == "GENROU") & m.geometry.notna()]
    smax = de["E_mws"].max()
    def size(e: Any) -> Any:
        return 4 + 110 * np.asarray(e) / smax
    for fu in ("lignite", "hard coal", "gas/oil", "nuclear", "hydro", "other"):
        s = de[de.fuel == fu]
        if len(s):
            ax.scatter(s.geometry.x, s.geometry.y, s=size(s["E_mws"]), color=FUEL_COLOUR[fu],
                       alpha=0.85, edgecolor="white", linewidth=0.4, zorder=6, label=fu)
    ex = m[(m.model == "GENCLS") & m.geometry.notna()]
    ax.scatter(ex.geometry.x, ex.geometry.y, s=14, facecolor="none", edgecolor=INK2, linewidth=0.6,
               marker="D", zorder=5, label="border equivalent (GENCLS)")
    from matplotlib.lines import Line2D

    hf = [Line2D([], [], ls="none", marker="o", ms=4.5, color=FUEL_COLOUR[fu], mec="white", mew=0.4,
                 label=fu) for fu in ("lignite", "hard coal", "gas/oil", "nuclear", "hydro", "other")
          if (de.fuel == fu).any()]
    hf.append(Line2D([], [], ls="none", marker="D", ms=3.5, mfc="none", mec=INK2, mew=0.6,
                     label="border equivalent"))
    ref = [1000, 5000, 10000]
    hs = [ax.scatter([], [], s=size(v), color="#bdbbb5", edgecolor="white", lw=0.4,
                     label=f"$HS_n$ = {v / 1000:g} GWs") for v in ref]
    fig.legend(handles=hf + hs, loc="outside lower center", ncol=3, fontsize=6, borderaxespad=0.1,
               handletextpad=0.3, labelspacing=0.45, columnspacing=0.8)
    s = f"ml_v41dyn/prep/{MAP_CASE}.xlsx (GENROU, GENCLS); grid_model.gpkg buses/plants/control_areas"
    _fact("machines.case", MAP_CASE, s)
    _fact("machines.total", int(len(m)), s)
    _fact("machines.genrou_de", int((m.model == "GENROU").sum()), s)
    _fact("machines.gencls_border", int((m.model == "GENCLS").sum()), s)
    _fact("machines.unplaced", int(m.geometry.isna().sum()), s)
    _fact("machines.de_kinetic_gws", round(float(de["E_mws"].sum()) / 1e3, 1), s)
    _fact("machines.de_sn_gva", round(float(m.loc[m.model == "GENROU", "Sn"].sum()) / 1e3, 1), s)
    _fact("machines.de_h_range_s", [float(de["H"].min()), float(de["H"].max())], s)
    _fact("machines.fuel_counts", de["fuel"].value_counts().to_dict(), s)
    _fact("machines.fuel_kinetic_share", (de.groupby("fuel")["E_mws"].sum() / de["E_mws"].sum())
          .round(3).to_dict(), s)
    print("map_inertia:", {k: v["value"] for k, v in FACTS.items() if k.startswith("machines")})
    _save(fig, out, "map_inertia")


# --------------------------------------------------------------------------------------------
# grid dynamics: one stable / one unstable event on the same line, straddling its CCT
# (traces from scripts/paper_dyn_sim.py: same engine and scenario as e4_sim cct_bracket.py)
# --------------------------------------------------------------------------------------------
def _dyn(src: Path) -> dict[str, Any]:
    d = src / "dyn"
    meta = json.load((d / "dyn_meta.json").open())
    x = _case_sheets(meta["case"], ["GENROU", "GENCLS"])
    par = {}
    for g in (x["GENROU"], x["GENCLS"]):
        for i, mm, sn in zip(g["idx"], g["M"], g["Sn"], strict=True):
            par[str(i)] = (float(mm) / 2.0, float(sn))
    gmap = {a: b for a, b, _ in json.load((HOME / "data" / "ml_v41dyn" / "gen_maps.json").open())
            [meta["case"]]["gen_map"]}
    runs = {}
    for tag in ("stable", "unstable"):
        z = np.load(d / f"dyn_{tag}.npz")
        names = [str(n) for n in z["names"]]
        V = z["values"].astype(np.float64)
        col = {n: j for j, n in enumerate(names)}
        gens = [n.split("|")[0] for n in names if n.endswith("c:firot in deg")]
        H = np.array([par[gmap[g]][0] for g in gens])
        Sn = np.array([par[gmap[g]][1] for g in gens])
        delta = np.deg2rad(V[:, [col[f"{g}|c:firot in deg"] for g in gens]])
        speed = V[:, [col[f"{g}|s:xspeed in p.u."] for g in gens]]
        pe = V[:, [col[f"{g}|s:P1 in MW"] for g in gens]]
        fb = meta["bus1"] if meta["fault_position"] == 0 else meta["bus2"]
        u = V[:, col[f"Bus {fb:04d}|m:u in p.u."]]
        Mw = H * Sn
        coi = (delta * Mw).sum(1) / Mw.sum()
        w_coi = (speed * Mw).sum(1) / Mw.sum()
        runs[tag] = dict(t=z["time"] - 1.0, delta=delta, rel=np.rad2deg(delta - coi[:, None]),
                         speed=speed, pe=pe, u=u, w_coi=w_coi, H=H, Sn=Sn,
                         clr=meta[tag]["clearing_s"], label=meta[tag]["label"],
                         german=np.array([gmap[g].startswith("M") for g in gens]))
    return {"meta": meta, "runs": runs}


def _sime(run: dict[str, Any], critical: Any = None) -> Any:
    from tranpy.conventional.sime import inertia_coefficient, omib, sime_margin

    t, clr = run["t"], run["clr"]
    M = inertia_coefficient(run["H"] * run["Sn"])
    omega = (run["speed"] - 1.0) * 2 * np.pi * 50.0
    pm = run["pe"][t < -0.02].mean(0)  # mechanical power = pre-fault electrical power
    k = np.searchsorted(t, clr + 1e-6)
    res = sime_margin(run["delta"][k:], omega[k:], pm, run["pe"][k:], M, time=t[k:], critical=critical)
    full = omib(run["delta"], omega, pm, run["pe"], M, res.critical, time=t)
    return res, full


def fig_dyn_angles(exp: Path, grid: Path, src: Path, out: Path) -> None:
    D = _dyn(src)
    meta, R = D["meta"], D["runs"]
    res_u, _ = _sime(R["unstable"])
    crit = res_u.critical
    fig, axes = plt.subplots(1, 2, figsize=(ACC_TXT, 2.3), sharey=True)
    for ax, tag, title in ((axes[0], "stable", "(a) stable"), (axes[1], "unstable", "(b) unstable")):
        r = R[tag]
        ax.grid(True, axis="y")
        ax.axvspan(0, r["clr"], color=C["red"], alpha=0.12, lw=0)
        for j in np.where(~crit)[0]:
            ax.plot(r["t"], r["rel"][:, j], color="#b9b7b0", lw=0.4, zorder=2)
        for j in np.where(crit)[0]:
            ax.plot(r["t"], r["rel"][:, j], color=C["orange"], lw=0.9, zorder=3)
        ax.axhline(180, color=INK2, ls=":", lw=0.6)
        ax.axhline(-180, color=INK2, ls=":", lw=0.6)
        ax.set_xlim(-0.2, 6.0)
        ax.set_ylim(-200, 400)
        ax.set_xlabel("time after fault inception (s)")
        ax.set_title(f"{title}: clearing {r['clr']:.3f} s", loc="left", fontsize=8)
    axes[0].set_ylabel("rotor angle − COI angle (deg)")
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([], [], color=C["orange"], lw=0.9,
                                   label=f"critical machines (SIME, {int(crit.sum())})"),
                            Line2D([], [], color="#b9b7b0", lw=0.6, label="other machines"),
                            Line2D([], [], color=INK2, ls=":", lw=0.6, label="±180° out-of-step rule")],
                   loc="upper left", fontsize=6.5)
    s = f"paper_src/dyn/dyn_*.npz ({meta['case']}, Line_{meta['fault_line']}); {meta['source']}"
    _fact("dyn.case", meta["case"], s)
    _fact("dyn.fault_line", meta["fault_line"], s)
    _fact("dyn.fault_position", meta["fault_position"], s)
    _fact("dyn.cct_bracket_s", [meta["cct_lo"], meta["cct_hi"]], s)
    _fact("dyn.clearing_stable_s", R["stable"]["clr"], s)
    _fact("dyn.clearing_unstable_s", R["unstable"]["clr"], s)
    _fact("dyn.n_machines", int(len(crit)), s)
    _fact("dyn.n_critical", int(crit.sum()), s)
    _fact("dyn.max_rel_angle_stable_deg", round(float(np.abs(R["stable"]["rel"]).max()), 1), s)
    _fact("dyn.labels", [R["stable"]["label"], R["unstable"]["label"]], s)
    print("dyn_angles:", {k: v["value"] for k, v in FACTS.items() if k.startswith("dyn.")})
    _save(fig, out, "dyn_angles")


def fig_dyn_frequency(exp: Path, grid: Path, src: Path, out: Path) -> None:
    D = _dyn(src)
    R = D["runs"]
    fig, axes = plt.subplots(2, 1, figsize=(ACC_COL, 3.3), sharex=True)
    sty = {"stable": (C["blue"], "-"), "unstable": (C["orange"], "-")}
    for tag, r in R.items():
        col, ls = sty[tag]
        lab = f"{tag}, clearing {r['clr']:.3f} s"
        axes[0].plot(r["t"], (r["w_coi"] - 1.0) * 50e3, color=col, ls=ls, label=lab)
        axes[1].plot(r["t"], r["u"], color=col, ls=ls, label=lab)
        axes[1].axvline(r["clr"], color=col, lw=0.5, ls=":")
    axes[0].grid(True, axis="y")
    axes[1].grid(True, axis="y")
    axes[0].set_ylabel("COI speed dev. (mHz)")
    axes[1].set_ylabel("faulted-bus voltage (p.u.)")
    axes[1].set_xlabel("time after fault inception (s)")
    axes[1].set_xlim(-0.2, 6.0)
    axes[0].legend(loc="upper right", fontsize=6.5)
    axes[0].set_title("(a) centre-of-inertia speed", loc="left", fontsize=8)
    axes[1].set_title("(b) voltage at the faulted terminal", loc="left", fontsize=8)
    s = "paper_src/dyn/dyn_*.npz"
    for tag, r in R.items():
        on = (r["t"] > 0.005) & (r["t"] < r["clr"] - 0.005)
        _fact(f"dyn.{tag}.u_fault_on_min_pu", round(float(r["u"][on].min()), 3), s)
        _fact(f"dyn.{tag}.coi_dev_max_mhz", round(float(np.abs((r["w_coi"] - 1) * 50e3).max()), 1), s)
        post = r["t"] > 4.0
        _fact(f"dyn.{tag}.u_after_4s_mean_pu", round(float(r["u"][post].mean()), 3), s)
    print("dyn_frequency:", {k: v["value"] for k, v in FACTS.items() if k.startswith("dyn.") and "." in k[4:]})
    _save(fig, out, "dyn_frequency")


def fig_dyn_sime(exp: Path, grid: Path, src: Path, out: Path) -> None:
    D = _dyn(src)
    R = D["runs"]
    res_u, full_u = _sime(R["unstable"])
    res_s, full_s = _sime(R["stable"], critical=res_u.critical)
    fig, axes = plt.subplots(1, 2, figsize=(ACC_TXT, 2.4), sharey=True)
    for ax, tag, res, o, title in ((axes[0], "stable", res_s, full_s, "(a) stable"),
                                   (axes[1], "unstable", res_u, full_u, "(b) unstable")):
        r = R[tag]
        t = r["t"]
        d = np.rad2deg(o.delta)
        pm = o.pm if np.ndim(o.pm) else np.full_like(d, o.pm)
        on = (t >= 0) & (t <= r["clr"] + 1e-9)
        end = len(t) if res.kind != "unstable" else int(np.searchsorted(t, res.t_u)) + 3
        post = (t > r["clr"]) & (np.arange(len(t)) < end)
        if res.kind != "unstable":  # stable: first swing up to the return point
            k = int(np.searchsorted(t, res.t_r)) if np.isfinite(res.t_r) else len(t)
            post &= np.arange(len(t)) <= k
        ax.grid(True)
        ax.plot(d[t < 0][-1:], o.pe[t < 0][-1:], "o", color=INK, ms=3, zorder=6)
        ax.plot(d[on], o.pe[on], color=C["red"], lw=1.0, label="$P_e$, fault on")
        ax.plot(d[post], o.pe[post], color=C["blue"], lw=1.0, label="$P_e$, after clearing")
        ax.plot(d[on | post], pm[on | post], color=INK2, lw=0.8, ls="--", label="$P_m$ (pre-fault)")
        acc = (on | post) & (pm > o.pe)
        dec = post & (o.pe > pm)
        ax.fill_between(d, o.pe, pm, where=acc, color=C["orange"], alpha=0.3, lw=0,
                        interpolate=True, label="accelerating area")
        ax.fill_between(d, o.pe, pm, where=dec, color=C["aqua"], alpha=0.3, lw=0,
                        interpolate=True, label="decelerating area")
        ax.set_xlabel("OMIB angle δ (deg)")
        eta = res.eta
        if res.kind == "unstable":
            info = f"η = {eta:,.0f} MW·rad"
            ax.plot(np.rad2deg(res.delta_u), np.interp(res.delta_u, o.delta[:end], o.pe[:end]), "x",
                    color=INK, ms=4, zorder=7)
        else:  # the triangle extrapolation (2.20) of this return is not robust to the fit window
            info = f"returns at δ = {np.rad2deg(res.delta_r):.0f}°"
            ax.plot(np.rad2deg(res.delta_r), np.interp(res.t_r, t, o.pe), "x", color=INK, ms=4, zorder=7)
        ax.set_title(f"{title}: clearing {r['clr']:.3f} s, {info}", loc="left", fontsize=7.5)
        _fact(f"sime.{tag}.delta_r_deg", round(float(np.rad2deg(res.delta_r)), 1) if np.isfinite(res.delta_r) else None,
              "tranpy.conventional.sime")
        _fact(f"sime.{tag}.t_after_clearing_s", round(float((res.t_r if np.isfinite(res.t_r) else res.t_u) - r["clr"]), 3),
              "tranpy.conventional.sime")
        _fact(f"sime.{tag}.eta_mw_rad", round(float(eta), 1), "paper_src/dyn/dyn_*.npz; tranpy.conventional.sime")
        _fact(f"sime.{tag}.kind", res.kind, "tranpy.conventional.sime")
    axes[0].set_ylabel("OMIB power (MW)")
    axes[1].legend(loc="upper right", fontsize=6.3)
    _fact("sime.n_critical", int(res_u.critical.sum()), "tranpy.conventional.sime")
    _fact("sime.m_eq", round(float(res_u.m_eq), 2), "tranpy.conventional.sime")
    print("dyn_sime:", {k: v["value"] for k, v in FACTS.items() if k.startswith("sime")})
    _save(fig, out, "dyn_sime")


# --------------------------------------------------------------------------------------------
# real events: 17 May 2021 traces; sensitivity of nadir / QSS for both events
# --------------------------------------------------------------------------------------------
VARIANT_LABEL = {
    "": "base", "_ramp": "import loss as 11 steps", "_winter_peak": "winter-peak German case",
    "_HRRO0": "HR/RO wholly in NW", "_HRRO1": "HR/RO wholly in SE",
    "_fpre50": "pre-event 50.000 Hz", "_fcr_measured": "NW FCR from measured 3,280 MW",
    "_no_measures": "no interruptible load/HVDC", "_de_gap": "German inertia = 2021 A75",
    "_gen_trips": "plus published NW gen. trips", "_summer_peak": "summer-peak German case",
    "_cet_hour": "inertia hour 15 UTC", "_ce_all": "step over all border nodes",
}


def fig_event_may2021(exp: Path, grid: Path, src: Path, out: Path) -> None:
    import pandas as pd

    d = src / "real_events" / "sim" / "events_v41"
    tags = ["may_P3322", "may_P3322_summer_peak", "may_P3322_cet_hour", "may_P3322_de_gap",
            "may_P3322_ce_all"]
    fig, ax = plt.subplots(figsize=(ACC_COL, 2.4))
    ax.grid(True, axis="y")
    cols = [INK, C["blue"], C["aqua"], C["yellow"], C["violet"]]
    nad: list[float] = []
    qss: list[float] = []
    for tag, col in zip(tags, cols, strict=True):
        p = d / f"{tag}.csv"
        if not p.exists():
            print(f"event_may2021: {p} missing")
            continue
        z = pd.read_csv(p)
        mt = json.load((d / f"{tag}.json").open())["metrics"]
        t = z["t_s"].to_numpy() - mt.get("t_ref_s", 1.0)
        f = (z["f_coi_hz"].to_numpy() - mt["f_pre_hz"]) * 1e3
        nad.append(f.min())
        qss.append(f[(t >= 20) & (t <= 52)].mean())
        ax.plot(t, f, color=col, lw=1.1 if tag == "may_P3322" else 0.7,
                label=VARIANT_LABEL[tag.replace("may_P3322", "")], zorder=5 if tag == "may_P3322" else 4)
    # published (R21 via FACTS.md): pre-event 49.984 Hz, nadir 49.842 Hz at about 13 s;
    # QSS only in a figure (49.86-49.89 Hz, [VISUAL])
    ax.plot([13], [-142], marker="v", ms=6, color=C["red"], ls="none", zorder=6,
            label="ENTSO-E nadir −142 mHz, ≈13 s")
    ax.axhspan(-124, -94, xmin=0.55, color=C["red"], alpha=0.12, lw=0)
    ax.text(58, -97, "report figure QSS\n(read off, 49.86–49.89 Hz)", ha="right", va="top", fontsize=6,
            color=INK2)
    ax.set_xlim(-3, 60)
    ax.set_ylim(-320, 20)
    ax.set_xlabel("time after the unit trips (s)")
    ax.set_ylabel("Δf from pre-event (mHz)")
    ax.legend(loc="lower right", fontsize=6, ncol=1, borderaxespad=0.1, labelspacing=0.25)
    s = "paper_src/real_events/sim/events_v41/may_*.csv; FACTS.md §2"
    _fact("may.sim_qss_20_52_mhz", [round(float(v), 0) for v in qss], s)
    _fact("may.sim_nadir_range_mhz", [round(float(min(nad)), 0), round(float(max(nad)), 0)], s)
    _fact("may.published_nadir_mhz", -142, "FACTS.md §2 (49.984 − 49.842, derived)")
    print("event_may2021:", FACTS["may.sim_nadir_range_mhz"])
    _save(fig, out, "event_may2021")


def fig_event_sensitivity(exp: Path, grid: Path, src: Path, out: Path) -> None:
    import pandas as pd

    p = src / "real_events" / "sim" / "events_v41" / "events_summary.csv"
    S = pd.read_csv(p)
    S["event"] = S["tag"].str[:3]
    S["var"] = S["tag"].str.replace(r"^(jan|may)_P\d+", "", regex=True)
    S["P"] = S["P_mw"].astype(int)
    # deviations in mHz relative to each run's own pre-event frequency (sim starts at 50 Hz;
    # jan_P5800_fpre50 also; the observed values are relative to the measured pre-event value)
    S["nadir_err"] = (S["nadir_df_sim"] - S["nadir_df_obs"]) * 1e3
    may_qss_obs = -0.109  # FACTS.md §2: QSS only in R21 Fig. 7, read off as 49.86-49.89 Hz -> mid -0.109
    qss_sim = np.where(S["event"] == "jan", S["qss_df_sim"], S["qss_20_52"] - 50.0)
    qss_obs = np.where(S["event"] == "jan", S["qss_df_obs"], may_qss_obs)
    S["qss_err"] = (qss_sim - qss_obs) * 1e3
    jan = S[S.event == "jan"].copy()
    jan_base = jan[jan["var"].isin(["", "_ramp"])].sort_values(["var", "P"])
    jan_var = jan[~jan["var"].isin(["", "_ramp"])]
    may = S[S.event == "may"]
    rows = ([(f"8 Jan: {r.P:,} MW" + (", 11 steps" if r.var else ""), r, "jan") for r in jan_base.itertuples()]
            + [(f"8 Jan: 5,800 MW, {VARIANT_LABEL[r.var]}", r, "jan") for r in jan_var.itertuples()]
            + [(f"17 May: {VARIANT_LABEL[r.var]}", r, "may") for r in may.itertuples()])
    n = len(rows)
    fig, axes = plt.subplots(1, 2, figsize=(ACC_TXT, 0.13 * n + 0.75), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1]})
    y = np.arange(n)[::-1]
    for ax, col, tol_lab, title in ((axes[0], "nadir_err", "±10 % of published", "(a) nadir error"),
                                    (axes[1], "qss_err", "±10 mHz", "(b) QSS (20–52 s) error")):
        ax.grid(True, axis="x")
        ax.axvline(0, color=INK2, lw=0.6)
        for yi, (lab, r, ev) in zip(y, rows, strict=True):
            v = getattr(r, col)
            c = C["blue"] if ev == "jan" else C["orange"]
            hollow = ev == "may" and col == "qss_err"
            ax.plot([0, v], [yi, yi], color=c, lw=0.8)
            ax.plot(v, yi, "o", ms=3.8, mfc="white" if hollow else c, mec=c, mew=0.8)
        ax.set_title(title, loc="left", fontsize=8)
        ax.set_xlabel("simulated − published Δf (mHz)")
    tol_j = 0.1 * 287
    tol_m = 0.1 * 142
    axes[0].axvspan(-tol_j, tol_j, ymin=(n - len(jan) - 0.5) / n + 0.5 / n, color=C["blue"], alpha=0.08, lw=0)
    axes[0].axvspan(-tol_m, tol_m, ymax=(len(may)) / n, color=C["orange"], alpha=0.1, lw=0)
    axes[1].axvspan(-10, 10, color=GRID, alpha=0.6, lw=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lab for lab, _, _ in rows], fontsize=6.3)
    axes[0].set_ylim(-0.7, n - 0.3)
    axes[0].tick_params(axis="y", length=0)
    axes[1].text(0.02, 0.02, "17 May (hollow): vs QSS read off\nthe report figure, 49.86–49.89 Hz",
                 transform=axes[1].transAxes, ha="left", va="bottom", fontsize=6, color=INK2)
    s = "paper_src/real_events/sim/events_v41/events_summary.csv"
    _fact("sens.jan.nadir_err_mhz_range", [round(float(jan.nadir_err.min()), 0), round(float(jan.nadir_err.max()), 0)], s)
    _fact("sens.jan.qss_err_mhz_range", [round(float(jan.qss_err.min()), 0), round(float(jan.qss_err.max()), 0)], s)
    _fact("sens.jan.base_nadir_err_mhz", [round(float(v), 0) for v in jan_base.nadir_err], s)
    _fact("sens.may.nadir_err_mhz_range", [round(float(may.nadir_err.min()), 0), round(float(may.nadir_err.max()), 0)], s)
    _fact("sens.may.qss_err_vs_visual_mhz_range", [round(float(may.qss_err.min()), 0), round(float(may.qss_err.max()), 0)], s)
    for r in S.itertuples():
        _fact(f"sens.{r.tag}", {"nadir_err_mhz": round(float(r.nadir_err), 1), "qss_err_mhz": round(float(r.qss_err), 1)}, s)
    print("event_sensitivity:", {k: v["value"] for k, v in FACTS.items() if k.startswith("sens.") and "range" in k})
    _save(fig, out, "event_sensitivity")


FIGURES = {
    "instants": fig_instants, "grid_map": fig_grid_map, "event_jan2021": fig_event_jan2021,
    "instant_curve": fig_instant_curve, "prefault_rank": fig_prefault_rank, "strata": fig_strata,
    "cct_lever": fig_cct_lever, "latency": fig_latency,
    "map_cct": fig_map_cct, "map_unstable": fig_map_unstable, "map_errors": fig_map_errors,
    "map_inertia": fig_map_inertia, "dyn_angles": fig_dyn_angles, "dyn_frequency": fig_dyn_frequency,
    "dyn_sime": fig_dyn_sime, "event_may2021": fig_event_may2021,
    "event_sensitivity": fig_event_sensitivity,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, default=Path.home() / "data" / "exp")
    ap.add_argument("--grid", type=Path, default=Path.home() / "data" / "grid" / "grid_model.gpkg")
    ap.add_argument("--paper-src", type=Path, default=Path.home() / "data" / "paper_src")
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--only", default="", help="comma-separated figure names")
    a = ap.parse_args(argv)
    names = [n for n in a.only.split(",") if n] or list(FIGURES)
    failed = 0
    for n in names:
        try:
            FIGURES[n](a.exp, a.grid, a.paper_src, a.out)
        except Exception as e:
            failed += 1
            print(f"FAILED {n}: {type(e).__name__}: {e}", file=sys.stderr)
    if FACTS:  # numbers stated in the T15 captions, with their sources
        p = a.out / "figure_numbers.json"
        old = json.loads(p.read_text()) if p.exists() else {}
        old.update(FACTS)
        p.write_text(json.dumps(old, indent=1, ensure_ascii=False, default=str) + "\n")
        print(f"wrote {p}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
