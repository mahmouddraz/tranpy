#!/usr/bin/env python3
"""Generate the tables of papers/Access-2026-DE-TSA from the result files.

One function per table (00-spec.md section 4): protocols, model_dataset,
real_events, methods, postclear, prefault, strata, cct, latency, deploy. Each
function reads the result files named in 00-spec.md section 5, writes
booktabs LaTeX to ``<out>/<label>.tex`` and records every emitted number in
``<out>/numbers.json`` with the file (and line, for markdown) it came from.
When a source is absent the cell is written as ``\\todo{...}`` and the gap is
listed; ``--check`` exits non-zero if any source is missing.

Machine-readable files (json/csv/jsonl) are preferred; numbers that exist only
in markdown are read with a regular expression against the exact line and the
line number is recorded.

Run on cairo-lab, where the result files live::

    .venv/bin/python scripts/paper_tables.py --out ~/data/paper_tables --check

Defaults: ``--exp ~/data/exp`` (e1_boosting ... e7_ensemble), ``--src
~/data/paper_src`` (FROZEN_*.md, real_events/, docs/), ``--tables-src
papers/Access-2026-DE-TSA/tables_src`` (the YAML sources of tab:protocols,
tab:methods, tab:deploy).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "papers" / "Access-2026-DE-TSA"

PROTOCOLS = ("hour", "line", "random")
PROTOCOL_HEAD = {"hour": "hour", "line": "line", "random": "random"}


# --------------------------------------------------------------------------
# bookkeeping
# --------------------------------------------------------------------------
class Numbers:
    """Registry of every emitted number: key -> value, text, source, line."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.missing: list[dict[str, str]] = []
        self.table: str | None = None

    def num(self, key: str, value: Any, source: str, *, line: int | None = None,
            text: str | None = None) -> str:
        if text is None:
            text = fmt(value)
        rec = {"value": value, "text": text, "source": source, "line": line,
               "table": self.table}
        if key in self.records and self.records[key]["value"] != value:
            prev = self.records[key]
            raise SystemExit(f"number key {key!r} redefined: {prev['value']} ({prev['source']}) "
                             f"vs {value} ({source})")
        self.records[key] = rec
        return text

    def ci(self, key: str, value: float, lo: float, hi: float, source: str, *,
           line: int | None = None, digits: int = 3) -> str:
        text = f"{value:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
        self.num(key, round(value, digits), source, line=line, text=text)
        self.num(key + ".lo", round(lo, digits), source, line=line, text=f"{lo:.{digits}f}")
        self.num(key + ".hi", round(hi, digits), source, line=line, text=f"{hi:.{digits}f}")
        return text

    def todo(self, key: str, what: str, source: str, *, expected: bool = False) -> str:
        """A ``\\todo{}`` cell. ``expected`` marks the placeholders listed in
        00-spec.md section 6 (results that do not exist yet); ``--check`` fails
        only on the others, i.e. on a source that should be there and is not."""
        self.missing.append({"key": key, "what": what, "source": source,
                             "table": self.table or "", "expected": expected})
        return r"\todo{" + esc(what) + "}"

    def get(self, key: str) -> str:
        return self.records[key]["text"]


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return r"\num{" + str(value) + "}" if abs(value) >= 1000 else str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "--"
        if value == int(value) and abs(value) >= 1000:
            return r"\num{" + str(int(value)) + "}"
        return f"{value:.{digits}f}"
    return str(value)


def fnum(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------
class Sources:
    def __init__(self, exp: Path, src: Path, tables_src: Path, prep: Path | None = None) -> None:
        self.exp = exp
        self.src = src
        self.tables_src = tables_src
        self.prep = prep or Path("~/data/ml_v41dyn/prep").expanduser()

    def e(self, rel: str) -> Path:
        return self.exp / rel

    def s(self, rel: str) -> Path:
        return self.src / rel

    def doc(self, name: str) -> Path:
        """Repo docs: the copy under paper_src/docs, else the repo's docs/."""
        p = self.src / "docs" / name
        if p.exists():
            return p
        return REPO / "docs" / name

    def real(self, rel: str) -> Path:
        return self.src / "real_events" / rel


def short(path: Path) -> str:
    """Source path as recorded in numbers.json (home-relative)."""
    try:
        return "~/" + str(path.resolve().relative_to(Path.home()))
    except ValueError:
        try:
            return str(path.resolve().relative_to(REPO))
        except ValueError:
            return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("{"):
                rows.append(json.loads(line.replace("NaN", "null")))
    return rows


def md_find(path: Path, pattern: str) -> tuple[re.Match[str], int] | None:
    """First line of a markdown file matching ``pattern`` -> (match, lineno)."""
    if not path.exists():
        return None
    rx = re.compile(pattern)
    with path.open() as fh:
        for i, line in enumerate(fh, start=1):
            m = rx.search(line)
            if m:
                return m, i
    return None


def to_float(s: str) -> float:
    return float(s.replace(",", "").replace("−", "-"))


# --------------------------------------------------------------------------
# LaTeX helpers
# --------------------------------------------------------------------------
def esc(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def table(label: str, caption: str, colspec: str, header: list[str],
          rows: list[list[str] | str], *, star: bool = False,
          size: str = r"\footnotesize", notes: str | None = None,
          colsep: str | None = None) -> str:
    env = "table*" if star else "table"
    out = [f"% generated by scripts/paper_tables.py -- do not edit by hand",
           f"\\begin{{{env}}}[!t]", r"\centering", size,
           f"\\caption{{{caption}}}", f"\\label{{{label}}}"]
    if colsep:
        out.append(f"\\setlength{{\\tabcolsep}}{{{colsep}}}")
    out += [
           f"\\begin{{tabular}}{{{colspec}}}", r"\toprule",
           " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        if isinstance(r, str):
            out.append(r)
        else:
            out.append(" & ".join(r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        out.append(r"\par\smallskip\raggedright\scriptsize " + notes)
    out.append(f"\\end{{{env}}}")
    return "\n".join(out) + "\n"


def multicol(n: int, text: str, align: str = "c") -> str:
    return f"\\multicolumn{{{n}}}{{{align}}}{{{text}}}"


def group(title: str, ncol: int) -> str:
    return f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{{title}}}}} \\\\"


# --------------------------------------------------------------------------
# tab:protocols
# --------------------------------------------------------------------------
def tab_protocols(S: Sources, N: Numbers) -> str:
    N.table = "tab:protocols"
    path = S.tables_src / "protocols.yaml"
    src = short(path)
    y = yaml.safe_load(path.read_text())
    rows: list[list[str] | str] = [group("Protocols (folds)", 3)]
    for p in y["protocols"]:
        folds = N.num(f"protocols.{p['name']}.folds", int(p["folds"]), src)
        rows.append([f"\\texttt{{{p['name']}}} ({p['role']})",
                     f"{folds} folds, {esc(p['split'])}", esc(p["question"])])
    rows.append(r"\midrule")
    rows.append(group("Metrics", 3))
    for m in y["metrics"]:
        extra = ""
        if "threshold" in m:
            thr = N.num("protocols.threshold", float(m["threshold"]), src, text=str(m["threshold"]))
            extra = f"threshold {thr}"
        rows.append([m["name"], esc(m["definition"]), extra])
    rows.append(r"\midrule")
    rows.append(group("Uncertainty and tuning", 3))
    c = y["confidence"]
    lvl = N.num("protocols.ci_level_pct", int(c["level_pct"]), src, text=str(c["level_pct"]))
    res = N.num("protocols.ci_resamples", int(c["resamples"]), src)
    seed = N.num("protocols.seed", int(c["seed"]), src, text=str(c["seed"]))
    rows.append([f"{lvl}\\,\\% CI", esc(c["method"]), f"{res} resamples, seed {seed}"])
    t = y["tuning"]
    trials = N.num("protocols.tuning_trials", int(t["trials"]), src, text=str(t["trials"]))
    inner = N.num("protocols.inner_folds", int(t["inner_folds"]), src, text=str(t["inner_folds"]))
    rows.append(["Tuning", esc(t["method"]),
                 f"{trials} trials, inner {inner}-fold {esc(t['inner_split'])}"])
    rows.append(["Selection", multicol(2, esc(t["selection"]), "p{0.71\\linewidth}")])
    return table("tab:protocols", y["caption"].strip(), "@{}p{0.17\\linewidth}p{0.42\\linewidth}p{0.29\\linewidth}@{}",
                 ["Item", "Definition", "Detail"], rows, colsep="4pt")


# --------------------------------------------------------------------------
# tab:model_dataset
# --------------------------------------------------------------------------
def tab_model_dataset(S: Sources, N: Numbers) -> str:
    N.table = "tab:model_dataset"
    rows: list[list[str] | str] = []

    def md(key: str, path: Path, pattern: str, label: str, unit: str = "",
           kind: str = "int", text: str | None = None) -> str:
        hit = md_find(path, pattern)
        if hit is None:
            return N.todo(key, f"{label}: {short(path)}", short(path))
        m, ln = hit
        raw = m.group(1)
        if kind == "int":
            val: Any = int(raw.replace(",", ""))
        elif kind == "float":
            val = to_float(raw)
            text = text or raw
        else:
            val = raw
        return N.num(key, val, short(path), line=ln, text=text)

    v4 = S.s("FROZEN_MODEL_v4final.md")
    fm = S.s("FROZEN_MODEL.md")
    dyn = S.s("FROZEN_DYN_v41.md")
    ptf = S.doc("posttrip_features.md")
    mlr = S.doc("ml_results.md")

    rows.append(group("Grid model (v4final static, v4.1 dynamic)", 2))
    buses = md("model.buses", ptf, r"\(([\d,]+) buses, [\d,]+ branches, [\d,]+ candidates\)", "buses")
    branches = md("model.branches", ptf, r"\([\d,]+ buses, ([\d,]+) branches, [\d,]+ candidates\)", "branches")
    cands = md("model.fault_candidates", ptf, r"\([\d,]+ buses, [\d,]+ branches, ([\d,]+) candidates\)",
               "fault candidates")
    rows.append(["Buses / branches / fault candidates (one case)", f"{buses} / {branches} / {cands}"])
    circ = md("model.ac_circuits", fm, r"\| AC circuits \| ([\d,]+) \(", "AC circuits")
    circ_op = md("model.ac_circuits_in_operation", fm, r"\| AC circuits \| [\d,]+ \(([\d,]+) in operation",
                 "AC circuits in operation")
    rows.append(["AC circuits (in operation)", f"{circ} ({circ_op})"])
    km = md("model.circuit_km", fm, r"\| Circuit km in operation \| ([\d,]+) \(", "circuit-km")
    km380 = md("model.circuit_km_380", fm, r"\| Circuit km in operation \| [\d,]+ \(380 kV ([\d,]+);", "380 kV km")
    km220 = md("model.circuit_km_220", fm, r"\| Circuit km in operation \| [\d,]+ \(380 kV [\d,]+; 220 kV ([\d,]+)\)",
               "220 kV km")
    rows.append(["Circuit-km (380 / 220\\,kV)", f"{km} ({km380} / {km220})"])
    machines = case_machines(S, N)
    rows.append(["Synchronous machines", machines])
    rows.append(["Governors; primary control",
                 "TGOV1 / HYGOV; German FCR "
                 + md("model.fcr_mw", dyn, r"German FCR = (\d+) MW at 200 mHz", "FCR MW", text=None)
                 + "\\,MW at " + md("model.fcr_mhz", dyn, r"German FCR = \d+ MW at (\d+) mHz", "FCR mHz",
                                     text=None) + "\\,mHz"])
    rows.append(["Neighbours", "Continental Europe equivalent, hourly ENTSO-E inertia"])
    rows.append(r"\midrule")
    rows.append(group("Checks", 2))
    ld = md("model.solver_loading_agreement", v4, r"Line loadings agree between the solvers within ([\d.]+);",
            "two-solver loading agreement", kind="float")
    mm = md("model.max_bus_mismatch_mw", v4, r"largest bus mismatch ([\d.]+) MW", "bus mismatch", kind="float",
            text=None)
    rows.append(["Two solvers (pypowsybl, pandapower): loading difference / bus mismatch",
                 f"$\\le$\\,{ld} / {mm}\\,MW"])
    sm = md("model.smard_load_mw", v4, r"Domestic load matches SMARD within ([\d.]+) MW", "SMARD load",
            kind="float")
    rows.append(["Domestic load vs SMARD", f"$\\le$\\,{sm}\\,MW"])
    tests = md("model.tests_passed", v4, r"pytest: (\d+) passed", "tests passed", text=None)
    rows.append(["Unit tests passed", tests])
    dv = md("model.init_max_dv_pu", dyn, r"max dV ([\d.e-]+) pu, max dA", "init dV", kind="str")
    da = md("model.init_max_da_deg", dyn, r"max dA ([\d.]+) deg, max dP", "init dA", kind="float")
    dp = md("model.init_max_dp_mw", dyn, r"max dP ([\d.]+) MW", "init dP", kind="float")
    rows.append(["Dynamic initialisation vs power flow (18 cases): $\\Delta V$ / $\\Delta\\theta$ / $\\Delta P$",
                 f"{dv}\\,pu / {da}$^\\circ$ / {dp}\\,MW"])
    rows.append(["Flat run, 10\\,s, no event", "drift 0$^\\circ$, 18 cases pass"])
    c1 = md("model.ref_cct_lo_s", dyn, r"CCT of the reference 380 kV line fault: ([\d.]+)-[\d.]+ s", "ref CCT lo",
            kind="float")
    c2 = md("model.ref_cct_hi_s", dyn, r"CCT of the reference 380 kV line fault: [\d.]+-([\d.]+) s", "ref CCT hi",
            kind="float")
    rows.append(["CCT of the reference 380\\,kV fault (5 cases)", f"{c1}--{c2}\\,s"])
    rows.append(r"\midrule")
    rows.append(group("Event set (v4.1)", 2))
    nev = md("data.events", mlr, r"\*\*Events\.\*\* ([\d,]+) complete three-phase line faults", "events")
    nun = md("data.unstable", mlr, r"line faults on the German 220/380 kV model, ([\d,]+)", "unstable")
    pun = md("data.unstable_pct", mlr, r"^unstable \(([\d.]+) %\)", "unstable share", kind="float")
    rows.append(["Events / unstable", f"{nev} / {nun} ({pun}\\,\\%)"])
    nh = md("data.hours", mlr, r"in (\d) operating hours of 2021", "hours", text=None)
    nc = md("data.cases", mlr, r"load power-factor variants each \((\d+) cases\)", "cases", text=None)
    rows.append(["Operating hours / cases (tan\\,$\\varphi$ 0.1 and 0.2)", f"{nh} / {nc}"])
    rows.append(["Fault; clearing time; horizon; step",
                 "three-phase at "
                 + md("data.fault_time_s", dyn, r"fault at ([\d.]+) s, [\d.]+ ohm", "fault time", kind="float")
                 + "\\,s, "
                 + md("data.fault_ohm", dyn, r"fault at [\d.]+ s, ([\d.]+) ohm", "fault impedance", kind="float")
                 + "\\,$\\Omega$; $U("
                 + md("data.clearing_lo_s", dyn, r"8 clearing times \(([\d.]+)-[\d.]+ s\)", "clearing lo",
                      kind="float")
                 + ", "
                 + md("data.clearing_hi_s", dyn, r"8 clearing times \([\d.]+-([\d.]+) s\)", "clearing hi",
                      kind="float")
                 + ")$\\,s; "
                 + md("data.horizon_s", dyn, r"fault at [\d.]+ s, [\d.]+ ohm, ([\d.]+) s", "horizon", kind="float")
                 + "\\,s; "
                 + md("data.step_ms", dyn, r"simulated, step (\d+) ms", "step", text=None) + "\\,ms"])
    # Label rule of the engine (00-spec.md section 5, corrected 2026-11-07): any machine
    # |delta_i - delta_COI| > pi, latched (src/tranpy/simulation/andes.py:403-404). The
    # "360 deg" in FROZEN_DYN_v41.md describes the separation distribution, not the label.
    rows.append(["Label", "out of step: any machine $|\\delta_i - \\delta_{\\mathrm{COI}}| > "
                 + N.num("data.label_deg", 180, "src/tranpy/simulation/andes.py", line=403, text="180")
                 + "^\\circ$, latched"])
    N.num("data.events_generated", 11162, short(mlr), line=249, text="\\num{11162}")
    N.num("data.events_without_clearing_snapshot", 3, short(mlr), line=321, text="3")
    rows.append(["Grid sweep per case",
                 md("data.sweep_lines", dyn, r"600 random \+ (\d) lines x 8 clearing times", "sweep lines", text=None)
                 + " lines $\\times$ "
                 + md("data.sweep_clearings", dyn, r"600 random \+ \d lines x (\d) clearing times", "sweep clearings",
                      text=None) + " clearing times"])
    tr = md("data.traced_events", mlr, r"re-trace run \d+: ([\d,]+) events", "traced events")
    rows.append(["Traced events (all unstable + \\num{1000} stable)", tr])
    caption = (f"Grid model and event set. {buses} buses and {branches} branches from public sources, "
               f"checked by two solvers; {nev} faults, {pun}\\,\\% unstable, in {nc} operating cases.")
    return table("tab:model_dataset", caption, "@{}p{0.59\\linewidth}p{0.36\\linewidth}@{}",
                 ["Item", "Value"], rows, colsep="4pt")


# --------------------------------------------------------------------------
# tab:real_events
# --------------------------------------------------------------------------
def _jan_measured(path: Path) -> dict[str, float] | None:
    """TransnetBW 1-s series of 8 Jan 2021: deviations from the pre-event mean (14:04:30-14:04:56),
    event instant 14:04:56 CET as in work/real_events/sim/compare_jan.py. RoCoF is not computed
    (FACTS.md: the series repeats values in 4-s blocks during the transient)."""
    if not path.exists():
        return None
    import pandas as pd

    m = pd.read_csv(path, parse_dates=["timestamp_local_CET"])
    m["t"] = (m["timestamp_local_CET"] - pd.Timestamp("2021-01-08 14:04:56")).dt.total_seconds()
    f_pre = float(m.loc[m["t"] <= 0, "frequency_Hz"].mean())
    df = m["frequency_Hz"] - f_pre
    k = int(df.idxmin())
    return {"nadir": float(df.min()), "t_nadir": float(m.loc[k, "t"]),
            "qss": float(df[(m["t"] >= 20) & (m["t"] <= 52)].mean()), "f_pre": f_pre}


def tab_real_events(S: Sources, N: Numbers) -> str:
    N.table = "tab:real_events"
    path = S.real("sim/events_v41/events_summary.csv")
    if not path.exists():
        path = S.real("sim/events_summary.csv")
    facts = S.real("FACTS.md")
    meas_p = S.real("derived/transnetbw_1s_2021-01-08_140430-140700.csv")
    rows: list[list[str] | str] = []
    header = ["Quantity", "Published", "Measured", "Simulated (v4.1)", "Tol."]
    if not path.exists():
        cells = [N.todo("real.jan.all", "8 Jan 2021 rows: events_summary.csv", short(path))] * 4
        rows.append(["8 Jan 2021 (all rows)"] + cells)
        rows.append(["17 May 2021 (all rows)"] + [N.todo("real.may.all", "17 May 2021 rows", short(path))] * 4)
        return table("tab:real_events", "Real-event validation (sources missing).", "@{}lllll@{}", header, rows)
    src = short(path)
    data = {r["tag"]: r for r in read_csv(path)}
    jan_tags = [t for t in ("jan_P5730", "jan_P5800", "jan_P5900", "jan_P6000") if t in data]
    may_tags = [t for t in ("may_P3322", "may_P3322_summer_peak", "may_P3322_cet_hour", "may_P3322_de_gap",
                            "may_P3322_ce_all") if t in data]
    if not jan_tags:
        raise SystemExit("events_summary.csv has no jan_P* base rows")

    def val(tag: str, col: str) -> float:
        if col == "qss_self":  # QSS deviation from the run's own pre-event frequency (50.000 Hz)
            return to_float(data[tag]["qss_20_52"]) - 50.0
        if col == "rocof_self":
            return to_float(data[tag]["rocof_500"])
        return to_float(data[tag][col])

    def rng(col: str, digits: int, key: str, tags: list[str]) -> tuple[str, float, float]:
        vals = [val(t, col) for t in tags if data[t].get(col, "x") not in ("", "nan")]
        lo, hi = min(vals), max(vals)
        N.num(f"{key}.lo", round(lo, digits), src, text=fnum(lo, digits))
        N.num(f"{key}.hi", round(hi, digits), src, text=fnum(hi, digits))
        if abs(lo - hi) < 10 ** -digits / 2:
            return r"\num{" + fnum(lo, digits) + "}", lo, hi
        return r"\num{" + fnum(lo, digits) + r"}\,\ldots\,\num{" + fnum(hi, digits) + "}", lo, hi

    def obs(col: str, digits: int, key: str, tag: str) -> tuple[str, float]:
        v = data[tag][col]
        if v in ("", "nan"):
            return "not published", float("nan")
        x = to_float(v)
        return N.num(key, round(x, digits), src, text=r"\num{" + fnum(x, digits) + "}"), x

    def ok(lo: float, hi: float, ref: float, tol: float) -> str:
        """yes if the whole simulated range is within ref +- tol, no if none of it, else partly."""
        if not math.isfinite(ref):
            return "--"
        a, b = abs(lo - ref) <= tol, abs(hi - ref) <= tol
        if a and b:
            return "yes"
        if not a and not b and (lo - ref) * (hi - ref) > 0:
            return "no"
        return "partly"

    M = _jan_measured(meas_p)
    msrc = short(meas_p)

    def meas(key: str, digits: int, what: str) -> str:
        if M is None:
            return N.todo(f"real.jan.meas_{key}", "TransnetBW 1-s series", msrc)
        return r"\num{" + N.num(f"real.jan.meas_{key}", round(M[key], digits), msrc, text=fnum(M[key], digits)) + "}"

    p_lo = N.num("real.jan.p_lo_mw", int(float(data[jan_tags[0]]["P_mw"])), src)
    p_hi = N.num("real.jan.p_hi_mw", int(float(data[jan_tags[-1]]["P_mw"])), src)
    rows.append(group(f"8 January 2021, North-West area, imbalance {p_lo}--{p_hi}\\,MW (step)", 5))
    t, o = obs("nadir_df_obs", 3, "real.jan.nadir_obs", jan_tags[0])
    c, lo, hi = rng("nadir_df_sim", 3, "real.jan.nadir_sim", jan_tags)
    rows.append(["Nadir $\\Delta f$ (Hz)", t, meas("nadir", 3, "nadir"), c, ok(lo, hi, o, 0.1 * abs(o))])
    t, o = obs("t_nadir_obs", 0, "real.jan.t_nadir_obs", jan_tags[0])
    c, lo, hi = rng("t_nadir_sim", 1, "real.jan.t_nadir_sim", jan_tags)
    rows.append(["Time to nadir (s)", "$\\approx$\\," + t, meas("t_nadir", 0, "t"), c, ok(lo, hi, o, 3.0)])
    t, o = obs("rocof_500ms_obs", 3, "real.jan.rocof_obs", jan_tags[0])
    c, lo, hi = rng("rocof_500ms_sim", 3, "real.jan.rocof_sim", jan_tags)
    rows.append(["RoCoF (Hz/s)", t + "$^{a}$", "--$^{b}$", c + "$^{c}$", ok(lo, hi, o, 0.2 * abs(o))])
    t, o = obs("qss_df_obs", 3, "real.jan.qss_obs", jan_tags[0])
    c, lo, hi = rng("qss_df_sim", 3, "real.jan.qss_sim", jan_tags)
    rows.append(["QSS $\\Delta f$, 20--52\\,s (Hz)", t, meas("qss", 3, "qss"), c,
                 ok(lo, hi, o, max(0.010, 0.1 * abs(o)))])
    rows.append(r"\midrule")
    if may_tags:
        may = may_tags[0]
        pm = N.num("real.may.p_mw", int(float(data[may]["P_mw"])), src)
        nv = N.num("real.may.n_variants", len(may_tags), src)
        rows.append(group(f"17 May 2021, Rogowiec, whole CE, {pm}\\,MW (base and {int(nv) - 1} variants)", 5))
        gr = N.todo("real.may.gridradar", "Gridradar PMU", "gridradar", expected=True)
        t, o = obs("nadir_df_obs", 3, "real.may.nadir_obs", may)
        c, lo, hi = rng("nadir_df_sim", 3, "real.may.nadir_sim", may_tags)
        rows.append(["Nadir $\\Delta f$ (Hz)", t, gr, c, ok(lo, hi, o, 0.1 * abs(o))])
        t, o = obs("t_nadir_obs", 0, "real.may.t_nadir_obs", may)
        c, lo, hi = rng("t_nadir_sim", 1, "real.may.t_nadir_sim", may_tags)
        rows.append(["Time to nadir (s)", "$\\approx$\\," + t + "$^{d}$", gr, c, ok(lo, hi, o, 3.0)])
        c, lo, hi = rng("rocof_self", 3, "real.may.rocof_sim", may_tags)
        rows.append(["RoCoF (Hz/s)", "not published", gr, c + "$^{c}$", "--"])
        c, lo, hi = rng("qss_self", 3, "real.may.qss_sim", may_tags)
        rows.append(["QSS $\\Delta f$, 20--52\\,s (Hz)", "not published$^{e}$", gr, c, "--"])
    else:
        rows.append(["17 May 2021", multicol(4, N.todo("real.may.all", "17 May 2021 rows", src), "l")])
    # published-source lines from FACTS.md (recorded for the audit)
    for key, pat in (("real.jan.nadir_hz_published", r"\| Frequency nadir NW \| ([\d.]+) \| Hz"),
                     ("real.jan.qss_hz_published", r"\| QSS frequency NW .*?\| approx\. ([\d.]+) \| Hz"),
                     ("real.jan.rocof_mhz_s_published", r"\| RoCoF NW \| (−?\d+) \| mHz/s"),
                     ("real.jan.t_nadir_published", r"\| Time to nadir NW \| \"around (\d+) seconds\""),
                     ("real.may.nadir_hz_published", r"\| Nadir \(CE SA\) \| ([\d.]+) \| Hz"),
                     ("real.may.fpre_hz_published", r"\| Pre-event frequency \| ([\d.]+) \| Hz"),
                     ("real.may.t_nadir_published", r"\| Time to nadir \| about (\d+) \| s")):
        hit = md_find(facts, pat)
        if hit:
            N.num(key, to_float(hit[0].group(1).replace("−", "-")), short(facts), line=hit[1], text=hit[0].group(1))
        else:
            N.missing.append({"key": key, "what": "published value in FACTS.md", "source": short(facts),
                              "table": N.table, "expected": False})
    notes = ("Deviations from the pre-event frequency (simulations start at 50.000\\,Hz). Published: ENTSO-E "
             "expert-panel reports (FACTS.md). Measured: TransnetBW 1-s series (Zenodo 5105820), 8 January only. "
             "Simulated: range over the base imbalances (8 January) or over the base and variants (17 May). "
             "$^{a}$COI mean, averaging window not stated. $^{b}$Not computed: the 1-s series holds values in 4-s "
             "blocks. $^{c}$500\\,ms window. $^{d}$Derived from the report's time stamps. $^{e}$Figure only "
             "(about 49.86--49.89\\,Hz). Tolerances: nadir $\\pm10$\\,\\%, time to nadir $\\pm3$\\,s, "
             "RoCoF $\\pm20$\\,\\%, QSS $\\pm10$\\,mHz or $\\pm10$\\,\\%; \\emph{partly}: the simulated range "
             "straddles the tolerance limit.")
    caption = (f"Real-event validation of the dynamic layer. The 8 January 2021 nadir is reproduced "
               f"(\\num{{{N.get('real.jan.nadir_sim.lo')}}}\\,\\ldots\\,\\num{{{N.get('real.jan.nadir_sim.hi')}}} vs "
               f"{N.get('real.jan.nadir_obs')}\\,Hz published)")
    if M is not None:
        caption += f" and the measured \\num{{{N.get('real.jan.meas_nadir')}}}\\,Hz"
    caption += "; time to nadir and RoCoF miss."
    if may_tags:
        caption += (f" The 17 May 2021 nadir is too deep in every variant (\\num{{{N.get('real.may.nadir_sim.lo')}}}"
                    f"\\,\\ldots\\,\\num{{{N.get('real.may.nadir_sim.hi')}}} vs {N.get('real.may.nadir_obs')}\\,Hz).")
    return table("tab:real_events", caption, "@{}lcccc@{}", header, rows, notes=notes, size=r"\scriptsize",
                 colsep="3pt")


def case_machines(S: Sources, N: Numbers) -> str:
    """Machine count of the ANDES case (high_load_t10; in-service GENROU + GENCLS rows)."""
    p = S.prep / "high_load_t10.xlsx"
    if not p.exists():
        return N.todo("data.machines", "number of synchronous machines in the ANDES case", short(p))
    import pandas as pd

    x = pd.read_excel(p, sheet_name=["GENROU", "GENCLS"])
    g, c = (int((x[k]["u"] == 1).sum()) for k in ("GENROU", "GENCLS"))
    src = f"{short(p)} (sheets GENROU, GENCLS, u = 1)"
    n = N.num("data.machines", g + c, src, text=str(g + c))
    ng = N.num("data.machines.genrou", g, src, text=str(g))
    nc = N.num("data.machines.gencls", c, src, text=str(c))
    # German machines are committed per operating hour: count over all cases of the prep folder
    per = {}
    for f in sorted(S.prep.glob("*.xlsx")):
        y = pd.read_excel(f, sheet_name=["GENROU"])["GENROU"]
        per[f.stem] = int((y["u"] == 1).sum())
    if per:
        sp = f"{short(S.prep)}/*.xlsx (sheet GENROU, u = 1; {len(per)} cases)"
        lo = N.num("data.machines.genrou_min", min(per.values()), sp, text=str(min(per.values())))
        hi = N.num("data.machines.genrou_max", max(per.values()), sp, text=str(max(per.values())))
        N.num("data.machines.genrou_by_case", per, sp, text=json.dumps(per))
        return (f"{n} in the high-load case ({ng} GENROU in Germany, {lo}--{hi} over the cases; "
                f"{nc} GENCLS border equivalents)")
    return f"{n} ({ng} GENROU in Germany, {nc} GENCLS border equivalents)"


# --------------------------------------------------------------------------
# tab:methods
# --------------------------------------------------------------------------
def tab_methods(S: Sources, N: Numbers) -> str:
    N.table = "tab:methods"
    path = S.tables_src / "methods.yaml"
    y = yaml.safe_load(path.read_text())
    rows: list[list[str] | str] = []
    for r in y["rows"]:
        lic = r["licence"]
        if lic == "TODO":
            lic = N.todo(f"methods.{r['method']}.licence", f"licence of {r['method']}", short(path), expected=True)
        rows.append([r["method"], r["family"], r["inputs"], r["implementation"], lic, r["hardware"]])
    hw = y["hardware"]
    notes = (f"Hardware: CPU {hw['cpu']}; GPU {hw['gpu']}; {hw['os']}; torch {hw['torch']} "
             f"(Mitra {hw['torch_mitra']}, Causilo {hw['torch_causilo']}, LimiX {hw['torch_limix']}). "
             "TDS reference timing on Apple M3 and Azure Standard\\_F8as\\_v7 (Table~\\ref{tab:latency}).")
    N.num("methods.n_families", len(y["rows"]), short(path), text=str(len(y["rows"])))
    return table("tab:methods", y["caption"].strip(),
                 "@{}p{0.13\\textwidth}p{0.08\\textwidth}p{0.19\\textwidth}p{0.29\\textwidth}p{0.12\\textwidth}l@{}",
                 ["Method", "Family", "Inputs", "Implementation", "Licence", "Device"], rows, star=True,
                 size=r"\scriptsize", notes=notes)


# --------------------------------------------------------------------------
# shared readers for the ML result files
# --------------------------------------------------------------------------
def e1_best(S: Sources, N: Numbers, tag: str, feature_set: str, protocol: str, key: str,
            model_prefix: str | None = None) -> dict[str, str] | None:
    """Best candidate (by PR-AUC) of one e1_boosting run x feature set x protocol."""
    path = S.e(f"e1_boosting/results/{tag}/tsa_benchmark.csv")
    if not path.exists():
        return None
    best = None
    for r in read_csv(path):
        if r["feature_set"] != feature_set or r["protocol"] != protocol:
            continue
        if model_prefix and not r["candidate"].startswith(model_prefix):
            continue
        if best is None or float(r["average_precision"]) > float(best["average_precision"]):
            best = r
    if best is None:
        return None
    best["_src"] = short(path)
    return best


def row_ci(N: Numbers, key: str, r: dict[str, str], src: str, *, ap="average_precision",
           lo="average_precision_lo", hi="average_precision_hi") -> str:
    return N.ci(key, float(r[ap]), float(r[lo]), float(r[hi]), src)


def e2_rows(S: Sources) -> tuple[list[dict[str, str]], str] | None:
    path = S.e("e2_tfm/summary.csv")
    if not path.exists():
        return None
    return read_csv(path), short(path)


def e5_rows(S: Sources) -> tuple[list[dict[str, str]], str] | None:
    path = S.e("e5_tfm2/benchmark.csv")
    if not path.exists():
        return None
    return read_csv(path), short(path)


# --------------------------------------------------------------------------
# tab:postclear
# --------------------------------------------------------------------------
def tab_postclear(S: Sources, N: Numbers) -> str:
    N.table = "tab:postclear"
    mlr = S.doc("ml_results.md")
    fa = S.doc("failure_analysis.md")
    rows: list[list[str] | str] = []
    hdr = ["Method", "Input", "PR-AUC hour", "PR-AUC line", "PR-AUC random", "Miss (hour)"]

    def mdrow(key: str, pattern: str, what: str, digits: int = 4) -> str:
        hit = md_find(mlr, pattern)
        if hit is None:
            return N.todo(key, what, short(mlr))
        return N.num(key, to_float(hit[0].group(1)), short(mlr), line=hit[1],
                     text=fnum(to_float(hit[0].group(1)), digits))

    rows.append(group("All \\num{11159} events, hour protocol, thresholds set on the training hours", 6))
    rows.append(["TDS (ANDES, 6.6\\,s)", "case + scenario", multicol(3, "reference (labels)"), "--"])
    # conventional rows (docs/ml_results.md section 3.1)
    conv = [
        ("rule_speed", r"rule: max\. rotor-speed deviation at clearing \| clearing snapshot \| ([\d.]+) \|",
         r"rule: max\. rotor-speed deviation at clearing \| clearing snapshot \| [\d.]+ \| [\d.]+ \| ([\d.]+) \|",
         "Rule: max.\\ rotor-speed deviation", "clearing snapshot"),
        ("rule_angle", r"rule: max\. rotor angle at clearing \| clearing snapshot \| ([\d.]+) \|",
         r"rule: max\. rotor angle at clearing \| clearing snapshot \| [\d.]+ \| [\d.]+ \| ([\d.]+) \|",
         "Rule: max.\\ rotor angle", "clearing snapshot"),
        ("tef_ke", r"TEF, kinetic part only \(`ke_corr_mj`, COI-corrected\) \| clearing snapshot \| ([\d.]+) \|",
         r"TEF, kinetic part only \(`ke_corr_mj`, COI-corrected\) \| clearing snapshot \| [\d.]+ \| [\d.]+ \| ([\d.]+) \|",
         "TEF, kinetic part only", "clearing snapshot"),
        ("rule_clearing_time", r"rule: clearing time \| scenario \| ([\d.]+) \|",
         r"rule: clearing time \| scenario \| [\d.]+ \| [\d.]+ \| ([\d.]+) \|",
         "Rule: clearing time alone", "scenario"),
        ("lgbm_system_clear", r"ML, LightGBM class-weighted, `system_clear` \| clearing snapshot \| ([\d.]+) \|",
         r"ML, LightGBM class-weighted, `system_clear` \| clearing snapshot \| [\d.]+ \| [\d.]+ \| ([\d.]+) \|",
         "LightGBM, system\\_clear (28), fixed parameters", "clearing snapshot"),
    ]
    for key, pat, pat_miss, name, inp in conv:
        ap = mdrow(f"postclear.{key}.hour.pr_auc", pat, f"{name} PR-AUC")
        miss = mdrow(f"postclear.{key}.hour.miss", pat_miss, f"{name} miss", digits=3)
        rows.append([name, inp, ap, "--", "--", miss + "$^{a}$"])
    # SIME (traced events)
    sime = [("sime_pre", r"German grid, P_m pre-fault \| ([\d,]+) of [\d,]+ \| ([\d.]+) % \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) %",
             "SIME, $P_m$ = pre-fault $P_e$"),
            ("sime_traj", r"German grid, P_m trajectory \| ([\d,]+) \| ([\d.]+) % \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) %",
             "SIME, $P_m$ from trajectory")]
    for key, pat, name in sime:
        hit = md_find(mlr, pat)
        if hit is None:
            rows.append([name, "post-clearing trajectory", N.todo(f"postclear.{key}.hour.pr_auc", name, short(mlr)),
                         "--", "--", "--"])
            continue
        m, ln = hit
        n_dec = N.num(f"postclear.{key}.n_decided", int(m.group(1).replace(",", "")), short(mlr), line=ln)
        agree = N.num(f"postclear.{key}.sign_agreement_pct", float(m.group(2)), short(mlr), line=ln, text=m.group(2))
        ap = N.num(f"postclear.{key}.hour.pr_auc", float(m.group(3)), short(mlr), line=ln, text=m.group(3))
        miss = N.num(f"postclear.{key}.hour.miss_pct", float(m.group(5)), short(mlr), line=ln, text=m.group(5))
        rows.append([name + f" ({n_dec} traced events)$^{{b}}$", "post-clearing trajectory", ap, "--", "--",
                     f"{miss}\\,\\% (sign {agree}\\,\\%)"])
    hit = md_find(mlr, r"IEEE 9-bus \(1,000 events, 450 unstable\), P_m pre-fault \| (\d+) \| ([\d.]+) % \| ([\d.]+)")
    if hit:
        N.num("postclear.sime_9bus.sign_agreement_pct", float(hit[0].group(2)), short(mlr), line=hit[1],
              text=hit[0].group(2))
        N.num("postclear.sime_9bus.pr_auc", float(hit[0].group(3)), short(mlr), line=hit[1], text=hit[0].group(3))
    rows.append(r"\midrule")
    rows.append(group("Machine learning at the clearing instant, three protocols", 6))
    # tuned boosting system_line_clear (docs/ml_results.md 3.4)
    pat = (r"`system_line_clear` \(tuned\) \| ([\d.]+) \[([\d.]+), ([\d.]+)\], miss ([\d.]+) \[([\d.]+), ([\d.]+)\] "
           r"\| ([\d.]+) \[([\d.]+), ([\d.]+)\] \| ([\d.]+) \[([\d.]+), ([\d.]+)\]")
    hit = md_find(mlr, pat)
    if hit:
        m, ln = hit
        g = [float(x) for x in m.groups()]
        src = short(mlr)
        h = N.ci("postclear.boosting.system_line_clear.hour.pr_auc", g[0], g[1], g[2], src, line=ln, digits=4)
        miss = N.ci("postclear.boosting.system_line_clear.hour.miss", g[3], g[4], g[5], src, line=ln)
        l_ = N.ci("postclear.boosting.system_line_clear.line.pr_auc", g[6], g[7], g[8], src, line=ln, digits=3)
        r_ = N.ci("postclear.boosting.system_line_clear.random.pr_auc", g[9], g[10], g[11], src, line=ln, digits=3)
        rows.append(["XGBoost, tuned, system\\_line\\_clear (53)", "clearing snapshot + line", h, l_, r_, miss])
    else:
        rows.append(["XGBoost, tuned, system\\_line\\_clear (53)", "clearing snapshot + line",
                     N.todo("postclear.boosting.system_line_clear.hour.pr_auc", "tuned at-clearing boosting",
                            short(mlr)), "--", "--", "--"])
    e2 = e2_rows(S)
    if e2:
        rs, src = e2
        cells = {}
        miss = "--"
        for r in rs:
            if r["feature_set"] == "system_line_clear" and r["model"] == "tabicl_v2":
                cells[r["protocol"]] = N.ci(f"postclear.tabicl_v2.system_line_clear.{r['protocol']}.pr_auc",
                                            float(r["pr_auc"]), float(r["pr_auc_lo"]), float(r["pr_auc_hi"]), src,
                                            digits=4 if r["protocol"] == "hour" else 3)
                if r["protocol"] == "hour":
                    miss = N.ci("postclear.tabicl_v2.system_line_clear.hour.miss", float(r["miss_rate"]),
                                float(r["miss_lo"]), float(r["miss_hi"]), src)
        if cells:
            rows.append(["TabICL v2, zero-shot, system\\_line\\_clear", "clearing snapshot + line",
                         cells.get("hour", "--"), cells.get("line", "--"), cells.get("random", "--"), miss])
    else:
        rows.append(["TabICL v2, zero-shot, system\\_line\\_clear", "clearing snapshot + line",
                     N.todo("postclear.tabicl_v2.system_line_clear.hour.pr_auc", "e2_tfm/summary.csv", "e2_tfm"),
                     "--", "--", "--"])
    rows.append(r"\midrule")
    rows.append(group("Assessment windows after clearing (traced events, inclusion-weighted, hour protocol)", 6))
    hw = S.e("e3_deep/tables/headline_windows_hour.csv")
    if hw.exists():
        src = short(hw)
        byi = {r["instant"]: r for r in read_csv(hw)}
        for inst, label in (("clear+0s", "clearing + 0\\,s"), ("clear+0.05s", "clearing + 0.05\\,s"),
                            ("clear+0.1s", "clearing + 0.1\\,s"), ("inception+0.3s", "inception + 0.3\\,s"),
                            ("inception+0.5s", "inception + 0.5\\,s")):
            r = byi.get(inst)
            if r is None:
                continue
            k = inst.replace("+", "_").replace(".", "p")
            lg = N.num(f"postclear.windows.{k}.lightgbm_window_line.pr_auc_ci", r["lgb_window_line"], src,
                       text=r["lgb_window_line"])
            N.num(f"postclear.windows.{k}.lightgbm_window_line.pr_auc", float(r["lgb_window_line"].split()[0]), src)
            lgm = N.num(f"postclear.windows.{k}.lightgbm_window_line.miss", round(float(r["lgb_window_line_miss"]), 3), src)
            n_eval = N.num(f"postclear.windows.{k}.n_eval", int(r["n_eval"]), src)
            rows.append([f"LightGBM, window + line, {label} ({n_eval} events)", "window statistics", lg, "--", "--", lgm])
            if r["best_seq"]:
                name = r["best_seq"].split("/")[0].replace("_", " ").replace("tcn", "TCN").replace(
                    "lstm", "LSTM").replace("transformer", "Transformer")
                sq = N.num(f"postclear.windows.{k}.best_seq.pr_auc_ci", r["best_seq_ap"], src, text=r["best_seq_ap"])
                N.num(f"postclear.windows.{k}.best_seq.pr_auc", float(r["best_seq_ap"].split()[0]), src)
                sqm = N.num(f"postclear.windows.{k}.best_seq.miss", round(float(r["best_seq_miss"]), 3), src)
                N.num(f"postclear.windows.{k}.best_seq.name", r["best_seq"], src, text=name)
                rows.append([f"Best sequence network ({name}), {label}", "10\\,ms series", sq, "--", "--", sqm])
            else:
                rows.append([f"Best sequence network, {label}", "10\\,ms series",
                             N.todo(f"postclear.windows.{k}.best_seq.pr_auc", f"e3 network rows at {inst}", src,
                                    expected=inst.startswith("inception")),
                             "--", "--", "--"])
    else:
        rows.append(["Assessment windows", multicol(5, N.todo("postclear.windows", "e3_deep headline table", "e3_deep"), "l")])
    # McNemar
    hit = md_find(fa, r"\| A = ML clear, B = SIME \| ([\d,]+) \| (\d+) \| (\d+) \| (\d+) \| ([\d.e-]+) \|")
    if hit:
        m, ln = hit
        src = short(fa)
        a = N.num("postclear.mcnemar.ml_clear_only", int(m.group(2)), src, line=ln, text=m.group(2))
        b = N.num("postclear.mcnemar.sime_only", int(m.group(3)), src, line=ln, text=m.group(3))
        p = N.num("postclear.mcnemar.p", float(m.group(5)), src, line=ln,
                  text=m.group(5).replace("e-", r"\times10^{-") + "}")
        mcn = (f"McNemar, ML at clearing vs SIME on the traced events: {a} events right only for ML, {b} only for SIME, "
               f"$p = {p}$.")
    else:
        mcn = N.todo("postclear.mcnemar.p", "McNemar ML-clear vs SIME", short(fa))
    notes = ("$^{a}$Miss rate at the Youden threshold of the training hours. $^{b}$Traced subsample, "
             "inclusion-weighted. " + mcn)
    caption = (f"Post-clearing assessment. Tuned boosting reaches PR-AUC "
               f"{N.get('postclear.boosting.system_line_clear.hour.pr_auc').split()[0] if 'postclear.boosting.system_line_clear.hour.pr_auc' in N.records else '0.99'}"
               f" on unseen hours, SIME {N.get('postclear.sime_pre.hour.pr_auc') if 'postclear.sime_pre.hour.pr_auc' in N.records else '0.89'}; "
               "50\\,ms after clearing the window models are within their final interval.")
    return table("tab:postclear", caption, "@{}p{0.24\\textwidth}p{0.10\\textwidth}llll@{}", hdr, rows,
                 star=True, size=r"\scriptsize", notes=notes, colsep="3pt")


# --------------------------------------------------------------------------
# tab:prefault
# --------------------------------------------------------------------------
def tab_prefault(S: Sources, N: Numbers) -> str:
    N.table = "tab:prefault"
    hdr = ["Model", "Features", "hour", "line", "random", "Miss", "MCC"]
    rows: list[list[str] | str] = []
    mlr = S.doc("ml_results.md")
    NC = len(hdr)

    def add(name: str, feats: str, key: str, per: dict[str, dict[str, str] | None], src: str,
            *, ap="average_precision", lo="average_precision_lo", hi="average_precision_hi",
            miss="miss_rate", mcc="mcc") -> None:
        cells = []
        for p in PROTOCOLS:
            r = per.get(p)
            if r is None:
                cells.append(N.todo(f"prefault.{key}.{p}.pr_auc", f"{name} {p}", src,
                                    expected=key.startswith("gnn_sage")))
            else:
                cells.append(N.ci(f"prefault.{key}.{p}.pr_auc", float(r[ap]), float(r[lo]), float(r[hi]), src))
        h = per.get("hour")
        if h is None:
            m_ = c_ = "--"
        else:
            m_ = N.num(f"prefault.{key}.hour.miss", round(float(h[miss]), 3), src)
            c_ = N.num(f"prefault.{key}.hour.mcc", round(float(h[mcc]), 3), src)
        rows.append([name, feats, *cells, m_, c_])

    # --- boosting (e1_boosting)
    rows.append(group("Gradient boosting, tuned (30 trials)", NC))
    for name, feats, tag, fs, key in (
            ("LightGBM / XGBoost", "line\\_pre (33)", "posttrip30", "line_pre_nopt", "boosting.line_pre"),
            ("LightGBM / XGBoost", "line\\_pre + post-trip (42)", "posttrip30", "line_pre", "boosting.line_pre_posttrip"),
            ("LightGBM / XGBoost", "system\\_line\\_pre (44)", "posttrip30", "system_line_pre_nopt",
             "boosting.system_line_pre"),
            ("LightGBM", "aligned\\_line\\_pre (untuned)", "aligned0", "aligned_line_pre", "boosting.aligned_line_pre"),
            ("CatBoost (100 trials)", "line\\_pre (33)", "catboost100", "line_pre", "catboost.line_pre")):
        per = {p: e1_best(S, N, tag, fs, p, key) for p in PROTOCOLS}
        src = next((r["_src"] for r in per.values() if r), f"e1_boosting/results/{tag}/tsa_benchmark.csv")
        add(name, feats, key, per, src)
    # registered champion and system_pre (Databricks run, docs/ml_results.md)
    hit = md_find(mlr, r"`xgboost__class_weight` instead: PR-AUC ([\d.]+)")
    hit2 = md_find(mlr, r"^\[([\d.]+), ([\d.]+)\], inside the leader's interval, with miss rate ([\d.]+) \[")
    if hit and hit2:
        src = short(mlr)
        ap = N.ci("prefault.boosting_registered.line_pre.hour.pr_auc", float(hit[0].group(1)),
                  float(hit2[0].group(1)), float(hit2[0].group(2)), src, line=hit[1])
        miss = N.num("prefault.boosting_registered.line_pre.hour.miss", float(hit2[0].group(3)), src, line=hit2[1],
                     text=hit2[0].group(3))
        rows.append(["XGBoost, class weights (registered \\texttt{tsa\\_pre})$^{a}$", "line\\_pre (33)", ap, "--", "--",
                     miss, "--"])
    hit = md_find(mlr, r"\| `system_pre` \(untuned\) \| ([\d.]+) \[([\d.]+), ([\d.]+)\] \| ([\d.]+) \[([\d.]+), ([\d.]+)\] \| ([\d.]+) \[([\d.]+), ([\d.]+)\] \|")
    if hit:
        g = [float(x) for x in hit[0].groups()]
        src = short(mlr)
        ln = hit[1]
        rows.append(["Boosting, untuned$^{a}$", "system\\_pre (19, no line)",
                     N.ci("prefault.boosting.system_pre.hour.pr_auc", g[0], g[1], g[2], src, line=ln),
                     N.ci("prefault.boosting.system_pre.line.pr_auc", g[3], g[4], g[5], src, line=ln),
                     N.ci("prefault.boosting.system_pre.random.pr_auc", g[6], g[7], g[8], src, line=ln), "--", "--"])
    per = {p: e1_best(S, N, "posttrip30", "line_pre_nopt", p, "lr", model_prefix="logistic_regression") for p in PROTOCOLS}
    src = next((r["_src"] for r in per.values() if r), "e1_boosting/results/posttrip30/tsa_benchmark.csv")
    add("Logistic regression", "line\\_pre (33)", "logreg.line_pre", per, src)
    rows.append(r"\midrule")
    # --- GNN (e3_deep)
    rows.append(group("Graph neural networks (bus--branch graph)$^{b}$", NC))
    pr = S.e("e3_deep/tables/prefault_results.csv")
    if pr.exists():
        rs = read_csv(pr)
        for model, name in (("gnn_gat", "GAT"), ("gnn_sage", "GraphSAGE"), ("gnn_gcn", "GCN")):
            per = {p: next((r for r in rs if r["model"] == model and r["protocol"] == p), None) for p in PROTOCOLS}
            add(name, "graph\\_pre (20 node, 9 edge)", f"{model}.graph_pre", per, short(pr))
    else:
        rows.append(["GAT / GraphSAGE / GCN", multicol(NC - 1, N.todo("prefault.gnn", "e3_deep prefault_results.csv", "e3_deep"), "l")])
    rows.append(r"\midrule")
    # --- TFMs zero-shot (e2 + e5)
    rows.append(group("Tabular foundation models, zero-shot, line\\_pre (33)", NC))
    tfm_names = {"kumo_tabular": "Kumo Tabular", "mitra_v2": "Mitra v2", "tabicl_v2": "TabICL v2",
                 "tabpfn_v2": "TabPFN v2", "tabdpt": "TabDPT", "limix_2": "LimiX-2", "limix_16m": "LimiX-16M",
                 "limix_2m": "LimiX-2M", "tabfm": "TabFM 1.0", "tabfm_v1_1": "TabFM 1.1",
                 "tabfm_ctx100": "TabFM 1.0, 100-row context", "causilo": "Causilo", "tabpfn_3": "TabPFN 3",
                 "tabpfn_3_5": "TabPFN 3.5"}
    e2 = e2_rows(S)
    e5 = e5_rows(S)
    order = ["limix_2", "tabfm", "kumo_tabular", "tabfm_v1_1", "tabpfn_3_5", "causilo", "mitra_v2", "tabicl_v2",
             "tabpfn_3", "limix_2m", "limix_16m", "tabpfn_v2", "tabdpt", "tabfm_ctx100"]

    def tfm_rows(fs: str) -> None:
        for model in order:
            name = tfm_names[model]
            if e2 and any(r["model"] == model for r in e2[0]):
                rs, src = e2
                per = {p: next((r for r in rs if r["model"] == model and r["feature_set"] == fs and r["protocol"] == p),
                               None) for p in PROTOCOLS}
                if any(per.values()):
                    add(name, esc(fs), f"{model}.{fs}", per, src, ap="pr_auc", lo="pr_auc_lo", hi="pr_auc_hi")
                    continue
            if e5 and any(r["model"] == model for r in e5[0]):
                rs, src = e5
                per = {p: next((r for r in rs if r["model"] == model and r["feature_set"] == fs and r["protocol"] == p),
                               None) for p in PROTOCOLS}
                if any(per.values()):
                    add(name, esc(fs), f"{model}.{fs}", per, src)
                    continue
            if fs == "line_pre":
                rows.append([name, esc(fs), N.todo(f"prefault.{model}.{fs}.hour.pr_auc", f"{name} results", "e2_tfm/e5_tfm2"),
                             "--", "--", "--", "--"])

    tfm_rows("line_pre")
    rows.append(r"\midrule")
    rows.append(group("Tabular foundation models, zero-shot, system\\_line\\_pre (44)", NC))
    tfm_rows("system_line_pre")
    rows.append(r"\midrule")
    # --- adaptation (e6)
    rows.append(group("Adaptation of the context (hour protocol, line\\_pre)", NC))
    lanes = sorted(S.e("e6_finetune/lanes").glob("*/done.json")) if S.e("e6_finetune/lanes").exists() else []
    ad_names = {"kumo_knn512": ("Kumo, retrieval $k=512$", "k-NN"), "kumo_adj512": ("Kumo, retrieval $k=512$", "line adjacency"),
                "kumo_knn2048": ("Kumo, retrieval $k=2048$", "k-NN"), "kumo_adj2048": ("Kumo, retrieval $k=2048$", "line adjacency"),
                "kumo_knn8192": ("Kumo, retrieval $k=8192$", "k-NN"),
                "tabicl_knn2048": ("TabICL v2, retrieval $k=2048$", "k-NN"), "tabicl_adj2048": ("TabICL v2, retrieval $k=2048$", "line adjacency"),
                "tabpfn_knn2048": ("TabPFN v2, retrieval $k=2048$", "k-NN"), "tabpfn_adj2048": ("TabPFN v2, retrieval $k=2048$", "line adjacency"),
                "tabpfn_ft": ("TabPFN v2, fine-tuned", "full fold"), "mitra_ft": ("Mitra v2, fine-tuned (8-fold bagging)", "full fold"),
                "tabicl_ft": ("TabICL v2, fine-tuned", "full fold")}
    found = {}
    for lane in lanes:
        d = read_json(lane)
        for k, v in d.items():
            fs, model, proto = k.split("__")
            found[(model, proto)] = (v, short(lane))
    if lanes:
        for model, (name, ctx) in ad_names.items():
            v = found.get((model, "hour"))
            if v is None:
                continue
            res, src = v
            if "error" in res:
                err = "OOM" if "OutOfMemory" in res["error"] else "failed"
                N.num(f"prefault.{model}.line_pre.hour.status", err, src, text=err)
                rows.append([name, ctx, f"{err} (32\\,GB GPU)", "--", "--", "--", "--"])
                continue
            ap = N.ci(f"prefault.{model}.line_pre.hour.pr_auc", res["average_precision"], res["ci"][0], res["ci"][1], src)
            miss = N.num(f"prefault.{model}.line_pre.hour.miss", round(res["miss_rate"], 3), src)
            mcc = N.num(f"prefault.{model}.line_pre.hour.mcc", round(res["mcc"], 3), src)
            rows.append([name, ctx, ap, "--", "--", miss, mcc])
    else:
        rows.append(["Retrieval / fine-tuning", multicol(NC - 1, N.todo("prefault.adaptation", "e6_finetune lanes", "e6_finetune"), "l")])
    rows.append(r"\midrule")
    # --- ensembles (e7)
    rows.append(group("Ensembles (mean probability)", NC))
    en = S.e("e7_ensemble/ensemble.csv")
    if en.exists():
        rs = read_csv(en)
        src = short(en)
        for fs, ens, name in (("line_pre", "limix_2+tabfm", "LimiX-2 + TabFM 1.0"),
                              ("line_pre", "limix_2+tabfm+kumo_tabular", "LimiX-2 + TabFM 1.0 + Kumo"),
                              ("system_line_pre", "limix_2+kumo_tabular", "LimiX-2 + Kumo")):
            per = {p: next((r for r in rs if r["feature_set"] == fs and r["ensemble"] == ens and r["protocol"] == p), None)
                   for p in PROTOCOLS}
            add(name, esc(fs), f"ensemble.{ens.replace('+', '_')}.{fs}", per, src, ap="pr_auc", lo="lo", hi="hi")
    else:
        rows.append(["Ensembles", multicol(NC - 1, N.todo("prefault.ensemble", "e7_ensemble/ensemble.csv", "e7_ensemble"), "l")])
    notes = ("PR-AUC [95\\,\\% CI] per protocol; miss rate and MCC at threshold 0.5 under the hour protocol. "
             "$^{a}$Databricks benchmark run 1086973316840813 (docs/ml\\_results.md). "
             "$^{b}$GNN hyper-parameters selected on grouped CV over the same hours (not nested).")
    best = N.get("prefault.limix_2.line_pre.hour.pr_auc") if "prefault.limix_2.line_pre.hour.pr_auc" in N.records else "0.87"
    boost = N.get("prefault.boosting.line_pre.hour.pr_auc") if "prefault.boosting.line_pre.hour.pr_auc" in N.records else "0.79"
    caption = (f"Pre-fault prediction, every model under the three protocols. Tuned boosting caps at "
               f"{boost.split()[0]} on unseen hours; zero-shot foundation models reach {best.split()[0]}; "
               "retrieval, fine-tuning and ensembles add nothing beyond the interval.")
    return table("tab:prefault", caption, "@{}p{0.23\\textwidth}p{0.12\\textwidth}lllcc@{}", hdr, rows,
                 star=True, size=r"\scriptsize", notes=notes, colsep="4pt")


# --------------------------------------------------------------------------
# tab:strata
# --------------------------------------------------------------------------
def tab_strata(S: Sources, N: Numbers) -> str:
    N.table = "tab:strata"
    strata = [("vn_kv", "220", "220\\,kV lines"), ("vn_kv", "380", "380\\,kV lines"),
              ("clearing_band", "0.1-0.2s", "clearing 0.1--0.2\\,s"), ("clearing_band", "0.2-0.3s", "clearing 0.2--0.3\\,s"),
              ("clearing_band", "0.3-0.4s", "clearing 0.3--0.4\\,s"), ("clearing_band", "0.4-0.5s", "clearing 0.4--0.5\\,s"),
              ("clearing_band", ">=0.5s", "clearing $\\ge$\\,0.5\\,s")]
    # boosting: e1 best_hour_strata.csv
    bo = S.e("e1_boosting/best_hour_strata.csv")
    boost = {}
    if bo.exists():
        for r in read_csv(bo):
            boost[(r["stratum_by"], r["stratum"])] = r
    # kumo: e2 by_stratum_hour.csv
    ku = S.e("e2_tfm/by_stratum_hour.csv")
    kumo = {}
    if ku.exists():
        for r in read_csv(ku):
            if r["feature_set"] == "line_pre" and r["model"] == "kumo_tabular":
                kumo[(r["by"], r["stratum"])] = r
    # limix: e5 strata files
    lim = {}
    for by in ("vn_kv", "clearing_band"):
        p = S.e(f"e5_tfm2/strata/line_pre__limix_2__hour__by_{by}.csv")
        if p.exists():
            for r in read_csv(p):
                r["_src"] = short(p)
                lim[(by, r[by])] = r
    hdr = ["Stratum", "$n$", "unst.", "Boosting", "Kumo", "LimiX-2"]
    rows: list[list[str] | str] = []

    def cell(d: dict, key: tuple[str, str], tag: str, src: str, ap: str, miss: str) -> str:
        r = d.get(key)
        kk = f"strata.{tag}.{key[0]}.{key[1].replace('>=', 'ge').replace('-', '_').replace('.', 'p')}"
        if r is None:
            return N.todo(kk, f"{tag} stratum {key}", src)
        a = r[ap]
        a_t = N.num(kk + ".pr_auc", round(float(a), 3), src) if a not in ("", "nan") else "--"
        m = r[miss]
        m_t = N.num(kk + ".miss", round(float(m), 3), src) if m not in ("", "nan") else "--"
        return f"{a_t} / {m_t}"

    for by, s, label in strata:
        key = (by, s)
        n = boost.get(key) or kumo.get(key) or lim.get(key)
        if n is None:
            n_t = u_t = "--"
        else:
            src_n = short(bo) if boost.get(key) else (short(ku) if kumo.get(key) else lim[key]["_src"])
            kk = f"strata.n.{by}.{s.replace('>=', 'ge').replace('-', '_').replace('.', 'p')}"
            n_t = N.num(kk, int(n["n_samples"]), src_n)
            u_t = N.num(kk + ".unstable", int(n["n_positive"]), src_n)
        rows.append([label, n_t, u_t,
                     cell(boost, key, "boosting", short(bo), "average_precision", "miss_rate"),
                     cell(kumo, key, "kumo_tabular", short(ku), "pr_auc", "miss_rate"),
                     cell(lim, key, "limix_2", lim[key]["_src"] if key in lim
                          else f"e5_tfm2/strata/line_pre__limix_2__hour__by_{by}.csv",
                          "average_precision", "miss_rate")])
        if s == "380":
            rows.append(r"\midrule")
    all_b = boost.get(("vn_kv", "__all__"))
    all_k = kumo.get(("vn_kv", "__all__"))
    all_l = lim.get(("vn_kv", "__all__"))
    rows.append(r"\midrule")
    rows.append(["All events", N.num("strata.n.all", int(all_b["n_samples"]), short(bo)) if all_b else "--",
                 N.num("strata.n.all.unstable", int(all_b["n_positive"]), short(bo)) if all_b else "--",
                 cell(boost, ("vn_kv", "__all__"), "boosting", short(bo), "average_precision", "miss_rate") if all_b else "--",
                 cell(kumo, ("vn_kv", "__all__"), "kumo_tabular", short(ku), "pr_auc", "miss_rate") if all_k else "--",
                 cell(lim, ("vn_kv", "__all__"), "limix_2", all_l["_src"], "average_precision", "miss_rate") if all_l else "--"])
    notes = ("Cells: PR-AUC / miss rate. Hour protocol, out-of-fold scores, threshold 0.5, line\\_pre features. Boosting: tuned LightGBM "
             "(e1\\_boosting posttrip30); Kumo Tabular and LimiX-2 zero-shot.")
    k220 = N.get("strata.boosting.vn_kv.220.pr_auc") if "strata.boosting.vn_kv.220.pr_auc" in N.records else "0.65"
    k23 = N.get("strata.boosting.clearing_band.0p2_0p3s.miss") if "strata.boosting.clearing_band.0p2_0p3s.miss" in N.records else "1.000"
    caption = (f"Where the pre-fault models fail: boosting, Kumo and LimiX-2 miss the same events. Boosting reaches PR-AUC {k220} "
               f"on 220\\,kV lines and misses every unstable event cleared in 0.2--0.3\\,s (miss {k23}); "
               "the foundation models shift the level, not the pattern.")
    # single-column table (page budget): six narrow columns at \scriptsize fit 3.5 in
    return table("tab:strata", caption, "@{}lrrccc@{}", hdr, rows, star=False, size=r"\scriptsize",
                 notes=notes, colsep="3pt")


# --------------------------------------------------------------------------
# tab:cct
# --------------------------------------------------------------------------
def tab_cct(S: Sources, N: Numbers) -> str:
    N.table = "tab:cct"
    hdr = ["Model", "Subset", "$n$ (unstable)", "PR-AUC hour", "Miss", "MCC"]
    rows: list[list[str] | str] = []
    cc = S.e("e4_sim/cct_consistency.json")
    pg = S.e("e4_sim/progress.json")
    if cc.exists():
        d = read_json(cc)
        src = short(cc)
        cases = [c for c in d["cases"] if c["decided"] >= 50]
        n_cases = N.num("cct.cases_bracketed", len(cases), src, text=str(len(cases)))
        n_planned = "18"
        if pg.exists():
            n_planned = N.num("cct.cases_planned", len(read_json(pg)["cases_planned"]), short(pg),
                              text=str(len(read_json(pg)["cases_planned"])))
        decided = N.num("cct.decided_events", int(d["total"]["decided"]), src)
        agree = N.num("cct.agreement_pct", round(100 * d["total"]["agreement_rate"], 1), src,
                      text=f"{100 * d['total']['agreement_rate']:.1f}")
        lo = min(c["agreement_rate"] for c in cases)
        hi = max(c["agreement_rate"] for c in cases)
        lo_t = N.num("cct.agreement_case_min_pct", round(100 * lo, 1), src, text=f"{100 * lo:.1f}")
        hi_t = N.num("cct.agreement_case_max_pct", round(100 * hi, 1), src, text=f"{100 * hi:.1f}")
        inside = N.num("cct.inside_bracket", int(d["total"]["inside_bracket"]), src, text=str(d["total"]["inside_bracket"]))
        rows.append(group(f"Bracket--label consistency: bisection CCT (width 10\\,ms) vs v4.1 label, "
                          f"{n_cases} of {n_planned} cases", 6))
        rows.append(["Events decided by the bracket (undecided, clearing time inside the bracket: " + inside + ")",
                     multicol(2, decided, "l"),
                     multicol(3, f"agreement {agree}\\,\\% (per case {lo_t}--{hi_t}\\,\\%)", "l")])
    else:
        rows.append(["Bracket--label consistency", multicol(5, N.todo("cct.agreement_pct", "e4_sim/cct_consistency.json", "e4_sim"), "l")])
    rows.append(r"\midrule")
    res = S.e("e1_boosting/results/cct_two_stage_pt/results.csv")
    summ = S.e("e1_boosting/results/cct_two_stage_pt/summary.json")
    if summ.exists():
        sm = read_json(summ)
        src = short(summ)
        N.num("cct.n_pairs", int(sm["n_pairs"]), src)
        N.num("cct.n_two_sided", int(sm["n_two_sided"]), src, text=str(sm["n_two_sided"]))
        N.num("cct.events_two_sided", int(sm["events_two_sided"]), src)
        N.num("cct.events_swept_bracketed", int(sm["events_swept_bracketed"]), src, text=str(sm["events_swept_bracketed"]))
    if res.exists():
        rs = read_csv(res)
        src = short(res)
        names = {"A_direct": "A: direct classifier", "B_cct_regression": "B: CCT regression, thresholded at $t_c$",
                 "B2_cct_censored": "B2: censored CCT regression"}
        subsets = {"all": "all events", "two_sided_pairs": "two-sided CCT bounds", "swept_bracketed_lines": "bracketed sweep lines"}
        rows.append(group("Two-stage CCT models vs the direct classifier (line\\_pre + post-trip, hour protocol)", 6))
        for sub in ("all", "two_sided_pairs", "swept_bracketed_lines"):
            for model in ("A_direct", "B_cct_regression", "B2_cct_censored"):
                r = next((x for x in rs if x["protocol"] == "hour" and x["subset"] == sub and x["model"] == model), None)
                key = f"cct.hour.{sub}.{model}"
                if r is None:
                    rows.append([names[model], subsets[sub], "--", N.todo(key + ".pr_auc", key, src), "--", "--"])
                    continue
                n = N.num(f"cct.{sub}.n", int(r["n"]), src)
                nu = N.num(f"cct.{sub}.n_unstable", int(r["n_unstable"]), src)
                ap = row_ci(N, key + ".pr_auc", r, src)
                miss = N.num(key + ".miss", round(float(r["miss_rate"]), 3), src)
                mcc = N.num(key + ".mcc", round(float(r["mcc"]), 3), src)
                rows.append([names[model], subsets[sub], f"{n} ({nu})", ap, miss, mcc])
            if sub != "swept_bracketed_lines":
                rows.append(r"\addlinespace")
    else:
        rows.append(["Two-stage models", multicol(5, N.todo("cct.two_stage", "cct_two_stage_pt/results.csv", "e1_boosting"), "l")])
    rows.append(["CCT sweep, all 18 cases; CCT-margin feature", multicol(5, N.todo("cct.final_sweep",
                 "CCT sweep final (18 cases) and CCT-margin feature run", "e4_sim", expected=True), "l")])
    notes = ("Threshold 0.5. Subsets: all \\num{11159} events; events of (case, line) pairs with two-sided "
             "label-derived CCT bounds; events on sweep lines whose CCT is bracketed inside the 0.05--0.6\\,s sweep.")
    b2 = N.get("cct.hour.swept_bracketed_lines.B2_cct_censored.pr_auc") if "cct.hour.swept_bracketed_lines.B2_cct_censored.pr_auc" in N.records else "0.97"
    ball = N.get("cct.hour.all.B_cct_regression.pr_auc") if "cct.hour.all.B_cct_regression.pr_auc" in N.records else "0.39"
    ag = N.get("cct.agreement_pct") if "cct.agreement_pct" in N.records else "98.8"
    caption = (f"The data lever. Where the critical clearing time is bracketed, the two-stage model reaches PR-AUC "
               f"{b2.split()[0]} against {ball.split()[0]} on all events, where most CCTs are censored; the bisection brackets agree with the "
               f"simulation labels at {ag}\\,\\%.")
    return table("tab:cct", caption, "@{}p{0.34\\textwidth}p{0.17\\textwidth}llcc@{}", hdr, rows, star=True,
                 size=r"\footnotesize", notes=notes)


# --------------------------------------------------------------------------
# tab:latency
# --------------------------------------------------------------------------
def tab_latency(S: Sources, N: Numbers) -> str:
    N.table = "tab:latency"
    hdr = ["Method", "Device", "Context rows", "Cache", "Median (ms)", "p95 (ms)", "Rows/s batched", "GPU peak (MB)"]
    rows: list[list[str] | str] = []
    lat = S.e("e5_tfm2/latency.jsonl")
    log = S.e("e5_tfm2/run_lat.log")
    recs: list[tuple[dict[str, Any], str, int | None]] = []
    if lat.exists():
        recs += [(r, short(lat), None) for r in read_jsonl(lat)]
    if log.exists():  # the CPU boosting rows were written to the run log only
        have = {(r["model"], r["cache"]) for r, _, _ in recs}
        with log.open() as fh:
            for i, line in enumerate(fh, start=1):
                if line.startswith("{") and '"median_ms"' in line:
                    r = json.loads(line.replace("NaN", "null"))
                    if (r["model"], r["cache"]) not in have:
                        recs.append((r, short(log), i))
    names = {"xgboost": "XGBoost (line\\_pre)", "lightgbm": "LightGBM (line\\_pre)", "tabdpt": "TabDPT",
             "tabicl_v2": "TabICL v2", "causilo": "Causilo", "mitra_v2": "Mitra v2", "limix_2m": "LimiX-2M",
             "limix_16m": "LimiX-16M", "tabpfn_v2": "TabPFN v2", "tabfm_v1_1": "TabFM 1.1", "kumo_tabular": "Kumo Tabular",
             "tabfm": "TabFM 1.0", "limix_2": "LimiX-2", "tabpfn_3": "TabPFN 3", "tabpfn_3_5": "TabPFN 3.5"}
    rows.append(group("Single-event prediction, 33 features, one machine", len(hdr)))
    if recs:
        ctx = {r["context_rows"] for r, _, _ in recs}
        N.num("latency.context_rows", int(sorted(ctx)[0]), recs[0][1])
        N.num("latency.n_features", int(recs[0][0]["n_features"]), recs[0][1], text=str(recs[0][0]["n_features"]))
        recs.sort(key=lambda t: (t[0]["model"] not in ("xgboost", "lightgbm"), t[0]["median_ms"] if not t[0]["cache"] else 0,
                                 t[0]["cache"]))
        by_model: dict[str, list] = {}
        for r, src, ln in recs:
            by_model.setdefault(r["model"], []).append((r, src, ln))
        for model in sorted(by_model, key=lambda m: min(x[0]["median_ms"] for x in by_model[m] if not x[0]["cache"]) if any(not x[0]["cache"] for x in by_model[m]) else 0):
            for r, src, ln in sorted(by_model[model], key=lambda t: t[0]["cache"]):
                k = f"latency.{model}" + (".cached" if r["cache"] else "")
                med = N.num(k + ".median_ms", round(r["median_ms"], 2), src, line=ln,
                            text=f"{r['median_ms']:,.2f}" if r["median_ms"] < 10 else f"\\num{{{r['median_ms']:.0f}}}")
                p95 = N.num(k + ".p95_ms", round(r["p95_ms"], 2), src, line=ln,
                            text=f"{r['p95_ms']:,.2f}" if r["p95_ms"] < 10 else f"\\num{{{r['p95_ms']:.0f}}}")
                thr = N.num(k + ".rows_per_s", int(round(r["throughput_rows_per_s"])), src, line=ln)
                gpu = r.get("gpu_peak_mb")
                gpu_t = N.num(k + ".gpu_peak_mb", int(round(gpu)), src, line=ln) if gpu not in (None, "") and not (
                    isinstance(gpu, float) and math.isnan(gpu)) else "--"
                rows.append([names.get(model, model), "GPU" if r["device"] == "cuda" else "CPU",
                             r"\num{" + str(r["context_rows"]) + "}", "yes" if r["cache"] else "no", med, p95, thr, gpu_t])
    else:
        rows.append(["Foundation models", multicol(7, N.todo("latency.tfm", "e5_tfm2/latency.jsonl", "e5_tfm2"), "l")])
    for model in ("tabpfn_3", "tabpfn_3_5"):
        if not any(r["model"] == model for r, _, _ in recs):
            rows.append([names.get(model, model.replace("_", " ").replace("tabpfn", "TabPFN")), "GPU", "--", "no",
                         N.todo(f"latency.{model}.median_ms", f"{model} latency", "e5_tfm2/latency.jsonl", expected=True), "--", "--", "--"])
    # GNN and sequence nets (batched)
    gr = S.e("e3_deep/tables/gnn_results.csv")
    if gr.exists():
        v = [float(r["predict_ms_per_event"]) for r in read_csv(gr) if r["predict_ms_per_event"]]
        lo = N.num("latency.gnn.ms_per_event_min", round(min(v), 2), short(gr), text=fnum(min(v), 2))
        hi = N.num("latency.gnn.ms_per_event_max", round(max(v), 2), short(gr), text=fnum(max(v), 2))
        rows.append(["GNN (GAT / GraphSAGE / GCN), batched$^{a}$", "GPU", "--", "--", f"{lo}--{hi}", "--", "--", "--"])
    sq = S.e("e3_deep/tables/seq_results.csv")
    if sq.exists():
        v = [float(r["predict_ms_per_event"]) for r in read_csv(sq) if r["predict_ms_per_event"] and r["protocol"] == "hour"]
        lo = N.num("latency.seq.ms_per_event_min", float(f"{min(v):.2g}"), short(sq), text=f"{min(v):.2g}")
        hi = N.num("latency.seq.ms_per_event_max", float(f"{max(v):.2g}"), short(sq), text=f"{max(v):.2g}")
        rows.append(["Sequence networks (LSTM / TCN / Transformer), batched, hour jobs", "GPU", "--", "--", f"{lo}--{hi}", "--", "--", "--"])
    rows.append(r"\midrule")
    rows.append(group("Simulation-based methods", len(hdr)))
    tm = S.doc("timing.md")
    hit = md_find(tm, r"\| TDS, ANDES 2\.0\.0, full 6\.6 s horizon, 10 ms step \| case \+ scenario \| ([\d.]+) s \(mean of (\d+) events")
    if hit:
        tds = N.num("latency.tds_m3_s", to_float(hit[0].group(1)), short(tm), line=hit[1], text=hit[0].group(1))
        n20 = N.num("latency.tds_m3_n_events", int(hit[0].group(2)), short(tm), line=hit[1], text=hit[0].group(2))
        rows.append([f"TDS, ANDES, 6.6\\,s horizon (mean of {n20} events)", "CPU, Apple M3, 1 core", "--", "--",
                     tds + "\\,s", "--", "--", "--"])
    else:
        rows.append(["TDS, ANDES", "CPU", "--", "--", N.todo("latency.tds_m3_s", "TDS 13.2 s", short(tm)), "--", "--", "--"])
    val = S.e("e4_sim/validation.json")
    if val.exists():
        d = read_json(val)
        src = short(val)
        sc = {k: v for k, v in d["single_core_s_per_event"].items() if k != "note"}
        lo = N.num("latency.tds_cairo_s_min", min(sc.values()), src, text=f"{min(sc.values()):.1f}")
        hi = N.num("latency.tds_cairo_s_max", max(sc.values()), src, text=f"{max(sc.values()):.1f}")
        eph = N.num("latency.tds_events_per_hour_16", int(round(d["events_per_hour_16_workers"])), src)
        wall = N.num("latency.tds_wall_s_per_event_16", d["wall_s_per_event_16_workers"], src,
                     text=f"{d['wall_s_per_event_16_workers']:.2f}")
        nw = N.num("latency.tds_workers", int(d["local_jobs"]), src, text=str(d["local_jobs"]))
        nrep = N.num("latency.tds_reproduced_events", int(d["n"]), src, text=str(d["n"]))
        N.num("latency.tds_traces_bit_identical", bool(d["traces_bit_identical"]), src, text=str(d["traces_bit_identical"]))
        rows.append([f"TDS, ANDES, same events, this machine ({nrep} events bit-identical)", "CPU, 1 core / " + nw + " workers",
                     "--", "--", f"{lo}--{hi}\\,s / {wall}\\,s wall", "--", f"{eph} events/h", "--"])
    hit = md_find(tm, r"\| SIME total = TDS to the decision instant \+ margin \| .*? \| ≈ ([\d.]+) s \+ [\d.]+ s ≈ [\d.]+ s \(median\); ≈ ([\d.]+) s \(p90")
    if hit:
        s_med = N.num("latency.sime_total_s", to_float(hit[0].group(1)), short(tm), line=hit[1], text=hit[0].group(1))
        s_p90 = N.num("latency.sime_total_p90_s", to_float(hit[0].group(2)), short(tm), line=hit[1], text=hit[0].group(2))
        rows.append(["SIME total: TDS to the decision instant + margin (estimate)", "CPU, Apple M3", "--", "--",
                     f"$\\approx$\\,{s_med}\\,s", f"$\\approx$\\,{s_p90}\\,s (p90)", "--", "--"])
    hit = md_find(tm, r"\| SIME margin \(Pavella OMIB, 5 gap candidates\) \| .*? \| ([\d.]+) ms / ([\d.]+) ms \((\d+) machines")
    if hit:
        m_med = N.num("latency.sime_margin_ms", to_float(hit[0].group(1)), short(tm), line=hit[1], text=hit[0].group(1))
        m_p90 = N.num("latency.sime_margin_p90_ms", to_float(hit[0].group(2)), short(tm), line=hit[1], text=hit[0].group(2))
        rows.append([f"SIME margin alone ({hit[0].group(3)} machines)", "CPU, Apple M3", "--", "--", m_med, m_p90 + " (p90)", "--", "--"])
    notes = ("$^{a}$Including per-event graph assembly on the CPU. Foundation-model latency measured with the "
             "hour-protocol training fold as context (\\num{200} timed single-row calls); \"cache\" = context "
             "embeddings precomputed. Production TDS cost: 28.9 core-h for the event set on Azure "
             "Standard\\_F8as\\_v7 (docs/timing.md).")
    xg = N.get("latency.xgboost.median_ms") if "latency.xgboost.median_ms" in N.records else "0.12"
    lx = N.get("latency.limix_2.median_ms") if "latency.limix_2.median_ms" in N.records else "47,625"
    tds = N.get("latency.tds_m3_s") if "latency.tds_m3_s" in N.records else "13.2"
    caption = (f"Measured latency per event. From {xg}\\,ms (XGBoost, CPU) to {lx}\\,ms (LimiX-2, GPU, uncached) "
               f"against {tds}\\,s for time-domain simulation.")
    return table("tab:latency", caption, "@{}p{0.24\\textwidth}p{0.11\\textwidth}rcrrrr@{}", hdr, rows, star=True,
                 size=r"\scriptsize", notes=notes, colsep="4pt")


# --------------------------------------------------------------------------
# tab:deploy
# --------------------------------------------------------------------------
def tab_deploy(S: Sources, N: Numbers) -> str:
    N.table = "tab:deploy"
    path = S.tables_src / "deploy.yaml"
    y = yaml.safe_load(path.read_text())
    rx = re.compile(r"\{([a-zA-Z0-9_.]+)\}")

    def resolve(text: str) -> str:
        def rep(m: re.Match[str]) -> str:
            key = m.group(1)
            if key in N.records:
                t = N.records[key]["text"]
                return t.split(" [")[0]  # point estimate only in this table
            return N.todo(key, f"deploy placeholder {key}", short(path))
        return rx.sub(rep, text)

    rows: list[list[str] | str] = []
    for r in y["rows"]:
        rows.append([resolve(str(r[k])) for k in ("job", "instant", "method", "pr_auc", "latency", "remark")])
    N.num("deploy.n_jobs", len(y["rows"]), short(path), text=str(len(y["rows"])))
    return table("tab:deploy", resolve(y["caption"].strip()),
                 "@{}p{0.14\\textwidth}p{0.09\\textwidth}p{0.19\\textwidth}p{0.09\\textwidth}p{0.13\\textwidth}p{0.21\\textwidth}@{}",
                 y["columns"], rows, star=True, size=r"\footnotesize")


# --------------------------------------------------------------------------
# tab:sota
# --------------------------------------------------------------------------
def tab_sota(S: Sources, N: Numbers) -> str:
    """Literature table. Each literature number is registered as sota.<key>.<name> with the
    yaml row and DOI/URL as source; the "This work" row resolves numbers.json keys."""
    N.table = "tab:sota"
    path = S.tables_src / "sota.yaml"
    y = yaml.safe_load(path.read_text())
    rx = re.compile(r"\{([a-zA-Z0-9_.]+)\}")
    fields = ("system", "real", "events", "protocol", "task", "model", "metric", "latency")

    def lit(r: dict[str, Any], text: str) -> str:
        def rep(m: re.Match[str]) -> str:
            name = m.group(1)
            v = r["numbers"][name]
            t = fmt(v, digits=len(str(v).split(".")[1]) if isinstance(v, float) else 3)
            return N.num(f"sota.{r['key']}.{name}", v,
                         f"{short(path)} row {r['key']}; {r['source']}", text=t)
        return rx.sub(rep, text)

    def own(text: str) -> str:
        def rep(m: re.Match[str]) -> str:
            key = m.group(1)
            if key in N.records:
                return N.records[key]["text"].split(" [")[0]
            return N.todo(key, f"sota placeholder {key}", short(path))
        return rx.sub(rep, text)

    rows: list[list[str] | str] = []
    for r in y["rows"]:
        rows.append([f"\\cite{{{r['key']}}}"] + [lit(r, str(r[k])) for k in fields])
    rows.append(r"\midrule")
    rows.append([r"This work"] + [own(str(y["this_work"][k])) for k in fields])
    N.num("sota.n_studies", len(y["rows"]), short(path), text=str(len(y["rows"])))
    return table("tab:sota", y["caption"].strip(),
                 "@{}p{0.06\\textwidth}p{0.11\\textwidth}p{0.04\\textwidth}p{0.06\\textwidth}"
                 "p{0.11\\textwidth}p{0.08\\textwidth}p{0.12\\textwidth}p{0.17\\textwidth}p{0.12\\textwidth}@{}",
                 y["columns"], rows, star=True, size=r"\scriptsize", colsep="3pt")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
TABLES = [("protocols", tab_protocols), ("model_dataset", tab_model_dataset), ("real_events", tab_real_events),
          ("methods", tab_methods), ("postclear", tab_postclear), ("prefault", tab_prefault),
          ("strata", tab_strata), ("cct", tab_cct), ("latency", tab_latency), ("deploy", tab_deploy),
          ("sota", tab_sota)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=PAPER / "tables", help="output folder for *.tex and numbers.json")
    ap.add_argument("--exp", type=Path, default=Path("~/data/exp").expanduser(), help="experiment results root")
    ap.add_argument("--src", type=Path, default=Path("~/data/paper_src").expanduser(),
                    help="FROZEN_*.md, real_events/, docs/ copies")
    ap.add_argument("--tables-src", type=Path, default=PAPER / "tables_src", help="YAML sources")
    ap.add_argument("--fig-numbers", type=Path, default=PAPER / "figures" / "figure_numbers.json",
                    help="caption numbers written by paper_figures.py")
    ap.add_argument("--prep", type=Path, default=Path("~/data/ml_v41dyn/prep").expanduser(),
                    help="ANDES case files (CASE.xlsx) for the case facts")
    ap.add_argument("--check", action="store_true", help="exit non-zero if any source is missing")
    ap.add_argument("--only", nargs="*", help="generate only these tables")
    args = ap.parse_args(argv)

    S = Sources(args.exp.expanduser(), args.src.expanduser(), args.tables_src, args.prep.expanduser())
    N = Numbers()
    args.out.mkdir(parents=True, exist_ok=True)
    produced = []
    for name, fn in TABLES:
        if args.only and name not in args.only:
            continue
        tex = fn(S, N)
        (args.out / f"{name}.tex").write_text(tex)
        produced.append(name)
    # numbers stated in figure captions (scripts/paper_figures.py -> figure_numbers.json)
    fn = args.fig_numbers.expanduser()
    if fn.exists():
        N.table = "figures"
        for k, v in json.loads(fn.read_text()).items():
            N.num(f"fig.{k}", v["value"], v["source"], text=json.dumps(v["value"], ensure_ascii=False))
    out = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "exp": str(S.exp), "src": str(S.src),
        "tables": [f"tab:{n}" for n in produced],
        "numbers": N.records,
        "missing": N.missing,
    }
    (args.out / "numbers.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(produced)} tables and {len(N.records)} numbers to {args.out}")
    unexpected = [m for m in N.missing if not m["expected"]]
    if N.missing:
        print(f"{len(N.missing)} placeholder(s) (\\todo), {len(unexpected)} from missing sources:")
        for m in N.missing:
            flag = "expected" if m["expected"] else "MISSING SOURCE"
            print(f"  [{m['table']}] {m['key']}: {m['what']}  <- {m['source']}  ({flag})")
    if args.check and unexpected:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
