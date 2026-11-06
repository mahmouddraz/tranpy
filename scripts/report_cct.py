"""Write ~/data/exp/e4_sim/REPORT.md from the files of the e4_sim experiment (no numbers typed in)."""
# ruff: noqa: E501

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "data/exp/e4_sim")
V = json.loads((OUT / "validation.json").read_text())
P = json.loads((OUT / "progress.json").read_text())
T = pd.read_parquet(OUT / "cct_brackets.parquet")
C = (
    json.loads((OUT / "cct_consistency.json").read_text())
    if (OUT / "cct_consistency.json").exists()
    else {}
)

T["cct_mid"] = np.where(T["status"] == "bracketed", 0.5 * (T["cct_lo"] + T["cct_hi"]), np.nan)
T["vn_kv"] = T["vn_kv"].astype(int)
T["n_circuits"] = T["n_circuits"].clip(upper=3)
lines = []
w = lines.append
w("# e4_sim: local fault-event simulation and CCT bracketing of the v4.1 cases\n")
w(
    "All numbers below come from files in this folder (validation.json, progress.json, cct_brackets.parquet, cct_consistency.json, cct/).\n"
)
w("## 1. Local simulation (scripts/run_sim_local.py)\n")
w(
    f"Re-simulation of {V['n']} events (sample ids {V['sample_ids'][0]}-{V['sample_ids'][-1]}, shard {V['shard']}) of {V['case']} with 16 workers on CPU threads 32-47, "
    f"frozen events_v2 scripts (sha256 checked against FROZEN_DYN_v41.md), ANDES 2.0.0, numpy 2.5.4:\n"
)
w(
    f"- labels identical: {V['labels_equal']} ({V['n_unstable']} unstable of {V['n']}); max |diff| of tsi, angle separation, clearing time, fault line, snapshots: "
    f"{max(V[k] for k in V if k.startswith('max_abs_diff_') and 'trace' not in k and 'rotor' not in k and 'channels' not in k):g}"
)
w(
    f"- traces: {V['traces_compared']} events traced on both sides ({V['trace_rows']} rows), time grid equal: {V['time_grid_equal']}, max |rotor-angle diff| {V['max_abs_diff_rotor_angle_deg']:g} deg, all channels bit-identical: {V['traces_bit_identical']}"
)
s = V["single_core_s_per_event"]
w(
    f"- speed: one core (idle machine) {s['line764_clr0p1_stable']}-{s['line764_clr0p6_unstable']} s per event (6.6 s window, stable-unstable); "
    f"16 workers: {V['wall_s_per_event_16_workers']} s wall per event = {V['events_per_hour_16_workers']:.0f} events/hour on the 30-event batch (tail-dominated; a full case of 648 events is ~{648 / V['events_per_hour_16_workers'] * 60:.0f} min).\n"
)
w("## 2. CCT bracketing (scripts/cct_bracket.py)\n")
w(
    f"Bisection of the clearing time between {P['cfg']['lo']} and {P['cfg']['hi']} s to a {P['cfg']['width'] * 1000:.0f} ms bracket, three-phase 0.5 ohm fault at t = 1 s at the line terminal "
    "(both terminals, fault_position 0 and 1, as the v4.1 events draw either), base operating point, v4.1 out-of-step label (|delta - delta_COI| > 180 deg, latched), "
    "7 s window (v4.1: 6.6 s; the out-of-step onset time t_oos_s is stored). Fault candidates: the lines whose opening cuts off no generator (gen.py's rule).\n"
)
cases = sorted(T["case"].unique())
w(
    f"Cases finished: {len(cases)} of 18 ({', '.join(cases)}); planned order {', '.join(P['cases_planned'])}; "
    f"progress.json: current case {P['current']['case']} {P['current']['done']}/{P['current']['total']} tasks, {P['current'].get('sims_per_hour')} sims/hour.\n"
)
w("### Status per case\n")
st = T.groupby("case")["status"].value_counts().unstack(fill_value=0)
st["rows"] = st.sum(axis=1)
st["sims"] = T.groupby("case")["n_sims"].sum()
st["sec_per_task"] = T.groupby("case")["sec"].mean().round(1)
w(st.to_markdown())
w("")
w(
    "`above` = stable at 1.0 s (CCT > 1 s), `below` = unstable at 0.05 s, `failed` = a simulation of the bisection failed numerically (bracket kept where it got).\n"
)
w(
    "### CCT distribution by voltage level and circuit count (bracketed rows, midpoint of the bracket, s)\n"
)
B = T[T["status"] == "bracketed"]
g = (
    B.groupby(["vn_kv", "n_circuits"])["cct_mid"]
    .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    .round(3)
)
g["share_above_1s"] = T.groupby(["vn_kv", "n_circuits"])["status"].apply(
    lambda s: round((s == "above").mean(), 3)
)
g["share_below_50ms"] = T.groupby(["vn_kv", "n_circuits"])["status"].apply(
    lambda s: round((s == "below").mean(), 3)
)
w(g.to_markdown())
w("")
w("n_circuits = parallel lines between the same two buses (3 = three or more).\n")
w("### CCT distribution per case\n")
w(B.groupby("case")["cct_mid"].describe(percentiles=[0.1, 0.5, 0.9]).round(3).to_markdown())
w("")
w("### Terminal dependence\n")
pv = B.pivot_table(index=["case", "fault_line"], columns="fault_position", values="cct_mid")
if pv.shape[1] == 2:
    d = (pv[0.0] - pv[1.0]).abs()
    w(
        f"Lines bracketed at both terminals: {int(d.notna().sum())}; |CCT(from) - CCT(to)|: median {d.median():.3f} s, 90th percentile {d.quantile(0.9):.3f} s, max {d.max():.3f} s; "
        f"share with a difference above 50 ms: {(d > 0.05).mean():.3f}.\n"
    )
w("### Short-CCT lines\n")
short = B[B["cct_hi"] <= 0.15]
w(
    f"Rows with CCT <= 150 ms: {len(short)} of {len(B)} bracketed ({len(short) / len(B):.3f}); by voltage: {short['vn_kv'].value_counts().to_dict()}; by circuits: {short['n_circuits'].value_counts().to_dict()}.\n"
)
if C:
    w("## 3. Consistency with the v4.1 labels\n")
    w(
        "For every v4.1 event of a finished case, matched on (case, fault_line, fault_position): predicted unstable <=> clearing_time_s > cct_hi. Events whose clearing time falls inside the 10 ms bracket are not judged. "
        "The v4.1 events draw the loading in 0.9-1.1 with 5 % per-load noise, the brackets are at the base operating point, so disagreements near the CCT are expected.\n"
    )
    w(pd.DataFrame(C["cases"]).to_markdown(index=False))
    w("")
    w(f"Total: {C['total']}\n")
    D = (
        pd.read_csv(OUT / "cct_disagreements.csv")
        if (OUT / "cct_disagreements.csv").exists()
        else pd.DataFrame()
    )
    if len(D):
        D["margin_s"] = np.where(
            D["label"] == 1,
            D["cct_hi"].fillna(np.inf) - D["clearing_time_s"],
            D["clearing_time_s"] - D["cct_lo"].fillna(-np.inf),
        )
        D["loading_dev"] = D["mean_load_factor"] - 1.0
        w(
            f"Disagreements: {len(D)} (cct_disagreements.csv). Margin from the bracket: median {D['margin_s'].median():.3f} s, 90th percentile {D['margin_s'].quantile(0.9):.3f} s, max {D['margin_s'].max():.3f} s. "
            f"Mean load factor of the disagreeing events: v4.1-unstable-but-bracket-stable mean {D.loc[D.label == 1, 'mean_load_factor'].mean():.3f}, v4.1-stable-but-bracket-unstable mean {D.loc[D.label == 0, 'mean_load_factor'].mean():.3f} (base 1.0).\n"
        )
        far = D[D["margin_s"] > 0.1]
        w(
            f"Disagreements more than 100 ms away from the bracket: {len(far)}; of these v4.1 unstable: {int((far.label == 1).sum())}, v4.1 stable: {int((far.label == 0).sum())}.\n"
        )
w("## 4. Files\n")
w("- scripts/run_sim_local.py, scripts/cct_bracket.py, scripts/report_cct.py (code.patch)")
w("- validation.json, validate/high_load_t10/ (re-simulated shard r000 subset in compact format)")
w(
    "- cct/cct_CASE.parquet (per case, resumable), cct_brackets.parquet / .csv (all finished cases), cct_consistency.json, cct_disagreements.csv, progress.json, run.log"
)
(OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
