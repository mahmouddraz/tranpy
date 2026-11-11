"""Connected power flow of the v5 grid model: table loading, reactive limits, both solvers, metrics.

Library module of the v5 checks. ``go4.py`` (``~/data/grid/v5/cache/work/pf_v5``) loads it by file
path as ``pfc``, and ``run_check.sh`` / ``gate.py`` run ``go4.py``. Moved here from
``~/data/grid/v5/cache/work/pf_connected/run.py`` (PLAN step 4, I1); the computations are unchanged
apart from I1: boundary reactive limits count in-operation lines only
(:func:`tranpy.reactive_power.boundary_q_limits`).

Run directly (the old pf_connected batch mode)::

    python scripts/v5_pf_connected.py TAG [TAG ...]

reads ``input_<TAG>.pkl`` from ``$PF_CONNECTED_DIR`` (default
``~/data/grid/v5/cache/work/pf_connected``) and writes ``results_<TAG>.csv`` and
``overloads_<TAG>.csv`` there.
"""

from __future__ import annotations

import os
import pickle
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandapower as pp
import pandas as pd
import pypowsybl as ps

from tranpy.grid_connections import set_taps
from tranpy.pandapower_model import line_loading as PL
from tranpy.pandapower_model import network_from_cim_tables as PN
from tranpy.powsybl_model import _ids
from tranpy.powsybl_model import line_loading as SL
from tranpy.powsybl_model import network_from_cim_tables as SN
from tranpy.reactive_power import boundary_q_limits

OUT_DIR = Path(
    os.environ.get("PF_CONNECTED_DIR", str(Path.home() / "data/grid/v5/cache/work/pf_connected"))
)
FL = ("_mw", "_mvar", "_ohm", "_kv", "_s", "_a", "_pu", "_mva")


def tables(d: dict[str, Any]) -> types.SimpleNamespace:
    """Tables from plain column dicts, with the numeric columns converted."""

    def frame(v: Any) -> pd.DataFrame:
        f = pd.DataFrame(v)
        for c in f:
            if c.endswith(FL) or c == "target_v_pu" or c in ("kv",):
                f[c] = pd.to_numeric(f[c], errors="coerce")
        return f

    return types.SimpleNamespace(**{k: frame(v) for k, v in d.items()})


def qlim(t: Any, tan: float) -> types.SimpleNamespace:
    """Load tan phi, machine reactive limits and boundary injection limits (``t.ext``)."""
    g = t.synchronous_machine.copy()
    t = types.SimpleNamespace(**vars(t))
    load = t.energy_consumer.copy()
    own = ~load.mrid.astype(str).str.startswith(("hvdc/", "pump/"))
    load["q_mvar"] = load.q_mvar.where(~own, load.p_mw * tan)
    t.energy_consumer = load
    g["q_max_mvar"] = g.q_max_mvar.where(g.q_max_mvar.notna(), 0.5 * g.p_mw / 0.8)
    g["q_min_mvar"] = g.q_min_mvar.where(g.q_min_mvar.notna(), -0.3 * g.p_mw / 0.8)
    t.synchronous_machine = g
    # I1: only in-operation lines at the boundary bus count
    t.ext = boundary_q_limits(t.ac_line_segment, t.external_network_injection)
    x = t.external_network_injection.copy()
    x["target_v_pu"] = 1.02
    t.external_network_injection = x
    return t


def sy(t: Any, lim: bool = True) -> tuple[Any, str]:
    """pypowsybl AC load flow with the reactive limits of :func:`qlim`."""
    n = SN(t)
    gens = n.get_generators()
    gm = t.synchronous_machine
    xm = t.external_network_injection
    ids, mi, ma = [], [], []
    for i, a, b in zip(_ids("G", gm.mrid), gm.q_min_mvar, gm.q_max_mvar, strict=True):
        ids.append(i)
        mi.append(float(a) if lim else -1e5)
        ma.append(float(b) if lim else 1e5)
    for i, m in zip(_ids("X", xm.mrid), xm.mrid, strict=True):
        ids.append(i)
        e = t.ext[m]
        mi.append(-e if lim else -1e5)
        ma.append(e if lim else 1e5)
    n.create_minmax_reactive_limits(
        pd.DataFrame({"min_q": mi, "max_q": ma}, index=pd.Index(ids, name="id"))
    )
    x = _ids("X", xm.mrid)
    tp = gens.loc[x, "target_p"]
    n.update_generators(id=x, min_p=(tp - 0.001).tolist(), max_p=(tp + 0.001).tolist())
    par = ps.loadflow.Parameters(
        distributed_slack=True,
        balance_type=ps.loadflow.BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX,
        use_reactive_limits=lim,
        provider_parameters={"maxOuterLoopIterations": "100", "maxNewtonRaphsonIterations": "60"},
    )
    r = ps.loadflow.run_ac(n, par)[0]
    return n, r.status.name


def volt(n: Any, t: Any) -> tuple[dict[str, float], pd.DataFrame, pd.Series]:
    """Busbar voltage (pu) by busbar mrid, plus the bus-breaker view and all bus voltages."""
    b = n.get_bus_breaker_view_buses()
    vl = n.get_voltage_levels().nominal_v
    ids = dict(zip(_ids("B", t.busbar_section.mrid), t.busbar_section.mrid, strict=True))
    v = b.v_mag / b.voltage_level_id.map(vl)
    return {ids[i]: x for i, x in v.items() if i in ids and x == x}, b, v


def tr_load(n: Any, t: Any, units: dict[str, Any]) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Loading of the lumped and step-up transformers, and the lumped N-1 loading."""
    c = n.get_2_windings_transformers()
    ids = dict(zip(_ids("T", t.power_transformer.mrid), t.power_transformer.mrid, strict=True))
    tt = t.power_transformer.set_index("mrid")
    m = c.index.map(ids)
    r = tt.rated_current_a.reindex(m).to_numpy()
    s = pd.Series(c.i1.to_numpy() / r, index=m)
    lu = s[s.index.str.startswith("lumped/tr")]
    su = s[s.index.str.startswith("stepup/")]
    u = lu.index.str.removeprefix("lumped/tr/").map(units)
    n1 = lu * u / (u - 1).where(u > 1, np.nan)
    return lu, su, n1


def metrics(n: Any, t: Any, units: dict[str, Any]) -> tuple[dict[str, Any], pd.Series, pd.Series]:
    """Voltage, angle, loss and loading metrics of a solved pypowsybl network."""
    _, b, v = volt(n, t)
    vn = b.voltage_level_id.map(n.get_voltage_levels().nominal_v)
    ok = v.notna()
    a = n.get_lines()
    c = n.get_2_windings_transformers()
    loss = float((a.p1 + a.p2).sum() + (c.p1 + c.p2).sum())
    gens = n.get_generators()
    gen = -gens.p[gens.p < 0].sum()
    lines = SL(n, t)
    lu, su, n1 = tr_load(n, t, units)
    ehv = (vn >= 200) & ok
    d = {
        "buses": int(ok.sum()),
        "v380_min": round(v[ehv].min(), 3),
        "v380_max": round(v[ehv].max(), 3),
        "v110_min": round(v[(vn == 110) & ok].min(), 3),
        "v110_max": round(v[(vn == 110) & ok].max(), 3),
        "vt_min": round(v[(vn < 100) & ok].min(), 3),
        "vt_max": round(v[(vn < 100) & ok].max(), 3),
        "ang": round(b.v_angle[ok].max() - b.v_angle[ok].min(), 1),
        "ang_ehv": round(b.v_angle[ehv].max() - b.v_angle[ehv].min(), 1),
        "loss": round(loss),
        "loss_pct": round(100 * loss / gen, 2),
        "over": int((lines > 1).sum()),
        "maxl": round(float(lines.max()), 2),
        "lum_over": int((lu > 1).sum()),
        "lum_max": round(float(lu.max()), 2),
        "lum_n1_over": int((n1 > 1).sum()),
        "su_over": int((su > 1).sum()),
        "su_max": round(float(su.max()), 2),
    }
    return d, lines, lu


def pand(t: Any) -> tuple[Any, str]:
    """pandapower AC load flow with an outer loop that fixes generators at their Q limits."""
    n = PN(t, voltage_control="table", q_limit_ratio=1.0)
    g = t.synchronous_machine
    on = g[g.voltage_control.astype(bool) & g.in_service.astype(bool)]
    up = on.groupby("bus_mrid").q_max_mvar.sum()
    dn = on.groupby("bus_mrid").q_min_mvar.sum()
    for i, r in n.gen.iterrows():
        if r["name"] in t.ext:
            n.gen.loc[i, ["min_q_mvar", "max_q_mvar"]] = [-t.ext[r["name"]], t.ext[r["name"]]]
        elif r["name"] in up.index:
            n.gen.loc[i, ["min_q_mvar", "max_q_mvar"]] = [dn[r["name"]], up[r["name"]]]
    try:
        pp.runpp(n, distributed_slack=True, max_iteration=60, init="dc", trafo_model="pi")
    except Exception:
        return n, "base_fail"
    for _ in range(80):
        q = n.res_gen.q_mvar
        v = (
            pd.concat([q - n.gen.max_q_mvar, n.gen.min_q_mvar - q], axis=1)
            .max(axis=1)
            .where(n.gen.in_service, 0)
        )
        bad = v[v > 1].sort_values(ascending=False)
        if not len(bad):
            return n, "ok"
        for i in bad.index[: max(1, len(bad) // 8)]:
            lim = n.gen.max_q_mvar[i] if q[i] > n.gen.max_q_mvar[i] else n.gen.min_q_mvar[i]
            n.gen.loc[i, "in_service"] = False
            pp.create_sgen(
                n, n.gen.bus[i], p_mw=float(n.res_gen.p_mw[i]), q_mvar=float(lim), name="lim"
            )
        try:
            pp.runpp(n, distributed_slack=True, max_iteration=60, init="results", trafo_model="pi")
        except Exception:
            try:
                pp.runpp(n, distributed_slack=True, max_iteration=80, init="dc", trafo_model="pi")
            except Exception:
                return n, "qloop_fail"
    return n, "qloop_max"


def _run_tag(tag: str) -> None:
    data = pickle.load(open(OUT_DIR / f"input_{tag}.pkl", "rb"))  # noqa: SIM115
    units = dict(zip(data["infeed"]["bus_hv"], data["infeed"]["units"], strict=True))
    rows, over = [], []
    for h, d in data["H"].items():
        for tan in (None, 0.33, 0.2, 0.1):
            t = tables(d)
            lim = tan is not None
            t = qlim(t, tan if lim else 0.329)
            row: dict[str, Any] = {"tag": tag, "h": h, "tan": tan}
            st = "?"
            for _ in range(4):
                n, st = sy(t, lim)
                if st != "CONVERGED":
                    break
                v, _b, _v = volt(n, t)
                t2 = set_taps(t, v, 1.02)
                t2.ext = t.ext
                if t2.power_transformer.rated_u2_kv.equals(t.power_transformer.rated_u2_kv):
                    break
                t = t2
            if st == "CONVERGED":
                n, st = sy(t, lim)
            row["sy"] = st
            lines = None
            if st == "CONVERGED":
                mm, lines, _lu = metrics(n, t, units)
                row.update(mm)
                for i, x in lines[lines > 1].items():
                    over.append((tag, h, tan, i, round(x, 2)))
            if lim:
                net, ps_ = pand(t)
            else:
                net = PN(t, voltage_control="table", q_limit_ratio=1.0)
                try:
                    pp.runpp(
                        net, distributed_slack=True, max_iteration=60, init="dc", trafo_model="pi"
                    )
                    ps_ = "ok"
                except Exception:
                    ps_ = "base_fail"
            row["pp"] = ps_
            if ps_ == "ok":
                vm = net.res_bus.vm_pu.dropna()
                row.update(
                    pp_vmin=round(vm.min(), 3),
                    pp_vmax=round(vm[vm.index.map(net.bus.vn_kv) >= 100].max(), 3),
                    pp_loss=round(net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum()),
                    pp_ang=round(net.res_bus.va_degree.max() - net.res_bus.va_degree.min(), 1),
                    pp_over=int((PL(net) > 1).sum()),
                )
                if lines is not None:
                    d2 = (PL(net).reindex(lines.index) - lines).abs()
                    row["dl_max"] = round(float(d2.max()), 3)
            rows.append(row)
            print(row, flush=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / f"results_{tag}.csv", index=False)
    pd.DataFrame(over, columns=["tag", "h", "tan", "line", "load"]).to_csv(
        OUT_DIR / f"overloads_{tag}.csv", index=False
    )


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        _run_tag(arg)
