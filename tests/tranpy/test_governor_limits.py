"""Governor limits follow the scenario's dispatch scaling (test e10 / F1b, hygov check)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from tests.tranpy.test_grid_case import _connected

from tranpy.simulation.andes import (
    _record_init_clamps,
    clamped_governors,
    shift_governor_limits,
)


def _fake(at: float = 1.0) -> SimpleNamespace:
    pv = SimpleNamespace(
        idx=SimpleNamespace(v=["G0", "X1"]), p0=SimpleNamespace(v=np.array([2.2, -0.5]))
    )
    gen = SimpleNamespace(idx=SimpleNamespace(v=["M0"]), gen=SimpleNamespace(v=["G0"]))
    gcls = SimpleNamespace(idx=SimpleNamespace(v=["XM1"]), gen=SimpleNamespace(v=["X1"]))

    def par(x: list[float]) -> SimpleNamespace:
        return SimpleNamespace(v=np.array(x, dtype=float))

    tgov = SimpleNamespace(
        n=2,
        syn=SimpleNamespace(v=["M0", "XM1"]),
        params={"VMAX": par([2.03, -0.3]), "VMIN": par([1.97, -0.7])},
    )
    hygov = SimpleNamespace(
        n=1,
        syn=SimpleNamespace(v=["M0"]),
        At=SimpleNamespace(v=np.array([at])),
        params={"GMAX": par([2.13]), "GMIN": par([0.0])},
    )
    return SimpleNamespace(PV=pv, GENROU=gen, GENCLS=gcls, TGOV1=tgov, HYGOV=hygov)


def test_limits_move_with_the_setpoint() -> None:
    ss = _fake(at=1.25)
    base = np.array([2.0, -0.4])  # the scenario scaled G0 by 1.1 and X1 by 1.25
    shift_governor_limits(ss, base)
    assert ss.TGOV1.params["VMAX"].v == pytest.approx([2.23, -0.4])
    # German VMIN stays >= 0 rule only binds at 0; a negative neighbour limit moves unclipped
    assert ss.TGOV1.params["VMIN"].v == pytest.approx([2.17, -0.8])
    assert ss.HYGOV.params["GMAX"].v == pytest.approx([2.13 + 0.2 / 1.25])
    assert ss.HYGOV.params["GMIN"].v == pytest.approx([0.16])


def test_lower_limit_does_not_go_negative() -> None:
    ss = _fake()
    shift_governor_limits(ss, np.array([4.2, -0.5]))  # G0 scaled down by 2.0
    assert ss.HYGOV.params["GMIN"].v[0] == 0.0
    assert ss.TGOV1.params["VMIN"].v[0] == 0.0
    assert ss.TGOV1.params["VMIN"].v[1] == pytest.approx(-0.7)  # X1 unchanged


def _run(tmp_path, fuel: str, op: float, shift: bool, demand_only: bool = False):  # type: ignore[no-untyped-def]
    andes = pytest.importorskip("andes")
    from tranpy.simulation.grid_case import build_connected_case, continental_europe_equivalent

    tables = _connected()
    tables.synchronous_machine = tables.synchronous_machine.assign(
        fuel=[fuel, "Windenergie (Onshore-Anlage)"]
    )
    path = build_connected_case(
        tables,
        {"a": 1.02, "b": 1.0, "x": 1.0},
        {"a": (200.0, 30.0)},
        {"a": (50.0, 0.0)},
        {"x": -10.0},
        tmp_path / f"{fuel}.xlsx",
        neighbour_equivalent=continental_europe_equivalent(50_000.0),
        german_fcr_mw=3.0,  # a +-1 % band, as narrow as the per-unit shares of the v4.1 cases
    )
    ss = andes.load(str(path), setup=True, no_output=True, default_config=True)
    base = np.array(ss.PV.p0.v, dtype=float)
    ss.PQ.p0.v[:] *= op
    ss.PQ.q0.v[:] *= op
    if not demand_only:
        ss.PV.p0.v[:] *= op  # as AndesEngine does for operating_point
    if shift:
        shift_governor_limits(ss, base)
    assert ss.PFlow.run()
    if shift:
        shift_governor_limits(ss, base, slack=True)
    ss.TDS.config.tf = 1.0
    ss.TDS.config.no_tqdm = 1
    clamped = _record_init_clamps(ss)
    assert ss.TDS.run()
    a = np.degrees(ss.dae.ts.y[:, list(ss.Bus.a.a)])
    a = a - a[:, [-1]]  # relative to the neighbour bus
    return clamped, float(np.abs(a[-1] - a[0]).max())


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("fuel", ["Erdgas", "Pumpspeicher"])
@pytest.mark.parametrize("op", [0.9, 0.95, 1.05, 1.1])
def test_no_clamp_and_flat_pre_fault_after_scaling(tmp_path, fuel: str, op: float) -> None:  # type: ignore[no-untyped-def]
    clamped, drift = _run(tmp_path, fuel, op, shift=True)
    assert clamped == []
    assert drift < 0.05


@pytest.mark.filterwarnings("ignore")
def test_fixed_limits_clamp_after_scaling(tmp_path) -> None:  # type: ignore[no-untyped-def]
    clamped, _ = _run(tmp_path, "Pumpspeicher", 1.1, shift=False)
    assert clamped and clamped[0][0] == "HYGOV"
    assert clamped_governors  # exported helper


@pytest.mark.filterwarnings("ignore")
def test_slack_governor_follows_the_solved_slack_power(tmp_path) -> None:  # type: ignore[no-untyped-def]
    andes = pytest.importorskip("andes")
    from tranpy.simulation.grid_case import build_connected_case, continental_europe_equivalent

    path = build_connected_case(
        _connected(),
        {"a": 1.02, "b": 1.0, "x": 1.0},
        {"a": (200.0, 30.0)},
        {"a": (50.0, 0.0)},
        {"x": -10.0},
        tmp_path / "slack.xlsx",
        neighbour_equivalent=continental_europe_equivalent(50_000.0),
    )
    ss = andes.load(str(path), setup=True, no_output=True, default_config=True)
    base = np.array(ss.PV.p0.v, dtype=float)
    ss.PQ.p0.v[:] *= 1.3  # demand change without re-dispatch: the slack neighbour takes it up
    assert ss.PFlow.run()
    k = [str(s).startswith("XM") for s in ss.TGOV1.syn.v].index(True)
    vmax, vmin = float(ss.TGOV1.VMAX.v[k]), float(ss.TGOV1.VMIN.v[k])
    dp = float(ss.Slack.p.v[0] - ss.Slack.p0.v[0])
    assert abs(dp) > 0.1
    shift_governor_limits(ss, base, slack=True)
    assert ss.TGOV1.VMAX.v[k] == pytest.approx(vmax + dp)
    assert ss.TGOV1.VMIN.v[k] == pytest.approx(vmin + dp if vmin < 0 else max(vmin + dp, 0.0))
    ss.TDS.config.tf = 0.5
    ss.TDS.config.no_tqdm = 1
    clamped = _record_init_clamps(ss)
    assert ss.TDS.run()
    assert clamped == []
