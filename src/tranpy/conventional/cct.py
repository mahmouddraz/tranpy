"""M4 groundwork: critical-clearing-time brackets from clearing-time sweeps.

The "exact" CCT of a fault is the longest clearing time for which the time-domain
simulation stays stable, found by repeated simulation (e.g. bisection). A sweep that has
already simulated the same fault at several clearing times brackets it: the CCT lies
between the longest stable clearing time below the first unstable one and that unstable
clearing time. A sweep is *monotone* if no stable clearing time follows an unstable one;
otherwise the bracket is reported but flagged (multi-swing or numerical effects).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from tranpy_core.errors import DataError

__all__ = ["cct_brackets"]


def cct_brackets(
    events: pd.DataFrame,
    by: Sequence[str],
    *,
    time_col: str = "clearing_time_s",
    label_col: str = "label",
) -> pd.DataFrame:
    """CCT bracket of every sweep (one group of ``by``), label 1 = unstable.

    Returns one row per group with ``n``, ``n_unstable``, ``t_stable_max`` (longest stable
    clearing time below the first unstable one, NaN if none), ``t_unstable_min`` (first
    unstable clearing time, NaN if all stable), ``monotone``, ``width_s`` and ``status``:
    ``"bracketed"``, ``"below_sweep"`` (unstable already at the shortest time) or
    ``"above_sweep"`` (stable at every time).
    """
    missing = [c for c in (*by, time_col, label_col) if c not in events]
    if missing:
        raise DataError(f"columns missing: {missing}")
    rows = []
    for key, g in events.groupby(list(by), sort=True):
        g = g.sort_values(time_col)
        t = g[time_col].to_numpy(np.float64)
        y = g[label_col].to_numpy().astype(int)
        unstable = np.flatnonzero(y == 1)
        first = int(unstable[0]) if unstable.size else None
        t_u = float(t[first]) if first is not None else float("nan")
        below = t[: first if first is not None else len(t)]
        t_s = float(below.max()) if below.size else float("nan")
        monotone = first is None or bool(np.all(y[first:] == 1))
        if first is None:
            status = "above_sweep"
        elif first == 0:
            status = "below_sweep"
        else:
            status = "bracketed"
        keys = key if isinstance(key, tuple) else (key,)
        rows.append(
            {
                **dict(zip(by, keys, strict=True)),
                "n": len(g),
                "n_unstable": int(unstable.size),
                "t_stable_max": t_s,
                "t_unstable_min": t_u,
                "monotone": monotone,
                "width_s": t_u - t_s,
                "status": status,
            }
        )
    return pd.DataFrame(rows)
