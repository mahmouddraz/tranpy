"""CCT brackets from clearing-time sweeps (M4 groundwork)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tranpy.conventional.cct import cct_brackets


def test_brackets_status_and_monotonicity() -> None:
    t = [0.05, 0.1, 0.2, 0.3]
    frame = pd.DataFrame(
        {
            "line": np.repeat([1, 2, 3, 4], 4),
            "clearing_time_s": t * 4,
            "label": [*(0, 0, 1, 1), *(0, 0, 0, 0), *(1, 1, 1, 1), *(0, 1, 0, 1)],
        }
    )
    br = cct_brackets(frame, ["line"]).set_index("line")
    assert br.loc[1, "status"] == "bracketed"
    assert (br.loc[1, "t_stable_max"], br.loc[1, "t_unstable_min"]) == (0.1, 0.2)
    assert br.loc[1, "monotone"]
    assert br.loc[2, "status"] == "above_sweep" and np.isnan(br.loc[2, "t_unstable_min"])
    assert br.loc[3, "status"] == "below_sweep" and np.isnan(br.loc[3, "t_stable_max"])
    assert br.loc[4, "status"] == "bracketed" and not br.loc[4, "monotone"]
    assert br.loc[4, "width_s"] == 0.05
