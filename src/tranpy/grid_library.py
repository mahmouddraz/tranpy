"""Standard component values from the pandapower 3.5.6 library, and checks against them.

The library gives a citable value where the model has none, and a band to flag suspect values.
It never replaces a published value, and agreement with it shows plausibility, not verification.

Library lines are per conductor, while real 220 and 380 kV circuits are bundled. :func:`bundle_rule`
takes the conductor and bundle closest to the published ratings of each voltage; R and the rating
scale with the bundle and X and C are kept as listed, which is a simplification. The library has
no step-up or 380/220 kV transformer. Its 380/110 kV unit of 160 MVA is smaller than published
units of 300 to 350 MVA, so :func:`match_transformer` keeps the capacity of a site and chooses the
unit count to suit: uk is on the unit rating, so the ohmic value depends only on total capacity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

LIBRARY = "pandapower 3.5.6 standard types"
LIBRARY_MATCH = "library_match"
SIBLING_RATING = "sibling_rating"
STUB_RATING_SHARE = 0.5
SAME_SITE_KM = 1.0
PLACEHOLDER_RATING = 9000.0
SHORT_KM = 1.0
"""Per-km values of a shorter circuit say little: its impedance can be a placeholder."""
LENGTH_TOLERANCE = 0.15
IMPLIED_TOLERANCE = 0.35
SHARE_RATIO = 1.5
BANDS: Mapping[str, tuple[float, float]] = {"x": (0.7, 1.4), "b": (0.6, 1.5), "r": (0.5, 1.5)}
"""Allowed factors on the lowest and highest library value of a quantity."""
_OMEGA = 2 * np.pi * 50.0


@dataclass(frozen=True)
class LineType:
    """One conductor of the library, in ohm/km, nF/km and A."""

    name: str
    kv: int
    r: float
    x: float
    c: float
    i: float


@dataclass(frozen=True)
class TransformerType:
    """One transformer of the library."""

    name: str
    kv_hv: int
    kv_lv: int
    mva: float
    uk: float
    ur: float


LINE_TYPES = (
    LineType("490-AL1/64-ST1A 220.0", 220, 0.059, 0.285, 10.0, 960.0),
    LineType("679-AL1/86-ST1A 220.0", 220, 0.042, 0.275, 11.7, 1150.0),
    LineType("490-AL1/64-ST1A 380.0", 380, 0.059, 0.253, 11.0, 960.0),
    LineType("679-AL1/86-ST1A 380.0", 380, 0.042, 0.250, 14.6, 1150.0),
)
TRANSFORMER_TYPES = (
    TransformerType("160 MVA 380/110 kV", 380, 110, 160.0, 0.122, 0.0025),
    TransformerType("100 MVA 220/110 kV", 220, 110, 100.0, 0.12, 0.0026),
)


def level(kv: Any) -> int:
    """220 or 380 for a nominal voltage."""
    return 380 if float(kv) >= 300 else 220


def bundle_rule(rating_a: float, kv: Any, max_bundle: int = 4) -> tuple[LineType, int]:
    """Conductor and bundle size whose rating is closest to ``rating_a``, the smaller on a tie."""
    options = [
        (abs(line.i * n - rating_a), n, -line.i, k)
        for k, line in enumerate(LINE_TYPES)
        if line.kv == level(kv)
        for n in range(1, max_bundle + 1)
    ]
    _, n, _, k = min(options)
    return LINE_TYPES[k], n


def line_values(line: LineType, bundle: int, km: float) -> dict[str, float]:
    """Totals of a circuit: ``r_ohm``, ``x_ohm``, ``bch_s``, ``rated_current_a``."""
    return {
        "r_ohm": line.r / bundle * km,
        "x_ohm": line.x * km,
        "bch_s": _OMEGA * line.c * 1e-9 * km,
        "rated_current_a": line.i * bundle,
    }


def match_transformer(kv_hv: Any, site_mva: float) -> tuple[TransformerType, int]:
    """Library type and the unit count that keeps at least ``site_mva`` at the site."""
    kind = next(t for t in TRANSFORMER_TYPES if t.kv_hv == level(kv_hv))
    return kind, max(1, int(np.ceil(site_mva / kind.mva - 1e-9)))


def route_km(route: Any) -> float:
    """Length along a route of ``lon``, ``lat`` points, NaN without one."""
    if not isinstance(route, list | np.ndarray) or len(route) < 2:
        return float("nan")
    lon = np.array([float(p["lon"]) for p in route])
    lat = np.array([float(p["lat"]) for p in route])
    east = np.diff(lon) * np.cos(np.radians((lat[1:] + lat[:-1]) / 2)) * 111.2
    return float(np.hypot(east, np.diff(lat) * 111.2).sum())


def _length_class(row: Any, typical: float) -> tuple[str, str, float]:
    km, route, implied = row.length_km, row.route_km, row.implied_km
    proposed = float("nan")
    if not km > 0:
        return "flagged", "no length", proposed
    if km < SHORT_KM:
        return "unconfirmed", "stub under 1 km", proposed
    if implied > km * SHARE_RATIO and bool(row.tie_line):
        return "flagged", "tie-line share: length covers one operator", implied
    if implied < km * (1 - IMPLIED_TOLERANCE) and row.x_ohm <= 0.15:
        return "flagged", "placeholder reactance", proposed
    if abs(implied - km) > km * IMPLIED_TOLERANCE:
        return "flagged", f"reactance implies {implied:.1f} km at {typical} ohm/km", proposed
    if row.route_quality not in ("verified", "mapped", "osm_only") or route != route:
        return "unconfirmed", "no real route to measure", proposed
    if abs(route - km) > km * LENGTH_TOLERANCE:
        return "flagged", "route and stated length differ", proposed
    return "confirmed", "", proposed


def length_ledger(lines: pd.DataFrame) -> pd.DataFrame:
    """Route, stated and reactance-implied length per circuit, with a class and its cause.

    ``proposed_km`` is a proposal only and changes no value. HVDC rows have no usable reactance
    and are left out by the caller.
    """
    typical = {kv: float(np.mean([t.x for t in LINE_TYPES if t.kv == kv])) for kv in (220, 380)}
    table = lines[
        ["mrid", "name", "nominal_voltage_kv", "length_km", "x_ohm", "tie_line", "route_quality"]
    ].copy()
    table["route_km"] = lines["route"].map(route_km)
    table["typical_x"] = table["nominal_voltage_kv"].map(level).map(typical)
    table["implied_km"] = pd.to_numeric(table["x_ohm"], errors="coerce") / table["typical_x"]
    records: Any = table.itertuples()
    found = [_length_class(r, r.typical_x) for r in records]
    table["length_class"] = [f[0] for f in found]
    table["cause"] = [f[1] for f in found]
    table["proposed_km"] = [f[2] for f in found]
    return table.reset_index(drop=True)


def _band(quantity: str, kv: int) -> tuple[float, float]:
    types = [t for t in LINE_TYPES if t.kv == kv]
    if quantity == "b":
        values = [_OMEGA * t.c * 1e-3 for t in types]
    elif quantity == "x":
        values = [t.x for t in types]
    elif quantity == "r":
        values = [t.r / 4 for t in types] + [t.r for t in types]
    else:
        return min(t.i for t in types), 4 * max(t.i for t in types)
    low, high = BANDS[quantity]
    return min(values) * low, max(values) * high


def line_plausibility(lines: pd.DataFrame) -> pd.DataFrame:
    """Values per km and rating outside the library band, one row per value.

    Stubs are checked on the rating only. A flag keeps the value and asks for a look.
    """
    rows = []
    records: Any = lines.itertuples()
    for line in records:
        kv, km = level(line.nominal_voltage_kv), float(line.length_km)
        values = {"rating": float(line.rated_current_a)}
        if km >= SHORT_KM:
            values |= {
                "r": float(line.r_ohm) / km,
                "x": float(line.x_ohm) / km,
                "b": float(line.bch_s) * 1e6 / km,
            }
        for name, value in values.items():
            low, high = _band(name, kv)
            if not low <= value <= high:
                rows.append(
                    {
                        "mrid": line.mrid,
                        "name": line.name,
                        "kv": kv,
                        "data_quality": line.data_quality,
                        "quantity": name,
                        "value": value,
                        "band_low": low,
                        "band_high": high,
                        "factor": max(value / high, low / value) if value > 0 else np.inf,
                    }
                )
    columns = [
        "mrid", "name", "kv", "data_quality", "quantity", "value", "band_low", "band_high", "factor"
    ]  # fmt: skip
    return pd.DataFrame(rows, columns=columns)


def transformer_uk(transformers: pd.DataFrame) -> pd.DataFrame:
    """Short-circuit voltage implied by reactance and rating, compared with the library.

    ``uk = x * S / U**2`` with S from the rated current at side 1, to which the values refer. A
    voltage pair without a library type is compared with nothing and marked so.
    """
    table = transformers[["mrid", "name", "rated_u1_kv", "rated_u2_kv", "phase_shifter"]].copy()
    u1 = pd.to_numeric(table["rated_u1_kv"], errors="coerce")
    current = pd.to_numeric(transformers["rated_current_a"], errors="coerce")
    table["mva"] = np.sqrt(3) * u1 * current / 1000
    table["uk"] = pd.to_numeric(transformers["x_ohm"], errors="coerce") * table["mva"] / u1**2
    table["library"] = "no library type: 380/220 kV and phase shifters"
    return table.reset_index(drop=True)


def apply_library(tables: Any) -> tuple[Any, pd.DataFrame]:
    """The tables with library values on estimated circuits, and a record of each change.

    Published circuits are untouched. Estimated circuits with a published rating keep it, and
    their bundle follows it; circuits added from the export have no published rating, so theirs
    comes from the median of published circuits of that voltage through :func:`bundle_rule`.
    """
    lines = tables.ac_line_segment.copy()
    published = lines[lines["data_quality"].eq("published")]
    median = pd.to_numeric(published["rated_current_a"], errors="coerce").groupby(
        published["nominal_voltage_kv"].map(level)
    ).median()  # fmt: skip
    records = []
    for index, line in lines[lines["data_quality"].eq("estimated")].iterrows():
        km = float(line["length_km"])
        if not km > 0 or line["bus1_mrid"] == line["bus2_mrid"]:
            continue
        added = line["route_quality"] == "osm_only"
        target = (
            median[level(line["nominal_voltage_kv"])] if added else float(line["rated_current_a"])
        )
        kind, bundle = bundle_rule(float(target), line["nominal_voltage_kv"])
        values = line_values(kind, bundle, km)
        if not added:
            values["rated_current_a"] = float(line["rated_current_a"])
        before = {name: float(line[name]) for name in values}
        for name, value in values.items():
            lines.loc[index, name] = value
        lines.loc[index, "data_quality"] = LIBRARY_MATCH
        records.append(
            {
                "mrid": line["mrid"],
                "name": line["name"],
                "source": f"{LIBRARY_MATCH}: {kind.name} x{bundle}",
                "rating_kept": not added,
                "length_km": km,
                **{f"{k}_before": v for k, v in before.items()},
                **{f"{k}_after": v for k, v in values.items()},
            }
        )
    return replace(tables, ac_line_segment=lines), pd.DataFrame(records)


def infeed_library(infeed: pd.DataFrame) -> pd.DataFrame:
    """Library type and unit count per infeed row; rated export units keep their capacity."""
    records: Any = infeed.itertuples()
    rows = [match_transformer(r.kv_hv, float(r.mva)) for r in records]
    return infeed.assign(
        library_type=[kind.name for kind, _ in rows],
        library_units=[units for _, units in rows],
        library_uk=[kind.uk for kind, _ in rows],
        source=[f"{LIBRARY_MATCH}, capacity kept" for _ in rows],
    )


def _override(line: Any, field: str, new: float, reason: str) -> dict[str, Any]:
    return {
        "mrid": line["mrid"],
        "name": line["name"],
        "field": field,
        "previous": float(line[field]),
        "new": new,
        "reason": reason,
    }


def resolve_flagged(tables: Any, ledger: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
    """Owner decisions on the ledger's flags; returns the tables and each value replaced.

    A placeholder reactance takes the library value and its circuit becomes ``library_match``,
    unless both ends are one site, where the stated length is what is wrong. A
    tie-line share takes the proposed length, with its published impedance unchanged. The previous
    values stay in the returned table.
    """
    lines = tables.ac_line_segment.copy()
    at = {m: i for i, m in zip(lines.index, lines["mrid"], strict=True)}
    records = []
    where = tables.busbar_section.set_index("mrid")[["lon", "lat"]]
    for row in ledger[ledger["cause"].eq("placeholder reactance")].itertuples():
        index = at[row.mrid]
        typical: Any = row.typical_x
        km: Any = row.length_km
        line: Any = lines.loc[index]
        ends = where.loc[[line["bus1_mrid"], line["bus2_mrid"]]].to_numpy(dtype=float)
        apart = np.hypot(*((ends[0] - ends[1]) * np.array([0.63 * 111.2, 111.2])))
        if apart < SAME_SITE_KM:
            # two yards of one site: the reactance is right, the length is the whole circuit's
            short = float(line["x_ohm"]) / float(typical)
            reason = "yard coupling: stated length is the whole circuit, reactance kept"
            records.append(_override(line, "length_km", short, reason))
            lines.loc[index, "length_km"] = short
            continue
        x = float(typical) * float(km)
        records.append(_override(line, "x_ohm", x, f"{LIBRARY_MATCH}: placeholder reactance"))
        lines.loc[index, "x_ohm"] = x
        lines.loc[index, "data_quality"] = LIBRARY_MATCH
    for row in ledger[ledger["proposed_km"].notna()].itertuples():
        index = at[row.mrid]
        reason = "tie-line share: length of the whole circuit, impedance unchanged"
        line = lines.loc[index]
        proposed: Any = row.proposed_km
        records.append(_override(line, "length_km", float(proposed), reason))
        lines.loc[index, "length_km"] = float(proposed)
    columns = ["mrid", "name", "field", "previous", "new", "reason"]
    return replace(tables, ac_line_segment=lines), pd.DataFrame(records, columns=columns)


def correct_stub_ratings(
    tables: Any, share: float = STUB_RATING_SHARE, km: float = SHORT_KM
) -> tuple[Any, pd.DataFrame]:
    """Tee stubs far below their circuit take the circuit's rating; returns the values replaced.

    A stub is one circuit with the segments at its tee, so it takes the lowest rating among them.
    A stub without a sibling keeps its value: the library band only reports on it. Only the rating
    changes, and the circuit becomes ``sibling_rating``.
    """
    lines = tables.ac_line_segment.copy()
    tee = set(
        tables.busbar_section.loc[
            tables.busbar_section["name"].astype(str).str.startswith("y-"), "mrid"
        ]
    )
    records = []
    for index, line in lines[lines["length_km"].astype(float) < km].iterrows():
        at_tee = {line["bus1_mrid"], line["bus2_mrid"]} & tee
        siblings = lines[
            (lines["bus1_mrid"].isin(at_tee) | lines["bus2_mrid"].isin(at_tee))
            & (lines["mrid"] != line["mrid"])
            & (lines["length_km"].astype(float) >= km)
        ]
        if siblings.empty:
            continue
        rating = float(siblings["rated_current_a"].min())
        if float(line["rated_current_a"]) >= share * float(siblings["rated_current_a"].max()):
            continue
        low, high = _band("rating", level(line["nominal_voltage_kv"]))
        reason = (
            f"{SIBLING_RATING}: tee stub {float(line['rated_current_a']):.0f} A below its circuit;"
            f" library band {low:.0f}-{high:.0f} A"
        )
        records.append(_override(line, "rated_current_a", rating, reason))
        lines.loc[index, "rated_current_a"] = rating
        lines.loc[index, "data_quality"] = SIBLING_RATING
    columns = ["mrid", "name", "field", "previous", "new", "reason"]
    return replace(tables, ac_line_segment=lines), pd.DataFrame(records, columns=columns)


def correct_placeholder_ratings(tables: Any) -> tuple[Any, pd.DataFrame]:
    """A rating of 9,999 A marks no value: such a circuit takes the largest library bundle of 3."""
    lines = tables.ac_line_segment.copy()
    records = []
    placeholder = lines["rated_current_a"].astype(float) >= PLACEHOLDER_RATING
    for index, line in lines[placeholder].iterrows():
        kv = level(line["nominal_voltage_kv"])
        kind = max((k for k in LINE_TYPES if k.kv == kv), key=lambda k: k.i)
        rating = kind.i * 3
        published = float(line["rated_current_a"])
        reason = f"{LIBRARY_MATCH}: {kind.name} x 3; {published:.0f} A is a placeholder"
        records.append(_override(line, "rated_current_a", rating, reason))
        lines.loc[index, "rated_current_a"] = rating
        lines.loc[index, "data_quality"] = LIBRARY_MATCH
    columns = ["mrid", "name", "field", "previous", "new", "reason"]
    return replace(tables, ac_line_segment=lines), pd.DataFrame(records, columns=columns)


def rating_review(
    plausibility: pd.DataFrame, lines: pd.DataFrame, factor: float = 1.5
) -> pd.DataFrame:
    """Published ratings of real lines far outside the library band, for the owner; none changed."""
    long = lines.loc[lines["length_km"].astype(float) >= SHORT_KM, "mrid"]
    chosen = (
        plausibility["quantity"].eq("rating")
        & plausibility["data_quality"].eq("published")
        & plausibility["mrid"].isin(long)
        & plausibility["factor"].ge(factor)
    )
    return plausibility[chosen].sort_values("factor", ascending=False).reset_index(drop=True)


def library_tables(tables: Any, hvdc_eics: set[str]) -> tuple[Any, dict[str, pd.DataFrame]]:
    """The tables with library defaults and owner decisions applied, and their silver tables."""
    lines = tables.ac_line_segment
    ac = lines[~lines["eic"].isin(hvdc_eics) & (lines["bus1_mrid"] != lines["bus2_mrid"])]
    ledger = length_ledger(ac)
    plausibility = line_plausibility(ac)
    uk = transformer_uk(tables.power_transformer)
    with_library, changes = apply_library(tables)
    resolved, overrides = resolve_flagged(with_library, ledger)
    final, stubs = correct_stub_ratings(resolved)
    final, placeholders = correct_placeholder_ratings(final)
    overrides = pd.concat([overrides, stubs, placeholders], ignore_index=True)
    return final, {
        "grid_de_rating_review": rating_review(plausibility, ac),
        "grid_de_length_ledger": ledger,
        "grid_de_plausibility": plausibility,
        "grid_de_transformer_uk": uk,
        "grid_de_library_changes": changes,
        "grid_de_value_overrides": overrides,
    }
