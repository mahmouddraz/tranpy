"""How generators and load connect to the 220/380 kV grid, without any 110 kV line.

Every plant follows its published connection level (MaStR ``voltage_level``):

* class A, extra-high voltage synchronous plants: a terminal bus, where the machine holds voltage,
  and a step-up transformer to the 220/380 kV bus. Machines of one node and fuel merge;
* class B, 110 kV synchronous plants of at least ``class_b_min_mw`` after merging by site: a
  terminal bus and a step-up transformer to the lumped 110 kV bus of their infeed;
* class C, everything else, including solar, onshore wind and feed-in: a static injection on the
  lumped bus, without voltage control;
* class D, offshore wind and inverters registered at extra-high voltage: a static injection on
  the 220/380 kV node. Plants abroad stay on their border equivalent.

A lumped bus stands for one site with EHV/110 kV transformers, its count coming from the export.
It ignores voltage drops inside the 110 kV network and flow between catchments: it is an
aggregation, not an equivalent of that network. A plant without a level counts as class C, which
is an assumption.

:data:`EHV_110` holds dataset values (SimBench 1.6.3), used where the export has no rating. The
values of :class:`ConnectionParameters` for step-up transformers and machines have no citable
source: they are assumptions for the owner to confirm, and are meant to be swept.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import linprog
from scipy.spatial import cKDTree

from tranpy_core._optional import require
from tranpy_core.errors import DataError

KV_110 = 110.0
EHV_LEVELS = ("Höchstspannung", "Umspannebene Höchstspannung/Hochspannung")
HV_LEVEL = "Hochspannung"
ASSUMPTION = "assumption, uncited, owner to confirm"
EHV_110: Mapping[int, Mapping[str, float]] = {
    380: {"mva": 350.0, "uk": 0.22, "ur": 0.00257},
    220: {"mva": 300.0, "uk": 0.12, "ur": 0.00128},
}
LIBRARY_UNIT_MVA: Mapping[int, float] = {380: 160.0, 220: 100.0}
"""pandapower standard types 160 MVA 380/110 kV and 100 MVA 220/110 kV; ``type_units`` of a site
is its capacity in these units, not a count of real transformers."""
LIBRARY_TYPES: Mapping[int, Mapping[str, float]] = {
    380: {"mva": 160.0, "uk": 0.122, "ur": 0.0025, "pfe_kw": 60.0, "i0": 0.0006},
    220: {"mva": 100.0, "uk": 0.12, "ur": 0.0026, "pfe_kw": 55.0, "i0": 0.0006},
}
"""pandapower 3.5.6 standard types "160 MVA 380/110 kV" and "100 MVA 220/110 kV" per unit: rating,
uk, ur, iron losses and no-load current (vk 12.2/12.0 %, vkr 0.25/0.26 %, 60/55 kW, i0 0.06 %)."""
LIBRARY_MATCH = "library_match"
MEDIAN_MATCH = "median_published"
"""Rating of an unrated unit: the median published unit rating of its voltage pair."""
COUNT_LABELS = ("owner_checked", "published", "export", "default")
"""Evidence of a unit count, strongest first."""
LOAD_CAP = 0.8
"""Share of a site's transformer capacity its catchment load may take under ``rule="capped"``,
an assumption meant to be swept."""
MIN_AREA_SAMPLES = 5
"""Fewest published unit ratings of a voltage pair in a control area for its own median; with
fewer, the national median fills its unrated units."""
MAX_CATCHMENT_KM = 60.0
"""Longest distance from a cell to its infeed unless no site is closer: an assumption."""
TYPICAL_UNITS = 2
"""Units at a station with no tagged transformer: an assumption, labelled in ``rating_source``."""
TAP_STEP = 0.01
TAP_RANGE = 0.16
"""On-load tap changer of the lumped transformers, -16..16 x 1 % as in SimBench 1.6.3."""
STATION_BUFFER_DEGREES = 0.001
YARD_DISTANCE_DEGREES = 0.0065
"""About 500 m: a 110 kV yard this close to an EHV station is taken as the same site."""
LIGHT_LOAD_MW = 40000.0
_TEE = re.compile(r"^y-(.+?)#")
EHV_DISTANCE_KM = 2.5
CELL_DEGREES = 0.2
NEAREST_INFEEDS = 8
HEADROOM = 1.1
UNTAGGED = "untagged, voltage inferred"
TOUCH_DEGREES = 0.00003
"""About 2 m: a transformer symbol is mapped on the end of the busbars it joins."""
_KM = np.array([0.63 * 111.2, 111.2])
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_INVERTER = ("Solare Strahlungsenergie", "Windenergie", "Speicher")
_OFFSHORE = "Windenergie (Offshore-Anlage)"
_KEEP_LOADS = ("hvdc/", "pump/")


@dataclass(frozen=True)
class ConnectionParameters:
    """Values of the connection model; see the module description for their status.

    Attributes:
        class_b_min_mw: least capacity of a 110 kV plant site to get a machine of its own.
        step_up_uk: short-circuit voltage of step-up transformers, on the machine rating.
        step_up_x_r: reactance over resistance of step-up transformers.
        power_factor: rated power factor; the machine rating is committed MW over it.
        leading_limit: reactive absorption as a share of the machine rating.
        terminal_kv: nominal voltage of terminal buses.
        terminal_v_pu: voltage a machine holds at its terminal.
        ehv_110_uk: short-circuit voltage of 380/110 kV units, None for :data:`EHV_110`.
        machine_loading: dispatched power as a share of committed capacity.
        embedded_q_ratio: reactive limit of embedded units per MW dispatched; they then hold the
            lumped bus voltage, standing in for distribution-level reactive support. None makes
            them purely static. Without it the case does not solve, see ``soundness_problems``.
        lumped_v_pu: voltage embedded units and tap changers aim for on a lumped bus.
        load_cap: under ``rule="capped"``, the share of a site's transformer capacity its load
            may take before the rest spills to the next-nearest sites of the control area.
        load_cap_tan_phi: load tan phi for that cap; MVA is MW times sqrt(1 + tan phi**2).
        backfeed_cap: under ``rule="capped"``, also cap net backfeed (embedded generation minus
            load) at ``load_cap`` of a site's capacity, spilling the excess like load.
    """

    class_b_min_mw: float = 25.0
    step_up_uk: float = 0.15
    step_up_x_r: float = 40.0
    power_factor: float = 0.85
    leading_limit: float = 0.3
    terminal_kv: float = 21.0
    terminal_v_pu: float = 1.03
    ehv_110_uk: float | None = None
    machine_loading: float = 0.8
    embedded_q_ratio: float | None = 0.33
    lumped_v_pu: float = 1.02
    load_cap: float = LOAD_CAP
    load_cap_tan_phi: float = 0.2
    backfeed_cap: bool = False
    split_levels: bool = True


def _numbers(value: Any) -> list[float]:
    if value is None or value != value:
        return []
    return [float(x) for x in _NUMBER.findall(str(value))]


def has_110(voltages: Any) -> bool:
    """True when a list of volts holds a level of 100 to 119 kV."""
    return any(100e3 <= v <= 119e3 for v in _numbers(voltages))


def has_ehv(voltages: Any) -> bool:
    """True when a list of volts holds 220 kV or more."""
    return any(v >= 220e3 for v in _numbers(voltages))


def rating_mva(value: Any) -> float | None:
    """MVA of a ``rating`` tag such as ``300 MVA``; values under 50 are not grid couplers."""
    text = str(value).lower()
    numbers = [v for v in _numbers(text) if v >= 50] if "mva" in text else []
    return numbers[0] if numbers else None


def _level(kv: float) -> int:
    return 380 if kv >= 300 else 220


def is_inverter(fuel: Any) -> bool:
    """Solar, wind and storage."""
    return str(fuel).startswith(_INVERTER)


def plant_class(voltage_level: Any, fuel: Any) -> str:
    """``A``, ``B``, ``C`` or ``D`` of a plant, before the size rule for class B."""
    if str(fuel) == _OFFSHORE:
        return "D"
    if voltage_level in EHV_LEVELS:
        return "D" if is_inverter(fuel) else "A"
    if voltage_level == HV_LEVEL and not is_inverter(fuel):
        return "B"
    return "C"


def transformer_impedance(
    kv: float, mva: float, uk: float, ur: float, units: int = 1
) -> tuple[float, float]:
    """Series r and x in ohm at ``kv`` of ``units`` equal parallel transformers of ``mva`` each.

    Per unit on the rating, z = uk and r = ur, so z in ohm is ``uk * kv**2 / mva``.
    """
    if not (kv > 0 and mva > 0 and uk > ur >= 0 and units >= 1):
        raise DataError(f"invalid transformer: {kv} kV, {mva} MVA, uk {uk}, ur {ur}, {units} units")
    base = kv**2 / mva / units
    return ur * base, float(np.sqrt(uk**2 - ur**2)) * base


def infeed_transformers(
    stations: pd.DataFrame,
    transformers: pd.DataFrame,
    buses: pd.DataFrame,
    lines: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per model EHV bus with transformers, and the stations that matched no model bus.

    With ``lines`` (``voltages``, ``geometry_wkt``), an object without a full voltage tag counts
    only where it touches both an EHV and a 110 kV line or busbar; a unit inside the station that
    touches only one voltage is a 110 kV/MV or other unit.

    ``stations``: ``id``, ``name``, ``voltages``, ``geometry_wkt``. ``transformers``: ``lon``,
    ``lat``, ``v_primary``, ``v_secondary``, ``rating``. ``buses``: ``mrid``, ``site``, ``lon``,
    ``lat``, ``kv``, all of 220 kV or more. ``mva`` is the sum over the units of the row.
    """
    shapely = require("shapely", "grid")
    # the export often tags the two yards of one site as separate polygons
    is_ehv = stations["voltages"].map(has_ehv).to_numpy(dtype=bool)
    is_110 = stations["voltages"].map(has_110).to_numpy(dtype=bool)
    outlines = shapely.from_wkt(stations["geometry_wkt"].to_numpy())
    near_110 = np.zeros(len(stations), dtype=bool)
    if is_110.any() and is_ehv.any():
        found = shapely.STRtree(outlines[is_110]).query(
            outlines[is_ehv], predicate="dwithin", distance=YARD_DISTANCE_DEGREES
        )
        near_110[np.flatnonzero(is_ehv)[np.unique(found[0])]] = True
    both = stations[near_110].reset_index(drop=True)
    shapes = shapely.from_wkt(both["geometry_wkt"].to_numpy())
    centres = shapely.get_coordinates(shapely.centroid(shapes))
    primary = pd.to_numeric(transformers["v_primary"], errors="coerce")
    secondary = pd.to_numeric(transformers["v_secondary"], errors="coerce")
    is_tagged = (primary >= 220e3) & secondary.between(100e3, 119e3)
    tagged = transformers[is_tagged]
    published: dict[int, list[float]] = {}
    for volts, rating in zip(tagged["v_primary"], tagged["rating"], strict=True):
        value = rating_mva(rating)
        if value is not None:
            published.setdefault(_level(float(volts) / 1e3), []).append(value)
    unit_mva = {
        level: float(np.median(published[level])) if published.get(level) else EHV_110[level]["mva"]
        for level in EHV_110
    }
    levels = buses["kv"].map(_level).to_numpy()
    trees = {
        level: (buses[levels == level], cKDTree(buses[levels == level][["lon", "lat"]] * _KM))
        for level in sorted(set(levels))
    }
    units: dict[str, list[float | None]] = {}
    station_of: dict[str, set[str]] = {}
    evidence: dict[str, str] = {}
    unit_hits = (
        shapely.STRtree(shapes).query(
            shapely.points(tagged[["lon", "lat"]].to_numpy()),
            predicate="dwithin",
            distance=STATION_BUFFER_DEGREES,
        )
        if len(tagged)
        else np.zeros((2, 0), dtype=int)
    )
    home = dict(zip(unit_hits[0].tolist(), unit_hits[1].tolist(), strict=True))
    lost = 0
    for k, unit in enumerate(tagged.itertuples()):
        level = _level(float(cast(Any, unit).v_primary) / 1e3)
        if level not in trees:
            lost += 1
            continue
        frame, tree = trees[level]
        km, i = tree.query(np.array([unit.lon, unit.lat]) * _KM)
        if km > EHV_DISTANCE_KM:
            lost += 1
            continue
        bus_id = str(frame["mrid"].iloc[int(i)])
        units.setdefault(bus_id, []).append(rating_mva(unit.rating))
        evidence[bus_id] = "export"
        if k in home:
            station_of.setdefault(bus_id, set()).add(str(both["name"].iloc[home[k]]))
    # an object without a full voltage tag inside a station of both voltages is a candidate unit:
    # its voltage is inferred from the station, and its own rating tag is used where present
    untagged = transformers[
        ~is_tagged
        & (primary.isna() | (primary >= 220e3))
        & (secondary.isna() | secondary.between(100e3, 119e3))
        & ~(primary.notna() & secondary.notna())
    ]
    if lines is not None and len(untagged):
        wires = shapely.from_wkt(lines["geometry_wkt"].to_numpy())
        both_sides = np.ones(len(untagged), dtype=bool)
        spots = shapely.points(untagged[["lon", "lat"]].to_numpy())
        for side in (has_ehv, has_110):
            tree_side = shapely.STRtree(wires[lines["voltages"].map(side).to_numpy(dtype=bool)])
            touch = tree_side.query(spots, predicate="dwithin", distance=TOUCH_DEGREES)
            both_sides &= np.isin(np.arange(len(untagged)), touch[0])
        untagged = untagged[both_sides]
    points = shapely.points(untagged[["lon", "lat"]].to_numpy()) if len(untagged) else None

    def inside(polygons: Any) -> dict[int, list[float | None]]:
        if points is None:
            return {}
        hit = shapely.STRtree(polygons).query(
            points, predicate="dwithin", distance=STATION_BUFFER_DEGREES
        )
        found: dict[int, list[float | None]] = {}
        for unit, station in dict(zip(hit[0].tolist(), hit[1].tolist(), strict=True)).items():
            found.setdefault(station, []).append(rating_mva(untagged["rating"].iloc[unit]))
        return found

    open_units = inside(shapes)
    tree = cKDTree(buses[["lon", "lat"]].to_numpy() * _KM)
    missed = []
    for k, (_, station) in enumerate(both.iterrows()):
        near = tree.query_ball_point(centres[k] * _KM, EHV_DISTANCE_KM)
        if not near:
            missed.append(station[["id", "name", "voltages"]].to_dict())
            continue
        here = buses.iloc[near]
        extra = open_units.get(k, [])
        known = here["mrid"].astype(str)[here["mrid"].astype(str).isin(units)]
        if len(known):
            if extra:
                bus_id = str(known.iloc[0])
                units[bus_id] = units[bus_id] + extra
                evidence[bus_id] = f"{evidence[bus_id]} + {UNTAGGED}"
            continue
        target = here.sort_values("kv", ascending=False).iloc[0]
        bus_id = str(target["mrid"])
        units[bus_id] = list(extra) or [None] * TYPICAL_UNITS
        evidence[bus_id] = UNTAGGED if extra else "assumed"
        station_of.setdefault(bus_id, set()).add(str(station["name"]))
    # an EHV bus beside a 110 kV yard counts even when the export has no EHV outline there
    if is_110.any():
        beside = shapely.STRtree(outlines[is_110]).query(
            shapely.points(buses[["lon", "lat"]].to_numpy()),
            predicate="dwithin",
            distance=YARD_DISTANCE_DEGREES,
        )
        served = set(buses.loc[buses["mrid"].astype(str).isin(units), "site"])
        in_yard = inside(outlines[is_110])
        yard_of = dict(zip(beside[0].tolist(), beside[1].tolist(), strict=True))
        for k in np.unique(beside[0]):
            target = buses.iloc[int(k)]
            if target["site"] in served or _TEE.match(str(target["site"])):
                continue
            served.add(target["site"])
            extra = in_yard.get(yard_of[int(k)], [])
            units[str(target["mrid"])] = list(extra) or [None] * TYPICAL_UNITS
            evidence[str(target["mrid"])] = UNTAGGED if extra else "assumed"
    info = buses.set_index("mrid")
    # a tee is no site: its units belong to the station the tee is named after
    by_site = {str(s): str(m) for m, s in zip(buses["mrid"], buses["site"], strict=True)}
    for bus_id in list(units):
        tee = _TEE.match(str(info.at[bus_id, "site"]))
        parent = by_site.get(tee.group(1)) if tee else None
        if parent is None:
            continue
        ratings = units.pop(bus_id)
        tag = evidence.pop(bus_id)
        names = station_of.pop(bus_id, set())
        if parent not in units or (evidence[parent] != "export" and tag == "export"):
            units[parent], evidence[parent] = ratings, tag
        station_of.setdefault(parent, set()).update(names)
    rows = []
    for bus_id, ratings in units.items():
        kv = float(cast(Any, info.at[bus_id, "kv"]))
        level = _level(kv)
        known = [r for r in ratings if r is not None]
        mva = sum(known) + (len(ratings) - len(known)) * unit_mva[level]
        rows.append(
            {
                "site": info.at[bus_id, "site"],
                "bus_hv": bus_id,
                "kv_hv": kv,
                "units": len(ratings),
                "units_from_export": evidence[bus_id] == "export",
                "units_rated": len(known),
                "mva": mva,
                "rated_mva": float(sum(known)),
                "type_units": int(np.ceil(mva / LIBRARY_UNIT_MVA[level])),
                "stations": ";".join(sorted(station_of.get(bus_id, set()))),
                "count_evidence": evidence[bus_id],
            }
        )
    grouped = pd.DataFrame(rows).sort_values(["site", "bus_hv"]).reset_index(drop=True)
    grouped["rating_source"] = np.select(
        [
            grouped["units_rated"].eq(grouped["units"]),
            grouped["count_evidence"].eq("export"),
            grouped["count_evidence"].str.contains("untagged"),
        ],
        [
            "export",
            "estimated: export count, median published unit rating",
            "estimated: untagged units in the station, median published unit rating",
        ],
        "estimated: assumed count, median published unit rating",
    )
    grouped.attrs["units_beyond_distance"] = lost
    return grouped, pd.DataFrame(missed)


_ONE_UNIT = re.compile(r"\bone\b[^;]*\bunit\b", re.IGNORECASE)
CALIBRATED = "calibrated to published total"


def calibrate_to_totals(
    infeed: pd.DataFrame, area_of_bus: Mapping[str, Any], totals: Mapping[str, float]
) -> pd.DataFrame:
    """Scale the median-rated units of each area so its EHV/110 kV capacity matches a published
    total; published and export ratings stay as they are, and an area without a total is kept.
    """
    out = infeed.copy()
    area = out["bus_hv"].map(area_of_bus)
    estimated = out["rating_source"].astype(str).str.contains("median")
    for name, total in totals.items():
        here = area.eq(name).to_numpy()
        fixed = float(out.loc[here & ~estimated, "mva"].sum())
        loose = here & estimated.to_numpy()
        if not loose.any() or total <= fixed:
            continue
        out.loc[loose, "mva"] *= (total - fixed) / float(out.loc[loose, "mva"].sum())
        out.loc[loose, "rating_source"] = f"{CALIBRATED}: {name} {total:.0f} MVA"
    return out


def _value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def apply_site_units(infeed: pd.DataFrame, site_units: pd.DataFrame) -> pd.DataFrame:
    """Per-site transformer data where researched: published counts and ratings first; elsewhere
    the export count and the rating already in the row. Planned units and unknown counts change
    nothing, so a count is never invented.

    ``site_units``: ``node`` (the EHV bus) or ``site`` and ``kv_hv``; ``units``; ``unit_mva`` or
    ``mva`` (the site total); ``uk_percent`` or ``uk``; and the evidence, as ``source_type`` with
    ``source_url`` or as ``count_source`` and ``rating_source``.
    """
    out = infeed.copy()
    if "uk" not in out:
        out["uk"] = np.nan
    site_units = site_units.rename(columns={"uk_percent": "uk", "node": "bus_hv"})
    if "source_type" in site_units:
        evidence = site_units["source_type"].astype(str) + (
            ", " + site_units["source_url"].astype(str) if "source_url" in site_units else ""
        )
        site_units = site_units.assign(count_source=evidence, rating_source=evidence)
    keys = ["bus_hv"] if "bus_hv" in site_units else ["site", "kv_hv"]
    keys = [key for key in keys if key in site_units]
    for row in site_units.to_dict("records"):
        if str(row.get("count_source", "")).startswith(("planned", "none")):
            continue  # planned units are not existing ones, and an unknown count stays as it is
        hit = np.ones(len(out), dtype=bool)
        for key in keys:
            hit &= (out[key] == row[key]).to_numpy()
        if not hit.any():
            continue
        # the mean unit rating so far: published where the export rates a unit, else the median
        previous = (out.loc[hit, "mva"] / out.loc[hit, "units"]).astype(float)
        units = _value(row.get("units"))
        if pd.notna(units) and units > 0:
            out.loc[hit, "units"] = int(units)
            out.loc[hit, "count_evidence"] = str(row.get("count_source") or "site research")
        count = out.loc[hit, "units"].astype(float)
        unit_mva = _value(row.get("unit_mva"))
        total = _value(row.get("mva"))
        published = str(row.get("rating_source") or "site research")
        if pd.notna(total) and total > 0:
            out.loc[hit, "mva"] = float(total)
            out.loc[hit, "rating_source"] = published
        elif pd.notna(unit_mva) and unit_mva > 0:
            # a source naming "one" unit rates that unit only; the others keep their rating
            rated = 1.0 if _ONE_UNIT.search(str(row.get("notes", ""))) else count
            out.loc[hit, "mva"] = rated * float(unit_mva) + (count - rated) * previous
            out.loc[hit, "rating_source"] = published
        else:
            out.loc[hit, "mva"] = count * previous
        uk = _value(row.get("uk"))
        if pd.notna(uk) and uk > 0:
            out.loc[hit, "uk"] = float(uk) / 100 if uk > 1 else float(uk)
    return out


def _published(note: Any) -> str:
    text = str(note)
    return text if text.startswith("published") else f"published: {text}"


def unit_counts(
    infeed: pd.DataFrame,
    owner: pd.DataFrame | None = None,
    site_units: pd.DataFrame | None = None,
    owner_stations: Mapping[str, str] | None = None,
    buses: pd.DataFrame | None = None,
    ratings: str = "median",
    areas: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unit count, rating and library values per infeed row, and the owner rows matching none.

    Counts: ``owner`` (``site``, ``kv_hv``, ``kv_lv``, ``ehv_units``, ``evidence``, ``note``)
    first, then published rows of ``site_units`` (``source_type`` starting with "published"),
    then the export count as it is (no floor), then :data:`TYPICAL_UNITS` where the station has
    no count at all. ``count_label`` names which (:data:`COUNT_LABELS`). Ratings: a published
    rating, else the export tag of each rated unit, else (``ratings="median"``) the median
    published unit rating of the voltage pair (export tags and published unit ratings,
    ``MEDIAN_MATCH``), or with ``ratings="library"`` the library unit of :data:`LIBRARY_TYPES`
    (``library_match``, a sensitivity). ``uk``, ``ur``, ``g_s`` and ``b_s`` come from the
    library type for the row's capacity, a published uk first. ``owner_stations`` maps a site to
    the node site its units sit at. With ``buses`` (``mrid``, ``site``, ``kv``), an owner row at a
    voltage the export placed no units at gets a row at that bus, and an assumed row of the same
    site at a voltage the owner did not list is dropped. With ``areas`` (``bus_hv`` to control
    area), the median is taken per area and voltage pair where the area has at least
    :data:`MIN_AREA_SAMPLES` published unit ratings, else nationally.
    """
    out = infeed.reset_index(drop=True)
    if owner is not None and buses is not None:
        moved_to = dict(owner_stations or {})
        listed = {(moved_to.get(str(a), str(a)), float(b)) for a, b in zip(
            owner["site"], owner["kv_hv"], strict=True)}  # fmt: skip
        added = []
        for site, kv in sorted(listed):
            if ((out["site"] == site) & (out["kv_hv"] == kv)).any():
                continue
            bus = buses[buses["site"].eq(site) & buses["kv"].astype(float).eq(kv)]
            if bus.empty:
                continue
            added.append(
                {"site": site, "bus_hv": str(bus["mrid"].iloc[0]), "kv_hv": kv, "units": 0,
                 "units_rated": 0, "rated_mva": 0.0, "mva": 0.0, "count_evidence": "assumed",
                 "stations": "added for an owner-checked count"}
            )  # fmt: skip
        drop = out["count_evidence"].astype(str).eq("assumed").to_numpy() & np.array(
            [
                any(site == a for a, _ in listed) and (site, kv) not in listed
                for site, kv in zip(out["site"], out["kv_hv"].astype(float), strict=True)
            ],
            dtype=bool,
        )
        out = pd.concat([out[~drop], pd.DataFrame(added)], ignore_index=True)
    raw = out["count_evidence"].astype(str).to_numpy()
    default = raw == "assumed"
    units = np.where(default, TYPICAL_UNITS, out["units"].to_numpy()).astype(int)
    label = np.where(default, "default", "export").astype(object)
    evidence = np.where(
        default,
        f"default: {TYPICAL_UNITS} units, the station has no counted unit",
        "export: " + raw,
    ).astype(object)
    if "units_rated" not in out:
        out["units_rated"] = 0
    if "rated_mva" not in out:
        out["rated_mva"] = np.nan
    unit_mva = np.full(len(out), np.nan)
    total_mva = np.full(len(out), np.nan)
    rating_note = np.full(len(out), "", dtype=object)
    uk_published = np.full(len(out), np.nan)
    if site_units is not None:
        rows = site_units[site_units["source_type"].astype(str).str.startswith("published")]
        for row in rows.to_dict("records"):
            hit = (
                out["bus_hv"].eq(row["node"])
                if "node" in row
                else (out["site"].eq(row["site"]) & out["kv_hv"].eq(float(row["kv_hv"])))
            ).to_numpy()
            if not hit.any():
                continue
            text = f"{row['source_type']}, {row.get('source_url', '')}".rstrip(", ")
            count = _value(row.get("units"))
            if count > 0:
                units[hit], label[hit], evidence[hit] = int(count), "published", text
            if _value(row.get("unit_mva")) > 0:
                unit_mva[hit] = _value(row.get("unit_mva"))
                rating_note[hit] = text
            if _value(row.get("uk_percent")) > 0:
                uk_published[hit] = _value(row.get("uk_percent")) / 100
    missed = []
    if owner is not None:
        moved = dict(owner_stations or {})
        for row in owner.to_dict("records"):
            kv_lv = _value(row.get("kv_lv"))
            if pd.notna(kv_lv) and kv_lv != KV_110:
                missed.append({**row, "reason": "not an EHV/110 kV pair"})
                continue
            site = moved.get(str(row["site"]), str(row["site"]))
            hit = (out["site"].eq(site) & out["kv_hv"].eq(float(row["kv_hv"]))).to_numpy()
            if not hit.any():
                missed.append({**row, "reason": "no infeed row at this site and voltage"})
                continue
            note = "; ".join(str(row[k]) for k in ("evidence", "note") if pd.notna(row.get(k)))
            units[hit], label[hit] = int(row["ehv_units"]), "owner_checked"
            evidence[hit] = f"owner_checked: {note}"
            if _value(row.get("mva")) > 0:
                total_mva[hit] = _value(row.get("mva"))
                rating_note[hit] = str(row.get("rating_source", "published"))
    out = out.assign(units=units, count_label=label, count_evidence=evidence)
    if ratings not in ("median", "library"):
        raise ValueError(f"ratings must be 'median' or 'library', not {ratings!r}")
    fill = {level: lib["mva"] for level, lib in LIBRARY_TYPES.items()}
    fill_label = LIBRARY_MATCH
    if ratings == "median":
        fill_label = MEDIAN_MATCH
        seen: dict[tuple[Any, int], list[float]] = {}
        levels = out["kv_hv"].astype(float).map(_level).to_numpy()
        zone = out["bus_hv"].map(areas).to_numpy() if areas is not None else [None] * len(out)
        n_rated = pd.to_numeric(out["units_rated"], errors="coerce").fillna(0).astype(int)
        for z, level, k, total in zip(zone, levels, n_rated, out["rated_mva"], strict=True):
            if k > 0 and pd.notna(total):
                seen.setdefault((z, int(level)), []).extend([float(total) / k] * int(k))
        for z, level, value in zip(zone, levels, unit_mva, strict=True):
            if pd.notna(value):
                seen.setdefault((z, int(level)), []).append(float(value))
        national: dict[int, list[float]] = {}
        for (_, lv), values in seen.items():
            national.setdefault(lv, []).extend(values)
        fill = {lv: float(np.median(national[lv])) if national.get(lv) else fill[lv] for lv in fill}
        area_fill = {
            key: float(np.median(values))
            for key, values in seen.items()
            if key[0] is not None and key[0] == key[0] and len(values) >= MIN_AREA_SAMPLES
        }
        row_fill = [
            area_fill.get((z, int(lv)), fill[int(lv)]) for z, lv in zip(zone, levels, strict=True)
        ]
    else:
        row_fill = [fill[_level(float(kv))] for kv in out["kv_hv"]]
    mva: list[float] = []
    sources: list[str] = []
    uk: list[float] = []
    ur: list[float] = []
    g: list[float] = []
    b: list[float] = []
    types: list[int] = []
    records: Any = out.to_dict("records")
    for i, row in enumerate(records):
        kv = float(row["kv_hv"])
        lib = LIBRARY_TYPES[_level(kv)]
        n = int(row["units"])
        rated = int(row.get("units_rated", 0) or 0)
        if pd.notna(total_mva[i]):
            value, source = float(total_mva[i]), _published(rating_note[i])
        elif pd.notna(unit_mva[i]):
            value, source = n * float(unit_mva[i]), _published(rating_note[i])
        else:
            r = min(n, rated)
            per = float(row["rated_mva"]) / rated if rated else 0.0
            value = r * per + (n - r) * row_fill[i]
            source = "export" if r == n else fill_label if r == 0 else f"export + {fill_label}"
        mva.append(value)
        sources.append(source)
        uk.append(float(uk_published[i]) if pd.notna(uk_published[i]) else lib["uk"])
        ur.append(lib["ur"])
        conductance = lib["pfe_kw"] / 1000.0 * value / lib["mva"] / kv**2
        admittance = lib["i0"] * value / kv**2
        g.append(conductance)
        b.append(-float(np.sqrt(max(admittance**2 - conductance**2, 0.0))))
        types.append(int(np.ceil(value / lib["mva"] - 1e-9)))
    out = out.assign(mva=mva, rating_source=sources, uk=uk, ur=ur, g_s=g, b_s=b, type_units=types)
    return out, pd.DataFrame(missed)


def _cell(lon: Any, lat: Any) -> pd.Series:
    x = np.round(np.asarray(lon, dtype=float) / CELL_DEGREES).astype(int)
    y = np.round(np.asarray(lat, dtype=float) / CELL_DEGREES).astype(int)
    return pd.Series(x * 100000 + y)


def catchment_shares(
    points: pd.DataFrame,
    sites: pd.DataFrame,
    rule: str = "capacity",
    cap: float = LOAD_CAP,
    load_factor: float = 1.0,
    backfeed: bool = False,
) -> pd.DataFrame:
    """Share of each grid cell per infeed site.

    ``points``: ``lon``, ``lat``, ``zone``, ``mw``, the size of what must be served. ``sites``:
    ``site``, ``lon``, ``lat``, ``zone``, ``mva``. With ``rule="capacity"`` each control area
    minimises MW x km while no site holds more than its share of the area's size, times
    :data:`HEADROOM`; with ``"nearest"`` each cell goes to its nearest site in the area. With
    ``"capped"``, the ``"capacity"`` shares are followed by :func:`spill_over_cap`: load (rows of
    ``kind`` "load") beyond ``cap`` of a site's ``mva``, at ``load_factor`` MVA per MW, moves to
    the next-nearest sites of the area with room, without a distance limit. With ``backfeed``,
    net backfeed (rows of other kinds minus load) beyond the same cap then spills the same way
    (:func:`spill_backfeed`). Returns ``cell``, ``site``, ``share``, ``km``.
    """
    table = points.assign(cell=_cell(points["lon"], points["lat"]).to_numpy())
    if "kind" not in table:
        table["kind"] = "all"
    cells = table.groupby(["zone", "cell"], as_index=False).agg(
        mw=("mw", "sum"), lon=("lon", "mean"), lat=("lat", "mean")
    )
    sizes = table.pivot_table(index=["zone", "cell"], columns="kind", values="mw", aggfunc="sum")
    sizes = sizes.fillna(0.0)
    rows = []
    for zone, frame in cells.groupby("zone"):
        here = sites[sites["zone"].eq(zone)].reset_index(drop=True)
        if here.empty:
            raise DataError(f"control area {zone} has no infeed site")
        tree = cKDTree(here[["lon", "lat"]].to_numpy() * _KM)
        count = min(NEAREST_INFEEDS, len(here))
        km, near = tree.query(frame[["lon", "lat"]].to_numpy() * _KM, k=count)
        km, near = km.reshape(len(frame), count), near.reshape(len(frame), count)
        share = None
        if rule in ("capacity", "capped") and count > 1:
            keys = pd.MultiIndex.from_arrays([frame["zone"], frame["cell"]])
            share = _transport(
                frame["mw"].to_numpy(), sizes.loc[keys].to_numpy(), km, near, here["mva"].to_numpy()
            )
        if share is None:
            share = np.zeros_like(km)
            share[:, 0] = 1.0
        if rule == "capped" and count > 1 and "load" in sizes:
            # spill may reach every site of the area, nearest first
            wide = len(here)
            km, near = tree.query(frame[["lon", "lat"]].to_numpy() * _KM, k=wide)
            km, near = km.reshape(len(frame), wide), near.reshape(len(frame), wide)
            share = np.hstack([share, np.zeros((len(frame), wide - count))])
            count = wide
            keys = pd.MultiIndex.from_arrays([frame["zone"], frame["cell"]])
            share = spill_over_cap(
                sizes.loc[keys, "load"].to_numpy(),
                share,
                near,
                here["mva"].to_numpy(dtype=float) * cap / load_factor,
            )
            if backfeed:
                load = sizes.loc[keys, "load"].to_numpy()
                generation = sizes.loc[keys].drop(columns="load").sum(axis=1).to_numpy()
                share = spill_backfeed(
                    generation - load,
                    share,
                    near,
                    here["mva"].to_numpy(dtype=float) * cap / load_factor,
                )
        for i, cell in enumerate(frame["cell"]):
            for j in range(count):
                if share[i, j] > 1e-6:
                    rows.append(
                        {
                            "zone": zone,
                            "cell": cell,
                            "site": here["site"].iloc[near[i, j]],
                            "share": share[i, j],
                            "km": km[i, j],
                        }
                    )
    out = pd.DataFrame(rows)
    out["share"] = out["share"] / out.groupby(["zone", "cell"])["share"].transform("sum")
    return out


def spill_over_cap(
    load: npt.NDArray[Any],
    share: npt.NDArray[Any],
    near: npt.NDArray[Any],
    limit: npt.NDArray[Any],
) -> npt.NDArray[Any]:
    """Shares after moving load beyond each site's ``limit`` (MW) to the next-nearest sites.

    ``load`` per cell, ``share`` and ``near`` cell by candidate (nearest first), ``limit`` per
    site. An overloaded site gives up the same fraction of every cell it serves; each cell passes
    its part to its nearest candidates with room, in order of distance. A site that reached its
    limit takes nothing more, so the loop ends; load that finds no room stays where it was.
    """
    share = share.astype(float).copy()
    full = np.zeros(len(limit), dtype=bool)
    for _ in range(len(limit) + 1):
        held = np.bincount(near.ravel(), (share * load[:, None]).ravel(), minlength=len(limit))
        over = np.flatnonzero(held > limit * (1 + 1e-9) + 1e-9)
        full |= held >= limit * (1 - 1e-9)
        moved = False
        for site in over:
            rows, cols = np.nonzero((near == site) & (share > 0) & (load[:, None] > 0))
            served = share[rows, cols] * load[rows]
            if served.sum() <= 0:
                continue
            part = served * min(1.0, (held[site] - limit[site]) / served.sum())
            for i, j, mw in zip(rows, cols, part, strict=True):
                rest = mw
                for k in range(near.shape[1]):
                    other = near[i, k]
                    if other == site or full[other]:
                        continue
                    give = min(rest, limit[other] - held[other])
                    if give <= 0:
                        continue
                    share[i, k] += give / load[i]
                    share[i, j] -= give / load[i]
                    held[other] += give
                    held[site] -= give
                    rest -= give
                    moved = True
                    if held[other] >= limit[other] * (1 - 1e-9):
                        full[other] = True
                    if rest <= 1e-12:
                        break
        if not moved:
            break
    return np.asarray(np.clip(share, 0.0, None))


def spill_backfeed(
    net: npt.NDArray[Any],
    share: npt.NDArray[Any],
    near: npt.NDArray[Any],
    limit: npt.NDArray[Any],
) -> npt.NDArray[Any]:
    """Shares after moving net backfeed beyond each site's ``limit`` (MW) to the nearest sites.

    ``net`` per cell is generation minus load. The mirror of :func:`spill_over_cap`: cells that
    feed back move, while the load of net-consuming cells at a site absorbs backfeed and so
    raises that site's limit. Shares of net-consuming cells do not change.
    """
    net = np.asarray(net, dtype=float)
    back = np.clip(net, 0.0, None)
    absorbed = np.bincount(
        near.ravel(), (share * np.clip(-net, 0.0, None)[:, None]).ravel(), minlength=len(limit)
    )
    return spill_over_cap(back, share, near, np.asarray(limit, dtype=float) + absorbed)


def _transport(
    total: npt.NDArray[Any],
    kinds: npt.NDArray[Any],
    km: npt.NDArray[Any],
    near: npt.NDArray[Any],
    mva: npt.NDArray[Any],
) -> npt.NDArray[Any] | None:
    """Shares minimising MW x km; each kind of power fills the transformers evenly.

    Load, wind and solar peak at different hours, so each has its own limit per site. The limit
    relaxes from :data:`HEADROOM` until the problem is feasible.
    """
    sparse = require("scipy.sparse", "grid")
    n, k = km.shape
    cost = (km * total[:, None]).ravel() + 1e-6
    equal = sparse.csr_matrix((np.ones(n * k), (np.repeat(np.arange(n), k), np.arange(n * k))))
    blocks = []
    rooms = []
    for j in range(kinds.shape[1]):
        size = kinds[:, j]
        if size.sum() <= 0:
            continue
        blocks.append(
            sparse.csr_matrix(
                ((size[:, None] * np.ones((n, k))).ravel(), (near.ravel(), np.arange(n * k))),
                shape=(len(mva), n * k),
            )
        )
        rooms.append(mva * size.sum() / mva.sum())
    limit = sparse.vstack(blocks).tocsr()
    room = np.concatenate(rooms)
    allowed = km <= MAX_CATCHMENT_KM
    allowed[:, 0] = True
    bounds = [(0.0, 1.0 if ok else 0.0) for ok in allowed.ravel()]
    for headroom in (HEADROOM, 1.3, 1.6, 2.0, 3.0):
        solved = linprog(
            cost,
            A_ub=limit,
            b_ub=room * headroom,
            A_eq=equal,
            b_eq=np.ones(n),
            bounds=bounds,
        )
        if solved.success:
            return np.asarray(solved.x).reshape(n, k)
    return None


def _merge_sites(plants: pd.DataFrame) -> pd.Series:
    """Site key of named plants: same fuel within about one km."""
    x = (plants["lon"].astype(float) / 0.015).round().astype(int).astype(str)
    y = (plants["lat"].astype(float) / 0.01).round().astype(int).astype(str)
    return plants["fuel"].astype(str) + "@" + x + "_" + y


def connect(
    tables: Any,
    infeed: pd.DataFrame,
    plants: pd.DataFrame,
    attachments: pd.DataFrame,
    cells: pd.DataFrame,
    population: pd.DataFrame,
    areas: pd.Series,
    parameters: ConnectionParameters | None = None,
    rule: str = "capacity",
) -> tuple[Any, pd.Series, pd.DataFrame]:
    """Tables with the connection model applied to their current ``p_mw``.

    Totals of power and capacity per control area and fuel stay. Returns the tables, ``areas``
    extended by the new nodes, and a ledger of capacity and power per fuel, class and path.
    """
    par = parameters or ConnectionParameters()
    bus = tables.busbar_section.set_index("mrid")
    vl_sub = tables.voltage_level.set_index("mrid")["substation_mrid"]
    sites = infeed.sort_values("kv_hv", ascending=False).drop_duplicates("site").copy()
    sites["lon"] = sites["bus_hv"].map(bus["lon"])
    sites["lat"] = sites["bus_hv"].map(bus["lat"])
    sites["zone"] = sites["bus_hv"].map(bus["name"]).map(areas)
    sites["mva"] = sites["site"].map(infeed.groupby("site")["mva"].sum())
    sites = sites.dropna(subset=["zone"])
    infeed = infeed[infeed["site"].isin(sites["site"])]
    bar = {s: f"lumped/{s}" for s in sites["site"]}
    node_of = {s: f"{s}|110" for s in sites["site"]}
    new_levels, new_bars, new_units = [], [], []
    extended = dict(areas.items())

    def add_bus(mrid: str, node: str, site: str, kv: float, sub: Any, at: Any, kind: str) -> None:
        new_levels.append(
            {
                "mrid": f"vl/{mrid}",
                "site": site,
                "kv": kv,
                "substation_mrid": sub,
                "nominal_voltage_kv": kv,
            }
        )
        new_bars.append(
            {
                "mrid": mrid,
                "name": node,
                "voltage_level_mrid": f"vl/{mrid}",
                "kind": kind,
                "tso": None,
                "lon": at[0],
                "lat": at[1],
                "location_source": "model EHV node",
                "data_quality": ASSUMPTION,
            }
        )

    def add_unit(
        mrid: str,
        hv: str,
        lv: str,
        kv: float,
        lv_kv: float,
        mva: float,
        r: float,
        x: float,
        q: str,
        g: float = 0.0,
        b: float = 0.0,
    ) -> None:
        new_units.append(
            {
                "mrid": mrid,
                "name": mrid,
                "eic": None,
                "tso": None,
                "bus1_mrid": hv,
                "bus2_mrid": lv,
                "rated_u1_kv": kv,
                "rated_u2_kv": lv_kv,
                "r_ohm": r,
                "x_ohm": x,
                "b_s": b,
                "g_s": g,
                "rated_current_a": mva / kv / np.sqrt(3) * 1000,
                "phase_shifter": False,
                "data_quality": q,
            }
        )

    site_rows: Any = sites.itertuples()
    for row in site_rows:
        add_bus(
            bar[row.site],
            node_of[row.site],
            row.site,
            KV_110,
            vl_sub[bus.at[row.bus_hv, "voltage_level_mrid"]],
            (row.lon, row.lat),
            "lumped_110",
        )
        extended[node_of[row.site]] = row.zone
    infeed_rows: Any = infeed.itertuples()
    for row in infeed_rows:
        typical = EHV_110[_level(row.kv_hv)]
        uk = typical["uk"]
        if par.ehv_110_uk is not None and _level(row.kv_hv) == 380:
            uk = par.ehv_110_uk
        researched = getattr(row, "uk", None)
        if researched is not None and pd.notna(researched):
            uk = float(researched)
        ur = getattr(row, "ur", None)
        ur = float(ur) if ur is not None and pd.notna(ur) else typical["ur"]
        r, x = transformer_impedance(row.kv_hv, row.mva, uk, ur)
        g, b = (_value(getattr(row, k, 0.0)) for k in ("g_s", "b_s"))
        add_unit(
            f"lumped/tr/{row.bus_hv}",
            row.bus_hv,
            bar[row.site],
            row.kv_hv,
            KV_110,
            row.mva,
            r,
            x,
            row.rating_source,
            g if pd.notna(g) else 0.0,
            b if pd.notna(b) else 0.0,
        )

    level = plants.drop_duplicates("mastr_id").set_index("mastr_id")
    named = attachments.copy()
    named["level"] = named["mastr_id"].map(level["voltage_level"])
    named["cls"] = [plant_class(a, b) for a, b in zip(named["level"], named["fuel"], strict=True)]
    named["lon"] = named["mastr_id"].map(level["lon"])
    named["lat"] = named["mastr_id"].map(level["lat"])
    hv = named[named["cls"].eq("B") & named["lon"].notna()].copy()
    hv["plant_site"] = _merge_sites(hv)
    size = hv.groupby("plant_site")["capacity_mw"].transform("sum")
    hv = hv[size >= par.class_b_min_mw]

    people = population.assign(mw=population["population"].astype(float))
    tree = cKDTree(sites[["lon", "lat"]].to_numpy() * _KM)
    people["zone"] = sites["zone"].to_numpy()[
        tree.query(people[["lon", "lat"]].to_numpy() * _KM)[1]
    ]
    embedded_cells = cells.assign(
        zone=sites["zone"].to_numpy()[tree.query(cells[["lon", "lat"]].to_numpy() * _KM)[1]]
    )
    load_scale: Any = float(tables.energy_consumer["p_mw"].sum()) / float(people["mw"].sum())
    if rule == "capped":
        # the cap needs each area's own load in MW, as the lumped loads below receive it
        consumers = tables.energy_consumer
        own_load = consumers[
            ~consumers["mrid"].astype(str).str.startswith(_KEEP_LOADS)
            & consumers["node"].map(areas).notna()
        ]
        zone_load = own_load.groupby(own_load["node"].map(areas))["p_mw"].sum()
        load_scale = people["zone"].map(zone_load / people.groupby("zone")["mw"].sum()).fillna(0.0)

    def split(frame: pd.DataFrame, value: str) -> pd.Series:
        """MW per zone and site of located rows."""
        keyed = frame[["lon", "lat", "zone", value]].copy()
        keyed["cell"] = _cell(keyed["lon"], keyed["lat"]).to_numpy()
        merged = keyed.merge(shares, on=["zone", "cell"], how="left")
        lost = merged["site"].isna()
        if lost.any():
            _, i = tree.query(merged.loc[lost, ["lon", "lat"]].to_numpy() * _KM)
            merged.loc[lost, "site"] = sites["site"].to_numpy()[i]
            merged.loc[lost, "share"] = 1.0
        merged["part"] = merged[value] * merged["share"]
        return merged.groupby(["zone", "site"])["part"].sum()

    machines = tables.synchronous_machine.copy()
    columns = list(machines.columns)
    keep_a = named[named["cls"].eq("A")].groupby(["node", "fuel"])["capacity_mw"].sum()
    keep_b = hv.groupby(["node", "fuel"])["capacity_mw"].sum()
    stay, ledger = [], []
    pool: dict[tuple[str, str], list[float]] = {}
    new_machines: list[dict[str, Any]] = []
    b_pools: dict[tuple[str, str], list[float]] = {}

    def add_machine(row: Any, tag: str, cap: float, p: float, bus_hv: str, kv: float) -> None:
        mrid = f"term/{tag}"
        at = (
            (bus.at[bus_hv, "lon"], bus.at[bus_hv, "lat"])
            if bus_hv in bus.index
            else (
                sites.set_index("site").at[bus_hv.removeprefix("lumped/"), "lon"],
                sites.set_index("site").at[bus_hv.removeprefix("lumped/"), "lat"],
            )
        )
        sub = (
            vl_sub[bus.at[bus_hv, "voltage_level_mrid"]]
            if bus_hv in bus.index
            else vl_sub[
                bus.at[sites.set_index("site").at[bus_hv.removeprefix("lumped/"), "bus_hv"],
                       "voltage_level_mrid"]
            ]
        )  # fmt: skip
        add_bus(mrid, f"{mrid}|{par.terminal_kv:g}", tag, par.terminal_kv, sub, at, "terminal")
        committed = min(cap, p / par.machine_loading)
        running = committed >= par.class_b_min_mw
        committed = committed if running else cap
        rating = committed / par.power_factor
        ur = par.step_up_uk / np.hypot(1.0, par.step_up_x_r)
        r, x = transformer_impedance(kv, rating, par.step_up_uk, ur)
        add_unit(f"stepup/{tag}", bus_hv, mrid, kv, par.terminal_kv, rating, r, x, ASSUMPTION)
        extended[f"{mrid}|{par.terminal_kv:g}"] = extended.get(row["node"])
        new_machines.append(
            {
                **row,
                "mrid": mrid,
                "bus_mrid": mrid,
                "node": f"{mrid}|{par.terminal_kv:g}",
                "max_p_mw": cap,
                "p_mw": p,
                "in_service": p > 0,
                "voltage_control": running,
                "target_v_pu": par.terminal_v_pu,
                "rating_mva": rating,
                "q_max_mvar": rating * float(np.sqrt(1.0 - par.power_factor**2)),
                "q_min_mvar": -par.leading_limit * rating,
                "data_quality": ASSUMPTION,
            }
        )

    kv_of = dict(
        zip(tables.voltage_level["mrid"], tables.voltage_level["nominal_voltage_kv"], strict=False)
    )
    for index, raw in machines.iterrows():
        row = raw.to_dict()
        cap, p = float(row["max_p_mw"]), float(row["p_mw"])
        zone = areas.get(row["node"])
        if zone is None or zone != zone or cap <= 0:
            stay.append(index)
            ledger.append((row["fuel"], "abroad or empty", cap, p))
            continue
        ratio = p / cap
        fuel = str(row["fuel"])
        if str(row["mrid"]).startswith("feed_in/"):
            pool.setdefault((zone, fuel), [0.0, 0.0])
            pool[(zone, fuel)][0] += cap
            pool[(zone, fuel)][1] += p
            ledger.append((fuel, "C feed-in", cap, p))
            continue
        cls = plant_class(None, fuel) if fuel == _OFFSHORE else None
        if cls == "D":
            stay.append(index)
            ledger.append((fuel, "D", cap, p))
            continue
        d_cap = (
            float(
                named[
                    named["cls"].eq("D") & named["node"].eq(row["node"]) & named["fuel"].eq(fuel)
                ]["capacity_mw"].sum()
            )
            if is_inverter(fuel)
            else 0.0
        )
        a_cap = min(float(keep_a.get((row["node"], fuel), 0.0)), cap)
        a_cap = a_cap if a_cap >= par.class_b_min_mw else 0.0
        b_cap = min(float(keep_b.get((row["node"], fuel), 0.0)), cap - a_cap)
        d_cap = min(d_cap, cap)
        if d_cap > 0:
            piece = dict(row)
            ledger.append((fuel, "D", d_cap, d_cap * ratio))
            piece.update(
                mrid=f"{row['mrid']}/ehv", max_p_mw=d_cap, p_mw=d_cap * ratio, voltage_control=False
            )
            new_machines.append(piece)
        if a_cap > 0:
            kv = float(kv_of[bus.at[row["bus_mrid"], "voltage_level_mrid"]])
            add_machine(row, f"{row['mrid']}/A", a_cap, a_cap * ratio, row["bus_mrid"], kv)
            ledger.append((fuel, "A", a_cap, a_cap * ratio))
        if b_cap > 0:
            entry = b_pools.setdefault((row["node"], fuel), [0.0, 0.0])
            entry[0] += b_cap
            entry[1] += b_cap * ratio
            ledger.append((fuel, "B", b_cap, b_cap * ratio))
        rest = cap - a_cap - b_cap - d_cap
        if rest > 1e-9:
            pool.setdefault((zone, fuel), [0.0, 0.0])
            pool[(zone, fuel)][0] += rest
            pool[(zone, fuel)][1] += rest * ratio
            ledger.append((fuel, "C", rest, rest * ratio))

    # every embedded injection (all attachment methods) enters the catchment at this hour's
    # power: register cells carry the shape, the C pool of their area and fuel the power
    registered = embedded_cells[["lon", "lat", "zone", "mw", "fuel"]]
    pooled = pd.Series({key: value[1] for key, value in pool.items()}, dtype=float)
    shape = registered.groupby(["zone", "fuel"])["mw"].transform("sum")
    keys = pd.MultiIndex.from_arrays([registered["zone"], registered["fuel"]])
    hourly = registered.assign(
        mw=registered["mw"] / shape.where(shape > 0) * pooled.reindex(keys).to_numpy()
    ).fillna({"mw": 0.0})
    served = pd.concat(
        [
            people[["lon", "lat", "zone"]].assign(mw=people["mw"] * load_scale, kind="load"),
            hourly[["lon", "lat", "zone", "mw"]].assign(
                kind=np.where(hourly["fuel"].astype(str).str.startswith("Solar"), "solar", "other")
            ),
        ],
        ignore_index=True,
    )
    shares = catchment_shares(
        served,
        sites,
        rule,
        par.load_cap,
        float(np.hypot(1.0, par.load_cap_tan_phi)),
        par.backfeed_cap,
    )

    template = machines.iloc[0].to_dict() if len(machines) else {}
    hv["zone"] = hv["node"].map(areas)
    hv = hv.dropna(subset=["zone"])
    for key, frame in hv.groupby(["node", "fuel"]):
        node, fuel = cast(tuple[str, str], key)
        info = b_pools.get((node, fuel))
        if info is None:
            continue
        scale = float(info[0]) / float(frame["capacity_mw"].sum())
        ratio = float(info[1]) / float(info[0])
        for plant_site, part in frame.groupby("plant_site"):
            # a plant has one physical connection: its nearest infeed, not a share of a cell
            where = (part["lon"].mean(), part["lat"].mean())
            options = sites[sites["zone"].eq(part["zone"].iloc[0])]
            gaps = np.hypot(*((options[["lon", "lat"]].to_numpy() - where) * _KM).T)
            target = options["site"].iloc[int(np.argmin(gaps))]
            cap = float(part["capacity_mw"].sum()) * scale
            row = machines[machines["node"].eq(node) & machines["fuel"].eq(fuel)].iloc[0].to_dict()
            add_machine(row, f"{node}/{plant_site}/B", cap, cap * ratio, bar[target], KV_110)

    site_zone = sites.set_index("site")["zone"]
    for (zone, fuel), (cap_total, p_total) in pool.items():
        if cap_total <= 0:
            continue
        here = registered[registered["zone"].eq(zone) & registered["fuel"].eq(fuel)]
        if here["mw"].sum() > 0:
            weight = split(here, "mw").xs(zone, level=0)
        else:
            weight = pd.Series(1.0, index=site_zone.index[site_zone.eq(zone)])
        weight = weight / weight.sum()
        for site, part in weight.items():
            if part <= 0:
                continue
            new_machines.append(
                {
                    **template,
                    "mrid": f"embedded/{zone}/{fuel}/{site}",
                    "bus_mrid": bar[site],
                    "node": node_of[site],
                    "fuel": fuel,
                    "max_p_mw": cap_total * part,
                    "min_p_mw": 0.0,
                    "p_mw": p_total * part,
                    "in_service": p_total * part > 0,
                    "voltage_control": par.embedded_q_ratio is not None and p_total * part > 0,
                    "target_v_pu": par.lumped_v_pu,
                    "q_max_mvar": (par.embedded_q_ratio or 0.0) * p_total * part,
                    "q_min_mvar": -(par.embedded_q_ratio or 0.0) * p_total * part,
                    "data_quality": "estimated: embedded, lumped by connection level",
                }
            )
    kept = machines.loc[stay]
    grown_machines = pd.concat([kept, pd.DataFrame(new_machines)], ignore_index=True)
    for name in columns:
        if name not in grown_machines:
            grown_machines[name] = None

    loads = tables.energy_consumer
    fixed = loads["mrid"].astype(str).str.startswith(_KEEP_LOADS) | loads["node"].map(areas).isna()
    own = loads[~fixed]
    per = split(people, "mw")
    rows = []
    for zone, frame in own.groupby(own["node"].map(areas)):
        weight = per.xs(zone, level=0)
        weight = weight / weight.sum()
        for site, part in weight.items():
            rows.append(
                {
                    "mrid": f"lumped/load/{zone}/{site}",
                    "bus_mrid": bar[site],
                    "node": node_of[site],
                    "p_mw": float(frame["p_mw"].sum()) * part,
                    "q_mvar": float(frame["q_mvar"].sum()) * part,
                    "data_quality": "estimated: share of population in the catchment",
                }
            )
    lumped_loads = pd.DataFrame(rows)
    if par.split_levels:
        grown_machines, lumped_loads = _split_levels(
            infeed, sites, bar, node_of, extended, new_levels, new_bars, new_units,
            grown_machines, lumped_loads,
        )  # fmt: skip
    result = replace(
        tables,
        voltage_level=pd.concat(
            [tables.voltage_level, pd.DataFrame(new_levels)], ignore_index=True
        ),
        busbar_section=pd.concat(
            [tables.busbar_section, pd.DataFrame(new_bars)], ignore_index=True
        ),
        power_transformer=pd.concat(
            [tables.power_transformer, pd.DataFrame(new_units)], ignore_index=True
        ),
        synchronous_machine=grown_machines,
        energy_consumer=pd.concat([loads[fixed], lumped_loads], ignore_index=True),
    )
    table = pd.DataFrame(ledger, columns=["fuel", "path", "capacity_mw", "p_mw"])
    summary = table.groupby(["fuel", "path"], as_index=False)[["capacity_mw", "p_mw"]].sum()
    return result, pd.Series(extended), summary


_SCALED_MACHINE = ("max_p_mw", "min_p_mw", "p_mw", "q_max_mvar", "q_min_mvar")


def _split_levels(
    infeed: pd.DataFrame,
    sites: pd.DataFrame,
    bar: Mapping[str, str],
    node_of: Mapping[str, str],
    extended: dict[Any, Any],
    new_levels: list[dict[str, Any]],
    new_bars: list[dict[str, Any]],
    new_units: list[dict[str, Any]],
    machines: pd.DataFrame,
    loads: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A lumped 110 kV bus per EHV level at sites with more than one; see ``split_levels``.

    Changes the bus, level and unit lists in place; returns machines and lumped loads.
    """
    machines, loads = machines.copy(), loads.copy()
    level_mva = infeed.groupby(["site", "kv_hv"])["mva"].sum()
    for site, upper in zip(sites["site"], sites["kv_hv"], strict=True):
        if site not in level_mva.index.get_level_values(0):
            continue
        per = level_mva.xs(site)
        if len(per) < 2 or per.sum() <= 0:
            continue
        share = per / per.sum()
        main = bar[site]
        template_bar = next(b for b in new_bars if b["mrid"] == main)
        template_level = next(v for v in new_levels if v["mrid"] == f"vl/{main}")
        moved_m = machines["bus_mrid"].eq(main) & machines["mrid"].astype(str).str.startswith(
            "embedded/"
        )
        moved_l = loads["bus_mrid"].eq(main) if len(loads) else pd.Series(dtype=bool)
        extra_m, extra_l = [], []
        for kv, part in share.items():
            if kv == upper:
                continue
            mrid, node = f"{main}/{kv:g}", f"{site}/{kv:g}|110"
            new_levels.append({**template_level, "mrid": f"vl/{mrid}"})
            new_bars.append(
                {**template_bar, "mrid": mrid, "name": node, "voltage_level_mrid": f"vl/{mrid}"}
            )
            extended[node] = extended.get(node_of[site])
            here = infeed[
                infeed["site"].eq(site) & infeed["kv_hv"].astype(float).eq(float(cast(float, kv)))
            ]
            hv = set(here["bus_hv"])
            for unit in new_units:
                if unit["bus2_mrid"] == main and unit["bus1_mrid"] in hv:
                    unit["bus2_mrid"] = mrid
            m = machines[moved_m].copy()
            m[list(_SCALED_MACHINE)] = m[list(_SCALED_MACHINE)].astype(float) * float(part)
            m["mrid"] = m["mrid"].astype(str) + f"/{kv:g}"
            m["bus_mrid"], m["node"] = mrid, node
            extra_m.append(m)
            if len(loads):
                ld = loads[moved_l].copy()
                ld[["p_mw", "q_mvar"]] = ld[["p_mw", "q_mvar"]] * float(part)
                ld["mrid"] = ld["mrid"].astype(str) + f"/{kv:g}"
                ld["bus_mrid"], ld["node"] = mrid, node
                extra_l.append(ld)
        keep = float(share[upper]) if upper in share.index else 0.0
        machines.loc[moved_m, list(_SCALED_MACHINE)] = (
            machines.loc[moved_m, list(_SCALED_MACHINE)].astype(float) * keep
        )
        if len(loads):
            loads.loc[moved_l, ["p_mw", "q_mvar"]] = loads.loc[moved_l, ["p_mw", "q_mvar"]] * keep
        machines = pd.concat([machines, *extra_m], ignore_index=True)
        loads = pd.concat([loads, *extra_l], ignore_index=True)
    return machines, loads


def set_taps(tables: Any, v_lumped: Mapping[str, float], target_pu: float = 1.02) -> Any:
    """Next tap position of the lumped transformers from the solved 110 kV voltages.

    A tap changer works in steps, so ``rated_u2_kv`` stays on the grid of :data:`TAP_STEP` within
    :data:`TAP_RANGE`. Parallel transformers of a site share the bus, so they move together.
    """
    table = tables.power_transformer.copy()
    chosen = table["mrid"].astype(str).str.startswith("lumped/tr/")
    for index in table.index[chosen]:
        voltage = v_lumped.get(table.at[index, "bus2_mrid"])
        if voltage is None or voltage != voltage or voltage <= 0:
            continue
        wanted = float(table.at[index, "rated_u2_kv"]) * target_pu / voltage
        step = np.round((wanted / KV_110 - 1.0) / TAP_STEP) * TAP_STEP
        table.at[index, "rated_u2_kv"] = KV_110 * (
            1.0 + float(np.clip(step, -TAP_RANGE, TAP_RANGE))
        )
    if isinstance(tables, SimpleNamespace):
        return SimpleNamespace(**{**vars(tables), "power_transformer": table})
    return replace(tables, power_transformer=table)


def voltage_setpoint_pu(load_mw: float) -> float:
    """Voltage machines and neighbours hold: lower in light hours, an assumption without source."""
    return 1.00 if load_mw < LIGHT_LOAD_MW else 1.02


def soundness_problems(
    tables: Any,
    before: Any | None = None,
    bus_balance: pd.Series | None = None,
    tolerance_mw: float = 1.0,
) -> list[str]:
    """Problems of a connected model; empty when every check of the plan's gates passes.

    Checks positive impedances, known and consistent buses and voltage bases, no isolated bus
    holding power, machine limits against ratings, preserved totals against ``before``, and
    power balance per bus of a solved network when ``bus_balance`` holds its mismatch in MW.
    """
    problems = []
    tr = tables.power_transformer
    new = tr[tr["mrid"].astype(str).str.startswith(("lumped/tr/", "stepup/"))]
    if ((new["x_ohm"].astype(float) <= 0) | (new["r_ohm"].astype(float) < 0)).any():
        problems.append("non-positive transformer impedance")
    bus = tables.busbar_section.set_index("mrid")
    kv = tables.voltage_level.set_index("mrid")["nominal_voltage_kv"].astype(float)
    bus_kv = bus["voltage_level_mrid"].map(kv)
    for name, table in (
        ("machine", tables.synchronous_machine),
        ("load", tables.energy_consumer),
    ):
        if not table["bus_mrid"].isin(bus.index).all():
            problems.append(f"{name} on an unknown bus")
    if not (new["bus1_mrid"].isin(bus.index) & new["bus2_mrid"].isin(bus.index)).all():
        problems.append("transformer on an unknown bus")
    else:
        ratio = new["rated_u1_kv"].astype(float) / new["bus1_mrid"].map(bus_kv)
        if (ratio - 1).abs().max() > 1e-9:
            problems.append("transformer high side does not match its bus voltage")
        ratio = new["rated_u2_kv"].astype(float) / new["bus2_mrid"].map(bus_kv)
        if (ratio - 1).abs().max() > TAP_RANGE + 1e-9:
            problems.append("transformer low side outside its tap range")
    pairs = tables.power_transformer[["bus1_mrid", "bus2_mrid"]].to_numpy().tolist()
    lines = tables.ac_line_segment
    pairs += lines[["bus1_mrid", "bus2_mrid"]].to_numpy().tolist()
    parent = {b: b for b in bus.index}

    def root(b: Any) -> Any:
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        return b

    for a, b in pairs:
        parent[root(a)] = root(b)
    groups = pd.Series({b: root(b) for b in bus.index})
    main = groups.value_counts().index[0]
    gen = tables.synchronous_machine
    off = gen["bus_mrid"].map(groups).ne(main) & gen["p_mw"].astype(float).gt(0)
    load = tables.energy_consumer
    load_off = load["bus_mrid"].map(groups).ne(main) & load["p_mw"].astype(float).ne(0)
    if off.sum() or load_off.sum():
        problems.append(
            f"{int(off.sum())} machines and {int(load_off.sum())} loads off the main network"
        )
    if (gen["p_mw"].astype(float) > gen["max_p_mw"].astype(float) + 1e-6).any():
        problems.append("machine above its rating")
    if "rating_mva" in gen:
        held = gen[gen["mrid"].astype(str).str.startswith("term/") & gen["in_service"].astype(bool)]
        if (held["p_mw"].astype(float) > held["rating_mva"].astype(float) + 1e-6).any():
            problems.append("machine power above its committed rating")
    if before is not None:
        for name, new_v, old_v in (
            (
                "generation",
                gen["p_mw"].astype(float).sum(),
                before.synchronous_machine["p_mw"].astype(float).sum(),
            ),
            (
                "capacity",
                gen["max_p_mw"].astype(float).sum(),
                before.synchronous_machine["max_p_mw"].astype(float).sum(),
            ),
            (
                "load",
                load["p_mw"].astype(float).sum(),
                before.energy_consumer["p_mw"].astype(float).sum(),
            ),
        ):
            if abs(new_v - old_v) > tolerance_mw:
                problems.append(f"{name} changed by {new_v - old_v:.1f} MW")
    if bus_balance is not None and bus_balance.abs().max() > tolerance_mw:
        problems.append(f"bus balance off by {bus_balance.abs().max():.1f} MW")
    return problems
