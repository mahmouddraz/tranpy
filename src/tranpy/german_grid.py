"""German transmission network from the operators' published data, with real routes.

:func:`build_german_grid` builds the 220/380 kV network of the four German TSOs:

* **Electrical model (authoritative)** - the Core Static Grid Model
  (:func:`tranpy.grid_data.load_core_grid_model`): every circuit and transformer with
  the operator's resistance, reactance, susceptance and rating. Nothing in the electrical
  model is estimated.
* **Geography** - OpenStreetMap, as prebuilt for PyPSA-Eur (Xiong et al., Sci. Data 12,
  277, 2025; ODbL), with substation names from the Overpass API. Substations are located
  by name; a circuit gets its real route if a path through the OpenStreetMap lines
  between its two substations has the official length (within ``length_tolerance``);
  otherwise it keeps the schematic route of the operators' map.

Cleaning of the source, each step recorded in :attr:`GermanGrid.issues`:

1. rows repeated in the line and tie-line sheets are removed (same EIC and same ends;
   segments of a teed circuit share one EIC and are kept);
2. a missing voltage is taken from the other segments of the circuit;
3. substation names are unified (umlauts and their transliterations, ``/N`` = ``/Nord``,
   ``/N`` = ``/Nord``) and corrected with the curated tables
   ``resources/grid/substation_aliases_de.csv`` and ``line_corrections_de.csv``, in
   which every entry states its evidence;
4. transformers are assigned to substations by name, voltages from the data; a
   transformer side without any circuit in the source (local 220 kV areas are not
   published) becomes a busbar of its own;
5. connectivity and parameter plausibility are checked.

Not in the source: generators, loads, spur lines and generator connections.
"""

from __future__ import annotations

import json
import re
import unicodedata
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from tranpy.grid_data import GERMAN_TSOS, CoreGridModel, load_core_grid_model
from tranpy_core.data.download import RemoteFile, cache_dir, fetch
from tranpy_core.errors import ConfigurationError

__all__ = [
    "OSM_GRID",
    "GermanGrid",
    "build_german_grid",
    "fetch_osm_substations",
    "load_osm_grid",
    "site_key",
]

_ZENODO = "https://zenodo.org/api/records/18619025/files/{}/content"
OSM_GRID: dict[str, RemoteFile] = {
    name: RemoteFile(filename=f"pypsa_eur_osm_0.7_{name}", url=_ZENODO.format(name), sha256=sha)
    for name, sha in (
        ("buses.csv", "f50c75ea0339f5578ae383b47a1c0564b371c5d0d60fd4181ed626c7f3a5a523"),
        ("lines.csv", "d3ee4f45a0d9889eee9e27376cb9cfc877514964f77ba406bea3b6eb336531cb"),
    )
}
"""Prebuilt OpenStreetMap network for PyPSA-Eur, version 0.7 (Zenodo 18619025)."""

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

_REGION = (4.0, 17.0, 46.0, 56.0)
"""Longitude and latitude bounds read from OpenStreetMap (Germany and its border)."""

_DIRECTIONS = {"n": "nord", "s": "sud", "o": "ost", "w": "west", "m": "mitte"}

_ABSOLUTE_TOLERANCE_KM = 1.5
"""Route length difference accepted for short circuits (in addition to the relative one)."""

_SNAP_KM = 1.5
"""Largest distance between a named OpenStreetMap substation and the line network node."""

_PLAUSIBLE_X_OHM_PER_KM = {220: (0.15, 0.45), 380: (0.15, 0.40)}
"""Range of the series reactance per circuit and km of overhead lines and cables."""


def _num(value: Any) -> float:
    """A table value as float (pandas types single values too loosely for mypy)."""
    return float(value)


def _text(value: Any) -> str:
    return str(value)


def _rows(frame: pd.DataFrame) -> Iterator[Any]:
    """Rows as named tuples with untyped fields."""
    return iter(frame.itertuples())


def site_key(name: str) -> str:
    """Normalised substation name: lower case, ASCII, ``/N`` -> ``/nord``.

    Suffixes such as ``50H`` are kept: "Hamburg/N 50H" (50Hertz) and "Hamburg/N"
    (TenneT) are different substations a few kilometres apart.
    """
    text = str(name).strip().lower().replace("ß", "ss")
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    # transliterated umlauts ("ue" for "ü") are reduced like the umlauts themselves
    text = re.sub(r"([aou])e", r"\1", text)
    text = re.sub(r"\s*\(abzweig\)", " abzweig", text)
    # a trailing direction word is the same as the "/direction" form ("Doerpen West")
    text = re.sub(r"^([a-z\- ]+?)\s+(nord|sud|ost|west|mitte)$", r"\1/\2", text.strip())
    parts = [re.sub(r"[^a-z0-9 \-]", "", p).strip() for p in text.split("/")]
    if len(parts) > 1 and parts[1] in _DIRECTIONS:
        parts[1] = _DIRECTIONS[parts[1]]
    return re.sub(r"\s+", " ", "/".join(p for p in parts if p))


_GENERIC_WORDS = {
    "umspannwerk", "uw", "station", "schaltanlage", "schaltwerk", "anlage", "kv",
    "umspannanlage", "kraftwerk", "transnetbw", "amprion", "tennet", "nord", "sud", "ost",
    "west", "mitte", "abzweig",
}  # fmt: skip


def _words(name: str) -> set[str]:
    """Significant words of a substation name (for comparing with OpenStreetMap names)."""
    key = site_key(name).replace("/", " ").replace("-", " ")
    return {w for w in key.split() if len(w) >= 4 and not w.isdigit() and w not in _GENERIC_WORDS}


def _voltage_class(kv: float) -> int:
    return 380 if kv >= 300 else 220


def _aliases() -> dict[str, str]:
    table = resources.files("tranpy.resources.grid").joinpath("substation_aliases_de.csv")
    with table.open(encoding="utf-8") as fh:
        frame = pd.read_csv(fh)
    return {site_key(a): site_key(b) for a, b in zip(frame["name"], frame["same_as"], strict=True)}


def _split_sites() -> dict[tuple[str, str], str]:
    table = resources.files("tranpy.resources.grid").joinpath("split_sites_de.csv")
    with table.open(encoding="utf-8") as fh:
        frame = pd.read_csv(fh, dtype=str)
    return {
        (a.strip(), b.strip()): c.strip()
        for a, b, c in zip(frame["name"], frame["reported_by"], frame["node_name"], strict=True)
    }


def _curated_locations(name: str) -> dict[str, tuple[float, float]]:
    table = resources.files("tranpy.resources.grid").joinpath(name)
    with table.open(encoding="utf-8") as fh:
        frame = pd.read_csv(fh)
    return {
        str(n): (float(x), float(y))
        for n, x, y in zip(frame["node"], frame["lon"], frame["lat"], strict=True)
    }


def _verified_locations() -> dict[str, tuple[float, float]]:
    """German substations located with orthophoto evidence."""
    return _curated_locations("substation_locations_de.csv")


def _corrections() -> pd.DataFrame:
    table = resources.files("tranpy.resources.grid").joinpath("line_corrections_de.csv")
    with table.open(encoding="utf-8") as fh:
        return pd.read_csv(fh, dtype=str)


def _transformer_site(name: str) -> str:
    """Substation in a transformer name, for the naming styles of the four TSOs."""
    text = re.sub(r"^\s*(TR|PST)\b", "", str(name).strip(), flags=re.I)
    parts = [p.strip() for p in re.split(r"\s+/\s+|^/\s*", text) if p.strip()]
    parts = [p for p in parts if not re.fullmatch(r"\d{3}\s*-\s*\d{3}\s*kV", p, re.I)]
    site = parts[0] if parts else ""
    site = re.sub(r"\b(Trafo|TR|T)\s*\d{1,3}[a-z]?\b.*$", "", site, flags=re.I)
    site = re.sub(r"\s+\d{3}\s*/\s*\d{3}\b.*$", "", site)
    return re.sub(r"\s+\d{3}\b.*$", "", site).strip()


def _points(wkt: str) -> npt.NDArray[np.float64]:
    numbers = re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", wkt)
    return np.array(numbers, dtype=np.float64).reshape(-1, 2)


def _osm_ids(tags: Any) -> list[str]:
    parts = (p.strip() for p in str(tags).split(";"))
    return [p for p in parts if re.match(r"(way|relation|node)/\d+$", p)]


def load_osm_grid(directory: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """OpenStreetMap AC buses and lines around Germany (lines with a ``route`` column).

    ``directory`` holds the CSV files; they are downloaded (checksum verified) if None.
    """
    paths = {
        name: (Path(directory) / name if directory is not None else fetch(remote))
        for name, remote in OSM_GRID.items()
    }
    buses = pd.read_csv(paths["buses.csv"])
    west, east, south, north = _REGION
    buses = buses[buses["x"].between(west, east) & buses["y"].between(south, north)]
    buses = buses[~buses["dc"].astype(str).str.lower().isin(("t", "true"))]
    inside = set(buses["bus_id"])
    lines = pd.read_csv(paths["lines.csv"], quotechar="'")
    lines = lines[lines["bus0"].isin(inside) & lines["bus1"].isin(inside)]
    lines = lines.assign(
        route=pd.Series([_points(g) for g in lines["geometry"]], index=lines.index, dtype=object)
    )
    return {
        "buses": buses.reset_index(drop=True),
        "lines": lines.drop(columns="geometry").reset_index(drop=True),
    }


def fetch_osm_substations(path: str | Path | None = None) -> Path:
    """220-400 kV substations (with names, voltages, operators) around Germany.

    Queried from the Overpass API and written to ``path`` (default: the cache); an
    existing file is reused, so a build can be repeated with identical input.
    """
    out = Path(path) if path is not None else cache_dir() / "osm_hv_substations.json"
    if out.exists():
        return out
    west, east, south, north = _REGION
    query = (
        "[out:json][timeout:300];"
        'nwr["power"~"substation|plant"]["voltage"~"220000|225000|380000|400000"]'
        f"({south},{west},{north},{east});out center tags;"
    )
    request = urllib.request.Request(
        OVERPASS_URL,
        data=urllib.parse.urlencode({"data": query}).encode(),
        headers={"User-Agent": "tranpy"},
    )
    with urllib.request.urlopen(request, timeout=400) as response:
        payload = response.read()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    return out


def _read_substations(path: Path) -> tuple[pd.DataFrame, str]:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    rows = []
    for element in payload["elements"]:
        tags = element.get("tags", {})
        centre = element.get("center", element)
        if "name" not in tags or "lat" not in centre:
            continue
        levels = {
            _voltage_class(int(v) / 1000)
            for v in re.findall(r"\d+", tags.get("voltage", ""))
            if int(v) >= 200000
        }
        rows.append(
            {
                "osm_id": f"{element['type']}/{element['id']}",
                "name": tags["name"],
                "operator": tags.get("operator"),
                "levels": levels,
                "lon": _num(centre["lon"]),
                "lat": _num(centre["lat"]),
            }
        )
    stamp = str(payload.get("osm3s", {}).get("timestamp_osm_base", "unknown date"))
    return pd.DataFrame(rows), stamp


@dataclass
class GermanGrid:
    """Nodes, circuits and transformers of the German transmission network.

    Attributes:
        nodes: ``node`` (``<site>|<kV>``), ``site``, ``kv``, ``names`` (as published),
            ``tso``, ``kind`` (``substation``, ``tee``, ``busbar`` or ``foreign``),
            ``lon``, ``lat``, ``location_source`` (``osm`` or ``schematic``),
            ``osm_bus``.
        lines: one row per circuit segment: ``eic``, ``name``, ``tso``, ``node0``,
            ``node1``, ``kv``, ``length_km``, ``r_ohm``, ``x_ohm``, ``b_us``,
            ``imax_a``, ``tie_line``, ``hvdc``, ``route``, ``route_source`` (``osm``,
            ``within substation`` or ``schematic``).
        transformers: ``eic``, ``name``, ``tso``, ``node_hv``, ``node_lv``, ``kv_hv``,
            ``kv_lv``, ``r_ohm``, ``x_ohm``, ``b_us``, ``g_us``, ``imax_a``,
            ``phase_shifter``.
        issues: every correction and finding: ``check``, ``element``, ``detail``,
            ``action``.
        sources: provenance of the inputs.
    """

    nodes: pd.DataFrame
    lines: pd.DataFrame
    transformers: pd.DataFrame
    issues: pd.DataFrame
    sources: dict[str, str] = field(default_factory=dict)

    def components(self) -> pd.Series:
        """Connected component of every node (circuits and transformers), 0 = largest."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        index = {n: i for i, n in enumerate(self.nodes["node"])}
        series = self.transformers[~self.transformers["phase_shifter"]]
        ac = self.lines[~self.lines["hvdc"]]
        a = [index[n] for n in ac["node0"]] + [index[n] for n in series["node_hv"]]
        b = [index[n] for n in ac["node1"]] + [index[n] for n in series["node_lv"]]
        size = len(index)
        graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(size, size))
        _, labels = connected_components(graph, directed=False)
        rank = {lab: r for r, lab in enumerate(pd.Series(labels).value_counts().index)}
        return pd.Series([rank[x] for x in labels], index=self.nodes["node"], name="component")

    def subset(self, tsos: Sequence[str]) -> GermanGrid:
        """Circuits and transformers of the given operators (with their end nodes)."""
        unknown = set(tsos) - set(GERMAN_TSOS)
        if unknown:
            raise ConfigurationError(f"unknown operators {sorted(unknown)}")
        lines = self.lines[self.lines["tso"].isin(tsos)].reset_index(drop=True)
        transformers = self.transformers[self.transformers["tso"].isin(tsos)]
        used = set(lines["node0"]) | set(lines["node1"])
        used |= set(transformers["node_hv"]) | set(transformers["node_lv"])
        nodes = self.nodes[self.nodes["node"].isin(used)].reset_index(drop=True)
        return GermanGrid(
            nodes, lines, transformers.reset_index(drop=True), self.issues, dict(self.sources)
        )

    def to_geojson(self, path: str | Path) -> Path:
        """Write circuits (with routes) and nodes as GeoJSON for GIS tools."""

        def plain(row: pd.Series, skip: Iterable[str]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for key, value in row.items():
                if str(key) in skip:
                    continue
                if isinstance(value, (list, tuple)):
                    out[str(key)] = list(value)
                elif pd.isna(value):
                    out[str(key)] = None
                else:
                    out[str(key)] = value.item() if hasattr(value, "item") else value
            return out

        features = [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": row["route"].tolist()},
                "properties": {"kind": "line", **plain(row, ("route",))},
            }
            for _, row in self.lines.iterrows()
            if len(row["route"]) >= 2
        ]
        features += [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [row["lon"], row["lat"]]},
                "properties": {"kind": "node", **plain(row, ())},
            }
            for _, row in self.nodes.iterrows()
            if np.isfinite(row["lon"])
        ]
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
        return out


class _Issues:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []

    def add(self, check: str, element: str, detail: str, action: str) -> None:
        self.rows.append({"check": check, "element": element, "detail": detail, "action": action})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["check", "element", "detail", "action"])


def _clean_lines(core: CoreGridModel, aliases: dict[str, str], issues: _Issues) -> pd.DataFrame:
    lines = core.lines.copy()
    lines["ends"] = [
        tuple(sorted((_text(a).strip(), _text(b).strip())))
        for a, b in zip(lines["from_substation"], lines["to_substation"], strict=True)
    ]
    repeated = lines.duplicated(["eic", "ends"])
    for name in lines.loc[repeated, "name"]:
        issues.add("duplicate", str(name), "listed in two sheets or by two TSOs", "removed")
    lines = lines[~repeated].reset_index(drop=True)
    lines["hvdc"] = False
    for fix in _rows(_corrections()):
        rows = lines.index[lines["name"].str.strip() == fix.name.strip()]
        if not len(rows):
            issues.add("correction", fix.name, "not found in this release", "skipped")
            continue
        value: Any = fix.value == "True" if fix.field == "hvdc" else fix.value
        lines.loc[rows, fix.field] = value
        issues.add("correction", fix.name, fix.evidence, f"{fix.field} set to {fix.value}")
    for i in lines.index[lines["kv"].isna()]:
        same = lines[(lines["eic"] == lines.at[i, "eic"]) & lines["kv"].notna()]
        if len(same):
            lines.at[i, "kv"] = same["kv"].iloc[0]
            detail = "voltage missing; taken from the other segments of the circuit"
        else:
            neighbours = lines[
                lines["from_substation"].isin(
                    [lines.at[i, "from_substation"], lines.at[i, "to_substation"]]
                )
                & lines["kv"].notna()
            ]
            lines.at[i, "kv"] = neighbours["kv"].mode().iloc[0] if len(neighbours) else 380.0
            detail = "voltage missing; taken from the circuits at its substations"
        issues.add(
            "voltage", _text(lines.at[i, "name"]), detail, f"set to {lines.at[i, 'kv']:.0f} kV"
        )

    merged: set[str] = set()
    split = _split_sites()

    def node(name: str, kv: float, reporter: str) -> str:
        key = site_key(split.get((str(name).strip(), reporter), name))
        if key in aliases:
            if name not in merged:
                merged.add(name)
                issues.add("alias", str(name), f"same substation as {aliases[key]}", "merged")
            key = aliases[key]
        return f"{key}|{_voltage_class(kv)}"

    lines["node0"] = [
        node(a, v, r)
        for a, v, r in zip(lines["from_substation"], lines["kv"], lines["tso"], strict=True)
    ]
    lines["node1"] = [
        node(b, v, r)
        for b, v, r in zip(lines["to_substation"], lines["kv"], lines["tso"], strict=True)
    ]
    lines = _merge_double_reports(lines, issues)
    for row in _rows(lines[lines["node0"] == lines["node1"]]):
        issues.add("self-loop", _text(row.name), f"both ends at {row.node0}", "check the source")
    for row in _rows(lines):
        length = _num(row.length_km)
        if not np.isfinite(length) or length <= 0:
            issues.add("length", _text(row.name), f"length {row.length_km}", "kept as published")
            continue
        x_per_km = _num(row.x_ohm) / length
        low, high = _PLAUSIBLE_X_OHM_PER_KM[_voltage_class(_num(row.kv))]
        if length >= 1.0 and not low <= x_per_km <= high:
            issues.add(
                "reactance",
                _text(row.name),
                f"{x_per_km:.3f} ohm/km outside {low}-{high}",
                "kept as published",
            )
    return lines


_AGREEMENT = 0.10
"""Largest relative difference of two operators' reports of a circuit's reactance."""


def _is_tee(node: str) -> bool:
    site = node.split("|")[0]
    return site.startswith("y-") or site.endswith("abzweig")


def _unify_reported_tees(lines: pd.DataFrame, issues: _Issues) -> pd.DataFrame:
    """Two operators' reports of one circuit (same EIC, agreeing reactance) that share one
    end describe the same segment, so their other ends are the same node. Applied when
    both of those ends are tees (operators name tees differently); otherwise reported."""
    rename: dict[str, str] = {}
    for eic, group in lines.groupby("eic", sort=False):
        if group["tso"].nunique() < 2 or len(group) != 2:
            continue
        a, b = group.iloc[0], group.iloc[1]
        x = group["x_ohm"].astype(float)
        if (x.max() - x.min()) > _AGREEMENT * x.max():
            continue
        ends_a, ends_b = {a["node0"], a["node1"]}, {b["node0"], b["node1"]}
        shared = ends_a & ends_b
        if len(shared) != 1:
            continue
        (other_a,), (other_b,) = ends_a - shared, ends_b - shared
        if _is_tee(other_a) and _is_tee(other_b):
            keep, drop = sorted((other_a, other_b))
            rename[drop] = keep
            issues.add(
                "tee name",
                f"{drop} = {keep}",
                f"circuit {eic} reported by {a['tso']} and {b['tso']} with the same reactance",
                "unified",
            )
        else:
            issues.add(
                "review",
                f"{a['name']} / {b['name']}",
                f"circuit {eic}: {a['tso']} ends at {other_a}, {b['tso']} at {other_b}",
                "kept both; check on imagery",
            )
    if rename:
        lines = lines.assign(
            node0=lines["node0"].replace(rename), node1=lines["node1"].replace(rename)
        )
    return lines


def _merge_double_reports(lines: pd.DataFrame, issues: _Issues) -> pd.DataFrame:
    """Keep one row per circuit reported by two operators (same EIC and same ends).

    The other report is kept in ``other_tso``, ``other_r_ohm``, ``other_x_ohm`` and
    ``other_b_us``: operators report either the whole circuit or their own section, and
    :func:`_reconcile_ties` decides between them using the route length. The published
    lengths are the operators' own sections and are not used.
    """
    lines = _unify_reported_tees(lines, issues)
    lines = lines.assign(
        pair=pd.Series(
            [frozenset((a, b)) for a, b in zip(lines["node0"], lines["node1"], strict=True)],
            index=lines.index,
            dtype=object,
        ),
        other_tso=None,
        other_r_ohm=np.nan,
        other_x_ohm=np.nan,
        other_b_us=np.nan,
    )
    keep: list[Any] = []
    for _, group in lines.groupby(["eic", "pair"], sort=False):
        if group["tso"].nunique() < 2:
            keep.extend(group.index)
            continue
        first, other = group.index[0], group.index[-1]
        lines.loc[first, ["other_tso", "other_r_ohm", "other_x_ohm", "other_b_us"]] = [
            lines.at[other, "tso"],
            lines.at[other, "r_ohm"],
            lines.at[other, "x_ohm"],
            lines.at[other, "b_us"],
        ]
        lines.at[first, "length_km"] = np.nan
        issues.add(
            "double report",
            _text(lines.at[first, "name"]),
            "reported by "
            + ", ".join(
                f"{t}: X={_num(v):.3f} ohm"
                for t, v in zip(group["tso"], group["x_ohm"], strict=True)
            ),
            "merged; parameters decided by the route length",
        )
        keep.append(first)
    return lines.loc[sorted(keep)].drop(columns="pair").reset_index(drop=True)


_TYPICAL_X_OHM_PER_KM = {220: 0.32, 380: 0.27}
"""Series reactance per circuit and km of typical overhead lines."""

_ROUTE_FACTOR = 1.15
"""Typical ratio of route length to straight-line distance."""


def _reconcile_ties(
    lines: pd.DataFrame, foreign: pd.DataFrame, nodes: pd.DataFrame, issues: _Issues
) -> None:
    """Complete the parameters of cross-border circuits, in place.

    Operators report tie-lines by different conventions: the whole circuit, or only
    their own section up to the border. Candidates are the German report, the foreign
    report of the same EIC and the sum of both sections; the one whose reactance per km
    over the estimated route (straight distance x ``_ROUTE_FACTOR``) is closest to a
    typical overhead line is kept. A German report covering clearly only a section, with
    no foreign report, is flagged.
    """
    where = nodes.set_index("node")[["lon", "lat", "location_source"]]
    reports = foreign.groupby("eic")
    located = ("osm", "verified")
    for i in lines.index[(lines["tie_line"] | lines["other_tso"].notna()) & ~lines["hvdc"]]:
        row = lines.loc[i]
        a, b = where.loc[row["node0"]], where.loc[row["node1"]]
        if not (
            _text(a["location_source"]).startswith(located)
            and _text(b["location_source"]).startswith(located)
        ):
            issues.add(
                "tie-line",
                _text(row["name"]),
                "an end is not located; the route length is unknown",
                "kept as published",
            )
            continue
        straight = _num(
            np.hypot((a["lon"] - b["lon"]) * np.cos(np.radians(a["lat"])), a["lat"] - b["lat"])
            * 111.2
        )
        if not np.isfinite(straight) or straight < 3.0:
            continue
        route_km = straight * _ROUTE_FACTOR
        typical = _TYPICAL_X_OHM_PER_KM[_voltage_class(_num(row["kv"]))]
        candidates = {f"{row['tso']} report": (row["r_ohm"], row["x_ohm"], row["b_us"])}
        second = None
        if isinstance(row["other_tso"], str):
            second = (row["other_tso"], row["other_r_ohm"], row["other_x_ohm"], row["other_b_us"])
        elif row["eic"] in reports.groups:
            f = reports.get_group(row["eic"]).iloc[0]
            second = (f["tso"], f["r_ohm"], f["x_ohm"], f["b_us"])
        if second is not None:
            candidates[f"{second[0]} report"] = second[1:]
            candidates["both sections"] = (
                _num(row["r_ohm"]) + _num(second[1]),
                _num(row["x_ohm"]) + _num(second[2]),
                _num(row["b_us"]) + _num(second[3]),
            )
        published = [_num(v[1]) / typical for v in candidates.values()]
        if route_km > 3 * max(published):
            issues.add(
                "tie-line",
                _text(row["name"]),
                f"ends {route_km:.0f} km apart, far more than any report implies",
                "kept as published; check the end locations",
            )
            continue
        scored = {
            name: abs(np.log(_num(v[1]) / route_km / typical))
            for name, v in candidates.items()
            if np.isfinite(_num(v[1])) and _num(v[1]) > 0
        }
        if not scored:
            continue
        best = min(scored, key=lambda k: scored[k])
        x_per_km = _num(candidates[best][1]) / route_km
        if best != f"{row['tso']} report":
            lines.loc[i, ["r_ohm", "x_ohm", "b_us"]] = [_num(v) for v in candidates[best]]
            issues.add(
                "tie-line",
                _text(row["name"]),
                "candidates "
                + ", ".join(f"{k}: X={_num(v[1]):.2f} ohm" for k, v in candidates.items())
                + f"; route about {route_km:.0f} km",
                f"used {best} ({x_per_km:.3f} ohm/km)",
            )
        elif len(candidates) > 1:
            issues.add(
                "tie-line",
                _text(row["name"]),
                "candidates "
                + ", ".join(f"{k}: X={_num(v[1]):.2f} ohm" for k, v in candidates.items())
                + f"; route about {route_km:.0f} km",
                f"kept {best} ({x_per_km:.3f} ohm/km)",
            )
        if best == f"{row['tso']} report" and x_per_km < 0.6 * typical and len(candidates) == 1:
            issues.add(
                "tie-line",
                _text(row["name"]),
                f"X={_num(row['x_ohm']):.2f} ohm over a route of about {route_km:.0f} km "
                f"({x_per_km:.3f} ohm/km): only the German section is reported",
                "kept; neighbouring section missing",
            )


def _transformers(
    core: CoreGridModel, sites: set[str], aliases: dict[str, str], issues: _Issues
) -> pd.DataFrame:
    rows = []
    for t in _rows(core.transformers):
        raw = _transformer_site(_text(t.name))
        site = site_key(_split_sites().get((raw, _text(t.tso)), raw))
        site = aliases.get(site, site)
        hv, lv = _voltage_class(_num(t.kv_primary)), _voltage_class(_num(t.kv_secondary))
        if site not in sites:
            issues.add(
                "transformer",
                _text(t.name).strip(),
                f"substation {site!r} has no circuit",
                "busbar added",
            )
        rows.append(
            {
                "eic": t.eic,
                "name": _text(t.name).strip(),
                "tso": t.tso,
                "node_hv": f"{site}|{hv}",
                "node_lv": f"{site}|{lv}",
                "kv_hv": _num(t.kv_primary),
                "kv_lv": _num(t.kv_secondary),
                "r_ohm": t.r_ohm,
                "x_ohm": t.x_ohm,
                "b_us": t.b_us,
                "g_us": t.g_us,
                "imax_a": t.imax_a,
                "phase_shifter": hv == lv,
            }
        )
    return pd.DataFrame(rows)


class _OsmNetwork:
    """OpenStreetMap substations (by name) and shortest paths over its lines."""

    def __init__(self, buses: pd.DataFrame, lines: pd.DataFrame, substations: pd.DataFrame) -> None:
        from scipy.sparse import coo_matrix

        self.buses = buses.reset_index(drop=True)
        self.index = {b: i for i, b in enumerate(self.buses["bus_id"])}
        self.xy = self.buses[["x", "y"]].to_numpy(np.float64)
        self.kv = np.array([_voltage_class(v) for v in self.buses["voltage"]])
        self.substations = substations.reset_index(drop=True)
        self.sub_xy = self.substations[["lon", "lat"]].to_numpy(np.float64)
        self.sub_words = [_words(n) for n in self.substations["name"]]
        self.lines = lines.reset_index(drop=True)
        a = np.array([self.index[b] for b in self.lines["bus0"]], dtype=np.int64)
        b = np.array([self.index[b] for b in self.lines["bus1"]], dtype=np.int64)
        km = self.lines["length"].to_numpy(np.float64) / 1000.0
        # parallel lines between the same substations: keep the shortest (a sparse matrix
        # would add their lengths)
        self.edge: dict[tuple[int, int], int] = {}
        for k, (i, j) in enumerate(zip(a.tolist(), b.tolist(), strict=True)):
            for pair in ((i, j), (j, i)):
                if pair not in self.edge or km[k] < km[self.edge[pair]]:
                    self.edge[pair] = k
        rows, cols = zip(*self.edge, strict=True) if self.edge else ((), ())
        weights = [km[k] for k in self.edge.values()]
        size = len(self.index)
        self.graph = coo_matrix((weights, (rows, cols)), shape=(size, size)).tocsr()

    def locate(
        self, site: str, kv: int, near: npt.NDArray[np.float64], radius_km: float
    ) -> tuple[int, str] | None:
        """Named OpenStreetMap substation for ``site``: a shared significant name word,
        within ``radius_km`` of ``near``; one with the voltage level is preferred."""
        wanted = _words(site)
        if site.startswith("y-") or not wanted or not np.isfinite(near).all():
            return None
        lon, lat = near
        d = (
            np.hypot((self.sub_xy[:, 0] - lon) * np.cos(np.radians(lat)), self.sub_xy[:, 1] - lat)
            * 111.2
        )
        named = np.array([bool(wanted & w) for w in self.sub_words]) & (d <= radius_km)
        level = np.array([kv in levels for levels in self.substations["levels"]])
        for mask, how in ((named & level, "osm"), (named, "osm (other voltage level)")):
            if mask.any():
                return int(np.argmin(np.where(mask, d, np.inf))), how
        return None

    def nearest(self, point: npt.NDArray[np.float64], kv: int, radius_km: float) -> int | None:
        lon, lat = point
        d = np.hypot((self.xy[:, 0] - lon) * np.cos(np.radians(lat)), self.xy[:, 1] - lat) * 111.2
        d[self.kv != kv] = np.inf
        i = int(np.argmin(d))
        return i if d[i] <= radius_km else None

    def route(
        self, a: int, b: int, official_km: float, tolerance: float
    ) -> npt.NDArray[np.float64] | None:
        from scipy.sparse.csgraph import dijkstra

        if a == b:
            return None
        allowed = max(tolerance * official_km, _ABSOLUTE_TOLERANCE_KM)
        dist, pred = dijkstra(
            self.graph, indices=a, limit=official_km + allowed + 1.0, return_predecessors=True
        )
        if not np.isfinite(dist[b]) or abs(dist[b] - official_km) > allowed:
            return None
        pieces = []
        node = b
        while node != a:
            prev = int(pred[node])
            points = np.asarray(self.lines.at[self.edge[(prev, node)], "route"], dtype=np.float64)
            start = self.xy[prev]
            if np.hypot(*(points[0] - start)) > np.hypot(*(points[-1] - start)):
                points = points[::-1]
            pieces.append(points)
            node = prev
        return np.vstack(pieces[::-1])


def build_german_grid(
    core: CoreGridModel | None = None,
    osm_directory: str | Path | None = None,
    *,
    substations: str | Path | None = None,
    locate_km: float = 25.0,
    unnamed_km: float = 0.0,
    length_tolerance: float = 0.15,
) -> GermanGrid:
    """Build the cleaned network (see module docs).

    Args:
        core: Core Static Grid Model of all Core TSOs (loaded if None); the German
            TSOs' elements form the network, the neighbours' reports complete the
            cross-border circuits (see :func:`_reconcile_ties`).
        osm_directory: folder with the OpenStreetMap CSV files (downloaded if None).
        substations: Overpass result with the 220-400 kV OpenStreetMap substations
            (queried and cached if None, see :func:`fetch_osm_substations`).
        locate_km: largest distance between the schematic position of a substation and
            the OpenStreetMap substation of the same name.
        unnamed_km: largest distance to an unnamed OpenStreetMap substation of the same
            voltage used when no name matches. Off by default: checked on orthophotos,
            the nearest unnamed substation was often a different site; such nodes are
            located in ``resources/grid/substation_locations_de.csv`` instead.
        length_tolerance: accepted relative difference between a route through
            OpenStreetMap and the official circuit length.
    """
    full = core if core is not None else load_core_grid_model(tsos=None)
    foreign = full.lines[~full.lines["tso"].isin(GERMAN_TSOS)]
    core = CoreGridModel(
        lines=full.lines[full.lines["tso"].isin(GERMAN_TSOS)].reset_index(drop=True),
        transformers=full.transformers[full.transformers["tso"].isin(GERMAN_TSOS)].reset_index(
            drop=True
        ),
        routes=full.routes,
        release=full.release,
    )
    issues = _Issues()
    aliases = _aliases()
    lines = _clean_lines(core, aliases, issues)
    sites = {n.split("|")[0] for n in set(lines["node0"]) | set(lines["node1"])}
    transformers = _transformers(core, sites, aliases, issues)

    internal = lines[~lines["tie_line"]]
    domestic = set(internal["node0"]) | set(internal["node1"])
    domestic |= set(transformers["node_hv"]) | set(transformers["node_lv"])
    names: dict[str, set[str]] = {}
    tsos: dict[str, set[str]] = {}
    schematic: dict[str, list[npt.NDArray[np.float64]]] = {}
    for row in _rows(lines):
        route = core.routes.get(_text(row.eic))
        for end, (node, raw) in enumerate(
            ((row.node0, row.from_substation), (row.node1, row.to_substation))
        ):
            names.setdefault(node, set()).add(str(raw).strip())
            tsos.setdefault(node, set()).add(_text(row.tso))
            if route is not None and len(route) >= 2:
                schematic.setdefault(node, []).append(route[0] if end == 0 else route[-1])
    located = core.transformers.set_index("eic")[["lon", "lat"]]
    for t in _rows(transformers):
        for node in (t.node_hv, t.node_lv):
            names.setdefault(node, set()).add(t.name)
            tsos.setdefault(node, set()).add(_text(t.tso))
            if t.eic in located.index and np.isfinite(located.loc[t.eic, "lon"]):
                schematic.setdefault(node, []).append(located.loc[t.eic].to_numpy(np.float64))

    osm = load_osm_grid(osm_directory)
    named, stamp = _read_substations(fetch_osm_substations(substations))
    network = _OsmNetwork(osm["buses"], osm["lines"], named)

    busbars = (
        (set(transformers["node_hv"]) | set(transformers["node_lv"]))
        - set(lines["node0"])
        - set(lines["node1"])
    )
    verified = _verified_locations()
    abroad = _curated_locations("substation_locations_foreign.csv")
    node_rows = []
    for node in sorted(names):
        site, kv = node.rsplit("|", 1)
        points = schematic.get(node)
        guess = np.median(np.array(points), axis=0) if points else np.array([np.nan, np.nan])
        if site.startswith("y-") or site.endswith("abzweig"):
            kind = "tee"
        elif node in busbars:
            kind = "busbar"
        elif node not in domestic:
            kind = "foreign"
        else:
            kind = "substation"
        found = network.locate(site, int(kv), guess, locate_km)
        source: str | None
        if node in verified:
            lon, lat = verified[node]
            osm_bus = network.nearest(np.array([lon, lat]), int(kv), _SNAP_KM)
            source = "verified"
        elif node in abroad:
            lon, lat = abroad[node]
            osm_bus = network.nearest(np.array([lon, lat]), int(kv), _SNAP_KM)
            source = "osm (curated name match)"
        elif found is not None:
            lon, lat = network.sub_xy[found[0]]
            osm_bus = network.nearest(network.sub_xy[found[0]], int(kv), _SNAP_KM)
            source = found[1]
        else:
            osm_bus = (
                network.nearest(guess, int(kv), unnamed_km)
                if points and unnamed_km > 0 and kind != "foreign"
                else None
            )
            lon, lat = network.xy[osm_bus] if osm_bus is not None else guess
            source = "osm (nearest unnamed)" if osm_bus is not None else None
        node_rows.append(
            {
                "node": node,
                "site": site,
                "kv": int(kv),
                "names": sorted(names[node]),
                "tso": ";".join(sorted(tsos.get(node, set()))),
                "kind": kind,
                "lon": _num(lon),
                "lat": _num(lat),
                "location_source": source or ("schematic" if points else "none"),
                "osm_bus": None if osm_bus is None else str(network.buses.at[osm_bus, "bus_id"]),
            }
        )
    nodes = pd.DataFrame(node_rows)

    _reconcile_ties(lines, foreign, nodes, issues)
    osm_of = {
        n: b for n, b in zip(nodes["node"], nodes["osm_bus"], strict=True) if isinstance(b, str)
    }
    _place_tees(nodes, lines, network, osm_of, length_tolerance)
    where = {
        n: np.array([x, y])
        for n, x, y in zip(nodes["node"], nodes["lon"], nodes["lat"], strict=True)
    }
    routes: list[npt.NDArray[np.float64]] = []
    route_sources = []
    for row in _rows(lines):
        a, b = osm_of.get(row.node0), osm_of.get(row.node1)
        route = None
        if a is not None and b is not None and np.isfinite(row.length_km) and row.length_km > 0:
            route = network.route(
                network.index[a], network.index[b], _num(row.length_km), length_tolerance
            )
        if route is not None:
            routes.append(route)
            route_sources.append("osm")
        elif a is not None and a == b and _num(row.length_km) <= _ABSOLUTE_TOLERANCE_KM:
            # both ends in the same substation area: busbar sections, couplers
            routes.append(np.array([where[row.node0], where[row.node1]]))
            route_sources.append("within substation")
        else:
            fallback = core.routes.get(_text(row.eic))
            routes.append(fallback if fallback is not None else np.empty((0, 2)))
            route_sources.append("schematic" if fallback is not None else "none")
    columns = [
        "eic", "name", "tso", "node0", "node1", "kv", "length_km", "r_ohm", "x_ohm",
        "b_us", "imax_a", "tie_line", "hvdc",
    ]  # fmt: skip
    lines = lines[columns].assign(
        route=pd.Series(routes, index=lines.index, dtype=object), route_source=route_sources
    )

    grid = GermanGrid(
        nodes=nodes,
        lines=lines.reset_index(drop=True),
        transformers=transformers,
        issues=pd.DataFrame(),
        sources={
            "electrical model": f"Core Static Grid Model ({core.release}), JAO",
            "routes": "OpenStreetMap via PyPSA-Eur prebuilt network 0.7 (Zenodo 18619025, ODbL)",
            "substation locations": f"OpenStreetMap via Overpass API, {stamp}",
        },
    )
    component = grid.components()
    for comp, members in component.groupby(component).groups.items():
        if comp != 0:
            issues.add(
                "connectivity",
                ", ".join(sorted(members)),
                f"not connected to the main network ({len(members)} nodes)",
                "kept; check the source",
            )
    grid.issues = issues.frame()
    return grid


def _place_tees(
    nodes: pd.DataFrame,
    lines: pd.DataFrame,
    network: _OsmNetwork,
    osm_of: dict[str, str],
    tolerance: float,
) -> None:
    """Locate tee points on the route between two located ends of their circuit.

    For a tee Y on A-Y-B the route A->B must have the official length L(A-Y)+L(Y-B); Y is
    the OpenStreetMap junction nearest to the point at distance L(A-Y) along it.
    """
    at = {n: i for i, n in enumerate(nodes["node"])}
    for _ in range(3):  # tees next to tees are placed once a neighbour is located
        for tee in nodes.loc[(nodes["kind"] == "tee") & ~nodes["node"].isin(osm_of), "node"]:
            touching = lines[(lines["node0"] == tee) | (lines["node1"] == tee)]
            ends = [
                (r.node1 if r.node0 == tee else r.node0, _num(r.length_km)) for r in _rows(touching)
            ]
            ends = [(n, km) for n, km in ends if n in osm_of and np.isfinite(km) and km > 0]
            placed = False
            for i, (a, la) in enumerate(ends):
                for b, lb in ends[i + 1 :]:
                    route = network.route(
                        network.index[osm_of[a]], network.index[osm_of[b]], la + lb, tolerance
                    )
                    if route is None:
                        continue
                    steps = np.hypot(
                        np.diff(route[:, 0]) * np.cos(np.radians(route[:-1, 1])),
                        np.diff(route[:, 1]),
                    )
                    cumulative = np.r_[0.0, np.cumsum(steps)] * 111.2
                    point = route[int(np.argmin(abs(cumulative - cumulative[-1] * la / (la + lb))))]
                    junction = network.nearest(point, int(_num(nodes.at[at[tee], "kv"])), 3.0)
                    if junction is None:
                        continue
                    bus = str(network.buses.at[junction, "bus_id"])
                    osm_of[tee] = bus
                    nodes.loc[at[tee], ["lon", "lat"]] = network.xy[junction]
                    nodes.loc[at[tee], ["osm_bus", "location_source"]] = [bus, "osm (tee on route)"]
                    placed = True
                    break
                if placed:
                    break
