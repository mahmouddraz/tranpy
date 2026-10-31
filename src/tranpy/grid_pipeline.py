"""Pipeline of the German transmission grid model: bronze -> silver -> gold -> verify.

Each step is a command (``tranpy-grid <step>``, also the entry point of the Databricks
job in ``resources/grid_pipeline.job.yml``):

* ``bronze`` downloads the sources into ``<files>/<run_date>/raw`` (Core Static Grid Model,
  PyPSA-Eur OpenStreetMap extract, Overpass substations and lines, country borders) and
  writes them as raw tables;
* ``silver`` builds the cleaned network, attaches power plants and population-based
  loads (:mod:`tranpy.german_grid`, :mod:`tranpy.grid_layers`);
* ``gold`` writes the CIM-aligned model, runs an AC power flow and exports CGMES;
* ``verify`` checks the written tables (row counts, references, keys, coordinates,
  power flow on the gold tables read back) and fails on any violation.

Configuration comes from environment variables (job environment variables on
Databricks) and can be overridden on the command line:

======================== =====================================================
``TRANPY_CATALOG``       Unity Catalog catalog
``TRANPY_BRONZE_SCHEMA`` schema of the raw tables (default ``grid_bronze``)
``TRANPY_SILVER_SCHEMA`` schema of the cleaned tables (default ``grid_silver``)
``TRANPY_GOLD_SCHEMA``   schema of the model (default ``grid_gold``)
``TRANPY_FILES``         directory for downloaded and exported files
``TRANPY_PLANTS_TABLE``  Bundesnetzagentur power plant list (table or Parquet)
``TRANPY_POPULATION_TABLE`` population cells ``lon``, ``lat``, ``population``
``TRANPY_RUN_DATE``      date folder of the sources (default: today, UTC)
======================== =====================================================

Without a catalog (local runs) every table is written as Parquet under
``<files>/<run_date>/<layer>/``; the steps are otherwise identical.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tranpy.german_grid import (
    OSM_GRID,
    build_german_grid,
    fetch_osm_hv_lines,
    fetch_osm_substations,
)
from tranpy.grid_data import CORE_SGM, load_core_grid_model
from tranpy.grid_layers import (
    LOAD_REFERENCE,
    TOTAL_LOAD_MW,
    attach_plants,
    cim_tables,
    load_shares,
    node_countries,
    plants_from_kraftwerksliste,
)
from tranpy.grid_register import area_load_shares, attach_register
from tranpy.maps import NATURAL_EARTH_COUNTRIES
from tranpy.osm_export import SOURCE as OSM_EXPORT_SOURCE
from tranpy.osm_export import latest_export, read_hv_layers
from tranpy.osm_routes import (
    dlm250_geometries,
    fetch_circuit_relations,
    fetch_dlm250_lines,
    load_relations,
    load_ways,
    route_grid,
)
from tranpy_core.data.download import fetch
from tranpy_core.errors import ConfigurationError, DataError

__all__ = ["PipelineConfig", "build_layer", "main", "verify_tables"]

logger = logging.getLogger(__name__)

LAYERS = ("bronze", "silver", "gold")

COMMENTS: dict[str, str] = {
    "grid_sgm_lines_raw": "Core Static Grid Model (JAO), sheet Lines, all Core TSOs, as published.",
    "grid_sgm_tielines_raw": "Core Static Grid Model (JAO), sheet Tielines, as published.",
    "grid_sgm_transformers_raw": "Core Static Grid Model (JAO), sheet Transformers, as published.",
    "grid_sgm_map_routes_raw": "Line polylines of the map published with the Core Static Grid "
    "Model (schematic), by EIC code.",
    "grid_osm_hv_substations_raw": "OpenStreetMap substations and plants tagged 220-400 kV "
    "around Germany (Overpass API). OpenStreetMap contributors, ODbL 1.0.",
    "grid_osm_hv_lines_raw": "OpenStreetMap power lines and cables tagged 220-400 kV in Germany "
    "with node ids and geometry (Overpass API). OpenStreetMap contributors, ODbL 1.0.",
    "grid_osm_pypsa_buses_raw": "PyPSA-Eur prebuilt OpenStreetMap network 0.7, buses. ODbL 1.0.",
    "grid_osm_pypsa_lines_raw": "PyPSA-Eur prebuilt OpenStreetMap network 0.7, lines. ODbL 1.0.",
    "grid_de_nodes": "German 220/380 kV network nodes with location, how it was located, country.",
    "grid_de_circuits": "German 220/380 kV circuits: operator parameters (R, X ohm; B "
    "microsiemens per circuit), rating, tie-line/HVDC flags, route.",
    "grid_de_transformers": "German 380/220 kV transformers and phase shifters.",
    "grid_de_issues": "Every correction and finding of the network build, with evidence.",
    "grid_de_sources": "Sources, licences and build time of the German grid model.",
    "grid_de_curation": "Curated corrections with evidence (aliases, split sites, circuit "
    "corrections, verified locations).",
    "grid_de_plants": "Power plant list normalised (status, capacity, WGS84 coordinates).",
    "grid_de_plant_attachments": "Power plants attached to transmission substations; method and "
    "data_quality (published or estimated) per row.",
    "grid_de_load_shares": "Population per German transmission substation (basis of the loads).",
    "cim_substation": "German HV grid model (CIM Substation).",
    "cim_voltage_level": "German HV grid model (CIM VoltageLevel).",
    "cim_busbar_section": "German HV grid model (CIM BusbarSection), one bus per node.",
    "cim_ac_line_segment": "German HV grid model (CIM ACLineSegment): r/x ohm, bch siemens per "
    "circuit, rated current, route. HVDC links left out (see cim_model_info).",
    "cim_power_transformer": "German HV grid model (CIM PowerTransformer), parameters at the "
    "high-voltage end.",
    "cim_synchronous_machine": "German HV grid model (CIM SynchronousMachine): generation per "
    "node and fuel; data_quality marks estimated placements.",
    "cim_energy_consumer": "German HV grid model (CIM EnergyConsumer): estimated loads.",
    "cim_external_network_injection": "German HV grid model (CIM ExternalNetworkInjection): "
    "neighbouring grids at tie-line ends outside Germany.",
    "cim_model_info": "German HV grid model: version, assumptions, sources, power flow result.",
}


@dataclass(frozen=True)
class PipelineConfig:
    """Where the pipeline reads from and writes to (see module docs)."""

    files: Path
    catalog: str | None = None
    schemas: Mapping[str, str] = field(
        default_factory=lambda: {
            "bronze": "grid_bronze",
            "silver": "grid_silver",
            "gold": "grid_gold",
        }
    )
    plants_table: str | None = None
    population_table: str | None = None
    run_date: str = field(default_factory=lambda: datetime.now(UTC).date().isoformat())

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ, **overrides: Any) -> PipelineConfig:
        values = {
            "files": env.get("TRANPY_FILES"),
            "catalog": env.get("TRANPY_CATALOG") or None,
            "plants_table": env.get("TRANPY_PLANTS_TABLE") or None,
            "population_table": env.get("TRANPY_POPULATION_TABLE") or None,
            "run_date": env.get("TRANPY_RUN_DATE") or None,
        }
        values.update({k: v for k, v in overrides.items() if v is not None})
        if not values["files"]:
            raise ConfigurationError("set TRANPY_FILES (or --files)")
        schemas = {
            layer: overrides.get(f"{layer}_schema")
            or env.get(f"TRANPY_{layer.upper()}_SCHEMA")
            or f"grid_{layer}"
            for layer in LAYERS
        }
        return cls(
            files=Path(values["files"]),
            catalog=values["catalog"],
            schemas=schemas,
            plants_table=values["plants_table"],
            population_table=values["population_table"],
            **({"run_date": values["run_date"]} if values["run_date"] else {}),
        )

    @property
    def run_dir(self) -> Path:
        return self.files / self.run_date

    def table(self, layer: str, name: str) -> str:
        if self.catalog is None:
            raise ConfigurationError("no catalog configured")
        return f"`{self.catalog}`.`{self.schemas[layer]}`.`{name}`"


# ---------------------------------------------------------------- sources (bronze)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_sources(raw: Path) -> dict[str, Path]:
    """Download every source into ``raw`` (kept if already there: reruns are reproducible)."""
    raw.mkdir(parents=True, exist_ok=True)
    paths = {
        "sgm": fetch(CORE_SGM, directory=raw),
        "borders": fetch(NATURAL_EARTH_COUNTRIES, directory=raw),
        "substations": fetch_osm_substations(raw / "osm_hv_substations.json"),
        "lines": fetch_osm_hv_lines(raw / "osm_hv_lines.json"),
        "circuits": fetch_circuit_relations(raw / "osm_circuits.json"),
        "dlm250": fetch_dlm250_lines(raw / "dlm250_lines.json"),
    }
    for name, remote in OSM_GRID.items():
        target = raw / "pypsa" / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            fetch(remote, directory=target.parent).rename(target)
        paths[f"pypsa_{name}"] = target
    return paths


def _safe_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Column names usable in Delta tables (letters, digits, underscore; unique)."""
    names: list[str] = []
    for i, c in enumerate(frame.columns):
        base = re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z]+", "_", str(c))).strip("_").lower()
        base = base or f"col{i}"
        name, k = base, 2
        while name in names:
            name, k = f"{base}_{k}", k + 1
        names.append(name)
    out = frame.copy()
    out.columns = names
    return out


def _points(route: Any) -> list[dict[str, float]]:
    return [{"lon": float(x), "lat": float(y)} for x, y in np.asarray(route).reshape(-1, 2)]


def bronze_tables(paths: Mapping[str, Path], retrieved: str) -> dict[str, pd.DataFrame]:
    """Raw tables: every published column, plus source, checksum and retrieval date."""
    tables: dict[str, pd.DataFrame] = {}
    sgm = paths["sgm"]
    meta = {"source": f"Core Static Grid Model ({sgm.name}), JAO", "source_sha256": _sha256(sgm)}
    with zipfile.ZipFile(sgm) as archive:
        name = next(n for n in archive.namelist() if n.endswith("_External.xlsx"))
        workbook = archive.read(name)
    for sheet, table in (
        ("Lines", "grid_sgm_lines_raw"),
        ("Tielines", "grid_sgm_tielines_raw"),
        ("Transformers", "grid_sgm_transformers_raw"),
    ):
        frame = pd.read_excel(io.BytesIO(workbook), sheet_name=sheet, header=1, dtype=str)
        tables[table] = _safe_columns(frame).dropna(how="all").assign(**meta)
    core = load_core_grid_model(sgm, tsos=None)
    tables["grid_sgm_map_routes_raw"] = pd.DataFrame(
        {"eic": list(core.routes), "route": [_points(r) for r in core.routes.values()]}
    ).assign(**meta)

    osm_meta = {"source": "OpenStreetMap via Overpass API (ODbL 1.0)"}
    payload = json.loads(paths["substations"].read_text(encoding="utf-8"))
    stamp = payload.get("osm3s", {}).get("timestamp_osm_base")
    rows = []
    for element in payload["elements"]:
        centre, tags = element.get("center", element), element.get("tags", {})
        rows.append(
            {
                "osm_id": f"{element['type']}/{element['id']}",
                "name": tags.get("name"),
                "power": tags.get("power"),
                "voltage": tags.get("voltage"),
                "operator": tags.get("operator"),
                "lon": centre.get("lon"),
                "lat": centre.get("lat"),
                "tags": json.dumps(tags, ensure_ascii=False),
            }
        )
    tables["grid_osm_hv_substations_raw"] = pd.DataFrame(rows).assign(
        osm_timestamp=stamp, source_sha256=_sha256(paths["substations"]), **osm_meta
    )
    payload = json.loads(paths["lines"].read_text(encoding="utf-8"))
    rows = []
    for element in payload["elements"]:
        tags = element.get("tags", {})
        rows.append(
            {
                "osm_id": f"way/{element['id']}",
                **{
                    k: tags.get(k)
                    for k in (
                        "power",
                        "name",
                        "ref",
                        "operator",
                        "voltage",
                        "circuits",
                        "cables",
                        "location",
                    )
                },
                "node_ids": [int(n) for n in element["nodes"]],
                "geometry": [{"lon": p["lon"], "lat": p["lat"]} for p in element["geometry"]],
                "tags": json.dumps(tags, ensure_ascii=False),
            }
        )
    tables["grid_osm_hv_lines_raw"] = pd.DataFrame(rows).assign(
        osm_timestamp=payload.get("osm3s", {}).get("timestamp_osm_base"),
        source_sha256=_sha256(paths["lines"]),
        **osm_meta,
    )
    for name, table in (
        ("buses.csv", "grid_osm_pypsa_buses_raw"),
        ("lines.csv", "grid_osm_pypsa_lines_raw"),
    ):
        path = paths[f"pypsa_{name}"]
        frame = pd.read_csv(path, quotechar="'" if name == "lines.csv" else '"', dtype=str)
        tables[table] = _safe_columns(frame).assign(
            source="PyPSA-Eur prebuilt OpenStreetMap network 0.7 (ODbL 1.0)",
            source_sha256=_sha256(path),
        )
    for frame in tables.values():
        frame["retrieved_at"] = retrieved
    relations = load_relations(paths["circuits"])
    tables["grid_osm_circuits_raw"] = (
        pd.DataFrame(
            [
                {
                    "relation_id": r["id"],
                    **{f"tag_{k}": v for k, v in r.get("tags", {}).items()},
                    "member_ways": [m["ref"] for m in r.get("members", []) if m["type"] == "way"],
                }
                for r in relations
            ]
        )
        .pipe(_safe_columns)
        .assign(
            source="OpenStreetMap route=power relations via Overpass API (ODbL)",
            source_sha256=_sha256(paths["circuits"]),
            retrieved_at=retrieved,
        )
    )
    return tables


# ---------------------------------------------------------------- silver and gold


def model_tables(
    paths: Mapping[str, Path],
    plants_raw: pd.DataFrame,
    population: pd.DataFrame,
    run_date: str,
    register: Mapping[str, pd.DataFrame] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], Any]:
    """Silver and gold tables from the downloaded sources and the two input tables."""
    core = load_core_grid_model(paths["sgm"], tsos=None)
    grid = build_german_grid(
        core, paths["pypsa_buses.csv"].parent, substations=paths["substations"]
    )
    countries = node_countries(grid.nodes, json.loads(paths["borders"].read_text(encoding="utf-8")))
    routed = route_grid(
        grid.lines,
        grid.nodes,
        load_ways(paths["lines"]),
        load_relations(paths["circuits"]),
        dlm250_geometries(paths["dlm250"]),
        foreign_nodes=[str(n) for n, c in countries.items() if c != "DEU"],
    )
    grid.lines, grid.nodes = routed.lines, routed.nodes
    plants = plants_from_kraftwerksliste(plants_raw)
    shares = load_shares(population, grid.nodes)
    total_load = TOTAL_LOAD_MW
    load_source = str(LOAD_REFERENCE["source"])
    extra_silver: dict[str, pd.DataFrame] = {}
    if register is not None:
        located = register["plant_locations"].drop_duplicates("mastr_id").set_index("mastr_id")
        plants = plants[~plants["aggregated"]].copy()
        for axis in ("lon", "lat"):
            plants[axis] = plants[axis].fillna(plants["mastr_id"].map(located[axis]))
        cells = register["cells"]
        attachments = pd.concat(
            [
                attach_plants(plants, grid.nodes, shares, countries),
                attach_register(cells, grid.nodes),
            ],
            ignore_index=True,
        )
        area_load = register["area_load"]
        shares, total_load = area_load_shares(
            shares, grid.nodes, dict(zip(area_load["area"], area_load["mean_mw"], strict=True))
        )
        load_source = str(area_load["source"].iloc[0])
        extra_silver = {"grid_de_register_cells": cells, "grid_de_area_load": area_load}
    else:
        attachments = attach_plants(plants, grid.nodes, shares, countries)

    sources = [{"source": k, "description": v} for k, v in grid.sources.items()]
    sources += [
        {"source": "power plants", "description": "Bundesnetzagentur Kraftwerksliste"},
        {"source": "population", "description": "population cells (lon, lat, population)"},
        {"source": "load", "description": load_source},
        {
            "source": "country borders",
            "description": "Natural Earth admin 0 v5.1.2 (public domain)",
        },
    ]
    curation = []
    for name in (
        "substation_aliases_de.csv",
        "split_sites_de.csv",
        "line_corrections_de.csv",
        "substation_locations_de.csv",
        "substation_locations_foreign.csv",
    ):
        with resources.files("tranpy.resources.grid").joinpath(name).open(encoding="utf-8") as fh:
            table = pd.read_csv(fh, dtype=str)
        content = table.drop(columns=[table.columns[0], "evidence"])
        curation.append(
            pd.DataFrame(
                {
                    "curation_table": name,
                    "key": table.iloc[:, 0],
                    "content": [
                        json.dumps(r.dropna().to_dict(), ensure_ascii=False)
                        for _, r in content.iterrows()
                    ],
                    "evidence": table["evidence"],
                }
            )
        )
    silver = {
        "grid_de_nodes": grid.nodes.assign(country=grid.nodes["node"].map(countries)),
        "grid_de_circuits": grid.lines.assign(route=grid.lines["route"].map(_points)),
        "grid_de_transformers": grid.transformers,
        "grid_de_issues": grid.issues,
        "grid_de_sources": pd.DataFrame(sources).assign(model_version=run_date),
        "grid_de_curation": pd.concat(curation, ignore_index=True),
        "grid_de_plants": plants,
        "grid_de_plant_attachments": attachments,
        "grid_de_load_shares": shares,
        "grid_de_circuit_routes": routed.routes,
        "grid_de_tee_positions": routed.tees,
        "grid_de_field_changes": routed.changes,
        "grid_de_route_candidates": routed.candidates,
        "grid_de_loose_ends": routed.loose_ends,
        **extra_silver,
    }
    for row in routed.loose_ends.to_dict("records"):
        gap = max(float(row["gap_km_node0"]), float(row["gap_km_node1"]))
        grid.issues.loc[len(grid.issues)] = [
            "loose end",
            str(row["name"]),
            f"route ends {gap:.1f} km from its bus",
            "kept; check the node, tee or route",
        ]
    model = cim_tables(
        grid,
        attachments,
        shares,
        total_load_mw=total_load,
        load_source=load_source,
        model_version=run_date,
        countries=countries,
    )
    gold = {name: frame.assign(model_version=run_date) for name, frame in model.tables().items()}
    return silver, gold, model


def gold_check(model: Any, out_dir: Path) -> dict[str, Any]:
    """AC power flow on the model and CGMES export (needs the ``[powsybl]`` extra)."""
    from tranpy.powsybl_model import (
        export_cgmes,
        line_loading,
        network_from_cim_tables,
        run_load_flow,
    )

    network = network_from_cim_tables(model)
    summary = run_load_flow(network, model)
    loading = line_loading(network, model)
    # pieces inside a station carry nominal ratings (150 A stubs): reported separately
    inside = set(
        model.ac_line_segment.loc[
            model.ac_line_segment["route_source"] == "within substation", "mrid"
        ]
    )
    stub = loading.index.isin(inside)
    archive = export_cgmes(
        network, out_dir / f"germany_hv_{model.model_info['model_version']}_cgmes.zip"
    )
    return {
        **{f"loadflow_{k}": v for k, v in summary.as_dict().items()},
        "loadflow_lines_over_100pct": int((loading[~stub] > 1).sum()),
        "loadflow_station_stubs_over_100pct": int((loading[stub] > 1).sum()),
        "loadflow_line_loading_p95": float(loading.quantile(0.95))
        if len(loading)
        else float("nan"),
        "cgmes_archive": str(archive),
    }


# ---------------------------------------------------------------- storage


class Store:
    """Writes tables as Parquet under the run directory and, with a catalog, as Delta
    tables (overwritten: the pipeline owns its schemas)."""

    def __init__(self, config: PipelineConfig, spark: Any = None) -> None:
        self.config = config
        self.spark = spark

    def write(self, layer: str, tables: Mapping[str, pd.DataFrame]) -> None:
        folder = self.config.run_dir / layer
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            path = folder / f"{name}.parquet"
            _parquet_ready(frame).to_parquet(path, index=False)
            logger.info("%s.%s: %d rows", layer, name, len(frame))
            if self.spark is not None:
                target = self.config.table(layer, name)
                (
                    self.spark.read.parquet(str(path))
                    .write.mode("overwrite")
                    .option("overwriteSchema", "true")
                    .saveAsTable(target)
                )
                comment = COMMENTS.get(name, "").replace("'", "''")
                self.spark.sql(f"COMMENT ON TABLE {target} IS '{comment}'")

    def read(self, layer: str, name: str) -> pd.DataFrame:
        if self.spark is not None:
            frame: pd.DataFrame = self.spark.table(self.config.table(layer, name)).toPandas()
            return frame
        return pd.read_parquet(self.config.run_dir / layer / f"{name}.parquet")


def _parquet_ready(frame: pd.DataFrame) -> pd.DataFrame:
    """Object columns with mixed values become strings; list columns stay lists."""
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype != object:
            continue
        sample = out[column].dropna().head(50).tolist()
        if sample and all(isinstance(v, (list, np.ndarray)) for v in sample):
            out[column] = out[column].map(
                lambda v: list(v) if isinstance(v, (list, np.ndarray)) else None
            )
        elif sample and not all(isinstance(v, str) for v in sample):
            out[column] = out[column].map(
                lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
            )
    return out


def _input_table(spark: Any, name: str | None, query: str) -> pd.DataFrame:
    if name is None:
        raise ConfigurationError("input table not configured")
    if name.endswith(".parquet") or Path(name).exists():
        return pd.read_parquet(name)
    if spark is None:
        raise ConfigurationError(f"{name} is a table: run on Databricks or pass a Parquet file")
    frame: pd.DataFrame = spark.sql(query.format(table=name)).toPandas()
    return frame


def _register_inputs(directory: Path) -> dict[str, pd.DataFrame] | None:
    """Prepared register and load tables, if present in the volume."""
    names = ("cells", "plant_locations", "area_load")
    if not all((directory / f"{name}.parquet").exists() for name in names):
        return None
    return {name: pd.read_parquet(directory / f"{name}.parquet") for name in names}


_EXPORT_TABLES = {
    "lines": ("grid_oim_lines_raw", "grid_de_osm_ways"),
    "towers": ("grid_oim_towers_raw", "grid_de_towers_hv"),
    "substations": ("grid_oim_substations_raw", "grid_de_osm_substations"),
    "transformers": ("grid_oim_transformers_raw", "grid_de_osm_transformers"),
    "switches": ("grid_oim_switches_raw", "grid_de_osm_switches"),
}


def export_tables(directory: Path) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Bronze and silver tables of the newest OpenStreetMap export (empty without one).

    Nothing is filtered out by status. The GeoPackage is read from a local copy because
    SQLite files need file locks that Unity Catalog volumes do not offer.
    """
    export = latest_export(directory)
    if export is None:
        return {}, {}
    snapshot = export.parent.name
    meta = {"source": OSM_EXPORT_SOURCE, "source_sha256": _sha256(export), "snapshot": snapshot}
    with tempfile.TemporaryDirectory() as folder:
        local = Path(folder) / export.name
        shutil.copyfile(export, local)
        layers = read_hv_layers(local)
    bronze = {_EXPORT_TABLES[k][0]: _safe_columns(v).assign(**meta) for k, v in layers.items()}
    silver = {
        _EXPORT_TABLES[k][1]: _safe_columns(v).assign(
            snapshot=snapshot,
            **({"status_source": f"OpenStreetMap tags, {snapshot}"} if "status" in v else {}),
        )
        for k, v in layers.items()
    }
    return bronze, silver


# ---------------------------------------------------------------- steps


def build_layer(step: str, config: PipelineConfig, spark: Any = None) -> dict[str, Any]:
    """Run one step; returns a short report."""
    store = Store(config, spark)
    raw = config.run_dir / "raw"
    paths = download_sources(raw)
    if step == "bronze":
        tables = {
            **bronze_tables(paths, config.run_date),
            **export_tables(config.files / "osm_export")[0],
        }
        store.write("bronze", tables)
        return {"bronze_tables": {k: len(v) for k, v in tables.items()}}
    plants = _input_table(spark, config.plants_table, "SELECT * FROM {table}")
    population = _input_table(
        spark,
        config.population_table,
        "SELECT round(lon / 0.05) * 0.05 AS lon, round(lat / 0.05) * 0.05 AS lat, "
        "sum(population) AS population FROM {table} GROUP BY 1, 2",
    ).astype(float)
    register = _register_inputs(config.files / "register")
    silver, gold, model = model_tables(paths, plants, population, config.run_date, register)
    if step == "silver":
        silver.update(export_tables(config.files / "osm_export")[1])
        store.write("silver", silver)
        return {"silver_tables": {k: len(v) for k, v in silver.items()}}
    if step == "gold":
        check = gold_check(model, config.run_dir / "gold")
        info = {
            **model.model_info,
            **check,
            "routes_with_loose_ends": len(silver["grid_de_loose_ends"]),
            "built_at": datetime.now(UTC).isoformat(),
        }
        gold["cim_model_info"] = pd.DataFrame(
            [{"key": k, "value": str(v)} for k, v in info.items()]
        ).assign(model_version=config.run_date)
        store.write("gold", gold)
        return info
    raise ConfigurationError(f"unknown step {step!r}")


def verify_tables(read: Callable[[str, str], pd.DataFrame]) -> list[str]:
    """Problems found in the written gold tables (empty: all checks passed)."""
    problems: list[str] = []
    bus = read("gold", "cim_busbar_section")
    buses = set(bus["mrid"])
    references = {
        "cim_ac_line_segment": ("bus1_mrid", "bus2_mrid"),
        "cim_power_transformer": ("bus1_mrid", "bus2_mrid"),
        "cim_synchronous_machine": ("bus_mrid",),
        "cim_energy_consumer": ("bus_mrid",),
        "cim_external_network_injection": ("bus_mrid",),
    }
    for name, columns in references.items():
        frame = read("gold", name)
        if frame["mrid"].isna().any() or frame["mrid"].duplicated().any():
            problems.append(f"{name}: missing or duplicate mrid")
        for column in columns:
            dangling = (~frame[column].isin(buses)).sum()
            if dangling:
                problems.append(f"{name}.{column}: {dangling} references to unknown buses")
    if bus["mrid"].duplicated().any():
        problems.append("cim_busbar_section: duplicate mrid")
    lat = pd.to_numeric(bus["lat"], errors="coerce")
    lon = pd.to_numeric(bus["lon"], errors="coerce")
    if lat.isna().any() or not lat.between(45.0, 60.0).all() or not lon.between(4.0, 17.5).all():
        problems.append("cim_busbar_section: missing coordinates or outside the region")
    levels = set(pd.to_numeric(read("gold", "cim_voltage_level")["nominal_voltage_kv"]))
    if not levels <= {220.0, 380.0}:
        problems.append(f"cim_voltage_level: unexpected nominal voltages {sorted(levels)}")
    info = read("gold", "cim_model_info").set_index("key")["value"]
    if info.get("loadflow_status") != "CONVERGED":
        problems.append(f"power flow: {info.get('loadflow_status')}")
    return problems


def _spark() -> Any:
    try:
        from databricks.connect import DatabricksSession

        return DatabricksSession.builder.getOrCreate()
    except ImportError:
        from pyspark.sql import SparkSession

        return SparkSession.builder.getOrCreate()


def main(argv: Sequence[str] | None = None) -> int:
    """``tranpy-grid {bronze,silver,gold,verify}`` (see module docs)."""
    parser = argparse.ArgumentParser(prog="tranpy-grid", description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=(*LAYERS, "verify"))
    parser.add_argument("--files")
    parser.add_argument("--catalog")
    parser.add_argument("--plants-table")
    parser.add_argument("--population-table")
    parser.add_argument("--run-date")
    for layer in LAYERS:
        parser.add_argument(f"--{layer}-schema")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = PipelineConfig.from_env(
        files=args.files,
        catalog=args.catalog,
        plants_table=args.plants_table,
        population_table=args.population_table,
        run_date=args.run_date,
        **{f"{layer}_schema": getattr(args, f"{layer}_schema") for layer in LAYERS},
    )
    spark = _spark() if config.catalog else None
    if args.step == "verify":
        problems = verify_tables(Store(config, spark).read)
        for problem in problems:
            logger.error(problem)
        if problems:
            raise DataError(f"{len(problems)} verification problems")
        logger.info("all checks passed")
        return 0
    report = build_layer(args.step, config, spark)
    sys.stdout.write(json.dumps(report, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
