"""Fill the CGMES tables from silver: ``tranpy-cgmes-fill --silver c.s --schema c.s``."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd

from tranpy.cim.mapping import Rows, conform, geometry, network
from tranpy.cim.schema import Table, tables

logger = logging.getLogger(__name__)


def to_records(frame: pd.DataFrame, table: Table) -> list[tuple[Any, ...]]:
    """Rows as Python values of each column's SQL type; a many-valued column gets a list."""
    converted = []
    for column in table.columns:
        values = frame[column.name].astype(object).where(frame[column.name].notna(), None)
        if column.sql_type == "BIGINT":
            values = values.map(lambda v: None if v is None else int(v))
        elif column.sql_type == "DOUBLE":
            values = values.map(lambda v: None if v is None else float(v))
        elif column.sql_type == "BOOLEAN":
            values = values.map(lambda v: None if v is None else bool(v))
        elif column.sql_type.startswith("ARRAY"):
            values = values.map(lambda v: None if v is None else [str(v)])
        else:
            values = values.map(lambda v: None if v is None else str(v))
        converted.append(values.tolist())
    return list(zip(*converted, strict=True))


def build(read: Any) -> Rows:
    """All filled classes from a ``read(silver_table)`` function."""
    nodes = read("grid_de_nodes")
    circuits = read("grid_de_circuits")
    transformers = read("grid_de_transformers")
    return conform({**network(nodes, circuits, transformers), **geometry(nodes, circuits)})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", required=True, help="catalog.schema of the silver tables")
    parser.add_argument("--schema", required=True, help="catalog.schema of the CGMES tables")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    rows = build(lambda name: spark.table(f"{args.silver}.{name}").toPandas())
    schema = {t.name: t for t in tables()}
    for name, frame in rows.items():
        target = f"{args.schema}.{name.lower()}"
        data = spark.createDataFrame(
            to_records(frame, schema[name]), schema=spark.table(target).schema
        )
        data.write.insertInto(target, overwrite=True)
        written = spark.table(target).count()
        logger.info("%s: %d rows", target, written)
        if written != len(frame):
            logger.error("%s: wrote %d of %d rows", target, written, len(frame))
            return 1
    return 0
