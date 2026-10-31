"""Create the empty CGMES 3.0 tables: ``tranpy-cgmes --schema catalog.cgmes_v3_0``."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from tranpy.cim.schema import create_statement, tables

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", required=True, help="catalog.schema")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    catalog, name = args.schema.split(".")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{name}")
    wanted = tables()
    for table in wanted:
        spark.sql(create_statement(table, args.schema))
    found = {r.tableName for r in spark.sql(f"SHOW TABLES IN {args.schema}").collect()}
    missing = {t.name.lower() for t in wanted} - found
    extra = found - {t.name.lower() for t in wanted}
    logger.info("%d tables wanted, %d found", len(wanted), len(found))
    if missing or extra:
        logger.error("missing %s; not in the schema %s", sorted(missing), sorted(extra))
        return 1
    return 0
