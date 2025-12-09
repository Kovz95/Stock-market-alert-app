#!/usr/bin/env python3
"""
Bulk-import the SQLite datasets used during local development into PostgreSQL.

Usage:
    python migration/import_sqlite_to_postgres.py \
        --database-url postgresql://user:pass@host:port/dbname

If --database-url is omitted the script will fall back to the DATABASE_URL
environment variable.  The script keeps the SQLite databases intact; it only
reads from them and performs upserts into PostgreSQL so it is safe to rerun.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Generator, Iterable, List, Mapping, MutableMapping, Sequence

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values, Json

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMA_FILE = BASE_DIR / "db" / "postgres_schema.sql"

logger = logging.getLogger("sqlite_to_postgres")

BOOL_COLUMNS = {
    "price_data_pulled",
    "conditions_evaluated",
    "alert_triggered",
    "cache_hit",
}


class ImportErrorWithContext(RuntimeError):
    """Raised when we hit an unrecoverable problem and want a clean message."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import local SQLite databases into PostgreSQL."
    )
    parser.add_argument(
        "--database-url",
        dest="database_url",
        default=os.getenv("DATABASE_URL"),
        help="PostgreSQL connection string (defaults to $DATABASE_URL).",
    )
    parser.add_argument(
        "--schema-file",
        dest="schema_file",
        type=Path,
        default=DEFAULT_SCHEMA_FILE,
        help="Path to the PostgreSQL schema SQL file to execute before import.",
    )
    parser.add_argument(
        "--skip-schema",
        dest="skip_schema",
        action="store_true",
        help="Skip applying the schema file before importing data.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=2_000,
        help="Number of rows to insert per batch.",
    )
    parser.add_argument(
        "--tables",
        dest="tables",
        nargs="*",
        help="Optional list of tables to import (defaults to all known tables).",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Load data from SQLite but do not write to PostgreSQL.",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Increase logging verbosity.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def apply_schema(conn, schema_path: Path) -> None:
    stmt_buffer: List[str] = []

    def flush_buffer():
        statement = "".join(stmt_buffer).strip()
        if statement:
            logger.debug("Executing schema statement: %s", statement)
            with conn.cursor() as cur:
                cur.execute(statement)
            conn.commit()

    for line in schema_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        stmt_buffer.append(line)
        if stripped.endswith(";"):
            flush_buffer()
            stmt_buffer.clear()

    if stmt_buffer:
        flush_buffer()


def iter_sqlite_rows(
    conn: sqlite3.Connection, table: str, columns: Sequence[str], batch_size: int
) -> Generator[List[tuple], None, None]:
    placeholder_cols = ", ".join(columns)
    query = f"SELECT {placeholder_cols} FROM {table}"
    cursor = conn.execute(query)
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        yield rows


def _convert_blob_value(value):
    """
    Convert SQLite BLOB/memoryview numeric encodings into Python numbers.

    SQLite sometimes stores integers as 8-byte blobs (little endian). Convert
    those into integers so PostgreSQL accepts them for numeric columns.
    """
    if value is None:
        return None

    if isinstance(value, memoryview):
        value = value.tobytes()

    if isinstance(value, (bytes, bytearray)):
        if len(value) == 0:
            return None

        # Attempt to interpret small blobs as little-endian integers
        if len(value) <= 16:
            try:
                return int.from_bytes(value, byteorder="little", signed=False)
            except Exception:
                pass

        # Fall back to UTF-8 decoding when possible
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            # As a last resort return hex string representation
            return value.hex()

    return value


def normalize_row(columns: Sequence[str], row: Sequence) -> Sequence:
    normalized = []
    for col_name, value in zip(columns, row):
        converted = _convert_blob_value(value)
        if col_name in BOOL_COLUMNS and converted is not None:
            if isinstance(converted, str):
                lowered = converted.strip().lower()
                if lowered in {"true", "false"}:
                    converted = lowered == "true"
                elif lowered.isdigit():
                    converted = bool(int(lowered))
                else:
                    converted = bool(converted)
            else:
                converted = bool(converted)
        normalized.append(converted)
    return normalized


def upsert_batch(
    pg_conn,
    table: str,
    columns: Sequence[str],
    conflict_cols: Sequence[str],
    rows: Iterable[Sequence],
) -> int:
    column_identifiers = sql.SQL(", ").join(sql.Identifier(col) for col in columns)
    conflict_identifiers = sql.SQL(", ").join(
        sql.Identifier(col) for col in conflict_cols
    )
    update_assignments = sql.SQL(", ").join(
        sql.SQL("{col} = EXCLUDED.{col}").format(col=sql.Identifier(col))
        for col in columns
        if col not in conflict_cols
    )

    query = sql.SQL(
        "INSERT INTO {table} ({columns}) VALUES %s "
        "ON CONFLICT ({conflict_cols}) DO UPDATE SET {updates}"
    ).format(
        table=sql.Identifier(table),
        columns=column_identifiers,
        conflict_cols=conflict_identifiers,
        updates=update_assignments,
    )

    with pg_conn.cursor() as cur:
        execute_values(cur, query.as_string(pg_conn), rows, template=None)
    pg_conn.commit()
    return len(list(rows))


def upsert_batch_json_safe(
    pg_conn,
    table: str,
    columns: Sequence[str],
    conflict_cols: Sequence[str],
    rows: Iterable[MutableMapping[str, object]],
) -> int:
    # Convert the dicts in-place to ordered tuples matching `columns`
    tuple_rows = []
    for row in rows:
        cleaned = []
        for col in columns:
            value = row.get(col)
            value = _convert_blob_value(value)
            if col in BOOL_COLUMNS and value is not None:
                if isinstance(value, str):
                    lowered = value.strip().lower()
                    if lowered in {"true", "false"}:
                        value = lowered == "true"
                    elif lowered.isdigit():
                        value = bool(int(lowered))
                    else:
                        value = bool(value)
                else:
                    value = bool(value)
            if col == "additional_data":
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        logger.debug(
                            "Leaving additional_data as plain text for row: %s", row
                        )
                if isinstance(value, (dict, list)):
                    value = Json(value)
            cleaned.append(value)
        tuple_rows.append(tuple(cleaned))
    return upsert_batch(pg_conn, table, columns, conflict_cols, tuple_rows)


def import_table(
    pg_conn,
    sqlite_db: Path,
    table: str,
    columns: Sequence[str],
    conflict_cols: Sequence[str],
    batch_size: int,
    *,
    json_columns: bool = False,
) -> int:
    sqlite_path = BASE_DIR / sqlite_db
    if not sqlite_path.exists():
        raise ImportErrorWithContext(f"SQLite database {sqlite_path} does not exist")

    logger.info("Importing %s.%s (%s)", sqlite_db, table, ", ".join(columns))
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    total_rows = 0

    try:
        for rows in iter_sqlite_rows(conn, table, columns, batch_size):
            total_rows += len(rows)
            if json_columns:
                dict_rows = [dict(row) for row in rows]
                # Normalize potential blob encodings before JSON handling
                for item in dict_rows:
                    for key in columns:
                        item[key] = _convert_blob_value(item.get(key))
                upsert_batch_json_safe(
                    pg_conn, table, columns, conflict_cols, dict_rows
                )
            else:
                tuple_rows = [
                    tuple(normalize_row(columns, row))
                    for row in rows
                ]
                upsert_batch(pg_conn, table, columns, conflict_cols, tuple_rows)
            logger.debug("Imported %s rows into %s", total_rows, table)
    finally:
        conn.close()

    return total_rows


def refresh_sequence(pg_conn, table: str, column: str) -> None:
    sequence_name = f"{table}_{column}_seq"
    query = sql.SQL(
        "SELECT setval({seq}, COALESCE((SELECT MAX({col}) FROM {table}), 0) + 1, false)"
    ).format(
        seq=sql.Literal(sequence_name),
        col=sql.Identifier(column),
        table=sql.Identifier(table),
    )
    with pg_conn.cursor() as cur:
        cur.execute(query)
    pg_conn.commit()


TABLE_SPECS: Mapping[str, Mapping[str, object]] = {
    "daily_prices": {
        "sqlite_db": Path("price_data.db"),
        "columns": [
            "ticker",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "updated_at",
        ],
        "conflict": ["ticker", "date"],
    },
    "hourly_prices": {
        "sqlite_db": Path("price_data.db"),
        "columns": [
            "ticker",
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "updated_at",
        ],
        "conflict": ["ticker", "datetime"],
    },
    "weekly_prices": {
        "sqlite_db": Path("price_data.db"),
        "columns": [
            "ticker",
            "week_ending",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "updated_at",
        ],
        "conflict": ["ticker", "week_ending"],
    },
    "ticker_metadata": {
        "sqlite_db": Path("price_data.db"),
        "columns": [
            "ticker",
            "first_date",
            "last_date",
            "total_records",
            "last_update",
            "exchange",
            "asset_type",
        ],
        "conflict": ["ticker"],
    },
    "daily_move_stats": {
        "sqlite_db": Path("price_data.db"),
        "columns": [
            "ticker",
            "date",
            "pct_change",
            "mean_change",
            "std_change",
            "zscore",
            "sigma_level",
            "direction",
            "magnitude",
            "updated_at",
        ],
        "conflict": ["ticker", "date"],
    },
    "alert_audits": {
        "sqlite_db": Path("alert_audit.db"),
        "columns": [
            "id",
            "timestamp",
            "alert_id",
            "ticker",
            "stock_name",
            "exchange",
            "timeframe",
            "action",
            "evaluation_type",
            "price_data_pulled",
            "price_data_source",
            "conditions_evaluated",
            "alert_triggered",
            "trigger_reason",
            "execution_time_ms",
            "cache_hit",
            "error_message",
            "additional_data",
        ],
        "conflict": ["id"],
        "json_columns": True,
    },
    "continuous_prices": {
        "sqlite_db": Path("futures_price_data.db"),
        "columns": [
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adjustment_method",
            "front_month",
            "roll_date",
            "created_at",
        ],
        "conflict": ["symbol", "date"],
    },
    "futures_metadata": {
        "sqlite_db": Path("futures_price_data.db"),
        "columns": [
            "symbol",
            "name",
            "exchange",
            "category",
            "multiplier",
            "min_tick",
            "currency",
            "contract_id",
            "last_update",
            "front_month",
            "next_roll_date",
            "data_quality_score",
        ],
        "conflict": ["symbol"],
    },
}


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    if not args.database_url:
        raise ImportErrorWithContext(
            "DATABASE_URL is not set. Provide --database-url or export the variable."
        )

    tables_to_process = args.tables or list(TABLE_SPECS.keys())
    unknown = set(tables_to_process) - set(TABLE_SPECS.keys())
    if unknown:
        raise ImportErrorWithContext(
            f"Unknown table(s): {', '.join(sorted(unknown))}. "
            f"Known tables: {', '.join(sorted(TABLE_SPECS))}"
        )

    logger.info("Connecting to PostgreSQL: %s", args.database_url)
    pg_conn = psycopg2.connect(args.database_url)

    try:
        if not args.skip_schema:
            logger.info("Applying schema from %s", args.schema_file)
            apply_schema(pg_conn, args.schema_file)

        total_imported: MutableMapping[str, int] = {}
        if args.dry_run:
            logger.warning("Running in dry-run mode; no data will be written.")

        for table in tables_to_process:
            spec = TABLE_SPECS[table]

            if args.dry_run:
                sqlite_path = BASE_DIR / spec["sqlite_db"]
                conn = sqlite3.connect(sqlite_path)
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                conn.close()
                logger.info("Would import %s rows into %s", count, table)
                total_imported[table] = count
                continue

            imported = import_table(
                pg_conn=pg_conn,
                sqlite_db=spec["sqlite_db"],
                table=table,
                columns=spec["columns"],
                conflict_cols=spec["conflict"],
                json_columns=spec.get("json_columns", False),
                batch_size=args.batch_size,
            )
            total_imported[table] = imported
            logger.info("Imported %s rows into %s", imported, table)

        if not args.dry_run and "alert_audits" in tables_to_process:
            refresh_sequence(pg_conn, "alert_audits", "id")

        logger.info("Import complete:")
        for table, count in total_imported.items():
            logger.info("  %s: %s rows", table, count)

    finally:
        pg_conn.close()


if __name__ == "__main__":
    try:
        main()
    except ImportErrorWithContext as exc:
        logger.error(str(exc))
        raise SystemExit(1)
BOOL_COLUMNS = {
    "price_data_pulled",
    "conditions_evaluated",
    "alert_triggered",
    "cache_hit",
}
