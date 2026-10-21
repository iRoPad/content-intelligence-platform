"""Warehouse connection adapter.

Defaults to SQLite from the standard library so the project runs anywhere with
zero setup. Set CIQ_ENGINE=duckdb (and `pip install duckdb`) to run the same
models on DuckDB. The SQL in /sql sticks to the portable subset both engines share.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager

import pandas as pd

from . import config

ENGINE = os.getenv("CIQ_ENGINE", "sqlite").lower()
duckdb = None
if ENGINE == "duckdb":  # pragma: no cover - depends on environment
    try:
        import duckdb  # type: ignore
    except ImportError:
        print("CIQ_ENGINE=duckdb but duckdb is not installed; falling back to sqlite")
        ENGINE = "sqlite"


def db_path():
    ext = "duckdb" if ENGINE == "duckdb" else "db"
    return config.WAREHOUSE_DIR / f"ciq.{ext}"


@contextmanager
def connect(read_only: bool = False):
    config.ensure_dirs()
    if ENGINE == "duckdb":
        con = duckdb.connect(str(db_path()), read_only=read_only)
    else:
        uri = f"file:{db_path()}{'?mode=ro' if read_only else ''}"
        con = sqlite3.connect(uri, uri=True)
    try:
        yield con
    finally:
        con.close()


def write_df(con, df: pd.DataFrame, table: str) -> None:
    if ENGINE == "duckdb":
        con.execute(f"DROP TABLE IF EXISTS {table}")
        # make missing values real NULLs (pandas uses NaN) so IS NULL logic matches SQLite
        df = df.astype(object).where(df.notna(), None)
        con.register("_tmp_df", df)
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM _tmp_df")
        con.unregister("_tmp_df")
    else:
        df.to_sql(table, con, if_exists="replace", index=False)
        con.commit()


def read_df(con, sql: str) -> pd.DataFrame:
    if ENGINE == "duckdb":
        return con.execute(sql).df()
    return pd.read_sql_query(sql, con)


def execute_script(con, sql: str) -> None:
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    for stmt in statements:
        con.execute(stmt)
    if ENGINE == "sqlite":
        con.commit()


def list_tables(con) -> list[str]:
    if ENGINE == "duckdb":
        rows = con.execute("SELECT table_name FROM information_schema.tables ORDER BY 1").fetchall()
    else:
        rows = con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1").fetchall()
    return [r[0] for r in rows]


def table_columns(con, table: str) -> list[str]:
    return list(read_df(con, f"SELECT * FROM {table} LIMIT 0").columns)
