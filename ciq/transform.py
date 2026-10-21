"""Run the SQL models in /sql in order (a tiny dbt-style runner).

Supports {{ variable }} templating from config so business rules such as the
breakout threshold live in one place.
"""
from __future__ import annotations

import re
import time

from . import config, db

VARIABLES = {"breakout_multiple": config.BREAKOUT_MULTIPLE}


def render(sql: str) -> str:
    def sub(m):
        name = m.group(1)
        if name not in VARIABLES:
            raise KeyError(f"undefined SQL variable: {name}")
        return str(VARIABLES[name])
    return re.sub(r"\{\{\s*(\w+)\s*\}\}", sub, sql)


def model_files():
    return sorted(config.SQL_DIR.glob("*.sql"))


def main():
    with db.connect() as con:
        for path in model_files():
            t0 = time.perf_counter()
            db.execute_script(con, render(path.read_text()))
            table = path.stem.split("_", 1)[1]
            n = db.read_df(con, f"SELECT COUNT(*) AS n FROM {table}")["n"].iloc[0]
            print(f"  built {table:<26} {n:>8,} rows  ({time.perf_counter() - t0:.2f}s)")
    print("transform complete")


if __name__ == "__main__":
    main()
