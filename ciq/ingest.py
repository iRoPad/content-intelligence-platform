"""Land raw CSV exports in the warehouse untouched (plus load metadata),
and build the calendar dimension. All cleaning happens in SQL staging models."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from . import config, db

RAW_FILES = {
    "raw_posts": "posts.csv",
    "raw_post_metrics": "post_metrics.csv",
    "raw_paid_media": "paid_media.csv",
}


def build_dim_date(start="2024-12-01", end="2027-01-31") -> pd.DataFrame:
    d = pd.date_range(start, end, freq="D")
    return pd.DataFrame({
        "date_key": d.strftime("%Y-%m-%d"),
        "prev_date_key": (d - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "week_start": (d - pd.to_timedelta(d.weekday, unit="D")).strftime("%Y-%m-%d"),
        "month_key": d.strftime("%Y-%m"),
        "weekday_num": d.weekday,
        "weekday_name": d.strftime("%a"),
        "is_weekend": (d.weekday >= 5).astype(int),
    })


def main():
    loaded_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with db.connect() as con:
        for table, fname in RAW_FILES.items():
            df = pd.read_csv(config.RAW_DIR / fname)
            df["_source_file"] = fname
            df["_loaded_at"] = loaded_at
            db.write_df(con, df, table)
            print(f"  loaded {table:<20} {len(df):>8,} rows")
        db.write_df(con, build_dim_date(), "dim_date")
    print(f"ingest complete ({db.ENGINE})")


if __name__ == "__main__":
    main()
