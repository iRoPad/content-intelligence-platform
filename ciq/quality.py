"""Data quality tests on the modeled layer.

Each test is a query returning the number of offending rows. Severity "error"
fails the pipeline (bad data never reaches the models or the dashboard);
"warn" is reported but tolerated.
"""
from __future__ import annotations

import json
import sys

from . import config, db

PLATFORMS = "('tiktok','instagram','youtube_shorts','youtube')"

TESTS = [
    # (name, severity, failing-rows SQL)
    ("stg_posts.post_id is unique", "error",
     "SELECT COUNT(*) FROM (SELECT post_id FROM stg_posts GROUP BY post_id HAVING COUNT(*) > 1) x"),
    ("stg_posts.platform in accepted values", "error",
     f"SELECT COUNT(*) FROM stg_posts WHERE platform NOT IN {PLATFORMS}"),
    ("stg_posts.creator_tier in accepted values", "error",
     "SELECT COUNT(*) FROM stg_posts WHERE creator_tier NOT IN ('none','micro','macro')"),
    ("stg_posts.published_hour_local between 0 and 23", "error",
     "SELECT COUNT(*) FROM stg_posts WHERE published_hour_local NOT BETWEEN 0 AND 23"),
    ("stg_post_metrics.views non-negative", "error",
     "SELECT COUNT(*) FROM stg_post_metrics WHERE views < 0"),
    ("stg_post_metrics engagement <= views", "error",
     "SELECT COUNT(*) FROM stg_post_metrics WHERE likes + comments + shares + saves > views"),
    ("every post has metrics", "error",
     "SELECT COUNT(*) FROM stg_posts p LEFT JOIN stg_post_metrics m ON m.post_id = p.post_id WHERE m.post_id IS NULL"),
    ("every post has AI tags", "error",
     "SELECT COUNT(*) FROM fct_post_performance WHERE theme IS NULL OR hook IS NULL OR tone IS NULL"),
    ("fct rowcount matches stg_posts", "error",
     "SELECT ABS((SELECT COUNT(*) FROM fct_post_performance) - (SELECT COUNT(*) FROM stg_posts))"),
    ("video posts missing duration", "warn",
     "SELECT COUNT(*) FROM stg_posts WHERE is_duration_missing = 1"),
    ("posts tracked < 14 days", "warn",
     "SELECT COUNT(*) FROM stg_post_metrics WHERE days_tracked < 13"),
    ("mmm weeks with zero total spend", "warn",
     "SELECT COUNT(*) FROM mart_mmm_weekly WHERE tiktok_ads + meta_ads + youtube_ads + creator_partnerships + paid_search = 0"),
]


def run() -> list[dict]:
    results = []
    with db.connect(read_only=True) as con:
        for name, severity, sql in TESTS:
            failing = int(con.execute(sql).fetchone()[0])
            status = "pass" if failing == 0 else ("fail" if severity == "error" else "warn")
            results.append({"test": name, "severity": severity, "failing_rows": failing, "status": status})
    return results


def main():
    results = run()
    (config.ARTIFACTS_DIR / "data_quality.json").write_text(json.dumps(results, indent=2))
    for r in results:
        icon = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[r["status"]]
        print(f"  [{icon}] {r['test']}" + (f"  ({r['failing_rows']} rows)" if r["failing_rows"] else ""))
    failed = [r for r in results if r["status"] == "fail"]
    if failed:
        print(f"{len(failed)} data quality test(s) failed")
        sys.exit(1)
    print("data quality: all blocking tests passed")


if __name__ == "__main__":
    main()
