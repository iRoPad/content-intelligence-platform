"""'Ask your data' AI analyst: natural language -> SQL -> answer.

With an API key the LLM writes SQL against a documented schema. Every query
passes guardrails before it touches the warehouse:
  - single read-only SELECT/WITH statement
  - only allow-listed tables
  - row limit enforced
  - executed on a read-only connection
The result rows (not the whole warehouse) go back to the LLM for a plain-English
answer. Without a key, a small intent router answers common questions with
vetted SQL templates, so the CLI and tests work offline.

Usage:  python -m ciq ask "which hook works best on tiktok?"
"""
from __future__ import annotations

import re
import sys

import pandas as pd

from . import db, llm

SCHEMA_DOC = """
fct_post_performance  -- one row per post
  post_id, brand, platform ('tiktok','instagram','youtube_shorts','youtube'), format ('short_video','long_video','carousel'),
  caption, duration_sec, creator_tier ('none','micro','macro'), published_at_utc, published_date_local (YYYY-MM-DD),
  published_hour_local (0-23, Pacific), week_start (YYYY-MM-DD, Monday), month_key (YYYY-MM), weekday_num (0=Mon),
  is_weekend, theme, hook, tone (AI tags), views, views_day1, likes, comments, shares, saves, engagements,
  engagement_rate, share_rate, avg_pct_viewed, baseline_views, views_vs_baseline,
  is_breakout (1/0, NULL when not enough history)
mart_brand_weekly        -- week_start, brand, platform, posts, views, engagements, engagement_rate, breakouts, views_per_post
mart_content_benchmarks  -- platform, theme, hook, posts, avg_views, avg_views_vs_baseline, avg_engagement_rate, breakout_rate
mart_mmm_weekly          -- week_start, tiktok_ads, meta_ads, youtube_ads, creator_partnerships, paid_search (USD spend), conversions, organic_views, organic_posts
ml_breakout_scores       -- post_id, brand, platform, theme, hook, is_breakout, breakout_probability, risk_decile (test period only)
""".strip()

ALLOWED_TABLES = {"fct_post_performance", "mart_brand_weekly", "mart_content_benchmarks",
                  "mart_mmm_weekly", "ml_breakout_scores", "dim_date"}
FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|copy|export|"
                       r"install|load|vacuum|grant|truncate|call)\b", re.I)
MAX_ROWS = 200


class UnsafeQuery(ValueError):
    pass


def validate_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";").strip()
    if ";" in sql:
        raise UnsafeQuery("multiple statements are not allowed")
    if not re.match(r"^(select|with)\b", sql, re.I):
        raise UnsafeQuery("only SELECT queries are allowed")
    if FORBIDDEN.search(sql):
        raise UnsafeQuery("query contains a forbidden keyword")
    ctes = {m.lower() for m in re.findall(r"(?:with|,)\s*(\w+)\s+as\s*\(", sql, re.I)}
    referenced = {t.lower() for t in re.findall(r"\b(?:from|join)\s+([a-zA-Z_][\w]*)", sql, re.I)}
    unknown = referenced - ALLOWED_TABLES - ctes
    if unknown:
        raise UnsafeQuery(f"table(s) not allowed: {', '.join(sorted(unknown))}")
    if not re.search(r"\blimit\s+\d+\s*$", sql, re.I):
        sql = f"SELECT * FROM ({sql}) AS q LIMIT {MAX_ROWS}"
    return sql


def run_sql(sql: str) -> pd.DataFrame:
    safe = validate_sql(sql)
    with db.connect(read_only=True) as con:
        return db.read_df(con, safe)


# ---------------------------------------------------------------- offline router
PLATFORM_WORDS = {"tiktok": "tiktok", "instagram": "instagram", "ig": "instagram", "reels": "instagram",
                  "shorts": "youtube_shorts", "youtube": "youtube"}


def _platform_filter(q: str) -> str:
    ql = q.lower()
    if "shorts" in ql:
        return "AND platform = 'youtube_shorts'"
    for word, p in PLATFORM_WORDS.items():
        if re.search(rf"\b{word}\b", ql):
            return f"AND platform = '{p}'"
    return ""


def template_sql(question: str) -> tuple[str, str] | None:
    q = question.lower()
    pf = _platform_filter(question)
    breakout = "ROUND(AVG(CAST(is_breakout AS DOUBLE)) * 100, 1) AS breakout_rate_pct"
    if any(w in q for w in ("spend", "channel", "roi", "budget", "ads", "media mix")):
        return ("Total paid spend by channel and the conversions recorded over the period.",
                "SELECT SUM(tiktok_ads) AS tiktok_ads, SUM(meta_ads) AS meta_ads, SUM(youtube_ads) AS youtube_ads, "
                "SUM(creator_partnerships) AS creator_partnerships, SUM(paid_search) AS paid_search, "
                "SUM(conversions) AS conversions FROM mart_mmm_weekly")
    for field in ("hook", "theme", "tone"):
        if field in q or (field == "theme" and "content type" in q):
            return (f"Breakout rate by {field}{' on ' + pf.split(chr(39))[1] if pf else ''}.",
                    f"SELECT {field}, COUNT(*) AS posts, {breakout}, ROUND(AVG(views)) AS avg_views "
                    f"FROM fct_post_performance WHERE is_breakout IS NOT NULL {pf} "
                    f"GROUP BY {field} ORDER BY breakout_rate_pct DESC")
    if any(w in q for w in ("hour", "time", "when", "post at")):
        return ("Breakout rate by local posting hour (Pacific).",
                f"SELECT published_hour_local AS hour, COUNT(*) AS posts, {breakout} FROM fct_post_performance "
                f"WHERE is_breakout IS NOT NULL {pf} GROUP BY published_hour_local ORDER BY hour")
    if any(w in q for w in ("collab", "creator", "partner")):
        return ("Performance by creator collaboration tier.",
                f"SELECT creator_tier, COUNT(*) AS posts, {breakout}, ROUND(AVG(views)) AS avg_views "
                f"FROM fct_post_performance WHERE is_breakout IS NOT NULL {pf} GROUP BY creator_tier ORDER BY breakout_rate_pct DESC")
    if any(w in q for w in ("length", "duration", "long", "seconds")):
        return ("Breakout rate by video length bucket.",
                "SELECT CASE WHEN duration_sec < 15 THEN 'a) <15s' WHEN duration_sec < 30 THEN 'b) 15-29s' "
                "WHEN duration_sec < 60 THEN 'c) 30-59s' WHEN duration_sec < 300 THEN 'd) 1-5m' ELSE 'e) 5m+' END AS length_bucket, "
                f"COUNT(*) AS posts, {breakout} FROM fct_post_performance WHERE is_breakout IS NOT NULL AND duration_sec IS NOT NULL {pf} "
                "GROUP BY length_bucket ORDER BY length_bucket")
    if any(w in q for w in ("brand", "account", "growing", "growth")):
        return ("Views by brand: last 8 weeks vs the 8 weeks before.",
                "WITH w AS (SELECT DISTINCT week_start FROM mart_brand_weekly ORDER BY week_start DESC LIMIT 16), "
                "r AS (SELECT week_start, ROW_NUMBER() OVER (ORDER BY week_start DESC) AS rn FROM w) "
                "SELECT b.brand, SUM(CASE WHEN r.rn <= 8 THEN b.views ELSE 0 END) AS views_last_8w, "
                "SUM(CASE WHEN r.rn > 8 THEN b.views ELSE 0 END) AS views_prior_8w, "
                "ROUND((SUM(CASE WHEN r.rn <= 8 THEN b.views ELSE 0 END) * 1.0 / "
                "NULLIF(SUM(CASE WHEN r.rn > 8 THEN b.views ELSE 0 END), 0) - 1) * 100, 1) AS growth_pct "
                f"FROM mart_brand_weekly b JOIN r ON r.week_start = b.week_start WHERE 1=1 {pf.replace('platform', 'b.platform')} "
                "GROUP BY b.brand ORDER BY growth_pct DESC")
    if any(w in q for w in ("top", "best post", "viral", "breakout")):
        return ("Top posts by views relative to their baseline.",
                f"SELECT post_id, brand, platform, theme, hook, views, ROUND(views_vs_baseline, 1) AS x_baseline "
                f"FROM fct_post_performance WHERE views_vs_baseline IS NOT NULL {pf} ORDER BY views_vs_baseline DESC LIMIT 10")
    return None


def offline_summary(rows: pd.DataFrame) -> str:
    """Deterministic one-line takeaway for the template answers."""
    if rows.empty:
        return "No rows matched."
    label = rows.columns[0]
    for metric, unit in (("breakout_rate_pct", "% breakout rate"), ("growth_pct", "% view growth")):
        if metric in rows.columns and len(rows) > 1:
            r = rows.dropna(subset=[metric])
            if "posts" in r.columns:
                r = r[r["posts"] >= 30]  # ignore tiny samples
            if len(r) < 2:
                break
            hi, lo = r.loc[r[metric].idxmax()], r.loc[r[metric].idxmin()]
            return (f"Best: {hi[label]} at {hi[metric]:g}{unit}; weakest: {lo[label]} at {lo[metric]:g}{unit}"
                    + (" (groups under 30 posts excluded)." if "posts" in rows.columns else "."))
    return f"{len(rows)} row(s) returned."


# ---------------------------------------------------------------- LLM path
SQL_SYSTEM = f"""You are a senior analytics engineer. Write ONE read-only SQL SELECT query that
answers the user's question using only these tables:

{SCHEMA_DOC}

Rules: portable SQL (works in both DuckDB and SQLite). Multiply by 1.0 before dividing.
Prefer breakout rate over raw average views when comparing content (views are heavy-tailed).
Exclude rows where is_breakout IS NULL when computing breakout rates. Keep result sets small.
Respond with JSON only: {{"sql": "...", "explanation": "one sentence on the approach"}}"""

ANSWER_SYSTEM = """You are a social analytics lead briefing a content team. Answer the question in
2-4 sentences using ONLY the query results provided. Lead with the takeaway, cite specific numbers,
and flag small sample sizes (under ~30 posts) as directional."""


def ask(question: str, use_llm: bool | None = None) -> dict:
    use_llm = llm.available() if use_llm is None else use_llm
    if use_llm:
        plan = llm.extract_json(llm.complete(SQL_SYSTEM, question, max_tokens=800))
        explanation, sql = plan.get("explanation", ""), plan["sql"]
    else:
        routed = template_sql(question)
        if routed is None:
            return {"question": question, "sql": None, "rows": pd.DataFrame(),
                    "answer": "Offline mode can answer questions about themes, hooks, tone, posting time, "
                              "video length, creator collabs, brand growth, top posts and paid spend. "
                              "Set ANTHROPIC_API_KEY for free-form questions."}
        explanation, sql = routed
    rows = run_sql(sql)
    if use_llm:
        answer = llm.complete(ANSWER_SYSTEM, f"Question: {question}\n\nResults:\n{rows.head(50).to_string(index=False)}",
                              max_tokens=400)
    else:
        answer = explanation + " " + offline_summary(rows)
    return {"question": question, "sql": sql, "rows": rows, "answer": answer.strip()}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    question = " ".join(argv) or "Which hook works best on TikTok?"
    res = ask(question)
    print(f"\nQ: {res['question']}\n")
    if res["sql"]:
        print(f"SQL:\n  {res['sql']}\n")
        print(res["rows"].to_string(index=False))
    print(f"\nA: {res['answer']}\n")


if __name__ == "__main__":
    main()
