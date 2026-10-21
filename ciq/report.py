"""Render the dashboard from the warehouse + model artifacts.

Writes docs/index.html, a single self-contained page (data embedded as JSON)
that GitHub Pages serves as the project's live demo.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone

from . import analyst, config, db

TEMPLATE = config.ROOT / "ciq" / "templates" / "dashboard.html"
SITE_URL = "https://iropad.github.io/content-intelligence-platform"
# Link-preview tags so LinkedIn, Slack, X etc. show docs/og-image.png when the dashboard is shared
SOCIAL_META = (
    '<meta property="og:type" content="website">\n'
    '<meta property="og:title" content="Content Intelligence Platform">\n'
    '<meta property="og:description" content="End-to-end analytics + AI for social video: SQL warehouse, LLM content tagging, '
    'breakout prediction and a marketing mix model, with a live dashboard.">\n'
    f'<meta property="og:url" content="{SITE_URL}/">\n'
    f'<meta property="og:image" content="{SITE_URL}/og-image.png">\n'
    '<meta property="og:image:width" content="1200">\n<meta property="og:image:height" content="627">\n'
    '<meta name="twitter:card" content="summary_large_image">\n'
    f'<meta name="twitter:image" content="{SITE_URL}/og-image.png">\n'
    '<meta name="description" content="End-to-end analytics + AI for social video.">\n'
)

SAMPLE_QUESTIONS = [
    "Which hook works best on TikTok?",
    "Do creator collabs help?",
    "Which brands are growing fastest?",
]


def _load(name):
    return json.loads((config.ARTIFACTS_DIR / name).read_text())


def collect() -> dict:
    with db.connect(read_only=True) as con:
        q = lambda sql: db.read_df(con, sql)
        k = q("""SELECT COUNT(*) AS posts, SUM(views) AS views, COUNT(DISTINCT brand) AS brands,
                        AVG(CAST(is_breakout AS DOUBLE)) AS breakout_rate,
                        MIN(published_date_local) AS period_start, MAX(published_date_local) AS period_end
                 FROM fct_post_performance""").iloc[0]
        weekly = q("""SELECT week_start, platform, SUM(views) AS views FROM mart_brand_weekly
                      GROUP BY week_start, platform ORDER BY week_start""")
        themes_p = q("""SELECT platform, theme, AVG(CAST(is_breakout AS DOUBLE)) AS rate, COUNT(*) AS n
                        FROM fct_post_performance WHERE is_breakout IS NOT NULL GROUP BY platform, theme""")
        themes_all = q("""SELECT theme, AVG(CAST(is_breakout AS DOUBLE)) AS rate, COUNT(*) AS n
                          FROM fct_post_performance WHERE is_breakout IS NOT NULL GROUP BY theme""")
        heat = q("""SELECT platform, hook, AVG(CAST(is_breakout AS DOUBLE)) AS rate, COUNT(*) AS n
                    FROM fct_post_performance WHERE is_breakout IS NOT NULL GROUP BY platform, hook""")
        hours = q("""SELECT published_hour_local AS hour, AVG(CAST(is_breakout AS DOUBLE)) AS rate, COUNT(*) AS n
                     FROM fct_post_performance WHERE is_breakout IS NOT NULL GROUP BY published_hour_local ORDER BY 1""")

    platforms = ["tiktok", "instagram", "youtube_shorts", "youtube"]
    hooks = ["hot_take", "countdown", "question", "story", "none"]
    piv = weekly.pivot(index="week_start", columns="platform", values="views").fillna(0)
    # drop the trailing partial week so the line doesn't dive at the end
    piv = piv.iloc[:-1] if len(piv) > 2 else piv
    themes = {"all": themes_all.round(4).to_dict(orient="records")}
    for p in platforms:
        themes[p] = themes_p[themes_p.platform == p][["theme", "rate", "n"]].round(4).to_dict(orient="records")
    hm = heat.set_index(["hook", "platform"])
    matrix = [[{"v": round(float(hm.loc[(h, p), "rate"]), 4), "n": int(hm.loc[(h, p), "n"])} for p in platforms] for h in hooks]

    qa = []
    for question in SAMPLE_QUESTIONS:
        res = analyst.ask(question)
        qa.append({"question": html.escape(question), "sql": html.escape(res["sql"] or ""), "answer": html.escape(res["answer"])})

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "engine": db.ENGINE,
        "breakout_multiple": config.BREAKOUT_MULTIPLE,
        "kpis": {"posts": int(k.posts), "views": int(k.views), "brands": int(k.brands),
                 "breakout_rate": round(float(k.breakout_rate), 4),
                 "period_start": k.period_start, "period_end": k.period_end},
        "weekly": {"weeks": list(piv.index), "series": {p: [int(v) for v in piv[p]] for p in platforms if p in piv}},
        "themes": themes,
        "hook_heat": {"hooks": hooks, "platforms": platforms, "matrix": matrix},
        "hours": [{"hour": int(r.hour), "rate": round(float(r.rate), 4), "n": int(r.n)} for r in hours.itertuples()],
        "tagger": _load("tagger_eval.json"),
        "model": _load("model_metrics.json"),
        "mmm": _load("mmm_results.json"),
        "dq": _load("data_quality.json"),
        "analyst": qa,
    }


def render(data: dict) -> str:
    """Page body (title, styles, markup, script) with the data embedded."""
    payload = json.dumps(data, separators=(",", ":"), default=str).replace("</", "<\\/")
    return TEMPLATE.read_text().replace("__DATA__", payload)


def main():
    data = collect()
    body = render(data)
    config.DOCS_DIR.mkdir(exist_ok=True)
    page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + SOCIAL_META +
            '<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
            'body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n</head>\n<body>\n'
            + body + "\n</body>\n</html>\n")
    (config.DOCS_DIR / "index.html").write_text(page)
    (config.ARTIFACTS_DIR / "dashboard_body.html").write_text(body)
    print(f"dashboard written to {config.DOCS_DIR / 'index.html'} ({len(page) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
