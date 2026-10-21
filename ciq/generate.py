"""Synthetic data generator.

Produces realistic, *messy* raw exports that look like what you'd pull from
social platform APIs and an ad-spend tracker:

- data/raw/posts.csv          one row per published post (with dupes + casing noise)
- data/raw/post_metrics.csv   daily cumulative metric snapshots, days 0-13
- data/raw/paid_media.csv     weekly paid spend by channel + conversions
- data/raw/_labels.csv        ground-truth content labels (used ONLY to evaluate the AI tagger)
- data/raw/_mmm_truth.json    true channel parameters (used ONLY to validate the MMM)

The performance signal is planted on purpose (theme, hook, timing, duration,
creator collabs) so the downstream AI + ML layers have something real to find.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import config

BRANDS = {
    "courtside_daily": {"vertical": "basketball", "effect": 0.30, "growth": 0.20},
    "pitchside_fc": {"vertical": "soccer", "effect": 0.15, "growth": 0.25},
    "respawn_plays": {"vertical": "gaming", "effect": 0.05, "growth": 0.10},
    "kitchen_heat": {"vertical": "food", "effect": -0.10, "growth": 0.35},
    "gridiron_lab": {"vertical": "football", "effect": 0.20, "growth": 0.05},
    "throws_nation": {"vertical": "track", "effect": -0.30, "growth": 0.40},
}

SUBJECTS = {
    "basketball": ["a deep three", "the crossover", "a poster dunk", "the fadeaway", "a no-look dime"],
    "soccer": ["a bicycle kick", "the free kick", "a nutmeg", "a last-minute winner", "the rabona"],
    "gaming": ["a 1-tap clutch", "the speedrun skip", "a ranked comeback", "the secret boss", "a 1v5"],
    "food": ["smash burgers", "birria tacos", "the perfect steak", "street-corn ribs", "a 5-minute pasta"],
    "football": ["a one-handed catch", "the pick-six", "a 60-yard bomb", "the goal-line stand", "a juke"],
    "track": ["a 70-meter discus throw", "the shot put spin", "a hammer PR", "the javelin release", "a 4x100 handoff"],
}

PLATFORMS = ["tiktok", "instagram", "youtube_shorts", "youtube"]
PLATFORM_WEIGHTS = [0.38, 0.30, 0.20, 0.12]
PLATFORM_BASE = {"tiktok": 9.6, "instagram": 9.0, "youtube_shorts": 9.2, "youtube": 8.7}

THEMES = ["highlight", "tutorial", "behind_the_scenes", "reaction", "trivia", "challenge", "interview"]
HOOKS = ["question", "countdown", "hot_take", "story", "none"]
TONES = ["hype", "funny", "emotional", "informative"]

# Planted effects on log(views). Platform-specific where it matters.
THEME_EFFECT = {
    "highlight": {"tiktok": 0.45, "instagram": 0.35, "youtube_shorts": 0.50, "youtube": 0.05},
    "tutorial": {"tiktok": -0.10, "instagram": 0.10, "youtube_shorts": 0.00, "youtube": 0.45},
    "behind_the_scenes": {"tiktok": 0.05, "instagram": 0.20, "youtube_shorts": -0.05, "youtube": 0.15},
    "reaction": {"tiktok": 0.20, "instagram": 0.00, "youtube_shorts": 0.15, "youtube": 0.10},
    "trivia": {"tiktok": 0.40, "instagram": 0.25, "youtube_shorts": 0.35, "youtube": 0.20},
    "challenge": {"tiktok": 0.55, "instagram": 0.15, "youtube_shorts": 0.30, "youtube": 0.00},
    "interview": {"tiktok": -0.30, "instagram": -0.10, "youtube_shorts": -0.25, "youtube": 0.30},
}
HOOK_EFFECT = {"question": 0.25, "countdown": 0.30, "hot_take": 0.35, "story": 0.15, "none": 0.0}
TONE_EFFECT = {"hype": 0.10, "funny": 0.25, "emotional": 0.15, "informative": 0.0}
COLLAB_EFFECT = {"none": 0.0, "micro": 0.20, "macro": 0.45}

THEME_TEMPLATES = {
    "highlight": ["{s} highlight you need to see", "best {s} moments this week", "clutch: {s} with no time left", "the most insane play: {s}"],
    "tutorial": ["how to master {s}, step by step", "{s} drill that actually works", "technique breakdown: how to nail {s}", "learn {s} in 60 seconds"],
    "behind_the_scenes": ["behind the scenes of {s}", "a day with the crew filming {s}", "BTS access: what you didn't see from {s}", "inside the prep for {s}"],
    "reaction": ["pro reacts to {s}", "our reaction to {s} has us speechless", "watching {s} for the first time", "coach reacts to {s}"],
    "trivia": ["trivia: guess who pulled off {s}", "quiz time: name every player with {s}", "guess the year of {s}", "only real fans get this {s} trivia"],
    "challenge": ["{s} challenge: 1v1 for $100", "we tried to recreate {s}", "challenge accepted: {s} on the first try", "can a rookie do {s}? challenge"],
    "interview": ["the GOAT sits down with us to talk about {s}", "interview: the truth about {s}", "Q&A on {s} with a legend", "she talks about {s} for the first time"],
}
# Keyword-free phrasings: a human (or an LLM) can tell what these are,
# a keyword rule cannot. This is what makes the AI tagger earn its keep.
SEMANTIC_TEMPLATES = {
    "highlight": ["nobody in the building was ready for {s}", "rewind it, then rewind it again: {s}"],
    "tutorial": ["fix your form with this: {s}", "do this every day and {s} gets easy"],
    "behind_the_scenes": ["what the cameras usually miss before {s}", "the hours before {s} looked like this"],
    "reaction": ["our faces when we saw {s} live", "first time seeing {s} and we lost it"],
    "trivia": ["how many players can you name who did {s}", "comment the answer: who did {s} first"],
    "challenge": ["first one to miss {s} buys dinner", "loser of {s} shaves their head"],
    "interview": ["we asked a legend what nobody asks about {s}", "an honest conversation about {s}"],
}
SEMANTIC_RATE = 0.35

HOOK_PREFIX = {
    "question": ["Would you believe this? ", "Can you do this? ", "Who did it better? "],
    "countdown": ["Top 5: ", "3 reasons why: ", "Top 10 ranked: "],
    "hot_take": ["Hot take: ", "Unpopular opinion: ", "Nobody wants to hear this but "],
    "story": ["The story behind this one: ", "Nobody expected what happened next. ", "Here's how it started: "],
    "none": [""],
}
TONE_SUFFIX = {
    "hype": [" 🔥🔥", " LET'S GO", " 🚀", " absolutely unreal", ""],
    "funny": [" 😂", " lol we can't", " 💀💀", " I'm crying", ""],
    "emotional": [" ❤️", " goosebumps every time", " 🥹", " this one hits different", ""],
    "informative": [" 📊", " (full breakdown)", " — save this", " notes in the comments", ""],
}


def _caption(rng, theme, hook, tone, vertical):
    subject = rng.choice(SUBJECTS[vertical])
    pool = SEMANTIC_TEMPLATES if rng.random() < SEMANTIC_RATE else THEME_TEMPLATES
    body = rng.choice(pool[theme]).format(s=subject)
    cap = rng.choice(HOOK_PREFIX[hook]) + body + rng.choice(TONE_SUFFIX[tone])
    return cap[0].upper() + cap[1:]


def _duration(rng, platform, fmt):
    if fmt == "carousel":
        return np.nan
    if platform == "youtube":
        return float(np.clip(rng.lognormal(np.log(600), 0.45), 180, 2400).round())
    return float(np.clip(rng.lognormal(np.log(26), 0.55), 6, 180).round())


def _log_views(rng, row, brand_meta, weeks_elapsed):
    p = row["platform"]
    lv = PLATFORM_BASE[p] + brand_meta["effect"]
    lv += brand_meta["growth"] * weeks_elapsed / 52.0
    lv += THEME_EFFECT[row["theme"]][p] + HOOK_EFFECT[row["hook"]] + TONE_EFFECT[row["tone"]]
    if p == "tiktok" and row["hook"] == "hot_take":
        lv += 0.15
    lv += COLLAB_EFFECT[row["creator_tier"]]
    dur = row["duration_sec"]
    if not np.isnan(dur):
        opt = 600.0 if p == "youtube" else 22.0
        lv -= 0.55 * (np.log(dur) - np.log(opt)) ** 2
    else:  # carousel
        lv -= 0.15
    hour = row["local_hour"]
    if 17 <= hour <= 21:
        lv += 0.25
    elif 1 <= hour <= 6:
        lv -= 0.30
    if row["weekday"] >= 5:
        lv += 0.10
    lv += rng.normal(0, 0.40)
    if rng.random() < 0.015:  # unpredictable viral luck
        lv += rng.uniform(1.2, 2.8)
    return lv


def generate_posts(rng):
    start = datetime.fromisoformat(config.START_DATE)
    rows, labels, metrics = [], [], []
    post_n = 0
    for week in range(config.N_WEEKS):
        n_posts = rng.poisson(config.POSTS_PER_WEEK)
        for _ in range(n_posts):
            post_n += 1
            brand = rng.choice(list(BRANDS))
            meta = BRANDS[brand]
            platform = rng.choice(PLATFORMS, p=PLATFORM_WEIGHTS)
            if platform == "youtube":
                fmt = "long_video"
            elif platform == "instagram":
                fmt = rng.choice(["short_video", "carousel"], p=[0.75, 0.25])
            else:
                fmt = "short_video"
            theme = rng.choice(THEMES)
            hook = rng.choice(HOOKS, p=[0.2, 0.15, 0.15, 0.15, 0.35])
            tone = rng.choice(TONES)
            creator_tier = rng.choice(["none", "micro", "macro"], p=[0.78, 0.15, 0.07])
            # Posting hours skew toward the afternoon/evening (Pacific time)
            local_hour = int(rng.choice(range(24), p=_hour_weights()))
            ts_local = start + timedelta(weeks=week, days=int(rng.integers(0, 7)), hours=local_hour,
                                         minutes=int(rng.integers(0, 60)))
            ts_utc = ts_local + timedelta(hours=7)
            row = {
                "post_id": f"p{post_n:06d}",
                "brand": brand,
                "platform": platform,
                "format": fmt,
                "duration_sec": _duration(rng, platform, fmt),
                "creator_tier": creator_tier,
                "local_hour": local_hour,
                "weekday": ts_local.weekday(),
                "theme": theme, "hook": hook, "tone": tone,
            }
            lv = _log_views(rng, row, meta, week)
            total_views = int(np.exp(lv))
            caption = _caption(rng, theme, hook, tone, meta["vertical"])
            rows.append({
                "post_id": row["post_id"],
                "account_handle": f"@{brand}",
                "platform": platform,
                "media_type": fmt,
                "caption": caption,
                "duration_sec": row["duration_sec"],
                "creator_collab_tier": creator_tier,
                "published_at": ts_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
            labels.append({"post_id": row["post_id"], "theme": theme, "hook": hook, "tone": tone})
            metrics.extend(_daily_snapshots(rng, row, total_views))
    return pd.DataFrame(rows), pd.DataFrame(labels), pd.DataFrame(metrics)


_HOUR_W = None


def _hour_weights():
    global _HOUR_W
    if _HOUR_W is None:
        w = np.array([0.4, 0.2, 0.1, 0.1, 0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 2.5,
                      3, 3, 3, 3, 3.5, 4, 4, 4, 3.5, 3, 2, 1], dtype=float)
        _HOUR_W = w / w.sum()
    return _HOUR_W


def _daily_snapshots(rng, row, total_views):
    k = rng.uniform(0.15, 0.3) if row["platform"] == "youtube" else rng.uniform(0.35, 0.9)
    days = np.arange(14)
    frac = (1 - np.exp(-k * (days + 1))) / (1 - np.exp(-k * 14))
    like_rate = {"funny": 0.075, "hype": 0.065, "emotional": 0.08, "informative": 0.045}[row["tone"]]
    share_rate = 0.012 + (0.01 if row["hook"] == "hot_take" else 0) + (0.008 if row["tone"] == "funny" else 0)
    save_rate = 0.004 + (0.02 if row["theme"] == "tutorial" else 0) + (0.006 if row["theme"] == "trivia" else 0)
    comment_rate = 0.003 + (0.006 if row["hook"] in ("question", "hot_take") else 0)
    dur = row["duration_sec"] if not np.isnan(row["duration_sec"]) else 0.0
    retention = rng.uniform(0.25, 0.7)
    jitter = rng.lognormal(0, 0.15, size=4)
    out = []
    for d, f in zip(days, frac):
        v = int(total_views * f)
        out.append({
            "post_id": row["post_id"],
            "snapshot_day": int(d),
            "views": v,
            "likes": int(v * like_rate * jitter[0]),
            "comments": int(v * comment_rate * jitter[1]),
            "shares": int(v * share_rate * jitter[2]),
            "saves": int(v * save_rate * jitter[3]),
            "watch_time_sec": round(v * dur * retention, 1),
        })
    return out


def make_messy(rng, posts, metrics):
    """Inject the kind of noise real API exports have."""
    posts = posts.copy()
    idx = rng.choice(posts.index, size=int(len(posts) * 0.05), replace=False)
    variants = {"tiktok": "TikTok", "instagram": "Instagram ", "youtube_shorts": "YouTube Shorts", "youtube": "YOUTUBE"}
    posts.loc[idx, "platform"] = posts.loc[idx, "platform"].map(variants)
    vid_idx = posts[posts["media_type"] != "carousel"].index
    null_idx = rng.choice(vid_idx, size=int(len(vid_idx) * 0.01), replace=False)
    posts.loc[null_idx, "duration_sec"] = np.nan
    dupes = posts.sample(frac=0.006, random_state=int(rng.integers(1e9)))
    posts = pd.concat([posts, dupes]).sort_values("published_at").reset_index(drop=True)

    metrics = metrics.copy()
    bad = rng.choice(metrics.index, size=int(len(metrics) * 0.002), replace=False)
    metrics.loc[bad, "views"] = -1  # API glitch rows
    return posts, metrics


def generate_paid_media(rng):
    """Weekly paid spend by channel with adstock + saturation response."""
    channels = {
        #                 decay  half_sat  beta    mean_spend
        "tiktok_ads": (0.30, 18000, 900, 14000),
        "meta_ads": (0.50, 22000, 700, 16000),
        "youtube_ads": (0.60, 25000, 650, 11000),
        "creator_partnerships": (0.70, 15000, 1100, 9000),
        "paid_search": (0.10, 9000, 350, 6000),
    }
    n = config.N_WEEKS
    t = np.arange(n)
    start = datetime.fromisoformat(config.START_DATE)
    df = pd.DataFrame({"week_start": [(start + timedelta(weeks=int(i))).strftime("%Y-%m-%d") for i in t]})
    contribution_total = np.zeros(n)
    truth = {}
    for ch, (decay, half, beta, mean) in channels.items():
        base = mean * (1 + 0.35 * np.sin(2 * np.pi * (t + rng.integers(0, 52)) / 52))
        on = rng.random(n) > 0.18  # flighting: some dark weeks
        spend = np.clip(base * rng.lognormal(0, 0.35, n) * on, 0, None).round(2)
        ad = np.zeros(n)
        for i in range(n):
            ad[i] = spend[i] + (decay * ad[i - 1] if i else 0)
        response = beta * ad / (ad + half)
        contribution_total += response
        df[ch] = spend
        truth[ch] = {"decay": decay, "half_saturation": half, "beta": beta,
                     "total_spend": float(spend.sum()), "total_contribution": float(response.sum()),
                     "roi_per_1k": float(response.sum() / spend.sum() * 1000)}
    base = 1500 + 4.0 * t + 180 * np.sin(2 * np.pi * t / 52 + 1.0)
    df["conversions"] = np.round(base + contribution_total + rng.normal(0, 70, n)).astype(int)
    truth["_base_total"] = float(base.sum())
    return df, truth


def main():
    config.ensure_dirs()
    rng = np.random.default_rng(config.SEED)
    posts, labels, metrics = generate_posts(rng)
    posts, metrics = make_messy(rng, posts, metrics)
    paid, truth = generate_paid_media(rng)
    posts.to_csv(config.RAW_DIR / "posts.csv", index=False)
    metrics.to_csv(config.RAW_DIR / "post_metrics.csv", index=False)
    paid.to_csv(config.RAW_DIR / "paid_media.csv", index=False)
    labels.to_csv(config.RAW_DIR / "_labels.csv", index=False)
    (config.RAW_DIR / "_mmm_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"generated {len(posts):,} post rows, {len(metrics):,} metric snapshots, {len(paid)} paid-media weeks")


if __name__ == "__main__":
    main()
