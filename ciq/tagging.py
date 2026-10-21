"""AI content tagging.

Classifies every caption into a content taxonomy (theme, hook, tone) so that
unstructured text becomes model features and reportable dimensions.

- With ANTHROPIC_API_KEY set: batched LLM classification with a strict JSON
  contract, results cached by caption hash so re-runs cost nothing.
- Without a key: a transparent keyword-rules baseline.

Either way, accuracy is measured against a held-out labeled set, which is how
you'd decide in practice whether the LLM is worth its cost over the rules.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone

import pandas as pd

from . import config, db, llm

TAXONOMY = {
    "theme": ["highlight", "tutorial", "behind_the_scenes", "reaction", "trivia", "challenge", "interview"],
    "hook": ["question", "countdown", "hot_take", "story", "none"],
    "tone": ["hype", "funny", "emotional", "informative"],
}

THEME_RULES = {
    "trivia": ["trivia", "quiz", "guess", "name every", "real fans"],
    "challenge": ["challenge", "1v1", "we tried", "recreate", "first try"],
    "tutorial": ["how to", "step by step", "drill", "technique", "learn "],
    "interview": ["sits down", "interview", "q&a", "talks about"],
    "reaction": ["reacts", "reaction", "watching"],
    "behind_the_scenes": ["behind the scenes", "bts", "a day with", "inside the"],
    "highlight": ["highlight", "moments", "clutch", "insane play"],
}
TONE_RULES = {
    "funny": ["😂", "lol", "💀"],
    "emotional": ["❤️", "goosebumps", "🥹"],
    "hype": ["🔥", "let's go", "🚀"],
    "informative": ["📊", "breakdown", "save this"],
}


def rules_tag(caption: str) -> dict:
    c = caption.lower()
    scores = {t: sum(k in c for k in kws) for t, kws in THEME_RULES.items()}
    theme = max(scores, key=scores.get) if max(scores.values()) else "highlight"

    if any(k in c for k in ("hot take", "unpopular opinion", "nobody wants to hear")):
        hook = "hot_take"
    elif re.match(r"^(top \d+|\d+ reasons)", c):
        hook = "countdown"
    elif any(k in c for k in ("story behind", "what happened next", "how it started")):
        hook = "story"
    elif "?" in c:
        hook = "question"
    else:
        hook = "none"

    tscores = {t: sum(k in c for k in kws) for t, kws in TONE_RULES.items()}
    tone = max(tscores, key=tscores.get) if max(tscores.values()) else "informative"
    return {"theme": theme, "hook": hook, "tone": tone}


SYSTEM_PROMPT = f"""You are a social video content analyst. Classify each post caption
using ONLY these labels:
{json.dumps(TAXONOMY, indent=2)}

Definitions:
- theme: what the content IS (e.g. trivia = guessing/quiz format; challenge = attempting a feat; highlight = showing a great play).
- hook: the opening device. countdown = ranked list ("Top 5", "3 reasons"); hot_take = provocative opinion;
  story = narrative tease; question = opens by asking the viewer something; none = no hook device.
- tone: overall emotional register.

Return ONLY a JSON array, one object per caption, in input order:
[{{"id": "<id>", "theme": "...", "hook": "...", "tone": "..."}}]"""


def _key(caption: str) -> str:
    return hashlib.sha1(caption.encode()).hexdigest()[:16]


def _validate(tag: dict) -> dict | None:
    if all(tag.get(f) in opts for f, opts in TAXONOMY.items()):
        return {f: tag[f] for f in TAXONOMY}
    return None


def llm_tag_batch(captions: list[tuple[str, str]]) -> dict[str, dict]:
    prompt = "\n".join(f'{{"id": "{pid}", "caption": {json.dumps(cap, ensure_ascii=False)}}}' for pid, cap in captions)
    out = llm.extract_json(llm.complete(SYSTEM_PROMPT, prompt, max_tokens=4000))
    results = {}
    for item in out:
        ok = _validate(item)
        if ok:
            results[str(item.get("id"))] = ok
    return results


def tag_captions(posts: pd.DataFrame, use_llm: bool | None = None) -> pd.DataFrame:
    use_llm = llm.available() if use_llm is None else use_llm
    cache_path = config.WAREHOUSE_DIR / "tag_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    tags, sources = {}, {}
    todo = []
    for pid, cap in posts[["post_id", "caption"]].itertuples(index=False):
        hit = cache.get(_key(cap)) if use_llm else None
        if hit:
            tags[pid], sources[pid] = hit, "llm_cache"
        else:
            todo.append((pid, cap))

    if use_llm and todo:
        n_batches = (len(todo) + config.LLM_BATCH_SIZE - 1) // config.LLM_BATCH_SIZE
        print(f"  tagging {len(todo):,} captions with {config.LLM_MODEL} in {n_batches} batches")
        caption_of = dict(todo)
        for i in range(0, len(todo), config.LLM_BATCH_SIZE):
            batch = todo[i:i + config.LLM_BATCH_SIZE]
            try:
                res = llm_tag_batch(batch)
            except Exception as exc:  # network, parse, rate limit -> degrade gracefully
                print(f"  batch {i // config.LLM_BATCH_SIZE} failed ({exc.__class__.__name__}), using rules")
                res = {}
            for pid, tag in res.items():
                if pid in caption_of:
                    tags[pid], sources[pid] = tag, "llm"
                    cache[_key(caption_of[pid])] = tag
        cache_path.write_text(json.dumps(cache))

    for pid, cap in todo:
        if pid not in tags:
            tags[pid], sources[pid] = rules_tag(cap), "rules"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return pd.DataFrame([
        {"post_id": pid, **tags[pid], "tag_source": sources[pid], "tagged_at": now}
        for pid in posts["post_id"]
    ])


def evaluate(tags: pd.DataFrame) -> dict:
    """Accuracy per field against the held-out ground-truth labels."""
    labels = pd.read_csv(config.RAW_DIR / "_labels.csv")
    m = tags.merge(labels, on="post_id", suffixes=("_pred", "_true"))
    return {f: round(float((m[f"{f}_pred"] == m[f"{f}_true"]).mean()), 4) for f in TAXONOMY} | {"n": len(m)}


def main(use_llm: bool | None = None):
    with db.connect() as con:
        posts = db.read_df(con, "SELECT DISTINCT post_id, caption FROM raw_posts")
        tags = tag_captions(posts, use_llm)
        db.write_df(con, tags, "raw_post_tags")
    scores = evaluate(tags)
    (config.ARTIFACTS_DIR / "tagger_eval.json").write_text(json.dumps(
        {"accuracy": scores, "source_mix": tags["tag_source"].value_counts().to_dict()}, indent=2))
    print(f"tagged {len(tags):,} posts | accuracy theme={scores['theme']:.1%} hook={scores['hook']:.1%} tone={scores['tone']:.1%}")


if __name__ == "__main__":
    main()
