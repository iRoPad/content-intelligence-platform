"""Breakout prediction model.

Question it answers: *before publishing*, how likely is this post to break out
(3x its brand x platform trailing average)? Only pre-publish features are used
(content tags, format, length, timing, collab tier) so the score is usable for
planning, and the train/test split is by time to mimic real deployment.

Also runs an ablation without the AI-generated tags to quantify how much
lift the LLM tagging layer adds to the model.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config, db

CATEGORICAL = ["brand", "platform", "format", "creator_tier", "theme", "hook", "tone"]
NUMERIC = ["log_duration", "is_duration_missing", "hour_sin", "hour_cos", "is_prime_time", "is_weekend", "weeks_since_start"]
AI_TAG_FEATURES = ["theme", "hook", "tone"]
TEST_WEEKS = 16


def load_features() -> pd.DataFrame:
    with db.connect(read_only=True) as con:
        df = db.read_df(con, """
            SELECT post_id, week_start, brand, platform, format, creator_tier, theme, hook, tone,
                   duration_sec, is_duration_missing, published_hour_local, is_weekend, is_breakout
            FROM fct_post_performance
            WHERE is_breakout IS NOT NULL
        """)
    df["week_start"] = pd.to_datetime(df["week_start"])
    # Impute missing durations with the platform median (carousels have no duration: 0)
    med = df.groupby("platform")["duration_sec"].transform("median")
    dur = df["duration_sec"].where(df["duration_sec"].notna() | (df["format"] == "carousel"), med)
    df["log_duration"] = np.log1p(dur.fillna(0))
    h = df["published_hour_local"]
    df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24)
    df["is_prime_time"] = h.between(17, 21).astype(int)
    df["weeks_since_start"] = (df["week_start"] - df["week_start"].min()).dt.days / 7
    df["is_breakout"] = df["is_breakout"].astype(int)
    return df


def time_split(df: pd.DataFrame):
    cutoff = df["week_start"].max() - pd.Timedelta(weeks=TEST_WEEKS - 1)
    return df[df["week_start"] < cutoff], df[df["week_start"] >= cutoff], cutoff


def build_pipeline(categorical, numeric, kind="gbm") -> Pipeline:
    if kind == "gbm":
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("num", "passthrough", numeric),
        ])
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=400, max_leaf_nodes=15, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=config.SEED)
    else:
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ])
        clf = LogisticRegression(max_iter=2000, C=0.5)
    return Pipeline([("pre", pre), ("clf", clf)])


def lift_at(y_true, scores, frac=0.10) -> float:
    n = max(1, int(len(scores) * frac))
    top = np.argsort(-scores)[:n]
    return float(np.mean(np.asarray(y_true)[top]) / np.mean(y_true))


def evaluate(pipe, X, y) -> dict:
    p = pipe.predict_proba(X)[:, 1]
    return {"roc_auc": round(roc_auc_score(y, p), 4),
            "pr_auc": round(average_precision_score(y, p), 4),
            "lift_top_decile": round(lift_at(y, p, 0.10), 2)}


def main():
    df = load_features()
    train, test, cutoff = time_split(df)
    features = CATEGORICAL + NUMERIC
    Xtr, ytr, Xte, yte = train[features], train["is_breakout"], test[features], test["is_breakout"]

    gbm = build_pipeline(CATEGORICAL, NUMERIC, "gbm").fit(Xtr, ytr)
    logit = build_pipeline(CATEGORICAL, NUMERIC, "logit").fit(Xtr, ytr)

    no_tag_cat = [c for c in CATEGORICAL if c not in AI_TAG_FEATURES]
    gbm_no_tags = build_pipeline(no_tag_cat, NUMERIC, "gbm").fit(train[no_tag_cat + NUMERIC], ytr)

    results = {
        "target": f"views >= {config.BREAKOUT_MULTIPLE}x trailing 30-post brand x platform average",
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "test_period_start": cutoff.strftime("%Y-%m-%d"),
        "base_rate_test": round(float(yte.mean()), 4),
        "models": {
            "gradient_boosting": evaluate(gbm, Xte, yte),
            "logistic_regression": evaluate(logit, Xte, yte),
            "gradient_boosting_without_ai_tags": evaluate(gbm_no_tags, test[no_tag_cat + NUMERIC], yte),
        },
    }

    imp = permutation_importance(gbm, Xte, yte, scoring="roc_auc", n_repeats=8, random_state=config.SEED)
    importance = (pd.DataFrame({"feature": features, "importance": imp.importances_mean})
                  .sort_values("importance", ascending=False))
    results["feature_importance"] = [
        {"feature": r.feature, "importance": round(float(r.importance), 4)} for r in importance.itertuples()]

    scored = test[["post_id", "brand", "platform", "theme", "hook", "is_breakout"]].copy()
    scored["breakout_probability"] = gbm.predict_proba(Xte)[:, 1].round(4)
    scored["risk_decile"] = pd.qcut(scored["breakout_probability"].rank(method="first"), 10, labels=False) + 1
    with db.connect() as con:
        db.write_df(con, scored, "ml_breakout_scores")

    (config.ARTIFACTS_DIR / "model_metrics.json").write_text(json.dumps(results, indent=2))
    with open(config.ARTIFACTS_DIR / "breakout_model.pkl", "wb") as f:
        pickle.dump({"pipeline": gbm, "features": features}, f)

    m = results["models"]
    print(f"breakout model | test base rate {results['base_rate_test']:.1%}")
    for name, s in m.items():
        print(f"  {name:<36} ROC-AUC {s['roc_auc']:.3f}  PR-AUC {s['pr_auc']:.3f}  top-decile lift {s['lift_top_decile']:.1f}x")
    print("  top drivers: " + ", ".join(f["feature"] for f in results["feature_importance"][:5]))


if __name__ == "__main__":
    main()
