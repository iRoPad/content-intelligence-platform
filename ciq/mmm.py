"""Lightweight marketing mix model (MMM).

conversions_t = base + trend + seasonality + organic
              + sum_c beta_c * hill(adstock_c(spend_c,t))

- Adstock: geometric carryover, a_t = spend_t + decay * a_{t-1}
- Saturation: hill curve a / (a + half_saturation)
- Fit: coordinate-descent grid search over (decay, half-saturation) per channel,
  bounded least squares for coefficients (media effects constrained >= 0)
- Output: channel contributions, ROI, marginal ROI, and a budget reallocation
  that holds total spend flat.

Because the data is synthetic, the true parameters are known, so the script
also reports how well the model recovers them. That's the kind of validation
you rarely get with real MMMs and it's useful for showing the method works.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from . import config, db

CHANNELS = ["tiktok_ads", "meta_ads", "youtube_ads", "creator_partnerships", "paid_search"]
DECAYS = np.round(np.arange(0.0, 0.85, 0.1), 2)
HALF_SAT_MULTIPLES = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
HOLDOUT_WEEKS = 13


def adstock(x: np.ndarray, decay: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    for i, v in enumerate(x):
        out[i] = v + (decay * out[i - 1] if i else 0.0)
    return out


def hill(a: np.ndarray, half: float) -> np.ndarray:
    return a / (a + half)


def controls(df: pd.DataFrame) -> pd.DataFrame:
    t = np.arange(len(df))
    ov = df["organic_views"].to_numpy(float)
    return pd.DataFrame({
        "intercept": 1.0,
        "trend": t / len(t),
        "season_sin": np.sin(2 * np.pi * t / 52),
        "season_cos": np.cos(2 * np.pi * t / 52),
        "organic_views": (ov - ov.mean()) / (ov.std() or 1),
    })


class MMM:
    def __init__(self):
        self.params: dict[str, dict] = {}
        self.coef: dict[str, float] = {}

    def _design(self, df, params):
        X = controls(df)
        for ch in CHANNELS:
            a = adstock(df[ch].to_numpy(float), params[ch]["decay"])
            X[ch] = hill(a, params[ch]["half_saturation"])
        return X

    def _solve(self, X, y):
        lb = np.array([0 if c in CHANNELS else -np.inf for c in X.columns])
        res = lsq_linear(X.to_numpy(), y, bounds=(lb, np.inf))
        pred = X.to_numpy() @ res.x
        return res.x, float(np.mean((y - pred) ** 2))

    def fit(self, df: pd.DataFrame, passes: int = 3):
        y = df["conversions"].to_numpy(float)
        mean_ad = {ch: max(1.0, df[ch].mean()) for ch in CHANNELS}
        params = {ch: {"decay": 0.5, "half_saturation": mean_ad[ch] / 0.5} for ch in CHANNELS}
        for _ in range(passes):
            for ch in CHANNELS:
                best = None
                for d in DECAYS:
                    steady = mean_ad[ch] / (1 - d)
                    for m in HALF_SAT_MULTIPLES:
                        trial = {**params, ch: {"decay": float(d), "half_saturation": float(steady * m)}}
                        _, mse = self._solve(self._design(df, trial), y)
                        if best is None or mse < best[0]:
                            best = (mse, trial[ch])
                params[ch] = best[1]
        X = self._design(df, params)
        coef, _ = self._solve(X, y)
        self.params, self.columns = params, list(X.columns)
        self.coef = dict(zip(self.columns, coef))
        return self

    def predict(self, df) -> np.ndarray:
        return self._design(df, self.params).to_numpy() @ np.array([self.coef[c] for c in self.columns])

    def contributions(self, df) -> pd.DataFrame:
        X = self._design(df, self.params)
        return pd.DataFrame({c: X[c] * self.coef[c] for c in self.columns}, index=df.index)

    def steady_state_response(self, ch: str, weekly_spend: float) -> float:
        p = self.params[ch]
        a = weekly_spend / (1 - p["decay"])
        return self.coef[ch] * a / (a + p["half_saturation"])

    def optimize_budget(self, current: dict[str, float], step: float = 250.0, flex: float = 0.5) -> dict[str, float]:
        """Greedy marginal-return allocation of the same total weekly budget,
        keeping each channel within +/- flex of its current level."""
        total = sum(current.values())
        alloc = {ch: current[ch] * (1 - flex) for ch in CHANNELS}
        cap = {ch: current[ch] * (1 + flex) for ch in CHANNELS}
        remaining = total - sum(alloc.values())
        while remaining > 1e-6:
            inc = min(step, remaining)
            gains = {ch: self.steady_state_response(ch, alloc[ch] + inc) - self.steady_state_response(ch, alloc[ch])
                     for ch in CHANNELS if alloc[ch] + inc <= cap[ch] + 1e-6}
            if not gains:
                break
            best = max(gains, key=gains.get)
            alloc[best] += inc
            remaining -= inc
        return alloc


def mape(y, p) -> float:
    return float(np.mean(np.abs((y - p) / y)))


def main():
    with db.connect(read_only=True) as con:
        df = db.read_df(con, "SELECT * FROM mart_mmm_weekly ORDER BY week_start").reset_index(drop=True)

    # Out-of-sample check on the final quarter, then refit on everything
    train = df.iloc[:-HOLDOUT_WEEKS]
    holdout_model = MMM().fit(train)
    full_pred_h = holdout_model.predict(df)
    y = df["conversions"].to_numpy(float)
    holdout_mape = mape(y[-HOLDOUT_WEEKS:], full_pred_h[-HOLDOUT_WEEKS:])

    model = MMM().fit(df)
    pred = model.predict(df)
    ss_res, ss_tot = np.sum((y - pred) ** 2), np.sum((y - y.mean()) ** 2)
    contrib = model.contributions(df)

    truth_path = config.RAW_DIR / "_mmm_truth.json"
    truth = json.loads(truth_path.read_text()) if truth_path.exists() else {}

    channels = []
    last_13 = df.tail(13)
    current = {ch: float(last_13[ch].mean()) for ch in CHANNELS}
    for ch in CHANNELS:
        spend = float(df[ch].sum())
        c = float(contrib[ch].sum())
        mroi = (model.steady_state_response(ch, current[ch] + 1000) - model.steady_state_response(ch, current[ch]))
        row = {
            "channel": ch,
            "total_spend": round(spend, 2),
            "contribution": round(c, 1),
            "share_of_paid_contribution": None,
            "roi_per_1k": round(c / spend * 1000, 2) if spend else None,
            "marginal_roi_per_1k": round(float(mroi), 2),
            "decay": model.params[ch]["decay"],
            "half_saturation": round(model.params[ch]["half_saturation"], 0),
        }
        if ch in truth:
            row["true_roi_per_1k"] = round(truth[ch]["roi_per_1k"], 2)
            row["true_decay"] = truth[ch]["decay"]
        channels.append(row)
    paid_total = sum(r["contribution"] for r in channels)
    for r in channels:
        r["share_of_paid_contribution"] = round(r["contribution"] / paid_total, 4)

    optimal = model.optimize_budget(current)
    cur_resp = sum(model.steady_state_response(ch, current[ch]) for ch in CHANNELS)
    opt_resp = sum(model.steady_state_response(ch, optimal[ch]) for ch in CHANNELS)

    results = {
        "fit": {"r2": round(float(1 - ss_res / ss_tot), 4), "holdout_mape": round(holdout_mape, 4),
                "holdout_weeks": HOLDOUT_WEEKS, "weeks": len(df)},
        "channels": channels,
        "budget": {
            "weekly_total": round(sum(current.values()), 2),
            "current": {k: round(v, 2) for k, v in current.items()},
            "recommended": {k: round(v, 2) for k, v in optimal.items()},
            "paid_conversions_per_week_current": round(cur_resp, 1),
            "paid_conversions_per_week_recommended": round(opt_resp, 1),
            "lift_pct": round((opt_resp / cur_resp - 1) * 100, 2),
        },
        "weekly": pd.DataFrame({
            "week_start": df["week_start"], "actual": y, "predicted": pred.round(1),
            "base": (contrib[[c for c in contrib.columns if c not in CHANNELS]].sum(axis=1)).round(1),
            **{ch: contrib[ch].round(1) for ch in CHANNELS},
        }).to_dict(orient="records"),
    }
    (config.ARTIFACTS_DIR / "mmm_results.json").write_text(json.dumps(results, indent=2))

    print(f"MMM | R^2 {results['fit']['r2']:.3f} | holdout MAPE {holdout_mape:.1%}")
    for r in channels:
        t = f" (true {r['true_roi_per_1k']:.1f})" if "true_roi_per_1k" in r else ""
        print(f"  {r['channel']:<22} ROI/$1k {r['roi_per_1k']:>6.1f}{t}   marginal {r['marginal_roi_per_1k']:>5.1f}   decay {r['decay']:.1f}")
    print(f"  budget reallocation (same spend): +{results['budget']['lift_pct']:.1f}% paid-driven conversions")


if __name__ == "__main__":
    main()
