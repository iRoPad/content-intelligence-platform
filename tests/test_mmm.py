import unittest

import numpy as np
import pandas as pd

from ciq.mmm import CHANNELS, MMM, adstock, hill


class TestTransforms(unittest.TestCase):
    def test_adstock_carryover(self):
        np.testing.assert_allclose(adstock(np.array([100.0, 0, 0]), 0.5), [100, 50, 25])

    def test_adstock_no_decay_is_identity(self):
        x = np.array([3.0, 1, 4, 1, 5])
        np.testing.assert_allclose(adstock(x, 0.0), x)

    def test_hill_saturates(self):
        self.assertAlmostEqual(hill(np.array([10.0]), 10.0)[0], 0.5)
        self.assertLess(hill(np.array([1e9]), 10.0)[0], 1.0)


class TestRecovery(unittest.TestCase):
    def test_recovers_strongest_channel(self):
        rng = np.random.default_rng(0)
        n = 104
        df = pd.DataFrame({ch: rng.uniform(2000, 20000, n) * (rng.random(n) > 0.2) for ch in CHANNELS})
        y = np.full(n, 1000.0)
        betas = {"tiktok_ads": 300, "meta_ads": 300, "youtube_ads": 300, "creator_partnerships": 1500, "paid_search": 100}
        for ch in CHANNELS:
            y += betas[ch] * hill(adstock(df[ch].to_numpy(), 0.4), 15000)
        df["conversions"] = y + rng.normal(0, 10, n)
        df["organic_views"] = rng.uniform(1e6, 2e6, n)
        model = MMM().fit(df, passes=2)
        contrib = model.contributions(df)[CHANNELS].sum()
        self.assertEqual(contrib.idxmax(), "creator_partnerships")
        pred = model.predict(df)
        self.assertLess(np.mean(np.abs(pred - df["conversions"]) / df["conversions"]), 0.02)

    def test_budget_optimizer_conserves_spend(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({ch: rng.uniform(1000, 9000, 60) for ch in CHANNELS})
        df["conversions"] = 500 + 0.02 * df["creator_partnerships"] + rng.normal(0, 5, 60)
        df["organic_views"] = 1.0
        model = MMM().fit(df, passes=1)
        current = {ch: 5000.0 for ch in CHANNELS}
        alloc = model.optimize_budget(current)
        self.assertAlmostEqual(sum(alloc.values()), sum(current.values()), places=4)
        for ch in CHANNELS:
            self.assertGreaterEqual(alloc[ch], 2500 - 1e-6)
            self.assertLessEqual(alloc[ch], 7500 + 1e-6)


if __name__ == "__main__":
    unittest.main()
