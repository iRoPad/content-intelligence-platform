"""Integration checks on the built warehouse. Run `python -m ciq run` first (CI does)."""
import json
import unittest

from ciq import config, db, transform

BUILT = db.db_path().exists()


@unittest.skipUnless(BUILT, "warehouse not built; run `python -m ciq run`")
class TestWarehouse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = db.connect(read_only=True)
        cls.con = cls.ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)

    def scalar(self, sql):
        return self.con.execute(sql).fetchone()[0]

    def test_dedup_removed_duplicate_raw_rows(self):
        raw = self.scalar("SELECT COUNT(*) FROM raw_posts")
        distinct = self.scalar("SELECT COUNT(DISTINCT post_id) FROM raw_posts")
        self.assertGreater(raw, distinct, "fixture should contain duplicates")
        self.assertEqual(self.scalar("SELECT COUNT(*) FROM stg_posts"), distinct)

    def test_platforms_normalized(self):
        rows = {r[0] for r in self.con.execute("SELECT DISTINCT platform FROM stg_posts").fetchall()}
        self.assertEqual(rows, {"tiktok", "instagram", "youtube_shorts", "youtube"})

    def test_breakout_label_has_no_lookahead(self):
        # the first post for each brand x platform can't have a baseline
        n = self.scalar("""SELECT COUNT(*) FROM (
            SELECT brand, platform, MIN(published_at_utc) AS first_ts FROM fct_post_performance GROUP BY 1, 2) f
            JOIN fct_post_performance p ON p.brand = f.brand AND p.platform = f.platform AND p.published_at_utc = f.first_ts
            WHERE p.baseline_views IS NOT NULL""")
        self.assertEqual(n, 0)

    def test_breakout_rate_is_plausible(self):
        rate = self.scalar("SELECT AVG(CAST(is_breakout AS DOUBLE)) FROM fct_post_performance WHERE is_breakout IS NOT NULL")
        self.assertTrue(0.02 < rate < 0.25, rate)

    def test_quality_artifact_has_no_failures(self):
        results = json.loads((config.ARTIFACTS_DIR / "data_quality.json").read_text())
        self.assertFalse([r for r in results if r["status"] == "fail"])

    def test_model_beats_random(self):
        m = json.loads((config.ARTIFACTS_DIR / "model_metrics.json").read_text())
        self.assertGreater(m["models"]["gradient_boosting"]["roc_auc"], 0.6)


class TestTemplating(unittest.TestCase):
    def test_render_substitutes_variables(self):
        self.assertIn(str(config.BREAKOUT_MULTIPLE), transform.render("x >= {{ breakout_multiple }}"))

    def test_render_rejects_unknown_variables(self):
        with self.assertRaises(KeyError):
            transform.render("{{ nope }}")


if __name__ == "__main__":
    unittest.main()
