"""The AI analyst must never be able to run anything but a bounded read-only query."""
import unittest

from ciq.analyst import UnsafeQuery, validate_sql


class TestSqlGuardrails(unittest.TestCase):
    def test_plain_select_gets_row_limit(self):
        out = validate_sql("SELECT brand, SUM(views) FROM fct_post_performance GROUP BY brand")
        self.assertTrue(out.endswith("LIMIT 200"))

    def test_existing_limit_kept(self):
        sql = "SELECT * FROM mart_brand_weekly LIMIT 5"
        self.assertEqual(validate_sql(sql), sql)

    def test_ctes_are_allowed(self):
        validate_sql("WITH x AS (SELECT * FROM fct_post_performance) SELECT COUNT(*) FROM x")

    def test_rejects_writes(self):
        for sql in ("DELETE FROM fct_post_performance",
                    "SELECT 1; DROP TABLE fct_post_performance",
                    "UPDATE mart_brand_weekly SET views = 0",
                    "SELECT * FROM fct_post_performance WHERE 1=1; ATTACH 'x.db' AS x"):
            with self.subTest(sql=sql), self.assertRaises(UnsafeQuery):
                validate_sql(sql)

    def test_rejects_unknown_tables(self):
        with self.assertRaises(UnsafeQuery):
            validate_sql("SELECT * FROM raw_posts")
        with self.assertRaises(UnsafeQuery):
            validate_sql("SELECT name FROM sqlite_master")

    def test_rejects_non_select(self):
        with self.assertRaises(UnsafeQuery):
            validate_sql("PRAGMA table_info(fct_post_performance)")


if __name__ == "__main__":
    unittest.main()
