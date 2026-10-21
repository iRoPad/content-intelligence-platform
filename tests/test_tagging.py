import unittest

from ciq import llm
from ciq.tagging import TAXONOMY, _validate, rules_tag


class TestRulesTagger(unittest.TestCase):
    def test_keyword_captions(self):
        self.assertEqual(rules_tag("Top 5: how to master the fadeaway, step by step 📊"),
                         {"theme": "tutorial", "hook": "countdown", "tone": "informative"})
        self.assertEqual(rules_tag("Hot take: pro reacts to a nutmeg 😂")["hook"], "hot_take")
        self.assertEqual(rules_tag("Trivia: guess who pulled off a rabona")["theme"], "trivia")

    def test_always_returns_valid_labels(self):
        for cap in ("", "???", "wow", "nobody in the building was ready for this"):
            self.assertIsNotNone(_validate(rules_tag(cap)))


class TestLlmContract(unittest.TestCase):
    def test_validate_rejects_out_of_taxonomy(self):
        self.assertIsNone(_validate({"theme": "vlog", "hook": "none", "tone": "hype"}))
        self.assertIsNone(_validate({"theme": "trivia"}))
        ok = {f: v[0] for f, v in TAXONOMY.items()}
        self.assertEqual(_validate({**ok, "extra": 1}), ok)

    def test_extract_json_handles_fences_and_chatter(self):
        self.assertEqual(llm.extract_json('Sure!\n```json\n[{"id": "a"}]\n```'), [{"id": "a"}])
        self.assertEqual(llm.extract_json('Result: {"sql": "SELECT 1"} hope that helps'), {"sql": "SELECT 1"})
        with self.assertRaises(ValueError):
            llm.extract_json("no json here")


if __name__ == "__main__":
    unittest.main()
