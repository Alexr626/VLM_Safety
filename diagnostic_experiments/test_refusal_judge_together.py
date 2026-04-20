"""Unit tests for verdict parsing (no Together API calls)."""

import unittest

from refusal_judge_together import verdict_to_refused


class VerdictTests(unittest.TestCase):
    def test_bare_yes_no(self):
        self.assertTrue(verdict_to_refused("YES", None))
        self.assertFalse(verdict_to_refused("NO", None))

    def test_short_answer_wrapped(self):
        self.assertFalse(verdict_to_refused("**Short answer:**\n\nNO", None))
        self.assertTrue(verdict_to_refused("**Short answer:**\n\nYES", None))

    def test_last_token_wins(self):
        self.assertTrue(verdict_to_refused("Maybe NO then YES", None))

    def test_reasoning_fallback_when_content_empty(self):
        self.assertTrue(verdict_to_refused("", "The answer is YES"))

    def test_ambiguous_empty(self):
        self.assertFalse(verdict_to_refused("", ""))


if __name__ == "__main__":
    unittest.main()
