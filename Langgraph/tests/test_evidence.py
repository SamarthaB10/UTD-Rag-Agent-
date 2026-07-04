import unittest

from rag_agent.evidence import (
    EvidenceGate,
    direct_support_score,
    freshness_penalty,
    requested_catalog_year,
)
from rag_agent.models import EvidencePassage


def passage(content: str, *, score: float = 0.9, tier: int = 2, **kwargs) -> EvidencePassage:
    return EvidencePassage(
        content=content,
        title="Policy",
        authority_tier=tier,
        score=score,
        source="database",
        **kwargs,
    )


class EvidenceGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = EvidenceGate(minimum_retrieval_score=0.6, minimum_direct_support=0.2)

    def test_direct_support_is_query_coverage(self):
        score = direct_support_score(
            "What is the course withdrawal deadline?",
            "The course withdrawal deadline appears in the academic calendar.",
        )
        self.assertGreaterEqual(score, 0.75)

    def test_accepts_supported_high_scoring_passage(self):
        decision = self.gate.evaluate(
            "What is the course withdrawal deadline?",
            [passage("The course withdrawal deadline appears in the academic calendar.")],
        )
        self.assertTrue(decision.sufficient)

    def test_rejects_low_score_or_wrong_student_type(self):
        question = "What is the graduate course withdrawal deadline?"
        decision = self.gate.evaluate(
            question,
            [
                passage("The graduate course withdrawal deadline is listed here.", score=0.2),
                passage(
                    "The undergraduate course withdrawal deadline is listed here.",
                    student_type="undergraduate",
                ),
            ],
        )
        self.assertFalse(decision.sufficient)

    def test_requires_requested_catalog_year(self):
        decision = self.gate.evaluate(
            "Under the 2025 catalog, what are degree credit requirements?",
            [passage("The degree credit requirements are described here.", catalog_year="2024")],
        )
        self.assertFalse(decision.sufficient)

    def test_finds_year_in_qualified_catalog_question(self):
        self.assertEqual(
            requested_catalog_year("What does the 2025 undergraduate catalog require?"),
            "2025",
        )

    def test_outdated_sources_receive_a_freshness_penalty(self):
        self.assertEqual(freshness_penalty("Mon, 01 Jan 2018 00:00:00 GMT"), 0.12)

    def test_flags_close_same_tier_numeric_conflict(self):
        decision = self.gate.evaluate(
            "How many days are allowed before the appeal deadline?",
            [
                passage("The policy allows 10 days before the appeal deadline is final."),
                passage("The policy allows 20 days before the appeal deadline is final."),
            ],
        )
        self.assertFalse(decision.sufficient)
        self.assertIn("conflicting", decision.reasons[0].lower())


if __name__ == "__main__":
    unittest.main()
