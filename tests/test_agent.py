import unittest

from rag_agent.agent import RAGAgent
from rag_agent.evidence import EvidenceGate
from rag_agent.models import EvidencePassage, GeneratedAnswer


class FakeRetriever:
    def __init__(self, evidence):
        self.evidence = evidence
        self.calls = 0

    def retrieve(self, question, *, limit=6):
        self.calls += 1
        return self.evidence


class FakeGenerator:
    def __init__(self, citation_ids=(1,)):
        self.citation_ids = list(citation_ids)
        self.calls = 0

    def generate(self, question, evidence):
        self.calls += 1
        return GeneratedAnswer(answer="Verified answer.", citation_ids=self.citation_ids)


def evidence(source="database", score=0.9):
    return EvidencePassage(
        content="The official course withdrawal policy explains the withdrawal deadline.",
        title="Official policy",
        url="https://policy.utdallas.edu/example" if source == "web" else None,
        authority_tier=2,
        score=score,
        source=source,
    )


class AgentWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.gate = EvidenceGate(minimum_retrieval_score=0.6, minimum_direct_support=0.2)

    def test_local_success_does_not_call_web(self):
        local = FakeRetriever([evidence()])
        web = FakeRetriever([evidence("web")])
        generator = FakeGenerator()
        agent = RAGAgent(local_retriever=local, web_retriever=web, generator=generator, evidence_gate=self.gate)

        response = agent.answer("What is the course withdrawal policy deadline?")

        self.assertEqual(response.evidence_source, "database")
        self.assertEqual(web.calls, 0)
        self.assertEqual(len(response.citations), 1)

    def test_insufficient_local_falls_back_to_validated_web(self):
        local = FakeRetriever([evidence(score=0.1)])
        web = FakeRetriever([evidence("web")])
        agent = RAGAgent(
            local_retriever=local,
            web_retriever=web,
            generator=FakeGenerator(),
            evidence_gate=self.gate,
        )

        response = agent.answer("What is the course withdrawal policy deadline?")

        self.assertEqual(web.calls, 1)
        self.assertEqual(response.evidence_source, "web")

    def test_abstains_without_sufficient_evidence(self):
        agent = RAGAgent(
            local_retriever=FakeRetriever([]),
            web_retriever=FakeRetriever([]),
            generator=FakeGenerator(),
            evidence_gate=self.gate,
        )
        response = agent.answer("What is the course withdrawal policy deadline?")
        self.assertEqual(response.evidence_source, "unverified")
        self.assertTrue(response.needs_official_confirmation)

    def test_crisis_bypasses_all_retrieval(self):
        local = FakeRetriever([])
        generator = FakeGenerator()
        agent = RAGAgent(local_retriever=local, generator=generator, evidence_gate=self.gate)
        response = agent.answer("I may hurt myself")
        self.assertIn("988", response.answer)
        self.assertEqual(local.calls, 0)
        self.assertEqual(generator.calls, 0)

    def test_invalid_model_citation_causes_abstention(self):
        agent = RAGAgent(
            local_retriever=FakeRetriever([evidence()]),
            generator=FakeGenerator(citation_ids=(99,)),
            evidence_gate=self.gate,
        )
        response = agent.answer("What is the course withdrawal policy deadline?")
        self.assertEqual(response.evidence_source, "unverified")


if __name__ == "__main__":
    unittest.main()
