from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol

from .evidence import EvidenceGate
from .generator import AnswerGenerator, OpenAIAnswerGenerator
from .local_retrieval import LocalHandbookRetriever
from .models import AnswerResponse, EvidencePassage, GeneratedAnswer
from .routing import classify_question, routed_response
from .web_search import GoogleCustomSearch, TavilyRestrictedSearch, TrustedWebRetriever


logger = logging.getLogger(__name__)


class Retriever(Protocol):
    def retrieve(self, question: str, *, limit: int = 6) -> list[EvidencePassage]: ...


class RAGAgent:
    def __init__(
        self,
        *,
        local_retriever: Retriever,
        generator: AnswerGenerator,
        web_retriever: Retriever | None = None,
        evidence_gate: EvidenceGate | None = None,
    ) -> None:
        self.local_retriever = local_retriever
        self.web_retriever = web_retriever
        self.generator = generator
        self.evidence_gate = evidence_gate or EvidenceGate()

    def answer(self, question: str) -> AnswerResponse:
        question = question.strip()
        if not question:
            return AnswerResponse(
                answer="Please enter a UT Dallas policy or student-services question.",
                evidence_source="unverified",
            )

        classification = classify_question(question)
        routed = routed_response(classification.route)
        if routed:
            return routed

        local_evidence = self._retrieve(self.local_retriever, question, limit=6)
        local_decision = self.evidence_gate.evaluate(question, local_evidence)
        if local_decision.sufficient:
            return self._generate(question, list(local_decision.evidence))

        web_evidence: list[EvidencePassage] = []
        if self.web_retriever is not None:
            web_evidence = self._retrieve(self.web_retriever, question, limit=5)

        combined_decision = self.evidence_gate.evaluate(
            question, local_evidence + web_evidence
        )
        if combined_decision.sufficient:
            return self._generate(question, list(combined_decision.evidence))

        conflict = any("conflict" in reason.lower() for reason in combined_decision.reasons)
        detail = (
            "I found potentially conflicting official information and could not safely choose between it."
            if conflict
            else "I could not verify the answer from the local knowledge base or trusted official websites."
        )
        return AnswerResponse(
            answer=detail,
            evidence_source="unverified",
            next_step=self._official_next_step(question),
            needs_official_confirmation=True,
        )

    @staticmethod
    def _retrieve(retriever: Retriever, question: str, *, limit: int) -> list[EvidencePassage]:
        try:
            return retriever.retrieve(question, limit=limit)
        except Exception as exc:
            logger.warning("Evidence retrieval failed: %s", exc)
            logger.debug("Retrieval traceback", exc_info=True)
            return []

    def _generate(self, question: str, evidence: list[EvidencePassage]) -> AnswerResponse:
        try:
            draft = self.generator.generate(question, evidence)
        except Exception as exc:
            logger.warning("Answer generation failed: %s", exc)
            logger.debug("Generation traceback", exc_info=True)
            return AnswerResponse(
                answer="I retrieved evidence but could not generate a verified answer.",
                evidence_source="unverified",
                next_step=self._official_next_step(question),
                needs_official_confirmation=True,
            )

        cited_indexes = self._valid_citation_indexes(draft, len(evidence))
        if not cited_indexes:
            return AnswerResponse(
                answer="I could not produce an answer with direct support from the retrieved evidence.",
                evidence_source="unverified",
                next_step=self._official_next_step(question),
                needs_official_confirmation=True,
            )
        cited_evidence = [evidence[index - 1] for index in cited_indexes]
        evidence_source = "web" if any(item.source == "web" for item in cited_evidence) else "database"
        return AnswerResponse(
            answer=draft.answer,
            evidence_source=evidence_source,
            citations=[item.citation() for item in cited_evidence],
            next_step=draft.next_step,
            needs_official_confirmation=draft.needs_official_confirmation,
        )

    @staticmethod
    def _valid_citation_indexes(draft: GeneratedAnswer, evidence_count: int) -> list[int]:
        result: list[int] = []
        for index in draft.citation_ids:
            if 1 <= index <= evidence_count and index not in result:
                result.append(index)
        return result

    @staticmethod
    def _official_next_step(question: str) -> str:
        lowered = question.lower()
        if any(word in lowered for word in ("financial aid", "fafsa", "scholarship")):
            return "Confirm with the UT Dallas Office of Financial Aid through its official contact page."
        if any(word in lowered for word in ("register", "transcript", "enrollment", "credit")):
            return "Confirm with the UT Dallas Registrar through its official contact page."
        if any(word in lowered for word in ("admission", "application", "applicant")):
            return "Confirm with UT Dallas Admissions through its official contact page."
        if any(word in lowered for word in ("housing", "dorm", "residence")):
            return "Confirm with University Housing through its official contact page."
        return "Contact the responsible UT Dallas office or the Comet Support Hub for an official answer."


def build_default_agent(base_dir: Path | None = None) -> RAGAgent:
    base_dir = base_dir or Path(__file__).resolve().parent.parent
    configured_path = os.getenv("UTD_HANDBOOK_PATH")
    handbook_path = Path(configured_path).expanduser() if configured_path else Path.home() / "Downloads/StudentHandbook.pdf"
    local = LocalHandbookRetriever(
        persist_directory=base_dir / "chroma_langchain_db",
        handbook_path=handbook_path,
    )

    search = GoogleCustomSearch.from_environment() or TavilyRestrictedSearch.from_environment()
    web = TrustedWebRetriever(search) if search else None
    return RAGAgent(
        local_retriever=local,
        web_retriever=web,
        generator=OpenAIAnswerGenerator(),
    )
