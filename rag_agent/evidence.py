from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from .models import EvidencePassage


_TOKEN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "from", "how",
    "i", "in", "is", "it", "my", "of", "on", "or", "the", "to", "what",
    "when", "where", "who", "why", "with", "ut", "utd", "utdallas",
}


def meaningful_tokens(text: str) -> set[str]:
    return {token for token in _TOKEN.findall(text.lower()) if token not in _STOP_WORDS}


def direct_support_score(question: str, passage: str) -> float:
    query_tokens = meaningful_tokens(question)
    if not query_tokens:
        return 0.0
    return len(query_tokens & meaningful_tokens(passage)) / len(query_tokens)


def infer_student_type(text: str) -> str | None:
    lowered = text.lower()
    if "undergraduate" in lowered or "bachelor" in lowered:
        return "undergraduate"
    if "graduate" in lowered or "master's" in lowered or "doctoral" in lowered:
        return "graduate"
    return None


def requested_catalog_year(question: str) -> str | None:
    before = re.search(r"\b(20\d{2})\b.{0,30}\bcatalog\b", question, re.I)
    if before:
        return before.group(1)
    after = re.search(r"\bcatalog\b.{0,30}\b(20\d{2})\b", question, re.I)
    return after.group(1) if after else None


def infer_catalog_year(text: str) -> str | None:
    match = re.search(r"(?:/|\b)(20\d{2})(?:/|\b)", text)
    return match.group(1) if match else None


@dataclass(frozen=True)
class GateDecision:
    sufficient: bool
    evidence: tuple[EvidencePassage, ...]
    reasons: tuple[str, ...]


class EvidenceGate:
    def __init__(
        self,
        *,
        minimum_retrieval_score: float = 0.58,
        minimum_direct_support: float = 0.18,
        max_passages: int = 6,
    ) -> None:
        self.minimum_retrieval_score = minimum_retrieval_score
        self.minimum_direct_support = minimum_direct_support
        self.max_passages = max_passages

    def evaluate(self, question: str, passages: list[EvidencePassage]) -> GateDecision:
        wanted_student_type = infer_student_type(question)
        wanted_catalog = requested_catalog_year(question)
        accepted: list[EvidencePassage] = []
        reasons: list[str] = []

        for passage in passages:
            support = direct_support_score(question, passage.content)
            if passage.score < self.minimum_retrieval_score:
                continue
            if support < self.minimum_direct_support:
                continue
            if (
                wanted_student_type
                and passage.student_type
                and passage.student_type not in ("all", wanted_student_type)
            ):
                continue
            if wanted_catalog and passage.catalog_year != wanted_catalog:
                continue
            accepted.append(passage)

        if not accepted:
            reasons.append("No passage met retrieval, direct-support, and applicability thresholds.")
            return GateDecision(False, (), tuple(reasons))

        # Authority is the primary ordering; relevance resolves sources in the same tier.
        accepted.sort(key=lambda item: (item.authority_tier, -item.score))
        accepted = accepted[: self.max_passages]

        if self._has_numeric_conflict(accepted):
            reasons.append("Potentially conflicting authoritative passages were found.")
            return GateDecision(False, tuple(accepted), tuple(reasons))

        return GateDecision(True, tuple(accepted), ())

    @staticmethod
    def _has_numeric_conflict(passages: list[EvidencePassage]) -> bool:
        """Flag contradictory numeric answers among similarly relevant same-tier sources."""
        if len(passages) < 2:
            return False
        conflict_terms = {
            "amount", "credits", "days", "deadline", "fee", "gpa", "hours",
            "maximum", "minimum", "percent", "semesters", "years",
        }
        for index, left in enumerate(passages):
            left_tokens = meaningful_tokens(left.content)
            left_numbers = set(re.findall(r"\b(?!20\d{2}\b)\d+(?:\.\d+)?%?\b", left.content))
            if not left_numbers or not (left_tokens & conflict_terms):
                continue
            for right in passages[index + 1 :]:
                if left.authority_tier != right.authority_tier:
                    continue
                right_tokens = meaningful_tokens(right.content)
                right_numbers = set(re.findall(r"\b(?!20\d{2}\b)\d+(?:\.\d+)?%?\b", right.content))
                if (
                    right_numbers
                    and left_numbers.isdisjoint(right_numbers)
                    and len(left_tokens & right_tokens) >= 4
                    and bool(right_tokens & conflict_terms)
                ):
                    return True
        return False


def freshness_penalty(updated_at: str | None, *, max_age_days: int = 1095) -> float:
    """A small deterministic penalty; missing dates stay visible to the gate."""
    if not updated_at:
        return 0.05
    try:
        value = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except ValueError:
        try:
            value = parsedate_to_datetime(updated_at)
        except (TypeError, ValueError):
            return 0.05
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - value).days
    return 0.12 if age > max_age_days else 0.0
