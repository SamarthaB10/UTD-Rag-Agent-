from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


EvidenceSource = Literal["database", "web", "unverified"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Citation(BaseModel):
    title: str
    url: str | None = None
    section: str | None = None
    retrieved_at: str
    authority_tier: int = Field(ge=1, le=5)


class AnswerResponse(BaseModel):
    answer: str
    evidence_source: EvidenceSource
    citations: list[Citation] = Field(default_factory=list)
    next_step: str | None = None
    needs_official_confirmation: bool = False


class EvidencePassage(BaseModel):
    content: str
    title: str
    url: str | None = None
    section: str | None = None
    retrieved_at: str = Field(default_factory=utc_now_iso)
    updated_at: str | None = None
    authority_tier: int = Field(ge=1, le=5)
    score: float = Field(ge=0.0, le=1.0)
    source: Literal["database", "web"]
    student_type: Literal["undergraduate", "graduate", "all"] | None = None
    catalog_year: str | None = None

    def citation(self) -> Citation:
        return Citation(
            title=self.title,
            url=self.url,
            section=self.section,
            retrieved_at=self.retrieved_at,
            authority_tier=self.authority_tier,
        )


class GeneratedAnswer(BaseModel):
    answer: str
    citation_ids: list[int] = Field(default_factory=list)
    next_step: str | None = None
    needs_official_confirmation: bool = False
