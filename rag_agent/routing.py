from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .models import AnswerResponse, Citation, utc_now_iso


class Route(str, Enum):
    NORMAL = "normal"
    PERSONAL_RECORD = "personal_record"
    IMMEDIATE_DANGER = "immediate_danger"
    MENTAL_HEALTH = "mental_health"
    SEXUAL_MISCONDUCT = "sexual_misconduct"
    MEDICAL_EMERGENCY = "medical_emergency"


@dataclass(frozen=True)
class Classification:
    route: Route


_PATTERNS: tuple[tuple[Route, re.Pattern[str]], ...] = (
    (
        Route.IMMEDIATE_DANGER,
        re.compile(
            r"\b(active shooter|bomb threat|someone (?:has a )?weapon|in immediate danger|"
            r"being attacked|call (?:the )?police|campus police|emergency police)\b",
            re.I,
        ),
    ),
    (
        Route.MEDICAL_EMERGENCY,
        re.compile(
            r"\b(overdose|not breathing|unconscious|severe bleeding|medical emergency|"
            r"chest pain|anaphylaxis)\b",
            re.I,
        ),
    ),
    (
        Route.MENTAL_HEALTH,
        re.compile(
            r"\b(suicid(?:e|al)|kill myself|hurt myself|self[- ]harm|mental health crisis|"
            r"crisis hotline|can't go on)\b",
            re.I,
        ),
    ),
    (
        Route.SEXUAL_MISCONDUCT,
        re.compile(
            r"\b(sexual assault|sexual misconduct|sexually harassed|dating violence|"
            r"domestic violence|stalking|title ix)\b",
            re.I,
        ),
    ),
    (
        Route.PERSONAL_RECORD,
        re.compile(
            r"\b(my (?:hold|holds|balance|bill|grade|grades|gpa|application|admission status|"
            r"financial aid status|refund|transcript|transfer credit|transfer credits)|"
            r"why (?:was|is) my|check my|what do i owe)\b",
            re.I,
        ),
    ),
)


def classify_question(question: str) -> Classification:
    for route, pattern in _PATTERNS:
        if pattern.search(question):
            return Classification(route=route)
    return Classification(route=Route.NORMAL)


def _citation(title: str, url: str, tier: int = 4) -> Citation:
    return Citation(
        title=title,
        url=url,
        section="Contact information",
        retrieved_at=utc_now_iso(),
        authority_tier=tier,
    )


def routed_response(route: Route) -> AnswerResponse | None:
    """Return manually verified, retrieval-independent safety/personal routing."""
    if route in (Route.IMMEDIATE_DANGER, Route.MEDICAL_EMERGENCY):
        return AnswerResponse(
            answer=(
                "If there is immediate danger or a medical emergency, call 911 now. "
                "For a campus police response, call UT Dallas Police at 972-883-2222."
            ),
            evidence_source="database",
            citations=[
                _citation("UT Dallas Safety", "https://www.utdallas.edu/safety/")
            ],
            next_step="Call 911 now if anyone may be in immediate danger.",
        )
    if route == Route.MENTAL_HEALTH:
        return AnswerResponse(
            answer=(
                "If you may act on thoughts of suicide or self-harm, call 911 or 988 now, "
                "or go to the nearest emergency room. UT Dallas's 24/7 crisis line is "
                "972-UTD-TALK (972-883-8255)."
            ),
            evidence_source="database",
            citations=[
                _citation(
                    "UT Dallas Student Counseling Center",
                    "https://counseling.utdallas.edu/appointments/",
                )
            ],
            next_step="Call 988 or 972-883-8255 for immediate crisis support.",
        )
    if route == Route.SEXUAL_MISCONDUCT:
        return AnswerResponse(
            answer=(
                "You can contact the UT Dallas Title IX Coordinator at "
                "TitleIXCoordinator@utdallas.edu or 972-883-2306. If there is immediate "
                "danger, call 911; UT Dallas Police is 972-883-2222."
            ),
            evidence_source="database",
            citations=[
                _citation(
                    "UT Dallas Title IX Compliance",
                    "https://institutional-compliance.utdallas.edu/title-ix/",
                )
            ],
            next_step="Contact Title IX or emergency services, depending on urgency.",
            needs_official_confirmation=False,
        )
    if route == Route.PERSONAL_RECORD:
        return AnswerResponse(
            answer=(
                "I can explain general UT Dallas policy, but I cannot access or determine "
                "your personal holds, balance, grades, transfer-credit decision, financial "
                "aid status, or application status. Do not post sensitive student records here."
            ),
            evidence_source="unverified",
            next_step=(
                "Check Orion for your record, then contact the office shown there (such as "
                "the Registrar, Financial Aid, Bursar, or Admissions) if it needs review."
            ),
            needs_official_confirmation=True,
        )
    return None
