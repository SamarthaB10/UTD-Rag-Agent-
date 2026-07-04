from __future__ import annotations

import logging

from dotenv import load_dotenv

from rag_agent import AnswerResponse, RAGAgent, build_default_agent


load_dotenv()
logging.basicConfig(level=logging.INFO)

_agent: RAGAgent | None = None


def get_agent() -> RAGAgent:
    global _agent
    if _agent is None:
        _agent = build_default_agent()
    return _agent


def ingest_handbook_if_needed() -> None:
    agent = get_agent()
    ingest = getattr(agent.local_retriever, "ingest_if_needed", None)
    if ingest:
        ingest()


def answer_question(question: str) -> AnswerResponse:
    return get_agent().answer(question)


def run_cli() -> None:
    ingest_handbook_if_needed()
    while True:
        question = input("Enter a UTD question (-1 to quit): ").strip()
        if question == "-1":
            break
        print(answer_question(question).model_dump_json(indent=2))


if __name__ == "__main__":
    run_cli()
