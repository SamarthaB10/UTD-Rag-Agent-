from __future__ import annotations

import os
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import EvidencePassage


def _canonical_policy_url(title: str) -> str | None:
    match = re.search(r"\b(UTD[A-Z]{2}\d{4})\b", title, re.I)
    return f"https://policy.utdallas.edu/{match.group(1).lower()}" if match else None


class LocalHandbookRetriever:
    def __init__(
        self,
        *,
        persist_directory: Path,
        handbook_path: Path | None = None,
        collection_name: str = "UTD_HANDBOOK",
    ) -> None:
        self.handbook_path = handbook_path
        embeddings = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        )
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_directory),
        )

    def ingest_if_needed(self) -> None:
        if self.vector_store.get(limit=1)["ids"]:
            return
        if not self.handbook_path or not self.handbook_path.is_file():
            raise FileNotFoundError(
                "The Chroma collection is empty. Set UTD_HANDBOOK_PATH to the handbook PDF."
            )
        pages = PyPDFLoader(str(self.handbook_path)).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=180)
        chunks = splitter.split_documents(pages)
        for chunk in chunks:
            chunk.metadata.update(
                {
                    "title": "UT Dallas Student Handbook",
                    "authority_tier": 2,
                    "student_type": "all",
                }
            )
        self.vector_store.add_documents(chunks)

    def retrieve(self, question: str, *, limit: int = 6) -> list[EvidencePassage]:
        results = self.vector_store.similarity_search_with_relevance_scores(question, k=limit)
        passages: list[EvidencePassage] = []
        for document, raw_score in results:
            metadata = document.metadata
            page = metadata.get("page_label")
            if page is None and metadata.get("page") is not None:
                page = int(metadata["page"]) + 1
            score = max(0.0, min(1.0, float(raw_score)))
            title = str(metadata.get("title", "UT Dallas Student Handbook"))
            source_url = metadata.get("url") or _canonical_policy_url(title)
            if source_url and not str(source_url).startswith("https://"):
                source_url = None
            passages.append(
                EvidencePassage(
                    content=document.page_content,
                    title=title,
                    url=source_url,
                    section=f"Page {page}" if page is not None else metadata.get("section"),
                    updated_at=metadata.get("updated_at"),
                    authority_tier=int(metadata.get("authority_tier", 2)),
                    score=score,
                    source="database",
                    student_type=metadata.get("student_type", "all"),
                    catalog_year=metadata.get("catalog_year"),
                )
            )
        return passages
