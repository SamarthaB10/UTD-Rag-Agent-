from __future__ import annotations

import io
import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Protocol

from pypdf import PdfReader

from .evidence import (
    direct_support_score,
    freshness_penalty,
    infer_catalog_year,
    infer_student_type,
)
from .models import EvidencePassage
from .source_policy import (
    ALLOWED_DOMAIN_FAMILIES,
    SourcePolicyError,
    authority_tier,
    validate_source_url,
)


SUPPORTED_CONTENT_TYPES = ("text/html", "text/plain", "application/pdf")
MAX_RESPONSE_BYTES = 5_000_000


@dataclass(frozen=True)
class SearchResult:
    url: str


@dataclass(frozen=True)
class FetchedPage:
    title: str
    url: str
    text: str
    updated_at: str | None
    content_type: str


class SearchProvider(Protocol):
    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]: ...


class GoogleCustomSearch:
    """Google CSE is used only for discovery; snippets are intentionally discarded."""

    endpoint = "https://www.googleapis.com/customsearch/v1"

    def __init__(self, api_key: str, search_engine_id: str, *, timeout: float = 10.0) -> None:
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.timeout = timeout

    @classmethod
    def from_environment(cls) -> GoogleCustomSearch | None:
        api_key = os.getenv("GOOGLE_CSE_API_KEY")
        search_engine_id = os.getenv("GOOGLE_CSE_ID")
        if not api_key or not search_engine_id:
            return None
        return cls(api_key, search_engine_id)

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        site_filter = " OR ".join(f"site:{domain}" for domain in ALLOWED_DOMAIN_FAMILIES)
        params = urllib.parse.urlencode(
            {
                "key": self.api_key,
                "cx": self.search_engine_id,
                "q": f"{query} ({site_filter})",
                "num": min(max(limit, 1), 10),
                "safe": "active",
            }
        )
        request = urllib.request.Request(
            f"{self.endpoint}?{params}",
            headers={"Accept": "application/json", "User-Agent": "UTD-RAG/1.0"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read(MAX_RESPONSE_BYTES).decode("utf-8"))

        results: list[SearchResult] = []
        for item in payload.get("items", []):
            url = item.get("link", "")
            try:
                validate_source_url(url)
            except SourcePolicyError:
                continue
            results.append(SearchResult(url=url))
        return results


class TavilyRestrictedSearch:
    """Tavily discovery constrained to official domains; returned text is discarded."""

    def __init__(self, tool=None) -> None:
        if tool is None:
            from langchain_tavily import TavilySearch

            tool = TavilySearch(
                max_results=10,
                topic="general",
                search_depth="basic",
                include_answer=False,
                include_raw_content=False,
                include_images=False,
                include_domains=list(ALLOWED_DOMAIN_FAMILIES),
            )
        self.tool = tool

    @classmethod
    def from_environment(cls) -> TavilyRestrictedSearch | None:
        if not os.getenv("TAVILY_API_KEY"):
            return None
        return cls()

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        payload = self.tool.invoke({"query": query})
        if not isinstance(payload, dict) or payload.get("error"):
            return []
        results: list[SearchResult] = []
        for item in payload.get("results", []):
            url = item.get("url", "")
            try:
                validate_source_url(url)
            except SourcePolicyError:
                continue
            # Search snippets and Tavily-provided raw content are never retained.
            results.append(SearchResult(url=url))
            if len(results) >= limit:
                break
        return results


class _RestrictedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        validate_source_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _MainTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.all_parts: list[str] = []
        self.main_parts: list[str] = []
        self.updated_at: str | None = None
        self._ignored_depth = 0
        self._main_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key.lower(): value for key, value in attrs}
        if tag in {"script", "style", "noscript", "svg", "form", "nav", "footer"}:
            self._ignored_depth += 1
        if tag in {"main", "article"}:
            self._main_depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            name = (attributes.get("name") or attributes.get("property") or "").lower()
            if name in {
                "article:modified_time",
                "date",
                "dc.date",
                "last-modified",
                "last_modified",
            }:
                self.updated_at = attributes.get("content") or self.updated_at

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg", "form", "nav", "footer"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
        if tag in {"main", "article"}:
            self._main_depth = max(0, self._main_depth - 1)
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text or self._ignored_depth:
            return
        if self._in_title:
            self.title_parts.append(text)
        self.all_parts.append(text)
        if self._main_depth:
            self.main_parts.append(text)

    def extracted_text(self) -> str:
        parts = self.main_parts if len(" ".join(self.main_parts)) >= 200 else self.all_parts
        return "\n".join(parts)


class TrustedPageFetcher:
    def __init__(self, *, timeout: float = 12.0, max_bytes: int = MAX_RESPONSE_BYTES) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.opener = urllib.request.build_opener(_RestrictedRedirectHandler())

    def fetch(self, url: str) -> FetchedPage:
        validate_source_url(url)
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "text/html,text/plain,application/pdf",
                "Accept-Encoding": "identity",
                "User-Agent": "UTD-RAG/1.0 (+controlled academic policy retrieval)",
            },
        )
        with self.opener.open(request, timeout=self.timeout) as response:
            final_url = response.geturl()
            validate_source_url(final_url)
            content_type = response.headers.get_content_type().lower()
            if content_type not in SUPPORTED_CONTENT_TYPES:
                raise SourcePolicyError(f"Unsupported content type: {content_type}")
            body = response.read(self.max_bytes + 1)
            if len(body) > self.max_bytes:
                raise SourcePolicyError("Source page exceeds the configured size limit")
            header_updated = response.headers.get("Last-Modified")

        if content_type == "application/pdf":
            reader = PdfReader(io.BytesIO(body))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            title = str(reader.metadata.title) if reader.metadata and reader.metadata.title else "Official PDF"
            updated_at = header_updated
        elif content_type == "text/plain":
            text = body.decode("utf-8", errors="replace")
            title = final_url
            updated_at = header_updated
        else:
            charset = response.headers.get_content_charset() or "utf-8"
            parser = _MainTextParser()
            parser.feed(body.decode(charset, errors="replace"))
            text = parser.extracted_text()
            title = " ".join(parser.title_parts).strip() or final_url
            updated_at = parser.updated_at or header_updated

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) < 200:
            raise SourcePolicyError("Source page has no meaningful extractable text")
        return FetchedPage(title, final_url, text, updated_at, content_type)


def _passages(text: str, *, size: int = 1800, overlap: int = 250) -> list[str]:
    paragraphs: list[str] = []
    for raw_part in text.splitlines():
        part = raw_part.strip()
        while len(part) > size:
            split_at = part.rfind(" ", 0, size)
            split_at = split_at if split_at > size // 2 else size
            paragraphs.append(part[:split_at].strip())
            part = part[max(0, split_at - overlap) :].strip()
        if part:
            paragraphs.append(part)
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 1 > size:
            chunks.append(current)
            current = current[-overlap:] + "\n" + paragraph
        else:
            current = f"{current}\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks


class TrustedWebRetriever:
    def __init__(self, search_provider: SearchProvider, fetcher: TrustedPageFetcher | None = None) -> None:
        self.search_provider = search_provider
        self.fetcher = fetcher or TrustedPageFetcher()

    def retrieve(self, question: str, *, limit: int = 5) -> list[EvidencePassage]:
        evidence: list[EvidencePassage] = []
        seen_urls: set[str] = set()
        for result in self.search_provider.search(question, limit=limit):
            if result.url in seen_urls:
                continue
            seen_urls.add(result.url)
            try:
                page = self.fetcher.fetch(result.url)
            except Exception:
                # A malformed, inaccessible, redirected, or unsupported page is
                # rejected independently so other official results can proceed.
                continue
            ranked_chunks = sorted(
                _passages(page.text),
                key=lambda chunk: direct_support_score(question, chunk),
                reverse=True,
            )[:2]
            for index, chunk in enumerate(ranked_chunks, start=1):
                lexical = direct_support_score(question, chunk)
                score = max(0.0, min(1.0, 0.45 + 0.55 * lexical - freshness_penalty(page.updated_at)))
                evidence.append(
                    EvidencePassage(
                        content=chunk,
                        title=page.title,
                        url=page.url,
                        section=f"Relevant passage {index}",
                        updated_at=page.updated_at,
                        authority_tier=authority_tier(page.url),
                        score=score,
                        source="web",
                        student_type=infer_student_type(f"{page.url} {page.title} {chunk}"),
                        catalog_year=infer_catalog_year(f"{page.url} {page.title}"),
                    )
                )
        evidence.sort(key=lambda item: (item.authority_tier, -item.score))
        return evidence
