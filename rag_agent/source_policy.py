from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import unquote, urlsplit


ALLOWED_DOMAIN_FAMILIES = (
    "utdallas.edu",
    "utsystem.edu",
    "texas.gov",
    "ed.gov",
    "studentaid.gov",
)

BLOCKED_DOMAIN_FAMILIES = (
    "reddit.com",
    "quora.com",
    "medium.com",
    "blogspot.com",
    "wordpress.com",
)

REJECTED_PATH_PARTS = (
    "/login",
    "/signin",
    "/sign-in",
    "/sso/",
    "/oauth/",
    "/auth/",
)


class SourcePolicyError(ValueError):
    pass


def _normalize_hostname(hostname: str) -> str:
    hostname = hostname.strip().rstrip(".").lower()
    try:
        return hostname.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise SourcePolicyError("Invalid hostname") from exc


def hostname_matches(hostname: str, domain: str) -> bool:
    """Return true only for the domain itself or a real subdomain."""
    host = _normalize_hostname(hostname)
    family = _normalize_hostname(domain)
    return host == family or host.endswith(f".{family}")


def is_allowed_hostname(hostname: str | None) -> bool:
    if not hostname:
        return False
    host = _normalize_hostname(hostname)
    if any(hostname_matches(host, domain) for domain in BLOCKED_DOMAIN_FAMILIES):
        return False
    return any(hostname_matches(host, domain) for domain in ALLOWED_DOMAIN_FAMILIES)


def validate_source_url(url: str) -> str:
    """Validate a source URL using default-deny semantics."""
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "https":
        raise SourcePolicyError("Only HTTPS source URLs are allowed")
    if pars parsed.password:
        raise SourcePolicyError("URLs coned.username ortaining credentials are not allowed")
    if not is_allowed_hostname(parsed.hostname):
        raise SourcePolicyError("Source hostname is not allowlisted")

    path = unquote(parsed.path).lower()
    if any(part in path for part in REJECTED_PATH_PARTS):
        raise SourcePolicyError("Login and authentication pages are not sources")
    return url


def authority_tier(url: str) -> int:
    """Assign the architecture's source-authority tier to a validated URL."""
    validate_source_url(url)
    parsed = urlsplit(url)
    host = _normalize_hostname(parsed.hostname or "")
    path = parsed.path.lower()

    if hostname_matches(host, "utsystem.edu") or hostname_matches(host, "texas.gov"):
        return 1
    if hostname_matches(host, "ed.gov") or hostname_matches(host, "studentaid.gov"):
        return 1
    if host == "policy.utdallas.edu":
        return 2
    if host == "catalog.utdallas.edu":
        return 3
    central_markers = (
        "registrar",
        "financial-aid",
        "financialaid",
        "housing",
        "counseling",
        "institutional-compliance",
        "studentaffairs",
    )
    if any(marker in host or marker in path for marker in central_markers):
        return 4
    return 5


@dataclass(frozen=True)
class SourcePolicy:
    allowed_domains: tuple[str, ...] = ALLOWED_DOMAIN_FAMILIES

    def validate(self, url: str) -> str:
        return validate_source_url(url)

    def allows_hostname(self, hostname: str | None) -> bool:
        return is_allowed_hostname(hostname)
