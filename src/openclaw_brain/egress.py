"""Mechanical, fail-closed egress policy enforcement.

``local-only`` deliberately means *this machine*, not "trusted network": only localhost,
127.0.0.0/8, and ::1 over HTTP(S) are accepted.  LAN addresses and ``*.local`` names are
reserved for a possible future ``lan`` policy and are not treated as local here.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
import sys
from dataclasses import dataclass
from urllib.parse import urlsplit

from openclaw_brain.config import (
    EGRESS_ENV_VAR,
    BrainConfig,
    ConfigError,
    ModelEntry,
    normalize_egress,
)

logger = logging.getLogger(__name__)

LOCAL_ONLY = "local-only"
DEFAULT_LOCAL_ENDPOINT = "http://localhost:8000/v1"

# These are enable switches, not credentials or mere project/endpoint selectors.  Values such
# as "false" and "0" are explicitly off and do not make a local-only start fail.
TRACING_ENV_VARS = (
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
)
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_PROXY_ENV_VARS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)


class EgressPolicyError(RuntimeError):
    """A runtime URL was blocked by the active egress policy."""


@dataclass(frozen=True)
class Violation:
    """One configured destination that would violate ``local-only``."""

    reference: str
    model_name: str
    provider: str
    endpoint: str
    host: str
    reason: str

    def __str__(self) -> str:
        endpoint = (self.endpoint if self.endpoint in {"<missing>", "offline-cache"}
                    else _safe_url(self.endpoint))
        return (
            f"{self.reference}: model={_safe_identifier(self.model_name)!r}, provider={self.provider!r}, "
            f"host={self.host!r}, endpoint={endpoint!r} — {self.reason}"
        )


def effective_egress(config: BrainConfig | None = None) -> str:
    """Return the effective policy, with ``OPENCLAW_EGRESS`` taking precedence."""
    if EGRESS_ENV_VAR in os.environ:
        return normalize_egress(os.environ[EGRESS_ENV_VAR], f"${EGRESS_ENV_VAR}")
    if config is None:
        return "any"
    return normalize_egress(getattr(getattr(config, "deployment", None), "egress", "any"))


def _url_host(url: str) -> str:
    try:
        return urlsplit(url).hostname or "<missing>"
    except ValueError:
        return "<invalid>"


def _safe_url(url: str) -> str:
    """Log only a URL's scheme, host and port, never credentials, path or query."""
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if not parsed.scheme or not host:
            return "<invalid>"
        port = parsed.port
        display_host = f"[{host}]" if ":" in host else host
        return f"{parsed.scheme}://{display_host}{f':{port}' if port is not None else ''}"
    except (ValueError, TypeError):
        return "<invalid>"


def _safe_identifier(value: str) -> str:
    return _safe_url(value) if "://" in value else value


def _loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        try:
            addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
            return bool(addresses) and all(
                ipaddress.ip_address(item[4][0]).is_loopback for item in addresses
            )
        except (OSError, ValueError, IndexError):
            return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv4Address):
        return address in ipaddress.ip_network("127.0.0.0/8")
    return address == ipaddress.ip_address("::1")


def is_local_url(url: str) -> bool:
    """True only for loopback HTTP(S) URLs on this machine.

    Host classification uses the parsed hostname, so user-info tricks such as
    ``http://127.0.0.1@evil.com`` and suffix tricks such as
    ``http://localhost.evil.com`` are rejected.  LAN IPs and mDNS ``*.local`` names are
    intentionally non-local under this policy; a future ``lan`` tier can define them.
    """
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            return False
        if parsed.username is not None or parsed.password is not None:
            return False
        host = parsed.hostname
        # Touch port too: malformed brackets/ports must fail classification, not be accepted
        # because the hostname happened to parse before the invalid suffix.
        _ = parsed.port
    except (ValueError, TypeError):
        return False
    if not host:
        return False
    return _loopback_host(host)


def check_url(url: str, policy: str | None = None) -> None:
    """Raise before a direct HTTP request when ``url`` violates ``local-only``."""
    active = effective_egress() if policy is None else normalize_egress(policy, "egress policy")
    if active == LOCAL_ONLY and not is_local_url(url):
        raise EgressPolicyError(
            f"Egress blocked: endpoint={_safe_url(url)!r}, host={_url_host(url)!r}, policy={active!r}; "
            "local-only permits only localhost, 127.0.0.0/8, and ::1 over HTTP(S)"
        )


def check_neo4j_uri(uri: str, policy: str | None = None) -> None:
    """Reject routing and non-loopback Bolt before constructing a driver."""
    active = effective_egress() if policy is None else normalize_egress(policy)
    if active != LOCAL_ONLY:
        return
    try:
        parsed = urlsplit(uri)
        host = parsed.hostname or "<missing>"
        _ = parsed.port
        valid = (parsed.scheme.lower() in {"bolt", "bolt+s", "bolt+ssc"}
                 and parsed.username is None and parsed.password is None
                 and _loopback_host(host))
    except (ValueError, TypeError):
        host, valid = "<invalid>", False
    if not valid:
        raise EgressPolicyError(
            f"Neo4j egress blocked: endpoint={_safe_url(uri)!r}, host={host!r}; "
            "local-only requires direct loopback Bolt"
        )


_PROVIDER_ENDPOINTS = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com",
    "xai": "https://api.x.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


def endpoint_for_model(entry: ModelEntry) -> str:
    if entry.provider == "local":
        return entry.endpoint or DEFAULT_LOCAL_ENDPOINT
    return entry.endpoint or _PROVIDER_ENDPOINTS.get(entry.provider, "<provider-managed>")


def violation_for_model(
    entry: ModelEntry,
    *,
    reference: str,
    policy: str,
) -> Violation | None:
    """Return the local-only violation for one resolved catalog entry, if any."""
    active = normalize_egress(policy, "egress policy")
    if active != LOCAL_ONLY:
        return None
    endpoint = endpoint_for_model(entry)
    safe_endpoint = _safe_url(endpoint)
    host = _url_host(endpoint)
    if entry.provider != "local":
        return Violation(
            reference, entry.name, entry.provider, safe_endpoint, host,
            "provider is not 'local'",
        )
    if not is_local_url(endpoint):
        return Violation(
            reference, entry.name, entry.provider, safe_endpoint, host,
            "local provider endpoint is not loopback HTTP(S)",
        )
    return None


def _configured_model_references(config: BrainConfig):
    for field_name in (
        "default_extraction",
        "default_reasoning",
        "default_matching",
        "default_vision",
        "default_figure_analysis",
    ):
        name = getattr(config.models, field_name)
        if name:
            yield f"[models].{field_name}", name
    for field_name in ("slide_analysis_model", "slide_analysis_fallback"):
        name = getattr(config.figures, field_name)
        if name:
            yield f"[figures].{field_name}", name
    for field_name in (
        "fallback_extraction",
        "fallback_reasoning",
        "fallback_matching",
    ):
        for index, name in enumerate(getattr(config.resilience, field_name)):
            if name:
                yield f"[resilience].{field_name}[{index}]", name


def validate_egress(
    config: BrainConfig,
    *,
    policy: str | None = None,
) -> list[Violation]:
    """Validate every configured LLM/VLM chain reference for the effective policy.

    Extraction, reasoning (also used by answer/summarize), matching, both PDF vision slots,
    both slide-analysis slots, and every resilience fallback are enumerated.  Embedding and
    MinerU/transformers model artifacts are not endpoint-routed catalog entries; local-only
    forces their loaders offline in :func:`enforce_startup_egress` instead.
    """
    active = effective_egress(config) if policy is None else normalize_egress(policy)
    if active != LOCAL_ONLY:
        return []

    violations: list[Violation] = []
    for reference, name in _configured_model_references(config):
        entry = config.models.get_model(name)
        if entry is None:
            violations.append(Violation(
                reference, name, "<missing>", "<missing>", "<missing>",
                "referenced model is absent from the catalog",
            ))
            continue
        violation = violation_for_model(entry, reference=reference, policy=active)
        if violation is not None:
            violations.append(violation)

    embedding_model = config.embedding.model
    if not isinstance(embedding_model, str) or not embedding_model.strip():
        violations.append(Violation(
            "[embedding].model", str(embedding_model), "local-artifact",
            "offline-cache", "<none>", "embedding model identifier is empty",
        ))
    elif "://" in embedding_model and not is_local_url(embedding_model):
        violations.append(Violation(
            "[embedding].model", _safe_identifier(embedding_model), "model-loader",
            _safe_url(embedding_model), _url_host(embedding_model),
            "embedding URL is not loopback HTTP(S)",
        ))
    neo4j_uri = config.neo4j.uri
    try:
        check_neo4j_uri(neo4j_uri, policy=active)
    except EgressPolicyError:
        violations.append(Violation(
            "[neo4j].uri", "<graph>", "neo4j", _safe_url(neo4j_uri), _url_host(neo4j_uri),
            "Neo4j URI must be a direct loopback Bolt endpoint (routing is forbidden)",
        ))
    return violations


def enabled_tracing_variables() -> list[str]:
    """Return tracing enable switches that are truthy in the current process."""
    return [
        name for name in TRACING_ENV_VARS
        if os.environ.get(name, "").strip().lower() in _TRUE_VALUES
    ]


def environment_violations() -> list[str]:
    """Inspect process switches without changing them (shared by doctor and startup)."""
    problems = [f"tracing is enabled by: {name}" for name in enabled_tracing_variables()]
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        value = os.environ.get(name, "").strip().lower()
        if value and value not in _TRUE_VALUES:
            problems.append(f"{name} must be 1, true, yes, or on")
    hub = sys.modules.get("huggingface_hub.constants")
    if hub is not None and getattr(hub, "HF_HUB_OFFLINE", False) is False:
        problems.append("huggingface_hub was imported online; restart with HF_HUB_OFFLINE=1")
    transformers = sys.modules.get("transformers.utils.hub")
    if transformers is not None and getattr(transformers, "_is_offline_mode", False) is False:
        problems.append("transformers was imported online; restart with TRANSFORMERS_OFFLINE=1")
    return problems


def _enable_offline_loaders() -> None:
    applied: list[str] = []
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name) != "1":
            applied.append(f"{name}=1")
        # The libraries read their environment without stripping whitespace. Normalize
        # every accepted spelling before either library can import its offline constant.
        os.environ[name] = "1"
    if applied:
        logger.info("local-only egress: enabled offline model loading (%s)", ", ".join(applied))


def format_violations(violations: list[Violation]) -> str:
    return "\n".join(f"  - {violation}" for violation in violations)


def enforce_startup_egress(config: BrainConfig) -> None:
    """Apply local-only process guards and reject every startup violation at once."""
    active = effective_egress(config)
    if active != LOCAL_ONLY:
        return

    proxies = [name for name in _PROXY_ENV_VARS if name in os.environ]
    if proxies:
        logger.warning("local-only egress: ignoring environment proxies (%s)", ", ".join(proxies))
    environment = environment_violations()
    if not any("OFFLINE" in problem for problem in environment):
        _enable_offline_loaders()
    model_violations = validate_egress(config, policy=active)
    if not model_violations and not environment:
        return

    sections = [f"egress policy {active!r} rejected startup"]
    if model_violations:
        sections.append(
            f"configured model violations ({len(model_violations)}):\n"
            f"{format_violations(model_violations)}"
        )
    sections.extend(environment)
    raise ConfigError("\n".join(sections))
