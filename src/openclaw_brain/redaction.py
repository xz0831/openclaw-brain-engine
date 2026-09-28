"""Scrub credential shapes from text before it reaches a capturable stream.

Defense-in-depth for the leak CLASS: a resolved API key lives in os.environ and the
outbound HTTP auth header, so it can never appear in our own data structures — EXCEPT
if a provider/endpoint reflects the submitted Authorization back into an error body,
which the resilience layer would then format into a log, a FallbackExhaustedError
message/traceback, or a persisted checkpoint (see the secret-leak audit). This module
redacts known key shapes at those sinks so no such reflection can surface a credential.

`redact_secrets` is deliberately shape-based (not value-based): it does not need the
live keys, so it works even for credentials this process never resolved.
"""

from __future__ import annotations

import re

# Known credential shapes. Order doesn't matter (all applied); over-redaction of a
# look-alike token is acceptable, under-redaction of a real key is not.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),                       # Anthropic
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),                          # OpenAI-style
    re.compile(r"xai-[A-Za-z0-9_\-]{8,}"),                          # xAI
    re.compile(r"AIza[A-Za-z0-9_\-]{20,}"),                         # Google
    re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),  # JWT (oauth)
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),               # Bearer <token>
    re.compile(r"(?i)(authorization|x-api-key)(\s*[:=]\s*)\S+"),    # header echo
)

_REDACTED = "[REDACTED]"


def redact_secrets(text: object) -> str:
    """Return `str(text)` with any credential-shaped substring replaced by [REDACTED]."""
    if text is None:
        return ""
    s = str(text)
    for pat in _SECRET_PATTERNS:
        # keep the header NAME, redact only its value, for the header-echo pattern
        if "authorization" in pat.pattern.lower():
            s = pat.sub(lambda m: f"{m.group(1)}{m.group(2)}{_REDACTED}", s)
        else:
            s = pat.sub(_REDACTED, s)
    return s
