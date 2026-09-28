"""Resilient LLM invocation — retry, backoff, and fallback chain.

Wraps LangChain ChatModel calls with:
- Exponential backoff on rate-limit (429) and transient errors (5xx)
- Automatic fallback to next model in chain when retries exhausted
- Auth error detection (401) triggering OAuth token refresh

Usage:
    # The LLMProvider returns ResilientChatModel transparently
    llm = provider.get_for_stage("extraction")  # already resilient
    result = await llm.ainvoke(messages)         # retries + fallbacks automatic
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from typing import Any, Callable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from openclaw_brain.config import ResilienceConfig
from openclaw_brain.redaction import redact_secrets

logger = logging.getLogger(__name__)


# ── Error classification ──

class RetryableError(Exception):
    """Wraps an error that should be retried (429, 5xx)."""
    def __init__(self, original: Exception, retry_after: float | None = None):
        super().__init__(str(original))
        self.original = original
        self.retry_after = retry_after


class AuthError(Exception):
    """Wraps an auth error (401) that may be resolved by token refresh."""
    def __init__(self, original: Exception):
        super().__init__(str(original))
        self.original = original


class FallbackExhaustedError(Exception):
    """All models in the fallback chain failed."""


class TruncatedOutputError(Exception):
    """A model returned a SUCCESSFUL response whose finish_reason says it was cut off at the
    output-token limit ('length' / 'max_tokens' / 'MAX_TOKENS' depending on provider).

    Deliberately a plain Exception, NOT a RetryableError: the retry/backoff loop below is
    driven by ``classify_error`` over raised provider EXCEPTIONS, and a truncation is not an
    exception at all — the HTTP call succeeded. Re-running the identical request would
    reproduce the identical truncation, so the only useful response is a ONE-SHOT retry at an
    escalated bound (handled by ``_retry_once_on_truncation``); after that the chain must move
    on. Raising this type is how that "move on" is expressed to the model loop.
    """

    def __init__(
        self,
        model_name: str,
        finish_reason: str | None,
        *,
        max_tokens: int | None = None,
        escalated: bool = False,
    ):
        self.model_name = model_name
        self.finish_reason = finish_reason
        self.max_tokens = max_tokens
        self.escalated = escalated
        detail = f"max_tokens={max_tokens}" if max_tokens else "max_tokens=unknown"
        super().__init__(
            f"model={model_name} returned truncated output "
            f"(finish_reason={finish_reason!r}, {detail}, "
            f"escalated_retry={'yes' if escalated else 'no'}) — partial output discarded"
        )


# Provider spellings that all mean "output hit the token cap". OpenAI/OpenRouter/oMLX emit
# 'length'; Anthropic emits stop_reason 'max_tokens'; Google emits 'MAX_TOKENS'. Compared
# case-insensitively with separators stripped, so 'MAX_TOKENS'/'max-tokens'/'maxTokens' all hit.
_TRUNCATION_FINISH_REASONS = {
    "length", "maxtokens", "maxoutputtokens", "modellength", "outputlimit",
}


def detect_finish_reason(response: Any) -> str | None:
    """Best-effort extraction of a provider finish/stop reason from a LangChain response.

    Shapes handled (all observed in this stack — see the langchain_openai chat-model result
    builder and experiments/reason_shootout.py::_finish_reason):
      - ``AIMessage.response_metadata`` → {"finish_reason": "length", ...}  (openai/openrouter/
        local oMLX and google, which uses "MAX_TOKENS"; anthropic uses "stop_reason")
      - ``AIMessage.additional_kwargs`` → same keys, older/aggregated shape
      - ``.generation_info``            → {"finish_reason": ...} on a Generation/ChatGeneration
      - ``.generations[0][0].generation_info`` → LLMResult/ChatResult from ``.agenerate()``

    Returns the raw string as reported (NOT normalized — callers log it verbatim), or None
    when no reason is reachable. Never raises: a structured-output RunnableSequence, a plain
    string, or a Mock exposes none of these and yields None.
    """
    if response is None:
        return None

    for attr in ("response_metadata", "additional_kwargs", "generation_info"):
        payload = getattr(response, attr, None)
        if not isinstance(payload, dict):
            continue
        for key in ("finish_reason", "stop_reason", "finishReason"):
            value = payload.get(key)
            if value is not None:
                return str(value)

    # LLMResult / ChatResult: reasons live on the nested generations.
    generations = getattr(response, "generations", None)
    if isinstance(generations, (list, tuple)):
        for gen_group in generations:
            group = gen_group if isinstance(gen_group, (list, tuple)) else [gen_group]
            for gen in group:
                nested = detect_finish_reason(gen)
                if nested is not None:
                    return nested
                info = getattr(gen, "generation_info", None)
                if isinstance(info, dict):
                    for key in ("finish_reason", "stop_reason", "finishReason"):
                        value = info.get(key)
                        if value is not None:
                            return str(value)
    return None


def is_truncation_reason(finish_reason: str | None) -> bool:
    """True when a finish_reason means "cut off at the output-token limit".

    Absent/None/normal reasons ('stop', 'end_turn', 'error', ...) are False — the
    finish_reason branch is strictly additive and must be a no-op for them.
    """
    if not finish_reason:
        return False
    normalized = re.sub(r"[^a-z]", "", str(finish_reason).lower())
    return normalized in _TRUNCATION_FINISH_REASONS


def response_is_truncated(response: Any) -> bool:
    """Convenience: detect + classify in one call."""
    return is_truncation_reason(detect_finish_reason(response))


def classify_error(exc: Exception) -> Exception:
    """Classify an LLM error into retryable, auth, or fatal.

    Inspects status codes from common LLM client libraries
    (httpx, openai, anthropic) to determine the error category.
    """
    status = _extract_status(exc)

    if status == 401:
        return AuthError(exc)

    if status == 429:
        retry_after = _extract_retry_after(exc)
        return RetryableError(exc, retry_after=retry_after)

    if status is not None and 500 <= status < 600:
        return RetryableError(exc)

    # Connection/timeout errors are retryable
    error_type = type(exc).__name__.lower()
    error_msg = str(exc).lower()
    retryable_patterns = [
        "timeout", "timed out", "connection", "reset by peer",
        "server disconnected", "overloaded", "service unavailable",
    ]
    if any(p in error_type or p in error_msg for p in retryable_patterns):
        return RetryableError(exc)

    # Everything else is fatal — don't retry
    return exc


def _extract_status(exc: Exception) -> int | None:
    """Extract HTTP status code from various exception types."""
    # httpx.HTTPStatusError, openai.APIStatusError, anthropic.APIStatusError
    for attr in ("status_code", "status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    # Nested response object
    response = getattr(exc, "response", None)
    if response:
        for attr in ("status_code", "status"):
            val = getattr(response, attr, None)
            if isinstance(val, int):
                return val
    # Last resort: parse from string
    match = re.search(r'\b(4\d{2}|5\d{2})\b', str(exc))
    if match:
        return int(match.group(1))
    return None


def _err_diag(exc: object) -> str:
    """Ops-debuggable, secret-safe error string: exception TYPE + status (neither is a secret)
    + the redacted message. Fixes a regression where redact_secrets() alone could blank an
    error whose message was empty/terse (e.g. a bare timeout), leaving logs undiagnosable."""
    type_name = type(exc).__name__ if isinstance(exc, BaseException) else "error"
    status = _extract_status(exc) if isinstance(exc, Exception) else None
    msg = redact_secrets(exc)
    parts = [type_name]
    if status is not None:
        parts.append(f"status={status}")
    if msg and msg != type_name:
        parts.append(msg)
    return " ".join(parts)


def _extract_retry_after(exc: Exception) -> float | None:
    """Extract Retry-After header value from the exception if available."""
    response = getattr(exc, "response", None)
    if response:
        headers = getattr(response, "headers", {})
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
    return None


# ── Backoff calculator ──

def compute_backoff(
    attempt: int,
    config: ResilienceConfig,
    retry_after: float | None = None,
) -> float:
    """Compute the wait time before the next retry.

    Uses exponential backoff with optional jitter, respecting
    server-specified Retry-After if present.
    """
    if retry_after is not None and retry_after > 0:
        # Server told us how long to wait — respect it, but cap
        return min(retry_after, config.max_backoff_s)

    base = config.initial_backoff_s * (config.backoff_multiplier ** attempt)
    base = min(base, config.max_backoff_s)

    if config.jitter:
        # Full jitter: uniform [0, base]
        base = random.uniform(0, base)

    return base


# ── Circuit breaker ──
#
# A bad-window primary (e.g. an upstream provider having a rough hour) makes every call to it
# fail, but invoke_with_resilience() has no memory across calls — each new chunk pays that
# model's full per-model retry budget (max_retries+1 attempts × request_timeout_s) before
# advancing to the fallback chain. The breaker below gives the resilience layer cross-call
# memory: once a model has failed circuit_breaker_threshold times in a row, later chunks skip
# it outright (straight to the next model in the chain) until circuit_breaker_cooldown_s has
# elapsed, at which point exactly one "half-open" trial call is allowed through to test recovery.


class _BreakerState:
    """Per-model circuit breaker state. consecutive_failures resets to 0 on any success;
    opened_at is set to the clock time the breaker tripped OPEN, and cleared on success."""

    __slots__ = ("consecutive_failures", "opened_at")

    def __init__(self) -> None:
        self.consecutive_failures = 0
        self.opened_at: float | None = None


# Keyed by the same `model_name` value invoke_with_resilience() already computes per model in
# its loop. asyncio is single-threaded (no preemption between awaits), so this plain dict needs
# no lock — concurrent invoke_with_resilience() calls interleave only at await points, and each
# read-modify-write of a given model's state below happens without an intervening await.
_circuit_breakers: dict[Any, _BreakerState] = {}


def _monotonic() -> float:
    """Indirection over time.monotonic() so tests can control elapsed time deterministically
    (monkeypatch this function) instead of sleeping for real cooldown windows."""
    return time.monotonic()


def reset_circuit_breakers() -> None:
    """Clear all circuit breaker state. Not called anywhere in production — for test isolation
    only (tests should call this in a fixture before each test to avoid cross-test leakage)."""
    _circuit_breakers.clear()


def _breaker_is_open(model_name: Any, config: ResilienceConfig) -> bool:
    """True if `model_name`'s breaker is OPEN and should be skipped. False when CLOSED (no
    tripped breaker) or HALF-OPEN (cooldown elapsed — exactly one trial call is allowed through;
    its outcome is recorded by `_breaker_record_success`/`_breaker_record_failure` below, which
    naturally re-arms the skip window on failure or fully closes it on success)."""
    state = _circuit_breakers.get(model_name)
    if state is None or state.opened_at is None:
        return False
    return (_monotonic() - state.opened_at) < config.circuit_breaker_cooldown_s


def _breaker_record_success(model_name: Any) -> None:
    """A model call succeeded — fully reset its breaker (covers both the ordinary closed path
    and a half-open trial succeeding)."""
    _circuit_breakers.pop(model_name, None)


def _breaker_record_failure(model_name: Any, config: ResilienceConfig) -> None:
    """A model call failed and the outer loop is advancing to the next model (or exhausting
    the chain). Increment the consecutive-failure count; once it reaches the threshold, (re-)open
    the breaker by stamping opened_at to now. Applies equally to a fresh trip and to a half-open
    trial failing again — the latter just restamps opened_at, restarting the cooldown window."""
    state = _circuit_breakers.setdefault(model_name, _BreakerState())
    state.consecutive_failures += 1
    if state.consecutive_failures >= config.circuit_breaker_threshold:
        state.opened_at = _monotonic()


# ── Core retry logic ──

# Type for the optional auth-refresh callback
AuthRefreshCallback = Callable[[], bool] | None


def resolve_model_name(model: Any) -> str:
    """Best-effort stable identifier for circuit-breaker keying and logs.

    ``with_structured_output()`` returns a RunnableSequence that exposes neither
    ``model_name`` nor ``model``, so a *wrapped* model resolves to "unknown". Because the
    breaker is keyed by this string, that silently collapses every wrapped call — primary
    AND fallbacks — onto ONE "unknown" key: consecutive-failure counting can no longer tell
    deepseek from its gemini fallback, so a bad-window model is never skipped and each chunk
    re-pays the full per-call timeout before falling back. Callers that pre-wrap MUST pass
    ``model_names=`` computed from the RAW models; this helper is both the shared fallback and
    the source of those names.
    """
    return getattr(model, "model_name", None) or getattr(model, "model", None) or "unknown"


async def _invoke_once(
    model: Any,
    messages: list[BaseMessage],
    config: ResilienceConfig,
    method: str,
    **kwargs: Any,
) -> Any:
    """One model call, bounded by the configured request timeout (0 = unbounded)."""
    coro = getattr(model, method)(messages, **kwargs)
    if config.request_timeout_s > 0:
        return await asyncio.wait_for(coro, timeout=config.request_timeout_s)
    return await coro


def _effective_max_tokens(model: Any, kwargs: dict[str, Any]) -> int | None:
    """The output bound currently in force for this call: the call-time kwarg if present,
    else the bound baked into the model instance at construction (LLMProvider does that from
    the catalog entry / the stage budget). None when neither is known.

    The instance attribute is provider-spelled: ``ChatOpenAI``/``ChatAnthropic``/``ChatXAI``
    expose ``max_tokens``, while ``ChatGoogleGenerativeAI`` exposes ``max_output_tokens``.
    Reading only the former made every google fallback look "unbounded" and silently skip the
    escalation branch even though its bound was perfectly knowable. When NEITHER exists the
    bound is genuinely unknown and the caller still skips escalation — never guess a number.
    """
    bound = _effective_output_bound(model, kwargs)
    return bound[1] if bound is not None else None


# The provider spellings of the output bound, in resolution order. A model is asked for BOTH
# because the same logical bound is `max_tokens` on ChatOpenAI/ChatAnthropic/ChatXAI and
# `max_output_tokens` on ChatGoogleGenerativeAI (declared as
# `max_output_tokens: int | None = Field(default=None, alias="max_tokens")` — the alias works
# at CONSTRUCTION but the call-time path reads `kwargs["max_output_tokens"]`, so an escalated
# bind must use the name the model will actually read back).
_MAX_TOKEN_PARAM_NAMES = ("max_tokens", "max_output_tokens")


def _effective_output_bound(model: Any, kwargs: dict[str, Any]) -> tuple[str, int] | None:
    """``(param_name, value)`` for the output bound in force, or None when unknowable.

    ``param_name`` is the spelling the escalated retry must bind so the model honors it.
    """
    for name in _MAX_TOKEN_PARAM_NAMES:
        value = kwargs.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return name, value
    for name in _MAX_TOKEN_PARAM_NAMES:
        value = getattr(model, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return name, value
    return None


async def _retry_once_on_truncation(
    model: Any,
    model_name: str,
    messages: list[BaseMessage],
    config: ResilienceConfig,
    *,
    method: str,
    finish_reason: str | None,
    escalation: float,
    **kwargs: Any,
) -> Any:
    """Handle a truncated (finish_reason='length') response: discard, escalate once, or give up.

    Contract (ONE-SHOT — no loop, no recursion):
      (a) The partial output is DISCARDED here and never returned; a half-written GraphDelta
          must never reach extract_json/normalize, where it would either raise or — worse —
          parse into a silently-amputated delta.
      (b) Exactly one retry on the SAME model with max_tokens × ``escalation``, bound at call
          time via ``Runnable.bind`` so the shared, cached model instance is not mutated.
      (c) Anything else — no known base bound to escalate from, a model that cannot bind, or a
          second truncation — raises TruncatedOutputError, which the caller's model loop treats
          like any other per-model failure: advance to the next model in the fallback chain.
    """
    bound = _effective_output_bound(model, kwargs)
    base = bound[1] if bound is not None else None
    logger.warning(
        "Truncated output on model=%s (finish_reason=%r, max_tokens=%s) — DISCARDING the "
        "partial response (never parsed) and retrying ONCE with an escalated bound",
        model_name, finish_reason, base,
    )

    binder = getattr(model, "bind", None)
    if bound is None or not callable(binder):
        raise TruncatedOutputError(model_name, finish_reason, max_tokens=base)

    param_name, base = bound
    escalated = max(int(base * escalation), base + 1)
    # The escalated bound MUST NOT be re-overridden by the caller's own max_tokens.
    # `RunnableBinding.ainvoke` merges `{**self.kwargs, **kwargs}` — CALL kwargs win over BOUND
    # kwargs — so forwarding the caller's `max_tokens` here would re-issue the byte-identical
    # request, guaranteeing the same truncation and burning a paid call for nothing. Stripped
    # from a COPY: the caller's dict is untouched for every other attempt/model in the chain.
    retry_kwargs = {k: v for k, v in kwargs.items() if k not in _MAX_TOKEN_PARAM_NAMES}
    result = await _invoke_once(
        binder(**{param_name: escalated}), messages, config, method, **retry_kwargs,
    )

    retry_reason = detect_finish_reason(result)
    if is_truncation_reason(retry_reason):
        # Still truncated at 1.5× — stop here. Escalating further is an unbounded cost/latency
        # loop; the fallback chain is the right next move.
        raise TruncatedOutputError(
            model_name, retry_reason, max_tokens=escalated, escalated=True,
        )

    logger.info(
        "Truncation retry succeeded on model=%s (max_tokens %d -> %d)",
        model_name, base, escalated,
    )
    return result


async def invoke_with_resilience(
    models: Sequence[BaseChatModel],
    messages: list[BaseMessage],
    config: ResilienceConfig,
    *,
    method: str = "ainvoke",
    auth_refresh: AuthRefreshCallback = None,
    model_names: Sequence[str] | None = None,
    truncation_retry: bool = False,
    truncation_escalation: float = 1.5,
    **kwargs: Any,
) -> Any:
    """Invoke an LLM with retry + fallback chain.

    Args:
        models: Ordered list of models to try (primary first, then fallbacks).
        messages: The messages to send.
        config: Resilience configuration.
        method: The method to call on the model ("ainvoke" or "with_structured_output").
        auth_refresh: Optional callback to refresh OAuth tokens on 401.
        model_names: Optional per-model breaker/log labels, positionally aligned with
            ``models``. Required when ``models`` are ``with_structured_output()``-wrapped
            (which hides ``model_name``); pass names resolved from the RAW models via
            ``resolve_model_name``. Falls back to ``resolve_model_name(model)`` per entry.
        truncation_retry: Opt-in finish_reason branch (OFF by default — every existing caller
            keeps byte-identical behavior). When True, a response whose finish_reason says
            'length'/'max_tokens' is DISCARDED and retried ONCE on the same model at 1.5× its
            bound; a second truncation advances to the next model. A response with no
            finish_reason, or a normal one, is returned untouched exactly as before.
            The escalation is ONE-SHOT per model per invocation (not per retry attempt), the
            escalated call never forwards the caller's own max_tokens (it would override the
            bind and reproduce the truncation verbatim), and a persistent truncation does NOT
            record a circuit-breaker failure — it is a content failure on a successful call,
            not an availability failure.
        truncation_escalation: Multiplier for that one-shot retry's max_tokens (default 1.5).
        **kwargs: Passed to the model method.

    Returns:
        The model response.

    Raises:
        FallbackExhaustedError: All models exhausted their retries.
    """
    last_error: Exception | None = None

    for model_idx, model in enumerate(models):
        model_name = (
            model_names[model_idx]
            if model_names is not None and model_idx < len(model_names)
            else resolve_model_name(model)
        )

        if config.circuit_breaker_enabled and _breaker_is_open(model_name, config):
            logger.warning(
                "Circuit breaker OPEN for model=%s — skipping (cooldown %.0fs)",
                model_name, config.circuit_breaker_cooldown_s,
            )
            last_error = RuntimeError(
                f"circuit breaker open for model={model_name} (consecutive failures "
                f"reached threshold; cooling down before a half-open retry)"
            )
            continue

        auth_refreshed = False
        # ONE-SHOT means one escalation per model per INVOCATION, not per attempt. The outer
        # `attempt` loop re-runs the whole body on a retryable error, so without this flag an
        # escalated call that hits a 429 would send the loop around and escalate again on the
        # next attempt (measured: 8 calls at max_retries=3). Scoped inside the model loop so
        # the next model in the chain still gets its own single escalation.
        truncation_escalated = False

        for attempt in range(config.max_retries + 1):
            try:
                result = await _invoke_once(model, messages, config, method, **kwargs)

                # Additive finish_reason branch (opt-in). A truncated response is a SUCCESSFUL
                # call carrying unusable output, so it is invisible to classify_error below —
                # this is the only place it can be caught. When the flag is off, or the
                # provider reports no/normal finish_reason, nothing changes here.
                if truncation_retry:
                    finish_reason = detect_finish_reason(result)
                    if is_truncation_reason(finish_reason):
                        if truncation_escalated:
                            # Budget already spent on this model — do not escalate twice.
                            raise TruncatedOutputError(
                                model_name, finish_reason,
                                max_tokens=_effective_max_tokens(model, kwargs),
                                escalated=True,
                            )
                        # Set BEFORE the call so a retryable failure inside the escalated
                        # request cannot buy a second escalation on the next attempt.
                        truncation_escalated = True
                        result = await _retry_once_on_truncation(
                            model, model_name, messages, config,
                            method=method,
                            finish_reason=finish_reason,
                            escalation=truncation_escalation,
                            **kwargs,
                        )

                if model_idx > 0:
                    logger.info(
                        "Fallback succeeded: model=%s (index %d)",
                        model_name, model_idx,
                    )
                if config.circuit_breaker_enabled:
                    _breaker_record_success(model_name)
                return result

            except TruncatedOutputError as exc:
                # Handled BEFORE the generic handler so classify_error is never fed a
                # non-provider exception (it would guess a status out of the message text).
                # A truncation is deterministic — retrying the same model at the same bound
                # reproduces it — so no backoff loop: fall through to the next model in the
                # chain, i.e. the existing cascade behavior.
                #
                # NO BREAKER FAILURE IS RECORDED — deliberate. The circuit breaker measures
                # AVAILABILITY: is this endpoint answering at all? A truncation is a CONTENT-
                # shaped failure on a call that SUCCEEDED (HTTP 200, tokens billed, the model
                # merely ran out of output budget). Counting it as unavailability meant a run
                # of long chunks — exactly the workload the reasoning stage exists for —
                # tripped the breaker and evicted the production reasoning model from the
                # chain for the whole cooldown, degrading every subsequent chunk over a bound
                # that is ours to set. Same reasoning for any other availability-style counter.
                logger.error(
                    "Truncated output persisted on model=%s after the one-shot escalation "
                    "(%s) — advancing to the next model in the chain (circuit breaker NOT "
                    "tripped: content-shaped failure, not an availability failure)",
                    model_name, exc,
                )
                last_error = exc
                break

            except Exception as exc:
                classified = classify_error(exc)

                if isinstance(classified, AuthError):
                    if auth_refresh and not auth_refreshed:
                        logger.warning(
                            "Auth error on model=%s, attempting token refresh",
                            model_name,
                        )
                        refreshed = auth_refresh()
                        if refreshed:
                            auth_refreshed = True
                            continue  # Retry same attempt (don't increment)
                    # Auth failed even after refresh — try next model
                    logger.error(
                        "Auth error on model=%s, no refresh available",
                        model_name,
                    )
                    last_error = classified.original
                    if config.circuit_breaker_enabled:
                        _breaker_record_failure(model_name, config)
                    break

                if isinstance(classified, RetryableError):
                    if attempt < config.max_retries:
                        wait = compute_backoff(
                            attempt, config,
                            retry_after=classified.retry_after,
                        )
                        logger.warning(
                            "Retryable error on model=%s (attempt %d/%d), "
                            "waiting %.1fs: %s",
                            model_name, attempt + 1, config.max_retries,
                            wait, _err_diag(exc),
                        )
                        await asyncio.sleep(wait)
                        continue
                    else:
                        logger.error(
                            "Retries exhausted for model=%s: %s",
                            model_name, _err_diag(exc),
                        )
                        last_error = classified.original
                        if config.circuit_breaker_enabled:
                            _breaker_record_failure(model_name, config)
                        break

                # Fatal error — skip to next model
                logger.error(
                    "Fatal error on model=%s: %s",
                    model_name, _err_diag(exc),
                )
                last_error = exc
                if config.circuit_breaker_enabled:
                    _breaker_record_failure(model_name, config)
                break

    # Redact the last error AND raise `from None`: a misbehaving endpoint can reflect the
    # submitted auth header into its error body, so neither the message nor the chained
    # traceback may carry the raw provider exception. Per-attempt logs above are redacted too.
    raise FallbackExhaustedError(
        f"All {len(models)} model(s) failed. Last error: {_err_diag(last_error)}"
    ) from None
