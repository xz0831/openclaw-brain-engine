"""Tests for LLM resilience — retry, backoff, and fallback chain."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openclaw_brain.config import ResilienceConfig
from openclaw_brain.llm import resilience as resilience_module
from openclaw_brain.llm.resilience import (
    AuthError,
    FallbackExhaustedError,
    RetryableError,
    classify_error,
    compute_backoff,
    invoke_with_resilience,
    reset_circuit_breakers,
    resolve_model_name,
)


@pytest.fixture(autouse=True)
def _reset_breakers():
    """Circuit breaker state is module-level (keyed by model_name) — isolate every test."""
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


# ── Error classification tests ──


def test_classify_429_as_retryable():
    exc = Exception("Rate limited")
    exc.status_code = 429
    result = classify_error(exc)
    assert isinstance(result, RetryableError)


def test_classify_500_as_retryable():
    exc = Exception("Server error")
    exc.status_code = 500
    result = classify_error(exc)
    assert isinstance(result, RetryableError)


def test_classify_503_as_retryable():
    exc = Exception("Service unavailable")
    exc.status_code = 503
    result = classify_error(exc)
    assert isinstance(result, RetryableError)


def test_classify_401_as_auth():
    exc = Exception("Unauthorized")
    exc.status_code = 401
    result = classify_error(exc)
    assert isinstance(result, AuthError)


def test_classify_400_as_fatal():
    exc = Exception("Bad request")
    exc.status_code = 400
    result = classify_error(exc)
    assert not isinstance(result, (RetryableError, AuthError))
    assert isinstance(result, Exception)


def test_classify_timeout_as_retryable():
    exc = TimeoutError("Connection timed out")
    result = classify_error(exc)
    assert isinstance(result, RetryableError)


def test_classify_connection_error_as_retryable():
    exc = ConnectionError("Connection reset by peer")
    result = classify_error(exc)
    assert isinstance(result, RetryableError)


def test_classify_generic_error_as_fatal():
    exc = ValueError("Invalid JSON")
    result = classify_error(exc)
    assert not isinstance(result, (RetryableError, AuthError))


def test_classify_status_from_response_object():
    exc = Exception("Error")
    exc.response = MagicMock()
    exc.response.status_code = 429
    result = classify_error(exc)
    assert isinstance(result, RetryableError)


def test_classify_status_from_string():
    exc = Exception("HTTP 429 Too Many Requests")
    result = classify_error(exc)
    assert isinstance(result, RetryableError)


def test_retry_after_extraction():
    exc = Exception("Rate limited")
    exc.status_code = 429
    exc.response = MagicMock()
    exc.response.headers = {"retry-after": "5.0"}
    result = classify_error(exc)
    assert isinstance(result, RetryableError)
    assert result.retry_after == 5.0


# ── Backoff computation tests ──


def test_backoff_exponential():
    config = ResilienceConfig(
        initial_backoff_s=1.0,
        backoff_multiplier=2.0,
        max_backoff_s=60.0,
        jitter=False,
    )
    assert compute_backoff(0, config) == 1.0
    assert compute_backoff(1, config) == 2.0
    assert compute_backoff(2, config) == 4.0
    assert compute_backoff(3, config) == 8.0


def test_backoff_capped():
    config = ResilienceConfig(
        initial_backoff_s=1.0,
        backoff_multiplier=2.0,
        max_backoff_s=10.0,
        jitter=False,
    )
    assert compute_backoff(10, config) == 10.0


def test_backoff_with_jitter():
    config = ResilienceConfig(
        initial_backoff_s=1.0,
        backoff_multiplier=2.0,
        max_backoff_s=60.0,
        jitter=True,
    )
    values = [compute_backoff(2, config) for _ in range(20)]
    # With jitter, values should vary (uniform [0, 4.0])
    assert min(values) < max(values)
    assert all(0 <= v <= 4.0 for v in values)


def test_backoff_respects_retry_after():
    config = ResilienceConfig(jitter=False)
    assert compute_backoff(0, config, retry_after=10.0) == 10.0


def test_backoff_caps_retry_after():
    config = ResilienceConfig(max_backoff_s=5.0, jitter=False)
    assert compute_backoff(0, config, retry_after=30.0) == 5.0


# ── invoke_with_resilience tests ──


@pytest.mark.asyncio
async def test_success_on_first_try():
    model = AsyncMock()
    model.ainvoke.return_value = "result"
    config = ResilienceConfig()

    result = await invoke_with_resilience([model], ["msg"], config)
    assert result == "result"
    model.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_retry_on_retryable_error():
    model = AsyncMock()
    retryable = Exception("Rate limited")
    retryable.status_code = 429
    model.ainvoke.side_effect = [retryable, "success"]
    config = ResilienceConfig(max_retries=3, initial_backoff_s=0.01)

    with patch("openclaw_brain.llm.resilience.asyncio.sleep"):
        result = await invoke_with_resilience([model], ["msg"], config)
    assert result == "success"
    assert model.ainvoke.call_count == 2


@pytest.mark.asyncio
async def test_fallback_on_retries_exhausted():
    primary = AsyncMock()
    retryable = Exception("overloaded")
    retryable.status_code = 503
    primary.ainvoke.side_effect = retryable

    fallback = AsyncMock()
    fallback.ainvoke.return_value = "fallback_result"

    config = ResilienceConfig(max_retries=1, initial_backoff_s=0.01)

    with patch("openclaw_brain.llm.resilience.asyncio.sleep"):
        result = await invoke_with_resilience([primary, fallback], ["msg"], config)
    assert result == "fallback_result"


@pytest.mark.asyncio
async def test_all_models_exhausted_raises():
    model1 = AsyncMock()
    model2 = AsyncMock()
    retryable = Exception("overloaded")
    retryable.status_code = 503
    model1.ainvoke.side_effect = retryable
    model2.ainvoke.side_effect = retryable

    config = ResilienceConfig(max_retries=0, initial_backoff_s=0.01)

    with pytest.raises(FallbackExhaustedError):
        await invoke_with_resilience([model1, model2], ["msg"], config)


@pytest.mark.asyncio
async def test_reflected_key_redacted_from_raised_error():
    """A misbehaving endpoint that reflects the submitted key into its error body must
    NOT surface that key in the FallbackExhaustedError message or its chained traceback.
    """
    model = AsyncMock()
    reflected = Exception(
        "401 Unauthorized: invalid credential 'sk-ant-api03-FAKEsecretDONOTUSE0000'"
    )
    reflected.status_code = 400  # fatal → straight to FallbackExhaustedError
    model.ainvoke.side_effect = reflected
    config = ResilienceConfig()

    with pytest.raises(FallbackExhaustedError) as ei:
        await invoke_with_resilience([model], ["msg"], config)

    msg = str(ei.value)
    assert "sk-ant-api03-FAKEsecretDONOTUSE0000" not in msg
    assert "[REDACTED]" in msg
    # `from None` suppresses the raw provider exception in the printed traceback
    assert ei.value.__cause__ is None
    assert ei.value.__suppress_context__ is True


@pytest.mark.asyncio
async def test_fatal_error_skips_to_fallback():
    primary = AsyncMock()
    primary.ainvoke.side_effect = ValueError("Bad JSON schema")

    fallback = AsyncMock()
    fallback.ainvoke.return_value = "ok"

    config = ResilienceConfig(max_retries=3)

    result = await invoke_with_resilience([primary, fallback], ["msg"], config)
    assert result == "ok"
    # Primary should only be called once (fatal = no retry)
    assert primary.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_auth_error_triggers_refresh():
    model = AsyncMock()
    auth_exc = Exception("Unauthorized")
    auth_exc.status_code = 401
    model.ainvoke.side_effect = [auth_exc, "success_after_refresh"]

    refresh = MagicMock(return_value=True)
    config = ResilienceConfig(max_retries=2)

    result = await invoke_with_resilience(
        [model], ["msg"], config, auth_refresh=refresh,
    )
    assert result == "success_after_refresh"
    refresh.assert_called_once()


@pytest.mark.asyncio
async def test_auth_error_refresh_fails_tries_fallback():
    primary = AsyncMock()
    auth_exc = Exception("Unauthorized")
    auth_exc.status_code = 401
    primary.ainvoke.side_effect = auth_exc

    fallback = AsyncMock()
    fallback.ainvoke.return_value = "fallback_ok"

    refresh = MagicMock(return_value=False)
    config = ResilienceConfig(max_retries=2)

    result = await invoke_with_resilience(
        [primary, fallback], ["msg"], config, auth_refresh=refresh,
    )
    assert result == "fallback_ok"


@pytest.mark.asyncio
async def test_single_model_no_fallback():
    model = AsyncMock()
    model.ainvoke.return_value = "solo"
    config = ResilienceConfig()

    result = await invoke_with_resilience([model], ["msg"], config)
    assert result == "solo"


# ── Circuit breaker tests ──


def _fake_clock(monkeypatch, start: float = 0.0) -> dict:
    """Monkeypatch resilience's clock indirection to a controllable value (no real sleeps)."""
    box = {"t": start}
    monkeypatch.setattr(resilience_module, "_monotonic", lambda: box["t"])
    return box


def _make_503(msg: str = "overloaded") -> Exception:
    exc = Exception(msg)
    exc.status_code = 503
    return exc


@pytest.mark.asyncio
async def test_breaker_opens_after_threshold_consecutive_failures(monkeypatch):
    box = _fake_clock(monkeypatch)
    model = AsyncMock()
    model.model_name = "flaky-model"
    model.ainvoke.side_effect = _make_503()

    config = ResilienceConfig(
        max_retries=0, circuit_breaker_threshold=3, circuit_breaker_cooldown_s=120.0,
    )

    for i in range(1, 4):
        with pytest.raises(FallbackExhaustedError):
            await invoke_with_resilience([model], ["msg"], config)
        state = resilience_module._circuit_breakers["flaky-model"]
        assert state.consecutive_failures == i
        if i < 3:
            assert state.opened_at is None, "must not open before threshold is reached"
        else:
            assert state.opened_at == box["t"], "must open on the failure that reaches threshold"


@pytest.mark.asyncio
async def test_breaker_open_model_is_skipped_and_next_model_tried(monkeypatch):
    _fake_clock(monkeypatch)
    primary = AsyncMock()
    primary.model_name = "bad-primary"
    primary.ainvoke.side_effect = _make_503()

    fallback = AsyncMock()
    fallback.model_name = "good-fallback"
    fallback.ainvoke.return_value = "fallback_result"

    config = ResilienceConfig(max_retries=0, circuit_breaker_threshold=2)

    # Two failing calls (primary alone in the chain) trip its breaker open.
    for _ in range(2):
        with pytest.raises(FallbackExhaustedError):
            await invoke_with_resilience([primary], ["msg"], config)
    assert primary.ainvoke.call_count == 2

    # Now the chain includes the fallback — the breaker must skip primary outright.
    result = await invoke_with_resilience([primary, fallback], ["msg"], config)
    assert result == "fallback_result"
    assert primary.ainvoke.call_count == 2, "open breaker must not call the model again"
    fallback.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_breaker_half_open_after_cooldown_allows_one_trial(monkeypatch):
    box = _fake_clock(monkeypatch, start=1000.0)
    model = AsyncMock()
    model.model_name = "recovering-model"
    model.ainvoke.side_effect = _make_503()

    config = ResilienceConfig(
        max_retries=0, circuit_breaker_threshold=1, circuit_breaker_cooldown_s=60.0,
    )

    # One failure trips the breaker open (threshold=1) at t=1000.
    with pytest.raises(FallbackExhaustedError):
        await invoke_with_resilience([model], ["msg"], config)
    assert model.ainvoke.call_count == 1

    # Still within cooldown -> skipped entirely, not called again.
    box["t"] = 1030.0
    with pytest.raises(FallbackExhaustedError):
        await invoke_with_resilience([model], ["msg"], config)
    assert model.ainvoke.call_count == 1

    # Cooldown elapsed -> exactly one half-open trial is let through.
    box["t"] = 1061.0
    model.ainvoke.side_effect = None
    model.ainvoke.return_value = "recovered"
    result = await invoke_with_resilience([model], ["msg"], config)
    assert result == "recovered"
    assert model.ainvoke.call_count == 2, "the half-open trial must actually reach the model"


@pytest.mark.asyncio
async def test_breaker_success_resets_that_models_breaker(monkeypatch):
    _fake_clock(monkeypatch)
    model = AsyncMock()
    model.model_name = "flaky-then-fine"
    model.ainvoke.side_effect = _make_503()

    config = ResilienceConfig(max_retries=0, circuit_breaker_threshold=3)

    # Two failures — below threshold, not yet open.
    for _ in range(2):
        with pytest.raises(FallbackExhaustedError):
            await invoke_with_resilience([model], ["msg"], config)
    assert resilience_module._circuit_breakers["flaky-then-fine"].consecutive_failures == 2

    # A success must fully clear the breaker state for this model.
    model.ainvoke.side_effect = None
    model.ainvoke.return_value = "ok"
    result = await invoke_with_resilience([model], ["msg"], config)
    assert result == "ok"
    assert "flaky-then-fine" not in resilience_module._circuit_breakers


@pytest.mark.asyncio
async def test_breaker_per_model_isolation(monkeypatch):
    _fake_clock(monkeypatch)
    model_a = AsyncMock()
    model_a.model_name = "model-a"
    model_a.ainvoke.side_effect = _make_503()

    model_b = AsyncMock()
    model_b.model_name = "model-b"
    model_b.ainvoke.return_value = "b-result"

    config = ResilienceConfig(max_retries=0, circuit_breaker_threshold=2)

    # Open model_a's breaker (alone in the chain so it can't fall through to model_b yet).
    for _ in range(2):
        with pytest.raises(FallbackExhaustedError):
            await invoke_with_resilience([model_a], ["msg"], config)
    assert resilience_module._circuit_breakers["model-a"].opened_at is not None

    # model_b must be entirely unaffected by model_a's open breaker.
    result = await invoke_with_resilience([model_a, model_b], ["msg"], config)
    assert result == "b-result"
    model_b.ainvoke.assert_called_once()
    assert "model-b" not in resilience_module._circuit_breakers


@pytest.mark.asyncio
async def test_breaker_disabled_is_pure_noop(monkeypatch):
    _fake_clock(monkeypatch)
    model = AsyncMock()
    model.model_name = "always-fails"
    model.ainvoke.side_effect = _make_503()

    config = ResilienceConfig(
        max_retries=0, circuit_breaker_enabled=False, circuit_breaker_threshold=1,
    )

    for _ in range(5):
        with pytest.raises(FallbackExhaustedError):
            await invoke_with_resilience([model], ["msg"], config)

    # No state is ever tracked, and every call actually reaches the model (no skipping).
    assert resilience_module._circuit_breakers == {}
    assert model.ainvoke.call_count == 5


@pytest.mark.asyncio
async def test_breaker_happy_path_never_opens(monkeypatch):
    _fake_clock(monkeypatch)
    model = AsyncMock()
    model.model_name = "reliable-model"
    model.ainvoke.return_value = "ok"
    config = ResilienceConfig()

    for _ in range(5):
        result = await invoke_with_resilience([model], ["msg"], config)
        assert result == "ok"

    assert resilience_module._circuit_breakers == {}


# ── model_name attribution (structured-output wrapper defeats breaker keying) ──


def test_resolve_model_name_raw_vs_wrapped():
    raw = AsyncMock()
    raw.model_name = "deepseek/deepseek-v4-flash"
    assert resolve_model_name(raw) == "deepseek/deepseek-v4-flash"

    # with_structured_output() returns a RunnableSequence exposing neither model_name nor
    # model → the value that silently collapsed every wrapped call onto one breaker key.
    wrapped = AsyncMock()
    wrapped.model_name = None
    wrapped.model = None
    assert resolve_model_name(wrapped) == "unknown"


@pytest.mark.asyncio
async def test_model_names_restores_per_model_breaker_for_wrapped_models(monkeypatch):
    """Regression: with_structured_output() hides model_name, so every wrapped call keyed the
    breaker under one shared "unknown". Once the primary tripped that key open, the *healthy*
    fallback (also "unknown") was skipped too — a chunk exhausted its chain despite a working
    fallback. Passing model_names= restores per-model attribution."""
    _fake_clock(monkeypatch)
    primary = AsyncMock()
    primary.model_name = None
    primary.model = None
    primary.ainvoke.side_effect = _make_503()
    fallback = AsyncMock()
    fallback.model_name = None
    fallback.model = None
    fallback.ainvoke.return_value = "fallback_result"

    config = ResilienceConfig(max_retries=0, circuit_breaker_threshold=2)

    # BUG shape: no model_names → both resolve to "unknown". The primary's two failures open the
    # shared key, which then skips the healthy fallback too.
    for _ in range(2):
        with pytest.raises(FallbackExhaustedError):
            await invoke_with_resilience([primary], ["m"], config)
    with pytest.raises(FallbackExhaustedError):
        await invoke_with_resilience([primary, fallback], ["m"], config)
    fallback.ainvoke.assert_not_called()  # collapsed under "unknown" → never reached

    # FIX: distinct model_names → only the primary's own breaker opens; the fallback is still tried.
    reset_circuit_breakers()
    for _ in range(2):
        with pytest.raises(FallbackExhaustedError):
            await invoke_with_resilience([primary], ["m"], config, model_names=["deepseek"])
    result = await invoke_with_resilience(
        [primary, fallback], ["m"], config, model_names=["deepseek", "gemini"],
    )
    assert result == "fallback_result"
    fallback.ainvoke.assert_called_once()
