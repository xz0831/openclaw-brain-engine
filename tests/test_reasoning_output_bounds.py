"""Reasoning-stage output bounds + parse-failure diagnosability (2026-07-25).

Three surgical changes are covered here:

1. **Parse-failure diagnostics** — `extract_json`'s ValueError carried only a LENGTH, so the
   2026-07 lecture batch's 24 parse failures (p50 10.8k chars, max 22.7k) were unclassifiable:
   truncated JSON or prose? `normalize.describe_parse_failure()` now emits length + last 200
   chars + brace presence + finish_reason at the pipeline call site.
2. **`[reasoning].output_token_budget` wiring** — declared in config.py since forever, read by
   NOTHING (verified dead 2026-07-25), so the production reasoning model (deepseek-v4-flash /
   openrouter, no catalog max_tokens) ran unbounded. Now injected by `LLMProvider._stage_kwargs`
   with an explicit precedence: caller kwarg > catalog entry max_tokens > stage budget.
3. **finish_reason-aware truncation branch** — `resilience.TruncatedOutputError` +
   `detect_finish_reason` / `is_truncation_reason`; a 'length' finish discards the partial
   output and retries ONCE at 1.5x on the same model before the chain moves on.

No network, no billable call: models are fakes mirroring the real ChatOpenAI surface that
`invoke_with_resilience` touches (`ainvoke(messages, **kwargs)` -> AIMessage with
`response_metadata`, `.bind(**kwargs)` -> bound runnable, `.max_tokens`, `.model_name`), and
provider tests instantiate `local`-provider catalog entries (ChatOpenAI against a localhost
base_url, constructed only — never invoked).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from openclaw_brain.config import (
    BrainConfig,
    ModelEntry,
    ModelsConfig,
    ResilienceConfig,
    save_config,
)
from openclaw_brain.knowledge.extraction.models import ExtractionResult, MatchResult
from openclaw_brain.knowledge.graph.schema import GraphDelta
from openclaw_brain.knowledge.pipeline import KnowledgePipeline, ReasoningDegradedError
from openclaw_brain.knowledge.reasoning import normalize as normalize_module
from openclaw_brain.knowledge.reasoning.normalize import (
    describe_parse_failure,
    extract_json,
)
from openclaw_brain.llm.provider import LLMProvider
from openclaw_brain.llm.resilience import (
    FallbackExhaustedError,
    TruncatedOutputError,
    _breaker_is_open,
    _circuit_breakers,
    detect_finish_reason,
    invoke_with_resilience,
    is_truncation_reason,
    reset_circuit_breakers,
)


@pytest.fixture(autouse=True)
def _reset_breakers():
    """Breaker state is module-level and keyed by model name — isolate every test."""
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


# ── Fakes mirroring the real shapes ──────────────────────────────────


class FakeBoundModel:
    """What `BaseChatModel.bind(**kwargs)` returns: a runnable that merges the bound kwargs
    into every call (langchain_core `_ChatModelBinding` semantics)."""

    def __init__(self, model: "FakeChatModel", bound: dict[str, Any]):
        self._model = model
        self.kwargs = bound

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        return await self._model.ainvoke(messages, **{**self.kwargs, **kwargs})


class FakeChatModel:
    """Minimal stand-in for ChatOpenAI, limited to the surface the resilience layer touches."""

    def __init__(
        self,
        responses: list[Any],
        *,
        max_tokens: int | None = None,
        model_name: str = "fake-reasoner",
        bound_attr: str = "max_tokens",
    ):
        self._responses = list(responses)
        # `bound_attr` mirrors the provider spelling of the output bound: ChatOpenAI/
        # ChatAnthropic/ChatXAI expose `max_tokens`, ChatGoogleGenerativeAI exposes
        # `max_output_tokens` (declared `Field(default=None, alias="max_tokens")`).
        setattr(self, bound_attr, max_tokens)
        self.bound_attr = bound_attr
        self.model_name = model_name
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        resp = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(resp, Exception):
            raise resp
        return resp

    def bind(self, **kwargs: Any) -> FakeBoundModel:
        return FakeBoundModel(self, kwargs)

    def with_structured_output(self, schema: Any) -> Any:
        # Real models that cannot honor the schema fail at invoke time; the pipeline's
        # structured-first attempt then falls through to raw + normalize.
        class _Broken:
            async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
                raise ValueError("structured output unsupported by this model")

        return _Broken()


def _msg(content: str, finish_reason: str | None = "stop") -> AIMessage:
    """A real AIMessage — response_metadata is the actual field providers populate."""
    metadata = {"finish_reason": finish_reason} if finish_reason is not None else {}
    return AIMessage(content=content, response_metadata=metadata)


_GOOD_DELTA_JSON = json.dumps(
    {
        "new_nodes": [
            {
                "proposed_id": "kt_c_noise",
                "label": "Concept",
                "canonical_name": "kTC Noise",
                "description": "Sampled thermal noise of a switched capacitor.",
                "domain": "analog_circuits",
                "confidence": 0.9,
                "evidence_chunk_ids": ["chunk_0"],
                "reasoning": "Stated in the chunk.",
            }
        ],
        "new_edges": [],
    }
)

# What a cut-off GraphDelta actually looks like: stops mid-token, no closing brace.
_TRUNCATED_JSON = _GOOD_DELTA_JSON[: len(_GOOD_DELTA_JSON) // 2]

# What a prose failure looks like: complete sentences, no JSON object at all.
_PROSE = (
    "I reviewed the chunk carefully. The passage describes correlated double sampling "
    "and its relationship to reset noise, but it does not contain enough grounded "
    "material to justify emitting new graph nodes at this time."
)


# ── 1. Budget wiring (provider) ──────────────────────────────────────


def _budget_config(
    *,
    primary_max_tokens: int | None = None,
    fallback_max_tokens: int | None = None,
) -> BrainConfig:
    cfg = BrainConfig()
    cfg.models = ModelsConfig(
        default_extraction="test-extract",
        default_reasoning="test-reason",
        default_matching="test-extract",
        catalog=[
            ModelEntry(
                name="test-reason",
                provider="local",
                model_id="reasoner-local",
                endpoint="http://localhost:8000/v1",
                tier="local",
                max_tokens=primary_max_tokens,
            ),
            ModelEntry(
                name="test-reason-fallback",
                provider="local",
                model_id="reasoner-fallback",
                endpoint="http://localhost:8000/v1",
                tier="local",
                max_tokens=fallback_max_tokens,
            ),
            ModelEntry(
                name="test-extract",
                provider="local",
                model_id="extractor-local",
                endpoint="http://localhost:8000/v1",
                tier="local",
            ),
        ],
    )
    cfg.resilience.fallback_reasoning = ["test-reason-fallback"]
    return cfg


def test_stage_budget_applies_when_entry_declares_no_max_tokens():
    """The production shape: deepseek-v4-flash declares no catalog max_tokens, so the
    reasoning call was unbounded. It must now carry [reasoning].output_token_budget."""
    cfg = _budget_config()
    llm = LLMProvider(cfg).get_chain("reasoning")[0]
    assert llm.max_tokens == cfg.reasoning.output_token_budget == 16000


def test_catalog_max_tokens_wins_over_stage_budget():
    """The exo models' 6000 runaway-guard must never be widened to the stage default."""
    cfg = _budget_config(primary_max_tokens=6000)
    llm = LLMProvider(cfg).get_chain("reasoning")[0]
    assert llm.max_tokens == 6000


def test_changed_output_token_budget_flows_through():
    cfg = _budget_config()
    cfg.reasoning.output_token_budget = 3000
    llm = LLMProvider(cfg).get_chain("reasoning")[0]
    assert llm.max_tokens == 3000


def test_explicit_caller_kwarg_beats_both():
    cfg = _budget_config(primary_max_tokens=6000)
    llm = LLMProvider(cfg).get_chain("reasoning", max_tokens=1234)[0]
    assert llm.max_tokens == 1234


def test_budget_applies_to_reasoning_fallbacks_per_model():
    """Fallbacks serve the same stage, so they get the same resolution — but per model: a
    fallback WITH its own catalog bound keeps it."""
    cfg = _budget_config()
    chain = LLMProvider(cfg).get_chain("reasoning")
    assert [m.max_tokens for m in chain] == [16000, 16000]

    bounded = _budget_config(fallback_max_tokens=6000)
    chain2 = LLMProvider(bounded).get_chain("reasoning")
    assert [m.max_tokens for m in chain2] == [16000, 6000]


def test_reasoning_override_still_gets_the_stage_budget():
    cfg = _budget_config()
    chain = LLMProvider(cfg).get_chain("reasoning", override="test-reason-fallback")
    assert len(chain) == 1
    assert chain[0].max_tokens == 16000


def test_default_output_token_budget_is_16000_with_the_measured_rationale(tmp_path):
    """F2 — the budget as originally set (8000) CREATED the truncation it was handling.

    Measured: the largest real GraphDelta bodies are ~22,700 chars ≈ 5.5-6.5k content tokens,
    and this model demonstrably emits reasoning tokens on top (observed 1,256) ≈ 7.8k total —
    so an 8000 TOTAL bound clipped real, successful outputs. 16000 is ~2x headroom and still
    bounds runaway. Asserted in the dataclass and its serialized TOML.
    """
    import tomllib
    cfg = BrainConfig()
    assert cfg.reasoning.output_token_budget == 16000

    toml_path = tmp_path / "default.toml"
    save_config(cfg, toml_path)
    with toml_path.open("rb") as fh:
        shipped = tomllib.load(fh)
    assert shipped["reasoning"]["output_token_budget"] == 16000
    # Scope guard: F2 changed exactly ONE default, nothing else in [reasoning].
    assert shipped["reasoning"]["context_token_budget"] == 8000
    assert shipped["reasoning"]["chunk_token_budget"] == 6000
    assert shipped["reasoning"]["max_graph_neighbors"] == 6
    assert shipped["reasoning"]["graph_hop_depth"] == 2


def test_extraction_and_matching_stages_untouched():
    """Deliberate scope limit: only the reasoning stage has a wired budget."""
    cfg = _budget_config()
    provider = LLMProvider(cfg)
    assert provider.get_chain("extraction")[0].max_tokens is None
    assert provider.get_chain("matching")[0].max_tokens is None
    assert provider.get_for_stage("extraction").max_tokens is None


def test_zero_budget_is_treated_as_unset():
    cfg = _budget_config()
    cfg.reasoning.output_token_budget = 0
    assert LLMProvider(cfg).get_chain("reasoning")[0].max_tokens is None


# ── 2. finish_reason detection ───────────────────────────────────────


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("length", True),           # openai / openrouter / oMLX
        ("max_tokens", True),       # anthropic stop_reason
        ("MAX_TOKENS", True),       # google
        ("max-tokens", True),
        ("stop", False),
        ("end_turn", False),
        ("error", False),           # the observed reasoning-model quirk — NOT truncation
        ("", False),
        (None, False),
    ],
)
def test_is_truncation_reason_spellings(reason, expected):
    assert is_truncation_reason(reason) is expected


def test_detect_finish_reason_from_response_metadata():
    assert detect_finish_reason(_msg("partial", "length")) == "length"
    assert detect_finish_reason(_msg("done", "stop")) == "stop"


def test_detect_finish_reason_from_additional_kwargs():
    msg = AIMessage(content="x")
    msg.additional_kwargs = {"stop_reason": "max_tokens"}
    assert detect_finish_reason(msg) == "max_tokens"
    assert is_truncation_reason(detect_finish_reason(msg)) is True


def test_detect_finish_reason_from_generation_info():
    gen = ChatGeneration(
        message=AIMessage(content="partial"),
        generation_info={"finish_reason": "length", "logprobs": None},
    )
    assert detect_finish_reason(gen) == "length"


def test_detect_finish_reason_from_llm_result_generations():
    """`.agenerate()` returns an LLMResult whose reason hides in the nested generations."""
    result = LLMResult(
        generations=[[
            ChatGeneration(
                message=AIMessage(content="partial"),
                generation_info={"finish_reason": "length"},
            )
        ]]
    )
    assert detect_finish_reason(result) == "length"


def test_detect_finish_reason_absent_is_none():
    assert detect_finish_reason(None) is None
    assert detect_finish_reason("a bare string") is None
    assert detect_finish_reason(AIMessage(content="no metadata")) is None
    assert detect_finish_reason(SimpleNamespace(response_metadata=None)) is None
    # A structured-output RunnableSequence returns the parsed model, not a message.
    assert detect_finish_reason(GraphDelta()) is None
    assert is_truncation_reason(detect_finish_reason(AIMessage(content="x"))) is False


# ── 3. Truncation retry ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_truncation_escalates_once_on_same_model():
    model = FakeChatModel(
        [_msg(_TRUNCATED_JSON, "length"), _msg(_GOOD_DELTA_JSON, "stop")],
        max_tokens=8000,
    )
    config = ResilienceConfig(request_timeout_s=0, circuit_breaker_enabled=False)

    result = await invoke_with_resilience(
        [model], ["msg"], config, truncation_retry=True,
    )

    # (a) the partial output is never returned to the parser
    assert result.content == _GOOD_DELTA_JSON
    # (b) exactly one escalated retry, same model, 1.5x the bound
    assert len(model.calls) == 2
    assert model.calls[0] == {}
    assert model.calls[1] == {"max_tokens": 12000}


@pytest.mark.asyncio
async def test_truncation_does_not_loop_and_falls_through_to_fallback():
    """Still truncated at 1.5x -> stop escalating, hand over to the next model (cascade)."""
    primary = FakeChatModel(
        [_msg(_TRUNCATED_JSON, "length")], max_tokens=8000, model_name="primary",
    )
    fallback = FakeChatModel([_msg(_GOOD_DELTA_JSON, "stop")], model_name="fallback")
    config = ResilienceConfig(request_timeout_s=0, circuit_breaker_enabled=False)

    result = await invoke_with_resilience(
        [primary, fallback], ["msg"], config, truncation_retry=True,
    )

    assert result.content == _GOOD_DELTA_JSON
    assert len(primary.calls) == 2  # original + ONE escalation, no unbounded loop
    assert primary.calls[1] == {"max_tokens": 12000}
    assert len(fallback.calls) == 1


@pytest.mark.asyncio
async def test_persistent_truncation_exhausts_chain_with_truncation_error():
    model = FakeChatModel([_msg(_TRUNCATED_JSON, "length")], max_tokens=8000)
    config = ResilienceConfig(
        request_timeout_s=0, max_retries=3, circuit_breaker_enabled=False,
    )

    with pytest.raises(FallbackExhaustedError) as exc_info:
        await invoke_with_resilience([model], ["msg"], config, truncation_retry=True)

    # A truncation is deterministic: no backoff retries, just original + one escalation.
    assert len(model.calls) == 2
    assert "TruncatedOutputError" in str(exc_info.value)


@pytest.mark.asyncio
async def test_truncation_without_known_bound_skips_escalation():
    """No max_tokens anywhere = nothing to escalate FROM; discard and move on, never guess."""
    primary = FakeChatModel([_msg(_TRUNCATED_JSON, "length")], model_name="primary")
    fallback = FakeChatModel([_msg(_GOOD_DELTA_JSON, "stop")], model_name="fallback")
    config = ResilienceConfig(request_timeout_s=0, circuit_breaker_enabled=False)

    result = await invoke_with_resilience(
        [primary, fallback], ["msg"], config, truncation_retry=True,
    )

    assert result.content == _GOOD_DELTA_JSON
    assert len(primary.calls) == 1


@pytest.mark.asyncio
async def test_normal_finish_reason_is_a_no_op():
    model = FakeChatModel([_msg(_GOOD_DELTA_JSON, "stop")], max_tokens=8000)
    config = ResilienceConfig(request_timeout_s=0, circuit_breaker_enabled=False)

    result = await invoke_with_resilience(
        [model], ["msg"], config, truncation_retry=True,
    )

    assert result.content == _GOOD_DELTA_JSON
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_absent_finish_reason_is_a_no_op():
    model = FakeChatModel([_msg(_GOOD_DELTA_JSON, None)], max_tokens=8000)
    config = ResilienceConfig(request_timeout_s=0, circuit_breaker_enabled=False)

    result = await invoke_with_resilience(
        [model], ["msg"], config, truncation_retry=True,
    )

    assert result.content == _GOOD_DELTA_JSON
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_truncation_branch_is_off_by_default():
    """Byte-identical legacy behavior: without the opt-in flag a truncated response is
    returned exactly as before (this is what the 2026-07 batch did)."""
    model = FakeChatModel([_msg(_TRUNCATED_JSON, "length")], max_tokens=8000)
    config = ResilienceConfig(request_timeout_s=0, circuit_breaker_enabled=False)

    result = await invoke_with_resilience([model], ["msg"], config)

    assert result.content == _TRUNCATED_JSON
    assert len(model.calls) == 1


def test_truncated_output_error_message_carries_the_facts():
    exc = TruncatedOutputError("deepseek-v4-flash", "length", max_tokens=12000, escalated=True)
    assert exc.model_name == "deepseek-v4-flash"
    assert exc.finish_reason == "length"
    assert exc.max_tokens == 12000
    assert exc.escalated is True
    assert "truncated output" in str(exc)


# ── 4. Parse-failure diagnostics ─────────────────────────────────────


def test_extract_json_error_prefix_is_preserved():
    """Log-greps and existing tests depend on this exact prefix — it must not drift."""
    with pytest.raises(ValueError) as exc_info:
        extract_json(_PROSE)
    assert str(exc_info.value).startswith("Could not extract JSON from LLM output (")
    assert str(exc_info.value) == (
        f"Could not extract JSON from LLM output ({len(_PROSE)} chars)"
    )


def test_diagnostics_distinguish_truncated_from_prose():
    truncated = describe_parse_failure(_TRUNCATED_JSON, finish_reason="length")
    prose = describe_parse_failure(_PROSE, finish_reason="stop")

    # Truncated: an opening brace exists, nothing closes it, provider says 'length'.
    assert "has_open_brace=True" in truncated
    assert "ends_with_close_brace=False" in truncated
    assert "finish_reason='length'" in truncated

    # Prose: no JSON object at all, ends on a sentence, provider says it finished normally.
    assert "has_open_brace=False" in prose
    assert "ends_with_close_brace=False" in prose
    assert "finish_reason='stop'" in prose
    assert prose.endswith("at this time.'")

    assert truncated != prose


def test_diagnostics_report_length_and_cap_the_tail_at_200():
    body = "x" * 5000 + "TAIL_MARKER"
    diag = describe_parse_failure(body)

    assert f"len={len(body)}" in diag
    assert "TAIL_MARKER" in diag
    tail = diag.split("tail_200=", 1)[1]
    # repr-quoted, so 200 chars + the two quotes; newlines can never break the log line.
    assert len(tail) == 202
    assert "x" * 5000 not in diag


def test_diagnostics_handle_empty_and_missing_bodies():
    """4 of the 24 measured incidents were under 2k chars, including 0-char empties."""
    empty = describe_parse_failure("")
    assert "len=0" in empty
    assert "has_open_brace=False" in empty
    assert "tail_200=''" in empty
    assert describe_parse_failure(None) == "parse_failure: response=None"


# ── 5. Pipeline call site (integration of 1 + 3) ─────────────────────


def _stub_pipeline(config: BrainConfig | None = None) -> KnowledgePipeline:
    """A pipeline with only what `_resilient_reason` touches — no Neo4j, no provider.

    Mirrors tests/test_pipeline_match_stage.py's stubbed-pipeline pattern; `_gather_context`
    and `_build_prompt` return the same types the real GraphReasoner does (both str).
    """
    pipeline = KnowledgePipeline.__new__(KnowledgePipeline)
    cfg = config or BrainConfig()
    cfg.resilience = ResilienceConfig(request_timeout_s=0, circuit_breaker_enabled=False)
    pipeline._config = cfg
    pipeline._auth_refresh = None

    async def _gather_context(match_result):
        return "(No existing graph context)"

    pipeline._reasoner = SimpleNamespace(
        _gather_context=_gather_context,
        _build_prompt=lambda extraction, match_result, context, chunk_text: "PROMPT",
    )
    return pipeline


@pytest.mark.asyncio
async def test_pipeline_logs_parse_failure_diagnostics(caplog):
    """A prose body must produce classifiable detail — the measurement that was impossible."""
    pipeline = _stub_pipeline()
    model = FakeChatModel([_msg(_PROSE, "stop")])

    with caplog.at_level("WARNING"):
        with pytest.raises(ReasoningDegradedError) as exc_info:
            await pipeline._resilient_reason(
                ExtractionResult(chunk_id="chunk_0"), MatchResult(chunk_id="chunk_0"), "body text", [model],
            )

    logged = caplog.text
    assert "Could not extract JSON from LLM output" in logged
    assert "parse_failure:" in logged
    assert f"len={len(_PROSE)}" in logged
    assert "has_open_brace=False" in logged
    assert "finish_reason='stop'" in logged
    # The diagnostic also reaches IngestResult.errors / the checkpoint via the cause.
    assert "parse_failure:" in exc_info.value.cause
    assert exc_info.value.chunk_id == "chunk_0"


@pytest.mark.asyncio
async def test_pipeline_diagnostics_flag_a_truncated_body(caplog):
    pipeline = _stub_pipeline()
    # finish_reason absent (the shape that made the 2026-07 logs ambiguous): the tail alone
    # still classifies it — an open brace with nothing closing it.
    model = FakeChatModel([_msg(_TRUNCATED_JSON, None)])

    with caplog.at_level("WARNING"):
        with pytest.raises(ReasoningDegradedError):
            await pipeline._resilient_reason(
                ExtractionResult(chunk_id="chunk_1"), MatchResult(chunk_id="chunk_0"), "body text", [model],
            )

    assert "has_open_brace=True" in caplog.text
    assert "ends_with_close_brace=False" in caplog.text
    assert "finish_reason=None" in caplog.text


@pytest.mark.asyncio
async def test_pipeline_never_parses_a_truncated_partial(monkeypatch):
    """The reasoning call site opts into the truncation branch: the partial GraphDelta is
    discarded and the escalated retry's output is what reaches extract_json."""
    seen: list[str] = []
    real_extract_json = normalize_module.extract_json

    def spy(text: str):
        seen.append(text)
        return real_extract_json(text)

    monkeypatch.setattr(normalize_module, "extract_json", spy)

    pipeline = _stub_pipeline()
    model = FakeChatModel(
        [_msg(_TRUNCATED_JSON, "length"), _msg(_GOOD_DELTA_JSON, "stop")],
        max_tokens=8000,
    )

    delta = await pipeline._resilient_reason(
        ExtractionResult(chunk_id="chunk_2"), MatchResult(chunk_id="chunk_2"), "body text",
        [model],
    )

    assert isinstance(delta, GraphDelta)
    assert [n.canonical_name for n in delta.new_nodes] == ["kTC Noise"]
    assert seen == [_GOOD_DELTA_JSON]           # the partial NEVER reached the parser
    assert model.calls[1] == {"max_tokens": 12000}


# ── 6. Adversarial-review remedies (F1, F3-F8) ───────────────────────
#
# Nine findings from the 2026-07-25 review of the change above; the eight with a testable
# surface are pinned here. (F2 — the 8000 -> 16000 default — is pinned in section 1;
# F9's blast-radius comments at the three sibling reasoning call sites are prose, not
# behavior, except for answer_from_context's opt-in, covered in test_answerer.py's chain.)


class _Rate429(Exception):
    """A provider rate-limit error, classified RetryableError by `classify_error`."""
    status_code = 429


_FAST_BREAKER = dict(
    request_timeout_s=0,
    initial_backoff_s=0.0,
    max_backoff_s=0.0,
    jitter=False,
)


# ── F1: truncation is a CONTENT failure, not an AVAILABILITY failure ──


@pytest.mark.asyncio
async def test_persistent_truncation_never_trips_the_circuit_breaker():
    """F1 — the production hazard: a batch of long chunks all truncate, and the breaker
    (an AVAILABILITY instrument) evicts the production reasoning model from the chain for the
    whole cooldown over a bound that is ours to set. The call SUCCEEDED; only its content was
    unusable. Nothing may be recorded against the model."""
    model = FakeChatModel(
        [_msg(_TRUNCATED_JSON, "length")], max_tokens=8000, model_name="reasoner",
    )
    config = ResilienceConfig(
        max_retries=0, circuit_breaker_enabled=True, circuit_breaker_threshold=3,
        **_FAST_BREAKER,
    )

    for _ in range(2 * config.circuit_breaker_threshold):
        with pytest.raises(FallbackExhaustedError):
            await invoke_with_resilience([model], ["msg"], config, truncation_retry=True)

    assert _circuit_breakers == {}                              # nothing recorded at all
    assert _breaker_is_open("reasoner", config) is False        # breaker stays CLOSED


@pytest.mark.asyncio
async def test_model_stays_in_the_chain_after_repeated_truncations():
    """F1, the consequence that matters: the model is still ATTEMPTED afterwards — the very
    next call that fits the bound succeeds instead of being skipped by an open breaker."""
    model = FakeChatModel(
        [_msg(_TRUNCATED_JSON, "length")], max_tokens=8000, model_name="reasoner",
    )
    config = ResilienceConfig(
        max_retries=0, circuit_breaker_enabled=True, circuit_breaker_threshold=3,
        **_FAST_BREAKER,
    )

    for _ in range(4):
        with pytest.raises(FallbackExhaustedError):
            await invoke_with_resilience([model], ["msg"], config, truncation_retry=True)
    calls_before = len(model.calls)

    model._responses = [_msg(_GOOD_DELTA_JSON, "stop")]
    result = await invoke_with_resilience([model], ["msg"], config, truncation_retry=True)

    assert result.content == _GOOD_DELTA_JSON
    assert len(model.calls) == calls_before + 1                 # it really was called


@pytest.mark.asyncio
async def test_availability_failures_still_trip_the_breaker():
    """F1 is a targeted exemption, not a blanket removal: a real availability failure on the
    same model still opens the breaker exactly as before."""
    model = FakeChatModel([ValueError("upstream is on fire")], model_name="reasoner")
    config = ResilienceConfig(
        max_retries=0, circuit_breaker_enabled=True, circuit_breaker_threshold=3,
        **_FAST_BREAKER,
    )

    for _ in range(config.circuit_breaker_threshold):
        with pytest.raises(FallbackExhaustedError):
            await invoke_with_resilience([model], ["msg"], config, truncation_retry=True)

    assert _breaker_is_open("reasoner", config) is True


# ── F4: the escalated retry must not be defeated by the caller's kwargs ──


@pytest.mark.asyncio
async def test_escalated_retry_ignores_the_callers_max_tokens():
    """F4 — `RunnableBinding.ainvoke` merges `{**self.kwargs, **kwargs}`: CALL kwargs override
    BOUND kwargs. Forwarding the caller's max_tokens re-issued the byte-identical request, so
    the "escalated" retry was a guaranteed-failing paid call."""
    model = FakeChatModel(
        [_msg(_TRUNCATED_JSON, "length"), _msg(_GOOD_DELTA_JSON, "stop")], max_tokens=4000,
    )
    config = ResilienceConfig(request_timeout_s=0, circuit_breaker_enabled=False)

    result = await invoke_with_resilience(
        [model], ["msg"], config, truncation_retry=True, max_tokens=8000, temperature=0.0,
    )

    assert result.content == _GOOD_DELTA_JSON
    # The caller's 8000 is the base (it outranks the instance's 4000)...
    assert model.calls[0] == {"max_tokens": 8000, "temperature": 0.0}
    # ...and the retry actually goes out at the ESCALATED value, other kwargs preserved.
    assert model.calls[1] == {"max_tokens": 12000, "temperature": 0.0}


@pytest.mark.asyncio
async def test_escalation_strips_max_tokens_only_for_that_one_call():
    """F4 — the strip is on a COPY: every other path (here, the next model in the chain) still
    gets the caller's own bound."""
    primary = FakeChatModel(
        [_msg(_TRUNCATED_JSON, "length")], max_tokens=8000, model_name="primary",
    )
    fallback = FakeChatModel([_msg(_GOOD_DELTA_JSON, "stop")], model_name="fallback")
    config = ResilienceConfig(request_timeout_s=0, circuit_breaker_enabled=False)

    result = await invoke_with_resilience(
        [primary, fallback], ["msg"], config, truncation_retry=True, max_tokens=8000,
    )

    assert result.content == _GOOD_DELTA_JSON
    assert primary.calls == [{"max_tokens": 8000}, {"max_tokens": 12000}]
    assert fallback.calls == [{"max_tokens": 8000}]


# ── F5: one escalation per invocation, not per retry attempt ─────────


@pytest.mark.asyncio
async def test_escalation_happens_at_most_once_per_invocation():
    """F5 — a retryable error on the escalated call sends the OUTER attempt loop around again;
    the escalation used to be re-entered once per attempt (measured: 8 calls at max_retries=3).
    The state is now threaded per model, so attempt 2's truncation goes straight to the chain."""
    model = FakeChatModel(
        [
            _msg(_TRUNCATED_JSON, "length"),   # call 1 — original, truncated
            _Rate429("429 rate limited"),      # call 2 — the ONE escalation, retryable failure
            _msg(_TRUNCATED_JSON, "length"),   # call 3 — attempt 2, truncated again
        ],
        max_tokens=8000,
    )
    config = ResilienceConfig(
        max_retries=3, circuit_breaker_enabled=False, **_FAST_BREAKER,
    )

    with pytest.raises(FallbackExhaustedError):
        await invoke_with_resilience([model], ["msg"], config, truncation_retry=True)

    assert model.calls == [{}, {"max_tokens": 12000}, {}]
    assert len(model.calls) == 3            # was 8 before the fix


@pytest.mark.asyncio
async def test_each_model_in_the_chain_gets_its_own_single_escalation():
    """F5's scope: once-per-model, not once-per-chain — a fallback is not punished for the
    primary having spent its escalation."""
    primary = FakeChatModel(
        [_msg(_TRUNCATED_JSON, "length")], max_tokens=8000, model_name="primary",
    )
    fallback = FakeChatModel(
        [_msg(_TRUNCATED_JSON, "length"), _msg(_GOOD_DELTA_JSON, "stop")],
        max_tokens=2000, model_name="fallback",
    )
    config = ResilienceConfig(request_timeout_s=0, circuit_breaker_enabled=False)

    result = await invoke_with_resilience(
        [primary, fallback], ["msg"], config, truncation_retry=True,
    )

    assert result.content == _GOOD_DELTA_JSON
    assert primary.calls == [{}, {"max_tokens": 12000}]
    assert fallback.calls == [{}, {"max_tokens": 3000}]


# ── F6: knowable bounds on non-openai models ─────────────────────────


@pytest.mark.asyncio
async def test_google_style_max_output_tokens_is_a_knowable_bound():
    """F6 — ChatGoogleGenerativeAI stores the bound as `max_output_tokens`, so reading only
    `max_tokens` made every google fallback look unbounded and silently skip escalation."""
    model = FakeChatModel(
        [_msg(_TRUNCATED_JSON, "MAX_TOKENS"), _msg(_GOOD_DELTA_JSON, "STOP")],
        max_tokens=8000, bound_attr="max_output_tokens", model_name="gemini-3.1-flash-lite",
    )
    config = ResilienceConfig(request_timeout_s=0, circuit_breaker_enabled=False)

    result = await invoke_with_resilience(
        [model], ["msg"], config, truncation_retry=True,
    )

    assert result.content == _GOOD_DELTA_JSON
    # Escalated under the name the model will actually read back at call time.
    assert model.calls == [{}, {"max_output_tokens": 12000}]


@pytest.mark.asyncio
async def test_no_bound_under_either_spelling_still_skips_escalation():
    """F6 keeps the "unknown bound -> never guess" behavior when NEITHER name is present."""
    model = FakeChatModel([_msg(_TRUNCATED_JSON, "length")], model_name="unbounded")
    fallback = FakeChatModel([_msg(_GOOD_DELTA_JSON, "stop")], model_name="fallback")
    config = ResilienceConfig(request_timeout_s=0, circuit_breaker_enabled=False)

    result = await invoke_with_resilience(
        [model, fallback], ["msg"], config, truncation_retry=True,
    )

    assert result.content == _GOOD_DELTA_JSON
    assert len(model.calls) == 1


# ── F7 / F8: the diagnostics must actually discriminate, and stay comparable ──


def test_diagnostics_see_through_markdown_fences():
    """F7 — a COMPLETE ```json … ``` body ends with a backtick, so `ends_with_close_brace` read
    False: byte-identical booleans to a genuinely truncated body, i.e. they classified
    nothing. Fences are stripped before the structural booleans are computed."""
    complete = describe_parse_failure(f"```json\n{_GOOD_DELTA_JSON}\n```", finish_reason="stop")
    truncated = describe_parse_failure(f"```json\n{_TRUNCATED_JSON}", finish_reason="length")
    prose = describe_parse_failure(_PROSE, finish_reason="stop")

    assert "has_open_brace=True" in complete
    assert "ends_with_close_brace=True" in complete       # was False before the fix

    assert "has_open_brace=True" in truncated
    assert "ends_with_close_brace=False" in truncated

    assert "has_open_brace=False" in prose
    assert "ends_with_close_brace=False" in prose

    # The three shapes are now actually distinguishable from the booleans alone.
    def _booleans(diag: str) -> tuple[str, ...]:
        return tuple(t for t in diag.split() if t.startswith(("has_open", "ends_with")))

    assert _booleans(complete) != _booleans(truncated) != _booleans(prose)

    # And the tail is still the RAW tail, fences included, capped at 200.
    assert complete.rstrip("'").endswith("```")


def test_diagnostics_keep_the_200_char_tail_after_the_fence_fix():
    import ast

    body = f"```json\n{'x' * 5000}TAIL_MARKER\n```"
    diag = describe_parse_failure(body)

    tail = ast.literal_eval(diag.split("tail_200=", 1)[1])   # repr'd, so newlines stay escaped
    assert len(tail) == 200
    assert "TAIL_MARKER" in tail
    assert "x" * 5000 not in diag


def test_diagnostics_report_raw_length_not_stripped():
    """F8 — `len=` must stay comparable to the 2026-07 series (p50 10,839 / max 22,700), which
    was measured on the body as received. Stripped length rides along only when it differs."""
    padded = f"\n\n  {_PROSE}  \n\n"
    diag = describe_parse_failure(padded)

    assert diag.startswith(f"parse_failure: len={len(padded)} ")
    assert f"stripped_len={len(_PROSE)}" in diag

    # No whitespace to strip -> no redundant field.
    assert describe_parse_failure(_PROSE).startswith(f"parse_failure: len={len(_PROSE)} ")
    assert "stripped_len" not in describe_parse_failure(_PROSE)


# ── F3: telemetry must never break the pipeline ──────────────────────


@pytest.mark.parametrize(
    "payload",
    [
        [{"type": "text", "text": "hi"}],                      # LangChain multi-part content
        [{"type": "thinking", "thinking": "…"}, "tail"],       # block with no "text" key
        ["a", "b"],                                            # list[str] content
        {"unexpected": "dict"},
        12345,
        object(),
        b"bytes",
    ],
)
def test_describe_parse_failure_never_raises_for_any_input_type(payload):
    """F3(a) — `text.strip()` raised AttributeError on a multi-part content list, INSIDE the
    except block whose whole job is graceful degradation."""
    diag = describe_parse_failure(payload, finish_reason="stop")
    assert isinstance(diag, str)
    assert diag.startswith("parse_failure:")


def test_describe_parse_failure_flattens_multipart_text():
    diag = describe_parse_failure(
        [{"type": "text", "text": "{\"new_nodes\": ["}], finish_reason="length",
    )
    assert "has_open_brace=True" in diag
    assert "ends_with_close_brace=False" in diag
    assert "finish_reason='length'" in diag


@pytest.mark.asyncio
async def test_multipart_content_does_not_crash_resilient_reason(caplog):
    """F3 end-to-end — `raw_text = response.content` is `str | list[str|dict]`, so a multi-part
    response used to convert a ReasoningDegradedError into an unhandled AttributeError that
    took the whole ingest down. The normal degradation path must survive it."""
    pipeline = _stub_pipeline()
    model = FakeChatModel([AIMessage(content=[{"type": "text", "text": "hi"}])])

    with caplog.at_level("WARNING"):
        with pytest.raises(ReasoningDegradedError) as exc_info:
            await pipeline._resilient_reason(
                ExtractionResult(chunk_id="chunk_mp"), MatchResult(chunk_id="chunk_mp"),
                "body text", [model],
            )

    assert exc_info.value.chunk_id == "chunk_mp"
    assert "parse_failure:" in exc_info.value.cause
    assert "tail_200='hi'" in exc_info.value.cause
    assert "parse_failure:" in caplog.text


@pytest.mark.asyncio
async def test_diagnostics_failure_can_never_escape_the_call_site(monkeypatch):
    """F3(b) — second layer: even a describe_parse_failure that blows up outright must leave
    the ReasoningDegradedError (and its original cause) intact."""
    def _explode(*args, **kwargs):
        raise RuntimeError("telemetry is broken")

    # `_resilient_reason` imports it from the normalize module at call time.
    monkeypatch.setattr(normalize_module, "describe_parse_failure", _explode)

    pipeline = _stub_pipeline()
    model = FakeChatModel([_msg(_PROSE, "stop")])

    with pytest.raises(ReasoningDegradedError) as exc_info:
        await pipeline._resilient_reason(
            ExtractionResult(chunk_id="chunk_x"), MatchResult(chunk_id="chunk_x"),
            "body text", [model],
        )

    assert "Could not extract JSON from LLM output" in exc_info.value.cause
    assert "diagnostics raised RuntimeError" in exc_info.value.cause
