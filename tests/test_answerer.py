"""Offline tests for the lean grounded answer layer."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import openclaw_brain.agent as agent_module
import openclaw_brain.knowledge.reasoning.answerer as answerer_module
from openclaw_brain.agent import BrainAgent
from openclaw_brain.config import ResilienceConfig
from openclaw_brain.knowledge.reasoning.answerer import (
    _ANSWER_SYSTEM_PROMPT,
    _MODEL_KNOWLEDGE_PROMPT,
    AnswerResult,
    answer_from_context,
)
from openclaw_brain.llm.resilience import reset_circuit_breakers


@pytest.fixture(autouse=True)
def _reset_breakers():
    """Circuit breaker state is module-level (keyed by model_name) — isolate every test. Matters
    here (W-D2 defect 9) because answer_from_context's structured-primary attempt now routes
    through invoke_with_resilience for real in several of these tests, not just the raw fallback."""
    reset_circuit_breakers()
    yield
    reset_circuit_breakers()


class _FakeStructuredResponder:
    def __init__(self, parent):
        self._parent = parent

    async def ainvoke(self, messages):
        self._parent.messages = messages
        return self._parent.result


class _FakePrimaryModel:
    def __init__(
        self, result=None, *, structured_error: Exception | None = None,
        model_name: str = "primary-fake",
    ):
        self.result = result
        self.structured_error = structured_error
        self.model_name = model_name   # resolve_model_name() reads this off the RAW model
        self.schema = None
        self.messages = None
        self.structured_calls = 0

    def with_structured_output(self, schema):
        self.structured_calls += 1
        self.schema = schema
        if self.structured_error:
            raise self.structured_error
        return _FakeStructuredResponder(self)


def _resilience_config():
    # A real ResilienceConfig (not a bare SimpleNamespace): the structured-primary attempt now
    # goes through invoke_with_resilience for real in some of these tests, which reads
    # circuit_breaker_enabled/max_retries/etc — mock fidelity per CLAUDE.md Testing Conventions.
    return ResilienceConfig(request_timeout_s=0)


def _context_kwargs(model_chain):
    return {
        "question": "What reduces comparator kickback?",
        "context_markdown": "Concept c1 cites chunk_1: isolation reduces kickback.",
        "concept_refs": [
            {
                "id": "c1",
                "name": "Kickback isolation",
                "confidence": 0.9,
                "layer": "L1",
                "domain": "analog",
                "cite": {"level": "chunk", "chunks": ["chunk_1"]},
            }
        ],
        "open_hypotheses": [],
        "active_decisions": [],
        "model_chain": model_chain,
        "resilience_config": _resilience_config(),
    }


@pytest.mark.asyncio
async def test_answer_from_context_structured_output_happy_path():
    expected = AnswerResult(
        answer="Use isolation around the sensitive node.",
        citations=["c1", "chunk_1"],
        abstained=False,
        used_concepts=["c1"],
    )
    primary = _FakePrimaryModel(expected)

    result = await answer_from_context(**_context_kwargs([primary]))

    assert result == expected
    assert primary.schema is AnswerResult
    assert primary.structured_calls == 1
    assert primary.messages[0].content == _ANSWER_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_answer_from_context_raw_json_fallback_accepts_a3_alias(monkeypatch):
    primary = _FakePrimaryModel(structured_error=RuntimeError("schema unsupported"))

    async def fake_invoke_with_resilience(
        models, messages, config, *, auth_refresh=None, truncation_retry=False, **kwargs,
    ):
        assert models == [primary]
        assert auth_refresh is None
        # F9(a), 2026-07-25: this chain resolves stage='reasoning', so it carries the stage's
        # output-token bound. A silently truncated ANSWER is the worst of the three inheriting
        # surfaces — it is what the citation audits run on — so the raw path opts in.
        assert truncation_retry is True
        return SimpleNamespace(
            content=json.dumps(
                {
                    "prose_answer": "Use source degeneration.",
                    "citations": ["c2", "chunk_2"],
                    "used_concepts": ["c2"],
                    "abstained": False,
                    "abstain_reason": "",
                }
            )
        )

    monkeypatch.setattr(
        answerer_module,
        "invoke_with_resilience",
        fake_invoke_with_resilience,
    )

    result = await answer_from_context(**_context_kwargs([primary]))

    assert result.answer == "Use source degeneration."
    assert result.citations == ["c2", "chunk_2"]
    assert result.used_concepts == ["c2"]
    assert result.abstained is False


@pytest.mark.asyncio
async def test_answer_from_context_total_failure_abstains(monkeypatch):
    primary = _FakePrimaryModel(structured_error=RuntimeError("schema unsupported"))

    async def fake_invoke_with_resilience(
        models, messages, config, *, auth_refresh=None, truncation_retry=False, **kwargs,
    ):
        return SimpleNamespace(content="not json")

    monkeypatch.setattr(
        answerer_module,
        "invoke_with_resilience",
        fake_invoke_with_resilience,
    )

    result = await answer_from_context(**_context_kwargs([primary]))

    assert result.abstained is True
    assert result.abstain_reason == "synthesis_failed"
    assert result.answer == ""


@pytest.mark.asyncio
async def test_answer_from_context_structured_attempt_skips_when_breaker_open():
    """W-D2 defect 9 regression: answer_from_context's structured-output primary attempt used to
    call .ainvoke() directly, bypassing invoke_with_resilience (and therefore the circuit breaker)
    entirely — a model whose breaker was already OPEN (tripped by OTHER callers sharing the
    process-global breaker, llm/README.md) still got a full, wasted attempt on every call. Fixed:
    the structured attempt now goes through invoke_with_resilience with model_names computed from
    the RAW primary, so it respects an already-open breaker and skips straight to the raw-chain
    fallback instead of paying the cost.

    Differentiates old vs new: under the OLD code, `structured.ainvoke()` was called
    unconditionally regardless of breaker state, which — via _FakeStructuredResponder — sets
    `primary.messages` and returns the primary's own fake result directly (not abstained). Under
    the fix, the breaker-open skip happens before any attempt, so `primary.messages` stays None
    and the (also breaker-blocked) raw-chain fallback is left to abstain.
    """
    from openclaw_brain.llm import resilience as resilience_module

    primary = _FakePrimaryModel(AnswerResult(answer="should never be reached"))
    config = _resilience_config()
    for _ in range(config.circuit_breaker_threshold):
        resilience_module._breaker_record_failure("primary-fake", config)
    assert resilience_module._breaker_is_open("primary-fake", config)

    result = await answer_from_context(**_context_kwargs([primary]))

    assert primary.messages is None   # the structured wrapper's ainvoke() was never reached
    assert primary.structured_calls == 1   # with_structured_output() itself is still cheap/called
    assert result.abstained is True
    assert result.abstain_reason == "synthesis_failed"


@pytest.mark.asyncio
async def test_brain_agent_answer_question_no_context_short_circuits(monkeypatch):
    agent = BrainAgent.__new__(BrainAgent)
    agent._started = True
    agent.query_knowledge = AsyncMock(
        return_value={
            "formatted": "",
            "concepts_found": 0,
            "memories_found": 0,
            "concepts": [],
            "neighbors": [],
            "concept_refs": [],
            "open_hypotheses": [],
            "active_decisions": [],
        }
    )
    synth = AsyncMock(side_effect=AssertionError("model should not be called"))
    monkeypatch.setattr(agent_module, "answer_from_context", synth)

    result = await agent.answer_question("What is gm?", on_insufficient="flag")

    assert {key: value for key, value in result.items() if key != "envelope"} == {
        "answer": "",
        "citations": [],
        "abstained": True,
        "abstain_reason": "no_context",
        "used_concepts": [],
        "concepts_found": 0,
        "concepts": [],
    }
    assert result["envelope"]["status"] == "abstained"
    assert result["envelope"]["reason_codes"] == ["no_context"]
    agent.query_knowledge.assert_awaited_once_with("What is gm?", session_id=None)
    synth.assert_not_awaited()


def test_answer_prompt_contains_cite_or_label_abstention_contract():
    assert "Every factual claim MUST be backed by a citation" in _ANSWER_SYSTEM_PROMPT
    assert "does not contain enough to answer faithfully" in _ANSWER_SYSTEM_PROMPT
    assert "`abstained=true`" in _ANSWER_SYSTEM_PROMPT
    assert "derived/unverified" in _ANSWER_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_no_context_model_knowledge_uses_one_raw_model_call():
    class RawModel:
        model_name = "raw-answer-fake"

        def __init__(self):
            self.calls = 0
            self.messages = None

        async def ainvoke(self, messages):
            self.calls += 1
            self.messages = messages
            return SimpleNamespace(content=json.dumps({
                "answer": "A general answer", "citations": [], "kb_coverage": "none",
                "missing_knowledge": ["missing explanation"],
            }))

    model = RawModel()
    kwargs = _context_kwargs([model])
    kwargs["concept_refs"] = []
    kwargs["context_markdown"] = ""
    result = await answer_from_context(
        **kwargs, on_insufficient="model_knowledge", no_context=True,
    )
    assert model.calls == 1
    assert model.messages[0].content == _MODEL_KNOWLEDGE_PROMPT
    assert result.kb_coverage == "none"
    assert result.missing_knowledge == ["missing explanation"]


@pytest.mark.asyncio
async def test_no_context_model_failure_does_not_retry():
    class FailingModel:
        model_name = "failing-answer-fake"

        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            raise RuntimeError("offline failure")

    model = FailingModel()
    kwargs = _context_kwargs([model])
    kwargs["concept_refs"] = []
    kwargs["context_markdown"] = ""
    result = await answer_from_context(
        **kwargs, on_insufficient="model_knowledge", no_context=True,
    )
    assert model.calls == 1
    assert result.abstain_reason == "synthesis_failed"


def test_missing_knowledge_is_capped_at_five():
    result = AnswerResult(missing_knowledge=[f"gap {i}" for i in range(8)])
    assert len(result.missing_knowledge) == 5
