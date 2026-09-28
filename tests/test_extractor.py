"""Tests for the two-pass extractor's structured-output schemas (extraction/extractor.py).

Focus: `_RelationExtractionOutput`'s tolerance for claude-sonnet-5's double-encoding quirk —
the `relationships` field (or the whole payload) comes back as a JSON *string* instead of a
native list, which used to make pydantic reject the whole relation-extraction pass ("relationships
should be a valid list but got a string") and silently drop every relation the model emitted.
"""

from __future__ import annotations

import json
import logging

import pytest
from pydantic import ValidationError

from openclaw_brain.knowledge.extraction.extractor import (
    _EntityExtractionOutput,
    _RelationExtractionOutput,
    extract_from_chunk,
    pass2_failure_count,
    reset_pass2_failure_tracking,
)
from openclaw_brain.knowledge.extraction.models import ConceptMention, ParameterMention, RawEdge


# ── Double-encoding tolerance (claude-sonnet-5 quirk) ──────────────────


def test_relationships_field_as_json_string_is_recovered():
    """`relationships` arrives as a JSON string encoding the list directly."""
    payload = json.dumps([
        {
            "source_name": "Cascode",
            "target_name": "Low Output Impedance",
            "relationship": "SOLVES_PROBLEM",
            "rationale": "Cascode stacks a common-gate stage to boost Rout",
            "confidence": 0.9,
        }
    ])
    out = _RelationExtractionOutput(relationships=payload)
    assert len(out.relationships) == 1
    assert out.relationships[0].source_name == "Cascode"
    assert out.relationships[0].target_name == "Low Output Impedance"


def test_relationships_field_double_wrapped_is_recovered():
    """`relationships` arrives as a JSON string whose parsed value is itself a dict wrapping
    another `relationships` key (the observed sonnet-5 double-encoding shape)."""
    payload = json.dumps({
        "relationships": [
            {
                "source_name": "A",
                "target_name": "B",
                "relationship": "DEPENDS_ON",
            }
        ]
    })
    out = _RelationExtractionOutput(relationships=payload)
    assert len(out.relationships) == 1
    assert out.relationships[0].source_name == "A"
    assert out.relationships[0].target_name == "B"


def test_whole_payload_stringified_is_recovered():
    """Belt-and-suspenders: the entire args dict comes back JSON-encoded as a single string."""
    payload = json.dumps({
        "relationships": [
            {"source_name": "A", "target_name": "B", "relationship": "RELATES_TO"}
        ]
    })
    out = _RelationExtractionOutput.model_validate(payload)
    assert len(out.relationships) == 1


def test_genuinely_unparseable_relationships_still_raises():
    """A string that isn't recoverable JSON must not be silently swallowed — it should still
    fail validation exactly as before this fix, so the caller's existing entities-only
    degradation path still applies when a list genuinely cannot be recovered."""
    with pytest.raises(ValidationError):
        _RelationExtractionOutput(relationships="this is not json")


def test_clean_list_input_unaffected_by_double_encoding_coercion():
    """REGRESSION: a normal, already-parsed list of relationship dicts (the deepseek /
    sonnet-4-6 path) must be byte-identical in behavior."""
    out = _RelationExtractionOutput(relationships=[
        {
            "source_name": "Cascode",
            "target_name": "Low Output Impedance",
            "relationship": "SOLVES_PROBLEM",
            "rationale": "Cascode stacks a common-gate stage to boost Rout",
            "confidence": 0.9,
        }
    ])
    assert len(out.relationships) == 1
    assert out.relationships[0].source_name == "Cascode"
    assert out.relationships[0].confidence == 0.9


def test_empty_relationships_defaults_to_empty_list():
    """REGRESSION: omitting `relationships` entirely still defaults to an empty list."""
    out = _RelationExtractionOutput()
    assert out.relationships == []


# ── extract_from_chunk() Pass 2 failure visibility ──────────────────────
#
# extract_from_chunk() is dead code in production today (pipeline.py's _resilient_extract
# reimplements the same two-pass flow inline) — see extraction/README.md. But the old bare
# `except Exception: pass` around Pass 2 meant ANY future caller (or a resurrection of this
# function as the canonical path) would silently lose every edge on a transient LLM error with
# zero signal. These tests exercise extract_from_chunk() directly.


class _FakeEntityLLM:
    """`.with_structured_output(_EntityExtractionOutput)` result — Pass 1 always succeeds."""

    async def ainvoke(self, messages):
        return _EntityExtractionOutput(
            concepts=[ConceptMention(name="Cascode", description="A cascode topology.")],
            equations=[],
            parameters=[ParameterMention(symbol="gm", name="Transconductance")],
        )


class _Pass2RaisingLLM:
    """`.with_structured_output(_RelationExtractionOutput)` result — Pass 2 always raises."""

    async def ainvoke(self, messages):
        raise TimeoutError("relation extraction timed out")


class _Pass2FailingChatModel:
    """Routes `.with_structured_output(schema)` to the Pass-1-succeeds / Pass-2-raises fakes."""

    def with_structured_output(self, schema):
        if schema is _EntityExtractionOutput:
            return _FakeEntityLLM()
        return _Pass2RaisingLLM()


class _Pass2SucceedingLLM:
    async def ainvoke(self, messages):
        return _RelationExtractionOutput(relationships=[
            RawEdge(source_name="Cascode", target_name="Transconductance", relationship="DEPENDS_ON"),
        ])


class _CleanChatModel:
    def with_structured_output(self, schema):
        if schema is _EntityExtractionOutput:
            return _FakeEntityLLM()
        return _Pass2SucceedingLLM()


@pytest.mark.asyncio
async def test_extract_from_chunk_pass2_failure_is_logged_with_chunk_id_and_type(caplog):
    """extract_from_chunk()'s Pass 2 failing must not silently zero out ALL relationships with
    no signal — it must log at WARNING with the chunk id + exception type, and Pass 1's entities
    must still come through (degrade, don't lose what already succeeded)."""
    reset_pass2_failure_tracking()
    try:
        with caplog.at_level(logging.WARNING, logger="openclaw_brain.knowledge.extraction.extractor"):
            result = await extract_from_chunk("some chunk text", "chunk_x1", _Pass2FailingChatModel())

        assert len(result.concepts) == 1
        assert len(result.parameters) == 1
        assert result.raw_edges == []  # Pass 2 lost, but visibly now

        messages = [r.getMessage() for r in caplog.records]
        assert any("chunk_x1" in m and "TimeoutError" in m for m in messages)
    finally:
        reset_pass2_failure_tracking()


@pytest.mark.asyncio
async def test_extract_from_chunk_pass2_failure_increments_aggregable_counter():
    """A caller must be able to aggregate Pass 2 failures without scraping logs."""
    reset_pass2_failure_tracking()
    try:
        assert pass2_failure_count() == 0
        await extract_from_chunk("text a", "chunk_a", _Pass2FailingChatModel())
        assert pass2_failure_count() == 1
        await extract_from_chunk("text b", "chunk_b", _Pass2FailingChatModel())
        assert pass2_failure_count() == 2
    finally:
        reset_pass2_failure_tracking()


@pytest.mark.asyncio
async def test_extract_from_chunk_pass2_success_does_not_count_as_failure():
    """REGRESSION: a normal, successful Pass 2 must not be counted or warned about."""
    reset_pass2_failure_tracking()
    try:
        result = await extract_from_chunk("text", "chunk_ok", _CleanChatModel())
        assert len(result.raw_edges) == 1
        assert pass2_failure_count() == 0
    finally:
        reset_pass2_failure_tracking()
