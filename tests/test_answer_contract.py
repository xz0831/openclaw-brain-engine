"""Offline contract fixtures: no network, model, or graph writes."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import openclaw_brain.agent as agent_module
from openclaw_brain.agent import BrainAgent
from openclaw_brain.knowledge.reasoning.answer_contract import (
    ContextManifest, NO_KB_PREFIX, PARTIAL_KB_PREFIX, build_envelope,
)
from openclaw_brain.knowledge.reasoning.answer_resolution import classify_rows
from openclaw_brain.knowledge.reasoning.answerer import AnswerResult


def _manifest(level="chunk"):
    return ContextManifest.from_refs([{"id": "c1", "cite": {"level": level, "chunks": ["ch1"]}}])


def _candidate(answer="Explanation [cite: c1]", citations=None, **kwargs):
    return AnswerResult(answer=answer, citations=["c1"] if citations is None else citations,
                        **kwargs)


def _row(kind="Concept", **kwargs):
    return {"resolution": "active", "exists": True, "kind": kind, "tier": 1,
            "scope": None, "verdict": None, **kwargs}


def _audit(*findings):
    return {"state": "completed", "passed": not findings,
            "findings": list(findings)}


@pytest.mark.parametrize("answer,citations,expected", [
    ("Explanation", ["c1"], ["array"]),
    ("Explanation [cite: c1]", [], ["body"]),
    ("Explanation [cite: c1]", ["c1"], ["array", "body"]),
])
def test_citation_origins_and_normal_state(answer, citations, expected):
    env = build_envelope(_candidate(answer, citations), _manifest(), {"c1": _row()}, _audit(), "flag")
    assert env.status == "unverified"
    assert env.checks.citation_integrity == "pass"
    assert env.citations[0].origins == expected
    assert env.status != "grounded"


@pytest.mark.parametrize("resolution,context,reason", [
    ("missing", True, "citation_missing"),
    ("retracted", True, "citation_retracted"),
    ("ambiguous", True, "citation_ambiguous"),
    ("error", True, "citation_error"),
    ("active", False, "citation_out_of_context"),
])
def test_bad_citations_are_invalid(resolution, context, reason):
    manifest = _manifest() if context else ContextManifest.from_refs([{"id": "other"}])
    row = _row(resolution=resolution)
    env = build_envelope(_candidate(), manifest, {"c1": row}, _audit(), "flag")
    assert env.status == "invalid"
    assert reason in env.reason_codes


def test_array_only_unresolved_and_used_concept_checks():
    env = build_envelope(_candidate("body", ["bad"], used_concepts=["absent"]),
                         _manifest(), {"bad": {"resolution": "missing", "exists": False}},
                         _audit(), "flag")
    assert env.status == "invalid"
    assert "citation_missing" in env.reason_codes
    assert "used_concepts_out_of_context" in env.reason_codes


def test_used_concept_must_exist_even_when_it_was_sent():
    candidate = _candidate("Explanation [cite: ch1]", ["ch1"], used_concepts=["c1"])
    env = build_envelope(candidate, _manifest(), {
        "ch1": _row("SourceChunk", tier=0),
        "c1": {"resolution": "missing", "exists": False},
    }, _audit(), "flag")
    assert env.status == "invalid"
    assert "used_concept_missing" in env.reason_codes


@pytest.mark.parametrize("candidate,audit,reason", [
    (_candidate(answer="", abstained=False), _audit(), "contradictory_payload"),
    (_candidate(answer="body", abstained=True), _audit(), "contradictory_payload"),
    (AnswerResult(abstained=True, abstain_reason="synthesis_failed"), _audit(), "synthesis_failed"),
    (_candidate(), _audit({"kind": "scope_overstatement", "advisory": False,
                           "detail": "candidate secret"}), "audit_finding"),
])
def test_operational_and_audit_failures(candidate, audit, reason):
    env = build_envelope(candidate, _manifest(), {"c1": _row()}, audit, "flag")
    assert env.status == "invalid"
    assert reason in env.reason_codes
    assert "candidate secret" not in env.model_dump_json()


@pytest.mark.parametrize("policy", ["flag", "abstain"])
def test_other_policies_still_reject_body_with_abstention(policy):
    env = build_envelope(AnswerResult(answer="body", abstained=True),
                         _manifest(), {}, _audit(), policy)
    assert env.status == "invalid"
    assert "contradictory_payload" in env.reason_codes
    assert "model_abstained_with_body" not in env.reason_codes


@pytest.mark.parametrize("resolution,reason", [
    ("missing", "citation_missing"),
    ("retracted", "citation_retracted"),
    ("ambiguous", "citation_ambiguous"),
    ("error", "citation_error"),
    ("active", "citation_out_of_context"),
])
def test_model_knowledge_body_with_abstention_does_not_override_bad_citation(resolution, reason):
    candidate = AnswerResult(answer="body [cite: forged]", citations=["forged"],
                             abstained=True)
    env = build_envelope(candidate, _manifest(), {"forged": _row(resolution=resolution)},
                         _audit(), "model_knowledge")
    assert env.status == "invalid"
    assert reason in env.reason_codes
    assert "model_abstained_with_body" in env.reason_codes


def test_no_citation_policy_derived_and_abstention():
    candidate = _candidate("Ordinary answer", [])
    assert build_envelope(candidate, _manifest(), {}, _audit(), "flag").status == "unverified"
    env = build_envelope(candidate, _manifest(), {}, _audit(), "abstain")
    assert env.status == "abstained"
    assert "insufficient_citations" in env.reason_codes
    assert build_envelope(_candidate(), _manifest("derived"), {"c1": _row()},
                          _audit(), "flag").checks.all_citations_derived
    model_abstained = AnswerResult(abstained=True)
    env = build_envelope(model_abstained, _manifest(), {}, _audit(), "flag")
    assert env.status == "abstained" and "model_abstained" in env.reason_codes


def test_id_limit_and_prefix_exactness():
    candidate = _candidate("body", [f"id{i}" for i in range(33)])
    env = build_envelope(candidate, _manifest(), {}, _audit(), "flag")
    assert env.status == "invalid" and "id_limit" in env.reason_codes
    assert classify_rows([])["resolution"] == "missing"
    assert classify_rows([{"labels": ["ClaimCard"]}, {"labels": ["Concept"]}])["resolution"] == "ambiguous"
    assert classify_rows([{"labels": ["ClaimCard"], "retracted": True}])["resolution"] == "retracted"
    # The resolver uses equality only; a long sha256 prefix never becomes a match.
    from openclaw_brain.knowledge.reasoning.answer_resolution import _QUERY
    assert "STARTS WITH" not in _QUERY


def test_card_and_law_tiers_and_scope():
    refuted = classify_rows([{"labels": ["ClaimCard"], "verdict": "REFUTED",
                             "engine": "ngspice", "scope": '{"pdk":"sky130"}'}])
    assert refuted["tier"] == 3 and refuted["verdict"] == "REFUTED"
    assert classify_rows([{"labels": ["ClaimCard"], "verdict": None, "engine": None}])["tier"] is None
    members = {p: {"verdict": "VERIFIED"} for p in ("sky130", "gf180", "ihp")}
    law = classify_rows([{"labels": ["Regularity"], "status": "law",
                          "member_summary": __import__("json").dumps(members)}])
    assert law["tier"] == 4 and len(law["scope_pdks"]) == 3
    assert classify_rows([{"labels": ["Regularity"], "status": "process_scoped",
                           "member_summary": __import__("json").dumps(members)}])["tier"] is None
    assert classify_rows([{"labels": ["Regularity"], "status": "law",
                           "member_summary": "bad", "pdks": ["sky130", "gf180", "ihp"]}])["tier"] is None


@pytest.mark.asyncio
async def test_agent_invalid_suppresses_candidate_and_one_synthesis(monkeypatch):
    agent = BrainAgent.__new__(BrainAgent)
    agent._started = True
    agent._config = SimpleNamespace(resilience=object())
    agent._llm_provider = SimpleNamespace(get_chain=MagicMock(return_value=[object()]))
    agent.query_knowledge = AsyncMock(return_value={
        "formatted": "concept summary", "concepts_found": 1,
        "concept_refs": [{"id": "c1", "cite": {"level": "derived"}}],
        "open_hypotheses": [], "active_decisions": [],
    })
    agent._graph = SimpleNamespace(run_read_query=AsyncMock(return_value=[]))
    synth = AsyncMock(return_value=_candidate("secret candidate", ["fake"]))
    monkeypatch.setattr(agent_module, "answer_from_context", synth)
    result = await agent.answer_question("question", on_insufficient="flag")
    assert result["answer"] == ""
    assert result["abstain_reason"] == "contract_invalid:citation_missing"
    assert "secret candidate" not in str(result)
    synth.assert_awaited_once()
    assert agent._graph.run_read_query.await_count == 1


@pytest.mark.asyncio
async def test_agent_no_context_and_bad_policy_never_call_model(monkeypatch):
    agent = BrainAgent.__new__(BrainAgent)
    agent._started = True
    agent.query_knowledge = AsyncMock(return_value={
        "formatted": "memory text", "concepts_found": 0, "concept_refs": [],
        "open_hypotheses": [], "active_decisions": [],
    })
    synth = AsyncMock()
    monkeypatch.setattr(agent_module, "answer_from_context", synth)
    result = await agent.answer_question("question", on_insufficient="flag")
    assert result["envelope"]["status"] == "abstained"
    synth.assert_not_awaited()
    abstain = await agent.answer_question("question", on_insufficient="abstain")
    assert abstain["envelope"]["status"] == "abstained"
    synth.assert_not_awaited()
    with pytest.raises(ValueError):
        await agent.answer_question("question", on_insufficient="ignore")


@pytest.mark.asyncio
@pytest.mark.parametrize("answer,expected", [
    ("REFUTED result [cite: card1]", "unverified"),
    ("Certified on sky130 [certified: card1]", "invalid"),
    ("sky130 및 미지 PDK 결과 [cite: card1]", "unverified"),
])
async def test_refuted_scope_and_language_limits(monkeypatch, answer, expected):
    agent = BrainAgent.__new__(BrainAgent)
    agent._started = True
    agent._config = SimpleNamespace(resilience=object())
    agent._llm_provider = SimpleNamespace(get_chain=MagicMock(return_value=[object()]))
    agent.query_knowledge = AsyncMock(return_value={
        "formatted": "card", "concepts_found": 1,
        "concept_refs": [{"id": "card1", "cite": {"level": "chunk"}}],
        "open_hypotheses": [], "active_decisions": [],
    })
    agent._graph = SimpleNamespace(run_read_query=AsyncMock(return_value=[{
        "labels": ["ClaimCard"], "verdict": "REFUTED", "engine": "ngspice",
        "scope": '{"pdk":"sky130"}',
    }]))
    synth = AsyncMock(return_value=_candidate(answer, ["card1"]))
    monkeypatch.setattr(agent_module, "answer_from_context", synth)
    result = await agent.answer_question("q", on_insufficient="flag")
    assert result["envelope"]["status"] == expected
    assert result["envelope"]["citations"][0]["verdict"] == "REFUTED"
    assert result["envelope"]["checks"]["semantic_support"] == "not_checked"
    if expected == "invalid":
        assert result["answer"] == ""


@pytest.mark.asyncio
async def test_broken_provenance_warns_without_tier_upgrade(monkeypatch):
    agent = BrainAgent.__new__(BrainAgent)
    agent._started = True
    agent._config = SimpleNamespace(resilience=object())
    agent._llm_provider = SimpleNamespace(get_chain=MagicMock(return_value=[object()]))
    agent.query_knowledge = AsyncMock(return_value={
        "formatted": "body effect", "concepts_found": 1,
        "concept_refs": [{"id": "body_effect", "cite": {"level": "chunk", "chunks": ["missing_chunk"]}}],
        "open_hypotheses": [], "active_decisions": [],
    })

    async def graph_read(query, params):
        return [{"labels": ["Concept"]}] if params["cid"] == "body_effect" else []

    agent._graph = SimpleNamespace(run_read_query=AsyncMock(side_effect=graph_read))
    monkeypatch.setattr(agent_module, "answer_from_context",
                        AsyncMock(return_value=_candidate("Body effect [cite: body_effect]", ["body_effect"])))
    result = await agent.answer_question("q", on_insufficient="flag")
    assert result["envelope"]["status"] == "unverified"
    assert result["envelope"]["citations"][0]["tier"] == 1
    assert "broken_provenance_ref" in result["envelope"]["audit"]["warnings"]
    assert agent._graph.run_read_query.await_count == 2


@pytest.mark.asyncio
async def test_db_timeout_is_invalid_without_error_text(monkeypatch):
    agent = BrainAgent.__new__(BrainAgent)
    agent._started = True
    agent._config = SimpleNamespace(resilience=object())
    agent._llm_provider = SimpleNamespace(get_chain=MagicMock(return_value=[object()]))
    agent.query_knowledge = AsyncMock(return_value={
        "formatted": "concept", "concepts_found": 1,
        "concept_refs": [{"id": "c1", "cite": {"level": "chunk"}}],
        "open_hypotheses": [], "active_decisions": [],
    })
    agent._graph = SimpleNamespace(run_read_query=AsyncMock(
        side_effect=TimeoutError("private database endpoint")))
    monkeypatch.setattr(agent_module, "answer_from_context",
                        AsyncMock(return_value=_candidate("secret body [cite: c1]")))
    result = await agent.answer_question("q", on_insufficient="flag")
    assert result["envelope"]["status"] == "invalid"
    assert result["envelope"]["checks"]["citation_integrity"] == "unknown"
    assert result["answer"] == ""
    assert "private database endpoint" not in str(result)
    assert "secret body" not in str(result)


def _policy_agent(tmp_path, *, with_context=False, gap_log=True):
    agent = BrainAgent.__new__(BrainAgent)
    agent._started = True
    agent._config = SimpleNamespace(
        resilience=object(), answer=SimpleNamespace(gap_log=gap_log), state_path=tmp_path,
    )
    agent._llm_provider = SimpleNamespace(get_chain=MagicMock(return_value=[object()]))
    refs = [{"id": "c1", "cite": {"level": "chunk"}}] if with_context else []
    agent.query_knowledge = AsyncMock(return_value={
        "formatted": "Known concept" if with_context else "",
        "concepts_found": len(refs), "concept_refs": refs,
        "open_hypotheses": [], "active_decisions": [],
    })
    agent._graph = SimpleNamespace(run_read_query=AsyncMock(return_value=[{"labels": ["Concept"]}]))
    return agent


def _gap_rows(tmp_path):
    path = tmp_path / "knowledge_gaps.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.asyncio
async def test_model_knowledge_no_context_calls_synth_once_and_records_gap(monkeypatch, tmp_path):
    agent = _policy_agent(tmp_path)
    synth = AsyncMock(return_value=AnswerResult(
        answer="Transconductance is current change per voltage change.",
        kb_coverage="none", missing_knowledge=["transconductance definition"],
    ))
    monkeypatch.setattr(agent_module, "answer_from_context", synth)

    result = await agent.answer_question("What is gm?")

    synth.assert_awaited_once()
    assert synth.await_args.kwargs["no_context"] is True
    assert synth.await_args.kwargs["on_insufficient"] == "model_knowledge"
    agent._graph.run_read_query.assert_not_awaited()
    env = result["envelope"]
    assert env["schema_version"] == "answer-envelope/0.1"
    assert env["status"] == "unverified"
    assert env["checks"]["kb_coverage"] == "none"
    assert env["checks"]["model_knowledge"] is True
    assert env["gap_recorded"] is True
    assert result["answer"].startswith(NO_KB_PREFIX + "\n")
    assert result["citations"] == []
    rows = _gap_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0] == {
        "schema": "knowledge-gap/0", "ts": rows[0]["ts"], "question": "What is gm?",
        "missing_knowledge": ["transconductance definition"],
        "kb_coverage": "none", "cited_ids": [],
        "context_sha256": env["context_sha256"],
    }


@pytest.mark.asyncio
async def test_model_knowledge_body_with_abstention_is_delivered_without_citations(monkeypatch, tmp_path):
    agent = _policy_agent(tmp_path)
    monkeypatch.setattr(agent_module, "answer_from_context", AsyncMock(return_value=AnswerResult(
        answer="Transconductance is current change per voltage change.",
        abstained=True, abstain_reason="no graph evidence", kb_coverage="cited",
        missing_knowledge=["transconductance definition"],
    )))

    result = await agent.answer_question("What is gm?")

    assert result["answer"] == (
        NO_KB_PREFIX + "\nTransconductance is current change per voltage change."
    )
    assert result["abstained"] is False
    assert result["abstain_reason"] == ""
    env = result["envelope"]
    assert env["status"] == "unverified"
    assert env["checks"]["kb_coverage"] == "none"
    assert env["checks"]["model_knowledge"] is True
    assert "model_abstained_with_body" in env["reason_codes"]
    assert "contradictory_payload" not in env["reason_codes"]
    assert env["gap_recorded"] is True
    assert len(_gap_rows(tmp_path)) == 1
    assert _gap_rows(tmp_path)[0]["kb_coverage"] == "none"


@pytest.mark.asyncio
async def test_model_knowledge_body_with_abstention_and_valid_citation_is_partial(monkeypatch, tmp_path):
    agent = _policy_agent(tmp_path, with_context=True)
    monkeypatch.setattr(agent_module, "answer_from_context", AsyncMock(return_value=AnswerResult(
        answer="Known [cite: c1]. [모델 지식] General explanation.",
        citations=["c1"], abstained=True, kb_coverage="cited",
        missing_knowledge=["general explanation source"],
    )))

    result = await agent.answer_question("Explain")

    assert result["answer"].startswith(PARTIAL_KB_PREFIX + "\n")
    assert result["abstained"] is False
    env = result["envelope"]
    assert env["status"] == "unverified"
    assert env["checks"]["kb_coverage"] == "partial"
    assert "model_abstained_with_body" in env["reason_codes"]
    assert env["gap_recorded"] is True
    assert len(_gap_rows(tmp_path)) == 1
    assert _gap_rows(tmp_path)[0]["cited_ids"] == ["c1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("with_context", [False, True])
async def test_model_knowledge_empty_abstention_preserves_reason_and_records_gap(
    monkeypatch, tmp_path, with_context,
):
    agent = _policy_agent(tmp_path, with_context=with_context)
    monkeypatch.setattr(agent_module, "answer_from_context", AsyncMock(return_value=AnswerResult(
        answer="", citations=["c1"] if with_context else [],
        abstained=True, abstain_reason="model_abstained", kb_coverage="cited",
        missing_knowledge=["missing explanation"],
    )))

    result = await agent.answer_question("Explain")

    assert result["answer"] == ""
    assert result["abstained"] is True
    assert result["abstain_reason"] == "model_abstained"
    env = result["envelope"]
    assert env["status"] == "abstained"
    assert env["checks"]["kb_coverage"] == "none"
    assert "model_abstained" in env["reason_codes"]
    assert env["gap_recorded"] is True
    assert len(_gap_rows(tmp_path)) == 1
    assert _gap_rows(tmp_path)[0]["kb_coverage"] == "none"


@pytest.mark.asyncio
@pytest.mark.parametrize("body,expected", [
    ("Known fact. [모델 지식]", "Known fact."),
    ("Known fact.\n[모델 지식]  ", "Known fact."),
    ("[모델 지식] Known fact.", "[모델 지식] Known fact."),
])
async def test_model_knowledge_trailing_empty_marker_is_removed(monkeypatch, tmp_path, body, expected):
    agent = _policy_agent(tmp_path)
    monkeypatch.setattr(agent_module, "answer_from_context", AsyncMock(return_value=AnswerResult(
        answer=body, kb_coverage="none",
    )))

    result = await agent.answer_question("Explain")

    assert result["answer"] == NO_KB_PREFIX + "\n" + expected


@pytest.mark.asyncio
async def test_model_knowledge_marker_only_is_empty_abstention(monkeypatch, tmp_path):
    agent = _policy_agent(tmp_path)
    monkeypatch.setattr(agent_module, "answer_from_context", AsyncMock(return_value=AnswerResult(
        answer="  [모델 지식]\n", abstained=True,
    )))

    result = await agent.answer_question("Explain")

    assert result["answer"] == ""
    assert result["abstained"] is True
    assert result["envelope"]["status"] == "abstained"
    assert "model_abstained" in result["envelope"]["reason_codes"]
    assert result["envelope"]["gap_recorded"] is True


@pytest.mark.asyncio
async def test_model_knowledge_forged_citation_with_abstention_is_suppressed(monkeypatch, tmp_path):
    agent = _policy_agent(tmp_path)
    monkeypatch.setattr(agent_module, "answer_from_context", AsyncMock(return_value=AnswerResult(
        answer="Invented [cite: forged]", citations=["forged"], abstained=True,
    )))

    result = await agent.answer_question("Explain")

    assert result["answer"] == ""
    assert result["abstained"] is True
    assert result["envelope"]["status"] == "invalid"
    assert "citation_error" in result["envelope"]["reason_codes"]


@pytest.mark.asyncio
async def test_model_knowledge_partial_and_fully_cited(monkeypatch, tmp_path):
    agent = _policy_agent(tmp_path, with_context=True)
    synth = AsyncMock(side_effect=[
        AnswerResult(answer="Known [cite: c1]. [모델 지식] More context.",
                     citations=["c1"], kb_coverage="partial",
                     missing_knowledge=["missing detail"]),
        AnswerResult(answer="Known [cite: c1].", citations=["c1"], kb_coverage="cited"),
    ])
    monkeypatch.setattr(agent_module, "answer_from_context", synth)

    partial = await agent.answer_question("Explain")
    cited = await agent.answer_question("Explain")

    assert partial["answer"].startswith(PARTIAL_KB_PREFIX + "\n")
    assert partial["envelope"]["checks"]["model_knowledge"] is True
    assert partial["envelope"]["gap_recorded"] is True
    assert cited["answer"] == "Known [cite: c1]."
    assert cited["envelope"]["checks"]["kb_coverage"] == "cited"
    assert cited["envelope"]["checks"]["model_knowledge"] is False
    assert cited["envelope"]["gap_recorded"] is False
    rows = _gap_rows(tmp_path)
    assert len(rows) == 1 and rows[0]["cited_ids"] == ["c1"]


@pytest.mark.asyncio
async def test_model_knowledge_cited_without_valid_citation_downgrades(monkeypatch, tmp_path):
    agent = _policy_agent(tmp_path, with_context=True)
    monkeypatch.setattr(agent_module, "answer_from_context", AsyncMock(return_value=AnswerResult(
        answer="General fact", kb_coverage="cited", missing_knowledge=["fact source"],
    )))
    result = await agent.answer_question("Explain")
    assert result["envelope"]["checks"]["kb_coverage"] == "none"
    assert result["answer"].startswith(NO_KB_PREFIX + "\n")
    assert result["envelope"]["gap_recorded"] is True


@pytest.mark.asyncio
async def test_model_knowledge_no_context_citation_is_invalid(monkeypatch, tmp_path):
    agent = _policy_agent(tmp_path)
    monkeypatch.setattr(agent_module, "answer_from_context", AsyncMock(return_value=AnswerResult(
        answer="Bad [cite: invented]", citations=["invented"], kb_coverage="none",
    )))
    result = await agent.answer_question("Explain")
    assert result["envelope"]["status"] == "invalid"
    assert result["answer"] == ""
    assert result["abstained"] is True
    assert "citation_error" in result["envelope"]["reason_codes"]
    assert result["envelope"]["citations"][0]["in_context"] is False


@pytest.mark.asyncio
async def test_gap_log_off_and_failure_are_nonblocking(monkeypatch, tmp_path):
    agent = _policy_agent(tmp_path, gap_log=False)
    monkeypatch.setattr(agent_module, "answer_from_context", AsyncMock(return_value=AnswerResult(
        answer="A model answer", kb_coverage="none",
        missing_knowledge=[f"gap {i}" for i in range(8)],
    )))
    off = await agent.answer_question("Explain")
    assert off["answer"].startswith(NO_KB_PREFIX)
    assert off["envelope"]["gap_recorded"] is False
    assert off["envelope"]["warnings"] == []
    assert _gap_rows(tmp_path) == []

    agent._config.answer.gap_log = True
    def fail_write(*args, **kwargs):
        raise OSError("private filesystem detail")
    monkeypatch.setattr(agent_module, "append_knowledge_gap", fail_write)
    failed = await agent.answer_question("Explain")
    assert failed["answer"].startswith(NO_KB_PREFIX)
    assert failed["envelope"]["warnings"] == ["gap_log_failed"]
    assert "private filesystem detail" not in str(failed)
    assert failed["envelope"]["gap_recorded"] is False
