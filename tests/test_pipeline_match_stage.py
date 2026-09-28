"""Unit tests for F6's Stage-4.5 match->reason handoff in KnowledgePipeline.ingest.

Before F6, matched_ids (pipeline.py) was built only from match_result.matched (Concepts) —
Equations/Parameters never entered matching at all, so every mention committed as new_nodes
with zero de-duplication signal (knowledge/reasoning/README.md "Known defects" #2). This file
verifies matched equations/parameters now route through the SAME mechanism concepts already
use: (1) a mechanical reinforcement_count NodeUpdate (M2), and (2) exemption from the
unmatched-node confidence scale-down (M5) — mirroring
tests/test_pipeline_growth_hook.py's stubbed-pipeline pattern. No real LLM/Neo4j.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import openclaw_brain.knowledge.pipeline as pipe_mod
from openclaw_brain.knowledge.extraction.models import (
    ChunkerResult,
    ConceptMention,
    EquationMention,
    MatchedConcept,
    MatchedEquation,
    MatchedParameter,
    MatchResult,
    ParameterMention,
    SourceChunkInfo,
)
from openclaw_brain.knowledge.graph.schema import GraphDelta, NodeLabel, NodeProposal
from openclaw_brain.knowledge.pipeline import KnowledgePipeline

_COUNTS = {"new_nodes": 0, "updated_nodes": 0, "new_edges": 0, "reinforced_edges": 0, "insights": 0}


def _aval(v):
    async def _coro():
        return v
    return _coro()


def _new_proposal(pid: str, label: NodeLabel, confidence: float = 0.8) -> NodeProposal:
    return NodeProposal(
        proposed_id=pid, label=label, canonical_name=f"name for {pid}",
        description="d", domain="analog_circuits", confidence=confidence,
        evidence_chunk_ids=["chunk_0"], reasoning="test fixture",
    )


def _build_pipeline(tmp_path, monkeypatch, match_result, new_nodes, existing_nodes):
    """A single-chunk pipeline whose stages are stubbed (mirrors
    tests/test_pipeline_growth_hook.py's _build_pipeline). `existing_nodes` maps
    (NodeLabel.value, id_value) -> node dict, feeding a stubbed get_node so Stage 4.5's
    reinforcement lookup has something to read. apply_delta CAPTURES the delta it receives
    (unlike the sibling files, which discard it) so tests can assert on Stage 4.5's mutations.
    """
    pipeline = KnowledgePipeline.__new__(KnowledgePipeline)
    pipeline._config = SimpleNamespace(
        mineru=SimpleNamespace(backend="pipeline"),
        models=SimpleNamespace(
            default_figure_analysis="", default_vision="",
            default_extraction="x-extract", default_reasoning="x-reason",
        ),
        embedding=SimpleNamespace(model="x-embed"),
        resilience=SimpleNamespace(checkpoint_enabled=False, request_timeout_s=30),
        state_path=tmp_path,
    )
    pipeline._provider = SimpleNamespace(
        get_chain=lambda stage, override=None: ["m"],
        get_for_stage=lambda stage: "m",
        get=lambda name: "m",
    )
    pipeline._matcher = SimpleNamespace(match=lambda extraction: _aval(match_result))

    captured_deltas: list[GraphDelta] = []

    async def fake_apply_delta(delta):
        captured_deltas.append(delta)
        return dict(_COUNTS)

    async def fake_get_node(label, id_field, id_value):
        return existing_nodes.get((label.value, id_value))

    pipeline._graph = SimpleNamespace(apply_delta=fake_apply_delta, get_node=fake_get_node)
    pipeline._journal = SimpleNamespace(log=lambda *a, **k: None)

    async def fake_extract(text, chunk_id, chain):
        return SimpleNamespace(
            entities=[], concepts=[], equations=[], parameters=[], chunk_id=chunk_id,
        )

    async def fake_reason(extraction, match_result, text, chain, errors=None):
        return GraphDelta(new_nodes=list(new_nodes))

    async def _anoop(*a, **k):
        return None

    pipeline._resilient_extract = fake_extract
    pipeline._resilient_reason = fake_reason
    pipeline._register_chunk = _anoop
    pipeline._register_source = _anoop
    pipeline._embed_new_nodes = _anoop
    pipeline._persist_mineru_outputs = lambda *a, **k: None
    pipeline._persist_pdf = lambda *a, **k: None

    chunks = [SourceChunkInfo(
        chunk_id="chunk_0", source_id="src_test", text="body", pages="1", section_title="S0",
    )]

    monkeypatch.setattr(
        pipe_mod, "mineru_parse_pdf",
        lambda path, backend=None: SimpleNamespace(figures=[], output_dir=str(tmp_path)),
    )
    monkeypatch.setattr(
        pipe_mod, "chunk_structured",
        lambda parsed, figure_analyses=None, max_tokens=1500: ChunkerResult(
            source_id="src_test", title="T", chunks=chunks, checksum="ck",
        ),
    )
    monkeypatch.setattr(pipe_mod, "verify_grounding", lambda extraction, text: extraction)
    monkeypatch.setattr(pipe_mod, "summarize_paper", _anoop)
    return pipeline, captured_deltas


@pytest.mark.asyncio
async def test_matched_equation_gets_reinforcement_update(tmp_path, monkeypatch):
    eq_mention = EquationMention(latex="GBW = g_m/(2*pi*C_L)")
    match_result = MatchResult(
        chunk_id="chunk_0",
        matched_equations=[MatchedEquation(mention=eq_mention, existing_node_id="eq_gbw")],
    )
    existing = {("Equation", "eq_gbw"): {"equation_id": "eq_gbw", "reinforcement_count": 3}}
    pipeline, deltas = _build_pipeline(tmp_path, monkeypatch, match_result, [], existing)

    result = await pipeline.ingest("paper.pdf")

    assert not result.errors
    assert len(deltas) == 1
    updates = deltas[0].updated_nodes
    assert len(updates) == 1
    assert updates[0].existing_node_id == "eq_gbw"
    assert updates[0].label == NodeLabel.EQUATION
    assert updates[0].updates == {"reinforcement_count": 4}
    assert updates[0].reasoning == "equation re-encountered in new source"


@pytest.mark.asyncio
async def test_matched_parameter_gets_reinforcement_update(tmp_path, monkeypatch):
    p_mention = ParameterMention(symbol="GBW", name="Gain-Bandwidth Product", units="Hz")
    match_result = MatchResult(
        chunk_id="chunk_0",
        matched_parameters=[MatchedParameter(mention=p_mention, existing_node_id="p_gbw")],
    )
    existing = {("Parameter", "p_gbw"): {"parameter_id": "p_gbw"}}  # no reinforcement_count yet
    pipeline, deltas = _build_pipeline(tmp_path, monkeypatch, match_result, [], existing)

    await pipeline.ingest("paper.pdf")

    updates = deltas[0].updated_nodes
    assert len(updates) == 1
    assert updates[0].existing_node_id == "p_gbw"
    assert updates[0].label == NodeLabel.PARAMETER
    # missing reinforcement_count defaults to 1, mirroring the pre-existing concept behavior
    assert updates[0].updates == {"reinforcement_count": 2}
    assert updates[0].reasoning == "parameter re-encountered in new source"


@pytest.mark.asyncio
async def test_matched_equation_new_nodes_proposal_not_penalized(tmp_path, monkeypatch):
    """M5: a new_nodes proposal whose id equals a MATCHED equation's existing_node_id must
    keep its confidence unscaled — mirrors the pre-existing concept behavior at pipeline.py's
    `proposed_id not in matched_ids` check, now extended to equations/parameters."""
    eq_mention = EquationMention(latex="GBW = g_m/(2*pi*C_L)")
    match_result = MatchResult(
        chunk_id="chunk_0",
        matched_equations=[MatchedEquation(mention=eq_mention, existing_node_id="eq_gbw")],
    )
    proposals = [
        _new_proposal("eq_gbw", NodeLabel.EQUATION, confidence=0.9),          # matched -> unscaled
        _new_proposal("eq_totally_new", NodeLabel.EQUATION, confidence=0.9),  # unmatched -> scaled
    ]
    pipeline, deltas = _build_pipeline(tmp_path, monkeypatch, match_result, proposals, {})

    await pipeline.ingest("paper.pdf")

    by_id = {p.proposed_id: p for p in deltas[0].new_nodes}
    assert by_id["eq_gbw"].confidence == 0.9
    assert by_id["eq_totally_new"].confidence == pytest.approx(0.9 * 0.78, rel=1e-6)


@pytest.mark.asyncio
async def test_matched_parameter_new_nodes_proposal_not_penalized(tmp_path, monkeypatch):
    p_mention = ParameterMention(symbol="GBW", name="Gain-Bandwidth Product", units="Hz")
    match_result = MatchResult(
        chunk_id="chunk_0",
        matched_parameters=[MatchedParameter(mention=p_mention, existing_node_id="p_gbw")],
    )
    proposals = [_new_proposal("p_gbw", NodeLabel.PARAMETER, confidence=0.85)]
    pipeline, deltas = _build_pipeline(tmp_path, monkeypatch, match_result, proposals, {})

    await pipeline.ingest("paper.pdf")

    assert deltas[0].new_nodes[0].confidence == 0.85  # unchanged -- matched, not penalized


@pytest.mark.asyncio
async def test_unmatched_equation_and_parameter_are_unaffected(tmp_path, monkeypatch):
    """No matched equations/parameters at all -> Stage 4.5 behaves exactly as it did before
    F6 (concept-only matched_ids) -- confirms F6 is dormant when there's nothing to reinforce."""
    match_result = MatchResult(chunk_id="chunk_0")
    proposals = [_new_proposal("eq_new", NodeLabel.EQUATION, confidence=0.9)]
    pipeline, deltas = _build_pipeline(tmp_path, monkeypatch, match_result, proposals, {})

    await pipeline.ingest("paper.pdf")

    assert deltas[0].updated_nodes == []
    assert deltas[0].new_nodes[0].confidence == pytest.approx(0.9 * 0.78, rel=1e-6)


@pytest.mark.asyncio
async def test_concept_and_equation_and_parameter_matches_all_reinforced_together(tmp_path, monkeypatch):
    """Integration: matched/matched_equations/matched_parameters combine into ONE
    reinforcement pass (the all_matched concatenation in pipeline.py), not three
    separate/competing mechanisms."""
    c_mention = ConceptMention(name="Cascode", description="A cascode topology.")
    eq_mention = EquationMention(latex="GBW = g_m/(2*pi*C_L)")
    p_mention = ParameterMention(symbol="GBW", name="Gain-Bandwidth Product", units="Hz")
    match_result = MatchResult(
        chunk_id="chunk_0",
        matched=[MatchedConcept(mention=c_mention, existing_node_id="c_cascode")],
        matched_equations=[MatchedEquation(mention=eq_mention, existing_node_id="eq_gbw")],
        matched_parameters=[MatchedParameter(mention=p_mention, existing_node_id="p_gbw")],
    )
    existing = {
        ("Concept", "c_cascode"): {"concept_id": "c_cascode", "reinforcement_count": 1},
        ("Equation", "eq_gbw"): {"equation_id": "eq_gbw", "reinforcement_count": 1},
        ("Parameter", "p_gbw"): {"parameter_id": "p_gbw", "reinforcement_count": 1},
    }
    pipeline, deltas = _build_pipeline(tmp_path, monkeypatch, match_result, [], existing)

    await pipeline.ingest("paper.pdf")

    updated_by_id = {u.existing_node_id: u for u in deltas[0].updated_nodes}
    assert set(updated_by_id) == {"c_cascode", "eq_gbw", "p_gbw"}
    assert all(u.updates["reinforcement_count"] == 2 for u in updated_by_id.values())
    assert updated_by_id["c_cascode"].reasoning == "concept re-encountered in new source"
    assert updated_by_id["eq_gbw"].reasoning == "equation re-encountered in new source"
    assert updated_by_id["p_gbw"].reasoning == "parameter re-encountered in new source"


@pytest.mark.asyncio
async def test_matched_equation_source_id_still_injected_on_new_nodes(tmp_path, monkeypatch):
    """M4 (source_id injection) is unconditional and label-agnostic already — this just
    confirms F6 didn't accidentally scope it down to concepts only."""
    match_result = MatchResult(chunk_id="chunk_0")
    proposals = [_new_proposal("eq_new", NodeLabel.EQUATION, confidence=0.9)]
    pipeline, deltas = _build_pipeline(tmp_path, monkeypatch, match_result, proposals, {})

    await pipeline.ingest("paper.pdf")

    assert deltas[0].new_nodes[0].properties["source_id"] == "src_test"
