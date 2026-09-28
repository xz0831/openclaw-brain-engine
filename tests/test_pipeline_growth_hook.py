"""Unit tests for the corpus-growth ingest hook (`KnowledgePipeline._enqueue_growth_candidates`),
spec §2b/§4 of docs/superpowers/specs/2026-07-03-corpus-growth-automation-design.md.

No real LLM / Neo4j: the per-chunk stages are stubbed (mirrors tests/test_pipeline_concurrency.py's
fake-pipeline pattern); the reasoning stage is faked to return a fixed GraphDelta carrying
CircuitTopology NodeProposals so the hook's registry-match / confidence / layer routing can be
exercised end-to-end through a real `ingest()` call, writing to a tmp state_dir.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import openclaw_brain.knowledge.pipeline as pipe_mod
from openclaw_brain.knowledge.executable import growth as growth_mod
from openclaw_brain.knowledge.extraction.models import ChunkerResult, SourceChunkInfo
from openclaw_brain.knowledge.graph.schema import GraphDelta, NodeLabel, NodeProposal
from openclaw_brain.knowledge.pipeline import KnowledgePipeline

_COUNTS = {"new_nodes": 1, "updated_nodes": 0, "new_edges": 0, "reinforced_edges": 0, "insights": 0}


def _aval(v):
    async def _coro():
        return v
    return _coro()


def _topology_proposal(
    pid: str, name: str, confidence: float, layer: int, chunk_id: str = "chunk_0",
) -> NodeProposal:
    return NodeProposal(
        proposed_id=pid, label=NodeLabel.CIRCUIT_TOPOLOGY, canonical_name=name,
        description="test topology", domain="analog_circuits", knowledge_layer=layer,
        confidence=confidence, evidence_chunk_ids=[chunk_id], reasoning="test fixture",
    )


def _build_pipeline(tmp_path, monkeypatch, new_nodes: list[NodeProposal]) -> KnowledgePipeline:
    """A single-chunk pipeline whose stages are stubbed; the reason stage returns `new_nodes`
    verbatim as the chunk's GraphDelta.new_nodes (pipeline stage 4.5 still runs on top — it injects
    source_id and, for any proposal not in matched_ids, scales confidence by 0.78)."""
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
    pipeline._matcher = SimpleNamespace(
        match=lambda extraction: _aval(
            SimpleNamespace(matched=[], matched_equations=[], matched_parameters=[])
        )
    )
    pipeline._graph = SimpleNamespace(
        apply_delta=lambda delta: _aval(dict(_COUNTS)),
        get_node=lambda *a, **k: _aval(None),
    )
    pipeline._journal = SimpleNamespace(log=lambda *a, **k: None)

    async def fake_extract(text, chunk_id, chain):
        return SimpleNamespace(entities=[], concepts=[], chunk_id=chunk_id)

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
    return pipeline


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_registry_matched_topology_enqueues_growth_queue(tmp_path, monkeypatch):
    nodes = [_topology_proposal("p_matched", "5T OTA", 0.9, 2)]
    pipeline = _build_pipeline(tmp_path, monkeypatch, nodes)
    result = await pipeline.ingest("paper.pdf")
    assert not result.errors

    growth_lines = _read_jsonl(tmp_path / "growth_queue.jsonl")
    assert len(growth_lines) == 1
    rec = growth_lines[0]
    assert rec["topology_class"] == "ota_5t_nmos_in"
    assert rec["canonical_name"] == "5T OTA"
    assert rec["status"] == "pending"
    assert rec["chunk_ids"] == ["chunk_0"]
    assert rec["source_id"] == "src_test"

    assert _read_jsonl(tmp_path / "template_queue.jsonl") == []


@pytest.mark.asyncio
async def test_unmatched_high_confidence_layer2_enqueues_template_queue(tmp_path, monkeypatch):
    nodes = [_topology_proposal("p_unmatched", "Totally Novel Circuit XYZ", 0.95, 2)]
    pipeline = _build_pipeline(tmp_path, monkeypatch, nodes)
    await pipeline.ingest("paper.pdf")

    template_lines = _read_jsonl(tmp_path / "template_queue.jsonl")
    assert len(template_lines) == 1
    rec = template_lines[0]
    assert rec["canonical_name"] == "Totally Novel Circuit XYZ"
    assert rec["layer"] == 2
    assert rec["confidence"] >= 0.7
    assert rec["chunk_ids"] == ["chunk_0"]

    assert _read_jsonl(tmp_path / "growth_queue.jsonl") == []


@pytest.mark.asyncio
async def test_unmatched_low_confidence_is_skipped(tmp_path, monkeypatch):
    # 0.5 pre-scale -> 0.39 post the pipeline's 0.78 unmatched-concept scaling, well under 0.7.
    nodes = [_topology_proposal("p_low_conf", "Some Other Circuit ABC", 0.5, 2)]
    pipeline = _build_pipeline(tmp_path, monkeypatch, nodes)
    await pipeline.ingest("paper.pdf")

    assert _read_jsonl(tmp_path / "growth_queue.jsonl") == []
    assert _read_jsonl(tmp_path / "template_queue.jsonl") == []


@pytest.mark.asyncio
async def test_unmatched_wrong_layer_is_skipped(tmp_path, monkeypatch):
    # High confidence but layer 1 (device physics) is outside the {2, 3} REVIEW-lane gate.
    nodes = [_topology_proposal("p_wrong_layer", "Yet Another Circuit DEF", 0.95, 1)]
    pipeline = _build_pipeline(tmp_path, monkeypatch, nodes)
    await pipeline.ingest("paper.pdf")

    assert _read_jsonl(tmp_path / "growth_queue.jsonl") == []
    assert _read_jsonl(tmp_path / "template_queue.jsonl") == []


@pytest.mark.asyncio
async def test_non_topology_nodes_never_enqueued(tmp_path, monkeypatch):
    # A Concept node that happens to share a registry-matchable name must NOT be enqueued — only
    # CircuitTopology-labeled proposals are eligible (spec §2b, "NEW CircuitTopology node").
    nodes = [
        NodeProposal(
            proposed_id="c_ignore", label=NodeLabel.CONCEPT, canonical_name="5T OTA",
            domain="analog_circuits", knowledge_layer=2, confidence=0.95,
            evidence_chunk_ids=["chunk_0"], reasoning="test fixture",
        ),
    ]
    pipeline = _build_pipeline(tmp_path, monkeypatch, nodes)
    await pipeline.ingest("paper.pdf")

    assert _read_jsonl(tmp_path / "growth_queue.jsonl") == []
    assert _read_jsonl(tmp_path / "template_queue.jsonl") == []


@pytest.mark.asyncio
async def test_hook_failure_is_non_fatal(tmp_path, monkeypatch):
    nodes = [_topology_proposal("p_matched", "5T OTA", 0.9, 2)]
    pipeline = _build_pipeline(tmp_path, monkeypatch, nodes)

    def _boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(growth_mod, "enqueue_candidates", _boom)

    result = await pipeline.ingest("paper.pdf")
    assert not result.errors
    assert result.new_nodes == 1
    # enqueue_candidates blew up before writing anything — no partial/garbage queue file either.
    assert not (tmp_path / "growth_queue.jsonl").is_file()


@pytest.mark.asyncio
async def test_no_topology_nodes_skips_hook_entirely(tmp_path, monkeypatch):
    pipeline = _build_pipeline(tmp_path, monkeypatch, [])
    result = await pipeline.ingest("paper.pdf")

    assert not result.errors
    assert not (tmp_path / "growth_queue.jsonl").is_file()
    assert not (tmp_path / "template_queue.jsonl").is_file()
