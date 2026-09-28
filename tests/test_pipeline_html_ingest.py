"""Pipeline-seam test: KnowledgePipeline.ingest_html routes through the IDENTICAL
chunk->extract->ground->match->reason->reconcile->commit->embed->summarize path as ingest()
(PDF) — only the parse stage (html_parser vs MinerU) and figure-analysis stage (skipped
entirely, never merely no-opped) differ. See knowledge/pipeline.py::_ingest_from_parsed, the
shared seam this test exists to pin.

Mock fidelity (CLAUDE.md Testing Conventions / PHILOSOPHY.md P11): every object below mirrors
the REAL shape at its call site — real ParsedDocument/ContentBlock (mineru_parser.py, reused
unchanged by html_parser.py), real ExtractionResult/MatchResult (extraction/models.py), real
GraphDelta/NodeProposal/NodeLabel enums (graph/schema.py). ``chunk_structured()`` itself runs
FOR REAL (pure/deterministic, no I/O) rather than being mocked, so this test proves genuine
parse-output -> chunk-boundary integration, not just "some stage was called".
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import openclaw_brain.knowledge.pipeline as pipe_mod
from openclaw_brain.knowledge.extraction.mineru_parser import ContentBlock, ParsedDocument
from openclaw_brain.knowledge.extraction.models import ExtractionResult, MatchResult
from openclaw_brain.knowledge.graph.schema import GraphDelta, NodeLabel, NodeProposal
from openclaw_brain.knowledge.pipeline import KnowledgePipeline


def _aval(v):
    async def _coro():
        return v
    return _coro()


class FakeGraph:
    """Records every call so assertions can inspect the REAL shapes reaching the commit path."""

    def __init__(self):
        self.merged_nodes: list[dict] = []
        self.merged_edges: list[dict] = []
        self.applied_deltas: list[GraphDelta] = []

    async def merge_node(self, **kwargs):
        self.merged_nodes.append(kwargs)
        return kwargs["id_value"]

    async def merge_edge(self, **kwargs):
        self.merged_edges.append(kwargs)

    async def apply_delta(self, delta):
        assert isinstance(delta, GraphDelta)
        self.applied_deltas.append(delta)
        return {
            "new_nodes": len(delta.new_nodes), "updated_nodes": len(delta.updated_nodes),
            "new_edges": len(delta.new_edges), "reinforced_edges": len(delta.reinforced_edges),
            "insights": len(delta.insights),
        }

    async def get_node(self, *a, **k):
        return None

    def _id_field_for_label(self, label):
        return "concept_id"


def _real_parsed_document(source_id: str = "src_html_test", n_pages: int = 2) -> ParsedDocument:
    """The exact ContentBlock shape html_parser.parse_lecture_html produces for N pages with
    real speech: one text_level=1 heading block + one text_level=0 body block per page."""
    blocks: list[ContentBlock] = []
    for i in range(n_pages):
        blocks.append(ContentBlock(
            type="text", text=f"p.{i + 1} (0:0{i}-0:0{i + 1})", page_idx=i, text_level=1,
        ))
        blocks.append(ContentBlock(
            type="text", text=f"synthetic transcript body for page {i + 1}", page_idx=i, text_level=0,
        ))
    return ParsedDocument(
        source_id=source_id, title="Synthetic HTML Lecture", total_pages=n_pages,
        checksum="deadbeef" * 8, blocks=blocks, figures=[], output_dir="",
    )


def _build_pipeline(tmp_path, monkeypatch):
    """A pipeline whose LLM-calling stages are stubbed; chunk_structured runs for real."""
    pipeline = KnowledgePipeline.__new__(KnowledgePipeline)
    pipeline._config = SimpleNamespace(
        mineru=SimpleNamespace(backend="pipeline"),
        models=SimpleNamespace(
            default_figure_analysis="", default_vision="",
            default_extraction="x-extract", default_reasoning="x-reason",
        ),
        embedding=SimpleNamespace(model="x-embed"),
        resilience=SimpleNamespace(checkpoint_enabled=True, request_timeout_s=30),
        state_path=tmp_path,
    )
    pipeline._provider = SimpleNamespace(
        get_chain=lambda stage, override=None: ["m"],
        get_for_stage=lambda stage: "m",
        # ingest_html must NEVER reach the figure-analysis model lookup — parse_lecture_html
        # never populates ParsedDocument.figures, and ingest_html doesn't even enter that code
        # block (see _ingest_from_parsed's docstring). Raise loudly instead of silently no-op'ing
        # so a future regression that re-adds a figure-analysis call for HTML fails a test.
        get=lambda name: (_ for _ in ()).throw(
            AssertionError(f"figure-analysis model {name!r} must never be requested for HTML ingest")
        ),
    )
    pipeline._matcher = SimpleNamespace(
        match=lambda extraction: _aval(MatchResult(chunk_id=extraction.chunk_id))
    )
    graph = FakeGraph()
    pipeline._graph = graph
    pipeline._journal = SimpleNamespace(log=lambda *a, **k: None)

    calls = {"extract": 0, "reason": 0, "persist_html": [], "persist_pdf": []}

    async def fake_extract(text, chunk_id, chain):
        calls["extract"] += 1
        return ExtractionResult(chunk_id=chunk_id, concepts=[], equations=[], parameters=[])

    async def fake_reason(extraction, match_result, text, chain, errors=None):
        calls["reason"] += 1
        return GraphDelta(new_nodes=[
            NodeProposal(
                proposed_id=f"concept_{extraction.chunk_id}",
                label=NodeLabel.CONCEPT,
                canonical_name=f"Concept from {extraction.chunk_id}",
                confidence=0.9,
                domain="analog_circuits",
                properties={},
            )
        ])

    async def _anoop(*a, **k):
        return None

    pipeline._resilient_extract = fake_extract
    pipeline._resilient_reason = fake_reason
    pipeline._register_chunk = _anoop
    pipeline._register_source = _anoop
    pipeline._embed_new_nodes = _anoop
    pipeline._persist_mineru_outputs = lambda *a, **k: None
    pipeline._persist_pdf = lambda *a, **k: calls["persist_pdf"].append(a)
    pipeline._persist_html = lambda *a, **k: calls["persist_html"].append(a)

    monkeypatch.setattr(pipe_mod, "verify_grounding", lambda extraction, text: extraction)
    monkeypatch.setattr(pipe_mod, "summarize_paper", _anoop)

    return pipeline, calls, graph


@pytest.mark.asyncio
async def test_ingest_html_calls_html_parser_not_mineru(tmp_path, monkeypatch):
    pipeline, calls, graph = _build_pipeline(tmp_path, monkeypatch)

    parse_calls = []

    def fake_parse_lecture_html(path):
        parse_calls.append(path)
        return _real_parsed_document()

    def _boom_mineru(*a, **k):
        raise AssertionError("mineru_parse_pdf must never be called by ingest_html")

    monkeypatch.setattr(pipe_mod, "parse_lecture_html", fake_parse_lecture_html)
    monkeypatch.setattr(pipe_mod, "mineru_parse_pdf", _boom_mineru)

    result = await pipeline.ingest_html("lecture.html")

    assert len(parse_calls) == 1
    assert result.source_id == "src_html_test"
    assert result.errors == []
    assert result.total_chunks == 2
    assert calls["extract"] == 2
    assert calls["reason"] == 2


@pytest.mark.asyncio
async def test_ingest_html_commits_through_same_apply_delta_path(tmp_path, monkeypatch):
    """Real-shape proof: the GraphDelta objects reaching GraphStore.apply_delta carry real
    NodeLabel enums and real NodeProposal instances — the SAME commit contract ingest() (PDF)
    uses, not a parallel/ad hoc write path (PHILOSOPHY.md P5: enforce once, at the choke
    point). Also proves the M4 source_id stamp (pipeline.py's reconcile stage) still applies
    unchanged for HTML-sourced proposals."""
    pipeline, calls, graph = _build_pipeline(tmp_path, monkeypatch)
    monkeypatch.setattr(pipe_mod, "parse_lecture_html", lambda path: _real_parsed_document())

    result = await pipeline.ingest_html("lecture.html")

    assert len(graph.applied_deltas) == 2  # one per chunk (2 pages -> 2 chunks)
    for delta in graph.applied_deltas:
        assert isinstance(delta, GraphDelta)
        assert len(delta.new_nodes) == 1
        proposal = delta.new_nodes[0]
        assert isinstance(proposal, NodeProposal)
        assert proposal.label is NodeLabel.CONCEPT  # real enum, not a string
        assert proposal.properties.get("source_id") == "src_html_test"
    assert result.new_nodes == 2


@pytest.mark.asyncio
async def test_ingest_html_persists_via_html_kind_not_pdf(tmp_path, monkeypatch):
    pipeline, calls, graph = _build_pipeline(tmp_path, monkeypatch)
    monkeypatch.setattr(pipe_mod, "parse_lecture_html", lambda path: _real_parsed_document())

    await pipeline.ingest_html("lecture.html")

    assert len(calls["persist_html"]) == 1
    assert calls["persist_pdf"] == []


@pytest.mark.asyncio
async def test_ingest_html_uses_same_checkpoint_semantics_as_pdf(tmp_path, monkeypatch):
    """Same PipelineCheckpoint file convention + archive-on-completion behavior as ingest()
    (see test_pipeline_concurrency.py's parallel assertion for the PDF path)."""
    pipeline, calls, graph = _build_pipeline(tmp_path, monkeypatch)
    monkeypatch.setattr(pipe_mod, "parse_lecture_html", lambda path: _real_parsed_document())

    result = await pipeline.ingest_html("lecture.html")

    assert result.source_id == "src_html_test"
    completed = tmp_path / "checkpoints" / "completed" / "src_html_test.json"
    assert completed.exists()
    live = tmp_path / "checkpoints" / "src_html_test.json"
    assert not live.exists()


@pytest.mark.asyncio
async def test_ingest_html_parse_failure_degrades_to_errors_not_raise(tmp_path, monkeypatch):
    """Mirrors ingest()'s "MinerU parse failed" degrade path exactly (PHILOSOPHY.md P9:
    degrade, but never silently) — a parse exception becomes a recorded error, never an
    unhandled raise, and commits nothing."""
    pipeline, calls, graph = _build_pipeline(tmp_path, monkeypatch)

    def _boom(path):
        raise ValueError("malformed HTML")

    monkeypatch.setattr(pipe_mod, "parse_lecture_html", _boom)

    result = await pipeline.ingest_html("lecture.html")

    assert result.errors and "HTML parse failed" in result.errors[0]
    assert graph.applied_deltas == []


@pytest.mark.asyncio
async def test_ingest_html_no_speech_pages_reports_no_text_extracted(tmp_path, monkeypatch):
    """An all-pdfonly HTML source (parse_lecture_html returns zero blocks) degrades exactly
    like an empty/scanned PDF does — a recorded error, not a crash, and zero commits."""
    pipeline, calls, graph = _build_pipeline(tmp_path, monkeypatch)
    empty_doc = ParsedDocument(
        source_id="src_empty", title="Empty", total_pages=5, checksum="c" * 64,
        blocks=[], figures=[], output_dir="",
    )
    monkeypatch.setattr(pipe_mod, "parse_lecture_html", lambda path: empty_doc)

    result = await pipeline.ingest_html("lecture.html")

    assert any("No text extracted from HTML" in e for e in result.errors)
    assert graph.applied_deltas == []
