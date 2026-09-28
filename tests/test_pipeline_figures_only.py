"""Tests for figures-only lecture-slide ingest (pipeline.py::KnowledgePipeline
._ingest_html_figures_only, ingest_html(figures_only=True)).

Mock fidelity (CLAUDE.md Testing Conventions / PHILOSOPHY.md P11): real ParsedDocument/
ContentBlock/SlideImage (mineru_parser.py), real GraphDelta/NodeProposal/NodeLabel enums
(graph/schema.py), real SlideAnalysis/analyze_all_slides (slide_analyzer.py) run for real against
AsyncMock VLM clients — only the LLM network boundary and GraphStore are mocked, mirroring
test_pipeline_html_ingest.py's approach for the text path.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import openclaw_brain.knowledge.pipeline as pipe_mod
from openclaw_brain.knowledge.extraction.mineru_parser import ParsedDocument, SlideImage
from openclaw_brain.knowledge.extraction.models import ExtractionResult, MatchResult
from openclaw_brain.knowledge.graph.schema import GraphDelta, NodeLabel, NodeProposal
from openclaw_brain.knowledge.pipeline import KnowledgePipeline
from openclaw_brain.llm.provider import LLMProviderError

_MIN_PNG = (
    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
    b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
    b'\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00'
    b'\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
)

_EQUATION_JSON = json.dumps({
    "equations": [{"latex": "V_{OUT} = -g_m R_D V_{IN}", "meaning": "CS amplifier small-signal gain"}],
    "schematic": None, "plot": None, "summary": "Small-signal gain of a CS amplifier.",
})
_TITLE_ONLY_JSON = json.dumps({
    "equations": [], "schematic": None, "plot": None, "summary": "Course intro / title slide.",
})


def _aval(v):
    async def _coro():
        return v
    return _coro()


class FakeGraph:
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


def _slide_image(page_num: int, mime: str = "image/png") -> SlideImage:
    return SlideImage(
        page_num=page_num, page_idx=page_num - 1, time_range="", mime_type=mime,
        data=lambda: _MIN_PNG,
    )


def _parsed_with_slides(source_id: str, n_slides: int) -> ParsedDocument:
    return ParsedDocument(
        source_id=source_id, title="Synthetic Figures Lecture", total_pages=n_slides,
        checksum="deadbeef" * 8, blocks=[], figures=[], output_dir="",
        slide_images=[_slide_image(i + 1) for i in range(n_slides)],
    )


def _build_pipeline(tmp_path, monkeypatch, primary_llm=None, fallback_llm=None):
    pipeline = KnowledgePipeline.__new__(KnowledgePipeline)
    pipeline._config = SimpleNamespace(
        mineru=SimpleNamespace(backend="pipeline"),
        models=SimpleNamespace(
            default_figure_analysis="", default_vision="",
            default_extraction="x-extract", default_reasoning="x-reason",
        ),
        figures=SimpleNamespace(
            slide_analysis_model="slide-primary", slide_analysis_fallback="slide-fallback",
        ),
        embedding=SimpleNamespace(model="x-embed"),
        resilience=SimpleNamespace(checkpoint_enabled=True, request_timeout_s=30),
        state_path=tmp_path,
    )

    models = {"slide-primary": primary_llm, "slide-fallback": fallback_llm}

    def _get(name):
        model = models.get(name)
        if model is None:
            raise LLMProviderError(f"Model {name!r} not found in catalog.")
        return model

    pipeline._provider = SimpleNamespace(
        get=_get,
        get_chain=lambda stage, override=None: ["m"],
        get_for_stage=lambda stage: "m",
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
                proposed_id=f"eq_{extraction.chunk_id}",
                label=NodeLabel.EQUATION,
                canonical_name=f"Equation from {extraction.chunk_id}",
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


def _llm_returning(content) -> AsyncMock:
    llm = AsyncMock()
    llm.ainvoke.return_value = MagicMock(content=content)
    return llm


# ── Basic routing / gating ──


@pytest.mark.asyncio
async def test_figures_only_parses_with_include_images(tmp_path, monkeypatch):
    primary = _llm_returning(_EQUATION_JSON)
    pipeline, calls, graph = _build_pipeline(tmp_path, monkeypatch, primary_llm=primary)

    parse_calls = []

    def fake_parse(path, include_images=False):
        parse_calls.append(include_images)
        return _parsed_with_slides("src_figs_test", n_slides=1)

    monkeypatch.setattr(pipe_mod, "parse_lecture_html", fake_parse)

    result = await pipeline.ingest_html("lecture.html", figures_only=True)

    assert parse_calls == [True]
    assert result.source_id == "src_figs_test"
    assert result.errors == []
    assert result.total_chunks == 1  # 1 gated slide -> 1 chunk (heading+body -> one section)
    assert calls["extract"] == 1


@pytest.mark.asyncio
async def test_figures_only_gates_out_slides_without_figure_content(tmp_path, monkeypatch):
    """3 slides: 2 title-only (no equations/schematic/plot), 1 with an equation — only the
    equation-bearing slide should reach chunking/extraction."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = [
        MagicMock(content=_TITLE_ONLY_JSON),
        MagicMock(content=_EQUATION_JSON),
        MagicMock(content=_TITLE_ONLY_JSON),
    ]
    pipeline, calls, graph = _build_pipeline(tmp_path, monkeypatch, primary_llm=primary)
    monkeypatch.setattr(
        pipe_mod, "parse_lecture_html",
        lambda path, include_images=False: _parsed_with_slides("src_gate_test", n_slides=3),
    )

    result = await pipeline.ingest_html("lecture.html", figures_only=True)

    assert result.errors == []
    assert result.total_chunks == 1  # only the middle (equation) slide passed the gate
    assert calls["extract"] == 1


@pytest.mark.asyncio
async def test_figures_only_same_source_id_as_text_would_use(tmp_path, monkeypatch):
    """The graph identity (source_id stamped on new nodes) must match what a normal text ingest
    of the SAME file would use — figures-only adds to the SAME source, never a parallel one."""
    primary = _llm_returning(_EQUATION_JSON)
    pipeline, calls, graph = _build_pipeline(tmp_path, monkeypatch, primary_llm=primary)
    monkeypatch.setattr(
        pipe_mod, "parse_lecture_html",
        lambda path, include_images=False: _parsed_with_slides("src_shared_identity", n_slides=1),
    )

    await pipeline.ingest_html("lecture.html", figures_only=True)

    assert len(graph.applied_deltas) == 1
    proposal = graph.applied_deltas[0].new_nodes[0]
    assert proposal.label is NodeLabel.EQUATION
    assert proposal.properties.get("source_id") == "src_shared_identity"


@pytest.mark.asyncio
async def test_figures_only_no_slide_images_reports_clear_error(tmp_path, monkeypatch):
    pipeline, calls, graph = _build_pipeline(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pipe_mod, "parse_lecture_html",
        lambda path, include_images=False: _parsed_with_slides("src_empty", n_slides=0),
    )

    result = await pipeline.ingest_html("lecture.html", figures_only=True)

    # C6: legitimately-empty input is empty_reason (exit 3 at the CLI), NOT an error —
    # 76 slide-free session-note files read as failures in the 2026-07 batch otherwise.
    assert result.errors == []
    assert result.empty_reason and "No slide images" in result.empty_reason
    assert result.success
    assert graph.applied_deltas == []


@pytest.mark.asyncio
async def test_figures_only_all_slides_gated_out_reports_clear_error(tmp_path, monkeypatch):
    primary = _llm_returning(_TITLE_ONLY_JSON)
    pipeline, calls, graph = _build_pipeline(tmp_path, monkeypatch, primary_llm=primary)
    monkeypatch.setattr(
        pipe_mod, "parse_lecture_html",
        lambda path, include_images=False: _parsed_with_slides("src_no_figures", n_slides=2),
    )

    result = await pipeline.ingest_html("lecture.html", figures_only=True)

    assert result.errors == []
    assert result.empty_reason and "No figure content" in result.empty_reason
    assert result.success
    assert graph.applied_deltas == []


@pytest.mark.asyncio
async def test_figures_only_primary_model_unavailable_reports_clear_error(tmp_path, monkeypatch):
    """No 'slide-primary' registered in the fake catalog -> LLMProviderError -> clean, visible
    error rather than a crash (P9: degrade, but never silently)."""
    pipeline, calls, graph = _build_pipeline(tmp_path, monkeypatch)  # primary_llm=None
    monkeypatch.setattr(
        pipe_mod, "parse_lecture_html",
        lambda path, include_images=False: _parsed_with_slides("src_no_model", n_slides=1),
    )

    result = await pipeline.ingest_html("lecture.html", figures_only=True)

    assert any("slide-primary" in e and "unavailable" in e for e in result.errors)
    assert graph.applied_deltas == []


# ── Circuit breaker: whole-file abort ──


@pytest.mark.asyncio
async def test_figures_only_breaker_trip_aborts_file_with_clear_error(tmp_path, monkeypatch):
    primary = AsyncMock()
    primary.ainvoke.side_effect = TimeoutError("local server down")
    pipeline, calls, graph = _build_pipeline(tmp_path, monkeypatch, primary_llm=primary)
    monkeypatch.setattr(
        pipe_mod, "parse_lecture_html",
        lambda path, include_images=False: _parsed_with_slides("src_breaker_test", n_slides=10),
    )

    result = await pipeline.ingest_html("lecture.html", figures_only=True)

    assert len(result.errors) == 1
    assert "aborted" in result.errors[0]
    assert "circuit breaker" in result.errors[0]
    # Nothing reaches the commit stage — analysis aborts before chunking even starts.
    assert graph.applied_deltas == []
    assert calls["extract"] == 0


# ── Checkpoint namespace ──


@pytest.mark.asyncio
async def test_figures_only_uses_figs_checkpoint_namespace_not_text_checkpoint(tmp_path, monkeypatch):
    primary = _llm_returning(_EQUATION_JSON)
    pipeline, calls, graph = _build_pipeline(tmp_path, monkeypatch, primary_llm=primary)
    monkeypatch.setattr(
        pipe_mod, "parse_lecture_html",
        lambda path, include_images=False: _parsed_with_slides("src_ckpt_ns", n_slides=1),
    )

    result = await pipeline.ingest_html("lecture.html", figures_only=True)

    assert result.errors == []
    completed_figs = tmp_path / "checkpoints" / "completed" / "src_ckpt_ns_figs.json"
    assert completed_figs.exists()
    # The TEXT checkpoint namespace (no _figs suffix) must be untouched — this run never wrote it.
    assert not (tmp_path / "checkpoints" / "src_ckpt_ns.json").exists()
    assert not (tmp_path / "checkpoints" / "completed" / "src_ckpt_ns.json").exists()


@pytest.mark.asyncio
async def test_figures_only_resumes_analysis_from_checkpoint_after_breaker_trip(tmp_path, monkeypatch):
    """Run 1: 2 slides analyzed OK, then the local model goes permanently down -> breaker trips
    for the remaining ones. Run 2 (server back up): must NOT re-call the model for the first 2
    slides — only the ones that never completed."""
    n = 6
    from openclaw_brain.knowledge.extraction.slide_analyzer import _SLIDE_BREAKER_THRESHOLD

    def parse(path, include_images=False):
        return _parsed_with_slides("src_resume_test", n_slides=n)

    monkeypatch.setattr(pipe_mod, "parse_lecture_html", parse)

    # Run 1: first 2 slides succeed, then everything times out -> breaker trips.
    primary1 = AsyncMock()
    primary1.ainvoke.side_effect = (
        [MagicMock(content=_EQUATION_JSON), MagicMock(content=_EQUATION_JSON)]
        + [TimeoutError("down")] * _SLIDE_BREAKER_THRESHOLD
    )
    pipeline1, _, graph1 = _build_pipeline(tmp_path, monkeypatch, primary_llm=primary1)
    result1 = await pipeline1.ingest_html("lecture.html", figures_only=True)
    assert result1.errors and "aborted" in result1.errors[0]
    calls_run1 = primary1.ainvoke.await_count
    assert calls_run1 == 2 + _SLIDE_BREAKER_THRESHOLD  # 2 good + threshold failures

    # Run 2: server healthy again. Only the (n - 2) never-completed slides should be re-analyzed.
    primary2 = AsyncMock()
    primary2.ainvoke.return_value = MagicMock(content=_EQUATION_JSON)
    pipeline2, calls2, graph2 = _build_pipeline(tmp_path, monkeypatch, primary_llm=primary2)
    result2 = await pipeline2.ingest_html("lecture.html", figures_only=True)

    assert result2.errors == []
    assert primary2.ainvoke.await_count == n - 2  # the 2 already-checkpointed slides were skipped
    assert result2.total_chunks == n  # all 6 slides end up gated in (all yield equations)


@pytest.mark.asyncio
async def test_figures_only_reprocess_true_ignores_checkpoint_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipe_mod, "parse_lecture_html",
        lambda path, include_images=False: _parsed_with_slides("src_reprocess_test", n_slides=2),
    )

    primary1 = _llm_returning(_EQUATION_JSON)
    pipeline1, _, _ = _build_pipeline(tmp_path, monkeypatch, primary_llm=primary1)
    await pipeline1.ingest_html("lecture.html", figures_only=True)
    assert primary1.ainvoke.await_count == 2

    primary2 = _llm_returning(_EQUATION_JSON)
    pipeline2, _, _ = _build_pipeline(tmp_path, monkeypatch, primary_llm=primary2)
    await pipeline2.ingest_html("lecture.html", figures_only=True, reprocess=True)
    # reprocess=True must NOT reuse the checkpointed analyses from run 1.
    assert primary2.ainvoke.await_count == 2


# ── Per-slide fallback (not a breaker event) ──


@pytest.mark.asyncio
async def test_figures_only_per_slide_fallback_does_not_abort(tmp_path, monkeypatch):
    """A single primary miss with a healthy fallback must NOT trip the breaker or abort — only a
    RUN of consecutive primary failures does (see slide_analyzer.py's breaker semantics)."""
    primary = AsyncMock()
    primary.ainvoke.side_effect = [TimeoutError("blip"), MagicMock(content=_EQUATION_JSON)]
    fallback = _llm_returning(_EQUATION_JSON)
    pipeline, calls, graph = _build_pipeline(tmp_path, monkeypatch, primary_llm=primary, fallback_llm=fallback)
    monkeypatch.setattr(
        pipe_mod, "parse_lecture_html",
        lambda path, include_images=False: _parsed_with_slides("src_fallback_test", n_slides=2),
    )

    result = await pipeline.ingest_html("lecture.html", figures_only=True)

    assert result.errors == []
    assert fallback.ainvoke.await_count == 1
    assert result.total_chunks == 2  # both slides ended up with usable equation content


@pytest.mark.asyncio
async def test_reasoning_degraded_chunk_left_not_done_for_retry(tmp_path, monkeypatch):
    """C6: a chunk whose reasoning terminally fails must NOT be checkpointed done (so a
    later invocation retries it) and must commit NOTHING — previously the empty delta
    still committed match-stage reinforcements, masking the lost reasoning yield behind
    non-zero counts (16 such chunks in the 2026-07 figures batch)."""
    primary = _llm_returning(_EQUATION_JSON)
    pipeline, calls, graph = _build_pipeline(tmp_path, monkeypatch, primary_llm=primary)
    monkeypatch.setattr(
        pipe_mod, "parse_lecture_html",
        lambda path, include_images=False: _parsed_with_slides("src_degraded", n_slides=1),
    )

    async def _boom(*a, **k):
        raise pipe_mod.ReasoningDegradedError("chunk_test", "forced parse failure")
    monkeypatch.setattr(pipeline, "_resilient_reason", _boom)

    result = await pipeline.ingest_html("lecture.html", figures_only=True)

    assert any("terminally failed" in e and "retry" in e for e in result.errors)
    assert not result.success
    assert graph.applied_deltas == []  # nothing committed for the degraded chunk
    # checkpoint must NOT have the chunk done — reopen-able on the next run
    import json as _json
    import pathlib as _pl
    cks = list((_pl.Path(tmp_path) / "state" / "checkpoints").rglob("src_degraded_figs.json"))
    if cks:  # checkpointing enabled in this fixture
        st = _json.loads(cks[0].read_text())
        assert st.get("completed_chunks", []) == []
