"""Unit tests for opt-in chunk concurrency + the multi-pass reprocess flag in KnowledgePipeline.ingest.

No real LLM / Neo4j: the per-chunk stages are stubbed and instrumented to observe in-flight
concurrency, that every chunk is committed, that counts sum, and that reprocess re-runs done chunks.
See experiments/CONCURRENT_INGEST_DESIGN.md.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import openclaw_brain.knowledge.pipeline as pipe_mod
from openclaw_brain.config import ResilienceConfig
from openclaw_brain.knowledge.extraction.models import ChunkerResult, SourceChunkInfo
from openclaw_brain.knowledge.graph.schema import GraphDelta, NodeLabel, NodeProposal
from openclaw_brain.knowledge.pipeline import KnowledgePipeline

_COUNTS = {"new_nodes": 1, "updated_nodes": 0, "new_edges": 2, "reinforced_edges": 0, "insights": 0}


def _make_chunks(n: int) -> list[SourceChunkInfo]:
    return [
        SourceChunkInfo(
            chunk_id=f"chunk_{i}", source_id="src_test", text=f"body {i}",
            pages="1", section_title=f"S{i}",
        )
        for i in range(n)
    ]


def _build_pipeline(tmp_path, monkeypatch, n_chunks: int):
    """A pipeline whose stages are stubbed; returns (pipeline, telemetry)."""
    tele = {"inflight": 0, "max_inflight": 0, "processed": []}

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

    # instrumented per-chunk extract: observe overlap
    async def fake_extract(text, chunk_id, chain):
        tele["inflight"] += 1
        tele["max_inflight"] = max(tele["max_inflight"], tele["inflight"])
        try:
            await asyncio.sleep(0.02)
            return SimpleNamespace(entities=[], concepts=[], chunk_id=chunk_id)
        finally:
            tele["inflight"] -= 1

    async def fake_reason(extraction, match_result, text, chain, errors=None):
        tele["processed"].append(extraction.chunk_id)
        return GraphDelta()

    async def _anoop(*a, **k):
        return None

    pipeline._resilient_extract = fake_extract
    pipeline._resilient_reason = fake_reason
    pipeline._register_chunk = _anoop
    pipeline._register_source = _anoop
    pipeline._embed_new_nodes = _anoop
    pipeline._persist_mineru_outputs = lambda *a, **k: None
    pipeline._persist_pdf = lambda *a, **k: None

    # module-level stage boundaries
    monkeypatch.setattr(pipe_mod, "mineru_parse_pdf",
                        lambda path, backend=None: SimpleNamespace(figures=[], output_dir=str(tmp_path)))
    monkeypatch.setattr(pipe_mod, "chunk_structured",
                        lambda parsed, figure_analyses=None, max_tokens=1500: ChunkerResult(
                            source_id="src_test", title="T", chunks=_make_chunks(n_chunks), checksum="ck"))
    monkeypatch.setattr(pipe_mod, "verify_grounding", lambda extraction, text: extraction)
    monkeypatch.setattr(pipe_mod, "summarize_paper", _anoop)
    return pipeline, tele


def _aval(v):
    async def _coro():
        return v
    return _coro()


@pytest.mark.asyncio
async def test_sequential_default_is_strictly_serial(tmp_path, monkeypatch):
    pipeline, tele = _build_pipeline(tmp_path, monkeypatch, n_chunks=6)
    result = await pipeline.ingest("paper.pdf")  # chunk_concurrency defaults to 1
    assert tele["max_inflight"] == 1
    assert len(tele["processed"]) == 6
    assert result.new_nodes == 6 and result.new_edges == 12  # counts summed over 6 chunks


@pytest.mark.asyncio
async def test_concurrent_bounded_and_overlaps(tmp_path, monkeypatch):
    pipeline, tele = _build_pipeline(tmp_path, monkeypatch, n_chunks=8)
    result = await pipeline.ingest("paper.pdf", chunk_concurrency=4)
    assert 1 < tele["max_inflight"] <= 4          # genuinely overlapped, never exceeds the cap
    assert len(tele["processed"]) == 8            # every chunk processed
    assert result.new_nodes == 8 and result.new_edges == 16


@pytest.mark.asyncio
async def test_concurrent_marks_all_chunks_done(tmp_path, monkeypatch):
    pipeline, tele = _build_pipeline(tmp_path, monkeypatch, n_chunks=5)
    await pipeline.ingest("paper.pdf", chunk_concurrency=3)
    # checkpoint was archived on completion → re-create points at the live dir; all 5 were committed
    assert sorted(tele["processed"]) == [f"chunk_{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_reprocess_reruns_already_done_chunk(tmp_path, monkeypatch):
    from openclaw_brain.knowledge.checkpoint import PipelineCheckpoint
    # Pre-seed an ACTIVE checkpoint with chunk 0 already done.
    ckpt = PipelineCheckpoint(tmp_path / "checkpoints", "src_test")
    ckpt.initialize("T", 3)
    ckpt.mark_chunk_done(0, dict(_COUNTS))

    # Control: a normal resume skips chunk 0.
    pipeline, tele = _build_pipeline(tmp_path, monkeypatch, n_chunks=3)
    await pipeline.ingest("paper.pdf")
    assert "chunk_0" not in tele["processed"]
    assert sorted(tele["processed"]) == ["chunk_1", "chunk_2"]

    # Re-seed and run with reprocess=True → chunk 0 IS re-run (multi-pass refine).
    ckpt2 = PipelineCheckpoint(tmp_path / "checkpoints", "src_test")
    ckpt2.initialize("T", 3)
    ckpt2.mark_chunk_done(0, dict(_COUNTS))
    pipeline2, tele2 = _build_pipeline(tmp_path, monkeypatch, n_chunks=3)
    await pipeline2.ingest("paper.pdf", reprocess=True)
    assert sorted(tele2["processed"]) == ["chunk_0", "chunk_1", "chunk_2"]


# ── KnowledgePipeline._resilient_reason — per-chunk crash isolation ─────────────────────────
#
# Unlike _build_pipeline above (which always fakes _resilient_reason itself), these tests
# exercise the REAL method to verify its own guarded fallback behavior: a chunk whose
# reasoning output can't be parsed into a valid GraphDelta must degrade to an empty GraphDelta
# instead of raising and crashing the whole multi-chunk ingest (spec: crash-isolation fix for
# claude-sonnet-5 structured-output quirks).


def _make_reason_pipeline():
    """Minimal KnowledgePipeline exposing the real, unfaked `_resilient_reason`."""
    pipeline = KnowledgePipeline.__new__(KnowledgePipeline)
    pipeline._config = SimpleNamespace(resilience=ResilienceConfig())
    pipeline._auth_refresh = None
    pipeline._reasoner = SimpleNamespace(
        _gather_context=lambda match_result: _aval("(no context)"),
        _build_prompt=lambda extraction, match_result, context, chunk_text: "prompt",
    )
    return pipeline


class _RaisingStructured:
    """A `.with_structured_output(...)`-wrapped runnable that always rejects the payload —
    simulates claude-sonnet-5 omitting a required-at-the-time field under tool-calling."""

    async def ainvoke(self, messages):
        raise ValueError("primary structured output rejected the payload")


class _UnparseableModel:
    """Structured output fails AND the raw free-text fallback isn't parseable JSON either —
    the case the guard must catch."""

    model_name = "fake-unparseable"

    def with_structured_output(self, schema):
        return _RaisingStructured()

    async def ainvoke(self, messages):
        return SimpleNamespace(content="I cannot comply with a rigid schema here — prose only.")


class _CleanStructuredModel:
    """Structured output succeeds on the first attempt — the guard must never engage."""

    model_name = "fake-clean"

    def __init__(self, delta):
        self._delta = delta

    def with_structured_output(self, schema):
        outer = self

        class _Structured:
            async def ainvoke(self, messages):
                return outer._delta

        return _Structured()


@pytest.mark.asyncio
async def test_resilient_reason_unparseable_output_raises_degraded_for_retry():
    """C6 (2026-07-22, 5b29a22): unparseable reasoning output RAISES ReasoningDegradedError
    instead of silently degrading to an empty GraphDelta. The old empty-delta path committed
    match-stage reinforcements, marked the chunk done, and permanently lost the reasoning
    yield (16 chunks recovered in the C6 incident); raising leaves the chunk un-done so a
    rerun retries it."""
    from openclaw_brain.knowledge.pipeline import ReasoningDegradedError

    pipeline = _make_reason_pipeline()
    extraction = SimpleNamespace(chunk_id="chunk_bad")
    match_result = SimpleNamespace(matched=[])

    with pytest.raises(ReasoningDegradedError) as exc:
        await pipeline._resilient_reason(
            extraction, match_result, "chunk text", [_UnparseableModel()],
        )
    assert exc.value.chunk_id == "chunk_bad"


@pytest.mark.asyncio
async def test_resilient_reason_clean_structured_output_unaffected_by_guard():
    """REGRESSION: a model whose structured output succeeds immediately still returns that
    result directly — the crash-isolation guard only wraps the raw+normalize fallback."""
    pipeline = _make_reason_pipeline()
    extraction = SimpleNamespace(chunk_id="chunk_good")
    match_result = SimpleNamespace(matched=[])
    expected = GraphDelta(new_nodes=[], new_edges=[])

    delta = await pipeline._resilient_reason(
        extraction, match_result, "chunk text", [_CleanStructuredModel(expected)],
    )

    assert delta is expected


@pytest.mark.asyncio
async def test_resilient_reason_terminal_failure_raises_with_structured_cause():
    """C6 evolution of the errors-list defect fix: terminal reasoning failure now surfaces as
    ReasoningDegradedError carrying chunk_id + the terminal cause (the ingest loop's handler
    records it into IngestResult.errors and checkpoint.record_error, and does NOT mark the
    chunk done — covered at loop level by test_pipeline_figures_only)."""
    from openclaw_brain.knowledge.pipeline import ReasoningDegradedError

    pipeline = _make_reason_pipeline()
    extraction = SimpleNamespace(chunk_id="chunk_bad")
    match_result = SimpleNamespace(matched=[])
    errors: list[str] = []

    with pytest.raises(ReasoningDegradedError) as exc:
        await pipeline._resilient_reason(
            extraction, match_result, "chunk text", [_UnparseableModel()], errors=errors,
        )
    msg = str(exc.value)
    assert "chunk_bad" in msg
    assert "reason" in msg.lower()


@pytest.mark.asyncio
async def test_resilient_reason_clean_output_does_not_touch_errors_list():
    """REGRESSION: a model whose structured output succeeds must not append anything to the
    errors list — the new bookkeeping only engages on genuine terminal failure."""
    pipeline = _make_reason_pipeline()
    extraction = SimpleNamespace(chunk_id="chunk_good")
    match_result = SimpleNamespace(matched=[])
    expected = GraphDelta(new_nodes=[], new_edges=[])
    errors: list[str] = []

    delta = await pipeline._resilient_reason(
        extraction, match_result, "chunk text", [_CleanStructuredModel(expected)], errors=errors,
    )

    assert delta is expected
    assert errors == []


@pytest.mark.asyncio
async def test_ingest_surfaces_resilient_reason_errors_into_result(tmp_path, monkeypatch):
    """`ingest()` must pass its own `result.errors` list into `_resilient_reason` BY REFERENCE
    so a terminal reasoning failure recorded there for one chunk survives into the final
    IngestResult, not just a log line. Uses a stub that appends via the `errors=` kwarg it
    receives, isolating the ingest()-level wiring from _resilient_reason's own except-branch
    logic (covered directly above)."""
    pipeline, tele = _build_pipeline(tmp_path, monkeypatch, n_chunks=3)

    async def fake_reason_with_one_failure(extraction, match_result, text, chain, errors=None):
        tele["processed"].append(extraction.chunk_id)
        if extraction.chunk_id == "chunk_1" and errors is not None:
            errors.append(f"Chunk {extraction.chunk_id}: reasoning terminally failed (simulated)")
        return GraphDelta()

    pipeline._resilient_reason = fake_reason_with_one_failure
    result = await pipeline.ingest("paper.pdf")

    assert len(result.errors) == 1
    assert "chunk_1" in result.errors[0]
    assert "reasoning terminally failed" in result.errors[0]
    assert result.success is False
    # degrade-don't-crash is preserved — only visibility changed, all chunks still processed
    assert sorted(tele["processed"]) == ["chunk_0", "chunk_1", "chunk_2"]


# ── KnowledgePipeline._embed_new_nodes — best-effort embedding, error visibility ────────────
#
# encode_batch / store_embedding failures must stay best-effort (never abort ingestion) but,
# unlike before this fix, must also be visible in the caller-supplied errors list — a node that
# commits with no embedding silently falls out of Tier 1/2 embedding-based matching forever.


class _FakeEmbedGraph:
    """Minimal GraphStore stand-in exposing only what _embed_new_nodes calls."""

    def __init__(self, store_embedding_fails_for: set[str] | None = None):
        self.stored: list[tuple] = []
        self._fail_for = store_embedding_fails_for or set()

    def _id_field_for_label(self, label):
        return "concept_id"

    async def store_embedding(self, label, id_field, id_value, vector):
        if id_value in self._fail_for:
            raise RuntimeError(f"store_embedding boom for {id_value}")
        self.stored.append((label, id_field, id_value, vector))


def _make_embed_pipeline(graph=None):
    pipeline = KnowledgePipeline.__new__(KnowledgePipeline)
    pipeline._config = SimpleNamespace(embedding=SimpleNamespace(model="x-embed"))
    pipeline._graph = graph if graph is not None else _FakeEmbedGraph()
    return pipeline


def _node_proposal(pid: str, name: str = "Threshold Voltage") -> NodeProposal:
    return NodeProposal(
        proposed_id=pid, label=NodeLabel.CONCEPT, canonical_name=name,
        description="d", domain="analog_circuits", confidence=0.8,
        evidence_chunk_ids=["chunk_0"], reasoning="test fixture",
    )


@pytest.mark.asyncio
async def test_embed_new_nodes_batch_failure_appends_to_errors(monkeypatch):
    pipeline = _make_embed_pipeline()
    proposals = [_node_proposal("p1")]
    errors: list[str] = []

    def _boom(texts, model):
        raise RuntimeError("embedding server unreachable")

    monkeypatch.setattr(pipe_mod, "encode_batch", _boom)
    await pipeline._embed_new_nodes(proposals, errors=errors)

    assert len(errors) == 1
    assert "Embedding batch failed" in errors[0]
    assert pipeline._graph.stored == []


@pytest.mark.asyncio
async def test_embed_new_nodes_store_embedding_failure_appends_to_errors(monkeypatch):
    graph = _FakeEmbedGraph(store_embedding_fails_for={"p_bad"})
    pipeline = _make_embed_pipeline(graph)
    proposals = [_node_proposal("p_ok", "OK Concept"), _node_proposal("p_bad", "Bad Concept")]
    errors: list[str] = []

    monkeypatch.setattr(pipe_mod, "encode_batch", lambda texts, model: [[0.1, 0.2] for _ in texts])
    await pipeline._embed_new_nodes(proposals, errors=errors)

    assert len(errors) == 1
    assert "p_bad" in errors[0]
    # best-effort semantics preserved: the OTHER node still got its embedding stored
    assert len(graph.stored) == 1 and graph.stored[0][2] == "p_ok"


@pytest.mark.asyncio
async def test_embed_new_nodes_success_does_not_touch_errors(monkeypatch):
    pipeline = _make_embed_pipeline()
    proposals = [_node_proposal("p1")]
    errors: list[str] = []

    monkeypatch.setattr(pipe_mod, "encode_batch", lambda texts, model: [[0.1, 0.2]])
    await pipeline._embed_new_nodes(proposals, errors=errors)

    assert errors == []
    assert len(pipeline._graph.stored) == 1


@pytest.mark.asyncio
async def test_embed_new_nodes_errors_none_default_does_not_raise(monkeypatch):
    """Backward-compat: omitting errors= (as ingest_chunk's path does) must not raise even when
    embedding fails — the append is simply skipped, matching the prior (log-only) behavior."""
    pipeline = _make_embed_pipeline()
    proposals = [_node_proposal("p1")]

    def _boom(texts, model):
        raise RuntimeError("embedding server unreachable")

    monkeypatch.setattr(pipe_mod, "encode_batch", _boom)
    await pipeline._embed_new_nodes(proposals)  # no errors kwarg — must not raise
