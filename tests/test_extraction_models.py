"""Unit tests for extraction pipeline data models."""

from openclaw_brain.knowledge.extraction.models import (
    AmbiguousMatch,
    ChunkerResult,
    ConceptMention,
    EquationMention,
    ExtractionResult,
    IngestResult,
    MatchedConcept,
    MatchedEquation,
    MatchedParameter,
    MatchResult,
    ParameterMention,
    PipelineProgress,
    SourceChunkInfo,
)


def test_source_chunk_info():
    chunk = SourceChunkInfo(
        chunk_id="chunk_abc",
        source_id="src_123",
        text="The transconductance gm...",
        pages="5-6",
        section_title="5.1 MOSFET Small-Signal Model",
    )
    assert chunk.token_estimate == 0  # default


def test_chunker_result():
    result = ChunkerResult(
        source_id="src_123",
        title="Sedra Smith Ch5",
        chunks=[
            SourceChunkInfo(chunk_id="c1", source_id="src_123", text="text1"),
            SourceChunkInfo(chunk_id="c2", source_id="src_123", text="text2"),
        ],
    )
    assert len(result.chunks) == 2


def test_extraction_result():
    result = ExtractionResult(
        chunk_id="c1",
        concepts=[
            ConceptMention(name="Transconductance", description="Small-signal parameter gm = 2ID/Vov relating gate voltage to drain current"),
        ],
        equations=[],
        parameters=[],
        raw_edges=[],
    )
    assert len(result.concepts) == 1


def test_match_result():
    result = MatchResult(
        chunk_id="c1",
        matched=[
            MatchedConcept(
                mention=ConceptMention(name="MOSFET", description="Metal-oxide-semiconductor field-effect transistor used in integrated circuits"),
                existing_node_id="concept_mosfet",
                similarity=0.95,
            ),
        ],
        new_concepts=[ConceptMention(name="Loop Gain", description="Total gain around a feedback loop, product of forward and feedback gains")],
        ambiguous=[],
    )
    assert len(result.matched) == 1
    assert len(result.new_concepts) == 1


def test_match_result_equation_parameter_defaults_are_empty():
    # F6 added 4 new fields to MatchResult; every pre-F6 call site constructs a MatchResult
    # without them, so the defaults must be empty lists (never None) to stay backward compatible.
    result = MatchResult(chunk_id="c1")
    assert result.matched_equations == []
    assert result.new_equations == []
    assert result.matched_parameters == []
    assert result.new_parameters == []


def test_match_result_matched_equation():
    eq_mention = EquationMention(latex="GBW = g_m/(2*pi*C_L)")
    result = MatchResult(
        chunk_id="c1",
        matched_equations=[MatchedEquation(mention=eq_mention, existing_node_id="eq_gbw")],
    )
    assert len(result.matched_equations) == 1
    m = result.matched_equations[0]
    assert m.existing_node_id == "eq_gbw"
    assert m.similarity == 1.0
    assert m.node_label == "Equation"
    assert m.node_id_field == "equation_id"
    assert m.mention.latex == "GBW = g_m/(2*pi*C_L)"


def test_match_result_matched_parameter():
    # This field previously existed as list[dict[str, Any]] but was declared, never written
    # or read anywhere in the repo (2026-07-10 architecture survey). F6 repurposes the same
    # field name with real semantics -- this test locks in the new typed shape.
    p_mention = ParameterMention(symbol="GBW", name="Gain-Bandwidth Product", units="Hz")
    result = MatchResult(
        chunk_id="c1",
        matched_parameters=[MatchedParameter(mention=p_mention, existing_node_id="p_gbw")],
    )
    assert len(result.matched_parameters) == 1
    m = result.matched_parameters[0]
    assert m.existing_node_id == "p_gbw"
    assert m.similarity == 1.0
    assert m.node_label == "Parameter"
    assert m.node_id_field == "parameter_id"
    assert m.mention.name == "Gain-Bandwidth Product"


def test_match_result_new_equations_and_parameters():
    result = MatchResult(
        chunk_id="c1",
        new_equations=[EquationMention(latex="A = B")],
        new_parameters=[ParameterMention(symbol="Av", name="Gain", units="V/V")],
    )
    assert len(result.new_equations) == 1
    assert len(result.new_parameters) == 1


def test_pipeline_progress():
    p = PipelineProgress(stage="extract", chunk_index=3, total_chunks=10)
    assert p.percent == 30.0


def test_pipeline_progress_zero():
    p = PipelineProgress(stage="chunk", chunk_index=0, total_chunks=0)
    assert p.percent == 0.0


def test_ingest_result_success():
    r = IngestResult(source_id="src_1", title="Test", new_nodes=5, new_edges=3)
    assert r.success
    assert "5 new concepts" in r.summary()
    assert "3 new connections" in r.summary()


def test_ingest_result_with_errors():
    r = IngestResult(source_id="src_1", title="Test", errors=["chunk 3 failed"])
    assert not r.success
    assert "1 errors" in r.summary()


def test_ingest_result_summary_all():
    r = IngestResult(
        source_id="src_1",
        title="Ch5",
        total_chunks=10,
        new_nodes=8,
        updated_nodes=2,
        new_edges=12,
        reinforced_edges=5,
        insights=3,
    )
    summary = r.summary()
    assert "10 chunks" in summary
    assert "8 new concepts" in summary
    assert "2 updated" in summary
    assert "12 new connections" in summary
    assert "5 reinforced" in summary
    assert "3 insights" in summary
