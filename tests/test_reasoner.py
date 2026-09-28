"""Tests for GraphReasoner's prompt construction (reasoning/reasoner.py).

No LLM/Neo4j: `_build_prompt` is a pure string-builder given an already-computed
extraction/match_result/context/chunk_text — exercised directly. `reasoner.py::GraphReasoner
.reason()` itself has zero production callers (pipeline.py's `_resilient_reason` reimplements
the same flow using `_gather_context`/`_build_prompt` directly — see knowledge/README.md and
knowledge/reasoning/README.md), so this file focuses on the prompt-building logic that IS shared
with production, not the dead `reason()` orchestration wrapper.
"""

from __future__ import annotations

from types import SimpleNamespace

from openclaw_brain.knowledge.extraction.models import (
    ConceptMention,
    EquationMention,
    ExtractionResult,
    MatchedConcept,
    MatchedEquation,
    MatchedParameter,
    MatchResult,
    ParameterMention,
)
from openclaw_brain.knowledge.reasoning.reasoner import GraphReasoner


def _make_reasoner() -> GraphReasoner:
    config = SimpleNamespace(reasoning=SimpleNamespace(chunk_token_budget=500))
    return GraphReasoner(graph=None, config=config)


def _extraction(chunk_id: str = "chunk_1") -> ExtractionResult:
    return ExtractionResult(
        chunk_id=chunk_id,
        concepts=[ConceptMention(name="Cascode", description="A cascode topology description.")],
    )


def test_cross_document_linking_prefers_reinforced_edges_for_existing_relationship():
    """The CROSS-DOCUMENT LINKING instruction must tell the model to use reinforced_edges (not
    new_edges) when the graph context already shows the same relationship — otherwise
    apply_delta's new_edges MERGE resets reinforcement_count/evidence_sources back to 1 on an
    edge that had already accumulated confirmation from prior sources (the duplicate-edge
    pressure defect)."""
    reasoner = _make_reasoner()
    extraction = _extraction()
    match_result = MatchResult(
        chunk_id="chunk_1",
        matched=[MatchedConcept(mention=extraction.concepts[0], existing_node_id="cascode_x")],
    )

    prompt = reasoner._build_prompt(extraction, match_result, "(no context)", "chunk text")

    assert "CROSS-DOCUMENT LINKING" in prompt
    assert "reinforced_edges" in prompt
    # names the schema fields a reinforced_edges entry needs, for the raw-JSON local-model
    # fallback path (which has no structured-output schema visibility)
    assert "confirming_chunk_ids" in prompt
    # explicitly names the failure mode being avoided, not just the field name
    assert "reinforcement_count" in prompt


def test_cross_document_linking_still_instructs_new_edges_for_genuinely_new_relationships():
    """The tightened instruction must still tell the model to propose new_edges when the
    relationship is NOT already in the graph context — reinforcement must not crowd out
    genuinely new cross-document edges."""
    reasoner = _make_reasoner()
    extraction = _extraction()
    match_result = MatchResult(
        chunk_id="chunk_1",
        matched=[MatchedConcept(mention=extraction.concepts[0], existing_node_id="cascode_x")],
    )

    prompt = reasoner._build_prompt(extraction, match_result, "(no context)", "chunk text")

    assert "new_edges" in prompt
    assert "existing_node_id" in prompt
    assert "no such relationship is listed" in prompt.lower()


def test_cross_document_linking_absent_when_no_matches():
    """REGRESSION: a chunk with no matched concepts must not get the cross-document linking
    instruction at all (unchanged from before this fix)."""
    reasoner = _make_reasoner()
    extraction = _extraction()
    match_result = MatchResult(chunk_id="chunk_1", matched=[])

    prompt = reasoner._build_prompt(extraction, match_result, "(no context)", "chunk text")

    assert "CROSS-DOCUMENT LINKING" not in prompt
    assert "reinforced_edges" not in prompt


def test_cross_document_linking_names_all_matched_ids():
    """REGRESSION: every matched existing_node_id must still be named in the instruction (the
    tightened wording must not drop the multi-match case)."""
    reasoner = _make_reasoner()
    extraction = ExtractionResult(
        chunk_id="chunk_1",
        concepts=[
            ConceptMention(name="Cascode", description="A cascode topology description."),
            ConceptMention(name="Current Mirror", description="A current mirror description."),
        ],
    )
    match_result = MatchResult(
        chunk_id="chunk_1",
        matched=[
            MatchedConcept(mention=extraction.concepts[0], existing_node_id="cascode_x"),
            MatchedConcept(mention=extraction.concepts[1], existing_node_id="mirror_y"),
        ],
    )

    prompt = reasoner._build_prompt(extraction, match_result, "(no context)", "chunk text")

    assert "cascode_x" in prompt
    assert "mirror_y" in prompt


# ── F6: matched equations/parameters in the MATCH RESULTS listing ──────────────────────────
#
# Equations/parameters previously never appeared in match_result at all (matcher.py never
# looked at them), so the reasoner had zero "already exists" signal for these two labels —
# see knowledge/reasoning/README.md "Known defects" #2. These tests cover the new listing
# slot ONLY (matcher.py's own matching logic is covered by test_matcher_equations_parameters.py).


def test_match_results_lists_matched_equations():
    reasoner = _make_reasoner()
    extraction = ExtractionResult(
        chunk_id="chunk_1",
        equations=[EquationMention(latex="GBW = g_m/(2*pi*C_L)")],
    )
    match_result = MatchResult(
        chunk_id="chunk_1",
        matched_equations=[
            MatchedEquation(mention=extraction.equations[0], existing_node_id="eq_gbw"),
        ],
    )

    prompt = reasoner._build_prompt(extraction, match_result, "(no context)", "chunk text")

    assert "Already in graph (equations)" in prompt
    assert "eq_gbw" in prompt
    assert "GBW = g_m/(2*pi*C_L)" in prompt


def test_match_results_lists_matched_parameters():
    reasoner = _make_reasoner()
    extraction = ExtractionResult(
        chunk_id="chunk_1",
        parameters=[ParameterMention(symbol="GBW", name="Gain-Bandwidth Product", units="Hz")],
    )
    match_result = MatchResult(
        chunk_id="chunk_1",
        matched_parameters=[
            MatchedParameter(mention=extraction.parameters[0], existing_node_id="p_gbw"),
        ],
    )

    prompt = reasoner._build_prompt(extraction, match_result, "(no context)", "chunk text")

    assert "Already in graph (parameters)" in prompt
    assert "p_gbw" in prompt
    assert "Gain-Bandwidth Product" in prompt


def test_match_results_omits_equation_parameter_sections_when_empty():
    """REGRESSION: a chunk with no matched equations/parameters must not get the new
    sections at all (mirrors the concept-side absent-when-empty convention already
    covered by test_cross_document_linking_absent_when_no_matches)."""
    reasoner = _make_reasoner()
    extraction = _extraction()
    match_result = MatchResult(chunk_id="chunk_1", matched=[])

    prompt = reasoner._build_prompt(extraction, match_result, "(no context)", "chunk text")

    assert "Already in graph (equations)" not in prompt
    assert "Already in graph (parameters)" not in prompt


def test_match_results_equations_and_parameters_alongside_matched_concepts():
    """Integration: all three 'Already in graph' blocks can coexist in one prompt."""
    reasoner = _make_reasoner()
    extraction = ExtractionResult(
        chunk_id="chunk_1",
        concepts=[ConceptMention(name="Cascode", description="A cascode topology description.")],
        equations=[EquationMention(latex="GBW = g_m/(2*pi*C_L)")],
        parameters=[ParameterMention(symbol="GBW", name="Gain-Bandwidth Product", units="Hz")],
    )
    match_result = MatchResult(
        chunk_id="chunk_1",
        matched=[MatchedConcept(mention=extraction.concepts[0], existing_node_id="cascode_x")],
        matched_equations=[
            MatchedEquation(mention=extraction.equations[0], existing_node_id="eq_gbw"),
        ],
        matched_parameters=[
            MatchedParameter(mention=extraction.parameters[0], existing_node_id="p_gbw"),
        ],
    )

    prompt = reasoner._build_prompt(extraction, match_result, "(no context)", "chunk text")

    assert "Already in graph:" in prompt
    assert "cascode_x" in prompt
    assert "Already in graph (equations)" in prompt
    assert "eq_gbw" in prompt
    assert "Already in graph (parameters)" in prompt
    assert "p_gbw" in prompt
