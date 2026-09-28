"""Tests for concept matcher (graph-based matching)."""

import pytest
from tests.conftest import require_live_graph

from openclaw_brain.config import load_config
from openclaw_brain.knowledge.extraction.models import (
    ConceptMention,
    ExtractionResult,
)
from openclaw_brain.knowledge.graph.schema import NodeLabel
from openclaw_brain.knowledge.graph.store import GraphStore
from openclaw_brain.knowledge.reasoning.matcher import ConceptMatcher, _name_similarity


# ── Unit tests (no Neo4j needed) ──


def test_name_similarity_exact():
    assert _name_similarity("MOSFET", "MOSFET") == 1.0


def test_name_similarity_case_insensitive():
    assert _name_similarity("mosfet", "MOSFET") == 1.0


def test_name_similarity_partial():
    sim = _name_similarity("common source amplifier", "common source")
    assert 0.5 < sim < 1.0


def test_name_similarity_no_overlap():
    assert _name_similarity("MOSFET", "resistor") < 0.3  # Low but non-zero due to LCS character overlap


def test_name_similarity_empty():
    assert _name_similarity("", "MOSFET") == 0.0
    assert _name_similarity("MOSFET", "") == 0.0


def test_name_similarity_abbreviation_expansion():
    # FD-SOI should match Fully Depleted Silicon On Insulator
    assert _name_similarity("FD-SOI", "Fully Depleted Silicon On Insulator") > 0.9
    # AER should match Address Event Representation
    assert _name_similarity("AER", "Address Event Representation") > 0.9


def test_name_similarity_snake_vs_title():
    assert _name_similarity("subthreshold_regime", "Subthreshold Regime") > 0.9


def test_name_similarity_near_duplicates():
    # These should be detected as very similar
    sim = _name_similarity("Subthreshold Regime", "Subthreshold Operation")
    assert sim > 0.4  # Partial overlap — should be flagged as ambiguous


# ── Integration tests (Neo4j) ──


@pytest.fixture
async def graph():
    require_live_graph()
    config = load_config()
    g = GraphStore(config.neo4j)
    try:
        await g.connect()
    except Exception:
        pytest.skip("Neo4j not available")

    # Seed test concepts. These run against the LIVE graph (the matcher searches the whole corpus),
    # so the deterministic ones use a SENTINEL name no ingested source could collide with — real
    # circuit terms (MOSFET, Feedback Factor) now exist in the production graph and would compete.
    for cid, name in [
        ("test_match_mosfet", "MOSFET"),
        ("test_match_cs_amp", "Common-Source Amplifier"),
        ("test_match_gm", "Transconductance"),
        ("test_match_sentinel", "Zzylene Sentinel Concept Qx"),
    ]:
        await g.merge_node(
            NodeLabel.CONCEPT, "concept_id", cid,
            {"concept_id": cid, "canonical_name": name, "domain": "analog_circuits"},
        )

    yield g

    # Cleanup
    async with await g._session() as session:
        await session.run(
            "MATCH (n:Concept) WHERE n.concept_id STARTS WITH 'test_match_' DETACH DELETE n"
        )
    await g.close()


@pytest.mark.asyncio
async def test_matcher_finds_exact_match(graph: GraphStore):
    # Exact-match MECHANIC: a mention whose name equals a seeded node must resolve to THAT node.
    # Uses the sentinel (Tier-0 name-exact, no production collision) so the assertion is deterministic.
    matcher = ConceptMatcher(graph)
    extraction = ExtractionResult(
        chunk_id="test_chunk",
        concepts=[ConceptMention(name="Zzylene Sentinel Concept Qx",
                                 description="A sentinel concept used only to test exact-match resolution.")],
    )
    result = await matcher.match(extraction)
    assert len(result.matched) == 1
    assert len(result.new_concepts) == 0
    assert result.matched[0].existing_node_id == "test_match_sentinel"


@pytest.mark.asyncio
async def test_matcher_detects_new_concept(graph: GraphStore):
    # Novel-concept MECHANIC: a mention with no name/semantic match anywhere in the corpus is NEW.
    # Uses a deliberately meaningless term so no ingested circuit concept can resolve it.
    matcher = ConceptMatcher(graph)
    novel = "Florble Quintessence Widget Vorp"
    extraction = ExtractionResult(
        chunk_id="test_chunk",
        concepts=[ConceptMention(name=novel,
                                 description="A nonsensical placeholder term with no semiconductor "
                                             "meaning, used only to test novel-concept detection.")],
    )
    result = await matcher.match(extraction)
    assert len(result.new_concepts) == 1
    assert result.new_concepts[0].name == novel


@pytest.mark.asyncio
async def test_matcher_mixed_results(graph: GraphStore):
    matcher = ConceptMatcher(graph)
    extraction = ExtractionResult(
        chunk_id="test_chunk",
        concepts=[
            ConceptMention(name="MOSFET", description="Metal-oxide-semiconductor field-effect transistor for integrated circuits"),
            ConceptMention(name="Loop Gain", description="Total gain around a feedback loop in amplifier circuits"),
        ],
    )
    result = await matcher.match(extraction)
    assert len(result.matched) + len(result.new_concepts) + len(result.ambiguous) == 2
