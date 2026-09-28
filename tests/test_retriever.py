"""Tests for the unified retriever."""

import pytest
from tests.conftest import require_live_graph

from openclaw_brain.config import load_config
from openclaw_brain.knowledge.graph.schema import NodeLabel
from openclaw_brain.knowledge.graph.store import GraphStore
from openclaw_brain.memory.models import MemoryEntry, MemoryLayer, PromotionTier
from openclaw_brain.memory.store import MemoryStore
from openclaw_brain.retriever import GraphContext, UnifiedContext, UnifiedRetriever


# ── Unit tests ──


def test_graph_context_empty():
    ctx = GraphContext()
    assert ctx.total_concepts_found == 0
    assert ctx.concepts == []


def test_unified_context_format_empty():
    ctx = UnifiedContext()
    assert ctx.format() == ""


def test_unified_context_format_with_concepts():
    ctx = UnifiedContext(
        graph=GraphContext(
            concepts=[
                {"canonical_name": "MOSFET", "description": "Metal-oxide semiconductor FET"},
            ],
            total_concepts_found=1,
        ),
    )
    text = ctx.format()
    assert "MOSFET" in text
    assert "Related Knowledge" in text


def test_unified_context_format_truncation():
    ctx = UnifiedContext(
        graph=GraphContext(
            concepts=[{"canonical_name": f"Concept_{i}"} for i in range(100)],
            total_concepts_found=100,
        ),
    )
    text = ctx.format(token_budget=50)  # Very small budget
    assert "truncated" in text


# ── Integration tests ──


@pytest.fixture
async def graph():
    require_live_graph()
    config = load_config()
    g = GraphStore(config.neo4j)
    try:
        await g.connect()
    except Exception:
        pytest.skip("Neo4j not available")

    # Seed test data
    await g.merge_node(
        NodeLabel.CONCEPT, "concept_id", "test_ret_mosfet",
        {"concept_id": "test_ret_mosfet", "canonical_name": "MOSFET", "domain": "semiconductor"},
    )
    await g.merge_node(
        NodeLabel.CONCEPT, "concept_id", "test_ret_gain",
        {"concept_id": "test_ret_gain", "canonical_name": "Voltage Gain", "domain": "analog_circuits"},
    )

    yield g

    async with await g._session() as session:
        await session.run("MATCH (n:Concept) WHERE n.concept_id STARTS WITH 'test_ret_' DETACH DELETE n")
        await session.run("MATCH (n:Memory) WHERE n.memory_id STARTS WITH 'test_ret_' DETACH DELETE n")
    await g.close()


@pytest.mark.asyncio
async def test_retriever_graph_only(graph: GraphStore):
    config = load_config()
    mem_store = MemoryStore(graph, config)
    retriever = UnifiedRetriever(mem_store, graph, config)

    ctx = await retriever.retrieve("MOSFET", include_memory=False)
    assert ctx.graph.total_concepts_found >= 1


@pytest.mark.asyncio
async def test_retriever_memory_only(graph: GraphStore):
    config = load_config()
    mem_store = MemoryStore(graph, config)

    await mem_store.save(MemoryEntry(
        memory_id="test_ret_mem1",
        layer=MemoryLayer.SEMANTIC,
        tier=PromotionTier.CORE,
        content="MOSFET is the foundation of modern circuits",
    ))

    retriever = UnifiedRetriever(mem_store, graph, config)
    ctx = await retriever.retrieve("MOSFET", include_graph=False)
    assert len(ctx.memories) >= 1


@pytest.mark.asyncio
async def test_retriever_combined(graph: GraphStore):
    config = load_config()
    mem_store = MemoryStore(graph, config)

    await mem_store.save(MemoryEntry(
        memory_id="test_ret_mem2",
        layer=MemoryLayer.SEMANTIC,
        tier=PromotionTier.CURATED,
        content="Voltage gain is the key metric for amplifiers",
    ))

    retriever = UnifiedRetriever(mem_store, graph, config)
    ctx = await retriever.retrieve("Voltage Gain")

    formatted = ctx.format()
    assert len(formatted) > 0


def test_compact_envelope_exposes_ids():
    """W1.5: agents need node IDs to close the read→write loop."""
    from openclaw_brain.retriever import DecisionContext, GraphContext, UnifiedContext

    ctx = UnifiedContext(
        graph=GraphContext(
            concepts=[
                {"concept_id": "body_effect", "canonical_name": "Body Effect",
                 "confidence": 0.82, "knowledge_layer": "L1", "domain": "device_physics",
                 "source_id": "src_body"},
                {"topology_id": "cascode_amp", "canonical_name": "Cascode Amplifier"},
                {"canonical_name": "no-id-node"},  # dropped: no usable ref
            ],
            total_concepts_found=3,
        ),
        design=DecisionContext(
            hypotheses=[{"id": "hyp_1", "statement": "s", "status": "open", "confidence": 0.5}],
            decisions=[{"id": "dec_1", "choice": "c", "status": "active", "rationale": "r"}],
        ),
    )
    c = ctx.compact()
    assert [x["id"] for x in c["concepts"]] == ["body_effect", "cascode_amp"]
    assert set(c["concepts"][0]) == {"id", "name", "confidence", "layer", "domain", "cite"}
    assert c["concepts"][0]["layer"] == "L1"
    assert c["open_hypotheses"][0]["id"] == "hyp_1"
    assert c["active_decisions"][0]["id"] == "dec_1"
    assert set(c["open_hypotheses"][0]) == {"id", "statement", "status", "confidence"}
    assert set(c["active_decisions"][0]) == {"id", "choice", "status"}
    # formatted text also carries ids inline
    text = ctx.format()
    assert "`id: body_effect`" in text
    assert "`id: hyp_1`" in text
    assert "`id: dec_1`" in text


def test_compact_cite_level_chunk():
    ctx = UnifiedContext(
        graph=GraphContext(
            concepts=[
                {
                    "concept_id": "mosfet",
                    "canonical_name": "MOSFET",
                    "source_id": "src_razavi",
                    "evidence_chunk_ids": ["chunk_1", "chunk_2", "chunk_3", "chunk_4"],
                },
            ],
        ),
    )

    assert ctx.compact()["concepts"][0]["cite"] == {
        "level": "chunk",
        "src": "src_razavi",
        "chunks": ["chunk_1", "chunk_2", "chunk_3"],
    }


def test_compact_cite_level_source():
    ctx = UnifiedContext(
        graph=GraphContext(
            concepts=[
                {
                    "concept_id": "body_effect",
                    "canonical_name": "Body Effect",
                    "source_id": "src_device_physics",
                },
            ],
        ),
    )

    assert ctx.compact()["concepts"][0]["cite"] == {
        "level": "source",
        "src": "src_device_physics",
    }


def test_compact_cite_level_derived():
    ctx = UnifiedContext(
        graph=GraphContext(
            concepts=[
                {"concept_id": "inferred_tradeoff", "canonical_name": "Inferred Tradeoff"},
            ],
        ),
    )

    assert ctx.compact()["concepts"][0]["cite"] == {"level": "derived"}
