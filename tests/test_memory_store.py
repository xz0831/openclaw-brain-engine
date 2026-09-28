"""Integration tests for the unified memory system.

Requires a running Neo4j instance (docker compose up -d).
Skipped automatically if Neo4j is not available.
"""

import pytest
from tests.conftest import require_live_graph

from openclaw_brain.config import load_config
from openclaw_brain.knowledge.graph.store import GraphStore
from openclaw_brain.memory.models import (
    EpisodicEvent,
    MemoryEntry,
    MemoryLayer,
    PromotionTier,
    SemanticFact,
)
from openclaw_brain.memory.store import MemoryStore
from openclaw_brain.memory.episodic import EpisodicMemory
from openclaw_brain.memory.semantic import SemanticMemory
from openclaw_brain.memory.procedural import ProceduralMemory
from openclaw_brain.memory.retriever import MemoryRetriever
from openclaw_brain.memory.promotion import PromotionPipeline


@pytest.fixture
async def graph():
    require_live_graph()
    config = load_config()
    g = GraphStore(config.neo4j)
    try:
        await g.connect()
    except Exception:
        pytest.skip("Neo4j not available")
    yield g
    # Cleanup test data
    async with await g._session() as session:
        await session.run("MATCH (n:Memory) WHERE n.memory_id STARTS WITH 'test_' OR n.memory_id STARTS WITH 'ep_' OR n.memory_id STARTS WITH 'sem_' OR n.memory_id STARTS WITH 'proc_' OR n.memory_id STARTS WITH 'lesson_' OR n.memory_id STARTS WITH 'ep_sum_' DETACH DELETE n")
        await session.run("MATCH (n:Session) WHERE n.session_id STARTS WITH 'test_' OR n.session_id STARTS WITH 'ses_' DETACH DELETE n")
        await session.run("MATCH (n:Entity) WHERE n.entity_id STARTS WITH 'test_' DETACH DELETE n")
        await session.run("MATCH (n:SkillRun) WHERE n.run_id STARTS WITH 'run_' DETACH DELETE n")
    await g.close()


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def mem_store(graph, config):
    return MemoryStore(graph, config)


# ── MemoryStore tests ──


@pytest.mark.asyncio
async def test_save_and_get(mem_store: MemoryStore):
    entry = MemoryEntry(
        memory_id="test_save_1",
        layer=MemoryLayer.EPISODIC,
        content="User asked about common-source amplifier gain",
        tags=["mosfet", "amplifier"],
    )
    mid = await mem_store.save(entry)
    assert mid == "test_save_1"

    retrieved = await mem_store.get("test_save_1")
    assert retrieved is not None
    assert retrieved.content == "User asked about common-source amplifier gain"
    assert retrieved.layer == MemoryLayer.EPISODIC


@pytest.mark.asyncio
async def test_search_by_content(mem_store: MemoryStore):
    await mem_store.save(MemoryEntry(
        memory_id="test_search_1",
        layer=MemoryLayer.SEMANTIC,
        content="Transconductance gm equals 2*ID/Vov for MOSFET in saturation",
    ))
    results = await mem_store.search_by_content("transconductance")
    assert any(m.memory_id == "test_search_1" for m in results)


@pytest.mark.asyncio
async def test_search_by_tags(mem_store: MemoryStore):
    await mem_store.save(MemoryEntry(
        memory_id="test_tags_1",
        layer=MemoryLayer.SEMANTIC,
        content="Lesson: always check operating point before AC analysis",
        tags=["lesson", "analog"],
    ))
    results = await mem_store.search_by_tags(["lesson"])
    assert any(m.memory_id == "test_tags_1" for m in results)


@pytest.mark.asyncio
async def test_promote(mem_store: MemoryStore):
    await mem_store.save(MemoryEntry(
        memory_id="test_promote_1",
        layer=MemoryLayer.SEMANTIC,
        tier=PromotionTier.RAW,
        content="A new observation",
    ))
    await mem_store.promote("test_promote_1", PromotionTier.RETAIN)
    updated = await mem_store.get("test_promote_1")
    assert updated is not None
    assert updated.tier == PromotionTier.RETAIN


@pytest.mark.asyncio
async def test_stats(mem_store: MemoryStore):
    await mem_store.save(MemoryEntry(
        memory_id="test_stats_1",
        layer=MemoryLayer.EPISODIC,
        content="stat test",
    ))
    stats = await mem_store.get_stats()
    assert stats["total"] >= 1


# ── EpisodicMemory tests ──


@pytest.mark.asyncio
async def test_episodic_session(mem_store: MemoryStore, graph: GraphStore):
    ep = EpisodicMemory(mem_store, graph)

    sid = await ep.start_session("test_session_1")
    assert sid == "test_session_1"

    mid = await ep.record_event(EpisodicEvent(
        event_type="user_message",
        content="Explain the Miller effect",
    ))
    assert mid.startswith("ep_")

    buf = ep.get_buffer()
    assert len(buf) == 1

    summary_id = await ep.end_session("Discussed Miller effect and capacitance multiplication")
    assert summary_id is not None


# ── SemanticMemory tests ──


@pytest.mark.asyncio
async def test_semantic_entity_and_fact(mem_store: MemoryStore, graph: GraphStore):
    sem = SemanticMemory(mem_store, graph)

    await sem.upsert_entity(
        entity_id="test_entity_minjong",
        entity_type="person",
        canonical_name="민종",
        summary="Example circuit engineer learning analog design",
        aliases=["Minjong"],
    )

    entity = await sem.get_entity("test_entity_minjong")
    assert entity is not None
    assert entity["canonical_name"] == "민종"

    await sem.record_fact(SemanticFact(
        fact_type="entity_property",
        subject_id="test_entity_minjong",
        predicate="primary_interest",
        value="CMOS image sensor pixel design",
    ))

    memories = await sem.get_entity_memories("test_entity_minjong")
    assert len(memories) >= 1


@pytest.mark.asyncio
async def test_semantic_lesson(mem_store: MemoryStore, graph: GraphStore):
    sem = SemanticMemory(mem_store, graph)
    mid = await sem.record_lesson("Always verify DC operating point before AC analysis")
    assert mid.startswith("lesson_")

    lessons = await sem.get_lessons()
    assert any(m.memory_id == mid for m in lessons)


# ── ProceduralMemory tests ──


@pytest.mark.asyncio
async def test_procedural_skill_run(mem_store: MemoryStore, graph: GraphStore):
    # Need a session first
    await graph.merge_node(
        label=__import__("openclaw_brain.knowledge.graph.schema", fromlist=["NodeLabel"]).NodeLabel.SESSION,
        id_field="session_id",
        id_value="test_proc_session",
        properties={"session_id": "test_proc_session", "status": "active"},
    )

    proc = ProceduralMemory(mem_store, graph)
    run_id = await proc.record_skill_run(
        skill_name="pdf_extract",
        session_id="test_proc_session",
        input_summary="Extract from chapter 3",
        output_summary="Extracted 45 concepts",
        success=True,
        duration_seconds=12.5,
    )
    assert run_id.startswith("run_")

    stats = await proc.get_skill_stats("pdf_extract")
    assert stats.total_runs >= 1
    assert stats.success_rate > 0


# ── MemoryRetriever tests ──


@pytest.mark.asyncio
async def test_retriever_format(mem_store: MemoryStore, config):
    # Seed a core memory
    await mem_store.save(MemoryEntry(
        memory_id="test_core_1",
        layer=MemoryLayer.SEMANTIC,
        tier=PromotionTier.CORE,
        content="민종 is learning analog circuit design with focus on CMOS image sensors",
    ))
    await mem_store.save(MemoryEntry(
        memory_id="test_curated_1",
        layer=MemoryLayer.SEMANTIC,
        tier=PromotionTier.CURATED,
        content="gm = 2*ID/Vov is the most important equation in analog design",
        tags=["equation"],
    ))

    retriever = MemoryRetriever(mem_store, config)
    text = await retriever.format_for_injection("analog design")
    assert "Core Memories" in text
    assert "민종" in text


# ── PromotionPipeline tests ──


@pytest.mark.asyncio
async def test_promotion_pipeline(mem_store: MemoryStore, config):
    # Create a raw memory that has been accessed 3 times
    await mem_store.save(MemoryEntry(
        memory_id="test_promo_1",
        layer=MemoryLayer.SEMANTIC,
        tier=PromotionTier.RAW,
        content="A frequently accessed fact",
        access_count=3,
    ))

    pipeline = PromotionPipeline(mem_store, config)
    counts = await pipeline.run()
    assert counts["raw_to_retain"] >= 1

    promoted = await mem_store.get("test_promo_1")
    assert promoted is not None
    assert promoted.tier == PromotionTier.RETAIN
