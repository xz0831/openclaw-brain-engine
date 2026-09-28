"""Unit tests for memory data models."""

from openclaw_brain.memory.models import (
    EpisodicEvent,
    MemoryEntry,
    MemoryLayer,
    ProceduralPattern,
    PromotionTier,
    RetrievalResult,
    SemanticFact,
)


def test_memory_entry_defaults():
    entry = MemoryEntry(
        memory_id="test_1",
        layer=MemoryLayer.EPISODIC,
        content="User asked about MOSFET biasing",
    )
    assert entry.tier == PromotionTier.RAW
    assert entry.access_count == 0
    assert entry.confidence == 0.7
    assert entry.embedding is None


def test_memory_entry_to_neo4j_props():
    entry = MemoryEntry(
        memory_id="test_2",
        layer=MemoryLayer.SEMANTIC,
        tier=PromotionTier.CURATED,
        content="gm = 2 * ID / Vov",
        tags=["equation", "mosfet"],
    )
    props = entry.to_neo4j_props()
    assert props["layer"] == "semantic"
    assert props["tier"] == "curated"
    assert "embedding" not in props
    assert isinstance(props["created_at"], str)


def test_episodic_event():
    event = EpisodicEvent(
        event_type="user_message",
        content="What is the small-signal model?",
    )
    assert event.event_type == "user_message"
    assert event.metadata == {}


def test_semantic_fact():
    fact = SemanticFact(
        fact_type="entity_property",
        subject_id="entity_minjong",
        predicate="role",
        value="Example circuit engineer",
    )
    assert fact.confidence == 0.8


def test_procedural_pattern_defaults():
    pattern = ProceduralPattern(skill_name="pdf_extract")
    assert pattern.total_runs == 0
    assert pattern.success_rate == 0.0
    assert pattern.common_errors == []


def test_retrieval_result():
    entry = MemoryEntry(
        memory_id="test_r",
        layer=MemoryLayer.EPISODIC,
        content="test",
    )
    result = RetrievalResult(
        memory=entry,
        relevance_score=0.85,
        source_layer=MemoryLayer.EPISODIC,
        retrieval_reason="content match",
    )
    assert result.relevance_score == 0.85


def test_promotion_tier_ordering():
    tiers = [PromotionTier.RAW, PromotionTier.RETAIN, PromotionTier.CURATED, PromotionTier.CORE]
    assert [t.value for t in tiers] == ["raw", "retain", "curated", "core"]
