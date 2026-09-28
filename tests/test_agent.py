"""Tests for the BrainAgent — the main orchestration layer."""

from unittest.mock import MagicMock

import pytest
from tests.conftest import require_live_graph

from openclaw_brain.agent import BrainAgent
from openclaw_brain.config import load_config
from openclaw_brain.knowledge.graph.schema import NodeLabel, RelType
from openclaw_brain.knowledge.graph.store import GraphStore


@pytest.fixture
async def agent():
    require_live_graph()
    config = load_config()
    a = BrainAgent(config)
    try:
        await a.start()
    except Exception:
        pytest.skip("Neo4j not available")
    yield a
    # Cleanup test data
    async with await a._graph._session() as session:
        await session.run("MATCH (n:Memory) WHERE n.memory_id STARTS WITH 'ep_' OR n.memory_id STARTS WITH 'sem_' OR n.memory_id STARTS WITH 'lesson_' DETACH DELETE n")
        await session.run("MATCH (n:Session) WHERE n.session_id STARTS WITH 'ses_' OR n.session_id STARTS WITH 'test_agent_' DETACH DELETE n")
        await session.run("MATCH (n:Entity) WHERE n.entity_id STARTS WITH 'test_agent_' DETACH DELETE n")
    await a.stop()


@pytest.mark.asyncio
async def test_agent_start_stop():
    require_live_graph()
    config = load_config()
    a = BrainAgent(config)
    try:
        await a.start()
    except Exception:
        pytest.skip("Neo4j not available")
    assert a.is_started
    await a.stop()
    assert not a.is_started


@pytest.mark.asyncio
async def test_agent_double_start(agent: BrainAgent):
    # Should be idempotent
    await agent.start()
    assert agent.is_started


@pytest.mark.asyncio
async def test_agent_not_started():
    a = BrainAgent()
    # RuntimeError, not AssertionError (W-D2 defect 3): _assert_started() is a real if/raise now,
    # not a bare `assert`, so the guard survives python -O / PYTHONOPTIMIZE instead of vanishing.
    with pytest.raises(RuntimeError, match="start"):
        await a.query_knowledge("test")


@pytest.mark.asyncio
async def test_agent_get_stats(agent: BrainAgent):
    stats = await agent.get_stats()
    assert "graph" in stats
    assert "memory" in stats
    assert "skills" in stats
    assert "models" in stats
    assert "pdf_ingest" in stats["skills"]["available"]


@pytest.mark.asyncio
async def test_agent_session_lifecycle(agent: BrainAgent):
    sid = await agent.start_session("test_agent_session_1")
    assert sid == "test_agent_session_1"

    mid = await agent.record_event("user_message", "What is gm?")
    assert mid.startswith("ep_")

    summary_id = await agent.end_session("Discussed transconductance")
    assert summary_id is not None


@pytest.mark.asyncio
async def test_agent_query_knowledge(agent: BrainAgent):
    result = await agent.query_knowledge("MOSFET amplifier")
    assert "formatted" in result
    assert "concepts_found" in result
    assert "memories_found" in result


@pytest.mark.asyncio
async def test_agent_get_evidence_preview_fallback():
    class FakeGraph:
        async def get_node(self, label, id_field, id_value):
            return {
                "chunk_id": id_value,
                "source_id": "src_1",
                "text_preview": "preview text",
                "section_title": "Section",
                "pages": "4",
            }

    config = load_config()
    a = BrainAgent(config)
    a._graph = FakeGraph()
    a._started = True

    result = await a.get_evidence("chunk_1")

    assert result == {
        "chunk_id": "chunk_1",
        "source_id": "src_1",
        "text": "preview text",
        "verbatim": False,
        "section_title": "Section",
        "pages": "4",
    }


@pytest.mark.asyncio
async def test_agent_recall(agent: BrainAgent):
    # Record a lesson first
    await agent.record_lesson("gm = 2*ID/Vov is fundamental", tags=["equation"])

    result = await agent.recall("gm equation")
    assert "formatted" in result
    assert "count" in result


@pytest.mark.asyncio
async def test_agent_upsert_entity(agent: BrainAgent):
    eid = await agent.upsert_entity(
        "test_agent_minjong", "person", "민종",
        summary="Example circuit engineer",
    )
    assert eid == "test_agent_minjong"


@pytest.mark.asyncio
async def test_agent_route_no_match(agent: BrainAgent):
    result = await agent.route_and_execute("What is the weather?")
    assert not result["routed"]


@pytest.mark.asyncio
async def test_agent_run_promotion(agent: BrainAgent):
    counts = await agent.run_promotion()
    assert isinstance(counts, dict)
    assert "raw_to_retain" in counts


@pytest.mark.asyncio
async def test_agent_find_bridges(agent: BrainAgent):
    bridges = await agent.find_bridges()
    assert isinstance(bridges, list)


@pytest.mark.asyncio
async def test_agent_pdf_ingest_missing_file(agent: BrainAgent):
    # Should handle gracefully through skill executor
    result = await agent.ingest_pdf("/nonexistent/file.pdf")
    # The executor wraps the handler result
    assert isinstance(result, ExecutionResult) or isinstance(result, dict)


# ── S5 learner model (record_assessment / get_learner_state) — mocked, no live Neo4j ──
#
# Follows test_agent_get_evidence_preview_fallback's pattern: a hand-rolled fake standing in for
# self._graph (not the low-level Neo4j driver), self._started set directly, self._journal a
# MagicMock (record_assessment logs through it — journal.log is sync, see journal.py).


class _FakeLearnerGraph:
    """Mirrors the GraphStore surface record_assessment/get_learner_state/
    _resolve_assessment_target actually call: run_read_query, write_batch,
    _id_field_for_label (a real @staticmethod delegate — mock-fidelity: the real one is
    exhaustive/enum-keyed, not worth re-faking)."""

    def __init__(self, read_results=None):
        self._read_results = list(read_results or [])
        self.read_queries: list[tuple[str, dict]] = []
        self.write_batch_calls: list[dict] = []

    async def run_read_query(self, query, params=None):
        self.read_queries.append((query, params or {}))
        return self._read_results.pop(0) if self._read_results else []

    async def write_batch(self, nodes=None, updates=None, edges=None):
        self.write_batch_calls.append({
            "nodes": list(nodes or []), "updates": list(updates or []), "edges": list(edges or []),
        })

    @staticmethod
    def _id_field_for_label(label):
        return GraphStore._id_field_for_label(label)


def _learner_agent(read_results=None) -> tuple[BrainAgent, _FakeLearnerGraph]:
    a = BrainAgent(load_config())
    graph = _FakeLearnerGraph(read_results)
    a._graph = graph
    a._journal = MagicMock()
    a._started = True
    return a, graph


@pytest.mark.asyncio
async def test_record_assessment_against_concept_target():
    """First-ever assessment of a Concept: Learner + Assessment nodes MERGEd, ASSESSES +
    UNDERSTANDS edges added with NodeLabel/RelType ENUMS (write_batch contract — CLAUDE.md
    mock-fidelity), assessment_count starts at 1, journal logged."""
    a, graph = _learner_agent(read_results=[
        [{"label": "Concept", "assessment_count": None, "confidence": None}],
    ])

    aid = await a.record_assessment("rick", "concept_gm", "understood", evidence="explained gm=2Id/Vov")

    assert aid.startswith("assess_")
    assert len(graph.write_batch_calls) == 1
    call = graph.write_batch_calls[0]

    node_labels = {(n["label"], n["id_field"], n["id_value"]) for n in call["nodes"]}
    assert (NodeLabel.LEARNER, "learner_id", "rick") in node_labels
    assert (NodeLabel.ASSESSMENT, "assessment_id", aid) in node_labels
    assessment_node = next(n for n in call["nodes"] if n["label"] == NodeLabel.ASSESSMENT)
    assert assessment_node["properties"]["verdict"] == "understood"
    assert assessment_node["properties"]["evidence"] == "explained gm=2Id/Vov"
    assert assessment_node["properties"]["learner_id"] == "rick"

    assert len(call["edges"]) == 2
    assesses = next(e for e in call["edges"] if e["rel_type"] == RelType.ASSESSES)
    assert assesses["source_label"] == NodeLabel.ASSESSMENT
    assert assesses["target_label"] == NodeLabel.CONCEPT
    assert assesses["target_id_field"] == "concept_id"
    assert assesses["target_id_value"] == "concept_gm"

    understands = next(e for e in call["edges"] if e["rel_type"] == RelType.UNDERSTANDS)
    assert understands["source_label"] == NodeLabel.LEARNER
    assert understands["source_id_value"] == "rick"
    assert understands["target_label"] == NodeLabel.CONCEPT
    assert understands["properties"]["status"] == "understood"
    assert understands["properties"]["confidence"] == 0.9
    assert understands["properties"]["assessment_count"] == 1

    a._journal.log.assert_called_once_with(
        "record_assessment", assessment_id=aid, learner_id="rick",
        target_id="concept_gm", verdict="understood",
    )


@pytest.mark.asyncio
async def test_record_assessment_against_regularity_target():
    """Acceptance criterion (design doc §6.1 item 1): target is label-agnostic — must work
    against a Regularity (law-tier) target too, not just Concept, since that pairing is the
    trust-tier interaction (§1.1) the design exists to prove out."""
    a, graph = _learner_agent(read_results=[
        [{"label": "Regularity", "assessment_count": 2, "confidence": 0.6}],
    ])

    await a.record_assessment("rick", "law-pelgrom", "understood")

    call = graph.write_batch_calls[0]
    understands = next(e for e in call["edges"] if e["rel_type"] == RelType.UNDERSTANDS)
    assert understands["target_label"] == NodeLabel.REGULARITY
    assert understands["target_id_field"] == "law_id"
    assert understands["properties"]["assessment_count"] == 3  # prior 2 + this one


@pytest.mark.asyncio
async def test_record_assessment_misconception_verdict_lowers_confidence():
    a, graph = _learner_agent(read_results=[
        [{"label": "Concept", "assessment_count": 0, "confidence": None}],
    ])

    await a.record_assessment("rick", "concept_noise", "misconception",
                              evidence="confused offset with noise")

    call = graph.write_batch_calls[0]
    understands = next(e for e in call["edges"] if e["rel_type"] == RelType.UNDERSTANDS)
    assert understands["properties"]["confidence"] == 0.15
    assert understands["properties"]["status"] == "misconception"


@pytest.mark.asyncio
async def test_record_assessment_unknown_target_records_event_but_skips_edges():
    """Never fabricates the thing it's about (mirrors record_bench_result's tests_hypothesis
    handling): a target_id that resolves to nothing still records the Assessment event, but
    adds no ASSESSES/UNDERSTANDS edges."""
    a, graph = _learner_agent(read_results=[[]])  # no node anywhere carries this id

    aid = await a.record_assessment("rick", "nonexistent_id", "understood")

    call = graph.write_batch_calls[0]
    assert len(call["nodes"]) == 2  # Learner + Assessment still written
    assert call["edges"] == []
    assert aid.startswith("assess_")


@pytest.mark.asyncio
async def test_resolve_assessment_target_unresolvable_label_treated_as_not_found():
    """A label string that isn't a real NodeLabel (shouldn't happen, but _resolve_assessment_target
    must not raise) degrades to 'not found' rather than propagating a ValueError."""
    a, graph = _learner_agent(read_results=[
        [{"label": "NotARealLabel", "assessment_count": 0, "confidence": None}],
    ])

    label, count, confidence = await a._resolve_assessment_target("rick", "some_id")
    assert label is None
    assert count == 0
    assert confidence is None


@pytest.mark.asyncio
async def test_get_learner_state_returns_rolled_up_entries():
    a, graph = _learner_agent(read_results=[
        [
            {
                "target_label": "Concept",
                "target_props": {"concept_id": "concept_gm", "canonical_name": "Transconductance"},
                "understands": {"confidence": 0.9, "status": "understood",
                                "last_assessed": "2026-07-10T00:00:00", "assessment_count": 1},
            },
            {
                "target_label": "Regularity",
                "target_props": {"law_id": "law-pelgrom", "canonical_name": None},
                "understands": {"confidence": 0.6, "status": "partial",
                                "last_assessed": "2026-07-09T00:00:00", "assessment_count": 2},
            },
        ],
    ])

    results = await a.get_learner_state("rick")

    assert len(results) == 2
    concept_entry = next(r for r in results if r["target_label"] == "Concept")
    assert concept_entry["target_id"] == "concept_gm"
    assert concept_entry["target_name"] == "Transconductance"
    assert concept_entry["confidence"] == 0.9
    assert concept_entry["assessment_count"] == 1

    law_entry = next(r for r in results if r["target_label"] == "Regularity")
    assert law_entry["target_id"] == "law-pelgrom"


@pytest.mark.asyncio
async def test_get_learner_state_filters_by_target_id():
    a, graph = _learner_agent(read_results=[
        [
            {"target_label": "Concept", "target_props": {"concept_id": "c1"},
             "understands": {"confidence": 0.9, "status": "understood",
                             "last_assessed": "t1", "assessment_count": 1}},
            {"target_label": "Concept", "target_props": {"concept_id": "c2"},
             "understands": {"confidence": 0.5, "status": "partial",
                             "last_assessed": "t2", "assessment_count": 1}},
        ],
    ])

    results = await a.get_learner_state("rick", target_id="c2")

    assert len(results) == 1
    assert results[0]["target_id"] == "c2"


@pytest.mark.asyncio
async def test_get_learner_state_empty_returns_empty_list():
    a, graph = _learner_agent(read_results=[[]])
    results = await a.get_learner_state("rick")
    assert results == []


# Import here to avoid issues if not installed
from openclaw_brain.skills.executor import ExecutionResult
