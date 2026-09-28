"""Tests for the reinforcement engine."""

import pytest
from tests.conftest import require_live_graph

from openclaw_brain.config import BrainConfig, Neo4jConfig, load_config
from openclaw_brain.knowledge.graph.schema import NodeLabel, RelType
from openclaw_brain.knowledge.graph.store import GraphStore
from openclaw_brain.knowledge.reinforcement import ReinforcementEngine


@pytest.fixture
async def graph():
    require_live_graph()
    config = load_config()
    g = GraphStore(config.neo4j)
    try:
        await g.connect()
    except Exception:
        pytest.skip("Neo4j not available")

    # Seed test concepts
    await g.merge_node(
        NodeLabel.CONCEPT, "concept_id", "test_reinf_gm",
        {
            "concept_id": "test_reinf_gm",
            "canonical_name": "Transconductance",
            "domain": "analog_circuits",
            "confidence": 0.7,
            "reinforcement_count": 1,
        },
    )
    await g.merge_node(
        NodeLabel.CONCEPT, "concept_id", "test_reinf_mosfet",
        {
            "concept_id": "test_reinf_mosfet",
            "canonical_name": "MOSFET",
            "domain": "semiconductor",
            "confidence": 0.8,
            "reinforcement_count": 3,
        },
    )

    # Create an edge
    await g.merge_edge(
        source_label=NodeLabel.CONCEPT,
        source_id_field="concept_id",
        source_id_value="test_reinf_mosfet",
        target_label=NodeLabel.CONCEPT,
        target_id_field="concept_id",
        target_id_value="test_reinf_gm",
        rel_type=RelType.HAS_PARAMETER,
        properties={"rationale": "gm is a key MOSFET parameter", "confidence": 0.8},
    )

    yield g

    async with await g._session() as session:
        await session.run("MATCH (n:Concept) WHERE n.concept_id STARTS WITH 'test_reinf_' DETACH DELETE n")
    await g.close()


@pytest.mark.asyncio
async def test_reinforce_concept(graph: GraphStore):
    config = load_config()
    engine = ReinforcementEngine(graph, config)

    await engine.reinforce_concept("test_reinf_gm", "User confirmed gm importance")

    node = await graph.get_node(NodeLabel.CONCEPT, "concept_id", "test_reinf_gm")
    assert node is not None
    assert node["reinforcement_count"] >= 2
    assert node["confidence"] > 0.7


@pytest.mark.asyncio
async def test_reinforce_edge(graph: GraphStore):
    config = load_config()
    engine = ReinforcementEngine(graph, config)

    await engine.reinforce_edge(
        "test_reinf_mosfet", "test_reinf_gm",
        "HAS_PARAMETER", "Confirmed in new PDF",
    )
    # Verify via neighborhood
    neighbors = await graph.get_neighborhood(
        NodeLabel.CONCEPT, "concept_id", "test_reinf_mosfet", hops=1,
    )
    has_param = [n for n in neighbors if n.get("rel_type") == "HAS_PARAMETER"]
    assert len(has_param) >= 1


@pytest.mark.asyncio
async def test_apply_decay(graph: GraphStore):
    config = load_config()
    engine = ReinforcementEngine(graph, config)

    # Set a concept's last_reinforced to far in the past
    async with await graph._session() as session:
        await session.run(
            """
            MATCH (c:Concept {concept_id: 'test_reinf_gm'})
            SET c.last_reinforced = datetime('2025-01-01T00:00:00')
            """,
        )

    affected = await engine.apply_decay(days_threshold=30)
    # Should have decayed at least one concept
    assert affected >= 1

    node = await graph.get_node(NodeLabel.CONCEPT, "concept_id", "test_reinf_gm")
    assert node["confidence"] < 0.7  # Was 0.7, should have decayed


@pytest.mark.asyncio
async def test_find_bridges_empty(graph: GraphStore):
    config = load_config()
    engine = ReinforcementEngine(graph, config)
    bridges = await engine.find_potential_bridges()
    # May or may not find bridges in test data, but should not error
    assert isinstance(bridges, list)


@pytest.mark.asyncio
async def test_reinforcement_stats(graph: GraphStore):
    config = load_config()
    engine = ReinforcementEngine(graph, config)
    stats = await engine.get_reinforcement_stats()
    assert "total_concepts" in stats
    assert stats["total_concepts"] >= 2


# ── reinforce_edge rel_type validation: mocked-driver unit tests ──
#
# reinforce_edge has zero callers today but is public API that interpolates
# rel_type into an f-string Cypher relationship pattern — it must validate
# against RelType before that interpolation happens. No live Neo4j needed.


class _FakeSession:
    """Records every (query, params) pair passed to session.run()."""

    def __init__(self, calls: list[tuple[str, dict]]):
        self._calls = calls

    async def run(self, query, params=None):
        self._calls.append((query, params or {}))
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeDriver:
    def __init__(self, calls: list[tuple[str, dict]]):
        self._calls = calls

    def session(self, database=None):
        return _FakeSession(self._calls)


def _mocked_engine() -> tuple[ReinforcementEngine, list[tuple[str, dict]]]:
    """A ReinforcementEngine wired to a fake driver — no live Neo4j required."""
    calls: list[tuple[str, dict]] = []
    graph = GraphStore(Neo4jConfig())
    graph._driver = _FakeDriver(calls)
    engine = ReinforcementEngine(graph, BrainConfig())
    return engine, calls


@pytest.mark.asyncio
async def test_reinforce_edge_accepts_known_rel_type_string():
    engine, calls = _mocked_engine()
    await engine.reinforce_edge("src_id", "tgt_id", "HAS_PARAMETER", "evidence text")
    assert len(calls) == 1
    query, params = calls[0]
    assert "HAS_PARAMETER" in query
    assert params == {
        "source_id": "src_id",
        "target_id": "tgt_id",
        "evidence": "evidence text",
    }


@pytest.mark.asyncio
async def test_reinforce_edge_accepts_rel_type_enum_member():
    engine, calls = _mocked_engine()
    await engine.reinforce_edge("src_id", "tgt_id", RelType.DEPENDS_ON, "evidence")
    assert len(calls) == 1
    assert "DEPENDS_ON" in calls[0][0]


@pytest.mark.asyncio
async def test_reinforce_edge_rejects_unknown_rel_type():
    engine, calls = _mocked_engine()
    with pytest.raises(ValueError):
        await engine.reinforce_edge(
            "src_id", "tgt_id", "DROP DATABASE neo4j //", "evidence",
        )
    # Nothing should have reached the driver for a rejected rel_type.
    assert calls == []
