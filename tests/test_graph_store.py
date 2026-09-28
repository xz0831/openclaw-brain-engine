"""Integration tests for Neo4j graph store.

Requires a running Neo4j instance (docker compose up -d).
Skipped automatically if Neo4j is not available.
"""

import re

import pytest
from tests.conftest import require_live_graph

from openclaw_brain.config import Neo4jConfig, load_config
from openclaw_brain.knowledge.graph.schema import (
    ConceptNode,
    EdgeProposal,
    GraphDelta,
    InsightProposal,
    NodeLabel,
    NodeProposal,
    NodeUpdate,
    RelType,
)
from openclaw_brain.knowledge.graph.store import (
    GraphStore,
    _cypher_key,
    _param_key_map,
    _set_assignments,
)


@pytest.fixture
async def store():
    require_live_graph()
    config = load_config()
    s = GraphStore(config.neo4j)
    try:
        await s.connect()
    except Exception:
        pytest.skip("Neo4j not available")
    yield s
    # Cleanup test data
    async with await s._session() as session:
        await session.run("MATCH (n) WHERE n.concept_id STARTS WITH 'test_' DETACH DELETE n")
        await session.run("MATCH (n) WHERE n.parameter_id STARTS WITH 'test_' DETACH DELETE n")
        await session.run("MATCH (n) WHERE n.insight_id STARTS WITH 'test_' DETACH DELETE n")
    await s.close()


@pytest.mark.asyncio
async def test_connect_and_stats(store: GraphStore):
    stats = await store.get_stats()
    assert isinstance(stats, dict)


@pytest.mark.asyncio
async def test_merge_and_get_node(store: GraphStore):
    await store.merge_node(
        label=NodeLabel.CONCEPT,
        id_field="concept_id",
        id_value="test_common_source",
        properties={
            "concept_id": "test_common_source",
            "canonical_name": "Common-Source Amplifier",
            "description": "Basic MOSFET amplifier topology",
            "domain": "analog_circuits",
            "confidence": 0.9,
        },
    )
    node = await store.get_node(NodeLabel.CONCEPT, "concept_id", "test_common_source")
    assert node is not None
    assert node["canonical_name"] == "Common-Source Amplifier"
    assert node["confidence"] == 0.9


@pytest.mark.asyncio
async def test_merge_edge(store: GraphStore):
    # Create two nodes
    await store.merge_node(
        NodeLabel.CONCEPT, "concept_id", "test_cs_amp",
        {"concept_id": "test_cs_amp", "canonical_name": "CS Amplifier"},
    )
    await store.merge_node(
        NodeLabel.CONCEPT, "concept_id", "test_voltage_gain",
        {"concept_id": "test_voltage_gain", "canonical_name": "Voltage Gain"},
    )
    # Create edge
    await store.merge_edge(
        source_label=NodeLabel.CONCEPT,
        source_id_field="concept_id",
        source_id_value="test_cs_amp",
        target_label=NodeLabel.CONCEPT,
        target_id_field="concept_id",
        target_id_value="test_voltage_gain",
        rel_type=RelType.DEPENDS_ON,
        properties={
            "rationale": "CS amplifier gain is characterized by voltage gain parameter",
            "confidence": 0.85,
        },
    )
    # Verify via neighborhood
    neighbors = await store.get_neighborhood(
        NodeLabel.CONCEPT, "concept_id", "test_cs_amp", hops=1,
    )
    assert len(neighbors) > 0
    assert any(n["rel_type"] == "DEPENDS_ON" for n in neighbors)


@pytest.mark.asyncio
async def test_write_batch_atomic_and_update_only(store: GraphStore):
    # Nodes + edge committed together in one transaction.
    await store.write_batch(
        nodes=[
            {"label": NodeLabel.CONCEPT, "id_field": "concept_id",
             "id_value": "test_wb_concept",
             "properties": {"concept_id": "test_wb_concept", "canonical_name": "WB Concept"}},
            {"label": NodeLabel.HYPOTHESIS, "id_field": "hypothesis_id",
             "id_value": "test_wb_hyp",
             "properties": {"hypothesis_id": "test_wb_hyp", "statement": "x", "status": "open"}},
        ],
        edges=[
            {"source_label": NodeLabel.HYPOTHESIS, "source_id_field": "hypothesis_id",
             "source_id_value": "test_wb_hyp", "target_label": NodeLabel.CONCEPT,
             "target_id_field": "concept_id", "target_id_value": "test_wb_concept",
             "rel_type": RelType.RELATES_TO, "properties": {"confidence": 0.5}},
        ],
    )
    hyp = await store.get_node(NodeLabel.HYPOTHESIS, "hypothesis_id", "test_wb_hyp")
    assert hyp is not None and hyp["status"] == "open"

    # MATCH-only update applies to an existing node.
    await store.write_batch(updates=[
        {"label": NodeLabel.HYPOTHESIS, "id_field": "hypothesis_id",
         "id_value": "test_wb_hyp", "properties": {"status": "confirmed"}},
    ])
    hyp = await store.get_node(NodeLabel.HYPOTHESIS, "hypothesis_id", "test_wb_hyp")
    assert hyp["status"] == "confirmed"

    # MATCH-only update on a missing node is a no-op — no phantom node is created.
    await store.write_batch(updates=[
        {"label": NodeLabel.HYPOTHESIS, "id_field": "hypothesis_id",
         "id_value": "test_wb_missing", "properties": {"status": "confirmed"}},
    ])
    assert await store.get_node(NodeLabel.HYPOTHESIS, "hypothesis_id", "test_wb_missing") is None

    # Cleanup (fixture only sweeps concept_/parameter_/insight_ ids).
    async with await store._session() as session:
        await session.run(
            "MATCH (n:Hypothesis) WHERE n.hypothesis_id STARTS WITH 'test_wb' DETACH DELETE n"
        )


def test_id_field_for_label_covers_executable_labels():
    # I3: retract_node resolves the id field via _id_field_for_label; without these it KeyErrors
    # on exactly the two node types the executable projection writes.
    assert GraphStore._id_field_for_label(NodeLabel.SPECIMEN) == "spec_id"
    assert GraphStore._id_field_for_label(NodeLabel.CLAIM_CARD) == "claim_id"


def test_id_field_for_label_covers_learner_model_labels():
    # S5 learner model (docs/specs/S5_LEARNER_MODEL_DESIGN.md §6.1): record_assessment /
    # get_learner_state resolve id fields via _id_field_for_label — without these entries they
    # KeyError on exactly the two node types the learner model writes.
    assert GraphStore._id_field_for_label(NodeLabel.LEARNER) == "learner_id"
    assert GraphStore._id_field_for_label(NodeLabel.ASSESSMENT) == "assessment_id"


def test_label_str_to_id_field_covers_learner_model_labels():
    # graph/README.md Known Issue #3 / Trap: _label_str_to_id_field (merge_concepts's edge-rewire)
    # is a second, independent, string-keyed map that must be extended alongside
    # _id_field_for_label whenever a NodeLabel is added — the exact trap the design doc's §2
    # Option A cons list names explicitly. Only asserting the NEW entries here (the map's
    # pre-existing gap for Specimen/ClaimCard/Regularity is out of scope — Known Issue #3, not
    # introduced by this change).
    from openclaw_brain.knowledge.graph.store import _label_str_to_id_field

    assert _label_str_to_id_field("Learner") == "learner_id"
    assert _label_str_to_id_field("Assessment") == "assessment_id"


@pytest.mark.asyncio
async def test_retract_projection_round_trip(store: GraphStore):
    # I3: a bad --apply must be reversible. retract_projection DETACH DELETEs the Specimen + its
    # ClaimCards (and thus their edges) for a spec_id.
    from openclaw_brain.knowledge.executable.projection import retract_projection

    sid = "sha256:test_retract_spec"
    await store.write_batch(
        nodes=[
            {"label": NodeLabel.SPECIMEN, "id_field": "spec_id", "id_value": sid,
             "properties": {"spec_id": sid, "topology_class": "x"}},
            {"label": NodeLabel.CLAIM_CARD, "id_field": "claim_id", "id_value": f"{sid}:c1",
             "properties": {"claim_id": f"{sid}:c1"}},
        ],
        edges=[
            {"source_label": NodeLabel.SPECIMEN, "source_id_field": "spec_id", "source_id_value": sid,
             "target_label": NodeLabel.CLAIM_CARD, "target_id_field": "claim_id",
             "target_id_value": f"{sid}:c1", "rel_type": RelType.HAS_CLAIM, "properties": {}},
        ],
    )
    assert await store.get_node(NodeLabel.SPECIMEN, "spec_id", sid) is not None
    # retract_node no longer KeyErrors on these labels either:
    assert GraphStore._id_field_for_label(NodeLabel.CLAIM_CARD) == "claim_id"

    stats = await retract_projection(store, sid)
    assert stats == {"specimens": 1, "claim_cards": 1}
    assert await store.get_node(NodeLabel.SPECIMEN, "spec_id", sid) is None
    assert await store.get_node(NodeLabel.CLAIM_CARD, "claim_id", f"{sid}:c1") is None


@pytest.mark.asyncio
async def test_projection_reapply_supersedes_links(store: GraphStore):
    # I4: re-projecting the SAME specimen must REPLACE its cross-links, not accrete a second edge.
    from openclaw_brain.knowledge.executable.corpus import compute_spec_id
    from openclaw_brain.knowledge.executable.models import (
        ClaimCard, AnalogPVT, MechanismClaim, QuantTest, Specimen)
    from openclaw_brain.knowledge.executable.projection import GraphProjector, retract_projection

    await store.write_batch(nodes=[
        {"label": NodeLabel.CIRCUIT_TOPOLOGY, "id_field": "topology_id", "id_value": "test_i4_topo",
         "properties": {"topology_id": "test_i4_topo", "canonical_name": "Test I4 Topo"}},
        {"label": NodeLabel.PARAMETER, "id_field": "parameter_id", "id_value": "test_i4_param",
         "properties": {"parameter_id": "test_i4_param", "canonical_name": "Test I4 Param"}},
    ])
    cond = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)
    spec = Specimen(topology_class="test_i4_class", netlist="* cell test_i4\n.end\n", claim_cards=[
        ClaimCard(id="c1", topology_class="test_i4_class", conditions=cond,
                  mechanism=MechanismClaim(knob="X", metric="m", series_ref="s",
                                           quant=QuantTest(kind="direction", sign="+")))])
    sid = compute_spec_id(spec)

    async def resolver(label, text):
        if label == NodeLabel.CIRCUIT_TOPOLOGY:
            return "test_i4_topo"
        if label == NodeLabel.PARAMETER:
            return "test_i4_param"
        return None

    projector = GraphProjector(store, resolver)
    try:
        await projector.project(spec)
        await projector.project(spec)           # re-apply
        r = await store.run_read_query(
            "MATCH (s:Specimen {spec_id:$sid}) "
            "OPTIONAL MATCH (s)-[re:REALIZES]->() "
            "OPTIONAL MATCH (s)-[:HAS_CLAIM]->(:ClaimCard)-[g:GROUNDS]->() "
            "RETURN count(DISTINCT re) AS realizes, count(DISTINCT g) AS grounds", {"sid": sid})
        assert r[0]["realizes"] == 1            # not 2 — superseded
        assert r[0]["grounds"] == 1
    finally:
        await retract_projection(store, sid)
        async with await store._session() as s:
            await s.run("MATCH (n) WHERE n.topology_id='test_i4_topo' OR n.parameter_id='test_i4_param' "
                        "DETACH DELETE n")


@pytest.mark.asyncio
async def test_write_batch_edge_without_properties(store: GraphStore):
    # Regression: an edge with EMPTY properties must commit, not raise. The merge built
    # `ON CREATE SET , r._created_at = ...` (leading comma -> Cypher SyntaxError) when the property
    # set was empty. The projection's HAS_CLAIM/REALIZES/GROUNDS edges all carry no properties, so
    # they were the first caller to hit this path.
    await store.write_batch(
        nodes=[
            {"label": NodeLabel.CONCEPT, "id_field": "concept_id", "id_value": "test_wbe_a",
             "properties": {"concept_id": "test_wbe_a", "canonical_name": "A"}},
            {"label": NodeLabel.CONCEPT, "id_field": "concept_id", "id_value": "test_wbe_b",
             "properties": {"concept_id": "test_wbe_b", "canonical_name": "B"}},
        ],
        edges=[
            {"source_label": NodeLabel.CONCEPT, "source_id_field": "concept_id",
             "source_id_value": "test_wbe_a", "target_label": NodeLabel.CONCEPT,
             "target_id_field": "concept_id", "target_id_value": "test_wbe_b",
             "rel_type": RelType.RELATES_TO, "properties": {}},
        ],
    )
    neighbors = await store.get_neighborhood(NodeLabel.CONCEPT, "concept_id", "test_wbe_a", hops=1)
    assert any(n["rel_type"] == "RELATES_TO" for n in neighbors)


@pytest.mark.asyncio
async def test_apply_delta_includes_node_evidence_chunk_ids_unit():
    class FakeTx:
        async def commit(self):
            pass

        async def rollback(self):
            pass

    class FakeSession:
        def __init__(self):
            self.tx = FakeTx()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def begin_transaction(self):
            return self.tx

    class CapturingGraphStore(GraphStore):
        def __init__(self):
            super().__init__(load_config().neo4j)
            self.merged_nodes = []

        async def _session(self):
            return FakeSession()

        async def _merge_node_tx(self, tx, label, id_field, id_value, properties):
            self.merged_nodes.append(
                {
                    "label": label,
                    "id_field": id_field,
                    "id_value": id_value,
                    "properties": properties,
                }
            )

    store = CapturingGraphStore()
    delta = GraphDelta(
        new_nodes=[
            NodeProposal(
                proposed_id="test_evidence_node",
                label=NodeLabel.CONCEPT,
                canonical_name="Test Evidence Node",
                evidence_chunk_ids=["chunk_a", "chunk_b"],
                reasoning="unit test",
            ),
        ],
    )

    counts = await store.apply_delta(delta)

    assert counts["new_nodes"] == 1
    assert store.merged_nodes[0]["properties"]["evidence_chunk_ids"] == ["chunk_a", "chunk_b"]


@pytest.mark.asyncio
async def test_apply_delta_equation_latex_backfills_canonical_latex_unit():
    """Equation proposals carry LaTeX under properties['latex'] (reasoner convention),
    but the schema field is 'canonical_latex'. apply_delta must backfill it so equations
    don't commit with an empty canonical_latex (root cause: 614/640 live nodes affected).
    """

    class FakeTx:
        async def commit(self):
            pass

        async def rollback(self):
            pass

    class FakeSession:
        def __init__(self):
            self.tx = FakeTx()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def begin_transaction(self):
            return self.tx

    class CapturingGraphStore(GraphStore):
        def __init__(self):
            super().__init__(load_config().neo4j)
            self.merged_nodes = []

        async def _session(self):
            return FakeSession()

        async def _merge_node_tx(self, tx, label, id_field, id_value, properties):
            self.merged_nodes.append(
                {
                    "label": label,
                    "id_field": id_field,
                    "id_value": id_value,
                    "properties": properties,
                }
            )

    store = CapturingGraphStore()
    delta = GraphDelta(
        new_nodes=[
            NodeProposal(
                proposed_id="test_equation_av",
                label=NodeLabel.EQUATION,
                canonical_name="CS Amplifier Voltage Gain",
                properties={"latex": "A_v = -g_m r_o"},
                reasoning="unit test",
            ),
            NodeProposal(
                proposed_id="test_concept_with_latex_prop",
                label=NodeLabel.CONCEPT,
                canonical_name="Not An Equation",
                properties={"latex": "should be left alone"},
                reasoning="unit test — cross-label guard",
            ),
        ],
    )

    counts = await store.apply_delta(delta)

    assert counts["new_nodes"] == 2
    eq_node = next(
        n for n in store.merged_nodes if n["id_value"] == "test_equation_av"
    )
    assert eq_node["properties"]["canonical_latex"] == "A_v = -g_m r_o"
    # 'latex' is kept too (backward-compat; things may still read it)
    assert eq_node["properties"]["latex"] == "A_v = -g_m r_o"

    # Non-Equation node with a 'latex' property is unaffected — no cross-label mapping.
    concept_node = next(
        n for n in store.merged_nodes if n["id_value"] == "test_concept_with_latex_prop"
    )
    assert "canonical_latex" not in concept_node["properties"]
    assert concept_node["properties"]["latex"] == "should be left alone"


@pytest.mark.asyncio
async def test_migrate_equation_latex_dry_run_and_apply_queries_are_scoped():
    """The recovery migration must only ever target Equation nodes with an empty/missing
    canonical_latex AND a populated latex. Dry-run must never write (no SET); --apply must
    write exactly 'SET n.canonical_latex = n.latex' scoped by the same WHERE clause.
    """
    captured = {}

    class FakeResult:
        def __init__(self, cnt):
            self._cnt = cnt

        async def single(self):
            return {"cnt": self._cnt}

    class FakeSession:
        def __init__(self, cnt):
            self._cnt = cnt

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def run(self, query, params=None):
            captured["query"] = query
            return FakeResult(self._cnt)

    class FakeStore(GraphStore):
        def __init__(self, cnt):
            super().__init__(load_config().neo4j)
            self._cnt = cnt

        async def _session(self):
            return FakeSession(self._cnt)

    dry_store = FakeStore(614)
    count = await dry_store.migrate_equation_latex(apply=False)
    assert count == 614
    dry_query = captured["query"]
    assert "MATCH (n:Equation)" in dry_query
    assert "canonical_latex IS NULL OR n.canonical_latex = ''" in dry_query
    assert "n.latex IS NOT NULL AND n.latex <> ''" in dry_query
    assert "SET" not in dry_query  # dry-run must never write

    apply_store = FakeStore(614)
    count = await apply_store.migrate_equation_latex(apply=True)
    assert count == 614
    apply_query = captured["query"]
    assert "SET n.canonical_latex = n.latex" in apply_query
    assert "canonical_latex IS NULL OR n.canonical_latex = ''" in apply_query
    assert "n.latex IS NOT NULL AND n.latex <> ''" in apply_query


@pytest.mark.asyncio
async def test_apply_graph_delta(store: GraphStore):
    delta = GraphDelta(
        new_nodes=[
            NodeProposal(
                proposed_id="test_delta_concept",
                label=NodeLabel.CONCEPT,
                canonical_name="Test Delta Concept",
                description="A test concept for delta application",
                domain="test",
                confidence=0.8,
                properties={},
                evidence_chunk_ids=["test_chunk_1"],
                reasoning="Test node for delta application",
            ),
            NodeProposal(
                proposed_id="test_delta_param",
                label=NodeLabel.PARAMETER,
                canonical_name="Test Transconductance",
                description="A test parameter for delta application",
                confidence=0.8,
                properties={
                    "symbol": "gm_test",
                    "name": "Test Transconductance",
                    "units": "A/V",
                },
                evidence_chunk_ids=["test_chunk_1"],
                reasoning="Test parameter for delta",
            ),
        ],
        new_edges=[
            EdgeProposal(
                source_ref="test_delta_concept",
                target_ref="test_delta_param",
                relationship_type=RelType.HAS_PARAMETER,
                rationale="Test concept has test parameter",
                confidence=0.8,
            ),
        ],
        insights=[
            InsightProposal(
                statement="Test insight connecting concepts",
                related_concept_ids=["test_delta_concept"],
                bridge_type="within-domain",
                confidence=0.6,
            ),
        ],
    )

    counts = await store.apply_delta(delta)
    assert counts["new_nodes"] == 2
    assert counts["new_edges"] == 1
    assert counts["insights"] == 1

    # Verify nodes exist
    node = await store.get_node(NodeLabel.CONCEPT, "concept_id", "test_delta_concept")
    assert node is not None
    assert node["canonical_name"] == "Test Delta Concept"
    assert node["evidence_chunk_ids"] == ["test_chunk_1"]

    # Cleanup extra test nodes
    async with await store._session() as session:
        await session.run(
            "MATCH (n:Parameter) WHERE n.parameter_id = 'test_delta_param' DETACH DELETE n"
        )


@pytest.mark.asyncio
async def test_find_similar_concepts(store: GraphStore):
    await store.merge_node(
        NodeLabel.CONCEPT, "concept_id", "test_mosfet_basics",
        {"concept_id": "test_mosfet_basics", "canonical_name": "MOSFET Basics"},
    )
    results = await store.find_similar_concepts("MOSFET", limit=100)
    assert any(r.get("concept_id") == "test_mosfet_basics" for r in results)


# ── Cypher-layer hygiene: mocked-driver unit tests (no live Neo4j) ──
#
# These exercise the dynamic-Cypher property-key sanitization in store.py.
# Reasoner output flowing into merge_node/merge_edge is LLM-controlled, so
# the key-quoting (_cypher_key) and param-naming (_param_key/_param_key_map)
# helpers are a real injection/clobbering surface — see _set_assignments.


class _FakeResult:
    """Minimal stand-in for a neo4j AsyncResult — no records needed here."""

    async def single(self):
        return None


class _FakeSession:
    """Records every (query, params) pair passed to session.run()."""

    def __init__(self, calls: list[tuple[str, dict]]):
        self._calls = calls

    async def run(self, query, params=None):
        self._calls.append((query, params or {}))
        return _FakeResult()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeDriver:
    def __init__(self, calls: list[tuple[str, dict]]):
        self._calls = calls

    def session(self, database=None):
        return _FakeSession(self._calls)


def _mocked_store() -> tuple[GraphStore, list[tuple[str, dict]]]:
    """A GraphStore wired to a fake driver — captures Cypher without Neo4j."""
    calls: list[tuple[str, dict]] = []
    mocked = GraphStore(Neo4jConfig())
    mocked._driver = _FakeDriver(calls)
    return mocked, calls


@pytest.mark.asyncio
async def test_merge_node_sanitizes_embedded_backtick_in_key():
    """A backtick inside an LLM-derived property key must not let it break
    out of the backtick-quoted Cypher identifier it's wrapped in, and the
    parameter name spliced in as ``$name`` must stay a safe bare identifier
    even though the original key also contains parens/slashes/spaces."""
    store, calls = _mocked_store()
    injected_key = "evil`) DETACH DELETE (n) //"
    await store.merge_node(
        NodeLabel.CONCEPT, "concept_id", "test_bt",
        {"concept_id": "test_bt", injected_key: "payload"},
    )
    assert len(calls) == 1
    query, params = calls[0]
    # Every backtick-quoted identifier in the query is well-formed (even
    # count) — no stray unmatched backtick from the embedded one closing an
    # identifier early.
    assert query.count("`") % 2 == 0
    # The dangerous fragment survives only *inside* a quoted identifier —
    # never as raw, executable-looking Cypher text outside one.
    assert "`evil) DETACH DELETE (n) //`" in query
    # Every $-parameter reference actually spliced into the query text is a
    # safe bare identifier — nothing that could break out of the `$...` slot.
    for pname in re.findall(r"\$(\w+)", query):
        assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", pname)
    assert "payload" in params.values()


@pytest.mark.asyncio
async def test_merge_node_all_backtick_key_is_dropped_not_written():
    """A key that sanitizes to empty (e.g. only backticks) must be dropped
    entirely rather than emitted as an empty backtick-quoted identifier."""
    store, calls = _mocked_store()
    await store.merge_node(
        NodeLabel.CONCEPT, "concept_id", "test_empty_key",
        {"concept_id": "test_empty_key", "```": "should never be written", "kept": "ok"},
    )
    assert len(calls) == 1
    query, params = calls[0]
    assert "should never be written" not in params.values()
    assert "``` " not in query
    assert "kept" in [k for k in params]
    assert params["kept"] == "ok"


@pytest.mark.asyncio
async def test_merge_node_colliding_param_keys_both_persist_distinct_params():
    """Two distinct property keys that sanitize to the same param name
    ("a-b" and "a.b" both -> "a_b") must each keep their own SET fragment
    and their own parameter — neither may silently clobber the other."""
    store, calls = _mocked_store()
    await store.merge_node(
        NodeLabel.CONCEPT, "concept_id", "test_collide",
        {"concept_id": "test_collide", "a-b": "dash-value", "a.b": "dot-value"},
    )
    assert len(calls) == 1
    query, params = calls[0]
    assert "n.`a-b` = $a_b" in query
    assert "n.`a.b` = $a_b_2" in query
    assert params["a_b"] == "dash-value"
    assert params["a_b_2"] == "dot-value"


@pytest.mark.asyncio
async def test_merge_edge_colliding_param_keys_both_persist_distinct_params():
    """Same collision guarantee on the edge-properties path (merge_edge)."""
    store, calls = _mocked_store()
    await store.merge_edge(
        source_label=NodeLabel.CONCEPT,
        source_id_field="concept_id",
        source_id_value="test_src",
        target_label=NodeLabel.CONCEPT,
        target_id_field="concept_id",
        target_id_value="test_tgt",
        rel_type=RelType.DEPENDS_ON,
        properties={"a-b": "dash-value", "a.b": "dot-value"},
    )
    assert len(calls) == 1
    query, params = calls[0]
    assert "r.`a-b` = $a_b" in query
    assert "r.`a.b` = $a_b_2" in query
    assert params["a_b"] == "dash-value"
    assert params["a_b_2"] == "dot-value"


def test_cypher_key_strips_backtick_and_control_chars():
    assert _cypher_key("plain_name") == "plain_name"
    assert _cypher_key("has space") == "`has space`"
    # Embedded backtick is stripped before quoting — it never survives into
    # the returned identifier. Here stripping it happens to leave a plain
    # identifier, so no quoting is needed at all.
    assert _cypher_key("a`b") == "ab"
    # With a character that still forces backtick-quoting, the embedded
    # backtick is still gone and no stray backtick remains inside the quotes.
    quoted = _cypher_key("a`b c")
    assert quoted == "`ab c`"
    assert "`" not in quoted[1:-1]


def test_cypher_key_returns_none_when_sanitized_to_empty():
    assert _cypher_key("```") is None
    assert _cypher_key("\x00\x01\x02") is None


def test_param_key_map_is_collision_free_and_deterministic():
    keys = ["a-b", "a.b", "a b"]
    mapping = _param_key_map(keys)
    assert len(set(mapping.values())) == len(keys)
    # Same input -> same output.
    assert _param_key_map(keys) == mapping


def test_param_key_map_disambiguator_never_collides_with_natural_key():
    """A disambiguated suffix (e.g. "a_b_2") must not collide with another
    key whose own natural mapping is literally "a_b_2"."""
    keys = ["a-b", "a.b", "a_b_2"]
    mapping = _param_key_map(keys)
    assert len(set(mapping.values())) == len(keys)


def test_set_assignments_drops_empty_key_keeps_valid_ones():
    props = {"```": "dropped", "good_key": "kept", "a-b": 1, "a.b": 2}
    parts, params = _set_assignments(props, "n")
    assert len(parts) == 3
    assert len(params) == 3
    assert set(params.values()) == {"kept", 1, 2}
    assert "dropped" not in params.values()


@pytest.mark.asyncio
async def test_merge_edge_property_key_cannot_shadow_match_anchor_params():
    """An LLM-controlled edge-property key like "source id" sanitizes to
    "source_id" — without the reserved-anchor guard it would merge AFTER the
    MATCH anchors in the params dict and silently redirect the edge write to
    a different node. The anchors must keep their values; the property gets
    a disambiguated parameter of its own."""
    store, calls = _mocked_store()
    await store.merge_edge(
        source_label=NodeLabel.CONCEPT,
        source_id_field="concept_id",
        source_id_value="real_src",
        target_label=NodeLabel.CONCEPT,
        target_id_field="concept_id",
        target_id_value="real_tgt",
        rel_type=RelType.DEPENDS_ON,
        properties={"source id": "evil_src", "target.id": "evil_tgt"},
    )
    assert len(calls) == 1
    query, params = calls[0]
    assert params["source_id"] == "real_src"
    assert params["target_id"] == "real_tgt"
    assert params["source_id_2"] == "evil_src"
    assert params["target_id_2"] == "evil_tgt"
    assert "r.`source id` = $source_id_2" in query
    assert "r.`target.id` = $target_id_2" in query


def test_param_key_map_reserved_names_are_never_issued():
    mapping = _param_key_map(["source id", "normal"], reserved={"source_id"})
    assert mapping["source id"] == "source_id_2"
    assert mapping["normal"] == "normal"


# ── Identity boundary (#20 family, 2026-07-10): protected fields never mutate ON MATCH ──


def _split_merge_clauses(query: str) -> tuple[str, str]:
    """Split a MERGE query into its ON CREATE and ON MATCH clause texts."""
    create = query.split("ON CREATE SET", 1)[1].split("ON MATCH SET", 1)[0]
    match = query.split("ON MATCH SET", 1)[1]
    return create, match


@pytest.mark.asyncio
async def test_merge_on_match_never_clobbers_identity_fields():
    """Regression for the 2026-07-09 incident: a NEW source's proposal whose
    proposed_id collides with an EXISTING node used to overwrite that node's
    source_id and canonical_name (observed live: a Razavi topology renamed by a
    PrimeSim chunk). The generated Cypher must set identity fields ON CREATE
    only — the ON MATCH branch may update descriptive props but never identity."""
    store, calls = _mocked_store()
    await store.merge_node(
        NodeLabel.CIRCUIT_TOPOLOGY, "topology_id", "Differential Pair",
        {
            "canonical_name": "Logic High Voltage (logichv)",   # the vandal name
            "source_id": "src_primesim_new",                     # provenance theft
            "retracted": False,                                   # would un-retract
            "description": "updated description is fine",        # legit enrichment
        },
    )
    assert len(calls) == 1
    query, params = calls[0]
    create, match = _split_merge_clauses(query)
    for protected in ("canonical_name", "source_id", "retracted"):
        assert f"n.{protected}" in create, f"{protected} must still be set on CREATE"
        assert f"n.{protected}" not in match, f"{protected} leaked into ON MATCH"
    assert "n.description" in create and "n.description" in match
    # param still bound (harmless — only the CREATE branch references it)
    assert params["source_id"] == "src_primesim_new"


@pytest.mark.asyncio
async def test_write_batch_merge_tx_applies_same_identity_boundary():
    """The tx twin (_merge_node_tx, used by apply_delta/write_batch/import) must
    enforce the same boundary as the session-scoped merge_node — the pair is kept
    in sync by hand, so this test pins the tx side independently."""

    class _FakeTx:
        def __init__(self, calls):
            self._calls = calls

        async def run(self, query, params=None):
            self._calls.append((query, params or {}))

    store, _ = _mocked_store()
    calls: list[tuple[str, dict]] = []
    await store._merge_node_tx(
        _FakeTx(calls), NodeLabel.CONCEPT, "concept_id", "c1",
        {"canonical_name": "Thief", "source_id": "src_thief", "confidence": 0.9},
    )
    query, _params = calls[0]
    create, match = _split_merge_clauses(query)
    assert "n.canonical_name" in create and "n.source_id" in create
    assert "n.canonical_name" not in match and "n.source_id" not in match
    assert "n.confidence" in match


@pytest.mark.asyncio
async def test_update_node_tx_drops_identity_fields_with_warning(caplog):
    """An explicit UPDATE naming an identity field (the LLM NodeUpdate.updates
    path) is dropped at the choke point with a WARNING; non-identity keys in the
    same request still apply."""
    import logging

    class _FakeTx:
        def __init__(self, calls):
            self._calls = calls

        async def run(self, query, params=None):
            self._calls.append((query, params or {}))

    store, _ = _mocked_store()
    calls: list[tuple[str, dict]] = []
    with caplog.at_level(logging.WARNING, logger="openclaw_brain.knowledge.graph.store"):
        await store._update_node_tx(
            _FakeTx(calls), NodeLabel.CONCEPT, "concept_id", "c1",
            {"source_id": "src_thief", "canonical_name": "Thief", "description": "ok"},
        )
    assert len(calls) == 1
    query, params = calls[0]
    assert "n.description" in query
    assert "source_id" not in query.split("SET", 1)[1].replace("$", " ")  # not in SET clause
    assert "n.canonical_name" not in query
    assert any("identity field" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_update_node_tx_all_identity_request_is_noop(caplog):
    """If EVERY requested key is protected, the update degrades to a no-op
    (plus warning) rather than emitting a SET-less broken query."""
    import logging

    class _FakeTx:
        def __init__(self, calls):
            self._calls = calls

        async def run(self, query, params=None):
            self._calls.append((query, params or {}))

    store, _ = _mocked_store()
    calls: list[tuple[str, dict]] = []
    with caplog.at_level(logging.WARNING, logger="openclaw_brain.knowledge.graph.store"):
        await store._update_node_tx(
            _FakeTx(calls), NodeLabel.CONCEPT, "concept_id", "c1",
            {"source_id": "src_thief"},
        )
    assert calls == []


def test_protected_set_covers_every_label_merge_key():
    """Exhaustiveness lock: every NodeLabel's merge-key field must be in
    _PROTECTED_ON_MATCH, so a FOREIGN id key riding inside a props dict can
    never overwrite a node's key. Adding a NodeLabel without extending the
    protected set fails here."""
    from openclaw_brain.knowledge.graph.store import _PROTECTED_ON_MATCH

    for label in NodeLabel:
        id_field = GraphStore._id_field_for_label(label)
        assert id_field in _PROTECTED_ON_MATCH, (
            f"{label.value}'s merge key {id_field!r} missing from _PROTECTED_ON_MATCH")


# ── LLM-proposable label allowlist (F8(a), 2026-07-11): apply_delta rejects an
# out-of-allowlist label at the write choke point ──
#
# Verified incident: 151 live nodes under Memory/Session/SkillRun/Entity were 100%
# LLM-mislabeled EXTRACTION knowledge (the reasoner chose a personal-memory label
# instead of a knowledge one) — those labels are stripped from the company export,
# so real knowledge silently vanished from every shipped artifact. Same
# unguarded-boundary FAMILY as the identity-boundary tests above (#20, commit
# 85575dc/W-A) — there an LLM-controlled property KEY reached MERGE/SET with no
# allowlist; here an LLM-controlled LABEL reaches MERGE with no allowlist.


class _F8AFakeTx:
    async def commit(self):
        pass

    async def rollback(self):
        pass


class _F8AFakeSession:
    def __init__(self):
        self.tx = _F8AFakeTx()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def begin_transaction(self):
        return self.tx


class _F8ACapturingGraphStore(GraphStore):
    """apply_delta builds its own session/tx internally (unlike merge_node /
    _merge_node_tx, which take a tx directly) — mirroring the existing
    test_apply_delta_*_unit pattern, this fakes _session() and captures
    _merge_node_tx / _update_node_tx calls directly instead of raw Cypher."""

    def __init__(self):
        super().__init__(load_config().neo4j)
        self.merged_nodes: list[dict] = []
        self.updated_calls: list[dict] = []

    async def _session(self):
        return _F8AFakeSession()

    async def _merge_node_tx(self, tx, label, id_field, id_value, properties):
        self.merged_nodes.append(
            {
                "label": label,
                "id_field": id_field,
                "id_value": id_value,
                "properties": properties,
            }
        )

    async def _update_node_tx(self, tx, label, id_field, id_value, properties):
        self.updated_calls.append(
            {
                "label": label,
                "id_field": id_field,
                "id_value": id_value,
                "properties": properties,
            }
        )


@pytest.mark.asyncio
async def test_apply_delta_rejects_non_allowlisted_label_in_new_nodes(caplog):
    """A Memory-labeled proposal — the actual live incident shape — is skipped:
    counts['rejected_labels'] == 1, a WARNING names the label/proposed_id/
    canonical_name, and no merge query is issued for it (_merge_node_tx is never
    called for the rejected proposal)."""
    import logging

    store = _F8ACapturingGraphStore()
    delta = GraphDelta(
        new_nodes=[
            NodeProposal(
                proposed_id="test_mislabeled_memory",
                label=NodeLabel.MEMORY,
                canonical_name="Should Have Been A Concept",
                reasoning="unit test — F8(a) rejection",
            ),
        ],
    )

    with caplog.at_level(logging.WARNING, logger="openclaw_brain.knowledge.graph.store"):
        counts = await store.apply_delta(delta)

    assert counts["new_nodes"] == 0
    assert counts["rejected_labels"] == 1
    assert store.merged_nodes == []  # no merge query issued for the rejected proposal
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "Memory" in r.getMessage()
        and "test_mislabeled_memory" in r.getMessage()
        and "Should Have Been A Concept" in r.getMessage()
        for r in warnings
    )


@pytest.mark.asyncio
async def test_apply_delta_rejects_non_allowlisted_label_in_updated_nodes(caplog):
    """updated_nodes gets the same treatment as new_nodes: an update targeting a
    non-allowlisted label (here SkillRun) is skipped+warned — the LLM shouldn't
    touch that node's fields either — and _update_node_tx is never called for it."""
    import logging

    store = _F8ACapturingGraphStore()
    delta = GraphDelta(
        updated_nodes=[
            NodeUpdate(
                existing_node_id="run_123",
                label=NodeLabel.SKILL_RUN,
                updates={"status": "hijacked"},
                reasoning="unit test — F8(a) rejection",
            ),
        ],
    )

    with caplog.at_level(logging.WARNING, logger="openclaw_brain.knowledge.graph.store"):
        counts = await store.apply_delta(delta)

    assert counts["updated_nodes"] == 0
    assert counts["rejected_labels"] == 1
    assert store.updated_calls == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("SkillRun" in r.getMessage() and "run_123" in r.getMessage() for r in warnings)


@pytest.mark.asyncio
async def test_apply_delta_commits_all_allowlisted_labels():
    """All seven LLM-proposable knowledge labels still commit via new_nodes — the
    allowlist must not accidentally exclude a legitimate knowledge label."""
    store = _F8ACapturingGraphStore()
    allowlisted = [
        NodeLabel.CONCEPT, NodeLabel.EQUATION, NodeLabel.PRINCIPLE,
        NodeLabel.CIRCUIT_TOPOLOGY, NodeLabel.PARAMETER, NodeLabel.ASSUMPTION,
        NodeLabel.INSIGHT,
    ]
    delta = GraphDelta(
        new_nodes=[
            NodeProposal(
                proposed_id=f"test_allow_{label.value.lower()}",
                label=label,
                canonical_name=f"Test {label.value}",
                reasoning="unit test — F8(a) allowlist coverage",
            )
            for label in allowlisted
        ],
    )

    counts = await store.apply_delta(delta)

    assert counts["new_nodes"] == 7
    assert counts["rejected_labels"] == 0
    assert len(store.merged_nodes) == 7
    assert {n["label"] for n in store.merged_nodes} == set(allowlisted)


def test_llm_proposable_labels_partition_covers_every_label():
    """Exhaustiveness lock: every NodeLabel must land in EXACTLY one of the four
    partitions — the LLM-proposable allowlist, or one of the three excluded
    groups (personal-memory/operational, curated-write-path-only, substrate-
    projection-only). A future NodeLabel addition that isn't explicitly routed
    into one of these fails here, forcing an explicit decision (mirrors
    test_protected_set_covers_every_label_merge_key's philosophy)."""
    from openclaw_brain.knowledge.graph.store import _LLM_PROPOSABLE_LABELS

    personal_memory_operational = {
        NodeLabel.MEMORY, NodeLabel.SESSION, NodeLabel.SKILL_RUN, NodeLabel.ENTITY,
        NodeLabel.SOURCE, NodeLabel.SOURCE_CHUNK,
    }
    curated_write_path_only = {
        NodeLabel.HYPOTHESIS, NodeLabel.DESIGN_DECISION, NodeLabel.BENCH_RESULT,
        NodeLabel.LEARNER, NodeLabel.ASSESSMENT,
    }
    substrate_projection_only = {
        NodeLabel.SPECIMEN, NodeLabel.CLAIM_CARD, NodeLabel.REGULARITY,
        # symbolic anchors (2026-07-28): written only by the derivation/anchor tooling,
        # never LLM-authored — same discipline as the other projection labels.
        NodeLabel.SYMBOLIC_DERIVATION,
    }
    excluded = personal_memory_operational | curated_write_path_only | substrate_projection_only

    # No label is both proposable and excluded...
    assert _LLM_PROPOSABLE_LABELS & excluded == set()
    # ...and together they cover every NodeLabel member with nothing left over.
    assert _LLM_PROPOSABLE_LABELS | excluded == set(NodeLabel)


# ── Vector-index query failure visibility (find_match_candidates / find_similar_concepts) ──
#
# Both methods wrap their vector-index query in a try/except so a genuinely-missing index
# degrades matching to text-only candidates rather than raising. The bare `except Exception:
# pass` made a REAL error (dimension mismatch, transient Neo4j fault) indistinguishable from
# "index not yet created" — these tests exercise the fix: log once per store instance (not per
# call — a multi-thousand-chunk ingest would spam) and expose a counter tests/callers can read.


class _EmptyAsyncResult:
    """Async-iterable Neo4j result stand-in yielding zero records (both the `async for` and
    `.data()` consumption styles used across store.py's query methods)."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def data(self):
        return []

    async def single(self):
        return None


class _FailingVectorSession:
    """Raises on any vector-index query; text/alias queries succeed with zero rows."""

    def __init__(self, calls: list[tuple[str, dict]]):
        self._calls = calls

    async def run(self, query, params=None):
        self._calls.append((query, params or {}))
        if "db.index.vector.queryNodes" in query:
            raise RuntimeError("vector index unavailable (dimension mismatch)")
        return _EmptyAsyncResult()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FailingVectorDriver:
    def __init__(self, calls: list[tuple[str, dict]]):
        self._calls = calls

    def session(self, database=None):
        return _FailingVectorSession(self._calls)


def _failing_vector_store() -> tuple[GraphStore, list[tuple[str, dict]]]:
    calls: list[tuple[str, dict]] = []
    store = GraphStore(Neo4jConfig())
    store._driver = _FailingVectorDriver(calls)
    return store, calls


@pytest.mark.asyncio
async def test_find_match_candidates_vector_failure_logs_once_and_counts(caplog):
    import logging

    store, _ = _failing_vector_store()
    with caplog.at_level(logging.WARNING, logger="openclaw_brain.knowledge.graph.store"):
        r1 = await store.find_match_candidates("cascode", embedding=[0.1, 0.2], limit=5)
        r2 = await store.find_match_candidates("mirror", embedding=[0.1, 0.2], limit=5)

    assert r1 == [] and r2 == []  # degrades to text-only (here: empty) — never raises
    assert store._vector_index_error_count == 2
    vector_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "vector index" in r.getMessage().lower()
    ]
    assert len(vector_warnings) == 1  # logged once per store instance, not per call


@pytest.mark.asyncio
async def test_find_similar_concepts_vector_failure_logs_once_and_counts(caplog):
    import logging

    store, _ = _failing_vector_store()
    with caplog.at_level(logging.WARNING, logger="openclaw_brain.knowledge.graph.store"):
        await store.find_similar_concepts("cascode", embedding=[0.1, 0.2], limit=5)
        await store.find_similar_concepts("mirror", embedding=[0.1, 0.2], limit=5)

    assert store._vector_index_error_count == 2
    vector_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "vector index" in r.getMessage().lower()
    ]
    assert len(vector_warnings) == 1


@pytest.mark.asyncio
async def test_vector_index_failure_counter_shared_across_both_methods(caplog):
    """The counter/flag live on the store instance, not per-method — a mix of
    find_match_candidates and find_similar_concepts calls still only logs once."""
    import logging

    store, _ = _failing_vector_store()
    with caplog.at_level(logging.WARNING, logger="openclaw_brain.knowledge.graph.store"):
        await store.find_match_candidates("cascode", embedding=[0.1, 0.2], limit=5)
        await store.find_similar_concepts("cascode", embedding=[0.1, 0.2], limit=5)

    assert store._vector_index_error_count == 2
    assert store._vector_index_warned is True
    vector_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "vector index" in r.getMessage().lower()
    ]
    assert len(vector_warnings) == 1


@pytest.mark.asyncio
async def test_vector_index_no_failure_when_no_embedding_given():
    """REGRESSION: without an embedding, neither method attempts the vector query at all — the
    counter must stay at zero (unchanged behavior)."""
    store, _ = _failing_vector_store()
    await store.find_match_candidates("cascode", embedding=None, limit=5)
    assert store._vector_index_error_count == 0


# ── C-bundle (F9): Neo4j array rules — the chunk-2084 incident class ──


def test_sanitize_drops_none_inside_arrays():
    """Neo4j rejects null inside arrays; a [str, None] list must ship without the None
    (the old code's 'primitive or None' check let it through → Neo.ClientError TypeError)."""
    from openclaw_brain.knowledge.graph.store import _sanitize_neo4j_value

    assert _sanitize_neo4j_value(["a", None, "b"]) == ["a", "b"]


def test_sanitize_coerces_mixed_primitive_arrays_to_homogeneous():
    """Neo4j arrays must be homogeneous. Mixed str/int → all-str; int/float → all-float;
    bool must not silently merge into int (bool is an int subclass in Python)."""
    from openclaw_brain.knowledge.graph.store import _sanitize_neo4j_value

    assert _sanitize_neo4j_value(["a", 1]) == ["a", "1"]
    assert _sanitize_neo4j_value([1, 2.5]) == [1.0, 2.5]
    assert _sanitize_neo4j_value([True, 1]) == ["True", "1"]
    assert _sanitize_neo4j_value([1, 2, 3]) == [1, 2, 3]  # already homogeneous — untouched


@pytest.mark.asyncio
async def test_apply_delta_equation_list_latex_is_joined_to_scalar():
    """LLMs occasionally emit LaTeX as a list of strings (2 live Razavi nodes shipped as
    LIST<STRING>); every downstream reader needs a scalar — apply_delta joins it."""
    from openclaw_brain.knowledge.graph.schema import GraphDelta, NodeProposal

    store, _ = _mocked_store()
    calls: list[tuple[str, dict]] = []

    class _Tx:
        async def run(self, q, p=None):
            calls.append((q, p or {}))

        async def commit(self): ...
        async def rollback(self): ...

    class _Sess:
        async def begin_transaction(self):
            return _Tx()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    async def _sess():
        return _Sess()

    store._session = _sess
    delta = GraphDelta(new_nodes=[NodeProposal(
        proposed_id="eq_list", label="Equation", canonical_name="Multi-line",
        properties={"latex": ["a=b", "c=d"]}, reasoning="")])
    await store.apply_delta(delta)
    params = calls[0][1]
    assert params.get("canonical_latex") == "a=b ; c=d"
    assert params.get("latex") == "a=b ; c=d"


@pytest.mark.asyncio
async def test_neighborhood_one_row_per_neighbor_with_its_own_edge(store: GraphStore):
    """Audit D-5 regression: the old query UNWOUND every relationship on the path, so a 2-hop
    neighbor came back once per hop AND was paired with the first hop's edge — a relationship
    not incident to it (measured live: 12 of 30 rows misattributed, 1.5-2.0x duplication, and
    the production limit of 6 bought only 4 distinct neighbors)."""
    for cid in ("nb_a", "nb_b", "nb_c"):
        await store.merge_node(NodeLabel.CONCEPT, "concept_id", cid, {"canonical_name": cid})
    # a -[DEPENDS_ON]-> b -[RELATES_TO]-> c : c is 2 hops from a, reached by RELATES_TO
    await store.merge_edge(NodeLabel.CONCEPT, "concept_id", "nb_a",
                           NodeLabel.CONCEPT, "concept_id", "nb_b",
                           RelType.DEPENDS_ON, properties={"rationale": "a->b"})
    await store.merge_edge(NodeLabel.CONCEPT, "concept_id", "nb_b",
                           NodeLabel.CONCEPT, "concept_id", "nb_c",
                           RelType.RELATES_TO, properties={"rationale": "b->c"})

    rows = await store.get_neighborhood(NodeLabel.CONCEPT, "concept_id", "nb_a", hops=2, limit=50)
    by_id = {r["node"]["concept_id"]: r for r in rows if r["node"].get("concept_id")}

    ids = [r["node"].get("concept_id") for r in rows if r["node"].get("concept_id")]
    assert len(ids) == len(set(ids)), f"neighbor duplicated: {ids}"
    assert "nb_a" not in ids, "start node returned as its own neighbor"

    # each neighbor carries the edge that actually reaches IT, not some other hop's
    assert by_id["nb_b"]["rel_type"] == "DEPENDS_ON"
    assert by_id["nb_b"]["hop_distance"] == 1
    assert by_id["nb_c"]["rel_type"] == "RELATES_TO", "2-hop neighbor mis-attributed to hop 1"
    assert by_id["nb_c"]["rationale"] == "b->c"
    assert by_id["nb_c"]["hop_distance"] == 2
    # traversal is undirected, so direction must be reported rather than lost
    assert by_id["nb_b"]["edge_direction"] == "->"


@pytest.mark.asyncio
async def test_counterfeit_evidence_edges_are_refused(store: GraphStore):
    """Audit A1/D-2: schema.py has always called REALIZES/HAS_CLAIM/GROUNDS/SUPPORTED_BY/ABOUT
    'projector-only discipline', but nothing enforced it, so the reasoning LLM minted ~664 of
    978 tier-3 edges between arbitrary nodes (e.g. (Concept)-[:SUPPORTED_BY]->(Concept)). Every
    'what is grounded by simulation' query reads these."""
    for cid in ("ev_a", "ev_b"):
        await store.merge_node(NodeLabel.CONCEPT, "concept_id", cid, {"canonical_name": cid})

    for rel in (RelType.SUPPORTED_BY, RelType.HAS_CLAIM, RelType.REALIZES,
                RelType.GROUNDS, RelType.ABOUT):
        await store.merge_edge(NodeLabel.CONCEPT, "concept_id", "ev_a",
                               NodeLabel.CONCEPT, "concept_id", "ev_b",
                               rel, properties={"rationale": "counterfeit"})
        rows = await store.run_read_query(
            f"MATCH (:Concept {{concept_id:'ev_a'}})-[r:`{rel.value}`]->"
            f"(:Concept {{concept_id:'ev_b'}}) RETURN count(r) AS n")
        assert rows[0]["n"] == 0, f"{rel.value} between two Concepts was written"

    # ...and a legitimate one still goes through untouched
    await store.merge_node(NodeLabel.SPECIMEN, "spec_id", "ev_spec", {})
    await store.merge_node(NodeLabel.CLAIM_CARD, "claim_id", "ev_spec:c1", {})
    await store.merge_edge(NodeLabel.SPECIMEN, "spec_id", "ev_spec",
                           NodeLabel.CLAIM_CARD, "claim_id", "ev_spec:c1",
                           RelType.HAS_CLAIM, properties={})
    rows = await store.run_read_query(
        "MATCH (:Specimen {spec_id:'ev_spec'})-[r:HAS_CLAIM]->"
        "(:ClaimCard {claim_id:'ev_spec:c1'}) RETURN count(r) AS n")
    assert rows[0]["n"] == 1, "legitimate projector edge was blocked"


def test_evidence_edge_endpoints_match_the_real_minters():
    """Unit: the allowlist must mirror executable/projection.py and executable/laws.py."""
    from openclaw_brain.knowledge.graph.store import _evidence_edge_violation
    assert _evidence_edge_violation(
        RelType.HAS_CLAIM, NodeLabel.SPECIMEN, NodeLabel.CLAIM_CARD) is None
    assert _evidence_edge_violation(
        RelType.SUPPORTED_BY, NodeLabel.REGULARITY, NodeLabel.CLAIM_CARD) is None
    assert _evidence_edge_violation(
        RelType.GROUNDS, NodeLabel.CLAIM_CARD, NodeLabel.PARAMETER) is None
    assert _evidence_edge_violation(
        RelType.ABOUT, NodeLabel.REGULARITY, NodeLabel.CIRCUIT_TOPOLOGY) is None
    assert _evidence_edge_violation(
        RelType.REALIZES, NodeLabel.SPECIMEN, NodeLabel.CIRCUIT_TOPOLOGY) is None
    # non-evidence types are never gated
    assert _evidence_edge_violation(
        RelType.RELATES_TO, NodeLabel.CONCEPT, NodeLabel.CONCEPT) is None
    # and the real-world counterfeit shape is caught
    v = _evidence_edge_violation(RelType.SUPPORTED_BY, NodeLabel.CONCEPT, NodeLabel.CONCEPT)
    assert v and "Regularity" in v and "Concept" in v


def test_filter_proposal_properties_blocks_identity_and_lifecycle_keys():
    """Migration step 0 write-channel closure (ADR-046 D6): an LLM proposal's free-form
    properties dict must not clobber merge keys, provenance, lifecycle state, internal
    fields, or assign ids — while domain content passes through untouched."""
    from openclaw_brain.knowledge.graph.store import _filter_proposal_properties

    kept, dropped = _filter_proposal_properties({
        "concept_id": "evil-rekey",          # merge key
        "source_id": "provenance-theft",     # protected
        "canonical_name": "rename-attack",   # protected
        "retracted": False,                  # lifecycle
        "_created_at": "2020-01-01",         # internal namespace
        "figure_id": "id-assignment",        # *_id blanket
        "canonical_latex": "A_v = -g_m R_D", # domain content — keep
        "units": "V/V",                      # domain content — keep
        "condition": "saturation",           # domain content — keep
    })
    assert kept == {
        "canonical_latex": "A_v = -g_m R_D", "units": "V/V", "condition": "saturation",
    }
    assert sorted(dropped) == [
        "_created_at", "canonical_name", "concept_id", "figure_id",
        "retracted", "source_id",
    ]


def test_filter_proposal_properties_empty_dict_passthrough():
    from openclaw_brain.knowledge.graph.store import _filter_proposal_properties
    kept, dropped = _filter_proposal_properties({})
    assert kept == {} and dropped == []
