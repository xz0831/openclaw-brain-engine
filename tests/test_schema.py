"""Tests for knowledge graph schema definitions."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from openclaw_brain.knowledge.graph.schema import (
    ConceptNode,
    EdgeProperties,
    EdgeProposal,
    GraphDelta,
    InsightProposal,
    NodeLabel,
    NodeProposal,
    NodeUpdate,
    RelType,
)


def test_concept_node_defaults():
    node = ConceptNode(
        concept_id="common_source_amplifier",
        canonical_name="Common-Source Amplifier",
        description="A basic MOSFET amplifier topology",
        domain="analog_circuits",
    )
    assert node.concept_id == "common_source_amplifier"
    assert node.confidence == 0.7
    assert node.reinforcement_count == 1
    assert node.granularity == "atomic"


def test_edge_properties():
    edge = EdgeProperties(
        rationale="CS amplifier gain is defined by Av = -gm*RD",
        confidence=0.9,
        evidence_sources=["chunk_razavi_ch4_p1"],
        created_by="reasoning",
    )
    assert edge.reinforcement_count == 1
    assert edge.confidence == 0.9
    assert isinstance(edge.created_at, datetime)


def test_graph_delta_construction():
    delta = GraphDelta(
        new_nodes=[
            NodeProposal(
                proposed_id="common_source_amplifier",
                label=NodeLabel.CIRCUIT_TOPOLOGY,
                canonical_name="Common-Source Amplifier",
                description="A single-transistor amplifier topology",
                domain="analog_circuits",
                confidence=0.9,
                properties={"function": "amplifier"},
                evidence_chunk_ids=["chunk_001"],
                reasoning="New topology not in graph",
            )
        ],
        new_edges=[
            EdgeProposal(
                source_ref="common_source_amplifier",
                target_ref="voltage_gain",
                relationship_type=RelType.HAS_PARAMETER,
                rationale="CS amplifier has voltage gain as key spec",
                confidence=0.85,
            )
        ],
        insights=[
            InsightProposal(
                statement="CS amplifier gain-bandwidth tradeoff mirrors RC pole behavior",
                related_concept_ids=["common_source_amplifier", "rc_time_constant"],
                bridge_type="cross-domain",
                confidence=0.6,
            )
        ],
    )
    assert len(delta.new_nodes) == 1
    assert len(delta.new_edges) == 1
    assert len(delta.insights) == 1
    assert delta.new_nodes[0].label == NodeLabel.CIRCUIT_TOPOLOGY


def test_node_update():
    update = NodeUpdate(
        existing_node_id="mosfet_iv",
        label=NodeLabel.EQUATION,
        updates={"assumptions": ["square-law region", "VDS > VGS - Vth"]},
        reasoning="Adding explicit saturation condition",
    )
    assert update.existing_node_id == "mosfet_iv"
    assert "square-law region" in update.updates["assumptions"]


def test_new_node_omits_reasoning_still_constructs():
    """claude-sonnet-5 (and any compliant-but-terse model) may omit NodeProposal.reasoning —
    it's a purely descriptive/rationale field, not structural identity, so it must default
    rather than raise `reasoning Field required`. Structural fields (proposed_id, label,
    canonical_name) stay required."""
    delta = GraphDelta(new_nodes=[
        NodeProposal(
            proposed_id="folded_cascode_ota",
            label=NodeLabel.CIRCUIT_TOPOLOGY,
            canonical_name="Folded Cascode OTA",
            # reasoning omitted entirely
        )
    ])
    assert len(delta.new_nodes) == 1
    assert delta.new_nodes[0].reasoning == ""


def test_new_edge_omits_rationale_still_constructs():
    """Same tolerance for EdgeProposal.rationale — structural fields (source_ref, target_ref,
    relationship_type) stay required."""
    edge = EdgeProposal(
        source_ref="a",
        target_ref="b",
        relationship_type=RelType.DEPENDS_ON,
        # rationale omitted entirely
    )
    assert edge.rationale == ""


def test_insight_omits_related_concept_ids_still_constructs():
    """InsightProposal.related_concept_ids may be omitted by a terse model; statement (the
    insight's actual content) stays required."""
    insight = InsightProposal(statement="A trades area for leakage mitigation")
    assert insight.related_concept_ids == []


def test_new_node_still_requires_structural_fields():
    """A genuinely malformed NodeProposal (missing structural identity) must still be
    rejected — defaulting descriptive fields must not weaken this."""
    with pytest.raises(ValidationError):
        NodeProposal(label=NodeLabel.CONCEPT)  # missing proposed_id + canonical_name


def test_new_edge_still_requires_structural_fields():
    """A genuinely malformed EdgeProposal (missing structural refs) must still be rejected."""
    with pytest.raises(ValidationError):
        EdgeProposal(relationship_type=RelType.DEPENDS_ON)  # missing source_ref + target_ref


def test_all_node_labels():
    """Verify all expected labels are defined."""
    expected = {
        "Concept", "Equation", "Principle", "CircuitTopology",
        "Parameter", "Assumption", "Source", "SourceChunk", "Insight",
        "Memory", "Entity", "Session", "SkillRun",
        "Hypothesis", "DesignDecision", "BenchResult",
        "Specimen", "ClaimCard",  # executable-circuit substrate (v2)
        "Regularity",  # law-tier graph representation (spec 2026-07-04)
        "Learner", "Assessment",  # S5 learner model (ADR-044 D4, spec 2026-07-10)
        "SymbolicDerivation",  # symbolic anchors (2026-07-20 structure, registered 2026-07-28)
    }
    actual = {label.value for label in NodeLabel}
    assert actual == expected


def test_all_rel_types():
    """Verify key relationship types are defined."""
    key_rels = {
        "USES_EQUATION", "DEPENDS_ON", "DERIVED_FROM",
        "TRADES_OFF", "CONFIRMS", "CONTRADICTS", "BRIDGES_TO",
    }
    actual = {rt.value for rt in RelType}
    assert key_rels.issubset(actual)


# ── S5 learner model ──


def test_learner_model_rel_types_defined():
    """UNDERSTANDS (Learner -> target, rolled-up) and ASSESSES (Assessment -> target,
    immutable event) — S5_LEARNER_MODEL_DESIGN.md §6.1 item 1."""
    assert RelType.UNDERSTANDS.value == "UNDERSTANDS"
    assert RelType.ASSESSES.value == "ASSESSES"


def test_learner_node_defaults():
    from openclaw_brain.knowledge.graph.schema import LearnerNode

    node = LearnerNode(learner_id="rick")
    assert node.learner_id == "rick"


def test_assessment_node_defaults():
    from openclaw_brain.knowledge.graph.schema import AssessmentNode

    node = AssessmentNode(
        assessment_id="assess_20260710_000000_000000",
        learner_id="rick",
        verdict="understood",
    )
    assert node.evidence == ""
    assert isinstance(node.created_at, datetime)
