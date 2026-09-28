"""Tests for deterministic scope-widening grounding guard."""

from openclaw_brain.knowledge.extraction.grounding import verify_grounding
from openclaw_brain.knowledge.extraction.models import ConceptMention, ExtractionResult
from openclaw_brain.knowledge.graph.schema import GraphDelta, NodeLabel, NodeProposal
from openclaw_brain.knowledge.pipeline import _apply_grounding_scope_penalties


def test_t2_scope_widening_fixture_flags_surviving_concept():
    chunk = "Under CMS the shot-noise integration variance increases with M, with alpha_shot = M/3."
    extraction = ExtractionResult(
        chunk_id="t2",
        concepts=[
            ConceptMention(
                name="Shot-Noise Integration Variance",
                description="Total noise variance increases linearly with M.",
            )
        ],
    )

    result = verify_grounding(extraction, chunk)

    assert len(result.concepts) == 1
    assert result.concepts[0].grounding_flags == ["scope_widened"]


def test_correctly_scoped_directional_claim_has_no_scope_flag():
    chunk = "Under CMS the shot-noise integration variance increases with M, with alpha_shot = M/3."
    extraction = ExtractionResult(
        chunk_id="t2",
        concepts=[
            ConceptMention(
                name="Shot-Noise Integration Variance",
                description="Under CMS, shot-noise integration variance increases linearly with M.",
            )
        ],
    )

    result = verify_grounding(extraction, chunk)

    assert len(result.concepts) == 1
    assert result.concepts[0].grounding_flags == []


def test_universalizing_token_present_in_chunk_has_no_false_positive():
    chunk = "Under CMS the total noise variance increases linearly with M."
    extraction = ExtractionResult(
        chunk_id="t2",
        concepts=[
            ConceptMention(
                name="Total Noise Variance",
                description="Total noise variance increases linearly with M.",
            )
        ],
    )

    result = verify_grounding(extraction, chunk)

    assert len(result.concepts) == 1
    assert result.concepts[0].grounding_flags == []


def test_pipeline_penalty_persists_scope_flag_for_matching_proposal():
    extraction = ExtractionResult(
        chunk_id="t2",
        concepts=[
            ConceptMention(
                name="Shot-Noise Integration Variance",
                description="Total noise variance increases linearly with M.",
                grounding_flags=["scope_widened"],
            )
        ],
    )
    delta = GraphDelta(
        new_nodes=[
            NodeProposal(
                proposed_id="shot_noise_integration_variance",
                label=NodeLabel.CONCEPT,
                canonical_name="Shot-Noise Integration Variance",
                description="Total noise variance increases linearly with M.",
                confidence=0.7,
                reasoning="New concept from chunk.",
            )
        ]
    )

    _apply_grounding_scope_penalties(delta, extraction)

    proposal = delta.new_nodes[0]
    assert proposal.confidence == 0.56
    assert proposal.properties["grounding_flags"] == ["scope_widened"]
