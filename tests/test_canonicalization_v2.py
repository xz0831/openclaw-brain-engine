"""Tests for the production precision MERGE path (merge_tier + guards + clustering).

Anchored on the ER-gold precision-critical cases (experiments/er_gold_set.json) and the live
false-merge patterns the multi-strategy comparison surfaced (experiments/CANONICALIZATION_PROMOTION.md).
The frozen grader path (same_entity) is covered by the conformance/shadow tests — not here.
"""
from openclaw_brain.knowledge.canonicalization import (
    acronym_same, numeric_conflict, qualifier_conflict, merge_tier, cluster_merge_edges,
)


# ── numeric conflict (Pitfall 27.2 vs 27.6 false-merge class) ──
def test_numeric_conflict_distinguishes_numbered_items():
    assert numeric_conflict("Pitfall Prevention 27.2", "Pitfall Prevention 27.6")
    assert numeric_conflict("Example 5.1", "Example 5.2")
    assert numeric_conflict("3T Pixel", "4T Pixel")
    assert not numeric_conflict("Threshold Voltage", "Vth")
    assert not numeric_conflict("MOSFET Triode Region", "MOSFET Linear Region")


# ── qualifier conflict (the precision fix the grader containment lacks) ──
def test_qualifier_conflict_rejects_specializations():
    assert qualifier_conflict("Threshold Voltage", "Threshold Voltage Mismatch")
    assert qualifier_conflict("Dark Current", "Dark Current Shot Noise")
    assert qualifier_conflict("Subthreshold Slope", "Subthreshold Slope Degradation")


def test_qualifier_conflict_tolerates_generic_extension():
    assert not qualifier_conflict("Autozero", "Autozero Operation")
    assert not qualifier_conflict("Floating Diffusion", "Floating Diffusion Node")
    # both-sides-unique is not a pure specialization → defer (no conflict)
    assert not qualifier_conflict("Dark Current", "Dark Signal")


# ── acronym / initialism ──
def test_acronym_same_catches_initialisms():
    assert acronym_same("FWC", "Full Well Capacity")
    assert acronym_same("Quantum Efficiency", "QE")
    assert acronym_same("S.S.", "Subthreshold Slope")
    assert acronym_same("FD Node", "Floating Diffusion")  # generic trailing word stripped


def test_acronym_same_rejects_noninitialisms():
    assert not acronym_same("Body Effect", "Backgate Bias Effect")
    assert not acronym_same("Rolling Shutter", "Global Shutter")


# ── merge_tier (the production decision) ──
def test_merge_tier_auto_merges_lexical_identity():
    assert merge_tier("Quantum Efficiency", "QE") == "MERGE"
    assert merge_tier("MOSFET Triode Region", "mosfet triode region") == "MERGE"


def test_merge_tier_rejects_conflicts_regardless_of_cosine():
    # high cosine must NOT override a qualifier/numeric guard
    assert merge_tier("Threshold Voltage", "Threshold Voltage Mismatch", cosine=0.95) == "REJECT"
    assert merge_tier("Pitfall Prevention 27.2", "Pitfall Prevention 27.6", cosine=0.94) == "REJECT"


def test_merge_tier_routes_embedding_candidates_to_verify_not_automerge():
    # a high-cosine non-lexical pair is a CANDIDATE for the LLM, never an auto-merge
    assert merge_tier("Rolling Shutter", "Global Shutter", cosine=0.82) == "VERIFY"
    assert merge_tier("Body Effect", "Backgate Bias Effect", cosine=0.84) == "VERIFY"


def test_merge_tier_rejects_below_floor():
    assert merge_tier("Shot Noise", "Thermal Noise", cosine=0.55) == "REJECT"
    assert merge_tier("Shot Noise", "Thermal Noise", cosine=None) == "REJECT"


# ── clustering ──
def test_cluster_merge_edges_union_find():
    ids = ["a", "b", "c", "d", "e"]
    edges = [("a", "b"), ("b", "c"), ("d", "e")]
    clusters = sorted(sorted(c) for c in cluster_merge_edges(ids, edges))
    assert clusters == [["a", "b", "c"], ["d", "e"]]


def test_cluster_merge_edges_ignores_singletons_and_unknown_ids():
    clusters = cluster_merge_edges(["a", "b"], [("a", "x")])  # x not in ids
    assert clusters == []
