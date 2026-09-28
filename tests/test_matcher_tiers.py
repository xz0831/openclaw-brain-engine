"""Unit tests for the embedding-first tier decision in ConceptMatcher.

No Neo4j and no LLM needed — GraphStore and MatchVerifier are stubbed.
"""

import pytest

from openclaw_brain.config import MatcherConfig
from openclaw_brain.knowledge.extraction.models import ConceptMention, ExtractionResult
from openclaw_brain.knowledge.graph.store import _vector_score_to_cosine
from openclaw_brain.knowledge.reasoning.matcher import ConceptMatcher, _conflict_guard
from openclaw_brain.knowledge.reasoning.verifier import VerifyResult


def test_vector_score_to_cosine_conversion():
    # Neo4j cosine index: score = (1 + cos) / 2
    assert _vector_score_to_cosine(1.0) == pytest.approx(1.0)
    assert _vector_score_to_cosine(0.5) == pytest.approx(0.0)
    assert _vector_score_to_cosine(0.0) == pytest.approx(-1.0)
    assert _vector_score_to_cosine(0.9) == pytest.approx(0.8)


class StubGraph:
    """find_match_candidates returns a fixed candidate list."""
    def __init__(self, entries):
        self._entries = entries

    async def find_match_candidates(self, name, embedding=None, limit=8):
        return self._entries

    async def find_similar_concepts(self, name, embedding=None, limit=5):
        return [e["node"] for e in self._entries]


class StubVerifier:
    def __init__(self, verdict):
        self._verdict = verdict
        self.calls = 0

    async def verify(self, name, description, candidate, neighbor_names=None):
        self.calls += 1
        return VerifyResult(verdict=self._verdict, candidate_id=candidate.get("concept_id", ""))


def _node(cid, name, aliases=None, description=""):
    return {"concept_id": cid, "canonical_name": name,
            "aliases": aliases or [], "description": description}


def _extraction(name, description="a placeholder description"):
    return ExtractionResult(
        chunk_id="c1", concepts=[ConceptMention(name=name, description=description)]
    )


def _matcher(entries, verdict=None, **cfg_kwargs):
    cfg = MatcherConfig(**cfg_kwargs)
    verifier = StubVerifier(verdict) if verdict else None
    m = ConceptMatcher(StubGraph(entries), matcher_config=cfg, verifier=verifier)
    return m, verifier


@pytest.mark.asyncio
async def test_tier0_alias_exact_match_wins_over_low_cosine():
    entries = [{"node": _node("c_be", "Body Effect", aliases=["Backgate Bias"]),
                "cos": 0.55, "text_hit": True}]
    m, _ = _matcher(entries)
    result = await m.match(_extraction("Backgate Bias"))
    assert len(result.matched) == 1
    assert result.matched[0].match_method == "name"


@pytest.mark.asyncio
async def test_tier1_high_cosine_auto_match():
    entries = [{"node": _node("c_kts", "kTC Noise", description="thermal reset noise"),
                "cos": 0.91, "text_hit": False}]
    m, _ = _matcher(entries)
    result = await m.match(_extraction("Reset Noise", "kTC thermal noise sampled on a capacitor"))
    assert len(result.matched) == 1
    assert result.matched[0].match_method == "embedding"
    assert result.matched[0].similarity == pytest.approx(0.91)


@pytest.mark.asyncio
async def test_tier1_guard_blocks_digit_designator_mismatch():
    # 3T vs 4T pixel: cosine high, but digit tokens differ → must NOT auto-match.
    entries = [{"node": _node("c_4t", "4T Pixel"), "cos": 0.93, "text_hit": False}]
    m, _ = _matcher(entries)  # no verifier → guard drops it to ambiguous band
    result = await m.match(_extraction("3T Pixel", "three transistor active pixel"))
    assert len(result.matched) == 0  # went to ambiguous (band) — never auto-matched
    assert len(result.ambiguous) == 1


def test_conflict_guard_unit():
    c3t = ConceptMention(name="3T Pixel", description="three transistor APS")
    assert _conflict_guard(c3t, _node("x", "4T Pixel"), name_sim=0.8) is True
    same = ConceptMention(name="Reset Noise", description="kTC noise on the floating diffusion")
    assert _conflict_guard(same, _node("x", "kTC Noise", description="reset noise"), name_sim=0.4) is False


@pytest.mark.asyncio
async def test_tier2_verifier_same_matches():
    entries = [{"node": _node("c_be", "Body Effect"), "cos": 0.72, "text_hit": False}]
    m, v = _matcher(entries, verdict="SAME")
    result = await m.match(_extraction("Backgate Bias Effect", "Vth shift from source-body voltage"))
    assert len(result.matched) == 1
    assert result.matched[0].match_method == "verified"
    assert v.calls == 1


@pytest.mark.asyncio
async def test_tier2_verifier_different_goes_new():
    entries = [{"node": _node("c_prnu", "PRNU"), "cos": 0.68, "text_hit": False}]
    m, v = _matcher(entries, verdict="DIFFERENT")
    result = await m.match(_extraction("DSNU", "dark signal non-uniformity"))
    assert len(result.new_concepts) == 1
    assert v.calls == 1


@pytest.mark.asyncio
async def test_tier2_unsure_falls_to_ambiguous():
    entries = [{"node": _node("c_x", "Charge Injection"), "cos": 0.7, "text_hit": False}]
    m, v = _matcher(entries, verdict="UNSURE")
    result = await m.match(_extraction("Clock Feedthrough", "switch charge coupling"))
    assert len(result.ambiguous) == 1


@pytest.mark.asyncio
async def test_tier2_budget_exhausted_falls_to_ambiguous():
    entries = [{"node": _node("c_x", "Some Concept"), "cos": 0.7, "text_hit": False}]
    m, v = _matcher(entries, verdict="SAME", max_verify_per_chunk=0)
    result = await m.match(_extraction("Other Concept", "another description"))
    assert len(result.ambiguous) == 1
    assert v.calls == 0  # budget 0 → verifier never invoked


@pytest.mark.asyncio
async def test_tier3_low_signal_is_new():
    entries = [{"node": _node("c_x", "Bandgap Reference"), "cos": 0.3, "text_hit": False}]
    m, _ = _matcher(entries)
    result = await m.match(_extraction("Quantum Efficiency", "photon-to-electron conversion ratio"))
    assert len(result.new_concepts) == 1


@pytest.mark.asyncio
async def test_legacy_mode_preserves_old_behavior():
    entries = [{"node": _node("c_m", "MOSFET"), "cos": None, "text_hit": True}]
    m, _ = _matcher(entries, use_embedding_decision=False)
    result = await m.match(_extraction("MOSFET", "transistor"))
    assert len(result.matched) == 1
    assert result.matched[0].match_method == "name"
