"""Unit tests for the consolidation engine (no Neo4j — store stubbed)."""

import json

import pytest

from openclaw_brain.config import BrainConfig
from openclaw_brain.knowledge.consolidation import ConsolidationEngine, MergeCandidate


def _concept(cid, name, embedding=None, aliases=None, description="", degree=0,
             reinforcement=0, created_at="2026-01-01"):
    return {"id": cid, "name": name, "embedding": embedding, "aliases": aliases or [],
            "description": description, "degree": degree,
            "reinforcement": reinforcement, "created_at": created_at}


class StubStore:
    def __init__(self, concepts):
        self._concepts = concepts
        self.merges = []

    async def run_read_query(self, query, params):
        return self._concepts

    async def merge_concepts(self, primary, dup):
        self.merges.append((primary, dup))
        return {"primary_id": primary, "deleted": dup, "rewired_out": 0, "rewired_in": 0,
                "gap_props_copied": 0, "duplicate_snapshot": {}, "rewired_edges": {"out": [], "in": []}}


def _engine(concepts, **cfg):
    config = BrainConfig()
    for k, v in cfg.items():
        setattr(config.consolidation, k, v)
    store = StubStore(concepts)
    return ConsolidationEngine(store, config), store


# Two nearly-identical unit vectors → cosine ≈ 1; an orthogonal one → 0.
V1 = [1.0, 0.0, 0.0]
V1B = [0.999, 0.04, 0.0]
V2 = [0.0, 1.0, 0.0]


@pytest.mark.asyncio
async def test_lexical_identity_is_auto():
    # auto-merge ONLY for lexical identity (here: normalized-name match) — not cosine.
    concepts = [
        _concept("c1", "kTC Noise", V1, description="thermal reset noise on a capacitor"),
        _concept("c2", "ktc noise", V1B, description="reset noise frozen at the node"),
        _concept("c3", "Quantum Efficiency", V2, description="photon conversion ratio"),
    ]
    engine, _ = _engine(concepts)
    report = await engine.run(dry_run=True)
    assert len(report.auto) == 1
    assert {report.auto[0].a_id, report.auto[0].b_id} == {"c1", "c2"}


@pytest.mark.asyncio
async def test_high_cosine_nonlexical_routes_to_verify_not_auto():
    # The core safety change: a high-cosine, non-lexical, non-conflict pair is a CANDIDATE for the
    # LLM verifier, never an auto-merge (embedding precision tops ~0.92 — a gate, not a decider).
    concepts = [
        _concept("c1", "Body Effect", V1, description="Vth shift from source-body bias"),
        _concept("c2", "Backgate Bias Effect", V1B, description="Vth modulation by substrate"),
    ]
    engine, _ = _engine(concepts)  # no verifier wired → verify band stays
    report = await engine.run(dry_run=True)
    assert len(report.auto) == 0
    assert len(report.verify) == 1


class _StubVerifier:
    def __init__(self, verdict):
        from openclaw_brain.knowledge.reasoning.verifier import VerifyResult
        self._verdict = verdict
        self._VR = VerifyResult
        self.calls = 0

    async def verify(self, name_a, desc_a, candidate, neighbor_names=None):
        self.calls += 1
        return self._VR(verdict=self._verdict, candidate_id=candidate.get("concept_id", ""))


@pytest.mark.asyncio
async def test_verifier_drains_verify_band_into_auto():
    concepts = [
        _concept("c1", "Body Effect", V1, description="Vth shift from source-body bias"),
        _concept("c2", "Backgate Bias Effect", V1B, description="Vth modulation by substrate"),
    ]
    engine, _ = _engine(concepts)
    engine._verifier = _StubVerifier("SAME")
    report = await engine.run(dry_run=True)
    assert engine._verifier.calls == 1
    assert len(report.verify) == 0          # drained
    assert len(report.auto) == 1            # verifier SAME → auto


@pytest.mark.asyncio
async def test_verifier_different_drops_candidate():
    concepts = [
        _concept("c1", "Body Effect", V1, description="Vth shift"),
        _concept("c2", "Backgate Bias Effect", V1B, description="Vth modulation"),
    ]
    engine, _ = _engine(concepts)
    engine._verifier = _StubVerifier("DIFFERENT")
    report = await engine.run(dry_run=True)
    assert len(report.auto) == 0
    assert len(report.verify) == 0
    assert len(report.review) == 0          # DIFFERENT → dropped entirely


@pytest.mark.asyncio
async def test_guard_blocks_digit_mismatch_into_review():
    concepts = [
        _concept("c1", "3T Pixel", V1, description="three transistor pixel"),
        _concept("c2", "4T Pixel", V1B, description="four transistor pixel"),
    ]
    engine, _ = _engine(concepts)
    report = await engine.run(dry_run=True)
    assert len(report.auto) == 0
    assert len(report.review) == 1  # strong signal + guard veto → human review


@pytest.mark.asyncio
async def test_alias_hit_is_auto_without_embedding():
    concepts = [
        _concept("c1", "Full Well Capacity", None, aliases=["FWC"]),
        _concept("c2", "FWC", None, description="saturation charge"),
    ]
    engine, _ = _engine(concepts)
    report = await engine.run(dry_run=True)
    assert len(report.auto) == 1


@pytest.mark.asyncio
async def test_dry_run_never_merges():
    concepts = [
        _concept("c1", "kTC Noise", V1, description="reset noise"),
        _concept("c2", "kTC Noise Source", V1B, description="reset noise source"),
    ]
    engine, store = _engine(concepts)
    await engine.run(dry_run=True, auto_merge=True)
    assert store.merges == []


@pytest.mark.asyncio
async def test_apply_merges_into_highest_degree_primary():
    concepts = [
        _concept("c_low", "kTC Noise", V1, degree=1, description="reset noise"),
        _concept("c_high", "ktc noise", V1B, degree=9, description="reset noise frozen"),
    ]
    engine, store = _engine(concepts)
    report = await engine.run(dry_run=False, auto_merge=True)
    assert store.merges == [("c_high", "c_low")]
    assert report.merged == [{"primary": "c_high", "duplicate": "c_low"}]


@pytest.mark.asyncio
async def test_auto_merge_off_queues_auto_band(tmp_path):
    concepts = [
        _concept("c1", "kTC Noise", V1, description="reset noise"),
        _concept("c2", "ktc noise", V1B, description="reset noise frozen"),
    ]
    engine, store = _engine(concepts)
    qp = tmp_path / "queue.jsonl"
    report = await engine.run(dry_run=False, auto_merge=False, queue_path=qp)
    assert store.merges == []  # auto-merge OFF → nothing merged
    lines = [json.loads(l) for l in qp.read_text().splitlines()]
    assert any(e["band"] == "auto" for e in lines)  # auto band queued for review


def test_cluster_cap_brakes_cascade():
    config = BrainConfig()
    config.consolidation.cluster_cap = 2
    engine = ConsolidationEngine(StubStore([]), config)
    cands = [
        MergeCandidate("a", "b", "A", "B", 0.95, 0.9, "auto"),
        MergeCandidate("b", "c", "B", "C", 0.95, 0.9, "auto"),
        MergeCandidate("c", "d", "C", "D", 0.95, 0.9, "auto"),
    ]
    by_id = {x: _concept(x, x.upper(), embedding=V1) for x in ("a", "b", "c", "d")}
    clusters = engine._cluster(cands, by_id)
    assert all(len(c) <= 2 for c in clusters)


def test_cohesion_guard_splits_transitive_chain():
    """A~B (cos≈1) and B~C (auto via verifier) but A≁C → C is split off the cluster."""
    config = BrainConfig()  # cluster_cap=4 (default) allows the size-3 union; cohesion=0.86
    engine = ConsolidationEngine(StubStore([]), config)
    cands = [
        MergeCandidate("a", "b", "A", "B", 0.99, 0.9, "auto"),
        MergeCandidate("b", "c", "B", "C", 0.86, 0.1, "auto"),  # verifier-SAME, low name-sim
    ]
    by_id = {
        "a": _concept("a", "A", embedding=V1),
        "b": _concept("b", "B", embedding=V1B),   # cos(a,b) ≈ 0.999
        "c": _concept("c", "C", embedding=V2),     # orthogonal to a → below cohesion
    }
    clusters = engine._cluster(cands, by_id)
    assert clusters == [["a", "b"]]  # c split off (singleton dropped), a+b kept


def test_cohesion_guard_keeps_cohesive_triple():
    """All three mutually ≥ cohesion_threshold → the size-3 cluster survives intact."""
    config = BrainConfig()
    engine = ConsolidationEngine(StubStore([]), config)
    v_close = [0.98, 0.0, 0.199]  # cos to V1 ≈ 0.98, to V1B ≈ 0.979 — all ≥ 0.86
    cands = [
        MergeCandidate("a", "b", "A", "B", 0.99, 0.9, "auto"),
        MergeCandidate("b", "c", "B", "C", 0.98, 0.9, "auto"),
    ]
    by_id = {
        "a": _concept("a", "A", embedding=V1),
        "b": _concept("b", "B", embedding=V1B),
        "c": _concept("c", "C", embedding=v_close),
    }
    clusters = engine._cluster(cands, by_id)
    assert len(clusters) == 1 and sorted(clusters[0]) == ["a", "b", "c"]
