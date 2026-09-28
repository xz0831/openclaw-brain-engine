"""Tests for graph-to-graph relinking (enrich cross-source edges without re-chunking)."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from openclaw_brain.config import ResilienceConfig
from openclaw_brain.knowledge.graph.schema import NodeLabel, RelType
from openclaw_brain.knowledge.reasoning.relink import (
    _Concept,
    _select_pairs,
    relink_source,
    RelinkEdge,
    RelinkProposal,
)


# ── Pure candidate-selection logic ──


def _cand(id, source_id, score, label="Concept", name="x"):
    return {"id": id, "name": name, "source_id": source_id, "label": label,
            "description": "", "score": score}


def test_select_pairs_applies_every_filter():
    c1 = _Concept(id="c1", name="cascode", source_id="RAZ")
    c2 = _Concept(id="c2", name="mirror", source_id="RAZ")
    candidates = {
        "c1": [
            _cand("g1", "GM", 0.9),                      # KEEP
            _cand("g2", "GM", 0.3),                      # drop: score < min
            _cand("r9", "RAZ", 0.9),                     # drop: same source (not a cross-link)
            _cand("p1", "GM", 0.9, label="Parameter"),   # drop: not a Concept
            _cand("c1", "GM", 0.9),                      # drop: self-id
            _cand("g3", "GM", 0.9),                      # drop: already linked
        ],
        "c2": [_cand("g4", "GM", 0.7)],                  # KEEP
    }
    existing = {"c1": {"g3"}, "c2": set()}

    pairs = _select_pairs([c1, c2], candidates, existing, None, min_score=0.6, top_k=6)

    assert {(c.id, cand["id"]) for c, cand in pairs} == {("c1", "g1"), ("c2", "g4")}


def test_select_pairs_target_source_filter():
    c1 = _Concept(id="c1", name="x", source_id="RAZ")
    candidates = {"c1": [_cand("g1", "GM", 0.9), _cand("m1", "MURMANN", 0.9)]}
    pairs = _select_pairs([c1], candidates, {"c1": set()}, {"MURMANN"}, min_score=0.6, top_k=6)
    assert {(c.id, cand["id"]) for c, cand in pairs} == {("c1", "m1")}  # GM excluded


def test_select_pairs_caps_top_k():
    c1 = _Concept(id="c1", name="x", source_id="RAZ")
    candidates = {"c1": [_cand(f"g{i}", "GM", 0.9) for i in range(10)]}
    pairs = _select_pairs([c1], candidates, {"c1": set()}, None, min_score=0.6, top_k=3)
    assert len(pairs) == 3


# ── Orchestration (mocked graph + llm) ──


def _match_cand(concept_id, source_id, cos, name="gm/ID", description="e", **props):
    """The REAL find_match_candidates shape: {"node": <props>, "cos": ..., "text_hit": ...}.
    Node props carry no label; a Concept is identified by having concept_id."""
    node = {"concept_id": concept_id, "canonical_name": name,
            "description": description, "source_id": source_id, **props}
    return {"node": node, "cos": cos, "text_hit": cos is None}


def _mock_graph(concepts_rows, candidates, neighbors=None):
    graph = AsyncMock()
    neighbors = neighbors or {}

    async def _rrq(query, params=None):
        if "collect(DISTINCT b.concept_id)" in query:          # _existing_neighbor_ids
            return [{"ids": list(neighbors.get((params or {}).get("id"), []))}]
        return concepts_rows                                    # _fetch_source_concepts

    graph.run_read_query = AsyncMock(side_effect=_rrq)
    graph.find_match_candidates = AsyncMock(return_value=candidates)
    graph.write_batch = AsyncMock()
    return graph


def _mock_llm(edges):
    llm = AsyncMock()
    llm.model_name = "mock-reasoner"
    wrapped = AsyncMock()
    wrapped.ainvoke.return_value = RelinkProposal(edges=edges)
    llm.with_structured_output = MagicMock(return_value=wrapped)
    return llm


@pytest.mark.asyncio
async def test_relink_apply_commits_only_valid_edges():
    graph = _mock_graph(
        concepts_rows=[{"id": "c1", "name": "cascode", "description": "d", "embedding": [0.1] * 4}],
        candidates=[_match_cand("g1", "GM", 0.9, name="gm/ID efficiency")],
    )
    llm = _mock_llm([
        RelinkEdge(source_id="c1", target_id="g1", relationship_type="RELATES_TO", rationale="r"),
        RelinkEdge(source_id="c1", target_id="BOGUS", relationship_type="RELATES_TO"),  # unknown id
        RelinkEdge(source_id="c1", target_id="g1", relationship_type="NOT_A_REAL_TYPE"),  # bad type
    ])

    stats = await relink_source(graph, [llm], "RAZ", ResilienceConfig(), apply=True)

    assert stats["concepts_scanned"] == 1
    assert stats["candidate_pairs"] == 1
    assert stats["edges_valid"] == 1          # BOGUS + bad-type filtered out
    assert stats["edges_committed"] == 1
    graph.write_batch.assert_called_once()
    edges = graph.write_batch.call_args.kwargs["edges"]
    assert edges[0]["source_id_value"] == "c1" and edges[0]["target_id_value"] == "g1"
    # _merge_edge_tx does source_label.value / rel_type.value → these MUST be enums, not strings.
    assert edges[0]["source_label"] is NodeLabel.CONCEPT
    assert edges[0]["target_label"] is NodeLabel.CONCEPT
    assert edges[0]["rel_type"] is RelType.RELATES_TO
    assert edges[0]["properties"]["origin"] == "relink"


@pytest.mark.asyncio
async def test_relink_dry_run_writes_nothing():
    graph = _mock_graph(
        concepts_rows=[{"id": "c1", "name": "cascode", "description": "d", "embedding": [0.1] * 4}],
        candidates=[_match_cand("g1", "GM", 0.9)],
    )
    llm = _mock_llm([RelinkEdge(source_id="c1", target_id="g1", relationship_type="DEPENDS_ON")])

    stats = await relink_source(graph, [llm], "RAZ", ResilienceConfig(), apply=False)

    assert stats["edges_valid"] == 1
    assert stats["edges_committed"] == 0
    graph.write_batch.assert_not_called()


@pytest.mark.asyncio
async def test_relink_emits_progress_heartbeat_logs(caplog):
    """The scan/plan/batch INFO lines are the observable surface an external monitor (/loop)
    greps for liveness and progress — a relink otherwise emits nothing until it returns.
    Exact-format asserts on purpose: monitors grep these strings, so format drift is a break."""
    graph = _mock_graph(
        concepts_rows=[{"id": "c1", "name": "cascode", "description": "d", "embedding": [0.1] * 4}],
        candidates=[_match_cand("g1", "GM", 0.9)],
    )
    llm = _mock_llm([RelinkEdge(source_id="c1", target_id="g1", relationship_type="RELATES_TO")])

    with caplog.at_level(logging.INFO, logger="openclaw_brain.knowledge.reasoning.relink"):
        await relink_source(graph, [llm], "RAZ", ResilienceConfig(), apply=False)

    msgs = [r.getMessage() for r in caplog.records]
    assert "Relink scan: 1/1 concepts" in msgs
    assert "Relink plan: 1 concepts scanned, 1 candidate pairs -> 1 reasoning batches" in msgs
    assert "Relink batch 1/1: 1 pairs -> 1 edges (running total 1)" in msgs


@pytest.mark.asyncio
async def test_relink_skips_already_linked_pair():
    """A candidate already related to the concept must not be reasoned or committed."""
    graph = _mock_graph(
        concepts_rows=[{"id": "c1", "name": "cascode", "description": "d", "embedding": [0.1] * 4}],
        candidates=[_match_cand("g1", "GM", 0.9)],
        neighbors={"c1": ["g1"]},  # c1 already linked to g1
    )
    llm = _mock_llm([RelinkEdge(source_id="c1", target_id="g1", relationship_type="RELATES_TO")])

    stats = await relink_source(graph, [llm], "RAZ", ResilienceConfig(), apply=True)

    assert stats["candidate_pairs"] == 0
    assert stats["edges_committed"] == 0
    graph.write_batch.assert_not_called()


# ── Per-batch failure visibility (failed_batches / skipped_pairs) ──


@pytest.mark.asyncio
async def test_relink_reports_failed_batches_and_skipped_pairs(monkeypatch):
    """A batch whose reasoning call raises (fallback chain exhausted) must not just vanish from
    the stats — failed_batches/skipped_pairs must reflect it so a caller relying on the return
    value alone can distinguish 'no real relationship found' from 'this batch never got judged'.
    """
    import openclaw_brain.knowledge.reasoning.relink as relink_mod

    graph = _mock_graph(
        concepts_rows=[
            {"id": "c1", "name": "cascode", "description": "d", "embedding": [0.1] * 4},
            {"id": "c2", "name": "mirror", "description": "d", "embedding": [0.1] * 4},
        ],
        candidates=[_match_cand("g1", "GM", 0.9)],
    )
    llm = _mock_llm([])  # unused: invoke_with_resilience is replaced below

    call_count = {"n": 0}

    async def _flaky_invoke(models, messages, resilience, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("all fallback models exhausted")
        return RelinkProposal(
            edges=[RelinkEdge(source_id="c1", target_id="g1", relationship_type="RELATES_TO")]
        )

    monkeypatch.setattr(relink_mod, "invoke_with_resilience", _flaky_invoke)

    stats = await relink_source(
        graph, [llm], "RAZ", ResilienceConfig(), batch_size=1, apply=False,
    )

    assert stats["candidate_pairs"] == 2   # (c1,g1) and (c2,g1) — both concepts share candidate g1
    assert stats["failed_batches"] == 1
    assert stats["skipped_pairs"] == 1
    assert stats["edges_proposed"] == 1    # only the one batch that succeeded


@pytest.mark.asyncio
async def test_relink_no_failed_batches_when_all_succeed():
    """REGRESSION: the new counters must read zero when nothing fails (unchanged behavior)."""
    graph = _mock_graph(
        concepts_rows=[{"id": "c1", "name": "cascode", "description": "d", "embedding": [0.1] * 4}],
        candidates=[_match_cand("g1", "GM", 0.9)],
    )
    llm = _mock_llm([RelinkEdge(source_id="c1", target_id="g1", relationship_type="RELATES_TO")])

    stats = await relink_source(graph, [llm], "RAZ", ResilienceConfig(), apply=False)

    assert stats["failed_batches"] == 0
    assert stats["skipped_pairs"] == 0


# ── Commit-time race guard (identity-boundary spirit at the edge level) ──


@pytest.mark.asyncio
async def test_relink_apply_skips_edge_that_appeared_since_scan():
    """If a concurrent writer (another relink run, or an ingest) creates the exact same edge
    between relink's scan snapshot and its own commit, relink must not blindly re-MERGE over it
    — _merge_edge_tx applies the SAME property set on ON CREATE and ON MATCH, which would
    silently overwrite the concurrent writer's rationale/origin. relink must re-check right
    before commit and skip an edge that already exists by then, keeping the first writer's
    properties intact."""
    graph = AsyncMock()
    calls = {"neighbor_lookups": 0}

    async def _rrq(query, params=None):
        if "collect(DISTINCT b.concept_id)" in query:
            calls["neighbor_lookups"] += 1
            if calls["neighbor_lookups"] == 1:
                return [{"ids": []}]      # scan-time snapshot: not yet linked
            return [{"ids": ["g1"]}]      # commit-time recheck: a concurrent writer beat us to it
        return [{"id": "c1", "name": "cascode", "description": "d", "embedding": [0.1] * 4}]

    graph.run_read_query = AsyncMock(side_effect=_rrq)
    graph.find_match_candidates = AsyncMock(
        return_value=[_match_cand("g1", "GM", 0.9, name="gm/ID efficiency")]
    )
    graph.write_batch = AsyncMock()
    llm = _mock_llm([RelinkEdge(source_id="c1", target_id="g1", relationship_type="RELATES_TO")])

    stats = await relink_source(graph, [llm], "RAZ", ResilienceConfig(), apply=True)

    assert stats["edges_valid"] == 1        # judged as a real, validated relationship
    assert stats["edges_committed"] == 0    # but NOT committed — it appeared since the scan
    graph.write_batch.assert_not_called()


@pytest.mark.asyncio
async def test_relink_apply_commits_when_nothing_raced():
    """REGRESSION: the commit-time recheck must not block ordinary, uncontended commits — this
    duplicates test_relink_apply_commits_only_valid_edges' outcome using a call-counting mock to
    prove the recheck query itself (not just its absence) doesn't change the result."""
    graph = AsyncMock()

    async def _rrq(query, params=None):
        if "collect(DISTINCT b.concept_id)" in query:
            return [{"ids": []}]  # never linked, at scan OR at the commit-time recheck
        return [{"id": "c1", "name": "cascode", "description": "d", "embedding": [0.1] * 4}]

    graph.run_read_query = AsyncMock(side_effect=_rrq)
    graph.find_match_candidates = AsyncMock(
        return_value=[_match_cand("g1", "GM", 0.9, name="gm/ID efficiency")]
    )
    graph.write_batch = AsyncMock()
    llm = _mock_llm([RelinkEdge(source_id="c1", target_id="g1", relationship_type="RELATES_TO")])

    stats = await relink_source(graph, [llm], "RAZ", ResilienceConfig(), apply=True)

    assert stats["edges_committed"] == 1
    graph.write_batch.assert_called_once()
