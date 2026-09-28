"""Regression test for the paper-summary context bug (knowledge/reasoning/summarizer.py).

_gather_paper_context previously collected the source's chunk_ids but never used them to filter the
node match, so it returned the ENTIRE graph's knowledge nodes — every paper got summarized from the
same global sample, cross-contaminating all Source summaries. This asserts the context is now scoped
to one source. Neo4j-gated.
"""

import pytest
from tests.conftest import require_live_graph

from openclaw_brain.config import load_config
from openclaw_brain.knowledge.graph.schema import NodeLabel
from openclaw_brain.knowledge.graph.store import GraphStore
from openclaw_brain.knowledge.reasoning.summarizer import _gather_paper_context


@pytest.fixture
async def graph():
    require_live_graph()
    g = GraphStore(load_config().neo4j)
    try:
        await g.connect()
    except Exception:
        pytest.skip("Neo4j not available")
    # two sources, each with a uniquely-named concept (sentinel names — no ingested source collides)
    for sid, cid, name in [
        ("src_testsumm_a", "test_summ_alpha", "Alpha Sentinel Topic Qx"),
        ("src_testsumm_b", "test_summ_beta", "Beta Sentinel Topic Qx"),
    ]:
        await g.merge_node(
            NodeLabel.CONCEPT, "concept_id", cid,
            {"concept_id": cid, "canonical_name": name, "source_id": sid,
             "description": f"description of {name}", "domain": "analog_circuits"})
    yield g
    async with await g._session() as s:
        await s.run("MATCH (n) WHERE n.concept_id STARTS WITH 'test_summ_' DETACH DELETE n")
    await g.close()


@pytest.mark.asyncio
async def test_gather_paper_context_is_source_scoped(graph: GraphStore):
    ctx = await _gather_paper_context("src_testsumm_a", graph)
    assert "Alpha Sentinel Topic Qx" in ctx           # this source's node IS in its own context
    assert "Beta Sentinel Topic Qx" not in ctx        # the OTHER source's node must NOT leak in
