"""Tests for the Neo4j projection of executable specimens (knowledge/executable/projection.py).

Pure projection (no DB) + the projector's safe linking (resolve to existing nodes, drop
unresolved -> no phantom). The store is mocked (repo convention: Neo4j tests don't need a DB).
"""

import pytest

from openclaw_brain.knowledge.executable.models import (
    ClaimCard, AnalogPVT, MechanismClaim, QuantTest, Specimen, VerdictClass,
)
from openclaw_brain.knowledge.executable.projection import (
    GraphProjector, project_specimen,
)
from openclaw_brain.knowledge.graph.schema import NodeLabel, RelType

COND = AnalogPVT(corner="tt", temp_c=27, vdd=1.8)
OTA = "miller_ota_2stage_nmos_in"


def _specimen():
    def claim(cid, metric, verdict):
        c = ClaimCard(id=cid, topology_class=OTA,
                      mechanism=MechanismClaim(knob="Cc", metric=metric, series_ref="cc",
                                               quant=QuantTest(kind="direction", sign="-"),
                                               narrative="why"),
                      conditions=COND)
        c.verdict = verdict
        return c
    return Specimen(topology_class=OTA, netlist="* net\n.end\n",
                    claim_cards=[claim("cc_gbw_inverse", "gbw_hz", VerdictClass.VERIFIED_WITH_CAVEAT),
                                 claim("cl_pm_down", "pm_deg", VerdictClass.VERIFIED)])


def test_project_specimen_pure():
    pw = project_specimen(_specimen())
    labels = [n["label"] for n in pw.nodes]
    assert labels.count(NodeLabel.SPECIMEN) == 1
    assert labels.count(NodeLabel.CLAIM_CARD) == 2
    spec_node = next(n for n in pw.nodes if n["label"] == NodeLabel.SPECIMEN)
    assert spec_node["properties"]["topology_class"] == OTA
    cc_node = next(n for n in pw.nodes if n["label"] == NodeLabel.CLAIM_CARD)
    assert cc_node["properties"]["verdict"] in {"VERIFIED_WITH_CAVEAT", "VERIFIED"}
    # internal edges: 2 HAS_CLAIM
    assert [e["rel_type"] for e in pw.edges] == [RelType.HAS_CLAIM, RelType.HAS_CLAIM]
    # links to existing nodes: 1 REALIZES(CircuitTopology) + 2 GROUNDS(Parameter)
    rels = sorted(l.rel_type.value for l in pw.links)
    assert rels == ["GROUNDS", "GROUNDS", "REALIZES"]
    realizes = next(l for l in pw.links if l.rel_type == RelType.REALIZES)
    assert realizes.target_label == NodeLabel.CIRCUIT_TOPOLOGY
    assert "miller ota" in realizes.match_text


class _MockStore:
    def __init__(self):
        self.calls = []

    async def write_batch(self, nodes=None, updates=None, edges=None):
        self.calls.append({"nodes": nodes or [], "edges": edges or []})


@pytest.mark.asyncio
async def test_projector_resolves_and_drops_unmatched():
    store = _MockStore()

    async def resolver(label, text):
        # CircuitTopology resolves; Parameter does NOT (simulates a metric with no existing node)
        return "topo-123" if label == NodeLabel.CIRCUIT_TOPOLOGY else None

    stats = await GraphProjector(store, resolver).project(_specimen())
    assert stats == {"nodes": 3, "internal_edges": 2, "links_resolved": 1, "links_total": 3}
    written = store.calls[0]
    assert len(written["nodes"]) == 3                                  # additive nodes
    rels = [e["rel_type"] for e in written["edges"]]
    assert rels.count(RelType.HAS_CLAIM) == 2                          # internal
    assert rels.count(RelType.REALIZES) == 1                           # the one resolved link
    assert RelType.GROUNDS not in rels                                # unresolved -> dropped (no phantom)
    realizes_edge = next(e for e in written["edges"] if e["rel_type"] == RelType.REALIZES)
    assert realizes_edge["target_id_value"] == "topo-123"
    assert realizes_edge["target_id_field"] == "topology_id"
