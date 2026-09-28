"""Neo4j projection (SPEC §13.1) — the DERIVED graph view of git-corpus specimens.

The corpus is the SSOT; this projects a specimen into the graph as Specimen + ClaimCard
nodes and LINKS them to the EXISTING text-knowledge graph (a Specimen REALIZES an existing
CircuitTopology; a ClaimCard GROUNDS an existing Parameter/Concept). This is the SEAM that
joins the executable layer to the 2268 CircuitTopology / 4336 Concept nodes already in the
graph — we ADD and LINK, never rebuild.

Two safety invariants (this writes to the live production graph):
  1. ADDITIVE — new nodes are MERGEd by content-hash id; existing nodes are never modified.
  2. NO PHANTOMS — a link to an existing node is created only if a resolver finds it; the
     edge MATCHes both endpoints (store._merge_edge_tx), so an unresolved/wrong target
     simply forms no edge. Linking is best-effort, never forced.
The projection is deterministic and rebuildable from the corpus.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from ..graph.schema import NodeLabel, RelType
from .corpus import compute_spec_id
from .models import Specimen

# id field per label (existing graph + the new executable labels)
_ID_FIELD = {
    NodeLabel.SPECIMEN: "spec_id",
    NodeLabel.CLAIM_CARD: "claim_id",
    NodeLabel.CIRCUIT_TOPOLOGY: "topology_id",
    NodeLabel.PARAMETER: "parameter_id",
    NodeLabel.CONCEPT: "concept_id",
}

# Resolver: (target_label, match_text) -> existing node's id_value, or None if no match.
Resolver = Callable[[NodeLabel, str], Awaitable[str | None]]


@dataclass
class LinkRequest:
    """A best-effort link from a new node to an EXISTING graph node (resolved by text)."""
    source_label: NodeLabel
    source_id_value: str
    rel_type: RelType
    target_label: NodeLabel
    match_text: str


@dataclass
class ProjectionWrite:
    nodes: list[dict] = field(default_factory=list)   # write_batch node dicts (MERGE, additive)
    edges: list[dict] = field(default_factory=list)   # internal edges (both endpoints in `nodes`)
    links: list[LinkRequest] = field(default_factory=list)  # to existing nodes (resolved separately)


def _readable(topology_class: str) -> str:
    """`miller_ota_2stage_nmos_in` -> `miller ota 2stage nmos in` (text for the resolver)."""
    return topology_class.replace("_", " ")


def _metric_text(metric: str) -> str:
    """`gbw_hz` -> `gbw`; `pm_deg` -> `pm`; `av0_db` -> `av0`."""
    return re.sub(r"_(hz|deg|db|v|a|ohm|s|f)$", "", metric).replace("_", " ")


def project_specimen(spec: Specimen) -> ProjectionWrite:
    """Pure: turn a specimen into the additive node/edge/link write plan (no DB)."""
    sid = spec.spec_id or compute_spec_id(spec)
    pw = ProjectionWrite()
    pw.nodes.append({
        "label": NodeLabel.SPECIMEN, "id_field": "spec_id", "id_value": sid,
        "properties": {"topology_class": spec.topology_class, "pdk": spec.pdk, "tool": spec.tool},
    })
    # Specimen REALIZES the existing (text) CircuitTopology for its class
    pw.links.append(LinkRequest(NodeLabel.SPECIMEN, sid, RelType.REALIZES,
                                NodeLabel.CIRCUIT_TOPOLOGY, _readable(spec.topology_class)))
    for c in spec.claim_cards:
        cid = f"{sid}:{c.id}"
        pw.nodes.append({
            "label": NodeLabel.CLAIM_CARD, "id_field": "claim_id", "id_value": cid,
            "properties": {
                "claim": c.id, "knob": c.mechanism.knob, "metric": c.mechanism.metric,
                "verdict": c.verdict.value if c.verdict else None,
                # quant_kind lets the citation audit tell a SHAPE claim (direction/invariance) from a
                # scalar claim (value/elasticity), so a magnitude presented as certified while the card
                # proved only a direction/invariance can be flagged (the teaching-eval leak).
                "quant_kind": c.mechanism.quant.kind,
                # scope-honesty props (ADR 4-5): engine + epistemic basis + machine-surfaced scope +
                # the untested dominant axis, so the teaching surfaces can render the verdict inline with
                # its scope and refuse to over-generalize it.
                "engine": c.engine,
                "basis": c.basis,
                "scope": json.dumps(c.scope) if c.scope else None,
                "dominant_risk_untested": c.dominant_risk_untested,
                "narrative": c.mechanism.narrative,
                # PVT props are ANALOG-only: getattr keeps this engine-safe — a digital ClaimCard's
                # DigitalUnits conditions have no corner/temp/vdd, so they project as null (not crash).
                "corner": getattr(c.conditions, "corner", None),
                "temp_c": getattr(c.conditions, "temp_c", None),
                "vdd": getattr(c.conditions, "vdd", None),
            },
        })
        pw.edges.append({
            "source_label": NodeLabel.SPECIMEN, "source_id_field": "spec_id", "source_id_value": sid,
            "target_label": NodeLabel.CLAIM_CARD, "target_id_field": "claim_id", "target_id_value": cid,
            "rel_type": RelType.HAS_CLAIM, "properties": {},
        })
        # ClaimCard GROUNDS the existing Parameter for the metric it measures
        pw.links.append(LinkRequest(NodeLabel.CLAIM_CARD, cid, RelType.GROUNDS,
                                    NodeLabel.PARAMETER, _metric_text(c.mechanism.metric)))
    return pw


async def retract_projection(store, spec_id: str) -> dict:
    """Undo a projection: DETACH DELETE the Specimen + its ClaimCards (and thus their REALIZES /
    HAS_CLAIM / GROUNDS edges) for one spec_id. Additive projection has no other footprint — the
    existing graph nodes the links pointed at are untouched. Reversibility for a bad --apply."""
    async with await store._session() as session:
        result = await session.run(
            """
            MATCH (s:Specimen {spec_id: $sid})
            OPTIONAL MATCH (s)-[:HAS_CLAIM]->(c:ClaimCard)
            WITH collect(DISTINCT s) AS specs, [x IN collect(DISTINCT c) WHERE x IS NOT NULL] AS claims
            FOREACH (s IN specs | DETACH DELETE s)
            FOREACH (c IN claims | DETACH DELETE c)
            RETURN size(specs) AS specimens, size(claims) AS claim_cards
            """,
            {"sid": spec_id},
        )
        rec = await result.single()
        return {"specimens": rec["specimens"] if rec else 0,
                "claim_cards": rec["claim_cards"] if rec else 0}


PROJECTION_TAG = "executable_projection"   # stamps cross-links so re-apply can supersede them


class GraphProjector:
    """Projects specimens into the live graph: writes additive nodes/internal edges, and
    resolves each LinkRequest against the existing graph (dropping unresolved ones)."""

    def __init__(self, store, resolver: Resolver):
        self._store = store
        self._resolver = resolver

    async def _supersede_links(self, spec_id: str) -> None:
        """I4: delete this specimen's PRIOR projection cross-links (REALIZES from the Specimen,
        GROUNDS from its ClaimCards) before re-writing. Cross-links re-resolve against the MUTABLE
        graph each apply, so without superseding a changed resolution would accrete a second,
        divergent edge instead of replacing the old one. Only edges this projector stamped are
        touched; internal HAS_CLAIM edges are content-stable and left alone."""
        session_cm = getattr(self._store, "_session", None)
        if session_cm is None:
            return    # a store without raw-session access (e.g. a unit mock) can't accrete anyway
        async with await self._store._session() as session:
            await session.run(
                """
                MATCH (s:Specimen {spec_id: $sid})
                OPTIONAL MATCH (s)-[r:REALIZES]->() WHERE r.created_by = $tag
                WITH s, collect(r) AS rs
                OPTIONAL MATCH (s)-[:HAS_CLAIM]->(:ClaimCard)-[g:GROUNDS]->() WHERE g.created_by = $tag
                WITH rs, collect(g) AS gs
                FOREACH (r IN rs | DELETE r)
                FOREACH (g IN gs | DELETE g)
                """,
                {"sid": spec_id, "tag": PROJECTION_TAG},
            )

    async def project(self, spec: Specimen) -> dict:
        pw = project_specimen(spec)
        spec_id = pw.nodes[0]["id_value"]
        await self._supersede_links(spec_id)
        edges = list(pw.edges)
        resolved = 0
        for lr in pw.links:
            target_id = await self._resolver(lr.target_label, lr.match_text)
            if target_id is None:
                continue  # no match -> no phantom, no forced link
            edges.append({
                "source_label": lr.source_label, "source_id_field": _ID_FIELD[lr.source_label],
                "source_id_value": lr.source_id_value,
                "target_label": lr.target_label, "target_id_field": _ID_FIELD[lr.target_label],
                "target_id_value": target_id,
                "rel_type": lr.rel_type, "properties": {"created_by": PROJECTION_TAG},
            })
            resolved += 1
        await self._store.write_batch(nodes=pw.nodes, edges=edges)
        return {"nodes": len(pw.nodes), "internal_edges": len(pw.edges),
                "links_resolved": resolved, "links_total": len(pw.links)}
