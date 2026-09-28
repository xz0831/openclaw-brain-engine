"""Graph-to-graph relinking — enrich an existing source's concepts with cross-source
relationships against the *current* graph, WITHOUT re-chunking or re-extracting.

Why this exists: a `reprocess` re-runs the full parse→chunk→extract→match→reason pipeline. But
chunking is not stable across runs (figure descriptions are non-deterministic; disabling figures
changes the text), so a reprocess whose only goal is new cross-links instead *duplicates* every
chunk and its concepts (observed on the gm/ID Razavi reprocess: 1508→2496 SourceChunks). Relink
sidesteps chunking entirely: it works on the concept nodes already in the graph, finds
newly-reachable cross-source neighbours by embedding similarity, asks the reasoner which pairs
carry a real relationship, and commits ONLY those edges. Purely additive — it never creates a
node, so it can be re-run any time the graph gains new knowledge with zero duplication.

Scope (MVP): Concept↔Concept edges. Parameter/Topology endpoints are a later extension.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from openclaw_brain.config import ResilienceConfig
from openclaw_brain.knowledge.graph.schema import NodeLabel, RelType
from openclaw_brain.knowledge.graph.store import GraphStore
from openclaw_brain.llm.resilience import invoke_with_resilience, resolve_model_name

logger = logging.getLogger(__name__)

_ALLOWED_RELS = [r.value for r in RelType]


class RelinkEdge(BaseModel):
    """One reasoned cross-source relationship between two existing concepts."""

    source_id: str = Field(description="concept_id of the FROM concept (the relationship's tail)")
    target_id: str = Field(description="concept_id of the TO concept (the relationship's head)")
    relationship_type: str = Field(description="one of the allowed relationship types")
    rationale: str = ""


class RelinkProposal(BaseModel):
    edges: list[RelinkEdge] = Field(default_factory=list)


class _Concept(BaseModel):
    """A concept row pulled from the graph (id, name, description, source, embedding)."""

    id: str
    name: str
    description: str = ""
    source_id: str = ""
    embedding: list[float] | None = None


def _select_pairs(
    source_concepts: list[_Concept],
    candidates_by_id: dict[str, list[dict]],
    existing_neighbors: dict[str, set[str]],
    target_sources: set[str] | None,
    min_score: float,
    top_k: int,
) -> list[tuple[_Concept, dict]]:
    """Pure selection: for each source concept, keep its candidate neighbours that are (a) a
    Concept from a DIFFERENT source (and in ``target_sources`` if given), (b) at or above
    ``min_score``, (c) not already linked, (d) not itself — capped at ``top_k`` per concept.

    Kept pure (no I/O) so the filtering rules are unit-testable in isolation.
    """
    pairs: list[tuple[_Concept, dict]] = []
    for c in source_concepts:
        seen = existing_neighbors.get(c.id, set())
        kept = 0
        for cand in candidates_by_id.get(c.id, []):
            cid = cand.get("id")
            csrc = cand.get("source_id")
            if not cid or cid == c.id:
                continue
            if cand.get("label") not in (None, "Concept"):
                continue
            if csrc == c.source_id:  # same source — not a cross-link
                continue
            if target_sources is not None and csrc not in target_sources:
                continue
            if cand.get("score", 0.0) < min_score:
                continue
            if cid in seen:  # already related
                continue
            pairs.append((c, cand))
            kept += 1
            if kept >= top_k:
                break
    return pairs


_SYSTEM_PROMPT = (
    "You are an expert analog IC design engineer curating a knowledge graph. You are given pairs "
    "of already-defined concepts: a SOURCE concept and a semantically nearby CANDIDATE concept "
    "from another document. For each pair, decide whether a genuine, directed technical "
    "relationship holds between them. Only emit an edge when you are confident the relationship is "
    "real and technically correct — it is far better to omit a doubtful pair than to assert a wrong "
    "link. Never invent concepts; use only the given concept_ids. Choose the relationship_type "
    "from the allowed list and orient it correctly (source_id = tail, target_id = head)."
)


def _build_prompt(pairs: list[tuple[_Concept, dict]]) -> str:
    lines = [
        "Allowed relationship_type values: " + ", ".join(_ALLOWED_RELS),
        "",
        "Pairs to judge (emit an edge only for real relationships):",
    ]
    for c, cand in pairs:
        lines.append(
            f"- SOURCE [{c.id}] \"{c.name}\": {(c.description or '')[:200]}\n"
            f"  CANDIDATE [{cand.get('id')}] \"{cand.get('name')}\": {(cand.get('description') or '')[:200]}"
        )
    lines.append(
        "\nReturn a JSON object {\"edges\": [{source_id, target_id, relationship_type, rationale}]}. "
        "source_id/target_id MUST be concept_ids from the pairs above. Omit pairs with no real link."
    )
    return "\n".join(lines)


async def _fetch_source_concepts(graph: GraphStore, source_id: str) -> list[_Concept]:
    rows = await graph.run_read_query(
        "MATCH (n:Concept) WHERE n.source_id = $s AND NOT coalesce(n.retracted, false) "
        "AND n.canonical_name IS NOT NULL "
        "RETURN n.concept_id AS id, n.canonical_name AS name, "
        "coalesce(n.description, '') AS description, n.embedding AS embedding",
        {"s": source_id},
    )
    return [
        _Concept(id=r["id"], name=r["name"], description=r["description"],
                 source_id=source_id, embedding=r.get("embedding"))
        for r in rows if r.get("id")
    ]


async def _existing_neighbor_ids(graph: GraphStore, concept_id: str) -> set[str]:
    rows = await graph.run_read_query(
        "MATCH (a:Concept {concept_id: $id})-[]-(b) WHERE b.concept_id IS NOT NULL "
        "RETURN collect(DISTINCT b.concept_id) AS ids",
        {"id": concept_id},
    )
    return set(rows[0]["ids"]) if rows and rows[0].get("ids") else set()


async def relink_source(
    graph: GraphStore,
    llm_chain: list[BaseChatModel],
    source_id: str,
    resilience: ResilienceConfig,
    target_source_ids: list[str] | None = None,
    top_k: int = 6,
    min_score: float = 0.6,
    batch_size: int = 12,
    apply: bool = False,
) -> dict:
    """Relink one source's concepts against the current graph.

    Args:
        graph: connected GraphStore.
        llm_chain: reasoning model chain (primary first, then fallbacks) — raw models; the
            resilient wrapper receives their names so the circuit breaker attributes correctly.
        source_id: the source whose concepts get enriched.
        resilience: ResilienceConfig for invoke_with_resilience.
        target_source_ids: only link against these sources (default: any other source).
        top_k: max candidate neighbours reasoned per concept.
        min_score: minimum embedding-similarity score for a candidate.
        batch_size: concept-pairs per reasoning call.
        apply: commit the reasoned edges (default: dry-run — reason but do not write).

    Returns:
        stats dict: concepts_scanned, candidate_pairs, edges_proposed, edges_committed.
    """
    targets = set(target_source_ids) if target_source_ids else None
    concepts = await _fetch_source_concepts(graph, source_id)

    candidates_by_id: dict[str, list[dict]] = {}
    existing: dict[str, set[str]] = {}
    for i, c in enumerate(concepts, 1):
        # Progress heartbeat for external monitors (/loop): the scan phase runs 2 queries per
        # concept and would otherwise emit nothing for thousands of concepts.
        if i % 200 == 0 or i == len(concepts):
            logger.info("Relink scan: %d/%d concepts", i, len(concepts))
        existing[c.id] = await _existing_neighbor_ids(graph, c.id)
        if not c.embedding:
            continue
        # find_match_candidates returns {"node": <props>, "cos": <cosine|None>, "text_hit": bool}.
        # Neo4j node props carry NO label, so infer Concept from the presence of concept_id (this
        # also enforces the Concept↔Concept MVP scope — Parameter/Topology/Equation nodes lack it).
        # Text-only hits have cos=None → score 0.0 → dropped by min_score; relink links by embedding
        # similarity, which is the semantic signal we want across differently-named concepts.
        cands = []
        for item in await graph.find_match_candidates(c.name, embedding=c.embedding, limit=top_k * 3):
            node = item.get("node") or {}
            if "concept_id" not in node:
                continue
            cos = item.get("cos")
            cands.append({
                "id": node.get("concept_id"),
                "name": node.get("canonical_name"),
                "description": node.get("description", ""),
                "source_id": node.get("source_id"),
                "label": "Concept",
                "score": cos if cos is not None else 0.0,
            })
        candidates_by_id[c.id] = cands

    pairs = _select_pairs(concepts, candidates_by_id, existing, targets, min_score, top_k)
    total_batches = (len(pairs) + batch_size - 1) // batch_size
    # The denominators a monitor needs, emitted once up front.
    logger.info("Relink plan: %d concepts scanned, %d candidate pairs -> %d reasoning batches",
                len(concepts), len(pairs), total_batches)

    model_names = [resolve_model_name(m) for m in llm_chain]
    proposed: list[RelinkEdge] = []
    failed_batches = 0
    skipped_pairs = 0
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        wrapped = [m.with_structured_output(RelinkProposal) for m in llm_chain]
        try:
            result = await invoke_with_resilience(
                wrapped,
                [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=_build_prompt(batch))],
                resilience,
                model_names=model_names,
            )
        except Exception as e:
            # This batch never got judged (fallback chain exhausted) — distinct from "the LLM
            # judged these pairs and found no real relationship". Without failed_batches/
            # skipped_pairs a caller relying on the returned stats alone can't tell a partial
            # infra failure from genuinely low candidate quality.
            logger.warning("Relink reasoning batch failed (skipping): %s", e)
            failed_batches += 1
            skipped_pairs += len(batch)
            continue
        batch_edges = result.edges if result and result.edges else []
        proposed.extend(batch_edges)
        logger.info("Relink batch %d/%d: %d pairs -> %d edges (running total %d)",
                    start // batch_size + 1, total_batches, len(batch), len(batch_edges),
                    len(proposed))

    valid_ids = {c.id for c in concepts} | {
        cand.get("id") for cands in candidates_by_id.values() for cand in cands
    }
    edges = [
        e for e in proposed
        if e.source_id in valid_ids and e.target_id in valid_ids
        and e.source_id != e.target_id
        and e.relationship_type in _ALLOWED_RELS
    ]

    committed = 0
    if apply and edges:
        # Commit-time race guard: the "already linked" filter feeding _select_pairs above is a
        # snapshot taken once, before any LLM calls; the reasoning batches between that snapshot
        # and this commit can take a while, so a concurrent writer (another relink run, or an
        # ingest) may create one of these exact edges in that window. write_batch's underlying
        # MERGE applies the SAME property set on ON CREATE and ON MATCH (graph/store.py::
        # _merge_edge_tx, shared with the main ingest commit path — not changed here), so
        # committing blind would silently overwrite that writer's rationale/origin. Re-check
        # right before commit and drop anything that already exists by now — mirrors the
        # identity-boundary spirit (first writer keeps its rationale/origin) for relink's own
        # edge properties, without touching the shared merge primitive.
        fresh_neighbors: dict[str, set[str]] = {}
        to_commit: list[RelinkEdge] = []
        raced = 0
        for e in edges:
            if e.source_id not in fresh_neighbors:
                fresh_neighbors[e.source_id] = await _existing_neighbor_ids(graph, e.source_id)
            if e.target_id in fresh_neighbors[e.source_id]:
                raced += 1
                logger.info(
                    "Relink: skipping %s -[%s]-> %s — appeared since scan "
                    "(keeping the first writer's rationale/origin)",
                    e.source_id, e.relationship_type, e.target_id,
                )
                continue
            to_commit.append(e)
        if raced:
            logger.warning(
                "Relink: %d edge(s) skipped at commit time — created by another writer since "
                "the scan snapshot", raced,
            )
        if to_commit:
            # write_batch → _merge_edge_tx uses source_label.value / rel_type.value, so these
            # MUST be NodeLabel / RelType enums, NOT plain strings (a str has no .value →
            # AttributeError, which write_batch's transaction would roll back — committing
            # nothing). e.relationship_type is already validated against _ALLOWED_RELS above,
            # so RelType(...) is safe.
            await graph.write_batch(edges=[
                {
                    "source_label": NodeLabel.CONCEPT, "source_id_field": "concept_id",
                    "source_id_value": e.source_id,
                    "target_label": NodeLabel.CONCEPT, "target_id_field": "concept_id",
                    "target_id_value": e.target_id,
                    "rel_type": RelType(e.relationship_type),
                    "properties": {"origin": "relink", "rationale": e.rationale[:500]},
                }
                for e in to_commit
            ])
            committed = len(to_commit)

    return {
        "concepts_scanned": len(concepts),
        "candidate_pairs": len(pairs),
        "edges_proposed": len(proposed),
        "edges_valid": len(edges),
        "edges_committed": committed,
        "failed_batches": failed_batches,
        "skipped_pairs": skipped_pairs,
    }
