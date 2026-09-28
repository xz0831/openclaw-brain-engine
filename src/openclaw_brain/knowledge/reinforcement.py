"""Reinforcement engine — strengthens knowledge through repeated confirmation.

Handles:
- Conversational reinforcement (user confirms a fact during chat)
- Confidence decay (unused knowledge fades over time)
- Cross-domain bridge detection (periodic scan for new connections)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from openclaw_brain.config import BrainConfig
from openclaw_brain.knowledge.graph.schema import NodeLabel, RelType
from openclaw_brain.knowledge.graph.store import GraphStore


class ReinforcementEngine:
    """Manages knowledge reinforcement, decay, and bridge detection."""

    def __init__(self, graph: GraphStore, config: BrainConfig):
        self._graph = graph
        self._config = config

    # ── Conversational reinforcement ──

    async def reinforce_concept(
        self,
        concept_id: str,
        evidence: str,
        source: str = "conversation",
    ) -> None:
        """Reinforce a concept's confidence based on new evidence.

        Called when: user confirms a concept in conversation, a new PDF
        mentions the same concept, or the agent uses a concept successfully.
        """
        async with await self._graph._session() as session:
            await session.run(
                """
                MATCH (c:Concept {concept_id: $concept_id})
                SET c.reinforcement_count = coalesce(c.reinforcement_count, 1) + 1,
                    c.last_reinforced = datetime(),
                    c.last_evidence = $evidence,
                    c.last_reinforcement_source = $source,
                    c.confidence = CASE
                        WHEN coalesce(c.confidence, 0.7) < 0.95
                        THEN coalesce(c.confidence, 0.7) + 0.02
                        ELSE 0.95
                    END
                """,
                {"concept_id": concept_id, "evidence": evidence, "source": source},
            )

    async def reinforce_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        evidence: str,
    ) -> None:
        """Reinforce an existing edge with new evidence.

        ``rel_type`` is validated against ``RelType`` before being
        interpolated into the Cypher relationship pattern. This method has
        no callers today, but it is public API — should it ever be wired up
        to LLM/PDF-derived input, an unvalidated ``rel_type`` would let an
        attacker-controlled string inject Cypher via the f-string
        relationship-type slot.
        """
        try:
            validated = RelType(rel_type)
        except ValueError:
            raise ValueError(f"Unknown relationship type: {rel_type!r}") from None
        async with await self._graph._session() as session:
            await session.run(
                f"""
                MATCH (a:Concept {{concept_id: $source_id}})
                      -[r:{validated.value}]->
                      (b:Concept {{concept_id: $target_id}})
                SET r.reinforcement_count = coalesce(r.reinforcement_count, 1) + 1,
                    r.last_reinforced = datetime(),
                    r.last_evidence = $evidence,
                    r.confidence = CASE
                        WHEN coalesce(r.confidence, 0.7) < 0.95
                        THEN coalesce(r.confidence, 0.7) + 0.03
                        ELSE 0.95
                    END
                """,
                {"source_id": source_id, "target_id": target_id, "evidence": evidence},
            )

    # ── Confidence decay ──

    async def apply_decay(self, days_threshold: int = 30) -> int:
        """Apply confidence decay to concepts not reinforced recently.

        Returns the number of concepts affected.
        """
        decay_rate = self._config.knowledge.confidence_decay_rate
        floor = self._config.knowledge.confidence_floor

        async with await self._graph._session() as session:
            result = await session.run(
                """
                MATCH (c:Concept)
                WHERE c.last_reinforced IS NOT NULL
                  AND datetime(c.last_reinforced) < datetime() - duration({days: $days})
                  AND coalesce(c.confidence, 0.7) > $floor
                SET c.confidence = CASE
                    WHEN coalesce(c.confidence, 0.7) - $decay > $floor
                    THEN coalesce(c.confidence, 0.7) - $decay
                    ELSE $floor
                END
                RETURN count(c) AS affected
                """,
                {"days": days_threshold, "decay": decay_rate, "floor": floor},
            )
            record = await result.single()
            return record["affected"] if record else 0

    # ── Cross-domain bridge detection ──

    async def find_potential_bridges(self, limit: int = 10) -> list[dict[str, Any]]:
        """Find concept pairs in different domains that share neighbors.

        These are candidates for cross-domain insights — concepts in
        different fields that connect through shared sub-concepts.
        """
        async with await self._graph._session() as session:
            result = await session.run(
                """
                MATCH (a:Concept)-[:DEPENDS_ON|USES_EQUATION|HAS_PARAMETER]->(shared)
                      <-[:DEPENDS_ON|USES_EQUATION|HAS_PARAMETER]-(b:Concept)
                WHERE a.domain <> b.domain
                  AND a.concept_id < b.concept_id
                  AND NOT (a)-[:BRIDGES_TO|TRADES_OFF]-(b)
                WITH a, b, collect(DISTINCT shared) AS shared_nodes, count(DISTINCT shared) AS shared_count
                WHERE shared_count >= 2
                RETURN
                    a.concept_id AS concept_a,
                    a.canonical_name AS name_a,
                    a.domain AS domain_a,
                    b.concept_id AS concept_b,
                    b.canonical_name AS name_b,
                    b.domain AS domain_b,
                    shared_count,
                    [s IN shared_nodes | coalesce(s.canonical_name, s.concept_id)][..5] AS shared_names
                ORDER BY shared_count DESC
                LIMIT $limit
                """,
                {"limit": limit},
            )
            return [dict(record) async for record in result]

    # ── Stats ──

    async def get_reinforcement_stats(self) -> dict[str, Any]:
        """Get reinforcement statistics for the knowledge graph."""
        async with await self._graph._session() as session:
            result = await session.run("""
                MATCH (c:Concept)
                WITH
                    count(c) AS total,
                    avg(coalesce(c.confidence, 0.7)) AS avg_confidence,
                    avg(coalesce(c.reinforcement_count, 1)) AS avg_reinforcement,
                    max(coalesce(c.reinforcement_count, 1)) AS max_reinforcement
                RETURN total, avg_confidence, avg_reinforcement, max_reinforcement
            """)
            record = await result.single()
            if not record:
                return {}
            return {
                "total_concepts": record["total"] or 0,
                "avg_confidence": round(record["avg_confidence"] or 0, 3),
                "avg_reinforcement": round(record["avg_reinforcement"] or 0, 1),
                "max_reinforcement": record["max_reinforcement"] or 0,
            }
