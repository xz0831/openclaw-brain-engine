"""Unified memory store backed by Neo4j.

Handles CRUD operations for all three memory layers (episodic, semantic,
procedural) using the Memory and Entity node types from the graph schema.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from openclaw_brain.config import BrainConfig
from openclaw_brain.knowledge.graph.schema import NodeLabel, RelType
from openclaw_brain.knowledge.graph.store import GraphStore
from openclaw_brain.memory.models import (
    MemoryEntry,
    MemoryLayer,
    PromotionTier,
)


class MemoryStore:
    """Async Neo4j-backed store for the unified memory system."""

    def __init__(self, graph: GraphStore, config: BrainConfig):
        self._graph = graph
        self._config = config

    # ── Write ──

    async def save(self, entry: MemoryEntry) -> str:
        """Save a memory entry to Neo4j. Returns the memory_id."""
        if not entry.memory_id:
            entry.memory_id = f"mem_{uuid.uuid4().hex[:12]}"

        props = entry.to_neo4j_props()
        await self._graph.merge_node(
            label=NodeLabel.MEMORY,
            id_field="memory_id",
            id_value=entry.memory_id,
            properties=props,
        )
        return entry.memory_id

    async def link_to_session(self, memory_id: str, session_id: str) -> None:
        """Link a memory to its originating session."""
        await self._graph.merge_edge(
            source_label=NodeLabel.SESSION,
            source_id_field="session_id",
            source_id_value=session_id,
            target_label=NodeLabel.MEMORY,
            target_id_field="memory_id",
            target_id_value=memory_id,
            rel_type=RelType.RECORDED,
            properties={"rationale": "Memory recorded during session", "confidence": 1.0},
        )

    async def link_to_entity(self, memory_id: str, entity_id: str, rationale: str) -> None:
        """Link a memory to a related entity."""
        await self._graph.merge_edge(
            source_label=NodeLabel.MEMORY,
            source_id_field="memory_id",
            source_id_value=memory_id,
            target_label=NodeLabel.ENTITY,
            target_id_field="entity_id",
            target_id_value=entity_id,
            rel_type=RelType.RELATES_TO,
            properties={"rationale": rationale, "confidence": 0.8},
        )

    async def link_to_concept(self, memory_id: str, concept_id: str, rationale: str) -> None:
        """Link a memory to a knowledge concept."""
        await self._graph.merge_edge(
            source_label=NodeLabel.MEMORY,
            source_id_field="memory_id",
            source_id_value=memory_id,
            target_label=NodeLabel.CONCEPT,
            target_id_field="concept_id",
            target_id_value=concept_id,
            rel_type=RelType.RELATES_TO,
            properties={"rationale": rationale, "confidence": 0.7},
        )

    # ── Read ──

    async def get(self, memory_id: str) -> MemoryEntry | None:
        """Fetch a single memory by ID."""
        node = await self._graph.get_node(NodeLabel.MEMORY, "memory_id", memory_id)
        if not node:
            return None
        return self._node_to_entry(node)

    async def get_by_session(self, session_id: str, limit: int = 50) -> list[MemoryEntry]:
        """Get all memories from a session."""
        query = """
        MATCH (s:Session {session_id: $session_id})-[:RECORDED]->(m:Memory)
        RETURN m ORDER BY m.created_at ASC
        LIMIT $limit
        """
        return await self._query_memories(query, {"session_id": session_id, "limit": limit})

    async def get_by_layer(
        self,
        layer: MemoryLayer,
        tier: PromotionTier | None = None,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        """Get memories filtered by layer and optionally tier."""
        if tier:
            query = """
            MATCH (m:Memory)
            WHERE m.layer = $layer AND m.tier = $tier
            RETURN m ORDER BY m.last_accessed DESC
            LIMIT $limit
            """
            params = {"layer": layer.value, "tier": tier.value, "limit": limit}
        else:
            query = """
            MATCH (m:Memory)
            WHERE m.layer = $layer
            RETURN m ORDER BY m.last_accessed DESC
            LIMIT $limit
            """
            params = {"layer": layer.value, "limit": limit}
        return await self._query_memories(query, params)

    async def get_recent(self, hours: int = 24, limit: int = 20) -> list[MemoryEntry]:
        """Get memories from the last N hours."""
        query = """
        MATCH (m:Memory)
        WHERE datetime(m.created_at) > datetime() - duration({hours: $hours})
        RETURN m ORDER BY m.created_at DESC
        LIMIT $limit
        """
        return await self._query_memories(query, {"hours": hours, "limit": limit})

    async def search_by_tags(self, tags: list[str], limit: int = 10) -> list[MemoryEntry]:
        """Find memories matching any of the given tags."""
        query = """
        MATCH (m:Memory)
        WHERE any(tag IN $tags WHERE tag IN m.tags)
        RETURN m ORDER BY m.last_accessed DESC
        LIMIT $limit
        """
        return await self._query_memories(query, {"tags": tags, "limit": limit})

    async def search_by_content(self, text: str, limit: int = 10) -> list[MemoryEntry]:
        """Simple text search in memory content."""
        query = """
        MATCH (m:Memory)
        WHERE toLower(m.content) CONTAINS toLower($text)
           OR toLower(m.summary) CONTAINS toLower($text)
        RETURN m ORDER BY m.last_accessed DESC
        LIMIT $limit
        """
        return await self._query_memories(query, {"text": text, "limit": limit})

    # ── Update ──

    async def record_access(self, memory_id: str) -> None:
        """Bump access count and last_accessed timestamp."""
        async with await self._graph._session() as session:
            await session.run(
                """
                MATCH (m:Memory {memory_id: $memory_id})
                SET m.access_count = coalesce(m.access_count, 0) + 1,
                    m.last_accessed = datetime()
                """,
                {"memory_id": memory_id},
            )

    async def promote(self, memory_id: str, new_tier: PromotionTier) -> None:
        """Promote a memory to a higher tier.

        Updates the tier in-place and appends a PROMOTED_FROM self-loop that
        records the full promotion history as a list of transition strings.
        The self-loop is the canonical way to visualise memory evolution in
        the graph (MATCH (m)-[r:PROMOTED_FROM]->(m) RETURN r.history).
        """
        from datetime import datetime, timezone
        entry = (
            f"{new_tier.value}@{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
        )
        async with await self._graph._session() as session:
            await session.run(
                """
                MATCH (m:Memory {memory_id: $memory_id})
                WITH m, coalesce(m.tier, 'raw') AS old_tier
                SET m.tier = $tier, m._updated_at = datetime()
                WITH m, old_tier
                MERGE (m)-[r:PROMOTED_FROM]->(m)
                ON CREATE SET r.history = [$entry], r._created_at = datetime()
                ON MATCH  SET r.history = coalesce(r.history, []) + [$entry],
                              r._updated_at = datetime()
                """,
                {"memory_id": memory_id, "tier": new_tier.value, "entry": entry},
            )

    # ── Delete ──

    async def expire_old_raw(self, retention_days: int | None = None) -> int:
        """Delete raw-tier memories older than retention period. Returns count deleted."""
        days = retention_days or self._config.memory.episodic_retention_days
        async with await self._graph._session() as session:
            result = await session.run(
                """
                MATCH (m:Memory)
                WHERE m.tier = 'raw'
                  AND datetime(m.created_at) < datetime() - duration({days: $days})
                DETACH DELETE m
                RETURN count(m) AS deleted
                """,
                {"days": days},
            )
            record = await result.single()
            return record["deleted"] if record else 0

    # ── Stats ──

    async def get_stats(self) -> dict[str, Any]:
        """Get memory counts by layer and tier."""
        async with await self._graph._session() as session:
            result = await session.run("""
                MATCH (m:Memory)
                WITH m.layer AS layer, m.tier AS tier, count(m) AS cnt
                RETURN layer, tier, cnt ORDER BY layer, tier
            """)
            stats: dict[str, Any] = {"total": 0, "by_layer": {}, "by_tier": {}}
            async for record in result:
                layer = record["layer"]
                tier = record["tier"]
                cnt = record["cnt"]
                stats["total"] += cnt
                stats["by_layer"][layer] = stats["by_layer"].get(layer, 0) + cnt
                stats["by_tier"][tier] = stats["by_tier"].get(tier, 0) + cnt
            return stats

    # ── Helpers ──

    async def _query_memories(self, query: str, params: dict[str, Any]) -> list[MemoryEntry]:
        """Run a query that returns Memory nodes and convert them."""
        async with await self._graph._session() as session:
            result = await session.run(query, params)
            entries = []
            async for record in result:
                entries.append(self._node_to_entry(dict(record["m"])))
            return entries

    @staticmethod
    def _node_to_entry(node: dict) -> MemoryEntry:
        """Convert a Neo4j node dict to a MemoryEntry."""
        return MemoryEntry(
            memory_id=node.get("memory_id", ""),
            layer=MemoryLayer(node.get("layer", "episodic")),
            tier=PromotionTier(node.get("tier", "raw")),
            content=node.get("content", ""),
            summary=node.get("summary", ""),
            session_id=node.get("session_id", ""),
            tags=node.get("tags", []),
            confidence=node.get("confidence", 0.7),
            created_at=_parse_dt(node.get("created_at")),
            last_accessed=_parse_dt(node.get("last_accessed")),
            access_count=node.get("access_count", 0),
        )


def _parse_dt(val: Any) -> datetime:
    """Parse a datetime from Neo4j (could be ISO string or datetime)."""
    if val is None:
        return datetime.now()
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        return datetime.fromisoformat(val)
    return datetime.now()
