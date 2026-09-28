"""Semantic memory — durable facts about entities, projects, and lessons.

Replaces the manual bank/ directory and MEMORY.md with graph-stored
facts that carry provenance and can be queried by entity or topic.
"""

from __future__ import annotations

import uuid
from typing import Any

from openclaw_brain.knowledge.graph.schema import NodeLabel, RelType
from openclaw_brain.knowledge.graph.store import GraphStore
from openclaw_brain.memory.models import (
    MemoryEntry,
    MemoryLayer,
    PromotionTier,
    SemanticFact,
)
from openclaw_brain.memory.store import MemoryStore


class SemanticMemory:
    """Manages semantic (fact-based) memories tied to entities."""

    def __init__(self, memory_store: MemoryStore, graph: GraphStore):
        self._store = memory_store
        self._graph = graph

    async def upsert_entity(
        self,
        entity_id: str,
        entity_type: str,
        canonical_name: str,
        summary: str = "",
        aliases: list[str] | None = None,
    ) -> str:
        """Create or update an entity node."""
        props: dict[str, Any] = {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "canonical_name": canonical_name,
        }
        if summary:
            props["summary"] = summary
        if aliases:
            props["aliases"] = aliases

        await self._graph.merge_node(
            label=NodeLabel.ENTITY,
            id_field="entity_id",
            id_value=entity_id,
            properties=props,
        )
        return entity_id

    async def record_fact(self, fact: SemanticFact) -> str:
        """Store a semantic fact as a curated memory linked to its subject entity."""
        entry = MemoryEntry(
            memory_id=f"sem_{uuid.uuid4().hex[:12]}",
            layer=MemoryLayer.SEMANTIC,
            tier=PromotionTier.CURATED,
            content=f"{fact.predicate}: {fact.value}",
            tags=[fact.fact_type, fact.subject_id],
            confidence=fact.confidence,
        )
        mid = await self._store.save(entry)
        await self._store.link_to_entity(mid, fact.subject_id, fact.predicate)
        return mid

    async def get_entity(self, entity_id: str) -> dict | None:
        """Get an entity node."""
        return await self._graph.get_node(NodeLabel.ENTITY, "entity_id", entity_id)

    async def get_entity_memories(self, entity_id: str, limit: int = 20) -> list[MemoryEntry]:
        """Get all memories linked to an entity."""
        query = """
        MATCH (m:Memory)-[:RELATES_TO]->(e:Entity {entity_id: $entity_id})
        RETURN m ORDER BY m.last_accessed DESC
        LIMIT $limit
        """
        return await self._store._query_memories(query, {"entity_id": entity_id, "limit": limit})

    async def find_entities(self, text: str, limit: int = 10) -> list[dict]:
        """Search entities by name or alias."""
        query = """
        MATCH (e:Entity)
        WHERE toLower(e.canonical_name) CONTAINS toLower($text)
           OR any(alias IN coalesce(e.aliases, []) WHERE toLower(alias) CONTAINS toLower($text))
        RETURN e
        LIMIT $limit
        """
        async with await self._graph._session() as session:
            result = await session.run(query, {"text": text, "limit": limit})
            return [dict(record["e"]) async for record in result]

    async def record_lesson(self, lesson: str, tags: list[str] | None = None) -> str:
        """Store a durable lesson (replaces bank/lessons.md entries)."""
        entry = MemoryEntry(
            memory_id=f"lesson_{uuid.uuid4().hex[:12]}",
            layer=MemoryLayer.SEMANTIC,
            tier=PromotionTier.CURATED,
            content=lesson,
            tags=["lesson"] + (tags or []),
            confidence=0.9,
        )
        return await self._store.save(entry)

    async def get_lessons(self, limit: int = 20) -> list[MemoryEntry]:
        """Get all stored lessons."""
        query = """
        MATCH (m:Memory)
        WHERE m.layer = 'semantic' AND 'lesson' IN m.tags
        RETURN m ORDER BY m.created_at DESC
        LIMIT $limit
        """
        return await self._store._query_memories(query, {"limit": limit})
