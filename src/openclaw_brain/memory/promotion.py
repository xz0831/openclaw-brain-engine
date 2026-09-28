"""Memory promotion pipeline: raw → retain → curated → core.

Replaces manual MEMORY.md rotation with automated, criteria-based promotion.
Runs periodically to evaluate raw memories and promote/expire them.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from openclaw_brain.config import BrainConfig
from openclaw_brain.memory.models import MemoryLayer, PromotionTier
from openclaw_brain.memory.store import MemoryStore


class PromotionPipeline:
    """Evaluates and promotes memories through tiers based on usage signals."""

    def __init__(self, store: MemoryStore, config: BrainConfig):
        self._store = store
        self._config = config

    async def run(self) -> dict[str, int]:
        """Execute one promotion cycle. Returns counts of actions taken."""
        counts = {
            "raw_to_retain": 0,
            "retain_to_curated": 0,
            "expired": 0,
        }

        # Promote raw → retain: accessed more than once, or tagged important
        raw_memories = await self._store.get_by_layer(
            MemoryLayer.EPISODIC, PromotionTier.RAW, limit=100,
        )
        raw_memories += await self._store.get_by_layer(
            MemoryLayer.SEMANTIC, PromotionTier.RAW, limit=100,
        )
        raw_memories += await self._store.get_by_layer(
            MemoryLayer.PROCEDURAL, PromotionTier.RAW, limit=100,
        )

        for mem in raw_memories:
            if self._should_promote_to_retain(mem):
                await self._store.promote(mem.memory_id, PromotionTier.RETAIN)
                counts["raw_to_retain"] += 1

        # Promote retain → curated: high access count or strong confidence
        retain_memories = await self._store.get_by_layer(
            MemoryLayer.EPISODIC, PromotionTier.RETAIN, limit=100,
        )
        retain_memories += await self._store.get_by_layer(
            MemoryLayer.SEMANTIC, PromotionTier.RETAIN, limit=100,
        )

        for mem in retain_memories:
            if self._should_promote_to_curated(mem):
                await self._store.promote(mem.memory_id, PromotionTier.CURATED)
                counts["retain_to_curated"] += 1

        # Expire old raw memories
        expired = await self._store.expire_old_raw()
        counts["expired"] = expired

        return counts

    @staticmethod
    def _should_promote_to_retain(mem) -> bool:
        """Raw → retain: accessed more than once, or has 'lesson'/'important' tag."""
        if mem.access_count >= 2:
            return True
        important_tags = {"lesson", "important", "session_summary", "insight"}
        if important_tags & set(mem.tags):
            return True
        return False

    @staticmethod
    def _should_promote_to_curated(mem) -> bool:
        """Retain → curated: accessed 5+ times, or confidence >= 0.9."""
        if mem.access_count >= 5:
            return True
        if mem.confidence >= 0.9:
            return True
        return False
