"""Multi-layer memory retriever with auto-inject support.

Retrieves relevant memories across all three layers and formats them
for injection into agent context before each turn.
"""

from __future__ import annotations

from openclaw_brain.config import BrainConfig
from openclaw_brain.memory.models import (
    MemoryEntry,
    MemoryLayer,
    PromotionTier,
    RetrievalResult,
)
from openclaw_brain.memory.store import MemoryStore


# Tier priority weights for ranking
_TIER_WEIGHT = {
    PromotionTier.CORE: 1.0,
    PromotionTier.CURATED: 0.8,
    PromotionTier.RETAIN: 0.5,
    PromotionTier.RAW: 0.2,
}


class MemoryRetriever:
    """Retrieves and ranks memories across all layers for context injection."""

    def __init__(self, store: MemoryStore, config: BrainConfig):
        self._store = store
        self._config = config

    async def retrieve(
        self,
        query: str,
        session_id: str | None = None,
        layers: list[MemoryLayer] | None = None,
        max_results: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant memories ranked by tier and relevance.

        Combines: text search + tag matching + session context + tier priority.
        """
        max_results = max_results or self._config.memory.semantic_max_results
        target_layers = layers or list(MemoryLayer)
        results: list[RetrievalResult] = []

        # 1. Core memories always included
        core = await self._store.get_by_layer(MemoryLayer.SEMANTIC, PromotionTier.CORE, limit=10)
        for m in core:
            results.append(RetrievalResult(
                memory=m,
                relevance_score=1.0,
                source_layer=MemoryLayer.SEMANTIC,
                retrieval_reason="core memory — always injected",
            ))

        # 2. Content search across layers
        if query:
            text_matches = await self._store.search_by_content(query, limit=max_results * 2)
            for m in text_matches:
                if m.layer in target_layers and not self._already_in(m.memory_id, results):
                    score = _TIER_WEIGHT.get(m.tier, 0.2) * 0.8
                    results.append(RetrievalResult(
                        memory=m,
                        relevance_score=score,
                        source_layer=m.layer,
                        retrieval_reason="content match",
                    ))

        # 3. Session context (recent episodic memories from current session)
        if session_id and MemoryLayer.EPISODIC in target_layers:
            session_mems = await self._store.get_by_session(session_id, limit=10)
            for m in session_mems:
                if not self._already_in(m.memory_id, results):
                    results.append(RetrievalResult(
                        memory=m,
                        relevance_score=0.6,
                        source_layer=MemoryLayer.EPISODIC,
                        retrieval_reason="current session context",
                    ))

        # 4. Recent high-tier memories
        for layer in target_layers:
            if layer == MemoryLayer.EPISODIC:
                continue  # Already handled via session
            curated = await self._store.get_by_layer(layer, PromotionTier.CURATED, limit=5)
            for m in curated:
                if not self._already_in(m.memory_id, results):
                    results.append(RetrievalResult(
                        memory=m,
                        relevance_score=_TIER_WEIGHT[PromotionTier.CURATED] * 0.6,
                        source_layer=layer,
                        retrieval_reason="curated memory",
                    ))

        # Sort by relevance (desc) and trim
        results.sort(key=lambda r: r.relevance_score, reverse=True)

        # Record access for returned memories
        for r in results[:max_results]:
            await self._store.record_access(r.memory.memory_id)

        return results[:max_results]

    async def format_for_injection(
        self,
        query: str,
        session_id: str | None = None,
        token_budget: int | None = None,
    ) -> str:
        """Retrieve and format memories as a string for agent context injection.

        Respects the configured token budget (approximate: 1 token ~ 4 chars).
        """
        budget = token_budget or self._config.memory.auto_inject_token_budget
        char_budget = budget * 4  # rough approximation

        results = await self.retrieve(query, session_id=session_id)
        if not results:
            return ""

        sections: dict[str, list[str]] = {
            "core": [],
            "semantic": [],
            "episodic": [],
            "procedural": [],
        }

        for r in results:
            tier_label = r.memory.tier.value.upper()
            line = f"- [{tier_label}] {r.memory.content}"
            if r.source_layer == MemoryLayer.SEMANTIC and r.memory.tier == PromotionTier.CORE:
                sections["core"].append(line)
            elif r.source_layer == MemoryLayer.SEMANTIC:
                sections["semantic"].append(line)
            elif r.source_layer == MemoryLayer.EPISODIC:
                sections["episodic"].append(line)
            elif r.source_layer == MemoryLayer.PROCEDURAL:
                sections["procedural"].append(line)

        parts = []
        if sections["core"]:
            parts.append("## Core Memories\n" + "\n".join(sections["core"]))
        if sections["semantic"]:
            parts.append("## Facts & Lessons\n" + "\n".join(sections["semantic"]))
        if sections["episodic"]:
            parts.append("## Session Context\n" + "\n".join(sections["episodic"]))
        if sections["procedural"]:
            parts.append("## Skill Patterns\n" + "\n".join(sections["procedural"]))

        text = "\n\n".join(parts)

        # Trim to budget
        if len(text) > char_budget:
            text = text[:char_budget] + "\n... (truncated)"

        return text

    @staticmethod
    def _already_in(memory_id: str, results: list[RetrievalResult]) -> bool:
        return any(r.memory.memory_id == memory_id for r in results)
