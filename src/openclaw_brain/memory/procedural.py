"""Procedural memory — skill execution pattern learning.

Tracks how skills are invoked, their success rates, and common patterns.
Uses SkillRun nodes linked to sessions and concepts.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from openclaw_brain.knowledge.graph.schema import NodeLabel, RelType
from openclaw_brain.knowledge.graph.store import GraphStore
from openclaw_brain.memory.models import (
    MemoryEntry,
    MemoryLayer,
    ProceduralPattern,
    PromotionTier,
)
from openclaw_brain.memory.store import MemoryStore


class ProceduralMemory:
    """Manages procedural (skill execution) memories."""

    def __init__(self, memory_store: MemoryStore, graph: GraphStore):
        self._store = memory_store
        self._graph = graph

    async def record_skill_run(
        self,
        skill_name: str,
        session_id: str,
        input_summary: str,
        output_summary: str,
        success: bool,
        duration_seconds: float = 0.0,
        error: str = "",
    ) -> str:
        """Record a skill execution as a SkillRun node + procedural memory."""
        run_id = f"run_{uuid.uuid4().hex[:12]}"

        # Create SkillRun node
        props: dict[str, Any] = {
            "run_id": run_id,
            "skill_name": skill_name,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "success": success,
            "duration_seconds": duration_seconds,
            "executed_at": datetime.now().isoformat(),
        }
        if error:
            props["error"] = error

        await self._graph.merge_node(
            label=NodeLabel.SKILL_RUN,
            id_field="run_id",
            id_value=run_id,
            properties=props,
        )

        # Link to session
        await self._graph.merge_edge(
            source_label=NodeLabel.SESSION,
            source_id_field="session_id",
            source_id_value=session_id,
            target_label=NodeLabel.SKILL_RUN,
            target_id_field="run_id",
            target_id_value=run_id,
            rel_type=RelType.EXECUTED,
            properties={"rationale": f"Skill '{skill_name}' executed during session", "confidence": 1.0},
        )

        # Store procedural memory
        content = f"Ran skill '{skill_name}': {'success' if success else 'failed'}"
        if error:
            content += f" — error: {error}"
        entry = MemoryEntry(
            memory_id=f"proc_{uuid.uuid4().hex[:12]}",
            layer=MemoryLayer.PROCEDURAL,
            tier=PromotionTier.RAW,
            content=content,
            summary=f"{skill_name}: {'ok' if success else 'fail'}",
            session_id=session_id,
            tags=["skill_run", skill_name],
            confidence=1.0,
        )
        await self._store.save(entry)

        return run_id

    async def get_skill_stats(self, skill_name: str) -> ProceduralPattern:
        """Aggregate stats for a skill across all runs."""
        query = """
        MATCH (sr:SkillRun)
        WHERE sr.skill_name = $skill_name
        RETURN
            count(sr) AS total,
            sum(CASE WHEN sr.success THEN 1 ELSE 0 END) AS successes,
            avg(sr.duration_seconds) AS avg_duration,
            collect(CASE WHEN NOT sr.success AND sr.error IS NOT NULL THEN sr.error END)[..5] AS errors
        """
        async with await self._graph._session() as session:
            result = await session.run(query, {"skill_name": skill_name})
            record = await result.single()

        if not record or record["total"] == 0:
            return ProceduralPattern(skill_name=skill_name)

        total = record["total"]
        successes = record["successes"]
        errors = [e for e in (record["errors"] or []) if e]

        return ProceduralPattern(
            skill_name=skill_name,
            success_rate=successes / total if total > 0 else 0.0,
            total_runs=total,
            successful_runs=successes,
            avg_duration_seconds=record["avg_duration"] or 0.0,
            common_errors=errors,
        )

    async def get_recent_runs(
        self,
        skill_name: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Get recent skill runs, optionally filtered by skill name."""
        if skill_name:
            query = """
            MATCH (sr:SkillRun)
            WHERE sr.skill_name = $skill_name
            RETURN sr ORDER BY sr.executed_at DESC
            LIMIT $limit
            """
            params = {"skill_name": skill_name, "limit": limit}
        else:
            query = """
            MATCH (sr:SkillRun)
            RETURN sr ORDER BY sr.executed_at DESC
            LIMIT $limit
            """
            params = {"limit": limit}

        async with await self._graph._session() as session:
            result = await session.run(query, params)
            return [dict(record["sr"]) async for record in result]
