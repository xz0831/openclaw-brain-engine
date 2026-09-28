"""Episodic memory — session event recording and summarization.

Captures timestamped events during a session, then compresses them
into summary memories at session end or rollover.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from openclaw_brain.knowledge.graph.schema import NodeLabel
from openclaw_brain.knowledge.graph.store import GraphStore
from openclaw_brain.memory.models import (
    EpisodicEvent,
    MemoryEntry,
    MemoryLayer,
    PromotionTier,
)
from openclaw_brain.memory.store import MemoryStore


class EpisodicMemory:
    """Manages episodic (session-scoped) memories."""

    def __init__(self, memory_store: MemoryStore, graph: GraphStore):
        self._store = memory_store
        self._graph = graph
        self._current_session_id: str | None = None
        self._buffer: list[EpisodicEvent] = []

    @property
    def session_id(self) -> str | None:
        return self._current_session_id

    async def start_session(self, session_id: str | None = None) -> str:
        """Start a new session. Creates a Session node in the graph."""
        sid = session_id or f"ses_{uuid.uuid4().hex[:12]}"
        self._current_session_id = sid
        self._buffer = []

        await self._graph.merge_node(
            label=NodeLabel.SESSION,
            id_field="session_id",
            id_value=sid,
            properties={
                "session_id": sid,
                "started_at": datetime.now().isoformat(),
                "status": "active",
            },
        )
        return sid

    async def record_event(self, event: EpisodicEvent, session_id: str | None = None) -> str:
        """Record a single event, storing it as a raw episodic memory.

        Args:
            event: The event to record.
            session_id: Session to attribute this event to. Defaults to the ambient "currently
                active" session (``self._current_session_id``) — unchanged behavior for every
                existing caller. Pass explicitly to target a session other than whichever one
                happens to be ambient right now: `_current_session_id`/`_buffer` are shared
                mutable state on this one process-wide EpisodicMemory instance (see
                memory/README.md Trap 1), so if a second `start_session()` clobbers the ambient
                id out from under an in-flight first session, that first session's caller can
                still route its events correctly by passing its own session_id here instead of
                silently landing on whatever session is now ambient.

                Note: `self._buffer` (used by get_buffer()/get_buffer_text()) is still a single
                shared buffer scoped to whatever is currently ambient, not per-session — this fix
                makes the durable Neo4j write/link session-correct; buffer isolation across
                overlapping sessions remains a known, documented gap (best-effort, not fixed
                here).
        """
        sid = session_id if session_id is not None else self._current_session_id
        assert sid, "Call start_session() first, or pass an explicit session_id"

        self._buffer.append(event)

        entry = MemoryEntry(
            memory_id=f"ep_{uuid.uuid4().hex[:12]}",
            layer=MemoryLayer.EPISODIC,
            tier=PromotionTier.RAW,
            content=event.content,
            session_id=sid,
            tags=[event.event_type],
            confidence=0.9,
            created_at=event.timestamp,
        )
        mid = await self._store.save(entry)
        await self._store.link_to_session(mid, sid)
        return mid

    async def end_session(self, summary: str = "", session_id: str | None = None) -> str | None:
        """End a session. Optionally store a session summary.

        Args:
            summary: Optional summary text to persist for the ended session.
            session_id: Session to end. Defaults to the ambient "currently active" session —
                unchanged behavior for every existing caller. Pass explicitly to end a session
                other than whichever one is currently ambient: previously an overlapped/clobbered
                session could NEVER be ended (nothing let you target a session by id), so it sat
                at status="active" in Neo4j forever. Ending a non-ambient session_id never clears
                `_current_session_id`/`_buffer` — only ending the session that IS currently
                ambient does, so a still-active ambient session is never disturbed by ending a
                different one.

        Returns:
            The summary memory_id if a summary was created.
        """
        sid = session_id if session_id is not None else self._current_session_id
        if not sid:
            return None
        is_ambient = sid == self._current_session_id

        # Mark session as ended
        await self._graph.merge_node(
            label=NodeLabel.SESSION,
            id_field="session_id",
            id_value=sid,
            properties={
                "session_id": sid,
                "ended_at": datetime.now().isoformat(),
                "status": "ended",
                # `_buffer` is shared/ambient-scoped, not per-session (see record_event's
                # docstring) — this count is only meaningful when `sid` is the ambient session;
                # for a non-ambient sid it best-effort reports the ambient buffer's size, which
                # may not reflect `sid`'s own event count.
                "event_count": len(self._buffer),
            },
        )

        summary_id = None
        if summary:
            entry = MemoryEntry(
                memory_id=f"ep_sum_{uuid.uuid4().hex[:12]}",
                layer=MemoryLayer.EPISODIC,
                tier=PromotionTier.RETAIN,
                content=summary,
                summary=summary,
                session_id=sid,
                tags=["session_summary"],
                confidence=0.85,
            )
            summary_id = await self._store.save(entry)
            await self._store.link_to_session(summary_id, sid)

        if is_ambient:
            self._current_session_id = None
            self._buffer = []
        return summary_id

    def get_buffer(self) -> list[EpisodicEvent]:
        """Get all events in the current session buffer."""
        return list(self._buffer)

    def get_buffer_text(self, last_n: int | None = None) -> str:
        """Get a text representation of recent buffer events."""
        events = self._buffer[-last_n:] if last_n else self._buffer
        lines = []
        for e in events:
            ts = e.timestamp.strftime("%H:%M:%S")
            lines.append(f"[{ts}] {e.event_type}: {e.content[:200]}")
        return "\n".join(lines)
