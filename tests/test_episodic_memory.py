"""Tests for memory/episodic.py — EpisodicMemory session identity (W-D2 defect 4).

Session identity used to be ambient mutable singleton state: record_event()/end_session() always
acted on self._current_session_id with no way to target a different session. Two overlapping
sessions (a second start_session() before the first one's end_session()) silently cross-linked
content to the wrong Session node with zero error, and the abandoned session could never be
end_session()'d at all (nothing let you target a session by id) — it sat at status="active" in
Neo4j forever. Fixed: both methods now accept an explicit session_id, defaulting to the ambient
session for every existing caller (unchanged behavior) — see episodic.py's docstrings for the
remaining best-effort gap (the in-process _buffer is still ambient-scoped, not per-session).

Mocked MemoryStore/GraphStore throughout (mock fidelity: real methods are async, spec'd so the
mocks are too — see CLAUDE.md Testing Conventions). No real Neo4j — pure unit tests of
EpisodicMemory's own session-routing logic, independent of this session's live ingest.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from openclaw_brain.knowledge.graph.store import GraphStore
from openclaw_brain.memory.episodic import EpisodicMemory
from openclaw_brain.memory.models import EpisodicEvent
from openclaw_brain.memory.store import MemoryStore


def _event(content: str = "hello") -> EpisodicEvent:
    return EpisodicEvent(event_type="user_message", content=content)


@pytest.fixture
def mock_store():
    store = MagicMock(spec=MemoryStore)
    store.save.side_effect = lambda entry: entry.memory_id   # mirrors MemoryStore.save's real return
    store.link_to_session.return_value = None
    return store


@pytest.fixture
def mock_graph():
    graph = MagicMock(spec=GraphStore)
    graph.merge_node.return_value = "node_id"
    return graph


@pytest.fixture
def episodic(mock_store, mock_graph):
    return EpisodicMemory(mock_store, mock_graph)


# ── record_event ──


@pytest.mark.asyncio
async def test_record_event_defaults_to_ambient_session(episodic, mock_store):
    await episodic.start_session("ses_A")

    mid = await episodic.record_event(_event())

    assert mid.startswith("ep_")
    saved_entry = mock_store.save.call_args.args[0]
    assert saved_entry.session_id == "ses_A"
    mock_store.link_to_session.assert_awaited_once_with(mid, "ses_A")


@pytest.mark.asyncio
async def test_record_event_without_any_session_raises(episodic):
    with pytest.raises(AssertionError):
        await episodic.record_event(_event())


@pytest.mark.asyncio
async def test_record_event_explicit_session_id_overrides_ambient(episodic, mock_store):
    await episodic.start_session("ses_A")

    mid = await episodic.record_event(_event(), session_id="ses_OTHER")

    saved_entry = mock_store.save.call_args.args[0]
    assert saved_entry.session_id == "ses_OTHER"
    mock_store.link_to_session.assert_awaited_once_with(mid, "ses_OTHER")
    assert episodic.session_id == "ses_A"   # explicit id never touches ambient state


@pytest.mark.asyncio
async def test_overlapping_sessions_no_longer_cross_contaminate(episodic, mock_store):
    """The exact scenario from the defect: session A starts, then session B starts before A ends
    — clobbering the ambient _current_session_id (wiping _buffer too — a documented, unchanged
    best-effort gap). Before the fix, A's own record_event() call would have silently
    link_to_session'd to B's Session node with zero error. Now A's caller can pass its own
    session_id explicitly and land correctly on A regardless of what's currently ambient."""
    await episodic.start_session("ses_A")
    await episodic.start_session("ses_B")   # clobbers ambient -> "ses_B" (pre-existing behavior)
    assert episodic.session_id == "ses_B"

    mid = await episodic.record_event(_event("A's event"), session_id="ses_A")

    saved_entry = mock_store.save.call_args.args[0]
    assert saved_entry.session_id == "ses_A"
    mock_store.link_to_session.assert_awaited_once_with(mid, "ses_A")   # NOT "ses_B"


# ── end_session ──


@pytest.mark.asyncio
async def test_end_session_without_any_session_returns_none(episodic):
    assert await episodic.end_session() is None


@pytest.mark.asyncio
async def test_end_session_defaults_to_ambient_and_clears_state(episodic, mock_graph):
    await episodic.start_session("ses_A")
    await episodic.record_event(_event())

    result = await episodic.end_session("summary text")

    assert result is not None
    merge_call = mock_graph.merge_node.call_args_list[-1]
    assert merge_call.kwargs["id_value"] == "ses_A"
    assert merge_call.kwargs["properties"]["status"] == "ended"
    assert episodic.session_id is None   # ambient state cleared, same as before the fix
    assert episodic.get_buffer() == []


@pytest.mark.asyncio
async def test_end_session_explicit_non_ambient_id_does_not_disturb_ambient_session(
    episodic, mock_graph,
):
    """DEFECT REGRESSION: previously nothing let you target a specific session_id, so an
    overlapped/abandoned session could never be end_session()'d — permanently stuck at
    status="active" in Neo4j. Now it can be ended explicitly, WITHOUT disturbing a still-active
    different ambient session (ending B's sibling A must not clear B's in-flight ambient state)."""
    await episodic.start_session("ses_A")
    await episodic.start_session("ses_B")   # ses_B now ambient; ses_A is orphaned-but-alive

    result = await episodic.end_session("wrap up A", session_id="ses_A")

    assert result is not None
    merge_call = mock_graph.merge_node.call_args_list[-1]
    assert merge_call.kwargs["id_value"] == "ses_A"
    assert merge_call.kwargs["properties"]["status"] == "ended"
    assert episodic.session_id == "ses_B"   # ambient session (B) untouched


@pytest.mark.asyncio
async def test_end_session_summary_linked_to_target_session(episodic, mock_store):
    await episodic.start_session("ses_A")

    summary_id = await episodic.end_session("did stuff", session_id="ses_A")

    saved_entry = mock_store.save.call_args.args[0]
    assert saved_entry.session_id == "ses_A"
    assert saved_entry.summary == "did stuff"
    mock_store.link_to_session.assert_awaited_once_with(summary_id, "ses_A")
