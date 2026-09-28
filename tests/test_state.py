"""Tests for unified state models and snapshot store."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from openclaw_brain.state.models import (
    FollowUp,
    Phase,
    RuntimeState,
    TaskRecord,
    TaskStatus,
)
from openclaw_brain.state.snapshot import SnapshotStore


def test_runtime_state_defaults():
    state = RuntimeState()
    assert state.phase == Phase.IDLE
    assert state.round_number == 0
    assert state.tasks == []
    assert state.followups == []


def test_task_record():
    task = TaskRecord(task_id="t1", title="Fix gain calculation")
    assert task.status == TaskStatus.PENDING
    assert task.priority == 5


def test_get_active_task():
    state = RuntimeState(
        active_task_id="t1",
        tasks=[
            TaskRecord(task_id="t1", title="Active task"),
            TaskRecord(task_id="t2", title="Other task"),
        ],
    )
    active = state.get_active_task()
    assert active is not None
    assert active.title == "Active task"


def test_get_active_task_none():
    state = RuntimeState(active_task_id="missing")
    assert state.get_active_task() is None


def test_pending_followups():
    past = datetime.now() - timedelta(hours=1)
    future = datetime.now() + timedelta(hours=1)
    state = RuntimeState(followups=[
        FollowUp(followup_id="f1", action="check", scheduled_for=past),
        FollowUp(followup_id="f2", action="review", scheduled_for=future),
        FollowUp(followup_id="f3", action="done", scheduled_for=past, completed=True),
    ])
    pending = state.get_pending_followups()
    assert len(pending) == 1
    assert pending[0].followup_id == "f1"


def test_to_dict_serializable():
    state = RuntimeState(
        session_id="s1",
        phase=Phase.EXECUTING,
        tasks=[TaskRecord(task_id="t1", title="Test")],
    )
    d = state.to_dict()
    # Should be JSON-serializable
    json_str = json.dumps(d)
    assert "s1" in json_str
    assert "executing" in json_str


# ── SnapshotStore tests ──


def test_snapshot_save_and_load():
    with tempfile.TemporaryDirectory() as tmp:
        store = SnapshotStore(Path(tmp))
        state = RuntimeState(
            session_id="snap_test",
            phase=Phase.EXECUTING,
            round_number=3,
        )
        store.save(state)

        loaded = store.load()
        assert loaded is not None
        assert loaded.session_id == "snap_test"
        assert loaded.phase == Phase.EXECUTING
        assert loaded.round_number == 3


def test_snapshot_load_empty():
    with tempfile.TemporaryDirectory() as tmp:
        store = SnapshotStore(Path(tmp))
        assert store.load() is None


def test_snapshot_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        store = SnapshotStore(Path(tmp))
        state = RuntimeState(session_id="cp_test", phase=Phase.REVIEW)

        path = store.save_checkpoint(state, label="before_fix")
        assert path.exists()
        assert "before_fix" in path.name

        checkpoints = store.list_checkpoints()
        assert len(checkpoints) == 1


def test_snapshot_atomic_write():
    """Verify the current state file is valid JSON even after save."""
    with tempfile.TemporaryDirectory() as tmp:
        store = SnapshotStore(Path(tmp))
        state = RuntimeState(session_id="atomic_test")
        store.save(state)

        # Read raw file and verify it's valid JSON
        with open(store.current_path) as f:
            data = json.load(f)
        assert data["session_id"] == "atomic_test"
