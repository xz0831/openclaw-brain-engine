"""Unified state models for the openclaw-brain runtime.

Replaces the scattered JSON state files (state.json, loop_state.json,
current_runtime_status.json) with a single typed state structure.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Phase(str, Enum):
    """Top-level runtime phase."""
    IDLE = "idle"
    BOOTSTRAP = "bootstrap"
    EXECUTING = "executing"
    RESOLVING = "resolving"
    REVIEW = "review"
    FOLLOWUP = "followup"
    ERROR = "error"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskRecord(BaseModel):
    """A single work item tracked by the runtime."""
    task_id: str
    title: str
    status: TaskStatus = TaskStatus.PENDING
    priority: int = Field(default=5, ge=1, le=10, description="1=highest, 10=lowest")
    branch: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FollowUp(BaseModel):
    """A scheduled follow-up action."""
    followup_id: str
    task_id: str = ""
    action: str
    scheduled_for: datetime
    created_at: datetime = Field(default_factory=datetime.now)
    completed: bool = False
    result: str = ""


class RuntimeState(BaseModel):
    """Complete runtime state — the single source of truth.

    Replaces: state.json + loop_state.json + current_runtime_status.json
    """
    # Identity
    session_id: str = ""
    phase: Phase = Phase.IDLE

    # Task tracking
    active_task_id: str = ""
    tasks: list[TaskRecord] = Field(default_factory=list)

    # Round management
    round_number: int = 0
    max_rounds: int = 10
    round_started_at: datetime | None = None

    # Follow-ups
    followups: list[FollowUp] = Field(default_factory=list)

    # Execution context
    current_branch: str = "main"
    last_checkpoint: str = ""
    error_message: str = ""

    # Timestamps
    started_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def get_active_task(self) -> TaskRecord | None:
        for t in self.tasks:
            if t.task_id == self.active_task_id:
                return t
        return None

    def get_pending_followups(self) -> list[FollowUp]:
        now = datetime.now()
        return [f for f in self.followups if not f.completed and f.scheduled_for <= now]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence (JSON-safe)."""
        data = self.model_dump()
        # Convert datetimes to ISO strings
        def convert(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        return convert(data)
