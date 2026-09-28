"""Follow-up scheduler — automatic scheduling and execution of deferred actions.

When a task review determines a followup is needed, this module creates
FollowUp records and checks for due items on each session start.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from openclaw_brain.state.models import FollowUp, RuntimeState


class FollowUpScheduler:
    """Manages follow-up scheduling and retrieval."""

    def __init__(self, state: RuntimeState):
        self._state = state

    def schedule(
        self,
        action: str,
        delay_hours: float = 24.0,
        task_id: str = "",
    ) -> FollowUp:
        """Schedule a followup action for the future."""
        followup = FollowUp(
            followup_id=f"fu_{uuid.uuid4().hex[:12]}",
            task_id=task_id or self._state.active_task_id,
            action=action,
            scheduled_for=datetime.now() + timedelta(hours=delay_hours),
        )
        self._state.followups.append(followup)
        self._state.updated_at = datetime.now()
        return followup

    def get_due(self) -> list[FollowUp]:
        """Get all followups that are due now."""
        return self._state.get_pending_followups()

    def complete(self, followup_id: str, result: str = "") -> bool:
        """Mark a followup as completed."""
        for fu in self._state.followups:
            if fu.followup_id == followup_id:
                fu.completed = True
                fu.result = result
                self._state.updated_at = datetime.now()
                return True
        return False

    def get_pending_count(self) -> int:
        """Number of incomplete followups."""
        return sum(1 for f in self._state.followups if not f.completed)

    def get_overdue(self) -> list[FollowUp]:
        """Get followups that are past their scheduled time."""
        now = datetime.now()
        return [
            f for f in self._state.followups
            if not f.completed and f.scheduled_for < now
        ]

    def cleanup_completed(self, keep_days: int = 7) -> int:
        """Remove completed followups older than keep_days. Returns count removed."""
        cutoff = datetime.now() - timedelta(days=keep_days)
        before = len(self._state.followups)
        self._state.followups = [
            f for f in self._state.followups
            if not (f.completed and f.created_at < cutoff)
        ]
        removed = before - len(self._state.followups)
        if removed:
            self._state.updated_at = datetime.now()
        return removed
