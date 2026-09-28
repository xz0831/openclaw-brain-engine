"""Tests for LangGraph bounded-round workflow and followup scheduler."""

from datetime import datetime, timedelta

from openclaw_brain.state.models import FollowUp, RuntimeState
from openclaw_brain.workflows.bounded_round import (
    bootstrap,
    build_bounded_round_graph,
    execute,
    handle_error,
    resolve,
    review,
)
from openclaw_brain.workflows.followup import FollowUpScheduler
from openclaw_brain.workflows.graph_state import WorkflowState


# ── Node function unit tests ──


def test_bootstrap_node():
    state: WorkflowState = {"task_id": "t1", "task_title": "Test Task", "max_rounds": 5}
    result = bootstrap(state)
    assert result["phase"] == "execute"
    assert result["round_number"] == 1
    assert result["max_rounds"] == 5


def test_execute_node():
    state: WorkflowState = {"round_number": 3, "messages": []}
    result = execute(state)
    assert result["round_number"] == 4
    assert len(result["messages"]) == 1
    assert "Round 3" in result["messages"][0]["content"]


def test_resolve_continues():
    state: WorkflowState = {"round_number": 2, "max_rounds": 10, "resolution": ""}
    result = resolve(state)
    assert result["phase"] == "execute"


def test_resolve_max_rounds():
    state: WorkflowState = {"round_number": 11, "max_rounds": 10, "resolution": ""}
    result = resolve(state)
    assert result["phase"] == "review"
    assert "Max rounds" in result["resolution"]


def test_resolve_explicit_resolution():
    state: WorkflowState = {"round_number": 2, "max_rounds": 10, "resolution": "Task done"}
    result = resolve(state)
    assert result["phase"] == "review"


def test_review_no_followup():
    state: WorkflowState = {"resolution": "completed", "needs_followup": False}
    result = review(state)
    assert result["phase"] == "done"


def test_review_with_followup():
    state: WorkflowState = {"resolution": "partial", "needs_followup": True}
    result = review(state)
    assert result["phase"] == "followup"


def test_handle_error():
    state: WorkflowState = {"error": "Neo4j connection failed"}
    result = handle_error(state)
    assert result["phase"] == "review"
    assert "Neo4j connection failed" in result["resolution"]


# ── Compiled graph tests ──


def test_graph_compiles():
    app = build_bounded_round_graph()
    assert app is not None


def test_graph_runs_to_completion():
    """Run the graph with a task that resolves after max rounds."""
    app = build_bounded_round_graph()

    initial_state: WorkflowState = {
        "task_id": "test_1",
        "task_title": "Integration Test",
        "max_rounds": 3,
        "messages": [],
        "tool_results": [],
        "artifacts": [],
        "resolution": "",
        "needs_followup": False,
        "error": "",
        "retry_count": 0,
    }

    result = app.invoke(
        initial_state,
        config={"configurable": {"thread_id": "test_run_1"}},
    )

    assert result["phase"] == "done"
    assert result["round_number"] > 1
    assert len(result["messages"]) > 0


def test_graph_with_early_resolution():
    """Test the graph when resolution is set during execution."""
    app = build_bounded_round_graph()

    initial_state: WorkflowState = {
        "task_id": "test_2",
        "task_title": "Early Resolve Test",
        "max_rounds": 10,
        "messages": [],
        "tool_results": [],
        "artifacts": [],
        "resolution": "Pre-resolved for testing",
        "needs_followup": False,
        "error": "",
        "retry_count": 0,
    }

    result = app.invoke(
        initial_state,
        config={"configurable": {"thread_id": "test_run_2"}},
    )

    assert result["phase"] == "done"


def test_graph_with_followup():
    """Test the graph when followup is needed."""
    app = build_bounded_round_graph()

    initial_state: WorkflowState = {
        "task_id": "test_3",
        "task_title": "Followup Test",
        "max_rounds": 2,
        "messages": [],
        "tool_results": [],
        "artifacts": [],
        "resolution": "",
        "needs_followup": True,
        "followup_action": "Check deployment status",
        "error": "",
        "retry_count": 0,
    }

    result = app.invoke(
        initial_state,
        config={"configurable": {"thread_id": "test_run_3"}},
    )

    # Should end after followup_exit
    assert result["phase"] == "followup"
    assert any("Followup scheduled" in m["content"] for m in result["messages"])


# ── FollowUpScheduler tests ──


def test_scheduler_schedule():
    state = RuntimeState()
    scheduler = FollowUpScheduler(state)
    fu = scheduler.schedule("Check test results", delay_hours=2.0, task_id="t1")
    assert fu.followup_id.startswith("fu_")
    assert fu.task_id == "t1"
    assert len(state.followups) == 1


def test_scheduler_get_due():
    state = RuntimeState(followups=[
        FollowUp(
            followup_id="f1",
            action="check",
            scheduled_for=datetime.now() - timedelta(hours=1),
        ),
        FollowUp(
            followup_id="f2",
            action="future",
            scheduled_for=datetime.now() + timedelta(hours=1),
        ),
    ])
    scheduler = FollowUpScheduler(state)
    due = scheduler.get_due()
    assert len(due) == 1
    assert due[0].followup_id == "f1"


def test_scheduler_complete():
    state = RuntimeState(followups=[
        FollowUp(
            followup_id="f1",
            action="check",
            scheduled_for=datetime.now(),
        ),
    ])
    scheduler = FollowUpScheduler(state)
    assert scheduler.complete("f1", result="All good")
    assert state.followups[0].completed
    assert state.followups[0].result == "All good"


def test_scheduler_pending_count():
    state = RuntimeState(followups=[
        FollowUp(followup_id="f1", action="a", scheduled_for=datetime.now()),
        FollowUp(followup_id="f2", action="b", scheduled_for=datetime.now(), completed=True),
        FollowUp(followup_id="f3", action="c", scheduled_for=datetime.now()),
    ])
    scheduler = FollowUpScheduler(state)
    assert scheduler.get_pending_count() == 2


def test_scheduler_cleanup():
    old = datetime.now() - timedelta(days=10)
    state = RuntimeState(followups=[
        FollowUp(followup_id="f1", action="old", scheduled_for=old, created_at=old, completed=True),
        FollowUp(followup_id="f2", action="recent", scheduled_for=datetime.now(), completed=True),
    ])
    scheduler = FollowUpScheduler(state)
    removed = scheduler.cleanup_completed(keep_days=7)
    assert removed == 1
    assert len(state.followups) == 1
    assert state.followups[0].followup_id == "f2"
