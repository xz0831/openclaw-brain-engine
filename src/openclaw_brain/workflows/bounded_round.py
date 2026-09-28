"""Bounded-round LangGraph workflow.

Implements the core task lifecycle:
  bootstrap → execute (loop, bounded) → resolve → review → done/followup

Each node is a pure function (state in → partial state out) that
LangGraph composes into a compiled graph with checkpointing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from openclaw_brain.workflows.graph_state import WorkflowState


# ── Node functions ──


def bootstrap(state: WorkflowState) -> dict[str, Any]:
    """Initialize the task: validate inputs, set up context."""
    task_id = state.get("task_id", "")
    task_title = state.get("task_title", "untitled")
    max_rounds = state.get("max_rounds", 10)

    return {
        "phase": "execute",
        "round_number": 1,
        "max_rounds": max_rounds,
        "messages": [{
            "role": "system",
            "content": f"Task '{task_title}' (id={task_id}) bootstrapped at {datetime.now().isoformat()}. "
                       f"Max rounds: {max_rounds}.",
        }],
        "error": "",
    }


def execute(state: WorkflowState) -> dict[str, Any]:
    """Execute one round of work.

    In the real system this calls the LLM + tools. For now it's a
    skeleton that advances the round counter.
    """
    round_num = state.get("round_number", 1)
    messages = state.get("messages", [])

    # Placeholder: in production this invokes the LLM agent
    return {
        "messages": [{
            "role": "assistant",
            "content": f"Round {round_num} executed.",
        }],
        "round_number": round_num + 1,
        "tool_results": [],
    }


def resolve(state: WorkflowState) -> dict[str, Any]:
    """Determine if the task is resolved or needs more work."""
    round_num = state.get("round_number", 1)
    max_rounds = state.get("max_rounds", 10)

    # Check for explicit resolution in the latest message
    messages = state.get("messages", [])
    resolution = state.get("resolution", "")

    if resolution:
        return {"phase": "review"}

    # Auto-resolve if we hit max rounds
    if round_num > max_rounds:
        return {
            "phase": "review",
            "resolution": f"Max rounds ({max_rounds}) reached without explicit resolution.",
        }

    # Continue executing
    return {"phase": "execute"}


def review(state: WorkflowState) -> dict[str, Any]:
    """Review the completed task, decide if followup is needed."""
    resolution = state.get("resolution", "completed")
    needs_followup = state.get("needs_followup", False)

    return {
        "phase": "done" if not needs_followup else "followup",
        "messages": [{
            "role": "system",
            "content": f"Task reviewed. Resolution: {resolution}. "
                       f"Followup needed: {needs_followup}.",
        }],
    }


def handle_error(state: WorkflowState) -> dict[str, Any]:
    """Handle errors by recording them and moving to review."""
    error = state.get("error", "unknown error")
    return {
        "phase": "review",
        "resolution": f"Error occurred: {error}",
        "messages": [{
            "role": "system",
            "content": f"Error handler invoked: {error}",
        }],
    }


# ── Routing functions ──


def route_after_resolve(state: WorkflowState) -> str:
    """Route based on phase after resolve node."""
    phase = state.get("phase", "")
    if phase == "execute":
        return "execute"
    if phase == "review":
        return "review"
    return "handle_error"


def route_after_review(state: WorkflowState) -> str:
    """Route based on phase after review node."""
    phase = state.get("phase", "")
    if phase == "followup":
        return "followup_exit"
    return END


def followup_exit(state: WorkflowState) -> dict[str, Any]:
    """Exit node for tasks that need followup scheduling."""
    return {
        "messages": [{
            "role": "system",
            "content": f"Followup scheduled: {state.get('followup_action', 'check status')}",
        }],
    }


# ── Graph construction ──


def build_bounded_round_graph(checkpointer=None) -> Any:
    """Build and compile the bounded-round workflow graph.

    Returns a compiled LangGraph application.
    """
    graph = StateGraph(WorkflowState)

    # Add nodes
    graph.add_node("bootstrap", bootstrap)
    graph.add_node("execute", execute)
    graph.add_node("resolve", resolve)
    graph.add_node("review", review)
    graph.add_node("handle_error", handle_error)
    graph.add_node("followup_exit", followup_exit)

    # Edges
    graph.add_edge(START, "bootstrap")
    graph.add_edge("bootstrap", "execute")
    graph.add_edge("execute", "resolve")
    graph.add_conditional_edges("resolve", route_after_resolve, ["execute", "review", "handle_error"])
    graph.add_conditional_edges("review", route_after_review, ["followup_exit", END])
    graph.add_edge("handle_error", "review")
    graph.add_edge("followup_exit", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
