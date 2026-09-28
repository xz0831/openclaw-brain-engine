"""LangGraph state definition for bounded-round workflows.

This is the TypedDict used as the state schema for LangGraph's StateGraph.
It carries everything the workflow nodes need to make decisions.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


class WorkflowState(TypedDict, total=False):
    """State flowing through the bounded-round LangGraph workflow.

    Uses Annotated reducers for list fields so LangGraph merges
    partial updates correctly.
    """

    # Task identity
    task_id: str
    task_title: str
    session_id: str

    # Phase tracking
    phase: str  # bootstrap | execute | resolve | review | done | error
    round_number: int
    max_rounds: int

    # Conversation / messages
    messages: Annotated[list[dict[str, Any]], operator.add]

    # Memory context (injected before each agent turn)
    memory_context: str

    # Execution results
    tool_results: Annotated[list[dict[str, Any]], operator.add]
    artifacts: Annotated[list[str], operator.add]

    # Resolution
    resolution: str
    review_notes: str
    needs_followup: bool
    followup_action: str

    # Error handling
    error: str
    retry_count: int
