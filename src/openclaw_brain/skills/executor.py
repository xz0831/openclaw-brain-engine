"""Skill executor with automatic instrumentation.

Wraps skill handler execution with timing, error capture, and
automatic recording to procedural memory.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from openclaw_brain.memory.procedural import ProceduralMemory
from openclaw_brain.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


class ExecutionResult(BaseModel):
    """Result of a skill execution."""

    skill_name: str
    success: bool
    output: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    duration_seconds: float = 0.0
    run_id: str = ""


class SkillExecutor:
    """Executes skills with automatic instrumentation and learning."""

    def __init__(
        self,
        registry: SkillRegistry,
        procedural: ProceduralMemory | None = None,
    ):
        self._registry = registry
        self._procedural = procedural

    async def execute(
        self,
        skill_name: str,
        params: dict[str, Any],
        session_id: str = "",
    ) -> ExecutionResult:
        """Execute a skill by name with full instrumentation.

        Records timing, success/failure, and updates procedural memory.
        """
        handler = self._registry.get_handler(skill_name)
        if not handler:
            return ExecutionResult(
                skill_name=skill_name,
                success=False,
                error=f"Skill '{skill_name}' not found in registry",
            )

        skill_def = self._registry.get(skill_name)
        if skill_def and not skill_def.enabled:
            return ExecutionResult(
                skill_name=skill_name,
                success=False,
                error=f"Skill '{skill_name}' is disabled",
            )

        # Execute with timing
        start = time.monotonic()
        try:
            output = await handler(params)
            duration = time.monotonic() - start
            success = True
            error = ""
        except Exception as e:
            duration = time.monotonic() - start
            success = False
            error = f"{type(e).__name__}: {e}"
            output = {}

        # Update registry stats
        self._registry.update_stats(skill_name, success)

        # Record to procedural memory. Exception-isolated from the handler's own result: a
        # failure here (transient Neo4j error, missing Session anchor, etc.) must not discard
        # visibility into an already-successful handler run — the caller would otherwise see a
        # hard tool-call failure for an operation (e.g. a multi-hour pdf_ingest) that actually
        # completed, risking a costly duplicate retry. run_id simply stays "" on failure.
        run_id = ""
        if self._procedural and session_id:
            input_summary = _summarize_params(params)
            output_summary = _summarize_output(output) if success else error
            try:
                run_id = await self._procedural.record_skill_run(
                    skill_name=skill_name,
                    session_id=session_id,
                    input_summary=input_summary,
                    output_summary=output_summary,
                    success=success,
                    duration_seconds=duration,
                    error=error if not success else "",
                )
            except Exception as exc:
                logger.warning(
                    "Procedural memory write failed for skill=%s session=%s "
                    "(handler result unaffected): %s",
                    skill_name, session_id, exc,
                )

        return ExecutionResult(
            skill_name=skill_name,
            success=success,
            output=output,
            error=error,
            duration_seconds=duration,
            run_id=run_id,
        )


def _summarize_params(params: dict[str, Any], max_len: int = 200) -> str:
    """Create a brief summary of input parameters."""
    parts = []
    for k, v in params.items():
        val_str = str(v)
        if len(val_str) > 50:
            val_str = val_str[:47] + "..."
        parts.append(f"{k}={val_str}")
    summary = ", ".join(parts)
    return summary[:max_len] if len(summary) > max_len else summary


def _summarize_output(output: dict[str, Any], max_len: int = 200) -> str:
    """Create a brief summary of output."""
    if not output:
        return "empty output"
    parts = []
    for k, v in output.items():
        val_str = str(v)
        if len(val_str) > 50:
            val_str = val_str[:47] + "..."
        parts.append(f"{k}={val_str}")
    summary = ", ".join(parts)
    return summary[:max_len] if len(summary) > max_len else summary
