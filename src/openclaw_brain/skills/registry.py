"""Code-based skill registry.

Replaces the manual SKILL_REGISTRY.json with typed skill definitions
that carry execution contracts, input/output schemas, and confidence.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Awaitable

from pydantic import BaseModel, Field


class SkillCategory(str, Enum):
    KNOWLEDGE = "knowledge"       # PDF extraction, graph queries, reasoning
    RENDERING = "rendering"       # Equations, diagrams, band diagrams
    MEMORY = "memory"             # Memory operations, recall
    COMMUNICATION = "communication"  # Telegram, notifications
    SYSTEM = "system"             # Health checks, state management
    ANALYSIS = "analysis"         # Code review, circuit analysis


class SkillDefinition(BaseModel):
    """A registered skill with its metadata and execution contract."""

    name: str
    description: str
    category: SkillCategory
    version: str = "0.1.0"

    # Input/output schema descriptions
    input_schema: dict[str, str] = Field(
        default_factory=dict,
        description="Parameter name → type description",
    )
    output_schema: dict[str, str] = Field(
        default_factory=dict,
        description="Output field name → type description",
    )

    # Routing hints
    trigger_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords that suggest this skill should be invoked",
    )
    trigger_patterns: list[str] = Field(
        default_factory=list,
        description="Regex patterns for input matching",
    )

    # Confidence and learning
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    total_runs: int = 0
    successful_runs: int = 0
    enabled: bool = True

    @property
    def success_rate(self) -> float:
        if self.total_runs == 0:
            return 0.0
        return self.successful_runs / self.total_runs


# Type alias for skill handler functions
SkillHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class SkillRegistry:
    """Registry of available skills with their handlers."""

    def __init__(self):
        self._skills: dict[str, SkillDefinition] = {}
        self._handlers: dict[str, SkillHandler] = {}

    def register(
        self,
        definition: SkillDefinition,
        handler: SkillHandler,
    ) -> None:
        """Register a skill with its handler function."""
        self._skills[definition.name] = definition
        self._handlers[definition.name] = handler

    def unregister(self, name: str) -> bool:
        """Remove a skill from the registry."""
        if name in self._skills:
            del self._skills[name]
            del self._handlers[name]
            return True
        return False

    def get(self, name: str) -> SkillDefinition | None:
        """Get a skill definition by name."""
        return self._skills.get(name)

    def get_handler(self, name: str) -> SkillHandler | None:
        """Get the handler function for a skill."""
        return self._handlers.get(name)

    def list_skills(
        self,
        category: SkillCategory | None = None,
        enabled_only: bool = True,
    ) -> list[SkillDefinition]:
        """List registered skills, optionally filtered."""
        skills = list(self._skills.values())
        if category:
            skills = [s for s in skills if s.category == category]
        if enabled_only:
            skills = [s for s in skills if s.enabled]
        return sorted(skills, key=lambda s: s.confidence, reverse=True)

    def find_by_keyword(self, keyword: str) -> list[SkillDefinition]:
        """Find skills matching a keyword in name, description, or trigger_keywords."""
        keyword_lower = keyword.lower()
        matches = []
        for skill in self._skills.values():
            if not skill.enabled:
                continue
            if keyword_lower in skill.name.lower():
                matches.append(skill)
            elif keyword_lower in skill.description.lower():
                matches.append(skill)
            elif any(keyword_lower in kw.lower() for kw in skill.trigger_keywords):
                matches.append(skill)
        return sorted(matches, key=lambda s: s.confidence, reverse=True)

    def update_stats(self, name: str, success: bool) -> None:
        """Update execution stats for a skill after a run."""
        skill = self._skills.get(name)
        if not skill:
            return
        skill.total_runs += 1
        if success:
            skill.successful_runs += 1
        # Adjust confidence based on recent performance
        skill.confidence = _recalculate_confidence(skill)

    @property
    def count(self) -> int:
        return len(self._skills)

    def to_prompt_description(self) -> str:
        """Format all enabled skills as a description for LLM context."""
        lines = []
        for skill in self.list_skills():
            params = ", ".join(f"{k}: {v}" for k, v in skill.input_schema.items())
            line = f"- **{skill.name}** ({skill.category.value}): {skill.description}"
            if params:
                line += f" | Params: {params}"
            line += f" | Confidence: {skill.confidence:.0%}"
            lines.append(line)
        return "\n".join(lines) if lines else "No skills registered."


def _recalculate_confidence(skill: SkillDefinition) -> float:
    """Recalculate skill confidence using exponential moving average.

    Weighs recent performance more heavily than historical.
    """
    if skill.total_runs == 0:
        return skill.confidence

    # Base: success rate
    rate = skill.success_rate

    # EMA: blend with prior confidence (alpha = 0.3 for recent weight)
    alpha = 0.3
    new_conf = alpha * rate + (1 - alpha) * skill.confidence

    # Clamp
    return max(0.1, min(1.0, new_conf))
