"""Skill router — selects the best skill for a given user input.

Combines keyword matching, pattern matching, and procedural memory
to route inputs to the most appropriate skill handler.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from openclaw_brain.memory.procedural import ProceduralMemory
from openclaw_brain.skills.registry import SkillDefinition, SkillRegistry


class RouteCandidate(BaseModel):
    """A scored candidate skill for routing."""

    skill_name: str
    score: float = Field(ge=0.0, le=1.0)
    match_reason: str = ""


class RouterResult(BaseModel):
    """Result of skill routing — the selected skill and alternatives."""

    selected: RouteCandidate | None = None
    candidates: list[RouteCandidate] = Field(default_factory=list)
    fallback: bool = False


class SkillRouter:
    """Routes user inputs to the best matching skill.

    Scoring combines:
    - Keyword match (0.3 weight)
    - Pattern match (0.3 weight)
    - Skill confidence (0.2 weight)
    - Procedural memory success rate (0.2 weight)
    """

    KEYWORD_WEIGHT = 0.3
    PATTERN_WEIGHT = 0.3
    CONFIDENCE_WEIGHT = 0.2
    HISTORY_WEIGHT = 0.2

    # Minimum score to consider a skill as a candidate
    MIN_SCORE = 0.2

    def __init__(
        self,
        registry: SkillRegistry,
        procedural: ProceduralMemory | None = None,
    ):
        self._registry = registry
        self._procedural = procedural
        self._history_cache: dict[str, float] = {}

    async def route(self, user_input: str) -> RouterResult:
        """Route a user input to the best matching skill.

        Returns the selected skill and ranked alternatives.
        """
        candidates: list[RouteCandidate] = []

        for skill in self._registry.list_skills(enabled_only=True):
            score, reason = await self._score_skill(skill, user_input)
            if score >= self.MIN_SCORE:
                candidates.append(RouteCandidate(
                    skill_name=skill.name,
                    score=score,
                    match_reason=reason,
                ))

        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)

        if not candidates:
            return RouterResult(fallback=True)

        return RouterResult(
            selected=candidates[0],
            candidates=candidates,
        )

    async def _score_skill(
        self,
        skill: SkillDefinition,
        user_input: str,
    ) -> tuple[float, str]:
        """Score how well a skill matches the input. Returns (score, reason)."""
        reasons = []
        total_score = 0.0
        input_lower = user_input.lower()

        # 1. Keyword matching
        keyword_score = 0.0
        matched_keywords = []
        for kw in skill.trigger_keywords:
            if kw.lower() in input_lower:
                keyword_score = 1.0
                matched_keywords.append(kw)
        if skill.name.lower() in input_lower:
            keyword_score = 1.0
            matched_keywords.append(skill.name)
        total_score += keyword_score * self.KEYWORD_WEIGHT
        if matched_keywords:
            reasons.append(f"keywords: {', '.join(matched_keywords)}")

        # 2. Pattern matching
        pattern_score = 0.0
        for pattern in skill.trigger_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                pattern_score = 1.0
                reasons.append(f"pattern: {pattern}")
                break
        total_score += pattern_score * self.PATTERN_WEIGHT

        # 3. Skill confidence
        total_score += skill.confidence * self.CONFIDENCE_WEIGHT
        if skill.confidence >= 0.8:
            reasons.append(f"high confidence ({skill.confidence:.0%})")

        # 4. Procedural memory (historical success)
        history_score = await self._get_history_score(skill.name)
        total_score += history_score * self.HISTORY_WEIGHT
        if history_score > 0:
            reasons.append(f"history: {history_score:.0%}")

        reason = "; ".join(reasons) if reasons else "low match"
        return total_score, reason

    async def _get_history_score(self, skill_name: str) -> float:
        """Get historical success rate from procedural memory."""
        if not self._procedural:
            return 0.0

        # Cache to avoid repeated DB queries within same routing call
        if skill_name in self._history_cache:
            return self._history_cache[skill_name]

        pattern = await self._procedural.get_skill_stats(skill_name)
        score = pattern.success_rate
        self._history_cache[skill_name] = score
        return score

    def clear_cache(self) -> None:
        """Clear the history score cache."""
        self._history_cache.clear()
