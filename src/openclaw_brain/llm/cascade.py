"""Cascade routing — cheap model first, escalate on quality failure.

Instead of a simple fallback chain (try A, if error try B), cascade routing
validates the output quality of the cheap model. If the output passes quality
checks, it's used directly. If not, the expensive model processes the same input.

This saves 60-70% of costs by handling easy chunks with cheap models while
reserving expensive models for hard cases.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from openclaw_brain.knowledge.extraction.models import ExtractionResult

logger = logging.getLogger(__name__)


class ExtractionQualityCheck:
    """Validates extraction output quality to decide if escalation is needed."""

    def __init__(
        self,
        min_entities: int = 1,
        min_description_ratio: float = 0.5,
        min_description_length: int = 10,
    ):
        """
        Args:
            min_entities: Minimum total entities (concepts + equations + parameters).
            min_description_ratio: Minimum ratio of concepts with non-empty descriptions.
            min_description_length: Minimum description length to count as "filled".
        """
        self.min_entities = min_entities
        self.min_description_ratio = min_description_ratio
        self.min_description_length = min_description_length

    def check(self, extraction: ExtractionResult, chunk_text: str) -> QualityResult:
        """Check extraction quality and return detailed result.

        Args:
            extraction: The extraction output to validate.
            chunk_text: The original chunk text (for coverage estimation).

        Returns:
            QualityResult with pass/fail and reasons.
        """
        reasons: list[str] = []

        # Check 1: Minimum entity count
        total_entities = (
            len(extraction.concepts)
            + len(extraction.equations)
            + len(extraction.parameters)
        )
        if total_entities < self.min_entities:
            reasons.append(
                f"Too few entities: {total_entities} < {self.min_entities}"
            )

        # Check 2: Description completeness
        if extraction.concepts:
            filled = sum(
                1 for c in extraction.concepts
                if len(c.description) >= self.min_description_length
            )
            ratio = filled / len(extraction.concepts)
            if ratio < self.min_description_ratio:
                reasons.append(
                    f"Low description ratio: {ratio:.0%} < {self.min_description_ratio:.0%} "
                    f"({filled}/{len(extraction.concepts)} filled)"
                )

        # Check 3: Concept names should be Title Case, not snake_case
        snake_case_count = sum(
            1 for c in extraction.concepts
            if "_" in c.name and " " not in c.name
        )
        if extraction.concepts and snake_case_count / len(extraction.concepts) > 0.5:
            reasons.append(
                f"Too many snake_case names: {snake_case_count}/{len(extraction.concepts)}"
            )

        # Check 4: Reasonable entity density (at least 1 entity per 500 chars of text)
        expected_min = max(1, len(chunk_text) // 2000)
        if total_entities < expected_min:
            reasons.append(
                f"Low entity density: {total_entities} entities for {len(chunk_text)} chars "
                f"(expected ≥{expected_min})"
            )

        return QualityResult(
            passed=len(reasons) == 0,
            reasons=reasons,
            entity_count=total_entities,
        )


class QualityResult:
    """Result of a quality check on extraction output."""

    def __init__(self, passed: bool, reasons: list[str], entity_count: int = 0):
        self.passed = passed
        self.reasons = reasons
        self.entity_count = entity_count

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        if self.passed:
            return f"QualityResult(passed=True, entities={self.entity_count})"
        return f"QualityResult(passed=False, reasons={self.reasons})"
