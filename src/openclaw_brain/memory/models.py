"""Memory system data models.

Three-layer memory: episodic (session events), semantic (entities/facts),
procedural (skill execution patterns). Each memory has a promotion tier
(raw → retain → curated → core) that determines persistence and retrieval priority.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryLayer(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class PromotionTier(str, Enum):
    RAW = "raw"          # Just captured, not yet reviewed
    RETAIN = "retain"    # Survived first promotion pass
    CURATED = "curated"  # Explicitly kept by agent or user
    CORE = "core"        # Identity-level, never expires


class MemoryEntry(BaseModel):
    """A single memory record stored in Neo4j."""

    memory_id: str
    layer: MemoryLayer
    tier: PromotionTier = PromotionTier.RAW
    content: str
    summary: str = ""
    session_id: str = ""
    tags: list[str] = Field(default_factory=list)
    confidence: float = 0.7
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    access_count: int = 0
    embedding: list[float] | None = None

    def to_neo4j_props(self) -> dict[str, Any]:
        """Convert to Neo4j-safe properties (no embedding, datetimes as ISO)."""
        props = self.model_dump(exclude={"embedding"})
        props["layer"] = self.layer.value
        props["tier"] = self.tier.value
        props["created_at"] = self.created_at.isoformat()
        props["last_accessed"] = self.last_accessed.isoformat()
        return props


class EpisodicEvent(BaseModel):
    """A timestamped event within a session."""

    event_type: str = Field(description="user_message | agent_response | tool_call | skill_run | error | observation")
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class SemanticFact(BaseModel):
    """A durable fact about an entity, project, or lesson."""

    fact_type: str = Field(description="entity_property | project_status | lesson | preference")
    subject_id: str
    predicate: str
    value: str
    source_memory_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.8


class ProceduralPattern(BaseModel):
    """A learned pattern from skill execution."""

    skill_name: str
    trigger_pattern: str = Field(default="", description="What input patterns lead to this skill being invoked")
    success_rate: float = 0.0
    total_runs: int = 0
    successful_runs: int = 0
    avg_duration_seconds: float = 0.0
    common_errors: list[str] = Field(default_factory=list)
    last_run: datetime | None = None


class RetrievalResult(BaseModel):
    """A memory returned by the retriever with relevance scoring."""

    memory: MemoryEntry
    relevance_score: float = 0.0
    source_layer: MemoryLayer
    retrieval_reason: str = ""
