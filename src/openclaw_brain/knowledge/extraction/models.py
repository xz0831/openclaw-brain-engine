"""Data models for the extraction and reasoning pipeline.

These are the intermediate representations flowing between pipeline stages.
Separate from the graph schema models (which represent stored state).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Stage 1: Chunker output ──


class SourceChunkInfo(BaseModel):
    """A chunk of text extracted from a source document."""

    chunk_id: str
    source_id: str
    text: str
    pages: str = ""
    section_title: str = ""
    raw_text_hash: str = ""
    chunk_index: int = 0
    token_estimate: int = 0


class ChunkerResult(BaseModel):
    """Output of the chunking stage."""

    source_id: str
    title: str
    author: str = ""
    total_pages: int = 0
    chunks: list[SourceChunkInfo] = Field(default_factory=list)
    checksum: str = ""


# ── Stage 2: Extractor output ──


class ConceptMention(BaseModel):
    """A concept mentioned in a chunk."""

    name: str = Field(
        description="Human-readable concept name in Title Case (e.g., 'Threshold Voltage', 'Common-Source Amplifier')"
    )
    description: str = Field(
        min_length=10,
        description="A concise but informative description (at least 10 characters). Explain what the concept IS, not just its name.",
    )
    domain: str = "general"
    granularity: str = Field(default="atomic", description="atomic | composite")
    grounding_flags: list[str] = Field(default_factory=list)


class EquationMention(BaseModel):
    """An equation found in a chunk."""

    latex: str
    equation_type: str = Field(
        default="definition",
        description="definition | law | derived | approximation",
    )
    variables: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class ParameterMention(BaseModel):
    """A parameter/variable found in a chunk."""

    symbol: str
    name: str
    units: str = ""
    typical_range: str = ""


class RawEdge(BaseModel):
    """A relationship detected within a single chunk (pre-matching)."""

    source_name: str
    target_name: str
    relationship: str
    rationale: str = ""
    confidence: float = 0.7


class ExtractionResult(BaseModel):
    """Structured extraction from a single chunk."""

    chunk_id: str
    concepts: list[ConceptMention] = Field(default_factory=list)
    equations: list[EquationMention] = Field(default_factory=list)
    parameters: list[ParameterMention] = Field(default_factory=list)
    raw_edges: list[RawEdge] = Field(default_factory=list)


# ── Stage 3: Matcher output ──


class MatchedConcept(BaseModel):
    """A concept matched to an existing graph node."""

    mention: ConceptMention
    existing_node_id: str
    similarity: float = 0.0
    match_method: str = Field(default="name", description="name | embedding | llm")
    # Label and id_field of the matched node (defaults to Concept for backward compat)
    node_label: str = Field(default="Concept")
    node_id_field: str = Field(default="concept_id")


class AmbiguousMatch(BaseModel):
    """A concept that partially matches existing nodes — needs LLM resolution."""

    mention: ConceptMention
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    best_similarity: float = 0.0


class MatchedEquation(BaseModel):
    """An equation matched to an existing graph node (F6).

    Exact match on NORMALIZED canonical_latex only (see
    reasoning/matcher.py::_normalize_latex / _match_equations) — deliberately no
    fuzzy/embedding tier, so unlike MatchedConcept.similarity this is always 1.0.
    """

    mention: EquationMention
    existing_node_id: str
    similarity: float = 1.0
    match_method: str = Field(
        default="canonical_latex", description="canonical_latex (normalized exact match)"
    )
    node_label: str = Field(default="Equation")
    node_id_field: str = Field(default="equation_id")


class MatchedParameter(BaseModel):
    """A parameter matched to an existing graph node (F6).

    Exact match on normalized canonical_name PLUS unit agreement (both sides
    carrying a DIFFERENT unit disqualifies the candidate entirely; one side missing
    a unit is gap-fill semantics) — see
    reasoning/matcher.py::_normalize_plain / _match_parameters. No fuzzy/embedding
    tier, so similarity is always 1.0.
    """

    mention: ParameterMention
    existing_node_id: str
    similarity: float = 1.0
    match_method: str = Field(
        default="canonical_name", description="canonical_name (normalized exact match, unit-checked)"
    )
    node_label: str = Field(default="Parameter")
    node_id_field: str = Field(default="parameter_id")


class MatchResult(BaseModel):
    """Output of the matching stage."""

    chunk_id: str
    matched: list[MatchedConcept] = Field(default_factory=list)
    new_concepts: list[ConceptMention] = Field(default_factory=list)
    ambiguous: list[AmbiguousMatch] = Field(default_factory=list)
    # F6: equations/parameters now flow through the matcher too (previously they bypassed
    # matching entirely and were ALWAYS proposed as new_nodes — see
    # knowledge/reasoning/README.md "Known defects" #2). Exact-normalized matching only, no
    # fuzzy/verify band, so there is no equation/parameter equivalent of `ambiguous` above —
    # every mention resolves to exactly one of matched_*/new_* (a strict partition of
    # ExtractionResult.equations / .parameters, mirroring the matched/new_concepts split).
    matched_equations: list[MatchedEquation] = Field(default_factory=list)
    new_equations: list[EquationMention] = Field(default_factory=list)
    # NOTE: this field previously existed as `list[dict[str, Any]]` but was declared, never
    # written or read anywhere in the repo (confirmed by the 2026-07-10 architecture survey —
    # see knowledge/reasoning/README.md and knowledge/extraction/README.md). F6 repurposes it
    # with the same name (no dangling near-duplicate field) and gives it real semantics; the
    # type change is safe because there were zero prior readers/writers to break.
    matched_parameters: list[MatchedParameter] = Field(default_factory=list)
    new_parameters: list[ParameterMention] = Field(default_factory=list)


# ── Pipeline-level ──


class PipelineProgress(BaseModel):
    """Progress report emitted during pipeline execution."""

    stage: str
    chunk_index: int
    total_chunks: int
    message: str = ""

    @property
    def percent(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return (self.chunk_index / self.total_chunks) * 100


class IngestResult(BaseModel):
    """Final result of the full PDF ingest pipeline."""

    source_id: str
    title: str
    total_chunks: int = 0
    new_nodes: int = 0
    updated_nodes: int = 0
    new_edges: int = 0
    reinforced_edges: int = 0
    insights: int = 0
    errors: list[str] = Field(default_factory=list)
    # Legitimately-empty input (figures-only: no slide images / every slide gated as
    # text-only). NOT an error: the pipeline correctly found nothing to ingest. CLI maps
    # this to exit 3 so batch telemetry can tell "empty" from "failed" — in the 2026-07
    # lecture batches 78/88 exit-1s were this, reading as a 76% failure rate that wasn't
    # (GROUNDING_REALIGNMENT C6).
    empty_reason: str | None = None

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        parts = [f"'{self.title}' processed ({self.total_chunks} chunks)"]
        if self.new_nodes:
            parts.append(f"{self.new_nodes} new concepts")
        if self.updated_nodes:
            parts.append(f"{self.updated_nodes} updated")
        if self.new_edges:
            parts.append(f"{self.new_edges} new connections")
        if self.reinforced_edges:
            parts.append(f"{self.reinforced_edges} reinforced")
        if self.insights:
            parts.append(f"{self.insights} insights")
        if self.errors:
            parts.append(f"{len(self.errors)} errors")
        return ", ".join(parts)
