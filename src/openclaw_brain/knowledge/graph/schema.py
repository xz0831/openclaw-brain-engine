"""Knowledge graph node and edge type definitions.

Domain: semiconductor / analog circuit design knowledge + agent memory.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Node Labels ──


class NodeLabel(str, Enum):
    CONCEPT = "Concept"
    EQUATION = "Equation"
    PRINCIPLE = "Principle"
    CIRCUIT_TOPOLOGY = "CircuitTopology"
    PARAMETER = "Parameter"
    ASSUMPTION = "Assumption"
    SOURCE = "Source"
    SOURCE_CHUNK = "SourceChunk"
    INSIGHT = "Insight"
    # Memory nodes
    MEMORY = "Memory"
    ENTITY = "Entity"
    SESSION = "Session"
    SKILL_RUN = "SkillRun"
    # Design reasoning nodes
    HYPOTHESIS = "Hypothesis"
    DESIGN_DECISION = "DesignDecision"
    BENCH_RESULT = "BenchResult"
    # executable-circuit substrate (v2) — derived projection of git-corpus specimens
    SPECIMEN = "Specimen"
    CLAIM_CARD = "ClaimCard"
    # law-tier graph representation (docs/superpowers/specs/2026-07-04-law-tier-graph-
    # representation.md §3) — a cross-PDK regularity derived from the replication runner's raw
    # JSONL (knowledge/executable/laws.py). Projector-only: deliberately NOT added to
    # reasoning/normalize.py's _VALID_LABELS (see that module's comment) so the LLM extraction/
    # reasoning path can never author one — same discipline already applied to SPECIMEN/CLAIM_CARD.
    REGULARITY = "Regularity"
    # Symbolic-anchor derivation record (2026-07-20 grounding-realignment work): an lcapy/sympy
    # derivation that verifies an Equation symbolically — (Equation)-[:DERIVED_BY]->
    # (SymbolicDerivation), keyed by derivation_id. Created via graph-surgery Cypher on
    # 2026-07-20 WITHOUT this registration; the gap made export_graph silently drop all 76
    # DERIVED_BY edges and import_graph drop the nodes too (found 2026-07-28 during the
    # ARCH_REVIEW backup/restore verification). Projector-only discipline: NOT added to
    # reasoning/normalize.py's _VALID_LABELS (same as SPECIMEN/CLAIM_CARD/REGULARITY).
    SYMBOLIC_DERIVATION = "SymbolicDerivation"
    # Learner model (S5, ADR-044 D4; docs/specs/S5_LEARNER_MODEL_DESIGN.md §2/§6.1) — a
    # persistent record of what a learner has been assessed on, kept fully separate from the
    # Memory subsystem (a conscious departure from D4's literal "build on the memory subsystem"
    # wording — see design doc §2 for why) and from the Concept-confidence truth layer (§3.2).
    # Learner = identity; Assessment = one immutable probe event (mirrors BenchResult). Home-only:
    # never added to reasoning/normalize.py's _VALID_LABELS (same projector-only discipline as
    # SPECIMEN/CLAIM_CARD/REGULARITY — record_assessment is the only author) and always excluded
    # from the company export (export/graph_io.py DEFAULT_PRIVATE_LABELS, §4's non-negotiable rule).
    LEARNER = "Learner"
    ASSESSMENT = "Assessment"


# ── Relationship Types ──


class RelType(str, Enum):
    # ── Knowledge relationships ──
    # Basic structural
    USES_EQUATION = "USES_EQUATION"
    HAS_PARAMETER = "HAS_PARAMETER"
    DEPENDS_ON = "DEPENDS_ON"
    ASSUMES = "ASSUMES"
    DERIVED_FROM = "DERIVED_FROM"
    APPROXIMATION_OF = "APPROXIMATION_OF"
    EXTRACTED_FROM = "EXTRACTED_FROM"
    CONFIRMS = "CONFIRMS"
    CONTRADICTS = "CONTRADICTS"
    REFINES = "REFINES"
    RELATES_TO = "RELATES_TO"

    # Design reasoning — WHY circuits are designed this way
    SOLVES_PROBLEM = "SOLVES_PROBLEM"           # "Cascode solves low output impedance"
    INTRODUCES_PROBLEM = "INTRODUCES_PROBLEM"   # "Cascode introduces headroom loss"
    TRADES_OFF = "TRADES_OFF"                   # "Gain vs bandwidth tradeoff"
    DESIGN_RULE = "DESIGN_RULE"                 # Practical guideline

    # Topology evolution — how circuit architectures relate
    EVOLVES_TO = "EVOLVES_TO"                   # "CS → Cascode (to increase gain)"
    SUB_BLOCK = "SUB_BLOCK"                     # "OTA contains Diff Pair as input stage"
    TOPOLOGY_VARIANT = "TOPOLOGY_VARIANT"       # "Folded cascode is a variant of cascode"
    COMPENSATED_BY = "COMPENSATED_BY"           # "Two-stage OTA compensated by Miller cap"

    # Equation-structure mapping — connecting math to physical circuit
    MODELS_BEHAVIOR = "MODELS_BEHAVIOR"         # "gm·ro equation models CS amplifier gain"
    VARIABLE_MAPS_TO = "VARIABLE_MAPS_TO"       # "gm in Av=-gm·RD maps to input transistor M1"

    # Cross-domain
    BRIDGES_TO = "BRIDGES_TO"

    # Design reasoning — hypothesis→bench→decision cycle
    TESTS_HYPOTHESIS = "TESTS_HYPOTHESIS"       # BenchResult → Hypothesis
    DECIDED_BY = "DECIDED_BY"                   # Concept/Topology → DesignDecision
    SUPERSEDES = "SUPERSEDES"                   # DesignDecision → DesignDecision
    FALSIFIED_BY = "FALSIFIED_BY"               # Hypothesis → BenchResult
    MOTIVATED_BY = "MOTIVATED_BY"               # DesignDecision → Hypothesis/Principle

    # ── Memory relationships ──
    RECORDED = "RECORDED"
    PROMOTED_FROM = "PROMOTED_FROM"
    EXECUTED = "EXECUTED"
    # executable-circuit substrate (v2) — seam to the existing text-knowledge graph
    REALIZES = "REALIZES"        # Specimen → CircuitTopology (executable realization of a text topology)
    HAS_CLAIM = "HAS_CLAIM"      # Specimen → ClaimCard
    GROUNDS = "GROUNDS"          # ClaimCard → Concept/Parameter (sim grounds a text concept)
    # law-tier graph representation — Regularity's own edges (spec §3). Same
    # projector-only discipline as REALIZES/HAS_CLAIM/GROUNDS: deliberately NOT added to
    # reasoning/normalize.py's _VALID_REL_TYPES.
    SUPPORTED_BY = "SUPPORTED_BY"  # Regularity → ClaimCard (one edge per EXISTING member card, NO-PHANTOM)
    ABOUT = "ABOUT"                # Regularity → CircuitTopology (resolver-linked, NO-PHANTOM)
    # Symbolic-anchor edges (2026-07-20 grounding-realignment; registered 2026-07-28 — see the
    # SYMBOLIC_DERIVATION NodeLabel comment for the export/import data-loss incident this gap
    # caused). Same projector-only discipline: NOT in normalize.py's _VALID_REL_TYPES.
    DERIVED_BY = "DERIVED_BY"      # Equation → SymbolicDerivation (lcapy/sympy symbolic anchor)
    MEASURED_BY = "MEASURED_BY"    # Equation → ClaimCard (triple anchor: textbook↔symbolic↔ngspice)

    # ── Learner model (S5) — kept fully separate from the truth-layer edges above; see
    # docs/specs/S5_LEARNER_MODEL_DESIGN.md §2/§3.2/§6.1 ──
    # Learner → target (label-agnostic: Concept, Regularity, ClaimCard, ...). Rolled-up state,
    # MATCH-only-updated (never fabricates target): confidence, status, last_assessed,
    # assessment_count. One edge per (learner, target) pair.
    UNDERSTANDS = "UNDERSTANDS"
    # Assessment → target (same label-agnostic target). Immutable per-probe historical edge,
    # mirroring BenchResult -[:TESTS_HYPOTHESIS]-> Hypothesis exactly (§1.5) — the event record,
    # kept separate from UNDERSTANDS's rolled-up state.
    ASSESSES = "ASSESSES"


# ── Shared Edge Properties ──


class EdgeProperties(BaseModel):
    """Every edge in the graph carries these base properties."""

    rationale: str = Field(description="Why this connection exists (1-2 sentences)")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    evidence_sources: list[str] = Field(
        default_factory=list,
        description="List of chunk_ids or memory_ids that support this edge",
    )
    created_at: datetime = Field(default_factory=datetime.now)
    last_reinforced: datetime = Field(default_factory=datetime.now)
    reinforcement_count: int = Field(default=1)
    created_by: str = Field(
        default="extraction",
        description="extraction | reasoning | human | reinforcement",
    )


# ── Knowledge Node Models ──


class ConceptNode(BaseModel):
    concept_id: str
    canonical_name: str
    description: str = ""
    domain: str = "general"
    granularity: str = Field(default="atomic", description="atomic | composite")
    confidence: float = 0.7
    knowledge_layer: int = Field(
        default=-1,
        description="Knowledge hierarchy layer: 0=math/physics, 1=device physics, 2=analog circuits/EDA, 3=CIS architecture. -1=unclassified.",
    )
    first_seen: datetime = Field(default_factory=datetime.now)
    last_reinforced: datetime = Field(default_factory=datetime.now)
    reinforcement_count: int = 1
    embedding: list[float] | None = None


class EquationNode(BaseModel):
    equation_id: str
    canonical_latex: str
    equation_type: str = Field(
        default="definition",
        description="definition | law | derived | approximation",
    )
    variable_signature: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence: float = 0.7
    # Structure-equation mapping
    physical_meaning: str = Field(
        default="",
        description="What does this equation physically represent? (e.g., 'small-signal voltage gain of a CS amplifier')",
    )
    component_mapping: list[str] = Field(
        default_factory=list,
        description="How variables map to circuit components (e.g., 'gm → input transistor M1', 'RD → drain load resistor')",
    )


class PrincipleNode(BaseModel):
    principle_id: str
    name: str
    statement: str
    domain: str = "general"
    scope: str = ""


class CircuitTopologyNode(BaseModel):
    topology_id: str
    name: str
    function: str = Field(
        default="",
        description="amplifier | bias | feedback | driver | comparator | ...",
    )
    transistor_count: int | None = None
    key_nodes: list[str] = Field(default_factory=list)
    # Design reasoning fields
    solves: str = Field(default="", description="What problem does this topology solve?")
    introduces: str = Field(default="", description="What new problem/limitation does it introduce?")
    when_to_use: str = Field(default="", description="Under what conditions should a designer choose this?")
    sub_blocks: list[str] = Field(default_factory=list, description="Constituent sub-circuits")


class ParameterNode(BaseModel):
    parameter_id: str
    symbol: str
    name: str
    units: str = ""
    typical_range: str = ""


class SourceNode(BaseModel):
    source_id: str
    title: str
    author: str = ""
    checksum: str = ""
    # Provenance: the PRIMARY models the source was ingested with (the configured chain heads).
    # Semantics: "ingested-with", not a per-chunk actual-model audit — a transient mid-run fallback
    # (rate-limit) is NOT reflected here. The systematic silent-auth-fallback case (the reason this
    # field exists) is caught separately by the ingestion smoke gate, so these together let a reader
    # tell frontier-built from fallback-built going forward without assuming. "" = pre-provenance node.
    extraction_model: str = ""
    reasoning_model: str = ""


class SourceChunkNode(BaseModel):
    chunk_id: str
    source_id: str
    pages: str = ""
    section_title: str = ""
    raw_text_hash: str = ""
    text_preview: str = Field(default="", max_length=500)


class InsightNode(BaseModel):
    insight_id: str
    statement: str
    confidence: float = 0.5
    bridge_type: str = Field(
        default="within-domain",
        description="within-domain | cross-domain",
    )
    source_id: str = ""
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    reviewed: bool = False
    created_at: datetime = Field(default_factory=datetime.now)


class AssumptionNode(BaseModel):
    assumption_id: str
    statement: str
    scope: str = ""


# ── Design Reasoning Node Models ──


class HypothesisNode(BaseModel):
    hypothesis_id: str
    statement: str
    status: str = Field(default="open", description="open | confirmed | falsified | superseded")
    confidence: float = 0.5
    assumptions: list[str] = Field(default_factory=list)
    test_plan: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class DesignDecisionNode(BaseModel):
    decision_id: str
    choice: str
    alternatives: list[str] = Field(default_factory=list)
    rationale: str = ""
    constraints: list[str] = Field(default_factory=list)
    status: str = Field(default="active", description="active | superseded | revisit")
    valid_until: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class BenchResultNode(BaseModel):
    bench_id: str
    bench_type: str = Field(default="simulation", description="simulation | measurement | calculation")
    setup: str = ""
    metric: str = ""
    corner: str = ""
    conclusion: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


# ── Learner Model Node Models (S5) ──
#
# Documentation-only, same as the Design Reasoning models above (HypothesisNode/
# DesignDecisionNode/BenchResultNode) — record_assessment (agent.py) builds the matching raw
# props dict directly rather than instantiating these; they exist to pin the shape. NOT part of
# GraphDelta (never LLM-authored — see NodeLabel.LEARNER/ASSESSMENT's comment).


class LearnerNode(BaseModel):
    """Identity only. No mutable descriptive fields — every per-call property record_assessment
    could set here (e.g. a "first_seen" timestamp) would otherwise be reset on every re-MERGE by
    _merge_node_tx's ON MATCH branch (only _PROTECTED_ON_MATCH fields survive that); the node's
    own system-managed ``_created_at`` already covers "when was this learner first seen" for
    free, so nothing else needs to ride in the properties dict."""

    learner_id: str


class AssessmentNode(BaseModel):
    """One immutable probe event — mirrors BenchResultNode. ``verdict`` is free text (e.g.
    "understood" | "partial" | "misconception" — deliberately not an enum; see design doc §3.1),
    read by record_assessment to derive the UNDERSTANDS-edge confidence/status roll-up.
    Misconception detail (design doc §3.3 Option A: anchored only on Assessment, never a
    Concept-to-Concept edge) rides in ``evidence`` as free text under the locked 4-arg tool
    signature (no separate confused_with/misconception_text param exists to populate them)."""

    assessment_id: str
    learner_id: str
    verdict: str
    evidence: str = Field(default="", description="Free text; optionally a teach-back transcript (§3.1)")
    created_at: datetime = Field(default_factory=datetime.now)


# ── Memory Node Models ──


class MemoryNode(BaseModel):
    memory_id: str
    memory_type: str = Field(description="episodic | semantic | procedural | core")
    content: str
    confidence: float = 0.7
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    access_count: int = 0
    embedding: list[float] | None = None


class EntityNode(BaseModel):
    entity_id: str
    entity_type: str = Field(description="person | project | tool | system | concept")
    canonical_name: str
    summary: str = ""
    aliases: list[str] = Field(default_factory=list)
    last_verified: datetime = Field(default_factory=datetime.now)


# ── Graph Delta (LLM structured output) ──


class NodeProposal(BaseModel):
    """Proposed new node from reasoning pass.

    ``label`` is open at the schema layer (any ``NodeLabel``), but only the
    knowledge labels a reasoning-pass delta may actually create are honored — see
    ``store._LLM_PROPOSABLE_LABELS`` (F8(a)). Same unguarded-boundary FAMILY as
    ``store._PROTECTED_ON_MATCH`` (the #20 incident, commit 85575dc/W-A) — there an
    LLM-controlled property KEY reached MERGE/SET with no allowlist; here an
    LLM-controlled LABEL reaches MERGE with no allowlist. Verified incident: 151
    live Memory/Session/SkillRun/Entity nodes were 100% LLM-mislabeled EXTRACTION
    knowledge (the reasoner chose a personal-memory label instead of a knowledge
    one) — those labels are stripped from the company export, so real knowledge
    silently vanished from every shipped artifact. A ``NodeProposal`` (or
    ``NodeUpdate``) naming a label outside the allowlist is DROPPED with a warning
    at the store boundary, not here — constraining once at the single write choke
    point covers every producer of this model (structured output, normalize
    fallback, direct callers) without duplicating the rule here.
    """

    proposed_id: str = Field(description="lowercase_snake_case identifier (e.g., 'folded_cascode_ota')")
    label: NodeLabel
    canonical_name: str = Field(
        description="Human-readable name in Title Case (e.g., 'Folded Cascode OTA'). REQUIRED for all nodes."
    )
    description: str = Field(
        default="",
        description="1-3 sentence explanation. What IS this concept/circuit/parameter?",
    )
    domain: str = Field(default="general", description="Domain classification (e.g., 'analog_circuits', 'semiconductor_physics', 'cis_architecture')")
    knowledge_layer: int = Field(
        default=-1,
        description="Knowledge hierarchy layer: 0=math/physics foundation, 1=device physics, 2=analog circuits/EDA, 3=CIS architecture. -1=unclassified (pipeline will assign from domain).",
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="0.9+ for textbook definitions, 0.7-0.9 for derived, 0.5-0.7 for inferred",
    )
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional type-specific properties (e.g., 'symbol', 'units' for Parameter, 'latex' for Equation)",
    )
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    reasoning: str = Field(
        default="",
        description="Why this should be a new node, not merged",
    )


class NodeUpdate(BaseModel):
    """Proposed update to an existing node.

    ``updates`` is deliberately an open dict at the schema layer, but identity and
    provenance fields (``source_id``, ``canonical_name``, merge-key ids, retraction
    state, ``_created_at``) are DROPPED with a warning at the store boundary — see
    ``store._PROTECTED_ON_MATCH`` (#20 incident family). Constraining once at the
    single write choke point covers every producer of this model (structured output,
    normalize fallback, direct callers) without duplicating the rule here.
    """

    existing_node_id: str
    label: NodeLabel
    updates: dict[str, Any]
    # Purely explanatory (why this update) — default "" so a terse-but-compliant frontier model
    # omitting it never crashes a structured-output ingest (same class as NodeProposal.reasoning).
    reasoning: str = ""


class EdgeProposal(BaseModel):
    """Proposed new edge with rationale."""

    source_ref: str = Field(description="proposed_id or existing_node_id of source node")
    target_ref: str = Field(description="proposed_id or existing_node_id of target node")
    relationship_type: RelType
    rationale: str = Field(
        default="",
        description="Specific 1-2 sentence explanation of WHY this connection exists. Must reference the text.",
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="REQUIRED. 0.9+ for explicitly stated, 0.7-0.9 for strongly implied, 0.5-0.7 for inferred",
    )
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class EdgeReinforcement(BaseModel):
    """Confirmation of an existing edge by new evidence."""

    source_ref: str
    target_ref: str
    relationship_type: RelType
    # Supplementary evidence list — default [] so a model omitting it never crashes the ingest
    # (source_ref/target_ref/relationship_type remain the load-bearing structural fields).
    confirming_chunk_ids: list[str] = Field(default_factory=list)
    new_evidence_note: str = ""


class InsightProposal(BaseModel):
    """Cross-concept observation from reasoning."""

    statement: str
    related_concept_ids: list[str] = Field(default_factory=list)
    bridge_type: str = "within-domain"
    confidence: float = 0.5
    source_id: str = ""
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class GraphDelta(BaseModel):
    """Complete graph change proposal from a reasoning pass."""

    new_nodes: list[NodeProposal] = Field(default_factory=list)
    updated_nodes: list[NodeUpdate] = Field(default_factory=list)
    new_edges: list[EdgeProposal] = Field(default_factory=list)
    reinforced_edges: list[EdgeReinforcement] = Field(default_factory=list)
    insights: list[InsightProposal] = Field(default_factory=list)
