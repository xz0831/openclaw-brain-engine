"""Two-pass knowledge extractor — entity extraction then relation extraction.

Pass 1: Extract entities (concepts, equations, parameters) from a chunk.
Pass 2: Given the resolved entities, extract relationships between them.

This two-pass approach reduces cognitive load on the LLM and produces
more consistent results (KGGen NeurIPS 2025: 66% vs 48% single-pass).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

from openclaw_brain.knowledge.extraction.models import (
    ConceptMention,
    EquationMention,
    ExtractionResult,
    ParameterMention,
    RawEdge,
)
from openclaw_brain.knowledge.ontology import ONTOLOGY_SNIPPET

logger = logging.getLogger(__name__)

# Pass 2 (relation extraction) failure tracking. extract_from_chunk() has no production caller
# today (pipeline.py reimplements this two-pass flow inline as _resilient_extract, which already
# logs+its own caller sees the failure) — this module-level counter exists so a future caller
# (or a resurrection of this function as the canonical extraction path) can aggregate how many
# chunks silently lost their edges to a Pass 2 exception, without requiring an ExtractionResult
# schema change. Scoped per-process; reset_pass2_failure_tracking() is for test isolation.
_pass2_failed_chunk_ids: set[str] = set()


def pass2_failure_count() -> int:
    """Number of distinct chunks whose Pass 2 (relation extraction) failed inside
    extract_from_chunk() this process — a flag/count callers can aggregate instead of scraping
    logs for the warning extract_from_chunk emits on failure."""
    return len(_pass2_failed_chunk_ids)


def reset_pass2_failure_tracking() -> None:
    """Clear Pass 2 failure tracking (test isolation; not called in production)."""
    _pass2_failed_chunk_ids.clear()


# ── Pass 1: Entity extraction ──

_SCOPE_FIDELITY_RULE = """

Scope fidelity rule:
- When a claim is directional or quantitative (increases/decreases/proportional/×N), preserve the exact scope stated in the text — which quantity, which noise type, which regime/condition. Never widen 'X-type noise' to 'noise' or 'total noise'. If the text scopes a claim, your output must carry that scope.

## EXHAUSTIVENESS (critical)
BE EXHAUSTIVE. Extract EVERY distinct technical concept, circuit block, device, phenomenon, technique,
noise source, equation, and parameter that is stated OR clearly implied. A dense technical paragraph
typically yields 8-20 concepts. Do NOT summarize, merge distinct ideas, or stop early — list each as
its own entity. Do NOT extract figure/table references (e.g. 'Fig. 3', 'Table II') as concepts.
Missing a real concept is worse than including a borderline one (measured to lift recall to frontier
parity without precision loss — experiments/CANONICALIZATION_PROMOTION.md model-selection study)."""


class _EntityExtractionOutput(BaseModel):
    """Schema for Pass 1 — entities only (no relationships)."""

    concepts: list[ConceptMention] = Field(default_factory=list)
    equations: list[EquationMention] = Field(default_factory=list)
    parameters: list[ParameterMention] = Field(default_factory=list)


# Keep for backward compat — pipeline.py imports this for resilient extraction
class _LLMExtractionOutput(BaseModel):
    """Full extraction output schema (used by single-pass fallback)."""

    concepts: list[ConceptMention] = Field(default_factory=list)
    equations: list[EquationMention] = Field(default_factory=list)
    parameters: list[ParameterMention] = Field(default_factory=list)
    relationships: list[RawEdge] = Field(default_factory=list)


_ENTITY_SYSTEM_PROMPT = """You are a knowledge extraction engine for semiconductor and analog circuit design.

Given a text chunk from a technical document, extract ALL entities:

## 1. Concepts
Named technical concepts (e.g., "Common-Source Amplifier", "Transconductance", "Miller Effect").
RULES:
- **name**: Human-readable, Title Case (e.g., "Threshold Voltage", NOT "threshold_voltage")
- **description**: REQUIRED. At least one full sentence explaining what the concept IS.
  Bad: "" or "FD-SOI". Good: "Fully-depleted silicon-on-insulator technology that provides reduced parasitic capacitance and improved electrostatic control."
- **domain**: Be specific (e.g., "analog_circuits", "semiconductor_physics", "neuromorphic", "device_fabrication")
- **granularity**: "atomic" for single concepts, "composite" for compound topics

## 2. Equations
Extract ALL mathematical equations, formulas, and quantitative relationships.
RULES:
- Write in LaTeX format (e.g., "I_D = \\frac{1}{2} \\mu_n C_{ox} \\frac{W}{L} (V_{GS} - V_{th})^2")
- Classify as: definition, law, derived, or approximation
- List ALL variables with their meanings
- Include assumptions (e.g., "saturation region", "long-channel approximation")
- Even approximate or empirical formulas count — do NOT skip them

## 3. Parameters
Extract ALL physical parameters, design specifications, and measured values.
RULES:
- **symbol**: The mathematical symbol (e.g., "V_th", "g_m", "I_D")
- **name**: Full descriptive name (e.g., "Threshold Voltage", "Transconductance")
- **units**: SI or common units (e.g., "V", "mA/V", "fF")
- **typical_range**: If mentioned, give range (e.g., "0.3-0.5 V", ">10 mA/V")
- Extract numeric values mentioned in the text (e.g., "28 nm technology node", "1.2 V supply")

Focus ONLY on entities. Do NOT extract relationships yet — that will be done in a separate step.
Be precise. Only extract what is explicitly stated or directly implied.
Aim for COMPLETENESS — extract every concept, equation, and parameter present in the text.

""" + ONTOLOGY_SNIPPET
_ENTITY_SYSTEM_PROMPT += _SCOPE_FIDELITY_RULE


# ── Pass 2: Relation extraction ──


def _recover_relationships_list(value: Any) -> Any:
    """Recover a `relationships` value that arrived double-encoded as a JSON string instead
    of a native list — e.g. '[{"source_name": ...}]' or a whole-object wrap like
    '{"relationships": [...]}' (observed from claude-sonnet-5). Returns the value unchanged
    (list, or anything else) when it isn't a recoverable string, so a genuinely malformed
    payload still fails normal field validation rather than being silently swallowed here."""
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return value
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and isinstance(parsed.get("relationships"), list):
        return parsed["relationships"]
    return value


class _RelationExtractionOutput(BaseModel):
    """Schema for Pass 2 — relationships between known entities."""

    relationships: list[RawEdge] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _tolerate_double_encoding(cls, data: Any) -> Any:
        """Some frontier models (claude-sonnet-5) double-encode the structured output: the
        `relationships` field comes back as a JSON string (sometimes itself wrapping another
        `relationships` key), or the whole payload comes back as a JSON string, instead of a
        native list. Recover it before field validation runs so a shape quirk doesn't force
        pipeline.py's Pass 2 to give up on relations entirely. Only reachable inputs are
        touched — anything that can't be parsed/recovered is passed through unchanged and
        still fails normal pydantic validation exactly as before this fix."""
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError, ValueError):
                return data
        if isinstance(data, dict) and "relationships" in data:
            data = dict(data)
            data["relationships"] = _recover_relationships_list(data["relationships"])
        return data


_RELATION_SYSTEM_PROMPT = """You are a knowledge graph relation extractor for semiconductor and analog circuit design.

You are given:
1. A text chunk from a technical document.
2. A list of entities (concepts, equations, parameters) already extracted from this chunk.

Your task: extract ALL relationships between the listed entities that are supported by the text.

RULES:
- **source_name** and **target_name** MUST match entity names from the provided list exactly.
- Include a **rationale** explaining WHY the relationship exists (1-2 sentences, specific to the text).
- **Relationship types**: USES_EQUATION, HAS_PARAMETER, DEPENDS_ON, ASSUMES, DERIVED_FROM, APPROXIMATION_OF, TOPOLOGY_VARIANT, TRADES_OFF, CONFIRMS, CONTRADICTS, REFINES, DESIGN_RULE
- Be thorough — if entity A and entity B appear in the same context, consider whether a relationship exists.
- Assign **confidence**: 0.9+ for explicitly stated relationships, 0.7-0.9 for strongly implied, 0.5-0.7 for inferred.
- Do NOT invent relationships not supported by the text.
- Do NOT create relationships involving entities not in the provided list.

""" + ONTOLOGY_SNIPPET + _SCOPE_FIDELITY_RULE


# ── Combined system prompt for single-pass fallback ──

_SYSTEM_PROMPT = _ENTITY_SYSTEM_PROMPT.replace(
    "Focus ONLY on entities. Do NOT extract relationships yet — that will be done in a separate step.\n",
    ""
) + """

## 4. Relationships
Connections between concepts found in THIS chunk.
RULES:
- Include a rationale explaining WHY the relationship exists
- Relationship types: USES_EQUATION, HAS_PARAMETER, DEPENDS_ON, ASSUMES, DERIVED_FROM, APPROXIMATION_OF, TOPOLOGY_VARIANT, TRADES_OFF, CONFIRMS, CONTRADICTS, REFINES, DESIGN_RULE
- Be thorough — if concept A mentions concept B in context, that IS a relationship"""


def _format_entity_list(entities: _EntityExtractionOutput) -> str:
    """Format extracted entities as a readable list for the relation extraction pass."""
    parts = []

    if entities.concepts:
        parts.append("### Concepts")
        for c in entities.concepts:
            parts.append(f"- **{c.name}**: {c.description} (domain: {c.domain})")

    if entities.equations:
        parts.append("\n### Equations")
        for eq in entities.equations:
            parts.append(f"- ${eq.latex}$ ({eq.equation_type})")

    if entities.parameters:
        parts.append("\n### Parameters")
        for p in entities.parameters:
            parts.append(f"- **{p.name}** ({p.symbol}): {p.units}")

    return "\n".join(parts) if parts else "(no entities extracted)"


async def extract_from_chunk(
    chunk_text: str,
    chunk_id: str,
    llm: BaseChatModel,
) -> ExtractionResult:
    """Extract structured knowledge using two-pass entity-then-relation approach.

    Pass 1: Extract entities (concepts, equations, parameters).
    Pass 2: Extract relationships given the resolved entities.

    Falls back to single-pass if Pass 2 fails.
    """
    # ── Pass 1: Entity extraction ──
    entity_llm = llm.with_structured_output(_EntityExtractionOutput)

    entities: _EntityExtractionOutput = await entity_llm.ainvoke([
        SystemMessage(content=_ENTITY_SYSTEM_PROMPT),
        HumanMessage(content=(
            "Extract ALL concepts, equations, and parameters from this text.\n"
            "Remember: every concept MUST have a descriptive name in Title Case and a non-empty description.\n"
            "Do NOT skip equations or parameters — extract every one you find.\n\n"
            f"{chunk_text}"
        )),
    ])

    # ── Pass 2: Relation extraction ──
    entity_list_str = _format_entity_list(entities)
    raw_edges: list[RawEdge] = []

    # Only run Pass 2 if we have at least 2 entities to relate
    entity_count = len(entities.concepts) + len(entities.equations) + len(entities.parameters)
    if entity_count >= 2:
        try:
            relation_llm = llm.with_structured_output(_RelationExtractionOutput)
            relations: _RelationExtractionOutput = await relation_llm.ainvoke([
                SystemMessage(content=_RELATION_SYSTEM_PROMPT),
                HumanMessage(content=(
                    f"## Source Text\n{chunk_text}\n\n"
                    f"## Extracted Entities\n{entity_list_str}\n\n"
                    "Extract ALL relationships between these entities that are supported by the text above.\n"
                    "source_name and target_name MUST match entity names from the list exactly."
                )),
            ])
            raw_edges = relations.relationships
        except Exception as e:
            # Pass 2 failing must not lose Pass 1's entities (still returned below) — but
            # silently dropping ALL relationships with zero signal (the previous bare
            # `except Exception: pass`) makes a transient LLM error (timeout/429/auth/
            # validation) indistinguishable from "the text genuinely has no relationships".
            # Log with the chunk id + exception type/message, and count it so a caller can
            # aggregate without scraping logs (see pass2_failure_count()).
            logger.warning(
                "Chunk %s: Pass 2 (relation extraction) failed (%s: %s) — returning Pass 1 "
                "entities only, zero relationships for this chunk",
                chunk_id, type(e).__name__, e,
            )
            _pass2_failed_chunk_ids.add(chunk_id)

    return ExtractionResult(
        chunk_id=chunk_id,
        concepts=entities.concepts,
        equations=entities.equations,
        parameters=entities.parameters,
        raw_edges=raw_edges,
    )
