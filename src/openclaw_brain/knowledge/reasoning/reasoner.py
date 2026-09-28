"""Graph-context-aware reasoner — produces GraphDelta from extraction + graph context.

This is the core intelligence stage. It takes extracted concepts,
their matches against the existing graph, and the graph neighborhood,
then uses an LLM to produce a structured GraphDelta.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from openclaw_brain.config import BrainConfig
from openclaw_brain.knowledge.extraction.models import (
    ExtractionResult,
    MatchResult,
)
from openclaw_brain.knowledge.graph.schema import GraphDelta, NodeLabel
from openclaw_brain.knowledge.graph.store import GraphStore
from openclaw_brain.knowledge.reasoning.normalize import (
    extract_json,
    normalize_graph_delta,
)

logger = logging.getLogger(__name__)

_SCOPE_FIDELITY_RULE = """

Scope fidelity rule:
- When a claim is directional or quantitative (increases/decreases/proportional/×N), preserve the exact scope stated in the text — which quantity, which noise type, which regime/condition. Never widen 'X-type noise' to 'noise' or 'total noise'. If the text scopes a claim, your output must carry that scope."""


_SYSTEM_PROMPT = """You are a knowledge graph reasoning engine for semiconductor and analog circuit design.
Your graph is an AI agent's brain — it must understand circuits the way an experienced designer does.

You receive extracted entities, match results, and graph context. Produce a GraphDelta.

## PHILOSOPHY: Capture DESIGN REASONING, not just facts.

A fact: "Cascode has high output impedance."
Design reasoning: "Cascode SOLVES_PROBLEM low output impedance of a single CS stage by stacking a common-gate transistor, multiplying Rout by gm·ro. But it INTRODUCES_PROBLEM reduced headroom (one extra VDS,sat). A designer should choose cascode WHEN gain is insufficient but supply voltage allows the headroom cost."

The second version is what makes a knowledge graph useful for an engineering mentor.

## CRITICAL REQUIREMENTS for every new_node:

- **canonical_name**: REQUIRED. Human-readable Title Case. "Folded Cascode OTA", NOT "folded_cascode_ota".
- **description**: REQUIRED. 1-3 sentences explaining what this IS and WHY it matters for a designer.
- **confidence**: REQUIRED. NEVER leave at 0.
  - 0.9-1.0: Textbook definitions, explicitly stated facts
  - 0.7-0.9: Well-supported claims, standard derivations
  - 0.5-0.7: Inferences, implications not directly stated
- **domain**: Specific classification (analog_circuits, semiconductor_physics, neuromorphic, etc.)

## Node types:

### CircuitTopology — the most important node type
For EVERY circuit topology, capture the designer's perspective:
- **canonical_name**: e.g., "Folded Cascode OTA"
- **description**: What it does AND why it exists
- In **properties**:
  - **function**: amplifier | bias | feedback | comparator | compensation | ...
  - **key_nodes**: Important transistors/nets (M1, M2, Vbias, ...)
  - **sub_blocks**: Constituent sub-circuits (e.g., ["Differential Pair", "Cascode Load", "Tail Current Source"])
  - **solves**: What problem does this topology address? (e.g., "Low gain of single-stage amplifier")
  - **introduces**: What new problem/limitation? (e.g., "Reduced output swing by VDS,sat")
  - **when_to_use**: Under what conditions? (e.g., "When single-stage gain is insufficient and supply voltage > 4·VDS,sat + 2·Vov")

### Equation — connect math to physical circuit
- **canonical_name**: Descriptive (e.g., "CS Amplifier Voltage Gain")
- In **properties**:
  - **latex**: The equation
  - **physical_meaning**: What does this equation represent physically?
  - **component_mapping**: How variables map to circuit components (e.g., ["gm → input transistor M1", "RD → drain load resistor"])
  - **assumptions**: Under what conditions is this valid?

### Parameter — include actual numbers
- In **properties**: symbol, units, typical_range WITH numeric values from the text

### Concept, Principle — standard fields

## Edge relationships — PRIORITIZE DESIGN REASONING:

The most valuable edges for an engineering mentor:

1. **SOLVES_PROBLEM**: Topology/technique → Problem it solves. Rationale: the MECHANISM.
2. **INTRODUCES_PROBLEM**: Topology/technique → Limitation it creates. Rationale: WHY this is a consequence.
3. **EVOLVES_TO**: Parent topology → Child topology. Rationale: WHAT limitation of the parent motivated the evolution.
   Example: "Common-Source EVOLVES_TO Cascode — because CS has Rout ≈ ro which limits gain to gm·ro ≈ 20-50 V/V"
4. **SUB_BLOCK**: Composite circuit → Sub-circuit. Rationale: WHAT role the sub-block plays.
   Example: "Two-Stage Miller OTA SUB_BLOCK Differential Pair — serves as the input transconductance stage"
5. **TRADES_OFF**: Parameter A ↔ Parameter B. Rationale: name BOTH sides AND why they're coupled.
6. **MODELS_BEHAVIOR**: Equation → Circuit. Rationale: what aspect the equation captures.
7. **VARIABLE_MAPS_TO**: Variable → Component. Rationale: the physical correspondence.
8. **COMPENSATED_BY**: Circuit → Compensation. Rationale: what instability it fixes.
9. **DESIGN_RULE**: Practical guideline with specific conditions.

Also use: USES_EQUATION, HAS_PARAMETER, DEPENDS_ON, ASSUMES, DERIVED_FROM when appropriate.

Every edge MUST have:
- **rationale**: Specific, referencing the text. Not "they are related."
- **confidence**: Per the scale above. NEVER 0.

## Insights — design-level observations:

Good: "The pseudo-cascode technique trades area for leakage mitigation — in 28nm FD-SOI where pico-Ampere leakage currents dominate bias, this technique enables time constants >100ms without impractically large capacitors."
Bad: "These concepts are related to neuromorphic computing."

## KNOWLEDGE LAYER HIERARCHY

The graph spans 4 layers. Set **knowledge_layer** on every new node:
- **0** (L0): Math/physics foundations — calculus, electromagnetism, thermodynamics, quantum mechanics
- **1** (L1): Device physics — semiconductor physics, transistor models (MOSFET, BJT), fabrication
- **2** (L2): Analog circuits & EDA — circuit topologies, OTAs, comparators, ADCs, simulation, layout
- **3** (L3): CIS architecture — pixel design, readout chains, ADC columns, image quality metrics

**Cross-layer edges are the most valuable** — they show HOW lower-layer physics enables higher-layer design:
- BRIDGES_TO: e.g., "Subthreshold Slope (L1) BRIDGES_TO Pixel Source Follower Noise (L3) — weak inversion gm determines the thermal noise floor read out from each pixel"
- DEPENDS_ON: e.g., "Correlated Double Sampling (L3) DEPENDS_ON kTC Noise (L1) — CDS was invented specifically to cancel the kT/C thermal noise set by the reset transistor"
When you see concepts from different layers in the same chunk, propose at least one cross-layer edge.

## Rules
- Resolve ambiguous matches. Prefer merging over duplicates.
- For each Parameter, include actual numeric values from the text.
- Think like a circuit design textbook author: explain the WHY, not just the WHAT.

OUTPUT FORMAT: Return pure JSON only. Never wrap in markdown code blocks or backticks.""" + _SCOPE_FIDELITY_RULE


class GraphReasoner:
    """Produces GraphDelta by reasoning over extractions + graph context."""

    def __init__(self, graph: GraphStore, config: BrainConfig):
        self._graph = graph
        self._config = config

    async def reason(
        self,
        extraction: ExtractionResult,
        match_result: MatchResult,
        chunk_text: str,
        llm: BaseChatModel,
    ) -> GraphDelta:
        """Reason over extraction + graph context to produce a GraphDelta.

        Args:
            extraction: Raw extraction from the chunk.
            match_result: Concept matching results.
            chunk_text: Original text of the chunk.
            llm: LLM to use for reasoning.

        Returns:
            A GraphDelta with proposed changes to the knowledge graph.
        """
        # Gather graph context for matched concepts
        context = await self._gather_context(match_result)

        # Build the prompt
        prompt = self._build_prompt(extraction, match_result, context, chunk_text)

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        # Try structured output first (works for Anthropic, OpenAI, etc.)
        try:
            structured_llm = llm.with_structured_output(GraphDelta)
            delta: GraphDelta = await structured_llm.ainvoke(messages)
            return delta
        except Exception as e:
            logger.info("Structured output failed (%s), falling back to raw + normalize", e)

        # Fallback: raw invocation → JSON extraction → normalization → validation
        response = await llm.ainvoke(messages)
        raw_text = response.content if hasattr(response, "content") else str(response)
        raw_json = extract_json(raw_text)
        normalized = normalize_graph_delta(raw_json)
        delta = GraphDelta(**normalized)
        return delta

    async def _gather_context(self, match_result: MatchResult) -> str:
        """Fetch graph neighborhood for matched concepts."""
        context_parts: list[str] = []
        seen_ids: set[str] = set()

        for m in match_result.matched:
            if m.existing_node_id in seen_ids:
                continue
            seen_ids.add(m.existing_node_id)

            _label = NodeLabel(m.node_label) if m.node_label in NodeLabel._value2member_map_ else NodeLabel.CONCEPT
            neighbors = await self._graph.get_neighborhood(
                label=_label,
                id_field=m.node_id_field,
                id_value=m.existing_node_id,
                hops=self._config.reasoning.graph_hop_depth,
                limit=self._config.reasoning.max_graph_neighbors,
            )

            if neighbors:
                context_parts.append(
                    f"\n### Neighborhood of '{m.mention.name}' "
                    f"(existing: {m.existing_node_id}):"
                )
                for n in neighbors:
                    node = n.get("node", {})
                    node_name = node.get("canonical_name", "") or node.get("name", "") or str(node.get("_labels", []))
                    rel = n.get("rel_type", "?")
                    rationale = n.get("rationale", "")
                    conf = n.get("confidence", "?")
                    reinf = n.get("reinforcement_count", 0)
                    line = f"  → {rel} → {node_name} (conf={conf}, reinf={reinf})"
                    if rationale:
                        line += f"\n    Rationale: {rationale}"
                    context_parts.append(line)

        return "\n".join(context_parts) if context_parts else "(No existing graph context)"

    def _build_prompt(
        self,
        extraction: ExtractionResult,
        match_result: MatchResult,
        context: str,
        chunk_text: str,
    ) -> str:
        """Build the reasoning prompt."""
        parts = []

        # 1. Source text (truncated)
        max_text = self._config.reasoning.chunk_token_budget * 4
        text_preview = chunk_text[:max_text]
        parts.append(f"## NEW TEXT (chunk: {extraction.chunk_id})\n{text_preview}")

        # 2. Extracted concepts
        parts.append("\n## EXTRACTED CONCEPTS")
        for c in extraction.concepts:
            parts.append(f"- {c.name}: {c.description} (domain: {c.domain})")

        # 3. Extracted equations
        if extraction.equations:
            parts.append("\n## EXTRACTED EQUATIONS")
            for eq in extraction.equations:
                parts.append(f"- ${eq.latex}$ ({eq.equation_type})")

        # 4. Extracted parameters
        if extraction.parameters:
            parts.append("\n## EXTRACTED PARAMETERS")
            for p in extraction.parameters:
                parts.append(f"- {p.symbol} ({p.name}): {p.units}")

        # 5. Raw relationships
        if extraction.raw_edges:
            parts.append("\n## DETECTED RELATIONSHIPS (within chunk)")
            for e in extraction.raw_edges:
                parts.append(f"- {e.source_name} → {e.relationship} → {e.target_name}: {e.rationale}")

        # 6. Match results
        parts.append("\n## MATCH RESULTS")
        if match_result.matched:
            parts.append("Already in graph:")
            for m in match_result.matched:
                parts.append(f"  ✓ '{m.mention.name}' = {m.existing_node_id} (sim={m.similarity:.2f})")
        # F6: equations/parameters now also flow through the matcher (exact-normalized match,
        # no fuzzy band — see reasoning/matcher.py). Mirrors the concept "Already in graph"
        # listing above so the LLM gets the same steer-away-from-duplication signal for these
        # two labels, which previously had ZERO matching (every mention landed in new_nodes —
        # see knowledge/reasoning/README.md "Known defects" #2). No similarity score printed:
        # unlike concept matching these are a binary exact-match decision, always 1.0.
        if match_result.matched_equations:
            parts.append("Already in graph (equations):")
            for m in match_result.matched_equations:
                parts.append(f"  ✓ '{m.mention.latex}' = {m.existing_node_id}")
        if match_result.matched_parameters:
            parts.append("Already in graph (parameters):")
            for m in match_result.matched_parameters:
                parts.append(f"  ✓ '{m.mention.name}' = {m.existing_node_id}")
        if match_result.new_concepts:
            parts.append("New (not in graph):")
            for c in match_result.new_concepts:
                parts.append(f"  + '{c.name}' — needs new node")
        if match_result.ambiguous:
            parts.append("Ambiguous (you decide):")
            for a in match_result.ambiguous:
                cands = ", ".join(
                    f"{c['canonical_name']} ({c['similarity']:.2f})"
                    for c in a.candidates
                )
                parts.append(f"  ? '{a.mention.name}' — candidates: {cands}")

        # 7. Graph context
        parts.append(f"\n## EXISTING GRAPH CONTEXT\n{context}")

        # 8. Instructions
        matched_ids = (
            [m.existing_node_id for m in match_result.matched]
            # H1 final-hunt fix: matched equations/parameters were shown in the "Already in
            # graph" listing but never entered this instruction, so the LLM was likelier to
            # mint duplicate new_edges (resetting reinforcement history) for them.
            + [e.existing_node_id for e in match_result.matched_equations]
            + [q.existing_node_id for q in match_result.matched_parameters]
        )
        cross_doc_instruction = ""
        if matched_ids:
            ids_str = ", ".join(matched_ids)
            cross_doc_instruction = (
                f"\n- CROSS-DOCUMENT LINKING (critical): For each matched existing concept "
                f"({ids_str}), connect it to something in THIS chunk. FIRST check the EXISTING "
                f"GRAPH CONTEXT above:\n"
                "    - If the SAME relationship (same two concepts, same direction, same "
                "relationship type) is already listed there, this chunk is CONFIRMING it, not "
                "discovering it — put it in reinforced_edges (fields: source_ref, target_ref, "
                "relationship_type, confirming_chunk_ids, new_evidence_note), NOT new_edges. "
                "Using new_edges for an already-existing relationship resets its accumulated "
                "reinforcement_count and evidence history back to just this one chunk — "
                "reinforced_edges is the only field that grows it correctly.\n"
                "    - Only when no such relationship is listed in the graph context, create a "
                "new_edges EdgeProposal that uses the existing_node_id as source_ref or "
                "target_ref. Use the graph context above to pick the correct relationship type "
                "(RELATES_TO, SOLVES_PROBLEM, DEPENDS_ON, etc.)."
            )
        parts.append(
            "\n## INSTRUCTIONS\n"
            "Produce a GraphDelta. Remember:\n"
            "- Resolve all ambiguous matches.\n"
            "- Every new edge must have a specific rationale.\n"
            "- Use evidence_chunk_ids = ['" + extraction.chunk_id + "'] for all proposals.\n"
            "- Prefer merging over creating duplicates."
            + cross_doc_instruction
        )

        return "\n".join(parts)
