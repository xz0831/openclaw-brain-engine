"""Paper-level summarizer — generates a high-level summary after all chunks are processed.

Produces a structured summary of the paper's key contributions, methodology,
limitations, and connections to existing knowledge. Updates the Source node
with this enriched information.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from openclaw_brain.knowledge.graph.schema import NodeLabel
from openclaw_brain.knowledge.graph.store import GraphStore

logger = logging.getLogger(__name__)


class PaperSummary(BaseModel):
    """Structured summary of a paper's contributions."""

    title: str = Field(description="Full paper title")
    one_line: str = Field(description="One-sentence summary of the paper's main contribution")
    key_contributions: list[str] = Field(
        description="2-5 bullet points of specific technical contributions",
    )
    methodology: str = Field(
        default="",
        description="Brief description of the approach/method (1-3 sentences)",
    )
    key_results: list[str] = Field(
        default_factory=list,
        description="Specific quantitative results mentioned (e.g., '3.2 mW power', '45° phase margin')",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Limitations or constraints acknowledged or implied",
    )
    technology: str = Field(
        default="",
        description="Technology node / process used (e.g., '28nm FD-SOI')",
    )
    application_domain: str = Field(
        default="",
        description="Target application (e.g., 'neuromorphic computing', 'sensor interface')",
    )
    design_insights: list[str] = Field(
        default_factory=list,
        description="Practical design insights that would help a circuit designer. These are the most valuable outputs.",
    )


_SUMMARY_SYSTEM_PROMPT = """You are an expert semiconductor engineering researcher.
Given a knowledge graph extracted from a paper (concepts, circuits, parameters, equations, and their relationships),
produce a structured paper summary that would be useful for a circuit design engineer.

Focus on:
1. What SPECIFIC technical contribution does this paper make?
2. What CONCRETE design insights can a practitioner take away?
3. What are the QUANTITATIVE results (numbers, not vague claims)?
4. What LIMITATIONS should a designer be aware of?

Be precise and technical. Avoid generic academic language."""


async def summarize_paper(
    source_id: str,
    graph: GraphStore,
    llm: BaseChatModel,
) -> PaperSummary | None:
    """Generate a paper-level summary from the extracted knowledge graph.

    Queries the graph for all nodes linked to this source, then asks an LLM
    to synthesize a high-level summary.

    Args:
        source_id: The source node ID.
        graph: GraphStore instance.
        llm: LLM for summarization.

    Returns:
        PaperSummary or None if generation fails.
    """
    # Gather all knowledge extracted from this paper
    context = await _gather_paper_context(source_id, graph)

    if not context:
        logger.warning("No context found for source %s, skipping summary", source_id)
        return None

    try:
        structured_llm = llm.with_structured_output(PaperSummary)
        summary: PaperSummary = await structured_llm.ainvoke([
            SystemMessage(content=_SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=f"Summarize the following knowledge extracted from a paper:\n\n{context}"),
        ])
        return summary
    except Exception as e:
        logger.warning("Paper summary generation failed: %s", e)
        return None


async def apply_summary_to_source(
    source_id: str,
    summary: PaperSummary,
    graph: GraphStore,
) -> None:
    """Update the Source node with the paper summary."""
    await graph.merge_node(
        label=NodeLabel.SOURCE,
        id_field="source_id",
        id_value=source_id,
        properties={
            "source_id": source_id,
            # Do NOT overwrite the real title (set from the filename by _register_source) with the
            # LLM's synthesized title — that is the title-hallucination bug that mislabeled the graph
            # (Razavi/Stewart content got generic "Digital Pixel Sensors" titles). Keep it separate.
            "synthesized_title": summary.title,
            "one_line_summary": summary.one_line,
            "key_contributions": summary.key_contributions,
            "methodology": summary.methodology,
            "key_results": summary.key_results,
            "limitations": summary.limitations,
            "technology": summary.technology,
            "application_domain": summary.application_domain,
            "design_insights": summary.design_insights,
        },
    )
    logger.info("Updated Source node %s with paper summary", source_id)


async def _gather_paper_context(source_id: str, graph: GraphStore) -> str:
    """Gather all extracted knowledge for a paper from the graph."""
    async with await graph._session() as session:
        # Knowledge nodes belonging to THIS source. (Bug fix: the previous query collected the
        # source's chunk_ids but never used them to filter `n`, so it returned the ENTIRE graph's
        # nodes — every paper got summarized from the same global sample, cross-contaminating all
        # Source summaries. All knowledge nodes carry source_id, so scope on that.)
        result = await session.run("""
            MATCH (n)
            WHERE n.source_id = $source_id
              AND NOT any(l IN labels(n) WHERE l IN ['Source', 'SourceChunk', 'Memory', 'Entity', 'Session', 'SkillRun'])
            WITH n, labels(n)[0] AS label
            RETURN label, collect(properties(n)) AS nodes
            ORDER BY label
        """, {"source_id": source_id})

        parts = []
        async for record in result:
            label = record["label"]
            nodes = record["nodes"]
            parts.append(f"\n## {label} ({len(nodes)} nodes)")
            for node in nodes[:20]:  # Limit per type to fit context
                name = node.get("canonical_name", "") or node.get("name", "") or node.get("concept_id", "?")
                desc = node.get("description", "") or node.get("statement", "")
                conf = node.get("confidence", "?")
                line = f"- **{name}** (conf={conf}): {desc[:150]}"
                # Add type-specific details
                if node.get("symbol"):
                    line += f" [{node['symbol']}]"
                if node.get("units"):
                    line += f" ({node['units']})"
                if node.get("typical_range"):
                    line += f" range: {node['typical_range']}"
                if node.get("canonical_latex"):
                    line += f" $${node['canonical_latex']}$$"
                parts.append(line)

        # Relationships among THIS source's nodes (same scoping fix — was global).
        result = await session.run("""
            MATCH (a)-[r]->(b)
            WHERE a.source_id = $source_id AND b.source_id = $source_id
              AND NOT any(l IN labels(a) WHERE l IN ['Source', 'SourceChunk', 'Memory', 'Entity', 'Session', 'SkillRun'])
              AND NOT any(l IN labels(b) WHERE l IN ['Source', 'SourceChunk', 'Memory', 'Entity', 'Session', 'SkillRun'])
            WITH a, b, r, labels(a)[0] AS at, labels(b)[0] AS bt
            RETURN
                coalesce(a.canonical_name, a.name, a.concept_id, '?') AS a_name,
                type(r) AS rel,
                r.rationale AS rationale,
                coalesce(b.canonical_name, b.name, b.concept_id, '?') AS b_name
            LIMIT 50
        """, {"source_id": source_id})

        parts.append("\n## Key Relationships")
        async for record in result:
            line = f"- {record['a_name']} --[{record['rel']}]--> {record['b_name']}"
            if record["rationale"]:
                line += f": {record['rationale'][:100]}"
            parts.append(line)

    return "\n".join(parts) if parts else ""
