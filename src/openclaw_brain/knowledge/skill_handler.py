"""Skill handler for PDF knowledge ingestion.

This is the entry point that M4's SkillExecutor calls.
Registers as a skill and wraps the KnowledgePipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openclaw_brain.config import BrainConfig
from openclaw_brain.knowledge.evidence import EvidenceVault
from openclaw_brain.knowledge.extraction.models import IngestResult, PipelineProgress
from openclaw_brain.knowledge.graph.store import GraphStore
from openclaw_brain.knowledge.pipeline import KnowledgePipeline
from openclaw_brain.llm.provider import LLMProvider
from openclaw_brain.skills.registry import SkillCategory, SkillDefinition, SkillRegistry


def register_pdf_ingest_skill(
    registry: SkillRegistry,
    graph: GraphStore,
    config: BrainConfig,
    llm_provider: LLMProvider,
    vault: EvidenceVault | None = None,
) -> None:
    """Register the pdf_ingest skill in the skill registry."""

    pipeline = KnowledgePipeline(graph, config, llm_provider, vault=vault)

    async def handler(params: dict[str, Any]) -> dict[str, Any]:
        return await pdf_ingest_handler(pipeline, params)

    definition = SkillDefinition(
        name="pdf_ingest",
        description="Extract knowledge from a PDF and integrate into the knowledge graph",
        category=SkillCategory.KNOWLEDGE,
        trigger_keywords=[
            "pdf", "extract", "ingest", "document", "지식", "변환",
            "정리", "연결", "분석", "읽어", "처리",
        ],
        trigger_patterns=[
            r"\.pdf",
            r"extract\s+from",
            r"process\s+(?:this|the)\s+(?:pdf|document)",
            r"(?:이|이거|이것)\s*(?:정리|분석|처리|변환)",
            r"지식.*(?:연결|변환|추가)",
        ],
        input_schema={
            "file_path": "str — path to the PDF file",
            "instruction": "str — natural language instruction (optional)",
            "extraction_model": "str — model name for extraction (optional)",
            "reasoning_model": "str — model name for reasoning (optional)",
            "chunk_concurrency": "int — max chunks processed concurrently (optional, default 1=sequential)",
            "reprocess": "bool — re-run every chunk even if checkpointed done (optional, multi-pass refine)",
        },
        output_schema={
            "success": "bool",
            "summary": "str — human-readable summary",
            "new_nodes": "int",
            "new_edges": "int",
            "reinforced_edges": "int",
            "insights": "int",
            "errors": "list[str]",
        },
        confidence=0.7,
    )

    registry.register(definition, handler)


async def pdf_ingest_handler(
    pipeline: KnowledgePipeline,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Handle a pdf_ingest skill invocation.

    Params:
        file_path: Path to the PDF file (required).
        instruction: Natural language instruction (optional, for future use).
        extraction_model: Model name override for extraction stage.
        reasoning_model: Model name override for reasoning stage.

    Returns:
        Dict with success status, summary, and change counts.
    """
    file_path = params.get("file_path", "")
    if not file_path:
        return {"success": False, "error": "file_path is required"}

    path = Path(file_path)
    if not path.exists():
        return {"success": False, "error": f"File not found: {file_path}"}

    if not path.suffix.lower() == ".pdf":
        return {"success": False, "error": f"Not a PDF file: {file_path}"}

    # Collect progress messages
    progress_log: list[str] = []

    async def on_progress(p: PipelineProgress) -> None:
        progress_log.append(p.message)

    result: IngestResult = await pipeline.ingest(
        pdf_path=path,
        extraction_model=params.get("extraction_model"),
        reasoning_model=params.get("reasoning_model"),
        on_progress=on_progress,
        chunk_concurrency=int(params.get("chunk_concurrency", 1) or 1),
        reprocess=bool(params.get("reprocess", False)),
    )

    return {
        "success": result.success,
        "summary": result.summary(),
        "source_id": result.source_id,
        "title": result.title,
        "total_chunks": result.total_chunks,
        "new_nodes": result.new_nodes,
        "updated_nodes": result.updated_nodes,
        "new_edges": result.new_edges,
        "reinforced_edges": result.reinforced_edges,
        "insights": result.insights,
        "errors": result.errors,
        "empty_reason": result.empty_reason,
        "progress_log": progress_log,
    }


def register_html_ingest_skill(
    registry: SkillRegistry,
    graph: GraphStore,
    config: BrainConfig,
    llm_provider: LLMProvider,
    vault: EvidenceVault | None = None,
) -> None:
    """Register the html_ingest skill (Rick's lecture-capture HTML decks) — mirrors
    register_pdf_ingest_skill exactly; see knowledge/extraction/html_parser.py for the input
    format and knowledge/pipeline.py::ingest_html for the shared chunk->...->summarize seam."""

    pipeline = KnowledgePipeline(graph, config, llm_provider, vault=vault)

    async def handler(params: dict[str, Any]) -> dict[str, Any]:
        return await html_ingest_handler(pipeline, params)

    definition = SkillDefinition(
        name="html_ingest",
        description="Extract knowledge from a Rick lecture-capture HTML deck and integrate into the knowledge graph",
        category=SkillCategory.KNOWLEDGE,
        trigger_keywords=[
            "html", "lecture", "정리본", "세션노트", "강의", "지식", "변환", "정리", "연결",
        ],
        trigger_patterns=[
            r"\.html?$",
            r"강의.*(?:정리|분석|처리|변환)",
            r"지식.*(?:연결|변환|추가)",
        ],
        input_schema={
            "file_path": "str — path to the lecture HTML file",
            "instruction": "str — natural language instruction (optional)",
            "extraction_model": "str — model name for extraction (optional)",
            "reasoning_model": "str — model name for reasoning (optional)",
            "chunk_concurrency": "int — max chunks processed concurrently (optional, default 1=sequential)",
            "reprocess": "bool — re-run every chunk even if checkpointed done (optional, multi-pass refine)",
            "figures_only": "bool — VLM-analyze slide IMAGES instead of speech transcript (optional, default False)",
        },
        output_schema={
            "success": "bool",
            "summary": "str — human-readable summary",
            "new_nodes": "int",
            "new_edges": "int",
            "reinforced_edges": "int",
            "insights": "int",
            "errors": "list[str]",
        },
        confidence=0.7,
    )

    registry.register(definition, handler)


async def html_ingest_handler(
    pipeline: KnowledgePipeline,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Handle an html_ingest skill invocation. Mirrors pdf_ingest_handler exactly, modulo the
    file-extension guard (.html/.htm) and calling pipeline.ingest_html instead of pipeline.ingest.

    Params: same as pdf_ingest_handler's (file_path required; instruction/extraction_model/
    reasoning_model/chunk_concurrency/reprocess/figures_only optional).

    Returns:
        Dict with success status, summary, and change counts — same shape as pdf_ingest_handler's.
    """
    file_path = params.get("file_path", "")
    if not file_path:
        return {"success": False, "error": "file_path is required"}

    path = Path(file_path)
    if not path.exists():
        return {"success": False, "error": f"File not found: {file_path}"}

    if path.suffix.lower() not in (".html", ".htm"):
        return {"success": False, "error": f"Not an HTML file: {file_path}"}

    progress_log: list[str] = []

    async def on_progress(p: PipelineProgress) -> None:
        progress_log.append(p.message)

    result: IngestResult = await pipeline.ingest_html(
        html_path=path,
        extraction_model=params.get("extraction_model"),
        reasoning_model=params.get("reasoning_model"),
        on_progress=on_progress,
        chunk_concurrency=int(params.get("chunk_concurrency", 1) or 1),
        reprocess=bool(params.get("reprocess", False)),
        figures_only=bool(params.get("figures_only", False)),
    )

    return {
        "success": result.success,
        "summary": result.summary(),
        "source_id": result.source_id,
        "title": result.title,
        "total_chunks": result.total_chunks,
        "new_nodes": result.new_nodes,
        "updated_nodes": result.updated_nodes,
        "new_edges": result.new_edges,
        "reinforced_edges": result.reinforced_edges,
        "insights": result.insights,
        "errors": result.errors,
        "empty_reason": result.empty_reason,
        "progress_log": progress_log,
    }
