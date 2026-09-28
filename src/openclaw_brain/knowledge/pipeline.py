"""Knowledge ingestion pipeline — full PDF→GraphDelta→Neo4j flow.

Orchestrates: chunk → extract → match → reason → commit
Each stage can use a different LLM model (via LLMProvider).
Emits progress callbacks for agent reporting.

Resilience features:
- Checkpoint/resume: saves per-chunk progress to disk
- Retry + fallback: LLM calls use exponential backoff with model fallback chain
- Auth refresh: automatically refreshes OAuth tokens on 401 errors
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Awaitable

from openclaw_brain.auth import create_auth_refresh_callback
from openclaw_brain.config import BrainConfig
from openclaw_brain.egress import effective_egress
from openclaw_brain.journal import ActionJournal
from openclaw_brain.knowledge.checkpoint import PipelineCheckpoint
from openclaw_brain.knowledge.embedding import encode_batch
from openclaw_brain.knowledge.evidence import EvidenceVault
from openclaw_brain.knowledge.extraction.chunker import chunk_structured
from openclaw_brain.knowledge.extraction.figure_analyzer import analyze_all_figures
from openclaw_brain.knowledge.extraction.grounding import strip_figure_markers, verify_grounding
from openclaw_brain.knowledge.extraction.html_parser import parse_lecture_html
from openclaw_brain.knowledge.extraction.mineru_parser import ContentBlock, ParsedDocument, SlideImage, parse_pdf as mineru_parse_pdf
from openclaw_brain.knowledge.extraction.models import (
    ChunkerResult,
    ExtractionResult,
    IngestResult,
    PipelineProgress,
)
from openclaw_brain.knowledge.extraction.slide_analyzer import (
    SlideAnalysis,
    SlideBreakerOpen,
    analyze_all_slides,
    slide_analysis_to_blocks,
)
from openclaw_brain.knowledge.graph.schema import GraphDelta, NodeLabel, NodeUpdate
from openclaw_brain.knowledge.graph.store import GraphStore
from openclaw_brain.knowledge.reasoning.matcher import ConceptMatcher
from openclaw_brain.knowledge.reasoning.reasoner import GraphReasoner
from openclaw_brain.knowledge.reasoning.summarizer import summarize_paper, apply_summary_to_source
from openclaw_brain.llm.cascade import ExtractionQualityCheck
from openclaw_brain.llm.provider import LLMProvider
from openclaw_brain.llm.resilience import (
    invoke_with_resilience, resolve_model_name, FallbackExhaustedError,
)
from openclaw_brain.redaction import redact_secrets

logger = logging.getLogger(__name__)

# Callback type for progress reporting
ProgressCallback = Callable[[PipelineProgress], Awaitable[None]] | None

# Labels that should have embeddings stored (knowledge nodes, not operational nodes)
_EMBEDDABLE_LABELS: frozenset = frozenset({
    NodeLabel.CONCEPT, NodeLabel.EQUATION, NodeLabel.PRINCIPLE,
    NodeLabel.CIRCUIT_TOPOLOGY, NodeLabel.PARAMETER,
})

# Domain → knowledge layer (L0–L3) mapping
_DOMAIN_TO_LAYER: dict[str, int] = {
    # L0: Math / physics foundations
    "mathematics": 0, "classical_physics": 0, "electromagnetism": 0,
    "thermodynamics": 0, "mechanics": 0, "quantum_mechanics": 0,
    "calculus": 0, "electromagnetics": 0, "probability_statistics": 0,
    # L1: Device physics
    "semiconductor_physics": 1, "device_fabrication": 1,
    # L2: Analog circuits / EDA / simulation
    "analog_circuits": 2, "digital_circuits": 2, "mixed_signal": 2,
    "rf_circuits": 2, "power_management": 2, "circuit_simulation": 2,
    "layout_design": 2, "reliability": 2, "signal_processing": 2,
    # L3: CIS / image sensor architecture
    "cis_architecture": 3, "image_sensors": 3, "sensor_interfaces": 3,
    "memory_circuits": 3,
    # Orphan domains
    "neuromorphic": 3, "general": -1,
}


def _normalize_grounding_name(name: str) -> str:
    """Normalize names for conservative mention/proposal matching."""
    return re.sub(r"\s+", " ", name.strip().lower())


def _apply_grounding_scope_penalties(delta, extraction: ExtractionResult) -> None:
    """Persist grounding scope flags and apply their confidence penalty to new nodes."""
    flagged_names = {
        _normalize_grounding_name(concept.name)
        for concept in extraction.concepts
        if "scope_widened" in concept.grounding_flags
    }
    if not flagged_names:
        return

    for proposal in delta.new_nodes:
        if _normalize_grounding_name(proposal.canonical_name) in flagged_names:
            proposal.confidence = round(proposal.confidence * 0.8, 4)
            proposal.properties["grounding_flags"] = ["scope_widened"]


def _inject_insight_evidence(delta: GraphDelta, chunk) -> None:
    """Attach current chunk evidence to insights when the reasoner omitted it."""
    for insight in delta.insights:
        if not insight.source_id:
            insight.source_id = chunk.source_id
        if not insight.evidence_chunk_ids:
            insight.evidence_chunk_ids = [chunk.chunk_id]


class ReasoningDegradedError(Exception):
    """The reasoning stage terminally failed for one chunk (all models/retries exhausted
    on structured output + the raw-JSON normalize fallback). Raised INSTEAD of committing
    an empty delta so that the chunk is NOT checkpointed done — a later invocation retries
    it naturally (the 2026-07 lecture batches showed these are usually transient deepseek
    bad-windows: 16 chunks lost their reasoning yield while match-stage reinforcements
    committed, masking the loss behind non-zero counts — GROUNDING_REALIGNMENT C6)."""

    def __init__(self, chunk_id: str, cause: str):
        self.chunk_id = chunk_id
        self.cause = cause
        super().__init__(f"Chunk {chunk_id}: reasoning terminally failed — {cause}")


class KnowledgePipeline:
    """Orchestrates the full PDF ingestion pipeline.

    Usage:
        pipeline = KnowledgePipeline(graph, config, provider)
        result = await pipeline.ingest(
            pdf_path="ch5.pdf",
            extraction_model="qwen3-vl-8b",    # optional override
            reasoning_model="claude-opus-4-6",   # optional override
            on_progress=my_callback,             # optional progress reports
        )
    """

    def __init__(
        self,
        graph: GraphStore,
        config: BrainConfig,
        llm_provider: LLMProvider,
        vault: EvidenceVault | None = None,
    ):
        self._graph = graph
        self._config = config
        self._provider = llm_provider
        self._vault = vault or EvidenceVault(config.state_path / "evidence")
        self._auth_refresh = create_auth_refresh_callback(llm_provider)
        from openclaw_brain.knowledge.reasoning.verifier import MatchVerifier
        self._matcher = ConceptMatcher(
            graph,
            high_threshold=config.knowledge.reinforcement_threshold,
            low_threshold=config.knowledge.bridge_similarity_low,
            embedding_model=config.embedding.model,
            matcher_config=config.matcher,
            verifier=MatchVerifier(llm_provider, config, auth_refresh=self._auth_refresh),
        )
        self._reasoner = GraphReasoner(graph, config)
        self._journal = ActionJournal(config.state_path)

    async def ingest(
        self,
        pdf_path: str | Path,
        extraction_model: str | None = None,
        reasoning_model: str | None = None,
        figure_analysis_model: str | None = None,
        max_tokens_per_chunk: int = 1500,
        on_progress: ProgressCallback = None,
        chunk_concurrency: int = 1,
        reprocess: bool = False,
    ) -> IngestResult:
        """Run the full ingestion pipeline on a PDF.

        Pipeline: parse(MinerU) → analyze_figures(VLM) → chunk → extract → match → reason → commit

        Args:
            pdf_path: Path to the PDF file.
            extraction_model: Model name for extraction (overrides config default).
            reasoning_model: Model name for reasoning (overrides config default).
            figure_analysis_model: Model name for figure analysis (overrides config default).
            max_tokens_per_chunk: Max tokens per chunk.
            on_progress: Async callback for progress reporting.
            chunk_concurrency: Max chunks processed concurrently. ``1`` (default) = the original
                strictly-sequential behavior. ``N > 1`` fans the per-chunk unit out under an
                ``asyncio.Semaphore(N)`` — the extract+reason (OpenRouter) cost overlaps while figures
                (local oMLX) and embed serialize. Trades a bounded intra-batch duplicate window
                (consolidation + a multi-pass reconcile it) for ~3–5× wall-clock. See
                ``experiments/CONCURRENT_INGEST_DESIGN.md``.
            reprocess: Re-run every chunk even if the checkpoint marks it done (the multi-pass refine
                path — re-matches against the now-complete graph to form cross-chunk edges the first
                concurrent pass missed and to reinforce duplicates onto one node).

        Returns:
            IngestResult with counts of changes applied.
        """
        pdf_path = Path(pdf_path)
        result = IngestResult(source_id="", title=pdf_path.stem)

        # ── Stage 0: Parse PDF with MinerU ──
        await self._emit(on_progress, "parse", 0, 1, f"Parsing {pdf_path.name} with MinerU")
        try:
            parsed = mineru_parse_pdf(
                pdf_path,
                backend=self._config.mineru.backend,
                **({"egress": "local-only"} if effective_egress(self._config) == "local-only" else {}),
            )
        except Exception as e:
            result.errors.append(f"MinerU parse failed: {e}")
            return result

        # ── Stage 0.5: Analyze figures with frontier VLM ──
        figure_analyses = []
        if parsed and parsed.figures:
            fa_model_name = figure_analysis_model or self._config.models.default_figure_analysis
            if fa_model_name:
                await self._emit(
                    on_progress, "figures", 0, len(parsed.figures),
                    f"Analyzing {len(parsed.figures)} figures with {fa_model_name}",
                )
                try:
                    analysis_llm = self._provider.get(fa_model_name)
                    # Use vision model for classification of unknowns
                    classify_llm = None
                    if self._config.models.default_vision:
                        try:
                            classify_llm = self._provider.get(self._config.models.default_vision)
                        except Exception:
                            pass

                    figure_analyses = await analyze_all_figures(
                        parsed.figures, analysis_llm, classify_llm,
                    )
                    logger.info("Analyzed %d figures", len(figure_analyses))
                except Exception as e:
                    logger.warning("Figure analysis failed (continuing without): %s", e)
                    result.errors.append(f"Figure analysis failed: {e}")

        return await self._ingest_from_parsed(
            parsed=parsed,
            original_path=pdf_path,
            result=result,
            figure_analyses=figure_analyses,
            extraction_model=extraction_model,
            reasoning_model=reasoning_model,
            max_tokens_per_chunk=max_tokens_per_chunk,
            on_progress=on_progress,
            chunk_concurrency=chunk_concurrency,
            reprocess=reprocess,
            source_kind="pdf",
        )

    async def ingest_html(
        self,
        html_path: str | Path,
        extraction_model: str | None = None,
        reasoning_model: str | None = None,
        max_tokens_per_chunk: int = 1500,
        on_progress: ProgressCallback = None,
        chunk_concurrency: int = 1,
        reprocess: bool = False,
        figures_only: bool = False,
    ) -> IngestResult:
        """Run the ingestion pipeline on one of Rick's lecture-capture HTML files.

        Pipeline: parse(html_parser) → chunk → extract → match → reason → reconcile → commit →
        embed → summarize — IDENTICAL to ingest() from the chunk stage onward (see
        ``_ingest_from_parsed``, the shared seam). The MinerU parse stage and the VLM
        figure-analysis stage are both skipped entirely rather than merely no-opped: this
        parser never populates ``ParsedDocument.figures`` (equations/diagrams live inside the
        base64 slide images — out of scope for this text-only adapter; see
        ``extraction/html_parser.py``'s module docstring for the full rationale and the two
        live-verified input-format variants it handles).

        Args:
            html_path: Path to the lecture HTML file.
            extraction_model / reasoning_model: same as ingest().
            max_tokens_per_chunk / on_progress / chunk_concurrency / reprocess: same as ingest()
                (checkpointing included — same PipelineCheckpoint semantics, keyed by the
                content-addressed source_id computed from the HTML file's own bytes).
            figures_only: when True, routes to ``_ingest_html_figures_only`` instead — VLM-analyzes
                every ``.pg`` slide IMAGE (equations/schematics/plots the text-only path above
                cannot see) and feeds only the slides that yield figure content through the SAME
                stage 1-6 seam, under a SEPARATE checkpoint namespace (``{source_id}_figs``) so it
                can run against a source this method has already text-ingested without disturbing
                that run's checkpoint. See ``_ingest_html_figures_only``'s docstring.

        Returns:
            IngestResult with counts of changes applied — same shape as ingest()'s.
        """
        html_path = Path(html_path)
        result = IngestResult(source_id="", title=html_path.stem)

        if figures_only:
            return await self._ingest_html_figures_only(
                html_path=html_path,
                result=result,
                extraction_model=extraction_model,
                reasoning_model=reasoning_model,
                max_tokens_per_chunk=max_tokens_per_chunk,
                on_progress=on_progress,
                chunk_concurrency=chunk_concurrency,
                reprocess=reprocess,
            )

        await self._emit(on_progress, "parse", 0, 1, f"Parsing {html_path.name} (lecture HTML)")
        try:
            parsed = parse_lecture_html(html_path)
        except Exception as e:
            result.errors.append(f"HTML parse failed: {e}")
            return result

        return await self._ingest_from_parsed(
            parsed=parsed,
            original_path=html_path,
            result=result,
            figure_analyses=[],
            extraction_model=extraction_model,
            reasoning_model=reasoning_model,
            max_tokens_per_chunk=max_tokens_per_chunk,
            on_progress=on_progress,
            chunk_concurrency=chunk_concurrency,
            reprocess=reprocess,
            source_kind="html",
        )

    async def _ingest_html_figures_only(
        self,
        *,
        html_path: Path,
        result: IngestResult,
        extraction_model: str | None,
        reasoning_model: str | None,
        max_tokens_per_chunk: int,
        on_progress: ProgressCallback,
        chunk_concurrency: int,
        reprocess: bool,
    ) -> IngestResult:
        """Figures-only lecture-slide ingest: parse WITH images → VLM-analyze every slide →
        gate to figure-bearing slides → build ContentBlocks → feed the UNCHANGED stage 1-6 seam
        (``_ingest_from_parsed``).

        Model routing (task-specified, see knowledge/extraction/slide_analyzer.py for the
        mechanics): primary = ``[figures].slide_analysis_model`` (local oMLX, free); per-slide
        transient failures fall back to ``[figures].slide_analysis_fallback`` (paid frontier);
        a run of CONSECUTIVE primary failures trips a circuit breaker that ABORTS THE WHOLE FILE
        (``SlideBreakerOpen``) rather than silently burning the paid fallback across every
        remaining slide of a ~4,000-slide corpus — see that module's docstring for the full cost
        rationale.

        Resumability: slide analysis (one VLM call per slide, ~10-30s each measured 2026-07-17)
        can legitimately take longer than a single operator/tool invocation wants to block for on
        a large deck, and a breaker trip deliberately stops mid-file. Rather than lose already-
        completed (free) analysis on any interruption, each slide's result is persisted
        incrementally to the SAME ``{source_id}_figs`` checkpoint the task specifies (via
        PipelineCheckpoint.save_extra/load_extra — a generic slot on the existing checkpoint
        file, not a second file) as soon as it completes; re-invoking this method on the same file
        skips slides already recorded there. ``reprocess=True`` bypasses this reuse (fresh
        analysis of every slide), mirroring what reprocess already means for stage 1-6.
        """
        await self._emit(
            on_progress, "parse", 0, 1, f"Parsing {html_path.name} (lecture HTML, figures-only)",
        )
        try:
            parsed = parse_lecture_html(html_path, include_images=True)
        except Exception as e:
            result.errors.append(f"HTML parse failed: {e}")
            return result
        result.source_id = parsed.source_id
        result.title = parsed.title

        if not parsed.slide_images:
            result.empty_reason = (
                f"No slide images found in {html_path.name} (figures-only mode) — nothing to analyze"
            )
            return result

        slide_model_name = self._config.figures.slide_analysis_model
        fallback_model_name = self._config.figures.slide_analysis_fallback
        try:
            primary_llm = self._provider.get(slide_model_name)
        except Exception as e:
            result.errors.append(
                f"Slide-analysis primary model {slide_model_name!r} unavailable: {e}"
            )
            return result
        fallback_llm = None
        if fallback_model_name:
            try:
                fallback_llm = self._provider.get(fallback_model_name)
            except Exception as e:
                logger.warning(
                    "Slide-analysis fallback model %r unavailable (per-slide transient failures "
                    "will have no fallback for this run): %s", fallback_model_name, e,
                )

        # ── Resumability: reuse any slide analyses already checkpointed under {source_id}_figs ──
        figs_checkpoint_id = f"{parsed.source_id}_figs"
        slide_checkpoint = self._create_checkpoint(figs_checkpoint_id)
        done_map: dict[str, dict] = {}
        if slide_checkpoint is not None:
            # Always load (populates self._state from disk if the file exists) so later
            # save_extra() calls MERGE onto any existing stage-1-6 checkpoint state instead of
            # starting from a blank dict — independent of whether we REUSE what was loaded below.
            slide_checkpoint.load()
            if not reprocess:
                done_map = dict(slide_checkpoint.load_extra("slide_analyses", {}) or {})
            # else (reprocess=True): deliberately ignore any cached analyses — re-analyze every
            # slide fresh, mirroring what reprocess already means for stage 1-6 below (a fresh
            # analysis naturally overwrites the stale cached entry per slide as it completes).

        todo: list[SlideImage] = []
        already_done: list[SlideAnalysis] = []
        for slide in parsed.slide_images:
            cached = done_map.get(str(slide.page_idx))
            if cached is not None:
                already_done.append(SlideAnalysis(**cached))
            else:
                todo.append(slide)

        if already_done:
            logger.info(
                "figures-only %s: reusing %d/%d already-analyzed slide(s) from checkpoint",
                html_path.name, len(already_done), len(parsed.slide_images),
            )

        def _persist_slide(slide: SlideImage, analysis: SlideAnalysis) -> None:
            # A slide that errored (both primary and fallback unusable) is deliberately NOT
            # cached — resumability exists to skip re-PAYING for a slide that already got a
            # real answer, not to permanently give up on one that failed. Leaving it out of
            # done_map means the next invocation retries it (exactly like the un-checkpointed
            # stage 1-6 chunk loop already does for any chunk that isn't marked done).
            if slide_checkpoint is None or analysis.error:
                return
            done_map[str(slide.page_idx)] = asdict(analysis)
            slide_checkpoint.save_extra("slide_analyses", done_map)

        await self._emit(
            on_progress, "figures", len(already_done), len(parsed.slide_images),
            f"Analyzing {len(todo)} slide image(s) with {slide_model_name} "
            f"({len(already_done)} already done)",
        )

        newly_analyzed: list[SlideAnalysis] = []
        try:
            if todo:
                newly_analyzed = await analyze_all_slides(
                    todo, primary_llm, fallback_llm, on_slide_done=_persist_slide,
                )
        except SlideBreakerOpen as e:
            result.errors.append(f"Slide analysis aborted for {html_path.name}: {e}")
            return result

        slide_analyses = sorted(already_done + newly_analyzed, key=lambda sa: sa.page_idx)

        # ── Gate: only slides with actual figure content become blocks (spec: local analysis is
        # free, so every slide is analyzed above, but only figure-bearing slides create graph work).
        blocks: list[ContentBlock] = []
        gated_count = 0
        failed_count = 0
        for sa in slide_analyses:
            if sa.error:
                failed_count += 1
            if not sa.has_figure_content:
                continue
            gated_count += 1
            heading, body = slide_analysis_to_blocks(sa)
            blocks.append(heading)
            blocks.append(body)

        logger.info(
            "figures-only %s: %d slide(s) analyzed, %d passed the figure-content gate, %d failed "
            "(both primary and fallback unusable)",
            html_path.name, len(slide_analyses), gated_count, failed_count,
        )

        if not blocks:
            result.empty_reason = (
                f"No figure content (equations/schematic/plot) found in any of "
                f"{len(slide_analyses)} analyzed slide(s) of {html_path.name} (figures-only "
                f"mode) — nothing to ingest"
            )
            return result

        figs_parsed = ParsedDocument(
            source_id=parsed.source_id, title=parsed.title, author=parsed.author,
            total_pages=parsed.total_pages, checksum=parsed.checksum, blocks=blocks,
            figures=[], output_dir="",
        )

        return await self._ingest_from_parsed(
            parsed=figs_parsed,
            original_path=html_path,
            result=result,
            figure_analyses=[],
            extraction_model=extraction_model,
            reasoning_model=reasoning_model,
            max_tokens_per_chunk=max_tokens_per_chunk,
            on_progress=on_progress,
            chunk_concurrency=chunk_concurrency,
            reprocess=reprocess,
            source_kind="html",
            checkpoint_source_id=figs_checkpoint_id,
        )

    async def _ingest_from_parsed(
        self,
        *,
        parsed: ParsedDocument,
        original_path: Path,
        result: IngestResult,
        figure_analyses: list,
        extraction_model: str | None,
        reasoning_model: str | None,
        max_tokens_per_chunk: int,
        on_progress: ProgressCallback,
        chunk_concurrency: int,
        reprocess: bool,
        source_kind: str,
        checkpoint_source_id: str | None = None,
    ) -> IngestResult:
        """The shared seam: chunk → extract → ground → match → reason → reconcile → commit →
        embed → summarize, EXACTLY as-is regardless of how ``parsed`` was produced (MinerU for a
        PDF, ``html_parser`` for a lecture HTML file). ``ingest()``/``ingest_html()`` only differ
        in stage 0 (parse) and — for ``ingest()`` only — stage 0.5 (figure VLM analysis, whose
        results arrive pre-computed in ``figure_analyses``; ``ingest_html`` always passes ``[]``
        since ``parse_lecture_html`` never populates ``ParsedDocument.figures``).

        ``source_kind`` ("pdf" | "html") selects only the evidence-vault artifact-persistence
        call (``_persist_pdf`` vs ``_persist_html``) and the source-noun used in the
        no-text-extracted error message — it has no effect on graph writes.

        ``checkpoint_source_id`` overrides ONLY which checkpoint FILE this run's per-chunk
        progress is tracked under (default: ``chunked.source_id``, i.e. unchanged behavior for
        every existing caller). Used by ``_ingest_html_figures_only`` to checkpoint under
        ``{source_id}_figs`` instead of the text ingest's ``{source_id}`` — so a figures-only run
        never touches the text run's checkpoint (or vice versa) even though both stamp the SAME
        ``chunked.source_id`` / ``result.source_id`` onto every graph node (identical identity,
        separate progress-tracking namespace).
        """
        # ── Stage 1: Chunk ──
        await self._emit(on_progress, "chunk", 0, 1, f"Chunking {original_path.name}")
        try:
            chunked = chunk_structured(
                parsed,
                figure_analyses=figure_analyses,
                max_tokens=max_tokens_per_chunk,
            )
        except Exception as e:
            result.errors.append(f"Chunking failed: {e}")
            return result

        result.source_id = chunked.source_id
        result.title = chunked.title
        result.total_chunks = len(chunked.chunks)

        self._persist_mineru_outputs(parsed.output_dir, chunked.checksum)

        # Register source in graph (tag with the primary models it is ingested with)
        await self._register_source(
            chunked,
            extraction_model=extraction_model or self._config.models.default_extraction,
            reasoning_model=reasoning_model or self._config.models.default_reasoning,
        )
        if source_kind == "html":
            self._persist_html(original_path, chunked.checksum)
        else:
            self._persist_pdf(original_path, chunked.checksum)

        if not chunked.chunks:
            result.errors.append(f"No text extracted from {source_kind.upper()}")
            return result

        # ── Checkpoint: load or initialize ──
        checkpoint = self._create_checkpoint(checkpoint_source_id or chunked.source_id)
        if checkpoint:
            if checkpoint.load() and not reprocess:
                # Resuming — recover accumulated counts. The file may have been created by
                # the figures pass's save_extra() before chunk tracking ever ran, so
                # backfill identity keys that initialize() would have stamped.
                checkpoint.ensure_meta(chunked.title, len(chunked.chunks))
                prev = checkpoint.get_accumulated_counts()
                result.new_nodes = prev["new_nodes"]
                result.updated_nodes = prev["updated_nodes"]
                result.new_edges = prev["new_edges"]
                result.reinforced_edges = prev["reinforced_edges"]
                result.insights = prev["insights"]
                result.errors.extend(checkpoint.get_errors())
                skipped = sum(1 for i in range(len(chunked.chunks)) if checkpoint.is_chunk_done(i))
                logger.info("Resuming: %d/%d chunks already done", skipped, len(chunked.chunks))
            else:
                # Fresh run, or reprocess=True (multi-pass refine) — re-run every chunk against the
                # now-complete graph regardless of any prior done-set.
                checkpoint.initialize(chunked.title, len(chunked.chunks))

        # Get model chains for each stage (primary + fallbacks)
        extract_chain = self._provider.get_chain("extraction", override=extraction_model)
        reason_chain = self._provider.get_chain("reasoning", override=reasoning_model)

        # Accumulates NEW CircuitTopology NodeProposals committed by any chunk this ingest, for the
        # document-level growth hook below (spec §2b/§4). Populated post-commit inside `_run_one`;
        # plain list.append is safe under chunk_concurrency > 1 because asyncio coroutines only
        # interleave at await points, never mid-statement.
        new_topology_nodes: list = []

        # ── Process each chunk ──
        # The per-chunk unit (extract → ground → match → reason → reconcile → commit → embed →
        # checkpoint) is independent and uses per-call Neo4j sessions + synchronous checkpoint/journal
        # mutations, so several can run at once under a bounded semaphore (chunk_concurrency > 1). The
        # full safety audit + known concurrency costs are in experiments/CONCURRENT_INGEST_DESIGN.md.
        async def _run_one(i: int, chunk) -> None:
            # Skip if checkpoint says this chunk is already done
            if checkpoint and checkpoint.is_chunk_done(i):
                await self._emit(
                    on_progress, "skip", i, len(chunked.chunks),
                    f"Skipping chunk {i+1}/{len(chunked.chunks)} (already processed)",
                )
                return

            await self._emit(
                on_progress, "process", i, len(chunked.chunks),
                f"Processing chunk {i+1}/{len(chunked.chunks)}: {chunk.section_title}",
            )

            try:
                # Stage 2: Extract (with resilience, two-pass). The LLM sees clean prose; the
                # figure-VLM sentinels are only used by grounding to exclude the description span.
                extraction = await self._resilient_extract(
                    strip_figure_markers(chunk.text), chunk.chunk_id, extract_chain,
                )

                # Stage 2.5: Grounding verification. Pass RAW chunk.text (with sentinels) so the
                # support set excludes the VLM figure-description span (anti self-grounding).
                extraction = verify_grounding(extraction, chunk.text)

                # Stage 3: Match
                match_result = await self._matcher.match(extraction)

                # Stage 4: Reason (with resilience). Pass result.errors by reference so a
                # terminal reasoning failure (degrades to an empty delta rather than crashing —
                # see _resilient_reason) is visible in the final IngestResult, not just logged.
                delta = await self._resilient_reason(
                    extraction, match_result, chunk.text, reason_chain, errors=result.errors,
                )

                # Stage 4.5: Inject source_id into new nodes (M4) +
                #            increment reinforcement_count for matched concepts/equations/
                #            parameters (M2, extended by F6) +
                #            scale confidence for single-source nodes (M5)
                # F6: equations/parameters now flow through the matcher too (exact-normalized
                # match — reasoning/matcher.py::_match_equations/_match_parameters). Mirror the
                # concept grain exactly: fold their matched-lists into the SAME matched_ids set
                # and the SAME reinforcement loop below rather than duplicating the mechanism.
                # This does not (and, mirroring concepts, deliberately does not) hard-filter a
                # matched item out of delta.new_nodes — that stays an LLM-compliance-dependent
                # signal via the reasoner prompt (see reasoner.py MATCH RESULTS listing); this
                # block only guarantees the mechanical M2/M5 side is always correct.
                all_matched = (
                    match_result.matched
                    + match_result.matched_equations
                    + match_result.matched_parameters
                )
                matched_ids = {m.existing_node_id for m in all_matched}
                for proposal in delta.new_nodes:
                    proposal.properties["source_id"] = chunk.source_id
                    # Assign knowledge_layer from domain if not already set by reasoner
                    if proposal.knowledge_layer < 0:
                        proposal.knowledge_layer = _DOMAIN_TO_LAYER.get(proposal.domain, -1)
                    # New (unmatched) node: scale down to improve confidence spread (M5)
                    if proposal.proposed_id not in matched_ids and proposal.confidence > 0:
                        proposal.confidence = round(proposal.confidence * 0.78, 4)
                _inject_insight_evidence(delta, chunk)
                _apply_grounding_scope_penalties(delta, extraction)

                for m in all_matched:
                    _mlabel = NodeLabel(m.node_label) if m.node_label in NodeLabel._value2member_map_ else NodeLabel.CONCEPT
                    node = await self._graph.get_node(
                        _mlabel, m.node_id_field, m.existing_node_id
                    )
                    if node is not None:
                        current = int(node.get("reinforcement_count") or 1)
                        delta.updated_nodes.append(NodeUpdate(
                            existing_node_id=m.existing_node_id,
                            label=_mlabel,
                            updates={"reinforcement_count": current + 1},
                            reasoning=f"{_mlabel.value.lower()} re-encountered in new source",
                        ))

                # Stage 5: Commit (register chunk + apply delta together)
                await self._register_chunk(chunk)
                counts = await self._graph.apply_delta(delta)
                result.new_nodes += counts["new_nodes"]
                result.updated_nodes += counts["updated_nodes"]
                result.new_edges += counts["new_edges"]
                result.reinforced_edges += counts["reinforced_edges"]
                result.insights += counts["insights"]

                # Post-commit: collect NEW CircuitTopology proposals for the document-level growth
                # hook (spec §2b) — only nodes that actually landed in apply_delta's commit, never
                # rejected/duplicate proposals.
                new_topology_nodes.extend(
                    p for p in delta.new_nodes if p.label == NodeLabel.CIRCUIT_TOPOLOGY
                )

                # Stage 5.5: Embed new knowledge nodes (non-blocking — errors are logged AND
                # recorded in result.errors; see _embed_new_nodes)
                await self._embed_new_nodes(delta.new_nodes, errors=result.errors)

                # Journal: record successful delta commit
                self._journal.log(
                    "apply_delta",
                    source_id=chunk.source_id,
                    chunk_id=chunk.chunk_id,
                    chunk_index=i + 1,
                    total_chunks=len(chunked.chunks),
                    **counts,
                )

                # Save checkpoint
                if checkpoint:
                    checkpoint.mark_chunk_done(i, counts)

            except ReasoningDegradedError as e:
                error_msg = (
                    f"Chunk {i+1} ({chunk.chunk_id}): reasoning terminally failed — no "
                    f"knowledge committed for this chunk; left un-checkpointed for retry "
                    f"— {e.cause}"
                )
                result.errors.append(error_msg)
                if checkpoint:
                    checkpoint.record_error(i, e.cause)

            except FallbackExhaustedError as e:
                safe = redact_secrets(e)
                error_msg = f"Chunk {i+1} ({chunk.chunk_id}): all models failed — {safe}"
                result.errors.append(error_msg)
                if checkpoint:
                    checkpoint.record_error(i, safe)

            except Exception as e:
                safe = redact_secrets(e)
                error_msg = f"Chunk {i+1} ({chunk.chunk_id}): {safe}"
                result.errors.append(error_msg)
                if checkpoint:
                    checkpoint.record_error(i, safe)

        if chunk_concurrency <= 1:
            # Original strictly-sequential path — exact prior behavior (default).
            for i, chunk in enumerate(chunked.chunks):
                await _run_one(i, chunk)
        else:
            # Bounded fan-out: at most chunk_concurrency chunks in flight.
            sem = asyncio.Semaphore(chunk_concurrency)

            async def _guarded(i: int, chunk) -> None:
                async with sem:
                    await _run_one(i, chunk)

            await asyncio.gather(
                *(_guarded(i, chunk) for i, chunk in enumerate(chunked.chunks))
            )

        # ── Corpus growth hook (spec §2b/§4) — document-level, ONCE per ingest rather than per
        # chunk: every new CircuitTopology node committed by any chunk's delta above is enqueued in
        # one batch. Cheaper than a per-chunk call (one file open/append instead of N) and the queue
        # semantics (append-only JSONL, matched-vs-unmatched routing) don't depend on chunk boundaries
        # — a topology node is a document-level fact regardless of which chunk introduced it. Pure
        # best-effort: any failure (bad state_dir, disk error, import error) is logged and swallowed,
        # never fails ingest (mirrors the embed/summarize best-effort pattern above).
        self._enqueue_growth_candidates(new_topology_nodes)

        # Clean up checkpoint on successful completion (no errors = all chunks done)
        all_done = checkpoint and all(
            checkpoint.is_chunk_done(i) for i in range(len(chunked.chunks))
        )
        if all_done:
            checkpoint.complete()

        # ── Stage 6: Paper-level summary ──
        await self._emit(
            on_progress, "summarize", len(chunked.chunks), len(chunked.chunks),
            "Generating paper-level summary",
        )
        try:
            # NOTE (2026-07-25): resolves stage='reasoning', so this call now inherits the
            # stage's [reasoning].output_token_budget bound — but summarize_paper does NOT go
            # through the truncation branch, so a clipped summary would be accepted silently.
            # Deliberate, not an oversight: a paper summary is a few hundred tokens against a
            # 16000 bound, and the whole block is already best-effort (`except` → warn, the
            # ingest result is unaffected). Revisit if summaries ever grow output-bound.
            summary_llm = self._provider.get_for_stage("reasoning")
            summary = await summarize_paper(result.source_id, self._graph, summary_llm)
            if summary:
                await apply_summary_to_source(result.source_id, summary, self._graph)
                logger.info("Paper summary: %s", summary.one_line)
        except Exception as e:
            logger.warning("Paper summary failed (non-critical): %s", e)

        await self._emit(
            on_progress, "done", len(chunked.chunks), len(chunked.chunks),
            f"Completed: {result.summary()}",
        )

        return result

    async def ingest_chunk(
        self,
        chunk_text: str,
        chunk_id: str,
        source_id: str,
        extraction_model: str | None = None,
        reasoning_model: str | None = None,
    ) -> dict[str, int]:
        """Process a single chunk (for fine-grained control or retries).

        Uses the resilience layer (retry + fallback) for LLM calls.
        """
        extract_chain = self._provider.get_chain("extraction", override=extraction_model)
        reason_chain = self._provider.get_chain("reasoning", override=reasoning_model)

        extraction = await self._resilient_extract(chunk_text, chunk_id, extract_chain)
        match_result = await self._matcher.match(extraction)
        delta = await self._resilient_reason(extraction, match_result, chunk_text, reason_chain)
        return await self._graph.apply_delta(delta)

    # ── Resilient LLM calls ──

    async def _resilient_extract(self, chunk_text, chunk_id, model_chain):
        """Two-pass extraction with cascade routing.

        Cascade: try primary (cheap) model → quality check → escalate if needed.
        Two-pass: entities first, then relationships given resolved entities.
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        from openclaw_brain.knowledge.extraction.extractor import (
            _ENTITY_SYSTEM_PROMPT,
            _EntityExtractionOutput,
            _RELATION_SYSTEM_PROMPT,
            _RelationExtractionOutput,
            _format_entity_list,
        )

        quality_check = ExtractionQualityCheck()

        # ── Pass 1: Entity extraction with cascade ──
        entity_messages = [
            SystemMessage(content=_ENTITY_SYSTEM_PROMPT),
            HumanMessage(content=(
                "Extract ALL concepts, equations, and parameters from this text.\n"
                "Remember: every concept MUST have a descriptive name in Title Case and a non-empty description.\n"
                "Do NOT skip equations or parameters — extract every one you find.\n\n"
                f"{chunk_text}"
            )),
        ]

        entities = None

        # Cascade: try primary model first, check quality, escalate if needed
        if len(model_chain) > 1:
            try:
                primary_model = model_chain[0].with_structured_output(_EntityExtractionOutput)
                entities = await invoke_with_resilience(
                    [primary_model],
                    entity_messages,
                    self._config.resilience,
                    auth_refresh=self._auth_refresh,
                    model_names=[resolve_model_name(model_chain[0])],
                )

                # Quality check — if primary model output is good enough, keep it
                preliminary = ExtractionResult(
                    chunk_id=chunk_id,
                    concepts=entities.concepts,
                    equations=entities.equations,
                    parameters=entities.parameters,
                )
                quality = quality_check.check(preliminary, chunk_text)
                if not quality:
                    logger.info(
                        "Cascade: escalating chunk %s — %s",
                        chunk_id, "; ".join(quality.reasons),
                    )
                    entities = None  # Force escalation
            except (FallbackExhaustedError, Exception):
                entities = None  # Primary failed, escalate

        # Escalation: use full fallback chain
        if entities is None:
            entity_models = [m.with_structured_output(_EntityExtractionOutput) for m in model_chain]
            entities = await invoke_with_resilience(
                entity_models,
                entity_messages,
                self._config.resilience,
                auth_refresh=self._auth_refresh,
                model_names=[resolve_model_name(m) for m in model_chain],
            )

        # ── Pass 2: Relation extraction (only if ≥2 entities) ──
        raw_edges = []
        entity_count = len(entities.concepts) + len(entities.equations) + len(entities.parameters)

        if entity_count >= 2:
            entity_list_str = _format_entity_list(entities)
            relation_messages = [
                SystemMessage(content=_RELATION_SYSTEM_PROMPT),
                HumanMessage(content=(
                    f"## Source Text\n{chunk_text}\n\n"
                    f"## Extracted Entities\n{entity_list_str}\n\n"
                    "Extract ALL relationships between these entities that are supported by the text above.\n"
                    "source_name and target_name MUST match entity names from the list exactly."
                )),
            ]

            try:
                relation_models = [m.with_structured_output(_RelationExtractionOutput) for m in model_chain]
                relations = await invoke_with_resilience(
                    relation_models,
                    relation_messages,
                    self._config.resilience,
                    auth_refresh=self._auth_refresh,
                    model_names=[resolve_model_name(m) for m in model_chain],
                )
                raw_edges = relations.relationships
            except FallbackExhaustedError:
                from openclaw_brain.knowledge.extraction import extractor as _ext
                # live-path bump — D1's counter instrumented extract_from_chunk(), which has no
                # production caller; the real Pass-2 flow is this inline reimplementation.
                _ext._pass2_failed_chunk_ids.add(chunk_id)
                logger.warning("Chunk %s: relation extraction failed, returning entities only", chunk_id)

        return ExtractionResult(
            chunk_id=chunk_id,
            concepts=entities.concepts,
            equations=entities.equations,
            parameters=entities.parameters,
            raw_edges=raw_edges,
        )

    async def _resilient_reason(
        self, extraction, match_result, chunk_text, model_chain,
        errors: list[str] | None = None,
    ):
        """Reason with retry + fallback, with normalization for local models.

        Args:
            errors: optional list (typically the caller's IngestResult.errors) that a terminal
                reasoning failure appends a structured entry to. Defaults to None (no-op) so
                callers that don't track per-chunk errors (e.g. ingest_chunk) are unaffected.
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        from openclaw_brain.knowledge.graph.schema import GraphDelta
        from openclaw_brain.knowledge.reasoning.reasoner import _SYSTEM_PROMPT
        from openclaw_brain.knowledge.reasoning.normalize import (
            describe_parse_failure,
            extract_json,
            normalize_graph_delta,
        )
        from openclaw_brain.llm.resilience import detect_finish_reason

        context = await self._reasoner._gather_context(match_result)
        prompt = self._reasoner._build_prompt(extraction, match_result, context, chunk_text)

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        # Try structured output on primary model (single attempt, no retries).
        # If it works (Anthropic, OpenAI, Google), return immediately.
        # If it fails (EXO/local models with schema deviations), fall through
        # to raw + normalize with full resilience.
        import asyncio as _asyncio
        primary = model_chain[0]
        try:
            structured = primary.with_structured_output(GraphDelta)
            coro = structured.ainvoke(messages)
            timeout = self._config.resilience.request_timeout_s
            result = await _asyncio.wait_for(coro, timeout=timeout) if timeout > 0 else await coro
            return result
        except Exception:
            logger.info("Structured output failed on primary model, using raw + normalize")

        # Fallback: raw invocation with full resilience → JSON extraction → normalization →
        # GraphDelta construction. A single chunk whose reasoning output still cannot be
        # parsed into a valid GraphDelta (all models exhausted, unparseable JSON, or a
        # validation error normalize_graph_delta couldn't tolerate) must not crash the whole
        # multi-chunk ingest — degrade to an empty GraphDelta (no new knowledge from this
        # chunk) and let the per-chunk loop continue, exactly as it already does for a chunk
        # that legitimately produces nothing new.
        # Held outside the try so the failure handler below can classify WHAT came back
        # (truncated JSON vs. prose vs. empty) — the response body is the one thing the
        # 2026-07 parse-failure logs never captured.
        # `.content` is typed `str | list[str | dict]` in LangChain — multi-part content is a
        # real shape, and the diagnostics below are built to survive it (see
        # describe_parse_failure), so this is deliberately NOT narrowed to str.
        raw_text: Any = None
        response = None
        try:
            response = await invoke_with_resilience(
                list(model_chain),
                messages,
                self._config.resilience,
                auth_refresh=self._auth_refresh,
                model_names=[resolve_model_name(m) for m in model_chain],
                # The reasoning stage emits the pipeline's largest output (a full GraphDelta),
                # so it is the one stage where hitting the output bound is a live failure mode.
                # A finish_reason='length' response is discarded and retried once at 1.5× the
                # bound before the chain moves on — a partial GraphDelta must never be parsed.
                truncation_retry=True,
            )
            raw_text = response.content if hasattr(response, "content") else str(response)
            raw_json = extract_json(raw_text)
            normalized = normalize_graph_delta(raw_json)
            return GraphDelta(**normalized)
        except Exception as e:
            chunk_id = getattr(extraction, "chunk_id", "?")
            safe = redact_secrets(e)
            # Diagnostics: length + last 200 chars + brace presence + finish_reason. Enough to
            # tell a truncated body (no closing brace, stops mid-token) from prose (ends on a
            # sentence, often no '{' at all) without storing the body. Empty when the call
            # itself failed (FallbackExhaustedError) — there is no response to describe.
            #
            # Belt AND braces: describe_parse_failure is written to be total, but it is being
            # called from INSIDE an except block whose entire job is graceful degradation. A
            # telemetry bug must never be able to replace `ReasoningDegradedError` (chunk
            # recorded, left un-done, retried next run) with an unhandled crash that takes the
            # whole ingest down — so the second layer is here, at the call site, and the
            # original error is re-raised with its normal cause either way.
            try:
                diag = (
                    describe_parse_failure(
                        raw_text, finish_reason=detect_finish_reason(response),
                    )
                    if raw_text is not None
                    else "parse_failure: no response (call failed before any body)"
                )
            except Exception as diag_exc:  # pragma: no cover — defensive
                diag = f"parse_failure: diagnostics raised {type(diag_exc).__name__}"
            logger.warning(
                "Chunk %s: reasoning output could not be parsed into a valid GraphDelta (%s) "
                "— degrading to an empty delta instead of crashing the ingest [%s]",
                chunk_id, safe, diag,
            )
            # Raise instead of returning an empty delta: an empty delta used to get
            # committed (match-stage reinforcements made its counts non-zero) and the chunk
            # was checkpointed done — so the chunk's reasoning yield was silently lost AND
            # never retried, while the stale error string kept exit codes at 1 on later,
            # fully-successful runs. Raising routes to the loop's per-chunk handler: error
            # recorded, chunk left NOT-done, next invocation retries it (C6).
            # The diagnostic rides along on the cause so it reaches IngestResult.errors and the
            # checkpoint's error record too — not just a log line nobody tails mid-batch.
            raise ReasoningDegradedError(chunk_id, f"{safe} [{diag}]")

    # ── Checkpoint helpers ──

    def _create_checkpoint(self, source_id: str) -> PipelineCheckpoint | None:
        """Create a checkpoint manager if checkpointing is enabled."""
        if not self._config.resilience.checkpoint_enabled:
            return None
        checkpoint_dir = self._config.state_path / "checkpoints"
        return PipelineCheckpoint(checkpoint_dir, source_id)

    # ── Embedding ──

    async def _embed_new_nodes(self, proposals: list, errors: list[str] | None = None) -> None:
        """Generate and store embeddings for new knowledge nodes (best-effort).

        Encodes canonical_name + description as a batch, then persists each
        embedding via GraphStore.store_embedding(). Errors are logged AND (when
        an ``errors`` list is supplied — typically the caller's IngestResult.errors)
        recorded there; they never propagate — embedding failures must not abort
        ingestion. A node that commits with no embedding is invisible to Tier 1/2
        (embedding-based) matching in every future ingest, so this is worth a
        visible record, not just a log line nobody watches mid-ingest.
        """
        embeddable = [p for p in proposals if p.label in _EMBEDDABLE_LABELS]
        if not embeddable:
            return

        texts = []
        for p in embeddable:
            text = p.canonical_name
            if p.description:
                text = f"{p.canonical_name}. {p.description}"
            texts.append(text)

        try:
            vectors = encode_batch(texts, self._config.embedding.model)
        except Exception as e:
            safe = redact_secrets(e)
            logger.warning("Embedding batch failed (skipping): %s", safe)
            if errors is not None:
                errors.append(
                    f"Embedding batch failed for {len(embeddable)} new node(s) — committed "
                    f"without embeddings (invisible to future embedding-based matching): {safe}"
                )
            return

        for proposal, vector in zip(embeddable, vectors):
            if not vector:
                continue
            try:
                id_field = self._graph._id_field_for_label(proposal.label)
                await self._graph.store_embedding(
                    proposal.label, id_field, proposal.proposed_id, vector
                )
            except Exception as e:
                safe = redact_secrets(e)
                logger.warning("store_embedding failed for %s: %s", proposal.proposed_id, safe)
                if errors is not None:
                    errors.append(
                        f"store_embedding failed for node {proposal.proposed_id!r} "
                        f"({proposal.label.value}) — committed without an embedding: {safe}"
                    )

    # ── Corpus growth hook ──

    def _enqueue_growth_candidates(self, topology_nodes: list) -> None:
        """Best-effort ingest hook into corpus growth automation (spec §2b/§4,
        docs/superpowers/specs/2026-07-03-corpus-growth-automation-design.md): match each NEW
        CircuitTopology proposal against the registered-template alias table via
        `growth.enqueue_candidates` — a registry match enqueues `growth_queue.jsonl` (AUTO lane,
        carrying source-chunk grounding); an unmatched high-confidence (>=0.55 post-scale,
        ~0.7 raw before this pipeline's x0.78 reconcile scaling) layer-2/3 node
        enqueues `template_queue.jsonl` (REVIEW lane, human template-authoring).
        Known trade-off (document-level batching): chunks resumed from a prior crashed run
        short-circuit before the post-commit accumulator, so their topology nodes are not
        re-enqueued — acceptable for a best-effort hook. Pure JSONL append
        (no Neo4j, no LLM, no simulation) — ingest latency is untouched.

        `enqueue_candidates` itself does NOT swallow exceptions (only its own file I/O is
        non-fatal by construction); this wrapper is what makes the hook non-fatal to ingest as a
        whole, per spec §4 ("failure is non-fatal and logged") — mirrors `_embed_new_nodes` /
        `_persist_pdf`'s best-effort discipline elsewhere in this file.
        """
        if not topology_nodes:
            return
        try:
            from openclaw_brain.knowledge.executable import growth
            growth.enqueue_candidates(topology_nodes, self._config.state_path)
        except Exception as e:
            logger.warning("Corpus growth enqueue failed (non-fatal): %s", e)

    def _persist_pdf(self, pdf_path: Path, checksum: str) -> None:
        """Best-effort copy of the original PDF into the evidence vault."""
        if not checksum:
            logger.warning("Evidence vault PDF copy skipped: missing source checksum")
            return
        try:
            self._vault.put_file(pdf_path, "pdf", checksum)
        except Exception as e:
            logger.warning("Evidence vault PDF copy failed for %s: %s", pdf_path, e)

    def _persist_html(self, html_path: Path, checksum: str) -> None:
        """Best-effort copy of the original lecture HTML into the evidence vault (mirrors
        _persist_pdf exactly; separate "html" kind so it lands under evidence/html/ rather than
        colliding with the "pdf" namespace — see EvidenceVault._file_dest)."""
        if not checksum:
            logger.warning("Evidence vault HTML copy skipped: missing source checksum")
            return
        try:
            self._vault.put_file(html_path, "html", checksum)
        except Exception as e:
            logger.warning("Evidence vault HTML copy failed for %s: %s", html_path, e)

    def _persist_mineru_outputs(self, output_dir: str, checksum: str) -> None:
        """Best-effort copy of MinerU markdown/content-list outputs into the vault."""
        if not output_dir or not checksum:
            return
        try:
            self._vault.put_dir_files(Path(output_dir), "mineru", checksum)
        except Exception as e:
            logger.warning("Evidence vault MinerU copy failed for %s: %s", output_dir, e)

    # ── Graph registration ──

    async def _register_source(
        self, chunked: ChunkerResult, extraction_model: str = "", reasoning_model: str = ""
    ) -> None:
        """Create/update Source node in the graph (with model provenance)."""
        await self._graph.merge_node(
            label=NodeLabel.SOURCE,
            id_field="source_id",
            id_value=chunked.source_id,
            properties={
                "source_id": chunked.source_id,
                "title": chunked.title,
                "author": chunked.author,
                "checksum": chunked.checksum,
                "extraction_model": extraction_model,
                "reasoning_model": reasoning_model,
            },
        )

    async def _register_chunk(self, chunk) -> None:
        """Create/update SourceChunk node and link to Source."""
        from openclaw_brain.knowledge.graph.schema import RelType

        raw_text_hash = ""
        try:
            raw_text_hash = self._vault.put_text(chunk.text)
            chunk.raw_text_hash = raw_text_hash
        except Exception as e:
            logger.warning("Evidence vault chunk text write failed for %s: %s", chunk.chunk_id, e)

        await self._graph.merge_node(
            label=NodeLabel.SOURCE_CHUNK,
            id_field="chunk_id",
            id_value=chunk.chunk_id,
            properties={
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "pages": chunk.pages,
                "section_title": chunk.section_title,
                "raw_text_hash": raw_text_hash,
                "text_preview": chunk.text[:500],
            },
        )
        await self._graph.merge_edge(
            source_label=NodeLabel.SOURCE_CHUNK,
            source_id_field="chunk_id",
            source_id_value=chunk.chunk_id,
            target_label=NodeLabel.SOURCE,
            target_id_field="source_id",
            target_id_value=chunk.source_id,
            rel_type=RelType.EXTRACTED_FROM,
            properties={
                "rationale": f"Chunk from pages {chunk.pages}",
                "confidence": 1.0,
            },
        )

    @staticmethod
    async def _emit(
        callback: ProgressCallback,
        stage: str,
        index: int,
        total: int,
        message: str,
    ) -> None:
        if callback:
            await callback(PipelineProgress(
                stage=stage,
                chunk_index=index,
                total_chunks=total,
                message=message,
            ))
