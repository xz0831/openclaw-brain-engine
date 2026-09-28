"""Agent core — the main orchestration layer.

Ties together: memory (M2), workflows (M3), skills (M4),
knowledge engine (M5), unified retriever, and reinforcement.

This is the primary interface that MCP Server (M7) will expose.
External agents call methods on BrainAgent, not on individual subsystems.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

from openclaw_brain.auth import inject_api_keys
from openclaw_brain.config import BrainConfig, load_config
from openclaw_brain.egress import LOCAL_ONLY, effective_egress, enforce_startup_egress
from openclaw_brain.journal import ActionJournal
from openclaw_brain.knowledge.evidence import EvidenceVault
from openclaw_brain.knowledge.extraction.figure_analyzer import (
    analyze_figure,
    classify_by_caption,
    classify_with_vlm,
)
from openclaw_brain.knowledge.extraction.mineru_parser import ContentBlock
from openclaw_brain.knowledge.graph.schema import EdgeProperties, NodeLabel, RelType
from openclaw_brain.knowledge.graph.store import GraphStore
from openclaw_brain.knowledge.reasoning.answerer import answer_from_context
from openclaw_brain.knowledge.reasoning.answer_contract import (
    ContextManifest, NO_KB_PREFIX, PARTIAL_KB_PREFIX, build_envelope, citation_ids,
)
from openclaw_brain.knowledge.reasoning.answer_feedback import append_knowledge_gap
from openclaw_brain.knowledge.reasoning.answer_resolution import resolve_exact
from openclaw_brain.knowledge.reinforcement import ReinforcementEngine
from openclaw_brain.knowledge.skill_handler import register_html_ingest_skill, register_pdf_ingest_skill
from openclaw_brain.llm.provider import LLMProvider
from openclaw_brain.memory.episodic import EpisodicMemory
from openclaw_brain.memory.models import EpisodicEvent
from openclaw_brain.memory.procedural import ProceduralMemory
from openclaw_brain.memory.promotion import PromotionPipeline
from openclaw_brain.memory.semantic import SemanticMemory
from openclaw_brain.memory.store import MemoryStore
from openclaw_brain.retriever import UnifiedRetriever
from openclaw_brain.skills.executor import SkillExecutor
from openclaw_brain.skills.router import SkillRouter
from openclaw_brain.skills.registry import SkillRegistry


def _scope_inline(scope: dict) -> str:
    """Render a claim-card scope as an undetachable verdict suffix (why()'s inline tag).
    Analog: pdk/corner[/statistical][/intervention:<id>]; digital: functional[/stimulus:<class>].

    E3-I2 (spec §4-I2): an intervention-family card's `scope` carries `intervention`/`idealization`
    (executor.py stamps both, additively, onto the SAME `scope` dict `summarize_scope` returns —
    no Conditions/summarize_scope change, so this is a pure rendering addition here). Appended LAST
    so the tag rides on the END of the existing pdk/corner/statistical/stimulus chain, exactly like
    every other scope axis above it — undetachable from the verdict, per the E1b standard."""
    pdk = scope.get("pdk")
    corners = scope.get("corners") or []
    inline = corners[0] if corners else (f"{pdk}/" if pdk else "") + (
        "functional" if scope.get("functional") else "nominal")
    if scope.get("statistical", "none") not in (None, "none", "n/a"):
        inline = f"{inline}/{scope['statistical']}"          # VERIFIED@sky130/tt_mm/27/1.8/3σ@200
    stim = scope.get("stimulus")
    if stim and stim != "unspecified":
        inline = f"{inline}/stimulus:{stim}"                 # VERIFIED@functional/stimulus:count 0->15
    iv = scope.get("intervention")
    if iv:
        idealized = "(idealized)" if scope.get("idealization") else ""
        inline = f"{inline}/intervention:{iv}{idealized}"    # VERIFIED@sky130/tt/27/1.8/intervention:ff_break(idealized)
    # E2a-I1 (docs/superpowers/specs/2026-07-05-e2a-analytic-envelope.md §2/§4): the analytic-envelope
    # verdict — a SECOND, independent instrument's concurrence/violation on the SAME claim — rides on
    # the END of the chain, exactly like intervention/idealization above (additive, undetachable; never
    # overrides the oracle's own verdict rendered by everything before it). `scope["envelope"]` is
    # stamped by envelope.py's `stamp_envelope_scope` (`{"status": ..., "ratio": ...}`); absent for
    # every pre-E2a card (backcompat, byte-identical to today). A VIOLATED status renders exactly like
    # a CONCUR one — this function never silently drops it.
    env = scope.get("envelope")
    if env:
        status = env.get("status")
        tag = "concur" if status == "CONCUR" else status   # CONCUR->"concur", VIOLATED/NA pass through
        ratio = env.get("ratio")
        suffix = f"@{ratio:.2f}" if ratio is not None else ""
        inline = f"{inline}/envelope:{tag}{suffix}"          # VERIFIED@sky130/tt/27/1.8/envelope:concur@1.08
    return inline


# S5 learner model — record_assessment's verdict -> UNDERSTANDS.confidence mapping. The design
# doc (docs/specs/S5_LEARNER_MODEL_DESIGN.md §3.1/§6.1) locks the UNDERSTANDS edge's property
# set (confidence/status/last_assessed/assessment_count) but leaves the confidence ARITHMETIC an
# implementation choice — this is that choice: a small, explicit, monotonic table rather than a
# prior-blending formula, so it's auditable and easy to retune once real usage data exists (§7).
# "misconception" is the one verdict value the doc gives explicit special handling (§3.3); any
# other verdict string (e.g. "understood") falls through to _DEFAULT_VERDICT_CONFIDENCE unless
# named here.
_VERDICT_CONFIDENCE: dict[str, float] = {
    "understood": 0.9,
    "misconception": 0.15,
}
_DEFAULT_VERDICT_CONFIDENCE = 0.5

logger = logging.getLogger(__name__)


class BrainAgent:
    """The main agent — single entry point to openclaw-brain capabilities.

    Lifecycle:
        agent = BrainAgent()
        await agent.start()
        ...
        await agent.stop()

    All 28 MCP tools delegate to methods on this class.
    Key public methods: ingest_pdf, ingest_html, analyze_circuit, query_knowledge,
    recall, reinforce, merge_concepts, retract_node, record_hypothesis,
    record_decision, record_bench_result, list_open_hypotheses,
    start_session, end_session, record_event, upsert_entity,
    record_lesson, run_maintenance, find_bridges, get_stats.
    """

    def __init__(self, config: BrainConfig | None = None):
        self._config = config or load_config()
        self._graph: GraphStore | None = None
        self._llm_provider: LLMProvider | None = None
        self._memory_store: MemoryStore | None = None
        self._episodic: EpisodicMemory | None = None
        self._semantic: SemanticMemory | None = None
        self._procedural: ProceduralMemory | None = None
        self._retriever: UnifiedRetriever | None = None
        self._reinforcement: ReinforcementEngine | None = None
        self._evidence_vault = EvidenceVault(self._config.state_path / "evidence")
        self._registry: SkillRegistry | None = None
        self._router: SkillRouter | None = None
        self._executor: SkillExecutor | None = None
        self._promotion: PromotionPipeline | None = None
        self._journal: ActionJournal | None = None
        self._started = False

    @property
    def is_started(self) -> bool:
        return self._started

    async def start(self) -> None:
        """Initialize all subsystems and connect to Neo4j."""
        if self._started:
            return

        # Fail before auth resolution, Neo4j connection, or client construction.  Besides
        # checking every configured model chain, this arms offline HF/transformers loading and
        # rejects enabled LangChain/LangSmith tracing under local-only.
        enforce_startup_egress(self._config)

        # Cloud credentials are irrelevant under local-only, and resolving them may execute a
        # configured SecretRef helper.  Do not touch that surface when every cloud model is
        # mechanically forbidden.  The local oMLX bearer key has its own call-time resolver.
        if effective_egress(self._config) != LOCAL_ONLY:
            inject_api_keys()

        # Action journal (file-based, always available)
        self._journal = ActionJournal(self._config.state_path)

        # Graph store
        self._graph = GraphStore(self._config.neo4j, egress=effective_egress(self._config))
        await self._graph.connect()

        # LLM provider
        self._llm_provider = LLMProvider(self._config)

        # Memory system
        self._memory_store = MemoryStore(self._graph, self._config)
        self._episodic = EpisodicMemory(self._memory_store, self._graph)
        self._semantic = SemanticMemory(self._memory_store, self._graph)
        self._procedural = ProceduralMemory(self._memory_store, self._graph)
        self._promotion = PromotionPipeline(self._memory_store, self._config)

        # Retriever
        self._retriever = UnifiedRetriever(self._memory_store, self._graph, self._config)

        # Reinforcement
        self._reinforcement = ReinforcementEngine(self._graph, self._config)

        # Skill system
        self._registry = SkillRegistry()
        self._router = SkillRouter(self._registry, self._procedural)
        self._executor = SkillExecutor(self._registry, self._procedural)

        # Register built-in skills
        register_pdf_ingest_skill(
            self._registry,
            self._graph,
            self._config,
            self._llm_provider,
            vault=self._evidence_vault,
        )
        register_html_ingest_skill(
            self._registry,
            self._graph,
            self._config,
            self._llm_provider,
            vault=self._evidence_vault,
        )

        self._started = True

    async def stop(self) -> None:
        """Shut down all subsystems."""
        if self._graph:
            await self._graph.close()
        self._started = False

    # ── Core operations (MCP-exposed) ──

    async def ingest_pdf(
        self,
        file_path: str,
        extraction_model: str | None = None,
        reasoning_model: str | None = None,
        session_id: str = "",
        chunk_concurrency: int = 1,
        reprocess: bool = False,
    ) -> dict[str, Any]:
        """Ingest a PDF into the knowledge graph.

        This is what gets called when a user sends a PDF via Telegram.

        chunk_concurrency: max chunks processed concurrently (1 = sequential, the default).
        reprocess: re-run every chunk even if checkpointed done (multi-pass refine).
        See experiments/CONCURRENT_INGEST_DESIGN.md.
        """
        self._assert_started()
        return await self._executor.execute(
            "pdf_ingest",
            {
                "file_path": file_path,
                "extraction_model": extraction_model,
                "reasoning_model": reasoning_model,
                "chunk_concurrency": chunk_concurrency,
                "reprocess": reprocess,
            },
            session_id=session_id,
        )

    async def ingest_html(
        self,
        file_path: str,
        extraction_model: str | None = None,
        reasoning_model: str | None = None,
        session_id: str = "",
        chunk_concurrency: int = 1,
        reprocess: bool = False,
        figures_only: bool = False,
    ) -> dict[str, Any]:
        """Ingest one of Rick's lecture-capture HTML decks into the knowledge graph.

        Mirrors ingest_pdf's contract exactly (same params, same routing through the
        SkillExecutor for timing/procedural-memory instrumentation, same checkpointing —
        PipelineCheckpoint is keyed by the content-addressed source_id, identical semantics to
        the PDF path). See knowledge/extraction/html_parser.py for the input format this expects
        (Rick's own lecture-capture pipeline's per-lecture HTML output) and
        knowledge/pipeline.py::ingest_html for the parse-stage substitution (html_parser instead
        of MinerU; no figure-VLM stage) ahead of the identical chunk->...->summarize seam.

        chunk_concurrency: max chunks processed concurrently (1 = sequential, the default).
        reprocess: re-run every chunk even if checkpointed done (multi-pass refine).
        figures_only: VLM-analyze the deck's slide IMAGES instead of its speech transcript — see
            knowledge/pipeline.py::KnowledgePipeline._ingest_html_figures_only for the full
            model-routing/circuit-breaker/checkpoint-namespace contract. Can run against a
            source this method has already text-ingested (same source_id, separate checkpoint).
        """
        self._assert_started()
        return await self._executor.execute(
            "html_ingest",
            {
                "file_path": file_path,
                "extraction_model": extraction_model,
                "reasoning_model": reasoning_model,
                "chunk_concurrency": chunk_concurrency,
                "reprocess": reprocess,
                "figures_only": figures_only,
            },
            session_id=session_id,
        )

    async def analyze_circuit(self, image_path: str) -> dict[str, Any]:
        """Analyze a circuit image and return knowledge-grounded insights.

        Classifies the image, extracts circuit topology/components via frontier VLM,
        then cross-references against the knowledge graph to surface related concepts,
        equations, and principles (e.g. from Razavi).

        Args:
            image_path: Absolute path to the image file (PNG, JPG).

        Returns:
            dict with: figure_type, visual_analysis, knowledge_context
        """
        self._assert_started()

        # Step 1: Caption-based classification (no LLM — usually "unknown" for standalone images)
        fig_type = classify_by_caption("")

        # Step 2: VLM classification if needed
        if fig_type == "unknown" and self._config.models.default_vision:
            vision_llm = self._llm_provider.get_model(self._config.models.default_vision)
            fig_type = await classify_with_vlm(image_path, vision_llm)
        if fig_type == "unknown":
            fig_type = "other"

        # Step 3: Deep analysis with frontier VLM
        if self._config.models.default_figure_analysis:
            analysis_llm = self._llm_provider.get_model(
                self._config.models.default_figure_analysis
            )
            figure = ContentBlock(
                type="image",
                page_idx=0,
                text="",
                img_path=image_path,
                caption="",
            )
            analysis = await analyze_figure(figure, analysis_llm, fig_type)
            visual_description = analysis.description
        else:
            visual_description = f"[Figure type: {fig_type}. No analysis model configured.]"

        # Step 4: Cross-reference against knowledge graph
        ctx = await self._retriever.retrieve(visual_description)
        knowledge_context = ctx.format(self._config.memory.auto_inject_token_budget)

        return {
            "figure_type": fig_type,
            "visual_analysis": visual_description,
            "knowledge_context": knowledge_context,
        }

    async def query_knowledge(
        self,
        query: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Query the knowledge graph and memory for relevant context.

        Returns structured context that an Agent can use to answer questions.
        """
        self._assert_started()
        ctx = await self._retriever.retrieve(query, session_id=session_id)
        compact = ctx.compact()
        return {
            "formatted": ctx.format(self._config.memory.auto_inject_token_budget),
            "concepts_found": ctx.graph.total_concepts_found,
            "memories_found": len(ctx.memories),
            "concepts": ctx.graph.concepts,
            "neighbors": ctx.graph.neighbors,
            # Compact refs (IDs) so callers can close the read→write loop.
            "concept_refs": compact["concepts"],
            "open_hypotheses": compact["open_hypotheses"],
            "active_decisions": compact["active_decisions"],
        }

    async def answer_question(
        self,
        question: str,
        session_id: str | None = None,
        *,
        on_insufficient: Literal["flag", "abstain", "model_knowledge"] = "model_knowledge",
    ) -> dict[str, Any]:
        """Answer and audit citations; gap feedback stores the original question locally."""
        self._assert_started()
        if on_insufficient not in ("flag", "abstain", "model_knowledge"):
            raise ValueError("on_insufficient must be 'flag', 'abstain', or 'model_knowledge'")
        kq = await self.query_knowledge(question, session_id=session_id)
        concepts = kq["concept_refs"]
        manifest = ContextManifest.from_refs(concepts)
        from openclaw_brain.knowledge.reasoning.answerer import AnswerResult
        from openclaw_brain.knowledge.executable.lesson import audit_answer_citations

        if not manifest.refs and on_insufficient != "model_knowledge":
            result = AnswerResult(abstained=True, abstain_reason="no_context")
        else:
            result = await answer_from_context(
                question=question,
                context_markdown=kq["formatted"],
                concept_refs=concepts,
                open_hypotheses=kq["open_hypotheses"],
                active_decisions=kq["active_decisions"],
                model_chain=self._llm_provider.get_chain("reasoning"),
                resilience_config=self._config.resilience,
                auth_refresh=None,
                on_insufficient=on_insufficient,
                no_context=not manifest.refs,
            )
        if not manifest.refs:
            resolved: dict[str, dict] = {}
            audit = {"state": "not_run", "passed": None, "findings": []}
        else:
            ids, _ = citation_ids(result)
            all_ids = list(dict.fromkeys(ids + result.used_concepts))
            if len(all_ids) > 32 or any(len(cid) > 512 for cid in all_ids):
                resolved = {}
                audit = {"state": "not_run", "passed": None, "findings": []}
            else:
                provenance = list(dict.fromkeys(
                    chunk for ref in concepts if ref.get("id") in ids
                    for chunk in ((ref.get("cite") or {}).get("chunks") or [])[:3]
                    if isinstance(chunk, str) and chunk not in ids
                ))
                checked = await resolve_exact(self._graph, all_ids + provenance)
                resolved = {cid: checked[cid] for cid in all_ids}
                for ref in concepts:
                    cid = ref.get("id")
                    if cid in resolved:
                        chunk_states = [checked.get(chunk, {}).get("resolution", "error")
                                        for chunk in ((ref.get("cite") or {}).get("chunks") or [])[:3]]
                        if "error" in chunk_states:
                            resolved[cid]["provenance_check_error"] = True
                        elif any(state != "active" for state in chunk_states):
                            resolved[cid]["broken_provenance_ref"] = True
                try:
                    def for_audit(cid: str) -> dict | None:
                        row = resolved.get(cid)
                        if not row or row.get("resolution") != "active":
                            return None
                        return {"labels": [row["kind"]],
                                "verdict": row.get("verdict") if row.get("tier") == 3 else None,
                                "status": "law" if row.get("tier") == 4 else None,
                                "scope_pdks": row.get("scope_pdks")}
                    report = audit_answer_citations(result.answer, for_audit)
                    audit = {"state": "completed", **report.model_dump()}
                except Exception:
                    audit = {"state": "error", "passed": None, "findings": []}
        if on_insufficient == "model_knowledge":
            # A trailing marker with no following content is not part of the answer.
            clean_answer = re.sub(r"\s*\[모델 지식\]\s*$", "", result.answer)
            result = result.model_copy(update={
                "answer": clean_answer if clean_answer.strip() else "",
            })
        env = build_envelope(result, manifest, resolved, audit, on_insufficient)
        answer = result.answer
        abstained = result.abstained
        reason = result.abstain_reason
        if env.status == "invalid":
            answer = ""
            abstained = True
            reason = f"contract_invalid:{next((code for code in env.reason_codes if code not in ('scope_unknown', 'no_inline_citations', 'broken_provenance_ref')), 'unknown')}"
        elif env.status == "abstained":
            answer = ""
            abstained = True
            if "policy_insufficient" in env.reason_codes:
                reason = "policy_insufficient"
        elif on_insufficient == "model_knowledge":
            if "model_abstained_with_body" in env.reason_codes:
                abstained = False
                reason = ""
            coverage = env.checks.kb_coverage
            if coverage == "none":
                answer = f"{NO_KB_PREFIX}\n{answer}"
            elif coverage == "partial":
                answer = f"{PARTIAL_KB_PREFIX}\n{answer}"
        if (on_insufficient == "model_knowledge"
                and env.checks.kb_coverage in ("none", "partial")
                and getattr(getattr(self._config, "answer", None), "gap_log", True)):
            missing = [item.strip() for item in result.missing_knowledge
                       if isinstance(item, str) and item.strip()][:5]
            cited = [c.id for c in env.citations
                     if c.resolution == "active" and c.in_context]
            try:
                append_knowledge_gap(
                    self._config.state_path, question=question,
                    missing_knowledge=missing, kb_coverage=env.checks.kb_coverage,
                    cited_ids=cited, context_sha256=env.context_sha256,
                )
                env.gap_recorded = True
            except OSError:
                env.warnings.append("gap_log_failed")
        return {
            "answer": answer,
            "citations": result.citations,
            "abstained": abstained,
            "abstain_reason": reason,
            "used_concepts": result.used_concepts,
            "concepts_found": kq["concepts_found"],
            "concepts": concepts,
            "envelope": env.model_dump(mode="json"),
        }

    async def get_evidence(self, chunk_id: str) -> dict[str, Any]:
        """Read original evidence text for a SourceChunk, falling back to preview."""
        self._assert_started()
        node = await self._graph.get_node(NodeLabel.SOURCE_CHUNK, "chunk_id", chunk_id)
        if node is None:
            return {
                "chunk_id": chunk_id,
                "source_id": "",
                "text": "",
                "verbatim": False,
                "section_title": "",
                "pages": "",
            }

        raw_text_hash = str(node.get("raw_text_hash") or "")
        text = None
        if raw_text_hash:
            text = self._evidence_vault.get_text(raw_text_hash)
        verbatim = text is not None
        return {
            "chunk_id": str(node.get("chunk_id") or chunk_id),
            "source_id": str(node.get("source_id") or ""),
            "text": text if text is not None else str(node.get("text_preview") or ""),
            "verbatim": verbatim,
            "section_title": str(node.get("section_title") or ""),
            "pages": str(node.get("pages") or ""),
        }

    async def recall(
        self,
        query: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Recall relevant memories.

        Simpler than query_knowledge — only searches memory, not graph.
        """
        self._assert_started()
        ctx = await self._retriever.retrieve(
            query, session_id=session_id,
            include_graph=False,
        )
        return {
            "formatted": ctx.format(),
            "count": len(ctx.memories),
            "memories": [
                {
                    "content": r.memory.content,
                    "layer": r.memory.layer.value,
                    "tier": r.memory.tier.value,
                    "relevance": r.relevance_score,
                }
                for r in ctx.memories
            ],
        }

    async def reinforce(
        self,
        concept_id: str,
        evidence: str = "",
        source: str = "conversation",
    ) -> dict[str, str]:
        """Reinforce a concept based on new evidence.

        Called when the user confirms a fact, or when a concept is used
        successfully in reasoning.
        """
        self._assert_started()
        await self._reinforcement.reinforce_concept(concept_id, evidence, source)
        return {"status": "reinforced", "concept_id": concept_id}

    async def route_and_execute(
        self,
        user_input: str,
        session_id: str = "",
    ) -> dict[str, Any]:
        """Route a user input to the best skill and execute it.

        This is the general-purpose entry point. The agent figures out
        what skill to use based on the input.
        """
        self._assert_started()

        # Route
        route_result = await self._router.route(user_input)

        if route_result.fallback:
            return {
                "routed": False,
                "message": "No matching skill found for this input.",
            }

        selected = route_result.selected
        result = await self._executor.execute(
            selected.skill_name,
            {"text": user_input, "instruction": user_input},
            session_id=session_id,
        )

        return {
            "routed": True,
            "skill": selected.skill_name,
            "match_reason": selected.match_reason,
            "score": selected.score,
            "execution": {
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "duration": result.duration_seconds,
            },
        }

    # ── Session management ──

    async def start_session(self, session_id: str | None = None) -> str:
        """Start a new episodic session."""
        self._assert_started()
        return await self._episodic.start_session(session_id)

    async def end_session(self, summary: str = "", session_id: str | None = None) -> str | None:
        """End a session with an optional summary.

        session_id: defaults to the ambient "currently active" session (unchanged behavior).
        Pass explicitly to end a specific session other than whichever one is currently ambient
        — see EpisodicMemory.end_session's docstring (memory/episodic.py) for why this matters
        when sessions can overlap.
        """
        self._assert_started()
        return await self._episodic.end_session(summary, session_id=session_id)

    async def record_event(
        self, event_type: str, content: str, session_id: str | None = None,
    ) -> str:
        """Record an event in a session.

        session_id: defaults to the ambient "currently active" session (unchanged behavior).
        Pass explicitly to attribute this event to a specific session other than whichever one
        is currently ambient — see EpisodicMemory.record_event's docstring for why this matters
        when sessions can overlap.
        """
        self._assert_started()
        return await self._episodic.record_event(
            EpisodicEvent(event_type=event_type, content=content), session_id=session_id,
        )

    # ── Entity management ──

    async def upsert_entity(
        self,
        entity_id: str,
        entity_type: str,
        name: str,
        summary: str = "",
    ) -> str:
        """Create or update a semantic entity."""
        self._assert_started()
        return await self._semantic.upsert_entity(entity_id, entity_type, name, summary)

    async def record_lesson(self, lesson: str, tags: list[str] | None = None) -> str:
        """Record a durable lesson."""
        self._assert_started()
        return await self._semantic.record_lesson(lesson, tags)

    # ── Design reasoning ──

    async def record_hypothesis(
        self,
        statement: str,
        assumptions: list[str] | None = None,
        test_plan: str = "",
        related_concepts: list[str] | None = None,
    ) -> str:
        """Record a testable hypothesis linked to concepts."""
        self._assert_started()
        from datetime import datetime
        hid = f"hyp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        props: dict[str, Any] = {
            "hypothesis_id": hid,
            "statement": statement,
            "status": "open",
            "confidence": 0.5,
            "test_plan": test_plan,
            "created_at": datetime.now().isoformat(),
        }
        if assumptions:
            props["assumptions"] = assumptions
        edges = [
            {
                "source_label": NodeLabel.HYPOTHESIS,
                "source_id_field": "hypothesis_id",
                "source_id_value": hid,
                "target_label": NodeLabel.CONCEPT,
                "target_id_field": "concept_id",
                "target_id_value": cid,
                "rel_type": RelType.RELATES_TO,
                "properties": EdgeProperties(
                    rationale=f"Hypothesis relates to concept {cid}",
                    confidence=0.5,
                    created_by="human",
                ).model_dump(),
            }
            for cid in (related_concepts or [])
        ]
        await self._graph.write_batch(
            nodes=[{
                "label": NodeLabel.HYPOTHESIS,
                "id_field": "hypothesis_id",
                "id_value": hid,
                "properties": props,
            }],
            edges=edges,
        )
        self._journal.log("record_hypothesis", hypothesis_id=hid, statement=statement,
                          related_concepts=related_concepts or [])
        return hid

    async def record_decision(
        self,
        choice: str,
        alternatives: list[str] | None = None,
        rationale: str = "",
        constraints: list[str] | None = None,
        related_concepts: list[str] | None = None,
        motivated_by: list[str] | None = None,
    ) -> str:
        """Record a design decision with rationale and alternatives."""
        self._assert_started()
        from datetime import datetime
        did = f"dec_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        props: dict[str, Any] = {
            "decision_id": did,
            "choice": choice,
            "rationale": rationale,
            "status": "active",
            "created_at": datetime.now().isoformat(),
        }
        if alternatives:
            props["alternatives"] = alternatives
        if constraints:
            props["constraints"] = constraints
        edges = [
            {
                "source_label": NodeLabel.CONCEPT,
                "source_id_field": "concept_id",
                "source_id_value": cid,
                "target_label": NodeLabel.DESIGN_DECISION,
                "target_id_field": "decision_id",
                "target_id_value": did,
                "rel_type": RelType.DECIDED_BY,
                "properties": EdgeProperties(
                    rationale=f"Design decision for {cid}",
                    confidence=0.8,
                    created_by="human",
                ).model_dump(),
            }
            for cid in (related_concepts or [])
        ]
        edges += [
            {
                "source_label": NodeLabel.DESIGN_DECISION,
                "source_id_field": "decision_id",
                "source_id_value": did,
                "target_label": NodeLabel.HYPOTHESIS,
                "target_id_field": "hypothesis_id",
                "target_id_value": mid,
                "rel_type": RelType.MOTIVATED_BY,
                "properties": EdgeProperties(
                    rationale=f"Decision motivated by {mid}",
                    confidence=0.8,
                    created_by="human",
                ).model_dump(),
            }
            for mid in (motivated_by or [])
        ]
        await self._graph.write_batch(
            nodes=[{
                "label": NodeLabel.DESIGN_DECISION,
                "id_field": "decision_id",
                "id_value": did,
                "properties": props,
            }],
            edges=edges,
        )
        self._journal.log("record_decision", decision_id=did, choice=choice,
                          rationale=rationale)
        return did

    async def record_bench_result(
        self,
        bench_type: str = "simulation",
        setup: str = "",
        metric: str = "",
        corner: str = "",
        conclusion: str = "",
        tests_hypothesis: str = "",
        confirms: bool = True,
    ) -> str:
        """Record a bench result, optionally confirming/falsifying a hypothesis."""
        self._assert_started()
        from datetime import datetime
        bid = f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        props: dict[str, Any] = {
            "bench_id": bid,
            "bench_type": bench_type,
            "setup": setup,
            "metric": metric,
            "corner": corner,
            "conclusion": conclusion,
            "created_at": datetime.now().isoformat(),
        }
        nodes = [{
            "label": NodeLabel.BENCH_RESULT,
            "id_field": "bench_id",
            "id_value": bid,
            "properties": props,
        }]
        updates: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        if tests_hypothesis:
            edges.append({
                "source_label": NodeLabel.BENCH_RESULT,
                "source_id_field": "bench_id",
                "source_id_value": bid,
                "target_label": NodeLabel.HYPOTHESIS,
                "target_id_field": "hypothesis_id",
                "target_id_value": tests_hypothesis,
                "rel_type": RelType.TESTS_HYPOTHESIS,
                "properties": EdgeProperties(
                    rationale=conclusion or "Bench tests hypothesis",
                    confidence=0.9,
                    created_by="human",
                ).model_dump(),
            })
            # MATCH-only update: never fabricate a hypothesis from a bad id.
            updates.append({
                "label": NodeLabel.HYPOTHESIS,
                "id_field": "hypothesis_id",
                "id_value": tests_hypothesis,
                "properties": {"status": "confirmed" if confirms else "falsified"},
            })
            if not confirms:
                edges.append({
                    "source_label": NodeLabel.HYPOTHESIS,
                    "source_id_field": "hypothesis_id",
                    "source_id_value": tests_hypothesis,
                    "target_label": NodeLabel.BENCH_RESULT,
                    "target_id_field": "bench_id",
                    "target_id_value": bid,
                    "rel_type": RelType.FALSIFIED_BY,
                    "properties": EdgeProperties(
                        rationale=conclusion or "Hypothesis falsified by bench result",
                        confidence=0.9,
                        created_by="human",
                    ).model_dump(),
                })
        await self._graph.write_batch(nodes=nodes, updates=updates, edges=edges)
        self._journal.log("record_bench_result", bench_id=bid, bench_type=bench_type,
                          tests_hypothesis=tests_hypothesis or "", confirms=confirms)
        return bid

    async def list_open_hypotheses(self, limit: int = 10) -> list[dict[str, Any]]:
        """List open hypotheses ordered by confidence."""
        self._assert_started()
        query = """
        MATCH (h:Hypothesis {status: 'open'})
        OPTIONAL MATCH (h)-[:RELATES_TO]->(c:Concept)
        WITH h, collect(c.canonical_name) AS concept_names
        RETURN h.hypothesis_id AS id, h.statement AS statement,
               h.confidence AS confidence, h.test_plan AS test_plan,
               h.created_at AS created_at, concept_names
        ORDER BY h.confidence DESC
        LIMIT $limit
        """
        return await self._graph.run_read_query(query, {"limit": limit})

    # ── Learner Model (S5, ADR-044 D4; docs/specs/S5_LEARNER_MODEL_DESIGN.md) ──
    #
    # Home-only (never registered in server/mcp_server.py's READONLY_TOOLS — §4.1). A fully
    # separate subsystem from Design Reasoning above and from memory/: Learner/Assessment are
    # new NodeLabels (schema.py), not MemoryEntry records (§2's Option A, a conscious departure
    # from D4's literal "build on the memory subsystem" wording — see the design doc for why).

    async def _resolve_assessment_target(
        self, learner_id: str, target_id: str
    ) -> tuple[NodeLabel | None, int, float | None]:
        """Resolve target_id's NodeLabel and read any existing UNDERSTANDS roll-up in one
        round trip.

        ``target`` is deliberately label-agnostic in the schema (§2/§6.1: it connects to
        whatever knowledge node was assessed — Concept, Regularity, ClaimCard, ... — the same
        way DEPENDS_ON/RELATES_TO already connect many label pairs) rather than enumerated, so
        this generically matches any node carrying target_id under an ``*_id``-suffixed
        property instead of requiring the caller to name the label (record_assessment's/
        get_learner_state's signatures have no such param). Verified against the live graph
        (read-only) that this resolves both a Concept id and a Regularity id correctly.

        Returns ``(None, 0, None)`` if no node anywhere carries target_id — callers must treat
        that as "do not fabricate an edge" (mirrors record_bench_result's tests_hypothesis
        handling, §1.5).
        """
        rows = await self._graph.run_read_query(
            """
            MATCH (t) WHERE $target_id IN [k IN keys(t) WHERE k ENDS WITH '_id' | t[k]]
            OPTIONAL MATCH (:Learner {learner_id: $learner_id})-[u:UNDERSTANDS]->(t)
            RETURN labels(t)[0] AS label, u.assessment_count AS assessment_count,
                   u.confidence AS confidence
            LIMIT 1
            """,
            {"learner_id": learner_id, "target_id": target_id},
        )
        if not rows:
            return None, 0, None
        row = rows[0]
        try:
            label = NodeLabel(row["label"])
        except ValueError:
            return None, 0, None
        return label, int(row.get("assessment_count") or 0), row.get("confidence")

    async def record_assessment(
        self,
        learner_id: str,
        target_id: str,
        verdict: str,
        evidence: str = "",
    ) -> str:
        """Record an explicit assessment verdict for a learner's understanding of a graph node.

        Mirrors record_bench_result's write_batch + MATCH-only-update shape exactly (§1.5):
        writes an immutable Assessment event node (-[:ASSESSES]-> target) and rolls the outcome
        up onto a Learner -[:UNDERSTANDS]-> target edge (confidence, status, last_assessed,
        assessment_count — MATCH-only-updated). A target_id that doesn't resolve to any node is
        a silent no-op on both edges (never fabricates the thing it's about) — the Assessment
        event itself is still recorded, exactly like record_bench_result still records a
        BenchResult for an unknown tests_hypothesis id.

        The verdict->confidence mapping below is an implementation judgment call the design doc
        leaves open (it locks the UNDERSTANDS edge's property set, not its arithmetic): a small,
        explicit, monotonic table rather than invented prior-blending math, so it stays easy to
        audit and retune once real usage data exists (§7's open question).

        Misconceptions (§3.3 Option A, locked): verdict="misconception" plus a description in
        `evidence` — nothing is ever written onto or between Concept/CircuitTopology/Regularity
        nodes themselves.
        """
        self._assert_started()
        from datetime import datetime

        aid = f"assess_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        now = datetime.now().isoformat()

        nodes: list[dict[str, Any]] = [
            {
                "label": NodeLabel.LEARNER,
                "id_field": "learner_id",
                "id_value": learner_id,
                "properties": {"learner_id": learner_id},
            },
            {
                "label": NodeLabel.ASSESSMENT,
                "id_field": "assessment_id",
                "id_value": aid,
                "properties": {
                    "assessment_id": aid,
                    "learner_id": learner_id,
                    "verdict": verdict,
                    "evidence": evidence,
                    "created_at": now,
                },
            },
        ]
        edges: list[dict[str, Any]] = []

        target_label, prior_count, _prior_confidence = await self._resolve_assessment_target(
            learner_id, target_id
        )
        if target_label is not None:
            target_id_field = self._graph._id_field_for_label(target_label)
            edges.append({
                "source_label": NodeLabel.ASSESSMENT,
                "source_id_field": "assessment_id",
                "source_id_value": aid,
                "target_label": target_label,
                "target_id_field": target_id_field,
                "target_id_value": target_id,
                "rel_type": RelType.ASSESSES,
                "properties": {"rationale": f"Assessment {aid} of {target_id}: {verdict}"},
            })
            edges.append({
                "source_label": NodeLabel.LEARNER,
                "source_id_field": "learner_id",
                "source_id_value": learner_id,
                "target_label": target_label,
                "target_id_field": target_id_field,
                "target_id_value": target_id,
                "rel_type": RelType.UNDERSTANDS,
                "properties": {
                    "confidence": _VERDICT_CONFIDENCE.get(
                        verdict.strip().lower(), _DEFAULT_VERDICT_CONFIDENCE
                    ),
                    "status": verdict,
                    "last_assessed": now,
                    "assessment_count": prior_count + 1,
                },
            })

        await self._graph.write_batch(nodes=nodes, edges=edges)
        self._journal.log("record_assessment", assessment_id=aid, learner_id=learner_id,
                          target_id=target_id, verdict=verdict)
        return aid

    async def get_learner_state(
        self, learner_id: str, target_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Read a learner's rolled-up UNDERSTANDS state — one entry per target assessed.

        Read-only. Structural ingredient for sequencing, not a sequencer: this plus ordinary
        query_knowledge-style reads of EVOLVES_TO/SUB_BLOCK/knowledge_layer are what a caller
        (Hermes) combines to decide "what's next" — openclaw-brain does not compute that itself
        (§5.2, locked: brain=evidence/Hermes=narrative+sequencing boundary preserved).

        Args:
            learner_id: the learner to read.
            target_id: if given, scope to just that target's edge.
        """
        self._assert_started()
        query = """
        MATCH (l:Learner {learner_id: $learner_id})-[u:UNDERSTANDS]->(target)
        RETURN labels(target)[0] AS target_label, properties(target) AS target_props,
               properties(u) AS understands
        ORDER BY u.last_assessed DESC
        """
        rows = await self._graph.run_read_query(query, {"learner_id": learner_id})
        results: list[dict[str, Any]] = []
        for row in rows:
            label_str = row.get("target_label")
            target_props = row.get("target_props") or {}
            try:
                id_field = self._graph._id_field_for_label(NodeLabel(label_str))
            except ValueError:
                id_field = None
            resolved_target_id = target_props.get(id_field) if id_field else None
            if target_id is not None and resolved_target_id != target_id:
                continue
            understands = row.get("understands") or {}
            results.append({
                "target_id": resolved_target_id,
                "target_label": label_str,
                "target_name": target_props.get("canonical_name") or target_props.get("name") or "",
                "confidence": understands.get("confidence"),
                "status": understands.get("status"),
                "last_assessed": understands.get("last_assessed"),
                "assessment_count": understands.get("assessment_count"),
            })
        return results

    # ── Knowledge Graph Curation ──

    async def merge_concepts(
        self,
        primary_id: str,
        duplicate_id: str,
    ) -> dict[str, Any]:
        """Merge a duplicate Concept node into the primary.

        All edges from/to the duplicate are re-wired to the primary.
        Gap properties (on duplicate but not on primary) are copied.
        The duplicate is deleted.

        Args:
            primary_id: The concept_id to keep.
            duplicate_id: The concept_id to absorb and delete.
        """
        self._assert_started()
        result = await self._graph.merge_concepts(primary_id, duplicate_id)
        if "error" not in result:
            self._journal.log("merge_concepts", **result)
        return result

    async def retract_node(
        self,
        node_id: str,
        label_str: str,
        reason: str = "",
        hard_delete: bool = False,
    ) -> dict[str, Any]:
        """Retract (soft-delete or hard-delete) a knowledge node.

        Soft delete: sets retracted=true — node stays in graph but is excluded
        from all searches and retrieval queries.
        Hard delete: permanently removes the node and all its edges.

        Args:
            node_id: The ID value for the node (e.g. concept_id value).
            label_str: Node type string — "Concept", "Equation", "Parameter", etc.
            reason: Human-readable reason for retraction.
            hard_delete: If True, permanently delete; default is soft-delete.
        """
        self._assert_started()
        try:
            label = NodeLabel(label_str)
        except ValueError:
            valid = [l.value for l in NodeLabel]
            return {"error": f"Unknown label '{label_str}'. Valid: {valid}"}
        found = await self._graph.retract_node(node_id, label, reason, hard_delete)
        action = "deleted" if hard_delete else "retracted"
        result = {
            "node_id": node_id,
            "label": label_str,
            "action": action,
            "found": found,
            "reason": reason,
        }
        if found:
            self._journal.log("retract_node", **result)
        return result

    # ── Maintenance ──

    async def run_promotion(self) -> dict[str, int]:
        """Run the memory promotion pipeline."""
        self._assert_started()
        return await self._promotion.run()

    async def run_decay(self, days_threshold: int = 30) -> int:
        """Apply confidence decay to stale concepts."""
        self._assert_started()
        return await self._reinforcement.apply_decay(days_threshold)

    async def find_bridges(self, limit: int = 10) -> list[dict[str, Any]]:
        """Find potential cross-domain bridges."""
        self._assert_started()
        return await self._reinforcement.find_potential_bridges(limit)

    async def get_stats(self) -> dict[str, Any]:
        """Get system-wide statistics."""
        self._assert_started()
        graph_stats = await self._graph.get_stats()
        memory_stats = await self._memory_store.get_stats()
        reinforcement_stats = await self._reinforcement.get_reinforcement_stats()

        return {
            "graph": graph_stats,
            "memory": memory_stats,
            "reinforcement": reinforcement_stats,
            "skills": {
                "registered": self._registry.count,
                "available": [s.name for s in self._registry.list_skills()],
            },
            "models": self._llm_provider.list_available(),
        }

    # ── Skill registration ──

    def register_skill(self, definition, handler) -> None:
        """Register a custom skill (for extensibility)."""
        self._assert_started()
        self._registry.register(definition, handler)

    # ── Executable-circuit substrate ──

    async def query_executable(self, topology_class: str | None = None) -> dict[str, Any]:
        """Read the executable-circuit substrate: simulation-verdicted specimens and their
        claim-cards, linked to the existing graph (REALIZES a CircuitTopology, GROUNDS a Parameter).

        This is the EVIDENCE layer a teaching agent narrates from: each claim-card's verdict was
        produced by a deterministic oracle re-running the ngspice sim; the QUANT assertion is
        certified, the mechanism narrative is interpretive (never oracle-certified). Read-only.

        Each claim additively carries `law_ids` (law-tier I2, docs/superpowers/specs/2026-07-04-
        law-tier-graph-representation.md §5 bullet 1): the ids of any `law`-tier Regularity
        SUPPORTED_BY this claim-card — ids only, `why_law(law_id)` renders the full evidence. One
        round trip (a pattern comprehension per claim, no N+1)."""
        self._assert_started()
        rows = await self._graph.run_read_query(
            """
            MATCH (s:Specimen)
            WHERE $tclass IS NULL OR s.topology_class = $tclass
            OPTIONAL MATCH (s)-[:REALIZES]->(t:CircuitTopology)
            OPTIONAL MATCH (s)-[:HAS_CLAIM]->(c:ClaimCard)
            OPTIONAL MATCH (c)-[:GROUNDS]->(p:Parameter)
            RETURN s.spec_id AS spec_id, s.topology_class AS topology_class,
                   s.pdk AS pdk, s.tool AS tool, t.canonical_name AS realizes,
                   collect(CASE WHEN c IS NULL THEN NULL ELSE {
                       claim_id: c.claim_id, claim: c.claim, knob: c.knob, metric: c.metric,
                       verdict: c.verdict, narrative: c.narrative,
                       engine: c.engine, basis: c.basis, dominant_risk_untested: c.dominant_risk_untested,
                       conditions: {corner: c.corner, temp_c: c.temp_c, vdd: c.vdd},
                       grounds: p.canonical_name,
                       law_ids: [(reg:Regularity)-[:SUPPORTED_BY]->(c) | reg.law_id]} END) AS claims
            ORDER BY s.topology_class
            """,
            {"tclass": topology_class},
        )
        specimens = [
            {
                "spec_id": r["spec_id"], "topology_class": r["topology_class"],
                "pdk": r.get("pdk"), "tool": r.get("tool"), "realizes": r.get("realizes"),
                "claims": [c for c in (r.get("claims") or []) if c],
            }
            for r in rows
        ]
        return {"topology_class": topology_class, "count": len(specimens), "specimens": specimens}

    async def project_executable(self, apply: bool = False, corpus_dir: str | None = None) -> dict[str, Any]:
        """Build + project ALL validated seed specimens across the executable-circuit substrate:
        the 15 analog seed recipes (ngspice), the 3 digital pedagogy recipes (iverilog), and the 4
        Stat-QT statistical/corner recipes (ngspice Monte Carlo) — `seed_recipes()` +
        `digital_seed_recipes()` + `statistical_seed_recipes()`, 22 recipes total.

        Each recipe's engine is resolved (`build.engine`, defaulting to the template's registered
        engine) and dispatched to that engine's runner from the ENGINES registry (engines.py) — no
        engine is hardcoded. A recipe whose engine's runner is unavailable (e.g. the ngspice sim
        image is missing but iverilog is present) is SKIPPED with a per-recipe report entry
        ({topology_class, engine, reason}) rather than failing the whole batch.

        Runs the real simulator to produce verdicted specimens, then projects them onto the graph —
        additive Specimen/ClaimCard nodes + REALIZES/GROUNDS links (NO-PHANTOM). Dry-run by default
        (apply=False simulates + reports but writes nothing); apply=True writes AND persists the
        verdicted specimens to the git-corpus SSOT (`[executable] corpus_dir` in config, overridable
        via `corpus_dir`). Long-running."""
        self._assert_started()
        from openclaw_brain.knowledge.executable.corpus import SpecimenCorpus, compute_spec_id
        from openclaw_brain.knowledge.executable.engines import ENGINES, engine_for_template, runner_for_engine
        from openclaw_brain.knowledge.executable.executor import run_recipe
        from openclaw_brain.knowledge.executable.projection import GraphProjector
        from openclaw_brain.knowledge.executable.recipe import capability_for
        from openclaw_brain.knowledge.executable.resolver import GraphResolver
        from openclaw_brain.knowledge.executable.seeds import (
            digital_seed_recipes, seed_recipes, statistical_seed_recipes,
        )

        projector = GraphProjector(self._graph, GraphResolver(self._graph).resolve)
        corpus = None
        if apply:
            cdir = corpus_dir or str(self._config.executable.corpus_path)
            corpus = SpecimenCorpus(cdir)

        out: dict[str, Any] = {"apply": apply, "specimens": [], "skipped": []}
        all_recipes = seed_recipes() + digital_seed_recipes() + statistical_seed_recipes()
        for recipe in all_recipes:
            template_ref = (recipe.build or {}).get("template_ref") or \
                capability_for(recipe.topology_class).template_ref
            engine = (recipe.build or {}).get("engine") or engine_for_template(template_ref)
            espec = ENGINES.get(engine)
            if espec is None:
                out["skipped"].append({"topology_class": recipe.topology_class, "engine": engine,
                                       "reason": f"unknown engine {engine!r}"})
                continue
            if espec.runner is None:
                out["skipped"].append({"topology_class": recipe.topology_class, "engine": engine,
                                       "reason": "engine is render-only (no runner)"})
                continue
            if not runner_for_engine(engine, self._config).available():
                out["skipped"].append({"topology_class": recipe.topology_class, "engine": engine,
                                       "reason": f"{engine} runner unavailable "
                                                 "(sim image/tooling not present)"})
                continue

            result = run_recipe(recipe, runner_for_engine(engine, self._config),
                                corpus=corpus, config=self._config)
            spec = result.specimen
            entry: dict[str, Any] = {
                "topology_class": recipe.topology_class,
                "engine": engine,
                "spec_id": spec.spec_id or compute_spec_id(spec),
                "verdicts": {c.id: (c.verdict.value if c.verdict else None) for c in result.claim_cards},
            }
            if apply:
                entry["written"] = await projector.project(spec)
                self._journal.log("project_executable", topology_class=recipe.topology_class,
                                  **entry["written"])
            out["specimens"].append(entry)
        return out

    async def retract_executable(self, spec_id: str) -> dict[str, Any]:
        """Reverse a project_executable apply: DETACH DELETE the Specimen + its ClaimCards (and their
        REALIZES/HAS_CLAIM/GROUNDS edges) for one spec_id. Existing graph nodes are untouched."""
        self._assert_started()
        from openclaw_brain.knowledge.executable.projection import retract_projection

        stats = await retract_projection(self._graph, spec_id)
        self._journal.log("retract_executable", spec_id=spec_id, **stats)
        return {"spec_id": spec_id, **stats}

    async def grow_executable(
        self,
        *,
        sources: list[str] | None = None,
        max_recipes: int = 5,
        max_per_class: int = 4,
        topology_class: str | None = None,
        apply: bool = False,
        corpus_dir: str | None = None,
    ) -> dict[str, Any]:
        """Corpus growth automation (D3 AUTO lane): author + simulate NEW claim-cards for registered
        topologies, sourced from coverage-gap enumeration over the template registry, the ingest
        growth_queue, and the TOPOLOGY_BACKLOG.md curriculum (`sources`, default all three) — then
        route each verdict (VERIFIED/REFUTED-family projects on apply=True; FLAGGED/REJECTED never
        project, they land in the report's `triage` for post-mortem).

        This is fully automated (author -> run -> judge -> corpus.store -> project, no human in the
        loop) because it only ever authors recipes against ALREADY-VALIDATED templates
        (`capability_for()` pre-gates; `enforce_executable()` snaps `template_ref`) — a NEW SPICE
        template is never auto-admitted (that stays a hand-validated `templates.py` change); an
        unmatched high-signal topology only ever surfaces via `review_lane_pending`. Digital
        (iverilog) classes are excluded from AUTO authoring in this wave.

        Dry-run by default (apply=False plans + authors + simulates + reports verdicts but writes
        nothing); apply=True writes the graph projection AND persists to the git-corpus SSOT
        (`[executable] corpus_dir`, overridable via `corpus_dir`). LONG-RUNNING: authoring calls an
        LLM per target and simulation takes real wall-clock time."""
        self._assert_started()
        from openclaw_brain.knowledge.executable import growth
        from openclaw_brain.knowledge.executable.corpus import SpecimenCorpus
        from openclaw_brain.knowledge.executable.projection import GraphProjector
        from openclaw_brain.knowledge.executable.resolver import GraphResolver

        plan = await growth.plan_growth(
            self._graph,
            sources=sources or ["coverage", "queue", "backlog"],
            max_recipes=max_recipes,
            max_new_cards_per_class=max_per_class,
            topology_class=topology_class,
            state_dir=self._config.state_path,
            apply=apply,
        )

        corpus = None
        projector = None
        if apply:
            cdir = corpus_dir or str(self._config.executable.corpus_path)
            corpus = SpecimenCorpus(cdir)
            projector = GraphProjector(self._graph, GraphResolver(self._graph).resolve)

        deps = growth.GrowthDeps(
            store=self._graph,
            model_chain=self._llm_provider.get_chain("extraction"),
            resilience_config=self._config.resilience,
            auth_refresh=None,
            vault=self._evidence_vault,
            corpus=corpus,
            projector=projector,
            journal=self._journal,
            state_dir=self._config.state_path,
            config=self._config,
        )
        report = await growth.execute_growth(deps, plan, apply=apply)
        return report.to_dict()

    async def why(self, claim_card_id: str) -> dict[str, Any]:
        """Grounding for one claim-card — the evidence behind a citation. Read-only.
        Returns the graph-stored fields (knob/metric/verdict/narrative/R1 conditions + the topology it
        realizes and the parameter it grounds), or {found: False} if the id is unknown.

        Additive (law-tier I2, docs/superpowers/specs/2026-07-04-law-tier-graph-representation.md §5
        bullet 1): "laws" carries every `law`-tier Regularity this card SUPPORTED_BY-links to, as
        light {law_id, status} refs — `why_law(law_id)` renders the full cross-PDK evidence. One
        round trip (a single query, aggregated)."""
        self._assert_started()
        rows = await self._graph.run_read_query(
            """
            MATCH (c:ClaimCard {claim_id: $cid})
            OPTIONAL MATCH (s:Specimen)-[:HAS_CLAIM]->(c)
            OPTIONAL MATCH (c)-[:GROUNDS]->(p:Parameter)
            OPTIONAL MATCH (reg:Regularity)-[:SUPPORTED_BY]->(c)
            WITH c, s, p, collect(DISTINCT CASE WHEN reg IS NULL THEN NULL
                                   ELSE {law_id: reg.law_id, status: reg.status} END) AS laws
            RETURN c.claim AS claim, c.knob AS knob, c.metric AS metric, c.verdict AS verdict,
                   c.quant_kind AS quant_kind, c.narrative AS narrative,
                   c.corner AS corner, c.temp_c AS temp_c, c.vdd AS vdd,
                   c.engine AS engine, c.basis AS basis, c.scope AS scope,
                   c.dominant_risk_untested AS dominant_risk_untested,
                   s.spec_id AS spec_id, s.topology_class AS topology_class, p.canonical_name AS grounds,
                   laws
            LIMIT 1
            """,
            {"cid": claim_card_id},
        )
        if not rows:
            return {"claim_card_id": claim_card_id, "found": False}
        r = rows[0]
        scope = json.loads(r["scope"]) if r.get("scope") else {}
        inline = _scope_inline(scope)
        laws = [law for law in (r.get("laws") or []) if law]
        result: dict[str, Any] = {
            "claim_card_id": claim_card_id, "found": True,
            "claim": r["claim"], "knob": r["knob"], "metric": r["metric"], "verdict": r["verdict"],
            # quant_kind tells the narrator WHAT the oracle certified: a direction/invariance (shape)
            # vs a value/elasticity (scalar). A magnitude is only a certified fact under value/elasticity.
            "quant_kind": r["quant_kind"],
            "narrative": r["narrative"], "grounds": r["grounds"],
            "conditions": {"corner": r["corner"], "temp_c": r["temp_c"], "vdd": r["vdd"]},
            # scope-honesty (ADR 4-5): the verdict rendered INLINE with its scope (undetachable), the
            # per-engine epistemic basis, and — when present — the untested dominant silicon/functional risk.
            "engine": r.get("engine"), "basis": r.get("basis"), "scope": scope,
            "verdict_scoped": f'{r["verdict"]}@{inline}' if r["verdict"] else None,
            "dominant_risk_untested": r.get("dominant_risk_untested"),
            "spec_id": r["spec_id"], "topology_class": r["topology_class"],
            "laws": laws,
        }
        self._attach_schematic(result, claim_card_id, r.get("topology_class"), r.get("spec_id"))
        return result

    def _attach_schematic(self, result: dict[str, Any], claim_card_id: str,
                           topology_class: str | None, spec_id: str | None) -> None:
        """Best-effort: add `schematic_path` to a `why()` result when the card's specimen is on
        disk AND its topology_class has a hand-authored schematic template (pilot: 3 of 19
        families). Lazy-renders + caches under `[storage].state_dir/schematics/`. Never raises —
        an unsupported family silently omits the field (not an error, see
        `schematic.UnsupportedTopologyError`'s docstring); a genuine connectivity mismatch between
        the corpus netlist and the template omits the field too, but is logged loudly (P9:
        degrade, never silently) rather than shipping a picture that might not match the circuit
        the claim was actually measured on."""
        if not topology_class or not spec_id:
            return
        from openclaw_brain.knowledge.executable.corpus import SpecimenCorpus
        from openclaw_brain.knowledge.executable.schematic import (
            SchematicMismatchError, UnsupportedTopologyError, is_supported, render_specimen,
        )

        if not is_supported(topology_class):
            return
        corpus = SpecimenCorpus(str(self._config.executable.corpus_path))
        spec_dir = Path(corpus.spec_dir(topology_class, spec_id))
        if not (spec_dir / "meta.yaml").is_file():
            return  # corpus doesn't have this specimen on disk (e.g. projected without --apply)
        try:
            out_dir = self._config.state_path / "schematics"
            result["schematic_path"] = str(render_specimen(
                spec_dir, out_dir, mos_style=self._config.figures.schematic_mos_style))
        except UnsupportedTopologyError:
            pass  # is_supported() already gated this; kept defensive, never fatal to why()
        except SchematicMismatchError as e:
            logger.warning(
                "why(%s): schematic connectivity self-check failed for topology_class=%r "
                "spec_id=%r — omitting schematic_path (verdict/narrative are unaffected): %s",
                claim_card_id, topology_class, spec_id, e,
            )

    async def why_law(self, law_id: str) -> dict[str, Any]:
        """Grounding for one law-tier Regularity — the cross-PDK replicated evidence behind a
        citation to a `law_id` instead of a single claim-card (docs/superpowers/specs/2026-07-04-
        law-tier-graph-representation.md §5 bullet 1). Read-only, no journal writes.

        Renders every member PDK's verdict SCOPE-HONESTLY and undetachably: a member whose
        ClaimCard has actually been projected onto the graph (a one-hop SUPPORTED_BY edge, matched
        to its PDK via the card's own Specimen) renders from that card's REAL scope, exactly like
        `why()` does (`_scope_inline`, reused not duplicated); a member recorded only in
        `member_summary` (not yet projected — every non-sky130A member of the current law set)
        renders "<verdict>@<pdk> (report-only: member card not yet projected)", honestly. The law
        NEVER renders as a bare universal — `statement` already carries the member list, and every
        member entry repeats its own PDK. A Pelgrom-type law's per-PDK `fitted_exponent` is member
        DATA (never folded into `statement`, magnitude-free discipline). `status != "law"`
        (process_scoped/demoted) surfaces `status_note` loudly, and a node that was ever demoted
        carries its prior-status `history` forward permanently (R5 — never silent).

        Returns {law_id, found, status, statement, topology_class, metric, knob, quant_kind,
        members:[{pdk, verdict, verdict_scoped, note?, fitted_exponent?, card_id?, card_projected}],
        status_note? (when status != "law"), history? (when the node was ever demoted),
        supported_by_count, derived_from}, or {law_id, found: False} if the id is unknown."""
        self._assert_started()
        rows = await self._graph.run_read_query(
            """
            MATCH (r:Regularity {law_id: $lid})
            OPTIONAL MATCH (r)-[:SUPPORTED_BY]->(c:ClaimCard)<-[:HAS_CLAIM]-(s:Specimen)
            RETURN r.status AS status, r.status_note AS status_note, r.statement AS statement,
                   r.topology_class AS topology_class, r.metric AS metric, r.knob AS knob,
                   r.quant_kind AS quant_kind, r.pdks AS pdks, r.member_summary AS member_summary,
                   r.derived_from AS derived_from,
                   collect(DISTINCT CASE WHEN c IS NULL THEN NULL
                           ELSE {pdk: s.pdk, claim_id: c.claim_id, verdict: c.verdict, scope: c.scope}
                           END) AS cards
            LIMIT 1
            """,
            {"lid": law_id},
        )
        if not rows:
            return {"law_id": law_id, "found": False}
        r = rows[0]
        member_summary: dict[str, Any] = json.loads(r["member_summary"]) if r.get("member_summary") else {}
        history = member_summary.pop("_history", None)
        pdks: list[str] = r.get("pdks") or []
        raw_cards = [c for c in (r.get("cards") or []) if c]

        # NO-PHANTOM member->card matching: a member's card, if projected, is matched by PDK (via
        # the card's own Specimen); deterministic-first (lexicographically-smallest claim_id) on the
        # rare case a PDK carries more than one card, mirroring laws.py's own `_find_member_card`.
        cards_by_pdk: dict[str, dict] = {}
        for entry in sorted(raw_cards, key=lambda e: e.get("claim_id") or ""):
            cards_by_pdk.setdefault(entry["pdk"], entry)

        members: list[dict[str, Any]] = []
        for pdk in pdks:
            info = member_summary.get(pdk, {})
            verdict = info.get("verdict")
            member: dict[str, Any] = {"pdk": pdk, "verdict": verdict}
            if info.get("note"):
                member["note"] = info["note"]
            if "fitted_exponent" in info:
                member["fitted_exponent"] = info["fitted_exponent"]
            card = cards_by_pdk.get(pdk)
            if card:
                card_scope = json.loads(card["scope"]) if card.get("scope") else {}
                inline = _scope_inline(card_scope)
                member["card_id"] = card["claim_id"]
                member["card_projected"] = True
                member["verdict_scoped"] = f'{card["verdict"]}@{inline}' if card.get("verdict") else None
            else:
                member["card_projected"] = False
                tag = verdict or "UNKNOWN"
                member["verdict_scoped"] = f"{tag}@{pdk} (report-only: member card not yet projected)"
            members.append(member)

        result: dict[str, Any] = {
            "law_id": law_id, "found": True, "status": r["status"], "statement": r["statement"],
            "topology_class": r["topology_class"], "metric": r["metric"], "knob": r["knob"],
            "quant_kind": r["quant_kind"], "members": members,
            "supported_by_count": len(raw_cards),
            "derived_from": json.loads(r["derived_from"]) if r.get("derived_from") else {},
        }
        if r["status"] != "law":
            result["status_note"] = r.get("status_note")
        if history:
            result["history"] = history
        return result

    async def audit_citations(self, lesson_plan: dict[str, Any]) -> dict[str, Any]:
        """Advisory self-check of a Hermes LessonPlan: every 'certified' claim must cite an existing
        certified claim-card (or, law-tier I2: a `law`-status Regularity, by its `law_id`); every
        claim must carry a tier label. Deterministic, NOT a hard gate — it judges only the
        certified-vs-interpretive boundary (+ the law-citation rules of docs/superpowers/specs/
        2026-07-04-law-tier-graph-representation.md §5 bullet 2), never lesson quality/ordering/
        coverage. Returns an AuditReport dict."""
        self._assert_started()
        from openclaw_brain.knowledge.executable.lesson import LessonPlan, audit_citations as _audit

        plan = LessonPlan.model_validate(lesson_plan)
        # Pre-resolve every cited id from the graph (the pure audit takes a SYNC resolver). A single
        # OPTIONAL-MATCH query checks BOTH id spaces (ClaimCard.claim_id / Regularity.law_id) in one
        # round trip per distinct citation — the two id formats never collide (claim_card ids carry
        # a ":", law_ids are a bare 40-hex sha1), so at most one side ever actually matches.
        cache: dict[str, dict | None] = {}
        for claim in plan.claims:
            if claim.tier == "certified" and claim.cites and claim.cites not in cache:
                rows = await self._graph.run_read_query(
                    """
                    OPTIONAL MATCH (c:ClaimCard {claim_id: $cid})
                    OPTIONAL MATCH (reg:Regularity {law_id: $cid})
                    RETURN c.verdict AS verdict, c.quant_kind AS quant_kind, c.basis AS basis,
                           c.metric AS metric, c.scope AS scope,
                           reg.status AS law_status, reg.quant_kind AS law_quant_kind,
                           reg.metric AS law_metric, reg.pdks AS law_pdks
                    LIMIT 1
                    """,
                    {"cid": claim.cites})
                r0 = rows[0] if rows else {}
                if r0.get("law_status") is not None:
                    # law-tier I2 (spec §5 bullet 2): the Regularity IS the cited thing — its
                    # `status` gates certification (R5's payoff: a demoted/process_scoped citation
                    # fails loudly) and its cross-PDK `pdks` become the audit's allowed
                    # generalization span (lesson.py's law_members-aware `_overgeneralizes` — never
                    # a parallel checker).
                    cache[claim.cites] = {
                        "kind": "law", "status": r0["law_status"],
                        "quant_kind": r0.get("law_quant_kind"), "metric": r0.get("law_metric"),
                        "pdks": r0.get("law_pdks") or [],
                    }
                elif r0.get("verdict") is not None:
                    # E3-I2: `scope` already carries `intervention`/`idealization` post-projection
                    # (executor.py stamps them into the SAME scope dict summarize_scope returns;
                    # projection.py persists the whole dict as one JSON-string property) — no new
                    # ClaimCard property, just this RETURN + the flat-key extraction below, mirroring
                    # the existing metric-field precedent (a scalar column added to the SAME query).
                    scope = json.loads(r0["scope"]) if r0.get("scope") else {}
                    cache[claim.cites] = {
                        "kind": "card", "verdict": r0["verdict"], "quant_kind": r0.get("quant_kind"),
                        "basis": r0.get("basis"), "metric": r0.get("metric"),
                        "intervention": scope.get("intervention"),
                        "idealization": scope.get("idealization"),
                    }
                else:
                    cache[claim.cites] = None
        report = _audit(plan, lambda cid: cache.get(cid))
        return report.model_dump()

    async def audit_answer(self, text: str) -> dict[str, Any]:
        """Advisory tier-honesty audit of free-form ANSWER text — the answer-surface twin of
        `audit_citations` (which guards structured LessonPlans only). Extracts every
        [cite:]/[certified:]/[assoc:] bracket id, resolves each against the graph ONCE
        (read-only), and lets the pure audit (lesson.py::audit_answer_citations) judge the tier:
        only a ClaimCard whose own verdict is in the certified band (VERIFIED/
        VERIFIED_WITH_CAVEAT) or a law-status Regularity licenses certified/verified/oracle
        language — a ClaimCard with any other verdict is `uncertified_card` (treated like
        associative for trust language, verdict named in the finding), and a Concept/Insight/
        Equation id is REAL but associative: citing one under a "Certified" heading is exactly
        the blind-eval leak this closes. Also carries the M5 scope-attribution inputs: each
        resolved row's scope columns are parsed into canonical `scope_pdks` (ClaimCard scope
        JSON / law-status Regularity licensed member set — see `_scope_pdks` below) so the pure
        audit can flag a paragraph attributing multi-PDK verification to single-PDK evidence
        (scope_overstatement). Deterministic, NOT a hard gate. Returns an AnswerAuditReport
        dict."""
        self._assert_started()
        from openclaw_brain.knowledge.executable.lesson import (
            _CERTIFIED_VERDICTS, audit_answer_citations as _audit_answer, canonical_pdk,
            extract_answer_citation_ids)

        def _scope_pdks(labels: list, row: dict) -> "list[str] | None":
            """Canonical PDK scope of one resolved row (M5; data shapes live-verified
            2026-07-22): a ClaimCard's `scope` is a JSON STRING whose "pdk" is a single name
            ("sky130"/"gf180mcuD"/"ihp-sg13g2") or null (5/84 cards) — null/unparseable -> None,
            an UNKNOWN scope is never guessed. A law-status Regularity's LICENSED set is its
            `member_summary` (JSON string {pdk: {"note", "verdict"}}) keys whose verdict is
            VERIFIED-family; fallback to the `pdks` property ONLY when member_summary is
            missing/unparseable — pdks records every PDK with ANY verdict (agree or not), so it
            over-licenses. pdks is a real Neo4j list property (all 27 nodes, live-checked
            2026-07-22); the repr-string branch is purely defensive, no live node has that
            shape. Every other label (and non-law Regularity) -> None."""
            if "ClaimCard" in labels:
                try:
                    scope = json.loads(row["scope"]) if row.get("scope") else {}
                    pdk = scope.get("pdk")
                except (TypeError, ValueError, AttributeError):
                    return None
                return [canonical_pdk(pdk)] if isinstance(pdk, str) and pdk else None
            if "Regularity" in labels and row.get("status") == "law":
                if row.get("member_summary"):
                    try:
                        members = json.loads(row["member_summary"])
                        return sorted({canonical_pdk(p) for p, m in members.items()
                                       if isinstance(m, dict)
                                       and m.get("verdict") in _CERTIFIED_VERDICTS})
                    except (TypeError, ValueError, AttributeError):
                        pass    # unparseable member_summary -> fall back to pdks
                pdks = row.get("pdks")
                if isinstance(pdks, str):
                    try:
                        pdks = ast.literal_eval(pdks)
                    except (ValueError, SyntaxError):
                        return None
                if isinstance(pdks, (list, tuple)):
                    return sorted({canonical_pdk(str(p)) for p in pdks if p})
                return None
            return None

        # Pre-resolve every DISTINCT cited id (the pure audit takes a SYNC resolver — the same
        # pre-resolve pattern audit_citations uses above). One read-only round trip per id,
        # OPTIONAL-MATCHing each label by its REAL id property (verified SSOT:
        # store.py::_id_field_for_label + projection.py::_ID_FIELD — claim_id/law_id/concept_id/
        # equation_id/insight_id/parameter_id/topology_id/principle_id/spec_id/hypothesis_id/
        # decision_id/bench_id/chunk_id). Every anchor excludes soft-retracted nodes
        # (`NOT coalesce(retracted, false)` — retract_node sets retracted=true; a retracted node
        # must resolve like a missing one, never silently lend its tier). The node's own `verdict`
        # is RETURNED so the pure audit can gate a ClaimCard's tier on it (a REFUTED card is
        # `uncertified_card`, not certified), and its scope columns (`scope` / `pdks` /
        # `member_summary` — null for labels that lack them) are RETURNED so `_scope_pdks` can
        # canonicalize the M5 scope in Python. ClaimCard ids are long
        # ("sha256:<spec-hash>:<claim>", corpus.py::compute_spec_id + projection.py), so a cited
        # id starting with "sha256:" may be a PREFIX of the full claim_id — honored ONLY when
        # (a) the prefix carries at least 12 hex chars past "sha256:" (shorter prefixes are too
        # collision-prone to trust) AND (b) it matches exactly ONE non-retracted ClaimCard (an
        # ambiguous prefix resolves to NOTHING — explicit collect/size==1, never ORDER BY+LIMIT 1
        # picking an arbitrary card). Exact matches take precedence: the prefix-matched card sits
        # LAST in the coalesce, so an id that exactly equals e.g. a Specimen.spec_id (which is a
        # strict prefix of that specimen's claim_ids) resolves as the Specimen (associative),
        # never as an arbitrary prefix-matched ClaimCard.
        cache: dict[str, dict | None] = {}
        min_prefix = len("sha256:") + 12
        for cited in extract_answer_citation_ids(text):
            rows = await self._graph.run_read_query(
                """
                OPTIONAL MATCH (ccx:ClaimCard {claim_id: $cid})
                    WHERE NOT coalesce(ccx.retracted, false)
                OPTIONAL MATCH (ccp:ClaimCard)
                    WHERE $prefix AND ccp.claim_id STARTS WITH $cid
                          AND NOT coalesce(ccp.retracted, false)
                WITH ccx, collect(DISTINCT ccp) AS prefix_matches
                WITH ccx,
                     CASE WHEN size(prefix_matches) = 1 THEN prefix_matches[0] ELSE null END AS ccp
                OPTIONAL MATCH (reg:Regularity {law_id: $cid}) WHERE NOT coalesce(reg.retracted, false)
                OPTIONAL MATCH (con:Concept {concept_id: $cid}) WHERE NOT coalesce(con.retracted, false)
                OPTIONAL MATCH (eq:Equation {equation_id: $cid}) WHERE NOT coalesce(eq.retracted, false)
                OPTIONAL MATCH (ins:Insight {insight_id: $cid}) WHERE NOT coalesce(ins.retracted, false)
                OPTIONAL MATCH (par:Parameter {parameter_id: $cid}) WHERE NOT coalesce(par.retracted, false)
                OPTIONAL MATCH (top:CircuitTopology {topology_id: $cid}) WHERE NOT coalesce(top.retracted, false)
                OPTIONAL MATCH (pri:Principle {principle_id: $cid}) WHERE NOT coalesce(pri.retracted, false)
                OPTIONAL MATCH (sp:Specimen {spec_id: $cid}) WHERE NOT coalesce(sp.retracted, false)
                OPTIONAL MATCH (hy:Hypothesis {hypothesis_id: $cid}) WHERE NOT coalesce(hy.retracted, false)
                OPTIONAL MATCH (dd:DesignDecision {decision_id: $cid}) WHERE NOT coalesce(dd.retracted, false)
                OPTIONAL MATCH (br:BenchResult {bench_id: $cid}) WHERE NOT coalesce(br.retracted, false)
                OPTIONAL MATCH (ch:SourceChunk {chunk_id: $cid}) WHERE NOT coalesce(ch.retracted, false)
                WITH coalesce(ccx, reg, con, eq, ins, par, top, pri, sp, hy, dd, br, ch, ccp) AS m
                RETURN CASE WHEN m IS NULL THEN null ELSE labels(m) END AS labels,
                       m.status AS status, m.verdict AS verdict,
                       m.scope AS scope, m.pdks AS pdks, m.member_summary AS member_summary
                LIMIT 1
                """,
                {"cid": cited,
                 "prefix": cited.startswith("sha256:") and len(cited) >= min_prefix})
            r0 = rows[0] if rows else {}
            labels = r0.get("labels")
            cache[cited] = ({"labels": labels, "status": r0.get("status"),
                             "verdict": r0.get("verdict"),
                             "scope_pdks": _scope_pdks(labels, r0)} if labels else None)
        report = _audit_answer(text, lambda cid: cache.get(cid))
        return report.model_dump()

    # ── Internal ──

    def _assert_started(self) -> None:
        # A real `if`/`raise`, not a bare `assert` — `assert` is stripped entirely under
        # `python -O`/PYTHONOPTIMIZE, which would silently disable this guard on all 31 gated
        # methods and let them proceed into bodies that assume self._graph/etc. are set (they're
        # only assigned in start()), surfacing several frames deeper as an opaque AttributeError.
        if not self._started:
            raise RuntimeError("Call agent.start() first")
