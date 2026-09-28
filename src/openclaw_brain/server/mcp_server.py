"""MCP Server for openclaw-brain.

Exposes BrainAgent methods as MCP tools and resources.
Any MCP-compatible agent (OpenClaw, Claude Code, Cursor, custom) can connect.

Usage:
    # stdio transport (for OpenClaw/Claude Code integration)
    openclaw-brain serve

    # Read-only profile (ADR-044 D1 — shared-service deployment: no write/ingest/
    # curation tools, no EvidenceVault verbatim text)
    openclaw-brain serve --readonly
    OPENCLAW_BRAIN_READONLY=1 openclaw-brain serve

    # Or programmatically
    from openclaw_brain.server.mcp_server import create_server
    server = create_server()
    server.run()
    server = create_server(readonly=True)
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Literal, TypeVar

from mcp.server.mcpserver import MCPServer  # mcp 2.x; FastMCP was renamed MCPServer

from openclaw_brain.agent import BrainAgent
from openclaw_brain.auth import inject_api_keys
from openclaw_brain.config import load_config, save_config
from openclaw_brain.egress import LOCAL_ONLY, effective_egress, enforce_startup_egress

# Global agent instance — initialized on server startup
_agent: BrainAgent | None = None

_F = TypeVar("_F", bound=Callable[..., Any])

# ADR-044 D1: the shared-service read-only MCP profile. Absence from this set is
# the enforcement — excluded tools are never registered with MCPServer, there is
# no per-tool auth check. Keep in sync with docs/DECISIONS.md ADR-044 D1.
READONLY_TOOLS: frozenset[str] = frozenset({
    # "startup" stays: a readonly consumer must be able to initialize the agent.
    # "shutdown" is EXCLUDED (2026-08-16): the shared readonly SSE daemon exposed it to
    # every remote reader, letting any remote consumer stop the shared agent for all
    # others (verified end-to-end by the cluster orchestrator). Lifecycle teardown is the
    # daemon operator's job, not a read surface.
    "startup",
    "query_knowledge", "answer_question", "recall_memory", "find_bridges",
    "list_open_hypotheses",
    "query_executable", "why", "why_law", "audit_citations", "audit_answer",
    "get_stats", "get_pipeline_config",
})

# Home-lab research-session profile (2026-08-16, home-lab pilot): everything a circuit
# research session needs — all reads, verbatim evidence (home-trusted, unlike the
# shared readonly profile), the design-reasoning record loop, the learner model, and
# oracle-gated claim-card projection. Deliberately EXCLUDED: graph mutation
# (merge/retract), ingestion (curation stays with the owner session), config mutation,
# maintenance/export/lifecycle teardown, grow_executable (D3 pending), and
# analyze_circuit_image (D-7 defect open). Rationale: the 2026-08-15 label-contamination
# census (350 mislabeled design-reasoning nodes) is what ungated writes cost — research
# sessions write through exactly the gated, journaled record_* path and nothing else.
RESEARCHER_TOOLS: frozenset[str] = READONLY_TOOLS | frozenset({
    "get_evidence",
    "record_hypothesis", "record_bench_result", "record_decision",
    "record_assessment", "get_learner_state",
    "reinforce_concept",
    "project_executable",
})

#: profile name -> allowed tool set (None = register everything).
_PROFILES: dict[str, frozenset[str] | None] = {
    "full": None,
    "researcher": RESEARCHER_TOOLS,
    "readonly": READONLY_TOOLS,
}


def _reject_if_not_routable(cfg, model_name: str, role: str) -> str | None:
    """Error string when `model_name` may not be ROUTED to, else None.

    The config-mutation tools used to gate on catalog EXISTENCE alone, so an agent could
    persist a fallback chain (or a stage default) naming a model that
    `LLMProvider.get_chain()` refuses to build — and be told it succeeded. The chain then
    silently runs shorter than configured, or the stage default never resolves. Anything a
    tool writes into a routing slot must therefore pass the same bar the router applies:
    the entry exists AND its status is active.

    `banned` is doubly blocked (the provider-keyed raise in `LLMProvider.get()` is the real
    mechanism); `deprecated`/`incompatible` are blocked here because get_chain drops them.
    Direct `provider.get(name)` is deliberately NOT gated — naming a retired model in code
    still works, with a warning. This gate is only about what gets WRITTEN TO CONFIG.
    """
    entry = cfg.models.get_model(model_name)
    if entry is None:
        available = [m.name for m in cfg.models.catalog]
        return f"Model '{model_name}' not in catalog. Available: {available}"
    if entry.status != "active":
        reason = f" — {entry.status_reason}" if entry.status_reason else ""
        return (
            f"Model '{model_name}' is status={entry.status}{reason} and cannot be routed to: "
            f"LLMProvider.get_chain() drops every non-active model, so writing it into the "
            f"{role} would produce a config that never runs. See the entry's comment in "
            f"config/default.toml; re-activating a model is an A/B-with-cost decision, not a "
            f"config edit."
        )
    return None


def _tool(mcp: MCPServer, allowed: frozenset[str] | None) -> Callable[[_F], _F]:
    """Decorator gate: register with MCPServer unless the active profile excludes this tool.

    Matches on the wrapped function's ``__name__`` against the profile's allowed set
    (None = full profile, everything registers), so the 41 tool bodies below are
    defined exactly once regardless of profile — the only difference between profiles
    is which functions get wrapped by ``mcp.tool()`` at all.
    """
    def decorator(fn: _F) -> _F:
        if allowed is not None and fn.__name__ not in allowed:
            return fn
        return mcp.tool()(fn)
    return decorator


def create_server(
    config_path: str | None = None,
    readonly: bool | None = None,
    profile: str | None = None,
) -> MCPServer:
    """Create and configure the MCP server with all tools and resources.

    Args:
        config_path: Optional path to a config TOML file.
        readonly: Back-compat flag. If True, equivalent to ``profile="readonly"``
            (ADR-044 D1 shared-service profile).
        profile: One of ``_PROFILES`` — "full" (default), "researcher" (home-lab
            research sessions: reads + evidence + record loop + learner +
            oracle-gated projection), "readonly". Resolution precedence:
            explicit ``profile`` > explicit ``readonly=True`` >
            ``$OPENCLAW_BRAIN_PROFILE`` > ``$OPENCLAW_BRAIN_READONLY=1`` > "full".
            An unknown profile raises rather than silently registering everything.
    """
    if profile is None:
        env_profile = os.environ.get("OPENCLAW_BRAIN_PROFILE", "").strip()
        if readonly is True:
            profile = "readonly"
        elif readonly is False:
            # Explicit False overrides OPENCLAW_BRAIN_READONLY (pre-profile contract,
            # pinned by test_explicit_readonly_false_overrides_env_var); a PROFILE env
            # naming a non-readonly tier is still honored.
            profile = env_profile or "full"
        else:
            profile = env_profile or (
                "readonly" if os.environ.get("OPENCLAW_BRAIN_READONLY", "") == "1"
                else "full"
            )
    if profile not in _PROFILES:
        raise ValueError(
            f"unknown MCP profile {profile!r} — expected one of {sorted(_PROFILES)}"
        )
    allowed = _PROFILES[profile]
    readonly = profile == "readonly"

    mcp = MCPServer(
        "openclaw-brain",
        instructions=(
            _READONLY_INSTRUCTIONS if readonly else
            "Semiconductor-engineering knowledge graph + memory (Neo4j-backed). "
            "Core loop: (1) query_knowledge(query) returns a JSON envelope whose "
            "concepts[].id / open_hypotheses[].id / active_decisions[].id are the "
            "handles every write tool accepts; (2) record what you learn — "
            "record_hypothesis(related_concepts=[concept ids]) for testable claims, "
            "record_bench_result(tests_hypothesis=<hypothesis id>) for evidence, "
            "record_decision(motivated_by=[hypothesis ids]) for design choices, "
            "reinforce_concept(id) when knowledge is confirmed in use. "
            "Read brain://guide for the ontology (node labels, design-reasoning "
            "edges, knowledge layers L0-L3) and when to use which tool. "
            "Never invent ids — only use ids returned by query_knowledge or "
            "returned from a record_* call. "
            "Treat concepts[].cite.level as citation discipline: verify chunk "
            "claims with get_evidence, and label derived claims as unverified."
        ),
    )

    # ── Lifecycle ──

    @_tool(mcp, allowed)
    async def startup() -> str:
        """Start the openclaw-brain agent. Must be called before other tools."""
        global _agent
        if _agent and _agent.is_started:
            return "Already running."
        config = load_config(config_path) if config_path else load_config()
        enforce_startup_egress(config)
        # SecretRef/keychain helpers may execute code. Never consult them for a
        # local-only deployment, and never before the active config passes preflight.
        injected = inject_api_keys() if effective_egress(config) != LOCAL_ONLY else {}
        _agent = BrainAgent(config)
        await _agent.start()
        parts = ["openclaw-brain started successfully."]
        if injected:
            parts.append(f"API keys loaded from OpenClaw: {', '.join(injected.keys())}")
        return " ".join(parts)

    @_tool(mcp, allowed)
    async def shutdown() -> str:
        """Shut down the openclaw-brain agent."""
        global _agent
        if _agent:
            await _agent.stop()
            _agent = None
        return "openclaw-brain shut down."

    # ── Knowledge Ingestion ──

    @_tool(mcp, allowed)
    async def ingest_pdf(
        file_path: str,
        extraction_model: str = "",
        reasoning_model: str = "",
        session_id: str = "",
    ) -> str:
        """Ingest a PDF document into the knowledge graph.

        Extracts concepts, equations, and parameters, matches them against
        existing knowledge, reasons about relationships, and commits changes.

        Args:
            file_path: Absolute path to the PDF file.
            extraction_model: Model name for extraction (optional, uses default).
            reasoning_model: Model name for reasoning (optional, uses default).
            session_id: Optional session id (from start_session) to attribute this run to for
                procedural memory / skill-routing history. Omit to keep today's behavior (no
                procedural-memory recording for this run).

        Returns:
            Summary of ingestion results.
        """
        _assert_running()
        result = await _agent.ingest_pdf(
            file_path=file_path,
            extraction_model=extraction_model or None,
            reasoning_model=reasoning_model or None,
            session_id=session_id,
        )
        if hasattr(result, 'output'):
            return json.dumps(result.output, ensure_ascii=False, indent=2)
        return json.dumps(result, default=str, ensure_ascii=False, indent=2)

    @_tool(mcp, allowed)
    async def ingest_html(
        file_path: str,
        extraction_model: str = "",
        reasoning_model: str = "",
        session_id: str = "",
        figures_only: bool = False,
    ) -> str:
        """Ingest one of Rick's lecture-capture HTML decks into the knowledge graph.

        Same pipeline as ingest_pdf (chunk/extract/ground/match/reason/reconcile/commit/embed/
        summarize) — only the parse stage differs: a stdlib HTML parser instead of MinerU, and
        no figure-VLM stage by default (equations/diagrams live inside the lecture's slide
        images; see knowledge/extraction/html_parser.py). Set figures_only=True to instead
        VLM-analyze the deck's slide IMAGES (equations/schematics/plots) and ingest THOSE — can
        run against a source already text-ingested via a normal call (same source_id, a separate
        checkpoint namespace so neither run disturbs the other's resume state); see
        knowledge/pipeline.py::KnowledgePipeline._ingest_html_figures_only.

        Args:
            file_path: Absolute path to the lecture HTML file.
            extraction_model: Model name for extraction (optional, uses default).
            reasoning_model: Model name for reasoning (optional, uses default).
            session_id: Optional session id (from start_session) to attribute this run to for
                procedural memory / skill-routing history. Omit to keep today's behavior (no
                procedural-memory recording for this run).
            figures_only: VLM-analyze slide images instead of the speech transcript (default
                False — the original text-only behavior, unchanged).

        Returns:
            Summary of ingestion results.
        """
        _assert_running()
        result = await _agent.ingest_html(
            file_path=file_path,
            extraction_model=extraction_model or None,
            reasoning_model=reasoning_model or None,
            session_id=session_id,
            figures_only=figures_only,
        )
        if hasattr(result, 'output'):
            return json.dumps(result.output, ensure_ascii=False, indent=2)
        return json.dumps(result, default=str, ensure_ascii=False, indent=2)

    # ── Circuit / Image Analysis ──

    @_tool(mcp, allowed)
    async def analyze_circuit_image(image_path: str) -> str:
        """Analyze a circuit schematic (or any technical image) using VLM + knowledge graph.

        Workflow:
          1. Classifies the image (circuit, plot, block_diagram, layout, etc.)
          2. Extracts circuit topology, components, signal flow, and design features
             using the frontier vision model (e.g. gemini-3.1-pro).
          3. Cross-references the extracted description against the knowledge graph
             (Razavi, ingested papers) to surface related concepts, equations,
             principles, and open hypotheses.

        Use this when a user sends a circuit schematic and wants engineering insights
        grounded in the knowledge base — NOT for ingesting the circuit as new knowledge.
        For PDFs/papers to be stored permanently, use ingest_pdf() instead.

        Args:
            image_path: Absolute path to the image file (PNG, JPG, etc.).
                        Typically the Telegram download path on disk.

        Returns:
            Formatted text with: figure type, visual analysis, and knowledge context.
        """
        _assert_running()
        result = await _agent.analyze_circuit(image_path)
        lines = [
            f"## Figure Type: {result['figure_type']}",
            "",
            "## Visual Analysis",
            result["visual_analysis"],
            "",
            "## Knowledge Context (from graph)",
            result["knowledge_context"] or "No matching knowledge found in graph.",
        ]
        return "\n".join(lines)

    # ── Knowledge Query ──

    @_tool(mcp, allowed)
    async def query_knowledge(query: str) -> str:
        """Query the knowledge graph and memory for relevant information.

        Searches both the Neo4j knowledge graph and the memory system.

        Args:
            query: Natural language query (e.g., "MOSFET amplifier gain").

        Returns:
            JSON envelope:
              context          — human-readable markdown of concepts/edges/memories
              concepts         — [{id, name, confidence, layer, domain, cite}] —
                                 pass these ids to
                                 record_hypothesis(related_concepts=[...]),
                                 record_decision, reinforce_concept, etc.
                                 Each concept carries cite.level — 'chunk' claims can
                                 be verified verbatim via get_evidence(chunk_id);
                                 when you use a 'derived' item in an answer, label
                                 that part of your answer as derived/unverified.
              open_hypotheses  — [{id, statement, status, confidence}] — ids usable
                                 in record_bench_result(tests_hypothesis=...) and
                                 record_decision(motivated_by=[...])
              active_decisions — [{id, choice, status}]
        """
        _assert_running()
        result = await _agent.query_knowledge(query)
        envelope = {
            "context": result.get("formatted") or "No relevant knowledge found.",
            "concepts": result.get("concept_refs", []),
            "open_hypotheses": result.get("open_hypotheses", []),
            "active_decisions": result.get("active_decisions", []),
        }
        return json.dumps(envelope, ensure_ascii=False, default=str)

    @_tool(mcp, allowed)
    async def answer_question(
        question: str, on_insufficient: Literal["flag", "abstain", "model_knowledge"] = "model_knowledge"
    ) -> str:
        """Answer with citations and a deterministic answer envelope.

        Cited chunk ids are verifiable via get_evidence(chunk_id). Returned concepts
        contain ids that feed the design-reasoning write tools for hypotheses and
        decisions. With model_knowledge, uncovered parts use marked model knowledge
        and a local knowledge-gap record; flag/abstain retain their earlier behavior.
        """
        _assert_running()
        result = await _agent.answer_question(question, on_insufficient=on_insufficient)
        envelope = {
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "abstained": result.get("abstained", False),
            "abstain_reason": result.get("abstain_reason", ""),
            "used_concepts": result.get("used_concepts", []),
            "concepts_found": result.get("concepts_found", 0),
            "concepts": result.get("concepts", []),
            "envelope": result.get("envelope"),
        }
        return json.dumps(envelope, ensure_ascii=False, default=str)

    @_tool(mcp, allowed)
    async def get_evidence(chunk_id: str) -> str:
        """Read SourceChunk evidence for citation verification.

        citation 검증용 — query_knowledge가 준 chunk id의 원문 열람.

        Args:
            chunk_id: SourceChunk ID returned by query_knowledge context/citations.

        Returns:
            JSON evidence payload with chunk_id, source_id, text, verbatim,
            section_title, and pages.
        """
        _assert_running()
        result = await _agent.get_evidence(chunk_id)
        return json.dumps(result, ensure_ascii=False, default=str)

    @_tool(mcp, allowed)
    async def recall_memory(query: str) -> str:
        """Search memories only (without knowledge graph).

        Useful for recalling past conversations, lessons, or facts.

        Args:
            query: What to search for in memories.

        Returns:
            Matching memories formatted as text.
        """
        _assert_running()
        result = await _agent.recall(query)
        return result["formatted"] or "No matching memories found."

    # ── Reinforcement ──

    @_tool(mcp, allowed)
    async def reinforce_concept(concept_id: str, evidence: str = "") -> str:
        """Strengthen a concept's confidence in the knowledge graph.

        Call this when a concept is confirmed during conversation,
        or when new evidence supports an existing concept.

        Args:
            concept_id: The concept ID to reinforce.
            evidence: Description of the confirming evidence.

        Returns:
            Confirmation message.
        """
        _assert_running()
        result = await _agent.reinforce(concept_id, evidence)
        return f"Concept '{concept_id}' reinforced."

    @_tool(mcp, allowed)
    async def find_bridges(limit: int = 10) -> str:
        """Find potential cross-domain connections in the knowledge graph.

        Discovers concept pairs in different domains that share neighbors
        but aren't directly connected — candidates for new insights.

        Args:
            limit: Maximum number of bridges to return.

        Returns:
            List of potential cross-domain bridges.
        """
        _assert_running()
        bridges = await _agent.find_bridges(limit)
        if not bridges:
            return "No cross-domain bridges found yet. Add more knowledge from different domains."
        return json.dumps(bridges, default=str, ensure_ascii=False, indent=2)

    # ── Session Management ──

    @_tool(mcp, allowed)
    async def start_session(session_id: str = "") -> str:
        """Start a new episodic memory session.

        Events recorded during the session are linked together and can
        be summarized at session end.

        Args:
            session_id: Optional custom session ID.

        Returns:
            The session ID.
        """
        _assert_running()
        sid = await _agent.start_session(session_id or None)
        return f"Session started: {sid}"

    @_tool(mcp, allowed)
    async def end_session(summary: str = "") -> str:
        """End the current session with an optional summary.

        Args:
            summary: Brief summary of what happened in the session.

        Returns:
            Confirmation.
        """
        _assert_running()
        result = await _agent.end_session(summary)
        return f"Session ended. Summary saved: {result}" if result else "Session ended."

    @_tool(mcp, allowed)
    async def record_event(event_type: str, content: str) -> str:
        """Record an event in the current session.

        Args:
            event_type: Type of event (user_message, agent_response, observation, etc.).
            content: The event content.

        Returns:
            The memory ID of the recorded event.
        """
        _assert_running()
        mid = await _agent.record_event(event_type, content)
        return f"Event recorded: {mid}"

    # ── Entity & Lesson Management ──

    @_tool(mcp, allowed)
    async def upsert_entity(
        entity_id: str,
        entity_type: str,
        name: str,
        summary: str = "",
    ) -> str:
        """Create or update a semantic entity (person, project, tool, etc.).

        Args:
            entity_id: Unique ID for the entity.
            entity_type: Type (person, project, tool, system, concept).
            name: Display name.
            summary: Brief description.

        Returns:
            The entity ID.
        """
        _assert_running()
        eid = await _agent.upsert_entity(entity_id, entity_type, name, summary)
        return f"Entity '{name}' saved: {eid}"

    @_tool(mcp, allowed)
    async def record_lesson(lesson: str, tags: str = "") -> str:
        """Record a durable lesson learned.

        Lessons are stored as curated semantic memories that persist
        across sessions.

        Args:
            lesson: The lesson text.
            tags: Comma-separated tags (optional).

        Returns:
            The memory ID.
        """
        _assert_running()
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
        mid = await _agent.record_lesson(lesson, tag_list)
        return f"Lesson recorded: {mid}"

    # ── Maintenance ──

    @_tool(mcp, allowed)
    async def run_maintenance() -> str:
        """Run maintenance tasks: memory promotion and confidence decay.

        Should be called periodically (e.g., daily) to keep the
        knowledge graph and memory system healthy.

        Returns:
            Summary of maintenance actions.
        """
        _assert_running()
        promo = await _agent.run_promotion()
        decayed = await _agent.run_decay()
        parts = []
        if promo["raw_to_retain"]:
            parts.append(f"{promo['raw_to_retain']} memories promoted raw→retain")
        if promo["retain_to_curated"]:
            parts.append(f"{promo['retain_to_curated']} memories promoted retain→curated")
        if promo["expired"]:
            parts.append(f"{promo['expired']} old memories expired")
        if decayed:
            parts.append(f"{decayed} concepts confidence decayed")
        return "; ".join(parts) if parts else "No maintenance actions needed."

    @_tool(mcp, allowed)
    async def get_stats() -> str:
        """Get system-wide statistics.

        Returns node counts, memory stats, reinforcement metrics,
        available skills, and configured models.
        """
        _assert_running()
        stats = await _agent.get_stats()
        return json.dumps(stats, default=str, ensure_ascii=False, indent=2)

    # ── Skill Routing ──

    @_tool(mcp, allowed)
    async def route_and_execute(user_input: str, session_id: str = "") -> str:
        """Route a natural language input to the best matching skill.

        The agent analyzes the input, selects the most appropriate skill,
        and executes it automatically.

        Args:
            user_input: Natural language instruction.
            session_id: Optional session id (from start_session) to attribute this run to for
                procedural memory / skill-routing history. Omit to keep today's behavior (no
                procedural-memory recording for this run).

        Returns:
            Execution result or fallback message.
        """
        _assert_running()
        result = await _agent.route_and_execute(user_input, session_id=session_id)
        return json.dumps(result, default=str, ensure_ascii=False, indent=2)

    # ── Pipeline Configuration ──

    @_tool(mcp, allowed)
    async def get_pipeline_config() -> str:
        """Get the current pipeline configuration: stage models, fallback chains, and model catalog.

        Returns a human-readable summary of:
        - Default model for each pipeline stage (extraction, reasoning, matching)
        - Fallback chain for each stage
        - Available models in the catalog
        - Resilience settings (retry, backoff)

        No arguments needed. Call startup() first.
        """
        _assert_running()
        cfg = _agent._config
        result = {
            "stages": {
                "extraction": {
                    "default_model": cfg.models.default_extraction,
                    "fallback_chain": cfg.resilience.fallback_extraction,
                },
                "reasoning": {
                    "default_model": cfg.models.default_reasoning,
                    "fallback_chain": cfg.resilience.fallback_reasoning,
                },
                "matching": {
                    "default_model": cfg.models.default_matching,
                    "fallback_chain": cfg.resilience.fallback_matching,
                },
            },
            "catalog": [
                {
                    "name": m.name,
                    "provider": m.provider,
                    "model_id": m.model_id,
                    "tier": m.tier,
                    **({"endpoint": m.endpoint} if m.endpoint else {}),
                }
                for m in cfg.models.catalog
            ],
            "resilience": {
                "max_retries": cfg.resilience.max_retries,
                "initial_backoff_s": cfg.resilience.initial_backoff_s,
                "max_backoff_s": cfg.resilience.max_backoff_s,
                "backoff_multiplier": cfg.resilience.backoff_multiplier,
                "jitter": cfg.resilience.jitter,
                "oauth_refresh_enabled": cfg.resilience.oauth_refresh_enabled,
                "checkpoint_enabled": cfg.resilience.checkpoint_enabled,
            },
            "pipeline_architecture": (
                "chunk → extract → match → reason → commit. "
                "Each stage uses its default model with fallback chain on failure. "
                "Retry with exponential backoff on 429/5xx, OAuth refresh on 401, "
                "checkpoint/resume per chunk for crash recovery."
            ),
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    @_tool(mcp, allowed)
    async def update_stage_model(
        stage: str,
        model_name: str,
    ) -> str:
        """Change the default LLM model for a pipeline stage.

        The model must exist in the catalog AND be status=active — a stage default is the
        model the stage actually runs, so pointing it at a retired/banned/incompatible entry
        is at least as broken as putting one in a fallback chain. Changes are applied
        immediately (in-memory) and persisted to config/default.toml.

        Args:
            stage: Pipeline stage — "extraction", "reasoning", or "matching".
            model_name: Name of an ACTIVE model from the catalog (e.g. "deepseek-v4-flash").

        Returns:
            Confirmation of the change, or the reason the model was rejected.
        """
        _assert_running()
        cfg = _agent._config

        valid_stages = ("extraction", "reasoning", "matching")
        if stage not in valid_stages:
            return f"Invalid stage '{stage}'. Must be one of: {', '.join(valid_stages)}"

        rejection = _reject_if_not_routable(cfg, model_name, f"'{stage}' stage default")
        if rejection:
            return rejection

        attr = f"default_{stage}"
        old_model = getattr(cfg.models, attr)
        setattr(cfg.models, attr, model_name)

        # Clear cached model instances so new default takes effect
        if _agent._llm_provider:
            _agent._llm_provider.clear_cache()

        save_config(cfg)
        return f"Stage '{stage}' default changed: {old_model} → {model_name} (saved to disk)"

    @_tool(mcp, allowed)
    async def update_fallback_chain(
        stage: str,
        models: str,
    ) -> str:
        """Set the fallback chain for a pipeline stage.

        When the primary model fails (after retries), the system tries each fallback in
        order. Every model must exist in the catalog AND be status=active: `get_chain()`
        drops non-active fallbacks, so persisting one would report success while producing
        a chain that is silently shorter than what was asked for.

        Args:
            stage: Pipeline stage — "extraction", "reasoning", or "matching".
            models: Comma-separated ACTIVE model names in fallback order
                    (e.g. "gemini-3.1-flash-lite, deepseek-v4-flash-mlx"). Empty string clears
                    the chain.

        Returns:
            Confirmation with the new chain, or the reason a model was rejected.
        """
        _assert_running()
        cfg = _agent._config

        valid_stages = ("extraction", "reasoning", "matching")
        if stage not in valid_stages:
            return f"Invalid stage '{stage}'. Must be one of: {', '.join(valid_stages)}"

        model_list = [m.strip() for m in models.split(",") if m.strip()] if models else []

        # Every entry must clear the same bar get_chain() applies (in catalog + active),
        # and the whole write is rejected if any one does not — a partially-applied chain
        # would be a config nobody asked for.
        for name in model_list:
            rejection = _reject_if_not_routable(cfg, name, f"'{stage}' fallback chain")
            if rejection:
                return rejection

        attr = f"fallback_{stage}"
        old_chain = getattr(cfg.resilience, attr)
        setattr(cfg.resilience, attr, model_list)

        save_config(cfg)
        return (
            f"Stage '{stage}' fallback chain updated: "
            f"{old_chain} → {model_list} (saved to disk)"
        )

    @_tool(mcp, allowed)
    async def add_catalog_model(
        name: str,
        provider: str,
        model_id: str,
        tier: str = "frontier",
        endpoint: str = "",
    ) -> str:
        """Add a new model to the catalog.

        Once added, the model can be used as a stage default or in fallback chains — so the
        tool refuses to mint an entry that could never be routed to. The new entry is always
        status=active: retiring a model is a decision recorded (with its reason) by editing
        config/default.toml, not something an agent flips through this tool.

        Args:
            name: Short name for the model (e.g. "gpt-5.4", "llama-70b").
            provider: "anthropic", "openai", "google", or "local".
            model_id: The actual model ID for the API (e.g. "gpt-5.4", "claude-opus-4-6-20250514").
            tier: "frontier", "fast", or "local".
            endpoint: API endpoint URL (required for "local" provider, e.g. "http://localhost:8000/v1").

        Returns:
            Confirmation with the updated catalog, or the reason the entry was rejected.
        """
        _assert_running()
        cfg = _agent._config

        if provider not in ("anthropic", "openai", "google", "local"):
            return f"Invalid provider '{provider}'. Must be: anthropic, openai, google, local"

        # Same class of gap as a non-active fallback: `LLMProvider.get()` raises on ANY
        # anthropic-provider model (Rick, 2026-07-22, cost), so adding one would confirm
        # success for an entry that cannot be constructed, let alone routed to.
        if provider == "anthropic" and os.environ.get("OPENCLAW_ALLOW_ANTHROPIC_API") != "1":
            return (
                f"Model '{name}' rejected: anthropic-API models are banned on cost "
                f"(Rick, 2026-07-22) and LLMProvider.get() raises on every one of them, so a "
                f"catalog entry for '{model_id}' could never be constructed or routed to. Set "
                f"OPENCLAW_ALLOW_ANTHROPIC_API=1 to override deliberately."
            )

        if cfg.models.get_model(name):
            return f"Model '{name}' already exists in catalog. Remove it first to re-add."

        from openclaw_brain.config import ModelEntry
        entry = ModelEntry(
            name=name, provider=provider, model_id=model_id,
            tier=tier, endpoint=endpoint,
        )
        cfg.models.catalog.append(entry)

        # Clear cache so new model is available
        if _agent._llm_provider:
            _agent._llm_provider.clear_cache()

        save_config(cfg)
        return f"Model '{name}' added to catalog ({provider}/{model_id}, tier={tier})"

    @_tool(mcp, allowed)
    async def remove_catalog_model(name: str) -> str:
        """Remove a model from the catalog.

        Cannot remove a model that is currently set as a stage default
        or in a fallback chain.

        Args:
            name: Name of the model to remove.

        Returns:
            Confirmation or error if model is in use.
        """
        _assert_running()
        cfg = _agent._config

        if not cfg.models.get_model(name):
            available = [m.name for m in cfg.models.catalog]
            return f"Model '{name}' not in catalog. Available: {available}"

        # Check if model is in use as a default. ALL routing slots count, not just the three
        # update_stage_model can write: removing the vision / figure-analysis / slide-analysis
        # default is the same class of gap as writing a dead name into a chain — the config is
        # persisted happily and the router then fails to resolve a name that is gone.
        routing_slots = {
            "extraction stage default": cfg.models.default_extraction,
            "reasoning stage default": cfg.models.default_reasoning,
            "matching stage default": cfg.models.default_matching,
            "vision (figure classify) default": cfg.models.default_vision,
            "figure_analysis default": cfg.models.default_figure_analysis,
            "[figures] slide_analysis_model": cfg.figures.slide_analysis_model,
            "[figures] slide_analysis_fallback": cfg.figures.slide_analysis_fallback,
        }
        for slot, current in routing_slots.items():
            if current == name:
                fix = ("Change the stage default first with update_stage_model."
                       if slot.endswith("stage default")
                       else "Repoint that setting in config/default.toml first.")
                return f"Cannot remove '{name}' — it's the {slot}. {fix}"

        # Check if model is in a fallback chain
        for stage in ("extraction", "reasoning", "matching"):
            chain = getattr(cfg.resilience, f"fallback_{stage}")
            if name in chain:
                return (
                    f"Cannot remove '{name}' — it's in the '{stage}' fallback chain. "
                    f"Update the fallback chain first with update_fallback_chain."
                )

        cfg.models.catalog = [m for m in cfg.models.catalog if m.name != name]

        if _agent._llm_provider:
            _agent._llm_provider.clear_cache()

        save_config(cfg)
        return f"Model '{name}' removed from catalog."

    # ── Design Reasoning ──

    @_tool(mcp, allowed)
    async def record_hypothesis(
        statement: str,
        assumptions: str = "",
        test_plan: str = "",
        related_concepts: str = "",
    ) -> str:
        """Record a testable hypothesis about a circuit or design issue.

        Hypotheses are tracked in the knowledge graph and can be confirmed
        or falsified by bench results. Use this when exploring root causes,
        predicting behavior, or proposing explanations.

        Args:
            statement: The hypothesis (e.g., "Body effect causes Vth shift > 50mV at SS-cold").
            assumptions: Comma-separated assumptions underlying this hypothesis.
            test_plan: How to verify this hypothesis via simulation or measurement.
            related_concepts: Comma-separated concept IDs to link to.

        Returns:
            The hypothesis ID.
        """
        _assert_running()
        assumption_list = [a.strip() for a in assumptions.split(",") if a.strip()] if assumptions else None
        concept_list = [c.strip() for c in related_concepts.split(",") if c.strip()] if related_concepts else None
        hid = await _agent.record_hypothesis(
            statement=statement,
            assumptions=assumption_list,
            test_plan=test_plan,
            related_concepts=concept_list,
        )
        return f"Hypothesis recorded: {hid}"

    @_tool(mcp, allowed)
    async def record_decision(
        choice: str,
        alternatives: str = "",
        rationale: str = "",
        constraints: str = "",
        related_concepts: str = "",
        motivated_by: str = "",
    ) -> str:
        """Record a design decision with rationale, alternatives, and constraints.

        Captures WHY a particular design choice was made, what was considered,
        and under what constraints. Enables re-evaluation when specs change.

        Args:
            choice: The selected design choice (e.g., "Folded cascode OTA").
            alternatives: Comma-separated alternatives considered.
            rationale: Why this choice was made.
            constraints: Comma-separated constraints (e.g., "VDD=1.2V, area<100um2").
            related_concepts: Comma-separated concept IDs this decision applies to.
            motivated_by: Comma-separated hypothesis IDs that motivated this decision.

        Returns:
            The decision ID.
        """
        _assert_running()
        alt_list = [a.strip() for a in alternatives.split(",") if a.strip()] if alternatives else None
        con_list = [c.strip() for c in constraints.split(",") if c.strip()] if constraints else None
        concept_list = [c.strip() for c in related_concepts.split(",") if c.strip()] if related_concepts else None
        motive_list = [m.strip() for m in motivated_by.split(",") if m.strip()] if motivated_by else None
        did = await _agent.record_decision(
            choice=choice,
            alternatives=alt_list,
            rationale=rationale,
            constraints=con_list,
            related_concepts=concept_list,
            motivated_by=motive_list,
        )
        return f"Decision recorded: {did}"

    @_tool(mcp, allowed)
    async def record_bench_result(
        setup: str,
        metric: str,
        conclusion: str,
        bench_type: str = "simulation",
        corner: str = "",
        tests_hypothesis: str = "",
        confirms: bool = True,
    ) -> str:
        """Record a simulation or measurement result, optionally confirming/falsifying a hypothesis.

        Links bench results to hypotheses to close the hypothesis→bench→feedback loop.
        When a hypothesis is tested, its status is automatically updated.

        Args:
            setup: Testbench setup description (e.g., "CS amp, W/L=10/0.18, VDD=1.8V").
            metric: Measured results (e.g., "gain=22dB, BW=150MHz").
            conclusion: What this result means.
            bench_type: "simulation", "measurement", or "calculation".
            corner: Process/temperature corner (e.g., "SS -40C").
            tests_hypothesis: Hypothesis ID to confirm or falsify (optional).
            confirms: True if the result confirms the hypothesis, False if it falsifies.

        Returns:
            The bench result ID.
        """
        _assert_running()
        bid = await _agent.record_bench_result(
            bench_type=bench_type,
            setup=setup,
            metric=metric,
            corner=corner,
            conclusion=conclusion,
            tests_hypothesis=tests_hypothesis,
            confirms=confirms,
        )
        return f"Bench result recorded: {bid}"

    @_tool(mcp, allowed)
    async def list_open_hypotheses(limit: int = 10) -> str:
        """List all open (unresolved) hypotheses, ordered by confidence.

        Use this to review pending hypotheses that need bench verification,
        or to find hypotheses related to a current issue.

        Args:
            limit: Maximum number of hypotheses to return.

        Returns:
            JSON list of open hypotheses with linked concepts.
        """
        _assert_running()
        results = await _agent.list_open_hypotheses(limit)
        if not results:
            return "No open hypotheses. Record one with record_hypothesis."
        return json.dumps(results, default=str, ensure_ascii=False, indent=2)

    # ── Learner Model (S5, ADR-044 D4) ──
    #
    # Home-only (ADR-044 D1 mechanism-safety grounds, design doc §4.1): deliberately NOT added to
    # READONLY_TOOLS above, for the whole S5 phase, not just S5a — see the frozenset literal.

    @_tool(mcp, allowed)
    async def record_assessment(
        learner_id: str,
        target_id: str,
        verdict: str,
        evidence: str = "",
    ) -> str:
        """Record an explicit assessment verdict for a learner's understanding of a graph node.

        Writes an immutable Assessment event and rolls the outcome up onto the learner's
        UNDERSTANDS state for that target. Use this after directly judging a learner's answer
        (e.g. a teach-back) — it is the only write path for learner-understanding state; there
        is no passive/inferred alternative.

        Args:
            learner_id: Identifies the learner (no registration step — first use creates it).
            target_id: The id of the assessed node (a concepts[].id from query_knowledge, an
                open_hypotheses[].id, a law_id, or any other existing graph node id). A
                non-existent target_id records the Assessment event but silently skips linking
                it to nothing — never fabricates the target.
            verdict: e.g. "understood", "partial", "misconception" (free text, not enumerated).
                "misconception" is treated specially (lower rolled-up confidence); describe the
                specific confusion in `evidence`, never as a separate Concept-to-Concept edge.
            evidence: Free text — optionally a teach-back transcript.

        Returns:
            The assessment ID.
        """
        _assert_running()
        aid = await _agent.record_assessment(
            learner_id=learner_id,
            target_id=target_id,
            verdict=verdict,
            evidence=evidence,
        )
        return f"Assessment recorded: {aid}"

    @_tool(mcp, allowed)
    async def get_learner_state(learner_id: str, target_id: str = "") -> str:
        """Read a learner's rolled-up understanding state (confidence/status/last_assessed/
        assessment_count per assessed target).

        This is a structural ingredient for "what to teach next," not a sequencer — combine
        with query_knowledge's EVOLVES_TO/SUB_BLOCK/knowledge_layer structure to decide that;
        openclaw-brain does not rank or prescribe a next lesson itself.

        Args:
            learner_id: The learner to read.
            target_id: Restrict to one target's state, or "" for every target assessed so far.

        Returns:
            JSON list of {target_id, target_label, target_name, confidence, status,
            last_assessed, assessment_count}, most recently assessed first.
        """
        _assert_running()
        results = await _agent.get_learner_state(learner_id, target_id or None)
        if not results:
            return f"No understanding recorded yet for learner '{learner_id}'."
        return json.dumps(results, default=str, ensure_ascii=False, indent=2)

    # ── Knowledge Graph Curation ──

    @_tool(mcp, allowed)
    async def merge_concepts(primary_id: str, duplicate_id: str) -> str:
        """Merge a duplicate concept node into the primary, re-wiring all edges.

        Use when the LLM extracted the same concept twice with slightly different names
        (e.g. "MOSFET" and "MOSFET Transistor") and you want to consolidate them.

        Steps performed automatically:
          1. Gap-fill: properties present on duplicate but absent on primary are copied.
          2. All outgoing edges from duplicate are re-created on primary (idempotent MERGE).
          3. All incoming edges to duplicate are re-pointed to primary (idempotent MERGE).
          4. Duplicate node is permanently deleted.

        To find candidates: use query_knowledge to search for near-duplicate concepts,
        or check find_bridges for concepts that reference each other indirectly.

        Args:
            primary_id: concept_id of the node to keep (the canonical one).
            duplicate_id: concept_id of the node to absorb and delete.

        Returns:
            JSON summary with rewired edge counts and deletion confirmation.
        """
        _assert_running()
        result = await _agent.merge_concepts(primary_id, duplicate_id)
        return json.dumps(result, default=str, ensure_ascii=False, indent=2)

    @_tool(mcp, allowed)
    async def retract_node(
        node_id: str,
        label: str,
        reason: str = "",
        hard_delete: bool = False,
    ) -> str:
        """Retract (soft-delete or hard-delete) a knowledge node.

        Use when a node was incorrectly extracted, is factually wrong, or is
        a hallucination that slipped through grounding checks.

        Soft delete (default, hard_delete=false):
          Sets retracted=true on the node. It remains in the graph for audit purposes
          but is excluded from all searches, retrieval, and concept matching.
          Recoverable: set retracted=false manually if needed.

        Hard delete (hard_delete=true):
          Permanently removes the node and ALL its edges. Irreversible.
          Use only for nodes with zero useful relationships.

        Supported labels: Concept, Equation, Principle, CircuitTopology, Parameter,
          Assumption, Insight, Hypothesis, DesignDecision, BenchResult.

        Args:
            node_id: The ID value (e.g. the concept_id, equation_id, etc.).
            label: Node type — "Concept", "Equation", "Parameter", etc.
            reason: Human-readable reason (stored on the node for audit trail).
            hard_delete: If True, permanently delete. Default: False (soft-delete).

        Returns:
            JSON with action taken and whether the node was found.
        """
        _assert_running()
        result = await _agent.retract_node(node_id, label, reason, hard_delete)
        return json.dumps(result, default=str, ensure_ascii=False, indent=2)

    # ── Obsidian Export ──

    @_tool(mcp, allowed)
    async def export_obsidian(vault_path: str = "", typed_links: bool = True) -> str:
        """Export the knowledge graph to an Obsidian vault as Markdown files.

        Each concept, equation, principle, etc. becomes a note with YAML
        frontmatter and [[wikilinks]] for relationships. Open the vault in
        Obsidian to browse and visualize the graph.

        Args:
            vault_path: Path to the Obsidian vault directory.
                        Defaults to ~/Semiconductor.
            typed_links: Also emit outgoing edges as rel_type-keyed
                        frontmatter [[wikilink]] properties (e.g.
                        "depends_on: [...]"), resolved to the same target
                        file as the body "## Relationships" section.
                        Default: True.

        Returns:
            Summary of exported nodes.
        """
        _assert_running()
        from pathlib import Path
        from openclaw_brain.export.obsidian import ObsidianExporter

        target = Path(vault_path) if vault_path else Path.home() / "Semiconductor"
        target.mkdir(parents=True, exist_ok=True)

        from openclaw_brain.egress import effective_egress
        exporter = ObsidianExporter(_agent._config.neo4j, target, typed_links=typed_links,
                                    egress=effective_egress(_agent._config))
        try:
            await exporter.connect()
            counts = await exporter.export()
            total = sum(counts.values())
            lines = [f"Exported {total} nodes to {target}:"]
            for label, count in sorted(counts.items()):
                lines.append(f"  {label}: {count}")
            lines.append(f"\nOpen '{target}' in Obsidian to browse.")
            return "\n".join(lines)
        finally:
            await exporter.close()

    # ── Executable-circuit substrate ──

    @_tool(mcp, allowed)
    async def query_executable(topology_class: str = "") -> str:
        """Read the executable-circuit substrate: simulation-VERIFIED circuit specimens and their
        claim-cards. Each claim-card is a falsifiable mechanism claim whose verdict was produced by a
        DETERMINISTIC oracle re-running a real ngspice simulation (sky130 PDK) under stated R1
        conditions (corner/temp/vdd). This is the evidence layer for teaching: the QUANT assertion
        (direction / elasticity / invariance / value) is oracle-certified; the mechanism NARRATIVE is
        interpretive and must NOT be taught as oracle-certified fact. Specimens link into the
        knowledge graph (REALIZES a CircuitTopology, GROUNDS a Parameter).

        Args:
            topology_class: restrict to one class (e.g. "miller_ota_2stage_nmos_in"), or "" for all.

        Returns:
            JSON {topology_class, count, specimens:[{spec_id, topology_class, realizes, pdk, tool,
              claims:[{claim, knob, metric, verdict, narrative, conditions:{corner,temp_c,vdd},
              grounds}]}]}.
        """
        _assert_running()
        result = await _agent.query_executable(topology_class or None)
        return json.dumps(result, ensure_ascii=False, default=str)

    @_tool(mcp, allowed)
    async def project_executable(apply: bool = False) -> str:
        """Build + project ALL validated executable-circuit seed specimens (analog + digital +
        statistical registries, 22 recipes) onto the knowledge graph. Engine-aware: each recipe is
        dispatched to its own engine (ngspice-in-docker for analog/statistical, native iverilog for
        digital); recipes whose engine is unavailable are SKIPPED per-recipe (reported in
        "skipped") — the batch never aborts. Verdicted specimens ADD Specimen/ClaimCard nodes +
        REALIZES/GROUNDS links to existing graph nodes — additive (existing nodes never modified)
        and NO-PHANTOM (a link forms only to a resolved existing node). On apply=True, specimens
        also persist to the git specimen corpus ([executable].corpus_dir). LONG-RUNNING:
        simulations take minutes. Dry-run by default (apply=False simulates + reports verdicts but
        writes nothing); apply=True writes.

        Args:
            apply: True to write the projection; False (default) for a dry-run preview.

        Returns:
            JSON {apply, specimens:[{topology_class, spec_id, verdicts:{claim_id: verdict},
              written?:{nodes, internal_edges, links_resolved, links_total}}]}.
        """
        _assert_running()
        result = await _agent.project_executable(apply)
        return json.dumps(result, ensure_ascii=False, default=str)

    @_tool(mcp, allowed)
    async def grow_executable(
        sources: str = "all", max_recipes: int = 5, max_per_class: int = 4,
        topology_class: str = "", apply: bool = False,
    ) -> str:
        """Corpus growth automation (D3 AUTO lane): author + simulate NEW claim-cards for registered
        topologies — coverage-gap enumeration over the template registry, the ingest growth_queue,
        and the TOPOLOGY_BACKLOG.md curriculum (`sources`, default all three) — then route each
        verdict: VERIFIED/REFUTED-family projects on apply=True (REFUTED is knowledge too, never
        hidden); FLAGGED/REJECTED never project and land in the report's "triage" list for
        post-mortem. Fully automated because it only ever authors against ALREADY-VALIDATED
        templates — a NEW SPICE template is never auto-admitted (that stays a hand-validated code
        change); an unmatched high-signal topology only ever surfaces via "review_lane_pending"
        ({template_queue, backlog_unmapped} counts). Digital (iverilog) classes are excluded from
        AUTO authoring in this wave. LONG-RUNNING: authoring calls an LLM per target and simulation
        takes real wall-clock time. Dry-run by default (apply=False plans+authors+simulates+reports
        but writes nothing); apply=True writes the graph projection and persists to the git corpus.

        Args:
            sources: comma-separated subset of "coverage,queue,backlog", or "all" (default).
            max_recipes: cap on recipes authored+simulated this run (must be > 0).
            max_per_class: cap on new claim-cards per topology_class this run.
            topology_class: restrict to one registered class, or "" for all eligible classes.
            apply: True to write the projection + corpus; False (default) for a dry-run preview.

        Returns:
            JSON GrowthReport: {planned, authored, simulated, projected, skipped_covered,
              triage:[...], review_lane_pending:{template_queue,backlog_unmapped},
              per_class_counts, duration_s, apply}.
        """
        _assert_running()
        parsed_sources = (
            None if sources == "all"
            else [s.strip() for s in sources.split(",") if s.strip()]
        )
        result = await _agent.grow_executable(
            sources=parsed_sources, max_recipes=max_recipes, max_per_class=max_per_class,
            topology_class=topology_class or None, apply=apply,
        )
        return json.dumps(result, ensure_ascii=False, default=str)

    @_tool(mcp, allowed)
    async def retract_executable(spec_id: str) -> str:
        """Reverse a project_executable apply: DETACH DELETE the Specimen + its ClaimCards (and their
        REALIZES / HAS_CLAIM / GROUNDS edges) for one spec_id. The existing graph nodes the links
        pointed at are untouched. Use this to undo a bad projection.

        Args:
            spec_id: the content-hash id of the specimen to retract (from query_executable).

        Returns:
            JSON {spec_id, specimens, claim_cards} — counts of nodes deleted.
        """
        _assert_running()
        result = await _agent.retract_executable(spec_id)
        return json.dumps(result, ensure_ascii=False, default=str)

    @_tool(mcp, allowed)
    async def why(claim_card_id: str) -> str:
        """Grounding for one executable-substrate claim-card — the evidence behind a citation. Use it
        to inspect (or cite) a specific sim-verdicted claim. Remember the boundary: the claim's QUANT
        verdict is oracle-certified, but its `narrative` is interpretive (a mechanism the engineer
        narrates, NEVER an oracle-certified fact).

        Args:
            claim_card_id: the id from query_executable, format {spec_id}:{claim}.

        Returns:
            JSON {claim_card_id, found, claim, knob, metric, verdict, narrative, grounds,
              conditions:{corner,temp_c,vdd}, spec_id, topology_class,
              schematic_path? (SVG circuit diagram, only when the specimen's topology_class has a
              hand-authored template — pilot: 3 of 19 families; omitted, not an error, otherwise)}.
        """
        _assert_running()
        return json.dumps(await _agent.why(claim_card_id), ensure_ascii=False, default=str)

    @_tool(mcp, allowed)
    async def why_law(law_id: str) -> str:
        """Grounding for one law-tier Regularity — the cross-PDK replicated evidence behind a claim
        that has become a LAW (replicated across >= 3 independent foundry model families), citable
        by `law_id` instead of a single claim-card. Use this to inspect (or cite) a cross-PDK
        regularity. Each member PDK's verdict renders scope-honestly: a member whose claim-card has
        actually been projected onto the graph shows its real scope tag; a member recorded only in
        the law's summary (not yet projected onto the graph as its own specimen) renders
        report-only, honestly — never as a bare universal. A Pelgrom-type (power-law) regularity's
        per-PDK fitted exponent is member data, never folded into the one-sentence statement
        (magnitude-free discipline). A law that is not (or no longer) status="law"
        (process_scoped/demoted) surfaces its status_note loudly, and — if it was ever demoted —
        its prior-status history.

        Args:
            law_id: the id from a claim-card's `why(claim_card_id)["laws"]` or
              `query_executable(...)["specimens"][].claims[].law_ids` — a deterministic 40-hex sha1.

        Returns:
            JSON {law_id, found, status, statement, topology_class, metric, knob, quant_kind,
              members:[{pdk, verdict, verdict_scoped, note?, fitted_exponent?, card_id?,
              card_projected}], status_note? (status != "law"), history? (ever demoted),
              supported_by_count, derived_from}, or {law_id, found: false} if unknown.
        """
        _assert_running()
        return json.dumps(await _agent.why_law(law_id), ensure_ascii=False, default=str)

    @_tool(mcp, allowed)
    async def audit_citations(lesson_plan_json: str) -> str:
        """Advisory self-check for a lesson before you teach it. Submit your LessonPlan and this
        verifies the boundaries that matter: a claim you marked `tier="certified"` must cite an
        existing certified claim-card OR a `law`-status Regularity by its `law_id` (so you never
        present a simulation-verified number, or a cross-PDK law, you don't actually have), and
        every claim must carry a tier label (so an interpretive mechanism is never disguised as
        certified fact). Citing a law licenses generalizing EXACTLY across its certified member PDK
        set — one step further (any untested process/foundry/node) still fails; citing a Regularity
        whose status is no longer "law" (process_scoped/demoted) fails loudly, naming the status.
        It is NOT a hard gate and judges nothing else — your lesson's structure, ordering, depth,
        and selection are entirely yours.

        Args:
            lesson_plan_json: JSON of {topology_class, spec_id?, title?, claims:[{text, tier:
              "certified"|"interpretive", cites: claim_card_id or law_id (required when
              certified)}]}.

        Returns:
            JSON AuditReport {passed, certified_total, certified_ok, interpretive_total,
              findings:[{index, ok, reason}]}.
        """
        _assert_running()
        plan = json.loads(lesson_plan_json)
        return json.dumps(await _agent.audit_citations(plan), ensure_ascii=False, default=str)

    @_tool(mcp, allowed)
    async def audit_answer(text: str) -> str:
        """Advisory tier-honesty self-check for a free-form ANSWER before you send it (the
        answer-surface twin of audit_citations, which takes a structured LessonPlan). Submit the
        answer text; every bracket citation — [cite: id], [certified: id], [assoc: id], with
        comma/semicolon-separated ids allowed inside one bracket — is resolved against the graph
        and tiered by what the id ACTUALLY is: a ClaimCard with a VERIFIED-family verdict is
        `certified` (oracle-backed), a ClaimCard with any other verdict is `uncertified_card`
        (a real card, but its verdict licenses no trust language), a law-status Regularity is
        `certified_law` (cross-PDK replication), any other real node
        (Concept/Insight/Equation/...) is `associative` — real and citable, but NEVER certified —
        and an unknown id is `unresolved`. Flags: a cited id that resolves to nothing
        (unresolved_citation); [certified:] syntax on a non-certified-tier id, or a
        certified/verified/oracle-language paragraph backed only by associative/uncertified ids
        (tier_misrepresentation — an id-existence check alone cannot catch this); trust language
        with no citation at all (bare_verified_claim); a paragraph that names PDKs
        (sky130A/sky130, gf180mcuD/gf180mcu/gf180, ihp-sg13g2/sg13g2/ihp — negated mentions
        excluded) beyond the union of what its certified-tier citations actually cover
        (scope_overstatement — e.g. "(verified on sky130A, gf180mcuD, ihp-sg13g2)" attributed
        to a single-PDK claim-card; skipped when no certified citation carries scope info);
        and a NON-FATAL advisory (mixed_certified_paragraph, advisory=true) when a trust
        paragraph mixes certified-tier and non-certified citations. Markdown headings are
        audited WITH the paragraph they label, and negated trust markers ("no certified
        evidence exists") do not count. It is NOT a hard gate and judges nothing else —
        content, structure, and narrative are entirely yours.

        Args:
            text: the full answer text (markdown fine; paragraphs are split on blank lines).

        Returns:
            JSON AnswerAuditReport {passed, citations:[{id, syntax, tier, scope_pdks}],
              findings:[{kind, detail, citation_id?, advisory}]} — passed ignores
              advisory findings.
        """
        _assert_running()
        return json.dumps(await _agent.audit_answer(text), ensure_ascii=False, default=str)

    # ── Resources ──

    @mcp.resource("brain://stats")
    async def resource_stats() -> str:
        """Current system statistics."""
        if not _agent or not _agent.is_started:
            return "openclaw-brain is not running. Call startup() first."
        stats = await _agent.get_stats()
        return json.dumps(stats, default=str, ensure_ascii=False, indent=2)

    @mcp.resource("brain://models")
    async def resource_models() -> str:
        """Available LLM models in the catalog."""
        if not _agent or not _agent.is_started:
            return "openclaw-brain is not running."
        stats = await _agent.get_stats()
        return json.dumps(stats.get("models", []), default=str, indent=2)

    @mcp.resource("brain://guide")
    async def resource_guide() -> str:
        """Orientation guide: ontology + tool flow for agents new to this brain."""
        return _AGENT_GUIDE_READONLY if readonly else _AGENT_GUIDE

    return mcp


def _assert_running() -> None:
    if not _agent or not _agent.is_started:
        raise RuntimeError(
            "openclaw-brain is not running. Call the 'startup' tool first."
        )


_AGENT_GUIDE = """\
# openclaw-brain — Agent Guide

A semiconductor-engineering knowledge graph + memory, built as the second
brain of a CIS (CMOS image sensor) circuit-design engineer. Everything is
stored locally in Neo4j. You are one of possibly several agents using it.

## The core loop (read → write)

1. `query_knowledge(query)` → JSON envelope. `concepts[].id`,
   `open_hypotheses[].id`, `active_decisions[].id` are the ONLY valid
   handles for write tools. Never fabricate an id.
2. Record what the conversation produces:
   - Testable claim → `record_hypothesis(statement, related_concepts=[concept ids])`
   - Simulation/measurement evidence → `record_bench_result(..., tests_hypothesis=<hypothesis id>, confirms=bool)`
   - Design choice → `record_decision(choice, rationale, motivated_by=[hypothesis ids], related_concepts=[concept ids])`
   - Knowledge confirmed useful → `reinforce_concept(id)`
3. `list_open_hypotheses()` shows what is awaiting evidence.

## Citation discipline

`query_knowledge` concepts carry `cite.level`. For `chunk`, verify verbatim
claims with `get_evidence(chunk_id)` using one of `cite.chunks`; for `source`,
treat the claim as source-level provenance; for `derived`, label any answer
text that relies on it as derived/unverified.

## Ontology (what the graph models)

- Knowledge nodes: Concept, Equation, Parameter, CircuitTopology, Principle —
  each carries confidence [0-1], reinforcement_count, knowledge_layer.
- Knowledge layers: L0 math/physics → L1 device physics → L2 analog circuits →
  L3 CIS architecture. Cross-layer edges (BRIDGES_TO) are prized.
- Design-reasoning edges (the point of this graph): SOLVES_PROBLEM,
  INTRODUCES_PROBLEM, TRADES_OFF, EVOLVES_TO — they encode WHY, not just what.
- Design cycle: Hypothesis --TESTS_HYPOTHESIS/FALSIFIED_BY--> BenchResult;
  DesignDecision --MOTIVATED_BY--> Hypothesis; SUPERSEDES between decisions.
- Provenance: every node traces to Source/SourceChunk; confidence decays
  without reinforcement.

## Learner model

- `record_assessment(learner_id, target_id, verdict, evidence="")` — record
  an explicit verdict (e.g. "understood", "partial", "misconception") for a
  learner's understanding of any graph node (a `concepts[].id`, a law id,
  etc.). This is the ONLY write path for learner-understanding state — no
  passive/inferred alternative. `evidence` may carry a teach-back transcript.
  A misconception is described in `evidence`, never as an edge between
  knowledge nodes.
- `get_learner_state(learner_id, target_id="")` — read a learner's rolled-up
  understanding (confidence/status/last_assessed/assessment_count),
  optionally scoped to one target.
- This is a separate subsystem from `recall_memory`/session tools — learner
  state lives on `Learner`/`Assessment` nodes, not `Memory`. Sequencing
  ("what should this learner study next") is NOT this graph's job: combine
  `get_learner_state` with `query_knowledge`'s EVOLVES_TO/SUB_BLOCK/
  knowledge_layer structure yourself, the same way you already sequence
  everything else.

## Etiquette

- Prefer reading (`query_knowledge`, `list_open_hypotheses`) freely; write
  only durable engineering knowledge, not conversational chatter.
- Destructive tools (`merge_concepts`, `retract_node`) require certainty —
  when unsure, record a hypothesis instead.
- `ingest_pdf` is long-running (minutes-hours); do not call it casually.
"""


_READONLY_INSTRUCTIONS = (
    "Semiconductor-engineering knowledge graph + memory (Neo4j-backed), READ-ONLY "
    "profile (ADR-044 D1 — shared-service deployment): no write, ingest, or curation "
    "tools are registered on this server; nothing you do here persists new "
    "knowledge. Core loop: query_knowledge(query) / answer_question(question) "
    "return concepts and citations grounded in the existing graph — "
    "concepts[].id / open_hypotheses[].id / active_decisions[].id identify graph "
    "entities but there is no write tool to feed them into. Read brain://guide "
    "for the ontology and the read-only tool list. Never invent ids. Citation "
    "ids are provenance pointers only — this profile does not serve verbatim "
    "SourceChunk text (EvidenceVault is not distributed shared-service side); label "
    "derived claims as unverified."
)

_AGENT_GUIDE_READONLY = """\
# openclaw-brain — Agent Guide (read-only profile)

A semiconductor-engineering knowledge graph + memory, built as the second
brain of a CIS (CMOS image sensor) circuit-design engineer. This server is
running in READ-ONLY mode (ADR-044 D1, shared-service deployment): no write,
ingest, or curation tools are registered here. You can query and reason over
the existing graph, but nothing you do in this profile persists new knowledge.

## The core loop (read only)

1. `query_knowledge(query)` → JSON envelope. `concepts[].id`,
   `open_hypotheses[].id`, `active_decisions[].id` identify graph entities;
   this profile has no write tool to feed them into.
2. `answer_question(question)` → grounded answer with citations (citation ids
   are provenance metadata — source id / chunk id — not a verbatim-text
   lookup; see Citation discipline below).
3. `recall_memory(query)` — search memories (read-only).
4. `list_open_hypotheses()` — review pending hypotheses awaiting evidence.
5. `find_bridges()` — surface candidate cross-domain connections.

## Citation discipline

`query_knowledge` / `answer_question` concepts and citations carry
`cite.level` / source ids. Treat them as provenance pointers, not verbatim
text — this profile does not serve SourceChunk text (EvidenceVault, the
verbatim copyrighted source text, is not distributed shared-service side; ADR-044
D1). Label derived claims as unverified.

## Executable-circuit substrate (read-only)

- `query_executable(topology_class="")` — read simulation-VERIFIED circuit
  specimens and their claim-cards.
- `why(claim_card_id)` — grounding for one claim-card; its QUANT verdict is
  oracle-certified, its narrative is interpretive (never oracle-certified fact).
- `why_law(law_id)` — grounding for one cross-PDK law-tier Regularity; renders
  each member PDK's verdict scope-honestly (report-only when its card is not
  yet projected), never as a bare universal.
- `audit_citations(lesson_plan_json)` — advisory self-check before teaching a
  lesson: a claim marked certified must cite a real claim-card, or a
  `law`-status Regularity generalized exactly across its member PDK set.

## Diagnostics

- `get_stats()` — system-wide statistics.
- `get_pipeline_config()` — current stage models / fallback chains / catalog.

## Etiquette

- This profile has no write surface: no ingestion, no reinforcement, no
  hypothesis/decision/bench recording, no curation (merge/retract), no config
  mutation. Adding or correcting knowledge requires the full (non-readonly)
  deployment.
"""
