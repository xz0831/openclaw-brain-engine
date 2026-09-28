"""Unified retriever — queries memory + knowledge graph in a single call.

Combines MemoryRetriever (M2) and GraphStore (M1) to provide a single
retrieval interface for the agent. Returns both relevant memories and
related knowledge graph context.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from openclaw_brain.config import BrainConfig
from openclaw_brain.knowledge.graph.schema import NodeLabel
from openclaw_brain.knowledge.graph.store import GraphStore
from openclaw_brain.memory.models import RetrievalResult
from openclaw_brain.memory.retriever import MemoryRetriever
from openclaw_brain.memory.store import MemoryStore


class GraphContext(BaseModel):
    """Knowledge graph context retrieved for a query."""

    concepts: list[dict[str, Any]] = Field(default_factory=list)
    neighbors: list[dict[str, Any]] = Field(default_factory=list)
    total_concepts_found: int = 0


class DecisionContext(BaseModel):
    """Active hypotheses and design decisions linked to retrieved concepts."""

    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    decisions: list[dict[str, Any]] = Field(default_factory=list)


class UnifiedContext(BaseModel):
    """Combined memory + knowledge graph context for agent injection."""

    memories: list[RetrievalResult] = Field(default_factory=list)
    graph: GraphContext = Field(default_factory=GraphContext)
    design: DecisionContext = Field(default_factory=DecisionContext)

    def format(self, token_budget: int = 4000) -> str:
        """Format the full context as a string for agent injection."""
        char_budget = token_budget * 4
        parts: list[str] = []

        # Memory section
        if self.memories:
            mem_lines = []
            for r in self.memories:
                tier = r.memory.tier.value.upper()
                mem_lines.append(f"- [{tier}] {r.memory.content}")
            parts.append("## Relevant Memories\n" + "\n".join(mem_lines))

        # Knowledge graph section — IDs are exposed so an agent can pass them
        # to write tools (record_hypothesis, record_decision, reinforce, ...).
        if self.graph.concepts:
            kg_lines = []
            for c in self.graph.concepts:
                name = c.get("canonical_name", c.get("concept_id", "?"))
                desc = c.get("description", "")
                nid = _node_ref(c)
                line = f"- **{name}** (`id: {nid}`)" if nid else f"- **{name}**"
                if desc:
                    line += f": {desc}"
                kg_lines.append(line)
            parts.append("## Related Knowledge\n" + "\n".join(kg_lines))

        if self.graph.neighbors:
            edge_lines = []
            for n in self.graph.neighbors:
                node = n.get("node", {})
                node_name = (
                    node.get("canonical_name", "")
                    or node.get("name", "")
                    or str(node.get("_labels", []))
                )
                rel = n.get("rel_type", "?")
                rationale = n.get("rationale", "")
                line = f"- {rel} → {node_name}"
                if rationale:
                    line += f" ({rationale})"
                edge_lines.append(line)
            parts.append("## Graph Connections\n" + "\n".join(edge_lines))

        # Decision context section
        if self.design.hypotheses or self.design.decisions:
            dc_lines = []
            for h in self.design.hypotheses:
                status_icon = {"open": "\U0001f534", "confirmed": "\U0001f7e2", "falsified": "\u274c"}.get(
                    h.get("status", ""), "\u2753"
                )
                dc_lines.append(
                    f"- {status_icon} Hypothesis (`id: {h.get('id', '?')}`): "
                    f"{h.get('statement', '?')} (confidence: {h.get('confidence', '?')})"
                )
            for d in self.design.decisions:
                status_icon = "\U0001f7e2" if d.get("status") == "active" else "\u26a0\ufe0f"
                dc_lines.append(
                    f"- {status_icon} Decision (`id: {d.get('id', '?')}`): "
                    f"{d.get('choice', '?')} — {d.get('rationale', '')}"
                )
            parts.append("## Decision Context\n" + "\n".join(dc_lines))

        text = "\n\n".join(parts)
        if len(text) > char_budget:
            text = text[:char_budget] + "\n... (truncated)"
        return text

    def compact(self) -> dict[str, Any]:
        """Machine-readable envelope so an agent can close the read→write loop.

        Every entry carries the node ID that the write tools accept
        (record_hypothesis(related_concepts=[...]), record_decision,
        reinforce_concept, merge_concepts, retract_node).
        """
        return {
            "concepts": [
                {
                    "id": _node_ref(c),
                    "name": c.get("canonical_name", ""),
                    "confidence": c.get("confidence"),
                    "layer": c.get("knowledge_layer"),
                    "domain": c.get("domain"),
                    "cite": _concept_cite(c),
                }
                for c in self.graph.concepts
                if _node_ref(c)
            ],
            "open_hypotheses": [
                {
                    "id": h.get("id"),
                    "statement": h.get("statement"),
                    "status": h.get("status"),
                    "confidence": h.get("confidence"),
                }
                for h in self.design.hypotheses
            ],
            "active_decisions": [
                {
                    "id": d.get("id"),
                    "choice": d.get("choice"),
                    "status": d.get("status"),
                }
                for d in self.design.decisions
            ],
        }


def _node_ref(node: dict[str, Any]) -> str:
    """Primary ID of a knowledge node dict, whichever label it carries."""
    return (
        node.get("concept_id") or node.get("topology_id")
        or node.get("parameter_id") or node.get("equation_id")
        or node.get("principle_id") or ""
    )


def _concept_cite(node: dict[str, Any]) -> dict[str, Any]:
    """Envelope citation label for a retrieved knowledge node."""
    evidence_chunk_ids = node.get("evidence_chunk_ids") or []
    if isinstance(evidence_chunk_ids, str):
        evidence_chunk_ids = [evidence_chunk_ids]
    source_id = node.get("source_id")

    if evidence_chunk_ids:
        cite = {"level": "chunk"}
        if source_id:
            cite["src"] = source_id
        cite["chunks"] = list(evidence_chunk_ids)[:3]
        return cite
    if source_id:
        return {"level": "source", "src": source_id}
    return {"level": "derived"}


class UnifiedRetriever:
    """Retrieves context from both memory and knowledge graph."""

    def __init__(
        self,
        memory_store: MemoryStore,
        graph: GraphStore,
        config: BrainConfig,
    ):
        self._memory_retriever = MemoryRetriever(memory_store, config)
        self._graph = graph
        self._config = config

    async def retrieve(
        self,
        query: str,
        session_id: str | None = None,
        include_memory: bool = True,
        include_graph: bool = True,
    ) -> UnifiedContext:
        """Retrieve relevant context from all sources.

        Args:
            query: The user's message or query text.
            session_id: Current session ID for episodic context.
            include_memory: Whether to include memory results.
            include_graph: Whether to include knowledge graph results.

        Returns:
            UnifiedContext with memories and graph context.
        """
        ctx = UnifiedContext()

        if include_memory:
            ctx.memories = await self._memory_retriever.retrieve(
                query, session_id=session_id,
            )

        if include_graph:
            ctx.graph = await self._retrieve_graph_context(query)
            ctx.design = await self._retrieve_decision_context(ctx.graph)

        return ctx

    # Korean → English technical term mapping.
    # Only include terms that are truly Korean-only (no English equivalent in the query).
    # Generic loanwords like 노이즈/noise, 픽셀/pixel are intentionally excluded.
    _KO_TO_EN: dict[str, str] = {
        # Math / Statistics (L0) — these typically appear only in Korean
        "가우시안": "Gaussian",
        "정규분포": "normal distribution",
        "확률분포": "probability distribution",
        "확률밀도": "probability density",
        "표준편차": "standard deviation",
        "시정수": "time constant",
        "미분방정식": "differential equation",
        "지수함수": "exponential function",
        "커패시터": "capacitor",
        # Device physics (L1) — compound Korean terms
        "문턱전압": "threshold voltage",
        "바디효과": "body effect",
        "채널핀치오프": "channel pinchoff",
        "드레인 전류": "drain current",
        "드레인전류": "drain current",
        "포화 영역": "saturation region",
        "주파수": "frequency",
    }

    @staticmethod
    def _extract_search_terms(query: str) -> list[str]:
        """Extract searchable technical terms from a free-form query.

        Handles mixed Korean/English queries by:
        1. Including the full query (for short queries / pure English)
        2. Extracting English words, bigrams, and technical abbreviations
        3. Translating known Korean technical terms to English
        """
        import re
        from openclaw_brain.knowledge.reasoning.matcher import _ABBREVIATIONS

        terms: list[str] = []

        # Short queries (≤40 chars, mostly English) → use as-is
        korean_chars = sum(1 for c in query if '\uAC00' <= c <= '\uD7A3')
        if len(query) <= 40 and korean_chars < 3:
            terms.append(query)
            return terms

        # Extract English words (technical terms, abbreviations)
        english_words = re.findall(r'[A-Za-z][A-Za-z0-9\-\.]+', query)
        clean_words = [w.strip('.-') for w in english_words if len(w.strip('.-')) >= 2]

        # Add bigrams first (consecutive word pairs) — enables exact-phrase matching
        # e.g. "thermal noise", "flicker noise", "single slope", "kTC noise"
        for i in range(len(clean_words) - 1):
            w1, w2 = clean_words[i], clean_words[i + 1]
            if w1.lower() != w2.lower():  # skip self-pairs like "RC RC"
                terms.append(f"{w1} {w2}")

        # Korean technical term translations (after bigrams — fires when English terms sparse)
        for ko, en in UnifiedRetriever._KO_TO_EN.items():
            if ko in query:
                terms.append(en)

        # Individual words and abbreviation expansions
        for clean in clean_words:
            terms.append(clean)
            low = clean.lower()
            if low in _ABBREVIATIONS:
                terms.append(_ABBREVIATIONS[low])

        # Add first significant English phrase (for ordering)
        if not terms:
            terms.append(query[:40])

        return list(dict.fromkeys(terms))  # deduplicate preserving order

    @staticmethod
    def _node_label_and_id(node: dict) -> tuple[NodeLabel, str, str] | None:
        """Return (NodeLabel, id_field, id_value) for a graph node dict.

        Returns None if the node has no recognised ID field.
        """
        for field, label in (
            ("concept_id", NodeLabel.CONCEPT),
            ("topology_id", NodeLabel.CIRCUIT_TOPOLOGY),
            ("parameter_id", NodeLabel.PARAMETER),
            ("equation_id", NodeLabel.EQUATION),
            ("principle_id", NodeLabel.PRINCIPLE),
        ):
            val = node.get(field)
            if val:
                return label, field, val
        return None

    async def _retrieve_graph_context(self, query: str) -> GraphContext:
        """Search the knowledge graph for concepts related to the query."""
        from openclaw_brain.knowledge.embedding import encode

        limit = self._config.reasoning.max_graph_neighbors
        search_terms = self._extract_search_terms(query)

        # Pre-generate query embedding once for vector supplement.
        # Must use the configured model — a hardcoded default would mismatch
        # the stored vectors' dimension/distribution after a model migration.
        try:
            query_embedding = encode(query[:512], self._config.embedding.model)
        except Exception:
            query_embedding = None

        # Gather unique nodes across all search terms (Concept + other knowledge types).
        # Fetch at most 2 candidates per term so diverse search terms all get a chance
        # to contribute before the total budget is exhausted.
        import re as _re

        per_term_limit = 2
        seen_ids: set[str] = set()
        seen_names: set[str] = set()
        concepts: list[dict] = []
        for term in search_terms:
            # Normalize hyphens/underscores so "single-slope" finds "Single Slope ADC"
            term_norm = term.replace("-", " ").replace("_", " ")
            term_is_short = len(term_norm.replace(" ", "")) <= 2

            # For very short terms (≤2 chars like "RC", "gm"), fetch extra candidates
            # and post-filter to word-boundary matches only — prevents "RC" matching
            # "source" (s-o-u-r-c-e) or "circuit" (ci-r-c-u-i-t) as substring.
            fetch_limit = per_term_limit if not term_is_short else per_term_limit + 4
            # Pass query embedding only on the first (most representative) term
            term_emb = query_embedding if term == search_terms[0] else None
            raw_candidates = await self._graph.find_similar_concepts(
                term_norm, embedding=term_emb, limit=fetch_limit
            )
            if term_is_short:
                term_lower = term_norm.lower()
                candidates = []
                for c in raw_candidates:
                    name_lower = c.get("canonical_name", "").lower()
                    words = set(_re.split(r"[\s\-_/().,;:]+", name_lower))
                    if term_lower in words:
                        candidates.append(c)
                candidates = candidates[:per_term_limit]
            else:
                candidates = raw_candidates
            for c in candidates:
                info = self._node_label_and_id(c)
                name = c.get("canonical_name", "").lower().strip()
                if info and info[2] not in seen_ids and (not name or name not in seen_names):
                    seen_ids.add(info[2])
                    if name:
                        seen_names.add(name)
                    concepts.append(c)
                    if len(concepts) >= limit:
                        break
            if len(concepts) >= limit:
                break

        # Backward compat: also try the raw query if we got nothing
        if not concepts:
            concepts = await self._graph.find_similar_concepts(query, limit=limit)

        if not concepts:
            return GraphContext()

        # Get neighborhoods for top concepts
        all_neighbors: list[dict[str, Any]] = []
        seen_ids_nb: set[str] = set()

        for concept in concepts[:3]:  # Top 3 concepts
            info = self._node_label_and_id(concept)
            if info is None:
                continue
            label, id_field, id_value = info
            if id_value in seen_ids_nb:
                continue
            seen_ids_nb.add(id_value)

            neighbors = await self._graph.get_neighborhood(
                label=label,
                id_field=id_field,
                id_value=id_value,
                hops=self._config.reasoning.graph_hop_depth,
                limit=self._config.reasoning.max_graph_neighbors,
            )
            all_neighbors.extend(neighbors)

        return GraphContext(
            concepts=concepts,
            neighbors=all_neighbors,
            total_concepts_found=len(concepts),
        )

    async def _retrieve_decision_context(self, graph_ctx: GraphContext) -> DecisionContext:
        """Find hypotheses and decisions linked to retrieved concepts."""
        concept_ids = []
        for c in graph_ctx.concepts:
            cid = c.get("concept_id")
            if cid:
                concept_ids.append(cid)
        if not concept_ids:
            return DecisionContext()

        # Query hypotheses linked to any of the retrieved concepts
        hyp_query = """
        MATCH (h:Hypothesis)-[:RELATES_TO]->(c:Concept)
        WHERE c.concept_id IN $concept_ids AND h.status IN ['open', 'confirmed']
        RETURN DISTINCT h.hypothesis_id AS id, h.statement AS statement,
               h.status AS status, h.confidence AS confidence
        ORDER BY h.confidence DESC
        LIMIT 5
        """
        # Query design decisions linked to any of the retrieved concepts
        dec_query = """
        MATCH (c:Concept)-[:DECIDED_BY]->(d:DesignDecision)
        WHERE c.concept_id IN $concept_ids AND d.status = 'active'
        RETURN DISTINCT d.decision_id AS id, d.choice AS choice,
               d.rationale AS rationale, d.status AS status
        LIMIT 5
        """
        params = {"concept_ids": concept_ids}
        hypotheses = await self._graph.run_read_query(hyp_query, params)
        decisions = await self._graph.run_read_query(dec_query, params)
        return DecisionContext(hypotheses=hypotheses, decisions=decisions)
