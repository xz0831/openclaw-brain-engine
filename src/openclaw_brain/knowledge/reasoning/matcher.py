"""Concept matcher — compares extracted concepts against the existing graph.

For each extracted concept, determines whether it matches an existing
node (merge), is ambiguous (needs LLM resolution), or is genuinely new.

Decision hierarchy (embedding-first, when enabled via MatcherConfig):
  Tier 0  normalized-name / alias exact hit          → MATCH (deterministic)
  Tier 1  cosine ≥ t_high and no conflict guard      → MATCH (embedding)
  Tier 2  cosine ≥ t_low or name-sim ≥ legacy low    → MatchVerifier (1-token
          SAME/DIFFERENT/UNSURE); UNSURE or budget exhausted → ambiguous
  Tier 3  otherwise                                   → NEW

The legacy pure name-similarity decision is preserved behind
``use_embedding_decision=False`` and as the fallback when no embedding
is available. ``_name_similarity`` survives as guard and tiebreaker.

F6: ``match()`` also resolves Equation/Parameter mentions (previously they bypassed
matching entirely and every mention was proposed as a new node — see
``knowledge/reasoning/README.md`` "Known defects" #2). These are a SEPARATE, binary
SAME/NEW decision — exact match on normalized ``canonical_latex`` for equations,
normalized ``canonical_name`` + unit agreement for parameters — deliberately with no
fuzzy/embedding tier (see ``_match_equations``/``_match_parameters`` below). Not part
of the tiered hierarchy above; not gated by ``use_embedding_decision``.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from openclaw_brain.knowledge.extraction.models import (
    AmbiguousMatch,
    ConceptMention,
    EquationMention,
    ExtractionResult,
    MatchedConcept,
    MatchedEquation,
    MatchedParameter,
    MatchResult,
    ParameterMention,
)
from openclaw_brain.knowledge.graph.store import GraphStore

if TYPE_CHECKING:
    from openclaw_brain.config import MatcherConfig
    from openclaw_brain.knowledge.reasoning.verifier import MatchVerifier

logger = logging.getLogger(__name__)


class ConceptMatcher:
    """Matches extracted concepts against the existing knowledge graph."""

    def __init__(
        self,
        graph: GraphStore,
        high_threshold: float = 0.85,
        low_threshold: float = 0.6,
        embedding_model: str = "all-MiniLM-L6-v2",
        matcher_config: "MatcherConfig | None" = None,
        verifier: "MatchVerifier | None" = None,
    ):
        self._graph = graph
        self._high = high_threshold
        self._low = low_threshold
        # Must match the model used to write node embeddings, or query vectors
        # and stored vectors would mismatch in dimension/distribution.
        self._embedding_model = embedding_model
        self._mcfg = matcher_config
        self._verifier = verifier

    async def match(self, extraction: ExtractionResult) -> MatchResult:
        """Match all concepts from an extraction against the graph."""
        from openclaw_brain.knowledge.embedding import encode_batch

        matched: list[MatchedConcept] = []
        new_concepts: list[ConceptMention] = []
        ambiguous: list[AmbiguousMatch] = []

        use_embedding = self._mcfg is not None and self._mcfg.use_embedding_decision

        # Batch-encode all mentions up front — mentions repeat across chunks
        # and per-mention encode loops were the real matching latency.
        emb_texts = [
            f"{c.name}. {c.description}" if c.description else c.name
            for c in extraction.concepts
        ]
        try:
            embeddings: list[list[float] | None] = encode_batch(
                emb_texts, self._embedding_model
            )
        except Exception:
            embeddings = [None] * len(extraction.concepts)

        verify_budget = self._mcfg.max_verify_per_chunk if self._mcfg else 0

        for concept, embedding in zip(extraction.concepts, embeddings):
            embedding = embedding or None
            if use_embedding:
                outcome, verify_budget = await self._match_one_tiered(
                    concept, embedding, verify_budget
                )
            else:
                outcome = await self._match_one_legacy(concept, embedding)

            kind, payload = outcome
            if kind == "matched":
                matched.append(payload)
            elif kind == "ambiguous":
                ambiguous.append(payload)
            else:
                new_concepts.append(payload)

        # F6: equations/parameters — exact-normalized matching, no fuzzy band (see module
        # docstring). Each helper no-ops (zero Cypher calls) when its mention list is empty,
        # the common per-chunk case.
        matched_equations, new_equations = await self._match_equations(extraction.equations)
        matched_parameters, new_parameters = await self._match_parameters(extraction.parameters)

        return MatchResult(
            chunk_id=extraction.chunk_id,
            matched=matched,
            new_concepts=new_concepts,
            ambiguous=ambiguous,
            matched_equations=matched_equations,
            new_equations=new_equations,
            matched_parameters=matched_parameters,
            new_parameters=new_parameters,
        )

    # ── tiered decision (embedding-first) ──

    async def _match_one_tiered(
        self,
        concept: ConceptMention,
        embedding: list[float] | None,
        verify_budget: int,
    ) -> tuple[tuple[str, object], int]:
        assert self._mcfg is not None
        entries = await self._graph.find_match_candidates(
            concept.name, embedding=embedding, limit=8,
        )
        if not entries:
            return ("new", concept), verify_budget

        # Tier 0 — deterministic name/alias layer keeps final authority.
        norm_mention = _normalize_name(concept.name)
        for entry in entries:
            node = entry["node"]
            if _normalize_name(node.get("canonical_name", "")) == norm_mention or any(
                _normalize_name(a) == norm_mention for a in node.get("aliases", []) or []
            ):
                return ("matched", self._as_match(concept, node, 1.0, "name")), verify_budget

        # Best embedding candidate and best name-sim candidate.
        scored = [e for e in entries if e["cos"] is not None]
        best_vec = scored[0] if scored else None
        ns_of = lambda e: _name_similarity(  # noqa: E731
            concept.name, e["node"].get("canonical_name", "")
        )
        best_ns_entry = max(entries, key=ns_of)
        best_ns = ns_of(best_ns_entry)

        # Tier 1 — embedding auto-match, guard permitting.
        if best_vec is not None and best_vec["cos"] >= self._mcfg.t_high:
            if not _conflict_guard(concept, best_vec["node"], ns_of(best_vec)):
                return (
                    ("matched", self._as_match(concept, best_vec["node"], best_vec["cos"], "embedding")),
                    verify_budget,
                )

        # Tier 2 — verification band.
        in_band = (best_vec is not None and best_vec["cos"] >= self._mcfg.t_low) or (
            best_ns >= self._low
        )
        if in_band:
            top = best_vec if best_vec is not None else best_ns_entry
            if self._verifier is not None and verify_budget > 0:
                verify_budget -= 1
                result = await self._verifier.verify(
                    concept.name, concept.description or "", top["node"]
                )
                if result.verdict == "SAME":
                    sim = top["cos"] if top["cos"] is not None else ns_of(top)
                    return (
                        ("matched", self._as_match(concept, top["node"], sim, "verified")),
                        verify_budget,
                    )
                if result.verdict == "DIFFERENT":
                    return ("new", concept), verify_budget
                # UNSURE falls through to ambiguous
            return ("ambiguous", self._as_ambiguous(concept, entries)), verify_budget

        # Tier 3 — new.
        return ("new", concept), verify_budget

    # ── legacy decision (pure name similarity) ──

    async def _match_one_legacy(
        self, concept: ConceptMention, embedding: list[float] | None
    ) -> tuple[str, object]:
        candidates = await self._graph.find_similar_concepts(
            concept.name, embedding=embedding, limit=3,
        )
        if not candidates:
            return ("new", concept)

        best = candidates[0]
        similarity = _name_similarity(concept.name, best.get("canonical_name", ""))
        if similarity >= self._high:
            return ("matched", self._as_match(concept, best, similarity, "name"))
        if similarity >= self._low:
            return (
                "ambiguous",
                self._as_ambiguous(
                    concept, [{"node": c, "cos": None, "text_hit": True} for c in candidates]
                ),
            )
        return ("new", concept)

    # ── shared constructors ──

    def _as_match(
        self, concept: ConceptMention, node: dict, similarity: float, method: str
    ) -> MatchedConcept:
        node_id, id_field, label = _node_primary_key(node)
        return MatchedConcept(
            mention=concept,
            existing_node_id=node_id,
            similarity=similarity,
            match_method=method,
            node_label=label,
            node_id_field=id_field,
        )

    def _as_ambiguous(self, concept: ConceptMention, entries: list[dict]) -> AmbiguousMatch:
        cands = []
        best_sim = 0.0
        for e in entries[:5]:
            node = e["node"]
            ns = _name_similarity(concept.name, node.get("canonical_name", ""))
            sim = e["cos"] if e.get("cos") is not None else ns
            best_sim = max(best_sim, sim)
            cands.append({
                "concept_id": node.get("concept_id", ""),
                "canonical_name": node.get("canonical_name", ""),
                "description": node.get("description", ""),
                "similarity": sim,
            })
        return AmbiguousMatch(mention=concept, candidates=cands, best_similarity=best_sim)

    # ── F6: equation matching — exact match on NORMALIZED canonical_latex ──
    #
    # Conservative first cut: no fuzzy/embedding tier (unlike concepts above), no attempt at
    # symbolic equivalence — see ``_normalize_latex``. Binary SAME/NEW, no ambiguous band.

    async def _match_equations(
        self, mentions: list[EquationMention],
    ) -> tuple[list[MatchedEquation], list[EquationMention]]:
        if not mentions:
            return [], []

        candidates = await self._fetch_equation_candidates()
        matched: list[MatchedEquation] = []
        new: list[EquationMention] = []
        for mention in mentions:
            norm = _normalize_latex(mention.latex)
            hit = next((c for c in candidates if c["norm_latex"] == norm), None) if norm else None
            if hit is not None:
                matched.append(MatchedEquation(mention=mention, existing_node_id=hit["equation_id"]))
            else:
                new.append(mention)
        return matched, new

    async def _fetch_equation_candidates(self) -> list[dict]:
        """Bulk-fetch ALL Equation nodes ONCE per match() call (not per mention) — mirrors the
        batch-encode-up-front pattern used for concept embeddings above. A lean 2-column
        projection, not full node dicts.
        """
        query = """
        MATCH (n:Equation)
        WHERE n.canonical_latex IS NOT NULL AND n.canonical_latex <> ''
          AND NOT coalesce(n.retracted, false)
        RETURN n.equation_id AS equation_id, n.canonical_latex AS canonical_latex
        ORDER BY n.equation_id
        """
        try:
            rows = await self._graph.run_read_query(query)
        except Exception as e:
            logger.warning("Equation candidate fetch failed — treating all as new: %s", e)
            return []
        candidates = []
        for r in rows:
            latex = r.get("canonical_latex")
            if not isinstance(latex, str):
                # Defensive: a handful of legacy nodes carry a non-string canonical_latex (a
                # pre-existing, unrelated data-quality defect — confirmed live, 2026-07-10
                # reconnaissance). Skip rather than crash the matcher on malformed legacy data.
                continue
            candidates.append({
                "equation_id": r.get("equation_id", ""),
                "norm_latex": _normalize_latex(latex),
            })
        return candidates

    # ── F6: parameter matching — exact match on normalized canonical_name + unit agreement ──
    #
    # Conservative first cut: normalization is case/trim/whitespace ONLY (deliberately
    # narrower than the concept matcher's abbreviation-expanding _normalize_name — see
    # _normalize_plain). A unit conflict (both sides carry a DIFFERENT unit) disqualifies a
    # candidate outright; a missing unit on either side is gap-fill semantics. No fuzzy/
    # embedding tier, no ambiguous band.

    async def _match_parameters(
        self, mentions: list[ParameterMention],
    ) -> tuple[list[MatchedParameter], list[ParameterMention]]:
        if not mentions:
            return [], []

        candidates = await self._fetch_parameter_candidates()
        matched: list[MatchedParameter] = []
        new: list[ParameterMention] = []
        for mention in mentions:
            norm_name = _normalize_plain(mention.name)
            norm_units = _normalize_plain(mention.units)
            best = _best_parameter_candidate(norm_name, norm_units, candidates)
            if best is not None:
                matched.append(MatchedParameter(mention=mention, existing_node_id=best["parameter_id"]))
            else:
                new.append(mention)
        return matched, new

    async def _fetch_parameter_candidates(self) -> list[dict]:
        """Bulk-fetch ALL Parameter nodes ONCE per match() call — same rationale as
        ``_fetch_equation_candidates``.
        """
        query = """
        MATCH (n:Parameter)
        WHERE n.canonical_name IS NOT NULL AND n.canonical_name <> ''
          AND NOT coalesce(n.retracted, false)
        RETURN n.parameter_id AS parameter_id, n.canonical_name AS canonical_name, n.units AS units
        ORDER BY n.parameter_id
        """
        try:
            rows = await self._graph.run_read_query(query)
        except Exception as e:
            logger.warning("Parameter candidate fetch failed — treating all as new: %s", e)
            return []
        candidates = []
        for r in rows:
            name = r.get("canonical_name")
            if not isinstance(name, str):
                continue  # defensive: same non-string-legacy-data guard as equations above
            units = r.get("units")
            candidates.append({
                "parameter_id": r.get("parameter_id", ""),
                "norm_name": _normalize_plain(name),
                "norm_units": _normalize_plain(units) if isinstance(units, str) else "",
            })
        return candidates


def _digit_tokens(normalized: str) -> set[str]:
    """Tokens containing digits — circuit designators like 3t, 4t, 8t, 2x."""
    return {t for t in normalized.split() if any(ch.isdigit() for ch in t)}


def _names_conflict(
    name_a: str, desc_a: str, name_b: str, desc_b: str, name_sim: float
) -> bool:
    """Primitive guard usable on any name/description pair.

    Guards: (1) differing numeric designator tokens ('3T pixel' vs '4T pixel'
    embed close but are different circuits); (2) near-zero surface similarity
    with no cross-mention in the descriptions — high cosine alone is not
    enough to silently merge two names that share nothing lexically.
    """
    na = _normalize_name(name_a)
    nb = _normalize_name(name_b)
    da, db = _digit_tokens(na), _digit_tokens(nb)
    if da and db and da != db:
        return True
    if name_sim < 0.15:
        cross = (nb and nb in (desc_a or "").lower()) or (na and na in (desc_b or "").lower())
        if not cross:
            return True
    return False


def _conflict_guard(concept: ConceptMention, node: dict, name_sim: float) -> bool:
    """Veto an embedding auto-match that smells like a near-miss (mention vs node)."""
    return _names_conflict(
        concept.name,
        concept.description or "",
        node.get("canonical_name", ""),
        node.get("description", "") or "",
        name_sim,
    )


def _node_primary_key(node: dict) -> tuple[str, str, str]:
    """Return (id_value, id_field, label_str) for a graph node dict."""
    for field, label in (
        ("concept_id", "Concept"),
        ("topology_id", "CircuitTopology"),
        ("parameter_id", "Parameter"),
        ("equation_id", "Equation"),
        ("principle_id", "Principle"),
    ):
        val = node.get(field)
        if val:
            return val, field, label
    return node.get("canonical_name", "unknown"), "concept_id", "Concept"


def _name_similarity(a: str, b: str) -> float:
    """Multi-signal name similarity: normalization + token Jaccard + character subsequence.

    Handles abbreviations (FD-SOI = Fully Depleted SOI), hyphenation variants,
    and snake_case vs Title Case differences.
    """
    if not a or not b:
        return 0.0

    na = _normalize_name(a)
    nb = _normalize_name(b)

    # Exact match after normalization
    if na == nb:
        return 1.0

    # Token-level Jaccard
    tokens_a = set(na.split())
    tokens_b = set(nb.split())

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    jaccard = len(intersection) / len(union)

    # Character-level longest common subsequence ratio (catches partial matches)
    lcs_ratio = _lcs_ratio(na, nb)

    # Combined score (weighted toward token match but boosted by character similarity)
    return max(jaccard, lcs_ratio * 0.8)


# ── Common semiconductor abbreviation expansions ──
_ABBREVIATIONS = {
    "fd": "fully depleted",
    "soi": "silicon on insulator",
    "fdsoi": "fully depleted silicon on insulator",
    "cmos": "complementary metal oxide semiconductor",
    "mosfet": "metal oxide semiconductor field effect transistor",
    "nmos": "n channel mosfet",
    "pmos": "p channel mosfet",
    "ota": "operational transconductance amplifier",
    "opamp": "operational amplifier",
    "adc": "analog to digital converter",
    "dac": "digital to analog converter",
    "pll": "phase locked loop",
    "vco": "voltage controlled oscillator",
    "ldo": "low dropout regulator",
    "esd": "electrostatic discharge",
    "snr": "signal to noise ratio",
    "thd": "total harmonic distortion",
    "bw": "bandwidth",
    "gbw": "gain bandwidth product",
    "aer": "address event representation",
    "snn": "spiking neural network",
    "dpi": "differential pair integrator",
    "ahp": "after hyperpolarization",
    "epsc": "excitatory post synaptic current",
    "relu": "rectified linear unit",
    "lif": "leaky integrate and fire",
    # CIS / image sensor abbreviations
    "cds": "correlated double sampling",
    "cis": "cmos image sensor",
    "ppd": "pinned photodiode",
    "tx": "transfer gate transistor",
    "fwc": "full well capacity",
    "qe": "quantum efficiency",
    "dsnu": "dark signal non uniformity",
    "prnu": "photo response non uniformity",
    "fpn": "fixed pattern noise",
}


def _normalize_name(name: str) -> str:
    """Normalize a concept name for comparison."""
    import re
    # Lowercase
    s = name.lower()
    # Replace separators with spaces
    s = re.sub(r'[-_/]+', ' ', s)
    # Remove parenthetical abbreviations like "(AER)"
    s = re.sub(r'\([^)]*\)', '', s)
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    # Expand known abbreviations (only if the name is short / likely an acronym)
    tokens = s.split()
    expanded = []
    for t in tokens:
        if t in _ABBREVIATIONS:
            expanded.append(_ABBREVIATIONS[t])
        else:
            expanded.append(t)
    return " ".join(expanded)


def _lcs_ratio(a: str, b: str) -> float:
    """Longest common subsequence ratio — handles partial name overlaps."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0.0

    # Optimized LCS length (O(n) space)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr

    lcs_len = prev[n]
    return (2.0 * lcs_len) / (m + n)


# ── F6: equation/parameter normalization (exact-match, no fuzzy tier) ──


def _normalize_latex(latex: str) -> str:
    """Conservative LaTeX normalization for exact-match equation comparison (F6).

    Deliberately NOT symbolic equivalence (no reordering, no algebraic rewriting) —
    only collapses formatting variance that renders identically:
      - a single layer of surrounding math delimiters ($...$, $$...$$, \\(...\\), \\[...\\])
        some extractions include and others don't
      - \\left / \\right — pure delimiter-sizing hints, semantically inert
      - a single alphanumeric character in a ^/_ script brace ('x^{2}' == 'x^2';
        multi-char scripts like 'V_{DS}' are left alone — removing those braces
        WOULD change meaning ('V_{DS}' vs 'V_D S'))
      - all whitespace — LaTeX math source treats it as insignificant

    Case is preserved deliberately: 'V_{GS}' and 'v_{gs}' are different circuit
    quantities in this domain (large-signal vs small-signal), unlike concept names.
    Live reconnaissance (2026-07-10, 842 Equation nodes) validated this catches real
    formatting-only duplicates a raw/trim-only comparison misses (29 normalized-duplicate
    groups vs 21 raw-exact groups) without needing anything beyond formatting collapse.
    """
    if not latex:
        return ""
    s = latex.strip()
    for open_d, close_d in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")):
        if s.startswith(open_d) and s.endswith(close_d) and len(s) >= len(open_d) + len(close_d):
            s = s[len(open_d): len(s) - len(close_d)].strip()
            break
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r'([_^])\{([A-Za-z0-9])\}', r'\1\2', s)
    s = re.sub(r'\s+', '', s)
    return s


def _normalize_plain(s: str) -> str:
    """Case/trim/whitespace-only normalization (F6 parameter name/unit comparison).

    Deliberately narrower than ``_normalize_name`` above: no abbreviation expansion,
    no punctuation stripping, no parenthetical removal. The parameter-matching
    constraint is an exact match on a normalized string, not a fuzzy concept-identity
    decision — reusing ``_normalize_name`` here would let e.g. a parameter literally
    named "GBW" auto-match one named "Gain Bandwidth Product" regardless of whether
    they denote the same quantity, which is exactly the kind of silent-merge risk
    this narrower function avoids.
    """
    if not s:
        return ""
    return re.sub(r'\s+', ' ', s.strip().lower())


def _best_parameter_candidate(
    norm_name: str, norm_units: str, candidates: list[dict],
) -> dict | None:
    """Pick the best name-matching candidate under the unit-agreement rule (F6).

    Among candidates whose ``norm_name`` matches exactly: a candidate whose
    ``norm_units`` conflicts with ``norm_units`` (both non-empty, different) is
    disqualified outright — it is NEVER chosen, even if no better candidate exists
    (unit conflict -> NEW, never SAME). Among the remaining eligible candidates, an
    exact non-empty unit match ranks above a gap-fill pairing (one or both sides
    missing a unit), so a same-named, same-unit node is preferred over a same-named,
    unit-less node when both are present in the graph.
    """
    if not norm_name:
        return None
    best: dict | None = None
    best_score = -1
    for c in candidates:
        if c["norm_name"] != norm_name:
            continue
        if norm_units and c["norm_units"]:
            if norm_units != c["norm_units"]:
                continue  # unit conflict -- disqualified, never SAME
            score = 2  # exact non-empty unit agreement
        else:
            score = 1  # gap-fill: one or both sides missing a unit
        if score > best_score:
            best_score = score
            best = c
    return best
