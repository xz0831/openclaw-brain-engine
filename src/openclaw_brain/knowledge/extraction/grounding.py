"""Source grounding verification — checks that extracted entities are grounded in source text.

After extraction, verifies that each concept, equation, and parameter can be
traced back to the original chunk text. Flags or removes ungrounded extractions
to prevent hallucination from entering the knowledge graph.
"""

from __future__ import annotations

import logging
import re

from openclaw_brain.knowledge.extraction.models import (
    ConceptMention,
    EquationMention,
    ExtractionResult,
    ParameterMention,
    RawEdge,
)

logger = logging.getLogger(__name__)

_DIRECTIONAL_RE = re.compile(
    r"\b(?:increas\w*|decreas\w*|improv\w*|degrad\w*|proportional|linear|scal\w*)\b",
    re.IGNORECASE,
)
# Sentinels wrapping a VLM-generated figure DESCRIPTION inside chunk text (added by the chunker).
# The caption/labels/body around them are independent (from the source); the description is the
# generator's own text. Grounding must NOT use the description as the support set for the edges it
# itself produced — that is self-grounding (a VLM hallucination validating itself). So the support
# set excludes these spans, requiring figure-derived edges to anchor to an INDEPENDENT source.
FIG_VLM_OPEN = "⟦FIGVLM⟧"
FIG_VLM_CLOSE = "⟦/FIGVLM⟧"
_FIG_VLM_SPAN_RE = re.compile(re.escape(FIG_VLM_OPEN) + r".*?" + re.escape(FIG_VLM_CLOSE), re.DOTALL)
_FIG_VLM_MARKER_RE = re.compile(re.escape(FIG_VLM_OPEN) + r"|" + re.escape(FIG_VLM_CLOSE))


def independent_support_text(chunk_text: str) -> str:
    """Support set for grounding: chunk text with VLM figure-DESCRIPTION spans removed (caption/
    body/labels kept). Prevents a figure-derived edge from grounding against the VLM's own prose."""
    return _FIG_VLM_SPAN_RE.sub("  ", chunk_text)


def strip_figure_markers(chunk_text: str) -> str:
    """Remove only the bare sentinels, keeping the description content — for the EXTRACTOR input and
    stored chunk text (the LLM/store should see clean prose, only grounding excludes the span)."""
    return _FIG_VLM_MARKER_RE.sub("", chunk_text)


_UNIVERSAL_RE = re.compile(r"\b(?:total|all|always|overall|every|any)\b", re.IGNORECASE)
_L_MIN_EQUATION_LCS = 4
_UNICODE_GREEK = {
    "\u03b1": "alpha",
    "\u03b2": "beta",
    "\u03b3": "gamma",
    "\u03b4": "delta",
    "\u03b7": "eta",
    "\u03b8": "theta",
    "\u03bb": "lambda",
    "\u03bc": "mu",
    "\u03c0": "pi",
    "\u03c1": "rho",
    "\u03c3": "sigma",
    "\u03c4": "tau",
    "\u03c6": "phi",
    "\u03d5": "phi",
    "\u03c9": "omega",
}
_LATEX_WORD_COMMANDS = {
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\gamma": "gamma",
    r"\delta": "delta",
    r"\eta": "eta",
    r"\theta": "theta",
    r"\lambda": "lambda",
    r"\mu": "mu",
    r"\pi": "pi",
    r"\rho": "rho",
    r"\sigma": "sigma",
    r"\tau": "tau",
    r"\phi": "phi",
    r"\omega": "omega",
    r"\sinc": "sinc",
    r"\sin": "sin",
    r"\cos": "cos",
    r"\tan": "tan",
    r"\log": "log",
    r"\ln": "ln",
    r"\exp": "exp",
    r"\sqrt": "sqrt",
    r"\min": "min",
    r"\max": "max",
}


def verify_grounding(extraction: ExtractionResult, chunk_text: str) -> ExtractionResult:
    """Verify that extracted entities are grounded in the source text.

    Removes or downgrades entities that cannot be traced to the text.
    Returns a filtered ExtractionResult.

    Args:
        extraction: The raw extraction output.
        chunk_text: The original chunk text.

    Returns:
        ExtractionResult with ungrounded entities removed.
    """
    # Support set EXCLUDES VLM figure-description spans (anti self-grounding); see module header.
    support = independent_support_text(chunk_text)
    text_lower = support.lower()
    eq_chunk_norm = _eq_normalize(support)

    grounded_concepts = []
    for concept in extraction.concepts:
        if _concept_is_grounded(concept, text_lower):
            _flag_scope_widening(concept, chunk_text)
            grounded_concepts.append(concept)
        else:
            logger.debug("Ungrounded concept removed: %s", concept.name)

    grounded_equations = []
    for eq in extraction.equations:
        if _equation_is_grounded(eq, eq_chunk_norm):
            grounded_equations.append(eq)
        else:
            logger.debug("Ungrounded equation removed: %s", eq.latex[:50])

    grounded_params = []
    for param in extraction.parameters:
        if _parameter_is_grounded(param, text_lower):
            grounded_params.append(param)
        else:
            logger.debug("Ungrounded parameter removed: %s (%s)", param.name, param.symbol)

    # Filter edges: both source and target must still exist
    grounded_names = {c.name for c in grounded_concepts}
    grounded_names |= {eq.latex for eq in grounded_equations}
    grounded_names |= {p.name for p in grounded_params}
    grounded_names |= {p.symbol for p in grounded_params}

    grounded_edges = []
    for edge in extraction.raw_edges:
        if _edge_is_grounded(edge, grounded_names):
            grounded_edges.append(edge)
        else:
            logger.debug("Ungrounded edge removed: %s -> %s", edge.source_name, edge.target_name)

    removed = (
        len(extraction.concepts) - len(grounded_concepts)
        + len(extraction.equations) - len(grounded_equations)
        + len(extraction.parameters) - len(grounded_params)
        + len(extraction.raw_edges) - len(grounded_edges)
    )
    if removed > 0:
        logger.info(
            "Grounding: removed %d ungrounded items from chunk %s",
            removed, extraction.chunk_id,
        )

    return ExtractionResult(
        chunk_id=extraction.chunk_id,
        concepts=grounded_concepts,
        equations=grounded_equations,
        parameters=grounded_params,
        raw_edges=grounded_edges,
    )


def _flag_scope_widening(concept: ConceptMention, chunk_text: str) -> None:
    """Flag directional/quantitative descriptions that universalize beyond the text."""
    description = concept.description or ""
    if not _DIRECTIONAL_RE.search(description):
        return

    universal_tokens = [m.group(0).lower() for m in _UNIVERSAL_RE.finditer(description)]
    if not universal_tokens:
        return

    chunk_lower = chunk_text.lower()
    for token in universal_tokens:
        if not re.search(rf"\b{re.escape(token)}\b", chunk_lower, re.IGNORECASE):
            if "scope_widened" not in concept.grounding_flags:
                concept.grounding_flags.append("scope_widened")
            return


def _concept_is_grounded(concept: ConceptMention, text_lower: str) -> bool:
    """Check if a concept name appears (or has significant overlap) in the text."""
    name_lower = concept.name.lower()

    # Direct substring match
    if name_lower in text_lower:
        return True

    # Check individual significant words (≥3 chars) — at least 50% must appear
    words = [w for w in re.split(r'[\s\-_/]+', name_lower) if len(w) >= 3]
    if not words:
        return True  # Too short to verify, keep it

    matches = sum(1 for w in words if w in text_lower)
    return matches / len(words) >= 0.5


def _equation_is_grounded(eq: EquationMention, chunk_norm: str) -> bool:
    """Check that an equation body has structural support in the chunk."""
    if not eq.variables:
        return True

    en = _eq_normalize(eq.latex)
    if not en:
        return True

    chunk_norm = _eq_normalize(chunk_norm)
    s = longest_common_substring(en, chunk_norm)
    if len(s) < _L_MIN_EQUATION_LCS:
        return False
    if any(ch.isdigit() for ch in s):
        return True
    for vt in _eq_var_tokens(eq.variables):
        if s in vt:
            return False
    return True


def _eq_normalize(s: str) -> str:
    """Normalize equation/chunk text to compact ASCII alphanumerics."""
    text = s.lower()
    for greek, replacement in _UNICODE_GREEK.items():
        text = text.replace(greek, replacement)
    for command, replacement in sorted(
        _LATEX_WORD_COMMANDS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        text = re.sub(re.escape(command) + r"(?![a-z])", replacement, text)
    text = re.sub(r"\\[a-z]+", " ", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def _eq_var_tokens(variables: list[str]) -> set[str]:
    tokens = set()
    for variable in variables:
        head = re.split(r"[:(]", variable, maxsplit=1)[0]
        token = _eq_normalize(head)
        if token:
            tokens.add(token)
    return tokens


def longest_common_substring(a: str, b: str) -> str:
    """Return the longest contiguous substring shared by a and b."""
    if not a or not b:
        return ""

    previous = [0] * (len(b) + 1)
    best_len = 0
    best_end = 0
    for i, char_a in enumerate(a, start=1):
        current = [0] * (len(b) + 1)
        for j, char_b in enumerate(b, start=1):
            if char_a == char_b:
                current[j] = previous[j - 1] + 1
                if current[j] > best_len:
                    best_len = current[j]
                    best_end = i
        previous = current

    return a[best_end - best_len:best_end]


def _parameter_is_grounded(param: ParameterMention, text_lower: str) -> bool:
    """Check if a parameter's symbol or name appears in the text."""
    # Check symbol
    if param.symbol:
        sym_lower = param.symbol.lower().replace("\\", "")
        if sym_lower in text_lower:
            return True
        sym_collapsed = sym_lower.replace("_", "")
        if len(sym_collapsed) >= 2 and sym_collapsed in text_lower:
            return True

    # Check name
    if param.name:
        name_lower = param.name.lower()
        if name_lower in text_lower:
            return True
        # Check individual words
        words = [w for w in name_lower.split() if len(w) >= 3]
        if words:
            matches = sum(1 for w in words if w in text_lower)
            if matches / len(words) >= 0.5:
                return True

    return False


def _edge_is_grounded(edge: RawEdge, entity_names: set[str]) -> bool:
    """Check if both endpoints of an edge exist in the grounded entity set."""
    source_ok = edge.source_name in entity_names
    target_ok = edge.target_name in entity_names

    # Also check case-insensitive
    if not source_ok:
        source_ok = any(edge.source_name.lower() == n.lower() for n in entity_names)
    if not target_ok:
        target_ok = any(edge.target_name.lower() == n.lower() for n in entity_names)

    return source_ok and target_ok
