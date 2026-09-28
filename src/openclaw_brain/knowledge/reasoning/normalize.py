"""Normalize raw LLM JSON output to match GraphDelta Pydantic schema.

Local/EXO models often produce structurally correct JSON but with
different field names (e.g. "id" instead of "proposed_id") or
simplified types (e.g. insights as plain strings).

This module maps common deviations to the canonical schema so that
Pydantic validation succeeds without requiring strict schema enforcement
from the inference server.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

# ── Valid enum values (kept in sync with schema.py) ──────────────────

_VALID_LABELS = {
    # Aligned with store._LLM_PROPOSABLE_LABELS (F8): the raw-JSON fallback path must not
    # accept labels the structured path's store boundary would reject anyway — operational
    # (Memory/Session/SkillRun/Entity/Source/SourceChunk), curated-write-path
    # (Hypothesis/DesignDecision/BenchResult/Learner/Assessment), and projector-owned
    # (Specimen/ClaimCard/Regularity) labels are all invalid in LLM-authored deltas.
    # Belt-and-suspenders: store.apply_delta enforces the same set with rejection counting.
    "Concept", "Equation", "Principle", "CircuitTopology", "Parameter",
    "Assumption", "Insight",
}

_VALID_REL_TYPES = {
    "USES_EQUATION", "HAS_PARAMETER", "DEPENDS_ON", "ASSUMES", "DERIVED_FROM",
    "APPROXIMATION_OF", "EXTRACTED_FROM", "CONFIRMS", "CONTRADICTS", "REFINES",
    "RELATES_TO", "SOLVES_PROBLEM", "INTRODUCES_PROBLEM", "TRADES_OFF",
    "DESIGN_RULE", "EVOLVES_TO", "SUB_BLOCK", "TOPOLOGY_VARIANT",
    "COMPENSATED_BY", "MODELS_BEHAVIOR", "VARIABLE_MAPS_TO", "BRIDGES_TO",
    "TESTS_HYPOTHESIS", "DECIDED_BY", "SUPERSEDES", "FALSIFIED_BY",
    "MOTIVATED_BY", "RECORDED", "PROMOTED_FROM", "EXECUTED",
}

# Map common LLM-generated labels to valid NodeLabel values
_LABEL_ALIASES: dict[str, str] = {
    "concept": "Concept",
    "equation": "Equation",
    "principle": "Principle",
    "circuit_topology": "CircuitTopology",
    "circuittopology": "CircuitTopology",
    "topology": "CircuitTopology",
    "circuit": "CircuitTopology",
    "parameter": "Parameter",
    "assumption": "Assumption",
    "insight": "Insight",
    "hypothesis": "Hypothesis",
    # Common LLM hallucinated labels → Concept
    "model": "Concept",
    "method": "Concept",
    "technique": "Concept",
    "algorithm": "Concept",
    "framework": "Concept",
    "architecture": "Concept",
    "component": "Concept",
    "module": "Concept",
    "system": "Concept",
    "process": "Concept",
    "mechanism": "Concept",
    "metric": "Parameter",
    "measurement": "Parameter",
    "formula": "Equation",
    "law": "Principle",
    "rule": "Principle",
    "theory": "Principle",
    "dataset": "Concept",
    "tool": "Concept",
    "technology": "Concept",
    "feature": "Concept",
    "task": "Concept",
    "problem": "Concept",
    "solution": "Concept",
}

# Map common LLM-generated rel types to valid RelType values
_REL_TYPE_ALIASES: dict[str, str] = {
    "uses": "USES_EQUATION",
    "has_parameter": "HAS_PARAMETER",
    "depends_on": "DEPENDS_ON",
    "relates_to": "RELATES_TO",
    "sub_block": "SUB_BLOCK",
    "part_of": "SUB_BLOCK",
    "contains": "SUB_BLOCK",
    "implements": "DEPENDS_ON",
    "extends": "EVOLVES_TO",
    "improves": "EVOLVES_TO",
    "based_on": "DERIVED_FROM",
    "derived_from": "DERIVED_FROM",
    "variant_of": "TOPOLOGY_VARIANT",
    "solves": "SOLVES_PROBLEM",
    "solves_problem": "SOLVES_PROBLEM",
    "introduces_problem": "INTRODUCES_PROBLEM",
    "trades_off": "TRADES_OFF",
    "bridges_to": "BRIDGES_TO",
    "is_a": "RELATES_TO",
    "related_to": "RELATES_TO",
    "associated_with": "RELATES_TO",
    "enables": "DEPENDS_ON",
    "requires": "DEPENDS_ON",
    "input_to": "DEPENDS_ON",
    "output_of": "DERIVED_FROM",
    "applied_to": "RELATES_TO",
    "compared_to": "RELATES_TO",
    "evaluates": "RELATES_TO",
    "optimizes": "RELATES_TO",
    "complements": "RELATES_TO",
}


# ── Field name mappings ──────────────────────────────────────────────

_NODE_FIELD_MAP: dict[str, str] = {
    "id": "proposed_id",
    "node_id": "proposed_id",
    "name": "canonical_name",
    "type": "label",
    "node_type": "label",
    "node_label": "label",
    "category": "label",
    "layer": "knowledge_layer",
    "reason": "reasoning",
    "justification": "reasoning",
    "explanation": "reasoning",
    "evidence": "evidence_chunk_ids",
    "chunk_ids": "evidence_chunk_ids",
    "props": "properties",
    "attributes": "properties",
    "metadata": "properties",
}

_EDGE_FIELD_MAP: dict[str, str] = {
    "source": "source_ref",
    "source_id": "source_ref",
    "source_node": "source_ref",
    "source_node_id": "source_ref",
    "src": "source_ref",
    "from": "source_ref",
    "from_id": "source_ref",
    "target": "target_ref",
    "target_id": "target_ref",
    "target_node": "target_ref",
    "target_node_id": "target_ref",
    "dst": "target_ref",
    "to": "target_ref",
    "to_id": "target_ref",
    "type": "relationship_type",
    "rel_type": "relationship_type",
    "relation": "relationship_type",
    "edge_type": "relationship_type",
    "reason": "rationale",
    "explanation": "rationale",
    "justification": "rationale",
    "evidence": "evidence_chunk_ids",
    "chunk_ids": "evidence_chunk_ids",
}

_NODE_UPDATE_FIELD_MAP: dict[str, str] = {
    "id": "existing_node_id",
    "node_id": "existing_node_id",
    "type": "label",
    "node_type": "label",
    "node_label": "label",
    "reason": "reasoning",
    "changes": "updates",
    "fields": "updates",
}

_EDGE_REINFORCEMENT_FIELD_MAP: dict[str, str] = {
    "source": "source_ref",
    "source_id": "source_ref",
    "target": "target_ref",
    "target_id": "target_ref",
    "type": "relationship_type",
    "rel_type": "relationship_type",
    "relation": "relationship_type",
    "chunk_ids": "confirming_chunk_ids",
    "evidence": "confirming_chunk_ids",
    "note": "new_evidence_note",
    "evidence_note": "new_evidence_note",
}

_INSIGHT_FIELD_MAP: dict[str, str] = {
    "title": "statement",
    "text": "statement",
    "insight": "statement",
    "observation": "statement",
    "content": "statement",
    "description": "statement",
    "concepts": "related_concept_ids",
    "concept_ids": "related_concept_ids",
    "related_concepts": "related_concept_ids",
    "node_ids": "related_concept_ids",
    "type": "bridge_type",
}

_DELTA_FIELD_MAP: dict[str, str] = {
    "nodes": "new_nodes",
    "edges": "new_edges",
    "updates": "updated_nodes",
    "node_updates": "updated_nodes",
    "reinforcements": "reinforced_edges",
    "reinforced": "reinforced_edges",
}


# ── Core normalization ───────────────────────────────────────────────

def _remap_fields(obj: dict[str, Any], field_map: dict[str, str]) -> dict[str, Any]:
    """Remap keys in a dict using field_map. Canonical names take priority."""
    result: dict[str, Any] = {}
    for key, value in obj.items():
        canonical = field_map.get(key, key)
        # Don't overwrite if the canonical key already exists in the original dict
        if canonical in obj and canonical != key:
            result[key] = value  # keep original, canonical version will be added naturally
        else:
            result[canonical] = value
    return result


def _normalize_label(raw_label: Any) -> str:
    """Normalize a node label to a valid NodeLabel enum value."""
    if not isinstance(raw_label, str):
        return "Concept"
    # Already valid?
    if raw_label in _VALID_LABELS:
        return raw_label
    # Try alias lookup (case-insensitive)
    lower = raw_label.lower().replace(" ", "_").replace("-", "_")
    if lower in _LABEL_ALIASES:
        return _LABEL_ALIASES[lower]
    # Try case-insensitive match against valid labels
    for valid in _VALID_LABELS:
        if valid.lower() == lower:
            return valid
    return "Concept"  # safe default


def _normalize_rel_type(raw_type: Any) -> str:
    """Normalize a relationship type to a valid RelType enum value."""
    if not isinstance(raw_type, str):
        return "RELATES_TO"
    # Already valid?
    upper = raw_type.upper().replace(" ", "_").replace("-", "_")
    if upper in _VALID_REL_TYPES:
        return upper
    # Try alias lookup
    lower = raw_type.lower().replace(" ", "_").replace("-", "_")
    if lower in _REL_TYPE_ALIASES:
        return _REL_TYPE_ALIASES[lower]
    return "RELATES_TO"  # safe default


def _normalize_node(node: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single node proposal."""
    node = _remap_fields(node, _NODE_FIELD_MAP)

    # Ensure proposed_id is snake_case string; derive from canonical_name when missing
    if "proposed_id" not in node and isinstance(node.get("canonical_name"), str) and node["canonical_name"].strip():
        node["proposed_id"] = "auto_" + node["canonical_name"]
    if "proposed_id" in node and isinstance(node["proposed_id"], str):
        node["proposed_id"] = node["proposed_id"].strip().replace(" ", "_").replace("-", "_").lower()

    # Normalize label to valid enum value
    if "label" in node:
        node["label"] = _normalize_label(node["label"])
    else:
        node["label"] = "Concept"  # default when missing

    # If reasoning is missing, synthesize from description
    if "reasoning" not in node and "description" in node:
        node["reasoning"] = f"New concept: {node.get('description', '')}"
    elif "reasoning" not in node:
        node["reasoning"] = f"New node: {node.get('canonical_name', node.get('proposed_id', 'unknown'))}"

    return node


def _normalize_edge(edge: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single edge proposal."""
    edge = _remap_fields(edge, _EDGE_FIELD_MAP)

    # Normalize relationship_type to valid enum value
    if "relationship_type" in edge:
        edge["relationship_type"] = _normalize_rel_type(edge["relationship_type"])
    else:
        edge["relationship_type"] = "RELATES_TO"

    # Ensure confidence exists
    if "confidence" not in edge:
        edge["confidence"] = 0.7

    # Ensure rationale exists
    if "rationale" not in edge:
        edge["rationale"] = f"{edge.get('source_ref', '?')} → {edge.get('relationship_type', '?')} → {edge.get('target_ref', '?')}"

    return edge


def _normalize_node_update(update: dict[str, Any]) -> dict[str, Any]:
    """Normalize a node update."""
    update = _remap_fields(update, _NODE_UPDATE_FIELD_MAP)
    if "reasoning" not in update:
        update["reasoning"] = "Update from extraction"
    return update


def _normalize_edge_reinforcement(reinf: dict[str, Any]) -> dict[str, Any]:
    """Normalize an edge reinforcement."""
    reinf = _remap_fields(reinf, _EDGE_REINFORCEMENT_FIELD_MAP)
    if "relationship_type" in reinf:
        reinf["relationship_type"] = _normalize_rel_type(reinf["relationship_type"])
    return reinf


def _normalize_insight(insight: Any) -> dict[str, Any]:
    """Normalize an insight — handles both dict and plain string."""
    if isinstance(insight, str):
        return {
            "statement": insight,
            "related_concept_ids": [],
            "bridge_type": "within-domain",
            "confidence": 0.5,
        }
    if isinstance(insight, dict):
        insight = _remap_fields(insight, _INSIGHT_FIELD_MAP)
        # Ensure related_concept_ids is a list
        if "related_concept_ids" not in insight:
            insight["related_concept_ids"] = []
        elif isinstance(insight["related_concept_ids"], str):
            insight["related_concept_ids"] = [insight["related_concept_ids"]]
        return insight
    return {"statement": str(insight), "related_concept_ids": [], "confidence": 0.5}


def _coerce_parsed(value: Any) -> Any:
    """Coerce a stringified-JSON value to its parsed (list/dict) form.

    Belt-and-suspenders for frontier models that double-encode: instead of emitting a
    native JSON list/object for a field (or the whole payload), they emit a JSON *string*
    whose contents are themselves JSON (e.g. `new_nodes` arrives as `'[{"proposed_id": ...}]'`
    instead of `[{"proposed_id": ...}]`). Applied defensively — non-string input, or a string
    that isn't valid JSON, passes through unchanged, so already-parsed clean input (the
    deepseek / sonnet-4-6 path) is byte-identical in behavior.
    """
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "{[":
        return value
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError, ValueError):
        return value


def _as_list(value: Any) -> list:
    """Best-effort coercion of a GraphDelta container field (new_nodes, new_edges, ...) to a
    list, tolerating one level of JSON-string double-encoding via `_coerce_parsed`. Anything
    that still isn't a list after that (missing field, wrong type) becomes an empty list —
    consistent with `raw.get(field, [])`'s existing default — rather than raising here; a
    malformed element inside a genuine list is still caught per-element by `_salvage`."""
    value = _coerce_parsed(value)
    return value if isinstance(value, list) else []


def _element_kind(item: Any) -> str:
    """Best-effort sub-classification of a raw (already field-remapped) element for
    salvage-drop diagnostics. Reads only a single small identifying field (node label, or edge
    relationship_type — both always present post-normalization, defaulted if the LLM omitted
    them) — never the full item — so the warning stays a short, safe aggregate rather than a
    payload dump of (possibly large or sensitive) LLM-generated text."""
    if not isinstance(item, dict):
        return "unknown"
    for key in ("label", "relationship_type"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _salvage(items: list, model: Any, kind: str) -> list[dict[str, Any]]:
    """Element-wise validation: keep what passes, drop (and log) what doesn't.

    A single malformed element from a local model must not discard the whole
    delta — losing one edge is recoverable, losing the chunk is not.
    """
    kept: list[dict[str, Any]] = []
    dropped_kinds: Counter[str] = Counter()
    for item in items:
        try:
            model(**item)
            kept.append(item)
        except Exception:
            dropped_kinds[_element_kind(item)] += 1
    if dropped_kinds:
        total_dropped = sum(dropped_kinds.values())
        breakdown = ", ".join(f"{k}×{n}" for k, n in sorted(dropped_kinds.items()))
        logger.warning(
            "normalize_graph_delta: dropped %d invalid %s element(s) [%s], kept %d",
            total_dropped, kind, breakdown, len(kept),
        )
    return kept


def normalize_graph_delta(raw: dict[str, Any] | str) -> dict[str, Any]:
    """Normalize a raw JSON dict to match GraphDelta schema.

    Handles:
    - Field name remapping (e.g. "id" → "proposed_id")
    - Insight strings → InsightProposal objects
    - Missing optional fields filled with defaults
    - Element-wise salvage: invalid elements are dropped instead of failing the delta
    - Double-encoding: `raw` itself, or any container field (new_nodes/new_edges/...),
      arriving as a JSON string instead of an already-parsed dict/list (some frontier models
      double-encode their structured output)
    """
    from openclaw_brain.knowledge.graph.schema import (
        EdgeProposal,
        EdgeReinforcement,
        InsightProposal,
        NodeProposal,
        NodeUpdate,
    )

    # Defensive: some frontier models double-encode — the whole payload arrives as a JSON
    # string rather than an already-parsed dict. Coerce once before remapping; a plain dict
    # (the deepseek / sonnet-4-6 path) passes through _coerce_parsed unchanged.
    raw = _coerce_parsed(raw)
    if not isinstance(raw, dict):
        raw = {}

    raw = _remap_fields(raw, _DELTA_FIELD_MAP)

    result: dict[str, Any] = {}
    result["new_nodes"] = _salvage([_normalize_node(n) for n in _as_list(raw.get("new_nodes", []))], NodeProposal, "new_nodes")
    result["updated_nodes"] = _salvage([_normalize_node_update(u) for u in _as_list(raw.get("updated_nodes", []))], NodeUpdate, "updated_nodes")
    result["new_edges"] = _salvage([_normalize_edge(e) for e in _as_list(raw.get("new_edges", []))], EdgeProposal, "new_edges")
    result["reinforced_edges"] = _salvage([_normalize_edge_reinforcement(r) for r in _as_list(raw.get("reinforced_edges", []))], EdgeReinforcement, "reinforced_edges")
    result["insights"] = _salvage([_normalize_insight(i) for i in _as_list(raw.get("insights", []))], InsightProposal, "insights")

    return result


# ── JSON extraction ──────────────────────────────────────────────────

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from LLM text output, stripping markdown fences if present."""
    text = text.strip()

    # Try direct parse first
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try extracting from markdown code block
    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Last resort: find the first { and last } and try parsing
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from LLM output ({len(text)} chars)")


# ── Parse-failure diagnostics ────────────────────────────────────────
#
# extract_json's ValueError carries only a LENGTH. The 2026-07 lecture batch hit it on 24
# distinct chunks (response length p50 10.8k, max 22.7k chars, 4 incidents under 2k incl.
# 0-char empties) and, because the body was never logged, it was IMPOSSIBLE to tell whether
# those long failures were TRUNCATED JSON (raise the bound) or well-formed PROSE with no JSON
# at all (fix the prompt / the model). Storing whole bodies is not the answer — these two
# shapes are distinguishable from the TAIL alone: truncated JSON stops mid-token with no
# closing brace, prose ends on a sentence. Hence a capped tail, not a body dump.
# Nothing is redacted: this is our own reasoning output, never a credential surface (and the
# call site already routes exception text through redact_secrets separately).

_PARSE_FAILURE_TAIL_CHARS = 200

# An UNTERMINATED opening fence — the shape a truncated fenced body ends up in once the model
# runs out of budget before it can close the block.
_OPEN_FENCE_RE = re.compile(r"^```[A-Za-z0-9_+-]*[ \t]*\r?\n?")


def _coerce_response_text(value: Any) -> str:
    """Flatten anything a LangChain ``.content`` can hold into a single string.

    ``AIMessage.content`` is typed ``str | list[str | dict]``: providers that emit multi-part
    content (tool/thinking/text blocks) hand back a LIST, and the original implementation's
    ``text.strip()`` raised AttributeError on it — INSIDE the pipeline's except block, turning
    graceful degradation into an unhandled crash. Diagnostics must never be able to do that,
    so every non-str shape is coerced here rather than assumed away.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # {"type": "text", "text": ...} is the common multi-part block; anything else
                # (thinking/tool_use blocks) is kept verbatim so the tail still classifies.
                inner = item.get("text")
                parts.append(inner if isinstance(inner, str) else repr(item))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(value)


def _strip_code_fences(text: str) -> str:
    """Remove markdown fences so the structural booleans describe the JSON, not the wrapper.

    Without this, a COMPLETE ```json … ``` body ends with a backtick, so
    ``ends_with_close_brace`` reads False — byte-identical booleans to a genuinely truncated
    body, i.e. the booleans classified nothing (the whole point of emitting them). A complete
    fenced block is unwrapped by the same regex ``extract_json`` uses; a truncated one has no
    closing fence to match, so only its opener is dropped and the body reads as still-open.
    """
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    opener = _OPEN_FENCE_RE.match(text)
    if opener:
        return text[opener.end():].strip()
    return text


def describe_parse_failure(
    text: Any = None,
    *,
    finish_reason: str | None = None,
    tail_chars: int = _PARSE_FAILURE_TAIL_CHARS,
) -> str:
    """Compact, greppable classification detail for an extract_json failure.

    Emits: RAW length, whether ANY '{' is present (no brace at all ⇒ prose/refusal, not a
    cut-off object), whether the body actually closes, the provider finish_reason when the
    caller could reach one ('length' ⇒ truncation, independently of the tail's shape), and the
    LAST ``tail_chars`` characters repr'd so newlines cannot break the log line.

    ``len=`` is the RAW length so it stays comparable to the historical series: the 2026-07
    distribution (p50 10,839 / max 22,700 chars) and ``extract_json``'s own ValueError count
    the body as received. ``stripped_len=`` is added only when whitespace makes them differ.

    TOTAL and never-raising by contract: this runs inside the pipeline's except block, where an
    exception would replace a graceful degradation with a crash. Any input type is accepted
    (``str``, a multi-part ``content`` list, None, an arbitrary object); if anything at all goes
    wrong the function still returns a string saying so.

    Pure and side-effect free; ``extract_json``'s signature, exception type and message prefix
    are deliberately untouched (tests and log-greps depend on the
    "Could not extract JSON from LLM output (N chars)" prefix).
    """
    if text is None:
        return "parse_failure: response=None"

    try:
        raw = _coerce_response_text(text)
        stripped = raw.strip()
        body = _strip_code_fences(stripped)
        tail = stripped[-tail_chars:] if tail_chars > 0 else ""
        length = f"len={len(raw)}"
        if len(stripped) != len(raw):
            length += f" stripped_len={len(stripped)}"
        return (
            f"parse_failure: {length} "
            f"has_open_brace={'{' in body} "
            f"ends_with_close_brace={body.endswith(('}', ']'))} "
            f"finish_reason={finish_reason!r} "
            f"tail_{tail_chars}={tail!r}"
        )
    except Exception as exc:  # pragma: no cover — belt-and-braces; the body cannot raise today
        return (
            f"parse_failure: diagnostics unavailable "
            f"({type(exc).__name__}: {exc}; response type={type(text).__name__})"
        )
