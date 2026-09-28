"""Production canonicalization layer — entity-identity + rel_type projection for KG construction.

Track 1 (production KG). This is the construction-side counterpart of the grader's
canonicalization (`experiments/reason_grade.py`): the SAME documented-alias registry,
generic-qualifier guard, salient/discriminator logic, and rich->v0 rel projection that the
grader uses for SCORING are promoted here for CONSTRUCTION, so the live pipeline stops
fragmenting one concept into many nodes ("Zero at the Origin of Frequency" vs canonical
"DC (Frequency Origin)"; "Auto-Zero Operation" vs "Autozero").

DELIBERATE TWO-COPY (Rick 2026-06-16): the grader copy stays sealed/stable; this is a faithful
port. `tests/test_canonicalization_conformance.py` runs identical cases through BOTH and asserts
behavioral identity, so the copies cannot silently diverge. Do NOT edit one without the other.

Public API:
  same_entity(a, b)        -> bool   : do two endpoint names denote the same physical entity?
  canonical_key(name)      -> str    : alias-normalized key for fast first-pass dedup (Tier 0)
  propose_merges(concepts) -> list[dict]: shadow-only merge proposals canonicalization would add
  project_rel_type(rel)    -> str    : rich (61) -> v0 (GraphDelta enum) projection
  alias_surface_forms(name)-> set[str]: registered entity cores present (for alias harvest)
"""
from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Mapping

# --- name canonicalization (verbatim from reason_grade) ---------------------
_SPACE_RE = re.compile(r"[\s_-]+")
_PUNCT_RE = re.compile(r"[^a-z0-9+/(). ]+")


def canonicalize_name(value: Any) -> str:
    """Canonical endpoint matching: lowercase and collapse spacing/hyphens."""
    text = str(value or "").lower()
    text = _SPACE_RE.sub(" ", text)
    text = _PUNCT_RE.sub("", text)
    return " ".join(text.split())


# --- §2 documented entity-alias registry (curated; Rick sign-off 2026-06-16) ---
# Each canonical key -> surface forms as canonicalized token-tuples (multi-token tuples
# matched as contiguous n-grams). CORE 5 only; refdes/symbols intentionally excluded.
_ALIAS_ENTITIES: dict[str, list[tuple[str, ...]]] = {
    "autozero": [("autozero",), ("auto", "zero"), ("az",), ("autozeroing",), ("auto", "zeroing")],
    "cds": [("cds",), ("correlated", "double", "sampling")],
    "sampleandhold": [("sample", "and", "hold"), ("sample", "hold"), ("s/h",), ("sh",)],
    "chopperstabilization": [("chs",), ("chopper", "stabilization"), ("chopper", "stabilisation")],
    "dcorigin": [("dc",), ("frequency", "origin")],
}
_ENTITY_CORES = set(_ALIAS_ENTITIES.keys())
_ALIAS_FORMS: list[tuple[tuple[str, ...], str]] = sorted(
    ((form, key) for key, forms in _ALIAS_ENTITIES.items() for form in forms),
    key=lambda kv: -len(kv[0]),
)
_GENERIC_QUALIFIERS = {
    "operation", "scheme", "technique", "method", "approach", "process",
    "step", "phase", "mode", "action", "circuit",
}

_SALIENT_STOP = {
    "a", "an", "the", "of", "to", "in", "at", "on", "and", "or", "for", "with",
    "as", "is", "by", "be", "via", "per", "from", "into", "that", "this",
    "these", "those", "its", "which", "than", "each", "such", "not", "but",
    "do", "does", "only", "any", "here",
}
_DISCRIMINATORS = {
    "factor", "threshold", "limit", "requirement", "technology", "inaccuracy",
    "error", "residual", "zero", "differential", "noise",
}


def _is_salient_short(token: str) -> bool:
    return any(ch.isdigit() for ch in token) or "/" in token or token in {"dc", "az"}


def _entity_normalize(tokens: list[str]) -> list[str]:
    """Replace recognized alias surface-form n-grams with their canonical key token."""
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        matched = False
        for form, key in _ALIAS_FORMS:
            k = len(form)
            if tuple(tokens[i : i + k]) == form:
                out.append(key)
                i += k
                matched = True
                break
        if not matched:
            out.append(tokens[i])
            i += 1
    return out


def _entity_salient(name: Any) -> set[str]:
    """Salient token set AFTER alias-canonicalizing entity surface forms."""
    raw = [t.strip("().") for t in canonicalize_name(name).split()]
    raw = [t for t in raw if t]
    norm = _entity_normalize(raw)
    out: set[str] = set()
    for token in norm:
        if not token or token in _SALIENT_STOP:
            continue
        if token in _ENTITY_CORES:
            out.add(token)
            continue
        if len(token) < 3:
            if _is_salient_short(token):
                out.add(token)
            continue
        if re.fullmatch(r"[0-9]+", token):
            continue
        out.add(token)
    return out


def _alias_match(gold_name: Any, model_name: Any) -> bool:
    gs, ms = _entity_salient(gold_name), _entity_salient(model_name)
    g_ent, m_ent = gs & _ENTITY_CORES, ms & _ENTITY_CORES
    if not g_ent or g_ent != m_ent:
        return False
    if (gs - _ENTITY_CORES) - _GENERIC_QUALIFIERS:
        return False
    if (ms - _ENTITY_CORES) - _GENERIC_QUALIFIERS:
        return False
    return True


def _salient(name: Any) -> list[str]:
    out: list[str] = []
    for token in canonicalize_name(str(name)).split():
        token = token.strip("().")
        if not token or token in _SALIENT_STOP:
            continue
        if len(token) < 3:
            if _is_salient_short(token):
                out.append(token)
            continue
        if re.fullmatch(r"[0-9]+", token):
            continue
        out.append(token)
    return out


def _endpoint_match(gold_name: Any, model_name: Any) -> bool:
    """True when two endpoint names denote the same physical concept."""
    if _alias_match(gold_name, model_name):
        return True
    gt, mt = set(_salient(gold_name)), set(_salient(model_name))
    if not gt or not mt:
        return False
    if gt == mt:
        return True
    inter = gt & mt
    if not inter:
        return False
    if (gt ^ mt) & _DISCRIMINATORS:
        return False
    short, long = (gt, mt) if len(gt) <= len(mt) else (mt, gt)
    if short <= long:
        if len(short) == 1:
            tok = next(iter(short))
            if tok.endswith("+") or tok.endswith("-"):
                return False
            return bool(any(ch.isdigit() for ch in tok) or "/" in tok or len(tok) <= 4)
        return True
    return len(inter) >= 2


# --- §6 rich -> v0 rel_type projection (verbatim from reason_grade) ----------
DROP_TYPES = frozenset({"OUTPERFORMS", "IS_DOMINANT_AT", "PRIORITIZES"})
PROJ: dict[str, str] = {
    "SOLVES_PROBLEM": "SOLVES_PROBLEM", "INTRODUCES_PROBLEM": "INTRODUCES_PROBLEM",
    "REDUCES": "DEPENDS_ON", "INCREASES": "DEPENDS_ON", "MODULATES": "DEPENDS_ON",
    "DETERMINES": "DEPENDS_ON", "HAS_ZERO_AT": "HAS_PARAMETER", "COMPOSED_OF": "SUB_BLOCK",
    "DRIVES": "DEPENDS_ON", "STRUCTURALLY_MODIFIES": "DEPENDS_ON", "VALID_WHEN": "ASSUMES",
    "CAUSES": "DEPENDS_ON", "ENABLES": "DEPENDS_ON", "SATISFIES": "DEPENDS_ON",
    "SIMILAR_TO": "RELATES_TO", "ALTERNATIVE_TO": "TOPOLOGY_VARIANT",
    "OUTPERFORMS": "DROP", "IS_DOMINANT_AT": "DROP", "PRIORITIZES": "DROP",
    "TRADES_OFF": "TRADES_OFF", "DEPENDS_ON": "DEPENDS_ON", "MODELS_BEHAVIOR": "MODELS_BEHAVIOR",
    "ASSUMES": "ASSUMES", "DERIVED_FROM": "DERIVED_FROM", "EVOLVES_TO": "EVOLVES_TO",
    "TOPOLOGY_VARIANT": "TOPOLOGY_VARIANT", "SUB_BLOCK": "SUB_BLOCK",
    "COMPENSATED_BY": "COMPENSATED_BY", "REFINES": "REFINES", "USES_EQUATION": "USES_EQUATION",
    "HAS_PARAMETER": "HAS_PARAMETER", "BRIDGES_TO": "BRIDGES_TO", "MOTIVATED_BY": "MOTIVATED_BY",
    "RELATES_TO": "RELATES_TO", "DESIGN_RULE": "DESIGN_RULE", "VARIABLE_MAPS_TO": "VARIABLE_MAPS_TO",
    "APPROXIMATION_OF": "APPROXIMATION_OF", "EXTRACTED_FROM": "EXTRACTED_FROM",
    "CONFIRMS": "CONFIRMS", "CONTRADICTS": "CONTRADICTS", "TESTS_HYPOTHESIS": "TESTS_HYPOTHESIS",
    "DECIDED_BY": "DECIDED_BY", "SUPERSEDES": "SUPERSEDES", "FALSIFIED_BY": "FALSIFIED_BY",
    "RECORDED": "RECORDED", "PROMOTED_FROM": "PROMOTED_FROM", "EXECUTED": "EXECUTED",
}


def _rel_type_value(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value or "").upper()


def project_rel_type(rel_type: Any) -> str:
    """Project a rich or current enum relation to the v0 GraphDelta enum."""
    value = _rel_type_value(rel_type)
    return PROJ.get(value, value)


# --- production-facing API ---------------------------------------------------
def same_entity(name_a: Any, name_b: Any) -> bool:
    """Do two endpoint/concept names denote the same physical entity? Alias-aware,
    generic-qualifier-tolerant, discriminator-guarded (technique 'CDS' != mechanism
    'CDS Baseband Transfer Function'). The authoritative identity check."""
    return _endpoint_match(name_a, name_b)


def canonical_key(name: Any) -> str:
    """Alias-normalized key for fast first-pass dedup. NOT authoritative (same_entity is) —
    a key collision is a strong dedup *candidate*; a key miss does NOT prove distinct."""
    raw = [t.strip("().") for t in canonicalize_name(name).split()]
    raw = [t for t in raw if t]
    return " ".join(_entity_normalize(raw))


def alias_surface_forms(name: Any) -> set[str]:
    """Registered entity cores present in a name (for confirmation-harvest discovery)."""
    return _entity_salient(name) & _ENTITY_CORES


# ── Production precision MERGE path (v2, 2026-06-16) ─────────────────────────
# The grader same_entity (token-containment) is recall-tuned for SCORING and is precision-poison
# for CONSTRUCTION at production all-pairs scale (experiments/canon/: 5046 live merges, gold P=0.50;
# embedding cosine alone tops at live-precision ~0.92 — a candidate gate, not a decider). The
# evidenced winner is TIERED:
#   MERGE  (auto, P≈1.0) : lexical identity — normalized-name / curated alias / acronym
#   REJECT               : qualifier or numeric conflict (the precision fix), or below candidate floor
#   VERIFY (LLM)         : embedding candidate (cos≥floor) not lexically decided — local qwen3.6-27b
#                          was the best verify judge (band agreement 0.90 vs gemini 0.85); the
#                          local Qwen families were deleted 2026-08-19/20, so cli.py consolidate
#                          now pins an unbenched-in-role stopgap — re-bench before live use
# These ADD a precision path; same_entity / canonical_key / project_rel_type stay frozen (grader
# conformance). Full comparison + evidence: experiments/CANONICALIZATION_PROMOTION.md.

# Generic words that, as the ONLY extra tokens of a longer name, do not change identity.
_GENERIC_EXTRA = _GENERIC_QUALIFIERS | {
    "node", "stage", "block", "structure", "effect", "technology", "model", "analysis",
}

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_ACRO_SPLIT = re.compile(r"[.\s_/-]+")


def numeric_conflict(name_a: Any, name_b: Any) -> bool:
    """Names carrying DIFFERENT numbers/section-refs are distinct items (Pitfall 27.2 vs 27.6,
    Example 5.1 vs 5.2, 3T Pixel vs 4T Pixel). Number-stripping makes their salient sets look
    identical, but the number IS the discriminator."""
    na = set(_NUM_RE.findall(str(name_a)))
    nb = set(_NUM_RE.findall(str(name_b)))
    return bool(na ^ nb)


def qualifier_conflict(name_a: Any, name_b: Any) -> bool:
    """One name is a SPECIALIZATION of the other → distinct entities. The precision fix that the
    grader's containment path lacks: 'Threshold Voltage' ⊊ 'Threshold Voltage Mismatch'
    (extra={mismatch}, non-generic) → conflict; 'Autozero' ⊊ 'Autozero Operation' (generic) → no."""
    a_s, b_s = _entity_salient(name_a), _entity_salient(name_b)
    if not a_s or not b_s:
        return False
    only_a, only_b = a_s - b_s, b_s - a_s
    if only_a and only_b:
        return False  # both sides unique — not a pure specialization (defer to embedding/LLM)
    extra = only_a or only_b
    if not extra:
        return False
    return not (extra <= _GENERIC_EXTRA)


def acronym_same(name_a: Any, name_b: Any) -> bool:
    """One name is the initialism of the other, modulo generic trailing words (FWC↔Full Well
    Capacity, S.S.↔Subthreshold Slope, FD Node↔Floating Diffusion). High precision: ≥2-letter
    initialisms rarely collide across distinct concepts."""
    wa = canonicalize_name(name_a).split()
    wb = canonicalize_name(name_b).split()
    if not wa or not wb:
        return False

    def _try(short_words: list[str], long_words: list[str]) -> bool:
        core = [w for w in short_words if w not in _GENERIC_EXTRA] or short_words
        if len(core) != 1:
            return False
        acro = _ACRO_SPLIT.sub("", core[0])
        if len(acro) < 2 or not acro.isalpha():
            return False
        long_core = [w for w in long_words if w not in _GENERIC_EXTRA] or long_words
        return "".join(w[0] for w in long_core if w).lower() == acro.lower()

    return _try(wa, wb) or _try(wb, wa)


def merge_tier(name_a: Any, name_b: Any, cosine: float | None = None,
               candidate_floor: float = 0.82) -> str:
    """Production merge decision for two concept names (+ optional embedding cosine).

    Returns 'MERGE' (auto, lexical identity), 'REJECT' (conflict / below floor), or 'VERIFY'
    (embedding candidate needing an LLM-verify decision). Auto-merge ONLY the MERGE tier; route
    VERIFY through the LLM verifier (and/or the review queue) — never auto-merge on cosine alone.
    """
    a, b = str(name_a), str(name_b)
    # hard precision guards first
    if qualifier_conflict(a, b) or numeric_conflict(a, b):
        return "REJECT"
    # lexical identity → safe auto-merge
    if canonicalize_name(a) == canonicalize_name(b):
        return "MERGE"
    if alias_surface_forms(a) and _alias_match(a, b):
        return "MERGE"
    if acronym_same(a, b):
        return "MERGE"
    # embedding candidate → LLM-verify, never auto on cosine
    if cosine is not None and cosine >= candidate_floor:
        return "VERIFY"
    return "REJECT"


def cluster_merge_edges(ids: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    """Union-find connected components over CONFIRMED merge edges → canonical-entity clusters.

    Pure (no embeddings): callers pass only edges they have already decided MERGE (or LLM-confirmed
    VERIFY). The chaining/cohesion guard against transitive over-merge lives in the caller, which
    holds the embeddings (see experiments/canon/discovery.py). Returns clusters of size ≥ 2."""
    index = {cid: i for i, cid in enumerate(ids)}
    parent = list(range(len(ids)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        if a in index and b in index:
            parent[find(index[a])] = find(index[b])
    comp: dict[int, list[str]] = {}
    for cid, i in index.items():
        comp.setdefault(find(i), []).append(cid)
    return [members for members in comp.values() if len(members) > 1]


# --- shadow-only merge proposal report --------------------------------------
_FIRING_RULES = ("alias", "generic_qualifier", "salient_containment")


def _matcher_normalized_forms(concept: Mapping[str, Any]) -> set[str]:
    """Current live Tier-0 normalized name/alias forms for shadow exclusion only."""
    from openclaw_brain.knowledge.reasoning.matcher import _normalize_name

    forms: list[Any] = [concept.get("name", "")]
    aliases = concept.get("aliases") or []
    if isinstance(aliases, str):
        forms.append(aliases)
    else:
        forms.extend(aliases)
    return {_normalize_name(str(form)) for form in forms if str(form or "").strip()}


def _already_live_name_match(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return bool(_matcher_normalized_forms(a) & _matcher_normalized_forms(b))


def _alias_firing_rule(name_a: Any, name_b: Any) -> str | None:
    """Classify registered-core matches without changing same_entity behavior."""
    salient_a = _entity_salient(name_a)
    salient_b = _entity_salient(name_b)
    cores_a = salient_a & _ENTITY_CORES
    cores_b = salient_b & _ENTITY_CORES
    if not cores_a or cores_a != cores_b:
        return None

    residue_a = salient_a - _ENTITY_CORES
    residue_b = salient_b - _ENTITY_CORES
    if (residue_a - _GENERIC_QUALIFIERS) or (residue_b - _GENERIC_QUALIFIERS):
        return None
    if residue_a or residue_b:
        return "generic_qualifier"
    return "alias"


def _firing_rule(name_a: Any, name_b: Any) -> str:
    return _alias_firing_rule(name_a, name_b) or "salient_containment"


def _blocking_tokens(name: Any) -> set[str]:
    """Blocking tokens covering both alias-aware and raw salient same_entity paths."""
    return _entity_salient(name) | set(_salient(name))


def _candidate_pairs(concepts: list[Mapping[str, Any]]) -> set[tuple[int, int]]:
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, concept in enumerate(concepts):
        name = concept.get("name", "")
        buckets[("canonical_key", canonical_key(name))].append(idx)
        for token in _blocking_tokens(name):
            buckets[("salient", token)].append(idx)

    pairs: set[tuple[int, int]] = set()
    for indexes in buckets.values():
        if len(indexes) < 2:
            continue
        ordered = sorted(set(indexes))
        for left_pos, left in enumerate(ordered):
            for right in ordered[left_pos + 1:]:
                pairs.add((left, right))
    return pairs


def _non_generic_residue(name: Any) -> set[str]:
    return _entity_salient(name) - _ENTITY_CORES - _GENERIC_QUALIFIERS


def _false_merge_watch_detail(name_a: Any, name_b: Any) -> dict[str, list[str]] | None:
    cores = alias_surface_forms(name_a) & alias_surface_forms(name_b)
    if not cores:
        return None
    residue_a = _non_generic_residue(name_a)
    residue_b = _non_generic_residue(name_b)
    if residue_a == residue_b:
        return None
    if not residue_a and not residue_b:
        return None
    return {
        "shared_cores": sorted(cores),
        "a_non_generic_residue": sorted(residue_a),
        "b_non_generic_residue": sorted(residue_b),
    }


def shadow_false_merge_watch(proposal: Mapping[str, Any]) -> dict[str, list[str]] | None:
    """Return false-merge watch metadata for a shadow proposal, or None if unflagged."""
    return _false_merge_watch_detail(proposal.get("a_name", ""), proposal.get("b_name", ""))


def propose_merges(concepts: list[Mapping]) -> list[dict]:
    """Shadow-only merge proposals canonicalization would add beyond live Tier-0 matching.

    The function is pure: it reads only the supplied concept records, never queries Neo4j,
    and never changes matcher decisions. Records need at least ``id`` and ``name``; optional
    ``aliases`` are used only to suppress proposals already covered by the current matcher's
    normalized-name/alias equality.
    """
    ordered = sorted(
        concepts,
        key=lambda c: (canonicalize_name(c.get("name", "")), str(c.get("id", ""))),
    )
    proposals: list[dict] = []
    for left, right in sorted(_candidate_pairs(ordered)):
        a = ordered[left]
        b = ordered[right]
        a_id = str(a.get("id", ""))
        b_id = str(b.get("id", ""))
        if a_id == b_id:
            continue
        a_name = str(a.get("name", ""))
        b_name = str(b.get("name", ""))
        if not same_entity(a_name, b_name):
            continue
        if _already_live_name_match(a, b):
            continue
        proposals.append(
            {
                "a_id": a_id,
                "a_name": a_name,
                "b_id": b_id,
                "b_name": b_name,
                "firing_rule": _firing_rule(a_name, b_name),
            }
        )

    return sorted(
        proposals,
        key=lambda p: (
            _FIRING_RULES.index(p["firing_rule"]),
            canonicalize_name(p["a_name"]),
            canonicalize_name(p["b_name"]),
            p["a_id"],
            p["b_id"],
        ),
    )
