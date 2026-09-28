"""Corpus growth automation (SPEC §2-§5, docs/superpowers/specs/2026-07-03-corpus-growth-automation-
design.md) — the D3 trust split's AUTO lane: recipes that reference an already-validated template
are authored, simulated, judged, and (on --apply) projected fully automatically, no human in the
loop. A NEW SPICE template is never auto-admitted (that stays a hand-validated, human code change to
templates.py) — an unmatched high-signal topology only ever SURFACES via `review_lane_pending`.

Three ordered sources feed one engine (spec §2):
  (a) coverage-gap enumeration over the registered templates' probe space (primary, immediately
      productive — multiplies claim density with zero new-template risk).
  (b) growth_queue.jsonl consumption — ingest-discovered CircuitTopology nodes that matched a
      registered class via the curated alias table (`_CLASS_ALIASES` / `match_topology_name`),
      carrying real source-chunk grounding.
  (c) TOPOLOGY_BACKLOG.md — curated curriculum rows mapped to a registered class via the same
      alias table (canonical-knowledge authoring; the backlog carries no source chunks).

Priority for the per-run `max_recipes` cap is queue > backlog > coverage: sources (b)/(c) represent
live signal (a topology the corpus actually needs right now) and should be authored first when the
budget is tight; (a) is the exhaustive fallback that eventually covers everything else. All three
draw from the SAME per-class uncovered-probe space (novelty-gated against existing VERIFIED-family
claim-cards), so a probe claimed by (b) is never re-offered by (c) or (a).

Two entry points mirror `agent.py::project_executable` / `executor.py::run_recipe`:
  plan_growth()    — pure planning (one Neo4j read per touched class + file I/O); no LLM, no sim.
  execute_growth() — author_recipe (LLM) -> run_recipe (engine-aware sim) -> verdict routing ->
                     corpus.store + project (apply=True only) -> journal.log per projected specimen.

Digital (iverilog) classes are excluded from AUTO authoring in v1 (spec §3 gate 6) — coverage
enumeration skips any TEMPLATES entry whose `engine` isn't "ngspice", every such class is recorded in
`GrowthPlan.skipped_digital`.
"""

from __future__ import annotations

import json
import logging
import re
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .engines import ENGINES, engine_for_template
from .executor import run_recipe
from .models import VerdictClass
from .recipe import author_recipe
from .templates import TEMPLATES

logger = logging.getLogger(__name__)

# The probe space is (metric, knob) PAIRS — a kind-agnostic cell (spec §2a revision). WHICH quant kind
# certifies a given (metric, knob) relationship (direction / elasticity / invariance / value / ...) is
# the AUTHOR's physical judgment to make (recipe.py::_FOCUS_INSTRUCTION), not the enumerator's mandate:
# a cell where the metric provably does NOT move with the knob is correctly certified as `invariance`,
# never forced into a `direction` claim the oracle would then FLAG. One certified VERIFIED-family
# claim-card (of ANY kind) on a (metric, knob) pair is enough to mark that cell covered.
_VERIFIED_FAMILY = {VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT, VerdictClass.VERIFIED_NEGATIVE}
_REFUTED_FAMILY = {VerdictClass.REFUTED, VerdictClass.REFUTED_MAGNITUDE}
_VERIFIED_FAMILY_STR = tuple(v.value for v in _VERIFIED_FAMILY)


# =====================================================================================================
# Topology-name matching (spec §2b/§3) — the alias table that decides AUTO-lane eligibility for both
# ingest-discovered CircuitTopology nodes (growth_queue) and TOPOLOGY_BACKLOG.md rows.
# =====================================================================================================


def _normalize_name(name: str) -> str:
    """Casefold + collapse underscores/hyphens to spaces + strip remaining punctuation. Both the
    alias-table keys and every name looked up through `match_topology_name` are normalized through
    this SAME function, so "5T OTA", "5t_ota", and "5-T OTA" all resolve identically."""
    name = (name or "").strip().casefold()
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


# Curated alias table (spec §2b): normalized name-variant -> the REGISTERED topology_class it refers
# to. Covers all 15 analog registered classes with (i) natural-language phrases a source PDF or LLM
# summary would use, and (ii) the matching TOPOLOGY_BACKLOG.md `topology_class` ids where that backlog
# row names the SAME circuit our template builds. Aliasing is a coarse "is this text plausibly about
# topology X" signal that only ROUTES which registered template gets grown/grounded — it is not a
# claim of exact circuit equivalence. Safety is enforced structurally downstream: enforce_executable()
# always snaps build.template_ref to the validated template regardless of what matched here, and the
# oracle only ever certifies what was actually simulated (the D3 trust split, spec §1). Deliberately
# NOT aliased: fully-differential/CMFB variants (e.g. backlog #71/#72 "+CMFB") and resistively-biased
# source-follower (#29) — those are materially different circuits from what our templates build, and a
# wrong alias there would misrepresent a backlog row as "growable" when it is not.
_CLASS_ALIASES: dict[str, str] = {
    # miller_ota_2stage_nmos_in
    "two stage miller ota": "miller_ota_2stage_nmos_in",
    "miller ota": "miller_ota_2stage_nmos_in",
    "two stage op amp": "miller_ota_2stage_nmos_in",
    "miller compensated op amp": "miller_ota_2stage_nmos_in",
    "two stage miller compensated op amp": "miller_ota_2stage_nmos_in",
    # current_mirror_simple_nmos
    "simple current mirror": "current_mirror_simple_nmos",
    "nmos current mirror": "current_mirror_simple_nmos",
    "current mirror": "current_mirror_simple_nmos",
    "current mirror simple": "current_mirror_simple_nmos",
    # common_source_active_load_nmos
    "common source active load": "common_source_active_load_nmos",
    "active load cs stage": "common_source_active_load_nmos",
    "common source amplifier": "common_source_active_load_nmos",
    "active loaded common source": "common_source_active_load_nmos",
    "common source current source load": "common_source_active_load_nmos",
    "cs current source load": "common_source_active_load_nmos",
    # common_gate_nmos
    "common gate": "common_gate_nmos",
    "cg stage": "common_gate_nmos",
    "common gate stage": "common_gate_nmos",
    "common gate amplifier": "common_gate_nmos",
    # source_follower_nmos
    "source follower": "source_follower_nmos",
    "voltage buffer": "source_follower_nmos",
    "common drain": "source_follower_nmos",
    "source follower current source": "source_follower_nmos",
    "pixel source follower": "source_follower_nmos",
    "ac coupled source follower": "source_follower_nmos",
    # diff_pair_resistive_nmos
    "differential pair": "diff_pair_resistive_nmos",
    "diff pair": "diff_pair_resistive_nmos",
    "basic differential pair": "diff_pair_resistive_nmos",
    "resistively loaded differential pair": "diff_pair_resistive_nmos",
    "differential pair resistive load": "diff_pair_resistive_nmos",
    # cascode_current_mirror_nmos
    "cascode current mirror": "cascode_current_mirror_nmos",
    "cascode mirror": "cascode_current_mirror_nmos",
    "nmos cascode mirror": "cascode_current_mirror_nmos",
    # ota_5t_nmos_in
    "5t ota": "ota_5t_nmos_in",
    "five transistor ota": "ota_5t_nmos_in",
    "5 transistor ota": "ota_5t_nmos_in",
    "single stage ota": "ota_5t_nmos_in",
    "active mirror differential pair": "ota_5t_nmos_in",
    # telescopic_cascode_ota_nmos_in (single-ended forward path only — see module note above)
    "telescopic cascode ota": "telescopic_cascode_ota_nmos_in",
    "telescopic ota": "telescopic_cascode_ota_nmos_in",
    "telescopic op amp": "telescopic_cascode_ota_nmos_in",
    "telescopic cascode diff pair": "telescopic_cascode_ota_nmos_in",
    "cascode amplifier": "telescopic_cascode_ota_nmos_in",
    # folded_cascode_ota_nmos_in (single-ended forward path only — see module note above)
    "folded cascode ota": "folded_cascode_ota_nmos_in",
    "folded cascode op amp": "folded_cascode_ota_nmos_in",
    "folded cascode operational amplifier": "folded_cascode_ota_nmos_in",
    "folded cascode stage": "folded_cascode_ota_nmos_in",
    "nmos input folded cascode op amp": "folded_cascode_ota_nmos_in",
    # regulated_cascode_nmos
    "regulated cascode": "regulated_cascode_nmos",
    "gain boosted cascode": "regulated_cascode_nmos",
    "regulated cascode source": "regulated_cascode_nmos",
    "gain boosting technique": "regulated_cascode_nmos",
    # comparator_continuous_nmos
    "comparator": "comparator_continuous_nmos",
    "continuous time comparator": "comparator_continuous_nmos",
    "analog comparator": "comparator_continuous_nmos",
    "column comparator": "comparator_continuous_nmos",
    "comparator continuous": "comparator_continuous_nmos",
    # cds_switched_cap_nmos
    "correlated double sampling": "cds_switched_cap_nmos",
    "cds": "cds_switched_cap_nmos",
    "cds amplifier": "cds_switched_cap_nmos",
    "column level cds amplifier": "cds_switched_cap_nmos",
    # single_slope_ramp_generator
    "ramp generator": "single_slope_ramp_generator",
    "global ramp reference": "single_slope_ramp_generator",
    "single slope ramp generator": "single_slope_ramp_generator",
    "ramp staircase generator": "single_slope_ramp_generator",
    # column_pga_inverting_nmos
    "programmable gain amplifier": "column_pga_inverting_nmos",
    "column pga": "column_pga_inverting_nmos",
    "column amplifier": "column_pga_inverting_nmos",
}


def match_topology_name(name: str) -> str | None:
    """Casefolded, punctuation/underscore-normalized lookup against the curated alias table, falling
    back to the registered template class names themselves (so an already-canonical class name always
    self-matches). Returns the registered topology_class, or None (no confident match — the D3 REVIEW
    lane, never guessed)."""
    if not name:
        return None
    key = _normalize_name(name)
    if key in _CLASS_ALIASES:
        return _CLASS_ALIASES[key]
    for cls in TEMPLATES:
        if _normalize_name(cls) == key:
            return cls
    return None


# =====================================================================================================
# TOPOLOGY_BACKLOG.md parsing (spec §2c/§3) — the curated curriculum table.
# =====================================================================================================


@dataclass
class BacklogRow:
    backlog_id: str            # the row's own topology_class id (backlog snake_case, e.g. "ptat_core")
    name: str                  # the row's human-readable name column (may carry "**[DONE]**")
    done: bool                 # True if the name column marks this row [DONE]
    mapped_class: str | None   # the registered topology_class this row maps to, or None (unmapped)


# One markdown table row: `| <#> | <rest of the row, pipe-separated> |`. Matches ONLY numbered data
# rows of the dedup table in TOPOLOGY_BACKLOG.md section (1) — the header/separator rows and the
# lettered gold-set table (G1..G8) don't match `\d+`, so they're skipped for free.
_BACKLOG_ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|(.+)\|\s*$")
_BACKLOG_CLASS_COL_RE = re.compile(r"^`([a-z0-9_]+)`$")


def parse_topology_backlog(path: str | Path) -> list[BacklogRow]:
    """Parse the section-(1) dedup table of TOPOLOGY_BACKLOG.md into BacklogRow entries. Returns []
    for a missing file (never raises — the backlog is an optional planning artifact)."""
    p = Path(path)
    if not p.is_file():
        return []
    rows: list[BacklogRow] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        m = _BACKLOG_ROW_RE.match(line.strip())
        if not m:
            continue
        cols = [c.strip() for c in m.group(2).split("|")]
        if len(cols) < 2:
            continue
        cm = _BACKLOG_CLASS_COL_RE.match(cols[0])
        if not cm:
            continue  # column 2 isn't a single `backtick_code` token -> not a topology_class data row
        backlog_id = cm.group(1)
        name_col = cols[1]
        done = "[done]" in name_col.lower()
        mapped = match_topology_name(backlog_id) or match_topology_name(name_col)
        rows.append(BacklogRow(backlog_id=backlog_id, name=name_col, done=done, mapped_class=mapped))
    return rows


def _default_backlog_path() -> Path:
    return (Path(__file__).resolve().parents[4] / "experiments"
            / "executable_circuit_specimens" / "TOPOLOGY_BACKLOG.md")


# =====================================================================================================
# growth_queue.jsonl / template_queue.jsonl — append-only JSONL (spec §3 queue records; the
# merge_queue.jsonl precedent). Writers never rewrite a line; status transitions are NEW lines.
# =====================================================================================================


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _append_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _pending_growth_queue_entries(state_dir: Path) -> list[dict[str, Any]]:
    """Latest-status-wins view of growth_queue.jsonl, filtered to status == "pending". Identity is
    `matched_from` (the source node id) — the SAME node's status transitions (pending -> consumed /
    skipped_covered) are later lines with the same `matched_from`, per the append-only convention."""
    records = _read_jsonl(state_dir / "growth_queue.jsonl")
    latest: dict[Any, dict[str, Any]] = {}
    for r in records:
        key = r.get("matched_from") or r.get("canonical_name")
        latest[key] = r
    return [r for r in latest.values() if r.get("status") == "pending"]


def _get(node: Any, attr: str, default: Any = None) -> Any:
    if isinstance(node, dict):
        return node.get(attr, default)
    return getattr(node, attr, default)


def _topology_node_view(node: Any) -> dict[str, Any]:
    """Normalize a NEW CircuitTopology node (a `NodeProposal`-shaped object OR an equivalent dict —
    G2's ingest hook may pass either) into the fields `enqueue_candidates` needs. Tolerant of both
    `NodeProposal.evidence_chunk_ids` and a flat `chunk_ids` key; `source_id` is read from
    `properties["source_id"]` first (where the pipeline injects it, `pipeline.py` stage 4.5) and a
    top-level `source_id` as a fallback."""
    props = _get(node, "properties", {}) or {}
    node_id = _get(node, "proposed_id") or _get(node, "topology_id") or _get(node, "node_id") or _get(node, "id") or ""
    canonical_name = _get(node, "canonical_name") or _get(node, "name") or ""
    chunk_ids = _get(node, "evidence_chunk_ids") or _get(node, "chunk_ids") or []
    source_id = (props.get("source_id") if isinstance(props, dict) else None) or _get(node, "source_id") or ""
    confidence = _get(node, "confidence", 0.0) or 0.0
    layer = _get(node, "knowledge_layer")
    if layer is None:
        layer = _get(node, "layer", -1)
    return {
        "node_id": node_id, "canonical_name": canonical_name, "source_id": source_id,
        "chunk_ids": list(chunk_ids), "confidence": float(confidence), "layer": int(layer),
    }


def enqueue_candidates(topology_nodes: Sequence[Any], state_dir: str | Path) -> dict[str, int]:
    """Match each NEW CircuitTopology node against the registered-template alias table and enqueue it
    for growth (AUTO lane, growth_queue.jsonl, carrying source-chunk grounding) or template drafting
    (REVIEW lane, template_queue.jsonl). Pure JSONL append — never touches Neo4j, never simulates.
    Safe to call with an empty sequence. Callers (the ingest post-commit hook) should treat any
    exception here as non-fatal per spec §4 ("failure is non-fatal and logged").

    Returns {"growth_queue": N, "template_queue": M, "skipped": K}.
    """
    state_dir = Path(state_dir)
    ts = _now_iso()
    growth_records: list[dict[str, Any]] = []
    template_records: list[dict[str, Any]] = []
    skipped = 0
    for node in topology_nodes:
        view = _topology_node_view(node)
        matched = match_topology_name(view["canonical_name"])
        if matched:
            growth_records.append({
                "ts": ts, "topology_class": matched, "matched_from": view["node_id"],
                "canonical_name": view["canonical_name"], "source_id": view["source_id"],
                "chunk_ids": view["chunk_ids"], "confidence": view["confidence"],
                "status": "pending",
            })
        # Spec §2b gate is "node confidence >= 0.7" in RAW LLM terms — but the pipeline's
        # reconcile stage scales unmatched new-node confidence x0.78 BEFORE the hook sees it
        # (pipeline.py stage 4.5, the M5 spread fix), so the equivalent post-scale threshold
        # is 0.7 * 0.78 ≈ 0.55. Gating at 0.7 post-scale would demand raw ~0.90 and starve
        # the template queue.
        elif view["confidence"] >= 0.55 and view["layer"] in (2, 3):
            template_records.append({
                "ts": ts, "canonical_name": view["canonical_name"], "node_id": view["node_id"],
                "layer": view["layer"], "confidence": view["confidence"],
                "source_id": view["source_id"], "chunk_ids": view["chunk_ids"], "llm_proposal": None,
            })
        else:
            skipped += 1
    _append_jsonl(state_dir / "growth_queue.jsonl", growth_records)
    _append_jsonl(state_dir / "template_queue.jsonl", template_records)
    return {"growth_queue": len(growth_records), "template_queue": len(template_records), "skipped": skipped}


# =====================================================================================================
# Coverage-gap enumeration (spec §2a) — the probe space over a class's registered metrics/knobs, minus
# probes already VERIFIED-family certified on the live graph.
# =====================================================================================================


async def _existing_verified_probes(store, topology_class: str) -> set[tuple[str, str]]:
    """(metric, knob) pairs already VERIFIED-family certified for this class, REGARDLESS of which
    quant `kind` certified them (kind is no longer part of the coverage key — spec revision: a cell
    certified as `invariance` is just as covered as one certified `direction`). ClaimCard nodes carry
    knob/metric/verdict (projection.py); topology_class lives on the owning Specimen, so the novelty
    check joins Specimen-[:HAS_CLAIM]->ClaimCard."""
    rows = await store.run_read_query(
        "MATCH (s:Specimen {topology_class: $tclass})-[:HAS_CLAIM]->(c:ClaimCard) "
        "WHERE c.verdict IN $verified "
        "RETURN DISTINCT c.metric AS metric, c.knob AS knob",
        {"tclass": topology_class, "verified": list(_VERIFIED_FAMILY_STR)},
    )
    return {(r["metric"], r["knob"]) for r in rows if r.get("metric") and r.get("knob")}


def _uncovered_probes(
    metrics: set[str], knobs: set[str], covered: set[tuple[str, str]],
) -> list[tuple[str, str]]:
    """(metric, knob) pairs not already covered, in deterministic order: metric (sorted) -> knob
    (sorted). Kind-agnostic: WHICH quant kind certifies the pair is the author's judgment, not this
    enumerator's mandate (spec revision, DEFECT 3)."""
    out: list[tuple[str, str]] = []
    for metric in sorted(metrics):
        for knob in sorted(knobs):
            if (metric, knob) not in covered:
                out.append((metric, knob))
    return out


# =====================================================================================================
# Planning (spec §3) — plan_growth()
# =====================================================================================================


@dataclass
class AuthoringTarget:
    """One probe to author + simulate: one metric x one knob on one topology_class. `kind_hint` is NOT
    an enumeration mandate — it is None for coverage/queue/backlog targets ("author chooses"): which
    quant kind (direction/elasticity/invariance/value/...) certifies the (metric, knob) relationship is
    the author's physical judgment (spec revision, DEFECT 3), not a planning-time label."""

    topology_class: str
    metric: str
    knob: str
    source: str                                         # "coverage" | "queue" | "backlog"
    kind_hint: str | None = None                        # None = author chooses the quant kind
    grounding_chunk_ids: list[str] = field(default_factory=list)
    enqueue_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class GrowthPlan:
    targets: list[AuthoringTarget] = field(default_factory=list)
    skipped_covered: int = 0                            # novelty-gate hits (already VERIFIED-family)
    review_lane_pending: dict[str, int] = field(
        default_factory=lambda: {"template_queue": 0, "backlog_unmapped": 0}
    )
    per_class_counts: dict[str, int] = field(default_factory=dict)   # topology_class -> planned count
    skipped_digital: list[str] = field(default_factory=list)         # engine != ngspice, gate 6
    sources: tuple[str, ...] = ()
    # topology_class -> the growth_queue.jsonl entries eligible to be marked "consumed" — i.e.
    # this class had >=1 queue-sourced AuthoringTarget that survived the max_recipes cap into
    # `targets`. NOT yet written as "consumed": that write is deferred to execute_growth, gated on
    # whether a queue-sourced target for this class actually got projected (author/simulate/
    # verdict/apply all succeeded) — see plan_growth's docstring and execute_growth's tail.
    queue_consumed_candidates: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


_ALL_SOURCES = ("coverage", "queue", "backlog")


async def plan_growth(
    store,
    *,
    sources: Sequence[str],
    max_recipes: int = 5,
    max_new_cards_per_class: int = 4,
    topology_class: str | None = None,
    state_dir: str | Path | None = None,
    backlog_path: str | Path | None = None,
    apply: bool = False,
) -> GrowthPlan:
    """Assemble an ordered, capped, novelty-gated list of AuthoringTargets (spec §3).

    `state_dir` gates growth_queue.jsonl (source "queue" is a no-op without it); `backlog_path`
    defaults to the repo's TOPOLOGY_BACKLOG.md. `review_lane_pending` is always populated (regardless
    of which `sources` were requested) so the review-lane backlog size never rots invisibly — it costs
    one extra file read, no LLM/sim spend. Digital (iverilog) classes never reach the LLM (gate 6):
    they're recorded in `skipped_digital`, not authored.

    `apply` gates growth_queue.jsonl STATUS WRITES only (spec §3 gate 5: dry-run writes nothing) — it
    never affects which targets are planned, only whether "skipped_covered" lines are appended here
    (see below) and, later, whether execute_growth appends "consumed" lines. Status writes are also
    deferred until AFTER the `max_recipes` truncation below: a queue entry is only a CANDIDATE for
    "consumed" if at least one of its targets survived into the final, capped target list (returned
    as `GrowthPlan.queue_consumed_candidates`, keyed by topology_class). An entry whose targets were
    entirely cut by the global cap is left "pending" so it is re-offered on the next run instead of
    being silently lost.

    IMPORTANT: this function never writes "consumed" itself — planning alone (surviving the cap)
    does not mean the target was ever actually authored/simulated/certified. That write is deferred
    to execute_growth, which only marks a candidate class "consumed" once one of its queue-sourced
    targets actually gets projected; any author/simulate/engine/verdict/apply failure leaves the
    entry "pending" (re-offered next run) instead of permanently losing its ingest-sourced grounding.
    """
    if max_recipes <= 0:
        raise ValueError("max_recipes must be > 0 (no cap = no run)")
    sources = set(sources)
    unknown = sources - set(_ALL_SOURCES)
    if unknown:
        raise ValueError(f"unknown growth source(s): {sorted(unknown)} (have {_ALL_SOURCES})")

    skipped_digital: list[str] = []
    eligible_classes: list[str] = []
    for tclass, entry in TEMPLATES.items():
        if topology_class and tclass != topology_class:
            continue
        if entry.get("engine", "ngspice") != "ngspice":
            skipped_digital.append(tclass)
            continue
        eligible_classes.append(tclass)

    # Precompute each eligible class's FULL uncovered-probe space once; every source draws from (and
    # depletes) the same working list, so a probe claimed by queue is never re-offered by backlog/
    # coverage. skipped_covered is the one-time novelty-gate count (total probe space - uncovered).
    remaining_probes: dict[str, list[tuple[str, str]]] = {}
    skipped_covered = 0
    for tclass in eligible_classes:
        entry = TEMPLATES[tclass]
        metrics, knobs = set(entry.get("metrics", [])), set(entry.get("knobs", []))
        existing = await _existing_verified_probes(store, tclass)
        uncovered = _uncovered_probes(metrics, knobs, existing)
        total = len(metrics) * len(knobs)
        skipped_covered += total - len(uncovered)
        remaining_probes[tclass] = uncovered

    def _take(tclass: str, n: int) -> list[tuple[str, str]]:
        if n <= 0:
            return []
        bucket = remaining_probes.get(tclass, [])
        taken, rest = bucket[:n], bucket[n:]
        remaining_probes[tclass] = rest
        return taken

    per_class_selected: dict[str, int] = defaultdict(int)
    ordered_targets: list[AuthoringTarget] = []

    # review_lane_pending is always computed (independent of `sources`) — see docstring.
    review_lane_pending = {"template_queue": 0, "backlog_unmapped": 0}
    if state_dir is not None:
        review_lane_pending["template_queue"] = len(_read_jsonl(Path(state_dir) / "template_queue.jsonl"))
    bpath = Path(backlog_path) if backlog_path else _default_backlog_path()
    backlog_rows = parse_topology_backlog(bpath)
    review_lane_pending["backlog_unmapped"] = sum(
        1 for r in backlog_rows if not r.done and r.mapped_class is None
    )

    # (b) queue consumption — priority 1 (live ingest signal), state_dir required. Status lines are
    # NOT written here — target selection must stay side-effect-free w.r.t. `apply` (spec §3 gate 5).
    # `queue_consumed_groups` becomes the returned `GrowthPlan.queue_consumed_candidates` (a
    # CANDIDATE list — execute_growth writes "consumed" only once it confirms actual success);
    # `queue_skipped_entries` is resolved into an immediate "skipped_covered" file write further
    # down (planning-time-only fact, independent of execution outcome), once the final
    # `max_recipes`-capped target list is known.
    queue_consumed_groups: dict[str, list[dict[str, Any]]] = {}
    queue_skipped_entries: list[dict[str, Any]] = []
    if "queue" in sources and state_dir is not None:
        pending = _pending_growth_queue_entries(Path(state_dir))
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in pending:
            tclass = entry.get("topology_class")
            if tclass in eligible_classes:
                grouped[tclass].append(entry)
        for tclass, entries in grouped.items():
            budget = max_new_cards_per_class - per_class_selected[tclass]
            take = _take(tclass, budget)
            if take:
                chunk_ids: list[str] = []
                for e in entries:
                    chunk_ids.extend(e.get("chunk_ids") or [])
                chunk_ids = list(dict.fromkeys(chunk_ids))
                for metric, knob in take:
                    ordered_targets.append(AuthoringTarget(
                        topology_class=tclass, metric=metric, knob=knob, kind_hint=None,
                        source="queue", grounding_chunk_ids=chunk_ids,
                        enqueue_meta={"matched_from": [e.get("matched_from") for e in entries]},
                    ))
                    per_class_selected[tclass] += 1
                queue_consumed_groups[tclass] = entries
            else:
                queue_skipped_entries.extend(entries)

    # (c) backlog — priority 2 (curated curriculum signal), no grounding.
    if "backlog" in sources:
        for row in backlog_rows:
            if row.done or row.mapped_class is None:
                continue
            tclass = row.mapped_class
            if tclass not in eligible_classes:
                continue
            budget = max_new_cards_per_class - per_class_selected[tclass]
            for metric, knob in _take(tclass, budget):
                ordered_targets.append(AuthoringTarget(
                    topology_class=tclass, metric=metric, knob=knob, kind_hint=None,
                    source="backlog", grounding_chunk_ids=[],
                    enqueue_meta={"backlog_id": row.backlog_id, "backlog_name": row.name},
                ))
                per_class_selected[tclass] += 1

    # (a) coverage — fallback, fills remaining per-class budget for every eligible class.
    if "coverage" in sources:
        for tclass in eligible_classes:
            budget = max_new_cards_per_class - per_class_selected[tclass]
            for metric, knob in _take(tclass, budget):
                ordered_targets.append(AuthoringTarget(
                    topology_class=tclass, metric=metric, knob=knob, kind_hint=None,
                    source="coverage", grounding_chunk_ids=[],
                ))
                per_class_selected[tclass] += 1

    final_targets = ordered_targets[:max_recipes]
    per_class_counts: dict[str, int] = defaultdict(int)
    for t in final_targets:
        per_class_counts[t.topology_class] += 1

    # A queue-sourced class is only a CANDIDATE for "consumed" if at least one of its targets
    # survived the max_recipes cap into `final_targets`; classes entirely cut by the cap are left
    # out of the candidate map entirely, so they stay "pending" (re-offered next run) — this part
    # is unchanged from before. What HAS changed (W-D2 defect 7): this function no longer writes
    # "consumed" itself — surviving the cap is a planning-time fact, not proof that anything was
    # ever actually certified. The candidate map is returned on the plan for execute_growth to
    # resolve into an actual "consumed" write, gated on real per-class projection success.
    surviving_queue_classes = {t.topology_class for t in final_targets if t.source == "queue"}
    queue_consumed_candidates = {
        tclass: entries for tclass, entries in queue_consumed_groups.items()
        if tclass in surviving_queue_classes
    }

    # "skipped_covered" IS still written here, immediately: it's a pure planning-time fact (this
    # class's uncovered-probe space was already fully novelty-gated, independent of whether
    # anything downstream ever executes) — resolved NOW that the final target list is known, and
    # ONLY when apply=True (spec §3 gate 5 — dry-run writes nothing).
    if apply and state_dir is not None and queue_skipped_entries:
        status_records: list[dict[str, Any]] = []
        for e in queue_skipped_entries:
            rec = dict(e)
            rec["ts"] = _now_iso()
            rec["status"] = "skipped_covered"
            status_records.append(rec)
        try:
            _append_jsonl(Path(state_dir) / "growth_queue.jsonl", status_records)
        except Exception as exc:
            logger.warning("growth_queue.jsonl status append failed: %s", exc)

    return GrowthPlan(
        targets=final_targets, skipped_covered=skipped_covered,
        review_lane_pending=review_lane_pending, per_class_counts=dict(per_class_counts),
        skipped_digital=skipped_digital, sources=tuple(sorted(sources)),
        queue_consumed_candidates=queue_consumed_candidates,
    )


# =====================================================================================================
# Execution (spec §3) — execute_growth()
# =====================================================================================================


@dataclass
class GrowthDeps:
    """Everything execute_growth needs, bundled so the caller (BrainAgent.grow_executable / the CLI)
    constructs it once. `corpus`/`projector` are None in a dry run (or when the caller chooses not to
    wire persistence) — execute_growth degrades to routing-only (no write) in that case."""

    store: Any                                          # GraphStore-like: run_read_query(query, params)
    model_chain: Sequence[Any]                          # author_recipe's model_chain (extraction chain)
    resilience_config: Any
    auth_refresh: Any = None
    vault: Any | None = None                             # EvidenceVault-like: get_text(hash) -> str|None
    corpus: Any | None = None                            # SpecimenCorpus | None
    projector: Any | None = None                         # GraphProjector-like: await .project(specimen)
    journal: Any | None = None                            # ActionJournal-like: .log(op, **kwargs)
    state_dir: str | Path | None = None                   # growth_log.jsonl append target
    config: Any | None = None                             # active deployment policy for simulator


@dataclass
class GrowthReport:
    planned: int
    authored: int
    simulated: int
    projected: int
    skipped_covered: int
    triage: list[dict[str, Any]]
    review_lane_pending: dict[str, int]
    per_class_counts: dict[str, int]
    duration_s: float
    apply: bool = False
    # Per-card preview of every projectable result — the spec §8 step-2 owner-review surface:
    # a dry run must show WHAT would be certified, not just how many.
    certified: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def _fetch_grounding_text(store, vault, chunk_ids: Sequence[str], *, max_chunks: int = 3) -> str:
    """Top-k source-chunk text for a target's grounding_chunk_ids, via SourceChunk.raw_text_hash ->
    EvidenceVault (spec §3). Empty string (no source_text) means canonical-knowledge authoring —
    author_recipe's documented fallback when no grounding is available. Never raises."""
    if not chunk_ids or vault is None or store is None:
        return ""
    ids = list(dict.fromkeys(chunk_ids))[:max_chunks]
    try:
        rows = await store.run_read_query(
            "MATCH (sc:SourceChunk) WHERE sc.chunk_id IN $ids "
            "RETURN sc.chunk_id AS chunk_id, sc.raw_text_hash AS raw_text_hash",
            {"ids": ids},
        )
    except Exception as exc:
        logger.warning("grounding chunk lookup failed: %s", exc)
        return ""
    texts: list[str] = []
    for row in rows:
        h = row.get("raw_text_hash")
        if not h:
            continue
        try:
            text = vault.get_text(h)
        except Exception as exc:
            logger.warning("EvidenceVault read failed for %s: %s", h, exc)
            text = None
        if text:
            texts.append(text)
    return "\n\n".join(texts)


def _compact_trace(exc: BaseException) -> str:
    """Last 3 frames of exc's traceback (compact string) — so a live author/simulate-stage failure is
    diagnosable straight from the report/growth_log without a rerun."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-3:])


def _triage_entry(target: AuthoringTarget, stage: str, **extra: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "topology_class": target.topology_class, "metric": target.metric, "knob": target.knob,
        "kind_hint": target.kind_hint, "source": target.source, "stage": stage,
    }
    recipe = extra.pop("recipe", None)
    if recipe is not None:
        try:
            entry["recipe"] = recipe.model_dump(mode="json")
        except Exception:
            entry["recipe"] = str(recipe)
    entry.update({k: v for k, v in extra.items() if v is not None})
    return entry


def _append_growth_log(state_dir: str | Path | None, report: GrowthReport) -> None:
    """Best-effort provenance append (spec §3) — never raises, mirrors journal.py's discipline."""
    if state_dir is None:
        return
    try:
        path = Path(state_dir) / "growth_log.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": _now_iso(), **report.to_dict()}, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        logger.warning("growth_log.jsonl append failed: %s", exc)


async def execute_growth(agent_deps: GrowthDeps, plan: GrowthPlan, *, apply: bool) -> GrowthReport:
    """Author -> simulate -> route -> (apply=True) project + persist, for every target in the plan
    (spec §3 pseudocode). `projected` counts the routing DECISION (VERIFIED/REFUTED-family got at
    least one certifiable claim) regardless of apply — a dry run still previews what would be written,
    exactly like `project_executable`'s dry-run does; actual corpus/graph writes only happen when
    apply=True AND `agent_deps.corpus`/`agent_deps.projector` are provided. FLAGGED/REJECTED verdicts
    (and author/engine/sim failures) never project; they land in `report.triage` for post-mortem.

    W-D2 defect 7: this is also where `plan.queue_consumed_candidates` gets resolved into an actual
    growth_queue.jsonl "consumed" write (moved here from plan_growth) — ONLY for classes that had a
    queue-sourced target actually get projected in THIS run. A class whose queue-sourced target(s)
    failed to author/simulate/engine, or got a FLAGGED/REJECTED (non-projectable) verdict, or whose
    apply-persistence raised, is left untouched (still "pending" from whenever it was enqueued) so it
    is re-offered on the next plan_growth run instead of permanently losing its ingest-sourced
    grounding for a target that was never actually certified from it. Gated on apply=True + state_dir,
    same as every other growth_queue.jsonl write (spec §3 gate 5: dry-run writes nothing)."""
    t0 = time.monotonic()
    authored = simulated = projected = 0
    triage: list[dict[str, Any]] = []
    certified: list[dict[str, Any]] = []
    per_class_projected: dict[str, int] = defaultdict(int)
    # Classes with >=1 queue-sourced target that actually got projected THIS run (source-specific,
    # mirroring plan_growth's own "source == queue" specificity — a coverage/backlog-sourced target
    # for the same class projecting must NOT falsely mark an unrelated queue entry "consumed").
    queue_projected_classes: set[str] = set()

    for target in plan.targets:
        cap_entry = TEMPLATES.get(target.topology_class)
        if cap_entry is None or cap_entry.get("engine", "ngspice") != "ngspice":
            triage.append(_triage_entry(target, "capability", reason="unregistered or non-ngspice class"))
            continue

        source_text = ""
        if target.grounding_chunk_ids:
            source_text = await _fetch_grounding_text(agent_deps.store, agent_deps.vault, target.grounding_chunk_ids)

        try:
            recipe = await author_recipe(
                topology_class=target.topology_class, source_text=source_text,
                model_chain=agent_deps.model_chain, resilience_config=agent_deps.resilience_config,
                auth_refresh=agent_deps.auth_refresh,
                focus={"metric": target.metric, "knob": target.knob, "kind_hint": target.kind_hint},
            )
        except Exception as exc:
            triage.append(_triage_entry(target, "author", error=repr(exc), trace=_compact_trace(exc)))
            continue
        authored += 1

        template_ref = (recipe.build or {}).get("template_ref") or cap_entry["template_ref"]
        engine = (recipe.build or {}).get("engine") or engine_for_template(template_ref)
        espec = ENGINES.get(engine)
        from .engines import runner_for_engine
        runner = runner_for_engine(engine, agent_deps.config) if espec is not None and espec.runner is not None else None
        if runner is None or not runner.available():
            triage.append(_triage_entry(target, "engine", recipe=recipe,
                                        reason=f"engine {engine!r} unavailable"))
            continue

        # corpus persistence is GATED by verdict routing (spec §3: "FLAGGED/REJECTED ... do NOT
        # project" groups corpus.store WITH graph projection) — never let run_recipe auto-store here;
        # a not-yet-routed specimen would otherwise land in the corpus regardless of verdict.
        try:
            result = run_recipe(recipe, runner, corpus=None, config=agent_deps.config)
        except Exception as exc:
            triage.append(_triage_entry(target, "simulate", recipe=recipe, error=repr(exc),
                                        trace=_compact_trace(exc)))
            continue
        simulated += 1

        verdicts = {c.id: c.verdict for c in result.claim_cards}
        projectable = any(v in _VERIFIED_FAMILY or v in _REFUTED_FAMILY for v in verdicts.values() if v is not None)
        verdict_summary = {cid: (v.value if v else None) for cid, v in verdicts.items()}
        if not projectable:
            triage.append(_triage_entry(target, "verdict", recipe=recipe, verdicts=verdict_summary))
            continue

        projected += 1
        per_class_projected[target.topology_class] += 1
        certified.append({
            "topology_class": target.topology_class, "source": target.source,
            "cards": [{
                "id": c.id, "metric": c.mechanism.metric, "knob": c.mechanism.knob,
                "kind": c.mechanism.quant.kind if c.mechanism.quant else None,
                "verdict": (c.verdict.value if c.verdict else None),
            } for c in result.claim_cards],
        })
        if apply:
            # Per-target containment (mirrors the author/simulate guards above): corpus.store does
            # raw unguarded filesystem I/O and projector.project can raise on a Neo4j hiccup, so one
            # bad target's apply-persistence must not abort the rest of the plan — it is triaged
            # (never silently swallowed) and the loop continues to the next target. `corpus_stored`
            # is recorded honestly: if corpus.store already succeeded before projector.project (or
            # journal.log) raised, the specimen may now be orphaned in the git-corpus SSOT with no
            # graph presence — that is a known, documented risk (EPISTEMOLOGY.md Known issues #2),
            # not something this guard can retroactively undo, so it is surfaced rather than hidden.
            corpus_stored = False
            try:
                if agent_deps.corpus is not None:
                    agent_deps.corpus.store(result.specimen)   # sets result.specimen.spec_id in place
                    corpus_stored = True
                projection = None
                if agent_deps.projector is not None:
                    projection = await agent_deps.projector.project(result.specimen)
                if agent_deps.journal is not None:
                    agent_deps.journal.log(
                        "grow_executable", topology_class=target.topology_class, source=target.source,
                        spec_id=result.specimen.spec_id, verdicts=verdict_summary, **(projection or {}),
                    )
                # Apply-persistence succeeded end-to-end for this target — only NOW is it safe to
                # treat a queue-sourced target as having actually delivered on its ingest-sourced
                # grounding (W-D2 defect 7: a verdict alone, or a persistence failure two lines
                # above, must not silently mark the source queue entry "consumed").
                if target.source == "queue":
                    queue_projected_classes.add(target.topology_class)
            except Exception as exc:
                triage.append(_triage_entry(
                    target, "apply", recipe=recipe, error=repr(exc), trace=_compact_trace(exc),
                    corpus_stored=corpus_stored,
                    spec_id=result.specimen.spec_id if corpus_stored else None,
                ))

    # Resolve plan.queue_consumed_candidates into an actual growth_queue.jsonl "consumed" write —
    # deferred here from plan_growth (W-D2 defect 7) so it reflects actual execution outcome, not
    # merely having survived the max_recipes cap. Same gate-5 discipline as plan_growth's own
    # "skipped_covered" write: apply=True + state_dir required, dry runs write nothing.
    if apply and agent_deps.state_dir is not None and plan.queue_consumed_candidates:
        status_records: list[dict[str, Any]] = []
        for tclass, entries in plan.queue_consumed_candidates.items():
            if tclass not in queue_projected_classes:
                continue   # author/simulate/engine/verdict/apply failed -> stays "pending", re-offered
            for e in entries:
                rec = dict(e)
                rec["ts"] = _now_iso()
                rec["status"] = "consumed"
                status_records.append(rec)
        if status_records:
            try:
                _append_jsonl(Path(agent_deps.state_dir) / "growth_queue.jsonl", status_records)
            except Exception as exc:
                logger.warning("growth_queue.jsonl status append failed: %s", exc)

    duration_s = time.monotonic() - t0
    report = GrowthReport(
        planned=len(plan.targets), authored=authored, simulated=simulated, projected=projected,
        skipped_covered=plan.skipped_covered, triage=triage, certified=certified,
        review_lane_pending=dict(plan.review_lane_pending), per_class_counts=dict(per_class_projected),
        duration_s=duration_s, apply=apply,
    )
    _append_growth_log(agent_deps.state_dir, report)
    return report
