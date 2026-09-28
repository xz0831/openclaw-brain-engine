"""Law-tier Regularity projector (docs/superpowers/specs/2026-07-04-law-tier-graph-representation.md
§3/§4) — turns the E1/E1b cross-PDK replication runner's raw JSONL into `Regularity` nodes.

Additive, idempotent MERGE by `law_id`. The graph never invents a law: every field on a
`LawRecord` traces either to a RunRecord row in the JSONL, or to the static claim-card
catalog (seeds.py) that defined the recipe the JSONL ran (R6 — Neo4j stays a derived view).

Design notes (read before editing)
-----------------------------------
- **The raw JSONL is RunRecord rows, not the replication script's ClaimRow/agreement table.**
  `scripts/replicate_cross_pdk.py` writes two artifacts per suite: `<basename>_raw.jsonl` (one
  JSON object per `(topology_class, pdk)` RunRecord: `ok`, `verdicts` {claim_id: verdict},
  `verdict_notes` {claim_id: note}, `canonical` {route_key: [(x, y), ...]}) and
  `<basename>_table.md` (the human-readable ClaimRow agreement table — markdown only, never
  serialized to JSON). So `agree`/`divergence_note` do not exist as data anywhere on disk; this
  module RE-DERIVES per-claim agreement from the RunRecord rows by applying spec §4's
  deterministic status rules (`classify_status`) to each PDK's oracle verdict. This is NOT
  `replicate_cross_pdk.py::classify_claim`'s baseline-vs-others table logic — that shape serves
  the markdown report; the law tier needs the spec's own agreeing/disagreeing partition (and
  `scripts/` is not a packaged dependency of the read-only company wheel anyway). This is
  deriving-from-the-JSONL (the producer's actual output), not the
  "recompute agreement at query time from graph state" anti-pattern the spec's alternative D
  rejected — the JSONL stays the one source of truth for each PDK's raw verdict.
- **`knob` / `metric` / `quant.kind` are NOT in the RunRecord JSONL** — only `claim_id` and its
  verdict/note are. Those three fields are static properties of the claim-card DEFINITION, not of
  any one run, so they are resolved from the seed-recipe catalog (`seeds.seed_recipes()` +
  `seeds.statistical_seed_recipes()` — the exact recipe universe `replicate_cross_pdk.py` draws
  from) by `claim_id`. This mirrors `classify_claim`'s own call site, which reads
  `card.mechanism.quant.kind` off `recipe.claim_cards`, never off a raw JSONL row either. A
  claim_id absent from the catalog is an honest, reported gap — skipped, never invented.
- **`statement` never carries a magnitude** (spec §3) — not even a pre-registered band/bound. Only
  qualitative shape language ("increases with", "is invariant to", "power-law scaling in") appears
  in `statement`; every number (per-PDK verdict, per-PDK fitted exponent) lives in
  `member_summary` only.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..graph.schema import NodeLabel, RelType
from .models import QuantTest
from .oracle import _loglog
from .seeds import seed_recipes, statistical_seed_recipes

logger = logging.getLogger(__name__)

# "Agrees with the claim" (spec §4) = the verdict AFFIRMS the claim. VERIFIED_NEGATIVE is a real,
# correct measurement of the OPPOSITE outcome (models.py: "e.g. spec FAIL") — trustworthy, but it
# does not affirm: mixed with VERIFIED members it is a process divergence (the outcome differs by
# foundry), and unanimous VERIFIED_NEGATIVE is a replicated negative ("does NOT hold"), never a
# positively-phrased law. Both routes fall out of classify_status by keeping it OUT of this set.
# (growth.py's same-named set gates projection-WORTHINESS — a different question — and rightly
# includes VERIFIED_NEGATIVE there.)
_AFFIRMING_FAMILY = {"VERIFIED", "VERIFIED_WITH_CAVEAT"}

Resolver = Callable[[NodeLabel, str], Awaitable[str | None]]


# ── Static claim-card catalog (topology/metric/knob/kind — never per-run data) ──


@dataclass(frozen=True)
class ClaimShape:
    topology_class: str
    knob: str
    metric: str
    quant: QuantTest


def _build_catalog() -> dict[str, ClaimShape]:
    """claim_id -> ClaimShape, from the SAME recipe universe replicate_cross_pdk.py draws from
    (seed_recipes() for the nominal/E1 suite, statistical_seed_recipes() for E1b). A claim_id
    defined twice (should not happen for a validated seed) keeps the first definition and logs a
    warning rather than silently overwriting."""
    catalog: dict[str, ClaimShape] = {}
    for recipe in seed_recipes() + statistical_seed_recipes():
        for card in recipe.claim_cards:
            if card.id in catalog and catalog[card.id].topology_class != recipe.topology_class:
                logger.warning("laws: claim_id %r redefined under a different topology_class "
                               "(%r vs %r) — keeping the first", card.id,
                               catalog[card.id].topology_class, recipe.topology_class)
                continue
            catalog[card.id] = ClaimShape(
                topology_class=recipe.topology_class, knob=card.mechanism.knob,
                metric=card.mechanism.metric, quant=card.mechanism.quant,
            )
    return catalog


def _route_key(knob: str, metric: str) -> str:
    """Mirrors replicate_cross_pdk.py's `_route_key` exactly (same 1-line convention, re-derived
    here for the same reason `classify_claim`'s logic is re-derived — see module docstring)."""
    return f"{knob.lower()}_{metric}"


def _pelgrom_exponent(series: list | None) -> float | None:
    """The fitted global log-log slope — reuses oracle._loglog (the SAME math the oracle judged
    the elasticity claim against, imported not re-derived: oracle.py IS packaged). None for a
    missing/degenerate (<2-point) series."""
    if not series or len(series) < 2:
        return None
    g, _local = _loglog(series)
    return g


# ── Raw JSONL parsing (RunRecord rows) ──


@dataclass
class RunRow:
    topology_class: str
    pdk: str
    ok: bool
    verdicts: dict[str, str] = field(default_factory=dict)
    verdict_notes: dict[str, str] = field(default_factory=dict)
    canonical: dict[str, list] = field(default_factory=dict)
    source_path: str = ""
    source_line: int = 0


@dataclass
class LoadResult:
    rows: list[RunRow]
    errors: list[dict[str, Any]]   # {"path", "line", "error"} — malformed/incomplete rows, reported not raised


def load_run_rows(jsonl_path: str | Path) -> LoadResult:
    """Parse one replication raw JSONL (a RunRecord per line). A malformed line (bad JSON, or
    missing one of the required keys) is skipped and reported — never raises, mirrors
    `run_one`'s own PORT-FAILED discipline (one bad row must not silence the rest of the file)."""
    path = Path(jsonl_path)
    rows: list[RunRow] = []
    errors: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append({"path": str(path), "line": lineno, "error": f"malformed JSON: {exc}"})
            continue
        if not isinstance(obj, dict):
            errors.append({"path": str(path), "line": lineno, "error": "row is not a JSON object"})
            continue
        missing = [k for k in ("topology_class", "pdk", "ok") if k not in obj]
        if missing:
            errors.append({"path": str(path), "line": lineno, "error": f"missing keys: {missing}"})
            continue
        rows.append(RunRow(
            topology_class=obj["topology_class"], pdk=obj["pdk"], ok=bool(obj["ok"]),
            verdicts=obj.get("verdicts") or {}, verdict_notes=obj.get("verdict_notes") or {},
            canonical=obj.get("canonical") or {}, source_path=str(path), source_line=lineno,
        ))
    return LoadResult(rows=rows, errors=errors)


# ── Per-claim agreement (re-derived from RunRows, per module docstring) ──


@dataclass
class ClaimAgreement:
    claim_id: str
    topology_class: str
    verdicts_by_pdk: dict[str, str]          # only PDKs that actually produced a verdict
    notes_by_pdk: dict[str, str]
    fitted_exponent_by_pdk: dict[str, float | None] | None   # elasticity claims only
    source_paths: set[str]


def _group_by_claim(rows: list[RunRow], catalog: dict[str, ClaimShape]) -> tuple[list[ClaimAgreement], list[str]]:
    """Every RunRow can carry verdicts for several claim_ids (one recipe run yields all its
    claim-cards' verdicts at once). Groups into one ClaimAgreement per claim_id. Returns
    (agreements, unknown_claim_ids) — a claim_id absent from the catalog is reported, not guessed."""
    by_claim: dict[str, ClaimAgreement] = {}
    unknown: set[str] = set()
    for row in rows:
        if not row.ok:
            continue   # the whole recipe run errored for this PDK -> no verdicts to contribute
        for claim_id, verdict in row.verdicts.items():
            shape = catalog.get(claim_id)
            if shape is None:
                unknown.add(claim_id)
                continue
            rec = by_claim.setdefault(claim_id, ClaimAgreement(
                claim_id=claim_id, topology_class=shape.topology_class,
                verdicts_by_pdk={}, notes_by_pdk={},
                fitted_exponent_by_pdk={} if shape.quant.kind == "elasticity" else None,
                source_paths=set(),
            ))
            rec.verdicts_by_pdk[row.pdk] = verdict
            note = row.verdict_notes.get(claim_id)
            if note:
                rec.notes_by_pdk[row.pdk] = note
            rec.source_paths.add(row.source_path)
            if shape.quant.kind == "elasticity":
                series = row.canonical.get(_route_key(shape.knob, shape.metric))
                rec.fitted_exponent_by_pdk[row.pdk] = _pelgrom_exponent(series)
    return list(by_claim.values()), sorted(unknown)


# ── Status rules (spec §4, deterministic) ──


@dataclass
class StatusResult:
    status: str          # "law" | "process_scoped" (pre-demotion; see resolve_status)
    note: str
    reason: str           # "law" | "two_member" | "insufficient" | "divergent" | "unanimous_non_verified"
    agreeing_pdks: list[str]
    disagreeing_pdks: list[str]


def classify_status(verdicts_by_pdk: dict[str, str]) -> StatusResult:
    """Pure classification off ALREADY-derived per-PDK verdicts (spec §4). No graph access.

    Spec §4 literally names 3 outcomes: `law` (>=3 agreeing), the 2-member `process_scoped`
    law-candidate, and `process_scoped` divergence (named axis). Two more `reason`s are honest,
    disclosed extensions of the SAME "process_scoped, name why" pattern for cases the spec's
    prose doesn't explicitly cover but the classifier must still handle deterministically rather
    than crash or silently drop the claim: `insufficient` (0 or 1 agreeing PDK, nothing disagrees
    — e.g. only one PDK's recipe ran) and `unanimous_non_verified` (no PDK that ran affirms the
    claim — REFUTED-family or a replicated VERIFIED_NEGATIVE: a non-law, not a divergence)."""
    agreeing = sorted(pdk for pdk, v in verdicts_by_pdk.items() if v in _AFFIRMING_FAMILY)
    disagreeing = sorted(pdk for pdk in verdicts_by_pdk if pdk not in agreeing)
    if agreeing and disagreeing:
        diffs = ", ".join(f"{pdk}={verdicts_by_pdk[pdk]}" for pdk in disagreeing)
        note = f"process-dependent: {', '.join(agreeing)} agree, but {diffs}"
        return StatusResult("process_scoped", note, "divergent", agreeing, disagreeing)
    if len(agreeing) >= 3:
        return StatusResult("law", "", "law", agreeing, disagreeing)
    if len(agreeing) == 2:
        note = (f"law-candidate (2 model families: {', '.join(agreeing)}; needs >=3) "
                f"— never over-badge on two")
        return StatusResult("process_scoped", note, "two_member", agreeing, disagreeing)
    if not disagreeing:
        note = f"insufficient replication ({len(agreeing)} model family/families: {', '.join(agreeing) or 'none'})"
        return StatusResult("process_scoped", note, "insufficient", agreeing, disagreeing)
    diffs = ", ".join(f"{pdk}={verdicts_by_pdk[pdk]}" for pdk in disagreeing)
    note = f"no replicated model family affirms the claim ({diffs}) — not a law"
    return StatusResult("process_scoped", note, "unanimous_non_verified", agreeing, disagreeing)


def resolve_status(raw: StatusResult, existing_status: str | None) -> tuple[str, bool]:
    """Overlay demotion (spec §4 `demoted`) on top of the raw classification.

    `demoted` is reserved for a node that WAS `law` (or already `demoted`, i.e. is currently
    diverging having once been a law) and whose freshest data diverges. A node that never reached
    `law` stays `process_scoped` even if it diverges — divergence alone does not manufacture a
    demotion out of nothing (R5: demotion is about a PREVIOUSLY-law node).

    Recovery is not blocked: if the freshest data agrees again (raw.status == "law"), the live
    status returns to "law" (R6 — the graph is a rebuildable, current-data view); the fact it was
    once demoted is preserved permanently in member_summary's `_history`, never erased.

    Returns (final_status, is_new_demotion) — `is_new_demotion` is True only at the exact
    transition moment (existing_status == "law"), so history/journal writes happen exactly once.
    """
    if raw.reason == "divergent" and existing_status in ("law", "demoted"):
        return "demoted", existing_status == "law"
    return raw.status, False


# ── law_id + statement ──


def compute_law_id(topology_class: str, metric: str, knob: str, quant_kind: str) -> str:
    """sha1(topology_class|metric|knob|quant_kind) — spec §3. Stable as new PDKs join (MERGE grows
    members; the id never encodes a PDK)."""
    key = f"{topology_class}|{metric}|{knob}|{quant_kind}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _shape_clause(knob: str, metric: str, quant: QuantTest) -> str:
    """Qualitative, magnitude-free shape description of the claim (never a number)."""
    k = quant.kind
    if k == "direction":
        verb = "increases" if quant.sign == "+" else "decreases"
        return f"{metric} {verb} with {knob}"
    if k == "direction_to_optimum":
        return f"{metric} moves toward an optimum as {knob} changes"
    if k == "invariance":
        return f"{metric} is invariant to {knob}"
    if k == "elasticity":
        return f"{metric} follows a power-law (Pelgrom-type) scaling in {knob}"
    if k == "statistical":
        return f"{metric} mismatch stays within its statistical bound as a function of {knob}"
    if k == "corner":
        return f"{metric} holds across the tested process corner(s) (knob {knob})"
    if k == "value":
        return f"{metric} matches its target value (knob {knob})"
    return f"{metric} vs {knob}"   # defensive fallback for a future QuantTest.kind


def build_statement(topology_class: str, knob: str, metric: str, quant: QuantTest,
                     status_result: StatusResult) -> str:
    """Shape-level, scope-honest prose (spec §3) — NEVER a magnitude. Divergence names the axis
    in the statement itself (spec §4's `process_scoped` rule); law/law-candidate/insufficient
    scope the sentence to the PDKs that actually agree."""
    shape = _shape_clause(knob, metric, quant)
    if status_result.reason == "divergent":
        return (f"For {topology_class}, {shape} — {status_result.note}; "
                f"process-dependent, not (yet) a cross-foundry law.")
    if status_result.reason == "unanimous_non_verified":
        return f"For {topology_class}, {shape} does NOT hold — {status_result.note}."
    if status_result.reason == "insufficient":
        return f"For {topology_class}, {shape}: {status_result.note}."
    pdks = ", ".join(status_result.agreeing_pdks)
    qualifier = "" if status_result.reason == "law" else " (law-candidate, not yet law-tier)"
    return f"For {topology_class}, {shape} — replicated across {pdks}{qualifier}."


# ── LawRecord: the full projector output for one law ──


@dataclass
class LawRecord:
    law_id: str
    topology_class: str
    metric: str
    knob: str
    quant_kind: str
    claim_ids: list[str]
    pdks: list[str]                       # every PDK with ANY recorded verdict (agree or not)
    member_summary: dict[str, Any]        # pdk -> {verdict, note, fitted_exponent?}; "_history" too
    status: str                           # RAW status (pre-demotion overlay — see resolve_status)
    status_note: str
    reason: str                           # RAW StatusResult.reason (pre-demotion) — "law" | "two_member" |
                                           # "insufficient" | "divergent" | "unanimous_non_verified"
    agreeing_pdks: list[str]
    disagreeing_pdks: list[str]
    statement: str
    derived_from: dict[str, Any]


def build_law_records(load_results: dict[str, LoadResult]) -> tuple[list[LawRecord], dict[str, Any]]:
    """Pure: JSONL rows (already loaded, keyed by source path) -> LawRecords + a gap report.
    No graph access — demotion overlay happens later (needs the existing node's status)."""
    catalog = _build_catalog()
    all_rows: list[RunRow] = []
    for result in load_results.values():
        all_rows.extend(result.rows)
    agreements, unknown_claim_ids = _group_by_claim(all_rows, catalog)

    # Group agreements by law_id (normally 1:1 with claim_id; MERGE-compatible if a future
    # full-registry claim_id happens to share topology_class/metric/knob/kind with another).
    by_law: dict[str, list[ClaimAgreement]] = {}
    for agr in agreements:
        shape = catalog[agr.claim_id]
        lid = compute_law_id(shape.topology_class, shape.metric, shape.knob, shape.quant.kind)
        by_law.setdefault(lid, []).append(agr)

    records: list[LawRecord] = []
    for law_id, group in sorted(by_law.items()):
        group = sorted(group, key=lambda a: a.claim_id)
        shape = catalog[group[0].claim_id]
        verdicts_by_pdk: dict[str, str] = {}
        notes_by_pdk: dict[str, str] = {}
        exponents_by_pdk: dict[str, float | None] = {}
        source_paths: set[str] = set()
        for agr in group:   # deterministic (sorted) claim_id order — later entries win on conflict
            verdicts_by_pdk.update(agr.verdicts_by_pdk)
            notes_by_pdk.update(agr.notes_by_pdk)
            if agr.fitted_exponent_by_pdk is not None:
                exponents_by_pdk.update(agr.fitted_exponent_by_pdk)
            source_paths |= agr.source_paths

        status_result = classify_status(verdicts_by_pdk)
        pdks = sorted(verdicts_by_pdk)
        member_summary: dict[str, Any] = {}
        for pdk in pdks:
            entry: dict[str, Any] = {"verdict": verdicts_by_pdk[pdk]}
            if notes_by_pdk.get(pdk):
                entry["note"] = notes_by_pdk[pdk]
            if shape.quant.kind == "elasticity":
                entry["fitted_exponent"] = exponents_by_pdk.get(pdk)
            member_summary[pdk] = entry

        statement = build_statement(shape.topology_class, shape.knob, shape.metric, shape.quant, status_result)
        records.append(LawRecord(
            law_id=law_id, topology_class=shape.topology_class, metric=shape.metric,
            knob=shape.knob, quant_kind=shape.quant.kind,
            claim_ids=[a.claim_id for a in group], pdks=pdks, member_summary=member_summary,
            status=status_result.status, status_note=status_result.note,
            reason=status_result.reason, agreeing_pdks=status_result.agreeing_pdks,
            disagreeing_pdks=status_result.disagreeing_pdks, statement=statement,
            derived_from={"jsonl_paths": sorted(source_paths)},
        ))

    gap_report = {"unknown_claim_ids": unknown_claim_ids}
    return records, gap_report


# ── Graph write (idempotent MERGE, read-before-write) ──


PROJECTION_OP = "project_law"


@dataclass
class LawWriteOutcome:
    law_id: str
    action: str            # "written" | "updated" | "demoted" | "unchanged"
    status: str
    supported_by_written: int
    about_written: int


async def _existing_regularity(store, law_id: str) -> dict | None:
    return await store.get_node(NodeLabel.REGULARITY, "law_id", law_id)


async def _existing_edge_targets(store, law_id: str, rel_type: RelType, target_label: NodeLabel,
                                  target_id_field: str) -> set[str]:
    rows = await store.run_read_query(
        f"MATCH (:{NodeLabel.REGULARITY.value} {{law_id: $lid}})"
        f"-[:{rel_type.value}]->(t:{target_label.value}) "
        f"RETURN t.{target_id_field} AS id",
        {"lid": law_id},
    )
    return {r["id"] for r in rows if r.get("id")}


async def _find_member_card(store, topology_class: str, pdk: str, claim_id: str) -> str | None:
    """The member ClaimCard for (topology_class, pdk, claim_id), if it has been projected onto the
    graph by the (separate) production corpus projector (projection.py). NO-PHANTOM: absent ->
    None -> the caller forms no SUPPORTED_BY edge (spec §3's "member cards must EXIST" rule).
    If several projected specimens of the same (topology_class, pdk) carry this claim, the
    lexicographically-first card gets the edge — deterministic but arbitrary; revisit when ③'s
    full-registry rollout makes multi-specimen members real."""
    rows = await store.run_read_query(
        "MATCH (s:Specimen {topology_class: $tc, pdk: $pdk})-[:HAS_CLAIM]->"
        "(c:ClaimCard {claim: $claim}) "
        "WHERE NOT coalesce(c.retracted, false) "
        "RETURN c.claim_id AS id ORDER BY c.claim_id LIMIT 1",
        {"tc": topology_class, "pdk": pdk, "claim": claim_id},
    )
    return rows[0]["id"] if rows else None


def _node_properties(record: LawRecord, final_status: str, member_summary: dict) -> dict[str, Any]:
    return {
        "law_id": record.law_id,
        "topology_class": record.topology_class,
        "metric": record.metric,
        "knob": record.knob,
        "quant_kind": record.quant_kind,
        "claim_ids": sorted(record.claim_ids),
        "pdks": record.pdks,
        "member_summary": json.dumps(member_summary, sort_keys=True),
        "status": final_status,
        "status_note": record.status_note,
        "statement": record.statement,
        "derived_from": json.dumps(record.derived_from, sort_keys=True),
    }


def _merge_member_summary(record: LawRecord, existing: dict | None, is_new_demotion: bool) -> dict:
    """Thread `_history` forward from the existing node's member_summary (spec §4: 'prior status
    kept in member_summary history'). Only appends a NEW entry at the exact demotion transition —
    an already-demoted node that stays demoted does not accumulate duplicate entries."""
    history: list[dict] = []
    if existing and existing.get("member_summary"):
        try:
            prior = json.loads(existing["member_summary"])
            history = list(prior.get("_history") or [])
        except (json.JSONDecodeError, TypeError):
            history = []
    summary = dict(record.member_summary)
    if is_new_demotion:
        history = history + [{
            "prior_status": "law",
            "demoted_reason": record.status_note,
            "pdks_at_demotion": record.pdks,
        }]
    if history:
        summary["_history"] = history
    return summary


async def project_law_records(
    store,
    records: list[LawRecord],
    *,
    about_resolver: Resolver | None = None,
    journal=None,
    apply: bool = False,
) -> list[LawWriteOutcome]:
    """Resolve SUPPORTED_BY/ABOUT targets and (if apply) write only what actually changed.

    Dry-run (apply=False, default): resolves everything (read-only) so the CLI can preview the
    full law table + link plan; issues NO writes.
    apply=True: reads each law's existing node + existing edges first, diffs, and MERGEs only the
    delta — so a second --apply run over UNCHANGED input data writes nothing (R6 idempotent
    re-projection; write_batch's own no-op guard makes an empty nodes/edges call a true no-op).

    Accumulate-field shrink guard (Known issues #1): `record` is computed ENTIRELY from THIS call's
    loaded JSONL rows (see build_law_records's docstring — "No graph access"), so if the caller's
    input this time covers fewer pdks/claim_ids than the law already has on the graph (e.g. a plain
    `project-laws --raw <one narrower file>`, not unioned with the standing default/historical
    JSONLs the way `reverify.py::refeed_project_laws` always does), writing `node_props` as-is would
    silently REPLACE (never union) the existing pdks/member_summary/status/derived_from — regressing
    a law from e.g. 3-PDK "law" down to 1-PDK "process_scoped" even though the other PDKs' underlying
    data never changed. Detected per-record below by simple set containment (existing minus incoming
    non-empty); when it fires, the record is refused outright (no node write, no new edges) with a
    loud warning naming the law_id — the existing node is left exactly as it was. Re-run with the
    historical JSONL(s) unioned in (as `reverify.py::refeed_project_laws` already does) to update the
    law safely instead.
    """
    outcomes: list[LawWriteOutcome] = []
    nodes: list[dict] = []
    edges: list[dict] = []

    for record in records:
        existing = await _existing_regularity(store, record.law_id)
        existing_status = existing.get("status") if existing else None

        if existing is not None:
            existing_pdks = set(existing.get("pdks") or [])
            existing_claim_ids = set(existing.get("claim_ids") or [])
            missing_pdks = existing_pdks - set(record.pdks)
            missing_claim_ids = existing_claim_ids - set(record.claim_ids)
            if missing_pdks or missing_claim_ids:
                logger.warning(
                    "laws: refusing to project law_id=%s (topology_class=%s metric=%s knob=%s "
                    "quant_kind=%s) — incoming member set would SHRINK the accumulated law "
                    "(missing pdks=%s, missing claim_ids=%s). existing pdks=%s claim_ids=%s "
                    "status=%s vs incoming pdks=%s claim_ids=%s — no write applied for this law; "
                    "union the historical JSONL(s) in (see cli.py::_default_law_jsonl_paths / "
                    "reverify.py::refeed_project_laws) to update it safely.",
                    record.law_id, record.topology_class, record.metric, record.knob,
                    record.quant_kind, sorted(missing_pdks), sorted(missing_claim_ids),
                    sorted(existing_pdks), sorted(existing_claim_ids), existing_status,
                    record.pdks, record.claim_ids,
                )
                outcomes.append(LawWriteOutcome(
                    law_id=record.law_id, action="refused_narrowing", status=existing_status,
                    supported_by_written=0, about_written=0,
                ))
                continue

        raw = StatusResult(record.status, record.status_note, record.reason,
                            record.agreeing_pdks, record.disagreeing_pdks)
        final_status, is_new_demotion = resolve_status(raw, existing_status)
        member_summary = _merge_member_summary(record, existing, is_new_demotion)
        node_props = _node_properties(record, final_status, member_summary)

        existing_supported = await _existing_edge_targets(
            store, record.law_id, RelType.SUPPORTED_BY, NodeLabel.CLAIM_CARD, "claim_id",
        ) if existing else set()
        existing_about = await _existing_edge_targets(
            store, record.law_id, RelType.ABOUT, NodeLabel.CIRCUIT_TOPOLOGY, "topology_id",
        ) if existing else set()

        new_supported: list[str] = []
        for pdk in record.pdks:
            for claim_id in record.claim_ids:
                target = await _find_member_card(store, record.topology_class, pdk, claim_id)
                if target and target not in existing_supported and target not in new_supported:
                    new_supported.append(target)

        new_about: str | None = None
        if about_resolver is not None:
            match_text = record.topology_class.replace("_", " ")
            resolved = await about_resolver(NodeLabel.CIRCUIT_TOPOLOGY, match_text)
            if resolved and resolved not in existing_about:
                new_about = resolved

        # Idempotency: compare computed node properties (minus bookkeeping) against what's
        # already stored. Identical + no new edges -> truly nothing to write.
        unchanged_node = existing is not None and _same_node(existing, node_props)
        node_changed = not unchanged_node
        has_new_edges = bool(new_supported) or new_about is not None

        if node_changed:
            nodes.append({
                "label": NodeLabel.REGULARITY, "id_field": "law_id",
                "id_value": record.law_id, "properties": node_props,
            })
        for target in new_supported:
            edges.append({
                "source_label": NodeLabel.REGULARITY, "source_id_field": "law_id",
                "source_id_value": record.law_id,
                "target_label": NodeLabel.CLAIM_CARD, "target_id_field": "claim_id",
                "target_id_value": target, "rel_type": RelType.SUPPORTED_BY, "properties": {},
            })
        if new_about is not None:
            edges.append({
                "source_label": NodeLabel.REGULARITY, "source_id_field": "law_id",
                "source_id_value": record.law_id,
                "target_label": NodeLabel.CIRCUIT_TOPOLOGY, "target_id_field": "topology_id",
                "target_id_value": new_about, "rel_type": RelType.ABOUT, "properties": {},
            })

        if is_new_demotion:
            action = "demoted"
        elif not node_changed and not has_new_edges:
            action = "unchanged"
        elif existing is None:
            action = "written"
        else:
            action = "updated"
        outcomes.append(LawWriteOutcome(
            law_id=record.law_id, action=action, status=final_status,
            supported_by_written=len(new_supported), about_written=1 if new_about else 0,
        ))
        if apply and journal is not None and action != "unchanged":
            journal.log(PROJECTION_OP, law_id=record.law_id, topology_class=record.topology_class,
                        claim_ids=record.claim_ids, action=action, status=final_status,
                        pdks=record.pdks)

    if apply:
        await store.write_batch(nodes=nodes, edges=edges)
    return outcomes


def _same_node(existing: dict, computed: dict) -> bool:
    keys = ("topology_class", "metric", "knob", "quant_kind", "claim_ids", "pdks",
            "member_summary", "status", "status_note", "statement", "derived_from")
    for key in keys:
        if existing.get(key) != computed.get(key):
            return False
    return True
