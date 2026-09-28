"""E4a — the re-verification cycle (ADR-045; docs/superpowers/specs/2026-07-05-e4-reverification-
coherence.md). "Right once" -> "self-correcting": re-runs a sample of the corpus's recipes via the
EXISTING `run_one` (scripts/replicate_cross_pdk.py — reused verbatim, never reimplemented) and
diffs each fresh verdict against the currently-projected ClaimCard (queried from Neo4j), producing
a drift table classified CONCORDANT / DRIFTED / VALUE-DRIFT / MISSING / ERROR (spec §2). `--apply`
(CLI: `openclaw-brain reverify`) re-projects ONLY the drifted/value-drift (topology_class, pdk)
pairs through the EXISTING projection path (GraphProjector — not reimplemented), journaled as a
`reverify` op distinct from a first projection, and re-feeds `project-laws` so an affected law
re-derives its status.

Graph/corpus split (read before editing — a real gap between the spec's mechanics prose and the
actual projection path, deliberately NOT closed here since projection.py is reused as-is, never
modified): `GraphProjector`/`project_specimen` (projection.py) does NOT persist `verdict_note` as
a ClaimCard property — only `verdict` (the class) and `narrative` (the SEPARATE, never-certified
mechanism prose) are projected. So the primary DRIFTED/CONCORDANT signal (`verdict`) comes from
Neo4j exactly as the spec describes; the SCALAR needed for VALUE-DRIFT comes instead from the
git-corpus specimen's `claim_cards.yaml` (corpus.py — the actual SSOT the graph's ClaimCard is
itself derived from, and it DOES persist `verdict_note`), keyed by the same `spec_id` the graph's
Specimen node carries. Pass `corpus=None` (CLI: omit is not possible — `--corpus` always defaults
to the configured corpus dir, see cli.py) to disable this and get verdict-CLASS-only diffing;
VALUE-DRIFT then never fires — it is never fabricated from data the instrument doesn't have.

Determinism honesty (spec §2): a VALUE-DRIFT on a claim whose measurement is Monte-Carlo (kind
`statistical`/`elasticity`) is a real signal ONLY if that recipe's build.template_ref is one of the
templates carrying the documented, hardened `setseed` stride (mc_templates.py — the fix that
corrected the Pelgrom founding artifact, E1b). Every other claim kind has no random source at all
(single-run deterministic AC/DC/corner sweeps) and is trivially "seeded" (no RNG to be dishonest
about). An MC-kind claim from any OTHER template is labelled UNSEEDED — its scalar is not
certified as reproducible, so a VALUE-DRIFT there is reported as instrument noise, never eligible
to be `--apply`-corrected as a genuine correction (see `is_known_seeded`).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import VerificationRecipe
from .seeds import seed_recipes

# ── Sample selection ──

PDKS = ["sky130A", "gf180mcuD", "ihp-sg13g2"]   # mirrors scripts/replicate_cross_pdk.py::PDKS exactly
BASELINE_PDK = "sky130A"

# The 5 NOMINAL law topologies (spec §7 Q1's live sample): the 3 E1 pilots (current_mirror_
# simple_nmos, common_source_active_load_nmos, ota_5t_nmos_in's nominal av0/gbw claims) + the next
# 2 alphabetically among the OTHER topology_classes whose Regularity carries status "law" on a
# NON-statistical/elasticity claim, per the live graph as of 2026-07-05 (26 projected Regularities;
# `MATCH (r:Regularity) WHERE r.status='law' AND NOT r.quant_kind IN ['statistical','elasticity']
# RETURN DISTINCT r.topology_class` -> 14 candidates; cascode_current_mirror_nmos and
# cds_switched_cap_nmos are the first 2 alphabetically not already a pilot). Sorted for determinism.
NOMINAL_SAMPLE_CLASSES = [
    "cascode_current_mirror_nmos",
    "cds_switched_cap_nmos",
    "common_source_active_load_nmos",
    "current_mirror_simple_nmos",
    "ota_5t_nmos_in",
]


def sample_recipes(sample_values: str | list[str] | tuple[str, ...],
                    recipes: list[VerificationRecipe] | None = None) -> list[VerificationRecipe]:
    """`--sample nominal|all|<topology_class>...` -> the filtered recipe list, in seed_recipes()
    order (deterministic). 'all' takes every seed_recipes() topology (unfiltered, mirrors
    replicate_cross_pdk.py::full_registry_recipes); 'nominal' selects NOMINAL_SAMPLE_CLASSES;
    anything else is treated as one or more explicit topology_class names (an unmatched name
    simply selects nothing — reported as 0 recipes, never an error)."""
    recipes = list(recipes) if recipes is not None else seed_recipes()
    values = [sample_values] if isinstance(sample_values, str) else list(sample_values)
    if "all" in values:
        return recipes
    classes = set(NOMINAL_SAMPLE_CLASSES) if "nominal" in values else set(values)
    return [r for r in recipes if r.topology_class in classes]


# ── Reuse run_one verbatim (scripts/ is not a packaged module — same dynamic-load pattern
#    tests/test_replicate_cross_pdk.py already uses) ──

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RCP_MODULE = None  # cached after first load


def _load_replicate_module():
    global _RCP_MODULE
    if _RCP_MODULE is not None:
        return _RCP_MODULE
    path = _REPO_ROOT / "scripts" / "replicate_cross_pdk.py"
    spec = importlib.util.spec_from_file_location("replicate_cross_pdk", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module   # required before exec: see test_replicate_cross_pdk.py comment
    spec.loader.exec_module(module)
    _RCP_MODULE = module
    return module


def default_run_one(recipe: VerificationRecipe, pdk: str):
    """The production re-run callable: `scripts/replicate_cross_pdk.py::run_one`, imported
    dynamically and called UNMODIFIED (its own NgspiceRunner default applies)."""
    return _load_replicate_module().run_one(recipe, pdk)


# ── Drift classification ──

DRIFT_CLASSES = ("CONCORDANT", "DRIFTED", "VALUE-DRIFT", "MISSING", "ERROR")


@dataclass
class DriftRow:
    """One (topology_class, pdk, claim_id) diff — a row of the drift table."""

    topology_class: str
    pdk: str
    claim_id: str
    quant_kind: str
    classification: str            # one of DRIFT_CLASSES
    old_verdict: str | None        # the currently-projected ClaimCard's verdict; None if MISSING
    new_verdict: str | None        # the fresh re-run's verdict; None if ERROR (re-run itself failed)
    detail: str                    # human-readable: old->new, scalar delta, error, or why concordant
    spec_id: str | None = None     # the projected specimen's spec_id (None if MISSING/no card found)
    seeded: bool | None = None     # set only on VALUE-DRIFT rows — True: known-seeded (real signal),
                                    # False: unseeded (instrument noise, never certified)


# Scalar patterns matched against oracle.py's own verdict_note format strings (judge_quant) — one
# source of truth, no re-derivation of the oracle's math, just parsing its OWN printed number.
_SCALAR_PATTERNS: dict[str, tuple[str, ...]] = {
    "elasticity": (r"global slope\s+([+-]?\d+(?:\.\d+)?)",),
    "invariance": (r"spread\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\(max",
                   r"CoV\s+(\d+(?:\.\d+)?)\s*%"),
    "statistical": (r"[σ]=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",),
    "value": (r"canonical\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",),
}


def _extract_scalar(quant_kind: str, note: str | None) -> float | None:
    """The one scalar (fitted slope / spread / CoV / k·σ / measured value) `judge_quant` printed
    into a verdict_note, for a claim kind that carries one. None for direction/direction_to_optimum/
    corner (pass/fail-shaped notes, no continuous scalar to track) or an unparseable/missing note —
    never fabricated, never raises."""
    if not note:
        return None
    for pattern in _SCALAR_PATTERNS.get(quant_kind, ()):
        m = re.search(pattern, note)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


_TOL_FRACTION = 0.25   # see _tolerance's docstring for the justification


def _tolerance(quant) -> float | None:
    """VALUE-DRIFT tolerance: a FRACTION of the SAME band/bound the oracle already certifies this
    claim against (re-registered, not a fabricated new number) — 25% of the registered width/bound.
    Justification: the Pelgrom founding artifact was a fitted exponent of -0.375 against the
    corrected ~-0.46..-0.48 (a move of ~0.085-0.105) inside an elasticity band typically ~0.3-0.4
    wide (e.g. (-0.6,-0.2)) — 25% of that width (~0.075-0.1) trips on a move of that size while
    routine solver/PDK-model float noise (<<5% of the band) stays CONCORDANT; the same fraction
    applied to invariance's spread_max/cov_max and statistical's bound gives an equally proportional
    guard band without inventing per-kind numbers. None for a kind/claim with no scalar tolerance
    to derive from (direction family, or a `value` claim missing its own `tol`)."""
    if quant.kind == "elasticity" and quant.band:
        lo, hi = quant.band
        return _TOL_FRACTION * (hi - lo)
    if quant.kind == "invariance":
        if quant.spread_max is not None:
            return _TOL_FRACTION * quant.spread_max
        if quant.cov_max is not None:
            return _TOL_FRACTION * quant.cov_max * 100.0   # note's CoV is printed in %, cov_max is fractional
    if quant.kind == "statistical" and quant.bound is not None:
        return _TOL_FRACTION * quant.bound
    if quant.kind == "value" and quant.tol is not None:
        return _TOL_FRACTION * quant.tol
    return None


# The ONLY templates carrying the documented, hardened `setseed` stride (mc_templates.py; the E1b
# fix that corrected the Pelgrom founding artifact) — statistical_seed_recipes()'s 2 MC pilots
# (mirrors scripts/replicate_cross_pdk.py::STATISTICAL_PILOT_TEMPLATE_REFS exactly).
_SEEDED_MC_TEMPLATE_REFS = {"ota5t_offset_mc", "comparator_fpn_mc"}


def is_known_seeded(quant_kind: str, template_ref: str | None) -> bool:
    """Determinism-honesty guard (spec §2): a claim kind with NO random source at all (every kind
    except statistical/elasticity — single-run deterministic AC/DC/corner sweeps) is trivially
    "seeded" (nothing to be dishonest about). A statistical/elasticity claim is "known-seeded" iff
    its recipe's build.template_ref is one of the documented seeded-MC templates; from any OTHER
    template it is UNSEEDED — its scalar is not certified reproducible, so a VALUE-DRIFT there must
    be labelled instrument noise, never certified as a correction (the Pelgrom-in-reverse guard)."""
    if quant_kind not in ("statistical", "elasticity"):
        return True
    return template_ref in _SEEDED_MC_TEMPLATE_REFS


async def query_stored_claim(store, topology_class: str, pdk: str, claim_id: str) -> dict | None:
    """The currently-projected ClaimCard for (topology_class, pdk, claim_id) — mirrors
    laws.py::_find_member_card's MATCH pattern but also returns the verdict + owning spec_id (not
    just presence). Deterministic tie-break on multiple candidates (same arbitrary-but-stable rule
    laws.py already uses): lexicographically-first by claim_id wins."""
    rows = await store.run_read_query(
        "MATCH (s:Specimen {topology_class: $tc, pdk: $pdk})-[:HAS_CLAIM]->"
        "(c:ClaimCard {claim: $claim}) "
        "WHERE NOT coalesce(c.retracted, false) "
        "RETURN s.spec_id AS spec_id, c.claim_id AS claim_card_id, c.verdict AS verdict "
        "ORDER BY c.claim_id LIMIT 1",
        {"tc": topology_class, "pdk": pdk, "claim": claim_id},
    )
    return rows[0] if rows else None


def stored_note_from_corpus(corpus, topology_class: str, spec_id: str | None, claim_id: str) -> str | None:
    """The stored verdict_note for (topology_class, spec_id, claim_id), read from the git-corpus
    specimen (the graph doesn't project this field — see module docstring). None if no corpus is
    wired, no spec_id, the specimen directory is missing, or the claim_id isn't on it — never
    raises (a missing scalar degrades the diff to a class-only comparison, it never crashes it)."""
    if corpus is None or not spec_id:
        return None
    try:
        spec = corpus.load(topology_class, spec_id)
    except Exception:
        return None
    for c in spec.claim_cards:
        if c.id == claim_id:
            return c.verdict_note
    return None


def classify_one(recipe: VerificationRecipe, pdk: str, card, rec_result, old: dict | None,
                  corpus) -> DriftRow:
    """Classify one (recipe, pdk, claim) diff: fresh `rec_result` (a run_one-shaped RunRecord —
    duck-typed: `.ok`, `.error`, `.verdicts` dict, `.verdict_notes` dict) vs `old` (this claim's
    currently-projected {spec_id, verdict}, or None if no card exists)."""
    claim_id = card.id
    quant = card.mechanism.quant
    if not rec_result.ok:
        return DriftRow(
            recipe.topology_class, pdk, claim_id, quant.kind, "ERROR",
            old_verdict=(old["verdict"] if old else None), new_verdict=None,
            detail=f"re-run PORT-FAILED: {rec_result.error}",
            spec_id=(old.get("spec_id") if old else None),
        )
    new_verdict = rec_result.verdicts.get(claim_id)
    new_note = (getattr(rec_result, "verdict_notes", None) or {}).get(claim_id)
    if old is None:
        return DriftRow(
            recipe.topology_class, pdk, claim_id, quant.kind, "MISSING",
            old_verdict=None, new_verdict=new_verdict,
            detail="no projected ClaimCard for this (topology_class, pdk, claim)",
        )
    old_verdict = old["verdict"]
    spec_id = old.get("spec_id")
    if old_verdict != new_verdict:
        return DriftRow(
            recipe.topology_class, pdk, claim_id, quant.kind, "DRIFTED",
            old_verdict=old_verdict, new_verdict=new_verdict,
            detail=f"{old_verdict} -> {new_verdict}", spec_id=spec_id,
        )
    # Same verdict class -- check whether the underlying scalar moved (the Pelgrom shape).
    old_note = stored_note_from_corpus(corpus, recipe.topology_class, spec_id, claim_id)
    tol = _tolerance(quant)
    old_scalar = _extract_scalar(quant.kind, old_note)
    new_scalar = _extract_scalar(quant.kind, new_note)
    if tol is not None and old_scalar is not None and new_scalar is not None:
        delta = abs(new_scalar - old_scalar)
        if delta > tol:
            seeded = is_known_seeded(quant.kind, (recipe.build or {}).get("template_ref"))
            label = ("known-seeded: real signal" if seeded
                     else "UNSEEDED: instrument noise, not a certified correction")
            return DriftRow(
                recipe.topology_class, pdk, claim_id, quant.kind, "VALUE-DRIFT",
                old_verdict=old_verdict, new_verdict=new_verdict,
                detail=(f"{old_verdict} held; scalar {old_scalar:+.4g} -> {new_scalar:+.4g} "
                        f"(delta {delta:.4g} > tol {tol:.4g}); {label}"),
                spec_id=spec_id, seeded=seeded,
            )
        detail = (f"{old_verdict} (scalar {old_scalar:+.4g} -> {new_scalar:+.4g}, "
                  f"delta {delta:.4g} within tol {tol:.4g})")
    else:
        reason = ("no scalar pattern for this quant kind" if tol is None
                  else "stored verdict_note unavailable (graph doesn't project it; pass "
                       "--corpus to read it from the git-corpus specimen)")
        detail = f"{old_verdict} (unchanged; scalar check skipped: {reason})"
    return DriftRow(
        recipe.topology_class, pdk, claim_id, quant.kind, "CONCORDANT",
        old_verdict=old_verdict, new_verdict=new_verdict, detail=detail, spec_id=spec_id,
    )


async def reverify_sample(store, recipes: list[VerificationRecipe], *, corpus=None,
                           run_one_fn=None, pdks: list[str] | None = None) -> list[DriftRow]:
    """The dry-run pass: for every recipe x every pdk, re-run via `run_one_fn` (default:
    `default_run_one`, i.e. the real `run_one`) and classify every one of the recipe's claim cards
    against the graph's currently-projected card. Deterministic order: recipes (caller's order,
    normally seed_recipes()) x pdks (PDKS order) x recipe.claim_cards (definition order)."""
    run_one_fn = run_one_fn or default_run_one
    pdks = pdks or PDKS
    rows: list[DriftRow] = []
    for recipe in recipes:
        for pdk in pdks:
            rec_result = run_one_fn(recipe, pdk)
            for card in recipe.claim_cards:
                old = await query_stored_claim(store, recipe.topology_class, pdk, card.id)
                rows.append(classify_one(recipe, pdk, card, rec_result, old, corpus))
    return rows


# ── Gated re-projection (--apply) ──

DRIFT_APPLY_CLASSES = ("DRIFTED", "VALUE-DRIFT")
REVERIFY_OP = "reverify"


def group_drifted(rows: list[DriftRow]) -> dict[tuple[str, str], list[DriftRow]]:
    """(topology_class, pdk) -> its DRIFTED/VALUE-DRIFT rows. MISSING/ERROR/CONCORDANT rows never
    trigger a re-projection (spec §2: nothing to correct, or nothing changed)."""
    pairs: dict[tuple[str, str], list[DriftRow]] = {}
    for row in rows:
        if row.classification in DRIFT_APPLY_CLASSES:
            pairs.setdefault((row.topology_class, row.pdk), []).append(row)
    return pairs


async def apply_reverify(store, projector, rows: list[DriftRow], recipes: list[VerificationRecipe],
                          *, corpus, runner_factory=None, journal=None,
                          run_recipe_fn=None) -> tuple[list[dict], list[dict]]:
    """Re-project ONLY the (topology_class, pdk) pairs carrying >=1 DRIFTED/VALUE-DRIFT row,
    through the EXISTING projection path: a FRESH runner instance + `run_recipe(rec, runner,
    corpus=corpus)` (project_ok_runs' own re-run+project pattern) followed by
    `projector.project(spec)` (GraphProjector — unmodified). This NEVER trusts the dry-run pass's
    cached `new_verdict` — it re-derives fresh at apply time, so what gets persisted is always the
    oracle's OWN current answer, never a stale or synthetic one (a forced/planted drift used only
    to trigger detection self-heals on --apply rather than getting "certified"). Journaled as
    `REVERIFY_OP` ("reverify"), distinct from project_executable's/`project-laws`'s own ops.

    Returns (outcomes, fresh_run_records): `fresh_run_records` are RunRecord-shaped dicts (one per
    re-projected pair) meant to be written to a JSONL and fed back through `project-laws` (see
    `write_reverify_jsonl`/`refeed_project_laws`) so a law whose member's verdict genuinely changed
    re-derives its status. A pair whose own re-run OR re-projection raises is reported as an error
    entry (`{"topology_class", "pdk", "error"}`) and does NOT abort the rest — both the sim-side
    `run_recipe_fn` call and the graph-side `projector.project` call are per-pair contained (one bad
    pair must not silence the others, and must not lose pairs already re-verified+journaled in
    earlier loop iterations — same discipline as `run_one`; see EPISTEMOLOGY.md Known issues #3).

    Zero drifted pairs (a second --apply over now-concordant data) -> the loop body never executes
    -> outcomes=[] and fresh_run_records=[] -> zero calls to run_recipe/project/journal.log: a true
    no-op, not just an empty write_batch call."""
    if runner_factory is None:
        from .runner import NgspiceRunner
        runner_factory = NgspiceRunner
    if run_recipe_fn is None:
        from .executor import run_recipe as run_recipe_fn

    by_class = {r.topology_class: r for r in recipes}
    pairs = group_drifted(rows)
    outcomes: list[dict] = []
    fresh_records: list[dict] = []
    for topology_class, pdk in sorted(pairs):
        drows = pairs[(topology_class, pdk)]
        recipe = by_class.get(topology_class)
        if recipe is None:
            outcomes.append({"topology_class": topology_class, "pdk": pdk,
                              "error": "recipe not present in the sample passed to apply_reverify"})
            continue
        rec = copy.deepcopy(recipe)
        if pdk != BASELINE_PDK:
            rec.conditions.pdk_profile = {"pdk": pdk}
        runner = runner_factory()
        try:
            result = run_recipe_fn(rec, runner, corpus=corpus)
        except Exception as exc:  # noqa: BLE001 — one bad pair must not silence the rest
            outcomes.append({"topology_class": topology_class, "pdk": pdk,
                              "error": f"{type(exc).__name__}: {exc}"})
            continue
        spec = result.specimen
        try:
            stats = await projector.project(spec)
        except Exception as exc:  # noqa: BLE001 — same discipline as the run_recipe_fn guard above:
            # a projection failure on ONE pair (e.g. a transient Neo4j blip) must not silence the
            # rest of the batch, and must not lose the pairs already re-verified+journaled in
            # earlier loop iterations (see EPISTEMOLOGY.md Known issues #3).
            outcomes.append({"topology_class": topology_class, "pdk": pdk,
                              "error": f"{type(exc).__name__}: {exc}"})
            continue
        claim_ids = sorted({row.claim_id for row in drows})
        old_new = {row.claim_id: {"old": row.old_verdict, "new": row.new_verdict,
                                   "classification": row.classification} for row in drows}
        if journal is not None:
            journal.log(REVERIFY_OP, topology_class=topology_class, pdk=pdk, spec_id=spec.spec_id,
                        claim_ids=claim_ids, old_new=old_new,
                        nodes=stats["nodes"], links_resolved=stats["links_resolved"])
        outcomes.append({"topology_class": topology_class, "pdk": pdk, "spec_id": spec.spec_id,
                          "claim_ids": claim_ids, "old_new": old_new,
                          "nodes": stats["nodes"], "links_resolved": stats["links_resolved"]})
        verdicts = {c.id: (c.verdict.value if c.verdict else "NONE") for c in result.claim_cards}
        notes = {c.id: (c.verdict_note or "") for c in result.claim_cards}
        canonical = {k: list(v) for k, v in result.canonical.items()}
        fresh_records.append({"topology_class": topology_class, "pdk": pdk, "ok": True,
                               "wall_time_s": None, "error": None, "verdicts": verdicts,
                               "verdict_notes": notes, "canonical": canonical})
    return outcomes, fresh_records


REVERIFY_JSONL_BASENAME = "e4_reverify_raw.jsonl"


def write_reverify_jsonl(fresh_records: list[dict[str, Any]], out_dir: str) -> str:
    """Write `apply_reverify`'s fresh_run_records to `<out_dir>/e4_reverify_raw.jsonl`, in the
    EXACT RunRecord JSONL shape scripts/replicate_cross_pdk.py's own outputs use, so
    laws.py::load_run_rows parses it unmodified."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, REVERIFY_JSONL_BASENAME)
    with open(out_path, "w") as f:
        for rec in fresh_records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    return out_path


async def refeed_project_laws(store, jsonl_paths: list[str], *, about_resolver, journal=None):
    """Re-run the EXISTING project-laws pipeline (laws.py — unmodified) over `jsonl_paths`. Callers
    pass the standing default JSONLs (e1/e1b/full-registry) PLUS the fresh reverify JSONL appended
    LAST — `build_law_records`'s per-pdk grouping keeps the LAST-seen verdict for a given
    (topology_class, pdk) claim (dict-update semantics in `_group_by_claim`), so the reverify pass's
    freshly re-derived verdict wins over whatever stale entry an earlier suite's JSONL carries for
    the same pair, letting an affected law's status genuinely re-derive. A path that doesn't exist
    is skipped (reported by the caller via CLI, not raised here)."""
    from .laws import build_law_records, load_run_rows, project_law_records

    load_results = {}
    for path in jsonl_paths:
        if os.path.isfile(path):
            load_results[path] = load_run_rows(path)
    records, gap_report = build_law_records(load_results)
    outcomes = await project_law_records(
        store, records, about_resolver=about_resolver, journal=journal, apply=True,
    )
    return outcomes, gap_report
