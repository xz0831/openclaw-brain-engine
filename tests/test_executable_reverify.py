"""Tests for the E4a re-verification cycle (knowledge/executable/reverify.py) —
docs/superpowers/specs/2026-07-05-e4-reverification-coherence.md.

All mocked (no DB, no docker, no ngspice): a fake graph store for query_stored_claim, a fake
corpus for the verdict_note scalar fallback, and stubbed run_one/run_recipe callables — mirrors
the repo convention in test_executable_laws.py / test_executable_projection.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from openclaw_brain.knowledge.executable.laws import build_law_records, load_run_rows
from openclaw_brain.knowledge.executable.models import (
    AnalogPVT, ClaimCard, MechanismClaim, QuantTest, VerificationRecipe,
)
from openclaw_brain.knowledge.executable.reverify import (
    DriftRow,
    apply_reverify,
    classify_one,
    group_drifted,
    is_known_seeded,
    query_stored_claim,
    reverify_sample,
    sample_recipes,
    write_reverify_jsonl,
    NOMINAL_SAMPLE_CLASSES,
    PDKS,
)
from openclaw_brain.knowledge.executable.seeds import seed_recipes

REPO_ROOT = Path(__file__).resolve().parent.parent
COND = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)


def _direction_card(cid="cm_iout", topology_class="current_mirror_simple_nmos"):
    return ClaimCard(id=cid, topology_class=topology_class, conditions=COND,
                     mechanism=MechanismClaim(knob="Vout", metric="iout_a", series_ref=cid,
                                              quant=QuantTest(kind="direction", sign="+")))


def _elasticity_card(cid="ota5t_pelgrom", topology_class="ota_5t_nmos_in", band=(-0.6, -0.2)):
    return ClaimCard(id=cid, topology_class=topology_class, conditions=COND,
                     mechanism=MechanismClaim(knob="area", metric="a_vos", series_ref=cid,
                                              quant=QuantTest(kind="elasticity", band=band)))


def _recipe(topology_class, cards, template_ref=None):
    build = {"template_ref": template_ref} if template_ref else {}
    return VerificationRecipe(topology_class=topology_class, conditions=COND,
                              claim_cards=cards, build=build)


def _run_record(ok=True, error=None, verdicts=None, verdict_notes=None):
    return SimpleNamespace(ok=ok, error=error, verdicts=verdicts or {}, verdict_notes=verdict_notes or {})


# ── classify_one: each drift class ──


def test_classify_one_concordant_same_verdict_no_scalar_kind():
    card = _direction_card()
    recipe = _recipe("current_mirror_simple_nmos", [card])
    old = {"spec_id": "sha256:abc", "verdict": "VERIFIED"}
    rec_result = _run_record(verdicts={"cm_iout": "VERIFIED"}, verdict_notes={"cm_iout": "trend +, monotonic=True"})
    row = classify_one(recipe, "sky130A", card, rec_result, old, corpus=None)
    assert row.classification == "CONCORDANT"
    assert row.old_verdict == row.new_verdict == "VERIFIED"


def test_classify_one_drifted_verdict_class_changed():
    card = _direction_card()
    recipe = _recipe("current_mirror_simple_nmos", [card])
    old = {"spec_id": "sha256:abc", "verdict": "VERIFIED"}
    rec_result = _run_record(verdicts={"cm_iout": "REFUTED"}, verdict_notes={"cm_iout": "trend - != predicted +"})
    row = classify_one(recipe, "sky130A", card, rec_result, old, corpus=None)
    assert row.classification == "DRIFTED"
    assert row.old_verdict == "VERIFIED" and row.new_verdict == "REFUTED"
    assert "VERIFIED -> REFUTED" in row.detail


def test_classify_one_missing_when_no_stored_card():
    card = _direction_card()
    recipe = _recipe("current_mirror_simple_nmos", [card])
    rec_result = _run_record(verdicts={"cm_iout": "VERIFIED"})
    row = classify_one(recipe, "sky130A", card, rec_result, old=None, corpus=None)
    assert row.classification == "MISSING"
    assert row.old_verdict is None
    assert row.new_verdict == "VERIFIED"


def test_classify_one_error_when_rerun_port_failed():
    card = _direction_card()
    recipe = _recipe("current_mirror_simple_nmos", [card])
    old = {"spec_id": "sha256:abc", "verdict": "VERIFIED"}
    rec_result = _run_record(ok=False, error="RuntimeError: ngspice nonconvergence")
    row = classify_one(recipe, "sky130A", card, rec_result, old, corpus=None)
    assert row.classification == "ERROR"
    assert row.new_verdict is None
    assert "PORT-FAILED" in row.detail
    assert "nonconvergence" in row.detail
    # the OLD (last-known) verdict is still surfaced, never silenced
    assert row.old_verdict == "VERIFIED"


class _FakeCorpus:
    def __init__(self):
        self.specs: dict[tuple[str, str], SimpleNamespace] = {}

    def load(self, topology_class, spec_id):
        key = (topology_class, spec_id)
        if key not in self.specs:
            raise FileNotFoundError(key)
        return self.specs[key]


def _wire_note(corpus, topology_class, spec_id, claim_id, note):
    key = (topology_class, spec_id)
    spec = corpus.specs.setdefault(key, SimpleNamespace(claim_cards=[]))
    spec.claim_cards.append(SimpleNamespace(id=claim_id, verdict_note=note))


def test_classify_one_value_drift_seeded_is_a_real_signal():
    card = _elasticity_card()
    recipe = _recipe("ota_5t_nmos_in", [card], template_ref="ota5t_offset_mc")   # a seeded MC template
    old = {"spec_id": "sha256:abc", "verdict": "VERIFIED"}
    corpus = _FakeCorpus()
    _wire_note(corpus, "ota_5t_nmos_in", "sha256:abc", "ota5t_pelgrom",
               "global slope -0.470 in band; locals_all_in=True")
    rec_result = _run_record(verdicts={"ota5t_pelgrom": "VERIFIED"},
                              verdict_notes={"ota5t_pelgrom": "global slope -0.300 in band; locals_all_in=True"})
    row = classify_one(recipe, "sky130A", card, rec_result, old, corpus=corpus)
    assert row.classification == "VALUE-DRIFT"
    assert row.seeded is True
    assert "known-seeded" in row.detail
    assert row.old_verdict == row.new_verdict == "VERIFIED"   # the Pelgrom shape: class held, number moved


def test_classify_one_value_drift_unseeded_is_labelled_instrument_noise():
    card = _elasticity_card()
    # NOT one of the documented seeded-MC templates -> unseeded
    recipe = _recipe("ota_5t_nmos_in", [card], template_ref="some_unrecorded_mc_variant")
    old = {"spec_id": "sha256:abc", "verdict": "VERIFIED"}
    corpus = _FakeCorpus()
    _wire_note(corpus, "ota_5t_nmos_in", "sha256:abc", "ota5t_pelgrom",
               "global slope -0.470 in band; locals_all_in=True")
    rec_result = _run_record(verdicts={"ota5t_pelgrom": "VERIFIED"},
                              verdict_notes={"ota5t_pelgrom": "global slope -0.300 in band; locals_all_in=True"})
    row = classify_one(recipe, "sky130A", card, rec_result, old, corpus=corpus)
    assert row.classification == "VALUE-DRIFT"
    assert row.seeded is False
    assert "UNSEEDED" in row.detail
    assert "instrument noise" in row.detail


def test_classify_one_concordant_when_scalar_within_tolerance():
    card = _elasticity_card()
    recipe = _recipe("ota_5t_nmos_in", [card], template_ref="ota5t_offset_mc")
    old = {"spec_id": "sha256:abc", "verdict": "VERIFIED"}
    corpus = _FakeCorpus()
    _wire_note(corpus, "ota_5t_nmos_in", "sha256:abc", "ota5t_pelgrom",
               "global slope -0.470 in band; locals_all_in=True")
    rec_result = _run_record(verdicts={"ota5t_pelgrom": "VERIFIED"},
                              verdict_notes={"ota5t_pelgrom": "global slope -0.480 in band; locals_all_in=True"})
    row = classify_one(recipe, "sky130A", card, rec_result, old, corpus=corpus)
    assert row.classification == "CONCORDANT"
    assert "within tol" in row.detail


def test_classify_one_concordant_when_no_corpus_wired_never_fabricates_value_drift():
    card = _elasticity_card()
    recipe = _recipe("ota_5t_nmos_in", [card], template_ref="ota5t_offset_mc")
    old = {"spec_id": "sha256:abc", "verdict": "VERIFIED"}
    rec_result = _run_record(verdicts={"ota5t_pelgrom": "VERIFIED"},
                              verdict_notes={"ota5t_pelgrom": "global slope -0.300 in band; locals_all_in=True"})
    row = classify_one(recipe, "sky130A", card, rec_result, old, corpus=None)   # no corpus -> no stored note
    assert row.classification == "CONCORDANT"
    assert "scalar check skipped" in row.detail


def test_is_known_seeded_direction_kind_always_true_no_random_source():
    assert is_known_seeded("direction", template_ref=None) is True
    assert is_known_seeded("invariance", template_ref="anything") is True


def test_is_known_seeded_statistical_kind_gated_by_template_ref():
    assert is_known_seeded("statistical", "comparator_fpn_mc") is True
    assert is_known_seeded("elasticity", "ota5t_offset_mc") is True
    assert is_known_seeded("elasticity", "not_a_seeded_template") is False
    assert is_known_seeded("statistical", None) is False


# ── sample_recipes ──


def test_sample_recipes_nominal_selects_the_5_named_classes():
    recs = sample_recipes("nominal", seed_recipes())
    got = {r.topology_class for r in recs}
    assert got == set(NOMINAL_SAMPLE_CLASSES)
    assert len(NOMINAL_SAMPLE_CLASSES) == 5


def test_sample_recipes_all_selects_every_seed_recipe():
    all_recipes = seed_recipes()
    assert sample_recipes("all", all_recipes) == all_recipes


def test_sample_recipes_explicit_topology_names():
    recs = sample_recipes(["current_mirror_simple_nmos"], seed_recipes())
    assert {r.topology_class for r in recs} == {"current_mirror_simple_nmos"}


def test_sample_recipes_unmatched_explicit_name_is_empty_not_an_error():
    recs = sample_recipes(["not_a_real_topology"], seed_recipes())
    assert recs == []


# ── query_stored_claim (fake store) ──


class _FakeGraphStore:
    def __init__(self):
        self.cards: dict[tuple[str, str, str], dict] = {}

    async def run_read_query(self, query, params=None):
        params = params or {}
        key = (params["tc"], params["pdk"], params["claim"])
        row = self.cards.get(key)
        return [row] if row else []


async def test_query_stored_claim_found_and_missing():
    store = _FakeGraphStore()
    store.cards[("current_mirror_simple_nmos", "sky130A", "cm_iout")] = {
        "spec_id": "sha256:abc", "claim_card_id": "sha256:abc:cm_iout", "verdict": "VERIFIED",
    }
    found = await query_stored_claim(store, "current_mirror_simple_nmos", "sky130A", "cm_iout")
    assert found["verdict"] == "VERIFIED"
    missing = await query_stored_claim(store, "current_mirror_simple_nmos", "gf180mcuD", "cm_iout")
    assert missing is None


# ── reverify_sample: deterministic dispatch + the Q2 planted-drift case ──


async def test_reverify_sample_dispatches_every_recipe_x_pdk_x_claim_deterministically():
    card1 = _direction_card("cm_iout")
    recipe = _recipe("current_mirror_simple_nmos", [card1])
    store = _FakeGraphStore()
    for pdk in PDKS:
        store.cards[("current_mirror_simple_nmos", pdk, "cm_iout")] = {
            "spec_id": f"sha256:{pdk}", "claim_card_id": "x", "verdict": "VERIFIED",
        }
    calls = []

    def fake_run_one(recipe, pdk):
        calls.append((recipe.topology_class, pdk))
        return _run_record(verdicts={"cm_iout": "VERIFIED"}, verdict_notes={"cm_iout": "trend +, monotonic=True"})

    rows = await reverify_sample(store, [recipe], run_one_fn=fake_run_one)
    assert calls == [("current_mirror_simple_nmos", pdk) for pdk in PDKS]   # PDKS order, deterministic
    assert len(rows) == 3
    assert all(r.classification == "CONCORDANT" for r in rows)
    assert [r.pdk for r in rows] == PDKS


async def test_reverify_sample_planted_drift_is_flagged_drifted_with_old_new_named():
    """Q2's unit-level proof: a stubbed run_one that flips ONE (recipe, pdk)'s verdict away from
    what the graph has stored must surface as DRIFTED with old->new named; the other 2 PDKs (which
    the stub reports unchanged) must stay CONCORDANT."""
    card = _direction_card("cm_iout")
    recipe = _recipe("current_mirror_simple_nmos", [card])
    store = _FakeGraphStore()
    for pdk in PDKS:
        store.cards[("current_mirror_simple_nmos", pdk, "cm_iout")] = {
            "spec_id": f"sha256:{pdk}", "claim_card_id": "x", "verdict": "VERIFIED",
        }

    def planted_run_one(recipe, pdk):
        if pdk == "gf180mcuD":   # the planted drift
            return _run_record(verdicts={"cm_iout": "REFUTED"}, verdict_notes={"cm_iout": "trend - != predicted +"})
        return _run_record(verdicts={"cm_iout": "VERIFIED"}, verdict_notes={"cm_iout": "trend +, monotonic=True"})

    rows = await reverify_sample(store, [recipe], run_one_fn=planted_run_one)
    by_pdk = {r.pdk: r for r in rows}
    assert by_pdk["gf180mcuD"].classification == "DRIFTED"
    assert by_pdk["gf180mcuD"].old_verdict == "VERIFIED" and by_pdk["gf180mcuD"].new_verdict == "REFUTED"
    assert by_pdk["sky130A"].classification == "CONCORDANT"
    assert by_pdk["ihp-sg13g2"].classification == "CONCORDANT"


# ── group_drifted ──


def test_group_drifted_only_groups_drifted_and_value_drift_rows():
    rows = [
        DriftRow("t", "sky130A", "c1", "direction", "CONCORDANT", "VERIFIED", "VERIFIED", "ok"),
        DriftRow("t", "sky130A", "c2", "direction", "DRIFTED", "VERIFIED", "REFUTED", "flip"),
        DriftRow("t", "gf180mcuD", "c1", "elasticity", "VALUE-DRIFT", "VERIFIED", "VERIFIED", "moved"),
        DriftRow("t", "ihp-sg13g2", "c1", "direction", "MISSING", None, "VERIFIED", "no card"),
        DriftRow("t", "ihp-sg13g2", "c2", "direction", "ERROR", "VERIFIED", None, "port-failed"),
    ]
    pairs = group_drifted(rows)
    assert set(pairs) == {("t", "sky130A"), ("t", "gf180mcuD")}
    assert len(pairs[("t", "sky130A")]) == 1 and pairs[("t", "sky130A")][0].claim_id == "c2"


# ── apply_reverify: re-projects ONLY drifted pairs, journals, idempotent ──


class _FakeProjector:
    def __init__(self):
        self.projected: list[str] = []

    async def project(self, spec):
        self.projected.append(spec.spec_id)
        return {"nodes": 2, "internal_edges": 1, "links_resolved": 0, "links_total": 1}


class _FakeJournal:
    def __init__(self):
        self.entries: list[dict] = []

    def log(self, op, **kwargs):
        self.entries.append({"op": op, **kwargs})


def _fake_run_recipe_factory(spec_by_pair):
    """Builds a run_recipe_fn stub keyed by (topology_class, pdk-if-overridden) — reads pdk off
    `rec.conditions.pdk_profile` the same way apply_reverify's own copy.deepcopy + override does."""
    def _fn(rec, runner, corpus=None):
        pdk = (rec.conditions.pdk_profile or {}).get("pdk", "sky130A")
        spec_id = spec_by_pair[(rec.topology_class, pdk)]
        specimen = SimpleNamespace(spec_id=spec_id, topology_class=rec.topology_class)
        claim_cards = [SimpleNamespace(id=c.id, verdict=SimpleNamespace(value="VERIFIED"),
                                       verdict_note="trend +, monotonic=True") for c in rec.claim_cards]
        return SimpleNamespace(specimen=specimen, claim_cards=claim_cards, canonical={})
    return _fn


async def test_apply_reverify_reprojects_only_drifted_pairs_and_journals():
    card = _direction_card("cm_iout")
    recipe_a = _recipe("topo_a", [card])
    recipe_b = _recipe("topo_b", [card])
    rows = [
        DriftRow("topo_a", "sky130A", "cm_iout", "direction", "DRIFTED", "VERIFIED", "REFUTED", "flip",
                 spec_id="sha256:old-a"),
        DriftRow("topo_b", "sky130A", "cm_iout", "direction", "CONCORDANT", "VERIFIED", "VERIFIED", "ok",
                 spec_id="sha256:old-b"),
    ]
    projector = _FakeProjector()
    journal = _FakeJournal()
    run_recipe_fn = _fake_run_recipe_factory({("topo_a", "sky130A"): "sha256:new-a"})

    outcomes, fresh = await apply_reverify(
        store=None, projector=projector, rows=rows, recipes=[recipe_a, recipe_b],
        corpus=object(), runner_factory=lambda: object(), journal=journal, run_recipe_fn=run_recipe_fn,
    )
    assert len(outcomes) == 1 and outcomes[0]["topology_class"] == "topo_a"   # topo_b (concordant) untouched
    assert projector.projected == ["sha256:new-a"]
    assert len(journal.entries) == 1
    assert journal.entries[0]["op"] == "reverify"
    assert journal.entries[0]["claim_ids"] == ["cm_iout"]
    assert len(fresh) == 1 and fresh[0]["topology_class"] == "topo_a"


class _FailingProjector:
    """Like _FakeProjector, but raises for one specific spec_id — proves apply_reverify's per-pair
    containment on the projection side (mirrors _FakeProjector's exact return shape otherwise)."""

    def __init__(self, fail_spec_id: str):
        self.projected: list[str] = []
        self.fail_spec_id = fail_spec_id

    async def project(self, spec):
        if spec.spec_id == self.fail_spec_id:
            raise RuntimeError("neo4j blip")
        self.projected.append(spec.spec_id)
        return {"nodes": 2, "internal_edges": 1, "links_resolved": 0, "links_total": 1}


async def test_apply_reverify_projection_failure_on_one_pair_does_not_abort_the_rest():
    """The reverify.py:383 defect: apply_reverify's graph-projection call used to be unguarded while
    the (riskier) simulation call two lines above it WAS guarded — a projection failure on one
    drifted pair aborted the whole apply_reverify call, silently losing outcomes/fresh_records for
    every pair already re-verified+journaled in earlier loop iterations. Fixed: per-pair containment
    matching the sim-side guard exactly — the failure is reported, not swallowed, and the rest of
    the batch still completes."""
    card = _direction_card("cm_iout")
    recipe_a = _recipe("topo_a", [card])
    recipe_c = _recipe("topo_c", [card])
    rows = [
        DriftRow("topo_a", "sky130A", "cm_iout", "direction", "DRIFTED", "VERIFIED", "REFUTED", "flip",
                 spec_id="sha256:old-a"),
        DriftRow("topo_c", "sky130A", "cm_iout", "direction", "DRIFTED", "VERIFIED", "REFUTED", "flip",
                 spec_id="sha256:old-c"),
    ]
    journal = _FakeJournal()
    run_recipe_fn = _fake_run_recipe_factory({
        ("topo_a", "sky130A"): "sha256:new-a", ("topo_c", "sky130A"): "sha256:new-c",
    })
    # apply_reverify iterates `sorted(pairs)` -> "topo_a" is processed before "topo_c"; failing
    # topo_a's projection first is the concrete regression case (today this would abort topo_c
    # entirely and lose its already-completed run_recipe_fn work).
    projector = _FailingProjector(fail_spec_id="sha256:new-a")

    outcomes, fresh = await apply_reverify(
        store=None, projector=projector, rows=rows, recipes=[recipe_a, recipe_c],
        corpus=object(), runner_factory=lambda: object(), journal=journal, run_recipe_fn=run_recipe_fn,
    )

    # topo_c's re-projection must still have happened and been reported, despite topo_a's failure.
    assert projector.projected == ["sha256:new-c"]
    assert len(outcomes) == 2
    by_class = {o["topology_class"]: o for o in outcomes}
    assert by_class["topo_a"] == {"topology_class": "topo_a", "pdk": "sky130A",
                                   "error": "RuntimeError: neo4j blip"}
    assert by_class["topo_c"]["spec_id"] == "sha256:new-c"
    # only the successfully-reprojected pair is journaled / fed forward for the project-laws re-feed
    assert len(journal.entries) == 1 and journal.entries[0]["topology_class"] == "topo_c"
    assert len(fresh) == 1 and fresh[0]["topology_class"] == "topo_c"


async def test_apply_reverify_second_pass_over_concordant_data_is_zero_writes():
    """The idempotency claim: once reverify_sample reports everything CONCORDANT (nothing in
    DRIFT_APPLY_CLASSES), apply_reverify must not call run_recipe/project/journal at all."""
    recipe_a = _recipe("topo_a", [_direction_card("cm_iout")])
    projector = _FakeProjector()
    journal = _FakeJournal()
    calls = []
    run_recipe_fn = lambda rec, runner, corpus=None: calls.append(1) or None  # noqa: E731 — must never run

    all_concordant_rows = [
        DriftRow("topo_a", "sky130A", "cm_iout", "direction", "CONCORDANT", "VERIFIED", "VERIFIED", "ok"),
    ]
    outcomes, fresh = await apply_reverify(
        store=None, projector=projector, rows=all_concordant_rows, recipes=[recipe_a],
        corpus=object(), runner_factory=lambda: object(), journal=journal, run_recipe_fn=run_recipe_fn,
    )
    assert outcomes == [] and fresh == []
    assert projector.projected == []
    assert journal.entries == []
    assert calls == []


# ── write_reverify_jsonl / project-laws re-feed plumbing ──


def test_write_reverify_jsonl_is_parseable_by_load_run_rows(tmp_path):
    fresh = [{"topology_class": "current_mirror_simple_nmos", "pdk": "sky130A", "ok": True,
              "wall_time_s": None, "error": None, "verdicts": {"cm_iout": "REFUTED"},
              "verdict_notes": {"cm_iout": "trend - != predicted +"}, "canonical": {}}]
    path = write_reverify_jsonl(fresh, str(tmp_path))
    result = load_run_rows(path)
    assert result.errors == []
    assert result.rows[0].verdicts == {"cm_iout": "REFUTED"}


def test_refeed_ordering_lets_the_fresh_jsonl_override_a_stale_verdict(tmp_path):
    """The exact mechanic refeed_project_laws relies on: appending the reverify JSONL LAST means
    its (fresher) per-pdk verdict wins in build_law_records' grouping, so a genuinely drifted
    member changes the recomputed law status."""
    stale_path = tmp_path / "stale.jsonl"
    with stale_path.open("w") as f:
        for pdk in PDKS:
            f.write(json.dumps({"topology_class": "current_mirror_simple_nmos", "pdk": pdk, "ok": True,
                                 "verdicts": {"cm_iout": "VERIFIED"}, "verdict_notes": {},
                                 "canonical": {}}) + "\n")
    fresh_path = write_reverify_jsonl(
        [{"topology_class": "current_mirror_simple_nmos", "pdk": "sky130A", "ok": True,
          "wall_time_s": None, "error": None, "verdicts": {"cm_iout": "REFUTED"},
          "verdict_notes": {}, "canonical": {}}],
        str(tmp_path),
    )
    load_results = {str(stale_path): load_run_rows(stale_path), fresh_path: load_run_rows(fresh_path)}
    records, gap_report = build_law_records(load_results)
    assert gap_report["unknown_claim_ids"] == []   # cm_iout is a real seed claim id
    rec = next(r for r in records if r.topology_class == "current_mirror_simple_nmos" and r.knob == "Vout")
    assert rec.member_summary["sky130A"]["verdict"] == "REFUTED"     # fresh wins over stale "VERIFIED"
    assert rec.member_summary["gf180mcuD"]["verdict"] == "VERIFIED"  # untouched PDKs keep the stale row
    assert rec.reason == "divergent"   # sky130A now disagrees with the other 2 -> law status changes


# ── Non-negotiable pins: oracle.py byte-unchanged, ClaimCard schema frozen ──


def test_oracle_py_is_byte_unchanged():
    oracle_path = REPO_ROOT / "src" / "openclaw_brain" / "knowledge" / "executable" / "oracle.py"
    digest = hashlib.sha256(oracle_path.read_bytes()).hexdigest()
    assert digest == "eebdbef495074b3cb0c728bb57435f72d4aab8c34ece4bb35962caa0ea9b1498", (
        "oracle.py changed — E4a must reuse the oracle as-is (spec non-negotiable)"
    )


def test_claim_card_schema_is_frozen():
    assert set(ClaimCard.model_fields.keys()) == {
        "id", "topology_class", "mechanism", "conditions", "grounds",
        "verdict", "verdict_note", "engine", "basis", "scope", "dominant_risk_untested",
    }


# ── CLI wiring smoke test ──


def test_cli_reverify_help_works():
    from click.testing import CliRunner

    from openclaw_brain.cli import main

    result = CliRunner().invoke(main, ["reverify", "--help"])
    assert result.exit_code == 0
    assert "--sample" in result.output
    assert "--apply" in result.output
    assert "--corpus" in result.output
