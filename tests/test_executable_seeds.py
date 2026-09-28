"""Validated seed recipes (knowledge/executable/seeds.py) — the SSOT the project-executable CLI
runs. Guard: every seed's sweeps + claims reference only knobs/metrics the template can actually
execute (so a real run can't fail on an unexecutable seed). No sim."""

from openclaw_brain.knowledge.executable.recipe import capability_for
from openclaw_brain.knowledge.executable.seeds import seed_recipes


def test_seed_recipes_are_executable():
    recipes = seed_recipes()
    assert {r.topology_class for r in recipes} == {
        "miller_ota_2stage_nmos_in", "current_mirror_simple_nmos",
        "common_source_active_load_nmos", "common_gate_nmos", "source_follower_nmos",
        "diff_pair_resistive_nmos", "cascode_current_mirror_nmos", "ota_5t_nmos_in",
        "telescopic_cascode_ota_nmos_in", "folded_cascode_ota_nmos_in",
        "regulated_cascode_nmos", "comparator_continuous_nmos", "cds_switched_cap_nmos",
        "single_slope_ramp_generator", "column_pga_inverting_nmos",
        "ptat_ctat_core_bjt"}   # S3-inc2a W2 — the first new template since the registry froze at 18
    for r in recipes:
        cap = capability_for(r.topology_class)          # raises UnknownTopologyClass if unregistered
        assert r.build["template_ref"] == cap.template_ref
        for sw in r.sweeps:
            assert sw["knob"] in cap.knobs, (r.topology_class, sw["knob"])
            for metric in sw["measure"]:
                assert metric in cap.metrics, (r.topology_class, metric)
        for c in r.claim_cards:
            assert c.mechanism.knob in cap.knobs
            assert c.mechanism.metric in cap.metrics
            assert c.conditions.corner == "tt"          # R1 conditions present


def test_conditions_objects_are_never_shared_by_reference():
    """seeds.py:15's former defect: every analog seed's every claim card (and its owning recipe)
    shared ONE mutable AnalogPVT object (`_TT`) by Python reference — an in-place mutation anywhere
    would silently corrupt the recorded PVT condition of every other card/recipe. Fixed: every use
    gets its own independent copy via the `_tt()` factory. Zero behavior change for current values —
    every copy must still be content-equal to the nominal TT/27C/1.8V point."""
    recipes = seed_recipes()
    conditions_ids = set()
    for r in recipes:
        conditions_ids.add(id(r.conditions))
        for c in r.claim_cards:
            conditions_ids.add(id(c.conditions))
    total_conditions = len(recipes) + sum(len(r.claim_cards) for r in recipes)
    assert total_conditions > 1          # otherwise this test proves nothing
    assert len(conditions_ids) == total_conditions   # every single one is a DISTINCT object

    for r in recipes:
        assert (r.conditions.corner, r.conditions.temp_c, r.conditions.vdd) == ("tt", 27.0, 1.8)
        for c in r.claim_cards:
            assert (c.conditions.corner, c.conditions.temp_c, c.conditions.vdd) == ("tt", 27.0, 1.8)


def test_mutating_one_claim_cards_conditions_never_leaks_into_another():
    """The concrete adversarial reproduction the audit performed against the pre-fix code: mutate
    ONE claim card's `.conditions` in place and confirm no other card or recipe anywhere in the
    standing curriculum picks up the mutation (before the fix, all 23/23 cards across all 16/16
    topology classes did)."""
    recipes = seed_recipes()
    all_cards = [c for r in recipes for c in r.claim_cards]
    assert len(all_cards) > 1

    target = all_cards[0]
    target.conditions.vdd = 2.0   # in-place mutation — exactly the hypothesized trigger

    assert target.conditions.vdd == 2.0   # the mutation DID apply to its own object
    for c in all_cards[1:]:
        assert c.conditions.vdd == 1.8    # ...but nothing else picked it up
    for r in recipes:
        assert r.conditions.vdd == 1.8    # every recipe's own top-level conditions is untouched too
