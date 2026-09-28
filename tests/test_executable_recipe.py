"""Tests for ① recipe-authoring (knowledge/executable/recipe.py).

All offline (no LLM, no docker): the template-registry capability read, the deterministic
executable-enforcement (snap template_ref, drop+record unexecutable sweeps), the raw-fallback
normalization (inject R1 conditions, nest flat claim-cards), and both author() paths with a
mocked model — mirroring the answerer.py test idiom.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from openclaw_brain.config import ResilienceConfig
from openclaw_brain.knowledge.executable.models import VerificationRecipe
from openclaw_brain.knowledge.executable.recipe import (
    UnknownTopologyClass,
    author_recipe,
    capability_for,
    enforce_executable,
    _build_human_prompt,
    _clamp_sweep_points,
    _materialize_sweep_points,
    _normalize_recipe_payload,
    _sweep_unexecutable_reasons,
)
from openclaw_brain.knowledge.executable.templates import (
    OTA_KNOB_ELEMENTS,
    OTA_METRIC_MEAS,
    _OTA_BODY,
    render_miller_ota_ac,
)

OTA = "miller_ota_2stage_nmos_in"


# ── capability registry ──


def test_capability_for_reads_registry():
    cap = capability_for(OTA)
    assert cap.template_ref == "miller_ota_ac"
    # S3-inc2a §2: VDD/IREFV (headroom knobs) + vout_swing_v/icmr_lo_v/icmr_hi_v (swing/ICMR metrics)
    # + the "dc" analysis, added alongside the pre-existing Cc/CL + gbw/pm/av0 + "ac" surface.
    assert cap.knobs == frozenset({"Cc", "CL", "VDD", "IREFV"})
    assert cap.metrics == frozenset({"gbw_hz", "pm_deg", "av0_db",
                                     "vout_swing_v", "icmr_lo_v", "icmr_hi_v"})
    assert cap.analyses == frozenset({"ac", "dc"})
    assert cap.default_vdd == 1.8
    assert cap.knob_ranges == {"VDD": (1.4, 2.2), "IREFV": (2e-6, 4e-5)}


def test_capability_for_unknown_raises():
    with pytest.raises(UnknownTopologyClass):
        capability_for("nonexistent_topology")


def test_registry_knobs_metrics_are_actually_executable():
    """Drift guard: every capability the registry advertises must be runnable on the real template.
    Each knob's element must be a device the netlist body defines; each metric must render
    (the meas map keys it) without a KeyError. The S3-inc2a swing/ICMR metrics don't route through
    OTA_METRIC_MEAS at all (a dedicated device-query control block instead) — checked separately by
    just confirming they render without raising and emit the expected control-script shape."""
    cap = capability_for(OTA)
    body_elements = {line.split()[0] for line in _OTA_BODY.splitlines() if line.strip()}
    for knob in cap.knobs:
        element = OTA_KNOB_ELEMENTS[knob]
        assert element in body_elements, f"knob {knob} -> {element} absent from _OTA_BODY"
    for metric in cap.metrics:
        if metric in ("vout_swing_v", "icmr_lo_v", "icmr_hi_v"):
            deck = render_miller_ota_ac(knob="VDD", metric=metric)   # must not KeyError/raise
            assert "@m.xm" in deck and "[vdsat]" in deck
            continue
        assert metric in OTA_METRIC_MEAS
        deck = render_miller_ota_ac(knob="Cc", metric=metric)   # must not KeyError
        assert OTA_METRIC_MEAS[metric].splitlines()[0] in deck


# ── executable enforcement ──


def _recipe(sweeps, *, topo=OTA, template_ref="miller_ota_ac"):
    return VerificationRecipe(
        topology_class=topo,
        build={"method": "template", "template_ref": template_ref},
        conditions={"corner": "tt", "temp_c": 27.0, "vdd": 1.8},
        sweeps=sweeps,
    )


def test_enforce_snaps_template_ref_and_class():
    # model proposed the wrong class and a hallucinated template_ref
    r = _recipe([], topo="WRONG_CLASS", template_ref="sky130_fd_sc_hd_ota")
    out = enforce_executable(r, capability_for(OTA))
    assert out.topology_class == OTA
    assert out.build["template_ref"] == "miller_ota_ac"
    assert out.build["method"] == "template"


def test_enforce_drops_unexecutable_sweep_and_records_it():
    good = {"analysis": "ac", "knob": "Cc", "measure": ["gbw_hz", "pm_deg"]}
    bad_knob = {"analysis": "ac", "knob": "Vbias", "measure": ["gbw_hz"]}
    bad_metric = {"analysis": "ac", "knob": "CL", "measure": ["thd_db"]}
    out = enforce_executable(_recipe([good, bad_knob, bad_metric]), capability_for(OTA))
    assert out.sweeps == [good]                                  # only the executable one kept
    dropped = out.build["unexecutable_sweeps"]
    assert len(dropped) == 2                                     # recorded, not silent
    dropped_sweeps = [d["sweep"] for d in dropped]
    assert bad_knob in dropped_sweeps and bad_metric in dropped_sweeps


def test_sweep_reasons_pinpoint_the_violation():
    cap = capability_for(OTA)
    assert _sweep_unexecutable_reasons({"analysis": "ac", "knob": "Cc", "measure": ["gbw_hz"]}, cap) == []
    reasons = _sweep_unexecutable_reasons({"analysis": "tran", "knob": "Cc", "measure": ["gbw_hz"]}, cap)
    assert any("analysis" in r for r in reasons)


# ── sizing sanitation (DEFECT 1 — executor.py::_resolve_sizing/_gmid_target crash on a loose string) ──


def test_enforce_sanitizes_string_sizing():
    r = _recipe([])
    r.sizing = "use template defaults"           # loose LLM output — not a dict at all
    out = enforce_executable(r, capability_for(OTA))
    assert out.sizing == {}
    assert out.build["sizing_dropped"] == {"sizing": "use template defaults"}


def test_enforce_sanitizes_string_seed():
    r = _recipe([])
    r.sizing = {"method": "gmid_lookup", "seed": "use template defaults", "targets": {"gm_id": 12.0}}
    out = enforce_executable(r, capability_for(OTA))
    assert out.sizing["method"] == "gmid_lookup"          # method left as-is
    assert "seed" not in out.sizing                       # dropped, not silently coerced
    assert out.sizing["targets"] == {"gm_id": 12.0}        # untouched
    assert out.build["sizing_dropped"] == {"seed": "use template defaults"}


def test_enforce_sanitizes_string_targets():
    r = _recipe([])
    r.sizing = {"method": "gmid_lookup", "seed": {"Cc": "200f"}, "targets": "high overdrive"}
    out = enforce_executable(r, capability_for(OTA))
    assert out.sizing["seed"] == {"Cc": "200f"}
    assert "targets" not in out.sizing
    assert out.build["sizing_dropped"] == {"targets": "high overdrive"}


def test_enforce_leaves_well_formed_sizing_untouched():
    r = _recipe([])
    r.sizing = {"method": "gmid_lookup", "seed": {"Cc": "200f", "CL": "5p"}, "targets": {"gm_id": 14.0}}
    out = enforce_executable(r, capability_for(OTA))
    assert out.sizing == {
        "method": "gmid_lookup", "seed": {"Cc": "200f", "CL": "5p"}, "targets": {"gm_id": 14.0},
    }
    assert "sizing_dropped" not in out.build


def test_enforce_sizing_seed_scalars_are_not_enforced():
    # Only the CONTAINER type is enforced — scalar values *inside* seed are left alone (the executor
    # str()s them).
    r = _recipe([])
    r.sizing = {"seed": {"Cc": 200e-15, "CL": "5p", "M1": 4}}
    out = enforce_executable(r, capability_for(OTA))
    assert out.sizing["seed"] == {"Cc": 200e-15, "CL": "5p", "M1": 4}
    assert "sizing_dropped" not in out.build


# ── focus probe (DEFECT 2 — the planning target cell reaching the prompt) ──


def test_build_human_prompt_without_focus_is_byte_identical():
    cap = capability_for(OTA)
    with_none = _build_human_prompt(cap, "source text")
    explicit_none = _build_human_prompt(cap, "source text", None)
    assert with_none == explicit_none
    payload = json.loads(with_none)
    assert "focus_probe" not in payload
    assert "focus_instruction" not in payload


def test_build_human_prompt_embeds_focus_probe_and_instruction():
    cap = capability_for(OTA)
    focus = {"metric": "gbw_hz", "knob": "Cc", "kind_hint": None}
    prompt = _build_human_prompt(cap, "", focus)
    payload = json.loads(prompt)
    assert payload["focus_probe"] == {"metric": "gbw_hz", "knob": "Cc", "kind_hint": None}
    assert "invariance" in payload["focus_instruction"]
    assert "never force" in payload["focus_instruction"]


# ── raw-fallback normalization ──


def test_normalize_injects_mandatory_conditions():
    cap = capability_for(OTA)
    payload = _normalize_recipe_payload({"sweeps": []}, cap)   # no conditions at all
    recipe = VerificationRecipe(**payload)                     # R1 would reject if absent
    assert recipe.conditions.corner == "tt"
    assert recipe.conditions.temp_c == 27.0
    assert recipe.conditions.vdd == 1.8


def test_normalize_nests_a_flat_claim_card():
    cap = capability_for(OTA)
    raw = {
        "claim_cards": [
            {"id": "cc_gbw", "knob": "Cc", "metric": "gbw_hz", "series_ref": "cc",
             "quant": {"kind": "direction", "sign": "-"}, "narrative": "Miller cap lowers GBW"}
        ]
    }
    recipe = VerificationRecipe(**_normalize_recipe_payload(raw, cap))
    card = recipe.claim_cards[0]
    assert card.mechanism.knob == "Cc"
    assert card.mechanism.metric == "gbw_hz"
    assert card.mechanism.quant.kind == "direction"
    assert card.topology_class == OTA            # class threaded onto the card
    assert card.conditions.corner == "tt"        # inherited recipe conditions


def test_normalize_collapses_pvt_grid_to_nominal():
    cap = capability_for(OTA)
    raw = {"conditions": {"corner": ["ss", "tt", "ff"], "temp_c": [-40, 27, 125], "vdd": [1.62, 1.8, 1.98]}}
    recipe = VerificationRecipe(**_normalize_recipe_payload(raw, cap))
    assert recipe.conditions.corner == "ss"      # first element of the grid
    assert recipe.conditions.temp_c == -40
    assert recipe.conditions.vdd == 1.62


# ── author() — mocked model, both paths ──


class _FakeStructuredResponder:
    def __init__(self, parent):
        self._parent = parent

    async def ainvoke(self, messages):
        self._parent.messages = messages
        return self._parent.result


class _FakePrimaryModel:
    """Mimics a ChatModel: structured-output path returns `result`; raw path (ainvoke) returns
    a response with `.content`. `structured_error` forces the raw fallback."""

    def __init__(self, *, result=None, raw_content=None, structured_error: Exception | None = None):
        self.result = result
        self.raw_content = raw_content
        self.structured_error = structured_error
        self.model = "fake-recipe-model"
        self.structured_calls = 0

    def with_structured_output(self, schema):
        self.structured_calls += 1
        if self.structured_error:
            raise self.structured_error
        return _FakeStructuredResponder(self)

    async def ainvoke(self, messages, **kwargs):
        return SimpleNamespace(content=self.raw_content)


@pytest.mark.asyncio
async def test_author_structured_path_enforces():
    recipe = _recipe(
        [{"analysis": "ac", "knob": "Cc", "measure": ["gbw_hz"]},
         {"analysis": "ac", "knob": "Vbias", "measure": ["gbw_hz"]}],   # one unexecutable
        topo="WRONG", template_ref="bogus",
    )
    model = _FakePrimaryModel(result=recipe)
    out = await author_recipe(
        topology_class=OTA, source_text="two-stage Miller OTA",
        model_chain=[model], resilience_config=SimpleNamespace(request_timeout_s=0),
    )
    assert model.structured_calls == 1
    assert out.topology_class == OTA                       # snapped
    assert out.build["template_ref"] == "miller_ota_ac"    # snapped
    assert len(out.sweeps) == 1                            # Vbias dropped
    assert "unexecutable_sweeps" in out.build


@pytest.mark.asyncio
async def test_author_structured_path_threads_focus_into_prompt():
    recipe = _recipe([{"analysis": "ac", "knob": "Cc", "measure": ["gbw_hz"]}])
    model = _FakePrimaryModel(result=recipe)
    focus = {"metric": "gbw_hz", "knob": "Cc", "kind_hint": None}
    await author_recipe(
        topology_class=OTA, source_text="", model_chain=[model],
        resilience_config=SimpleNamespace(request_timeout_s=0), focus=focus,
    )
    human_payload = json.loads(model.messages[1].content)
    assert human_payload["focus_probe"] == focus
    assert "focus_instruction" in human_payload


@pytest.mark.asyncio
async def test_author_raw_fallback_path_normalizes_and_enforces():
    raw = {
        "topology_class": "whatever",
        "build": {"template_ref": "hallucinated"},
        "sweeps": [
            {"analysis": "ac", "knob": "Cc", "measure": ["gbw_hz"]},
            {"analysis": "noise", "knob": "Cc", "measure": ["vn"]},   # unexecutable
        ],
        "claim_cards": [
            {"id": "c0", "knob": "Cc", "metric": "gbw_hz", "series_ref": "cc",
             "quant": {"kind": "elasticity", "band": [-1.1, -0.7]}, "narrative": "GBW≈gm1/2πCc"}
        ],
        # no conditions -> normalizer must inject them
    }
    model = _FakePrimaryModel(
        raw_content="```json\n" + json.dumps(raw) + "\n```",     # fenced, extract_json must strip
        structured_error=RuntimeError("model has no structured output"),
    )
    out = await author_recipe(
        topology_class=OTA,
        model_chain=[model],
        resilience_config=ResilienceConfig(max_retries=0, request_timeout_s=0),
    )
    assert out.topology_class == OTA
    assert out.build["template_ref"] == "miller_ota_ac"
    assert out.conditions.corner == "tt"                  # injected
    assert len(out.sweeps) == 1                           # noise sweep dropped
    assert out.claim_cards[0].mechanism.quant.kind == "elasticity"
    assert out.claim_cards[0].topology_class == OTA


# ── _materialize_sweep_points (count+range -> explicit value list; the live TypeError fix) ──


def test_materialize_points_count_plus_log_range():
    sw = {"analysis": "ac", "knob": "CL", "points": 5,
          "range": [2e-13, 5e-12], "log": True, "measure": ["av0_db"]}
    out = _materialize_sweep_points(sw)
    assert isinstance(out["points"], list) and len(out["points"]) == 5
    vals = [float(p) for p in out["points"]]
    assert abs(vals[0] - 2e-13) / 2e-13 < 1e-4 and abs(vals[-1] - 5e-12) / 5e-12 < 1e-4
    # log spacing: constant ratio between consecutive points
    ratios = [vals[i + 1] / vals[i] for i in range(4)]
    assert max(ratios) / min(ratios) < 1.001
    # original dict untouched (copy semantics)
    assert sw["points"] == 5


def test_materialize_points_linear_range_without_log_flag():
    out = _materialize_sweep_points({"points": 3, "range": [1.0, 2.0]})
    assert [float(p) for p in out["points"]] == [1.0, 1.5, 2.0]


def test_materialize_points_count_without_range_drops_to_renderer_default():
    out = _materialize_sweep_points({"points": 7, "measure": ["gbw_hz"]})
    assert "points" not in out
    assert out["points_dropped"] == 7


def test_materialize_points_list_passthrough_and_none():
    sw = {"points": ["1p", "2p"], "measure": ["pm_deg"]}
    assert _materialize_sweep_points(sw) is sw
    sw2 = {"measure": ["pm_deg"]}
    assert _materialize_sweep_points(sw2) is sw2


# ── knob-range clamp (S3-inc2a spec §4: enforce_executable, next to _materialize_sweep_points) ──


def test_clamp_drops_out_of_range_points_and_records_them():
    cap = capability_for(OTA)   # VDD knob_ranges == (1.4, 2.2) — templates.py registry
    sw = {"analysis": "dc", "knob": "VDD", "points": ["0.5", "1.8", "2.0", "5.0"],
          "measure": ["vout_swing_v"]}
    out, record = _clamp_sweep_points(sw, cap)
    assert out["points"] == ["1.8", "2.0"]              # in-range points survive, in order
    assert record == {"knob": "VDD", "range": [1.4, 2.2], "dropped": ["0.5", "5.0"]}
    assert sw["points"] == ["0.5", "1.8", "2.0", "5.0"]  # original untouched (copy semantics)


def test_clamp_accepts_unit_suffixed_points_for_a_ranged_knob():
    cap = capability_for(OTA)   # IREFV knob_ranges == (2e-6, 4e-5)
    sw = {"analysis": "dc", "knob": "IREFV", "points": ["1u", "10u", "40u", "100u"],
          "measure": ["icmr_lo_v"]}
    out, record = _clamp_sweep_points(sw, cap)
    assert out["points"] == ["10u", "40u"]
    assert record["dropped"] == ["1u", "100u"]


def test_clamp_all_points_out_of_range_leaves_an_empty_points_list():
    cap = capability_for(OTA)
    sw = {"analysis": "dc", "knob": "VDD", "points": ["0.1", "9.9"], "measure": ["vout_swing_v"]}
    out, record = _clamp_sweep_points(sw, cap)
    assert out["points"] == []
    assert record["dropped"] == ["0.1", "9.9"]


def test_clamp_no_declared_range_is_additive_noop():
    # Cc has no knob_ranges entry — untouched regardless of how extreme the authored value is.
    cap = capability_for(OTA)
    sw = {"analysis": "ac", "knob": "Cc", "points": ["1f", "999p"], "measure": ["gbw_hz"]}
    out, record = _clamp_sweep_points(sw, cap)
    assert out is sw          # not even copied — a true no-op
    assert record is None


def test_clamp_no_explicit_points_is_a_noop():
    # points=None (renderer default applies) -- nothing to clamp, even for a ranged knob.
    cap = capability_for(OTA)
    sw = {"analysis": "dc", "knob": "VDD", "measure": ["vout_swing_v"]}
    out, record = _clamp_sweep_points(sw, cap)
    assert out is sw
    assert record is None


def test_enforce_clamp_partial_keeps_sweep_executable_and_records_clamp():
    r = _recipe([{"analysis": "dc", "knob": "VDD", "points": ["0.5", "1.8", "2.0", "5.0"],
                  "measure": ["vout_swing_v"]}])
    out = enforce_executable(r, capability_for(OTA))
    assert out.sweeps == [{"analysis": "dc", "knob": "VDD", "points": ["1.8", "2.0"],
                           "measure": ["vout_swing_v"]}]
    assert out.build["points_clamped"] == [
        {"knob": "VDD", "range": [1.4, 2.2], "dropped": ["0.5", "5.0"]},
    ]
    assert "unexecutable_sweeps" not in out.build


def test_enforce_clamp_all_points_makes_sweep_unexecutable_with_reason():
    """The all-clamped case must never become a silent empty sweep (which would fall through to
    the renderer's own default, hiding the clamp entirely) — it becomes unexecutable, with the
    clamp reason on record."""
    r = _recipe([{"analysis": "dc", "knob": "VDD", "points": ["0.1", "9.9"],
                  "measure": ["vout_swing_v"]}])
    out = enforce_executable(r, capability_for(OTA))
    assert out.sweeps == []                             # never silently kept
    assert out.build["points_clamped"] == [
        {"knob": "VDD", "range": [1.4, 2.2], "dropped": ["0.1", "9.9"]},
    ]
    dropped = out.build["unexecutable_sweeps"]
    assert len(dropped) == 1
    assert "clamped out of range" in dropped[0]["reasons"][0]
    assert "VDD" in dropped[0]["reasons"][0]


def test_enforce_no_declared_range_never_clamps_existing_knobs():
    """Additive guarantee: a template/knob with no declared range keeps its pre-existing enforcement
    behavior byte-for-byte — no build.points_clamped key appears at all."""
    r = _recipe([{"analysis": "ac", "knob": "Cc", "points": ["1f", "999p"], "measure": ["gbw_hz"]}])
    out = enforce_executable(r, capability_for(OTA))
    assert out.sweeps == [{"analysis": "ac", "knob": "Cc", "points": ["1f", "999p"],
                           "measure": ["gbw_hz"]}]
    assert "points_clamped" not in out.build


# ── raw-path chain walk on JSON-extraction failure (the live 561-chars-of-prose triage) ──


@pytest.mark.asyncio
async def test_author_recipe_raw_path_advances_chain_on_garbage_json(monkeypatch):
    """A model can return API-success whose content is NOT JSON — that must advance the
    chain to the next model, not raise (observed live: deepseek returned prose once)."""
    import openclaw_brain.knowledge.executable.recipe as recipe_mod

    class _NoStructured:
        def with_structured_output(self, schema):
            raise RuntimeError("no structured output")

    garbage_model, good_model = _NoStructured(), _NoStructured()
    good_payload = {
        "topology_class": "miller_ota_2stage_nmos_in",
        "sweeps": [{"analysis": "ac", "knob": "CL", "points": ["1p", "2p"], "measure": ["av0_db"]}],
        "claim_cards": [{"id": "c1", "metric": "av0_db", "knob": "CL",
                         "quant": {"kind": "invariance", "cov_max": 0.05}}],
    }
    calls = []

    async def fake_invoke(chain, messages, cfg, auth_refresh=None):
        calls.append(chain[0])
        class R:
            content = "I think the recipe should..." if chain[0] is garbage_model \
                else __import__("json").dumps(good_payload)
        return R()

    monkeypatch.setattr(recipe_mod, "invoke_with_resilience", fake_invoke)
    recipe = await recipe_mod.author_recipe(
        topology_class="miller_ota_2stage_nmos_in", source_text="",
        model_chain=[garbage_model, good_model],
        resilience_config=ResilienceConfig(request_timeout_s=0),
    )
    assert calls == [garbage_model, good_model]
    assert recipe.claim_cards and recipe.claim_cards[0].mechanism.quant.kind == "invariance"


@pytest.mark.asyncio
async def test_author_recipe_raw_path_raises_after_whole_chain_fails(monkeypatch):
    import openclaw_brain.knowledge.executable.recipe as recipe_mod

    class _NoStructured:
        def with_structured_output(self, schema):
            raise RuntimeError("no structured output")

    async def fake_invoke(chain, messages, cfg, auth_refresh=None):
        class R:
            content = "not json at all"
        return R()

    monkeypatch.setattr(recipe_mod, "invoke_with_resilience", fake_invoke)
    with pytest.raises(Exception):
        await recipe_mod.author_recipe(
            topology_class="miller_ota_2stage_nmos_in", source_text="",
            model_chain=[_NoStructured(), _NoStructured()],
            resilience_config=ResilienceConfig(request_timeout_s=0),
        )


# ── conditions coercion + role_map sanitation (live growth-run failures, 2026-07-04) ──


def test_enforce_coerces_digital_conditions_to_analog_nominal():
    """Live failure: the structured-output path (which bypasses _normalize_conditions) authored
    DigitalUnits conditions for the Miller OTA — corner lost -> dead deck -> FLAGGED empty series."""
    from openclaw_brain.knowledge.executable.conditions import AnalogPVT, DigitalUnits

    cap = capability_for("miller_ota_2stage_nmos_in")
    recipe = VerificationRecipe(
        topology_class="miller_ota_2stage_nmos_in",
        conditions=DigitalUnits(clock_period_ns=5.0, bit_width=8, stimulus="sine"),
        sweeps=[{"analysis": "ac", "knob": "CL", "points": ["1p", "2p"], "measure": ["av0_db"]}],
        claim_cards=[{"id": "c1", "topology_class": "miller_ota_2stage_nmos_in",
                      "conditions": {"kind": "digital", "clock_period_ns": 5.0, "bit_width": 8,
                                     "stimulus": "sine"},
                      "mechanism": {"knob": "CL", "metric": "av0_db", "series_ref": "s",
                                    "quant": {"kind": "invariance", "cov_max": 0.01}}}],
    )
    out = enforce_executable(recipe, cap)
    assert isinstance(out.conditions, AnalogPVT)
    assert out.conditions.corner == "tt" and out.conditions.vdd == cap.default_vdd
    assert out.build["conditions_coerced"]["kind"] == "digital"
    assert all(isinstance(c.conditions, AnalogPVT) for c in out.claim_cards)


def test_enforce_analog_conditions_pass_through_untouched():
    from openclaw_brain.knowledge.executable.conditions import AnalogPVT

    cap = capability_for("miller_ota_2stage_nmos_in")
    cond = AnalogPVT(corner="ss", temp_c=85.0, vdd=1.62)
    recipe = VerificationRecipe(topology_class="miller_ota_2stage_nmos_in", conditions=cond,
                                sweeps=[], claim_cards=[])
    out = enforce_executable(recipe, cap)
    assert out.conditions is cond
    assert "conditions_coerced" not in out.build


def test_enforce_drops_prose_role_map_to_empty_dict():
    """Live failure: structured path emitted role_map as prose -> Specimen dict_type validation
    error at the simulate stage."""
    cap = capability_for("miller_ota_2stage_nmos_in")
    recipe = VerificationRecipe(
        topology_class="miller_ota_2stage_nmos_in",
        build={"role_map": "M1,M2: diff pair; M5: tail current source"},
        conditions={"corner": "tt", "temp_c": 27.0, "vdd": 1.8},
        sweeps=[], claim_cards=[],
    )
    out = enforce_executable(recipe, cap)
    assert out.build["role_map"] == {}
    assert "diff pair" in out.build["role_map_dropped"]


def test_enforce_keeps_dict_role_map():
    cap = capability_for("miller_ota_2stage_nmos_in")
    recipe = VerificationRecipe(
        topology_class="miller_ota_2stage_nmos_in",
        build={"role_map": {"M1": "diff pair"}},
        conditions={"corner": "tt", "temp_c": 27.0, "vdd": 1.8},
        sweeps=[], claim_cards=[],
    )
    out = enforce_executable(recipe, cap)
    assert out.build["role_map"] == {"M1": "diff pair"}
    assert "role_map_dropped" not in out.build
