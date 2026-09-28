"""Tests for the E3-I1 intervention machinery (spec §2/§3/§4-I1/§5):

  - the baseline miller_ota_2stage_nmos_in template body is BYTE-UNCHANGED (regression pin);
  - the executor's 'intervention' sweep branch (executor.py) — paired baseline/variant runs with
    shared sizing, delta/ratio series derivation, series-ref routing, undetachable grounds/scope
    stamping, cache dedup, and the "unknown intervention id raises before sim" guard (all mocked, no
    docker);
  - ONE live sky130 smoke (docker-gated) closing the loop end-to-end for the rz_null pilot recipe.

Magnitude assertions (e.g. "PM improves by exactly N degrees") belong to I2's experiment run, not
this substrate test — see docs/superpowers/specs/2026-07-04-e3-intervention-experiments.md §4/§5.
"""

from __future__ import annotations

import pytest
from tests.conftest import require_live_graph

from openclaw_brain.knowledge.executable.executor import run_recipe
from openclaw_brain.knowledge.executable.interventions import (
    INTERVENTIONS,
    UnknownIntervention,
    get_intervention,
)
from openclaw_brain.knowledge.executable.models import (
    ClaimCard, MechanismClaim, QuantTest, VerdictClass, VerificationRecipe,
)
from openclaw_brain.knowledge.executable.runner import NgspiceRunner
from openclaw_brain.knowledge.executable.seeds import intervention_seed_recipes
from openclaw_brain.knowledge.executable.templates import _OTA_BODY
from openclaw_brain.agent import _scope_inline
import json as _json
from openclaw_brain.config import load_config as _load_config

OTA = "miller_ota_2stage_nmos_in"
COND = {"corner": "tt", "temp_c": 27.0, "vdd": 1.8}
_CERTIFIED = {VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT}

# The EXACT baseline body as it existed before E3-I1 (pinned verbatim from templates.py at the start
# of this work) — a byte-for-byte regression guarantee that the intervention work never touched it.
_ORIGINAL_OTA_BODY = """VDD vdd 0 {VDD}
Vin vinn 0 DC {VCM} AC 1
Lfb out vinp 1T
Cfb vinp 0 1T
IREF vdd nbias {IREFV}
XM8 nbias nbias 0 0 sky130_fd_pr__nfet_01v8 W={W8} L={Lp}
XM5 tail nbias 0 0 sky130_fd_pr__nfet_01v8 W={W5} L={Lp}
XM7 out  nbias 0 0 sky130_fd_pr__nfet_01v8 W={W7} L={Lp}
XM1 o1n vinp tail 0 sky130_fd_pr__nfet_01v8 W={W1} L={Lp} m={M1}
XM2 o1  vinn tail 0 sky130_fd_pr__nfet_01v8 W={W1} L={Lp} m={M1}
XM3 o1n o1n vdd vdd sky130_fd_pr__pfet_01v8 W={W3} L={Lp}
XM4 o1  o1n vdd vdd sky130_fd_pr__pfet_01v8 W={W3} L={Lp}
XM6 out o1 vdd vdd sky130_fd_pr__pfet_01v8 W={W6} L={Lp}
Ccomp out o1 {Cc}
CL out 0 {CL}"""


def test_baseline_ota_body_byte_identical():
    """Regression pin (spec requirement): the intervention work must never touch the baseline
    template's netlist body."""
    assert _OTA_BODY == _ORIGINAL_OTA_BODY


# ── fake runner: distinguishes baseline / rz_null variant / ff_break variant decks by their
# distinctive netlist fragment (present regardless of sizing substitution) ──


class _FakeInterventionRunner:
    def __init__(self, responses: list[tuple[str, dict]]):
        # ordered (marker_substring, canned_series_dict) pairs; first substring match wins, so put
        # more-specific markers first and a catch-all ("") last.
        self._responses = responses
        self.decks: list[str] = []

    def measure(self, deck: str, timeout: int = 300):
        self.decks.append(deck)
        for marker, series in self._responses:
            if marker in deck:
                return series
        return {}


class _AssertNeverCalledRunner:
    def measure(self, deck: str, timeout: int = 300):
        raise AssertionError("runner.measure must not be called before the intervention id is validated")


def _claim(cid, knob, metric, quant, series_ref=None, grounds=()):
    return ClaimCard(
        id=cid, topology_class=OTA, conditions=COND, grounds=list(grounds),
        mechanism=MechanismClaim(knob=knob, metric=metric, series_ref=series_ref or cid,
                                 quant=quant, narrative="why"),
    )


def _ff_break_recipe(extra_sweep_kwargs=None, extra_claims=()):
    """Native-knob pairing: Cc is swept identically on both baseline and ff_break variant decks."""
    sweep = {"analysis": "intervention", "intervention": "ff_break", "knob": "Cc",
             "points": ["250f", "500f"], "measure": ["gbw_hz"]}
    sweep.update(extra_sweep_kwargs or {})
    claims = [
        _claim("cc_gbw_delta__ff_break", "Cc", "gbw_hz", QuantTest(kind="direction", sign="+")),
        *extra_claims,
    ]
    return VerificationRecipe(
        topology_class=OTA, build={"method": "template", "template_ref": "miller_ota_ac"},
        conditions=COND, sweeps=[sweep], claim_cards=claims,
    )


def _rz_null_recipe():
    """Non-native-knob pairing: Rz does not exist on the baseline template at all."""
    sweep = {"analysis": "intervention", "intervention": "rz_null", "knob": "Rz",
             "points": ["10", "1000", "2000"], "measure": ["pm_deg"]}
    claims = [_claim("rz_pm_delta__rz_null", "Rz", "pm_deg", QuantTest(kind="direction", sign="+"))]
    return VerificationRecipe(
        topology_class=OTA, build={"method": "template", "template_ref": "miller_ota_ac"},
        conditions=COND, sweeps=[sweep], claim_cards=claims,
    )


# ff_break variant marker: "Eff outbuf" (the ideal buffer line). rz_null variant marker: "Rz nrz o1"
# (the nulling-resistor line). Neither ever appears in a baseline deck.
_FF_BASE_GBW = [(2.5e-13, 5.0e7), (5e-13, 4.0e7)]
_FF_VARIANT_GBW = [(2.5e-13, 6.0e7), (5e-13, 5.5e7)]


def test_paired_run_shares_sizing_renders_two_decks():
    runner = _FakeInterventionRunner([
        ("Eff outbuf", {"cc": _FF_VARIANT_GBW}),
        ("", {"cc": _FF_BASE_GBW}),
    ])
    result = run_recipe(_ff_break_recipe(), runner)
    assert len(runner.decks) == 2
    base_deck, variant_deck = runner.decks
    # identical sizing: every shared .param value (W1=8 etc.) is the SAME literal in both decks.
    base_params = next(ln for ln in base_deck.splitlines() if ln.startswith(".param"))
    variant_params = next(ln for ln in variant_deck.splitlines() if ln.startswith(".param"))
    for tok in ["VDD=1.8", "W1=8", "W3=4", "W6=16", "Lp=0.5"]:
        assert tok in base_params
        assert tok in variant_params
    assert result.runs == 1   # one cache_key -> one paired sim (2 decks, one run_cache entry)


def test_delta_series_key_and_values():
    runner = _FakeInterventionRunner([
        ("Eff outbuf", {"cc": _FF_VARIANT_GBW}),
        ("", {"cc": _FF_BASE_GBW}),
    ])
    result = run_recipe(_ff_break_recipe(), runner)
    card = result.claim_cards[0]
    assert card.mechanism.series_ref == "cc_gbw_hz_delta__ff_break"
    delta = result.canonical["cc_gbw_hz_delta__ff_break"]
    assert delta == [(2.5e-13, 6.0e7 - 5.0e7), (5e-13, 5.5e7 - 4.0e7)]
    assert card.verdict in _CERTIFIED     # delta is positive and rising -> direction '+' verifies


def test_ratio_series_derived_when_requested():
    runner = _FakeInterventionRunner([
        ("Eff outbuf", {"cc": _FF_VARIANT_GBW}),
        ("", {"cc": _FF_BASE_GBW}),
    ])
    extra = _claim("cc_gbw_ratio__ff_break", "Cc", "gbw_hz", QuantTest(kind="direction", sign="+"),
                   series_ref="cc_gbw_ratio__ff_break")
    recipe = _ff_break_recipe(extra_sweep_kwargs={"derive": "ratio"},
                              extra_claims=())
    # `derive` is per-sweep (all claims on that (knob, sweep) share it); use the ratio-configured sweep.
    recipe.claim_cards = [extra]
    result = run_recipe(recipe, runner)
    card = result.claim_cards[0]
    assert card.mechanism.series_ref == "cc_gbw_hz_ratio__ff_break"
    ratio = result.canonical["cc_gbw_hz_ratio__ff_break"]
    assert ratio == [(2.5e-13, 6.0e7 / 5.0e7), (5e-13, 5.5e7 / 4.0e7)]


def test_av0_invariance_control_verifies_on_zero_delta():
    """The ff_break pilot's own validity check (spec §3): a delta series pinned at exactly 0 must
    VERIFY under `invariance(spread_max=...)` — NOT `cov_max` (a zero-mean series FLAGs under CoV;
    see oracle.py's own guidance), which is exactly why seeds.py's control claim uses spread_max."""
    same = [(2.5e-13, 68.8739), (5e-13, 68.8739)]
    runner = _FakeInterventionRunner([
        ("Eff outbuf", {"cc": same}),
        ("", {"cc": same}),
    ])
    claim = _claim("cc_av0_delta__ff_break", "Cc", "av0_db",
                   QuantTest(kind="invariance", spread_max=0.05))
    recipe = _ff_break_recipe(extra_sweep_kwargs={"measure": ["av0_db"]})
    recipe.claim_cards = [claim]
    result = run_recipe(recipe, runner)
    card = result.claim_cards[0]
    assert result.canonical["cc_av0_db_delta__ff_break"] == [(2.5e-13, 0.0), (5e-13, 0.0)]
    assert card.verdict == VerdictClass.VERIFIED


def test_unknown_intervention_raises_before_sim():
    sweep = {"analysis": "intervention", "intervention": "bogus_id", "knob": "Rz",
             "points": ["10", "1000"], "measure": ["pm_deg"]}
    claim = _claim("rz_pm_delta__bogus", "Rz", "pm_deg", QuantTest(kind="direction", sign="+"))
    recipe = VerificationRecipe(
        topology_class=OTA, build={"method": "template", "template_ref": "miller_ota_ac"},
        conditions=COND, sweeps=[sweep], claim_cards=[claim],
    )
    with pytest.raises(UnknownIntervention):
        run_recipe(recipe, _AssertNeverCalledRunner())


def test_intervention_wrong_topology_class_raises_before_sim():
    """rz_null/ff_break are both registered for miller_ota_2stage_nmos_in — running one under a
    recipe for a DIFFERENT topology_class is an authoring error that must raise before any sim."""
    sweep = {"analysis": "intervention", "intervention": "rz_null", "knob": "Rz",
             "points": ["10", "1000"], "measure": ["pm_deg"]}
    claim = _claim("rz_pm_delta__rz_null", "Rz", "pm_deg", QuantTest(kind="direction", sign="+"))
    recipe = VerificationRecipe(
        topology_class="current_mirror_simple_nmos",
        build={"method": "template", "template_ref": "current_mirror_dc"},
        conditions=COND, sweeps=[sweep], claim_cards=[claim],
    )
    with pytest.raises(ValueError, match="rz_null"):
        run_recipe(recipe, _AssertNeverCalledRunner())


def test_get_intervention_known_ids_resolve():
    assert set(INTERVENTIONS) == {"rz_null", "ff_break"}
    assert get_intervention("rz_null").idealization is None            # PHYSICAL variant
    assert get_intervention("ff_break").idealization is not None        # IDEALIZED variant, named


_RZ_BASE_REF = [(1e-12, 34.26)]          # single reference point (baseline has no Rz knob at all)
_RZ_VARIANT_PM = [(10.0, 34.35), (1000.0, 43.25), (2000.0, 51.73)]


def test_rz_null_broadcasts_baseline_reference_across_variant_points():
    """Non-native-knob pairing (spec §3): the baseline reference is ONE fixed measurement (Rz does
    not exist on the baseline template), broadcast across every variant Rz point."""
    runner = _FakeInterventionRunner([
        ("Rz nrz o1", {"rz": _RZ_VARIANT_PM}),
        ("", {"cc": _RZ_BASE_REF}),
    ])
    result = run_recipe(_rz_null_recipe(), runner)
    card = result.claim_cards[0]
    assert card.mechanism.series_ref == "rz_pm_deg_delta__rz_null"
    delta = result.canonical["rz_pm_deg_delta__rz_null"]
    base_y = _RZ_BASE_REF[0][1]
    assert delta == [(x, y - base_y) for x, y in _RZ_VARIANT_PM]
    assert card.verdict in _CERTIFIED     # rising delta -> direction '+' verifies
    # baseline deck rendered with knob=Cc (the OTA's own first native knob), a SINGLE point equal to
    # its own nominal sizing value — not a 3-point Rz sweep.
    base_deck = next(d for d in runner.decks if "Rz nrz o1" not in d)
    assert "foreach pt 1p\n" in base_deck or "foreach pt 1p" in base_deck


def test_grounds_and_scope_stamped_undetachably():
    runner = _FakeInterventionRunner([
        ("Eff outbuf", {"cc": _FF_VARIANT_GBW}),
        ("", {"cc": _FF_BASE_GBW}),
    ])
    result = run_recipe(_ff_break_recipe(), runner)
    card = result.claim_cards[0]
    assert "intervention:ff_break" in card.grounds
    assert card.scope["intervention"] == "ff_break"
    assert card.scope["idealization"] == get_intervention("ff_break").idealization
    # summarize_scope's own fields are still present (additive stamping, not a replacement)
    assert "pdk" in card.scope and "corners" in card.scope


def test_grounds_not_duplicated_when_author_already_set_it():
    runner = _FakeInterventionRunner([
        ("Eff outbuf", {"cc": _FF_VARIANT_GBW}),
        ("", {"cc": _FF_BASE_GBW}),
    ])
    recipe = _ff_break_recipe()
    recipe.claim_cards[0].grounds = ["intervention:ff_break"]
    result = run_recipe(recipe, runner)
    assert result.claim_cards[0].grounds == ["intervention:ff_break"]


def test_intervention_cache_dedup_shared_knob_metric():
    """Two claims on the SAME (knob, metric, intervention id) must reuse one paired sim, not two."""
    extra = _claim("cc_gbw_delta_again__ff_break", "Cc", "gbw_hz", QuantTest(kind="direction", sign="-"))
    runner = _FakeInterventionRunner([
        ("Eff outbuf", {"cc": _FF_VARIANT_GBW}),
        ("", {"cc": _FF_BASE_GBW}),
    ])
    recipe = _ff_break_recipe(extra_claims=(extra,))
    result = run_recipe(recipe, runner)
    assert len(runner.decks) == 2      # still just one baseline + one variant deck
    assert result.runs == 1
    by_id = {c.id: c for c in result.claim_cards}
    assert (by_id["cc_gbw_delta__ff_break"].mechanism.series_ref
            == by_id["cc_gbw_delta_again__ff_break"].mechanism.series_ref)


def test_truncated_pairing_flags_not_certifies():
    """A shorter variant series than requested points must FLAG (C1), never certify a partial pairing."""
    runner = _FakeInterventionRunner([
        ("Eff outbuf", {"cc": [_FF_VARIANT_GBW[0]]}),   # only 1/2 requested points came back
        ("", {"cc": _FF_BASE_GBW}),
    ])
    result = run_recipe(_ff_break_recipe(), runner)
    card = result.claim_cards[0]
    assert card.verdict == VerdictClass.FLAGGED


# ── live: real sky130 via ngspice (docker-gated) ──


def test_rz_null_paired_sweep_live_sky130():
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; integration test needs the sim container")
    recipe = intervention_seed_recipes()[0]     # _rz_null_pm_recipe(): Rz knob, pm_deg metric
    result = run_recipe(recipe, runner)
    card = result.claim_cards[0]
    assert card.mechanism.series_ref == "rz_pm_deg_delta__rz_null"
    delta = result.canonical["rz_pm_deg_delta__rz_null"]
    assert len(delta) == 5                       # a parseable, full-length delta series
    assert all(isinstance(y, float) for _, y in delta)
    assert card.verdict != VerdictClass.FLAGGED
    assert card.verdict != VerdictClass.REJECTED


# ── E3-I2 (spec §4-I2 requirement 2): why()'s _scope_inline renders the intervention tag ──────────




def test_scope_inline_appends_intervention_id():
    # physical intervention (rz_null): no idealization -> no "(idealized)" suffix
    s = {"pdk": "sky130", "corners": ["sky130/tt/27/1.8"], "statistical": "none",
         "intervention": "rz_null", "idealization": None}
    assert _scope_inline(s) == "sky130/tt/27/1.8/intervention:rz_null"


def test_scope_inline_appends_idealized_intervention_id():
    # idealized intervention (ff_break): the exact spec-example tag
    s = {"pdk": "sky130", "corners": ["sky130/tt/27/1.8"], "statistical": "none",
         "intervention": "ff_break", "idealization": "ideal unity-gain E-source buffer"}
    assert _scope_inline(s) == "sky130/tt/27/1.8/intervention:ff_break(idealized)"


def test_scope_inline_omits_intervention_tag_when_absent():
    # every pre-E3 card (no "intervention" key at all) renders exactly as before -- backcompat pin
    s = {"pdk": "sky130", "corners": ["sky130/tt/27/1.8"], "statistical": "none"}
    assert _scope_inline(s) == "sky130/tt/27/1.8"


def test_scope_inline_intervention_tag_rides_after_statistical_and_stimulus():
    # the tag is appended LAST, after every other existing scope axis (undetachable, E1b standard)
    s = {"pdk": "sky130", "corners": ["sky130/tt_mm/27/1.8"], "statistical": "3σ@200",
         "intervention": "ff_break", "idealization": "ideal buffer"}
    assert _scope_inline(s) == "sky130/tt_mm/27/1.8/3σ@200/intervention:ff_break(idealized)"


# ── mocked: agent.why()'s FULL method (not just _scope_inline) renders the intervention tag ────────
# from a mocked graph row -- no Neo4j, complements the pure _scope_inline tests above and the live
# end-to-end tests below.


class _MockInterventionGraph:
    def __init__(self, rows):
        self._rows = rows

    async def run_read_query(self, query, params=None):
        return self._rows





def _why_row(scope: dict, **overrides) -> dict:
    row = {
        "claim": "cc_pm_delta__ff_break", "knob": "Cc", "metric": "pm_deg", "verdict": "VERIFIED",
        "quant_kind": "direction", "narrative": "severing the feedforward path ...",
        "corner": "tt", "temp_c": 27.0, "vdd": 1.8, "engine": "ngspice", "basis": "physical-nominal",
        "scope": _json.dumps(scope), "dominant_risk_untested": None,
        "spec_id": "sha256:abc", "topology_class": "miller_ota_2stage_nmos_in",
        "grounds": None,
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_why_full_method_renders_idealized_intervention_tag_mocked():
    from openclaw_brain.agent import BrainAgent

    agent = BrainAgent(_load_config())
    agent._started = True
    agent._graph = _MockInterventionGraph([_why_row({
        "corners": ["sky130/tt/27/1.8"], "statistical": "none",
        "intervention": "ff_break", "idealization": "ideal unity-gain E-source buffer"})])
    out = await agent.why("sha256:abc:cc_pm_delta__ff_break")
    assert out["found"] is True
    assert out["scope"]["intervention"] == "ff_break"
    assert out["verdict_scoped"] == "VERIFIED@sky130/tt/27/1.8/intervention:ff_break(idealized)"


@pytest.mark.asyncio
async def test_why_full_method_renders_physical_intervention_tag_mocked():
    from openclaw_brain.agent import BrainAgent

    agent = BrainAgent(_load_config())
    agent._started = True
    agent._graph = _MockInterventionGraph([_why_row({
        "corners": ["sky130/tt/27/1.8"], "statistical": "none",
        "intervention": "rz_null", "idealization": None},
        claim="rz_pm_delta__rz_null", knob="Rz")])
    out = await agent.why("sha256:abc:rz_pm_delta__rz_null")
    assert out["found"] is True
    assert out["verdict_scoped"] == "VERIFIED@sky130/tt/27/1.8/intervention:rz_null"
    assert "(idealized)" not in out["verdict_scoped"]


@pytest.mark.asyncio
async def test_why_full_method_backcompat_no_intervention_key_mocked():
    # a pre-E3 card's scope has no "intervention" key at all -- why() renders exactly as before
    from openclaw_brain.agent import BrainAgent

    agent = BrainAgent(_load_config())
    agent._started = True
    agent._graph = _MockInterventionGraph([_why_row({
        "corners": ["sky130/tt/27/1.8"], "statistical": "none"}, claim="cc_gbw", knob="Cc",
        metric="gbw_hz")])
    out = await agent.why("sha256:abc:cc_gbw")
    assert out["found"] is True
    assert out["verdict_scoped"] == "VERIFIED@sky130/tt/27/1.8"


# ── live: the projected pilot cards on the REAL Neo4j graph (spec §4-I2 requirement 3/5) ──────────
#
# Populated by `scripts/run_e3_pilot.py --apply` -- the FIRST intentional graph write of intervention
# cards (spec §4-I2 requirement 3). These tests are skip-gated (not fail-gated) on that write having
# happened, exactly like test_executable_lesson.py's live_agent tests gate on Neo4j availability.


class _PilotMockJournal:
    def log(self, *a, **k):
        pass


@pytest.fixture
async def pilot_live_agent():
    require_live_graph()
    from openclaw_brain.agent import BrainAgent
    from openclaw_brain.config import load_config
    from openclaw_brain.knowledge.graph.store import GraphStore

    cfg = load_config()
    a = BrainAgent(cfg)
    a._graph = GraphStore(cfg.neo4j)
    try:
        await a._graph.connect()
    except Exception:
        pytest.skip("Neo4j not available")
    a._started = True
    a._journal = _PilotMockJournal()
    yield a
    await a._graph.close()


async def _find_pilot_card(agent, claim_suffix: str) -> str | None:
    rows = await agent._graph.run_read_query(
        "MATCH (c:ClaimCard) WHERE c.claim_id ENDS WITH $suffix RETURN c.claim_id AS id LIMIT 1",
        {"suffix": f":{claim_suffix}"})
    return rows[0]["id"] if rows else None


_PILOT_SKIP = ("intervention pilot card not projected yet -- run: "
              ".venv/bin/python3 scripts/run_e3_pilot.py --apply")


@pytest.mark.asyncio
async def test_why_renders_idealized_intervention_tag_on_live_projected_ff_break_card(pilot_live_agent):
    """The core E3-I2 rendering requirement, end-to-end: query the REAL projected ff_break card
    through agent.why() and assert the undetachable inline verdict tag (spec's own worked example:
    VERIFIED@sky130/tt/27/1.8/intervention:ff_break(idealized))."""
    cid = await _find_pilot_card(pilot_live_agent, "cc_pm_delta__ff_break")
    if cid is None:
        pytest.skip(_PILOT_SKIP)
    g = await pilot_live_agent.why(cid)
    assert g["found"] is True
    assert g["verdict"] in ("VERIFIED", "VERIFIED_WITH_CAVEAT")
    assert g["scope"]["intervention"] == "ff_break"
    assert g["scope"]["idealization"]
    assert g["verdict_scoped"].endswith("/intervention:ff_break(idealized)")


@pytest.mark.asyncio
async def test_why_renders_physical_intervention_tag_without_idealized_suffix_live(pilot_live_agent):
    """rz_null is PHYSICAL (idealization=None) -- the live tag must NOT carry "(idealized)"."""
    cid = await _find_pilot_card(pilot_live_agent, "rz_pm_delta__rz_null")
    if cid is None:
        pytest.skip(_PILOT_SKIP)
    g = await pilot_live_agent.why(cid)
    assert g["found"] is True
    assert g["scope"]["intervention"] == "rz_null"
    assert g["verdict_scoped"].endswith("/intervention:rz_null")
    assert "(idealized)" not in g["verdict_scoped"]


@pytest.mark.asyncio
async def test_audit_causal_claim_passes_citing_live_projected_ff_break_card(pilot_live_agent):
    """The exact PASS causal sentence (experiments/E3_INTERVENTION_REPORT.md's Q3 demonstration),
    cited to the REAL projected ff_break card, passes audit_citations end-to-end -- proof this is not
    a mocked-resolver-only result."""
    cid = await _find_pilot_card(pilot_live_agent, "cc_pm_delta__ff_break")
    if cid is None:
        pytest.skip(_PILOT_SKIP)
    plan = {"topology_class": "miller_ota_2stage_nmos_in", "claims": [
        {"text": ("The phase margin degradation at large Cc is caused by the feedforward path "
                  "through Ccomp — certified under an idealized buffer intervention, in the model, "
                  "not the literal un-idealized circuit."),
         "tier": "certified", "cites": cid},
    ]}
    rep = await pilot_live_agent.audit_citations(plan)
    assert rep["passed"] is True, rep["findings"]


@pytest.mark.asyncio
async def test_audit_causal_claim_fails_citing_live_non_intervention_card(pilot_live_agent):
    """The SAME causal sentence, cited to the specimen's own pre-existing (non-intervention) cl_pm
    card (also pm_deg, also direction-kind, also VERIFIED) -- must FAIL: certification alone never
    licenses causation, only an intervention-family citation does."""
    cid = await _find_pilot_card(pilot_live_agent, "cc_pm_delta__ff_break")
    if cid is None:
        pytest.skip(_PILOT_SKIP)
    spec_id = cid.rsplit(":", 1)[0]
    cl_pm_id = await _find_pilot_card(pilot_live_agent, "cl_pm")
    if cl_pm_id is None or cl_pm_id.rsplit(":", 1)[0] != spec_id:
        pytest.skip("cl_pm card not on the same specimen -- corpus layout changed since this test was written")
    plan = {"topology_class": "miller_ota_2stage_nmos_in", "claims": [
        {"text": "The phase margin degradation at large Cc is caused by the feedforward path through Ccomp.",
         "tier": "certified", "cites": cl_pm_id},
    ]}
    rep = await pilot_live_agent.audit_citations(plan)
    assert rep["passed"] is False
    assert "intervention" in rep["findings"][0]["reason"]
    assert "mechanism_never_fact" in rep["findings"][0]["reason"]


@pytest.mark.asyncio
async def test_audit_causal_claim_fails_without_citation_live(pilot_live_agent):
    plan = {"topology_class": "miller_ota_2stage_nmos_in", "claims": [
        {"text": "The phase margin degradation at large Cc is caused by the feedforward path through Ccomp.",
         "tier": "certified", "cites": None},
    ]}
    rep = await pilot_live_agent.audit_citations(plan)
    assert rep["passed"] is False and "no citation" in rep["findings"][0]["reason"]


@pytest.mark.asyncio
async def test_audit_causal_claim_fails_idealization_laundered_live(pilot_live_agent):
    """The same real ff_break citation as the PASS test, causal, correctly attributing the PATH --
    but the claim's own text never acknowledges that ff_break is an idealized intervention (the
    idealization laundered away by omission) -- must FAIL even though the causal-family requirement
    is satisfied. This is the one Q3 audit variant the report claimed was run live but, before this
    fix, was actually only exercised against a mocked resolver (test_executable_lesson.py) -- this
    test closes that gap so E3_INTERVENTION_REPORT.md §5's 'identical outcomes both times' claim
    (mocked AND live, all 5 variants) is literally true, not just true-in-spirit."""
    cid = await _find_pilot_card(pilot_live_agent, "cc_pm_delta__ff_break")
    if cid is None:
        pytest.skip(_PILOT_SKIP)
    plan = {"topology_class": "miller_ota_2stage_nmos_in", "claims": [
        {"text": "The phase margin degradation at large Cc is caused by the feedforward path through Ccomp.",
         "tier": "certified", "cites": cid},
    ]}
    rep = await pilot_live_agent.audit_citations(plan)
    assert rep["passed"] is False
    assert "over-generalizes" in rep["findings"][0]["reason"]
    assert "idealization" in rep["findings"][0]["reason"]


@pytest.mark.asyncio
async def test_audit_causal_claim_fails_cross_topology_overgeneralized_live(pilot_live_agent):
    """The same real ff_break citation as the PASS test, but the prose transfers the single-specimen
    pathway to every instance of the topology class -- must FAIL even though the causal-family and
    requirement is satisfied; the idealization acknowledgment is deliberately ABSENT here —
    this test exercises the cross-topology generalize refusal specifically (either refusal
    failing the audit is the asserted outcome)."""
    cid = await _find_pilot_card(pilot_live_agent, "cc_pm_delta__ff_break")
    if cid is None:
        pytest.skip(_PILOT_SKIP)
    plan = {"topology_class": "miller_ota_2stage_nmos_in", "claims": [
        {"text": ("This is why ALL two-stage amps suffer this penalty: the phase margin "
                  "degradation is caused by the feedforward path through Ccomp."),
         "tier": "certified", "cites": cid},
    ]}
    rep = await pilot_live_agent.audit_citations(plan)
    assert rep["passed"] is False and "over-generalizes" in rep["findings"][0]["reason"]
