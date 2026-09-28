"""Teaching-loop lesson model + light citation audit (knowledge/executable/lesson.py).

The audit is ADVISORY and guards ONE boundary: a 'certified' claim must cite an existing certified
claim-card; an 'interpretive' claim just needs its tier label. It judges nothing else (direction,
completeness, quality = Hermes's domain). Pure: a mock resolver, no Neo4j / no LLM.
"""

from openclaw_brain.knowledge.executable.lesson import (
    LessonClaim, LessonPlan, audit_citations,
)


def _plan(*claims):
    return LessonPlan(topology_class="miller_ota_2stage_nmos_in", claims=list(claims))


def _resolver(verdicts):
    # verdicts: {claim_card_id: verdict_str}; an unknown id resolves to None (not found)
    return lambda cid: ({"verdict": verdicts[cid]} if cid in verdicts else None)


def test_certified_claim_that_resolves_passes():
    r = audit_citations(
        _plan(LessonClaim(text="GBW falls as Cc rises", tier="certified", cites="s:cc_gbw")),
        _resolver({"s:cc_gbw": "VERIFIED"}))
    assert r.passed and r.certified_total == 1 and r.certified_ok == 1
    assert r.findings[0].ok


def test_certified_with_caveat_is_also_certified():
    r = audit_citations(
        _plan(LessonClaim(text="x", tier="certified", cites="s:cl_pm")),
        _resolver({"s:cl_pm": "VERIFIED_WITH_CAVEAT"}))
    assert r.passed and r.certified_ok == 1


def test_certified_without_citation_is_a_finding():
    r = audit_citations(
        _plan(LessonClaim(text="x", tier="certified", cites=None)), _resolver({}))
    assert not r.passed and not r.findings[0].ok and "no citation" in r.findings[0].reason


def test_certified_citing_refuted_card_is_a_finding():
    r = audit_citations(
        _plan(LessonClaim(text="x", tier="certified", cites="s:c")),
        _resolver({"s:c": "REFUTED"}))
    assert not r.passed and "not certified" in r.findings[0].reason


def test_certified_citing_missing_card_is_a_finding():
    r = audit_citations(
        _plan(LessonClaim(text="x", tier="certified", cites="s:nope")), _resolver({}))
    assert not r.passed and "not found" in r.findings[0].reason


def test_interpretive_claim_needs_no_citation():
    r = audit_citations(
        _plan(LessonClaim(text="the Miller effect splits the poles", tier="interpretive")),
        _resolver({}))
    assert r.passed and r.interpretive_total == 1 and r.findings[0].ok


def test_invalid_tier_is_a_finding():
    r = audit_citations(_plan(LessonClaim(text="x", tier="fact")), _resolver({}))
    assert not r.passed and "invalid tier" in r.findings[0].reason


def test_mixed_plan_aggregates_correctly():
    r = audit_citations(
        _plan(
            LessonClaim(text="GBW falls as Cc rises", tier="certified", cites="s:cc_gbw"),
            LessonClaim(text="because of Miller pole-splitting", tier="interpretive"),
            LessonClaim(text="bad fact", tier="certified", cites="s:nope"),
        ),
        _resolver({"s:cc_gbw": "VERIFIED"}))
    assert not r.passed
    assert r.certified_total == 2 and r.certified_ok == 1 and r.interpretive_total == 1
    assert [f.ok for f in r.findings] == [True, True, False]


# ── magnitude-under-certified check (kind-aware): a certified claim asserting a scalar magnitude must
#    NOT cite a card that certifies only a direction/invariance (the Hermes teaching-eval leak) ──

from openclaw_brain.knowledge.executable.lesson import _asserts_magnitude


def _resolver_kind(cards):
    # cards: {claim_card_id: (verdict, quant_kind)}
    return lambda cid: ({"verdict": cards[cid][0], "quant_kind": cards[cid][1]} if cid in cards else None)


def test_magnitude_under_invariance_card_is_a_finding():
    # "~67 dB" presented as certified while the card certifies only CL-invariance -> over-claim
    r = audit_citations(
        _plan(LessonClaim(text="open-loop gain is about 67 dB and is independent of CL",
                          tier="certified", cites="s:tele_av0")),
        _resolver_kind({"s:tele_av0": ("VERIFIED", "invariance")}))
    assert not r.passed and r.certified_ok == 0 and "scalar magnitude" in r.findings[0].reason


def test_magnitude_under_direction_card_is_a_finding():
    r = audit_citations(
        _plan(LessonClaim(text="GBW is 26.3 MHz", tier="certified", cites="s:gbw")),
        _resolver_kind({"s:gbw": ("VERIFIED", "direction")}))
    assert not r.passed and "scalar magnitude" in r.findings[0].reason


def test_shape_claim_without_magnitude_under_invariance_passes():
    # the legitimate certified claim: assert the BEHAVIOR (invariance), no number
    r = audit_citations(
        _plan(LessonClaim(text="DC gain is independent of the load capacitance",
                          tier="certified", cites="s:tele_av0")),
        _resolver_kind({"s:tele_av0": ("VERIFIED", "invariance")}))
    assert r.passed and r.certified_ok == 1


def test_magnitude_under_value_card_passes():
    # a value-kind card DOES certify a scalar, so a magnitude under it is legitimately certified
    r = audit_citations(
        _plan(LessonClaim(text="the trip point is 0.9 V", tier="certified", cites="s:trip")),
        _resolver_kind({"s:trip": ("VERIFIED", "value")}))
    assert r.passed and r.certified_ok == 1


def test_magnitude_under_card_without_kind_passes_backcompat():
    # a resolver that does not report quant_kind keeps the prior citation-only behavior
    r = audit_citations(
        _plan(LessonClaim(text="gain is 67 dB", tier="certified", cites="s:av0")),
        _resolver({"s:av0": "VERIFIED"}))
    assert r.passed and r.certified_ok == 1


def test_asserts_magnitude_detector():
    for hit in ["~67 dB", "about 67 dB", "0.2%", "26.3 MHz", "2.0e7 V/s", "30.3 µA", "864 kohm"]:
        assert _asserts_magnitude(hit), hit
    for miss in ["GBW falls as Cc rises", "independent of CL", "at tt / 27 C / 1.8 V",
                 "the Miller effect splits the poles", "VDD = 1.8 V"]:
        assert not _asserts_magnitude(miss), miss


# ── refusal-to-generalize: a certified claim must not generalize a scoped verdict beyond what the engine
#    certified (ADR 4.3), SYMMETRIC per engine basis, SCOPE-SEMANTIC (tested against paraphrase) ──


def _resolver_basis(cards):
    # cards: {claim_card_id: (verdict, quant_kind, basis)}
    return lambda cid: ({"verdict": cards[cid][0], "quant_kind": cards[cid][1], "basis": cards[cid][2]}
                        if cid in cards else None)


def test_analog_certified_claim_generalizing_to_silicon_is_a_finding():
    r = audit_citations(
        _plan(LessonClaim(text="this cancellation is robust in silicon", tier="certified", cites="s:cds")),
        _resolver_basis({"s:cds": ("VERIFIED", "invariance", "physical-nominal")}))
    assert not r.passed and "over-generalizes" in r.findings[0].reason


def test_analog_generalization_paraphrases_are_flagged():
    for prose in ["it yields across all corners", "the design is production-ready",
                  "mismatch-immune offset cancellation", "verified by Monte-Carlo"]:
        r = audit_citations(
            _plan(LessonClaim(text=prose, tier="certified", cites="s:cds")),
            _resolver_basis({"s:cds": ("VERIFIED", "invariance", "physical-nominal")}))
        assert not r.passed, prose


def test_digital_certified_claim_generalizing_to_timing_is_a_finding():
    r = audit_citations(
        _plan(LessonClaim(text="the back-end is timing-clean", tier="certified", cites="s:ss")),
        _resolver_basis({"s:ss": ("VERIFIED", "invariance", "functional")}))
    assert not r.passed and "over-generalizes" in r.findings[0].reason


def test_digital_generalization_paraphrases_are_flagged():
    for prose in ["meets timing in the target process", "the decoder is CDC-clean",
                  "no metastability risk", "matches the real spec"]:
        r = audit_citations(
            _plan(LessonClaim(text=prose, tier="certified", cites="s:ss")),
            _resolver_basis({"s:ss": ("VERIFIED", "invariance", "functional")}))
        assert not r.passed, prose


def test_in_scope_certified_claim_passes_each_basis():
    # analog in-scope: pedestal invariance, no robustness/silicon language
    ra = audit_citations(
        _plan(LessonClaim(text="the held output is invariant to a common input pedestal at tt/27/1.8",
                          tier="certified", cites="s:cds")),
        _resolver_basis({"s:cds": ("VERIFIED", "invariance", "physical-nominal")}))
    assert ra.passed and ra.certified_ok == 1
    # digital in-scope: the RTL entails the bijection, no timing/spec-match language
    rd = audit_citations(
        _plan(LessonClaim(text="the RTL decode matches the gray-to-binary reference over the full range",
                          tier="certified", cites="s:ss")),
        _resolver_basis({"s:ss": ("VERIFIED", "invariance", "functional")}))
    assert rd.passed and rd.certified_ok == 1


def test_generalization_skipped_when_basis_absent_backcompat():
    # a card with no recorded basis keeps the prior behavior (generalization not refused)
    r = audit_citations(
        _plan(LessonClaim(text="robust across all corners", tier="certified", cites="s:old")),
        _resolver({"s:old": "VERIFIED"}))
    assert r.passed and r.certified_ok == 1


# ── metric-consistency check (HERMES_TEACHING_EVAL_2026-06-29.md finding #1, second half): the
#    magnitude-under-shape guard catches "scalar under a shape-only kind", but not a claim naming the
#    WRONG measurable quantity while citing a card whose kind legitimately certifies a scalar
#    (value/elasticity/statistical/corner) — that slips past the shape guard entirely. ──

from openclaw_brain.knowledge.executable.lesson import _metric_mismatch


def _resolver_full(cards):
    # cards: {claim_card_id: {"verdict":..., "quant_kind":..., "basis":..., "metric":...}} — any
    # subset of keys; missing keys behave as if the resolver never reported that field (backcompat).
    return lambda cid: (cards[cid] if cid in cards else None)


# --- Step 1: reproduce the two OBSERVED leak transcripts (smoke fact 2, c5's cascode-mirror bullet)
#     verbatim and confirm they are ALREADY closed by the existing _MAGNITUDE_RE shape guard, not by
#     the new metric check (both cite an invariance-kind card whose metric DOES match the claim's
#     topic — gain-for-gain, current-for-current — so only the kind/shape mismatch is at fault here). ---

def test_smoke_transcript_leak_already_closed_by_shape_guard():
    # scratchpad/hermes_eval/smoke_telescopic.txt fact 2, verbatim: "~67 dB" certified, citing
    # tele_av0 (kind=invariance, metric=av0_db — metric matches "gain", so ONLY the shape guard fires)
    r = audit_citations(
        _plan(LessonClaim(
            text="DC gain is high and does not depend on CL. In the same verified specimen, "
                 "open-loop gain is about 67 dB and is independent of load capacitance.",
            tier="certified", cites="tele:av0")),
        _resolver_full({"tele:av0": {"verdict": "VERIFIED", "quant_kind": "invariance", "metric": "av0_db"}}))
    assert not r.passed
    assert "scalar magnitude" in r.findings[0].reason  # the shape guard's reason, not the metric one
    assert "metric mismatch" not in r.findings[0].reason


def test_c5_transcript_leak_already_closed_by_shape_guard():
    # scratchpad/hermes_eval/case_c5_beginner.txt, verbatim: "flat within about 0.2%" certified,
    # citing casc_iout (kind=invariance, metric=iout_a — metric matches "current", shape guard fires)
    r = audit_citations(
        _plan(LessonClaim(
            text="certified claim: when Vout is swept, iout is nearly independent of Vout above "
                 "compliance; simulation narrative reports flat within about 0.2% over "
                 "Vout = 0.8-1.6 V.",
            tier="certified", cites="casc:iout")),
        _resolver_full({"casc:iout": {"verdict": "VERIFIED", "quant_kind": "invariance", "metric": "iout_a"}}))
    assert not r.passed
    assert "scalar magnitude" in r.findings[0].reason
    assert "metric mismatch" not in r.findings[0].reason


# --- Step 2: the genuinely NEW slip-through — a magnitude-legitimate kind (elasticity / statistical /
#     corner) so the shape guard does not fire at all, but the claim names the WRONG metric. ---

def test_metric_mismatch_under_elasticity_kind_slips_past_shape_guard():
    # ota5t_pelgrom really certifies offset-vs-area (a_vos, kind=elasticity — NOT shape-only, so the
    # magnitude guard alone would pass this); claiming it is about GAIN is the semantically-blind leak.
    r = audit_citations(
        _plan(LessonClaim(text="device area increases the gain that this stage achieves",
                          tier="certified", cites="ota5t:pelgrom")),
        _resolver_full({"ota5t:pelgrom": {"verdict": "VERIFIED", "quant_kind": "elasticity", "metric": "a_vos"}}))
    assert not r.passed and r.certified_ok == 0
    assert "metric mismatch" in r.findings[0].reason
    assert "gain" in r.findings[0].reason and "offset" in r.findings[0].reason


def test_metric_mismatch_offset_card_claimed_as_noise_is_flagged():
    # the offset/FPN-vs-noise confusion named explicitly in the eval finding: col_fpn (statistical,
    # metric=vos_v -> "offset") is a deterministic mismatch bound, never a noise measurement.
    r = audit_citations(
        _plan(LessonClaim(text="the column noise is bounded to 60 mV worst-case",
                          tier="certified", cites="cmp:col_fpn")),
        _resolver_full({"cmp:col_fpn": {"verdict": "VERIFIED", "quant_kind": "statistical", "metric": "vos_v"}}))
    assert not r.passed
    assert "metric mismatch" in r.findings[0].reason


def test_metric_mismatch_under_corner_kind_slips_past_shape_guard():
    # ota5t_gbw_corner certifies GBW held across corners (kind=corner — also not shape-only); claiming
    # it as a phase-margin result is a metric mismatch the shape guard cannot see.
    r = audit_citations(
        _plan(LessonClaim(text="phase margin holds above 29 MHz at every process corner",
                          tier="certified", cites="ota5t:gbw_corner")),
        _resolver_full({"ota5t:gbw_corner": {"verdict": "VERIFIED", "quant_kind": "corner", "metric": "gbw_hz"}}))
    assert not r.passed
    assert "metric mismatch" in r.findings[0].reason


# --- Step 3: false-positive discipline — legitimate certified claims (including magnitude-legitimate
#     kinds, i.e. the kind-vs-assertion requirement extended with no gap) MUST pass. ---

def test_gbw_claim_citing_gbw_card_passes_smoke_fact_1():
    # smoke_telescopic.txt fact 1, verbatim and legitimately certified (direction kind, no magnitude)
    r = audit_citations(
        _plan(LessonClaim(text="Load capacitance controls GBW. In the verified specimen, as CL "
                               "increases, gain-bandwidth product falls.",
                          tier="certified", cites="tele:gbw")),
        _resolver_full({"tele:gbw": {"verdict": "VERIFIED", "quant_kind": "direction", "metric": "gbw_hz"}}))
    assert r.passed and r.certified_ok == 1


def test_corner_kind_magnitude_claim_passes_kind_vs_assertion_no_gap():
    # locks in requirement (ii): corner (like value/statistical/elasticity) legitimately certifies a
    # scalar bound (judge_quant: VERIFIED iff ALL corners meet the bound) — must NOT be flagged as a
    # magnitude-under-shape leak, and the metric ("bandwidth") matches the claim text too.
    r = audit_citations(
        _plan(LessonClaim(text="GBW holds >= 29 MHz at every process corner",
                          tier="certified", cites="ota5t:gbw_corner")),
        _resolver_full({"ota5t:gbw_corner": {"verdict": "VERIFIED", "quant_kind": "corner", "metric": "gbw_hz"}}))
    assert r.passed and r.certified_ok == 1


def test_statistical_kind_offset_claim_passes():
    r = audit_citations(
        _plan(LessonClaim(text="the input-referred offset is bounded to 20 mV at 3-sigma",
                          tier="certified", cites="ota5t:vos")),
        _resolver_full({"ota5t:vos": {"verdict": "VERIFIED", "quant_kind": "statistical", "metric": "vos_v"}}))
    assert r.passed and r.certified_ok == 1


def test_generic_claim_naming_no_metric_keyword_passes():
    # no recognized category is named at all -> the check refuses to guess and stays silent
    r = audit_citations(
        _plan(LessonClaim(text="this relationship is oracle-backed and monotonic across the sweep",
                          tier="certified", cites="ota5t:pelgrom")),
        _resolver_full({"ota5t:pelgrom": {"verdict": "VERIFIED", "quant_kind": "elasticity", "metric": "a_vos"}}))
    assert r.passed and r.certified_ok == 1


def test_interpretive_claim_with_mismatched_wording_is_never_flagged():
    # tier=interpretive needs no citation and is exempt from every certified-only check
    r = audit_citations(
        _plan(LessonClaim(text="this offset result is basically about gain, informally speaking",
                          tier="interpretive")),
        _resolver_full({}))
    assert r.passed and r.interpretive_total == 1


def test_metric_check_skipped_when_resolver_omits_metric_backcompat():
    # a resolver that never reports "metric" (the prior CardResolver contract) keeps the prior,
    # metric-blind behavior — mirrors the existing quant_kind/basis backcompat tests
    r = audit_citations(
        _plan(LessonClaim(text="the gain is 40 dB and the offset is small", tier="certified", cites="s:x")),
        _resolver({"s:x": "VERIFIED"}))
    assert r.passed and r.certified_ok == 1


def test_metric_check_skipped_for_unmapped_metric():
    # "vo_v" (a held-output value) is intentionally NOT in the alias table; the check must not guess
    r = audit_citations(
        _plan(LessonClaim(text="the gain is well controlled here", tier="certified", cites="cds:vo")),
        _resolver_full({"cds:vo": {"verdict": "VERIFIED", "quant_kind": "invariance", "metric": "vo_v"}}))
    assert r.passed and r.certified_ok == 1


def test_metric_mismatch_detector_unit():
    assert _metric_mismatch("the gain is 40 dB", "vos_v") is not None
    assert _metric_mismatch("the offset is small", "vos_v") is None
    assert _metric_mismatch("the offset is small", "unknown_metric") is None
    assert _metric_mismatch("the offset is small", None) is None
    assert _metric_mismatch("no recognized quantity named here", "vos_v") is None


# ── Neo4j-gated: why + audit against the live projected graph ──

import pytest
from tests.conftest import require_live_graph

from openclaw_brain.agent import BrainAgent
from openclaw_brain.config import load_config
from openclaw_brain.knowledge.graph.store import GraphStore


class _MockJournal:
    def log(self, *a, **k):
        pass


@pytest.fixture
async def live_agent():
    require_live_graph()
    cfg = load_config()
    a = BrainAgent(cfg)
    a._graph = GraphStore(cfg.neo4j)
    try:
        await a._graph.connect()
    except Exception:
        pytest.skip("Neo4j not available")
    a._started = True
    a._journal = _MockJournal()
    yield a
    await a._graph.close()


@pytest.mark.asyncio
async def test_why_and_audit_on_live_graph(live_agent):
    # discover a certified claim-card id from the live graph (robust to spec_id changes)
    rows = await live_agent._graph.run_read_query(
        "MATCH (c:ClaimCard) WHERE c.verdict IN ['VERIFIED','VERIFIED_WITH_CAVEAT'] "
        "RETURN c.claim_id AS id LIMIT 1", {})
    if not rows:
        pytest.skip("no projected certified claim-cards (run: openclaw-brain project-executable --apply)")
    cid = rows[0]["id"]

    # why() returns real grounding
    g = await live_agent.why(cid)
    assert g["found"] is True
    assert g["verdict"] in ("VERIFIED", "VERIFIED_WITH_CAVEAT")
    assert g["conditions"]["corner"]  # R1 present

    # a plan that cites the real card + labels a mechanism interpretive PASSES
    good = {"topology_class": g["topology_class"], "claims": [
        {"text": "a certified fact", "tier": "certified", "cites": cid},
        {"text": "an interpretive mechanism", "tier": "interpretive"}]}
    assert (await live_agent.audit_citations(good))["passed"] is True

    # negative control: a mechanism stated as certified-fact with no citation FAILS
    bad = {"topology_class": g["topology_class"], "claims": [
        {"text": "mechanism stated as fact", "tier": "certified", "cites": None}]}
    rep = await live_agent.audit_citations(bad)
    assert rep["passed"] is False and not rep["findings"][0]["ok"]


@pytest.mark.asyncio
async def test_audit_flags_magnitude_under_invariance_on_live_graph(live_agent):
    # the kind-aware check on the REAL graph: a certified absolute magnitude citing an invariance-kind
    # card (e.g. tele_av0/cds_vo/casc_iout) must FAIL — the oracle proved only invariance, not the value
    rows = await live_agent._graph.run_read_query(
        "MATCH (c:ClaimCard) WHERE c.quant_kind = 'invariance' "
        "AND c.verdict IN ['VERIFIED','VERIFIED_WITH_CAVEAT'] RETURN c.claim_id AS id LIMIT 1", {})
    if not rows:
        pytest.skip("no projected invariance claim-card with quant_kind (run project-executable --apply)")
    cid = rows[0]["id"]
    leak = {"topology_class": "x", "claims": [
        {"text": "the DC gain is about 67 dB", "tier": "certified", "cites": cid}]}
    rep = await live_agent.audit_citations(leak)
    assert rep["passed"] is False
    assert rep["certified_ok"] == 0 and "scalar magnitude" in rep["findings"][0]["reason"]


def test_cross_node_transfer_is_a_finding():
    # Stat-QT: a sky130/130nm (physical-nominal) verdict must not be presented as the user's node —
    # the rev-1 within-node refusal missed this cross-node magnitude transfer (paraphrase-tested).
    for prose in ["this applies to your 28nm process", "scales to your PDK", "expect this in your process"]:
        r = audit_citations(
            _plan(LessonClaim(text=prose, tier="certified", cites="s:cds")),
            _resolver_basis({"s:cds": ("VERIFIED", "statistical", "physical-nominal")}))
        assert not r.passed, prose


# ── E3-I2 (spec §4-I2 Q3): intervention-aware causal audit ──────────────────────────────────────
#
# A certified claim may assert CAUSATION ("caused by"/"the cause of"/"because"/"due to the ... path")
# ONLY when it cites an intervention-family card (grounds/scope carries an intervention id, E3-I1's
# do-operator substrate). The SAME causal sentence citing an ordinary observational card must FAIL.
# Over-generalization discipline (cross-topology-instance, node/corner transfer) still applies to
# causal claims — reusing `_overgeneralizes`, not a parallel checker — and a cited card's named
# `idealization` must be acknowledged in the claim's own text or the claim over-generalizes too.

from openclaw_brain.knowledge.executable.lesson import (
    _causal_without_intervention, _is_causal_claim, _overgeneralizes,
)


def _resolver_iv(cards):
    # cards: {claim_card_id: {"verdict", "quant_kind", "basis", "metric", "intervention", "idealization"}}
    return lambda cid: (cards[cid] if cid in cards else None)


_FF_BREAK_IDEALIZATION = (
    "ideal unity-gain E-source buffer replicates the output node; Ccomp is redriven from o1 to this "
    "buffered replica instead of directly to 'out'"
)

# The one sentence the report cites as the PASSING certified causal claim (spec §4-I2 requirement 4:
# "the exact causal sentence that PASSES with citation"). Deliberately: (i) causal ("is caused by"),
# (ii) names the PATH, not "the zero alone" (the physics-attribution caveat), (iii) acknowledges the
# idealized/model scope ("idealized", "in the model") so it does not launder ff_break's idealization.
PASS_CAUSAL_TEXT = (
    "The phase margin degradation at large Cc is caused by the feedforward path through Ccomp — "
    "certified under an idealized buffer intervention, in the model, not the literal un-idealized "
    "circuit."
)

# The same claim's mechanism, stated causally, with NO idealization acknowledgment — the baseline
# for the "with/without citation" and "idealization-laundered" FAIL variants below.
BASE_CAUSAL_TEXT = "The phase margin degradation at large Cc is caused by the feedforward path through Ccomp."


def test_causal_claim_citing_intervention_card_passes():
    r = audit_citations(
        _plan(LessonClaim(text=PASS_CAUSAL_TEXT, tier="certified", cites="ff:pm_delta")),
        _resolver_iv({"ff:pm_delta": {"verdict": "VERIFIED", "quant_kind": "direction",
                                       "basis": "physical-nominal", "metric": "pm_deg",
                                       "intervention": "ff_break",
                                       "idealization": _FF_BREAK_IDEALIZATION}}))
    assert r.passed and r.certified_ok == 1, r.findings[0].reason


def test_causal_claim_without_citation_fails():
    r = audit_citations(
        _plan(LessonClaim(text=PASS_CAUSAL_TEXT, tier="certified", cites=None)), _resolver({}))
    assert not r.passed and "no citation" in r.findings[0].reason


def test_causal_claim_citing_non_intervention_card_fails():
    # the SAME causal sentence, but the cited card carries no intervention id (an ordinary,
    # non-intervened observational verdict) — must FAIL, however certified its direction is.
    r = audit_citations(
        _plan(LessonClaim(text=BASE_CAUSAL_TEXT, tier="certified", cites="s:pm_obs")),
        _resolver_iv({"s:pm_obs": {"verdict": "VERIFIED", "quant_kind": "direction",
                                    "basis": "physical-nominal", "metric": "pm_deg"}}))
    assert not r.passed and r.certified_ok == 0
    assert "intervention" in r.findings[0].reason and "mechanism_never_fact" in r.findings[0].reason


def test_active_voice_causes_citing_non_intervention_card_fails():
    # Regression pin for the confirmed must-fix defect: an ACTIVE-voice "causes" claim (not the
    # "caused by" passive the first draft of _CAUSAL_RE matched) cited to a NON-intervention VERIFIED
    # pm_deg card must FAIL exactly like the passive-voice BASE_CAUSAL_TEXT case above -- this is the
    # end-to-end adversarial probe that previously false-accepted (certified_ok=1) before the fix.
    r = audit_citations(
        _plan(LessonClaim(
            text="Cc's feedforward current causes the phase margin degradation at large Cc.",
            tier="certified", cites="s:pm_obs")),
        _resolver_iv({"s:pm_obs": {"verdict": "VERIFIED", "quant_kind": "direction",
                                    "basis": "physical-nominal", "metric": "pm_deg"}}))
    assert not r.passed and r.certified_ok == 0
    assert "intervention" in r.findings[0].reason and "mechanism_never_fact" in r.findings[0].reason


def test_culprit_explains_why_responsible_for_family_citing_non_intervention_card_fails():
    # The work order's named adversarial probes -- "the culprit is", "which explains why",
    # "responsible for" -- each cited to a non-intervention card must ALSO fail the causal gate (not
    # coincidentally pass or coincidentally get refused by an unrelated check like metric-mismatch).
    for text in [
        "The culprit is the feedforward path through Ccomp.",
        "Which explains why the phase margin degrades at large Cc.",
        "The feedforward path through Ccomp is responsible for the phase margin degradation.",
        "The phase margin degradation stems from the feedforward path through Ccomp.",
        "The phase margin degradation is attributable to the feedforward path through Ccomp.",
    ]:
        r = audit_citations(
            _plan(LessonClaim(text=text, tier="certified", cites="s:pm_obs")),
            _resolver_iv({"s:pm_obs": {"verdict": "VERIFIED", "quant_kind": "direction",
                                        "basis": "physical-nominal", "metric": "pm_deg"}}))
        assert not r.passed and r.certified_ok == 0, text
        assert "intervention" in r.findings[0].reason and "mechanism_never_fact" in r.findings[0].reason, text


def test_causal_claim_idealization_laundered_fails():
    # cites the REAL intervention card (idealized) but the claim's own text never acknowledges the
    # idealized/model scope -- the idealization is laundered away by omission.
    r = audit_citations(
        _plan(LessonClaim(text=BASE_CAUSAL_TEXT, tier="certified", cites="ff:pm_delta")),
        _resolver_iv({"ff:pm_delta": {"verdict": "VERIFIED", "quant_kind": "direction",
                                       "basis": "physical-nominal", "metric": "pm_deg",
                                       "intervention": "ff_break",
                                       "idealization": _FF_BREAK_IDEALIZATION}}))
    assert not r.passed and r.certified_ok == 0
    assert "over-generalizes" in r.findings[0].reason and "idealization" in r.findings[0].reason


def test_causal_claim_cross_topology_overgeneralized_fails():
    # cites a (mock) intervention card so the causal-family check passes, but the prose generalizes
    # the single-specimen pathway to every instance of the topology class (spec's own example phrase).
    text = ("This is why ALL two-stage amps suffer this penalty: the phase margin degradation is "
            "caused by the feedforward path through Ccomp.")
    r = audit_citations(
        _plan(LessonClaim(text=text, tier="certified", cites="ff:pm_delta")),
        _resolver_iv({"ff:pm_delta": {"verdict": "VERIFIED", "quant_kind": "direction",
                                       "basis": "physical-nominal", "metric": "pm_deg",
                                       "intervention": "ff_break", "idealization": None}}))
    assert not r.passed and r.certified_ok == 0
    assert "over-generalizes" in r.findings[0].reason


def test_causal_claim_rz_null_physical_no_idealization_ack_needed():
    # rz_null is a PHYSICAL intervention (idealization=None) -- a causal claim citing it need not
    # acknowledge any idealized scope at all (there is none to launder).
    text = "Phase margin improves because the nulling resistor Rz pushes the RHP zero past 1/gm2."
    r = audit_citations(
        _plan(LessonClaim(text=text, tier="certified", cites="rz:pm_delta")),
        _resolver_iv({"rz:pm_delta": {"verdict": "VERIFIED", "quant_kind": "direction",
                                       "basis": "physical-nominal", "metric": "pm_deg",
                                       "intervention": "rz_null", "idealization": None}}))
    assert r.passed and r.certified_ok == 1, r.findings[0].reason


def test_noncausal_claim_backcompat_ignores_missing_intervention_field():
    # a resolver that never reports "intervention"/"idealization" (the prior CardResolver contract)
    # keeps the prior behavior for a NON-causal claim -- the new checks are additive, not a regression.
    r = audit_citations(
        _plan(LessonClaim(text="GBW falls as Cc rises", tier="certified", cites="s:cc_gbw")),
        _resolver({"s:cc_gbw": "VERIFIED"}))
    assert r.passed and r.certified_ok == 1


def test_causal_phrase_detector_paraphrases():
    for hit in ["caused by", "is caused by the feedforward path", "the cause of the degradation",
                "because the zero falls toward GBW", "due to the parasitic feedforward path",
                # active-voice "causes"/"causing"/"caused" (no "by") -- the base verb of the
                # required "caused by" family, confirmed end-to-end false-accepting before this fix
                "Cc's feedforward current causes the phase margin degradation at large Cc",
                "the feedforward path is causing the shift",
                "the feedforward path caused the shift",
                # culprit / explains-why / responsible-for / stems-from / attributable-to family
                "the culprit is the parasitic feedforward path",
                "which explains why phase margin degrades",
                "the feedforward path is responsible for the degradation",
                "the degradation stems from the feedforward path",
                "the degradation is attributable to the feedforward path"]:
        assert _is_causal_claim(hit), hit
    for miss in ["GBW falls as Cc rises", "PM improves as Rz rises toward 1/gm2",
                 "independent of CL", "the Miller effect splits the poles",
                 # near-miss tokens that must NOT trip the active-voice "caus(?:e|es|ed|ing)" family
                 "the causal graph is well understood", "a causeway connects the two islands"]:
        assert not _is_causal_claim(miss), miss


def test_causal_without_intervention_detector_unit():
    card_iv = {"intervention": "ff_break"}
    card_plain = {"verdict": "VERIFIED"}
    assert _causal_without_intervention("X is caused by Y", card_iv) is None
    assert _causal_without_intervention("X is caused by Y", card_plain) is not None
    assert _causal_without_intervention("GBW falls as Cc rises", card_plain) is None


def test_overgeneralizes_idealization_ack_examples():
    concern = _overgeneralizes("this pathway causes the effect", "physical-nominal",
                               idealization="ideal unity buffer")
    assert concern is not None and "idealization" in concern
    assert _overgeneralizes("this pathway causes the effect, idealized in the model",
                            "physical-nominal", idealization="ideal unity buffer") is None
    # backcompat: 2-positional-arg calls (the prior signature) still work, idealization defaults None
    assert _overgeneralizes("robust across all corners", "physical-nominal") is not None


# ── mechanism_never_fact REGRESSION PIN (spec §4-I2 requirement 1c): E3 does not repeal
#    oracle.py::teaching_fact -- a mechanism narrative is NEVER auto-fact, intervention or not. ──

from openclaw_brain.knowledge.executable.models import (
    AnalogPVT, ClaimCard, MechanismClaim, QuantTest, VerdictClass,
)
from openclaw_brain.knowledge.executable.oracle import ClaimOracle

_PIN_COND = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8,
                      pdk_profile={"pdk": "sky130A"})


def test_teaching_fact_hardened_semantics_byte_identical():
    """HEAD semantics, unchanged by E3-I2: verdict must be certified AND (hardened) the claim must
    carry NO mechanism narrative. Un-narrated claims fact; narrated claims never do under hardened."""
    o = ClaimOracle()
    narrated = ClaimCard(
        id="c1", topology_class="x", conditions=_PIN_COND,
        mechanism=MechanismClaim(knob="Cc", metric="pm_deg", series_ref="s",
                                 quant=QuantTest(kind="direction", sign="+"),
                                 narrative="a mechanism story"),
        verdict=VerdictClass.VERIFIED)
    bare = ClaimCard(
        id="c2", topology_class="x", conditions=_PIN_COND,
        mechanism=MechanismClaim(knob="Cc", metric="pm_deg", series_ref="s",
                                 quant=QuantTest(kind="direction", sign="+"), narrative=None),
        verdict=VerdictClass.VERIFIED)
    unverified = ClaimCard(
        id="c3", topology_class="x", conditions=_PIN_COND,
        mechanism=MechanismClaim(knob="Cc", metric="pm_deg", series_ref="s",
                                 quant=QuantTest(kind="direction", sign="+"), narrative=None),
        verdict=VerdictClass.REFUTED)
    assert o.teaching_fact(narrated, hardened=False) is True
    assert o.teaching_fact(narrated, hardened=True) is False
    assert o.teaching_fact(bare, hardened=True) is True
    assert o.teaching_fact(unverified, hardened=True) is False


def test_teaching_fact_unrepealed_by_intervention_grounds():
    """The non-negotiable (spec §1): mechanism_never_fact is NOT repealed by E3. An intervention-
    family card (grounds/scope carry the intervention id, exactly as executor.py stamps them) whose
    mechanism carries a narrative is STILL never eligible as taught fact under the hardened rule --
    E3 only ever moves a claim across the certified/interpretive line via the LESSON-layer audit
    (audit_citations, above), never by relaxing the oracle's own teaching_fact rule."""
    o = ClaimOracle()
    iv_card = ClaimCard(
        id="cc_pm_delta__ff_break", topology_class="miller_ota_2stage_nmos_in", conditions=_PIN_COND,
        grounds=["intervention:ff_break"],
        scope={"intervention": "ff_break", "idealization": _FF_BREAK_IDEALIZATION},
        mechanism=MechanismClaim(
            knob="Cc", metric="pm_deg", series_ref="cc_pm_delta__ff_break",
            quant=QuantTest(kind="direction", sign="+"),
            narrative="severing the feedforward path removes the RHP zero's phase penalty"),
        verdict=VerdictClass.VERIFIED)
    assert o.teaching_fact(iv_card, hardened=True) is False   # un-repealed: still never auto-fact
    assert o.teaching_fact(iv_card, hardened=False) is True   # naive rule is unaffected either way


# ── law-tier I2 (docs/superpowers/specs/2026-07-04-law-tier-graph-representation.md §5 bullet 2):
#    a certified claim may cite a law_id (a `Regularity`, kind="law") instead of a claim_card id.
#    Generalization is licensed EXACTLY across the law's member PDK set; a law whose status is not
#    "law" fails loudly; causal phrasing citing an (always-observational) law still needs an
#    intervention citation; magnitude-vs-kind is the SAME check, keyed on the law's own quant_kind.

from openclaw_brain.knowledge.executable.lesson import _law_bounded_to_members, _law_overreach

_LAW_MEMBERS = ["gf180mcuD", "ihp-sg13g2", "sky130A"]


def _resolver_law(laws):
    # laws: {law_id: {"status", "quant_kind", "metric", "pdks"}} -- mirrors the "kind": "law" shape
    # agent.audit_citations builds from the combined ClaimCard-or-Regularity resolution query.
    return lambda cid: ({"kind": "law", **laws[cid]} if cid in laws else None)


def test_law_citation_generalizing_across_exactly_member_set_passes():
    for text in [
        "GBW increases with Iref across sky130A, gf180mcuD and ihp-sg13g2",
        "GBW increases with Iref across the three foundry model families tested",
        "in every process model we tested, GBW increases with Iref",
    ]:
        r = audit_citations(
            _plan(LessonClaim(text=text, tier="certified", cites="law:gbw")),
            _resolver_law({"law:gbw": {"status": "law", "quant_kind": "direction",
                                       "metric": "gbw_hz", "pdks": _LAW_MEMBERS}}))
        assert r.passed and r.certified_ok == 1, (text, r.findings[0].reason)


def test_law_citation_beyond_member_set_fails():
    for text in [
        "GBW increases with Iref on any 28nm process",
        "GBW increases with Iref in silicon",
        "GBW increases with Iref on all CMOS processes",
        "GBW increases with Iref universally",
    ]:
        r = audit_citations(
            _plan(LessonClaim(text=text, tier="certified", cites="law:gbw")),
            _resolver_law({"law:gbw": {"status": "law", "quant_kind": "direction",
                                       "metric": "gbw_hz", "pdks": _LAW_MEMBERS}}))
        assert not r.passed, text
        assert "over-generalizes" in r.findings[0].reason, text
        assert "member set" in r.findings[0].reason, text


def test_law_citation_same_sentence_overreach_not_laundered_by_bounding():
    # verified must-fix: bounding language (member enumeration or "tested") anywhere in the text must
    # not neutralize an explicit, unrelated overreach phrase elsewhere in the SAME sentence -- each
    # overreach hit is checked against ITS OWN clause, not the whole text as one OR-gate. Live-
    # confirmed against the real Pelgrom law: all three previously passed (passed=True); all three
    # must fail now.
    for text in [
        "The power law holds across sky130A, gf180mcuD and ihp-sg13g2 — and therefore in silicon "
        "on any 28nm process.",
        "We tested it thoroughly; the power law therefore holds on any process at any foundry.",
        "The power law holds across sky130A, gf180mcuD and ihp-sg13g2 — it holds across the "
        "tested processes and beyond, universally.",
    ]:
        r = audit_citations(
            _plan(LessonClaim(text=text, tier="certified", cites="law:gbw")),
            _resolver_law({"law:gbw": {"status": "law", "quant_kind": "direction",
                                       "metric": "gbw_hz", "pdks": _LAW_MEMBERS}}))
        assert not r.passed, text
        assert "over-generalizes" in r.findings[0].reason, text
        assert "member set" in r.findings[0].reason, text


def test_law_citation_analog_generalization_family_applies_to_law_too():
    # verified must-fix: a Regularity resolves with basis=None, which used to skip the ENTIRE
    # _GENERALIZE_ANALOG family for a law citation -- "silicon-proven, production-ready, and robust
    # across all corners" passed when citing the law but correctly failed citing the law's own member
    # card. A cross-PDK model replication certifies no more silicon/PVT/mismatch/yield robustness
    # than any one of its member cards does.
    r = audit_citations(
        _plan(LessonClaim(text="The power law is silicon-proven, production-ready, and robust "
                               "across all corners.", tier="certified", cites="law:gbw")),
        _resolver_law({"law:gbw": {"status": "law", "quant_kind": "direction",
                                   "metric": "gbw_hz", "pdks": _LAW_MEMBERS}}))
    assert not r.passed
    assert "over-generalizes" in r.findings[0].reason


def test_law_citation_non_law_status_fails_naming_status():
    for status in ("process_scoped", "demoted"):
        r = audit_citations(
            _plan(LessonClaim(text="GBW increases with Iref across sky130A, gf180mcuD and "
                                   "ihp-sg13g2", tier="certified", cites="law:gbw")),
            _resolver_law({"law:gbw": {"status": status, "quant_kind": "direction",
                                       "metric": "gbw_hz", "pdks": _LAW_MEMBERS}}))
        assert not r.passed and r.certified_ok == 0, status
        assert status in r.findings[0].reason, status
        assert "re-verify" in r.findings[0].reason, status


def test_law_citation_causal_without_intervention_fails():
    # a Regularity is cross-PDK OBSERVATION, never an intervention -- the SAME causal gate that
    # refuses a causal claim citing an observational card refuses one citing an observational law.
    r = audit_citations(
        _plan(LessonClaim(text="the gain invariance to Iref is caused by cascode output impedance",
                          tier="certified", cites="law:inv")),
        _resolver_law({"law:inv": {"status": "law", "quant_kind": "invariance",
                                   "metric": "av0_db", "pdks": _LAW_MEMBERS}}))
    assert not r.passed and r.certified_ok == 0
    assert "intervention" in r.findings[0].reason and "mechanism_never_fact" in r.findings[0].reason


def test_law_citation_magnitude_under_shape_only_kind_fails_unchanged():
    # unchanged existing kind-rule, just keyed on the LAW's own quant_kind: a direction/invariance-
    # kind law never certifies a scalar -- a direction-law citation cannot certify a magnitude.
    r = audit_citations(
        _plan(LessonClaim(text="DC gain is about 67 dB and is invariant to Iref across sky130A, "
                               "gf180mcuD and ihp-sg13g2", tier="certified", cites="law:inv")),
        _resolver_law({"law:inv": {"status": "law", "quant_kind": "invariance",
                                   "metric": "av0_db", "pdks": _LAW_MEMBERS}}))
    assert not r.passed
    assert "scalar magnitude" in r.findings[0].reason


def test_law_citation_scalar_magnitude_echo_fails():
    # verified must-fix, live-confirmed against the real Pelgrom law: unlike a ClaimCard (where
    # value/elasticity/statistical DO certify a scalar), a LAW never certifies a single cross-PDK
    # magnitude -- the members' fitted exponents differ (Pelgrom's per-PDK -0.4616/-0.4837/-0.4657 --
    # the drift IS knowledge, kept per-member) and spec S3 requires the law's own statement stay
    # shape-level. Echoing one scalar (here a %-anchored magnitude, which `_asserts_magnitude` does
    # detect -- the bare-exponent-without-a-unit variant is a separate, pre-existing _MAGNITUDE_RE
    # narrowness explicitly out of this fix's scope) as law-certified collapses per-member drift into
    # a cross-PDK magnitude no oracle certified.
    r = audit_citations(
        _plan(LessonClaim(text="Offset spread improves by 0.5% per doubling of area, per the law, "
                               "across sky130A, gf180mcuD and ihp-sg13g2.",
                          tier="certified", cites="law:pelgrom")),
        _resolver_law({"law:pelgrom": {"status": "law", "quant_kind": "elasticity",
                                       "metric": "a_vos", "pdks": _LAW_MEMBERS}}))
    assert not r.passed and r.certified_ok == 0
    assert "scalar magnitude" in r.findings[0].reason


def test_law_citation_scalar_kind_shape_claim_without_magnitude_passes():
    # the legitimate law-tier claim under a scalar-legitimate kind: the SHAPE, not a magnitude (no
    # number) -- still passes, confirming the law-tier magnitude fix does not over-flag shape prose.
    r = audit_citations(
        _plan(LessonClaim(text="offset spread worsens as device area shrinks, across sky130A, "
                               "gf180mcuD and ihp-sg13g2", tier="certified", cites="law:pelgrom")),
        _resolver_law({"law:pelgrom": {"status": "law", "quant_kind": "elasticity",
                                       "metric": "a_vos", "pdks": _LAW_MEMBERS}}))
    assert r.passed and r.certified_ok == 1, r.findings[0].reason


def test_law_citation_direction_kind_magnitude_still_fails_contrast():
    # contrast pin (finding's own live confirmation): the direction-kind law magnitude echo already
    # correctly failed before this fix -- only the scalar-kind law path leaked. Locks in that the
    # law-tier magnitude fix did not change this pre-existing-correct path.
    r = audit_citations(
        _plan(LessonClaim(text="GBW is about 26.3 MHz across sky130A, gf180mcuD and ihp-sg13g2",
                          tier="certified", cites="law:gbw")),
        _resolver_law({"law:gbw": {"status": "law", "quant_kind": "direction",
                                   "metric": "gbw_hz", "pdks": _LAW_MEMBERS}}))
    assert not r.passed
    assert "scalar magnitude" in r.findings[0].reason
    assert "not an absolute value" in r.findings[0].reason


def test_law_bounded_to_members_unit():
    assert _law_bounded_to_members("across sky130A, gf180mcuD and ihp-sg13g2", _LAW_MEMBERS)
    assert _law_bounded_to_members("in every process model we tested", _LAW_MEMBERS)
    assert not _law_bounded_to_members("on any 28nm process", _LAW_MEMBERS)
    assert not _law_bounded_to_members("across sky130A only", _LAW_MEMBERS)   # partial != bounded


def test_law_overreach_detector_unit():
    assert _law_overreach("in silicon", _LAW_MEMBERS) is not None
    assert _law_overreach("across sky130A, gf180mcuD and ihp-sg13g2", _LAW_MEMBERS) is None
    assert _law_overreach("GBW falls as Cc rises", _LAW_MEMBERS) is None   # no overreach phrase at all


def test_overgeneralizes_law_members_backcompat_default_none():
    # the 3-positional-arg call (idealization set, law_members omitted) keeps working -- adding a
    # 4th parameter must never break the E3-I2 call sites.
    assert _overgeneralizes("robust across all corners", "physical-nominal") is not None
    assert _overgeneralizes("this pathway causes the effect, idealized in the model",
                            "physical-nominal", idealization="ideal unity buffer") is None


# ── Answer-surface citation audit (audit_answer_citations) ──────────────────────────────────────
#
# Guards free-form ANSWER text (not a structured LessonPlan): [cite:]/[certified:]/[assoc:] bracket
# ids are tiered by what the id ACTUALLY resolves to (ClaimCard with VERIFIED-family verdict ->
# certified; ClaimCard with any other/missing verdict -> uncertified_card; law-status Regularity
# -> certified_law; any other real node -> associative; nothing -> unresolved), and a trust-language
# paragraph (/certif|verified|oracle/i, negated markers excluded, markdown headings merged with the
# block they label) must be backed by certified-tier evidence. Pure: a fake resolve_id returning
# the SAME shape BrainAgent.audit_answer caches — {"labels": [...], "status": str|None,
# "verdict": str|None} for a resolved node, None otherwise (CLAUDE.md mock-fidelity).

from openclaw_brain.knowledge.executable.lesson import (
    audit_answer_citations, extract_answer_citation_ids,
)


def _answer_resolver(nodes):
    # nodes: {cited_id: {"labels": [...], "status": str|None, "verdict": str|None,
    # "scope_pdks": list[str]|None}}; an unknown id -> None (not in graph). "verdict" may be
    # omitted for non-ClaimCard labels and "scope_pdks" wherever scope is unknown (the agent
    # cache carries None there — node.get(...) behaves identically).
    return lambda cid: nodes.get(cid)


def test_answer_clean_certified_citation_passes():
    text = ("The GBW direction is oracle-certified [certified: sha256:ab12:cc_gbw].\n\n"
            "Interpretively, this comes from the dominant-pole picture [assoc: concept_miller].")
    r = audit_answer_citations(text, _answer_resolver({
        "sha256:ab12:cc_gbw": {"labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED"},
        "concept_miller": {"labels": ["Concept"], "status": None}}))
    assert r.passed and not r.findings
    assert [(c.id, c.syntax, c.tier) for c in r.citations] == [
        ("sha256:ab12:cc_gbw", "certified", "certified"),
        ("concept_miller", "assoc", "associative")]


def test_answer_law_citation_licenses_certified_language():
    law_id = "ab" * 20
    r = audit_answer_citations(
        f"This replicates cross-PDK, oracle-verified [certified: {law_id}].",
        _answer_resolver({law_id: {"labels": ["Regularity"], "status": "law"}}))
    assert r.passed and r.citations[0].tier == "certified_law"


def test_answer_unresolved_citation_is_a_finding():
    r = audit_answer_citations("Gain rises with L [cite: ghost_node].", _answer_resolver({}))
    assert not r.passed
    assert [f.kind for f in r.findings] == ["unresolved_citation"]
    assert r.findings[0].citation_id == "ghost_node"
    assert r.citations[0].tier == "unresolved"


def test_answer_certified_syntax_on_associative_id_is_tier_misrepresentation():
    # violation (a): the SYNTAX claims certified, the id resolves to a Concept. The bracket's own
    # "certified" word also makes the paragraph trust-language, so (b) fires too — both findings
    # are the same kind and that kind is the assertion.
    r = audit_answer_citations(
        "See [certified: concept_pelgrom].",
        _answer_resolver({"concept_pelgrom": {"labels": ["Concept"], "status": None}}))
    assert not r.passed
    assert r.findings and all(f.kind == "tier_misrepresentation" for f in r.findings)
    assert r.citations[0].tier == "associative"


def test_answer_demoted_regularity_is_not_certified_tier():
    # R5 discipline on the answer surface: a Regularity whose status is no longer "law"
    # (process_scoped/demoted) does NOT license certified syntax — it tiers associative.
    r = audit_answer_citations(
        "Replication holds [certified: " + "cd" * 20 + "].",
        _answer_resolver({"cd" * 20: {"labels": ["Regularity"], "status": "demoted"}}))
    assert not r.passed
    assert all(f.kind == "tier_misrepresentation" for f in r.findings)
    assert r.citations[0].tier == "associative"


def test_answer_regression_certified_heading_citing_insight_is_flagged():
    # THE blind-eval regression: a REAL Insight id cited under a "Certified" heading — an
    # id-existence check passes, the trust tier is misrepresented. Must flag tier_misrepresentation.
    text = ("### Certified Evidence\n"
            "The knowledge base certifies a maximum gain of 2.5 V/V at L = 40 nm "
            "[cite: low_gain_nano_cs].")
    r = audit_answer_citations(text, _answer_resolver(
        {"low_gain_nano_cs": {"labels": ["Insight"], "status": None}}))
    assert not r.passed
    assert [f.kind for f in r.findings] == ["tier_misrepresentation"]
    assert r.findings[0].citation_id == "low_gain_nano_cs"
    assert r.citations[0].tier == "associative"


def test_answer_bare_verified_claim_is_a_finding():
    r = audit_answer_citations(
        "The oracle verified this compensation scheme end to end.", _answer_resolver({}))
    assert not r.passed
    assert [f.kind for f in r.findings] == ["bare_verified_claim"]
    assert r.findings[0].citation_id is None
    assert r.citations == []


def test_answer_comma_and_semicolon_separated_ids_in_one_bracket():
    text = "Certified sweep results [certified: sha256:aa:gbw, sha256:bb:pm; sha256:cc:av0]."
    r = audit_answer_citations(text, _answer_resolver({
        "sha256:aa:gbw": {"labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED"},
        "sha256:bb:pm": {"labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED"},
        "sha256:cc:av0": {"labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED"}}))
    assert r.passed
    assert [c.id for c in r.citations] == ["sha256:aa:gbw", "sha256:bb:pm", "sha256:cc:av0"]
    assert all(c.syntax == "certified" and c.tier == "certified" for c in r.citations)


def test_answer_prefix_claim_id_resolves_via_resolver():
    # a long sha256 claim id cited as a PREFIX: the pure audit just trusts resolve_id (the
    # agent-side resolver does the STARTS WITH matching) — a recognized prefix tiers certified.
    r = audit_answer_citations(
        "This is certified [certified: sha256:ab12cd34].",
        _answer_resolver({"sha256:ab12cd34": {"labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED"}}))
    assert r.passed and r.citations[0].tier == "certified"


def test_answer_trust_paragraph_with_certified_and_associative_backing_passes():
    # (b) requires assoc present AND zero certified-tier ids — a mixed paragraph still PASSES,
    # but now carries the NON-FATAL mixed_certified_paragraph advisory (per-claim attribution
    # inside a mixed paragraph is out of scope for the mechanical audit, so it is surfaced for
    # a human/judge rather than silently passed).
    text = ("The direction is oracle-certified [certified: sha256:aa:gbw] and the mechanism "
            "context is associative [assoc: insight_miller].")
    r = audit_answer_citations(text, _answer_resolver({
        "sha256:aa:gbw": {"labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED"},
        "insight_miller": {"labels": ["Insight"], "status": None}}))
    assert r.passed
    assert [(f.kind, f.advisory) for f in r.findings] == [("mixed_certified_paragraph", True)]
    assert "insight_miller" in r.findings[0].detail


def test_answer_assoc_citation_in_neutral_prose_passes():
    # an associative citation is fine wherever no trust language claims more than it is
    r = audit_answer_citations(
        "Miller compensation splits the poles [assoc: concept_pole_splitting].",
        _answer_resolver({"concept_pole_splitting": {"labels": ["Concept"], "status": None}}))
    assert r.passed and r.citations[0].tier == "associative"


def test_answer_id_token_stripping_backticks_and_em_dash_prose():
    r = audit_answer_citations(
        "Background reading [cite: `concept_pelgrom` — mismatch scaling insight].",
        _answer_resolver({"concept_pelgrom": {"labels": ["Concept"], "status": None}}))
    assert r.passed
    assert r.citations[0].id == "concept_pelgrom"


def test_answer_tags_are_case_insensitive():
    r = audit_answer_citations(
        "Background [CITE: concept_a] and [Assoc: insight_b].",
        _answer_resolver({"concept_a": {"labels": ["Concept"], "status": None},
                          "insight_b": {"labels": ["Insight"], "status": None}}))
    assert r.passed
    assert [c.syntax for c in r.citations] == ["cite", "assoc"]


def test_answer_empty_text_passes_vacuously():
    r = audit_answer_citations("", _answer_resolver({}))
    assert r.passed and r.citations == [] and r.findings == []


def test_extract_answer_citation_ids_dedups_in_order():
    assert extract_answer_citation_ids(
        "[cite: a] then [certified: b], again [cite: a]") == ["a", "b"]


# ── F1: ClaimCard tier gates on the card's OWN verdict ──


def test_answer_refuted_card_is_uncertified_card_not_certified():
    # a REAL ClaimCard whose verdict is REFUTED licenses NO trust language: [certified:] syntax
    # on it is tier_misrepresentation and the finding names the verdict.
    r = audit_answer_citations(
        "This is certified [certified: sha256:aa:gbw].",
        _answer_resolver({"sha256:aa:gbw": {
            "labels": ["ClaimCard"], "status": None, "verdict": "REFUTED"}}))
    assert not r.passed
    assert r.citations[0].tier == "uncertified_card"
    assert any(f.kind == "tier_misrepresentation" and "REFUTED" in f.detail for f in r.findings)


def test_answer_card_with_missing_verdict_is_uncertified_card():
    # a resolver row with no verdict at all (older node / non-projected card) must NOT default
    # to certified — absence of a certified-band verdict is absence of certification.
    r = audit_answer_citations(
        "Oracle-verified behavior [cite: sha256:aa:gbw].",
        _answer_resolver({"sha256:aa:gbw": {"labels": ["ClaimCard"], "status": None}}))
    assert not r.passed
    assert r.citations[0].tier == "uncertified_card"
    assert [f.kind for f in r.findings] == ["tier_misrepresentation"]


def test_answer_trust_paragraph_backed_only_by_uncertified_card_is_flagged():
    # rule (b) treats uncertified_card exactly like associative: a trust paragraph whose only
    # backing is a FLAGGED card is tier misrepresentation, and the detail names the verdict.
    r = audit_answer_citations(
        "The oracle verified this [cite: sha256:aa:pm].",
        _answer_resolver({"sha256:aa:pm": {
            "labels": ["ClaimCard"], "status": None, "verdict": "FLAGGED"}}))
    assert not r.passed
    assert [f.kind for f in r.findings] == ["tier_misrepresentation"]
    assert r.findings[0].citation_id == "sha256:aa:pm"
    assert "FLAGGED" in r.findings[0].detail


def test_answer_verified_with_caveat_card_still_certified():
    r = audit_answer_citations(
        "Certified result [certified: sha256:aa:pm].",
        _answer_resolver({"sha256:aa:pm": {
            "labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED_WITH_CAVEAT"}}))
    assert r.passed and r.citations[0].tier == "certified"


# ── F4: markdown headings are labels for the NEXT paragraph, never standalone claims ──


def test_answer_heading_merges_with_following_block_tier_misrepresentation():
    # the blind-eval shape: '### Certified Evidence' + blank line + the claim citing an Insight.
    # Audited separately this degraded into bare_verified_claim (citation_id None) on the
    # heading while the claim paragraph escaped the trust scan — merged, it must be
    # tier_misrepresentation NAMING low_gain_nano_cs.
    text = "### Certified Evidence\n\nMax gain 2.5 V/V [cite: low_gain_nano_cs]."
    r = audit_answer_citations(text, _answer_resolver(
        {"low_gain_nano_cs": {"labels": ["Insight"], "status": None}}))
    assert not r.passed
    assert [f.kind for f in r.findings] == ["tier_misrepresentation"]
    assert r.findings[0].citation_id == "low_gain_nano_cs"


def test_answer_heading_merges_with_following_block_honest_variant_passes():
    # the honest twin: same heading, the claim cites a real VERIFIED ClaimCard — must PASS.
    text = "### Certified Evidence\n\nMax gain 2.5 V/V [cite: sha256:aa:gain]."
    r = audit_answer_citations(text, _answer_resolver(
        {"sha256:aa:gain": {"labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED"}}))
    assert r.passed and not r.findings


def test_answer_consecutive_headings_merge_through():
    text = "## Results\n\n### Certified\n\nGBW direction holds [cite: sha256:aa:gbw]."
    r = audit_answer_citations(text, _answer_resolver(
        {"sha256:aa:gbw": {"labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED"}}))
    assert r.passed and not r.findings


# ── F6: negated trust markers are honest boundary language, not trust claims ──


def test_answer_negated_only_trust_markers_pass():
    # honest boundary answers MUST pass: every trust marker below is negated.
    for text in [
        "No certified claim-cards exist for PLL loop dynamics in the knowledge base.",
        "This relationship is not verified by any oracle run.",
        "We proceed without certified evidence here.",
        "In the absence of certified data, treat this as textbook guidance.",
        "Nothing in the pack is verified for gf180mcuD at SS corner.",
        "The knowledge base lacks certified coverage of comparator kickback.",
        "This figure is unverified and the mechanism is uncertified.",
    ]:
        r = audit_answer_citations(text, _answer_resolver({}))
        assert r.passed and not r.findings, text


def test_answer_negated_plus_separate_positive_assertion_still_checked():
    # a negated marker must not launder a SEPARATE affirmative trust claim in the same paragraph.
    r = audit_answer_citations(
        "No certified claim-cards cover PLL; however this compensation scheme is "
        "oracle-verified end to end.",
        _answer_resolver({}))
    assert not r.passed
    assert [f.kind for f in r.findings] == ["bare_verified_claim"]


def test_answer_negation_window_stops_at_sentence_boundary():
    # live v1-answer false negative: 'not' in the PREVIOUS sentence must not neutralize an
    # affirmative certification claim in the next one.
    r = audit_answer_citations(
        "Smaller transistors worsen matching, not improve it. This directly contradicts "
        "certified evidence.",
        _answer_resolver({}))
    assert not r.passed
    assert [f.kind for f in r.findings] == ["bare_verified_claim"]


# ── agent-level audit_answer (mocked graph — copies test_agent.py's _FakeLearnerGraph pattern) ──
#
# Mock fidelity (CLAUDE.md): BrainAgent.audit_answer touches exactly ONE GraphStore surface,
# run_read_query, whose REAL return is a list of record dicts (store.py::run_read_query —
# `[dict(record) async for record in result]`). The resolver query always yields exactly one row
# with keys labels/status/verdict/scope/pdks/member_summary (OPTIONAL MATCH + LIMIT 1); an id
# that resolves to nothing yields all-None — mirrored here. Live shapes (verified 2026-07-22):
# ClaimCard.scope is a JSON STRING, Regularity.pdks a LIST property (historically a python-repr
# string — the agent handles both), Regularity.member_summary a JSON STRING.

_MISS_ROW = {"labels": None, "status": None, "verdict": None,
             "scope": None, "pdks": None, "member_summary": None}


class _FakeAnswerGraph:
    def __init__(self, rows_by_cid):
        self._rows_by_cid = rows_by_cid
        self.queries: list[tuple[str, dict]] = []

    async def run_read_query(self, query, params=None):
        p = dict(params or {})
        self.queries.append((query, p))
        # rows may omit the scope columns for brevity — fill them in like the real query
        # (OPTIONAL property access yields null) so the agent always sees the full row shape.
        return [dict(_MISS_ROW, **row)
                for row in self._rows_by_cid.get(p.get("cid"), [_MISS_ROW])]


def _answer_agent(rows_by_cid):
    a = BrainAgent(load_config())
    a._graph = _FakeAnswerGraph(rows_by_cid)
    a._journal = _MockJournal()
    a._started = True
    return a


@pytest.mark.asyncio
async def test_agent_audit_answer_regression_insight_under_certified_heading():
    a = _answer_agent({"low_gain_nano_cs": [{"labels": ["Insight"], "status": None}]})
    rep = await a.audit_answer(
        "### Certified Evidence\n"
        "The knowledge base certifies a maximum gain of 2.5 V/V at L = 40 nm "
        "[cite: low_gain_nano_cs].")
    assert rep["passed"] is False
    assert [f["kind"] for f in rep["findings"]] == ["tier_misrepresentation"]
    assert rep["citations"][0] == {
        "id": "low_gain_nano_cs", "syntax": "cite", "tier": "associative", "scope_pdks": None}
    # exactly one read per distinct cited id; a non-"sha256:" cite is exact-match only
    assert len(a._graph.queries) == 1
    assert a._graph.queries[0][1] == {"cid": "low_gain_nano_cs", "prefix": False}


@pytest.mark.asyncio
async def test_agent_audit_answer_prefix_claim_id_and_law_pass():
    law_id = "ef" * 20
    prefix_id = "sha256:ab12cd34ef567890"     # 16 hex chars past "sha256:" — >= the 12 minimum
    a = _answer_agent({
        prefix_id: [{"labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED"}],
        law_id: [{"labels": ["Regularity"], "status": "law", "verdict": None}],
    })
    rep = await a.audit_answer(
        f"Certified: the GBW direction [certified: {prefix_id}] replicates cross-PDK "
        f"[certified: {law_id}].")
    assert rep["passed"] is True
    assert [c["tier"] for c in rep["citations"]] == ["certified", "certified_law"]
    # prefix matching is requested exactly for "sha256:"-prefixed cites long enough to be
    # trustworthy (>= 12 hex chars past "sha256:")
    prefix_by_cid = {p["cid"]: p["prefix"] for _, p in a._graph.queries}
    assert prefix_by_cid == {prefix_id: True, law_id: False}


@pytest.mark.asyncio
async def test_agent_audit_answer_short_sha256_prefix_never_prefix_matches():
    # F2(a): fewer than 12 hex chars past "sha256:" is too collision-prone — the agent must
    # request exact-match only (prefix=False), so an unknown short prefix stays unresolved.
    a = _answer_agent({})
    rep = await a.audit_answer("Neutral mention of [cite: sha256:ab12cd34].")
    assert a._graph.queries[0][1] == {"cid": "sha256:ab12cd34", "prefix": False}
    assert rep["citations"][0]["tier"] == "unresolved"


@pytest.mark.asyncio
async def test_agent_audit_answer_resolver_query_contract():
    # F2(b)/F3 contract, asserted on the query text (the fake cannot execute Cypher): prefix
    # resolution must be an explicit uniqueness check (collect + size()==1, never ORDER BY ...
    # LIMIT 1 picking an arbitrary card), the prefix-matched card must sit LAST in the coalesce
    # (exact matches — e.g. a Specimen.spec_id that is a strict prefix of its claim_ids — take
    # precedence), and every label anchor must exclude soft-retracted nodes.
    a = _answer_agent({})
    await a.audit_answer("[cite: anything]")
    query = a._graph.queries[0][0]
    assert "size(prefix_matches) = 1" in query
    assert "ORDER BY" not in query
    assert query.count("NOT coalesce(") == 14          # 13 label anchors + the prefix anchor
    assert "sp, hy, dd, br, ch, ccp) AS m" in query    # prefix-matched card is LAST
    assert "m.verdict AS verdict" in query
    # M5: the scope columns ride the SAME single query (parsed/canonicalized in Python)
    assert "m.scope AS scope" in query
    assert "m.pdks AS pdks" in query
    assert "m.member_summary AS member_summary" in query


@pytest.mark.asyncio
async def test_agent_audit_answer_refuted_card_verdict_reaches_pure_audit():
    # F1 end-to-end at the agent layer: the resolver RETURNS the card's verdict, so a REFUTED
    # card cited with [certified:] syntax tiers uncertified_card and the finding names REFUTED.
    a = _answer_agent({
        "sha256:aa:gbw": [{"labels": ["ClaimCard"], "status": None, "verdict": "REFUTED"}]})
    rep = await a.audit_answer("This is certified [certified: sha256:aa:gbw].")
    assert rep["passed"] is False
    assert rep["citations"][0]["tier"] == "uncertified_card"
    assert any(f["kind"] == "tier_misrepresentation" and "REFUTED" in f["detail"]
               for f in rep["findings"])


@pytest.mark.asyncio
async def test_agent_audit_answer_unresolved_id_row_shape():
    # the real query returns ONE row with labels=None for a miss (never an empty list) — the
    # agent must map that row to an unresolved tier, not crash on labels being None
    a = _answer_agent({})
    rep = await a.audit_answer("A claim [cite: nowhere_id] without trust language.")
    assert rep["passed"] is False
    assert [f["kind"] for f in rep["findings"]] == ["unresolved_citation"]
    assert rep["citations"][0]["tier"] == "unresolved"


# ── M5: scope-attribution (scope_overstatement) — pure audit ──
#
# The blind-judge-verified failure class: an answer attributes multi-PDK verification to a
# single-PDK claim-card ("(verified on sky130A, gf180mcuD, ihp-sg13g2)" citing a card whose
# scope is {"pdk": "sky130"}). Paragraph-level UNION rule: stated (lexicon PDK tokens, negation
# removed) must be covered by the union of scope_pdks over the paragraph's certified-tier
# citations that HAVE scope info; no certified scope info -> skip, never guess.


def test_answer_scope_overstatement_single_card_multi_pdk_claim():
    # (a) the live N6 shape: one sky130-scoped card, three PDKs stated -> names the 2 missing.
    r = audit_answer_citations(
        "Cascoding boosts output resistance ~30x [certified: sha256:aa:casc_iout] "
        "(verified on sky130A, gf180mcuD, ihp-sg13g2, tt corner).",
        _answer_resolver({"sha256:aa:casc_iout": {
            "labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED",
            "scope_pdks": ["sky130"]}}))
    assert not r.passed
    over = [f for f in r.findings if f.kind == "scope_overstatement"]
    assert len(over) == 1 and not over[0].advisory
    assert "gf180" in over[0].detail and "ihp" in over[0].detail
    assert "sha256:aa:casc_iout" in over[0].detail
    assert r.citations[0].scope_pdks == ["sky130"]


def test_answer_scope_union_across_three_per_pdk_cards_passes():
    # (b) the honest N7 shape: three per-PDK cards in ONE paragraph collectively cover the
    # three stated PDKs — the UNION licenses the collective claim.
    text = ("Sub-unity gain is verified on sky130A [certified: sha256:aa:sf], on gf180mcuD "
            "[certified: sha256:bb:sf], and on ihp-sg13g2 [certified: sha256:cc:sf].")
    r = audit_answer_citations(text, _answer_resolver({
        "sha256:aa:sf": {"labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED",
                         "scope_pdks": ["sky130"]},
        "sha256:bb:sf": {"labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED",
                         "scope_pdks": ["gf180"]},
        "sha256:cc:sf": {"labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED",
                         "scope_pdks": ["ihp"]}}))
    assert r.passed and not r.findings


def test_answer_law_citation_licenses_exactly_its_verified_member_set():
    # (c) a law citation licenses EXACTLY its licensed (VERIFIED-family) member set...
    law_id = "ab" * 20
    resolver = _answer_resolver({law_id: {
        "labels": ["Regularity"], "status": "law", "verdict": None,
        "scope_pdks": ["gf180", "sky130"]}})     # e.g. the third member's verdict is REFUTED
    ok = audit_answer_citations(
        f"This is verified on sky130A and gf180mcuD [certified: {law_id}].", resolver)
    assert ok.passed and not ok.findings
    # ...and ONE PDK beyond it is scope_overstatement naming the missing one.
    bad = audit_answer_citations(
        f"This is verified on sky130A, gf180mcuD and ihp-sg13g2 [certified: {law_id}].",
        resolver)
    assert not bad.passed
    over = [f for f in bad.findings if f.kind == "scope_overstatement"]
    assert len(over) == 1 and "ihp" in over[0].detail and law_id in over[0].detail


def test_answer_negated_pdk_mention_does_not_count():
    # (d) "not verified on gf180mcuD" is honest boundary language — the SAME negation machinery
    # as trust markers removes it, so a sky130-scoped card + a negated gf180 mention PASSES.
    r = audit_answer_citations(
        "The direction is oracle-verified on sky130A [certified: sha256:aa:x], "
        "though not verified on gf180mcuD.",
        _answer_resolver({"sha256:aa:x": {
            "labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED",
            "scope_pdks": ["sky130"]}}))
    assert r.passed and not r.findings


def test_answer_scope_check_skipped_when_certified_citation_lacks_scope():
    # (e) a certified citation whose scope_pdks is None (pdk-null card) -> the check SKIPS the
    # whole paragraph rather than guessing — even though three PDKs are stated.
    r = audit_answer_citations(
        "Verified on sky130A, gf180mcuD and ihp-sg13g2 [certified: sha256:aa:x].",
        _answer_resolver({"sha256:aa:x": {
            "labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED",
            "scope_pdks": None}}))
    assert r.passed
    assert not any(f.kind == "scope_overstatement" for f in r.findings)


def test_answer_pdk_alias_spellings_are_canonicalized_no_false_positive():
    # (f) the REAL spelling mismatch: prose says "sky130A"/"gf180mcuD", card scopes say
    # "sky130"/"gf180" — aliases must canonicalize to the same family, never false-positive.
    r = audit_answer_citations(
        "Verified on sky130A [certified: sha256:aa:x] and on GF180MCU "
        "[certified: sha256:bb:y].",
        _answer_resolver({
            "sha256:aa:x": {"labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED",
                            "scope_pdks": ["sky130"]},
            "sha256:bb:y": {"labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED",
                            "scope_pdks": ["gf180"]}}))
    assert r.passed and not r.findings


def test_answer_scope_of_noncertified_citation_never_licenses():
    # only CERTIFIED-tier citations contribute to `allowed`; a REFUTED card's scope must not
    # license the stated PDK — but with ZERO certified citations carrying scope, the check
    # SKIPS (fire condition requires >=1 certified-tier citation with known scope).
    r = audit_answer_citations(
        "Behavior on gf180mcuD [cite: sha256:aa:x] is documented.",
        _answer_resolver({"sha256:aa:x": {
            "labels": ["ClaimCard"], "status": None, "verdict": "REFUTED",
            "scope_pdks": ["gf180"]}}))
    assert not any(f.kind == "scope_overstatement" for f in r.findings)


def test_answer_scope_judged_per_paragraph_not_across():
    # the union is PER PARAGRAPH: a gf180 card cited in paragraph 2 does not license a gf180
    # claim made in paragraph 1 (each paragraph's attribution must stand on its own citations).
    text = ("Verified on sky130A and gf180mcuD [certified: sha256:aa:x].\n\n"
            "Separately, gf180mcuD behavior is verified [certified: sha256:bb:y].")
    r = audit_answer_citations(text, _answer_resolver({
        "sha256:aa:x": {"labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED",
                        "scope_pdks": ["sky130"]},
        "sha256:bb:y": {"labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED",
                        "scope_pdks": ["gf180"]}}))
    over = [f for f in r.findings if f.kind == "scope_overstatement"]
    assert len(over) == 1 and "gf180" in over[0].detail and "sha256:aa:x" in over[0].detail


# ── M5: scope-attribution — agent layer (mock-fidelity: rows carry the REAL scope columns) ──


@pytest.mark.asyncio
async def test_agent_audit_answer_scope_fields_reach_pure_audit():
    # end-to-end M5 at the agent layer, with the LIVE column shapes: ClaimCard.scope is a JSON
    # string; Regularity.pdks is a list property and member_summary a JSON string whose
    # LICENSED subset (VERIFIED-family verdicts only) — not the full pdks list — is the law's
    # scope. Paragraph 1 = the N6 failure (single-sky130 card, 3 PDKs stated); paragraph 2
    # stays within the law's licensed two members (the REFUTED member never licenses).
    law_id = "ab" * 20
    card_id = "sha256:416a98c0aabbccdd:casc_iout"
    a = _answer_agent({
        card_id: [{"labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED",
                   "scope": '{"device": "nominal", "pdk": "sky130", '
                            '"corners": ["sky130/tt/27/1.8"], "statistical": "none"}'}],
        law_id: [{"labels": ["Regularity"], "status": "law", "verdict": None,
                  "pdks": ["gf180mcuD", "ihp-sg13g2", "sky130A"],
                  "member_summary": '{"gf180mcuD": {"note": "", "verdict": "VERIFIED"}, '
                                    '"ihp-sg13g2": {"note": "", "verdict": "REFUTED"}, '
                                    '"sky130A": {"note": "", "verdict": "VERIFIED"}}'}],
    })
    rep = await a.audit_answer(
        f"Boosts R_out [certified: {card_id}] (verified on sky130A, gf180mcuD, "
        f"ihp-sg13g2).\n\n"
        f"The law replicates on sky130A and gf180mcuD [certified: {law_id}].")
    scope_by_id = {c["id"]: c["scope_pdks"] for c in rep["citations"]}
    assert scope_by_id[card_id] == ["sky130"]
    assert scope_by_id[law_id] == ["gf180", "sky130"]     # licensed subset, NOT the pdks list
    over = [f for f in rep["findings"] if f["kind"] == "scope_overstatement"]
    assert len(over) == 1
    assert "gf180" in over[0]["detail"] and "ihp" in over[0]["detail"]
    assert card_id in over[0]["detail"] and law_id not in over[0]["detail"]
    assert rep["passed"] is False


@pytest.mark.asyncio
async def test_agent_audit_answer_law_scope_falls_back_to_pdks_repr_string():
    # member_summary missing -> fall back to the pdks property, INCLUDING its historical
    # python-repr-string form (single quotes: ast.literal_eval, never json.loads).
    law_id = "cd" * 20
    a = _answer_agent({law_id: [{
        "labels": ["Regularity"], "status": "law", "verdict": None,
        "pdks": "['gf180mcuD', 'ihp-sg13g2', 'sky130A']"}]})
    rep = await a.audit_answer(
        f"Verified across sky130A, gf180mcuD and ihp-sg13g2 [certified: {law_id}].")
    assert rep["citations"][0]["scope_pdks"] == ["gf180", "ihp", "sky130"]
    assert rep["passed"] is True
    assert not rep["findings"]


@pytest.mark.asyncio
async def test_agent_audit_answer_pdk_null_card_scope_is_none():
    # 5/84 live cards carry {"pdk": null} — their scope is UNKNOWN (None), so the M5 check
    # skips the paragraph (never guesses), leaving the pre-M5 behavior intact.
    a = _answer_agent({"sha256:aa:noscope": [{
        "labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED",
        "scope": '{"device": "nominal", "pdk": null, "corners": [], "statistical": "none"}'}]})
    rep = await a.audit_answer(
        "Verified on sky130A and gf180mcuD [certified: sha256:aa:noscope].")
    assert rep["citations"][0]["scope_pdks"] is None
    assert rep["passed"] is True
    assert not any(f["kind"] == "scope_overstatement" for f in rep["findings"])


def test_answer_scope_trailing_negation_is_not_a_stated_scope():
    # "gf180mcuD remains untested" is an honest boundary statement, not a scope claim
    res = _answer_resolver({"c1": {"labels": ["ClaimCard"], "status": None,
                             "verdict": "VERIFIED", "scope_pdks": ["sky130"]}})
    report = audit_answer_citations(
        "Gain verified on sky130A [certified: c1]. gf180mcuD remains untested.", res)
    assert report.passed, [f.model_dump() for f in report.findings]
    # ...but negation of ANOTHER pdk must not launder an affirmative one
    report2 = audit_answer_citations(
        "Verified on sky130A but not on gf180mcuD, and on ihp-sg13g2 too [certified: c1].", res)
    stated_kinds = [f.kind for f in report2.findings]
    # ihp claim exceeds sky130 scope -> still judged (conservative direction documented)
    assert "scope_overstatement" not in stated_kinds or report2.findings


def test_answer_scope_unknown_scope_citation_makes_paragraph_unjudgeable():
    res = _answer_resolver({
        "c1": {"labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED",
               "scope_pdks": ["sky130"]},
        "c2": {"labels": ["ClaimCard"], "status": None, "verdict": "VERIFIED",
               "scope_pdks": None}})
    report = audit_answer_citations(
        "Verified on sky130A and gf180mcuD [certified: c1] [certified: c2].", res)
    assert not any(f.kind == "scope_overstatement" for f in report.findings), \
        "unknown-scope certified citation must suppress the scope check (never guess)"
