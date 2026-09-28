"""Digital stimulus-coverage scope axis (ADR-041 corollary c): a functional verdict declares the
stimulus class it exercised and names the dominant UNTESTED stimulus class — analog<->digital symmetry."""
from openclaw_brain.knowledge.executable.conditions import DigitalUnits, summarize_scope


def test_digital_scope_renders_declared_stimulus():
    s = summarize_scope(DigitalUnits(bit_width=8, stimulus="count 0->15, one wrap"))
    assert s["stimulus"] == "count 0->15, one wrap"
    assert s["functional"] is True                 # unchanged axis preserved


def test_digital_scope_stimulus_defaults_unspecified():
    s = summarize_scope(DigitalUnits(bit_width=8))
    assert s["stimulus"] == "unspecified"
    assert s["functional"] is True


from openclaw_brain.knowledge.executable.corpus import SpecimenCorpus
from openclaw_brain.knowledge.executable.executor import run_recipe
from openclaw_brain.knowledge.executable.models import (
    ClaimCard, MechanismClaim, QuantTest, VerificationRecipe, VerdictClass)


class _FakeDigital:
    def measure(self, deck, timeout=120):
        return {"ped": [(0.0, 16.0), (64.0, 16.0), (128.0, 16.0), (192.0, 16.0)]}


def _digital_recipe(tclass, cond):
    return VerificationRecipe(
        topology_class=tclass, build={"method": "template", "template_ref": "digital_cds_tb", "engine": "iverilog"},
        conditions=cond,
        sweeps=[{"analysis": "tran", "knob": "ped", "points": ["0", "64", "128", "192"], "measure": ["ped"]}],
        claim_cards=[ClaimCard(id="c", topology_class=tclass, conditions=cond,
            mechanism=MechanismClaim(knob="ped", metric="ped", series_ref="c",
                quant=QuantTest(kind="invariance", cov_max=0.001), narrative="functional"))])


def test_executor_stamps_declared_stimulus_untested(tmp_path):
    cond = DigitalUnits(bit_width=8, stimulus="ramp full range",
                        stimulus_untested="counter wrap at ramp end")
    res = run_recipe(_digital_recipe("ss_adc_digital_backend", cond),
                     _FakeDigital(), corpus=SpecimenCorpus(str(tmp_path)), project=False)
    assert res.claim_cards[0].dominant_risk_untested == "counter wrap at ramp end"


def test_executor_preserves_existing_digital_banner_when_unspecified(tmp_path):
    # digital_cds_nmos HAS a hand-written _DOMINANT_RISK banner; a recipe that declares no
    # stimulus_untested must keep it (or-fallback), not lose it to silent None (defect §2.1).
    cond = DigitalUnits(bit_width=8)   # no stimulus_untested
    res = run_recipe(_digital_recipe("digital_cds_nmos", cond),
                     _FakeDigital(), corpus=SpecimenCorpus(str(tmp_path)), project=False)
    risk = res.claim_cards[0].dominant_risk_untested
    assert risk is not None and "FUNCTIONAL model omits" in risk


from openclaw_brain.knowledge.executable.lesson import _overgeneralizes


def test_refuses_stimulus_completeness_overgeneralization():
    r = _overgeneralizes("this counter works for all inputs", "functional")
    assert r is not None and "un-exercised stimulus" in r
    assert _overgeneralizes("verified exhaustively over every sequence", "functional") is not None


def test_does_not_flag_legit_finite_driven_claim():
    # a finite exhaustive description is NOT over-generalization for a functional sim (defect §2.3)
    assert _overgeneralizes("handles all 256 codes it was driven with", "functional") is None


from openclaw_brain.knowledge.executable.seeds import (
    _digital_cds_recipe, _gray_recipe, _ss_adc_backend_recipe)


def test_digital_recipes_own_distinct_stimulus():
    dcds = _digital_cds_recipe()
    ssadc = _ss_adc_backend_recipe()
    gray = _gray_recipe()
    # each declares a stimulus, and DCDS != SSADC (proves no shared-singleton mutation, §2.2)
    assert dcds.conditions.stimulus and ssadc.conditions.stimulus and gray.conditions.stimulus
    assert dcds.conditions.stimulus != ssadc.conditions.stimulus
    assert dcds.conditions is not ssadc.conditions          # distinct instances
    # the claim cards see the same instance as their recipe
    assert dcds.claim_cards[0].conditions is dcds.conditions


from openclaw_brain.agent import _scope_inline


def test_scope_inline_appends_digital_stimulus():
    s = {"functional": True, "statistical": "n/a", "corners": [], "stimulus": "count 0->15, one wrap"}
    assert _scope_inline(s) == "functional/stimulus:count 0->15, one wrap"   # no stray '/n/a'


def test_scope_inline_omits_unspecified_stimulus():
    s = {"functional": True, "statistical": "n/a", "corners": [], "stimulus": "unspecified"}
    assert _scope_inline(s) == "functional"


def test_scope_inline_analog_unchanged():
    s = {"corners": ["sky130/tt_mm/27/1.8"], "statistical": "3σ@200"}
    assert _scope_inline(s) == "sky130/tt_mm/27/1.8/3σ@200"
