"""C1 + I1 (adversarial-review must-fixes): a degenerate / truncated / empty measured series must
FLAG the claim, never crash the run or certify a partial curve. Covers oracle.judge_quant guards
and the executor's count-guard + broadened exception handling. No docker / no model."""

from __future__ import annotations

import pytest

from openclaw_brain.knowledge.executable.executor import run_recipe
from openclaw_brain.knowledge.executable.models import (
    ClaimCard, AnalogPVT, MechanismClaim, QuantTest, VerdictClass, VerificationRecipe,
)
from openclaw_brain.knowledge.executable.oracle import judge_quant

OTA = "miller_ota_2stage_nmos_in"
COND = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)


# ── oracle.judge_quant degenerate-input guards ──


def test_judge_quant_flags_empty_invariance_series():
    v, note = judge_quant(QuantTest(kind="invariance", cov_max=0.1), [])
    assert v == VerdictClass.FLAGGED          # was ValueError: max([]) -> crash


def test_judge_quant_flags_value_kind_given_a_series():
    # canonical only ever stores (x,y) lists; a value-kind claim must FLAG, not float(list) -> TypeError
    v, note = judge_quant(QuantTest(kind="value", value=1.0, tol=0.1), [(1.0, 2.0)])
    assert v == VerdictClass.FLAGGED


def test_judge_quant_flags_single_point_direction():
    v, note = judge_quant(QuantTest(kind="direction", sign="+"), [(1.0, 5.0)])
    assert v == VerdictClass.FLAGGED          # n<2 cannot establish a trend


# ── I6: oracle soundness (direction zero-trend asymmetry; dB invariance) ──


def test_direction_flat_series_flags_both_signs():
    # a non-responsive (flat) metric satisfies NO direction claim — copysign(+0.0) must not certify '+'
    flat = [(1.0, 5.0), (2.0, 5.0), (3.0, 5.0)]
    assert judge_quant(QuantTest(kind="direction", sign="+"), flat)[0] == VerdictClass.FLAGGED
    assert judge_quant(QuantTest(kind="direction", sign="-"), flat)[0] == VerdictClass.FLAGGED


def test_direction_real_trend_still_verifies():
    rising = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    assert judge_quant(QuantTest(kind="direction", sign="+"), rising)[0] == VerdictClass.VERIFIED


def test_invariance_spread_max_scale_aware():
    # av0_db ~69 dB flat: a 0.5 dB spread bound is meaningful; a fractional CoV would be ~3x looser
    flat_db = [(1.0, 68.9), (2.0, 69.0), (3.0, 68.8)]
    assert judge_quant(QuantTest(kind="invariance", spread_max=0.5), flat_db)[0] == VerdictClass.VERIFIED
    wide_db = [(1.0, 68.0), (2.0, 72.0)]              # 4 dB spread -> refuted at 0.5 dB
    assert judge_quant(QuantTest(kind="invariance", spread_max=0.5), wide_db)[0] == VerdictClass.REFUTED


def test_invariance_requires_a_bound():
    with pytest.raises(ValueError):
        QuantTest(kind="invariance")                 # neither cov_max nor spread_max
    QuantTest(kind="invariance", spread_max=0.5)      # spread_max alone is valid


# ── executor count-guard + no-crash ──


def _ota_recipe(points, claim_kind="direction"):
    quant = (QuantTest(kind="direction", sign="-") if claim_kind == "direction"
             else QuantTest(kind="invariance", cov_max=0.1))
    return VerificationRecipe(
        topology_class=OTA, build={"method": "template", "template_ref": "miller_ota_ac"},
        conditions=COND,
        sweeps=[{"analysis": "ac", "knob": "Cc", "points": points, "measure": ["gbw_hz"]}],
        claim_cards=[ClaimCard(id="cc_gbw", topology_class=OTA, conditions=COND,
                               mechanism=MechanismClaim(knob="Cc", metric="gbw_hz", series_ref="x",
                                                        quant=quant, narrative="..."))],
    )


class _Runner:
    def __init__(self, series):
        self._series = series

    def measure(self, deck, timeout=300):
        return {"cc": self._series}


def test_run_recipe_flags_truncated_series():
    # recipe asked for 3 points; the sim returned only 2 (a .meas failed at one) -> FLAG, don't judge
    recipe = _ota_recipe(["500f", "1000f", "2000f"])
    result = run_recipe(recipe, _Runner([(500e-15, 48e6), (1000e-15, 24e6)]), project=False)
    c = result.claim_cards[0]
    assert c.verdict == VerdictClass.FLAGGED
    assert "truncated" in (c.verdict_note or "")


def test_run_recipe_flags_empty_series():
    recipe = _ota_recipe(["500f", "1000f", "2000f"])
    result = run_recipe(recipe, _Runner([]), project=False)
    assert result.claim_cards[0].verdict == VerdictClass.FLAGGED


def test_run_recipe_full_series_still_judged():
    # control: a complete series is judged normally (gbw falls with Cc -> VERIFIED)
    recipe = _ota_recipe(["500f", "1000f", "2000f"])
    result = run_recipe(recipe, _Runner([(500e-15, 48e6), (1000e-15, 24e6), (2000e-15, 12e6)]),
                        project=False)
    assert result.claim_cards[0].verdict == VerdictClass.VERIFIED


def test_run_recipe_does_not_crash_when_oracle_raises():
    # elasticity does math.log(y); a negative y raises ValueError inside judge -> must FLAG, not abort
    recipe = VerificationRecipe(
        topology_class=OTA, build={"method": "template", "template_ref": "miller_ota_ac"},
        conditions=COND,
        sweeps=[{"analysis": "ac", "knob": "Cc", "points": ["500f", "1000f"], "measure": ["gbw_hz"]}],
        claim_cards=[ClaimCard(id="e", topology_class=OTA, conditions=COND,
                               mechanism=MechanismClaim(knob="Cc", metric="gbw_hz", series_ref="x",
                                                        quant=QuantTest(kind="elasticity", band=(-1.2, -0.8)),
                                                        narrative="..."))],
    )
    result = run_recipe(recipe, _Runner([(500e-15, -1.0), (1000e-15, -2.0)]), project=False)
    assert result.claim_cards[0].verdict == VerdictClass.FLAGGED
