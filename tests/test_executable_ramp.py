"""(CIS readout-chain conquest #3) 14th topology class: single_slope_ramp_generator — the last analog
block of the CIS column readout chain (the single-slope-ADC ramp).

A PMOS current-source mirror sources I into a capacitor Cramp -> a linear ramp v(t) = (I/Cramp)*t; an
nfet reset switch discharges Cramp then releases. The ramp slope is the SS-ADC's volts-per-code-time;
the defining behaviour is slope proportional to the charging current (slope = I/Cramp). Validated
sky130: slope 2.0e7 V/s @ 10uA -> 7.9e7 V/s @ 40uA, slope/I ~ 1/Cramp constant within ~1% (the
linearity the converter relies on). The full SS-ADC's counter is digital logic (out of analog scope);
this is its analog front. Unit + docker integration.
"""

from __future__ import annotations

import pytest

from openclaw_brain.knowledge.executable.corpus import SpecimenCorpus
from openclaw_brain.knowledge.executable.executor import run_recipe
from openclaw_brain.knowledge.executable.models import (
    ClaimCard, AnalogPVT, MechanismClaim, QuantTest, VerdictClass, VerificationRecipe,
)
from openclaw_brain.knowledge.executable.recipe import capability_for
from openclaw_brain.knowledge.executable.runner import NgspiceRunner
from openclaw_brain.knowledge.executable.templates import (
    _RAMP_BODY, render_ramp_tran, render_ramp_cell,
)
from openclaw_brain.knowledge.graph.schema import NodeLabel

RAMP = "single_slope_ramp_generator"
COND = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)
_CERTIFIED = {VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT}


def test_capability_registered_for_ramp():
    cap = capability_for(RAMP)
    assert cap.template_ref == "ramp_tran"
    assert cap.analyses == frozenset({"tran"})
    assert cap.knobs == frozenset({"Iref"})
    assert cap.metrics == frozenset({"slope_vps"})


def test_render_emits_iref_slope_sweep():
    deck = render_ramp_tran(points=["10u", "40u"])
    assert "foreach pt 10u 40u" in deck
    assert "alter IREF = $pt" in deck                    # the charging current is swept
    assert "tran 0.02n 60n" in deck
    assert "meas tran t1 when v(vramp)=0.3 rise=1" in deck
    assert "meas tran t2 when v(vramp)=1.0 rise=1" in deck
    assert "let y = 0.7/(t2-t1)" in deck                 # slope in the linear region
    assert "echo RDATA iref $pt $&y" in deck
    assert "sky130_fd_pr__pfet_01v8" in deck and "sky130_fd_pr__nfet_01v8" in deck


def test_body_is_current_into_cap_with_reset():
    assert "XMch vramp irefn vdd vdd sky130_fd_pr__pfet_01v8" in _RAMP_BODY  # PMOS current source
    assert "Cramp vramp 0" in _RAMP_BODY                                     # the integrating cap
    assert "Xrst vramp phirst 0 0 sky130_fd_pr__nfet_01v8" in _RAMP_BODY     # nfet reset switch
    assert "Vrst phirst 0 PWL(" in _RAMP_BODY                                # reset then release


class _FakeRampRunner:
    def __init__(self, slope):
        self._slope = slope

    def measure(self, deck, timeout=300):
        if "let y = 0.7/(t2-t1)" in deck:
            return {"iref": self._slope}
        return {}


# measured shape (sky130 probe): slope rises ~linearly with Iref (slope = I/Cramp)
_SLOPE = [(10e-6, 1.994e7), (20e-6, 3.966e7), (40e-6, 7.889e7), (80e-6, 1.57e8)]


def _ramp_recipe():
    return VerificationRecipe(
        topology_class=RAMP, build={"method": "template", "template_ref": "ramp_tran"},
        conditions=COND,
        sweeps=[{"analysis": "tran", "knob": "Iref",
                 "points": ["10u", "20u", "40u", "80u"], "measure": ["slope_vps"]}],
        claim_cards=[
            ClaimCard(id="ramp_slope", topology_class=RAMP, conditions=COND,
                      mechanism=MechanismClaim(knob="Iref", metric="slope_vps", series_ref="ramp_slope",
                                               quant=QuantTest(kind="direction", sign="+"),
                                               narrative="slope = I/Cramp rises with Iref")),
        ],
    )


def test_run_recipe_ramp_unit(tmp_path):
    result = run_recipe(_ramp_recipe(), _FakeRampRunner(_SLOPE),
                        corpus=SpecimenCorpus(str(tmp_path)), project=True)
    assert result.runs == 1
    assert set(result.canonical) == {"iref_slope_vps"}
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["ramp_slope"].verdict == VerdictClass.VERIFIED          # slope rises with Iref
    labels = [n["label"] for n in result.projection.nodes]
    assert labels.count(NodeLabel.SPECIMEN) == 1
    assert labels.count(NodeLabel.CLAIM_CARD) == 1


def test_run_recipe_ramp_flags_wrong_direction(tmp_path):
    falling = [(10e-6, 8e7), (20e-6, 6e7), (40e-6, 4e7), (80e-6, 2e7)]
    result = run_recipe(_ramp_recipe(), _FakeRampRunner(falling),
                        corpus=SpecimenCorpus(str(tmp_path)), project=False)
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["ramp_slope"].verdict != VerdictClass.VERIFIED


def test_cell_renderer_is_bare():
    cell = render_ramp_cell()
    assert ".control" not in cell and "foreach" not in cell and "tran" not in cell
    assert "XMch vramp irefn vdd vdd" in cell


def test_run_recipe_ramp_end_to_end_sky130(tmp_path):
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; integration test needs the sim container")
    result = run_recipe(_ramp_recipe(), runner, corpus=SpecimenCorpus(str(tmp_path)), project=True)
    slope = result.canonical["iref_slope_vps"]
    ys = [y for _, y in slope]
    assert slope[0][1] < slope[-1][1]                    # slope rises with Iref
    # slope/Iref should be ~constant (= 1/Cramp) -> linearity
    ratios = [y / x for x, y in slope]
    assert (max(ratios) - min(ratios)) / (sum(ratios) / len(ratios)) < 0.1
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["ramp_slope"].verdict in _CERTIFIED
    assert result.spec_id.startswith("sha256:")
