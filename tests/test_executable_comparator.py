"""(CIS readout-chain conquest #1) 12th topology class: comparator_continuous_nmos — the column
single-slope-ADC comparator, and the substrate's first TIME-DOMAIN (.tran) specimen.

A high-gain open-loop 5T core (NMOS diff pair + PMOS active mirror, no feedback/compensation): vinn =
reference, vinp = a step crossing the trip. The defining behaviour is propagation delay tpd FALLING as
input overdrive rises (more overdrive -> faster slew to the decision threshold). The .tran render
alters the input PULSE high level per overdrive point and measures the input-cross-to-output-decision
delay. Validated sky130: trip 0.90V, tpd 9.2ns @ 50mV -> 3.4ns @ 400mV. Demonstrates that adding a new
analysis (.tran) costs zero core plumbing — runner/oracle/executor are analysis-agnostic.
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
    _CMP_BODY, render_comparator_tran, render_comparator_cell,
)
from openclaw_brain.knowledge.graph.schema import NodeLabel

CMP = "comparator_continuous_nmos"
COND = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)
_CERTIFIED = {VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT}


def test_capability_registered_for_comparator():
    cap = capability_for(CMP)
    assert cap.template_ref == "comparator_tran"
    assert cap.analyses == frozenset({"tran"})         # the new time-domain analysis primitive
    assert cap.knobs == frozenset({"Vov"})
    assert cap.metrics == frozenset({"tpd_s"})


def test_render_emits_tran_overdrive_sweep():
    deck = render_comparator_tran(points=["0.05", "0.2"])
    assert "foreach pt 0.05 0.2" in deck
    assert "let vhi = 0.9 + $pt" in deck                # overdrive above the trip (VREF=0.9)
    assert "alter @Vin[pulse] = [ 0.6 $&vhi" in deck    # the input PULSE high level is swept
    assert "tran 0.005n 20n" in deck                    # time-domain analysis
    assert "meas tran y trig v(vinp) val=0.9 rise=1 targ v(out) val=0.9 rise=1" in deck
    assert "echo RDATA vov $pt $&y" in deck
    assert "sky130_fd_pr__nfet_01v8" in deck and "sky130_fd_pr__pfet_01v8" in deck


def test_body_is_open_loop_with_pulse_input():
    # a comparator is OPEN loop (no Lfb feedback like the OTAs) with a PULSE-driven input
    assert "Lfb" not in _CMP_BODY
    assert "Vin vinp 0 PULSE(" in _CMP_BODY
    assert "Vref vinn 0" in _CMP_BODY                   # the other input is a fixed reference


class _FakeCmpRunner:
    def __init__(self, tpd):
        self._tpd = tpd

    def measure(self, deck, timeout=300):
        if "meas tran y trig v(vinp)" in deck:
            return {"vov": self._tpd}
        return {}


# measured shape (sky130 probe): tpd falls as overdrive rises (9.2 -> 3.4 ns)
_TPD = [(0.05, 9.25e-9), (0.1, 5.31e-9), (0.2, 3.73e-9), (0.4, 3.36e-9)]


def _cmp_recipe():
    return VerificationRecipe(
        topology_class=CMP, build={"method": "template", "template_ref": "comparator_tran"},
        conditions=COND,
        sweeps=[{"analysis": "tran", "knob": "Vov",
                 "points": ["0.05", "0.1", "0.2", "0.4"], "measure": ["tpd_s"]}],
        claim_cards=[
            ClaimCard(id="cmp_tpd", topology_class=CMP, conditions=COND,
                      mechanism=MechanismClaim(knob="Vov", metric="tpd_s", series_ref="cmp_tpd",
                                               quant=QuantTest(kind="direction", sign="-"),
                                               narrative="tpd falls as overdrive rises")),
        ],
    )


def test_run_recipe_comparator_unit(tmp_path):
    result = run_recipe(_cmp_recipe(), _FakeCmpRunner(_TPD),
                        corpus=SpecimenCorpus(str(tmp_path)), project=True)
    assert result.runs == 1
    assert set(result.canonical) == {"vov_tpd_s"}
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["cmp_tpd"].verdict == VerdictClass.VERIFIED          # tpd decreasing -> direction -
    labels = [n["label"] for n in result.projection.nodes]
    assert labels.count(NodeLabel.SPECIMEN) == 1
    assert labels.count(NodeLabel.CLAIM_CARD) == 1


def test_run_recipe_comparator_flags_wrong_direction(tmp_path):
    # if tpd INCREASED with overdrive it would contradict the comparator law -> not VERIFIED
    rising = [(0.05, 3.3e-9), (0.1, 3.7e-9), (0.2, 5.3e-9), (0.4, 9.2e-9)]
    result = run_recipe(_cmp_recipe(), _FakeCmpRunner(rising),
                        corpus=SpecimenCorpus(str(tmp_path)), project=False)
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["cmp_tpd"].verdict != VerdictClass.VERIFIED


def test_cell_renderer_is_bare():
    cell = render_comparator_cell()
    assert ".control" not in cell and "foreach" not in cell and "tran" not in cell
    assert "Vin vinp 0 PULSE(" in cell


def test_run_recipe_comparator_end_to_end_sky130(tmp_path):
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; integration test needs the sim container")
    result = run_recipe(_cmp_recipe(), runner, corpus=SpecimenCorpus(str(tmp_path)), project=True)
    tpd = result.canonical["vov_tpd_s"]
    ys = [y for _, y in tpd]
    assert tpd[0][1] > tpd[-1][1]                        # tpd falls as overdrive rises
    assert all(1e-9 < y < 50e-9 for y in ys)             # nanosecond-scale propagation delay
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["cmp_tpd"].verdict in _CERTIFIED
    assert result.spec_id.startswith("sha256:")
