"""(Razavi conquest #2) 4th topology class: common_gate_nmos — the current buffer.

Defining property: a LOW input resistance Rin ~ 1/gm at the source. The deck drives the source with
an AC current and reads Rin = |v(source)|, sweeping the bias current — Rin falls as gm (and Id) rise.
A new metric (rin_ohm) and a new analysis idiom (input-impedance probe), same frozen executor core.
Unit (fake runner) + docker-gated integration on real sky130.
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
    CG_KNOB_ELEMENTS, CG_METRIC_MEAS, _CG_BODY,
    render_common_gate_ac, render_common_gate_cell,
)
from openclaw_brain.knowledge.graph.schema import NodeLabel

CG = "common_gate_nmos"
COND = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)
_CERTIFIED = {VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT}


def test_capability_registered_for_common_gate():
    cap = capability_for(CG)
    assert cap.template_ref == "common_gate_ac"
    assert cap.analyses == frozenset({"ac"})
    assert cap.knobs == frozenset({"Iref"})
    assert cap.metrics == frozenset({"rin_ohm"})


def test_render_emits_input_resistance_sweep():
    deck = render_common_gate_ac(points=["2u", "10u", "40u"])
    assert "foreach pt 2u 10u 40u" in deck
    assert "alter Itail = $pt" in deck
    assert "Iin s 0 AC 1" in deck                      # the AC current probe at the source
    assert "find vm(s)" in deck                        # Rin = |v(source)|
    assert "echo RDATA iref $pt $&y" in deck
    assert "sky130_fd_pr__nfet_01v8" in deck


def test_template_knob_is_executable():
    cap = capability_for(CG)
    body_elements = {line.split()[0] for line in _CG_BODY.splitlines() if line.strip()}
    for knob in cap.knobs:
        assert CG_KNOB_ELEMENTS[knob] in body_elements
    for metric in cap.metrics:
        assert metric in CG_METRIC_MEAS


class _FakeCGRunner:
    def __init__(self, rin):
        self._rin = rin

    def measure(self, deck, timeout=300):
        if CG_METRIC_MEAS["rin_ohm"].splitlines()[-1].strip() in deck:   # the meas line
            return {"iref": self._rin}
        return {}


# measured shape (sky130 probe): Rin falls 18.5k -> 1.6k as Id 2u -> 40u (Rin ~ 1/gm)
_RIN = [(2e-6, 18469.0), (5e-6, 8030.0), (10e-6, 4442.0), (20e-6, 2595.0), (40e-6, 1636.0)]


def _cg_recipe():
    return VerificationRecipe(
        topology_class=CG, build={"method": "template", "template_ref": "common_gate_ac"},
        conditions=COND,
        sweeps=[{"analysis": "ac", "knob": "Iref",
                 "points": ["2u", "5u", "10u", "20u", "40u"], "measure": ["rin_ohm"]}],
        claim_cards=[ClaimCard(id="cg_rin", topology_class=CG, conditions=COND,
                               mechanism=MechanismClaim(knob="Iref", metric="rin_ohm", series_ref="cg_rin",
                                                        quant=QuantTest(kind="direction", sign="-"),
                                                        narrative="Rin ~ 1/gm falls with bias current"))],
    )


def test_run_recipe_common_gate_unit(tmp_path):
    result = run_recipe(_cg_recipe(), _FakeCGRunner(_RIN),
                        corpus=SpecimenCorpus(str(tmp_path)), project=True)
    assert result.runs == 1
    assert set(result.canonical) == {"iref_rin_ohm"}
    assert result.claim_cards[0].verdict == VerdictClass.VERIFIED       # Rin monotonic falling
    labels = [n["label"] for n in result.projection.nodes]
    assert labels.count(NodeLabel.SPECIMEN) == 1
    assert labels.count(NodeLabel.CLAIM_CARD) == 1


def test_cell_renderer_is_bare():
    cell = render_common_gate_cell()
    assert ".control" not in cell and "foreach" not in cell
    assert "XM1 d g s 0" in cell


def test_run_recipe_common_gate_end_to_end_sky130(tmp_path):
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; integration test needs the sim container")
    result = run_recipe(_cg_recipe(), runner, corpus=SpecimenCorpus(str(tmp_path)), project=True)
    rin = result.canonical["iref_rin_ohm"]
    assert len(rin) == 5
    assert rin[0][1] > rin[-1][1]                       # Rin falls as bias current rises
    assert all(500 < y < 50000 for _, y in rin)         # ~1/gm, in a sane kohm range
    assert result.claim_cards[0].verdict in _CERTIFIED
    assert result.spec_id.startswith("sha256:")
