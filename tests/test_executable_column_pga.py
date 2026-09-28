"""(CIS readout-chain conquest #4) 15th topology class: column_pga_inverting_nmos — the programmable-
gain column amplifier, closing the CIS readout chain's amplification block.

A 5T OTA core in inverting resistive feedback: + input = Vcm, - input = the summing node vx (virtual
ground), input through Rin to vx, feedback Rf from out to vx. Closed-loop gain = -Rf/Rin, set by the
feedback RATIO (the programmable-gain property). Feedback resistors are >> the OTA output resistance
(ro ~350k): a transconductance amp drives resistive loads poorly, so small Rf would tank the gain
(which is why CIS PGAs often use SC feedback). Validated sky130 (Rin=500k): gain -0.4 dB @ Rf=500k ->
16.9 dB @ Rf=4M, tracking 20log(Rf/Rin) within ~1 dB. Unit + docker integration.
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
    OTA_METRIC_MEAS, _PGA_BODY, render_column_pga_ac, render_column_pga_cell,
)
from openclaw_brain.knowledge.graph.schema import NodeLabel

PGA = "column_pga_inverting_nmos"
COND = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)
_CERTIFIED = {VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT}


def test_capability_registered_for_pga():
    cap = capability_for(PGA)
    assert cap.template_ref == "column_pga_ac"
    assert cap.analyses == frozenset({"ac"})
    assert cap.knobs == frozenset({"Rf"})
    assert cap.metrics == frozenset({"acl_db"})


def test_acl_db_is_closed_loop_metric():
    # closed-loop gain reuses the av0_db measurement (peak |vout/vin|) under a distinct key, so a
    # closed-loop specimen grounds to "Closed-Loop Gain", not the open-loop "av0"
    assert "acl_db" in OTA_METRIC_MEAS
    assert OTA_METRIC_MEAS["acl_db"] == OTA_METRIC_MEAS["av0_db"]


def test_render_emits_rf_sweep():
    deck = render_column_pga_ac(points=["500k", "4000k"])
    assert "foreach pt 500k 4000k" in deck
    assert "alter Rf = $pt" in deck                     # the feedback resistor programs the gain
    assert "Rin vin vx" in deck and "Rf  out vx" in deck  # inverting feedback network
    assert "XM1 o1  vcm tail 0" in deck                  # + input = Vcm, - input = vx (summing node)
    assert "meas ac y max vdb(out)" in deck
    assert "echo RDATA rf $pt $&y" in deck


def test_body_is_inverting_feedback():
    assert "Vin vin 0 DC" in _PGA_BODY and "AC 1" in _PGA_BODY  # AC drive through Rin
    assert "Rin vin vx" in _PGA_BODY                            # input resistor to the summing node
    assert "Rf  out vx" in _PGA_BODY                            # feedback resistor (programs gain)
    assert "XM2 out vx  tail 0" in _PGA_BODY                    # inverting input = summing node vx


class _FakePgaRunner:
    def __init__(self, gain):
        self._gain = gain

    def measure(self, deck, timeout=300):
        if "max vdb(out)" in deck:
            return {"rf": self._gain}
        return {}


# measured shape (sky130 probe, Rin=500k): gain rises with Rf, ~ 20log(Rf/Rin)
_GAIN = [(500e3, -0.40), (1000e3, 5.53), (2000e3, 11.35), (4000e3, 16.94)]


def _pga_recipe():
    return VerificationRecipe(
        topology_class=PGA, build={"method": "template", "template_ref": "column_pga_ac"},
        conditions=COND,
        sweeps=[{"analysis": "ac", "knob": "Rf",
                 "points": ["500k", "1000k", "2000k", "4000k"], "measure": ["acl_db"]}],
        claim_cards=[
            ClaimCard(id="pga_gain", topology_class=PGA, conditions=COND,
                      mechanism=MechanismClaim(knob="Rf", metric="acl_db", series_ref="pga_gain",
                                               quant=QuantTest(kind="direction", sign="+"),
                                               narrative="closed-loop gain = Rf/Rin rises with Rf")),
        ],
    )


def test_run_recipe_pga_unit(tmp_path):
    result = run_recipe(_pga_recipe(), _FakePgaRunner(_GAIN),
                        corpus=SpecimenCorpus(str(tmp_path)), project=True)
    assert result.runs == 1
    assert set(result.canonical) == {"rf_acl_db"}
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["pga_gain"].verdict == VerdictClass.VERIFIED          # gain rises with Rf
    labels = [n["label"] for n in result.projection.nodes]
    assert labels.count(NodeLabel.SPECIMEN) == 1
    assert labels.count(NodeLabel.CLAIM_CARD) == 1


def test_run_recipe_pga_flags_wrong_direction(tmp_path):
    falling = [(500e3, 16.0), (1000e3, 11.0), (2000e3, 5.0), (4000e3, -1.0)]
    result = run_recipe(_pga_recipe(), _FakePgaRunner(falling),
                        corpus=SpecimenCorpus(str(tmp_path)), project=False)
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["pga_gain"].verdict != VerdictClass.VERIFIED


def test_cell_renderer_is_bare():
    cell = render_column_pga_cell()
    assert ".control" not in cell and "foreach" not in cell
    assert "Rf  out vx" in cell


def test_run_recipe_pga_end_to_end_sky130(tmp_path):
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; integration test needs the sim container")
    result = run_recipe(_pga_recipe(), runner, corpus=SpecimenCorpus(str(tmp_path)), project=True)
    gain = result.canonical["rf_acl_db"]
    ys = [y for _, y in gain]
    assert gain[0][1] < gain[-1][1]                      # gain rises with Rf (programmable)
    assert ys[-1] - ys[0] > 10                           # ~18 dB span over the 8x Rf range
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["pga_gain"].verdict in _CERTIFIED
    assert result.spec_id.startswith("sha256:")
