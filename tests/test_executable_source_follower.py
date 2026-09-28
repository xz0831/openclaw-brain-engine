"""(Razavi conquest #3) 5th topology class: source_follower_nmos — the voltage buffer / pixel SF.

Defining property: a near-unity, SUB-unity, bias-independent voltage gain (body effect makes Av<1).
Drain at VDD, gate = input, source = output to a tail current; reads av0_db = max vdb(out). Validated
sky130: Av ~ -1.6 dB (~0.83) and flat (<0.03 dB) over 20x bias. Unit (fake runner) + docker integration.
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
    OTA_METRIC_MEAS, SF_KNOB_ELEMENTS, _SF_BODY,
    render_source_follower_ac, render_source_follower_cell,
)
from openclaw_brain.knowledge.graph.schema import NodeLabel

SF = "source_follower_nmos"
COND = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)
_CERTIFIED = {VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT}


def test_capability_registered_for_source_follower():
    cap = capability_for(SF)
    assert cap.template_ref == "source_follower_ac"
    assert cap.analyses == frozenset({"ac"})
    assert cap.knobs == frozenset({"Iref"})
    assert cap.metrics == frozenset({"av0_db"})


def test_render_emits_gain_sweep():
    deck = render_source_follower_ac(points=["2u", "10u", "40u"])
    assert "foreach pt 2u 10u 40u" in deck
    assert "alter Itail = $pt" in deck
    assert "XM1 vdd g out 0" in deck                    # drain at VDD = follower
    assert "max vdb(out)" in deck
    assert "echo RDATA iref $pt $&y" in deck


def test_template_knob_is_executable():
    cap = capability_for(SF)
    body_elements = {line.split()[0] for line in _SF_BODY.splitlines() if line.strip()}
    for knob in cap.knobs:
        assert SF_KNOB_ELEMENTS[knob] in body_elements
    for metric in cap.metrics:
        assert metric in OTA_METRIC_MEAS


def _meas_line(metric: str) -> str:
    for ln in OTA_METRIC_MEAS[metric].splitlines():
        if "meas " in ln:
            return ln.strip()
    return OTA_METRIC_MEAS[metric].strip()


class _FakeSFRunner:
    def __init__(self, av0):
        self._av0 = av0

    def measure(self, deck, timeout=300):
        return {"iref": self._av0} if _meas_line("av0_db") in deck else {}


# measured shape (sky130 probe): sub-unity gain ~-1.6 dB, flat over bias
_AV0 = [(2e-6, -1.601), (5e-6, -1.600), (10e-6, -1.603), (20e-6, -1.610), (40e-6, -1.625)]


def _sf_recipe():
    return VerificationRecipe(
        topology_class=SF, build={"method": "template", "template_ref": "source_follower_ac"},
        conditions=COND,
        sweeps=[{"analysis": "ac", "knob": "Iref",
                 "points": ["2u", "5u", "10u", "20u", "40u"], "measure": ["av0_db"]}],
        claim_cards=[ClaimCard(id="sf_av0", topology_class=SF, conditions=COND,
                               mechanism=MechanismClaim(knob="Iref", metric="av0_db", series_ref="sf_av0",
                                                        quant=QuantTest(kind="invariance", spread_max=0.5),
                                                        narrative="sub-unity buffer gain, bias-independent"))],
    )


def test_run_recipe_source_follower_unit(tmp_path):
    result = run_recipe(_sf_recipe(), _FakeSFRunner(_AV0),
                        corpus=SpecimenCorpus(str(tmp_path)), project=True)
    assert result.runs == 1
    assert set(result.canonical) == {"iref_av0_db"}
    assert result.claim_cards[0].verdict == VerdictClass.VERIFIED       # spread 0.025 dB < 0.5
    labels = [n["label"] for n in result.projection.nodes]
    assert labels.count(NodeLabel.SPECIMEN) == 1
    assert labels.count(NodeLabel.CLAIM_CARD) == 1


def test_cell_renderer_is_bare():
    cell = render_source_follower_cell()
    assert ".control" not in cell and "foreach" not in cell
    assert "XM1 vdd g out 0" in cell


def test_run_recipe_source_follower_end_to_end_sky130(tmp_path):
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; integration test needs the sim container")
    result = run_recipe(_sf_recipe(), runner, corpus=SpecimenCorpus(str(tmp_path)), project=True)
    av0 = result.canonical["iref_av0_db"]
    assert len(av0) == 5
    assert all(-4 < y < 0 for _, y in av0)              # sub-unity (negative dB), near unity
    assert max(y for _, y in av0) - min(y for _, y in av0) < 0.5   # bias-independent (buffer)
    assert result.claim_cards[0].verdict in _CERTIFIED
    assert result.spec_id.startswith("sha256:")
