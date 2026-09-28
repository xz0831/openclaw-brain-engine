"""(Razavi conquest #4) 6th topology class: diff_pair_resistive_nmos — the differential pair.

Matched NMOS pair + resistive loads + tail current; the differential gain Adm = gm*RD rises with the
tail current. Differential AC input (|vid|=1), differential output read as db|v(o1)-v(o2)| (new
metric adm_db). Resistive loads set the DC point (no self-bias). Validated sky130: Adm 7.9 -> 19.6 dB
as Itail 5u -> 80u. Unit (fake runner) + docker integration.
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
    DP_KNOB_ELEMENTS, DP_METRIC_MEAS, _DP_BODY,
    render_diff_pair_ac, render_diff_pair_cell,
)
from openclaw_brain.knowledge.graph.schema import NodeLabel

DP = "diff_pair_resistive_nmos"
COND = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)
_CERTIFIED = {VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT}


def test_capability_registered_for_diff_pair():
    cap = capability_for(DP)
    assert cap.template_ref == "diff_pair_ac"
    assert cap.analyses == frozenset({"ac"})
    assert cap.knobs == frozenset({"Iref"})
    assert cap.metrics == frozenset({"adm_db"})


def test_render_emits_diff_gain_sweep():
    deck = render_diff_pair_ac(points=["5u", "20u", "80u"])
    assert "foreach pt 5u 20u 80u" in deck
    assert "alter Itail = $pt" in deck
    assert "db(abs(v(o1)-v(o2)))" in deck               # differential output gain
    assert "echo RDATA iref $pt $&y" in deck
    assert "XM1 o1 vinp tail 0" in deck and "XM2 o2 vinn tail 0" in deck


def test_template_knob_is_executable():
    cap = capability_for(DP)
    body_elements = {line.split()[0] for line in _DP_BODY.splitlines() if line.strip()}
    for knob in cap.knobs:
        assert DP_KNOB_ELEMENTS[knob] in body_elements
    for metric in cap.metrics:
        assert metric in DP_METRIC_MEAS


class _FakeDPRunner:
    def __init__(self, adm):
        self._adm = adm

    def measure(self, deck, timeout=300):
        return {"iref": self._adm} if "meas ac y max admdb" in deck else {}


# measured shape (sky130 probe): Adm rises 7.9 -> 19.6 dB as Itail 5u -> 80u
_ADM = [(5e-6, 7.89), (10e-6, 9.6), (20e-6, 11.36), (40e-6, 15.0), (80e-6, 19.61)]


def _dp_recipe():
    return VerificationRecipe(
        topology_class=DP, build={"method": "template", "template_ref": "diff_pair_ac"},
        conditions=COND,
        sweeps=[{"analysis": "ac", "knob": "Iref",
                 "points": ["5u", "10u", "20u", "40u", "80u"], "measure": ["adm_db"]}],
        claim_cards=[ClaimCard(id="dp_adm", topology_class=DP, conditions=COND,
                               mechanism=MechanismClaim(knob="Iref", metric="adm_db", series_ref="dp_adm",
                                                        quant=QuantTest(kind="direction", sign="+"),
                                                        narrative="Adm = gm*RD rises with tail current"))],
    )


def test_run_recipe_diff_pair_unit(tmp_path):
    result = run_recipe(_dp_recipe(), _FakeDPRunner(_ADM),
                        corpus=SpecimenCorpus(str(tmp_path)), project=True)
    assert result.runs == 1
    assert set(result.canonical) == {"iref_adm_db"}
    assert result.claim_cards[0].verdict == VerdictClass.VERIFIED       # Adm monotonic rising
    labels = [n["label"] for n in result.projection.nodes]
    assert labels.count(NodeLabel.SPECIMEN) == 1
    assert labels.count(NodeLabel.CLAIM_CARD) == 1


def test_cell_renderer_is_bare():
    cell = render_diff_pair_cell()
    assert ".control" not in cell and "foreach" not in cell
    assert "XM1 o1 vinp tail 0" in cell


def test_run_recipe_diff_pair_end_to_end_sky130(tmp_path):
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; integration test needs the sim container")
    result = run_recipe(_dp_recipe(), runner, corpus=SpecimenCorpus(str(tmp_path)), project=True)
    adm = result.canonical["iref_adm_db"]
    assert len(adm) == 5
    assert adm[-1][1] > adm[0][1]                       # Adm = gm*RD rises with tail current
    assert all(0 < y < 40 for _, y in adm)              # sane resistive-load differential gain
    assert result.claim_cards[0].verdict in _CERTIFIED
    assert result.spec_id.startswith("sha256:")
