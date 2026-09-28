"""(Razavi conquest #5) 7th topology class: cascode_current_mirror_nmos — output-resistance boost.

A cascode device shields the output and raises Rout to ~gm*ro^2 (~30x a simple mirror), so the
mirrored current is nearly independent of the output voltage. Same .dc/op + iout idiom as the simple
mirror; the distinctive claim is iout INVARIANCE (vs the simple mirror's rising iout). Validated
sky130: iout flat 9.98-10.00 uA over Vout 0.8-1.6 (cov ~0.2%). Unit + docker integration.
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
    CM_KNOB_SOURCES, CM_METRIC_LET, _CASC_BODY,
    render_cascode_mirror_cell, render_cascode_mirror_dc,
)
from openclaw_brain.knowledge.graph.schema import NodeLabel

CASC = "cascode_current_mirror_nmos"
COND = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)
_CERTIFIED = {VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT}


def test_capability_registered_for_cascode_mirror():
    cap = capability_for(CASC)
    assert cap.template_ref == "cascode_mirror_dc"
    assert cap.analyses == frozenset({"dc"})
    assert cap.knobs == frozenset({"Vout"})
    assert cap.metrics == frozenset({"iout_a"})


def test_render_emits_dc_sweep():
    deck = render_cascode_mirror_dc(points=["0.8", "1.2", "1.6"])
    assert "foreach pt 0.8 1.2 1.6" in deck
    assert "alter Vout = $pt" in deck
    assert "echo RDATA vout $pt $&y" in deck
    assert deck.count("sky130_fd_pr__nfet_01v8") == 4   # the 4-device cascode mirror


def test_template_knob_is_executable():
    cap = capability_for(CASC)
    body_elements = {line.split()[0] for line in _CASC_BODY.splitlines() if line.strip()}
    for knob in cap.knobs:
        assert CM_KNOB_SOURCES[knob] in body_elements
    for metric in cap.metrics:
        assert metric in CM_METRIC_LET


class _FakeCascRunner:
    def __init__(self, iout):
        self._iout = iout

    def measure(self, deck, timeout=300):
        return {"vout": self._iout} if CM_METRIC_LET["iout_a"] in deck else {}


# measured shape (sky130 probe): iout flat ~10 uA over Vout 0.8-1.6 (very high Rout)
_IOUT = [(0.8, 9.984e-6), (1.0, 9.994e-6), (1.2, 9.997e-6), (1.4, 9.999e-6), (1.6, 1.0001e-5)]


def _casc_recipe():
    return VerificationRecipe(
        topology_class=CASC, build={"method": "template", "template_ref": "cascode_mirror_dc"},
        conditions=COND,
        sweeps=[{"analysis": "dc", "knob": "Vout",
                 "points": ["0.8", "1.0", "1.2", "1.4", "1.6"], "measure": ["iout_a"]}],
        claim_cards=[ClaimCard(id="casc_iout", topology_class=CASC, conditions=COND,
                               mechanism=MechanismClaim(knob="Vout", metric="iout_a", series_ref="casc_iout",
                                                        quant=QuantTest(kind="invariance", cov_max=0.02),
                                                        narrative="cascode Rout ~ gm*ro^2 -> iout flat"))],
    )


def test_run_recipe_cascode_mirror_unit(tmp_path):
    result = run_recipe(_casc_recipe(), _FakeCascRunner(_IOUT),
                        corpus=SpecimenCorpus(str(tmp_path)), project=True)
    assert result.runs == 1
    assert set(result.canonical) == {"vout_iout_a"}
    assert result.claim_cards[0].verdict == VerdictClass.VERIFIED       # CoV ~0.2% < 2%
    labels = [n["label"] for n in result.projection.nodes]
    assert labels.count(NodeLabel.SPECIMEN) == 1
    assert labels.count(NodeLabel.CLAIM_CARD) == 1


def test_cell_renderer_is_bare():
    cell = render_cascode_mirror_cell()
    assert ".control" not in cell and "foreach" not in cell
    assert "XM4 out nc nx 0" in cell                    # the output cascode device


def test_run_recipe_cascode_mirror_end_to_end_sky130(tmp_path):
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; integration test needs the sim container")
    result = run_recipe(_casc_recipe(), runner, corpus=SpecimenCorpus(str(tmp_path)), project=True)
    iout = result.canonical["vout_iout_a"]
    assert len(iout) == 5
    assert all(8e-6 < y < 12e-6 for _, y in iout)       # mirrors ~10 uA
    spread = max(y for _, y in iout) - min(y for _, y in iout)
    assert spread / 10e-6 < 0.02                        # iout flat (cov < 2%) -> high Rout
    assert result.claim_cards[0].verdict in _CERTIFIED
    assert result.spec_id.startswith("sha256:")
