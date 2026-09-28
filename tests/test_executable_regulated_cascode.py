"""(Razavi conquest #9, Ch.9 gain-boosting) 11th topology class: regulated_cascode_nmos.

The regulated cascode (RGC) is gain-boosting in its purest form: an auxiliary amplifier (Mp/Mn) holds
the cascode source node constant, so the bottom device sees a fixed Vds regardless of the output
voltage -> its ro is boosted by the aux gain -> Rout ~ gm*ro^2 * A_aux. Measured as a current source
(.dc Vout -> iout): the regulated cascode held iout flat within ~0.005% over Vout 0.8-1.6V vs ~1.4%
for the same five transistors with the aux loop replaced by a fixed gate (a ~270x Rout boost — the
gain-boosting payoff as a falsifiable A/B). The aux is a DC regulation loop only (no main signal
loop), so no AC stability concern. Reuses the cascode-mirror DC infra. Unit + docker integration.
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
    CM_KNOB_SOURCES, CM_METRIC_LET, _RGC_BODY,
    render_regulated_cascode_dc, render_regulated_cascode_cell,
)
from openclaw_brain.knowledge.graph.schema import NodeLabel

RGC = "regulated_cascode_nmos"
COND = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)
_CERTIFIED = {VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT}


def test_capability_registered_for_regulated_cascode():
    cap = capability_for(RGC)
    assert cap.template_ref == "regulated_cascode_dc"
    assert cap.analyses == frozenset({"dc"})
    assert cap.knobs == frozenset({"Vout"})
    assert cap.metrics == frozenset({"iout_a"})


def test_render_emits_vout_sweep():
    deck = render_regulated_cascode_dc(points=["0.8", "1.2"])
    assert "foreach pt 0.8 1.2" in deck
    assert "alter Vout = $pt" in deck
    assert "XMp g1c ns vdd vdd" in deck and "XMn g1c vba 0 0" in deck  # the auxiliary regulation amp
    assert "XM1c out g1c ns 0" in deck                                 # cascode gate driven by aux (g1c)
    assert "let y = -i(vout)" in deck
    assert "echo RDATA vout $pt $&y" in deck
    assert "sky130_fd_pr__nfet_01v8" in deck and "sky130_fd_pr__pfet_01v8" in deck


def test_template_knob_is_executable():
    cap = capability_for(RGC)
    body_elements = {line.split()[0] for line in _RGC_BODY.splitlines() if line.strip()}
    for knob in cap.knobs:
        assert CM_KNOB_SOURCES[knob] in body_elements
    for metric in cap.metrics:
        assert metric in CM_METRIC_LET


class _FakeRgcRunner:
    def __init__(self, iout):
        self._iout = iout

    def measure(self, deck, timeout=300):
        if "let y = -i(vout)" in deck:
            return {"vout": self._iout}
        return {}


# measured shape (sky130 probe, vba=0.8): iout essentially constant (boosted Rout, cov ~0.00005)
_IOUT = [(0.8, 9.6722e-6), (1.0, 9.6724e-6), (1.2, 9.6725e-6), (1.4, 9.6726e-6), (1.6, 9.6727e-6)]


def _rgc_recipe():
    return VerificationRecipe(
        topology_class=RGC, build={"method": "template", "template_ref": "regulated_cascode_dc"},
        conditions=COND,
        sweeps=[{"analysis": "dc", "knob": "Vout",
                 "points": ["0.8", "1.0", "1.2", "1.4", "1.6"], "measure": ["iout_a"]}],
        claim_cards=[
            ClaimCard(id="rgc_iout", topology_class=RGC, conditions=COND,
                      mechanism=MechanismClaim(knob="Vout", metric="iout_a", series_ref="rgc_iout",
                                               quant=QuantTest(kind="invariance", cov_max=0.001),
                                               narrative="boosted Rout -> iout independent of Vout")),
        ],
    )


def test_run_recipe_regulated_cascode_unit(tmp_path):
    result = run_recipe(_rgc_recipe(), _FakeRgcRunner(_IOUT),
                        corpus=SpecimenCorpus(str(tmp_path)), project=True)
    assert result.runs == 1
    assert set(result.canonical) == {"vout_iout_a"}
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["rgc_iout"].verdict == VerdictClass.VERIFIED          # iout flat (boosted Rout)
    labels = [n["label"] for n in result.projection.nodes]
    assert labels.count(NodeLabel.SPECIMEN) == 1
    assert labels.count(NodeLabel.CLAIM_CARD) == 1


def test_run_recipe_regulated_cascode_flags_unboosted(tmp_path):
    # a plain (unboosted) cascode slopes ~1.4% over Vout -> must FAIL the cov_max=0.001 invariance
    plain = [(0.8, 7.857e-6), (1.0, 7.888e-6), (1.2, 7.915e-6), (1.4, 7.941e-6), (1.6, 7.965e-6)]
    result = run_recipe(_rgc_recipe(), _FakeRgcRunner(plain),
                        corpus=SpecimenCorpus(str(tmp_path)), project=False)
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["rgc_iout"].verdict != VerdictClass.VERIFIED          # too sloped -> not boosted


def test_cell_renderer_is_bare():
    cell = render_regulated_cascode_cell()
    assert ".control" not in cell and "foreach" not in cell
    assert "XMp g1c ns vdd vdd" in cell


def test_run_recipe_regulated_cascode_end_to_end_sky130(tmp_path):
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; integration test needs the sim container")
    result = run_recipe(_rgc_recipe(), runner, corpus=SpecimenCorpus(str(tmp_path)), project=True)
    iout = result.canonical["vout_iout_a"]
    ys = [y for _, y in iout]
    mean = sum(ys) / len(ys)
    cov = (max(ys) - min(ys)) / abs(mean)
    assert cov < 0.001                                  # boosted Rout -> iout flat within 0.1%
    assert all(5e-6 < y < 15e-6 for y in ys)            # ~10uA reference current
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["rgc_iout"].verdict in _CERTIFIED
    assert result.spec_id.startswith("sha256:")
