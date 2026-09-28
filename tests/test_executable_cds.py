"""(CIS readout-chain conquest #2) 13th topology class: cds_switched_cap_nmos — correlated double
sampling, the SIGNATURE CIS readout technique and the substrate's first switched-capacitor specimen.

A sampling cap Cs in series with a real sky130 nfet reset switch (gate = clock phase phi1): during
RESET phi1 shorts vo to Vcm while the input sits at the pedestal -> Cs stores (pedestal - Vcm); then
phi1 goes low (hold) and the input steps up by DIFF -> charge conservation gives vo = Vcm + DIFF *
Cs/(Cs+Cpar), INDEPENDENT of the pedestal. A common offset / fixed-pattern pedestal cancels; only the
(signal - reset) difference survives — the deterministic half of CDS. Validated sky130: held vo
identical (cov 0) across a 0.5->1.3 V pedestal sweep while tracking DIFF. Unit + docker integration.
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
    _CDS_BODY, render_cds_tran, render_cds_cell,
)
from openclaw_brain.knowledge.graph.schema import NodeLabel

CDS = "cds_switched_cap_nmos"
COND = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)
_CERTIFIED = {VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT}


def test_capability_registered_for_cds():
    cap = capability_for(CDS)
    assert cap.template_ref == "cds_tran"
    assert cap.analyses == frozenset({"tran"})
    assert cap.knobs == frozenset({"Vped"})
    assert cap.metrics == frozenset({"vo_v"})


def test_render_emits_pedestal_sweep():
    deck = render_cds_tran(points=["0.5", "1.3"])
    assert "foreach pt 0.5 1.3" in deck
    assert "alter Vped = $pt" in deck                   # the input pedestal (common offset) is swept
    assert "tran 0.01n 15n" in deck
    assert "meas tran y find v(vo) at=14n" in deck       # the held output after the signal phase
    assert "echo RDATA vped $pt $&y" in deck
    assert "sky130_fd_pr__nfet_01v8" in deck             # the real sampling switch


def test_body_is_switched_cap_with_clock():
    # CDS is switched-cap: a sampling cap, an nfet switch gated by a clock phase, a reset->signal step
    assert "Cs vin vo" in _CDS_BODY                      # sampling capacitor
    assert "Xsw vo phi1 vcm 0 sky130_fd_pr__nfet_01v8" in _CDS_BODY  # nfet reset switch on clock phi1
    assert "Vphi1 phi1 0 PWL(" in _CDS_BODY              # the clock phase
    assert "Vstep vin a PWL(" in _CDS_BODY               # reset->signal step


class _FakeCdsRunner:
    def __init__(self, vo):
        self._vo = vo

    def measure(self, deck, timeout=300):
        if "meas tran y find v(vo) at=14n" in deck:
            return {"vped": self._vo}
        return {}


# measured shape (sky130 probe, DIFF=0.2): held vo identical across the pedestal sweep (cov ~0)
_VO = [(0.5, 1.0975), (0.7, 1.0975), (0.9, 1.0975), (1.1, 1.0975), (1.3, 1.0975)]


def _cds_recipe():
    return VerificationRecipe(
        topology_class=CDS, build={"method": "template", "template_ref": "cds_tran"},
        conditions=COND,
        sweeps=[{"analysis": "tran", "knob": "Vped",
                 "points": ["0.5", "0.7", "0.9", "1.1", "1.3"], "measure": ["vo_v"]}],
        claim_cards=[
            ClaimCard(id="cds_vo", topology_class=CDS, conditions=COND,
                      mechanism=MechanismClaim(knob="Vped", metric="vo_v", series_ref="cds_vo",
                                               quant=QuantTest(kind="invariance", cov_max=0.001),
                                               narrative="held output independent of pedestal (CDS)")),
        ],
    )


def test_run_recipe_cds_unit(tmp_path):
    result = run_recipe(_cds_recipe(), _FakeCdsRunner(_VO),
                        corpus=SpecimenCorpus(str(tmp_path)), project=True)
    assert result.runs == 1
    assert set(result.canonical) == {"vped_vo_v"}
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["cds_vo"].verdict == VerdictClass.VERIFIED          # vo invariant -> pedestal cancelled
    labels = [n["label"] for n in result.projection.nodes]
    assert labels.count(NodeLabel.SPECIMEN) == 1
    assert labels.count(NodeLabel.CLAIM_CARD) == 1


def test_run_recipe_cds_flags_pedestal_leak(tmp_path):
    # if the output tracked the pedestal (no cancellation) it would NOT be invariant -> not VERIFIED
    leaks = [(0.5, 0.70), (0.7, 0.90), (0.9, 1.10), (1.1, 1.30), (1.3, 1.50)]
    result = run_recipe(_cds_recipe(), _FakeCdsRunner(leaks),
                        corpus=SpecimenCorpus(str(tmp_path)), project=False)
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["cds_vo"].verdict != VerdictClass.VERIFIED


def test_cell_renderer_is_bare():
    cell = render_cds_cell()
    assert ".control" not in cell and "foreach" not in cell and "tran" not in cell
    assert "Xsw vo phi1 vcm 0" in cell


def test_run_recipe_cds_end_to_end_sky130(tmp_path):
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; integration test needs the sim container")
    result = run_recipe(_cds_recipe(), runner, corpus=SpecimenCorpus(str(tmp_path)), project=True)
    vo = result.canonical["vped_vo_v"]
    ys = [y for _, y in vo]
    mean = sum(ys) / len(ys)
    cov = (max(ys) - min(ys)) / abs(mean)
    assert cov < 0.001                                  # pedestal cancelled -> held vo invariant
    assert all(1.0 < y < 1.2 for y in ys)               # ~ Vcm(0.9) + DIFF(0.2)
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["cds_vo"].verdict in _CERTIFIED
    assert result.spec_id.startswith("sha256:")
