"""(Razavi conquest #8, Ch.9) 10th topology class: folded_cascode_ota_nmos_in.

NMOS input pair with drains at the FOLD nodes, current folded down through PMOS cascodes into an NMOS
cascode-mirror load. Single-ended output. Cascode-boosted gain (measured ~71 dB) with a wider input
common-mode range / output swing than the telescopic (the fold un-stacks the input device from the
output cascode — its defining advantage). Single-stage, output-pole-limited (GBW = gm1/2*pi*CL).

The load-bearing probe finding baked into the template: the PMOS-top current IREFP must be only
SLIGHTLY above the input-branch current (Itail/2), NOT equal to the tail. Over-biasing pushes the top
source into TRIODE at a railed fold node, shunting the signal to VDD and collapsing GBW ~30x while DC
gain still reads ~50 dB — a silent trap the measure-first probe caught. Unit + docker integration.
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
    OTA_KNOB_ELEMENTS, OTA_METRIC_MEAS, _FC_BODY,
    render_folded_cascode_ac, render_folded_cascode_cell,
)
from openclaw_brain.knowledge.graph.schema import NodeLabel

FC = "folded_cascode_ota_nmos_in"
COND = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)
_CERTIFIED = {VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT}


def test_capability_registered_for_folded_cascode():
    cap = capability_for(FC)
    assert cap.template_ref == "folded_cascode_ac"
    assert cap.analyses == frozenset({"ac"})
    assert cap.knobs == frozenset({"CL"})
    assert cap.metrics == frozenset({"av0_db", "gbw_hz"})


def test_render_emits_cl_sweep():
    deck = render_folded_cascode_ac(points=["500f", "2000f"])
    assert "foreach pt 500f 2000f" in deck
    assert "alter CL = $pt" in deck
    assert "Lfb out vinn 1T" in deck                       # feedback to the INVERTING input
    assert "IREFP vbp 0" in deck and "XMRP vbp vbp vdd vdd" in deck  # decoupled PMOS-top reference leg
    assert "XM3 f1 vbp vdd vdd" in deck and "XM3c o1  gpc f1 vdd" in deck  # fold node + PMOS cascode
    assert "echo RDATA cl $pt $&y" in deck
    assert "sky130_fd_pr__nfet_01v8" in deck and "sky130_fd_pr__pfet_01v8" in deck


def test_template_knob_is_executable():
    cap = capability_for(FC)
    body_elements = {line.split()[0] for line in _FC_BODY.splitlines() if line.strip()}
    for knob in cap.knobs:
        assert OTA_KNOB_ELEMENTS[knob] in body_elements
    for metric in cap.metrics:
        assert metric in OTA_METRIC_MEAS


def _meas_line(metric: str) -> str:
    for ln in OTA_METRIC_MEAS[metric].splitlines():
        if "meas " in ln:
            return ln.strip()
    return OTA_METRIC_MEAS[metric].strip()


class _FakeFcRunner:
    def __init__(self, av0, gbw):
        self._by = {_meas_line("av0_db"): av0, _meas_line("gbw_hz"): gbw}

    def measure(self, deck, timeout=300):
        for sig, series in self._by.items():
            if sig in deck:
                return {"cl": series}
        return {}


# measured shapes (sky130 probe, IREFP=12u): cascode-boosted Av0 flat ~71.3 dB; GBW ∝ 1/CL
_AV0 = [(5e-13, 71.26), (1e-12, 71.26), (2e-12, 71.26), (4e-12, 71.26)]
_GBW = [(5e-13, 10.3e6), (1e-12, 5.26e6), (2e-12, 2.65e6), (4e-12, 1.33e6)]


def _fc_recipe():
    def claim(cid, metric, quant, narrative):
        return ClaimCard(id=cid, topology_class=FC, conditions=COND,
                         mechanism=MechanismClaim(knob="CL", metric=metric, series_ref=cid,
                                                  quant=quant, narrative=narrative))
    return VerificationRecipe(
        topology_class=FC, build={"method": "template", "template_ref": "folded_cascode_ac"},
        conditions=COND,
        sweeps=[{"analysis": "ac", "knob": "CL",
                 "points": ["500f", "1000f", "2000f", "4000f"], "measure": ["av0_db", "gbw_hz"]}],
        claim_cards=[
            claim("fc_gbw", "gbw_hz", QuantTest(kind="direction", sign="-"), "GBW = gm1/2piCL"),
            claim("fc_av0", "av0_db", QuantTest(kind="invariance", spread_max=0.5),
                  "cascode-boosted DC gain independent of CL"),
        ],
    )


def test_run_recipe_folded_cascode_unit(tmp_path):
    result = run_recipe(_fc_recipe(), _FakeFcRunner(_AV0, _GBW),
                        corpus=SpecimenCorpus(str(tmp_path)), project=True)
    assert result.runs == 2
    assert set(result.canonical) == {"cl_av0_db", "cl_gbw_hz"}
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["fc_gbw"].verdict == VerdictClass.VERIFIED          # GBW falls with CL
    assert by_id["fc_av0"].verdict == VerdictClass.VERIFIED          # av0 flat
    labels = [n["label"] for n in result.projection.nodes]
    assert labels.count(NodeLabel.SPECIMEN) == 1
    assert labels.count(NodeLabel.CLAIM_CARD) == 2


def test_cell_renderer_is_bare():
    cell = render_folded_cascode_cell()
    assert ".control" not in cell and "foreach" not in cell
    assert "XM7c out gnc nb2 0" in cell


def test_run_recipe_folded_cascode_end_to_end_sky130(tmp_path):
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; integration test needs the sim container")
    result = run_recipe(_fc_recipe(), runner, corpus=SpecimenCorpus(str(tmp_path)), project=True)
    av0 = result.canonical["cl_av0_db"]
    gbw = result.canonical["cl_gbw_hz"]
    assert all(60 < y < 80 for _, y in av0)            # cascode-boosted single-stage gain ~71 dB
    assert max(y for _, y in av0) - min(y for _, y in av0) < 0.5   # gain independent of CL
    assert gbw[0][1] > gbw[-1][1]                       # GBW falls as CL rises
    assert gbw[0][1] > 1e6                              # MHz GBW (not the kHz triode-shunt trap)
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["fc_gbw"].verdict in _CERTIFIED
    assert by_id["fc_av0"].verdict in _CERTIFIED
    assert result.spec_id.startswith("sha256:")
