"""(Razavi conquest #7, Ch.9 deep-tier) 9th topology class: telescopic_cascode_ota_nmos_in.

NMOS input pair + NMOS cascodes + PMOS cascode current-source loads. Cascoding both sides boosts
Rout ~ (gm*ro^2), so DC gain is MUCH higher than the simple-mirror 5T OTA (measured ~67 dB vs ~37 dB,
same input pair) — the defining telescopic trade (high gain, at the cost of output swing). Two probe
lessons baked into the template: the PMOS-top mirror reference is DECOUPLED into its own leg (folding
it into a signal branch starves the DC operating point), and the cascode gates ride SEPARATE tuned
rails, NOT diode stacks (a diode cascode burns a full Vgs of headroom and collapses the stack in
sky130's 1.8 V budget). Single-stage, output-pole-limited (GBW = gm1/2*pi*CL). Unit + docker integration.
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
    OTA_KNOB_ELEMENTS, OTA_METRIC_MEAS, _TELE_BODY,
    render_telescopic_cascode_ac, render_telescopic_cascode_cell,
)
from openclaw_brain.knowledge.graph.schema import NodeLabel

TELE = "telescopic_cascode_ota_nmos_in"
COND = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)
_CERTIFIED = {VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT}


def test_capability_registered_for_telescopic():
    cap = capability_for(TELE)
    assert cap.template_ref == "telescopic_cascode_ac"
    assert cap.analyses == frozenset({"ac"})
    assert cap.knobs == frozenset({"CL"})
    assert cap.metrics == frozenset({"av0_db", "gbw_hz"})


def test_render_emits_cl_sweep():
    deck = render_telescopic_cascode_ac(points=["500f", "2000f"])
    assert "foreach pt 500f 2000f" in deck
    assert "alter CL = $pt" in deck
    assert "Lfb out vinn 1T" in deck                       # feedback to the INVERTING input
    assert "IREFP vbp 0" in deck and "XMRP vbp vbp vdd vdd" in deck  # decoupled PMOS-top reference leg
    assert "Vbpc gpc 0" in deck and "Vbnc gnc 0" in deck   # cascode gates on tuned rails, not diodes
    assert "echo RDATA cl $pt $&y" in deck
    assert "sky130_fd_pr__nfet_01v8" in deck and "sky130_fd_pr__pfet_01v8" in deck


def test_template_knob_is_executable():
    cap = capability_for(TELE)
    body_elements = {line.split()[0] for line in _TELE_BODY.splitlines() if line.strip()}
    for knob in cap.knobs:
        assert OTA_KNOB_ELEMENTS[knob] in body_elements
    for metric in cap.metrics:
        assert metric in OTA_METRIC_MEAS


def _meas_line(metric: str) -> str:
    for ln in OTA_METRIC_MEAS[metric].splitlines():
        if "meas " in ln:
            return ln.strip()
    return OTA_METRIC_MEAS[metric].strip()


class _FakeTeleRunner:
    def __init__(self, av0, gbw):
        self._by = {_meas_line("av0_db"): av0, _meas_line("gbw_hz"): gbw}

    def measure(self, deck, timeout=300):
        for sig, series in self._by.items():
            if sig in deck:
                return {"cl": series}
        return {}


# measured shapes (sky130 probe): cascode-boosted Av0 flat ~67.4 dB; GBW ∝ 1/CL (26.3 -> 3.2 MHz)
_AV0 = [(5e-13, 67.42), (1e-12, 67.42), (2e-12, 67.42), (4e-12, 67.42)]
_GBW = [(5e-13, 26.3e6), (1e-12, 13.1e6), (2e-12, 6.46e6), (4e-12, 3.22e6)]


def _tele_recipe():
    def claim(cid, metric, quant, narrative):
        return ClaimCard(id=cid, topology_class=TELE, conditions=COND,
                         mechanism=MechanismClaim(knob="CL", metric=metric, series_ref=cid,
                                                  quant=quant, narrative=narrative))
    return VerificationRecipe(
        topology_class=TELE, build={"method": "template", "template_ref": "telescopic_cascode_ac"},
        conditions=COND,
        sweeps=[{"analysis": "ac", "knob": "CL",
                 "points": ["500f", "1000f", "2000f", "4000f"], "measure": ["av0_db", "gbw_hz"]}],
        claim_cards=[
            claim("tele_gbw", "gbw_hz", QuantTest(kind="direction", sign="-"), "GBW = gm1/2piCL"),
            claim("tele_av0", "av0_db", QuantTest(kind="invariance", spread_max=0.5),
                  "cascode-boosted DC gain independent of CL"),
        ],
    )


def test_run_recipe_telescopic_unit(tmp_path):
    result = run_recipe(_tele_recipe(), _FakeTeleRunner(_AV0, _GBW),
                        corpus=SpecimenCorpus(str(tmp_path)), project=True)
    assert result.runs == 2
    assert set(result.canonical) == {"cl_av0_db", "cl_gbw_hz"}
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["tele_gbw"].verdict == VerdictClass.VERIFIED          # GBW falls with CL
    assert by_id["tele_av0"].verdict == VerdictClass.VERIFIED          # av0 flat
    labels = [n["label"] for n in result.projection.nodes]
    assert labels.count(NodeLabel.SPECIMEN) == 1
    assert labels.count(NodeLabel.CLAIM_CARD) == 2


def test_cell_renderer_is_bare():
    cell = render_telescopic_cascode_cell()
    assert ".control" not in cell and "foreach" not in cell
    assert "XM4c out gpc p4 vdd" in cell


def test_run_recipe_telescopic_end_to_end_sky130(tmp_path):
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; integration test needs the sim container")
    result = run_recipe(_tele_recipe(), runner, corpus=SpecimenCorpus(str(tmp_path)), project=True)
    av0 = result.canonical["cl_av0_db"]
    gbw = result.canonical["cl_gbw_hz"]
    assert all(55 < y < 80 for _, y in av0)            # cascode-boosted single-stage gain ~67 dB
    assert max(y for _, y in av0) - min(y for _, y in av0) < 0.5   # gain independent of CL
    assert gbw[0][1] > gbw[-1][1]                       # GBW falls as CL rises
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["tele_gbw"].verdict in _CERTIFIED
    assert by_id["tele_av0"].verdict in _CERTIFIED
    assert result.spec_id.startswith("sha256:")
