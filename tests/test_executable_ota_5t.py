"""(Razavi conquest #6) 8th topology class: ota_5t_nmos_in — the canonical single-stage OTA.

NMOS diff pair + PMOS mirror load + NMOS tail (the active-mirror differential pair / 5T OTA). The DC
self-bias feeds back to the INVERTING input (a single stage is non-inverting from vinp; feeding
out->vinp would latch). Validated sky130: Av0 37.1 dB (flat vs CL), GBW 60->7.6 MHz as CL 500f->4p
(GBW = gm1/2piCL). Reuses the OTA av0/gbw meas + the CL knob. Unit + docker integration.
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
    OTA_KNOB_ELEMENTS, OTA_METRIC_MEAS, _OTA5T_BODY,
    render_ota_5t_ac, render_ota_5t_cell,
)
from openclaw_brain.knowledge.graph.schema import NodeLabel

OTA5T = "ota_5t_nmos_in"
COND = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)
_CERTIFIED = {VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT}


def test_capability_registered_for_ota_5t():
    cap = capability_for(OTA5T)
    assert cap.template_ref == "ota_5t_ac"
    # S3-inc2a §2: + "dc" analysis, + VDD/IREFV knobs, + vout_swing_v/icmr_lo_v/icmr_hi_v metrics
    # (the swing/ICMR device-query sweeps), alongside the pre-existing CL/av0/gbw/"ac" surface.
    assert cap.analyses == frozenset({"ac", "dc"})
    assert cap.knobs == frozenset({"CL", "VDD", "IREFV"})
    assert cap.metrics == frozenset({"av0_db", "gbw_hz", "vout_swing_v", "icmr_lo_v", "icmr_hi_v"})


def test_render_emits_cl_sweep():
    deck = render_ota_5t_ac(points=["500f", "2000f"])
    assert "foreach pt 500f 2000f" in deck
    assert "alter CL = $pt" in deck
    assert "Lfb out vinn 1T" in deck                   # feedback to the INVERTING input (1-stage)
    assert "echo RDATA cl $pt $&y" in deck
    assert "sky130_fd_pr__nfet_01v8" in deck and "sky130_fd_pr__pfet_01v8" in deck


def test_template_knob_is_executable():
    cap = capability_for(OTA5T)
    body_elements = {line.split()[0] for line in _OTA5T_BODY.splitlines() if line.strip()}
    for knob in cap.knobs:
        assert OTA_KNOB_ELEMENTS[knob] in body_elements
    for metric in cap.metrics:
        if metric in ("vout_swing_v", "icmr_lo_v", "icmr_hi_v"):
            # S3-inc2a §2: these route through a dedicated device-query control block, not
            # OTA_METRIC_MEAS — just confirm they render without raising.
            assert "[vdsat]" in render_ota_5t_ac(knob="VDD", metric=metric)
            continue
        assert metric in OTA_METRIC_MEAS


def _meas_line(metric: str) -> str:
    for ln in OTA_METRIC_MEAS[metric].splitlines():
        if "meas " in ln:
            return ln.strip()
    return OTA_METRIC_MEAS[metric].strip()


class _FakeOta5tRunner:
    def __init__(self, av0, gbw):
        self._by = {_meas_line("av0_db"): av0, _meas_line("gbw_hz"): gbw}

    def measure(self, deck, timeout=300):
        for sig, series in self._by.items():
            if sig in deck:
                return {"cl": series}
        return {}


# measured shapes (sky130 probe): Av0 flat 37.11 dB; GBW ∝ 1/CL (60 -> 7.6 MHz)
_AV0 = [(5e-13, 37.11), (1e-12, 37.11), (2e-12, 37.11), (4e-12, 37.11)]
_GBW = [(5e-13, 60.0e6), (1e-12, 30.3e6), (2e-12, 15.2e6), (4e-12, 7.6e6)]


def _ota5t_recipe():
    def claim(cid, metric, quant, narrative):
        return ClaimCard(id=cid, topology_class=OTA5T, conditions=COND,
                         mechanism=MechanismClaim(knob="CL", metric=metric, series_ref=cid,
                                                  quant=quant, narrative=narrative))
    return VerificationRecipe(
        topology_class=OTA5T, build={"method": "template", "template_ref": "ota_5t_ac"},
        conditions=COND,
        sweeps=[{"analysis": "ac", "knob": "CL",
                 "points": ["500f", "1000f", "2000f", "4000f"], "measure": ["av0_db", "gbw_hz"]}],
        claim_cards=[
            claim("ota5t_gbw", "gbw_hz", QuantTest(kind="direction", sign="-"), "GBW = gm1/2piCL"),
            claim("ota5t_av0", "av0_db", QuantTest(kind="invariance", spread_max=0.5),
                  "single-stage DC gain independent of CL"),
        ],
    )


def test_run_recipe_ota_5t_unit(tmp_path):
    result = run_recipe(_ota5t_recipe(), _FakeOta5tRunner(_AV0, _GBW),
                        corpus=SpecimenCorpus(str(tmp_path)), project=True)
    assert result.runs == 2
    assert set(result.canonical) == {"cl_av0_db", "cl_gbw_hz"}
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["ota5t_gbw"].verdict == VerdictClass.VERIFIED          # GBW falls with CL
    assert by_id["ota5t_av0"].verdict == VerdictClass.VERIFIED          # av0 flat
    labels = [n["label"] for n in result.projection.nodes]
    assert labels.count(NodeLabel.SPECIMEN) == 1
    assert labels.count(NodeLabel.CLAIM_CARD) == 2


def test_cell_renderer_is_bare():
    cell = render_ota_5t_cell()
    assert ".control" not in cell and "foreach" not in cell
    assert "XM4 out o1 vdd vdd" in cell


def test_run_recipe_ota_5t_end_to_end_sky130(tmp_path):
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; integration test needs the sim container")
    result = run_recipe(_ota5t_recipe(), runner, corpus=SpecimenCorpus(str(tmp_path)), project=True)
    av0 = result.canonical["cl_av0_db"]
    gbw = result.canonical["cl_gbw_hz"]
    assert all(25 < y < 45 for _, y in av0)            # single-stage gain ~37 dB
    assert max(y for _, y in av0) - min(y for _, y in av0) < 0.5   # gain independent of CL
    assert gbw[0][1] > gbw[-1][1]                       # GBW falls as CL rises
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["ota5t_gbw"].verdict in _CERTIFIED
    assert by_id["ota5t_av0"].verdict in _CERTIFIED
    assert result.spec_id.startswith("sha256:")
