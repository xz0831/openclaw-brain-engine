"""(Razavi conquest #1) Third topology class: common_source_active_load_nmos — the gain primitive.

NMOS common-source + PMOS active load, swept by bias current (.ac, av0/gbw). Same frozen executor
core; the measured claims encode a textbook-vs-sim nuance: GBW rises with Id (clean), but Av0 is only
weakly dependent on Id in sky130 short-channel (the long-channel Av∝1/√Id law softens). Unit (fake
runner) + docker-gated integration on real sky130.
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
    CS_KNOB_ELEMENTS, OTA_METRIC_MEAS, _CS_BODY,
    render_common_source_ac, render_common_source_cell,
)
from openclaw_brain.knowledge.graph.schema import NodeLabel

CS = "common_source_active_load_nmos"
COND = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)
_CERTIFIED = {VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT}


# ── template + capability ──


def test_capability_registered_for_common_source():
    cap = capability_for(CS)
    assert cap.template_ref == "common_source_ac"
    assert cap.analyses == frozenset({"ac"})
    assert cap.knobs == frozenset({"Iref"})
    assert cap.metrics == frozenset({"av0_db", "gbw_hz"})


def test_render_emits_ac_sweep_rdata():
    deck = render_common_source_ac(points=["2u", "10u", "40u"])
    assert "foreach pt 2u 10u 40u" in deck
    assert "alter IREF = $pt" in deck
    assert "ac dec 40 1 10G" in deck
    assert "echo RDATA iref $pt $&y" in deck
    assert "sky130_fd_pr__nfet_01v8" in deck and "sky130_fd_pr__pfet_01v8" in deck


def test_template_knob_is_executable():
    """Drift guard: the knob's swept source must exist in the cell body."""
    cap = capability_for(CS)
    body_elements = {line.split()[0] for line in _CS_BODY.splitlines() if line.strip()}
    for knob in cap.knobs:
        assert CS_KNOB_ELEMENTS[knob] in body_elements
    for metric in cap.metrics:
        assert metric in OTA_METRIC_MEAS          # reuses the OTA's av0/gbw meas


# ── executor: unit (fake runner) ──


def _meas_line(metric: str) -> str:
    for ln in OTA_METRIC_MEAS[metric].splitlines():
        if "meas " in ln:
            return ln.strip()
    return OTA_METRIC_MEAS[metric].strip()


class _FakeCSRunner:
    """Returns the av0 (flat) or gbw (rising) series by which metric's meas line the deck contains."""

    def __init__(self, av0, gbw):
        self._by = {_meas_line("av0_db"): av0, _meas_line("gbw_hz"): gbw}

    def measure(self, deck, timeout=300):
        for sig, series in self._by.items():
            if sig in deck:
                return {"iref": series}
        return {}


# measured shapes (sky130 probe): Av0 flat ~37 dB (spread ~1.1 dB), GBW rising 7->82 MHz
_AV0 = [(2e-6, 36.73), (5e-6, 36.98), (10e-6, 37.01), (20e-6, 36.73), (40e-6, 35.93)]
_GBW = [(2e-6, 7.2e6), (5e-6, 16.6e6), (10e-6, 29.9e6), (20e-6, 51.1e6), (40e-6, 81.7e6)]


def _cs_recipe():
    def claim(cid, metric, quant, narrative):
        return ClaimCard(id=cid, topology_class=CS, conditions=COND,
                         mechanism=MechanismClaim(knob="Iref", metric=metric, series_ref=cid,
                                                  quant=quant, narrative=narrative))
    return VerificationRecipe(
        topology_class=CS, build={"method": "template", "template_ref": "common_source_ac"},
        conditions=COND,
        sweeps=[{"analysis": "ac", "knob": "Iref",
                 "points": ["2u", "5u", "10u", "20u", "40u"], "measure": ["av0_db", "gbw_hz"]}],
        claim_cards=[
            claim("cs_gbw", "gbw_hz", QuantTest(kind="direction", sign="+"), "GBW rises with Id"),
            claim("cs_av0", "av0_db", QuantTest(kind="invariance", spread_max=1.5),
                  "gain weakly dependent on Id in sky130 (textbook 1/sqrt(Id) softens)"),
        ],
    )


def test_run_recipe_common_source_unit(tmp_path):
    result = run_recipe(_cs_recipe(), _FakeCSRunner(_AV0, _GBW),
                        corpus=SpecimenCorpus(str(tmp_path)), project=True)
    assert result.runs == 2                                  # one deck per (Iref, av0) and (Iref, gbw)
    assert set(result.canonical) == {"iref_av0_db", "iref_gbw_hz"}
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["cs_gbw"].verdict == VerdictClass.VERIFIED           # GBW monotonic rising
    assert by_id["cs_av0"].verdict == VerdictClass.VERIFIED           # spread 1.08 dB < 1.5
    labels = [n["label"] for n in result.projection.nodes]
    assert labels.count(NodeLabel.SPECIMEN) == 1
    assert labels.count(NodeLabel.CLAIM_CARD) == 2


def test_cell_renderer_is_bare():
    cell = render_common_source_cell()
    assert ".control" not in cell and "foreach" not in cell
    assert "XM1 out   g 0 0" in cell


# ── executor: docker-gated integration on real sky130 ──


def test_run_recipe_common_source_end_to_end_sky130(tmp_path):
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; integration test needs the sim container")
    result = run_recipe(_cs_recipe(), runner, corpus=SpecimenCorpus(str(tmp_path)), project=True)

    av0 = result.canonical["iref_av0_db"]
    gbw = result.canonical["iref_gbw_hz"]
    assert len(av0) == 5 and len(gbw) == 5
    assert all(25 < y < 45 for _, y in av0)                  # sane single-stage gain ~37 dB
    assert max(y for _, y in av0) - min(y for _, y in av0) < 1.5   # weakly dependent on Id
    assert gbw[-1][1] > gbw[0][1]                            # GBW rises with bias current
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["cs_gbw"].verdict in _CERTIFIED
    assert by_id["cs_av0"].verdict in _CERTIFIED
    assert result.spec_id.startswith("sha256:")
