"""Tests for the executable-circuit runner + T-template (knowledge/executable).

Unit tests (no docker): template rendering + RDATA parsing.
Integration test (skipped without the IIC-OSIC-TOOLS image): the full ② slice —
template -> ngspice (container) -> parse -> increment-1 oracle — reproduces
Specimen-One's cc_gbw_inverse verdict (VERIFIED_WITH_CAVEAT) from code, not hand-SPICE.
"""

import pytest

from openclaw_brain.knowledge.executable.models import (
    ClaimCard, AnalogPVT, MechanismClaim, QuantTest, VerdictClass,
)
from openclaw_brain.knowledge.executable.oracle import ClaimOracle
from openclaw_brain.knowledge.executable.runner import NgspiceRunner, parse_rdata, parse_unit
from openclaw_brain.knowledge.executable.templates import (
    DEFAULT_OTA_SIZING, LIB_PLACEHOLDER, render_miller_ota_ac,
)


# ── unit: template ──

def test_template_renders_valid_deck():
    deck = render_miller_ota_ac()
    # fixed connectivity uses the ANALOG primitives (not digital std-cells), 1.8V core
    assert "sky130_fd_pr__nfet_01v8" in deck
    assert "sky130_fd_pr__pfet_01v8" in deck
    assert "sky130_fd_sc_hd" not in deck            # the 9B failure mode must be impossible
    assert f'.lib "{LIB_PLACEHOLDER}" tt' in deck    # runner substitutes the lib path
    assert "VDD=1.8" in deck and "W1=8" in deck      # validated sizing
    assert "Ccomp out o1" in deck                    # Miller cap connectivity
    assert "RDATA cc" in deck                        # machine-parseable sweep output


def test_template_sizing_override():
    deck = render_miller_ota_ac(sizing={"W1": "12", "Cc": "2p"})
    assert "W1=12" in deck and "Cc=2p" in deck
    assert "W3=4" in deck                            # untouched defaults remain


# ── unit: parsing ──

def test_parse_unit_suffixes():
    assert parse_unit("250f") == pytest.approx(250e-15)
    assert parse_unit("1p") == pytest.approx(1e-12)
    assert parse_unit("8k") == pytest.approx(8000.0)
    assert parse_unit("1.8") == pytest.approx(1.8)


def test_parse_rdata_builds_sorted_series():
    out = (
        "noise...\nRDATA cc 1000f 2.42641E+07\nRDATA cc 250f 5.47476E+07\n"
        "junk\nRDATA cc 4000f 7.52958E+06\n"
    )
    series = parse_rdata(out)
    assert set(series) == {"cc"}
    xs = [x for x, _ in series["cc"]]
    assert xs == sorted(xs)                          # sorted by knob value
    assert series["cc"][0] == (pytest.approx(250e-15), pytest.approx(5.47476e7))


# ── integration: full ② slice (template -> ngspice -> parse -> oracle) ──

def _cc_claim() -> ClaimCard:
    return ClaimCard(
        id="cc_gbw_inverse", topology_class="miller_ota_2stage_nmos_in",
        mechanism=MechanismClaim(knob="Cc", metric="gbw_hz", series_ref="cc",
                                 quant=QuantTest(kind="elasticity", target=-1.0, band=(-1.3, -0.7)),
                                 narrative="Miller compensation sets the dominant pole; GBW=gm1/(2*pi*Cc)"),
        conditions=AnalogPVT(corner="tt", temp_c=27, vdd=1.8, cl_f=2e-12,
                              pdk_profile={"pdk_id": "sky130A", "device": "planar"}),
    )


def test_runner_end_to_end_reproduces_cc_verdict():
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; integration test needs the sim container")
    series = runner.measure(render_miller_ota_ac(knob="Cc", metric="gbw_hz"))
    assert "cc" in series and len(series["cc"]) == 5
    # GBW @ Cc=1p reproduces the committed Specimen-One measurement (~24.3 MHz)
    gbw_1p = next(y for x, y in series["cc"] if abs(x - 1e-12) < 1e-15)
    assert gbw_1p == pytest.approx(2.426e7, rel=0.03)
    # the oracle reaches the same verdict as the hand-run Specimen-One
    claim = ClaimOracle().judge(_cc_claim(), series)
    assert claim.verdict == VerdictClass.VERIFIED_WITH_CAVEAT, claim.verdict_note
