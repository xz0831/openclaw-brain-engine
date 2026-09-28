"""Tier-B Task 5: the NON-VACUOUS headline — composed single-slope-ADC digital back-end.

The oracle's ground truth is SPEC-DERIVED: each testbench computes an INDEPENDENT reference (the
textbook XOR-cascade gray->binary, the monotonicity property, the integer-difference identity) and the
DUT is judged against it. A deliberately-bugged decoder REFUTES — proving the oracle catches a real
functional bug, not a tautology (the load-bearing escape from the self-fulfilling-oracle, DV must-fix).
"""
from __future__ import annotations

import shutil

import pytest

from openclaw_brain.knowledge.executable.corpus import SpecimenCorpus
from openclaw_brain.knowledge.executable.digital_templates import render_ss_adc_backend_tran
from openclaw_brain.knowledge.executable.executor import run_recipe
from openclaw_brain.knowledge.executable.models import QuantTest, VerdictClass
from openclaw_brain.knowledge.executable.oracle import judge_quant
from openclaw_brain.knowledge.executable.recipe import capability_for
from openclaw_brain.knowledge.executable.seeds import _ss_adc_backend_recipe
from openclaw_brain.knowledge.executable.verilog_runner import VerilogRunner

_VERIFIED = VerdictClass.VERIFIED


def test_capability_registered_for_ss_adc():
    cap = capability_for("ss_adc_digital_backend")
    assert cap.template_ref == "ss_adc_backend_tb"
    assert cap.knobs == frozenset({"code", "trip", "ped"})
    assert cap.metrics == frozenset({"g2b_match", "code", "diff_match"})


def test_render_branches_compute_independent_reference():
    g2b = render_ss_adc_backend_tran(metric="g2b_match", knob="code")
    assert "ref_g2b" in g2b                      # textbook XOR-cascade reference computed IN the tb
    code = render_ss_adc_backend_tran(metric="code", knob="trip")
    assert "ss_adc_backend" in code and "run_trip" in code
    diff = render_ss_adc_backend_tran(metric="diff_match", knob="ped")
    assert "module digital_cds" in diff


class _GoodSS:
    """The correct DUT: bijection matches everywhere, code monotonic in trip, diff matches incl. wrap."""
    def measure(self, deck, timeout=120):
        if "ref_g2b" in deck:                     # g2b_match: all 256 codes match the reference
            return {"code": [(float(i), 1.0) for i in range(256)]}
        if "run_trip" in deck:                    # monotonicity: code rises with trip time
            return {"trip": [(10.0, 11.0), (50.0, 51.0), (100.0, 101.0), (200.0, 201.0)]}
        return {"ped": [(float(p), 1.0) for p in (0, 32, 64, 96, 128, 160, 192, 224)]}  # diff incl. wrap


def test_ss_adc_three_properties_verified(tmp_path):
    result = run_recipe(_ss_adc_backend_recipe(), _GoodSS(), corpus=SpecimenCorpus(str(tmp_path)), project=False)
    v = {c.id: c.verdict for c in result.claim_cards}
    assert v["ss_g2b"] == _VERIFIED
    assert v["ss_mono"] == _VERIFIED
    assert v["ss_diff"] == _VERIFIED


class _BuggedSS:
    """A wrong gray->binary decoder: the bijection breaks for most codes -> REFUTED (the other two,
    which don't exercise the decoder, still pass — the failure is localized to the buggy property)."""
    def measure(self, deck, timeout=120):
        if "ref_g2b" in deck:
            return {"code": [(0.0, 1.0), (1.0, 1.0)] + [(float(i), 0.0) for i in range(2, 256)]}
        if "run_trip" in deck:
            return {"trip": [(10.0, 11.0), (50.0, 51.0), (100.0, 101.0), (200.0, 201.0)]}
        return {"ped": [(float(p), 1.0) for p in (0, 32, 64, 96, 128, 160, 192, 224)]}


def test_ss_adc_bugged_decoder_refuted(tmp_path):
    result = run_recipe(_ss_adc_backend_recipe(), _BuggedSS(), corpus=SpecimenCorpus(str(tmp_path)), project=False)
    assert {c.id: c.verdict for c in result.claim_cards}["ss_g2b"] != _VERIFIED


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog not installed")
def test_ss_adc_end_to_end_iverilog(tmp_path):
    # NO runner injected -> the iverilog engine runs the REAL composed RTL; all 3 spec-derived props hold
    result = run_recipe(_ss_adc_backend_recipe(), corpus=SpecimenCorpus(str(tmp_path)), project=False)
    v = {c.id: c.verdict for c in result.claim_cards}
    assert v["ss_g2b"] == _VERIFIED
    assert v["ss_mono"] == _VERIFIED
    assert v["ss_diff"] == _VERIFIED


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog not installed")
def test_ss_adc_bugged_rtl_refuted_iverilog(tmp_path):
    # render the BUGGED decoder for REAL, run iverilog, judge -> not VERIFIED (the oracle catches a real
    # functional bug — this is the proof the headline is non-vacuous, not the author grading the author)
    deck = render_ss_adc_backend_tran(metric="g2b_match", knob="code", bugged=True)
    series = VerilogRunner(workdir=str(tmp_path)).measure(deck)["code"]
    verdict, _ = judge_quant(QuantTest(kind="invariance", cov_max=0.001), series)
    assert verdict != _VERIFIED
