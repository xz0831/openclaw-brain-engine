"""Tier-B Task 4: digital_cds pedagogy demo (iverilog) — the first non-ngspice specimen run end-to-end.

Proves the engine dispatch routes a digital template_ref to the iverilog engine and that the
artifact-agnostic claim-card + oracle judge a Verilog series exactly as they do an ngspice one.
"""
from __future__ import annotations

import shutil

import pytest

from openclaw_brain.knowledge.executable.corpus import SpecimenCorpus
from openclaw_brain.knowledge.executable.digital_templates import render_digital_cds_tran
from openclaw_brain.knowledge.executable.engines import engine_for_template
from openclaw_brain.knowledge.executable.executor import run_recipe
from openclaw_brain.knowledge.executable.models import VerdictClass
from openclaw_brain.knowledge.executable.recipe import capability_for
from openclaw_brain.knowledge.executable.seeds import _digital_cds_recipe


def test_capability_registered_for_digital_cds():
    cap = capability_for("digital_cds_nmos")
    assert cap.template_ref == "digital_cds_tb"
    assert cap.knobs == frozenset({"ped"})
    assert cap.metrics == frozenset({"diff_lsb"})


def test_digital_cds_template_routes_to_iverilog():
    assert engine_for_template("digital_cds_tb") == "iverilog"


def test_render_emits_self_contained_verilog():
    deck = render_digital_cds_tran(points=["0", "64"])
    assert "module digital_cds" in deck          # the DUT is inlined (self-contained tb)
    assert "sig - rst" in deck                   # the CDS subtraction
    assert '$display("RDATA ped' in deck          # byte-compatible with parse_rdata


class _FakeDigitalRunner:
    # held diff invariant to the common pedestal (cov 0) — the digital CDS cancellation
    def measure(self, deck, timeout=120):
        if "sig - rst" in deck:
            return {"ped": [(0.0, 16.0), (64.0, 16.0), (128.0, 16.0), (192.0, 16.0)]}
        return {}


def test_run_recipe_digital_cds_unit(tmp_path):
    result = run_recipe(_digital_cds_recipe(), _FakeDigitalRunner(),
                        corpus=SpecimenCorpus(str(tmp_path)), project=False)
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["dcds_inv"].verdict == VerdictClass.VERIFIED      # pedestal cancelled -> invariant
    # the Specimen records the digital engine's provenance, NOT sky130/ngspice
    assert result.specimen.tool == "iverilog-13"
    assert result.specimen.pdk == "n/a"


def test_run_recipe_digital_cds_pedestal_leak_not_verified(tmp_path):
    class _Leak:
        def measure(self, deck, timeout=120):
            return {"ped": [(0.0, 16.0), (64.0, 80.0), (128.0, 144.0), (192.0, 208.0)]}  # tracks pedestal
    result = run_recipe(_digital_cds_recipe(), _Leak(), corpus=SpecimenCorpus(str(tmp_path)), project=False)
    assert {c.id: c.verdict for c in result.claim_cards}["dcds_inv"] != VerdictClass.VERIFIED


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog not installed")
def test_run_recipe_digital_cds_end_to_end_iverilog(tmp_path):
    # NO runner injected -> run_recipe picks the iverilog engine's VerilogRunner and runs real iverilog
    result = run_recipe(_digital_cds_recipe(), corpus=SpecimenCorpus(str(tmp_path)), project=False)
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["dcds_inv"].verdict == VerdictClass.VERIFIED
    assert result.spec_id.startswith("sha256:")
