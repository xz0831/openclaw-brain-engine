"""Tier-B Task 4: gray_code_counter pedagogy demo (iverilog) — Hamming-1 invariance."""
from __future__ import annotations

import shutil

import pytest

from openclaw_brain.knowledge.executable.corpus import SpecimenCorpus
from openclaw_brain.knowledge.executable.digital_templates import render_gray_counter_tran
from openclaw_brain.knowledge.executable.executor import run_recipe
from openclaw_brain.knowledge.executable.models import VerdictClass
from openclaw_brain.knowledge.executable.recipe import capability_for
from openclaw_brain.knowledge.executable.seeds import _gray_recipe


def test_capability_registered_for_gray():
    cap = capability_for("gray_code_counter")
    assert cap.template_ref == "gray_counter_tb"
    assert cap.knobs == frozenset({"idx"})
    assert cap.metrics == frozenset({"hamming"})


def test_render_emits_bin2gray_and_hamming():
    deck = render_gray_counter_tran()
    assert "module bin2gray" in deck and "b ^ (b >> 1)" in deck
    assert '$display("RDATA idx' in deck


class _FakeGrayRunner:
    # every consecutive Gray pair differs by exactly one bit -> hamming == 1 for all indices
    def measure(self, deck, timeout=120):
        return {"idx": [(float(i), 1.0) for i in range(1, 16)]}


def test_run_recipe_gray_unit(tmp_path):
    result = run_recipe(_gray_recipe(), _FakeGrayRunner(), corpus=SpecimenCorpus(str(tmp_path)), project=False)
    assert {c.id: c.verdict for c in result.claim_cards}["gray_h1"] == VerdictClass.VERIFIED


def test_run_recipe_gray_non_hamming1_not_verified(tmp_path):
    class _Bad:  # a non-Gray sequence: distances vary -> not invariant
        def measure(self, deck, timeout=120):
            return {"idx": [(1.0, 1.0), (2.0, 2.0), (3.0, 1.0), (4.0, 3.0)]}
    result = run_recipe(_gray_recipe(), _Bad(), corpus=SpecimenCorpus(str(tmp_path)), project=False)
    assert {c.id: c.verdict for c in result.claim_cards}["gray_h1"] != VerdictClass.VERIFIED


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog not installed")
def test_run_recipe_gray_end_to_end_iverilog(tmp_path):
    result = run_recipe(_gray_recipe(), corpus=SpecimenCorpus(str(tmp_path)), project=False)
    assert {c.id: c.verdict for c in result.claim_cards}["gray_h1"] == VerdictClass.VERIFIED
