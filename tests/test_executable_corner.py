"""Stat-QT Task 3: the `corner` QuantTest kind — VERIFIED-iff-all-corners off a corner-swept series."""
from openclaw_brain.knowledge.executable.models import QuantTest, VerdictClass
from openclaw_brain.knowledge.executable.oracle import judge_quant


def _s(vals):
    return [(float(i), v) for i, v in enumerate(vals)]


def test_corner_all_pass_verified():
    v, _ = judge_quant(QuantTest(kind="corner", sign="+", bound=29e6),
                       _s([30.28e6, 29.97e6, 30.44e6, 30.32e6, 30.01e6]))
    assert v == VerdictClass.VERIFIED


def test_corner_one_fail_refuted_names_index():
    v, note = judge_quant(QuantTest(kind="corner", sign="+", bound=30.2e6),
                          _s([30.28e6, 29.97e6, 30.44e6, 30.32e6, 30.01e6]))   # idx 1,4 fail
    assert v == VerdictClass.REFUTED and "1" in note and "4" in note


def test_corner_too_few_corners_flagged():
    v, _ = judge_quant(QuantTest(kind="corner", sign="+", bound=1.0), _s([2.0]))
    assert v == VerdictClass.FLAGGED
