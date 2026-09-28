"""Stat-QT Task 4: Pelgrom-TYPE power law = the existing `elasticity` kind on a (area, σ) series.

The executable substrate cannot claim ideal Pelgrom (σ∝area^-0.5) — sky130's open mismatch model scales
as ~area^-0.375, so an exact 'σ·√area = constant' invariance is REFUTED by the model (honest finding).
Instead the claim certifies the log-log SCALING EXPONENT lies in a Pelgrom-type band: σ decreases as a
power of area (bigger device -> less mismatch), the exponent being model/node-specific.
"""
import math

from openclaw_brain.knowledge.executable.models import QuantTest, VerdictClass
from openclaw_brain.knowledge.executable.oracle import judge_quant

_CERT = {VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT}


def _series(areas, sigmas):
    return [(a, s) for a, s in zip(areas, sigmas)]


def test_pelgrom_powerlaw_is_elasticity_verified():
    areas = [0.25, 1.0, 4.0]
    sigmas = [5e-3 / math.sqrt(a) for a in areas]            # ideal Pelgrom σ ∝ area^-0.5 -> slope -0.5
    v, _ = judge_quant(QuantTest(kind="elasticity", band=(-0.6, -0.2)), _series(areas, sigmas))
    assert v in _CERT


def test_skylike_exponent_minus_0p375_verified():
    areas = [0.25, 1.0, 4.0]
    sigmas = [5e-3 * a ** (-0.375) for a in areas]           # sky130-like exponent -> in band
    v, _ = judge_quant(QuantTest(kind="elasticity", band=(-0.6, -0.2)), _series(areas, sigmas))
    assert v in _CERT


def test_no_area_scaling_is_refuted():
    areas = [0.25, 1.0, 4.0]
    sigmas = [5e-3, 5e-3, 5e-3]                              # σ area-INDEPENDENT -> slope 0 -> out of band
    v, _ = judge_quant(QuantTest(kind="elasticity", band=(-0.6, -0.2)), _series(areas, sigmas))
    assert v == VerdictClass.REFUTED
