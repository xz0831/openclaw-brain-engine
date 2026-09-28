"""Stat-QT Task 2: the `statistical` QuantTest kind — 3σ and worst-of-N k·σ off a Monte-Carlo series."""
import math

from openclaw_brain.knowledge.executable.models import QuantTest, VerdictClass
from openclaw_brain.knowledge.executable.oracle import judge_quant


def _series(vals):
    return [(float(i), v) for i, v in enumerate(vals)]


def test_three_sigma_verified_when_under_bound():
    ys = [0.005, -0.006, 0.004, -0.005, 0.006, -0.004, 0.005, -0.005]   # ~5mV σ -> 3σ≈15mV < 20mV
    v, note = judge_quant(
        QuantTest(kind="statistical", reducer="three_sigma", bound=0.020, absolute=True), _series(ys))
    assert v == VerdictClass.VERIFIED and "σ" in note


def test_three_sigma_refuted_when_over_bound():
    ys = [0.05, -0.05, 0.04, -0.06, 0.05, -0.04, 0.06, -0.05]
    v, _ = judge_quant(
        QuantTest(kind="statistical", reducer="three_sigma", bound=0.005, absolute=True), _series(ys))
    assert v == VerdictClass.REFUTED


def test_k_sigma_worst_of_n_uses_probit():
    # N_col=2048 -> k = Φ⁻¹(1-1/2048) ≈ 3.3; worst-column FPN = k·σ
    q = QuantTest(kind="statistical", reducer="k_sigma", n_col=2048, bound=0.060, absolute=True)
    ys = [0.005 * math.sin(i) for i in range(40)]    # σ ≈ 3.5mV -> k·σ ≈ 11.6mV < 60mV
    v, note = judge_quant(q, _series(ys))
    assert v == VerdictClass.VERIFIED and "k=3.3" in note.replace("k=3.30", "k=3.3")


def test_statistical_too_few_samples_flagged():
    v, _ = judge_quant(
        QuantTest(kind="statistical", reducer="three_sigma", bound=0.02), _series([0.01]))
    assert v == VerdictClass.FLAGGED


# ── Task 6: the executor outer-loop builds the Pelgrom σ·√WL series (mock per-area runner) ──
import re

from openclaw_brain.knowledge.executable.corpus import SpecimenCorpus
from openclaw_brain.knowledge.executable.executor import run_recipe
from openclaw_brain.knowledge.executable.models import (
    AnalogPVT, ClaimCard, MechanismClaim, VerificationRecipe)

_PEL = "ota_5t_nmos_in"


class _AreaRunner:
    # returns an MC series whose σ obeys Pelgrom for the W1 encoded in the deck (σ ∝ 1/√W)
    def measure(self, deck, timeout=300):
        m = re.search(r"W1=(\d+(?:\.\d+)?)", deck)
        w1 = float(m.group(1)) if m else 8.0
        sigma = 5e-3 / math.sqrt(w1)
        return {"vos": [(float(i), sigma * (1 if i % 2 else -1)) for i in range(12)]}


def test_executor_pelgrom_outer_loop_builds_sigma_sqrtwl_series(tmp_path):
    cond = AnalogPVT(corner="tt_mm", temp_c=27.0, vdd=1.8, mc_runs=12, areas=[0.25, 1.0, 4.0])
    rec = VerificationRecipe(
        topology_class=_PEL, build={"method": "template", "template_ref": "ota5t_offset_mc"},
        conditions=cond,
        sweeps=[{"analysis": "pelgrom", "knob": "vos", "areas": [0.25, 1.0, 4.0], "measure": ["a_vos"]}],
        claim_cards=[ClaimCard(id="pel", topology_class=_PEL, conditions=cond,
            mechanism=MechanismClaim(knob="vos", metric="a_vos", series_ref="pel",
                quant=QuantTest(kind="elasticity", band=(-0.6, -0.2)),
                narrative="σ_Vos decreases as a power law with device area (Pelgrom-type); the exponent is "
                          "model/node-specific"))])
    res = run_recipe(rec, _AreaRunner(), corpus=SpecimenCorpus(str(tmp_path)), project=False)
    assert {c.id: c.verdict for c in res.claim_cards}["pel"] in (
        VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT)

