"""Tier-B Task 9: the infra-agnostic demo (ADR Decision 5).

The SAME claim-card shape (invariance) and the SAME oracle certify a held-output-invariant-to-pedestal
property on TWO engines — analog CDS (ngspice) and digital CDS (iverilog) — and both are VERIFIED. The
verdicts carry DISTINCT basis tags (physical-nominal vs functional). This proves the *infrastructure*
generalizes (one schema, one oracle, one projection handle every engine) — NOT that the two verdicts
are the same epistemic kind. Artifact-agnosticism is a claim about the plumbing, not the truth.
"""
from __future__ import annotations

import shutil

import pytest

from openclaw_brain.knowledge.executable.corpus import SpecimenCorpus
from openclaw_brain.knowledge.executable.executor import run_recipe
from openclaw_brain.knowledge.executable.models import (
    AnalogPVT, ClaimCard, MechanismClaim, QuantTest, VerdictClass, VerificationRecipe,
)
from openclaw_brain.knowledge.executable.runner import NgspiceRunner
from openclaw_brain.knowledge.executable.seeds import _digital_cds_recipe

_CERT = {VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT}


def _analog_cds_recipe() -> VerificationRecipe:
    cond = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)
    return VerificationRecipe(
        topology_class="cds_switched_cap_nmos",
        build={"method": "template", "template_ref": "cds_tran"},
        conditions=cond,
        sweeps=[{"analysis": "tran", "knob": "Vped", "points": ["0.5", "0.7", "0.9", "1.1", "1.3"],
                 "measure": ["vo_v"]}],
        claim_cards=[ClaimCard(
            id="cds_vo", topology_class="cds_switched_cap_nmos", conditions=cond,
            mechanism=MechanismClaim(knob="Vped", metric="vo_v", series_ref="cds_vo",
                                     quant=QuantTest(kind="invariance", cov_max=0.001),
                                     narrative="held output independent of the input pedestal (CDS)"))],
    )


class _InvariantRunner:
    """Returns an invariant 5-point series under either engine's knob key (executor picks by
    knob.lower()); 5 points clears the truncation guard for both the 4-point digital and 5-point
    analog sweeps."""
    def __init__(self, y=1.0975):
        self._s = [(float(i), y) for i in range(5)]

    def measure(self, deck, timeout=300):
        return {"ped": self._s, "vped": self._s}


def test_plumbing_generalizes_one_schema_one_oracle_two_engines(tmp_path):
    # deterministic (mock runners): the basis is set by the ENGINE, not the sim result, so this proves
    # the infrastructure generalizes without needing iverilog/docker.
    dres = run_recipe(_digital_cds_recipe(), _InvariantRunner(16.0),
                      corpus=SpecimenCorpus(str(tmp_path / "d")), project=False)
    ares = run_recipe(_analog_cds_recipe(), _InvariantRunner(1.0975),
                      corpus=SpecimenCorpus(str(tmp_path / "a")), project=False)
    d = {c.id: c for c in dres.claim_cards}["dcds_inv"]
    a = {c.id: c for c in ares.claim_cards}["cds_vo"]

    # same invariance property, certified by the same oracle, on both engines
    assert d.verdict == VerdictClass.VERIFIED and a.verdict == VerdictClass.VERIFIED
    assert d.mechanism.quant.kind == a.mechanism.quant.kind == "invariance"
    # ...but the epistemic BASIS tags are DISTINCT (the plumbing generalizes, the truth-kind does not)
    assert d.engine == "iverilog" and a.engine == "ngspice"
    assert d.basis == "functional" and a.basis == "physical-nominal"
    assert d.basis != a.basis


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog not installed")
def test_cross_engine_demo_with_real_sims(tmp_path):
    # the full proof: digital CDS through REAL iverilog VERIFIED@functional, and (if the ngspice docker
    # image is present) analog CDS through REAL ngspice VERIFIED@physical-nominal — distinct basis tags.
    dres = run_recipe(_digital_cds_recipe(), corpus=SpecimenCorpus(str(tmp_path / "d")), project=False)
    d = {c.id: c for c in dres.claim_cards}["dcds_inv"]
    assert d.verdict == VerdictClass.VERIFIED and d.basis == "functional"

    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("ngspice docker image not present; digital half proven, analog half needs the sim container")
    ares = run_recipe(_analog_cds_recipe(), runner, corpus=SpecimenCorpus(str(tmp_path / "a")), project=False)
    a = {c.id: c for c in ares.claim_cards}["cds_vo"]
    assert a.verdict in _CERT and a.basis == "physical-nominal"
    assert d.basis != a.basis        # same invariance, two real engines, distinct epistemic basis
