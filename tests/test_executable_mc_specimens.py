"""Stat-QT Tasks 5/7/9: MC specimen renderers (emit-checks), recipes (mock verdicts), and native
sky130 integration (docker-gated). The recipe + integration cases are appended in T7/T9."""
from openclaw_brain.knowledge.executable.mc_templates import (
    render_comparator_fpn_mc,
    render_ota5t_offset_mc,
)


def test_ota_offset_mc_deck_is_tt_mm_montecarlo():
    d = render_ota5t_offset_mc(mc_runs=30)
    assert '.lib "__LIBPATH__" tt_mm' in d              # mismatch corner
    assert "dowhile mc < 30" in d and "reset" in d       # resample per run
    assert "v(out)" in d and "echo RDATA vos" in d


def test_comparator_fpn_mc_deck_sweeps_trip():
    d = render_comparator_fpn_mc(mc_runs=30)
    assert '.lib "__LIBPATH__" tt_mm' in d
    assert "dc Vin" in d and "meas dc" in d and "v(out)=0.9" in d
    assert "echo RDATA vos" in d


# ── Task 7: recipes (mock-runner verdicts) ──
from openclaw_brain.knowledge.executable.corpus import SpecimenCorpus
from openclaw_brain.knowledge.executable.executor import run_recipe
from openclaw_brain.knowledge.executable.models import VerdictClass
from openclaw_brain.knowledge.executable.seeds import _comparator_fpn_recipe, _ota5t_offset_recipe


class _OffsetRunner:
    def measure(self, deck, timeout=300):   # ~5mV σ -> 3σ≈15mV / k·σ≈17mV
        return {"vos": [(float(i), 0.005 * (1 if i % 2 else -1) - 0.002) for i in range(30)]}


def test_ota_offset_specimen_verified_and_caveated(tmp_path):
    res = run_recipe(_ota5t_offset_recipe(), _OffsetRunner(), corpus=SpecimenCorpus(str(tmp_path)), project=False)
    card = {c.id: c for c in res.claim_cards}["ota5t_vos"]
    assert card.verdict == VerdictClass.VERIFIED
    assert "CDS" in (card.mechanism.narrative or "")          # faithfulness caveat present
    assert card.scope.get("statistical", "none") != "none"     # mismatch axis recorded
    assert card.scope.get("pdk") == "sky130"                   # node first-class


def test_comparator_fpn_worst_of_n(tmp_path):
    res = run_recipe(_comparator_fpn_recipe(), _OffsetRunner(), corpus=SpecimenCorpus(str(tmp_path)), project=False)
    assert {c.id: c.verdict for c in res.claim_cards}["col_fpn"] in (
        VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT)


# ── Task 9: native sky130 MC integration (docker-gated; each test uses its own $HOME workdir) ──
import os

import pytest


def _wd(tmp_path):
    # NgspiceRunner workdir MUST be under $HOME — Colima mounts $HOME, NOT pytest's /private/var tmp_path.
    # tmp_path.name is unique per test, so these are concurrency-safe.
    return os.path.expanduser(f"~/.openclaw_brain/sim_{tmp_path.name}")

from openclaw_brain.knowledge.executable.runner import NgspiceRunner
from openclaw_brain.knowledge.executable.seeds import (
    _comparator_fpn_recipe, _ota5t_gbw_corner_recipe, _ota5t_pelgrom_recipe)

_HAS_NGSPICE = NgspiceRunner().available()


@pytest.mark.skipif(not _HAS_NGSPICE, reason="IIC-OSIC-TOOLS image not present")
def test_ota_offset_real_mc_verified_sky130(tmp_path):
    rec = _ota5t_offset_recipe()
    rec.conditions.mc_runs = 30                       # 30 runs: 3σ≈14mV has big margin to the 20mV bound
    res = run_recipe(rec, NgspiceRunner(workdir=_wd(tmp_path)), corpus=SpecimenCorpus(str(tmp_path)), project=False)
    assert {c.id: c.verdict for c in res.claim_cards}["ota5t_vos"] in (
        VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT)


@pytest.mark.skipif(not _HAS_NGSPICE, reason="IIC-OSIC-TOOLS image not present")
def test_ota_pelgrom_real_powerlaw_sky130(tmp_path):
    # real sky130 MC over 3 areas -> elasticity certifies σ decreases as a power law of area (sky130
    # exponent ~-0.375, inside the Pelgrom-type band [-0.6,-0.2]); NOT ideal Pelgrom (which the open
    # model does not obey — see the design spec's honest rev).
    rec = _ota5t_pelgrom_recipe()
    rec.conditions.mc_runs = 120
    res = run_recipe(rec, NgspiceRunner(workdir=_wd(tmp_path)), corpus=SpecimenCorpus(str(tmp_path)), project=False)
    assert {c.id: c.verdict for c in res.claim_cards}["ota5t_pelgrom"] in (
        VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT)


@pytest.mark.skipif(not _HAS_NGSPICE, reason="IIC-OSIC-TOOLS image not present")
def test_statistical_refuted_tight_bound_sky130(tmp_path):
    rec = _ota5t_offset_recipe()
    rec.conditions.mc_runs = 30
    rec.claim_cards[0].mechanism.quant.bound = 0.005   # 5mV vs ~14mV measured -> REFUTED
    res = run_recipe(rec, NgspiceRunner(workdir=_wd(tmp_path)), corpus=SpecimenCorpus(str(tmp_path)), project=False)
    assert {c.id: c.verdict for c in res.claim_cards}["ota5t_vos"] == VerdictClass.REFUTED


@pytest.mark.skipif(not _HAS_NGSPICE, reason="IIC-OSIC-TOOLS image not present")
def test_corner_refuted_tight_bound_sky130(tmp_path):
    rec = _ota5t_gbw_corner_recipe()
    rec.claim_cards[0].mechanism.quant.bound = 30.2e6   # ss(29.97)/fs(30.01) fail the all-corners gate
    res = run_recipe(rec, NgspiceRunner(workdir=_wd(tmp_path)), corpus=SpecimenCorpus(str(tmp_path)), project=False)
    assert {c.id: c.verdict for c in res.claim_cards}["ota5t_gbw_corner"] == VerdictClass.REFUTED
