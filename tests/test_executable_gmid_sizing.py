"""(d) gm/ID-driven input-pair sizing wired into the executor.

Focused scope: the executor sizes the Miller-OTA INPUT PAIR by gm/ID (sets gm1 -> GBW) while the
tail-fixed current id1 and downstream bias stay put. Full multi-device synthesis is a separate
milestone.

Unit (synthetic GmidTable, no docker): the sizing arithmetic — float multiplicity (NOT int(), which
would round a valid fractional device to 0 and delete the input pair), W1:=w_char so the device is
m units of the characterized unit, and the surfaced GBW prediction. Plus backward-compat: a recipe
without method='gmid_lookup' takes the pure seed/default path (no characterization, no prediction).

Integration (docker-gated): real clean-room characterize -> size -> real sky130 OTA AC -> the
realized GBW tracks the gm/ID prediction (within the first-order body-effect/VDS tolerance).
"""

from __future__ import annotations

import math

import pytest

from openclaw_brain.knowledge.executable import executor as executor_mod
from openclaw_brain.knowledge.executable.corpus import SpecimenCorpus
from openclaw_brain.knowledge.executable.executor import run_recipe
from openclaw_brain.knowledge.executable.gmid import GmidPoint, GmidTable
from openclaw_brain.knowledge.executable.models import (
    ClaimCard, AnalogPVT, MechanismClaim, QuantTest, VerdictClass, VerificationRecipe,
)
from openclaw_brain.knowledge.executable.runner import NgspiceRunner

OTA = "miller_ota_2stage_nmos_in"
COND = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)
_CERTIFIED = {VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT}


def _pt(gm_id, id_w, w_char=10.0, gds=2e-6, cgg=1e-14, vgs=0.7):
    idd = id_w * w_char
    return GmidPoint(vgs=vgs, idd=idd, gm=gm_id * idd, gds=gds, cgg=cgg)


# Synthetic table: ascending in gm/ID, [10 -> 30µA unit, 20 -> 10µA unit]; id_unit(15) interpolates
# to 20µA. With the OTA default bias (IREF=10u, W5/W8=4/2) -> Itail=20µA -> id1=10µA, so the sized
# multiplicity m = id1/id_unit(15) = 10/20 = 0.5 (deliberately fractional — the int()-rounds-to-0 trap).
_SYNTH_TABLE = GmidTable([_pt(20, 1e-6, vgs=0.6), _pt(10, 3e-6, vgs=1.0)],
                         w_char_um=10.0, l_um=0.5, device="nfet", vds=0.9)


class _FakeOtaRunner:
    """Returns a GBW-vs-Cc series (gbw ∝ 1/Cc) for the AC sweep; characterize is monkeypatched out."""

    def measure(self, deck, timeout=300):
        return {"cc": [(500e-15, 48e6), (1000e-15, 24e6), (2000e-15, 12e6)]}


def _gmid_ota_recipe(gm_id=15.0):
    return VerificationRecipe(
        topology_class=OTA,
        build={"method": "template", "template_ref": "miller_ota_ac"},
        sizing={"method": "gmid_lookup", "targets": {"gm_id": gm_id}},
        conditions=COND,
        sweeps=[{"analysis": "ac", "knob": "Cc",
                 "points": ["500f", "1000f", "2000f"], "measure": ["gbw_hz"]}],
        claim_cards=[ClaimCard(
            id="cc_gbw", topology_class=OTA, conditions=COND,
            mechanism=MechanismClaim(knob="Cc", metric="gbw_hz", series_ref="x",
                                     quant=QuantTest(kind="direction", sign="-"),
                                     narrative="Miller compensation: GBW = gm1/(2*pi*Cc), so GBW falls as Cc rises"))],
    )


def _seed_ota_recipe():
    """An OTA recipe with NO gmid_lookup — must take the pure seed/default path (no prediction)."""
    r = _gmid_ota_recipe()
    r.sizing = {"seed": {"Cc": "1p"}}
    return r


# ── unit ──


def test_gmid_lookup_sizes_input_pair(monkeypatch, tmp_path):
    monkeypatch.setattr(executor_mod, "characterize", lambda runner, **kw: _SYNTH_TABLE)
    result = run_recipe(_gmid_ota_recipe(gm_id=15.0), _FakeOtaRunner(),
                        corpus=SpecimenCorpus(str(tmp_path)), project=False)

    pred = result.sizing_prediction
    assert pred is not None
    assert pred["gm_id"] == 15.0
    assert pred["id1_a"] == pytest.approx(10e-6)            # IREF·(W5/W8)/2 = 10µ·2/2
    assert pred["m"] == pytest.approx(0.5)                  # 10µA / id_unit(15)=20µA
    assert pred["w_char_um"] == 10.0
    assert pred["gm1_pred_s"] == pytest.approx(150e-6)      # gm/ID · id1 = 15 · 10µA
    assert pred["gbw_pred_hz"] == pytest.approx(150e-6 / (2 * math.pi * 1e-12), rel=1e-6)

    # the sizing flowed into the specimen netlist: W1 snapped to the unit width, M1 = the float m
    netlist = result.specimen.netlist
    assert "W1=10" in netlist
    assert "M1=0.5" in netlist
    # claim still judged against the (rewritten) routed series
    assert result.claim_cards[0].verdict == VerdictClass.VERIFIED   # gbw falls with Cc


def test_fractional_multiplicity_is_not_truncated(monkeypatch, tmp_path):
    """Regression: m=0.5 must survive as a float — int() would make it 0 and delete the input pair."""
    monkeypatch.setattr(executor_mod, "characterize", lambda runner, **kw: _SYNTH_TABLE)
    result = run_recipe(_gmid_ota_recipe(gm_id=15.0), _FakeOtaRunner(), project=False)
    assert 0 < result.sizing_prediction["m"] < 1
    assert "M1=0\n" not in result.specimen.netlist
    assert "m=0 " not in result.specimen.netlist        # the rendered device line, not zeroed


def test_higher_gmid_predicts_higher_gbw(monkeypatch, tmp_path):
    """gm/ID is the knob: a higher target predicts proportionally higher gm1 (hence GBW)."""
    monkeypatch.setattr(executor_mod, "characterize", lambda runner, **kw: _SYNTH_TABLE)
    lo = run_recipe(_gmid_ota_recipe(gm_id=10.0), _FakeOtaRunner(), project=False).sizing_prediction
    hi = run_recipe(_gmid_ota_recipe(gm_id=20.0), _FakeOtaRunner(), project=False).sizing_prediction
    # id1 is fixed (tail-set); gm1 = gm/ID · id1, so the ratio is exactly the gm/ID ratio
    assert hi["gbw_pred_hz"] / lo["gbw_pred_hz"] == pytest.approx(2.0)


def test_non_gmid_recipe_skips_sizing(monkeypatch, tmp_path):
    """Backward compat: without method='gmid_lookup', characterize is never called and there is
    no prediction — the pure seed/default path is byte-for-byte the prior behavior."""
    def _boom(*a, **k):
        raise AssertionError("characterize must not run for a non-gmid recipe")
    monkeypatch.setattr(executor_mod, "characterize", _boom)
    result = run_recipe(_seed_ota_recipe(), _FakeOtaRunner(), project=False)
    assert result.sizing_prediction is None
    assert "M1=1" in result.specimen.netlist          # default multiplicity unchanged


# ── integration (docker-gated): real sky130 ──


def test_gmid_input_pair_sizing_end_to_end_sky130(tmp_path):
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; integration test needs the sim container")
    recipe = _gmid_ota_recipe(gm_id=15.0)
    result = run_recipe(recipe, runner, corpus=SpecimenCorpus(str(tmp_path)), project=False)

    pred = result.sizing_prediction
    assert pred is not None and pred["gm_id"] == 15.0
    gbw_series = result.canonical["cc_gbw_hz"]
    assert len(gbw_series) == 3
    gbw_at_1p = min(gbw_series, key=lambda xy: abs(xy[0] - 1e-12))[1]
    # The realized GBW tracks the gm/ID prediction with a STABLE ~0.81 factor (measured: 0.811–0.816
    # across gm/ID 12/15/18). That constant offset is body effect — the table is characterized at
    # VSB=0 but the in-circuit input source sits at Vtail>0 — so gm/ID stays a calibratable
    # first-order tool, not a coincidence. Band brackets the measured 0.81 with PVT margin.
    assert 0.70 * pred["gbw_pred_hz"] < gbw_at_1p < 0.95 * pred["gbw_pred_hz"]
    # Non-tautological: the gm/ID-sized GBW (~19.5MHz) is distinctly below the default M1=1 point
    # (~24.3MHz) — sizing genuinely re-set the input-pair inversion level.
    assert gbw_at_1p < 22e6
    # GBW still falls with Cc (the Miller claim holds on the gm/ID-sized device)
    assert result.claim_cards[0].verdict in _CERTIFIED
    assert result.spec_id.startswith("sha256:")
