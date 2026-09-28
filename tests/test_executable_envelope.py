"""Tests for the E2a-I1 analytic-envelope substrate (spec: docs/superpowers/specs/
2026-07-05-e2a-analytic-envelope.md):

  - each of the 4 pilot EnvelopeModel formulas returns its hand-anchored value on a FIXED op dict
    (a regression pin against the live sky130 numbers captured in scratchpad/probe_envelope_*.py);
  - CONCUR / VIOLATED / NA band logic;
  - `envelope_check` independence: predicted depends ONLY on op_quantities, never on measured_value
    (the load-bearing non-circularity property a reviewer must be able to verify);
  - `oracle.py` byte-immutability guard (sha256 pin — the envelope is a SEPARATE instrument, never a
    change to the oracle);
  - the envelope scope-tag renders in `_scope_inline` (agent.py), following the intervention/
    idealization pattern exactly, undetachable and never silently dropping a VIOLATED status;
  - ONE live sky130 smoke (docker-gated) closing the loop for all 4 pilots end-to-end (Q1 smoke).
"""

from __future__ import annotations

import hashlib
import inspect
import math

import pytest

from openclaw_brain.agent import _scope_inline
from openclaw_brain.knowledge.executable import oracle as oracle_module
from openclaw_brain.knowledge.executable.envelope import (
    REGISTRY,
    EnvelopeVerdict,
    envelope_check,
    measure_op_quantities,
    render_cs_op_dump,
    render_miller_ota_op_dump,
    render_ota5t_op_dump,
    stamp_envelope_scope,
)
from openclaw_brain.knowledge.executable.runner import NgspiceRunner, parse_unit
from openclaw_brain.knowledge.executable.templates import (
    DEFAULT_CS_SIZING,
    DEFAULT_OTA5T_SIZING,
    DEFAULT_OTA_SIZING,
    render_common_source_ac,
    render_ota_5t_ac,
    render_miller_ota_ac,
)

MILLER = "miller_ota_2stage_nmos_in"
OTA5T = "ota_5t_nmos_in"
CS = "common_source_active_load_nmos"


# ===================================================================================================
# §1 — hand-anchored fixed-op-dict pins (live sky130 numbers frozen 2026-07-05, see envelope.py's
# per-formula docstrings + scratchpad/probe_envelope_*.py for the live transcripts these come from)
# ===================================================================================================

# miller_ota_2stage_nmos_in / gbw_hz -- probe_envelope_miller_gbw.py
_OP_MILLER = {"gm1": 1.86053e-4, "Cc": 1e-12}
_MILLER_MEASURED_GBW = 2.42641e7    # oracle's own AC sweep, nominal Cc=1p point

# ota_5t_nmos_in / av0_db + gbw_hz -- probe_envelope_ota5t.py
_OP_OTA5T = {"gm1": 1.9643e-4, "gds_n": 1.71937e-6, "gds_p": 9.70258e-7, "CL": 1e-12}
_OTA5T_MEASURED_AV0_DB = 37.1104
_OTA5T_MEASURED_GBW = 3.02783e7

# common_source_active_load_nmos / av0_db -- probe_envelope_cs.py
_OP_CS = {"gm": 1.89283e-4, "gds_n": 1.72924e-6, "gds_p": 9.41343e-7}
_CS_MEASURED_AV0_DB = 37.0101


def test_miller_gbw_formula_hand_anchored():
    model = REGISTRY[(MILLER, "gbw_hz")]
    predicted, band = model.fn(_OP_MILLER)
    expected = _OP_MILLER["gm1"] / (2 * math.pi * _OP_MILLER["Cc"])
    assert predicted == pytest.approx(expected, rel=1e-12)
    assert predicted == pytest.approx(2.96113e7, rel=1e-4)   # the live-anchored figure
    assert band == pytest.approx(0.25)


def test_ota5t_av0_formula_hand_anchored():
    model = REGISTRY[(OTA5T, "av0_db")]
    predicted, band = model.fn(_OP_OTA5T)
    expected = 20 * math.log10(_OP_OTA5T["gm1"] / (_OP_OTA5T["gds_n"] + _OP_OTA5T["gds_p"]))
    assert predicted == pytest.approx(expected, rel=1e-12)
    assert predicted == pytest.approx(37.2703, abs=1e-3)
    assert band == pytest.approx(0.15)


def test_ota5t_gbw_formula_hand_anchored():
    model = REGISTRY[(OTA5T, "gbw_hz")]
    predicted, band = model.fn(_OP_OTA5T)
    expected = _OP_OTA5T["gm1"] / (2 * math.pi * _OP_OTA5T["CL"])
    assert predicted == pytest.approx(expected, rel=1e-12)
    assert predicted == pytest.approx(3.12628e7, rel=1e-4)
    assert band == pytest.approx(0.15)


def test_cs_av0_formula_hand_anchored():
    model = REGISTRY[(CS, "av0_db")]
    predicted, band = model.fn(_OP_CS)
    expected = 20 * math.log10(_OP_CS["gm"] / (_OP_CS["gds_n"] + _OP_CS["gds_p"]))
    assert predicted == pytest.approx(expected, rel=1e-12)
    assert predicted == pytest.approx(37.011, abs=1e-2)
    assert band == pytest.approx(0.15)


# ===================================================================================================
# §2 — CONCUR / VIOLATED / NA band logic (using the pinned op dicts + their live-measured values)
# ===================================================================================================


def test_envelope_check_concur_on_all_4_pilots_at_live_anchored_values():
    """The 4 pilots' own hand-anchored live numbers (§1) all land inside their formula's band --
    Q1's "already-VERIFIED laws should concur" prediction, on sky130, at the substrate level."""
    v1 = envelope_check(MILLER, "gbw_hz", "sky130A", _MILLER_MEASURED_GBW, _OP_MILLER)
    assert v1.status == "CONCUR"
    v2 = envelope_check(OTA5T, "av0_db", "sky130A", _OTA5T_MEASURED_AV0_DB, _OP_OTA5T)
    assert v2.status == "CONCUR"
    v3 = envelope_check(OTA5T, "gbw_hz", "sky130A", _OTA5T_MEASURED_GBW, _OP_OTA5T)
    assert v3.status == "CONCUR"
    v4 = envelope_check(CS, "av0_db", "sky130A", _CS_MEASURED_AV0_DB, _OP_CS)
    assert v4.status == "CONCUR"


def test_envelope_check_violated_far_outside_band():
    # a measured value wildly off from the closed-form prediction (e.g. a planted wrong-node
    # measurement, spec Q2) must VIOLATE, never be silently rounded into CONCUR.
    v = envelope_check(MILLER, "gbw_hz", "sky130A", _MILLER_MEASURED_GBW * 5, _OP_MILLER)
    assert v.status == "VIOLATED"
    assert v.ratio == pytest.approx(5 * (_MILLER_MEASURED_GBW / (_OP_MILLER["gm1"] / (2 * math.pi * _OP_MILLER["Cc"]))))


def test_envelope_check_na_for_unregistered_topology_metric():
    v = envelope_check("current_mirror_simple_nmos", "iout_a", "sky130A", 1.0e-5, {"gm": 1e-4})
    assert v.status == "NA"
    assert v.predicted is None
    assert v.ratio is None
    assert v.band is None
    # NA is honest, not a pass -- must not accidentally coincide with a CONCUR-shaped record
    assert v.status != "CONCUR"


def test_envelope_check_db_metric_ratio_is_on_the_linear_quantity():
    # a +6.02dB measured-vs-predicted gap is exactly a 2x linear ratio (20*log10(2) = 6.02...)
    op = {"gm1": _OP_OTA5T["gm1"], "gds_n": _OP_OTA5T["gds_n"], "gds_p": _OP_OTA5T["gds_p"]}
    model = REGISTRY[(OTA5T, "av0_db")]
    predicted_db, _ = model.fn(op)
    v = envelope_check(OTA5T, "av0_db", "sky130A", predicted_db + 20 * math.log10(2), op)
    assert v.ratio == pytest.approx(2.0, rel=1e-9)
    assert v.status == "VIOLATED"   # 2x is far outside the +-15% band


# ===================================================================================================
# §3 — independence: `predicted` depends ONLY on op_quantities, never on measured_value (the
# reviewer-verifiable non-circularity property, spec §2's "CRITICAL INDEPENDENCE RULE")
# ===================================================================================================


def test_envelope_check_predicted_independent_of_measured_value():
    """Same op_quantities, wildly different measured_value -> IDENTICAL predicted + band. If
    `predicted` ever depended on the oracle's measured series/value, this would fail -- the whole
    point of a second, independent instrument is that path B never reads path A's answer before
    computing its own."""
    v_low = envelope_check(OTA5T, "av0_db", "sky130A", 0.001, _OP_OTA5T)
    v_high = envelope_check(OTA5T, "av0_db", "sky130A", 500.0, _OP_OTA5T)  # a dB metric: stays in float64 range
    assert v_low.predicted == v_high.predicted
    assert v_low.band == v_high.band
    # only ratio/status differ, driven purely by the differing measured_value
    assert v_low.ratio != v_high.ratio


def test_envelope_check_never_passed_a_series():
    """`envelope_check`'s 4th positional arg is a scalar (`measured_value`); this test documents +
    enforces that calling it with a SERIES (a list of (x, y) tuples, what the oracle actually
    measures) is a caller error the function does not silently tolerate as a valid scalar path --
    it must raise on the arithmetic (TypeError), never coerce/degrade into a wrong-but-quiet number."""
    with pytest.raises(TypeError):
        envelope_check(MILLER, "gbw_hz", "sky130A", [(1e-12, 2.4e7)], _OP_MILLER)  # a series, not a scalar


# ===================================================================================================
# §4 — oracle.py byte-immutability guard (E2a is a SEPARATE instrument; oracle.py stays untouched)
# ===================================================================================================

_ORACLE_SHA256 = "eebdbef495074b3cb0c728bb57435f72d4aab8c34ece4bb35962caa0ea9b1498"


def test_oracle_py_byte_immutable():
    path = inspect.getsourcefile(oracle_module)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert digest == _ORACLE_SHA256, (
        "oracle.py changed during E2a-I1 -- the envelope is a SEPARATE instrument alongside the "
        "oracle, never a modification of it (the mechanism-never-fact / same-input-same-verdict "
        "invariant). If oracle.py genuinely needed to change, that is a deliberate, reviewed, dated "
        "decision -- not something E2a-I1 should do silently."
    )


# ===================================================================================================
# §5 — scope-tag render: `_scope_inline` renders the envelope tag, following the intervention/
# idealization pattern exactly (additive, undetachable, VIOLATED never silently dropped)
# ===================================================================================================


def test_scope_inline_appends_envelope_concur():
    s = {"pdk": "sky130", "corners": ["sky130/tt/27/1.8"], "statistical": "none",
         "envelope": {"status": "CONCUR", "ratio": 1.08}}
    assert _scope_inline(s) == "sky130/tt/27/1.8/envelope:concur@1.08"


def test_scope_inline_appends_envelope_violated():
    s = {"pdk": "sky130", "corners": ["sky130/tt/27/1.8"], "statistical": "none",
         "envelope": {"status": "VIOLATED", "ratio": 3.4}}
    assert _scope_inline(s) == "sky130/tt/27/1.8/envelope:VIOLATED@3.40"


def test_scope_inline_envelope_na_renders_without_ratio():
    s = {"pdk": "sky130", "corners": ["sky130/tt/27/1.8"], "statistical": "none",
         "envelope": {"status": "NA"}}
    assert _scope_inline(s) == "sky130/tt/27/1.8/envelope:NA"


def test_scope_inline_omits_envelope_tag_when_absent():
    # every pre-E2a card (no "envelope" key at all) renders exactly as before -- backcompat pin
    s = {"pdk": "sky130", "corners": ["sky130/tt/27/1.8"], "statistical": "none"}
    assert _scope_inline(s) == "sky130/tt/27/1.8"


def test_scope_inline_envelope_tag_rides_after_intervention_tag():
    # appended LAST, after every other existing scope axis including intervention (undetachable)
    s = {"pdk": "sky130", "corners": ["sky130/tt/27/1.8"], "statistical": "none",
         "intervention": "ff_break", "idealization": "ideal buffer",
         "envelope": {"status": "CONCUR", "ratio": 1.02}}
    assert _scope_inline(s) == "sky130/tt/27/1.8/intervention:ff_break(idealized)/envelope:concur@1.02"


def test_stamp_envelope_scope_is_additive_and_does_not_mutate_input():
    original = {"pdk": "sky130", "corners": ["sky130/tt/27/1.8"], "statistical": "none"}
    verdict = EnvelopeVerdict(OTA5T, "av0_db", "sky130A", 37.27, 37.11, 0.9818, 0.15, "CONCUR")
    stamped = stamp_envelope_scope(original, verdict)
    assert "envelope" not in original                       # no in-place mutation of the caller's dict
    assert stamped["envelope"] == {"status": "CONCUR", "ratio": pytest.approx(0.9818)}
    assert stamped["pdk"] == "sky130"                        # every other key preserved


def test_stamp_envelope_scope_na_has_no_ratio_key():
    verdict = EnvelopeVerdict("current_mirror_simple_nmos", "iout_a", "sky130A", None, 1e-5, None, None, "NA")
    stamped = stamp_envelope_scope({}, verdict)
    assert stamped["envelope"] == {"status": "NA"}
    assert "ratio" not in stamped["envelope"]


# ===================================================================================================
# §6 — live (docker-gated) smoke: all 4 pilot envelopes CONCUR on real sky130, end-to-end through
# the OP-dump probes + NgspiceRunner (Q1 smoke, spec §5's "Live (gated)" test-plan item)
# ===================================================================================================


def test_live_all_4_pilots_concur_on_sky130():
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; integration test needs the sim container")

    # miller_ota_2stage_nmos_in / gbw_hz
    op = measure_op_quantities(MILLER, DEFAULT_OTA_SIZING, runner=runner)
    ac = runner.measure(render_miller_ota_ac(DEFAULT_OTA_SIZING, knob="Cc", metric="gbw_hz",
                                              points=["250f", "500f", "1000f", "2000f", "4000f"]))
    measured = dict(ac["cc"])[parse_unit(DEFAULT_OTA_SIZING["Cc"])]
    v = envelope_check(MILLER, "gbw_hz", "sky130A", measured, op)
    assert v.status == "CONCUR", v

    # ota_5t_nmos_in / av0_db + gbw_hz
    op5t = measure_op_quantities(OTA5T, DEFAULT_OTA5T_SIZING, runner=runner)
    for metric in ("av0_db", "gbw_hz"):
        ac5t = runner.measure(render_ota_5t_ac(DEFAULT_OTA5T_SIZING, knob="CL", metric=metric,
                                                points=["500f", "1000f", "2000f", "4000f"]))
        measured5t = dict(ac5t["cl"])[parse_unit(DEFAULT_OTA5T_SIZING["CL"])]
        v5t = envelope_check(OTA5T, metric, "sky130A", measured5t, op5t)
        assert v5t.status == "CONCUR", (metric, v5t)

    # common_source_active_load_nmos / av0_db
    opcs = measure_op_quantities(CS, DEFAULT_CS_SIZING, runner=runner)
    accs = runner.measure(render_common_source_ac(DEFAULT_CS_SIZING, knob="Iref", metric="av0_db",
                                                   points=["2u", "5u", "10u", "20u", "40u"]))
    measuredcs = dict(accs["iref"])[parse_unit(DEFAULT_CS_SIZING["IREFV"])]
    vcs = envelope_check(CS, "av0_db", "sky130A", measuredcs, opcs)
    assert vcs.status == "CONCUR", vcs


def test_live_op_dump_renderers_reuse_unmutated_bodies():
    """Sanity: the OP-dump render helpers embed the SAME body strings templates.py's own AC-sweep
    renderers use (imported, not copy-pasted) -- a change to the pilot bodies is automatically picked
    up here rather than silently diverging."""
    from openclaw_brain.knowledge.executable.templates import _CS_BODY, _OTA5T_BODY, _OTA_BODY

    assert _OTA_BODY in render_miller_ota_op_dump(DEFAULT_OTA_SIZING)
    assert _OTA5T_BODY in render_ota5t_op_dump(DEFAULT_OTA5T_SIZING)
    assert _CS_BODY in render_cs_op_dump(DEFAULT_CS_SIZING)
