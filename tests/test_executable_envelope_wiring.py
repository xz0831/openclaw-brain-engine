"""Tests for the E2a-I2 executor wiring (spec: docs/superpowers/specs/
2026-07-05-e2a-analytic-envelope.md §2/§4, I2 deliverables 1/3/5):

  §1 — executor wiring (fake runner, no docker): a pilot (topology_class, metric) claim gets a
       CONCUR/VIOLATED envelope tag stamped additively on its scope; a non-pilot claim in the SAME
       recipe gets NO tag at all (the gate is tight — envelope.REGISTRY membership, nothing broader);
       an OP-dump sim failure degrades to "no tag", never crashes run_recipe or touches the oracle's
       own verdict; two pilot claims sharing one op-point (OTA5T's av0_db + gbw_hz) cost exactly ONE
       extra `.op` sim (cached), not two.
  §2 — why()/scope render end-to-end: `_scope_inline` (agent.py) renders a REAL envelope tag produced
       by the executor's own stamping loop (not just a hand-built literal dict, which
       test_executable_envelope.py's §5 already covers at the unit level).
  §3 — Q2, the independence/planted-error proof (docker-gated): a SCRATCH (test-local, never
       templates.py) mutation of the ota_5t_nmos_in av0_db AC deck that measures the WRONG node
       (the diode-loaded mirror node `o1` instead of the true high-impedance output `out`) — the
       oracle's own invariance verdict on that (wrong) series still VERIFIES (o1's gain does not
       depend on CL at all, so it is even FLATTER than the true node), while `envelope_check` against
       the SAME (unmutated, path-B) op-quantities VIOLATES — proving the envelope is a genuinely
       independent second instrument, not a redundant restatement of the sim.
"""

from __future__ import annotations

import pytest

from openclaw_brain.agent import _scope_inline
from openclaw_brain.knowledge.executable.envelope import envelope_check, measure_op_quantities
from openclaw_brain.knowledge.executable.executor import run_recipe
from openclaw_brain.knowledge.executable.models import (
    ClaimCard, MechanismClaim, QuantTest, VerdictClass, VerificationRecipe,
)
from openclaw_brain.knowledge.executable.oracle import judge_quant
from openclaw_brain.knowledge.executable.runner import NgspiceRunner, parse_unit
from openclaw_brain.knowledge.executable.templates import DEFAULT_OTA5T_SIZING, render_ota_5t_ac

MILLER = "miller_ota_2stage_nmos_in"
OTA5T = "ota_5t_nmos_in"
COND = {"corner": "tt", "temp_c": 27.0, "vdd": 1.8}
_CERTIFIED = {VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT}


def _claim(tclass, cid, knob, metric, series_ref, quant):
    return ClaimCard(id=cid, topology_class=tclass, conditions=COND,
                     mechanism=MechanismClaim(knob=knob, metric=metric, series_ref=series_ref,
                                              quant=quant, narrative="wiring test"))


class _EnvelopeAwareFakeRunner:
    """Recognizes the E2a OP-dump probe deck (a bare `.control\\nop\\n` analysis, no `foreach` sweep
    loop) and returns canned gm/gds device-query values; every other deck falls back to matching its
    metric's distinguishing `.meas` line, exactly like test_executable_executor.py's own `_FakeRunner`
    (`entries`: {meas_signature: (series_key, series)})."""

    def __init__(self, entries: dict, op_quantities: dict, fail_op: bool = False):
        self._entries = entries
        self._op = op_quantities
        self._fail_op = fail_op
        self.op_dump_calls = 0
        self.decks: list[str] = []

    def measure(self, deck: str, pdk: str = "sky130A", timeout: int = 300):
        self.decks.append(deck)
        if "\nop\n" in deck:
            self.op_dump_calls += 1
            if self._fail_op:
                raise RuntimeError("simulated OP-dump sim failure")
            return {k: [(0.0, v)] for k, v in self._op.items()}
        for meas_sig, (series_key, series) in self._entries.items():
            if meas_sig in deck:
                return {series_key: series}
        return {}


def _meas_sig(metric: str) -> str:
    from openclaw_brain.knowledge.executable.templates import OTA_METRIC_MEAS
    for ln in OTA_METRIC_MEAS[metric].splitlines():
        if "meas " in ln:
            return ln.strip()
    return OTA_METRIC_MEAS[metric].strip()


# hand-anchored live sky130 numbers (SAME as envelope.py's per-formula docstrings /
# test_executable_envelope.py's §1/§6 — reused, not re-derived, so every number below is independently
# checkable against the substrate's own committed hand-anchor).
_MILLER_GM1 = 1.86053e-4
_MILLER_GBW_AT_NOMINAL = 2.42641e7   # CONCUR: ratio 0.8194, band .25
_OTA5T_OP = {"gm1": 1.9643e-4, "gds_n": 1.71937e-6, "gds_p": 9.70258e-7}
_OTA5T_AV0_AT_NOMINAL = 37.1104      # CONCUR: ratio 0.9818, band .15
_OTA5T_GBW_AT_NOMINAL = 3.02783e7    # CONCUR: ratio 0.9685, band .15


def _miller_recipe():
    return VerificationRecipe(
        topology_class=MILLER, build={"method": "template", "template_ref": "miller_ota_ac"},
        conditions=COND, sizing={"method": "given", "seed": {}},
        sweeps=[{"analysis": "ac", "knob": "Cc", "points": ["250f", "500f", "1000f", "2000f", "4000f"],
                 "measure": ["gbw_hz", "av0_db"]}],
        claim_cards=[
            _claim(MILLER, "cc_gbw_pilot", "Cc", "gbw_hz", "cc_gbw", QuantTest(kind="direction", sign="-")),
            _claim(MILLER, "cc_av0_nonpilot", "Cc", "av0_db", "cc_av0", QuantTest(kind="invariance", cov_max=0.05)),
        ],
    )


def _gbw_series(nominal_hz):
    """Monotonically-decreasing ~1/Cc series, anchored to `nominal_hz` at the Cc=1p (1000f) point."""
    return [(2.5e-13, nominal_hz * 4), (5e-13, nominal_hz * 2), (1e-12, nominal_hz),
            (2e-12, nominal_hz / 2), (4e-12, nominal_hz / 4)]


_AV0_FLAT_MILLER = [(2.5e-13, 68.9), (5e-13, 68.9), (1e-12, 68.9), (2e-12, 68.9), (4e-12, 68.9)]


# ===================================================================================================
# §1 — executor wiring: gate tightness, CONCUR/VIOLATED, degrade-on-failure, op-dump sim dedup
# ===================================================================================================


def test_pilot_claim_gets_concur_tag_non_pilot_claim_gets_none():
    runner = _EnvelopeAwareFakeRunner(
        entries={_meas_sig("gbw_hz"): ("cc", _gbw_series(_MILLER_GBW_AT_NOMINAL)),
                 _meas_sig("av0_db"): ("cc", _AV0_FLAT_MILLER)},
        op_quantities={"gm1": _MILLER_GM1},
    )
    result = run_recipe(_miller_recipe(), runner)
    by_id = {c.id: c for c in result.claim_cards}

    pilot = by_id["cc_gbw_pilot"]
    assert pilot.verdict in _CERTIFIED                       # the oracle's own verdict, untouched
    assert pilot.scope["envelope"]["status"] == "CONCUR"
    assert pilot.scope["envelope"]["ratio"] == pytest.approx(0.8194, abs=1e-3)

    nonpilot = by_id["cc_av0_nonpilot"]
    assert nonpilot.verdict in _CERTIFIED
    assert "envelope" not in nonpilot.scope                  # miller_ota/av0_db is NOT in the registry

    # PERFORMANCE gate (spec §2 requirement 1): exactly ONE extra `.op` sim, only for the pilot claim.
    assert runner.op_dump_calls == 1


def test_pilot_claim_gets_violated_tag_when_measured_far_off_envelope_but_verdict_unaffected():
    # a GBW series scaled 5x off the closed-form prediction, but STILL perfectly monotonic decreasing
    # -- the oracle's own direction verdict does not care about absolute magnitude, only shape.
    runner = _EnvelopeAwareFakeRunner(
        entries={_meas_sig("gbw_hz"): ("cc", _gbw_series(_MILLER_GBW_AT_NOMINAL * 5)),
                 _meas_sig("av0_db"): ("cc", _AV0_FLAT_MILLER)},
        op_quantities={"gm1": _MILLER_GM1},
    )
    result = run_recipe(_miller_recipe(), runner)
    pilot = next(c for c in result.claim_cards if c.id == "cc_gbw_pilot")

    assert pilot.verdict in _CERTIFIED                       # shape still passes...
    assert pilot.scope["envelope"]["status"] == "VIOLATED"   # ...but the envelope catches the magnitude
    assert pilot.scope["envelope"]["ratio"] == pytest.approx(0.8194 * 5, abs=1e-2)


def test_op_dump_sim_failure_degrades_to_no_tag_never_crashes_and_never_touches_verdict():
    runner = _EnvelopeAwareFakeRunner(
        entries={_meas_sig("gbw_hz"): ("cc", _gbw_series(_MILLER_GBW_AT_NOMINAL)),
                 _meas_sig("av0_db"): ("cc", _AV0_FLAT_MILLER)},
        op_quantities={"gm1": _MILLER_GM1}, fail_op=True,
    )
    result = run_recipe(_miller_recipe(), runner)             # must not raise
    pilot = next(c for c in result.claim_cards if c.id == "cc_gbw_pilot")

    assert pilot.verdict in _CERTIFIED                        # oracle path entirely unaffected
    assert "envelope" not in pilot.scope                      # degraded silently, no tag
    assert any("envelope check skipped" in n and "cc_gbw_pilot" in n for n in result.notes)


def test_two_pilot_claims_sharing_one_op_point_cost_exactly_one_op_dump_sim():
    """OTA5T's av0_db + gbw_hz BOTH read gm1/gds_n/gds_p at the SAME CL=1p nominal op-point -- the
    executor's cache must not re-run the `.op` sim for the second claim ("be economical")."""
    runner = _EnvelopeAwareFakeRunner(
        entries={_meas_sig("gbw_hz"): ("cl", _gbw_series(_OTA5T_GBW_AT_NOMINAL)),
                 _meas_sig("av0_db"): ("cl", [(5e-13, _OTA5T_AV0_AT_NOMINAL), (1e-12, _OTA5T_AV0_AT_NOMINAL),
                                              (2e-12, _OTA5T_AV0_AT_NOMINAL), (4e-12, _OTA5T_AV0_AT_NOMINAL)])},
        op_quantities=_OTA5T_OP,
    )
    recipe = VerificationRecipe(
        topology_class=OTA5T, build={"method": "template", "template_ref": "ota_5t_ac"},
        conditions=COND,
        sweeps=[{"analysis": "ac", "knob": "CL", "points": ["500f", "1000f", "2000f", "4000f"],
                 "measure": ["av0_db", "gbw_hz"]}],
        claim_cards=[
            _claim(OTA5T, "ota5t_gbw_pilot", "CL", "gbw_hz", "cl_gbw", QuantTest(kind="direction", sign="-")),
            _claim(OTA5T, "ota5t_av0_pilot", "CL", "av0_db", "cl_av0", QuantTest(kind="invariance", spread_max=0.5)),
        ],
    )
    result = run_recipe(recipe, runner)
    by_id = {c.id: c for c in result.claim_cards}

    assert by_id["ota5t_gbw_pilot"].scope["envelope"]["status"] == "CONCUR"
    assert by_id["ota5t_av0_pilot"].scope["envelope"]["status"] == "CONCUR"
    assert runner.op_dump_calls == 1                          # shared op-point, cached -- not 2


# ===================================================================================================
# §2 — why()/scope render end-to-end: _scope_inline on a REAL executor-produced scope dict
# ===================================================================================================


def test_scope_inline_renders_a_real_executor_produced_envelope_tag():
    runner = _EnvelopeAwareFakeRunner(
        entries={_meas_sig("gbw_hz"): ("cc", _gbw_series(_MILLER_GBW_AT_NOMINAL)),
                 _meas_sig("av0_db"): ("cc", _AV0_FLAT_MILLER)},
        op_quantities={"gm1": _MILLER_GM1},
    )
    result = run_recipe(_miller_recipe(), runner)
    pilot = next(c for c in result.claim_cards if c.id == "cc_gbw_pilot")

    inline = _scope_inline(pilot.scope)
    assert inline.endswith("/envelope:concur@0.82")           # undetachable, rides the END of the chain
    # and the non-pilot card (no "envelope" key at all) renders exactly as it always did
    nonpilot = next(c for c in result.claim_cards if c.id == "cc_av0_nonpilot")
    assert "envelope" not in _scope_inline(nonpilot.scope)


# ===================================================================================================
# §3 — Q2: the planted-error independence proof (docker-gated live sim)
# ===================================================================================================


def test_q2_planted_wrong_node_envelope_violates_while_oracle_verdict_still_passes():
    """The independence proof (spec §1 Q2 / §2). SCRATCH mutation ONLY (never templates.py): the
    rendered ota_5t_nmos_in av0_db AC deck's `.meas ... vdb(out)` is string-replaced, on the ALREADY-
    RENDERED deck text, to `vdb(o1)` -- the diode-loaded mirror node (XM1/XM3's drain) instead of the
    true high-impedance output `out` (XM2/XM4's drain). `o1`'s local gain (~gm1/gm3, a low-impedance
    diode-load node) is NOT the design's real av0, so the reported magnitude is wrong -- but CL only
    loads `out` in `_OTA5T_BODY` (`CL out 0 {CL}`), so `o1`'s own transfer function does not depend on
    CL AT ALL, making the wrong-node series even MORE invariant than the true node's own ~0.5dB
    spread. The oracle's `invariance` verdict (spread_max=0.5, the real ota5t_av0 claim's own bound)
    therefore VERIFIES on the wrong series -- exactly the "self-consistent but wrong" shape spec Q2
    demands -- while `envelope_check`, comparing that SAME wrong measured value against the UNMUTATED
    path-B op-quantities (gm1/gds_n/gds_p, read from the real `out`-referenced circuit, never touched
    by this mutation), VIOLATES.
    """
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; Q2 proof needs the sim container")

    points = ["500f", "1000f", "2000f", "4000f"]
    real_deck = render_ota_5t_ac(DEFAULT_OTA5T_SIZING, knob="CL", metric="av0_db", points=points)
    assert real_deck.count("vdb(out)") == 1                  # exactly one occurrence -- safe to replace
    mutated_deck = real_deck.replace("vdb(out)", "vdb(o1)")
    assert "vdb(o1)" in mutated_deck and "vdb(out)" not in mutated_deck

    wrong_node_series = runner.measure(mutated_deck)["cl"]
    assert len(wrong_node_series) == len(points)

    # --- Path A (oracle), mutated: the SAME QuantTest the real ota5t_av0 seed claim carries ---
    quant = QuantTest(kind="invariance", spread_max=0.5)
    verdict, note = judge_quant(quant, wrong_node_series)
    assert verdict in _CERTIFIED, (verdict, note, wrong_node_series)

    # --- Path B (envelope), UNMUTATED: gm1/gds_n/gds_p from the real circuit's own OP-dump ---
    op = measure_op_quantities(OTA5T, DEFAULT_OTA5T_SIZING, runner=runner)
    nominal_cl = parse_unit(DEFAULT_OTA5T_SIZING["CL"])
    wrong_measured_at_nominal = dict(wrong_node_series)[nominal_cl]
    env_verdict = envelope_check(OTA5T, "av0_db", "sky130A", wrong_measured_at_nominal, op)

    assert env_verdict.status == "VIOLATED", env_verdict
    # report the concrete numbers this proof rests on (also echoed in the task's final report)
    print(f"\nQ2 planted-error proof: wrong-node (o1) series={wrong_node_series}")
    print(f"oracle invariance verdict={verdict} ({note})")
    print(f"envelope: measured={wrong_measured_at_nominal:.6g} predicted={env_verdict.predicted:.6g} "
          f"ratio={env_verdict.ratio:.4f} band=±{env_verdict.band:.2f} status={env_verdict.status}")
