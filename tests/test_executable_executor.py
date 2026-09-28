"""Tests for the ② executor (knowledge/executable/executor.py) — the full closed loop.

Unit (no docker): a fake runner returns canned series, so we test the executor's own logic —
series routing (one deck per (knob, metric), re-keyed to each claim's series_ref), dedup,
oracle judging, corpus store, pure projection.

Integration (docker-gated): run a real Cc-sweep recipe on the sky130 Miller OTA through ngspice
and confirm the closed loop reproduces the GBW∝1/Cc finding end-to-end — recipe → render → sim →
oracle → corpus → projection. This is the "loop closes" demonstration.
"""

from __future__ import annotations

import pytest

from openclaw_brain.knowledge.executable.corpus import SpecimenCorpus
from openclaw_brain.knowledge.executable.executor import (
    UnsupportedTemplate,
    run_recipe,
)
from openclaw_brain.knowledge.executable.models import (
    ClaimCard, MechanismClaim, QuantTest, VerdictClass, VerificationRecipe,
)
from openclaw_brain.knowledge.executable.runner import NgspiceRunner
from openclaw_brain.knowledge.executable.templates import OTA_METRIC_MEAS
from openclaw_brain.knowledge.graph.schema import NodeLabel

OTA = "miller_ota_2stage_nmos_in"
COND = {"corner": "tt", "temp_c": 27.0, "vdd": 1.8}
_CERTIFIED = {VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT}


def _claim(cid, metric, series_ref, quant, narrative="why"):
    return ClaimCard(
        id=cid, topology_class=OTA, conditions=COND,
        mechanism=MechanismClaim(knob="Cc", metric=metric, series_ref=series_ref,
                                 quant=quant, narrative=narrative),
    )


def _cc_recipe(extra_claims=()):
    """The Specimen-One Cc sweep: GBW∝1/Cc (elasticity) + Av0 invariant to Cc."""
    claims = [
        _claim("cc_gbw_inverse", "gbw_hz", "cc_gbw",
               QuantTest(kind="elasticity", band=(-1.3, -0.4)),
               "GBW≈gm1/2πCc — the Miller cap sets the dominant pole"),
        _claim("cc_av0_invariant", "av0_db", "cc_av0",
               QuantTest(kind="invariance", cov_max=0.05),
               "DC gain is gm·ro, independent of Cc"),
        *extra_claims,
    ]
    return VerificationRecipe(
        topology_class=OTA,
        build={"method": "template", "template_ref": "miller_ota_ac"},
        conditions=COND,
        sizing={"method": "given", "seed": {}},
        sweeps=[{"analysis": "ac", "knob": "Cc",
                 "points": ["250f", "500f", "1000f", "2000f", "4000f"],
                 "measure": ["gbw_hz", "av0_db"]}],
        claim_cards=claims,
    )


# ── unit: fake runner ──


def _meas_line(metric: str) -> str:
    """The distinguishing `meas ...` line of a metric's extract string (skips the reset line)."""
    for ln in OTA_METRIC_MEAS[metric].splitlines():
        if "meas " in ln:
            return ln.strip()
    return OTA_METRIC_MEAS[metric].strip()


class _FakeRunner:
    """Returns a canned `cc` series chosen by which metric's meas string the deck contains —
    so it exercises the real render→series-key path without ngspice."""

    def __init__(self, gbw: list, av0: list):
        # key off each metric's distinguishing `meas` line (the first line is now a shared
        # `let y = 0/0` reset, so it no longer disambiguates).
        self._by_meas = {
            _meas_line("gbw_hz"): gbw,
            _meas_line("av0_db"): av0,
        }
        self.decks: list[str] = []

    def measure(self, deck: str, timeout: int = 300):
        self.decks.append(deck)
        for meas_sig, series in self._by_meas.items():
            if meas_sig in deck:
                return {"cc": series}
        return {}


_GBW = [(2.5e-13, 9.6e7), (5e-13, 4.8e7), (1e-12, 2.4e7), (2e-12, 1.2e7), (4e-12, 6.0e6)]  # ∝ 1/Cc
_AV0 = [(2.5e-13, 68.9), (5e-13, 68.9), (1e-12, 68.9), (2e-12, 68.9), (4e-12, 68.9)]        # flat


def test_run_recipe_routes_series_judges_stores_projects(tmp_path):
    corpus = SpecimenCorpus(str(tmp_path))
    runner = _FakeRunner(_GBW, _AV0)
    result = run_recipe(_cc_recipe(), runner, corpus=corpus, project=True)

    # one deck per (knob, metric): two runs, each bound to its canonical (knob, metric) routing key
    assert result.runs == 2
    assert set(result.canonical) == {"cc_gbw_hz", "cc_av0_db"}
    assert result.canonical["cc_gbw_hz"] == _GBW

    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["cc_gbw_inverse"].verdict == VerdictClass.VERIFIED          # slope -1 in band
    assert by_id["cc_av0_invariant"].verdict == VerdictClass.VERIFIED        # CoV 0 < 5%

    # corpus stored + projection produced (Specimen + 2 ClaimCard nodes)
    assert result.spec_id and result.spec_id.startswith("sha256:")
    assert (tmp_path / "specimens" / OTA).is_dir()
    labels = [n["label"] for n in result.projection.nodes]
    assert labels.count(NodeLabel.SPECIMEN) == 1
    assert labels.count(NodeLabel.CLAIM_CARD) == 2


def test_run_recipe_dedups_shared_knob_metric(tmp_path):
    # a third claim measuring the SAME (Cc, gbw_hz) must NOT trigger a second gbw run
    extra = _claim("cc_gbw_again", "gbw_hz", "cc_gbw2",
                   QuantTest(kind="direction", sign="-"))
    runner = _FakeRunner(_GBW, _AV0)
    result = run_recipe(_cc_recipe(extra_claims=(extra,)), runner, corpus=SpecimenCorpus(str(tmp_path)))
    assert result.runs == 2                                  # gbw + av0 only; gbw reused
    by_id = {c.id: c for c in result.claim_cards}            # both gbw claims share the routing key
    assert (by_id["cc_gbw_inverse"].mechanism.series_ref
            == by_id["cc_gbw_again"].mechanism.series_ref == "cc_gbw_hz")


def test_shared_author_series_ref_does_not_collide(tmp_path):
    """Regression for the bug the (a) live probe surfaced: a real author labels by sweep, so two
    claims on the same knob but different metrics share series_ref ('Cc_sweep'). The executor must
    still judge each against ITS OWN metric's series — not let the second overwrite the first."""
    c_gbw = _claim("C1", "gbw_hz", "Cc_sweep", QuantTest(kind="direction", sign="-"))
    c_av0 = _claim("C2", "av0_db", "Cc_sweep", QuantTest(kind="invariance", cov_max=0.05))
    recipe = VerificationRecipe(
        topology_class=OTA, build={"method": "template", "template_ref": "miller_ota_ac"},
        conditions=COND,
        sweeps=[{"analysis": "ac", "knob": "Cc", "points": ["250f", "4000f"],
                 "measure": ["gbw_hz", "av0_db"]}],
        claim_cards=[c_gbw, c_av0],
    )
    result = run_recipe(recipe, _FakeRunner(_GBW, _AV0), corpus=SpecimenCorpus(str(tmp_path)))
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["C1"].verdict == VerdictClass.VERIFIED       # GBW falls (not refuted vs the av0 series)
    assert by_id["C2"].verdict == VerdictClass.VERIFIED       # Av0 flat
    assert by_id["C1"].mechanism.series_ref != by_id["C2"].mechanism.series_ref  # disambiguated


def test_run_recipe_rejects_unknown_template(tmp_path):
    recipe = _cc_recipe()
    # forge a class with no executor renderer by monkeying the registry path: use an unsupported
    # template via a class that capability_for knows but executor has no renderer for is hard to
    # construct, so assert the guard fires for a template_ref the executor cannot run.
    from openclaw_brain.knowledge.executable import executor as ex
    saved = ex.RENDERERS                                     # registry relocated to templates.RENDERERS (Tier-B §1a)
    ex.RENDERERS = {}                                        # simulate "no renderers registered"
    try:
        with pytest.raises(UnsupportedTemplate):
            run_recipe(recipe, _FakeRunner(_GBW, _AV0))
    finally:
        ex.RENDERERS = saved


def test_empty_series_flags_claim_not_crash(tmp_path):
    # runner returns no data for gbw -> the claim is flagged, the loop still completes
    runner = _FakeRunner([], _AV0)
    result = run_recipe(_cc_recipe(), runner, corpus=SpecimenCorpus(str(tmp_path)))
    gbw = next(c for c in result.claim_cards if c.id == "cc_gbw_inverse")
    assert gbw.verdict == VerdictClass.FLAGGED
    assert "empty" in (gbw.verdict_note or "")          # the flag reason rides the verdict_note
    assert any("gbw_hz" in n for n in result.notes)


# ── oracle.judge() failure triage: expected (data/measurement) vs UNEXPECTED (code regression) ──
#
# The blanket `except Exception` around oracle.judge() converted ANY failure — including a
# genuine code regression inside the oracle — into an indistinguishable per-card FLAGGED verdict
# (I1: one bad claim FLAGs; must never abort the batch). These tests exercise the narrowed
# except split: expected failure types (KeyError/ValueError/ArithmeticError, traced to
# oracle.py's `canonical[series_ref]` lookup / math-domain errors / degenerate regressions) stay
# a quiet FLAG; anything else is marked UNEXPECTED and logged at ERROR so a systematic oracle bug
# reads as "go look at the oracle" instead of blending into ordinary sim noise.


def test_oracle_judge_expected_failure_flags_quietly(tmp_path, monkeypatch, caplog):
    """A KeyError (the same shape oracle.judge raises for a missing series_ref) is an EXPECTED
    failure type — FLAGGED, verdict_note carries the exception type, but it is NOT marked
    UNEXPECTED and nothing is logged at ERROR (an ordinary data/measurement gap, not a
    regression)."""
    import logging

    from openclaw_brain.knowledge.executable.oracle import ClaimOracle

    def _boom(self, claim, canonical):
        raise KeyError("cc_gbw_hz")

    monkeypatch.setattr(ClaimOracle, "judge", _boom)
    runner = _FakeRunner(_GBW, _AV0)

    with caplog.at_level(logging.ERROR, logger="openclaw_brain.knowledge.executable.executor"):
        result = run_recipe(_cc_recipe(), runner, corpus=SpecimenCorpus(str(tmp_path)))

    assert result.claim_cards  # the batch still completed (I1 — never aborts)
    for card in result.claim_cards:
        assert card.verdict == VerdictClass.FLAGGED
        assert "KeyError" in card.verdict_note
        assert "UNEXPECTED" not in card.verdict_note
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)


def test_oracle_judge_unexpected_failure_is_marked_and_logged(tmp_path, monkeypatch, caplog):
    """A non-expected exception type (e.g. a bare RuntimeError, not one of KeyError/ValueError/
    ArithmeticError) from oracle.judge() must still FLAG the card (I1: never abort the batch),
    but the verdict_note is marked UNEXPECTED and an ERROR is logged — so a corpus-wide wave of
    these is distinguishable from an ordinary sim/data FLAG."""
    import logging

    from openclaw_brain.knowledge.executable.oracle import ClaimOracle

    def _boom(self, claim, canonical):
        raise RuntimeError("boom: oracle internals broke")

    monkeypatch.setattr(ClaimOracle, "judge", _boom)
    runner = _FakeRunner(_GBW, _AV0)

    with caplog.at_level(logging.ERROR, logger="openclaw_brain.knowledge.executable.executor"):
        result = run_recipe(_cc_recipe(), runner, corpus=SpecimenCorpus(str(tmp_path)))

    assert result.runs == 2  # the batch still completed (I1 — never aborts)
    for card in result.claim_cards:
        assert card.verdict == VerdictClass.FLAGGED
        assert "UNEXPECTED" in card.verdict_note
        assert "RuntimeError" in card.verdict_note
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert error_records
    assert all("UNEXPECTED" in r.getMessage() for r in error_records)


def test_oracle_judge_success_unaffected_by_narrowed_except(tmp_path):
    """REGRESSION: the ordinary success path (no exception at all) is untouched by the except
    split — verdicts certify normally."""
    runner = _FakeRunner(_GBW, _AV0)
    result = run_recipe(_cc_recipe(), runner, corpus=SpecimenCorpus(str(tmp_path)))
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["cc_gbw_inverse"].verdict == VerdictClass.VERIFIED
    assert "UNEXPECTED" not in (by_id["cc_gbw_inverse"].verdict_note or "")


# ── integration: real sky130 ngspice ──


def test_run_recipe_end_to_end_sky130(tmp_path):
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; integration test needs the sim container")
    result = run_recipe(_cc_recipe(), runner, corpus=SpecimenCorpus(str(tmp_path)), project=True)

    by_id = {c.id: c for c in result.claim_cards}
    # GBW∝1/Cc: certified, and the measured series actually falls as Cc rises
    assert by_id["cc_gbw_inverse"].verdict in _CERTIFIED
    gbw_series = result.canonical["cc_gbw_hz"]
    assert len(gbw_series) == 5
    assert gbw_series[0][1] > gbw_series[-1][1]
    # DC gain ~invariant to Cc
    assert by_id["cc_av0_invariant"].verdict in _CERTIFIED

    # the loop closed: stored in the corpus + projected as graph nodes
    assert result.spec_id.startswith("sha256:")
    loaded = SpecimenCorpus(str(tmp_path)).load(OTA, result.spec_id)
    assert len(loaded.claim_cards) == 2
    assert [n["label"] for n in result.projection.nodes].count(NodeLabel.CLAIM_CARD) == 2
