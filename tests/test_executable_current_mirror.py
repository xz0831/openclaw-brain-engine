"""(b) Second topology class: current_mirror_simple_nmos — proves the frozen core generalizes.

A DIFFERENT analysis (.dc/op, not .ac) and a DIFFERENT metric (iout, not gbw/pm) flow through the
SAME registry → recipe-capability → executor → oracle → corpus → projection path, with zero changes
to that core. Unit (fake runner) + docker-gated integration on the real sky130 mirror.
"""

from __future__ import annotations

import pytest

from openclaw_brain.knowledge.executable.corpus import SpecimenCorpus
from openclaw_brain.knowledge.executable.executor import run_recipe
from openclaw_brain.knowledge.executable.models import (
    ClaimCard, MechanismClaim, QuantTest, VerdictClass, VerificationRecipe,
)
from openclaw_brain.knowledge.executable.recipe import capability_for
from openclaw_brain.knowledge.executable.runner import NgspiceRunner
from openclaw_brain.knowledge.executable.templates import (
    CM_KNOB_SOURCES, CM_METRIC_LET, _CM_SIMPLE_BODY,
    render_current_mirror_cell, render_current_mirror_dc,
)
from openclaw_brain.knowledge.graph.schema import NodeLabel

CM = "current_mirror_simple_nmos"
COND = {"corner": "tt", "temp_c": 27.0, "vdd": 1.8}
_CERTIFIED = {VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT}


# ── template + capability ──


def test_capability_registered_for_current_mirror():
    cap = capability_for(CM)
    assert cap.template_ref == "current_mirror_dc"
    assert cap.analyses == frozenset({"dc"})
    assert cap.knobs == frozenset({"Vout"})
    assert cap.metrics == frozenset({"iout_a"})


def test_render_emits_dc_op_sweep_rdata():
    deck = render_current_mirror_dc(points=["0.4", "1.0", "1.6"])
    assert "foreach pt 0.4 1.0 1.6" in deck
    assert "alter Vout = $pt" in deck
    assert "op" in deck
    assert "echo RDATA vout $pt $&y" in deck
    assert "sky130_fd_pr__nfet_01v8" in deck          # analog primitive, not std-cell


def test_template_knob_and_metric_are_executable():
    """Drift guard: the knob's swept source must be defined in the body; the metric's let is known."""
    cap = capability_for(CM)
    body_elements = {line.split()[0] for line in _CM_SIMPLE_BODY.splitlines() if line.strip()}
    for knob in cap.knobs:
        assert CM_KNOB_SOURCES[knob] in body_elements
    for metric in cap.metrics:
        assert metric in CM_METRIC_LET


# ── executor: unit (fake runner) ──


class _FakeMirrorRunner:
    def __init__(self, iout_series):
        self._iout = iout_series
        self.decks = []

    def measure(self, deck, timeout=300):
        self.decks.append(deck)
        if CM_METRIC_LET["iout_a"] in deck:
            return {"vout": self._iout}
        return {}


# finite output resistance: iout rises slightly with Vout (channel-length modulation)
_IOUT = [(0.4, 9.4e-6), (0.7, 9.7e-6), (1.0, 10.0e-6), (1.3, 10.3e-6), (1.6, 10.6e-6)]


def _cm_recipe():
    def claim(cid, quant, narrative):
        return ClaimCard(id=cid, topology_class=CM, conditions=COND,
                         mechanism=MechanismClaim(knob="Vout", metric="iout_a",
                                                  series_ref="out_char", quant=quant, narrative=narrative))
    return VerificationRecipe(
        topology_class=CM,
        build={"method": "template", "template_ref": "current_mirror_dc"},
        conditions=COND,
        sweeps=[{"analysis": "dc", "knob": "Vout",
                 "points": ["0.4", "0.7", "1.0", "1.3", "1.6"], "measure": ["iout_a"]}],
        claim_cards=[
            claim("cm_iout_clm", QuantTest(kind="direction", sign="+"),
                  "iout rises with Vout — finite output resistance (channel-length modulation)"),
            claim("cm_iout_hold", QuantTest(kind="invariance", cov_max=0.30),
                  "the mirror holds iout approximately constant across the saturation region"),
        ],
    )


def test_run_recipe_current_mirror_unit(tmp_path):
    runner = _FakeMirrorRunner(_IOUT)
    result = run_recipe(_cm_recipe(), runner, corpus=SpecimenCorpus(str(tmp_path)), project=True)

    # two claims on the SAME (Vout, iout_a) -> one run, both routed to the canonical key
    assert result.runs == 1
    assert set(result.canonical) == {"vout_iout_a"}
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["cm_iout_clm"].verdict == VerdictClass.VERIFIED      # monotonic rising
    assert by_id["cm_iout_hold"].verdict == VerdictClass.VERIFIED     # CoV 12% < 30%

    # the frozen core handled a new class: stored + projected
    assert result.spec_id.startswith("sha256:")
    labels = [n["label"] for n in result.projection.nodes]
    assert labels.count(NodeLabel.SPECIMEN) == 1
    assert labels.count(NodeLabel.CLAIM_CARD) == 2


def test_cell_renderer_is_bare(tmp_path):
    cell = render_current_mirror_cell()
    assert ".control" not in cell and "foreach" not in cell      # no testbench in the identity
    assert "XM1 nd nd 0 0" in cell


# ── executor: docker-gated integration on real sky130 ──


def test_run_recipe_current_mirror_end_to_end_sky130(tmp_path):
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; integration test needs the sim container")
    result = run_recipe(_cm_recipe(), runner, corpus=SpecimenCorpus(str(tmp_path)), project=True)

    iout = result.canonical["vout_iout_a"]
    assert len(iout) == 5
    assert all(8e-6 < y < 12e-6 for _, y in iout)        # mirrors the ~10µA reference
    assert iout[-1][1] >= iout[0][1]                      # finite Rout: iout rises with Vout
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["cm_iout_clm"].verdict in _CERTIFIED
    assert by_id["cm_iout_hold"].verdict in _CERTIFIED
    assert result.spec_id.startswith("sha256:")
