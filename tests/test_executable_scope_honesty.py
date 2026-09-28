"""Tier-B Task 6: scope-honesty — machine-surfaced scope + per-engine basis + dominant-risk banner.

A VERIFIED verdict is nominal/single-corner (analog) or functional (digital) unless it says otherwise;
mismatch-dominated classes also name the untested dominant axis. These are stamped by the executor,
projected onto the ClaimCard node, and surfaced by why() (ADR Decisions 4-5).
"""
from __future__ import annotations

import json

from openclaw_brain.knowledge.executable.conditions import summarize_scope
from openclaw_brain.knowledge.executable.corpus import SpecimenCorpus
from openclaw_brain.knowledge.executable.executor import run_recipe
from openclaw_brain.knowledge.executable.models import (
    AnalogPVT, ClaimCard, MechanismClaim, QuantTest, Specimen, VerdictClass,
)
from openclaw_brain.knowledge.executable.projection import project_specimen
from openclaw_brain.knowledge.executable.seeds import _digital_cds_recipe
from openclaw_brain.knowledge.graph.schema import NodeLabel


class _FakeDigital:
    def measure(self, deck, timeout=120):
        return {"ped": [(0.0, 16.0), (64.0, 16.0), (128.0, 16.0), (192.0, 16.0)]}


def test_executor_stamps_functional_basis_and_risk_on_digital(tmp_path):
    res = run_recipe(_digital_cds_recipe(), _FakeDigital(), corpus=SpecimenCorpus(str(tmp_path)), project=True)
    card = res.claim_cards[0]
    assert card.engine == "iverilog"
    assert card.basis == "functional"
    assert card.scope.get("functional") is True
    assert card.dominant_risk_untested and "kTC" in card.dominant_risk_untested  # told, not just inferable


def test_digital_specimen_projects_without_crashing_on_pvt(tmp_path):
    # project=True exercises projection.py's ClaimCard node write: a digital card has NO corner/temp/vdd,
    # so the engine-safe getattr must project them as null (this is the reviewer's line-90 fix).
    res = run_recipe(_digital_cds_recipe(), _FakeDigital(), corpus=SpecimenCorpus(str(tmp_path)), project=True)
    props = next(n["properties"] for n in res.projection.nodes if n["label"] == NodeLabel.CLAIM_CARD)
    assert props["engine"] == "iverilog" and props["basis"] == "functional"
    assert props["corner"] is None and props["temp_c"] is None and props["vdd"] is None
    assert json.loads(props["scope"])["functional"] is True


def test_analog_card_projects_pvt_scope_and_physical_basis():
    cond = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)
    card = ClaimCard(
        id="a", topology_class="cds_switched_cap_nmos", conditions=cond,
        mechanism=MechanismClaim(knob="Vped", metric="vo_v", series_ref="a",
                                 quant=QuantTest(kind="invariance", cov_max=0.001)),
        verdict=VerdictClass.VERIFIED, engine="ngspice", basis="physical-nominal",
        scope=summarize_scope(cond),
        dominant_risk_untested="device/cap mismatch -> column FPN, kTC, charge-injection",
    )
    spec = Specimen(topology_class="cds_switched_cap_nmos", netlist="* cell",
                    claim_cards=[card], spec_id="sha256:test")
    props = next(n["properties"] for n in project_specimen(spec).nodes if n["label"] == NodeLabel.CLAIM_CARD)
    assert props["corner"] == "tt" and props["temp_c"] == 27.0 and props["vdd"] == 1.8
    assert props["engine"] == "ngspice" and props["basis"] == "physical-nominal"
    assert json.loads(props["scope"])["corners"] == ["sky130/tt/27/1.8"]   # node first-class (Stat-QT T1)
    assert "kTC" in props["dominant_risk_untested"]


def test_digital_specimen_forms_additive_realizes_and_grounds_links():
    # the digital projection reuses the SAME additive REALIZES (Specimen->CircuitTopology) + GROUNDS
    # (ClaimCard->Parameter) machinery as analog; the match_text targets a DIGITAL topology/parameter
    # (resolution to a node is live + NO-PHANTOM, so we verify the link REQUEST, not the bound node).
    from openclaw_brain.knowledge.graph.schema import RelType
    res = run_recipe(_digital_cds_recipe(), _FakeDigital(), corpus=None, project=True)
    realizes = [l for l in res.projection.links if l.rel_type == RelType.REALIZES]
    assert any(l.target_label == NodeLabel.CIRCUIT_TOPOLOGY and "digital cds" in l.match_text
               for l in realizes)
    grounds = [l for l in res.projection.links if l.rel_type == RelType.GROUNDS]
    assert any(l.target_label == NodeLabel.PARAMETER and l.match_text == "diff lsb" for l in grounds)


def test_claimcard_model_validate_backfills_kind_for_legacy_dict():
    # corpus load of a pre-Tier-B conditions dict (no `kind`) coerces via the union BeforeValidator —
    # the forward-compat trap the T1 reviewer flagged is already closed by Task 1.
    card = ClaimCard.model_validate({
        "id": "x", "topology_class": "cds_switched_cap_nmos",
        "mechanism": {"knob": "Vped", "metric": "vo_v", "series_ref": "x",
                      "quant": {"kind": "invariance", "cov_max": 0.001}},
        "conditions": {"corner": "tt", "temp_c": 27.0, "vdd": 1.8},
    })
    assert card.conditions.kind == "analog_pvt" and card.conditions.corner == "tt"


def test_statistical_verdict_clears_mismatch_banner(tmp_path):
    # a certified statistical verdict means the mismatch axis IS now tested -> the dominant-risk banner
    # (which ota_5t_nmos_in carries) is cleared for that card (Stat-QT pays down the scope-debt).
    from openclaw_brain.knowledge.executable.seeds import _ota5t_offset_recipe

    class _OR:
        def measure(self, deck, timeout=300):
            return {"vos": [(float(i), 0.004 * (1 if i % 2 else -1)) for i in range(30)]}

    res = run_recipe(_ota5t_offset_recipe(), _OR(), corpus=SpecimenCorpus(str(tmp_path)), project=False)
    card = {c.id: c for c in res.claim_cards}["ota5t_vos"]
    assert card.verdict == VerdictClass.VERIFIED
    assert card.dominant_risk_untested is None
