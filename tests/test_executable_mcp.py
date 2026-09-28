"""(e) MCP surface — the BrainAgent methods the executable-substrate MCP tools delegate to.
query_executable (read) / project_executable (write) / retract_executable (write). Mock graph +
journal, no Neo4j / no sim."""

from __future__ import annotations

import json

import pytest

from openclaw_brain.agent import BrainAgent
from openclaw_brain.config import load_config


class _MockGraph:
    def __init__(self, rows=None):
        self._rows = rows or []
        self.queries = []

    async def run_read_query(self, query, params=None):
        self.queries.append((query, params))
        return self._rows


class _MockJournal:
    def __init__(self):
        self.logs = []

    def log(self, action, **kw):
        self.logs.append((action, kw))


@pytest.fixture
def agent(tmp_path):
    cfg = load_config()
    cfg.openclaw.state_dir = str(tmp_path)      # state_path is a read-only property derived from this
    a = BrainAgent(cfg)
    a._started = True
    a._journal = _MockJournal()
    a._graph = _MockGraph()
    return a


_ROW = {
    "spec_id": "sha256:abc", "topology_class": "miller_ota_2stage_nmos_in",
    "pdk": "sky130A", "tool": "ngspice-46", "realizes": "Two-Stage Miller OTA",
    "claims": [
        {"claim_id": "sha256:abc:cc_gbw", "claim": "cc_gbw", "knob": "Cc", "metric": "gbw_hz",
         "verdict": "VERIFIED", "narrative": "GBW=gm1/2piCc",
         "conditions": {"corner": "tt", "temp_c": 27.0, "vdd": 1.8},
         "grounds": "Gain-Bandwidth Product"},
        None,    # specimen-with-some-but-not-all-claims artifact — must be filtered
    ],
}


@pytest.mark.asyncio
async def test_query_executable_shapes_and_filters(agent):
    agent._graph = _MockGraph(rows=[_ROW])
    out = await agent.query_executable()
    assert out["count"] == 1
    s = out["specimens"][0]
    assert s["spec_id"] == "sha256:abc"
    assert s["realizes"] == "Two-Stage Miller OTA"
    assert len(s["claims"]) == 1                          # the None entry is filtered
    assert s["claims"][0]["verdict"] == "VERIFIED"
    assert s["claims"][0]["grounds"] == "Gain-Bandwidth Product"


@pytest.mark.asyncio
async def test_query_executable_passes_topology_filter(agent):
    agent._graph = _MockGraph(rows=[])
    out = await agent.query_executable("current_mirror_simple_nmos")
    assert out == {"topology_class": "current_mirror_simple_nmos", "count": 0, "specimens": []}
    # the class flowed into the query params
    assert agent._graph.queries[0][1] == {"tclass": "current_mirror_simple_nmos"}


@pytest.mark.asyncio
async def test_project_executable_skips_per_recipe_when_engines_unavailable(agent, monkeypatch):
    # New contract (all-registry, engine-aware dispatch): an unavailable engine skips ITS recipes
    # with a per-recipe report entry — the batch never aborts with a whole-batch error.
    from openclaw_brain.knowledge.executable import runner as runner_mod
    from openclaw_brain.knowledge.executable import verilog_runner as vrunner_mod
    monkeypatch.setattr(runner_mod.NgspiceRunner, "available", lambda self: False)
    monkeypatch.setattr(vrunner_mod.VerilogRunner, "available", lambda self: False)
    from openclaw_brain.knowledge.executable.seeds import (
        digital_seed_recipes, seed_recipes, statistical_seed_recipes,
    )
    out = await agent.project_executable(apply=False)
    assert "error" not in out
    assert out["specimens"] == []
    # every seed recipe (analog + digital + statistical) reported as skipped — derived, not
    # hardcoded, so adding a seed recipe (e.g. S3-inc2a's ptat_ctat_core_bjt) never rots this test.
    all_recipes_count = (len(seed_recipes()) + len(digital_seed_recipes())
                         + len(statistical_seed_recipes()))
    assert len(out["skipped"]) == all_recipes_count
    engines_seen = {s["engine"] for s in out["skipped"]}
    assert engines_seen == {"ngspice", "iverilog"}
    assert all(s["reason"] and s["topology_class"] for s in out["skipped"])


@pytest.mark.asyncio
async def test_retract_executable_delegates_and_journals(agent, monkeypatch):
    async def _fake_retract(store, spec_id):
        return {"specimens": 1, "claim_cards": 3}
    monkeypatch.setattr(
        "openclaw_brain.knowledge.executable.projection.retract_projection", _fake_retract)
    out = await agent.retract_executable("sha256:abc")
    assert out == {"spec_id": "sha256:abc", "specimens": 1, "claim_cards": 3}
    assert agent._journal.logs[0][0] == "retract_executable"


@pytest.mark.asyncio
async def test_why_shapes_grounding(agent):
    agent._graph = _MockGraph(rows=[{
        "claim": "cc_gbw", "knob": "Cc", "metric": "gbw_hz", "verdict": "VERIFIED",
        "quant_kind": "direction",
        "narrative": "GBW = gm1/(2*pi*Cc)", "corner": "tt", "temp_c": 27.0, "vdd": 1.8,
        "engine": "ngspice", "basis": "physical-nominal",
        "scope": '{"device": "nominal", "corners": ["tt/27/1.8"], "statistical": "none"}',
        "dominant_risk_untested": None,
        "spec_id": "sha256:abc", "topology_class": "miller_ota_2stage_nmos_in",
        "grounds": "Gain-Bandwidth Product"}])
    out = await agent.why("sha256:abc:cc_gbw")
    assert out["found"] is True
    assert out["verdict"] == "VERIFIED"
    assert out["quant_kind"] == "direction"
    assert out["conditions"] == {"corner": "tt", "temp_c": 27.0, "vdd": 1.8}
    assert out["grounds"] == "Gain-Bandwidth Product"
    assert out["topology_class"] == "miller_ota_2stage_nmos_in"
    # scope-honesty: the verdict renders INLINE with its scope (undetachable), and the basis is surfaced
    assert out["basis"] == "physical-nominal"
    assert out["verdict_scoped"] == "VERIFIED@tt/27/1.8"


@pytest.mark.asyncio
async def test_why_not_found(agent):
    agent._graph = _MockGraph(rows=[])
    out = await agent.why("sha256:none:x")
    assert out == {"claim_card_id": "sha256:none:x", "found": False}


@pytest.mark.asyncio
async def test_audit_citations_passes_resolved_certified(agent):
    # the graph resolves the one cited card as VERIFIED (a direction-kind card)
    agent._graph = _MockGraph(rows=[{"verdict": "VERIFIED", "quant_kind": "direction"}])
    plan = {"topology_class": "miller_ota_2stage_nmos_in", "claims": [
        {"text": "GBW falls as Cc rises", "tier": "certified", "cites": "sha256:abc:cc_gbw"},
        {"text": "because of Miller pole-splitting", "tier": "interpretive"}]}
    out = await agent.audit_citations(plan)
    assert out["passed"] is True
    assert out["certified_ok"] == 1 and out["interpretive_total"] == 1


@pytest.mark.asyncio
async def test_audit_citations_flags_magnitude_under_shape_card(agent):
    # a certified claim that asserts an ABSOLUTE magnitude (~67 dB) while citing an invariance-kind
    # card over-claims: the oracle proved only CL-invariance, never the scalar. Audit must flag it.
    agent._graph = _MockGraph(rows=[{"verdict": "VERIFIED", "quant_kind": "invariance"}])
    plan = {"topology_class": "telescopic_cascode_ota_nmos_in", "claims": [
        {"text": "open-loop gain is about 67 dB and is independent of load capacitance",
         "tier": "certified", "cites": "sha256:abc:tele_av0"}]}
    out = await agent.audit_citations(plan)
    assert out["passed"] is False
    assert out["certified_total"] == 1 and out["certified_ok"] == 0
    assert "scalar magnitude" in out["findings"][0]["reason"]


@pytest.mark.asyncio
async def test_audit_citations_flags_missing_card(agent):
    # the graph resolves nothing -> the cited card is "not found"
    agent._graph = _MockGraph(rows=[])
    plan = {"topology_class": "t", "claims": [
        {"text": "stated as fact", "tier": "certified", "cites": "sha256:abc:nope"}]}
    out = await agent.audit_citations(plan)
    assert out["passed"] is False
    assert "not found" in out["findings"][0]["reason"]


# ── law-tier I2 (docs/superpowers/specs/2026-07-04-law-tier-graph-representation.md §5) ──
# query_executable's additive law_ids, why()'s additive laws key, and the new why_law() method —
# all against a mocked graph (no Neo4j / no sim, mirroring every test above).


@pytest.mark.asyncio
async def test_query_executable_passes_through_law_ids_additively(agent):
    # law_ids is produced entirely by the Cypher pattern comprehension in production; at the
    # BrainAgent level this is a pure passthrough — a claim dict carrying "law_ids" flows through
    # query_executable's filtering unmodified, exactly like every other claim field.
    row = {**_ROW, "claims": [
        {**_ROW["claims"][0], "law_ids": ["04dcee1d0410e97468da53d6bc8dde0ff6d039a5"]},
        None,
    ]}
    agent._graph = _MockGraph(rows=[row])
    out = await agent.query_executable()
    assert out["specimens"][0]["claims"][0]["law_ids"] == ["04dcee1d0410e97468da53d6bc8dde0ff6d039a5"]


@pytest.mark.asyncio
async def test_why_includes_laws_key_additively(agent):
    # the "laws" key is additive: existing fields (test_why_shapes_grounding, above) are unchanged,
    # and a row that DOES carry the collected laws list (as the real Cypher's `collect(...)` would,
    # nulls included for the CASE-WHEN-NULL branch) surfaces it, nulls filtered.
    agent._graph = _MockGraph(rows=[{
        "claim": "cc_gbw", "knob": "Cc", "metric": "gbw_hz", "verdict": "VERIFIED",
        "quant_kind": "direction",
        "narrative": "GBW = gm1/(2*pi*Cc)", "corner": "tt", "temp_c": 27.0, "vdd": 1.8,
        "engine": "ngspice", "basis": "physical-nominal",
        "scope": '{"device": "nominal", "corners": ["tt/27/1.8"], "statistical": "none"}',
        "dominant_risk_untested": None,
        "spec_id": "sha256:abc", "topology_class": "miller_ota_2stage_nmos_in",
        "grounds": "Gain-Bandwidth Product",
        "laws": [{"law_id": "04dcee1d0410e97468da53d6bc8dde0ff6d039a5", "status": "law"}, None],
    }])
    out = await agent.why("sha256:abc:cc_gbw")
    assert out["laws"] == [{"law_id": "04dcee1d0410e97468da53d6bc8dde0ff6d039a5", "status": "law"}]


@pytest.mark.asyncio
async def test_why_laws_key_empty_when_row_omits_it_backcompat(agent):
    # a row that never reports "laws" at all (the old query shape) still yields an empty list, not
    # a KeyError -- additive means "absent -> []", never a break.
    agent._graph = _MockGraph(rows=[{
        "claim": "cc_gbw", "knob": "Cc", "metric": "gbw_hz", "verdict": "VERIFIED",
        "quant_kind": "direction", "narrative": "x", "corner": "tt", "temp_c": 27.0, "vdd": 1.8,
        "engine": "ngspice", "basis": "physical-nominal", "scope": None,
        "dominant_risk_untested": None, "spec_id": "sha256:abc",
        "topology_class": "miller_ota_2stage_nmos_in", "grounds": None,
    }])
    out = await agent.why("sha256:abc:cc_gbw")
    assert out["laws"] == []


@pytest.mark.asyncio
async def test_why_law_not_found(agent):
    agent._graph = _MockGraph(rows=[])
    out = await agent.why_law("deadbeef")
    assert out == {"law_id": "deadbeef", "found": False}


_PELGROM_LAW_ROW = {
    "status": "law", "status_note": "",
    "statement": ("For ota_5t_nmos_in, a_vos follows a power-law (Pelgrom-type) scaling in vos "
                  "— replicated across gf180mcuD, ihp-sg13g2, sky130A."),
    "topology_class": "ota_5t_nmos_in", "metric": "a_vos", "knob": "vos", "quant_kind": "elasticity",
    "pdks": ["gf180mcuD", "ihp-sg13g2", "sky130A"],
    "member_summary": json.dumps({
        "gf180mcuD": {"verdict": "VERIFIED", "note": "global slope -0.484 in band",
                      "fitted_exponent": -0.4837253530318265},
        "ihp-sg13g2": {"verdict": "VERIFIED", "note": "global slope -0.466 in band",
                       "fitted_exponent": -0.4656522453987968},
        "sky130A": {"verdict": "VERIFIED", "note": "global slope -0.462 in band",
                    "fitted_exponent": -0.4616473823724754},
    }),
    "derived_from": json.dumps({"jsonl_paths": ["experiments/e1b_statistical_cross_pdk_raw.jsonl"]}),
    "cards": [
        {"pdk": "sky130A", "claim_id": "sha256:xyz:ota5t_pelgrom", "verdict": "VERIFIED",
         "scope": json.dumps({"pdk": "sky130", "corners": ["sky130/tt_mm/27/1.8"],
                              "statistical": "3σ@200"})},
        None,   # the CASE-WHEN-c-IS-NULL branch when a law has zero/partial SUPPORTED_BY edges
    ],
}


@pytest.mark.asyncio
async def test_why_law_found_mixes_card_backed_and_report_only_members(agent):
    # mirrors the live 8-law state exactly: SUPPORTED_BY exists only to the sky130A card; gf180mcuD
    # / ihp-sg13g2 are member_summary-only (report-only) until rollout (3) projects their cards.
    agent._graph = _MockGraph(rows=[_PELGROM_LAW_ROW])
    out = await agent.why_law("8c4834b2fc70134f355efa5f850a4024e64c5821")
    assert out["found"] is True
    assert out["status"] == "law"
    assert "status_note" not in out          # status == "law" -> not rendered loudly
    assert "history" not in out
    assert out["supported_by_count"] == 1
    assert out["derived_from"] == {"jsonl_paths": ["experiments/e1b_statistical_cross_pdk_raw.jsonl"]}
    # the law never renders as a bare universal -- the member list is always attached
    assert "gf180mcuD" in out["statement"] and "sky130A" in out["statement"]

    members = {m["pdk"]: m for m in out["members"]}
    assert set(members) == {"gf180mcuD", "ihp-sg13g2", "sky130A"}

    # card-backed member: scope-honest, undetachable verdict tag reusing _scope_inline exactly
    sky = members["sky130A"]
    assert sky["card_projected"] is True
    assert sky["card_id"] == "sha256:xyz:ota5t_pelgrom"
    assert sky["verdict_scoped"] == "VERIFIED@sky130/tt_mm/27/1.8/3σ@200"
    # Pelgrom: fitted_exponent is MEMBER data, never folded into the (magnitude-free) statement
    assert sky["fitted_exponent"] == -0.4616473823724754
    assert "0.46" not in out["statement"] and "0.48" not in out["statement"]

    # report-only members: honest, never a bare universal, never silently identical to a real scope
    gf = members["gf180mcuD"]
    assert gf["card_projected"] is False
    assert "card_id" not in gf
    assert gf["verdict_scoped"] == "VERIFIED@gf180mcuD (report-only: member card not yet projected)"
    assert gf["fitted_exponent"] == -0.4837253530318265
    ihp = members["ihp-sg13g2"]
    assert ihp["card_projected"] is False
    assert ihp["verdict_scoped"] == "VERIFIED@ihp-sg13g2 (report-only: member card not yet projected)"


@pytest.mark.asyncio
async def test_why_law_demoted_rendering_includes_status_note_and_history(agent):
    row = {
        "status": "demoted",
        "status_note": "process-dependent: gf180mcuD, sky130A agree, but ihp-sg13g2=REFUTED",
        "statement": "For current_mirror_simple_nmos, iout_a increases with Vout "
                     "— process-dependent, not (yet) a cross-foundry law.",
        "topology_class": "current_mirror_simple_nmos", "metric": "iout_a", "knob": "Vout",
        "quant_kind": "direction", "pdks": ["gf180mcuD", "ihp-sg13g2", "sky130A"],
        "member_summary": json.dumps({
            "gf180mcuD": {"verdict": "VERIFIED", "note": "trend +"},
            "ihp-sg13g2": {"verdict": "REFUTED", "note": "trend flat"},
            "sky130A": {"verdict": "VERIFIED", "note": "trend +"},
            "_history": [{"prior_status": "law", "demoted_reason": "ihp-sg13g2 diverged",
                          "pdks_at_demotion": ["gf180mcuD", "ihp-sg13g2", "sky130A"]}],
        }),
        "derived_from": json.dumps({"jsonl_paths": ["experiments/e1_cross_pdk_raw.jsonl"]}),
        "cards": [],
    }
    agent._graph = _MockGraph(rows=[row])
    out = await agent.why_law("04dcee1d0410e97468da53d6bc8dde0ff6d039a5")
    assert out["status"] == "demoted"
    # status != "law" renders loudly: status_note is present and names the divergence
    assert out["status_note"] == row["status_note"]
    assert "ihp-sg13g2" in out["status_note"]
    # a node that was EVER demoted carries its prior-status history forward (R5, never silent)
    assert out["history"] == [{"prior_status": "law", "demoted_reason": "ihp-sg13g2 diverged",
                               "pdks_at_demotion": ["gf180mcuD", "ihp-sg13g2", "sky130A"]}]
    # _history is never leaked as if it were a real member's data
    members = {m["pdk"]: m for m in out["members"]}
    assert set(members) == {"gf180mcuD", "ihp-sg13g2", "sky130A"}
    assert all(m["card_projected"] is False for m in members.values())
    assert members["ihp-sg13g2"]["verdict"] == "REFUTED"


# ── law-tier I2 audit wiring: agent.audit_citations resolves BOTH ClaimCard AND Regularity id
#    spaces in the SAME query (a citation is either a claim_card_id or a law_id, never both) ──


@pytest.mark.asyncio
async def test_audit_citations_resolves_law_citation_within_member_set_passes(agent):
    agent._graph = _MockGraph(rows=[{
        "verdict": None, "quant_kind": None, "basis": None, "metric": None, "scope": None,
        "law_status": "law", "law_quant_kind": "direction", "law_metric": "iout_a",
        "law_pdks": ["gf180mcuD", "ihp-sg13g2", "sky130A"],
    }])
    plan = {"topology_class": "current_mirror_simple_nmos", "claims": [
        {"text": "output current increases with Vout across gf180mcuD, ihp-sg13g2 and sky130A",
         "tier": "certified", "cites": "04dcee1d0410e97468da53d6bc8dde0ff6d039a5"}]}
    out = await agent.audit_citations(plan)
    assert out["passed"] is True and out["certified_ok"] == 1


@pytest.mark.asyncio
async def test_audit_citations_law_citation_beyond_member_set_fails(agent):
    agent._graph = _MockGraph(rows=[{
        "verdict": None, "quant_kind": None, "basis": None, "metric": None, "scope": None,
        "law_status": "law", "law_quant_kind": "direction", "law_metric": "iout_a",
        "law_pdks": ["gf180mcuD", "ihp-sg13g2", "sky130A"],
    }])
    plan = {"topology_class": "current_mirror_simple_nmos", "claims": [
        {"text": "output current increases with Vout on any 28nm process",
         "tier": "certified", "cites": "04dcee1d0410e97468da53d6bc8dde0ff6d039a5"}]}
    out = await agent.audit_citations(plan)
    assert out["passed"] is False
    assert "over-generalizes" in out["findings"][0]["reason"]


@pytest.mark.asyncio
async def test_audit_citations_law_citation_non_law_status_fails(agent):
    agent._graph = _MockGraph(rows=[{
        "verdict": None, "quant_kind": None, "basis": None, "metric": None, "scope": None,
        "law_status": "demoted", "law_quant_kind": "direction", "law_metric": "iout_a",
        "law_pdks": ["gf180mcuD", "ihp-sg13g2", "sky130A"],
    }])
    plan = {"topology_class": "current_mirror_simple_nmos", "claims": [
        {"text": "output current increases with Vout across gf180mcuD, ihp-sg13g2 and sky130A",
         "tier": "certified", "cites": "04dcee1d0410e97468da53d6bc8dde0ff6d039a5"}]}
    out = await agent.audit_citations(plan)
    assert out["passed"] is False
    assert "demoted" in out["findings"][0]["reason"]
    assert "re-verify" in out["findings"][0]["reason"]


@pytest.mark.asyncio
async def test_audit_citations_law_citation_causal_without_intervention_fails(agent):
    agent._graph = _MockGraph(rows=[{
        "verdict": None, "quant_kind": None, "basis": None, "metric": None, "scope": None,
        "law_status": "law", "law_quant_kind": "invariance", "law_metric": "av0_db",
        "law_pdks": ["gf180mcuD", "ihp-sg13g2", "sky130A"],
    }])
    plan = {"topology_class": "common_source_active_load_nmos", "claims": [
        {"text": "the invariance to Iref is caused by cascode output impedance",
         "tier": "certified", "cites": "984274c86a29e61d8543c43e52ccdb51e36989e1"}]}
    out = await agent.audit_citations(plan)
    assert out["passed"] is False
    assert "intervention" in out["findings"][0]["reason"]
    assert "mechanism_never_fact" in out["findings"][0]["reason"]
