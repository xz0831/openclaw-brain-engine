"""Tests for the MCP server.

Direct ``Tool.run(args, None)`` / ``get_resource(uri, None)`` calls pass an explicit
None context: mcp 2.x made the context parameter required, and none of these tools or
resources declares a Context parameter, so the value is never dereferenced (mcp 1.x
defaults it to None, so the same call shape runs on both majors).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openclaw_brain.server.mcp_server import create_server, _assert_running


@pytest.fixture(autouse=True)
def _no_ambient_readonly(monkeypatch):
    """A shell-exported OPENCLAW_BRAIN_READONLY=1 / OPENCLAW_BRAIN_PROFILE must not
    flip default-mode tests."""
    monkeypatch.delenv("OPENCLAW_BRAIN_READONLY", raising=False)
    monkeypatch.delenv("OPENCLAW_BRAIN_PROFILE", raising=False)


@pytest.fixture
def mcp():
    """Create a fresh MCP server instance."""
    return create_server()


def test_create_server_returns_mcpserver(mcp):
    # mcp 2.x renamed FastMCP -> MCPServer; the server module resolves the name once.
    from openclaw_brain.server.mcp_server import MCPServer
    assert isinstance(mcp, MCPServer)


def test_server_has_expected_tools(mcp):
    """Verify all expected tools are registered."""
    tool_names = {t.name for t in mcp._tool_manager.list_tools()}  # internal Tool objects
    expected = {
        "startup", "shutdown",
        "ingest_pdf", "query_knowledge", "get_evidence", "recall_memory",
        "reinforce_concept", "find_bridges",
        "start_session", "end_session", "record_event",
        "upsert_entity", "record_lesson",
        "run_maintenance", "get_stats", "route_and_execute",
        "get_pipeline_config",
        "update_stage_model", "update_fallback_chain",
        "add_catalog_model", "remove_catalog_model",
    }
    assert expected.issubset(tool_names), f"Missing tools: {expected - tool_names}"


def test_assert_running_raises_when_not_started():
    """Should raise RuntimeError when agent is not running."""
    import openclaw_brain.server.mcp_server as mod
    original = mod._agent
    mod._agent = None
    try:
        with pytest.raises(RuntimeError, match="not running"):
            _assert_running()
    finally:
        mod._agent = original


def test_assert_running_raises_when_not_is_started():
    """Should raise RuntimeError when agent exists but is_started is False."""
    import openclaw_brain.server.mcp_server as mod
    original = mod._agent
    mock_agent = MagicMock()
    mock_agent.is_started = False
    mod._agent = mock_agent
    try:
        with pytest.raises(RuntimeError, match="not running"):
            _assert_running()
    finally:
        mod._agent = original


@pytest.mark.asyncio
async def test_startup_tool(mcp):
    """Test startup tool creates and starts the agent."""
    import openclaw_brain.server.mcp_server as mod

    mock_agent = AsyncMock()
    mock_agent.is_started = True

    with patch.object(mod, "load_config") as mock_config, \
         patch.object(mod, "inject_api_keys", return_value={}), \
         patch.object(mod, "BrainAgent", return_value=mock_agent) as mock_cls:
        from openclaw_brain.config import BrainConfig
        mock_config.return_value = BrainConfig()

        # Get the startup tool function
        tool = mcp._tool_manager.get_tool("startup")
        result = await tool.run({}, None)

        assert "started successfully" in result
        mock_cls.assert_called_once()
        mock_agent.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_startup_already_running(mcp):
    """Test startup when agent is already running."""
    import openclaw_brain.server.mcp_server as mod
    original = mod._agent

    mock_agent = MagicMock()
    mock_agent.is_started = True
    mod._agent = mock_agent

    try:
        tool = mcp._tool_manager.get_tool("startup")
        result = await tool.run({}, None)
        assert "Already running" in result
    finally:
        mod._agent = original


@pytest.mark.asyncio
async def test_shutdown_tool(mcp):
    """Test shutdown tool stops and clears the agent."""
    import openclaw_brain.server.mcp_server as mod
    original = mod._agent

    mock_agent = AsyncMock()
    mod._agent = mock_agent

    try:
        tool = mcp._tool_manager.get_tool("shutdown")
        result = await tool.run({}, None)
        assert "shut down" in result
        mock_agent.stop.assert_awaited_once()
        assert mod._agent is None
    finally:
        mod._agent = original


@pytest.mark.asyncio
async def test_query_knowledge_tool(mcp):
    """Test query_knowledge delegates to agent."""
    import openclaw_brain.server.mcp_server as mod
    original = mod._agent

    mock_agent = AsyncMock()
    mock_agent.is_started = True
    mock_agent.query_knowledge.return_value = {
        "formatted": "MOSFET: a field-effect transistor",
        "concepts_found": 1,
        "memories_found": 0,
        "concept_refs": [{"id": "mosfet", "name": "MOSFET", "confidence": 0.9,
                          "layer": "L1", "domain": "device_physics",
                          "cite": {"level": "chunk", "src": "src_1",
                                   "chunks": ["chunk_1"]}}],
        "open_hypotheses": [],
        "active_decisions": [],
    }
    mod._agent = mock_agent

    try:
        tool = mcp._tool_manager.get_tool("query_knowledge")
        result = await tool.run({"query": "MOSFET"}, None)
        assert "MOSFET" in result
        # W1.5 contract: JSON envelope with IDs for the read→write loop
        import json as _json
        envelope = _json.loads(result)
        assert set(envelope) == {"context", "concepts", "open_hypotheses", "active_decisions"}
        assert envelope["concepts"][0]["id"] == "mosfet"
        assert set(envelope["concepts"][0]) == {
            "id", "name", "confidence", "layer", "domain", "cite",
        }
        assert envelope["concepts"][0]["cite"]["level"] == "chunk"
        assert "open_hypotheses" in envelope and "active_decisions" in envelope
        mock_agent.query_knowledge.assert_awaited_once_with("MOSFET")
    finally:
        mod._agent = original


@pytest.mark.asyncio
async def test_get_evidence_tool(mcp):
    """Test get_evidence delegates to agent."""
    import openclaw_brain.server.mcp_server as mod
    original = mod._agent

    mock_agent = AsyncMock()
    mock_agent.is_started = True
    mock_agent.get_evidence.return_value = {
        "chunk_id": "chunk_1",
        "source_id": "src_1",
        "text": "verbatim text",
        "verbatim": True,
        "section_title": "Intro",
        "pages": "1",
    }
    mod._agent = mock_agent

    try:
        tool = mcp._tool_manager.get_tool("get_evidence")
        result = await tool.run({"chunk_id": "chunk_1"}, None)
        parsed = json.loads(result)
        assert parsed["text"] == "verbatim text"
        assert parsed["verbatim"] is True
        mock_agent.get_evidence.assert_awaited_once_with("chunk_1")
    finally:
        mod._agent = original


@pytest.mark.asyncio
async def test_recall_memory_tool(mcp):
    """Test recall_memory delegates to agent."""
    import openclaw_brain.server.mcp_server as mod
    original = mod._agent

    mock_agent = AsyncMock()
    mock_agent.is_started = True
    mock_agent.recall.return_value = {
        "formatted": "gm = 2*ID/Vov",
        "count": 1,
    }
    mod._agent = mock_agent

    try:
        tool = mcp._tool_manager.get_tool("recall_memory")
        result = await tool.run({"query": "gm equation"}, None)
        assert "gm" in result
    finally:
        mod._agent = original


@pytest.mark.asyncio
async def test_ingest_pdf_tool(mcp):
    """Test ingest_pdf delegates to agent."""
    import openclaw_brain.server.mcp_server as mod
    original = mod._agent

    mock_agent = AsyncMock()
    mock_agent.is_started = True
    mock_result = MagicMock()
    mock_result.output = {"concepts_added": 5, "edges_added": 3}
    mock_agent.ingest_pdf.return_value = mock_result
    mod._agent = mock_agent

    try:
        tool = mcp._tool_manager.get_tool("ingest_pdf")
        result = await tool.run({"file_path": "/tmp/test.pdf"}, None)
        parsed = json.loads(result)
        assert parsed["concepts_added"] == 5
    finally:
        mod._agent = original


@pytest.mark.asyncio
async def test_ingest_pdf_tool_forwards_session_id(mcp):
    """W-D2 defect 6: ingest_pdf previously had no session_id parameter at all, so
    SkillExecutor's session_id-gated procedural-memory write (executor.py:85) was structurally
    unreachable from this production entry point even though agent.py/SkillExecutor already
    supported it end-to-end."""
    import openclaw_brain.server.mcp_server as mod
    original = mod._agent

    mock_agent = AsyncMock()
    mock_agent.is_started = True
    mock_result = MagicMock()
    mock_result.output = {"concepts_added": 1}
    mock_agent.ingest_pdf.return_value = mock_result
    mod._agent = mock_agent

    try:
        tool = mcp._tool_manager.get_tool("ingest_pdf")
        await tool.run({"file_path": "/tmp/test.pdf", "session_id": "ses_abc123"}, None)
        mock_agent.ingest_pdf.assert_awaited_once_with(
            file_path="/tmp/test.pdf", extraction_model=None, reasoning_model=None,
            session_id="ses_abc123",
        )
    finally:
        mod._agent = original


@pytest.mark.asyncio
async def test_ingest_pdf_tool_defaults_session_id_to_empty_string(mcp):
    """Omitting session_id must preserve today's exact behavior (empty-string default, matching
    BrainAgent.ingest_pdf's own default) — additive, non-breaking."""
    import openclaw_brain.server.mcp_server as mod
    original = mod._agent

    mock_agent = AsyncMock()
    mock_agent.is_started = True
    mock_result = MagicMock()
    mock_result.output = {}
    mock_agent.ingest_pdf.return_value = mock_result
    mod._agent = mock_agent

    try:
        tool = mcp._tool_manager.get_tool("ingest_pdf")
        await tool.run({"file_path": "/tmp/test.pdf"}, None)
        mock_agent.ingest_pdf.assert_awaited_once_with(
            file_path="/tmp/test.pdf", extraction_model=None, reasoning_model=None,
            session_id="",
        )
    finally:
        mod._agent = original


@pytest.mark.asyncio
async def test_reinforce_concept_tool(mcp):
    """Test reinforce_concept delegates to agent."""
    import openclaw_brain.server.mcp_server as mod
    original = mod._agent

    mock_agent = AsyncMock()
    mock_agent.is_started = True
    mod._agent = mock_agent

    try:
        tool = mcp._tool_manager.get_tool("reinforce_concept")
        result = await tool.run({"concept_id": "gm_001", "evidence": "confirmed"}, None)
        assert "reinforced" in result
        mock_agent.reinforce.assert_awaited_once()
    finally:
        mod._agent = original


@pytest.mark.asyncio
async def test_get_stats_tool(mcp):
    """Test get_stats returns JSON."""
    import openclaw_brain.server.mcp_server as mod
    original = mod._agent

    mock_agent = AsyncMock()
    mock_agent.is_started = True
    mock_agent.get_stats.return_value = {"graph": {"concepts": 10}, "memory": {"total": 5}}
    mod._agent = mock_agent

    try:
        tool = mcp._tool_manager.get_tool("get_stats")
        result = await tool.run({}, None)
        parsed = json.loads(result)
        assert parsed["graph"]["concepts"] == 10
    finally:
        mod._agent = original


@pytest.mark.asyncio
async def test_start_session_tool(mcp):
    """Test start_session delegates to agent."""
    import openclaw_brain.server.mcp_server as mod
    original = mod._agent

    mock_agent = AsyncMock()
    mock_agent.is_started = True
    mock_agent.start_session.return_value = "ses_abc123"
    mod._agent = mock_agent

    try:
        tool = mcp._tool_manager.get_tool("start_session")
        result = await tool.run({"session_id": "ses_abc123"}, None)
        assert "ses_abc123" in result
    finally:
        mod._agent = original


@pytest.mark.asyncio
async def test_record_lesson_tool(mcp):
    """Test record_lesson parses tags correctly."""
    import openclaw_brain.server.mcp_server as mod
    original = mod._agent

    mock_agent = AsyncMock()
    mock_agent.is_started = True
    mock_agent.record_lesson.return_value = "lesson_001"
    mod._agent = mock_agent

    try:
        tool = mcp._tool_manager.get_tool("record_lesson")
        result = await tool.run({"lesson": "Always check Vgs", "tags": "mosfet, design"}, None)
        assert "lesson_001" in result
        # Verify tags were split correctly
        call_args = mock_agent.record_lesson.call_args
        assert call_args[0][1] == ["mosfet", "design"]
    finally:
        mod._agent = original


@pytest.mark.asyncio
async def test_record_assessment_tool(mcp):
    """Test record_assessment delegates to agent with the exact 4 args (design doc §6.1's
    locked tool signature)."""
    import openclaw_brain.server.mcp_server as mod
    original = mod._agent

    mock_agent = AsyncMock()
    mock_agent.is_started = True
    mock_agent.record_assessment.return_value = "assess_20260710_000000_000000"
    mod._agent = mock_agent

    try:
        tool = mcp._tool_manager.get_tool("record_assessment")
        result = await tool.run({
            "learner_id": "rick", "target_id": "concept_gm",
            "verdict": "understood", "evidence": "explained gm=2Id/Vov",
        }, None)
        assert "assess_20260710_000000_000000" in result
        mock_agent.record_assessment.assert_awaited_once_with(
            learner_id="rick", target_id="concept_gm",
            verdict="understood", evidence="explained gm=2Id/Vov",
        )
    finally:
        mod._agent = original


@pytest.mark.asyncio
async def test_get_learner_state_tool(mcp):
    """Test get_learner_state delegates to agent and renders JSON; empty target_id ("") maps
    to None, matching query_executable's `topology_class or None` convention."""
    import openclaw_brain.server.mcp_server as mod
    original = mod._agent

    mock_agent = AsyncMock()
    mock_agent.is_started = True
    mock_agent.get_learner_state.return_value = [
        {"target_id": "concept_gm", "target_label": "Concept", "target_name": "Transconductance",
         "confidence": 0.9, "status": "understood", "last_assessed": "2026-07-10T00:00:00",
         "assessment_count": 1},
    ]
    mod._agent = mock_agent

    try:
        tool = mcp._tool_manager.get_tool("get_learner_state")
        result = await tool.run({"learner_id": "rick"}, None)
        parsed = json.loads(result)
        assert parsed[0]["target_id"] == "concept_gm"
        mock_agent.get_learner_state.assert_awaited_once_with("rick", None)
    finally:
        mod._agent = original


@pytest.mark.asyncio
async def test_get_learner_state_tool_empty_state(mcp):
    import openclaw_brain.server.mcp_server as mod
    original = mod._agent

    mock_agent = AsyncMock()
    mock_agent.is_started = True
    mock_agent.get_learner_state.return_value = []
    mod._agent = mock_agent

    try:
        tool = mcp._tool_manager.get_tool("get_learner_state")
        result = await tool.run({"learner_id": "rick", "target_id": "concept_gm"}, None)
        assert "No understanding recorded" in result
        mock_agent.get_learner_state.assert_awaited_once_with("rick", "concept_gm")
    finally:
        mod._agent = original


@pytest.mark.asyncio
async def test_route_and_execute_tool(mcp):
    """Test route_and_execute delegates to agent."""
    import openclaw_brain.server.mcp_server as mod
    original = mod._agent

    mock_agent = AsyncMock()
    mock_agent.is_started = True
    mock_agent.route_and_execute.return_value = {"routed": True, "skill": "pdf_ingest"}
    mod._agent = mock_agent

    try:
        tool = mcp._tool_manager.get_tool("route_and_execute")
        result = await tool.run({"user_input": "ingest this PDF"}, None)
        parsed = json.loads(result)
        assert parsed["routed"] is True
    finally:
        mod._agent = original


@pytest.mark.asyncio
async def test_route_and_execute_tool_forwards_session_id(mcp):
    """W-D2 defect 6: route_and_execute previously had no session_id parameter either — the
    other of the only two production paths into SkillExecutor.execute()."""
    import openclaw_brain.server.mcp_server as mod
    original = mod._agent

    mock_agent = AsyncMock()
    mock_agent.is_started = True
    mock_agent.route_and_execute.return_value = {"routed": True, "skill": "pdf_ingest"}
    mod._agent = mock_agent

    try:
        tool = mcp._tool_manager.get_tool("route_and_execute")
        await tool.run({"user_input": "ingest this PDF", "session_id": "ses_xyz"}, None)
        mock_agent.route_and_execute.assert_awaited_once_with(
            "ingest this PDF", session_id="ses_xyz",
        )
    finally:
        mod._agent = original


@pytest.mark.asyncio
async def test_route_and_execute_tool_defaults_session_id_to_empty_string(mcp):
    import openclaw_brain.server.mcp_server as mod
    original = mod._agent

    mock_agent = AsyncMock()
    mock_agent.is_started = True
    mock_agent.route_and_execute.return_value = {"routed": False}
    mod._agent = mock_agent

    try:
        tool = mcp._tool_manager.get_tool("route_and_execute")
        await tool.run({"user_input": "ingest this PDF"}, None)
        mock_agent.route_and_execute.assert_awaited_once_with(
            "ingest this PDF", session_id="",
        )
    finally:
        mod._agent = original


@pytest.mark.asyncio
async def test_run_maintenance_tool(mcp):
    """Test run_maintenance combines promotion and decay."""
    import openclaw_brain.server.mcp_server as mod
    original = mod._agent

    mock_agent = AsyncMock()
    mock_agent.is_started = True
    mock_agent.run_promotion.return_value = {
        "raw_to_retain": 2, "retain_to_curated": 1, "expired": 0,
    }
    mock_agent.run_decay.return_value = 3
    mod._agent = mock_agent

    try:
        tool = mcp._tool_manager.get_tool("run_maintenance")
        result = await tool.run({}, None)
        assert "2 memories promoted raw→retain" in result
        assert "1 memories promoted retain→curated" in result
        assert "3 concepts confidence decayed" in result
    finally:
        mod._agent = original


@pytest.mark.asyncio
async def test_run_maintenance_no_actions(mcp):
    """Test run_maintenance when nothing needs doing."""
    import openclaw_brain.server.mcp_server as mod
    original = mod._agent

    mock_agent = AsyncMock()
    mock_agent.is_started = True
    mock_agent.run_promotion.return_value = {
        "raw_to_retain": 0, "retain_to_curated": 0, "expired": 0,
    }
    mock_agent.run_decay.return_value = 0
    mod._agent = mock_agent

    try:
        tool = mcp._tool_manager.get_tool("run_maintenance")
        result = await tool.run({}, None)
        assert result == "No maintenance actions needed."
    finally:
        mod._agent = original


# ── ADR-044 D1: read-only MCP server profile ──


def test_default_mode_registers_all_39_tools(mcp):
    """Default (non-readonly) surface: all 41 tools registered (37 + record_assessment +
    get_learner_state, S5a learner model + ingest_html, the HTML lecture-notes ingest adapter +
    audit_answer, the answer-surface citation audit)."""
    tool_names = {t.name for t in mcp._tool_manager.list_tools()}
    assert len(tool_names) == 41
    assert "why_law" in tool_names
    assert "record_assessment" in tool_names
    assert "get_learner_state" in tool_names
    assert "ingest_html" in tool_names
    assert "audit_answer" in tool_names


def test_readonly_registers_only_allowed_subset():
    """ADR-044 D1: readonly mode registers exactly the documented tool subset —
    absence is the enforcement (excluded tools are never registered)."""
    from openclaw_brain.server.mcp_server import READONLY_TOOLS

    expected = {
        "startup", "query_knowledge", "answer_question",
        "recall_memory", "find_bridges", "list_open_hypotheses",
        "query_executable", "why", "why_law", "audit_citations", "audit_answer",
        "get_stats", "get_pipeline_config",
    }
    assert READONLY_TOOLS == expected
    # 2026-08-16 shared-SSE incident: shutdown on the readonly surface let any remote
    # reader stop the shared daemon's agent for every other consumer. Pinned out.
    assert "shutdown" not in READONLY_TOOLS

    mcp_ro = create_server(readonly=True)
    tool_names = {t.name for t in mcp_ro._tool_manager.list_tools()}
    assert tool_names == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["full", "researcher", "readonly"])
async def test_answer_envelope_roundtrip_all_profiles(profile):
    import openclaw_brain.server.mcp_server as mod
    original = mod._agent
    mock_agent = AsyncMock()
    mock_agent.is_started = True
    mock_agent.answer_question.return_value = {
        "answer": "", "citations": ["bad"], "abstained": True,
        "abstain_reason": "contract_invalid:citation_missing", "used_concepts": [],
        "concepts_found": 1, "concepts": [],
        "envelope": {"schema_version": "answer-envelope/0.1", "status": "invalid",
                     "reason_codes": ["citation_missing"], "gap_recorded": False,
                     "checks": {"kb_coverage": "none", "model_knowledge": True}},
    }
    mod._agent = mock_agent
    try:
        server = create_server(profile=profile)
        result = await server._tool_manager.get_tool("answer_question").run(
            {"question": "q", "on_insufficient": "abstain"}, None)
        assert json.loads(result)["envelope"] == mock_agent.answer_question.return_value["envelope"]
        mock_agent.answer_question.assert_awaited_once_with("q", on_insufficient="abstain")
        mock_agent.answer_question.reset_mock()
        default_result = await server._tool_manager.get_tool("answer_question").run(
            {"question": "q"}, None)
        assert json.loads(default_result)["envelope"] == mock_agent.answer_question.return_value["envelope"]
        mock_agent.answer_question.assert_awaited_once_with("q", on_insufficient="model_knowledge")
        if profile == "readonly":
            assert server._tool_manager.get_tool("get_evidence") is None
    finally:
        mod._agent = original


def test_readonly_excludes_write_and_heavy_tools():
    """Every write/mutating/heavy/policy tool (ADR-044 D1) must be absent —
    including get_evidence, since EvidenceVault verbatim text is not
    distributed shared-service side."""
    mcp_ro = create_server(readonly=True)
    tool_names = {t.name for t in mcp_ro._tool_manager.list_tools()}

    excluded = {
        "ingest_pdf", "ingest_html", "analyze_circuit_image", "get_evidence",
        "reinforce_concept",
        "start_session", "end_session", "record_event",
        "upsert_entity", "record_lesson",
        "record_hypothesis", "record_decision", "record_bench_result",
        "record_assessment", "get_learner_state",
        "merge_concepts", "retract_node",
        "project_executable", "retract_executable", "grow_executable",
        "run_maintenance", "route_and_execute", "export_obsidian",
        "update_stage_model", "update_fallback_chain",
        "add_catalog_model", "remove_catalog_model",
    }
    assert tool_names.isdisjoint(excluded), f"Leaked write tools: {tool_names & excluded}"
    assert len(tool_names) == 13  # 14 -> 13 on 2026-08-16: shutdown pinned out (shared-SSE incident)


def test_readonly_excludes_learner_model_tools():
    """S5_LEARNER_MODEL_DESIGN.md §4.1 (locked): home-only for the WHOLE S5 phase, on
    mechanism-safety grounds — record_assessment/get_learner_state must never reach the
    shared-service read-only profile, pinned as its own dedicated assertion (not just bundled
    into the generic excluded-tools set above) since this is the specific privacy-boundary
    decision §1.4's contamination finding motivates."""
    from openclaw_brain.server.mcp_server import READONLY_TOOLS

    assert "record_assessment" not in READONLY_TOOLS
    assert "get_learner_state" not in READONLY_TOOLS

    mcp_ro = create_server(readonly=True)
    tool_names = {t.name for t in mcp_ro._tool_manager.list_tools()}
    assert "record_assessment" not in tool_names
    assert "get_learner_state" not in tool_names


def test_readonly_resources_unchanged():
    """brain://stats, brain://models, brain://guide stay registered in readonly mode."""
    mcp_ro = create_server(readonly=True)
    resource_uris = {str(r.uri) for r in mcp_ro._resource_manager.list_resources()}
    assert resource_uris == {"brain://stats", "brain://models", "brain://guide"}


@pytest.mark.asyncio
async def test_readonly_guide_omits_write_tools():
    """brain://guide in readonly mode must not teach tools that don't exist
    in this profile (e.g. record_hypothesis, get_evidence, merge_concepts)."""
    mcp_ro = create_server(readonly=True)
    resource = await mcp_ro._resource_manager.get_resource("brain://guide", None)
    guide = await resource.read()
    for banned in ("record_hypothesis", "record_decision", "record_bench_result",
                   "reinforce_concept", "get_evidence", "merge_concepts",
                   "retract_node", "ingest_pdf",
                   "record_assessment", "get_learner_state"):
        assert banned not in guide, f"readonly guide leaks write-tool name: {banned}"


@pytest.mark.asyncio
async def test_default_guide_still_teaches_write_tools(mcp):
    """Non-readonly guide is unchanged — still teaches the write loop."""
    resource = await mcp._resource_manager.get_resource("brain://guide", None)
    guide = await resource.read()
    assert "record_hypothesis" in guide
    assert "get_evidence" in guide


@pytest.mark.asyncio
async def test_default_guide_teaches_learner_model_tools(mcp):
    """S5.0 (docs/specs/S5_LEARNER_MODEL_DESIGN.md §6.1 item 6): closes ADR-044 D4's
    documentation clause — _AGENT_GUIDE must teach record_assessment/get_learner_state."""
    resource = await mcp._resource_manager.get_resource("brain://guide", None)
    guide = await resource.read()
    assert "record_assessment" in guide
    assert "get_learner_state" in guide
    assert "Learner model" in guide


def test_readonly_via_env_var(monkeypatch):
    """OPENCLAW_BRAIN_READONLY=1 activates readonly mode when readonly= is not
    passed explicitly."""
    monkeypatch.setenv("OPENCLAW_BRAIN_READONLY", "1")
    mcp_env = create_server()
    tool_names = {t.name for t in mcp_env._tool_manager.list_tools()}
    assert len(tool_names) == 13  # 14 -> 13 on 2026-08-16: shutdown pinned out (shared-SSE incident)
    assert "ingest_pdf" not in tool_names


def test_env_var_readonly_off_by_default(monkeypatch):
    """Without the env var (and without readonly=True), the full surface is served."""
    monkeypatch.delenv("OPENCLAW_BRAIN_READONLY", raising=False)
    mcp_default = create_server()
    tool_names = {t.name for t in mcp_default._tool_manager.list_tools()}
    assert len(tool_names) == 41


def test_explicit_readonly_false_overrides_env_var(monkeypatch):
    """An explicit readonly=False wins over a truthy env var."""
    monkeypatch.setenv("OPENCLAW_BRAIN_READONLY", "1")
    mcp_explicit = create_server(readonly=False)
    tool_names = {t.name for t in mcp_explicit._tool_manager.list_tools()}
    assert len(tool_names) == 41


# ── Researcher profile (2026-08-16 home-lab pilot) ──


def test_researcher_registers_exactly_expected_subset():
    """Home-lab research sessions get all reads + verbatim evidence + the gated
    record loop + learner model + oracle-gated projection — and nothing else."""
    from openclaw_brain.server.mcp_server import RESEARCHER_TOOLS, READONLY_TOOLS

    expected = READONLY_TOOLS | {
        "get_evidence",
        "record_hypothesis", "record_bench_result", "record_decision",
        "record_assessment", "get_learner_state",
        "reinforce_concept",
        "project_executable",
    }
    assert RESEARCHER_TOOLS == expected

    mcp_r = create_server(profile="researcher")
    tool_names = {t.name for t in mcp_r._tool_manager.list_tools()}
    assert tool_names == expected
    assert len(tool_names) == 21


def test_researcher_excludes_curation_and_mutation_tools():
    """The 2026-08-15 label-contamination census (350 mislabeled nodes) is what
    ungated writes cost — research sessions must not see graph mutation, ingestion,
    config mutation, lifecycle teardown, or the D3/D-7-gated tools."""
    mcp_r = create_server(profile="researcher")
    tool_names = {t.name for t in mcp_r._tool_manager.list_tools()}
    excluded = {
        "shutdown", "ingest_pdf", "ingest_html", "merge_concepts", "retract_node",
        "retract_executable", "grow_executable", "analyze_circuit_image",
        "run_maintenance", "route_and_execute", "export_obsidian",
        "update_stage_model", "update_fallback_chain",
        "add_catalog_model", "remove_catalog_model",
        "start_session", "end_session", "record_event", "record_lesson",
        "upsert_entity",
    }
    assert tool_names.isdisjoint(excluded), f"Leaked tools: {tool_names & excluded}"


def test_profile_env_var_and_precedence(monkeypatch):
    monkeypatch.setenv("OPENCLAW_BRAIN_PROFILE", "researcher")
    from openclaw_brain.server.mcp_server import RESEARCHER_TOOLS

    mcp_r = create_server()
    assert {t.name for t in mcp_r._tool_manager.list_tools()} == RESEARCHER_TOOLS

    # Explicit profile beats the env var; explicit readonly=True maps to readonly.
    mcp_full = create_server(profile="full")
    assert len({t.name for t in mcp_full._tool_manager.list_tools()}) > len(RESEARCHER_TOOLS)
    from openclaw_brain.server.mcp_server import READONLY_TOOLS
    mcp_ro = create_server(readonly=True, profile="readonly")
    assert {t.name for t in mcp_ro._tool_manager.list_tools()} == READONLY_TOOLS


def test_unknown_profile_raises():
    with pytest.raises(ValueError, match="unknown MCP profile"):
        create_server(profile="admin")
