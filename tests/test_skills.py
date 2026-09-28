"""Tests for skill registry, router, and executor."""

from unittest.mock import AsyncMock

import pytest
from tests.conftest import require_live_graph

from openclaw_brain.config import load_config
from openclaw_brain.knowledge.graph.schema import NodeLabel
from openclaw_brain.knowledge.graph.store import GraphStore
from openclaw_brain.memory.procedural import ProceduralMemory
from openclaw_brain.memory.store import MemoryStore
from openclaw_brain.skills.executor import SkillExecutor
from openclaw_brain.skills.registry import (
    SkillCategory,
    SkillDefinition,
    SkillRegistry,
)
from openclaw_brain.skills.router import SkillRouter


# ── Test helpers ──


async def _echo_handler(params):
    return {"echo": params.get("text", "")}


async def _fail_handler(params):
    raise ValueError("intentional failure")


async def _slow_handler(params):
    return {"result": "done"}


def _make_registry() -> SkillRegistry:
    """Create a registry with test skills."""
    reg = SkillRegistry()

    reg.register(
        SkillDefinition(
            name="pdf_extract",
            description="Extract concepts from a PDF document",
            category=SkillCategory.KNOWLEDGE,
            trigger_keywords=["pdf", "extract", "document"],
            trigger_patterns=[r"extract\s+from\s+", r"process\s+pdf"],
            confidence=0.8,
            input_schema={"file_path": "str", "pages": "str (optional)"},
            output_schema={"concepts": "list[str]", "count": "int"},
        ),
        _echo_handler,
    )

    reg.register(
        SkillDefinition(
            name="render_equation",
            description="Render a LaTeX equation as an image",
            category=SkillCategory.RENDERING,
            trigger_keywords=["render", "equation", "latex", "plot"],
            trigger_patterns=[r"\$.*\$", r"render\s+"],
            confidence=0.9,
        ),
        _echo_handler,
    )

    reg.register(
        SkillDefinition(
            name="graph_query",
            description="Query the knowledge graph for related concepts",
            category=SkillCategory.KNOWLEDGE,
            trigger_keywords=["graph", "related", "connections", "neighbors"],
            trigger_patterns=[r"what\s+is\s+related\s+to", r"connections?\s+of"],
            confidence=0.7,
        ),
        _echo_handler,
    )

    return reg


# ── SkillRegistry tests ──


def test_registry_register_and_get():
    reg = _make_registry()
    assert reg.count == 3
    skill = reg.get("pdf_extract")
    assert skill is not None
    assert skill.category == SkillCategory.KNOWLEDGE


def test_registry_unregister():
    reg = _make_registry()
    assert reg.unregister("pdf_extract")
    assert reg.count == 2
    assert reg.get("pdf_extract") is None
    assert not reg.unregister("nonexistent")


def test_registry_list_by_category():
    reg = _make_registry()
    knowledge = reg.list_skills(category=SkillCategory.KNOWLEDGE)
    assert len(knowledge) == 2
    assert all(s.category == SkillCategory.KNOWLEDGE for s in knowledge)


def test_registry_find_by_keyword():
    reg = _make_registry()
    matches = reg.find_by_keyword("equation")
    assert len(matches) >= 1
    assert matches[0].name == "render_equation"


def test_registry_update_stats():
    reg = _make_registry()
    reg.update_stats("pdf_extract", success=True)
    reg.update_stats("pdf_extract", success=True)
    reg.update_stats("pdf_extract", success=False)
    skill = reg.get("pdf_extract")
    assert skill is not None
    assert skill.total_runs == 3
    assert skill.successful_runs == 2


def test_registry_confidence_adjustment():
    reg = SkillRegistry()
    reg.register(
        SkillDefinition(
            name="test_skill",
            description="test",
            category=SkillCategory.SYSTEM,
            confidence=0.5,
        ),
        _echo_handler,
    )
    # Run 10 successes → confidence should increase
    for _ in range(10):
        reg.update_stats("test_skill", success=True)
    skill = reg.get("test_skill")
    assert skill is not None
    assert skill.confidence > 0.5


def test_registry_to_prompt():
    reg = _make_registry()
    prompt = reg.to_prompt_description()
    assert "pdf_extract" in prompt
    assert "render_equation" in prompt
    assert "Confidence" in prompt


def test_registry_disabled_skill():
    reg = _make_registry()
    skill = reg.get("pdf_extract")
    skill.enabled = False
    enabled = reg.list_skills(enabled_only=True)
    assert not any(s.name == "pdf_extract" for s in enabled)


# ── SkillRouter tests ──


@pytest.mark.asyncio
async def test_router_keyword_match():
    reg = _make_registry()
    router = SkillRouter(reg)
    result = await router.route("Please extract concepts from this PDF")
    assert result.selected is not None
    assert result.selected.skill_name == "pdf_extract"
    assert not result.fallback


@pytest.mark.asyncio
async def test_router_pattern_match():
    reg = _make_registry()
    router = SkillRouter(reg)
    result = await router.route("What is related to MOSFET?")
    assert result.selected is not None
    assert result.selected.skill_name == "graph_query"


@pytest.mark.asyncio
async def test_router_no_match():
    reg = _make_registry()
    router = SkillRouter(reg)
    result = await router.route("How is the weather today?")
    assert result.fallback


@pytest.mark.asyncio
async def test_router_multiple_candidates():
    reg = _make_registry()
    router = SkillRouter(reg)
    # "render equation" matches render_equation clearly
    result = await router.route("Can you render this equation for me?")
    assert result.selected is not None
    assert len(result.candidates) >= 1


@pytest.mark.asyncio
async def test_router_with_empty_registry():
    reg = SkillRegistry()
    router = SkillRouter(reg)
    result = await router.route("anything")
    assert result.fallback


# ── SkillExecutor tests ──


@pytest.mark.asyncio
async def test_executor_success():
    reg = _make_registry()
    executor = SkillExecutor(reg)
    result = await executor.execute("pdf_extract", {"text": "hello"})
    assert result.success
    assert result.output == {"echo": "hello"}
    assert result.duration_seconds > 0


@pytest.mark.asyncio
async def test_executor_failure():
    reg = SkillRegistry()
    reg.register(
        SkillDefinition(name="bad_skill", description="fails", category=SkillCategory.SYSTEM),
        _fail_handler,
    )
    executor = SkillExecutor(reg)
    result = await executor.execute("bad_skill", {})
    assert not result.success
    assert "intentional failure" in result.error


@pytest.mark.asyncio
async def test_executor_missing_skill():
    reg = SkillRegistry()
    executor = SkillExecutor(reg)
    result = await executor.execute("nonexistent", {})
    assert not result.success
    assert "not found" in result.error


@pytest.mark.asyncio
async def test_executor_disabled_skill():
    reg = _make_registry()
    skill = reg.get("pdf_extract")
    skill.enabled = False
    executor = SkillExecutor(reg)
    result = await executor.execute("pdf_extract", {})
    assert not result.success
    assert "disabled" in result.error


@pytest.mark.asyncio
async def test_executor_updates_registry_stats():
    reg = _make_registry()
    executor = SkillExecutor(reg)
    await executor.execute("pdf_extract", {"text": "a"})
    await executor.execute("pdf_extract", {"text": "b"})
    skill = reg.get("pdf_extract")
    assert skill.total_runs == 2
    assert skill.successful_runs == 2


# ── SkillExecutor + procedural memory (mocked — W-D2 defects 5/6) ──


@pytest.mark.asyncio
async def test_executor_records_procedural_memory_on_success():
    """session_id + a working ProceduralMemory -> record_skill_run fires and run_id surfaces."""
    reg = _make_registry()
    mock_procedural = AsyncMock(spec=ProceduralMemory)
    mock_procedural.record_skill_run.return_value = "run_abc123"
    executor = SkillExecutor(reg, procedural=mock_procedural)

    result = await executor.execute("pdf_extract", {"text": "hello"}, session_id="ses_1")

    assert result.success is True
    assert result.run_id == "run_abc123"
    mock_procedural.record_skill_run.assert_awaited_once()
    call_kwargs = mock_procedural.record_skill_run.call_args.kwargs
    assert call_kwargs["skill_name"] == "pdf_extract"
    assert call_kwargs["session_id"] == "ses_1"
    assert call_kwargs["success"] is True


@pytest.mark.asyncio
async def test_executor_without_session_id_skips_procedural_memory():
    """Defect 6 context: an empty session_id (today's MCP-tool default before the fix) means
    record_skill_run is never even attempted — this is the existing, intentional gate
    (`if self._procedural and session_id`), unchanged; the fix is that production entry points
    now CAN pass a real session_id (see server/mcp_server.py's ingest_pdf/route_and_execute)."""
    reg = _make_registry()
    mock_procedural = AsyncMock(spec=ProceduralMemory)
    executor = SkillExecutor(reg, procedural=mock_procedural)

    result = await executor.execute("pdf_extract", {"text": "hello"})   # session_id="" default

    assert result.success is True
    assert result.run_id == ""
    mock_procedural.record_skill_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_executor_procedural_memory_failure_does_not_discard_handler_result():
    """W-D2 defect 5: a failure in the post-handler procedural-memory write must not discard
    visibility into an already-successful handler result. Previously the exception propagated out
    of execute() uncaught, so ExecutionResult(success=True, ...) was never constructed even though
    the handler itself had already succeeded (and, for e.g. a multi-hour pdf_ingest, already
    committed to the graph) — a false-failure signal risking a costly duplicate retry."""
    reg = _make_registry()
    mock_procedural = AsyncMock(spec=ProceduralMemory)
    mock_procedural.record_skill_run.side_effect = RuntimeError("neo4j blip")
    executor = SkillExecutor(reg, procedural=mock_procedural)

    result = await executor.execute("pdf_extract", {"text": "hello"}, session_id="ses_1")

    assert result.success is True
    assert result.output == {"echo": "hello"}
    assert result.run_id == ""   # procedural write failed -> no run_id, but the result still surfaces


@pytest.mark.asyncio
async def test_executor_procedural_memory_failure_does_not_mask_handler_failure():
    """Same isolation, mirrored for a handler that itself failed: the procedural-memory write
    failing on top of that must not change or obscure the original handler error."""
    reg = SkillRegistry()
    reg.register(
        SkillDefinition(name="bad_skill", description="fails", category=SkillCategory.SYSTEM),
        _fail_handler,
    )
    mock_procedural = AsyncMock(spec=ProceduralMemory)
    mock_procedural.record_skill_run.side_effect = RuntimeError("neo4j blip")
    executor = SkillExecutor(reg, procedural=mock_procedural)

    result = await executor.execute("bad_skill", {}, session_id="ses_1")

    assert result.success is False
    assert "intentional failure" in result.error
    assert result.run_id == ""


# ── Integration: executor + procedural memory ──


@pytest.fixture
async def graph():
    require_live_graph()
    config = load_config()
    g = GraphStore(config.neo4j)
    try:
        await g.connect()
    except Exception:
        pytest.skip("Neo4j not available")
    yield g
    async with await g._session() as session:
        await session.run("MATCH (n:Memory) WHERE n.memory_id STARTS WITH 'proc_' DETACH DELETE n")
        await session.run("MATCH (n:SkillRun) WHERE n.run_id STARTS WITH 'run_' DETACH DELETE n")
        await session.run("MATCH (n:Session) WHERE n.session_id = 'test_exec_session' DETACH DELETE n")
    await g.close()


@pytest.mark.asyncio
async def test_executor_records_to_procedural_memory(graph: GraphStore):
    config = load_config()
    mem_store = MemoryStore(graph, config)
    proc = ProceduralMemory(mem_store, graph)

    # Create test session
    await graph.merge_node(
        label=NodeLabel.SESSION,
        id_field="session_id",
        id_value="test_exec_session",
        properties={"session_id": "test_exec_session", "status": "active"},
    )

    reg = _make_registry()
    executor = SkillExecutor(reg, procedural=proc)
    result = await executor.execute(
        "pdf_extract",
        {"text": "test doc"},
        session_id="test_exec_session",
    )
    assert result.success
    assert result.run_id.startswith("run_")

    # Verify procedural memory was recorded
    stats = await proc.get_skill_stats("pdf_extract")
    assert stats.total_runs >= 1
