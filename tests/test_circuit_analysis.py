"""Tests for analyze_circuit / analyze_circuit_image MCP tool."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openclaw_brain.knowledge.extraction.figure_analyzer import FigureAnalysis


# ── Helpers ──

def _mock_agent(
    classify_result: str = "circuit",
    analysis_description: str = "Two-stage Miller OTA with NMOS input pair.",
    knowledge_context: str = "## Related Knowledge\nConcept: differential_pair",
    figure_analysis_model: str = "grok-4.20",
    vision_model: str = "qwen3-vl-8b",
):
    """Create a minimal BrainAgent mock for analyze_circuit tests."""
    from openclaw_brain.config import BrainConfig, ModelsConfig
    from openclaw_brain.retriever import GraphContext, UnifiedContext

    cfg = BrainConfig()
    cfg.models.default_vision = vision_model
    cfg.models.default_figure_analysis = figure_analysis_model

    ctx = MagicMock(spec=UnifiedContext)
    ctx.format.return_value = knowledge_context

    agent = MagicMock()
    agent._config = cfg
    agent._started = True
    agent.is_started = True
    agent._assert_started = lambda: None

    # Fake retriever
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=ctx)
    agent._retriever = retriever

    return agent


# ── figure_analyzer module tests ──

class TestClassifyByCaption:
    def test_circuit_caption(self):
        from openclaw_brain.knowledge.extraction.figure_analyzer import classify_by_caption
        assert classify_by_caption("Schematic of two-stage OTA") == "circuit"

    def test_plot_caption(self):
        from openclaw_brain.knowledge.extraction.figure_analyzer import classify_by_caption
        assert classify_by_caption("Frequency response of the amplifier") == "plot"

    def test_layout_caption(self):
        from openclaw_brain.knowledge.extraction.figure_analyzer import classify_by_caption
        assert classify_by_caption("Layout of the OTA core") == "layout"

    def test_block_diagram_caption(self):
        from openclaw_brain.knowledge.extraction.figure_analyzer import classify_by_caption
        assert classify_by_caption("Block diagram of the system") == "block_diagram"

    def test_empty_caption_returns_unknown(self):
        from openclaw_brain.knowledge.extraction.figure_analyzer import classify_by_caption
        assert classify_by_caption("") == "unknown"

    def test_unrecognized_caption_returns_unknown(self):
        from openclaw_brain.knowledge.extraction.figure_analyzer import classify_by_caption
        assert classify_by_caption("Some random unrelated text here") == "unknown"


# ── BrainAgent.analyze_circuit tests ──

class TestAnalyzeCircuit:
    """Tests for BrainAgent.analyze_circuit() orchestration."""

    @pytest.fixture
    def circuit_image(self, tmp_path) -> Path:
        """Create a minimal valid PNG file for testing."""
        import base64
        # 1x1 white PNG
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
            "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        img_path = tmp_path / "test_circuit.png"
        img_path.write_bytes(png_bytes)
        return img_path

    @pytest.mark.asyncio
    async def test_analyze_circuit_returns_expected_keys(self, circuit_image):
        """analyze_circuit should return figure_type, visual_analysis, knowledge_context."""
        from openclaw_brain.agent import BrainAgent

        fake_analysis = FigureAnalysis(
            figure_type="circuit",
            description="Folded cascode OTA with PMOS input pair and tail current source.",
            page_idx=0,
        )

        with (
            patch("openclaw_brain.agent.classify_by_caption", return_value="unknown"),
            patch("openclaw_brain.agent.classify_with_vlm", new_callable=AsyncMock, return_value="circuit"),
            patch("openclaw_brain.agent.analyze_figure", new_callable=AsyncMock, return_value=fake_analysis),
        ):
            agent = _mock_agent()
            # Inject fake llm_provider
            fake_provider = MagicMock()
            fake_provider.get_model.return_value = MagicMock()
            agent._llm_provider = fake_provider

            # Patch _assert_started
            with patch.object(BrainAgent, "_assert_started", lambda self: None):
                # Call directly on agent (bypassing __class__ check)
                result = await BrainAgent.analyze_circuit(agent, str(circuit_image))

        assert "figure_type" in result
        assert "visual_analysis" in result
        assert "knowledge_context" in result
        assert result["figure_type"] == "circuit"
        assert "Folded cascode" in result["visual_analysis"]

    @pytest.mark.asyncio
    async def test_analyze_circuit_no_vision_model(self, circuit_image):
        """When default_vision is empty, should skip VLM classification."""
        from openclaw_brain.agent import BrainAgent

        fake_analysis = FigureAnalysis(
            figure_type="other",
            description="Unable to classify image.",
            page_idx=0,
        )

        with (
            patch("openclaw_brain.agent.classify_by_caption", return_value="unknown"),
            patch("openclaw_brain.agent.analyze_figure", new_callable=AsyncMock, return_value=fake_analysis),
        ):
            agent = _mock_agent(vision_model="")  # no vision model
            fake_provider = MagicMock()
            fake_provider.get_model.return_value = MagicMock()
            agent._llm_provider = fake_provider

            with patch.object(BrainAgent, "_assert_started", lambda self: None):
                result = await BrainAgent.analyze_circuit(agent, str(circuit_image))

        assert result["figure_type"] == "other"

    @pytest.mark.asyncio
    async def test_analyze_circuit_no_figure_analysis_model(self, circuit_image):
        """When default_figure_analysis is empty, skip analysis and return placeholder."""
        from openclaw_brain.agent import BrainAgent

        with (
            patch("openclaw_brain.agent.classify_by_caption", return_value="circuit"),
        ):
            agent = _mock_agent(figure_analysis_model="")  # disabled
            fake_provider = MagicMock()
            agent._llm_provider = fake_provider

            with patch.object(BrainAgent, "_assert_started", lambda self: None):
                result = await BrainAgent.analyze_circuit(agent, str(circuit_image))

        assert result["figure_type"] == "circuit"
        assert "No analysis model configured" in result["visual_analysis"]

    @pytest.mark.asyncio
    async def test_analyze_circuit_knowledge_context_used(self, circuit_image):
        """Retrieved knowledge context should appear in result."""
        from openclaw_brain.agent import BrainAgent

        fake_analysis = FigureAnalysis(
            figure_type="circuit",
            description="CS amplifier with NMOS transistor.",
            page_idx=0,
        )
        expected_context = "## Related Knowledge\nConcept: common_source_amplifier"

        with (
            patch("openclaw_brain.agent.classify_by_caption", return_value="circuit"),
            patch("openclaw_brain.agent.analyze_figure", new_callable=AsyncMock, return_value=fake_analysis),
        ):
            agent = _mock_agent(knowledge_context=expected_context)
            fake_provider = MagicMock()
            fake_provider.get_model.return_value = MagicMock()
            agent._llm_provider = fake_provider

            with patch.object(BrainAgent, "_assert_started", lambda self: None):
                result = await BrainAgent.analyze_circuit(agent, str(circuit_image))

        assert result["knowledge_context"] == expected_context
        # Verify retriever was called with the VLM analysis text
        agent._retriever.retrieve.assert_called_once_with(fake_analysis.description)


# ── MCP tool surface test ──

def test_analyze_circuit_image_tool_registered():
    """Verify analyze_circuit_image is registered as an MCP tool."""
    from openclaw_brain.server.mcp_server import create_server
    server = create_server()
    # FastMCP stores tools in ._tool_manager or similar; check by duck-typing
    tool_names = [t.name for t in server._tool_manager.list_tools()]
    assert "analyze_circuit_image" in tool_names, (
        f"analyze_circuit_image not found in tools: {tool_names}"
    )
