"""Tests for the pdf_ingest skill handler and registration."""

import tempfile
from pathlib import Path

import pytest

from openclaw_brain.config import load_config
from openclaw_brain.knowledge.skill_handler import register_pdf_ingest_skill, pdf_ingest_handler
from openclaw_brain.knowledge.pipeline import KnowledgePipeline
from openclaw_brain.knowledge.graph.store import GraphStore
from openclaw_brain.llm.provider import LLMProvider
from openclaw_brain.skills.registry import SkillRegistry


def test_skill_registration():
    """Test that pdf_ingest skill registers correctly in the registry."""
    config = load_config()
    registry = SkillRegistry()

    # We can't fully initialize without Neo4j, but we can test registration
    # by mocking the dependencies
    from unittest.mock import MagicMock, AsyncMock
    mock_graph = MagicMock(spec=GraphStore)
    mock_provider = MagicMock(spec=LLMProvider)

    register_pdf_ingest_skill(registry, mock_graph, config, mock_provider)

    skill = registry.get("pdf_ingest")
    assert skill is not None
    assert skill.name == "pdf_ingest"
    assert "pdf" in skill.trigger_keywords
    assert "지식" in skill.trigger_keywords
    assert "file_path" in skill.input_schema


def test_skill_has_korean_triggers():
    """Verify Korean trigger keywords for agent compatibility."""
    config = load_config()
    registry = SkillRegistry()

    from unittest.mock import MagicMock
    register_pdf_ingest_skill(
        registry, MagicMock(spec=GraphStore), config, MagicMock(spec=LLMProvider),
    )
    skill = registry.get("pdf_ingest")
    korean_triggers = [kw for kw in skill.trigger_keywords if any('\uac00' <= c <= '\ud7a3' for c in kw)]
    assert len(korean_triggers) >= 3  # 지식, 변환, 정리, 연결, 분석, etc.


@pytest.mark.asyncio
async def test_handler_missing_file_path():
    """Handler should reject calls without file_path."""
    from unittest.mock import MagicMock
    mock_pipeline = MagicMock(spec=KnowledgePipeline)
    result = await pdf_ingest_handler(mock_pipeline, {})
    assert not result["success"]
    assert "file_path is required" in result["error"]


@pytest.mark.asyncio
async def test_handler_nonexistent_file():
    """Handler should reject nonexistent files."""
    from unittest.mock import MagicMock
    mock_pipeline = MagicMock(spec=KnowledgePipeline)
    result = await pdf_ingest_handler(mock_pipeline, {"file_path": "/nonexistent/file.pdf"})
    assert not result["success"]
    assert "not found" in result["error"].lower()


@pytest.mark.asyncio
async def test_handler_non_pdf_file():
    """Handler should reject non-PDF files."""
    from unittest.mock import MagicMock
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"not a pdf")
        f.flush()
        mock_pipeline = MagicMock(spec=KnowledgePipeline)
        result = await pdf_ingest_handler(mock_pipeline, {"file_path": f.name})
    assert not result["success"]
    assert "not a pdf" in result["error"].lower()
    Path(f.name).unlink()
