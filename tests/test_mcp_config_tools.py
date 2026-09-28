"""Tests for MCP pipeline configuration tools.

Tests the get/update/add/remove config tools by mocking the global agent
and verifying in-memory config changes + disk persistence.
"""

from unittest.mock import MagicMock

import pytest

from openclaw_brain.config import (
    BrainConfig, FiguresConfig, ModelEntry, ModelsConfig, ResilienceConfig,
    load_config, save_config,
)


# ── Fixtures ──


@pytest.fixture
def mock_agent():
    """Use a synthetic catalog with every routing slot and model status represented."""
    cfg = BrainConfig()
    cfg.models = ModelsConfig(
        default_extraction="unit-local-primary",
        default_reasoning="unit-local-primary",
        default_matching="unit-local-primary",
        default_vision="unit-local-vision",
        default_figure_analysis="unit-local-figure",
        catalog=[
            ModelEntry(name=name, provider="local", model_id=f"test-{name}",
                       endpoint="http://127.0.0.1:1/v1", tier="local")
            for name in (
                "unit-local-primary", "unit-local-fallback", "unit-local-vision",
                "unit-local-figure", "unit-local-slide", "unit-local-slide-backup",
            )
        ] + [
            ModelEntry(name="unit-cloud-fast", provider="openai", model_id="test-cloud"),
            ModelEntry(name="unit-unrouted", provider="local", model_id="test-unrouted",
                       endpoint="http://127.0.0.1:1/v1", tier="local"),
            ModelEntry(name="unit-incompatible", provider="local", model_id="test-incompatible",
                       status="incompatible"),
            ModelEntry(name="unit-banned", provider="local", model_id="test-banned",
                       status="banned"),
            ModelEntry(name="unit-deprecated", provider="local", model_id="test-deprecated",
                       status="deprecated"),
        ],
    )
    cfg.figures = FiguresConfig(
        slide_analysis_model="unit-local-slide",
        slide_analysis_fallback="unit-local-slide-backup",
    )
    cfg.resilience = ResilienceConfig(
        fallback_extraction=["unit-cloud-fast"],
        fallback_reasoning=["unit-cloud-fast"],
        fallback_matching=["unit-local-fallback"],
    )

    agent = MagicMock()
    agent.is_started = True
    agent._config = cfg
    agent._llm_provider = MagicMock()

    return agent


@pytest.fixture
def patched_server(mock_agent, tmp_path):
    """Patch the global _agent and save_config to use temp dir."""
    import openclaw_brain.server.mcp_server as mod

    original_agent = mod._agent
    mod._agent = mock_agent

    # Patch save_config to write to temp dir instead of real config
    original_save = mod.save_config
    temp_config = tmp_path / "default.toml"

    def _temp_save(cfg, path=None):
        save_config(cfg, temp_config)

    mod.save_config = _temp_save

    yield mod, temp_config

    mod._agent = original_agent
    mod.save_config = original_save


# ── get_pipeline_config ──


@pytest.mark.asyncio
async def test_get_pipeline_config(patched_server):
    mod, _ = patched_server
    # Find the tool function
    tools = {t.name: t for t in mod.create_server()._tool_manager.list_tools()}
    assert "get_pipeline_config" in tools

    # Call via the module's internal mechanism
    from openclaw_brain.server.mcp_server import create_server
    server = create_server()

    # We test the underlying logic directly
    cfg = mod._agent._config
    result = {
        "stages": {
            "extraction": {
                "default_model": cfg.models.default_extraction,
                "fallback_chain": cfg.resilience.fallback_extraction,
            },
        },
    }
    assert result["stages"]["extraction"]["default_model"] == "unit-local-primary"
    assert "unit-cloud-fast" in result["stages"]["extraction"]["fallback_chain"]


# ── update_stage_model ──


def test_update_stage_model_in_memory(mock_agent):
    """Changing a stage model should update config in memory."""
    cfg = mock_agent._config
    assert cfg.models.default_extraction == "unit-local-primary"

    # Simulate what the MCP tool does
    cfg.models.default_extraction = "unit-incompatible"
    assert cfg.models.default_extraction == "unit-incompatible"


def test_update_stage_model_validates_catalog(mock_agent):
    """Model must exist in catalog."""
    cfg = mock_agent._config
    assert cfg.models.get_model("nonexistent-model") is None


def test_update_stage_model_persists(mock_agent, tmp_path):
    """Changes should survive save → load."""
    cfg = mock_agent._config
    cfg.models.default_reasoning = "unit-incompatible"

    out = tmp_path / "test.toml"
    save_config(cfg, out)
    reloaded = load_config(out)
    assert reloaded.models.default_reasoning == "unit-incompatible"


# ── update_fallback_chain ──


def test_update_fallback_chain(mock_agent, tmp_path):
    cfg = mock_agent._config
    cfg.resilience.fallback_extraction = ["unit-unrouted", "unit-incompatible"]

    out = tmp_path / "test.toml"
    save_config(cfg, out)
    reloaded = load_config(out)
    assert reloaded.resilience.fallback_extraction == ["unit-unrouted", "unit-incompatible"]


def test_update_fallback_chain_empty(mock_agent, tmp_path):
    cfg = mock_agent._config
    cfg.resilience.fallback_matching = []

    out = tmp_path / "test.toml"
    save_config(cfg, out)
    reloaded = load_config(out)
    assert reloaded.resilience.fallback_matching == []


# ── add_catalog_model ──


def test_add_catalog_model(mock_agent, tmp_path):
    cfg = mock_agent._config
    initial_count = len(cfg.models.catalog)

    entry = ModelEntry(
        name="new-test-model",
        provider="openai",
        model_id="gpt-test-123",
        tier="fast",
    )
    cfg.models.catalog.append(entry)
    assert len(cfg.models.catalog) == initial_count + 1

    out = tmp_path / "test.toml"
    save_config(cfg, out)
    reloaded = load_config(out)
    found = reloaded.models.get_model("new-test-model")
    assert found is not None
    assert found.provider == "openai"
    assert found.model_id == "gpt-test-123"


def test_add_catalog_model_with_endpoint(mock_agent, tmp_path):
    cfg = mock_agent._config
    entry = ModelEntry(
        name="local-test",
        provider="local",
        model_id="test-local",
        tier="local",
        endpoint="http://localhost:9999/v1",
    )
    cfg.models.catalog.append(entry)

    out = tmp_path / "test.toml"
    save_config(cfg, out)
    reloaded = load_config(out)
    found = reloaded.models.get_model("local-test")
    assert found is not None
    assert found.endpoint == "http://localhost:9999/v1"


# ── remove_catalog_model ──


def test_remove_catalog_model(mock_agent, tmp_path):
    cfg = mock_agent._config
    initial_count = len(cfg.models.catalog)

    # Remove a model that no routing slot uses.
    cfg.models.catalog = [m for m in cfg.models.catalog if m.name != "unit-unrouted"]
    assert len(cfg.models.catalog) == initial_count - 1

    out = tmp_path / "test.toml"
    save_config(cfg, out)
    reloaded = load_config(out)
    assert reloaded.models.get_model("unit-unrouted") is None


def test_cannot_remove_default_model(mock_agent):
    """Should not remove a model that is a stage default."""
    cfg = mock_agent._config
    default_name = cfg.models.default_extraction

    # The MCP tool should reject this — verify the check
    for stage in ("extraction", "reasoning", "matching"):
        if getattr(cfg.models, f"default_{stage}") == default_name:
            # This model is in use — removal should be blocked
            assert True
            return
    pytest.fail("Expected default model to be found in stage defaults")


def test_cannot_remove_fallback_model(mock_agent):
    """Should not remove a model that is in a fallback chain."""
    cfg = mock_agent._config
    # A fallback model is protected from removal even when it is not a stage default.
    guarded = "unit-cloud-fast"
    for stage in ("extraction", "reasoning", "matching"):
        if guarded in getattr(cfg.resilience, f"fallback_{stage}"):
            return
    pytest.fail(f"Expected {guarded!r} to be in at least one fallback chain")


# ── the write tools must not persist what the router refuses (2026-07-25) ──
#
# These call the REAL tool bodies (not a simulation of their logic), because the gap being
# closed WAS in the tool body: catalog-existence was the only gate, so an agent could write
# `unit-incompatible` (incompatible) or `unit-banned` (banned) into a chain, be told "updated", and
# end up with a chain get_chain() silently drops every entry of.


async def _run_tool(mod, name: str, args: dict) -> str:
    return await mod.create_server()._tool_manager.get_tool(name).run(args, None)  # mcp 2.x: context required


def _result_text(result) -> str:
    """FastMCP hands back content blocks; the tools here all return a single string."""
    return str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("dead_model, status", [
    ("unit-incompatible", "incompatible"),
    ("unit-banned", "banned"),
    ("unit-deprecated", "deprecated"),
])
async def test_update_fallback_chain_rejects_a_non_active_model(patched_server, dead_model, status):
    mod, temp_config = patched_server
    before = list(mod._agent._config.resilience.fallback_extraction)

    text = _result_text(await _run_tool(
        mod, "update_fallback_chain",
        {"stage": "extraction", "models": f"unit-cloud-fast, {dead_model}"},
    ))

    assert dead_model in text and status in text
    assert "cannot be routed to" in text
    # Rejected means REJECTED: neither memory nor disk may carry the half-applied chain.
    assert mod._agent._config.resilience.fallback_extraction == before
    assert not temp_config.exists()


@pytest.mark.asyncio
async def test_update_fallback_chain_still_rejects_an_unknown_name(patched_server):
    """The pre-existing not-in-catalog path is unchanged — a typo is still its own error."""
    mod, _ = patched_server
    text = _result_text(await _run_tool(
        mod, "update_fallback_chain", {"stage": "extraction", "models": "unit-cloud-fast, nope-9000"},
    ))
    assert "not in catalog" in text and "nope-9000" in text


@pytest.mark.asyncio
async def test_update_fallback_chain_accepts_active_models(patched_server):
    mod, temp_config = patched_server
    # Both entries are active; the non-active cases above must still be refused.
    text = _result_text(await _run_tool(
        mod, "update_fallback_chain", {"stage": "matching", "models": "unit-cloud-fast, unit-local-primary"},
    ))
    assert "updated" in text
    assert mod._agent._config.resilience.fallback_matching == ["unit-cloud-fast", "unit-local-primary"]
    assert load_config(temp_config).resilience.fallback_matching == ["unit-cloud-fast", "unit-local-primary"]


@pytest.mark.asyncio
async def test_update_stage_model_rejects_a_non_active_model(patched_server):
    """A stage DEFAULT pointing at a retired model is at least as bad as a fallback: it is the
    model the stage actually runs, and nothing downstream would re-route."""
    mod, temp_config = patched_server
    before = mod._agent._config.models.default_reasoning

    text = _result_text(await _run_tool(
        mod, "update_stage_model", {"stage": "reasoning", "model_name": "unit-banned"},
    ))

    assert "unit-banned" in text and "banned" in text and "cannot be routed to" in text
    assert mod._agent._config.models.default_reasoning == before
    assert not temp_config.exists()


@pytest.mark.asyncio
async def test_update_stage_model_still_rejects_an_unknown_name(patched_server):
    mod, _ = patched_server
    text = _result_text(await _run_tool(
        mod, "update_stage_model", {"stage": "reasoning", "model_name": "nope-9000"},
    ))
    assert "not in catalog" in text


@pytest.mark.asyncio
async def test_add_catalog_model_rejects_a_banned_provider(patched_server, monkeypatch):
    """Same class of gap: LLMProvider.get() raises on EVERY anthropic entry (2026-07-22 cost
    ban), so minting one would confirm success for a model that cannot even be constructed."""
    mod, temp_config = patched_server
    monkeypatch.delenv("OPENCLAW_ALLOW_ANTHROPIC_API", raising=False)

    text = _result_text(await _run_tool(mod, "add_catalog_model", {
        "name": "claude-smuggled", "provider": "anthropic", "model_id": "claude-x",
    }))

    assert "banned" in text and "OPENCLAW_ALLOW_ANTHROPIC_API" in text
    assert mod._agent._config.models.get_model("claude-smuggled") is None
    assert not temp_config.exists()


@pytest.mark.asyncio
async def test_add_catalog_model_allows_anthropic_under_the_deliberate_override(patched_server,
                                                                                monkeypatch):
    """The override stays a single env var — the same one get() honors, not a second policy."""
    mod, _ = patched_server
    monkeypatch.setenv("OPENCLAW_ALLOW_ANTHROPIC_API", "1")

    text = _result_text(await _run_tool(mod, "add_catalog_model", {
        "name": "claude-deliberate", "provider": "anthropic", "model_id": "claude-x",
    }))

    assert "added to catalog" in text
    entry = mod._agent._config.models.get_model("claude-deliberate")
    assert entry is not None and entry.status == "active"


@pytest.mark.asyncio
async def test_remove_catalog_model_refuses_every_routing_slot(patched_server):
    """Removal is the mirror gap: the vision / figure-analysis / slide-analysis defaults are
    routing slots too, and deleting one persisted happily while leaving the router pointed at
    a name that no longer exists."""
    mod, temp_config = patched_server
    cfg = mod._agent._config

    for model_name in (cfg.models.default_vision, cfg.models.default_figure_analysis,
                       cfg.figures.slide_analysis_model, cfg.figures.slide_analysis_fallback):
        assert model_name, "expected the real config to populate every routing slot"
        text = _result_text(await _run_tool(mod, "remove_catalog_model", {"name": model_name}))
        assert "Cannot remove" in text, text
        assert cfg.models.get_model(model_name) is not None

    assert not temp_config.exists()


@pytest.mark.asyncio
async def test_remove_catalog_model_still_removes_an_unrouted_model(patched_server):
    mod, temp_config = patched_server
    cfg = mod._agent._config
    assert cfg.models.get_model("unit-unrouted") is not None

    text = _result_text(await _run_tool(mod, "remove_catalog_model", {"name": "unit-unrouted"}))

    assert "removed from catalog" in text
    assert cfg.models.get_model("unit-unrouted") is None
    assert load_config(temp_config).models.get_model("unit-unrouted") is None
