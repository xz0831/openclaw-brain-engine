"""Tests for LLM provider — model catalog and runtime selection."""

import logging

import pytest

from openclaw_brain.config import BrainConfig, ModelEntry, ModelsConfig, load_config
from openclaw_brain.llm import provider as provider_module
from openclaw_brain.llm.provider import LLMProvider, LLMProviderError

@pytest.fixture(autouse=True)
def _allow_anthropic_instantiation(monkeypatch):
    """These tests exercise FACTORY mechanics with anthropic catalog fixtures — instantiation
    only, zero API calls, zero billing. The 2026-07-22 anthropic-API ban guard is therefore
    bypassed here; the ban itself is covered by tests/test_provider_anthropic_ban.py."""
    monkeypatch.setenv("OPENCLAW_ALLOW_ANTHROPIC_API", "1")



@pytest.fixture(autouse=True)
def _reset_missing_fallback_warnings():
    """W-D2 defect 8's seen-set is module-level (warn once per process, not once per get_chain()
    call) — isolate every test from whatever other tests warned about first. The two
    catalog-staleness seen-sets (2026-07-25) follow the same pattern and need the same isolation."""
    for seen in (provider_module._warned_missing_fallbacks,
                 provider_module._warned_nonactive_fallbacks,
                 provider_module._warned_nonactive_models):
        seen.clear()
    yield
    for seen in (provider_module._warned_missing_fallbacks,
                 provider_module._warned_nonactive_fallbacks,
                 provider_module._warned_nonactive_models):
        seen.clear()


def _make_config() -> BrainConfig:
    """Create a config with test catalog entries."""
    cfg = BrainConfig()
    cfg.models = ModelsConfig(
        default_extraction="test-sonnet",
        default_reasoning="test-opus",
        default_matching="test-sonnet",
        catalog=[
            ModelEntry(
                name="test-opus",
                provider="anthropic",
                model_id="claude-opus-4-6-20250514",
                tier="frontier",
            ),
            ModelEntry(
                name="test-sonnet",
                provider="anthropic",
                model_id="claude-sonnet-4-6-20250514",
                tier="frontier",
            ),
            ModelEntry(
                name="test-local",
                provider="local",
                model_id="qwen3-vl-8b-instruct",
                endpoint="http://localhost:11434/v1",
                tier="local",
            ),
            ModelEntry(
                name="test-openai",
                provider="openai",
                model_id="gpt-5.4",
                tier="frontier",
            ),
        ],
    )
    return cfg


def test_provider_get_anthropic():
    provider = LLMProvider(_make_config())
    llm = provider.get("test-sonnet")
    assert llm is not None
    assert llm.model == "claude-sonnet-4-6-20250514"


def test_provider_get_local():
    provider = LLMProvider(_make_config())
    llm = provider.get("test-local")
    assert llm is not None


def test_provider_get_openai():
    provider = LLMProvider(_make_config())
    llm = provider.get("test-openai")
    assert llm is not None


def test_provider_get_not_found():
    provider = LLMProvider(_make_config())
    with pytest.raises(LLMProviderError, match="not found in catalog"):
        provider.get("nonexistent-model")


def test_provider_get_for_stage_default():
    provider = LLMProvider(_make_config())
    # extraction default is test-sonnet
    llm = provider.get_for_stage("extraction")
    assert llm.model == "claude-sonnet-4-6-20250514"


def test_provider_get_for_stage_override():
    provider = LLMProvider(_make_config())
    # Override extraction with opus
    llm = provider.get_for_stage("extraction", override="test-opus")
    assert llm.model == "claude-opus-4-6-20250514"


def test_provider_get_for_stage_unknown():
    provider = LLMProvider(_make_config())
    with pytest.raises(LLMProviderError, match="Unknown stage"):
        provider.get_for_stage("nonexistent_stage")


def test_provider_caching():
    provider = LLMProvider(_make_config())
    llm1 = provider.get("test-sonnet")
    llm2 = provider.get("test-sonnet")
    assert llm1 is llm2  # Same instance (cached)


def test_provider_kwargs_different_cache():
    provider = LLMProvider(_make_config())
    llm1 = provider.get("test-sonnet", temperature=0.0)
    llm2 = provider.get("test-sonnet", temperature=1.0)
    assert llm1 is not llm2  # Different kwargs → different cache entry


def test_provider_clear_cache():
    provider = LLMProvider(_make_config())
    llm1 = provider.get("test-sonnet")
    provider.clear_cache()
    llm2 = provider.get("test-sonnet")
    assert llm1 is not llm2  # Cache cleared → new instance


def test_provider_list_available():
    provider = LLMProvider(_make_config())
    models = provider.list_available()
    assert len(models) == 4
    names = [m["name"] for m in models]
    assert "test-opus" in names
    assert "test-local" in names


def test_list_available_carries_status_to_the_agent_facing_channel():
    """list_available() IS the agent-facing view of the catalog (BrainAgent.get_stats()["models"]
    → the `brain://models` MCP resource). Without `status` there, an outside agent cannot tell a
    retired entry from a live one — reading the catalog as 4 equal options is exactly the
    folklore this field was added to end. `status_reason` rides along only when it is set."""
    cfg = _make_config()
    cfg.models.get_model("test-openai").status = "incompatible"
    cfg.models.get_model("test-openai").status_reason = "Codex OAuth speaks chatgpt.com"

    rows = {m["name"]: m for m in LLMProvider(cfg).list_available()}

    assert set(rows["test-local"]) == {"name", "provider", "tier", "model_id", "status"}
    assert rows["test-local"]["status"] == "active"
    assert rows["test-openai"]["status"] == "incompatible"
    assert rows["test-openai"]["status_reason"] == "Codex OAuth speaks chatgpt.com"


@pytest.mark.asyncio
async def test_get_stats_models_expose_status():
    """One step further down the REAL channel: what `brain://models` serializes is
    `get_stats()["models"]`, so the status has to survive that hop, not just exist on the
    provider's return value in isolation."""
    from unittest.mock import AsyncMock, MagicMock

    from openclaw_brain.agent import BrainAgent

    cfg = _make_config()
    cfg.models.get_model("test-openai").status = "incompatible"

    agent = BrainAgent.__new__(BrainAgent)      # no Neo4j: only the stats view is under test
    agent._started = True
    agent._graph = MagicMock(get_stats=AsyncMock(return_value={}))
    agent._memory_store = MagicMock(get_stats=AsyncMock(return_value={}))
    agent._reinforcement = MagicMock(get_reinforcement_stats=AsyncMock(return_value={}))
    agent._registry = MagicMock(count=0, list_skills=MagicMock(return_value=[]))
    agent._llm_provider = LLMProvider(cfg)

    stats = await agent.get_stats()
    by_name = {m["name"]: m for m in stats["models"]}

    assert by_name["test-openai"]["status"] == "incompatible"
    assert by_name["test-local"]["status"] == "active"


def test_provider_unknown_provider():
    cfg = BrainConfig()
    cfg.models = ModelsConfig(
        catalog=[
            ModelEntry(name="bad", provider="unknown_provider", model_id="x"),
        ],
    )
    provider = LLMProvider(cfg)
    with pytest.raises(LLMProviderError, match="Unknown provider"):
        provider.get("bad")


def test_get_chain_returns_primary_only_when_no_fallbacks_configured():
    provider = LLMProvider(_make_config())
    chain = provider.get_chain("extraction")
    assert len(chain) == 1
    assert chain[0].model == "claude-sonnet-4-6-20250514"


def test_get_chain_includes_available_fallbacks():
    cfg = _make_config()
    cfg.resilience.fallback_extraction = ["test-opus", "test-local"]
    provider = LLMProvider(cfg)

    chain = provider.get_chain("extraction")

    assert len(chain) == 3   # primary (test-sonnet) + both fallbacks


def test_get_chain_override_returns_single_model():
    provider = LLMProvider(_make_config())
    chain = provider.get_chain("extraction", override="test-opus")
    assert len(chain) == 1
    assert chain[0].model == "claude-opus-4-6-20250514"


def test_get_chain_warns_once_for_missing_fallback_model(caplog):
    """W-D2 defect 8: get_chain() used to silently drop a configured fallback model missing from
    the catalog (bare `except LLMProviderError: pass`) — a typo'd/renamed fallback name shortened
    the chain forever with zero signal. Fixed: logs a WARNING naming the model + stage, once."""
    cfg = _make_config()
    cfg.resilience.fallback_extraction = ["nonexistent-fallback", "test-opus"]
    provider = LLMProvider(cfg)

    with caplog.at_level(logging.WARNING, logger="openclaw_brain.llm.provider"):
        chain = provider.get_chain("extraction")

    # The chain is still silently SHORTER than configured (unchanged, documented behavior) —
    # what's new is that the gap is now visible.
    assert len(chain) == 2   # primary + test-opus; nonexistent-fallback skipped
    warnings = [r for r in caplog.records if "nonexistent-fallback" in r.message]
    assert len(warnings) == 1
    assert "extraction" in warnings[0].message


def test_get_chain_does_not_warn_again_for_the_same_stage_and_model(caplog):
    cfg = _make_config()
    cfg.resilience.fallback_extraction = ["nonexistent-fallback"]
    provider = LLMProvider(cfg)

    with caplog.at_level(logging.WARNING, logger="openclaw_brain.llm.provider"):
        provider.get_chain("extraction")
        caplog.clear()
        provider.get_chain("extraction")   # same stage+model gap, called again

    assert not any("nonexistent-fallback" in r.message for r in caplog.records)


def test_get_chain_warns_separately_per_stage(caplog):
    """Same missing model name in two DIFFERENT stages' fallback chains must warn for each stage —
    the seen-set key is (stage, name), not just name."""
    cfg = _make_config()
    cfg.resilience.fallback_extraction = ["nonexistent-fallback"]
    cfg.resilience.fallback_reasoning = ["nonexistent-fallback"]
    provider = LLMProvider(cfg)

    with caplog.at_level(logging.WARNING, logger="openclaw_brain.llm.provider"):
        provider.get_chain("extraction")
        provider.get_chain("reasoning")

    stages_warned = {
        "extraction" if "extraction" in r.message else "reasoning"
        for r in caplog.records if "nonexistent-fallback" in r.message
    }
    assert stages_warned == {"extraction", "reasoning"}


# ── catalog staleness: ModelEntry.status (2026-07-25) ──
#
# The catalog accretes and nothing expired, so a model retired by an evaluation (grok-4.20 after
# the figure bake-off) sat in it indistinguishable from a live one. `status` labels that; these
# tests pin what the label DOES: it makes a fallback chain refuse to fall onto a retired model
# (loudly), and makes an explicit request for one say so (once) — and nothing else. It is not a
# router: no test here may show a status change altering which model a stage picks.


def _retired(name: str = "test-retired", status: str = "deprecated") -> ModelEntry:
    """A local catalog entry mirroring the real oMLX shape (endpoint + bare model_id)."""
    return ModelEntry(
        name=name,
        provider="local",
        model_id="Qwen3.5-Retired-4bit",
        endpoint="http://localhost:8000/v1",
        tier="local",
        status=status,
    )


def test_get_chain_skips_deprecated_fallback_and_warns(caplog):
    cfg = _make_config()
    cfg.models.catalog.append(_retired())
    cfg.resilience.fallback_extraction = ["test-retired", "test-opus"]
    provider = LLMProvider(cfg)

    with caplog.at_level(logging.WARNING, logger="openclaw_brain.llm.provider"):
        chain = provider.get_chain("extraction")

    assert len(chain) == 2   # primary (test-sonnet) + test-opus; the deprecated one is skipped
    warnings = [r for r in caplog.records if "test-retired" in r.message]
    assert len(warnings) == 1
    assert "deprecated" in warnings[0].message
    assert "shorter than configured" in warnings[0].message   # same visible-gap language as a typo


def test_get_chain_skips_banned_and_incompatible_fallbacks_too(caplog):
    """Any non-active status is disqualified from a chain — the pipeline falls onto these
    unattended, so 'still constructible' is not the bar."""
    cfg = _make_config()
    cfg.models.catalog.append(_retired("test-banned", status="banned"))
    cfg.models.catalog.append(_retired("test-incompatible", status="incompatible"))
    cfg.resilience.fallback_extraction = ["test-banned", "test-incompatible", "test-opus"]
    provider = LLMProvider(cfg)

    with caplog.at_level(logging.WARNING, logger="openclaw_brain.llm.provider"):
        chain = provider.get_chain("extraction")

    assert len(chain) == 2
    messages = [r.message for r in caplog.records]
    assert sum("status='banned'" in m for m in messages) == 1
    assert sum("status='incompatible'" in m for m in messages) == 1


def test_get_chain_warns_once_per_stage_and_name_for_a_deprecated_fallback(caplog):
    cfg = _make_config()
    cfg.models.catalog.append(_retired())
    cfg.resilience.fallback_extraction = ["test-retired"]
    cfg.resilience.fallback_reasoning = ["test-retired"]
    provider = LLMProvider(cfg)

    with caplog.at_level(logging.WARNING, logger="openclaw_brain.llm.provider"):
        provider.get_chain("extraction")
        provider.get_chain("extraction")   # same (stage, name) gap again — must stay quiet
        provider.get_chain("reasoning")    # different stage = a different gap = its own warning

    hits = [r.message for r in caplog.records if "test-retired" in r.message]
    assert len(hits) == 2
    assert sum("extraction" in m for m in hits) == 1
    assert sum("reasoning" in m for m in hits) == 1


def test_get_returns_a_deprecated_model_and_warns_once(caplog):
    """Direct get(name) still CONSTRUCTS a retired model — code that names a model means it, and
    the guard must not become a second silent router — but the staleness is stated once."""
    cfg = _make_config()
    cfg.models.catalog.append(_retired())
    provider = LLMProvider(cfg)

    with caplog.at_level(logging.WARNING, logger="openclaw_brain.llm.provider"):
        first = provider.get("test-retired")
        provider.clear_cache()
        second = provider.get("test-retired")   # cache cleared, but the warning is per-process

    assert first is not None and second is not None
    assert first.model == "Qwen3.5-Retired-4bit"
    hits = [r for r in caplog.records if "test-retired" in r.message]
    assert len(hits) == 1
    assert "deprecated" in hits[0].message


def test_get_names_the_status_reason_when_the_entry_carries_one(caplog):
    cfg = _make_config()
    entry = _retired()
    entry.status_reason = "lost the 2026-06-20 figure bake-off"
    cfg.models.catalog.append(entry)
    provider = LLMProvider(cfg)

    with caplog.at_level(logging.WARNING, logger="openclaw_brain.llm.provider"):
        provider.get("test-retired")

    assert any("2026-06-20 figure bake-off" in r.message for r in caplog.records)


def test_get_does_not_warn_for_an_active_model(caplog):
    provider = LLMProvider(_make_config())
    with caplog.at_level(logging.WARNING, logger="openclaw_brain.llm.provider"):
        provider.get("test-local")
    assert [r.message for r in caplog.records] == []


def test_status_does_not_change_which_model_a_stage_resolves(caplog):
    """The one thing this feature must NOT do: alter routing. A deprecated stage default still
    resolves to that model (changing it is an evaluated decision, not a config-status side effect)."""
    cfg = _make_config()
    cfg.models.catalog.append(_retired())
    cfg.models.default_extraction = "test-retired"
    provider = LLMProvider(cfg)

    with caplog.at_level(logging.WARNING, logger="openclaw_brain.llm.provider"):
        llm = provider.get_for_stage("extraction")

    assert llm.model == "Qwen3.5-Retired-4bit"
    assert any("test-retired" in r.message for r in caplog.records)


def test_provider_with_synthetic_catalog():
    """Catalog listing, named selection, and stage selection share one config."""
    cfg = _make_config()
    provider = LLMProvider(cfg)

    available = provider.list_available()
    assert len(available) == len(cfg.models.catalog)

    llm = provider.get("test-sonnet")
    assert llm.model == cfg.models.get_model("test-sonnet").model_id

    assert provider.get_for_stage("extraction") is llm
