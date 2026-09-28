"""2026-07-22 Rick directive: anthropic API models banned (cost). Mechanical guard tests."""
import pytest

from openclaw_brain.config import BrainConfig, ModelEntry, ModelsConfig, load_config
from openclaw_brain.llm.provider import LLMProvider, LLMProviderError


def _provider_policy_config() -> BrainConfig:
    cfg = BrainConfig()
    cfg.models = ModelsConfig(catalog=[
        ModelEntry(name="unit-anthropic", provider="anthropic", model_id="test-anthropic",
                   status="banned"),
        ModelEntry(name="unit-google", provider="google", model_id="test-google"),
    ])
    return cfg


def test_anthropic_model_blocked_by_default(monkeypatch):
    monkeypatch.delenv("OPENCLAW_ALLOW_ANTHROPIC_API", raising=False)
    prov = LLMProvider(_provider_policy_config())
    with pytest.raises(LLMProviderError, match="banned"):
        prov.get("unit-anthropic")


def test_anthropic_model_allowed_with_explicit_override(monkeypatch):
    monkeypatch.setenv("OPENCLAW_ALLOW_ANTHROPIC_API", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    prov = LLMProvider(_provider_policy_config())
    assert prov.get("unit-anthropic") is not None


def test_status_banned_does_not_weaken_or_replace_the_provider_keyed_raise(monkeypatch):
    """2026-07-25: the anthropic entries also carry status="banned". That field is DOCUMENTATION —
    the mechanism stays the provider-keyed raise, so an anthropic entry left un-labelled (a
    deployment-specific config, a hand-added entry) is still blocked, and a labelled one is still
    overridable by the same single env var. Neither is softened into a warning."""
    cfg = _provider_policy_config()
    assert cfg.models.get_model("unit-anthropic").status == "banned"

    # Label removed in memory → still raises (the ban does not depend on the label).
    cfg.models.get_model("unit-anthropic").status = "active"
    monkeypatch.delenv("OPENCLAW_ALLOW_ANTHROPIC_API", raising=False)
    with pytest.raises(LLMProviderError, match="banned"):
        LLMProvider(cfg).get("unit-anthropic")

    # A non-anthropic entry labelled banned is NOT raised on — status alone never blocks.
    cfg.models.get_model("unit-google").status = "banned"
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-real")
    assert LLMProvider(cfg).get("unit-google") is not None


def test_no_claude_in_any_fallback_chain():
    cfg = load_config()
    for chain_attr in ("fallback_extraction", "fallback_reasoning", "fallback_matching"):
        chain = getattr(cfg.resilience, chain_attr, [])
        assert not any("claude" in m for m in chain), f"{chain_attr} still has claude: {chain}"
