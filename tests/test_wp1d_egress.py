"""WP1d startup order regressions; all external boundaries are replaced by fakes."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import types
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

from openclaw_brain import cli
from openclaw_brain.config import (
    BrainConfig, ConfigError, DeploymentConfig, FiguresConfig, ModelEntry,
    ModelsConfig, ResilienceConfig, save_config,
)
from openclaw_brain.server import mcp_server


@pytest.fixture(autouse=True)
def _offline_import_state(monkeypatch):
    # Other suite modules deliberately import these libraries in online mode.
    # These startup-order tests begin at the fresh-process offline boundary.
    for module_name, constant in (("huggingface_hub.constants", "HF_HUB_OFFLINE"),
                                  ("transformers.utils.hub", "_is_offline_mode")):
        module = sys.modules.get(module_name)
        if module is not None:
            monkeypatch.setattr(module, constant, True)
    for name in ("OPENCLAW_EGRESS", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(name, raising=False)


def _local_config() -> BrainConfig:
    config = BrainConfig(deployment=DeploymentConfig(egress="local-only"))
    config.models = ModelsConfig(
        default_extraction="local", default_reasoning="local",
        default_matching="local", default_vision="local",
        default_figure_analysis="local",
        catalog=[ModelEntry(name="local", provider="local", model_id="local-model",
                            endpoint="http://127.0.0.1:8000/v1", tier="local")],
    )
    config.figures = FiguresConfig(slide_analysis_model="local", slide_analysis_fallback="local")
    config.resilience = ResilienceConfig()
    config.embedding.model = "active-config-only-model"
    return config


@pytest.mark.parametrize("valid", [True, False])
def test_n1_doctor_full_preflights_active_config_before_loader(monkeypatch, valid):
    from openclaw_brain.knowledge import embedding

    cfg = _local_config()
    if not valid:
        cfg.neo4j.uri = "neo4j://remote.invalid:7687"
    seen = []

    def constructor(model, **kwargs):
        seen.append((model, kwargs))
        raise RuntimeError("model loading stopped at fake constructor")

    monkeypatch.setattr(cli, "load_config", lambda path=None: cfg)
    monkeypatch.setattr(cli, "_doctor_checks", lambda full=False: [])
    monkeypatch.setattr(embedding, "_model", None)
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=constructor))
    for name in ("OPENCLAW_EGRESS", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(name, raising=False)
    result = CliRunner().invoke(cli.main, ["--config", "/active/config.toml", "doctor", "--full"])
    assert result.exit_code == 1
    assert "FAIL" in result.output
    if valid:
        assert len(seen) == 1
        assert seen[0][0] == "active-config-only-model"
        assert seen[0][1]["local_files_only"] is True
    else:
        assert seen == []
        assert "blocked by egress preflight" in result.output


@pytest.mark.asyncio
@pytest.mark.parametrize("valid", [True, False])
async def test_n2_direct_mcp_startup_preflights_before_injection(monkeypatch, valid):
    from openclaw_brain.egress import enforce_startup_egress

    cfg = _local_config()
    if not valid:
        cfg.neo4j.uri = "neo4j://remote.invalid:7687"
    events = []
    agent = AsyncMock()
    agent.is_started = False
    monkeypatch.setattr(mcp_server, "load_config", lambda path=None: events.append("load") or cfg)
    def preflight(config):
        events.append("preflight")
        enforce_startup_egress(config)
    monkeypatch.setattr(mcp_server, "enforce_startup_egress", preflight)
    monkeypatch.setattr(mcp_server, "inject_api_keys", lambda: events.append("inject") or {})
    monkeypatch.setattr(mcp_server, "BrainAgent", lambda config: events.append("agent") or agent)
    monkeypatch.setattr(mcp_server, "_agent", None)
    server = mcp_server.create_server()
    startup = server._tool_manager.get_tool("startup").fn
    try:
        if valid:
            assert "started successfully" in await startup()
            assert events == ["load", "preflight", "agent"]
        else:
            with pytest.raises(ConfigError):
                await startup()
            assert events == ["load", "preflight"]
    finally:
        mcp_server._agent = None


def test_n2_cli_serve_preflights_before_mcp_startup(monkeypatch):
    from openclaw_brain import egress

    cfg = _local_config()
    events = []
    monkeypatch.setattr(cli, "load_config", lambda path=None: events.append("cli load") or cfg)
    monkeypatch.setattr(mcp_server, "load_config", lambda path=None: events.append("mcp load") or cfg)
    original = egress.enforce_startup_egress
    def cli_preflight(config):
        events.append("cli preflight")
        original(config)
    def mcp_preflight(config):
        events.append("mcp preflight")
        original(config)
    monkeypatch.setattr(egress, "enforce_startup_egress", cli_preflight)
    monkeypatch.setattr(mcp_server, "enforce_startup_egress", mcp_preflight)
    monkeypatch.setattr(mcp_server, "inject_api_keys", lambda: events.append("inject") or {})
    agent = AsyncMock()
    agent.is_started = False
    monkeypatch.setattr(mcp_server, "BrainAgent", lambda config: events.append("agent") or agent)
    monkeypatch.setattr(mcp_server, "_agent", None)

    def fake_run(server, **kwargs):
        events.append("run")
        asyncio.run(server._tool_manager.get_tool("startup").fn())

    monkeypatch.setattr(mcp_server.MCPServer, "run", fake_run)
    try:
        result = CliRunner().invoke(cli.main, ["--config", "/active/config.toml", "serve"])
        assert result.exit_code == 0, result.output
        assert events == ["cli load", "cli preflight", "run", "mcp load", "mcp preflight", "agent"]
    finally:
        mcp_server._agent = None


def test_n2_cli_serve_rejects_before_server_or_injection(monkeypatch):
    cfg = _local_config()
    cfg.neo4j.uri = "neo4j://remote.invalid:7687"
    events = []
    monkeypatch.setattr(cli, "load_config", lambda path=None: cfg)
    monkeypatch.setattr(mcp_server, "inject_api_keys", lambda: events.append("inject") or {})
    monkeypatch.setattr(mcp_server.MCPServer, "run", lambda *a, **k: events.append("run"))
    result = CliRunner().invoke(cli.main, ["serve"])
    assert result.exit_code != 0
    assert events == []


def test_n3_whitespace_true_values_arm_real_library_constants_in_fresh_process(tmp_path):
    path = tmp_path / "local.toml"
    save_config(_local_config(), path)
    script = """
import os, sys
from openclaw_brain.config import load_config
from openclaw_brain.egress import enforce_startup_egress
enforce_startup_egress(load_config(sys.argv[1]))
import huggingface_hub.constants as hub
from transformers.utils import hub as transformers_hub
assert os.environ['HF_HUB_OFFLINE'] == '1'
assert os.environ['TRANSFORMERS_OFFLINE'] == '1'
assert hub.HF_HUB_OFFLINE is True
assert transformers_hub._is_offline_mode is True
"""
    env = os.environ.copy()
    env.pop("OPENCLAW_EGRESS", None)
    env["HF_HUB_OFFLINE"] = " true "
    env["TRANSFORMERS_OFFLINE"] = " true "
    result = subprocess.run([sys.executable, "-c", script, str(path)], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
