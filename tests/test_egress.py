"""WP1 local-only egress policy tests (offline: no model or network calls)."""

from __future__ import annotations

import logging
import os
import socket
import sys
import types
from pathlib import Path

import httpx
import pytest

from openclaw_brain.agent import BrainAgent
from openclaw_brain.auth import _do_token_refresh
from openclaw_brain.cli import _served_model_ids
from openclaw_brain.config import (
    BrainConfig,
    ConfigError,
    DeploymentConfig,
    FiguresConfig,
    ModelEntry,
    ModelsConfig,
    ResilienceConfig,
    StorageConfig,
    load_config,
    save_config,
)
from openclaw_brain.egress import (
    EgressPolicyError,
    check_url,
    enforce_startup_egress,
    is_local_url,
    validate_egress,
    effective_egress,
)
from openclaw_brain.llm import provider as provider_module
from openclaw_brain.llm.provider import LLMProvider, LLMProviderError
from openclaw_brain.llm.resilience import FallbackExhaustedError, invoke_with_resilience


@pytest.fixture(autouse=True)
def _isolated_egress_environment(monkeypatch):
    for name in (
        "OPENCLAW_EGRESS",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_TRACING",
        "LANGSMITH_TRACING",
        "LANGSMITH_TRACING_V2",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ):
        monkeypatch.delenv(name, raising=False)
    # Other test modules may have imported these libraries before this fixture runs. Keep
    # unrelated unit cases in the offline-import state; the explicit F3 test overrides it.
    hub = sys.modules.get("huggingface_hub.constants")
    if hub is not None:
        monkeypatch.setattr(hub, "HF_HUB_OFFLINE", True)
    transformers = sys.modules.get("transformers.utils.hub")
    if transformers is not None:
        monkeypatch.setattr(transformers, "_is_offline_mode", True)


@pytest.mark.parametrize("url", [
    "http://localhost:8000/v1",
    "https://LOCALHOST/path",
    "http://127.0.0.1:8000/v1",
    "http://127.255.255.254/path",
    "http://[::1]:8000/v1",
])
def test_loopback_url_classification_accepts_only_this_machine(url):
    assert is_local_url(url)


@pytest.mark.parametrize("url", [
    "http://127.0.0.1@evil.com",
    "http://localhost.evil.com",
    "http://192.168.1.10:8000/v1",
    "http://brain.local:8000/v1",
    "ftp://localhost/model",
    "http://[::ffff:127.0.0.1]/v1",
    "not-a-url",
])
def test_loopback_url_classification_rejects_spoofs_lan_and_other_schemes(url):
    assert not is_local_url(url)


def _local_only_config(tmp_path: Path | None = None) -> BrainConfig:
    cfg = BrainConfig(deployment=DeploymentConfig(egress="local-only"))
    cfg.models = ModelsConfig(
        default_extraction="local",
        default_reasoning="local",
        default_matching="local",
        default_vision="local",
        default_figure_analysis="local",
        catalog=[
            ModelEntry(
                name="local", provider="local", model_id="local-model",
                endpoint="http://localhost:8000/v1", tier="local",
            ),
        ],
    )
    cfg.figures = FiguresConfig(
        slide_analysis_model="local",
        slide_analysis_fallback="local",
    )
    cfg.resilience = ResilienceConfig()
    if tmp_path is not None:
        cfg.storage = StorageConfig(
            workspace=str(tmp_path / "workspace"),
            state_dir=str(tmp_path / "state"),
        )
    return cfg


def _cloud_route_config(tmp_path: Path | None = None) -> BrainConfig:
    """Synthetic mixed routing graph for startup egress validation."""
    cfg = _local_only_config(tmp_path)
    cfg.models.catalog.extend([
        ModelEntry(name="cloud-a", provider="openai", model_id="test-cloud-a"),
        ModelEntry(name="cloud-b", provider="google", model_id="test-cloud-b"),
    ])
    cfg.models.default_matching = "cloud-a"
    cfg.models.default_figure_analysis = "cloud-b"
    cfg.figures.slide_analysis_fallback = "cloud-a"
    cfg.resilience.fallback_extraction = ["cloud-a", "cloud-b"]
    cfg.resilience.fallback_reasoning = ["cloud-b"]
    return cfg


def test_deployment_config_env_override_and_save_roundtrip(monkeypatch, tmp_path):
    path = tmp_path / "config.toml"
    cfg = _local_only_config(tmp_path)
    save_config(cfg, path)
    assert load_config(path).deployment.egress == "local-only"

    monkeypatch.setenv("OPENCLAW_EGRESS", "any")
    assert load_config(path).deployment.egress == "any"


@pytest.mark.parametrize("value", ["cloud", "", "local", "local_only"])
def test_invalid_egress_values_fail_closed(monkeypatch, tmp_path, value):
    path = tmp_path / "config.toml"
    save_config(BrainConfig(), path)
    monkeypatch.setenv("OPENCLAW_EGRESS", value)
    with pytest.raises(ConfigError, match="OPENCLAW_EGRESS"):
        load_config(path)


def test_invalid_toml_egress_value_fails_closed():
    with pytest.raises(ConfigError, match="deployment"):
        DeploymentConfig(egress="offsite")


@pytest.mark.parametrize("provider", ["openrouter", "google", "openai", "anthropic"])
def test_get_blocks_cloud_before_client_creation(monkeypatch, provider, tmp_path):
    cfg = _local_only_config(tmp_path)
    cfg.models.catalog.append(ModelEntry(name="cloud", provider=provider, model_id="remote"))
    created = []
    monkeypatch.setattr(
        provider_module,
        "_create_chat_model",
        lambda *args, **kwargs: created.append(args) or object(),
    )

    with pytest.raises(LLMProviderError) as exc_info:
        LLMProvider(cfg).get("cloud")

    message = str(exc_info.value)
    assert "model='cloud'" in message
    assert f"provider='{provider}'" in message
    assert "host=" in message and "policy='local-only'" in message
    assert created == []


def test_get_blocks_nonloopback_local_and_allows_default_loopback(monkeypatch, tmp_path):
    cfg = _local_only_config(tmp_path)
    cfg.models.catalog.extend([
        ModelEntry(
            name="lan-local", provider="local", model_id="lan",
            endpoint="http://192.168.1.20:8000/v1",
        ),
        ModelEntry(name="implicit-local", provider="local", model_id="implicit", endpoint=""),
    ])
    sentinel = object()
    monkeypatch.setattr(provider_module, "_create_chat_model", lambda *a, **k: sentinel)
    provider = LLMProvider(cfg)

    with pytest.raises(LLMProviderError, match="192.168.1.20"):
        provider.get("lan-local")
    assert provider.get("implicit-local") is sentinel


def test_get_chain_raises_for_cloud_fallback_instead_of_dropping_it(monkeypatch, tmp_path):
    cfg = _local_only_config(tmp_path)
    cfg.models.catalog.append(
        ModelEntry(name="cloud", provider="openrouter", model_id="remote")
    )
    cfg.resilience.fallback_extraction = ["cloud"]
    monkeypatch.setattr(provider_module, "_create_chat_model", lambda *a, **k: object())

    with pytest.raises(LLMProviderError, match="cloud"):
        LLMProvider(cfg).get_chain("extraction")


def test_cached_cloud_client_is_blocked_after_runtime_policy_flip(monkeypatch, tmp_path):
    cfg = _local_only_config(tmp_path)
    cfg.deployment.egress = "any"
    cfg.models.catalog.append(
        ModelEntry(name="cloud", provider="openrouter", model_id="remote")
    )
    sentinel = object()
    monkeypatch.setattr(provider_module, "_create_chat_model", lambda *a, **k: sentinel)
    provider = LLMProvider(cfg)
    assert provider.get("cloud") is sentinel

    cfg.deployment.egress = "local-only"
    with pytest.raises(LLMProviderError, match="local-only"):
        provider.get("cloud")


def test_validate_default_shaped_config_lists_every_cloud_reference():
    cfg = _cloud_route_config()

    violations = validate_egress(cfg)
    references = {v.reference for v in violations}

    assert "[models].default_matching" in references
    assert "[models].default_figure_analysis" in references
    assert "[figures].slide_analysis_fallback" in references
    assert {
        "[resilience].fallback_extraction[0]",
        "[resilience].fallback_extraction[1]",
        "[resilience].fallback_reasoning[0]",
    } <= references
    assert all(v.provider != "local" for v in violations)


@pytest.mark.asyncio
async def test_local_only_agent_start_fails_before_graph_connect(tmp_path, monkeypatch):
    cfg = _cloud_route_config(tmp_path)
    connected = False

    async def _unexpected_connect(self):
        nonlocal connected
        connected = True

    auth_touched = False

    def _unexpected_auth():
        nonlocal auth_touched
        auth_touched = True

    monkeypatch.setattr(
        "openclaw_brain.knowledge.graph.store.GraphStore.connect", _unexpected_connect
    )
    monkeypatch.setattr("openclaw_brain.agent.inject_api_keys", _unexpected_auth)

    with pytest.raises(ConfigError) as exc_info:
        await BrainAgent(cfg).start()

    assert not connected
    assert not auth_touched
    assert "configured model violations" in str(exc_info.value)
    assert "fallback_extraction[0]" in str(exc_info.value)


@pytest.mark.asyncio
async def test_local_401_exhausts_without_constructing_cloud_model(monkeypatch, tmp_path):
    class Unauthorized(Exception):
        status_code = 401

    class Local401:
        model_name = "local"

        async def ainvoke(self, messages, **kwargs):
            raise Unauthorized("local endpoint rejected bearer token")

    cfg = _local_only_config(tmp_path)
    cfg.models.catalog.append(
        ModelEntry(name="unused-cloud", provider="openrouter", model_id="remote")
    )
    cfg.resilience.max_retries = 0
    cloud_creations = []

    def _factory(entry, **kwargs):
        if entry.provider != "local":
            cloud_creations.append(entry.name)
        return Local401()

    monkeypatch.setattr(provider_module, "_create_chat_model", _factory)
    chain = LLMProvider(cfg).get_chain("extraction")

    with pytest.raises(FallbackExhaustedError):
        await invoke_with_resilience(chain, [], cfg.resilience)
    assert cloud_creations == []


@pytest.mark.asyncio
async def test_tracing_enabled_rejects_local_only_start_and_arms_offline_mode(
    monkeypatch, tmp_path
):
    cfg = _local_only_config(tmp_path)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    with pytest.raises(ConfigError, match="LANGSMITH_TRACING"):
        await BrainAgent(cfg).start()

    assert __import__("os").environ["HF_HUB_OFFLINE"] == "1"
    assert __import__("os").environ["TRANSFORMERS_OFFLINE"] == "1"


def test_direct_http_guard_raises_before_nonlocal_request():
    with pytest.raises(EgressPolicyError, match="1.1.1.1"):
        check_url("https://1.1.1.1/path", policy="local-only")
    check_url("http://127.0.0.1:8000/v1/models", policy="local-only")


def test_auth_http_call_is_blocked_before_urlopen(monkeypatch):
    opened = []
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: opened.append(a))
    with pytest.raises(EgressPolicyError, match="auth.openai.com"):
        _do_token_refresh(
            "https://auth.openai.com/oauth/token", "not-a-real-secret", egress="local-only"
        )
    assert opened == []


def test_doctor_http_call_is_blocked_before_httpx(monkeypatch):
    requested = []
    monkeypatch.setattr("httpx.get", lambda *a, **k: requested.append(a))
    with pytest.raises(EgressPolicyError, match="192.168.1.20"):
        _served_model_ids("http://192.168.1.20:8000/v1", egress="local-only")
    assert requested == []


def test_any_preserves_existing_provider_behavior(monkeypatch, tmp_path):
    cfg = _local_only_config(tmp_path)
    cfg.deployment.egress = "any"
    cfg.models.catalog.append(ModelEntry(name="cloud", provider="openrouter", model_id="remote"))
    sentinel = object()
    monkeypatch.setattr(provider_module, "_create_chat_model", lambda *a, **k: sentinel)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    assert validate_egress(cfg) == []
    enforce_startup_egress(cfg)
    assert LLMProvider(cfg).get("cloud") is sentinel


def test_localhost_resolution_and_userinfo_fail_closed(monkeypatch):
    real_getaddrinfo = socket.getaddrinfo

    def poisoned(host, port, *args, **kwargs):
        if host.lower() == "localhost":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
                    (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("198.51.100.10", 0))]
        return real_getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", poisoned)
    assert not is_local_url("http://localhost:8000/v1")
    assert not is_local_url("http://user:password@127.0.0.1:8000/v1")


def test_url_and_violation_messages_redact_credentials_path_and_query():
    url = "http://user:password@127.0.0.1:8000/private?token=hidden"
    with pytest.raises(EgressPolicyError) as exc_info:
        check_url(url, policy="local-only")
    assert "http://127.0.0.1:8000" in str(exc_info.value)
    for secret in ("user", "password", "private", "token", "hidden"):
        assert secret not in str(exc_info.value)

    cfg = _local_only_config()
    cfg.models.catalog[0].endpoint = url
    cfg.neo4j.uri = "bolt://user:password@remote.invalid:7687/private?token=hidden"
    text = "\n".join(map(str, validate_egress(cfg)))
    assert "bolt://remote.invalid:7687" in text
    for secret in ("password", "private", "token", "hidden"):
        assert secret not in text


@pytest.mark.parametrize("offline_name", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_proxy_warning_once_and_false_offline_value_rejected(monkeypatch, caplog, offline_name):
    cfg = _local_only_config()
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:9")
    monkeypatch.setenv(offline_name, "0")
    with caplog.at_level(logging.WARNING), pytest.raises(ConfigError, match=offline_name):
        enforce_startup_egress(cfg)
    assert len([r for r in caplog.records if "ignoring environment proxies" in r.message]) == 1
    assert "proxy.invalid" not in caplog.text

    monkeypatch.delenv(offline_name)
    enforce_startup_egress(cfg)
    assert __import__("os").environ[offline_name] == "1"


def test_local_clients_ignore_proxy_and_guard_both_request_paths(monkeypatch):
    cfg = _local_only_config()
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:9")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.invalid:9")
    captured = {}
    monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **params: captured.update(params) or params)
    LLMProvider(cfg).get("local")
    sync_client = captured["http_client"]
    async_client = captured["http_async_client"]
    assert sync_client._trust_env is False
    assert async_client._trust_env is False
    assert sync_client.follow_redirects is False
    assert async_client.follow_redirects is False

    calls = []
    sync_client._transport = httpx.MockTransport(
        lambda request: calls.append(str(request.url)) or httpx.Response(
            302, headers={"Location": "https://remote.invalid/private?token=hidden"}
        )
    )
    response = sync_client.get("http://127.0.0.1:8000/redirect")
    assert response.status_code == 302 and len(calls) == 1
    with pytest.raises(EgressPolicyError):
        sync_client.get("https://remote.invalid/private?token=hidden")
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_async_local_client_hook_blocks_before_transport(monkeypatch):
    cfg = _local_only_config()
    captured = {}
    monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **params: captured.update(params) or params)
    LLMProvider(cfg).get("local")
    client = captured["http_async_client"]
    calls = []
    client._transport = httpx.MockTransport(
        lambda request: calls.append(str(request.url)) or httpx.Response(200)
    )
    with pytest.raises(EgressPolicyError):
        await client.get("https://remote.invalid/private?token=hidden")
    assert calls == []
    await client.aclose()


def test_neo4j_uri_validated_at_startup():
    cfg = _local_only_config()
    cfg.neo4j.uri = "neo4j+s://remote.invalid:7687"
    assert any(v.reference == "[neo4j].uri" for v in validate_egress(cfg))
    with pytest.raises(ConfigError, match=r"\[neo4j\]\.uri"):
        enforce_startup_egress(cfg)
    cfg.neo4j.uri = "bolt://localhost:7687"
    assert validate_egress(cfg) == []


def test_direct_http_clients_disable_environment_proxy(monkeypatch):
    import urllib.request

    seen = {}

    class FakeOpener:
        def open(self, *args, **kwargs):
            seen["opened"] = True
            class Response:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    return None
                def read(self):
                    return b"{}"
            return Response()

    def build_opener(*handlers):
        seen["handlers"] = handlers
        return FakeOpener()

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    _do_token_refresh("http://127.0.0.1:8000/token", "placeholder", egress="local-only")
    assert seen["opened"]
    assert any(isinstance(h, urllib.request.ProxyHandler) and h.proxies == {}
               for h in seen["handlers"])

    requested = {}
    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: requested.update(kwargs) or
                        httpx.Response(200, json={"data": []}, request=httpx.Request("GET", args[0])))
    assert _served_model_ids("http://127.0.0.1:8000/v1", egress="local-only") == []
    assert requested["trust_env"] is False and requested["follow_redirects"] is False


@pytest.mark.parametrize("policy,expected", [("local-only", True), ("any", False)])
def test_docker_run_network_flag_follows_policy(monkeypatch, tmp_path, policy, expected):
    from types import SimpleNamespace

    from openclaw_brain.knowledge.executable.runner import NgspiceRunner

    monkeypatch.setenv("OPENCLAW_EGRESS", policy)
    if policy == "local-only":
        monkeypatch.setenv("DOCKER_HOST", "unix:///tmp/docker-test.sock")
        monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    commands = []
    monkeypatch.setattr("subprocess.run", lambda cmd, **kwargs:
                        commands.append(cmd) or SimpleNamespace(stdout="", stderr="", returncode=0))
    NgspiceRunner(workdir=str(tmp_path)).run_deck("* fake deck")
    assert ("--network" in commands[0] and "none" in commands[0]) is expected


def test_f1_cli_preflight_blocks_direct_status_before_graph(monkeypatch, tmp_path):
    from click.testing import CliRunner
    from openclaw_brain.cli import main
    cfg = _local_only_config()
    cfg.neo4j.uri = "bolt://remote.invalid:7687"
    path = tmp_path / "config.toml"
    save_config(cfg, path)
    calls = []
    monkeypatch.setattr("openclaw_brain.knowledge.graph.store.AsyncGraphDatabase.driver",
                        lambda *a, **k: calls.append(a))
    result = CliRunner().invoke(main, ["--config", str(path), "status"])
    assert result.exit_code != 0 and calls == []
    assert "[neo4j].uri" in str(result.exception)


@pytest.mark.asyncio
async def test_f1_graph_and_exporter_check_before_driver_or_credentials(monkeypatch, tmp_path):
    from openclaw_brain.knowledge.graph.store import GraphStore
    from openclaw_brain.export.obsidian import ObsidianExporter
    cfg = _local_only_config()
    cfg.neo4j.uri = "bolt://remote.invalid:7687"
    calls = []
    monkeypatch.setattr("openclaw_brain.knowledge.graph.store.AsyncGraphDatabase.driver",
                        lambda *a, **k: calls.append(a))
    for obj in (GraphStore(cfg.neo4j, egress="local-only"),
                ObsidianExporter(cfg.neo4j, tmp_path, egress="local-only")):
        with pytest.raises(EgressPolicyError):
            await obj.connect()
    assert calls == []


def test_f2_mineru_cache_miss_stops_before_import_or_download(monkeypatch, tmp_path):
    import builtins
    import json
    from openclaw_brain.knowledge.extraction.mineru_parser import _parse_with_mineru
    imports = []
    original = builtins.__import__
    def watched(name, *args, **kwargs):
        if name.startswith("mineru.cli.common"):
            imports.append(name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", watched)
    monkeypatch.setenv("MINERU_MODEL_SOURCE", "local")
    config_path = tmp_path / "mineru.json"
    config_path.write_text(json.dumps({"models-dir": {"pipeline": str(tmp_path / "missing")}}))
    monkeypatch.setenv("MINERU_TOOLS_CONFIG_JSON", str(config_path))
    with pytest.raises(ConfigError, match="prepared models-dir"):
        _parse_with_mineru(tmp_path / "fake.pdf", tmp_path, "s", "x", "pipeline", egress="local-only")
    assert imports == []
    monkeypatch.setenv("MINERU_MODEL_SOURCE", "modelscope")
    with pytest.raises(ConfigError, match="MINERU_MODEL_SOURCE=local"):
        _parse_with_mineru(tmp_path / "fake.pdf", tmp_path, "s", "x", "pipeline", egress="local-only")
    assert imports == []


@pytest.mark.parametrize("value,accepted", [("", True), ("YES", True), ("0", False),
                                              ("maybe", False)])
def test_f3_offline_values_and_imported_online_constant(monkeypatch, value, accepted):
    cfg = _local_only_config()
    monkeypatch.setenv("HF_HUB_OFFLINE", value)
    if accepted:
        enforce_startup_egress(cfg)
        assert os.environ["HF_HUB_OFFLINE"] == "1"
    else:
        with pytest.raises(ConfigError, match="HF_HUB_OFFLINE"):
            enforce_startup_egress(cfg)


def test_f3_preimported_online_hub_requires_restart(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants",
                        types.SimpleNamespace(HF_HUB_OFFLINE=False))
    with pytest.raises(ConfigError, match="restart"):
        enforce_startup_egress(_local_only_config())


def test_f3_embedding_loader_gets_local_files_only(monkeypatch):
    from openclaw_brain.knowledge import embedding
    seen = {}
    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = lambda *a, **k: seen.update(k) or object()
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(embedding, "_model", None)
    embedding.get_embedder("fake-model")
    assert seen["local_files_only"] is True


@pytest.mark.parametrize("scheme", ["neo4j", "neo4j+s", "neo4j+ssc"])
def test_f4_routing_uri_rejected_even_on_loopback(scheme):
    cfg = _local_only_config()
    cfg.neo4j.uri = f"{scheme}://127.0.0.1:7687"
    assert any(v.reference == "[neo4j].uri" for v in validate_egress(cfg))


def test_f5_runner_uses_active_config_not_implicit_disk(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from openclaw_brain.knowledge.executable.runner import NgspiceRunner
    cfg = _local_only_config()
    monkeypatch.delenv("OPENCLAW_EGRESS", raising=False)
    monkeypatch.setenv("DOCKER_HOST", "unix:///tmp/docker-test.sock")
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.setattr("openclaw_brain.config.load_config",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("disk reload")))
    commands = []
    monkeypatch.setattr("subprocess.run", lambda cmd, **kw:
                        commands.append(cmd) or SimpleNamespace(stdout="", stderr="", returncode=0))
    NgspiceRunner(workdir=str(tmp_path), egress=effective_egress(cfg)).run_deck("* synthetic")
    assert commands[0][2:4] == ["--network", "none"]


def test_f5_engine_registry_passes_active_config(monkeypatch):
    from openclaw_brain.knowledge.executable.engines import runner_for_engine
    cfg = _local_only_config()
    monkeypatch.delenv("OPENCLAW_EGRESS", raising=False)
    assert runner_for_engine("ngspice", cfg).egress == "local-only"


@pytest.mark.parametrize("host,context", [("tcp://remote.invalid:2375", ""),
                                          ("", ""), ("unix:///tmp/docker.sock", "remote")])
def test_f9_remote_or_ambiguous_docker_daemon_rejected(monkeypatch, tmp_path, host, context):
    from openclaw_brain.knowledge.executable.runner import NgspiceRunner
    monkeypatch.setenv("DOCKER_HOST", host)
    monkeypatch.setenv("DOCKER_CONTEXT", context)
    with pytest.raises(EgressPolicyError, match="unix"):
        NgspiceRunner(workdir=str(tmp_path), egress="local-only").run_deck("* synthetic")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("listing", ["blocked", "unreachable", "served", "malformed"])
def test_f6_doctor_redacts_every_endpoint_display(monkeypatch, capsys, listing):
    from openclaw_brain import cli
    cfg = _local_only_config()
    cfg.models.catalog[0].endpoint = "http://user:secret@127.0.0.1:8000/private?token=marker"
    result = {"blocked": cli.BLOCKED_LISTING, "unreachable": None,
              "served": ["local-model"], "malformed": cli.MALFORMED_LISTING}[listing]
    if listing != "blocked":
        monkeypatch.setattr(cli, "_served_model_ids", lambda *a, **k: result)
    cli._print_model_report(cfg)
    output = capsys.readouterr().out
    assert "http://127.0.0.1:8000" in output
    assert not any(marker in output for marker in ("user", "secret", "private", "token=", "marker"))


@pytest.mark.parametrize("name,value", [("LANGSMITH_TRACING", "true"),
                                        ("HF_HUB_OFFLINE", "0")])
def test_f7_doctor_reports_startup_environment_conflict(monkeypatch, tmp_path, name, value):
    from click.testing import CliRunner
    from openclaw_brain import cli
    path = tmp_path / "config.toml"
    save_config(_local_only_config(), path)
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(cli, "_doctor_checks", lambda full=False: [])
    result = CliRunner().invoke(cli.main, ["--config", str(path), "doctor"])
    assert result.exit_code == 1
    assert name in result.output and "FAIL" in result.output
    assert "all 1 checks green" not in result.output
