"""Tests for the `doctor` env-drift command.

The green-run test doubles as the suite's own drift alarm: it executes the real checks
against the real venv, so a dependency drift like the 2026-07-08 incident (ad-hoc install
bumped transformers to 5.13, silently breaking both MinerU backends) fails the test suite
instead of surfacing days later inside a live ingest.
"""

from __future__ import annotations

import socket

import httpx
import pytest
from click.testing import CliRunner

from openclaw_brain import cli as cli_module
from openclaw_brain.cli import (
    MALFORMED_LISTING, main, _doctor_checks, _model_id_keys, _model_report,
    _print_model_report, _served_model_ids,
)
from openclaw_brain.config import BrainConfig, ModelEntry, ModelsConfig, save_config
from openclaw_brain.llm import provider as provider_module


def test_doctor_help_works():
    result = CliRunner().invoke(main, ["doctor", "--help"])
    assert result.exit_code == 0
    assert "drift" in result.output


def test_doctor_green_in_this_env():
    """Real checks, real venv — this test IS the drift alarm (fast mode: imports only)."""
    result = CliRunner().invoke(main, ["doctor"])
    assert result.exit_code == 0, f"doctor found drift:\n{result.output}"
    assert "all" in result.output and "green" in result.output


def test_doctor_checks_report_shape():
    rows = _doctor_checks(full=False)
    assert len(rows) >= 4  # transformers range, ABI, AutoProcessor, mineru (+ mlx on darwin)
    for name, ok, detail in rows:
        assert isinstance(name, str) and isinstance(ok, bool) and isinstance(detail, str)


# ── doctor --models: catalog liveness + candidacy (2026-07-25) ──
#
# Everything below runs against a FAKE server listing. Mock fidelity (CLAUDE.md): the payload
# mirrors the real oMLX/EXO envelope measured on this machine 2026-07-25 —
#   {"object": "list", "data": [{"id": "Qwen3.5-27B-4bit", "object": "model",
#                                "created": 1784953434, "owned_by": "omlx", ...}]}
# — because the report's whole job is to compare `id` fields, and a flat list-of-strings mock
# would pass while hiding exactly the shape bug it exists to catch.

_OMLX = "http://localhost:8000/v1"
_EXO = "http://localhost:52415/v1"

# The served ids deliberately cover all three namespacing conventions in play, because the
# report's match key normalizes them (`cli._model_id_keys`):
#   bare               — oMLX's usual shape, matched against a bare catalog model_id
#   bare + '/' catalog — oMLX bare id vs the catalog's `mlx-community/...` form
#   '--'-flattened     — what :8000 actually returns for some entries (measured 2026-07-25);
#                        3 catalogued models were reported as "uncatalogued candidates" until
#                        this separator was folded too.
_SERVED_OMLX = [
    "Qwen3.5-27B-4bit",                            # catalogued bare-to-bare
    "Qwen3.6-27B-8bit",                            # catalogued bare-vs-'/'-namespaced
    "mlx-community--Qwen3-VL-32B-Instruct-4bit",   # catalogued '--'-flattened-vs-'/'
    "Qwen3-VL-8B-Instruct-4bit",                   # uncatalogued → candidate
    "gemma-4-26b-a4b-it-6bit",                     # uncatalogued → candidate
]


class _FakeListingResponse:
    def __init__(self, payload) -> None:
        self._payload = payload

    @classmethod
    def of_ids(cls, ids: list[str]) -> "_FakeListingResponse":
        return cls({
            "object": "list",
            "data": [
                {"id": i, "object": "model", "created": 1784953434, "owned_by": "omlx"}
                for i in ids
            ],
        })

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _NetworkAttempt(BaseException):
    """Tripwire for any real network use inside the report.

    Deliberately a BaseException, NOT an Exception: `_model_report` wraps model construction
    in `except Exception` and turns it into a NOT-CONSTRUCTIBLE row, so an Exception-derived
    tripwire would be SWALLOWED — the test would pass green while the call went out. Only a
    BaseException escapes that handler and fails the test.
    """


@pytest.fixture
def fake_servers(monkeypatch):
    """oMLX (:8000) up and serving 5 models; EXO (:52415) refusing connections."""
    calls: list[str] = []

    def _fake_get(url, *args, **kwargs):
        calls.append(url)
        if ":8000" in url:
            return _FakeListingResponse.of_ids(_SERVED_OMLX)
        raise httpx.ConnectError("[Errno 61] Connection refused")

    monkeypatch.setattr(httpx, "get", _fake_get)

    # Network guard: nothing in this report may issue a real request — a model is judged by the
    # server's LISTING or by CONSTRUCTION, never by inference.
    def _no_network(*a, **k):
        raise _NetworkAttempt("doctor --models issued a network request (no inference allowed)")

    monkeypatch.setattr(httpx.Client, "send", _no_network)
    monkeypatch.setattr(httpx.AsyncClient, "send", _no_network)
    # …and a transport-agnostic backstop under httpx, because a provider SDK's transport is a
    # version-dependent implementation detail: langchain-google-genai has shipped both a grpc
    # stack (google-ai-generativelanguage) and an httpx one (google-genai 2.x, what is installed
    # here), and grpc would sail straight past the two patches above. Sockets are the floor every
    # transport must reach through.
    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", _no_network)

    # Belt and braces on the same point: NO google entry is ever constructed in this file. The
    # google row exists to pin the NO-KEY verdict (GOOGLE_API_KEY is unset in every test here),
    # so making the construction itself a tripwire proves that verdict is reached by not
    # constructing, rather than by constructing something that quietly dialled out.
    def _no_google(*a, **k):
        raise _NetworkAttempt(
            "a google model was CONSTRUCTED in the doctor test path — google-genai's transport "
            "is version-dependent; assert the NO-KEY path instead"
        )

    monkeypatch.setattr(provider_module, "_create_google", _no_google)
    return calls


def _report_config() -> BrainConfig:
    cfg = BrainConfig()
    cfg.models = ModelsConfig(
        default_extraction="live-local",
        default_reasoning="live-local",
        default_matching="live-local",
        catalog=[
            ModelEntry(name="live-local", provider="local", model_id="Qwen3.5-27B-4bit",
                       endpoint=_OMLX, tier="local"),
            # Namespaced in the catalog, bare in the listing — exercises the '/'-basename
            # branch on a REACHABLE endpoint (the only namespaced entry used to sit behind the
            # deliberately-down EXO, so that branch was never actually executed by a test).
            ModelEntry(name="namespaced-local", provider="local",
                       model_id="mlx-community/Qwen3.6-27B-8bit", endpoint=_OMLX, tier="local"),
            # Namespaced in the catalog, '--'-flattened in the listing: the real :8000 shape
            # that made 3 catalogued models show up as "uncatalogued candidates".
            ModelEntry(name="flattened-local", provider="local",
                       model_id="mlx-community/Qwen3-VL-32B-Instruct-4bit",
                       endpoint=_OMLX, tier="local"),
            ModelEntry(name="ghost-local", provider="local", model_id="Qwen3.5-Ghost-4bit",
                       endpoint=_OMLX, tier="local"),
            ModelEntry(name="exo-local", provider="local",
                       model_id="mlx-community/GLM-4.7-4bit", endpoint=_EXO, tier="local"),
            ModelEntry(name="keyed-remote", provider="openai", model_id="gpt-x",
                       tier="frontier"),
            ModelEntry(name="keyless-remote", provider="google", model_id="gemini-x",
                       tier="frontier"),
            ModelEntry(name="blocked-remote", provider="anthropic", model_id="claude-x",
                       tier="frontier", status="banned"),
            ModelEntry(name="retired-remote", provider="xai", model_id="grok-x",
                       tier="frontier", status="deprecated"),
        ],
    )
    return cfg


def test_model_report_verdicts(fake_servers, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("XAI_API_KEY", "test-key-not-real")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    rows, listings = _model_report(_report_config())
    verdicts = {r[0]: r[4] for r in rows}

    assert verdicts["live-local"] == "SERVED"
    # Both namespacing conventions the servers actually use resolve to the same model.
    assert verdicts["namespaced-local"] == "SERVED"   # catalog 'mlx-community/…' vs bare id
    assert verdicts["flattened-local"] == "SERVED"    # catalog 'mlx-community/…' vs '--' id
    assert verdicts["ghost-local"] == "NOT-SERVED"
    # The EXO server is DOWN — its model is UNKNOWN, never NOT-SERVED. Reporting a down server
    # as evidence against a model is how a healthy entry gets deleted for being probed mid-restart.
    assert verdicts["exo-local"] == "UNKNOWN"
    assert verdicts["keyed-remote"] == "CONSTRUCTIBLE"
    assert verdicts["keyless-remote"] == "NO-KEY"
    assert verdicts["blocked-remote"] == "BANNED"
    assert verdicts["retired-remote"] == "CONSTRUCTIBLE"   # deprecated ≠ unusable

    assert dict(zip([r[0] for r in rows], [r[3] for r in rows]))["retired-remote"] == "deprecated"
    assert listings[_OMLX] == _SERVED_OMLX
    assert listings[_EXO] is None
    # One listing GET per distinct endpoint, not per entry (4 local entries share :8000).
    assert sorted(fake_servers) == [f"{_EXO}/models", f"{_OMLX}/models"]


def test_model_report_says_nothing_about_a_key_it_cannot_see(fake_servers, monkeypatch):
    """NO-KEY must not be reported as NOT-CONSTRUCTIBLE: a missing key is an environment fact,
    not a verdict on the model — and the report must not crash trying to construct anyway."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    rows, _ = _model_report(_report_config())
    row = next(r for r in rows if r[0] == "keyless-remote")
    assert row[4] == "NO-KEY"
    assert "GOOGLE_API_KEY" in row[5]


def test_print_model_report_renders_candidates_with_the_operator_caveat(fake_servers, monkeypatch,
                                                                       capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("XAI_API_KEY", "test-key-not-real")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    _print_model_report(_report_config(), sample_cap=1)
    out = capsys.readouterr().out

    assert "CANDIDATES ONLY" in out
    assert "A/B evaluation with cost measured" in out
    assert "operator decision" in out
    # 5 served on :8000; 3 catalogued (live-local bare, namespaced-local via '/', flattened-local
    # via '--') → 2 uncatalogued, sample capped at 1.
    assert f"http://localhost:8000 — 5 served, 3 in catalog, 2 uncatalogued" in out
    assert "+1 more (sample capped at 1)" in out
    # A model the catalog HAS must never be printed as a candidate to adopt, whichever
    # separator the server used to name it.
    assert "mlx-community--Qwen3-VL-32B-Instruct-4bit" not in out
    assert "Qwen3.6-27B-8bit" not in out
    assert "http://localhost:52415 — UNREACHABLE (no candidacy judgment)" in out
    assert "SERVED" in out and "NOT-SERVED" in out and "BANNED" in out


def test_doctor_models_flag_does_not_affect_the_exit_code(fake_servers, monkeypatch, tmp_path):
    """A deprecated entry, a down server, and an unserved model are DIAGNOSTIC — only the
    environment checks decide the exit code (unchanged semantics)."""
    monkeypatch.setattr(cli_module, "_doctor_checks", lambda full=False: [("fake check", True, "ok")])
    monkeypatch.setattr("openclaw_brain.auth.inject_api_keys", lambda: {})
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("XAI_API_KEY", "test-key-not-real")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    config_path = tmp_path / "models.toml"
    save_config(_report_config(), config_path)

    result = CliRunner().invoke(main, ["--config", str(config_path), "doctor", "--models"])

    assert result.exit_code == 0, result.output
    assert "ZERO inference calls" in result.output
    assert "ghost-local" in result.output and "NOT-SERVED" in result.output
    assert "CANDIDATES ONLY" in result.output


def test_doctor_models_still_exits_nonzero_when_an_env_check_fails(fake_servers, monkeypatch,
                                                                   tmp_path):
    monkeypatch.setattr(cli_module, "_doctor_checks",
                        lambda full=False: [("fake check", False, "drifted")])
    monkeypatch.setattr("openclaw_brain.auth.inject_api_keys", lambda: {})
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("XAI_API_KEY", "test-key-not-real")

    config_path = tmp_path / "models2.toml"
    save_config(_report_config(), config_path)

    result = CliRunner().invoke(main, ["--config", str(config_path), "doctor", "--models"])

    assert result.exit_code == 1
    assert "1/1 check(s) FAILED" in result.output
    # ...and the model report still ran: a failing env check must not hide the catalog state.
    assert "CANDIDATES ONLY" in result.output


# ── namespacing: '/' and '--' name the same model ──


def test_model_id_keys_folds_both_namespace_separators():
    """oMLX at :8000 flattens `owner/model` to `owner--model`; EXO and the catalog keep the '/'.
    Folding only '/' left catalogued models reported as adoption CANDIDATES (measured
    2026-07-25: 3 of 14), which is a report arguing against its own purpose."""
    slashed = _model_id_keys("mlx-community/Qwen3-VL-32B-Instruct-4bit")
    flattened = _model_id_keys("mlx-community--Qwen3-VL-32B-Instruct-4bit")

    assert slashed & flattened                      # they must be recognized as one model
    assert "Qwen3-VL-32B-Instruct-4bit" in slashed
    assert "Qwen3-VL-32B-Instruct-4bit" in flattened
    # A bare id keeps working, and an empty id yields no keys (never matches anything).
    assert _model_id_keys("Qwen3.5-27B-4bit") == {"Qwen3.5-27B-4bit"}
    assert _model_id_keys("") == set()


# ── malformed listings: an unreadable payload is UNKNOWN, never NOT-SERVED ──


@pytest.mark.parametrize("payload, shape", [
    (["Qwen3.5-27B-4bit", "Qwen3.6-27B-8bit"], "flat list of name strings"),
    ({"object": "list", "data": [{"model": "Qwen3.5-27B-4bit"}]}, "entries not keyed on `id`"),
])
def test_malformed_listing_is_unknown_not_not_served(monkeypatch, payload, shape):
    """A reachable server whose listing shape drifts must not be read as evidence AGAINST every
    model on it — that is the same slander the down-server branch exists to prevent. Only a
    listing we actually parsed can say a model is absent."""
    monkeypatch.setattr(httpx, "get", lambda url, *a, **k: _FakeListingResponse(payload))

    assert _served_model_ids(_OMLX) is MALFORMED_LISTING, shape

    cfg = BrainConfig()
    cfg.models = ModelsConfig(catalog=[
        ModelEntry(name="live-local", provider="local", model_id="Qwen3.5-27B-4bit",
                   endpoint=_OMLX, tier="local"),
    ])
    rows, listings = _model_report(cfg)
    assert rows[0][4] == "UNKNOWN"
    assert "unreadable" in rows[0][5]
    assert listings[_OMLX] is MALFORMED_LISTING

    _print_model_report(cfg)


def test_reachable_server_serving_nothing_still_says_not_served(monkeypatch):
    """The counter-case that keeps the fix honest: `{"data": []}` IS a parsed listing, so a
    model absent from it really is absent — that must stay NOT-SERVED, not become UNKNOWN."""
    monkeypatch.setattr(httpx, "get",
                        lambda url, *a, **k: _FakeListingResponse({"object": "list", "data": []}))

    assert _served_model_ids(_OMLX) == []

    cfg = BrainConfig()
    cfg.models = ModelsConfig(catalog=[
        ModelEntry(name="live-local", provider="local", model_id="Qwen3.5-27B-4bit",
                   endpoint=_OMLX, tier="local"),
    ])
    rows, _ = _model_report(cfg)
    assert rows[0][4] == "NOT-SERVED"


def test_print_model_report_makes_no_candidacy_claim_on_a_malformed_listing(monkeypatch, capsys):
    monkeypatch.setattr(httpx, "get",
                        lambda url, *a, **k: _FakeListingResponse(["Qwen3.5-27B-4bit"]))

    cfg = BrainConfig()
    cfg.models = ModelsConfig(catalog=[
        ModelEntry(name="live-local", provider="local", model_id="Qwen3.5-27B-4bit",
                   endpoint=_OMLX, tier="local"),
    ])
    _print_model_report(cfg)
    out = capsys.readouterr().out

    assert "http://localhost:8000 — LISTING UNREADABLE (no candidacy judgment)" in out
    assert "uncatalogued" not in out
