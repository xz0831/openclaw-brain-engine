"""Tests for harness-neutral Brain auth resolution."""

import json
import os
from unittest.mock import patch

import pytest

from openclaw_brain.auth import get_api_key, inject_api_keys, _from_openclaw_auth


def _reveal(secret):
    """Reveal a resolved SecretStr for comparison against a KNOWN TEST CONSTANT only.

    Resolution returns SecretStr (masked repr), so a real key can never spill via an
    assertion message. We only ever reveal to compare against fixture constants here.
    """
    return secret.get_secret_value() if secret is not None else None


@pytest.fixture
def mock_auth_file(tmp_path):
    """Create a mock Brain auth-profiles.json."""
    auth_dir = tmp_path / ".config" / "openclaw-brain"
    auth_dir.mkdir(parents=True)
    auth_file = auth_dir / "auth-profiles.json"
    auth_file.write_text(json.dumps({
        "version": 1,
        "profiles": {
            "anthropic:default": {
                "type": "api_key",
                "provider": "anthropic",
                "key": "sk-ant-test-key-12345",
            },
            "openai-codex:default": {
                "type": "oauth",
                "provider": "openai-codex",
                "access": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.test",
                "refresh": "rt_test_refresh_token",
                "expires": 9999999999999,  # Far future
            },
        },
    }))
    return auth_file


def test_env_var_takes_priority():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key-123"}):
        key = get_api_key("anthropic")
        assert _reveal(key) == "env-key-123"


def test_anthropic_from_auth_store(mock_auth_file):
    with patch("openclaw_brain.auth._find_auth_file", return_value=mock_auth_file):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            key = _from_openclaw_auth("anthropic")
            assert _reveal(key) == "sk-ant-test-key-12345"


def test_secretstr_repr_does_not_render_value(mock_auth_file):
    """The structural close: a resolved secret's repr/str/format are masked."""
    with patch("openclaw_brain.auth._find_auth_file", return_value=mock_auth_file):
        key = _from_openclaw_auth("anthropic")
        assert key is not None
        assert "sk-ant-test-key-12345" not in repr(key)
        assert "sk-ant-test-key-12345" not in str(key)
        assert "sk-ant-test-key-12345" not in f"{key}"
        assert "sk-ant-test-key-12345" not in "{}".format(key)


def test_openai_from_auth_store(mock_auth_file):
    with patch("openclaw_brain.auth._find_auth_file", return_value=mock_auth_file):
        key = _from_openclaw_auth("openai")
        assert _reveal(key) == "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.test"


def test_unknown_provider():
    key = _from_openclaw_auth("unknown-provider")
    assert key is None


def test_inject_api_keys(mock_auth_file):
    with patch("openclaw_brain.auth._find_auth_file", return_value=mock_auth_file):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            injected = inject_api_keys()
            assert "ANTHROPIC_API_KEY" in injected
            assert "OPENAI_API_KEY" in injected
            assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test-key-12345"


def test_inject_skips_existing_env():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "already-set"}):
        injected = inject_api_keys()
        assert "ANTHROPIC_API_KEY" not in injected
        assert os.environ["ANTHROPIC_API_KEY"] == "already-set"


def test_missing_auth_file():
    with patch("openclaw_brain.auth._find_auth_file", return_value=None):
        key = _from_openclaw_auth("anthropic")
        assert key is None


# ── SecretRef (source: exec, e.g. macOS keychain) resolution ──
#
# OpenClaw stores anthropic/google/xai keys as keyRefs, not inline. The resolver
# is a script declared in secret-providers.json that speaks a stdio JSON protocol:
#   stdin {"ids":[...]} -> stdout {"protocolVersion":1,"values":{id:secret},"errors":{}}

_RESOLVED_SECRET = "sk-ant-keychain-RESOLVED-do-not-log"
_ANTHROPIC_ID = "openclaw/auth-profiles/anthropic/default/key"

_FAKE_RESOLVER = '''#!/usr/bin/env python3
import sys, json
req = json.load(sys.stdin)
KNOWN = {"%s": "%s"}
values, errors = {}, {}
for i in req.get("ids", []):
    if i in KNOWN:
        values[i] = KNOWN[i]
    else:
        errors[i] = {"message": "not found in fake keychain"}
print(json.dumps({"protocolVersion": 1, "values": values, "errors": errors}))
''' % (_ANTHROPIC_ID, _RESOLVED_SECRET)


@pytest.fixture
def mock_keychain_auth(tmp_path):
    """auth-profiles.json with a keyRef + Brain secret config + fake resolver."""
    import shlex
    import sys

    auth_dir = tmp_path / ".config" / "openclaw-brain"
    auth_dir.mkdir(parents=True)
    auth_file = auth_dir / "auth-profiles.json"
    auth_file.write_text(json.dumps({
        "version": 1,
        "profiles": {
            "anthropic:default": {
                "type": "api_key",
                "provider": "anthropic",
                "keyRef": {
                    "source": "exec",
                    "provider": "macos-keychain",
                    "id": _ANTHROPIC_ID,
                },
            },
        },
    }))

    resolver = tmp_path / "resolver.py"
    resolver.write_text(_FAKE_RESOLVER)

    secret_config = auth_dir / "secret-providers.json"
    # command as a STRING (matches reality) that shlex-splits to [python, resolver]
    secret_config.write_text(json.dumps({
        "secrets": {
            "providers": {
                "macos-keychain": {
                    "source": "exec",
                    "command": shlex.join([sys.executable, str(resolver)]),
                    "jsonOnly": True,
                },
            },
        },
    }))
    return auth_file


def test_keyref_resolves_via_exec(mock_keychain_auth):
    with patch("openclaw_brain.auth._find_auth_file", return_value=mock_keychain_auth):
        with patch.dict(os.environ, {}, clear=True):
            key = _from_openclaw_auth("anthropic")
            assert _reveal(key) == _RESOLVED_SECRET


def test_keyref_inline_key_takes_priority(mock_keychain_auth):
    """An inline `key` must win over a keyRef (legacy profiles)."""
    data = json.loads(mock_keychain_auth.read_text())
    data["profiles"]["anthropic:default"]["key"] = "inline-wins"
    mock_keychain_auth.write_text(json.dumps(data))
    with patch("openclaw_brain.auth._find_auth_file", return_value=mock_keychain_auth):
        assert _reveal(_from_openclaw_auth("anthropic")) == "inline-wins"


def test_keyref_unknown_id_returns_none(mock_keychain_auth):
    """Resolver reports an error for an unknown id → graceful None, no crash."""
    data = json.loads(mock_keychain_auth.read_text())
    data["profiles"]["anthropic:default"]["keyRef"]["id"] = "openclaw/does/not/exist"
    mock_keychain_auth.write_text(json.dumps(data))
    with patch("openclaw_brain.auth._find_auth_file", return_value=mock_keychain_auth):
        assert _from_openclaw_auth("anthropic") is None


def test_keyref_missing_secret_config_returns_none(mock_keychain_auth):
    """No resolver config reachable → None, no crash.

    Patch the compatibility resolver lookup directly so no machine config can
    affect this unit test.
    """
    with patch("openclaw_brain.auth._find_auth_file", return_value=mock_keychain_auth):
        with patch("openclaw_brain.auth._find_openclaw_config", return_value=None):
            assert _from_openclaw_auth("anthropic") is None


def test_default_auth_lookup_never_reads_openclaw_agent_home(tmp_path):
    legacy = tmp_path / ".openclaw" / "agents" / "main" / "agent"
    legacy.mkdir(parents=True)
    (legacy / "auth-profiles.json").write_text('{"profiles": {}}')
    with patch("openclaw_brain.auth.Path.home", return_value=tmp_path), \
            patch.dict(os.environ, {}, clear=True):
        from openclaw_brain.auth import _find_auth_file
        assert _find_auth_file() is None


def test_inject_api_keys_resolves_keyref(mock_keychain_auth):
    with patch("openclaw_brain.auth._find_auth_file", return_value=mock_keychain_auth):
        with patch.dict(os.environ, {}, clear=True):
            injected = inject_api_keys()
            assert "ANTHROPIC_API_KEY" in injected
            assert os.environ["ANTHROPIC_API_KEY"] == _RESOLVED_SECRET
            # masked summary must not leak the full secret
            assert _RESOLVED_SECRET not in injected["ANTHROPIC_API_KEY"]
