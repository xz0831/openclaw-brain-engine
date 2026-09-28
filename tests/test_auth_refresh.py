"""Tests for OAuth token refresh in auth module."""

import json
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from openclaw_brain.auth import (
    refresh_oauth_token,
    create_auth_refresh_callback,
    _atomic_write_json,
)


@pytest.fixture
def mock_auth_file(tmp_path):
    """Create a mock auth-profiles.json with an expired OAuth token."""
    auth_dir = tmp_path / ".openclaw" / "agents" / "main" / "agent"
    auth_dir.mkdir(parents=True)
    auth_file = auth_dir / "auth-profiles.json"
    auth_file.write_text(json.dumps({
        "version": 1,
        "profiles": {
            "openai-codex:default": {
                "type": "oauth",
                "provider": "openai-codex",
                "access": "old-expired-token",
                "refresh": "rt_test_refresh_abc",
                "expires": 1000000000000,  # Past
            },
        },
    }))
    return auth_file


def test_refresh_success(mock_auth_file):
    new_tokens = {
        "access_token": "new-fresh-token-xyz",
        "refresh_token": "rt_new_refresh_123",
        "expires_in": 3600,
    }

    with patch("openclaw_brain.auth._find_auth_file", return_value=mock_auth_file), \
         patch("openclaw_brain.auth._do_token_refresh", return_value=new_tokens), \
         patch.dict(os.environ, {}, clear=True):

        result = refresh_oauth_token("openai")
        assert result is True

        # Verify file was updated
        data = json.loads(mock_auth_file.read_text())
        profile = data["profiles"]["openai-codex:default"]
        assert profile["access"] == "new-fresh-token-xyz"
        assert profile["refresh"] == "rt_new_refresh_123"
        assert profile["expires"] > int(time.time() * 1000)

        # Verify env var was updated
        assert os.environ.get("OPENAI_API_KEY") == "new-fresh-token-xyz"


def test_refresh_no_auth_file():
    with patch("openclaw_brain.auth._find_auth_file", return_value=None):
        assert refresh_oauth_token("openai") is False


def test_refresh_no_refresh_token(tmp_path):
    auth_dir = tmp_path / ".openclaw" / "agents" / "main" / "agent"
    auth_dir.mkdir(parents=True)
    auth_file = auth_dir / "auth-profiles.json"
    auth_file.write_text(json.dumps({
        "version": 1,
        "profiles": {
            "openai-codex:default": {
                "type": "oauth",
                "provider": "openai-codex",
                "access": "token-no-refresh",
            },
        },
    }))

    with patch("openclaw_brain.auth._find_auth_file", return_value=auth_file):
        assert refresh_oauth_token("openai") is False


def test_refresh_network_error(mock_auth_file):
    with patch("openclaw_brain.auth._find_auth_file", return_value=mock_auth_file), \
         patch("openclaw_brain.auth._do_token_refresh", side_effect=Exception("Network error")):

        assert refresh_oauth_token("openai") is False


def test_refresh_no_access_token_in_response(mock_auth_file):
    with patch("openclaw_brain.auth._find_auth_file", return_value=mock_auth_file), \
         patch("openclaw_brain.auth._do_token_refresh", return_value={"error": "invalid_grant"}):

        assert refresh_oauth_token("openai") is False


def test_refresh_non_oauth_profile(tmp_path):
    auth_dir = tmp_path / ".openclaw" / "agents" / "main" / "agent"
    auth_dir.mkdir(parents=True)
    auth_file = auth_dir / "auth-profiles.json"
    auth_file.write_text(json.dumps({
        "version": 1,
        "profiles": {
            "anthropic:default": {
                "type": "api_key",
                "provider": "anthropic",
                "key": "sk-ant-123",
            },
        },
    }))

    with patch("openclaw_brain.auth._find_auth_file", return_value=auth_file):
        assert refresh_oauth_token("anthropic") is False


def test_create_auth_refresh_callback():
    callback = create_auth_refresh_callback()
    assert callable(callback)

    with patch("openclaw_brain.auth.refresh_oauth_token", return_value=True) as mock_refresh:
        result = callback()
        assert result is True
        mock_refresh.assert_called_once_with("openai")


def test_create_auth_refresh_callback_failure():
    callback = create_auth_refresh_callback()

    with patch("openclaw_brain.auth.refresh_oauth_token", return_value=False):
        result = callback()
        assert result is False


def test_create_auth_refresh_callback_clears_provider_cache():
    """After successful refresh, provider cache must be cleared."""
    mock_provider = MagicMock()
    callback = create_auth_refresh_callback(llm_provider=mock_provider)

    with patch("openclaw_brain.auth.refresh_oauth_token", return_value=True):
        result = callback()
        assert result is True
        mock_provider.clear_cache.assert_called_once()


def test_create_auth_refresh_callback_no_clear_on_failure():
    """On failed refresh, provider cache should NOT be cleared."""
    mock_provider = MagicMock()
    callback = create_auth_refresh_callback(llm_provider=mock_provider)

    with patch("openclaw_brain.auth.refresh_oauth_token", return_value=False):
        result = callback()
        assert result is False
        mock_provider.clear_cache.assert_not_called()


# ── Profile-scoped atomic write (file-race defect fix) ──
#
# refresh_oauth_token() used to read the whole auth-profiles.json, mutate one
# profile in memory, and write the whole (now possibly stale) snapshot back with a
# plain write_text() — no temp file, no fsync, no re-read. Two processes racing
# (this module + OpenClaw's own agent, which shares the same file) could silently
# revert each other's changes to unrelated profiles, and a crash mid-write could
# truncate the file. These tests pin the fix: a fresh re-read immediately before
# writing, a scoped merge of only the refreshed profile's keys, and an atomic
# temp-file + fsync + os.replace() write.


def test_refresh_uses_atomic_replace(mock_auth_file):
    """The write goes through os.replace() (a same-directory temp file, not a
    direct write_text over the target) — wraps the real os.replace so the file
    is still genuinely written, while recording the call for inspection."""
    new_tokens = {"access_token": "new-fresh-token-xyz", "expires_in": 3600}

    with patch("openclaw_brain.auth._find_auth_file", return_value=mock_auth_file), \
         patch("openclaw_brain.auth._do_token_refresh", return_value=new_tokens), \
         patch("openclaw_brain.auth.os.replace", wraps=os.replace) as mock_replace, \
         patch.dict(os.environ, {}, clear=True):

        assert refresh_oauth_token("openai") is True

    mock_replace.assert_called_once()
    tmp_src, dst = mock_replace.call_args[0]
    assert Path(dst) == mock_auth_file
    assert Path(tmp_src).parent == mock_auth_file.parent  # same dir -> atomic rename
    assert Path(tmp_src) != mock_auth_file  # wrote to a distinct temp file first
    assert not Path(tmp_src).exists()  # temp file consumed by the (real) rename

    data = json.loads(mock_auth_file.read_text())
    assert data["profiles"]["openai-codex:default"]["access"] == "new-fresh-token-xyz"


def test_refresh_preserves_concurrent_write_to_other_profile(tmp_path):
    """Core regression test for the race: if another process (e.g. OpenClaw's own
    agent, sharing this same auth-profiles.json) writes a DIFFERENT profile while
    our token-refresh HTTP call is in flight, that write must survive — not be
    silently reverted by our own now-stale full-file snapshot.

    Simulated deterministically: _do_token_refresh's mock has a side effect that
    mutates the file (standing in for a concurrent writer) before returning the
    new tokens. The old whole-file read-modify-write captured `data` BEFORE this
    call and wrote that stale snapshot back, clobbering exactly this change; the
    fix re-reads immediately before writing, so the concurrent change is present
    in the copy it merges into.
    """
    auth_dir = tmp_path / ".openclaw" / "agents" / "main" / "agent"
    auth_dir.mkdir(parents=True)
    auth_file = auth_dir / "auth-profiles.json"
    auth_file.write_text(json.dumps({
        "version": 1,
        "profiles": {
            "openai-codex:default": {
                "type": "oauth",
                "provider": "openai-codex",
                "access": "old-openai-token",
                "refresh": "rt_openai_abc",
                "expires": 1000000000000,
            },
            "anthropic:default": {
                "type": "api_key",
                "provider": "anthropic",
                "key": "original-anthropic-key",
            },
        },
    }))

    def _fake_refresh(token_url, refresh_token):
        # Stand-in for a concurrent writer touching a DIFFERENT profile mid-flight.
        data = json.loads(auth_file.read_text())
        data["profiles"]["anthropic:default"]["key"] = "concurrently-rotated-key"
        auth_file.write_text(json.dumps(data, indent=2))
        return {"access_token": "new-openai-token", "expires_in": 3600}

    with patch("openclaw_brain.auth._find_auth_file", return_value=auth_file), \
         patch("openclaw_brain.auth._do_token_refresh", side_effect=_fake_refresh), \
         patch.dict(os.environ, {}, clear=True):

        assert refresh_oauth_token("openai") is True

    final = json.loads(auth_file.read_text())
    assert final["profiles"]["openai-codex:default"]["access"] == "new-openai-token"
    # The concurrent writer's change to a DIFFERENT profile must survive.
    assert final["profiles"]["anthropic:default"]["key"] == "concurrently-rotated-key"


def test_two_sequential_refreshes_of_different_profiles_both_persist(tmp_path):
    """Two profile-scoped refreshes, each for a different provider/profile, must
    each leave the other's change intact — neither call's write may clobber the
    other's, even sequentially (not just under mid-flight concurrency)."""
    auth_dir = tmp_path / ".openclaw" / "agents" / "main" / "agent"
    auth_dir.mkdir(parents=True)
    auth_file = auth_dir / "auth-profiles.json"
    auth_file.write_text(json.dumps({
        "version": 1,
        "profiles": {
            "openai-codex:default": {
                "type": "oauth", "provider": "openai-codex",
                "access": "old-openai", "refresh": "rt_openai", "expires": 1,
            },
            "test-provider-2:default": {
                "type": "oauth", "provider": "test-provider-2",
                "access": "old-provider-2", "refresh": "rt_provider_2", "expires": 1,
            },
        },
    }))

    with patch("openclaw_brain.auth._find_auth_file", return_value=auth_file), \
         patch("openclaw_brain.auth._token_endpoint_for_provider",
               return_value="https://example.invalid/token"), \
         patch.dict(os.environ, {}, clear=True):

        with patch("openclaw_brain.auth._do_token_refresh",
                    return_value={"access_token": "new-openai", "expires_in": 3600}):
            assert refresh_oauth_token("openai") is True

        with patch("openclaw_brain.auth._do_token_refresh",
                    return_value={"access_token": "new-provider-2", "expires_in": 3600}):
            assert refresh_oauth_token("test-provider-2") is True

    final = json.loads(auth_file.read_text())
    assert final["profiles"]["openai-codex:default"]["access"] == "new-openai"
    assert final["profiles"]["test-provider-2:default"]["access"] == "new-provider-2"


def test_atomic_write_json_cleans_up_temp_file_on_replace_failure(tmp_path):
    """If os.replace() itself fails (e.g. disk full, permission error), the temp
    file must not be left behind and the original file must be untouched — a
    failed write must not corrupt or duplicate state."""
    target = tmp_path / "data.json"
    target.write_text('{"original": true}')

    with patch("openclaw_brain.auth.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            _atomic_write_json(target, {"new": "data"})

    assert json.loads(target.read_text()) == {"original": True}
    leftovers = [p for p in tmp_path.iterdir() if p.name != "data.json"]
    assert leftovers == [], f"temp file(s) left behind: {leftovers}"


def test_atomic_write_json_writes_correct_content(tmp_path):
    """Direct unit test of the happy path: target file ends up with exactly the
    given data, valid JSON, no temp file left behind."""
    target = tmp_path / "data.json"
    payload = {"profiles": {"a": {"access": "token-value"}}}

    _atomic_write_json(target, payload)

    assert json.loads(target.read_text()) == payload
    leftovers = [p for p in tmp_path.iterdir() if p.name != "data.json"]
    assert leftovers == []
