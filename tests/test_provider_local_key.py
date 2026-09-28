"""_create_local reads the oMLX bearer key from OMLX_API_KEY (2026-09-27).

Without it, a key-protected oMLX endpoint answers 401 and the reasoning chain drops to
its cloud fallback. Mock fidelity (CLAUDE.md): assert on the real ChatOpenAI object the
factory returns, not on a stand-in.
"""
from openclaw_brain.config import ModelEntry
from openclaw_brain.llm.provider import _create_local


def _entry():
    return ModelEntry(name="local-test", provider="local", model_id="Some-Local-Model",
                      tier="local", endpoint="http://localhost:8000/v1")


def test_local_uses_omlx_api_key_when_set(monkeypatch):
    monkeypatch.setenv("OMLX_API_KEY", "test-omlx-key")
    model = _create_local(_entry())
    assert model.openai_api_key.get_secret_value() == "test-omlx-key"
    assert str(model.openai_api_base).startswith("http://localhost:8000")


def test_local_keeps_placeholder_without_key(monkeypatch):
    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    model = _create_local(_entry())
    assert model.openai_api_key.get_secret_value() == "not-needed"


def test_key_file_json_reads_auth_api_key_only(monkeypatch, tmp_path):
    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    f = tmp_path / "settings.json"
    f.write_text('{"server": {"port": 8000}, "auth": {"api_key": "file-json-key"}}', encoding="utf-8")
    monkeypatch.setenv("OMLX_API_KEY_FILE", str(f))
    assert _create_local(_entry()).openai_api_key.get_secret_value() == "file-json-key"


def test_key_file_plain_text_first_line(monkeypatch, tmp_path):
    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    f = tmp_path / "key.txt"
    f.write_text("plain-key\nignored\n", encoding="utf-8")
    monkeypatch.setenv("OMLX_API_KEY_FILE", str(f))
    assert _create_local(_entry()).openai_api_key.get_secret_value() == "plain-key"


def test_env_value_wins_over_file(monkeypatch, tmp_path):
    f = tmp_path / "key.txt"
    f.write_text("file-key", encoding="utf-8")
    monkeypatch.setenv("OMLX_API_KEY_FILE", str(f))
    monkeypatch.setenv("OMLX_API_KEY", "env-key")
    assert _create_local(_entry()).openai_api_key.get_secret_value() == "env-key"


def test_unreadable_or_keyless_file_falls_back_without_leaking(monkeypatch, tmp_path, caplog):
    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    bad = tmp_path / "broken.json"
    bad.write_text('{"auth": {"api_key": "SECRET-SHOULD-NOT-LEAK"', encoding="utf-8")  # invalid JSON
    monkeypatch.setenv("OMLX_API_KEY_FILE", str(bad))
    with caplog.at_level("WARNING"):
        model = _create_local(_entry())
    assert model.openai_api_key.get_secret_value() == "not-needed"
    assert "SECRET-SHOULD-NOT-LEAK" not in caplog.text
    assert "JSONDecodeError" in caplog.text
    monkeypatch.setenv("OMLX_API_KEY_FILE", str(tmp_path / "missing.json"))
    assert _create_local(_entry()).openai_api_key.get_secret_value() == "not-needed"
