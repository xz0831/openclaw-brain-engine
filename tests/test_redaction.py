"""Tests for redaction.redact_secrets — the sink-side scrub for credential shapes.

All test vectors embed the literal "FAKE" so they are self-evidently not live keys
(and the repo's pre-commit live-key scan can exclude them by that marker).
"""

from openclaw_brain.redaction import redact_secrets


def test_redacts_anthropic_key():
    out = redact_secrets("boom: sk-ant-api03-FAKE0000000000000000 failed")
    assert "sk-ant-api03-FAKE0000000000000000" not in out
    assert "[REDACTED]" in out


def test_redacts_xai_openai_google_jwt():
    keys = [
        "xai-FAKE00000000abcd5678",
        "sk-proj-FAKE000000000000000000",
        "AIzaSyFAKE0000000000000000000",
        "eyJFAKEhdr0000000.eyJFAKEpyld0000.FAKEsig00000000",
    ]
    for key in keys:
        out = redact_secrets(f"error context {key} more text")
        assert key not in out, f"{key!r} survived redaction"
        assert "[REDACTED]" in out


def test_redacts_bearer_and_authorization_header_echo():
    out = redact_secrets("headers={'Authorization': 'Bearer sk-ant-api03-FAKE0000aaaa'}")
    assert "sk-ant-api03-FAKE0000aaaa" not in out
    assert "[REDACTED]" in out
    out2 = redact_secrets("x-api-key: sk-ant-api03-FAKE7777000000")
    assert "FAKE7777000000" not in out2
    assert "x-api-key" in out2  # header NAME preserved, value redacted


def test_preserves_innocuous_diagnostic_text():
    # The common, SAFE cloud 401 body must pass through untouched (keeps diagnostics).
    msg = "Error code: 401 - {'error': {'message': 'Incorrect API key provided'}}"
    assert redact_secrets(msg) == msg


def test_handles_none_and_exception_objects():
    assert redact_secrets(None) == ""
    out = redact_secrets(Exception("token sk-ant-api03-FAKE00000000eeee here"))
    assert "sk-ant-api03-FAKE00000000eeee" not in out
    assert "[REDACTED]" in out
