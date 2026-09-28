"""Resolve API keys from environment variables or a Brain-owned auth store.

The optional store defaults to:
    ~/.config/openclaw-brain/auth-profiles.json

Set ``OPENCLAW_BRAIN_AUTH_FILE`` to use another explicit file. Nothing is
implicitly read from a host agent such as OpenClaw, Claude Code, or Codex.

Also handles OAuth token refresh for providers that use JWT tokens
(e.g., OpenAI Codex with OAuth flow).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from openclaw_brain.redaction import redact_secrets

logger = logging.getLogger(__name__)


# Resolution returns SecretStr, NOT str, so a resolved credential can never be
# rendered in plaintext by repr()/str()/logging/an assertion message — repr is
# "SecretStr('**********')". This is a structural close of the leak CLASS: callers
# must call .get_secret_value() to reveal (a greppable, intentional act), so no
# accidental print/log/test-assert can spill the value. The single sanctioned
# reveal point is inject_api_keys() writing into os.environ (env vars must be str).


def _mask(value: str) -> str:
    """Short, non-reversible preview for logging an injection summary."""
    return f"{value[:12]}...{value[-4:]}" if len(value) > 16 else "****"


def get_api_key(provider: str) -> SecretStr | None:
    """Get an API key for a provider, checking env vars first, then Brain config.

    Args:
        provider: "anthropic", "openai", or "google"

    Returns:
        A SecretStr wrapping the key (reveal with .get_secret_value()), or None.
    """
    # 1. Environment variable takes priority
    env_key = _env_key_for_provider(provider)
    if env_key:
        val = os.environ.get(env_key)
        if val:
            return SecretStr(val)

    # 2. Fall back to the optional Brain-owned auth store
    return _from_brain_auth(provider)


def _env_key_for_provider(provider: str) -> str | None:
    return {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "xai": "XAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(provider)


def _from_brain_auth(provider: str) -> SecretStr | None:
    """Read credentials from Brain's optional auth-profiles.json."""
    auth_file = _find_auth_file()
    if not auth_file:
        return None

    try:
        data = json.loads(auth_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    profiles = data.get("profiles", {})

    # Try provider:default profile
    profile = profiles.get(f"{provider}:default")
    if not profile:
        # Try openai-codex:default for openai provider
        if provider == "openai":
            profile = profiles.get("openai-codex:default")
        if not profile:
            return None

    profile_type = profile.get("type", "")

    if profile_type == "api_key":
        # Inline key (legacy) takes priority; otherwise resolve a keyRef
        # (Brain profiles may point at a macOS-keychain SecretRef resolver).
        inline = profile.get("key")
        if inline:
            return SecretStr(inline)
        key_ref = profile.get("keyRef")
        if key_ref:
            return _resolve_secret_ref(key_ref, auth_file)
        return None

    if profile_type == "oauth":
        access = profile.get("access")
        # Valid or expired, return it; caller may handle refresh on 401.
        return SecretStr(access) if access else None

    if profile_type == "token":
        token = profile.get("token")
        return SecretStr(token) if token else None

    return None


# Deprecated internal name for source compatibility. It no longer reads an
# OpenClaw home; resolution is Brain-owned and harness-neutral.
_from_openclaw_auth = _from_brain_auth


def _find_auth_file() -> Path | None:
    """Find an explicitly configured or Brain-owned auth-profiles.json file."""
    explicit = os.environ.get("OPENCLAW_BRAIN_AUTH_FILE")
    candidates = ([Path(explicit).expanduser()] if explicit else []) + [
        Path.home() / ".config" / "openclaw-brain" / "auth-profiles.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


# ── SecretRef resolution (source: exec, e.g. macOS keychain) ──
#
# Brain auth profiles may store API keys as keyRefs:
#   {"source": "exec", "provider": "macos-keychain",
#    "id": "openclaw/auth-profiles/anthropic/default/key"}
# The exec resolver is declared in the Brain-owned secret-providers.json under
#   secrets.providers.<provider> = {"source": "exec", "command": "<path>", "jsonOnly": true}
# and speaks a tiny JSON protocol over stdio:
#   stdin :  {"ids": ["<id>", ...]}
#   stdout:  {"protocolVersion": 1, "values": {"<id>": "<secret>"}, "errors": {...}}
# We read the command FROM the config (not hardcoded) so we follow OpenClaw if it
# changes the resolver. Resolved secrets are never logged.


def _find_secret_config(auth_file: Path | None) -> Path | None:
    """Locate the harness-neutral secrets-resolver declaration."""
    candidates: list[Path] = []
    explicit = os.environ.get("OPENCLAW_BRAIN_SECRET_CONFIG")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if auth_file is not None:
        candidates.append(auth_file.parent / "secret-providers.json")
    candidates.append(Path.home() / ".config" / "openclaw-brain" / "secret-providers.json")
    for path in candidates:
        if path.is_file():
            return path
    return None


def _find_openclaw_config(auth_file: Path | None) -> Path | None:
    """Deprecated internal alias retained for test/source compatibility."""
    return _find_secret_config(auth_file)


def _exec_resolver_command(provider: str, config_path: Path) -> list[str] | None:
    """Read the exec-resolver argv for `provider` from the secrets config."""
    import shlex

    try:
        cfg = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    spec = (
        cfg.get("secrets", {})
        .get("providers", {})
        .get(provider, {})
    )
    if spec.get("source") != "exec":
        return None
    command = spec.get("command")
    if not command:
        return None
    argv = command if isinstance(command, list) else shlex.split(command)
    return argv or None


def _resolve_secret_ref(key_ref: dict[str, Any], auth_file: Path | None) -> SecretStr | None:
    """Resolve a {source:exec, provider, id} keyRef with Brain's configured resolver.

    Returns the secret (as SecretStr), or None if anything fails (missing config,
    resolver error, malformed output). Never raises, never logs the secret.
    """
    import subprocess

    if not isinstance(key_ref, dict) or key_ref.get("source") != "exec":
        return None
    provider = key_ref.get("provider")
    ref_id = key_ref.get("id")
    if not provider or not ref_id:
        return None

    config_path = _find_openclaw_config(auth_file)
    if not config_path:
        return None
    argv = _exec_resolver_command(provider, config_path)
    if not argv:
        return None

    request = json.dumps({"ids": [ref_id]})

    def _run(cmd: list[str]) -> subprocess.CompletedProcess | None:
        try:
            return subprocess.run(
                cmd, input=request, capture_output=True, text=True, timeout=15
            )
        except (OSError, subprocess.SubprocessError):
            return None

    result = _run(argv)
    # If the resolver script isn't directly executable (no +x / ENOEXEC) but is a
    # .js file, retry via node — mirrors how OpenClaw would spawn it.
    if (result is None or result.returncode != 0) and argv and argv[0].endswith(".js"):
        node = _which_node()
        if node:
            result = _run([node, *argv])
    if result is None or result.returncode != 0:
        if result is not None and result.stderr:
            # stderr carries the security(1) diagnostic (e.g. errSecItemNotFound),
            # never the secret (that goes to stdout) — cap length + redact as defence anyway.
            logger.warning(
                "SecretRef resolver failed for %s: %s",
                provider, redact_secrets(result.stderr.strip()[:200]),
            )
        return None

    try:
        resp = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    value = resp.get("values", {}).get(ref_id)
    if value:
        return SecretStr(value)
    err = resp.get("errors", {}).get(ref_id)
    if err:
        logger.warning("SecretRef resolver could not resolve %s: %s", ref_id, err.get("message"))
    return None


def _which_node() -> str | None:
    """Find a node interpreter (PATH first, then the common Homebrew location)."""
    import shutil

    return shutil.which("node") or next(
        (p for p in ("/opt/homebrew/bin/node", "/usr/local/bin/node") if Path(p).is_file()),
        None,
    )


def inject_api_keys() -> dict[str, str]:
    """Inject discovered API keys into environment variables.

    Returns a dict of what was injected.
    """
    injected = {}
    for provider, env_var in [
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
        ("google", "GOOGLE_API_KEY"),
        ("xai", "XAI_API_KEY"),
    ]:
        if not os.environ.get(env_var):
            secret = _from_brain_auth(provider)
            if secret:
                value = secret.get_secret_value()  # sanctioned reveal: env vars must be str
                if value:
                    os.environ[env_var] = value
                    injected[env_var] = _mask(value)
    # OpenRouter key may live in the macOS keychain ('openrouter-api-key').
    if not os.environ.get("OPENROUTER_API_KEY"):
        ork = _openrouter_from_keychain()
        if ork:
            os.environ["OPENROUTER_API_KEY"] = ork
            injected["OPENROUTER_API_KEY"] = _mask(ork)
    return injected


def _openrouter_from_keychain() -> str | None:
    """Read the OpenRouter key from the macOS keychain generic-password 'openrouter-api-key'
    (service set up by ai-litellm-fabric; account = current user). Returns None if absent."""
    import getpass
    import subprocess

    try:
        r = subprocess.run(
            ["/usr/bin/security", "find-generic-password", "-s", "openrouter-api-key",
             "-a", getpass.getuser(), "-w"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    val = (r.stdout or "").strip()
    return val or None


# ── OAuth Token Refresh ──


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write `data` as JSON to `path` atomically.

    Writes to a temp file in the same directory (so the final `os.replace()` is a
    same-filesystem rename, not a cross-filesystem copy), fsyncs it so the bytes are
    durable before the rename is visible, then `os.replace()`s it over `path`.
    `os.replace()` is atomic on POSIX: a reader (or another writer's fresh read)
    always sees either the fully-old or fully-new file, never a truncated/partial
    one, and there is no window where the target path is missing.

    Raises OSError on any failure; on failure the temp file is removed rather than
    left behind.
    """
    directory = path.parent
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(data, indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def refresh_oauth_token(provider: str = "openai", *, egress: str | None = None) -> bool:
    """Refresh an expired OAuth token and update the auth store + env var.

    Reads the refresh_token from auth-profiles.json, calls the token endpoint, and
    writes back the new access/refresh tokens.

    The auth file can be shared by multiple Brain processes, so this does a
    profile-scoped atomic update rather than a whole-file read-modify-write: the
    initial read below only decides WHETHER to refresh and supplies the refresh
    token; after the (potentially slow) network round-trip, the file is re-read
    immediately before writing, and only this profile's changed keys (`access`,
    `refresh`, `expires`) are merged into that fresh copy. A concurrent writer's
    changes to any other profile — or to this same profile, made while our network
    call was in flight — are therefore never clobbered. The write itself goes
    through `_atomic_write_json()` (temp file + fsync + `os.replace()`), so a crash
    mid-write can never leave a truncated auth-profiles.json.

    Args:
        provider: The provider to refresh ("openai").

    Returns:
        True if refresh succeeded, False otherwise.
    """
    auth_file = _find_auth_file()
    if not auth_file:
        logger.warning("No auth file found for token refresh")
        return False

    try:
        data = json.loads(auth_file.read_text())
    except (json.JSONDecodeError, OSError):
        return False

    profiles = data.get("profiles", {})

    # Find the OAuth profile
    profile_key = f"{provider}:default"
    if profile_key not in profiles and provider == "openai":
        profile_key = "openai-codex:default"
    if profile_key not in profiles:
        return False

    profile = profiles[profile_key]
    if profile.get("type") != "oauth":
        return False

    refresh_token = profile.get("refresh")
    if not refresh_token:
        logger.warning("No refresh token available for %s", provider)
        return False

    # Determine token endpoint based on provider
    token_url = _token_endpoint_for_provider(provider)
    if not token_url:
        return False

    try:
        if egress is None:
            new_tokens = _do_token_refresh(token_url, refresh_token)
        else:
            new_tokens = _do_token_refresh(token_url, refresh_token, egress=egress)
    except Exception as e:
        logger.error("Token refresh failed for %s: %s", provider, e)
        return False

    if not new_tokens or "access_token" not in new_tokens:
        logger.error("Token refresh returned no access_token")
        return False

    # Only these keys get merged into the profile — never the rest of `data`/
    # `profiles` captured by the read above, which may now be stale.
    new_values: dict[str, Any] = {"access": new_tokens["access_token"]}
    if "refresh_token" in new_tokens:
        new_values["refresh"] = new_tokens["refresh_token"]
    if "expires_in" in new_tokens:
        # Convert seconds-from-now to epoch milliseconds
        new_values["expires"] = int((time.time() + new_tokens["expires_in"]) * 1000)

    # Re-read immediately before writing: the network round-trip above may have
    # taken long enough for another process to have written this same file.
    try:
        fresh_data = json.loads(auth_file.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to re-read auth file before write: %s", e)
        return False

    fresh_profiles = fresh_data.setdefault("profiles", {})
    # Fall back to the profile dict as it looked at the top of this call if it's
    # gone from the fresh read (e.g. a concurrent delete) — merges our refreshed
    # keys onto a real template (preserving `type`/`provider`) instead of
    # fabricating a bare dict that would fail the `type == "oauth"` check next time.
    fresh_profile = fresh_profiles.get(profile_key, dict(profile))
    fresh_profile.update(new_values)
    fresh_profiles[profile_key] = fresh_profile
    fresh_data["profiles"] = fresh_profiles

    try:
        _atomic_write_json(auth_file, fresh_data)
    except OSError as e:
        logger.error("Failed to write updated auth file: %s", e)
        return False

    # Update environment variable
    env_key = _env_key_for_provider(provider)
    if env_key:
        os.environ[env_key] = new_tokens["access_token"]

    logger.info("OAuth token refreshed for %s", provider)
    return True


def create_auth_refresh_callback(
    llm_provider: Any = None,
) -> callable:
    """Create a callback for resilience layer to call on 401 errors.

    After a successful token refresh, clears the LLMProvider's model cache
    so new requests pick up the fresh token (ChatModel instances capture
    the API key at init time).

    Args:
        llm_provider: Optional LLMProvider whose cache to clear after refresh.

    Returns a function that attempts to refresh OAuth tokens for all
    known providers and returns True if any succeeded.
    """
    def _refresh() -> bool:
        refreshed = False
        egress = None
        config = getattr(llm_provider, "_config", None)
        from openclaw_brain.config import BrainConfig

        if isinstance(config, BrainConfig):
            from openclaw_brain.egress import effective_egress

            egress = effective_egress(config)
        for provider in ("openai",):  # Only OpenAI uses OAuth currently
            if (refresh_oauth_token(provider) if egress is None
                    else refresh_oauth_token(provider, egress=egress)):
                refreshed = True
        if refreshed and llm_provider is not None:
            llm_provider.clear_cache()
            logger.info("Cleared LLM provider cache after token refresh")
        return refreshed

    return _refresh


def _token_endpoint_for_provider(provider: str) -> str | None:
    """Return the OAuth token endpoint URL for a provider."""
    endpoints = {
        "openai": "https://auth.openai.com/oauth/token",
    }
    return endpoints.get(provider)


def _do_token_refresh(
    token_url: str,
    refresh_token: str,
    *,
    egress: str | None = None,
) -> dict[str, Any]:
    """Perform the actual HTTP token refresh request.

    Uses urllib to avoid adding httpx/requests as a dependency.
    """
    import urllib.parse
    import urllib.request

    from openclaw_brain.egress import check_url, effective_egress

    # Guard before Request construction/urlopen so a blocked destination cannot emit traffic.
    active_egress = effective_egress() if egress is None else egress
    check_url(token_url, policy=active_egress)

    body = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }).encode()

    req = urllib.request.Request(
        token_url,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    if active_egress == "local-only":
        class GuardedRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, request, fp, code, msg, headers, newurl):
                check_url(newurl, policy=active_egress)
                return super().redirect_request(request, fp, code, msg, headers, newurl)

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), GuardedRedirect())
        response = opener.open(req, timeout=10)
    else:
        response = urllib.request.urlopen(req, timeout=10)
    with response as resp:
        return json.loads(resp.read())
