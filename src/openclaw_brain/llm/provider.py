"""LLM Provider — model catalog with runtime model selection and resilience.

Central factory that creates LangChain ChatModel instances from the
config catalog. Any pipeline stage can request a model by name, and
the provider returns the correct ChatModel with proper auth/endpoint.

When resilience is configured (fallback chains), the provider builds
ordered model lists for invoke_with_resilience() to iterate through.

Usage:
    provider = LLMProvider(config)

    # Get by catalog name
    llm = provider.get("claude-opus-4-6")

    # Get the default for a pipeline stage
    llm = provider.get_for_stage("extraction")

    # Get the full fallback chain for a stage (for resilient invocation)
    models = provider.get_chain("extraction")
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from openclaw_brain.config import BrainConfig, ModelEntry
from openclaw_brain.egress import (
    LOCAL_ONLY,
    _safe_identifier,
    check_url,
    effective_egress,
    violation_for_model,
)

logger = logging.getLogger(__name__)

# (stage, model_name) pairs already warned about by get_chain() below — module-level so the
# warning fires once per distinct gap for the life of the process, not once per get_chain() call
# (a hot pipeline stage would otherwise re-log the same typo'd fallback name on every chunk).
_warned_missing_fallbacks: set[tuple[str, str]] = set()

# Same once-per-process discipline, for the OTHER kind of chain gap: a fallback that IS in the
# catalog but is no longer a candidate (status != "active"). Keyed by (stage, name) like the set
# above — the same dead model configured in two stages is two distinct gaps, each worth a signal.
_warned_nonactive_fallbacks: set[tuple[str, str]] = set()

# Model names already warned about by get() for a non-active status. Keyed by NAME alone: a direct
# get() is not stage-scoped, and the point is one deprecation notice per model per process.
_warned_nonactive_models: set[str] = set()

# Statuses that make get() complain but still construct. `banned` is deliberately NOT here: its
# mechanism is the provider-keyed raise in get() (2026-07-22 anthropic ban), which fires first and
# is not weakened by, or dependent on, this field. See config.MODEL_STATUSES.
_WARN_ON_STATUSES = ("deprecated", "incompatible")


class LLMProviderError(Exception):
    """Raised when a model cannot be resolved or instantiated."""


def _reason_suffix(entry: ModelEntry) -> str:
    """` (reason)` when the entry carries a machine-readable status_reason, else "".

    The reason for a retirement normally lives as a one-line comment on the entry in
    config/default.toml (unreadable at runtime — TOML comments are not data), so this is
    usually empty and the warning points at the config instead. A deployment that wants the
    reason IN the log sets `status_reason` on the entry.
    """
    reason = getattr(entry, "status_reason", "") or ""
    return f" ({reason})" if reason else ""


class LLMProvider:
    """Factory for LangChain ChatModel instances from the config catalog.

    Lazily instantiates models — a model is only created when first
    requested, then cached for reuse.
    """

    # Maps pipeline stage names to config default fields
    _STAGE_DEFAULTS = {
        "extraction": "default_extraction",
        "reasoning": "default_reasoning",
        "matching": "default_matching",
        "vision": "default_vision",
    }

    # Maps pipeline stage names to resilience fallback fields
    _STAGE_FALLBACKS = {
        "extraction": "fallback_extraction",
        "reasoning": "fallback_reasoning",
        "matching": "fallback_matching",
    }

    # Stage → (config section attr, field) supplying that stage's OUTPUT-token bound.
    # Only the reasoning stage has one today: `[reasoning].output_token_budget` existed in
    # config.py since forever but NOTHING read it (dead config, verified 2026-07-25), so the
    # production reasoning model (deepseek-v4-flash / openrouter — no catalog max_tokens) ran
    # effectively UNBOUNDED. Extraction/matching deliberately stay untouched: their outputs are
    # schema-bounded structured objects, and no budget is declared for them.
    _STAGE_OUTPUT_BUDGETS = {
        "reasoning": ("reasoning", "output_token_budget"),
    }

    def __init__(self, config: BrainConfig):
        self._config = config
        self._cache: dict[str, BaseChatModel] = {}
        self._cache_policy: dict[str, str] = {}

    def _enforce_egress(self, entry: ModelEntry, *, reference: str) -> None:
        """Reject a non-local catalog entry before any client object is created."""
        policy = effective_egress(self._config)
        violation = violation_for_model(entry, reference=reference, policy=policy)
        if violation is not None:
            raise LLMProviderError(
                f"Egress policy violation: model={_safe_identifier(entry.name)!r}, provider={entry.provider!r}, "
                f"host={violation.host!r}, endpoint={violation.endpoint!r}, "
                f"policy={policy!r} ({violation.reason})"
            )

    def get(self, model_name: str, **kwargs: Any) -> BaseChatModel:
        """Get a ChatModel by catalog name.

        Args:
            model_name: Name from the model catalog (e.g. "claude-opus-4-6")
            **kwargs: Extra params passed to the ChatModel constructor
                      (e.g. temperature, max_tokens)

        Returns:
            A LangChain BaseChatModel instance. A catalog entry marked deprecated/incompatible
            is still constructed — an explicit request is honored — but logs a WARNING naming
            the status, once per model name per process.

        Raises:
            LLMProviderError: If the model is not in the catalog, or is an anthropic-API model
                (banned 2026-07-22 on cost) without OPENCLAW_ALLOW_ANTHROPIC_API=1.
        """
        cache_key = f"{model_name}:{_stable_hash(kwargs)}" if kwargs else model_name

        # Resolve and enforce on EVERY call, including a cache hit.  Otherwise a client cached
        # while policy=any could be returned after a runtime flip to local-only.
        entry = self._config.models.get_model(model_name)
        if not entry:
            available = [m.name for m in self._config.models.catalog]
            raise LLMProviderError(
                f"Model '{model_name}' not found in catalog. "
                f"Available: {available}"
            )
        self._enforce_egress(entry, reference=f"get({model_name!r})")

        policy = effective_egress(self._config)
        # A local client built under `any` may trust proxy environment variables. Rebuild it
        # when policy changes so a cached object cannot bypass the local-only HTTP transport.
        if (entry.provider == "local" and cache_key in self._cache and
                self._cache_policy.get(cache_key) != policy):
            self._cache.pop(cache_key)
        if cache_key not in self._cache:
            # 2026-07-22 Rick directive: anthropic API models are BANNED on cost. Mechanical
            # guard (not advisory): any code path — fallback chain, eval harness, MCP override —
            # that resolves an anthropic-provider model fails loudly here. Deliberate one-off
            # use requires OPENCLAW_ALLOW_ANTHROPIC_API=1 in the environment.
            if (entry.provider == "anthropic"
                    and os.environ.get("OPENCLAW_ALLOW_ANTHROPIC_API") != "1"):
                raise LLMProviderError(
                    f"Model '{model_name}' uses the anthropic API, which is banned "
                    f"(Rick, 2026-07-22, cost). Set OPENCLAW_ALLOW_ANTHROPIC_API=1 to "
                    f"override deliberately."
                )
            # A non-active catalog entry (deprecated/incompatible) still CONSTRUCTS here — code
            # that names a model explicitly means it, and this must not become a second, silent
            # router. But it never resolves silently: the staleness is stated once per process.
            # (`banned` is handled by the raise above; the status field only documents it.)
            if (entry.status in _WARN_ON_STATUSES
                    and model_name not in _warned_nonactive_models):
                _warned_nonactive_models.add(model_name)
                logger.warning(
                    "Model %r resolved from the catalog is marked status=%r%s — it is no longer "
                    "an active candidate (see the entry's comment in config/default.toml). "
                    "Constructing it anyway because it was requested by name.",
                    model_name, entry.status, _reason_suffix(entry),
                )
            factory_kwargs = dict(kwargs)
            if entry.provider == "local" and policy == LOCAL_ONLY:
                factory_kwargs["_egress_policy"] = policy
            self._cache[cache_key] = _create_chat_model(entry, **factory_kwargs)
            self._cache_policy[cache_key] = policy

        return self._cache[cache_key]

    def get_for_stage(
        self,
        stage: str,
        override: str | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Get the ChatModel for a pipeline stage.

        Uses the stage's configured default model unless an override
        is provided.

        Args:
            stage: Pipeline stage name ("extraction", "reasoning", "matching")
            override: Optional model name to use instead of the default
            **kwargs: Extra params for the ChatModel

        Returns:
            A LangChain BaseChatModel instance.
        """
        if override:
            return self.get(override, **self._stage_kwargs(stage, override, kwargs))

        attr = self._STAGE_DEFAULTS.get(stage)
        if not attr:
            raise LLMProviderError(
                f"Unknown stage '{stage}'. "
                f"Available: {list(self._STAGE_DEFAULTS.keys())}"
            )

        default_name = getattr(self._config.models, attr)
        return self.get(default_name, **self._stage_kwargs(stage, default_name, kwargs))

    def _stage_kwargs(
        self,
        stage: str,
        model_name: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Fold the stage's output-token budget into the ChatModel kwargs.

        PRECEDENCE (deliberate — 2026-07-25, wiring the previously-dead
        ``[reasoning].output_token_budget``). Highest wins:

        1. **Explicit caller kwarg** — ``get_chain("reasoning", max_tokens=N)``. A caller that
           states a bound means it; never second-guessed.
        2. **Catalog entry ``max_tokens``** (``config/default.toml [[models.catalog]]``) — a
           per-model bound that exists for a model-specific reason: the three local EXO
           reasoning models (DeepSeek-V3.2-4bit / GLM-5.1 / Kimi) IGNORE
           ``enable_thinking=False`` and run away without their 6000 bound (see
           ``_create_local``). A stage-wide budget must never silently WIDEN that to 8000,
           so when the entry declares one, nothing is injected here and ``_create_local``
           applies the entry value as before.
        3. **Stage budget** — applied ONLY when the entry declares no bound of its own. This
           is the case that matters in production: ``deepseek-v4-flash`` (openrouter, the
           reasoning + extraction default) declares no ``max_tokens``, so the reasoning call
           was unbounded until now.

        Stages with no entry in ``_STAGE_OUTPUT_BUDGETS`` (extraction, matching, vision) are
        returned untouched.
        """
        budget_ref = self._STAGE_OUTPUT_BUDGETS.get(stage)
        if budget_ref is None:
            return kwargs
        if "max_tokens" in kwargs:  # precedence 1 — caller wins
            return kwargs

        entry = self._config.models.get_model(model_name)
        if entry is None or getattr(entry, "max_tokens", None):
            # precedence 2 — the entry's own bound wins (or the name is unknown, in which
            # case get() raises the catalog error and the budget is moot).
            return kwargs

        section, field = budget_ref
        budget = getattr(getattr(self._config, section, None), field, None)
        if not isinstance(budget, int) or budget <= 0:
            return kwargs
        return {**kwargs, "max_tokens": budget}  # precedence 3 — stage budget

    def get_chain(
        self,
        stage: str,
        override: str | None = None,
        **kwargs: Any,
    ) -> list[BaseChatModel]:
        """Get the ordered fallback chain for a pipeline stage.

        Returns [primary, fallback_1, fallback_2, ...] as instantiated
        ChatModel objects. If no fallback chain is configured, returns
        just the primary model.

        A configured fallback is DROPPED (with a warning naming the gap, once per stage+name)
        when it is missing from the catalog, or present but non-active — the chain is a list of
        models the pipeline may fall onto unattended, so a retired one has no business in it.
        The primary is never dropped: which model a stage uses is a decision made in config,
        not here.

        Args:
            stage: Pipeline stage name.
            override: If set, returns only this model (no fallbacks).
            **kwargs: Extra params for all models.

        Returns:
            Ordered list of ChatModel instances.
        """
        if override:
            return [self.get(override, **self._stage_kwargs(stage, override, kwargs))]

        # Primary model
        primary = self.get_for_stage(stage, **kwargs)
        chain = [primary]

        # Fallback models from resilience config
        fallback_attr = self._STAGE_FALLBACKS.get(stage)
        if fallback_attr:
            fallback_names = getattr(self._config.resilience, fallback_attr, [])
            for name in fallback_names:
                # A fallback that IS in the catalog but is no longer a candidate
                # (deprecated/banned/incompatible) is skipped BEFORE construction — a resilience
                # chain is a list of things the pipeline may silently fall onto unattended, so a
                # retired model must never be one of them. Same visible-gap treatment as the
                # missing-name branch below: the chain really is shorter than configured, and
                # says so once per (stage, name). Direct get(name) is unaffected.
                entry = self._config.models.get_model(name)
                if entry is not None:
                    # Do this before the status-based skip and before get()'s catch below.  A
                    # local-only violation is a policy failure, never an unavailable fallback
                    # that may be silently shortened out of the chain.
                    self._enforce_egress(
                        entry,
                        reference=f"[resilience].{fallback_attr}",
                    )
                if entry is not None and entry.status != "active":
                    warn_key = (stage, name)
                    if warn_key not in _warned_nonactive_fallbacks:
                        _warned_nonactive_fallbacks.add(warn_key)
                        logger.warning(
                            "Configured fallback model %r for stage=%r is marked status=%r%s — "
                            "SKIPPED, so the chain is shorter than configured (a non-active model "
                            "must not be fallen onto unattended; re-activating it is an "
                            "A/B-with-cost decision, not a config edit)",
                            name, stage, entry.status, _reason_suffix(entry),
                        )
                    continue
                try:
                    # Fallbacks serve the SAME stage as the primary, so they get the same
                    # budget resolution (per-model precedence still applies: a local exo
                    # fallback keeps its own catalog bound).
                    model = self.get(name, **self._stage_kwargs(stage, name, kwargs))
                    chain.append(model)
                except LLMProviderError:
                    # Skip unavailable fallback models — but make the gap visible (once per
                    # stage+name) instead of silently shortening the chain forever. A hand-edited
                    # config/default.toml (or a shared-service deployment's own copy) with a
                    # typo'd/renamed fallback name would otherwise drop a fallback with zero
                    # signal at the point of the drop.
                    warn_key = (stage, name)
                    if warn_key not in _warned_missing_fallbacks:
                        _warned_missing_fallbacks.add(warn_key)
                        logger.warning(
                            "Configured fallback model %r for stage=%r not found in catalog — "
                            "chain is silently shorter than configured (check config/default.toml "
                            "[resilience] fallback_%s for a typo'd or removed model name)",
                            name, stage, stage,
                        )

        return chain

    def list_available(self) -> list[dict[str, str]]:
        """List all models in the catalog, each with its standing.

        This feeds `BrainAgent.get_stats()["models"]` → the `brain://models` MCP resource,
        i.e. it is the ONE channel an outside agent reads the catalog through. Omitting
        `status` there would leave that agent unable to tell a retired entry from a live
        one — the exact folklore the field exists to end — so it is always present.
        `status_reason` is included only when non-empty (the usual reason channel is the
        entry's comment in config/default.toml, which is not readable at runtime).
        """
        listing = []
        for m in self._config.models.catalog:
            row = {
                "name": m.name,
                "provider": m.provider,
                "tier": m.tier,
                "model_id": m.model_id,
                "status": m.status,
            }
            if m.status_reason:
                row["status_reason"] = m.status_reason
            listing.append(row)
        return listing

    def clear_cache(self) -> None:
        """Clear the model instance cache."""
        self._cache.clear()
        self._cache_policy.clear()


def _create_chat_model(entry: ModelEntry, **kwargs: Any) -> BaseChatModel:
    """Instantiate the appropriate LangChain ChatModel for a catalog entry."""
    if entry.provider == "anthropic":
        return _create_anthropic(entry, **kwargs)
    elif entry.provider == "openai":
        return _create_openai(entry, **kwargs)
    elif entry.provider == "google":
        return _create_google(entry, **kwargs)
    elif entry.provider == "xai":
        return _create_xai(entry, **kwargs)
    elif entry.provider == "openrouter":
        return _create_openrouter(entry, **kwargs)
    elif entry.provider == "local":
        policy = kwargs.pop("_egress_policy", None)
        return _create_local(entry, _egress_policy=policy, **kwargs)
    else:
        raise LLMProviderError(
            f"Unknown provider '{entry.provider}' for model '{entry.name}'. "
            f"Supported: anthropic, openai, google, xai, openrouter, local"
        )


def _create_anthropic(entry: ModelEntry, **kwargs: Any) -> BaseChatModel:
    """Create a ChatAnthropic instance."""
    from langchain_anthropic import ChatAnthropic

    params: dict[str, Any] = {"model": entry.model_id}
    params.update(kwargs)
    return ChatAnthropic(**params)


def _create_openai(entry: ModelEntry, **kwargs: Any) -> BaseChatModel:
    """Create a ChatOpenAI instance."""
    import os

    from langchain_openai import ChatOpenAI

    params: dict[str, Any] = {"model": entry.model_id}
    if entry.endpoint:
        params["base_url"] = entry.endpoint
    if not os.environ.get("OPENAI_API_KEY"):
        params.setdefault("api_key", "not-set")
    params.update(kwargs)
    return ChatOpenAI(**params)


def _create_google(entry: ModelEntry, **kwargs: Any) -> BaseChatModel:
    """Create a ChatGoogleGenerativeAI instance."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    params: dict[str, Any] = {"model": entry.model_id}
    params.update(kwargs)
    return ChatGoogleGenerativeAI(**params)


def _create_xai(entry: ModelEntry, **kwargs: Any) -> BaseChatModel:
    """Create a ChatOpenAI instance pointing to xAI's OpenAI-compatible endpoint."""
    import os
    from langchain_openai import ChatOpenAI

    params: dict[str, Any] = {
        "model": entry.model_id,
        "base_url": entry.endpoint or "https://api.x.ai/v1",
        "api_key": os.environ.get("XAI_API_KEY", "not-set"),
    }
    params.update(kwargs)
    return ChatOpenAI(**params)


def _create_openrouter(entry: ModelEntry, **kwargs: Any) -> BaseChatModel:
    """Create a ChatOpenAI instance pointing to OpenRouter's OpenAI-compatible endpoint.

    OpenRouter aggregates many providers (deepseek/glm/kimi/...). Key resolved into OPENROUTER_API_KEY
    by auth.inject_api_keys (from the macOS keychain entry 'openrouter-api-key').
    """
    import os
    from langchain_openai import ChatOpenAI

    params: dict[str, Any] = {
        "model": entry.model_id,
        "base_url": entry.endpoint or "https://openrouter.ai/api/v1",
        "api_key": os.environ.get("OPENROUTER_API_KEY", "not-set"),
    }
    params.update(kwargs)
    return ChatOpenAI(**params)


def _omlx_api_key() -> str:
    """Bearer key for local oMLX endpoints, resolved at call time.

    Order: ``OMLX_API_KEY`` (the value) -> ``OMLX_API_KEY_FILE`` (a path; a JSON file
    yields ``.auth.api_key`` only, e.g. ``~/.omlx/settings.json``; otherwise the first
    line) -> the historical placeholder ``"not-needed"`` (Ollama / keyless servers).
    The key itself never reaches a log line or an exception message — failures log the
    exception *type* only (PHILOSOPHY P9: visible, but no secret in the record).
    """
    key = os.environ.get("OMLX_API_KEY", "").strip()
    if key:
        return key
    path = os.environ.get("OMLX_API_KEY_FILE", "").strip()
    if path:
        try:
            raw = Path(path).expanduser().read_text(encoding="utf-8").strip()
            if raw.startswith("{"):
                data = json.loads(raw)
                auth = data.get("auth") if isinstance(data, dict) else None
                val = auth.get("api_key") if isinstance(auth, dict) else None
                key = val.strip() if isinstance(val, str) else ""
            else:
                key = raw.splitlines()[0].strip() if raw else ""
        except Exception as exc:  # noqa: BLE001 — report the type, never the content
            logger.warning("OMLX_API_KEY_FILE could not be read (%s); using placeholder key",
                           type(exc).__name__)
            key = ""
        if key:
            return key
        logger.warning("OMLX_API_KEY_FILE is set but holds no key; using placeholder key")
    return "not-needed"


def _create_local(entry: ModelEntry, *, _egress_policy: str | None = None,
                  **kwargs: Any) -> BaseChatModel:
    """Create a ChatOpenAI instance pointing to a local endpoint (oMLX/Ollama)."""
    from langchain_openai import ChatOpenAI

    params: dict[str, Any] = {
        "model": entry.model_id,
        "base_url": entry.endpoint or "http://localhost:8000/v1",
        # oMLX servers may require a bearer key (rick :8000 does since the 2026-09 oMLX
        # hardening); an unauthenticated call fails with 401 and the chain leaves the local
        # tier for its fallback. See _omlx_api_key() for the resolution order.
        "api_key": _omlx_api_key(),
        # Qwen3.x thinking mode generates thousands of hidden reasoning tokens per
        # call, which breaks structured-output budgets and made pipeline chunks take
        # ~36 min each (observed 2026-06-11). Ollama ignores unknown extra_body keys.
        # NOTE: EXO reasoning models (DeepSeek-V3.2 / Kimi) IGNORE this flag and always
        # reason — they need an explicit max_tokens bound (catalog field) to avoid runaway.
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if getattr(entry, "max_tokens", None):
        params["max_tokens"] = entry.max_tokens
    # kwargs still win over the entry here (an explicit caller bound is deliberate), but a
    # STAGE budget never reaches this line when the entry declares max_tokens —
    # LLMProvider._stage_kwargs skips injection in that case precisely so the exo models'
    # 6000 runaway-guard cannot be silently widened by a stage-wide default.
    params.update(kwargs)
    if (_egress_policy or effective_egress()) == LOCAL_ONLY:
        import httpx

        # Directly injected SDK clients and proxy settings bypass our request hook.
        for key in ("client", "async_client", "root_client", "root_async_client",
                    "http_client", "http_async_client", "openai_proxy"):
            if params.get(key) is not None:
                raise LLMProviderError(f"local-only does not allow {key} override")
        check_url(str(params["base_url"]), policy=LOCAL_ONLY)

        def guard_request(request: httpx.Request) -> None:
            check_url(str(request.url), policy=LOCAL_ONLY)

        async def guard_async_request(request: httpx.Request) -> None:
            check_url(str(request.url), policy=LOCAL_ONLY)

        params["http_client"] = httpx.Client(
            trust_env=False, follow_redirects=False,
            event_hooks={"request": [guard_request]},
        )
        params["http_async_client"] = httpx.AsyncClient(
            trust_env=False, follow_redirects=False,
            event_hooks={"request": [guard_async_request]},
        )
    return ChatOpenAI(**params)


def _stable_hash(d: dict) -> str:
    """Simple deterministic hash for kwargs dict."""
    if not d:
        return ""
    items = sorted(d.items(), key=lambda x: x[0])
    return str(hash(tuple((k, str(v)) for k, v in items)))
