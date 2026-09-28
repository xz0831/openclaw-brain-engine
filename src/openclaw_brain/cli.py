"""OpenClaw Brain CLI entry point."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

from openclaw_brain.config import load_config
from openclaw_brain.egress import effective_egress


@click.group()
@click.option("--config", "config_path", default=None, help="Path to config TOML file")
@click.pass_context
def main(ctx: click.Context, config_path: str | None) -> None:
    """OpenClaw Brain — LangGraph-powered agent runtime."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_path)
    if ctx.invoked_subcommand != "doctor":
        from openclaw_brain.egress import enforce_startup_egress
        enforce_startup_egress(ctx.obj["config"])


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show Neo4j connection status and graph statistics."""
    asyncio.run(_status(ctx.obj["config"]))


async def _status(config):
    from openclaw_brain.knowledge.graph.store import GraphStore

    store = GraphStore(config.neo4j, egress=effective_egress(config))
    try:
        await store.connect()
        click.echo("Neo4j: connected")
        stats = await store.get_stats()
        if store.vector_index_errors:
            click.echo(f"WARNING: vector-index failures this session: {store.vector_index_errors} "
                       f"(matching degraded to text-only on each)")
        if stats:
            click.echo("Graph statistics:")
            for label, count in stats.items():
                click.echo(f"  {label}: {count}")
        else:
            click.echo("Graph is empty.")
    except Exception as e:
        click.echo(f"Neo4j: connection failed — {e}")
        raise SystemExit(1)
    finally:
        await store.close()


@main.command()
@click.option("--apply", "apply_", is_flag=True, help="Apply merges (default: dry-run report only)")
@click.option("--auto-merge", is_flag=True,
              help="Merge the auto band (default OFF: auto band goes to the review queue)")
@click.option("--queue-path", default="~/.local/state/openclaw-brain/merge_queue.jsonl",
              help="JSONL review-queue path")
@click.pass_context
def consolidate(ctx: click.Context, apply_: bool, auto_merge: bool, queue_path: str) -> None:
    """Find (and optionally merge) duplicate Concept nodes."""
    asyncio.run(_consolidate(ctx.obj["config"], apply_, auto_merge, queue_path))


async def _consolidate(config, apply_: bool, auto_merge: bool, queue_path: str):
    from openclaw_brain.journal import ActionJournal
    from openclaw_brain.knowledge.consolidation import ConsolidationEngine
    from openclaw_brain.knowledge.graph.store import GraphStore

    store = GraphStore(config.neo4j, egress=effective_egress(config))
    try:
        await store.connect()
        journal = ActionJournal(config.state_path)
        # Wire a local verifier to drain the embedding-candidate band. The evidenced best judge
        # was qwen3.6-27b (band agreement 0.90 vs gemini 0.85 —
        # experiments/CANONICALIZATION_PROMOTION.md), but the user deleted every local Qwen3.5/3.6
        # model from the oMLX host (2026-08-19/20). deepseek-v4-flash-local is an UNBENCHED-in-role
        # stopgap (it IS the bench-confirmed extraction/reasoning default, but was never scored as
        # a verify judge): re-score on experiments/er_gold_adjudicated_827.json — alongside the
        # new Qwen3.8-27B candidates — before trusting the VERIFY band in a live (--apply) run.
        from openclaw_brain.auth import inject_api_keys
        from openclaw_brain.llm.provider import LLMProvider
        from openclaw_brain.knowledge.reasoning.verifier import MatchVerifier

        if effective_egress(config) != "local-only":
            inject_api_keys()
        verifier = MatchVerifier(LLMProvider(config), config, model_override="deepseek-v4-flash-local")
        engine = ConsolidationEngine(store, config, journal=journal, verifier=verifier)
        report = await engine.run(
            dry_run=not apply_, auto_merge=auto_merge, queue_path=queue_path,
        )
        click.echo("Consolidation report:")
        for k, v in report.summary().items():
            click.echo(f"  {k}: {v}")
        for c in (report.auto + report.verify + report.review)[:30]:
            cos = f"{c.cos:.3f}" if c.cos is not None else "  —  "
            click.echo(f"  [{c.band:6s}] cos={cos} ns={c.name_sim:.2f}  "
                       f"{c.a_name!r} ↔ {c.b_name!r}  ({c.reason})")
        if not apply_:
            click.echo("(dry run — re-run with --apply to write the queue"
                       " / --apply --auto-merge to merge the auto band)")
    except Exception as e:
        click.echo(f"Consolidation failed: {e}")
        raise SystemExit(1)
    finally:
        await store.close()


@main.command("apply-schema")
@click.pass_context
def apply_schema(ctx: click.Context) -> None:
    """Apply Neo4j schema constraints and indexes."""
    asyncio.run(_apply_schema(ctx.obj["config"]))


async def _apply_schema(config):
    from openclaw_brain.config import get_schema_path
    from openclaw_brain.knowledge.graph.store import GraphStore

    schema_path = get_schema_path()
    store = GraphStore(config.neo4j, egress=effective_egress(config))
    try:
        await store.connect()
        await store.apply_schema(schema_path)
        click.echo(f"Schema applied from {schema_path}")
    except Exception as e:
        click.echo(f"Schema application failed: {e}")
        raise SystemExit(1)
    finally:
        await store.close()


@main.command("migrate-equation-latex")
@click.option("--apply", "apply_", is_flag=True,
              help="Write canonical_latex := latex for recoverable Equation nodes "
                   "(default: dry-run report only)")
@click.pass_context
def migrate_equation_latex(ctx: click.Context, apply_: bool) -> None:
    """One-time recovery for Equation nodes committed before apply_delta learned to map
    the reasoner's 'latex' property onto the schema's 'canonical_latex' field — those nodes
    have LaTeX under 'latex' but an empty canonical_latex. Additive & idempotent: never
    touches an already-populated canonical_latex, never modifies 'latex'; re-running --apply
    after a successful run writes 0. Dry-run by default."""
    asyncio.run(_migrate_equation_latex(ctx.obj["config"], apply_))


async def _migrate_equation_latex(config, apply_: bool) -> None:
    from openclaw_brain.knowledge.graph.store import GraphStore

    store = GraphStore(config.neo4j, egress=effective_egress(config))
    try:
        await store.connect()
        count = await store.migrate_equation_latex(apply=apply_)
        if apply_:
            click.echo(f"migrate-equation-latex — APPLIED: {count} Equation node(s) updated "
                       f"(canonical_latex := latex).")
        else:
            click.echo(f"migrate-equation-latex — DRY-RUN: {count} Equation node(s) recoverable "
                       f"(populated 'latex', empty 'canonical_latex'). Re-run with --apply to write.")
    except Exception as e:
        click.echo(f"migrate-equation-latex failed: {e}")
        raise SystemExit(1)
    finally:
        await store.close()


@main.command("dedupe-concepts")
@click.option("--source-id", "source_id", default=None,
              help="Scope to one source (SAFE: same source_id + same name = true duplicate). "
                   "Omit to dedupe globally by name — merges ACROSS sources, collapsing provenance.")
@click.option("--apply", "apply_", is_flag=True,
              help="Merge each exact-name duplicate group into its highest-degree node "
                   "(default: dry-run count only)")
@click.pass_context
def dedupe_concepts(ctx: click.Context, source_id: str | None, apply_: bool) -> None:
    """Merge Concept nodes sharing an exact canonical_name into one (highest-degree survivor,
    edges folded via APOC). Cleans up exact-name duplicates left by a non-deterministic
    re-chunk/reprocess. Additive-safe (no node loses a connection) & idempotent. Requires APOC.
    Dry-run by default. Scope with --source-id to stay intra-source (recommended)."""
    asyncio.run(_dedupe_concepts(ctx.obj["config"], source_id, apply_))


async def _dedupe_concepts(config, source_id: str | None, apply_: bool) -> None:
    from openclaw_brain.knowledge.graph.store import GraphStore

    store = GraphStore(config.neo4j, egress=effective_egress(config))
    scope = f"source_id={source_id}" if source_id else "ALL sources (cross-source)"
    try:
        await store.connect()
        count = await store.dedupe_exact_concepts(source_id=source_id, apply=apply_)
        if apply_:
            click.echo(f"dedupe-concepts — APPLIED ({scope}): {count} duplicate group(s) merged.")
        else:
            click.echo(f"dedupe-concepts — DRY-RUN ({scope}): {count} exact-name duplicate "
                       f"group(s) found. Re-run with --apply to merge.")
    except Exception as e:
        click.echo(f"dedupe-concepts failed: {e}")
        raise SystemExit(1)
    finally:
        await store.close()


def _doctor_checks(full: bool = False) -> list[tuple[str, bool, str]]:
    """Environment-drift checks for the fragile ML stack (see pyproject's ingest extra).

    Each check targets a FAILURE MODE actually observed on 2026-07-08/09, when an ad-hoc
    install silently bumped transformers to 5.13 and broke both MinerU backends for a day.
    The full embedding probe runs separately after egress preflight; ``full`` remains for
    callers that used the earlier check signature.
    Returns (name, ok, detail) rows; import errors are captured, never raised.
    """
    checks: list[tuple[str, bool, str]] = []

    # 1. transformers in the verified range (keep in sync with pyproject [ingest]).
    try:
        from importlib.metadata import version as _v

        from packaging.specifiers import SpecifierSet

        tv = _v("transformers")
        ok = SpecifierSet(">=4.57.0,!=4.57.2,<5").contains(tv)
        checks.append(("transformers range", ok, f"{tv}" + ("" if ok else " — outside >=4.57,!=4.57.2,<5 (5.x kills both MinerU backends)")))
    except Exception as e:
        checks.append(("transformers range", False, f"{type(e).__name__}: {e}"))

    # 2. torch↔torchvision ABI pair (mismatch = torchvision::nms operator missing).
    try:
        from torchvision.ops import nms  # noqa: F401

        checks.append(("torch/torchvision ABI", True, "nms operator present"))
    except Exception as e:
        checks.append(("torch/torchvision ABI", False, f"{type(e).__name__}: {e}"))

    # 3. transformers lazy surface MinerU's models touch.
    try:
        from transformers import AutoProcessor  # noqa: F401

        checks.append(("transformers AutoProcessor", True, "importable"))
    except Exception as e:
        checks.append(("transformers AutoProcessor", False, f"{type(e).__name__}: {e}"))

    # 3b. tomlkit — load-bearing for save_config's comment-preserving patch (every
    # config-mutating MCP tool depends on it silently). Final-hunt H3 gap.
    try:
        import tomlkit  # noqa: F401

        checks.append(("tomlkit", True, "importable"))
    except Exception as e:
        checks.append(("tomlkit", False, f"{type(e).__name__}: {e}"))

    # 4. MinerU entrypoint.
    try:
        from mineru.cli.common import do_parse  # noqa: F401

        from importlib.metadata import version as _mv

        from packaging.specifiers import SpecifierSet as _SS

        mv = _mv("mineru")
        ok_range = _SS(">=3.4.3,<4").contains(mv)
        checks.append(("mineru do_parse+range", ok_range,
                       f"importable, {mv}" + ("" if ok_range else " — outside >=3.4.3,<4")))
    except Exception as e:
        checks.append(("mineru do_parse", False, f"{type(e).__name__}: {e}"))

    # 5. darwin MLX engine chain (hybrid/vlm backends), under the compat shim.
    import sys as _sys

    if _sys.platform == "darwin":
        try:
            from openclaw_brain.knowledge.extraction.mineru_parser import (
                _mlx_transformers_compat_shim,
            )

            _mlx_transformers_compat_shim()
            import mlx_vlm  # noqa: F401

            checks.append(("mlx engine chain", True, "mlx_vlm importable (shim armed)"))
        except Exception as e:
            checks.append(("mlx engine chain", False, f"{type(e).__name__}: {e}"))

    return checks


def _doctor_embedding_check(config) -> tuple[str, bool, str]:
    """Run the optional loader probe only after doctor has checked active policy."""
    try:
        from openclaw_brain.knowledge.embedding import encode_batch

        dims = len(encode_batch(["doctor probe"], config.embedding.model)[0])
        ok = dims == config.embedding.dimensions
        return "embedding encode", ok, f"dims={dims} (expected {config.embedding.dimensions})"
    except Exception as e:
        return "embedding encode", False, f"{type(e).__name__}: {e}"


# ── doctor --models: catalog liveness + candidacy (ZERO inference) ──
#
# The catalog ACCRETES: every model ever tried stays in config/default.toml, nothing expires, and
# nothing recorded whether an entry is still served or still constructible. This report makes that
# gap mechanical instead of folklore. Two deliberate non-goals:
#   1. It NEVER calls a model. Local entries are checked against the server's own /v1/models
#      listing; remote entries are CONSTRUCTED and discarded. No tokens, no bill.
#   2. It NEVER promotes anything. Served-but-uncatalogued models are printed as CANDIDATES —
#      adopting one is an A/B-with-cost operator decision (the figure bake-off, the extraction
#      A/B), not a config edit this command could make for you.

_MODELS_LIST_TIMEOUT_S = 5.0


def _model_id_keys(model_id: str) -> set[str]:
    """Match keys for a model id: the id itself, its '/'-normalized form, and its basename.

    Servers disagree about namespacing, in TWO ways — EXO lists `mlx-community/GLM-4.7-4bit`,
    the oMLX server at :8000 flattens the same namespace to `mlx-community--Qwen3-VL-32B-...`,
    and the catalog stores the '/' form. Normalizing only '/' left three catalogued oMLX models
    reported as uncatalogued CANDIDATES (measured 2026-07-25) — a candidacy report that
    recommends models we already have discredits itself, so both separators are folded.
    """
    mid = (model_id or "").strip()
    normalized = mid.replace("--", "/")
    return {mid, normalized, normalized.split("/")[-1]} - {""}


class _MalformedListing:
    """Sentinel: the endpoint ANSWERED, but its payload could not be read as a model listing.

    A third state next to `list[str]` (parsed) and `None` (unreachable), because collapsing it
    into either one slanders a model: as [] every model on that endpoint reads NOT-SERVED
    ("we looked, it isn't there") on the strength of a payload we could not parse, and as None
    the report claims the server is down when it demonstrably answered.
    """
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<malformed listing>"


MALFORMED_LISTING = _MalformedListing()


class _BlockedListing:
    """Sentinel: local-only blocked this non-loopback model-list endpoint."""
    __slots__ = ()


BLOCKED_LISTING = _BlockedListing()


def _served_model_ids(
    endpoint: str,
    timeout: float = _MODELS_LIST_TIMEOUT_S,
    *,
    egress: str = "any",
):
    """GET {endpoint}/v1/models → served model ids, or a marker for the two non-answers.

    Three outcomes, deliberately distinct (see `_MalformedListing`):
      - ``list[str]``          — parsed. `[]` means reachable and serving nothing, which IS
                                 evidence a model is absent.
      - ``None``               — unreachable. Never evidence AGAINST a model: that is how a
                                 healthy entry gets deleted for being probed during a restart.
      - ``MALFORMED_LISTING``  — answered, but the shape drifted (no `data` list, or entries
                                 that carry no `id`). Also not evidence against a model.
    """
    import httpx

    from openclaw_brain.egress import check_url

    base = (endpoint or "http://localhost:8000/v1").rstrip("/")
    url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
    # Guard outside the broad request-error handler: policy failures must not be disguised as
    # endpoint downtime.
    check_url(url, policy=egress)
    try:
        resp = httpx.get(
            url, timeout=timeout,
            **({"trust_env": False, "follow_redirects": False} if egress == "local-only" else {}),
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return None
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        return MALFORMED_LISTING
    ids = [str(m["id"]) for m in data if isinstance(m, dict) and m.get("id")]
    if data and not ids:
        # Entries are present but none is keyed on `id` (e.g. a flat list of name strings, or
        # a {"model": ...} envelope): we cannot say what this server serves.
        return MALFORMED_LISTING
    return ids


def _model_report(config, timeout: float = _MODELS_LIST_TIMEOUT_S):
    """Build (rows, listings) for the catalog liveness report — no inference, ever.

    rows: (name, provider, tier, status, verdict, detail) per catalog entry.
    listings: endpoint -> served ids (None = unreachable, MALFORMED_LISTING = unreadable
    payload), for the candidacy summary.
    """
    import os

    from openclaw_brain.auth import _env_key_for_provider  # SSOT for provider -> env var
    from openclaw_brain.egress import EgressPolicyError, _safe_url, effective_egress
    from openclaw_brain.llm.provider import LLMProvider

    provider = LLMProvider(config)
    egress = effective_egress(config)
    listings: dict[str, object] = {}
    rows: list[tuple[str, str, str, str, str, str]] = []

    for entry in config.models.catalog:
        if entry.provider == "local":
            ep = entry.endpoint or "http://localhost:8000/v1"
            shown_ep = _safe_url(ep)
            if ep not in listings:
                try:
                    listings[ep] = _served_model_ids(ep, timeout, egress=egress)
                except EgressPolicyError:
                    listings[ep] = BLOCKED_LISTING
            served = listings[ep]
            if served is BLOCKED_LISTING:
                verdict, detail = (
                    "BLOCKED",
                    f"local-only policy blocked non-loopback endpoint — {shown_ep}",
                )
            elif served is None:
                verdict, detail = "UNKNOWN", f"server down/unreachable — {shown_ep} (model NOT judged)"
            elif served is MALFORMED_LISTING:
                verdict, detail = "UNKNOWN", (
                    f"{shown_ep} answered but its /v1/models payload is unreadable (no `id` fields) "
                    f"— model NOT judged"
                )
            elif _model_id_keys(entry.model_id) & {k for s in served for k in _model_id_keys(s)}:
                verdict, detail = "SERVED", shown_ep
            else:
                verdict, detail = "NOT-SERVED", (
                    f"{entry.model_id!r} absent from {shown_ep} ({len(served)} model(s) served)"
                )
        elif entry.status == "banned":
            # Policy-blocked: report it, do not construct it. (Construction is free, but a
            # doctor that quietly instantiates a banned model teaches the wrong reflex.)
            verdict, detail = "BANNED", "policy-blocked — not constructed"
        else:
            env_var = _env_key_for_provider(entry.provider)
            if env_var and not os.environ.get(env_var):
                verdict, detail = "NO-KEY", f"{env_var} not set — construction not attempted"
            else:
                try:
                    provider.get(entry.name)   # constructed, cached, discarded — no request
                    verdict, detail = "CONSTRUCTIBLE", "constructed + discarded (no inference)"
                except Exception as exc:
                    verdict, detail = "NOT-CONSTRUCTIBLE", type(exc).__name__
        rows.append((entry.name, entry.provider, entry.tier, entry.status, verdict, detail))

    provider.clear_cache()
    return rows, listings


def _print_model_report(config, sample_cap: int = 15,
                        timeout: float = _MODELS_LIST_TIMEOUT_S) -> None:
    rows, listings = _model_report(config, timeout=timeout)
    from openclaw_brain.egress import _safe_url

    click.echo("")
    click.echo(f"models — {len(rows)} catalog entries "
               f"(liveness = server listing + construction only; ZERO inference calls)")
    header = ("MODEL", "PROVIDER", "TIER", "STATUS", "VERDICT", "DETAIL")
    widths = [max(len(str(r[i])) for r in (*rows, header)) for i in range(5)]
    click.echo("  " + "  ".join(h.ljust(w) for h, w in zip(header[:5], widths)) + "  " + header[5])
    for r in rows:
        click.echo("  " + "  ".join(str(v).ljust(w) for v, w in zip(r[:5], widths)) + "  " + r[5])

    non_active = [r for r in rows if r[3] != "active"]
    click.echo(f"  ({len(rows) - len(non_active)} active, "
               f"{len(non_active)} non-active: "
               f"{', '.join(f'{r[0]}={r[3]}' for r in non_active) or 'none'})")

    click.echo("")
    click.echo("candidacy — models the local servers expose that the catalog does NOT know.")
    click.echo("  CANDIDATES ONLY: nothing here is adopted, promoted, or routed to. Taking one on")
    click.echo("  requires an A/B evaluation with cost measured, and is an operator decision.")
    for ep, served in listings.items():
        shown_ep = _safe_url(ep)
        if served is BLOCKED_LISTING:
            click.echo(f"  {shown_ep} — BLOCKED BY EGRESS POLICY (no candidacy judgment)")
            continue
        if served is None:
            click.echo(f"  {shown_ep} — UNREACHABLE (no candidacy judgment)")
            continue
        if served is MALFORMED_LISTING:
            click.echo(f"  {shown_ep} — LISTING UNREADABLE (no candidacy judgment)")
            continue
        ep_keys = {
            k
            for e in config.models.catalog
            if e.provider == "local" and (e.endpoint or "http://localhost:8000/v1") == ep
            for k in _model_id_keys(e.model_id)
        }
        uncatalogued = [s for s in served if not (_model_id_keys(s) & ep_keys)]
        click.echo(f"  {shown_ep} — {len(served)} served, {len(served) - len(uncatalogued)} in catalog, "
                   f"{len(uncatalogued)} uncatalogued")
        for mid in uncatalogued[:sample_cap]:
            click.echo(f"      {mid}")
        if len(uncatalogued) > sample_cap:
            click.echo(f"      … +{len(uncatalogued) - sample_cap} more (sample capped at {sample_cap})")


@main.command("doctor")
@click.option("--full", "full", is_flag=True,
              help="Also load the embedding model and encode a probe (slower).")
@click.option("--models", "models", is_flag=True,
              help="Also report per-catalog-entry liveness (local: served by its own endpoint; "
                   "remote: constructible) plus served-but-uncatalogued CANDIDATES. Makes no "
                   "inference call and never changes routing; does not affect the exit code.")
@click.pass_context
def doctor(ctx: click.Context, full: bool, models: bool) -> None:
    """Detect environment drift in the fragile ML stack (transformers/torch/mineru/mlx).

    One command replacing the 90-minute hand diagnosis of the 2026-07-08 incident: an ad-hoc
    install silently bumped transformers to 5.13 and broke BOTH MinerU backends for a day.
    Run after any dependency change, and whenever ingest starts failing strangely. Exits 1 on
    any failed check.

    --models adds a catalog staleness report. Its rows are DIAGNOSTIC ONLY: a deprecated entry,
    a down local server, or an uncatalogued candidate is never an exit-code failure — only the
    environment checks above decide that."""
    from openclaw_brain.egress import (
        LOCAL_ONLY,
        effective_egress,
        enforce_startup_egress,
        environment_violations,
        format_violations,
        validate_egress,
    )

    config = ctx.obj["config"]
    policy = effective_egress(config)
    potential = validate_egress(config, policy=LOCAL_ONLY)
    env_problems = environment_violations() if policy == LOCAL_ONLY else []
    egress_ok = policy != LOCAL_ONLY or (not potential and not env_problems)
    # This probe loads a model. Resolve the active --config and all local-only
    # violations before dependency imports or loader calls. A failed preflight is
    # diagnostic only: report it below and leave the loader untouched.
    embedding_blocked = False
    preflight_failed = False
    if full and policy == LOCAL_ONLY:
        if egress_ok:
            try:
                enforce_startup_egress(config)
            except Exception:
                embedding_blocked = True
                preflight_failed = True
                egress_ok = False
        else:
            embedding_blocked = True
    rows = _doctor_checks(full=False)
    if full:
        rows.append(("embedding encode", False, "blocked by egress preflight")
                    if embedding_blocked else _doctor_embedding_check(config))
    base_bad = 0
    for name, ok, detail in rows:
        click.echo(f"  {'ok  ' if ok else 'FAIL'}  {name}: {detail}")
        base_bad += 0 if ok else 1
    if policy == LOCAL_ONLY:
        detail = ("local-only; all configured model and Neo4j endpoints are loopback"
                  if egress_ok else
                  f"local-only; {len(potential) + len(env_problems)} violation(s):\n"
                  + "\n".join(filter(None, [format_violations(potential), *env_problems,
                                                "startup preflight failed" if preflight_failed else ""])))
    else:
        detail = (f"any; local-only would block {len(potential)} configured reference(s)"
                  + (f":\n{format_violations(potential)}" if potential else ""))
    click.echo(f"  {'ok  ' if egress_ok else 'FAIL'}  egress policy: {detail}")

    if base_bad:
        # Preserve the established dependency-drift count; egress is reported separately above.
        click.echo(f"doctor — {base_bad}/{len(rows)} check(s) FAILED. Restore with: uv sync (or see requirements-known-good.txt)")
    elif not egress_ok:
        click.echo("doctor — egress policy check FAILED.")
    else:
        click.echo(f"doctor — all {len(rows) + 1} checks green.")

    if models:
        # Best-effort key resolution first, so NO-KEY means "no key anywhere this repo looks",
        # not "not exported in this shell". Never fatal: the report degrades to NO-KEY rows.
        if policy != LOCAL_ONLY:
            try:
                from openclaw_brain.auth import inject_api_keys

                inject_api_keys()
            except Exception as e:  # pragma: no cover - defensive
                click.echo(f"  (api-key injection skipped: {type(e).__name__}: {e})")
        _print_model_report(ctx.obj["config"])

    if base_bad or not egress_ok:
        raise SystemExit(1)


@main.command("relink-source")
@click.option("--source-id", "source_id", required=True,
              help="Source whose concepts get enriched with cross-source relationships.")
@click.option("--against", "against", multiple=True,
              help="Target source_id to link against (repeatable). Default: any other source.")
@click.option("--min-score", "min_score", default=0.6, type=float,
              help="Minimum embedding-similarity score for a candidate neighbour (default 0.6).")
@click.option("--top-k", "top_k", default=6, type=int,
              help="Max candidate neighbours reasoned per concept (default 6).")
@click.option("--apply", "apply_", is_flag=True,
              help="Commit the reasoned cross-links (default: dry-run — reason but do not write).")
@click.pass_context
def relink_source_cmd(ctx: click.Context, source_id: str, against: tuple[str, ...],
                      min_score: float, top_k: int, apply_: bool) -> None:
    """Enrich an existing source's concepts with cross-source relationships against the CURRENT
    graph, without re-chunking or re-extracting (which a reprocess does, duplicating chunks).
    Finds newly-reachable neighbours by embedding similarity, reasons which pairs carry a real
    relationship, and commits ONLY those edges. Purely additive; re-runnable with zero
    duplication. Dry-run by default."""
    asyncio.run(_relink_source(ctx.obj["config"], source_id, list(against) or None,
                               min_score, top_k, apply_))


async def _relink_source(config, source_id, against, min_score, top_k, apply_) -> None:
    import logging

    from openclaw_brain.agent import BrainAgent
    from openclaw_brain.knowledge.reasoning.relink import relink_source

    # The scan/plan/batch heartbeat lines are the observable surface an external monitor
    # (/loop) greps for liveness and progress — the default root level (WARNING) would hide
    # them entirely. Enable INFO for the relink logger only; everything else stays at WARNING.
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("openclaw_brain.knowledge.reasoning.relink").setLevel(logging.INFO)

    agent = BrainAgent(config)
    await agent.start()
    try:
        # NOTE (2026-07-25): stage='reasoning', so this chain inherits the stage's
        # [reasoning].output_token_budget bound — and relink_source does NOT opt into the
        # truncation branch, so a clipped batch verdict would be parsed as-is. Deliberate, not
        # an oversight: relink emits one small verdict object per pair batch (orders of
        # magnitude under the bound), it is an operator-run CLI with a visible dry-run before
        # --apply, and its per-batch failures are already counted in `failed_batches`.
        chain = agent._llm_provider.get_chain("reasoning")
        stats = await relink_source(
            agent._graph, chain, source_id, config.resilience,
            target_source_ids=against, top_k=top_k, min_score=min_score, apply=apply_,
        )
        verb = "APPLIED" if apply_ else "DRY-RUN (no write)"
        click.echo(
            f"relink-source — {verb}: scanned {stats['concepts_scanned']} concepts, "
            f"{stats['candidate_pairs']} candidate pairs, {stats['edges_valid']} valid edges "
            f"reasoned, {stats['edges_committed']} committed"
            f" (failed_batches={stats.get('failed_batches', 0)},"
            f" skipped_pairs={stats.get('skipped_pairs', 0)})."
        )
    except Exception as e:
        click.echo(f"relink-source failed: {e}")
        raise SystemExit(1)
    finally:
        await agent.stop()


@main.command("ingest-html")
@click.argument("html_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--extraction-model", default=None, help="Model name override for extraction stage")
@click.option("--reasoning-model", default=None, help="Model name override for reasoning stage")
@click.option("--chunk-concurrency", default=1, type=int,
              help="Max chunks processed concurrently (default 1 = sequential)")
@click.option("--reprocess", is_flag=True,
              help="Re-run every chunk even if checkpointed done (multi-pass refine)")
@click.option("--figures-only", is_flag=True,
              help="VLM-analyze slide IMAGES (equations/schematics/plots) instead of the speech "
                   "transcript. Can run against a source already text-ingested (same source_id, "
                   "a separate {source_id}_figs checkpoint) — see knowledge/pipeline.py "
                   "KnowledgePipeline._ingest_html_figures_only.")
@click.pass_context
def ingest_html_cmd(ctx: click.Context, html_path: str, extraction_model: str | None,
                    reasoning_model: str | None, chunk_concurrency: int, reprocess: bool,
                    figures_only: bool) -> None:
    """Ingest one of Rick's lecture-capture HTML decks (see
    knowledge/extraction/html_parser.py for the expected input format) through the SAME
    chunk->extract->ground->match->reason->reconcile->commit->embed->summarize pipeline a PDF
    uses — only the parse stage differs (html_parser instead of MinerU; no figure-VLM stage by
    default, since equations/diagrams live inside the slide images — pass --figures-only to
    analyze those instead of the speech transcript)."""
    asyncio.run(_ingest_html(
        ctx.obj["config"], html_path, extraction_model, reasoning_model,
        chunk_concurrency, reprocess, figures_only,
    ))


async def _ingest_html(config, html_path: str, extraction_model: str | None,
                       reasoning_model: str | None, chunk_concurrency: int, reprocess: bool,
                       figures_only: bool = False) -> None:
    from openclaw_brain.agent import BrainAgent

    agent = BrainAgent(config)
    await agent.start()
    try:
        result = await agent.ingest_html(
            file_path=html_path,
            extraction_model=extraction_model,
            reasoning_model=reasoning_model,
            chunk_concurrency=chunk_concurrency,
            reprocess=reprocess,
            figures_only=figures_only,
        )
        output = result.output if hasattr(result, "output") else result
        if not output.get("success", False):
            click.echo(f"ingest-html failed: {output.get('error') or output.get('errors')}")
            raise SystemExit(1)
        if output.get("empty_reason"):
            # Legitimately empty (no slides / all slides gated) — NOT a failure. Exit 3 so
            # batch drivers count "empty" separately from "failed" (C6: 78 of the lecture
            # batch's 88 exit-1s were this, reading as a 76% failure rate that wasn't).
            click.echo(f"ingest-html — nothing to ingest: {output['empty_reason']}")
            raise SystemExit(3)
        click.echo(f"ingest-html — {output.get('summary', '')}")
        click.echo(f"  source_id={output.get('source_id')} title={output.get('title')!r} "
                   f"chunks={output.get('total_chunks')}")
        click.echo(f"  new_nodes={output.get('new_nodes')} updated_nodes={output.get('updated_nodes')} "
                   f"new_edges={output.get('new_edges')} reinforced_edges={output.get('reinforced_edges')} "
                   f"insights={output.get('insights')}")
        errors = output.get("errors") or []
        if errors:
            click.echo(f"  {len(errors)} error(s):")
            for err in errors[:20]:
                click.echo(f"    - {err}")
    except SystemExit:
        raise
    except Exception as e:
        click.echo(f"ingest-html failed: {e}")
        raise SystemExit(1)
    finally:
        await agent.stop()


@main.command("export-obsidian")
@click.option("--vault", default=str(Path.home() / "Semiconductor"),
              help="Path to Obsidian vault directory")
@click.option("--typed-links/--no-typed-links", default=True,
              help="Also emit outgoing edges as rel_type-keyed frontmatter "
                   "[[wikilink]] properties, e.g. 'depends_on: [...]' "
                   "(default: on).")
@click.pass_context
def export_obsidian(ctx: click.Context, vault: str, typed_links: bool) -> None:
    """Export knowledge graph to an Obsidian vault."""
    asyncio.run(_export_obsidian(ctx.obj["config"], Path(vault), typed_links))


async def _export_obsidian(config, vault_path: Path, typed_links: bool = True):
    from openclaw_brain.export.obsidian import ObsidianExporter

    vault_path.mkdir(parents=True, exist_ok=True)
    exporter = ObsidianExporter(config.neo4j, vault_path, typed_links=typed_links,
                                egress=effective_egress(config))
    try:
        await exporter.connect()
        click.echo(f"Exporting to {vault_path} ...")
        counts = await exporter.export()
        total = sum(counts.values())
        click.echo(f"Exported {total} nodes:")
        for label, count in sorted(counts.items()):
            click.echo(f"  {label}: {count}")
        click.echo(f"\nOpen '{vault_path}' in Obsidian to browse the knowledge graph.")
    except Exception as e:
        click.echo(f"Export failed: {e}")
        raise SystemExit(1)
    finally:
        await exporter.close()


@main.command("backfill-embeddings")
@click.option("--batch-size", default=128, help="Nodes per embedding batch")
@click.option("--force", is_flag=True,
              help="Null ALL embeddings and re-embed (required on model change)")
@click.option("--recreate-indexes", is_flag=True,
              help="Drop + recreate vector indexes for all 5 labels at the configured dimension first")
@click.pass_context
def backfill_embeddings(ctx: click.Context, batch_size: int, force: bool,
                        recreate_indexes: bool) -> None:
    """Generate embeddings for all knowledge nodes that lack one."""
    asyncio.run(_backfill_embeddings(ctx.obj["config"], batch_size, force, recreate_indexes))


async def _backfill_embeddings(config, batch_size: int, force: bool = False,
                               recreate_indexes: bool = False):
    from openclaw_brain.knowledge.graph.store import GraphStore

    store = GraphStore(config.neo4j, egress=effective_egress(config))
    try:
        await store.connect()

        def _progress(label, done, total):
            click.echo(f"  {label}: {done}/{total}", nl=True)

        if recreate_indexes:
            created = await store.recreate_vector_indexes(config.embedding.dimensions)
            click.echo(f"Recreated vector indexes @ {config.embedding.dimensions}d: "
                       f"{', '.join(created)}")

        click.echo(f"Backfilling embeddings (model={config.embedding.model}, "
                   f"batch_size={batch_size}, force={force}) ...")
        totals = await store.backfill_embeddings(
            model_name=config.embedding.model,
            batch_size=batch_size,
            on_progress=_progress,
            force=force,
        )
        grand_total = sum(totals.values())
        click.echo(f"\nDone — {grand_total} nodes embedded:")
        for label, cnt in totals.items():
            if cnt:
                click.echo(f"  {label}: {cnt}")
    except Exception as e:
        click.echo(f"Backfill failed: {e}")
        raise
    finally:
        await store.close()


@main.command("quality-audit")
@click.option("--uri", default=None, help="Neo4j URI override")
@click.option("--sample", default=100, type=int, help="Number of nodes to audit")
@click.option("--labels", default="Concept,Principle,Insight",
              help="Comma-separated Neo4j labels to sample")
@click.option("--model", default="deepseek-v4-flash-local", help="Primary audit judge model")
@click.option("--fallback-model", default="deepseek-v4-flash-mlx", help="Fallback audit judge model")
@click.option("--seed", default=42, type=int, help="Python shuffle seed")
@click.option("--out", "out_path", default=None, type=click.Path(dir_okay=False),
              help="Write full JSON audit output to PATH")
@click.option("--fail-below", default=0.0, type=float,
              help="Exit 1 if precision is below this threshold")
@click.option("--evidence-selection", "evidence_selection", default="overlap",
              type=click.Choice(["overlap", "embedding"]),
              help="Chunk evidence ranking: overlap (token) or embedding (semantic cosine)")
@click.pass_context
def quality_audit(
    ctx: click.Context,
    uri: str | None,
    sample: int,
    labels: str,
    model: str,
    fallback_model: str,
    seed: int,
    out_path: str | None,
    fail_below: float,
    evidence_selection: str,
) -> None:
    """Audit node descriptions against source evidence for scope fidelity."""
    asyncio.run(_quality_audit(
        ctx.obj["config"],
        uri=uri,
        sample=sample,
        labels=labels,
        model=model,
        fallback_model=fallback_model,
        seed=seed,
        out_path=out_path,
        fail_below=fail_below,
        evidence_selection=evidence_selection,
    ))


async def _quality_audit(
    config,
    *,
    uri: str | None,
    sample: int,
    labels: str,
    model: str,
    fallback_model: str,
    seed: int,
    out_path: str | None,
    fail_below: float,
    evidence_selection: str = "overlap",
) -> None:
    from openclaw_brain.knowledge.graph.store import GraphStore
    from openclaw_brain.knowledge.evidence import EvidenceVault
    from openclaw_brain.knowledge.quality import (
        LLMQualityJudge,
        build_semantic_ranker,
        fetch_chunks_from_graph,
        fetch_nodes_from_graph,
        format_compact_table,
        run_quality_audit,
        should_fail_precision,
        write_quality_audit_json,
    )
    from openclaw_brain.llm.provider import LLMProvider

    if uri:
        config.neo4j.uri = uri
    config.resilience.request_timeout_s = 900

    parsed_labels = [label.strip() for label in labels.split(",") if label.strip()]
    store = GraphStore(config.neo4j, egress=effective_egress(config))
    vault = EvidenceVault(config.state_path / "evidence")
    provider = LLMProvider(config)
    judge = LLMQualityJudge(
        provider,
        model=model,
        fallback_model=fallback_model,
        timeout_s=config.resilience.request_timeout_s,
    )
    embed_rank = build_semantic_ranker(config) if evidence_selection == "embedding" else None

    # Cache hydrated chunks per source: a textbook source can hold ~1500 chunks, and
    # re-fetching + re-hydrating them from the vault for every sampled node dominates
    # wall-clock. Each source is fetched once per run.
    chunk_cache: dict[str, list] = {}

    async def fetch_chunks_cached(source_id: str) -> list:
        if source_id not in chunk_cache:
            chunk_cache[source_id] = await fetch_chunks_from_graph(store, source_id, vault)
        return chunk_cache[source_id]

    await store.connect()
    try:
        result = await run_quality_audit(
            labels=parsed_labels,
            sample=sample,
            seed=seed,
            fetch_nodes=lambda label: fetch_nodes_from_graph(store, label),
            fetch_chunks=fetch_chunks_cached,
            judge=judge,
            embed_rank=embed_rank,
            params={
                "uri": config.neo4j.uri,
                "sample": sample,
                "labels": parsed_labels,
                "model": model,
                "fallback_model": fallback_model,
                "seed": seed,
                "fail_below": fail_below,
                "evidence_selection": evidence_selection,
            },
        )
    finally:
        await store.close()

    if out_path:
        write_quality_audit_json(result, out_path)

    click.echo(format_compact_table(result))
    if out_path:
        click.echo(f"wrote {out_path}")

    if should_fail_precision(result["summary"], fail_below):
        raise SystemExit(1)


@main.command("project-executable")
@click.option("--apply", "apply_", is_flag=True,
              help="Write the projection (default: dry-run, read-only)")
@click.option("--corpus", "corpus_root", default=None,
              help="Persist verdicted specimens to this corpus dir (git SSOT); overrides the "
                   "config default ([executable] corpus_dir)")
@click.option("--retract", "retract_spec_id", default=None,
              help="Reverse a projection: DETACH DELETE the Specimen + its ClaimCards for this spec_id")
@click.pass_context
def project_executable(ctx: click.Context, apply_: bool, corpus_root: str | None,
                       retract_spec_id: str | None) -> None:
    """Run ALL validated seed recipes — analog (ngspice), digital (iverilog), and Stat-QT
    statistical/corner (ngspice MC), 22 total — dispatched to each recipe's engine, and project the
    verdicted specimens onto the live graph: additive Specimen/ClaimCard nodes + REALIZES/GROUNDS
    links to existing CircuitTopology/Parameter nodes (hybrid resolver, additive + NO-PHANTOM). A
    recipe whose engine's runner is unavailable is skipped (reported), not fatal. Dry-run by
    default. --retract SPEC_ID reverses a prior apply for that specimen."""
    asyncio.run(_project_executable(ctx.obj["config"], apply_, corpus_root, retract_spec_id))


async def _project_executable(config, apply_: bool, corpus_root: str | None,
                              retract_spec_id: str | None = None):
    from openclaw_brain.knowledge.executable.corpus import SpecimenCorpus
    from openclaw_brain.knowledge.executable.engines import ENGINES, engine_for_template, runner_for_engine
    from openclaw_brain.knowledge.executable.executor import run_recipe
    from openclaw_brain.knowledge.executable.projection import (
        GraphProjector, project_specimen, retract_projection,
    )
    from openclaw_brain.knowledge.executable.recipe import capability_for
    from openclaw_brain.knowledge.executable.resolver import GraphResolver
    from openclaw_brain.knowledge.executable.seeds import (
        digital_seed_recipes, seed_recipes, statistical_seed_recipes,
    )
    from openclaw_brain.knowledge.graph.store import GraphStore

    # Retraction path: no sim needed, just undo a prior projection.
    if retract_spec_id:
        store = GraphStore(config.neo4j, egress=effective_egress(config))
        await store.connect()
        try:
            stats = await retract_projection(store, retract_spec_id)
            click.echo(f"retracted {retract_spec_id}: "
                       f"{stats['specimens']} Specimen + {stats['claim_cards']} ClaimCard deleted.")
        finally:
            await store.close()
        return

    store = GraphStore(config.neo4j, egress=effective_egress(config))
    await store.connect()
    resolver = GraphResolver(store)
    projector = GraphProjector(store, resolver.resolve)
    # Dry-run writes NOTHING (graph or corpus) — only --apply persists. (Dry-run still runs the
    # sim; that's how it produces the verdicts it previews.) --corpus overrides the config default
    # ([executable] corpus_dir, resolved against the repo root).
    corpus_root_final = corpus_root or str(config.executable.corpus_path)
    corpus = SpecimenCorpus(corpus_root_final) if apply_ else None
    if corpus_root and not apply_:
        click.echo("note: --corpus is ignored in dry-run (nothing is written); pass --apply to persist.")

    click.echo(f"project-executable — "
               f"{'APPLY (writes graph + corpus)' if apply_ else 'DRY-RUN (runs sims; writes nothing)'}\n")
    tot_nodes = tot_links = tot_written = 0
    skipped: list[dict] = []
    try:
        all_recipes = seed_recipes() + digital_seed_recipes() + statistical_seed_recipes()
        for recipe in all_recipes:
            template_ref = (recipe.build or {}).get("template_ref") or \
                capability_for(recipe.topology_class).template_ref
            engine = (recipe.build or {}).get("engine") or engine_for_template(template_ref)
            espec = ENGINES.get(engine)
            if espec is None:
                reason = f"unknown engine {engine!r}"
            elif espec.runner is None:
                reason = "engine is render-only (no runner)"
            elif not runner_for_engine(engine, config).available():
                reason = f"{engine} runner unavailable (sim image/tooling not present)"
            else:
                reason = None
            if reason:
                click.echo(f"=== {recipe.topology_class} ({engine}) === SKIPPED: {reason}\n")
                skipped.append({"topology_class": recipe.topology_class, "engine": engine, "reason": reason})
                continue

            click.echo(f"=== {recipe.topology_class} ({engine}) ===")
            result = run_recipe(recipe, runner_for_engine(engine, config), corpus=corpus, config=config)
            spec = result.specimen
            verdicts = ", ".join(f"{c.id}={c.verdict.value if c.verdict else '?'}" for c in result.claim_cards)
            click.echo(f"  sim: {result.runs} decks; verdicts: {verdicts}")
            if corpus:
                click.echo(f"  corpus: {result.spec_id} -> {result.spec_dir}")
            pw = project_specimen(spec)
            tot_nodes += len(pw.nodes); tot_links += len(pw.links)
            for lr in pw.links:
                rid = await resolver.resolve(lr.target_label, lr.match_text)
                click.echo(f"  {lr.rel_type.value:9s} {lr.source_label.value} -> "
                           f"{lr.target_label.value}: {rid or '(no match -> NO EDGE)'}")
            if apply_:
                stats = await projector.project(spec)
                tot_written += stats["links_resolved"]
                click.echo(f"  WROTE {stats['nodes']} nodes, {stats['internal_edges']} internal edges, "
                           f"{stats['links_resolved']}/{stats['links_total']} links")
            click.echo("")

        if apply_:
            click.echo(f"APPLIED: {tot_nodes} additive nodes + {tot_written}/{tot_links} cross-links written.")
        else:
            click.echo(f"DRY-RUN: {tot_nodes} additive nodes, {tot_links} cross-links would resolve. "
                       "Nothing written — re-run with --apply to write.")
        if skipped:
            click.echo(f"SKIPPED {len(skipped)} recipe(s): " +
                       ", ".join(f"{s['topology_class']} ({s['engine']}: {s['reason']})" for s in skipped))
    finally:
        await store.close()      # always release the Neo4j driver, even if a recipe raises (I1)


@main.command("render-schematic")
@click.argument("target", required=False)
@click.option("--list", "list_", is_flag=True,
              help="List every corpus topology_class and whether a schematic template is "
                   "registered for it (pilot: 3 of 19 families), instead of rendering.")
@click.option("--mos-style", default=None, type=click.Choice(["razavi", "ieee"]),
              help="MOSFET symbol style (default: config [figures].schematic_mos_style)")
@click.option("--out-dir", "out_dir", default=None,
              help="Directory to write SVG(s) to (default: [storage] state_dir/schematics)")
@click.option("--corpus", "corpus_root", default=None,
              help="Corpus dir to read specimens from; overrides the config default "
                   "([executable] corpus_dir)")
@click.pass_context
def render_schematic_cmd(ctx: click.Context, target: str | None, list_: bool,
                         out_dir: str | None, corpus_root: str | None, mos_style: str | None) -> None:
    """Render a corpus specimen's cell.spice to a hand-authored SVG circuit diagram (schemdraw;
    read-only, no Neo4j/git writes). TARGET is either a topology_class (renders every specimen on
    disk for that class) or a spec_id / its 16-hex short-hash (renders that one specimen, searched
    across all classes). A topology_class with no registered template (most of the 19 — pilot
    scope is 3) is reported unsupported, never silently drawn generically."""
    from openclaw_brain.knowledge.executable.corpus import SpecimenCorpus
    from openclaw_brain.knowledge.executable.schematic import (
        SchematicMismatchError, UnsupportedTopologyError, is_supported, render_specimen,
    )

    config = ctx.obj["config"]
    cdir = corpus_root or str(config.executable.corpus_path)
    corpus = SpecimenCorpus(cdir)
    out = Path(out_dir) if out_dir else config.state_path / "schematics"

    if list_:
        classes = corpus.list_classes()
        click.echo(f"render-schematic --list — {len(classes)} topology_class(es) in corpus ({cdir})\n")
        for tc in classes:
            n = len(corpus.list_class(tc))
            status = "supported" if is_supported(tc) else "unsupported"
            click.echo(f"  [{status:11s}] {tc}  ({n} specimen(s))")
        return

    if not target:
        raise click.UsageError("provide a topology_class or spec_id, or pass --list")

    # TARGET is either a topology_class (every specimen on disk for it) or a spec_id/short-hash
    # (searched across every topology_class dir).
    targets: list[tuple[str, str]] = []
    if target in corpus.list_classes():
        targets = [(target, spec_id) for spec_id in corpus.list_class(target)]
    else:
        for tc in corpus.list_classes():
            for spec_id in corpus.list_class(tc):
                if spec_id == target or spec_id.split(":", 1)[-1][:16] == target:
                    targets.append((tc, spec_id))
    if not targets:
        click.echo(f"render-schematic: no specimen found on disk matching {target!r} (in {cdir})")
        raise SystemExit(1)

    bad = 0
    for tc, spec_id in targets:
        spec_dir = Path(corpus.spec_dir(tc, spec_id))
        try:
            path = render_specimen(spec_dir, out, mos_style=mos_style or ctx.obj["config"].figures.schematic_mos_style)
            click.echo(f"  {tc} {spec_id} -> {path}")
        except UnsupportedTopologyError as e:
            click.echo(f"  {tc} {spec_id}: UNSUPPORTED — {e}")
            bad += 1
        except SchematicMismatchError as e:
            click.echo(f"  {tc} {spec_id}: MISMATCH — {e}")
            bad += 1
    if bad:
        raise SystemExit(1)


@main.command("grow-corpus")
@click.option("--source", "sources_", multiple=True, default=("all",),
              type=click.Choice(["coverage", "queue", "backlog", "all"]),
              help="Growth source(s) to plan from (repeatable). Default: all.")
@click.option("--max-recipes", default=5, type=int,
              help="Cap on recipes authored+simulated this run (must be > 0).")
@click.option("--max-per-class", default=4, type=int,
              help="Cap on new claim-cards per topology_class this run.")
@click.option("--class", "topology_class_", default=None,
              help="Restrict to one registered topology_class.")
@click.option("--apply", "apply_", is_flag=True,
              help="Write the projection + persist to the git corpus (default: dry-run)")
@click.option("--corpus", "corpus_root", default=None,
              help="Persist verdicted specimens to this corpus dir on --apply; overrides the "
                   "config default ([executable] corpus_dir)")
@click.pass_context
def grow_corpus(ctx: click.Context, sources_: tuple[str, ...], max_recipes: int, max_per_class: int,
                topology_class_: str | None, apply_: bool, corpus_root: str | None) -> None:
    """Corpus growth automation (D3 AUTO lane, spec 2026-07-03-corpus-growth-automation-design.md):
    author + simulate NEW claim-cards for registered topologies, sourced from coverage-gap
    enumeration over the template registry, the ingest growth_queue, and the TOPOLOGY_BACKLOG.md
    curriculum. VERIFIED/REFUTED-family verdicts project on --apply (REFUTED is knowledge too,
    never hidden); FLAGGED/REJECTED never project — they land in the triage report for post-mortem.
    A NEW SPICE template is never auto-admitted; an unmatched high-signal topology only ever
    surfaces via the review-lane pending counts. Dry-run by default."""
    sources = ["coverage", "queue", "backlog"] if "all" in sources_ else list(dict.fromkeys(sources_))
    asyncio.run(_grow_corpus(
        ctx.obj["config"], sources, max_recipes, max_per_class, topology_class_, apply_, corpus_root,
    ))


async def _grow_corpus(config, sources: list[str], max_recipes: int, max_per_class: int,
                       topology_class: str | None, apply_: bool, corpus_root: str | None) -> None:
    from openclaw_brain.auth import inject_api_keys
    from openclaw_brain.journal import ActionJournal
    from openclaw_brain.knowledge.evidence import EvidenceVault
    from openclaw_brain.knowledge.executable import growth
    from openclaw_brain.knowledge.executable.corpus import SpecimenCorpus
    from openclaw_brain.knowledge.executable.projection import GraphProjector
    from openclaw_brain.knowledge.executable.resolver import GraphResolver
    from openclaw_brain.knowledge.graph.store import GraphStore
    from openclaw_brain.llm.provider import LLMProvider

    if effective_egress(config) != "local-only":
        inject_api_keys()
    store = GraphStore(config.neo4j, egress=effective_egress(config))
    try:
        await store.connect()
        click.echo(f"grow-corpus — sources={sources} max_recipes={max_recipes} "
                   f"max_per_class={max_per_class} "
                   f"{'APPLY (writes graph + corpus)' if apply_ else 'DRY-RUN (writes nothing)'}\n")

        plan = await growth.plan_growth(
            store, sources=sources, max_recipes=max_recipes,
            max_new_cards_per_class=max_per_class, topology_class=topology_class,
            state_dir=config.state_path, apply=apply_,
        )
        click.echo(f"planned {len(plan.targets)} target(s) "
                   f"(skipped_covered={plan.skipped_covered}, "
                   f"skipped_digital={plan.skipped_digital}, "
                   f"review_lane_pending={plan.review_lane_pending})")
        for t in plan.targets:
            grounded = f"  grounded x{len(t.grounding_chunk_ids)}" if t.grounding_chunk_ids else ""
            kind = f"({t.kind_hint})" if t.kind_hint else "(kind: author's choice)"
            click.echo(f"  [{t.source:8s}] {t.topology_class}: {t.metric} vs {t.knob} "
                       f"{kind}{grounded}")

        corpus = None
        projector = None
        if apply_:
            cdir = corpus_root or str(config.executable.corpus_path)
            corpus = SpecimenCorpus(cdir)
            projector = GraphProjector(store, GraphResolver(store).resolve)

        provider = LLMProvider(config)
        deps = growth.GrowthDeps(
            store=store, model_chain=provider.get_chain("extraction"),
            resilience_config=config.resilience, auth_refresh=None,
            vault=EvidenceVault(config.state_path / "evidence"),
            corpus=corpus, projector=projector,
            journal=ActionJournal(config.state_path), state_dir=config.state_path,
            config=config,
        )
        report = await growth.execute_growth(deps, plan, apply=apply_)

        click.echo(f"\nauthored={report.authored} simulated={report.simulated} "
                   f"projected={report.projected} triage={len(report.triage)} "
                   f"duration={report.duration_s:.1f}s")
        for entry in report.certified:
            cards = ", ".join(
                f"{c['id']}: {c['metric']} vs {c['knob']} [{c['kind']}] -> {c['verdict']}"
                for c in entry["cards"])
            click.echo(f"  CERTIFIED [{entry['source']}] {entry['topology_class']}: {cards}")
        for entry in report.triage:
            detail = entry.get("reason") or entry.get("error") or entry.get("verdicts")
            click.echo(f"  TRIAGE [{entry.get('stage')}] {entry.get('topology_class')}: "
                       f"{entry.get('metric')} vs {entry.get('knob')} — {detail}")
        if not apply_:
            click.echo("\n(dry run — re-run with --apply to write the projection + corpus)")
    finally:
        await store.close()


@main.command("export-graph")
@click.argument("out_path", type=click.Path(dir_okay=False))
@click.option("--include-embeddings", is_flag=True,
              help="Include embedding vectors (default: stripped — rebuildable via "
                   "backfill-embeddings; keeps the artifact small)")
@click.option("--include-private", is_flag=True,
              help="Include personal-memory-subsystem labels (Memory/Session/SkillRun/Entity). "
                   "NOT for the ADR-044 D1 company artifact — local/debug use only.")
@click.pass_context
def export_graph_cmd(ctx: click.Context, out_path: str, include_embeddings: bool,
                     include_private: bool) -> None:
    """Export the knowledge graph to a streaming JSONL artifact (ADR-044 D1 downloadable
    graph). SourceChunk citation metadata (source ids/hashes) is included; verbatim source
    text and the personal memory subsystem are not."""
    asyncio.run(_export_graph(ctx.obj["config"], out_path, include_embeddings, include_private))


async def _export_graph(config, out_path: str, include_embeddings: bool, include_private: bool):
    from openclaw_brain.export.graph_io import DEFAULT_PRIVATE_LABELS, export_graph
    from openclaw_brain.knowledge.graph.store import GraphStore

    store = GraphStore(config.neo4j, egress=effective_egress(config))
    try:
        await store.connect()
        exclude = frozenset() if include_private else DEFAULT_PRIVATE_LABELS

        def _progress(label, done, total):
            click.echo(f"  {label}: {done}/{total}", nl=True)

        click.echo(f"Exporting graph to {out_path} "
                   f"(embeddings={'included' if include_embeddings else 'stripped'}, "
                   f"private={'included' if include_private else 'excluded'}) ...")
        header = await export_graph(
            store, out_path,
            exclude_labels=exclude,
            include_embeddings=include_embeddings,
            on_progress=_progress,
        )
        node_total = sum(header["counts"]["nodes"].values())
        edge_total = sum(header["counts"]["edges"].values())
        click.echo(f"\nExported {node_total} nodes, {edge_total} edges -> {out_path}")
    except Exception as e:
        click.echo(f"Export failed: {e}")
        raise
    finally:
        await store.close()


@main.command("import-graph")
@click.argument("in_path", type=click.Path(exists=True, dir_okay=False))
@click.pass_context
def import_graph_cmd(ctx: click.Context, in_path: str) -> None:
    """Import a JSONL graph artifact (produced by export-graph) via idempotent MERGE
    writes — safe to re-run over an existing graph without duplicating nodes/edges."""
    asyncio.run(_import_graph(ctx.obj["config"], in_path))


async def _import_graph(config, in_path: str):
    from openclaw_brain.export.graph_io import import_graph
    from openclaw_brain.knowledge.graph.store import GraphStore

    store = GraphStore(config.neo4j, egress=effective_egress(config))
    try:
        await store.connect()

        def _progress(label, done, total):
            click.echo(f"  imported: {done}/{total}", nl=True)

        click.echo(f"Importing graph from {in_path} ...")
        counts = await import_graph(store, in_path, on_progress=_progress)
        click.echo(
            f"\nImported {counts['nodes_imported']} nodes, {counts['edges_imported']} edges "
            f"({counts['nodes_skipped']} nodes skipped, {counts['edges_skipped']} edges skipped)."
        )
    except Exception as e:
        click.echo(f"Import failed: {e}")
        raise
    finally:
        await store.close()


@main.command("project-laws")
@click.option("--raw", "raw_paths", multiple=True, type=click.Path(exists=True, dir_okay=False),
              help="Replication runner raw JSONL path(s) (repeatable). Default: the E1 + E1b + "
                   "full-registry (E-track ③) outputs (experiments/e1_cross_pdk_raw.jsonl + "
                   "experiments/e1b_statistical_cross_pdk_raw.jsonl + "
                   "experiments/e_rollout_full_registry_raw.jsonl).")
@click.option("--apply", "apply_", is_flag=True,
              help="Write the projection (default: dry-run, prints the law table only)")
@click.pass_context
def project_laws(ctx: click.Context, raw_paths: tuple[str, ...], apply_: bool) -> None:
    """Project law-tier `Regularity` nodes from the cross-PDK replication runner's raw JSONL
    (docs/superpowers/specs/2026-07-04-law-tier-graph-representation.md). Additive, idempotent
    MERGE by law_id — SUPPORTED_BY only links to member ClaimCards that already exist in the
    graph (NO-PHANTOM); ABOUT resolves via the same hybrid resolver project-executable uses.
    Dry-run by default; --apply writes and journals every written/updated/demoted law."""
    paths = list(raw_paths) or _default_law_jsonl_paths()
    asyncio.run(_project_laws(ctx.obj["config"], paths, apply_))


def _default_law_jsonl_paths() -> list[str]:
    repo_root = Path(__file__).resolve().parent.parent.parent
    return [
        str(repo_root / "experiments" / "e1_cross_pdk_raw.jsonl"),
        str(repo_root / "experiments" / "e1b_statistical_cross_pdk_raw.jsonl"),
        str(repo_root / "experiments" / "e_rollout_full_registry_raw.jsonl"),
    ]


async def _project_laws(config, raw_paths: list[str], apply_: bool) -> None:
    from openclaw_brain.journal import ActionJournal
    from openclaw_brain.knowledge.executable.laws import build_law_records, load_run_rows, project_law_records
    from openclaw_brain.knowledge.executable.resolver import GraphResolver
    from openclaw_brain.knowledge.graph.store import GraphStore

    load_results = {}
    for path in raw_paths:
        load_results[path] = load_run_rows(path)
        n_errors = len(load_results[path].errors)
        if n_errors:
            click.echo(f"  {path}: {n_errors} malformed row(s) skipped")
            for err in load_results[path].errors[:10]:
                click.echo(f"    line {err['line']}: {err['error']}")

    records, gap_report = build_law_records(load_results)
    if gap_report["unknown_claim_ids"]:
        click.echo(f"  unknown claim_ids (no catalog entry, skipped): "
                   f"{', '.join(gap_report['unknown_claim_ids'])}")

    click.echo(f"\nproject-laws — {'APPLY' if apply_ else 'DRY-RUN'} — "
               f"{len(records)} law(s) derived from {len(raw_paths)} JSONL file(s)\n")

    store = GraphStore(config.neo4j, egress=effective_egress(config))
    await store.connect()
    try:
        resolver = GraphResolver(store)
        journal = ActionJournal(config.state_path) if apply_ else None
        outcomes = await project_law_records(
            store, records, about_resolver=resolver.resolve, journal=journal, apply=apply_,
        )
        by_law_id = {o.law_id: o for o in outcomes}
        click.echo(f"{'law_id':14s} {'topology_class':32s} {'metric':12s} {'knob':8s} "
                   f"{'kind':12s} {'status':16s} {'action':10s} pdks")
        for record in records:
            outcome = by_law_id[record.law_id]
            click.echo(f"{record.law_id[:12]:14s} {record.topology_class:32s} "
                       f"{record.metric:12s} {record.knob:8s} {record.quant_kind:12s} "
                       f"{outcome.status:16s} {outcome.action:10s} {','.join(record.pdks)}")
            click.echo(f"    {record.statement}")
            if outcome.action != "unchanged":
                click.echo(f"    SUPPORTED_BY +{outcome.supported_by_written} "
                           f"ABOUT +{outcome.about_written}")

        counts: dict[str, int] = {}
        for o in outcomes:
            counts[o.action] = counts.get(o.action, 0) + 1
        click.echo(f"\n{'APPLIED' if apply_ else 'DRY-RUN'}: " +
                   ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        if not apply_:
            click.echo("(dry run — re-run with --apply to write)")
    finally:
        await store.close()


@main.command("reverify")
@click.option("--sample", "sample_values", multiple=True, default=("nominal",),
              help="'nominal' (default: the 5 nominal law topologies x 3 PDKs), 'all' (every "
                   "seed_recipes() topology_class), or one or more explicit topology_class names "
                   "(repeatable).")
@click.option("--apply", "apply_", is_flag=True,
              help="Re-project ONLY the drifted/value-drift cards through the existing projection "
                   "path and re-feed project-laws (default: dry-run, prints the drift table only)")
@click.option("--corpus", "corpus_root", default=None,
              help="git-corpus root for reading stored verdict_note scalars (VALUE-DRIFT scalar "
                   "check — the graph does not project this field) and, on --apply, for "
                   "re-storing the re-verified specimen. Default: config's [executable] corpus_dir.")
@click.pass_context
def reverify(ctx: click.Context, sample_values: tuple[str, ...], apply_: bool,
             corpus_root: str | None) -> None:
    """Re-verification cycle (E4a, docs/superpowers/specs/2026-07-05-e4-reverification-coherence.md):
    re-run a sample of the corpus's recipes and diff each fresh verdict against the currently-
    projected ClaimCard — CONCORDANT / DRIFTED / VALUE-DRIFT / MISSING / ERROR. Dry-run by default
    (read-only). --apply re-projects ONLY drifted/value-drift cards through the existing projection
    path (journaled as a 'reverify' op, distinct from a first projection) and re-feeds
    project-laws so any law whose member drifted re-derives its status. A second --apply over
    now-concordant data writes nothing."""
    asyncio.run(_reverify(ctx.obj["config"], list(sample_values), apply_, corpus_root))


async def _reverify(config, sample_values: list[str], apply_: bool, corpus_root: str | None) -> None:
    from openclaw_brain.journal import ActionJournal
    from openclaw_brain.knowledge.executable.corpus import SpecimenCorpus
    from openclaw_brain.knowledge.executable.projection import GraphProjector
    from openclaw_brain.knowledge.executable.resolver import GraphResolver
    from openclaw_brain.knowledge.executable.reverify import (
        apply_reverify, refeed_project_laws, reverify_sample, sample_recipes, write_reverify_jsonl,
    )
    from openclaw_brain.knowledge.executable.seeds import seed_recipes
    from openclaw_brain.knowledge.graph.store import GraphStore

    corpus_root_final = corpus_root or str(config.executable.corpus_path)
    corpus = SpecimenCorpus(corpus_root_final)
    recipes = sample_recipes(sample_values or ["nominal"], seed_recipes())

    store = GraphStore(config.neo4j, egress=effective_egress(config))
    await store.connect()
    try:
        click.echo(f"reverify — {'APPLY' if apply_ else 'DRY-RUN'} — "
                   f"sample={','.join(sample_values) or 'nominal'} "
                   f"({len(recipes)} recipe(s) x 3 PDKs)\n")
        from openclaw_brain.knowledge.executable.reverify import _load_replicate_module
        from openclaw_brain.knowledge.executable.runner import NgspiceRunner
        policy = effective_egress(config)
        run_one = lambda recipe, pdk: _load_replicate_module().run_one(
            recipe, pdk, runner_factory=lambda: NgspiceRunner(egress=policy))
        rows = await reverify_sample(store, recipes, corpus=corpus, run_one_fn=run_one)
        counts: dict[str, int] = {}
        for row in rows:
            counts[row.classification] = counts.get(row.classification, 0) + 1
            click.echo(f"{row.classification:11s} {row.topology_class:32s} {row.pdk:10s} "
                       f"{row.claim_id:16s} {row.detail}")
        click.echo(f"\n{'APPLY' if apply_ else 'DRY-RUN'} summary: " +
                   ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) +
                   f" (total {len(rows)})")

        if not apply_:
            click.echo("(dry run — re-run with --apply to re-project drifted/value-drift cards)")
            return

        resolver = GraphResolver(store)
        projector = GraphProjector(store, resolver.resolve)
        journal = ActionJournal(config.state_path)
        outcomes, fresh_records = await apply_reverify(
            store, projector, rows, recipes, corpus=corpus, journal=journal,
            runner_factory=lambda: NgspiceRunner(egress=policy),
        )
        if not outcomes:
            click.echo("\nAPPLY: nothing drifted -- zero writes.")
            return
        for o in outcomes:
            if o.get("error"):
                click.echo(f"  REPROJECT-FAILED {o['topology_class']}/{o['pdk']}: {o['error']}")
            else:
                click.echo(f"  reprojected {o['topology_class']}/{o['pdk']} -> {o['spec_id']} "
                           f"(claims: {', '.join(o['claim_ids'])})")
        click.echo(f"APPLY: {len(outcomes)} (topology_class, pdk) pair(s) re-projected.")

        if fresh_records:
            out_dir = str(Path(__file__).resolve().parent.parent.parent / "experiments")
            jsonl_path = write_reverify_jsonl(fresh_records, out_dir)
            paths = _default_law_jsonl_paths() + [jsonl_path]
            law_outcomes, gap_report = await refeed_project_laws(
                store, paths, about_resolver=resolver.resolve, journal=journal,
            )
            changed = [o for o in law_outcomes if o.action != "unchanged"]
            click.echo(f"project-laws re-feed: {len(changed)}/{len(law_outcomes)} law(s) "
                       f"re-derived (action != unchanged) from {jsonl_path}")
            for o in changed:
                click.echo(f"  {o.law_id[:12]} -> status={o.status} action={o.action}")
    finally:
        await store.close()


@main.command("render-bench")
@click.option("--out", "out_dir", default=None, type=click.Path(file_okay=False),
              help="Output directory for rendered decks (default: bench/primesim_pilot under "
                   "the repo root)")
@click.option("--pilot/--no-pilot", default=True,
              help="Render the S4 PrimeSim pilot benches (ADR-044 D2; currently the only "
                   "supported bench set, so this is on by default -- --no-pilot is accepted "
                   "but has nothing else to render yet).")
@click.pass_context
def render_bench(ctx: click.Context, out_dir: str | None, pilot: bool) -> None:
    """RENDER-ONLY: emit PrimeSim/HSPICE-dialect .sp decks + scope-honest EXPECTATION.md cards
    for the S4 pilot benches (ADR-044 D2's render-only teaching-mode bench drafting). Each
    EXPECTATION.md's expectation values are sourced live (READ-ONLY) from the graph's
    Regularity/ClaimCard nodes, never hand-copied from docs. NEVER invokes ngspice/docker/
    primesim -- pure text generation plus one read-only graph query per bench."""
    if not pilot:
        raise click.UsageError(
            "only --pilot is supported today -- exactly the 4 S4 pilot benches "
            "(render_hspice.PILOT_BENCHES); there is no other bench set yet to render."
        )
    asyncio.run(_render_bench(ctx.obj["config"], out_dir))


async def _render_bench(config, out_dir: str | None) -> None:
    from openclaw_brain.knowledge.executable import render_hspice as rh
    from openclaw_brain.knowledge.graph.store import GraphStore

    repo_root = Path(__file__).resolve().parent.parent.parent
    out_path = Path(out_dir).expanduser() if out_dir else repo_root / "bench" / "primesim_pilot"

    store = GraphStore(config.neo4j, egress=effective_egress(config))
    await store.connect()
    try:
        results = await rh.render_all_pilot_benches(store, out_path)
    except Exception as e:
        click.echo(f"render-bench failed: {e}")
        raise SystemExit(1)
    finally:
        await store.close()      # always release the Neo4j driver (I1), even on a raise above

    for r in results:
        click.echo(
            f"{r['bench_id']}: wrote bench.sp ({r['deck_bytes']}B) + EXPECTATION.md "
            f"({r['card_bytes']}B) -- law={'yes' if r['is_law'] else 'no'}, "
            f"claim_cards={r['claim_cards']}"
        )
    click.echo(f"\nrender-bench: {len(results)} bench(es) written to {out_path}")


@main.command()
@click.option("--transport", default="stdio", type=click.Choice(["stdio", "sse"]),
              help="MCP transport (stdio for CLI agents, sse for HTTP)")
@click.option("--readonly", is_flag=True,
              help="Serve the read-only tool subset (ADR-044 D1 shared-service "
                   "deployment: no write/ingest/curation tools, no EvidenceVault "
                   "verbatim text). Also settable via OPENCLAW_BRAIN_READONLY=1. "
                   "Equivalent to --profile readonly.")
@click.option("--profile", "profile", default=None,
              type=click.Choice(["full", "researcher", "readonly"]),
              help="Tool-surface profile: full (owner session), researcher "
                   "(home-lab research sessions: reads + evidence + record loop + "
                   "learner + oracle-gated projection), readonly (shared-service profile). "
                   "Also settable via OPENCLAW_BRAIN_PROFILE.")
@click.pass_context
def serve(ctx: click.Context, transport: str, readonly: bool, profile: str | None) -> None:
    """Start the MCP server for agent integration."""
    from openclaw_brain.server.mcp_server import create_server

    if readonly and profile and profile != "readonly":
        raise click.UsageError(f"--readonly conflicts with --profile {profile}")
    config_path = ctx.parent.params.get("config_path") if ctx.parent else None
    # An explicit --profile/--readonly always wins; otherwise create_server() falls
    # back to OPENCLAW_BRAIN_PROFILE / OPENCLAW_BRAIN_READONLY=1.
    server = create_server(
        config_path,
        readonly=True if readonly else None,
        profile=profile,
    )
    server.run(transport=transport)


if __name__ == "__main__":
    main()
