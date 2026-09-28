"""Slide analysis — VLM analysis of lecture-slide images (figures-only HTML ingest).

Mirrors ``figure_analyzer.py``'s structure deliberately (timeout constant, consecutive-failure
circuit breaker, ``_extract_text`` response-shape handling) per the task's explicit instruction to
reuse that module's breaker/timeout PATTERN — but this is a separate module, not an import of
figure_analyzer.py's internals: the two analyze fundamentally different inputs (an on-disk MinerU
figure file vs. an in-memory base64 lecture-slide image; see ``SlideImage`` in
``mineru_parser.py``) with a different prompt/output shape (one fixed 4-part JSON extraction, not
a caption-classified prompt table), and this repo's own precedent (html_parser.py's
``_file_checksum`` docstring) is to duplicate small leading-underscore-private helpers across
modules rather than cross-import them — only PUBLIC names (``FigureAnalysis``, ``analyze_figure``,
etc.) are meant to be imported across modules, and none of those fit this module's job.

One deliberate divergence from figure_analyzer.py's breaker semantics (read this before changing
either): figure_analyzer's breaker DEGRADES (tripped figures are kept as light/caption-only stubs,
ingest continues). This module's breaker ABORTS THE WHOLE FILE (raises ``SlideBreakerOpen``)
instead. Rationale: figure_analyzer has no per-figure paid fallback to spiral into — a tripped
figure just stops calling the (single) analysis model. This module's primary is LOCAL (free); its
fallback is a PAID frontier model (``gemini-3.1-pro`` by default). If the local oMLX server is
actually down, "degrade to fallback" would silently re-route potentially thousands of remaining
slides (the undergrad corpus is ~4,000 lecture-slide images) onto the paid model, unattended — a
real, unbounded cost blowup for a machine that just needs a restart. Aborting the file with a
loud, specific error is strictly safer: a human sees exactly why and how many slides were spared,
fixes the local server, and re-runs (which resumes for free — see the checkpoint-driven
resumability in ``pipeline.py::_ingest_html_figures_only``, not duplicated here).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from openclaw_brain.knowledge.extraction.mineru_parser import ContentBlock, SlideImage
from openclaw_brain.knowledge.reasoning.normalize import extract_json
from openclaw_brain.llm.resilience import detect_finish_reason, response_is_truncated

logger = logging.getLogger(__name__)

# Mirrors figure_analyzer.py's _LLM_TIMEOUT exactly. The 2026-07-17 bake-off measured 13-33s/slide
# on the local qwen3-vl-32b primary (real lecture slides); 120s leaves large margin for the paid
# gemini-3.1-pro fallback leg too, without inventing a second, untested constant.
_LLM_TIMEOUT = 120.0

# Mirrors figure_analyzer.py's _FIGURE_BREAKER_THRESHOLD exactly (same reasoning: bound the
# damage of a dead endpoint to a small, fixed number of doomed round-trips before reacting).
_SLIDE_BREAKER_THRESHOLD = 4

# Sequential by default — NOT figure_analyzer.py's _FIGURE_CONCURRENCY=4. Two reasons this
# module chooses differently rather than blindly mirroring: (1) the primary model here is LOCAL
# oMLX, which figure_analyzer.py's own comment already notes "still serializes on the one GPU"
# under concurrent requests — concurrency buys ~nothing on the path that matters; (2) this
# module's breaker must ABORT (see module docstring), and a crisp, zero-overshoot trip (exactly
# N consecutive failures, not "N to 4N depending on in-flight concurrency") is worth more here
# than in figure_analyzer.py's degrade-and-continue case. concurrency remains a real parameter
# (not hardcoded to 1) so a future caller can opt into overlap on the fallback/cloud leg.
_SLIDE_CONCURRENCY = 1

_SLIDE_SYSTEM_PROMPT = (
    "You are an expert analog/semiconductor circuit design instructor analyzing a single lecture "
    "slide image. Extract ONLY what is visually present on the slide — never infer connectivity, "
    "values, or structure beyond what is drawn. Respond with STRICT JSON only: no markdown code "
    "fences, no commentary before or after the JSON object."
)

# Validated 2026-07-17 (orchestrator bake-off, 3 real slides, 3/3 correct LaTeX transcription +
# topology ID, 0 hallucination — see config/default.toml [figures] comment). Kept close to that
# validated wording; safe to iterate on, per the task brief ("이대로 시작, 개선 가능").
_SLIDE_ANALYSIS_PROMPT = """Analyze this lecture slide image. Extract:

1. EQUATIONS: transcribe every mathematical equation visible on the slide to LaTeX, each with a
   short plain-language meaning.
2. SCHEMATIC: if the slide contains a circuit schematic/diagram, identify the EXACT topology name
   (e.g. "common-source amplifier", "five-transistor OTA", "telescopic cascode") and the ROLE of
   each visible device (e.g. "M1: input transistor", "M3/M4: current mirror load"). Pay careful
   attention to single-ended vs differential topology — do not assume a differential structure
   unless two clearly symmetric signal paths are drawn. Do NOT infer wiring/connectivity beyond
   what is visibly drawn — if exact structure is ambiguous, say so rather than guessing. If there
   is no schematic on the slide, this field is null.
3. PLOT: if the slide contains a graph/plot, ONE line describing the axes plus the key trend. If
   there is no plot, this field is null.
4. SUMMARY: a 1-2 sentence summary of what this slide teaches.

Respond with STRICT JSON in exactly this shape (no markdown fences, no extra text before/after):
{"equations": [{"latex": "...", "meaning": "..."}], "schematic": {"topology": "...", "roles": "..."} or null, "plot": "..." or null, "summary": "..."}

If a slide has no equations, use an empty list for "equations". If it has no schematic or no
plot, use null for that field (not an empty string, not an empty object)."""


@dataclass
class SlideAnalysis:
    """Result of analyzing one lecture-slide image."""

    page_num: int
    page_idx: int
    time_range: str = ""
    equations: list[dict[str, str]] = field(default_factory=list)  # [{"latex": ..., "meaning": ...}]
    schematic: dict[str, Any] | None = None
    plot: str | None = None
    summary: str = ""
    model_used: str = ""   # "primary" | "fallback" | "" (neither answered usably)
    latency_s: float = 0.0
    error: str = ""          # non-empty iff both primary and fallback failed (or were skipped)

    @property
    def has_figure_content(self) -> bool:
        """The pipeline gate (spec: "로컬 분석은 공짜이므로 전 슬라이드 분석하되 산출만 게이트") —
        True iff equations/schematic/plot carries SOMETHING. A pure-text/title slide (summary
        only, or a failed slide) must not create a ContentBlock — see
        ``pipeline.py::_ingest_html_figures_only``, the only caller of this property."""
        return bool(self.equations) or self.schematic is not None or self.plot is not None


class SlideBreakerOpen(RuntimeError):
    """Raised by analyze_all_slides() when the local-model consecutive-failure circuit breaker
    trips — see module docstring for why this ABORTS rather than degrades. Callers must treat
    this as a whole-FILE stop, not a per-slide skip."""

    def __init__(self, consec_fail: int, analyzed: int, total: int):
        self.consec_fail = consec_fail
        self.analyzed = analyzed
        self.total = total
        super().__init__(
            f"slide-analysis circuit breaker OPEN after {consec_fail} consecutive primary-model "
            f"failures ({analyzed}/{total} slides attempted before the trip) — the local model "
            f"appears down. Aborting the whole file rather than silently falling back to the "
            f"paid model for the remaining {total - analyzed} slide(s)."
        )


class _ModelCallFailed(Exception):
    """Internal: the primary model cannot produce an answer for STRUCTURAL reasons — the call
    raised or timed out (connectivity), or it came back having hit the output-token ceiling. The
    signal analyze_all_slides() uses to drive its circuit breaker.

    The ceiling case joined this class on 2026-08-31. Liveness is not the only thing the breaker
    protects: a model that never closes its reasoning channel answers every slide, proves the
    endpoint is up, and is unusable on all of them (measured that day — a reasoning VLM burned its
    whole budget on schematic reading with no conclusion, purely because the chat template opened
    a <think> block it could not close). Routing that to _ModelOutputInvalid would do exactly what
    this module's breaker exists to prevent: silently re-route every remaining slide to the PAID
    frontier fallback, one full timeout at a time, while the counter stays at zero.
    """


class _ModelOutputInvalid(Exception):
    """Internal: the LLM call SUCCEEDED and COMPLETED, but its response text contained no parseable
    JSON. Deliberately NOT a breaker signal — mirrors figure_analyzer.py's "a missing/unreadable
    image is neutral, only a model-side failure trips the breaker" distinction: a finished
    response, even a useless one, is evidence about THIS slide rather than about the model."""


def _extract_text(response: Any) -> str:
    """Extract text from an LLM response, handling both str and list content — duplicated from
    figure_analyzer.py's identical private helper rather than cross-imported (see module
    docstring; matches html_parser.py's _file_checksum precedent for small private helpers)."""
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts).strip()
    return str(content).strip()


async def _invoke_slide_model(slide: SlideImage, llm: BaseChatModel) -> dict[str, Any]:
    """Call one VLM on one slide image; return the raw parsed JSON dict (whatever shape it came
    back in — field coercion happens in _coerce_slide_analysis).

    Raises _ModelCallFailed (connectivity/timeout) or _ModelOutputInvalid (call succeeded, no
    parseable JSON) — see those classes' docstrings for why callers treat them differently.
    """
    raw_bytes = slide.data()
    b64 = base64.b64encode(raw_bytes).decode("ascii")
    data_url = f"data:{slide.mime_type};base64,{b64}"

    try:
        response = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(content=_SLIDE_SYSTEM_PROMPT),
                HumanMessage(content=[
                    {"type": "text", "text": _SLIDE_ANALYSIS_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]),
            ]),
            timeout=_LLM_TIMEOUT,
        )
    except Exception as e:
        raise _ModelCallFailed(str(e)) from e

    # Checked BEFORE the JSON parse: a ceiling-hit response usually fails extract_json anyway, but
    # it would land as _ModelOutputInvalid — "bad answer for this slide" — when it is really "this
    # model cannot finish an answer", which repeats identically on every remaining slide.
    if response_is_truncated(response):
        raise _ModelCallFailed(
            f"output ceiling hit (finish_reason={detect_finish_reason(response)!r}) — "
            "response never completed"
        )

    text = _extract_text(response)
    try:
        # extract_json (knowledge/reasoning/normalize.py) already tolerates markdown code fences
        # around the JSON — the "mock-fidelity" lesson (PHILOSOPHY.md P11) is exactly that local
        # models often wrap structured output in ```json ... ``` despite being told not to.
        return extract_json(text)
    except ValueError as e:
        raise _ModelOutputInvalid(str(e)) from e


def _coerce_slide_analysis(
    raw: dict[str, Any], slide: SlideImage, model_used: str, latency_s: float,
) -> SlideAnalysis:
    """Tolerantly coerce a raw (possibly partially-malformed) LLM JSON dict into SlideAnalysis
    fields — lenient, never raises: a slide-analysis JSON with one bad field should still yield
    whatever DID parse, not nuke the whole slide (mirrors reasoning/normalize.py's general
    philosophy of tolerance for local-model JSON, applied to this module's own small schema)."""
    equations: list[dict[str, str]] = []
    for item in raw.get("equations") or []:
        if isinstance(item, dict) and item.get("latex"):
            equations.append({
                "latex": str(item["latex"]),
                "meaning": str(item.get("meaning") or ""),
            })

    schematic = raw.get("schematic")
    if not isinstance(schematic, dict):
        schematic = None

    plot = raw.get("plot")
    if not isinstance(plot, str) or not plot.strip():
        plot = None

    summary = raw.get("summary")
    summary = str(summary).strip() if summary else ""

    return SlideAnalysis(
        page_num=slide.page_num,
        page_idx=slide.page_idx,
        time_range=slide.time_range,
        equations=equations,
        schematic=schematic,
        plot=plot,
        summary=summary,
        model_used=model_used,
        latency_s=latency_s,
    )


async def analyze_slide(
    slide: SlideImage,
    primary_llm: BaseChatModel,
    fallback_llm: BaseChatModel | None = None,
) -> tuple[SlideAnalysis, bool]:
    """Analyze one slide: try primary_llm, then fallback_llm (if given) on failure.

    Returns (analysis, primary_failed). ``primary_failed`` is True only for a _ModelCallFailed on
    the PRIMARY attempt specifically — the breaker-relevant signal analyze_all_slides() drives its
    consecutive-failure counter from. A _ModelOutputInvalid on the primary (it responded, just
    unusably) does NOT set this — see that class's docstring.
    """
    start = time.monotonic()
    primary_failed = False
    primary_error = ""
    try:
        raw = await _invoke_slide_model(slide, primary_llm)
        return _coerce_slide_analysis(raw, slide, "primary", time.monotonic() - start), False
    except _ModelCallFailed as e:
        primary_failed = True
        primary_error = str(e)
    except _ModelOutputInvalid as e:
        primary_error = str(e)

    if fallback_llm is None:
        return SlideAnalysis(
            page_num=slide.page_num, page_idx=slide.page_idx, time_range=slide.time_range,
            error=f"primary failed: {primary_error}",
            latency_s=time.monotonic() - start,
        ), primary_failed

    try:
        raw = await _invoke_slide_model(slide, fallback_llm)
        analysis = _coerce_slide_analysis(raw, slide, "fallback", time.monotonic() - start)
        return analysis, primary_failed
    except (_ModelCallFailed, _ModelOutputInvalid) as e:
        return SlideAnalysis(
            page_num=slide.page_num, page_idx=slide.page_idx, time_range=slide.time_range,
            error=f"primary failed: {primary_error}; fallback failed: {e}",
            latency_s=time.monotonic() - start,
        ), primary_failed


async def analyze_all_slides(
    slides: list[SlideImage],
    primary_llm: BaseChatModel,
    fallback_llm: BaseChatModel | None = None,
    concurrency: int = _SLIDE_CONCURRENCY,
    breaker_threshold: int = _SLIDE_BREAKER_THRESHOLD,
    on_slide_done: Callable[[SlideImage, SlideAnalysis], None] | None = None,
) -> list[SlideAnalysis]:
    """Analyze every slide, in page order, with the local-server-down circuit breaker (see module
    docstring). Every slide is attempted regardless of eventual gate content — the caller
    (pipeline.py) is what decides whether a given SlideAnalysis becomes a ContentBlock.

    Args:
        slides: slide images to analyze (already filtered by the caller if resuming — this
            function has no checkpoint awareness of its own, see PipelineCheckpoint.save_extra/
            load_extra usage in pipeline.py::_ingest_html_figures_only).
        primary_llm / fallback_llm: as analyze_slide().
        concurrency: bounded fan-out (default 1 — see _SLIDE_CONCURRENCY's rationale above).
        breaker_threshold: consecutive PRIMARY-call failures before tripping (default 4).
        on_slide_done: optional callback invoked after each slide that was ACTUALLY attempted
            (never for a slide skipped because the breaker had already tripped) with
            (slide, analysis) — the pipeline's resumability checkpoint persists here. Not firing
            for skipped slides matters: persisting a "skipped" stub as if it were a real result
            would make a later resume wrongly treat that slide as permanently done. Exceptions
            from this callback are NOT caught — a checkpoint-write bug should surface loudly, not
            be swallowed alongside VLM-call failures.

    Returns:
        One SlideAnalysis per input slide, in input order — UNLESS the breaker trips, in which
        case SlideBreakerOpen is raised instead of returning (see that class's docstring; any
        completed results are still visible to on_slide_done as they happen).
    """
    if not slides:
        return []

    sem = asyncio.Semaphore(max(1, concurrency))
    breaker = {"consec_fail": 0, "tripped": False}

    async def _process(slide: SlideImage) -> SlideAnalysis | None:
        """None means "skipped, breaker already open" — deliberately distinct from a real
        SlideAnalysis so the caller never mistakes a skip for a completed attempt (see
        on_slide_done's docstring above)."""
        async with sem:
            if breaker["tripped"]:
                return None
            analysis, primary_failed = await analyze_slide(slide, primary_llm, fallback_llm)
            if primary_failed:
                breaker["consec_fail"] += 1
                if breaker["consec_fail"] >= breaker_threshold:
                    breaker["tripped"] = True
            else:
                breaker["consec_fail"] = 0
            if on_slide_done is not None:
                on_slide_done(slide, analysis)
            return analysis

    if concurrency <= 1 or len(slides) <= 1:
        # Sequential default: stop launching new work the INSTANT the breaker trips (zero
        # overshoot) rather than looping through every remaining slide just to skip it.
        attempted: list[SlideAnalysis] = []
        for slide in slides:
            if breaker["tripped"]:
                break
            attempted.append(await _process(slide))
    else:
        # Bounded fan-out: tasks already scheduled before the trip may still run past it (some
        # in-flight overshoot, bounded by `concurrency`); any task that starts AFTER the trip
        # returns None immediately without a model call (see _process). Filter Nones out — same
        # "only real attempts count" contract as the sequential path above.
        raw = await asyncio.gather(*(_process(slide) for slide in slides))
        attempted = [r for r in raw if r is not None]

    if breaker["tripped"]:
        raise SlideBreakerOpen(breaker["consec_fail"], len(attempted), len(slides))

    return attempted


# ── SlideAnalysis -> ContentBlock (feeds the existing chunk->...->commit pipeline unchanged) ──


def _format_slide_body(sa: SlideAnalysis) -> str:
    """Structured plain-text rendering of one slide's analysis — this IS the extractor's input
    (no FIG_VLM sentinel wrapping: unlike a PDF figure embedded beside independent page prose,
    here the ENTIRE block text is VLM-derived, so grounding.py's anti-self-grounding sentinel
    would exclude everything and gut the ingest — see pipeline.py::_ingest_html_figures_only for
    the full rationale)."""
    parts: list[str] = []
    if sa.equations:
        lines = []
        for eq in sa.equations:
            meaning = eq.get("meaning", "")
            lines.append(f"- ${eq['latex']}$: {meaning}" if meaning else f"- ${eq['latex']}$")
        parts.append("Equations:\n" + "\n".join(lines))
    if sa.schematic:
        topology = sa.schematic.get("topology", "")
        roles = sa.schematic.get("roles", "")
        text = f"Schematic — topology: {topology}" if topology else "Schematic"
        if roles:
            text += f"\nDevice roles: {roles}"
        parts.append(text)
    if sa.plot:
        parts.append(f"Plot: {sa.plot}")
    if sa.summary:
        parts.append(f"Summary: {sa.summary}")
    return "\n\n".join(parts)


def slide_analysis_to_blocks(sa: SlideAnalysis) -> tuple[ContentBlock, ContentBlock]:
    """One (heading, body) ContentBlock pair for a gated SlideAnalysis — same text_level=1/0
    convention html_parser.py's text path uses, so chunk_structured() groups exactly one
    chunk-section per slide, unchanged. Caller (pipeline.py) is responsible for gating on
    ``sa.has_figure_content`` BEFORE calling this — it does not re-check the gate itself."""
    heading_text = f"p.{sa.page_num} slide (figure)"
    if sa.time_range:
        heading_text += f" ({sa.time_range})"
    heading = ContentBlock(type="text", text=heading_text, page_idx=sa.page_idx, text_level=1)
    body = ContentBlock(type="text", text=_format_slide_body(sa), page_idx=sa.page_idx, text_level=0)
    return heading, body
