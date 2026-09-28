"""Figure analysis — classify and analyze figures from parsed PDFs.

Step 1: Caption-based classification (no LLM needed)
Step 2: VLM classification for unknown figures (qwen3-vl:8b)
Step 3: Domain-specific analysis with the configured figure-analysis VLM (config: default_figure_analysis)

Figure types: circuit, plot, block_diagram, layout, photo, other
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LLM_TIMEOUT = 120.0  # seconds per API call
# Figure analysis is a SERIAL PREFIX (pipeline Stage 0.5, before the concurrent chunk loop), so its
# wall-clock is sum-of-figures and is NOT helped by chunk_concurrency. Process figures with bounded
# concurrency so the cloud analysis round-trips overlap (local oMLX classify still serializes on the
# one GPU, which a modest cap respects). Model-independent latency win. 1 = original sequential.
_FIGURE_CONCURRENCY = 4
# Only these figure types get the EXPENSIVE VLM deep-analysis (the grok/vision round-trip). Other
# types (plot, layout, other, equation-render fragments) are still parsed + classified + kept (their
# caption is already in the MinerU text), they just skip the costly per-figure VLM call. This is what
# makes a figure-dense book (Razavi: 1175 image blocks) tractable: deep-analyze the schematics, not
# every rendered fragment. NOT excluding images — only gating which ones cost a VLM call.
_DEEP_FIGURE_TYPES = ("circuit", "block_diagram")
# Figure-analysis circuit breaker. `analyze_figure` already bounds each call at `_LLM_TIMEOUT`, but
# when the remote deep-analysis model is entirely down EVERY schematic times out — a figure-dense book
# then grinds for hours (one full timeout per figure) before the pipeline reaches its first text chunk
# (observed live 2026-07-07: gemini-3.1-pro bad window stalled a resume ~35min at 0% CPU). Once this
# many deep analyses fail in a row the model is treated as down and the remaining schematics skip the
# doomed VLM round-trip (kept as light/caption-only), bounding the waste to threshold×_LLM_TIMEOUT.
# Mirrors the consecutive-failure breaker on the text path (llm/resilience.py).
_FIGURE_BREAKER_THRESHOLD = 4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from openclaw_brain.knowledge.extraction.mineru_parser import ContentBlock
from openclaw_brain.llm.resilience import detect_finish_reason, response_is_truncated


# A figure analysis that hit the output ceiling did not finish, and an unfinished analysis must not
# be injected into a chunk as if it had. Two shapes reach that ceiling and they are NOT the same
# defect (THR spec ④(d)): budget starvation truncates a coherent answer, non-convergence emits a
# huge self-reversing monologue with no conclusion (measured 2026-08-31: GLM-5.3-Flash spent 12,288
# tokens on a schematic with no final answer, while converging cleanly on plots). Telling them apart
# matters when CHOOSING a model; it does not matter when ACCEPTING one response — either way the
# description is unfit, and this file's own circuit prompt already states the governing principle
# ("a WRONG connection is worse than a missing one here").
# Backstop for providers that report no finish_reason. Measured 2026-08-31 across models: a
# no-schematic control answers in 680 chars, a plot in ~2.2K, a full block-level schematic
# description in 3-5K; the runaway ran 45K. 20K is >4x the legitimate ceiling and <half the
# runaway, so it has margin on both sides.
_MAX_DESCRIPTION_CHARS = 20_000


def _completion_defect(response, text: str) -> str | None:
    """Return why this response is unusable as a figure analysis, or None if it completed.

    Checks the ceiling, not the content: a response that stopped because it ran out of output
    budget is incomplete regardless of whether it was starved or never converging.

    The finish_reason half delegates to the resilience layer, which already normalizes every
    provider spelling and response shape in this stack — the text path has had that detection
    since the truncation work; figure analysis simply never went through it, because it calls
    `llm.ainvoke` directly rather than `invoke_with_resilience`.

    Borrowing the DETECTOR but not the REMEDY is deliberate. `_retry_once_on_truncation` retries
    once at 1.5x the bound, which is the right move for budget starvation (the text path's usual
    cause) and the wrong one for non-convergence: a model that cannot reach a conclusion returns
    the same non-answer for 1.5x the spend. Here the breaker is the remedy instead — it stops
    trying rather than trying harder.
    """
    if response_is_truncated(response):
        reason = detect_finish_reason(response)
        return f"incomplete response (finish_reason={reason!r}, {len(text)} chars)"
    if len(text) > _MAX_DESCRIPTION_CHARS:
        return f"runaway response ({len(text)} chars > {_MAX_DESCRIPTION_CHARS} cap)"
    return None


def _extract_text(response) -> str:
    """Extract text from an LLM response, handling both str and list content."""
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        # Multimodal response — extract text parts
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts).strip()
    return str(content).strip()

logger = logging.getLogger(__name__)


# ── Figure type definitions ──

FIGURE_TYPES = ("circuit", "plot", "block_diagram", "layout", "photo", "other")


@dataclass
class FigureAnalysis:
    """Result of analyzing a single figure."""

    figure_type: str  # One of FIGURE_TYPES
    description: str  # Detailed analysis text
    page_idx: int
    caption: str = ""
    footnote: str = ""  # real document anchor (independent of VLM prose)
    img_path: str = ""


# ── Step 1: Caption-based classification ──

_CAPTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Order matters: more specific patterns first to avoid false matches.
    # "layout" before "circuit" (both could match "OTA")
    ("layout", re.compile(
        r"(?:layout|die[\s\-]?photo|micrograph|chip[\s\-]?photo|floorplan|"
        r"cross[\s\-]?section)",
        re.IGNORECASE,
    )),
    # "photo" before "plot" ("measurement setup" should be photo, not plot)
    ("photo", re.compile(
        r"(?:photograph|test[\s\-]?bench|pcb[\s\-]?board|equipment[\s\-]?setup|"
        r"measurement[\s\-]?setup)",
        re.IGNORECASE,
    )),
    # "plot" before "circuit" ("frequency response" should be plot, not circuit via "amplifier")
    ("plot", re.compile(
        r"(?:response|characteristic|plot|curve|spectrum|waveform|transient|"
        r"bode|nyquist|gain[\s\-]?vs|phase[\s\-]?vs|magnitude|vs\.[\s]|versus|"
        r"simulation[\s\-]?result|monte[\s\-]carlo|corner[\s\-]?analysis|"
        r"i[\-_]v|c[\-_]v|transfer[\s]function|frequency[\s\-]?response|"
        r"measured[\s\-]?result|noise[\s\-]?spectrum)",
        re.IGNORECASE,
    )),
    ("block_diagram", re.compile(
        r"(?:block[\s\-]?diagram|system[\s\-]?level|overview|"
        r"signal[\s\-]?flow|data[\s\-]?path|functional[\s\-]?diagram)",
        re.IGNORECASE,
    )),
    # "circuit" last — broadest pattern, catches remaining schematics
    ("circuit", re.compile(
        r"(?:schematic|circuit[\s\-]?diagram|transistor[\s\-]?level|"
        r"opamp|op[\-\s]?amp|ota[\s\-]|ldo|bandgap|bias[\s\-]?circuit|"
        r"current[\s\-]?mirror|cascode|differential[\s\-]pair)",
        re.IGNORECASE,
    )),
]


def classify_by_caption(caption: str) -> str:
    """Classify a figure by its caption text using regex patterns.

    Returns figure type or "unknown" if no pattern matches.
    """
    if not caption:
        return "unknown"

    for fig_type, pattern in _CAPTION_PATTERNS:
        if pattern.search(caption):
            return fig_type

    return "unknown"


# ── Step 2: VLM classification for unknowns ──

_CLASSIFY_PROMPT = """Look at this image from a semiconductor/analog circuit design paper.
Classify it as exactly ONE of these types:
- circuit: circuit schematic or transistor-level diagram
- plot: graph, chart, measurement plot, simulation result
- block_diagram: system-level block diagram or signal flow
- layout: chip layout, die photo, cross-section
- photo: equipment photo, test setup, PCB
- other: none of the above

Respond with ONLY the type name, nothing else."""


async def classify_with_vlm(
    img_path: str,
    llm: BaseChatModel,
) -> str:
    """Classify a figure using a VLM when caption-based classification fails."""
    b64_image = _load_image_b64(img_path)
    if not b64_image:
        return "other"

    try:
        response = await asyncio.wait_for(
            llm.ainvoke([
                HumanMessage(content=[
                    {"type": "text", "text": _CLASSIFY_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{_get_mime_type(img_path)};base64,{b64_image}"}},
                ]),
            ]),
            timeout=_LLM_TIMEOUT,
        )
        # No ceiling check here, unlike analyze_figure: this call's output is matched against a
        # CLOSED vocabulary, so a truncated or runaway response simply misses every type and lands
        # on the safe "other" default (which skips deep analysis and keeps the figure caption-only).
        # A constrained output space is its own truncation guard; free text is what needs the check.
        result = _extract_text(response).lower()
        if result in FIGURE_TYPES:
            return result
        # Try to find a valid type in the response
        for ft in FIGURE_TYPES:
            if ft in result:
                return ft
        return "other"
    except Exception as e:
        logger.warning("VLM classification failed for %s: %s", img_path, e)
        return "other"


# ── Step 3: Type-specific analysis with frontier model ──

_ANALYSIS_PROMPTS = {
    "circuit": """Analyze this circuit schematic from a semiconductor/analog design paper at the
BLOCK / ROLE level — this feeds a circuit-reasoning knowledge graph, NOT an EDA netlist tool.

Extract:
1. **Topology**: the circuit architecture (e.g., folded cascode OTA, two-stage Miller, telescopic
   cascode, current mirror).
2. **Functional blocks and their ROLE**: name each block by the JOB it does — input pair, tail
   current source, cascode device, active/diode load, compensation network, CMFB — not by transistor
   reference designators.
3. **Block-level functional relations (the "why it works")**: state cause/role relations between
   blocks in plain terms, e.g. "the tail current source SETS the bias current and gm of the input
   pair", "the cascode device RAISES the output impedance and therefore the DC gain", "Miller
   compensation SETS the dominant pole". These causal/structural relations are the load-bearing output.
4. **Key design features / bias**: notable techniques and any clearly-labeled bias voltages/currents.

CRITICAL — do NOT assert transistor-level wiring or netlist connectivity (which device terminal
connects to which). Schematic-precise connectivity is not reliably readable and a WRONG connection is
worse than a missing one here. Describe only block-level signal flow you can clearly see; if exact
structure is ambiguous, say so rather than guessing. Use standard analog design terminology.""",

    "plot": """Analyze this measurement/simulation plot from a semiconductor/analog design paper.

Extract:
1. **Axes**: What are the X and Y axes? Include units and scale type (linear/log/dB)
2. **Curves**: How many curves? What does each represent?
3. **Key data points**: Read specific numeric values where visible:
   - For Bode plots: gain, bandwidth, unity-gain frequency, phase margin, gain margin
   - For I-V curves: threshold voltage, saturation current, on/off ratio
   - For transient: rise/fall times, settling time, overshoot
   - For noise: noise floor, corner frequency, integrated noise
4. **Trends**: Describe the overall behavior and any notable features
5. **Conditions**: Note any labeled conditions (temperature, process corner, load)

Read numeric values as precisely as possible from the graph.""",

    "block_diagram": """Analyze this block diagram from a semiconductor/analog design paper.

Extract:
1. **Functional blocks**: List each block and its function
2. **Signal flow**: Describe the connections and data/signal flow between blocks
3. **Feedback loops**: Identify any feedback paths
4. **Interfaces**: Note input/output signals and their types (analog, digital, clock)
5. **Architecture**: Classify the overall architecture type (pipeline, parallel, feedback, etc.)

Use standard engineering terminology.""",

    "layout": """Analyze this layout/die photo from a semiconductor/analog design paper.

Extract:
1. **Die/block area**: Note dimensions if visible
2. **Identified blocks**: Label visible functional blocks
3. **Technology features**: Note visible technology indicators (metal layers, routing density)
4. **Symmetry**: Note any symmetry in the layout (common for analog matching)

Be factual about what is visually identifiable.""",

    "photo": """Describe this photograph from a semiconductor/analog design paper.

Identify what is shown: test equipment, PCB, measurement setup, etc.
Note any visible labels, instrument models, or notable features.""",

    "other": """Describe the content of this figure from a technical engineering paper.

Extract all relevant technical information visible in the image.""",
}


async def analyze_figure(
    figure: ContentBlock,
    llm: BaseChatModel,
    figure_type: str | None = None,
) -> FigureAnalysis:
    """Analyze a single figure with a frontier VLM.

    Args:
        figure: ContentBlock with type="image" and img_path.
        llm: the configured figure-analysis VLM (config: default_figure_analysis).
        figure_type: Pre-classified type. If None, defaults to "other".

    Returns:
        FigureAnalysis with type and description.
    """
    if figure_type is None:
        figure_type = "other"

    b64_image = _load_image_b64(figure.img_path)
    if not b64_image:
        return FigureAnalysis(
            figure_type=figure_type,
            description=f"[Figure could not be loaded: {figure.img_path}]",
            page_idx=figure.page_idx,
            caption=figure.caption,
            footnote=figure.footnote,
            img_path=figure.img_path,
        )

    prompt = _ANALYSIS_PROMPTS.get(figure_type, _ANALYSIS_PROMPTS["other"])

    # Add caption context if available
    if figure.caption:
        prompt = f"Caption: \"{figure.caption}\"\n\n{prompt}"

    try:
        response = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(content="You are an expert semiconductor and analog circuit design engineer analyzing figures from technical papers."),
                HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{_get_mime_type(figure.img_path)};base64,{b64_image}"}},
                ]),
            ]),
            timeout=_LLM_TIMEOUT,
        )
        description = _extract_text(response)
        # Reuse the "[Analysis failed" marker deliberately: the breaker below counts CONSECUTIVE
        # occurrences, so an isolated truncation is absorbed while a model that never converges on
        # schematics trips the breaker and stops burning a full budget per figure — which is exactly
        # what the breaker exists to bound.
        defect = _completion_defect(response, description)
        if defect:
            logger.warning("Figure analysis %s for %s", defect, figure.img_path)
            description = f"[Analysis failed: {defect}]"
    except Exception as e:
        logger.warning("Figure analysis failed for %s: %s", figure.img_path, e)
        description = f"[Analysis failed: {e}]"

    return FigureAnalysis(
        figure_type=figure_type,
        description=description,
        page_idx=figure.page_idx,
        caption=figure.caption,
        footnote=figure.footnote,
        img_path=figure.img_path,
    )


async def analyze_all_figures(
    figures: list[ContentBlock],
    analysis_llm: BaseChatModel,
    classify_llm: BaseChatModel | None = None,
    concurrency: int = _FIGURE_CONCURRENCY,
    deep_types: tuple[str, ...] = _DEEP_FIGURE_TYPES,
) -> list[FigureAnalysis]:
    """Classify and analyze all figures from a parsed document.

    Every figure is classified and kept (results stay 1:1 with ``figures``), but only those whose
    type is in ``deep_types`` (schematics / block diagrams) pay the expensive per-figure VLM
    deep-analysis. Other types are returned as a lightweight classified stub (no VLM call, empty
    description — their caption is already in the MinerU text). This is what makes a figure-dense
    book tractable (Razavi: 1175 image blocks → only the schematics cost a VLM round-trip).

    Args:
        figures: List of ContentBlock with type="image".
        analysis_llm: the configured figure-analysis VLM (config: default_figure_analysis).
        classify_llm: Small VLM for classifying unknowns (qwen3-vl:8b).
                      If None, unknowns default to "other".
        concurrency: Max figures processed at once (``1`` = original strictly-sequential behavior).
                     Bounded so cloud analysis round-trips overlap; results stay in figure order.
        deep_types: Figure types that get the expensive VLM deep-analysis (default circuit/block).

    Returns:
        List of FigureAnalysis results, in the same order as ``figures``.
    """
    sem = asyncio.Semaphore(max(1, concurrency))
    stats = {"deep": 0, "light": 0, "skipped_model_down": 0}
    # Shared consecutive-failure breaker (see _FIGURE_BREAKER_THRESHOLD). Plain dict mutated only at
    # await-free points inside the coroutines, so no lock is needed under the bounded-concurrency gather.
    fig_breaker = {"consec_fail": 0, "tripped": False}

    def _light(figure: ContentBlock, fig_type: str) -> FigureAnalysis:
        # Classified + kept (not excluded), but no VLM call and nothing to inject.
        return FigureAnalysis(
            figure_type=fig_type, description="", page_idx=figure.page_idx,
            caption=figure.caption, footnote=figure.footnote, img_path=figure.img_path,
        )

    async def _process(figure: ContentBlock) -> FigureAnalysis:
        async with sem:
            try:
                # Step 1: Caption-based classification (cheap)
                fig_type = classify_by_caption(figure.caption)

                # Step 2: VLM classification only for caption-less unknowns
                if fig_type == "unknown" and classify_llm is not None:
                    fig_type = await classify_with_vlm(figure.img_path, classify_llm)

                if fig_type == "unknown":
                    fig_type = "other"

                # Step 3: Expensive VLM deep-analysis — ONLY for schematics / block diagrams.
                if fig_type in deep_types:
                    if fig_breaker["tripped"]:
                        # Model already judged down — skip the doomed round-trip, keep caption-only.
                        stats["skipped_model_down"] += 1
                        return _light(figure, fig_type)
                    analysis = await analyze_figure(figure, analysis_llm, fig_type)
                    stats["deep"] += 1
                    # Only a model-side failure ("[Analysis failed…") trips the breaker; a
                    # missing/unreadable image ("[Figure could not be loaded…") is a local issue.
                    if analysis.description.startswith("[Analysis failed"):
                        fig_breaker["consec_fail"] += 1
                        if fig_breaker["consec_fail"] >= _FIGURE_BREAKER_THRESHOLD and not fig_breaker["tripped"]:
                            fig_breaker["tripped"] = True
                            logger.warning(
                                "Figure-analysis circuit breaker TRIPPED after %d consecutive failures "
                                "— deep-analysis model appears down; remaining schematics skip it "
                                "(kept as light/caption-only).",
                                fig_breaker["consec_fail"],
                            )
                    elif not analysis.description.startswith("[Figure could not be loaded"):
                        # A real description proves the model is up → reset the streak. A
                        # missing/unreadable image made NO model call, so it is NEUTRAL: neither
                        # increment nor reset (resetting on it would let interspersed missing images
                        # clear a genuine model-outage streak and stop the breaker from ever tripping).
                        fig_breaker["consec_fail"] = 0
                    return analysis

                stats["light"] += 1
                return _light(figure, fig_type)
            except Exception as e:  # one bad figure must never sink the whole prefix
                logger.warning("Figure processing failed (page %d): %s", figure.page_idx, e)
                return _light(figure, "other")

    if concurrency <= 1 or len(figures) <= 1:
        # Original strictly-sequential path.
        results = [await _process(figure) for figure in figures]
        logger.info("Figure analysis: %d deep (%s), %d light, %d skipped(model-down), of %d",
                    stats["deep"], "/".join(deep_types), stats["light"],
                    stats["skipped_model_down"], len(figures))
        return results

    # Bounded fan-out — asyncio.gather preserves input order, so results stay figure-aligned.
    results = list(await asyncio.gather(*(_process(figure) for figure in figures)))
    logger.info("Figure analysis: %d deep (%s), %d light, of %d",
                stats["deep"], "/".join(deep_types), stats["light"], len(figures))
    return results


def _get_mime_type(img_path: str) -> str:
    """Detect image MIME type from file bytes, falling back to extension."""
    try:
        header = Path(img_path).read_bytes()[:16]
    except Exception:
        header = b""

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    if header.startswith(b"BM"):
        return "image/bmp"

    ext = Path(img_path).suffix.lower()
    return "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"


def _load_image_b64(img_path: str) -> str | None:
    """Load an image file and return base64-encoded string."""
    path = Path(img_path)
    if not path.exists():
        logger.warning("Image not found: %s", img_path)
        return None
    try:
        data = path.read_bytes()
        return base64.b64encode(data).decode("utf-8")
    except Exception as e:
        logger.warning("Failed to load image %s: %s", img_path, e)
        return None
