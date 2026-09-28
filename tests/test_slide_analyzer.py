"""Tests for lecture-slide VLM analysis (knowledge/extraction/slide_analyzer.py).

Mock fidelity (CLAUDE.md Testing Conventions / PHILOSOPHY.md P11): LLM mocks use AsyncMock for
the model (``.ainvoke`` is awaited) with a plain MagicMock(content=...) return value — the exact
shape figure_analyzer.py's own call site consumes (``response.content``), matching
tests/test_figure_analyzer.py's established convention. JSON payloads are exercised both as raw
text and wrapped in markdown code fences (local models routinely emit ```json ... ``` despite
being told not to — the mock-fidelity lesson this repo already paid for once).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from openclaw_brain.knowledge.extraction.mineru_parser import SlideImage
from openclaw_brain.knowledge.extraction.slide_analyzer import (
    SlideAnalysis,
    SlideBreakerOpen,
    _SLIDE_BREAKER_THRESHOLD,
    analyze_all_slides,
    analyze_slide,
    slide_analysis_to_blocks,
)

_MIN_PNG = (
    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
    b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
    b'\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00'
    b'\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
)


def _slide(page_num: int = 1, time_range: str = "") -> SlideImage:
    return SlideImage(
        page_num=page_num, page_idx=page_num - 1, time_range=time_range,
        mime_type="image/png", data=lambda: _MIN_PNG,
    )


def _llm_returning(content) -> AsyncMock:
    llm = AsyncMock()
    llm.ainvoke.return_value = MagicMock(content=content)
    return llm


def _llm_raising(exc: Exception) -> AsyncMock:
    llm = AsyncMock()
    llm.ainvoke.side_effect = exc
    return llm


_FULL_JSON = (
    '{"equations": [{"latex": "I_D = k(V_{GS}-V_{TH})^2", "meaning": "MOSFET saturation current"}], '
    '"schematic": {"topology": "common-source amplifier", "roles": "M1: input transistor"}, '
    '"plot": "Gain rolls off above the corner frequency", "summary": "CS amplifier overview"}'
)
_TITLE_ONLY_JSON = '{"equations": [], "schematic": null, "plot": null, "summary": "Title slide"}'


# ── analyze_slide: primary/fallback routing ──


@pytest.mark.asyncio
async def test_analyze_slide_primary_success_no_fallback_called():
    primary = _llm_returning(_FULL_JSON)
    fallback = AsyncMock()

    analysis, primary_failed = await analyze_slide(_slide(), primary, fallback)

    assert primary_failed is False
    assert analysis.model_used == "primary"
    assert analysis.equations == [{"latex": "I_D = k(V_{GS}-V_{TH})^2", "meaning": "MOSFET saturation current"}]
    assert analysis.schematic == {"topology": "common-source amplifier", "roles": "M1: input transistor"}
    assert analysis.plot == "Gain rolls off above the corner frequency"
    assert analysis.summary == "CS amplifier overview"
    assert analysis.error == ""
    fallback.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_slide_response_wrapped_in_markdown_fence():
    """Mock-fidelity target: local models routinely wrap JSON in ```json fences."""
    fenced = f"```json\n{_FULL_JSON}\n```"
    primary = _llm_returning(fenced)

    analysis, primary_failed = await analyze_slide(_slide(), primary)

    assert primary_failed is False
    assert analysis.equations[0]["latex"] == "I_D = k(V_{GS}-V_{TH})^2"


@pytest.mark.asyncio
async def test_analyze_slide_multimodal_list_content_shape():
    """A response whose .content is a list of content parts (some providers' multimodal shape)
    rather than a plain string — _extract_text must handle both, mirroring figure_analyzer.py."""
    primary = _llm_returning([{"type": "text", "text": _TITLE_ONLY_JSON}])

    analysis, primary_failed = await analyze_slide(_slide(), primary)

    assert primary_failed is False
    assert analysis.has_figure_content is False
    assert analysis.summary == "Title slide"


@pytest.mark.asyncio
async def test_analyze_slide_primary_timeout_falls_back():
    primary = _llm_raising(TimeoutError("connection refused"))
    fallback = _llm_returning(_FULL_JSON)

    analysis, primary_failed = await analyze_slide(_slide(), primary, fallback)

    assert primary_failed is True
    assert analysis.model_used == "fallback"
    assert analysis.has_figure_content is True
    fallback.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_analyze_slide_primary_timeout_no_fallback_configured():
    primary = _llm_raising(TimeoutError("down"))

    analysis, primary_failed = await analyze_slide(_slide(), primary, fallback_llm=None)

    assert primary_failed is True
    assert analysis.model_used == ""
    assert "primary failed" in analysis.error
    assert analysis.has_figure_content is False


@pytest.mark.asyncio
async def test_analyze_slide_both_primary_and_fallback_fail():
    primary = _llm_raising(TimeoutError("primary down"))
    fallback = _llm_raising(TimeoutError("fallback down too"))

    analysis, primary_failed = await analyze_slide(_slide(), primary, fallback)

    assert primary_failed is True
    assert "primary failed" in analysis.error and "fallback failed" in analysis.error
    assert analysis.has_figure_content is False


@pytest.mark.asyncio
async def test_analyze_slide_primary_unparseable_json_is_not_a_call_failure():
    """The primary RESPONDS (proving liveness) but with garbage — must fall back, but must NOT
    be reported as a primary call failure (see _ModelOutputInvalid's docstring: this is the
    signal analyze_all_slides() uses to avoid tripping the breaker on a quality miss)."""
    primary = _llm_returning("I cannot analyze this image.")
    fallback = _llm_returning(_FULL_JSON)

    analysis, primary_failed = await analyze_slide(_slide(), primary, fallback)

    assert primary_failed is False  # NOT a connectivity failure
    assert analysis.model_used == "fallback"


# ── SlideAnalysis.has_figure_content gate ──


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"equations": [{"latex": "V=IR", "meaning": ""}]}, True),
        ({"schematic": {"topology": "CS amp", "roles": ""}}, True),
        ({"plot": "linear ramp"}, True),
        ({"summary": "just a title slide, nothing else"}, False),
        ({"error": "primary failed: x; fallback failed: y"}, False),
        ({}, False),
    ],
)
def test_has_figure_content_gate(kwargs, expected):
    sa = SlideAnalysis(page_num=1, page_idx=0, **kwargs)
    assert sa.has_figure_content is expected


# ── analyze_all_slides: circuit breaker ──


@pytest.mark.asyncio
async def test_breaker_trips_after_threshold_consecutive_primary_failures():
    n = _SLIDE_BREAKER_THRESHOLD + 5
    slides = [_slide(i + 1) for i in range(n)]
    primary = _llm_raising(TimeoutError("down"))

    with pytest.raises(SlideBreakerOpen) as exc_info:
        await analyze_all_slides(slides, primary, fallback_llm=None)

    err = exc_info.value
    assert err.consec_fail == _SLIDE_BREAKER_THRESHOLD
    assert err.analyzed == _SLIDE_BREAKER_THRESHOLD
    assert err.total == n
    # Sequential default (concurrency=1) -> EXACTLY threshold calls, no overshoot.
    assert primary.ainvoke.await_count == _SLIDE_BREAKER_THRESHOLD


@pytest.mark.asyncio
async def test_breaker_stays_closed_when_primary_healthy():
    n = _SLIDE_BREAKER_THRESHOLD + 3
    slides = [_slide(i + 1) for i in range(n)]
    primary = _llm_returning(_FULL_JSON)

    results = await analyze_all_slides(slides, primary, fallback_llm=None)

    assert len(results) == n
    assert all(r.model_used == "primary" for r in results)
    assert primary.ainvoke.await_count == n


@pytest.mark.asyncio
async def test_ceiling_hit_trips_the_breaker_instead_of_draining_to_the_paid_fallback():
    """A primary that never finishes must NOT be treated as a per-slide bad answer.

    Before 2026-08-31 a ceiling-hit response raised _ModelOutputInvalid ("it responded, just
    badly"), which leaves primary_failed False: the breaker counter never moved and every slide
    drained to the PAID frontier fallback one full timeout at a time — the exact outcome this
    module's breaker exists to prevent. It answers every slide, so liveness is not the test.
    """
    n = _SLIDE_BREAKER_THRESHOLD + 5
    slides = [_slide(i + 1) for i in range(n)]
    primary = AsyncMock()
    primary.ainvoke.return_value = MagicMock(
        content="Hmm, let me reconsider which block drives the output",
        response_metadata={"finish_reason": "length"},
    )
    fallback = _llm_returning(_FULL_JSON)

    with pytest.raises(SlideBreakerOpen) as exc_info:
        await analyze_all_slides(slides, primary, fallback_llm=fallback)

    assert exc_info.value.consec_fail == _SLIDE_BREAKER_THRESHOLD
    assert primary.ainvoke.await_count == _SLIDE_BREAKER_THRESHOLD
    # The paid leg is still allowed to rescue the slides actually attempted — but only those.
    assert fallback.ainvoke.await_count == _SLIDE_BREAKER_THRESHOLD


@pytest.mark.asyncio
async def test_completed_but_unparseable_output_still_does_not_trip_the_breaker():
    """The complement: a FINISHED response with no JSON is evidence about the slide, not the
    model, and must keep its existing non-breaker treatment."""
    n = _SLIDE_BREAKER_THRESHOLD + 3
    slides = [_slide(i + 1) for i in range(n)]
    primary = _llm_returning("sorry, I cannot read this slide")
    fallback = _llm_returning(_FULL_JSON)

    results = await analyze_all_slides(slides, primary, fallback_llm=fallback)

    assert len(results) == n  # no breaker
    assert all(r.model_used == "fallback" for r in results)


@pytest.mark.asyncio
async def test_breaker_resets_on_interleaved_primary_success():
    """(threshold-1) fails, a success, then (threshold-1) fails — never threshold IN A ROW, so
    the breaker must never trip even though total failures exceed the threshold."""
    k = _SLIDE_BREAKER_THRESHOLD
    n = 2 * (k - 1) + 1
    slides = [_slide(i + 1) for i in range(n)]
    primary = AsyncMock()
    ok = MagicMock(content=_TITLE_ONLY_JSON)
    primary.ainvoke.side_effect = [TimeoutError()] * (k - 1) + [ok] + [TimeoutError()] * (k - 1)

    results = await analyze_all_slides(slides, primary, fallback_llm=None)

    assert len(results) == n  # never tripped -> every slide attempted
    assert primary.ainvoke.await_count == n


@pytest.mark.asyncio
async def test_breaker_not_tripped_by_unparseable_json_only():
    """A primary that always responds with garbage (never times out/errors) must NEVER trip the
    breaker — it proves liveness every time, it just never produces usable content."""
    n = _SLIDE_BREAKER_THRESHOLD + 5
    slides = [_slide(i + 1) for i in range(n)]
    primary = _llm_returning("not json at all, sorry")

    results = await analyze_all_slides(slides, primary, fallback_llm=None)

    assert len(results) == n
    assert primary.ainvoke.await_count == n
    assert all(r.has_figure_content is False for r in results)


@pytest.mark.asyncio
async def test_breaker_trip_leaves_partial_results_visible_via_callback():
    """Even though analyze_all_slides raises on trip, work already done before the trip must
    still have reached on_slide_done — this is the hook pipeline.py's resumability checkpoint
    relies on (nothing already-analyzed is lost just because the file ultimately aborts)."""
    n = _SLIDE_BREAKER_THRESHOLD + 4
    slides = [_slide(i + 1) for i in range(n)]
    primary = _llm_raising(TimeoutError("down"))
    seen: list[int] = []

    def _on_done(slide, analysis):
        seen.append(slide.page_num)

    with pytest.raises(SlideBreakerOpen):
        await analyze_all_slides(slides, primary, fallback_llm=None, on_slide_done=_on_done)

    # Every slide up to and including the trip point was reported via the callback.
    assert seen == list(range(1, _SLIDE_BREAKER_THRESHOLD + 1))


@pytest.mark.asyncio
async def test_analyze_all_slides_empty_input():
    assert await analyze_all_slides([], AsyncMock()) == []


@pytest.mark.asyncio
async def test_analyze_all_slides_preserves_order_under_concurrency():
    import asyncio as _asyncio

    n = 6
    slides = [_slide(i + 1) for i in range(n)]
    primary = AsyncMock()

    async def _delayed(*args, **kwargs):
        # Later-indexed calls return FASTER, so naive gather ordering would scramble results if
        # analyze_all_slides didn't preserve input order itself.
        idx = primary.ainvoke.await_count
        await _asyncio.sleep(0.01 * (n - idx))
        return MagicMock(content=_TITLE_ONLY_JSON)

    primary.ainvoke.side_effect = _delayed

    results = await analyze_all_slides(slides, primary, fallback_llm=None, concurrency=4)

    assert [r.page_num for r in results] == [1, 2, 3, 4, 5, 6]


# ── slide_analysis_to_blocks / ContentBlock rendering ──


def test_slide_analysis_to_blocks_full_content_and_heading_with_time_range():
    sa = SlideAnalysis(
        page_num=5, page_idx=4, time_range="1:00-2:00",
        equations=[{"latex": "I_D = k(V_{GS}-V_{TH})^2", "meaning": "MOSFET saturation current"}],
        schematic={"topology": "common-source amplifier", "roles": "M1: input transistor"},
        plot="Gain rolls off above the corner frequency",
        summary="CS amplifier overview",
    )
    heading, body = slide_analysis_to_blocks(sa)

    assert heading.text == "p.5 slide (figure) (1:00-2:00)"
    assert heading.text_level == 1
    assert heading.page_idx == 4
    assert body.text_level == 0
    assert body.page_idx == 4
    assert "$I_D = k(V_{GS}-V_{TH})^2$: MOSFET saturation current" in body.text
    assert "Schematic — topology: common-source amplifier" in body.text
    assert "Device roles: M1: input transistor" in body.text
    assert "Plot: Gain rolls off above the corner frequency" in body.text
    assert "Summary: CS amplifier overview" in body.text
    # Order: equations, schematic, plot, summary.
    assert (
        body.text.index("Equations:") < body.text.index("Schematic")
        < body.text.index("Plot:") < body.text.index("Summary:")
    )


def test_slide_analysis_to_blocks_heading_without_time_range():
    sa = SlideAnalysis(page_num=1, page_idx=0, time_range="", summary="x", plot="y")
    heading, _ = slide_analysis_to_blocks(sa)
    assert heading.text == "p.1 slide (figure)"


def test_slide_analysis_to_blocks_equation_without_meaning_omits_colon():
    sa = SlideAnalysis(page_num=1, page_idx=0, equations=[{"latex": "V=IR", "meaning": ""}])
    _, body = slide_analysis_to_blocks(sa)
    assert body.text == "Equations:\n- $V=IR$"


def test_slide_analysis_to_blocks_schematic_without_roles():
    sa = SlideAnalysis(page_num=1, page_idx=0, schematic={"topology": "current mirror", "roles": ""})
    _, body = slide_analysis_to_blocks(sa)
    assert body.text == "Schematic — topology: current mirror"
    assert "Device roles" not in body.text
