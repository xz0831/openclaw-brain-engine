"""Tests for figure analysis module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from openclaw_brain.knowledge.extraction.figure_analyzer import (
    classify_by_caption,
    analyze_figure,
    analyze_all_figures,
    FigureAnalysis,
    FIGURE_TYPES,
    _FIGURE_BREAKER_THRESHOLD,
    _MAX_DESCRIPTION_CHARS,
    _get_mime_type,
)
from openclaw_brain.knowledge.extraction.mineru_parser import ContentBlock


# ── Caption classification tests ──


def test_classify_circuit_captions():
    assert classify_by_caption("Fig. 3: Proposed cascode OTA schematic") == "circuit"
    assert classify_by_caption("Transistor-level circuit diagram") == "circuit"
    assert classify_by_caption("Differential pair schematic") == "circuit"


def test_classify_plot_captions():
    assert classify_by_caption("Frequency response of the amplifier") == "plot"
    assert classify_by_caption("Fig. 5: Gain vs. frequency Bode plot") == "plot"
    assert classify_by_caption("Measured I-V characteristics") == "plot"
    assert classify_by_caption("Monte Carlo simulation results") == "plot"


def test_classify_block_diagram_captions():
    assert classify_by_caption("System-level block diagram") == "block_diagram"
    assert classify_by_caption("Architecture overview of the ADC") == "block_diagram"


def test_classify_layout_captions():
    assert classify_by_caption("Die photo of the fabricated chip") == "layout"
    assert classify_by_caption("Layout of the OTA") == "layout"
    assert classify_by_caption("Cross-section of the FD-SOI process") == "layout"


def test_classify_photo_captions():
    assert classify_by_caption("Measurement setup photograph") == "photo"
    assert classify_by_caption("Test bench equipment setup") == "photo"


def test_classify_unknown_caption():
    assert classify_by_caption("") == "unknown"
    assert classify_by_caption("Figure 7") == "unknown"
    assert classify_by_caption("Some generic title") == "unknown"


def test_get_mime_type_sniffs_magic_bytes_over_extension(tmp_path):
    jpeg_with_png_extension = tmp_path / "mineru_output.png"
    jpeg_with_png_extension.write_bytes(b"\xff\xd8\xff\xe0jpeg data")

    png_with_jpg_extension = tmp_path / "plot.jpg"
    png_with_jpg_extension.write_bytes(b"\x89PNG\r\n\x1a\npng data")

    assert _get_mime_type(str(jpeg_with_png_extension)) == "image/jpeg"
    assert _get_mime_type(str(png_with_jpg_extension)) == "image/png"


def test_get_mime_type_recognizes_png_and_jpeg_magic(tmp_path):
    png_path = tmp_path / "figure.png"
    png_path.write_bytes(b"\x89PNG\r\n\x1a\npng data")

    jpg_path = tmp_path / "figure.jpg"
    jpg_path.write_bytes(b"\xff\xd8\xff\xe1jpeg data")

    assert _get_mime_type(str(png_path)) == "image/png"
    assert _get_mime_type(str(jpg_path)) == "image/jpeg"


def test_get_mime_type_falls_back_to_extension_without_raising(tmp_path):
    unknown_jpeg = tmp_path / "unknown.jpeg"
    unknown_jpeg.write_bytes(b"not a known image header")

    unknown_png = tmp_path / "unknown.bin"
    unknown_png.write_bytes(b"not a known image header")

    assert _get_mime_type(str(tmp_path / "missing.jpg")) == "image/jpeg"
    assert _get_mime_type(str(unknown_jpeg)) == "image/jpeg"
    assert _get_mime_type(str(unknown_png)) == "image/png"


# ── Figure analysis tests ──


@pytest.fixture
def sample_figure() -> ContentBlock:
    # Create a tiny PNG file for testing
    tmp = Path(tempfile.mktemp(suffix=".png"))
    # Minimal valid PNG (1x1 pixel, red)
    tmp.write_bytes(
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
        b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
        b'\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00'
        b'\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    yield ContentBlock(
        type="image",
        caption="Fig. 3: Proposed folded cascode OTA",
        img_path=str(tmp),
        page_idx=2,
    )
    tmp.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_analyze_figure(sample_figure):
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(
        content="This is a folded cascode OTA with PMOS input pair and NMOS cascode loads."
    )

    result = await analyze_figure(sample_figure, mock_llm, "circuit")

    assert result.figure_type == "circuit"
    assert "folded cascode" in result.description
    assert result.page_idx == 2
    assert result.caption == "Fig. 3: Proposed folded cascode OTA"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attr,payload",
    [
        ("response_metadata", {"finish_reason": "length"}),        # openai / openrouter / oMLX
        ("response_metadata", {"stop_reason": "max_tokens"}),      # anthropic
        ("response_metadata", {"finishReason": "MAX_TOKENS"}),     # google, raw camelCase
        ("additional_kwargs", {"finish_reason": "length"}),        # older aggregated shape
        ("generation_info", {"finish_reason": "maxOutputTokens"}), # generation-level
    ],
)
async def test_analyze_figure_rejects_response_that_hit_the_ceiling(sample_figure, attr, payload):
    """A response that ran out of output budget is incomplete and must not be injected.

    Both shapes that reach the ceiling (THR spec ④(d)) are covered by one guard on purpose —
    neither is usable, so accepting one response never needs the starvation/non-convergence
    diagnosis. The provider matrix here is why detection delegates to the resilience layer
    rather than reading `response_metadata["finish_reason"]` directly: three of these five
    shapes are invisible to that naive lookup.
    """
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(
        content="Hmm, this looks like a folded cascode. Actually wait, let me reconsider",
        **{attr: payload},
    )

    result = await analyze_figure(sample_figure, mock_llm, "circuit")

    assert result.description.startswith("[Analysis failed")
    assert "folded cascode" not in result.description


@pytest.mark.asyncio
async def test_analyze_figure_rejects_runaway_without_finish_reason(sample_figure):
    """Backstop for providers that report no finish_reason at all."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(
        content="x" * (_MAX_DESCRIPTION_CHARS + 1),
        response_metadata={},
    )

    result = await analyze_figure(sample_figure, mock_llm, "circuit")

    assert result.description.startswith("[Analysis failed")
    assert "runaway" in result.description


@pytest.mark.asyncio
async def test_analyze_figure_accepts_completed_response(sample_figure):
    """The guard must be inert for a model that converges — today's production path."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(
        content="Folded cascode OTA: PMOS input pair, NMOS cascode load.",
        response_metadata={"finish_reason": "stop"},
    )

    result = await analyze_figure(sample_figure, mock_llm, "circuit")

    assert result.description.startswith("Folded cascode OTA")


@pytest.mark.asyncio
async def test_analyze_figure_missing_image():
    figure = ContentBlock(
        type="image",
        caption="Missing figure",
        img_path="/nonexistent/path.png",
        page_idx=0,
    )
    mock_llm = AsyncMock()

    result = await analyze_figure(figure, mock_llm, "other")

    assert "could not be loaded" in result.description
    mock_llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_all_figures(sample_figure):
    figures = [
        sample_figure,
        ContentBlock(
            type="image",
            caption="Fig. 5: Gain vs frequency response",
            img_path=sample_figure.img_path,
            page_idx=4,
        ),
    ]

    mock_analysis_llm = AsyncMock()
    mock_analysis_llm.ainvoke.return_value = MagicMock(content="Analysis result text.")

    results = await analyze_all_figures(figures, mock_analysis_llm, classify_llm=None)

    assert len(results) == 2
    # First should be classified as circuit (from caption)
    assert results[0].figure_type == "circuit"
    # Second should be classified as plot (from caption)
    assert results[1].figure_type == "plot"


@pytest.mark.asyncio
async def test_analyze_all_figures_concurrency(sample_figure, monkeypatch):
    """Bounded fan-out: overlaps, never exceeds the cap, preserves figure order; N=1 stays serial."""
    import asyncio
    import openclaw_brain.knowledge.extraction.figure_analyzer as fa

    figures = [
        ContentBlock(type="image", caption=f"Fig {i}: circuit schematic",
                     img_path=sample_figure.img_path, page_idx=i)
        for i in range(6)
    ]

    def make_fake():
        st = {"inflight": 0, "max": 0}

        async def fake_analyze(figure, llm, fig_type):
            st["inflight"] += 1
            st["max"] = max(st["max"], st["inflight"])
            try:
                await asyncio.sleep(0.02)
                return fa.FigureAnalysis(figure_type=fig_type,
                                         description=f"desc-{figure.page_idx}",
                                         page_idx=figure.page_idx)
            finally:
                st["inflight"] -= 1
        return st, fake_analyze

    # concurrency=4 → genuinely overlaps, bounded, order preserved
    st, fake = make_fake()
    monkeypatch.setattr(fa, "analyze_figure", fake)
    results = await fa.analyze_all_figures(figures, MagicMock(), classify_llm=None, concurrency=4)
    assert [r.description for r in results] == [f"desc-{i}" for i in range(6)]  # order preserved
    assert 1 < st["max"] <= 4                                                    # overlapped, capped

    # concurrency=1 → strictly sequential
    st1, fake1 = make_fake()
    monkeypatch.setattr(fa, "analyze_figure", fake1)
    results1 = await fa.analyze_all_figures(figures, MagicMock(), classify_llm=None, concurrency=1)
    assert [r.description for r in results1] == [f"desc-{i}" for i in range(6)]
    assert st1["max"] == 1


@pytest.mark.asyncio
async def test_analyze_all_figures_deep_type_filter(sample_figure, monkeypatch):
    """Only circuit/block figures pay the VLM deep-analysis; others are kept as light stubs."""
    import openclaw_brain.knowledge.extraction.figure_analyzer as fa

    figures = [
        ContentBlock(type="image", caption="Fig 1: cascode OTA schematic",
                     img_path=sample_figure.img_path, page_idx=1),       # → circuit
        ContentBlock(type="image", caption="Fig 2: Gain vs frequency Bode plot",
                     img_path=sample_figure.img_path, page_idx=2),       # → plot
        ContentBlock(type="image", caption="Fig 3: System-level block diagram",
                     img_path=sample_figure.img_path, page_idx=3),       # → block_diagram
    ]
    deep_calls = []

    async def fake_analyze(figure, llm, fig_type):
        deep_calls.append(fig_type)
        return fa.FigureAnalysis(figure_type=fig_type, description="DEEP", page_idx=figure.page_idx)

    monkeypatch.setattr(fa, "analyze_figure", fake_analyze)
    results = await fa.analyze_all_figures(figures, MagicMock(), classify_llm=None, concurrency=2)

    # deep-analysis fired ONLY for circuit + block_diagram
    assert sorted(deep_calls) == ["block_diagram", "circuit"]
    # all figures kept (not excluded), order + types preserved
    assert [r.figure_type for r in results] == ["circuit", "plot", "block_diagram"]
    # circuit/block deep-analyzed; plot kept as a light stub (no VLM call, empty description)
    assert results[0].description == "DEEP" and results[2].description == "DEEP"
    assert results[1].description == ""



def test_all_figure_types_have_prompts():
    from openclaw_brain.knowledge.extraction.figure_analyzer import _ANALYSIS_PROMPTS
    for ft in FIGURE_TYPES:
        assert ft in _ANALYSIS_PROMPTS, f"Missing analysis prompt for figure type: {ft}"


# ── Figure-analysis circuit breaker ──

_MIN_PNG = (
    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
    b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
    b'\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00'
    b'\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
)


def _circuit_figures(tmp_path, n: int) -> list[ContentBlock]:
    """n schematic figures (caption → "circuit" → deep-analysis path), each a real PNG."""
    figs = []
    for i in range(n):
        p = tmp_path / f"fig{i}.png"
        p.write_bytes(_MIN_PNG)
        figs.append(ContentBlock(
            type="image", caption=f"Fig. {i}: cascode OTA schematic",
            img_path=str(p), page_idx=i,
        ))
    return figs


@pytest.mark.asyncio
async def test_figure_breaker_skips_remaining_when_model_down(tmp_path):
    """Deep-analysis model fails every call → breaker trips after _FIGURE_BREAKER_THRESHOLD
    consecutive failures and the remaining schematics skip the doomed VLM round-trip, bounding
    a bad-window from O(N) per-figure timeouts to O(threshold)."""
    n = _FIGURE_BREAKER_THRESHOLD + 5
    figures = _circuit_figures(tmp_path, n)
    mock_llm = AsyncMock()
    mock_llm.ainvoke.side_effect = TimeoutError("model down")

    results = await analyze_all_figures(figures, mock_llm, classify_llm=None, concurrency=1)

    assert len(results) == n  # still 1:1 with input — every figure kept
    assert mock_llm.ainvoke.call_count == _FIGURE_BREAKER_THRESHOLD  # rest skipped, no call
    assert all(f.figure_type == "circuit" for f in results)  # classified/kept regardless


@pytest.mark.asyncio
async def test_figure_breaker_stays_closed_when_model_healthy(tmp_path):
    """A healthy model must analyze every schematic — the breaker never trips."""
    n = _FIGURE_BREAKER_THRESHOLD + 3
    figures = _circuit_figures(tmp_path, n)
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(content="cascode OTA description")

    results = await analyze_all_figures(figures, mock_llm, classify_llm=None, concurrency=1)

    assert mock_llm.ainvoke.call_count == n
    assert all("cascode OTA description" in f.description for f in results)


@pytest.mark.asyncio
async def test_figure_breaker_missing_image_is_neutral_not_reset(tmp_path):
    """A missing-image deep figure makes NO model call, so it must not reset the failure streak.
    With the model down, real-fail/missing-image interspersed must still trip the breaker (and then
    skip the remaining real schematics) — the buggy 'reset on any non-failure' would never trip and
    call the model for every real figure."""
    k = _FIGURE_BREAKER_THRESHOLD
    figures = []
    for i in range(3 * k):
        if i % 2 == 0:  # real schematic (model will fail)
            p = tmp_path / f"real{i}.png"
            p.write_bytes(_MIN_PNG)
            path = str(p)
        else:            # missing image — load fails, no model call, must be neutral
            path = str(tmp_path / f"missing{i}.png")
        figures.append(ContentBlock(type="image", caption=f"Fig {i}: cascode OTA schematic",
                                    img_path=path, page_idx=i))
    mock_llm = AsyncMock()
    mock_llm.ainvoke.side_effect = TimeoutError("model down")

    await analyze_all_figures(figures, mock_llm, classify_llm=None, concurrency=1)

    # Only the k real failures needed to reach the threshold called the model; once tripped, the
    # later real schematics are skipped. Buggy reset-on-missing would call every real figure (> k).
    assert mock_llm.ainvoke.call_count == k


@pytest.mark.asyncio
async def test_figure_breaker_resets_streak_on_success(tmp_path):
    """An interleaved success resets the consecutive-failure counter, so a flaky (not down)
    model never trips: (k-1) fails, a success, then (k-1) fails never hits k in a row."""
    k = _FIGURE_BREAKER_THRESHOLD
    n = 2 * (k - 1) + 1
    figures = _circuit_figures(tmp_path, n)
    mock_llm = AsyncMock()
    ok = MagicMock(content="ok description")
    mock_llm.ainvoke.side_effect = (
        [TimeoutError()] * (k - 1) + [ok] + [TimeoutError()] * (k - 1)
    )

    results = await analyze_all_figures(figures, mock_llm, classify_llm=None, concurrency=1)

    assert mock_llm.ainvoke.call_count == n  # never trips → all attempted
    assert len(results) == n


@pytest.mark.asyncio
async def test_figure_breaker_trips_on_consecutive_ceiling_hits(tmp_path):
    """A model that never converges on schematics is, for this task, a down model.

    It returns promptly (no timeout to catch it) but burns a full output budget per figure, so
    the ceiling guard must feed the same breaker as a model-side failure. The consecutive count
    is what makes this safe — an isolated truncation is absorbed by the reset above.
    """
    n = _FIGURE_BREAKER_THRESHOLD + 3
    figures = _circuit_figures(tmp_path, n)
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(
        content="Hmm, wait, let me reconsider the arrow directions once more",
        response_metadata={"finish_reason": "length"},
    )

    results = await analyze_all_figures(figures, mock_llm, classify_llm=None, concurrency=1)

    assert mock_llm.ainvoke.call_count == _FIGURE_BREAKER_THRESHOLD  # rest skipped
    assert len(results) == n  # every figure still kept, caption-only
    assert not any("reconsider" in f.description for f in results)  # monologue never injected
