"""Tests for structured chunking (MinerU output → chunks)."""

from openclaw_brain.knowledge.extraction.chunker import (
    chunk_structured,
    _group_blocks_into_sections,
    _pages_to_range,
    _parse_page_range_set,
)
from openclaw_brain.knowledge.extraction.mineru_parser import ContentBlock, ParsedDocument
from openclaw_brain.knowledge.extraction.figure_analyzer import FigureAnalysis


def _make_parsed(blocks, figures=None):
    return ParsedDocument(
        source_id="src_test",
        title="Test Paper",
        total_pages=3,
        checksum="abc123",
        blocks=blocks,
        figures=figures or [],
    )


def test_chunk_structured_basic():
    blocks = [
        ContentBlock(type="text", text="Introduction", text_level=1, page_idx=0),
        ContentBlock(type="text", text="This paper presents a novel analog circuit design.", page_idx=0),
        ContentBlock(type="text", text="Background", text_level=1, page_idx=1),
        ContentBlock(type="text", text="Previous work on OTA design used folded cascode topologies.", page_idx=1),
    ]
    parsed = _make_parsed(blocks)
    result = chunk_structured(parsed)

    assert result.source_id == "src_test"
    assert len(result.chunks) == 2
    assert "Introduction" in result.chunks[0].section_title
    assert "Background" in result.chunks[1].section_title
    assert "novel analog" in result.chunks[0].text
    assert "folded cascode" in result.chunks[1].text


def test_chunk_structured_preserves_equations():
    blocks = [
        ContentBlock(type="text", text="Analysis", text_level=1, page_idx=0),
        ContentBlock(type="text", text="The drain current is:", page_idx=0),
        ContentBlock(type="equation", text="$$I_D = \\frac{1}{2} \\mu_n C_{ox} \\frac{W}{L} (V_{GS} - V_{th})^2$$", page_idx=0),
        ContentBlock(type="text", text="where V_th is the threshold voltage.", page_idx=0),
    ]
    parsed = _make_parsed(blocks)
    result = chunk_structured(parsed)

    assert len(result.chunks) == 1
    assert "$$I_D" in result.chunks[0].text
    assert "threshold voltage" in result.chunks[0].text


def test_chunk_structured_preserves_tables():
    blocks = [
        ContentBlock(type="text", text="Results", text_level=1, page_idx=0),
        ContentBlock(type="table", html="<table><tr><td>Power</td><td>1.2 mW</td></tr></table>",
                     caption="Table I: Performance", page_idx=0),
    ]
    parsed = _make_parsed(blocks)
    result = chunk_structured(parsed)

    assert len(result.chunks) == 1
    assert "<table>" in result.chunks[0].text
    assert "**Table I: Performance**" in result.chunks[0].text


def test_chunk_structured_with_figure_analysis():
    blocks = [
        ContentBlock(type="text", text="Proposed Circuit", text_level=1, page_idx=1),
        ContentBlock(type="text", text="Figure 3 shows the proposed OTA.", page_idx=1),
        ContentBlock(type="image", caption="Fig. 3: OTA schematic", img_path="fig3.png", page_idx=1),
    ]
    figures_analysis = [
        FigureAnalysis(
            figure_type="circuit",
            description="Folded cascode OTA with PMOS input pair, NMOS cascode loads, and Miller compensation.",
            page_idx=1,
            caption="Fig. 3: OTA schematic",
        ),
    ]
    parsed = _make_parsed(blocks)
    result = chunk_structured(parsed, figure_analyses=figures_analysis)

    assert len(result.chunks) == 1
    # Figure analysis should be injected
    assert "Folded cascode OTA" in result.chunks[0].text
    assert "Miller compensation" in result.chunks[0].text


def test_chunk_structured_routes_footnote_into_grounding_support():
    """The MinerU image_footnote is a real document anchor — it must land OUTSIDE the
    FIG_VLM sentinels so grounding's independent support set sees it (free recall win),
    while the VLM description stays INSIDE (excluded)."""
    from openclaw_brain.knowledge.extraction.grounding import independent_support_text

    blocks = [
        ContentBlock(type="text", text="Adjustable Mirror", text_level=1, page_idx=1),
        ContentBlock(type="image", caption="Fig. 5: current mirror", footnote="M3, M4 set by VNUL+/VNUL-.",
                     img_path="fig5.png", page_idx=1),
    ]
    figures_analysis = [
        FigureAnalysis(
            figure_type="circuit",
            description="A VLM-described degeneration current mirror.",
            page_idx=1,
            caption="Fig. 5: current mirror",
            footnote="M3, M4 set by VNUL+/VNUL-.",
        ),
    ]
    result = chunk_structured(_make_parsed(blocks), figure_analyses=figures_analysis)
    text = result.chunks[0].text
    support = independent_support_text(text)  # what grounding actually anchors against

    assert "M3, M4 set by VNUL" in support, "footnote must be an independent anchor"
    assert "VLM-described" not in support, "VLM description must stay excluded from support"


def test_chunk_structured_strips_vlm_preamble_on_figure_injection():
    blocks = [
        ContentBlock(type="text", text="System", text_level=1, page_idx=0),
        ContentBlock(type="text", text="Figure 1 shows the readout chain.", page_idx=0),
    ]
    figures_analysis = [
        FigureAnalysis(
            figure_type="block_diagram",
            description=(
                "Based on the provided block diagram, here is a technical analysis of the circuit:"
                "\n\n### 1. Functional Blocks\nThe chain includes a sampler and ADC."
            ),
            page_idx=0,
            caption="Fig. 1: Readout chain",
        ),
    ]
    parsed = _make_parsed(blocks)
    result = chunk_structured(parsed, figure_analyses=figures_analysis)

    assert "Based on the provided block diagram" not in result.chunks[0].text
    assert "### 1. Functional Blocks" in result.chunks[0].text
    assert "The chain includes a sampler and ADC." in result.chunks[0].text


def test_chunk_structured_does_not_duplicate_figure_across_sections_sharing_a_page():
    """A heading landing mid-page (a chapter transition) can leave two adjacent sections
    legitimately sharing one page_idx: the outgoing section's trailing blocks (including a
    figure) and the new section's heading block are both on that page. Without cross-section
    dedup, figure_map[page_idx]'s full VLM analysis gets appended to BOTH chunks' text —
    duplicating the same schematic's evidence under two chunk_ids. It must be injected once."""
    blocks = [
        ContentBlock(type="text", text="Old Chapter", text_level=1, page_idx=0),
        ContentBlock(type="text", text="End of the old chapter's discussion.", page_idx=1),
        ContentBlock(type="image", caption="Fig. 9: schematic", img_path="fig9.png", page_idx=1),
        # Heading lands on the SAME page_idx as the figure above — starts a new section.
        ContentBlock(type="text", text="New Chapter", text_level=1, page_idx=1),
        ContentBlock(type="text", text="Start of the new chapter's discussion.", page_idx=1),
    ]
    figures_analysis = [
        FigureAnalysis(
            figure_type="circuit",
            description="A shared schematic analysis that must not be duplicated across chunks.",
            page_idx=1,
            caption="Fig. 9: schematic",
        ),
    ]
    parsed = _make_parsed(blocks, figures=figures_analysis)
    result = chunk_structured(parsed, figure_analyses=figures_analysis)

    assert len(result.chunks) == 2
    occurrences = sum(
        "shared schematic analysis" in c.text for c in result.chunks
    )
    assert occurrences == 1, "figure analysis must be injected into exactly one section"
    # It lands in the section that actually contains the figure's own block (document order:
    # the outgoing "Old Chapter" section), not silently dropped from both.
    old_chapter = next(c for c in result.chunks if "Old Chapter" in c.section_title)
    new_chapter = next(c for c in result.chunks if "New Chapter" in c.section_title)
    assert "shared schematic analysis" in old_chapter.text
    assert "shared schematic analysis" not in new_chapter.text


def test_chunk_structured_single_section_per_page_unaffected_by_dedup():
    """REGRESSION: the common case (a page belongs to exactly one section) must inject the
    figure exactly as before this fix — the dedup tracking must be a no-op here."""
    blocks = [
        ContentBlock(type="text", text="Proposed Circuit", text_level=1, page_idx=1),
        ContentBlock(type="text", text="Figure 3 shows the proposed OTA.", page_idx=1),
        ContentBlock(type="image", caption="Fig. 3: OTA schematic", img_path="fig3.png", page_idx=1),
    ]
    figures_analysis = [
        FigureAnalysis(
            figure_type="circuit",
            description="Folded cascode OTA with PMOS input pair.",
            page_idx=1,
            caption="Fig. 3: OTA schematic",
        ),
    ]
    parsed = _make_parsed(blocks, figures=figures_analysis)
    result = chunk_structured(parsed, figure_analyses=figures_analysis)

    assert len(result.chunks) == 1
    assert result.chunks[0].text.count("Folded cascode OTA") == 1


def test_group_blocks_into_sections():
    blocks = [
        ContentBlock(type="text", text="Intro", text_level=1, page_idx=0),
        ContentBlock(type="text", text="Body 1", page_idx=0),
        ContentBlock(type="text", text="Methods", text_level=1, page_idx=1),
        ContentBlock(type="text", text="Body 2", page_idx=1),
        ContentBlock(type="text", text="Body 3", page_idx=2),
    ]
    sections = _group_blocks_into_sections(blocks)

    assert len(sections) == 2
    assert sections[0][0] == "Intro"
    assert sections[1][0] == "Methods"


def test_pages_to_range():
    assert _pages_to_range({0}) == "1"
    assert _pages_to_range({0, 1, 2}) == "1-3"
    assert _pages_to_range({2, 4}) == "3-5"
    assert _pages_to_range(set()) == "0"


def test_parse_page_range_set():
    assert _parse_page_range_set("1") == {0}
    assert _parse_page_range_set("1-3") == {0, 1, 2}
    assert _parse_page_range_set("") == set()
    assert _parse_page_range_set("0") == set()
