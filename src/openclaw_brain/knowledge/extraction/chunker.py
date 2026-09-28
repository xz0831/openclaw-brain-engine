"""Split MinerU parsed documents into structured, grounded chunks."""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from openclaw_brain.knowledge.extraction.grounding import FIG_VLM_CLOSE, FIG_VLM_OPEN
from openclaw_brain.knowledge.extraction.models import ChunkerResult, SourceChunkInfo

if TYPE_CHECKING:
    from openclaw_brain.knowledge.extraction.figure_analyzer import FigureAnalysis
    from openclaw_brain.knowledge.extraction.mineru_parser import ContentBlock, ParsedDocument


# Approximate tokens per character (conservative)
_CHARS_PER_TOKEN = 4

_CONTENT_HEADING_RE = re.compile(r"^\s*(?:#{1,6}\s*|\*\*|(?:#{1,6}\s*)?\d+\.)")
_VLM_PREAMBLE_RE = re.compile(
    r"""
    ^\s*
    (?:(?:of\s+course|sure|certainly)[.!?,]?\s*)?
    (?:
        based\s+on\b.*(?:\bhere\s+(?:is|are)\b|\bhere's\b|\banalysis\b|\bbreakdown\b|:).*
        |
        here\s+(?:is|are)\b.*
        |
        here's\b.*
        |
        as\s+(?:a|an|your)\b.*\bhere\s+(?:is|are)\b.*
        |
        i(?:'ll|\s+will)\b.*
        |
        this\s+is\s+an?\s+analysis\b.*
    )
    \s*[:.]?\s*$
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


def _strip_vlm_preamble(description: str) -> str:
    """Drop a leading conversational lead-in from VLM figure prose so it isn't
    re-parsed as content by extract. Conservative: only strips a short leading
    paragraph that matches a known preamble pattern AND is followed by real content.
    """
    if not description:
        return description

    leading, separator, content = description.partition("\n\n")
    if not separator or not content.strip():
        return description

    if _CONTENT_HEADING_RE.match(leading):
        return description

    if _VLM_PREAMBLE_RE.match(leading):
        return content.lstrip()

    return description


def chunk_structured(
    parsed: ParsedDocument,
    figure_analyses: list[FigureAnalysis] | None = None,
    max_tokens: int = 1500,
    overlap_tokens: int = 100,
) -> ChunkerResult:
    """Chunk a MinerU-parsed document using its structural information.

    Uses heading hierarchy from MinerU's content blocks to split at section
    boundaries. Equations and tables are kept as atomic units.
    Figure analysis results are injected into relevant chunks.

    Args:
        parsed: ParsedDocument from MinerU parser.
        figure_analyses: Optional figure analysis results to inject.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Token overlap between chunks.

    Returns:
        ChunkerResult with structured chunks.
    """
    from openclaw_brain.knowledge.extraction.mineru_parser import blocks_to_markdown

    # Build a page→figure_analysis map
    figure_map: dict[int, list[str]] = {}
    if figure_analyses:
        for fa in figure_analyses:
            # Wrap the VLM-generated description in sentinels so grounding EXCLUDES it from its
            # support set (anti self-grounding); caption AND footnote stay OUTSIDE as independent,
            # document-real anchors (the footnote was previously discarded — the free recall win).
            description = f"{FIG_VLM_OPEN}{_strip_vlm_preamble(fa.description)}{FIG_VLM_CLOSE}"
            header = f"[Figure ({fa.figure_type}): {fa.caption}]" if fa.caption else f"[Figure ({fa.figure_type})]"
            footnote = f"\n[Figure note: {fa.footnote}]" if getattr(fa, "footnote", "") else ""
            desc = f"{header}{footnote}\n{description}"
            figure_map.setdefault(fa.page_idx, []).append(desc)

    # Group blocks into sections by heading hierarchy
    sections = _group_blocks_into_sections(parsed.blocks)

    max_chars = max_tokens * _CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * _CHARS_PER_TOKEN
    chunks: list[SourceChunkInfo] = []

    # A heading can land mid-page, so two adjacent sections can legitimately share one
    # page_idx (the outgoing section's trailing blocks and the new section's heading block are
    # both on that page). Without this tracking set, figure_map[page_idx]'s full VLM analysis
    # would be appended verbatim into BOTH sections' text — duplicating the same schematic's
    # evidence under two chunk_ids (double LLM spend, and if the matcher folds both mentions
    # onto one node, a double-counted reinforcement bump). Inject each figure at most once, into
    # the first (document-order) section whose reconstructed page range covers it — a no-op
    # change for the common case where a page belongs to exactly one section.
    injected_figure_pages: set[int] = set()

    for section_title, page_range, section_blocks in sections:
        # Convert blocks to markdown text
        section_text = blocks_to_markdown(section_blocks)

        # Inject figure analyses for pages covered by this section, skipping any page already
        # injected into an earlier section (see injected_figure_pages above).
        pages_in_section = _parse_page_range_set(page_range)
        for page_idx in sorted(pages_in_section):
            if page_idx in figure_map and page_idx not in injected_figure_pages:
                for fig_desc in figure_map[page_idx]:
                    section_text += "\n\n" + fig_desc
                injected_figure_pages.add(page_idx)

        if not section_text.strip():
            continue

        if len(section_text) <= max_chars:
            chunks.append(SourceChunkInfo(
                chunk_id=f"chunk_{uuid.uuid4().hex[:12]}",
                source_id=parsed.source_id,
                text=section_text.strip(),
                pages=page_range,
                section_title=section_title,
                chunk_index=len(chunks),
                token_estimate=len(section_text) // _CHARS_PER_TOKEN,
            ))
        else:
            sub_chunks = _split_text(section_text, max_chars, overlap_chars)
            for sub_text in sub_chunks:
                chunks.append(SourceChunkInfo(
                    chunk_id=f"chunk_{uuid.uuid4().hex[:12]}",
                    source_id=parsed.source_id,
                    text=sub_text.strip(),
                    pages=page_range,
                    section_title=section_title,
                    chunk_index=len(chunks),
                    token_estimate=len(sub_text) // _CHARS_PER_TOKEN,
                ))

    return ChunkerResult(
        source_id=parsed.source_id,
        title=parsed.title,
        author=parsed.author,
        total_pages=parsed.total_pages,
        chunks=chunks,
        checksum=parsed.checksum,
    )


def _group_blocks_into_sections(
    blocks: list[ContentBlock],
) -> list[tuple[str, str, list[ContentBlock]]]:
    """Group content blocks into sections based on heading hierarchy.

    Returns list of (section_title, page_range, blocks).
    """
    sections: list[tuple[str, str, list]] = []
    current_title = "Untitled"
    current_blocks: list = []
    current_pages: set[int] = set()

    for block in blocks:
        # New section starts at heading blocks (text_level > 0)
        if block.type == "text" and block.text_level > 0 and current_blocks:
            # Flush current section
            page_range = _pages_to_range(current_pages)
            sections.append((current_title, page_range, current_blocks))
            current_title = block.text.strip()
            current_blocks = [block]
            current_pages = {block.page_idx}
        else:
            if block.type == "text" and block.text_level > 0:
                current_title = block.text.strip()
            current_blocks.append(block)
            current_pages.add(block.page_idx)

    # Flush last section
    if current_blocks:
        page_range = _pages_to_range(current_pages)
        sections.append((current_title, page_range, current_blocks))

    return sections


def _pages_to_range(pages: set[int]) -> str:
    """Convert a set of page indices to a page range string."""
    if not pages:
        return "0"
    sorted_pages = sorted(pages)
    # Convert 0-indexed to 1-indexed
    first = sorted_pages[0] + 1
    last = sorted_pages[-1] + 1
    return str(first) if first == last else f"{first}-{last}"


def _parse_page_range_set(page_range: str) -> set[int]:
    """Parse a page range string to a set of 0-indexed page numbers."""
    if not page_range or page_range == "0":
        return set()
    if "-" in page_range:
        parts = page_range.split("-", 1)
        try:
            # Convert 1-indexed to 0-indexed
            return set(range(int(parts[0]) - 1, int(parts[1])))
        except ValueError:
            return set()
    try:
        return {int(page_range) - 1}
    except ValueError:
        return set()


def _split_pages_into_chunks(
    pages: list[tuple[int, str]],
    source_id: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[SourceChunkInfo]:
    """Split page texts into chunks, trying to break at section boundaries."""
    # First, try to split by section headings
    sections = _extract_sections(pages)

    chunks: list[SourceChunkInfo] = []
    max_chars = max_tokens * _CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * _CHARS_PER_TOKEN

    for section_title, page_range, text in sections:
        if len(text) <= max_chars:
            # Section fits in one chunk
            chunks.append(SourceChunkInfo(
                chunk_id=f"chunk_{uuid.uuid4().hex[:12]}",
                source_id=source_id,
                text=text.strip(),
                pages=page_range,
                section_title=section_title,
                chunk_index=len(chunks),
                token_estimate=len(text) // _CHARS_PER_TOKEN,
            ))
        else:
            # Split large section by paragraphs, respecting max_tokens
            sub_chunks = _split_text(text, max_chars, overlap_chars)
            for sub_text in sub_chunks:
                chunks.append(SourceChunkInfo(
                    chunk_id=f"chunk_{uuid.uuid4().hex[:12]}",
                    source_id=source_id,
                    text=sub_text.strip(),
                    pages=page_range,
                    section_title=section_title,
                    chunk_index=len(chunks),
                    token_estimate=len(sub_text) // _CHARS_PER_TOKEN,
                ))

    return chunks


def _extract_sections(
    pages: list[tuple[int, str]],
) -> list[tuple[str, str, str]]:
    """Try to detect section headings and group text.

    Returns list of (section_title, page_range, text).
    Falls back to per-page grouping if no headings detected.
    """
    # Common heading patterns in textbooks and academic papers
    heading_pattern = re.compile(
        r'^(?:'
        # Numbered sections: 1. Title, 1.2 Title, 1.2.3 Title
        r'\d+\.[\d.]*\s+.{3,80}'
        # Roman numeral sections: I. Title, II. TITLE, IV. Title
        r'|[IVX]+\.\s+.{3,80}'
        # ALL-CAPS headings (5-60 chars)
        r'|[A-Z][A-Z\s]{5,60}'
        # Chapter headings
        r'|Chapter\s+\d+'
        # Common academic paper section keywords
        r'|(?:Abstract|Introduction|Background|Related\s+Work|Methodology|Methods?'
        r'|Experiment(?:al)?(?:\s+Results)?|Results?(?:\s+and\s+Discussion)?'
        r'|Discussion|Conclusion|Conclusions|Summary|Acknowledgment|References'
        r'|Appendix(?:\s+[A-Z])?)\s*$'
        r')',
        re.MULTILINE,
    )

    all_text = ""
    page_map: list[tuple[int, int]] = []  # (start_char, page_num)

    for page_num, text in pages:
        page_map.append((len(all_text), page_num))
        all_text += text + "\n"

    # Find headings
    matches = list(heading_pattern.finditer(all_text))

    if len(matches) < 2:
        # No clear sections — group by pages (1 page per chunk for finer granularity)
        return _group_by_pages(pages, group_size=1)

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(all_text)
        title = match.group().strip()
        text = all_text[start:end]

        # Determine page range
        start_page = _char_to_page(start, page_map)
        end_page = _char_to_page(end - 1, page_map)
        page_range = f"{start_page}" if start_page == end_page else f"{start_page}-{end_page}"

        sections.append((title, page_range, text))

    return sections


def _group_by_pages(
    pages: list[tuple[int, str]],
    group_size: int = 3,
) -> list[tuple[str, str, str]]:
    """Fallback: group consecutive pages together."""
    sections = []
    for i in range(0, len(pages), group_size):
        group = pages[i:i + group_size]
        first_page = group[0][0]
        last_page = group[-1][0]
        page_range = f"{first_page}" if first_page == last_page else f"{first_page}-{last_page}"
        text = "\n".join(t for _, t in group)
        sections.append((f"Pages {page_range}", page_range, text))
    return sections


def _split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split text into chunks, trying to break at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append(current)
            # Keep overlap from end of previous chunk
            if overlap_chars > 0 and len(current) > overlap_chars:
                current = current[-overlap_chars:] + "\n\n" + para
            else:
                current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current)

    return chunks


def _char_to_page(char_pos: int, page_map: list[tuple[int, int]]) -> int:
    """Convert a character position to a page number."""
    page = page_map[0][1]
    for start_char, page_num in page_map:
        if start_char <= char_pos:
            page = page_num
        else:
            break
    return page


def _file_checksum(path: Path) -> str:
    """SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()
