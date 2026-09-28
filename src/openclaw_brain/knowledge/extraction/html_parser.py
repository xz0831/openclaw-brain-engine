"""HTML lecture-notes parser — Rick's own lecture-capture pipeline output → ParsedDocument.

Mirrors ``mineru_parser.parse_pdf``'s EXACT return contract (``ParsedDocument``/``ContentBlock``,
same ``source_id = f"src_{sha256(file_bytes)[:12]}"`` convention — see ``_file_checksum`` below,
duplicated locally rather than imported, matching the SAME precedent already set by
``chunker.py::_file_checksum``) so the ingest pipeline can treat an HTML lecture deck identically
to a MinerU-parsed PDF from ``chunk_structured()`` onward — see ``knowledge/pipeline.py::ingest_html``.

INPUT FORMAT (verified against live samples, 2026-07-13 — see the two variants below)::

    <h1>...</h1>                             lecture title
    <div class="banner">...</div>            methodology boilerplate — SKIPPED (by omission: it
                                              never matches any target-setting rule below)
    <section class="pg" id="pN">             one per lecture page / topic-segment
      <div class="slide">                    OPTIONAL — absent in the session-note variant
        <img src="data:image/...">           SKIPPED entirely; counted only
        <div class="pgmeta">p.N · 5:15–10:30, 25:21–25:49 · 동기 85%</div>
      </div>
      <div class="note">
        EITHER <div class="pdfonly">이 페이지 구간의 발화가 없거나...</div>   -> NO blocks emitted
        OR     <ul class="gist">...</ul><div class="speech">...</div>
               [<details class="terms">...</details>] [<div class="flags">...</div>]
      </div>
    </section>

Two variants observed live, both handled by the SAME code path below:

  - "정리본" (lecture-notes): section carries ``id="pN"``; has a ``.slide`` (base64 image +
    ``.pgmeta`` = ``"p.N · <time-range> · 동기 NN%"``, the time-range segment itself possibly
    containing several comma-separated ``H:MM–H:MM`` spans, or absent entirely on a 0%-sync
    page). ~99.9% of file bytes is base64 slide images; pure text per deck is ~10-15k chars.
  - "세션노트" (session-notes — tablet/zoom captures with no slide-timestamp sync): section has
    NO ``id`` attribute and NO ``.slide``; ``.pgmeta`` is a bare time range directly inside
    ``.note`` (e.g. ``"0:00 – 1:31"`` — no "p." prefix, no 동기 sync%). This is a real,
    load-bearing correction to an earlier brief description of this variant as a flat "h1 + li
    outline": live files (36/36 sampled in one course directory) all use the IDENTICAL
    .pg/.note/.gist/.speech shape as the main variant, just without id/.slide. Page numbering
    falls back to a 1-based counter over section document-order.

Text-block policy within ``.note`` (see ``_LectureHTMLParser``):
  - A ``.pdfonly`` placeholder marks the whole page as having no speech -> the page contributes
    ZERO ContentBlocks (matches the task's "OR" framing: real transcript XOR a pdfonly
    placeholder — never observed mixed in any sampled file).
  - Everything else inside ``.note`` is kept: ``.gist`` (bullet summary), ``.speech``
    (transcript prose), ``.terms`` (ASR term-correction log, e.g. "be -> V_BE" — genuinely
    useful corrected domain vocabulary). ``.flags`` (pipeline-diagnostic tags like
    "asr-loop-trimmed", "translated-from-en" — never lecture content) is explicitly EXCLUDED:
    it is also the one place adjacent inline ``<span>`` siblings abut with zero whitespace in
    the source, which would otherwise concatenate unreadably.
  - ``<sub>``/``<sup>`` are rendered with a bare ``_``/``^`` prefix (no closing marker) so that
    e.g. ``I<sub>E</sub>`` becomes ``I_E`` — standard EE notation, and free semantic signal for
    the downstream extraction LLM.

OUT OF SCOPE for the DEFAULT (``include_images=False``) text path — by design, not oversight (see
knowledge/extraction/README.md Traps for the equivalent PDF-side scoping precedent):
  - Equations/diagrams: they live inside the slide *images* (base64 PNGs), never decoded by the
    default text-only path — only counted (``images_skipped``). ``include_images=True`` (2026-07,
    see ``SlideImage``/``ParsedDocument.slide_images`` below) is the VLM follow-up: it decodes
    each page's slide image LAZILY for ``knowledge/extraction/slide_analyzer.py`` /
    ``pipeline.py``'s figures-only ingest mode, still without touching the text path at all.
  - ``index.html`` course-listing pages: no ``.pg`` sections at all — a different, unhandled
    page shape; ``parse_lecture_html`` on one simply yields a title and zero blocks (degrades,
    does not crash — pipeline.ingest_html then reports "no text extracted", same as an
    empty/scanned PDF would).
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
from html.parser import HTMLParser
from pathlib import Path

from openclaw_brain.knowledge.extraction.mineru_parser import ContentBlock, ParsedDocument, SlideImage

logger = logging.getLogger(__name__)

# Block-level tags: a start OR end tag in this set forces a paragraph break in the accumulated
# note text. Deliberately excludes inline formatting tags (span/strong/code/a/b/i/em) so that
# e.g. ".terms"'s "<span class=asr>be</span> -> <strong>V_BE</strong>" stays on one line.
_BLOCK_TAGS: frozenset[str] = frozenset({
    "div", "p", "li", "ul", "ol", "section", "details", "summary",
    "h1", "h2", "h3", "h4", "h5", "h6", "br", "table", "tr", "td", "th",
})

# Void (self-closing, no matching end tag) elements — never pushed onto the element stack.
_VOID_TAGS: frozenset[str] = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
    "param", "source", "track", "wbr",
})

_PAGE_ID_RE = re.compile(r"^p(\d+)$", re.IGNORECASE)
_PAGE_PREFIX_RE = re.compile(r"^p\.?\s*\d+", re.IGNORECASE)
_SYNC_PREFIX_RE = re.compile(r"^동기")
_WS_RE = re.compile(r"[ \t\r\n\f\v]+")

# A `.slide` <img src="..."> is always a data: URI in live samples (verified 2026-07-13, see
# module docstring) — "image/<subtype>;base64,<payload>". DOTALL: the base64 payload never
# legitimately contains a literal newline, but a malformed/truncated attribute shouldn't make
# the regex silently stop early either.
_DATA_URI_RE = re.compile(r"^data:([\w.+-]+/[\w.+-]+);base64,(.*)$", re.DOTALL)

# Sentinel for an intentional paragraph break, inserted by block-tag boundaries. Not whitespace
# (so _WS_RE's collapse of real source pretty-printing indentation never eats it), and chosen
# from a range that can never appear in real HTML text content.
_BREAK = "\x00"


def _file_checksum(path: Path) -> str:
    """SHA-256 checksum of a file (byte-identical convention to mineru_parser.py's helper —
    reimplemented locally rather than imported, mirroring chunker.py's own precedent of a local
    duplicate rather than a cross-module import of a leading-underscore "private" helper)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _class_set(attrs: list[tuple[str, str | None]]) -> set[str]:
    for name, value in attrs:
        if name == "class" and value:
            return set(value.split())
    return set()


def _attr(attrs: list[tuple[str, str | None]], name: str) -> str | None:
    for n, v in attrs:
        if n == name:
            return v
    return None


def _normalize_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _normalize_blocks(raw: str) -> str:
    """Collapse source pretty-printing whitespace to single spaces, then turn ``_BREAK``
    sentinels into paragraph breaks (dropping any empty paragraphs from adjacent/leading/
    trailing breaks)."""
    collapsed = _WS_RE.sub(" ", raw)
    paragraphs = [p.strip() for p in collapsed.split(_BREAK)]
    paragraphs = [p for p in paragraphs if p]
    return "\n\n".join(paragraphs)


class _Page:
    """Accumulator for one ``<section class="pg">``."""

    __slots__ = ("page_num", "pdfonly", "_pgmeta_buf", "_note_buf", "_last_was_break", "slide_src")

    def __init__(self, page_num: int) -> None:
        self.page_num = page_num
        self.pdfonly = False
        self._pgmeta_buf: list[str] = []
        self._note_buf: list[str] = []
        self._last_was_break = True  # suppress a leading empty paragraph
        # The raw `src="data:...;base64,..."` of this page's `.slide` <img>, captured only when
        # the parser is constructed with include_images=True (see _LectureHTMLParser). Note this
        # is populated REGARDLESS of `.pdfonly` — a page with zero narration can still carry a
        # slide worth analyzing (arguably the MOST valuable case: an un-narrated equation-heavy
        # slide the professor wrote on but didn't speak through — see pipeline.py's figures-only
        # mode, which is exactly the "무발화" 5-file class this exists for).
        self.slide_src: str | None = None

    def write_pgmeta(self, text: str) -> None:
        self._pgmeta_buf.append(text)

    def write_note(self, text: str) -> None:
        self._note_buf.append(text)
        if text:
            self._last_was_break = False

    def break_paragraph(self) -> None:
        if not self._last_was_break:
            self._note_buf.append(_BREAK)
            self._last_was_break = True

    def pgmeta_text(self) -> str:
        return _normalize_ws("".join(self._pgmeta_buf))

    def note_text(self) -> str:
        return _normalize_blocks("".join(self._note_buf))


class _Frame:
    __slots__ = ("tag", "target", "is_page_root")

    def __init__(self, tag: str, target: str, is_page_root: bool = False) -> None:
        self.tag = tag
        self.target = target
        self.is_page_root = is_page_root


class _LectureHTMLParser(HTMLParser):
    """Single-pass, stdlib-only (``html.parser``) walker.

    ``bs4`` is present in this venv (uv.lock pin ``beautifulsoup4==4.15.0``) but only as a
    TRANSITIVE dependency of something else — not declared in pyproject.toml — so taking a hard
    import on it here would be an undeclared dependency risk (P10: the 2026-07-08 transformers-5.13
    incident was exactly a transitive sibling package silently changing this venv's resolved set
    out from under a load-bearing import). ``html.parser`` is stdlib and the task's own suggested
    approach; used here.

    Text routing is a small stack machine: each pushed ``_Frame`` carries a ``target`` — one of
    "discard" (default; covers <style>/<head>/.banner/.topnav/.flags/.pdfonly/the trailing
    generation-metadata footer, none of which ever match a target-setting rule below, so they are
    excluded BY OMISSION rather than an explicit denylist), "title" (inside the first <h1>),
    "pgmeta" (inside a .pgmeta div — lives under .slide in the main variant, directly under .note
    in the session-note variant), or "note" (the kept transcript text). A child inherits its
    parent's target unless a rule below overrides it.
    """

    def __init__(self, include_images: bool = False) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str = ""
        self.images_skipped: int = 0
        self.pages: list[_Page] = []
        # Gates ONLY whether an <img> src gets captured onto its _Page (see _on_open_tag) — every
        # other code path is byte-identical to before, so the default (False) text-only callers
        # pay zero cost for this feature (task requirement: "기본 False로 기존 텍스트 경로 무변경").
        self._include_images = include_images

        self._stack: list[_Frame] = [_Frame("<root>", "discard")]
        self._title_buf: list[str] = []
        self._title_done = False
        self._current_page: _Page | None = None
        self._sub_depth = 0
        self._sup_depth = 0
        self._li_pending = False

    @property
    def _target(self) -> str:
        return self._stack[-1].target

    # -- HTMLParser callbacks --

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._on_open_tag(tag, attrs)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Self-closed tag (e.g. <br/>): open then immediately close, so any state the open
        # side sets (sub/sup depth, li_pending) never leaks past this single element.
        self._on_open_tag(tag, attrs)
        if tag not in _VOID_TAGS:
            self._on_close_tag(tag)

    def handle_endtag(self, tag: str) -> None:
        self._on_close_tag(tag)

    def handle_data(self, data: str) -> None:
        target = self._target
        if target == "title":
            self._title_buf.append(data)
        elif target == "pgmeta" and self._current_page is not None:
            self._current_page.write_pgmeta(data)
        elif target == "note" and self._current_page is not None:
            text = data
            if self._li_pending and text.strip():
                text = "- " + text.lstrip()
                self._li_pending = False
            if self._sub_depth > 0:
                text = "_" + text
            elif self._sup_depth > 0:
                text = "^" + text
            self._current_page.write_note(text)

    # -- internals --

    def _on_open_tag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = _class_set(attrs)
        target = self._target
        is_page_root = False

        if tag == "h1" and not self._title_done and self._current_page is None:
            target = "title"
        elif tag == "section" and "pg" in classes and self._current_page is None:
            page_id = _PAGE_ID_RE.match(_attr(attrs, "id") or "")
            page = _Page(page_num=int(page_id.group(1)) if page_id else len(self.pages) + 1)
            self.pages.append(page)
            self._current_page = page
            is_page_root = True
            target = "discard"
        elif self._current_page is not None:
            if tag == "div" and "pgmeta" in classes:
                target = "pgmeta"
            elif "pdfonly" in classes:
                target = "discard"
                self._current_page.pdfonly = True
            elif "flags" in classes:
                target = "discard"  # pipeline-diagnostic tags, never lecture content
            elif tag == "div" and "note" in classes:
                target = "note"
            elif tag == "img":
                self.images_skipped += 1
                if self._include_images and self._current_page.slide_src is None:
                    src = _attr(attrs, "src")
                    if src:
                        self._current_page.slide_src = src

        if tag not in _VOID_TAGS:
            self._stack.append(_Frame(tag, target, is_page_root=is_page_root))

        if target == "note":
            if tag == "li":
                self._current_page.break_paragraph()
                self._li_pending = True
            elif tag == "sub":
                self._sub_depth += 1
            elif tag == "sup":
                self._sup_depth += 1
            elif tag in _BLOCK_TAGS:
                self._current_page.break_paragraph()

    def _on_close_tag(self, tag: str) -> None:
        if tag == "h1":
            self._title_done = True

        popped: _Frame | None = None
        for i in range(len(self._stack) - 1, 0, -1):
            if self._stack[i].tag == tag:
                popped = self._stack[i]
                del self._stack[i:]
                break
        # An unmatched end tag (malformed input) is a silent no-op — never raises; this parser
        # runs over Rick's own generator output, but must degrade, not crash, on anything odd.

        if popped is not None and popped.target == "note":
            if tag in _BLOCK_TAGS:
                if self._current_page is not None:
                    self._current_page.break_paragraph()
            if tag == "sub":
                self._sub_depth = max(0, self._sub_depth - 1)
            elif tag == "sup":
                self._sup_depth = max(0, self._sup_depth - 1)

        if popped is not None and popped.is_page_root:
            self._current_page = None

    def close(self) -> None:
        super().close()
        self.title = _normalize_ws("".join(self._title_buf))


def _extract_time_range(pgmeta_text: str) -> str:
    """Strip the leading "p.N" anchor and trailing "동기 NN%" sync marker (both optional —
    the session-note variant's .pgmeta is a bare time range with neither), keeping whatever
    remains as the provenance time-range string verbatim (it may itself contain several
    comma-separated ``H:MM–H:MM`` spans)."""
    parts = [p.strip() for p in re.split(r"\s*·\s*", pgmeta_text) if p.strip()]
    if parts and _PAGE_PREFIX_RE.match(parts[0]):
        parts = parts[1:]
    if parts and _SYNC_PREFIX_RE.match(parts[-1]):
        parts = parts[:-1]
    return ", ".join(parts)


def _page_heading(page: _Page) -> str:
    time_range = _extract_time_range(page.pgmeta_text())
    if time_range:
        return f"p.{page.page_num} ({time_range})"
    return f"p.{page.page_num}"


def parse_lecture_html(html_path: str | Path, include_images: bool = False) -> ParsedDocument:
    """Parse one of Rick's lecture-capture HTML files into the SAME ``ParsedDocument`` shape
    ``mineru_parser.parse_pdf`` returns, so it can feed the identical
    chunk->extract->ground->match->reason->reconcile->commit->embed->summarize pipeline
    unchanged (see ``knowledge/pipeline.py::ingest_html``).

    Rules (see module docstring for the full rationale):
      - title <- first ``<h1>`` (falls back to the file stem, matching mineru_parser's own
        no-title fallback, if no ``<h1>`` is found).
      - one heading ContentBlock + one body ContentBlock per ``<section class="pg">`` that has
        real speech (``text_level=1``/``0`` respectively, so the EXISTING, unmodified
        ``chunk_structured()`` groups exactly one chunk-section per lecture page). Heading text
        preserves the time-range provenance: ``"p.N (5:15-10:30, 25:21-25:49)"``, or bare
        ``"p.N"`` when the page carries no time range.
      - a page whose ``.note`` is a ``.pdfonly`` placeholder (no speech in that segment)
        contributes ZERO text blocks.
      - images are never decoded, only counted (``ParsedDocument.figures`` is always empty —
        there is nothing for figure_analyzer.py to do, and pipeline.ingest_html skips that
        stage entirely rather than calling it on an empty list).

    Args:
        html_path: path to the lecture HTML file.
        include_images: when True, additionally populates ``ParsedDocument.slide_images`` with
            one ``SlideImage`` per ``.pg`` page that carries a ``.slide`` image — REGARDLESS of
            that page's ``.pdfonly``/speech status (a silent slide can still hold an equation the
            professor never narrated; see ``pipeline.py``'s figures-only mode, the consumer this
            flag exists for). Each image is decoded LAZILY (``SlideImage.data`` is a callable, not
            bytes — see that class's docstring); this function itself never calls
            ``base64.b64decode`` on the full slide set, only regex-splits each page's already-
            in-memory data: URI into (mime, still-base64 payload). Default False leaves the
            existing text-only path byte-for-byte unchanged (no capture, no regex, no new list).

    Deterministic: same bytes -> same output (same ``source_id`` convention as
    mineru_parser.parse_pdf: ``f"src_{sha256(file_bytes)[:12]}"``).
    """
    html_path = Path(html_path)
    checksum = _file_checksum(html_path)
    source_id = f"src_{checksum[:12]}"
    raw = html_path.read_text(encoding="utf-8", errors="replace")

    parser = _LectureHTMLParser(include_images=include_images)
    parser.feed(raw)
    parser.close()

    title = parser.title or html_path.stem

    blocks: list[ContentBlock] = []
    slide_images: list[SlideImage] = []
    pages_with_speech = 0
    for page in parser.pages:
        if include_images and page.slide_src:
            match = _DATA_URI_RE.match(page.slide_src)
            if match:
                mime_type, b64_payload = match.group(1), match.group(2)
                slide_images.append(SlideImage(
                    page_num=page.page_num,
                    page_idx=page.page_num - 1,
                    time_range=_extract_time_range(page.pgmeta_text()),
                    mime_type=mime_type,
                    # Default-arg capture (not a closure over the loop variable) — each lambda
                    # decodes ITS OWN page's payload, not whichever `b64_payload` the loop last
                    # left behind.
                    data=lambda payload=b64_payload: base64.b64decode(payload),
                ))
            else:
                logger.warning(
                    "parse_lecture_html %s: p.%d <img> src is not a recognizable data: URI — "
                    "slide image skipped",
                    html_path.name, page.page_num,
                )

        if page.pdfonly:
            continue
        note_text = page.note_text()
        if not note_text:
            continue
        pages_with_speech += 1
        page_idx = page.page_num - 1
        blocks.append(ContentBlock(
            type="text", text=_page_heading(page), page_idx=page_idx, text_level=1,
        ))
        blocks.append(ContentBlock(
            type="text", text=note_text, page_idx=page_idx, text_level=0,
        ))

    total_pages = max((p.page_num for p in parser.pages), default=0)

    logger.info(
        "parse_lecture_html %s: pages=%d pages_with_speech=%d images_skipped=%d slide_images=%d",
        html_path.name, total_pages, pages_with_speech, parser.images_skipped, len(slide_images),
    )

    return ParsedDocument(
        source_id=source_id,
        title=title,
        total_pages=total_pages,
        checksum=checksum,
        blocks=blocks,
        figures=[],   # equations/diagrams live inside the slide images — out of scope (see module docstring)
        output_dir="",  # no MinerU-style side-output directory for this parser
        slide_images=slide_images,
    )
