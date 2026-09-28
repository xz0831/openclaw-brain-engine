"""Tests for the lecture-notes HTML parser.

All fixtures below are SYNTHETIC — invented BJT/diode-style placeholder sentences that mirror
the structural shape of Rick's real lecture-capture HTML output (verified live against real
sample files during implementation), never verbatim content copied from an actual lecture file.
"""

from __future__ import annotations

import hashlib

from openclaw_brain.knowledge.extraction.html_parser import parse_lecture_html

# A 1x1 transparent PNG, base64-encoded — stands in for a real (much larger) slide image.
_TINY_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _main_variant_html(*, sections: str) -> str:
    """A minimal skeleton of the "정리본" (slide-synced lecture-notes) variant: h1 + banner +
    N <section class="pg" id="pN"> blocks, matching the live-verified structure exactly."""
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>Synthetic Lecture — 강의 정리본</title></head>
<body><div class="wrap">
<div class="topnav"><a href="index.html">back</a></div>
<h1>Synthetic BJT Lecture</h1>
<div class="banner"><strong>시간축 동기 강의 정리본</strong> — methodology boilerplate that must
never appear in any extracted block.</div>
{sections}
</div></body></html>"""


def _pg_section(page_num: int, pgmeta: str, note_html: str) -> str:
    return f"""<section class="pg" id="p{page_num}">
  <div class="slide"><img loading="lazy" src="{_TINY_PNG}" alt="p.{page_num}">
    <div class="pgmeta">{pgmeta}</div></div>
  <div class="note">{note_html}</div>
</section>"""


def _pdfonly_note() -> str:
    return '<div class="pdfonly">이 페이지 구간의 발화가 없거나 화면에 표시되지 않은 페이지입니다 (교안만 제공).</div>'


def _real_note(gist_items: list[str], speech: str, extra: str = "") -> str:
    gist = "".join(f"<li>{item}</li>" for item in gist_items)
    return f'<ul class="gist">{gist}</ul><div class="speech">{speech}</div>{extra}'


# ── Main ("정리본") variant ──


def test_title_from_h1():
    html = _main_variant_html(sections=_pg_section(1, "p.1 · 동기 0%", _pdfonly_note()))
    doc = parse_lecture_html_from_string(html, "lecture.html")
    assert doc.title == "Synthetic BJT Lecture"


def test_banner_text_never_leaks_into_blocks():
    html = _main_variant_html(sections=_pg_section(
        1, "p.1 · 0:00–5:00 · 동기 79%",
        _real_note(["gist point one"], "some synthetic transcript prose about diodes."),
    ))
    doc = parse_lecture_html_from_string(html, "lecture.html")
    all_text = "\n".join(b.text for b in doc.blocks)
    assert "methodology boilerplate" not in all_text
    assert "시간축 동기" not in all_text


def test_pdfonly_page_yields_no_blocks():
    """A page whose .note is a bare .pdfonly placeholder must contribute ZERO ContentBlocks —
    but must still count toward total_pages and images_skipped (the slide/img still exists)."""
    html = _main_variant_html(sections=(
        _pg_section(1, "p.1 · 동기 0%", _pdfonly_note())
        + _pg_section(2, "p.2 · 5:15–10:30 · 동기 85%", _pdfonly_note())
    ))
    doc = parse_lecture_html_from_string(html, "lecture.html")
    assert doc.blocks == []
    assert doc.total_pages == 2
    assert doc.figures == []


def test_real_speech_page_produces_heading_and_body_pair():
    html = _main_variant_html(sections=(
        _pg_section(1, "p.1 · 동기 0%", _pdfonly_note())
        + _pg_section(
            2, "p.2 · 5:15–10:30, 25:21–25:49 · 동기 85%",
            _real_note(
                ["요약 항목 하나", "요약 항목 둘"],
                "이것은 합성된 예시 발화입니다. BJT의 동작을 설명합니다.",
            ),
        )
    ))
    doc = parse_lecture_html_from_string(html, "lecture.html")

    assert len(doc.blocks) == 2
    heading, body = doc.blocks
    assert heading.type == "text" and heading.text_level == 1
    assert body.type == "text" and body.text_level == 0
    assert heading.page_idx == 1 and body.page_idx == 1   # p.2 -> 0-indexed page_idx=1

    # Time-range provenance preserved verbatim (multi-range, comma-separated); sync% dropped.
    assert heading.text == "p.2 (5:15–10:30, 25:21–25:49)"
    assert "동기" not in heading.text

    assert "- 요약 항목 하나" in body.text
    assert "- 요약 항목 둘" in body.text
    assert "이것은 합성된 예시 발화입니다" in body.text


def test_page_with_no_time_range_gets_bare_heading():
    """"p.N · 동기 0%" (no middle time-range segment) -> heading has no parens."""
    html = _main_variant_html(sections=_pg_section(
        1, "p.1 · 동기 0%",
        _real_note(["점"], "발화가 있지만 동기 시간축이 없는 경우의 합성 예시."),
    ))
    doc = parse_lecture_html_from_string(html, "lecture.html")
    heading = doc.blocks[0]
    assert heading.text == "p.1"


def test_images_are_counted_not_decoded():
    html = _main_variant_html(sections=(
        _pg_section(1, "p.1 · 동기 0%", _pdfonly_note())
        + _pg_section(2, "p.2 · 동기 0%", _pdfonly_note())
        + _pg_section(3, "p.3 · 1:00–2:00 · 동기 50%",
                      _real_note(["점"], "합성 발화 예시 텍스트입니다."))
    ))
    doc = parse_lecture_html_from_string(html, "lecture.html")
    for b in doc.blocks:
        assert "base64" not in b.text
        assert "iVBOR" not in b.text
    assert doc.figures == []


def test_flags_excluded_from_body_text():
    note = _real_note(
        ["gist"], "합성 발화 예시입니다.",
        extra='<div class="flags"><span>translated-from-en</span><span>asr-loop-trimmed</span></div>',
    )
    html = _main_variant_html(sections=_pg_section(1, "p.1 · 0:00–1:00 · 동기 90%", note))
    doc = parse_lecture_html_from_string(html, "lecture.html")
    body_text = doc.blocks[1].text
    assert "translated-from-en" not in body_text
    assert "asr-loop-trimmed" not in body_text


def test_terms_correction_included_and_readable():
    note = _real_note(
        ["gist"], "합성 발화 예시입니다.",
        extra=(
            '<details class="terms"><summary>용어 교정 1건</summary>'
            '<ul><li><span class="asr">bee</span> → <strong>V_BE</strong></li></ul></details>'
        ),
    )
    html = _main_variant_html(sections=_pg_section(1, "p.1 · 0:00–1:00 · 동기 90%", note))
    doc = parse_lecture_html_from_string(html, "lecture.html")
    body_text = doc.blocks[1].text
    assert "bee → V_BE" in body_text


def test_subscript_and_superscript_prefix_convention():
    note = _real_note(
        ["gist"],
        "이것은 I<sub>E</sub> 와 V<sup>2</sup> 를 포함하는 합성 발화입니다.",
    )
    html = _main_variant_html(sections=_pg_section(1, "p.1 · 0:00–1:00 · 동기 90%", note))
    doc = parse_lecture_html_from_string(html, "lecture.html")
    body_text = doc.blocks[1].text
    assert "I_E" in body_text
    assert "V^2" in body_text


def test_source_id_and_checksum_match_mineru_convention(tmp_path):
    """source_id = f"src_{sha256(file_bytes)[:12]}" — same content-addressed convention as
    mineru_parser.parse_pdf, computed over the RAW FILE BYTES (not the parsed text)."""
    html = _main_variant_html(sections=_pg_section(
        1, "p.1 · 0:00–1:00 · 동기 90%", _real_note(["g"], "합성 발화."),
    ))
    path = tmp_path / "lecture.html"
    path.write_text(html, encoding="utf-8")

    doc = parse_lecture_html(path)

    expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert doc.checksum == expected_sha
    assert doc.source_id == f"src_{expected_sha[:12]}"


def test_deterministic_same_bytes_same_output(tmp_path):
    html = _main_variant_html(sections=(
        _pg_section(1, "p.1 · 동기 0%", _pdfonly_note())
        + _pg_section(2, "p.2 · 1:00–2:00 · 동기 90%", _real_note(["g"], "합성 발화 텍스트."))
    ))
    path = tmp_path / "lecture.html"
    path.write_text(html, encoding="utf-8")

    doc1 = parse_lecture_html(path)
    doc2 = parse_lecture_html(path)

    assert doc1.source_id == doc2.source_id
    assert [b.text for b in doc1.blocks] == [b.text for b in doc2.blocks]


def test_title_falls_back_to_file_stem_when_no_h1(tmp_path):
    html = """<!doctype html><html><head><title>x</title></head>
<body><div class="wrap"><section class="pg" id="p1">
<div class="note"><div class="pdfonly">이 페이지 구간의 발화가 없거나 화면에 표시되지 않은 페이지입니다.</div></div>
</section></div></body></html>"""
    path = tmp_path / "no_h1_lecture.html"
    path.write_text(html, encoding="utf-8")
    doc = parse_lecture_html(path)
    assert doc.title == "no_h1_lecture"


# ── "세션노트" (session-note) variant — no id, no .slide, bare time-range pgmeta ──


def _session_note_html(sections: str) -> str:
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>Synthetic Session Note — 세션 노트</title></head>
<body><div class="wrap">
<div class="topnav"><a href="index.html">back</a></div>
<h1>Synthetic Session-Note Lecture</h1>
<div class="banner"><strong>세션 노트 (페이지 미동기)</strong> — boilerplate, must never leak.</div>
{sections}
<div style="color:#888;font-size:12px">생성 2026-07-13 · Synthetic - main · 발화 3분</div>
</div></body></html>"""


def _session_note_section(pgmeta_time_range: str, gist_items: list[str], speech: str) -> str:
    # No id attribute, no .slide wrapper — matches the LIVE session-note variant exactly.
    return f"""<section class="pg" style="grid-template-columns:1fr">
  <div class="note"><div class="pgmeta">{pgmeta_time_range}</div>
  {_real_note(gist_items, speech)}</div>
</section>"""


def test_session_note_variant_no_id_no_slide():
    html = _session_note_html(
        _session_note_section("0:00 – 1:31", ["요약1"], "합성 세션노트 발화 예시 1.")
        + _session_note_section("1:31 – 2:57", ["요약2"], "합성 세션노트 발화 예시 2.")
    )
    doc = parse_lecture_html_from_string(html, "session.html")

    assert doc.title == "Synthetic Session-Note Lecture"
    assert doc.figures == []
    assert doc.total_pages == 2
    assert len(doc.blocks) == 4  # 2 pages x (heading + body)

    h1, b1, h2, b2 = doc.blocks
    # No id attribute -> page numbering falls back to a 1-based document-order counter.
    assert h1.page_idx == 0 and h2.page_idx == 1
    # Bare time range preserved verbatim (no "p." prefix, no 동기% in the source to strip).
    assert h1.text == "p.1 (0:00 – 1:31)"
    assert h2.text == "p.2 (1:31 – 2:57)"
    assert "합성 세션노트 발화 예시 1" in b1.text
    assert "합성 세션노트 발화 예시 2" in b2.text

    # No .slide in this variant -> zero images.
    assert doc.blocks and "iVBOR" not in "".join(b.text for b in doc.blocks)


def test_session_note_banner_excluded():
    html = _session_note_html(
        _session_note_section("0:00 – 1:00", ["g"], "합성 발화.")
    )
    doc = parse_lecture_html_from_string(html, "session.html")
    all_text = "\n".join(b.text for b in doc.blocks)
    assert "boilerplate" not in all_text
    assert "페이지 미동기" not in all_text


# ── include_images=True (lazily-decoded slide images for the figures-only VLM pipeline) ──


def test_include_images_false_default_yields_no_slide_images():
    """Default path is byte-for-byte unchanged — the task's explicit requirement."""
    html = _main_variant_html(sections=(
        _pg_section(1, "p.1 · 동기 0%", _pdfonly_note())
        + _pg_section(2, "p.2 · 1:00–2:00 · 동기 90%", _real_note(["g"], "합성 발화."))
    ))
    doc = parse_lecture_html_from_string(html, "lecture.html")
    assert doc.slide_images == []


def test_include_images_true_captures_every_slide_including_pdfonly_pages():
    """A silent (pdfonly) page can still hold a valuable un-narrated equation slide — slide
    images must be captured REGARDLESS of pdfonly/speech status (only the TEXT blocks are gated
    on speech). This is the load-bearing behavior for the "무발화" (no-speech) file class."""
    html = _main_variant_html(sections=(
        _pg_section(1, "p.1 · 동기 0%", _pdfonly_note())
        + _pg_section(2, "p.2 · 5:15–10:30 · 동기 85%", _real_note(["g"], "합성 발화 예시."))
    ))
    doc = parse_lecture_html_from_string(html, "lecture.html", include_images=True)

    assert len(doc.slide_images) == 2
    assert [s.page_num for s in doc.slide_images] == [1, 2]
    assert [s.page_idx for s in doc.slide_images] == [0, 1]
    # Text blocks are still gated on speech — only page 2 (unaffected by include_images).
    assert len(doc.blocks) == 2


def test_include_images_lazy_decode_matches_source_png():
    html = _main_variant_html(sections=_pg_section(
        1, "p.1 · 0:00–1:00 · 동기 90%", _real_note(["g"], "합성 발화."),
    ))
    doc = parse_lecture_html_from_string(html, "lecture.html", include_images=True)

    assert len(doc.slide_images) == 1
    slide = doc.slide_images[0]
    assert slide.mime_type == "image/png"
    assert callable(slide.data)  # not eagerly decoded — a zero-arg callable
    decoded = slide.data()
    assert isinstance(decoded, bytes)
    assert decoded.startswith(b"\x89PNG\r\n\x1a\n")  # real PNG magic bytes, not the base64 text
    # Decoding twice must be idempotent/repeatable (a plain closure, not a one-shot iterator).
    assert slide.data() == decoded


def test_include_images_time_range_matches_heading_provenance():
    """SlideImage.time_range uses the SAME _extract_time_range() string as the text heading
    block — one provenance source, not two independently-parsed copies."""
    html = _main_variant_html(sections=_pg_section(
        2, "p.2 · 5:15–10:30, 25:21–25:49 · 동기 85%",
        _real_note(["g"], "합성 발화."),
    ))
    doc = parse_lecture_html_from_string(html, "lecture.html", include_images=True)

    heading = doc.blocks[0]
    assert heading.text == "p.2 (5:15–10:30, 25:21–25:49)"
    assert doc.slide_images[0].time_range == "5:15–10:30, 25:21–25:49"


def test_include_images_session_note_variant_has_no_slide_images():
    """The 세션노트 variant has NO ``.slide`` wrapper at all (module docstring) — zero slide
    images even with include_images=True, matching its existing zero-figures behavior."""
    html = _session_note_html(
        _session_note_section("0:00 – 1:31", ["요약1"], "합성 세션노트 발화 예시.")
    )
    doc = parse_lecture_html_from_string(html, "session.html", include_images=True)
    assert doc.slide_images == []


def test_include_images_non_data_uri_src_skipped_not_crashed():
    """A malformed/non-data: <img src> (e.g. a relative file path) must degrade — skip that
    page's slide image — never raise (mirrors the parser's overall malformed-input tolerance)."""
    html = _main_variant_html(sections=f"""<section class="pg" id="p1">
  <div class="slide"><img src="./slide1.png" alt="p.1">
    <div class="pgmeta">p.1 · 동기 0%</div></div>
  <div class="note">{_real_note(["g"], "합성 발화.")}</div>
</section>""")
    doc = parse_lecture_html_from_string(html, "lecture.html", include_images=True)
    assert doc.slide_images == []
    assert len(doc.blocks) == 2  # text path is unaffected by the skipped image


# ── helper: parse from an in-memory string without requiring every test to touch disk ──


def parse_lecture_html_from_string(
    html_text: str, filename: str, tmp_path_factory=None, include_images: bool = False,
):
    """Write ``html_text`` to a throwaway temp file and parse it — parse_lecture_html's public
    contract is path-based (mirroring mineru_parser.parse_pdf), so tests go through the real
    file-read + checksum path rather than reaching into parser internals."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / filename
        path.write_text(html_text, encoding="utf-8")
        return parse_lecture_html(path, include_images=include_images)
