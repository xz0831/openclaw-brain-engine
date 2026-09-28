"""MinerU-based structured PDF parser.

Uses MinerU to parse PDFs into structured content blocks (text, equations,
tables, figures) with layout analysis, equation→LaTeX conversion, and
table→HTML extraction.

PDF parsing uses the optional MinerU backend; no legacy PDF fallback is bundled.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class SlideImage:
    """One lazily-decoded lecture-slide image (populated only by
    ``html_parser.parse_lecture_html(..., include_images=True)`` — see that module).

    Lives here (alongside ``ContentBlock``/``ParsedDocument``, not in ``html_parser.py``) purely
    to avoid a circular import: ``html_parser.py`` already imports ``ParsedDocument`` from this
    module, and ``ParsedDocument.slide_images`` below needs the type — putting it in the OTHER
    direction (mineru_parser importing from html_parser) would create a cycle. The shape itself
    is HTML-specific today (nothing here is MinerU-specific); this module is simply the shared
    "parser output contract" home for both parsers.

    ``data`` is a zero-argument callable rather than a plain ``bytes`` field so that parsing an
    N-slide deck never holds N decoded images in memory at once — the parser keeps only the
    (still-base64-encoded, i.e. already much cheaper) text slice per slide; each image's raw
    bytes materialize lazily, exactly when a consumer (``slide_analyzer.py``) calls ``.data()``
    for that ONE slide, and are free to be garbage-collected again once that call returns.
    """

    page_num: int                  # 1-based — matches the text heading block's "p.N" convention
    page_idx: int                   # 0-based — matches ContentBlock.page_idx convention elsewhere
    time_range: str                   # same provenance string as the text heading block ("" if none)
    mime_type: str                      # e.g. "image/png" — parsed from the data: URI itself
    data: Callable[[], bytes]             # lazy decode; call once per slide when actually analyzing it


@dataclass
class ContentBlock:
    """A single content block from MinerU's structured output."""

    type: str  # "text", "equation", "table", "image", "code"
    text: str = ""  # Text content, LaTeX for equations, verbatim body for code
    html: str = ""  # HTML for tables
    caption: str = ""
    footnote: str = ""  # MinerU image_footnote — a real document anchor (not VLM-generated)
    img_path: str = ""  # Path to extracted image file
    page_idx: int = 0
    bbox: list[float] = field(default_factory=list)
    # For text blocks, heading level (0=body, 1=h1, 2=h2, etc.)
    text_level: int = 0
    # For code blocks, MinerU's guessed language (e.g. "python", "spice"); empty if unknown
    language: str = ""


@dataclass
class ParsedDocument:
    """Complete structured output from MinerU parsing."""

    source_id: str
    title: str
    author: str = ""
    total_pages: int = 0
    checksum: str = ""
    blocks: list[ContentBlock] = field(default_factory=list)
    # Figure images that need VLM analysis
    figures: list[ContentBlock] = field(default_factory=list)
    # Output directory for extracted images
    output_dir: str = ""
    # Lecture-slide images (html_parser.py's include_images=True path only — always empty for a
    # MinerU-parsed PDF and for a default-mode HTML parse). Deliberately a SEPARATE field from
    # `figures`: `figures` specifically means "ContentBlocks still needing figure_analyzer.py's
    # external-file VLM analysis", a different contract (file-path-based, MinerU-figure-shaped)
    # that ingest_html()'s docstring documents as permanently unpopulated for HTML sources —
    # reusing it here would silently falsify that documented invariant.
    slide_images: list[SlideImage] = field(default_factory=list)

    @property
    def text_blocks(self) -> list[ContentBlock]:
        return [b for b in self.blocks if b.type == "text"]

    @property
    def equation_blocks(self) -> list[ContentBlock]:
        return [b for b in self.blocks if b.type == "equation"]

    @property
    def table_blocks(self) -> list[ContentBlock]:
        return [b for b in self.blocks if b.type == "table"]

    @property
    def code_blocks(self) -> list[ContentBlock]:
        return [b for b in self.blocks if b.type == "code"]


def _file_checksum(path: Path) -> str:
    """SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def parse_pdf(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    backend: str = "hybrid-auto-engine",
    *, egress: str | None = None,
) -> ParsedDocument:
    """Parse a PDF using MinerU into structured content blocks.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory for MinerU output (images, etc.).
                    If None, uses a temp directory.
        backend: MinerU backend to use.

    Returns:
        ParsedDocument with structured blocks.
    """
    pdf_path = Path(pdf_path)
    checksum = _file_checksum(pdf_path)
    source_id = f"src_{checksum[:12]}"

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="mineru_"))
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return _parse_with_mineru(pdf_path, output_dir, source_id, checksum, backend, egress=egress)


_MLX_COMPAT_SHIM_APPLIED = False


def _mlx_transformers_compat_shim() -> None:
    """Tolerate mlx-lm's import-time tokenizer registration under transformers 5.x.

    MinerU's hybrid/vlm backends on macOS auto-select the MLX engine, whose import chain
    (``mlx_vlm`` → ``mlx_lm.tokenizer_utils``) registers a tokenizer BY STRING::

        AutoTokenizer.register("NewlineTokenizer", fast_tokenizer_class=NewlineTokenizer)

    transformers 5.x requires a config *class* there (its ``register()`` reads
    ``key.__module__``) → ``AttributeError`` at import time, killing the whole engine before
    any parsing. The venv is pinned to transformers 4.57.x (MinerU requires ``<5``), where the
    string registration works and this shim is a harmless no-op — but 2026-07-08 proved the
    pin is soft: an unrelated ``sentence-transformers`` install silently bumped transformers
    to 5.13 and broke BOTH MinerU backends until caught a day later. mlx-lm ≤ 0.31.3 ships no
    5.x fix, and the registration is benign for MinerU's own VLM (NewlineTokenizer only
    matters to certain community LLMs), so if 5.x ever sneaks back this wrapper swallows
    exactly that failure and nothing else — the vlm path then fails soft instead of at
    import. Remove once mlx-lm registers with a real config class.
    """
    global _MLX_COMPAT_SHIM_APPLIED
    if _MLX_COMPAT_SHIM_APPLIED:
        return
    try:
        from transformers import AutoTokenizer
    except Exception:
        return
    orig = AutoTokenizer.register

    def _tolerant_register(config_class, *args, **kwargs):
        try:
            return orig(config_class, *args, **kwargs)
        except AttributeError:
            logger.warning(
                "AutoTokenizer.register rejected %r (transformers 5.x vs mlx-lm string key) "
                "— skipping benign registration",
                config_class,
            )
            return None

    AutoTokenizer.register = _tolerant_register
    _MLX_COMPAT_SHIM_APPLIED = True


def _parse_with_mineru(
    pdf_path: Path,
    output_dir: Path,
    source_id: str,
    checksum: str,
    backend: str,
    *, egress: str | None = None,
) -> ParsedDocument:
    """Parse using MinerU's do_parse API."""
    import sys

    from openclaw_brain.egress import LOCAL_ONLY, effective_egress
    if (effective_egress() if egress is None else egress) == LOCAL_ONLY:
        _require_local_mineru(backend)

    if backend != "pipeline" and sys.platform == "darwin":
        _mlx_transformers_compat_shim()

    from mineru.cli.common import do_parse, read_fn

    file_name = pdf_path.stem
    pdf_bytes = read_fn(str(pdf_path))

    # Run MinerU — outputs markdown, content_list.json, images, etc.
    do_parse(
        output_dir=str(output_dir),
        pdf_file_names=[file_name],
        pdf_bytes_list=[pdf_bytes],
        p_lang_list=["en"],
        backend=backend,
        parse_method="auto",
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_orig_pdf=False,
        f_dump_model_output=False,
        f_dump_md=True,
        f_dump_middle_json=False,
        f_dump_content_list=True,
    )

    # Read the content_list.json that MinerU produced
    content_list_path = output_dir / file_name / f"{file_name}_content_list.json"
    if not content_list_path.exists():
        # Try alternative path patterns
        for p in output_dir.rglob("*_content_list.json"):
            content_list_path = p
            break

    if not content_list_path.exists():
        raise FileNotFoundError(
            f"MinerU did not produce content_list.json in {output_dir}"
        )

    content_list = json.loads(content_list_path.read_text(encoding="utf-8"))

    # Resolve image paths relative to output dir
    content_dir = content_list_path.parent
    for item in content_list:
        if item.get("img_path"):
            img_abs = content_dir / item["img_path"]
            if img_abs.exists():
                item["img_path"] = str(img_abs)

    return _convert_content_list(content_list, source_id, checksum, output_dir, pdf_path)


def _require_local_mineru(backend: str) -> None:
    """Fail before MinerU import: its auto source probe and remote downloaders do HTTP."""
    from openclaw_brain.config import ConfigError

    if os.environ.get("MINERU_MODEL_SOURCE", "").strip().lower() != "local":
        raise ConfigError("local-only MinerU requires MINERU_MODEL_SOURCE=local")
    if backend not in {"pipeline", "vlm-auto-engine", "hybrid-auto-engine"}:
        raise ConfigError("local-only MinerU requires a local pipeline, VLM, or hybrid backend")
    config_file = Path(os.environ.get("MINERU_TOOLS_CONFIG_JSON", "mineru.json")).expanduser()
    if not config_file.is_absolute():
        config_file = Path.home() / config_file
    try:
        settings = json.loads(config_file.read_text(encoding="utf-8"))
        models_dir = settings["models-dir"]
        modes = ("pipeline",) if backend == "pipeline" else (("vlm",) if backend == "vlm-auto-engine" else ("pipeline", "vlm"))
        for mode in modes:
            root = Path(models_dir[mode]).expanduser()
            if not root.is_dir() or not any(root.iterdir()):
                raise ValueError("unprepared model directory")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ConfigError("local-only MinerU requires prepared models-dir paths in MINERU_TOOLS_CONFIG_JSON") from exc


def _convert_content_list(
    content_list: list[dict[str, Any]],
    source_id: str,
    checksum: str,
    output_dir: Path,
    pdf_path: Path,
) -> ParsedDocument:
    """Convert MinerU's content_list JSON to our ParsedDocument."""
    blocks: list[ContentBlock] = []
    figures: list[ContentBlock] = []

    for item in content_list:
        item_type = item.get("type", "text")
        page_idx = item.get("page_idx", 0)
        bbox = item.get("bbox", [])

        if item_type == "text":
            block = ContentBlock(
                type="text",
                text=item.get("text", ""),
                page_idx=page_idx,
                bbox=bbox,
                text_level=item.get("text_level", 0),
            )
            blocks.append(block)

        elif item_type == "equation":
            block = ContentBlock(
                type="equation",
                text=item.get("text", ""),
                page_idx=page_idx,
                bbox=bbox,
            )
            blocks.append(block)

        elif item_type == "table":
            block = ContentBlock(
                type="table",
                html=item.get("table_body", ""),
                caption=_join_captions(item.get("table_caption", [])),
                img_path=item.get("img_path", ""),
                page_idx=page_idx,
                bbox=bbox,
            )
            blocks.append(block)

        elif item_type == "code":
            # MinerU (vlm/hybrid backends, "added in vlm 2.5") emits code and
            # algorithm blocks under the shared type "code", distinguished by
            # "sub_type" ("code" | "algorithm"). Body is always "code_body";
            # "guess_lang" is only present for sub_type == "code".
            block = ContentBlock(
                type="code",
                text=item.get("code_body", ""),
                caption=_join_captions(item.get("code_caption", [])),
                language=item.get("guess_lang", ""),
                page_idx=page_idx,
                bbox=bbox,
            )
            blocks.append(block)

        elif item_type == "image":
            figure = ContentBlock(
                type="image",
                caption=_join_captions(item.get("image_caption", [])),
                footnote=_join_captions(item.get("image_footnote", [])),
                img_path=item.get("img_path", ""),
                page_idx=page_idx,
                bbox=bbox,
            )
            figures.append(figure)
            # Also add a placeholder in blocks for ordering
            blocks.append(ContentBlock(
                type="image",
                caption=figure.caption,
                img_path=figure.img_path,
                page_idx=page_idx,
                bbox=bbox,
            ))

    # Extract metadata
    title = pdf_path.stem
    total_pages = max((b.page_idx for b in blocks), default=0) + 1

    return ParsedDocument(
        source_id=source_id,
        title=title,
        total_pages=total_pages,
        checksum=checksum,
        blocks=blocks,
        figures=figures,
        output_dir=str(output_dir),
    )


def _join_captions(captions: list[str] | str) -> str:
    """Join caption list into a single string."""
    if isinstance(captions, str):
        return captions
    return " ".join(captions).strip()


def blocks_to_markdown(blocks: list[ContentBlock]) -> str:
    """Convert structured content blocks back to a Markdown string.

    Used for feeding to the LLM extraction stage.
    """
    parts: list[str] = []

    for block in blocks:
        if block.type == "text":
            if block.text_level > 0:
                prefix = "#" * block.text_level + " "
                parts.append(prefix + block.text)
            else:
                parts.append(block.text)

        elif block.type == "equation":
            parts.append(block.text)  # Already LaTeX with $$ delimiters

        elif block.type == "table":
            if block.caption:
                parts.append(f"**{block.caption}**")
            if block.html:
                parts.append(block.html)

        elif block.type == "code":
            if block.caption:
                parts.append(f"**{block.caption}**")
            # Preserve the body verbatim inside a fenced code block; tag the
            # fence with MinerU's guessed language when one was provided
            # (e.g. SPICE netlists, shell command examples).
            parts.append(f"```{block.language}\n{block.text}\n```")

        elif block.type == "image":
            if block.caption:
                parts.append(f"[Figure: {block.caption}]")
            # Figure analysis text will be injected later

    return "\n\n".join(parts)
