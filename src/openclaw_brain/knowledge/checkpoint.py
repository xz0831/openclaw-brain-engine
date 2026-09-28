"""Checkpoint/resume for PDF ingestion pipeline.

Saves per-chunk state to disk so a failed ingestion can resume from
the last successful chunk instead of restarting from scratch.

Checkpoint file structure:
    {state_dir}/checkpoints/{source_id}.json
    {
        "source_id": "...",
        "title": "...",
        "total_chunks": 10,
        "completed_chunks": [0, 1, 2, 3],  // indices
        "chunk_results": {
            "0": {"new_nodes": 3, "new_edges": 5, ...},
            "1": {"new_nodes": 1, "new_edges": 2, ...}
        },
        "errors": ["Chunk 5: timeout"],
        "started_at": "2026-03-21T10:00:00Z",
        "updated_at": "2026-03-21T10:05:00Z"
    }
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PipelineCheckpoint:
    """Manages checkpoint state for a single ingestion run."""

    def __init__(self, checkpoint_dir: Path, source_id: str):
        self._dir = checkpoint_dir
        self._source_id = source_id
        self._file = checkpoint_dir / f"{source_id}.json"
        self._state: dict[str, Any] = {}

    @property
    def exists(self) -> bool:
        return self._file.is_file()

    def load(self) -> bool:
        """Load existing checkpoint. Returns True if found."""
        if not self._file.is_file():
            return False
        try:
            self._state = json.loads(self._file.read_text())
            logger.info(
                "Resuming checkpoint: %s (%d/%d chunks done)",
                self._source_id,
                len(self._state.get("completed_chunks", [])),
                self._state.get("total_chunks", 0),
            )
            return True
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Corrupt checkpoint %s, starting fresh: %s", self._file, e)
            return False

    # Keys this class's chunk-tracking lifecycle owns. initialize() may reset THESE and
    # nothing else: sibling writers share the same file via save_extra() (e.g. the figures
    # pass caches "slide_analyses" in {source_id}_figs.json before chunk tracking starts),
    # and a wholesale state replacement here silently threw that cache away on any
    # --reprocess re-initialization — hours of slide-VLM work per file (review, 2026-07-18).
    _OWNED_KEYS = frozenset({
        "source_id", "title", "total_chunks", "completed_chunks",
        "chunk_results", "errors", "started_at", "updated_at", "completed_at",
    })

    def initialize(self, title: str, total_chunks: int) -> None:
        """Create a new checkpoint for a fresh ingestion (foreign keys survive)."""
        preserved = {k: v for k, v in self._state.items() if k not in self._OWNED_KEYS}
        self._state = {
            "source_id": self._source_id,
            "title": title,
            "total_chunks": total_chunks,
            "completed_chunks": [],
            "chunk_results": {},
            "errors": [],
            "started_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        self._state.update(preserved)
        self._save()

    def ensure_meta(self, title: str, total_chunks: int) -> None:
        """Backfill identity keys if a sibling save_extra() writer created the file
        before chunk tracking ever initialize()d it (the resume branch skips
        initialize(), so without this the checkpoint permanently lacks
        source_id/title/total_chunks)."""
        changed = False
        for key, value in (("source_id", self._source_id), ("title", title),
                           ("total_chunks", total_chunks)):
            if not self._state.get(key):
                self._state[key] = value
                changed = True
        if "started_at" not in self._state:
            self._state["started_at"] = _now_iso()
            changed = True
        if changed:
            self._state["updated_at"] = _now_iso()
            self._save()

    def is_chunk_done(self, chunk_index: int) -> bool:
        """Check if a chunk has already been processed."""
        return chunk_index in self._state.get("completed_chunks", [])

    def mark_chunk_done(self, chunk_index: int, counts: dict[str, int]) -> None:
        """Record a successfully processed chunk."""
        completed = self._state.setdefault("completed_chunks", [])
        if chunk_index not in completed:
            completed.append(chunk_index)
        self._state.setdefault("chunk_results", {})[str(chunk_index)] = counts
        self._state["updated_at"] = _now_iso()
        self._save()

    def record_error(self, chunk_index: int, error: str) -> None:
        """Record a chunk error without marking it done."""
        self._state.setdefault("errors", []).append(
            f"Chunk {chunk_index}: {error}"
        )
        self._state["updated_at"] = _now_iso()
        self._save()

    def get_accumulated_counts(self) -> dict[str, int]:
        """Sum up counts from all completed chunks."""
        totals: dict[str, int] = {
            "new_nodes": 0, "updated_nodes": 0, "new_edges": 0,
            "reinforced_edges": 0, "insights": 0,
        }
        for counts in self._state.get("chunk_results", {}).values():
            for key in totals:
                totals[key] += counts.get(key, 0)
        return totals

    def get_errors(self) -> list[str]:
        return list(self._state.get("errors", []))

    # ── Generic extra-data slot ──
    #
    # Added for the figures-only slide-analysis pre-pass (knowledge/extraction/slide_analyzer.py
    # / pipeline.py::_ingest_html_figures_only), which needs to persist progress on a unit of
    # work (one VLM call per slide, ~10-30s each) that happens BEFORE stage 1-6's per-chunk
    # checkpointing even starts. Rather than invent a second checkpoint file, this reuses the
    # SAME `{source_id}_figs.json` file the task already specifies (chunk tracking still uses
    # `completed_chunks`/`chunk_results` untouched) with one extra, generically-named top-level
    # key per caller — deliberately untyped/uninterpreted here (this class stays a plain JSON
    # blob manager; slide_analyzer.py owns what "slide_analyses" as a key means and how to
    # (de)serialize its values). Any JSON-safe value works; a caller that wants dataclasses
    # round-trips them itself (e.g. via dataclasses.asdict()/**kwargs reconstruction).
    def save_extra(self, key: str, value: Any) -> None:
        """Persist an arbitrary JSON-safe value under a caller-owned key, immediately (same
        atomic temp-file + os.replace write as mark_chunk_done/record_error) — safe to call
        after every single unit of expensive work without batching."""
        self._state[key] = value
        self._state["updated_at"] = _now_iso()
        self._save()

    def load_extra(self, key: str, default: Any = None) -> Any:
        """Read back a value saved by save_extra(). Returns `default` if never saved (including
        on a fresh/never-loaded checkpoint) — never raises KeyError."""
        return self._state.get(key, default)

    def complete(self) -> None:
        """Mark the checkpoint as fully done and archive it to completed/.

        The file is moved to {checkpoint_dir}/completed/ rather than deleted,
        preserving a permanent record of every successful ingestion run.
        """
        self._state["completed_at"] = _now_iso()
        self._save()

        completed_dir = self._dir / "completed"
        try:
            completed_dir.mkdir(parents=True, exist_ok=True)
            dest = completed_dir / self._file.name
            shutil.move(str(self._file), str(dest))
            logger.info("Checkpoint archived: %s → %s", self._source_id, dest)
        except OSError as e:
            logger.warning("Failed to archive checkpoint %s: %s", self._source_id, e)

    def _save(self) -> None:
        """Write checkpoint to disk atomically (temp file + fsync + os.replace).

        A crash mid-write must never leave a truncated/corrupt checkpoint file. Unlike
        ``evidence.py``'s ``_atomic_write_bytes`` (already atomic), this used to be a plain
        ``write_text`` — a process killed between open() and the bytes hitting disk could leave
        a half-written JSON file. ``load()`` tolerates a corrupt file by starting fresh (see its
        docstring), which means an already-committed chunk can be silently re-marked as not-done
        and reprocessed on resume. Writing to a temp file in the same directory and
        ``os.replace``-ing it in means a crash always leaves either the prior complete state or
        the new complete state, never a partial one.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self._state, indent=2)
        tmp_path: Path | None = None
        try:
            fd, tmp_name = tempfile.mkstemp(
                dir=self._dir,
                prefix=f".{self._file.name}.",
                suffix=".tmp",
            )
            tmp_path = Path(tmp_name)
            with os.fdopen(fd, "w") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._file)
        except OSError as e:
            logger.error("Failed to save checkpoint: %s", e)
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
