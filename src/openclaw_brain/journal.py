"""Lightweight action journal — append-only JSONL log for graph changes.

Records significant write operations so that ingestion history, concept merges,
and node retractions can be audited without querying Neo4j.

Journal file:   {state_dir}/journal.jsonl
Rotation:       auto-rotate at 10 MB; journal.jsonl -> .jsonl.1 -> .jsonl.2 (2 generations kept —
                a bounded, best-effort archive, not unlimited history; see
                _rotate_if_needed()'s docstring).

Log entries (one JSON object per line):
  apply_delta   — chunk committed to graph during ingestion
  merge_concepts — duplicate concept merged into primary
  retract_node  — node soft- or hard-deleted
  record_*      — design reasoning writes (hypothesis, decision, bench)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_BYTES = 10 * 1024 * 1024  # rotate at 10 MB


class ActionJournal:
    """Append-only JSONL journal.  All writes are best-effort — errors are logged."""

    def __init__(self, state_dir: Path):
        self._file = Path(state_dir) / "journal.jsonl"

    def log(self, op: str, **kwargs: Any) -> None:
        """Append one journal entry.  Never raises."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "op": op,
            **kwargs,
        }
        try:
            self._rotate_if_needed()
            self._file.parent.mkdir(parents=True, exist_ok=True)
            with self._file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("ActionJournal write failed: %s", exc)

    def tail(self, n: int = 50) -> list[dict[str, Any]]:
        """Return the last *n* journal entries (most recent last)."""
        if not self._file.is_file():
            return []
        try:
            lines = self._file.read_text(encoding="utf-8").splitlines()
            entries = []
            for line in lines[-n:]:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            return entries
        except Exception:
            return []

    def _rotate_if_needed(self) -> None:
        """Best-effort size-based rotation: journal.jsonl -> .jsonl.1 -> .jsonl.2 (2 generations).

        Shifts an existing .jsonl.1 down to .jsonl.2 (unconditionally overwriting any prior
        .jsonl.2 — os.replace() is fine for that overwrite-intent, we only ever keep 2 rotated
        generations by design, not an unlimited archive) BEFORE moving the active file into the
        now-vacated .jsonl.1 slot. This is the fix for the defect where a bare
        ``journal.jsonl.rename(journal.jsonl.1)`` unconditionally clobbered an existing
        journal.jsonl.1 on POSIX: a second rotation used to permanently destroy everything the
        first rotation had preserved, with no trace. Now a second rotation preserves it one level
        deeper (.jsonl.2); only a THIRD rotation ever drops history beyond that, which is the
        documented, accepted bound.

        Concurrent-writer note (best-effort, not fully guarded): each step is a single atomic
        os.replace(), and the active journal.jsonl path is only ever briefly missing between the
        second replace() call and this method returning — any other process's log() call in that
        window just creates a fresh journal.jsonl via its own open(mode="a"), so no in-flight
        entry is ever lost. What is NOT guarded: two processes rotating at the same instant can
        interleave their generation shifts in an unspecified order (which entries land in .1 vs
        .2 after such a race is not deterministic) — acceptable for an audit trail whose own
        module docstring already documents itself as best-effort, not a durable archive.
        """
        try:
            if self._file.is_file() and self._file.stat().st_size >= _MAX_BYTES:
                gen1 = self._file.with_suffix(".jsonl.1")
                gen2 = self._file.with_suffix(".jsonl.2")
                if gen1.is_file():
                    os.replace(gen1, gen2)
                os.replace(self._file, gen1)
        except OSError:
            pass
