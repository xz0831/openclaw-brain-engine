"""Local feedback for gaps in openclaw-brain answer coverage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def append_knowledge_gap(
    state_path: Path, *, question: str, missing_knowledge: list[str],
    kb_coverage: str, cited_ids: list[str], context_sha256: str,
) -> None:
    """Append one local JSONL record, including the user's original question verbatim.

    This only writes under ``state_path``; it never changes the knowledge graph.
    Callers decide whether a write failure should affect the answer.
    """
    record = {
        "schema": "knowledge-gap/0",
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "missing_knowledge": missing_knowledge[:5],
        "kb_coverage": kb_coverage,
        "cited_ids": cited_ids,
        "context_sha256": context_sha256,
    }
    state_path.mkdir(parents=True, exist_ok=True)
    with (state_path / "knowledge_gaps.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
