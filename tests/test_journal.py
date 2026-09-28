"""Tests for journal.py — ActionJournal append-only audit log + size-based rotation.

W-D2 defect 2: `_rotate_if_needed()` renamed journal.jsonl -> journal.jsonl.1 via `Path.rename()`,
which on POSIX unconditionally overwrites an existing journal.jsonl.1 — so a SECOND rotation
permanently destroyed everything the FIRST rotation had preserved, with zero generations kept
beyond .1 and no warning. Fixed: rotation now cascades .1 -> .2 (also an unconditional overwrite —
2 generations is the documented, accepted bound, not unlimited archival) before the active file
takes the vacated .1 slot, so a second rotation no longer destroys the first rotation's history
outright — only a third rotation drops history beyond the 2-generation bound.

Pure filesystem tests (tmp_path) — no Neo4j/LLM involved, ActionJournal has no such dependency.
`_MAX_BYTES` is monkeypatched down to a few bytes so a single small log entry already exceeds it,
triggering rotation on every subsequent call without writing megabytes of fixture data.
"""

from __future__ import annotations

import json

import openclaw_brain.journal as journal_module
from openclaw_brain.journal import ActionJournal


def _read_lines(path):
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ── basic log()/tail() behavior ──


def test_log_appends_jsonl_entry(tmp_path):
    j = ActionJournal(tmp_path)
    j.log("apply_delta", chunk_id="c1", nodes=3)

    lines = _read_lines(tmp_path / "journal.jsonl")
    assert len(lines) == 1
    assert lines[0]["op"] == "apply_delta"
    assert lines[0]["chunk_id"] == "c1"
    assert lines[0]["nodes"] == 3
    assert "ts" in lines[0]


def test_log_never_raises_when_state_dir_is_blocked(tmp_path):
    """log()'s own documented contract: best-effort, never raises — even when the state dir
    can't be created (here: a plain file already occupies the path mkdir() needs)."""
    blocker = tmp_path / "blocked"
    blocker.write_text("occupies the path journal.py wants as a directory", encoding="utf-8")
    j = ActionJournal(blocker)   # self._file = blocked/journal.jsonl -> mkdir(blocked) raises
    j.log("op")   # must not raise
    assert not (blocker / "journal.jsonl").exists()   # the write genuinely failed, silently


def test_tail_returns_last_n_entries_most_recent_last(tmp_path):
    j = ActionJournal(tmp_path)
    for i in range(5):
        j.log("op", i=i)

    tail = j.tail(2)
    assert [e["i"] for e in tail] == [3, 4]


def test_tail_on_missing_file_returns_empty(tmp_path):
    j = ActionJournal(tmp_path)
    assert j.tail() == []


# ── rotation ──


def test_rotate_moves_oversized_file_to_gen1(tmp_path, monkeypatch):
    monkeypatch.setattr(journal_module, "_MAX_BYTES", 10)  # any single entry already exceeds this
    j = ActionJournal(tmp_path)

    j.log("first", note="before rotation")
    assert not (tmp_path / "journal.jsonl.1").is_file()   # nothing to rotate yet on the 1st write

    j.log("second", note="triggers rotation")

    gen1 = _read_lines(tmp_path / "journal.jsonl.1")
    active = _read_lines(tmp_path / "journal.jsonl")
    assert [e["op"] for e in gen1] == ["first"]
    assert [e["op"] for e in active] == ["second"]


def test_second_rotation_preserves_first_as_gen2(tmp_path, monkeypatch):
    """DEFECT REGRESSION: before the fix, Path.rename() unconditionally overwriting an existing
    journal.jsonl.1 meant this second rotation would have silently destroyed the 'first' entry
    with zero trace anywhere. Fixed: it survives one level deeper, as journal.jsonl.2."""
    monkeypatch.setattr(journal_module, "_MAX_BYTES", 10)
    j = ActionJournal(tmp_path)

    j.log("first", note="rotation A source")
    j.log("second", note="rotation B source")   # rotation A fires here: first -> .1
    j.log("third", note="survives in active")   # rotation B fires here: second -> .1, first -> .2

    gen2 = _read_lines(tmp_path / "journal.jsonl.2")
    gen1 = _read_lines(tmp_path / "journal.jsonl.1")
    active = _read_lines(tmp_path / "journal.jsonl")
    assert [e["op"] for e in gen2] == ["first"]   # preserved, NOT destroyed
    assert [e["op"] for e in gen1] == ["second"]
    assert [e["op"] for e in active] == ["third"]


def test_third_rotation_drops_beyond_two_generations_documented_bound(tmp_path, monkeypatch):
    """The accepted, documented best-effort bound: exactly 2 rotated generations are kept, not
    unlimited history — a fourth log line's rotation replaces .2 too. This is NOT a regression:
    the fix's promise is only that a second rotation no longer destroys the first rotation's
    history outright (see test_second_rotation_preserves_first_as_gen2 above); unlimited
    generations were never in scope for this best-effort audit log."""
    monkeypatch.setattr(journal_module, "_MAX_BYTES", 10)
    j = ActionJournal(tmp_path)

    j.log("first", note="a")
    j.log("second", note="b")    # rotates: first -> .1
    j.log("third", note="c")     # rotates: second -> .1, first -> .2
    j.log("fourth", note="d")    # rotates: third -> .1, second -> .2 (first is now gone)

    gen2 = _read_lines(tmp_path / "journal.jsonl.2")
    gen1 = _read_lines(tmp_path / "journal.jsonl.1")
    active = _read_lines(tmp_path / "journal.jsonl")
    assert [e["op"] for e in gen2] == ["second"]
    assert [e["op"] for e in gen1] == ["third"]
    assert [e["op"] for e in active] == ["fourth"]


def test_no_rotation_when_under_threshold(tmp_path):
    j = ActionJournal(tmp_path)   # real _MAX_BYTES (10 MB) — tiny test entries never approach it
    j.log("first")
    j.log("second")

    assert not (tmp_path / "journal.jsonl.1").is_file()
    active = _read_lines(tmp_path / "journal.jsonl")
    assert [e["op"] for e in active] == ["first", "second"]
