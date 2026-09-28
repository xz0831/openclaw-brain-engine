"""Tests for pipeline checkpoint/resume."""

import json
import os

import pytest

from openclaw_brain.knowledge.checkpoint import PipelineCheckpoint


@pytest.fixture
def checkpoint_dir(tmp_path):
    return tmp_path / "checkpoints"


@pytest.fixture
def checkpoint(checkpoint_dir):
    return PipelineCheckpoint(checkpoint_dir, "test-source-abc123")


def test_initialize_creates_file(checkpoint, checkpoint_dir):
    checkpoint.initialize("Test PDF", 5)
    assert (checkpoint_dir / "test-source-abc123.json").is_file()


def test_load_empty(checkpoint):
    assert not checkpoint.load()


def test_load_after_initialize(checkpoint):
    checkpoint.initialize("Test PDF", 5)
    # Create a new checkpoint instance to test loading from disk
    fresh = PipelineCheckpoint(checkpoint._dir, "test-source-abc123")
    assert fresh.load()


def test_mark_chunk_done(checkpoint):
    checkpoint.initialize("Test PDF", 3)
    assert not checkpoint.is_chunk_done(0)

    checkpoint.mark_chunk_done(0, {"new_nodes": 3, "new_edges": 2, "updated_nodes": 0, "reinforced_edges": 0, "insights": 1})
    assert checkpoint.is_chunk_done(0)
    assert not checkpoint.is_chunk_done(1)


def test_accumulated_counts(checkpoint):
    checkpoint.initialize("Test PDF", 3)
    checkpoint.mark_chunk_done(0, {"new_nodes": 3, "new_edges": 2, "updated_nodes": 1, "reinforced_edges": 0, "insights": 1})
    checkpoint.mark_chunk_done(1, {"new_nodes": 1, "new_edges": 4, "updated_nodes": 0, "reinforced_edges": 2, "insights": 0})

    totals = checkpoint.get_accumulated_counts()
    assert totals["new_nodes"] == 4
    assert totals["new_edges"] == 6
    assert totals["updated_nodes"] == 1
    assert totals["reinforced_edges"] == 2
    assert totals["insights"] == 1


def test_record_error(checkpoint):
    checkpoint.initialize("Test PDF", 3)
    checkpoint.record_error(2, "timeout after 30s")
    errors = checkpoint.get_errors()
    assert len(errors) == 1
    assert "Chunk 2" in errors[0]


def test_complete_archives_file(checkpoint, checkpoint_dir):
    checkpoint.initialize("Test PDF", 3)
    checkpoint.mark_chunk_done(0, {"new_nodes": 1, "new_edges": 0, "updated_nodes": 0, "reinforced_edges": 0, "insights": 0})
    checkpoint.complete()
    # File should be gone from the active directory
    assert not (checkpoint_dir / "test-source-abc123.json").exists()
    # File should be archived in completed/
    archived = checkpoint_dir / "completed" / "test-source-abc123.json"
    assert archived.exists()
    state = json.loads(archived.read_text())
    assert "completed_at" in state


def test_resume_from_disk(checkpoint_dir):
    """Simulate: write checkpoint, create new instance, load."""
    cp1 = PipelineCheckpoint(checkpoint_dir, "resume-test")
    cp1.initialize("Resume PDF", 5)
    cp1.mark_chunk_done(0, {"new_nodes": 2, "new_edges": 1, "updated_nodes": 0, "reinforced_edges": 0, "insights": 0})
    cp1.mark_chunk_done(1, {"new_nodes": 1, "new_edges": 3, "updated_nodes": 0, "reinforced_edges": 0, "insights": 1})
    cp1.record_error(2, "rate limit")

    # Simulate restart
    cp2 = PipelineCheckpoint(checkpoint_dir, "resume-test")
    assert cp2.load()
    assert cp2.is_chunk_done(0)
    assert cp2.is_chunk_done(1)
    assert not cp2.is_chunk_done(2)
    assert not cp2.is_chunk_done(3)
    assert len(cp2.get_errors()) == 1

    totals = cp2.get_accumulated_counts()
    assert totals["new_nodes"] == 3


def test_corrupt_checkpoint_resets(checkpoint_dir):
    """If the checkpoint file is corrupt, load() returns False and we start fresh."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "corrupt.json").write_text("not valid json{{{")
    cp = PipelineCheckpoint(checkpoint_dir, "corrupt")
    assert not cp.load()


def test_save_writes_via_temp_file_and_os_replace(checkpoint, checkpoint_dir, monkeypatch):
    """_save() must go through a temp file + os.replace(), not a direct write_text() — a crash
    between open() and the final bytes hitting disk must never leave a truncated/corrupt
    checkpoint that load() then has to discard, silently re-processing already-committed
    chunks."""
    calls: list[tuple[str, str]] = []
    real_replace = os.replace

    def _spy_replace(src, dst):
        calls.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr("openclaw_brain.knowledge.checkpoint.os.replace", _spy_replace)
    checkpoint.initialize("Test PDF", 3)

    assert len(calls) == 1
    src, dst = calls[0]
    assert dst == str(checkpoint_dir / "test-source-abc123.json")
    assert src != dst  # a distinct temp path, not the live file written in place
    tmp_name = os.path.basename(src)
    assert tmp_name.startswith(".test-source-abc123.json.")
    assert tmp_name.endswith(".tmp")
    # the final file is valid, complete JSON — the temp file was fully written before replace
    state = json.loads((checkpoint_dir / "test-source-abc123.json").read_text())
    assert state["title"] == "Test PDF"


def test_save_failure_does_not_corrupt_prior_checkpoint(checkpoint, checkpoint_dir, monkeypatch):
    """A crash/error partway through a save (simulated here via a failing os.replace) must
    leave the PRIOR valid checkpoint completely intact — never a partial/corrupt overwrite —
    and must not leave a stray temp file behind."""
    checkpoint.initialize("Test PDF", 3)
    checkpoint.mark_chunk_done(
        0, {"new_nodes": 1, "new_edges": 0, "updated_nodes": 0, "reinforced_edges": 0, "insights": 0}
    )
    good_path = checkpoint_dir / "test-source-abc123.json"
    good_content = good_path.read_text()

    def _boom_replace(src, dst):
        raise OSError("disk full mid-replace")

    monkeypatch.setattr("openclaw_brain.knowledge.checkpoint.os.replace", _boom_replace)

    # This write fails partway (at the replace step) — must not raise out of mark_chunk_done
    # (matches the pre-existing "log and swallow" contract every caller relies on) and must not
    # touch the already-good file on disk.
    checkpoint.mark_chunk_done(
        1, {"new_nodes": 9, "new_edges": 9, "updated_nodes": 9, "reinforced_edges": 9, "insights": 9}
    )

    assert good_path.read_text() == good_content
    leftover_tmp = [
        p for p in checkpoint_dir.iterdir()
        if p.name.startswith(".test-source-abc123.json.") and p.name.endswith(".tmp")
    ]
    assert leftover_tmp == []


def test_idempotent_mark(checkpoint):
    """Marking the same chunk twice doesn't duplicate it."""
    checkpoint.initialize("Test PDF", 3)
    checkpoint.mark_chunk_done(0, {"new_nodes": 1, "new_edges": 0, "updated_nodes": 0, "reinforced_edges": 0, "insights": 0})
    checkpoint.mark_chunk_done(0, {"new_nodes": 1, "new_edges": 0, "updated_nodes": 0, "reinforced_edges": 0, "insights": 0})

    # Load from disk and verify
    fresh = PipelineCheckpoint(checkpoint._dir, "test-source-abc123")
    fresh.load()
    state = json.loads((checkpoint._dir / "test-source-abc123.json").read_text())
    assert state["completed_chunks"].count(0) == 1


# ── Generic extra-data slot (save_extra/load_extra) — added for
# pipeline.py::_ingest_html_figures_only's slide-analysis resumability, reusing this SAME
# checkpoint file rather than inventing a second one. ──


def test_load_extra_default_on_never_saved(checkpoint):
    assert checkpoint.load_extra("slide_analyses") is None
    assert checkpoint.load_extra("slide_analyses", {}) == {}
    assert checkpoint.load_extra("nonexistent_key", "fallback") == "fallback"


def test_save_extra_then_load_extra_round_trips_in_memory(checkpoint):
    checkpoint.save_extra("slide_analyses", {"0": {"page_num": 1}, "1": {"page_num": 2}})
    assert checkpoint.load_extra("slide_analyses") == {"0": {"page_num": 1}, "1": {"page_num": 2}}


def test_save_extra_persists_to_disk_without_initialize(checkpoint, checkpoint_dir):
    """save_extra must work on a brand-new checkpoint that was never initialize()'d — the
    slide-analysis pre-pass runs BEFORE stage 1-6's checkpoint.initialize() call."""
    checkpoint.save_extra("slide_analyses", {"0": {"page_num": 1}})
    assert (checkpoint_dir / "test-source-abc123.json").is_file()

    fresh = PipelineCheckpoint(checkpoint_dir, "test-source-abc123")
    assert fresh.load()
    assert fresh.load_extra("slide_analyses") == {"0": {"page_num": 1}}


def test_save_extra_survives_alongside_chunk_tracking(checkpoint):
    """The SAME checkpoint file/object correctly carries BOTH a caller's extra key AND the
    standard chunk-tracking state — mirrors the real sequence in
    _ingest_html_figures_only (extra data saved first) followed by _ingest_from_parsed
    (mark_chunk_done saved later, same underlying file)."""
    checkpoint.save_extra("slide_analyses", {"0": {"page_num": 1}})
    checkpoint.initialize("Test PDF", 2)
    checkpoint.mark_chunk_done(0, {"new_nodes": 1, "new_edges": 0, "updated_nodes": 0, "reinforced_edges": 0, "insights": 0})

    fresh = PipelineCheckpoint(checkpoint._dir, "test-source-abc123")
    assert fresh.load()
    assert fresh.is_chunk_done(0)
    # initialize() resets only its OWNED chunk-tracking keys; foreign keys written via
    # save_extra survive. (Before 2026-07-18 it wholesale-replaced the state dict, which
    # silently wiped hours of cached slide-VLM analyses on any --reprocess run — found in
    # review; this assertion guards the fix.)
    assert fresh.load_extra("slide_analyses") == {"0": {"page_num": 1}}


def test_load_extra_reads_value_saved_before_initialize_if_not_reinitialized(checkpoint_dir):
    """The realistic resumability sequence: save_extra (slide analysis phase) → load() by a
    LATER, separate PipelineCheckpoint instance (stage 1-6) that finds the file already exists
    and therefore does NOT call initialize() (mirrors pipeline.py's `if checkpoint.load() and
    not reprocess: resume else: initialize()` branch) — the extra key must still be there."""
    cp1 = PipelineCheckpoint(checkpoint_dir, "figs-source")
    cp1.save_extra("slide_analyses", {"3": {"page_num": 4, "equations": []}})

    cp2 = PipelineCheckpoint(checkpoint_dir, "figs-source")
    assert cp2.load()  # file exists -> True -> caller skips initialize()
    assert cp2.load_extra("slide_analyses") == {"3": {"page_num": 4, "equations": []}}
    # Chunk-tracking methods degrade gracefully (empty/false) even though initialize() never ran.
    assert cp2.get_accumulated_counts() == {
        "new_nodes": 0, "updated_nodes": 0, "new_edges": 0, "reinforced_edges": 0, "insights": 0,
    }
    assert not cp2.is_chunk_done(0)
    cp2.mark_chunk_done(0, {"new_nodes": 1, "new_edges": 0, "updated_nodes": 0, "reinforced_edges": 0, "insights": 0})
    assert cp2.is_chunk_done(0)
    # The extra key survives mark_chunk_done (which only touches completed_chunks/chunk_results).
    assert cp2.load_extra("slide_analyses") == {"3": {"page_num": 4, "equations": []}}


def test_initialize_preserves_foreign_keys_across_objects(checkpoint_dir):
    """The exact --reprocess hazard from review (2026-07-18): the slide phase caches
    analyses via save_extra on one PipelineCheckpoint object, then _ingest_from_parsed
    creates a SECOND object on the same file and (reprocess=True) calls initialize().
    The cache must survive that re-initialization."""
    slide_cp = PipelineCheckpoint(checkpoint_dir, "figs-shared")
    slide_cp.save_extra("slide_analyses", {"3": {"page_num": 4, "equations": ["E"]}})

    chunk_cp = PipelineCheckpoint(checkpoint_dir, "figs-shared")
    assert chunk_cp.load()
    chunk_cp.initialize("Reprocessed Deck", 7)  # reprocess=True path

    fresh = PipelineCheckpoint(checkpoint_dir, "figs-shared")
    assert fresh.load()
    assert fresh.load_extra("slide_analyses") == {"3": {"page_num": 4, "equations": ["E"]}}
    assert not fresh.is_chunk_done(0)  # owned chunk state genuinely reset
    assert fresh._state["total_chunks"] == 7


def test_ensure_meta_backfills_identity(checkpoint_dir):
    """A file created only by save_extra() lacks source_id/title/total_chunks (the resume
    branch skips initialize()); ensure_meta backfills them without touching other state."""
    slide_cp = PipelineCheckpoint(checkpoint_dir, "figs-meta")
    slide_cp.save_extra("slide_analyses", {"0": {}})

    chunk_cp = PipelineCheckpoint(checkpoint_dir, "figs-meta")
    assert chunk_cp.load()
    chunk_cp.ensure_meta("Deck Title", 12)

    fresh = PipelineCheckpoint(checkpoint_dir, "figs-meta")
    assert fresh.load()
    assert fresh._state["source_id"] == "figs-meta"
    assert fresh._state["title"] == "Deck Title"
    assert fresh._state["total_chunks"] == 12
    assert fresh.load_extra("slide_analyses") == {"0": {}}
