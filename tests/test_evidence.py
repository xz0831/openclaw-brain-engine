"""Tests for the filesystem evidence vault."""

from __future__ import annotations

from openclaw_brain.knowledge.evidence import EvidenceVault


def test_put_get_text_roundtrip(tmp_path):
    vault = EvidenceVault(tmp_path / "evidence")
    text = "verbatim source text\nwith multiple lines"

    sha = vault.put_text(text)

    assert len(sha) == 64
    assert vault.has(sha)
    assert vault.get_text(sha) == text
    assert (tmp_path / "evidence" / "chunks" / f"{sha}.txt").exists()


def test_put_text_is_content_addressed_and_skips_duplicates(tmp_path):
    vault = EvidenceVault(tmp_path / "evidence")

    sha1 = vault.put_text("same body")
    sha2 = vault.put_text("same body")

    chunk_files = list((tmp_path / "evidence" / "chunks").glob("*.txt"))
    tmp_files = list((tmp_path / "evidence" / "chunks").glob("*.tmp"))
    assert sha1 == sha2
    assert len(chunk_files) == 1
    assert tmp_files == []


def test_put_file_skips_duplicate_key(tmp_path):
    vault = EvidenceVault(tmp_path / "evidence")
    src = tmp_path / "source.pdf"
    src.write_bytes(b"first")

    dest = vault.put_file(src, "pdf", "abc123")
    src.write_bytes(b"second")
    duplicate = vault.put_file(src, "pdf", "abc123")

    assert duplicate == dest
    assert dest.read_bytes() == b"first"


def test_put_file_html_kind_lands_under_html_namespace(tmp_path):
    """The "html" evidence kind (added for the lecture-HTML ingest adapter's _persist_html)
    mirrors "pdf" exactly — separate namespace/extension, same content-addressed dedup — see
    EvidenceVault._file_dest."""
    vault = EvidenceVault(tmp_path / "evidence")
    src = tmp_path / "lecture.html"
    src.write_text("<html>synthetic lecture fixture</html>", encoding="utf-8")

    dest = vault.put_file(src, "html", "deadbeef123")

    assert dest == tmp_path / "evidence" / "html" / "deadbeef123.html"
    assert dest.exists()
    assert dest.read_text(encoding="utf-8") == "<html>synthetic lecture fixture</html>"


def test_put_dir_files_copies_mineru_outputs(tmp_path):
    vault = EvidenceVault(tmp_path / "evidence")
    out = tmp_path / "mineru_tmp" / "paper"
    out.mkdir(parents=True)
    (out / "paper.md").write_text("# Paper\n", encoding="utf-8")
    (out / "paper_content_list.json").write_text("[]\n", encoding="utf-8")
    (out / "ignored.txt").write_text("ignore\n", encoding="utf-8")

    copied = vault.put_dir_files(tmp_path / "mineru_tmp", "mineru", "checksum")

    assert sorted(path.name for path in copied) == [
        "paper.md",
        "paper_content_list.json",
    ]
    assert (tmp_path / "evidence" / "mineru" / "checksum" / "paper.md").exists()
    assert not (tmp_path / "evidence" / "mineru" / "checksum" / "ignored.txt").exists()
