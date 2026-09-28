"""Filesystem-backed evidence vault for verbatim source artifacts."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path


class EvidenceVault:
    """Content-addressed storage for extracted evidence and source artifacts."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser()

    def put_text(self, text: str) -> str:
        """Store verbatim text under chunks/<sha>.txt and return its SHA-256."""
        data = text.encode("utf-8")
        sha = hashlib.sha256(data).hexdigest()
        dest = self.root / "chunks" / f"{sha}.txt"
        if dest.exists():
            return sha
        self._atomic_write_bytes(dest, data)
        return sha

    def get_text(self, sha: str) -> str | None:
        """Return verbatim chunk text for a SHA-256 key, or None when absent."""
        if Path(sha).name != sha:
            return None
        path = self.root / "chunks" / f"{sha}.txt"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def put_file(self, src: Path, kind: str, key: str) -> Path:
        """Store a source artifact by kind/key and return the destination path."""
        src = Path(src)
        dest = self._file_dest(src, kind, key)
        if dest.exists():
            return dest

        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=dest.parent,
            prefix=f".{dest.name}.",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as out, src.open("rb") as inp:
                shutil.copyfileobj(inp, out)
                out.flush()
                os.fsync(out.fileno())
            tmp_path.replace(dest)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        return dest

    def put_dir_files(
        self,
        src_dir: Path,
        kind: str,
        key: str,
        patterns: tuple[str, ...] = ("*.md", "*_content_list.json"),
    ) -> list[Path]:
        """Store selected files from a directory under the artifact namespace."""
        src_dir = Path(src_dir)
        copied: list[Path] = []
        seen: set[Path] = set()
        for pattern in patterns:
            for src in sorted(src_dir.rglob(pattern)):
                if not src.is_file() or src in seen:
                    continue
                seen.add(src)
                copied.append(self.put_file(src, kind, key))
        return copied

    def has(self, sha: str) -> bool:
        """Return whether a verbatim text chunk exists for the SHA-256 key."""
        if Path(sha).name != sha:
            return False
        return (self.root / "chunks" / f"{sha}.txt").exists()

    def _file_dest(self, src: Path, kind: str, key: str) -> Path:
        if Path(key).name != key:
            raise ValueError(f"Invalid evidence key: {key!r}")
        if kind == "pdf":
            return self.root / "pdf" / f"{key}.pdf"
        if kind == "html":
            return self.root / "html" / f"{key}.html"
        if kind == "mineru":
            return self.root / "mineru" / key / src.name
        raise ValueError(f"Unsupported evidence kind: {kind!r}")

    @staticmethod
    def _atomic_write_bytes(dest: Path, data: bytes) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=dest.parent,
            prefix=f".{dest.name}.",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(dest)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
