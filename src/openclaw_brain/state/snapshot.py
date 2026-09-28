"""Atomic state snapshots and recovery.

Persists RuntimeState to disk as JSON, with atomic writes (write to
temp file, then rename) to prevent corruption on crash.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from openclaw_brain.state.models import RuntimeState


class SnapshotStore:
    """Manages atomic state snapshots on disk."""

    def __init__(self, state_dir: Path):
        self._state_dir = state_dir
        self._state_dir.mkdir(parents=True, exist_ok=True)

    @property
    def current_path(self) -> Path:
        return self._state_dir / "runtime_state.json"

    @property
    def history_dir(self) -> Path:
        d = self._state_dir / "snapshots"
        d.mkdir(exist_ok=True)
        return d

    def save(self, state: RuntimeState) -> Path:
        """Atomically save state to disk. Returns the path written."""
        state.updated_at = datetime.now()
        data = state.to_dict()

        # Atomic write: temp file → rename
        fd, tmp_path = tempfile.mkstemp(
            dir=self._state_dir, suffix=".json.tmp"
        )
        try:
            with open(fd, "w") as f:
                json.dump(data, f, indent=2, default=str)
            Path(tmp_path).rename(self.current_path)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

        return self.current_path

    def save_checkpoint(self, state: RuntimeState, label: str = "") -> Path:
        """Save a named checkpoint to the history directory."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{ts}_{label}.json" if label else f"{ts}.json"
        path = self.history_dir / name

        data = state.to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return path

    def load(self) -> RuntimeState | None:
        """Load the current state from disk. Returns None if no state exists."""
        if not self.current_path.exists():
            return None
        with open(self.current_path) as f:
            data = json.load(f)
        return self._parse_state(data)

    def load_checkpoint(self, name: str) -> RuntimeState | None:
        """Load a specific checkpoint by filename."""
        path = self.history_dir / name
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return self._parse_state(data)

    def list_checkpoints(self) -> list[str]:
        """List available checkpoint filenames, newest first."""
        if not self.history_dir.exists():
            return []
        return sorted(
            [p.name for p in self.history_dir.glob("*.json")],
            reverse=True,
        )

    @staticmethod
    def _parse_state(data: dict[str, Any]) -> RuntimeState:
        """Parse a JSON dict into RuntimeState, handling datetime strings."""
        return RuntimeState.model_validate(data)
