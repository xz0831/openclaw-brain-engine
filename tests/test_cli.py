"""Tests for cli.py exit-code behavior (W-D2 defect 1).

`status`, `apply-schema`, and `export-obsidian` used to catch all exceptions, print an error
string, and fall through with no `raise SystemExit(1)` / re-raise — so a total failure (e.g. Neo4j
unreachable) still exited 0, breaking any cron/health-check/deploy script that trusts the exit
code. Fixed to match every other mutating command's convention (`raise SystemExit(1)` after the
error echo). Success output/behavior is untouched — covered here too as a regression guard.

No real Neo4j needed: GraphStore/ObsidianExporter's connect()/apply_schema()/export() are mocked
at the class level (they're imported fresh inside each cli.py command function, so patching the
class object directly — same object those local imports bind to — is sufficient).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from click.testing import CliRunner

from openclaw_brain.cli import main
from openclaw_brain.export.obsidian import ObsidianExporter
from openclaw_brain.knowledge.graph.store import GraphStore


# ── status ──


def test_status_exits_nonzero_on_connection_failure(monkeypatch):
    monkeypatch.setattr(
        GraphStore, "connect", AsyncMock(side_effect=RuntimeError("connection refused")),
    )
    monkeypatch.setattr(GraphStore, "close", AsyncMock(return_value=None))

    result = CliRunner().invoke(main, ["status"])

    assert result.exit_code != 0
    assert "Neo4j: connection failed" in result.output


def test_status_exits_zero_on_success(monkeypatch):
    monkeypatch.setattr(GraphStore, "connect", AsyncMock(return_value=None))
    monkeypatch.setattr(GraphStore, "close", AsyncMock(return_value=None))
    monkeypatch.setattr(GraphStore, "get_stats", AsyncMock(return_value={"Concept": 3}))

    result = CliRunner().invoke(main, ["status"])

    assert result.exit_code == 0
    assert "Neo4j: connected" in result.output
    assert "Concept: 3" in result.output


# ── apply-schema ──


def test_apply_schema_exits_nonzero_on_failure(monkeypatch):
    monkeypatch.setattr(
        GraphStore, "connect", AsyncMock(side_effect=RuntimeError("connection refused")),
    )
    monkeypatch.setattr(GraphStore, "close", AsyncMock(return_value=None))

    result = CliRunner().invoke(main, ["apply-schema"])

    assert result.exit_code != 0
    assert "Schema application failed" in result.output


def test_apply_schema_exits_zero_on_success(monkeypatch):
    monkeypatch.setattr(GraphStore, "connect", AsyncMock(return_value=None))
    monkeypatch.setattr(GraphStore, "close", AsyncMock(return_value=None))
    monkeypatch.setattr(GraphStore, "apply_schema", AsyncMock(return_value=None))

    result = CliRunner().invoke(main, ["apply-schema"])

    assert result.exit_code == 0
    assert "Schema applied from" in result.output


# ── export-obsidian ──


def test_export_obsidian_exits_nonzero_on_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ObsidianExporter, "connect", AsyncMock(side_effect=RuntimeError("connection refused")),
    )
    monkeypatch.setattr(ObsidianExporter, "close", AsyncMock(return_value=None))

    result = CliRunner().invoke(main, ["export-obsidian", "--vault", str(tmp_path / "vault")])

    assert result.exit_code != 0
    assert "Export failed" in result.output


def test_export_obsidian_exits_zero_on_success(monkeypatch, tmp_path):
    monkeypatch.setattr(ObsidianExporter, "connect", AsyncMock(return_value=None))
    monkeypatch.setattr(ObsidianExporter, "close", AsyncMock(return_value=None))
    monkeypatch.setattr(ObsidianExporter, "export", AsyncMock(return_value={"Concept": 2}))

    result = CliRunner().invoke(main, ["export-obsidian", "--vault", str(tmp_path / "vault")])

    assert result.exit_code == 0
    assert "Exported 2 nodes" in result.output
