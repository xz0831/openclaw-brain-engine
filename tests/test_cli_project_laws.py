"""Unit test for cli.py's `project-laws` default `--raw` JSONL list (E-track ③ I2): the
full-registry rollout's raw JSONL must join the E1/E1b pilot outputs as a THIRD standing input,
never replace them — `project-laws --apply` (no `--raw`) is expected to keep picking up all three
files as new suites are added over time."""

from __future__ import annotations

from pathlib import Path

from openclaw_brain.cli import _default_law_jsonl_paths

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_default_law_jsonl_paths_includes_e1_e1b_and_full_registry():
    paths = _default_law_jsonl_paths()
    names = [Path(p).name for p in paths]
    assert names == [
        "e1_cross_pdk_raw.jsonl",
        "e1b_statistical_cross_pdk_raw.jsonl",
        "e_rollout_full_registry_raw.jsonl",
    ]


def test_default_law_jsonl_paths_are_absolute_under_experiments_dir():
    for p in _default_law_jsonl_paths():
        path = Path(p)
        assert path.is_absolute()
        assert path.parent == REPO_ROOT / "experiments"
