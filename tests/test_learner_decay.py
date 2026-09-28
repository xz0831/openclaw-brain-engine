"""Tests for knowledge/learner_decay.py — UNDERSTANDS-edge confidence decay (S5 learner model,
docs/specs/S5_LEARNER_MODEL_DESIGN.md §3.2 Option C).

Mocked-driver only (no live Neo4j) — mirrors tests/test_reinforcement.py's mocked-driver section
(_FakeSession/_FakeDriver), extended with a configurable `.single()` return value since
apply_understands_decay needs an actual `{"affected": n}` row back, unlike that file's
write-only `_FakeResult.single() -> None`.
"""

import math

import pytest

from openclaw_brain.config import LearnerConfig, Neo4jConfig
from openclaw_brain.knowledge.graph.store import GraphStore
from openclaw_brain.knowledge.learner_decay import _LN2, apply_understands_decay


class _FakeResult:
    def __init__(self, record: dict | None = None):
        self._record = record

    async def single(self):
        return self._record


class _FakeSession:
    def __init__(self, calls: list[tuple[str, dict]], record: dict | None):
        self._calls = calls
        self._record = record

    async def run(self, query, params=None):
        self._calls.append((query, params or {}))
        return _FakeResult(self._record)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeDriver:
    def __init__(self, calls: list[tuple[str, dict]], record: dict | None):
        self._calls = calls
        self._record = record

    def session(self, database=None):
        return _FakeSession(self._calls, self._record)


def _mocked_store(record: dict | None = None) -> tuple[GraphStore, list[tuple[str, dict]]]:
    """A GraphStore wired to a fake driver — no live Neo4j required."""
    calls: list[tuple[str, dict]] = []
    store = GraphStore(Neo4jConfig())
    store._driver = _FakeDriver(calls, record)
    return store, calls


@pytest.mark.asyncio
async def test_apply_understands_decay_returns_affected_count():
    store, calls = _mocked_store(record={"affected": 3})
    cfg = LearnerConfig(understands_half_life_days=30.0, understands_confidence_floor=0.05)

    affected = await apply_understands_decay(store, cfg)

    assert affected == 3
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_apply_understands_decay_query_targets_understands_edges_only():
    store, calls = _mocked_store(record={"affected": 0})
    await apply_understands_decay(store, LearnerConfig())

    query, _params = calls[0]
    assert ":UNDERSTANDS" in query
    assert "Learner" in query
    # Never touches the truth-layer decay mechanism's own label/property.
    assert "ReinforcementEngine" not in query


@pytest.mark.asyncio
async def test_apply_understands_decay_passes_config_driven_half_life_and_floor():
    store, calls = _mocked_store(record={"affected": 0})
    cfg = LearnerConfig(understands_half_life_days=10.0, understands_confidence_floor=0.2)

    await apply_understands_decay(store, cfg)

    _, params = calls[0]
    assert params["half_life"] == 10.0
    assert params["floor"] == 0.2


@pytest.mark.asyncio
async def test_apply_understands_decay_no_record_returns_zero():
    store, _calls = _mocked_store(record=None)
    affected = await apply_understands_decay(store, LearnerConfig())
    assert affected == 0


def test_half_life_constant_matches_ln2():
    """Regression pin for the exponential-decay formula's correctness — verified live
    (read-only) against the running Neo4j instance during implementation:
    RETURN exp(-ln2 * half_life / half_life) evaluated to exactly 0.5, the definition of
    half-life. Re-asserted here in pure Python so a future edit to _LN2 can't silently drift."""
    assert math.isclose(_LN2, math.log(2), rel_tol=1e-12)
    assert math.isclose(math.exp(-_LN2 * 30.0 / 30.0), 0.5, rel_tol=1e-12)
