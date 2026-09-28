"""project_executable must dispatch ALL seed recipes (seed_recipes + digital_seed_recipes +
statistical_seed_recipes), engine-aware (no hardcoded NgspiceRunner), skip a recipe whose engine's
runner is unavailable per-recipe (never fail the whole batch), and persist to the git-corpus SSOT
on apply=True. Mocked runners + mocked graph — no real ngspice/iverilog, no Neo4j.
"""

from __future__ import annotations

import pytest

from openclaw_brain.agent import BrainAgent
from openclaw_brain.config import load_config
from openclaw_brain.knowledge.executable import engines as engines_mod
from openclaw_brain.knowledge.executable.corpus import SpecimenCorpus
from openclaw_brain.knowledge.executable.seeds import (
    digital_seed_recipes, seed_recipes, statistical_seed_recipes,
)

_DIGITAL_CLASSES = {r.topology_class for r in digital_seed_recipes()}
_ANALOG_CLASSES = {r.topology_class for r in seed_recipes()}
_STAT_CLASSES = {r.topology_class for r in statistical_seed_recipes()}

# topology_class is NOT a unique key across registries (e.g. "ota_5t_nmos_in" and
# "comparator_continuous_nmos" each appear in BOTH seed_recipes() and statistical_seed_recipes())
# so per-source coverage is verified via each recipe's globally-unique claim-card ids instead.
_DIGITAL_CLAIM_IDS = {c.id for r in digital_seed_recipes() for c in r.claim_cards}
_ANALOG_CLAIM_IDS = {c.id for r in seed_recipes() for c in r.claim_cards}
_STAT_CLAIM_IDS = {c.id for r in statistical_seed_recipes() for c in r.claim_cards}
_ALL_CLAIM_IDS = _DIGITAL_CLAIM_IDS | _ANALOG_CLAIM_IDS | _STAT_CLAIM_IDS

# The total recipe count is DERIVED (not hardcoded) so adding a seed recipe (e.g. S3-inc2a's
# ptat_ctat_core_bjt) never silently rots this test — it was 22 before that addition, 23 after.
_ALL_RECIPES_COUNT = len(seed_recipes()) + len(digital_seed_recipes()) + len(statistical_seed_recipes())

# 12 monotonic points is ample headroom for every seed's largest explicit sweep (5 points).
_SERIES = [(float(i), 5.0 + i) for i in range(1, 13)]


class _AnyKeySeries(dict):
    """`.get(any_key)` always returns the same deterministic series — lets one fake runner drive
    every seed recipe (each keys its series by a different knob) with no per-recipe wiring."""

    def __init__(self, series):
        super().__init__()
        self._series = series

    def get(self, key, default=None):
        return self._series


class _FakeRunner:
    """Records dispatch; deterministic; never touches a real simulator."""

    def __init__(self, avail: bool = True):
        self.calls = 0
        self._avail = avail

    def available(self) -> bool:
        return self._avail

    def measure(self, deck, timeout=300):
        self.calls += 1
        return _AnyKeySeries(_SERIES)


class _MockGraph:
    """Supports both the resolver (`run_read_query`, no candidates -> every link unresolved, no
    phantom edges) and the projector (`write_batch`)."""

    def __init__(self):
        self.queries = []
        self.writes = []

    async def run_read_query(self, query, params=None):
        self.queries.append((query, params))
        return []

    async def write_batch(self, nodes=None, updates=None, edges=None):
        self.writes.append({"nodes": nodes or [], "edges": edges or []})


class _MockJournal:
    def __init__(self):
        self.logs = []

    def log(self, action, **kw):
        self.logs.append((action, kw))


@pytest.fixture
def fake_engines(monkeypatch):
    """Swap the ENGINES registry's runners for deterministic fakes. `agent.project_executable`
    re-imports `ENGINES` (the same module-level dict object) on every call, so mutating entries
    in place here is visible regardless of import timing."""
    ng = _FakeRunner()
    iv = _FakeRunner()
    monkeypatch.setitem(engines_mod.ENGINES, "ngspice", engines_mod.EngineSpec("ngspice", ng))
    monkeypatch.setitem(engines_mod.ENGINES, "iverilog", engines_mod.EngineSpec("iverilog", iv))
    return {"ngspice": ng, "iverilog": iv}


@pytest.fixture
def agent(tmp_path):
    cfg = load_config()
    cfg.openclaw.state_dir = str(tmp_path / "state")
    cfg.executable.corpus_dir = str(tmp_path / "corpus")
    a = BrainAgent(cfg)
    a._started = True
    a._journal = _MockJournal()
    a._graph = _MockGraph()
    return a


@pytest.mark.asyncio
async def test_all_seed_recipes_dispatched(agent, fake_engines):
    out = await agent.project_executable(apply=False)
    assert out["skipped"] == []
    assert len(out["specimens"]) == _ALL_RECIPES_COUNT   # one entry per RECIPE, not deduped by class
    assert fake_engines["ngspice"].calls > 0
    assert fake_engines["iverilog"].calls > 0
    # every registry's claims are represented (claim-card id is the globally-unique identity;
    # topology_class collides across registries, see the comment above)
    got_claim_ids = {cid for e in out["specimens"] for cid in e["verdicts"]}
    assert got_claim_ids == _ALL_CLAIM_IDS
    assert got_claim_ids >= _DIGITAL_CLAIM_IDS
    assert got_claim_ids >= _ANALOG_CLAIM_IDS
    assert got_claim_ids >= _STAT_CLAIM_IDS


@pytest.mark.asyncio
async def test_correct_runner_type_per_engine(agent, fake_engines):
    out = await agent.project_executable(apply=False)
    for entry in out["specimens"]:
        expected = "iverilog" if entry["topology_class"] in _DIGITAL_CLASSES else "ngspice"
        assert entry["engine"] == expected, entry
    # nothing hardcoded to NgspiceRunner: the digital recipes' deck(s) actually ran on the
    # iverilog fake (>=1 measure() call per digital recipe — a recipe can sweep >1 (knob, metric)
    # pair, e.g. the composed SS-ADC back-end, so this is a lower bound, not an exact count).
    assert fake_engines["iverilog"].calls >= len(digital_seed_recipes())


@pytest.mark.asyncio
async def test_unavailable_engine_skips_per_recipe_not_whole_batch(agent, fake_engines):
    fake_engines["iverilog"]._avail = False   # simulate: iverilog missing, ngspice present
    out = await agent.project_executable(apply=False)
    assert len(out["skipped"]) == len(digital_seed_recipes())
    assert {s["topology_class"] for s in out["skipped"]} == _DIGITAL_CLASSES
    for s in out["skipped"]:
        assert s["engine"] == "iverilog"
        assert "unavailable" in s["reason"]
    # the batch did NOT fail: every ngspice-engine recipe (analog + statistical) still ran
    assert len(out["specimens"]) == len(seed_recipes()) + len(statistical_seed_recipes())
    assert fake_engines["iverilog"].calls == 0
    assert fake_engines["ngspice"].calls > 0


@pytest.mark.asyncio
async def test_both_engines_unavailable_skips_everything_no_exception(agent, fake_engines):
    fake_engines["ngspice"]._avail = False
    fake_engines["iverilog"]._avail = False
    out = await agent.project_executable(apply=False)
    assert out["specimens"] == []
    assert len(out["skipped"]) == _ALL_RECIPES_COUNT


@pytest.mark.asyncio
async def test_apply_true_persists_to_config_corpus_dir(agent, fake_engines, tmp_path):
    out = await agent.project_executable(apply=True)
    assert out["skipped"] == []
    assert len(out["specimens"]) == _ALL_RECIPES_COUNT
    # every recipe wrote through the projector (mocked graph) ...
    assert len(agent._graph.writes) == _ALL_RECIPES_COUNT
    # ...and persisted into the CONFIG-DRIVEN corpus dir (no explicit corpus_dir override passed)
    corpus_dir = tmp_path / "corpus"
    assert corpus_dir.is_dir()
    corpus = SpecimenCorpus(str(corpus_dir))
    stored_classes = set(corpus.list_classes())
    assert stored_classes == _ANALOG_CLASSES | _DIGITAL_CLASSES | _STAT_CLASSES
    for entry in out["specimens"]:
        assert "written" in entry


@pytest.mark.asyncio
async def test_apply_false_does_not_write_corpus(agent, fake_engines, tmp_path):
    await agent.project_executable(apply=False)
    assert not (tmp_path / "corpus").exists()
    assert agent._graph.writes == []


@pytest.mark.asyncio
async def test_explicit_corpus_dir_overrides_config_default(agent, fake_engines, tmp_path):
    override_dir = tmp_path / "override_corpus"
    out = await agent.project_executable(apply=True, corpus_dir=str(override_dir))
    assert len(out["specimens"]) == _ALL_RECIPES_COUNT
    assert override_dir.is_dir()
    assert not (tmp_path / "corpus").exists()   # the config default dir was never used
