"""Tests for knowledge/executable/growth.py (S3 — corpus growth automation, D3 AUTO lane).

`author_recipe` is MOCKED at the growth layer throughout (monkeypatch `growth.author_recipe`) — these
tests never call an LLM (its own structured/raw paths are covered by test_executable_recipe.py).
Runners/projectors/graph stores are fakes — no real ngspice/iverilog, no Neo4j.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openclaw_brain.agent import BrainAgent
from openclaw_brain.config import load_config
from openclaw_brain.knowledge.executable import engines as engines_mod
from openclaw_brain.knowledge.executable import growth
from openclaw_brain.knowledge.executable.corpus import SpecimenCorpus
from openclaw_brain.knowledge.executable.models import (
    AnalogPVT, ClaimCard, MechanismClaim, QuantTest, VerificationRecipe,
)
from openclaw_brain.knowledge.executable.templates import TEMPLATES as TEMPLATES_FOR_COUNT
from openclaw_brain.knowledge.graph.schema import NodeLabel, NodeProposal


# =====================================================================================================
# match_topology_name / normalization
# =====================================================================================================


class TestMatchTopologyName:
    def test_curated_alias_variants(self):
        assert growth.match_topology_name("5T OTA") == "ota_5t_nmos_in"
        assert growth.match_topology_name("five-transistor OTA") == "ota_5t_nmos_in"
        assert growth.match_topology_name("Five_Transistor_OTA") == "ota_5t_nmos_in"

    def test_backlog_id_alias(self):
        assert growth.match_topology_name("current_mirror_simple") == "current_mirror_simple_nmos"
        assert growth.match_topology_name("two_stage_miller_ota") == "miller_ota_2stage_nmos_in"

    def test_self_match_registered_class(self):
        assert growth.match_topology_name("miller_ota_2stage_nmos_in") == "miller_ota_2stage_nmos_in"
        assert growth.match_topology_name("Miller Ota 2Stage Nmos In") == "miller_ota_2stage_nmos_in"

    def test_unknown_name_returns_none(self):
        assert growth.match_topology_name("some totally unrelated circuit") is None

    def test_empty_name_returns_none(self):
        assert growth.match_topology_name("") is None
        assert growth.match_topology_name(None) is None  # type: ignore[arg-type]


# =====================================================================================================
# TOPOLOGY_BACKLOG.md parsing
# =====================================================================================================


_BACKLOG_FIXTURE = """# fixture backlog

## (1) TOPOLOGY BACKLOG

| # | topology_class | name | primary_analysis | CIS_rel | source |
|---|---|---|---|---|---|
| 1 | `foo_amp` | Foo amplifier **[DONE]** | .ac | high | X1 |
| 2 | `bar_mirror` | Bar mirror | .dc | high | X2 |
| 3 | `unmapped_thing` | Some unmapped thing | .ac | low | X3 |

## (2) gold set (must NOT be parsed as backlog rows)

| pick | topology | analysis | tract. | what it stresses |
|---|---|---|---|---|
| G1 | `foo_amp` (#4) | .dc | EASY | not a data row (non-digit id) |
"""


class TestParseTopologyBacklog:
    @pytest.fixture
    def backlog_path(self, tmp_path):
        p = tmp_path / "TOPOLOGY_BACKLOG.md"
        p.write_text(_BACKLOG_FIXTURE, encoding="utf-8")
        return p

    def test_missing_file_returns_empty(self, tmp_path):
        assert growth.parse_topology_backlog(tmp_path / "nope.md") == []

    def test_done_row_flagged(self, backlog_path):
        rows = {r.backlog_id: r for r in growth.parse_topology_backlog(backlog_path)}
        assert rows["foo_amp"].done is True
        assert rows["bar_mirror"].done is False

    def test_gold_set_table_not_parsed(self, backlog_path):
        rows = growth.parse_topology_backlog(backlog_path)
        assert len(rows) == 3   # only the 3 numbered dedup-table rows, not the G1 gold-set row

    def test_mapped_vs_unmapped(self, backlog_path, monkeypatch):
        # self-match: TEMPLATES contains "bar_mirror" verbatim -> match_topology_name self-matches it.
        monkeypatch.setattr(growth, "TEMPLATES", {
            "bar_mirror": {"template_ref": "x", "engine": "ngspice", "metrics": [], "knobs": []},
        })
        rows = {r.backlog_id: r for r in growth.parse_topology_backlog(backlog_path)}
        assert rows["bar_mirror"].mapped_class == "bar_mirror"
        assert rows["unmapped_thing"].mapped_class is None
        assert rows["foo_amp"].mapped_class is None  # not in this fake TEMPLATES, no alias either


# =====================================================================================================
# plan_growth — coverage-gap enumeration, novelty gate, caps, ordering, digital exclusion
# =====================================================================================================

FAKE_TEMPLATES = {
    "foo_amp": {"template_ref": "foo_tmpl", "engine": "ngspice", "metrics": ["m1", "m2"], "knobs": ["k1"]},
    "bar_mirror": {"template_ref": "bar_tmpl", "engine": "ngspice", "metrics": ["m3"], "knobs": ["k2", "k3"]},
    "baz_digital": {"template_ref": "baz_tmpl", "engine": "iverilog", "metrics": ["m4"], "knobs": ["k4"]},
}


class _FakeStore:
    """run_read_query returns pre-seeded rows keyed by topology_class (the novelty-gate query)."""

    def __init__(self, verified_by_class: dict[str, list[dict]] | None = None):
        self._verified = verified_by_class or {}
        self.queries: list[tuple[str, dict]] = []

    async def run_read_query(self, query, params=None):
        self.queries.append((query, params or {}))
        tclass = (params or {}).get("tclass")
        return self._verified.get(tclass, [])


@pytest.fixture
def fake_templates(monkeypatch):
    monkeypatch.setattr(growth, "TEMPLATES", FAKE_TEMPLATES)


@pytest.fixture
def empty_backlog(tmp_path):
    """A backlog path with no file -> review_lane_pending.backlog_unmapped stays 0, isolating
    coverage-only tests from the real repo TOPOLOGY_BACKLOG.md."""
    return tmp_path / "no_backlog.md"


class TestPlanGrowthCoverage:
    @pytest.mark.asyncio
    async def test_max_recipes_must_be_positive(self, fake_templates, empty_backlog):
        with pytest.raises(ValueError, match="max_recipes"):
            await growth.plan_growth(
                _FakeStore(), sources=["coverage"], max_recipes=0, backlog_path=empty_backlog,
            )

    @pytest.mark.asyncio
    async def test_unknown_source_rejected(self, fake_templates, empty_backlog):
        with pytest.raises(ValueError, match="unknown growth source"):
            await growth.plan_growth(
                _FakeStore(), sources=["not_a_source"], max_recipes=5, backlog_path=empty_backlog,
            )

    @pytest.mark.asyncio
    async def test_digital_classes_excluded(self, fake_templates, empty_backlog):
        plan = await growth.plan_growth(
            _FakeStore(), sources=["coverage"], max_recipes=100, max_new_cards_per_class=100,
            backlog_path=empty_backlog,
        )
        assert plan.skipped_digital == ["baz_digital"]
        assert all(t.topology_class != "baz_digital" for t in plan.targets)

    @pytest.mark.asyncio
    async def test_full_probe_space_enumerated_when_uncapped(self, fake_templates, empty_backlog):
        # foo_amp: 2 metrics x 1 knob = 2; bar_mirror: 1 metric x 2 knobs = 2 -> 4 total (kind is no
        # longer an enumeration axis — DEFECT 3 revision).
        plan = await growth.plan_growth(
            _FakeStore(), sources=["coverage"], max_recipes=100, max_new_cards_per_class=100,
            backlog_path=empty_backlog,
        )
        assert len(plan.targets) == 4
        assert plan.skipped_covered == 0
        assert plan.per_class_counts == {"foo_amp": 2, "bar_mirror": 2}
        assert all(t.kind_hint is None for t in plan.targets)   # author chooses, never enumerator-mandated

    @pytest.mark.asyncio
    async def test_deterministic_ordering_metric_knob(self, fake_templates, empty_backlog):
        plan = await growth.plan_growth(
            _FakeStore(), sources=["coverage"], max_recipes=100, max_new_cards_per_class=100,
            topology_class="foo_amp", backlog_path=empty_backlog,
        )
        got = [(t.metric, t.knob) for t in plan.targets]
        assert got == [("m1", "k1"), ("m2", "k1")]

    @pytest.mark.asyncio
    async def test_novelty_gate_excludes_existing_verified_probe(self, fake_templates, empty_backlog):
        store = _FakeStore({"foo_amp": [{"metric": "m1", "knob": "k1", "quant_kind": "direction"}]})
        plan = await growth.plan_growth(
            store, sources=["coverage"], max_recipes=100, max_new_cards_per_class=100,
            topology_class="foo_amp", backlog_path=empty_backlog,
        )
        probes = [(t.metric, t.knob) for t in plan.targets]
        assert ("m1", "k1") not in probes
        assert plan.skipped_covered == 1
        assert len(plan.targets) == 1   # 2 - 1 novelty-gated

    @pytest.mark.asyncio
    async def test_invariance_verdict_covers_cell_kind_agnostically(self, fake_templates, empty_backlog):
        # A cell certified as `invariance` is just as covered as one certified `direction` — kind is
        # not part of the coverage key, so no further probe (of any kind) is re-planned for it.
        store = _FakeStore({"foo_amp": [{"metric": "m1", "knob": "k1", "quant_kind": "invariance"}]})
        plan = await growth.plan_growth(
            store, sources=["coverage"], max_recipes=100, max_new_cards_per_class=100,
            topology_class="foo_amp", backlog_path=empty_backlog,
        )
        probes = [(t.metric, t.knob) for t in plan.targets]
        assert ("m1", "k1") not in probes
        assert plan.skipped_covered == 1
        assert len(plan.targets) == 1

    @pytest.mark.asyncio
    async def test_max_new_cards_per_class_caps(self, fake_templates, empty_backlog):
        plan = await growth.plan_growth(
            _FakeStore(), sources=["coverage"], max_recipes=100, max_new_cards_per_class=1,
            backlog_path=empty_backlog,
        )
        assert plan.per_class_counts == {"foo_amp": 1, "bar_mirror": 1}
        assert len(plan.targets) == 2

    @pytest.mark.asyncio
    async def test_max_recipes_caps_overall(self, fake_templates, empty_backlog):
        plan = await growth.plan_growth(
            _FakeStore(), sources=["coverage"], max_recipes=3, max_new_cards_per_class=100,
            backlog_path=empty_backlog,
        )
        assert len(plan.targets) == 3

    @pytest.mark.asyncio
    async def test_topology_class_filter(self, fake_templates, empty_backlog):
        plan = await growth.plan_growth(
            _FakeStore(), sources=["coverage"], max_recipes=100, max_new_cards_per_class=100,
            topology_class="bar_mirror", backlog_path=empty_backlog,
        )
        assert all(t.topology_class == "bar_mirror" for t in plan.targets)
        assert len(plan.targets) == 2


# =====================================================================================================
# plan_growth — growth_queue.jsonl consumption
# =====================================================================================================


def _write_queue_line(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


class TestPlanGrowthQueue:
    @pytest.mark.asyncio
    async def test_pending_entry_becomes_queue_targets_with_grounding(
        self, fake_templates, empty_backlog, tmp_path,
    ):
        _write_queue_line(tmp_path / "growth_queue.jsonl", {
            "ts": "2026-07-03T00:00:00Z", "topology_class": "foo_amp", "matched_from": "node_1",
            "canonical_name": "Foo Amp", "source_id": "src1", "chunk_ids": ["c1", "c2"],
            "confidence": 0.9, "status": "pending",
        })
        plan = await growth.plan_growth(
            _FakeStore(), sources=["queue"], max_recipes=100, max_new_cards_per_class=2,
            state_dir=tmp_path, backlog_path=empty_backlog,
        )
        assert len(plan.targets) == 2
        assert all(t.source == "queue" for t in plan.targets)
        assert all(t.grounding_chunk_ids == ["c1", "c2"] for t in plan.targets)
        assert all(t.topology_class == "foo_amp" for t in plan.targets)

    @pytest.mark.asyncio
    async def test_consumed_entry_not_replanned(self, fake_templates, empty_backlog, tmp_path):
        _write_queue_line(tmp_path / "growth_queue.jsonl", {
            "ts": "t", "topology_class": "foo_amp", "matched_from": "node_1",
            "canonical_name": "Foo Amp", "source_id": "src1", "chunk_ids": [],
            "confidence": 0.9, "status": "pending",
        })
        plan = await growth.plan_growth(
            _FakeStore(), sources=["queue"], max_recipes=100, max_new_cards_per_class=2,
            state_dir=tmp_path, backlog_path=empty_backlog, apply=True,
        )
        # W-D2 defect 7 fix: plan_growth alone must NOT mark the entry consumed anymore — surviving
        # the max_recipes cap is a planning-time fact, not proof anything was ever authored/
        # simulated/certified. It's only staged as a CANDIDATE; execute_growth writes "consumed"
        # (see TestExecuteGrowthQueueConsumption) once it confirms actual success.
        lines = (tmp_path / "growth_queue.jsonl").read_text().splitlines()
        assert len(lines) == 1   # still just the original pending line — no status line yet
        assert json.loads(lines[0])["status"] == "pending"
        assert list(plan.queue_consumed_candidates.keys()) == ["foo_amp"]
        assert [e["matched_from"] for e in plan.queue_consumed_candidates["foo_amp"]] == ["node_1"]

        # Simulate execute_growth having later confirmed a successful projection for this class —
        # append the "consumed" status line the way execute_growth's own write does.
        _write_queue_line(tmp_path / "growth_queue.jsonl", {
            "ts": "t2", "topology_class": "foo_amp", "matched_from": "node_1",
            "canonical_name": "Foo Amp", "source_id": "src1", "chunk_ids": [],
            "confidence": 0.9, "status": "consumed",
        })

        plan2 = await growth.plan_growth(
            _FakeStore(), sources=["queue"], max_recipes=100, max_new_cards_per_class=2,
            state_dir=tmp_path, backlog_path=empty_backlog,
        )
        assert plan2.targets == []   # nothing pending anymore

    @pytest.mark.asyncio
    async def test_dry_run_does_not_consume_queue(self, fake_templates, empty_backlog, tmp_path):
        # CONFIRMED regression: a dry-run (apply=False, the default) must not permanently consume the
        # growth queue — it only PREVIEWS what would be planned. A subsequent --apply run must still
        # see the entry as pending and be able to author it for real.
        _write_queue_line(tmp_path / "growth_queue.jsonl", {
            "ts": "t", "topology_class": "foo_amp", "matched_from": "node_1",
            "canonical_name": "Foo Amp", "source_id": "src1", "chunk_ids": ["c1"],
            "confidence": 0.9, "status": "pending",
        })
        plan = await growth.plan_growth(
            _FakeStore(), sources=["queue"], max_recipes=100, max_new_cards_per_class=2,
            state_dir=tmp_path, backlog_path=empty_backlog,   # apply defaults False
        )
        assert len(plan.targets) == 2   # dry-run still PLANS targets (preview) ...
        lines = (tmp_path / "growth_queue.jsonl").read_text().splitlines()
        assert len(lines) == 1          # ... but writes nothing to the queue file
        assert json.loads(lines[0])["status"] == "pending"

        plan2 = await growth.plan_growth(
            _FakeStore(), sources=["queue"], max_recipes=100, max_new_cards_per_class=2,
            state_dir=tmp_path, backlog_path=empty_backlog, apply=True,
        )
        assert len(plan2.targets) == 2   # still pending -> still plannable on the real --apply run

    @pytest.mark.asyncio
    async def test_fully_covered_class_marks_skipped_covered_not_consumed(
        self, fake_templates, empty_backlog, tmp_path,
    ):
        # foo_amp is fully novelty-gated -> the queue entry has nothing new to offer. Kind is not part
        # of the coverage key, so one VERIFIED-family card per (metric, knob) pair suffices.
        verified = [{"metric": m, "knob": "k1", "quant_kind": "direction"} for m in ("m1", "m2")]
        store = _FakeStore({"foo_amp": verified})
        _write_queue_line(tmp_path / "growth_queue.jsonl", {
            "ts": "t", "topology_class": "foo_amp", "matched_from": "node_1",
            "canonical_name": "Foo Amp", "source_id": "src1", "chunk_ids": [],
            "confidence": 0.9, "status": "pending",
        })
        plan = await growth.plan_growth(
            store, sources=["queue"], max_recipes=100, max_new_cards_per_class=100,
            state_dir=tmp_path, backlog_path=empty_backlog, apply=True,
        )
        assert plan.targets == []
        lines = (tmp_path / "growth_queue.jsonl").read_text().splitlines()
        assert json.loads(lines[-1])["status"] == "skipped_covered"

    @pytest.mark.asyncio
    async def test_max_recipes_cap_leaves_cut_entries_pending(
        self, fake_templates, empty_backlog, tmp_path,
    ):
        # Two DIFFERENT classes each have a pending queue entry; max_recipes is small enough that only
        # the first class's targets survive the global cap. The second class's entry must stay
        # "pending" (never a consumed CANDIDATE) even though its group's `take` was non-empty at plan
        # time — its targets were cut by the [:max_recipes] truncation, so nothing was ever authored.
        _write_queue_line(tmp_path / "growth_queue.jsonl", {
            "ts": "t", "topology_class": "foo_amp", "matched_from": "node_1",
            "canonical_name": "Foo Amp", "source_id": "src1", "chunk_ids": [],
            "confidence": 0.9, "status": "pending",
        })
        _write_queue_line(tmp_path / "growth_queue.jsonl", {
            "ts": "t", "topology_class": "bar_mirror", "matched_from": "node_2",
            "canonical_name": "Bar Mirror", "source_id": "src2", "chunk_ids": [],
            "confidence": 0.9, "status": "pending",
        })
        plan = await growth.plan_growth(
            _FakeStore(), sources=["queue"], max_recipes=1, max_new_cards_per_class=100,
            state_dir=tmp_path, backlog_path=empty_backlog, apply=True,
        )
        assert len(plan.targets) == 1
        assert plan.targets[0].topology_class == "foo_amp"

        # W-D2 defect 7 fix: neither entry is written "consumed" by plan_growth alone anymore.
        # node_1 (survived the cap) is only a CANDIDATE now (queue_consumed_candidates), pending
        # execute_growth's actual outcome; node_2 (cut by the cap) was never a candidate at all,
        # same as before the fix.
        records = [json.loads(line) for line in (tmp_path / "growth_queue.jsonl").read_text().splitlines()]
        by_node = {r["matched_from"]: r["status"] for r in records}
        assert by_node["node_1"] == "pending"
        assert by_node["node_2"] == "pending"
        assert "foo_amp" in plan.queue_consumed_candidates
        assert "bar_mirror" not in plan.queue_consumed_candidates

    @pytest.mark.asyncio
    async def test_entry_for_unregistered_class_left_pending(
        self, fake_templates, empty_backlog, tmp_path,
    ):
        _write_queue_line(tmp_path / "growth_queue.jsonl", {
            "ts": "t", "topology_class": "not_a_real_class", "matched_from": "node_1",
            "canonical_name": "Something Else", "source_id": "src1", "chunk_ids": [],
            "confidence": 0.9, "status": "pending",
        })
        plan = await growth.plan_growth(
            _FakeStore(), sources=["queue"], max_recipes=100, max_new_cards_per_class=100,
            state_dir=tmp_path, backlog_path=empty_backlog,
        )
        assert plan.targets == []
        lines = (tmp_path / "growth_queue.jsonl").read_text().splitlines()
        assert len(lines) == 1   # untouched — no status line appended

    @pytest.mark.asyncio
    async def test_queue_source_without_state_dir_is_noop(self, fake_templates, empty_backlog):
        plan = await growth.plan_growth(
            _FakeStore(), sources=["queue"], max_recipes=100, max_new_cards_per_class=100,
            state_dir=None, backlog_path=empty_backlog,
        )
        assert plan.targets == []


# =====================================================================================================
# plan_growth — TOPOLOGY_BACKLOG.md-sourced targets + review_lane_pending
# =====================================================================================================


class TestPlanGrowthBacklog:
    @pytest.fixture
    def backlog_path(self, tmp_path):
        p = tmp_path / "backlog.md"
        p.write_text(_BACKLOG_FIXTURE, encoding="utf-8")
        return p

    @pytest.mark.asyncio
    async def test_mapped_backlog_row_generates_targets(self, fake_templates, backlog_path):
        plan = await growth.plan_growth(
            _FakeStore(), sources=["backlog"], max_recipes=100, max_new_cards_per_class=100,
            backlog_path=backlog_path,
        )
        assert any(t.source == "backlog" and t.topology_class == "bar_mirror" for t in plan.targets)
        assert all(t.grounding_chunk_ids == [] for t in plan.targets if t.source == "backlog")
        # foo_amp is [DONE] -> never planned from backlog even though FAKE_TEMPLATES registers it
        assert not any(t.topology_class == "foo_amp" for t in plan.targets)

    @pytest.mark.asyncio
    async def test_backlog_unmapped_counted_in_review_lane(self, fake_templates, backlog_path):
        plan = await growth.plan_growth(
            _FakeStore(), sources=["coverage"], max_recipes=100, max_new_cards_per_class=100,
            backlog_path=backlog_path,
        )
        # review_lane_pending is populated regardless of which `sources` were requested
        assert plan.review_lane_pending["backlog_unmapped"] == 1   # "unmapped_thing"

    @pytest.mark.asyncio
    async def test_template_queue_pending_counted(self, fake_templates, backlog_path, tmp_path):
        _write_queue_line(tmp_path / "template_queue.jsonl", {
            "ts": "t", "canonical_name": "Novel Topology", "node_id": "n1", "layer": 2,
            "confidence": 0.8, "source_id": "s1", "chunk_ids": [], "llm_proposal": None,
        })
        plan = await growth.plan_growth(
            _FakeStore(), sources=["coverage"], max_recipes=1, max_new_cards_per_class=1,
            state_dir=tmp_path, backlog_path=backlog_path,
        )
        assert plan.review_lane_pending["template_queue"] == 1

    @pytest.mark.asyncio
    async def test_priority_queue_then_backlog_then_coverage(self, monkeypatch, backlog_path, tmp_path):
        # A third class untouched by both the queue entry and the backlog fixture, so a genuine
        # "coverage"-sourced target is guaranteed to survive (foo_amp/bar_mirror would otherwise be
        # fully drained by queue/backlog alone at an unbounded per-class cap, leaving nothing for
        # coverage to add and making the ordering assertion vacuous).
        templates = dict(FAKE_TEMPLATES)
        templates["qux_stage"] = {"template_ref": "qux_tmpl", "engine": "ngspice",
                                  "metrics": ["m5"], "knobs": ["k5"]}
        monkeypatch.setattr(growth, "TEMPLATES", templates)
        _write_queue_line(tmp_path / "growth_queue.jsonl", {
            "ts": "t", "topology_class": "foo_amp", "matched_from": "node_1",
            "canonical_name": "Foo Amp", "source_id": "src1", "chunk_ids": ["c1"],
            "confidence": 0.9, "status": "pending",
        })
        plan = await growth.plan_growth(
            _FakeStore(), sources=["coverage", "queue", "backlog"], max_recipes=100,
            max_new_cards_per_class=100, state_dir=tmp_path, backlog_path=backlog_path,
        )
        sources_in_order = [t.source for t in plan.targets]
        first_queue_idx = sources_in_order.index("queue")
        first_backlog_idx = sources_in_order.index("backlog")
        first_coverage_idx = sources_in_order.index("coverage")
        assert first_queue_idx < first_backlog_idx < first_coverage_idx


# =====================================================================================================
# enqueue_candidates
# =====================================================================================================


class TestEnqueueCandidates:
    def test_matched_node_enqueues_growth_queue(self, tmp_path, monkeypatch):
        monkeypatch.setattr(growth, "TEMPLATES", FAKE_TEMPLATES)
        monkeypatch.setitem(growth._CLASS_ALIASES, "foo amp topology", "foo_amp")
        node = NodeProposal(
            proposed_id="topo_1", label=NodeLabel.CIRCUIT_TOPOLOGY, canonical_name="Foo Amp Topology",
            knowledge_layer=2, confidence=0.9, properties={"source_id": "src1"},
            evidence_chunk_ids=["c1", "c2"], reasoning="test",
        )
        counts = growth.enqueue_candidates([node], tmp_path)
        assert counts == {"growth_queue": 1, "template_queue": 0, "skipped": 0}
        lines = (tmp_path / "growth_queue.jsonl").read_text().splitlines()
        rec = json.loads(lines[0])
        assert rec["topology_class"] == "foo_amp"
        assert rec["matched_from"] == "topo_1"
        assert rec["source_id"] == "src1"
        assert rec["chunk_ids"] == ["c1", "c2"]
        assert rec["status"] == "pending"

    def test_unmatched_high_confidence_layer_enqueues_template_queue(self, tmp_path, monkeypatch):
        monkeypatch.setattr(growth, "TEMPLATES", FAKE_TEMPLATES)
        node = {
            "proposed_id": "topo_2", "canonical_name": "Totally Novel Circuit",
            "knowledge_layer": 3, "confidence": 0.85, "properties": {"source_id": "src2"},
            "evidence_chunk_ids": ["c3"],
        }
        counts = growth.enqueue_candidates([node], tmp_path)
        assert counts == {"growth_queue": 0, "template_queue": 1, "skipped": 0}
        rec = json.loads((tmp_path / "template_queue.jsonl").read_text().splitlines()[0])
        assert rec["canonical_name"] == "Totally Novel Circuit"
        assert rec["node_id"] == "topo_2"
        assert rec["layer"] == 3
        assert rec["llm_proposal"] is None

    def test_unmatched_low_confidence_is_skipped_entirely(self, tmp_path, monkeypatch):
        monkeypatch.setattr(growth, "TEMPLATES", FAKE_TEMPLATES)
        node = {"proposed_id": "topo_3", "canonical_name": "Low Confidence Thing",
                "knowledge_layer": 2, "confidence": 0.3, "properties": {}}
        counts = growth.enqueue_candidates([node], tmp_path)
        assert counts == {"growth_queue": 0, "template_queue": 0, "skipped": 1}
        assert not (tmp_path / "growth_queue.jsonl").exists()
        assert not (tmp_path / "template_queue.jsonl").exists()

    def test_unmatched_wrong_layer_is_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(growth, "TEMPLATES", FAKE_TEMPLATES)
        node = {"proposed_id": "topo_4", "canonical_name": "High Confidence But Wrong Layer",
                "knowledge_layer": 0, "confidence": 0.95, "properties": {}}
        counts = growth.enqueue_candidates([node], tmp_path)
        assert counts == {"growth_queue": 0, "template_queue": 0, "skipped": 1}

    def test_append_only_across_calls(self, tmp_path, monkeypatch):
        monkeypatch.setattr(growth, "TEMPLATES", FAKE_TEMPLATES)
        monkeypatch.setitem(growth._CLASS_ALIASES, "foo amp topology", "foo_amp")
        node = {"proposed_id": "n1", "canonical_name": "Foo Amp Topology",
                "knowledge_layer": 2, "confidence": 0.9, "properties": {}}
        growth.enqueue_candidates([node], tmp_path)
        growth.enqueue_candidates([node], tmp_path)
        lines = (tmp_path / "growth_queue.jsonl").read_text().splitlines()
        assert len(lines) == 2

    def test_empty_sequence_is_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr(growth, "TEMPLATES", FAKE_TEMPLATES)
        counts = growth.enqueue_candidates([], tmp_path)
        assert counts == {"growth_queue": 0, "template_queue": 0, "skipped": 0}


# =====================================================================================================
# S3-inc2a INTEGRATION PROOF (spec §4 / ORDER OF WORK item 4): plan_growth against the REAL
# templates.py registry (no monkeypatch — growth.py itself is UNCHANGED, "the coverage enumerator
# reads the registry, that is the whole point") must now plan genuinely NEW cells. A fake store
# simulates the EXACT "coverage-saturated" scenario the spec's Problem statement measured: every
# OLD (metric, knob) pair already VERIFIED-family certified, so BEFORE this task's registry changes
# the uncovered-probe space for these classes was empty (0 targets) — confirmed by computing it
# against the OLD cartesian product below; AFTER, the new metrics/knobs open real, uncovered cells.
# =====================================================================================================


class TestPlanGrowthS3Inc2aIntegration:
    _OLD_MILLER_METRICS = {"gbw_hz", "pm_deg", "av0_db"}
    _OLD_MILLER_KNOBS = {"Cc", "CL"}
    _OLD_OTA5T_METRICS = {"av0_db", "gbw_hz"}
    _OLD_OTA5T_KNOBS = {"CL"}

    def _verified_rows(self, metrics: set, knobs: set) -> list[dict]:
        return [{"metric": m, "knob": k} for m in metrics for k in knobs]

    @pytest.mark.asyncio
    async def test_before_state_the_old_cartesian_was_already_fully_saturated(self, empty_backlog):
        """Sanity check on the FAKE store's own construction: marking every OLD (metric, knob) pair
        VERIFIED-family leaves the OLD cartesian space with ZERO uncovered probes — i.e. before this
        task's registry changes, plan_growth on this class's old surface would have planned nothing
        (skipped_covered == the full old cartesian size), the exact saturation the spec measured."""
        old_rows = self._verified_rows(self._OLD_MILLER_METRICS, self._OLD_MILLER_KNOBS)
        store = _FakeStore(verified_by_class={"miller_ota_2stage_nmos_in": old_rows})
        plan = await growth.plan_growth(
            store, sources=("coverage",), max_recipes=100, max_new_cards_per_class=100, topology_class="miller_ota_2stage_nmos_in",
            backlog_path=empty_backlog,
        )
        old_cartesian_size = len(self._OLD_MILLER_METRICS) * len(self._OLD_MILLER_KNOBS)
        # every NEW target must involve a metric or knob that did NOT exist in the OLD surface —
        # otherwise the "before" state (old-only) would have had something left to plan too.
        assert all(
            t.metric not in self._OLD_MILLER_METRICS or t.knob not in self._OLD_MILLER_KNOBS
            for t in plan.targets
        )
        assert plan.skipped_covered == old_cartesian_size   # old space: fully covered, nothing wasted

    @pytest.mark.asyncio
    async def test_miller_ota_swing_icmr_cells_now_planned(self, empty_backlog):
        old_rows = self._verified_rows(self._OLD_MILLER_METRICS, self._OLD_MILLER_KNOBS)
        store = _FakeStore(verified_by_class={"miller_ota_2stage_nmos_in": old_rows})
        plan = await growth.plan_growth(
            store, sources=("coverage",), max_recipes=100, max_new_cards_per_class=100, topology_class="miller_ota_2stage_nmos_in",
            backlog_path=empty_backlog,
        )
        new_pairs = {(t.metric, t.knob) for t in plan.targets}
        # the headline S3-inc2a cells: new metrics against the (also new) VDD knob.
        assert ("vout_swing_v", "VDD") in new_pairs
        assert ("icmr_lo_v", "VDD") in new_pairs
        assert ("icmr_hi_v", "VDD") in new_pairs
        assert len(plan.targets) > 0
        assert plan.per_class_counts["miller_ota_2stage_nmos_in"] == len(plan.targets)

    @pytest.mark.asyncio
    async def test_ota5t_swing_icmr_cells_now_planned(self, empty_backlog):
        old_rows = self._verified_rows(self._OLD_OTA5T_METRICS, self._OLD_OTA5T_KNOBS)
        store = _FakeStore(verified_by_class={"ota_5t_nmos_in": old_rows})
        plan = await growth.plan_growth(
            store, sources=("coverage",), max_recipes=100, max_new_cards_per_class=100, topology_class="ota_5t_nmos_in",
            backlog_path=empty_backlog,
        )
        new_pairs = {(t.metric, t.knob) for t in plan.targets}
        assert ("vout_swing_v", "VDD") in new_pairs
        assert ("icmr_lo_v", "IREFV") in new_pairs
        assert len(plan.targets) > 0

    @pytest.mark.asyncio
    async def test_ptat_ctat_core_bjt_is_a_brand_new_plannable_class(self, empty_backlog):
        """The FIRST new template since the registry froze at 18 — with NO existing verified probes
        at all (a brand-new class has nothing to be covered by), every (metric, knob) cell plans."""
        store = _FakeStore(verified_by_class={})
        plan = await growth.plan_growth(
            store, sources=("coverage",), max_recipes=100, max_new_cards_per_class=100, topology_class="ptat_ctat_core_bjt",
            backlog_path=empty_backlog,
        )
        new_pairs = {(t.metric, t.knob) for t in plan.targets}
        assert new_pairs == {("iptat_a", "temp"), ("vbe_v", "temp")}
        assert plan.skipped_covered == 0
        assert "ptat_ctat_core_bjt" not in plan.skipped_digital   # ngspice, not iverilog -> AUTO-eligible

    @pytest.mark.asyncio
    async def test_before_and_after_planned_skipped_counts_across_all_three_classes(self, empty_backlog):
        """The closing summary this ORDER OF WORK item asks for: BEFORE (old-surface-only, fully
        saturated) planned=0 across the three touched classes combined; AFTER (real registry, same
        fake 'old fully covered' state) a non-zero, exactly-accounted-for set of new cells plans."""
        store = _FakeStore(verified_by_class={
            "miller_ota_2stage_nmos_in": self._verified_rows(self._OLD_MILLER_METRICS, self._OLD_MILLER_KNOBS),
            "ota_5t_nmos_in": self._verified_rows(self._OLD_OTA5T_METRICS, self._OLD_OTA5T_KNOBS),
            "ptat_ctat_core_bjt": [],
        })
        before_planned = 0   # by construction: the old cartesian for each class is 100% covered above
        after_plan = await growth.plan_growth(
            store, sources=("coverage",), max_recipes=1000, max_new_cards_per_class=1000, backlog_path=empty_backlog,
        )
        touched = {"miller_ota_2stage_nmos_in", "ota_5t_nmos_in", "ptat_ctat_core_bjt"}
        after_targets_touched = [t for t in after_plan.targets if t.topology_class in touched]
        assert before_planned == 0
        assert len(after_targets_touched) > 0
        # exact accounting: miller (6 metrics x 4 knobs - 6 old-covered) + ota5t (5x4 - 5 old-covered... )
        miller_cap = TEMPLATES_FOR_COUNT["miller_ota_2stage_nmos_in"]
        ota5t_cap = TEMPLATES_FOR_COUNT["ota_5t_nmos_in"]
        ptat_cap = TEMPLATES_FOR_COUNT["ptat_ctat_core_bjt"]
        expect_miller = len(miller_cap["metrics"]) * len(miller_cap["knobs"]) - len(
            self._OLD_MILLER_METRICS) * len(self._OLD_MILLER_KNOBS)
        expect_ota5t = len(ota5t_cap["metrics"]) * len(ota5t_cap["knobs"]) - len(
            self._OLD_OTA5T_METRICS) * len(self._OLD_OTA5T_KNOBS)
        expect_ptat = len(ptat_cap["metrics"]) * len(ptat_cap["knobs"])
        by_class = {}
        for t in after_targets_touched:
            by_class[t.topology_class] = by_class.get(t.topology_class, 0) + 1
        assert by_class.get("miller_ota_2stage_nmos_in", 0) == expect_miller
        assert by_class.get("ota_5t_nmos_in", 0) == expect_ota5t
        assert by_class.get("ptat_ctat_core_bjt", 0) == expect_ptat


# =====================================================================================================
# execute_growth — verdict routing, triage, dry-run vs apply, journal/growth_log
# =====================================================================================================


_TT = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)


def _cm_recipe():
    """A recipe on the REAL current_mirror_simple_nmos class/template so run_recipe's real render +
    executor pipeline executes end-to-end against a fake runner."""
    return VerificationRecipe(
        topology_class="current_mirror_simple_nmos", build={"method": "template", "template_ref": "current_mirror_dc"},
        conditions=_TT, sweeps=[{"analysis": "dc", "knob": "Vout", "measure": ["iout_a"]}],
        claim_cards=[ClaimCard(
            id="c1", topology_class="current_mirror_simple_nmos", conditions=_TT,
            mechanism=MechanismClaim(knob="Vout", metric="iout_a", series_ref="vout_iout_a",
                                     quant=QuantTest(kind="direction", sign="+")),
        )],
    )


class _FakeRunner:
    def __init__(self, series=None, avail: bool = True):
        self.calls = 0
        self._avail = avail
        self._series = series if series is not None else [
            (0.4, 1e-5), (0.7, 1.1e-5), (1.0, 1.2e-5), (1.3, 1.25e-5), (1.6, 1.3e-5),
        ]

    def available(self):
        return self._avail

    def measure(self, deck, timeout=300):
        self.calls += 1
        return {"vout": self._series}


class _FakeCorpus:
    def __init__(self):
        self.stored = []

    def store(self, spec):
        spec.spec_id = "sha256:" + str(len(self.stored))
        self.stored.append(spec)
        return f"/fake/{spec.spec_id}"


class _FakeProjector:
    def __init__(self):
        self.projected = []

    async def project(self, spec):
        self.projected.append(spec)
        return {"nodes": 2, "internal_edges": 1, "links_resolved": 1, "links_total": 1}


class _FailingProjector:
    """Like _FakeProjector, but raises on the Nth call (1-indexed) — proves execute_growth's
    per-target apply-block containment (mirrors _FakeProjector's exact return shape otherwise)."""

    def __init__(self, fail_on_call: int = 1):
        self.projected = []
        self.calls = 0
        self.fail_on_call = fail_on_call

    async def project(self, spec):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("neo4j blip")
        self.projected.append(spec)
        return {"nodes": 2, "internal_edges": 1, "links_resolved": 1, "links_total": 1}


class _FakeJournal:
    def __init__(self):
        self.entries = []

    def log(self, op, **kw):
        self.entries.append((op, kw))


def _target(topology_class="current_mirror_simple_nmos", metric="iout_a", knob="Vout", source="coverage"):
    return growth.AuthoringTarget(topology_class=topology_class, metric=metric, knob=knob,
                                  kind_hint="direction", source=source)


@pytest.fixture
def fake_ngspice(monkeypatch):
    runner = _FakeRunner()
    monkeypatch.setitem(engines_mod.ENGINES, "ngspice", engines_mod.EngineSpec("ngspice", runner))
    monkeypatch.setattr(growth, "ENGINES", engines_mod.ENGINES)
    return runner


def _mock_author(monkeypatch, recipe_or_exc):
    async def _fake(*, topology_class, source_text, model_chain, resilience_config, auth_refresh, focus=None):
        if isinstance(recipe_or_exc, Exception):
            raise recipe_or_exc
        return recipe_or_exc
    monkeypatch.setattr(growth, "author_recipe", _fake)


class TestExecuteGrowthRouting:
    @pytest.mark.asyncio
    async def test_verified_projects_on_apply(self, fake_ngspice, monkeypatch, tmp_path):
        _mock_author(monkeypatch, _cm_recipe())
        corpus, projector, journal = _FakeCorpus(), _FakeProjector(), _FakeJournal()
        deps = growth.GrowthDeps(store=None, model_chain=[object()], resilience_config=object(),
                                 corpus=corpus, projector=projector, journal=journal, state_dir=tmp_path)
        plan = growth.GrowthPlan(targets=[_target()])
        report = await growth.execute_growth(deps, plan, apply=True)

        assert report.authored == 1
        assert report.simulated == 1
        assert report.projected == 1
        assert report.triage == []
        assert len(corpus.stored) == 1
        assert len(projector.projected) == 1
        assert len(journal.entries) == 1
        assert journal.entries[0][0] == "grow_executable"

    @pytest.mark.asyncio
    async def test_apply_persistence_failure_is_triaged_and_does_not_abort_the_run(
        self, fake_ngspice, monkeypatch, tmp_path,
    ):
        """The growth.py:786 defect: execute_growth's per-target apply block (corpus.store +
        projector.project) used to have zero error containment — one bad target's Neo4j hiccup would
        propagate out of execute_growth entirely, so NO GrowthReport was ever built and every
        remaining target in the plan was silently never attempted. Fixed: per-target try/except that
        records the failure into report.triage and continues to the next target."""
        _mock_author(monkeypatch, _cm_recipe())
        corpus = _FakeCorpus()
        projector = _FailingProjector(fail_on_call=1)   # target #1's projector.project raises
        journal = _FakeJournal()
        deps = growth.GrowthDeps(store=None, model_chain=[object()], resilience_config=object(),
                                 corpus=corpus, projector=projector, journal=journal, state_dir=tmp_path)
        plan = growth.GrowthPlan(targets=[_target(), _target()])   # two targets

        report = await growth.execute_growth(deps, plan, apply=True)

        # the routing decision is counted for BOTH targets regardless of the apply-write outcome
        # (matches execute_growth's own documented "projected counts the routing decision...
        # regardless of apply" contract) -- the run was never aborted by target #1's failure.
        assert report.authored == 2
        assert report.simulated == 2
        assert report.projected == 2
        # target #1's apply failure is recorded, not silently swallowed
        assert len(report.triage) == 1
        assert report.triage[0]["stage"] == "apply"
        assert "neo4j blip" in report.triage[0]["error"]
        # honestly reports that corpus.store DID complete before projector.project raised -- the
        # documented orphaned-specimen risk (EPISTEMOLOGY.md Known issues #2) is surfaced, not hidden
        assert report.triage[0]["corpus_stored"] is True
        # both targets' corpus.store ran (only target #1's projector.project raised); only target #2
        # made it all the way through to a successful project()/journal.log
        assert len(corpus.stored) == 2
        assert len(projector.projected) == 1
        assert len(journal.entries) == 1
        # the report still gets built and persisted -- not lost, unlike before this fix
        log_path = tmp_path / "growth_log.jsonl"
        assert log_path.is_file()

    @pytest.mark.asyncio
    async def test_refuted_also_projects(self, fake_ngspice, monkeypatch):
        # sign="-" but the fake series is monotonically INCREASING with Vout -> oracle REFUTES it.
        recipe = _cm_recipe()
        recipe.claim_cards[0].mechanism.quant = QuantTest(kind="direction", sign="-")
        _mock_author(monkeypatch, recipe)
        corpus, projector = _FakeCorpus(), _FakeProjector()
        deps = growth.GrowthDeps(store=None, model_chain=[object()], resilience_config=object(),
                                 corpus=corpus, projector=projector)
        plan = growth.GrowthPlan(targets=[_target()])
        report = await growth.execute_growth(deps, plan, apply=True)

        assert report.triage == []
        assert report.projected == 1
        assert len(corpus.stored) == 1   # REFUTED is knowledge — the Pelgrom precedent, still projects

    @pytest.mark.asyncio
    async def test_flagged_never_projects(self, fake_ngspice, monkeypatch):
        fake_ngspice._series = []   # empty series -> executor FLAGs (C1: no valid RDATA)
        _mock_author(monkeypatch, _cm_recipe())
        corpus, projector = _FakeCorpus(), _FakeProjector()
        deps = growth.GrowthDeps(store=None, model_chain=[object()], resilience_config=object(),
                                 corpus=corpus, projector=projector)
        plan = growth.GrowthPlan(targets=[_target()])
        report = await growth.execute_growth(deps, plan, apply=True)

        assert report.projected == 0
        assert len(report.triage) == 1
        assert report.triage[0]["stage"] == "verdict"
        assert report.triage[0]["verdicts"]["c1"] == "FLAGGED"
        assert corpus.stored == []
        assert projector.projected == []

    @pytest.mark.asyncio
    async def test_author_failure_goes_to_triage(self, fake_ngspice, monkeypatch):
        _mock_author(monkeypatch, RuntimeError("boom"))
        deps = growth.GrowthDeps(store=None, model_chain=[object()], resilience_config=object())
        plan = growth.GrowthPlan(targets=[_target()])
        report = await growth.execute_growth(deps, plan, apply=True)

        assert report.authored == 0
        assert report.simulated == 0
        assert len(report.triage) == 1
        assert report.triage[0]["stage"] == "author"
        assert "boom" in report.triage[0]["error"]

    @pytest.mark.asyncio
    async def test_engine_unavailable_goes_to_triage(self, fake_ngspice, monkeypatch):
        fake_ngspice._avail = False
        _mock_author(monkeypatch, _cm_recipe())
        deps = growth.GrowthDeps(store=None, model_chain=[object()], resilience_config=object())
        plan = growth.GrowthPlan(targets=[_target()])
        report = await growth.execute_growth(deps, plan, apply=True)

        assert report.authored == 1
        assert report.simulated == 0
        assert len(report.triage) == 1
        assert report.triage[0]["stage"] == "engine"

    @pytest.mark.asyncio
    async def test_dry_run_previews_but_writes_nothing(self, fake_ngspice, monkeypatch):
        _mock_author(monkeypatch, _cm_recipe())
        corpus, projector, journal = _FakeCorpus(), _FakeProjector(), _FakeJournal()
        deps = growth.GrowthDeps(store=None, model_chain=[object()], resilience_config=object(),
                                 corpus=corpus, projector=projector, journal=journal)
        plan = growth.GrowthPlan(targets=[_target()])
        report = await growth.execute_growth(deps, plan, apply=False)

        assert report.projected == 1        # previews the routing decision
        assert report.apply is False
        assert corpus.stored == []          # but never actually writes
        assert projector.projected == []
        assert journal.entries == []

    @pytest.mark.asyncio
    async def test_growth_log_appended(self, fake_ngspice, monkeypatch, tmp_path):
        _mock_author(monkeypatch, _cm_recipe())
        deps = growth.GrowthDeps(store=None, model_chain=[object()], resilience_config=object(),
                                 state_dir=tmp_path)
        plan = growth.GrowthPlan(targets=[_target()])
        await growth.execute_growth(deps, plan, apply=False)

        log_path = tmp_path / "growth_log.jsonl"
        assert log_path.is_file()
        rec = json.loads(log_path.read_text().splitlines()[0])
        assert rec["planned"] == 1
        assert rec["projected"] == 1

    @pytest.mark.asyncio
    async def test_grounding_text_fetched_when_chunk_ids_present(self, fake_ngspice, monkeypatch):
        captured = {}

        async def _fake(*, topology_class, source_text, model_chain, resilience_config, auth_refresh, focus=None):
            captured["source_text"] = source_text
            return _cm_recipe()
        monkeypatch.setattr(growth, "author_recipe", _fake)

        class _StoreWithChunks:
            async def run_read_query(self, query, params=None):
                return [{"chunk_id": "c1", "raw_text_hash": "hash1"}]

        class _Vault:
            def get_text(self, h):
                return "grounding text from source" if h == "hash1" else None

        deps = growth.GrowthDeps(store=_StoreWithChunks(), model_chain=[object()],
                                 resilience_config=object(), vault=_Vault())
        target = growth.AuthoringTarget(topology_class="current_mirror_simple_nmos", metric="iout_a",
                                        knob="Vout", kind_hint="direction", source="queue",
                                        grounding_chunk_ids=["c1"])
        plan = growth.GrowthPlan(targets=[target])
        await growth.execute_growth(deps, plan, apply=False)
        assert captured["source_text"] == "grounding text from source"

    @pytest.mark.asyncio
    async def test_no_grounding_chunk_ids_means_empty_source_text(self, fake_ngspice, monkeypatch):
        captured = {}

        async def _fake(*, topology_class, source_text, model_chain, resilience_config, auth_refresh, focus=None):
            captured["source_text"] = source_text
            return _cm_recipe()
        monkeypatch.setattr(growth, "author_recipe", _fake)

        deps = growth.GrowthDeps(store=None, model_chain=[object()], resilience_config=object())
        plan = growth.GrowthPlan(targets=[_target(source="coverage")])
        await growth.execute_growth(deps, plan, apply=False)
        assert captured["source_text"] == ""

    @pytest.mark.asyncio
    async def test_focus_passed_through_to_author_recipe(self, fake_ngspice, monkeypatch):
        captured = {}

        async def _fake(*, topology_class, source_text, model_chain, resilience_config, auth_refresh, focus=None):
            captured["focus"] = focus
            return _cm_recipe()
        monkeypatch.setattr(growth, "author_recipe", _fake)

        deps = growth.GrowthDeps(store=None, model_chain=[object()], resilience_config=object())
        target = growth.AuthoringTarget(topology_class="current_mirror_simple_nmos", metric="iout_a",
                                        knob="Vout", kind_hint=None, source="coverage")
        plan = growth.GrowthPlan(targets=[target])
        await growth.execute_growth(deps, plan, apply=False)
        assert captured["focus"] == {"metric": "iout_a", "knob": "Vout", "kind_hint": None}

    @pytest.mark.asyncio
    async def test_author_failure_triage_carries_trace(self, fake_ngspice, monkeypatch):
        def _raise_in_stage(*a, **k):
            raise RuntimeError("kaboom")

        async def _fake(**kwargs):
            _raise_in_stage()
        monkeypatch.setattr(growth, "author_recipe", _fake)

        deps = growth.GrowthDeps(store=None, model_chain=[object()], resilience_config=object())
        plan = growth.GrowthPlan(targets=[_target()])
        report = await growth.execute_growth(deps, plan, apply=True)

        assert len(report.triage) == 1
        assert report.triage[0]["stage"] == "author"
        assert "trace" in report.triage[0]
        assert "_raise_in_stage" in report.triage[0]["trace"]

    @pytest.mark.asyncio
    async def test_simulate_failure_triage_carries_trace(self, fake_ngspice, monkeypatch):
        def _boom_in_run_recipe(*a, **k):
            raise ValueError("sim exploded")

        _mock_author(monkeypatch, _cm_recipe())
        monkeypatch.setattr(growth, "run_recipe", _boom_in_run_recipe)

        deps = growth.GrowthDeps(store=None, model_chain=[object()], resilience_config=object())
        plan = growth.GrowthPlan(targets=[_target()])
        report = await growth.execute_growth(deps, plan, apply=True)

        assert len(report.triage) == 1
        assert report.triage[0]["stage"] == "simulate"
        assert "trace" in report.triage[0]
        assert "_boom_in_run_recipe" in report.triage[0]["trace"]


# =====================================================================================================
# execute_growth — growth_queue.jsonl "consumed" write (W-D2 defect 7). plan_growth used to mark a
# queue entry "consumed" purely for surviving the max_recipes cap, BEFORE execute_growth ever ran --
# so any author/simulate/engine/verdict/apply-write failure permanently and silently dropped the
# ingest-sourced grounding for that topology (routing FLAGGED/REJECTED verdicts are documented as a
# routine, expected outcome, not exceptional). Fixed: the write is deferred to execute_growth and
# only fires for a class that had a QUEUE-sourced target actually get projected THIS run; any of the
# above failures leaves the entry untouched (still "pending" from whenever it was enqueued), so it is
# re-offered on the next plan_growth run instead of being lost.
# =====================================================================================================


class TestExecuteGrowthQueueConsumption:
    _CANDIDATE_ENTRIES = [{
        "ts": "t", "topology_class": "current_mirror_simple_nmos", "matched_from": "node_1",
        "canonical_name": "Current Mirror", "source_id": "src1", "chunk_ids": [],
        "confidence": 0.9, "status": "pending",
    }]

    def _plan(self, targets):
        return growth.GrowthPlan(
            targets=targets,
            queue_consumed_candidates={"current_mirror_simple_nmos": list(self._CANDIDATE_ENTRIES)},
        )

    @pytest.mark.asyncio
    async def test_marks_consumed_only_when_queue_sourced_target_actually_projects(
        self, fake_ngspice, monkeypatch, tmp_path,
    ):
        _mock_author(monkeypatch, _cm_recipe())
        corpus, projector, journal = _FakeCorpus(), _FakeProjector(), _FakeJournal()
        deps = growth.GrowthDeps(store=None, model_chain=[object()], resilience_config=object(),
                                 corpus=corpus, projector=projector, journal=journal, state_dir=tmp_path)
        plan = self._plan([_target(source="queue")])

        report = await growth.execute_growth(deps, plan, apply=True)

        assert report.projected == 1
        lines = (tmp_path / "growth_queue.jsonl").read_text().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["matched_from"] == "node_1"
        assert rec["status"] == "consumed"

    @pytest.mark.asyncio
    async def test_author_failure_leaves_queue_entry_pending_not_consumed(
        self, fake_ngspice, monkeypatch, tmp_path,
    ):
        """DEFECT REGRESSION: before the fix, plan_growth would have already marked this entry
        'consumed' the moment it survived the max_recipes cap, BEFORE execute_growth ever ran — so
        this author_recipe failure would have permanently and silently dropped the ingest-sourced
        grounding. Fixed: nothing is written for this class, so the entry stays whatever plan_growth
        itself left it as (pending) and is re-offered on the next run."""
        _mock_author(monkeypatch, RuntimeError("boom"))
        deps = growth.GrowthDeps(store=None, model_chain=[object()], resilience_config=object(),
                                 state_dir=tmp_path)
        plan = self._plan([_target(source="queue")])

        report = await growth.execute_growth(deps, plan, apply=True)

        assert report.authored == 0
        assert len(report.triage) == 1
        assert report.triage[0]["stage"] == "author"
        assert not (tmp_path / "growth_queue.jsonl").exists()   # nothing written -> stays "pending"

    @pytest.mark.asyncio
    async def test_flagged_verdict_leaves_queue_entry_pending_not_consumed(
        self, fake_ngspice, monkeypatch, tmp_path,
    ):
        """FLAGGED/REJECTED is a routine, expected outcome per execute_growth's own docstring — not
        exceptional — but must still leave the queue entry re-offerable, not consumed."""
        fake_ngspice._series = []   # empty series -> oracle FLAGs (no valid RDATA)
        _mock_author(monkeypatch, _cm_recipe())
        deps = growth.GrowthDeps(store=None, model_chain=[object()], resilience_config=object(),
                                 state_dir=tmp_path)
        plan = self._plan([_target(source="queue")])

        report = await growth.execute_growth(deps, plan, apply=True)

        assert report.projected == 0
        assert report.triage[0]["stage"] == "verdict"
        assert not (tmp_path / "growth_queue.jsonl").exists()

    @pytest.mark.asyncio
    async def test_apply_persistence_failure_leaves_queue_entry_pending_not_consumed(
        self, fake_ngspice, monkeypatch, tmp_path,
    ):
        """The verdict itself succeeded (VERIFIED-family), but apply-persistence (corpus.store /
        projector.project / journal.log) failed — must NOT be marked consumed either: the defect's
        own evidence names 'write failure' explicitly alongside author/simulate/engine/verdict as a
        case that must not silently drop the ingest-sourced grounding."""
        _mock_author(monkeypatch, _cm_recipe())
        corpus = _FakeCorpus()
        projector = _FailingProjector(fail_on_call=1)
        deps = growth.GrowthDeps(store=None, model_chain=[object()], resilience_config=object(),
                                 corpus=corpus, projector=projector, state_dir=tmp_path)
        plan = self._plan([_target(source="queue")])

        report = await growth.execute_growth(deps, plan, apply=True)

        assert report.projected == 1                 # the routing decision still counts (unchanged)
        assert report.triage[0]["stage"] == "apply"   # but persistence failed
        assert not (tmp_path / "growth_queue.jsonl").exists()   # -> still not consumed

    @pytest.mark.asyncio
    async def test_dry_run_never_writes_consumed_even_on_success(
        self, fake_ngspice, monkeypatch, tmp_path,
    ):
        """Spec §3 gate 5 (dry-run writes nothing) extends to the deferred consumed-write too."""
        _mock_author(monkeypatch, _cm_recipe())
        corpus, projector = _FakeCorpus(), _FakeProjector()
        deps = growth.GrowthDeps(store=None, model_chain=[object()], resilience_config=object(),
                                 corpus=corpus, projector=projector, state_dir=tmp_path)
        plan = self._plan([_target(source="queue")])

        report = await growth.execute_growth(deps, plan, apply=False)

        assert report.projected == 1   # preview still shows what WOULD be certified
        assert not (tmp_path / "growth_queue.jsonl").exists()

    @pytest.mark.asyncio
    async def test_coverage_sourced_success_does_not_falsely_consume_unrelated_queue_entry(
        self, fake_ngspice, monkeypatch, tmp_path,
    ):
        """Precision check: per_class_projected counts a projected target from ANY source, but the
        queue "consumed" write must be gated on a QUEUE-sourced target SPECIFICALLY projecting — a
        coverage-sourced target for the same class succeeding must not falsely consume a different,
        still-failing queue-sourced target for that same class."""
        call_count = {"n": 0}

        async def _fake_author(*, topology_class, source_text, model_chain, resilience_config,
                               auth_refresh, focus=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("queue-sourced target's author failed")
            return _cm_recipe()
        monkeypatch.setattr(growth, "author_recipe", _fake_author)

        queue_target = _target(source="queue")         # processed first -> author fails
        coverage_target = _target(source="coverage")   # processed second -> succeeds
        corpus, projector = _FakeCorpus(), _FakeProjector()
        deps = growth.GrowthDeps(store=None, model_chain=[object()], resilience_config=object(),
                                 corpus=corpus, projector=projector, state_dir=tmp_path)
        plan = self._plan([queue_target, coverage_target])

        report = await growth.execute_growth(deps, plan, apply=True)

        assert report.authored == 1             # only the coverage-sourced target authored
        assert report.projected == 1            # only the coverage-sourced target certified
        assert len(report.triage) == 1
        assert report.triage[0]["source"] == "queue"
        assert not (tmp_path / "growth_queue.jsonl").exists()   # queue entry NOT falsely consumed


# =====================================================================================================
# BrainAgent.grow_executable — end-to-end wiring (mirrors test_executable_project_all.py conventions)
# =====================================================================================================


class _MockGraph:
    def __init__(self):
        self.queries = []
        self.writes = []

    async def run_read_query(self, query, params=None):
        self.queries.append((query, params))
        return []

    async def write_batch(self, nodes=None, updates=None, edges=None):
        self.writes.append({"nodes": nodes or [], "edges": edges or []})


@pytest.fixture
def agent(tmp_path):
    cfg = load_config()
    cfg.openclaw.state_dir = str(tmp_path / "state")
    cfg.executable.corpus_dir = str(tmp_path / "corpus")
    a = BrainAgent(cfg)
    a._started = True
    a._journal = _FakeJournal()
    a._graph = _MockGraph()
    a._llm_provider = type("FakeProvider", (), {"get_chain": lambda self, stage: [object()]})()
    return a


class TestBrainAgentGrowExecutable:
    @pytest.mark.asyncio
    async def test_dry_run_writes_nothing(self, agent, fake_ngspice, monkeypatch, tmp_path):
        _mock_author(monkeypatch, _cm_recipe())
        result = await agent.grow_executable(
            sources=["coverage"], max_recipes=1, max_per_class=1,
            topology_class="current_mirror_simple_nmos", apply=False,
        )
        assert result["apply"] is False
        assert result["projected"] == 1
        assert not (tmp_path / "corpus").exists()
        assert agent._graph.writes == []

    @pytest.mark.asyncio
    async def test_apply_persists_to_corpus_and_graph(self, agent, fake_ngspice, monkeypatch, tmp_path):
        _mock_author(monkeypatch, _cm_recipe())
        result = await agent.grow_executable(
            sources=["coverage"], max_recipes=1, max_per_class=1,
            topology_class="current_mirror_simple_nmos", apply=True,
        )
        assert result["apply"] is True
        assert result["projected"] == 1
        corpus_dir = tmp_path / "corpus"
        assert corpus_dir.is_dir()
        corpus = SpecimenCorpus(str(corpus_dir))
        assert "current_mirror_simple_nmos" in corpus.list_classes()
        assert len(agent._graph.writes) == 1
        assert len(agent._journal.entries) == 1


@pytest.mark.asyncio
async def test_execute_growth_dry_run_report_carries_certified_card_preview(fake_ngspice, monkeypatch):
    """Spec §8 step 2: a DRY run must show WHAT would be certified (per-card preview),
    not just the projected count — the owner-review surface before any --apply."""
    _mock_author(monkeypatch, _cm_recipe())
    deps = growth.GrowthDeps(store=None, model_chain=[object()], resilience_config=object())
    plan = growth.GrowthPlan(targets=[_target()])
    report = await growth.execute_growth(deps, plan, apply=False)

    assert report.projected == 1
    assert len(report.certified) == 1
    entry = report.certified[0]
    assert entry["topology_class"] == plan.targets[0].topology_class
    card = entry["cards"][0]
    assert set(card) == {"id", "metric", "knob", "kind", "verdict"}
    assert card["verdict"] is not None
    assert report.to_dict()["certified"] == report.certified
