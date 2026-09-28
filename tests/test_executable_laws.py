"""Tests for the law-tier Regularity projector (knowledge/executable/laws.py) —
docs/superpowers/specs/2026-07-04-law-tier-graph-representation.md §3/§4/§6.

Unit tests (no DB): JSONL -> LawRecord extraction (incl. malformed rows, unknown claim_ids),
status rules (law / 2-member / divergence / insufficient / unanimous-non-verified), law_id
stability, and idempotent MERGE against a stateful mocked store (NO-PHANTOM linking, demotion,
history threading). A gated live-acceptance test at the bottom exercises the SAME projector
against the real E1/E1b JSONL over a live Neo4j (explicit opt-in required).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from tests.conftest import require_live_graph

from openclaw_brain.knowledge.executable.laws import (
    LawRecord,
    build_law_records,
    classify_status,
    compute_law_id,
    load_run_rows,
    project_law_records,
    resolve_status,
)
from openclaw_brain.knowledge.executable.oracle import _loglog
from openclaw_brain.knowledge.graph.schema import NodeLabel, RelType

REPO_ROOT = Path(__file__).resolve().parent.parent
E1_JSONL = REPO_ROOT / "experiments" / "e1_cross_pdk_raw.jsonl"
E1B_JSONL = REPO_ROOT / "experiments" / "e1b_statistical_cross_pdk_raw.jsonl"
FULL_JSONL = REPO_ROOT / "experiments" / "e_rollout_full_registry_raw.jsonl"


def _write_jsonl(tmp_path, name, rows):
    path = tmp_path / name
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _run_row(topology_class, pdk, verdicts, notes=None, canonical=None, ok=True):
    return {
        "topology_class": topology_class, "pdk": pdk, "ok": ok, "wall_time_s": 0.1, "error": None,
        "verdicts": verdicts, "verdict_notes": notes or {}, "canonical": canonical or {},
    }


# ── load_run_rows ──


def test_load_run_rows_parses_well_formed_rows(tmp_path):
    path = _write_jsonl(tmp_path, "ok.jsonl", [
        _run_row("current_mirror_simple_nmos", "sky130A", {"cm_iout": "VERIFIED"}),
        _run_row("current_mirror_simple_nmos", "gf180mcuD", {"cm_iout": "VERIFIED"}),
    ])
    result = load_run_rows(path)
    assert result.errors == []
    assert len(result.rows) == 2
    assert result.rows[0].pdk == "sky130A"
    assert result.rows[0].verdicts == {"cm_iout": "VERIFIED"}


def test_load_run_rows_skips_malformed_json_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(_run_row("current_mirror_simple_nmos", "sky130A", {"cm_iout": "VERIFIED"})) + "\n"
        + "{not valid json\n"
        + json.dumps(_run_row("current_mirror_simple_nmos", "gf180mcuD", {"cm_iout": "VERIFIED"})) + "\n"
    )
    result = load_run_rows(path)
    assert len(result.rows) == 2       # the one bad line is skipped, not fatal
    assert len(result.errors) == 1
    assert result.errors[0]["line"] == 2
    assert "malformed JSON" in result.errors[0]["error"]


def test_load_run_rows_skips_row_missing_required_keys(tmp_path):
    path = tmp_path / "missing.jsonl"
    path.write_text(
        json.dumps({"topology_class": "x", "ok": True}) + "\n"   # missing "pdk"
        + json.dumps(_run_row("current_mirror_simple_nmos", "sky130A", {"cm_iout": "VERIFIED"})) + "\n"
    )
    result = load_run_rows(path)
    assert len(result.rows) == 1
    assert len(result.errors) == 1
    assert "pdk" in result.errors[0]["error"]


def test_load_run_rows_skips_non_object_row_and_blank_lines(tmp_path):
    path = tmp_path / "weird.jsonl"
    path.write_text("\n[1, 2, 3]\n\n"
                     + json.dumps(_run_row("current_mirror_simple_nmos", "sky130A", {"cm_iout": "VERIFIED"})) + "\n")
    result = load_run_rows(path)
    assert len(result.rows) == 1
    assert len(result.errors) == 1
    assert "not a JSON object" in result.errors[0]["error"]


# ── classify_status (spec §4 status rules) ──


def test_classify_status_law_on_three_agreeing():
    r = classify_status({"sky130A": "VERIFIED", "gf180mcuD": "VERIFIED", "ihp-sg13g2": "VERIFIED"})
    assert r.status == "law" and r.reason == "law"
    assert r.agreeing_pdks == ["gf180mcuD", "ihp-sg13g2", "sky130A"]
    assert r.disagreeing_pdks == []


def test_classify_status_caveat_affirms_but_verified_negative_does_not():
    # VERIFIED_WITH_CAVEAT affirms the claim; VERIFIED_NEGATIVE is a correct measurement of the
    # OPPOSITE outcome — mixing them is process divergence (outcome differs by foundry), not
    # unanimity. Guards against re-adding VERIFIED_NEGATIVE to _AFFIRMING_FAMILY.
    r = classify_status({"sky130A": "VERIFIED", "gf180mcuD": "VERIFIED_WITH_CAVEAT",
                          "ihp-sg13g2": "VERIFIED_NEGATIVE"})
    assert r.status == "process_scoped" and r.reason == "divergent"
    assert "ihp-sg13g2=VERIFIED_NEGATIVE" in r.note
    assert r.agreeing_pdks == ["gf180mcuD", "sky130A"]
    assert r.disagreeing_pdks == ["ihp-sg13g2"]


def test_classify_status_unanimous_verified_negative_is_replicated_negative_not_law():
    # All three PDKs correctly measure the effect ABSENT: replicated knowledge, but of the
    # negative — routes to the "does NOT hold" statement path, never a positively-phrased law.
    r = classify_status({"sky130A": "VERIFIED_NEGATIVE", "gf180mcuD": "VERIFIED_NEGATIVE",
                          "ihp-sg13g2": "VERIFIED_NEGATIVE"})
    assert r.status == "process_scoped" and r.reason == "unanimous_non_verified"
    assert "not a law" in r.note


def test_classify_status_two_member_never_over_badges():
    r = classify_status({"sky130A": "VERIFIED", "gf180mcuD": "VERIFIED"})
    assert r.status == "process_scoped" and r.reason == "two_member"
    assert "never over-badge on two" in r.note


def test_classify_status_divergence_names_the_axis():
    r = classify_status({"sky130A": "VERIFIED", "gf180mcuD": "VERIFIED", "ihp-sg13g2": "REFUTED"})
    assert r.status == "process_scoped" and r.reason == "divergent"
    assert "ihp-sg13g2=REFUTED" in r.note
    assert r.agreeing_pdks == ["gf180mcuD", "sky130A"]
    assert r.disagreeing_pdks == ["ihp-sg13g2"]


def test_classify_status_insufficient_replication_zero_or_one_member():
    r0 = classify_status({})
    assert r0.status == "process_scoped" and r0.reason == "insufficient"
    r1 = classify_status({"sky130A": "VERIFIED"})
    assert r1.status == "process_scoped" and r1.reason == "insufficient"


def test_classify_status_unanimous_non_verified():
    r = classify_status({"sky130A": "REFUTED", "gf180mcuD": "REFUTED"})
    assert r.status == "process_scoped" and r.reason == "unanimous_non_verified"
    assert "not a law" in r.note


# ── resolve_status (demotion overlay, spec §4 `demoted`) ──


def test_resolve_status_never_law_stays_process_scoped_even_if_divergent():
    raw = classify_status({"sky130A": "VERIFIED", "gf180mcuD": "REFUTED"})
    final, is_new = resolve_status(raw, existing_status=None)
    assert final == "process_scoped" and is_new is False
    final2, is_new2 = resolve_status(raw, existing_status="process_scoped")
    assert final2 == "process_scoped" and is_new2 is False


def test_resolve_status_demotes_a_previously_law_node_on_new_divergence():
    raw = classify_status({"sky130A": "VERIFIED", "gf180mcuD": "VERIFIED", "ihp-sg13g2": "REFUTED"})
    final, is_new = resolve_status(raw, existing_status="law")
    assert final == "demoted" and is_new is True


def test_resolve_status_stays_demoted_without_re_flagging_history():
    raw = classify_status({"sky130A": "VERIFIED", "gf180mcuD": "VERIFIED", "ihp-sg13g2": "REFUTED"})
    final, is_new = resolve_status(raw, existing_status="demoted")
    assert final == "demoted" and is_new is False   # already demoted -> no NEW transition


def test_resolve_status_recovers_to_law_when_agreement_returns():
    raw = classify_status({"sky130A": "VERIFIED", "gf180mcuD": "VERIFIED", "ihp-sg13g2": "VERIFIED"})
    final, is_new = resolve_status(raw, existing_status="demoted")
    assert final == "law" and is_new is False


# ── compute_law_id ──


def test_compute_law_id_stable_and_order_sensitive():
    a = compute_law_id("cm", "iout_a", "Vout", "direction")
    b = compute_law_id("cm", "iout_a", "Vout", "direction")
    assert a == b
    c = compute_law_id("cm", "iout_a", "Vout", "invariance")
    assert a != c
    assert len(a) == 40   # sha1 hex digest


# ── build_law_records (pure extraction) ──


def test_build_law_records_direction_claim_end_to_end(tmp_path):
    path = _write_jsonl(tmp_path, "cm.jsonl", [
        _run_row("current_mirror_simple_nmos", "sky130A", {"cm_iout": "VERIFIED"}, {"cm_iout": "trend +"}),
        _run_row("current_mirror_simple_nmos", "gf180mcuD", {"cm_iout": "VERIFIED"}),
        _run_row("current_mirror_simple_nmos", "ihp-sg13g2", {"cm_iout": "VERIFIED"}),
    ])
    records, gaps = build_law_records({str(path): load_run_rows(path)})
    assert gaps["unknown_claim_ids"] == []
    assert len(records) == 1
    rec = records[0]
    assert rec.topology_class == "current_mirror_simple_nmos"
    assert rec.metric == "iout_a" and rec.knob == "Vout" and rec.quant_kind == "direction"
    assert rec.status == "law"
    assert rec.pdks == ["gf180mcuD", "ihp-sg13g2", "sky130A"]
    assert rec.law_id == compute_law_id("current_mirror_simple_nmos", "iout_a", "Vout", "direction")
    assert "increases with Vout" in rec.statement
    assert rec.member_summary["sky130A"]["note"] == "trend +"


def test_build_law_records_unknown_claim_id_reported_not_invented(tmp_path):
    path = _write_jsonl(tmp_path, "unk.jsonl", [
        _run_row("current_mirror_simple_nmos", "sky130A", {"totally_unknown_claim": "VERIFIED"}),
    ])
    records, gaps = build_law_records({str(path): load_run_rows(path)})
    assert records == []
    assert gaps["unknown_claim_ids"] == ["totally_unknown_claim"]


def test_build_law_records_pelgrom_elasticity_carries_per_pdk_exponent_never_in_statement(tmp_path):
    series_sky = [[0.25, 0.00982], [1.0, 0.00541], [4.0, 0.00273]]
    series_gf = [[0.25, 0.01032], [1.0, 0.00493], [4.0, 0.00270]]
    series_ihp = [[0.25, 0.00528], [1.0, 0.00295], [4.0, 0.00145]]
    path = _write_jsonl(tmp_path, "pelgrom.jsonl", [
        _run_row("ota_5t_nmos_in", "sky130A", {"ota5t_pelgrom": "VERIFIED"}, {},
                 {"vos_a_vos": series_sky}),
        _run_row("ota_5t_nmos_in", "gf180mcuD", {"ota5t_pelgrom": "VERIFIED"}, {},
                 {"vos_a_vos": series_gf}),
        _run_row("ota_5t_nmos_in", "ihp-sg13g2", {"ota5t_pelgrom": "VERIFIED"}, {},
                 {"vos_a_vos": series_ihp}),
    ])
    records, gaps = build_law_records({str(path): load_run_rows(path)})
    assert len(records) == 1
    rec = records[0]
    assert rec.quant_kind == "elasticity"
    assert rec.status == "law"
    exps = {pdk: rec.member_summary[pdk]["fitted_exponent"] for pdk in rec.pdks}
    expected_sky = _loglog(series_sky)[0]
    assert exps["sky130A"] == pytest.approx(expected_sky)
    assert all(-0.6 <= v <= -0.2 for v in exps.values())   # in-band, matches E1b's registered band
    # statement is shape-level only — the measured exponent value never leaks into it
    for v in exps.values():
        assert f"{v:.3f}" not in rec.statement
    assert "power-law" in rec.statement


def test_build_law_records_pelgrom_exponent_none_for_degenerate_series(tmp_path):
    path = _write_jsonl(tmp_path, "degenerate.jsonl", [
        _run_row("ota_5t_nmos_in", "sky130A", {"ota5t_pelgrom": "VERIFIED"}, {}, {"vos_a_vos": [[1.0, 0.005]]}),
    ])
    records, _ = build_law_records({str(path): load_run_rows(path)})
    assert records[0].member_summary["sky130A"]["fitted_exponent"] is None


def test_build_law_records_malformed_row_does_not_crash_the_file(tmp_path):
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        json.dumps(_run_row("current_mirror_simple_nmos", "sky130A", {"cm_iout": "VERIFIED"})) + "\n"
        + "not json at all\n"
    )
    records, gaps = build_law_records({str(path): load_run_rows(path)})
    assert len(records) == 1
    assert records[0].status == "process_scoped"   # only 1 PDK ran -> insufficient, not a crash


def test_build_law_records_ok_false_row_contributes_nothing():
    """A recipe run that errored on a PDK (ok=False) has no verdicts to contribute — that PDK
    reads as simply absent (not a fabricated PORT-FAILED verdict string in member_summary)."""
    rows = [
        _run_row("current_mirror_simple_nmos", "sky130A", {"cm_iout": "VERIFIED"}),
        _run_row("current_mirror_simple_nmos", "gf180mcuD", {}, ok=False),
    ]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = _write_jsonl(Path(d), "err.jsonl", rows)
        records, _ = build_law_records({str(path): load_run_rows(path)})
    assert records[0].pdks == ["sky130A"]


def test_build_law_records_merges_two_claim_ids_sharing_law_id(tmp_path):
    """Full-registry generality: if two DIFFERENT claim_ids happen to share
    (topology_class, metric, knob, quant_kind), they MERGE onto the same law_id (deterministic,
    sorted-claim_id-order union) rather than colliding silently."""
    from openclaw_brain.knowledge.executable import laws as laws_mod

    catalog_patch = dict(laws_mod._build_catalog())
    shape = catalog_patch["cm_iout"]
    catalog_patch["cm_iout_dup"] = shape   # identical topology/metric/knob/kind, different id

    orig = laws_mod._build_catalog
    laws_mod._build_catalog = lambda: catalog_patch
    try:
        path = _write_jsonl(tmp_path, "dup.jsonl", [
            _run_row("current_mirror_simple_nmos", "sky130A", {"cm_iout": "VERIFIED", "cm_iout_dup": "VERIFIED"}),
            _run_row("current_mirror_simple_nmos", "gf180mcuD", {"cm_iout": "VERIFIED"}),
            _run_row("current_mirror_simple_nmos", "ihp-sg13g2", {"cm_iout_dup": "VERIFIED"}),
        ])
        records, _ = build_law_records({str(path): load_run_rows(path)})
    finally:
        laws_mod._build_catalog = orig

    assert len(records) == 1   # same law_id -> ONE Regularity, not two
    rec = records[0]
    assert set(rec.claim_ids) == {"cm_iout", "cm_iout_dup"}
    assert rec.pdks == ["gf180mcuD", "ihp-sg13g2", "sky130A"]
    assert rec.status == "law"


# ── build_law_records against the REAL E1/E1b JSONL (no DB) ──


@pytest.mark.skipif(not E1_JSONL.exists() or not E1B_JSONL.exists(),
                     reason="E1/E1b replication JSONL not present in this checkout")
def test_build_law_records_against_real_e1_e1b_jsonl():
    load_results = {
        str(E1_JSONL): load_run_rows(E1_JSONL),
        str(E1B_JSONL): load_run_rows(E1B_JSONL),
    }
    records, gaps = build_law_records(load_results)
    assert gaps["unknown_claim_ids"] == []
    by_claim = {tuple(r.claim_ids): r for r in records}
    # spec §6: 5 nominal + 1 pelgrom + 2 statistical-bound = 8, all "law" (3/3, unanimous VERIFIED)
    assert len(records) == 8
    assert all(r.status == "law" for r in records)
    expected_claims = {("cm_iout",), ("cs_gbw",), ("cs_av0",), ("ota5t_gbw",), ("ota5t_av0",),
                       ("ota5t_vos",), ("ota5t_pelgrom",), ("col_fpn",)}
    assert set(by_claim) == expected_claims
    pelgrom = by_claim[("ota5t_pelgrom",)]
    assert pelgrom.quant_kind == "elasticity"
    assert len(pelgrom.member_summary) == 3
    for pdk in ("sky130A", "gf180mcuD", "ihp-sg13g2"):
        exp = pelgrom.member_summary[pdk]["fitted_exponent"]
        assert -0.6 <= exp <= -0.2


# ── build_law_records against the REAL full-registry JSONL (E-track ③ I2) ──
#
# The real e_rollout_full_registry_raw.jsonl (48 RunRecord rows: 16 topology classes x 3 PDKs, all
# ok=True, zero PORT-FAILED rows) shows ihp-sg13g2 producing a real REFUTED/FLAGGED verdict for 3
# claims rather than the pre-run brief's guessed PORT-FAILED — confirmed by direct inspection of
# the JSONL (not assumed): telescopic_cascode_ota_nmos_in/tele_av0 = REFUTED, .../tele_gbw =
# FLAGGED, regulated_cascode_nmos/rgc_iout = REFUTED, all on ihp-sg13g2 only (sky130A and
# gf180mcuD both VERIFIED on all three). classify_status's _AFFIRMING_FAMILY excludes both REFUTED
# and FLAGGED, so all three classify as agreeing=[gf180mcuD, sky130A] / disagreeing=[ihp-sg13g2] ->
# reason="divergent", status="process_scoped" (NOT "two_member" — two_member only fires when there
# is nothing disagreeing, spec §4) — matching pdks.py's own documented I1a finding (gf180 VERIFIED
# via VBA=1.8; ihp REFUTED at every explored VBA/WA point, CoV floor ~0.28% vs 0.1% bound) and the
# telescopic OTA's analogous ihp divergence.


@pytest.mark.skipif(not FULL_JSONL.exists(),
                     reason="full-registry replication JSONL not present in this checkout")
def test_build_law_records_against_real_full_registry_jsonl():
    load_results = {str(FULL_JSONL): load_run_rows(FULL_JSONL)}
    records, gaps = build_law_records(load_results)
    assert gaps["unknown_claim_ids"] == []
    by_claim = {tuple(r.claim_ids): r for r in records}

    rgc_iout = by_claim[("rgc_iout",)]
    assert rgc_iout.status == "process_scoped"
    assert rgc_iout.reason == "divergent"
    assert "ihp-sg13g2" in rgc_iout.status_note
    assert rgc_iout.agreeing_pdks == ["gf180mcuD", "sky130A"]
    assert rgc_iout.disagreeing_pdks == ["ihp-sg13g2"]

    for claim_id in ("tele_av0", "tele_gbw"):
        tele = by_claim[(claim_id,)]
        assert tele.status == "process_scoped", claim_id
        assert tele.reason == "divergent", claim_id
        assert "ihp-sg13g2" in tele.status_note, claim_id
        assert tele.agreeing_pdks == ["gf180mcuD", "sky130A"], claim_id
        assert tele.disagreeing_pdks == ["ihp-sg13g2"], claim_id

    # every OTHER claim in the full registry (13 of the 16 classes' claims plus the 2 miller_ota
    # E3 deltas) is a clean 3/3 agreement -- these 3 are the ONLY divergences.
    divergent = [r for r in records if r.reason == "divergent"]
    assert {tuple(r.claim_ids) for r in divergent} == {("rgc_iout",), ("tele_av0",), ("tele_gbw",)}


# ── project_law_records: idempotent MERGE, NO-PHANTOM, demotion (mocked store) ──


class _FakeStore:
    """Minimal stateful stand-in for GraphStore's get_node/run_read_query/write_batch surface —
    stateful so the SAME instance can be re-used across two project_law_records() calls to prove
    idempotency for real (not just "we didn't call write_batch this one time")."""

    def __init__(self):
        self.regularities: dict[str, dict] = {}
        self.supported_by: dict[str, set[str]] = {}
        self.about: dict[str, set[str]] = {}
        self.member_cards: dict[tuple[str, str, str], str] = {}   # (topology_class, pdk, claim) -> claim_card id
        self.write_batch_calls: list[dict] = []

    async def get_node(self, label, id_field, id_value):
        assert label == NodeLabel.REGULARITY and id_field == "law_id"
        node = self.regularities.get(id_value)
        return dict(node) if node else None

    async def run_read_query(self, query, params=None):
        params = params or {}
        if "HAS_CLAIM" in query:
            key = (params["tc"], params["pdk"], params["claim"])
            cid = self.member_cards.get(key)
            return [{"id": cid}] if cid else []
        if f"{RelType.SUPPORTED_BY.value}" in query:
            return [{"id": cid} for cid in sorted(self.supported_by.get(params["lid"], set()))]
        if f"{RelType.ABOUT.value}" in query:
            return [{"id": tid} for tid in sorted(self.about.get(params["lid"], set()))]
        raise AssertionError(f"unexpected query shape: {query}")

    async def write_batch(self, nodes=None, updates=None, edges=None):
        self.write_batch_calls.append({"nodes": list(nodes or []), "edges": list(edges or [])})
        for n in (nodes or []):
            self.regularities[n["id_value"]] = dict(n["properties"])
        for e in (edges or []):
            if e["rel_type"] == RelType.SUPPORTED_BY:
                self.supported_by.setdefault(e["source_id_value"], set()).add(e["target_id_value"])
            elif e["rel_type"] == RelType.ABOUT:
                self.about.setdefault(e["source_id_value"], set()).add(e["target_id_value"])


def _cm_record(verdicts_by_pdk, notes=None):
    from openclaw_brain.knowledge.executable.laws import classify_status, build_statement
    from openclaw_brain.knowledge.executable.models import QuantTest
    status_result = classify_status(verdicts_by_pdk)
    quant = QuantTest(kind="direction", sign="+")
    member_summary = {pdk: {"verdict": v, **({"note": notes[pdk]} if notes and pdk in notes else {})}
                       for pdk, v in verdicts_by_pdk.items()}
    return LawRecord(
        law_id=compute_law_id("current_mirror_simple_nmos", "iout_a", "Vout", "direction"),
        topology_class="current_mirror_simple_nmos", metric="iout_a", knob="Vout", quant_kind="direction",
        claim_ids=["cm_iout"], pdks=sorted(verdicts_by_pdk), member_summary=member_summary,
        status=status_result.status, status_note=status_result.note, reason=status_result.reason,
        agreeing_pdks=status_result.agreeing_pdks, disagreeing_pdks=status_result.disagreeing_pdks,
        statement=build_statement("current_mirror_simple_nmos", "Vout", "iout_a", quant, status_result),
        derived_from={"jsonl_paths": ["fake.jsonl"]},
    )


async def _noop_resolver(label, text):
    return None


async def test_project_law_records_dry_run_writes_nothing():
    store = _FakeStore()
    rec = _cm_record({"sky130A": "VERIFIED", "gf180mcuD": "VERIFIED", "ihp-sg13g2": "VERIFIED"})
    outcomes = await project_law_records(store, [rec], about_resolver=_noop_resolver, apply=False)
    assert store.write_batch_calls == []
    assert outcomes[0].status == "law"
    assert outcomes[0].action == "written"


async def test_project_law_records_no_phantom_missing_member_card_and_unresolved_about():
    store = _FakeStore()   # no member cards registered at all
    rec = _cm_record({"sky130A": "VERIFIED", "gf180mcuD": "VERIFIED", "ihp-sg13g2": "VERIFIED"})
    outcomes = await project_law_records(store, [rec], about_resolver=_noop_resolver, apply=True)
    assert outcomes[0].supported_by_written == 0
    assert outcomes[0].about_written == 0
    written_edges = [e for c in store.write_batch_calls for e in c["edges"]]
    assert written_edges == []   # NO-PHANTOM: no member card, no topology match -> no edges at all
    # but the PDK is still honestly listed in member_summary
    node = store.regularities[rec.law_id]
    assert json.loads(node["member_summary"]).keys() == {"sky130A", "gf180mcuD", "ihp-sg13g2"}


async def test_project_law_records_supported_by_only_existing_member_cards():
    store = _FakeStore()
    store.member_cards[("current_mirror_simple_nmos", "sky130A", "cm_iout")] = "sha256:abc:cm_iout"
    # gf180mcuD/ihp-sg13g2 member cards do NOT exist (report-only pilot cards, per spec)

    async def about_resolver(label, text):
        return "topo-cm" if label == NodeLabel.CIRCUIT_TOPOLOGY else None

    rec = _cm_record({"sky130A": "VERIFIED", "gf180mcuD": "VERIFIED", "ihp-sg13g2": "VERIFIED"})
    outcomes = await project_law_records(store, [rec], about_resolver=about_resolver, apply=True)
    assert outcomes[0].supported_by_written == 1
    assert outcomes[0].about_written == 1
    assert store.supported_by[rec.law_id] == {"sha256:abc:cm_iout"}
    assert store.about[rec.law_id] == {"topo-cm"}


async def test_project_law_records_second_apply_is_a_true_no_op():
    store = _FakeStore()
    store.member_cards[("current_mirror_simple_nmos", "sky130A", "cm_iout")] = "sha256:abc:cm_iout"

    async def about_resolver(label, text):
        return "topo-cm"

    rec = _cm_record({"sky130A": "VERIFIED", "gf180mcuD": "VERIFIED", "ihp-sg13g2": "VERIFIED"})
    first = await project_law_records(store, [rec], about_resolver=about_resolver, apply=True)
    assert first[0].action == "written"
    assert len(store.write_batch_calls) == 1
    assert store.write_batch_calls[0]["nodes"] and store.write_batch_calls[0]["edges"]

    second = await project_law_records(store, [rec], about_resolver=about_resolver, apply=True)
    assert second[0].action == "unchanged"
    # write_batch WAS called again (unconditionally, per apply=True), but with NOTHING to write —
    # the true idempotency claim is about the CONTENT of that call, not whether it's invoked.
    assert len(store.write_batch_calls) == 2
    assert store.write_batch_calls[1]["nodes"] == []
    assert store.write_batch_calls[1]["edges"] == []


async def test_project_law_records_refuses_to_narrow_pdks_and_leaves_existing_node_untouched(caplog):
    """The Known issues #1 accumulate-field-shrink defect: a later invocation whose loaded JSONL
    happens to carry fewer pdks than the law already has on the graph (e.g. a plain
    `project-laws --raw <one narrower file>`, not unioned with the standing/historical JSONLs the
    way `reverify.py::refeed_project_laws` always does) must NOT silently regress the accumulated
    pdks/member_summary/status — it must refuse, loudly, naming the law_id."""
    store = _FakeStore()
    rec_full = _cm_record({"sky130A": "VERIFIED", "gf180mcuD": "VERIFIED", "ihp-sg13g2": "VERIFIED"})
    first = await project_law_records(store, [rec_full], about_resolver=_noop_resolver, apply=True)
    assert first[0].status == "law"
    assert store.regularities[rec_full.law_id]["pdks"] == ["gf180mcuD", "ihp-sg13g2", "sky130A"]

    # compute_law_id is a pure function of (topology_class, metric, knob, quant_kind) — both records
    # target the SAME law_id even though this one's verdicts_by_pdk only covers sky130A.
    rec_narrow = _cm_record({"sky130A": "VERIFIED"})
    assert rec_narrow.law_id == rec_full.law_id

    with caplog.at_level(logging.WARNING):
        second = await project_law_records(store, [rec_narrow], about_resolver=_noop_resolver, apply=True)

    assert second[0].action == "refused_narrowing"
    assert second[0].status == "law"   # reports the TRUE, unchanged current status, not the (rejected) narrower one

    # the existing node is left EXACTLY as it was — not shrunk to 1 pdk, not regressed to process_scoped
    node = store.regularities[rec_full.law_id]
    assert node["pdks"] == ["gf180mcuD", "ihp-sg13g2", "sky130A"]
    assert node["status"] == "law"
    assert json.loads(node["member_summary"]).keys() == {"sky130A", "gf180mcuD", "ihp-sg13g2"}

    # loud: names the law_id so an operator can find exactly which law was protected
    assert rec_full.law_id in caplog.text
    assert "SHRINK" in caplog.text

    # write_batch WAS still called (unconditionally, per apply=True) but with nothing for this record
    assert len(store.write_batch_calls) == 2
    assert store.write_batch_calls[1]["nodes"] == []
    assert store.write_batch_calls[1]["edges"] == []


async def test_project_law_records_narrowing_guard_does_not_affect_normal_idempotent_reapply():
    """Sanity check that the new guard is scoped to genuine narrowing only — re-applying the exact
    SAME (non-narrower) record twice must still be a true no-op (existing idempotency contract,
    unaffected by the accumulate-field-shrink guard)."""
    store = _FakeStore()
    rec = _cm_record({"sky130A": "VERIFIED", "gf180mcuD": "VERIFIED", "ihp-sg13g2": "VERIFIED"})
    first = await project_law_records(store, [rec], about_resolver=_noop_resolver, apply=True)
    assert first[0].action == "written"

    second = await project_law_records(store, [rec], about_resolver=_noop_resolver, apply=True)
    assert second[0].action == "unchanged"   # NOT "refused_narrowing" — identical pdks, no shrink


async def test_project_law_records_journals_written_and_updated_not_unchanged():
    store = _FakeStore()
    journal_entries = []

    class _FakeJournal:
        def log(self, op, **kwargs):
            journal_entries.append({"op": op, **kwargs})

    rec = _cm_record({"sky130A": "VERIFIED", "gf180mcuD": "VERIFIED", "ihp-sg13g2": "VERIFIED"})
    await project_law_records(store, [rec], about_resolver=_noop_resolver,
                               journal=_FakeJournal(), apply=True)
    assert len(journal_entries) == 1
    assert journal_entries[0]["op"] == "project_law"
    assert journal_entries[0]["action"] == "written"
    assert journal_entries[0]["law_id"] == rec.law_id

    # a genuine content change (a new note appears) -> "updated", also journaled
    journal_entries.clear()
    rec2 = _cm_record({"sky130A": "VERIFIED", "gf180mcuD": "VERIFIED", "ihp-sg13g2": "VERIFIED"},
                       notes={"sky130A": "trend +, monotonic=True"})
    await project_law_records(store, [rec2], about_resolver=_noop_resolver,
                               journal=_FakeJournal(), apply=True)
    assert len(journal_entries) == 1 and journal_entries[0]["action"] == "updated"

    # re-running the SAME (now current) data logs nothing — "unchanged" never journaled
    journal_entries.clear()
    await project_law_records(store, [rec2], about_resolver=_noop_resolver,
                               journal=_FakeJournal(), apply=True)
    assert journal_entries == []


async def test_project_law_records_demotes_loudly_then_stays_demoted():
    store = _FakeStore()
    journal_entries = []

    class _FakeJournal:
        def log(self, op, **kwargs):
            journal_entries.append({"op": op, **kwargs})

    rec_law = _cm_record({"sky130A": "VERIFIED", "gf180mcuD": "VERIFIED", "ihp-sg13g2": "VERIFIED"})
    await project_law_records(store, [rec_law], about_resolver=_noop_resolver,
                               journal=_FakeJournal(), apply=True)
    assert store.regularities[rec_law.law_id]["status"] == "law"

    rec_divergent = _cm_record({"sky130A": "VERIFIED", "gf180mcuD": "VERIFIED", "ihp-sg13g2": "REFUTED"})
    outcomes = await project_law_records(store, [rec_divergent], about_resolver=_noop_resolver,
                                          journal=_FakeJournal(), apply=True)
    assert outcomes[0].action == "demoted"
    node = store.regularities[rec_divergent.law_id]
    assert node["status"] == "demoted"
    history = json.loads(node["member_summary"])["_history"]
    assert len(history) == 1
    assert history[0]["prior_status"] == "law"
    demotion_logs = [e for e in journal_entries if e.get("action") == "demoted"]
    assert len(demotion_logs) == 1   # loud — exactly one journal entry for the transition

    # Re-run with the SAME divergent data: stays "demoted", no duplicate history entry, no new log.
    journal_entries.clear()
    outcomes2 = await project_law_records(store, [rec_divergent], about_resolver=_noop_resolver,
                                           journal=_FakeJournal(), apply=True)
    assert outcomes2[0].action == "unchanged"
    history2 = json.loads(store.regularities[rec_divergent.law_id]["member_summary"])["_history"]
    assert len(history2) == 1   # unchanged, not duplicated
    assert not [e for e in journal_entries if e.get("action") == "demoted"]


# ── live acceptance (opt-in gate, then reachable Neo4j) ──


@pytest.mark.asyncio
async def test_live_acceptance_requires_opt_in_before_connect(monkeypatch):
    from openclaw_brain.knowledge.graph.store import GraphStore

    monkeypatch.delenv("RUN_LIVE_GRAPH_TESTS", raising=False)
    connect_calls = []

    async def unexpected_connect(self) -> None:
        connect_calls.append(self)
        raise AssertionError("live Neo4j connection attempted without opt-in")

    monkeypatch.setattr(GraphStore, "connect", unexpected_connect)
    with pytest.raises(pytest.skip.Exception, match="live-graph test gated"):
        await test_live_acceptance_project_laws_against_real_neo4j()
    assert connect_calls == []


@pytest.mark.asyncio
async def test_live_acceptance_project_laws_against_real_neo4j():
    require_live_graph()
    from openclaw_brain.config import load_config
    from openclaw_brain.knowledge.executable.resolver import GraphResolver
    from openclaw_brain.knowledge.graph.store import GraphStore

    if not E1_JSONL.exists() or not E1B_JSONL.exists():
        pytest.skip("E1/E1b replication JSONL not present in this checkout")

    config = load_config()
    store = GraphStore(config.neo4j)
    try:
        await store.connect()
    except Exception:
        pytest.skip("Neo4j not reachable; skipping live law-tier acceptance")
        return

    try:
        load_results = {
            str(E1_JSONL): load_run_rows(E1_JSONL),
            str(E1B_JSONL): load_run_rows(E1B_JSONL),
        }
        records, gaps = build_law_records(load_results)
        assert gaps["unknown_claim_ids"] == []
        assert len(records) == 8

        resolver = GraphResolver(store)
        # NOTE: this is a real, persistent Neo4j — a prior run of this same test (or a manual
        # `project-laws --apply`) may already have written these laws, so `first` can legitimately
        # be "unchanged" too. The idempotency claim under test is that the SECOND call below is
        # unchanged REGARDLESS of what the first one was — proven independent of prior DB state.
        first = await project_law_records(store, records, about_resolver=resolver.resolve, apply=True)
        assert all(o.action in ("written", "updated", "unchanged") for o in first)
        assert all(o.status == "law" for o in first)   # unanimous VERIFIED across all 3 PDKs

        # idempotent re-run: zero content written the second time
        second = await project_law_records(store, records, about_resolver=resolver.resolve, apply=True)
        assert all(o.action == "unchanged" for o in second)

        # read back one law node + its SUPPORTED_BY edge
        node = await store.get_node(NodeLabel.REGULARITY, "law_id", records[0].law_id)
        assert node is not None
        assert node["status"] == "law"
        edge_rows = await store.run_read_query(
            "MATCH (:Regularity {law_id: $lid})-[:SUPPORTED_BY]->(c:ClaimCard) RETURN c.claim_id AS id",
            {"lid": records[0].law_id},
        )
        assert len(edge_rows) >= 1   # sky130A member card is projected in the production corpus
    finally:
        await store.close()
