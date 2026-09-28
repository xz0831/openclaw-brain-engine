"""Tests for the S4 render-only PrimeSim/HSPICE dialect adapter
(knowledge/executable/render_hspice.py) -- ADR-044 D2.

All mocked/pure (no DB, no docker, no ngspice, no primesim): dialect-transformation unit tests
against render_hspice_deck directly, byte-stability + golden-file checks against the 4 committed
bench/primesim_pilot/*/bench.sp artifacts, and a mocked-store test proving EXPECTATION.md content
is actually threaded through from graph fields (Regularity/ClaimCard), not hardcoded -- mirrors
the repo's _FakeStore convention (test_executable_laws.py / test_executable_reverify.py) with
Mock-fidelity real shapes (raw property dicts as GraphStore.get_node/run_read_query actually
return them, JSON sub-fields left as strings).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openclaw_brain.knowledge.executable.laws import compute_law_id
from openclaw_brain.knowledge.executable.render_hspice import (
    COMPANY_PDK_LIB_PLACEHOLDER,
    CORNER_PLACEHOLDER,
    NMOS_MODEL_PLACEHOLDER,
    PMOS_MODEL_PLACEHOLDER,
    ExpectationData,
    HOME_MEASURED_NOTES,
    HspiceBenchSpec,
    MeasureTranslation,
    PILOT_BENCHES,
    _split_ngspice_deck,
    _substitute_device_models,
    build_expectation_markdown,
    dialect_citations,
    fetch_claim_cards,
    fetch_expectation_data,
    fetch_regularity,
    render_all_pilot_benches,
    render_hspice_deck,
    render_pilot_deck,
)
from openclaw_brain.knowledge.graph.schema import NodeLabel

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_ROOT = REPO_ROOT / "bench" / "primesim_pilot"

_SIMPLE_NGSPICE_DECK = """* T-template widget_nmos (AC, CL sweep)
.lib "__LIBPATH__" tt
.param VDD=1.8 CL=1p W=8 Lp=0.5
VDD vdd 0 {VDD}
XM1 out in 0 0 sky130_fd_pr__nfet_01v8 W={W} L={Lp}
XM2 out in vdd vdd sky130_fd_pr__pfet_01v8 W={W} L={Lp}
CL out 0 {CL}
.control
foreach pt 1p 2p
  alter CL = $pt
  ac dec 40 1 10G
  meas ac y max vdb(out)
  echo RDATA cl $pt $&y
end
.endc
.end
"""

_SIMPLE_SPEC = HspiceBenchSpec(
    bench_id="widget_test",
    topology_class="widget_nmos",
    claim_id="widget_claim",
    corner="tt",
    temp_c=27.0,
    sweep_param="CL",
    sweep_points=("1p", "2p"),
    analysis_lines=(".AC DEC 40 1 10G",),
    measures=(
        MeasureTranslation(
            ngspice_source="meas ac y max vdb(out)",
            hspice_lines=(".MEASURE AC av0_db MAX VDB(out)",),
            citation_keys=("measure_func_max",),
        ),
    ),
    probe_lines=(".PROBE AC V(out)",),
)


# ============================================================================
# _split_ngspice_deck / _substitute_device_models -- mechanical extraction
# ============================================================================


def test_split_ngspice_deck_extracts_title_param_and_device_lines():
    title, param_body, device_lines = _split_ngspice_deck(_SIMPLE_NGSPICE_DECK)
    assert title == "T-template widget_nmos (AC, CL sweep)"
    assert param_body == "VDD=1.8 CL=1p W=8 Lp=0.5"
    assert device_lines == [
        "VDD vdd 0 {VDD}",
        "XM1 out in 0 0 sky130_fd_pr__nfet_01v8 W={W} L={Lp}",
        "XM2 out in vdd vdd sky130_fd_pr__pfet_01v8 W={W} L={Lp}",
        "CL out 0 {CL}",
    ]


def test_split_ngspice_deck_raises_without_title_comment():
    with pytest.raises(ValueError, match="title comment"):
        _split_ngspice_deck('.lib "x" tt\n.param A=1\nVDD vdd 0 {A}\n.end\n')


def test_split_ngspice_deck_raises_without_lib_line():
    with pytest.raises(ValueError, match=r"\.lib"):
        _split_ngspice_deck("* title\n.param A=1\nVDD vdd 0 {A}\n.end\n")


def test_split_ngspice_deck_raises_without_param_line():
    with pytest.raises(ValueError, match=r"\.param"):
        _split_ngspice_deck('* title\n.lib "x" tt\nVDD vdd 0 1\n.end\n')


def test_substitute_device_models_replaces_known_tokens():
    lines = [
        "XM1 out in 0 0 sky130_fd_pr__nfet_01v8 W={W} L={Lp}",
        "XM2 out in vdd vdd sky130_fd_pr__pfet_01v8 W={W} L={Lp}",
        "CL out 0 {CL}",
    ]
    out = _substitute_device_models(lines)
    assert out[0] == f"XM1 out in 0 0 {NMOS_MODEL_PLACEHOLDER} W={{W}} L={{Lp}}"
    assert out[1] == f"XM2 out in vdd vdd {PMOS_MODEL_PLACEHOLDER} W={{W}} L={{Lp}}"
    assert out[2] == "CL out 0 {CL}"  # untouched, no device-model token present


def test_substitute_device_models_raises_on_unrecognized_sky130_token():
    lines = ["XM9 a b c d sky130_fd_pr__pnp_05v5_W0p68L0p68 area=1"]
    with pytest.raises(ValueError, match="unrecognized"):
        _substitute_device_models(lines)


# ============================================================================
# render_hspice_deck -- the dialect adapter itself
# ============================================================================


def test_render_hspice_deck_contains_company_placeholders():
    deck = render_hspice_deck(_SIMPLE_NGSPICE_DECK, _SIMPLE_SPEC)
    assert f".LIB '{COMPANY_PDK_LIB_PLACEHOLDER}' {CORNER_PLACEHOLDER}" in deck
    assert NMOS_MODEL_PLACEHOLDER in deck
    assert PMOS_MODEL_PLACEHOLDER in deck
    assert "TODO(engineer)" in deck


def test_render_hspice_deck_never_leaks_sky130_literal():
    deck = render_hspice_deck(_SIMPLE_NGSPICE_DECK, _SIMPLE_SPEC)
    assert "sky130_fd_pr__" not in deck


def test_render_hspice_deck_keeps_param_values_unchanged():
    deck = render_hspice_deck(_SIMPLE_NGSPICE_DECK, _SIMPLE_SPEC)
    # geometry/electrical .PARAM values kept verbatim from the template (task requirement)
    assert ".PARAM VDD=1.8 CL=1p W=8 Lp=0.5" in deck


def test_render_hspice_deck_never_contains_ngspice_control_block():
    deck = render_hspice_deck(_SIMPLE_NGSPICE_DECK, _SIMPLE_SPEC)
    assert ".control" not in deck
    assert ".endc" not in deck
    assert "foreach" not in deck
    assert "echo RDATA" not in deck


def test_render_hspice_deck_emits_measure_probe_option_alter():
    deck = render_hspice_deck(_SIMPLE_NGSPICE_DECK, _SIMPLE_SPEC)
    assert ".MEASURE AC av0_db MAX VDB(out)" in deck
    assert ".OPTION POST PROBE" in deck
    assert ".PROBE AC V(out)" in deck
    assert ".ALTER pt0_1p" in deck
    assert ".PARAM CL=1p" in deck
    assert ".ALTER pt1_2p" in deck
    assert ".PARAM CL=2p" in deck
    assert deck.rstrip().endswith(".END")


def test_render_hspice_deck_cites_guide_chunks_for_its_measure():
    deck = render_hspice_deck(_SIMPLE_NGSPICE_DECK, _SIMPLE_SPEC)
    for chunk_id in dialect_citations()["measure_func_max"]["chunks"]:
        assert chunk_id in deck


def test_render_hspice_deck_is_byte_stable():
    """Same (ngspice_deck, spec) -> byte-identical output, every call (no timestamps/wall-clock/
    randomness anywhere in the render path)."""
    a = render_hspice_deck(_SIMPLE_NGSPICE_DECK, _SIMPLE_SPEC)
    b = render_hspice_deck(_SIMPLE_NGSPICE_DECK, _SIMPLE_SPEC)
    assert a == b


def test_render_hspice_deck_never_invokes_a_simulator():
    """Grep-able invariant per the module's own docstring claim. Checks actual import/call
    patterns, not bare word occurrence (the module's own docstring discusses 'subprocess' by name
    while explaining this very invariant, so a naive 'word not in text' check would self-trip).
    NOTE: 'os.system(' below is a string literal searched for (never called) -- this test asserts
    render_hspice.py's SOURCE TEXT does not contain a shell-exec sink, it does not itself invoke
    a shell."""
    src = (REPO_ROOT / "src" / "openclaw_brain" / "knowledge" / "executable"
           / "render_hspice.py").read_text()
    assert "import subprocess" not in src
    assert "from subprocess" not in src
    assert "import docker" not in src
    banned_calls = ["Popen(", "os.system(", "run_deck(", "NgspiceRunner(", "VerilogRunner("]
    for banned in banned_calls:
        assert banned not in src, f"render_hspice.py must never call a simulator ({banned!r} found)"


# ============================================================================
# The 4 pilot benches -- byte-stability + golden-file regression
# ============================================================================


@pytest.mark.parametrize("bench", PILOT_BENCHES, ids=lambda b: b.bench_id)
def test_pilot_bench_renders_without_error_and_is_byte_stable(bench):
    first = render_pilot_deck(bench)
    second = render_pilot_deck(bench)
    assert first == second
    assert "sky130_fd_pr__" not in first
    assert COMPANY_PDK_LIB_PLACEHOLDER in first
    assert NMOS_MODEL_PLACEHOLDER in first
    assert PMOS_MODEL_PLACEHOLDER in first
    assert first.rstrip().endswith(".END")


@pytest.mark.parametrize("bench", PILOT_BENCHES, ids=lambda b: b.bench_id)
def test_pilot_bench_golden_file_matches_committed_deck(bench):
    """Regression pin: the renderer's current output must match the checked-in
    bench/primesim_pilot/<bench_id>/bench.sp exactly. A deliberate dialect change updates BOTH
    render_hspice.py and the committed deck in the same commit (mirrors
    test_executable_interventions.py::test_baseline_ota_body_byte_identical's discipline)."""
    golden_path = BENCH_ROOT / bench.bench_id / "bench.sp"
    if not golden_path.exists():
        pytest.skip(f"no committed golden deck at {golden_path} yet")
    assert render_pilot_deck(bench) == golden_path.read_text()


def test_pilot_benches_have_exactly_four_entries_in_the_stated_scope():
    assert len(PILOT_BENCHES) == 4
    ids = {b.bench_id for b in PILOT_BENCHES}
    assert ids == {
        "ota5t_av0_vs_cl",
        "miller_ota_gbw_vs_cl_elasticity",
        "miller_ota_pm_vs_cl",
        "cascode_mirror_iout_accuracy",
    }


def test_bench_2_is_the_refuted_elasticity_misconception_card():
    b2 = next(b for b in PILOT_BENCHES if b.bench_id == "miller_ota_gbw_vs_cl_elasticity")
    assert b2.quant_kind == "elasticity"
    assert b2.misconception_note is not None
    assert "-0.428" in b2.misconception_note or "-0.43" in b2.misconception_note


def test_every_measure_translation_cites_a_real_dialect_citation_key():
    citations = dialect_citations()
    for bench in PILOT_BENCHES:
        for measure in bench.measures:
            for key in measure.citation_keys:
                assert key in citations, f"{bench.bench_id}: unknown citation key {key!r}"


def test_dialect_citations_are_well_formed():
    for key, cite in dialect_citations().items():
        assert cite["rule"], f"{key}: empty rule"
        assert cite["chunks"], f"{key}: no chunk_id citations"
        for chunk_id in cite["chunks"]:
            assert chunk_id.startswith("chunk_"), f"{key}: malformed chunk_id {chunk_id!r}"


def test_home_measured_notes_cover_every_pilot_bench():
    for bench in PILOT_BENCHES:
        assert bench.bench_id in HOME_MEASURED_NOTES
        assert HOME_MEASURED_NOTES[bench.bench_id].strip()


# ============================================================================
# EXPECTATION.md sourcing -- mocked store, Mock-fidelity (real Neo4j property shapes)
# ============================================================================


class _FakeStore:
    """Minimal stand-in for GraphStore's get_node/run_read_query read surface. Shapes mirror
    what a live probe of the real graph actually returned during this task's research (raw
    property dicts; member_summary/scope/derived_from are JSON-ENCODED STRINGS, exactly as
    Neo4j returns them via projection.py/laws.py's own writers -- never pre-parsed dicts)."""

    def __init__(self, regularities: dict[str, dict] | None = None,
                 claim_rows: list[dict] | None = None):
        self._regularities = regularities or {}
        self._claim_rows = claim_rows or []
        self.get_node_calls: list[tuple] = []
        self.run_read_query_calls: list[tuple] = []

    async def get_node(self, label, id_field, id_value):
        self.get_node_calls.append((label, id_field, id_value))
        assert label == NodeLabel.REGULARITY
        assert id_field == "law_id"
        node = self._regularities.get(id_value)
        return dict(node) if node else None

    async def run_read_query(self, query, params=None):
        self.run_read_query_calls.append((query, params))
        params = params or {}
        assert "HAS_CLAIM" in query
        return [
            {"pdk": r["pdk"], "spec_id": r["spec_id"], "card": dict(r["card"])}
            for r in self._claim_rows
            if r["card"].get("claim") == params.get("claim")
        ]


def _real_shape_regularity(law_id: str) -> dict:
    """A hand-built dict matching the EXACT property shape observed live on a real Regularity
    node (see render_hspice.py's fetch_regularity docstring) -- deliberately using FAKE pdk/note
    values that do not match any real production law, so a test asserting these exact strings
    appear in the rendered card can only pass if the data genuinely flowed through from here."""
    return {
        "law_id": law_id,
        "topology_class": "widget_nmos",
        "metric": "av0_db",
        "knob": "CL",
        "quant_kind": "invariance",
        "claim_ids": ["widget_claim"],
        "pdks": ["fakepdkA", "fakepdkB"],
        "member_summary": json.dumps({
            "fakepdkA": {"verdict": "VERIFIED", "note": "spread 0.01 (max 0.5) [FAKE-TEST-VALUE]"},
            "fakepdkB": {"verdict": "VERIFIED", "note": "spread 0.02 (max 0.5) [FAKE-TEST-VALUE]"},
        }, sort_keys=True),
        "status": "law",
        "status_note": "",
        "statement": "For widget_nmos, av0_db is invariant to CL [FAKE-TEST-STATEMENT].",
        "derived_from": json.dumps({"jsonl_paths": ["experiments/fake_test_raw.jsonl"]}),
    }


def _real_shape_claim_card(pdk: str, spec_id: str, claim: str, verdict: str) -> dict:
    """Mirrors projection.py::project_specimen's exact ClaimCard node property dict (the fields
    it actually appends to pw.nodes -- claim/knob/metric/verdict/quant_kind/engine/basis/scope/
    dominant_risk_untested/narrative/corner/temp_c/vdd; NOTE verdict_note is deliberately absent
    -- projection.py never writes it, so a real ClaimCard node never carries it either)."""
    return {
        "claim": claim,
        "claim_id": f"{spec_id}:{claim}",
        "knob": "CL",
        "metric": "av0_db",
        "verdict": verdict,
        "quant_kind": "invariance",
        "engine": "ngspice",
        "basis": "physical-nominal",
        "scope": json.dumps({"device": "nominal", "pdk": pdk, "corners": [f"{pdk}/tt/27/1.8"],
                              "statistical": "none"}),
        "dominant_risk_untested": None,
        "narrative": "[FAKE-TEST-NARRATIVE] av0 is set by output resistances, independent of CL.",
        "corner": "tt",
        "temp_c": 27.0,
        "vdd": 1.8,
    }


@pytest.fixture
def fake_widget_bench():
    return next(iter(PILOT_BENCHES)).__class__(
        bench_id="widget_test", title="widget test", topology_class="widget_nmos",
        claim="widget_claim", knob="CL", metric="av0_db", quant_kind="invariance",
        corner="tt", temp_c=27.0, sweep_param="CL", sweep_points=("1p", "2p"),
        render_fn_name="render_ota_5t_ac", measures=(), probe_lines=(),
    )


@pytest.mark.asyncio
async def test_fetch_regularity_keys_by_compute_law_id(fake_widget_bench):
    law_id = compute_law_id("widget_nmos", "av0_db", "CL", "invariance")
    store = _FakeStore(regularities={law_id: _real_shape_regularity(law_id)})

    result = await fetch_regularity(store, "widget_nmos", "av0_db", "CL", "invariance")

    assert result is not None
    assert result["law_id"] == law_id
    assert store.get_node_calls == [(NodeLabel.REGULARITY, "law_id", law_id)]


@pytest.mark.asyncio
async def test_fetch_regularity_returns_none_when_not_a_law():
    store = _FakeStore(regularities={})
    result = await fetch_regularity(store, "widget_nmos", "gbw_hz", "CL", "elasticity")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_claim_cards_filters_by_topology_and_claim():
    rows = [
        {"pdk": "fakepdkA", "spec_id": "sha256:aaa",
         "card": _real_shape_claim_card("fakepdkA", "sha256:aaa", "widget_claim", "VERIFIED")},
        {"pdk": "fakepdkB", "spec_id": "sha256:bbb",
         "card": _real_shape_claim_card("fakepdkB", "sha256:bbb", "widget_claim", "VERIFIED")},
        {"pdk": "fakepdkA", "spec_id": "sha256:ccc",
         "card": _real_shape_claim_card("fakepdkA", "sha256:ccc", "other_claim", "REFUTED")},
    ]
    store = _FakeStore(claim_rows=rows)

    result = await fetch_claim_cards(store, "widget_nmos", "widget_claim")

    assert len(result) == 2
    assert {r["pdk"] for r in result} == {"fakepdkA", "fakepdkB"}
    assert all(r["card"]["claim"] == "widget_claim" for r in result)
    assert store.run_read_query_calls[0][1] == {"tc": "widget_nmos", "claim": "widget_claim"}


@pytest.mark.asyncio
async def test_expectation_card_is_sourced_from_graph_fields_not_hardcoded(fake_widget_bench):
    """The load-bearing test: build an ExpectationData from FAKE-marked graph field values (via
    the mocked store) and assert those exact injected strings -- not any real production number
    -- appear in the rendered EXPECTATION.md text. A version of build_expectation_markdown that
    silently used hardcoded/real values instead of its `data` argument would fail this test."""
    law_id = compute_law_id("widget_nmos", "av0_db", "CL", "invariance")
    store = _FakeStore(
        regularities={law_id: _real_shape_regularity(law_id)},
        claim_rows=[
            {"pdk": "fakepdkA", "spec_id": "sha256:aaa",
             "card": _real_shape_claim_card("fakepdkA", "sha256:aaa", "widget_claim", "VERIFIED")},
            {"pdk": "fakepdkB", "spec_id": "sha256:bbb",
             "card": _real_shape_claim_card("fakepdkB", "sha256:bbb", "widget_claim", "VERIFIED")},
        ],
    )

    data = await fetch_expectation_data(store, fake_widget_bench, "[FAKE-TEST-HOME-NOTE] see nowhere")
    card = build_expectation_markdown(data)

    # graph-sourced fields actually appear in the rendered card:
    assert "fakepdkA" in card and "fakepdkB" in card
    assert "spread 0.01 (max 0.5) [FAKE-TEST-VALUE]" in card
    assert "spread 0.02 (max 0.5) [FAKE-TEST-VALUE]" in card
    assert "[FAKE-TEST-STATEMENT]" in card
    assert law_id in card
    assert "[FAKE-TEST-HOME-NOTE]" in card
    assert "sha256:aaa:widget_claim" in card
    assert "sha256:bbb:widget_claim" in card
    # and the status/law framing follows the graph, not a hardcoded assumption:
    assert "**status: LAW**" in card


@pytest.mark.asyncio
async def test_expectation_card_scope_honest_when_not_a_law(fake_widget_bench):
    """A claim with no Regularity node (single-PDK, e.g. bench 2's shape) must render as
    'single-PDK ClaimCard', never silently claim LAW status."""
    store = _FakeStore(
        regularities={},
        claim_rows=[
            {"pdk": "fakepdkA", "spec_id": "sha256:aaa",
             "card": _real_shape_claim_card("fakepdkA", "sha256:aaa", "widget_claim", "REFUTED")},
        ],
    )
    data = await fetch_expectation_data(store, fake_widget_bench, "[FAKE-TEST-HOME-NOTE]")
    card = build_expectation_markdown(data)
    assert "**status: LAW**" not in card
    assert "single-PDK ClaimCard" in card


def test_expectation_data_is_a_plain_dataclass_no_hidden_graph_access():
    """ExpectationData itself must be a pure data holder -- build_expectation_markdown takes no
    store argument, so it structurally cannot reach back out to the graph."""
    import inspect

    sig = inspect.signature(build_expectation_markdown)
    assert list(sig.parameters) == ["data"]
    assert not inspect.iscoroutinefunction(build_expectation_markdown)


def test_expectation_data_can_be_hand_constructed_for_direct_unit_tests(fake_widget_bench):
    """Confirms the dataclass's field shape directly (not just via fetch_expectation_data) --
    a caller/test can build one by hand without touching a store at all."""
    data = ExpectationData(
        bench=fake_widget_bench, regularity=None, claim_cards=[], home_measured_note="n/a"
    )
    assert data.bench is fake_widget_bench
    assert data.regularity is None
    assert data.claim_cards == []
    card = build_expectation_markdown(data)
    assert "single-PDK ClaimCard" in card


# ============================================================================
# render_all_pilot_benches -- the cli.py orchestration entry point
# ============================================================================


@pytest.mark.asyncio
async def test_render_all_pilot_benches_writes_deck_and_card_per_bench(tmp_path):
    law_id_av0 = compute_law_id("ota_5t_nmos_in", "av0_db", "CL", "invariance")
    store = _FakeStore(
        regularities={law_id_av0: _real_shape_regularity(law_id_av0)},
        claim_rows=[],
    )

    results = await render_all_pilot_benches(store, tmp_path)

    assert len(results) == 4
    for bench in PILOT_BENCHES:
        bench_dir = tmp_path / bench.bench_id
        assert (bench_dir / "bench.sp").exists()
        assert (bench_dir / "EXPECTATION.md").exists()
        deck_text = (bench_dir / "bench.sp").read_text()
        assert "sky130_fd_pr__" not in deck_text
