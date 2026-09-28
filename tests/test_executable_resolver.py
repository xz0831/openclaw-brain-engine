"""Hybrid projection resolver (knowledge/executable/resolver.py).

The dry-run found pure top-1 embedding mis-ranks near-synonym distractors (pm/av0 -> "DC Loop Gain"
even though "Phase Margin"/"DC Open-Loop Gain" exist). The hybrid filters text-first (curated
canonical-vocabulary CONTAINS) then ranks by embedding — so the exact-kind node wins. Fallback to
gated vector search forms no edge when nothing resolves (NO-PHANTOM). Mock store + injected encode,
no DB / no model.
"""

from __future__ import annotations

import pytest
from tests.conftest import require_live_graph

from openclaw_brain.knowledge.executable.projection import _metric_text, _readable
from openclaw_brain.knowledge.executable.resolver import GraphResolver, _expansions
from openclaw_brain.knowledge.graph.schema import NodeLabel


class _MockStore:
    def __init__(self, text_rows=None, vec_rows=None):
        self.text_rows = text_rows if text_rows is not None else []
        self.vec_rows = vec_rows if vec_rows is not None else []
        self.queries = []

    async def run_read_query(self, query, params):
        self.queries.append((query, params))
        return self.vec_rows if "queryNodes" in query else self.text_rows


# Three phase-margin nodes (the real graph has exactly these); the distractor "DC Loop Gain" would
# only appear via pure vector search, which the text filter never reaches here.
_PM_ROWS = [
    {"id": "P_cm_pm", "name": "Common-Mode Loop Phase Margin", "embedding": [1.0, 0.0, 0.0]},
    {"id": "P_pm", "name": "Phase Margin", "embedding": [0.0, 1.0, 0.0]},
    {"id": "P_diff_pm", "name": "Differential Signal Phase Margin", "embedding": [0.0, 0.0, 1.0]},
]


def _encode_to(target_vec):
    return lambda text: target_vec


@pytest.mark.asyncio
async def test_text_filter_then_embedding_rank_picks_exact():
    store = _MockStore(text_rows=_PM_ROWS)
    # the query embedding aligns with the exact "Phase Margin" node, not the CM/diff variants
    r = GraphResolver(store, encode_fn=_encode_to([0.0, 1.0, 0.0]))
    assert await r.resolve(NodeLabel.PARAMETER, "pm") == "P_pm"
    # it used the TEXT path (CONTAINS), never the vector index
    assert all("queryNodes" not in q for q, _ in store.queries)


@pytest.mark.asyncio
async def test_exact_name_beats_embedding_distractor():
    # iout regression: a verbose context-specific node ("...in Inverter Delay Model") embeds closer
    # to the query than the general "Output Current", but the exact-name match must win.
    rows = [
        {"id": "P_inv", "name": "Output Current (I) in Inverter Delay Model", "embedding": [0.0, 1.0]},
        {"id": "P_out", "name": "Output Current", "embedding": [1.0, 0.0]},
    ]
    store = _MockStore(text_rows=rows)
    r = GraphResolver(store, encode_fn=_encode_to([0.0, 1.0]))   # embedding favors the distractor
    assert await r.resolve(NodeLabel.PARAMETER, "iout") == "P_out"


@pytest.mark.asyncio
async def test_unmapped_hint_falls_back_to_gated_vector():
    # vec_rows scores are Neo4j NORMALIZED (1+cos)/2; the floor is in cosine space.
    # raw 0.70 -> cos 0.40 (below the 0.60 cosine floor) -> None (no edge)
    store = _MockStore(vec_rows=[{"id": "P_x", "score": 0.70}])
    r = GraphResolver(store, encode_fn=_encode_to([1.0, 0.0]), vec_threshold=0.60)
    assert await r.resolve(NodeLabel.PARAMETER, "some unmapped metric") is None
    # raw 0.85 -> cos 0.70 (above) -> resolves
    store2 = _MockStore(vec_rows=[{"id": "P_y", "score": 0.85}])
    r2 = GraphResolver(store2, encode_fn=_encode_to([1.0, 0.0]), vec_threshold=0.60)
    assert await r2.resolve(NodeLabel.PARAMETER, "some unmapped metric") == "P_y"


@pytest.mark.asyncio
async def test_vector_fallback_converts_neo4j_score_to_cosine():
    # Regression (C2): Neo4j normalizes cosine to (1+cos)/2, so a raw index score of 0.72 is
    # cosine 0.44 — BELOW the 0.60 cosine floor. Before the fix the resolver gated the RAW score
    # (0.72 >= 0.60) and wrote a near-orthogonal link to the live graph.
    store = _MockStore(vec_rows=[{"id": "P_near_orthogonal", "score": 0.72}])
    r = GraphResolver(store, encode_fn=_encode_to([1.0, 0.0]), vec_threshold=0.60)
    assert await r.resolve(NodeLabel.PARAMETER, "unmapped") is None


@pytest.mark.asyncio
async def test_mapped_hint_with_no_text_hit_falls_back_to_vector():
    # curated expansion exists but CONTAINS returns nothing -> vector fallback (still gated)
    store = _MockStore(text_rows=[], vec_rows=[{"id": "T_z", "score": 0.80}])
    r = GraphResolver(store, encode_fn=_encode_to([1.0, 0.0]), vec_threshold=0.60)
    assert await r.resolve(NodeLabel.CIRCUIT_TOPOLOGY, "miller ota 2stage nmos in") == "T_z"
    assert any("queryNodes" in q for q, _ in store.queries)


def test_expansions_cover_every_validated_hint():
    """Drift guard: the resolver must map the EXACT munged hints the projection emits for the
    validated specimens — if projection._metric_text/_readable change, this catches the break."""
    for metric in ("gbw_hz", "av0_db", "pm_deg", "iout_a"):
        assert _expansions(NodeLabel.PARAMETER, _metric_text(metric)), metric
    for tclass in ("miller_ota_2stage_nmos_in", "current_mirror_simple_nmos"):
        assert _expansions(NodeLabel.CIRCUIT_TOPOLOGY, _readable(tclass)), tclass


def test_every_registered_class_and_metric_has_a_curated_expansion():
    """I2 coverage: a topology added to TEMPLATES without a curated expansion would silently route
    to the distrusted vector fallback. Derive the assertion from the registry so that regresses
    LOUDLY at test time, not silently at projection time."""
    from openclaw_brain.knowledge.executable.templates import TEMPLATES
    for tclass, meta in TEMPLATES.items():
        assert _expansions(NodeLabel.CIRCUIT_TOPOLOGY, _readable(tclass)), f"no topology expansion: {tclass}"
        for metric in meta.get("metrics", []):
            assert _expansions(NodeLabel.PARAMETER, _metric_text(metric)), f"no metric expansion: {tclass}/{metric}"


@pytest.mark.asyncio
async def test_vos_hint_resolves_via_curated_text_no_vector_fallback():
    """G4: 'vos' (projection._metric_text('vos_v')) must resolve deterministically off the curated
    exact-match text path, never the distrusted vector fallback — even with a decoy duplicate node
    sharing the exact same canonical_name (the live graph's 'param_vos' stub, see resolver.py
    comment). The row order pins which duplicate wins; the encode_fn raising proves no embedding
    call is needed for the exact-match branch."""
    rows = [
        {"id": "Input-Referred Offset Voltage", "name": "Input-Referred Offset Voltage",
         "embedding": [1.0, 0.0]},
        {"id": "param_vos", "name": "Input-Referred Offset Voltage", "embedding": [0.9, 0.1]},
    ]
    store = _MockStore(text_rows=rows)

    def _no_encode(text):
        raise AssertionError("exact-match branch should not need embedding")

    r = GraphResolver(store, encode_fn=_no_encode)
    assert await r.resolve(NodeLabel.PARAMETER, "vos") == "Input-Referred Offset Voltage"
    assert all("queryNodes" not in q for q, _ in store.queries)


@pytest.mark.asyncio
async def test_a_vos_stays_uncurated_and_refuses():
    """G4: 'a vos' (the OTA5T Pelgrom fitted scaling EXPONENT) has no faithful graph node — the closest
    candidates (A_VTH / A_K mismatch coefficients) are a different physical quantity (units, not a
    dimensionless exponent). It is deliberately left OUT of _METRIC_EXPANSIONS, so it must fall through
    to the gated vector fallback and refuse when nothing clears the floor/margin (NO-PHANTOM)."""
    assert _expansions(NodeLabel.PARAMETER, "a vos") == []
    store = _MockStore(vec_rows=[{"id": "A_VTH", "score": 0.80}, {"id": "A_K", "score": 0.79}])
    r = GraphResolver(store, encode_fn=_encode_to([1.0, 0.0]))
    assert await r.resolve(NodeLabel.PARAMETER, "a vos") is None


@pytest.mark.asyncio
async def test_av0_hint_prefers_rich_open_loop_gain_over_thin_dc_duplicate():
    """F5 (2026-07-09 graph-to-graph relink, 3,359 new edges): a thin, structurally-disconnected
    'DC Open-Loop Gain' duplicate (id dc_open_loop_gain, degree 1 on the live graph -- its only edge
    is DERIVED_FROM 'DC Open-Loop Gain Increase', ZERO HAS_PARAMETER edges from any CircuitTopology)
    became resolver-visible (newly embedded / un-retracted) and, with the old phrase order
    ["dc open-loop gain", "open-loop gain"], won the exact-match tie-break over the graph's actual,
    richly-connected 'Open-Loop Gain' node (HAS_PARAMETER from "Two-Stage Op Amp" / "Operational
    Amplifier" / etc) -- even though both are exact matches, just for different curated phrases.
    'open-loop gain' is now PRIMARY so the established node wins regardless of row order or which
    thin duplicates the graph accumulates. encode_fn raises to prove no embedding call is needed."""
    rows = [
        {"id": "dc_open_loop_gain", "name": "DC Open-Loop Gain", "embedding": [1.0, 0.0]},
        {"id": "open_loop_gain_parameter", "name": "Open-Loop Gain", "embedding": [0.0, 1.0]},
    ]
    store = _MockStore(text_rows=rows)

    def _no_encode(text):
        raise AssertionError("exact-match branch should not need embedding")

    r = GraphResolver(store, encode_fn=_no_encode)
    assert await r.resolve(NodeLabel.PARAMETER, "av0") == "open_loop_gain_parameter"


@pytest.mark.asyncio
async def test_adm_hint_prefers_established_differential_voltage_gain_over_new_duplicate():
    """F5 (2026-07-09 relink): a new thin duplicate 'Differential Gain' (id new_differential_gain,
    degree 4 on the live graph -- its edges are all about a DIFFERENT context: cascode internal-node
    gain, HAS_PARAMETER to/from "Cascode Amplifier Gain Mechanism" / "Differential Voltage Gain at
    Cascode Nodes (A and B)" -- not the resistively-loaded differential pair adm_db characterizes)
    became resolver-visible and, with the old phrase order, won the exact-match tie-break over
    'Differential Voltage Gain' (degree 40; HAS_PARAMETER from "Differential Pair" / "Differential
    Amplifier" / etc, and already the target of 3 live GROUNDS edges from prior claim-cards).
    'differential voltage gain' is now PRIMARY so the established, already-grounded node wins."""
    rows = [
        {"id": "new_differential_gain", "name": "Differential Gain", "embedding": [1.0, 0.0]},
        {"id": "differential_voltage_gain", "name": "Differential Voltage Gain", "embedding": [0.0, 1.0]},
    ]
    store = _MockStore(text_rows=rows)

    def _no_encode(text):
        raise AssertionError("exact-match branch should not need embedding")

    r = GraphResolver(store, encode_fn=_no_encode)
    assert await r.resolve(NodeLabel.PARAMETER, "adm") == "differential_voltage_gain"


@pytest.mark.asyncio
async def test_cascode_current_mirror_hint_prefers_poor_mans_over_generic_decoy():
    """F5 (2026-07-09 relink): the bare 'cascode' catch-all phrase alone matches 167 live
    CircuitTopology nodes (far past the resolver's LIMIT 40), so whether the exact-match target and
    the generic decoy both land inside a given 40-row window is an order-dependent accident of
    Neo4j's (unordered) row return, not something the old phrase list controlled -- a latent drift
    risk even though it happened not to flip in this session. This pins the now-deterministic outcome
    directly: even with the generic umbrella 'Cascode Current Mirror' node (degree 73 live -- could
    mean any cascode-mirror bias variant) ALSO present as a candidate, 'poor man's cascode current
    mirror' -- the netlist-faithful match for the bare 2-diode self-biased stack _CASC_BODY renders
    (M1+M2 diode-stacked reference, both bias nodes reused directly by M3/M4; no resistor, no
    regulation, no wide-swing sizing -- Razavi's "poor man's" scheme) -- wins because it is now the
    PRIMARY curated phrase."""
    rows = [
        {"id": "cascode_current_mirror", "name": "Cascode Current Mirror", "embedding": [1.0, 0.0]},
        {"id": "poor_mans_cascode_current_mirror", "name": "Poor Man's Cascode Current Mirror",
         "embedding": [0.0, 1.0]},
    ]
    store = _MockStore(text_rows=rows)

    def _no_encode(text):
        raise AssertionError("exact-match branch should not need embedding")

    r = GraphResolver(store, encode_fn=_no_encode)
    assert (await r.resolve(NodeLabel.CIRCUIT_TOPOLOGY, "cascode current mirror nmos")
            == "poor_mans_cascode_current_mirror")


@pytest.mark.asyncio
async def test_text_candidates_query_orders_by_degree_then_name():
    """F7 (2026-07-10 near-cap sweep): the candidate-pool query had a LIMIT with NO ORDER BY, so for
    any hint whose true pool exceeds the cap, WHICH $lim-of-N rows Neo4j even returns was an
    unspecified, order-dependent accident of internal storage/scan order -- the exact same mechanism
    that flipped 3 live bindings, fixed in 44a4eb4 (bare 'cascode' matched 167 nodes with no ORDER
    BY). Root-fixed at the QUERY level: ORDER BY relationship degree DESC (established, well-
    integrated nodes surface first -- live-verified below to both preserve every then-flagged hint's
    binding AND correct a previously-hidden one, see test_rout_hint_...), then canonical_name ASC as
    a fully deterministic tiebreak, replacing row-order luck entirely. This pins the exact clause so
    a future edit can't silently drop it back to unordered."""
    store = _MockStore(text_rows=[])
    r = GraphResolver(store)
    await r._text_candidates(NodeLabel.PARAMETER, ["output resistance"])
    assert len(store.queries) == 1
    query, params = store.queries[0]
    assert "ORDER BY size([(n)--()|1]) DESC, n.canonical_name ASC" in query
    # ORDER BY must rank the pool BEFORE LIMIT truncates it, never the other way around.
    assert query.index("ORDER BY size([(n)--()|1]) DESC") < query.index("LIMIT $lim")
    assert params == {"phrases": ["output resistance"], "lim": 40}


@pytest.mark.asyncio
async def test_rout_hint_prefers_exact_output_resistance_over_scoped_decoy():
    """F7 (2026-07-10 near-cap sweep, live-verified): 'rout' (curated to the single phrase
    'output resistance') has an 85-node live Parameter pool -- over the LIMIT-40 cap. Before the
    ORDER BY fix, the unordered query STABLY excluded the exact-match target -- 'Output Resistance'
    (id output_resistance, degree 51 live: HAS_PARAMETER from Source Follower / Second Stage
    Amplifier / Noninverting Amplifier / etc, USES_EQUATION for both cascode Rout formulas) -- from
    its top-40 window in 5/5 repeated live reads (a fixed storage-order accident, not run-to-run
    flakiness -- exactly the kind of drift a schema change/reindex/write is not obligated to
    preserve, per this file's own G6/vos precedent). Resolution fell through to embedding-rank among
    the visible 40 and landed on 'Closed-Loop Output Resistance' (id
    auto_closed_loop_output_resistance, degree 2 live -- a narrow feedback-amplifier-specific
    quantity, not the generic small-signal Rout the bare curated phrase means). 'rout' is not wired
    to any live template metric today (grepped TEMPLATES' metrics lists) and carries zero live
    GROUNDS edges either way -- this was a pre-emptive catch, not a shipped bad edge. Query-level
    ORDER BY alone fixes it (no phrase re-curation needed: 'output resistance' was already the right
    primary phrase, it just couldn't reliably reach the window). This pins the corrected outcome at
    the resolve()-logic level: given both rows, the exact match wins outright, no embedding needed
    (encode_fn raises to prove it) -- see test_resolver_seed_hints_resolve_on_live_graph for the
    live-graph end-to-end pin."""
    rows = [
        {"id": "auto_closed_loop_output_resistance", "name": "Closed-Loop Output Resistance",
         "embedding": [1.0, 0.0]},
        {"id": "output_resistance", "name": "Output Resistance", "embedding": [0.0, 1.0]},
    ]
    store = _MockStore(text_rows=rows)

    def _no_encode(text):
        raise AssertionError("exact-match branch should not need embedding")

    r = GraphResolver(store, encode_fn=_no_encode)
    assert await r.resolve(NodeLabel.PARAMETER, "rout") == "output_resistance"


@pytest.mark.asyncio
async def test_vector_fallback_rejects_near_tie():
    # I2: two un-curated candidates both above the cosine floor but within the margin = a
    # confidently-wrong near-tie (the av0 "DC Loop Gain" 0.921 vs 0.912 case) -> NO edge.
    store = _MockStore(vec_rows=[{"id": "A", "score": 0.96}, {"id": "B", "score": 0.955}])  # cos .92 vs .91
    r = GraphResolver(store, encode_fn=_encode_to([1.0, 0.0]), vec_threshold=0.60, margin_min=0.03)
    assert await r.resolve(NodeLabel.PARAMETER, "uncurated metric") is None


@pytest.mark.asyncio
async def test_vector_fallback_accepts_clear_winner():
    # a clear winner (margin well over the runner-up) still resolves
    store = _MockStore(vec_rows=[{"id": "A", "score": 0.96}, {"id": "B", "score": 0.70}])  # cos .92 vs .40
    r = GraphResolver(store, encode_fn=_encode_to([1.0, 0.0]), vec_threshold=0.60, margin_min=0.03)
    assert await r.resolve(NodeLabel.PARAMETER, "uncurated metric") == "A"


# ── Neo4j-gated: the curated expansions must match the REAL graph vocabulary ──


@pytest.fixture
async def live_store():
    require_live_graph()
    from openclaw_brain.config import load_config
    from openclaw_brain.knowledge.graph.store import GraphStore
    s = GraphStore(load_config().neo4j)
    try:
        await s.connect()
    except Exception:
        pytest.skip("Neo4j not available")
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_resolver_seed_hints_resolve_on_live_graph(live_store):
    """The curated expansions are only useful if they actually substring-match real canonical_names.
    This resolves every seed hint against the live graph and asserts the expected node — and injects
    an encode_fn that RAISES, proving the seeds resolve via the curated text/exact path with NO
    embedding (so this stays fast and the curated coverage is genuinely exercised)."""
    from openclaw_brain.knowledge.executable.projection import _ID_FIELD

    def _no_encode(text):
        raise AssertionError(f"embedding path hit for {text!r}; the curated text path should cover seeds")

    r = GraphResolver(live_store, encode_fn=_no_encode)

    async def name_of(label, idv):
        idf = _ID_FIELD[label]
        rows = await live_store.run_read_query(
            f"MATCH (n:{label.value}) WHERE n.{idf} = $id RETURN n.canonical_name AS name", {"id": idv})
        return rows[0]["name"] if rows else None

    cases = [
        (NodeLabel.CIRCUIT_TOPOLOGY, _readable("miller_ota_2stage_nmos_in"), "Two-Stage Miller OTA"),
        (NodeLabel.CIRCUIT_TOPOLOGY, _readable("current_mirror_simple_nmos"), "NMOS Current Mirror"),
        (NodeLabel.PARAMETER, _metric_text("gbw_hz"), "Gain-Bandwidth Product"),
        (NodeLabel.PARAMETER, _metric_text("av0_db"), "Open-Loop Gain"),
        (NodeLabel.PARAMETER, _metric_text("pm_deg"), "Phase Margin"),
        (NodeLabel.PARAMETER, _metric_text("iout_a"), "Output Current"),
        # F7 (2026-07-10 near-cap sweep): "rout"/"gain" are curated (_METRIC_EXPANSIONS) but not
        # wired to any live template metric today -- generic, forward-looking vocabulary (see
        # resolver.py's own "broader metrics" comment). Both exact-match on the live graph now that
        # the LIMIT-40 window is ordered by degree DESC instead of an unordered accident. "rout" is
        # the live-verified F7 finding (mock-pinned in test_rout_hint_prefers_exact_output_
        # resistance_over_scoped_decoy): before this fix the exact match "Output Resistance" (degree
        # 51) was STABLY excluded from the unordered 40-row window in 5/5 live reads, landing on a
        # thin "Closed-Loop Output Resistance" decoy (degree 2) instead.
        (NodeLabel.PARAMETER, "rout", "Output Resistance"),
        (NodeLabel.PARAMETER, "gain", "Open-Loop Gain"),
    ]
    for label, hint, expected in cases:
        rid = await r.resolve(label, hint)
        assert rid is not None, f"seed hint {hint!r} did not resolve on the live graph"
        assert await name_of(label, rid) == expected, f"{hint!r} resolved to the wrong node"


@pytest.mark.asyncio
async def test_resolver_all_conquest_hints_bind_to_expected_node(live_store):
    """Expected-NODE guard for every conquest class + metric (the gap that let the ramp slope_vps ->
    'Slew Rate' confidently-wrong edge ship: the old guard only asserted an expansion EXISTS). Uses the
    REAL encoder because several metric hints (acl/tpd/vo/adm) resolve via embedding-rank among CONTAINS
    candidates rather than exact-match. `None` is the CORRECT, asserted outcome where the graph has no
    faithful target (the ramp slope is dV/dt = I/Cramp, NOT amplifier slew rate — no edge beats a wrong
    edge). Pins the verified bindings so a future expansion edit can't silently mis-resolve."""
    from openclaw_brain.knowledge.executable.projection import _ID_FIELD

    r = GraphResolver(live_store)   # real embedding path

    async def name_of(label, idv):
        if idv is None:
            return None
        idf = _ID_FIELD[label]
        rows = await live_store.run_read_query(
            f"MATCH (n:{label.value}) WHERE n.{idf} = $id RETURN n.canonical_name AS name", {"id": idv})
        return rows[0]["name"] if rows else None

    T, P = NodeLabel.CIRCUIT_TOPOLOGY, NodeLabel.PARAMETER
    cases = [
        (T, _readable("common_source_active_load_nmos"), "Active Load CS Stage"),
        (T, _readable("common_gate_nmos"), "CG Stage"),
        (T, _readable("source_follower_nmos"), "Source Follower"),
        (T, _readable("diff_pair_resistive_nmos"), "Basic Differential Pair"),
        (T, _readable("cascode_current_mirror_nmos"), "Poor Man's Cascode Current Mirror"),
        (T, _readable("ota_5t_nmos_in"), "Five-Transistor OTA"),
        (T, _readable("telescopic_cascode_ota_nmos_in"), "Telescopic Cascode OTA"),
        (T, _readable("folded_cascode_ota_nmos_in"), "NMOS Input Folded-Cascode Op Amp"),
        (T, _readable("regulated_cascode_nmos"), "Regulated Cascode"),
        (T, _readable("comparator_continuous_nmos"), "Comparator"),
        (T, _readable("cds_switched_cap_nmos"), "Correlated Double-Sampling Amplifier"),
        (T, _readable("single_slope_ramp_generator"), "Global Ramp Reference"),
        (T, _readable("column_pga_inverting_nmos"), "Programmable-Gain Amplifier"),
        (P, _metric_text("rin_ohm"), "Input Resistance"),
        (P, _metric_text("adm_db"), "Differential Voltage Gain"),
        (P, _metric_text("tpd_s"), "Propagation Delay"),
        (P, _metric_text("vo_v"), "DC Output Offset Voltage"),
        (P, _metric_text("acl_db"), "Closed-Loop Gain Ratio"),
        # slope_vps now binds to the FAITHFUL 'Ramp Slope' Parameter the CIS ingest added (it previously
        # had no faithful node -> None). The guard's point holds: a faithful node iff one exists, never
        # the wrong 'Slew Rate'. (Digital classes are NOT projected to the live graph — their projection
        # mechanism is unit-tested in test_executable_projection — so they aren't pinned here.)
        (P, _metric_text("slope_vps"), "Ramp Slope"),
        # G4: the 2026-07-03 NO-PHANTOM refusals. 'vos' now has a curated exact-match target;
        # 'a vos' stays uncurated (no faithful node — the Pelgrom fit is a dimensionless exponent,
        # the closest graph nodes are per-transistor mismatch COEFFICIENTS) and must still resolve
        # to None, same discipline as the slope_vps precedent above.
        (P, _metric_text("vos_v"), "Input-Referred Offset Voltage"),
        (P, _metric_text("a_vos"), None),
    ]
    for label, hint, expected in cases:
        rid = await r.resolve(label, hint)
        assert await name_of(label, rid) == expected, (
            f"{hint!r} resolved to {await name_of(label, rid)!r}, expected {expected!r}")
