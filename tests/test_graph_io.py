"""Tests for the ADR-044 D1 streaming JSONL graph export/import (export/graph_io.py)."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pytest
from tests.conftest import require_live_graph

from openclaw_brain.export.graph_io import (
    DEFAULT_PRIVATE_LABELS,
    FORMAT_VERSION,
    _clean_node_props,
    export_graph,
    import_graph,
)
from openclaw_brain.knowledge.graph.schema import NodeLabel, RelType
from openclaw_brain.knowledge.graph.store import GraphStore


# ── Fake store: mirrors the exact query SHAPES export_graph/import_graph issue
# (verified against graph/store.py's retracted-filter + pagination conventions)
# so these tests exercise the real translation logic, not a trivial stub. ──


def _classify_query(query: str) -> str:
    q = " ".join(query.split())
    if "WITH labels(n)[0] AS label, count(n) AS cnt" in q:
        return "count_nodes"
    if "WITH type(r) AS rel_type, count(r) AS cnt" in q:
        return "count_edges"
    if q.startswith("MATCH (n:"):
        return "page_nodes"
    if q.startswith("MATCH (a)-[r]->(b)"):
        return "page_edges"
    raise AssertionError(f"unrecognized query shape: {query!r}")


class FakeStore:
    """In-memory stand-in for GraphStore's run_read_query/write_batch surface."""

    def __init__(self, nodes: dict[str, list[dict]], edges: list[dict]):
        self._nodes = {label: [dict(n) for n in rows] for label, rows in nodes.items()}
        self._edges = [dict(e) for e in edges]
        self.queries: list[tuple[str, dict]] = []
        self.write_batch_calls: list[dict] = []

    async def run_read_query(self, query, params=None):
        params = params or {}
        self.queries.append((query, params))
        kind = _classify_query(query)
        exclude = set(params.get("exclude", []))

        if kind == "count_nodes":
            return [
                {"label": label, "cnt": len(rows)}
                for label, rows in self._nodes.items()
                if label not in exclude and rows
            ]
        if kind == "count_edges":
            counts: dict[str, int] = {}
            for e in self._edges:
                if e["source_label"] in exclude or e["target_label"] in exclude:
                    continue
                counts[e["rel_type"]] = counts.get(e["rel_type"], 0) + 1
            return [{"rel_type": rt, "cnt": c} for rt, c in counts.items()]
        if kind == "page_nodes":
            label = re.search(r"MATCH \(n:(\w+)\)", query).group(1)
            skip, limit = params["skip"], params["limit"]
            rows = self._nodes.get(label, [])
            return [{"props": dict(p)} for p in rows[skip:skip + limit]]
        if kind == "page_edges":
            skip, limit = params["skip"], params["limit"]
            filtered = [
                e for e in self._edges
                if e["source_label"] not in exclude and e["target_label"] not in exclude
            ]
            page = filtered[skip:skip + limit]
            return [
                {
                    "rel_type": e["rel_type"],
                    "source_label": e["source_label"],
                    "a_props": dict(e["source_props"]),
                    "target_label": e["target_label"],
                    "b_props": dict(e["target_props"]),
                    "props": dict(e.get("props") or {}),
                }
                for e in page
            ]
        raise AssertionError(kind)  # pragma: no cover

    async def write_batch(self, nodes=None, updates=None, edges=None):
        self.write_batch_calls.append({
            "nodes": list(nodes or []), "updates": list(updates or []), "edges": list(edges or []),
        })


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def sample_store() -> FakeStore:
    nodes = {
        "Concept": [
            {"concept_id": "c1", "canonical_name": "CS Amp", "confidence": 0.9,
             "embedding": [0.1, 0.2], "embedding_model": "qwen3-embed"},
            {"concept_id": "c2", "canonical_name": "Voltage Gain", "confidence": 0.8},
        ],
        "SourceChunk": [
            {"chunk_id": "sc1", "source_id": "s1", "raw_text_hash": "sha256:abc",
             "text_preview": "This is a verbatim excerpt from a copyrighted textbook."},
        ],
        "Source": [
            {"source_id": "s1", "title": "Razavi Analog"},
        ],
        "Memory": [
            {"memory_id": "m1", "content": "personal memory, never leaves home"},
        ],
    }
    edges = [
        {"rel_type": "RELATES_TO", "source_label": "Concept", "source_props": {"concept_id": "c1"},
         "target_label": "Concept", "target_props": {"concept_id": "c2"},
         "props": {"rationale": "gain depends on bias", "confidence": 0.7}},
        {"rel_type": "EXTRACTED_FROM", "source_label": "SourceChunk",
         "source_props": {"chunk_id": "sc1"},
         "target_label": "Source", "target_props": {"source_id": "s1"},
         "props": {"rationale": "chunk from pages 1-2", "confidence": 1.0}},
        {"rel_type": "RECORDED", "source_label": "Memory", "source_props": {"memory_id": "m1"},
         "target_label": "Concept", "target_props": {"concept_id": "c1"},
         "props": {}},
    ]
    return FakeStore(nodes, edges)


# ── export_graph ──


async def test_export_graph_header_integrity(sample_store, tmp_path):
    out = tmp_path / "graph.jsonl"
    header = await export_graph(sample_store, out)
    lines = _read_jsonl(out)

    assert lines[0] == header
    assert header["format_version"] == FORMAT_VERSION
    datetime.fromisoformat(header["exported_at"])  # parses as ISO-8601
    assert header["exclude_labels"] == sorted(DEFAULT_PRIVATE_LABELS)
    assert header["include_embeddings"] is False

    node_lines = [ln for ln in lines[1:] if ln["kind"] == "node"]
    edge_lines = [ln for ln in lines[1:] if ln["kind"] == "edge"]
    assert sum(header["counts"]["nodes"].values()) == len(node_lines)
    assert sum(header["counts"]["edges"].values()) == len(edge_lines)


async def test_export_graph_excludes_private_labels_and_their_edges(sample_store, tmp_path):
    out = tmp_path / "graph.jsonl"
    await export_graph(sample_store, out)  # default exclude_labels = DEFAULT_PRIVATE_LABELS
    lines = _read_jsonl(out)

    node_labels = {ln["label"] for ln in lines[1:] if ln["kind"] == "node"}
    assert "Memory" not in node_labels

    edge_rel_types = {ln["rel_type"] for ln in lines[1:] if ln["kind"] == "edge"}
    assert "RECORDED" not in edge_rel_types  # Memory -> Concept edge dropped with its endpoint

    # SourceChunk is NOT in DEFAULT_PRIVATE_LABELS — it carries citation metadata,
    # explicitly allowed company-side (ADR-044 D1) — so it and its edge survive.
    assert "SourceChunk" in node_labels
    assert "EXTRACTED_FROM" in edge_rel_types


async def test_export_graph_include_private_disables_exclusion(sample_store, tmp_path):
    out = tmp_path / "graph.jsonl"
    await export_graph(sample_store, out, exclude_labels=set())
    lines = _read_jsonl(out)

    node_labels = {ln["label"] for ln in lines[1:] if ln["kind"] == "node"}
    assert "Memory" in node_labels
    edge_rel_types = {ln["rel_type"] for ln in lines[1:] if ln["kind"] == "edge"}
    assert "RECORDED" in edge_rel_types


async def test_export_graph_strips_embeddings_by_default(sample_store, tmp_path):
    out = tmp_path / "graph.jsonl"
    await export_graph(sample_store, out)
    lines = _read_jsonl(out)
    c1 = next(ln for ln in lines[1:] if ln.get("label") == "Concept" and ln["props"]["concept_id"] == "c1")
    assert "embedding" not in c1["props"]
    assert "embedding_model" not in c1["props"]


async def test_export_graph_include_embeddings_flag_keeps_them(sample_store, tmp_path):
    out = tmp_path / "graph.jsonl"
    await export_graph(sample_store, out, include_embeddings=True)
    lines = _read_jsonl(out)
    c1 = next(ln for ln in lines[1:] if ln.get("label") == "Concept" and ln["props"]["concept_id"] == "c1")
    assert c1["props"]["embedding"] == [0.1, 0.2]
    assert c1["props"]["embedding_model"] == "qwen3-embed"


async def test_export_graph_strips_source_chunk_text_preview_always(sample_store, tmp_path):
    # Even with private labels included AND embeddings included, text_preview
    # (verbatim source text, per knowledge/pipeline.py::_register_chunk) must
    # never survive — it is content, not the citation metadata ADR-044 D1 allows.
    out = tmp_path / "graph.jsonl"
    await export_graph(sample_store, out, exclude_labels=set(), include_embeddings=True)
    lines = _read_jsonl(out)
    sc = next(ln for ln in lines[1:] if ln.get("label") == "SourceChunk")
    assert "text_preview" not in sc["props"]
    assert sc["props"]["raw_text_hash"] == "sha256:abc"  # pointer/hash — allowed
    assert sc["props"]["chunk_id"] == "sc1"


def test_clean_node_props_strips_embedding_and_source_chunk_text():
    props = {"concept_id": "c1", "embedding": [1.0], "embedding_model": "x"}
    assert _clean_node_props("Concept", props, include_embeddings=False) == {"concept_id": "c1"}
    assert _clean_node_props("Concept", props, include_embeddings=True) == props

    sc_props = {"chunk_id": "sc1", "text_preview": "verbatim", "raw_text_hash": "h"}
    cleaned = _clean_node_props("SourceChunk", sc_props, include_embeddings=True)
    assert cleaned == {"chunk_id": "sc1", "raw_text_hash": "h"}


async def test_export_graph_queries_filter_retracted_nodes(sample_store, tmp_path):
    out = tmp_path / "graph.jsonl"
    await export_graph(sample_store, out)
    assert sample_store.queries  # sanity: queries were actually issued
    for query, _params in sample_store.queries:
        assert "retracted" in query


async def test_export_graph_pages_within_a_label(sample_store, tmp_path):
    out = tmp_path / "graph.jsonl"
    await export_graph(sample_store, out, page_size=1)
    lines = _read_jsonl(out)
    concept_lines = [ln for ln in lines[1:] if ln.get("label") == "Concept"]
    assert {ln["props"]["concept_id"] for ln in concept_lines} == {"c1", "c2"}

    page_node_calls = [q for q, _p in sample_store.queries if "MATCH (n:Concept)" in q]
    assert len(page_node_calls) >= 2  # streamed across multiple SKIP/LIMIT pages, not one shot


async def test_export_graph_progress_callback(sample_store, tmp_path):
    out = tmp_path / "graph.jsonl"
    calls: list[tuple] = []
    await export_graph(sample_store, out, on_progress=lambda *a: calls.append(a))
    assert ("SourceChunk", 1, 1) in calls
    assert ("Source", 1, 1) in calls
    assert ("Concept", 2, 2) in calls
    assert any(c[0] == "<edges>" for c in calls)


# ── import_graph ──


async def test_import_graph_replays_nodes_and_edges_via_write_batch(sample_store, tmp_path):
    out = tmp_path / "graph.jsonl"
    await export_graph(sample_store, out, exclude_labels=set())  # full round trip

    sink = FakeStore({}, [])
    counts = await import_graph(sink, out, batch_size=2)

    assert counts["nodes_skipped"] == 0
    assert counts["edges_skipped"] == 0
    assert counts["nodes_imported"] == 5   # c1, c2, sc1, s1, m1
    assert counts["edges_imported"] == 3

    total_nodes_written = sum(len(c["nodes"]) for c in sink.write_batch_calls)
    total_edges_written = sum(len(c["edges"]) for c in sink.write_batch_calls)
    assert total_nodes_written == counts["nodes_imported"]
    assert total_edges_written == counts["edges_imported"]

    all_nodes = [n for call in sink.write_batch_calls for n in call["nodes"]]
    concept_node = next(n for n in all_nodes if n["id_value"] == "c1")
    assert concept_node["label"] == NodeLabel.CONCEPT
    assert concept_node["id_field"] == "concept_id"
    assert concept_node["properties"]["canonical_name"] == "CS Amp"

    all_edges = [e for call in sink.write_batch_calls for e in call["edges"]]
    relates = next(e for e in all_edges if e["rel_type"] == RelType.RELATES_TO)
    assert relates["source_label"] == NodeLabel.CONCEPT
    assert relates["source_id_value"] == "c1"
    assert relates["target_id_value"] == "c2"


async def test_export_graph_includes_regularity_label(tmp_path):
    """Law-tier graph representation (spec 2026-07-04) — Regularity is company-shareable
    knowledge (not in DEFAULT_PRIVATE_LABELS) and needs no static enumeration here: labels are
    discovered dynamically from the live graph, so this only needs `NodeLabel.REGULARITY` +
    `GraphStore._id_field_for_label` registered (graph/store.py) for the id-field lookup to
    resolve instead of silently dropping the node/edge."""
    nodes = {
        "Regularity": [
            {"law_id": "law-1", "topology_class": "current_mirror_simple_nmos",
             "metric": "iout_a", "knob": "Vout", "quant_kind": "direction", "status": "law"},
        ],
        "ClaimCard": [
            {"claim_id": "sha256:abc:cm_iout", "verdict": "VERIFIED"},
        ],
    }
    edges = [
        {"rel_type": "SUPPORTED_BY", "source_label": "Regularity", "source_props": {"law_id": "law-1"},
         "target_label": "ClaimCard", "target_props": {"claim_id": "sha256:abc:cm_iout"},
         "props": {}},
    ]
    store = FakeStore(nodes, edges)
    out = tmp_path / "graph.jsonl"
    header = await export_graph(store, out, exclude_labels=set())
    lines = _read_jsonl(out)

    assert header["counts"]["nodes"]["Regularity"] == 1
    reg_line = next(ln for ln in lines[1:] if ln.get("label") == "Regularity")
    assert reg_line["props"]["law_id"] == "law-1"

    edge_line = next(ln for ln in lines[1:] if ln.get("rel_type") == "SUPPORTED_BY")
    assert edge_line["source_label"] == "Regularity"
    assert edge_line["source_id_field"] == "law_id"
    assert edge_line["source_id"] == "law-1"
    assert edge_line["target_id_field"] == "claim_id"

    # Round-trips through import_graph (idempotent MERGE) without being skipped.
    sink = FakeStore({}, [])
    counts = await import_graph(sink, out)
    assert counts["nodes_skipped"] == 0
    assert counts["edges_skipped"] == 0
    all_nodes = [n for call in sink.write_batch_calls for n in call["nodes"]]
    reg_node = next(n for n in all_nodes if n["id_value"] == "law-1")
    assert reg_node["label"] == NodeLabel.REGULARITY
    assert reg_node["id_field"] == "law_id"


async def test_export_graph_round_trips_symbolic_anchor_structure(tmp_path):
    """Regression for the 2026-07-28 backup data-loss incident: SymbolicDerivation /
    DERIVED_BY / MEASURED_BY were created via graph-surgery Cypher on 2026-07-20 without
    NodeLabel/RelType registration, so export_graph silently dropped all 76 DERIVED_BY edges
    (unrecognized endpoint label) and import_graph would have dropped the 76 nodes AND the 11
    MEASURED_BY edges (RelType gate) — a restore would have lost the entire triple-anchor
    structure while reporting success. This test pins the full round trip."""
    nodes = {
        "Equation": [
            {"equation_id": "eq-1", "canonical_latex": "A_v = -g_m R_D"},
        ],
        "SymbolicDerivation": [
            {"derivation_id": "sd-1", "tool": "lcapy", "match_class": "exact",
             "quantity": "gain"},
        ],
        "ClaimCard": [
            {"claim_id": "sha256:abc:cs_gain", "verdict": "VERIFIED"},
        ],
    }
    edges = [
        {"rel_type": "DERIVED_BY", "source_label": "Equation",
         "source_props": {"equation_id": "eq-1"},
         "target_label": "SymbolicDerivation", "target_props": {"derivation_id": "sd-1"},
         "props": {}},
        {"rel_type": "MEASURED_BY", "source_label": "Equation",
         "source_props": {"equation_id": "eq-1"},
         "target_label": "ClaimCard", "target_props": {"claim_id": "sha256:abc:cs_gain"},
         "props": {}},
    ]
    store = FakeStore(nodes, edges)
    out = tmp_path / "graph.jsonl"
    header = await export_graph(store, out, exclude_labels=set())
    lines = _read_jsonl(out)

    assert header["counts"]["nodes"]["SymbolicDerivation"] == 1
    assert header["counts"]["edges"]["DERIVED_BY"] == 1

    # The exact defect: the edge BODY must contain what the header promises.
    body_edges = [ln for ln in lines[1:] if ln.get("kind") == "edge"]
    by_type = {e["rel_type"]: e for e in body_edges}
    assert set(by_type) == {"DERIVED_BY", "MEASURED_BY"}
    assert by_type["DERIVED_BY"]["target_id_field"] == "derivation_id"
    assert by_type["DERIVED_BY"]["target_id"] == "sd-1"
    assert by_type["MEASURED_BY"]["target_id_field"] == "claim_id"

    # Restore path: nothing skipped, node label/id resolve to the registered enum.
    sink = FakeStore({}, [])
    counts = await import_graph(sink, out)
    assert counts["nodes_skipped"] == 0
    assert counts["edges_skipped"] == 0
    all_nodes = [n for call in sink.write_batch_calls for n in call["nodes"]]
    sd_node = next(n for n in all_nodes if n["id_value"] == "sd-1")
    assert sd_node["label"] == NodeLabel.SYMBOLIC_DERIVATION
    assert sd_node["id_field"] == "derivation_id"
    all_edges = [e for call in sink.write_batch_calls for e in call["edges"]]
    assert {e["rel_type"] for e in all_edges} == {RelType.DERIVED_BY, RelType.MEASURED_BY}


def test_default_private_labels_includes_learner_model():
    """S5_LEARNER_MODEL_DESIGN.md §4.1/§6.1 non-negotiable rule: Learner/Assessment must be
    excluded from the company export in the SAME change as the schema addition."""
    assert {"Learner", "Assessment"} <= DEFAULT_PRIVATE_LABELS


async def test_export_graph_excludes_learner_model_labels(tmp_path):
    """Learner/Assessment (S5 learner model) must never appear in a company export artifact —
    mirrors test_export_graph_excludes_private_labels_and_their_edges's pattern, isolated in its
    own FakeStore so it doesn't perturb the shared sample_store fixture's other assertions."""
    nodes = {
        "Learner": [{"learner_id": "rick"}],
        "Assessment": [{"assessment_id": "assess_1", "learner_id": "rick", "verdict": "understood"}],
        "Concept": [{"concept_id": "c1", "canonical_name": "CS Amp"}],
    }
    edges = [
        {"rel_type": "UNDERSTANDS", "source_label": "Learner", "source_props": {"learner_id": "rick"},
         "target_label": "Concept", "target_props": {"concept_id": "c1"}, "props": {}},
        {"rel_type": "ASSESSES", "source_label": "Assessment",
         "source_props": {"assessment_id": "assess_1"},
         "target_label": "Concept", "target_props": {"concept_id": "c1"}, "props": {}},
    ]
    store = FakeStore(nodes, edges)
    out = tmp_path / "graph.jsonl"
    await export_graph(store, out)  # default exclude_labels = DEFAULT_PRIVATE_LABELS
    lines = _read_jsonl(out)

    node_labels = {ln["label"] for ln in lines[1:] if ln["kind"] == "node"}
    assert "Learner" not in node_labels
    assert "Assessment" not in node_labels
    assert "Concept" in node_labels  # sanity: exclusion isn't dropping everything

    edge_rel_types = {ln["rel_type"] for ln in lines[1:] if ln["kind"] == "edge"}
    assert "UNDERSTANDS" not in edge_rel_types
    assert "ASSESSES" not in edge_rel_types


async def test_import_graph_flushes_in_batches(tmp_path):
    lines = [{"format_version": FORMAT_VERSION, "exported_at": "now", "exclude_labels": [],
              "include_embeddings": False, "counts": {"nodes": {"Concept": 7}, "edges": {}}}]
    for i in range(7):
        lines.append({"kind": "node", "label": "Concept",
                       "props": {"concept_id": f"c{i}", "canonical_name": f"C{i}"}})
    in_path = tmp_path / "in.jsonl"
    in_path.write_text("\n".join(json.dumps(ln) for ln in lines) + "\n")

    sink = FakeStore({}, [])
    counts = await import_graph(sink, in_path, batch_size=3)
    assert counts["nodes_imported"] == 7
    sizes = [len(c["nodes"]) for c in sink.write_batch_calls]
    assert sizes == [3, 3, 1]


async def test_import_graph_skips_malformed_and_invalid_lines(tmp_path):
    header = {"format_version": FORMAT_VERSION, "exported_at": "now", "exclude_labels": [],
              "include_embeddings": False, "counts": {"nodes": {}, "edges": {}}}
    good_node = {"kind": "node", "label": "Concept",
                 "props": {"concept_id": "c1", "canonical_name": "C1"}}
    bad_label_node = {"kind": "node", "label": "NotARealLabel", "props": {"foo": "bar"}}
    missing_id_node = {"kind": "node", "label": "Concept", "props": {"canonical_name": "no id"}}
    good_edge = {"kind": "edge", "rel_type": "RELATES_TO", "source_label": "Concept",
                 "source_id_field": "concept_id", "source_id": "c1",
                 "target_label": "Concept", "target_id_field": "concept_id", "target_id": "c1",
                 "props": {}}
    bad_rel_edge = {"kind": "edge", "rel_type": "NOT_A_REL_TYPE", "source_label": "Concept",
                    "source_id_field": "concept_id", "source_id": "c1",
                    "target_label": "Concept", "target_id_field": "concept_id", "target_id": "c1",
                    "props": {}}
    missing_target_edge = {"kind": "edge", "rel_type": "RELATES_TO", "source_label": "Concept",
                           "source_id_field": "concept_id", "source_id": "c1",
                           "target_label": "Concept", "target_id_field": "concept_id",
                           "target_id": "", "props": {}}

    raw_lines = [
        json.dumps(header), "{not valid json", json.dumps(good_node),
        json.dumps(bad_label_node), json.dumps(missing_id_node),
        json.dumps(good_edge), json.dumps(bad_rel_edge), json.dumps(missing_target_edge),
    ]
    in_path = tmp_path / "in.jsonl"
    in_path.write_text("\n".join(raw_lines) + "\n")

    sink = FakeStore({}, [])
    counts = await import_graph(sink, in_path)
    assert counts["nodes_imported"] == 1
    assert counts["nodes_skipped"] == 2
    assert counts["edges_imported"] == 1
    assert counts["edges_skipped"] == 2


async def test_import_graph_rejects_unsupported_format_version(tmp_path):
    header = {"format_version": FORMAT_VERSION + 1, "exported_at": "now",
              "exclude_labels": [], "include_embeddings": False, "counts": {}}
    in_path = tmp_path / "in.jsonl"
    in_path.write_text(json.dumps(header) + "\n")

    sink = FakeStore({}, [])
    with pytest.raises(ValueError, match="format_version"):
        await import_graph(sink, in_path)


async def test_import_graph_progress_callback(sample_store, tmp_path):
    out = tmp_path / "graph.jsonl"
    await export_graph(sample_store, out, exclude_labels=set())
    sink = FakeStore({}, [])
    calls: list[tuple] = []
    await import_graph(sink, out, batch_size=2, on_progress=lambda *a: calls.append(a))
    assert calls
    assert calls[-1][0] == "import"


# ── Live Neo4j round trip (skip-gated; kept small) ──


@pytest.fixture
async def live_store():
    require_live_graph()
    from openclaw_brain.config import load_config

    config = load_config()
    s = GraphStore(config.neo4j)
    try:
        await s.connect()
    except Exception:
        pytest.skip("Neo4j not available")
    yield s
    async with await s._session() as session:
        await session.run(
            "MATCH (n) WHERE n.concept_id STARTS WITH 'test_giotest_' DETACH DELETE n"
        )
    await s.close()


async def test_export_then_import_round_trip_live(live_store, tmp_path):
    # Export side: exercises the real Cypher (elementId ordering, retracted
    # filter, count aggregates) against a real Neo4j 5 server — read-only, so
    # safe to run against the shared graph. Assertion scope stays small (just
    # the two nodes this test creates) even though the scan covers whatever
    # else is present — export_graph has no id-prefix filter; it's a
    # full-graph tool by design.
    await live_store.merge_node(
        NodeLabel.CONCEPT, "concept_id", "test_giotest_export_a",
        {"concept_id": "test_giotest_export_a", "canonical_name": "GIO Export A"},
    )
    await live_store.merge_node(
        NodeLabel.CONCEPT, "concept_id", "test_giotest_export_b",
        {"concept_id": "test_giotest_export_b", "canonical_name": "GIO Export B"},
    )
    await live_store.merge_edge(
        NodeLabel.CONCEPT, "concept_id", "test_giotest_export_a",
        NodeLabel.CONCEPT, "concept_id", "test_giotest_export_b",
        RelType.RELATES_TO, {"rationale": "gio round trip", "confidence": 0.6},
    )

    out = tmp_path / "live_export.jsonl"
    await export_graph(live_store, out)
    lines = _read_jsonl(out)
    node_ids = {
        ln["props"].get("concept_id") for ln in lines[1:]
        if ln.get("kind") == "node" and ln.get("label") == "Concept"
    }
    assert {"test_giotest_export_a", "test_giotest_export_b"} <= node_ids

    # Import side: a small, hand-built artifact with a DISTINCT new id,
    # replayed against real Neo4j via write_batch/MERGE, then replayed AGAIN
    # to prove MERGE idempotency (no duplicate node created).
    header = {"format_version": FORMAT_VERSION, "exported_at": "now",
              "exclude_labels": [], "include_embeddings": False, "counts": {}}
    node_line = {"kind": "node", "label": "Concept",
                 "props": {"concept_id": "test_giotest_import_x", "canonical_name": "GIO Import X"}}
    in_path = tmp_path / "live_import.jsonl"
    in_path.write_text(json.dumps(header) + "\n" + json.dumps(node_line) + "\n")

    await import_graph(live_store, in_path)
    node = await live_store.get_node(NodeLabel.CONCEPT, "concept_id", "test_giotest_import_x")
    assert node is not None and node["canonical_name"] == "GIO Import X"

    await import_graph(live_store, in_path)  # re-import — MERGE must not duplicate
    count_rows = await live_store.run_read_query(
        "MATCH (n:Concept {concept_id: $id}) RETURN count(n) AS cnt",
        {"id": "test_giotest_import_x"},
    )
    assert count_rows[0]["cnt"] == 1
