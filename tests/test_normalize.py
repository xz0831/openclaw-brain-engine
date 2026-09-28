"""Tests for reasoning output normalization (EXO/local model schema compat)."""

from __future__ import annotations

import json
import logging

import pytest

from openclaw_brain.knowledge.reasoning.normalize import (
    extract_json,
    normalize_graph_delta,
)
from openclaw_brain.knowledge.graph.schema import GraphDelta


# ── Qwen3.5-style output (field name deviations) ──


QWEN_RAW = {
    "nodes": [  # should be "new_nodes"
        {
            "id": "folded_cascode_ota",  # should be "proposed_id"
            "name": "Folded Cascode OTA",  # should be "canonical_name"
            "type": "CircuitTopology",  # should be "label"
            "description": "A high-gain amplifier topology",
            "domain": "analog_circuits",
            "layer": 2,  # should be "knowledge_layer"
            "confidence": 0.9,
            "properties": {"function": "amplifier"},
            "evidence_chunk_ids": ["chunk_001"],
            # "reasoning" is missing — should be synthesized
        }
    ],
    "edges": [  # should be "new_edges"
        {
            "source": "folded_cascode_ota",  # should be "source_ref"
            "target": "differential_pair",  # should be "target_ref"
            "type": "SUB_BLOCK",  # should be "relationship_type"
            "rationale": "The differential pair is the input stage",
            "confidence": 0.85,
        }
    ],
    "insights": [
        {
            "title": "Folded cascode trades headroom for gain",  # should be "statement"
            "concepts": ["folded_cascode_ota", "headroom"],  # should be "related_concept_ids"
            "confidence": 0.7,
        }
    ],
}


def test_qwen_field_remap():
    """Qwen3.5-style output with renamed fields should validate."""
    normalized = normalize_graph_delta(QWEN_RAW)
    delta = GraphDelta(**normalized)

    assert len(delta.new_nodes) == 1
    node = delta.new_nodes[0]
    assert node.proposed_id == "folded_cascode_ota"
    assert node.canonical_name == "Folded Cascode OTA"
    assert node.label.value == "CircuitTopology"
    assert node.knowledge_layer == 2
    assert node.reasoning  # should be auto-generated

    assert len(delta.new_edges) == 1
    edge = delta.new_edges[0]
    assert edge.source_ref == "folded_cascode_ota"
    assert edge.target_ref == "differential_pair"
    assert edge.relationship_type.value == "SUB_BLOCK"

    assert len(delta.insights) == 1
    insight = delta.insights[0]
    assert "trades headroom" in insight.statement
    assert "folded_cascode_ota" in insight.related_concept_ids


# ── GLM-5-style output (insights as strings) ──


GLM5_RAW = {
    "new_nodes": [
        {
            "proposed_id": "pseudo_cascode",
            "label": "CircuitTopology",
            "canonical_name": "Pseudo-Cascode",
            "description": "A leakage mitigation technique",
            "confidence": 0.85,
            "reasoning": "Novel topology from the paper",
        }
    ],
    "new_edges": [],
    "insights": [
        "The pseudo-cascode technique trades area for leakage mitigation",
        "In 28nm FD-SOI, pico-Ampere leakage currents dominate bias design",
    ],
}


def test_glm5_string_insights():
    """GLM-5-style plain string insights should be wrapped into InsightProposal."""
    normalized = normalize_graph_delta(GLM5_RAW)
    delta = GraphDelta(**normalized)

    assert len(delta.insights) == 2
    assert delta.insights[0].statement == "The pseudo-cascode technique trades area for leakage mitigation"
    assert delta.insights[0].related_concept_ids == []
    assert delta.insights[0].confidence == 0.5
    assert delta.insights[1].statement.startswith("In 28nm")


# ── Mixed field names (some canonical, some alternative) ──


MIXED_RAW = {
    "new_nodes": [
        {
            "proposed_id": "cds_technique",
            "label": "Concept",
            "canonical_name": "Correlated Double Sampling",
            "reasoning": "CDS is a fundamental CIS readout technique",
        }
    ],
    "new_edges": [
        {
            "source_ref": "cds_technique",  # canonical
            "target": "ktc_noise",  # alternative
            "relationship_type": "SOLVES_PROBLEM",  # canonical
            "reason": "CDS cancels kTC reset noise",  # alternative for rationale
        }
    ],
    "updated_nodes": [
        {
            "id": "existing_node_123",  # alternative for existing_node_id
            "type": "Concept",  # alternative for label
            "changes": {"confidence": 0.95},  # alternative for updates
            "reason": "Higher confidence after second reference",  # alternative for reasoning
        }
    ],
    "insights": [],
}


def test_mixed_field_names():
    """Mix of canonical and alternative field names should all normalize."""
    normalized = normalize_graph_delta(MIXED_RAW)
    delta = GraphDelta(**normalized)

    assert len(delta.new_edges) == 1
    edge = delta.new_edges[0]
    assert edge.source_ref == "cds_technique"
    assert edge.target_ref == "ktc_noise"
    assert edge.rationale == "CDS cancels kTC reset noise"

    assert len(delta.updated_nodes) == 1
    update = delta.updated_nodes[0]
    assert update.existing_node_id == "existing_node_123"
    assert update.reasoning == "Higher confidence after second reference"


# ── Empty / minimal output ──


def test_empty_delta():
    """Empty dict should produce valid empty GraphDelta."""
    normalized = normalize_graph_delta({})
    delta = GraphDelta(**normalized)
    assert delta.new_nodes == []
    assert delta.new_edges == []


# ── JSON extraction ──


def test_extract_json_plain():
    raw = '{"new_nodes": [], "new_edges": []}'
    result = extract_json(raw)
    assert result == {"new_nodes": [], "new_edges": []}


def test_extract_json_markdown_fenced():
    raw = '```json\n{"new_nodes": [], "insights": ["test"]}\n```'
    result = extract_json(raw)
    assert result["insights"] == ["test"]


def test_extract_json_with_preamble():
    raw = 'Here is the result:\n\n{"new_nodes": [{"proposed_id": "x"}]}'
    result = extract_json(raw)
    assert len(result["new_nodes"]) == 1


def test_extract_json_invalid_raises():
    with pytest.raises(ValueError, match="Could not extract JSON"):
        extract_json("This is not JSON at all")


# ── Node proposed_id normalization ──


def test_proposed_id_snake_case():
    """proposed_id should be normalized to lowercase snake_case."""
    raw = {
        "new_nodes": [{
            "id": "Folded-Cascode OTA",
            "label": "CircuitTopology",
            "name": "Folded Cascode OTA",
            "reasoning": "test",
        }]
    }
    normalized = normalize_graph_delta(raw)
    delta = GraphDelta(**normalized)
    assert delta.new_nodes[0].proposed_id == "folded_cascode_ota"


# ── Label normalization ──


def test_unknown_label_maps_to_concept():
    """Non-standard labels like 'Model', 'Method' should map to Concept."""
    raw = {
        "new_nodes": [
            {
                "proposed_id": "weather_model",
                "label": "Model",  # not a valid NodeLabel
                "canonical_name": "Weather Prediction Model",
                "reasoning": "A machine learning model",
            },
            {
                "proposed_id": "fusion_method",
                "label": "Method",
                "canonical_name": "Data Fusion Method",
                "reasoning": "Technique for combining data",
            },
            {
                "proposed_id": "rmse_metric",
                "label": "Metric",  # should map to Parameter
                "canonical_name": "RMSE",
                "reasoning": "Error measurement",
            },
        ]
    }
    normalized = normalize_graph_delta(raw)
    delta = GraphDelta(**normalized)
    assert delta.new_nodes[0].label.value == "Concept"
    assert delta.new_nodes[1].label.value == "Concept"
    assert delta.new_nodes[2].label.value == "Parameter"


def test_missing_label_defaults_to_concept():
    """Nodes without a label field should default to Concept."""
    raw = {
        "new_nodes": [{
            "proposed_id": "test_node",
            "canonical_name": "Test Node",
            "reasoning": "test",
        }]
    }
    normalized = normalize_graph_delta(raw)
    delta = GraphDelta(**normalized)
    assert delta.new_nodes[0].label.value == "Concept"


def test_case_insensitive_label():
    """Labels should match case-insensitively."""
    raw = {
        "new_nodes": [{
            "proposed_id": "ota",
            "label": "circuit_topology",  # lowercase underscore
            "canonical_name": "OTA",
            "reasoning": "test",
        }]
    }
    normalized = normalize_graph_delta(raw)
    delta = GraphDelta(**normalized)
    assert delta.new_nodes[0].label.value == "CircuitTopology"


def test_unknown_rel_type_maps_to_relates_to():
    """Non-standard rel types should map to RELATES_TO."""
    raw = {
        "new_edges": [{
            "source_ref": "a",
            "target_ref": "b",
            "relationship_type": "IMPROVES",  # not standard
            "rationale": "A improves B",
            "confidence": 0.8,
        }]
    }
    normalized = normalize_graph_delta(raw)
    delta = GraphDelta(**normalized)
    assert delta.new_edges[0].relationship_type.value == "EVOLVES_TO"


# ── Edge reinforcement normalization ──


# ── Double-encoding tolerance (claude-sonnet-5 quirk) ──────────────────


def test_new_nodes_as_json_string_is_recovered():
    """A frontier model that double-encodes: `new_nodes` arrives as a JSON string containing
    the list, instead of an already-parsed list."""
    raw = {
        "new_nodes": json.dumps([{
            "proposed_id": "cds_technique",
            "label": "Concept",
            "canonical_name": "Correlated Double Sampling",
            "reasoning": "CDS is a fundamental CIS readout technique",
        }]),
        "new_edges": [],
    }
    normalized = normalize_graph_delta(raw)
    delta = GraphDelta(**normalized)
    assert len(delta.new_nodes) == 1
    assert delta.new_nodes[0].proposed_id == "cds_technique"


def test_new_edges_as_json_string_is_recovered():
    """Same double-encoding tolerance for `new_edges`."""
    raw = {
        "new_nodes": [],
        "new_edges": json.dumps([{
            "source_ref": "a",
            "target_ref": "b",
            "relationship_type": "DEPENDS_ON",
            "rationale": "A depends on B",
            "confidence": 0.8,
        }]),
    }
    normalized = normalize_graph_delta(raw)
    delta = GraphDelta(**normalized)
    assert len(delta.new_edges) == 1
    assert delta.new_edges[0].source_ref == "a"
    assert delta.new_edges[0].target_ref == "b"


def test_whole_payload_as_json_string_is_recovered():
    """Belt-and-suspenders: the entire raw payload itself arrives as a JSON string rather than
    an already-parsed dict."""
    raw = json.dumps({
        "new_nodes": [{
            "proposed_id": "x",
            "label": "Concept",
            "canonical_name": "X",
            "reasoning": "test",
        }],
        "new_edges": [],
    })
    normalized = normalize_graph_delta(raw)
    delta = GraphDelta(**normalized)
    assert len(delta.new_nodes) == 1
    assert delta.new_nodes[0].proposed_id == "x"


def test_non_json_string_container_field_falls_back_to_empty():
    """A container field that's a string but NOT valid/recoverable JSON must not raise — it
    degrades to an empty list for that field, same as a missing field would."""
    raw = {"new_nodes": "not json at all", "new_edges": []}
    normalized = normalize_graph_delta(raw)
    delta = GraphDelta(**normalized)
    assert delta.new_nodes == []


def test_clean_dict_input_unaffected_by_double_encoding_coercion():
    """REGRESSION: already-parsed clean input (the deepseek / sonnet-4-6 path) must be
    byte-identical in behavior — the JSON-string coercion is a no-op on non-string values."""
    normalized_before = normalize_graph_delta(QWEN_RAW)
    normalized_after = normalize_graph_delta(dict(QWEN_RAW))  # fresh dict, same content
    assert normalized_before == normalized_after
    delta = GraphDelta(**normalized_after)
    assert len(delta.new_nodes) == 1
    assert delta.new_nodes[0].proposed_id == "folded_cascode_ota"


def test_edge_reinforcement_remap():
    raw = {
        "reinforced_edges": [{
            "source": "node_a",
            "target": "node_b",
            "relation": "DEPENDS_ON",
            "chunk_ids": ["chunk_003"],
            "note": "Confirmed by second paper",
        }]
    }
    normalized = normalize_graph_delta(raw)
    delta = GraphDelta(**normalized)
    assert len(delta.reinforced_edges) == 1
    r = delta.reinforced_edges[0]
    assert r.source_ref == "node_a"
    assert r.target_ref == "node_b"
    assert r.confirming_chunk_ids == ["chunk_003"]
    assert r.new_evidence_note == "Confirmed by second paper"


# ── _salvage drop-count diagnostics ─────────────────────────────────────


def test_salvage_warning_includes_dropped_element_kinds(caplog):
    """A malformed element's kind (node label / edge relationship_type) must be aggregated into
    the drop-count warning — not just a bare count — so an operator scanning logs can tell WHICH
    kinds of proposals a chunk lost, without a full (possibly large/sensitive) payload dump."""
    raw = {
        "new_nodes": [
            {"canonical_name": "Threshold Voltage", "label": "Concept"},  # valid
            {"label": "Equation"},  # invalid: no canonical_name
            {"label": "Equation"},  # invalid: no canonical_name
        ],
        "new_edges": [
            {"source_ref": "a", "target_ref": "b", "relationship_type": "DEPENDS_ON"},  # valid
            {"relationship_type": "SOLVES_PROBLEM"},  # invalid: no source_ref/target_ref
        ],
    }

    with caplog.at_level(logging.WARNING, logger="openclaw_brain.knowledge.reasoning.normalize"):
        normalized = normalize_graph_delta(raw)

    assert len(normalized["new_nodes"]) == 1
    assert len(normalized["new_edges"]) == 1

    messages = [r.getMessage() for r in caplog.records]
    node_warning = next(m for m in messages if "new_nodes" in m)
    assert "dropped 2" in node_warning
    assert "Equation" in node_warning  # WHICH kind was dropped, not just a bare count
    # aggregate only — no payload leakage of the dropped items' other (absent) fields
    assert "canonical_name" not in node_warning

    edge_warning = next(m for m in messages if "new_edges" in m)
    assert "dropped 1" in edge_warning
    assert "SOLVES_PROBLEM" in edge_warning


def test_salvage_no_warning_when_nothing_dropped(caplog):
    """REGRESSION: a fully-valid delta must not log anything (unchanged from before this fix)."""
    raw = {
        "new_nodes": [{"canonical_name": "Threshold Voltage", "label": "Concept"}],
    }
    with caplog.at_level(logging.WARNING, logger="openclaw_brain.knowledge.reasoning.normalize"):
        normalize_graph_delta(raw)
    assert caplog.records == []
