"""Tests for semiconductor domain ontology."""

from openclaw_brain.knowledge.ontology import (
    ENTITY_TYPES,
    RELATIONSHIP_TYPES,
    DOMAINS,
    ONTOLOGY_SNIPPET,
    build_ontology_snippet,
)


def test_entity_types_defined():
    assert "Concept" in ENTITY_TYPES
    assert "Equation" in ENTITY_TYPES
    assert "Parameter" in ENTITY_TYPES
    assert "CircuitTopology" in ENTITY_TYPES
    assert "Principle" in ENTITY_TYPES


def test_entity_types_have_required_fields():
    for etype, info in ENTITY_TYPES.items():
        assert "description" in info, f"{etype} missing description"
        assert "properties" in info, f"{etype} missing properties"
        assert "examples" in info, f"{etype} missing examples"
        assert len(info["examples"]) >= 3, f"{etype} needs at least 3 examples"


def test_relationship_types_defined():
    expected = [
        "USES_EQUATION", "HAS_PARAMETER", "DEPENDS_ON", "ASSUMES",
        "DERIVED_FROM", "TOPOLOGY_VARIANT", "TRADES_OFF", "DESIGN_RULE",
    ]
    for rtype in expected:
        assert rtype in RELATIONSHIP_TYPES, f"Missing relationship type: {rtype}"


def test_domains_include_key_areas():
    assert "analog_circuits" in DOMAINS
    assert "semiconductor_physics" in DOMAINS
    assert "neuromorphic" in DOMAINS
    assert "device_fabrication" in DOMAINS


def test_build_ontology_snippet():
    snippet = build_ontology_snippet()
    assert "Entity Types" in snippet
    assert "Relationship Types" in snippet
    assert "Domain Classifications" in snippet
    assert "Concept" in snippet
    assert "USES_EQUATION" in snippet


def test_ontology_snippet_prebuilt():
    assert len(ONTOLOGY_SNIPPET) > 100
    assert "Semiconductor" in ONTOLOGY_SNIPPET
