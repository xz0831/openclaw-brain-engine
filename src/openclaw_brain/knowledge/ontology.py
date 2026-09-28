"""Semiconductor domain ontology — constrains extraction to known entity/relation types.

Designed to capture not just WHAT exists in a circuit design document,
but WHY circuits are designed the way they are — the design reasoning,
evolution of topologies, and structure-equation connections that form
the core of analog circuit design knowledge.
"""

from __future__ import annotations

# ── Entity type definitions with expected properties ──

ENTITY_TYPES = {
    "Concept": {
        "description": "A named technical concept, phenomenon, or technique.",
        "properties": ["name", "description", "domain", "granularity"],
        "examples": [
            "Transconductance", "Miller Effect", "Channel Length Modulation",
            "Body Effect", "Threshold Voltage", "Sub-threshold Operation",
            # CIS-specific concepts
            "Pixel Reset Noise (kTC Noise)", "Correlated Double Sampling (CDS)",
            "Column ADC", "Photodiode Full Well Capacity", "Quantum Efficiency",
            "Dark Current", "Rolling Shutter Distortion", "Global Shutter",
            "Pixel Source Follower", "Readout Chain Noise Floor",
        ],
    },
    "Equation": {
        "description": "A mathematical equation connecting circuit parameters. Include physical_meaning and component_mapping.",
        "properties": ["latex", "equation_type", "variables", "assumptions", "physical_meaning", "component_mapping"],
        "examples": [
            "Av = -gm·RD (CS amplifier gain: gm from input transistor, RD from load)",
            "gm = 2·ID/Vov (transconductance in saturation, set by bias current and overdrive)",
            "Rout ≈ gm2·ro2·ro1 (cascode output impedance: impedance multiplication by stacking)",
        ],
    },
    "Parameter": {
        "description": "A physical parameter with symbol, units, and typical numeric range.",
        "properties": ["symbol", "name", "units", "typical_range"],
        "examples": [
            "gm (Transconductance, mA/V, 1-50 mA/V)",
            "Av (Voltage Gain, V/V or dB, 10-1000 V/V)",
            "f_T (Unity Gain Frequency, GHz, 1-300 GHz)",
            "VDS,sat (Saturation Voltage, V, 0.1-0.3 V)",
        ],
    },
    "CircuitTopology": {
        "description": "A circuit architecture. MUST include: what problem it solves, what limitation it introduces, and when to use it.",
        "properties": ["name", "function", "key_nodes", "sub_blocks", "solves", "introduces", "when_to_use"],
        "examples": [
            "Common-Source Amplifier (solves: basic voltage amplification; introduces: limited gain by ro; when_to_use: simple gain stages)",
            "Cascode (solves: low output impedance of CS; introduces: reduced voltage headroom by one VDS,sat; when_to_use: when gain is insufficient but headroom is available)",
            "Folded Cascode OTA (solves: headroom problem of telescopic; introduces: extra current branches and noise; when_to_use: low-supply designs needing high gain)",
            # CIS-specific topologies
            "4T Pixel (solves: kTC noise via CDS-capable transfer gate; introduces: layout area for TX transistor; when_to_use: low-noise CIS designs)",
            "Single-Slope Column ADC (solves: per-column analog-to-digital conversion; introduces: linearity dependent on ramp quality; when_to_use: area-efficient CIS readout)",
            "Pixel Source Follower (solves: buffered readout of floating diffusion voltage; introduces: gain < 1 and 1/f noise contribution; when_to_use: standard 4T pixel readout chain)",
        ],
    },
    "Principle": {
        "description": "A fundamental law, theorem, or design principle.",
        "properties": ["name", "statement"],
        "examples": [
            "Kirchhoff's Current Law",
            "Barkhausen Stability Criterion",
            "Gain-Bandwidth Product Conservation",
            "Matching by Common-Centroid Layout",
        ],
    },
}


# ── Relationship type definitions ──

RELATIONSHIP_TYPES = {
    # Basic structural
    "USES_EQUATION": "Circuit or concept is modeled by this equation.",
    "HAS_PARAMETER": "Entity has this parameter as a key specification.",
    "DEPENDS_ON": "Entity's behavior fundamentally depends on another.",
    "ASSUMES": "Equation or analysis assumes this condition (e.g., 'saturation region', 'long-channel').",
    "DERIVED_FROM": "Mathematically or conceptually derived from another entity.",
    "APPROXIMATION_OF": "A simplified version of a more exact relationship.",

    # Design reasoning — the most valuable for an engineer
    "SOLVES_PROBLEM": "This topology/technique solves a specific design problem. (e.g., 'Cascode SOLVES_PROBLEM low output impedance'). Rationale must explain the mechanism.",
    "INTRODUCES_PROBLEM": "This topology/technique creates a new limitation. (e.g., 'Cascode INTRODUCES_PROBLEM reduced voltage headroom'). Rationale must quantify if possible.",
    "TRADES_OFF": "Improving one parameter degrades another. Rationale must name BOTH sides and explain why they're coupled.",
    "DESIGN_RULE": "Practical design guideline. (e.g., 'Use cascode current mirrors when output impedance > 10·ro is needed').",

    # Topology evolution — how circuit architectures build on each other
    "EVOLVES_TO": "One topology evolves into another to address a limitation. Rationale must explain the motivation. (e.g., 'Telescopic OTA EVOLVES_TO Folded Cascode OTA because telescopic has insufficient output swing')",
    "SUB_BLOCK": "One circuit contains another as a functional sub-block. (e.g., 'Two-Stage Miller OTA SUB_BLOCK Differential Pair (as input stage)')",
    "TOPOLOGY_VARIANT": "A variant or specialization of a parent topology.",
    "COMPENSATED_BY": "A circuit is stabilized by a compensation technique. (e.g., 'Two-Stage OTA COMPENSATED_BY Miller Capacitor')",

    # Equation-structure mapping
    "MODELS_BEHAVIOR": "An equation models the behavior of a circuit/concept. Rationale explains what aspect it captures.",
    "VARIABLE_MAPS_TO": "A variable in an equation maps to a physical circuit component. (e.g., 'gm VARIABLE_MAPS_TO input transistor M1')",

    # Evidence and cross-reference
    "CONFIRMS": "New evidence confirms an existing relationship.",
    "CONTRADICTS": "New evidence contradicts an existing relationship.",
    "REFINES": "New information adds detail to existing knowledge.",
    "BRIDGES_TO": "Cross-domain connection between different engineering areas.",
}


# ── Domain classifications ──

DOMAINS = [
    "semiconductor_physics",
    "analog_circuits",
    "digital_circuits",
    "mixed_signal",
    "rf_circuits",
    "power_management",
    "device_fabrication",
    "neuromorphic",
    "memory_circuits",
    "sensor_interfaces",
    "signal_processing",
    "circuit_simulation",
    "layout_design",
    "reliability",
    # CIS / image sensor domains
    "cis_architecture",
    "image_sensors",
    # Math / physics foundations
    "mathematics",
    "classical_physics",
    "electromagnetism",
    "thermodynamics",
    "quantum_mechanics",
    "calculus",
    "electromagnetics",
    "probability_statistics",
    "general",
]


def build_ontology_snippet(context: str = "general") -> str:
    """Build an ontology snippet to inject into extraction prompts."""
    lines = [
        "## Domain Ontology (Semiconductor & Analog Circuit Design)",
        "",
        "Use the following entity types and relationship types.",
        "Focus on capturing DESIGN REASONING — not just what exists, but WHY.",
        "",
        "### Entity Types",
    ]

    for etype, info in ENTITY_TYPES.items():
        examples = "\n    ".join(f'- {e}' for e in info["examples"][:3])
        lines.append(f"- **{etype}**: {info['description']}")
        lines.append(f"  Examples:\n    {examples}")

    lines.append("")
    lines.append("### Relationship Types (prioritize design reasoning types)")
    for rtype, desc in RELATIONSHIP_TYPES.items():
        lines.append(f"- **{rtype}**: {desc}")

    lines.append("")
    lines.append("### Domain Classifications")
    lines.append(f"Use one of: {', '.join(DOMAINS)}")

    return "\n".join(lines)


# Pre-built snippet for injection into prompts
ONTOLOGY_SNIPPET = build_ontology_snippet()
