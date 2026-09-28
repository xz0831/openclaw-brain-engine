"""Tests for source grounding verification."""

from openclaw_brain.knowledge.extraction.grounding import (
    verify_grounding,
    _concept_is_grounded,
    _equation_is_grounded,
    _parameter_is_grounded,
    _edge_is_grounded,
)
from openclaw_brain.knowledge.extraction.models import (
    ConceptMention,
    EquationMention,
    ExtractionResult,
    ParameterMention,
    RawEdge,
)


SAMPLE_TEXT = """
The common-source amplifier uses a MOSFET transistor in saturation region.
The drain current is given by I_D = (1/2) mu_n C_ox (W/L) (V_GS - V_th)^2.
The transconductance g_m is approximately 2*I_D / V_ov, where V_ov is the
overdrive voltage. Typical threshold voltage is 0.3-0.5 V in 28nm FD-SOI.
"""


def test_concept_grounded_direct_match():
    assert _concept_is_grounded(
        ConceptMention(name="Common-Source Amplifier", description="An amplifier topology using a MOSFET"),
        SAMPLE_TEXT.lower(),
    )


def test_concept_grounded_partial_match():
    assert _concept_is_grounded(
        ConceptMention(name="Threshold Voltage", description="Gate voltage at which MOSFET conducts"),
        SAMPLE_TEXT.lower(),
    )


def test_concept_not_grounded():
    assert not _concept_is_grounded(
        ConceptMention(name="Phase-Locked Loop", description="A feedback control system for frequency synthesis"),
        SAMPLE_TEXT.lower(),
    )


def test_equation_grounded():
    assert _equation_is_grounded(
        EquationMention(
            latex=r"I_D = \frac{1}{2} \mu_n C_{ox} (V_{GS} - V_{th})^2",
            variables=["I_D", "mu_n", "V_GS", "V_th"],
        ),
        SAMPLE_TEXT.lower(),
    )


def test_equation_not_grounded():
    assert not _equation_is_grounded(
        EquationMention(
            latex=r"z_q = \frac{x_y}{r_s}",
            variables=["z_q", "x_y"],
        ),
        SAMPLE_TEXT.lower(),
    )


def test_parameter_grounded_by_symbol():
    assert _parameter_is_grounded(
        ParameterMention(symbol="g_m", name="Transconductance"),
        SAMPLE_TEXT.lower(),
    )


def test_parameter_grounded_by_name():
    assert _parameter_is_grounded(
        ParameterMention(symbol="V_th", name="Threshold Voltage"),
        SAMPLE_TEXT.lower(),
    )


def test_parameter_not_grounded():
    assert not _parameter_is_grounded(
        ParameterMention(symbol="f_T", name="Unity Gain Frequency"),
        SAMPLE_TEXT.lower(),
    )


def test_edge_grounded():
    names = {"Common-Source Amplifier", "Threshold Voltage", "g_m"}
    assert _edge_is_grounded(
        RawEdge(source_name="Common-Source Amplifier", target_name="Threshold Voltage", relationship="HAS_PARAMETER"),
        names,
    )


def test_edge_not_grounded():
    names = {"Common-Source Amplifier"}
    assert not _edge_is_grounded(
        RawEdge(source_name="Common-Source Amplifier", target_name="Cascode Amplifier", relationship="TOPOLOGY_VARIANT"),
        names,
    )


def test_verify_grounding_full():
    extraction = ExtractionResult(
        chunk_id="test_chunk",
        concepts=[
            ConceptMention(name="Common-Source Amplifier", description="An amplifier topology using a MOSFET in saturation"),
            ConceptMention(name="Transconductance", description="Small-signal parameter relating gate voltage to drain current change"),
            ConceptMention(name="Phase-Locked Loop", description="A feedback control system for frequency synthesis not in this text"),
        ],
        equations=[
            EquationMention(
                latex=r"I_D = \frac{1}{2} \mu_n C_{ox} (V_{GS} - V_{th})^2",
                variables=["I_D", "mu_n", "V_GS", "V_th"],
            ),
        ],
        parameters=[
            ParameterMention(symbol="V_th", name="Threshold Voltage", units="V", typical_range="0.3-0.5 V"),
            ParameterMention(symbol="f_T", name="Unity Gain Frequency", units="GHz"),
        ],
        raw_edges=[
            RawEdge(source_name="Common-Source Amplifier", target_name="Transconductance", relationship="HAS_PARAMETER", rationale="gm is a key param"),
            RawEdge(source_name="Phase-Locked Loop", target_name="Transconductance", relationship="HAS_PARAMETER", rationale="bogus"),
        ],
    )

    result = verify_grounding(extraction, SAMPLE_TEXT)

    # Cascode Amplifier should be removed (not in text)
    assert len(result.concepts) == 2
    concept_names = {c.name for c in result.concepts}
    assert "Phase-Locked Loop" not in concept_names
    assert "Common-Source Amplifier" in concept_names
    assert "Transconductance" in concept_names

    # f_T parameter should be removed
    assert len(result.parameters) == 1
    assert result.parameters[0].symbol == "V_th"

    # Equation should remain
    assert len(result.equations) == 1

    # Edge to Cascode should be removed, edge to Transconductance should remain
    assert len(result.raw_edges) == 1
    assert result.raw_edges[0].source_name == "Common-Source Amplifier"


def test_verify_grounding_empty():
    extraction = ExtractionResult(chunk_id="empty")
    result = verify_grounding(extraction, "some text")
    assert len(result.concepts) == 0
    assert len(result.raw_edges) == 0


# --- figure-edge self-grounding gate (Rick run-directive 2026-06-16) ---------
# A VLM figure DESCRIPTION must NOT serve as the grounding support for the edges
# extracted from it (self-grounding launders fabrication). The chunker wraps the
# description in sentinels; verify_grounding excludes that span from the support set.
from openclaw_brain.knowledge.extraction.grounding import (  # noqa: E402
    FIG_VLM_CLOSE, FIG_VLM_OPEN, strip_figure_markers, verify_grounding,
)
from openclaw_brain.knowledge.extraction.models import (  # noqa: E402
    ConceptMention, ExtractionResult, RawEdge,
)


def _chunk_with_fig(description: str, caption_body: str) -> str:
    return caption_body + f"{FIG_VLM_OPEN}{description}{FIG_VLM_CLOSE}"


def test_figure_description_only_edge_is_rejected_not_self_grounded():
    chunk = _chunk_with_fig(
        description="The gain-boosting amplifier determines the input common-mode range.",
        caption_body="[Figure (circuit): Fig. 7. Telescopic cascode OTA.]\nThe cascode raises gain.\n",
    )
    ext = ExtractionResult(
        chunk_id="c", concepts=[
            ConceptMention(name="Gain-Boosting Amplifier", description="auxiliary gain-boost block stage"),
            ConceptMention(name="Input Common-Mode Range", description="input common-mode voltage range icmr"),
        ],
        raw_edges=[RawEdge(source_name="Gain-Boosting Amplifier",
                           target_name="Input Common-Mode Range", relationship="DETERMINES")])
    out = verify_grounding(ext, chunk)
    assert out.raw_edges == [], "figure-description-only edge must NOT self-ground"


def test_independent_anchored_figure_edge_survives():
    chunk = _chunk_with_fig(
        description="The telescopic cascode raises output impedance.",
        caption_body="[Figure (circuit): Fig. 7.]\nThe telescopic cascode increases the output impedance.\n",
    )
    ext = ExtractionResult(
        chunk_id="c", concepts=[
            ConceptMention(name="Telescopic Cascode", description="telescopic cascode output stage topology"),
            ConceptMention(name="Output Impedance", description="small-signal output impedance of the stage"),
        ],
        raw_edges=[RawEdge(source_name="Telescopic Cascode", target_name="Output Impedance",
                           relationship="DETERMINES")])
    out = verify_grounding(ext, chunk)
    assert len(out.raw_edges) == 1, "edge anchored in independent caption/body must survive"


def test_extractor_input_has_no_sentinels():
    chunk = _chunk_with_fig("desc text here", "[Figure (circuit): cap]\nbody\n")
    assert FIG_VLM_OPEN not in strip_figure_markers(chunk)
    assert FIG_VLM_CLOSE not in strip_figure_markers(chunk)
    assert "desc text here" in strip_figure_markers(chunk)  # content kept
