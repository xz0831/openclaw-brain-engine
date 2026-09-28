"""Tests for VLM preamble handling in structured chunks."""

from openclaw_brain.knowledge.extraction.chunker import _strip_vlm_preamble


def test_strip_vlm_preamble_removes_real_leadins():
    cases = [
        (
            "Based on the provided block diagram, here is a technical analysis of the circuit:"
            "\n\n### 1. Functional Blocks\nThe amplifier contains an input stage.",
            "### 1. Functional Blocks\nThe amplifier contains an input stage.",
        ),
        (
            "Based on the provided image, here is an analysis of the plot:"
            "\n\n**1. Axes**\nThe x-axis is frequency.",
            "**1. Axes**\nThe x-axis is frequency.",
        ),
        (
            "Of course. Based on my expertise in analog circuit design, here is a detailed "
            "analysis of the provided schematic."
            "\n\n### Analysis of the Circuit\nThe schematic is a folded cascode OTA.",
            "### Analysis of the Circuit\nThe schematic is a folded cascode OTA.",
        ),
        (
            "Based on my analysis as a semiconductor and analog circuit design expert, here is "
            "a detailed breakdown of the provided plot:"
            "\n\nThis image is not a circuit schematic; it is a measured response.",
            "This image is not a circuit schematic; it is a measured response.",
        ),
    ]

    for description, expected in cases:
        assert _strip_vlm_preamble(description) == expected


def test_strip_vlm_preamble_leaves_clean_content_unchanged():
    clean_block = "**Analysis of the Block Diagram**\n\nThe signal path starts at the input."
    clean_topology = "**1. Topology**\nThe circuit is a common-source stage."

    assert _strip_vlm_preamble(clean_block) == clean_block
    assert _strip_vlm_preamble(clean_topology) == clean_topology


def test_strip_vlm_preamble_leaves_non_preamble_and_empty_text_unchanged():
    no_blankline = "The plot shows gain roll-off without a conversational preamble."

    assert _strip_vlm_preamble(no_blankline) == no_blankline
    assert _strip_vlm_preamble("") == ""
