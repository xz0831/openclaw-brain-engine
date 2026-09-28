"""Regression tests for R0b equation confabulation grounding."""

from __future__ import annotations

import pytest

from openclaw_brain.knowledge.extraction.grounding import (
    _eq_normalize,
    _equation_is_grounded,
)
from openclaw_brain.knowledge.extraction.models import EquationMention


def _grounded(latex: str, variables: list[str], chunk_text: str) -> bool:
    return _equation_is_grounded(
        EquationMention(latex=latex, variables=variables),
        _eq_normalize(chunk_text),
    )


@pytest.mark.parametrize(
    ("latex", "variables", "chunk_text"),
    [
        (
            r"V_{os}\cong\frac{2\tau}{T}V_{spike}",
            ["V_{os}", r"\tau", "T", "V_{spike}"],
            r"The displayed body is $$V_{os} \cong \frac{2\tau}{T}V_{spike}$$.",
        ),
        (
            r"(8/\pi^2)A_0 \approx 0.81 A_0",
            ["A_0"],
            r"The paper writes $(8/\pi^2)A_0 \approx 0.81A_0$ for this gain.",
        ),
        (
            r"\sinc(x)\equiv\sin(x)/x",
            ["sinc(x)", "x"],
            r"The definition is shown inline as $\sinc(x)\equiv\sin(x)/x$.",
        ),
        (
            r"S_{Nin}=0.8525kT/C",
            ["S_{Nin}", "kT", "C"],
            r"The input-referred noise is displayed as $S_{Nin}=0.8525kT/C$.",
        ),
        (
            r"C_P=C_{Tov}+C_{Rov}+C_J+C_W",
            ["C_P", "C_Tov", "C_Rov", "C_J", "C_W"],
            r"The parasitic capacitance body is $C_P=C_{Tov}+C_{Rov}+C_J+C_W$.",
        ),
    ],
)
def test_equation_is_grounded_keeps_displayed_equation_bodies(latex, variables, chunk_text):
    assert _grounded(latex, variables, chunk_text)


@pytest.mark.parametrize(
    ("latex", "variables", "chunk_text"),
    [
        (
            r"R_3\propto V_{NUL+}",
            ["R_3", "V_{NUL+}"],
            "Two resistors R_3 and R_4 are controlled by voltages V_NUL+ and V_NUL-.",
        ),
        (
            r"T_{conv}=\frac{2^N}{f_{clk}}",
            ["T_{conv}", "N", "f_{clk}"],
            "The total A/D conversion time can be expressed as (1), where T_conv is defined.",
        ),
        (
            r"k_f/(C_{ox} A W L f)",
            ["k_f", "C_{ox}", "A", "W", "L", "f"],
            "Equation (5) is the starting point for the flicker-noise discussion.",
        ),
        (
            r"V_{range}=V_{sat}",
            ["V_{range}", "V_{sat}"],
            "The range of the input is limited by saturation, and V_range is discussed.",
        ),
    ],
)
def test_equation_is_grounded_drops_prose_only_or_ocr_gap_confabs(
    latex,
    variables,
    chunk_text,
):
    assert not _grounded(latex, variables, chunk_text)
