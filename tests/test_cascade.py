"""Tests for cascade routing quality checks."""

from openclaw_brain.llm.cascade import ExtractionQualityCheck, QualityResult
from openclaw_brain.knowledge.extraction.models import (
    ConceptMention,
    ExtractionResult,
    EquationMention,
    ParameterMention,
)


SAMPLE_TEXT = "The threshold voltage of the MOSFET transistor in 28nm FD-SOI technology is approximately 0.3V."


def _make_result(**kwargs) -> ExtractionResult:
    return ExtractionResult(chunk_id="test", **kwargs)


def test_quality_check_passes_good_extraction():
    check = ExtractionQualityCheck()
    extraction = _make_result(
        concepts=[
            ConceptMention(name="Threshold Voltage", description="The gate voltage at which a MOSFET begins to conduct current"),
            ConceptMention(name="FD-SOI Technology", description="Fully-depleted silicon-on-insulator technology for reduced parasitics"),
        ],
        parameters=[
            ParameterMention(symbol="V_th", name="Threshold Voltage", units="V", typical_range="0.3 V"),
        ],
    )
    result = check.check(extraction, SAMPLE_TEXT)
    assert result.passed
    assert result.entity_count == 3


def test_quality_check_fails_too_few_entities():
    check = ExtractionQualityCheck(min_entities=2)
    extraction = _make_result()
    result = check.check(extraction, SAMPLE_TEXT)
    assert not result.passed
    assert any("Too few entities" in r for r in result.reasons)


def test_quality_check_fails_empty_descriptions():
    check = ExtractionQualityCheck(min_description_ratio=0.8, min_description_length=50)
    extraction = _make_result(
        concepts=[
            ConceptMention(name="Threshold Voltage", description="The gate voltage at which a MOSFET begins to conduct current in saturation mode"),
            ConceptMention(name="FD-SOI Technology", description="A technology node variant"),
            ConceptMention(name="MOSFET Transistor", description="A transistor type"),
        ],
    )
    result = check.check(extraction, SAMPLE_TEXT)
    assert not result.passed
    assert any("description ratio" in r.lower() for r in result.reasons)


def test_quality_check_fails_snake_case_names():
    check = ExtractionQualityCheck()
    extraction = _make_result(
        concepts=[
            ConceptMention(name="threshold_voltage", description="The gate voltage at which a MOSFET begins to conduct"),
            ConceptMention(name="fd_soi_technology", description="Fully-depleted silicon-on-insulator technology for circuits"),
        ],
    )
    result = check.check(extraction, SAMPLE_TEXT)
    assert not result.passed
    assert any("snake_case" in r for r in result.reasons)


def test_quality_check_passes_mixed_names():
    check = ExtractionQualityCheck()
    extraction = _make_result(
        concepts=[
            ConceptMention(name="Threshold Voltage", description="The gate voltage at which a MOSFET begins to conduct current"),
            ConceptMention(name="fd_soi", description="Fully-depleted silicon-on-insulator technology used in advanced circuits"),
        ],
    )
    result = check.check(extraction, SAMPLE_TEXT)
    # 50% snake_case is at the boundary, should pass (>50% fails)
    assert result.passed


def test_quality_result_bool():
    assert QualityResult(passed=True, reasons=[])
    assert not QualityResult(passed=False, reasons=["low quality"])


def test_quality_result_repr():
    r = QualityResult(passed=True, reasons=[], entity_count=5)
    assert "True" in repr(r)
    r2 = QualityResult(passed=False, reasons=["low quality"])
    assert "False" in repr(r2)
