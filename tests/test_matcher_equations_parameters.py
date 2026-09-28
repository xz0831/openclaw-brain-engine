"""Unit tests for F6: equation/parameter matching in ConceptMatcher (reasoning/matcher.py).

Equations/parameters previously bypassed matching entirely — every mention was proposed as
a new node regardless of whether an equivalent one already existed (see
knowledge/reasoning/README.md "Known defects" #2: matcher.py:60-107 never looked at
extraction.equations/.parameters at all). This file covers the exact-normalized matching
added for F6: ``_normalize_latex`` / ``_normalize_plain`` / ``_best_parameter_candidate``
(pure functions) and ``ConceptMatcher._match_equations`` / ``_match_parameters`` (exercised
via ``match()``).

No Neo4j, no LLM, no real embedding model: every extraction here uses ``concepts=[]``, so
``match()``'s concept loop (and its ``encode_batch`` call) never executes —
``embedding.encode_batch([])`` is a documented no-op for an empty list regardless (see
``knowledge/embedding.py::encode_batch``, short-circuits before loading any model when
``miss_texts`` stays empty), so no monkeypatching is needed either way.
"""

from __future__ import annotations

import pytest

from openclaw_brain.knowledge.extraction.models import (
    EquationMention,
    ExtractionResult,
    ParameterMention,
)
from openclaw_brain.knowledge.reasoning.matcher import (
    ConceptMatcher,
    _best_parameter_candidate,
    _normalize_latex,
    _normalize_plain,
)


# ── _normalize_latex ──────────────────────────────────────────────────────────────────────


def test_normalize_latex_whitespace_insignificant():
    assert _normalize_latex("a + b") == _normalize_latex("a+b")


def test_normalize_latex_strips_leading_trailing_whitespace():
    assert _normalize_latex("  V = IR  ") == _normalize_latex("V = IR")


def test_normalize_latex_left_right_are_inert():
    assert _normalize_latex(r"\left( x \right)") == _normalize_latex("(x)")


def test_normalize_latex_single_char_script_braces_collapse():
    assert _normalize_latex("x^{2}") == _normalize_latex("x^2")
    assert _normalize_latex("a_{1}") == _normalize_latex("a_1")


def test_normalize_latex_multi_char_script_braces_preserved():
    # V_{DS} must NOT collapse to V_DS -- unbracing a multi-char script would change
    # which characters are subscripted (a different, not-equivalent LaTeX string).
    assert _normalize_latex("V_{DS}") != _normalize_latex("V_DS")
    assert _normalize_latex("V_{DS}") == "V_{DS}"


def test_normalize_latex_strips_dollar_delimiters():
    assert _normalize_latex("$V=IR$") == _normalize_latex("V=IR")


def test_normalize_latex_case_preserved():
    # V_{GS} (large-signal) vs v_{gs} (small-signal) are different circuit quantities in
    # this domain -- unlike concept names, equations must NOT be case-folded.
    assert _normalize_latex("V_{GS}") != _normalize_latex("v_{gs}")


def test_normalize_latex_empty_string():
    assert _normalize_latex("") == ""


def test_normalize_latex_no_symbolic_equivalence():
    # Deliberately conservative: must NOT know a+b == b+a (that's symbolic equivalence,
    # explicitly out of scope for this cut).
    assert _normalize_latex("a+b") != _normalize_latex("b+a")


def test_normalize_latex_real_duplicate_pair_from_live_recon():
    """Regression, live-data-grounded: this exact pair was found as a raw-string-distinct
    but conceptually-identical duplicate during the 2026-07-10 reconnaissance
    (source_follower_input_impedance_equation / new_source_follower_input_impedance_equation)."""
    a = r"Z_{in} = \frac{1}{C_{GS}s} + \left(1 + \frac{g_m}{C_{GS}s}\right) \frac{1}{g_{mb} + C_L s}"
    b = r"Z_{in} = \frac{1}{C_{GS} s} + \left(1 + \frac{g_m}{C_{GS} s}\right) \frac{1}{g_{mb} + C_L s}"
    assert _normalize_latex(a) == _normalize_latex(b)


# ── _normalize_plain ──────────────────────────────────────────────────────────────────────


def test_normalize_plain_case_insensitive():
    assert _normalize_plain("GBW") == _normalize_plain("gbw")


def test_normalize_plain_trims_and_collapses_whitespace():
    assert _normalize_plain("  Gain   Bandwidth  ") == "gain bandwidth"


def test_normalize_plain_empty_string():
    assert _normalize_plain("") == ""


def test_normalize_plain_does_not_expand_abbreviations():
    # Contrast with matcher._normalize_name (concept matching), which WOULD expand
    # "gbw" -> "gain bandwidth product". Parameter matching is exact-string, not fuzzy
    # concept identity -- see the design-constraint discussion in the module docstring.
    assert _normalize_plain("GBW") == "gbw"


# ── _best_parameter_candidate ─────────────────────────────────────────────────────────────


def _cand(pid: str, name: str, units: str) -> dict:
    return {"parameter_id": pid, "norm_name": name, "norm_units": units}


def test_best_parameter_candidate_exact_unit_match():
    cands = [_cand("p1", "gain", "v/v")]
    best = _best_parameter_candidate("gain", "v/v", cands)
    assert best is not None and best["parameter_id"] == "p1"


def test_best_parameter_candidate_gapfill_mention_missing_unit():
    cands = [_cand("p1", "gain", "v/v")]
    best = _best_parameter_candidate("gain", "", cands)
    assert best is not None and best["parameter_id"] == "p1"


def test_best_parameter_candidate_gapfill_graph_missing_unit():
    cands = [_cand("p1", "gain", "")]
    best = _best_parameter_candidate("gain", "v/v", cands)
    assert best is not None and best["parameter_id"] == "p1"


def test_best_parameter_candidate_both_missing_units():
    cands = [_cand("p1", "gain", "")]
    best = _best_parameter_candidate("gain", "", cands)
    assert best is not None and best["parameter_id"] == "p1"


def test_best_parameter_candidate_unit_conflict_excluded():
    cands = [_cand("p1", "gain", "db")]
    assert _best_parameter_candidate("gain", "v/v", cands) is None


def test_best_parameter_candidate_conflict_never_beats_a_compatible_candidate():
    # Order matters not: even if the conflicting candidate is listed first, the
    # unit-compatible one must win -- a conflict is disqualified, never merely "worse".
    cands = [_cand("p_conflict", "gain", "db"), _cand("p_ok", "gain", "v/v")]
    best = _best_parameter_candidate("gain", "v/v", cands)
    assert best is not None and best["parameter_id"] == "p_ok"


def test_best_parameter_candidate_prefers_exact_unit_over_gapfill():
    cands = [_cand("p_gapfill", "gain", ""), _cand("p_exact", "gain", "v/v")]
    best = _best_parameter_candidate("gain", "v/v", cands)
    assert best is not None and best["parameter_id"] == "p_exact"


def test_best_parameter_candidate_no_name_match():
    cands = [_cand("p1", "bandwidth", "hz")]
    assert _best_parameter_candidate("gain", "v/v", cands) is None


def test_best_parameter_candidate_empty_name_never_matches():
    cands = [_cand("p1", "", "")]
    assert _best_parameter_candidate("", "", cands) is None


# ── ConceptMatcher._match_equations / _match_parameters via match() ────────────────────────


class StubGraphEP:
    """Stub exposing only ``run_read_query`` — the sole GraphStore method F6's equation/
    parameter matching calls — dispatched by query content. Return shape mirrors the real
    ``GraphStore.run_read_query(query, params=None) -> list[dict]`` (see graph/store.py:1123)."""

    def __init__(self, equation_rows=None, parameter_rows=None, raise_on: str | None = None):
        self._equation_rows = equation_rows if equation_rows is not None else []
        self._parameter_rows = parameter_rows if parameter_rows is not None else []
        self._raise_on = raise_on
        self.queries: list[str] = []

    async def run_read_query(self, query, params=None):
        self.queries.append(query)
        if "MATCH (n:Equation)" in query:
            if self._raise_on == "equation":
                raise RuntimeError("simulated Neo4j failure")
            return self._equation_rows
        if "MATCH (n:Parameter)" in query:
            if self._raise_on == "parameter":
                raise RuntimeError("simulated Neo4j failure")
            return self._parameter_rows
        raise AssertionError(f"unexpected query: {query}")


def _extraction_eq(*latex_list: str) -> ExtractionResult:
    return ExtractionResult(chunk_id="c1", equations=[EquationMention(latex=lx) for lx in latex_list])


def _extraction_param(*mentions: ParameterMention) -> ExtractionResult:
    return ExtractionResult(chunk_id="c1", parameters=list(mentions))


@pytest.mark.asyncio
async def test_match_equations_empty_extraction_makes_no_graph_call():
    graph = StubGraphEP()
    matcher = ConceptMatcher(graph)
    result = await matcher.match(ExtractionResult(chunk_id="c1"))
    assert result.matched_equations == []
    assert result.new_equations == []
    assert graph.queries == []


@pytest.mark.asyncio
async def test_match_equations_exact_match():
    graph = StubGraphEP(equation_rows=[
        {"equation_id": "eq_gbw", "canonical_latex": "GBW = g_m/(2\\pi C_L)"},
    ])
    matcher = ConceptMatcher(graph)
    result = await matcher.match(_extraction_eq("GBW = g_m/(2\\pi C_L)"))
    assert len(result.matched_equations) == 1
    assert result.matched_equations[0].existing_node_id == "eq_gbw"
    assert result.matched_equations[0].similarity == 1.0
    assert result.new_equations == []


@pytest.mark.asyncio
async def test_match_equations_whitespace_only_variant_still_matches():
    """Regression for the live-data finding (2026-07-10 recon): two equations differing
    only in whitespace/spacing must resolve to the SAME node, not duplicate."""
    graph = StubGraphEP(equation_rows=[
        {"equation_id": "eq_zin", "canonical_latex":
            "Z_{in} = \\frac{1}{C_{GS}s} + \\left(1 + \\frac{g_m}{C_{GS}s}\\right) \\frac{1}{g_{mb} + C_L s}"},
    ])
    matcher = ConceptMatcher(graph)
    result = await matcher.match(_extraction_eq(
        "Z_{in} = \\frac{1}{C_{GS} s} + \\left(1 + \\frac{g_m}{C_{GS} s}\\right) \\frac{1}{g_{mb} + C_L s}"
    ))
    assert len(result.matched_equations) == 1
    assert result.matched_equations[0].existing_node_id == "eq_zin"


@pytest.mark.asyncio
async def test_match_equations_no_candidate_is_new():
    graph = StubGraphEP(equation_rows=[{"equation_id": "eq_x", "canonical_latex": "A = B"}])
    matcher = ConceptMatcher(graph)
    result = await matcher.match(_extraction_eq("C = D"))
    assert result.matched_equations == []
    assert len(result.new_equations) == 1
    assert result.new_equations[0].latex == "C = D"


@pytest.mark.asyncio
async def test_match_equations_no_symbolic_equivalence():
    graph = StubGraphEP(equation_rows=[{"equation_id": "eq_x", "canonical_latex": "a+b"}])
    matcher = ConceptMatcher(graph)
    result = await matcher.match(_extraction_eq("b+a"))
    assert result.matched_equations == []
    assert len(result.new_equations) == 1


@pytest.mark.asyncio
async def test_match_equations_candidate_fetch_failure_degrades_to_all_new():
    graph = StubGraphEP(raise_on="equation")
    matcher = ConceptMatcher(graph)
    result = await matcher.match(_extraction_eq("A = B"))
    assert result.matched_equations == []
    assert len(result.new_equations) == 1


@pytest.mark.asyncio
async def test_match_equations_non_string_canonical_latex_skipped_not_crashed():
    """Defensive: live data (2026-07-10 recon) has 3 legacy Equation nodes whose
    canonical_latex is a LIST, not a string -- a pre-existing, unrelated data-quality
    defect. The matcher must skip them, never crash."""
    graph = StubGraphEP(equation_rows=[
        {"equation_id": "eq_bad", "canonical_latex": ["not", "a", "string"]},
        {"equation_id": "eq_good", "canonical_latex": "A = B"},
    ])
    matcher = ConceptMatcher(graph)
    result = await matcher.match(_extraction_eq("A = B"))
    assert len(result.matched_equations) == 1
    assert result.matched_equations[0].existing_node_id == "eq_good"


@pytest.mark.asyncio
async def test_match_parameters_empty_extraction_makes_no_graph_call():
    graph = StubGraphEP()
    matcher = ConceptMatcher(graph)
    result = await matcher.match(ExtractionResult(chunk_id="c1"))
    assert result.matched_parameters == []
    assert result.new_parameters == []
    assert graph.queries == []


@pytest.mark.asyncio
async def test_match_parameters_exact_name_and_unit():
    graph = StubGraphEP(parameter_rows=[
        {"parameter_id": "p_gbw", "canonical_name": "Gain-Bandwidth Product", "units": "Hz"},
    ])
    matcher = ConceptMatcher(graph)
    mention = ParameterMention(symbol="GBW", name="Gain-Bandwidth Product", units="Hz")
    result = await matcher.match(_extraction_param(mention))
    assert len(result.matched_parameters) == 1
    assert result.matched_parameters[0].existing_node_id == "p_gbw"
    assert result.new_parameters == []


@pytest.mark.asyncio
async def test_match_parameters_case_and_whitespace_insensitive_name():
    graph = StubGraphEP(parameter_rows=[
        {"parameter_id": "p_gbw", "canonical_name": "gain-bandwidth product", "units": ""},
    ])
    matcher = ConceptMatcher(graph)
    mention = ParameterMention(symbol="GBW", name="  Gain-Bandwidth   Product  ", units="")
    result = await matcher.match(_extraction_param(mention))
    assert len(result.matched_parameters) == 1


@pytest.mark.asyncio
async def test_match_parameters_gapfill_mention_has_unit_graph_missing():
    graph = StubGraphEP(parameter_rows=[
        {"parameter_id": "p1", "canonical_name": "Feedback Factor", "units": ""},
    ])
    matcher = ConceptMatcher(graph)
    mention = ParameterMention(symbol="beta", name="Feedback Factor", units="unitless")
    result = await matcher.match(_extraction_param(mention))
    assert len(result.matched_parameters) == 1
    assert result.matched_parameters[0].existing_node_id == "p1"


@pytest.mark.asyncio
async def test_match_parameters_unit_conflict_is_new_never_same():
    """Regression for the live-data finding (2026-07-10 recon): 'Unity-Gain Frequency' at
    Hz vs kHz -- same name, genuinely different unit -- must NOT be silently merged."""
    graph = StubGraphEP(parameter_rows=[
        {"parameter_id": "p_hz", "canonical_name": "Unity-Gain Frequency", "units": "Hz"},
    ])
    matcher = ConceptMatcher(graph)
    mention = ParameterMention(symbol="f_u", name="Unity-Gain Frequency", units="kHz")
    result = await matcher.match(_extraction_param(mention))
    assert result.matched_parameters == []
    assert len(result.new_parameters) == 1


@pytest.mark.asyncio
async def test_match_parameters_no_name_match_is_new():
    graph = StubGraphEP(parameter_rows=[
        {"parameter_id": "p1", "canonical_name": "Bandwidth", "units": "Hz"},
    ])
    matcher = ConceptMatcher(graph)
    mention = ParameterMention(symbol="Av", name="Gain", units="V/V")
    result = await matcher.match(_extraction_param(mention))
    assert result.matched_parameters == []
    assert len(result.new_parameters) == 1


@pytest.mark.asyncio
async def test_match_parameters_candidate_fetch_failure_degrades_to_all_new():
    graph = StubGraphEP(raise_on="parameter")
    matcher = ConceptMatcher(graph)
    mention = ParameterMention(symbol="Av", name="Gain", units="V/V")
    result = await matcher.match(_extraction_param(mention))
    assert result.matched_parameters == []
    assert len(result.new_parameters) == 1


@pytest.mark.asyncio
async def test_match_parameters_non_string_canonical_name_skipped_not_crashed():
    graph = StubGraphEP(parameter_rows=[
        {"parameter_id": "p_bad", "canonical_name": ["not", "a", "string"], "units": ""},
        {"parameter_id": "p_good", "canonical_name": "Gain", "units": "V/V"},
    ])
    matcher = ConceptMatcher(graph)
    mention = ParameterMention(symbol="Av", name="Gain", units="V/V")
    result = await matcher.match(_extraction_param(mention))
    assert len(result.matched_parameters) == 1
    assert result.matched_parameters[0].existing_node_id == "p_good"


@pytest.mark.asyncio
async def test_match_dormant_when_graph_lacks_run_read_query_and_lists_empty():
    """Backward-compat guard: the (pre-F6) StubGraph used by test_matcher_tiers.py has no
    run_read_query at all. Confirms match() never even attempts to call it when
    equations/parameters are both empty -- F6 is fully dormant for concept-only
    extractions, so every pre-existing concept-matching test stays valid unmodified."""

    class NoRunReadQueryGraph:
        async def find_match_candidates(self, name, embedding=None, limit=8):
            return []

    matcher = ConceptMatcher(NoRunReadQueryGraph())
    result = await matcher.match(ExtractionResult(chunk_id="c1"))
    assert result.matched_equations == []
    assert result.matched_parameters == []
