"""Pure tests for quality audit logic."""

import pytest
from click.testing import CliRunner

from openclaw_brain.cli import main
from openclaw_brain.knowledge.quality import (
    Verdict,
    run_quality_audit,
    sample_nodes,
    select_evidence,
    should_fail_precision,
    summarize,
)


def test_select_evidence_ranks_by_max_overlap_and_marks_truncated():
    chunks = [
        {
            "chunk_id": "c1",
            "text_preview": "The reset switch contributes kTC noise.",
            "section_title": "Reset Noise",
        },
        {
            "chunk_id": "c2",
            "text_preview": "Under CMS the shot-noise integration variance increases with M.",
            "section_title": "CMS Shot Noise",
        },
        {
            "chunk_id": "c3",
            "text_preview": "Column ADC ramp timing is discussed.",
            "section_title": "ADC",
        },
    ]

    selected, truncated = select_evidence(
        "Shot-noise integration variance increases linearly with M.",
        chunks,
    )

    assert truncated is True
    assert [chunk["chunk_id"] for chunk in selected] == ["c2", "c1"]
    assert selected[0]["overlap_score"] > selected[1]["overlap_score"]
    assert selected[0]["verbatim"] is False


def test_select_evidence_uses_verbatim_text_without_truncation():
    chunks = [
        {
            "chunk_id": "c1",
            "text_preview": "short preview",
            "text": "Full verbatim evidence says shot noise variance increases with M.",
            "section_title": "Noise",
            "raw_text_hash": "sha",
            "verbatim": True,
        }
    ]

    selected, truncated = select_evidence("shot noise variance", chunks)

    assert truncated is False
    assert selected[0]["chunk_id"] == "c1"
    assert selected[0]["verbatim"] is True


def test_select_evidence_empty_chunks_has_no_truncation():
    selected, truncated = select_evidence("anything", [])

    assert selected == []
    assert truncated is False


def test_summarize_excludes_judge_error_from_precision_denominator():
    summary = summarize([
        {"verdict": "SUPPORTED", "evidence_truncated": True},
        {"verdict": "OVERGENERALIZED", "evidence_truncated": True},
        {"verdict": "UNSUPPORTED", "evidence_truncated": True},
        {"verdict": "NO_EVIDENCE", "evidence_truncated": False},
        {"verdict": "JUDGE_ERROR", "evidence_truncated": True},
    ])

    assert summary["n"] == 5
    assert summary["judged"] == 4
    assert summary["judged_with_evidence"] == 3
    assert summary["precision"] == pytest.approx(1 / 3)
    assert summary["overgeneralized_rate"] == pytest.approx(1 / 3)
    assert summary["unsupported_rate"] == pytest.approx(1 / 3)
    assert summary["no_evidence_rate"] == pytest.approx(1 / 4)
    assert summary["judge_error_count"] == 1
    assert summary["evidence_truncated_count"] == 4
    assert summary["verbatim_evidence_count"] == 0


def test_seeded_sampling_is_deterministic():
    nodes = [{"id": str(i)} for i in range(20)]

    first = sample_nodes(nodes, sample=8, seed=42)
    second = sample_nodes(nodes, sample=8, seed=42)
    different = sample_nodes(nodes, sample=8, seed=7)

    assert first == second
    assert first != different
    assert len(first) == 8


def test_fail_below_decision_uses_strict_less_than():
    assert should_fail_precision({"precision": 0.49}, 0.5)
    assert not should_fail_precision({"precision": 0.5}, 0.5)
    assert not should_fail_precision({"precision": 0.0}, 0.0)


@pytest.mark.asyncio
async def test_orchestrator_marks_no_evidence_without_calling_judge():
    judge_calls = 0

    async def fetch_nodes(label):
        return [
            {
                "id": "n1",
                "canonical_name": "Scoped Claim",
                "description": "Shot noise variance increases with M.",
                "source_id": "src1",
            },
            {
                "id": "n2",
                "canonical_name": "Missing Source",
                "description": "Total noise variance increases with M.",
                "source_id": "",
            },
        ]

    async def fetch_chunks(source_id):
        assert source_id == "src1"
        return [{
            "chunk_id": "c1",
            "text_preview": "Shot noise variance increases with M.",
            "section_title": "Noise",
        }]

    async def judge(node, evidence):
        nonlocal judge_calls
        judge_calls += 1
        return Verdict(verdict="SUPPORTED", reason="The evidence supports the claim.")

    result = await run_quality_audit(
        labels=["Concept"],
        sample=10,
        seed=1,
        fetch_nodes=fetch_nodes,
        fetch_chunks=fetch_chunks,
        judge=judge,
    )

    verdicts = {item["id"]: item["verdict"] for item in result["items"]}
    assert verdicts == {"n1": "SUPPORTED", "n2": "NO_EVIDENCE"}
    assert judge_calls == 1


def test_quality_audit_help_works():
    result = CliRunner().invoke(main, ["quality-audit", "--help"])

    assert result.exit_code == 0
    assert "quality-audit" in result.output
    assert "--fallback-model" in result.output
