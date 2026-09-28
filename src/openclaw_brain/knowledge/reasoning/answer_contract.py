"""Deterministic answer citation contract. No model, graph, or file I/O here."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from openclaw_brain.knowledge.executable.lesson import extract_answer_citation_ids

Policy = Literal["flag", "abstain", "model_knowledge"]

NO_KB_PREFIX = "openclaw-brain에 이 질문의 근거가 없습니다. 아래는 모델 내부 지식입니다(검증되지 않음)."
PARTIAL_KB_PREFIX = "일부 내용은 openclaw-brain 근거 없이 모델 내부 지식으로 답했습니다([모델 지식] 표시)."


class ContextManifest(BaseModel):
    refs: dict[str, Literal["summary", "reference_only"]] = Field(default_factory=dict)
    levels: dict[str, str] = Field(default_factory=dict)
    concepts: list[str] = Field(default_factory=list)

    @classmethod
    def from_refs(cls, refs: list[dict]) -> "ContextManifest":
        exposed: dict[str, Literal["summary", "reference_only"]] = {}
        levels: dict[str, str] = {}
        concepts: list[str] = []
        for ref in refs:
            cid = ref.get("id")
            if not isinstance(cid, str) or not cid:
                continue
            concepts.append(cid)
            exposed[cid] = "summary"
            cite = ref.get("cite") or {}
            levels[cid] = str(cite.get("level") or "derived")
            for chunk in (cite.get("chunks") or [])[:3]:
                if isinstance(chunk, str) and chunk:
                    exposed.setdefault(chunk, "reference_only")
                    levels.setdefault(chunk, levels[cid])
        return cls(refs=exposed, levels=levels, concepts=list(dict.fromkeys(concepts)))

    def digest(self) -> str:
        payload = {"refs": self.refs, "levels": self.levels, "concepts": self.concepts}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Citation(BaseModel):
    id: str
    kind: str | None = None
    tier: int | None = None
    exists: bool | None = None
    resolution: Literal["active", "retracted", "missing", "ambiguous", "error"]
    in_context: bool
    exposure: Literal["summary", "reference_only", "none"]
    origins: list[Literal["array", "body"]] = Field(default_factory=list)
    verdict: str | None = None
    scope: dict | None = None


class Checks(BaseModel):
    citation_integrity: Literal["pass", "fail", "unknown"]
    citation_count: int = 0
    all_citations_derived: bool = False
    semantic_support: Literal["not_checked"] = "not_checked"
    scope_check: Literal["lexical_pdk_only"] = "lexical_pdk_only"
    context_coverage: Literal["refs_only"] = "refs_only"
    kb_coverage: Literal["cited", "partial", "none"] = "none"
    model_knowledge: bool = False


class Envelope(BaseModel):
    schema_version: Literal["answer-envelope/0.1"] = "answer-envelope/0.1"
    status: Literal["grounded", "unverified", "abstained", "invalid"]
    reason_codes: list[str] = Field(default_factory=list)
    policy: Policy
    citations: list[Citation] = Field(default_factory=list)
    audit: dict[str, Any]
    checks: Checks
    related_evidence: list = Field(default_factory=list)
    related_evidence_state: Literal["not_run"] = "not_run"
    context_sha256: str
    gap_recorded: bool = False
    warnings: list[str] = Field(default_factory=list)
    checked_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def citation_ids(candidate: Any) -> tuple[list[str], dict[str, list[str]]]:
    origins: dict[str, list[str]] = {}
    for cid in candidate.citations:
        if isinstance(cid, str):
            origins.setdefault(cid, []).append("array")
    for cid in extract_answer_citation_ids(candidate.answer):
        if cid not in origins:
            origins[cid] = []
        origins[cid].append("body")
    return list(origins), origins


def build_envelope(
    candidate: Any,
    manifest: ContextManifest,
    resolved: dict[str, dict],
    audit: dict[str, Any],
    policy: Policy,
) -> Envelope:
    """Apply priority rules to a candidate using already resolved exact IDs."""
    if policy not in ("flag", "abstain", "model_knowledge"):
        raise ValueError("on_insufficient must be 'flag', 'abstain', or 'model_knowledge'")
    ids, origins = citation_ids(candidate)
    used = candidate.used_concepts
    reasons: list[str] = []
    citations: list[Citation] = []
    too_many = len(set(ids + used)) > 32 or any(
        not isinstance(cid, str) or len(cid) > 512 or not cid for cid in ids + used
    )
    if too_many:
        reasons.append("id_limit")
    for cid in ids:
        row = resolved.get(cid, {"resolution": "error", "exists": None})
        exposure = manifest.refs.get(cid, "none")
        resolution = row.get("resolution", "error")
        citations.append(Citation(
            id=cid, kind=row.get("kind"), tier=row.get("tier"),
            exists=row.get("exists"), resolution=resolution,
            in_context=exposure != "none", exposure=exposure,
            origins=list(dict.fromkeys(origins[cid])), verdict=row.get("verdict"),
            scope=row.get("scope"),
        ))
        if resolution != "active":
            reasons.append(f"citation_{resolution}")
        elif exposure == "none":
            reasons.append("citation_out_of_context")
        if row.get("broken_provenance_ref"):
            reasons.append("broken_provenance_ref")
        if row.get("provenance_check_error"):
            reasons.append("db_check_failed")
        if row.get("scope_unknown"):
            reasons.append("scope_unknown")
    if any(cid not in manifest.concepts for cid in used):
        reasons.append("used_concepts_out_of_context")
    for cid in used:
        if cid in manifest.concepts and resolved.get(cid, {}).get("resolution") != "active":
            reasons.append(f"used_concept_{resolved.get(cid, {}).get('resolution', 'error')}")
    if audit.get("state") == "error":
        reasons.append("audit_error")
    elif any(not finding.get("advisory", False) for finding in audit.get("findings", [])):
        reasons.append("audit_finding")
    if candidate.abstain_reason == "synthesis_failed":
        reasons.append("synthesis_failed")
    model_answer_with_abstention = (policy == "model_knowledge" and candidate.abstained
                                    and bool(candidate.answer.strip()))
    if model_answer_with_abstention:
        reasons.append("model_abstained_with_body")
    elif ((candidate.abstained and candidate.answer)
          or (not candidate.abstained and not candidate.answer)):
        reasons.append("contradictory_payload")
    if audit.get("state") == "completed" and audit.get("passed") and not extract_answer_citation_ids(candidate.answer):
        reasons.append("no_inline_citations")
    if any(c.scope is None for c in citations):
        reasons.append("scope_unknown")
    if not manifest.refs:
        reasons.append("no_context")
    if any(c.resolution == "error" for c in citations):
        reasons.append("db_check_failed")
    hard = [r for r in reasons if r in {
        "id_limit", "citation_missing", "citation_retracted", "citation_ambiguous",
        "citation_error", "citation_out_of_context", "used_concepts_out_of_context",
        "used_concept_missing", "used_concept_retracted", "used_concept_ambiguous",
        "used_concept_error",
        "audit_error", "audit_finding", "synthesis_failed", "contradictory_payload",
        "db_check_failed",
    }]
    valid_citations = [c for c in citations if c.resolution == "active" and c.in_context]
    if policy == "model_knowledge":
        kb_coverage = ("none" if (candidate.abstained and not candidate.answer.strip())
                       or not valid_citations or not manifest.refs else
                       "partial" if model_answer_with_abstention else candidate.kb_coverage)
    else:
        kb_coverage = "cited" if valid_citations else "none"
    if hard:
        status = "invalid"
    elif ((candidate.abstained and not model_answer_with_abstention)
          or (not manifest.refs and policy != "model_knowledge")):
        status = "abstained"
        if candidate.abstained and (not candidate.abstain_reason or policy == "model_knowledge"):
            reasons.append("model_abstained")
    elif not citations or all(manifest.levels.get(c.id) == "derived" for c in citations):
        reasons.append("insufficient_citations")
        status = "abstained" if policy == "abstain" else "unverified"
        if status == "abstained":
            reasons.append("policy_insufficient")
    else:
        status = "unverified"
    integrity = "unknown" if (any(c.resolution == "error" for c in citations)
                              or audit.get("state") == "error"
                              or "used_concept_error" in reasons
                              or "db_check_failed" in reasons) else (
        "fail" if hard else "pass")
    # Finding details can quote the candidate body; expose only stable diagnostic codes.
    safe_audit = {"state": audit.get("state", "not_run"), "passed": audit.get("passed"),
                  "citations": [{"id": c.get("id"), "syntax": c.get("syntax"),
                                 "tier": c.get("tier"), "scope_pdks": c.get("scope_pdks")}
                                for c in audit.get("citations", [])],
                  "findings": [{"kind": f.get("kind"), "advisory": f.get("advisory", False),
                                "citation_id": f.get("citation_id")}
                               for f in audit.get("findings", [])],
                  "warnings": [code for code in ("no_inline_citations", "scope_unknown",
                                                   "broken_provenance_ref") if code in reasons]}
    return Envelope(status=status, reason_codes=list(dict.fromkeys(reasons)), policy=policy,
                    citations=citations, audit=safe_audit,
                    checks=Checks(citation_integrity=integrity, citation_count=len(citations),
                                  all_citations_derived=bool(citations) and all(
                                      manifest.levels.get(c.id) == "derived" for c in citations),
                                  kb_coverage=kb_coverage,
                                  model_knowledge=(policy == "model_knowledge" and
                                                   kb_coverage != "cited")),
                    context_sha256=manifest.digest())
