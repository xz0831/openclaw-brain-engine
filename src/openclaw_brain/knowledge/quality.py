"""Triplet precision quality audit for distilled knowledge nodes."""

from __future__ import annotations

import asyncio
import logging
import random
import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Mapping, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from openclaw_brain.knowledge.evidence import EvidenceVault
from openclaw_brain.knowledge.graph.store import GraphStore
from openclaw_brain.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

AuditFetchNodes = Callable[[str], Awaitable[list[dict[str, Any]]]]
AuditFetchChunks = Callable[[str], Awaitable[list[dict[str, Any]]]]
AuditJudge = Callable[[dict[str, Any], list[dict[str, Any]]], Awaitable["Verdict"]]

SUPPORTED_VERDICTS = {"SUPPORTED", "OVERGENERALIZED", "UNSUPPORTED", "NO_EVIDENCE"}
AUDIT_VERDICTS = SUPPORTED_VERDICTS | {"JUDGE_ERROR"}

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from",
    "has", "have", "in", "into", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "this", "to", "under", "using", "via", "when", "where",
    "which", "with", "within",
})


class Verdict(BaseModel):
    """Structured LLM judge result for one audited node."""

    verdict: Literal["SUPPORTED", "OVERGENERALIZED", "UNSUPPORTED", "NO_EVIDENCE"]
    # No max_length: a verbose-but-valid judge verdict should not be discarded as a
    # JUDGE_ERROR just because its reason ran long. The reason is truncated to one
    # sentence post-validation (see _one_sentence at the call site).
    reason: str = Field(default="")


class JudgeFailedError(RuntimeError):
    """Raised after all judge attempts have failed."""


def select_evidence(
    description: str,
    chunks: Sequence[Mapping[str, Any]],
    *,
    embed_rank: Any = None,
    embed_prefilter: int = 80,
) -> tuple[list[dict[str, Any]], bool]:
    """Rank SourceChunk evidence text and return the top two.

    Token overlap by default. When ``embed_rank`` is provided, the top
    ``embed_prefilter`` chunks by token overlap are re-ranked by semantic
    similarity (``embed_rank(description, [chunk_texts]) -> [score]``). This is a
    retrieve-then-rerank: the supporting chunk shares the concept's surface terms so
    it survives the coarse overlap filter, then embeddings pull it to the top — fixing
    the failure mode where overlap alone buried the supporting chunk below top-2 and
    inflated UNSUPPORTED verdicts. The prefilter caps embedding cost/memory (a whole
    textbook source can hold thousands of chunks).
    """
    if not chunks:
        return [], False

    description_tokens = _tokens(description)
    texts = [str(c.get("text") or c.get("text_preview") or "") for c in chunks]
    overlaps = [
        len(description_tokens & _tokens(f"{str(c.get('section_title') or '')} {texts[i]}"))
        for i, c in enumerate(chunks)
    ]

    semantic: dict[int, float] | None = None
    if embed_rank is not None:
        cand_idx = sorted(range(len(chunks)), key=lambda i: -overlaps[i])[: max(2, embed_prefilter)]
        cand_scores = embed_rank(description, [texts[i] for i in cand_idx])
        semantic = {cand_idx[j]: float(cand_scores[j]) for j in range(len(cand_idx))}

    ranked: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        item = dict(chunk)
        item["overlap_score"] = overlaps[idx]
        if semantic is not None:
            item["semantic_score"] = round(semantic.get(idx, 0.0), 4)
        item["_rank_index"] = idx
        item["_sort_score"] = (
            semantic.get(idx, -1.0) if semantic is not None else float(overlaps[idx])
        )
        item["verbatim"] = bool(
            item.get("verbatim")
            or (item.get("raw_text_hash") and item.get("text"))
        )
        ranked.append(item)

    ranked.sort(key=lambda item: (-item["_sort_score"], int(item["_rank_index"])))
    selected = []
    for item in ranked[:2]:
        item = dict(item)
        item.pop("_rank_index", None)
        item.pop("_sort_score", None)
        selected.append(item)
    truncated = any(not bool(item.get("verbatim")) for item in selected)
    return selected, truncated


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def build_semantic_ranker(config: Any):
    """Return an ``embed_rank(description, texts) -> [cosine]`` over the graph's embedder.

    Embeds the node description as a query and each candidate chunk as a document
    with the same model the graph was built with, then ranks by cosine similarity.
    """
    from openclaw_brain.knowledge.embedding import encode, encode_batch

    emb = config.embedding

    def embed_rank(description: str, texts: Sequence[str]) -> list[float]:
        query = encode(
            description,
            emb.model,
            is_query=True,
            query_instruction=emb.query_instruction,
            truncate_dim=emb.truncate_dim,
        )
        docs = encode_batch(list(texts), emb.model, truncate_dim=emb.truncate_dim)
        return [_cosine(query, d) for d in docs]

    return embed_rank


def sample_nodes(
    nodes: Sequence[Mapping[str, Any]],
    sample: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Return a deterministic Python-shuffled sample of candidate nodes."""
    pool = [dict(node) for node in nodes]
    random.Random(seed).shuffle(pool)
    if sample < 0:
        return pool
    return pool[:sample]


def summarize(verdicts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize audit verdicts, excluding judge errors from precision math."""
    counts = {verdict: 0 for verdict in sorted(AUDIT_VERDICTS)}
    evidence_truncated_count = 0
    verbatim_evidence_count = 0
    for item in verdicts:
        verdict = str(item.get("verdict") or "")
        if verdict in counts:
            counts[verdict] += 1
        if item.get("evidence_truncated"):
            evidence_truncated_count += 1
        verbatim_evidence_count += int(item.get("verbatim_evidence_count") or 0)

    n = len(verdicts)
    judge_error_count = counts["JUDGE_ERROR"]
    judged = n - judge_error_count
    judged_with_evidence = (
        counts["SUPPORTED"] + counts["OVERGENERALIZED"] + counts["UNSUPPORTED"]
    )

    return {
        "n": n,
        "judged": judged,
        "judged_with_evidence": judged_with_evidence,
        "precision": _rate(counts["SUPPORTED"], judged_with_evidence),
        "overgeneralized_rate": _rate(counts["OVERGENERALIZED"], judged_with_evidence),
        "unsupported_rate": _rate(counts["UNSUPPORTED"], judged_with_evidence),
        "no_evidence_rate": _rate(counts["NO_EVIDENCE"], judged),
        "judge_error_count": judge_error_count,
        "evidence_truncated_count": evidence_truncated_count,
        "evidence_truncated": evidence_truncated_count > 0,
        "verbatim_evidence_count": verbatim_evidence_count,
        "verdict_counts": counts,
    }


def should_fail_precision(summary: Mapping[str, Any], fail_below: float) -> bool:
    """Return whether the CLI should exit non-zero for the precision threshold."""
    return float(summary.get("precision") or 0.0) < fail_below


async def run_quality_audit(
    *,
    labels: Sequence[str],
    sample: int,
    seed: int,
    fetch_nodes: AuditFetchNodes,
    fetch_chunks: AuditFetchChunks,
    judge: AuditJudge,
    params: Mapping[str, Any] | None = None,
    embed_rank: Any = None,
) -> dict[str, Any]:
    """Run the quality audit using injected storage and judge callables."""
    candidates: list[dict[str, Any]] = []
    for label in labels:
        label_nodes = await fetch_nodes(label)
        for node in label_nodes:
            description = str(node.get("description") or "")
            if not description.strip():
                continue
            item = dict(node)
            item["label"] = label
            candidates.append(item)

    items: list[dict[str, Any]] = []
    for node in sample_nodes(candidates, sample, seed):
        item_base = _item_base(node)
        source_id = node.get("source_id")
        if not source_id:
            items.append({
                **item_base,
                "verdict": "NO_EVIDENCE",
                "reason": "Node has no source_id.",
                "evidence_chunk_ids": [],
                "overlap_score": 0,
                "evidence_truncated": False,
                "verbatim_evidence_count": 0,
            })
            continue

        chunks = await fetch_chunks(str(source_id))
        evidence, truncated = select_evidence(
            str(node.get("description") or ""), chunks, embed_rank=embed_rank
        )
        if not evidence:
            items.append({
                **item_base,
                "verdict": "NO_EVIDENCE",
                "reason": "No SourceChunk previews found for source_id.",
                "evidence_chunk_ids": [],
                "overlap_score": 0,
                "evidence_truncated": False,
                "verbatim_evidence_count": 0,
            })
            continue

        try:
            verdict = await judge(node, evidence)
            verdict_value = verdict.verdict
            reason = _one_sentence(verdict.reason)
        except Exception as exc:
            logger.warning("Quality judge failed for %s: %s", item_base["id"], exc)
            verdict_value = "JUDGE_ERROR"
            reason = _one_sentence(f"Judge failed after retries: {exc}")

        items.append({
            **item_base,
            "verdict": verdict_value,
            "reason": reason,
            "evidence_chunk_ids": [
                str(chunk.get("chunk_id"))
                for chunk in evidence
                if chunk.get("chunk_id") is not None
            ],
            "overlap_score": int(evidence[0].get("overlap_score") or 0),
            "evidence_truncated": truncated,
            "verbatim_evidence_count": sum(1 for chunk in evidence if chunk.get("verbatim")),
        })

    summary = summarize(items)
    summary["params"] = dict(params or {})
    return {"items": items, "summary": summary}


async def fetch_nodes_from_graph(store: GraphStore, label: str) -> list[dict[str, Any]]:
    """Fetch audit candidates for one Neo4j label."""
    _validate_label(label)
    query = f"""
    MATCH (n:{label})
    WITH n,
         coalesce(
            n.concept_id, n.principle_id, n.insight_id, n.topology_id,
            n.equation_id, n.parameter_id, n.assumption_id, n.hypothesis_id,
            n.decision_id, n.bench_id, n.memory_id, n.entity_id, elementId(n)
         ) AS id,
         coalesce(n.canonical_name, n.name, n.statement, n.symbol, elementId(n)) AS canonical_name,
         coalesce(n.description, n.statement, n.rationale, n.conclusion, n.scope, n.name) AS description
    WHERE description IS NOT NULL AND trim(toString(description)) <> ''
    RETURN toString(id) AS id,
           toString(canonical_name) AS canonical_name,
           toString(description) AS description,
           n.source_id AS source_id
    ORDER BY id
    """
    return await store.run_read_query(query)


async def fetch_chunks_from_graph(
    store: GraphStore,
    source_id: str,
    vault: EvidenceVault | None = None,
) -> list[dict[str, Any]]:
    """Fetch SourceChunk evidence for a source_id, hydrating vault text when available."""
    query = """
    MATCH (sc:SourceChunk {source_id: $sid})
    RETURN sc.chunk_id AS chunk_id,
           sc.source_id AS source_id,
           sc.pages AS pages,
           sc.text_preview AS text_preview,
           sc.section_title AS section_title,
           sc.raw_text_hash AS raw_text_hash
    ORDER BY sc.chunk_id
    """
    chunks = await store.run_read_query(query, {"sid": source_id})
    if vault is None:
        return chunks

    hydrated: list[dict[str, Any]] = []
    for chunk in chunks:
        item = dict(chunk)
        item["verbatim"] = False
        raw_text_hash = str(item.get("raw_text_hash") or "")
        if raw_text_hash:
            try:
                text = vault.get_text(raw_text_hash)
            except Exception as exc:
                logger.warning("Evidence vault read failed for %s: %s", raw_text_hash, exc)
                text = None
            if text is not None:
                item["text"] = text
                item["verbatim"] = True
        hydrated.append(item)
    return hydrated


class LLMQualityJudge:
    """LLM-backed scope-fidelity judge with one retry and a fallback model."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        fallback_model: str | None = None,
        timeout_s: float = 900.0,
    ):
        self._provider = provider
        self._model = model
        self._fallback_model = fallback_model
        self._timeout_s = timeout_s

    async def __call__(self, node: dict[str, Any], evidence: list[dict[str, Any]]) -> Verdict:
        messages = _judge_messages(node, evidence)
        errors: list[str] = []

        attempts: list[tuple[str, int]] = [(self._model, 2)]
        if self._fallback_model and self._fallback_model != self._model:
            attempts.append((self._fallback_model, 1))

        for model_name, attempt_count in attempts:
            llm = self._provider.get(model_name)
            structured = llm.with_structured_output(Verdict)
            for attempt in range(1, attempt_count + 1):
                try:
                    result = structured.ainvoke(messages)
                    raw = (
                        await asyncio.wait_for(result, timeout=self._timeout_s)
                        if self._timeout_s > 0
                        else await result
                    )
                    verdict = raw if isinstance(raw, Verdict) else Verdict.model_validate(raw)
                    verdict.reason = _one_sentence(verdict.reason)
                    return verdict
                except Exception as exc:
                    errors.append(f"{model_name} attempt {attempt}: {exc}")
                    logger.debug("Quality judge attempt failed: %s", errors[-1])

        raise JudgeFailedError("; ".join(errors))


def format_compact_table(result: Mapping[str, Any]) -> str:
    """Render a compact stdout summary for the CLI."""
    summary = dict(result.get("summary") or {})
    counts = dict(summary.get("verdict_counts") or {})
    n = int(summary.get("n") or 0)
    lines = [
        "verdict             count   share",
        "------------------  -----  ------",
    ]
    for verdict in (
        "SUPPORTED", "OVERGENERALIZED", "UNSUPPORTED", "NO_EVIDENCE", "JUDGE_ERROR",
    ):
        count = int(counts.get(verdict) or 0)
        lines.append(f"{verdict:<18} {count:>5}  {_rate(count, n):>6.1%}")

    lines.extend([
        "",
        f"judged={summary.get('judged', 0)} "
        f"judged_with_evidence={summary.get('judged_with_evidence', 0)} "
        f"precision={float(summary.get('precision') or 0.0):.3f}",
        f"overgeneralized_rate={float(summary.get('overgeneralized_rate') or 0.0):.3f} "
        f"unsupported_rate={float(summary.get('unsupported_rate') or 0.0):.3f} "
        f"no_evidence_rate={float(summary.get('no_evidence_rate') or 0.0):.3f}",
        f"evidence_truncated_items={summary.get('evidence_truncated_count', 0)}",
        f"verbatim_evidence_count={summary.get('verbatim_evidence_count', 0)}",
    ])
    return "\n".join(lines)


def write_quality_audit_json(result: Mapping[str, Any], out_path: str | Path) -> None:
    """Write audit output JSON."""
    import json

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in (match.group(0).lower() for match in _TOKEN_RE.finditer(text))
        if token not in _STOPWORDS
    }


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _item_base(node: Mapping[str, Any]) -> dict[str, Any]:
    node_id = str(node.get("id") or node.get("canonical_name") or "")
    name = str(node.get("canonical_name") or node.get("name") or node_id)
    return {
        "id": node_id,
        "name": name,
        "label": str(node.get("label") or ""),
    }


def _validate_label(label: str) -> None:
    if not _LABEL_RE.fullmatch(label):
        raise ValueError(f"Invalid Neo4j label: {label!r}")


def _judge_messages(node: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]):
    evidence_text = "\n\n".join(
        f"[{chunk.get('chunk_id')}] {chunk.get('section_title') or ''}\n"
        f"{chunk.get('text') or chunk.get('text_preview') or ''}"
        for chunk in evidence
    )
    system = """You are a precision audit judge for a technical knowledge graph.

Classify whether the node description is faithfully supported by the provided source evidence.

Verdicts:
- SUPPORTED = the description is supported and preserves the evidence scope.
- OVERGENERALIZED = the description widens or drops a scope the evidence states (e.g. evidence scopes a claim to shot noise / a regime / a device type, description claims it for 'total noise' / unconditionally).
- UNSUPPORTED = the description contradicts the evidence or makes a claim not present in it.
- NO_EVIDENCE = the evidence is empty or unusable.

Few-shot 1:
Evidence: Under CMS the shot-noise integration variance increases with M, with alpha_shot = M/3.
Description: Total noise variance increases linearly with M.
Verdict: OVERGENERALIZED
Reason: The evidence scopes the increase to shot-noise integration variance, not total noise.

Few-shot 2:
Evidence: Under CMS the shot-noise integration variance increases with M, with alpha_shot = M/3.
Description: Under CMS, shot-noise integration variance increases with M.
Verdict: SUPPORTED
Reason: The description preserves the CMS and shot-noise scope.

Return a one-sentence reason."""
    human = (
        f"Node label: {node.get('label')}\n"
        f"Node name: {node.get('canonical_name')}\n"
        f"Description: {node.get('description')}\n\n"
        f"Evidence previews:\n{evidence_text}"
    )
    return [SystemMessage(content=system), HumanMessage(content=human)]


def _one_sentence(reason: str) -> str:
    compact = " ".join(str(reason or "").split())
    if not compact:
        return ""
    match = re.match(r"^(.+?[.!?])(?:\s|$)", compact)
    return match.group(1) if match else compact
