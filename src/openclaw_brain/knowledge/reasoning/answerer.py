"""Lean grounded answer synthesis over retrieved knowledge context."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any, Literal, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from openclaw_brain.llm.resilience import invoke_with_resilience, resolve_model_name

logger = logging.getLogger(__name__)


class AnswerResult(BaseModel):
    """Structured grounded answer result."""

    answer: str = Field(
        default="",
        description="Grounded prose answer; empty string if abstained",
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Concept ids and/or chunk ids from the provided context that back the answer",
    )
    abstained: bool = Field(
        default=False,
        description="True if the context was insufficient to answer faithfully",
    )
    abstain_reason: str = Field(
        default="",
        description="If abstained, what specifically was missing",
    )
    used_concepts: list[str] = Field(
        default_factory=list,
        description="Concept ids actually used in the answer (subset of provided concept ids)",
    )
    kb_coverage: Literal["cited", "partial", "none"] = Field(
        default="none",
        description="How much of the answer is supported by openclaw-brain citations",
    )
    missing_knowledge: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Up to five short phrases describing knowledge missing from openclaw-brain",
    )

    @field_validator("missing_knowledge", mode="before")
    @classmethod
    def _limit_missing_knowledge(cls, value):
        return value[:5] if isinstance(value, list) else value


_ANSWER_SYSTEM_PROMPT = """You are a precise semiconductor engineering assistant.
Answer ONLY from the provided knowledge-graph context: concepts, typed graph connections,
decision context, and memories.

Every factual claim MUST be backed by a citation drawn from the provided concept ids or
chunk ids. Put those ids in `citations`.

If the provided context does not contain enough to answer faithfully, set
`abstained=true` and state precisely what is missing in `abstain_reason`. Do NOT
fabricate facts or cite ids that are not in the context.

If you must rely on a `derived`/un-cited item, explicitly label that sentence in the
answer as *derived/unverified*.

Return strict JSON with keys: answer, citations, used_concepts, abstained,
abstain_reason.

Abstention is YOUR judgment; no external rule decides it. Prefer a faithful abstention
over a confident but ungrounded answer."""

_MODEL_KNOWLEDGE_PROMPT = """You are a precise semiconductor engineering assistant.
Answer the question using the supplied openclaw-brain context where it supports a claim.
Cite only concept or chunk ids actually supplied in the context, in both the answer and
`citations`. For unsupported parts, use your general model knowledge and mark each such
part [모델 지식]. If there is no openclaw-brain context, answer from general model knowledge,
explicitly recognize "openclaw-brain 근거 없음", set kb_coverage="none", and return no
citations or used_concepts. The application adds the fixed disclosure to the answer.
Under this policy, do not set abstained=true when providing an answer. If there is no
openclaw-brain evidence, disclose that absence and answer from general model knowledge.

Return strict JSON with answer, citations, used_concepts, abstained, abstain_reason,
kb_coverage ("cited", "partial", or "none"), and missing_knowledge (at most five short
phrases describing what openclaw-brain should learn). Use "cited" only when the whole
answer is supported by citations; "partial" for a mixture; "none" for no supported
claims. Do not present model knowledge as an openclaw-brain measurement or verified result.
If asked for a graph measurement absent from context, say "그래프에 없음". Never invent a
measurement. Do not invent citations."""


async def answer_from_context(
    *,
    question: str,
    context_markdown: str,
    concept_refs: list[dict],
    open_hypotheses: list[dict],
    active_decisions: list[dict],
    model_chain: Sequence[BaseChatModel],
    resilience_config,
    auth_refresh=None,
    on_insufficient: Literal["flag", "abstain", "model_knowledge"] = "flag",
    no_context: bool = False,
) -> AnswerResult:
    """Synthesize an answer from retrieved context under the selected policy."""
    messages = [
        SystemMessage(content=(_MODEL_KNOWLEDGE_PROMPT if on_insufficient == "model_knowledge"
                               else _ANSWER_SYSTEM_PROMPT)),
        HumanMessage(
            content=json.dumps(
                {
                    "question": question,
                    "context": context_markdown,
                    "concepts": concept_refs,
                    "open_hypotheses": open_hypotheses,
                    "active_decisions": active_decisions,
                },
                ensure_ascii=False,
            )
        ),
    ]

    primary = model_chain[0]
    if on_insufficient == "model_knowledge" and no_context:
        # The empty-context policy permits exactly one model attempt, with no retry,
        # fallback, or second structured-output request.
        try:
            from openclaw_brain.knowledge.reasoning.normalize import extract_json

            response = await invoke_with_resilience(
                [primary], messages, replace(resilience_config, max_retries=0),
                auth_refresh=auth_refresh,
            )
            raw_text = response.content if hasattr(response, "content") else str(response)
            return AnswerResult(**_normalize_answer_payload(extract_json(raw_text)))
        except Exception:
            logger.warning("Answer synthesis failed for empty context")
            return AnswerResult(abstained=True, abstain_reason="synthesis_failed",
                                kb_coverage="none")
    try:
        structured = primary.with_structured_output(AnswerResult)
        # Route through invoke_with_resilience (was a bare .ainvoke() + manual wait_for) so this
        # attempt gets retry/backoff, OAuth-refresh-on-401, and circuit-breaker participation like
        # every other resilience-covered call site — a persistently bad-window primary no longer
        # pays its full request_timeout_s on every call before falling through to the raw-chain
        # path below (llm/README.md Traps). model_names is computed from the RAW `primary`
        # (before with_structured_output() wraps it): with_structured_output() returns a
        # RunnableSequence that exposes neither .model_name nor .model, so resolve_model_name on
        # the WRAPPED object would collapse to "unknown" and cross-contaminate the breaker with
        # every other wrapped caller (see resolve_model_name's docstring).
        # No truncation_retry here (it IS set on the raw path below): with_structured_output()
        # returns a RunnableSequence whose result is the PARSED pydantic object, which exposes
        # no response_metadata/generations — detect_finish_reason can only ever return None, so
        # the flag would be a guaranteed no-op. A truncation on this path surfaces as a
        # parse/validation error, which the `except` already routes to the raw path.
        result = await invoke_with_resilience(
            [structured],
            messages,
            resilience_config,
            model_names=[resolve_model_name(primary)],
            auth_refresh=auth_refresh,
        )
        return AnswerResult.model_validate(result)
    except Exception:
        logger.info("Structured output failed on primary model, using raw + normalize")

    try:
        from openclaw_brain.knowledge.reasoning.normalize import extract_json

        response = await invoke_with_resilience(
            list(model_chain),
            messages,
            resilience_config,
            auth_refresh=auth_refresh,
            # This chain resolves stage='reasoning', so it now carries the stage's
            # [reasoning].output_token_budget bound — and a truncated ANSWER is the worst
            # failure of the three reasoning-stage surfaces: this is the Hermes-facing answer
            # text the citation audits run on, so a silently clipped body would be audited as
            # if it were the whole claim (a citation cut off mid-sentence still "cites").
            # Discard + one escalated retry + fall through the chain instead.
            truncation_retry=True,
        )
        raw_text = response.content if hasattr(response, "content") else str(response)
        raw_json = extract_json(raw_text)
        normalized = _normalize_answer_payload(raw_json)
        return AnswerResult(**normalized)
    except Exception as exc:
        logger.warning("Answer synthesis failed: %s", exc)
        return AnswerResult(abstained=True, abstain_reason="synthesis_failed")


def _normalize_answer_payload(raw_json: dict[str, Any]) -> dict[str, Any]:
    """Accept the production schema plus the earlier A3 prototype aliases."""
    normalized = dict(raw_json)
    if "answer" not in normalized and "prose_answer" in normalized:
        normalized["answer"] = normalized["prose_answer"]
    return normalized
