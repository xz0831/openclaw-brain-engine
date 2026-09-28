"""Match verifier — LLM arbitration for the ambiguous entity-resolution band.

When the matcher's embedding/name signals land between t_low and t_high,
neither auto-match nor auto-new is safe (e.g. "body effect" vs "backgate
bias" — same physics, "3T pixel" vs "4T pixel" — different circuits).
The verifier asks the matching-stage LLM chain a single constrained
question and parses a one-word verdict: SAME / DIFFERENT / UNSURE.

UNSURE (or any parse failure) falls through to the existing
AmbiguousMatch path, so the worst case equals current behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from openclaw_brain.config import BrainConfig
from openclaw_brain.llm.provider import LLMProvider
from openclaw_brain.llm.resilience import invoke_with_resilience, FallbackExhaustedError

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are an entity-resolution judge for a semiconductor circuit-design "
    "knowledge graph. Decide whether two entries denote the SAME underlying "
    "concept (mergeable: synonyms, abbreviations, notation variants of one "
    "thing) or DIFFERENT concepts (distinct devices, topologies, parameters, "
    "or phenomena — even if closely related). Numeric or letter designators "
    "matter: '3T pixel' and '4T pixel' are DIFFERENT. A parameter and the "
    "phenomenon it quantifies are DIFFERENT. Answer with exactly one word: "
    "SAME, DIFFERENT, or UNSURE."
)


def _response_text(content) -> str:
    """Flatten LLM response content to text. Some providers (gemini, anthropic) return a list of
    structured parts, not a string — a bare str() of that drops the text and degrades to UNSURE."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(getattr(content, "content", content) or "")


@dataclass
class VerifyResult:
    verdict: str  # "SAME" | "DIFFERENT" | "UNSURE"
    candidate_id: str


class MatchVerifier:
    """1-token SAME/DIFFERENT/UNSURE arbitration via the matching LLM chain."""

    def __init__(self, provider: LLMProvider, config: BrainConfig, auth_refresh=None,
                 model_override: str | None = None):
        self._provider = provider
        self._config = config
        self._auth_refresh = auth_refresh
        # model_override pins a single verify model (e.g. consolidation pins a local dense-27B
        # judge — see cli.py consolidate for the current pin and its evidence status) instead of
        # the matching chain.
        self._model_override = model_override

    async def verify(
        self,
        mention_name: str,
        mention_description: str,
        candidate: dict,
        neighbor_names: list[str] | None = None,
    ) -> VerifyResult:
        """Judge one mention/candidate pair. Failures degrade to UNSURE."""
        cand_id = (
            candidate.get("concept_id") or candidate.get("topology_id")
            or candidate.get("parameter_id") or candidate.get("equation_id")
            or candidate.get("principle_id") or candidate.get("canonical_name", "")
        )
        neighbors = ""
        if neighbor_names:
            neighbors = f"\nExisting entry's graph neighbors: {', '.join(neighbor_names[:6])}"
        prompt = (
            f"Entry A (newly extracted):\n"
            f"  name: {mention_name}\n"
            f"  description: {mention_description or '(none)'}\n\n"
            f"Entry B (existing in graph):\n"
            f"  name: {candidate.get('canonical_name', '?')}\n"
            f"  aliases: {', '.join(candidate.get('aliases', [])) or '(none)'}\n"
            f"  description: {candidate.get('description', '') or '(none)'}"
            f"{neighbors}\n\n"
            f"One word — SAME, DIFFERENT, or UNSURE:"
        )
        try:
            models = ([self._provider.get(self._model_override)] if self._model_override
                      else self._provider.get_chain("matching"))
            response = await invoke_with_resilience(
                models,
                [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)],
                self._config.resilience,
                auth_refresh=self._auth_refresh,
            )
            text = _response_text(getattr(response, "content", None) or response).strip().upper()
            for verdict in ("SAME", "DIFFERENT", "UNSURE"):
                if verdict in text[:24]:
                    return VerifyResult(verdict=verdict, candidate_id=cand_id)
            logger.debug("Verifier unparseable verdict: %r", text[:80])
        except FallbackExhaustedError:
            logger.warning("Verifier chain exhausted for %r", mention_name)
        except Exception as e:  # degrade, never block the pipeline
            logger.warning("Verifier error for %r: %s", mention_name, e)
        return VerifyResult(verdict="UNSURE", candidate_id=cand_id)
