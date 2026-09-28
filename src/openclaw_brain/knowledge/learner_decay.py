"""UNDERSTANDS-edge confidence decay — S5 learner model (docs/specs/S5_LEARNER_MODEL_DESIGN.md
§3.2, Option C, the locked recommendation).

This is deliberately a small, standalone module — NOT a method on ``ReinforcementEngine``
(``knowledge/reinforcement.py``) and NOT part of ``PromotionPipeline`` (``memory/promotion.py``).
Both were considered and rejected by the design doc:

- ``ReinforcementEngine.apply_decay()`` decays a Concept's own truth-confidence ("how sure is
  the system this fact is correct"). An UNDERSTANDS edge answers a different question entirely
  ("does *this learner* still remember it") — routing it through the same mechanism would let a
  purely pedagogical signal leak into the truth layer's confidence number, exactly the kind of
  contamination ``mechanism_never_fact``/``audit_citations`` exist to prevent elsewhere in this
  codebase.
- ``PromotionPipeline``'s RAW→RETAIN→CURATED sweep is a one-way, usage-based ratchet-forward — a
  tier ratchet is not a decay curve.

So: a new, tiny, ``UNDERSTANDS``-edge-scoped decay function, config-driven half-life
(``[learner]`` in ``config/default.toml`` — see ``config.LearnerConfig``), touching neither of
the above. Not wired into ``run_maintenance`` or any MCP tool by this change — the design doc's
approved S5a scope (§6.1) does not call for a call site, so this ships as a standalone, tested,
callable primitive (mirrors how ``SemanticMemory.record_fact``/``link_to_entity``/
``link_to_concept`` already exist as callable-but-uncalled code in this codebase, per
``memory/README.md`` Trap 4) — wiring it up (a CLI command, a new MCP tool, or a
``run_maintenance`` composition change) is left as a deliberate follow-up decision, not silently
bundled into S5a.
"""

from __future__ import annotations

from openclaw_brain.config import LearnerConfig
from openclaw_brain.knowledge.graph.store import GraphStore

# ln(2) — exponential half-life decay: confidence = floor + (confidence - floor) * 2^(-t/half_life),
# rewritten as exp(-ln(2) * t / half_life) so the Cypher only needs the one built-in exp().
_LN2 = 0.6931471805599453


async def apply_understands_decay(graph: GraphStore, learner_config: LearnerConfig) -> int:
    """Fade every ``Learner -[:UNDERSTANDS]-> target`` edge's ``confidence`` toward
    ``learner_config.understands_confidence_floor``, exponentially, with a half-life of
    ``learner_config.understands_half_life_days`` since ``last_assessed``.

    An edge that has never been assessed (no ``last_assessed``) or is already at/under the
    floor is left untouched. Returns the number of edges affected.
    """
    half_life = learner_config.understands_half_life_days
    floor = learner_config.understands_confidence_floor
    async with await graph._session() as session:
        result = await session.run(
            """
            MATCH (:Learner)-[u:UNDERSTANDS]->()
            WHERE u.last_assessed IS NOT NULL
              AND coalesce(u.confidence, $floor) > $floor
            WITH u, duration.between(datetime(u.last_assessed), datetime()).days AS days_since
            WHERE days_since > 0
            SET u.confidence = $floor + (coalesce(u.confidence, $floor) - $floor)
                * exp(-$ln2 * toFloat(days_since) / $half_life)
            RETURN count(u) AS affected
            """,
            {"half_life": half_life, "floor": floor, "ln2": _LN2},
        )
        record = await result.single()
        return record["affected"] if record else 0
