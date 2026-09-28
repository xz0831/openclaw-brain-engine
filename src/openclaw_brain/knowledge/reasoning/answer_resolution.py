"""Bounded read-only exact-ID resolution for the answer contract."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from neo4j import Query

from openclaw_brain.knowledge.executable.lesson import _CERTIFIED_VERDICTS, canonical_pdk

_LABEL_FIELDS = {
    "ClaimCard": "claim_id", "Regularity": "law_id", "Concept": "concept_id",
    "Equation": "equation_id", "Insight": "insight_id", "Parameter": "parameter_id",
    "CircuitTopology": "topology_id", "Principle": "principle_id", "Specimen": "spec_id",
    "SourceChunk": "chunk_id",
}
_PREDICATE = " OR ".join(f"(n:{label} AND n.{field} = $cid)"
                         for label, field in _LABEL_FIELDS.items())
_QUERY = (f"MATCH (n) WHERE {_PREDICATE} "
          "RETURN labels(n) AS labels, n.retracted AS retracted, n.verdict AS verdict, "
          "n.engine AS engine, n.status AS status, n.scope AS scope, "
          "n.member_summary AS member_summary LIMIT 2")


def _object(value: Any) -> dict | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else None
        except ValueError:
            return None
    return None


def classify_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"resolution": "missing", "exists": False}
    if len(rows) > 1:
        return {"resolution": "ambiguous", "exists": True}
    row = rows[0]
    labels = row.get("labels") or []
    if not labels:
        return {"resolution": "error", "exists": None}
    label = next((label for label in _LABEL_FIELDS if label in labels), None)
    if label is None:
        return {"resolution": "error", "exists": None}
    if row.get("retracted"):
        return {"resolution": "retracted", "exists": True, "kind": label}
    result = {"resolution": "active", "exists": True, "kind": label,
              "tier": 0 if label == "SourceChunk" else 1,
              "verdict": row.get("verdict") if label == "ClaimCard" else None,
              "scope": None, "scope_pdks": None}
    if label == "ClaimCard":
        scope = _object(row.get("scope"))
        result["scope"] = scope
        pdk = scope.get("pdk") if scope else None
        result["scope_pdks"] = [canonical_pdk(pdk)] if isinstance(pdk, str) and pdk else None
        if row.get("verdict") and row.get("engine"):
            result["tier"] = 3
        else:
            result["tier"] = None
    elif label == "Regularity":
        members = _object(row.get("member_summary"))
        positive = set()
        negative = False
        if members is not None:
            for pdk, member in members.items():
                if not isinstance(member, dict):
                    negative = True
                    continue
                verdict = member.get("verdict")
                if verdict in _CERTIFIED_VERDICTS:
                    positive.add(canonical_pdk(pdk))
                elif verdict:
                    negative = True
            result["scope_pdks"] = sorted(positive)
            result["scope"] = {"pdks": sorted(positive)}
        else:
            result["scope_unknown"] = True
        result["tier"] = (4 if row.get("status") == "law" and len(positive) >= 3
                          and not negative else None)
    return result


async def resolve_exact(graph: Any, ids: list[str], *, budget_s: float = 2.0) -> dict[str, dict]:
    """At most one graph read per distinct ID, with both client and server deadlines."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget_s
    resolved: dict[str, dict] = {}
    for cid in dict.fromkeys(ids):
        remaining = deadline - loop.time()
        if remaining <= 0:
            resolved[cid] = {"resolution": "error", "exists": None}
            continue
        try:
            rows = await asyncio.wait_for(
                graph.run_read_query(Query(_QUERY, timeout=remaining), {"cid": cid}),
                timeout=remaining)
            resolved[cid] = classify_rows(rows)
        except Exception:
            resolved[cid] = {"resolution": "error", "exists": None}
    return resolved
