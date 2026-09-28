"""Offline consolidation — retro-dedup of co-referent Concept nodes.

The graph accumulated duplicates while entity resolution was surface-form
only ('body effect' vs 'backgate bias').  Duplicates fragment edges, split
reinforcement counts, and silently degrade retrieval and bridge discovery —
so this pass finds co-referent Concept pairs and merges them.

Bands (evidence: experiments/CANONICALIZATION_PROMOTION.md — embedding cosine tops at live
precision ~0.92, so it is a candidate GATE, not a merge decision):
  auto    merge_tier == MERGE (lexical identity: normalized-name / alias / acronym) OR graph alias
  verify  merge_tier == VERIFY (embedding candidate ≥ t_low) — drained by an LLM verifier (qwen)
  review  conflict-guard veto on a strong signal → JSONL queue for a human
NOTE: a high cosine alone NEVER auto-merges — it routes to verify.

Safety: dry-run by default; auto band only merges when ``auto_merge=True``
(the dossier mandates ~1 week of auto-merge-OFF operation first); union-find
clusters are capped (merge-cascade brake); every merge is journaled with the
duplicate's full snapshot AND every rewired edge so it can be undone.

v1 scope is Concept-only (GraphStore.merge_concepts is Concept-only).
Generalizing to Equation/Parameter is an explicit follow-up.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from openclaw_brain.config import BrainConfig
from openclaw_brain.knowledge.graph.store import GraphStore
from openclaw_brain.knowledge.canonicalization import merge_tier, qualifier_conflict
from openclaw_brain.knowledge.reasoning.matcher import _name_similarity, _names_conflict, _normalize_name

logger = logging.getLogger(__name__)

# Pairs below this in BOTH cosine and name-sim are not even queued for review.
_REVIEW_FLOOR_COS = 0.60
_REVIEW_FLOOR_NS = 0.70


@dataclass
class MergeCandidate:
    a_id: str
    b_id: str
    a_name: str
    b_name: str
    cos: float | None
    name_sim: float
    band: str  # "auto" | "verify" | "review"
    reason: str = ""


@dataclass
class ConsolidationReport:
    concepts_scanned: int = 0
    pairs_considered: int = 0
    auto: list[MergeCandidate] = field(default_factory=list)
    verify: list[MergeCandidate] = field(default_factory=list)
    review: list[MergeCandidate] = field(default_factory=list)
    merged: list[dict[str, Any]] = field(default_factory=list)
    dry_run: bool = True

    def summary(self) -> dict[str, Any]:
        return {
            "concepts_scanned": self.concepts_scanned,
            "pairs_considered": self.pairs_considered,
            "auto_band": len(self.auto),
            "verify_band": len(self.verify),
            "review_band": len(self.review),
            "merged": len(self.merged),
            "dry_run": self.dry_run,
        }


class ConsolidationEngine:
    """Find and merge co-referent Concept nodes."""

    def __init__(self, graph: GraphStore, config: BrainConfig, journal=None, verifier=None):
        self._graph = graph
        self._config = config
        self._journal = journal
        self._verifier = verifier

    async def _fetch_concepts(self) -> list[dict]:
        query = """
        MATCH (c:Concept)
        WHERE NOT coalesce(c.retracted, false)
        OPTIONAL MATCH (c)-[r]-()
        WITH c, count(r) AS degree
        RETURN c.concept_id AS id, c.canonical_name AS name,
               coalesce(c.aliases, []) AS aliases,
               coalesce(c.description, '') AS description,
               c.embedding AS embedding,
               degree,
               coalesce(c.reinforcement_count, 0) AS reinforcement,
               toString(c._created_at) AS created_at
        """
        return await self._graph.run_read_query(query, {})

    def _classify_pairs(self, concepts: list[dict]) -> tuple[list[MergeCandidate], int]:
        """All-pairs scoring: vector cosine (numpy) + lexical pass."""
        import numpy as np

        candidates: list[MergeCandidate] = []
        n = len(concepts)
        pairs_considered = 0
        # Dedicated VERIFY-tier gate (NOT matcher.t_low — that is the per-chunk match boundary).
        # Still collect cosines down to the review floor so the conflict-veto review band works.
        candidate_floor = self._config.consolidation.candidate_floor

        # Vector pass — only nodes that have embeddings.
        embedded = [(i, c) for i, c in enumerate(concepts) if c.get("embedding")]
        cos_lookup: dict[tuple[int, int], float] = {}
        if len(embedded) >= 2:
            idxs = [i for i, _ in embedded]
            mat = np.array([c["embedding"] for _, c in embedded], dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
            sims = mat @ mat.T
            hi, hj = np.where(np.triu(sims, k=1) >= min(candidate_floor, _REVIEW_FLOOR_COS))
            for a, b in zip(hi.tolist(), hj.tolist()):
                cos_lookup[(idxs[a], idxs[b])] = float(sims[a, b])

        # Lexical pass — name-sim catches pairs whose embeddings are missing.
        seen: set[tuple[int, int]] = set(cos_lookup.keys())
        for i in range(n):
            for j in range(i + 1, n):
                key = (i, j)
                cos = cos_lookup.get(key)
                if cos is None and key in seen:
                    continue
                ns = _name_similarity(concepts[i]["name"] or "", concepts[j]["name"] or "")
                if cos is None and ns < _REVIEW_FLOOR_NS:
                    continue
                pairs_considered += 1
                ci, cj = concepts[i], concepts[j]
                na, nb = ci["name"] or "", cj["name"] or ""
                alias_hit = self._alias_hit(ci, cj)
                # Conflict = matcher primitive guard (digit mismatch / near-zero-sim) PLUS the
                # canonicalization qualifier guard (base vs specialization — the precision fix).
                conflict = _names_conflict(na, ci["description"], nb, cj["description"], ns) \
                    or qualifier_conflict(na, nb)
                tier = merge_tier(na, nb, cosine=cos, candidate_floor=candidate_floor)
                band, reason = self._band_for(tier, conflict, cos, ns, alias_hit)
                if band is None:
                    continue
                candidates.append(MergeCandidate(
                    a_id=ci["id"], b_id=cj["id"], a_name=ci["name"], b_name=cj["name"],
                    cos=cos, name_sim=round(ns, 3), band=band, reason=reason,
                ))
        return candidates, pairs_considered

    @staticmethod
    def _alias_hit(a: dict, b: dict) -> bool:
        na, nb = _normalize_name(a["name"] or ""), _normalize_name(b["name"] or "")
        aliases_a = {_normalize_name(x) for x in a.get("aliases", []) if x}
        aliases_b = {_normalize_name(x) for x in b.get("aliases", []) if x}
        return nb in aliases_a or na in aliases_b or bool(aliases_a & aliases_b)

    @staticmethod
    def _band_for(
        tier: str, conflict: bool, cos: float | None, ns: float, alias_hit: bool,
    ) -> tuple[str | None, str]:
        # A conflict (qualifier/numeric/specialization) never auto-merges; a strong signal
        # still earns human review (it might be a true synonym the guard over-vetoed).
        if conflict:
            if (cos is not None and cos >= _REVIEW_FLOOR_COS) or ns >= 0.85 or alias_hit:
                return "review", "conflict-guard veto on strong signal"
            return None, ""
        # Auto-merge ONLY lexical identity — cosine is a candidate gate, not a merge decision.
        if tier == "MERGE":
            return "auto", "lexical identity (alias/acronym/normalized-name)"
        if alias_hit:
            return "auto", "graph alias hit"
        if tier == "VERIFY":
            return "verify", f"embedding candidate cos={round(cos, 3) if cos is not None else None}"
        if ns >= _REVIEW_FLOOR_NS:
            return "review", f"lexical near-match ns={ns:.2f} (no embedding)"
        return None, ""

    def _cluster(self, candidates: list[MergeCandidate], by_id: dict[str, dict]) -> list[list[str]]:
        """Union-find over auto-band pairs with a size brake, then an embedding-cohesion guard.

        The size cap brakes runaway cascades; the cohesion guard (size > 2 only) splits a component
        whose members are not all within ``cohesion_threshold`` of a medoid — stopping A~B~C
        transitive over-merge where the verifier confirmed A~B and B~C but A≁C (the clustering
        analogue of the 5966 all-pairs false-merge). Pairs (size 2) carry no chaining risk and pass
        through unchanged."""
        cap = self._config.consolidation.cluster_cap
        cohesion = self._config.consolidation.cohesion_threshold
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        size: dict[str, int] = {}
        for c in candidates:
            ra, rb = find(c.a_id), find(c.b_id)
            if ra == rb:
                continue
            sa, sb = size.get(ra, 1), size.get(rb, 1)
            if sa + sb > cap:
                logger.info("cluster cap: skipping union %s + %s", c.a_id, c.b_id)
                continue
            parent[rb] = ra
            size[ra] = sa + sb
        clusters: dict[str, list[str]] = {}
        for node in parent:
            clusters.setdefault(find(node), []).append(node)

        out: list[list[str]] = []
        for members in clusters.values():
            if len(members) <= 2:
                if len(members) > 1:
                    out.append(members)
                continue
            out.extend(self._cohesion_split(members, by_id, cohesion))
        return out

    @staticmethod
    def _cohesion_split(members: list[str], by_id: dict[str, dict], cohesion: float) -> list[list[str]]:
        """Split a >2 cluster into sub-clusters cohesive around a medoid (cos ≥ cohesion).

        Conservative on missing data: if any member lacks an embedding the cluster cannot be safely
        refined, so it is left intact (auto-band membership is already lexical/verifier-confirmed;
        the fresh graph embeds every node, so this fallback is effectively unreachable)."""
        import numpy as np

        embs = [by_id.get(m, {}).get("embedding") for m in members]
        if any(e is None for e in embs):
            logger.info("cohesion guard: cluster has unembedded member(s); kept intact (%d)", len(members))
            return [members]
        mat = np.asarray(embs, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms

        out: list[list[str]] = []
        remaining = set(range(len(members)))
        while remaining:
            rem = sorted(remaining)
            sub = mat[rem]
            sims = sub @ sub.T
            medoid = rem[int(np.argmax(sims.sum(axis=1)))]
            keep = [j for j in rem if float(mat[medoid] @ mat[j]) >= cohesion]
            out.append([members[j] for j in keep])
            remaining -= set(keep)
        return [c for c in out if len(c) > 1]

    @staticmethod
    def _pick_primary(cluster: list[str], by_id: dict[str, dict]) -> str:
        """Highest degree → highest reinforcement → oldest."""
        def keyf(cid: str):
            c = by_id[cid]
            return (-c.get("degree", 0), -c.get("reinforcement", 0), c.get("created_at") or "9999")
        return sorted(cluster, key=keyf)[0]

    async def run(
        self,
        dry_run: bool = True,
        auto_merge: bool = False,
        queue_path: str | Path | None = None,
    ) -> ConsolidationReport:
        """Scan, classify, optionally merge the auto band, queue the rest."""
        report = ConsolidationReport(dry_run=dry_run)
        concepts = await self._fetch_concepts()
        report.concepts_scanned = len(concepts)
        by_id = {c["id"]: c for c in concepts}

        candidates, report.pairs_considered = self._classify_pairs(concepts)
        report.auto = [c for c in candidates if c.band == "auto"]
        report.verify = [c for c in candidates if c.band == "verify"]
        report.review = [c for c in candidates if c.band == "review"]

        # Optional verifier pass drains the verify band into auto/review.
        if self._verifier is not None and report.verify:
            still_verify: list[MergeCandidate] = []
            for c in report.verify:
                result = await self._verifier.verify(
                    c.a_name, by_id[c.a_id]["description"], {
                        "concept_id": c.b_id,
                        "canonical_name": c.b_name,
                        "aliases": by_id[c.b_id].get("aliases", []),
                        "description": by_id[c.b_id]["description"],
                    },
                )
                if result.verdict == "SAME":
                    c.band, c.reason = "auto", "verifier SAME"
                    report.auto.append(c)
                elif result.verdict == "DIFFERENT":
                    continue  # dropped
                else:
                    c.band = "review"
                    report.review.append(c)
            report.verify = still_verify

        # Queue review band (and, when auto-merge is off, the auto band too).
        # Dry runs report only — they must not append to the queue, or
        # repeated dry-runs would pile up duplicate entries.
        to_queue = report.review + ([] if auto_merge else report.auto)
        if queue_path and to_queue and not dry_run:
            qp = Path(queue_path).expanduser()
            qp.parent.mkdir(parents=True, exist_ok=True)
            with open(qp, "a") as f:
                for c in to_queue:
                    f.write(json.dumps({
                        "ts": datetime.now().isoformat(), "a_id": c.a_id, "b_id": c.b_id,
                        "a_name": c.a_name, "b_name": c.b_name, "cos": c.cos,
                        "name_sim": c.name_sim, "band": c.band, "reason": c.reason,
                        "status": "pending",
                    }, ensure_ascii=False) + "\n")

        if dry_run or not auto_merge:
            return report

        # Apply: merge auto-band clusters into their primaries.
        for cluster in self._cluster(report.auto, by_id):
            primary = self._pick_primary(cluster, by_id)
            for dup in cluster:
                if dup == primary:
                    continue
                result = await self._graph.merge_concepts(primary, dup)
                if "error" in result:
                    logger.warning("merge failed %s←%s: %s", primary, dup, result["error"])
                    continue
                if self._journal is not None:
                    self._journal.log("consolidation_merge", **result)
                report.merged.append({"primary": primary, "duplicate": dup})
        return report
