"""Neo4j graph store — CRUD, merge, and query operations."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession, AsyncTransaction

from openclaw_brain.config import Neo4jConfig
from openclaw_brain.knowledge.graph.schema import (
    ConceptNode,
    EdgeProperties,
    GraphDelta,
    NodeLabel,
    NodeProposal,
    EdgeProposal,
    EdgeReinforcement,
    NodeUpdate,
    InsightProposal,
    RelType,
)

logger = logging.getLogger(__name__)


class GraphStore:
    """Async Neo4j graph store for knowledge and memory nodes."""

    def __init__(self, config: Neo4jConfig, *, egress: str | None = None):
        self._config = config
        self._egress = egress
        self._driver: AsyncDriver | None = None
        # Vector-index query failures (find_match_candidates / find_similar_concepts) degrade
        # matching to text-only candidates silently by design — the index may legitimately not
        # exist yet. But a REAL error (embedding-dimension mismatch after a partial migration, a
        # transient Neo4j fault) looks identical without this. Logged at WARNING once per store
        # instance (a multi-thousand-chunk ingest calls these per chunk; logging every time
        # would spam) — the counter increments on every occurrence so a caller/test can tell an
        # ongoing degradation from one-off flakiness.
        self._vector_index_error_count: int = 0
        self._vector_index_warned: bool = False

    async def connect(self) -> None:
        from openclaw_brain.egress import check_neo4j_uri
        from openclaw_brain.config import resolve_neo4j_password

        check_neo4j_uri(self._config.uri, policy=self._egress)
        self._driver = AsyncGraphDatabase.driver(
            self._config.uri,
            auth=(self._config.user, resolve_neo4j_password(self._config.password)),
        )
        await self._driver.verify_connectivity()

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()

    async def _session(self) -> AsyncSession:
        assert self._driver, "Call connect() first"
        return self._driver.session(database=self._config.database)

    # ── Schema ──

    async def apply_schema(self, schema_path: str | Path) -> None:
        """Apply Cypher schema constraints and indexes."""
        schema_path = Path(schema_path)
        cypher = schema_path.read_text()

        # Strip comment lines BEFORE splitting on ';' — a semicolon INSIDE a comment
        # (e.g. the 2026-07-12 "NO uniqueness constraint; latent until..." note) would
        # otherwise split mid-comment, leaving a non-'//' fragment that reaches Neo4j as a
        # syntax error and aborts every statement after it (broke apply-schema 07-12→07-28).
        lines = [
            line for line in cypher.split("\n")
            if not line.strip().startswith("//")
        ]
        async with await self._session() as session:
            for raw_statement in "\n".join(lines).split(";"):
                statement = raw_statement.strip()
                if statement:
                    await session.run(statement)

    # ── Node CRUD ──

    async def merge_node(
        self,
        label: NodeLabel,
        id_field: str,
        id_value: str,
        properties: dict[str, Any],
    ) -> str:
        """Merge (upsert) a node by its ID field. Returns the node ID.

        Kept in sync BY HAND with ``_merge_node_tx`` (session/tx pair rule — see the
        package README). Identity boundary: full props ON CREATE only; ON MATCH the
        ``_PROTECTED_ON_MATCH`` fields are excluded (no rename / source_id theft /
        un-retract / history rewrite via generic merges — the #20 incident class).
        """
        # Filter out None values and embedding (stored separately)
        props = {
            k: _sanitize_neo4j_value(v)
            for k, v in properties.items()
            if v is not None and k != "embedding"
        }
        # Keep raw float vectors out of normal props — handled via vector index
        for k, v in list(props.items()):
            if isinstance(v, list) and v and isinstance(v[0], float):
                continue

        create_parts, match_parts, sanitized_props = _set_assignments_split(
            props, "n", exclude=id_field, reserved={id_field})
        create_clause = (", ".join(create_parts) + ", ") if create_parts else ""
        match_clause = (", ".join(match_parts) + ", ") if match_parts else ""
        query = f"""
        MERGE (n:{label.value} {{{id_field}: ${id_field}}})
        ON CREATE SET {create_clause}n._created_at = datetime()
        ON MATCH SET {match_clause}n._updated_at = datetime()
        RETURN n.{id_field} AS id
        """

        async with await self._session() as session:
            result = await session.run(query, {id_field: id_value, **sanitized_props})
            record = await result.single()
            return record["id"] if record else id_value

    async def get_node(self, label: NodeLabel, id_field: str, id_value: str) -> dict | None:
        """Fetch a single node by ID."""
        query = f"MATCH (n:{label.value} {{{id_field}: $id_value}}) RETURN n"
        async with await self._session() as session:
            result = await session.run(query, {"id_value": id_value})
            record = await result.single()
            return dict(record["n"]) if record else None

    def _note_vector_index_failure(self, where: str, exc: Exception) -> None:
        """Record a vector-index query failure (shared by find_match_candidates and
        find_similar_concepts). Logs at WARNING once per store instance — a multi-thousand-chunk
        ingest calls these per chunk, so logging every occurrence would spam — while the counter
        increments every time so a caller/test can tell a real, ongoing degradation (matching
        silently falling back to text-only candidates) from one-off flakiness.
        """
        self._vector_index_error_count += 1
        if not self._vector_index_warned:
            self._vector_index_warned = True
            logger.warning(
                "%s: vector index query failed — degrading to text-only candidates for this "
                "and any subsequent call this session (further occurrences are counted in "
                "_vector_index_error_count but not re-logged): %s",
                where, exc,
            )

    async def find_similar_concepts(
        self,
        name: str,
        embedding: list[float] | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """Find concepts by name similarity + optional vector similarity (hybrid).

        Text search: CONTAINS on canonical_name + alias, expanded abbreviations.
        Vector search: cosine similarity via Neo4j vector index (if index exists and
        embedding provided). Falls back to text-only if index unavailable.

        Text results take priority; vector results fill remaining slots.
        """
        from openclaw_brain.knowledge.reasoning.matcher import _normalize_name

        _BASE_QUERY = """
        MATCH (c)
        WHERE (c:Concept OR c:CircuitTopology OR c:Parameter OR c:Equation OR c:Principle)
          AND c.canonical_name IS NOT NULL AND c.canonical_name <> ''
          AND NOT coalesce(c.retracted, false)
          AND (toLower(c.canonical_name) CONTAINS toLower($name)
           OR any(alias IN coalesce(c.aliases, []) WHERE toLower(alias) CONTAINS toLower($name)))
        RETURN c
        ORDER BY
          CASE WHEN toLower(c.canonical_name) = toLower($name) THEN 0 ELSE 1 END,
          c.canonical_name
        LIMIT $limit
        """

        def _node_uid(node: dict) -> str:
            return (
                node.get("concept_id") or node.get("topology_id")
                or node.get("parameter_id") or node.get("equation_id")
                or node.get("principle_id") or node.get("canonical_name", "")
            )

        async def _text_query(session, term: str) -> list[dict]:
            result = await session.run(_BASE_QUERY, {"name": term, "limit": limit})
            return [dict(record["c"]) async for record in result]

        async with await self._session() as session:
            # 1. Text search: name CONTAINS (exact name matching, fast)
            results = await _text_query(session, name)
            seen_ids = {_node_uid(r) for r in results}

            # 2. Abbreviation-expanded fallback
            if len(results) < limit:
                expanded = _normalize_name(name)
                if expanded != name.lower() and expanded != name:
                    for node in await _text_query(session, expanded):
                        uid = _node_uid(node)
                        if uid not in seen_ids:
                            seen_ids.add(uid)
                            results.append(node)
                            if len(results) >= limit:
                                break

            # 3. Vector search supplement (if embedding provided and slots remain)
            if embedding and len(results) < limit:
                try:
                    vec_query = """
                    CALL db.index.vector.queryNodes('concept_embedding', $k, $embedding)
                    YIELD node AS c, score
                    WHERE NOT coalesce(c.retracted, false)
                    RETURN c
                    """
                    vec_result = await session.run(
                        vec_query, {"k": limit * 2, "embedding": embedding}
                    )
                    for record in await vec_result.data():
                        node = dict(record["c"])
                        uid = _node_uid(node)
                        if uid not in seen_ids:
                            seen_ids.add(uid)
                            results.append(node)
                            if len(results) >= limit:
                                break
                except Exception as e:
                    self._note_vector_index_failure("find_similar_concepts", e)

        return results[:limit]

    async def find_match_candidates(
        self,
        name: str,
        embedding: list[float] | None = None,
        limit: int = 8,
    ) -> list[dict]:
        """Find entity-resolution candidates WITH similarity scores.

        Unlike find_similar_concepts (which discards the vector score and is
        used for retrieval), this returns candidates for the matcher's tier
        decision: each item is {"node": {...}, "cos": float|None, "text_hit": bool}.
        Vector hits come from the Concept vector index with true cosine
        (converted from Neo4j's normalized score); text/alias hits across the
        five knowledge labels carry cos=None and rely on name-tier logic.
        """
        from openclaw_brain.knowledge.reasoning.matcher import _normalize_name

        _TEXT_QUERY = """
        MATCH (c)
        WHERE (c:Concept OR c:CircuitTopology OR c:Parameter OR c:Equation OR c:Principle)
          AND c.canonical_name IS NOT NULL AND c.canonical_name <> ''
          AND NOT coalesce(c.retracted, false)
          AND (toLower(c.canonical_name) CONTAINS toLower($name)
           OR any(alias IN coalesce(c.aliases, []) WHERE toLower(alias) CONTAINS toLower($name)))
        RETURN c
        ORDER BY
          CASE WHEN toLower(c.canonical_name) = toLower($name) THEN 0 ELSE 1 END,
          c.canonical_name
        LIMIT $limit
        """

        def _node_uid(node: dict) -> str:
            return (
                node.get("concept_id") or node.get("topology_id")
                or node.get("parameter_id") or node.get("equation_id")
                or node.get("principle_id") or node.get("canonical_name", "")
            )

        candidates: list[dict] = []
        by_uid: dict[str, dict] = {}

        async with await self._session() as session:
            # Vector candidates first — they carry the decision signal.
            if embedding:
                try:
                    vec_query = """
                    CALL db.index.vector.queryNodes('concept_embedding', $k, $embedding)
                    YIELD node AS c, score
                    WHERE NOT coalesce(c.retracted, false)
                    RETURN c, score
                    """
                    vec_result = await session.run(
                        vec_query, {"k": limit, "embedding": embedding}
                    )
                    for record in await vec_result.data():
                        node = dict(record["c"])
                        entry = {
                            "node": node,
                            "cos": _vector_score_to_cosine(record["score"]),
                            "text_hit": False,
                        }
                        by_uid[_node_uid(node)] = entry
                        candidates.append(entry)
                except Exception as e:
                    self._note_vector_index_failure("find_match_candidates", e)

            # Text/alias candidates (exact name + abbreviation-expanded).
            terms = [name]
            expanded = _normalize_name(name)
            if expanded not in (name, name.lower()):
                terms.append(expanded)
            for term in terms:
                result = await session.run(_TEXT_QUERY, {"name": term, "limit": limit})
                for record in [r async for r in result]:
                    node = dict(record["c"])
                    uid = _node_uid(node)
                    if uid in by_uid:
                        by_uid[uid]["text_hit"] = True
                    else:
                        entry = {"node": node, "cos": None, "text_hit": True}
                        by_uid[uid] = entry
                        candidates.append(entry)

        # Highest cosine first; text-only hits after.
        candidates.sort(key=lambda e: e["cos"] if e["cos"] is not None else -2.0, reverse=True)
        return candidates[: limit * 2]

    async def store_embedding(
        self,
        label: NodeLabel,
        id_field: str,
        id_value: str,
        embedding: list[float],
    ) -> None:
        """Store an embedding vector on a node for vector index search.

        Uses SET so the value is indexable by the Neo4j vector index.
        No-op if the node doesn't exist or embedding is empty.
        """
        if not embedding:
            return
        query = f"""
        MATCH (n:{label.value} {{{id_field}: $id_value}})
        SET n.embedding = $embedding
        """
        async with await self._session() as session:
            await session.run(query, {"id_value": id_value, "embedding": embedding})

    async def create_vector_index(self, dimensions: int = 384) -> None:
        """Create Neo4j vector index for concept embeddings (idempotent).

        Safe to call multiple times — uses IF NOT EXISTS.
        Run once after schema setup or on first embedding storage.
        """
        query = f"""
        CREATE VECTOR INDEX concept_embedding IF NOT EXISTS
        FOR (c:Concept) ON (c.embedding)
        OPTIONS {{indexConfig: {{`vector.dimensions`: {dimensions}, `vector.similarity_function`: 'cosine'}}}}
        """
        async with await self._session() as session:
            await session.run(query)

    async def recreate_vector_indexes(self, dimensions: int) -> list[str]:
        """Drop and recreate vector indexes for ALL five embeddable labels.

        Required when the embedding model (and thus dimension) changes —
        Neo4j vector index config is immutable.  Also closes the historical
        gap where only Concept had an index while embeddings were written
        to five labels.  The Concept index keeps the name 'concept_embedding'
        (queried by find_similar_concepts / find_match_candidates).
        """
        created: list[str] = []
        async with await self._session() as session:
            for index_name, label in _VECTOR_INDEXES:
                await session.run(f"DROP INDEX {index_name} IF EXISTS")
                await session.run(f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (n:{label}) ON (n.embedding)
                OPTIONS {{indexConfig: {{`vector.dimensions`: {dimensions},
                                         `vector.similarity_function`: 'cosine'}}}}
                """)
                created.append(index_name)
        return created

    async def backfill_embeddings(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 128,
        on_progress: Any = None,
        force: bool = False,
    ) -> dict[str, int]:
        """Generate and store embeddings for all knowledge nodes that lack one.

        Covers Concept, Equation, Principle, CircuitTopology, and Parameter nodes.
        Processes in batches for efficiency.  Returns counts per label.

        Args:
            model_name: sentence-transformers model name.
            batch_size: Number of nodes to embed per batch.
            on_progress: Optional callable(label, done, total) for progress reporting.
            force: Null ALL existing embeddings first and re-embed everything —
                required on an embedding-model change (old vectors are not
                comparable to new ones, in dimension or distribution).
        """
        from openclaw_brain.knowledge.embedding import encode_batch

        _LABELS = [
            ("Concept",         "concept_id"),
            ("Equation",        "equation_id"),
            ("Principle",       "principle_id"),
            ("CircuitTopology", "topology_id"),
            ("Parameter",       "parameter_id"),
        ]

        if force:
            async with await self._session() as session:
                for label_str, _ in _LABELS:
                    await session.run(
                        f"MATCH (n:{label_str}) WHERE n.embedding IS NOT NULL "
                        f"SET n.embedding = null, n.embedding_model = null"
                    )

        totals: dict[str, int] = {}

        for label_str, id_field in _LABELS:
            # Fetch nodes with a canonical_name but no embedding
            fetch_query = f"""
            MATCH (n:{label_str})
            WHERE n.canonical_name IS NOT NULL
              AND n.canonical_name <> ''
              AND n.embedding IS NULL
            RETURN n.{id_field} AS node_id, n.canonical_name AS name,
                   coalesce(n.description, '') AS description
            """
            async with await self._session() as session:
                result = await session.run(fetch_query)
                rows = [dict(r) async for r in result]

            if not rows:
                totals[label_str] = 0
                continue

            label_enum = NodeLabel(label_str)
            done = 0

            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                texts = [
                    f"{r['name']}. {r['description']}" if r["description"]
                    else r["name"]
                    for r in batch
                ]
                vectors = encode_batch(texts, model_name)

                for row, vector in zip(batch, vectors):
                    if not vector:
                        continue
                    try:
                        await self.store_embedding(
                            label_enum, id_field, row["node_id"], vector
                        )
                        done += 1
                    except Exception:
                        pass  # Skip individual failures silently

                if on_progress:
                    on_progress(label_str, done, len(rows))

            totals[label_str] = done

        # Stamp which model produced the vectors — guards against silent
        # mixed-model graphs after a future model change.
        async with await self._session() as session:
            for label_str, _ in _LABELS:
                await session.run(
                    f"MATCH (n:{label_str}) WHERE n.embedding IS NOT NULL "
                    f"SET n.embedding_model = $model",
                    {"model": model_name},
                )

        return totals

    # ── Edge CRUD ──

    async def merge_edge(
        self,
        source_label: NodeLabel,
        source_id_field: str,
        source_id_value: str,
        target_label: NodeLabel,
        target_id_field: str,
        target_id_value: str,
        rel_type: RelType,
        properties: dict[str, Any],
    ) -> None:
        """Merge (upsert) a relationship between two nodes.

        There are TWO edge write paths in this class — this standalone one and `_merge_edge_tx`
        (used inside apply_delta/write_batch transactions). Both are gated on tier-3 evidence
        endpoints; gating only one would leave the counterfeits arriving through the other.
        """
        violation = _evidence_edge_violation(rel_type, source_label, target_label)
        if violation is not None:
            logger.warning(
                "merge_edge: refusing counterfeit evidence edge %s -[:%s]-> %s — %s "
                "(source=%r target=%r; see store._EVIDENCE_EDGE_ENDPOINTS)",
                source_label.value, rel_type.value, target_label.value, violation,
                source_id_value, target_id_value,
            )
            return
        props = {
            k: _sanitize_neo4j_value(v)
            for k, v in properties.items()
            if v is not None
        }

        set_parts, sanitized_props = _set_assignments(props, "r",
                                                      reserved={"source_id", "target_id"})
        set_clause = ", ".join(set_parts)
        # An edge with no properties (e.g. projection's HAS_CLAIM/REALIZES/GROUNDS) yields an empty
        # set_clause — prepending it with a comma would produce `SET , r._created_at = ...` (a Cypher
        # syntax error). Only prefix the timestamp assignment when there are properties to set.
        prefix = f"{set_clause}, " if set_clause else ""
        query = f"""
        MATCH (a:{source_label.value} {{{source_id_field}: $source_id}})
        MATCH (b:{target_label.value} {{{target_id_field}: $target_id}})
        MERGE (a)-[r:{rel_type.value}]->(b)
        ON CREATE SET {prefix}r._created_at = datetime()
        ON MATCH SET {prefix}r._updated_at = datetime()
        """
        async with await self._session() as session:
            await session.run(
                query,
                {"source_id": source_id_value, "target_id": target_id_value, **sanitized_props},
            )

    async def reinforce_edge(
        self,
        source_label: NodeLabel,
        source_id_field: str,
        source_id_value: str,
        target_label: NodeLabel,
        target_id_field: str,
        target_id_value: str,
        rel_type: RelType,
        new_evidence: str,
        confirming_chunks: list[str],
    ) -> None:
        """Increment reinforcement count and add evidence to an existing edge."""
        query = f"""
        MATCH (a:{source_label.value} {{{source_id_field}: $source_id}})
              -[r:{rel_type.value}]->
              (b:{target_label.value} {{{target_id_field}: $target_id}})
        SET r.reinforcement_count = coalesce(r.reinforcement_count, 1) + 1,
            r.last_reinforced = datetime(),
            r.evidence_sources = coalesce(r.evidence_sources, []) + $new_chunks
        """
        async with await self._session() as session:
            await session.run(
                query,
                {
                    "source_id": source_id_value,
                    "target_id": target_id_value,
                    "new_chunks": confirming_chunks,
                },
            )

    # ── Neighborhood Queries ──

    async def get_neighborhood(
        self,
        label: NodeLabel,
        id_field: str,
        id_value: str,
        hops: int = 2,
        limit: int = 50,
    ) -> list[dict]:
        """Get N-hop neighborhood around a node, one row per neighbor, with the rationale of
        the edge that actually reaches it.

        Each neighbor appears ONCE, described by its own incident edge — the LAST relationship
        on the shortest path that reached it — plus how far away it is and which way that edge
        points. `edge_direction` is "->" when the edge runs from the preceding node into the
        neighbor, "<-" when it runs the other way; a caller that cares about semantic direction
        (DEPENDS_ON, SUB_BLOCK) must read it, because the traversal itself is undirected.

        Defect this replaces (measured 2026-07-26, audit D-5): the previous query did
        `UNWIND relationships(path)`, emitting one row per relationship ON the path rather than
        per neighbor. A 2-hop neighbor was therefore reported once per hop, and — worse — paired
        with the FIRST hop's relationship, which is not incident to it at all. Live measurement
        on `Miller_Approximation`: 1.5-2.0x duplication, and **12 of 30 sampled rows carried a
        rel_type/rationale belonging to some other edge**. Because `LIMIT` is applied after the
        explosion and the production caller uses `max_graph_neighbors = 6`, that limit bought
        only 4 distinct neighbors, a third of the budget spent on duplicates.
        """
        query = f"""
        MATCH path = (start:{label.value} {{{id_field}: $id_value}})-[*1..{hops}]-(neighbor)
        WHERE neighbor <> start
        WITH neighbor,
             labels(neighbor) AS lbls,
             last(relationships(path)) AS r,
             length(path) AS dist
        ORDER BY dist ASC
        WITH neighbor, lbls, collect({{r: r, dist: dist}})[0] AS best
        RETURN
            neighbor {{.*, _labels: lbls}} AS node,
            type(best.r) AS rel_type,
            best.r.rationale AS rationale,
            best.r.confidence AS confidence,
            best.r.reinforcement_count AS reinforcement_count,
            best.dist AS hop_distance,
            CASE WHEN endNode(best.r) = neighbor THEN '->' ELSE '<-' END AS edge_direction
        LIMIT $limit
        """
        async with await self._session() as session:
            result = await session.run(query, {"id_value": id_value, "limit": limit})
            return [dict(record) async for record in result]

    # ── Graph Delta Application ──

    async def _merge_node_tx(
        self,
        tx: AsyncTransaction,
        label: NodeLabel,
        id_field: str,
        id_value: str,
        properties: dict[str, Any],
    ) -> None:
        """Merge a node within an existing transaction.

        Identity boundary (see ``_PROTECTED_ON_MATCH``): the full property set is
        written only ON CREATE; ON MATCH — the node already exists — identity and
        provenance fields are excluded, so a colliding id from another source can
        gap-in new descriptive properties but can never rename the node, steal its
        source_id, resurrect it from retraction, or rewrite its history.
        """
        properties = _coerce_scalar_text_fields(dict(properties))
        props = {
            k: _sanitize_neo4j_value(v)
            for k, v in properties.items()
            if v is not None and k != "embedding"
        }
        create_parts, match_parts, sanitized_props = _set_assignments_split(
            props, "n", exclude=id_field, reserved={id_field})
        create_clause = (", ".join(create_parts) + ", ") if create_parts else ""
        match_clause = (", ".join(match_parts) + ", ") if match_parts else ""
        query = f"""
        MERGE (n:{label.value} {{{id_field}: ${id_field}}})
        ON CREATE SET {create_clause}n._created_at = datetime()
        ON MATCH SET {match_clause}n._updated_at = datetime()
        """
        await tx.run(query, {id_field: id_value, **sanitized_props})

    async def _merge_edge_tx(
        self,
        tx: AsyncTransaction,
        source_label: NodeLabel,
        source_id_field: str,
        source_id_value: str,
        target_label: NodeLabel,
        target_id_field: str,
        target_id_value: str,
        rel_type: RelType,
        properties: dict[str, Any],
    ) -> None:
        """Merge an edge within an existing transaction.

        Tier-3 evidence edges are endpoint-checked here (see `_EVIDENCE_EDGE_ENDPOINTS`): a
        mis-typed one is SKIPPED with a loud warning rather than written, mirroring the node
        path's `_LLM_PROPOSABLE_LABELS` gate. This is the single door every writer passes
        through — apply_delta and write_batch both land here.
        """
        violation = _evidence_edge_violation(rel_type, source_label, target_label)
        if violation is not None:
            logger.warning(
                "merge_edge: refusing counterfeit evidence edge %s -[:%s]-> %s — %s "
                "(source=%r target=%r; see store._EVIDENCE_EDGE_ENDPOINTS)",
                source_label.value, rel_type.value, target_label.value, violation,
                source_id_value, target_id_value,
            )
            return
        props = {
            k: _sanitize_neo4j_value(v)
            for k, v in properties.items()
            if v is not None
        }
        set_parts, sanitized_props = _set_assignments(props, "r",
                                                      reserved={"source_id", "target_id"})
        set_clause = ", ".join(set_parts)
        # An edge with no properties (e.g. projection's HAS_CLAIM/REALIZES/GROUNDS) yields an empty
        # set_clause — prepending it with a comma would produce `SET , r._created_at = ...` (a Cypher
        # syntax error). Only prefix the timestamp assignment when there are properties to set.
        prefix = f"{set_clause}, " if set_clause else ""
        query = f"""
        MATCH (a:{source_label.value} {{{source_id_field}: $source_id}})
        MATCH (b:{target_label.value} {{{target_id_field}: $target_id}})
        MERGE (a)-[r:{rel_type.value}]->(b)
        ON CREATE SET {prefix}r._created_at = datetime()
        ON MATCH SET {prefix}r._updated_at = datetime()
        """
        await tx.run(
            query,
            {"source_id": source_id_value, "target_id": target_id_value, **sanitized_props},
        )

    async def _reinforce_edge_tx(
        self,
        tx: AsyncTransaction,
        source_label: NodeLabel,
        source_id_field: str,
        source_id_value: str,
        target_label: NodeLabel,
        target_id_field: str,
        target_id_value: str,
        rel_type: RelType,
        confirming_chunks: list[str],
    ) -> None:
        """Reinforce an edge within an existing transaction."""
        query = f"""
        MATCH (a:{source_label.value} {{{source_id_field}: $source_id}})
              -[r:{rel_type.value}]->
              (b:{target_label.value} {{{target_id_field}: $target_id}})
        SET r.reinforcement_count = coalesce(r.reinforcement_count, 1) + 1,
            r.last_reinforced = datetime(),
            r.evidence_sources = coalesce(r.evidence_sources, []) + $new_chunks
        """
        await tx.run(
            query,
            {
                "source_id": source_id_value,
                "target_id": target_id_value,
                "new_chunks": confirming_chunks,
            },
        )

    async def _update_node_tx(
        self,
        tx: AsyncTransaction,
        label: NodeLabel,
        id_field: str,
        id_value: str,
        properties: dict[str, Any],
    ) -> None:
        """Update an existing node's properties within a transaction (MATCH-only, never creates).

        Identity boundary: ``_PROTECTED_ON_MATCH`` keys are dropped with a WARNING —
        unlike the merge path (where identical identity values ride along on every
        idempotent re-merge), an explicit update REQUEST naming an identity field is
        always suspect: it is how an LLM-authored ``NodeUpdate.updates`` dict would
        rename a node or steal its provenance (#20 family; ``updates`` is an
        unconstrained ``dict[str, Any]`` by schema, constrained here at the choke
        point instead so every write path inherits the rule).
        """
        blocked = sorted(k for k in properties if k in _PROTECTED_ON_MATCH)
        if blocked:
            logger.warning(
                "update on %s(%s=%s) tried to set identity field(s) %s — dropped "
                "(identity changes go through merge_concepts/retract_node, never "
                "generic updates)",
                label.value, id_field, id_value, blocked,
            )
        properties = _coerce_scalar_text_fields(dict(properties))
        props = {
            k: _sanitize_neo4j_value(v)
            for k, v in properties.items()
            if v is not None and k != "embedding" and k not in _PROTECTED_ON_MATCH
        }
        if not props:
            return
        set_parts, sanitized_props = _set_assignments(props, "n", reserved={id_field})
        if not set_parts:
            return
        set_clause = ", ".join(set_parts)
        query = f"""
        MATCH (n:{label.value} {{{id_field}: ${id_field}}})
        SET {set_clause}, n._updated_at = datetime()
        """
        await tx.run(query, {id_field: id_value, **sanitized_props})

    async def write_batch(
        self,
        nodes: list[dict[str, Any]] | None = None,
        updates: list[dict[str, Any]] | None = None,
        edges: list[dict[str, Any]] | None = None,
    ) -> None:
        """Apply multiple node merges, node updates, and edge merges atomically.

        All operations run inside a single Neo4j transaction and roll back together
        on any error — use this instead of issuing separate ``merge_node``/``merge_edge``
        calls when a logical write spans several nodes/edges (e.g. design-reasoning records).

        Args:
            nodes:   create-or-update via MERGE. Each: ``{label, id_field, id_value, properties}``.
            updates: update-only via MATCH (no-op if the node is absent — avoids phantom nodes).
                     Same shape as ``nodes``.
            edges:   Each: ``{source_label, source_id_field, source_id_value, target_label,
                     target_id_field, target_id_value, rel_type, properties}``.

        Nodes and updates are applied before edges so edge endpoints already exist.
        """
        if not nodes and not updates and not edges:
            return
        async with await self._session() as session:
            tx = await session.begin_transaction()
            try:
                for n in (nodes or []):
                    await self._merge_node_tx(
                        tx, n["label"], n["id_field"], n["id_value"], n["properties"]
                    )
                for u in (updates or []):
                    await self._update_node_tx(
                        tx, u["label"], u["id_field"], u["id_value"], u["properties"]
                    )
                for e in (edges or []):
                    await self._merge_edge_tx(
                        tx,
                        e["source_label"], e["source_id_field"], e["source_id_value"],
                        e["target_label"], e["target_id_field"], e["target_id_value"],
                        e["rel_type"], e["properties"],
                    )
                await tx.commit()
            except Exception:
                await tx.rollback()
                raise

    async def apply_delta(self, delta: GraphDelta) -> dict[str, int]:
        """Apply a GraphDelta atomically within a single Neo4j transaction."""
        counts = {
            "new_nodes": 0,
            "updated_nodes": 0,
            "new_edges": 0,
            "reinforced_edges": 0,
            "insights": 0,
            "rejected_labels": 0,
        }

        async with await self._session() as session:
            tx = await session.begin_transaction()
            try:
                # New nodes
                for proposal in delta.new_nodes:
                    # F8(a) — LLM-proposable label allowlist (see
                    # _LLM_PROPOSABLE_LABELS below for the verified incident and
                    # exclusion rationale). A reasoning-pass proposal naming a
                    # personal-memory/curated-write-path/projection-only label is
                    # skipped, not committed — never crashes the chunk (W-D style).
                    if proposal.label not in _LLM_PROPOSABLE_LABELS:
                        logger.warning(
                            "apply_delta: new_nodes proposal label=%s proposed_id=%s "
                            "canonical_name=%r is outside _LLM_PROPOSABLE_LABELS — "
                            "skipped (F8(a); see store._LLM_PROPOSABLE_LABELS)",
                            proposal.label.value, proposal.proposed_id,
                            proposal.canonical_name,
                        )
                        counts["rejected_labels"] += 1
                        continue
                    id_field = self._id_field_for_label(proposal.label)
                    node_props = {id_field: proposal.proposed_id}
                    if proposal.canonical_name:
                        node_props["canonical_name"] = proposal.canonical_name
                    if proposal.description:
                        node_props["description"] = proposal.description
                    if proposal.domain and proposal.domain != "general":
                        node_props["domain"] = proposal.domain
                    if proposal.confidence > 0:
                        node_props["confidence"] = proposal.confidence
                    if proposal.knowledge_layer >= 0:
                        node_props["knowledge_layer"] = proposal.knowledge_layer
                    if proposal.evidence_chunk_ids:
                        node_props["evidence_chunk_ids"] = proposal.evidence_chunk_ids
                    extra_props, dropped_keys = _filter_proposal_properties(
                        proposal.properties
                    )
                    if dropped_keys:
                        logger.warning(
                            "apply_delta: dropped %d protected/structural key(s) from "
                            "LLM-proposed properties for %s %r: %s (migration step 0 "
                            "write-channel closure, ADR-046 D6)",
                            len(dropped_keys), proposal.label.value,
                            proposal.proposed_id, sorted(dropped_keys),
                        )
                    node_props.update(extra_props)
                    if proposal.label == NodeLabel.EQUATION:
                        # LLMs occasionally emit LaTeX as a LIST of strings (multi-line
                        # equations — 2 live Razavi nodes shipped this way as LIST<STRING>
                        # arrays). canonical_latex/latex must be scalar strings for every
                        # downstream reader (normalization, matching, export) — join lists.
                        for k in ("canonical_latex", "latex"):
                            v = node_props.get(k)
                            if isinstance(v, list):
                                node_props[k] = " ; ".join(str(x) for x in v if x)
                    if proposal.label == NodeLabel.EQUATION and not node_props.get(
                        "canonical_latex"
                    ):
                        # The reasoner prompt (and schema.py's documented convention) has the
                        # LLM emit an equation's LaTeX under properties['latex'], but the
                        # EquationNode schema field / all downstream readers use
                        # 'canonical_latex'. Nothing else maps latex -> canonical_latex, so
                        # without this every Equation node commits with an empty
                        # canonical_latex (confirmed live: 614/640 nodes affected). Backfill
                        # from 'latex' when canonical_latex is absent/empty; keep 'latex' too
                        # for backward-compat. Equation-specific — no other label is touched.
                        latex = node_props.get("latex")
                        if latex:
                            node_props["canonical_latex"] = latex
                    await self._merge_node_tx(
                        tx, proposal.label, id_field, proposal.proposed_id, node_props
                    )
                    counts["new_nodes"] += 1

                # Updated nodes — MATCH-only: an "update" must target an existing
                # node, so a stale/wrong id is a no-op rather than a phantom create.
                for update in delta.updated_nodes:
                    # F8(a) — same allowlist as new_nodes: the LLM shouldn't touch a
                    # non-knowledge node's fields either.
                    if update.label not in _LLM_PROPOSABLE_LABELS:
                        logger.warning(
                            "apply_delta: updated_nodes update label=%s "
                            "existing_node_id=%s is outside _LLM_PROPOSABLE_LABELS — "
                            "skipped (F8(a); see store._LLM_PROPOSABLE_LABELS)",
                            update.label.value, update.existing_node_id,
                        )
                        counts["rejected_labels"] += 1
                        continue
                    id_field = self._id_field_for_label(update.label)
                    await self._update_node_tx(
                        tx, update.label, id_field, update.existing_node_id, update.updates
                    )
                    counts["updated_nodes"] += 1

                # New edges
                for edge in delta.new_edges:
                    source_label, source_id_field = self._resolve_ref(edge.source_ref, delta)
                    target_label, target_id_field = self._resolve_ref(edge.target_ref, delta)
                    props = EdgeProperties(
                        rationale=edge.rationale,
                        confidence=edge.confidence,
                        evidence_sources=edge.evidence_chunk_ids,
                        created_by="reasoning",
                    )
                    await self._merge_edge_tx(
                        tx,
                        source_label=source_label,
                        source_id_field=source_id_field,
                        source_id_value=edge.source_ref,
                        target_label=target_label,
                        target_id_field=target_id_field,
                        target_id_value=edge.target_ref,
                        rel_type=edge.relationship_type,
                        properties=props.model_dump(),
                    )
                    counts["new_edges"] += 1

                # Reinforced edges
                for reinf in delta.reinforced_edges:
                    source_label, source_id_field = self._resolve_ref(reinf.source_ref, delta)
                    target_label, target_id_field = self._resolve_ref(reinf.target_ref, delta)
                    await self._reinforce_edge_tx(
                        tx,
                        source_label=source_label,
                        source_id_field=source_id_field,
                        source_id_value=reinf.source_ref,
                        target_label=target_label,
                        target_id_field=target_id_field,
                        target_id_value=reinf.target_ref,
                        rel_type=reinf.relationship_type,
                        confirming_chunks=reinf.confirming_chunk_ids,
                    )
                    counts["reinforced_edges"] += 1

                # Insights
                for insight in delta.insights:
                    insight_id = f"insight_{hash(insight.statement) % 10**8:08d}"
                    await self._merge_node_tx(
                        tx,
                        NodeLabel.INSIGHT,
                        "insight_id",
                        insight_id,
                        {
                            "insight_id": insight_id,
                            "statement": insight.statement,
                            "confidence": insight.confidence,
                            "bridge_type": insight.bridge_type,
                            "source_id": insight.source_id,
                            "evidence_chunk_ids": insight.evidence_chunk_ids,
                        },
                    )
                    for cid in insight.related_concept_ids:
                        await self._merge_edge_tx(
                            tx,
                            source_label=NodeLabel.INSIGHT,
                            source_id_field="insight_id",
                            source_id_value=insight_id,
                            target_label=NodeLabel.CONCEPT,
                            target_id_field="concept_id",
                            target_id_value=cid,
                            rel_type=RelType.BRIDGES_TO,
                            properties={
                                "rationale": insight.statement,
                                "confidence": insight.confidence,
                            },
                        )
                    counts["insights"] += 1

                await tx.commit()

            except Exception:
                await tx.rollback()
                raise

        return counts

    # ── Node Identity Operations ──

    async def merge_concepts(
        self,
        primary_id: str,
        duplicate_id: str,
    ) -> dict[str, Any]:
        """Merge a duplicate Concept node into the primary, re-wiring all edges.

        Steps:
        1. Gap-fill: properties present on duplicate but absent on primary are copied.
        2. Outgoing edges from duplicate are re-created on primary (MERGE — idempotent).
        3. Incoming edges to duplicate are re-created pointing to primary (MERGE).
        4. Duplicate node is DETACH DELETE'd.

        Returns counts of rewired edges and a confirmation.
        """
        if primary_id == duplicate_id:
            return {"error": "primary_id and duplicate_id are the same"}

        # ── Step 1: Gap-fill properties ──
        primary_node = await self.get_node(NodeLabel.CONCEPT, "concept_id", primary_id)
        dup_node = await self.get_node(NodeLabel.CONCEPT, "concept_id", duplicate_id)

        if primary_node is None:
            return {"error": f"Primary concept '{primary_id}' not found"}
        if dup_node is None:
            return {"error": f"Duplicate concept '{duplicate_id}' not found"}

        _SKIP = {"concept_id", "_created_at", "_updated_at", "embedding", "retracted",
                 "retracted_at", "retracted_reason"}
        gap = {
            k: v for k, v in dup_node.items()
            if k not in primary_node and k not in _SKIP and v is not None
        }
        if gap:
            set_parts, gap_params = _set_assignments(gap, "n", param_prefix="gap_")
            if set_parts:
                async with await self._session() as session:
                    await session.run(
                        f"MATCH (n:Concept {{concept_id: $pid}}) SET {', '.join(set_parts)}",
                        {"pid": primary_id, **gap_params},
                    )

        # Copy embedding if primary lacks one
        if dup_node.get("embedding") and not primary_node.get("embedding"):
            await self.store_embedding(NodeLabel.CONCEPT, "concept_id", primary_id,
                                       dup_node["embedding"])

        # Alias union — gap-fill skips `aliases` when the primary already has
        # the key, but text search depends on aliases, so the duplicate's
        # aliases AND its canonical name must survive on the primary.
        alias_union = list(dict.fromkeys(
            (primary_node.get("aliases") or [])
            + (dup_node.get("aliases") or [])
            + ([dup_node["canonical_name"]] if dup_node.get("canonical_name") else [])
        ))
        primary_name = primary_node.get("canonical_name", "")
        alias_union = [a for a in alias_union if a and a != primary_name]
        if alias_union != (primary_node.get("aliases") or []):
            await self.add_aliases(primary_id, alias_union, replace=True)

        # ── Step 2: Fetch edges ──
        _NODE_ID_EXPR = (
            "coalesce(n.concept_id, n.equation_id, n.parameter_id, n.topology_id, "
            "n.principle_id, n.chunk_id, n.source_id, n.insight_id, n.memory_id, "
            "n.entity_id, n.hypothesis_id, n.decision_id, n.bench_id, n.assumption_id)"
        )
        out_edges = await self.run_read_query(
            f"""
            MATCH (dup:Concept {{concept_id: $dup_id}})-[r]->(target)
            WHERE NOT (target:Concept AND target.concept_id = $primary_id)
            RETURN type(r) AS rel_type,
                   labels(target)[0] AS node_label,
                   {_NODE_ID_EXPR.replace('n.', 'target.')} AS node_id,
                   {{rationale: r.rationale, confidence: r.confidence,
                     evidence_sources: r.evidence_sources,
                     reinforcement_count: r.reinforcement_count,
                     created_by: r.created_by}} AS props
            """,
            {"dup_id": duplicate_id, "primary_id": primary_id},
        )
        in_edges = await self.run_read_query(
            f"""
            MATCH (source)-[r]->(dup:Concept {{concept_id: $dup_id}})
            WHERE NOT (source:Concept AND source.concept_id = $primary_id)
            RETURN type(r) AS rel_type,
                   labels(source)[0] AS node_label,
                   {_NODE_ID_EXPR.replace('n.', 'source.')} AS node_id,
                   {{rationale: r.rationale, confidence: r.confidence,
                     evidence_sources: r.evidence_sources,
                     reinforcement_count: r.reinforcement_count,
                     created_by: r.created_by}} AS props
            """,
            {"dup_id": duplicate_id, "primary_id": primary_id},
        )

        # ── Step 3: Re-wire + delete in a single transaction ──
        rewired_out = rewired_in = 0
        async with await self._session() as session:
            tx = await session.begin_transaction()
            try:
                for edge in out_edges:
                    node_id = edge.get("node_id")
                    node_label = edge.get("node_label")
                    rel_type = edge.get("rel_type")
                    if not node_id or not node_label or not rel_type:
                        continue
                    id_field = _label_str_to_id_field(node_label)
                    props = _clean_edge_props(edge.get("props") or {})
                    set_clause, p_params = _edge_set_clause(props, "ep_")
                    q = (
                        f"MATCH (p:Concept {{concept_id: $primary_id}})\n"
                        f"MATCH (t:{node_label} {{{id_field}: $t_id}})\n"
                        f"MERGE (p)-[r:{rel_type}]->(t)\n"
                        f"ON CREATE SET {set_clause}"
                    )
                    await tx.run(q, {"primary_id": primary_id, "t_id": node_id, **p_params})
                    rewired_out += 1

                for edge in in_edges:
                    node_id = edge.get("node_id")
                    node_label = edge.get("node_label")
                    rel_type = edge.get("rel_type")
                    if not node_id or not node_label or not rel_type:
                        continue
                    id_field = _label_str_to_id_field(node_label)
                    props = _clean_edge_props(edge.get("props") or {})
                    set_clause, p_params = _edge_set_clause(props, "ep_")
                    q = (
                        f"MATCH (s:{node_label} {{{id_field}: $s_id}})\n"
                        f"MATCH (p:Concept {{concept_id: $primary_id}})\n"
                        f"MERGE (s)-[r:{rel_type}]->(p)\n"
                        f"ON CREATE SET {set_clause}"
                    )
                    await tx.run(q, {"primary_id": primary_id, "s_id": node_id, **p_params})
                    rewired_in += 1

                await tx.run(
                    "MATCH (n:Concept {concept_id: $dup_id}) DETACH DELETE n",
                    {"dup_id": duplicate_id},
                )
                await tx.commit()
            except Exception:
                await tx.rollback()
                raise

        # Full un-merge record: node snapshot + every rewired edge.  A node
        # snapshot alone cannot restore a merge (edges were rewired and the
        # duplicate's aliases were unioned), so callers journal this dict.
        return {
            "primary_id": primary_id,
            "deleted": duplicate_id,
            "rewired_out": rewired_out,
            "rewired_in": rewired_in,
            "gap_props_copied": len(gap),
            "duplicate_snapshot": {k: v for k, v in dup_node.items() if k != "embedding"},
            "rewired_edges": {
                "out": [dict(e) for e in out_edges],
                "in": [dict(e) for e in in_edges],
            },
        }

    async def add_aliases(
        self, concept_id: str, aliases: list[str], replace: bool = False
    ) -> None:
        """Add (or replace) the aliases list on a Concept node."""
        if replace:
            query = "MATCH (c:Concept {concept_id: $cid}) SET c.aliases = $aliases"
        else:
            query = (
                "MATCH (c:Concept {concept_id: $cid}) "
                "SET c.aliases = [a IN coalesce(c.aliases, []) + $aliases "
                "WHERE a IS NOT NULL | a]"
            )
        async with await self._session() as session:
            await session.run(query, {"cid": concept_id, "aliases": aliases})

    async def retract_node(
        self,
        node_id: str,
        label: NodeLabel,
        reason: str = "",
        hard_delete: bool = False,
    ) -> bool:
        """Retract (soft-delete or hard-delete) a knowledge node.

        Soft delete (default): sets retracted=true + retracted_at + retracted_reason.
          The node remains in the graph but is excluded from all searches and retrieval.
        Hard delete: DETACH DELETE — permanently removes node and all its edges.

        Returns True if the node was found and acted upon.
        """
        id_field = self._id_field_for_label(label)

        if hard_delete:
            async with await self._session() as session:
                result = await session.run(
                    f"MATCH (n:{label.value} {{{id_field}: $node_id}}) "
                    f"WITH n, count(n) AS found DETACH DELETE n RETURN found",
                    {"node_id": node_id},
                )
                record = await result.single()
                return bool(record and record["found"] > 0)

        # Soft delete
        async with await self._session() as session:
            result = await session.run(
                f"""
                MATCH (n:{label.value} {{{id_field}: $node_id}})
                SET n.retracted = true,
                    n.retracted_at = datetime(),
                    n.retracted_reason = $reason
                RETURN n IS NOT NULL AS found
                """,
                {"node_id": node_id, "reason": reason},
            )
            record = await result.single()
            return bool(record and record["found"])

    # ── Stats ──

    async def run_read_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict]:
        """Execute a read-only Cypher query and return records as dicts."""
        async with await self._session() as session:
            result = await session.run(query, params or {})
            return [dict(record) async for record in result]

    @property
    def vector_index_errors(self) -> int:
        """Count of vector-index query failures this store instance has swallowed (matching
        silently degrades to text-only on each). Surfaced by `status` — W-D observability."""
        return getattr(self, "_vector_index_error_count", 0)

    async def get_stats(self) -> dict[str, int]:
        """Get node and edge counts by type."""
        query = """
        MATCH (n)
        WITH labels(n)[0] AS label, count(n) AS cnt
        RETURN label, cnt ORDER BY cnt DESC
        """
        async with await self._session() as session:
            result = await session.run(query)
            return {record["label"]: record["cnt"] async for record in result}

    async def migrate_equation_latex(self, apply: bool = False) -> int:
        """One-time recovery for Equation nodes committed before the apply_delta
        latex -> canonical_latex backfill (see the Equation branch in ``apply_delta``):
        those nodes have a populated ``latex`` property but an empty/missing
        ``canonical_latex``. Dry-run (``apply=False``) reports how many nodes are
        recoverable; ``apply=True`` writes ``canonical_latex = latex`` for exactly those
        nodes. Additive & idempotent — never touches an already-populated
        ``canonical_latex``, never modifies ``latex``; a second ``apply=True`` run
        reports/writes 0 because ``canonical_latex`` is now populated.
        """
        where = (
            "(n.canonical_latex IS NULL OR n.canonical_latex = '') "
            "AND n.latex IS NOT NULL AND n.latex <> ''"
        )
        if apply:
            query = (
                f"MATCH (n:Equation) WHERE {where} "
                "SET n.canonical_latex = n.latex RETURN count(n) AS cnt"
            )
        else:
            query = f"MATCH (n:Equation) WHERE {where} RETURN count(n) AS cnt"
        async with await self._session() as session:
            result = await session.run(query)
            record = await result.single()
            return record["cnt"] if record else 0

    async def dedupe_exact_concepts(
        self, source_id: str | None = None, apply: bool = False
    ) -> int:
        """Merge Concept nodes that share an exact (case-insensitive, trimmed) ``canonical_name``
        into one. The survivor is the highest-degree node in each group; APOC folds the others'
        relationships onto it (``mergeRels: true`` dedupes parallels, ``properties: 'discard'``
        keeps the survivor's props). Exact-name duplicates arise when a non-deterministic
        re-chunk/reprocess re-extracts a concept the matcher then fails to merge (see the
        gm/ID Razavi reprocess). Dry-run (``apply=False``) returns the number of duplicate
        groups; ``apply=True`` performs the merges and returns the number merged.

        Scope to ONE source with ``source_id`` — this is the safe mode: same source_id + same
        name is a true intra-source duplicate. ``source_id=None`` dedupes globally by name,
        which merges ACROSS sources and collapses provenance (e.g. Razavi's "Transconductance"
        into Murmann's) — only use that if you specifically want cross-source concept unification.

        Requires APOC. Connectivity-safe: a merge only ever moves edges onto the survivor, so no
        node loses a *connection* — every neighbour the group reached, the survivor still reaches.
        Idempotent (a second run finds 0 groups). CAVEAT: ``mergeRels: true`` collapses parallel
        edges (survivor and duplicate both linking the same neighbour with the same type) into one
        and keeps only the survivor's edge properties, so accumulate-style edge fields
        (``reinforcement_count``, ``evidence_sources``) on a shared edge are truncated to the
        survivor's copy rather than summed/unioned. Those are soft ranking/provenance signals, not
        stored facts, and connectivity is unaffected — acceptable for an offline maintenance merge,
        but do not rely on it to preserve exact edge-reinforcement counts.
        """
        # Use `is not None` (not truthiness): source_id="" must NOT silently fall through to the
        # dangerous cross-source global merge — an empty string is a caller error, not "all sources".
        src_filter = "AND n.source_id = $source_id " if source_id is not None else ""
        base_params = {"source_id": source_id} if source_id is not None else {}
        find_groups = (
            f"MATCH (n:Concept) WHERE n.canonical_name IS NOT NULL {src_filter}"
            "WITH toLower(trim(n.canonical_name)) AS nm, count(*) AS c "
            "WHERE c > 1 RETURN nm"
        )
        async with await self._session() as session:
            result = await session.run(find_groups, base_params)
            names = [record["nm"] async for record in result]
            if not apply:
                return len(names)
            merge_q = (
                f"MATCH (n:Concept) WHERE toLower(trim(n.canonical_name)) = $nm {src_filter}"
                "WITH n ORDER BY size([(n)--()|1]) DESC WITH collect(n) AS ns "
                "CALL apoc.refactor.mergeNodes(ns, {properties: 'discard', mergeRels: true}) "
                "YIELD node RETURN node"
            )
            for nm in names:
                await (await session.run(merge_q, {**base_params, "nm": nm})).consume()
            return len(names)

    # ── Helpers ──

    @staticmethod
    def _id_field_for_label(label: NodeLabel) -> str:
        mapping = {
            NodeLabel.CONCEPT: "concept_id",
            NodeLabel.EQUATION: "equation_id",
            NodeLabel.PRINCIPLE: "principle_id",
            NodeLabel.CIRCUIT_TOPOLOGY: "topology_id",
            NodeLabel.PARAMETER: "parameter_id",
            NodeLabel.ASSUMPTION: "assumption_id",
            NodeLabel.SOURCE: "source_id",
            NodeLabel.SOURCE_CHUNK: "chunk_id",
            NodeLabel.INSIGHT: "insight_id",
            NodeLabel.MEMORY: "memory_id",
            NodeLabel.ENTITY: "entity_id",
            NodeLabel.SESSION: "session_id",
            NodeLabel.SKILL_RUN: "run_id",
            NodeLabel.HYPOTHESIS: "hypothesis_id",
            NodeLabel.DESIGN_DECISION: "decision_id",
            NodeLabel.BENCH_RESULT: "bench_id",
            NodeLabel.SPECIMEN: "spec_id",
            NodeLabel.CLAIM_CARD: "claim_id",
            NodeLabel.REGULARITY: "law_id",
            NodeLabel.SYMBOLIC_DERIVATION: "derivation_id",
            NodeLabel.LEARNER: "learner_id",
            NodeLabel.ASSESSMENT: "assessment_id",
        }
        return mapping[label]

    def _resolve_ref(
        self, ref: str, delta: GraphDelta
    ) -> tuple[NodeLabel, str]:
        """Resolve a node reference to its label and ID field.

        Checks proposed nodes first, then updated nodes, then
        falls back to Concept (most common case in the graph).
        """
        # Check new nodes in this delta
        for proposal in delta.new_nodes:
            if proposal.proposed_id == ref:
                return proposal.label, self._id_field_for_label(proposal.label)
        # Check updated nodes in this delta
        for update in delta.updated_nodes:
            if update.existing_node_id == ref:
                return update.label, self._id_field_for_label(update.label)
        # Default: most refs are concept_id in practice
        return NodeLabel.CONCEPT, "concept_id"


_VECTOR_INDEXES: list[tuple[str, str]] = [
    ("concept_embedding",   "Concept"),
    ("equation_embedding",  "Equation"),
    ("principle_embedding", "Principle"),
    ("topology_embedding",  "CircuitTopology"),
    ("parameter_embedding", "Parameter"),
]


def _vector_score_to_cosine(score: float) -> float:
    """Convert Neo4j vector-index score to true cosine similarity.

    Neo4j's cosine vector index normalizes scores to [0, 1] via
    score = (1 + cos) / 2, so cos = 2*score - 1.  This is the SINGLE
    place that conversion happens — matcher thresholds are calibrated
    in cosine space and must never see raw index scores.
    """
    return 2.0 * score - 1.0


def _cypher_key(k: str) -> str | None:
    """Backtick-quote a property name that isn't a safe Cypher identifier.

    Strips backticks and non-printable/control characters from the key before
    quoting: Neo4j has no in-identifier escape for a backtick, so a raw one in
    an LLM-controlled key could otherwise close the quoted identifier early
    and inject arbitrary Cypher into a dynamically built SET clause. Returns
    None if the key sanitizes to an empty string — callers must drop such a
    property rather than emit an empty backtick-quoted identifier.
    """
    if k.isidentifier():
        return k
    cleaned = "".join(ch for ch in k if ch.isprintable() and ch != "`")
    if not cleaned:
        return None
    if cleaned.isidentifier():
        return cleaned
    return f"`{cleaned}`"


def _param_key(k: str) -> str:
    """Convert a property name to a safe Cypher parameter key (no special chars).

    The result is spliced unquoted into the query text as ``$<result>`` (see
    ``_set_assignments``), so every character that isn't safe in a bare
    Cypher parameter reference must be neutralized — not just the
    hyphen/dot/space LLM output happens to use most often. Any other
    character (backticks, parens, slashes, control chars, ...) is replaced
    with ``_`` too, and a leading digit is guarded with a ``_`` prefix so the
    result is always a legal bare identifier.
    """
    cleaned = "".join(ch if ch.isascii() and (ch.isalnum() or ch == "_") else "_" for ch in k)
    if cleaned and cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned or "_"


def _param_key_map(keys: Iterable[str], reserved: Iterable[str] = ()) -> dict[str, str]:
    """Map property keys to unique Cypher parameter names.

    ``_param_key`` alone is not injective — e.g. "a-b" and "a.b" both
    sanitize to "a_b" — so two distinct keys passed to it independently can
    collide on one parameter name: a params dict built with a plain dict
    comprehension would silently keep only the last value, while a SET
    clause built the same way still emits two separate assignments that both
    end up reading the survivor. Walking the keys in order and appending a
    stable ``_2``, ``_3``, ... suffix on collision — checked against every
    parameter name already handed out, not just same-base counts, so a
    disambiguated suffix can never accidentally coincide with another key's
    own natural mapping (e.g. keys "a.b" and "a_b_2" both present) — keeps
    every original key mapped to its own parameter name. Deterministic for a
    given ordered sequence of keys (i.e. the same input dict every time).

    ``reserved`` pre-seeds the used set with anchor parameter names the query
    binds itself (e.g. "source_id"/"target_id" in a MERGE-edge MATCH, or a
    node's id_field): an LLM-controlled property key like "source id" would
    otherwise sanitize to "source_id" and — merged after the anchors in the
    params dict — silently overwrite the MATCH anchor, redirecting the write
    to a different node.
    """
    mapping: dict[str, str] = {}
    used: set[str] = set(reserved)
    for k in keys:
        base = _param_key(k)
        candidate = base
        n = 1
        while candidate in used:
            n += 1
            candidate = f"{base}_{n}"
        used.add(candidate)
        mapping[k] = candidate
    return mapping


def _set_assignments(
    props: dict[str, Any],
    entity_ref: str,
    *,
    exclude: str | None = None,
    param_prefix: str = "",
    reserved: Iterable[str] = (),
) -> tuple[list[str], dict[str, Any]]:
    """Build Cypher SET-clause fragments plus a matching, collision-free params dict.

    The single choke point where LLM-controlled property keys enter a
    dynamically built Cypher SET clause. Combines ``_cypher_key`` (identifier
    quoting/sanitizing — drops keys that sanitize to empty) with
    ``_param_key_map`` (collision-free parameter naming) so the returned SET
    fragments and params dict always agree: every surviving original key gets
    exactly one fragment and one dedicated parameter.
    """
    keys: list[str] = []
    for k in props:
        if k == exclude:
            continue
        if _cypher_key(k) is None:
            logger.debug("dropping property with empty-sanitized Cypher key: %r", k)
            continue
        keys.append(k)
    pmap = _param_key_map(keys, reserved=reserved)
    parts: list[str] = []
    params: dict[str, Any] = {}
    for k in keys:
        pkey = f"{param_prefix}{pmap[k]}"
        parts.append(f"{entity_ref}.{_cypher_key(k)} = ${pkey}")
        params[pkey] = props[k]
    return parts, params


#: Identity/provenance fields that generic write paths must NEVER mutate on an
#: EXISTING node. Root cause of the 2026-07-09 incident (#20 family, 9 confirmed
#: defects): LLM-authored deltas flow into MERGE/SET with no created-vs-matched
#: distinction, so a colliding proposed_id from a NEW source silently overwrote a
#: pre-existing node's ``source_id`` and ``canonical_name`` (observed live: a Razavi
#: topology renamed to "Logic High Voltage (logichv)" by a PrimeSim chunk). The set
#: covers: provenance (``source_id``), identity (``canonical_name``), history
#: (``_created_at``), soft-delete state (a generic update must not un-retract), and
#: every label's merge-key field (a FOREIGN id key riding inside a props dict must
#: not overwrite this node's key — the node's OWN key is already excluded as the
#: MERGE anchor). Enforced at the store boundary — the single choke point — rather
#: than in schema/normalize, so every caller (pipeline apply_delta, write_batch,
#: graph_io import, relink, projection) inherits it. Legitimate identity changes go
#: through dedicated audited paths (merge_concepts, retract_node), never these.
#: Fields that MUST be scalar strings but that LLMs occasionally emit as LISTS of strings
#: (proven live: canonical_latex ×3, function ×8, symbol ×1, units ×5 — the units case defeats
#: F6's unit-conflict disqualification because a list normalizes to ""). Coerced (joined) at the
#: tx write choke points so EVERY path — new_nodes, updated_nodes, write_batch — inherits it.
_SCALAR_TEXT_FIELDS: frozenset[str] = frozenset({
    "canonical_latex", "latex", "units", "function", "symbol",
})


def _coerce_scalar_text_fields(props: dict[str, Any]) -> dict[str, Any]:
    for k in _SCALAR_TEXT_FIELDS:
        v = props.get(k)
        if isinstance(v, list):
            props[k] = " ; ".join(str(x) for x in v if x)
    return props


def _filter_proposal_properties(
    props: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Allowlist gate for an LLM-authored proposal's free-form ``properties`` dict
    (migration step 0 write-channel closure, ADR-046 D6; the #20/F8 incident family).

    Drops, rather than writes, any key that could clobber identity or lifecycle
    state on the node being created: everything in ``_PROTECTED_ON_MATCH``
    (merge keys, canonical_name, source_id, retracted*), any ``*_id`` key (id
    assignment belongs to the typed proposal fields, never the free-form dict),
    and any ``_``-prefixed key (internal bookkeeping namespace). Domain content
    (latex, units, condition, description-class keys, ...) passes through.
    Returns (kept, dropped_keys).
    """
    kept: dict[str, Any] = {}
    dropped: list[str] = []
    for k, v in props.items():
        if k in _PROTECTED_ON_MATCH or k.endswith("_id") or k.startswith("_"):
            dropped.append(k)
        else:
            kept[k] = v
    return kept, dropped


_PROTECTED_ON_MATCH: frozenset[str] = frozenset({
    "source_id", "canonical_name", "_created_at",
    "retracted", "retracted_at", "retracted_reason",
    # merge-key fields, one per NodeLabel (exhaustiveness locked by test)
    "concept_id", "equation_id", "principle_id", "topology_id", "parameter_id",
    "assumption_id", "chunk_id", "insight_id", "memory_id", "entity_id",
    "session_id", "run_id", "hypothesis_id", "decision_id", "bench_id",
    "spec_id", "claim_id", "law_id",
    # learner model (S5) — learner_id/assessment_id, added together with NodeLabel.LEARNER/
    # ASSESSMENT (schema.py) per the 9-site checklist (graph/README.md); required for
    # test_protected_set_covers_every_label_merge_key to keep passing.
    "learner_id", "assessment_id",
    # symbolic anchors (registered 2026-07-28 with NodeLabel.SYMBOLIC_DERIVATION)
    "derivation_id",
})


#: Knowledge labels an LLM-authored reasoning-pass delta (``GraphDelta.new_nodes``/
#: ``updated_nodes``) may create or touch via ``apply_delta`` — the pipeline's LLM
#: write path (graph/README.md's dataflow: reasoner -> GraphDelta -> apply_delta).
#: F8(a), same unguarded-boundary FAMILY as ``_PROTECTED_ON_MATCH`` above (the #20
#: incident, commit 85575dc/W-A) — there an LLM-controlled property KEY reached
#: MERGE/SET with no allowlist; here an LLM-controlled LABEL reaches MERGE with no
#: allowlist. Verified incident: 151 live nodes under Memory/Session/SkillRun/
#: Entity were 100% LLM-mislabeled EXTRACTION knowledge — the reasoner emitted a
#: NodeProposal choosing a personal-memory label instead of a knowledge one, and
#: those labels are stripped from the company export (export/graph_io.py
#: DEFAULT_PRIVATE_LABELS), so the underlying knowledge silently vanished from
#: every shipped artifact. reasoning/normalize.py's _VALID_LABELS still accepts
#: every one of these as a "valid" raw-JSON label (Source, SourceChunk, Memory,
#: Entity, Session, SkillRun, Hypothesis, DesignDecision, BenchResult all appear
#: there) — this allowlist is the enforcement the normalize layer doesn't give, at
#: the actual write choke point.
#:
#: EXCLUDED, in three groups (exhaustiveness locked by
#: test_llm_proposable_labels_partition_covers_every_label):
#:   - personal-memory/operational (Memory, Session, SkillRun, Entity, Source,
#:     SourceChunk): not knowledge. Memory/Session/SkillRun/Entity are the labels
#:     the live incident actually hit and are export-stripped by design (they are
#:     supposed to be company-private, not merely mis-shipped); Source/SourceChunk
#:     are pipeline-owned document/chunk bookkeeping with structural ids assigned
#:     by the parse/chunk stages, not reasoner-authored knowledge — the reasoner
#:     has no legitimate reason to mint one via a NodeProposal either.
#:   - curated-write-path-only (Hypothesis, DesignDecision, BenchResult, Learner,
#:     Assessment): each has a dedicated, audited BrainAgent method
#:     (record_hypothesis / record_decision / record_bench_result /
#:     record_assessment) that commits via write_batch with its own edges and
#:     status-transition logic — a reasoning-pass NodeProposal minting one
#:     directly bypasses that audit trail.
#:   - substrate-projection-only (Specimen, ClaimCard, Regularity): additive
#:     projections owned by executable/projection.py / executable/laws.py,
#:     already excluded from reasoning/normalize.py's _VALID_LABELS by the same
#:     "projector-only" discipline (schema.py's NodeLabel.REGULARITY comment) —
#:     this is the second, store-side enforcement of that same rule.
#:
#: Enforced ONLY in apply_delta (the LLM path) — write_batch/merge_node are NOT
#: restricted here; their callers (agent.py, executable/projection.py,
#: executable/laws.py, export/graph_io.py, reasoning/relink.py) are trusted
#: internal code, not LLM output.
_LLM_PROPOSABLE_LABELS: frozenset[NodeLabel] = frozenset({
    NodeLabel.CONCEPT, NodeLabel.EQUATION, NodeLabel.PRINCIPLE,
    NodeLabel.CIRCUIT_TOPOLOGY, NodeLabel.PARAMETER, NodeLabel.ASSUMPTION,
    NodeLabel.INSIGHT,
})


# Tier-3 EVIDENCE edges: the relationship types that assert "a simulation/oracle result grounds
# this" — the only edges in the graph that carry certification weight. schema.py has always
# documented them as "projector-only discipline" with their required endpoints, but nothing
# enforced it, so the reasoning LLM could mint them between arbitrary nodes and did: an audit on
# 2026-07-26 found ~664 of 978 tier-3 edges structurally meaningless — e.g.
# (Concept)'PrimeSim Tool' -[:SUPPORTED_BY]-> (Concept)'Inductor', 169 (Concept)-[:REALIZES]->
# (Concept), 43 (Concept)-[:HAS_CLAIM]->(Concept). Every "what is grounded by simulation" query
# reads these. The endpoints below are taken from the two legitimate minters — executable/
# projection.py (REALIZES/HAS_CLAIM/GROUNDS) and executable/laws.py (SUPPORTED_BY/ABOUT) — and
# are enforced for ALL callers, projector included, because the projector already conforms.
_EVIDENCE_EDGE_ENDPOINTS: dict[RelType, tuple[frozenset[NodeLabel], frozenset[NodeLabel]]] = {
    RelType.REALIZES: (frozenset({NodeLabel.SPECIMEN}),
                       frozenset({NodeLabel.CIRCUIT_TOPOLOGY})),
    RelType.HAS_CLAIM: (frozenset({NodeLabel.SPECIMEN}),
                        frozenset({NodeLabel.CLAIM_CARD})),
    RelType.GROUNDS: (frozenset({NodeLabel.CLAIM_CARD}),
                      frozenset({NodeLabel.CONCEPT, NodeLabel.PARAMETER})),
    RelType.SUPPORTED_BY: (frozenset({NodeLabel.REGULARITY}),
                           frozenset({NodeLabel.CLAIM_CARD})),
    RelType.ABOUT: (frozenset({NodeLabel.REGULARITY}),
                    frozenset({NodeLabel.CIRCUIT_TOPOLOGY})),
    # Symbolic-anchor evidence edges — same counterfeit-protection rationale: a DERIVED_BY or
    # MEASURED_BY from/to the wrong label would fabricate verification provenance.
    RelType.DERIVED_BY: (frozenset({NodeLabel.EQUATION}),
                         frozenset({NodeLabel.SYMBOLIC_DERIVATION})),
    RelType.MEASURED_BY: (frozenset({NodeLabel.EQUATION}),
                          frozenset({NodeLabel.CLAIM_CARD})),
}


def _evidence_edge_violation(
    rel_type: RelType, source_label: NodeLabel, target_label: NodeLabel,
) -> str | None:
    """Return a reason string if this edge would be a counterfeit evidence edge, else None."""
    allowed = _EVIDENCE_EDGE_ENDPOINTS.get(rel_type)
    if allowed is None:
        return None
    src_ok, tgt_ok = allowed
    if source_label in src_ok and target_label in tgt_ok:
        return None
    return (f"{rel_type.value} requires "
            f"({'|'.join(sorted(l.value for l in src_ok))})->"
            f"({'|'.join(sorted(l.value for l in tgt_ok))}), "
            f"got ({source_label.value})->({target_label.value})")


def _set_assignments_split(
    props: dict[str, Any],
    entity_ref: str,
    *,
    exclude: str | None = None,
    param_prefix: str = "",
    reserved: Iterable[str] = (),
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Like ``_set_assignments`` but returns (create_parts, match_parts, params),
    where ``match_parts`` omits fragments for ``_PROTECTED_ON_MATCH`` keys.

    Both clause lists are derived from ONE ``_param_key_map`` call, so a ``$param``
    name always means the same value in the ON CREATE and ON MATCH branches —
    calling ``_set_assignments`` twice with different key subsets could assign the
    same param name to different keys (its collision suffixes depend on the key
    set), which inside a single MERGE query would silently bind the wrong value.
    Unused params in the MATCH branch are harmless (Neo4j ignores extra bindings).
    """
    keys: list[str] = []
    for k in props:
        if k == exclude:
            continue
        if _cypher_key(k) is None:
            logger.debug("dropping property with empty-sanitized Cypher key: %r", k)
            continue
        keys.append(k)
    pmap = _param_key_map(keys, reserved=reserved)
    create_parts: list[str] = []
    match_parts: list[str] = []
    params: dict[str, Any] = {}
    blocked: list[str] = []
    for k in keys:
        pkey = f"{param_prefix}{pmap[k]}"
        fragment = f"{entity_ref}.{_cypher_key(k)} = ${pkey}"
        create_parts.append(fragment)
        if k in _PROTECTED_ON_MATCH:
            blocked.append(k)
        else:
            match_parts.append(fragment)
        params[pkey] = props[k]
    if blocked:
        # debug, not warning: every idempotent re-merge (re-ingest, law re-projection)
        # legitimately carries identical identity values — spamming warnings on the
        # happy path would bury real signals during a 3k-chunk ingest.
        logger.debug("identity fields excluded from ON MATCH: %s", blocked)
    return create_parts, match_parts, params


def _sanitize_neo4j_value(value: Any) -> Any:
    """Normalize values into Neo4j property-compatible primitives/arrays.

    Neo4j properties may only be primitives or arrays of primitives.
    Nested dict/list structures are serialized to compact JSON strings so
    extraction metadata is preserved instead of crashing the commit.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return json.dumps(value, default=str, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (tuple, set)):
        value = list(value)
    if isinstance(value, list):
        sanitized = [_sanitize_neo4j_value(v) for v in value]
        # Neo4j array rules are stricter than "all primitives" (the S4 chunk-2084 incident:
        # one commit in 3,223 chunks died on Neo.ClientError.Statement.TypeError):
        #   * arrays may NOT contain null -> drop Nones;
        #   * arrays must be HOMOGENEOUS -> a mixed-primitive list ([str, int], [True, 1]...)
        #     is coerced to all-str (bool is an int subclass in Python, so group by exact
        #     type(); int+float may co-exist only as all-float).
        sanitized = [v for v in sanitized if v is not None]
        if all(isinstance(v, (str, int, float, bool)) for v in sanitized):
            kinds = {type(v) for v in sanitized}
            if len(kinds) <= 1:
                return sanitized
            if kinds <= {int, float}:
                return [float(v) for v in sanitized]
            return [str(v) for v in sanitized]
        return json.dumps(sanitized, default=str, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _label_str_to_id_field(label: str) -> str:
    """Map a node label string to its primary ID field name.

    NOTE (graph/README.md Known Issue #3): this second, string-keyed map predates the
    executable substrate and is stale for SPECIMEN/CLAIM_CARD/REGULARITY — not fixed here
    (out of scope for the learner-model change that touched this function next). Learner/
    Assessment ARE included below: S5_LEARNER_MODEL_DESIGN.md §2 Option A's cons list names
    this exact map as one half of the "9-site checklist" a new label triggers, warning
    explicitly against repeating the merge_concepts trap of updating only one of the two
    label->id-field maps — so the new labels are added to both here and in
    GraphStore._id_field_for_label, even though the map's PRE-EXISTING gap for the three
    executable labels is left as found.
    """
    return {
        "Concept": "concept_id",
        "Equation": "equation_id",
        "Principle": "principle_id",
        "CircuitTopology": "topology_id",
        "Parameter": "parameter_id",
        "Assumption": "assumption_id",
        "Source": "source_id",
        "SourceChunk": "chunk_id",
        "Insight": "insight_id",
        "Memory": "memory_id",
        "Entity": "entity_id",
        "Session": "session_id",
        "SkillRun": "run_id",
        "Hypothesis": "hypothesis_id",
        "DesignDecision": "decision_id",
        "BenchResult": "bench_id",
        "Learner": "learner_id",
        "Assessment": "assessment_id",
        # symbolic anchors (2026-07-28): registered here alongside NodeLabel to avoid the
        # falls-back-to-concept_id trap; the SPECIMEN/CLAIM_CARD/REGULARITY gap (Known
        # Issue #3) is still left as found.
        "SymbolicDerivation": "derivation_id",
    }.get(label, "concept_id")


def _clean_edge_props(props: dict) -> dict:
    """Extract transferable edge properties — skip internal/datetime fields."""
    _KEEP = {"rationale", "confidence", "evidence_sources", "reinforcement_count", "created_by"}
    result = {}
    for k, v in props.items():
        if k not in _KEEP or v is None:
            continue
        if isinstance(v, list):
            result[k] = [i for i in v if isinstance(i, (str, int, float, bool))]
        elif isinstance(v, (str, int, float, bool)):
            result[k] = v
    return result


def _edge_set_clause(props: dict, prefix: str = "ep_") -> tuple[str, dict]:
    """Build a Cypher SET clause and matching params dict for edge properties.

    Returns (set_clause_str, params_dict).  Always includes r._created_at.
    """
    key_parts, params = _set_assignments(props, "r", param_prefix=prefix)
    parts = ["r._created_at = datetime()", *key_parts]
    return ", ".join(parts), params
