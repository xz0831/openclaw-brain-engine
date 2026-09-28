"""Streaming JSONL graph export/import — the ADR-044 D1 downloadable graph artifact.

The shared-service profile does not run a full openclaw-brain instance (docs/DECISIONS.md
ADR-044 D1): it gets a read-only MCP server plus this artifact — a graph snapshot
it can `import-graph` into its own Neo4j. The EvidenceVault (verbatim source text)
is never distributed; only citation metadata (source ids / chunk hashes) travels.

Format — JSONL, one JSON object per line:
    1. Header (always first, no ``kind`` key):
       ``{format_version, exported_at, exclude_labels, include_embeddings, counts}``
    2. One line per node:
       ``{kind: "node", label, props}``
    3. One line per edge:
       ``{kind: "edge", rel_type, source_label, source_id_field, source_id,
          target_label, target_id_field, target_id, props}``

Both directions stream: ``export_graph`` pages through Neo4j one label at a time
(SKIP/LIMIT, never materializing more than one page of rows), and ``import_graph``
reads the file line by line, flushing accumulated node/edge batches to
``GraphStore.write_batch()`` every ``batch_size`` items. Neither function holds
the whole graph in memory — the live graph is ~12k nodes, but a 159MB JSON export
precedent (`experiments/ARCH_SURVEY_2026-07-03.md`) is exactly the failure mode
this avoids.

Writes go through GraphStore's existing ``write_batch``/``merge_node``/``merge_edge``
surface only — ``_set_assignments`` (graph/store.py) stays the single choke point
for dynamic-key SET clauses; this module never builds its own Cypher for writes.

Verbatim-text note (verified against graph/schema.py + knowledge/pipeline.py): a
SourceChunk node's ``raw_text_hash`` is a hash/pointer (safe to export — it is
exactly the "source ids / chunk hashes" ADR-044 D1 allows), but its
``text_preview`` field holds up to 500 CHARACTERS OF VERBATIM SOURCE TEXT
(``knowledge/pipeline.py::_register_chunk`` sets it to ``chunk.text[:500]``, and
``agent.py::get_evidence`` falls back to serving exactly that text when the
EvidenceVault doesn't have the hash). That is content, not metadata — despite
SourceChunk nodes being exported (citation metadata is explicitly allowed),
``text_preview`` is stripped unconditionally, independent of ``include_embeddings``
or any other flag, so the artifact never carries copyrighted excerpts.

Type-fidelity note: Neo4j temporal properties (``_created_at``,
``last_reinforced``, ...) serialize via ``json.dumps(default=str)`` and
re-import as plain STRINGS. Harmless for the read-only shared-service profile, but
datetime-comparison maintenance (decay/promotion) must not be run against
an imported graph without re-typing those properties first.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openclaw_brain.knowledge.graph.schema import NodeLabel, RelType
from openclaw_brain.knowledge.graph.store import GraphStore

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1

# Personal-memory-subsystem labels — mirrors export/obsidian.py's _SKIP_LABELS
# minus SourceChunk. SourceChunk carries citation metadata (allowed shared-service side
# per ADR-044 D1) so it is exported; Memory/Session/SkillRun/Entity are the
# agent's personal memory and must never leave the home instance.
#
# Learner/Assessment (S5 learner model, docs/specs/S5_LEARNER_MODEL_DESIGN.md §2/§4.1/§6.1):
# home-only for the whole S5 phase, mechanism-safety grounds (§4.1) — Rick's personal learning
# history must never leave the home instance. Added in the SAME change as the schema addition
# (schema.py NodeLabel.LEARNER/ASSESSMENT), per §1.4's finding that "exclude by label" already
# failed silently once for four labels when this discipline lapsed — this is a non-negotiable
# same-commit companion, not a follow-up.
DEFAULT_PRIVATE_LABELS: frozenset[str] = frozenset(
    {"Memory", "Session", "SkillRun", "Entity", "Learner", "Assessment"}
)

# Embeddings are a rebuildable index (README: "rebuildable index, not the asset
# of record") — stripped by default to keep the artifact small; shared-service side
# regenerates via `openclaw-brain backfill-embeddings`. `embedding_model` is
# paired metadata that's meaningless without the vector, so it travels with it.
_EMBEDDING_PROPS: frozenset[str] = frozenset({"embedding", "embedding_model"})

# See module docstring "Verbatim-text note" — verbatim source text, always stripped.
_SOURCE_CHUNK_TEXT_PROPS: frozenset[str] = frozenset({"text_preview"})

ProgressFn = Callable[[str, int, int], None]


# ── Export ──


async def export_graph(
    store: GraphStore,
    out_path: str | Path,
    *,
    exclude_labels: Iterable[str] = DEFAULT_PRIVATE_LABELS,
    include_embeddings: bool = False,
    page_size: int = 500,
    on_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Stream the graph to a JSONL artifact at ``out_path``.

    Retracted (soft-deleted) nodes are excluded, mirroring every other reader
    of the graph (`query_knowledge`, the Obsidian exporter). Returns the header
    dict actually written (includes per-label/per-rel-type counts).
    """
    exclude = {label for label in exclude_labels if label}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    node_counts = await _count_nodes(store, exclude)
    edge_counts = await _count_edges(store, exclude)
    total_edges = sum(edge_counts.values())

    header = {
        "format_version": FORMAT_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "exclude_labels": sorted(exclude),
        "include_embeddings": include_embeddings,
        "counts": {"nodes": node_counts, "edges": edge_counts},
    }

    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header, default=str, ensure_ascii=False) + "\n")

        for label in sorted(node_counts):
            if not label.isidentifier():
                logger.warning("export_graph: skipping label with unsafe identifier: %r", label)
                continue
            label_total = node_counts[label]
            label_done = 0
            skip = 0
            while True:
                rows = await store.run_read_query(
                    f"""
                    MATCH (n:{label})
                    WHERE NOT coalesce(n.retracted, false)
                    RETURN properties(n) AS props
                    ORDER BY elementId(n)
                    SKIP $skip LIMIT $limit
                    """,
                    {"skip": skip, "limit": page_size},
                )
                if not rows:
                    break
                for row in rows:
                    props = _clean_node_props(label, row["props"], include_embeddings)
                    f.write(json.dumps(
                        {"kind": "node", "label": label, "props": props},
                        default=str, ensure_ascii=False,
                    ) + "\n")
                label_done += len(rows)
                if on_progress:
                    on_progress(label, label_done, label_total)
                if len(rows) < page_size:
                    break
                skip += page_size

        done_edges = 0
        skip = 0
        while True:
            rows = await store.run_read_query(
                """
                MATCH (a)-[r]->(b)
                WHERE NOT any(l IN labels(a) WHERE l IN $exclude)
                  AND NOT any(l IN labels(b) WHERE l IN $exclude)
                  AND NOT coalesce(a.retracted, false)
                  AND NOT coalesce(b.retracted, false)
                RETURN type(r) AS rel_type,
                       labels(a)[0] AS source_label, properties(a) AS a_props,
                       labels(b)[0] AS target_label, properties(b) AS b_props,
                       properties(r) AS props
                ORDER BY elementId(r)
                SKIP $skip LIMIT $limit
                """,
                {"exclude": sorted(exclude), "skip": skip, "limit": page_size},
            )
            if not rows:
                break
            for row in rows:
                edge_line = _edge_to_line(row)
                if edge_line is None:
                    continue
                f.write(json.dumps(edge_line, default=str, ensure_ascii=False) + "\n")
            done_edges += len(rows)
            if on_progress:
                on_progress("<edges>", done_edges, total_edges)
            if len(rows) < page_size:
                break
            skip += page_size

    return header


async def _count_nodes(store: GraphStore, exclude: set[str]) -> dict[str, int]:
    rows = await store.run_read_query(
        """
        MATCH (n)
        WHERE NOT any(l IN labels(n) WHERE l IN $exclude)
          AND NOT coalesce(n.retracted, false)
        WITH labels(n)[0] AS label, count(n) AS cnt
        RETURN label, cnt
        """,
        {"exclude": sorted(exclude)},
    )
    return {r["label"]: r["cnt"] for r in rows}


async def _count_edges(store: GraphStore, exclude: set[str]) -> dict[str, int]:
    rows = await store.run_read_query(
        """
        MATCH (a)-[r]->(b)
        WHERE NOT any(l IN labels(a) WHERE l IN $exclude)
          AND NOT any(l IN labels(b) WHERE l IN $exclude)
          AND NOT coalesce(a.retracted, false)
          AND NOT coalesce(b.retracted, false)
        WITH type(r) AS rel_type, count(r) AS cnt
        RETURN rel_type, cnt
        """,
        {"exclude": sorted(exclude)},
    )
    return {r["rel_type"]: r["cnt"] for r in rows}


def _clean_node_props(label: str, props: dict[str, Any], include_embeddings: bool) -> dict[str, Any]:
    props = dict(props)
    if not include_embeddings:
        for k in _EMBEDDING_PROPS:
            props.pop(k, None)
    if label == "SourceChunk":
        for k in _SOURCE_CHUNK_TEXT_PROPS:
            props.pop(k, None)
    return props


def _id_field_for_label_str(label: str) -> str | None:
    try:
        return GraphStore._id_field_for_label(NodeLabel(label))
    except ValueError:
        return None


def _edge_to_line(row: dict[str, Any]) -> dict[str, Any] | None:
    source_label = row["source_label"]
    target_label = row["target_label"]
    source_id_field = _id_field_for_label_str(source_label)
    target_id_field = _id_field_for_label_str(target_label)
    if source_id_field is None or target_id_field is None:
        logger.warning(
            "export_graph: skipping edge with unrecognized endpoint label(s): %r -> %r",
            source_label, target_label,
        )
        return None
    source_id = (row["a_props"] or {}).get(source_id_field)
    target_id = (row["b_props"] or {}).get(target_id_field)
    if not source_id or not target_id:
        logger.warning("export_graph: skipping edge with missing endpoint id field")
        return None
    return {
        "kind": "edge",
        "rel_type": row["rel_type"],
        "source_label": source_label,
        "source_id_field": source_id_field,
        "source_id": source_id,
        "target_label": target_label,
        "target_id_field": target_id_field,
        "target_id": target_id,
        "props": row["props"] or {},
    }


# ── Import ──


async def import_graph(
    store: GraphStore,
    in_path: str | Path,
    *,
    batch_size: int = 500,
    on_progress: ProgressFn | None = None,
) -> dict[str, int]:
    """Replay a JSONL artifact into ``store``.

    MERGE semantics (via `GraphStore.write_batch`) make this idempotent — an
    import run twice, or over a graph that already has some of these nodes,
    never duplicates. Malformed lines and edges/nodes referencing a label
    outside the current schema are skipped (counted, not fatal) rather than
    aborting the whole import.
    """
    in_path = Path(in_path)
    counts = {"nodes_imported": 0, "edges_imported": 0, "nodes_skipped": 0, "edges_skipped": 0}
    header: dict[str, Any] | None = None
    header_total = 0

    node_batch: list[dict[str, Any]] = []
    edge_batch: list[dict[str, Any]] = []

    async def _flush() -> None:
        nonlocal node_batch, edge_batch
        if not node_batch and not edge_batch:
            return
        await store.write_batch(nodes=node_batch, edges=edge_batch)
        counts["nodes_imported"] += len(node_batch)
        counts["edges_imported"] += len(edge_batch)
        node_batch = []
        edge_batch = []
        if on_progress:
            on_progress(
                "import",
                counts["nodes_imported"] + counts["edges_imported"],
                header_total,
            )

    with in_path.open("r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("import_graph: skipping malformed JSON at line %d", lineno)
                continue

            if header is None:
                header = item
                if header.get("format_version") != FORMAT_VERSION:
                    raise ValueError(
                        f"import_graph: unsupported format_version "
                        f"{header.get('format_version')!r} (expected {FORMAT_VERSION}) in {in_path}"
                    )
                node_hdr = (header.get("counts") or {}).get("nodes") or {}
                edge_hdr = (header.get("counts") or {}).get("edges") or {}
                header_total = sum(node_hdr.values()) + sum(edge_hdr.values())
                continue

            kind = item.get("kind")
            if kind == "node":
                node = _node_from_line(item)
                if node is None:
                    counts["nodes_skipped"] += 1
                    continue
                node_batch.append(node)
            elif kind == "edge":
                edge = _edge_from_line(item)
                if edge is None:
                    counts["edges_skipped"] += 1
                    continue
                edge_batch.append(edge)
            else:
                logger.warning("import_graph: skipping line %d with unknown kind %r", lineno, kind)
                continue

            if len(node_batch) + len(edge_batch) >= batch_size:
                await _flush()

    await _flush()
    return counts


def _node_from_line(item: dict[str, Any]) -> dict[str, Any] | None:
    label = item.get("label")
    props = item.get("props") or {}
    if not label:
        return None
    try:
        node_label = NodeLabel(label)
    except ValueError:
        logger.warning("import_graph: skipping node with unknown label %r", label)
        return None
    id_field = GraphStore._id_field_for_label(node_label)
    id_value = props.get(id_field)
    if not id_value:
        logger.warning("import_graph: skipping %s node missing id field %r", label, id_field)
        return None
    return {"label": node_label, "id_field": id_field, "id_value": id_value, "properties": props}


def _edge_from_line(item: dict[str, Any]) -> dict[str, Any] | None:
    try:
        rel_type = RelType(item["rel_type"])
        source_label = NodeLabel(item["source_label"])
        target_label = NodeLabel(item["target_label"])
    except (KeyError, ValueError) as exc:
        logger.warning("import_graph: skipping edge with invalid type/label: %s", exc)
        return None
    source_id = item.get("source_id")
    target_id = item.get("target_id")
    if not source_id or not target_id:
        logger.warning("import_graph: skipping edge missing source/target id")
        return None
    source_id_field = item.get("source_id_field") or GraphStore._id_field_for_label(source_label)
    target_id_field = item.get("target_id_field") or GraphStore._id_field_for_label(target_label)
    return {
        "source_label": source_label,
        "source_id_field": source_id_field,
        "source_id_value": source_id,
        "target_label": target_label,
        "target_id_field": target_id_field,
        "target_id_value": target_id,
        "rel_type": rel_type,
        "properties": item.get("props") or {},
    }
