"""Export Neo4j knowledge graph to an Obsidian vault.

Writes each knowledge node as a Markdown file with YAML frontmatter.
Relationships become [[wikilinks]] so Obsidian's graph view works.

Typed frontmatter links (``ObsidianExporter(typed_links=True)``, the
default): outgoing edges are additionally grouped by ``rel_type`` (lowered,
e.g. ``DEPENDS_ON`` -> ``depends_on``) into frontmatter list properties whose
values are the same ``[[wikilink]]`` strings the body "## Relationships"
section renders — both resolve through the identical `_wikilink_for` /
`_id_to_filename` choke point, so they can never point at different files.
Pass ``typed_links=False`` (CLI: ``--no-typed-links``; MCP tool:
``typed_links=False``) to omit them. See `../llm/README.md` ("Obsidian
exporter integrity") for the full writeup.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from neo4j import AsyncDriver, AsyncSession

from openclaw_brain.config import Neo4jConfig


# ── Node type → subfolder mapping ──

_FOLDER_MAP = {
    "Concept": "Concepts",
    "Equation": "Equations",
    "Principle": "Principles",
    "CircuitTopology": "Circuits",
    "Parameter": "Parameters",
    "Assumption": "Assumptions",
    "Insight": "Insights",
    "Source": "Sources",
    "Regularity": "Regularities",
}

# Node types to skip (internal bookkeeping, not useful in Obsidian)
_SKIP_LABELS = {"SourceChunk", "Memory", "Entity", "Session", "SkillRun"}

# label -> id property name. Mirrors ObsidianExporter._get_id (which is the
# canonical consumer); kept at module level so _MANAGED_LABELS below can be
# derived from it without duplicating the mapping.
_ID_FIELDS = {
    "Concept": "concept_id",
    "Equation": "equation_id",
    "Principle": "principle_id",
    "CircuitTopology": "topology_id",
    "Parameter": "parameter_id",
    "Assumption": "assumption_id",
    "Source": "source_id",
    "SourceChunk": "chunk_id",
    "Insight": "insight_id",
    # executable substrate + design-reasoning labels carry a <label>_id, not `id`
    "ClaimCard": "claim_id",
    "Specimen": "spec_id",
    "Hypothesis": "hypothesis_id",
    "DesignDecision": "decision_id",
    "BenchResult": "bench_id",
    # law-tier graph representation (spec 2026-07-04): projector-only label, id is `law_id`.
    "Regularity": "law_id",
    # symbolic anchors (2026-07-28): lcapy/sympy derivation records, exported like the other
    # machine-evidence labels (ClaimCard/Specimen) into a label-named folder.
    "SymbolicDerivation": "derivation_id",
}

# Every label the exporter ever writes files for. `_fetch_all_nodes` filters
# retracted nodes in its WHERE clause *before* the collect() aggregation, so
# a label whose every node is currently retracted produces NO row at all
# (not an empty group) — the label key is simply absent from the fetch
# result. Phase 3 of export() only iterates labels present in that result,
# so without this set such a label's subfolder would never be visited again
# and its stale notes would persist forever. See the post-loop prune step
# in export().
_MANAGED_LABELS = (set(_FOLDER_MAP) | set(_ID_FIELDS)) - _SKIP_LABELS


def _sanitize_filename(name: str) -> str:
    """Make a string safe for use as a filename."""
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name[:200] if name else "untitled"


def _short_id_suffix(node_id: str) -> str:
    """Short deterministic suffix derived from a node id.

    Used to disambiguate filename collisions (2+ node ids that sanitize to
    the same display name — e.g. Parameter symbols 'gm', 'W', 'L' shared by
    many distinct parameter_ids). Pure function of the id content, so it is
    stable regardless of fetch/iteration order — mirrors the id-suffix
    pattern already used for ClaimCard/Specimen display names.
    """
    return hashlib.sha1(node_id.encode("utf-8")).hexdigest()[:8]


def _snake_to_title(name: str) -> str:
    """Convert snake_case to Title Case if the name looks like a snake_case ID.

    Examples:
        'fd_soi' → 'FD SOI'
        'common_source_amplifier' → 'Common Source Amplifier'
        'Threshold Voltage' → 'Threshold Voltage' (no change)
    """
    if "_" not in name or " " in name:
        return name
    # Common abbreviations that should stay uppercase
    _UPPER = {"fd", "soi", "cmos", "mosfet", "nmos", "pmos", "vlsi", "ic",
              "dc", "ac", "rf", "ldo", "adc", "dac", "pll", "vco", "esd",
              "io", "pcb", "bga", "tsv", "snr", "thd", "opamp", "bjt", "jfet"}
    parts = name.split("_")
    titled = []
    for part in parts:
        if part.lower() in _UPPER:
            titled.append(part.upper())
        else:
            titled.append(part.capitalize())
    return " ".join(titled)


class _WikiLink(str):
    """A string marking an Obsidian ``[[wikilink]]`` for frontmatter rendering.

    A bare ``- [[Name]]`` list item is invalid as YAML wants it: `[[` opens a
    nested flow sequence, so it must be quoted. `_frontmatter()` quotes and
    escapes only list items of this type — plain string list items (e.g.
    `variable_signature`, `pdks`) keep their existing, unquoted rendering, so
    that behavior never regresses.
    """

    __slots__ = ()


def _frontmatter(props: dict[str, Any]) -> str:
    """Build YAML frontmatter block."""
    lines = ["---"]
    for key, val in props.items():
        if val is None or val == "" or val == []:
            continue
        if isinstance(val, list):
            lines.append(f"{key}:")
            for item in val:
                if isinstance(item, _WikiLink):
                    # Quote + escape only wikilink items — `\` before `"` so
                    # an already-escaped backslash is never double-escaped.
                    escaped = item.replace("\\", "\\\\").replace('"', '\\"')
                    lines.append(f'  - "{escaped}"')
                else:
                    lines.append(f"  - {item}")
        elif isinstance(val, bool):
            lines.append(f"{key}: {'true' if val else 'false'}")
        elif isinstance(val, (int, float)):
            lines.append(f"{key}: {val}")
        else:
            # Escape strings that could break YAML
            s = str(val)
            if any(c in s for c in ":#{}[]|>&*!") or s.startswith(("'", '"')):
                s = f'"{s}"'
            lines.append(f"{key}: {s}")
    lines.append("---")
    return "\n".join(lines)


class ObsidianExporter:
    """Exports a Neo4j knowledge graph to Obsidian-compatible Markdown."""

    def __init__(
        self, config: Neo4jConfig, vault_path: Path, typed_links: bool = True,
        *, egress: str | None = None,
    ):
        self._config = config
        self._egress = egress
        self._vault = vault_path
        # Group outgoing edges by rel_type into frontmatter [[wikilink]]
        # list properties (default on). See module docstring.
        self._typed_links = typed_links
        self._driver: AsyncDriver | None = None
        # canonical_name/title → filename (for wikilink resolution)
        self._id_to_filename: dict[str, str] = {}
        # Stats from the most recent export() call — files_written/collisions_suffixed/pruned
        # per label, for honest reporting beyond the dict[str,int] the public API returns.
        self.last_export_stats: dict[str, Any] = {}

    async def connect(self) -> None:
        from neo4j import AsyncGraphDatabase

        from openclaw_brain.config import resolve_neo4j_password
        from openclaw_brain.egress import check_neo4j_uri

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
        assert self._driver
        return self._driver.session(database=self._config.database)

    # ── Main export ──

    async def export(self) -> dict[str, int]:
        """Export all knowledge nodes and their relationships.

        Returns counts by node type — the number of files actually written
        for that label (post retracted-filtering and collision-safe naming),
        not the raw node count fetched from Neo4j.
        """
        # Phase 1: fetch all nodes, drop retracted ones, then build a
        # collision-safe ID→filename index BEFORE any note body is rendered
        # (so wikilinks, built from that same index, always resolve).
        raw_nodes = await self._fetch_all_nodes()
        nodes = {
            label: [n for n in node_list if not n.get("retracted")]
            for label, node_list in raw_nodes.items()
        }
        collisions_by_label = self._build_filename_map(nodes)

        # Phase 2: fetch all relationships
        edges = await self._fetch_all_edges()

        # Phase 3: write files, then prune stale files left by earlier runs
        # (merged/retracted nodes) — scoped to each label's own subfolder only.
        counts: dict[str, int] = {}
        pruned_by_label: dict[str, int] = {}
        for label, node_list in nodes.items():
            if label in _SKIP_LABELS:
                continue
            folder = self._vault / _FOLDER_MAP.get(label, label)
            folder.mkdir(parents=True, exist_ok=True)
            written_names: set[str] = set()
            for node in node_list:
                node_id = self._get_id(label, node)
                filename = self._id_to_filename[node_id]
                filepath = folder / f"{filename}.md"
                content = self._render_node(label, node, node_id, edges)
                filepath.write_text(content, encoding="utf-8")
                written_names.add(filepath.name)
            counts[label] = len(written_names)
            pruned_by_label[label] = self._prune_stale_files(folder, written_names)

        # A managed label can be entirely absent from `nodes` even though it
        # has files on disk from a previous run: `_fetch_all_nodes` filters
        # retracted nodes before the collect() aggregation, so a label whose
        # every node just got retracted emits no row at all (never an empty
        # group) and the loop above never visits it. Still prune those
        # subfolders to empty here, or the orphaned notes persist forever.
        for label in sorted(_MANAGED_LABELS - nodes.keys()):
            folder = self._vault / _FOLDER_MAP.get(label, label)
            if not folder.is_dir():
                continue  # never populated by an export — nothing to prune
            pruned = self._prune_stale_files(folder, written_names=set())
            if pruned:
                pruned_by_label[label] = pruned
            counts[label] = 0

        self.last_export_stats = {
            "files_written": dict(counts),
            "collisions_suffixed": collisions_by_label,
            "pruned": pruned_by_label,
        }

        # Write vault index — counts here MUST equal files actually written.
        self._write_index(nodes, counts, collisions_by_label, pruned_by_label)

        return counts

    # ── Data fetching ──

    async def _fetch_all_nodes(self) -> dict[str, list[dict[str, Any]]]:
        """Fetch all nodes grouped by label.

        Excludes retracted nodes — mirrors graph/store.py's
        `NOT coalesce(n.retracted, false)` search filter, so soft-deleted
        (merged/retracted) nodes never resurface as Obsidian notes.
        """
        query = """
        MATCH (n)
        WHERE NOT any(l IN labels(n) WHERE l IN $skip)
          AND NOT coalesce(n.retracted, false)
        WITH n, labels(n)[0] AS label
        RETURN label, collect(properties(n)) AS nodes
        ORDER BY label
        """
        result_map: dict[str, list[dict]] = {}
        async with await self._session() as session:
            result = await session.run(query, {"skip": list(_SKIP_LABELS)})
            async for record in result:
                result_map[record["label"]] = record["nodes"]
        return result_map

    async def _fetch_all_edges(self) -> dict[str, list[dict[str, Any]]]:
        """Fetch all edges, keyed by source node ID.

        Returns {source_id: [{target_id, rel_type, rationale, confidence}, ...]}
        """
        query = """
        MATCH (a)-[r]->(b)
        WHERE NOT any(l IN labels(a) WHERE l IN $skip)
          AND NOT any(l IN labels(b) WHERE l IN $skip)
          AND NOT coalesce(a.retracted, false)
          AND NOT coalesce(b.retracted, false)
        WITH a, b, r, labels(a)[0] AS a_label, labels(b)[0] AS b_label
        RETURN
            properties(a) AS a_props,
            a_label,
            properties(b) AS b_props,
            b_label,
            type(r) AS rel_type,
            r.rationale AS rationale,
            r.confidence AS confidence,
            r.reinforcement_count AS reinforcement_count
        """
        edges: dict[str, list[dict]] = {}
        async with await self._session() as session:
            result = await session.run(query, {"skip": list(_SKIP_LABELS)})
            async for record in result:
                source_id = self._get_id(record["a_label"], record["a_props"])
                target_id = self._get_id(record["b_label"], record["b_props"])
                edge_info = {
                    "target_id": target_id,
                    "target_label": record["b_label"],
                    "rel_type": record["rel_type"],
                    "rationale": record["rationale"] or "",
                    "confidence": record["confidence"],
                    "reinforcement_count": record["reinforcement_count"],
                }
                edges.setdefault(source_id, []).append(edge_info)
        return edges

    # ── Filename index ──

    def _build_filename_map(
        self, nodes: dict[str, list[dict[str, Any]]]
    ) -> dict[str, int]:
        """Build `self._id_to_filename`, giving every node a unique filename.

        Distinct node ids can sanitize to the same display name (e.g.
        Parameter symbols 'gm', 'W', 'L' shared across many parameter_ids) —
        left alone, later writes silently overwrite earlier ones. When 2+
        node ids within a label collide on the same base filename, EVERY one
        of them gets a short id-derived suffix appended (mirrors the
        id-suffix pattern used for ClaimCard/Specimen display names).

        Deterministic: the suffix is a pure function of the node id, and
        colliding ids are processed in sorted order, so the same graph
        produces the same filenames regardless of fetch/iteration order.

        Must run to completion before any note body is rendered — the
        [[wikilinks]] emitted by `_render_node` are resolved via
        `self._id_to_filename`, so they only match final (possibly
        suffixed) filenames if this index is fully built first.

        Returns {label: number_of_node_ids_suffixed}.
        """
        self._id_to_filename = {}
        collisions_by_label: dict[str, int] = {}
        for label, node_list in nodes.items():
            base_by_id: dict[str, str] = {}
            for node in node_list:
                node_id = self._get_id(label, node)
                base_by_id[node_id] = _sanitize_filename(self._display_name(label, node))

            # Group case-insensitively (casefold): the live vault sits on a
            # case-insensitive filesystem (APFS), where 'GM.md' and 'gm.md'
            # are the SAME file — names differing only in case must also be
            # treated as colliding or one node silently overwrites the other.
            by_base: dict[str, list[str]] = {}
            for node_id, base in base_by_id.items():
                by_base.setdefault(base.casefold(), []).append(node_id)

            suffixed = 0
            for ids in by_base.values():
                if len(ids) == 1:
                    self._id_to_filename[ids[0]] = base_by_id[ids[0]]
                    continue
                for node_id in sorted(ids):
                    base = base_by_id[node_id]
                    self._id_to_filename[node_id] = f"{base} ({_short_id_suffix(node_id)})"
                suffixed += len(ids)
            if suffixed:
                collisions_by_label[label] = suffixed

        return collisions_by_label

    @staticmethod
    def _prune_stale_files(folder: Path, written_names: set[str]) -> int:
        """Delete `.md` files in `folder` that this export run did not write.

        Scoped to a single managed label subfolder ONLY — never touches
        INDEX.md, the vault root, or any other folder — so nodes that were
        merged/retracted since the last export don't leave orphan files
        behind forever. Returns the number of files pruned.
        """
        pruned = 0
        for existing in folder.glob("*.md"):
            if existing.name not in written_names:
                existing.unlink()
                pruned += 1
        return pruned

    # ── Rendering ──

    def _render_node(
        self,
        label: str,
        node: dict[str, Any],
        node_id: str,
        edges: dict[str, list[dict]],
    ) -> str:
        """Render a single node as Markdown with frontmatter."""
        name = self._display_name(label, node)
        node_edges = edges.get(node_id, [])
        fm_props = self._frontmatter_props(label, node)
        if self._typed_links:
            fm_props.update(self._typed_link_props(node_edges, set(fm_props)))
        sections: list[str] = [_frontmatter(fm_props), f"# {name}", ""]

        # Body varies by node type
        if label == "Concept":
            if node.get("description"):
                sections.append(node["description"])
                sections.append("")
        elif label == "Equation":
            if node.get("canonical_latex"):
                sections.append(f"$${node['canonical_latex']}$$")
                sections.append("")
            if node.get("variable_signature"):
                sections.append("**Variables:** " + ", ".join(node["variable_signature"]))
                sections.append("")
            if node.get("assumptions"):
                sections.append("**Assumptions:** " + ", ".join(node["assumptions"]))
                sections.append("")
        elif label == "Principle":
            if node.get("statement"):
                sections.append(node["statement"])
                sections.append("")
        elif label == "CircuitTopology":
            if node.get("function"):
                sections.append(f"**Function:** {node['function']}")
            if node.get("transistor_count") is not None:
                sections.append(f"**Transistor count:** {node['transistor_count']}")
            if node.get("key_nodes"):
                sections.append("**Key nodes:** " + ", ".join(node["key_nodes"]))
            sections.append("")
        elif label == "Parameter":
            parts = []
            if node.get("symbol"):
                parts.append(f"**Symbol:** {node['symbol']}")
            if node.get("units"):
                parts.append(f"**Units:** {node['units']}")
            if node.get("typical_range"):
                parts.append(f"**Typical range:** {node['typical_range']}")
            sections.extend(parts)
            sections.append("")
        elif label == "Insight":
            if node.get("statement"):
                sections.append(f"> {node['statement']}")
                sections.append("")
        elif label == "Source":
            if node.get("author"):
                sections.append(f"**Author:** {node['author']}")
                sections.append("")
        elif label == "ClaimCard":
            if node.get("verdict"):
                sections.append(f"**Verdict:** `{node['verdict']}`")
            meta = [f"{k} {node[k]}" for k in ("basis", "engine", "topology_class") if node.get(k)]
            if meta:
                sections.append(" · ".join(meta))
            if node.get("knob") and node.get("metric"):
                sections.append(f"**Mechanism:** {node['knob']} → {node['metric']}")
            if node.get("scope"):
                sections.append(f"**Scope:** `{node['scope']}`")
            if node.get("dominant_risk_untested"):
                sections.append(f"**Dominant untested risk:** {node['dominant_risk_untested']}")
            sections.append("")
        elif label == "Specimen":
            for k in ("topology_class", "pdk", "tool"):
                if node.get(k):
                    sections.append(f"**{k.replace('_', ' ').title()}:** {node[k]}")
            sections.append("")
        elif label == "Hypothesis":
            if node.get("statement"):
                sections.append(node["statement"])
            if node.get("status"):
                sections.append(f"\n**Status:** {node['status']}")
            sections.append("")
        elif label == "DesignDecision":
            if node.get("choice"):
                sections.append(f"**Choice:** {node['choice']}")
            if node.get("rationale"):
                sections.append(f"**Rationale:** {node['rationale']}")
            if node.get("status"):
                sections.append(f"**Status:** {node['status']}")
            sections.append("")
        elif label == "BenchResult":
            for k in ("bench_type", "metric", "corner", "conclusion"):
                if node.get(k):
                    sections.append(f"**{k.replace('_', ' ').title()}:** {node[k]}")
            sections.append("")
        elif label == "Regularity":
            if node.get("statement"):
                sections.append(node["statement"])
            status_line = f"**Status:** `{node['status']}`" if node.get("status") else ""
            if status_line and node.get("status_note"):
                status_line += f" — {node['status_note']}"
            if status_line:
                sections.append("")
                sections.append(status_line)
            if node.get("pdks"):
                sections.append("**Member PDKs:** " + ", ".join(node["pdks"]))
            sections.append("")

        # Relationships section
        if node_edges:
            sections.append("## Relationships")
            sections.append("")
            for edge in sorted(node_edges, key=lambda e: e["rel_type"]):
                link = self._wikilink_for(edge["target_id"])
                rel = edge["rel_type"].replace("_", " ").lower()
                line = f"- **{rel}** → {link}"
                if edge.get("rationale"):
                    line += f" — {edge['rationale']}"
                sections.append(line)
            sections.append("")

        return "\n".join(sections)

    def _wikilink_for(self, target_id: str) -> str:
        """Resolve a node id to its Obsidian wikilink string.

        Single choke point for target-id -> wikilink resolution (PHILOSOPHY
        P5 — enforce once): both the body "## Relationships" section and the
        frontmatter typed-link keys call this, so they can never drift onto
        different target filenames. `self._id_to_filename` is the same
        collision-safe index either way (built by `_build_filename_map`,
        id-suffixed on a filename collision) — falls back to the raw id if
        the target somehow never got indexed (e.g. it was retracted).
        """
        target_name = self._id_to_filename.get(target_id, target_id)
        return f"[[{target_name}]]"

    def _typed_link_props(
        self, node_edges: list[dict[str, Any]], existing_keys: set[str]
    ) -> dict[str, list[_WikiLink]]:
        """Group a node's outgoing edges by rel_type into frontmatter link lists.

        Key = ``rel_type.lower()`` (e.g. ``DEPENDS_ON`` -> ``depends_on``);
        value = the deduplicated (first-occurrence order preserved),
        `_WikiLink`-wrapped list of target wikilinks — resolved via
        `_wikilink_for`, the exact same target filename the body
        "## Relationships" section links to.

        A key that collides with an already-present frontmatter key (`id`,
        `type`, `confidence`, ...) is prefixed with `rel_` so it never
        clobbers that property. In the (untested-in-practice) case that two
        distinct rel_types still land on the same final key, their link
        lists are merged + deduped rather than one silently overwriting the
        other (PHILOSOPHY P9 — degrade, but never silently). Returned dict
        is ordered by key, ascending.
        """
        by_rel: dict[str, list[_WikiLink]] = {}
        for edge in node_edges:
            rel_key = edge["rel_type"].lower()
            link = _WikiLink(self._wikilink_for(edge["target_id"]))
            bucket = by_rel.setdefault(rel_key, [])
            if link not in bucket:
                bucket.append(link)

        keyed: dict[str, list[_WikiLink]] = {}
        for rel_key, links in by_rel.items():
            key = f"rel_{rel_key}" if rel_key in existing_keys else rel_key
            bucket = keyed.setdefault(key, [])
            for link in links:
                if link not in bucket:
                    bucket.append(link)

        return {key: keyed[key] for key in sorted(keyed)}

    def _frontmatter_props(self, label: str, node: dict[str, Any]) -> dict[str, Any]:
        """Select properties to include in YAML frontmatter."""
        node_id = self._get_id(label, node)
        props: dict[str, Any] = {
            "id": node_id,
            "type": label,
        }

        # Type-specific frontmatter
        if label == "Concept":
            if node.get("domain"):
                props["domain"] = node["domain"]
            if node.get("granularity"):
                props["granularity"] = node["granularity"]
        elif label == "Equation":
            if node.get("equation_type"):
                props["equation_type"] = node["equation_type"]
        elif label == "Insight":
            if node.get("bridge_type"):
                props["bridge_type"] = node["bridge_type"]
            props["reviewed"] = node.get("reviewed", False)
        elif label == "ClaimCard":
            for k in ("verdict", "basis", "engine", "topology_class"):
                if node.get(k):
                    props[k] = node[k]
        elif label in ("Hypothesis", "DesignDecision", "BenchResult"):
            if node.get("status"):
                props["status"] = node["status"]
        elif label == "Regularity":
            for k in ("status", "topology_class", "metric", "knob", "quant_kind"):
                if node.get(k):
                    props[k] = node[k]

        # Common fields
        if node.get("confidence") is not None:
            props["confidence"] = node["confidence"]
        if node.get("reinforcement_count") is not None:
            props["reinforcement_count"] = node["reinforcement_count"]

        return props

    def _write_index(
        self,
        nodes: dict[str, list[dict]],
        counts: dict[str, int],
        collisions_by_label: dict[str, int] | None = None,
        pruned_by_label: dict[str, int] | None = None,
    ) -> None:
        """Write a vault-level index file.

        `counts` must be the number of files actually written per label
        (not the raw node count) — retracted-node exclusion and collision
        suffixing can make the two diverge, and this index is the honesty
        surface for that.
        """
        collisions_by_label = collisions_by_label or {}
        pruned_by_label = pruned_by_label or {}
        lines = [
            "# Knowledge Graph Index",
            "",
            f"*Exported from openclaw-brain on {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            "",
            "## Summary",
            "",
        ]
        for label, count in sorted(counts.items()):
            folder = _FOLDER_MAP.get(label, label)
            lines.append(f"- **{label}**: {count} nodes → `{folder}/`")
        lines.append("")

        total_suffixed = sum(collisions_by_label.values())
        total_pruned = sum(pruned_by_label.values())
        if total_suffixed or total_pruned:
            lines.append("## Export Integrity")
            lines.append("")
            if total_suffixed:
                lines.append(
                    f"- {total_suffixed} filename collision(s) disambiguated with an id suffix:"
                )
                for label, n in sorted(collisions_by_label.items()):
                    if n:
                        lines.append(f"  - {label}: {n}")
            if total_pruned:
                lines.append(f"- {total_pruned} stale file(s) pruned (node no longer in the graph):")
                for label, n in sorted(pruned_by_label.items()):
                    if n:
                        lines.append(f"  - {label}: {n}")
            lines.append("")

        # Top concepts by reinforcement
        concepts = nodes.get("Concept", [])
        if concepts:
            top = sorted(
                concepts,
                key=lambda c: c.get("reinforcement_count", 1),
                reverse=True,
            )[:20]
            lines.append("## Top Concepts (by reinforcement)")
            lines.append("")
            for c in top:
                concept_id = self._get_id("Concept", c)
                default_name = _snake_to_title(c.get("canonical_name", c.get("concept_id", "?")))
                # Resolve through the collision-safe filename index so this link
                # matches the file actually on disk (same rule as edge wikilinks).
                fname = self._id_to_filename.get(concept_id, _sanitize_filename(default_name))
                rc = c.get("reinforcement_count", 1)
                conf = c.get("confidence", 0)
                lines.append(f"- [[{fname}]] (reinforced ×{rc}, confidence {conf:.2f})")
            lines.append("")

        self._vault.mkdir(parents=True, exist_ok=True)
        (self._vault / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")

    # ── Helpers ──

    @staticmethod
    def _get_id(label: str, node: dict[str, Any]) -> str:
        """Extract the ID field from a node based on its label."""
        field = _ID_FIELDS.get(label, "id")
        return str(node.get(field, node.get("id", "")))

    @staticmethod
    def _display_name(label: str, node: dict[str, Any]) -> str:
        """Human-readable name for a node."""
        if label == "Concept":
            name = node.get("canonical_name", node.get("concept_id", "Untitled"))
            return _snake_to_title(name)
        if label == "Equation":
            return node.get("canonical_latex", node.get("equation_id", "Untitled"))[:80]
        if label == "Principle":
            return node.get("name", node.get("principle_id", "Untitled"))
        if label == "CircuitTopology":
            return node.get("name", node.get("topology_id", "Untitled"))
        if label == "Parameter":
            name = node.get("name", "")
            symbol = node.get("symbol", "")
            if name and symbol:
                return f"{name} ({symbol})"
            return name or symbol or node.get("parameter_id", "Untitled")
        if label == "Source":
            return node.get("title", node.get("source_id", "Untitled"))
        if label == "Insight":
            stmt = node.get("statement", "")
            return stmt[:80] if stmt else node.get("insight_id", "Untitled")
        if label == "Assumption":
            stmt = node.get("statement", "")
            return stmt[:80] if stmt else node.get("assumption_id", "Untitled")
        if label == "ClaimCard":
            cid = str(node.get("claim_id", "Untitled"))
            parts = cid.split(":")            # projected as "<spec_id = sha256:hash>:<card_id>"
            if len(parts) >= 3 and parts[0] == "sha256":
                return f"{parts[-1]} ({parts[1][:8]})"   # readable: "ota5t_av0 (088e3b66)"
            return cid
        if label == "Specimen":
            tc = node.get("topology_class") or ""
            suf = str(node.get("spec_id", "")).split(":")[-1][:8]
            return f"{tc} ({suf})" if (tc and suf) else (tc or node.get("spec_id") or "Untitled")
        if label == "Hypothesis":
            stmt = (node.get("statement") or "")[:60].strip()
            hid = node.get("hypothesis_id", "")
            return f"{stmt} ({hid})" if stmt else (hid or "Untitled")
        if label == "DesignDecision":
            choice = (node.get("choice") or "")[:60].strip()
            did = node.get("decision_id", "")
            return f"{choice} ({did})" if choice else (did or "Untitled")
        if label == "BenchResult":
            metric, corner = node.get("metric") or "", node.get("corner") or ""
            bid = node.get("bench_id", "")
            head = f"{metric} @ {corner}" if (metric and corner) else (
                metric or (node.get("conclusion") or "")[:50] or "bench")
            return f"{head} ({bid})" if bid else head
        if label == "Regularity":
            tc = node.get("topology_class") or ""
            metric, knob = node.get("metric") or "", node.get("knob") or ""
            lid = str(node.get("law_id", ""))[:8]
            if tc and metric and knob:
                return f"{tc}: {metric} vs {knob} ({lid})"
            return (node.get("statement") or "")[:80] or lid or "Untitled"
        return str(node.get("id", "Untitled"))
