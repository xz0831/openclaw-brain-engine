"""Integrity regression tests for the Obsidian exporter.

Covers three verified defects found against the live ~/Semiconductor vault:

(a) Filename-collision silent loss — distinct node ids (e.g. Parameter
    symbols 'gm', 'W', 'L') sanitize to the same filename and silently
    overwrite each other (2,286 Parameter nodes -> 1,430 files observed).
(b) Retracted nodes exported as normal notes — no `retracted` filter,
    unlike graph/store.py's search queries.
(c) No stale-file pruning + dishonest counts — merged/retracted nodes leave
    orphan .md files forever, and counts reported pre-collision node counts
    as "exported" instead of files actually written.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from openclaw_brain.export.obsidian import ObsidianExporter, _short_id_suffix
from tests.test_obsidian_export import _assert_frontmatter_list, _extract_frontmatter


@pytest.fixture
def vault_dir(tmp_path):
    return tmp_path / "vault"


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.uri = "bolt://localhost:7687"
    cfg.user = "neo4j"
    cfg.password = "test"
    cfg.database = "neo4j"
    return cfg


# ── (a) Filename-collision silent loss ──


@pytest.fixture
def colliding_parameter_nodes():
    # Four distinct parameter_ids; three share symbol 'gm' (no `name` field,
    # so _display_name falls back to the bare symbol — the real-world shape
    # of the 2,286 -> 1,430 collapse).
    return {
        "Parameter": [
            {"parameter_id": "gm_nmos_45nm_paper1", "symbol": "gm"},
            {"parameter_id": "gm_pmos_28nm_paper2", "symbol": "gm"},
            {"parameter_id": "gm_nmos_180nm_bookA", "symbol": "gm"},
            {"parameter_id": "W_channel_width_std", "symbol": "W"},
        ],
    }


class TestCollisionSuffixing:
    @pytest.mark.asyncio
    async def test_colliding_ids_each_get_a_suffix_no_overwrite(
        self, vault_dir, mock_config, colliding_parameter_nodes
    ):
        exporter = ObsidianExporter(mock_config, vault_dir)
        exporter._fetch_all_nodes = AsyncMock(return_value=colliding_parameter_nodes)
        exporter._fetch_all_edges = AsyncMock(return_value={})

        counts = await exporter.export()

        # All 3 colliding 'gm' nodes survive as separate files — none lost.
        param_files = list((vault_dir / "Parameters").glob("*.md"))
        assert len(param_files) == 4
        assert counts["Parameter"] == 4

        # Every colliding id got suffixed (not just the losers) — the bare
        # "gm.md" must not exist since ALL 3 collided ids were renamed.
        names = {f.name for f in param_files}
        assert "gm.md" not in names
        assert "W.md" in names  # no collision -> unsuffixed
        gm_names = {n for n in names if n.startswith("gm ")}
        assert len(gm_names) == 3

        stats = exporter.last_export_stats
        assert stats["collisions_suffixed"]["Parameter"] == 3

    @pytest.mark.asyncio
    async def test_suffixing_is_deterministic_regardless_of_fetch_order(
        self, vault_dir, mock_config, colliding_parameter_nodes
    ):
        forward = colliding_parameter_nodes
        reversed_nodes = {
            "Parameter": list(reversed(colliding_parameter_nodes["Parameter"]))
        }

        exp_a = ObsidianExporter(mock_config, vault_dir / "a")
        exp_a._fetch_all_nodes = AsyncMock(return_value=forward)
        exp_a._fetch_all_edges = AsyncMock(return_value={})
        await exp_a.export()

        exp_b = ObsidianExporter(mock_config, vault_dir / "b")
        exp_b._fetch_all_nodes = AsyncMock(return_value=reversed_nodes)
        exp_b._fetch_all_edges = AsyncMock(return_value={})
        await exp_b.export()

        # Same graph (just fetched/iterated in a different order) -> same
        # id->filename map and the same files on disk.
        assert exp_a._id_to_filename == exp_b._id_to_filename
        names_a = {f.name for f in (vault_dir / "a" / "Parameters").glob("*.md")}
        names_b = {f.name for f in (vault_dir / "b" / "Parameters").glob("*.md")}
        assert names_a == names_b

    def test_short_id_suffix_is_a_pure_function_of_the_id(self):
        assert _short_id_suffix("gm_nmos_45nm_paper1") == _short_id_suffix("gm_nmos_45nm_paper1")
        assert _short_id_suffix("gm_nmos_45nm_paper1") != _short_id_suffix("gm_pmos_28nm_paper2")


class TestWikilinkFilenameConsistency:
    @pytest.mark.asyncio
    async def test_edge_wikilink_resolves_to_the_suffixed_filename(
        self, vault_dir, mock_config, colliding_parameter_nodes
    ):
        nodes = dict(colliding_parameter_nodes)
        nodes["Source"] = [
            {"source_id": "razavi_analog", "title": "Design of Analog CMOS ICs"},
        ]
        edges = {
            "razavi_analog": [
                {
                    "target_id": "gm_pmos_28nm_paper2",
                    "target_label": "Parameter",
                    "rel_type": "USES_PARAMETER",
                    "rationale": "",
                    "confidence": None,
                    "reinforcement_count": None,
                },
            ],
        }

        exporter = ObsidianExporter(mock_config, vault_dir)
        exporter._fetch_all_nodes = AsyncMock(return_value=nodes)
        exporter._fetch_all_edges = AsyncMock(return_value=edges)
        await exporter.export()

        target_filename = exporter._id_to_filename["gm_pmos_28nm_paper2"]
        assert target_filename != "gm"  # it was suffixed

        source_body = (vault_dir / "Sources" / "Design of Analog CMOS ICs.md").read_text()
        assert f"[[{target_filename}]]" in source_body

        # And the linked file actually exists at that exact name.
        assert (vault_dir / "Parameters" / f"{target_filename}.md").is_file()

    @pytest.mark.asyncio
    async def test_frontmatter_typed_link_resolves_to_the_suffixed_filename(
        self, vault_dir, mock_config, colliding_parameter_nodes
    ):
        """Same setup as the body-link test above, but for the frontmatter
        typed-link key -- it must resolve through the exact same
        `_id_to_filename` choke point, so a collision-suffixed target is
        never a place the two (body vs. frontmatter) disagree."""
        nodes = dict(colliding_parameter_nodes)
        nodes["Source"] = [
            {"source_id": "razavi_analog", "title": "Design of Analog CMOS ICs"},
        ]
        edges = {
            "razavi_analog": [
                {
                    "target_id": "gm_pmos_28nm_paper2",
                    "target_label": "Parameter",
                    "rel_type": "USES_PARAMETER",
                    "rationale": "",
                    "confidence": None,
                    "reinforcement_count": None,
                },
            ],
        }

        exporter = ObsidianExporter(mock_config, vault_dir)
        exporter._fetch_all_nodes = AsyncMock(return_value=nodes)
        exporter._fetch_all_edges = AsyncMock(return_value=edges)
        await exporter.export()

        target_filename = exporter._id_to_filename["gm_pmos_28nm_paper2"]
        assert target_filename != "gm"  # confirms the collision suffix actually applied

        source_body = (vault_dir / "Sources" / "Design of Analog CMOS ICs.md").read_text()
        fm_text = _extract_frontmatter(source_body)

        _assert_frontmatter_list(fm_text, "uses_parameter", [f"[[{target_filename}]]"])
        assert f"[[{target_filename}]]" in source_body  # body "## Relationships" line

        assert (vault_dir / "Parameters" / f"{target_filename}.md").is_file()

    @pytest.mark.asyncio
    async def test_top_concepts_index_link_resolves_to_suffixed_filename(
        self, vault_dir, mock_config
    ):
        # Two distinct concept_ids whose canonical_name sanitizes identically.
        nodes = {
            "Concept": [
                {
                    "concept_id": "vth_paper_a",
                    "canonical_name": "Vth",
                    "reinforcement_count": 5,
                    "confidence": 0.9,
                },
                {
                    "concept_id": "vth_paper_b",
                    "canonical_name": "Vth",
                    "reinforcement_count": 1,
                    "confidence": 0.5,
                },
            ],
        }
        exporter = ObsidianExporter(mock_config, vault_dir)
        exporter._fetch_all_nodes = AsyncMock(return_value=nodes)
        exporter._fetch_all_edges = AsyncMock(return_value={})
        await exporter.export()

        top_filename = exporter._id_to_filename["vth_paper_a"]
        assert top_filename != "Vth"  # suffixed due to collision

        index = (vault_dir / "INDEX.md").read_text()
        assert f"[[{top_filename}]]" in index
        assert (vault_dir / "Concepts" / f"{top_filename}.md").is_file()


# ── (b) Retracted-node exclusion ──


class TestRetractedExclusion:
    @pytest.mark.asyncio
    async def test_retracted_node_is_not_exported(self, vault_dir, mock_config):
        nodes = {
            "Concept": [
                {"concept_id": "live_concept", "canonical_name": "Live Concept"},
                {
                    "concept_id": "dead_concept",
                    "canonical_name": "Dead Concept",
                    "retracted": True,
                },
            ],
        }
        exporter = ObsidianExporter(mock_config, vault_dir)
        exporter._fetch_all_nodes = AsyncMock(return_value=nodes)
        exporter._fetch_all_edges = AsyncMock(return_value={})

        counts = await exporter.export()

        assert counts["Concept"] == 1
        assert (vault_dir / "Concepts" / "Live Concept.md").is_file()
        assert not (vault_dir / "Concepts" / "Dead Concept.md").exists()
        assert "dead_concept" not in exporter._id_to_filename

    @pytest.mark.asyncio
    async def test_all_nodes_retracted_leaves_empty_but_present_folder(
        self, vault_dir, mock_config
    ):
        nodes = {
            "Concept": [
                {"concept_id": "c1", "canonical_name": "One", "retracted": True},
            ],
        }
        exporter = ObsidianExporter(mock_config, vault_dir)
        exporter._fetch_all_nodes = AsyncMock(return_value=nodes)
        exporter._fetch_all_edges = AsyncMock(return_value={})

        counts = await exporter.export()

        assert counts["Concept"] == 0
        assert list((vault_dir / "Concepts").glob("*.md")) == []


# ── (c) Stale-file pruning + honest counts ──


class TestPruneAndHonestCounts:
    @pytest.mark.asyncio
    async def test_prunes_stale_file_within_label_folder_only(
        self, vault_dir, mock_config
    ):
        # Pre-populate the vault as if from a previous export run.
        (vault_dir / "Concepts").mkdir(parents=True)
        (vault_dir / "Concepts" / "Merged Away.md").write_text("stale", encoding="utf-8")
        (vault_dir / "Attachments").mkdir(parents=True)
        (vault_dir / "Attachments" / "diagram.md").write_text("user note", encoding="utf-8")
        (vault_dir / "My Root Note.md").write_text("user note", encoding="utf-8")
        (vault_dir / "INDEX.md").write_text("old index", encoding="utf-8")

        nodes = {
            "Concept": [
                {"concept_id": "survivor", "canonical_name": "Survivor"},
            ],
        }
        exporter = ObsidianExporter(mock_config, vault_dir)
        exporter._fetch_all_nodes = AsyncMock(return_value=nodes)
        exporter._fetch_all_edges = AsyncMock(return_value={})

        await exporter.export()

        # Stale file inside the managed Concepts folder is gone.
        assert not (vault_dir / "Concepts" / "Merged Away.md").exists()
        assert (vault_dir / "Concepts" / "Survivor.md").is_file()

        # Nothing outside the managed label folder was touched.
        assert (vault_dir / "Attachments" / "diagram.md").is_file()
        assert (vault_dir / "My Root Note.md").is_file()
        assert (vault_dir / "INDEX.md").is_file()  # rewritten, not deleted

        stats = exporter.last_export_stats
        assert stats["pruned"]["Concept"] == 1

    @pytest.mark.asyncio
    async def test_counts_equal_actual_files_written_not_raw_node_count(
        self, vault_dir, mock_config
    ):
        nodes = {
            "Concept": [
                {"concept_id": "live1", "canonical_name": "Live One"},
                {"concept_id": "live2", "canonical_name": "Live Two"},
                {"concept_id": "dead1", "canonical_name": "Dead One", "retracted": True},
            ],
        }
        exporter = ObsidianExporter(mock_config, vault_dir)
        exporter._fetch_all_nodes = AsyncMock(return_value=nodes)
        exporter._fetch_all_edges = AsyncMock(return_value={})

        counts = await exporter.export()

        # 3 raw nodes fetched, but only 2 survive retraction -> honest count is 2.
        assert counts["Concept"] == 2
        actual_files = list((vault_dir / "Concepts").glob("*.md"))
        assert counts["Concept"] == len(actual_files)

        index = (vault_dir / "INDEX.md").read_text()
        assert "**Concept**: 2 nodes" in index

    @pytest.mark.asyncio
    async def test_prunes_folder_for_label_absent_from_fetch_result(
        self, vault_dir, mock_config
    ):
        # Mirrors the real Cypher's aggregation semantics: `_fetch_all_nodes`
        # filters retracted rows in its WHERE clause *before* collect(), so a
        # label whose every node is retracted emits NO row — the label key
        # is entirely absent from the fetch result, not present with an
        # empty list. A prior export run left files in that label's folder;
        # they must still be pruned even though the label key is gone this
        # time (otherwise the folder's stale notes persist forever and
        # INDEX.md silently drops the label).
        (vault_dir / "Circuits").mkdir(parents=True)
        (vault_dir / "Circuits" / "Extinct Topology.md").write_text(
            "stale", encoding="utf-8"
        )

        exporter = ObsidianExporter(mock_config, vault_dir)
        # No "CircuitTopology" key at all in the fetch result.
        exporter._fetch_all_nodes = AsyncMock(
            return_value={
                "Concept": [{"concept_id": "c1", "canonical_name": "Survivor"}],
            }
        )
        exporter._fetch_all_edges = AsyncMock(return_value={})

        counts = await exporter.export()

        assert list((vault_dir / "Circuits").glob("*.md")) == []
        assert counts.get("CircuitTopology") == 0

        stats = exporter.last_export_stats
        assert stats["pruned"]["CircuitTopology"] == 1

    @pytest.mark.asyncio
    async def test_never_populated_label_folder_is_left_uncreated(
        self, vault_dir, mock_config
    ):
        # A label that never had a folder written for it (fresh vault) must
        # not be created just to prune nothing -- only labels with an
        # existing folder on disk get visited by the fallback prune step.
        exporter = ObsidianExporter(mock_config, vault_dir)
        exporter._fetch_all_nodes = AsyncMock(
            return_value={
                "Concept": [{"concept_id": "c1", "canonical_name": "Survivor"}],
            }
        )
        exporter._fetch_all_edges = AsyncMock(return_value={})

        counts = await exporter.export()

        assert not (vault_dir / "Circuits").exists()
        assert "CircuitTopology" not in counts

    @pytest.mark.asyncio
    async def test_second_export_run_prunes_files_from_nodes_removed_since_first_run(
        self, vault_dir, mock_config
    ):
        exporter = ObsidianExporter(mock_config, vault_dir)

        first_nodes = {
            "Concept": [
                {"concept_id": "c1", "canonical_name": "One"},
                {"concept_id": "c2", "canonical_name": "Two"},
            ],
        }
        exporter._fetch_all_nodes = AsyncMock(return_value=first_nodes)
        exporter._fetch_all_edges = AsyncMock(return_value={})
        await exporter.export()
        assert len(list((vault_dir / "Concepts").glob("*.md"))) == 2

        # c2 is gone from the graph on the next run (merged/retracted upstream).
        second_nodes = {
            "Concept": [
                {"concept_id": "c1", "canonical_name": "One"},
            ],
        }
        exporter._fetch_all_nodes = AsyncMock(return_value=second_nodes)
        counts = await exporter.export()

        assert counts["Concept"] == 1
        remaining = list((vault_dir / "Concepts").glob("*.md"))
        assert len(remaining) == 1
        assert remaining[0].name == "One.md"


class TestCaseInsensitiveCollision:
    @pytest.mark.asyncio
    async def test_case_only_name_difference_treated_as_collision(
        self, vault_dir, mock_config
    ):
        """The live vault sits on APFS (case-insensitive): 'GM.md' and
        'gm.md' are the SAME file, so names differing only in case must be
        suffixed like any other collision or one node silently overwrites
        the other."""
        nodes = {
            "Parameter": [
                {"parameter_id": "gm_lowercase_paperA", "symbol": "gm"},
                {"parameter_id": "GM_uppercase_paperB", "symbol": "GM"},
            ],
        }
        exporter = ObsidianExporter(mock_config, vault_dir)
        exporter._fetch_all_nodes = AsyncMock(return_value=nodes)
        exporter._fetch_all_edges = AsyncMock(return_value={})

        counts = await exporter.export()

        param_files = list((vault_dir / "Parameters").glob("*.md"))
        assert len(param_files) == 2
        assert counts["Parameter"] == 2
        names = {f.name for f in param_files}
        # both collided -> both suffixed; the bare single-case names are gone
        assert "gm.md" not in names and "GM.md" not in names
        # suffixed names remain distinct even when casefolded (suffix differs)
        assert len({n.casefold() for n in names}) == 2
        assert exporter.last_export_stats["collisions_suffixed"]["Parameter"] == 2
