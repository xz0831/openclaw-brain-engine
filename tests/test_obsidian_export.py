"""Tests for Obsidian vault export."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openclaw_brain.export.obsidian import (
    ObsidianExporter,
    _FOLDER_MAP,
    _ID_FIELDS,
    _MANAGED_LABELS,
    _frontmatter,
    _sanitize_filename,
    _snake_to_title,
)


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


def test_sanitize_filename():
    assert _sanitize_filename("MOSFET I-V Characteristics") == "MOSFET I-V Characteristics"
    assert _sanitize_filename('a/b\\c:d"e') == "a_b_c_d_e"
    assert _sanitize_filename("  lots   of   spaces  ") == "lots of spaces"
    assert _sanitize_filename("") == "untitled"


def test_regularity_label_registration():
    """Law-tier graph representation (spec 2026-07-04) — the 9-site checklist's Obsidian
    registration site: folder + id-field entries for the new Regularity label."""
    assert _FOLDER_MAP["Regularity"] == "Regularities"
    assert _ID_FIELDS["Regularity"] == "law_id"
    assert "Regularity" in _MANAGED_LABELS
    assert ObsidianExporter._get_id("Regularity", {"law_id": "abc123"}) == "abc123"


def test_snake_to_title():
    assert _snake_to_title("fd_soi") == "FD SOI"
    assert _snake_to_title("common_source_amplifier") == "Common Source Amplifier"
    assert _snake_to_title("cmos_inverter") == "CMOS Inverter"
    assert _snake_to_title("mosfet_threshold") == "MOSFET Threshold"
    # Already Title Case (has spaces) — no change
    assert _snake_to_title("Threshold Voltage") == "Threshold Voltage"
    # No underscores — no change
    assert _snake_to_title("MOSFET") == "MOSFET"
    # Mixed abbreviations
    assert _snake_to_title("pll_vco_design") == "PLL VCO Design"


@pytest.fixture
def sample_nodes():
    return {
        "Concept": [
            {
                "concept_id": "mosfet_threshold_voltage",
                "canonical_name": "Threshold Voltage",
                "description": "The gate voltage at which a MOSFET begins to conduct.",
                "domain": "semiconductor",
                "granularity": "atomic",
                "confidence": 0.85,
                "reinforcement_count": 3,
            },
            {
                "concept_id": "channel_length_modulation",
                "canonical_name": "Channel Length Modulation",
                "description": "Shortening of effective channel length at high VDS.",
                "domain": "semiconductor",
                "granularity": "atomic",
                "confidence": 0.7,
                "reinforcement_count": 1,
            },
        ],
        "Equation": [
            {
                "equation_id": "mosfet_ids_saturation",
                "canonical_latex": r"I_D = \frac{1}{2} \mu_n C_{ox} \frac{W}{L} (V_{GS} - V_{th})^2",
                "equation_type": "derived",
                "variable_signature": ["I_D", "mu_n", "C_ox", "W", "L", "V_GS", "V_th"],
                "assumptions": ["saturation region", "long-channel"],
                "confidence": 0.9,
            },
        ],
        "Source": [
            {
                "source_id": "razavi_analog",
                "title": "Design of Analog CMOS Integrated Circuits",
                "author": "Behzad Razavi",
            },
        ],
    }


@pytest.fixture
def sample_edges():
    return {
        "mosfet_threshold_voltage": [
            {
                "target_id": "mosfet_ids_saturation",
                "target_label": "Equation",
                "rel_type": "USES_EQUATION",
                "rationale": "V_th appears in the saturation current equation",
                "confidence": 0.8,
                "reinforcement_count": 2,
            },
            {
                "target_id": "channel_length_modulation",
                "target_label": "Concept",
                "rel_type": "DEPENDS_ON",
                "rationale": "CLM modifies the saturation current which depends on V_th",
                "confidence": 0.6,
                "reinforcement_count": 1,
            },
        ],
    }


class TestObsidianExporter:
    @pytest.mark.asyncio
    async def test_export_creates_folders_and_files(
        self, vault_dir, mock_config, sample_nodes, sample_edges
    ):
        exporter = ObsidianExporter(mock_config, vault_dir)
        exporter._fetch_all_nodes = AsyncMock(return_value=sample_nodes)
        exporter._fetch_all_edges = AsyncMock(return_value=sample_edges)

        counts = await exporter.export()

        # Check counts
        assert counts["Concept"] == 2
        assert counts["Equation"] == 1
        assert counts["Source"] == 1

        # Check folders exist
        assert (vault_dir / "Concepts").is_dir()
        assert (vault_dir / "Equations").is_dir()
        assert (vault_dir / "Sources").is_dir()

        # Check files exist
        assert (vault_dir / "Concepts" / "Threshold Voltage.md").is_file()
        assert (vault_dir / "Concepts" / "Channel Length Modulation.md").is_file()
        assert (vault_dir / "Sources" / "Design of Analog CMOS Integrated Circuits.md").is_file()

        # Check index
        index = (vault_dir / "INDEX.md").read_text()
        assert "Knowledge Graph Index" in index
        assert "Concept" in index

    @pytest.mark.asyncio
    async def test_concept_file_content(
        self, vault_dir, mock_config, sample_nodes, sample_edges
    ):
        exporter = ObsidianExporter(mock_config, vault_dir)
        exporter._fetch_all_nodes = AsyncMock(return_value=sample_nodes)
        exporter._fetch_all_edges = AsyncMock(return_value=sample_edges)

        await exporter.export()

        content = (vault_dir / "Concepts" / "Threshold Voltage.md").read_text()

        # Frontmatter
        assert "id: mosfet_threshold_voltage" in content
        assert "type: Concept" in content
        assert "domain: semiconductor" in content
        assert "confidence: 0.85" in content

        # Body
        assert "The gate voltage at which a MOSFET begins to conduct." in content

        # Relationships (wikilinks)
        assert "[[" in content
        assert "uses equation" in content.lower()
        assert "depends on" in content.lower()

    @pytest.mark.asyncio
    async def test_equation_file_content(
        self, vault_dir, mock_config, sample_nodes, sample_edges
    ):
        exporter = ObsidianExporter(mock_config, vault_dir)
        exporter._fetch_all_nodes = AsyncMock(return_value=sample_nodes)
        exporter._fetch_all_edges = AsyncMock(return_value=sample_edges)

        await exporter.export()

        # Equation filename comes from canonical_latex (truncated)
        eq_files = list((vault_dir / "Equations").glob("*.md"))
        assert len(eq_files) == 1

        content = eq_files[0].read_text()
        assert "$$" in content  # LaTeX block
        assert "equation_type: derived" in content
        assert "Variables" in content
        assert "Assumptions" in content

    @pytest.mark.asyncio
    async def test_empty_graph(self, vault_dir, mock_config):
        exporter = ObsidianExporter(mock_config, vault_dir)
        exporter._fetch_all_nodes = AsyncMock(return_value={})
        exporter._fetch_all_edges = AsyncMock(return_value={})

        counts = await exporter.export()
        assert counts == {}
        assert (vault_dir / "INDEX.md").is_file()

    @pytest.mark.asyncio
    async def test_index_top_concepts(
        self, vault_dir, mock_config, sample_nodes, sample_edges
    ):
        exporter = ObsidianExporter(mock_config, vault_dir)
        exporter._fetch_all_nodes = AsyncMock(return_value=sample_nodes)
        exporter._fetch_all_edges = AsyncMock(return_value=sample_edges)

        await exporter.export()

        index = (vault_dir / "INDEX.md").read_text()
        assert "Top Concepts" in index
        assert "Threshold Voltage" in index
        # Higher reinforcement should come first
        lines = index.split("\n")
        vth_line = next(l for l in lines if "Threshold Voltage" in l)
        assert "×3" in vth_line

    def test_display_name_variants(self, mock_config, vault_dir):
        exporter = ObsidianExporter(mock_config, vault_dir)

        assert exporter._display_name("Concept", {"canonical_name": "MOSFET"}) == "MOSFET"
        assert exporter._display_name("Equation", {"canonical_latex": "E=mc^2"}) == "E=mc^2"
        assert exporter._display_name("Parameter", {"name": "Mobility", "symbol": "μ"}) == "Mobility (μ)"
        assert exporter._display_name("Parameter", {"symbol": "μ"}) == "μ"
        assert exporter._display_name("Source", {"title": "Razavi"}) == "Razavi"

    def test_display_name_and_body_regularity(self, mock_config, vault_dir):
        # A projected law must never render as "Untitled": title carries the semantic key
        # (topology: metric vs knob + law_id suffix), body carries statement/status/members.
        exporter = ObsidianExporter(mock_config, vault_dir)
        node = {
            "law_id": "04dcee1d" + "0" * 32,
            "topology_class": "ota_5t_nmos_in", "metric": "gbw_hz", "knob": "CL",
            "quant_kind": "direction", "status": "law",
            "statement": "For ota_5t_nmos_in, gbw_hz decreases with CL — replicated across ...",
            "pdks": ["gf180mcuD", "ihp-sg13g2", "sky130A"],
        }
        assert exporter._display_name("Regularity", node) == \
            "ota_5t_nmos_in: gbw_hz vs CL (04dcee1d)"
        body = exporter._render_node("Regularity", node, node["law_id"], edges={})
        assert "gbw_hz decreases with CL" in body
        assert "**Status:** `law`" in body
        assert "gf180mcuD, ihp-sg13g2, sky130A" in body
        props = exporter._frontmatter_props("Regularity", node)
        assert props["status"] == "law" and props["quant_kind"] == "direction"
        # statement-only fallback (no topology fields) still never yields "Untitled"
        assert exporter._display_name(
            "Regularity", {"law_id": "ab" * 20, "statement": "s" * 100}) == "s" * 80

    def test_display_name_snake_case_conversion(self, mock_config, vault_dir):
        exporter = ObsidianExporter(mock_config, vault_dir)
        # snake_case IDs should be converted to Title Case
        assert exporter._display_name("Concept", {"canonical_name": "fd_soi"}) == "FD SOI"
        assert exporter._display_name("Concept", {"canonical_name": "common_source_amplifier"}) == "Common Source Amplifier"
        assert exporter._display_name("Concept", {"canonical_name": "cmos_inverter"}) == "CMOS Inverter"
        # Already human-readable should be unchanged
        assert exporter._display_name("Concept", {"canonical_name": "Threshold Voltage"}) == "Threshold Voltage"

    def test_frontmatter_props(self, mock_config, vault_dir):
        exporter = ObsidianExporter(mock_config, vault_dir)
        props = exporter._frontmatter_props("Concept", {
            "concept_id": "test",
            "domain": "semiconductor",
            "granularity": "atomic",
            "confidence": 0.85,
            "reinforcement_count": 3,
        })
        assert props["id"] == "test"
        assert props["type"] == "Concept"
        assert props["domain"] == "semiconductor"
        assert props["confidence"] == 0.85


# ── Executable-substrate + design-reasoning labels ──
# Regression: these labels carry claim_id/spec_id/hypothesis_id/decision_id/bench_id (NOT `id`), so
# _get_id/_display_name fell to ""/"Untitled" and EVERY node of a label overwrote one "Untitled.md".


@pytest.fixture
def executable_nodes():
    return {
        "ClaimCard": [
            {"claim_id": "ota5t_vos", "topology_class": "ota_5t_nmos_in", "verdict": "VERIFIED",
             "basis": "physical-nominal", "engine": "ngspice",
             "scope": '{"corners": ["sky130/tt_mm/27/1.8"], "statistical": "3sigma@200"}',
             "dominant_risk_untested": "ff-cold corner untested"},
            {"claim_id": "ota5t_pelgrom", "topology_class": "ota_5t_nmos_in",
             "verdict": "VERIFIED_WITH_CAVEAT", "basis": "physical-nominal", "engine": "ngspice",
             "scope": '{"pelgrom": "areas@3"}'},
        ],
        "Specimen": [
            {"spec_id": "sha256:aaaaaaaa1111", "topology_class": "ota_5t_ac", "pdk": "sky130A", "tool": "ngspice-46"},
            {"spec_id": "sha256:bbbbbbbb2222", "topology_class": "current_mirror_dc", "pdk": "sky130A", "tool": "ngspice-46"},
        ],
        "DesignDecision": [
            {"decision_id": "d1", "choice": "cascode the mirror for PSRR", "status": "active", "rationale": "raises r_out"},
        ],
        "BenchResult": [
            {"bench_id": "b1", "bench_type": "simulation", "metric": "GBW", "corner": "tt", "conclusion": "meets 30MHz"},
        ],
        "Hypothesis": [
            {"hypothesis_id": "h1", "statement": "body effect shifts Vth > 50mV at SS-cold", "status": "open"},
        ],
    }


@pytest.mark.asyncio
async def test_executable_and_design_nodes_do_not_collapse(vault_dir, mock_config, executable_nodes):
    exporter = ObsidianExporter(mock_config, vault_dir)
    exporter._fetch_all_nodes = AsyncMock(return_value=executable_nodes)
    exporter._fetch_all_edges = AsyncMock(return_value={})
    await exporter.export()

    cc_files = list((vault_dir / "ClaimCard").glob("*.md"))
    assert len(cc_files) == 2                                    # was 1 ("Untitled.md")
    assert not (vault_dir / "ClaimCard" / "Untitled.md").exists()
    assert len(list((vault_dir / "Specimen").glob("*.md"))) == 2
    assert len(list((vault_dir / "Hypothesis").glob("*.md"))) == 1
    assert len(list((vault_dir / "DesignDecision").glob("*.md"))) == 1
    assert len(list((vault_dir / "BenchResult").glob("*.md"))) == 1


@pytest.mark.asyncio
async def test_claimcard_note_surfaces_scope_honest_evidence(vault_dir, mock_config, executable_nodes):
    exporter = ObsidianExporter(mock_config, vault_dir)
    exporter._fetch_all_nodes = AsyncMock(return_value=executable_nodes)
    exporter._fetch_all_edges = AsyncMock(return_value={})
    await exporter.export()

    body = (vault_dir / "ClaimCard" / "ota5t_vos.md").read_text()
    assert "VERIFIED" in body                       # verdict
    assert "sky130" in body                         # scope tag
    assert "untested risk" in body.lower()          # dominant_risk_untested named
    assert "ngspice" in body                        # engine


def test_display_name_and_id_executable_labels(mock_config, vault_dir):
    ex = ObsidianExporter(mock_config, vault_dir)
    assert ex._get_id("ClaimCard", {"claim_id": "ota5t_vos"}) == "ota5t_vos"
    assert ex._display_name("ClaimCard", {"claim_id": "ota5t_vos"}) == "ota5t_vos"
    # real projected claim_id is "<spec_id = sha256:hash>:<card_id>" — render it readable, not the raw hash
    assert ex._display_name("ClaimCard", {"claim_id": "sha256:088e3b66aabbccdd:ota5t_av0"}) == "ota5t_av0 (088e3b66)"
    assert ex._get_id("Specimen", {"spec_id": "sha256:aaa"}) == "sha256:aaa"
    assert "ota_5t_ac" in ex._display_name("Specimen", {"spec_id": "sha256:aaa", "topology_class": "ota_5t_ac"})
    assert ex._get_id("DesignDecision", {"decision_id": "d1"}) == "d1"
    assert ex._get_id("Hypothesis", {"hypothesis_id": "h1"}) == "h1"
    assert ex._get_id("BenchResult", {"bench_id": "b1"}) == "b1"
    for label, node in [
        ("Hypothesis", {"hypothesis_id": "h1", "statement": "body effect"}),
        ("DesignDecision", {"decision_id": "d1", "choice": "use cascode"}),
        ("BenchResult", {"bench_id": "b1", "metric": "GBW", "corner": "tt"}),
    ]:
        assert ex._display_name(label, node) != "Untitled"


# ── Typed frontmatter links (rel_type -> [[wikilink]] frontmatter keys) ──

try:
    import yaml as _yaml
except ImportError:  # pragma: no cover - exercised only when PyYAML is absent
    _yaml = None


def _extract_frontmatter(content: str) -> str:
    """Return just the YAML frontmatter body, without the `---` fences."""
    lines = content.splitlines()
    assert lines[0] == "---", "note must start with a frontmatter fence"
    end = lines[1:].index("---") + 1
    return "\n".join(lines[1:end])


def _assert_frontmatter_list(fm_text: str, key: str, expected: list[str]) -> None:
    """Assert `key:` in the frontmatter YAML-parses to exactly `expected`.

    Uses PyYAML when available in this venv -- the authoritative check that
    the quoted `- "[[...]]"` items round-trip through a real YAML parser
    into plain strings, not nested sequences (a bare `- [[...]]` would
    parse as one). Falls back to a small manual unescaper (unquote, then
    `\\"` -> `"` and `\\\\` -> `\\`, matching the escape order `_frontmatter()`
    applies) when PyYAML isn't installed, verifying the same equivalence
    by hand.
    """
    if _yaml is not None:
        parsed = _yaml.safe_load(fm_text)
        assert parsed[key] == expected
        assert all(isinstance(item, str) for item in parsed[key])
        return

    lines = fm_text.splitlines()
    idx = lines.index(f"{key}:")
    items = []
    for line in lines[idx + 1:]:
        stripped = line.strip()
        if not stripped.startswith("- "):
            break
        raw = stripped[2:].strip()
        assert raw.startswith('"') and raw.endswith('"'), f"link item not quoted: {raw!r}"
        inner = raw[1:-1]
        items.append(inner.replace('\\"', '"').replace("\\\\", "\\"))
    assert items == expected


def test_frontmatter_plain_list_items_remain_unquoted_regression():
    """`_frontmatter()`'s bare list-item rendering (`- {item}`) must be
    unchanged for any plain (non-wikilink) list value -- only
    `_WikiLink`-wrapped items (added for typed frontmatter links) take the
    new quoted+escaped path. `_frontmatter_props()` doesn't currently emit
    any list-valued property itself, so this exercises `_frontmatter()`
    directly -- the actual mechanism the no-regression requirement is about."""
    text = _frontmatter({"id": "x", "tags": ["alpha", "beta"]})
    lines = text.splitlines()
    assert "tags:" in lines
    assert "  - alpha" in lines
    assert "  - beta" in lines
    assert '  - "alpha"' not in text  # no quoting introduced for plain items


class TestTypedFrontmatterLinks:
    @pytest.mark.asyncio
    async def test_typed_link_keys_grouped_sorted_and_quoted(
        self, vault_dir, mock_config, sample_nodes, sample_edges
    ):
        exporter = ObsidianExporter(mock_config, vault_dir)
        exporter._fetch_all_nodes = AsyncMock(return_value=sample_nodes)
        exporter._fetch_all_edges = AsyncMock(return_value=sample_edges)
        await exporter.export()

        content = (vault_dir / "Concepts" / "Threshold Voltage.md").read_text()
        fm_text = _extract_frontmatter(content)

        # Keys are rel_type.lower(), sorted ascending.
        assert "depends_on:" in fm_text
        assert "uses_equation:" in fm_text
        assert fm_text.index("depends_on:") < fm_text.index("uses_equation:")

        # Values match the body section's own wikilink resolution exactly.
        eq_filename = exporter._id_to_filename["mosfet_ids_saturation"]
        _assert_frontmatter_list(fm_text, "depends_on", ["[[Channel Length Modulation]]"])
        _assert_frontmatter_list(fm_text, "uses_equation", [f"[[{eq_filename}]]"])

        # List items are quoted in the raw text -- not the bare `- [[...]]`
        # form that would break YAML (nested sequence) or that any
        # pre-existing plain-string frontmatter list still uses.
        assert '  - "[[Channel Length Modulation]]"' in fm_text
        assert "  - [[Channel Length Modulation]]" not in fm_text

        # Body "## Relationships" section is untouched by any of this.
        assert f"[[{eq_filename}]]" in content
        assert "[[Channel Length Modulation]]" in content
        assert "## Relationships" in content

    @pytest.mark.asyncio
    async def test_typed_links_disabled_omits_keys_but_keeps_body_section(
        self, vault_dir, mock_config, sample_nodes, sample_edges
    ):
        exporter = ObsidianExporter(mock_config, vault_dir, typed_links=False)
        exporter._fetch_all_nodes = AsyncMock(return_value=sample_nodes)
        exporter._fetch_all_edges = AsyncMock(return_value=sample_edges)
        await exporter.export()

        content = (vault_dir / "Concepts" / "Threshold Voltage.md").read_text()
        fm_text = _extract_frontmatter(content)

        assert "depends_on" not in fm_text
        assert "uses_equation" not in fm_text
        assert "## Relationships" in content
        assert "depends on" in content.lower()

    @pytest.mark.asyncio
    async def test_duplicate_targets_deduped_within_a_rel_type(
        self, vault_dir, mock_config
    ):
        nodes = {
            "Concept": [
                {"concept_id": "c1", "canonical_name": "Alpha"},
                {"concept_id": "c2", "canonical_name": "Beta"},
            ],
        }
        # Two distinct relationship instances, same source/target/rel_type
        # (e.g. reinforced twice) -- the real shape `MATCH (a)-[r]->(b)`
        # returns one row per relationship, so duplicates like this occur.
        edges = {
            "c1": [
                {"target_id": "c2", "target_label": "Concept", "rel_type": "RELATES_TO",
                 "rationale": "first observation", "confidence": 0.5, "reinforcement_count": 1},
                {"target_id": "c2", "target_label": "Concept", "rel_type": "RELATES_TO",
                 "rationale": "reinforced again", "confidence": 0.7, "reinforcement_count": 2},
            ],
        }
        exporter = ObsidianExporter(mock_config, vault_dir)
        exporter._fetch_all_nodes = AsyncMock(return_value=nodes)
        exporter._fetch_all_edges = AsyncMock(return_value=edges)
        await exporter.export()

        content = (vault_dir / "Concepts" / "Alpha.md").read_text()
        fm_text = _extract_frontmatter(content)
        _assert_frontmatter_list(fm_text, "relates_to", ["[[Beta]]"])

    @pytest.mark.asyncio
    async def test_rel_type_colliding_with_existing_key_gets_rel_prefix(
        self, vault_dir, mock_config
    ):
        nodes = {
            "Concept": [
                {"concept_id": "c1", "canonical_name": "Alpha"},
                {"concept_id": "c2", "canonical_name": "Beta"},
            ],
        }
        edges = {
            "c1": [
                # rel_type.lower() == "id" collides with the universal `id` frontmatter key.
                {"target_id": "c2", "target_label": "Concept", "rel_type": "ID",
                 "rationale": "", "confidence": None, "reinforcement_count": None},
            ],
        }
        exporter = ObsidianExporter(mock_config, vault_dir)
        exporter._fetch_all_nodes = AsyncMock(return_value=nodes)
        exporter._fetch_all_edges = AsyncMock(return_value=edges)
        await exporter.export()

        content = (vault_dir / "Concepts" / "Alpha.md").read_text()
        fm_text = _extract_frontmatter(content)

        assert "id: c1" in fm_text  # the real id -- untouched, not clobbered
        _assert_frontmatter_list(fm_text, "rel_id", ["[[Beta]]"])

    @pytest.mark.asyncio
    async def test_escapes_quotes_and_backslashes_in_link_target(
        self, vault_dir, mock_config
    ):
        nodes = {
            "Concept": [{"concept_id": "c1", "canonical_name": "Source Concept"}],
        }
        # Target id deliberately absent from `nodes` (a dangling edge), so
        # `_wikilink_for` falls back to the RAW id -- the same fallback the
        # pre-existing body "## Relationships" section already uses
        # (`_id_to_filename.get(target_id, target_id)`), unsanitized. This is
        # the one realistic path where a `"` or `\` can reach a wikilink
        # string: any normally-indexed name is filename-sanitized first
        # (`_sanitize_filename` strips both characters), so it can never
        # carry either into a resolved link.
        raw_target_id = 'weird"target\\name'
        edges = {
            "c1": [
                {"target_id": raw_target_id, "target_label": "Concept", "rel_type": "REFERENCES",
                 "rationale": "", "confidence": None, "reinforcement_count": None},
            ],
        }
        exporter = ObsidianExporter(mock_config, vault_dir)
        exporter._fetch_all_nodes = AsyncMock(return_value=nodes)
        exporter._fetch_all_edges = AsyncMock(return_value=edges)
        await exporter.export()

        content = (vault_dir / "Concepts" / "Source Concept.md").read_text()
        fm_text = _extract_frontmatter(content)

        # Body section keeps the raw, unescaped link (pre-existing, unaffected).
        assert f"[[{raw_target_id}]]" in content
        # Frontmatter must NOT contain that same bare (unquoted) sequence --
        # confirms quoting/escaping actually happened, not a silent no-op.
        assert f"  - [[{raw_target_id}]]" not in fm_text
        assert '  - "' in fm_text

        # It round-trips through a real YAML parser back to the exact same
        # unescaped wikilink the body section uses.
        _assert_frontmatter_list(fm_text, "references", [f"[[{raw_target_id}]]"])
