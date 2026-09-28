"""Tests for the specimen -> schematic (SVG) renderer (knowledge/executable/schematic.py).

Offline where possible (synthetic cell.spice/meta.yaml fixtures for the parser, self-check, the
TOTAL-guard, and the cache) — plus a smoke pass over the 3 REAL pilot specimens already committed
under corpus/specimens/ (repo-local files, no simulator/docker/Neo4j needed), verifying the SVG is
valid XML and carries the expected device labels.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from openclaw_brain.knowledge.executable.schematic import (
    SchematicMismatchError,
    UnsupportedTopologyError,
    ParsedCell,
    is_supported,
    parse_cell_spice,
    render_specimen,
    supported_topologies,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS = REPO_ROOT / "corpus" / "specimens"

# The 3 pilot specimens (sky130A), already committed to the repo's corpus/ SSOT.
PILOT_SPECIMENS = {
    "common_source_active_load_nmos": "e426c2f5f9647540",
    "cascode_current_mirror_nmos": "416a98c03326721a",
    "diff_pair_resistive_nmos": "12727727068b9da6",
}
_CS_FULL_SPEC_ID = "sha256:e426c2f5f9647540cc6ae6b35e0e5a502880167ce9fc4c49a90999021a425489"

_CS_CELL_SPICE = """\
* cell common_source_active_load_nmos
.lib "__LIBPATH__" tt
.param VDD=1.8 Wn=8 Wp=16 Lp=0.5 IREFV=10u CL=1p
VDD vdd 0 {VDD}
XM3 pbias pbias vdd vdd sky130_fd_pr__pfet_01v8 W={Wp} L={Lp}
XM2 out   pbias vdd vdd sky130_fd_pr__pfet_01v8 W={Wp} L={Lp}
XM1 out   g 0 0 sky130_fd_pr__nfet_01v8 W={Wn} L={Lp}
IREF pbias 0 {IREFV}
Lfb out g 1T
Vin in 0 DC 0 AC 1
Cin in g 1T
CL out 0 {CL}
.end
"""


def _write_specimen(tmp_path: Path, topology_class: str, spec_id: str, cell_spice: str) -> Path:
    spec_dir = tmp_path / "specimens" / topology_class / spec_id.split(":", 1)[-1][:16]
    spec_dir.mkdir(parents=True)
    (spec_dir / "cell.spice").write_text(cell_spice)
    (spec_dir / "meta.yaml").write_text(yaml.safe_dump({
        "spec_id": spec_id, "topology_class": topology_class,
        "pdk": "sky130A", "tool": "ngspice-46", "role_map": {},
    }))
    return spec_dir


# ── parser ──────────────────────────────────────────────────────────────────────────────────


def test_parse_cell_spice_params():
    p = parse_cell_spice(_CS_CELL_SPICE)
    assert p.params == {
        "VDD": "1.8", "Wn": "8", "Wp": "16", "Lp": "0.5", "IREFV": "10u", "CL": "1p",
    }


def test_parse_cell_spice_devices():
    p = parse_cell_spice(_CS_CELL_SPICE)
    assert set(p.devices) == {"VDD", "XM3", "XM2", "XM1", "IREF", "Lfb", "Vin", "Cin", "CL"}
    assert p.devices["XM1"].nodes == ("out", "g", "0", "0")
    assert p.devices["XM3"].nodes == ("pbias", "pbias", "vdd", "vdd")
    assert p.devices["XM3"].model == "sky130_fd_pr__pfet_01v8"
    assert p.devices["IREF"].nodes == ("pbias", "0")
    assert p.devices["CL"].nodes == ("out", "0")


def test_parse_cell_spice_ignores_directives_and_comments():
    text = "* a comment\n.lib \"x\" tt\n.param VDD=1.8\n.include \"y\"\nVDD vdd 0 {VDD}\n.end\n"
    p = parse_cell_spice(text)
    assert p.params == {"VDD": "1.8"}
    assert set(p.devices) == {"VDD"}


def test_param_helper_default():
    p = ParsedCell(params={"VDD": "1.8"})
    assert p.param("VDD") == "1.8"
    assert p.param("MISSING") == "?"
    assert p.param("MISSING", default="0") == "0"


# ── TOTAL-guard: unsupported topology_class ────────────────────────────────────────────────


def test_is_supported_and_registry():
    assert is_supported("common_source_active_load_nmos")
    assert is_supported("cascode_current_mirror_nmos")
    assert is_supported("diff_pair_resistive_nmos")
    assert not is_supported("folded_cascode_ota_nmos_in")
    assert supported_topologies() == {
        "common_source_active_load_nmos", "cascode_current_mirror_nmos", "diff_pair_resistive_nmos",
    }


def test_render_specimen_raises_for_unsupported_topology(tmp_path):
    spec_dir = _write_specimen(tmp_path, "folded_cascode_ota_nmos_in", "sha256:" + "ab" * 32,
                                "* not even parsed\n.end\n")
    with pytest.raises(UnsupportedTopologyError, match="folded_cascode_ota_nmos_in"):
        render_specimen(spec_dir, tmp_path / "out")
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").glob("*.svg"))


# ── connectivity self-check ─────────────────────────────────────────────────────────────────


def test_render_specimen_valid_cell_renders(tmp_path):
    spec_dir = _write_specimen(tmp_path, "common_source_active_load_nmos",
                                "sha256:" + "11" * 32, _CS_CELL_SPICE)
    out_dir = tmp_path / "out"
    path = render_specimen(spec_dir, out_dir)
    assert path.is_file()
    assert path.suffix == ".svg"
    svg_text = path.read_text()
    assert svg_text.startswith("<svg") or "<svg" in svg_text[:200]
    ET.fromstring(svg_text)  # valid XML
    for expected_label in ("M1", "M2", "M3", "VDD"):
        assert expected_label in svg_text


def test_render_specimen_rejects_mismatched_node(tmp_path):
    """A node rewired (XM1's source 0 -> gndx) must refuse to render — the connectivity
    self-check catching exactly the defect class the module docstring names (a topology_class
    string routed to a hand-drawn picture the real netlist doesn't match)."""
    bad_cell = _CS_CELL_SPICE.replace(
        "XM1 out   g 0 0 sky130_fd_pr__nfet_01v8", "XM1 out   g gndx 0 sky130_fd_pr__nfet_01v8"
    )
    spec_dir = _write_specimen(tmp_path, "common_source_active_load_nmos",
                                "sha256:" + "22" * 32, bad_cell)
    with pytest.raises(SchematicMismatchError, match="XM1"):
        render_specimen(spec_dir, tmp_path / "out")
    assert not list((tmp_path / "out").glob("*.svg"))


def test_render_specimen_rejects_missing_device(tmp_path):
    """A dropped device line (CL entirely missing) must refuse to render, not draw a picture
    missing the load capacitor."""
    bad_cell = "\n".join(
        line for line in _CS_CELL_SPICE.splitlines() if not line.startswith("CL ")
    ) + "\n"
    spec_dir = _write_specimen(tmp_path, "common_source_active_load_nmos",
                                "sha256:" + "33" * 32, bad_cell)
    with pytest.raises(SchematicMismatchError, match="missing device"):
        render_specimen(spec_dir, tmp_path / "out")


def test_render_specimen_rejects_extra_device(tmp_path):
    """An extra, unexpected device line must also refuse — the check is exact-set, not subset."""
    bad_cell = _CS_CELL_SPICE.replace(".end\n", "RX1 out 0 1k\n.end\n")
    spec_dir = _write_specimen(tmp_path, "common_source_active_load_nmos",
                                "sha256:" + "44" * 32, bad_cell)
    with pytest.raises(SchematicMismatchError, match="unexpected device"):
        render_specimen(spec_dir, tmp_path / "out")


def test_render_specimen_wrong_topology_class_with_foreign_netlist_is_rejected(tmp_path):
    """The specific incident-shaped case named in the module docstring: a specimen whose
    meta.yaml CLAIMS topology_class=diff_pair_resistive_nmos but whose cell.spice is actually a
    common_source_active_load_nmos netlist. The self-check must refuse, not silently draw the
    diff-pair template over a netlist that isn't one."""
    spec_dir = _write_specimen(tmp_path, "diff_pair_resistive_nmos",
                                "sha256:" + "55" * 32, _CS_CELL_SPICE)
    with pytest.raises(SchematicMismatchError):
        render_specimen(spec_dir, tmp_path / "out")


# ── cache ───────────────────────────────────────────────────────────────────────────────────


def test_render_specimen_caches_by_output_filename(tmp_path):
    spec_dir = _write_specimen(tmp_path, "common_source_active_load_nmos",
                                "sha256:" + "66" * 32, _CS_CELL_SPICE)
    out_dir = tmp_path / "out"
    path1 = render_specimen(spec_dir, out_dir)
    original_bytes = path1.read_bytes()

    # Corrupt the source cell.spice so a re-parse/re-render would fail the self-check. A cache
    # hit must never touch it — same spec_id (same filename) can only ever mean byte-identical
    # netlist content, so skipping re-parse on a hit is safe by construction.
    (spec_dir / "cell.spice").write_text("* corrupted\n.end\n")

    path2 = render_specimen(spec_dir, out_dir)
    assert path2 == path1
    assert path2.read_bytes() == original_bytes  # untouched, not re-rendered


def test_render_specimen_output_filename_is_short_spec_id(tmp_path):
    spec_dir = _write_specimen(tmp_path, "common_source_active_load_nmos",
                                "sha256:" + "77" * 32, _CS_CELL_SPICE)
    path = render_specimen(spec_dir, tmp_path / "out")
    # style is part of the cache key: a mos_style switch must never serve
    # the other style's cached picture
    assert path.name == ("77" * 32)[:16] + ".razavi.svg"


# ── real corpus smoke (3 pilot specimens, repo-local files) ──────────────────────────────────


@pytest.mark.parametrize("topology_class,short_hash", sorted(PILOT_SPECIMENS.items()))
def test_pilot_specimen_renders_from_real_corpus(tmp_path, topology_class, short_hash):
    spec_dir = CORPUS / topology_class / short_hash
    if not spec_dir.is_dir():
        pytest.skip(f"pilot specimen not present in this checkout: {spec_dir}")
    path = render_specimen(spec_dir, tmp_path / "out")
    assert path.is_file()
    svg_text = path.read_text()
    root = ET.fromstring(svg_text)  # raises on invalid XML
    assert root.tag.endswith("svg")
    assert "M1" in svg_text
    assert "VDD" in svg_text


def test_pilot_specimen_cache_skips_reparse_on_second_call(tmp_path):
    spec_dir = CORPUS / "diff_pair_resistive_nmos" / PILOT_SPECIMENS["diff_pair_resistive_nmos"]
    if not spec_dir.is_dir():
        pytest.skip(f"pilot specimen not present in this checkout: {spec_dir}")
    out_dir = tmp_path / "out"
    p1 = render_specimen(spec_dir, out_dir)
    p2 = render_specimen(spec_dir, out_dir)
    assert p1 == p2


# ── agent.why() attachment (real pilot specimen, mocked graph — no Neo4j) ────────────────────


class _MockGraph:
    """Mirrors test_executable_mcp.py's mock — a fixed row set, read-only."""

    def __init__(self, rows):
        self._rows = rows

    async def run_read_query(self, query, params=None):
        return self._rows


@pytest.mark.asyncio
async def test_why_attaches_schematic_path_for_real_pilot_specimen(tmp_path):
    """End-to-end (minus Neo4j, which is mocked): why() on a claim-card whose Specimen is a REAL,
    repo-committed common_source_active_load_nmos specimen gets a schematic_path pointing at a
    real, freshly-rendered SVG under state_dir/schematics/."""
    if not (CORPUS / "common_source_active_load_nmos" /
            PILOT_SPECIMENS["common_source_active_load_nmos"]).is_dir():
        pytest.skip("pilot specimen not present in this checkout")
    from openclaw_brain.agent import BrainAgent
    from openclaw_brain.config import load_config

    cfg = load_config()
    cfg.openclaw.state_dir = str(tmp_path)   # state_path is derived from this; keeps writes local
    agent = BrainAgent(cfg)
    agent._started = True
    agent._graph = _MockGraph(rows=[{
        "claim": "cs_gbw", "knob": "Iref", "metric": "gbw_hz", "verdict": "VERIFIED",
        "quant_kind": "direction", "narrative": "GBW rises with Iref",
        "corner": "tt", "temp_c": 27.0, "vdd": 1.8,
        "engine": "ngspice", "basis": "physical-nominal",
        "scope": '{"device": "nominal", "pdk": "sky130", '
                 '"corners": ["sky130/tt/27/1.8"], "statistical": "none"}',
        "dominant_risk_untested": None,
        "spec_id": _CS_FULL_SPEC_ID, "topology_class": "common_source_active_load_nmos",
        "grounds": None, "laws": [],
    }])

    out = await agent.why(f"{_CS_FULL_SPEC_ID}:cs_gbw")

    assert out["found"] is True
    assert out["verdict"] == "VERIFIED"
    assert "schematic_path" in out
    svg_path = Path(out["schematic_path"])
    assert svg_path.is_file()
    assert svg_path.suffix == ".svg"
    assert str(tmp_path) in str(svg_path)   # rendered under the (test-local) state_dir, not corpus/
    ET.fromstring(svg_path.read_text())      # valid XML


@pytest.mark.asyncio
async def test_why_omits_schematic_path_for_unsupported_topology(tmp_path):
    """No template registered (18 of 19 families) -> field is absent, not an error."""
    from openclaw_brain.agent import BrainAgent
    from openclaw_brain.config import load_config

    cfg = load_config()
    cfg.openclaw.state_dir = str(tmp_path)
    agent = BrainAgent(cfg)
    agent._started = True
    agent._graph = _MockGraph(rows=[{
        "claim": "cc_gbw", "knob": "Cc", "metric": "gbw_hz", "verdict": "VERIFIED",
        "quant_kind": "direction", "narrative": None,
        "corner": "tt", "temp_c": 27.0, "vdd": 1.8,
        "engine": "ngspice", "basis": "physical-nominal",
        "scope": '{"device": "nominal", "corners": ["tt/27/1.8"], "statistical": "none"}',
        "dominant_risk_untested": None,
        "spec_id": "sha256:abc", "topology_class": "miller_ota_2stage_nmos_in",
        "grounds": None, "laws": [],
    }])

    out = await agent.why("sha256:abc:cc_gbw")

    assert out["found"] is True
    assert "schematic_path" not in out


def test_render_specimen_mos_style_ieee_and_unknown(tmp_path):
    """ieee style renders to its own cache file; unknown style is a loud ValueError."""
    spec_dir = _write_specimen(tmp_path, "common_source_active_load_nmos",
                               "sha256:" + "88" * 32, _CS_CELL_SPICE)
    p_ieee = render_specimen(spec_dir, tmp_path / "out", mos_style="ieee")
    assert p_ieee.name.endswith(".ieee.svg")
    p_razavi = render_specimen(spec_dir, tmp_path / "out")
    assert p_razavi != p_ieee and p_razavi.exists() and p_ieee.exists()
    import pytest as _pytest
    with _pytest.raises(ValueError, match="unknown mos_style"):
        render_specimen(spec_dir, tmp_path / "out", mos_style="sedra")
