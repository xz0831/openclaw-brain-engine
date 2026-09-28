"""specimen -> circuit-diagram (SVG) renderer — the "why() shows the schematic" pilot.

`why()` gives an engineer the oracle-certified verdict, but a claim-card cites a `spec_id`, not a
picture. This module turns a corpus specimen's `cell.spice` into a hand-authored, textbook-style
SVG schematic (schemdraw 0.23, a pure-SVG backend — **never import matplotlib** here) so the
teaching surface can show the circuit a claim was measured on, not just name it.

Two invariants carried over from this package's own conventions (see EPISTEMOLOGY.md/README.md):

1. **TOTAL-function guard, not silent generic fallback** (mirrors `conditions.py::summarize_scope`,
   which raises `UnknownConditionsKind` on an unhandled `Conditions.kind` rather than defaulting to
   a nullable PVT shape). `render_specimen` raises `UnsupportedTopologyError` for any
   `topology_class` with no registered template — it never falls back to a generic auto-layout.
   A generic renderer would silently draw *a* circuit, not *the* circuit; for a teaching surface
   that is worse than refusing.
2. **Connectivity self-check before every render** (`_check_connectivity`). The specimen's
   `topology_class` (a plain string field on `meta.yaml`) is trusted nowhere else in this module —
   it only selects a template. Before that template's hand-drawn devices are trusted to represent
   the netlist, the parsed `cell.spice` device inventory (ref -> node tuple) is compared against
   the template's own declared expectation. A mismatch raises `SchematicMismatchError` and refuses
   to render. This is the same defect *shape* PHILOSOPHY.md P5 names for the main ingest pipeline
   (an unverified identity field trusted by many write paths let one topology's provenance overwrite
   another's) and the same shape this package's own README documents as "Known risk area #1" for
   `corpus.py`'s `spec_id` hash (two different `topology_class` registrations whose netlists ever
   coincide would silently MERGE onto one Neo4j node). A schematic renderer sits exactly on that
   same seam — `topology_class` selects a hand-drawn picture from a string a corpus write path
   supplied — so it gets its own explicit, tested guard rather than inheriting trust from upstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import schemdraw
import schemdraw.elements as elm
import yaml

_PARAM_LINE_RE = re.compile(r"^\s*\.param\s+(.*)$", re.IGNORECASE)
_PARAM_KV_RE = re.compile(r"(\w+)\s*=\s*(\S+)")

# SPICE element-type prefixes this parser understands, and how many leading tokens (after the ref
# designator) are NODE tokens (as opposed to model-name / value tokens). `X` (subckt instance —
# every MOSFET in this corpus is instantiated as a 4-terminal `X<ref> d g s b <model> W=.. L=..`
# subckt call, never a bare `M` device line) gets 4; two-terminal primitives (current/voltage
# sources, R/L/C) get 2. Anything else (`.lib`, `.include`, `.end`, comments, unrecognized
# prefixes) is skipped — this parser is deliberately narrow to the corpus's actual netlist style,
# not a general SPICE parser.
_NODE_COUNT_BY_KIND = {"X": 4, "I": 2, "V": 2, "R": 2, "L": 2, "C": 2}


class UnsupportedTopologyError(Exception):
    """No schematic template is registered for this topology_class (the TOTAL-guard firing).

    Not a bug — most of the corpus's 19 families have no hand-authored layout yet (pilot scope:
    3 of 19). Callers (agent.why(), the CLI) treat this as "omit the schematic," never as an error
    to surface loudly — but this module itself never silently substitutes a generic drawing for it.
    """


class SchematicMismatchError(Exception):
    """The parsed cell.spice device inventory does not match the template's declared expectation.

    Raised by `_check_connectivity` before any drawing happens. See the module docstring's
    invariant (2) for why this exists: a `topology_class` string is the only thing that routes a
    specimen to a hand-drawn picture, and nothing upstream of this module re-verifies it against
    the actual netlist. Never caught-and-ignored inside this module — a mismatch always refuses to
    render rather than emitting a picture that doesn't match the simulated circuit.
    """


@dataclass(frozen=True)
class DeviceLine:
    """One parsed SPICE element line: its ref designator and the *node* tokens (never the model
    name or W/L value tokens — those legitimately vary per PDK/specimen; nodes are the
    connectivity signature that must not)."""

    ref: str
    kind: str
    nodes: tuple[str, ...]
    model: str = ""


@dataclass(frozen=True)
class ParsedCell:
    """The result of parsing one `cell.spice`: `.param` values (raw strings, unit suffix intact —
    e.g. `"10u"`, `"1.8"`) and every recognized device line, keyed by ref designator."""

    params: dict[str, str] = field(default_factory=dict)
    devices: dict[str, DeviceLine] = field(default_factory=dict)

    def param(self, name: str, default: str = "?") -> str:
        return self.params.get(name, default)


def parse_cell_spice(text: str) -> ParsedCell:
    """Parse a specimen's `cell.spice` into `.param` values + device connectivity.

    Deliberately narrow (see `_NODE_COUNT_BY_KIND`): this corpus's cell decks are machine-rendered
    by `templates.py` in one consistent style (one `.param` line, then `X`-subckt MOSFETs and
    `I`/`V`/`R`/`L`/`C` two-terminal primitives, one instance per line). It is not a general SPICE
    parser and does not need to be — the connectivity self-check only needs to catch a
    topology_class/netlist mismatch, not parse arbitrary decks.
    """
    params: dict[str, str] = {}
    devices: dict[str, DeviceLine] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("*"):
            continue
        m = _PARAM_LINE_RE.match(line)
        if m:
            for k, v in _PARAM_KV_RE.findall(m.group(1)):
                params[k] = v
            continue
        if line.startswith("."):
            continue  # .lib / .include / .end / other directives — not connectivity
        tokens = line.split()
        ref = tokens[0]
        kind = ref[0].upper()
        n_nodes = _NODE_COUNT_BY_KIND.get(kind)
        if n_nodes is None or len(tokens) < 1 + n_nodes:
            continue  # unrecognized element type / malformed line — not fatal, just not tracked
        nodes = tuple(tokens[1 : 1 + n_nodes])
        model = tokens[1 + n_nodes] if kind == "X" and len(tokens) > 1 + n_nodes else ""
        devices[ref] = DeviceLine(ref=ref, kind=kind, nodes=nodes, model=model)
    return ParsedCell(params=params, devices=devices)


# ── MOSFET symbol style ─────────────────────────────────────────────────────
# "razavi": minimal textbook symbols (schemdraw NFet/PFet — no source arrow, no bulk
#   terminal, PFet carries the gate bubble): the house style of Razavi's "Design of
#   Analog CMOS Integrated Circuits" and the operator's stated preference (2026-07-20).
# "ieee": schemdraw's detailed NMos/PMos (segmented channel + source arrow).
# The .theta(0) intrinsic-vertical quirk documented at the template sites applies to
# BOTH styles. Style flows through a module-level active pair set (and restored) by
# render_specimen(); rendering is synchronous and single-call today — if renders ever
# run concurrently, thread a parameter through the templates instead.
_MOS_STYLES: dict[str, tuple[type, type]] = {
    "razavi": (elm.NFet, elm.PFet),
    "ieee": (elm.NMos, elm.PMos),
}
_active_mos: tuple[type, type] = _MOS_STYLES["razavi"]


def _nmos():
    return _active_mos[0]()


def _pmos():
    return _active_mos[1]()


def _short_hash(spec_id: str) -> str:
    """The same short-hash form `corpus.SpecimenCorpus._dir` uses for its on-disk directory name
    (16 hex chars after the `sha256:` prefix) — reused here so a schematic's filename is
    recognizably the same specimen as its corpus directory, without importing corpus.py's private
    path helper into a rendering-only module."""
    return spec_id.split(":", 1)[-1][:16]


# ── connectivity self-check ──────────────────────────────────────────────────────────────────


def _check_connectivity(topology_class: str, expected: dict[str, tuple[str, ...]],
                         parsed: ParsedCell) -> None:
    """Compare the parsed device inventory against a template's declared expectation. TOTAL over
    the expected ref set: every expected ref must be present with an EXACT node tuple, and no
    unexpected device may be present either (a netlist with an extra/missing device is not "the"
    circuit this template draws, even if every declared ref happens to match)."""
    got_refs = set(parsed.devices)
    want_refs = set(expected)
    missing = want_refs - got_refs
    extra = got_refs - want_refs
    mismatched = {
        ref: (expected[ref], parsed.devices[ref].nodes)
        for ref in (want_refs & got_refs)
        if parsed.devices[ref].nodes != expected[ref]
    }
    if missing or extra or mismatched:
        parts = [f"schematic connectivity self-check failed for topology_class={topology_class!r}:"]
        if missing:
            parts.append(f"  missing device(s): {sorted(missing)}")
        if extra:
            parts.append(f"  unexpected device(s): {sorted(extra)}")
        for ref, (want, got) in sorted(mismatched.items()):
            parts.append(f"  {ref}: expected nodes {want}, got {got}")
        raise SchematicMismatchError("\n".join(parts))


# ── template registry ────────────────────────────────────────────────────────────────────────

RenderFn = Callable[[ParsedCell], "schemdraw.Drawing"]


@dataclass(frozen=True)
class TemplateSpec:
    topology_class: str
    expected_devices: dict[str, tuple[str, ...]]
    render: RenderFn


def _drawing() -> schemdraw.Drawing:
    d = schemdraw.Drawing()
    d.config(unit=2.0, fontsize=11, lw=1.5)
    return d


# ── template 1: common_source_active_load_nmos ──────────────────────────────────────────────
#
# cell.spice (all 3 PDKs, node names stable):
#   VDD  vdd 0 {VDD}
#   XM3  pbias pbias vdd vdd <pfet>  W={Wp} L={Lp}     # diode-connected PMOS current-mirror ref
#   XM2  out   pbias vdd vdd <pfet>  W={Wp} L={Lp}     # PMOS active load
#   XM1  out   g     0   0   <nfet>  W={Wn} L={Lp}     # NMOS common-source input device
#   IREF pbias 0 {IREFV}
#   Lfb  out g 1T                                       # AC-open feedback (self-bias)
#   Vin  in 0 DC 0 AC 1
#   Cin  in g 1T                                         # DC-block into the gate
#   CL   out 0 {CL}
#
# Textbook layout: PMOS mirror (diode ref + active load) hangs from VDD; NMOS input device below
# the load, output at the shared node between them; CL at the output; AC stimulus (Vin/Cin) and
# the self-bias feedback (Lfb) drawn as satellite branches so the picture stays faithful to what
# was actually simulated, not just the DUT's three transistors.

_CS_EXPECTED: dict[str, tuple[str, ...]] = {
    "VDD": ("vdd", "0"),
    "XM3": ("pbias", "pbias", "vdd", "vdd"),
    "XM2": ("out", "pbias", "vdd", "vdd"),
    "XM1": ("out", "g", "0", "0"),
    "IREF": ("pbias", "0"),
    "Lfb": ("out", "g"),
    "Vin": ("in", "0"),
    "Cin": ("in", "g"),
    "CL": ("out", "0"),
}


def _render_common_source_active_load_nmos(p: ParsedCell) -> schemdraw.Drawing:
    d = _drawing()
    rail_y = 8.0
    col_a, col_b = 2.0, 6.0

    d += elm.Line().at((0, rail_y)).to((9, rail_y))
    d += elm.Dot().at((col_a, rail_y))
    d += elm.Dot().at((col_b, rail_y))
    d += elm.Label().label(f"VDD = {p.param('VDD')}").at((4.3, rail_y + 0.4))

    m3 = d.add(_pmos().theta(0).anchor("source").at((col_a, rail_y))
               .label(f"M3\nW={p.param('Wp')} L={p.param('Lp')}", loc="left"))
    d += elm.Dot().at(m3.drain).label("pbias", loc="bottom")
    d += elm.Wire("-|").at(m3.gate).to(m3.drain)  # diode-connect: gate tied to drain

    iref = d.add(elm.SourceI().down().at(m3.drain).label(f"IREF\n{p.param('IREFV')}", loc="right"))
    d += elm.Ground().at(iref.end)

    m2 = d.add(_pmos().theta(0).anchor("source").at((col_b, rail_y))
               .label(f"M2\nW={p.param('Wp')} L={p.param('Lp')}", loc="right"))
    d += elm.Line().at(m3.gate).to(m2.gate)  # shared pbias gate bus (same y by construction)

    m1 = d.add(_nmos().theta(0).anchor("drain").at(m2.drain)
               .label(f"M1\nW={p.param('Wn')} L={p.param('Lp')}", loc="right"))
    d += elm.Dot().at(m1.drain).label("out", loc="left")
    d += elm.Ground().at(m1.source)

    d += elm.Line().right().at(m1.drain).length(1.2)
    cl = d.add(elm.Capacitor().down().label(f"CL\n{p.param('CL')}", loc="right"))
    d += elm.Ground().at(cl.end)

    # AC stimulus into the gate: Vin -> Cin -> g, routed on its own row well below both the IREF
    # branch's and M1's grounds so it never crosses column A's devices/labels.
    stim_y = min(iref.end[1], m1.source[1]) - 2.0
    g_drop = (m1.gate[0], stim_y)
    d += elm.Line().at(m1.gate).to(g_drop).label("g", loc="top")
    d += elm.Line().left().at(g_drop).length(1.0)
    d += elm.Capacitor().left().label("Cin", loc="top")
    vin = d.add(elm.SourceV().left().label("Vin", loc="bottom"))
    d += elm.Ground().at(vin.end)

    # Self-bias feedback: Lfb from out back to g, routed above the rail
    top_y = rail_y + 2.2
    out_top = (m1.drain[0], top_y)
    g_top = (m1.gate[0], top_y)
    d += elm.Line().at(m1.drain).to(out_top)
    d += elm.Inductor().at(out_top).to(g_top).label("Lfb", loc="top")
    d += elm.Line().at(g_top).to(m1.gate)

    return d


# ── template 2: cascode_current_mirror_nmos ─────────────────────────────────────────────────
#
# cell.spice:
#   VDD  vdd 0 {VDD}
#   IREF vdd nc {IREFV}
#   XM2  nc nc nb 0 <nfet> W={W} L={Lp}     # top device, reference stack (diode-connected)
#   XM1  nb nb 0  0 <nfet> W={W} L={Lp}     # bottom device, reference stack (diode-connected)
#   XM4  out nc nx 0 <nfet> W={W} L={Lp}    # top device, output (mirror) stack — gate <- nc
#   XM3  nx nb 0  0 <nfet> W={W} L={Lp}     # bottom device, output (mirror) stack — gate <- nb
#   Vout out 0 {VOUT}
#
# Textbook layout: two matched NMOS stacks (cascode "upward" — bottom device, then top device on
# top of it), reference stack on the left (self-biased via IREF, both devices diode-connected),
# output/mirror stack on the right with gates cross-wired from the reference stack; a DC test
# source (Vout) sweeps the output-compliance node.

_CASCODE_EXPECTED: dict[str, tuple[str, ...]] = {
    "VDD": ("vdd", "0"),
    "IREF": ("vdd", "nc"),
    "XM2": ("nc", "nc", "nb", "0"),
    "XM1": ("nb", "nb", "0", "0"),
    "XM4": ("out", "nc", "nx", "0"),
    "XM3": ("nx", "nb", "0", "0"),
    "Vout": ("out", "0"),
}


def _render_cascode_current_mirror_nmos(p: ParsedCell) -> schemdraw.Drawing:
    d = _drawing()
    rail_y = 8.0
    col_a, col_b = 2.0, 6.0
    label = f"W={p.param('W')} L={p.param('Lp')}"

    d += elm.Line().at((0, rail_y)).to((4, rail_y))
    d += elm.Dot().at((col_a, rail_y))
    d += elm.Label().label(f"VDD = {p.param('VDD')}").at((0.3, rail_y + 0.4))

    # NOTE: `.down()` rotates the loc frame — for a `.down()` 2-terminal element, loc="right"
    # renders near its START (here: the rail end), not visually-right. Used deliberately here to
    # keep the IREF value away from the "nc" tap label at the opposite (end) side.
    iref = d.add(elm.SourceI().down().at((col_a, rail_y)).label(f"IREF\n{p.param('IREFV')}", loc="right"))
    d += elm.Dot().at(iref.end)
    d += elm.Label().label("nc").at((iref.end[0] + 0.5, iref.end[1] - 0.1))

    m2 = d.add(_nmos().theta(0).anchor("drain").at(iref.end).label(f"M2\n{label}", loc="left"))
    d += elm.Wire("-|").at(m2.gate).to(m2.drain)  # diode-connect
    d += elm.Dot().at(m2.source).label("nb", loc="right")

    m1 = d.add(_nmos().theta(0).anchor("drain").at(m2.source).label(f"M1\n{label}", loc="left"))
    d += elm.Wire("-|").at(m1.gate).to(m1.drain)  # diode-connect
    d += elm.Ground().at(m1.source)

    m4 = d.add(_nmos().theta(0).anchor("drain").at((col_b, m2.drain[1]))
               .label(f"M4\n{label}", loc="right", ofst=(0.45, 0.4)))
    d += elm.Line().at(m2.gate).to(m4.gate)  # nc gate bus (same y by construction)
    d += elm.Dot().at(m4.drain).label("out", loc="right")

    m3 = d.add(_nmos().theta(0).anchor("drain").at((col_b, m1.drain[1]))
               .label(f"M3\n{label}", loc="right", ofst=(0.45, 0.4)))
    d += elm.Line().at(m1.gate).to(m3.gate)  # nb gate bus (same y by construction)
    d += elm.Ground().at(m3.source)

    d += elm.Line().right().at(m4.drain).length(2.6)
    vout = d.add(elm.SourceV().down().reverse().label(f"Vout\n{p.param('VOUT')}", loc="right"))
    d += elm.Ground().at(vout.end)

    return d


# ── template 3: diff_pair_resistive_nmos ────────────────────────────────────────────────────
#
# cell.spice:
#   VDD  vdd 0 {VDD}
#   RD1  vdd o1 {RD}
#   RD2  vdd o2 {RD}
#   XM1  o1 vinp tail 0 <nfet> W={Wn} L={Lp}
#   XM2  o2 vinn tail 0 <nfet> W={Wn} L={Lp}
#   Itail tail 0 {IB}
#   Vinp vinp 0 DC {VCM} AC 0.5
#   Vinn vinn 0 DC {VCM} AC -0.5
#
# Textbook layout: left-right symmetric differential pair — matched RD1/RD2 loads from VDD,
# matched M1/M2 inputs sharing a common tail node, tail current sink to ground, differential
# stimulus (Vinp/Vinn around VCM) driving the two gates from outside the pair.

_DIFFPAIR_EXPECTED: dict[str, tuple[str, ...]] = {
    "VDD": ("vdd", "0"),
    "RD1": ("vdd", "o1"),
    "RD2": ("vdd", "o2"),
    "XM1": ("o1", "vinp", "tail", "0"),
    "XM2": ("o2", "vinn", "tail", "0"),
    "Itail": ("tail", "0"),
    "Vinp": ("vinp", "0"),
    "Vinn": ("vinn", "0"),
}


def _render_diff_pair_resistive_nmos(p: ParsedCell) -> schemdraw.Drawing:
    d = _drawing()
    rail_y = 8.0
    col_a, col_b = 2.0, 6.0

    d += elm.Line().at((0, rail_y)).to((8, rail_y))
    d += elm.Dot().at((col_a, rail_y))
    d += elm.Dot().at((col_b, rail_y))
    d += elm.Label().label(f"VDD = {p.param('VDD')}").at((3.3, rail_y + 0.4))

    # NOTE: `.down()` rotates the loc frame — loc="top"/"bottom" land at true center-left/
    # center-right for a `.down()` element (not top/bottom); loc="left"/"right" would instead land
    # at the end/start points, colliding with the o1/o2 dot labels there. See the cascode
    # template's IREF label for the same quirk.
    rd1 = d.add(elm.Resistor().down().at((col_a, rail_y)).length(3.4)
                .label(f"RD1\n{p.param('RD')}", loc="top"))
    rd2 = d.add(elm.Resistor().down().at((col_b, rail_y)).length(3.4)
                .label(f"RD2\n{p.param('RD')}", loc="bottom"))
    d += elm.Dot().at(rd1.end).label("o1", loc="bottom")
    d += elm.Dot().at(rd2.end).label("o2", loc="bottom")

    m1 = d.add(_nmos().theta(0).anchor("drain").at(rd1.end)
               .label(f"M1\nW={p.param('Wn')} L={p.param('Lp')}", loc="left", ofst=(-0.2, 0.55)))
    m2 = d.add(_nmos().theta(0).anchor("drain").at(rd2.end)
               .label(f"M2\nW={p.param('Wn')} L={p.param('Lp')}", loc="right", ofst=(0.2, 0.55)))
    d += elm.Line().at(m1.source).to(m2.source)  # shared tail node (same y by construction)
    d += elm.Dot().at(((m1.source[0] + m2.source[0]) / 2, m1.source[1])).label("tail", loc="bottom")

    itail = d.add(elm.SourceI().down().at(((m1.source[0] + m2.source[0]) / 2, m1.source[1]))
                  .label(f"Itail\n{p.param('IB')}", loc="right"))
    d += elm.Ground().at(itail.end)

    # (no separate "vinp"/"vinn" node-name label here — it would sit right on top of M1/M2's own
    # label at the gate; the adjacent Vinp/Vinn source label already names the net unambiguously)
    d += elm.Line().left().at(m1.gate).length(2.8)
    vinp = d.add(elm.SourceV().left().label(f"Vinp\nVCM={p.param('VCM')}", loc="bottom"))
    d += elm.Ground().at(vinp.end)

    d += elm.Line().right().at(m2.gate).length(2.8)
    vinn = d.add(elm.SourceV().right().label(f"Vinn\nVCM={p.param('VCM')}", loc="bottom"))
    d += elm.Ground().at(vinn.end)

    return d


_TEMPLATES: dict[str, TemplateSpec] = {
    "common_source_active_load_nmos": TemplateSpec(
        "common_source_active_load_nmos", _CS_EXPECTED, _render_common_source_active_load_nmos,
    ),
    "cascode_current_mirror_nmos": TemplateSpec(
        "cascode_current_mirror_nmos", _CASCODE_EXPECTED, _render_cascode_current_mirror_nmos,
    ),
    "diff_pair_resistive_nmos": TemplateSpec(
        "diff_pair_resistive_nmos", _DIFFPAIR_EXPECTED, _render_diff_pair_resistive_nmos,
    ),
}


def is_supported(topology_class: str) -> bool:
    return topology_class in _TEMPLATES


def supported_topologies() -> frozenset[str]:
    return frozenset(_TEMPLATES)


# ── entry point ───────────────────────────────────────────────────────────────────────────────


def render_specimen(spec_dir: Path, out_dir: Path, mos_style: str = "razavi") -> Path:
    """Render one corpus specimen (`<spec_dir>/{meta.yaml,cell.spice}`) to
    `<out_dir>/<spec_id short-hash>.svg`. Cached: if the target file already exists, returns it
    without re-parsing or re-rendering (safe because `spec_id` is a content hash of the netlist —
    an identical filename can only ever come from byte-identical netlist content, so a cache hit
    can never be stale). Raises `UnsupportedTopologyError` for an unregistered `topology_class` and
    `SchematicMismatchError` if the netlist doesn't match the template's declared connectivity —
    never silently degrades to a generic or partial drawing.
    """
    meta = yaml.safe_load((spec_dir / "meta.yaml").read_text())
    topology_class = meta["topology_class"]
    spec_id = meta["spec_id"]

    template = _TEMPLATES.get(topology_class)
    if template is None:
        raise UnsupportedTopologyError(
            f"no schematic template registered for topology_class={topology_class!r} "
            f"(supported: {sorted(_TEMPLATES)})"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    if mos_style not in _MOS_STYLES:
        raise ValueError(f"unknown mos_style {mos_style!r}; known: {sorted(_MOS_STYLES)}")
    # style is part of the cache key — a style switch must never serve the other style's picture
    out_path = out_dir / f"{_short_hash(spec_id)}.{mos_style}.svg"
    if out_path.is_file():
        return out_path

    cell_text = (spec_dir / "cell.spice").read_text()
    parsed = parse_cell_spice(cell_text)
    _check_connectivity(topology_class, template.expected_devices, parsed)

    global _active_mos
    prev = _active_mos
    _active_mos = _MOS_STYLES[mos_style]
    try:
        drawing = template.render(parsed)
    finally:
        _active_mos = prev
    svg_bytes = drawing.get_imagedata("svg")
    out_path.write_bytes(svg_bytes)
    return out_path
