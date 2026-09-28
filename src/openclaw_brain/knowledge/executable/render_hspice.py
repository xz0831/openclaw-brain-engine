"""render_hspice.py -- S4 RENDER-ONLY PrimeSim/HSPICE dialect adapter (ADR-044 D2, docs/DECISIONS.md).

D2 (revising ADR-041's undefined design-mode crossing): "'Draft simulation benches' = render-only
single-probe bench drafting (PrimeSim render-only engine slot) INSIDE teaching-mode -- buildable
now." This module is exactly that slot's render half, and ONLY that half:

    home (no PrimeSim license) renders a claim's netlist in PrimeSim/HSPICE syntax
        -> an engineer at the company runs it on PrimeSim + the company PDK
        -> the COMPARISON against home's expectation bands happens COMPANY-SIDE; company
           magnitudes do NOT come home by default (ADR-046 boundary-direction clause,
           2026-08-16 — this line previously promised a home-side importer, which the
           architecture review flagged as a boundary violation). At most a pass/fail
           direction may return; a magnitude importer would require its own ADR.

This module NEVER shells out to ngspice, docker, or primesim -- it is a pure text
transformation (ngspice-dialect deck string in, HSPICE/PrimeSim-dialect deck string out) plus a
thin, explicitly-marked READ-ONLY Neo4j reader for the live ClaimCard/Regularity fields an
EXPECTATION.md card cites. `render_hspice_deck` is a pure function: identical inputs always
produce a byte-identical deck (no timestamps, no wall-clock, no random anything) -- mirroring
runner.py's own RUN_SH byte-pinning discipline, applied to this module's one rendered artifact.

Dialect ground truth -- queried READ-ONLY from the ingested PrimeSim/HSPICE user guide
(Source.title="03_primesim_user_guide", source_id=src_6c1cf6b9c845, 3155 SourceChunks) via the
.claude/skills/verify-ingest connect pattern (GraphStore.run_read_query, CONTAINS search over
SourceChunk.text_preview). Every syntax decision below cites the chunk_id(s) it came from in
_DIALECT_CITATIONS and in the rendered deck's own header comments -- syntax CONVENTIONS only
(facts: keyword names, argument shapes), never quoted guide prose, per the task's no-vendor-
content rule (bench/primesim_pilot/ ships only our own netlists + our own measured numbers).

Where the guide chunks I queried did not settle a question (e.g. whether PrimeSim's VP() phase
function returns radians or degrees, matching ngspice, or degrees natively), this module says so
explicitly in a TODO rather than silently guessing -- see _DIALECT_CITATIONS' `todo` entries and
each MeasureTranslation.todo field, surfaced as `* TODO:` comments in the rendered deck itself.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from openclaw_brain.knowledge.executable.laws import compute_law_id
from openclaw_brain.knowledge.graph.schema import NodeLabel

# ================================================================================================
# Dialect ground truth -- chunk_id citations from src_6c1cf6b9c845 (queried 2026-07-11, read-only).
# Keyed by the dialect decision it grounds; consumed by render_hspice_deck's header comments and by
# tests/test_executable_render_hspice.py::test_dialect_citations_are_well_formed.
# ================================================================================================
_DIALECT_CITATIONS: dict[str, dict[str, Any]] = {
    "measure_when_cross": {
        "rule": ".MEASURE <AC|DC|TRAN> <result> WHEN <outvar>=<val> [CROSS=n] [TD=..] [RISE=..] "
                "[FALL=..] [PRINT=0|1] -- bare WHEN (no FIND clause) translates ngspice's "
                "'meas ac y when vdb(out)=0 cross=1' 1:1 at the token level.",
        "chunks": ["chunk_681ce6875222", "chunk_6b3374abc8bb"],
        "todo": "the guide chunks sampled confirm the FIND-outvar1-WHEN-outvar2 two-variable form; "
                "whether the bare (FIND-omitted) form returns the sweep variable (frequency, for "
                ".AC) the same way ngspice's own bare 'when' does was not independently confirmed "
                "in the chunks queried -- verify on PrimeSim, or add an explicit FIND freq clause.",
    },
    "measure_find_when": {
        "rule": ".MEASURE <AC|DC|TRAN> <result> FIND <outvar1> WHEN <outvar2>=<val> [CROSS=n] "
                "[TD=..] -- the two-variable form: measure outvar1 at the point outvar2 crosses val.",
        "chunks": ["chunk_681ce6875222"],
        "todo": None,
    },
    "measure_func_max": {
        "rule": ".MEASURE <AC|DC|TRAN> <result> <func> <outvar> [FROM=..] [TO=..] [PRINT=0|1]; "
                "func in {AVG, RMS, MIN, MAX, PP, INTEGRAL, ...} -- MAX is a documented func value.",
        "chunks": ["chunk_4daa01db05c5", "chunk_b609acd99134"],
        "todo": None,
    },
    "measure_param_expr": {
        "rule": "a .MEASURE (or .PROBE) result may be an algebraic expression via the PAR(...) "
                "keyword/PARAM= form, not just a bare node/branch variable.",
        "chunks": ["chunk_b609acd99134", "chunk_594050ae52ee"],
        "todo": None,
    },
    "probe_and_post": {
        "rule": ".PROBE <antype> V(node) / I(dev) saves listed waveforms; .OPTION POST alone "
                "already saves ALL node voltages + supply currents without .PROBE; .OPTION POST "
                "PROBE combines both (documented combined form).",
        "chunks": ["chunk_218c6b7859e3", "chunk_594050ae52ee", "chunk_a13ba4ec95f5",
                   "chunk_46d7dde94988"],
        "todo": None,
    },
    "ac_analysis": {
        "rule": ".AC [swept_param [@dev]] <type> <np> [START=start] [STOP=stop] -- positional "
                "'.AC DEC <np> <fstart> <fstop>' matches ngspice's own '.ac dec <np> <f0> <f1>' "
                "already; no dialect change beyond keyword case.",
        "chunks": ["chunk_392928920e8d", "chunk_64581f4a2983"],
        "todo": None,
    },
    "op_analysis": {
        "rule": ".OP -- bare operating-point analysis keyword (listed in the PrimeSim HSPICE-"
                "compatible command index).",
        "chunks": ["chunk_815866c60d64"],
        "todo": None,
    },
    "temp_statement": {
        "rule": ".TEMP / TEMPERATURE sets the simulation temperature; PrimeSim names .ALTER/.TEMP "
                "sweeps in its output-file naming convention (confirms .TEMP is a real, "
                "recognized sweep-capable statement).",
        "chunks": ["chunk_2e1fb8f8c39f", "chunk_255f4ca8ef3a", "chunk_26a1a69f4562"],
        "todo": "the bare single-value argument form '.TEMP <celsius>' is standard SPICE-family "
                "convention but was not matched against a direct .TEMP syntax block in the chunks "
                "queried -- low-risk, flagged for completeness.",
    },
    "lib_statement": {
        "rule": ".LIB <path> <section> includes one named section (a corner) from a foundry model "
                "library file; PDK corners are defined INSIDE that vendor file's Variation Block, "
                "selected by the .LIB call's second argument -- never invented by this renderer.",
        "chunks": ["chunk_c4782ad84e26", "chunk_edbb888dbac8", "chunk_3133ec9e9362"],
        "todo": None,
    },
    "alter_sweep": {
        "rule": ".ALTER [name] reruns the deck's analyses/measures with newly redefined .PARAM "
                "values -- 'precise control over parameter values across multiple simulation "
                "runs'; each .ALTER pass gets its own indexed output (a<alter_id>), confirming "
                ".MEASURE/.PROBE/analysis statements declared once are re-evaluated per pass "
                "rather than needing to be repeated inside each block.",
        "chunks": ["chunk_017de5db2eed", "chunk_255f4ca8ef3a", "chunk_26a1a69f4562",
                   "chunk_205f4d86d483"],
        "todo": None,
    },
    "model_and_subckt": {
        "rule": ".MODEL <name> NMOS|PMOS LEVEL=<n> <par=val ...> defines a device model; a "
                "subcircuit instance (.SUBCKT-defined or a foundry-provided device wrapper, as "
                "sky130's nfet_01v8/pfet_01v8 primitives are) is instantiated X-prefixed "
                "('prefix them with the subcircuit call name Xyyyy'), matching this deck's own "
                "XM<n> convention unchanged.",
        "chunks": ["chunk_1d5f4fdacde9", "chunk_21232827a1cd", "chunk_073b42461e63",
                   "chunk_f2c56bf476c5"],
        "todo": "if the company PDK exposes bare NMOS/PMOS .MODEL primitives instead of a "
                "subcircuit wrapper, the X-prefix + W/L-as-subckt-args shape below will need "
                "adjusting to bare M-device syntax -- see the TODO block in each rendered deck.",
    },
}


def dialect_citations() -> dict[str, dict[str, Any]]:
    """Public, read-only accessor (tests + docs use this rather than reaching into the private
    module dict directly)."""
    return _DIALECT_CITATIONS


# ================================================================================================
# Company-PDK placeholders (deliverable A's required literal tokens)
# ================================================================================================
COMPANY_PDK_LIB_PLACEHOLDER = "<COMPANY_PDK_LIB>"
CORNER_PLACEHOLDER = "<CORNER>"
NMOS_MODEL_PLACEHOLDER = "<NMOS_MODEL>"
PMOS_MODEL_PLACEHOLDER = "<PMOS_MODEL>"

# The only two device-model literals any of templates.py's analog T-template bodies use for our 3
# pilot topology classes (miller_ota_2stage_nmos_in, ota_5t_nmos_in, cascode_current_mirror_nmos --
# verified by inspection of _OTA_BODY/_OTA5T_BODY/_CASC_BODY). Substitution is exact-literal (not a
# generic 'sky130' prefix rewrite) and _substitute_device_models raises loudly if a sky130_fd_pr__
# token survives that isn't one of these two -- never silently ship home PDK residue in a company
# deck (same discipline pdks.py's BJT exact-literal trap documents for the CORE HALF).
_DEVICE_MODEL_TOKENS: dict[str, str] = {
    "sky130_fd_pr__nfet_01v8": NMOS_MODEL_PLACEHOLDER,
    "sky130_fd_pr__pfet_01v8": PMOS_MODEL_PLACEHOLDER,
}
_SKY130_TOKEN_PREFIX = "sky130_fd_pr__"


# ================================================================================================
# Deliverable A -- the dialect adapter
# ================================================================================================


@dataclass(frozen=True)
class MeasureTranslation:
    """One ngspice `let`/`meas` idiom, hand-translated to HSPICE/PrimeSim `.MEASURE` syntax and
    cited against _DIALECT_CITATIONS. Hand-curated per bench (not regex-guessed from arbitrary
    ngspice control-script text) -- the 4 pilot benches use exactly 4 measurement idioms
    (WHEN/CROSS, FIND+WHEN, func MAX, PARAM algebraic expression); see PILOT_BENCHES below."""

    ngspice_source: str
    hspice_lines: tuple[str, ...]
    citation_keys: tuple[str, ...] = ()
    extra_todo: str | None = None


@dataclass(frozen=True)
class HspiceBenchSpec:
    """Everything render_hspice_deck needs beyond the ngspice deck text itself. `sweep_param` is
    the EXACT `.param` token the swept knob resolves to in the template body -- deliberately
    explicit rather than derived from the ngspice `alter <element> = $pt` line, because the
    ngspice `alter` command targets an already-elaborated element/source by its instance name
    (e.g. cascode_current_mirror_nmos's knob "Vout" alters the *source* ref-designator "Vout"),
    which is not always spelled identically to the `.param` name the body's `{...}` braces expand
    from (that same template's symbolic sizing param is "VOUT", not "Vout") -- see PILOT_BENCHES'
    per-bench comments for the verified mapping."""

    bench_id: str
    topology_class: str
    claim_id: str
    corner: str
    temp_c: float
    sweep_param: str
    sweep_points: tuple[str, ...]
    analysis_lines: tuple[str, ...]
    measures: tuple[MeasureTranslation, ...]
    probe_lines: tuple[str, ...]
    header_notes: tuple[str, ...] = ()


_LIB_LINE_RE = re.compile(r'^\.lib\s+"([^"]*)"\s+(\S+)\s*$', re.IGNORECASE)
_PARAM_LINE_RE = re.compile(r"^\.param\s+(.*)$", re.IGNORECASE)


def _split_ngspice_deck(ngspice_deck: str) -> tuple[str, str, list[str]]:
    """Pure, mechanical extraction (no semantic guessing) of the 3 pieces of a templates.py T-
    template deck this renderer reuses verbatim: the '* title' comment, the '.param ...' body
    (kept byte-for-byte -- same keys, same values, same order, per the task's 'geometry params
    kept from the template' requirement), and the device/source connectivity lines between
    '.param' and the first '.control'/'.end' marker. Raises ValueError on an unrecognized shape
    rather than silently emitting a malformed deck."""
    lines = ngspice_deck.splitlines()
    if not lines or not lines[0].startswith("*"):
        raise ValueError("expected a '* ...' title comment as the deck's first line")
    title = lines[0][1:].strip()

    idx_lib = next((i for i, ln in enumerate(lines) if _LIB_LINE_RE.match(ln)), None)
    if idx_lib is None:
        raise ValueError('expected a \'.lib "<path>" <corner>\' line')

    idx_param = next(
        (i for i in range(idx_lib + 1, len(lines)) if _PARAM_LINE_RE.match(lines[i])), None
    )
    if idx_param is None:
        raise ValueError("expected a '.param ...' line after the .lib line")
    param_body = _PARAM_LINE_RE.match(lines[idx_param]).group(1).strip()

    body_end = next(
        (
            i
            for i in range(idx_param + 1, len(lines))
            if lines[i].strip().lower() in (".control", ".end")
        ),
        len(lines),
    )
    device_lines = [ln for ln in lines[idx_param + 1 : body_end] if ln.strip()]
    if not device_lines:
        raise ValueError("no device/source lines found between .param and .control/.end")
    return title, param_body, device_lines


def _substitute_device_models(device_lines: list[str]) -> list[str]:
    """Exact-literal token substitution (see _DEVICE_MODEL_TOKENS' module comment). Raises loudly
    if a sky130 device-model token survives that this renderer doesn't recognize -- never ship an
    un-substituted home PDK literal into a company-facing deck."""
    out = []
    for ln in device_lines:
        new_ln = ln
        for token, placeholder in _DEVICE_MODEL_TOKENS.items():
            new_ln = new_ln.replace(token, placeholder)
        if _SKY130_TOKEN_PREFIX in new_ln:
            raise ValueError(
                f"unrecognized {_SKY130_TOKEN_PREFIX}* device-model token survived substitution "
                f"in line {ln!r} -- extend _DEVICE_MODEL_TOKENS before rendering this template"
            )
        out.append(new_ln)
    return out


def _todo_block() -> str:
    bar = "*" + "=" * 98
    return "\n".join(
        [
            bar,
            "* TODO(engineer): fill in the company PDK before running this deck on PrimeSim/HSPICE.",
            f"*  1) {COMPANY_PDK_LIB_PLACEHOLDER} -> your PDK's HSPICE-format model library file path.",
            f"*  2) {CORNER_PLACEHOLDER} -> a corner SECTION NAME defined inside that library file",
            "*     (PDK corners live in the vendor's own Variation Block, selected by .LIB's 2nd",
            "*     arg -- never invented here; home used a 'tt' (typical) corner, see below).",
            f"*  3) {NMOS_MODEL_PLACEHOLDER} / {PMOS_MODEL_PLACEHOLDER} -> your PDK's NMOS/PMOS",
            "*     device or subcircuit model names. This deck's X-prefixed instances assume a",
            "*     subcircuit-wrapped primitive (matching sky130's own nfet_01v8/pfet_01v8 convention).",
            "*     If your PDK exposes bare .MODEL NMOS/PMOS primitives instead, drop the X-prefix",
            "*     and re-check W/L argument order against your PDK's device usage guide.",
            "*  Geometry (W/L) and electrical (Cc/CL/IREFV/...) .PARAM values below are the HOME",
            "*  sky130A-VALIDATED sizing, carried through UNCHANGED -- almost certainly NOT portable",
            "*  as-is to your process node. Re-derive sizing for your PDK before trusting any",
            "*  absolute number this deck reports. See EXPECTATION.md in this bench's directory:",
            "*  the certified claim is the STRUCTURE (invariance / direction / elasticity), never",
            "*  the home magnitude.",
            bar,
        ]
    )


def render_hspice_deck(ngspice_deck: str, spec: HspiceBenchSpec) -> str:
    """Deliverable A's entry point. Deterministic: render_hspice_deck(deck, spec) called twice
    returns byte-identical text (no timestamps/randomness/wall-clock anywhere in this function or
    its helpers) -- see tests/test_executable_render_hspice.py::test_render_is_byte_stable.

    NEVER executes anything: pure str-in/str-out. This file itself never spells 'subprocess' or
    'docker' or shells out to ngspice/primesim (grep -c 'subprocess\\|docker' render_hspice.py
    == 0) -- the only imports are laws.compute_law_id (a pure function) and, lazily inside
    _render_target_deck, templates.py's own pure render_* functions."""
    title, param_body, device_lines = _split_ngspice_deck(ngspice_deck)
    device_lines = _substitute_device_models(device_lines)

    lines: list[str] = []
    lines.append(f"* {title} -- PrimeSim/HSPICE dialect (RENDER-ONLY; home never runs this deck)")
    lines.append(f"* bench_id={spec.bench_id}  topology_class={spec.topology_class}  "
                 f"claim={spec.claim_id}")
    lines.append(f"* home corner (informational only, NOT a company value): {spec.corner!r}")
    for note in spec.header_notes:
        lines.append(f"* {note}")
    lines.append(_todo_block())
    lines.append(f".LIB '{COMPANY_PDK_LIB_PLACEHOLDER}' {CORNER_PLACEHOLDER}")
    lines.append(f".TEMP {spec.temp_c:g}")
    lines.append(f".PARAM {param_body}")
    lines.extend(device_lines)
    lines.extend(spec.analysis_lines)
    for m in spec.measures:
        lines.append(f"* ngspice source: {m.ngspice_source}")
        for key in m.citation_keys:
            cite = _DIALECT_CITATIONS[key]
            lines.append(f"* guide chunks: {', '.join(cite['chunks'])}")
            if cite["todo"]:
                lines.append(f"* TODO: {cite['todo']}")
        if m.extra_todo:
            lines.append(f"* TODO: {m.extra_todo}")
        lines.extend(m.hspice_lines)
    lines.append(".OPTION POST PROBE")
    lines.extend(spec.probe_lines)
    for i, pt in enumerate(spec.sweep_points):
        safe = re.sub(r"[^A-Za-z0-9]", "_", pt).strip("_") or f"idx{i}"
        lines.append(f".ALTER pt{i}_{safe}")
        lines.append(f".PARAM {spec.sweep_param}={pt}")
    lines.append(".END")
    lines.append("")
    return "\n".join(lines)


# ================================================================================================
# Deliverable B support -- the 4 pilot bench definitions (exactly the PILOT SCOPE, no more)
# ================================================================================================


@dataclass(frozen=True)
class PilotBenchDef:
    """Static (non-graph) definition of one pilot bench: which template renderer to call, with
    which args, to reproduce the SAME ngspice deck that actually produced the cited claim/law on
    the graph -- verified byte-for-byte against the git-corpus SSOT / raw E1 JSONL during this
    task's research (see EXPECTATION.md's Provenance section per bench for exact citations)."""

    bench_id: str
    title: str
    topology_class: str
    claim: str  # ClaimCard.claim / MechanismClaim id, e.g. "ota5t_av0"
    knob: str  # MechanismClaim.knob, as recorded on the graph (e.g. "CL", "Vout")
    metric: str
    quant_kind: str
    corner: str
    temp_c: float
    sweep_param: str  # the exact .param token in the template body (see HspiceBenchSpec docstring)
    sweep_points: tuple[str, ...]
    render_fn_name: str  # templates.py function name (imported lazily to avoid a hard import cycle)
    measures: tuple[MeasureTranslation, ...]
    probe_lines: tuple[str, ...]
    misconception_note: str | None = None  # only set for the REFUTED bench (bench 2)


def _render_target_deck(bench: "PilotBenchDef") -> str:
    """Calls the SAME (unmodified) templates.py render function that produced the graph-verified
    claim, with the SAME corner/knob/metric/points -- the ngspice-dialect input this module
    transforms. templates.py is imported here, not at module top level, so importing
    render_hspice.py never has a side effect on templates.py's own import order."""
    from openclaw_brain.knowledge.executable import templates

    fn = getattr(templates, bench.render_fn_name)
    return fn(corner=bench.corner, knob=bench.knob, metric=bench.metric,
              points=list(bench.sweep_points))


# --- Bench 1: ota_5t_nmos_in -- av0 vs CL invariance (LAW, 3-PDK) -----------------------------------
_BENCH_1 = PilotBenchDef(
    bench_id="ota5t_av0_vs_cl",
    title="ota_5t_nmos_in: av0_db invariance to CL",
    topology_class="ota_5t_nmos_in",
    claim="ota5t_av0",
    knob="CL",
    metric="av0_db",
    quant_kind="invariance",
    corner="tt",
    temp_c=27.0,
    sweep_param="CL",
    sweep_points=("500f", "1000f", "2000f", "4000f"),
    render_fn_name="render_ota_5t_ac",
    measures=(
        MeasureTranslation(
            ngspice_source="let y = 0/0 \\ meas ac y max vdb(out)",
            hspice_lines=(".MEASURE AC av0_db MAX VDB(out)",),
            citation_keys=("measure_func_max",),
        ),
    ),
    probe_lines=(".PROBE AC V(out)",),
)

# --- Bench 2: miller_ota_2stage_nmos_in -- gbw vs CL elasticity, REFUTED (misconception correction) -
_BENCH_2 = PilotBenchDef(
    bench_id="miller_ota_gbw_vs_cl_elasticity",
    title="miller_ota_2stage_nmos_in: gbw_hz elasticity vs CL (REFUTED -1.0 misconception)",
    topology_class="miller_ota_2stage_nmos_in",
    claim="C1",
    knob="CL",
    metric="gbw_hz",
    quant_kind="elasticity",
    corner="tt",
    temp_c=27.0,
    sweep_param="CL",
    sweep_points=("2e-12", "5e-12", "1e-11", "2e-11", "5e-11"),
    render_fn_name="render_miller_ota_ac",
    measures=(
        MeasureTranslation(
            ngspice_source="let y = 0/0 \\ meas ac y when vdb(out)=0 cross=1",
            hspice_lines=(".MEASURE AC gbw_hz WHEN VDB(out)=0 CROSS=1",),
            citation_keys=("measure_when_cross",),
        ),
    ),
    probe_lines=(".PROBE AC V(out)",),
    misconception_note=(
        "an LLM-authored recipe (S3 corpus-growth automation) hypothesized elasticity ~= -1.0 "
        "(band [-1.1,-0.9]) reasoning that CL's non-dominant pole eventually dominates GBW; the "
        "deterministic oracle REFUTED it -- measured global log-log slope -0.428 (~-0.43), well "
        "outside the claimed band. Teaching point: for a Miller-compensated 2-stage OTA, GBW is "
        "set primarily by Cc (gm1/(2*pi*Cc)), not CL -- CL has a real but much weaker effect than "
        "a naive -1 elasticity guess. Trust the oracle over an LLM's a-priori physical reasoning."
    ),
)

# --- Bench 3: miller_ota_2stage_nmos_in -- phase margin vs CL direction (LAW, 3-PDK) ----------------
_BENCH_3 = PilotBenchDef(
    bench_id="miller_ota_pm_vs_cl",
    title="miller_ota_2stage_nmos_in: pm_deg direction vs CL",
    topology_class="miller_ota_2stage_nmos_in",
    claim="cl_pm",
    knob="CL",
    metric="pm_deg",
    quant_kind="direction",
    corner="tt",
    temp_c=27.0,
    sweep_param="CL",
    sweep_points=("1p", "2p", "4p"),
    render_fn_name="render_miller_ota_ac",
    measures=(
        MeasureTranslation(
            ngspice_source="let ph = 0/0 \\ meas ac ph find vp(out) when vdb(out)=0 cross=1 "
            "\\ let y = 180 + ph*180/pi",
            hspice_lines=(
                ".MEASURE AC ph_raw_deg FIND VP(out) WHEN VDB(out)=0 CROSS=1",
                ".MEASURE AC pm_deg PARAM='180 + ph_raw_deg*180/PI'",
            ),
            citation_keys=("measure_find_when", "measure_param_expr"),
            extra_todo=(
                "ngspice's VP() returns phase in RADIANS, hence the *180/PI conversion; whether "
                "PrimeSim's VP() already returns DEGREES was not confirmed in the guide chunks "
                "queried -- if so, drop the *180/PI term (pm_deg = 180 + ph_raw_deg directly)."
            ),
        ),
    ),
    probe_lines=(".PROBE AC V(out)",),
)

# --- Bench 4: cascode_current_mirror_nmos -- iout accuracy (invariance to Vout, LAW, 3-PDK) --------
_BENCH_4 = PilotBenchDef(
    bench_id="cascode_mirror_iout_accuracy",
    title="cascode_current_mirror_nmos: iout_a invariance to Vout",
    topology_class="cascode_current_mirror_nmos",
    claim="casc_iout",
    knob="Vout",
    metric="iout_a",
    quant_kind="invariance",
    corner="tt",
    temp_c=27.0,
    sweep_param="VOUT",  # NOTE: the template's .param is "VOUT" (all-caps); the swept SOURCE ref-
    # designator ngspice `alter`s is "Vout" (mixed-case) -- see HspiceBenchSpec's docstring. Verified
    # against templates.py's _CASC_BODY ("Vout out 0 {VOUT}") and DEFAULT_CASC_SIZING's "VOUT" key.
    sweep_points=("0.8", "1.0", "1.2", "1.4", "1.6"),
    render_fn_name="render_cascode_mirror_dc",
    measures=(
        MeasureTranslation(
            ngspice_source="let y = -i(vout)",
            hspice_lines=(".MEASURE DC iout_a PARAM='-I(Vout)'",),
            citation_keys=("measure_param_expr",),
        ),
    ),
    probe_lines=(".PROBE DC V(out) I(Vout)",),
)

PILOT_BENCHES: tuple[PilotBenchDef, ...] = (_BENCH_1, _BENCH_2, _BENCH_3, _BENCH_4)

# Home-measured raw (knob, metric) magnitude series, transcribed once from the exact artifact each
# bench's live Regularity.derived_from field (when a law exists) or git-corpus claim_cards.yaml
# (when it doesn't -- bench 2) points at. NOT a live Neo4j property (see build_expectation_markdown's
# own note on why) -- these are frozen historical measurements, cited by file path, never re-derived
# by re-simulating (this module never simulates anything). Verified against the live graph's
# Regularity.derived_from.jsonl_paths / the corpus YAML during this pilot's construction
# (2026-07-11); keyed by bench_id so cli.py's render-bench command can pass the right note through
# fetch_expectation_data without duplicating this table.
HOME_MEASURED_NOTES: dict[str, str] = {
    "ota5t_av0_vs_cl": (
        "experiments/e1_cross_pdk_raw.jsonl (route_key cl_av0_db; also replicated in "
        "experiments/e_rollout_full_registry_raw.jsonl) -- av0_db flat across CL in {500f, 1000f, "
        "2000f, 4000f}: sky130A=37.1104 dB, gf180mcuD=41.1185 dB, ihp-sg13g2=24.0808 dB "
        "(constant per PDK to the printed precision -- the invariance claim itself)."
    ),
    "miller_ota_gbw_vs_cl_elasticity": (
        "corpus/specimens/miller_ota_2stage_nmos_in/6cff964f94d9f0db/claim_cards.yaml (claim id "
        "'C1', git-tracked corpus SSOT -- this claim is sky130A-only and NOT part of the E1/③ "
        "cross-PDK JSONL replication, since it was authored by grow-corpus after the seed catalog "
        "was frozen): fitted global log-log slope of gbw_hz vs CL = -0.428, over CL in "
        "{2p,5p,10p,20p,50p} -- band claimed was [-1.1,-0.9], REFUTED."
    ),
    "miller_ota_pm_vs_cl": (
        "experiments/e_rollout_full_registry_raw.jsonl (route_key cl_pm_deg): pm_deg at CL="
        "{1p,2p,4p} -- sky130A: 43.7473, 34.2617, 25.7381 deg; gf180mcuD: 42.3512, 32.8053, "
        "24.3345 deg; ihp-sg13g2: 46.0591, 37.571, 29.9625 deg (monotonically decreasing on all "
        "3 PDKs -- the direction claim itself)."
    ),
    "cascode_mirror_iout_accuracy": (
        "experiments/e_rollout_full_registry_raw.jsonl (route_key vout_iout_a): iout_a at Vout="
        "{0.8,1.0,1.2,1.4,1.6} V -- sky130A: 9.98394e-6..1.00005e-5 A (CoV 0.2%); gf180mcuD: "
        "9.94719e-6..9.99895e-6 A (CoV 0.5%); ihp-sg13g2: 1.00271e-5..1.02159e-5 A (CoV 1.9%, "
        "closest to the 2% bound of the 3 PDKs) -- IREFV=10u nominal, near-ideal 1:1 mirror ratio."
    ),
}


def _analysis_lines_for(bench: PilotBenchDef, ngspice_deck: str) -> tuple[str, ...]:
    """Extracts the ONE analysis-primitive line ('ac dec 40 1 10G' or 'op') from inside the
    ngspice deck's own .control block and translates it -- citation: ac_analysis / op_analysis."""
    m = re.search(r"^\s*ac\s+dec\s+(\S+)\s+(\S+)\s+(\S+)\s*$", ngspice_deck,
                  re.IGNORECASE | re.MULTILINE)
    if m:
        return (f".AC DEC {m.group(1)} {m.group(2)} {m.group(3)}",)
    if re.search(r"^\s*op\s*$", ngspice_deck, re.IGNORECASE | re.MULTILINE):
        return (".OP",)
    raise ValueError("could not find a recognized analysis primitive ('ac dec ...' or 'op') "
                      "in the ngspice deck to translate")


def render_pilot_deck(bench: PilotBenchDef) -> str:
    """Renders ONE pilot bench's HSPICE/PrimeSim deck text (pure; no I/O). Calls the unmodified
    templates.py render function to get the ngspice-dialect input, then render_hspice_deck to
    translate it."""
    ngspice_deck = _render_target_deck(bench)
    analysis_lines = _analysis_lines_for(bench, ngspice_deck)
    spec = HspiceBenchSpec(
        bench_id=bench.bench_id,
        topology_class=bench.topology_class,
        claim_id=bench.claim,
        corner=bench.corner,
        temp_c=bench.temp_c,
        sweep_param=bench.sweep_param,
        sweep_points=bench.sweep_points,
        analysis_lines=analysis_lines,
        measures=bench.measures,
        probe_lines=bench.probe_lines,
    )
    return render_hspice_deck(ngspice_deck, spec)


# ================================================================================================
# Deliverable B support -- EXPECTATION.md cards, graph-sourced (Regularity + ClaimCard, READ-ONLY)
# ================================================================================================


async def fetch_regularity(
    store: Any, topology_class: str, metric: str, knob: str, quant_kind: str
) -> dict | None:
    """READ-ONLY. law_id is derived via laws.compute_law_id (the SSOT -- never re-implemented
    here) so this can never drift from how laws.py itself keys a Regularity node. Returns the raw
    node property dict (member_summary/derived_from still JSON-encoded strings exactly as Neo4j
    returns them) or None if this shape was never promoted to a cross-PDK law."""
    law_id = compute_law_id(topology_class, metric, knob, quant_kind)
    return await store.get_node(NodeLabel.REGULARITY, "law_id", law_id)


async def fetch_claim_cards(store: Any, topology_class: str, claim: str) -> list[dict]:
    """READ-ONLY. Every (Specimen.pdk, ClaimCard) pair on the live graph for one (topology_class,
    claim) shape -- same MATCH shape projection.py's own writer uses to link Specimen-[:HAS_CLAIM]->
    ClaimCard, read back here rather than reinvented."""
    rows = await store.run_read_query(
        "MATCH (sp:Specimen)-[:HAS_CLAIM]->(c:ClaimCard) "
        "WHERE sp.topology_class = $tc AND c.claim = $claim "
        "RETURN sp.pdk AS pdk, sp.spec_id AS spec_id, c AS card "
        "ORDER BY sp.pdk",
        {"tc": topology_class, "claim": claim},
    )
    return [{"pdk": r["pdk"], "spec_id": r["spec_id"], "card": dict(r["card"])} for r in rows]


@dataclass(frozen=True)
class ExpectationData:
    """Everything build_expectation_markdown needs, already extracted from the live graph fields
    (regularity/claim_cards -- raw property dicts, JSON sub-fields left as-is so the pure builder
    below stays trivially testable against a hand-built dict of exactly this shape)."""

    bench: PilotBenchDef
    regularity: dict | None  # raw Regularity node properties, or None if not a law
    claim_cards: list[dict]  # [{"pdk", "spec_id", "card": {...raw ClaimCard properties...}}]
    home_measured_note: str  # citation-carrying prose pointing at the raw JSONL/corpus artifact


async def fetch_expectation_data(
    store: Any, bench: PilotBenchDef, home_measured_note: str
) -> ExpectationData:
    """READ-ONLY orchestration: one Regularity lookup + one ClaimCard sweep, both against the live
    graph via GraphStore's documented read surface (get_node / run_read_query) -- never a write."""
    regularity = await fetch_regularity(
        store, bench.topology_class, bench.metric, bench.knob, bench.quant_kind
    )
    claim_cards = await fetch_claim_cards(store, bench.topology_class, bench.claim)
    return ExpectationData(
        bench=bench, regularity=regularity, claim_cards=claim_cards,
        home_measured_note=home_measured_note,
    )


def _fmt_member_summary(member_summary_json: str | None) -> list[str]:
    if not member_summary_json:
        return []
    try:
        summary = json.loads(member_summary_json)
    except (TypeError, json.JSONDecodeError):
        return []
    lines = []
    for pdk in sorted(k for k in summary if not k.startswith("_")):
        entry = summary[pdk]
        bits = [f"verdict={entry.get('verdict')}"]
        if entry.get("note"):
            bits.append(f"note={entry['note']!r}")
        if "fitted_exponent" in entry:
            bits.append(f"fitted_exponent={entry['fitted_exponent']}")
        lines.append(f"- **{pdk}**: {', '.join(bits)}")
    return lines


def build_expectation_markdown(data: ExpectationData) -> str:
    """Pure function: ExpectationData -> the EXPECTATION.md text. No graph access here -- keeps
    this trivially unit-testable against a hand-built ExpectationData (see
    tests/test_executable_render_hspice.py::test_expectation_card_sourced_from_graph_fields)."""
    b = data.bench
    reg = data.regularity
    lines: list[str] = []
    lines.append(f"# EXPECTATION -- {b.title}")
    lines.append("")
    lines.append("## Claim under test")
    lines.append(f"- topology_class: `{b.topology_class}`")
    lines.append(f"- claim id: `{b.claim}`")
    lines.append(f"- knob -> metric: `{b.knob}` -> `{b.metric}`")
    lines.append(f"- quant_kind: `{b.quant_kind}`")
    if reg:
        member_pdks = ", ".join(reg.get("pdks") or [])
        lines.append(f"- **status: LAW** (`law_id={reg.get('law_id')}`, "
                      f"status={reg.get('status')!r}) -- replicated across {member_pdks}")
    else:
        lines.append("- **status: single-PDK ClaimCard** (not promoted to a cross-PDK law -- "
                      "see scope note below)")
    lines.append("")
    lines.append("## SCOPE-HONEST statement (this is what \"verified at home\" means)")
    pdks_tested = sorted({cc["pdk"] for cc in data.claim_cards}) or (reg.get("pdks") if reg else [])
    if b.quant_kind in ("invariance", "direction"):
        shape_note = "this metric holds its value/direction across the swept knob"
    else:
        shape_note = "the fitted scaling exponent"
    lines.append(
        f"Verified by **ngspice** on **{{{', '.join(pdks_tested)}}}** (nominal TT corner, 27C, "
        f"1.8V) -- see measured values below. **PrimeSim + company-PDK numbers WILL differ.** "
        f"The certified claim is the **{b.quant_kind.upper()} STRUCTURE** ({shape_note}), "
        f"never the absolute home number."
    )
    if b.misconception_note:
        lines.append("")
        lines.append("### Misconception-correction framing")
        lines.append(b.misconception_note)
    lines.append("")
    lines.append("## Home-measured values (per PDK, live graph fields)")
    if reg and reg.get("statement"):
        lines.append(f"- Regularity statement (magnitude-free by construction): {reg['statement']!r}")
    member_lines = _fmt_member_summary(reg.get("member_summary")) if reg else []
    if member_lines:
        lines.append("- Per-PDK member_summary (Regularity node, live):")
        lines.extend(f"  {ln}" for ln in member_lines)
    if data.claim_cards:
        lines.append("- ClaimCard verdicts (live graph, one per PDK specimen):")
        for cc in data.claim_cards:
            card = cc["card"]
            lines.append(
                f"  - **{cc['pdk']}**: verdict={card.get('verdict')}, "
                f"claim_id={card.get('claim_id')}"
            )
    lines.append(f"- Raw magnitude series (NOT a live Neo4j property -- ClaimCard nodes carry "
                 f"verdict/scope/quant_kind but not verdict_note or the measured (x,y) series; "
                 f"the number below is transcribed from the SAME artifact the Regularity node's "
                 f"own `derived_from` field points at): {data.home_measured_note}")
    lines.append("")
    lines.append("## What to measure on PrimeSim + the company PDK")
    lines.append(f"- Sweep `.PARAM {b.sweep_param}` across the points in `bench.sp` "
                 f"({', '.join(b.sweep_points)}), or your own PDK-appropriate range.")
    lines.append(f"- Record `{b.metric}` at each point (see `bench.sp`'s `.MEASURE` statements).")
    lines.append("- Fill in `results.json` per this bench's schema (see the pilot README) and "
                 "bring it back for comparison against the home ngspice bands above.")
    lines.append("")
    lines.append("## PASS-shape guidance")
    if b.quant_kind == "invariance":
        lines.append(f"- PASS-shape: `{b.metric}` stays essentially flat across the `{b.knob}` "
                     f"sweep (small spread/CoV, not necessarily the SAME absolute value as home).")
    elif b.quant_kind == "direction":
        lines.append(f"- PASS-shape: `{b.metric}` moves monotonically in the SAME direction as "
                     f"`{b.knob}` increases (sign match), regardless of absolute magnitude.")
    elif b.quant_kind == "elasticity":
        lines.append(f"- PASS-shape: fit the log-log slope of `{b.metric}` vs `{b.knob}` across "
                     f"the swept points; compare the SIGN and rough MAGNITUDE of the exponent "
                     f"against the home fitted value noted above -- do not expect it to match "
                     f"home's sky130-specific exponent exactly.")
    lines.append("- A different absolute number on the company PDK is EXPECTED and not itself a "
                 "failure -- the law is the structure, not the magnitude (ADR-044 D2 / "
                 "the executable substrate's scope-honesty invariant).")
    lines.append("")
    lines.append("## Provenance")
    lines.append("- Regularity/ClaimCard fields above were queried live (READ-ONLY) from the "
                 "openclaw-brain Neo4j graph via `GraphStore.get_node` / `run_read_query` "
                 "(`.claude/skills/verify-ingest` connect pattern) -- not hand-copied from docs.")
    if reg:
        lines.append(f"- `law_id = compute_law_id({b.topology_class!r}, {b.metric!r}, {b.knob!r}, "
                     f"{b.quant_kind!r}) = {reg.get('law_id')}`")
        if reg.get("derived_from"):
            lines.append(f"- law derived_from: {reg['derived_from']}")
    lines.append(f"- Rendered netlist: `bench.sp` (this directory), produced by "
                 f"`render_hspice.render_pilot_deck` from `templates.{b.render_fn_name}` "
                 f"(unmodified) -- same sizing/knob/points as the graph-verified claim.")
    lines.append("")
    return "\n".join(lines)


# ================================================================================================
# Deliverable D support -- one orchestration entry point cli.py's `render-bench` wraps
# ================================================================================================


async def render_all_pilot_benches(store: Any, out_dir: Any) -> list[dict]:
    """Renders every PILOT_BENCHES entry's bench.sp + EXPECTATION.md under out_dir/<bench_id>/.
    `store` is a GraphStore (or any duck-typed object exposing get_node/run_read_query -- see
    tests/test_executable_render_hspice.py's _FakeStore for the mocked shape); `out_dir` is a
    pathlib.Path-like. READ-ONLY against the graph; writes only under out_dir (never to the
    graph, never to templates.py/corpus/experiments). Returns one summary dict per bench (for the
    CLI to echo) -- never raises on a per-bench basis beyond what render_pilot_deck/
    fetch_expectation_data themselves raise (a genuinely malformed bench def should fail loud, not
    be swallowed -- there are exactly 4 of these, hand-curated, not a batch where one bad entry
    should not block the rest)."""
    results: list[dict] = []
    for bench in PILOT_BENCHES:
        bench_dir = out_dir / bench.bench_id
        bench_dir.mkdir(parents=True, exist_ok=True)

        deck_text = render_pilot_deck(bench)
        (bench_dir / "bench.sp").write_text(deck_text)

        note = HOME_MEASURED_NOTES.get(bench.bench_id, "(no static home-measured note registered)")
        data = await fetch_expectation_data(store, bench, note)
        card = build_expectation_markdown(data)
        (bench_dir / "EXPECTATION.md").write_text(card)

        results.append({
            "bench_id": bench.bench_id,
            "dir": str(bench_dir),
            "deck_bytes": len(deck_text),
            "card_bytes": len(card),
            "is_law": data.regularity is not None,
            "claim_cards": len(data.claim_cards),
        })
    return results
