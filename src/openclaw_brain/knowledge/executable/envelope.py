"""The analytic envelope — E-track ④ E2a, a SECOND, mathematically INDEPENDENT instrument (ADR-045).

See docs/superpowers/specs/2026-07-05-e2a-analytic-envelope.md. Every one of the 26 Regularities
rests on a single kind of instrument: the ngspice measurement (the oracle, `oracle.py`). A template
that renders the wrong topology can be self-consistent under simulation (pass a `direction`/
`invariance` verdict) yet violate the physics; only a SECOND, differently-derived instrument catches
that. This module supplies it: a closed-form small-signal PREDICTION of a metric's value, computed
from operating-point device quantities (gm, gds, and the design's own C/sizing) via textbook
small-signal theory, compared against the oracle's independently-measured value.

THE INDEPENDENCE DISCIPLINE (spec §2, load-bearing):
    (path A, oracle, UNTOUCHED)   render -> AC/tran sweep  -> measure GBW/gain directly (a series)
    (path B, envelope, THIS file) render -> `.op` device query -> gm/gds -> Python closed form
Two DISJOINT computation paths reach the same number. `envelope_check` below computes `predicted`
ONLY from `op_quantities` — it takes `measured_value` as a single scalar to COMPARE AFTER predicted
is already computed, and it NEVER receives, reads, or is passed the oracle's measured (x, y) SERIES.
This is a NEW instrument ALONGSIDE the oracle, never inside it: `oracle.py` stays byte-untouched (see
test_executable_envelope.py's immutability guard), and an envelope verdict is recorded ADDITIVELY on
a claim's `scope` (`stamp_envelope_scope`) — it NEVER overrides an oracle verdict.

Pilot scope (spec §3 — deliberately small, textbook-clean closed forms, each hand-anchored live
against real sky130 in scratchpad/probe_envelope_*.py BEFORE being wired here, per the spec's Q3
hand-anchor discipline):
    miller_ota_2stage_nmos_in / gbw_hz   -> GBW ~= gm1 / (2*pi*Cc)
    ota_5t_nmos_in            / av0_db   -> A0  ~= gm1 / (gds_n + gds_p), in dB
    ota_5t_nmos_in            / gbw_hz   -> GBW ~= gm1 / (2*pi*CL)
    common_source_active_load_nmos/av0_db-> A0  ~= gm  / (gds_n + gds_p), in dB
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from .pdks import PDKProfile, get_profile, substitute_devices
from .runner import NgspiceRunner, parse_unit
from .templates import (
    _CS_BODY,
    _OTA5T_BODY,
    _OTA_BODY,
    _cs_param_lines,
    _ota5t_param_lines,
    _param_lines,
    LIB_PLACEHOLDER,
    TEMPLATES,
)

# ===================================================================================================
# §A — EnvelopeModel registry: (topology_class, metric) -> a pure fn op_quantities -> (predicted, band)
# ===================================================================================================

EnvelopeFn = Callable[[dict[str, float]], tuple[float, float]]


def _miller_ota_gbw(op: dict[str, float]) -> tuple[float, float]:
    """GBW ~= gm1 / (2*pi*Cc) — the classic dominant-pole Miller-compensation estimate (input-pair
    transconductance over the compensation cap). Hand-anchored live 2026-07-05 sky130A,
    DEFAULT_OTA_SIZING (Cc=1p) — scratchpad/probe_envelope_miller_gbw.py:
        gm1 = 1.86053e-4 S  ->  predicted = gm1/(2*pi*1e-12) = 2.96113e7 Hz
        oracle-measured GBW (SAME nominal Cc=1p sweep point) = 2.42641e7 Hz
        ratio measured/predicted = 0.8194  (predicted overshoots by ~18%)
    Band +-25% (WIDE relative to the other 3 pilots): this formula deliberately omits (a) the RHP
    zero from the direct Cc feedforward path (at ~gm6/Cc, not far above the dominant pole for this
    Rz-less compensation, so it measurably erodes the 0dB-crossing frequency) and (b) the finite
    output impedance of stage 1 loading Cc. The oracle measures GBW as the AC deck's actual 0dB-
    crossing of the FULL two-pole-plus-zero transfer function; gm1/(2*pi*Cc) is a first-order stand-
    in for that, not the transfer function itself — the ~18% gap is the KNOWN, physically-named
    correction this envelope's band absorbs (Q4's "envelope x correction-factor" story), not slop.
    """
    gm1 = op["gm1"]
    cc = op["Cc"]
    predicted = gm1 / (2 * math.pi * cc)
    return predicted, 0.25


def _ota5t_av0_db(op: dict[str, float]) -> tuple[float, float]:
    """A0 = gm1*(ro_n||ro_p) = gm1/(gds_n+gds_p), reported in dB. Hand-anchored live 2026-07-05
    sky130A, DEFAULT_OTA5T_SIZING (CL=1p) — scratchpad/probe_envelope_ota5t.py:
        gm1=1.9643e-4 S, gds_n=1.71937e-6 S, gds_p=9.70258e-7 S
        predicted = 20*log10(1.9643e-4/(1.71937e-6+9.70258e-7)) = 20*log10(73.03) = 37.2703 dB
        oracle-measured av0_db (SAME nominal CL=1p sweep point) = 37.1104 dB
        ratio (linear) = 10**((37.1104-37.2703)/20) = 0.9818  (deviation ~1.8%)
    Band +-15%: a single-stage OTA's output node has NO second pole/zero between DC and the
    dominant pole to omit -- av0_db IS, to leading order, exactly this gm/gds ratio (unlike the
    Miller GBW above, there is no comparably-sized missing term); the residual ~2% is ngspice's own
    .op vs. .ac-at-DC numerical difference, not a structural approximation. Kept looser than the raw
    1.8% observation so a modest per-PDK/per-corner shift does not manufacture a false VIOLATED.
    """
    gm1 = op["gm1"]
    gds_n = op["gds_n"]
    gds_p = op["gds_p"]
    predicted_db = 20 * math.log10(gm1 / (gds_n + gds_p))
    return predicted_db, 0.15


def _ota5t_gbw(op: dict[str, float]) -> tuple[float, float]:
    """GBW ~= gm1 / (2*pi*CL) — the single dominant output pole (no Miller path, no RHP zero to
    omit: the 5T OTA is a genuine single-stage amplifier). Hand-anchored live 2026-07-05 sky130A,
    DEFAULT_OTA5T_SIZING (CL=1p) — scratchpad/probe_envelope_ota5t.py:
        gm1 = 1.9643e-4 S  ->  predicted = gm1/(2*pi*1e-12) = 3.12628e7 Hz
        oracle-measured GBW (SAME nominal CL=1p sweep point) = 3.02783e7 Hz
        ratio measured/predicted = 0.9685  (deviation ~3.2%)
    Band +-15%: tighter than the Miller GBW's (no zero/2nd-pole to omit) but a bit looser than the
    5T Av0's, since the .meas 0dB-crossing search is itself a discretized (dec 40) AC sweep and GBW
    is more sensitive to that discretization than a flat DC-gain plateau is.
    """
    gm1 = op["gm1"]
    cl = op["CL"]
    predicted = gm1 / (2 * math.pi * cl)
    return predicted, 0.15


def _cs_av0_db(op: dict[str, float]) -> tuple[float, float]:
    """A0 = gm*(ro_n||ro_p) = gm/(gds_n+gds_p), in dB — same structural formula as the 5T OTA's gain
    (a single high-impedance output node, no omitted pole/zero). Hand-anchored live 2026-07-05
    sky130A, DEFAULT_CS_SIZING (IREFV=10u) — scratchpad/probe_envelope_cs.py:
        gm=1.89283e-4 S, gds_n=1.72924e-6 S, gds_p=9.41343e-7 S
        predicted = 20*log10(1.89283e-4/(1.72924e-6+9.41343e-7)) = 37.0108 dB
        oracle-measured av0_db (SAME nominal Iref=10u sweep point) = 37.0101 dB
        ratio (linear) = 0.9999... ~= 1.0000  (deviation < 0.01%)
    Band +-15% (matches the 5T Av0 pilot for the identical structural reason -- consistency across
    the two textbook-exact gain formulas), even though the live anchor here is essentially exact:
    the tight observed match is itself evidence FOR the formula, but the band stays conservative
    against per-PDK/per-corner shift (this envelope runs cross-PDK in I2, sky130 is only one point).
    """
    gm = op["gm"]
    gds_n = op["gds_n"]
    gds_p = op["gds_p"]
    predicted_db = 20 * math.log10(gm / (gds_n + gds_p))
    return predicted_db, 0.15


@dataclass(frozen=True)
class EnvelopeModel:
    """One (topology_class, metric)'s closed-form prediction. `fn` is pure, deterministic, and reads
    ONLY the op-quantities dict it is given (see the independence discipline in the module docstring).
    `band` is the fractional tolerance HAND-JUSTIFIED per formula (see each `fn`'s docstring for the
    physical reason) -- a formula that omits real 2nd-order physics (the Miller GBW's RHP zero) gets
    a wider band than one that doesn't (the two textbook-exact gain formulas)."""

    topology_class: str
    metric: str
    fn: EnvelopeFn


REGISTRY: dict[tuple[str, str], EnvelopeModel] = {
    ("miller_ota_2stage_nmos_in", "gbw_hz"): EnvelopeModel(
        "miller_ota_2stage_nmos_in", "gbw_hz", _miller_ota_gbw),
    ("ota_5t_nmos_in", "av0_db"): EnvelopeModel(
        "ota_5t_nmos_in", "av0_db", _ota5t_av0_db),
    ("ota_5t_nmos_in", "gbw_hz"): EnvelopeModel(
        "ota_5t_nmos_in", "gbw_hz", _ota5t_gbw),
    ("common_source_active_load_nmos", "av0_db"): EnvelopeModel(
        "common_source_active_load_nmos", "av0_db", _cs_av0_db),
}


# ===================================================================================================
# §B — EnvelopeVerdict + envelope_check: the independent comparison (NEVER touches oracle.py/verdict)
# ===================================================================================================


@dataclass
class EnvelopeVerdict:
    """The envelope's own verdict on ONE (topology_class, metric, pdk) cell -- a SEPARATE record from
    the oracle's ClaimCard.verdict, never merged into it. `status` is 'NA' (no envelope registered --
    honest, not a pass), 'CONCUR' (measured/predicted inside `band`), or 'VIOLATED' (outside it)."""

    topology_class: str
    metric: str
    pdk: str
    predicted: float | None
    measured: float
    ratio: float | None
    band: float | None
    status: str  # "CONCUR" | "VIOLATED" | "NA"


def envelope_check(
    topology_class: str,
    metric: str,
    pdk: str,
    measured_value: float,
    op_quantities: dict[str, float],
) -> EnvelopeVerdict:
    """The independent comparison. CRITICAL INDEPENDENCE RULE (spec §2, non-negotiable): `predicted`
    below is computed from `op_quantities` ONLY -- this function does not accept, and the registered
    `EnvelopeModel.fn`s above do not read, the oracle's measured (x, y) series at any point. Line
    order is intentional and load-bearing for a reviewer to verify by inspection: the registry lookup
    and `model.fn(op_quantities)` call happen BEFORE `measured_value` is touched at all; the only use
    of `measured_value` in this entire function is the ratio computed AFTER that line.

    `metric` ending in "_db" is treated as a log-scale quantity: the ratio is computed on the
    underlying LINEAR quantity (10**((measured-predicted)/20)) so a "how far off" ratio near 1.0
    means the same thing regardless of whether the metric's own units are linear (Hz) or dB -- a
    uniform "envelope x correction-factor" story (spec Q4) independent of the metric's unit."""
    model = REGISTRY.get((topology_class, metric))
    if model is None:
        return EnvelopeVerdict(topology_class, metric, pdk, None, measured_value, None, None, "NA")

    predicted, band = model.fn(op_quantities)   # <-- ONLY input: op_quantities. Never the series.

    if predicted == 0:
        ratio = math.inf
    elif metric.endswith("_db"):
        # A wildly-wrong measured_value (e.g. a planted-error injection, spec Q2) can push the dB
        # gap far enough that 10**(...) overflows float64 -- that must still resolve to an emphatic
        # VIOLATED, never a crash (the whole point of Q2 is the envelope CATCHES a bad measurement,
        # not that it dies on one).
        try:
            ratio = 10 ** ((measured_value - predicted) / 20.0)
        except OverflowError:
            ratio = math.inf
    else:
        ratio = measured_value / predicted
    status = "CONCUR" if abs(ratio - 1.0) <= band else "VIOLATED"
    return EnvelopeVerdict(topology_class, metric, pdk, predicted, measured_value, ratio, band, status)


def stamp_envelope_scope(scope: dict[str, Any], verdict: EnvelopeVerdict) -> dict[str, Any]:
    """Additive scope-tag stamp -- mirrors the executor's OWN intervention/idealization stamp pattern
    exactly (`card.scope = {**card.scope, ...}`, a new merged dict, never an in-place mutation of the
    caller's `scope`). This is the I1 substrate I2's executor wiring calls per-claim after judging;
    I1 itself only needs `_scope_inline` (agent.py) to render the key, which this function supplies
    in the exact shape `_scope_inline` expects. NEVER touches verdict/basis/engine -- purely additive,
    and it never overrides the oracle's own ClaimCard.verdict."""
    env: dict[str, Any] = {"status": verdict.status}
    if verdict.ratio is not None:
        env["ratio"] = verdict.ratio
    return {**scope, "envelope": env}


# ===================================================================================================
# §C — OP-dump probes: render the pilot cell (UNMUTATED body) + `.op` + the established
# `@m.<inst>[gm]`/`[gds]` hierarchical device query (E3's gm2 probe / inc2a's swing-icmr precedent),
# echoing RDATA lines the runner's EXISTING `parse_rdata` already parses -- no new sim machinery.
# ===================================================================================================


def render_miller_ota_op_dump(sizing: dict | None = None, corner: str = "tt") -> str:
    """OP-dump probe for miller_ota_2stage_nmos_in: gm1 (input-pair XM1) at a single `.op` point (no
    sweep). Reuses the UNMUTATED `_OTA_BODY` cell body -- only the analysis (a bare `.op` + one device
    query) differs from the oracle's own AC sweep deck (render_miller_ota_ac in templates.py)."""
    return "\n".join([
        "* T-envelope OP-dump miller_ota_2stage_nmos_in (gm1)",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _param_lines(sizing or {}),
        _OTA_BODY,
        ".control",
        "op",
        "let gm1 = @m.xm1.msky130_fd_pr__nfet_01v8[gm]",
        "echo RDATA gm1 0 $&gm1",
        ".endc",
        ".end",
        "",
    ])


def render_ota5t_op_dump(sizing: dict | None = None, corner: str = "tt") -> str:
    """OP-dump probe for ota_5t_nmos_in: gm1 (input-pair XM1), gds_n (output-node NMOS leg XM2),
    gds_p (output-node PMOS mirror XM4) at a single `.op` point. Reuses the UNMUTATED `_OTA5T_BODY`.
    XM1/XM2 are the matched differential pair (identical bias in the balanced DC operating point), so
    XM1's gm stands in for "gm1" in the classic gm1*(ro2||ro4) formula -- the SAME matched-pair
    convention the existing `_icmr_control_block` already relies on (M1 used for both templates)."""
    return "\n".join([
        "* T-envelope OP-dump ota_5t_nmos_in (gm1, gds_n, gds_p)",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _ota5t_param_lines(sizing or {}),
        _OTA5T_BODY,
        ".control",
        "op",
        "let gm1 = @m.xm1.msky130_fd_pr__nfet_01v8[gm]",
        "let gds_n = @m.xm2.msky130_fd_pr__nfet_01v8[gds]",
        "let gds_p = @m.xm4.msky130_fd_pr__pfet_01v8[gds]",
        "echo RDATA gm1 0 $&gm1",
        "echo RDATA gds_n 0 $&gds_n",
        "echo RDATA gds_p 0 $&gds_p",
        ".endc",
        ".end",
        "",
    ])


def render_cs_op_dump(sizing: dict | None = None, corner: str = "tt") -> str:
    """OP-dump probe for common_source_active_load_nmos: gm + gds_n (XM1, the NMOS gain device) and
    gds_p (XM2, the PMOS mirror load) at a single `.op` point. Reuses the UNMUTATED `_CS_BODY`."""
    return "\n".join([
        "* T-envelope OP-dump common_source_active_load_nmos (gm, gds_n, gds_p)",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _cs_param_lines(sizing or {}),
        _CS_BODY,
        ".control",
        "op",
        "let gm = @m.xm1.msky130_fd_pr__nfet_01v8[gm]",
        "let gds_n = @m.xm1.msky130_fd_pr__nfet_01v8[gds]",
        "let gds_p = @m.xm2.msky130_fd_pr__pfet_01v8[gds]",
        "echo RDATA gm 0 $&gm",
        "echo RDATA gds_n 0 $&gds_n",
        "echo RDATA gds_p 0 $&gds_p",
        ".endc",
        ".end",
        "",
    ])


_OP_DUMP_RENDERERS: dict[str, Callable[[dict | None, str], str]] = {
    "miller_ota_2stage_nmos_in": render_miller_ota_op_dump,
    "ota_5t_nmos_in": render_ota5t_op_dump,
    "common_source_active_load_nmos": render_cs_op_dump,
}

# I2 cross-PDK finding (live-verified 2026-07-05, `find -L`-anchored reads of each PDK's OWN model
# file -- same methodology as pdks.py's own ground truth): the hierarchical `@<prefix>.<Xinst>.
# <internal-instance>[gm]` device-query convention this module's OP-dump renderers use (E3's gm2
# probe / inc2a's swing-icmr precedent) is a PDK-MODEL-AUTHORSHIP fact, NOT the same thing as
# `pdks.PDKProfile.nfet`/`.pfet` (the SUBCKT NAME `substitute_devices` rewrites on the X-card
# instantiation line -- e.g. "sky130_fd_pr__nfet_01v8" -> "nfet_03v3"). It is WHICH internal element
# each foundry's own subckt body wraps, and what ngspice `@`-prefix that element's type character
# needs:
#   sky130A:    an `m`-type MOSFET instance literally named "m" + the subckt's own name (SkyWater's
#               generated-model convention -- sky130_fd_pr__nfet_01v8__tt.pm3.spice:37, "msky130_fd_
#               pr__nfet_01v8 d g s b sky130_fd_pr__nfet_01v8__model ..."), hence the ALREADY-shipped
#               I1 literal `@m.<Xinst>.msky130_fd_pr__nfet_01v8[gm]`.
#   gf180mcuD:  an `m`-type instance named plainly "m0" (sm141064.spice's nfet_03v3/pfet_03v3
#               .subckt bodies each wrap exactly one `m0 d g s b nfet_03v3/pfet_03v3 ...` line) --
#               `@m.<Xinst>.m0[gm]`. Live-verified: gm1=1.26486e-4 S on the miller OTA OP-dump.
#   ihp-sg13g2: an OSDI/Verilog-A PSP103 instance -- an `n`-type element (ngspice's numerical-device
#               class for an ADMS/OSDI-loaded model), named "N" + the subckt's own name
#               (sg13g2_moslv_mod.lib:66-90's `Nsg13_lv_nmos d g s b sg13g2_lv_nmos_psp ...`, inside
#               conditional `.if` branches keyed on ng/as/rfmode -- the DEFAULT ng=1/as=0/rfmode=0
#               case every recipe in this substrate renders lands on THIS branch, live-verified). The
#               QUERY PREFIX changes too: `@n.<Xinst>.nsg13_lv_nmos[gm]`, not `@m.`. Live-verified:
#               gm1=2.33614e-4 S on the miller OTA OP-dump.
# `substitute_devices` ALREADY rewrites the device-INSTANTIATION lines (XM1 ... sky130_fd_pr__
# nfet_01v8 -> nfet_03v3/sg13_lv_nmos) correctly for every profile -- its blanket nfet/pfet text
# replace ALSO reaches inside this module's `@m.xm1.msky130_fd_pr__nfet_01v8[gm]` line (turning it
# into e.g. `@m.xm1.mnfet_03v3[gm]`), but that is the WRONG internal-instance name for anything but
# sky130A. `_op_query_fixup` (below) corrects it, post-`substitute_devices`, using this table.
_OP_QUERY_CONVENTION: dict[str, dict[str, str]] = {
    "gf180mcuD": {"prefix": "m", "nfet": "m0", "pfet": "m0"},
    "ihp-sg13g2": {"prefix": "n", "nfet": "nsg13_lv_nmos", "pfet": "nsg13_lv_pmos"},
}


def _op_query_fixup(deck: str, profile: PDKProfile) -> str:
    """Correct the OP-dump's `@m.<Xinst>.m<devicename>[gm/gds]` query lines for a non-sky130A profile
    -- see `_OP_QUERY_CONVENTION`'s ground truth above. Raises ValueError (caught by the executor's
    best-effort envelope wiring, degrading to "no tag") for a profile with no verified convention
    entry -- an unverified query would either error loudly inside ngspice (safe but noisy) or, worse,
    silently resolve to the wrong element; refusing is the honest choice, exactly `BjtUnavailable`'s
    and `MismatchUnavailable`'s precedent elsewhere in this substrate."""
    query = _OP_QUERY_CONVENTION.get(profile.pdk)
    if query is None:
        raise ValueError(
            f"{profile.pdk!r}: no verified OP-dump gm/gds device-query convention (envelope.py's "
            f"_OP_QUERY_CONVENTION) -- the analytic envelope's OP-dump probe is not yet available "
            f"on this PDK profile")
    deck = deck.replace(f"m{profile.nfet}", query["nfet"]).replace(f"m{profile.pfet}", query["pfet"])
    if query["prefix"] != "m":
        deck = deck.replace("@m.", f"@{query['prefix']}.")
    return deck

# Which sizing keys pass through into op_quantities AS-IS (parsed via the existing parse_unit) --
# the design's own C values, taken from the recipe's sizing, NEVER from the sim (spec §2's rule).
_SIZING_PASSTHROUGH: dict[str, dict[str, str]] = {
    "miller_ota_2stage_nmos_in": {"Cc": "Cc"},
    "ota_5t_nmos_in": {"CL": "CL"},
    "common_source_active_load_nmos": {},  # av0_db needs no C
}


# ===================================================================================================
# §D — thin runner path: execute the OP-dump, parse gm/gds into an op_quantities dict. Reuses
# NgspiceRunner.measure (+ its existing parse_rdata machinery) -- nothing reinvented here.
# ===================================================================================================


def measure_op_quantities(
    topology_class: str,
    sizing: dict | None = None,
    pdk: str = "sky130A",
    corner: str = "tt",
    runner: NgspiceRunner | None = None,
) -> dict[str, float]:
    """Render `topology_class`'s OP-dump probe, run it through NgspiceRunner.measure (the EXISTING
    runner + RDATA parser -- no new sim machinery), and return a flat {quantity: value} dict: the
    LIVE gm/gds from the `.op` query, plus the design's OWN Cc/CL sizing (never the sim) for the
    formulas that need a capacitor value. Raises KeyError for a topology with no OP-dump probe.

    I2 cross-PDK wiring (spec §5-I2 requirement 4): the OP-dump renders per-PDK EXACTLY like the
    metric decks the executor already runs -- `get_profile` resolves (and validates, raising
    UnknownPDK before any sim) the PDKProfile, and `substitute_devices` applies the SAME post-render
    device/VDD/`.lib`-section rewrite the metric decks get (gf180/ihp geometry comes from `sizing`
    itself -- the caller passes the ALREADY pdks.sizing_overrides_for-resolved dict, exactly as
    executor.run_recipe does for the AC-sweep decks). `runner.measure`'s own `pdk` kwarg is passed
    ONLY for a genuinely non-default profile (mirrors executor._measure's own rationale exactly:
    every existing sky130-only test double, `measure(self, deck, timeout=300)` with no `pdk` param,
    keeps working unmodified on the default path -- this function's own live-anchored I1 callers
    and every existing test therefore need no change to go through this cross-PDK-aware path)."""
    render_fn = _OP_DUMP_RENDERERS.get(topology_class)
    if render_fn is None:
        raise KeyError(f"no OP-dump probe registered for topology_class {topology_class!r}")
    profile = get_profile(pdk)   # raises UnknownPDK before any render/sim (mirrors executor.run_recipe)
    runner = runner or NgspiceRunner()
    deck = render_fn(sizing, corner)
    if profile.pdk != "sky130A":
        deck = substitute_devices(deck, profile)
        deck = _op_query_fixup(deck, profile)   # correct the @m. device-query line (see ground truth above)
        series = runner.measure(deck, pdk=profile.pdk)
    else:
        series = runner.measure(deck)
    op: dict[str, float] = {key: pts[0][1] for key, pts in series.items() if pts}

    default_sizing = TEMPLATES[topology_class]["default_sizing"]
    merged_sizing = {**default_sizing, **{k: str(v) for k, v in (sizing or {}).items()}}
    for op_key, sizing_key in _SIZING_PASSTHROUGH.get(topology_class, {}).items():
        op[op_key] = parse_unit(str(merged_sizing[sizing_key]))
    return op
