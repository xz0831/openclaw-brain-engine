"""T-templates — known-good parametric netlists (SPEC §5, the T-template path).

The connectivity is FIXED (validated against ngspice/sky130); only the sizing fills the
`.param` values. This is what makes recipe-authoring model-independent: the golden-split
showed a small model can get the topology right but the wrong PDK device library
(`sky130_fd_sc_hd` vs `sky130_fd_pr`) — the template guarantees executable correctness
regardless of who authored the recipe. The Miller-OTA body here is the code form of the
validated experiments/executable_circuit_specimens/ota_ac.spice.
"""

from __future__ import annotations

from .runner import parse_unit  # unit-suffixed sizing -> float (S3-inc2a's PTAT core: resistor value
                                 # for the iptat_a = dVBE/R control-script arithmetic). No cycle:
                                 # runner.py only imports from .pdks, never from .templates.

LIB_PLACEHOLDER = "__LIBPATH__"  # the runner substitutes the in-container sky130 ngspice lib path

# Default sizing for miller_ota_2stage_nmos_in (the validated all-saturation point).
# M1 = input-pair multiplicity (default 1 = electrically identical to the validated point). gm/ID
# sizing (executor, method='gmid_lookup') overrides W1:=w_char and M1:=multiplicity to set the
# input-pair inversion level — and thus gm1 and GBW — while the tail-set current id1 stays fixed.
DEFAULT_OTA_SIZING: dict[str, str] = {
    "VDD": "1.8", "VCM": "0.9", "Cc": "1p", "CL": "2p", "Lp": "0.5", "IREFV": "10u",
    "W1": "8", "W3": "4", "W5": "4", "W6": "16", "W7": "8", "W8": "2", "M1": "1",
}

# Fixed connectivity (ngspice {...} braces are LITERAL here — resolved by ngspice from .param):
# NMOS diff pair (M1,M2) + PMOS mirror load (M3 diode, M4); NMOS tail (M5) + bias mirror (M8->M5,M7);
# PMOS common-source 2nd stage (M6) + NMOS sink (M7). Miller Ccomp out->o1. Open-loop AC TB via
# DC-short/AC-open feedback (Lfb into inverting input vinp, Cfb AC-grounds vinp).
_OTA_BODY = """VDD vdd 0 {VDD}
Vin vinn 0 DC {VCM} AC 1
Lfb out vinp 1T
Cfb vinp 0 1T
IREF vdd nbias {IREFV}
XM8 nbias nbias 0 0 sky130_fd_pr__nfet_01v8 W={W8} L={Lp}
XM5 tail nbias 0 0 sky130_fd_pr__nfet_01v8 W={W5} L={Lp}
XM7 out  nbias 0 0 sky130_fd_pr__nfet_01v8 W={W7} L={Lp}
XM1 o1n vinp tail 0 sky130_fd_pr__nfet_01v8 W={W1} L={Lp} m={M1}
XM2 o1  vinn tail 0 sky130_fd_pr__nfet_01v8 W={W1} L={Lp} m={M1}
XM3 o1n o1n vdd vdd sky130_fd_pr__pfet_01v8 W={W3} L={Lp}
XM4 o1  o1n vdd vdd sky130_fd_pr__pfet_01v8 W={W3} L={Lp}
XM6 out o1 vdd vdd sky130_fd_pr__pfet_01v8 W={W6} L={Lp}
Ccomp out o1 {Cc}
CL out 0 {CL}"""

_PARAM_ORDER = ["VDD", "VCM", "Cc", "CL", "Lp", "IREFV", "W1", "W3", "W5", "W6", "W7", "W8", "M1"]

# Capability SSOT — the recipe registry reads these so a recipe can only ever ask the executor to
# sweep a knob the netlist actually defines / measure a metric the deck can extract.
# Knob -> the netlist element the runner `alter`s. Only Ccomp/CL exist in _OTA_BODY; Rz lives in the
# separate ota_rz specimen, NOT this body, so it is deliberately ABSENT (sweeping it would fail).
# VDD/IREFV (S3-inc2a §2/§4 — the swing/ICMR headroom knobs) map to the "VDD" supply source and the
# "IREF" bias-current source — both element ref designators are literally spelled that way in
# _OTA_BODY *and* _OTA5T_BODY (verified by inspection), so this ONE shared dict serves both templates,
# exactly like the pre-existing CL entry already does.
OTA_KNOB_ELEMENTS: dict[str, str] = {"Cc": "Ccomp", "CL": "CL", "VDD": "VDD", "IREFV": "IREF"}
# Metric -> the ngspice .meas/.let that extracts it. Each entry RESETS its measured vector to 0/0
# (a non-finite no-value) BEFORE the .meas: ngspice leaves a FAILED .meas's result vector at its
# prior value (verified), so without the reset a failed point would silently re-emit the previous
# sweep point's number. With the reset, a failed .meas echoes an empty/non-finite y that
# runner.parse_rdata drops -> the series comes back short -> the executor FLAGs it (C1), never
# certifying a number off a failed run.
OTA_METRIC_MEAS: dict[str, str] = {
    "gbw_hz": "let y = 0/0\n  meas ac y when vdb(out)=0 cross=1",
    "pm_deg": "let ph = 0/0\n  meas ac ph find vp(out) when vdb(out)=0 cross=1\n  let y = 180 + ph*180/pi",
    "av0_db": "let y = 0/0\n  meas ac y max vdb(out)",
    # closed-loop gain: SAME measurement as av0_db (peak |vout/vin| in dB), but a distinct metric key
    # so a closed-loop specimen (the PGA) grounds to "Closed-Loop Gain", not the open-loop "av0".
    "acl_db": "let y = 0/0\n  meas ac y max vdb(out)",
}

# Template registry: topology_class → metadata + render function + executable capabilities.
TEMPLATES = {
    "miller_ota_2stage_nmos_in": {
        "name": "miller_ota_2stage_nmos_in",
        "description": "two-stage Miller OTA: NMOS diff pair + PMOS mirror + PMOS CS + CC",
        "devices": ["nfet", "pfet"],
        "default_sizing": DEFAULT_OTA_SIZING,
        "template_ref": "miller_ota_ac",          # the render function the executor calls
        "analyses": ["ac", "dc"],                 # "dc": the S3-inc2a swing/ICMR device-query sweeps
        "knobs": list(OTA_KNOB_ELEMENTS),          # {Cc, CL, VDD, IREFV} — what _OTA_BODY can sweep
        # explicit (NOT list(OTA_METRIC_MEAS)): the shared meas map also holds acl_db for the closed-
        # loop PGA, which the open-loop Miller OTA does not measure.
        "metrics": ["gbw_hz", "pm_deg", "av0_db", "vout_swing_v", "icmr_lo_v", "icmr_hi_v"],
        # S3-inc2a spec §4: declared RANGES for the two NEW knobs only (Cc/CL stay unranged — additive,
        # no clamping where a template doesn't declare a range). 1.4-2.2V is a sane sky130 1.8V-nominal
        # headroom band; 2u-40u brackets the validated bias sweeps (probe: scratchpad/probe_11_irefv_
        # trend.py) without licensing an absurd authored point (the carried "VDD 0->10V" risk, spec §4).
        "knob_ranges": {"VDD": (1.4, 2.2), "IREFV": (2e-6, 4e-5)},
    },
    "current_mirror_simple_nmos": {
        "name": "current_mirror_simple_nmos",
        "description": "simple NMOS current mirror: diode-connected reference + output mirror device",
        "devices": ["nfet"],
        "default_sizing": {"VDD": "1.8", "W": "4", "Lp": "0.5", "IREFV": "10u", "VOUT": "0.9"},
        "template_ref": "current_mirror_dc",
        "analyses": ["dc"],
        "knobs": ["Vout"],                         # the swept output-voltage source
        "metrics": ["iout_a"],                     # mirrored output current
    },
    "common_source_active_load_nmos": {
        "name": "common_source_active_load_nmos",
        "description": "NMOS common-source gain stage with a PMOS current-mirror active load",
        "devices": ["nfet", "pfet"],
        "default_sizing": {"VDD": "1.8", "Wn": "8", "Wp": "16", "Lp": "0.5", "IREFV": "10u", "CL": "1p"},
        "template_ref": "common_source_ac",
        "analyses": ["ac"],
        "knobs": ["Iref"],                         # the swept bias-current source
        "metrics": ["av0_db", "gbw_hz"],           # DC gain + unity-gain frequency (reuse OTA meas)
    },
    "common_gate_nmos": {
        "name": "common_gate_nmos",
        "description": "NMOS common-gate current buffer (low input resistance Rin ~ 1/gm)",
        "devices": ["nfet"],
        "default_sizing": {"VDD": "1.8", "VG": "0.9", "Wn": "8", "Lp": "0.5", "RL": "10k", "IB": "10u"},
        "template_ref": "common_gate_ac",
        "analyses": ["ac"],
        "knobs": ["Iref"],                         # the swept bias-current source
        "metrics": ["rin_ohm"],                    # source input resistance (the defining CG property)
    },
    "source_follower_nmos": {
        "name": "source_follower_nmos",
        "description": "NMOS source follower — unity-gain voltage buffer (CIS pixel source follower)",
        "devices": ["nfet"],
        "default_sizing": {"VDD": "1.8", "VG": "0.9", "Wn": "8", "Lp": "0.5", "IB": "10u", "CL": "1p"},
        "template_ref": "source_follower_ac",
        "analyses": ["ac"],
        "knobs": ["Iref"],                         # the swept tail-current source
        "metrics": ["av0_db"],                     # the sub-unity voltage gain (reuse OTA meas)
    },
    "diff_pair_resistive_nmos": {
        "name": "diff_pair_resistive_nmos",
        "description": "resistively-loaded NMOS differential pair (differential gain Adm = gm*RD)",
        "devices": ["nfet"],
        "default_sizing": {"VDD": "1.8", "VCM": "0.9", "Wn": "8", "Lp": "0.5", "RD": "20k", "IB": "20u"},
        "template_ref": "diff_pair_ac",
        "analyses": ["ac"],
        "knobs": ["Iref"],                         # the swept tail-current source
        "metrics": ["adm_db"],                     # differential-mode voltage gain
    },
    "cascode_current_mirror_nmos": {
        "name": "cascode_current_mirror_nmos",
        "description": "cascode NMOS current mirror — boosted output resistance Rout ~ gm*ro^2",
        "devices": ["nfet"],
        "default_sizing": {"VDD": "1.8", "W": "4", "Lp": "0.5", "IREFV": "10u", "VOUT": "1.0"},
        "template_ref": "cascode_mirror_dc",
        "analyses": ["dc"],
        "knobs": ["Vout"],                         # the swept output-voltage source
        "metrics": ["iout_a"],                     # mirrored current (flat -> high Rout)
    },
    "ota_5t_nmos_in": {
        "name": "ota_5t_nmos_in",
        "description": "5-transistor single-stage OTA (active-mirror NMOS differential pair)",
        "devices": ["nfet", "pfet"],
        "default_sizing": {"VDD": "1.8", "VCM": "0.9", "W1": "8", "W3": "8", "W5": "8", "W5b": "4",
                           "Lp": "0.5", "IREFV": "10u", "CL": "1p"},
        "template_ref": "ota_5t_ac",
        "analyses": ["ac", "dc"],                  # "dc": the S3-inc2a swing/ICMR device-query sweeps
        "knobs": ["CL", "VDD", "IREFV"],            # + the S3-inc2a headroom knobs (shared element map)
        "metrics": ["av0_db", "gbw_hz", "vout_swing_v", "icmr_lo_v", "icmr_hi_v"],
        # same declared ranges as the Miller OTA (S3-inc2a spec §4) — see that entry's comment.
        "knob_ranges": {"VDD": (1.4, 2.2), "IREFV": (2e-6, 4e-5)},
    },
    "telescopic_cascode_ota_nmos_in": {
        "name": "telescopic_cascode_ota_nmos_in",
        "description": "telescopic cascode OTA: NMOS input pair + NMOS+PMOS cascodes (Rout ~ gm*ro^2)",
        "devices": ["nfet", "pfet"],
        "default_sizing": {"VDD": "1.8", "VCM": "0.9", "VBNC": "1.1", "VBPC": "0.3", "WN": "8",
                           "WP": "16", "WT": "16", "Lp": "0.5", "IREFV": "20u", "IREFP": "10u", "CL": "1p"},
        "template_ref": "telescopic_cascode_ac",
        "analyses": ["ac"],
        "knobs": ["CL"],                           # the swept load capacitor
        "metrics": ["av0_db", "gbw_hz"],           # cascode-boosted DC gain + unity-gain frequency
    },
    "folded_cascode_ota_nmos_in": {
        "name": "folded_cascode_ota_nmos_in",
        "description": "folded-cascode OTA: NMOS input pair, current folded down through PMOS cascodes "
                       "into an NMOS cascode-mirror load (wider swing / input CM range than telescopic)",
        "devices": ["nfet", "pfet"],
        "default_sizing": {"VDD": "1.8", "VCM": "0.9", "VBNC": "1.0", "VBPC": "0.75", "WN": "8",
                           "WP": "16", "WT": "16", "Lp": "0.5", "IREFV": "20u", "IREFP": "12u", "CL": "1p"},
        "template_ref": "folded_cascode_ac",
        "analyses": ["ac"],
        "knobs": ["CL"],                           # the swept load capacitor
        "metrics": ["av0_db", "gbw_hz"],           # cascode-boosted DC gain + unity-gain frequency
    },
    "regulated_cascode_nmos": {
        "name": "regulated_cascode_nmos",
        "description": "regulated (gain-boosted) cascode current source: an auxiliary amplifier holds "
                       "the cascode source constant, boosting Rout to ~gm*ro^2*A_aux",
        "devices": ["nfet", "pfet"],
        "default_sizing": {"VDD": "1.8", "W": "4", "WA": "4", "Lp": "0.5", "IREFV": "10u",
                           "VBA": "0.8", "VOUT": "1.0"},
        "template_ref": "regulated_cascode_dc",
        "analyses": ["dc"],
        "knobs": ["Vout"],                         # the swept output-voltage source
        "metrics": ["iout_a"],                     # output current (flat -> boosted Rout)
    },
    "comparator_continuous_nmos": {
        "name": "comparator_continuous_nmos",
        "description": "continuous-time comparator (high-gain open-loop 5T core) for the CIS column "
                       "single-slope ADC; propagation delay vs input overdrive",
        "devices": ["nfet", "pfet"],
        "default_sizing": {"VDD": "1.8", "VREF": "0.9", "VLO": "0.6", "W1": "8", "W3": "8", "W5": "8",
                           "W5b": "4", "Lp": "0.5", "IREFV": "20u", "CL": "200f"},
        "template_ref": "comparator_tran",
        "analyses": ["tran"],                      # NEW analysis primitive: time-domain step response
        "knobs": ["Vov"],                          # input overdrive (step amplitude above the trip)
        "metrics": ["tpd_s"],                      # propagation delay (input-cross -> output-decision)
    },
    "cds_switched_cap_nmos": {
        "name": "cds_switched_cap_nmos",
        "description": "correlated double sampling (switched-cap, nfet sampling switch): samples reset "
                       "then signal so a common pedestal (offset / FPN) cancels at the output",
        "devices": ["nfet"],
        "default_sizing": {"VDD": "1.8", "VCM": "0.9", "VPED": "0.9", "DIFF": "0.2", "CS": "1000f",
                           "WSW": "4", "Lp": "0.15"},
        "template_ref": "cds_tran",
        "analyses": ["tran"],                      # time-domain: clock the sampling switch, sample-hold
        "knobs": ["Vped"],                         # the input pedestal (common offset) being swept
        "metrics": ["vo_v"],                       # held output (invariant to pedestal -> FPN rejected)
    },
    "single_slope_ramp_generator": {
        "name": "single_slope_ramp_generator",
        "description": "single-slope-ADC ramp generator: a PMOS current source charges a capacitor "
                       "into a linear ramp (slope = I/C); the global ramp reference for column ADCs",
        "devices": ["nfet", "pfet"],
        "default_sizing": {"VDD": "1.8", "IREFV": "10u", "CR": "500f", "WP": "8", "WS": "4", "Lp": "0.5"},
        "template_ref": "ramp_tran",
        "analyses": ["tran"],                      # time-domain: charge the cap, measure the ramp slope
        "knobs": ["Iref"],                         # the charging-current reference
        "metrics": ["slope_vps"],                  # ramp slope dV/dt (= I/Cramp)
    },
    "column_pga_inverting_nmos": {
        "name": "column_pga_inverting_nmos",
        "description": "programmable-gain column amplifier: a 5T OTA in inverting resistive feedback; "
                       "closed-loop gain = Rf/Rin set by the feedback ratio (programmable gain)",
        "devices": ["nfet", "pfet"],
        "default_sizing": {"VDD": "1.8", "VCM": "0.9", "W1": "8", "W3": "8", "W5": "8", "W5b": "4",
                           "Lp": "0.5", "IREFV": "10u", "CL": "500f", "RIN": "500k", "RF": "1000k"},
        "template_ref": "column_pga_ac",
        "analyses": ["ac"],
        "knobs": ["Rf"],                           # the feedback resistor (programs the gain)
        "metrics": ["acl_db"],                     # closed-loop gain (= Rf/Rin)
    },
    "ptat_ctat_core_bjt": {
        "name": "ptat_ctat_core_bjt",
        "description": "dVBE reference core: two IDENTICAL sky130 parasitic-PNP diodes at a 1:N "
                       "CURRENT ratio (not an emitter-area ratio — set by a 3-leg PMOS mirror's `m=` "
                       "device multiplicity, Is-independent by construction), with vbe_v = the "
                       "reference diode's own VBE (CTAT voltage, at an essentially fixed bias current) "
                       "and iptat_a = the derived (dVBE)/R PTAT-like current preview — the bandgap "
                       "family's smallest member (TOPOLOGY_BACKLOG.md #22/#23; no amp-in-loop, "
                       "sky130-only for now; REWORKED post-review from an emitter-area-ratio design "
                       "that failed its own hand-anchor — see templates.py's module comment)",
        "devices": ["pnp_bjt", "pfet"],
        # kept as a literal (not a reference to DEFAULT_PTAT_SIZING, defined far below this dict) —
        # same convention every other entry in this TEMPLATES dict already uses.
        "default_sizing": {"VDD": "1.8", "IBIAS": "2u", "R": "2k", "Wp": "8", "Lp": "0.5", "N": "5"},
        "template_ref": "ptat_ctat_core_bjt",
        "analyses": ["dc"],                        # native `.dc temp` sweep — temp IS the knob
        "knobs": ["temp"],
        "knob_ranges": {"temp": (-40.0, 125.0)},    # model validity envelope — the enforce clamp
                                                     # drops authored points outside (spec §4)
        "metrics": ["vbe_v", "iptat_a"],            # CTAT voltage / derived PTAT-like current
    },
}


# ===================================================================================================
# S3-inc2a §2 — vout_swing_v / icmr_lo_v / icmr_hi_v (shared by miller_ota_2stage_nmos_in and
# ota_5t_nmos_in). Hand-anchored + probe-validated live against real sky130 BEFORE this code existed
# (scratchpad/probe_swing_*.py, probe_icmr_*.py) — the discipline the spec's HAND-ANCHOR rule requires:
#
#   vout_swing_v = (VDD - Vdsat(top output device)) - Vdsat(bottom output device)[- v(tail)], read via
#   ngspice's hierarchical `@m.<inst>.<model>[vdsat]` device query (the E3 gm2 probe's own precedent).
#   Miller OTA: top=M6 (PMOS CS), bottom=M7 (NMOS sink) — BOTH rail-referenced (M7's source is literal
#   ground), so no tail term. 5T OTA: top=M4 (PMOS mirror), bottom=M2 (NMOS diff-pair leg, source=tail,
#   NOT ground) — the `- v(tail)` correction is required (probe #1/#2: dropping it is off by ~13%).
#   Hand value @ DEFAULT sizing (probe #1, python arithmetic on live op-point device queries) vs the
#   SAME formula computed live inside a foreach+alter+op sweep (probe #2): miller 1.4609297 vs
#   1.46093 (5 sig figs); 5T 1.38411531 vs 1.38412 (5 sig figs) — both templates match to their probes'
#   own rounding, confirming the formula (not just "a" formula) is what the deck computes.
#
#   icmr_lo_v / icmr_hi_v = the VCM boundary where a v1 criterion (device VDS margin crossing zero)
#   first trips, found by a COARSE (0.05V step) inner scan over Vin — the SAME element the AC deck's
#   own DC self-bias loop already drives (Lfb/Cfb makes v(out)~v(vinp)~v(vinn) track together at DC, so
#   sweeping Vin's DC level genuinely sweeps the whole loop's input common-mode point — no separate
#   stimulus network needed). icmr_lo: first VCM (scanning up) where the TAIL device M5's VDS margin
#   (vds-vdsat) reaches >=0 (Razavi's classic tail-headroom lower bound). icmr_hi: first VCM where the
#   INPUT-PAIR device M1's VDS margin falls <0 (the classic input-pair/mirror-headroom upper bound).
#   M5/M1 carry the IDENTICAL ref designator on both templates (verified by inspection), so ONE control
#   block serves both. Hand value @ VDD=1.8 (probe #4, 0.01V-resolution refine) vs the SAME 0.05V-grid
#   "first crossing" rule computed live (probe #6/#9): miller icmr_lo/hi = 0.85/1.40 (fine crossing
#   ~0.81/~1.38, within one 0.05V grid step — the declared tolerance); 5T icmr_lo/hi = 0.80/1.45 (fine
#   crossing ~0.77/~1.40, same one-step tolerance). A crossing never found in-range echoes 0/0 (NaN),
#   which runner.parse_rdata drops (C1) rather than certifying an un-observed boundary.
#
#   ngspice control-script syntax notes the probes pinned (probe #7/#8, a real live gotcha): equality
#   in an `if` condition is `=`, NOT `==` (the latter is a silent PARSE ERROR that aborts just that
#   line, easy to miss without adversarial verification against real ngspice).
# ===================================================================================================
_SWING_ICMR_METRICS = frozenset({"vout_swing_v", "icmr_lo_v", "icmr_hi_v"})

# Fixed inner VCM scan grid (0.05V steps, 0.05..2.55V) — literal (ngspice `foreach` takes a token list,
# not a runtime-computed range), and deliberately INDEPENDENT of the outer knob's current altered value
# so one grid serves every outer sweep point without per-point range math. 2.55V comfortably covers
# every VDD point any sane authored sweep would pick (the declared knob_ranges cap VDD at 2.2V).
_VCM_SCAN_POINTS: list[str] = [f"{0.05 + i * 0.05:.2f}" for i in range(51)]

# Per-knob defaults for the two new metrics' OUTER sweep (probe-validated ranges/trends: swing/icmr
# both move sensibly over these — see the module comment above and probe_swing_2_*.py / probe_11_*.py).
_SWING_ICMR_DEFAULT_POINTS: dict[str, list[str]] = {
    "VDD": ["1.6", "1.8", "2.0", "2.2"],
    "IREFV": ["5u", "10u", "20u", "40u"],
}


def _swing_icmr_default_points(knob: str, fallback: list[str]) -> list[str]:
    return _SWING_ICMR_DEFAULT_POINTS.get(knob, fallback)


def _swing_control_block(
    element: str, knob: str, points: list[str], top_device: str, bottom_device: str,
    tail_term: str = "",
) -> str:
    """`.control` body for vout_swing_v — see the module comment block above for the hand-anchored
    formula + probe transcript. `tail_term` is `" - v(tail)"` for the 5T OTA (bottom device sits on the
    tail node, not ground) and `""` for the 2-stage Miller OTA (both output devices are rail-referenced)."""
    foreach = " ".join(points)
    return (
        ".control\n"
        f"foreach pt {foreach}\n"
        f"  alter {element} = $pt\n"
        "  op\n"
        f"  let vdsat_top = @m.{top_device}[vdsat]\n"
        f"  let vdsat_bot = @m.{bottom_device}[vdsat]\n"
        f"  let y = (v(vdd) - vdsat_top - vdsat_bot){tail_term}\n"
        f"  echo RDATA {knob.lower()} $pt $&y\n"
        "end\n"
        ".endc"
    )


def _icmr_control_block(
    element: str, knob: str, points: list[str], metric: str,
    tail_device: str = "xm5.msky130_fd_pr__nfet_01v8",
    input_device: str = "xm1.msky130_fd_pr__nfet_01v8",
) -> str:
    """`.control` body for icmr_lo_v / icmr_hi_v — see the module comment block above for the v1
    criterion + probe transcript. Shared by both templates: M5 (tail current sink) and M1 (the first
    input-pair device) carry the identical ref designator on miller_ota_2stage_nmos_in AND
    ota_5t_nmos_in (verified by inspection of both bodies)."""
    foreach_outer = " ".join(points)
    vcm_pts = " ".join(_VCM_SCAN_POINTS)
    out_var = "icmr_lo" if metric == "icmr_lo_v" else "icmr_hi"
    return (
        ".control\n"
        f"foreach pt {foreach_outer}\n"
        f"  alter {element} = $pt\n"
        "  let icmr_lo = -1\n"
        "  let icmr_hi = -1\n"
        "  let found_lo = 0\n"
        "  let found_hi = 0\n"
        f"  foreach vcm {vcm_pts}\n"
        "    alter Vin = $vcm\n"
        "    op\n"
        f"    let vdsm5 = @m.{tail_device}[vds]\n"
        f"    let vdsatm5 = @m.{tail_device}[vdsat]\n"
        "    let margin5 = vdsm5 - vdsatm5\n"
        f"    let vdsm1 = @m.{input_device}[vds]\n"
        f"    let vdsatm1 = @m.{input_device}[vdsat]\n"
        "    let margin1 = vdsm1 - vdsatm1\n"
        "    if margin5 >= 0 & found_lo = 0\n"
        "      let icmr_lo = $vcm\n"
        "      let found_lo = 1\n"
        "    end\n"
        "    if margin1 < 0 & found_hi = 0\n"
        "      let icmr_hi = $vcm\n"
        "      let found_hi = 1\n"
        "    end\n"
        "  end\n"
        "  if found_lo = 0\n"
        "    let icmr_lo = 0/0\n"
        "  end\n"
        "  if found_hi = 0\n"
        "    let icmr_hi = 0/0\n"
        "  end\n"
        f"  echo RDATA {knob.lower()} $pt $&{out_var}\n"
        "end\n"
        ".endc"
    )


def _param_lines(sizing: dict) -> str:
    s = {**DEFAULT_OTA_SIZING, **{k: str(v) for k, v in (sizing or {}).items()}}
    items = " ".join(f"{k}={s[k]}" for k in _PARAM_ORDER)
    return f".param {items}"


def render_miller_ota_ac(
    sizing: dict | None = None,
    corner: str = "tt",
    knob: str = "Cc",
    metric: str = "gbw_hz",
    points: list[str] | None = None,
) -> str:
    """Render the open-loop AC deck for a knob-sweep, emitting machine-parseable `RDATA`
    lines (one per sweep point: `RDATA <series> <x_raw> <y>`). Default = the validated
    Cc sweep that produced the committed cc series.

    S3-inc2a §2: `metric` in {vout_swing_v, icmr_lo_v, icmr_hi_v} branches to the device-query DC
    sweep (see the module comment block above `_swing_control_block`/`_icmr_control_block`) instead
    of the AC deck below — every EXISTING metric's rendering is byte-unchanged."""
    if metric in _SWING_ICMR_METRICS:
        element = OTA_KNOB_ELEMENTS.get(knob, knob)
        pts = points or _swing_icmr_default_points(knob, ["250f", "500f", "1000f", "2000f", "4000f"])
        if metric == "vout_swing_v":
            control = _swing_control_block(
                element, knob, pts,
                top_device="xm6.msky130_fd_pr__pfet_01v8",
                bottom_device="xm7.msky130_fd_pr__nfet_01v8",
            )
        else:
            control = _icmr_control_block(element, knob, pts, metric)
        return "\n".join([
            f'* T-template miller_ota_2stage_nmos_in ({metric}, {knob} sweep)',
            f'.lib "{LIB_PLACEHOLDER}" {corner}',
            _param_lines(sizing or {}),
            _OTA_BODY,
            control,
            ".end",
            "",
        ])

    # per-knob defaults: VDD/IREFV are legal AC-path knobs since S3-inc2a — the capacitor-
    # valued fallback must never be altered onto a voltage/bias source (verified live:
    # alter VDD = 250f renders a dead operating point).
    points = points or _swing_icmr_default_points(knob, ["250f", "500f", "1000f", "2000f", "4000f"])
    element = OTA_KNOB_ELEMENTS.get(knob, knob)
    meas = OTA_METRIC_MEAS[metric]
    foreach = " ".join(points)
    control = (
        ".control\n"
        f"foreach pt {foreach}\n"
        f"  alter {element} = $pt\n"
        "  ac dec 40 1 10G\n"
        f"  {meas}\n"
        f"  echo RDATA {knob.lower()} $pt $&y\n"
        "end\n"
        ".endc"
    )
    return "\n".join([
        f'* T-template miller_ota_2stage_nmos_in (AC, {knob} sweep)',
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _param_lines(sizing or {}),
        _OTA_BODY,
        control,
        ".end",
        "",
    ])


def render_miller_ota_cell(sizing: dict | None = None, corner: str = "tt") -> str:
    """Render the bare cell (connectivity + sizing, no testbench) — the Specimen.netlist that
    is the corpus identity. The sweep control block lives separately as the testbench, so a
    specimen's identity is its circuit, not the analysis run against it."""
    return "\n".join([
        "* cell miller_ota_2stage_nmos_in",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _param_lines(sizing or {}),
        _OTA_BODY,
        ".end",
        "",
    ])


# ===================================================================================================
# E3-I1 INTERVENTION VARIANTS (docs/superpowers/specs/2026-07-04-e3-intervention-experiments.md §2-§3).
# Two hand-authored, probe-validated renderer bodies for the Miller OTA (baseline body above is
# BYTE-UNTOUCHED — see test_executable_interventions.py's regression pin). Each variant is registered
# under its OWN template_ref ("miller_ota_ac__<id>") in RENDERERS at the bottom of this file; the
# executor's 'intervention' sweep branch (executor.py) resolves BOTH the baseline template_ref (from
# the recipe's own topology_class/capability) and the variant template_ref (from the InterventionSpec
# registry, interventions.py) and runs a PAIRED sim with identical sizing. Variants deliberately do NOT
# get their own TEMPLATES entry — they are never a recipe's own topology_class, only reachable via an
# InterventionSpec, exactly as the spec's §2 "Mechanics" describes.
# ===================================================================================================

# ── intervention 'rz_null' — nulling resistor Rz in series with Ccomp, Rz itself sweepable. Physical
# variant (built FIRST per spec §3: lower convergence risk than an ideal-element intervention).
#
# Live-validated 2026-07-04 against real sky130 (probe: scratchpad/probe_rz_gm2.py + probe_rz_null.py):
#   gm2 (M6, the 2nd-stage PMOS CS device) measured via ngspice's hierarchical op-point access
#   `@m.xm6.msky130_fd_pr__pfet_01v8[gm]` at the nominal operating point: gm2 ~= 4.458e-4 S
#   (id2 ~= 51.17 uA) -> 1/gm2 ~= 2243 ohm -> the spec's "~2/gm2" sweep ceiling ~= 4486 ohm.
#
#   Rz=0 EXACTLY does NOT converge cleanly on this netlist: ngspice clamps a literal 0-ohm resistor to
#   1e-12 ohm, and dynamic-gmin / true-gmin / source stepping ALL fail, landing on a degenerate
#   transient-op fallback far from the real bias point (v(o1) collapsed to ~0V vs the true ~0.618V) —
#   this is why the sane sweep floor is "0-ish" (spec's own wording), not literal 0. Rz=10 ohm (a small
#   fraction of 1/gm2) reproduces the baseline (Rz-absent) operating point to 4 significant figures:
#   v(o1)=6.184011e-01 (baseline cell, no Rz at all) vs v(o1)=6.184e-01 at Rz=10 in this variant.
#
#   PM vs Rz @ nominal Cc=1p/CL=2p (points 10/1000/2000/3000/4500 ohm): 34.354 -> 43.251 -> 51.729 ->
#   59.746 -> 70.925 deg — monotonically RISING (the classic nulling-resistor RHP-zero cancellation:
#   pushing Rz toward 1/gm2 cancels the zero; past it, the zero moves into the LHP and adds phase
#   LEAD). Av0 stays flat at 68.8739 dB at EVERY Rz point (identical to the true Rz-absent baseline
#   68.8739 dB) — Rz lives only in the AC compensation path, so the DC operating point is untouched,
#   confirmed live (not merely assumed).
DEFAULT_OTA_RZ_SIZING: dict[str, str] = {**DEFAULT_OTA_SIZING, "Rz": "10"}
_OTA_RZ_PARAM_ORDER = [*_PARAM_ORDER, "Rz"]
OTA_RZ_KNOB_ELEMENTS: dict[str, str] = {**OTA_KNOB_ELEMENTS, "Rz": "Rz"}

# Identical to _OTA_BODY except Ccomp now drives an intermediate node `nrz`, bridged to o1 through the
# new Rz resistor (Ccomp out nrz {Cc}; Rz nrz o1 {Rz}) — Rz=0-ish recovers the baseline connectivity.
_OTA_RZ_BODY = """VDD vdd 0 {VDD}
Vin vinn 0 DC {VCM} AC 1
Lfb out vinp 1T
Cfb vinp 0 1T
IREF vdd nbias {IREFV}
XM8 nbias nbias 0 0 sky130_fd_pr__nfet_01v8 W={W8} L={Lp}
XM5 tail nbias 0 0 sky130_fd_pr__nfet_01v8 W={W5} L={Lp}
XM7 out  nbias 0 0 sky130_fd_pr__nfet_01v8 W={W7} L={Lp}
XM1 o1n vinp tail 0 sky130_fd_pr__nfet_01v8 W={W1} L={Lp} m={M1}
XM2 o1  vinn tail 0 sky130_fd_pr__nfet_01v8 W={W1} L={Lp} m={M1}
XM3 o1n o1n vdd vdd sky130_fd_pr__pfet_01v8 W={W3} L={Lp}
XM4 o1  o1n vdd vdd sky130_fd_pr__pfet_01v8 W={W3} L={Lp}
XM6 out o1 vdd vdd sky130_fd_pr__pfet_01v8 W={W6} L={Lp}
Ccomp out nrz {Cc}
Rz nrz o1 {Rz}
CL out 0 {CL}"""


def _rz_param_lines(sizing: dict) -> str:
    s = {**DEFAULT_OTA_RZ_SIZING, **{k: str(v) for k, v in (sizing or {}).items()}}
    items = " ".join(f"{k}={s[k]}" for k in _OTA_RZ_PARAM_ORDER)
    return f".param {items}"


def render_miller_ota_rz_null_ac(
    sizing: dict | None = None,
    corner: str = "tt",
    knob: str = "Rz",
    metric: str = "pm_deg",
    points: list[str] | None = None,
) -> str:
    """VARIANT template_ref 'miller_ota_ac__rz_null' — see the module-level comment block above for
    the live-validation transcript. Default sweep is the probe-validated Rz range: 10 ohm ("0-ish" —
    the numerically-safe floor, not literal 0, see comment above) to 4500 ohm (~2/gm2)."""
    points = points or ["10", "1000", "2000", "3000", "4500"]
    element = OTA_RZ_KNOB_ELEMENTS.get(knob, knob)
    meas = OTA_METRIC_MEAS[metric]
    foreach = " ".join(points)
    control = (
        ".control\n"
        f"foreach pt {foreach}\n"
        f"  alter {element} = $pt\n"
        "  ac dec 40 1 10G\n"
        f"  {meas}\n"
        f"  echo RDATA {knob.lower()} $pt $&y\n"
        "end\n"
        ".endc"
    )
    return "\n".join([
        f'* T-template miller_ota_2stage_nmos_in__rz_null (AC, {knob} sweep) — intervention variant',
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _rz_param_lines(sizing or {}),
        _OTA_RZ_BODY,
        control,
        ".end",
        "",
    ])


def render_miller_ota_rz_null_cell(sizing: dict | None = None, corner: str = "tt") -> str:
    """Bare cell for the rz_null variant — its OWN Specimen.netlist identity (kept separate from the
    baseline miller_ota_2stage_nmos_in cell; the two never merge onto one specimen)."""
    return "\n".join([
        "* cell miller_ota_2stage_nmos_in__rz_null",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _rz_param_lines(sizing or {}),
        _OTA_RZ_BODY,
        ".end",
        "",
    ])


# ── intervention 'ff_break' — ideal unity-gain E-source buffer severs the feedforward CURRENT path
# while preserving Miller pole-splitting as seen from o1. Idealized variant (built SECOND per spec §3).
#
# DESIGN NOTE (a rejected first attempt, kept here because it is the load-bearing lesson): buffering
# o1 itself (`Eff o1buf 0 o1 0 1; Ccomp out o1buf {Cc}`) was probed FIRST and REJECTED live — it
# isolates o1 from ANY of Cc's loading current (the E-source's control input draws none), which
# REMOVES Miller pole-splitting entirely (o1's own pole reverts to its fast, natural frequency) rather
# than preserving it. Symptom: wildly out-of-range "PM" readings (280-330 deg) traced to ngspice's
# vp() phase wrapping at +/-pi once the loop's true unwrapped phase lag exceeded 180 deg at the (much
# higher, now-unstable-adjacent) unity-gain crossover — a materially DIFFERENT loop, not just a faster
# one. The CORRECT placement buffers the OUTPUT node instead: `Eff outbuf 0 out 0 1` (outbuf tracks
# "out" with zero output impedance) and `Ccomp outbuf o1 {Cc}` (Cc now runs from o1 to the buffered
# replica, not to the real output). From o1's perspective this is IDENTICAL to the baseline (Cc's far
# terminal still tracks "out"'s voltage exactly), so Miller multiplication onto o1 is preserved; but
# the CURRENT that Cc carries terminates at the ideal buffer's output, never reaching the real "out"
# node's KCL — severing the direct feedforward path that (with the 2nd stage's inversion) produces the
# RHP zero.
#
# Live-validated 2026-07-04 against real sky130, sweeping the baseline's OWN Cc points on BOTH decks:
#   baseline   PM: 15.117 -> 23.747 -> 34.262 -> 45.400 -> 54.448 deg (Cc 250f..4000f)
#   ff_break   PM: 26.409 -> 39.523 -> 54.506 -> 68.552 -> 78.479 deg (SAME Cc points)
#   delta-PM (ff_break - baseline): +11.29 -> +15.78 -> +20.24 -> +23.15 -> +24.03 deg — POSITIVE and
#   GROWING with Cc, exactly the pre-registered Q2b prediction (the RHP zero z ~= gm2/Cc falls toward
#   GBW as Cc grows, so it costs the baseline more phase at larger Cc; ff_break has no such zero).
#   Av0 (both decks): 68.8739 dB at every Cc point — delta EXACTLY 0 (the intervention's own
#   DC-gain invariance control: severing the feedforward path must not touch DC gain).
_OTA_FF_BODY = """VDD vdd 0 {VDD}
Vin vinn 0 DC {VCM} AC 1
Lfb out vinp 1T
Cfb vinp 0 1T
IREF vdd nbias {IREFV}
XM8 nbias nbias 0 0 sky130_fd_pr__nfet_01v8 W={W8} L={Lp}
XM5 tail nbias 0 0 sky130_fd_pr__nfet_01v8 W={W5} L={Lp}
XM7 out  nbias 0 0 sky130_fd_pr__nfet_01v8 W={W7} L={Lp}
XM1 o1n vinp tail 0 sky130_fd_pr__nfet_01v8 W={W1} L={Lp} m={M1}
XM2 o1  vinn tail 0 sky130_fd_pr__nfet_01v8 W={W1} L={Lp} m={M1}
XM3 o1n o1n vdd vdd sky130_fd_pr__pfet_01v8 W={W3} L={Lp}
XM4 o1  o1n vdd vdd sky130_fd_pr__pfet_01v8 W={W3} L={Lp}
XM6 out o1 vdd vdd sky130_fd_pr__pfet_01v8 W={W6} L={Lp}
Eff outbuf 0 out 0 1
Ccomp outbuf o1 {Cc}
CL out 0 {CL}"""


def render_miller_ota_ff_break_ac(
    sizing: dict | None = None,
    corner: str = "tt",
    knob: str = "Cc",
    metric: str = "gbw_hz",
    points: list[str] | None = None,
) -> str:
    """VARIANT template_ref 'miller_ota_ac__ff_break' — see the module-level comment block above for
    the live-validation transcript. Reuses the baseline's OWN sizing/knob/param plumbing (the buffer's
    unity gain is a netlist literal, not a `.param`) so the paired baseline/variant sweep in
    executor.py's 'intervention' branch renders both decks off the IDENTICAL sizing dict + points."""
    # per-knob defaults: VDD/IREFV are legal AC-path knobs since S3-inc2a — the capacitor-
    # valued fallback must never be altered onto a voltage/bias source (verified live:
    # alter VDD = 250f renders a dead operating point).
    points = points or _swing_icmr_default_points(knob, ["250f", "500f", "1000f", "2000f", "4000f"])
    element = OTA_KNOB_ELEMENTS.get(knob, knob)
    meas = OTA_METRIC_MEAS[metric]
    foreach = " ".join(points)
    control = (
        ".control\n"
        f"foreach pt {foreach}\n"
        f"  alter {element} = $pt\n"
        "  ac dec 40 1 10G\n"
        f"  {meas}\n"
        f"  echo RDATA {knob.lower()} $pt $&y\n"
        "end\n"
        ".endc"
    )
    return "\n".join([
        f'* T-template miller_ota_2stage_nmos_in__ff_break (AC, {knob} sweep) — intervention variant',
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _param_lines(sizing or {}),
        _OTA_FF_BODY,
        control,
        ".end",
        "",
    ])


def render_miller_ota_ff_break_cell(sizing: dict | None = None, corner: str = "tt") -> str:
    """Bare cell for the ff_break variant — its OWN Specimen.netlist identity."""
    return "\n".join([
        "* cell miller_ota_2stage_nmos_in__ff_break",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _param_lines(sizing or {}),
        _OTA_FF_BODY,
        ".end",
        "",
    ])


# ── current_mirror_simple_nmos (the 2nd template — a different analysis (.dc/op) + metric (iout),
# proving the frozen core generalizes; the code form of the validated cm_simple.spice) ──

DEFAULT_CM_SIZING: dict[str, str] = {"VDD": "1.8", "W": "4", "Lp": "0.5", "IREFV": "10u", "VOUT": "0.9"}

# Simple NMOS mirror: diode-connected reference XM1 sets the gate bias; XM2 mirrors the current to
# the output node, swept by the Vout source. (Reference current set by IREF into the diode.)
_CM_SIMPLE_BODY = """VDD vdd 0 {VDD}
IREF vdd nd {IREFV}
XM1 nd nd 0 0 sky130_fd_pr__nfet_01v8 W={W} L={Lp}
XM2 out nd 0 0 sky130_fd_pr__nfet_01v8 W={W} L={Lp}
Vout out 0 DC {VOUT}"""

_CM_PARAM_ORDER = ["VDD", "W", "Lp", "IREFV", "VOUT"]
CM_KNOB_SOURCES = {"Vout": "Vout"}                  # knob -> the swept independent source
CM_METRIC_LET = {"iout_a": "let y = -i(vout)"}      # metric -> the measurement (binds `y`)


def _cm_param_lines(sizing: dict) -> str:
    s = {**DEFAULT_CM_SIZING, **{k: str(v) for k, v in (sizing or {}).items()}}
    return ".param " + " ".join(f"{k}={s[k]}" for k in _CM_PARAM_ORDER)


def render_current_mirror_dc(
    sizing: dict | None = None,
    corner: str = "tt",
    knob: str = "Vout",
    metric: str = "iout_a",
    points: list[str] | None = None,
) -> str:
    """Render the output-characteristic sweep: step Vout across the saturation region, measure the
    mirrored current via op at each point, emit `RDATA vout <V> <iout>`. Same foreach+op+let+echo
    pattern as the OTA (the validated batch-safe form), different analysis + metric — so the SAME
    runner/oracle/executor consume it. Default points sit above compliance (all in saturation)."""
    points = points or ["0.4", "0.7", "1.0", "1.3", "1.6"]
    src = CM_KNOB_SOURCES.get(knob, knob)
    let = CM_METRIC_LET[metric]
    foreach = " ".join(points)
    control = (
        ".control\n"
        f"foreach pt {foreach}\n"
        f"  alter {src} = $pt\n"
        "  op\n"
        f"  {let}\n"
        f"  echo RDATA {knob.lower()} $pt $&y\n"
        "end\n"
        ".endc"
    )
    return "\n".join([
        f"* T-template current_mirror_simple_nmos (DC, {knob} sweep)",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _cm_param_lines(sizing or {}),
        _CM_SIMPLE_BODY,
        control,
        ".end",
        "",
    ])


def render_current_mirror_cell(sizing: dict | None = None, corner: str = "tt") -> str:
    """Bare cell (connectivity + sizing) for the simple NMOS mirror — the Specimen.netlist identity."""
    return "\n".join([
        "* cell current_mirror_simple_nmos",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _cm_param_lines(sizing or {}),
        _CM_SIMPLE_BODY,
        ".end",
        "",
    ])


# ── common_source_active_load_nmos (3rd template — the gain primitive Razavi teaches first;
# NMOS common-source with a PMOS current-mirror active load, sweeping the bias current. Validated on
# sky130: Av0 ~37 dB (weakly dependent on Id — the textbook Av∝1/√Id is a long-channel law sky130
# short-channel softens), GBW ∝ ~√Id. Same .ac path + metrics as the OTA, different netlist + knob) ──

DEFAULT_CS_SIZING: dict[str, str] = {"VDD": "1.8", "Wn": "8", "Wp": "16", "Lp": "0.5", "IREFV": "10u", "CL": "1p"}

# NMOS common-source (M1) with a PMOS mirror active load (M3 diode reference -> M2 load). The bias
# current is set by IREF; the high-impedance output is self-biased to its operating point by a DC
# feedback inductor (Lfb out->g is a DC short / AC open), with the AC input coupled in through Cin.
_CS_BODY = """VDD vdd 0 {VDD}
XM3 pbias pbias vdd vdd sky130_fd_pr__pfet_01v8 W={Wp} L={Lp}
XM2 out   pbias vdd vdd sky130_fd_pr__pfet_01v8 W={Wp} L={Lp}
XM1 out   g 0 0 sky130_fd_pr__nfet_01v8 W={Wn} L={Lp}
IREF pbias 0 {IREFV}
Lfb out g 1T
Vin in 0 DC 0 AC 1
Cin in g 1T
CL out 0 {CL}"""

_CS_PARAM_ORDER = ["VDD", "Wn", "Wp", "Lp", "IREFV", "CL"]
CS_KNOB_ELEMENTS = {"Iref": "IREF"}             # knob -> the swept independent source (bias current)
# metric -> meas: reuses the OTA's (av0_db = DC gain, gbw_hz = unity-gain freq) — same vdb(out) probe.


def _cs_param_lines(sizing: dict) -> str:
    s = {**DEFAULT_CS_SIZING, **{k: str(v) for k, v in (sizing or {}).items()}}
    return ".param " + " ".join(f"{k}={s[k]}" for k in _CS_PARAM_ORDER)


def render_common_source_ac(
    sizing: dict | None = None,
    corner: str = "tt",
    knob: str = "Iref",
    metric: str = "av0_db",
    points: list[str] | None = None,
) -> str:
    """Render the open-loop AC deck for a bias-current sweep (op re-biases the self-biased output at
    each point), measuring av0_db / gbw_hz. RDATA per point. Default = the validated Iref sweep."""
    points = points or ["2u", "5u", "10u", "20u", "40u"]
    element = CS_KNOB_ELEMENTS.get(knob, knob)
    meas = OTA_METRIC_MEAS[metric]
    foreach = " ".join(points)
    control = (
        ".control\n"
        f"foreach pt {foreach}\n"
        f"  alter {element} = $pt\n"
        "  op\n"
        "  ac dec 40 1 10G\n"
        f"  {meas}\n"
        f"  echo RDATA {knob.lower()} $pt $&y\n"
        "end\n"
        ".endc"
    )
    return "\n".join([
        f"* T-template common_source_active_load_nmos (AC, {knob} sweep)",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _cs_param_lines(sizing or {}),
        _CS_BODY,
        control,
        ".end",
        "",
    ])


def render_common_source_cell(sizing: dict | None = None, corner: str = "tt") -> str:
    """Bare cell for the active-loaded common-source stage — the Specimen.netlist identity."""
    return "\n".join([
        "* cell common_source_active_load_nmos",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _cs_param_lines(sizing or {}),
        _CS_BODY,
        ".end",
        "",
    ])


# ── common_gate_nmos (4th template — the current buffer; defining property is a LOW input resistance
# Rin ≈ 1/gm. Drive the source with an AC current, measure v(source)=Rin, sweep the bias current.
# Validated sky130: Rin 18.5k->1.6k as Id 2u->40u (~0.83/gm, falls with bias). New metric rin_ohm) ──

DEFAULT_CG_SIZING: dict[str, str] = {"VDD": "1.8", "VG": "0.9", "Wn": "8", "Lp": "0.5", "RL": "10k", "IB": "10u"}

# NMOS with the gate AC-grounded (DC-biased by Vg); the source is the input, the drain drives RL.
# A DC tail current (Itail) sets Id; an AC current (Iin) probes the source so Rin = |v(s)| / |Iin|.
_CG_BODY = """VDD vdd 0 {VDD}
Vg g 0 {VG}
XM1 d g s 0 sky130_fd_pr__nfet_01v8 W={Wn} L={Lp}
RL vdd d {RL}
Itail s 0 {IB}
Iin s 0 AC 1"""

_CG_PARAM_ORDER = ["VDD", "VG", "Wn", "Lp", "RL", "IB"]
CG_KNOB_ELEMENTS = {"Iref": "Itail"}                            # knob -> the swept bias-current source
CG_METRIC_MEAS = {"rin_ohm": "let y = 0/0\n  meas ac y find vm(s) at=1"}   # Rin = |v(source)| / |Iin=1A|


def _cg_param_lines(sizing: dict) -> str:
    s = {**DEFAULT_CG_SIZING, **{k: str(v) for k, v in (sizing or {}).items()}}
    return ".param " + " ".join(f"{k}={s[k]}" for k in _CG_PARAM_ORDER)


def render_common_gate_ac(
    sizing: dict | None = None,
    corner: str = "tt",
    knob: str = "Iref",
    metric: str = "rin_ohm",
    points: list[str] | None = None,
) -> str:
    """Render the input-resistance sweep: step the bias current, op-rebias, and measure the source
    input resistance via a low-frequency AC current probe (Rin = |v(source)|, since Iin = 1 A)."""
    points = points or ["2u", "5u", "10u", "20u", "40u"]
    element = CG_KNOB_ELEMENTS.get(knob, knob)
    meas = CG_METRIC_MEAS[metric]
    foreach = " ".join(points)
    control = (
        ".control\n"
        f"foreach pt {foreach}\n"
        f"  alter {element} = $pt\n"
        "  op\n"
        "  ac dec 10 1 1k\n"
        f"  {meas}\n"
        f"  echo RDATA {knob.lower()} $pt $&y\n"
        "end\n"
        ".endc"
    )
    return "\n".join([
        f"* T-template common_gate_nmos (AC input-resistance, {knob} sweep)",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _cg_param_lines(sizing or {}),
        _CG_BODY,
        control,
        ".end",
        "",
    ])


def render_common_gate_cell(sizing: dict | None = None, corner: str = "tt") -> str:
    """Bare cell for the common-gate stage — the Specimen.netlist identity."""
    return "\n".join([
        "* cell common_gate_nmos",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _cg_param_lines(sizing or {}),
        _CG_BODY,
        ".end",
        "",
    ])


# ── source_follower_nmos (5th template — the voltage buffer / CIS pixel source follower; defining
# property is a near-unity, sub-unity voltage gain that is bias-independent. Drain at VDD, gate =
# input, source = output to a tail current. Validated sky130: Av ~ -1.6 dB (~0.83, sub-unity from
# body effect) and FLAT (<0.03 dB) over 20x bias — the buffer hallmark. Reuses the av0_db meas) ──

DEFAULT_SF_SIZING: dict[str, str] = {"VDD": "1.8", "VG": "0.9", "Wn": "8", "Lp": "0.5", "IB": "10u", "CL": "1p"}

_SF_BODY = """VDD vdd 0 {VDD}
XM1 vdd g out 0 sky130_fd_pr__nfet_01v8 W={Wn} L={Lp}
Itail out 0 {IB}
Vin g 0 DC {VG} AC 1
CL out 0 {CL}"""

_SF_PARAM_ORDER = ["VDD", "VG", "Wn", "Lp", "IB", "CL"]
SF_KNOB_ELEMENTS = {"Iref": "Itail"}            # knob -> the swept tail-current source
# metric -> meas: reuses the OTA's av0_db (= max vdb(out)); for a follower this reads the sub-unity gain.


def _sf_param_lines(sizing: dict) -> str:
    s = {**DEFAULT_SF_SIZING, **{k: str(v) for k, v in (sizing or {}).items()}}
    return ".param " + " ".join(f"{k}={s[k]}" for k in _SF_PARAM_ORDER)


def render_source_follower_ac(
    sizing: dict | None = None,
    corner: str = "tt",
    knob: str = "Iref",
    metric: str = "av0_db",
    points: list[str] | None = None,
) -> str:
    """Render the follower voltage-gain sweep: step the tail current, op-rebias, and read the
    (sub-unity) voltage gain av0_db = max vdb(out)."""
    points = points or ["2u", "5u", "10u", "20u", "40u"]
    element = SF_KNOB_ELEMENTS.get(knob, knob)
    meas = OTA_METRIC_MEAS[metric]
    foreach = " ".join(points)
    control = (
        ".control\n"
        f"foreach pt {foreach}\n"
        f"  alter {element} = $pt\n"
        "  op\n"
        "  ac dec 40 1 10G\n"
        f"  {meas}\n"
        f"  echo RDATA {knob.lower()} $pt $&y\n"
        "end\n"
        ".endc"
    )
    return "\n".join([
        f"* T-template source_follower_nmos (AC voltage gain, {knob} sweep)",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _sf_param_lines(sizing or {}),
        _SF_BODY,
        control,
        ".end",
        "",
    ])


def render_source_follower_cell(sizing: dict | None = None, corner: str = "tt") -> str:
    """Bare cell for the source-follower buffer — the Specimen.netlist identity."""
    return "\n".join([
        "* cell source_follower_nmos",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _sf_param_lines(sizing or {}),
        _SF_BODY,
        ".end",
        "",
    ])


# ── diff_pair_resistive_nmos (6th template — the differential transconductance Razavi opens the
# diff-amp chapter with; matched NMOS pair, resistive loads, tail current. Differential gain
# Adm = gm*RD rises with bias (gm ~ sqrt(Id)). Resistive loads set the DC point (no self-bias needed).
# Validated sky130: Adm 7.9 -> 19.6 dB as Itail 5u -> 80u. New metric adm_db = db|v(o1)-v(o2)|) ──

DEFAULT_DP_SIZING: dict[str, str] = {"VDD": "1.8", "VCM": "0.9", "Wn": "8", "Lp": "0.5", "RD": "20k", "IB": "20u"}

# Matched NMOS pair (M1,M2) with resistive loads RD1/RD2 and a tail current Itail. A differential AC
# input (Vinp +0.5, Vinn -0.5, so |vid|=1) lets the differential output gain read directly off o1,o2.
_DP_BODY = """VDD vdd 0 {VDD}
RD1 vdd o1 {RD}
RD2 vdd o2 {RD}
XM1 o1 vinp tail 0 sky130_fd_pr__nfet_01v8 W={Wn} L={Lp}
XM2 o2 vinn tail 0 sky130_fd_pr__nfet_01v8 W={Wn} L={Lp}
Itail tail 0 {IB}
Vinp vinp 0 DC {VCM} AC 0.5
Vinn vinn 0 DC {VCM} AC -0.5"""

_DP_PARAM_ORDER = ["VDD", "VCM", "Wn", "Lp", "RD", "IB"]
DP_KNOB_ELEMENTS = {"Iref": "Itail"}            # knob -> the swept tail-current source
# metric -> meas. No `let y = 0/0` reset prefix: 0/0 errors on a let-derived vector in this ngspice,
# and `max` over the finite db vector cannot fail — so the executor's point-count guard is the
# sufficient C1 protection here.
DP_METRIC_MEAS = {"adm_db": "let admdb = db(abs(v(o1)-v(o2)))\n  meas ac y max admdb"}


def _dp_param_lines(sizing: dict) -> str:
    s = {**DEFAULT_DP_SIZING, **{k: str(v) for k, v in (sizing or {}).items()}}
    return ".param " + " ".join(f"{k}={s[k]}" for k in _DP_PARAM_ORDER)


def render_diff_pair_ac(
    sizing: dict | None = None,
    corner: str = "tt",
    knob: str = "Iref",
    metric: str = "adm_db",
    points: list[str] | None = None,
) -> str:
    """Render the differential-gain sweep: step the tail current, op-rebias, and read the
    differential-mode gain Adm = db|v(o1)-v(o2)| (the differential AC input has |vid| = 1)."""
    points = points or ["5u", "10u", "20u", "40u", "80u"]
    element = DP_KNOB_ELEMENTS.get(knob, knob)
    meas = DP_METRIC_MEAS[metric]
    foreach = " ".join(points)
    control = (
        ".control\n"
        f"foreach pt {foreach}\n"
        f"  alter {element} = $pt\n"
        "  op\n"
        "  ac dec 40 1 10G\n"
        f"  {meas}\n"
        f"  echo RDATA {knob.lower()} $pt $&y\n"
        "end\n"
        ".endc"
    )
    return "\n".join([
        f"* T-template diff_pair_resistive_nmos (AC differential gain, {knob} sweep)",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _dp_param_lines(sizing or {}),
        _DP_BODY,
        control,
        ".end",
        "",
    ])


def render_diff_pair_cell(sizing: dict | None = None, corner: str = "tt") -> str:
    """Bare cell for the resistively-loaded differential pair — the Specimen.netlist identity."""
    return "\n".join([
        "* cell diff_pair_resistive_nmos",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _dp_param_lines(sizing or {}),
        _DP_BODY,
        ".end",
        "",
    ])


# ── cascode_current_mirror_nmos (7th template — the cascode Razavi uses to BOOST output resistance.
# Stacking a cascode device makes Rout ~ gm*ro^2 (here ~30M, ~35x a simple mirror), so the mirrored
# current is nearly independent of the output voltage. Same .dc/op + iout idiom as the simple mirror,
# different netlist; the distinctive claim is iout INVARIANCE (vs the simple mirror's rising iout).
# Validated sky130: iout flat 9.98-10.00uA over Vout 0.8-1.6 (cov ~0.2%)) ──

DEFAULT_CASC_SIZING: dict[str, str] = {"VDD": "1.8", "W": "4", "Lp": "0.5", "IREFV": "10u", "VOUT": "1.0"}

# Standard cascode NMOS mirror: a bottom mirror (M1 ref diode, M3 output) sets the current; a top
# cascode (M2 ref diode, M4 on the output, gate=nc) shields the output and boosts Rout to gm*ro^2.
_CASC_BODY = """VDD vdd 0 {VDD}
IREF vdd nc {IREFV}
XM2 nc nc nb 0 sky130_fd_pr__nfet_01v8 W={W} L={Lp}
XM1 nb nb 0 0 sky130_fd_pr__nfet_01v8 W={W} L={Lp}
XM4 out nc nx 0 sky130_fd_pr__nfet_01v8 W={W} L={Lp}
XM3 nx nb 0 0 sky130_fd_pr__nfet_01v8 W={W} L={Lp}
Vout out 0 {VOUT}"""

_CASC_PARAM_ORDER = ["VDD", "W", "Lp", "IREFV", "VOUT"]
# knob -> swept source (Vout) and metric -> let reuse the simple mirror's (CM_KNOB_SOURCES / CM_METRIC_LET).


def _casc_param_lines(sizing: dict) -> str:
    s = {**DEFAULT_CASC_SIZING, **{k: str(v) for k, v in (sizing or {}).items()}}
    return ".param " + " ".join(f"{k}={s[k]}" for k in _CASC_PARAM_ORDER)


def render_cascode_mirror_dc(
    sizing: dict | None = None,
    corner: str = "tt",
    knob: str = "Vout",
    metric: str = "iout_a",
    points: list[str] | None = None,
) -> str:
    """Render the output-characteristic sweep (Vout above the cascode compliance, where Rout is high),
    measuring the mirrored current per point. Default points sit in the high-Rout region."""
    points = points or ["0.8", "1.0", "1.2", "1.4", "1.6"]
    src = CM_KNOB_SOURCES.get(knob, knob)
    let = CM_METRIC_LET[metric]
    foreach = " ".join(points)
    control = (
        ".control\n"
        f"foreach pt {foreach}\n"
        f"  alter {src} = $pt\n"
        "  op\n"
        f"  {let}\n"
        f"  echo RDATA {knob.lower()} $pt $&y\n"
        "end\n"
        ".endc"
    )
    return "\n".join([
        f"* T-template cascode_current_mirror_nmos (DC, {knob} sweep)",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _casc_param_lines(sizing or {}),
        _CASC_BODY,
        control,
        ".end",
        "",
    ])


def render_cascode_mirror_cell(sizing: dict | None = None, corner: str = "tt") -> str:
    """Bare cell for the cascode NMOS mirror — the Specimen.netlist identity."""
    return "\n".join([
        "* cell cascode_current_mirror_nmos",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _casc_param_lines(sizing or {}),
        _CASC_BODY,
        ".end",
        "",
    ])


# ── ota_5t_nmos_in (8th template — the canonical single-stage OTA: NMOS diff pair + PMOS mirror load
# + NMOS tail = the active-mirror differential pair (5T OTA). Single-stage gain gm1*(ro2||ro4), output
# pole sets GBW = gm1/2*pi*CL. NOTE the DC self-bias feeds back to the INVERTING input (vinn) — a
# single stage is non-inverting from vinp, so feeding out->vinp would be POSITIVE feedback (it latches);
# the 2-stage OTA can feed out->vinp only because its 2nd stage inverts. Validated sky130: Av0 37.1 dB
# (flat vs CL), GBW 60->7.6 MHz as CL 500f->4p (∝1/CL). Reuses the OTA av0/gbw meas + the CL knob) ──

DEFAULT_OTA5T_SIZING: dict[str, str] = {
    "VDD": "1.8", "VCM": "0.9", "W1": "8", "W3": "8", "W5": "8", "W5b": "4", "Lp": "0.5",
    "IREFV": "10u", "CL": "1p",
}

# NMOS diff pair (M1=non-inv vinp, M2=inv vinn) + PMOS mirror load (M3 diode, M4 mirror -> single
# ended out at M2/M4 drain) + NMOS tail mirror (M5b ref, M5 tail). Open-loop AC via DC self-bias:
# Lfb out->vinn (negative fb for a 1-stage), Cfb AC-grounds vinn, AC input drives vinp.
_OTA5T_BODY = """VDD vdd 0 {VDD}
Vin vinp 0 DC {VCM} AC 1
Lfb out vinn 1T
Cfb vinn 0 1T
IREF vdd nbias {IREFV}
XM5b nbias nbias 0 0 sky130_fd_pr__nfet_01v8 W={W5b} L={Lp}
XM5 tail nbias 0 0 sky130_fd_pr__nfet_01v8 W={W5} L={Lp}
XM1 o1  vinp tail 0 sky130_fd_pr__nfet_01v8 W={W1} L={Lp}
XM2 out vinn tail 0 sky130_fd_pr__nfet_01v8 W={W1} L={Lp}
XM3 o1  o1 vdd vdd sky130_fd_pr__pfet_01v8 W={W3} L={Lp}
XM4 out o1 vdd vdd sky130_fd_pr__pfet_01v8 W={W3} L={Lp}
CL out 0 {CL}"""

_OTA5T_PARAM_ORDER = ["VDD", "VCM", "W1", "W3", "W5", "W5b", "Lp", "IREFV", "CL"]
# knob -> element and metric -> meas reuse the OTA's (OTA_KNOB_ELEMENTS[CL], OTA_METRIC_MEAS).


def _ota5t_param_lines(sizing: dict) -> str:
    s = {**DEFAULT_OTA5T_SIZING, **{k: str(v) for k, v in (sizing or {}).items()}}
    return ".param " + " ".join(f"{k}={s[k]}" for k in _OTA5T_PARAM_ORDER)


def render_ota_5t_ac(
    sizing: dict | None = None,
    corner: str = "tt",
    knob: str = "CL",
    metric: str = "av0_db",
    points: list[str] | None = None,
) -> str:
    """Render the open-loop AC deck for a CL sweep, measuring av0_db / gbw_hz (the single-stage gain
    and the output-pole-set unity-gain frequency).

    S3-inc2a §2: `metric` in {vout_swing_v, icmr_lo_v, icmr_hi_v} branches to the device-query DC
    sweep shared with the Miller OTA (see the module comment above `_swing_control_block` /
    `_icmr_control_block`) — every EXISTING metric's rendering below is byte-unchanged."""
    if metric in _SWING_ICMR_METRICS:
        element = OTA_KNOB_ELEMENTS.get(knob, knob)
        pts = points or _swing_icmr_default_points(knob, ["500f", "1000f", "2000f", "4000f"])
        if metric == "vout_swing_v":
            # M2 (bottom, NMOS diff-pair leg) sits on the TAIL node, not ground (unlike the Miller
            # OTA's rail-referenced M7) — probe #1/#2's tail_term correction is required here.
            control = _swing_control_block(
                element, knob, pts,
                top_device="xm4.msky130_fd_pr__pfet_01v8",
                bottom_device="xm2.msky130_fd_pr__nfet_01v8",
                tail_term=" - v(tail)",
            )
        else:
            control = _icmr_control_block(element, knob, pts, metric)
        return "\n".join([
            f"* T-template ota_5t_nmos_in ({metric}, {knob} sweep)",
            f'.lib "{LIB_PLACEHOLDER}" {corner}',
            _ota5t_param_lines(sizing or {}),
            _OTA5T_BODY,
            control,
            ".end",
            "",
        ])

    points = points or ["500f", "1000f", "2000f", "4000f"]
    element = OTA_KNOB_ELEMENTS.get(knob, knob)
    meas = OTA_METRIC_MEAS[metric]
    foreach = " ".join(points)
    control = (
        ".control\n"
        f"foreach pt {foreach}\n"
        f"  alter {element} = $pt\n"
        "  ac dec 40 1 10G\n"
        f"  {meas}\n"
        f"  echo RDATA {knob.lower()} $pt $&y\n"
        "end\n"
        ".endc"
    )
    return "\n".join([
        f"* T-template ota_5t_nmos_in (AC, {knob} sweep)",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _ota5t_param_lines(sizing or {}),
        _OTA5T_BODY,
        control,
        ".end",
        "",
    ])


def render_ota_5t_cell(sizing: dict | None = None, corner: str = "tt") -> str:
    """Bare cell for the 5T (active-mirror differential pair) OTA — the Specimen.netlist identity."""
    return "\n".join([
        "* cell ota_5t_nmos_in",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _ota5t_param_lines(sizing or {}),
        _OTA5T_BODY,
        ".end",
        "",
    ])


# ── telescopic_cascode_ota_nmos_in (9th template — Razavi Ch.9 deep-tier: the telescopic cascode OTA.
# NMOS input pair, NMOS cascodes (M1c/M2c) and PMOS cascodes (M3c/M4c) stacked, so Rout ~= (gm*ro^2)_n
# || (gm*ro^2)_p — the cascode boost makes DC gain MUCH higher than the simple-mirror 5T OTA (measured
# ~67 dB vs ~37 dB, same input pair) at the cost of output swing. Two design notes baked in from the
# measure-first probe: (1) the PMOS-top mirror reference is DECOUPLED into its own leg (IREFP+XMRP) so
# the signal branches are clean current sources — folding the reference into a signal branch starves
# the DC operating point (circular start). (2) The cascode gates ride SEPARATE tuned rails (VBNC/VBPC),
# NOT diode stacks: in sky130's 1.8 V budget a diode cascode burns a full Vgs (~0.8 V) of headroom and
# collapses the stack; a biased cascode burns only Vdsat. Single-stage, output-pole-limited: GBW =
# gm1/2*pi*CL. DC self-bias to the INVERTING input (vinn), same as the 5T OTA. Reuses OTA av0/gbw meas
# + the CL knob. Validated sky130: Av0 67.4 dB (flat vs CL), GBW 26.3->3.2 MHz as CL 500f->4p. ──

DEFAULT_TELE_SIZING: dict[str, str] = {
    "VDD": "1.8", "VCM": "0.9", "VBNC": "1.1", "VBPC": "0.3", "WN": "8", "WP": "16", "WT": "16",
    "Lp": "0.5", "IREFV": "20u", "IREFP": "10u", "CL": "1p",
}

_TELE_BODY = """VDD vdd 0 {VDD}
Vin vinp 0 DC {VCM} AC 1
Lfb out vinn 1T
Cfb vinn 0 1T
Vbnc gnc 0 {VBNC}
Vbpc gpc 0 {VBPC}
IREF vdd nbias {IREFV}
XM5b nbias nbias 0 0 sky130_fd_pr__nfet_01v8 W={WT} L={Lp}
XM5  tail nbias 0 0 sky130_fd_pr__nfet_01v8 W={WT} L={Lp}
IREFP vbp 0 {IREFP}
XMRP vbp vbp vdd vdd sky130_fd_pr__pfet_01v8 W={WP} L={Lp}
XM1 n1 vinp tail 0 sky130_fd_pr__nfet_01v8 W={WN} L={Lp}
XM2 n2 vinn tail 0 sky130_fd_pr__nfet_01v8 W={WN} L={Lp}
XM1c o1  gnc n1 0 sky130_fd_pr__nfet_01v8 W={WN} L={Lp}
XM2c out gnc n2 0 sky130_fd_pr__nfet_01v8 W={WN} L={Lp}
XM3  p3 vbp vdd vdd sky130_fd_pr__pfet_01v8 W={WP} L={Lp}
XM3c o1 gpc p3  vdd sky130_fd_pr__pfet_01v8 W={WP} L={Lp}
XM4  p4 vbp vdd vdd sky130_fd_pr__pfet_01v8 W={WP} L={Lp}
XM4c out gpc p4 vdd sky130_fd_pr__pfet_01v8 W={WP} L={Lp}
CL out 0 {CL}"""

_TELE_PARAM_ORDER = ["VDD", "VCM", "VBNC", "VBPC", "WN", "WP", "WT", "Lp", "IREFV", "IREFP", "CL"]
# knob -> element and metric -> meas reuse the OTA's (OTA_KNOB_ELEMENTS[CL], OTA_METRIC_MEAS).


def _tele_param_lines(sizing: dict) -> str:
    s = {**DEFAULT_TELE_SIZING, **{k: str(v) for k, v in (sizing or {}).items()}}
    return ".param " + " ".join(f"{k}={s[k]}" for k in _TELE_PARAM_ORDER)


def render_telescopic_cascode_ac(
    sizing: dict | None = None,
    corner: str = "tt",
    knob: str = "CL",
    metric: str = "av0_db",
    points: list[str] | None = None,
) -> str:
    """Render the open-loop AC deck for a CL sweep, measuring av0_db / gbw_hz (the cascode-boosted
    DC gain and the output-pole-set unity-gain frequency)."""
    points = points or ["500f", "1000f", "2000f", "4000f"]
    element = OTA_KNOB_ELEMENTS.get(knob, knob)
    meas = OTA_METRIC_MEAS[metric]
    foreach = " ".join(points)
    control = (
        ".control\n"
        f"foreach pt {foreach}\n"
        f"  alter {element} = $pt\n"
        "  ac dec 40 1 10G\n"
        f"  {meas}\n"
        f"  echo RDATA {knob.lower()} $pt $&y\n"
        "end\n"
        ".endc"
    )
    return "\n".join([
        f"* T-template telescopic_cascode_ota_nmos_in (AC, {knob} sweep)",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _tele_param_lines(sizing or {}),
        _TELE_BODY,
        control,
        ".end",
        "",
    ])


def render_telescopic_cascode_cell(sizing: dict | None = None, corner: str = "tt") -> str:
    """Bare cell for the telescopic cascode OTA — the Specimen.netlist identity."""
    return "\n".join([
        "* cell telescopic_cascode_ota_nmos_in",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _tele_param_lines(sizing or {}),
        _TELE_BODY,
        ".end",
        "",
    ])


# ── folded_cascode_ota_nmos_in (10th template — Razavi Ch.9: the folded-cascode OTA. NMOS input pair
# drains sit at the FOLD nodes (f1/f2), shared with PMOS top current sources; the signal current folds
# DOWN through PMOS cascodes (M3c/M4c) into a bottom NMOS cascode-mirror load. Single-ended output. The
# fold un-stacks the input device from the output cascode, so the input common-mode range and output
# swing are wider than the telescopic's — its raison d'etre. Single-stage, output-pole-limited (GBW =
# gm1/2*pi*CL). DC self-bias to the INVERTING input (vinn).
#
# ★ The load-bearing measure-first finding, baked in: the PMOS-top current IREFP must be only SLIGHTLY
# above the input-branch current (Itail/2), NOT equal to the tail. Over-biasing the top source (IREFP =
# Itail) makes it self-limit into TRIODE at a railed fold node (f1 -> VDD), which shunts the signal to
# VDD and collapses GBW ~30x (measured 0.3 MHz instead of 5 MHz) while DC gain still reads ~50 dB — a
# silent trap. With IREFP = Itail/2 + a small fold margin (here 12u vs a 20u tail = 10u/branch input),
# the top source sits in saturation and the fold works: Av0 71.3 dB (flat vs CL), GBW 10.3->1.3 MHz as
# CL 500f->4p. Same decoupled-reference + tuned-cascode-rail discipline as the telescopic. ──

DEFAULT_FC_SIZING: dict[str, str] = {
    "VDD": "1.8", "VCM": "0.9", "VBNC": "1.0", "VBPC": "0.75", "WN": "8", "WP": "16", "WT": "16",
    "Lp": "0.5", "IREFV": "20u", "IREFP": "12u", "CL": "1p",
}

_FC_BODY = """VDD vdd 0 {VDD}
Vin vinp 0 DC {VCM} AC 1
Lfb out vinn 1T
Cfb vinn 0 1T
Vbnc gnc 0 {VBNC}
Vbpc gpc 0 {VBPC}
IREF vdd nbias {IREFV}
XM5b nbias nbias 0 0 sky130_fd_pr__nfet_01v8 W={WT} L={Lp}
XM5  tail nbias 0 0 sky130_fd_pr__nfet_01v8 W={WT} L={Lp}
IREFP vbp 0 {IREFP}
XMRP vbp vbp vdd vdd sky130_fd_pr__pfet_01v8 W={WP} L={Lp}
XM3 f1 vbp vdd vdd sky130_fd_pr__pfet_01v8 W={WP} L={Lp}
XM4 f2 vbp vdd vdd sky130_fd_pr__pfet_01v8 W={WP} L={Lp}
XM1 f1 vinp tail 0 sky130_fd_pr__nfet_01v8 W={WN} L={Lp}
XM2 f2 vinn tail 0 sky130_fd_pr__nfet_01v8 W={WN} L={Lp}
XM3c o1  gpc f1 vdd sky130_fd_pr__pfet_01v8 W={WP} L={Lp}
XM4c out gpc f2 vdd sky130_fd_pr__pfet_01v8 W={WP} L={Lp}
XM6c o1  gnc nb1 0 sky130_fd_pr__nfet_01v8 W={WN} L={Lp}
XM6  nb1 nb1 0  0 sky130_fd_pr__nfet_01v8 W={WN} L={Lp}
XM7c out gnc nb2 0 sky130_fd_pr__nfet_01v8 W={WN} L={Lp}
XM7  nb2 nb1 0  0 sky130_fd_pr__nfet_01v8 W={WN} L={Lp}
CL out 0 {CL}"""

_FC_PARAM_ORDER = ["VDD", "VCM", "VBNC", "VBPC", "WN", "WP", "WT", "Lp", "IREFV", "IREFP", "CL"]
# knob -> element and metric -> meas reuse the OTA's (OTA_KNOB_ELEMENTS[CL], OTA_METRIC_MEAS).


def _fc_param_lines(sizing: dict) -> str:
    s = {**DEFAULT_FC_SIZING, **{k: str(v) for k, v in (sizing or {}).items()}}
    return ".param " + " ".join(f"{k}={s[k]}" for k in _FC_PARAM_ORDER)


def render_folded_cascode_ac(
    sizing: dict | None = None,
    corner: str = "tt",
    knob: str = "CL",
    metric: str = "av0_db",
    points: list[str] | None = None,
) -> str:
    """Render the open-loop AC deck for a CL sweep, measuring av0_db / gbw_hz (the cascode-boosted
    DC gain and the output-pole-set unity-gain frequency)."""
    points = points or ["500f", "1000f", "2000f", "4000f"]
    element = OTA_KNOB_ELEMENTS.get(knob, knob)
    meas = OTA_METRIC_MEAS[metric]
    foreach = " ".join(points)
    control = (
        ".control\n"
        f"foreach pt {foreach}\n"
        f"  alter {element} = $pt\n"
        "  ac dec 40 1 10G\n"
        f"  {meas}\n"
        f"  echo RDATA {knob.lower()} $pt $&y\n"
        "end\n"
        ".endc"
    )
    return "\n".join([
        f"* T-template folded_cascode_ota_nmos_in (AC, {knob} sweep)",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _fc_param_lines(sizing or {}),
        _FC_BODY,
        control,
        ".end",
        "",
    ])


def render_folded_cascode_cell(sizing: dict | None = None, corner: str = "tt") -> str:
    """Bare cell for the folded-cascode OTA — the Specimen.netlist identity."""
    return "\n".join([
        "* cell folded_cascode_ota_nmos_in",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _fc_param_lines(sizing or {}),
        _FC_BODY,
        ".end",
        "",
    ])


# ── regulated_cascode_nmos (11th template — Razavi Ch.9: gain-boosting in its purest, most-isolated
# form, the REGULATED CASCODE / RGC). An auxiliary amplifier (Mp/Mn, a PMOS-input common-source stage)
# senses the cascode source node ns (= the bottom device M1's drain) and drives the cascode gate g1c to
# hold ns CONSTANT regardless of the output voltage. M1 then sees a fixed Vds -> its ro is boosted by
# the aux gain -> Rout ~ gm*ro^2 * A_aux. Measured as a current source (.dc Vout sweep -> iout): the
# regulated cascode held iout flat to ~0.005% over Vout 0.8-1.6V vs ~1.4% for the same five transistors
# with the aux loop replaced by a fixed gate (a ~270x Rout boost — the gain-boosting payoff, made into
# a falsifiable A/B). The aux is a DC regulation loop only (no main signal loop), so there is no AC
# stability concern for the Rout measurement. Reuses the cascode-mirror DC infra (CM_KNOB_SOURCES /
# CM_METRIC_LET). ──

DEFAULT_RGC_SIZING: dict[str, str] = {
    "VDD": "1.8", "W": "4", "WA": "4", "Lp": "0.5", "IREFV": "10u", "VBA": "0.8", "VOUT": "1.0",
}

_RGC_BODY = """VDD vdd 0 {VDD}
IREF vdd vg {IREFV}
XM0 vg vg 0 0 sky130_fd_pr__nfet_01v8 W={W} L={Lp}
XM1 ns vg 0 0 sky130_fd_pr__nfet_01v8 W={W} L={Lp}
XM1c out g1c ns 0 sky130_fd_pr__nfet_01v8 W={W} L={Lp}
XMp g1c ns vdd vdd sky130_fd_pr__pfet_01v8 W={WA} L={Lp}
XMn g1c vba 0 0 sky130_fd_pr__nfet_01v8 W={WA} L={Lp}
Vba vba 0 {VBA}
Vout out 0 {VOUT}"""

_RGC_PARAM_ORDER = ["VDD", "W", "WA", "Lp", "IREFV", "VBA", "VOUT"]
# knob -> swept source (Vout) and metric -> let reuse the simple mirror's (CM_KNOB_SOURCES / CM_METRIC_LET).


def _rgc_param_lines(sizing: dict) -> str:
    s = {**DEFAULT_RGC_SIZING, **{k: str(v) for k, v in (sizing or {}).items()}}
    return ".param " + " ".join(f"{k}={s[k]}" for k in _RGC_PARAM_ORDER)


def render_regulated_cascode_dc(
    sizing: dict | None = None,
    corner: str = "tt",
    knob: str = "Vout",
    metric: str = "iout_a",
    points: list[str] | None = None,
) -> str:
    """Render the output-characteristic sweep: step Vout across the saturation region, measure iout via
    op at each point. The regulated cascode's iout is essentially Vout-independent (boosted Rout)."""
    points = points or ["0.8", "1.0", "1.2", "1.4", "1.6"]
    src = CM_KNOB_SOURCES.get(knob, knob)
    let = CM_METRIC_LET[metric]
    foreach = " ".join(points)
    control = (
        ".control\n"
        f"foreach pt {foreach}\n"
        f"  alter {src} = $pt\n"
        "  op\n"
        f"  {let}\n"
        f"  echo RDATA {knob.lower()} $pt $&y\n"
        "end\n"
        ".endc"
    )
    return "\n".join([
        f"* T-template regulated_cascode_nmos (DC, {knob} sweep)",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _rgc_param_lines(sizing or {}),
        _RGC_BODY,
        control,
        ".end",
        "",
    ])


def render_regulated_cascode_cell(sizing: dict | None = None, corner: str = "tt") -> str:
    """Bare cell for the regulated cascode — the Specimen.netlist identity."""
    return "\n".join([
        "* cell regulated_cascode_nmos",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _rgc_param_lines(sizing or {}),
        _RGC_BODY,
        ".end",
        "",
    ])


# ── comparator_continuous_nmos (12th template — the CIS column single-slope-ADC comparator, and the
# substrate's first TIME-DOMAIN (.tran) specimen). A high-gain open-loop 5T core (NMOS diff pair +
# PMOS active mirror, NO feedback, NO compensation): vinn = reference, vinp = a step that crosses the
# trip. The defining behaviour is propagation delay tpd FALLING as input overdrive rises (more
# overdrive -> faster output slew). The .tran sweep alters the input PULSE high level per overdrive
# point and measures input-crossing-to-output-decision delay. Validated sky130: trip 0.90 V, output
# swings ~0->1.2 V, tpd 9.2 ns @ 50 mV -> 3.4 ns @ 400 mV overdrive. The frozen core needs NO new
# plumbing for .tran: the runner parses RDATA and the oracle judges (knob->value) series regardless of
# analysis — a new analysis is just a new render + .meas, exactly as .dc/.ac already coexist. ──

DEFAULT_CMP_SIZING: dict[str, str] = {
    "VDD": "1.8", "VREF": "0.9", "VLO": "0.6", "W1": "8", "W3": "8", "W5": "8", "W5b": "4",
    "Lp": "0.5", "IREFV": "20u", "CL": "200f",
}

# VHI is the PULSE high level — a placeholder here; the .tran render OVERRIDES it per overdrive point
# via `alter @Vin[pulse]`. The body must still parse with a valid initial PULSE spec.
_CMP_BODY = """VDD vdd 0 {VDD}
Vref vinn 0 {VREF}
Vin vinp 0 PULSE({VLO} 1.0 2n 0.02n 0.02n 30n 60n)
IREF vdd nbias {IREFV}
XM5b nbias nbias 0 0 sky130_fd_pr__nfet_01v8 W={W5b} L={Lp}
XM5  tail nbias 0 0 sky130_fd_pr__nfet_01v8 W={W5} L={Lp}
XM1 o1  vinp tail 0 sky130_fd_pr__nfet_01v8 W={W1} L={Lp}
XM2 out vinn tail 0 sky130_fd_pr__nfet_01v8 W={W1} L={Lp}
XM3 o1  o1 vdd vdd sky130_fd_pr__pfet_01v8 W={W3} L={Lp}
XM4 out o1 vdd vdd sky130_fd_pr__pfet_01v8 W={W3} L={Lp}
CL out 0 {CL}"""

_CMP_PARAM_ORDER = ["VDD", "VREF", "VLO", "W1", "W3", "W5", "W5b", "Lp", "IREFV", "CL"]


def _cmp_param_lines(sizing: dict) -> str:
    s = {**DEFAULT_CMP_SIZING, **{k: str(v) for k, v in (sizing or {}).items()}}
    return ".param " + " ".join(f"{k}={s[k]}" for k in _CMP_PARAM_ORDER)


def render_comparator_tran(
    sizing: dict | None = None,
    corner: str = "tt",
    knob: str = "Vov",
    metric: str = "tpd_s",
    points: list[str] | None = None,
) -> str:
    """Render the .tran propagation-delay sweep: for each overdrive point, set the input PULSE high
    level to (trip + overdrive), run a transient, and measure the input-crossing-to-output-decision
    delay tpd. tpd falls as overdrive rises (the comparator speed law)."""
    points = points or ["0.05", "0.1", "0.2", "0.4"]
    s = {**DEFAULT_CMP_SIZING, **{k: str(v) for k, v in (sizing or {}).items()}}
    vref, vlo = s["VREF"], s["VLO"]
    foreach = " ".join(points)
    # trip ~ VREF; both the input trigger and the output decision threshold are taken at VREF.
    control = (
        ".control\n"
        f"foreach pt {foreach}\n"
        f"  let vhi = {vref} + $pt\n"
        f"  alter @Vin[pulse] = [ {vlo} $&vhi 2n 0.02n 0.02n 30n 60n ]\n"
        "  tran 0.005n 20n\n"
        "  let y = 0/0\n"
        f"  meas tran y trig v(vinp) val={vref} rise=1 targ v(out) val={vref} rise=1\n"
        f"  echo RDATA {knob.lower()} $pt $&y\n"
        "end\n"
        ".endc"
    )
    return "\n".join([
        f"* T-template comparator_continuous_nmos (TRAN, {knob} sweep)",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _cmp_param_lines(sizing or {}),
        _CMP_BODY,
        control,
        ".end",
        "",
    ])


def render_comparator_cell(sizing: dict | None = None, corner: str = "tt") -> str:
    """Bare cell for the continuous-time comparator — the Specimen.netlist identity."""
    return "\n".join([
        "* cell comparator_continuous_nmos",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _cmp_param_lines(sizing or {}),
        _CMP_BODY,
        ".end",
        "",
    ])


# ── cds_switched_cap_nmos (13th template — correlated double sampling, the SIGNATURE CIS readout
# technique, and the substrate's first SWITCHED-CAPACITOR / clocked specimen). A sampling cap Cs in
# series with a real sky130 nfet reset switch (gate = clock phase phi1): during the RESET phase phi1
# shorts the output node vo to Vcm while the input sits at the pedestal -> Cs stores (pedestal - Vcm);
# then phi1 goes low (switch off, vo holds on Cs) and the input steps up by DIFF -> charge conservation
# gives vo = Vcm + DIFF*Cs/(Cs+Cpar), INDEPENDENT of the pedestal. So a common offset / fixed-pattern
# pedestal cancels and only the (signal - reset) difference survives — the deterministic, simulatable
# half of CDS (the kTC-noise reduction is a separate noise-analysis story). Validated sky130: held vo
# IDENTICAL (cov 0) across a 0.5->1.3 V pedestal sweep while tracking DIFF (1.0975 V at DIFF=0.2,
# 1.1973 V at DIFF=0.3). Rleak (1G) gives the hold node a DC path (negligible droop over the sim). ──

DEFAULT_CDS_SIZING: dict[str, str] = {
    "VDD": "1.8", "VCM": "0.9", "VPED": "0.9", "DIFF": "0.2", "CS": "1000f", "WSW": "4", "Lp": "0.15",
}

# The input is a swept pedestal DC source (Vped) in series with a fixed reset->signal step (Vstep PWL,
# rising by DIFF at t=5n). phi1 is high (switch on, vo=Vcm) through the reset phase, then low (hold)
# before the signal step. Charge conservation on the floating Cs node carries only the DIFF to vo.
_CDS_BODY = """VDD vdd 0 {VDD}
Vcm vcm 0 {VCM}
Vped a 0 {VPED}
Vstep vin a PWL(0 0 5n 0 5.05n {DIFF} 20n {DIFF})
Cs vin vo {CS}
Xsw vo phi1 vcm 0 sky130_fd_pr__nfet_01v8 W={WSW} L={Lp}
Rleak vo vcm 1G
Vphi1 phi1 0 PWL(0 {VDD} 4.8n {VDD} 4.9n 0 20n 0)"""

_CDS_PARAM_ORDER = ["VDD", "VCM", "VPED", "DIFF", "CS", "WSW", "Lp"]


def _cds_param_lines(sizing: dict) -> str:
    s = {**DEFAULT_CDS_SIZING, **{k: str(v) for k, v in (sizing or {}).items()}}
    return ".param " + " ".join(f"{k}={s[k]}" for k in _CDS_PARAM_ORDER)


def render_cds_tran(
    sizing: dict | None = None,
    corner: str = "tt",
    knob: str = "Vped",
    metric: str = "vo_v",
    points: list[str] | None = None,
) -> str:
    """Render the .tran pedestal sweep: for each pedestal level, run reset->signal and measure the held
    output after settling. The held vo is invariant to the pedestal (CDS cancels the common offset)."""
    points = points or ["0.5", "0.7", "0.9", "1.1", "1.3"]
    foreach = " ".join(points)
    control = (
        ".control\n"
        f"foreach pt {foreach}\n"
        f"  alter {knob} = $pt\n"
        "  tran 0.01n 15n\n"
        "  let y = 0/0\n"
        "  meas tran y find v(vo) at=14n\n"
        f"  echo RDATA {knob.lower()} $pt $&y\n"
        "end\n"
        ".endc"
    )
    return "\n".join([
        f"* T-template cds_switched_cap_nmos (TRAN, {knob} sweep)",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _cds_param_lines(sizing or {}),
        _CDS_BODY,
        control,
        ".end",
        "",
    ])


def render_cds_cell(sizing: dict | None = None, corner: str = "tt") -> str:
    """Bare cell for the CDS sampler — the Specimen.netlist identity."""
    return "\n".join([
        "* cell cds_switched_cap_nmos",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _cds_param_lines(sizing or {}),
        _CDS_BODY,
        ".end",
        "",
    ])


# ── single_slope_ramp_generator (14th template — the single-slope-ADC ramp, the last analog block of
# the CIS column readout chain). A PMOS current-source mirror sources I into a capacitor Cramp -> a
# linear voltage ramp v(t) = (I/Cramp)*t; an nfet reset switch discharges Cramp to 0 then releases.
# The ramp slope is the SS-ADC's volts-per-code-time and (with the comparator + a digital counter,
# which is out of the analog substrate's scope) sets the conversion. The defining behaviour: the slope
# is proportional to the charging current (slope = I/Cramp). Validated sky130: slope 2.0e7 V/s at 10uA
# -> 7.9e7 V/s at 40uA, with slope/I ~= 1/Cramp constant within ~1% (the linearity the ADC relies on).
# Reuses the .tran path; the slope metric is measured as 0.7V / (t@1.0V - t@0.3V) in the linear region. ──

DEFAULT_RAMP_SIZING: dict[str, str] = {
    "VDD": "1.8", "IREFV": "10u", "CR": "500f", "WP": "8", "WS": "4", "Lp": "0.5",
    # LS = reset-switch length, its own param (NOT Lp — the switch has always been shorter than the
    # current-source devices). Was a bare hardcoded `L=0.15` literal until 2026-07-05, which made the
    # switch length unreachable by PDK_SIZING_OVERRIDES and PORT-FAILED the template on both
    # non-sky130 PDKs (0.15 resolves to METERS under gf180's unit convention). sky130 numeric
    # behavior unchanged: LS=0.15.
    "LS": "0.15",
}

_RAMP_BODY = """VDD vdd 0 {VDD}
IREF irefn 0 {IREFV}
XMrp irefn irefn vdd vdd sky130_fd_pr__pfet_01v8 W={WP} L={Lp}
XMch vramp irefn vdd vdd sky130_fd_pr__pfet_01v8 W={WP} L={Lp}
Cramp vramp 0 {CR}
Xrst vramp phirst 0 0 sky130_fd_pr__nfet_01v8 W={WS} L={LS}
Vrst phirst 0 PWL(0 {VDD} 1n {VDD} 1.05n 0 200n 0)"""

_RAMP_PARAM_ORDER = ["VDD", "IREFV", "CR", "WP", "WS", "Lp", "LS"]


def _ramp_param_lines(sizing: dict) -> str:
    s = {**DEFAULT_RAMP_SIZING, **{k: str(v) for k, v in (sizing or {}).items()}}
    return ".param " + " ".join(f"{k}={s[k]}" for k in _RAMP_PARAM_ORDER)


def render_ramp_tran(
    sizing: dict | None = None,
    corner: str = "tt",
    knob: str = "Iref",
    metric: str = "slope_vps",
    points: list[str] | None = None,
) -> str:
    """Render the .tran ramp-slope sweep: for each charging current, run a transient and measure the
    ramp slope as 0.7V/(t@1.0V - t@0.3V) in the linear region. Slope rises with the current (= I/Cramp)."""
    points = points or ["10u", "20u", "40u", "80u"]
    foreach = " ".join(points)
    control = (
        ".control\n"
        f"foreach pt {foreach}\n"
        "  alter IREF = $pt\n"
        "  tran 0.02n 60n\n"
        "  let t1 = 0/0\n"
        "  meas tran t1 when v(vramp)=0.3 rise=1\n"
        "  let t2 = 0/0\n"
        "  meas tran t2 when v(vramp)=1.0 rise=1\n"
        "  let y = 0.7/(t2-t1)\n"
        f"  echo RDATA {knob.lower()} $pt $&y\n"
        "end\n"
        ".endc"
    )
    return "\n".join([
        f"* T-template single_slope_ramp_generator (TRAN, {knob} sweep)",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _ramp_param_lines(sizing or {}),
        _RAMP_BODY,
        control,
        ".end",
        "",
    ])


def render_ramp_cell(sizing: dict | None = None, corner: str = "tt") -> str:
    """Bare cell for the ramp generator — the Specimen.netlist identity."""
    return "\n".join([
        "* cell single_slope_ramp_generator",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _ramp_param_lines(sizing or {}),
        _RAMP_BODY,
        ".end",
        "",
    ])


# ── column_pga_inverting_nmos (15th template — the CIS column programmable-gain amplifier, closing the
# readout chain's amplification block). A 5T OTA core in an INVERTING resistive feedback loop: + input
# = Vcm, - input = the summing node vx (virtual ground), input driven through Rin to vx, feedback Rf
# from out to vx. The closed-loop gain is -Rf/Rin — set by the resistor RATIO, the defining
# programmable-gain property (the gain is programmed by the feedback network, not the device sizing).
# The feedback resistors are deliberately >> the OTA output resistance (ro ~350k): a transconductance
# amp drives a resistive load poorly, so small Rf loads the output and tanks the gain (which is exactly
# why real CIS column PGAs use switched-capacitor feedback). Validated sky130 (Rin=500k): gain -0.4 dB
# @ Rf=500k -> 16.9 dB @ Rf=4M, tracking 20log(Rf/Rin) within ~1 dB (the small shortfall is the finite
# open-loop gain, growing with the programmed gain). Reuses the OTA ac path + the acl_db meas. ──

DEFAULT_PGA_SIZING: dict[str, str] = {
    "VDD": "1.8", "VCM": "0.9", "W1": "8", "W3": "8", "W5": "8", "W5b": "4", "Lp": "0.5",
    "IREFV": "10u", "CL": "500f", "RIN": "500k", "RF": "1000k",
}

_PGA_BODY = """VDD vdd 0 {VDD}
Vcm vcm 0 {VCM}
Vin vin 0 DC {VCM} AC 1
Rin vin vx {RIN}
Rf  out vx {RF}
IREF vdd nbias {IREFV}
XM5b nbias nbias 0 0 sky130_fd_pr__nfet_01v8 W={W5b} L={Lp}
XM5  tail nbias 0 0 sky130_fd_pr__nfet_01v8 W={W5} L={Lp}
XM1 o1  vcm tail 0 sky130_fd_pr__nfet_01v8 W={W1} L={Lp}
XM2 out vx  tail 0 sky130_fd_pr__nfet_01v8 W={W1} L={Lp}
XM3 o1  o1 vdd vdd sky130_fd_pr__pfet_01v8 W={W3} L={Lp}
XM4 out o1 vdd vdd sky130_fd_pr__pfet_01v8 W={W3} L={Lp}
CL out 0 {CL}"""

_PGA_PARAM_ORDER = ["VDD", "VCM", "W1", "W3", "W5", "W5b", "Lp", "IREFV", "CL", "RIN", "RF"]


def _pga_param_lines(sizing: dict) -> str:
    s = {**DEFAULT_PGA_SIZING, **{k: str(v) for k, v in (sizing or {}).items()}}
    return ".param " + " ".join(f"{k}={s[k]}" for k in _PGA_PARAM_ORDER)


def render_column_pga_ac(
    sizing: dict | None = None,
    corner: str = "tt",
    knob: str = "Rf",
    metric: str = "acl_db",
    points: list[str] | None = None,
) -> str:
    """Render the closed-loop AC sweep: for each feedback resistor Rf, measure the closed-loop gain
    (max vdb(out), with AC=1 driving the input through Rin). The gain rises with Rf (= -Rf/Rin)."""
    points = points or ["500k", "1000k", "2000k", "4000k"]
    meas = OTA_METRIC_MEAS[metric]
    foreach = " ".join(points)
    control = (
        ".control\n"
        f"foreach pt {foreach}\n"
        f"  alter {knob} = $pt\n"
        "  ac dec 40 1 1G\n"
        f"  {meas}\n"
        f"  echo RDATA {knob.lower()} $pt $&y\n"
        "end\n"
        ".endc"
    )
    return "\n".join([
        f"* T-template column_pga_inverting_nmos (AC, {knob} sweep)",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _pga_param_lines(sizing or {}),
        _PGA_BODY,
        control,
        ".end",
        "",
    ])


def render_column_pga_cell(sizing: dict | None = None, corner: str = "tt") -> str:
    """Bare cell for the column PGA — the Specimen.netlist identity."""
    return "\n".join([
        "* cell column_pga_inverting_nmos",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _pga_param_lines(sizing or {}),
        _PGA_BODY,
        ".end",
        "",
    ])


# ── ptat_ctat_core_bjt (16th template — S3-inc2a §3, the FIRST new template since the registry froze
# at 18: a dVBE-based CTAT/(virtual-)PTAT reference core, the smallest member of the bandgap family
# (the full amp-in-loop bandgap stays MULTI-STAGE-HARD backlog per TOPOLOGY_BACKLOG.md's reconciled
# tags). REWORKED post-review (must-fix finding, S3-inc2a code review): the original design used the
# two DIFFERENT-SIZE sky130 pnp_05v5 subckts at equal collector current, on the theory that dVBE =
# VT*ln(emitter-area ratio) — this does NOT hand-anchor (scratchpad/verify_ptat_anchor_adversarial.py,
# live @27C): (1) the two discrete sizes carry INDEPENDENTLY-FIT `.model` cards, not a geometry-scaled
# pair (implied N from an ideal equal-current bias drifts 2.06->3.01 over 0-85C, elasticity ~2.5 — not
# a clean ln(N) law); (2) the mirror topology double-injected IBIAS onto the SAME node the reference
# BJT sat on, so the two BJTs did NOT carry equal current at all (Q1 = IBIAS + I_mirror, Q2 = I_mirror
# alone — 29.0uA vs 19.0uA measured, not equal); (3) the shipped resistor read the WRONG node pair
# (not the two BJTs' own VEBs), so its "PTAT current" was actually a PMOS square-law current with
# measured elasticity ~6.9, nothing like a dVBE/R law.
#
# THE FIX — same-DEVICE, CURRENT-ratio dVBE core (scratchpad/probe_ptat_5.. through _8.., converging
# on probe #18's design, then verified end-to-end against THIS render function,
# scratchpad/probe_ptat_9_final_hand_anchor.py): Q1 and Q2 are the IDENTICAL sky130 pnp_05v5 "unit"
# subckt (Is cancels EXACTLY in the dVBE law — no dependence on sky130's two-size catalog at all).
# The N:1 emitter-CURRENT ratio (not area ratio) is set by a clean 3-leg PMOS mirror: IBIAS is wired
# as a SINK from a diode-connected reference device (XMP0, `IBIAS nref 0`, not `IBIAS vdd nref`) so
# KCL forces I(XMP0) = IBIAS exactly, no other element shares that node (the double-injection bug is
# structurally impossible here — no BJT sits at the diode-reference node at all). XMP1 (m=1) mirrors
# that current into Q1; XMP2 (m={N}, the SAME W/L as XMP1, N parallel unit fingers — sky130's `m=`
# device-multiplicity parameter, already used elsewhere in this file for the Miller-OTA input pair)
# mirrors N times that current into Q2. dVBE = V(n2) - V(n1) is then genuinely VT*ln(Ic2/Ic1) ~=
# VT*ln(N), Is-independent by construction.
#
# HAND ANCHOR (probe #18/#9, DEFAULT sizing IBIAS=2u/N=5, @27C): ideal dVBE = VT*ln(5) = 41.628 mV;
# measured dVBE = 48.541 mV — a CONSTANT +6.9 mV offset, present at every swept temperature (0C:
# +6.85mV, 27C: +6.91mV, 85C: +7.04mV — varies <3% over the whole range) and hand-explained as an
# ordinary series-resistance IR drop: offset/(Ic2-Ic1) = 846.95 / 846.62 / 845.57 ohm at 0/27/85C
# respectively — i.e. an IMPLIED constant ~846-ohm emitter/collector series resistance inside the
# sky130 pnp_05v5 model, not a mystery. Elasticity of dVBE vs ABSOLUTE temperature (log-log slope,
# probe #9): 0.876 — close to the ideal PTAT law's 1.0 (the constant ~7mV offset is a real but MINOR
# correction on top of the dominant VT*ln(N) term, unlike the old design where an unrelated ~34mV
# offset at higher bias currents swamped the signal, giving elasticity ~0.60 — bias current level
# matters here, which is why the default IBIAS dropped from 10u to 2u: the offset scales with the
# ABSOLUTE current difference Ic2-Ic1, not the ratio, so a smaller absolute bias keeps the offset a
# small correction on the dVBE law instead of a comparable-magnitude confound). This elasticity number
# is NOT shipped as a QuantTest(kind="elasticity") claim (spec §3: "only if it certifies cleanly, do
# not force") — the growth-engine's canonical x-series for this template is `temp` recorded in
# CELSIUS (0..85, spanning zero for the default sweep), and the oracle's `_loglog` takes `math.log(x)`
# directly (oracle.py) — `log(0)` is a domain error, so a temp-in-Celsius x-axis cannot host a formal
# elasticity claim without re-basing the recorded x to Kelvin (out of scope for this fix; the number
# above is reported in this comment for anyone reading the mechanism, and is a probe finding, not a
# certified claim-card assertion).
#
# `iptat_a` IS NOW EXPLICITLY A DERIVED (virtual) QUANTITY, not a separately-instantiated resistor
# branch: `iptat = (V(n2)-V(n1)) / R` computed in the `.control` block from the two BJTs' OWN node
# voltages (the same convention every other template in this file uses for a `.meas`/`let`-derived
# metric — e.g. `rin_ohm`, the swing/ICMR device queries). `R` (default 2k, DEFAULT_PTAT_SIZING) is a
# NOMINAL scaling constant — "the current a resistor of this value would carry if tapped across the
# two reference nodes" — not a physically-instantiated element the mirror's own bias currents flow
# through (so it does not perturb the current-ratio mirror it is reading from). Because elasticity is
# scale-invariant under division by a constant, `iptat_a`'s own elasticity vs T is IDENTICAL to
# dVBE's (~0.876) — it is a real, honestly-derived PTAT-like current preview, not a re-labeled square-
# law artifact.
#
# `vbe_v` (CTAT) is Q1's OWN VEB (V(n1), the 1x/reference leg) — "at an ESSENTIALLY FIXED bias
# current" is now an honest description (probe #9: I(XQ1) varies 2.03uA->2.10uA over 0-85C, <4%,
# because IBIAS is an ideal fixed source and the 1:1 mirror leg tracks it closely) — unlike the OLD
# design, where the shipped circuit's actual bias current rose ~6.4x over the same sweep (an artifact
# of the double-injection bug coupling bias level to VEB itself), the old vbe_v narrative's "at a
# given bias current" was materially false.
#
# TEMPERATURE IS STILL THE KNOB (spec §3) — the same native `.dc temp <lo> <hi> <step>` mechanism as
# before, indexing per-node vectors inside the SAME `.control` block (`v(n1)[i]`, `v(n2)[i]`) rather
# than reading back ngspice's own sweep-variable vector (the `temp-sweep` hyphen-parses-as-subtraction
# gotcha probe #13 caught still applies and is still avoided the same way — the x-value at each index
# is recomputed directly from the known lo/step, never read back from ngspice). Validated sky130
# (probe #9, DEFAULT sizing, native 0-85C@5C sweep): vbe_v (CTAT) FALLS monotonically; iptat_a
# (PTAT-derived) RISES monotonically — both signs match seeds.py's claims (iptat_a vs temp [direction
# +], vbe_v vs temp [direction -]). Exact endpoint values are recorded in the test file's docstring
# (tests/test_executable_templates_metrics.py), sourced from this exact render function's live output,
# not a hand-typed guess. ──

DEFAULT_PTAT_SIZING: dict[str, str] = {
    "VDD": "1.8", "IBIAS": "2u", "R": "2k", "Wp": "8", "Lp": "0.5", "N": "5",
}

# 3-leg PMOS current mirror (XMP0 diode-connected reference, sunk by IBIAS — no BJT shares that node,
# so the old double-injection bug is structurally impossible here) fanning out to two IDENTICAL
# sky130 pnp_05v5 "unit" diode-connected PNPs (base=collector=0, emitter=n1/n2) at a 1:{N} CURRENT
# ratio (XMP2's `m={N}` device-multiplicity parameter, not an area/W ratio and not a BJT-size ratio).
# V(n2)-V(n1) is genuinely VT*ln(Ic2/Ic1) ~= VT*ln(N), Is-independent (same model card for both BJTs).
_PTAT_BODY = """VDD vdd 0 {VDD}
IBIAS nref 0 {IBIAS}
XMP0 nref nref vdd vdd sky130_fd_pr__pfet_01v8 W={Wp} L={Lp} m=1
XMP1 n1   nref vdd vdd sky130_fd_pr__pfet_01v8 W={Wp} L={Lp} m=1
XMP2 n2   nref vdd vdd sky130_fd_pr__pfet_01v8 W={Wp} L={Lp} m={N}
XQ1  n1 0 0 sky130_fd_pr__pnp_05v5_W0p68L0p68
XQ2  n2 0 0 sky130_fd_pr__pnp_05v5_W0p68L0p68"""

_PTAT_PARAM_ORDER = ["VDD", "IBIAS", "R", "Wp", "Lp", "N"]


def _ptat_param_lines(sizing: dict) -> str:
    s = {**DEFAULT_PTAT_SIZING, **{k: str(v) for k, v in (sizing or {}).items()}}
    return ".param " + " ".join(f"{k}={s[k]}" for k in _PTAT_PARAM_ORDER)


def render_ptat_ctat_core_bjt(
    sizing: dict | None = None,
    corner: str = "tt",
    knob: str = "temp",
    metric: str = "iptat_a",
    points: list[str] | None = None,
) -> str:
    """Render the native `.dc temp <lo> <hi> <step>` sweep (see the module comment block above for the
    hand-anchored topology + probe transcript). `points` gives [lo, hi] in Celsius (only the first and
    last elements are used — a count/list-of-many authored via `_materialize_sweep_points` still works,
    since a range is all this analysis needs); default 0..85C at a fixed 5C step (finer for a narrower
    authored span). Emits ONE `RDATA temp <T> <y>` line per internal sweep point, `y` = vbe_v (Q1's own
    VEB, the CTAT reference, at an essentially fixed bias current) or iptat_a (the derived (V(n2)-
    V(n1))/R PTAT-like current preview) depending on `metric` — both computed from the SAME completed
    sweep, cheaply, so no second sim is needed."""
    points = points or ["0", "85"]
    lo, hi = float(points[0]), float(points[-1])
    if lo > hi:
        lo, hi = hi, lo
    span = hi - lo
    step = 5.0 if span >= 20 else max(span / 10.0, 1.0)
    s = {**DEFAULT_PTAT_SIZING, **{k: str(v) for k, v in (sizing or {}).items()}}
    r_ohms = parse_unit(s["R"])
    y_var = "vbe1" if metric == "vbe_v" else "iptat"
    control = (
        ".control\n"
        f"dc temp {lo:g} {hi:g} {step:g}\n"
        "let n = length(v(n1))\n"
        "let i = 0\n"
        "while i < n\n"
        f"  let t = {lo:g} + i * {step:g}\n"
        "  let vbe1 = v(n1)[i]\n"
        "  let vbe2 = v(n2)[i]\n"
        f"  let iptat = (vbe2 - vbe1) / {r_ohms:g}\n"
        f"  echo RDATA {knob.lower()} $&t $&{y_var}\n"
        "  let i = i + 1\n"
        "end\n"
        ".endc"
    )
    return "\n".join([
        f"* T-template ptat_ctat_core_bjt (DC temp sweep, {metric})",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _ptat_param_lines(sizing or {}),
        _PTAT_BODY,
        control,
        ".end",
        "",
    ])


def render_ptat_ctat_core_bjt_cell(sizing: dict | None = None, corner: str = "tt") -> str:
    """Bare cell for the PTAT/CTAT core — the Specimen.netlist identity."""
    return "\n".join([
        "* cell ptat_ctat_core_bjt",
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        _ptat_param_lines(sizing or {}),
        _PTAT_BODY,
        ".end",
        "",
    ])


# ---------------------------------------------------------------------------------------------------
# Renderer registry (Tier-B §1a unification): template_ref -> (sweep-deck renderer, cell renderer,
# default sizing). Relocated here from executor.py so that ADDING AN ENGINE touches only the registry
# (this file + its engine's render module) and seeds.py — never run_recipe's control flow. Each renderer
# shares the (sizing, corner=, knob=, metric=, points=) signature, so the executor loop is engine-agnostic.
# Digital (iverilog) entries are appended by their own register step (T4/T5).
# ---------------------------------------------------------------------------------------------------
RENDERERS: dict[str, tuple] = {
    "miller_ota_ac": (render_miller_ota_ac, render_miller_ota_cell, DEFAULT_OTA_SIZING),
    "current_mirror_dc": (render_current_mirror_dc, render_current_mirror_cell, DEFAULT_CM_SIZING),
    "common_source_ac": (render_common_source_ac, render_common_source_cell, DEFAULT_CS_SIZING),
    "common_gate_ac": (render_common_gate_ac, render_common_gate_cell, DEFAULT_CG_SIZING),
    "source_follower_ac": (render_source_follower_ac, render_source_follower_cell, DEFAULT_SF_SIZING),
    "diff_pair_ac": (render_diff_pair_ac, render_diff_pair_cell, DEFAULT_DP_SIZING),
    "cascode_mirror_dc": (render_cascode_mirror_dc, render_cascode_mirror_cell, DEFAULT_CASC_SIZING),
    "ota_5t_ac": (render_ota_5t_ac, render_ota_5t_cell, DEFAULT_OTA5T_SIZING),
    "telescopic_cascode_ac": (render_telescopic_cascode_ac, render_telescopic_cascode_cell,
                              DEFAULT_TELE_SIZING),
    "folded_cascode_ac": (render_folded_cascode_ac, render_folded_cascode_cell, DEFAULT_FC_SIZING),
    "regulated_cascode_dc": (render_regulated_cascode_dc, render_regulated_cascode_cell,
                             DEFAULT_RGC_SIZING),
    "comparator_tran": (render_comparator_tran, render_comparator_cell, DEFAULT_CMP_SIZING),
    "cds_tran": (render_cds_tran, render_cds_cell, DEFAULT_CDS_SIZING),
    "ramp_tran": (render_ramp_tran, render_ramp_cell, DEFAULT_RAMP_SIZING),
    "column_pga_ac": (render_column_pga_ac, render_column_pga_cell, DEFAULT_PGA_SIZING),
    "ptat_ctat_core_bjt": (render_ptat_ctat_core_bjt, render_ptat_ctat_core_bjt_cell, DEFAULT_PTAT_SIZING),
}


# ===================================================================================================
# DIGITAL (iverilog) specimens — Tier-B §3. Registered as ONE contained block: a digital template_ref
# declares its engine + renderers in the SAME registries the analog specimens use, so the iverilog
# engine's whole footprint is this block + its render module (digital_templates.py) + seeds.py. This is
# the unification's payoff (§1a): adding/extending an engine never touches run_recipe.
# ===================================================================================================
from openclaw_brain.knowledge.executable.digital_templates import (  # noqa: E402
    render_digital_cds_cell,
    render_digital_cds_tran,
    render_gray_counter_cell,
    render_gray_counter_tran,
    render_ss_adc_backend_cell,
    render_ss_adc_backend_tran,
)

DEFAULT_DIGITAL_CDS_SIZING: dict = {"W": 8, "DELTA": 16}
DEFAULT_GRAY_SIZING: dict = {"W": 4}
DEFAULT_SS_ADC_SIZING: dict = {"W": 8, "DELTA": 16, "REF": 200}

TEMPLATES.update({
    "digital_cds_nmos": {
        "name": "digital_cds_nmos",
        "description": "digital correlated double sampling: out = signal_code - reset_code (pedestal cancels)",
        "devices": ["digital"],
        "default_sizing": DEFAULT_DIGITAL_CDS_SIZING,
        "template_ref": "digital_cds_tb",
        "engine": "iverilog",
        "analyses": ["tran"],
        "knobs": ["ped"],
        "metrics": ["diff_lsb"],
    },
    "gray_code_counter": {
        "name": "gray_code_counter",
        "description": "reflected-binary Gray counter: consecutive codes differ by exactly one bit",
        "devices": ["digital"],
        "default_sizing": DEFAULT_GRAY_SIZING,
        "template_ref": "gray_counter_tb",
        "engine": "iverilog",
        "analyses": ["tran"],
        "knobs": ["idx"],
        "metrics": ["hamming"],
    },
    "ss_adc_digital_backend": {
        "name": "ss_adc_digital_backend",
        "description": "single-slope-ADC digital back-end: gray counter + latch + gray->binary + digital CDS",
        "devices": ["digital"],
        "default_sizing": DEFAULT_SS_ADC_SIZING,
        "template_ref": "ss_adc_backend_tb",
        "engine": "iverilog",
        "analyses": ["tran"],
        "knobs": ["code", "trip", "ped"],          # gray-code index / comparator-trip time / pedestal
        "metrics": ["g2b_match", "code", "diff_match"],
    },
})

RENDERERS.update({
    "digital_cds_tb": (render_digital_cds_tran, render_digital_cds_cell, DEFAULT_DIGITAL_CDS_SIZING),
    "gray_counter_tb": (render_gray_counter_tran, render_gray_counter_cell, DEFAULT_GRAY_SIZING),
    "ss_adc_backend_tb": (render_ss_adc_backend_tran, render_ss_adc_backend_cell, DEFAULT_SS_ADC_SIZING),
})

# Monte-Carlo mismatch renderers (Stat-QT). Bottom-import (cells are defined above) resolves the cycle:
# mc_templates imports render_ota_5t_cell/render_comparator_cell from here.
from openclaw_brain.knowledge.executable.mc_templates import (  # noqa: E402
    render_comparator_fpn_mc,
    render_ota5t_gbw,
    render_ota5t_offset_mc,
)

RENDERERS.update({
    "ota5t_offset_mc": (render_ota5t_offset_mc, render_ota_5t_cell, DEFAULT_OTA5T_SIZING),
    "comparator_fpn_mc": (render_comparator_fpn_mc, render_comparator_cell, DEFAULT_CMP_SIZING),
    "ota5t_gbw": (render_ota5t_gbw, render_ota_5t_cell, DEFAULT_OTA5T_SIZING),
})

# E3-I1 intervention variant renderers (see the module-level comment block above render_miller_ota_
# rz_null_ac / render_miller_ota_ff_break_ac for the live-validation transcripts). Registered under
# their OWN template_ref ("miller_ota_ac__<intervention id>") — never under a TEMPLATES entry, since a
# variant is reachable only via an InterventionSpec (interventions.py), never as a recipe's own
# topology_class (spec §2's "Mechanics": the baseline stays the class's normal template).
RENDERERS.update({
    "miller_ota_ac__rz_null": (render_miller_ota_rz_null_ac, render_miller_ota_rz_null_cell,
                               DEFAULT_OTA_RZ_SIZING),
    "miller_ota_ac__ff_break": (render_miller_ota_ff_break_ac, render_miller_ota_ff_break_cell,
                                DEFAULT_OTA_SIZING),
})
