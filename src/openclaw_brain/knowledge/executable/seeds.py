"""Validated seed recipes for projection into the live graph (SPEC §13.1).

These are the topology classes the executable substrate has validated end-to-end against real
sky130 — the recipes the `project-executable` CLI runs (real ngspice) to produce verdicted
specimens, then projects additively onto the existing text-knowledge graph. Sizing = the validated
all-saturation point (the default). Adding a validated class here = one more entry, no new wiring.
"""

from __future__ import annotations

from .models import (
    ClaimCard, AnalogPVT, DigitalUnits, MechanismClaim, QuantTest, VerificationRecipe,
)

_TT = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)


def _tt() -> AnalogPVT:
    """A FRESH, independent copy of the nominal TT/27C/1.8V analog condition.

    `_TT` above is a single module-level value template — every call site that wants "the nominal
    TT point" must go through this factory rather than passing `_TT` itself, so no two claim-cards
    (or a claim-card and its owning recipe) ever alias the SAME AnalogPVT object. `AnalogPVT` is a
    plain (non-frozen) pydantic model, so an in-place mutation on one shared reference would
    silently corrupt every other claim/recipe still pointing at it — a direct violation of ADR
    Decision 4 (scope-honesty). `model_copy(deep=True)` (not a shallow copy) because AnalogPVT's
    `pdk_profile` field defaults to a mutable `{}` — a shallow copy would still alias that nested
    dict across every "copy" of `_TT`. Zero behavior change for current values: every copy compares
    equal to `_TT` by content, only object identity differs."""
    return _TT.model_copy(deep=True)


OTA = "miller_ota_2stage_nmos_in"
CM = "current_mirror_simple_nmos"
CS = "common_source_active_load_nmos"
CG = "common_gate_nmos"
SF = "source_follower_nmos"
DP = "diff_pair_resistive_nmos"
CASC = "cascode_current_mirror_nmos"
OTA5T = "ota_5t_nmos_in"
TELE = "telescopic_cascode_ota_nmos_in"
FC = "folded_cascode_ota_nmos_in"
RGC = "regulated_cascode_nmos"
CMP = "comparator_continuous_nmos"
CDS = "cds_switched_cap_nmos"
RAMP = "single_slope_ramp_generator"
PGA = "column_pga_inverting_nmos"


def _claim(cid, tclass, knob, metric, quant, narrative):
    return ClaimCard(id=cid, topology_class=tclass, conditions=_tt(),
                     mechanism=MechanismClaim(knob=knob, metric=metric, series_ref=cid,
                                              quant=quant, narrative=narrative))


# --- Digital (iverilog) pedagogy specimens (Tier-B §3b): artifact-agnostic claim-cards over a
#     FUNCTIONAL engine. Conditions are DigitalUnits (no PVT); the basis is functional, not physical. ---
DCDS = "digital_cds_nmos"
GRAY = "gray_code_counter"


def _dig_claim(cid, tclass, knob, metric, quant, narrative, cond):
    return ClaimCard(id=cid, topology_class=tclass, conditions=cond,
                     mechanism=MechanismClaim(knob=knob, metric=metric, series_ref=cid,
                                              quant=quant, narrative=narrative))


def _digital_cds_recipe() -> VerificationRecipe:
    cond = DigitalUnits(bit_width=8, stimulus="pedestal levels 0/64/128/192, single polarity")
    return VerificationRecipe(
        topology_class=DCDS,
        source_ref="Tier-B digital pedagogy demo (artifact-agnostic claim-card)",
        build={"method": "template", "template_ref": "digital_cds_tb", "engine": "iverilog"},
        conditions=cond,
        sweeps=[{"analysis": "tran", "knob": "ped", "points": ["0", "64", "128", "192"],
                 "measure": ["diff_lsb"]}],
        claim_cards=[
            _dig_claim("dcds_inv", DCDS, "ped", "diff_lsb", QuantTest(kind="invariance", cov_max=0.001),
                       "digital subtraction cancels the common pedestal exactly (functional)", cond),
        ],
    )


def _gray_recipe() -> VerificationRecipe:
    cond = DigitalUnits(bit_width=4,
        stimulus="count 0->15 incl. one wrap 15->0",
        stimulus_untested="async reset mid-count, enable de-assert glitch")
    return VerificationRecipe(
        topology_class=GRAY,
        source_ref="Tier-B digital pedagogy demo (Hamming-1 invariance)",
        build={"method": "template", "template_ref": "gray_counter_tb", "engine": "iverilog"},
        conditions=cond,
        sweeps=[{"analysis": "tran", "knob": "idx", "measure": ["hamming"]}],
        claim_cards=[
            _dig_claim("gray_h1", GRAY, "idx", "hamming", QuantTest(kind="invariance", cov_max=0.001),
                       "consecutive Gray codes differ by exactly one bit (functional)", cond),
        ],
    )


SSADC = "ss_adc_digital_backend"


def _ss_adc_backend_recipe() -> VerificationRecipe:
    """The NON-VACUOUS headline (Tier-B §3a): a composed single-slope-ADC digital back-end whose oracle
    ground truth is SPEC-DERIVED (textbook XOR-cascade reference / monotonicity / integer-difference
    identity), computed INDEPENDENTLY in each testbench — not author-supplied expected vectors."""
    cond = DigitalUnits(bit_width=8,
        stimulus="ramp codes over the full count range",
        stimulus_untested="counter wrap at ramp end, metastable comparator strobe")
    return VerificationRecipe(
        topology_class=SSADC,
        source_ref="Tier-B composed digital specimen (single-slope-ADC back-end; spec-derived oracle)",
        build={"method": "template", "template_ref": "ss_adc_backend_tb", "engine": "iverilog"},
        conditions=cond,
        sweeps=[
            {"analysis": "tran", "knob": "code", "measure": ["g2b_match"]},
            {"analysis": "tran", "knob": "trip", "measure": ["code"]},
            {"analysis": "tran", "knob": "ped", "measure": ["diff_match"]},
        ],
        claim_cards=[
            _dig_claim("ss_g2b", SSADC, "code", "g2b_match", QuantTest(kind="invariance", cov_max=0.001),
                       "gray->binary decode matches the textbook XOR-cascade reference over the full range", cond),
            _dig_claim("ss_mono", SSADC, "trip", "code", QuantTest(kind="direction", sign="+"),
                       "the latched+decoded code is monotonic non-decreasing in the comparator-trip time", cond),
            _dig_claim("ss_diff", SSADC, "ped", "diff_match", QuantTest(kind="invariance", cov_max=0.001),
                       "the digital CDS difference equals the integer-difference identity incl. counter wrap", cond),
        ],
    )


def digital_seed_recipes() -> list[VerificationRecipe]:
    """Tier-B digital specimens (iverilog), kept separate from the analog seed_recipes()."""
    return [_digital_cds_recipe(), _gray_recipe(), _ss_adc_backend_recipe()]


def _ota_recipe() -> VerificationRecipe:
    return VerificationRecipe(
        topology_class=OTA,
        source_ref="executable-substrate validated specimen (Specimen-One)",
        build={"method": "template", "template_ref": "miller_ota_ac"},
        conditions=_tt(),
        sweeps=[
            {"analysis": "ac", "knob": "Cc", "points": ["250f", "500f", "1000f", "2000f", "4000f"],
             "measure": ["gbw_hz", "av0_db"]},
            {"analysis": "ac", "knob": "CL", "points": ["1p", "2p", "4p"], "measure": ["pm_deg"]},
        ],
        claim_cards=[
            _claim("cc_gbw", OTA, "Cc", "gbw_hz", QuantTest(kind="direction", sign="-"),
                   "Miller compensation: GBW = gm1/(2*pi*Cc), so GBW falls as Cc rises"),
            _claim("cc_av0", OTA, "Cc", "av0_db", QuantTest(kind="invariance", spread_max=0.5),
                   "DC gain Av0 = gm1*ro1*gm6*ro6 is set by output resistances, ~independent of Cc"),
            _claim("cl_pm", OTA, "CL", "pm_deg", QuantTest(kind="direction", sign="-"),
                   "increasing CL pushes the output pole toward the origin, eroding phase margin"),
        ],
    )


def _cm_recipe() -> VerificationRecipe:
    return VerificationRecipe(
        topology_class=CM,
        source_ref="executable-substrate validated specimen (second topology class)",
        build={"method": "template", "template_ref": "current_mirror_dc"},
        conditions=_tt(),
        sweeps=[{"analysis": "dc", "knob": "Vout", "points": ["0.4", "0.7", "1.0", "1.3", "1.6"],
                 "measure": ["iout_a"]}],
        claim_cards=[
            _claim("cm_iout", CM, "Vout", "iout_a", QuantTest(kind="direction", sign="+"),
                   "iout rises with Vout — finite output resistance (channel-length modulation)"),
        ],
    )


def _cs_recipe() -> VerificationRecipe:
    return VerificationRecipe(
        topology_class=CS,
        source_ref="Razavi — active-loaded common-source gain stage (the gain primitive)",
        build={"method": "template", "template_ref": "common_source_ac"},
        conditions=_tt(),
        sweeps=[{"analysis": "ac", "knob": "Iref",
                 "points": ["2u", "5u", "10u", "20u", "40u"], "measure": ["av0_db", "gbw_hz"]}],
        claim_cards=[
            _claim("cs_gbw", CS, "Iref", "gbw_hz", QuantTest(kind="direction", sign="+"),
                   "GBW = gm/(2*pi*CL); gm rises with bias current (~sqrt(Id)), so GBW rises with Iref"),
            _claim("cs_av0", CS, "Iref", "av0_db", QuantTest(kind="invariance", spread_max=1.5),
                   "Textbook Av0 = gm*(ro1||ro2) ~ 1/sqrt(Id); in sky130 short-channel the gain is only "
                   "weakly dependent on Id — flat within ~1 dB over a 20x current range (the long-channel "
                   "law softens), which the simulation quantifies."),
        ],
    )


def _cg_recipe() -> VerificationRecipe:
    return VerificationRecipe(
        topology_class=CG,
        source_ref="Razavi — common-gate stage (the current buffer)",
        build={"method": "template", "template_ref": "common_gate_ac"},
        conditions=_tt(),
        sweeps=[{"analysis": "ac", "knob": "Iref",
                 "points": ["2u", "5u", "10u", "20u", "40u"], "measure": ["rin_ohm"]}],
        claim_cards=[
            _claim("cg_rin", CG, "Iref", "rin_ohm", QuantTest(kind="direction", sign="-"),
                   "The common-gate input resistance looking into the source is Rin ~ 1/gm (its "
                   "defining low-impedance property); gm rises with bias current, so Rin falls as Iref rises."),
        ],
    )


def _sf_recipe() -> VerificationRecipe:
    return VerificationRecipe(
        topology_class=SF,
        source_ref="Razavi — source follower (the voltage buffer / CIS pixel source follower)",
        build={"method": "template", "template_ref": "source_follower_ac"},
        conditions=_tt(),
        sweeps=[{"analysis": "ac", "knob": "Iref",
                 "points": ["2u", "5u", "10u", "20u", "40u"], "measure": ["av0_db"]}],
        claim_cards=[
            _claim("sf_av0", SF, "Iref", "av0_db", QuantTest(kind="invariance", spread_max=0.5),
                   "The source follower's voltage gain Av = gm/(gm+gmb+gds) is sub-unity (~0.83 here, "
                   "about -1.6 dB, from the body effect gmb) and nearly independent of bias current — "
                   "the hallmark of a good unity-gain buffer, which the simulation confirms."),
        ],
    )


def _dp_recipe() -> VerificationRecipe:
    return VerificationRecipe(
        topology_class=DP,
        source_ref="Razavi — resistively-loaded differential pair (the differential transconductance)",
        build={"method": "template", "template_ref": "diff_pair_ac"},
        conditions=_tt(),
        sweeps=[{"analysis": "ac", "knob": "Iref",
                 "points": ["5u", "10u", "20u", "40u", "80u"], "measure": ["adm_db"]}],
        claim_cards=[
            _claim("dp_adm", DP, "Iref", "adm_db", QuantTest(kind="direction", sign="+"),
                   "The differential gain of a resistively-loaded pair is Adm = gm*RD; gm rises with "
                   "the tail current (~sqrt(Id)), so Adm rises with Iref (RD fixed)."),
        ],
    )


def _casc_recipe() -> VerificationRecipe:
    return VerificationRecipe(
        topology_class=CASC,
        source_ref="Razavi — cascode current mirror (output-resistance boosting)",
        build={"method": "template", "template_ref": "cascode_mirror_dc"},
        conditions=_tt(),
        sweeps=[{"analysis": "dc", "knob": "Vout",
                 "points": ["0.8", "1.0", "1.2", "1.4", "1.6"], "measure": ["iout_a"]}],
        claim_cards=[
            _claim("casc_iout", CASC, "Vout", "iout_a", QuantTest(kind="invariance", cov_max=0.02),
                   "Cascoding boosts the output resistance to Rout ~ gm*ro^2 (~30x a simple mirror), so "
                   "the mirrored current is nearly independent of the output voltage above the cascode "
                   "compliance — the simulation shows iout flat within ~0.2% over Vout 0.8-1.6 V."),
        ],
    )


def _ota5t_recipe() -> VerificationRecipe:
    return VerificationRecipe(
        topology_class=OTA5T,
        source_ref="Razavi — five-transistor (single-stage) OTA / active-mirror differential pair",
        build={"method": "template", "template_ref": "ota_5t_ac"},
        conditions=_tt(),
        sweeps=[{"analysis": "ac", "knob": "CL",
                 "points": ["500f", "1000f", "2000f", "4000f"], "measure": ["av0_db", "gbw_hz"]}],
        claim_cards=[
            _claim("ota5t_gbw", OTA5T, "CL", "gbw_hz", QuantTest(kind="direction", sign="-"),
                   "The single-stage OTA's output pole sets GBW = gm1/(2*pi*CL), so GBW falls as CL rises."),
            _claim("ota5t_av0", OTA5T, "CL", "av0_db", QuantTest(kind="invariance", spread_max=0.5),
                   "The DC gain Av0 = gm1*(ro2||ro4) is set by the device output resistances and is "
                   "independent of the load capacitance (CL moves the pole, not the DC gain)."),
        ],
    )


def _tele_recipe() -> VerificationRecipe:
    return VerificationRecipe(
        topology_class=TELE,
        source_ref="Razavi Ch.9 — telescopic cascode OTA (cascode-boosted single-stage amplifier)",
        build={"method": "template", "template_ref": "telescopic_cascode_ac"},
        conditions=_tt(),
        sweeps=[{"analysis": "ac", "knob": "CL",
                 "points": ["500f", "1000f", "2000f", "4000f"], "measure": ["av0_db", "gbw_hz"]}],
        claim_cards=[
            _claim("tele_gbw", TELE, "CL", "gbw_hz", QuantTest(kind="direction", sign="-"),
                   "The telescopic cascode OTA is single-stage: its output pole sets GBW = "
                   "gm1/(2*pi*CL), so GBW falls as CL rises."),
            _claim("tele_av0", TELE, "CL", "av0_db", QuantTest(kind="invariance", spread_max=0.5),
                   "Cascoding both the input pair and the load boosts the output resistance to "
                   "Rout ~ (gm*ro^2)_n || (gm*ro^2)_p, so the DC gain Av0 = gm1*Rout is much higher "
                   "than a simple-mirror single stage (measured ~67 dB here vs ~37 dB for the 5T OTA) "
                   "and is independent of the load capacitance (CL moves the pole, not the DC gain)."),
        ],
    )


def _fc_recipe() -> VerificationRecipe:
    return VerificationRecipe(
        topology_class=FC,
        source_ref="Razavi Ch.9 — folded-cascode OTA (NMOS input, current-folded single stage)",
        build={"method": "template", "template_ref": "folded_cascode_ac"},
        conditions=_tt(),
        sweeps=[{"analysis": "ac", "knob": "CL",
                 "points": ["500f", "1000f", "2000f", "4000f"], "measure": ["av0_db", "gbw_hz"]}],
        claim_cards=[
            _claim("fc_gbw", FC, "CL", "gbw_hz", QuantTest(kind="direction", sign="-"),
                   "The folded-cascode OTA is single-stage: its output pole sets GBW = gm1/(2*pi*CL), "
                   "so GBW falls as CL rises."),
            _claim("fc_av0", FC, "CL", "av0_db", QuantTest(kind="invariance", spread_max=0.5),
                   "Folding the input-pair current down through PMOS cascodes into an NMOS "
                   "cascode-mirror load keeps the output resistance cascode-high (Rout ~ gm*ro^2), so "
                   "the DC gain Av0 = gm1*Rout is high (measured ~71 dB here) and independent of the "
                   "load capacitance. The fold un-stacks the input device from the output cascode, so "
                   "this stage tolerates a wider input common-mode range and output swing than the "
                   "telescopic — its defining advantage (CL moves the pole, not the DC gain)."),
        ],
    )


def _rgc_recipe() -> VerificationRecipe:
    return VerificationRecipe(
        topology_class=RGC,
        source_ref="Razavi Ch.9 — regulated cascode (gain-boosting technique)",
        build={"method": "template", "template_ref": "regulated_cascode_dc"},
        conditions=_tt(),
        sweeps=[{"analysis": "dc", "knob": "Vout",
                 "points": ["0.8", "1.0", "1.2", "1.4", "1.6"], "measure": ["iout_a"]}],
        claim_cards=[
            _claim("rgc_iout", RGC, "Vout", "iout_a", QuantTest(kind="invariance", cov_max=0.001),
                   "Gain-boosting: an auxiliary amplifier holds the cascode source node constant, so "
                   "the bottom device sees a fixed Vds regardless of the output voltage. This boosts "
                   "the output resistance to Rout ~ gm*ro^2 * A_aux, so the output current is "
                   "essentially independent of Vout — measured flat within ~0.005% here, ~270x tighter "
                   "than the same five transistors with the auxiliary loop replaced by a fixed gate "
                   "(the plain cascode), demonstrating the resistance boost the regulation provides."),
        ],
    )


def _cmp_recipe() -> VerificationRecipe:
    return VerificationRecipe(
        topology_class=CMP,
        source_ref="Razavi Ch.8 / CIS column single-slope ADC — continuous-time comparator",
        build={"method": "template", "template_ref": "comparator_tran"},
        conditions=_tt(),
        sweeps=[{"analysis": "tran", "knob": "Vov",
                 "points": ["0.05", "0.1", "0.2", "0.4"], "measure": ["tpd_s"]}],
        claim_cards=[
            _claim("cmp_tpd", CMP, "Vov", "tpd_s", QuantTest(kind="direction", sign="-"),
                   "A comparator resolves faster with more input overdrive: a larger differential step "
                   "drives the high-gain stage harder, so the output slews to its decision threshold "
                   "sooner. The propagation delay therefore falls as the input overdrive rises — the "
                   "speed-vs-overdrive law that sets the column single-slope ADC's settling budget."),
        ],
    )


def _cds_recipe() -> VerificationRecipe:
    return VerificationRecipe(
        topology_class=CDS,
        source_ref="CIS column readout — correlated double sampling (offset / FPN cancellation)",
        build={"method": "template", "template_ref": "cds_tran"},
        conditions=_tt(),
        sweeps=[{"analysis": "tran", "knob": "Vped",
                 "points": ["0.5", "0.7", "0.9", "1.1", "1.3"], "measure": ["vo_v"]}],
        claim_cards=[
            _claim("cds_vo", CDS, "Vped", "vo_v", QuantTest(kind="invariance", cov_max=0.001),
                   "Correlated double sampling stores the reset level on the sampling capacitor and "
                   "subtracts it from the signal, so a common input pedestal (the per-pixel offset / "
                   "fixed-pattern noise) cancels: the held output carries only the (signal - reset) "
                   "difference and is independent of the pedestal — held identical (cov ~0) across a "
                   "0.5-1.3 V pedestal sweep here. This offset/FPN rejection is the deterministic half "
                   "of CDS (the kTC reset-noise reduction is a separate noise-analysis property)."),
        ],
    )


def _ramp_recipe() -> VerificationRecipe:
    return VerificationRecipe(
        topology_class=RAMP,
        source_ref="CIS column single-slope ADC — current-into-capacitor ramp generator",
        build={"method": "template", "template_ref": "ramp_tran"},
        conditions=_tt(),
        sweeps=[{"analysis": "tran", "knob": "Iref",
                 "points": ["10u", "20u", "40u", "80u"], "measure": ["slope_vps"]}],
        claim_cards=[
            _claim("ramp_slope", RAMP, "Iref", "slope_vps", QuantTest(kind="direction", sign="+"),
                   "A constant current I charging a capacitor Cramp makes a linear ramp with slope = "
                   "I/Cramp, so the ramp slope (the single-slope ADC's volts-per-code-time) rises in "
                   "direct proportion to the charging current — measured slope/I constant within ~1% "
                   "over a 4x current range (the linearity the converter's code spacing relies on)."),
        ],
    )


def _pga_recipe() -> VerificationRecipe:
    return VerificationRecipe(
        topology_class=PGA,
        source_ref="CIS column readout — programmable-gain amplifier (OTA in resistive feedback)",
        build={"method": "template", "template_ref": "column_pga_ac"},
        conditions=_tt(),
        sweeps=[{"analysis": "ac", "knob": "Rf",
                 "points": ["500k", "1000k", "2000k", "4000k"], "measure": ["acl_db"]}],
        claim_cards=[
            _claim("pga_gain", PGA, "Rf", "acl_db", QuantTest(kind="direction", sign="+"),
                   "The closed-loop gain of the inverting column PGA is -Rf/Rin, set by the feedback "
                   "RATIO rather than the device transconductances, so the gain is programmed by the "
                   "feedback network and rises with Rf (the programmable-gain property). Measured "
                   "tracking 20log(Rf/Rin) within ~1 dB; the small shortfall is the OTA's finite "
                   "open-loop gain (Rf is kept >> the OTA output resistance so resistive loading does "
                   "not tank the gain — the loading sensitivity is why CIS PGAs often use "
                   "switched-capacitor feedback instead)."),
        ],
    )


PTAT = "ptat_ctat_core_bjt"


def _ptat_recipe() -> VerificationRecipe:
    """S3-inc2a W2 — dVBE reference core (TOPOLOGY_BACKLOG.md #22/#23; the bandgap family's smallest,
    no-amp-in-loop member; sky130-only for now — see pdks.py's BjtUnavailable guard). REWORKED
    post-review (must-fix finding): the original design used two DIFFERENT-SIZE sky130 pnp_05v5
    subckts at equal collector current (an emitter-AREA ratio) — this did NOT hand-anchor
    (scratchpad/verify_ptat_anchor_adversarial.py): the two discrete sizes carry independently-fit
    `.model` cards (not a geometry-scaled pair), and a topology bug meant the two BJTs never actually
    carried equal current at all. The fix uses the IDENTICAL sky130 pnp_05v5 "unit" subckt for BOTH
    BJTs, with the 1:5 emitter-CURRENT ratio set cleanly by a 3-leg PMOS mirror's `m=` device
    multiplicity (Is-independent by construction — see templates.py's module comment for the full
    probe trail, scratchpad/probe_ptat_5..9). Only the two DIRECTION claims ship: the
    elasticity/near-linearity-in-absolute-temperature claim (probe #9: log-log slope of dVBE vs
    absolute temperature = 0.876, close to the idealized PTAT law's 1.0 — the reworked circuit DOES
    certify cleanly, unlike the old one's ~6.9/~0.60) is still NOT shipped as a formal
    QuantTest(kind="elasticity"): the growth engine's canonical x-series for this template is `temp`
    recorded in CELSIUS (spanning zero over the default 0-85C sweep), and the oracle's log-log
    regression takes `math.log(x)` directly (oracle.py) — a Celsius x-axis crossing zero cannot host
    an elasticity claim without re-basing the recorded x to Kelvin (out of scope here), so per spec
    §3 ('do not force') it stays a probe-documented finding, not a certified claim-card assertion."""
    return VerificationRecipe(
        topology_class=PTAT,
        source_ref="S3-inc2a W2 — dVBE reference core (TOPOLOGY_BACKLOG.md #22/#23), realized on "
                   "sky130's parasitic PNP at a 1:5 CURRENT ratio (same device, mirror-set)",
        build={"method": "template", "template_ref": "ptat_ctat_core_bjt"},
        conditions=_tt(),
        sweeps=[{"analysis": "dc", "knob": "temp", "points": ["0", "85"],
                 "measure": ["vbe_v", "iptat_a"]}],
        claim_cards=[
            _claim("ptat_iptat", PTAT, "temp", "iptat_a", QuantTest(kind="direction", sign="+"),
                   "iptat_a is the derived (V(n2)-V(n1))/R quantity — dVBE developed between two "
                   "IDENTICAL sky130 parasitic PNPs biased at a 1:5 CURRENT ratio (a mirror `m=` "
                   "device multiplicity, not an emitter-area ratio, so Is cancels exactly): dVBE = "
                   "VT*ln(5) plus a small, hand-explained constant series-resistance offset (~6.9mV, "
                   "stable within ~0.2% across 0-85C — probe #9/#18); VT = kT/q rises with absolute "
                   "temperature, so iptat_a rises with temp (measured 22.36uA @ 0C -> 28.36uA @ 85C "
                   "on the default sizing — a gentle ~1.27x rise, not the old design's spurious 6.4x, "
                   "because the bias current level itself stays nearly fixed here)."),
            _claim("ptat_vbe", PTAT, "temp", "vbe_v", QuantTest(kind="direction", sign="-"),
                   "The reference diode's own base-emitter voltage falls with temperature (the classic "
                   "CTAT ~-2mV/K law) at an ESSENTIALLY FIXED bias current (probe #18's device query: "
                   "I(XQ1) varies only ~3% over the whole 0-85C sweep, 2.03uA->2.10uA, since IBIAS is "
                   "an ideal fixed source and the 1:1 mirror leg tracks it closely) — measured "
                   "0.619294V @ 0C -> 0.448367V @ 85C on the default sizing."),
        ],
    )


def seed_recipes() -> list[VerificationRecipe]:
    """The validated seed recipes, in projection order."""
    return [_ota_recipe(), _cm_recipe(), _cs_recipe(), _cg_recipe(), _sf_recipe(), _dp_recipe(),
            _casc_recipe(), _ota5t_recipe(), _tele_recipe(), _fc_recipe(), _rgc_recipe(),
            _cmp_recipe(), _cds_recipe(), _ramp_recipe(), _pga_recipe(), _ptat_recipe()]


# --- Stat-QT statistical/corner/Pelgrom specimens (off-nominal: device mismatch + process corners) ---
_CAVEAT = ("NOTE: this is the PRE-CDS static mismatch; a real CIS readout's CDS/auto-zero cancels most "
           "static offset (see cds_switched_cap_nmos), and small-node residual FPN is kTC/RTS-dominated, "
           "which a deterministic DC-mismatch MC does not capture.")
_SKY = {"pdk": "sky130", "node": "130nm"}
OTA5T = "ota_5t_nmos_in"
CMP = "comparator_continuous_nmos"


def _ota5t_offset_recipe() -> VerificationRecipe:
    cond = AnalogPVT(corner="tt_mm", temp_c=27.0, vdd=1.8, mc_runs=200, pdk_profile=_SKY)
    return VerificationRecipe(
        topology_class=OTA5T, build={"method": "template", "template_ref": "ota5t_offset_mc"},
        conditions=cond, sweeps=[{"analysis": "mc", "knob": "vos", "measure": ["vos_v"]}],
        claim_cards=[ClaimCard(id="ota5t_vos", topology_class=OTA5T, conditions=cond,
            mechanism=MechanismClaim(knob="vos", metric="vos_v", series_ref="ota5t_vos",
                quant=QuantTest(kind="statistical", reducer="three_sigma", bound=0.020, absolute=True),
                narrative="input-referred offset is mismatch-dominated (Pelgrom); 3σ bounds it. " + _CAVEAT))])


def _ota5t_pelgrom_recipe() -> VerificationRecipe:
    cond = AnalogPVT(corner="tt_mm", temp_c=27.0, vdd=1.8, mc_runs=200, areas=[0.25, 1.0, 4.0], pdk_profile=_SKY)
    return VerificationRecipe(
        topology_class=OTA5T, build={"method": "template", "template_ref": "ota5t_offset_mc"},
        conditions=cond,
        sweeps=[{"analysis": "pelgrom", "knob": "vos", "areas": [0.25, 1.0, 4.0], "measure": ["a_vos"]}],
        claim_cards=[ClaimCard(id="ota5t_pelgrom", topology_class=OTA5T, conditions=cond,
            mechanism=MechanismClaim(knob="vos", metric="a_vos", series_ref="ota5t_pelgrom",
                quant=QuantTest(kind="elasticity", band=(-0.6, -0.2)),
                narrative="input-referred offset σ decreases as a POWER LAW with device area (Pelgrom-type: "
                          "bigger device -> less mismatch). Seeded n=200 measurement (E1b, 2026-07-04): "
                          "sky130 exponent -0.46 over 0.25-4x — and gf180 (-0.48) / ihp PSP103 (-0.47) "
                          "cluster there too, i.e. open foundry models sit CLOSE to ideal Pelgrom (-0.5), "
                          "only slightly shallow. (The earlier -0.375 was superseded: an unseeded small-n "
                          "instrument artifact, cov 34% vs 4% seeded.) The node-portable PRINCIPLE is the "
                          "power law; the exponent is set by YOUR foundry's mismatch model. " + _CAVEAT))])


def _ota5t_gbw_corner_recipe() -> VerificationRecipe:
    cond = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8, corners=["tt", "ss", "ff", "sf", "fs"], pdk_profile=_SKY)
    return VerificationRecipe(
        topology_class=OTA5T, build={"method": "template", "template_ref": "ota5t_gbw"},
        conditions=cond, sweeps=[{"analysis": "corner", "knob": "gbw", "measure": ["gbw_hz"]}],
        claim_cards=[ClaimCard(id="ota5t_gbw_corner", topology_class=OTA5T, conditions=cond,
            mechanism=MechanismClaim(knob="gbw", metric="gbw_hz", series_ref="ota5t_gbw_corner",
                quant=QuantTest(kind="corner", sign="+", bound=29e6),
                narrative="GBW ≥ 29 MHz at every process corner (process-robust; mismatch is the dominant axis)"))])


def _comparator_fpn_recipe() -> VerificationRecipe:
    cond = AnalogPVT(corner="tt_mm", temp_c=27.0, vdd=1.8, mc_runs=200, pdk_profile=_SKY)
    return VerificationRecipe(
        topology_class=CMP, build={"method": "template", "template_ref": "comparator_fpn_mc"},
        conditions=cond, sweeps=[{"analysis": "mc", "knob": "vos", "measure": ["vos_v"]}],
        claim_cards=[ClaimCard(id="col_fpn", topology_class=CMP, conditions=cond,
            mechanism=MechanismClaim(knob="vos", metric="vos_v", series_ref="col_fpn",
                quant=QuantTest(kind="statistical", reducer="k_sigma", n_col=2048, bound=0.060, absolute=True),
                narrative="column FPN = the WORST of N_col comparators' trip-offsets (k=Φ⁻¹(1-1/2048)·σ). " + _CAVEAT))])


def statistical_seed_recipes() -> list[VerificationRecipe]:
    """Stat-QT off-nominal specimens (device mismatch + process corners), judged in-memory."""
    return [_ota5t_offset_recipe(), _ota5t_pelgrom_recipe(), _ota5t_gbw_corner_recipe(), _comparator_fpn_recipe()]


# --- E3-I1 intervention pilot recipes (spec §3/§5) — REPORT-LEVEL PILOT ONLY. Deliberately NOT wired
#     into seed_recipes()/statistical_seed_recipes(): these are the do-operator pilot cards I2 runs for
#     real via the experiment path (docs/superpowers/specs/2026-07-04-e3-intervention-experiments.md
#     §7), not part of the standing project_executable() projection registry yet. Each claim's `id`
#     embeds the intervention id (spec §2: "the intervention id in its claim id and grounds") — the
#     executor ALSO stamps `grounds`/`scope` defensively (executor.py's scope-stamping loop), so this
#     is belt-and-suspenders, not the only place the identity lives. ---

_RZ_NULL = "rz_null"
_FF_BREAK = "ff_break"


def _rz_null_pm_recipe() -> VerificationRecipe:
    """Q2a (spec §1/§3): the nulling resistor. Rz swept 10ohm ("0-ish" — see templates.py's
    render_miller_ota_rz_null_ac module comment for why literal 0 does not converge) to 4500ohm
    (~2/gm2, gm2 measured live ~=4.458e-4 S) at fixed nominal Cc/CL. Prediction: delta-PM(Rz) is
    positive-going (live-measured 2026-07-04: +0.09..+36.66 deg over the sweep — direction '+')."""
    return VerificationRecipe(
        topology_class=OTA,
        source_ref="E3-I1 pilot — intervention 'rz_null' (Q2a: the nulling resistor)",
        build={"method": "template", "template_ref": "miller_ota_ac"},
        conditions=_tt(),
        sweeps=[{"analysis": "intervention", "intervention": _RZ_NULL, "knob": "Rz",
                 "points": ["10", "1000", "2000", "3000", "4500"], "measure": ["pm_deg"]}],
        claim_cards=[
            ClaimCard(
                id="rz_pm_delta__rz_null", topology_class=OTA, conditions=_tt(),
                grounds=[f"intervention:{_RZ_NULL}"],
                mechanism=MechanismClaim(
                    knob="Rz", metric="pm_deg", series_ref="rz_pm_delta__rz_null",
                    quant=QuantTest(kind="direction", sign="+"),
                    narrative=(
                        "the nulling resistor Rz, placed in series with the Miller cap Ccomp, pushes "
                        "the feedforward RHP zero toward (and past) 1/gm2, so phase margin at fixed "
                        "nominal Cc IMPROVES as Rz rises toward 1/gm2 — an intervention-certified "
                        "causal claim (Rz's own effect on PM, isolated from every other sizing "
                        "choice), not a narrative inference over an un-intervened sweep."
                    ),
                ),
            ),
        ],
    )


def _ff_break_pm_recipe() -> VerificationRecipe:
    """Q2b (spec §1/§3): the sharper do-operator. Cc swept over the baseline's OWN range under the
    ff_break intervention (ideal unity buffer severing the feedforward path — see templates.py's
    render_miller_ota_ff_break_ac module comment for the live-validation transcript, incl. a rejected
    first attempt). Prediction: delta-PM(Cc) = PM_ff_break(Cc) - PM_baseline(Cc) is positive and GROWS
    with Cc (live-measured 2026-07-04: +11.29 deg @ 250f -> +24.03 deg @ 4000f — direction '+')."""
    return VerificationRecipe(
        topology_class=OTA,
        source_ref="E3-I1 pilot — intervention 'ff_break' (Q2b: the feedforward break)",
        build={"method": "template", "template_ref": "miller_ota_ac"},
        conditions=_tt(),
        sweeps=[{"analysis": "intervention", "intervention": _FF_BREAK, "knob": "Cc",
                 "points": ["250f", "500f", "1000f", "2000f", "4000f"], "measure": ["pm_deg"]}],
        claim_cards=[
            ClaimCard(
                id="cc_pm_delta__ff_break", topology_class=OTA, conditions=_tt(),
                grounds=[f"intervention:{_FF_BREAK}"],
                mechanism=MechanismClaim(
                    knob="Cc", metric="pm_deg", series_ref="cc_pm_delta__ff_break",
                    quant=QuantTest(kind="direction", sign="+"),
                    narrative=(
                        "severing the feedforward current path through Cc (an ideal unity-gain "
                        "buffer, in the model, replicating the output node so Cc is driven from the "
                        "buffered replica rather than the real output — Miller pole-splitting as seen "
                        "from the first-stage node is preserved) removes the RHP zero's phase "
                        "penalty: delta-PM = PM_ff_break(Cc) - PM_baseline(Cc) is positive and GROWS "
                        "with Cc, since the zero z~=gm2/Cc falls toward GBW at larger Cc and so costs "
                        "the un-intervened baseline more phase there. Certified under an IDEALIZED "
                        "intervention (an ideal buffer is itself an abstraction) — never taught as "
                        "more than 'in the model, with this idealized intervention'. PHYSICS-"
                        "ATTRIBUTION CAVEAT: severing Cc's feedforward current necessarily also "
                        "removes Cc's loading of the output node (the same current), so the measured "
                        "delta-PM contains a small output-pole-unloading component alongside the "
                        "RHP-zero removal — the certifiable statement is that the feedforward PATH "
                        "causes the PM degradation component, never that the zero alone accounts for "
                        "the full delta."
                    ),
                ),
            ),
        ],
    )


def _ff_break_av0_invariance_recipe() -> VerificationRecipe:
    """The intervention's OWN validity check (spec §3): severing the feedforward path must NOT change
    DC gain — a nonzero delta would mean the ff_break variant altered more than the hypothesized
    pathway. Uses `spread_max`, not `cov_max`: the derived DELTA series is expected to sit at (or
    extremely near) zero, and oracle.judge_quant's invariance branch FLAGs a zero-mean series under
    `cov_max` ("CoV undefined") by design (oracle.py's own I6 guidance: use the scale-aware absolute
    bound for exactly this shape) — `spread_max` is the correct existing field for a near-zero delta,
    not a new oracle kind (Q1's bet stays unbent). Live-measured 2026-07-04: delta EXACTLY 0.000 dB at
    every swept Cc point (68.8739 dB on both the baseline and the ff_break variant)."""
    return VerificationRecipe(
        topology_class=OTA,
        source_ref="E3-I1 pilot — intervention 'ff_break' DC-gain invariance control",
        build={"method": "template", "template_ref": "miller_ota_ac"},
        conditions=_tt(),
        sweeps=[{"analysis": "intervention", "intervention": _FF_BREAK, "knob": "Cc",
                 "points": ["250f", "500f", "1000f", "2000f", "4000f"], "measure": ["av0_db"]}],
        claim_cards=[
            ClaimCard(
                id="cc_av0_delta__ff_break", topology_class=OTA, conditions=_tt(),
                grounds=[f"intervention:{_FF_BREAK}"],
                mechanism=MechanismClaim(
                    knob="Cc", metric="av0_db", series_ref="cc_av0_delta__ff_break",
                    quant=QuantTest(kind="invariance", spread_max=0.05),
                    narrative=(
                        "the ff_break intervention's own validity check: an ideal buffer that only "
                        "reroutes the feedforward current path must NOT move the DC operating point "
                        "or the DC gain — delta-Av0 = Av0_ff_break(Cc) - Av0_baseline(Cc) is bounded "
                        "near zero at every swept Cc (live-measured: exactly 0.000 dB), confirming the "
                        "intervention did not disturb the DC operating point. PHYSICS-ATTRIBUTION "
                        "CAVEAT: this DC-gain invariance confirms only that the DC bias point held — "
                        "it does NOT show the severed feedforward current has no OTHER (AC) effect. "
                        "Severing that same current also removes Cc's loading of the output node, so "
                        "the paired cc_pm_delta__ff_break card's measured delta-PM contains a small "
                        "output-pole-unloading component alongside the RHP-zero removal; the "
                        "certifiable statement there is that the feedforward PATH causes the PM "
                        "degradation component, never that the zero alone accounts for the full delta."
                    ),
                ),
            ),
        ],
    )


def intervention_seed_recipes() -> list[VerificationRecipe]:
    """The §3 pilot cards (Q2a direction, Q2b direction, ff_break's own av0 invariance control) —
    NOT wired into seed_recipes()/statistical_seed_recipes() (report-level pilot; I2 runs these via
    the experiment path, per spec §4's work split)."""
    return [_rz_null_pm_recipe(), _ff_break_pm_recipe(), _ff_break_av0_invariance_recipe()]
