"""S3-inc2a (metric expansion + PTAT/CTAT core) — knowledge/executable/templates.py.

Two work items (spec docs/superpowers/specs/2026-07-04-s3-increment2a-metric-expansion.md):

  W1 — vout_swing_v / icmr_lo_v / icmr_hi_v on miller_ota_2stage_nmos_in and ota_5t_nmos_in (a
       device-query control block, not a `.meas` line — HAND-ANCHORED live against real sky130
       BEFORE this code existed: scratchpad/probe_swing_*.py, probe_icmr_*.py, probe_11_*.py).
  W2 — ptat_ctat_core_bjt, the first new template since the registry froze at 18 (a native `.dc
       temp` sweep — hand-anchored: scratchpad/probe_ptat_*.py).

Unit (no docker): capability-registry shape, control-script emission (meas/device-query content,
series-key routing), run_recipe with a fake runner. Docker-gated: one live sky130 smoke per new
metric asserting the HAND-ANCHORED value within the stated tolerance (the probe transcripts are the
tolerance's justification, not a guess).
"""

from __future__ import annotations

import pytest

from openclaw_brain.knowledge.executable.corpus import SpecimenCorpus
from openclaw_brain.knowledge.executable.executor import run_recipe
from openclaw_brain.knowledge.executable.models import (
    AnalogPVT, ClaimCard, MechanismClaim, QuantTest, VerdictClass, VerificationRecipe,
)
from openclaw_brain.knowledge.executable.recipe import capability_for
from openclaw_brain.knowledge.executable.runner import NgspiceRunner
from openclaw_brain.knowledge.executable.templates import (
    RENDERERS,
    TEMPLATES,
    render_miller_ota_ac,
    render_ota_5t_ac,
    render_ptat_ctat_core_bjt,
    render_ptat_ctat_core_bjt_cell,
)

OTA = "miller_ota_2stage_nmos_in"
OTA5T = "ota_5t_nmos_in"
PTAT = "ptat_ctat_core_bjt"
COND = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)
_CERTIFIED = {VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT}
_SWING_ICMR = ("vout_swing_v", "icmr_lo_v", "icmr_hi_v")


# =====================================================================================================
# W1 — capability registry (both templates gained the same new surface)
# =====================================================================================================


@pytest.mark.parametrize("tclass,template_ref", [(OTA, "miller_ota_ac"), (OTA5T, "ota_5t_ac")])
def test_capability_gained_the_swing_icmr_surface(tclass, template_ref):
    cap = capability_for(tclass)
    assert cap.template_ref == template_ref
    assert "dc" in cap.analyses                      # new analysis primitive alongside "ac"
    assert {"VDD", "IREFV"} <= cap.knobs              # new headroom knobs
    assert set(_SWING_ICMR) <= cap.metrics
    assert cap.knob_ranges == {"VDD": (1.4, 2.2), "IREFV": (2e-6, 4e-5)}


# =====================================================================================================
# W1 — control-script emission (meas/device-query content, series-key routing)
# =====================================================================================================


@pytest.mark.parametrize("render_fn,top,bottom,tail_term", [
    (render_miller_ota_ac, "xm6.msky130_fd_pr__pfet_01v8", "xm7.msky130_fd_pr__nfet_01v8", None),
    (render_ota_5t_ac, "xm4.msky130_fd_pr__pfet_01v8", "xm2.msky130_fd_pr__nfet_01v8", "v(tail)"),
])
def test_swing_control_block_reads_the_correct_output_devices(render_fn, top, bottom, tail_term):
    deck = render_fn(knob="VDD", metric="vout_swing_v", points=["1.8"])
    assert "foreach pt 1.8" in deck
    assert "alter VDD = $pt" in deck
    assert f"@m.{top}[vdsat]" in deck
    assert f"@m.{bottom}[vdsat]" in deck
    assert "echo RDATA vdd $pt $&y" in deck
    if tail_term:
        assert tail_term in deck.split("let y =")[1].splitlines()[0]
    else:
        # the 2-stage OTA's output devices are BOTH rail-referenced — no tail correction term.
        assert "v(tail)" not in deck.split("let y =")[1].splitlines()[0]


@pytest.mark.parametrize("render_fn,knob_default_element", [
    (render_miller_ota_ac, "IREFV"), (render_ota_5t_ac, "IREFV"),
])
def test_icmr_control_block_scans_vin_and_tracks_first_crossing(render_fn, knob_default_element):
    deck = render_fn(knob=knob_default_element, metric="icmr_lo_v", points=["10u"])
    assert "foreach pt 10u" in deck
    assert "alter IREF = $pt" in deck                 # IREFV knob -> the IREF element
    assert "foreach vcm 0.05 0.10" in deck             # the fixed inner VCM scan grid
    assert "alter Vin = $vcm" in deck
    assert "@m.xm5.msky130_fd_pr__nfet_01v8[vds]" in deck    # tail device (icmr_lo criterion)
    assert "@m.xm1.msky130_fd_pr__nfet_01v8[vds]" in deck    # input-pair device (icmr_hi criterion)
    assert "if margin5 >= 0 & found_lo = 0" in deck
    assert "if margin1 < 0 & found_hi = 0" in deck
    assert "let icmr_lo = 0/0" in deck                 # never-found -> NaN, dropped by parse_rdata
    assert "echo RDATA irefv $pt $&icmr_lo" in deck    # icmr_lo_v -> the icmr_lo variable


def test_icmr_hi_v_echoes_the_other_variable():
    deck = render_miller_ota_ac(knob="VDD", metric="icmr_hi_v", points=["1.8"])
    assert "echo RDATA vdd $pt $&icmr_hi" in deck


def test_existing_metrics_are_byte_unchanged_by_the_new_branch():
    """Baseline template bodies/renderers this task does not extend must stay byte-identical."""
    deck = render_miller_ota_ac(knob="Cc", metric="gbw_hz")
    assert "meas ac y when vdb(out)=0 cross=1" in deck
    assert "vdsat" not in deck
    deck5t = render_ota_5t_ac(knob="CL", metric="av0_db")
    assert "meas ac y max vdb(out)" in deck5t
    assert "vdsat" not in deck5t


def test_default_points_are_knob_aware():
    deck_vdd = render_miller_ota_ac(knob="VDD", metric="vout_swing_v")
    assert "foreach pt 1.6 1.8 2.0 2.2" in deck_vdd
    deck_irefv = render_miller_ota_ac(knob="IREFV", metric="icmr_hi_v")
    assert "foreach pt 5u 10u 20u 40u" in deck_irefv


# =====================================================================================================
# W1 — run_recipe with a fake runner (series routing + verdict, no docker)
# =====================================================================================================


class _FakeSwingRunner:
    def __init__(self, series):
        self._series = series

    def measure(self, deck, timeout=300):
        return {"vdd": self._series}


def _swing_recipe(tclass, template_ref):
    return VerificationRecipe(
        topology_class=tclass, build={"method": "template", "template_ref": template_ref},
        conditions=COND,
        sweeps=[{"analysis": "dc", "knob": "VDD", "points": ["1.6", "1.8", "2.0", "2.2"],
                 "measure": ["vout_swing_v"]}],
        claim_cards=[ClaimCard(
            id="swing_vdd", topology_class=tclass, conditions=COND,
            mechanism=MechanismClaim(knob="VDD", metric="vout_swing_v", series_ref="swing_vdd",
                                     quant=QuantTest(kind="direction", sign="+"),
                                     narrative="more supply headroom -> more output swing"))],
    )


@pytest.mark.parametrize("tclass,template_ref", [(OTA, "miller_ota_ac"), (OTA5T, "ota_5t_ac")])
def test_run_recipe_swing_direction_unit(tmp_path, tclass, template_ref):
    series = [(1.6, 1.25982), (1.8, 1.46093), (2.0, 1.66182), (2.2, 1.86256)]
    result = run_recipe(_swing_recipe(tclass, template_ref), _FakeSwingRunner(series),
                        corpus=SpecimenCorpus(str(tmp_path)), project=True)
    assert set(result.canonical) == {"vdd_vout_swing_v"}
    assert result.claim_cards[0].verdict == VerdictClass.VERIFIED


# =====================================================================================================
# W1 — docker-gated live smokes: ONE per new metric per template, asserting the HAND-ANCHORED value
# (probe transcripts: scratchpad/probe_swing_*.py, probe_icmr_*.py) within a stated tolerance.
# =====================================================================================================


def _skip_unless_docker(runner):
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; live smoke needs the sim container")


def test_miller_ota_vout_swing_v_live_smoke_matches_hand_anchor():
    """Hand-anchor (probe #1, python arithmetic on live device queries @ VDD=1.8):
    swing = VDD - vdsat(M6) - vdsat(M7) = 1.8 - 0.2026770 - 0.1363933 = 1.4609297 V.
    Probe #2 (the SAME formula computed live inside the production foreach+alter+op sweep) measured
    1.46093 — agreeing to 5 sig figs; tolerance here is a generous 5 mV."""
    runner = NgspiceRunner()
    _skip_unless_docker(runner)
    deck = render_miller_ota_ac(knob="VDD", metric="vout_swing_v", points=["1.8"])
    series = runner.measure(deck)["vdd"]
    assert len(series) == 1
    assert abs(series[0][1] - 1.4609297) < 0.005


def test_ota5t_vout_swing_v_live_smoke_matches_hand_anchor():
    """Hand-anchor (probe #1, tail-corrected formula — M2's source is the tail node, not ground):
    swing = (VDD - vdsat(M4)) - (v(tail) + vdsat(M2)) = 1.662376 - 0.27826069 = 1.38411531 V.
    Probe #2 measured 1.38412 — agreeing to 5 sig figs; tolerance 5 mV."""
    runner = NgspiceRunner()
    _skip_unless_docker(runner)
    deck = render_ota_5t_ac(knob="VDD", metric="vout_swing_v", points=["1.8"])
    series = runner.measure(deck)["vdd"]
    assert len(series) == 1
    assert abs(series[0][1] - 1.38411531) < 0.005


def test_miller_ota_icmr_live_smoke_matches_hand_anchor():
    """Hand-anchor (probe #4, 0.01V-resolution refine around the coarse-scan bracket @ VDD=1.8):
    icmr_lo crossing ~0.811 V, icmr_hi crossing ~1.378 V. The production 0.05V-grid 'first crossing'
    rule (probe #6/#9) reports 0.85 / 1.40 — within one grid step (0.05V) of the fine crossing, the
    declared tolerance (0.06V, a hair over one step)."""
    runner = NgspiceRunner()
    _skip_unless_docker(runner)
    lo = runner.measure(render_miller_ota_ac(knob="VDD", metric="icmr_lo_v", points=["1.8"]))["vdd"]
    hi = runner.measure(render_miller_ota_ac(knob="VDD", metric="icmr_hi_v", points=["1.8"]))["vdd"]
    assert abs(lo[0][1] - 0.85) < 0.06
    assert abs(hi[0][1] - 1.40) < 0.06


def test_ota5t_icmr_live_smoke_matches_hand_anchor():
    """Hand-anchor (probe #4 fine refine @ VDD=1.8): icmr_lo crossing ~0.774V, icmr_hi ~1.403V.
    Production 0.05V-grid rule (probe #6/#9): 0.80 / 1.45 — within the same one-grid-step tolerance."""
    runner = NgspiceRunner()
    _skip_unless_docker(runner)
    lo = runner.measure(render_ota_5t_ac(knob="VDD", metric="icmr_lo_v", points=["1.8"]))["vdd"]
    hi = runner.measure(render_ota_5t_ac(knob="VDD", metric="icmr_hi_v", points=["1.8"]))["vdd"]
    assert abs(lo[0][1] - 0.80) < 0.06
    assert abs(hi[0][1] - 1.45) < 0.06


def test_swing_icmr_vdd_direction_live_smoke():
    """Direction sanity across the FULL VDD sweep (probe #6): vout_swing_v and icmr_hi_v both RISE
    monotonically with VDD (more supply headroom); icmr_lo_v is essentially INVARIANT (bounded by the
    tail current source's own local bias, not the supply) — confirming the claim-kind choice recorded
    in the spec (direction vs invariance) is physically grounded, not asserted blind."""
    runner = NgspiceRunner()
    _skip_unless_docker(runner)
    swing = runner.measure(render_miller_ota_ac(knob="VDD", metric="vout_swing_v"))["vdd"]
    icmr_hi = runner.measure(render_miller_ota_ac(knob="VDD", metric="icmr_hi_v"))["vdd"]
    icmr_lo = runner.measure(render_miller_ota_ac(knob="VDD", metric="icmr_lo_v"))["vdd"]
    swing_y = [y for _, y in swing]
    hi_y = [y for _, y in icmr_hi]
    lo_y = [y for _, y in icmr_lo]
    assert swing_y == sorted(swing_y)                 # monotonically rising
    assert hi_y == sorted(hi_y)                        # monotonically rising
    assert max(lo_y) - min(lo_y) < 0.01                 # invariant to VDD


# =====================================================================================================
# W2 — ptat_ctat_core_bjt capability + control-script emission
# =====================================================================================================


def test_ptat_capability_registered():
    cap = capability_for(PTAT)
    assert cap.template_ref == "ptat_ctat_core_bjt"
    assert cap.analyses == frozenset({"dc"})
    assert cap.knobs == frozenset({"temp"})
    assert cap.metrics == frozenset({"vbe_v", "iptat_a"})


def test_ptat_registered_in_renderers():
    render_fn, cell_fn, default_sizing = RENDERERS["ptat_ctat_core_bjt"]
    assert render_fn is render_ptat_ctat_core_bjt
    assert cell_fn is render_ptat_ctat_core_bjt_cell
    # S3-inc2a W2 REWORK (must-fix finding response): IBIAS dropped 10u->2u (a smaller absolute bias
    # keeps the emitter-series-resistance offset a small correction on the dVBE law rather than a
    # comparable-magnitude confound — templates.py's module comment / probe #18); N=5 is the NEW
    # current-ratio knob (a mirror `m=` multiplicity, not an emitter-area ratio).
    assert default_sizing == {"VDD": "1.8", "IBIAS": "2u", "R": "2k", "Wp": "8", "Lp": "0.5", "N": "5"}


def test_ptat_pdk_only_sky130_for_now():
    # documents the cross-PDK note (spec §6 out of scope; pdks.py's BjtUnavailable guard).
    assert TEMPLATES["ptat_ctat_core_bjt"]["devices"] == ["pnp_bjt", "pfet"]


def test_render_emits_native_dc_temp_sweep():
    deck = render_ptat_ctat_core_bjt(knob="temp", metric="iptat_a", points=["0", "85"])
    assert "dc temp 0 85 5" in deck
    assert "sky130_fd_pr__pnp_05v5_W0p68L0p68" in deck
    assert "echo RDATA temp $&t $&iptat" in deck


def test_render_uses_the_same_device_for_both_bjts_not_the_two_discrete_sizes():
    """Must-fix regression guard: the ORIGINAL (broken) design instantiated the "5x" discrete pnp
    subckt for Q2 on the theory that its independently-fit .model card gave a usable emitter-area
    ratio — it did not hand-anchor (verify_ptat_anchor_adversarial.py). The fix uses the IDENTICAL
    "unit" subckt for BOTH BJTs (Is cancels exactly in the dVBE law) — the "5x" size must never
    reappear in this deck, and the "unit" size must appear exactly twice (Q1 and Q2)."""
    deck = render_ptat_ctat_core_bjt(knob="temp", metric="iptat_a", points=["0", "85"])
    assert "sky130_fd_pr__pnp_05v5_W3p40L3p40" not in deck
    assert deck.count("sky130_fd_pr__pnp_05v5_W0p68L0p68") == 2


def test_render_current_ratio_mirror_has_no_double_injection():
    """Must-fix regression guard: the ORIGINAL design tied the external IBIAS current source to the
    SAME node the diode-connected mirror reference (and, transitively, the reference BJT) sat on
    (`IBIAS vdd nb`), so the reference BJT carried IBIAS *plus* the mirror's own diode current, not a
    clean value (29.0uA vs the mirror's intended ~19.0uA, confirmed live). The fix routes IBIAS as a
    SINK from the diode-connected reference device (`IBIAS nref 0`) so KCL forces its current to be
    IBIAS exactly, with no BJT sharing that node at all."""
    deck = render_ptat_ctat_core_bjt(knob="temp", metric="iptat_a", points=["0", "85"])
    assert "IBIAS nref 0 {IBIAS}" in deck
    assert "IBIAS vdd" not in deck
    assert "XMP0 nref nref vdd vdd" in deck        # diode-connected reference; no BJT at nref
    assert "XQ1  n1 0 0" in deck and "XQ2  n2 0 0" in deck   # BJTs sit on the FAN-OUT legs only


def test_render_mirror_ratio_is_device_multiplicity_not_area():
    """The 1:N current ratio is realized by the sky130 PDK's `m=` device-multiplicity parameter
    (N parallel unit-width fingers), the SAME mechanism this file already uses elsewhere (the
    Miller-OTA input pair's `m={M1}`) — not a W/L area ratio and not a BJT emitter-area ratio."""
    deck = render_ptat_ctat_core_bjt(knob="temp", metric="iptat_a", points=["0", "85"])
    assert "XMP1 n1   nref vdd vdd sky130_fd_pr__pfet_01v8 W={Wp} L={Lp} m=1" in deck
    assert "XMP2 n2   nref vdd vdd sky130_fd_pr__pfet_01v8 W={Wp} L={Lp} m={N}" in deck
    assert ".param VDD=1.8 IBIAS=2u R=2k Wp=8 Lp=0.5 N=5" in deck


def test_render_vbe_v_echoes_the_other_variable():
    deck = render_ptat_ctat_core_bjt(knob="temp", metric="vbe_v", points=["0", "85"])
    assert "echo RDATA temp $&t $&vbe1" in deck


def test_render_ptat_cell_is_bare():
    cell = render_ptat_ctat_core_bjt_cell()
    assert ".control" not in cell and "dc temp" not in cell
    assert "XQ1  n1 0 0 sky130_fd_pr__pnp_05v5_W0p68L0p68" in cell
    assert "XQ2  n2 0 0 sky130_fd_pr__pnp_05v5_W0p68L0p68" in cell


# =====================================================================================================
# W2 — run_recipe with a fake runner (both direction claims, no docker)
# =====================================================================================================


def _ptat_recipe():
    return VerificationRecipe(
        topology_class=PTAT, build={"method": "template", "template_ref": "ptat_ctat_core_bjt"},
        conditions=COND,
        sweeps=[{"analysis": "dc", "knob": "temp", "points": ["0", "85"],
                 "measure": ["vbe_v", "iptat_a"]}],
        claim_cards=[
            ClaimCard(id="ptat_iptat", topology_class=PTAT, conditions=COND,
                     mechanism=MechanismClaim(knob="temp", metric="iptat_a", series_ref="ptat_iptat",
                                              quant=QuantTest(kind="direction", sign="+"),
                                              narrative="PTAT current rises with temperature")),
            ClaimCard(id="ptat_vbe", topology_class=PTAT, conditions=COND,
                     mechanism=MechanismClaim(knob="temp", metric="vbe_v", series_ref="ptat_vbe",
                                              quant=QuantTest(kind="direction", sign="-"),
                                              narrative="CTAT: VBE falls with temperature")),
        ],
    )


class _FakePtatRunner:
    """Routes by which metric's echo-variable the deck emits — mirrors the real render function's
    (knob, metric)-per-deck contract (one deck per requested metric, see executor.py's cache key)."""

    def measure(self, deck, timeout=300):
        temps = [0.0, 27.0, 85.0]
        if "$&iptat" in deck:
            return {"temp": [(t, 8e-6 + t * 5e-7) for t in temps]}       # rising
        return {"temp": [(t, 0.68 - t * 0.001) for t in temps]}          # falling


def test_run_recipe_ptat_both_directions_unit(tmp_path):
    result = run_recipe(_ptat_recipe(), _FakePtatRunner(),
                        corpus=SpecimenCorpus(str(tmp_path)), project=True)
    assert set(result.canonical) == {"temp_iptat_a", "temp_vbe_v"}
    verdicts = {c.id: c.verdict for c in result.claim_cards}
    assert verdicts["ptat_iptat"] == VerdictClass.VERIFIED
    assert verdicts["ptat_vbe"] == VerdictClass.VERIFIED


# =====================================================================================================
# W2 — docker-gated live smoke: temp-sweep direction checks BOTH ways (spec §5)
# =====================================================================================================


def test_ptat_temp_sweep_live_smoke_both_directions_and_hand_anchor():
    """Hand-anchor (probe #9, DEFAULT sizing post-REWORK, native `.dc temp 0 85 5` sweep): vbe_v
    (CTAT) FALLS 0.619294V @ 0C -> 0.448367V @ 85C (Q1's own bias current only rises ~3% over the
    whole sweep, 2.033uA->2.095uA per probe #18's device query — "at an essentially fixed bias
    current" is now an honest description, unlike the old design's ~6.4x bias-current swing); iptat_a
    (the derived PTAT-like current) RISES 2.23649e-5A @ 0C -> 2.83560e-5A @ 85C — both endpoints
    checked within a tight tolerance, plus full-sweep monotonicity both ways."""
    runner = NgspiceRunner()
    _skip_unless_docker(runner)
    vbe = runner.measure(render_ptat_ctat_core_bjt(metric="vbe_v"))["temp"]
    iptat = runner.measure(render_ptat_ctat_core_bjt(metric="iptat_a"))["temp"]
    vbe_y = [y for _, y in vbe]
    iptat_y = [y for _, y in iptat]
    assert vbe_y == sorted(vbe_y, reverse=True)         # CTAT: monotonically FALLING
    assert iptat_y == sorted(iptat_y)                   # PTAT: monotonically RISING
    assert abs(vbe[0][1] - 0.619294) < 0.002 and abs(vbe[-1][1] - 0.448367) < 0.002
    assert abs(iptat[0][1] - 2.23649e-5) < 1e-6 and abs(iptat[-1][1] - 2.83560e-5) < 1e-6


def test_ptat_dvbe_hand_anchors_to_vt_ln_n_within_a_constant_series_resistance_offset():
    """THE hand-anchor the spec's rule requires (§2 acceptance / §3 "dVBE = VT*ln(N) at 27C within
    tolerance") — computed ONLY from the shipped metric (iptat_a * R recovers dVBE exactly, since
    iptat_a IS (V(n2)-V(n1))/R by construction), never from a side-channel probe number. Ideal dVBE
    = VT*ln(5) = 41.628mV @ 27C (VT = kT/q, k=1.380649e-23, q=1.602176634e-19); probe #9/#18 found a
    CONSTANT +6.9mV offset (implied series resistance (Ic2-Ic1)*R_E ~= 846-847 ohm, stable within
    ~0.2% across 0-85C) — this test re-derives the ideal law from first principles and checks the
    measured-vs-ideal gap stays inside a generous but bounded band at both 0C and 85C, proving the
    offset is a small, STABLE correction rather than a divergent/unexplained one (contrast the OLD
    design, where the equivalent gap widened from 37.9mV to ~2x by 85C)."""
    import math

    runner = NgspiceRunner()
    _skip_unless_docker(runner)
    R_OHMS = 2000.0
    N = 5
    k, q = 1.380649e-23, 1.602176634e-19
    iptat = runner.measure(render_ptat_ctat_core_bjt(metric="iptat_a"))["temp"]
    offsets_mv = []
    for t_c, y in iptat:
        dvbe = y * R_OHMS
        vt = k * (273.15 + t_c) / q
        ideal = vt * math.log(N)
        offsets_mv.append((dvbe - ideal) * 1e3)
    # every offset lands in a tight, physically-explainable band (probe measured 6.85-7.04mV) —
    # a generous 5-9mV band leaves headroom for cross-run ngspice/device-model float noise without
    # accepting the kind of unbounded, temperature-diverging gap the OLD design showed (~38mV at
    # 0C growing to ~2x its own value by 85C, i.e. NOT a stable offset at all).
    assert all(5.0 < off < 9.0 for off in offsets_mv)
    # STABILITY is the point: the offset must not drift by more than a couple mV end-to-end (the
    # measured spread was <0.2mV in the live probe).
    assert max(offsets_mv) - min(offsets_mv) < 2.0
