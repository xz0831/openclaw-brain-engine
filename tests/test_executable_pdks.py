"""Tests for the PDK profile layer (knowledge/executable/pdks.py) — spec
docs/superpowers/specs/2026-07-04-e1-cross-pdk-triangulation.md §5-I1, §6.

Unit (no docker): sky130A identity/byte-identity regression, substitution totality, unknown-pdk
raises before any sim, runner lib-glob resolution (mocked filesystem/subprocess), the PDK_SIZING_
OVERRIDES scaffold, and the executor plumb-through (fake runners — no real ngspice).

Integration (docker-gated): one live smoke per new PDK, rendering the current_mirror_simple_nmos
.dc deck (the E1 pilot's simplest template) and running it through real ngspice in the IIC-OSIC-
TOOLS container. Per spec §6 the bar is deck rendering + lib resolution + ngspice invocation — NOT
a physically-sane number (bias correction is I2's job). Both new PDKs were verified live during I1
(see pdks.py's module docstring for the full ground-truth transcripts): gf180mcuD actually converges
to a sane current-mirror number; ihp-sg13g2 converges too, but to a KNOWN-wrong magnitude (its
OSDI/PSP103 devices need explicitly u-suffixed geometry, not the bare numbers I1 leaves untouched —
that gap is documented, not silently accepted as correct).
"""

from __future__ import annotations

import copy
import os

import pytest

from openclaw_brain.knowledge.executable.executor import run_recipe
from openclaw_brain.knowledge.executable.models import (
    AnalogPVT, ClaimCard, MechanismClaim, QuantTest, VerificationRecipe,
)
from openclaw_brain.knowledge.executable.pdks import (
    MM_CORNER_TOKEN,
    PDK_PROFILES,
    PDK_SIZING_OVERRIDES,
    UnknownPDK,
    get_profile,
    sizing_overrides_for,
    substitute_devices,
)
from openclaw_brain.knowledge.executable.runner import NgspiceRunner, RUN_SH, render_run_sh
from openclaw_brain.knowledge.executable.seeds import seed_recipes
from openclaw_brain.knowledge.executable.templates import (
    render_cascode_mirror_dc,
    render_cds_tran,
    render_column_pga_ac,
    render_common_gate_ac,
    render_common_source_ac,
    render_current_mirror_dc,
    render_diff_pair_ac,
    render_folded_cascode_ac,
    render_miller_ota_ac,
    render_ota_5t_ac,
    render_ptat_ctat_core_bjt,
    render_ramp_tran,
    render_regulated_cascode_dc,
    render_source_follower_ac,
    render_telescopic_cascode_ac,
)

CM = "current_mirror_simple_nmos"
CS = "common_source_active_load_nmos"
OTA5T = "ota_5t_nmos_in"
CMP = "comparator_continuous_nmos"
_PILOT_RENDERERS = [render_current_mirror_dc, render_common_source_ac, render_ota_5t_ac]

# I1a wave A (spec docs/superpowers/specs/2026-07-05-full-registry-cross-pdk-rollout.md §3) — the 6
# "simpler bias structure" templates ported to gf180mcuD/ihp-sg13g2 in this increment. Kept as ITS OWN
# set of names/lists (not merged into the E1-pilot constants above) so the E1 parametrize ids and
# coverage stay byte-stable; wave-A coverage is purely additive.
OTA = "miller_ota_2stage_nmos_in"
CG = "common_gate_nmos"
SF = "source_follower_nmos"
DP = "diff_pair_resistive_nmos"
CASC = "cascode_current_mirror_nmos"
RGC = "regulated_cascode_nmos"
_WAVE_A_CLASSES = [OTA, CG, SF, DP, CASC, RGC]
_WAVE_A_RENDERERS = [
    render_miller_ota_ac, render_common_gate_ac, render_source_follower_ac,
    render_diff_pair_ac, render_cascode_mirror_dc, render_regulated_cascode_dc,
]
_WAVE_A_RENDER_FNS: dict[str, object] = dict(zip(_WAVE_A_CLASSES, _WAVE_A_RENDERERS))
# The u-suffixed geometry overrides I1a probed (pdks.py's PDK_SIZING_OVERRIDES comment has the full
# transcript) — identical magnitude on BOTH non-sky130 PDKs for every wave-A class except
# regulated_cascode_nmos, whose aux-amp bias VOLTAGE (`VBA`) is genuinely PDK-specific (see below).
_WAVE_A_EXPECTED_OVERRIDES: dict[str, dict[str, str]] = {
    OTA: {"W1": "8u", "W3": "4u", "W5": "4u", "W6": "16u", "W7": "8u", "W8": "2u", "Lp": "0.5u"},
    CG: {"Wn": "8u", "Lp": "0.5u"},
    SF: {"Wn": "8u", "Lp": "0.5u"},
    DP: {"Wn": "8u", "Lp": "0.5u"},
    CASC: {"W": "4u", "Lp": "0.5u"},
}
_RGC_EXPECTED_OVERRIDES: dict[str, dict[str, str]] = {
    "gf180mcuD": {"W": "4u", "WA": "4u", "Lp": "0.5u", "VBA": "1.8"},
    "ihp-sg13g2": {"W": "4u", "WA": "4u", "Lp": "0.5u", "VBA": "0.55"},
}

# I1b wave B (spec docs/superpowers/specs/2026-07-05-full-registry-cross-pdk-rollout.md §3) — the 6
# "headroom/switched/BJT" templates. Kept as its own set again (same reasoning as wave A above).
TELE = "telescopic_cascode_ota_nmos_in"
FC = "folded_cascode_ota_nmos_in"
CDS = "cds_switched_cap_nmos"
RAMP = "single_slope_ramp_generator"
PGA = "column_pga_inverting_nmos"
PTAT = "ptat_ctat_core_bjt"
_WAVE_B_CLASSES = [TELE, FC, CDS, RAMP, PGA, PTAT]
_WAVE_B_RENDERERS = [
    render_telescopic_cascode_ac, render_folded_cascode_ac, render_cds_tran,
    render_ramp_tran, render_column_pga_ac, render_ptat_ctat_core_bjt,
]
_WAVE_B_RENDER_FNS: dict[str, object] = dict(zip(_WAVE_B_CLASSES, _WAVE_B_RENDERERS))
# The wave-B overrides actually wired in pdks.py — 5/6 classes need ONLY the u-suffixed geometry
# override (folded_cascode_ota_nmos_in, cds_switched_cap_nmos/ihp, column_pga_inverting_nmos,
# ptat_ctat_core_bjt on BOTH pdks; cds_switched_cap_nmos/gf180mcuD needed a genuine RESIZE, Lp=0.3u
# not 0.15u — see pdks.py's ground truth); telescopic_cascode_ota_nmos_in/ihp-sg13g2 carries the same
# geometry override as gf180 but is STILL PORT-FAILED (a non-empty override here is not a success
# signal — see the live-smoke regression test below); single_slope_ramp_generator's I1b-era
# PORT-FAILED (hardcoded template literal) was CLOSED 2026-07-05 post-I1b: the switch length became
# the `LS` param and both pdks now carry the usual entry (gf180's LS=0.3u above nfet_03v3's lmin).
_WAVE_B_EXPECTED_OVERRIDES: dict[tuple[str, str], dict[str, str]] = {
    (TELE, "gf180mcuD"): {"WN": "8u", "WP": "16u", "WT": "16u", "Lp": "0.5u"},
    (TELE, "ihp-sg13g2"): {"WN": "8u", "WP": "16u", "WT": "16u", "Lp": "0.5u"},
    (FC, "gf180mcuD"): {"WN": "8u", "WP": "16u", "WT": "16u", "Lp": "0.5u"},
    (FC, "ihp-sg13g2"): {"WN": "8u", "WP": "16u", "WT": "16u", "Lp": "0.5u"},
    (CDS, "gf180mcuD"): {"WSW": "4u", "Lp": "0.3u"},
    (CDS, "ihp-sg13g2"): {"WSW": "4u", "Lp": "0.15u"},
    (RAMP, "gf180mcuD"): {"WP": "8u", "WS": "4u", "Lp": "0.5u", "LS": "0.3u"},
    (RAMP, "ihp-sg13g2"): {"WP": "8u", "WS": "4u", "Lp": "0.5u", "LS": "0.15u"},
    (PGA, "gf180mcuD"): {"W1": "8u", "W3": "8u", "W5": "8u", "W5b": "4u", "Lp": "0.5u"},
    (PGA, "ihp-sg13g2"): {"W1": "8u", "W3": "8u", "W5": "8u", "W5b": "4u", "Lp": "0.5u"},
    (PTAT, "gf180mcuD"): {"Wp": "8u", "Lp": "0.5u"},
    (PTAT, "ihp-sg13g2"): {"Wp": "8u", "Lp": "0.5u"},
}


# ── registry shape / ground truth ──


def test_registry_has_the_three_pilot_pdks():
    assert set(PDK_PROFILES) == {"sky130A", "gf180mcuD", "ihp-sg13g2"}


def test_sky130A_and_gf180_are_not_experimental_ihp_is():
    assert PDK_PROFILES["sky130A"].experimental is False
    assert PDK_PROFILES["gf180mcuD"].experimental is False
    assert PDK_PROFILES["ihp-sg13g2"].experimental is True


def test_verified_device_and_lib_facts():
    # spec's best-guess table, corrected/confirmed against the live image (pdks.py docstring has the
    # docker-run transcripts) — pin the values so a regression here is caught.
    gf = PDK_PROFILES["gf180mcuD"]
    assert (gf.lib_glob, gf.lib_section, gf.nfet, gf.pfet, gf.vdd_nom) == (
        "sm141064.ngspice", "typical", "nfet_03v3", "pfet_03v3", 3.3,
    )
    assert gf.extra_include_glob == "design.ngspice"     # NOT in the spec's original guess
    # E1b §4-I2: gf180 migrated wholesale to u-suffixed PDK_SIZING_OVERRIDES (dropping the I1
    # needs_scale_1u/.option scale=1.0u mechanism entirely) — ONE sizing convention across both
    # non-sky130 profiles instead of two half-conventions (see pdks.py module docstring, the "ONE
    # CONVENTION" I2 addendum). The field no longer exists on PDKProfile.
    assert not hasattr(gf, "needs_scale_1u")

    ihp = PDK_PROFILES["ihp-sg13g2"]
    assert (ihp.lib_glob, ihp.lib_section, ihp.nfet, ihp.pfet) == (
        "cornerMOSlv.lib", "mos_tt", "sg13_lv_nmos", "sg13_lv_pmos",
    )
    assert ihp.needs_osdi is True                        # NOT in the spec's original guess


# ── get_profile ──


def test_get_profile_known_pdks_round_trip():
    for key in ("sky130A", "gf180mcuD", "ihp-sg13g2"):
        assert get_profile(key).pdk == key


def test_get_profile_legacy_sky130_alias_resolves_to_sky130A_identity():
    # pre-E1 seed recipes (seeds.py's `_SKY = {"pdk": "sky130", ...}`) set pdk_profile.pdk to the
    # bare string "sky130", not the "sky130A" registry key. Must resolve, not raise (§5-I1
    # requirement 4: default/legacy behavior stays byte-identical).
    assert get_profile("sky130").pdk == "sky130A"


def test_get_profile_unknown_pdk_raises_before_any_sim():
    with pytest.raises(UnknownPDK):
        get_profile("not_a_real_pdk")


# ── substitute_devices: sky130A identity (the §6 byte-identity regression guarantee) ──


@pytest.mark.parametrize("render_fn", _PILOT_RENDERERS)
def test_sky130A_substitution_is_byte_identical(render_fn):
    deck = render_fn()
    assert substitute_devices(deck, get_profile("sky130A")) == deck


def test_sky130A_substitution_is_a_true_noop_not_just_equal():
    # spec: "sky130A profile = identity (byte-identical output)" — verify it's a genuine short-
    # circuit (same object), not a rewrite-that-happens-to-match.
    deck = render_current_mirror_dc()
    assert substitute_devices(deck, get_profile("sky130A")) is deck


# ── substitute_devices: totality on non-sky130 profiles ──


@pytest.mark.parametrize("pdk", ["gf180mcuD", "ihp-sg13g2"])
@pytest.mark.parametrize("render_fn", _PILOT_RENDERERS)
def test_non_sky130_substitution_totality(render_fn, pdk):
    """No 'sky130_fd_pr' substring survives a non-sky130 render (spec §6) — and the profile's own
    device names ARE present (substitution actually happened, not just deletion)."""
    deck = render_fn()
    profile = get_profile(pdk)
    out = substitute_devices(deck, profile)
    assert "sky130_fd_pr" not in out
    assert profile.nfet in out
    if "sky130_fd_pr__pfet_01v8" in deck:            # not every pilot template uses a pfet
        assert profile.pfet in out
    # devices/lib/VDD/section are ALL touched together — spot-check the .lib line landed correctly
    assert f".lib \"__LIBPATH__\" {profile.lib_section}" in out


def test_gf180_substitution_injects_extra_include_and_no_scale_option():
    # E1b §4-I2: gf180 dropped needs_scale_1u/.option scale=1.0u wholesale (u-suffixed
    # PDK_SIZING_OVERRIDES supply the geometry instead — see module docstring "ONE CONVENTION").
    # The extra_include (design.ngspice) is still required on EVERY gf180 deck, nominal or mismatch.
    deck = render_current_mirror_dc()
    out = substitute_devices(deck, get_profile("gf180mcuD"))
    assert '.include "__EXTRA_LIBPATH__"' in out
    assert "scale" not in out
    assert out.index('.include "__EXTRA_LIBPATH__"') < out.index('.lib "__LIBPATH__"')


def test_gf180_mismatch_token_injects_sw_stat_mismatch_param():
    # E1b §4-I2 gf180 adapter: the mismatch (tt_mm) token maps to lib_section_mm ("typical" — NOT a
    # distinct section, see pdks.py module docstring) plus a deck-level sw_stat_mismatch=1 override,
    # placed after the extra_include and before the .lib call (design.ngspice's default is 0).
    deck = f'.lib "__LIBPATH__" {MM_CORNER_TOKEN}\n.end\n'
    out = substitute_devices(deck, get_profile("gf180mcuD"))
    assert ".param sw_stat_mismatch=1" in out
    assert '.lib "__LIBPATH__" typical' in out
    assert (out.index('.include "__EXTRA_LIBPATH__"')
            < out.index(".param sw_stat_mismatch=1")
            < out.index('.lib "__LIBPATH__" typical'))


def test_ihp_mismatch_token_injects_mm_ok_on_every_mos_instance_line():
    # E1b §4-I2 ihp adapter: mm_ok=1 must land on EVERY X-card instantiating nfet/pfet — verified
    # against a real rendered deck's exact instance shape before this test/regex were written (see
    # pdks.py module docstring I2 addendum).
    deck = (
        f'.lib "__LIBPATH__" {MM_CORNER_TOKEN}\n'
        ".param VDD=1.5 W1=8u\n"
        "XM1 o1 vinp tail 0 sky130_fd_pr__nfet_01v8 W={W1} L={Lp}\n"
        "XM3 o1 o1 vdd vdd sky130_fd_pr__pfet_01v8 W={W3} L={Lp}\n"
        ".end\n"
    )
    out = substitute_devices(deck, get_profile("ihp-sg13g2"))
    assert '.lib "__LIBPATH__" mos_tt_mismatch' in out
    assert "sg13_lv_nmos W={W1} L={Lp} mm_ok=1" in out
    assert "sg13_lv_pmos W={W3} L={Lp} mm_ok=1" in out


def test_mm_ok_injection_does_not_touch_a_nominal_ihp_deck():
    # the instance-param injection is gated on the mismatch TOKEN, not unconditional — a nominal
    # ("tt") ihp deck must render exactly as before (no mm_ok anywhere).
    deck = 'XM1 o1 vinp tail 0 sky130_fd_pr__nfet_01v8 W={W1} L={Lp}\n.lib "__LIBPATH__" tt\n.end\n'
    out = substitute_devices(deck, get_profile("ihp-sg13g2"))
    assert "mm_ok" not in out


def test_ihp_substitution_has_no_scale_option_and_no_extra_include():
    # verified live: .option scale=1.0u does NOT fix ihp's OSDI/PSP103 bare-number geometry (unlike
    # gf180) — I1 does not inject a broken fix; that gap is I2's sizing-override job.
    deck = render_current_mirror_dc()
    out = substitute_devices(deck, get_profile("ihp-sg13g2"))
    assert "scale" not in out
    assert "__EXTRA_LIBPATH__" not in out
    assert '.lib "__LIBPATH__" mos_tt' in out


def test_vdd_default_literal_is_rewritten_to_profile_nominal():
    deck = render_current_mirror_dc()          # DEFAULT_CM_SIZING VDD == "1.8"
    assert "VDD=1.8" in deck
    out = substitute_devices(deck, get_profile("gf180mcuD"))
    assert "VDD=3.3" in out
    assert "VDD=1.8" not in out


def test_vdd_explicit_recipe_override_is_left_untouched():
    # a recipe-level VDD override renders a DIFFERENT numeric literal than the template default;
    # substitute_devices only rewrites the recognized DEFAULT (sizing correctness is I2's job).
    deck = render_current_mirror_dc(sizing={"VDD": "1.62"})
    out = substitute_devices(deck, get_profile("gf180mcuD"))
    assert "VDD=1.62" in out
    assert "VDD=3.3" not in out


# ── PDK_SIZING_OVERRIDES scaffold (§5-I1 requirement 5; POPULATED in I2 §5-I2 PART A — the scaffold
#    was deliberately empty at I1, this is the expected fill, not a regression) ──


def test_pdk_sizing_overrides_scaffold_is_populated_for_the_3_i2_pilot_ports():
    # E1b §4-I2: gf180mcuD migrated wholesale to u-suffixed overrides (dropping needs_scale_1u — see
    # pdks.py module docstring "ONE CONVENTION"), so BOTH non-sky130 profiles now carry an entry per
    # ported topology_class: the 3 original E1 nominal pilots (CM/CS/OTA5T) plus the E1b comparator
    # port (CMP, new for both PDKs — comparator_fpn_mc), plus I1a's 6 wave-A ports (OTA/CG/SF/DP/
    # CASC/RGC, docs/superpowers/specs/2026-07-05-full-registry-cross-pdk-rollout.md §3). Every
    # geometry value is an explicit u-suffixed micron literal on both profiles (one sizing convention,
    # not two) — EXCEPT `VBA` (regulated_cascode_nmos's aux-amp bias VOLTAGE, first introduced by
    # I1a), which must NEVER be u-suffixed (a suffix there would parse as microvolts, not volts).
    # I1b wave B added 10 pairs (5 of its 6 classes x both pdks); single_slope_ramp_generator joined
    # 2026-07-05 post-I1b when the hardcoded-literal template gap was closed (`LS` param) — all 6
    # wave-B classes are now in the union (12 pairs).
    assert set(PDK_SIZING_OVERRIDES) == {
        (CM, "ihp-sg13g2"), (CS, "ihp-sg13g2"), (OTA5T, "ihp-sg13g2"), (CMP, "ihp-sg13g2"),
        (CM, "gf180mcuD"), (CS, "gf180mcuD"), (OTA5T, "gf180mcuD"), (CMP, "gf180mcuD"),
    } | {(tc, pdk) for tc in _WAVE_A_CLASSES for pdk in ("gf180mcuD", "ihp-sg13g2")} | {
        (tc, pdk) for tc in _WAVE_B_CLASSES for pdk in ("gf180mcuD", "ihp-sg13g2")
    }
    for key, overrides in PDK_SIZING_OVERRIDES.items():
        assert overrides, f"{key} is registered but empty"
        for param, value in overrides.items():
            if param == "VBA":
                assert not value.endswith("u"), (
                    f"{key} VBA={value!r} must NOT be u-suffixed — it is a bias VOLTAGE "
                    "(regulated_cascode_nmos's aux-amp gate reference), not a device length"
                )
            else:
                assert value.endswith("u"), (
                    f"{key} override {param}={value!r} must be an explicit u-suffixed micron literal "
                    "(both non-sky130 profiles now share the ONE u-suffixed convention — pdks.py ground truth)"
                )


def test_sizing_overrides_for_gf180_matches_ihp_magnitudes_just_suffixed():
    # gf180's I2 overrides are the SAME numeric magnitude as the template defaults (and, not
    # coincidentally, identical to ihp's own overrides for these 3 classes) — the migration is a
    # unit-representation change, not a resizing (verified live; see pdks.py module docstring).
    assert sizing_overrides_for(CM, "gf180mcuD") == {"W": "4u", "Lp": "0.5u"}
    assert sizing_overrides_for(CS, "gf180mcuD") == {"Wn": "8u", "Wp": "16u", "Lp": "0.5u"}
    assert sizing_overrides_for(OTA5T, "gf180mcuD") == {
        "W1": "8u", "W3": "8u", "W5": "8u", "W5b": "4u", "Lp": "0.5u",
    }
    assert sizing_overrides_for(CMP, "gf180mcuD") == {
        "W1": "8u", "W3": "8u", "W5": "8u", "W5b": "4u", "Lp": "0.5u",
    }


def test_sizing_overrides_for_sky130A_is_empty_for_all_3_pilots():
    for tclass in (CM, CS, OTA5T):
        assert sizing_overrides_for(tclass, "sky130A") == {}


def test_sizing_overrides_for_ihp_current_mirror():
    assert sizing_overrides_for(CM, "ihp-sg13g2") == {"W": "4u", "Lp": "0.5u"}


def test_sizing_overrides_for_ihp_common_source():
    assert sizing_overrides_for(CS, "ihp-sg13g2") == {"Wn": "8u", "Wp": "16u", "Lp": "0.5u"}


def test_sizing_overrides_for_ihp_ota5t():
    assert sizing_overrides_for(OTA5T, "ihp-sg13g2") == {
        "W1": "8u", "W3": "8u", "W5": "8u", "W5b": "4u", "Lp": "0.5u",
    }


def test_sizing_overrides_for_ihp_comparator():
    # E1b §4-I2 PART B: comparator_continuous_nmos is a NEW port (comparator_fpn_mc pilot) — same
    # body shape as ota_5t_nmos_in, same DEFAULT_CMP_SIZING magnitudes, probe-validated with the
    # identical suffixed values.
    assert sizing_overrides_for(CMP, "ihp-sg13g2") == {
        "W1": "8u", "W3": "8u", "W5": "8u", "W5b": "4u", "Lp": "0.5u",
    }


# ── runner: render_run_sh (mocked / pure-text — no docker) ──


def test_render_run_sh_sky130A_returns_the_byte_identical_validated_script():
    assert render_run_sh(get_profile("sky130A")) is RUN_SH


def test_render_run_sh_gf180_parameterizes_lib_glob_and_extra_include():
    script = render_run_sh(get_profile("gf180mcuD"))
    assert "sm141064.ngspice" in script
    assert "design.ngspice" in script
    assert "__EXTRA_LIBPATH__" in script
    assert "RUNNER_FAIL" in script                  # the fail-loud guards are preserved


def test_render_run_sh_ihp_writes_a_local_spiceinit_before_invoking_ngspice():
    script = render_run_sh(get_profile("ihp-sg13g2"))
    assert "cornerMOSlv.lib" in script
    assert ".spiceinit" in script
    assert "osdi" in script
    assert 'cd "$(dirname "$OUT")"' in script        # local .spiceinit only auto-sources from cwd


# ── runner: NgspiceRunner lib-glob resolution + per-PDK workdir (mocked subprocess) ──


class _FakeCompleted:
    def __init__(self, stdout=""):
        self.stdout = stdout
        self.stderr = ""


def test_run_deck_writes_profile_lib_glob_into_a_per_pdk_workdir(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, capture_output, text, timeout):
        workdir = cmd[cmd.index("-v") + 1].split(":")[0]
        with open(os.path.join(workdir, "run.sh")) as f:
            captured["run_sh"] = f.read()
        captured["workdir"] = workdir
        return _FakeCompleted(stdout="RDATA vout 1.0 1.0e-05\n")

    monkeypatch.setattr("openclaw_brain.knowledge.executable.runner.subprocess.run", fake_run)
    runner = NgspiceRunner(workdir=str(tmp_path))
    series = runner.measure("* deck\n.end\n", pdk="gf180mcuD")

    assert "sm141064.ngspice" in captured["run_sh"]
    # isolated per-PDK BASE, with a further per-CALL subdir nested directly under it (the
    # runner.py:171/183 hardening pass — a bare-equal path would mean two concurrent gf180mcuD
    # runs share one deck.spice/run.sh again).
    assert os.path.dirname(captured["workdir"]) == str(tmp_path / "gf180mcuD")
    assert series == {"vout": [(1.0, 1.0e-05)]}


def test_run_deck_sky130A_gets_isolated_per_run_subdir_under_the_base_workdir(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, capture_output, text, timeout):
        workdir = cmd[cmd.index("-v") + 1].split(":")[0]
        with open(os.path.join(workdir, "run.sh")) as f:
            captured["run_sh"] = f.read()
        captured["workdir"] = workdir
        return _FakeCompleted(stdout="RDATA vout 1.0 1.0e-05\n")

    monkeypatch.setattr("openclaw_brain.knowledge.executable.runner.subprocess.run", fake_run)
    runner = NgspiceRunner(workdir=str(tmp_path))
    runner.measure("* deck\n.end\n")   # default pdk="sky130A"

    assert captured["run_sh"] == RUN_SH             # byte-identical script, not just equivalent
    # sky130A's BASE stays exactly self.workdir (byte-path-compatible, per _workdir_for's docstring),
    # but run_deck now writes into an ISOLATED subdirectory one level under it — never the base
    # itself — so two concurrent sky130A runs can no longer race on the same deck.spice/run.sh
    # (the runner.py:171/183 hardening pass; formerly this asserted `== str(tmp_path)` exactly,
    # which pinned the unscoped-write defect this test now guards against).
    assert captured["workdir"] != str(tmp_path)
    assert os.path.dirname(captured["workdir"]) == str(tmp_path)


def test_run_deck_concurrent_same_pdk_calls_never_share_a_workdir(tmp_path, monkeypatch):
    """The direct regression test for the sky130A race (EPISTEMOLOGY.md Known risk areas #3):
    two `measure()` calls for the SAME pdk (sky130A, the profile that used to have NO isolation at
    all) must land in two different directories, and the first call's deck.spice must still hold
    the FIRST call's content after the second call has run — proving the second call can never
    silently clobber (or read) the first call's still-in-flight files."""
    seen_workdirs = []
    seen_decks = []

    def fake_run(cmd, capture_output, text, timeout):
        workdir = cmd[cmd.index("-v") + 1].split(":")[0]
        seen_workdirs.append(workdir)
        with open(os.path.join(workdir, "deck.spice")) as f:
            seen_decks.append(f.read())
        return _FakeCompleted(stdout="RDATA vout 1.0 1.0e-05\n")

    monkeypatch.setattr("openclaw_brain.knowledge.executable.runner.subprocess.run", fake_run)
    runner = NgspiceRunner(workdir=str(tmp_path))
    runner.measure("* deck one\n.end\n")
    runner.measure("* deck two\n.end\n")

    assert len(seen_workdirs) == 2
    assert seen_workdirs[0] != seen_workdirs[1]                    # never the same directory
    assert seen_decks == ["* deck one\n.end\n", "* deck two\n.end\n"]  # each call saw its OWN deck
    # both calls' artifacts are still on disk afterward (no new cleanup/retention behavior) —
    # the first call's deck.spice was never overwritten by the second.
    with open(os.path.join(seen_workdirs[0], "deck.spice")) as f:
        assert f.read() == "* deck one\n.end\n"


def test_run_deck_unknown_pdk_raises_before_touching_the_filesystem(tmp_path):
    runner = NgspiceRunner(workdir=str(tmp_path))
    with pytest.raises(UnknownPDK):
        runner.run_deck("* deck\n.end\n", pdk="not_a_real_pdk")
    assert os.listdir(tmp_path) == []               # nothing written


# ── executor plumb-through (fake runners — no real ngspice) ──


def _cm_claim():
    return ClaimCard(
        id="cm_iout", topology_class=CM,
        mechanism=MechanismClaim(knob="Vout", metric="iout_a", series_ref="cm_iout",
                                 quant=QuantTest(kind="direction", sign="+"),
                                 narrative="iout rises with Vout"),
        conditions=AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8),
    )


def _cm_recipe(pdk_profile: dict | None = None) -> VerificationRecipe:
    cond = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8, pdk_profile=pdk_profile or {})
    return VerificationRecipe(
        topology_class=CM,
        build={"method": "template", "template_ref": "current_mirror_dc"},
        conditions=cond,
        sweeps=[{"analysis": "dc", "knob": "Vout", "points": ["0.4", "0.7", "1.0", "1.3", "1.6"],
                 "measure": ["iout_a"]}],
        claim_cards=[_cm_claim()],
    )


class _FakeRunnerLegacy:
    """Mirrors every pre-E1 test double: measure(self, deck, timeout=300), NO pdk parameter."""

    def __init__(self):
        self.decks: list[str] = []

    def measure(self, deck: str, timeout: int = 300):
        self.decks.append(deck)
        return {"vout": [(0.4, 9.6e-6), (0.7, 9.9e-6), (1.0, 1.0e-5), (1.3, 1.02e-5), (1.6, 1.04e-5)]}


class _FakePdkAwareRunner:
    """A pdk-aware double, for recipes that select a non-default profile."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []       # (pdk, deck)

    def measure(self, deck: str, pdk: str = "sky130A", timeout: int = 300):
        self.calls.append((pdk, deck))
        return {"vout": [(0.4, 9.6e-6), (0.7, 9.9e-6), (1.0, 1.0e-5), (1.3, 1.02e-5), (1.6, 1.04e-5)]}


def test_default_no_pdk_specified_is_byte_identical_to_pre_e1(tmp_path):
    """§5-I1 requirement 4: default behavior (no pdk specified) is unaffected. Uses the SAME
    legacy fake-runner shape every pre-E1 executable test double already has (no pdk kwarg) — if
    the executor ever started requiring pdk= unconditionally, this would TypeError."""
    runner = _FakeRunnerLegacy()
    result = run_recipe(_cm_recipe(), runner)
    assert result.specimen.pdk == "sky130A"
    assert "sky130_fd_pr__nfet_01v8" in result.specimen.netlist
    assert result.claim_cards[0].verdict is not None


def test_legacy_sky130_scope_alias_also_stays_on_the_no_pdk_kwarg_path(tmp_path):
    # seeds.py's pre-E1 convention: pdk_profile={"pdk": "sky130", "node": "130nm"} (scope-only,
    # informational). Must resolve to the sky130A identity profile and NOT require a pdk-aware
    # runner (the alias check happens on the RESOLVED profile, not the raw string).
    runner = _FakeRunnerLegacy()
    result = run_recipe(_cm_recipe(pdk_profile={"pdk": "sky130", "node": "130nm"}), runner)
    assert result.specimen.pdk == "sky130A"
    assert result.claim_cards[0].scope["pdk"] == "sky130"  # scope display is untouched (pre-existing field)


def test_pdk_profile_pdk_selects_the_profile_and_stamps_specimen(tmp_path):
    runner = _FakePdkAwareRunner()
    result = run_recipe(_cm_recipe(pdk_profile={"pdk": "gf180mcuD"}), runner)

    assert result.specimen.pdk == "gf180mcuD"
    assert "nfet_03v3" in result.specimen.netlist
    assert "sky130_fd_pr" not in result.specimen.netlist
    # the runner was actually invoked with the resolved pdk (proves the plumb-through, not just the
    # Specimen stamp)
    assert all(pdk == "gf180mcuD" for pdk, _deck in runner.calls)
    assert len(runner.calls) == 1                         # one deck for the one (knob, metric) pair
    assert "nfet_03v3" in runner.calls[0][1]               # the deck ngspice actually saw was substituted


def test_unknown_pdk_raises_before_any_render_or_sim(tmp_path):
    class _ExplodingRunner:
        def measure(self, deck, pdk="sky130A", timeout=300):
            raise AssertionError("runner.measure must never be called for an unknown pdk")

    with pytest.raises(UnknownPDK):
        run_recipe(_cm_recipe(pdk_profile={"pdk": "not_a_real_pdk"}), _ExplodingRunner())


# ── docker-gated live smoke: one per new PDK (spec §6) ──


def test_gf180_live_smoke_current_mirror_dc():
    """Deck rendering + lib resolution (incl. the required design.ngspice extra include) + ngspice
    invocation, for real, using the E1b §4-I2 u-suffixed PDK_SIZING_OVERRIDES (gf180 dropped
    needs_scale_1u/.option scale=1.0u wholesale — see pdks.py module docstring "ONE CONVENTION").
    Verified live: this profile actually converges to a physically sane current-mirror curve
    (~10uA), not just "doesn't crash" — the SAME magnitude as the pre-migration bare+scale probe."""
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; smoke test needs the sim container")

    profile = get_profile("gf180mcuD")
    sizing = sizing_overrides_for(CM, "gf180mcuD")
    assert sizing == {"W": "4u", "Lp": "0.5u"}
    deck = substitute_devices(render_current_mirror_dc(sizing=sizing), profile)
    series = runner.measure(deck, pdk="gf180mcuD")

    assert "vout" in series and len(series["vout"]) == 5
    for _x, iout in series["vout"]:
        assert 1e-6 < iout < 1e-4, f"iout {iout} far outside a sane current-mirror range"


def test_gf180_live_smoke_common_source_ac():
    """I2 port re-confirmation post-migration: common_source_active_load_nmos with the u-suffixed
    override (probed 2026-07-04) is sane: av0 positive gain ~40dB, GBW well below the AC sweep's
    10GHz Nyquist — the SAME magnitudes the pre-migration bare+scale convention produced."""
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; smoke test needs the sim container")

    profile = get_profile("gf180mcuD")
    sizing = sizing_overrides_for(CS, "gf180mcuD")
    assert sizing == {"Wn": "8u", "Wp": "16u", "Lp": "0.5u"}
    deck_av0 = substitute_devices(render_common_source_ac(sizing=sizing, metric="av0_db"), profile)
    av0 = runner.measure(deck_av0, pdk="gf180mcuD")["iref"]
    for _x, db in av0:
        assert 10.0 < db < 60.0, f"av0 {db} dB outside a sane single-stage gain range"

    deck_gbw = substitute_devices(render_common_source_ac(sizing=sizing, metric="gbw_hz"), profile)
    gbw = runner.measure(deck_gbw, pdk="gf180mcuD")["iref"]
    for _x, hz in gbw:
        assert 1e5 < hz < 1e9, f"gbw {hz} Hz outside a sane range (well below the 10GHz AC Nyquist)"


def test_gf180_live_smoke_ota5t_ac():
    """I2 port re-confirmation post-migration: ota_5t_nmos_in with the u-suffixed override — av0
    flat positive gain, GBW falling with CL (the 1/CL law), both well below the AC sweep's Nyquist."""
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; smoke test needs the sim container")

    profile = get_profile("gf180mcuD")
    sizing = sizing_overrides_for(OTA5T, "gf180mcuD")
    assert sizing == {"W1": "8u", "W3": "8u", "W5": "8u", "W5b": "4u", "Lp": "0.5u"}
    deck_av0 = substitute_devices(render_ota_5t_ac(sizing=sizing, metric="av0_db"), profile)
    av0 = runner.measure(deck_av0, pdk="gf180mcuD")["cl"]
    for _x, db in av0:
        assert 10.0 < db < 60.0, f"av0 {db} dB outside a sane single-stage gain range"

    deck_gbw = substitute_devices(render_ota_5t_ac(sizing=sizing, metric="gbw_hz"), profile)
    gbw = runner.measure(deck_gbw, pdk="gf180mcuD")["cl"]
    ys = [hz for _x, hz in gbw]
    for hz in ys:
        assert 1e5 < hz < 1e9, f"gbw {hz} Hz outside a sane range (well below the 10GHz AC Nyquist)"
    assert ys == sorted(ys, reverse=True), "GBW must fall monotonically as CL rises (1/CL law)"


def test_ihp_live_smoke_current_mirror_dc_with_sizing_override_is_now_sane():
    """I2 port confirmation: with the explicit u-suffixed PDK_SIZING_OVERRIDES entry (W=4u Lp=0.5u),
    ihp-sg13g2's current mirror converges to a SANE, monotonically-rising current (~10-18uA vs a
    10uA IREF) — the fix the mechanics-only smoke above documents as still-needed absent an override."""
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; smoke test needs the sim container")

    profile = get_profile("ihp-sg13g2")
    sizing = sizing_overrides_for(CM, "ihp-sg13g2")
    assert sizing == {"W": "4u", "Lp": "0.5u"}
    deck = substitute_devices(render_current_mirror_dc(sizing=sizing), profile)
    series = runner.measure(deck, pdk="ihp-sg13g2")

    assert "vout" in series and len(series["vout"]) == 5
    ys = [iout for _x, iout in series["vout"]]
    for iout in ys:
        assert 1e-6 < iout < 1e-4, f"iout {iout} far outside a sane current-mirror range"
    assert ys == sorted(ys), "iout must rise monotonically with Vout (finite output resistance)"


def test_ihp_live_smoke_common_source_ac_with_sizing_override_is_now_sane():
    """I2 port confirmation: with the u-suffixed override, ihp's common-source stage converges FAST
    (bare/unsuffixed sizing did not converge inside a 300s ngspice budget in probing — 2 independent
    timeouts) to a sane positive gain."""
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; smoke test needs the sim container")

    profile = get_profile("ihp-sg13g2")
    sizing = sizing_overrides_for(CS, "ihp-sg13g2")
    assert sizing == {"Wn": "8u", "Wp": "16u", "Lp": "0.5u"}

    deck_av0 = substitute_devices(render_common_source_ac(sizing=sizing, metric="av0_db"), profile)
    av0 = runner.measure(deck_av0, pdk="ihp-sg13g2")["iref"]
    for _x, db in av0:
        assert 5.0 < db < 60.0, f"av0 {db} dB outside a sane single-stage gain range"

    deck_gbw = substitute_devices(render_common_source_ac(sizing=sizing, metric="gbw_hz"), profile)
    gbw = runner.measure(deck_gbw, pdk="ihp-sg13g2")["iref"]
    for _x, hz in gbw:
        assert 1e5 < hz < 1e9, f"gbw {hz} Hz outside a sane range (well below the 10GHz AC Nyquist)"


def test_ihp_live_smoke_ota5t_ac_with_sizing_override_is_now_sane():
    """I2 port confirmation: with the u-suffixed override, ihp's 5T OTA converges to a POSITIVE gain
    (bare sizing gave a degenerate -88.6dB point in probing — a mis-biased operating point, not just
    a wrong number) and a GBW falling with CL (the 1/CL law)."""
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; smoke test needs the sim container")

    profile = get_profile("ihp-sg13g2")
    sizing = sizing_overrides_for(OTA5T, "ihp-sg13g2")
    assert sizing == {"W1": "8u", "W3": "8u", "W5": "8u", "W5b": "4u", "Lp": "0.5u"}

    deck_av0 = substitute_devices(render_ota_5t_ac(sizing=sizing, metric="av0_db"), profile)
    av0 = runner.measure(deck_av0, pdk="ihp-sg13g2")["cl"]
    for _x, db in av0:
        assert 5.0 < db < 60.0, f"av0 {db} dB outside a sane single-stage gain range"

    deck_gbw = substitute_devices(render_ota_5t_ac(sizing=sizing, metric="gbw_hz"), profile)
    gbw = runner.measure(deck_gbw, pdk="ihp-sg13g2")["cl"]
    ys = [hz for _x, hz in gbw]
    for hz in ys:
        assert 1e5 < hz < 1e9, f"gbw {hz} Hz outside a sane range (well below the 10GHz AC Nyquist)"
    assert ys == sorted(ys, reverse=True), "GBW must fall monotonically as CL rises (1/CL law)"


def test_ihp_live_smoke_current_mirror_dc_mechanics_only():
    """Deck rendering + lib resolution (incl. the OSDI local-.spiceinit mechanism) + ngspice
    invocation succeed for real — the §6 bar for an EXPERIMENTAL profile. The magnitude is NOT
    asserted to be physically sane: ihp's bare (unsuffixed) W/L geometry is a known, documented gap
    (PDKProfile.notes) that only an explicitly u-suffixed PDK_SIZING_OVERRIDES entry (I2) fixes —
    this test does not paper over that; it only checks the mechanics I1 owns."""
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; smoke test needs the sim container")

    profile = get_profile("ihp-sg13g2")
    assert profile.experimental
    deck = substitute_devices(render_current_mirror_dc(), profile)
    series = runner.measure(deck, pdk="ihp-sg13g2")

    # mechanics succeeded: a RUNNER_FAIL or fatal ngspice parse error would have raised in
    # runner.measure (RuntimeError) or left `series` empty — neither happened.
    assert "vout" in series and len(series["vout"]) == 5


def test_bjt_template_available_on_gf180_and_ihp_as_of_i1b():
    """I1b (spec docs/superpowers/specs/2026-07-05-full-registry-cross-pdk-rollout.md §3): as of this
    increment BOTH gf180mcuD and ihp-sg13g2 registered a usable `bjt_pnp` (see pdks.py's "BJT
    AVAILABILITY" ground truth) — substitute_devices no longer raises BjtUnavailable for either, and
    substitutes the sky130 PNP literal like any other device. sky130A stays the byte-identical no-op."""
    from openclaw_brain.knowledge.executable.pdks import get_profile, substitute_devices

    deck = "xq1 c b e sky130_fd_pr__pnp_05v5_W0p68L0p68\n.lib \"__LIBPATH__\" tt\n.end\n"
    assert substitute_devices(deck, get_profile("sky130A")) is deck
    for pdk, expected_pnp in (("gf180mcuD", "pnp_05p00x00p42"), ("ihp-sg13g2", "pnpMPA")):
        out = substitute_devices(deck, get_profile(pdk))
        assert "sky130_fd_pr__pnp_05v5" not in out
        assert expected_pnp in out


def test_bjt_unavailable_guard_still_raises_for_a_profile_without_bjt_pnp():
    """Defense-in-depth (BjtUnavailable's updated docstring): a profile that does NOT register a
    `bjt_pnp` still gets a clear, early refusal rather than a hybrid deck referencing an undefined
    subckt — verified against a synthetic profile so this guard is exercised even though neither
    currently-registered non-sky130 profile takes this path any more."""
    import dataclasses

    from openclaw_brain.knowledge.executable.pdks import (
        BjtUnavailable, PDK_PROFILES, substitute_devices,
    )

    deck = "xq1 c b e sky130_fd_pr__pnp_05v5_W0p68L0p68\n.lib \"__LIBPATH__\" tt\n.end\n"
    no_bjt_profile = dataclasses.replace(PDK_PROFILES["gf180mcuD"], bjt_pnp=None)
    with pytest.raises(BjtUnavailable):
        substitute_devices(deck, no_bjt_profile)


def test_bjt_totality_guard_raises_for_a_non_w0p68l0p68_sky130_pnp_literal():
    """Verifier finding (E-track ③ I1b must-fix): the `is_bjt_deck` availability gate is
    PREFIX-based (`_PNP_SKY130_PREFIX in deck_text`), but the actual rewrite only ever targeted the
    ONE exact literal `_PNP_SKY130_FULL` (W0p68L0p68). A deck instantiating any OTHER sky130
    pnp_05v5 size — e.g. W3p40L3p40, a real sky130 pnp size (a two-size pnp design is recorded in
    seeds.py's own S3-inc2a history) — used to sail past the availability gate, survive the
    exact-literal replace untouched, and ship sky130 residue silently on both gf180mcuD and
    ihp-sg13g2. `substitute_devices` must now raise `BjtUnavailable` instead, on BOTH profiles,
    rather than returning a hybrid deck that still references an undefined sky130 subckt."""
    from openclaw_brain.knowledge.executable.pdks import (
        BjtUnavailable, get_profile, substitute_devices,
    )

    deck = "xq1 c b e sky130_fd_pr__pnp_05v5_W3p40L3p40\n.lib \"__LIBPATH__\" tt\n.end\n"
    for pdk in ("gf180mcuD", "ihp-sg13g2"):
        with pytest.raises(BjtUnavailable):
            substitute_devices(deck, get_profile(pdk))


# ===================================================================================================
# I1a — wave A port (docs/superpowers/specs/2026-07-05-full-registry-cross-pdk-rollout.md §3): the 6
# "simpler bias structure" templates x {gf180mcuD, ihp-sg13g2}. Unit (render/substitution) coverage
# mirrors the E1 pilot's exact pattern above (totality, sky130A byte-identity, sizing-overrides shape)
# so it stays additive rather than a parallel convention; docker-gated live smokes confirm the actual
# wired PDK_SIZING_OVERRIDES entries reproduce the probe transcript (pdks.py module docstring /
# PDK_SIZING_OVERRIDES comment) end to end through the real executor + oracle.
# ===================================================================================================


# ── substitution: sky130A byte-identity + totality (same guarantee as the E1 pilot, additive) ──


@pytest.mark.parametrize("render_fn", _WAVE_A_RENDERERS)
def test_wave_a_sky130A_substitution_is_byte_identical(render_fn):
    deck = render_fn()
    assert substitute_devices(deck, get_profile("sky130A")) == deck


@pytest.mark.parametrize("pdk", ["gf180mcuD", "ihp-sg13g2"])
@pytest.mark.parametrize("render_fn", _WAVE_A_RENDERERS)
def test_wave_a_non_sky130_substitution_totality(render_fn, pdk):
    """No 'sky130_fd_pr' substring survives a non-sky130 render of any of the 6 wave-A templates —
    and the profile's own device name IS present (substitution actually happened)."""
    deck = render_fn()
    profile = get_profile(pdk)
    out = substitute_devices(deck, profile)
    assert "sky130_fd_pr" not in out
    assert profile.nfet in out
    if "sky130_fd_pr__pfet_01v8" in deck:            # not every wave-A template uses a pfet
        assert profile.pfet in out
    assert f'.lib "__LIBPATH__" {profile.lib_section}' in out


# ── PDK_SIZING_OVERRIDES: per-port shape (mirrors the E1 pilot's per-class tests) ──


@pytest.mark.parametrize("pdk", ["gf180mcuD", "ihp-sg13g2"])
@pytest.mark.parametrize("tclass", [OTA, CG, SF, DP, CASC])
def test_sizing_overrides_for_wave_a_ports_share_the_same_magnitude_both_pdks(tclass, pdk):
    # every wave-A class except regulated_cascode_nmos needed ONLY a geometry override, and that
    # override is the identical u-suffixed magnitude on BOTH non-sky130 PDKs (I1a probe finding —
    # pdks.py's PDK_SIZING_OVERRIDES comment has the transcript).
    assert sizing_overrides_for(tclass, pdk) == _WAVE_A_EXPECTED_OVERRIDES[tclass]


@pytest.mark.parametrize("pdk", ["gf180mcuD", "ihp-sg13g2"])
def test_sizing_overrides_for_regulated_cascode_carries_a_pdk_specific_vba(pdk):
    # regulated_cascode_nmos's aux-amp bias VOLTAGE genuinely differs per PDK's rail (probed: gf180's
    # 3.3V needs VBA=1.8 to reproduce sky130-grade regulation; ihp's 1.5V needs VBA=0.55, its scan
    # optimum — unlike every OTHER wave-A override, which is identical across both non-sky130 PDKs).
    assert sizing_overrides_for(RGC, pdk) == _RGC_EXPECTED_OVERRIDES[pdk]


# ── render-level regression per port: the WIRED override actually reaches the rendered deck ──


@pytest.mark.parametrize("pdk", ["gf180mcuD", "ihp-sg13g2"])
@pytest.mark.parametrize("tclass", _WAVE_A_CLASSES)
def test_wave_a_port_render_regression(tclass, pdk):
    """Per-(template, pdk) render regression (I1a): sizing_overrides_for's registered entry — the
    SAME lookup run_recipe() uses, not a hand override — renders into the deck's `.param` line
    verbatim, and the substituted deck carries no sky130 residue and the profile's own nfet."""
    profile = get_profile(pdk)
    render_fn = _WAVE_A_RENDER_FNS[tclass]
    sizing = sizing_overrides_for(tclass, pdk)
    assert sizing, f"{tclass}/{pdk} must have a registered override (I1a ported every wave-A class)"
    deck = render_fn(sizing=sizing)
    for param, value in sizing.items():
        assert f"{param}={value}" in deck, f"{tclass}/{pdk}: override {param}={value!r} did not reach the .param line"
    out = substitute_devices(deck, profile)
    assert "sky130_fd_pr" not in out
    assert profile.nfet in out


# ── docker-gated live smoke: the actual wired path, end to end through run_recipe()/oracle ──


def _wave_a_recipe(tclass: str, pdk: str) -> VerificationRecipe:
    recipe = next(r for r in seed_recipes() if r.topology_class == tclass)
    rec = copy.deepcopy(recipe)
    rec.conditions.pdk_profile = {"pdk": pdk}
    return rec


@pytest.mark.parametrize("pdk", ["gf180mcuD", "ihp-sg13g2"])
@pytest.mark.parametrize("tclass", [OTA, CG, SF, DP, CASC])
def test_wave_a_live_smoke_all_claims_verified(tclass, pdk):
    """I1a docker-gated live smoke (probe step b+d combined): every seed-recipe claim for this
    (template, pdk) VERIFIES on real IIC-OSIC-TOOLS ngspice, through the exact production path
    (run_recipe -> sizing_overrides_for -> substitute_devices -> NgspiceRunner -> oracle) — not a
    hand-rendered probe deck. 5 of 6 wave-A classes replicate VERIFIED on both new PDKs (agreeing
    with the sky130A baseline); regulated_cascode_nmos is the one exception, covered separately below."""
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; smoke test needs the sim container")

    result = run_recipe(_wave_a_recipe(tclass, pdk), runner)
    assert result.specimen.pdk == pdk
    assert "sky130_fd_pr" not in result.specimen.netlist
    for card in result.claim_cards:
        assert card.verdict is not None and card.verdict.value == "VERIFIED", (
            f"{tclass}/{pdk} claim {card.id} expected VERIFIED, got "
            f"{card.verdict.value if card.verdict else None} ({card.verdict_note})"
        )


@pytest.mark.parametrize("pdk,expected_verdict", [("gf180mcuD", "VERIFIED"), ("ihp-sg13g2", "REFUTED")])
def test_wave_a_live_smoke_regulated_cascode_verdict_per_pdk(pdk, expected_verdict):
    """regulated_cascode_nmos is I1a's one process_scoped divergence (Q1, rollout spec §1): gf180mcuD's
    gain-boosting loop, with the probe-found VBA=1.8 override, reproduces sky130-grade regulation
    (VERIFIED, CoV ~0.01% against the claim's 0.1% bound). ihp-sg13g2's IDENTICAL claim REFUTES even at
    its scan-optimum VBA=0.55 (CoV floor ~0.28%, ~2.8x the bound — confirmed not sizing-fixable by an
    additional aux-device-width scan; pdks.py's PDK_SIZING_OVERRIDES comment has the full transcript).
    This regression-pins the exact verdict per PDK so a silent instrument change is caught rather than
    silently re-rationalized as "expected divergence" without re-checking the actual numbers."""
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; smoke test needs the sim container")

    result = run_recipe(_wave_a_recipe(RGC, pdk), runner)
    card = result.claim_cards[0]
    assert card.verdict is not None and card.verdict.value == expected_verdict, (
        f"rgc_iout/{pdk} expected {expected_verdict}, got "
        f"{card.verdict.value if card.verdict else None} ({card.verdict_note})"
    )


# ===================================================================================================
# I1b — wave B port (docs/superpowers/specs/2026-07-05-full-registry-cross-pdk-rollout.md §3): the 6
# "headroom/switched/BJT" templates x {gf180mcuD, ihp-sg13g2}. 9/12 ports VERIFIED; 3/12 are honest
# PORT-FAILED (telescopic_cascode_ota_nmos_in/ihp-sg13g2 — headroom, the spec's own Q3 guess;
# single_slope_ramp_generator on BOTH pdks — a template-authoring gap unrelated to any pdk). Unit
# coverage mirrors wave A's pattern (totality, sky130A byte-identity, sizing-overrides shape); the
# PORT-FAILED classes get their OWN shape/regression assertions rather than being silently skipped —
# pdks.py's ground truth (module docstring + PDK_SIZING_OVERRIDES comment) has the full transcripts.
# ===================================================================================================


# ── substitution: sky130A byte-identity + totality (same guarantee as every prior wave) ──


@pytest.mark.parametrize("render_fn", _WAVE_B_RENDERERS)
def test_wave_b_sky130A_substitution_is_byte_identical(render_fn):
    deck = render_fn()
    assert substitute_devices(deck, get_profile("sky130A")) == deck


@pytest.mark.parametrize("pdk", ["gf180mcuD", "ihp-sg13g2"])
@pytest.mark.parametrize("render_fn", _WAVE_B_RENDERERS)
def test_wave_b_non_sky130_substitution_totality(render_fn, pdk):
    """No 'sky130_fd_pr' substring survives a non-sky130 render of any of the 6 wave-B templates
    (this covers BOTH the nfet/pfet prefix AND the pnp_05v5 BJT prefix, since both share the
    'sky130_fd_pr' root) — and the profile's own device name IS present. This is a purely MECHANICAL
    (render/substitute) guarantee: it holds even for telescopic_cascode_ota_nmos_in/ihp-sg13g2, which
    is PORT-FAILED at the SIMULATION/verdict level (see the live-smoke regression test below) — a
    clean substitution is necessary but not sufficient for a working port, exactly the distinction
    this increment's ground truth insists on."""
    deck = render_fn()
    profile = get_profile(pdk)
    out = substitute_devices(deck, profile)
    assert "sky130_fd_pr" not in out
    assert profile.nfet in out or profile.pfet in out
    if "sky130_fd_pr__pfet_01v8" in deck:
        assert profile.pfet in out
    if "sky130_fd_pr__pnp_05v5" in deck:
        assert profile.bjt_pnp in out
    assert f'.lib "__LIBPATH__" {profile.lib_section}' in out


def test_ptat_ctat_core_bjt_substitution_adds_the_second_bjt_lib_line():
    """The BJT corner section is a SECOND `.lib` call (pdks.py ground truth "BJT AVAILABILITY") —
    gf180mcuD reuses the resolved main libpath token (same file, `bjt_typical` section); ihp-sg13g2
    resolves a genuinely separate file via BJT_LIB_PLACEHOLDER (`hbt_typ` section)."""
    from openclaw_brain.knowledge.executable.pdks import BJT_LIB_PLACEHOLDER

    deck = render_ptat_ctat_core_bjt()
    gf180_out = substitute_devices(deck, get_profile("gf180mcuD"))
    assert '.lib "__LIBPATH__" bjt_typical' in gf180_out
    ihp_out = substitute_devices(deck, get_profile("ihp-sg13g2"))
    assert f'.lib "{BJT_LIB_PLACEHOLDER}" hbt_typ' in ihp_out


# ── PDK_SIZING_OVERRIDES: per-port shape, including the 3 PORT-FAILED pairs ──


@pytest.mark.parametrize("pdk", ["gf180mcuD", "ihp-sg13g2"])
@pytest.mark.parametrize("tclass", _WAVE_B_CLASSES)
def test_sizing_overrides_for_wave_b_ports_match_the_wired_transcript(tclass, pdk):
    assert sizing_overrides_for(tclass, pdk) == _WAVE_B_EXPECTED_OVERRIDES[(tclass, pdk)]


def test_single_slope_ramp_generator_ls_override_reaches_the_switch():
    """I1b recorded ramp as PORT-FAILED on BOTH pdks: `_RAMP_BODY`'s reset switch hardcoded a bare
    `L=0.15` literal no override could reach. 2026-07-05 post-I1b the switch length became the `LS`
    param (templates.py) and both ports now carry the usual entry — this test pins the CLOSED state:
    the entry exists, the rendered deck routes it through `.param`, and no bare `L=0.15` remains.
    (Flipped from the original absence-pin test when the template gap was fixed; live 3-PDK
    validation: sky130 slopes numerically unchanged, gf180/ihp ramp_slope VERIFIED.)"""
    assert sizing_overrides_for(RAMP, "gf180mcuD") == {
        "WP": "8u", "WS": "4u", "Lp": "0.5u", "LS": "0.3u"}     # LS above nfet_03v3 lmin=0.28um
    assert sizing_overrides_for(RAMP, "ihp-sg13g2") == {
        "WP": "8u", "WS": "4u", "Lp": "0.5u", "LS": "0.15u"}
    for pdk in ("gf180mcuD", "ihp-sg13g2"):
        deck = render_ramp_tran(sizing=sizing_overrides_for(RAMP, pdk))
        assert "L={LS}" in deck and "L=0.15\n" not in deck
        assert f"LS={sizing_overrides_for(RAMP, pdk)['LS']}" in deck
    # sky130 default stays numerically identical to the historic literal
    assert "LS=0.15" in render_ramp_tran()


def test_telescopic_cascode_ihp_override_is_present_but_not_a_success_signal():
    """telescopic_cascode_ota_nmos_in/ihp-sg13g2 carries the SAME u-suffixed geometry override as
    gf180mcuD (bare geometry is a separate, unrelated hard failure on ihp) — but is STILL PORT-FAILED
    at the verdict level (see the live-smoke regression test below). A non-empty entry in
    PDK_SIZING_OVERRIDES is necessary-but-not-sufficient evidence of a working port; this test pins
    that the entry stays geometry-only (no VBA/VBNC/VBPC override was found to fix it — pdks.py's
    ground truth has the full bias/width scan transcript)."""
    assert sizing_overrides_for(TELE, "ihp-sg13g2") == {"WN": "8u", "WP": "16u", "WT": "16u", "Lp": "0.5u"}


# ── render-level regression per port: the WIRED override actually reaches the rendered deck ──


@pytest.mark.parametrize("pdk", ["gf180mcuD", "ihp-sg13g2"])
@pytest.mark.parametrize("tclass", [TELE, FC, CDS, PGA, PTAT])   # RAMP excluded: no override to check
def test_wave_b_port_render_regression(tclass, pdk):
    """Per-(template, pdk) render regression (I1b): sizing_overrides_for's registered entry — the
    SAME lookup run_recipe() uses — renders into the deck's `.param` line verbatim, and the
    substituted deck carries no sky130 residue and the profile's own nfet. RAMP is excluded: it has
    no registered override at all (PORT-FAILED, see above), so there is nothing to regression-check
    at the render level."""
    profile = get_profile(pdk)
    render_fn = _WAVE_B_RENDER_FNS[tclass]
    sizing = sizing_overrides_for(tclass, pdk)
    assert sizing, f"{tclass}/{pdk} must have a registered override"
    deck = render_fn(sizing=sizing)
    for param, value in sizing.items():
        assert f"{param}={value}" in deck, f"{tclass}/{pdk}: override {param}={value!r} did not reach the .param line"
    out = substitute_devices(deck, profile)
    assert "sky130_fd_pr" not in out
    assert profile.nfet in out or profile.pfet in out


# ── docker-gated live smoke: the actual wired path, end to end through run_recipe()/oracle ──


def _wave_b_recipe(tclass: str, pdk: str) -> VerificationRecipe:
    recipe = next(r for r in seed_recipes() if r.topology_class == tclass)
    rec = copy.deepcopy(recipe)
    rec.conditions.pdk_profile = {"pdk": pdk}
    return rec


_WAVE_B_VERIFIED_PAIRS = [
    (TELE, "gf180mcuD"), (FC, "gf180mcuD"), (FC, "ihp-sg13g2"),
    (CDS, "gf180mcuD"), (CDS, "ihp-sg13g2"),
    (PGA, "gf180mcuD"), (PGA, "ihp-sg13g2"),
    (PTAT, "gf180mcuD"), (PTAT, "ihp-sg13g2"),
]


@pytest.mark.parametrize("tclass,pdk", _WAVE_B_VERIFIED_PAIRS)
def test_wave_b_live_smoke_all_claims_verified(tclass, pdk):
    """I1b docker-gated live smoke: every seed-recipe claim for this (template, pdk) VERIFIES on real
    IIC-OSIC-TOOLS ngspice, through the exact production path (run_recipe -> sizing_overrides_for ->
    substitute_devices -> NgspiceRunner -> oracle) — not a hand-rendered probe deck. 9 of the 12
    wave-B ports replicate VERIFIED (agreeing with the sky130A baseline); the 2 PORT-FAILED classes
    (telescopic_cascode_ota_nmos_in/ihp-sg13g2, single_slope_ramp_generator on both pdks) are covered
    separately below."""
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; smoke test needs the sim container")

    result = run_recipe(_wave_b_recipe(tclass, pdk), runner)
    assert result.specimen.pdk == pdk
    assert "sky130_fd_pr" not in result.specimen.netlist
    for card in result.claim_cards:
        assert card.verdict is not None and card.verdict.value == "VERIFIED", (
            f"{tclass}/{pdk} claim {card.id} expected VERIFIED, got "
            f"{card.verdict.value if card.verdict else None} ({card.verdict_note})"
        )


def test_wave_b_live_smoke_telescopic_cascode_ihp_is_port_failed():
    """telescopic_cascode_ota_nmos_in/ihp-sg13g2 is I1b's one process_scoped PORT-FAILED (the rollout
    spec's own Q3 headroom guess, confirmed on the guessed template this time): tele_gbw cannot even
    find a 0dB crossing (FLAGGED, empty series) and tele_av0 is wildly non-flat (REFUTED, spread far
    past the 0.5dB bound) — NOT a tight-but-sane-OP divergence like I1a's regulated_cascode. A
    systematic VBNC/VBPC/WN scan (pdks.py's ground truth has the full transcript) never found a
    positive, CL-flat gain; this regression-pins the exact degenerate verdict shape so a silent
    instrument change is caught rather than re-rationalized as "expected" without re-checking."""
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; smoke test needs the sim container")

    result = run_recipe(_wave_b_recipe(TELE, "ihp-sg13g2"), runner)
    by_id = {c.id: c for c in result.claim_cards}
    assert by_id["tele_gbw"].verdict is not None and by_id["tele_gbw"].verdict.value == "FLAGGED"
    assert by_id["tele_av0"].verdict is not None and by_id["tele_av0"].verdict.value == "REFUTED"


@pytest.mark.parametrize("pdk", ["gf180mcuD", "ihp-sg13g2"])
def test_wave_b_live_smoke_ramp_generator_verifies_post_ls_fix(pdk):
    """I1b pinned ramp as cleanly-FLAGGED PORT-FAILED "until a future increment fixes the template
    body" — that increment landed 2026-07-05 (`LS` param, templates.py): this is the SAME smoke,
    flipped to pin the fixed state. slope=I/Cramp must now certify on both pdks (live 3-PDK
    validation at fix time: sky130 slopes numerically unchanged; gf180/ihp VERIFIED, slope within a
    few % of the ideal I/C=2e7 V/s at 10uA/500fF)."""
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; smoke test needs the sim container")

    result = run_recipe(_wave_b_recipe(RAMP, pdk), runner)
    card = result.claim_cards[0]
    assert card.verdict is not None and card.verdict.value == "VERIFIED", (
        f"ramp_slope/{pdk} expected VERIFIED post-LS-fix, got "
        f"{card.verdict.value if card.verdict else None} ({card.verdict_note})"
    )
