"""E1b §4-I1/§4-I2 tests: seeded Monte-Carlo determinism, the mismatch-token (`tt_mm`) ->
`lib_section_mm` mapping, the mismatch-unavailable guard, suffixed-geometry area scaling for the
Pelgrom loop, and (I2) the two live adapters that flip `lib_section_mm` ON for gf180mcuD/ihp-sg13g2.
See docs/superpowers/specs/2026-07-04-e1b-statistical-cross-pdk.md §0/§3/§4-I1/§4-I2/§5.

Unit (no docker): lib_section_mm registry values, substitute_devices' mm-token mapping (incl. the
I2 mm_param_injection/mm_instance_params adapters), require_mismatch_available, the executor's
early mismatch-unavailable guard genericized against a SYNTHETIC no-mismatch profile (I1's original
tests pinned gf180/ihp specifically — both are now Q1-available, so the guard's "raises before any
render/sim" BEHAVIOR is re-targeted onto a synthetic profile rather than deleted; this is
spec-intended evolution, not test-weakening — the guard code path is unchanged, only which profile
exercises it), the Pelgrom per-area seed-offset distinctness, and _scale_geometry_value (plain +
u-suffixed).

Live (docker-gated): a sky130 seeded-determinism regression pin (spec's STOP-rule requirement — two
runs of the identical small-mc_runs deck must be byte-identical) and the I2 gf180/ihp mismatch
smokes (mechanics + determinism + genuinely nonzero spread, through the FULL render->
substitute_devices->runner.measure path — the adapters this file's unit tests exercise mechanically).
"""
from __future__ import annotations

import dataclasses
import re
import statistics

import pytest

from openclaw_brain.knowledge.executable.executor import _scale_geometry_value, run_recipe
from openclaw_brain.knowledge.executable.mc_templates import (
    MC_SEED_AREA_STRIDE,
    MC_SEED_BASE,
    render_ota5t_offset_mc,
)
from openclaw_brain.knowledge.executable.models import (
    AnalogPVT, ClaimCard, MechanismClaim, QuantTest, VerificationRecipe,
)
from openclaw_brain.knowledge.executable import pdks
from openclaw_brain.knowledge.executable.pdks import (
    MM_CORNER_TOKEN,
    MismatchUnavailable,
    PDK_PROFILES,
    get_profile,
    require_mismatch_available,
    sizing_overrides_for,
    substitute_devices,
)

OTA5T = "ota_5t_nmos_in"


def _no_mm_profile(base_pdk: str = "sky130A", key: str = "synthetic-no-mm"):
    """A synthetic PDKProfile with lib_section_mm=None, for testing the mismatch-unavailable GUARD
    mechanism itself (spec-intended after both real non-sky130 profiles flipped lib_section_mm ON
    in I2 — see module docstring)."""
    return dataclasses.replace(get_profile(base_pdk), pdk=key, lib_section_mm=None)


# ── lib_section_mm registry values (spec §3) ──


def test_lib_section_mm_registry_values():
    # E1b §4-I2: BOTH new PDKs are FLIPPED ON — the I2 adapters (gf180's sw_stat_mismatch=1 deck-
    # level .param injection; ihp's mm_ok=1 per-MOS-instance injection) were live-verified against
    # a hand-deck control+treatment pair AND through the full render->substitute_devices->
    # runner.measure code path (see pdks.py module docstring's I2 addendum). gf180's mm section is
    # the SAME as its nominal one ("typical" — fets_mm is pulled into every corner already; only
    # sw_stat_mismatch differentiates a mismatch deck), modeled honestly rather than inventing a
    # section that doesn't exist.
    assert PDK_PROFILES["sky130A"].lib_section_mm == "tt_mm"
    assert PDK_PROFILES["gf180mcuD"].lib_section_mm == "typical"
    assert PDK_PROFILES["ihp-sg13g2"].lib_section_mm == "mos_tt_mismatch"


# ── substitute_devices: mm-token mapping (spec §3/§4-I1 requirement 2, §4-I2 adapters) ──


def test_substitute_devices_ihp_mm_token_now_works_with_mm_ok_injection():
    # E1b §4-I2: ihp's mm_ok=1 per-instance adapter landed — the mismatch token no longer raises;
    # it maps to mos_tt_mismatch AND appends mm_ok=1 to every MOS instance line.
    deck = (f'.lib "__LIBPATH__" {MM_CORNER_TOKEN}\n'
            "XM1 o1 vinp tail 0 sky130_fd_pr__nfet_01v8 W={W1} L={Lp}\n.end\n")
    out = substitute_devices(deck, get_profile("ihp-sg13g2"))
    assert '.lib "__LIBPATH__" mos_tt_mismatch' in out
    assert "sg13_lv_nmos W={W1} L={Lp} mm_ok=1" in out


def test_substitute_devices_sky130A_mm_token_is_untouched_identity():
    # sky130A is the identity profile — a tt_mm deck passes through byte-identical (the section
    # already IS "tt_mm" in sky130's own lib file; no mapping needed).
    deck = f'.lib "__LIBPATH__" {MM_CORNER_TOKEN}\n.end\n'
    assert substitute_devices(deck, get_profile("sky130A")) is deck


def test_substitute_devices_gf180_mm_token_now_works_with_sw_stat_mismatch_injection():
    # E1b §4-I2: gf180's sw_stat_mismatch=1 deck-level adapter landed — the mismatch token no
    # longer raises; it maps to "typical" (the SAME section as nominal — see pdks.py module
    # docstring) plus the .param override.
    deck = f'.lib "__LIBPATH__" {MM_CORNER_TOKEN}\n.end\n'
    out = substitute_devices(deck, get_profile("gf180mcuD"))
    assert '.lib "__LIBPATH__" typical' in out
    assert ".param sw_stat_mismatch=1" in out


def test_substitute_devices_gf180_nominal_token_still_works_unaffected():
    # the mismatch adapter (mm_param_injection) only fires for the mismatch TOKEN — a nominal-
    # corner deck on gf180mcuD (unaffected by this feature) must still substitute exactly as before,
    # with NO sw_stat_mismatch override.
    deck = '.lib "__LIBPATH__" tt\n.end\n'
    out = substitute_devices(deck, get_profile("gf180mcuD"))
    assert '.lib "__LIBPATH__" typical' in out
    assert "sw_stat_mismatch" not in out


def test_substitute_devices_raises_mismatch_unavailable_for_a_synthetic_no_mm_profile():
    # the guard itself (spec §4-I1 requirement 3) still exists and still fires — re-targeted onto a
    # SYNTHETIC profile now that both real non-sky130 profiles have a working I2 adapter (see module
    # docstring: this is the intended I1->I2 evolution, not a weakened test).
    deck = f'.lib "__LIBPATH__" {MM_CORNER_TOKEN}\n.end\n'
    with pytest.raises(MismatchUnavailable):
        substitute_devices(deck, _no_mm_profile())


# ── require_mismatch_available (spec §4-I1 requirement 3) ──


def test_require_mismatch_available_passes_for_all_3_pdks_now_adapters_landed():
    # E1b §4-I2: gf180/ihp both flipped lib_section_mm ON — none of the 3 registered PDKs raise.
    for pdk in ("sky130A", "gf180mcuD", "ihp-sg13g2"):
        require_mismatch_available(get_profile(pdk))   # must not raise


def test_require_mismatch_available_raises_for_a_synthetic_no_mm_profile():
    with pytest.raises(MismatchUnavailable, match="synthetic-no-mm"):
        require_mismatch_available(_no_mm_profile())


# ── executor early guard: raises BEFORE any render/sim (spec §4-I1 requirement 3) ──


def _mm_recipe(pdk: str, mc_runs: int = 30) -> VerificationRecipe:
    cond = AnalogPVT(corner=MM_CORNER_TOKEN, temp_c=27.0, vdd=1.8, mc_runs=mc_runs,
                      pdk_profile={"pdk": pdk})
    return VerificationRecipe(
        topology_class=OTA5T, build={"method": "template", "template_ref": "ota5t_offset_mc"},
        conditions=cond, sweeps=[{"analysis": "mc", "knob": "vos", "measure": ["vos_v"]}],
        claim_cards=[ClaimCard(id="vos", topology_class=OTA5T, conditions=cond,
            mechanism=MechanismClaim(knob="vos", metric="vos_v", series_ref="vos",
                quant=QuantTest(kind="statistical", reducer="three_sigma", bound=0.020, absolute=True),
                narrative="offset"))])


def test_executor_raises_mismatch_unavailable_for_a_synthetic_no_mm_pdk_before_any_render_or_sim(monkeypatch):
    # E1b §4-I2: gf180/ihp both flipped lib_section_mm ON, so the "raises before any render/sim"
    # BEHAVIOR is re-targeted onto a synthetic no-mismatch profile registered temporarily into
    # PDK_PROFILES (spec-intended I1->I2 evolution: the guard CODE is unchanged and still tested,
    # only the profile that exercises it changed because both real ones now have a working adapter).
    monkeypatch.setitem(pdks.PDK_PROFILES, "synthetic-no-mm", _no_mm_profile())

    class _ExplodingRunner:
        def measure(self, deck, pdk="sky130A", timeout=300):
            raise AssertionError("runner.measure must never be called when mismatch is unavailable")

    with pytest.raises(MismatchUnavailable):
        run_recipe(_mm_recipe("synthetic-no-mm"), _ExplodingRunner())


def test_executor_pelgrom_mode_also_gated_for_a_synthetic_no_mm_pdk(monkeypatch):
    # Pelgrom mode always renders under MM_CORNER_TOKEN internally regardless of conditions.corner
    # (executor.py's pelgrom branch) — the guard must catch it via the sweep-analysis check too.
    monkeypatch.setitem(pdks.PDK_PROFILES, "synthetic-no-mm", _no_mm_profile())
    cond = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8, mc_runs=12, areas=[0.25, 1.0],
                      pdk_profile={"pdk": "synthetic-no-mm"})
    rec = VerificationRecipe(
        topology_class=OTA5T, build={"method": "template", "template_ref": "ota5t_offset_mc"},
        conditions=cond, sweeps=[{"analysis": "pelgrom", "knob": "vos", "areas": [0.25, 1.0],
                                   "measure": ["a_vos"]}],
        claim_cards=[ClaimCard(id="pel", topology_class=OTA5T, conditions=cond,
            mechanism=MechanismClaim(knob="vos", metric="a_vos", series_ref="pel",
                quant=QuantTest(kind="elasticity", band=(-0.6, -0.2)), narrative="pelgrom"))])

    class _ExplodingRunner:
        def measure(self, deck, pdk="sky130A", timeout=300):
            raise AssertionError("runner.measure must never be called when mismatch is unavailable")

    with pytest.raises(MismatchUnavailable):
        run_recipe(rec, _ExplodingRunner())


def test_executor_gf180_mismatch_recipe_is_no_longer_gated():
    # E1b §4-I2: gf180's sw_stat_mismatch=1 adapter landed — the recipe now RUNS (and judges) rather
    # than raising. Complements test_executor_sky130_mismatch_recipe_is_not_gated below.
    class _FakeRunner:
        def measure(self, deck, pdk="sky130A", timeout=300):
            assert ".param sw_stat_mismatch=1" in deck
            return {"vos": [(float(i), 0.001 * (1 if i % 2 else -1)) for i in range(30)]}

    res = run_recipe(_mm_recipe("gf180mcuD"), _FakeRunner())
    assert res.claim_cards[0].verdict is not None


def test_executor_ihp_mismatch_recipe_is_no_longer_gated():
    # E1b §4-I2: ihp's mm_ok=1 instance adapter landed — the recipe now RUNS (and judges) rather
    # than raising before any sim.
    calls: list[str] = []

    class _FakeRunner:
        def measure(self, deck, pdk="sky130A", timeout=300):
            calls.append(deck)
            assert "mm_ok=1" in deck
            return {"vos": [(float(i), 0.001 * (1 if i % 2 else -1)) for i in range(30)]}

    res = run_recipe(_mm_recipe("ihp-sg13g2"), _FakeRunner())
    assert res.claim_cards[0].verdict is not None
    assert len(calls) == 1   # exactly one sim invocation — no early raise, no double-run


def test_executor_sky130_mismatch_recipe_is_not_gated():
    class _FakeRunner:
        def measure(self, deck, pdk="sky130A", timeout=300):
            return {"vos": [(float(i), 0.001 * (1 if i % 2 else -1)) for i in range(30)]}

    res = run_recipe(_mm_recipe("sky130A"), _FakeRunner())
    assert res.claim_cards[0].verdict is not None


# ── Pelgrom per-area seed-offset distinctness (spec §4-I1 requirement 4) ──


def _seedval(deck: str) -> int:
    m = re.search(r"let seedval = (\d+) \+ mc", deck)
    assert m, f"no seed line found in deck:\n{deck}"
    return int(m.group(1))


def test_pelgrom_areas_get_distinct_seed_offsets():
    captured: list[str] = []

    class _AreaRunner:
        def measure(self, deck, timeout=300):
            captured.append(deck)
            return {"vos": [(float(i), 0.001) for i in range(12)]}

    cond = AnalogPVT(corner=MM_CORNER_TOKEN, temp_c=27.0, vdd=1.8, mc_runs=12, areas=[0.25, 1.0, 4.0])
    rec = VerificationRecipe(
        topology_class=OTA5T, build={"method": "template", "template_ref": "ota5t_offset_mc"},
        conditions=cond, sweeps=[{"analysis": "pelgrom", "knob": "vos", "areas": [0.25, 1.0, 4.0],
                                   "measure": ["a_vos"]}],
        claim_cards=[ClaimCard(id="pel", topology_class=OTA5T, conditions=cond,
            mechanism=MechanismClaim(knob="vos", metric="a_vos", series_ref="pel",
                quant=QuantTest(kind="elasticity", band=(-0.6, -0.2)), narrative="pelgrom"))])

    run_recipe(rec, _AreaRunner())

    assert len(captured) == 3
    seeds = [_seedval(d) for d in captured]
    assert seeds == [MC_SEED_BASE + i * MC_SEED_AREA_STRIDE for i in range(3)]
    assert len(set(seeds)) == 3   # no two areas share a base seed


def test_render_ota5t_offset_mc_seed_offset_wires_into_the_control_block():
    d = render_ota5t_offset_mc(mc_runs=5, seed_offset=0)
    assert f"let seedval = {MC_SEED_BASE} + mc" in d
    assert "setseed $&seedval" in d
    d2 = render_ota5t_offset_mc(mc_runs=5, seed_offset=MC_SEED_AREA_STRIDE)
    assert f"let seedval = {MC_SEED_BASE + MC_SEED_AREA_STRIDE} + mc" in d2


# ── _scale_geometry_value: plain + u-suffixed (spec §4-I1 requirement 5) ──


def test_scale_geometry_value_plain_numeric():
    assert _scale_geometry_value("8", 0.25) == "2"
    assert _scale_geometry_value("4", 4.0) == "16"


def test_scale_geometry_value_suffixed_preserves_suffix():
    # ihp's PDK_SIZING_OVERRIDES ship u-suffixed geometry (e.g. "8u") — a plain float() would raise
    # ValueError on this; the scaled value must keep the SAME suffix (spec: "reuse parse_unit,
    # re-emit with the suffix").
    assert _scale_geometry_value("8u", 0.25) == "2u"
    assert _scale_geometry_value("4u", 0.5) == "2u"
    assert _scale_geometry_value("500f", 2.0) == "1000f"


def test_scale_geometry_value_matches_ihp_override_shape():
    override = sizing_overrides_for(OTA5T, "ihp-sg13g2")
    scaled = {k: _scale_geometry_value(v, 0.25) for k, v in override.items()}
    assert scaled == {"W1": "2u", "W3": "2u", "W5": "2u", "W5b": "1u", "Lp": "0.125u"}


def test_executor_pelgrom_scales_ihp_suffixed_sizing_without_raising(tmp_path):
    # regression: the PRE-E1b executor did `str(float(a_sizing[_k]) * area)`, which ValueErrors on
    # "8u" — this is the exact wrong-unit-area-sweep mis-instrument the STOP rule guards against.
    captured: list[str] = []

    class _AreaRunner:
        def measure(self, deck, pdk="sky130A", timeout=300):
            captured.append(deck)
            return {"vos": [(float(i), 0.001) for i in range(8)]}

    # suffixed-geometry scaling is PDK-independent executor logic; exercised here under the
    # sky130A profile (ihp's mm path is gated until its I2 injector) using ihp's u-suffixed
    # override VALUES as the test data that used to ValueError pre-E1b.
    override = sizing_overrides_for(OTA5T, "ihp-sg13g2")
    cond = AnalogPVT(corner=MM_CORNER_TOKEN, temp_c=27.0, vdd=1.5, mc_runs=8, areas=[0.25, 1.0],
                      pdk_profile={"pdk": "sky130A"})
    rec = VerificationRecipe(
        topology_class=OTA5T, build={"method": "template", "template_ref": "ota5t_offset_mc"},
        conditions=cond, sizing={"seed": override},
        sweeps=[{"analysis": "pelgrom", "knob": "vos", "areas": [0.25, 1.0], "measure": ["a_vos"]}],
        claim_cards=[ClaimCard(id="pel", topology_class=OTA5T, conditions=cond,
            mechanism=MechanismClaim(knob="vos", metric="a_vos", series_ref="pel",
                quant=QuantTest(kind="elasticity", band=(-0.6, -0.2)), narrative="pelgrom"))])

    run_recipe(rec, _AreaRunner())   # must not raise ValueError

    assert len(captured) == 2
    assert "W1=2u" in captured[0]     # area=0.25: 8u * 0.25 -> 2u, suffix preserved
    assert "W1=8u" in captured[1]     # area=1.0: unchanged


# ── live-gated: seeded MC determinism (docker) ──

import os  # noqa: E402

from openclaw_brain.knowledge.executable.runner import NgspiceRunner  # noqa: E402

_HAS_NGSPICE = NgspiceRunner().available()


def _wd(tmp_path, tag: str) -> str:
    # NgspiceRunner workdir MUST be under $HOME — Colima mounts $HOME, NOT pytest's /private/var
    # tmp_path (verified live — matches runner.py's own comment). tmp_path.name is unique per test.
    return os.path.expanduser(f"~/.openclaw_brain/sim_{tmp_path.name}_{tag}")


@pytest.mark.skipif(not _HAS_NGSPICE, reason="IIC-OSIC-TOOLS image not present")
def test_seeded_mc_sky130_is_byte_identical_across_two_separate_runs(tmp_path):
    """REGRESSION PIN (spec §0/§4-I1/§5, the STOP-rule requirement): the SAME tiny seeded MC deck,
    run twice through two independent NgspiceRunner invocations, must produce byte-identical RDATA.
    Verified live during I1 (see mc_templates.py's module docstring for the probe transcript):
    `setseed $&seedval` (a `let`-computed, nonzero value) makes ngspice's `agauss()` draws
    reproducible across separate process invocations; `setseed 0` does NOT (verified — it behaves
    like "no override"), which is why MC_SEED_BASE is pinned >= 1."""
    deck = render_ota5t_offset_mc(mc_runs=8)
    runner_a = NgspiceRunner(workdir=_wd(tmp_path, "a"))
    runner_b = NgspiceRunner(workdir=_wd(tmp_path, "b"))

    series_a = runner_a.measure(deck, pdk="sky130A")
    series_b = runner_b.measure(deck, pdk="sky130A")

    assert series_a == series_b
    assert len(series_a["vos"]) == 8


# ── live-gated: I2 mismatch adapters, through the FULL render->substitute_devices->
#    runner.measure->run_recipe path (not a hand deck) — spec §4-I2 items 1/2 ──

from openclaw_brain.knowledge.executable.executor import run_recipe as _run_recipe_live  # noqa: E402


def _mm_offset_recipe(pdk: str, mc_runs: int = 8) -> VerificationRecipe:
    sizing = sizing_overrides_for(OTA5T, pdk) or None
    cond = AnalogPVT(corner=MM_CORNER_TOKEN, temp_c=27.0, vdd=1.8, mc_runs=mc_runs,
                      pdk_profile={"pdk": pdk})
    return VerificationRecipe(
        topology_class=OTA5T, build={"method": "template", "template_ref": "ota5t_offset_mc"},
        conditions=cond, sizing=({"seed": sizing} if sizing else {}),
        sweeps=[{"analysis": "mc", "knob": "vos", "measure": ["vos_v"]}],
        claim_cards=[ClaimCard(id="vos", topology_class=OTA5T, conditions=cond,
            mechanism=MechanismClaim(knob="vos", metric="vos_v", series_ref="vos",
                quant=QuantTest(kind="statistical", reducer="three_sigma", bound=0.020, absolute=True),
                narrative="offset"))])


@pytest.mark.skipif(not _HAS_NGSPICE, reason="IIC-OSIC-TOOLS image not present")
def test_gf180_mismatch_adapter_live_smoke_through_full_framework_path(tmp_path):
    """I2 gf180 adapter (mm_param_injection={"sw_stat_mismatch": "1"}), exercised through the REAL
    render->substitute_devices->runner.measure->run_recipe path (not a hand deck — pdks.py's module
    docstring I2 addendum has the hand-deck control/treatment probe this reproduces mechanically).
    I1's null result (zero variance) is root-caused as a unit-domain gap, now fixed; this asserts
    the fix holds end-to-end: genuinely nonzero, seeded-reproducible spread."""
    runner = NgspiceRunner(workdir=_wd(tmp_path, "gf180mm"))
    res = _run_recipe_live(_mm_offset_recipe("gf180mcuD"), runner)
    ys = [y for _, y in res.canonical["vos_vos_v"]]
    assert len(ys) == 8
    assert len(set(ys)) > 1, "mismatch draws must vary run-to-run (sw_stat_mismatch=1 must be live)"
    assert statistics.pstdev(ys) > 1e-6   # genuinely nonzero, not float noise


@pytest.mark.skipif(not _HAS_NGSPICE, reason="IIC-OSIC-TOOLS image not present")
def test_ihp_mismatch_adapter_live_smoke_through_full_framework_path(tmp_path):
    """I2 ihp adapter (mm_instance_params={"mm_ok": "1"}), exercised through the REAL render->
    substitute_devices->runner.measure->run_recipe path. I1's live seeded smoke measured a ZERO-
    variance null (bit-identical to 15 sig figs) root-caused to the per-instance mm_ok gate; this
    asserts the fix holds end-to-end."""
    runner = NgspiceRunner(workdir=_wd(tmp_path, "ihpmm"))
    res = _run_recipe_live(_mm_offset_recipe("ihp-sg13g2"), runner)
    ys = [y for _, y in res.canonical["vos_vos_v"]]
    assert len(ys) == 8
    assert len(set(ys)) > 1, "mismatch draws must vary run-to-run (mm_ok=1 must be live)"
    assert statistics.pstdev(ys) > 1e-6


@pytest.mark.skipif(not _HAS_NGSPICE, reason="IIC-OSIC-TOOLS image not present")
def test_gf180_and_ihp_mismatch_adapters_stay_seeded_deterministic(tmp_path):
    """The I2 adapters must not break the I1 seeding guarantee (spec's STOP rule) — two independent
    NgspiceRunner invocations of the identical seeded mismatch deck must be byte-identical, for
    BOTH new adapters, exactly as already pinned for sky130A above."""
    for pdk in ("gf180mcuD", "ihp-sg13g2"):
        runner_a = NgspiceRunner(workdir=_wd(tmp_path, f"{pdk}_det_a"))
        runner_b = NgspiceRunner(workdir=_wd(tmp_path, f"{pdk}_det_b"))
        res_a = _run_recipe_live(_mm_offset_recipe(pdk), runner_a)
        res_b = _run_recipe_live(_mm_offset_recipe(pdk), runner_b)
        assert res_a.canonical["vos_vos_v"] == res_b.canonical["vos_vos_v"], pdk
