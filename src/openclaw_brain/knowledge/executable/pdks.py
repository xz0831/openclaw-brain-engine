"""PDK profile layer (E1 §3, §5-I1) — ports a validated sky130 T-template to a different foundry
model card by TOTAL, mechanical POST-RENDER substitution. Templates keep their sky130 device
literals (the T-template guarantee is untouched); a profile only ever rewrites already-rendered
deck TEXT. sky130A is the IDENTITY profile — `substitute_devices` short-circuits to a byte-identical
no-op for it, which is the regression guarantee spec §6 tests against.

S3-inc2a NOTE: `ptat_ctat_core_bjt` (templates.py) is the first template outside the "exactly 2
device strings (nfet/pfet)" substitution surface ADR-045 verified — it instantiates a sky130 BJT
(`sky130_fd_pr__pnp_05v5_*`). This registry carries no BJT literals for gf180mcuD/ihp-sg13g2, so the
template is SKY130-ONLY for now (cross-PDK BJT support rides increment 3's full-registry rollout);
`substitute_devices` raises `BjtUnavailable` rather than silently emitting a MOS-substituted deck that
still references an undefined sky130 BJT subckt on the target PDK. See `_PNP_SKY130_PREFIX` below.

GROUND TRUTH (this is the deliverable, not a side effect — spec §5-I1 item 1): the spec's
PDK_PROFILES table was explicitly a best-guess. Verified live 2026-07-04 against
`hpretl/iic-osic-tools:latest` (image id `7371bae55da4`, `docker images -q`). Reproduce with:

    docker run --rm --entrypoint bash hpretl/iic-osic-tools:latest -c "
      find /foss/pdks -path '*gf180*ngspice*' -iname '*.ngspice'
      grep -inE '^\\.lib ' /foss/pdks/gf180mcuD/libs.tech/ngspice/sm141064.ngspice
      grep -n '^\\.subckt nfet_03v3 ' /foss/pdks/gf180mcuD/libs.tech/ngspice/sm141064.spice
      find /foss/pdks/ihp-sg13g2/libs.tech/ngspice/models -name 'cornerMOSlv.lib'
      grep -inE '^\\.LIB ' /foss/pdks/ihp-sg13g2/libs.tech/ngspice/models/cornerMOSlv.lib
      grep -n '^\\.subckt sg13_lv_nmos ' /foss/pdks/ihp-sg13g2/libs.tech/ngspice/models/sg13g2_moslv_mod.lib
    "

**sky130A** (baseline, unchanged) — `sky130.lib.spice`, sections tt/ff/ss/sf/fs, devices
`sky130_fd_pr__nfet_01v8`/`sky130_fd_pr__pfet_01v8`, 1.8V. Bare `W=`/`L=` numbers (no unit suffix)
resolve to MICRONS because `corners/tt.spice` transitively `.include`s `../all.spice`, which sets
`.option scale=1.0u` — this is why the existing validated templates can write `W=8` and mean 8µm.

**gf180mcuD** (spec's guess CONFIRMED, plus two structural facts the guess did not capture) —
`sm141064.ngspice`, section `typical` (also verified: `ff`/`ss`/`sf`/`fs`), devices `nfet_03v3`/
`pfet_03v3` (4-terminal subckts `d g s b`, X-prefix instantiation — same shape as sky130, which is
exactly why literal device-name substitution is sufficient), 3.3V, GF 180nm MCU process (doc header:
"0.18um 3.3V/6V ... process"). NOT in the spec's guess, found only by live probing:
  1. `sm141064.ngspice`'s `.lib typical` section references globals (`sw_stat_mismatch`,
     `nfet_03v3_noia`, ...) that are defined in a SEPARATE file, `design.ngspice`, in the same
     directory — `.include`ing it is REQUIRED before the `.lib` call or ngspice fails with
     `Undefined parameter [sw_stat_mismatch]` (verified fatal, reproduced live). See
     `extra_include_glob`.
  2. Unlike sky130, gf180's ngspice tree sets NO `.option scale`. Bare `W=4 L=0.5` (the sky130-style
     unsuffixed numbers the templates render) are then interpreted as 4 METERS / 0.5 METERS —
     verified fatal: `could not find a valid modelname` (the geometry falls outside every binned
     model's W/L range). gf180's own shipped testbenches
     (`libs.tech/xschem/tests/test_nfet_03v3.sch`, `/foss/examples/demo_gf180mcuD/ana/inv.sch`)
     always write `W=1u L=0.28u` — explicit micron suffixes. Injecting `.option scale=1.0u`
     ourselves (rather than rewriting every per-template geometry parameter name, which differs
     across templates — W/Wn/Wp/WN/WP/WT/W1/W3/...) reproduces sky130's convention exactly and
     needs no template-body change: verified live, `iout` came back 1.0559E-05 (sane, ~10µA
     reference) instead of the fatal model-not-found error. See `needs_scale_1u`.

**ihp-sg13g2** (spec's guess for lib/section/devices CONFIRMED exactly; marked EXPERIMENTAL,
honestly, per spec §5-I1's explicit escape hatch) — `cornerMOSlv.lib`, section `mos_tt` (also
`mos_ss`/`mos_ff`/`mos_sf`/`mos_fs` + `_mismatch`/`_stat` variants), devices `sg13_lv_nmos`/
`sg13_lv_pmos`. This PDK IS structurally different, verified two ways:
  1. The devices are Verilog-A/OSDI PSP103 compact models (`libs.tech/ngspice/osdi/*.osdi`), not
     plain BSIM `.model` cards. ngspice must load each `.osdi` plugin via its `osdi 'path'` control
     command BEFORE the netlist is parsed (a `.model ... psp103` line fails to parse otherwise) —
     `osdi` is not a valid netlist-level directive (verified fatal: `Undefined parameter [foss]`,
     ngspice mis-parses it as a device-instantiation line). The PDK's own `libs.tech/ngspice/
     install.py` confirms the intended mechanism: it symlinks a `.spiceinit` (containing 4 `osdi`
     lines) into `$HOME`. ngspice auto-sources a LOCAL `./.spiceinit` from its cwd, which SHADOWS (replaces, not augments)
     `$HOME`'s — writing one into the run's workdir and invoking ngspice from there is the verified
     -working substitute (op sim converged, `iout` 1.3276E-05 A). See `needs_osdi`.
  2. Unlike gf180, `.option scale=1.0u` does NOT fix ihp's bare-number geometry: `w=4 l=0.5` gave
     the identical (wrong, ~100x high) 1.42556E-03 A result WITH and WITHOUT the scale option —
     the OSDI/Verilog-A parameter path evidently does not consult the classic SPICE `scale` option.
     Only an explicit `u` suffix on every override (`w=4u l=0.5u`, verified sane: 1.3276E-05 A)
     works. That is a genuine sizing concern (which literal values to emit), not a mechanical
     render-substitution one — I1 does not attempt it (§5-I1 scope: infra, not bias); I2's
     PDK_SIZING_OVERRIDES entry for ihp-sg13g2 MUST supply explicitly-suffixed geometry, unlike
     gf180mcuD where a bare numeric override plus `needs_scale_1u` suffices. `node_nm=130` reflects
     IHP SG13G2's well-known 130nm SiGe BiCMOS process (not independently re-derived from an
     in-image spec doc); `vdd_nom=1.5` is the spec's LV-device nominal, likewise not independently
     re-confirmed against a voltage datasheet in-image — both are lower-confidence than the
     `lib_glob`/`lib_section`/device-name facts above, which were read directly off file contents.

GROUND TRUTH — E1b §4-I1 item 1, gf180mcuD PER-DEVICE MISMATCH (the required deliverable; decides
Q1's gf180 half — see docs/superpowers/specs/2026-07-04-e1b-statistical-cross-pdk.md §1). VERDICT:
`lib_section_mm=None` (N/A-by-instrument), but NOT because the section is absent — it is genuinely
more interesting than that, and the distinction matters for anyone revisiting this later:

  1. gf180 DOES ship real per-device MOS mismatch machinery: `sm141064.ngspice`'s `fets_mm` .lib
     section (pulled into EVERY corner, including the nominal `typical` one, via `.lib
     'sm141064.spice' fets_mm`) defines `nfet_03v3`/`pfet_03v3` with
     `delvto='mis_vth*sw_stat_mismatch'` / `mulu0='1-mis_k*sw_stat_mismatch'`, where
     `mis_vth='agauss(0,var_vth,1)'` and `var_vth` is a real, nonzero, area-dependent Pelgrom-style
     term (`0.7071*par_vth*1e-6/sqrt(leff*weff)`, par_vth=0.007148 for nfet_03v3 — ~4.2mV σ at
     W=4u/L=0.5u). This is gated by `.param sw_stat_mismatch` — 0 by DEFAULT in `design.ngspice`
     (`.param sw_stat_mismatch=0`; the file's own header table documents `sw_stat_mismatch=1` as
     "most realistic" mismatch-on). This is a genuine "mc_mismatch toggle in design.ngspice" (the
     exact alternative shape the spec told I1 to look for) — gf180 is NOT simply absent.
  2. The `delvto`/`mulu0` instance-parameter PATHWAY is functionally live in this ngspice/BSIM4
     build: verified live by hardcoding `mis_vth=0.1` (a fixed 100mV offset, bypassing agauss
     entirely) on a copy of `sm141064.ngspice` with `sw_stat_mismatch=1` — iout on a 2-device
     current-mirror probe shifted from -1.00890E-05 (no mismatch) to -1.00020E-05 (forced offset),
     confirming ngspice actually applies `delvto` for this model card.
  3. ROOT CAUSE (adversarial verify, live-falsified — supersedes I1's initial "does not resample"
     claim, which was WRONG): the null result is a UNIT-DOMAIN defect in THIS FRAMEWORK'S gf180
     convention, not a PDK or ngspice limitation. `sm141064.ngspice`'s sigma arithmetic
     (`var_vth = '0.7071*par_vth*1e-06/p_sqrtarea'`, with `par_leff = 'l - par_l'`, par_l=1.5e-7)
     expects w/l in METERS inside `.param` math — but the framework renders bare micron-count
     numbers (`W=4 L=0.5`) and relies on `.option scale=1.0u`, which scales DEVICE geometry only,
     never `.param` arithmetic. Result: σ computed ~1e6× too small (~3.6nV instead of ~3.6mV) —
     and the ~2.3e-7-relative run-to-run drift the probe DID show at 15 digits IS that attenuated
     mismatch signal resampling on every `reset` (initially misread as solver noise). Live control:
     the IDENTICAL probe with u-suffixed geometry (`W=4u L=0.5u`) and NO scale option shows
     FULL-STRENGTH resampling — iout spread -9.313e-06..-1.083e-05 (~15%) over 8 resets, on par
     with the sky130 `tt_mm` control (~20%). (Also noted: the point-2 forced-offset probe's small
     0.86% shift is common-mode cancellation — BOTH mirror devices got the same fixed offset — not
     evidence of weak delvto coupling.)
  4. Net verdict: gf180's mismatch statistics ARE usable under ngspice — Q1's gf180 half is
     "available, framework adapter required", NOT "absent from the PDK". `lib_section_mm` stays
     None TODAY because the current adapter (bare numbers + scale option) silently attenuates σ by
     1e6× — a recipe run now would certify a zero-variance non-signal, the exact mis-instrument
     the STOP rule exists to catch. Flipping it on requires the I2 adapter: u-suffixed geometry
     for gf180 (drop `needs_scale_1u` for mismatch decks or move gf180 to suffixed sizing like
     ihp) + a `sw_stat_mismatch=1` injection mechanism.

DOWNGRADED TO None PENDING THE I2 ADAPTER — ihp-sg13g2 `mos_tt_mismatch`: the `.LIB` sections
`mos_tt_mismatch`/`mos_tt_stat` are present and correctly named (line-verified), and I1's live
seeded smoke reproduced a ZERO-variance null (bit-identical to 15 significant digits across 8
seeded runs). ROOT CAUSE (adversarial verify, live-verified both ways — supersedes I1's
OSDI/PSP103-plugin-state hypothesis, which was WRONG): every mismatch `agauss` draw in
`sg13g2_moslv_mod_mismatch.lib` is gated by `(mm_ok != 1 ? 0 : 1)` (e.g. `delvto =
'agauss(0, sg13g2_lv_nmos_delvto_mm/sqrt(m*l*w*1e12), (mm_ok != 1 ? 0 : 1))'`, lines 78-101),
where `mm_ok` is a PER-INSTANCE subckt parameter DEFAULTING TO 0 (subckt line 67). Framework-
rendered decks never pass `mm_ok`, so ihp statistical runs are GUARANTEED zero-variance — with
`mm_ok=1` on the MOS instances the identical probe shows a genuine ~7% mirror-iout spread over 8
resets (-1.176e-05..-1.412e-05); with `mm_ok=0`, bit-identical all 8. A global `.param` CANNOT
override a subckt formal default — the I2 adapter must inject `mm_ok=1` at the INSTANCE level
(substitute_devices-style rewrite of MOS instance lines) for mismatch decks. Until that adapter
exists, `lib_section_mm=None` — because the alternative is worse than unavailability: a
mismatch recipe would silently certify σ=0 as a passing statistical claim.

CONFIRMATION — sky130A `tt_mm` (baseline instrument sanity, same probe methodology as above): the
IDENTICAL 2-device current-mirror + `reset`+`op` loop against sky130's `tt_mm` section showed a
genuine ~20% iout spread across 8 resets within ONE ngspice process (proving the `reset`-resampling
mechanism itself is sound in this image) — this is the control that makes the gf180 null result
above a real finding rather than a broken test methodology.

E1b §4-I2 ADAPTERS LANDED (both PDKs flipped `lib_section_mm` ON) — 2026-07-04, adversarial-verified
via a fresh hand-deck probe (control + treatment, both live) BEFORE any code was written, then
re-verified through the actual render→substitute_devices→runner.measure code path:

  gf180mcuD (`mm_param_injection={"sw_stat_mismatch": "1"}`): a deck-level `.param sw_stat_mismatch=1`
  line, placed AFTER the profile's `extra_include_glob` (design.ngspice, which sets the default to 0)
  and BEFORE the `.lib ... typical` call, overrides the default — ngspice's last-`.param`-wins
  behavior applies across separate `.include`d files, not just within one. Verified on the SAME
  2-device current-mirror probe as the I1 finding above: 8/8 resets bit-identical at 1.0089E-05 A
  with the default (0) left alone; genuine spread 9.7289E-06..1.13334E-05 A (~13-15%, consistent with
  I1's hand-edited-file finding of ~15%) with the override. `lib_section_mm` is set to `"typical"` —
  NOT a distinct section (see PDK_PROFILES entry: fets_mm is pulled into every corner already; the
  override is what activates it, not a section choice).

  ihp-sg13g2 (`mm_instance_params={"mm_ok": "1"}`): every MOS instance (X-card) line instantiating
  `sg13_lv_nmos`/`sg13_lv_pmos` gets ` mm_ok=1` appended — verified against an ACTUAL rendered deck
  (mc_templates.render_ota5t_offset_mc + substitute_devices' nfet/pfet substitution) before the regex
  was written, confirming the exact line shape (`XM5b nbias nbias 0 0 sg13_lv_nmos W={W5b} L={Lp}`,
  the device name as a bare token before the `W=`/`L=` pairs — appending is safe and total). Same
  probe methodology: 8/8 resets bit-identical at 1.29581E-05 A without the injection; genuine spread
  1.19077E-05..1.41997E-05 A (~14-18% peak-to-peak; consistent with I1's ~7%-sigma finding — same
  magnitude range, different unseeded draw) with `mm_ok=1` on both X-cards.

  ONE SIZING CONVENTION (gf180's `needs_scale_1u` DROPPED): gf180 migrated wholesale from
  bare-numbers-plus-`.option scale=1.0u` to explicit u-suffixed PDK_SIZING_OVERRIDES entries,
  matching ihp's convention exactly, rather than keeping two different unit conventions for two
  profiles that both originate from the SAME unit-domain gap. This is a representation change, not a
  magnitude one (verified: ordinary device geometry resolves to the identical meters value either
  way; the fets_mm `.param` arithmetic is the ONLY place the two conventions diverge, which is
  exactly why the old bare+scale convention silently zeroed mismatch sigma). See
  PDK_SIZING_OVERRIDES' module-level comment for the re-validation transcript and
  experiments/E1B_STATISTICAL_CROSS_PDK_REPORT.md §5 for the full port log.

GROUND TRUTH — I1a wave-A port (docs/superpowers/specs/2026-07-05-full-registry-cross-pdk-rollout.md
§3): 6 new templates (miller_ota_2stage_nmos_in, common_gate_nmos, source_follower_nmos,
diff_pair_resistive_nmos, cascode_current_mirror_nmos, regulated_cascode_nmos) x {gf180mcuD,
ihp-sg13g2} — 12/12 ported, ZERO PORT-FAILED. Full transcripts: PDK_SIZING_OVERRIDES' inline comment
above the 12 new entries (below); a compact summary here:

  1. gf180mcuD needed a u-suffixed geometry override on ALL 6 (0/6 zero-override), not >=5/6 as the
     rollout spec's Q2 predicted — that prediction cited pre-"ONE CONVENTION" I1 behavior (when the
     fix lived in a profile-level `.option scale=1.0u` injection, needing no PDK_SIZING_OVERRIDES
     entry at all) and was already stale the day it was frozen (the ONE CONVENTION migration two
     paragraphs up, dated 2026-07-04, retired that mechanism for gf180 entirely the day before).
     11/12 ports (both PDKs, 5 of 6 templates) needed ONLY the geometry override and VERIFIED every
     seed-recipe claim on the FIRST probe (1 probe/port).
  2. regulated_cascode_nmos is the one exception on BOTH axes: it needed a SECOND override kind
     (`VBA`, the gain-boosting auxiliary amplifier's bias voltage — never u-suffixed, unlike every
     other key in this table) on both PDKs, found by a live VBA scan after the sky130-inherited
     default (0.8V) starved the aux loop's NMOS current source of overdrive on both foreign rails
     (verified via direct node-voltage query, scratchpad/probe_rgc_gf180_diag.py: g1c pinned near
     VDD). gf180's fix (VBA=1.8V) fully restores sky130-grade regulation (rgc_iout CoV ~0.013%,
     VERIFIED against the claim's 0.1% bound). ihp-sg13g2's identical claim REFUTES even at its
     scan-optimum bias (VBA=0.55V, CoV floor ~0.28% — confirmed NOT sizing-fixable by an additional
     WA/aux-device-width scan) — a real, probe-confirmed `process_scoped` divergence: the regulated
     cascode's gain-boosting margin is headroom-limited at ihp's 1.5V domain, the SAME axis the
     rollout spec's Q1 pre-registered as its expected divergence candidate (on a different template
     than the spec's own guess). This is I1a's one Q1 divergence: 15/16 (claim, non-baseline-pdk)
     comparisons agree with sky130A, 1/16 (rgc_iout, ihp-sg13g2) diverges — see the increment's port
     table (task handoff) for the full per-claim transcript.

GROUND TRUTH — I1b wave-B port (docs/superpowers/specs/2026-07-05-full-registry-cross-pdk-rollout.md
§3): 6 new templates (telescopic_cascode_ota_nmos_in, folded_cascode_ota_nmos_in,
cds_switched_cap_nmos, single_slope_ramp_generator, column_pga_inverting_nmos, ptat_ctat_core_bjt) x
{gf180mcuD, ihp-sg13g2} — 9/12 ports VERIFIED (matching sky130A), 3/12 PORT-FAILED (0 REFUTED this
wave — the divergence this time is availability/unreachability, not a passing-but-different verdict).
Full transcripts: PDK_SIZING_OVERRIDES' inline comment above the wave-B entries; a compact summary:

  1. telescopic_cascode_ota_nmos_in/ihp-sg13g2 is I1b's ONE process_scoped PORT-FAILED: this topology
     stacks FIVE devices rail-to-rail (tail -> input -> NMOS cascode -> PMOS cascode -> PMOS mirror),
     the tallest stack in the registry. A systematic VBNC/VBPC bias scan (30 points) + a WN width axis
     (re-tested via fresh renders after `alter` on a subckt-internal `.param` proved unreliable inside
     a `.control` loop — a real ngspice limitation, not a probe mistake) never reached a positive,
     CL-flat gain (best -9.7 dB, i.e. still attenuating); direct node query at the best point showed
     the NMOS cascode pinned into deep triode (Vds ~0.017V) at every explored bias — genuinely
     starved, not merely sub-optimal. This IS the rollout spec's own Q3 headroom guess, confirmed on
     the guessed template this time (I1a's regulated_cascode divergence hit the SAME axis, Q1, but on
     a template neither guess named). folded_cascode_ota_nmos_in (one fewer stacked device — the fold
     un-stacks the input from the output cascode) VERIFIED on ihp on the FIRST probe (av0=41.28dB
     flat) — a live, direct confirmation that stack HEIGHT, not the 1.5V rail per se, is the limiting
     factor, since both templates share the same WN/WP/WT/Lp magnitudes and PDK.
  2. [CLOSED 2026-07-05 post-I1b: `LS` became a .param and both ports VERIFY — entries registered
     below; original finding kept verbatim.] single_slope_ramp_generator PORT-FAILED on BOTH pdks
     — NOT a bias/headroom divergence at all,
     an honest INSTRUMENT gap outside this increment's mandate: `_RAMP_BODY`'s reset-switch instance
     hardcodes its length as a bare `L=0.15` literal (not a `{...}` placeholder), so no
     PDK_SIZING_OVERRIDES entry can reach it. Verified fatal on gf180mcuD (bare `0.15` -> 0.15 METERS,
     "could not find a valid modelname") and silently wrong on ihp-sg13g2 (same unit error, but the
     switch never actually resets `vramp`, so both `.meas` calls report "out of interval" and the
     series comes back empty — correctly FLAGGED, never a fake pass). Confirmed fixable by a one-
     character-class template edit (`L=0.15` -> `L={Lp}`) in a scratch-only hand patch (both PDKs then
     produce clean, monotonic ramps) — but templates.py is out of this increment's touch-scope, so the
     port is recorded PORT-FAILED with the root cause named, not silently patched around.
  3. cds_switched_cap_nmos/gf180mcuD needed a genuine RESIZE (not just a unit suffix) on its SECOND
     probe: the naive Lp=0.15u (sky130's near-minimum length, u-suffixed per convention) sits BELOW
     nfet_03v3's own `lmin=0.28um` (verified: sm141064.spice), causing the fets_mm mismatch-sigma
     `.param` arithmetic to divide into an invalid effective length and fatal with `delvto`/`mulu0` =
     nan. Lp=0.3u (a round value above the floor) fixes it cleanly (cov 0, VERIFIED).
  4. ptat_ctat_core_bjt is available on BOTH pdks (see "BJT AVAILABILITY" below) — the rollout spec's
     Q2 guessed THIS template as the most likely honest availability DROP, and that guess did not
     hold; this wave's honest DROP-shaped outcome (3 PORT-FAILED ports) shows up elsewhere instead,
     for a different, headroom/instrument reason — reported as found, not forced either way.
  5. Every other port (folded_cascode_ota_nmos_in x2, cds_switched_cap_nmos/ihp-sg13g2,
     column_pga_inverting_nmos x2, ptat_ctat_core_bjt x2 — 7 of the 9 non-failed ports) VERIFIED on
     the FIRST probe with ONLY the u-suffixed geometry override, same convention as every prior wave.

GROUND TRUTH — I1b "BJT AVAILABILITY" (the required per-PDK PNP probe, spec §3's wave-B item):
verified live 2026-07-05 by direct PDK model-tree search (`find -L` inside the IIC-OSIC-TOOLS
container, same methodology as every other ground-truth section in this docstring) BEFORE writing any
substitution code, then re-verified end to end through `ptat_ctat_core_bjt`'s actual render+substitute
path:

  1. gf180mcuD ships FOUR discrete-size vertical PNP subckts (`pnp_10p00x00p42`, `pnp_05p00x00p42`,
     `pnp_10p00x10p00`, `pnp_05p00x05p00`, all `.subckt <name> c b e par=1 dtemp=0` — the SAME 3-pin
     Collector/Base/Emitter positional order sky130's `pnp_05v5` subckt uses, so a direct name swap
     needs no connectivity change). They are NOT reachable via the MOS "typical" `.lib` section
     already selected for every other template — they live in a SEPARATE `bjt_typical` corner
     (verified: `.lib bjt_typical` / `.lib 'sm141064.spice' bjt_mc` / `.endl bjt_typical`, lines
     265-297 of sm141064.ngspice), needing its OWN second `.lib` call (`PDKProfile.bjt_lib_section`).
     `pnp_05p00x00p42` (the smallest-area option) was chosen, matching sky130's own "small unit
     device" convention. HAND ANCHOR (live, `ptat_ctat_core_bjt` DEFAULT sizing + u-suffixed Wp/Lp,
     27C interpolated from the native 0-85C@5C sweep): measured dVBE = 42.03mV vs ideal VT*ln(5) =
     41.64mV — a +0.4mV (~0.9%) offset, CLEANER than sky130's own +6.9mV (~16.6%) anchor. Both
     direction claims (iptat_a rises, vbe_v falls with temp) VERIFIED on the first probe.
  2. ihp-sg13g2 (a SiGe-BiCMOS process, NPN-centric by design — its high-fT HBTs, npn13G2/npn13G2l/
     npn13G2v, are the whole point of the process) turned out to ALSO ship one discrete PNP subckt,
     `pnpMPA` (`.subckt pnpMPA c b e`, a plain — non-OSDI — Gummel-Poon `.model pnpMPA_mod pnp`, same
     3-pin C/B/E order). It lives in a genuinely SEPARATE model file from the MOS corner
     (`cornerHBT.lib`, not `cornerMOSlv.lib`; verified unique under this PDK's own `*ngspice*`-
     anchored find, the vacask/xyce copies of the same filename living outside that path), needing
     its own `bjt_lib_glob`/`BJT_LIB_PLACEHOLDER` resolution mechanism (parallel to, but distinct
     from, `extra_include_glob` — a different FILE, not just a different `.include`) plus its own
     `.lib` section (`hbt_typ`, verified alongside `hbt_bcs`/`hbt_wcs` + `_mismatch` variants in
     `cornerHBT.lib`). HAND ANCHOR (same methodology): measured dVBE = 44.86mV vs ideal 41.64mV — a
     +3.2mV (~7.7%) offset, the same order of magnitude as sky130's own +6.9mV anchor (and smaller in
     relative terms). Both direction claims VERIFIED on the first probe.
  3. NET: this increment's ONE permitted structural change beyond sizing overrides (the ptat_pnp /
     bjt_lib_glob / bjt_lib_section fields + `substitute_devices`'s BJT branch) was needed on BOTH
     pdks, not neither and not just one — the rollout spec's Q2 prediction of "most likely an honest
     DROP here" did NOT hold; reported exactly as probed, not forced to match or contradict the
     prediction. `BjtUnavailable` stays wired (defense-in-depth) for any FUTURE pdk profile that
     registers no `bjt_pnp`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The post-render placeholder for a REQUIRED extra `.include` (gf180's design.ngspice) — resolved by
# runner.py's render_run_sh alongside templates.LIB_PLACEHOLDER ("__LIBPATH__"). Kept here (not
# templates.py) because it is an E1/profile concept, not a per-template authoring one.
EXTRA_LIB_PLACEHOLDER = "__EXTRA_LIBPATH__"

# I1b (docs/superpowers/specs/2026-07-05-full-registry-cross-pdk-rollout.md §3, wave B): the
# placeholder for a BJT-bearing deck's SECOND `.lib` call, used ONLY when the BJT corner section
# lives in a file DIFFERENT from `lib_glob` (ihp-sg13g2's HBT models: `cornerHBT.lib`, separate from
# `cornerMOSlv.lib`). gf180mcuD's BJT section (`bjt_typical`) lives in the SAME file as its MOS
# section (`sm141064.ngspice`), so gf180's PDKProfile leaves `bjt_lib_glob` None and reuses the
# already-resolved main `__LIBPATH__` token for its second `.lib` line instead (see
# PDKProfile.bjt_lib_glob). Resolved by runner.py's render_run_sh exactly like EXTRA_LIB_PLACEHOLDER.
BJT_LIB_PLACEHOLDER = "__BJTLIBPATH__"

# The two sky130 device literals every template body carries (verified, ADR-045 evidence line:
# "template PDK-coupling surface = exactly 2 device strings (48x nfet_01v8, 24x pfet_01v8)").
_NFET_SKY130 = "sky130_fd_pr__nfet_01v8"
_PFET_SKY130 = "sky130_fd_pr__pfet_01v8"

# S3-inc2a §3/§4: `ptat_ctat_core_bjt` is the FIRST template to instantiate a sky130 BJT
# (`sky130_fd_pr__pnp_05v5_*`) — deliberately OUTSIDE the 2-device nfet/pfet substitution table above
# (ADR-045's "exactly 2 device strings" evidence predates this template). AS OF I1b (see this module's
# "BJT AVAILABILITY" ground-truth section below): gf180mcuD and ihp-sg13g2 BOTH turned out to carry a
# usable PNP (`PDKProfile.bjt_pnp`), so `substitute_devices` now substitutes it like any other device.
# `BjtUnavailable` stays wired for defense-in-depth — any FUTURE profile registered without a
# `bjt_pnp` still gets a clear, early refusal instead of a hybrid deck referencing an undefined
# subckt — but it is no longer raised for either of the two currently-registered non-sky130 profiles.
_PNP_SKY130_PREFIX = "sky130_fd_pr__pnp_05v5"
_PNP_SKY130_FULL = _PNP_SKY130_PREFIX + "_W0p68L0p68"   # the exact literal templates.py instantiates
                                                          # twice (XQ1/XQ2) — substituted wholesale.

# The CANONICAL mismatch corner token every recipe/template renders (seeds.py's AnalogPVT(corner=
# "tt_mm", ...), mc_templates.py's default `corner` kwarg) — spec E1b §3: recipes/seeds stay BYTE-
# UNCHANGED across PDKs; substitute_devices maps THIS token to profile.lib_section_mm exactly as it
# already maps the nominal corner token to profile.lib_section.
MM_CORNER_TOKEN = "tt_mm"

# Every template's DEFAULT_*_SIZING sets VDD to this literal (verified across templates.py) — the
# "VDD default" substitute_devices rewrites. A recipe-level VDD OVERRIDE renders a different numeric
# literal and is deliberately left untouched here: choosing a PDK-appropriate operating point is a
# sizing/bias decision (I2's PDK_SIZING_OVERRIDES + probe-validate loop), not a mechanical rewrite.
_VDD_DEFAULT_LITERAL = "1.8"


class UnknownPDK(Exception):
    """get_profile() was asked for a pdk key not in PDK_PROFILES — raised BEFORE any render/sim."""


class MismatchUnavailable(Exception):
    """A recipe requires per-device MOS mismatch statistics (mc_runs>0, or an explicit
    `MM_CORNER_TOKEN` corner request) on a PDKProfile whose `lib_section_mm` is None — spec E1b
    §4-I1 requirement 3. Raised BEFORE any render/sim (executor.run_recipe's early guard) and again,
    defense-in-depth, inside substitute_devices itself if ever called directly on a mismatch-token
    deck for such a profile."""


class BjtUnavailable(Exception):
    """A deck instantiates a sky130 BJT (`sky130_fd_pr__pnp_05v5_*`, the `ptat_ctat_core_bjt`
    template) and `substitute_devices` either (a) was asked to port it to a profile whose `bjt_pnp`
    is None, or (b) would have left sky130 `pnp_05v5` residue behind after substitution because the
    deck's exact literal wasn't `_PNP_SKY130_FULL` (the totality guard — the availability check above
    is PREFIX-based but the rewrite itself only ever targets that one exact literal) — silently
    shipping either case would emit a hybrid deck referencing an undefined subckt on the target PDK
    (a confusing deep-ngspice failure instead of a clear, early one). Raised inside
    `substitute_devices`, before the `.lib`/section rewrite proceeds. AS OF I1b, case (a) is no
    longer reachable for gf180mcuD/ihp-sg13g2 (both registered a usable `bjt_pnp` — see the module's
    "BJT AVAILABILITY" ground-truth section) — it stays wired as defense-in-depth for any FUTURE
    profile added without a characterized PNP entry. Case (b) IS reachable today for any deck that
    instantiates a sky130 pnp_05v5 size other than W0p68L0p68 (only unreachable from the currently-
    registered templates, which instantiate exactly that one size)."""


@dataclass(frozen=True)
class PDKProfile:
    """One foundry model card's device/lib identity. See module docstring for the verified values."""

    pdk: str                          # registry key == Specimen.pdk == conditions.pdk_profile.pdk
    lib_glob: str                     # filename glob the runner searches for under PDK_ROOT
    lib_section: str                  # the nominal `.lib "<file>" <section>` token (E1 pilot: nominal-only)
    vdd_nom: float
    nfet: str
    pfet: str
    node_nm: int
    pdk_dir: str | None = None        # directory name under $PDK_ROOT (anchors the runner's `find` so a
                                       # lib/osdi glob can never match a SIBLING PDK's same-named file —
                                       # cornerMOSlv.lib exists in both ihp-sg13g2 and ihp-sg13cmos5l)
    extra_include_glob: str | None = None   # a REQUIRED second `.include` before `.lib` (verified: gf180's
                                             # design.ngspice defines globals sm141064.ngspice's `typical`
                                             # section reads; omitting it is a verified-fatal parse error)
    needs_osdi: bool = False          # verified: devices are Verilog-A/OSDI (PSP103) — ngspice needs each
                                       # *.osdi plugin loaded via a local ./.spiceinit before netlist parse
    experimental: bool = False        # honestly-flagged: structurally different, not I1-hardened (ihp)
    lib_section_mm: str | None = None  # E1b §3/§4-I1: the mismatch-capable `.lib` section token that
                                        # MM_CORNER_TOKEN ("tt_mm") maps to. None = Q1-unavailable —
                                        # verified-by-instrument, not merely "section not found" (see
                                        # module docstring ground truth, esp. gf180mcuD's case).
    mm_param_injection: dict[str, str] | None = None
        # E1b §4-I2 gf180 adapter: extra `.param name=value` line(s) injected into the mismatch-token
        # deck's .lib block (after any extra_include, before the `.lib` call itself) — gf180's
        # `fets_mm` mismatch machinery is gated by design.ngspice's `.param sw_stat_mismatch=0`
        # default; a deck-level `.param sw_stat_mismatch=1` placed AFTER the design.ngspice include
        # overrides it (verified live: a 2-device current-mirror probe went from bit-identical
        # 1.0089E-05 A across 8 resets (sw_stat_mismatch left at 0) to a genuine ~13% spread
        # (9.7289E-06..1.13334E-05) with this override — see module docstring I2 addendum). None for
        # sky130A/ihp-sg13g2 (ihp's gate is per-INSTANCE, not a global .param — see mm_instance_params).
    mm_instance_params: dict[str, str] | None = None
        # E1b §4-I2 ihp adapter: extra param=value pair(s) appended to every MOS instance (X-card)
        # line that instantiates `nfet`/`pfet` in a mismatch-token deck — ihp's mos_tt_mismatch
        # section gates every agauss() draw by a PER-INSTANCE subckt formal `mm_ok` (default 0,
        # verified: `.subckt sg13_lv_nmos d g s b\n+ w=... mm_ok=0 ...` in sg13g2_moslv_mod.lib line
        # 66-67); a global `.param mm_ok=1` CANNOT reach it (subckt formals shadow globals of the
        # same name), so the injection must land ON the instance line itself: `XM1 nd nd 0 0
        # sg13_lv_nmos W={W} L={Lp} mm_ok=1`. Verified live: bit-identical 1.29581E-05 A across 8
        # resets without the injection, genuine ~14-18% spread (1.19077E-05..1.41997E-05, in line
        # with the module docstring's earlier ~7%-sigma finding) with it. None for sky130A/gf180mcuD.
    bjt_pnp: str | None = None
        # I1b (spec §3, wave B): the substitution target for `_PNP_SKY130_FULL`
        # (`ptat_ctat_core_bjt`'s XQ1/XQ2 device) — None means this profile has no characterized PNP
        # (BjtUnavailable stays wired). Both non-sky130 profiles registered one — see the module's
        # "BJT AVAILABILITY" ground-truth section for the live-probed transcript per PDK.
    bjt_lib_glob: str | None = None
        # I1b: filename glob for the file carrying `bjt_lib_section`, ONLY if it differs from
        # `lib_glob` — gf180mcuD's BJT corner (`bjt_typical`) lives in the SAME file as its MOS corner
        # (`sm141064.ngspice`), so this stays None and the second `.lib` line reuses the ALREADY-
        # resolved main libpath token; ihp-sg13g2's HBT models live in a genuinely separate file
        # (`cornerHBT.lib`, not `cornerMOSlv.lib`), so this is set and the runner resolves it via a
        # second `find`, substituting `BJT_LIB_PLACEHOLDER`.
    bjt_lib_section: str | None = None
        # I1b: the `.lib "<file>" <section>` token a BJT-bearing deck ALSO needs, appended as a
        # SECOND `.lib` line after the main MOS one (gf180 "bjt_typical", ihp "hbt_typ"). None for
        # sky130A / any profile without `bjt_pnp`.
    notes: str = ""


PDK_PROFILES: dict[str, PDKProfile] = {
    "sky130A": PDKProfile(
        pdk="sky130A",
        pdk_dir="sky130A", lib_glob="sky130.lib.spice", lib_section="tt", vdd_nom=1.8,
        nfet=_NFET_SKY130, pfet=_PFET_SKY130, node_nm=130,
        lib_section_mm="tt_mm",
        notes="Identity profile — substitute_devices() is a byte-identical no-op (the §6 regression guarantee).",
    ),
    "gf180mcuD": PDKProfile(
        pdk="gf180mcuD",
        pdk_dir="gf180mcuD", lib_glob="sm141064.ngspice", lib_section="typical", vdd_nom=3.3,
        nfet="nfet_03v3", pfet="pfet_03v3", node_nm=180,
        extra_include_glob="design.ngspice",
        lib_section_mm="typical",   # E1b §4-I2: NOT a distinct section — fets_mm (the mismatch machinery)
                                     # is pulled into EVERY corner section INCLUDING typical (verified,
                                     # module docstring), so the "mismatch section" for gf180 genuinely
                                     # equals the nominal one; what differentiates a mismatch deck is
                                     # mm_param_injection (below), not a different .lib token. Modeled
                                     # honestly as such rather than inventing a section that doesn't exist.
        mm_param_injection={"sw_stat_mismatch": "1"},   # design.ngspice defaults this to 0; a deck-level
                                                         # override placed AFTER the design.ngspice include
                                                         # (this profile's extra_include) and BEFORE the
                                                         # `.lib ... typical` call wins — verified live (see
                                                         # module docstring I2 addendum): 8/8 resets bit-
                                                         # identical (1.0089E-05 A) at the design.ngspice
                                                         # default, genuine ~13% spread with the override.
        bjt_pnp="pnp_05p00x00p42",   # I1b: the smallest-area discrete PNP subckt in gf180's catalog
                                      # (`c b e par=1 dtemp=0`, SAME 3-pin Collector/Base/Emitter
                                      # positional order as sky130's pnp_05v5 — a direct name swap,
                                      # no connectivity change needed). See "BJT AVAILABILITY" below.
        bjt_lib_section="bjt_typical",   # lives in the SAME sm141064 file as the MOS "typical" section
                                          # (verified: `.lib bjt_typical` ... `.lib 'sm141064.spice'
                                          # bjt_mc` ... `.endl bjt_typical`) — bjt_lib_glob stays None.
        notes=("VERIFIED 2026-07-04 live (see module docstring): needs design.ngspice included before "
               "the .lib call (undefined-parameter otherwise). Sizing convention (I2): u-suffixed "
               "geometry on every override (PDK_SIZING_OVERRIDES), matching ihp's convention — no "
               ".option scale injection (dropped; see module docstring 'ONE CONVENTION' note). Mismatch "
               "(lib_section_mm): AVAILABLE as of I2 — sw_stat_mismatch=1 deck-level override activates "
               "gf180's own fets_mm agauss() statistics; verified ~13-15% mirror-iout spread over 8 "
               "resets vs bit-identical at the design.ngspice default (sw_stat_mismatch=0). BJT "
               "(bjt_pnp): AVAILABLE as of I1b — pnp_05p00x00p42, a real Gummel-Poon PNP `.model` "
               "gated by a SEPARATE `bjt_typical` corner section (not pulled into the MOS `typical` "
               "one); verified live end-to-end through ptat_ctat_core_bjt (see module docstring)."),
    ),
    "ihp-sg13g2": PDKProfile(
        pdk="ihp-sg13g2",
        pdk_dir="ihp-sg13g2", lib_glob="cornerMOSlv.lib", lib_section="mos_tt", vdd_nom=1.5,
        nfet="sg13_lv_nmos", pfet="sg13_lv_pmos", node_nm=130,
        needs_osdi=True,
        experimental=True,
        lib_section_mm="mos_tt_mismatch",   # E1b §4-I2: section name was already line-verified correct
                                             # at I1 — the I1 null was the per-instance mm_ok gate (default
                                             # 0), not the section. Flipped on now that mm_instance_params
                                             # injects mm_ok=1 on every MOS instance line (below).
        mm_instance_params={"mm_ok": "1"},  # mos_tt_mismatch gates every agauss() draw by a PER-INSTANCE
                                             # subckt formal `mm_ok` (default 0, subckt line 67 of
                                             # sg13g2_moslv_mod.lib) that a global .param cannot reach —
                                             # verified live (module docstring I2 addendum): 8/8 resets
                                             # bit-identical (1.29581E-05 A) without the injection, genuine
                                             # ~14-18% spread with `mm_ok=1` appended to each X-card line.
        bjt_pnp="pnpMPA",   # I1b: the ONE discrete PNP subckt in ihp's HBT model file (`c b e`, no
                             # required keyword params — a direct name swap, same 3-pin positional
                             # order as sky130's pnp_05v5). A SiGe-BiCMOS process is NPN-centric
                             # (npn13G2/npn13G2l/npn13G2v — the high-fT HBTs this process exists for)
                             # but genuinely ships a parasitic/substrate PNP too — see "BJT
                             # AVAILABILITY" below for the live probe that found it.
        bjt_lib_glob="cornerHBT.lib",   # a SEPARATE model file from cornerMOSlv.lib (verified: the
                                         # HBT models live under libs.tech/ngspice/models/cornerHBT.lib,
                                         # unique under the `*ngspice*`-anchored find within this PDK's
                                         # own dir — the vacask/xyce copies of the same filename live
                                         # outside the ngspice path and are never matched).
        bjt_lib_section="hbt_typ",   # the nominal HBT corner (`.LIB "cornerHBT.lib" hbt_typ`, verified
                                      # line-present alongside hbt_bcs/hbt_wcs + _mismatch variants).
        notes=("EXPERIMENTAL (spec §5-I1 escape hatch, invoked honestly, not faked): sg13_lv_nmos/"
               "sg13_lv_pmos are Verilog-A/OSDI PSP103 compact models, not plain BSIM .model cards. "
               "Verified working mechanism: a LOCAL ./.spiceinit (auto-sourced by ngspice alongside "
               "$HOME's) loading the 4 osdi/*.osdi plugins before the netlist parses (op sim of the "
               "current-mirror probe with w=4u l=0.5u converged: iout=1.3276E-05 A). Verified NOT "
               "working: .option scale=1.0u does not fix bare (unsuffixed) W/L for this PDK's OSDI "
               "path — w=4 l=0.5 gave the identical wrong ~1.4mA result with or without scale=1u. "
               "PDK_SIZING_OVERRIDES entries for this pdk supply explicitly u-suffixed geometry. "
               "Mismatch (lib_section_mm): AVAILABLE as of I2 — mm_ok=1 instance injection activates "
               "the per-device agauss() draws; verified ~14-18% mirror-iout spread vs bit-identical "
               "without it. BJT (bjt_pnp): AVAILABLE as of I1b — pnpMPA, a plain (non-OSDI) Gummel-"
               "Poon PNP `.model` in a genuinely separate file (cornerHBT.lib) needing its own `.lib` "
               "call (`hbt_typ`); verified live end-to-end through ptat_ctat_core_bjt (see module "
               "docstring)."),
    ),
}

# Legacy alias: `conditions.pdk_profile.pdk` was already a first-class scope field (pre-E1,
# summarize_scope) and existing seed recipes (seeds.py's `_SKY = {"pdk": "sky130", ...}`) set it to
# the bare string "sky130" — NOT the "sky130A" registry key. Since sky130A is the identity profile
# anyway, aliasing preserves those recipes' behavior exactly (byte-identical, §5-I1 requirement 4)
# instead of making them raise UnknownPDK the first time executor.py starts calling get_profile()
# unconditionally.
_ALIASES: dict[str, str] = {"sky130": "sky130A"}


def get_profile(pdk: str) -> PDKProfile:
    """The PDKProfile for `pdk`. Raises UnknownPDK for anything not in PDK_PROFILES (aliases
    resolved first) — BEFORE any render or sim happens (spec §5-I1 requirement 2)."""
    key = _ALIASES.get(pdk, pdk)
    try:
        return PDK_PROFILES[key]
    except KeyError:
        raise UnknownPDK(f"unknown pdk {pdk!r}; known: {sorted(PDK_PROFILES)}") from None


def require_mismatch_available(profile: PDKProfile) -> None:
    """Raise MismatchUnavailable BEFORE any render/sim if `profile` has no usable per-device MOS
    mismatch section (spec E1b §4-I1 requirement 3). Call this as soon as a recipe's conditions are
    known to require mismatch — executor.run_recipe does this immediately after resolving the
    profile, ahead of sizing/render/measure."""
    if profile.lib_section_mm is None:
        raise MismatchUnavailable(
            f"{profile.pdk!r} has no usable per-device MOS mismatch statistics "
            f"(lib_section_mm=None; verified ground truth — see pdks.py module docstring). "
            f"A recipe requiring mismatch (mc_runs>0 / corner={MM_CORNER_TOKEN!r}) cannot run "
            f"on this profile."
        )


def _inject_mm_instance_params(deck_text: str, profile: PDKProfile) -> str:
    """Append `profile.mm_instance_params` (e.g. ihp's `{"mm_ok": "1"}`) to every MOS instance
    (X-card) line that instantiates `profile.nfet` or `profile.pfet` — mismatch-token decks ONLY
    (spec E1b §4-I2). ihp's `mos_tt_mismatch` gates every agauss() draw by a PER-INSTANCE subckt
    formal (`mm_ok`, default 0) that a global `.param` cannot reach; the exact instance-line shape
    (`XM5b nbias nbias 0 0 sg13_lv_nmos W={W5b} L={Lp}`) was verified by rendering a real deck
    (mc_templates.render_ota5t_offset_mc + substitute_devices) before this regex was written — the
    device name is always the last non-`{...}` token before the `W=`/`L=` param pairs, so matching
    the whole line by device-name word-boundary and appending is safe and total."""
    if not profile.mm_instance_params:
        return deck_text
    extra = "".join(f" {k}={v}" for k, v in profile.mm_instance_params.items())
    names = "|".join(re.escape(n) for n in {profile.nfet, profile.pfet} if n)
    pattern = re.compile(rf"^(X\S+.*\b(?:{names})\b.*)$", re.MULTILINE)
    return pattern.sub(lambda m: m.group(1) + extra, deck_text)


def substitute_devices(deck_text: str, profile: PDKProfile) -> str:
    """Post-render, TOTAL, mechanical substitution of the two sky130 device literals + the VDD
    default + the `.lib` section, per `profile`. Never touches a template body (templates.py is
    untouched by E1) — this runs on already-rendered deck TEXT.

    sky130A is the IDENTITY: returns `deck_text` completely unchanged (not even re-parsed), which is
    the byte-identical regression guarantee spec §6 tests directly.

    E1b §3/§4-I1 requirement 2: a deck whose `.lib` line carries the CANONICAL mismatch token
    (`MM_CORNER_TOKEN`, "tt_mm") maps to `profile.lib_section_mm` — exactly as any other token maps
    to `profile.lib_section` — instead of collapsing onto the nominal section. Raises
    MismatchUnavailable (defense-in-depth; executor.run_recipe's early guard is the primary gate)
    if the profile's `lib_section_mm` is None.

    E1b §4-I2: a mismatch-token deck additionally gets `profile.mm_param_injection` (a global
    `.param` override placed right before the `.lib` call, gf180's `sw_stat_mismatch=1`) and/or
    `profile.mm_instance_params` (per-MOS-instance param=value pairs, ihp's `mm_ok=1`) — whichever
    the profile carries. Both are None for sky130A/every non-mismatch deck, so this is a strict
    no-op addition over the pre-I2 behavior otherwise.

    I1b (spec §3, wave B): if `deck_text` instantiates the sky130 BJT (`ptat_ctat_core_bjt`) and
    `profile.bjt_pnp` is None, raises `BjtUnavailable` BEFORE any further rewrite (defense-in-depth —
    both currently-registered non-sky130 profiles carry a `bjt_pnp`, so this is unreachable for them;
    see the module's "BJT AVAILABILITY" ground-truth section). Otherwise the sky130 PNP literal
    (`_PNP_SKY130_FULL`) is substituted like any other device, and a SECOND `.lib` line for
    `profile.bjt_lib_section` is appended after the main one (gf180: same file, reuses the resolved
    main libpath; ihp: a separate file, resolved via `BJT_LIB_PLACEHOLDER`). Every existing (BJT-free)
    template is completely unaffected — the check is a no-op unless the deck actually contains a
    `pnp_05v5` instantiation.

    TOTALITY GUARD (post-verifier fix): the `is_bjt_deck` availability check above is PREFIX-based
    (`_PNP_SKY130_PREFIX in deck_text`), but the substitution below only ever rewrites the ONE exact
    literal `_PNP_SKY130_FULL`. A deck carrying any OTHER sky130 pnp_05v5 size would otherwise pass
    the availability gate and then have its sky130 device string survive the rewrite untouched —
    silently shipping sky130 residue on a non-sky130 PDK. `substitute_devices` re-checks for the
    prefix immediately after the exact-literal replace and raises `BjtUnavailable` if it still
    appears, rather than returning a hybrid deck.
    """
    if profile.pdk == "sky130A":
        return deck_text

    is_bjt_deck = _PNP_SKY130_PREFIX in deck_text
    if is_bjt_deck and profile.bjt_pnp is None:
        raise BjtUnavailable(
            f"{profile.pdk!r} has no BJT device substitution (sky130 {_PNP_SKY130_PREFIX!r} is "
            "outside the nfet/pfet substitution table); ptat_ctat_core_bjt is unavailable on this profile."
        )

    out = deck_text.replace(_NFET_SKY130, profile.nfet)
    out = out.replace(_PFET_SKY130, profile.pfet)
    if is_bjt_deck:
        out = out.replace(_PNP_SKY130_FULL, profile.bjt_pnp)
        if _PNP_SKY130_PREFIX in out:
            # Totality guard (post-verifier fix): the availability gate above is PREFIX-based
            # (`_PNP_SKY130_PREFIX in deck_text`), but the substitution itself only ever rewrote the
            # ONE exact literal `_PNP_SKY130_FULL` (the W0p68L0p68 size templates.py happens to
            # instantiate). A deck carrying any OTHER sky130 pnp_05v5 size (e.g. a hand-authored or
            # future-template W3p40L3p40 instance — a real sky130 size, see seeds.py's S3-inc2a
            # history) would sail past the availability gate and then ship sky130 residue silently on
            # both gf180mcuD and ihp-sg13g2, reproducing exactly the "hybrid deck / confusing deep-
            # ngspice failure" BjtUnavailable's docstring exists to prevent. Raise rather than emit a
            # deck that still references an undefined sky130 subckt on the target PDK.
            raise BjtUnavailable(
                f"{profile.pdk!r}: BJT substitution left sky130 residue ({_PNP_SKY130_PREFIX!r}) in "
                "the deck — the deck instantiates a sky130 pnp_05v5 size other than the exact "
                f"{_PNP_SKY130_FULL!r} literal this profile's `bjt_pnp` substitution covers; refusing "
                "rather than shipping a hybrid deck referencing an undefined subckt on this PDK."
            )

    # VDD default handling: only the recognized DEFAULT literal is rewritten (see _VDD_DEFAULT_LITERAL).
    out = re.sub(
        rf"(?<![A-Za-z0-9_])(VDD=){re.escape(_VDD_DEFAULT_LITERAL)}\b",
        lambda m: f"{m.group(1)}{profile.vdd_nom:g}",
        out,
    )

    # .lib line: swap in the profile's section (nominal, UNLESS the incoming token is the canonical
    # mismatch token — then map it to lib_section_mm instead, plus mm_param_injection if the profile
    # carries one), and prepend the extra .include this PDK needs (verified per-profile — see module
    # docstring). E1's pilot scope is nominal-only otherwise (spec §4): a corner-sweep recipe run
    # under a non-sky130 profile would have every per-corner deck collapse onto the SAME nominal
    # section — out of scope for I1, not guarded against here.
    is_mismatch_deck = False

    def _lib_block(m: re.Match) -> str:
        nonlocal is_mismatch_deck
        libpath_expr, token = m.group(1), m.group(2)
        if token == MM_CORNER_TOKEN:
            require_mismatch_available(profile)
            section = profile.lib_section_mm
            is_mismatch_deck = True
        else:
            section = profile.lib_section
        block = []
        if profile.extra_include_glob:
            block.append(f'.include "{EXTRA_LIB_PLACEHOLDER}"')
        if is_mismatch_deck and profile.mm_param_injection:
            for k, v in profile.mm_param_injection.items():
                block.append(f".param {k}={v}")
        block.append(f".lib {libpath_expr} {section}")
        if is_bjt_deck and profile.bjt_lib_section:
            bjt_libpath_expr = f'"{BJT_LIB_PLACEHOLDER}"' if profile.bjt_lib_glob else libpath_expr
            block.append(f".lib {bjt_libpath_expr} {profile.bjt_lib_section}")
        return "\n".join(block)

    out = re.sub(r'\.lib\s+("[^"]*")\s+(\S+)', _lib_block, out, count=1)

    if is_mismatch_deck:
        out = _inject_mm_instance_params(out, profile)

    return out


# Per-(topology_class, pdk) sizing overlay, applied AFTER the template default and BEFORE the
# recipe's own sizing.seed (so an author's explicit override always wins). Scaffold was empty for I1
# (sky130A needs no entries: its "override" IS the template defaults already, i.e. identity by
# absence). I2 (spec §5-I2) fills gf180mcuD/ihp-sg13g2 entries per pilot port, each probe->validated
# against REAL ngspice (probe transcripts: scratchpad/e1b_probe/ (repo scratchpad, gitignored-but-
# persistent); probe counts feed experiments/E1_CROSS_PDK_REPORT.md's Q4 table and
# experiments/E1B_STATISTICAL_CROSS_PDK_REPORT.md's port-effort table).
#
# ONE CONVENTION (E1b §4-I2 decision — see pdks.py module docstring for the full justification):
# gf180mcuD MIGRATED WHOLESALE to u-suffixed sizing overrides, dropping the I1 `needs_scale_1u`
# mechanism entirely. Both PDKs now carry the SAME sizing convention (every override an explicit
# micron literal) instead of gf180 keeping bare-numbers-plus-a-global-scale-option while ihp used
# suffixed literals for the identical reason (a unit-domain gap). Verified live that the migration
# is a pure REPRESENTATION change, not a magnitude one: `.option scale=1.0u` and an explicit `u`
# suffix both resolve a bare micron count to the same meters value for ordinary device geometry
# (scale converts unitless MOSFET L/W at elaboration; `u` is already explicit) — the ONLY place they
# differ is `.param`-level arithmetic (gf180's fets_mm mismatch sigma formula), which never saw the
# scale conversion at all, which is exactly the E1b §4-I1 gf180 root cause. So gf180's 3 nominal
# pilots were re-validated (not re-derived) in one probe/live-smoke iteration each against the SAME
# tests/test_executable_pdks.py smokes I1 already had (test_gf180_live_smoke_current_mirror_dc/
# _common_source_ac/_ota5t_ac, re-pointed at the u-suffixed PDK_SIZING_OVERRIDES entries below) — all
# 3 still land in the identical sane ranges those tests already assert (a ~10µA current-mirror
# current, ~10-60dB single-stage gain, GBW well below the AC sweep's Nyquist); exact re-validation
# transcripts (not re-derived, freshly re-run this session): experiments/E1B_STATISTICAL_CROSS_PDK_
# REPORT.md §5. comparator_continuous_nmos is a NEW port for gf180 (E1b's 3rd statistical pilot) —
# probe-validated fresh, not re-derived.
#
# gf180mcuD entries: every value is the template DEFAULT numeric magnitude with a `u` suffix added
# (no rescaling — the defaults were already sane microns, as gf180's I1 finding established).
#
# ihp-sg13g2: EVERY pilot needs an entry — confirms pdks.py's own ground truth (needs_osdi's PSP103
# path does not consult `.option scale`; bare W/L are wrong regardless) generalizes across all 3 I1
# pilots. Bare (unsuffixed) sizing was probed first for all 3 and failed in three DIFFERENT ways (all
# traced to the same root cause: PSP103 misreads the geometry): current_mirror_simple_nmos converged
# to a ~100x-high, still-monotonic current (misleading, not obviously broken); common_source_
# active_load_nmos failed to produce a clean result twice on identical inputs — a hard 300s ngspice-
# runner timeout, then (a second, overlapping attempt) a 204s completion with only 1/5 requested
# points parsed (degenerate, would FLAG not VERIFY) — at bias points that DO converge in <0.3s once
# geometry is explicit; ota_5t_nmos_in converged fast but to a NEGATIVE av0 (-88.6dB, a degenerate/
# mis-biased point, not just a wrong number). Explicit `u`-suffixed geometry resolves all three: iout
# 10.5->17.5uA (~1-1.75x the 10uA IREF, monotonic +); CS av0 25.3-25.7dB / gbw 9.0-83.4MHz; OTA5T av0
# 24.1dB flat / gbw 76.8->10.0MHz (clean 1/CL scaling) — all physically sane for a 1.5V low-voltage
# domain (lower gain than the 1.8V/3.3V ports, consistent with less ro headroom).
# comparator_continuous_nmos on ihp-sg13g2 is a NEW port for E1b (§4-I2 PART B item 4) — same body
# shape as ota_5t_nmos_in (X-card W/L, same DEFAULT_CMP_SIZING magnitudes), probe-validated with the
# SAME suffixed values.
# Full transcripts + the probe-iteration/wall-time tables: experiments/E1_CROSS_PDK_REPORT.md §5a
# (the 3 nominal pilots) and experiments/E1B_STATISTICAL_CROSS_PDK_REPORT.md §5 (the 3 statistical
# pilots + gf180's migration re-validation).
PDK_SIZING_OVERRIDES: dict[tuple[str, str], dict[str, str]] = {
    ("current_mirror_simple_nmos", "ihp-sg13g2"): {"W": "4u", "Lp": "0.5u"},
    ("common_source_active_load_nmos", "ihp-sg13g2"): {"Wn": "8u", "Wp": "16u", "Lp": "0.5u"},
    ("ota_5t_nmos_in", "ihp-sg13g2"): {"W1": "8u", "W3": "8u", "W5": "8u", "W5b": "4u", "Lp": "0.5u"},
    ("comparator_continuous_nmos", "ihp-sg13g2"): {"W1": "8u", "W3": "8u", "W5": "8u", "W5b": "4u", "Lp": "0.5u"},
    ("current_mirror_simple_nmos", "gf180mcuD"): {"W": "4u", "Lp": "0.5u"},
    ("common_source_active_load_nmos", "gf180mcuD"): {"Wn": "8u", "Wp": "16u", "Lp": "0.5u"},
    ("ota_5t_nmos_in", "gf180mcuD"): {"W1": "8u", "W3": "8u", "W5": "8u", "W5b": "4u", "Lp": "0.5u"},
    ("comparator_continuous_nmos", "gf180mcuD"): {"W1": "8u", "W3": "8u", "W5": "8u", "W5b": "4u", "Lp": "0.5u"},

    # I1a (spec docs/superpowers/specs/2026-07-05-full-registry-cross-pdk-rollout.md, wave A — 6
    # simpler bias structures). PROBE FINDING (both PDKs, all 6 templates): every geometry parameter
    # is a BARE (unsuffixed) micron literal in every template's DEFAULT_*_SIZING (Lp=0.5, W/Wn/W1../
    # WA=4..16) — the SAME unit-domain gap the 4 existing entries above already established at the
    # PDK level (gf180 has no `.option scale` at all post the E1b "ONE CONVENTION" migration; ihp's
    # PSP103/OSDI path ignores `.option scale` regardless). That gap is a property of the PDK's
    # render convention, not of any one topology, so it was NOT re-derived per template — every gf180
    # port here needed an override too, contradicting the rollout spec's Q2 prediction ("≥9/12 gf180
    # ports need ZERO overrides", written citing pre-ONE-CONVENTION I1 behavior) — see this module's
    # ONE CONVENTION note above (dated 2026-07-04, one day before the rollout spec's freeze) for why:
    # 0/6 zero-override, not >=5/6, because gf180's scale mechanism was fully retired, not merely
    # unneeded here. Each geometry override is the template DEFAULT numeric magnitude with a `u`
    # suffix added (no resizing) — verified live 2026-07-04/05 via `run_recipe()` against the real
    # IIC-OSIC-TOOLS ngspice (probe_i1a_port.py, scratchpad). 5 of 6 templates on both PDKs (10/12
    # ports) needed ONLY the geometry override: OP sane (no dead/railed bias, tail/bias currents
    # in-range for the VDD domain) and every seed-recipe claim VERIFIED on the FIRST probe.
    #
    # regulated_cascode_nmos NEEDED A SECOND OVERRIDE KIND — `VBA`, the auxiliary gain-boosting
    # amplifier's bias VOLTAGE (Mn's gate, fixed absolute reference) — the first sizing-override key
    # in this table that is NOT a device length: it must NEVER carry a `u` suffix (that would parse
    # as microvolts, not volts). ROOT CAUSE (probed): the template's default VBA=0.8 is a value
    # tuned for sky130's 1.8V rail; reused verbatim, `@node` voltage inspection (scratchpad/
    # probe_rgc_gf180_diag.py) showed gf180's aux-loop output node g1c pinned near VDD (v(g1c)=3.263V
    # of a 3.3V rail) — Mn (the aux stage's NMOS current source, gf180's higher-Vt nfet_03v3) is
    # starved of overdrive at Vgs=0.8V and can't sink even its small bias current, collapsing the
    # loop's regulation (rgc_iout measured CoV 6.0% vs the claim's 0.1% bound — REFUTED on the FIRST
    # probe). A VBA scan (scratchpad/probe_rgc_vba_scan.py, 4 points 1.2-2.0V) found VBA=1.8V restores
    # tight regulation (CoV ~0.013%, on par with sky130's own ~0.005-0.02% and PASSING the SAME 0.1%
    # bound) — gf180's 3.3V rail needs a proportionally higher aux bias, not a different topology.
    #
    # ihp-sg13g2's SAME claim REFUTES even after the identical fix pattern — a REAL, probe-confirmed
    # divergence, not a mis-port: naively reusing gf180's fixed VBA=1.8 (probe 1) on ihp's 1.5V rail
    # (VBA > VDD!) collapsed iout to ~40-60nA (a dead-ish, ~250x-low bias) — clearly wrong, so a
    # dedicated VBA scan (scratchpad/probe_rgc_vba_scan_ihp.py, 6 points 0.4-1.1V) + a WA (aux device
    # width) scan (4 points 4-32u, crossed with the best VBA candidates) were run to find ihp's true
    # optimum: CoV bottoms out at ~0.276-0.281% across the ENTIRE explored VBA/WA grid (best:
    # VBA=0.55V, WA=4u — no larger WA improves it, ruling out "aux amp too weak" as a fixable sizing
    # gap) — a hard floor ~2.8x the claim's 0.1% bound, so `rgc_iout` REFUTES on ihp-sg13g2 even at
    # its best found bias point. Read as `process_scoped`: the regulated-cascode's gain-boosting
    # margin is headroom-limited at ihp's 1.5V domain (vs sky130's 1.8V / gf180's 3.3V) exactly the
    # axis the rollout spec's Q1 pre-registered as the expected divergence candidate — this is that
    # candidate, on a DIFFERENT wave-A template than the spec's own guess (a headroom-hungry OTA
    # cascode), not a broken port (the OP itself is sane and in-range at every probed bias point;
    # only the tight cov_max=0.001 invariance bound fails). VBA=0.55 (not the wider-margin-looking but
    # WRONG-metric-derived 0.6) is kept as the override because it is the systematic-scan optimum
    # under the oracle's ACTUAL invariance metric ((max-min)/mean, oracle.py — NOT the naive
    # population-stdev/mean this probe first miscomputed by hand; the wired run_recipe()/oracle path
    # was what caught the discrepancy, re-confirming why probes must go through the real judge, not a
    # hand-rolled approximation of it).
    ("miller_ota_2stage_nmos_in", "gf180mcuD"):
        {"W1": "8u", "W3": "4u", "W5": "4u", "W6": "16u", "W7": "8u", "W8": "2u", "Lp": "0.5u"},
    ("miller_ota_2stage_nmos_in", "ihp-sg13g2"):
        {"W1": "8u", "W3": "4u", "W5": "4u", "W6": "16u", "W7": "8u", "W8": "2u", "Lp": "0.5u"},
    ("common_gate_nmos", "gf180mcuD"): {"Wn": "8u", "Lp": "0.5u"},
    ("common_gate_nmos", "ihp-sg13g2"): {"Wn": "8u", "Lp": "0.5u"},
    ("source_follower_nmos", "gf180mcuD"): {"Wn": "8u", "Lp": "0.5u"},
    ("source_follower_nmos", "ihp-sg13g2"): {"Wn": "8u", "Lp": "0.5u"},
    ("diff_pair_resistive_nmos", "gf180mcuD"): {"Wn": "8u", "Lp": "0.5u"},
    ("diff_pair_resistive_nmos", "ihp-sg13g2"): {"Wn": "8u", "Lp": "0.5u"},
    ("cascode_current_mirror_nmos", "gf180mcuD"): {"W": "4u", "Lp": "0.5u"},
    ("cascode_current_mirror_nmos", "ihp-sg13g2"): {"W": "4u", "Lp": "0.5u"},
    ("regulated_cascode_nmos", "gf180mcuD"): {"W": "4u", "WA": "4u", "Lp": "0.5u", "VBA": "1.8"},
    ("regulated_cascode_nmos", "ihp-sg13g2"): {"W": "4u", "WA": "4u", "Lp": "0.5u", "VBA": "0.55"},

    # I1b (spec docs/superpowers/specs/2026-07-05-full-registry-cross-pdk-rollout.md, wave B — 6
    # "headroom/switched/BJT" templates). PROBE METHOD (I1a discipline, live via run_recipe()/oracle,
    # scratchpad/probe_i1b_wave_b.py + hand-decks for the two divergent cases): 9/12 ports needed ONLY
    # the u-suffixed geometry override (same convention, same magnitude as every prior wave) and
    # VERIFIED every seed-recipe claim, matching sky130A. 3/12 did NOT: 1 genuine process_scoped
    # PORT-FAILED (telescopic_cascode_ota_nmos_in/ihp-sg13g2 — the rollout spec's own Q3 headroom
    # guess, confirmed) + 2 PORT-FAILED from a template-authoring gap unrelated to any PDK bias
    # (single_slope_ramp_generator, both PDKs — see below). Per-port detail:
    #
    # telescopic_cascode_ota_nmos_in: gf180mcuD (3.3V) VERIFIED both claims on the FIRST probe
    # (geometry-only). ihp-sg13g2 (1.5V) is PORT-FAILED, not REFUTED: this topology stacks FIVE
    # devices rail-to-rail on the signal path (M5 tail -> M1 input -> M1c NMOS cascode -> M3c PMOS
    # cascode -> M3 PMOS mirror) — tighter than every other wave-A/B port (folded, below, needs only
    # 4). A systematic scan (scratchpad/probe_i1b_wave_b.py + hand .op/.ac decks): VBNC 0.5-1.1V x
    # VBPC 0.2-0.9V (30-point grid) + a WN width axis (8u-64u, re-tested via FRESH separate
    # run_recipe() calls after discovering mid-scan that ngspice's `alter` does not reliably
    # propagate into a subckt W= parameter expression inside a `.control` loop — a real limitation,
    # not a probe error, recorded here so it isn't rediscovered) never once produced a positive,
    # CL-flat gain: every explored point measured NEGATIVE av0 (-9.7 to -33 dB, i.e. attenuation, not
    # amplification) and CL-DEPENDENT (not the flat single-pole response a working cascode gives).
    # Direct node-voltage query at the best-found point (WN=32u, VBNC=0.6, VBPC=0.2, av0=-9.8dB
    # still negative): v(o1)-v(n1) = 0.017V — the NMOS cascode M1c is pinned into deep triode (near-
    # zero Vds) at EVERY explored bias, i.e. genuinely starved, not merely sub-optimal. Read as
    # `process_scoped`: the telescopic cascode's 5-device stack does not fit ihp's 1.5V domain at
    # this recipe's fixed topology — the SAME axis the rollout spec's Q3 pre-registered as its
    # explicit guess, confirmed on the guessed template this time (contrast I1a's regulated_cascode,
    # which diverged on the spec's Q1 axis but a DIFFERENT template than either guess).
    #
    # folded_cascode_ota_nmos_in: VERIFIED both claims on BOTH pdks on the FIRST probe (geometry-only)
    # — gf180mcuD av0 71.3dB-class / ihp-sg13g2 av0=41.28dB flat (spread 0.0002, well inside the
    # 0.5dB bound). The fold un-stacks the input pair from the output cascode (max 4 devices on any
    # one signal path, one fewer than telescopic), exactly the headroom advantage the claim's own
    # narrative predicts — confirmed live, not just asserted, and the reason folded succeeds where
    # telescopic (same WN/WP/WT/Lp magnitudes, same ihp process) does not.
    #
    # cds_switched_cap_nmos: ihp-sg13g2 VERIFIED on the FIRST probe (geometry-only, Lp=0.15u — cov
    # 0.0%). gf180mcuD needed a SECOND probe: the naive "template default + u suffix" convention
    # (Lp=0.15u, sky130's near-minimum 1.8V-device length) hit a genuine gf180-specific floor —
    # `nfet_03v3`'s binned model sets `lmin=2.8e-07` (0.28um, verified: sm141064.spice line 806) for
    # this 3.3V THICK-oxide device, well above sky130's/ihp's shorter-channel minimums; Lp=0.15u is
    # BELOW it, and the fets_mm mismatch-sigma `.param` arithmetic (module docstring's earlier ground
    # truth) divides by an effective-length term that goes invalid there, giving `delvto`/`mulu0` =
    # nan and a hard "could not find a valid modelname" ngspice error (verified live). Lp=0.3u (a
    # round value with margin above the 0.28um floor, no other resizing) fixes it: vo held IDENTICAL
    # (cov 0) across the 0.5-1.3V pedestal sweep at 1.09331V, VERIFIED. This is a genuine, probe-
    # confirmed RESIZE (not just a suffix) — the one wave-B case besides regulated_cascode_nmos (I1a)
    # where the sky130-inherited magnitude itself, not merely its units, needed to change.
    #
    # single_slope_ramp_generator: [CLOSED 2026-07-05 post-I1b — see the LS entries below; original
    # transcript kept verbatim.] PORT-FAILED on BOTH pdks — NOT a bias/headroom divergence, an
    # INSTRUMENT gap this increment cannot fix within its own mandate (no template-body edits
    # permitted). `_RAMP_BODY`'s reset-switch instance (`Xrst vramp phirst 0 0 sky130_fd_pr__nfet_01v8
    # W={WS} L=0.15`) hardcodes its LENGTH as a bare, non-`.param` literal — unlike every other
    # geometry field in every template ported so far, `PDK_SIZING_OVERRIDES` cannot reach it (there is
    # no `{...}` placeholder for a sizing dict to fill). On gf180mcuD (no `.option scale`, per this
    # module's ground truth) the bare `0.15` resolves to 0.15 METERS -> a hard, fatal "could not find
    # a valid modelname" (verified live, identical failure mode to the historic W=4/L=0.5-bare-number
    # ihp findings above). On ihp-sg13g2 the same bare literal also resolves to 0.15 meters (PSP103
    # ignores `.option scale` too) but fails SILENTLY rather than fatally: the reset switch never
    # actually discharges `vramp` (op query: vramp settles near 1.6V, well above the 0.3V/1.0V measure
    # thresholds, and never crosses through them), so both `.meas` calls report "out of interval" and
    # the series comes back empty (correctly FLAGGED by the executor's series-validity guard, C1 —
    # never a fake pass). Root cause CONFIRMED on both PDKs by hand-patching the literal to `L=0.15u`
    # in a scratch copy of the rendered deck (scratchpad probe, not a shipped change): gf180 converges
    # immediately; ihp produces a clean, monotonic ramp (slope 2.03e7 V/s @10uA -> 1.57e8 V/s @80uA,
    # matching the claim's direction and ~2-4% linear in I, on par with sky130's own ~1%). This is
    # unambiguous evidence the topology itself is portable — the failure is a ONE-CHARACTER-CLASS
    # template-authoring gap (`L=0.15` should read `L={Lp}`, mirroring `cds_switched_cap_nmos`'s own
    # `Xsw ... L={Lp}` pattern one template up) OUTSIDE this increment's touch-scope (templates.py);
    # flagged for a follow-up increment rather than fixed here. No `PDK_SIZING_OVERRIDES` entry is
    # registered for either pdk — absence here means exactly what `sizing_overrides_for`'s docstring
    # already says it means ("not yet ported"), not a silently-accepted broken port.
    #
    # column_pga_inverting_nmos: VERIFIED on BOTH pdks on the FIRST probe (geometry-only) — the
    # closed-loop gain is set by the Rin/Rf resistor RATIO (unaffected by any PDK's device-geometry
    # unit convention), so only the OTA core's W/L needed the usual u-suffix treatment.
    #
    # ptat_ctat_core_bjt: BJT AVAILABLE ON BOTH PDKS — see this module's "BJT AVAILABILITY" ground-
    # truth section (near `PDKProfile.bjt_pnp`) for the full probe transcript; the rollout spec's Q2
    # guessed this template as the MOST LIKELY honest availability DROP, and that guess did not hold
    # (both PDKs genuinely ship a usable parasitic/substrate PNP) — reported as found, not forced
    # either way. VERIFIED on BOTH pdks on the FIRST probe once the bjt_pnp substitution + extra
    # `.lib` line were wired (geometry-only sizing beyond that — Wp/Lp, the same PMOS-mirror
    # convention as every other template).
    ("telescopic_cascode_ota_nmos_in", "gf180mcuD"): {"WN": "8u", "WP": "16u", "WT": "16u", "Lp": "0.5u"},
    ("telescopic_cascode_ota_nmos_in", "ihp-sg13g2"): {"WN": "8u", "WP": "16u", "WT": "16u", "Lp": "0.5u"},
        # PORT-FAILED even with this (universally-needed, every-other-class) geometry override applied
        # — kept registered because bare geometry is a hard, unrelated failure mode on ihp (see this
        # module's ground truth) that would otherwise mask the REAL finding. VBNC/VBPC are deliberately
        # left at the template's own sky130-inherited defaults (1.1V/0.3V): the systematic scan above
        # found no (VBNC, VBPC, WN) combination that fixes it, so no candidate is more "correct" than
        # the default — the failure is the headroom finding itself, not a specific wrong bias choice.
    ("folded_cascode_ota_nmos_in", "gf180mcuD"): {"WN": "8u", "WP": "16u", "WT": "16u", "Lp": "0.5u"},
    ("folded_cascode_ota_nmos_in", "ihp-sg13g2"): {"WN": "8u", "WP": "16u", "WT": "16u", "Lp": "0.5u"},
    ("cds_switched_cap_nmos", "gf180mcuD"): {"WSW": "4u", "Lp": "0.3u"},   # Lp bumped above nfet_03v3's
                                                                            # lmin=0.28um floor — see above.
    ("cds_switched_cap_nmos", "ihp-sg13g2"): {"WSW": "4u", "Lp": "0.15u"},
    # single_slope_ramp_generator: I1b recorded PORT-FAILED on both PDKs (hardcoded `L=0.15` switch
    # length, unreachable by overrides). 2026-07-05 post-I1b: the template gap was closed (`LS` became
    # a .param — templates.py DEFAULT_RAMP_SIZING note) and both ports re-run live: VERIFIED.
    # gf180's LS=0.3u sits above nfet_03v3's lmin=0.28um floor (same constraint as cds' Lp above).
    ("single_slope_ramp_generator", "gf180mcuD"): {"WP": "8u", "WS": "4u", "Lp": "0.5u", "LS": "0.3u"},
    ("single_slope_ramp_generator", "ihp-sg13g2"): {"WP": "8u", "WS": "4u", "Lp": "0.5u", "LS": "0.15u"},
    ("column_pga_inverting_nmos", "gf180mcuD"): {"W1": "8u", "W3": "8u", "W5": "8u", "W5b": "4u", "Lp": "0.5u"},
    ("column_pga_inverting_nmos", "ihp-sg13g2"): {"W1": "8u", "W3": "8u", "W5": "8u", "W5b": "4u", "Lp": "0.5u"},
    ("ptat_ctat_core_bjt", "gf180mcuD"): {"Wp": "8u", "Lp": "0.5u"},
    ("ptat_ctat_core_bjt", "ihp-sg13g2"): {"Wp": "8u", "Lp": "0.5u"},
}


def sizing_overrides_for(topology_class: str, pdk: str) -> dict[str, str]:
    """The PDK-specific sizing overlay for (topology_class, pdk), or {} if none registered — this wires
    the lookup into the render path per spec §5-I1 requirement 5. sky130A is the ONLY identity-by-
    absence profile (its "override" IS the template defaults already, per the PDK_SIZING_OVERRIDES
    module comment above); as of I2 BOTH non-sky130 profiles (gf180mcuD and ihp-sg13g2) carry an
    explicit u-suffixed entry for every ported topology_class — the 3 nominal E1 pilots plus the
    comparator class added in E1b, plus I1a's 6 wave-A ports (miller_ota_2stage_nmos_in,
    common_gate_nmos, source_follower_nmos, diff_pair_resistive_nmos, cascode_current_mirror_nmos,
    regulated_cascode_nmos — the last of which additionally carries a non-geometry `VBA` bias-voltage
    override, the first key in this table that is deliberately NOT u-suffixed) — so {} for either of
    those two now means "not yet ported", not "this PDK needs no override". I1b (wave B) recorded
    (single_slope_ramp_generator, EITHER pdk) as PORT-FAILED on a hardcoded, non-`.param` template
    literal; 2026-07-05 post-I1b that template gap was CLOSED (`LS` param) and both ramp ports now
    carry the usual entry and VERIFY — see the dated note at their entries.
    (telescopic_cascode_ota_nmos_in, "ihp-sg13g2") is subtly DIFFERENT: it DOES carry the usual
    geometry entry (bare numbers are a separate, unrelated hard failure on ihp that would otherwise
    mask the real finding) and is STILL genuinely PORT-FAILED — a non-empty entry here is not a
    success signal by itself; check the module docstring's ground truth / the increment's port table
    for the actual verdict. Every OTHER wave-B class (folded_cascode_ota_nmos_in, cds_switched_cap_nmos,
    column_pga_inverting_nmos, ptat_ctat_core_bjt) carries the usual explicit entry AND succeeds."""
    return dict(PDK_SIZING_OVERRIDES.get((topology_class, pdk), {}))
