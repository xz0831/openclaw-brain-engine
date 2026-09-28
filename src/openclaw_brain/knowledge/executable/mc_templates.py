"""Monte-Carlo mismatch renderers (statistical / Pelgrom). Reuse the existing analog cells, swap the
`.lib` corner to `tt_mm`, and wrap a `.control` loop with `reset` per run (which re-samples the agauss
device statistics). Each run echoes one `RDATA vos <run> <value>` — the oracle sees an ordinary
(run, value) series. Validated against real sky130: OTA offset 3σ≈14mV, comparator column-FPN 3σ≈16.7mV.

NOTE on the import cycle: this module imports the cells from templates.py, and templates.py registers
these renderers at its BOTTOM (after the cells are defined) — the same bottom-import pattern the digital
block uses, so the cycle resolves.

SEEDED MC (spec E1b §0/§4-I1) — INSTRUMENT DEFECT FIXED HERE. Before this, these `.control` loops were
UNSEEDED: `reset` genuinely resamples ngspice's agauss() draws (verified: a 2-device sky130 tt_mm
current-mirror probe spread ~20% across 8 resets within one process), but that draw sequence differs
across SEPARATE ngspice invocations — the measured symptom was the Pelgrom slope drifting -0.353 ->
-0.359 between two production runs of the identical recipe. E1's verification standard (byte-
reproduction of every reported number) is impossible on an unseeded instrument.

Fix, VERIFIED LIVE 2026-07-04 against sky130 tt_mm (2-device current-mirror probe, mc_runs=6):
  - `setseed <N>` inside the `.control` loop, called BEFORE `reset` each iteration, makes the
    subsequent `agauss()` draw reproducible: two SEPARATE `ngspice -b` process invocations of the
    IDENTICAL seeded deck produced byte-identical RDATA at every iteration.
  - `setseed 0` is NOT a valid fixed seed — verified it behaves like "no seed override" (the mc=0
    iteration differed between the two runs while mc=1..5, seeded with nonzero values, matched
    exactly). So MC_SEED_BASE (and any per-area offset) MUST stay >= 1 for every run index.
  - The seed value must be computed via a `.control` `let` variable and substituted with `$&name`
    (e.g. `let seedval = 1042\n setseed $&seedval`) — `setseed 1000+mc` as bare inline text does NOT
    evaluate the arithmetic (ngspice does not expand a bare `.control`-scope variable name inside an
    arbitrary command's argument without the `$&` substitution prefix) and silently produces a
    constant (non-varying) seed every iteration instead.

BASE + per-area offset scheme: MC_SEED_BASE anchors every renderer's inner run-index loop
(`setseed = MC_SEED_BASE + seed_offset + mc`); the executor's Pelgrom outer loop passes a distinct
`seed_offset = area_index * MC_SEED_AREA_STRIDE` per area so no two decks in the SAME elasticity fit
ever replay the identical underlying agauss() draw sequence (spec §4-I1 requirement 4).
"""
from __future__ import annotations

# Cells are imported LAZILY inside each renderer (not at module level) to avoid an import cycle:
# templates.py registers these renderers at its bottom, so a module-level `from templates import …`
# here would deadlock when mc_templates is imported first.

MC_SEED_BASE = 1000            # fixed, documented base seed (spec §4-I1) — verified >=1 stays out of
                                # ngspice's "0 = no override" pitfall (see module docstring).
MC_SEED_AREA_STRIDE = 100_000  # per-Pelgrom-area seed offset (executor.py) — keeps every area's run-
                                # index range (0..mc_runs-1) from overlapping another area's.


def _seed_lines(seed_offset: int) -> str:
    """The two `.control` lines that seed ngspice's PRNG deterministically for run `mc` (spec
    §4-I1): compute the seed via a `let` variable (bare inline arithmetic in `setseed`'s argument is
    NOT evaluated — verified live, see module docstring) and pass it with the `$&` substitution."""
    base = MC_SEED_BASE + int(seed_offset)
    return f"    let seedval = {base} + mc\n    setseed $&seedval\n"


def _cell_body(cell_text: str) -> list[str]:
    return [l for l in cell_text.splitlines()
            if not l.startswith("* cell") and l.strip() != ".end"]


def render_ota5t_offset_mc(sizing: dict | None = None, corner: str = "tt_mm", knob: str = "vos",
                           metric: str = "vos_v", points=None, mc_runs: int = 200,
                           seed_offset: int = 0) -> str:
    from openclaw_brain.knowledge.executable.templates import render_ota_5t_cell
    # the cell is ALREADY DC-unity-gain (Lfb shorts out->vinn at DC); offset = V(out)-VCM via op.
    vcm = float((sizing or {}).get("VCM", 0.9))
    body = [(f'.lib "__LIBPATH__" {corner}' if l.startswith(".lib") else l)
            for l in _cell_body(render_ota_5t_cell(sizing))]
    ctrl = (f"let mc=0\n  dowhile mc < {int(mc_runs)}\n"
            f"{_seed_lines(seed_offset)}    reset\n    op\n"
            f"    let vos = v(out)-{vcm}\n    echo RDATA {knob} $&mc $&vos\n    let mc=mc+1\n  end")
    return "\n".join(body + [".control", ctrl, ".endc", ".end", ""])


def render_comparator_fpn_mc(sizing: dict | None = None, corner: str = "tt_mm", knob: str = "vos",
                             metric: str = "vos_v", points=None, mc_runs: int = 200,
                             seed_offset: int = 0) -> str:
    from openclaw_brain.knowledge.executable.templates import render_comparator_cell
    vref = float((sizing or {}).get("VREF", 0.9))
    body = []
    for l in _cell_body(render_comparator_cell(sizing)):
        if l.startswith(".lib"):
            l = f'.lib "__LIBPATH__" {corner}'
        if l.startswith("Vin vinp"):
            l = "Vin vinp 0 DC %.3f" % vref            # swept in .control
        body.append(l)
    ctrl = (f"let mc=0\n  dowhile mc < {int(mc_runs)}\n"
            f"{_seed_lines(seed_offset)}    reset\n"
            f"    dc Vin {vref - 0.06:.3f} {vref + 0.06:.3f} 0.0004\n"
            f"    meas dc vtrip when v(out)={vref} cross=1\n"
            f"    let vos = vtrip-{vref}\n    echo RDATA {knob} $&mc $&vos\n    let mc=mc+1\n  end")
    return "\n".join(body + [".control", ctrl, ".endc", ".end", ""])


def render_ota5t_gbw(sizing: dict | None = None, corner: str = "tt", knob: str = "gbw",
                     metric: str = "gbw_hz", points=None) -> str:
    """One GBW value per corner (the unity-gain crossover of the open-loop AC) — for the `corner` kind.
    The cell's Lfb/Cfb breaks the feedback at AC, so this measures the open-loop response."""
    from openclaw_brain.knowledge.executable.templates import render_ota_5t_cell
    body = [(f'.lib "__LIBPATH__" {corner}' if l.startswith(".lib") else l)
            for l in _cell_body(render_ota_5t_cell(sizing))]
    ctrl = (f"ac dec 40 1 10G\n  let g=0\n  meas ac g when vdb(out)=0 fall=1\n"
            f"  echo RDATA {knob} 0 $&g")
    return "\n".join(body + [".control", ctrl, ".endc", ".end", ""])
