"""ngspice runner — the ② executor's lab (SPEC §5). Runs a deck through ngspice in the
IIC-OSIC-TOOLS container (Colima) and parses the machine-readable `RDATA` lines into the
canonical (knob -> [(x, y), ...]) series the oracle consumes. Code form of the validated
experiments/executable_circuit_specimens/run.sh + the docker invocation + output parsing.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile

from .pdks import BJT_LIB_PLACEHOLDER, EXTRA_LIB_PLACEHOLDER, PDKProfile, get_profile

# Validated run.sh: locate the sky130 ngspice lib in-container, substitute it for the
# template's __LIBPATH__ placeholder, run ngspice in batch. ngspice is not on the default
# PATH in the image (it lives under /foss/tools/ngspice/bin).
#
# NOTE (verified 2026-06-29): the IIC-OSIC-TOOLS image already ships $HOME/.spiceinit
# (/headless/.spiceinit) with the robust sky130 settings the Ngspice-on-Colab harvest
# recommended — `set ngbehavior=hsa`, `set ng_nomodcheck`, PLUS `set enable_noisy_r`
# (required for sky130 .noise analyses). run.sh runs ngspice from /tmp (no local
# .spiceinit), so these $HOME settings apply automatically — DO NOT write a competing
# .spiceinit (it would shadow enable_noisy_r and break noise sims).
#
# This is the sky130A path ONLY (kept byte-identical — E1's regression guarantee). Other PDKs get a
# generated script from render_run_sh(); see pdks.py for the verified per-PDK facts it encodes.
RUN_SH = r"""#!/usr/bin/env bash
set -uo pipefail
DECK="${1:-/work/deck.spice}"
command -v ngspice >/dev/null 2>&1 || export PATH="$PATH:/foss/tools/ngspice/bin"
command -v ngspice >/dev/null 2>&1 || { echo "RUNNER_FAIL ngspice_missing"; exit 3; }
LIB=$(find "${PDK_ROOT:-/foss/pdks}" -path '*ngspice*' -name 'sky130.lib.spice' 2>/dev/null | head -1)
[ -z "$LIB" ] && { echo "RUNNER_FAIL sky130_lib_not_found"; exit 4; }
OUT="/tmp/$(basename "$DECK").run.spice"
sed "s|__LIBPATH__|$LIB|" "$DECK" > "$OUT"
ngspice -b "$OUT" 2>&1
"""


def render_run_sh(profile: PDKProfile) -> str:
    """The run.sh for `profile`. sky130A returns the byte-identical validated RUN_SH above (the
    regression guarantee); other profiles are generated from verified per-PDK facts (pdks.py's
    module docstring has the docker-run transcripts each field encodes):

      - `lib_glob` — the filename `find` searches for under PDK_ROOT (parameterizes the one
        sky130-hardcoded `find` in RUN_SH).
      - `extra_include_glob` — gf180mcuD needs `design.ngspice`'s globals defined before its
        `.lib ... typical` call resolves; RUN_SH substitutes it for the deck's
        `__EXTRA_LIBPATH__` placeholder (which pdks.substitute_devices only emits when this is set).
      - `needs_osdi` — ihp-sg13g2's OSDI/PSP103 devices need each `osdi/*.osdi` plugin loaded via a
        LOCAL `./.spiceinit` (verified: ngspice auto-sources one from its cwd, SHADOWING $HOME's — home settings are merged into the generated file) before the netlist
        parses; RUN_SH writes that file next to the deck and `cd`s there before invoking ngspice.
    """
    if profile.pdk == "sky130A":
        return RUN_SH

    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        'DECK="${1:-/work/deck.spice}"',
        'command -v ngspice >/dev/null 2>&1 || export PATH="$PATH:/foss/tools/ngspice/bin"',
        'command -v ngspice >/dev/null 2>&1 || { echo "RUNNER_FAIL ngspice_missing"; exit 3; }',
        f'PDKDIR="${{PDK_ROOT:-/foss/pdks}}/{profile.pdk_dir or profile.pdk}"',
        f"LIB=$(find -L \"$PDKDIR\" -path '*ngspice*' -name '{profile.lib_glob}' "
        "2>/dev/null | head -1)",
        '[ -z "$LIB" ] && { echo "RUNNER_FAIL lib_not_found"; exit 4; }',
    ]
    sed_exprs = ["s|__LIBPATH__|$LIB|"]
    if profile.extra_include_glob:
        lines += [
            f"EXTRALIB=$(find -L \"$PDKDIR\" -path '*ngspice*' "
            f"-name '{profile.extra_include_glob}' 2>/dev/null | head -1)",
            '[ -z "$EXTRALIB" ] && { echo "RUNNER_FAIL extra_lib_not_found"; exit 5; }',
        ]
        sed_exprs.append(f"s|{EXTRA_LIB_PLACEHOLDER}|$EXTRALIB|")
    if profile.bjt_lib_glob:
        # I1b: a BJT-bearing deck's second `.lib` call, ONLY when that section lives in a file
        # DIFFERENT from `lib_glob` (ihp-sg13g2's cornerHBT.lib — see pdks.py ground truth). gf180mcuD
        # reuses the already-resolved `$LIB`/__LIBPATH__ token instead (bjt_lib_glob stays None there).
        lines += [
            f"BJTLIB=$(find -L \"$PDKDIR\" -path '*ngspice*' "
            f"-name '{profile.bjt_lib_glob}' 2>/dev/null | head -1)",
            '[ -z "$BJTLIB" ] && { echo "RUNNER_FAIL bjt_lib_not_found"; exit 6; }',
        ]
        sed_exprs.append(f"s|{BJT_LIB_PLACEHOLDER}|$BJTLIB|")
    lines.append('OUT="/tmp/$(basename "$DECK").run.spice"')
    lines.append(f'sed "{"; ".join(sed_exprs)}" "$DECK" > "$OUT"')

    if profile.needs_osdi:
        lines += [
            # NOTE: /foss/pdks/<pdk> entries are ciel-managed SYMLINKS for gf180mcuD/sky130A
            # (ihp is a real dir) — plain `find` does not descend a symlinked root, so every
            # anchored find here uses -L (verified: without -L, gf180's lib is invisible).
            # verified: ihp's ngspice *.osdi plugins live under libs.tech/ngspice/osdi/ — the find
            # is anchored to THIS profile's pdk_dir AND to */ngspice/osdi/* so it can never load the
            # vacask-built .osdi copies or a sibling PDK's plugins (cornerMOSlv.lib exists in two
            # PDKs). ngspice auto-sources a LOCAL ./.spiceinit at startup — which SHADOWS
            # $HOME/.spiceinit entirely (verified: with a local file, only local settings apply) —
            # so the home settings are prepended into the generated file rather than lost. This is
            # the only verified-working point to `osdi`-load a Verilog-A model (a `.control` block
            # inside the deck itself runs too late, after netlist parsing has already started).
            'SPINIT="$(dirname "$OUT")/.spiceinit"',
            'cat "$HOME/.spiceinit" > "$SPINIT" 2>/dev/null || : > "$SPINIT"',
            'find -L "$PDKDIR" -path \'*/ngspice/osdi/*\' -name \'*.osdi\' 2>/dev/null '
            '| while read -r f; do echo "osdi \'$f\'"; done >> "$SPINIT"',
            'cd "$(dirname "$OUT")" && ngspice -b "$(basename "$OUT")" 2>&1',
        ]
    else:
        lines.append('ngspice -b "$OUT" 2>&1')

    return "\n".join(lines) + "\n"


DEFAULT_IMAGE = "hpretl/iic-osic-tools:latest"
_UNIT = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "K": 1e3, "x": 1e6, "g": 1e9}


def parse_unit(tok: str) -> float:
    """SPICE-style suffixed number -> float ('250f' -> 2.5e-13, '8k' -> 8000, '1.8' -> 1.8)."""
    tok = tok.strip()
    if tok and tok[-1] in _UNIT and not tok[-1].isdigit():
        try:
            return float(tok[:-1]) * _UNIT[tok[-1]]
        except ValueError:
            pass
    return float(tok)


def parse_rdata(output: str) -> dict[str, list[tuple[float, float]]]:
    """Parse `RDATA <series> <x_raw> <y>` lines into sorted (x, y) series."""
    series: dict[str, list[tuple[float, float]]] = {}
    for line in output.splitlines():
        m = re.match(r"\s*RDATA\s+(\S+)\s+(\S+)\s+(\S+)\s*$", line)
        if not m:
            continue
        key, xraw, yraw = m.group(1), m.group(2), m.group(3)
        try:
            y = float(yraw)
        except ValueError:
            continue
        # Drop non-finite points: a failed ngspice `.meas` (no crossing / non-convergence) leaves the
        # result vector at a reset sentinel that echoes as nan/inf or an empty token — never certify it.
        if not math.isfinite(y):
            continue
        series.setdefault(key, []).append((parse_unit(xraw), y))
    for k in series:
        series[k].sort()
    return series


class NgspiceRunner:
    """Runs a deck via ngspice in the IIC-OSIC-TOOLS container and returns canonical data."""

    def __init__(self, image: str = DEFAULT_IMAGE, workdir: str | None = None,
                 docker: str = "docker", *, egress: str | None = None):
        self.image = image
        self.docker = docker
        self.egress = egress
        # $HOME is mounted into the Colima VM (validated); /private/tmp is NOT.
        if workdir is None:
            from openclaw_brain.config import brain_state_home
            workdir = str(brain_state_home() / "sim")
        self.workdir = workdir

    def available(self) -> bool:
        """True iff the docker CLI works and the image is present locally (gates integration tests)."""
        try:
            self._check_daemon()
            r = subprocess.run([self.docker, "images", "-q", self.image],
                               capture_output=True, text=True, timeout=30)
            return r.returncode == 0 and bool(r.stdout.strip())
        except Exception:
            return False

    def _check_daemon(self) -> None:
        from openclaw_brain.egress import LOCAL_ONLY, effective_egress, EgressPolicyError
        if (self.egress or effective_egress()) != LOCAL_ONLY:
            return
        host = os.environ.get("DOCKER_HOST", "")
        if os.environ.get("DOCKER_CONTEXT") or not host.startswith("unix://") or not host[7:].startswith("/"):
            raise EgressPolicyError("local-only Docker requires explicit unix:// DOCKER_HOST and no DOCKER_CONTEXT")

    def _workdir_for(self, profile: PDKProfile) -> str:
        """Per-PDK BASE sim directory (§5-I1 item 3) — the namespace `run_deck` allocates each
        call's own ISOLATED subdirectory under (see `run_deck`); no longer written to directly.
        sky130A keeps the exact base path (self.workdir, no `/<pdk>` segment) it always has —
        byte-compatible at the base-path level; other profiles get a `<workdir>/<pdk>` base so a
        cross-PDK run (E1's replication script) never clobbers another profile's deck/run.sh/
        .spiceinit even before per-call isolation is applied on top."""
        if profile.pdk == "sky130A":
            return self.workdir
        return os.path.join(self.workdir, profile.pdk)

    def run_deck(self, deck: str, pdk: str = "sky130A", timeout: int = 300) -> str:
        """Write the deck + run.sh into a FRESH, per-call isolated subdirectory under the (per-PDK)
        mounted workdir, run ngspice in the container, return combined stdout/stderr. `pdk` selects
        the PDKProfile (default "sky130A" — every existing caller is unaffected); raises UnknownPDK
        before touching the filesystem for an unrecognized pdk.

        Per-call isolation (`tempfile.mkdtemp`, nested under `_workdir_for`'s per-PDK base — NEVER
        the bare system tmpdir, since only $HOME is mounted into the Colima VM, not /private/tmp)
        closes the sky130A same-PDK concurrent-run race documented in EPISTEMOLOGY.md's Known risk
        areas #3: two overlapping `run_deck` calls used to share one `deck.spice`/`run.sh` for the
        sky130A profile (no isolation at all) — whichever call's docker container actually read the
        file could silently pick up the OTHER call's netlist. Every profile (not just sky130A) now
        gets its own directory per call; no cleanup/retention behavior changes — exactly as before,
        the directory (and its deck.spice/run.sh) is left on disk after the run for post-hoc
        debugging, nothing new is deleted."""
        self._check_daemon()
        profile = get_profile(pdk)
        base = self._workdir_for(profile)
        os.makedirs(base, exist_ok=True)
        workdir = tempfile.mkdtemp(prefix="run-", dir=base)
        deck_path = os.path.join(workdir, "deck.spice")
        runsh_path = os.path.join(workdir, "run.sh")
        with open(deck_path, "w") as f:
            f.write(deck)
        with open(runsh_path, "w") as f:
            f.write(render_run_sh(profile))
        os.chmod(runsh_path, 0o755)
        cmd = [self.docker, "run", "--rm", "--entrypoint", "bash",
               "-v", f"{workdir}:/work", self.image, "/work/run.sh", "/work/deck.spice"]
        from openclaw_brain.egress import LOCAL_ONLY, effective_egress
        if (self.egress or effective_egress()) == LOCAL_ONLY:
            cmd[2:2] = ["--network", "none"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or "") + (r.stderr or "")
        if "RUNNER_FAIL" in out:
            raise RuntimeError(f"ngspice runner failed: {out.splitlines()[-1] if out else 'no output'}")
        return out

    def measure(self, deck: str, pdk: str = "sky130A", timeout: int = 300) -> dict[str, list[tuple[float, float]]]:
        """Run a deck and parse its RDATA into canonical series for the oracle. `pdk` defaults to
        "sky130A" (byte-identical to pre-E1 behavior) — pass a PDK_PROFILES key to run under a
        different foundry model card (the deck must already be substituted via
        pdks.substitute_devices for that profile; the runner does not substitute)."""
        return parse_rdata(self.run_deck(deck, pdk=pdk, timeout=timeout))
