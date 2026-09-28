"""iverilog/vvp runner — the digital engine's lab (Tier-B §2). Writes a testbench, compiles with
iverilog, runs with vvp, and parses the SAME `RDATA <series> <x> <y>` lines the ngspice runner emits
(`runner.parse_rdata`). Native (no docker): iverilog 13.0 is an Apple-Silicon brew binary.

The render/run split (ADR Decision 1) keeps all Verilog-specific knowledge in the renderer (the deck is
opaque here); this runner only compiles+runs+parses, exactly mirroring NgspiceRunner.measure's contract.
"""
from __future__ import annotations

import os
import shutil
import subprocess

from openclaw_brain.knowledge.executable.runner import parse_rdata


class VerilogRunner:
    """Compiles a self-contained Verilog testbench with iverilog, runs it with vvp, parses RDATA."""

    def __init__(self, workdir: str | None = None):
        # $HOME-rooted like NgspiceRunner.workdir; native run needs no mount.
        if workdir is None:
            from openclaw_brain.config import brain_state_home
            workdir = str(brain_state_home() / "vlog")
        self.workdir = workdir

    def available(self) -> bool:
        """True iff both iverilog and vvp are on PATH (gates the integration tests)."""
        return shutil.which("iverilog") is not None and shutil.which("vvp") is not None

    def measure(self, testbench: str, timeout: int = 120) -> dict[str, list[tuple[float, float]]]:
        """Write tb.v, compile (-g2012), run, and parse RDATA into canonical (knob -> [(x, y)]) series."""
        os.makedirs(self.workdir, exist_ok=True)
        tb = os.path.join(self.workdir, "tb.v")
        vvp_out = os.path.join(self.workdir, "tb.vvp")
        with open(tb, "w") as f:
            f.write(testbench)
        comp = subprocess.run(
            ["iverilog", "-g2012", "-o", vvp_out, tb],
            capture_output=True, text=True, timeout=timeout,
        )
        if comp.returncode != 0:
            # A compile failure is a REAL signal (e.g. a deliberately-bugged RTL that won't elaborate);
            # surface it so the executor's guard turns it into FLAGGED, never a silent VERIFIED.
            raise RuntimeError(f"iverilog compile failed: {(comp.stderr or comp.stdout)[-400:]}")
        run = subprocess.run(["vvp", vvp_out], capture_output=True, text=True, timeout=timeout)
        return parse_rdata((run.stdout or "") + (run.stderr or ""))
