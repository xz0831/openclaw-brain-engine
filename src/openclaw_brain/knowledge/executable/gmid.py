"""gm/ID sizing engine (SPEC §3 sizing seed) — attacks the "hardest part" (sizing).

CLEAN-ROOM by design: we characterize the sky130 devices with our OWN ngspice sweep
(not Murmann's AGPL .mat) — the data is fact derived from the Apache-2.0 sky130 PDK, and
self-generated LUTs carry better provenance (we record corner/VDS/VSB). See SPEC §14.

The gm/ID flow: pick a gm/ID target (the inversion-level knob) -> look up current density
ID/W -> set W from the required current; read gm/gds (intrinsic gain) and ft (bandwidth)
at that operating point. This turns sizing from per-topology hand-iteration into a table walk.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .runner import NgspiceRunner
from .templates import LIB_PLACEHOLDER

_DEVICE = {
    "nfet": "sky130_fd_pr__nfet_01v8",
    "pfet": "sky130_fd_pr__pfet_01v8",
}


@dataclass
class GmidPoint:
    vgs: float          # the swept gate drive (gate node voltage)
    idd: float          # device drain current (signed)
    gm: float
    gds: float
    cgg: float

    @property
    def gm_id(self) -> float:
        return abs(self.gm) / abs(self.idd)

    @property
    def gm_gds(self) -> float:                      # intrinsic gain
        return abs(self.gm) / abs(self.gds)

    @property
    def ft(self) -> float:                          # gm / (2*pi*Cgg)
        return abs(self.gm) / (2 * math.pi * abs(self.cgg))


class GmidTable:
    """A characterized device: query design quantities by gm/ID, and size a device."""

    def __init__(self, points: list[GmidPoint], w_char_um: float, l_um: float,
                 device: str, vds: float, corner: str = "tt"):
        # ascending in gm/ID (weak inversion = high gm/ID at low VGS)
        self.points = sorted(points, key=lambda p: p.gm_id)
        self.w_char_um = w_char_um
        self.l_um = l_um
        self.device = device
        self.vds = vds
        self.corner = corner

    def _interp(self, gm_id: float, fn) -> float:
        pts = self.points
        xs = [p.gm_id for p in pts]
        if gm_id <= xs[0]:
            return fn(pts[0])
        if gm_id >= xs[-1]:
            return fn(pts[-1])
        for i in range(len(xs) - 1):
            if xs[i] <= gm_id <= xs[i + 1]:
                t = (gm_id - xs[i]) / (xs[i + 1] - xs[i])
                return fn(pts[i]) + t * (fn(pts[i + 1]) - fn(pts[i]))
        return fn(pts[-1])

    def id_unit(self, gm_id: float) -> float:
        """Current (A) of the characterization UNIT device (W=w_char_um) at a gm/ID point."""
        return self._interp(gm_id, lambda p: abs(p.idd))

    def gm_gds(self, gm_id: float) -> float:
        return self._interp(gm_id, lambda p: p.gm_gds)

    def ft(self, gm_id: float) -> float:
        return self._interp(gm_id, lambda p: p.ft)

    def vgs_at(self, gm_id: float) -> float:
        return self._interp(gm_id, lambda p: p.vgs)

    def size(self, gm_id: float, id_target: float) -> float:
        """Return the MULTIPLICITY m (of the unit device W=w_char_um) carrying id_target at gm/ID.

        IMPORTANT (measured sky130 caveat): id is NOT ∝ W — id/W varies ~25% from W=10µm→3.3µm
        (foundry W-dependence), so the gm/ID "id ∝ W" assumption fails. But id IS ∝ m EXACTLY for
        m identical parallel unit devices (verified). So we size by multiplicity of a fixed unit W,
        not by continuous W. The device is built as `W={w_char_um} m={m}`."""
        return id_target / self.id_unit(gm_id)

    def total_width_um(self, gm_id: float, id_target: float) -> float:
        """Informational total width = m × w_char_um (the device is m units of w_char_um, not a slab)."""
        return self.size(gm_id, id_target) * self.w_char_um


_GMID_RE = re.compile(
    r"GMID vg=(\S+) id=(\S+) gm=(\S+) gds=(\S+) cgg=(\S+)")


def parse_gmid(output: str) -> list[GmidPoint]:
    pts = []
    for line in output.splitlines():
        m = _GMID_RE.search(line)
        if m:
            try:
                pts.append(GmidPoint(*(float(x) for x in m.groups())))
            except ValueError:
                continue
    return pts


def _deck(device: str, l_um: float, w_char_um: float, vds: float, vgs_points: list[float],
          corner: str) -> str:
    dev = _DEVICE[device]
    minst = f"m.xm1.m{dev}"
    cap = " ".join(f"{v:g}" for v in vgs_points)
    body = "\n".join([
        f'* clean-room gm/ID sweep: {device} L={l_um} VDS={vds} VSB=0',
        f'.lib "{LIB_PLACEHOLDER}" {corner}',
        f".param L={l_um} W={w_char_um}",
    ])
    if device == "nfet":
        body += f"\nXM1 d g 0 0 {dev} W={{W}} L={{L}}\nVd d 0 {vds}\nVg g 0 0.9"
    else:  # pfet: source/bulk at 1.8; drain VDS below source; gate swept down
        body += (f"\nVdd vdd 0 1.8\nXM1 d g vdd vdd {dev} W={{W}} L={{L}}"
                 f"\nVd d 0 {1.8 - vds:g}\nVg g 0 0.9")
    ctrl = "\n".join([
        ".control",
        f"foreach vg {cap}",
        "  alter Vg = $vg",
        "  op",
        f"  let id = @{minst}[id]",
        f"  let gm = @{minst}[gm]",
        f"  let gds = @{minst}[gds]",
        f"  let cgg = @{minst}[cgg]",
        "  echo GMID vg=$vg id=$&id gm=$&gm gds=$&gds cgg=$&cgg",
        "end",
        ".endc", ".end", "",
    ])
    return body + "\n" + ctrl


def characterize(runner: NgspiceRunner, device: str = "nfet", l_um: float = 0.5,
                 vds: float = 0.9, w_char_um: float = 10.0,
                 vgs_points: list[float] | None = None, corner: str = "tt",
                 timeout: int = 180) -> GmidTable:
    """Run a clean-room ngspice sweep and return a GmidTable for the device."""
    if vgs_points is None:
        if device == "nfet":
            vgs_points = [round(0.40 + 0.025 * i, 3) for i in range(0, 41)]   # 0.40..1.40
        else:
            vgs_points = [round(1.40 - 0.025 * i, 3) for i in range(0, 41)]   # 1.40..0.40 (|VGS| up)
    deck = _deck(device, l_um, w_char_um, vds, vgs_points, corner)
    out = runner.run_deck(deck, timeout=timeout)
    pts = parse_gmid(out)
    if len(pts) < 5:
        raise RuntimeError(f"gm/ID characterization produced too few points ({len(pts)})")
    return GmidTable(pts, w_char_um, l_um, device, vds, corner)
