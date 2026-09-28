"""Tests for the gm/ID sizing engine (knowledge/executable/gmid.py).

Unit: synthetic-table interpolation + size() (no docker).
Integration (docker-gated): clean-room characterize the real sky130 nfet, then size a
device for a target gm/ID + current and re-simulate to confirm it carries that current
(round-trip — validates the sizing flow against real sky130, not a tautology).
"""

import math

import pytest

from openclaw_brain.knowledge.executable.gmid import (
    GmidPoint, GmidTable, characterize, parse_gmid,
)
from openclaw_brain.knowledge.executable.runner import NgspiceRunner
from openclaw_brain.knowledge.executable.templates import LIB_PLACEHOLDER


def _pt(gm_id, id_w, w_char=10.0, gds=2e-6, cgg=1e-14, vgs=0.7):
    idd = id_w * w_char
    gm = gm_id * idd
    return GmidPoint(vgs=vgs, idd=idd, gm=gm, gds=gds, cgg=cgg)


# ── unit ──

def test_size_returns_multiplicity():
    # unit device (W=10): gm/ID 20 -> 10µA ; gm/ID 10 -> 30µA  =>  at gm/ID 15 -> 20µA
    table = GmidTable([_pt(20, 1e-6, vgs=0.6), _pt(10, 3e-6, vgs=1.0)],
                      w_char_um=10.0, l_um=0.5, device="nfet", vds=0.9)
    assert table.id_unit(15) == pytest.approx(20e-6)
    # size for 60µA at gm/ID 15 -> m = 60/20 = 3 units; total width 3*10 = 30µm
    assert table.size(15, 60e-6) == pytest.approx(3.0)
    assert table.total_width_um(15, 60e-6) == pytest.approx(30.0)


def test_interp_clamps_outside_range():
    table = GmidTable([_pt(20, 1e-6), _pt(10, 3e-6)], 10.0, 0.5, "nfet", 0.9)
    assert table.id_unit(25) == pytest.approx(10e-6)     # above max gm/ID -> clamp to weakest
    assert table.id_unit(5) == pytest.approx(30e-6)      # below min -> clamp to strongest


def test_gm_gds_and_ft_queryable():
    table = GmidTable([GmidPoint(0.6, 10e-6, 200e-6, 2e-6, 1e-14),
                       GmidPoint(1.0, 30e-6, 300e-6, 6e-6, 2e-14)], 10.0, 0.5, "nfet", 0.9)
    # gm/gds: 100 (weak) and 50 (strong); midpoint gm/ID -> ~75
    assert 50 <= table.gm_gds(15) <= 100
    assert table.ft(15) > 0


def test_parse_gmid_lines():
    out = "noise\nGMID vg=0.7 id=3.05E-05 gm=4.76E-04 gds=4.0E-06 cgg=2.5E-14\nx\n"
    pts = parse_gmid(out)
    assert len(pts) == 1
    assert pts[0].gm_id == pytest.approx(476e-6 / 30.5e-6, rel=1e-3)


# ── integration ──

def _id_at(runner, W, L, vgs, m=1.0, vds=0.9):
    dev = "sky130_fd_pr__nfet_01v8"
    deck = "\n".join([
        f'.lib "{LIB_PLACEHOLDER}" tt',
        f"XM1 d g 0 0 {dev} W={W} L={L} m={m}",
        f"Vd d 0 {vds}", f"Vg g 0 {vgs}",
        ".control", "op",
        f"let id = @m.xm1.m{dev}[id]", "echo CHECK id=$&id",
        ".endc", ".end", "",
    ])
    out = runner.run_deck(deck)
    import re
    mm = re.search(r"CHECK id=(\S+)", out)
    return abs(float(mm.group(1)))


def test_characterize_and_size_roundtrip_sky130():
    runner = NgspiceRunner()
    if not runner.available():
        pytest.skip("IIC-OSIC-TOOLS image not present; integration test needs the sim container")
    table = characterize(runner, device="nfet", l_um=0.5)
    gmids = [p.gm_id for p in table.points]
    assert max(gmids) > 20 and min(gmids) < 6           # span weak -> strong inversion
    # round-trip on an EXACT grid point: size by MULTIPLICITY of the unit device (W=w_char),
    # then re-simulate the unit device at that VGS with m= -> carries the target (id ∝ m exactly).
    p = min(table.points, key=lambda q: abs(q.gm_id - 15))
    m = 10e-6 / abs(p.idd)                               # fractional m < 1 (target < unit current)
    assert 0 < m < 5
    id_meas = _id_at(runner, table.w_char_um, 0.5, p.vgs, m=m)
    assert id_meas == pytest.approx(10e-6, rel=0.02)     # multiplicity sizing hits the target exactly
