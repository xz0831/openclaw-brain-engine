"""Tier-B Task 3: VerilogRunner — native iverilog/vvp, reusing parse_rdata."""
from __future__ import annotations

import shutil

import pytest

from openclaw_brain.knowledge.executable.runner import parse_rdata
from openclaw_brain.knowledge.executable.verilog_runner import VerilogRunner


def test_parse_reuses_rdata_format():
    # the digital runner emits the SAME RDATA <series> <x> <y> the ngspice runner does
    out = "RDATA cnt 0 1\nRDATA cnt 1 2\n"
    assert parse_rdata(out) == {"cnt": [(0.0, 1.0), (1.0, 2.0)]}


def test_available_reflects_iverilog_presence():
    assert VerilogRunner().available() == (shutil.which("iverilog") is not None and shutil.which("vvp") is not None)


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog not installed")
def test_runs_a_counter_testbench(tmp_path):
    tb = (
        "module tb;\n"
        "  integer i; reg [3:0] q;\n"
        "  initial begin\n"
        "    for (i=0;i<4;i=i+1) begin q=i+1; $display(\"RDATA cnt %0d %0d\", i, q); end\n"
        "    $finish;\n"
        "  end\n"
        "endmodule\n"
    )
    series = VerilogRunner(workdir=str(tmp_path)).measure(tb)
    assert series["cnt"] == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
