"""Verilog renderers for the iverilog engine (Tier-B §3). Each `render_*_tran` returns a SELF-CONTAINED
testbench (DUT module + tb) that `$display`s `RDATA <knob> <x> <y>` — byte-compatible with the ngspice
runner's parser. Each `render_*_cell` returns just the DUT module source (the Specimen.netlist SSOT).

digital_cds / gray_code_counter are explicitly PEDAGOGY demos (Tier-B §3b): near-tautological, they
teach the artifact-agnostic claim-card and the per-engine basis distinction — they are NOT the evidence
the tier is non-vacuous (that is ss_adc_digital_backend, §3a). The renderers accept the same
`(sizing, corner=, knob=, metric=, points=)` signature the analog renderers use; `corner` is accepted
and ignored (a functional sim has no PVT corner).
"""
from __future__ import annotations


# ---------------------------------------------------------------------------------------------------
# digital_cds — pedestal-invariant difference (the digital analogue of switched-cap CDS)
# ---------------------------------------------------------------------------------------------------
def render_digital_cds_cell(sizing: dict | None = None, corner: str = "n/a") -> str:
    w = (sizing or {}).get("W", 8)
    return (
        f"module digital_cds #(parameter W={w}) "
        f"(input [W-1:0] rst, input [W-1:0] sig, output [W-1:0] diff);\n"
        f"  assign diff = sig - rst;   // CDS: subtract the reset sample from the signal sample\n"
        f"endmodule\n"
    )


def render_digital_cds_tran(
    sizing: dict | None = None, corner: str = "n/a", knob: str = "ped",
    metric: str = "diff_lsb", points: list[str] | None = None,
) -> str:
    w = (sizing or {}).get("W", 8)
    delta = (sizing or {}).get("DELTA", 16)
    pts = points or ["0", "64", "128", "192"]
    lines = "\n".join(
        f"    ped = {int(p)}; rst = ped + 8'd32; sig = ped + 8'd32 + 8'd{delta}; #1;\n"
        f"    $display(\"RDATA {knob} %0d %0d\", ped, diff);"
        for p in pts
    )
    return (
        render_digital_cds_cell(sizing, corner) + "\n"
        f"module tb;\n"
        f"  reg [{w - 1}:0] rst, sig; integer ped;\n"
        f"  wire [{w - 1}:0] diff;\n"
        f"  digital_cds #(.W({w})) dut(.rst(rst), .sig(sig), .diff(diff));\n"
        f"  initial begin\n{lines}\n    $finish;\n  end\n"
        f"endmodule\n"
    )


# ---------------------------------------------------------------------------------------------------
# gray_code_counter — consecutive Gray codes differ by exactly one bit (Hamming-1 invariance)
# ---------------------------------------------------------------------------------------------------
def render_gray_counter_cell(sizing: dict | None = None, corner: str = "n/a") -> str:
    w = (sizing or {}).get("W", 4)
    return (
        f"module bin2gray #(parameter W={w}) (input [W-1:0] b, output [W-1:0] g);\n"
        f"  assign g = b ^ (b >> 1);   // reflected binary -> Gray\n"
        f"endmodule\n"
    )


def render_gray_counter_tran(
    sizing: dict | None = None, corner: str = "n/a", knob: str = "idx",
    metric: str = "hamming", points: list[str] | None = None,
) -> str:
    w = (sizing or {}).get("W", 4)
    n = 1 << w
    return (
        render_gray_counter_cell(sizing, corner) + "\n"
        f"module tb;\n"
        f"  integer i, j, x, h;\n"
        f"  reg [{w - 1}:0] b, gprev;\n"
        f"  wire [{w - 1}:0] g;\n"
        f"  bin2gray #(.W({w})) dut(.b(b), .g(g));\n"
        f"  initial begin\n"
        f"    gprev = 0; b = 0; #1;\n"
        f"    for (i=0;i<{n};i=i+1) begin\n"
        f"      b = i; #1;\n"
        f"      x = g ^ gprev; h = 0;\n"
        f"      for (j=0;j<{w};j=j+1) h = h + ((x>>j) & 1);\n"
        f"      if (i>0) $display(\"RDATA {knob} %0d %0d\", i, h);\n"
        f"      gprev = g;\n"
        f"    end\n    $finish;\n  end\n"
        f"endmodule\n"
    )


# ---------------------------------------------------------------------------------------------------
# ss_adc_digital_backend — the NON-VACUOUS headline (Tier-B §3a): gray counter + latch + gray->binary
# + digital CDS, end-to-end. The oracle's ground truth is SPEC-DERIVED (the textbook XOR-cascade
# reference, the monotonicity property, the integer-difference identity) computed INDEPENDENTLY in the
# testbench — never author-supplied expected vectors. `bugged=True` injects a wrong gray->binary decoder
# (identity instead of XOR-reduction): the g2b_match assertion then REFUTES, proving the oracle catches a
# real functional bug rather than rubber-stamping a tautology.
# ---------------------------------------------------------------------------------------------------
def _gray2bin_module(w: int, bugged: bool) -> str:
    if bugged:
        # WRONG: pass the Gray code through unchanged (no XOR-cascade decode). A genuine RTL bug.
        body = f"  genvar k;\n  generate for (k=0;k<{w};k=k+1) begin: gb assign b[k] = g[k]; end endgenerate\n"
    else:
        # CORRECT: b[k] = XOR-reduction of g[W-1..k]  ==  textbook gray->binary.
        body = f"  genvar k;\n  generate for (k=0;k<{w};k=k+1) begin: gb assign b[k] = ^(g >> k); end endgenerate\n"
    return f"module gray2bin #(parameter W={w}) (input [W-1:0] g, output [W-1:0] b);\n{body}endmodule\n"


def _ss_adc_module(w: int, bugged: bool) -> str:
    return (
        _gray2bin_module(w, bugged) + "\n"
        f"module ss_adc_backend #(parameter W={w})\n"
        f"  (input clk, input rst_n, input trip, output [W-1:0] code_out);\n"
        f"  reg [W-1:0] bin;\n"
        f"  wire [W-1:0] gray = bin ^ (bin >> 1);\n"
        f"  reg [W-1:0] gray_lat;\n"
        f"  always @(posedge clk or negedge rst_n)\n"
        f"    if (!rst_n) bin <= 0; else bin <= bin + 1'b1;   // single-slope up-counter\n"
        f"  always @(posedge clk) if (trip) gray_lat <= gray; // latch the Gray code at comparator trip\n"
        f"  gray2bin #(.W(W)) dec(.g(gray_lat), .b(code_out)); // decode latched Gray -> binary\n"
        f"endmodule\n"
    )


def render_ss_adc_backend_cell(sizing: dict | None = None, corner: str = "n/a", bugged: bool = False) -> str:
    w = (sizing or {}).get("W", 8)
    return _ss_adc_module(w, bugged)


def render_ss_adc_backend_tran(
    sizing: dict | None = None, corner: str = "n/a", knob: str = "code",
    metric: str = "g2b_match", points: list[str] | None = None, bugged: bool = False,
) -> str:
    w = (sizing or {}).get("W", 8)
    n = 1 << w

    if metric == "g2b_match":
        # Decoder bijection: DUT gray2bin output == textbook XOR-cascade reference, over the FULL range.
        return (
            _gray2bin_module(w, bugged) + "\n"
            f"module tb;\n"
            f"  reg [{w - 1}:0] g; wire [{w - 1}:0] b_dut; integer i;\n"
            f"  gray2bin #(.W({w})) dut(.g(g), .b(b_dut));\n"
            f"  function [{w - 1}:0] ref_g2b; input [{w - 1}:0] gg; integer k; reg [{w - 1}:0] bb; begin\n"
            f"    bb[{w - 1}] = gg[{w - 1}];\n"
            f"    for (k={w - 2};k>=0;k=k-1) bb[k] = bb[k+1] ^ gg[k];\n"
            f"    ref_g2b = bb; end endfunction\n"
            f"  initial begin\n"
            f"    for (i=0;i<{n};i=i+1) begin\n"
            f"      g = i ^ (i>>1); #1;\n"
            f"      $display(\"RDATA {knob} %0d %0d\", i, (b_dut == ref_g2b(g)) ? 1 : 0);\n"
            f"    end\n    $finish;\n  end\n"
            f"endmodule\n"
        )

    if metric == "code":
        # Monotonicity: the latched+decoded code is non-decreasing in the comparator-trip time.
        pts = points or ["10", "50", "100", "200"]
        runs = "\n".join(f"    run_trip({int(p)});" for p in pts)
        return (
            _ss_adc_module(w, bugged) + "\n"
            f"module tb;\n"
            f"  reg clk, rst_n, trip; wire [{w - 1}:0] code_out; integer cyc;\n"
            f"  ss_adc_backend #(.W({w})) dut(.clk(clk), .rst_n(rst_n), .trip(trip), .code_out(code_out));\n"
            f"  initial clk = 0;\n  always #5 clk = ~clk;\n"
            f"  task run_trip(input integer tp); begin\n"
            f"    trip = 0; rst_n = 1; @(negedge clk);\n"
            f"    rst_n = 0; @(negedge clk);\n"       # clean 1->0 negedge fires the async reset (bin<=0)
            f"    rst_n = 1; @(posedge clk);\n"
            f"    for (cyc=0;cyc<tp;cyc=cyc+1) @(posedge clk);\n"
            f"    trip = 1; @(posedge clk); trip = 0; @(posedge clk);\n"
            f"    $display(\"RDATA {knob} %0d %0d\", tp, code_out);\n"
            f"  end endtask\n"
            f"  initial begin\n{runs}\n    $finish;\n  end\n"
            f"endmodule\n"
        )

    if metric == "diff_match":
        # Full-range incl. overflow: the W-bit (sig - rst) equals the integer-difference identity DELTA
        # (mod 2^W) even where rst/sig individually WRAP. The reference is the identity, computed in tb.
        delta = (sizing or {}).get("DELTA", 16)
        ref = (sizing or {}).get("REF", 200)
        pts = points or [str(p) for p in range(0, n, max(1, n // 8))]
        mask = n - 1
        lines = "\n".join(
            f"    ped = {int(p)}; rst = (ped + {ref}) & {mask}; sig = (ped + {ref} + {delta}) & {mask}; #1;\n"
            f"    $display(\"RDATA {knob} %0d %0d\", ped, (((sig - rst) & {mask}) == ({delta} & {mask})) ? 1 : 0);"
            for p in pts
        )
        return (
            render_digital_cds_cell(sizing, corner) + "\n"
            f"module tb;\n"
            f"  reg [{w - 1}:0] rst, sig; integer ped;\n"
            f"  initial begin\n{lines}\n    $finish;\n  end\n"
            f"endmodule\n"
        )

    raise ValueError(f"ss_adc_backend has no testbench for metric {metric!r}")
