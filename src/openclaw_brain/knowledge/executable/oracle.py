"""Functional-correctness oracle for the executable-circuit substrate (v2).

Deterministic claim-falsification (no model) — the code form of the validated
experiments/executable_circuit_specimens/oracle_zero.py. The oracle:
  1. RE-DERIVES every claim's verdict from canonical sim data with its own testbench
     (never trusts a self-reported number) — catches direction (Type-1) and value/TB
     (Type-2) errors.
  2. Certifies only the QUANT assertion (direction / magnitude / value / invariance).
  3. Enforces two structural rules (SPEC §4, measured leak-rate 1/3 naive -> 0/3):
       - mechanism_never_fact: a mechanism narrative is NEVER an oracle 'fact' — sign
         tests cannot refute a wrong cause that predicts the same numbers (Type-3).
       - reject_conditionless_measured: a measured value without complete R1 conditions
         is not eligible to become fact.
See docs/specs/SPEC_EXECUTABLE_CIRCUIT_SUBSTRATE.md §4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist, pstdev


def _probit_k(n_col: int) -> float:
    """Expected worst-of-N extreme in σ units: Φ⁻¹(1 − 1/N). For an N_col-column array this is the
    multiplier on σ that gives the WORST column's offset — what column FPN actually is."""
    return NormalDist().inv_cdf(1.0 - 1.0 / max(2, n_col))

from .models import ClaimCard, Conditions, QuantTest, VerdictClass

Series = list[tuple[float, float]]

# Verdicts that certify the quant assertion (eligible to back a taught 'fact').
_CERTIFIED = {VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT}


def _loglog(series: Series) -> tuple[float, list[float]]:
    """Global log-log regression slope + per-segment local slopes."""
    lx = [math.log(x) for x, _ in series]
    ly = [math.log(y) for _, y in series]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    denom = sum((lx[i] - mx) ** 2 for i in range(n))
    g = sum((lx[i] - mx) * (ly[i] - my) for i in range(n)) / denom
    local = [(ly[i + 1] - ly[i]) / (lx[i + 1] - lx[i]) for i in range(n - 1)]
    return g, local


def conditions_complete(c: Conditions) -> bool:
    """R1, engine-aware: an ANALOG verdict needs load-bearing corner/temp/vdd (a conditionless measured
    value is as context-free as a generic law -> REJECTED). A DIGITAL (functional) verdict has no PVT
    axis at all, so its conditions are complete by construction — the engine-typed Conditions union is
    exactly what lets R1 stay strict for analog without falsely rejecting digital."""
    if getattr(c, "kind", None) == "digital":
        return True
    return bool(c.corner) and math.isfinite(c.temp_c) and math.isfinite(c.vdd)


def judge_quant(quant: QuantTest, data) -> tuple[VerdictClass, str]:
    """Falsify one QUANT assertion against canonical data (a (x,y) series, or a scalar
    for kind='value'). Pure, deterministic."""
    k = quant.kind
    # Degenerate-input guard (C1): never certify an empty / too-short / wrong-shape series. A failed
    # ngspice run drops points (see runner.parse_rdata), so a claim must FLAG rather than crash or
    # certify a truncated curve. `value` needs a scalar; the executor only ever binds series, so a
    # value-kind claim with a list FLAGs instead of raising TypeError on float(list).
    if k == "value":
        if not isinstance(data, (int, float)):
            return VerdictClass.FLAGGED, "value-kind needs a scalar; got a series (scalar extraction unsupported)"
    elif not isinstance(data, (list, tuple)) or len(data) < 2:
        n = len(data) if hasattr(data, "__len__") else 0
        return VerdictClass.FLAGGED, f"degenerate series (n={n}); not enough points to judge"
    if k == "elasticity":
        g, local = _loglog(data)
        lo, hi = quant.band
        if not (lo <= g <= hi):
            return VerdictClass.REFUTED, f"global slope {g:+.3f} out of band {quant.band}"
        allin = all(lo <= s <= hi for s in local)
        v = VerdictClass.VERIFIED if allin else VerdictClass.VERIFIED_WITH_CAVEAT
        return v, f"global slope {g:+.3f} in band; locals_all_in={allin}"
    if k == "direction":
        ys = [y for _, y in data]
        trend = ys[-1] - ys[0]
        want = 1 if quant.sign == "+" else -1
        # I6: a flat (zero-trend) series satisfies NO direction claim. copysign(1, +0.0) is +1, which
        # would wrongly certify a '+' claim (and refute '-') on a non-responsive metric — asymmetric
        # and unsound. Require the trend to be significant relative to the series scale.
        scale = (abs(ys[0]) + abs(ys[-1])) / 2 or 1.0
        if abs(trend) <= 1e-6 * scale:
            return VerdictClass.FLAGGED, f"no significant trend (Δ={trend:.3g}, scale={scale:.3g})"
        if math.copysign(1, trend) != want:
            return VerdictClass.REFUTED, f"trend {'+' if trend > 0 else '-'} != predicted {quant.sign}"
        mono = all((ys[i + 1] - ys[i]) * want > 0 for i in range(len(ys) - 1))
        v = VerdictClass.VERIFIED if mono else VerdictClass.VERIFIED_WITH_CAVEAT
        return v, f"trend {quant.sign}, monotonic={mono}"
    if k == "direction_to_optimum":
        ys = [y for _, y in data]
        want = 1 if quant.sign == "+" else -1
        idx = max(range(len(ys)), key=lambda i: ys[i]) if want > 0 else min(range(len(ys)), key=lambda i: ys[i])
        rises = all((ys[i + 1] - ys[i]) * want > 0 for i in range(idx))
        if not rises:
            return VerdictClass.REFUTED, "does not rise toward predicted optimum"
        v = VerdictClass.VERIFIED_WITH_CAVEAT if idx < len(ys) - 1 else VerdictClass.VERIFIED
        return v, f"rises to interior optimum at index {idx}"
    if k == "invariance":
        if data and isinstance(data[0], (tuple, list)):
            ys = [y for x, y in data if quant.range_max is None or x <= quant.range_max]
        else:
            ys = list(data)
        if not ys:
            return VerdictClass.FLAGGED, "invariance: series empty after range_max filter"
        # I6: absolute spread is scale-aware — use it for dB/log metrics, where a fractional CoV on
        # the logarithmic value is meaningless (15% CoV on a 70 dB baseline ≈ 3.3× linear gain).
        if quant.spread_max is not None:
            spread = max(ys) - min(ys)
            v = VerdictClass.VERIFIED if spread <= quant.spread_max else VerdictClass.REFUTED
            return v, f"spread {spread:.3g} (max {quant.spread_max:.3g})"
        mean = sum(ys) / len(ys)
        if mean == 0:
            return VerdictClass.FLAGGED, "invariance: zero-mean series (CoV undefined; use spread_max)"
        cov = (max(ys) - min(ys)) / mean
        v = VerdictClass.VERIFIED if cov <= quant.cov_max else VerdictClass.REFUTED
        return v, f"CoV {cov * 100:.1f}% (max {quant.cov_max * 100:.0f}%)"
    if k == "value":
        meas = float(data)
        err = abs(quant.value - meas)
        v = VerdictClass.VERIFIED if err <= quant.tol else VerdictClass.REFUTED
        return v, f"asserted {quant.value} vs canonical {meas:.3g} (tol {quant.tol}) err={err:.3g}"
    if k == "statistical":
        ys = [y for _, y in data] if (data and isinstance(data[0], (tuple, list))) else list(data)
        if len(ys) < 8:
            return VerdictClass.FLAGGED, f"statistical: only {len(ys)} samples (<8); cannot bound a 3σ"
        sigma = pstdev(ys)
        kk = 3.0 if quant.reducer == "three_sigma" else (quant.k or _probit_k(quant.n_col))
        stat = kk * sigma
        if quant.absolute:
            metric, unit = stat, ""
        else:
            mu = sum(ys) / len(ys)
            if mu == 0:
                return VerdictClass.FLAGGED, "statistical: zero-mean series; use an absolute bound for offset"
            metric, unit = stat / abs(mu), "frac"
        verdict = VerdictClass.VERIFIED if metric <= quant.bound else VerdictClass.REFUTED
        return verdict, f"k={kk:.3g}·σ={stat:.3g}{unit} vs bound {quant.bound:.3g} (n={len(ys)})"
    if k == "corner":
        ys = [y for _, y in data] if (data and isinstance(data[0], (tuple, list))) else list(data)
        if len(ys) < 2:
            return VerdictClass.FLAGGED, f"corner: only {len(ys)} corner(s); need >=2"
        meets = (lambda y: y >= quant.bound) if quant.sign == "+" else (lambda y: y <= quant.bound)
        fails = [i for i, y in enumerate(ys) if not meets(y)]
        if not fails:
            return VerdictClass.VERIFIED, f"all {len(ys)} corners meet {quant.sign}{quant.bound:.3g}"
        return VerdictClass.REFUTED, f"corners {fails} fail {quant.sign}{quant.bound:.3g}"
    raise ValueError(f"unknown quant kind: {k}")


@dataclass
class Scorecard:
    leak_naive: int = 0       # injected claims reaching 'fact' under the NAIVE rule (mechanism rides quant)
    leak_hardened: int = 0    # injected claims reaching 'fact' under the HARDENED rule (mechanism never fact)
    usefulness_ok: int = 0    # real claims whose quant is certified
    usefulness_total: int = 0
    rows: list[dict] = field(default_factory=list)


class ClaimOracle:
    """Judges claim-cards against canonical sim data and scores teaching-eligibility."""

    def judge(self, claim: ClaimCard, canonical: dict) -> ClaimCard:
        """Set claim.verdict / verdict_note. REJECTED if R1 conditions are incomplete."""
        if not conditions_complete(claim.conditions):
            claim.verdict, claim.verdict_note = VerdictClass.REJECTED, "conditionless measured value (R1)"
            return claim
        data = canonical[claim.mechanism.series_ref]
        claim.verdict, claim.verdict_note = judge_quant(claim.mechanism.quant, data)
        return claim

    def teaching_fact(self, claim: ClaimCard, hardened: bool = True) -> bool:
        """Is this claim eligible to be TAUGHT as fact? Under the hardened rule a claim
        carrying a mechanism narrative is never auto-fact (only its quant is certified)."""
        if claim.verdict not in _CERTIFIED:
            return False
        if hardened and claim.has_mechanism_narrative():
            return False
        return True

    def scorecard(self, claims: list[ClaimCard], canonical: dict, injected_ids: set[str]) -> Scorecard:
        """Reproduce the Specimen-Zero leak-rate measurement. `injected_ids` = the
        deliberately-wrong claims (revealed only here, never during judging)."""
        sc = Scorecard()
        for c in claims:
            self.judge(c, canonical)
            naive_fact = self.teaching_fact(c, hardened=False)
            hard_fact = self.teaching_fact(c, hardened=True)
            if c.id in injected_ids:
                if naive_fact:
                    sc.leak_naive += 1
                if hard_fact:
                    sc.leak_hardened += 1
            else:
                sc.usefulness_total += 1
                if c.verdict in _CERTIFIED:
                    sc.usefulness_ok += 1
            sc.rows.append({"id": c.id, "verdict": c.verdict.value, "naive_fact": naive_fact,
                            "hardened_fact": hard_fact, "injected": c.id in injected_ids,
                            "note": c.verdict_note})
        return sc
