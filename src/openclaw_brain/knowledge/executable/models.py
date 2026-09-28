"""Data models for the executable-circuit substrate (v2).

The unit of knowledge is the **claim-card**: a falsifiable mechanism claim bound to a
reproducible deck, carrying its measurement conditions (R1). The oracle (oracle.py)
certifies the QUANT assertion (direction / magnitude / value / invariance) only — the
mechanism narrative is Interpretive and never oracle-certified (the measured leak-rate
finding, SPEC §4). See docs/specs/SPEC_EXECUTABLE_CIRCUIT_SUBSTRATE.md §2.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# Conditions is a CLOSED discriminated union (Tier-B §1b). Re-exported here so the 15 analog
# specimens keep importing it (and AnalogPVT/DigitalUnits) from .models unchanged.
from openclaw_brain.knowledge.executable.conditions import (  # noqa: F401  (re-export)
    AnalogPVT,
    Conditions,
    DigitalUnits,
)


class VerdictClass(str, Enum):
    """Oracle verdict on a claim-card's QUANT assertion."""

    VERIFIED = "VERIFIED"
    VERIFIED_WITH_CAVEAT = "VERIFIED_WITH_CAVEAT"  # global law holds; regime-dependent deviation
    VERIFIED_NEGATIVE = "VERIFIED_NEGATIVE"        # a real, correct negative result (e.g. spec FAIL)
    REFUTED = "REFUTED"
    REFUTED_MAGNITUDE = "REFUTED_MAGNITUDE"        # direction right, magnitude law wrong
    FLAGGED = "FLAGGED"                            # cannot certify; teacher-narrated only
    REJECTED = "REJECTED"                          # not eligible (e.g. conditionless measured value)


class QuantTest(BaseModel):
    """The machine-certifiable assertion. `kind` selects which fields apply."""

    kind: Literal["direction", "direction_to_optimum", "elasticity", "value", "invariance",
                  "statistical", "corner"]
    # direction / direction_to_optimum
    sign: str | None = None                       # "+" | "-" (also: corner ≥ / ≤)
    # elasticity
    target: float | None = None
    band: tuple[float, float] | None = None
    # value
    value: float | None = None
    tol: float | None = None
    # invariance
    cov_max: float | None = None                  # fractional CoV bound (linear-scale metrics)
    spread_max: float | None = None               # absolute max-min bound (scale-aware; use for dB/log metrics)
    range_max: float | None = None                # filter: include series x <= range_max
    # statistical (Monte-Carlo mismatch) / corner (process)
    reducer: str | None = None                    # "three_sigma" | "k_sigma"
    k: float | None = None                        # explicit k for k_sigma
    n_col: int | None = None                      # column count -> k = Φ⁻¹(1 - 1/n_col) (worst-of-N FPN)
    bound: float | None = None                    # statistical (≤) + corner bound
    absolute: bool = True                         # statistical: absolute (offset) vs fractional

    @model_validator(mode="after")
    def _check_fields(self) -> "QuantTest":
        if self.kind in ("direction", "direction_to_optimum") and self.sign not in ("+", "-"):
            raise ValueError(f"{self.kind} requires sign '+' or '-'")
        if self.kind == "elasticity" and self.band is None:
            raise ValueError("elasticity requires a band")
        if self.kind == "value" and (self.value is None or self.tol is None):
            raise ValueError("value requires value + tol")
        if self.kind == "invariance" and self.cov_max is None and self.spread_max is None:
            raise ValueError("invariance requires cov_max (fractional) or spread_max (absolute)")
        if self.kind == "statistical":
            if self.reducer not in ("three_sigma", "k_sigma"):
                raise ValueError("statistical requires reducer 'three_sigma' or 'k_sigma'")
            if self.bound is None:
                raise ValueError("statistical requires a bound")
            if self.reducer == "k_sigma" and self.k is None and self.n_col is None:
                raise ValueError("k_sigma requires k or n_col")
        if self.kind == "corner" and (self.sign not in ("+", "-") or self.bound is None):
            raise ValueError("corner requires sign '+'/'-' and a bound")
        return self


class MechanismClaim(BaseModel):
    """A falsifiable mechanism claim. `quant` is certifiable; `narrative` is NOT (Interpretive)."""

    knob: str                         # swept parameter (e.g. "Cc")
    metric: str                       # measured quantity (e.g. "gbw_hz")
    series_ref: str                   # key into canonical sim data (a (x,y) series, or a scalar for `value`)
    quant: QuantTest
    narrative: str | None = None      # mechanism prose — never oracle-certified


class ClaimCard(BaseModel):
    """The knowledge atom. R1 conditions are mandatory by construction."""

    id: str
    topology_class: str
    mechanism: MechanismClaim
    conditions: Conditions
    grounds: list[str] = Field(default_factory=list)
    # populated by the oracle:
    verdict: VerdictClass | None = None
    verdict_note: str | None = None
    # scope-honesty (ADR Decisions 4-5), stamped by the executor at judging time:
    engine: str = "ngspice"                                  # which engine certified this
    basis: str | None = None                                 # "physical-nominal" (ngspice) | "functional" (iverilog)
    scope: dict[str, Any] = Field(default_factory=dict)      # summarize_scope(conditions): {device, corners, statistical}
    dominant_risk_untested: str | None = None                # named untested dominant axis (mismatch-driven classes)

    def has_mechanism_narrative(self) -> bool:
        return bool(self.mechanism.narrative and self.mechanism.narrative.strip())


class Specimen(BaseModel):
    """A content-addressed specimen: a netlist (the connectivity+sizing SSOT) with the
    verdicted claim-cards that accrete onto it. Its identity (`spec_id`) is the NETLIST
    (+ tb + role_map + pdk + tool) — NOT the claim-cards, so later sources can MERGE
    additional claim-cards onto the same specimen (topology_class is the merge key).
    See docs/specs/SPEC_EXECUTABLE_CIRCUIT_SUBSTRATE.md §2-§3."""

    topology_class: str
    netlist: str                          # cell.spice — connectivity + sizing SSOT
    testbench: str = ""                   # tb deck (optional)
    # The rendered sweep decks that produced the canonical data (route_key -> deck). Persisted for
    # reproducibility (a specimen is self-contained / re-runnable from the corpus) but NOT part of
    # spec_id — identity is the NETLIST so claims from DIFFERENT sweeps accrete onto one specimen.
    testbenches: dict[str, str] = Field(default_factory=dict)
    pdk: str = "sky130A"
    tool: str = "ngspice-46"
    role_map: dict[str, str] = Field(default_factory=dict)  # device ref -> role (exact for templates)
    claim_cards: list[ClaimCard] = Field(default_factory=list)
    spec_id: str | None = None            # content hash, set by the corpus on store


class VerificationRecipe(BaseModel):
    """The ① recipe-authoring output (SPEC §5.1): a protocol-conforming plan that the
    ② executor runs mechanically. Authoring is amortized per topology; execution is per specimen."""

    topology_class: str
    source_ref: str = ""
    build: dict[str, Any] = Field(default_factory=dict)        # {method, template_ref, role_map}
    sizing: dict[str, Any] = Field(default_factory=dict)       # {method, targets, seed}
    conditions: Conditions
    sweeps: list[dict[str, Any]] = Field(default_factory=list)  # [{analysis, knob, points, measure}]
    claim_cards: list[ClaimCard] = Field(default_factory=list)
    oracle_rules: list[str] = Field(
        default_factory=lambda: ["mechanism_never_fact", "reject_conditionless_measured", "redrive_dont_trust"]
    )
    escalation: dict[str, Any] = Field(
        default_factory=lambda: {"on": ["oracle_inconsistency", "nonconvergence", "tb_validity_fail"], "to": "simulation"}
    )
