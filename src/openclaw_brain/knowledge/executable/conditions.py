"""Engine-typed measurement conditions — a CLOSED discriminated union (ADR Decision 4, Tier-B §1b).

The PVT triple was an analog over-fit; digital has no PVT. `Conditions` is therefore a discriminated
union keyed by `kind`, and EVERY consumer routes through `summarize_scope` — a TOTAL function that
RAISES on an unhandled kind. So registering engine #3 with a new conditions variant forces a new branch
HERE (a loud failure) rather than silently defaulting to nullable PVT. This is the anti-rot core that
makes the ADR's "engines 3..N touch only the registry" claim true and enforced.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, BeforeValidator, Field


class UnknownConditionsKind(Exception):
    """summarize_scope met a Conditions kind it has no branch for (the engine-3 guard fired)."""


class AnalogPVT(BaseModel):
    """ngspice conditions. corner/temp/vdd are load-bearing (R1): a value without them is as
    context-free as a generic law. Field-identical to the pre-Tier-B `Conditions`."""

    kind: Literal["analog_pvt"] = "analog_pvt"
    corner: str                                   # tt | ss | ff | sf | fs
    temp_c: float
    vdd: float
    cl_f: float | None = None
    pdk_profile: dict[str, Any] = Field(default_factory=dict)  # class, vdd_nom, vth_approx, device, pdk_id, pdk, node
    mc_runs: int | None = None            # statistical: Monte-Carlo sample count
    corners: list[str] | None = None      # corner: the process corners to require (VERIFIED-iff-all)
    areas: list[float] | None = None      # pelgrom: input-device WL multipliers to fit σ across


class DigitalUnits(BaseModel):
    """iverilog conditions. No PVT: a functional sim's scope is units/timing, not silicon corners."""

    kind: Literal["digital"] = "digital"
    clock_period_ns: float | None = None
    bit_width: int | None = None
    stimulus: str | None = None            # exercised input class, e.g. "count 0->15, single wrap"
    stimulus_untested: str | None = None   # dominant candidate falsifier the exercised scope doesn't cover


def _default_analog_kind(v: Any) -> Any:
    """Back-compat coercion: a bare PVT dict (the pre-Tier-B shape, no `kind`) IS the analog variant.

    This rescues only the legacy analog shape — a kind-less DIGITAL dict (e.g. {clock_period_ns: 10})
    would be stamped analog_pvt and then FAIL AnalogPVT validation (no corner/temp/vdd), so digital must
    still declare itself explicitly. The discriminator stays meaningful; the engine-3 guard
    (summarize_scope raising on an unknown kind) is untouched.
    """
    if isinstance(v, dict) and "kind" not in v:
        return {**v, "kind": "analog_pvt"}
    return v


# Discriminated union: pydantic resolves the variant by the `kind` literal (the before-validator
# supplies the legacy analog default). Construct the CONCRETE class (AnalogPVT/DigitalUnits) in new
# code — `Conditions` itself is a typing alias, used only as a field type.
Conditions = Annotated[
    Union[AnalogPVT, DigitalUnits],
    BeforeValidator(_default_analog_kind),
    Field(discriminator="kind"),
]


def summarize_scope(c: Any) -> dict[str, Any]:
    """The machine-surfaced verdict scope (Decision 4.1). TOTAL over the union; raises on unknown kind.

    Analog: nominal device, a single PVT corner string, no statistical coverage.
    Digital: device/statistical inapplicable; the verdict is a functional (RTL-entails-property) one.
    """
    kind = getattr(c, "kind", None)
    if kind == "analog_pvt":
        pdk = (c.pdk_profile or {}).get("pdk", "sky130")    # node FIRST-CLASS: undetachable from the magnitude
        node = (c.pdk_profile or {}).get("node")
        scope: dict[str, Any] = {
            "device": "mismatch" if c.mc_runs else "nominal",
            "pdk": pdk,
            "corners": ([f"{pdk}/{x}" for x in c.corners]
                        if c.corners else [f"{pdk}/{c.corner}/{c.temp_c:g}/{c.vdd:g}"]),
            "statistical": f"3σ@{c.mc_runs}" if c.mc_runs else "none",
        }
        if node:
            scope["node"] = node
        if c.areas:
            scope["pelgrom"] = f"areas@{len(c.areas)}"
        return scope
    if kind == "digital":
        return {"device": "n/a", "corners": [], "statistical": "n/a", "functional": True,
                "stimulus": c.stimulus or "unspecified"}
    raise UnknownConditionsKind(f"no scope branch for conditions kind {kind!r}")
