"""InterventionSpec registry (E3-I1, spec §2/§4-I1) — the do-operator's causal-component catalogue.

A NEW, deliberately small module rather than folding this into templates.py or executor.py:
templates.py's registries (TEMPLATES/RENDERERS) answer "how do I render+run a topology's own deck";
this registry answers a DIFFERENT question — "which two decks (baseline, variant) does an
intervention id pair, and what did we idealize to build the variant" — a metadata axis with no deck-
rendering logic of its own (`idealization` is prose, not a netlist fragment). Keeping it separate
means `grep interventions.py` finds every registered causal component without wading through 1500+
lines of template bodies, mirrors the spec's own vocabulary (`InterventionSpec`, distinct from
`TEMPLATES`/`RENDERERS`), and keeps executor.py's new 'intervention' sweep branch resolving from ONE
place — exactly the "registry, never run_recipe's control flow" discipline the engine registry
(engines.py) and the digital/MC template blocks already established.

Adding intervention N+1 = one more entry here (id, base_topology_class, variant_template_ref,
idealization) + the variant's renderer bodies in templates.py (probe-validated, human-authored, the
D3-line restraint spec §6 calls out) — this file is never touched by anything except a new
registration.
"""

from __future__ import annotations

from dataclasses import dataclass


class UnknownIntervention(Exception):
    """get_intervention() was asked for an id not in INTERVENTIONS — raised BEFORE any render/sim
    (executor.run_recipe validates every sweep's intervention id up front, mirroring the existing
    mismatch-availability / unsupported-template guards)."""


@dataclass(frozen=True)
class InterventionSpec:
    """One causal-component intervention (spec §2): pairs a topology's own BASELINE template with a
    hand-validated VARIANT renderer that structurally removes/neutralizes the hypothesized pathway.

    `idealization`: None for a PHYSICAL variant (rz_null — every element is a real, simulatable
    device); a short, named phrase for an IDEALIZED variant (ff_break — an ideal E-source buffer is
    itself an abstraction, so the certified statement is scope-honestly "in the model, with this
    idealized intervention" — spec §2's scope-honesty requirement, never silently more than that).
    """

    id: str
    base_topology_class: str
    variant_template_ref: str
    idealization: str | None = None
    notes: str = ""


INTERVENTIONS: dict[str, InterventionSpec] = {
    "rz_null": InterventionSpec(
        id="rz_null",
        base_topology_class="miller_ota_2stage_nmos_in",
        variant_template_ref="miller_ota_ac__rz_null",
        idealization=None,   # PHYSICAL: Rz is a real series resistor, not an idealized element.
        notes=(
            "nulling resistor Rz in series with Ccomp (knob Rz, 10ohm..4500ohm ~= 0-ish..2/gm2 — "
            "gm2 measured live ~=4.458e-4 S). Live-validated 2026-07-04 (templates.py module comment "
            "above render_miller_ota_rz_null_ac has the full transcript): PM rises 34.35->70.92 deg "
            "over the sweep at fixed nominal Cc; Av0 unchanged (68.8739 dB at every point)."
        ),
    ),
    "ff_break": InterventionSpec(
        id="ff_break",
        base_topology_class="miller_ota_2stage_nmos_in",
        variant_template_ref="miller_ota_ac__ff_break",
        idealization=(
            "ideal unity-gain E-source buffer replicates the output node ('out'); the compensation "
            "cap Ccomp is redriven from the first-stage node o1 to this buffered replica instead of "
            "directly to 'out' — Miller multiplication as seen from o1 is preserved (o1 still charges "
            "Cc against a node that tracks 'out' exactly) while the feedforward CURRENT that Cc would "
            "otherwise inject into the real 'out' node's KCL is diverted into the ideal buffer instead, "
            "severing the RHP-zero-causing feedforward path"
        ),
        notes=(
            "knob Cc (the baseline's own sweep range). Live-validated 2026-07-04 (templates.py module "
            "comment above render_miller_ota_ff_break_ac has the full transcript, incl. a REJECTED "
            "first attempt that buffered o1 instead of 'out' and destroyed pole-splitting): delta-PM "
            "= PM_ff_break(Cc) - PM_baseline(Cc) is positive and grows with Cc (+11.29..+24.03 deg "
            "over 250f..4000f); delta-Av0 is exactly 0 at every point (the intervention's own DC-gain "
            "invariance control)."
        ),
    ),
}


def get_intervention(intervention_id: str) -> InterventionSpec:
    """The InterventionSpec for `intervention_id`. Raises UnknownIntervention for anything not in
    INTERVENTIONS — BEFORE any render or sim happens (executor.run_recipe's early guard calls this
    for every 'intervention'-analysis sweep before its per-card loop begins)."""
    try:
        return INTERVENTIONS[intervention_id]
    except KeyError:
        raise UnknownIntervention(
            f"unknown intervention {intervention_id!r}; known: {sorted(INTERVENTIONS)}"
        ) from None
