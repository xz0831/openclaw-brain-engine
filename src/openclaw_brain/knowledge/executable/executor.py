"""② Executor (SPEC §5) — the deterministic loop that turns a VerificationRecipe into verdicted,
stored, projected claim-cards. No model: this is the cheap/local bulk path that runs per specimen.

    recipe → (sizing) → templates.render → runner.measure → oracle.judge → corpus.store → projection

The intelligence was already spent authoring the recipe (① recipe.py); everything here is code.
The one subtlety the executor owns is SERIES ROUTING: the template emits each sweep keyed by the
knob (`RDATA <knob> ...`), but two claims can share a knob while measuring different metrics
(GBW-vs-Cc and PM-vs-Cc both sweep Cc) — those need DIFFERENT decks. So the executor runs ONE deck
per (knob, metric, points) and binds each claim to its own series. It does NOT trust the author's
`series_ref` to be unique — a live model labels by sweep ("Cc_sweep"), which collides across
metrics — so it REWRITES series_ref to the canonical (knob, metric) routing key, which is then
exactly what oracle.judge looks up. (Found by the (a) live-author probe: deepseek labelled both Cc
claims "Cc_sweep"; without the rewrite the GBW claim was judged against the Av0 series.)

Sizing (honest scope): the executor resolves sizing from recipe.sizing.seed over the template
default (a known all-saturation point). When recipe.sizing.method == 'gmid_lookup', it additionally
sizes the OTA INPUT PAIR by gm/ID (_size_input_pair_from_gmid): the input devices' current id1 is
tail-fixed, so choosing a gm/ID target sets gm1 — and thus GBW = gm1/2πCc — without disturbing
downstream bias. That is the focused, decoupled step. FULL multi-device synthesis (sizing the tail,
load, and 2nd stage from gain/PM targets together) remains a separate milestone. See SPEC §5.1.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from statistics import pstdev

from .envelope import REGISTRY as ENVELOPE_REGISTRY, envelope_check, measure_op_quantities, stamp_envelope_scope
from .gmid import characterize
from .interventions import get_intervention
from .mc_templates import MC_SEED_AREA_STRIDE
from .models import ClaimCard, Specimen, VerdictClass, VerificationRecipe
from .oracle import ClaimOracle, Series
from .pdks import (
    MM_CORNER_TOKEN,
    PDKProfile,
    get_profile,
    require_mismatch_available,
    sizing_overrides_for,
    substitute_devices,
)
from .projection import ProjectionWrite, project_specimen
from .recipe import TemplateCapability, capability_for
from .runner import _UNIT, parse_unit
from .conditions import summarize_scope
from .engines import engine_for_template, runner_for_engine
from .templates import RENDERERS, TEMPLATES

logger = logging.getLogger(__name__)


class UnsupportedTemplate(Exception):
    """The recipe's template_ref has no executor renderer (cannot run mechanically)."""


# Exception types oracle.judge() can raise for an ordinary bad/missing-measurement input, traced
# to oracle.py: `canonical[claim.mechanism.series_ref]` -> KeyError when a metric never got
# measured; `math.log` on a non-positive series value inside the elasticity regression, or the
# explicit `raise ValueError(f"unknown quant kind: {k}")` -> ValueError; a degenerate/all-equal-x
# regression denominator -> ZeroDivisionError (ArithmeticError). These are data/measurement-shape
# problems the oracle is meant to FLAG, not a claim the executor should treat as a code
# regression. Anything NOT in this set (AttributeError, TypeError, a raw assert, ...) is much
# more likely a genuine bug in oracle.py or a caller-side contract break — see the except split
# below.
_EXPECTED_JUDGE_FAILURES: tuple[type[Exception], ...] = (KeyError, ValueError, ArithmeticError)


# Mismatch-dominated classes whose nominal/functional verdict is SILENT on the real silicon risk (ADR
# Decision 4.2): the verdict carries an active "dominant risk NOT tested" banner naming the untested axis,
# so a learner is told, not merely able to infer. Classes absent here have no flagged dominant risk.
_DOMINANT_RISK: dict[str, str] = {
    "cds_switched_cap_nmos": "device/cap mismatch -> column FPN, kTC, charge-injection (the dominant CIS readout risk)",
    "comparator_continuous_nmos": "input-referred offset from device mismatch -> decision error; outside nominal scope",
    "diff_pair_resistive_nmos": "input-pair Vth mismatch -> input-referred offset; outside nominal scope",
    "ota_5t_nmos_in": "input-pair + mirror mismatch -> input offset; outside nominal scope",
    "digital_cds_nmos": "in silicon the analog S/H + comparator offset & kTC dominate; this FUNCTIONAL model omits them",
}

# E2a-I2: a sweep's `knob` label is not always the SAME string as its own `.param`/sizing dict key —
# most pilots' knob equals its sizing key exactly (Cc, CL), but common_source_active_load_nmos's
# bias-current knob is labelled "Iref" (CS_KNOB_ELEMENTS maps it to the alter-element "IREF") while
# its OWN sizing/.param key is "IREFV" (templates.py's DEFAULT_CS_SIZING / _CS_PARAM_ORDER) — found
# live 2026-07-05 running the I2 cross-PDK experiment (a bare `sizing["Iref"]` KeyError on cs_av0, on
# EVERY pdk, sky130A included). Scoped to exactly what the envelope wiring below needs; any OTHER
# knob not listed here is assumed to equal its own sizing key (true for every other pilot: Cc, CL).
_ENVELOPE_KNOB_SIZING_KEY: dict[str, str] = {"Iref": "IREFV"}


@dataclass
class ExecutionResult:
    recipe: VerificationRecipe
    specimen: Specimen
    canonical: dict[str, Series]
    claim_cards: list[ClaimCard]                 # verdicted (judged in place)
    spec_id: str | None = None
    spec_dir: str | None = None
    projection: ProjectionWrite | None = None
    runs: int = 0                                # decks actually simulated (after dedup)
    sizing_prediction: dict | None = None        # gm/ID sizing record (None unless method='gmid_lookup')
    notes: list[str] = field(default_factory=list)


def _resolve_sizing(recipe: VerificationRecipe, default_sizing: dict, pdk: str = "sky130A") -> dict:
    """Concrete .param values: template default, overlaid with the PDK-specific sizing override
    (pdks.sizing_overrides_for — populated by E1-I2 for the pilot ports), overlaid
    with the recipe's own sizing seed (an author's explicit choice always wins last).
    gm/ID-driven input-pair sizing (method='gmid_lookup') is applied separately in run_recipe,
    where the runner is available to characterize the device."""
    sizing = dict(default_sizing)
    sizing.update(sizing_overrides_for(recipe.topology_class, pdk))
    seed = (recipe.sizing or {}).get("seed") or {}
    sizing.update({k: str(v) for k, v in seed.items()})
    return sizing


def _resolve_pdk(recipe: VerificationRecipe) -> str:
    """The pdk key an ngspice recipe selects via conditions.pdk_profile.pdk (the existing scope
    field, first-class since pre-E1 — spec §5-I1 requirement 4 reuses it rather than adding schema).
    Defaults to "sky130A" when unset — byte-identical default behavior. DigitalUnits conditions
    carry no pdk_profile at all; callers only invoke this for the ngspice engine."""
    pdk_profile = getattr(recipe.conditions, "pdk_profile", None) or {}
    return pdk_profile.get("pdk") or "sky130A"


def _measure(runner, engine: str, profile: PDKProfile | None, deck: str) -> dict[str, Series]:
    """Route to runner.measure with the pdk kwarg ONLY when a genuinely non-default PDK was
    resolved (engine=='ngspice' and profile.pdk != 'sky130A', checked on the ALREADY-RESOLVED
    profile so the legacy pdk_profile.pdk="sky130" alias — which resolves to the sky130A identity
    profile — takes this branch too). Every existing fake-runner test double
    (`measure(self, deck, timeout=300)`, no pdk param) keeps working unmodified on the default path;
    only a genuinely-selected non-sky130A profile needs a pdk-aware runner (the real NgspiceRunner,
    or a test double written for it)."""
    if engine == "ngspice" and profile is not None and profile.pdk != "sky130A":
        return runner.measure(deck, pdk=profile.pdk)
    return runner.measure(deck)


def _gmid_target(recipe: VerificationRecipe) -> float | None:
    """Read the input-pair gm/ID target. Canonical home is sizing.targets.gm_id (models.py),
    with flat sizing.gmid_target / target_gmid accepted as aliases."""
    s = recipe.sizing or {}
    targets = s.get("targets") or {}
    raw = targets.get("gm_id") or targets.get("input_pair_gm_id") or s.get("gmid_target") or s.get("target_gmid")
    return float(raw) if raw is not None else None


def _size_input_pair_from_gmid(recipe: VerificationRecipe, sizing: dict, runner, corner: str):
    """gm/ID-driven sizing of the Miller-OTA input pair (focused scope: the input pair only).

    The input devices carry id1 = Itail/2, a current the TAIL (M5) fixes — changing the input
    W/multiplicity changes the inversion level (and thus gm1, hence GBW = gm1/2πCc) but NOT id1,
    so downstream bias is untouched. That decoupling is what keeps this a clean per-device step
    rather than a full multi-device synthesis.

    Steps: derive id1 from the bias mirror (IREF·W5/W8 / 2) → characterize the nfet (clean-room
    ngspice sweep) → pick the multiplicity m that carries id1 at the target gm/ID → set the input
    device to W={w_char} m={m}. Returns (updated sizing, prediction dict).

    m is the EXACT float multiplicity, not int(): id ∝ m exactly (gmid round-trip), and the
    realistic case is fractional m (target current < unit-device current) — int() would round a
    valid 0.7 device to 0 and delete the input pair. W1 is set to the characterized unit width so
    the device is exactly m of THAT unit (sizing m for one unit width and applying it to another
    would scale id1 wrongly). gm/ID is first-order: characterization is at VDS=0.9/VSB=0 while the
    in-circuit device sees body effect + a different VDS, so realized gm1 (GBW) tracks the
    prediction to within tens of percent — the predicted GBW is recorded so the loop is checkable."""
    gm_id = _gmid_target(recipe)
    if gm_id is None:
        gm_id = 15.0   # moderate inversion — a sane default if the recipe omitted a target

    # id1 = Itail/2, with Itail estimated from the bias mirror as IREF·(W5/W8). NOTE this uses an
    # id∝W proportionality for the M8->M5 mirror — the same approximation gmid.size() declares
    # invalid in sky130 (~10-25% W-dependence), so id1 (and gm1_pred/gbw_pred below) carries that
    # error on top of body-effect/VDS. Acceptable for a first-order sizing seed; disclosed in the note.
    iref = parse_unit(str(sizing["IREFV"]))
    itail = iref * (parse_unit(str(sizing["W5"])) / parse_unit(str(sizing["W8"])))
    id1 = itail / 2.0

    table = characterize(runner, device="nfet", l_um=parse_unit(str(sizing["Lp"])), corner=corner)
    m = table.size(gm_id, id1)

    sizing = dict(sizing)
    sizing["W1"] = f"{table.w_char_um:g}"   # input device is m units of the characterized unit width
    sizing["M1"] = f"{m:g}"

    gm1_pred = gm_id * id1
    gbw_pred = gm1_pred / (2 * math.pi * parse_unit(str(sizing["Cc"])))
    prediction = {
        "method": "gmid_lookup", "gm_id": gm_id, "id1_a": id1, "m": m,
        "w_char_um": table.w_char_um, "gm1_pred_s": gm1_pred, "gbw_pred_hz": gbw_pred,
        "note": (f"gm/ID: input pair sized W={table.w_char_um:g} m={m:.3f} for gm/ID={gm_id:g} "
                 f"@ id1≈{id1 * 1e6:.2f}µA (id1 via W-ratio mirror, approx) -> gm1≈{gm1_pred * 1e6:.1f}µS, "
                 f"GBW_pred≈{gbw_pred / 1e6:.2f}MHz @ Cc={sizing['Cc']}"),
    }
    return sizing, prediction


def _points_for(recipe: VerificationRecipe, knob: str, metric: str) -> list[str] | None:
    """The sweep points the recipe specified for this (knob, metric), or None (renderer default)."""
    for sw in recipe.sweeps:
        if sw.get("knob") != knob:
            continue
        measure = sw.get("measure") or sw.get("metrics") or []
        if isinstance(measure, str):
            measure = [measure]
        if metric in measure and sw.get("points"):
            return [str(p) for p in sw["points"]]
    return None


def _series_invalid_reason(series: Series, points: list[str] | None) -> str | None:
    """Why this measured series must not be judged (C1), or None if it is sound.
    Empty -> the deck produced no parseable RDATA. Short -> a `.meas` failed at some sweep
    point(s) and those points dropped (runner.parse_rdata skips the non-finite reset value),
    so the curve is partial; certifying a partial curve would teach a number off a failed run."""
    if not series:
        return "empty series (no valid RDATA parsed)"
    if points is not None and len(series) < len(points):
        return f"truncated series: {len(series)}/{len(points)} points (a .meas likely failed)"
    return None


def _sweep_mode(recipe: VerificationRecipe, knob: str) -> str | None:
    """The sweep's analysis mode for this knob — 'mc'|'corner'|'pelgrom' (Stat-QT off-nominal axes)
    or 'intervention' (E3-I1) or 'ac'/'dc'/'tran' (analog) or None. Drives the series-source branch
    in run_recipe."""
    for sw in recipe.sweeps:
        if sw.get("knob") == knob:
            return sw.get("analysis")
    return None


def _intervention_id_for(recipe: VerificationRecipe, knob: str) -> str | None:
    """The 'intervention' id an {analysis: 'intervention', ...} sweep registered for `knob`, or None.
    Mirrors `_sweep_mode`'s lookup pattern (spec §2's sweep shape carries `intervention` alongside
    `analysis`/`knob`/`points`/`measure`)."""
    for sw in recipe.sweeps:
        if sw.get("knob") == knob and sw.get("analysis") == "intervention":
            return sw.get("intervention")
    return None


def _derive_kind_for(recipe: VerificationRecipe, knob: str, metric: str) -> str:
    """Which derived series (spec §2) an intervention claim on (knob, metric) routes to: 'delta'
    (default — the pilot's own choice for every §3 card) or 'ratio' (opt-in per sweep, `derive:
    'ratio'`, for physics that fits a multiplicative comparison better)."""
    for sw in recipe.sweeps:
        if sw.get("knob") != knob or sw.get("analysis") != "intervention":
            continue
        measure = sw.get("measure") or sw.get("metrics") or []
        if isinstance(measure, str):
            measure = [measure]
        if metric in measure:
            return sw.get("derive", "delta")
    return "delta"


def _paired_series(base: Series, variant: Series) -> dict[str, Series]:
    """Elementwise delta (variant - baseline) and ratio (variant / baseline) over two ALREADY-ALIGNED
    series (same swept points, same order — both were rendered from the SAME `points` list). Truncates
    to the shorter length: a length mismatch means a `.meas` failed on one side at some point(s); the
    caller's existing C1 guard (`_series_invalid_reason`, comparing len(series) to len(points)) then
    FLAGs the claim rather than certifying a partially-paired curve."""
    n = min(len(base), len(variant))
    delta = [(variant[i][0], variant[i][1] - base[i][1]) for i in range(n)]
    ratio = [(variant[i][0], (variant[i][1] / base[i][1]) if base[i][1] else float("nan"))
             for i in range(n)]
    return {"delta": delta, "ratio": ratio}


def _broadcast_paired_series(base_y: float, variant: Series) -> dict[str, Series]:
    """Delta/ratio against a SINGLE reference value, broadcast across every variant point — the
    pairing an intervention's own knob (e.g. rz_null's Rz) is not native to the baseline template at
    all (spec §3: the baseline reference is one fixed measurement, not a sweep)."""
    delta = [(x, y - base_y) for x, y in variant]
    ratio = [(x, (y / base_y) if base_y else float("nan")) for x, y in variant]
    return {"delta": delta, "ratio": ratio}


def _run_intervention_pair(
    spec, cap: TemplateCapability, sizing: dict, corner: str, knob: str, metric: str,
    points: list[str] | None, engine: str, profile: PDKProfile | None, runner,
) -> dict[str, Series]:
    """Render+run the BASELINE deck (the recipe's own class/template — `cap`) and the VARIANT deck
    (`spec.variant_template_ref`) with IDENTICAL sizing (spec §2/§3), then derive BOTH the delta and
    ratio series; the caller routes each claim card to whichever derive kind it asked for
    (`_derive_kind_for`).

    Baseline pairing is knob-aware (spec §3's two pilot shapes):
      - `knob` NATIVE to the baseline template (e.g. ff_break's Cc — miller_ota_2stage_nmos_in already
        sweeps Cc): both decks sweep the SAME knob over the SAME points, paired point-for-point.
      - `knob` NOT native to the baseline at all (e.g. rz_null's Rz — deliberately absent from the
        baseline netlist, templates.py's OTA_KNOB_ELEMENTS comment): the baseline is measured ONCE, at
        a fixed reference point (the topology's own first native knob, held at its CURRENT resolved
        sizing value — i.e. "PM without Rz, at the SAME nominal Cc the variant also holds fixed"), and
        broadcast across every variant sweep point.

    Neither pilot intervention renders any MC/random element (no `agauss`/`reset`/`setseed` in either
    variant body), so "identical seeds" (spec §3) reduces to "identical sizing" here — there is no
    seed axis to diverge. A future MC-bearing intervention passes the SAME seed_offset/mc_runs kwargs
    to both render_sweep calls below; this structure supports that directly (uniform **kwargs).
    """
    base_renderers = RENDERERS.get(cap.template_ref)
    if base_renderers is None:          # defense-in-depth; run_recipe already resolved this once
        raise UnsupportedTemplate(f"no renderer for base template_ref {cap.template_ref!r}")
    base_render_sweep, _, _ = base_renderers
    variant_renderers = RENDERERS.get(spec.variant_template_ref)
    if variant_renderers is None:
        raise UnsupportedTemplate(
            f"no renderer for variant_template_ref {spec.variant_template_ref!r} "
            f"(intervention {spec.id!r}); have {sorted(RENDERERS)}"
        )
    variant_render_sweep, _, variant_default_sizing = variant_renderers
    # Variant-only params (e.g. rz_null's Rz) come from the variant's own defaults; every param the
    # BASELINE template also defines (Cc, CL, W1, ...) is OVERRIDDEN by the already PDK/recipe-resolved
    # `sizing` — so baseline and variant agree exactly on every shared parameter.
    variant_sizing = {**variant_default_sizing, **sizing}

    if knob in cap.knobs:
        base_deck = base_render_sweep(sizing, corner=corner, knob=knob, metric=metric, points=points)
        variant_deck = variant_render_sweep(variant_sizing, corner=corner, knob=knob, metric=metric,
                                            points=points)
        if profile is not None:
            base_deck = substitute_devices(base_deck, profile)
            variant_deck = substitute_devices(variant_deck, profile)
        base_series = _measure(runner, engine, profile, base_deck).get(knob.lower(), [])
        variant_series = _measure(runner, engine, profile, variant_deck).get(knob.lower(), [])
        return _paired_series(base_series, variant_series)

    # NOTE: deliberately reads TEMPLATES' own ORDERED knobs list, not `cap.knobs` (a frozenset —
    # `next(iter(...))` over a set of strings is NOT order-stable across hash-randomized processes;
    # the reference knob must be deterministic run-to-run, e.g. always "Cc" for the OTA).
    ordered_knobs = (TEMPLATES.get(cap.topology_class) or {}).get("knobs") or []
    ref_knob = next(iter(ordered_knobs), None)
    if ref_knob is None:
        raise ValueError(
            f"intervention {spec.id!r}: base topology {cap.topology_class!r} has no native knob to "
            f"hold a reference point (knob {knob!r} exists only on the variant)"
        )
    ref_points = [str(sizing.get(ref_knob, "0"))]
    base_deck = base_render_sweep(sizing, corner=corner, knob=ref_knob, metric=metric, points=ref_points)
    variant_deck = variant_render_sweep(variant_sizing, corner=corner, knob=knob, metric=metric,
                                        points=points)
    if profile is not None:
        base_deck = substitute_devices(base_deck, profile)
        variant_deck = substitute_devices(variant_deck, profile)
    base_ref = _measure(runner, engine, profile, base_deck).get(ref_knob.lower(), [])
    variant_series = _measure(runner, engine, profile, variant_deck).get(knob.lower(), [])
    if not base_ref:
        return {"delta": [], "ratio": []}
    return _broadcast_paired_series(base_ref[0][1], variant_series)


def _recipe_requires_mismatch(recipe: VerificationRecipe) -> bool:
    """True iff `recipe` requires per-device MOS mismatch statistics (spec E1b §4-I1 requirement 3):
    mc_runs set, an explicit mismatch-corner request, or any sweep whose analysis is 'mc'/'pelgrom'
    (both axes are mismatch-only — the Pelgrom branch always renders under MM_CORNER_TOKEN
    regardless of what conditions.corner literally says, so it must be caught here too)."""
    if getattr(recipe.conditions, "mc_runs", None):
        return True
    if getattr(recipe.conditions, "corner", None) == MM_CORNER_TOKEN:
        return True
    return any(sw.get("analysis") in ("mc", "pelgrom") for sw in recipe.sweeps)


def _scale_geometry_value(raw: str, factor: float) -> str:
    """Scale a Pelgrom-area W-style sizing value by `factor`, preserving any SPICE unit suffix (spec
    E1b §4-I1 requirement 5). ihp's PDK_SIZING_OVERRIDES ship EXPLICITLY u-suffixed geometry (e.g.
    "8u") — a plain `float(raw) * factor` (the pre-E1b behavior) raises ValueError on those, which
    is exactly the wrong-unit area sweep the STOP rule exists to catch. Reuses runner.parse_unit for
    the magnitude and re-emits with the SAME suffix; a bare (sky130/gf180-style) numeric value has no
    suffix to preserve and scales exactly as before."""
    raw = raw.strip()
    value = parse_unit(raw)   # magnitude in base SI units ("8u" -> 8e-6, "4" -> 4.0); also validates
    if raw and raw[-1] in _UNIT and not raw[-1].isdigit():
        suffix = raw[-1]
        return f"{(value / _UNIT[suffix]) * factor:g}{suffix}"
    return f"{value * factor:g}"


def run_recipe(
    recipe: VerificationRecipe,
    runner=None,
    *,
    corpus=None,
    project: bool = False,
    config=None,
) -> ExecutionResult:
    """Execute a recipe end-to-end and return verdicted claim-cards (+ optional corpus store /
    pure projection).

    The runner is selected BY ENGINE (Tier-B §1c): `build.engine` (default: the template's engine,
    default ngspice) picks the runner from the engine registry. `runner` is an optional OVERRIDE —
    mainly tests injecting a fake — and must expose `measure(deck) -> {series_key: [(x, y), ...]}`.
    """
    cap = capability_for(recipe.topology_class)
    # honor the recipe's explicit template_ref so ONE topology_class can carry multiple measurement
    # templates (e.g. ota_5t: analog AC gain + ota5t_offset_mc + corner GBW); default to the class's.
    template_ref = (recipe.build or {}).get("template_ref") or cap.template_ref
    renderers = RENDERERS.get(template_ref)
    if renderers is None:
        raise UnsupportedTemplate(
            f"no renderer for template_ref '{template_ref}' "
            f"(class {recipe.topology_class}); have {sorted(RENDERERS)}"
        )
    render_sweep, render_cell, default_sizing = renderers

    engine = (recipe.build or {}).get("engine") or engine_for_template(template_ref)
    if runner is None:
        runner = runner_for_engine(engine)   # pick the engine's runner; a render-only engine raises
    from .runner import NgspiceRunner
    if config is not None and isinstance(runner, NgspiceRunner):
        from openclaw_brain.egress import effective_egress
        runner = NgspiceRunner(image=runner.image, workdir=runner.workdir,
                               docker=runner.docker, egress=effective_egress(config))

    # PDK profile selection (§5-I1 requirement 4): ngspice-only (digital has no PDK). Resolving +
    # validating the profile HERE — before any render or sim — is what makes an unknown pdk raise
    # before any sim (§5-I1 requirement 2). Default ("no pdk specified") resolves to "sky130A", whose
    # profile is the identity substitution, so unset-pdk behavior is byte-identical to pre-E1.
    pdk_key = _resolve_pdk(recipe) if engine == "ngspice" else "n/a"
    profile: PDKProfile | None = get_profile(pdk_key) if engine == "ngspice" else None

    # Mismatch-unavailable guard (spec E1b §4-I1 requirement 3): raise BEFORE any sizing/render/sim
    # if this recipe requires per-device mismatch statistics on a profile that has none (verified
    # ground truth — pdks.py module docstring). substitute_devices carries the same guard as
    # defense-in-depth, but checking here catches it before ANY compute, not just before the sim.
    if profile is not None and _recipe_requires_mismatch(recipe):
        require_mismatch_available(profile)

    # Unknown-intervention guard (spec E3-I1 §4/§5, test requirement "unknown intervention id raises
    # before sim"): resolve EVERY 'intervention'-analysis sweep's id up front, before any render/sim —
    # mirroring the mismatch guard above rather than discovering a bad id mid-way through a batch that
    # may already have run other claims' sims. Also catch a wrong-topology_class authoring error here
    # (an intervention registered for a DIFFERENT base class than this recipe's own) before any sim.
    for sw in recipe.sweeps:
        if sw.get("analysis") == "intervention":
            iv_spec = get_intervention(sw.get("intervention"))
            if iv_spec.base_topology_class != recipe.topology_class:
                raise ValueError(
                    f"intervention {iv_spec.id!r} is registered for base_topology_class "
                    f"{iv_spec.base_topology_class!r}, not this recipe's {recipe.topology_class!r}"
                )

    sizing = _resolve_sizing(recipe, default_sizing, profile.pdk if profile else pdk_key)
    corner = getattr(recipe.conditions, "corner", "n/a")   # engine-safe: DigitalUnits has no PVT corner

    # gm/ID-driven input-pair sizing (OTA, focused scope): set gm1 (hence GBW) by inversion level.
    sizing_prediction = None
    notes: list[str] = []
    if (recipe.sizing or {}).get("method") == "gmid_lookup" and cap.template_ref == "miller_ota_ac":
        sizing, sizing_prediction = _size_input_pair_from_gmid(recipe, sizing, runner, corner)
        notes.append(sizing_prediction["note"])

    # Run one deck per (knob, metric, points); re-key each series under the claims' series_refs.
    run_cache: dict[tuple, Series] = {}
    canonical: dict[str, Series] = {}
    flagged: dict[str, str] = {}      # route_key -> reason (an invalid series FLAGs, never judged)
    testbenches: dict[str, str] = {}  # route_key -> rendered sweep deck (persisted for reproducibility, I5)
    for card in recipe.claim_cards:
        knob = card.mechanism.knob
        metric = card.mechanism.metric
        points = _points_for(recipe, knob, metric)
        mode = _sweep_mode(recipe, knob)
        # intervention (spec E3-I1 §2): fold the id into the cache key so a paired baseline/variant
        # run never collides with an ordinary sweep sharing the same (knob, metric, points) — cheap
        # since iv_id is None for every non-intervention card.
        iv_id = _intervention_id_for(recipe, knob) if mode == "intervention" else None
        cache_key = (knob, metric, tuple(points) if points else None, iv_id)
        if cache_key not in run_cache:
            if mode == "corner":
                # one deck per process corner; aggregate (corner_idx, value) for the corner kind.
                # NOTE (§5-I1 scope): E1's pilot is nominal-only off sky130 (spec §4) — a non-sky130
                # profile's substitute_devices collapses EVERY per-corner deck onto the SAME nominal
                # lib_section, which would defeat corner differentiation. Not guarded here; unsupported.
                pts = []
                for ci, corner_name in enumerate(recipe.conditions.corners or [corner]):
                    deck = render_sweep(sizing, corner=corner_name, knob=knob, metric=metric, points=points)
                    if profile is not None:
                        deck = substitute_devices(deck, profile)
                    s = _measure(runner, engine, profile, deck)
                    ys = s.get(knob.lower(), [])
                    pts.append((float(ci), ys[-1][1] if ys else float("nan")))
                run_cache[cache_key] = pts
            elif mode == "pelgrom":
                # UNIFORM area scaling (scale every device width by the area factor) so every Vth mismatch
                # scales with area; reduce each area's inner MC to σ and emit (area, σ). The elasticity kind
                # then judges the log-log SCALING EXPONENT — a Pelgrom-TYPE power law (σ decreases as a power
                # of area). The exponent is model/node-specific (sky130's open model ~ -0.375, not the
                # textbook -0.5), so this honestly certifies the measured scaling, not an idealized constant.
                # seed_offset per AREA (spec §4-I1 requirement 4): each area's inner MC run-index
                # range (0..mc_runs-1) gets its own MC_SEED_AREA_STRIDE-sized block, so no two areas
                # in the SAME elasticity fit ever replay the identical agauss() draw sequence.
                pts = []
                for area_idx, area in enumerate(recipe.conditions.areas or [1.0]):
                    a_sizing = dict(sizing)
                    for _k in list(a_sizing):
                        if _k.startswith("W"):
                            a_sizing[_k] = _scale_geometry_value(a_sizing[_k], area)
                    deck = render_sweep(a_sizing, corner=MM_CORNER_TOKEN, knob=knob, metric=metric,
                                        mc_runs=recipe.conditions.mc_runs or 200,
                                        seed_offset=area_idx * MC_SEED_AREA_STRIDE)
                    if profile is not None:
                        deck = substitute_devices(deck, profile)
                    s = _measure(runner, engine, profile, deck)
                    ys = [y for _, y in s.get(knob.lower(), [])]
                    sigma = pstdev(ys) if len(ys) > 1 else 0.0
                    pts.append((float(area), sigma))
                run_cache[cache_key] = pts
            elif mode == "intervention":
                # E3-I1 §2/§3: paired baseline/variant run, IDENTICAL sizing (see _run_intervention_
                # pair's docstring for the native-knob vs broadcast-reference split). `spec` was
                # already validated (raises before ANY sim otherwise) by the early guard above.
                spec = get_intervention(iv_id)
                run_cache[cache_key] = _run_intervention_pair(
                    spec, cap, sizing, corner, knob, metric, points, engine, profile, runner,
                )
            else:   # "mc" (the renderer self-loops) or analog — one deck, the existing path
                kwargs = {"mc_runs": recipe.conditions.mc_runs} if (mode == "mc" and recipe.conditions.mc_runs) else {}
                deck = render_sweep(sizing, corner=corner, knob=knob, metric=metric, points=points, **kwargs)
                if profile is not None:
                    deck = substitute_devices(deck, profile)
                testbenches[f"{knob.lower()}_{metric}"] = deck
                run_cache[cache_key] = _measure(runner, engine, profile, deck).get(knob.lower(), [])
        # The executor OWNS the claim→series binding; it does not trust the author's series_ref to be
        # unique. A live author labels by sweep (e.g. "Cc_sweep"), which collides across metrics — two
        # claims on the same knob but different metrics would overwrite one series and misjudge. Rewrite
        # series_ref to the canonical routing key so judging + the stored card are unambiguous; the
        # author's intent (which sweep) is preserved in `knob`. Intervention claims route to the
        # DERIVED series (spec §2's f"{knob}_{metric}_{delta|ratio}__{id}" key), never the raw baseline
        # or variant series — the oracle only ever sees the derived comparison, exactly Q1's bet.
        if mode == "intervention":
            derive = _derive_kind_for(recipe, knob, metric)
            route_key = f"{knob.lower()}_{metric}_{derive}__{iv_id}"
            series_for_card = run_cache[cache_key].get(derive, [])
        else:
            route_key = f"{knob.lower()}_{metric}"
            series_for_card = run_cache[cache_key]
        card.mechanism.series_ref = route_key
        canonical[route_key] = series_for_card
        # C1: a truncated/empty series must FLAG, never be judged. A failed `.meas` drops points
        # (runner.parse_rdata skips the non-finite reset value), so fewer points back than requested
        # means a sim failure — certifying that curve as VERIFIED would teach a number off a partial run.
        reason = _series_invalid_reason(series_for_card, points)
        if reason and route_key not in flagged:
            flagged[route_key] = reason
            notes.append(f"({knob},{metric}) flagged: {reason}")

    # Judge every claim against its routed series (oracle is deterministic; mechanism never certified).
    oracle = ClaimOracle()
    for card in recipe.claim_cards:
        rk = card.mechanism.series_ref
        if rk in flagged:
            card.verdict, card.verdict_note = VerdictClass.FLAGGED, flagged[rk]
            continue
        try:
            oracle.judge(card, canonical)
        except _EXPECTED_JUDGE_FAILURES as exc:
            # Missing/degenerate measurement data reaching the oracle (a metric that never got
            # measured, or math that's undefined on the series it got) — one bad claim FLAGs; it
            # must never abort the batch (I1). Expected shape, not a code bug.
            card.verdict, card.verdict_note = (
                VerdictClass.FLAGGED,
                f"unjudgeable ({type(exc).__name__}): {exc!r}",
            )
            notes.append(f"claim '{card.id}' flagged: {exc!r}")
        except Exception as exc:      # one bad claim FLAGs; it must never abort the batch (I1)
            # Anything outside _EXPECTED_JUDGE_FAILURES is much more likely a genuine code
            # regression in oracle.py (or a caller-side contract break) than a data/measurement
            # problem. Still must never abort the batch (I1) — but the verdict_note is marked
            # UNEXPECTED (grep-distinguishable from an ordinary sim/data FLAG) and logged at
            # ERROR, so a corpus-wide wave of these reads as "go look at the oracle", not noise.
            card.verdict, card.verdict_note = (
                VerdictClass.FLAGGED,
                f"unjudgeable (UNEXPECTED {type(exc).__name__}): {exc!r}",
            )
            notes.append(f"claim '{card.id}' flagged (UNEXPECTED {type(exc).__name__}): {exc!r}")
            logger.error(
                "run_recipe: UNEXPECTED %s from oracle.judge() on claim %r — likely an oracle "
                "code regression, not a data/measurement issue: %r",
                type(exc).__name__, card.id, exc,
            )

    # Stamp scope-honesty (ADR 4-5): a verdict is nominal/single-corner (analog) or functional (digital)
    # unless said otherwise; mismatch-dominated classes also name their untested dominant axis.
    basis = "functional" if engine == "iverilog" else "physical-nominal"
    scope = summarize_scope(recipe.conditions)
    risk = _DOMINANT_RISK.get(recipe.topology_class)
    _digital = getattr(recipe.conditions, "kind", None) == "digital"
    # E2a-I2 (docs/superpowers/specs/2026-07-05-e2a-analytic-envelope.md §2/§4): the analytic-envelope
    # setup for the per-card stamping loop below. `envelope_pdk` mirrors the specimen's own pdk stamp
    # (profile.pdk when a profile was resolved, else the sky130A default); `envelope_op_cache` is keyed
    # (topology_class, sizing, pdk, corner) so two pilot claims sharing the SAME op-point (ota5t_av0 +
    # ota5t_gbw both read gm1/gds_n/gds_p at the SAME CL=1p nominal) cost exactly ONE extra `.op` sim,
    # not two — "be economical" (this increment's own mandate) without changing the per-claim contract.
    envelope_pdk = profile.pdk if profile is not None else "sky130A"
    envelope_op_cache: dict = {}
    for card in recipe.claim_cards:
        card.engine = engine
        card.basis = basis
        card.scope = scope
        # E3-I1 §2/§4 scope-honesty design point: the intervention id + its named idealization must be
        # UNDETACHABLE from the verdict surface without touching the Conditions closed union,
        # summarize_scope's total-function tripwire, or ClaimCard's shape. The executor (never trusting
        # the author, same discipline as the series_ref rewrite above) additively stamps BOTH:
        #   (i)  `grounds` gets an "intervention:<id>" entry (spec §2: "the derived-claim card carries
        #        the intervention id in its claim id and grounds" — id is the author's job at seed-
        #        recipe time; grounds is enforced here so it is undetachable even if an author forgot).
        #   (ii) `scope` (already `dict[str, Any]` — additive keys are not a shape change) gets
        #        `intervention`/`idealization` merged in AFTER summarize_scope runs — summarize_scope
        #        itself, and the Conditions union it switches on, are UNTOUCHED; this is a post-hoc
        #        enrichment of the returned dict, not a new discriminated-union branch.
        # projection.py already does `json.dumps(c.scope)` generically (no per-key allowlist) and
        # agent.py's why() already round-trips `scope` back to a dict, so both keys ride the EXISTING
        # plumbing to the verdict surface with zero projection.py/agent.py changes — why()'s
        # _scope_inline rendering of them is left to I2, per spec.
        iv_id_for_card = _intervention_id_for(recipe, card.mechanism.knob)
        if iv_id_for_card:
            iv_spec = get_intervention(iv_id_for_card)   # pre-validated by the early guard; safe here
            ground_tag = f"intervention:{iv_id_for_card}"
            if ground_tag not in card.grounds:
                card.grounds = [*card.grounds, ground_tag]
            card.scope = {**card.scope, "intervention": iv_id_for_card, "idealization": iv_spec.idealization}
        # E2a-I2: the analytic envelope, a SECOND independent instrument, stamped ADDITIVELY and
        # BEST-EFFORT (spec §2/§4 — deliverable 1). Gate is a plain dict-key membership check on
        # envelope.REGISTRY keyed (topology_class, metric); it matches EXACTLY the spec §3 pilot cells
        # (cc_gbw, ota5t_gbw, ota5t_av0, cs_av0) and none of the ~20 other claims — no OP-dump render,
        # no extra sim, for anything else. ANY failure below (a bad sizing lookup, a sim error, a
        # missing sweep point, an unavailable runner/image) degrades SILENTLY to "no envelope tag" —
        # never touches `card.verdict`, never raises out of run_recipe. It NEVER reads the oracle's
        # verdict either — `measured_pt` is the SAME (x, y) series the oracle already judged, read
        # independently here only to pick the ONE nominal-sizing point envelope_check compares.
        if engine == "ngspice" and (card.topology_class, card.mechanism.metric) in ENVELOPE_REGISTRY:
            try:
                sizing_key = _ENVELOPE_KNOB_SIZING_KEY.get(card.mechanism.knob, card.mechanism.knob)
                nominal_x = parse_unit(str(sizing[sizing_key]))
                series = canonical.get(card.mechanism.series_ref) or []
                if not series:
                    raise ValueError("no measured series to compare the envelope against")
                measured_pt = min(series, key=lambda pt: abs(pt[0] - nominal_x))
                if abs(measured_pt[0] - nominal_x) > 1e-6 * max(abs(nominal_x), 1e-30):
                    raise ValueError(
                        f"no sweep point at the nominal {sizing_key}="
                        f"{sizing[sizing_key]!r} to anchor the envelope comparison")
                cache_key = (card.topology_class, tuple(sorted(sizing.items())), envelope_pdk, corner)
                if cache_key not in envelope_op_cache:
                    try:
                        envelope_op_cache[cache_key] = measure_op_quantities(
                            card.topology_class, sizing, pdk=envelope_pdk, corner=corner, runner=runner)
                    except Exception as op_exc:   # sim error / missing instance / unavailable runner
                        envelope_op_cache[cache_key] = op_exc
                op_quantities = envelope_op_cache[cache_key]
                if isinstance(op_quantities, Exception):
                    raise op_quantities
                verdict = envelope_check(card.topology_class, card.mechanism.metric, envelope_pdk,
                                          measured_pt[1], op_quantities)
                card.scope = stamp_envelope_scope(card.scope, verdict)
            except Exception as exc:
                notes.append(f"envelope check skipped for claim {card.id!r}: {exc!r}")
        # a certified statistical/corner verdict means the mismatch/process axis IS now tested ->
        # clear the "DOMINANT RISK NOT TESTED" banner for that class (Stat-QT pays down the scope-debt).
        if card.mechanism.quant.kind in ("statistical", "corner") and \
                card.verdict in (VerdictClass.VERIFIED, VerdictClass.VERIFIED_WITH_CAVEAT):
            card.dominant_risk_untested = None
        else:
            card.dominant_risk_untested = (
                (getattr(recipe.conditions, "stimulus_untested", None)
                 or _DOMINANT_RISK.get(recipe.topology_class))   # preserve existing digital banner (§2.1)
                if _digital else risk)

    cell_netlist = render_cell(sizing, corner=corner)
    if profile is not None:
        cell_netlist = substitute_devices(cell_netlist, profile)  # sky130A: no-op (byte-identical)
    specimen = Specimen(
        topology_class=recipe.topology_class,
        netlist=cell_netlist,
        testbench="",
        testbenches=testbenches,        # the rendered sweep decks (I5: reproducible, not in spec_id)
        pdk=profile.pdk if profile is not None else "n/a",     # digital has no PDK; stamped from the profile
        tool={"ngspice": "ngspice-46", "iverilog": "iverilog-13"}.get(engine, engine),
        role_map=(recipe.build or {}).get("role_map", {}) or {},
        claim_cards=recipe.claim_cards,
    )

    result = ExecutionResult(
        recipe=recipe, specimen=specimen, canonical=canonical,
        claim_cards=recipe.claim_cards, runs=len(run_cache),
        sizing_prediction=sizing_prediction, notes=notes,
    )
    if corpus is not None:
        result.spec_dir = corpus.store(specimen)   # sets specimen.spec_id in place
        result.spec_id = specimen.spec_id
    if project:
        result.projection = project_specimen(specimen)
    return result
