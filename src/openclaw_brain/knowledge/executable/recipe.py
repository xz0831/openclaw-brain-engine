"""① Recipe-authoring (SPEC §5.1) — the irreducible-intelligence step of the executable substrate.

This is the ONE place per topology_class where frontier judgment is spent (amortized over every
specimen of that class): the model decides WHICH sweeps, claim-cards, sizing targets, and nominal
measurement conditions characterize a topology. Everything downstream (the §5 ② executor) is
deterministic code — gmid.size → templates.render → runner → oracle.judge → corpus.store →
projection.

The golden-split (§11) showed deepseek-v4-flash authors correct recipes, but it ALSO showed a small
model gets PDK device details wrong (`sky130_fd_sc_hd` std-cell vs `sky130_fd_pr` analog). So
EXECUTABLE CORRECTNESS IS NOT TRUSTED TO THE AUTHOR MODEL — it is enforced structurally here:

  - `build.template_ref` is SNAPPED to the registry's known-good template for the class (the model
    never gets to invent a netlist) — the T-template guarantee made concrete.
  - sweeps the template cannot execute (unknown analysis/knob/metric for that class) are DROPPED and
    RECORDED in `build.unexecutable_sweeps` — never silently kept — so the executor only ever runs
    what the validated template supports.
  - R1 conditions are mandatory: the schema enforces it on the structured path; the raw-fallback
    path injects a nominal TT default (the "no corner stated ⇒ TT" rule).

Authoring is model-independent by construction. The caller supplies the model chain (production
default = the extraction stage chain, deepseek-v4-flash primary — SPEC §5.2). This module imports
no provider/auth, mirroring answerer.py's separation.

See docs/specs/SPEC_EXECUTABLE_CIRCUIT_SUBSTRATE.md §5.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from openclaw_brain.llm.resilience import invoke_with_resilience

from .models import VerificationRecipe
from .runner import parse_unit
from .templates import TEMPLATES

logger = logging.getLogger(__name__)


# ── Template capability (read from the templates.py registry SSOT) ──


@dataclass(frozen=True)
class TemplateCapability:
    """What the validated template for a topology_class can actually build and measure.
    Constraints the recipe is forced to obey so it stays executable."""

    topology_class: str
    template_ref: str
    analyses: frozenset[str]
    knobs: frozenset[str]
    metrics: frozenset[str]
    default_sizing: dict[str, str]
    # S3-inc2a §4: declared {knob: (lo, hi)} operating ranges — additive (a knob absent here gets no
    # clamping at all). Populated from the registry's own `knob_ranges` entry (templates.py); {} for
    # every template that doesn't declare one (byte-identical enforcement behavior otherwise).
    knob_ranges: dict[str, tuple[float, float]]

    @property
    def default_vdd(self) -> float:
        try:
            return float(self.default_sizing.get("VDD", 1.8))
        except (TypeError, ValueError):
            return 1.8


class UnknownTopologyClass(Exception):
    """No validated template exists for the requested topology_class (cannot author safely)."""


def capability_for(topology_class: str) -> TemplateCapability:
    """Resolve the executable capability for a class from the templates registry."""
    entry = TEMPLATES.get(topology_class)
    if not entry:
        raise UnknownTopologyClass(
            f"no template registered for '{topology_class}'; "
            f"known: {sorted(TEMPLATES)}"
        )
    return TemplateCapability(
        topology_class=topology_class,
        template_ref=entry["template_ref"],
        analyses=frozenset(entry.get("analyses", [])),
        knobs=frozenset(entry.get("knobs", [])),
        metrics=frozenset(entry.get("metrics", [])),
        default_sizing=dict(entry.get("default_sizing", {})),
        knob_ranges={k: tuple(v) for k, v in entry.get("knob_ranges", {}).items()},
    )


# ── Prompt ──


_SYSTEM_PROMPT = """You author a VERIFICATION RECIPE for one analog-circuit topology class.

A verification recipe is a protocol that a DETERMINISTIC executor runs mechanically: it instantiates
a known-good netlist template, sizes it, runs the sweeps you specify in ngspice, and lets an oracle
falsify each claim-card against the measured data. You supply the JUDGMENT (which sweeps and
claim-cards characterize this topology, what sizing targets and nominal conditions); code supplies
execution. You do NOT write a netlist — a validated template is fixed for this class.

HARD CONSTRAINTS (the executor cannot run a recipe that breaks these):
- Use ONLY the allowed analyses / knobs / metrics listed for this class. Do not invent device names,
  PDK libraries, or knobs the template does not expose.
- conditions are MANDATORY (R1): give a single nominal operating point — corner (tt|ss|ff|sf|fs),
  temp_c, vdd. If the source states no corner, use tt at 27C and the nominal vdd.
- Every claim-card carries a falsifiable `quant` assertion. `kind` is one of:
    direction            (sign "+"/"-": metric moves which way as knob increases)
    direction_to_optimum (sign: metric moves toward an optimum)
    elasticity           (band [lo,hi]: d ln(metric)/d ln(knob) lies in this band)
    value                (value + tol: metric equals value within tol)
    invariance           (cov_max: metric is ~constant across the sweep)
  Pick the WEAKEST kind the textbook actually supports (prefer `direction`/`elasticity` over `value`).
- `narrative` is the mechanism prose. It is INTERPRETIVE and never machine-certified — keep claims you
  are unsure of in the narrative, never as the quant.

Return STRICT JSON only (no markdown fences) with keys:
  topology_class, source_ref,
  build: {method, template_ref, role_map},
  sizing: {method, targets, seed},
  conditions: {corner, temp_c, vdd, cl_f},
  sweeps: [ {analysis, knob, points, measure: [metric,...]} ],
  claim_cards: [ {id, metric, knob, series_ref, quant: {kind, sign, target, band, value, tol, cov_max}, narrative} ]
"""


_FOCUS_INSTRUCTION = (
    "author claim-card(s) certifying THIS metric-vs-knob relationship; CHOOSE the physically correct "
    "quant kind yourself — in particular, if the metric does not move with this knob, the correct "
    "certifiable claim is 'invariance' (that independence is itself teachable knowledge), never force "
    "a direction; at most one companion claim-card beyond the focus if pedagogically essential."
)


def _build_human_prompt(cap: TemplateCapability, source_text: str, focus: dict | None = None) -> str:
    spec = {
        "topology_class": cap.topology_class,
        "allowed_analyses": sorted(cap.analyses),
        "allowed_knobs": sorted(cap.knobs),
        "allowed_metrics": sorted(cap.metrics),
        "nominal_vdd": cap.default_vdd,
        "source_text": source_text or "(no source text supplied — author from canonical knowledge of this topology)",
    }
    if focus:
        spec["focus_probe"] = {
            "metric": focus.get("metric"),
            "knob": focus.get("knob"),
            "kind_hint": focus.get("kind_hint"),
        }
        spec["focus_instruction"] = _FOCUS_INSTRUCTION
    return json.dumps(spec, ensure_ascii=False)


# ── Public API ──


async def author_recipe(
    *,
    topology_class: str,
    source_text: str = "",
    model_chain: Sequence[BaseChatModel],
    resilience_config,
    auth_refresh=None,
    focus: dict | None = None,
) -> VerificationRecipe:
    """Author an executable-enforced VerificationRecipe for a topology_class.

    Structured-output first (Anthropic/OpenAI/OpenRouter); on failure, raw invocation through the
    full resilience chain + normalize. The returned recipe is always executable: template_ref snapped
    to the registry, unsupported sweeps dropped+recorded, R1 conditions guaranteed.

    `focus` (optional: {"metric", "knob", "kind_hint"|None}) names the ONE planned (metric, knob) cell
    this call exists to certify — without it, three same-class planning targets would author ~identical
    generic recipes (the planning label never reaching the model). When set, the prompt asks the author
    to certify that specific relationship, but leaves WHICH quant kind is physically correct to the
    model's judgment (never forcing a direction where invariance is the truth). `focus=None` keeps the
    prompt byte-identical to the pre-focus generic behavior (backward compat).
    """
    cap = capability_for(topology_class)  # raises UnknownTopologyClass before any LLM spend
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_build_human_prompt(cap, source_text, focus)),
    ]

    primary = model_chain[0]
    try:
        structured = primary.with_structured_output(VerificationRecipe)
        coro = structured.ainvoke(messages)
        timeout = resilience_config.request_timeout_s
        result = await asyncio.wait_for(coro, timeout=timeout) if timeout > 0 else await coro
        recipe = VerificationRecipe.model_validate(result)
        return enforce_executable(recipe, cap)
    except Exception:
        logger.info("Structured recipe output failed on primary model, using raw + normalize")

    from openclaw_brain.knowledge.reasoning.normalize import extract_json

    # Raw path: a model can return an API-level SUCCESS whose content is still not JSON
    # (observed live: deepseek returned 561 chars of prose — invoke_with_resilience only
    # falls over on API errors, so a garbage-but-successful response used to raise here
    # and triage the whole target). Walk the chain per-model: API resilience per model
    # stays inside invoke_with_resilience; JSON-extraction failure advances the chain.
    last_exc: Exception | None = None
    for i, model in enumerate(model_chain):
        try:
            response = await invoke_with_resilience(
                [model], messages, resilience_config, auth_refresh=auth_refresh,
            )
            raw_text = response.content if hasattr(response, "content") else str(response)
            raw_json = extract_json(raw_text)
            normalized = _normalize_recipe_payload(raw_json, cap)
            recipe = VerificationRecipe(**normalized)
            return enforce_executable(recipe, cap)
        except Exception as exc:
            last_exc = exc
            if i + 1 < len(model_chain):
                logger.info(
                    "recipe(%s): raw authoring failed on chain model %d (%s) — trying next",
                    cap.topology_class, i, type(exc).__name__,
                )
    raise last_exc if last_exc is not None else RuntimeError("empty model chain")


# ── Executable enforcement (deterministic; the load-bearing structural guarantee) ──


def enforce_executable(recipe: VerificationRecipe, cap: TemplateCapability) -> VerificationRecipe:
    """Force the recipe to what the validated template can actually run.

    1. topology_class + build.template_ref are snapped to the registry (the model does not pick the
       netlist). 2. Sweeps with an unknown analysis/knob/metric for this class are dropped and the
       dropped set is recorded in build.unexecutable_sweeps (honest provenance — never silent).
    3. recipe.sizing (and its seed/targets sub-containers) is sanitized to the dict shape the executor
       (executor.py::_resolve_sizing / _gmid_target) requires — never silently coerced away, always
       recorded in build.sizing_dropped when something didn't fit.
    4. recipe.conditions is coerced to AnalogPVT — authoring is analog/ngspice-only (v1), but the
       STRUCTURED-output path bypasses _normalize_conditions (raw-fallback only), and a live run
       authored DigitalUnits conditions for the Miller OTA (clock/stimulus fields; no corner ->
       `.lib` section lost -> dead deck -> FLAGGED empty series). Non-analog conditions are replaced
       with the nominal AnalogPVT (tt/27C/cap VDD) and recorded in build.conditions_coerced;
       every claim card's conditions are re-pointed likewise.
    5. build.role_map is sanitized to a dict (a live structured-path run emitted prose —
       "M1,M2: diff pair; ..." — which explodes Specimen validation at the simulate stage);
       a non-dict is dropped to {} and recorded in build.role_map_dropped.
    6. Each claim-card inherits the class + the recipe's conditions if it lacks them.
    """
    recipe.topology_class = cap.topology_class
    recipe.build = dict(recipe.build or {})
    recipe.build["method"] = "template"
    recipe.build["template_ref"] = cap.template_ref
    recipe.sizing = _sanitize_sizing(recipe.sizing, recipe.build)

    # (4) Conditions coercion — see docstring. Import here to avoid a module cycle at import time.
    from .conditions import AnalogPVT
    if not isinstance(recipe.conditions, AnalogPVT):
        raw = recipe.conditions
        recipe.build["conditions_coerced"] = (
            raw.model_dump() if hasattr(raw, "model_dump") else repr(raw)
        )
        recipe.conditions = AnalogPVT(corner="tt", temp_c=27.0, vdd=cap.default_vdd)
        for card in recipe.claim_cards:
            card.conditions = recipe.conditions
        logger.warning(
            "recipe(%s): non-analog conditions coerced to nominal AnalogPVT (recorded in "
            "build.conditions_coerced)", cap.topology_class,
        )

    # (5) role_map sanitation — see docstring.
    rm = recipe.build.get("role_map")
    if rm is not None and not isinstance(rm, dict):
        recipe.build["role_map_dropped"] = str(rm)[:500]
        recipe.build["role_map"] = {}
        logger.warning(
            "recipe(%s): non-dict build.role_map dropped (recorded in build.role_map_dropped)",
            cap.topology_class,
        )

    kept: list[dict] = []
    dropped: list[dict] = []
    points_clamped: list[dict] = []
    for sw in recipe.sweeps:
        sw = _materialize_sweep_points(sw)
        sw, clamp_record = _clamp_sweep_points(sw, cap)
        if clamp_record is not None:
            points_clamped.append(clamp_record)
            if not sw.get("points"):
                # every authored point was out of the declared range — this is NOT a silent empty
                # sweep (which would fall back to the renderer's own default and hide the clamp
                # entirely, since `sw.get("points")` on `[]` is falsy — executor._points_for would
                # return None and quietly re-adopt the renderer default): it is unexecutable, with
                # the reason on record (S3-inc2a §4).
                dropped.append({
                    "sweep": sw,
                    "reasons": [
                        f"all authored points clamped out of range for knob {sw.get('knob')!r} "
                        f"(declared range {clamp_record['range']})"
                    ],
                })
                continue
        reasons = _sweep_unexecutable_reasons(sw, cap)
        if reasons:
            dropped.append({"sweep": sw, "reasons": reasons})
        else:
            kept.append(sw)
    recipe.sweeps = kept
    if dropped:
        recipe.build["unexecutable_sweeps"] = dropped
        logger.warning(
            "recipe(%s): dropped %d unexecutable sweep(s): %s",
            cap.topology_class, len(dropped),
            [d["reasons"] for d in dropped],
        )
    if points_clamped:
        recipe.build["points_clamped"] = points_clamped
        logger.warning(
            "recipe(%s): clamped out-of-range sweep point(s) for %d knob(s): %s",
            cap.topology_class, len(points_clamped),
            [(c["knob"], c["dropped"]) for c in points_clamped],
        )

    for card in recipe.claim_cards:
        card.topology_class = cap.topology_class
    return recipe


def _sanitize_sizing(sizing: Any, build: dict) -> dict[str, Any]:
    """Coerce recipe.sizing (and its `seed`/`targets` sub-containers) to the dict shape the executor
    requires — `executor.py::_resolve_sizing` does `(recipe.sizing or {}).get("seed").items()` and
    `_gmid_target` reads `sizing.targets` as a dict; an LLM-authored recipe carrying sizing (or its
    seed/targets) as a loose string ("use template defaults") crashes both at the simulate stage.

    Only the CONTAINER types (sizing itself, seed, targets) are enforced — values *inside* seed may be
    any scalar (the executor `str()`s them). Anything that didn't fit is dropped and recorded in
    `build["sizing_dropped"]`, never silently discarded (mirrors the unexecutable_sweeps provenance
    discipline). `method` is left as-is (a free-form string is fine there).
    """
    dropped: dict[str, Any] = {}
    if not isinstance(sizing, dict):
        dropped["sizing"] = sizing
        sizing = {}
    else:
        sizing = dict(sizing)
    for key in ("seed", "targets"):
        if key in sizing and not isinstance(sizing[key], dict):
            dropped[key] = sizing.pop(key)
    if dropped:
        build["sizing_dropped"] = dropped
    return sizing


def _materialize_sweep_points(sw: dict) -> dict:
    """Coerce an authored sweep's `points` into the explicit value list the executor expects.

    LLM authors legitimately specify sweeps as a COUNT plus a range — e.g.
    ``{"points": 5, "range": [2e-13, 5e-12], "log": true}`` — while
    ``executor._points_for`` iterates ``points`` as a list of values (the seed-recipe shape).
    A count+range is materialized here (log-spaced when ``log`` is truthy or the range spans
    >= 2 decades, else linear); a count without a usable range can't be honored, so ``points``
    is removed (the renderer's default sweep applies) and the raw value is recorded on the
    sweep as ``points_dropped`` — never silently kept to crash downstream.
    """
    points = sw.get("points")
    if points is None or isinstance(points, (list, tuple)):
        return sw
    sw = dict(sw)
    try:
        n = int(points)
    except (TypeError, ValueError):
        n = 0
    rng = sw.get("range") or sw.get("span")
    lo = hi = None
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        try:
            lo, hi = float(rng[0]), float(rng[1])
        except (TypeError, ValueError):
            lo = hi = None
    if n >= 2 and lo is not None and hi is not None and lo > 0 and hi > lo:
        log_spaced = bool(sw.get("log")) or (hi / lo >= 100)
        if log_spaced:
            vals = [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]
        else:
            vals = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
        sw["points"] = [f"{v:.6g}" for v in vals]
    else:
        sw["points_dropped"] = points
        sw.pop("points", None)
    return sw


def _clamp_sweep_points(sw: dict, cap: TemplateCapability) -> tuple[dict, dict | None]:
    """Drop authored sweep points outside a template-declared knob RANGE (S3-inc2a spec §4).

    A template may declare `{knob: (lo, hi)}` in its registry entry's `knob_ranges`
    (`templates.py`, read into `TemplateCapability.knob_ranges` by `capability_for`) — e.g. the OTA
    templates' new VDD/IREFV headroom knobs, whose authored sweep points could otherwise pick a
    physically-absurd operating point (the carried risk from the first `--apply` growth run: "VDD
    0->10V"). Points outside the declared range are dropped and returned as a `points_clamped` record
    (never silently discarded); a knob with NO declared range is untouched — additive, so every
    pre-existing template/knob's enforcement behavior is unaffected.

    Only ever touches an EXPLICIT, already-materialized points LIST (never the renderer's own
    default — a `None`/missing `points` means the author didn't specify one, and the template's own
    default is presumed already sane). Returns `(possibly-updated sw, clamp record or None)`.
    """
    knob = sw.get("knob")
    rng = cap.knob_ranges.get(knob)
    points = sw.get("points")
    if rng is None or not isinstance(points, list) or not points:
        return sw, None
    lo, hi = rng
    kept: list = []
    out_of_range: list = []
    for p in points:
        try:
            value = parse_unit(str(p))
        except ValueError:
            kept.append(p)          # an unparsable literal is not this guard's job to interpret
            continue
        (kept if lo <= value <= hi else out_of_range).append(p)
    if not out_of_range:
        return sw, None
    sw = dict(sw)
    sw["points"] = kept
    record = {"knob": knob, "range": [lo, hi], "dropped": out_of_range}
    return sw, record


def _sweep_unexecutable_reasons(sw: dict, cap: TemplateCapability) -> list[str]:
    """Why (if at all) this sweep cannot run on the class template. Empty list = executable."""
    reasons: list[str] = []
    analysis = sw.get("analysis")
    knob = sw.get("knob")
    metrics = sw.get("measure") or sw.get("metrics") or []
    if isinstance(metrics, str):
        metrics = [metrics]
    if analysis not in cap.analyses:
        reasons.append(f"analysis '{analysis}' not in {sorted(cap.analyses)}")
    if knob not in cap.knobs:
        reasons.append(f"knob '{knob}' not in {sorted(cap.knobs)}")
    bad = [m for m in metrics if m not in cap.metrics]
    if bad:
        reasons.append(f"metric(s) {bad} not in {sorted(cap.metrics)}")
    return reasons


# ── Raw-fallback normalization (coerce loose local-model JSON to the schema) ──


def _normalize_recipe_payload(raw: dict[str, Any], cap: TemplateCapability) -> dict[str, Any]:
    """Make a raw JSON payload constructable as a VerificationRecipe.

    Local models drop/flatten fields; we inject the mandatory ones (conditions) and nest flattened
    claim-cards. We do NOT fix executability here — enforce_executable() owns that after construction.
    """
    out = dict(raw or {})
    out["topology_class"] = cap.topology_class

    out["conditions"] = _normalize_conditions(out.get("conditions"), cap)

    cards = out.get("claim_cards") or []
    out["claim_cards"] = [
        _normalize_claim_card(c, cap, idx, out["conditions"])
        for idx, c in enumerate(cards)
        if isinstance(c, dict)
    ]
    return out


def _normalize_conditions(cond: Any, cap: TemplateCapability) -> dict[str, Any]:
    if not isinstance(cond, dict):
        return {"kind": "analog_pvt", "corner": "tt", "temp_c": 27.0, "vdd": cap.default_vdd}
    cond = dict(cond)
    # The recipe-authoring path is ngspice/analog only — stamp the discriminator so the bare dict
    # coerces into the closed Conditions union (AnalogPVT). Without `kind` the union can't discriminate.
    cond.setdefault("kind", "analog_pvt")
    cond.setdefault("corner", "tt")
    cond.setdefault("temp_c", 27.0)
    cond.setdefault("vdd", cap.default_vdd)
    # §5.1 envisions a PVT grid (corners[]/temps_c[]); the implemented Conditions is scalar nominal
    # (PVT cube is SPEC §12-deferred). Collapse a list to its first element so the model's grid still
    # constructs as the nominal point.
    for k in ("corner", "temp_c", "vdd"):
        if isinstance(cond.get(k), list) and cond[k]:
            cond[k] = cond[k][0]
    return cond


def _normalize_claim_card(
    c: dict, cap: TemplateCapability, idx: int, conditions: dict
) -> dict[str, Any]:
    card = dict(c)
    card.setdefault("id", f"{cap.topology_class}_claim_{idx}")
    card["topology_class"] = cap.topology_class
    card.setdefault("conditions", conditions)
    # Nest a flattened claim: {knob, metric, quant, narrative} at top level -> mechanism{...}.
    if "mechanism" not in card:
        card["mechanism"] = {
            "knob": card.get("knob", ""),
            "metric": card.get("metric", ""),
            "series_ref": card.get("series_ref") or card.get("knob", ""),
            "quant": card.get("quant", {}),
            "narrative": card.get("narrative"),
        }
    return card
