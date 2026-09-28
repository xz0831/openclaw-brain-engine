"""Pluggable engine registry (ADR Decision 1, Tier-B §1a). An engine is the ONLY structural axis:
`EngineSpec = {name, runner}` where `runner` is an object exposing `measure(artifact, timeout) -> series`
and `available()` (NgspiceRunner / VerilogRunner). The existing ngspice path is wrapped unchanged;
iverilog is the first non-ngspice member. A render-only engine (future PrimeSim: we draft the bench, the
user runs it on a licensed tool) sets `runner=None`.

`engine_for_template(template_ref)` reads the engine off the template registry (TEMPLATES); entries with
no explicit `engine` default to ngspice, so the 15 analog specimens are untouched. `runner_for_engine`
returns the engine's runner object — the seam run_recipe uses to pick a runner by engine when the caller
injects none. This is what makes the ADR's "engines 3..N touch only the registry" claim hold.
"""
from __future__ import annotations

from dataclasses import dataclass

from openclaw_brain.knowledge.executable.runner import NgspiceRunner
from openclaw_brain.knowledge.executable.templates import TEMPLATES
from openclaw_brain.knowledge.executable.verilog_runner import VerilogRunner


@dataclass(frozen=True)
class EngineSpec:
    """One simulator/language. `runner=None` is a first-class render-only engine."""

    name: str
    runner: object | None        # NgspiceRunner | VerilogRunner instance; None = render-only


_NGSPICE = NgspiceRunner()
_IVERILOG = VerilogRunner()

ENGINES: dict[str, EngineSpec] = {
    "ngspice": EngineSpec("ngspice", _NGSPICE),
    "iverilog": EngineSpec("iverilog", _IVERILOG),
}


def engine_for_template(template_ref: str) -> str:
    """The engine that renders+runs a template_ref. Defaults to ngspice (analog specimens carry no
    explicit engine tag), so this is backward-compatible by construction."""
    for entry in TEMPLATES.values():
        if entry.get("template_ref") == template_ref:
            return entry.get("engine", "ngspice")
    return "ngspice"


def runner_for_engine(engine: str, config=None):
    """The engine's runner object (the seam run_recipe uses when no runner is injected). Raises for an
    unknown engine or a render-only one — a render-only engine cannot run, by design."""
    spec = ENGINES.get(engine)
    if spec is None:
        raise KeyError(f"no engine {engine!r} registered (have {sorted(ENGINES)})")
    if spec.runner is None:
        raise RuntimeError(f"engine {engine!r} is render-only; cannot run an artifact")
    if config is not None and isinstance(spec.runner, NgspiceRunner):
        from openclaw_brain.egress import effective_egress
        return NgspiceRunner(image=spec.runner.image, workdir=spec.runner.workdir,
                             docker=spec.runner.docker, egress=effective_egress(config))
    return spec.runner
