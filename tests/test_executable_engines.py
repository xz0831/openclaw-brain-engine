"""Tier-B Task 2: pluggable engine registry + engine-by-template resolution + runner-by-engine."""
from __future__ import annotations

import pytest

from openclaw_brain.knowledge.executable.engines import (
    ENGINES, EngineSpec, engine_for_template, runner_for_engine,
)
from openclaw_brain.knowledge.executable.runner import NgspiceRunner
from openclaw_brain.knowledge.executable.verilog_runner import VerilogRunner


def test_ngspice_engine_registered_with_runner():
    assert "ngspice" in ENGINES
    assert isinstance(ENGINES["ngspice"], EngineSpec)
    assert isinstance(ENGINES["ngspice"].runner, NgspiceRunner)


def test_iverilog_engine_registered_with_runner():
    assert "iverilog" in ENGINES
    assert isinstance(ENGINES["iverilog"].runner, VerilogRunner)


def test_existing_analog_template_defaults_to_ngspice():
    # the 15 analog specimens carry no explicit engine tag -> default ngspice (back-compat)
    assert engine_for_template("miller_ota_ac") == "ngspice"
    assert engine_for_template("cds_tran") == "ngspice"


def test_unknown_template_ref_defaults_to_ngspice():
    assert engine_for_template("does_not_exist") == "ngspice"


def test_runner_for_engine_returns_the_object():
    assert runner_for_engine("ngspice") is ENGINES["ngspice"].runner
    assert runner_for_engine("iverilog") is ENGINES["iverilog"].runner


def test_runner_for_engine_rejects_unknown():
    with pytest.raises(KeyError):
        runner_for_engine("does_not_exist")
