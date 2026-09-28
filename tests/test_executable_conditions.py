"""Tier-B Task 1: Conditions as a CLOSED discriminated union + the total summarize_scope.

The engine-3 guard is the `summarize_scope` total function raising on an unhandled kind — adding a
third engine with a new Conditions variant fails loudly here until its scope branch exists.
"""
from __future__ import annotations

import pytest

from openclaw_brain.knowledge.executable.conditions import (
    AnalogPVT,
    DigitalUnits,
    UnknownConditionsKind,
    summarize_scope,
)


def test_analog_pvt_summary_is_pvt_scoped():
    c = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)
    s = summarize_scope(c)
    assert s["device"] == "nominal"
    assert s["corners"] == ["sky130/tt/27/1.8"]   # node first-class (Stat-QT T1)
    assert s["statistical"] == "none"


def test_digital_units_summary_is_functional_scoped():
    c = DigitalUnits(clock_period_ns=10.0, bit_width=14)
    s = summarize_scope(c)
    assert s["device"] == "n/a"
    assert s["statistical"] == "n/a"
    assert s["functional"] is True


def test_summarize_scope_is_total_unknown_kind_raises():
    class _Fake:
        kind = "quantum"

    with pytest.raises(UnknownConditionsKind):
        summarize_scope(_Fake())


def test_analog_pvt_keeps_former_conditions_fields():
    # back-compat: the analog variant is field-identical to the pre-Tier-B Conditions
    c = AnalogPVT(corner="ss", temp_c=-40.0, vdd=1.62, cl_f=2e-12, pdk_profile={"device": "sky130_fd_pr__nfet_01v8"})
    assert c.kind == "analog_pvt"
    assert c.corner == "ss" and c.temp_c == -40.0 and c.vdd == 1.62 and c.cl_f == 2e-12
    assert c.pdk_profile["device"] == "sky130_fd_pr__nfet_01v8"


def test_summarize_scope_renders_pdk_node_first_class():
    c = AnalogPVT(corner="tt_mm", temp_c=27.0, vdd=1.8, mc_runs=200,
                  pdk_profile={"pdk": "sky130", "node": "130nm"})
    s = summarize_scope(c)
    assert s["pdk"] == "sky130" and s["node"] == "130nm"
    assert s["corners"][0] == "sky130/tt_mm/27/1.8"      # node-locality undetachable from the magnitude
    assert s["statistical"] == "3σ@200" and s["device"] == "mismatch"


def test_summarize_scope_defaults_pdk_for_legacy_analog():
    c = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8)      # no pdk_profile, no mc
    s = summarize_scope(c)
    assert s["pdk"] == "sky130" and s["statistical"] == "none"
    assert s["corners"][0] == "sky130/tt/27/1.8"


def test_summarize_scope_corner_and_pelgrom_axes():
    cc = AnalogPVT(corner="tt", temp_c=27.0, vdd=1.8, corners=["tt", "ss", "ff", "sf", "fs"])
    assert summarize_scope(cc)["corners"] == [f"sky130/{x}" for x in ("tt", "ss", "ff", "sf", "fs")]
    cp = AnalogPVT(corner="tt_mm", temp_c=27.0, vdd=1.8, mc_runs=200, areas=[0.25, 1.0, 4.0])
    assert summarize_scope(cp)["pelgrom"] == "areas@3"
