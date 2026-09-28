from __future__ import annotations

from openclaw_brain.knowledge.canonicalization import propose_merges, same_entity


def _concept(cid: str, name: str, aliases: list[str] | None = None) -> dict:
    concept = {"id": cid, "name": name}
    if aliases is not None:
        concept["aliases"] = aliases
    return concept


def test_autozero_operation_is_shadow_proposed():
    proposals = propose_merges(
        [
            _concept("c1", "Autozero"),
            _concept("c2", "Auto-Zero Operation"),
        ]
    )

    assert len(proposals) == 1
    assert {proposals[0]["a_id"], proposals[0]["b_id"]} == {"c1", "c2"}
    assert proposals[0]["firing_rule"] in {"alias", "generic_qualifier"}


def test_cds_baseband_transfer_function_is_guarded():
    proposals = propose_merges(
        [
            _concept("c1", "CDS (Correlated Double Sampling)"),
            _concept("c2", "CDS Baseband Transfer Function"),
        ]
    )

    assert proposals == []


def test_dc_gain_is_guarded_from_frequency_origin():
    proposals = propose_merges(
        [
            _concept("c1", "DC (Frequency Origin)"),
            _concept("c2", "DC Gain"),
        ]
    )

    assert proposals == []


def test_identical_live_normalized_names_are_not_reproposed():
    proposals = propose_merges(
        [
            _concept("c1", "Auto-Zero Operation"),
            _concept("c2", "Auto Zero Operation"),
        ]
    )

    assert same_entity("Auto-Zero Operation", "Auto Zero Operation")
    assert proposals == []


def test_live_alias_normalized_match_is_not_reproposed():
    proposals = propose_merges(
        [
            _concept("c1", "Autozero", aliases=["Auto Zero Operation"]),
            _concept("c2", "Auto-Zero Operation"),
        ]
    )

    assert same_entity("Autozero", "Auto-Zero Operation")
    assert proposals == []


def test_thermal_noise_factor_boundary_tracks_same_entity():
    concepts = [
        _concept("c1", "Thermal Noise"),
        _concept("c2", "Thermal Noise Factor"),
    ]

    proposals = propose_merges(concepts)
    if same_entity("Thermal Noise", "Thermal Noise Factor"):
        assert len(proposals) == 1
        assert proposals[0]["firing_rule"] == "salient_containment"
    else:
        assert proposals == []
