"""Tests for the git-corpus SSOT (knowledge/executable/corpus.py).

Content-addressed store + the accretion model: a specimen's identity is its NETLIST, so
later claim-cards MERGE onto the same specimen (topology_class merge key, content-hash
dedup). Pure file I/O under tmp_path — no docker / Neo4j / LLM / git required.
"""

import logging

from openclaw_brain.knowledge.executable.corpus import SpecimenCorpus, compute_spec_id
from openclaw_brain.knowledge.executable.models import (
    ClaimCard, AnalogPVT, MechanismClaim, QuantTest, Specimen, VerdictClass,
)

COND = AnalogPVT(corner="tt", temp_c=27, vdd=1.8, cl_f=2e-12)
OTA = "miller_ota_2stage_nmos_in"
NETLIST = "* miller ota\nXM1 o1n vinp tail 0 sky130_fd_pr__nfet_01v8 W=8 L=0.5\n.end\n"


def _claim(cid, knob, series, kind_kw, verdict=None):
    c = ClaimCard(
        id=cid, topology_class=OTA,
        mechanism=MechanismClaim(knob=knob, metric="gbw_hz", series_ref=series,
                                 quant=QuantTest(**kind_kw), narrative="physics"),
        conditions=COND,
    )
    c.verdict = verdict
    return c


def _spec(claims):
    return Specimen(topology_class=OTA, netlist=NETLIST, role_map={"M1": "input_pair_a"},
                    pdk="sky130A", tool="ngspice-46", claim_cards=claims)


def test_spec_id_deterministic_and_netlist_bound():
    a = _spec([_claim("c1", "Cc", "cc", dict(kind="elasticity", target=-1.0, band=(-1.3, -0.7)))])
    assert compute_spec_id(a) == compute_spec_id(a)
    # identity = netlist+role+pdk+tool, NOT claim-cards
    b = _spec([_claim("c1", "Cc", "cc", dict(kind="elasticity", target=-1.0, band=(-1.3, -0.7))),
               _claim("c2", "Rz", "rz_pm", dict(kind="direction_to_optimum", sign="+"))])
    assert compute_spec_id(a) == compute_spec_id(b)        # claim-cards don't change identity
    c = _spec([])
    c.netlist = NETLIST + "* edit\n"
    assert compute_spec_id(c) != compute_spec_id(a)        # netlist DOES


def test_store_load_roundtrip(tmp_path):
    corpus = SpecimenCorpus(str(tmp_path))
    spec = _spec([_claim("cc_gbw_inverse", "Cc", "cc",
                         dict(kind="elasticity", target=-1.0, band=(-1.3, -0.7)),
                         verdict=VerdictClass.VERIFIED_WITH_CAVEAT)])
    corpus.store(spec)
    back = corpus.load(OTA, spec.spec_id)
    assert back.netlist == NETLIST
    assert back.role_map == {"M1": "input_pair_a"}
    assert len(back.claim_cards) == 1
    assert back.claim_cards[0].verdict == VerdictClass.VERIFIED_WITH_CAVEAT   # enum survives yaml
    assert back.claim_cards[0].mechanism.quant.band == (-1.3, -0.7)


def test_accretion_merges_claim_cards(tmp_path):
    corpus = SpecimenCorpus(str(tmp_path))
    # source 1 contributes claim cc; source 2 (same netlist) contributes rz_pm
    corpus.store(_spec([_claim("cc", "Cc", "cc", dict(kind="elasticity", target=-1.0, band=(-1.3, -0.7)),
                               verdict=VerdictClass.VERIFIED)]))
    corpus.store(_spec([_claim("rz_pm", "Rz", "rz_pm", dict(kind="direction_to_optimum", sign="+"),
                               verdict=VerdictClass.VERIFIED_WITH_CAVEAT)]))
    sid = corpus.list_class(OTA)
    assert len(sid) == 1                                   # one specimen (same netlist), not two
    merged = corpus.load(OTA, sid[0])
    assert {c.id for c in merged.claim_cards} == {"cc", "rz_pm"}   # claims accreted


def test_reverdict_updates_in_place(tmp_path):
    corpus = SpecimenCorpus(str(tmp_path))
    corpus.store(_spec([_claim("cc", "Cc", "cc", dict(kind="elasticity", target=-1.0, band=(-1.3, -0.7)),
                               verdict=VerdictClass.VERIFIED)]))
    # re-run the same claim id with a new verdict -> updates, does not duplicate
    corpus.store(_spec([_claim("cc", "Cc", "cc", dict(kind="elasticity", target=-1.0, band=(-1.3, -0.7)),
                               verdict=VerdictClass.REFUTED)]))
    merged = corpus.load(OTA, corpus.list_class(OTA)[0])
    assert len(merged.claim_cards) == 1
    assert merged.claim_cards[0].verdict == VerdictClass.REFUTED


def test_same_signature_collision_still_updates_narrative_and_verdict(tmp_path):
    """Sanity check that the id-collision guard is scoped to genuine content changes only: the
    normal re-verification path (same knob/metric/quant.kind, verdict and/or narrative legitimately
    change) must keep updating in place, exactly like test_reverdict_updates_in_place."""
    corpus = SpecimenCorpus(str(tmp_path))
    corpus.store(_spec([_claim("cc", "Cc", "cc", dict(kind="elasticity", target=-1.0, band=(-1.3, -0.7)),
                               verdict=VerdictClass.VERIFIED)]))
    updated = ClaimCard(
        id="cc", topology_class=OTA,
        mechanism=MechanismClaim(knob="Cc", metric="gbw_hz", series_ref="cc",
                                 quant=QuantTest(kind="elasticity", target=-1.0, band=(-1.3, -0.7)),
                                 narrative="reworded narrative, same physical claim"),
        conditions=COND,
    )
    updated.verdict = VerdictClass.REFUTED
    corpus.store(_spec([updated]))

    merged = corpus.load(OTA, corpus.list_class(OTA)[0])
    assert len(merged.claim_cards) == 1
    assert merged.claim_cards[0].verdict == VerdictClass.REFUTED
    assert merged.claim_cards[0].mechanism.narrative == "reworded narrative, same physical claim"


def test_id_collision_with_different_content_is_refused_not_silently_overwritten(tmp_path, caplog):
    """recipe.py:499's root cause reaching corpus.py's accretion: the raw-fallback authoring path
    defaults an unset claim id to a list-POSITION-based string
    (`f"{topology_class}_claim_{idx}"`, recipe.py::_normalize_claim_card) — not derived from
    (knob, metric), so two semantically DIFFERENT claims can land on the SAME id across two
    authoring calls. Before this fix, `_merge_claim_cards` would silently replace the old claim
    with the new one (identity-clobber). Fixed: a same-id collision is only treated as an update
    when the (knob, metric, quant.kind) signature matches; otherwise the OLD claim is kept, a loud
    warning names both, and the incoming claim is skipped."""
    corpus = SpecimenCorpus(str(tmp_path))
    old_card = _claim("shared_id", "Cc", "cc", dict(kind="elasticity", target=-1.0, band=(-1.3, -0.7)),
                       verdict=VerdictClass.VERIFIED)
    corpus.store(_spec([old_card]))

    # a DIFFERENT semantic claim (different knob + metric) collides on the same id.
    new_card = ClaimCard(
        id="shared_id", topology_class=OTA,
        mechanism=MechanismClaim(knob="VDD", metric="av0_db", series_ref="vdd_av0",
                                 quant=QuantTest(kind="direction", sign="+"),
                                 narrative="a totally different physical claim"),
        conditions=COND,
    )
    new_card.verdict = VerdictClass.VERIFIED

    with caplog.at_level(logging.WARNING):
        corpus.store(_spec([new_card]))

    merged = corpus.load(OTA, corpus.list_class(OTA)[0])
    assert len(merged.claim_cards) == 1          # never silently duplicated either
    kept = merged.claim_cards[0]
    assert kept.id == "shared_id"
    assert kept.mechanism.knob == "Cc" and kept.mechanism.metric == "gbw_hz"   # the OLD claim survives
    assert kept.verdict == VerdictClass.VERIFIED
    # loud: names the colliding id and both narratives, so nothing is silently lost
    assert "shared_id" in caplog.text
    assert "physics" in caplog.text                              # the OLD claim's narrative
    assert "a totally different physical claim" in caplog.text   # the skipped incoming narrative


def test_list_classes(tmp_path):
    corpus = SpecimenCorpus(str(tmp_path))
    corpus.store(_spec([]))
    mirror = Specimen(topology_class="current_mirror_simple", netlist="* mirror\n.end\n")
    corpus.store(mirror)
    assert set(corpus.list_classes()) == {OTA, "current_mirror_simple"}


def test_testbenches_round_trip_and_excluded_from_spec_id(tmp_path):
    # I5: the rendered sweep decks make a specimen self-contained, but must NOT change spec_id —
    # identity is the NETLIST so claims from different sweeps still accrete onto one specimen.
    base = Specimen(topology_class=OTA, netlist=NETLIST)
    withtb = Specimen(topology_class=OTA, netlist=NETLIST,
                      testbenches={"cc_gbw_hz": "* gbw sweep\n.end\n", "cl_pm_deg": "* pm sweep\n.end\n"})
    assert compute_spec_id(base) == compute_spec_id(withtb)        # accretion identity preserved

    corpus = SpecimenCorpus(str(tmp_path))
    corpus.store(withtb)
    loaded = corpus.load(OTA, withtb.spec_id)
    assert loaded.testbenches == {"cc_gbw_hz": "* gbw sweep\n.end\n", "cl_pm_deg": "* pm sweep\n.end\n"}
