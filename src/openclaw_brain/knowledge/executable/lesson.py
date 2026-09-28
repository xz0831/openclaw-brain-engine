"""Teaching-loop lesson model + the light citation audit (SPEC:
docs/superpowers/specs/2026-06-29-teaching-loop-v1-design.md; extended by E3-I2 §4:
docs/superpowers/specs/2026-07-04-e3-intervention-experiments.md; extended again by law-tier I2 §5:
docs/superpowers/specs/2026-07-04-law-tier-graph-representation.md).

openclaw-brain provides DRY, self-labeled evidence; Hermes writes the lesson. This module is the
only audit the brain performs, and it is ADVISORY (a self-check tool, not a hard gate). It guards
exactly one boundary — a lesson must not present an interpretive MECHANISM as a sim-CERTIFIED fact:
  - a 'certified' claim must cite an existing claim-card whose verdict is in the certified band;
  - an 'interpretive' claim needs no citation, only its tier label (so it is allowed, but never
    disguised as certified).
It judges NOTHING else — not direction, not completeness, not ordering, not quality. Those are
Hermes's (narrative) domain; checking them here would collapse the brain=evidence / Hermes=narrative
division.

E3-I2 extends the SAME boundary one notch, for CAUSATION specifically: an observational verdict
(however certified) proves WHAT moved, never WHY — so a 'certified' claim making a CAUSAL assertion
("caused by"/"the cause of"/"causes"/"because"/"due to the ... path"/"the culprit is"/"explains
why"/"responsible for"/"stems from"/"attributable to") must cite an INTERVENTION-family card
(a do-operator paired baseline/variant run, E3-I1's substrate; the citation carries an intervention
id in its `scope`) — the same causal sentence citing an ordinary observational card still fails, and
`mechanism_never_fact` (oracle.py) is UNCHANGED: a card's own mechanism narrative is still never
auto-fact regardless of intervention. This module never touches oracle.py's hardened rule; it only
lets the LESSON layer's causal PROSE pass when — and only when — it is grounded in an intervention
citation. An intervention card's named `idealization` (when present) must also be acknowledged in the
claim's own text, reusing the existing refuse-to-generalize machinery (`_overgeneralizes`) rather than
a parallel checker — the idealization must not be laundered away by omission.

Law-tier I2 extends the citation surface, not the boundary: a 'certified' claim may cite a `law_id`
(a `Regularity` node's 40-hex sha1, docs/superpowers/specs/2026-07-04-law-tier-graph-representation.md
§3) instead of a claim_card id. A law's own `status` gates certification exactly as a card's `verdict`
does — citing a Regularity whose status is not "law" (process_scoped/demoted) FAILS, naming the status
(R5's payoff: a stale law citation breaks loudly). A law-status citation's certified generalization
span is EXACTLY its member PDK set — refuse-to-generalize relaxes to that union and not one step
further ("across sky130A, gf180mcuD and ihp-sg13g2" passes; "on any 28nm process" still fails) — again
by parameterizing `_overgeneralizes` (its `law_members` argument), never a parallel checker. A causal
claim citing an observational law fails exactly like one citing an observational card (a Regularity is
cross-PDK OBSERVATION, never an intervention) — `_causal_without_intervention` needs no law-specific
branch at all.
"""

from __future__ import annotations

import re
from typing import Callable

from pydantic import BaseModel, Field

# A claim-card's verdict (string property on the graph node) is "certified" when it is one of these.
_CERTIFIED_VERDICTS = {"VERIFIED", "VERIFIED_WITH_CAVEAT"}

# QuantTest kinds that certify only the SHAPE of a relationship (a direction or an invariance), NOT an
# absolute scalar value. A 'certified' claim that asserts a specific magnitude while citing a card of
# one of these kinds over-claims: the oracle proved "rises/falls" or "is independent", never the number
# (a PDK-bound magnitude is interpretive, "R4"). 'value'/'elasticity' DO certify a scalar/slope, so a
# magnitude under those is legitimately certified and is not flagged.
_SHAPE_ONLY_KINDS = {"direction", "direction_to_optimum", "invariance"}

# Law-tier-only extension of the shape-only set (must-fix, law-tier I2 spec §3: a law's own statement
# is "shape-level, never a magnitude"): under a card citation, 'elasticity'/'value'/'statistical' DO
# certify a scalar (a single specimen's fitted number). Under a LAW citation they do NOT — the members'
# fitted scalars differ across PDKs (e.g. Pelgrom's per-PDK exponent -0.4616/-0.4837/-0.4657 — the
# drift IS knowledge, kept per-member) and a law certifies only the shared shape + in-band replication,
# never one cross-PDK number no oracle ever certified. 'corner' is deliberately NOT added here (no
# live/spec evidence a corner-kind law even exists; out of this fix's scope).
_LAW_SCALAR_KINDS_STILL_SHAPE_ONLY = {"elasticity", "value", "statistical"}

# A magnitude assertion = a number immediately followed by a RESULT unit (gain dB, freq Hz, spread %,
# resistance Ω, slope V/s, current A). Deliberately NARROW: condition units (bare V for VDD, C for temp,
# F for caps) are excluded so the audit does not false-flag the R1 conditions a claim may restate.
_MAG_UNIT = r"(?:dB/dec|dB|GHz|MHz|kHz|Hz|V/s|[µu]A|mA|nA|[kMG]?ohms?|[kMG]?Ω|%)"
_MAGNITUDE_RE = re.compile(r"[~≈]?\s*[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*" + _MAG_UNIT)


def _asserts_magnitude(text: str) -> bool:
    """True if the prose asserts a specific RESULT magnitude (e.g. '~67 dB', '0.2%', '26.3 MHz')."""
    return bool(_MAGNITUDE_RE.search(text or ""))


# Refusal-to-generalize (ADR Decision 4.3), SYMMETRIC per engine basis and SCOPE-SEMANTIC (a curated
# phrase set covering paraphrase, not an exact-token blocklist). A nominal/functional verdict must not be
# presented as more than it is:
#   - analog (physical-nominal): NOT robust/yields/production-ready/in-silicon/across-corners/mismatch-immune
#   - digital (functional): NOT timing/CDC/metastability-clean, NOT "matches the real spec" / silicon-correct
_GENERALIZE_ANALOG = re.compile(
    r"\b(robust(?:ness|ly)?|production[ -]ready|in[ -]silicon|silicon[ -]proven|"
    r"across (?:all )?(?:corners?|pvt|process|temperature)|all[ -]corners?|every[ -]corner|"
    r"pvt[ -]?(?:robust|invariant|immune|safe)|mismatch[ -]?(?:immune|free|insensitive|robust)|"
    r"yield(?:s|ing|[ -]safe)?|monte[ -]?carlo|guarantee[ds]?|"
    # cross-node transfer (Stat-QT): a sky130/130nm verdict must not be presented as the user's node
    r"your[ -](?:process|pdk|node)|applies[ -]to[ -]your|scales[ -]to|"
    r"\b\d{1,3}\s?nm\b|advanced[ -]node|"
    # cross-TOPOLOGY-instance generalization (E3-I2 spec Q3-b: "this is why ALL two-stage amps...") —
    # a single specimen's intervention-certified pathway does not transfer to every instance of its
    # topology CLASS, any more than a single corner transfers to every corner.
    r"\b(?:all|every)\s+(?:[\w-]+\s+){0,3}"
    r"(?:amps?|amplifiers?|op-?amps?|otas?|topologies|designs?|circuits?|stages?))\b", re.I)
_GENERALIZE_DIGITAL = re.compile(
    r"\b(timing[ -]?(?:clean|safe|closed?|met|correct)|meets?[ -]timing|setup[ -/]hold|"
    r"cdc[ -]?(?:clean|safe|correct)|clock[ -]domain|metastab\w*|"
    r"silicon[ -]?correct|matches?[ -]the[ -](?:real|full|actual)[ -]spec|spec[ -]?(?:compliant|correct)|"
    r"production[ -]ready|"
    r"for[ -]all[ -]inputs?|every[ -](?:case|input|sequence|value)|exhaustiv(?:e|ely)|"
    r"any[ -](?:input|sequence)|fully[ -]verified|complete[ -]coverage)\b", re.I)

# Idealization acknowledgment (E3-I2 spec Q3-b): phrases that acknowledge a claim is stated under a
# named idealized/model abstraction rather than the literal un-idealized physical circuit — the exact
# vocabulary seeds.py's own ff_break narrative uses ("IDEALIZED", "ideal buffer... abstraction", "in
# the model"). Deliberately small (this is an ACK check, not a paraphrase-detection net): the claim
# just has to signal "I know this is the idealized/model version", not phrase it any particular way.
_IDEALIZATION_ACK_RE = re.compile(
    r"\bideal(?:iz(?:ed|ation|es|ing)?)?\b|\babstraction\b|\bin the model\b", re.I)


# Law-tier citation overreach (docs/superpowers/specs/2026-07-04-law-tier-graph-representation.md
# §5 bullet 2): a certified claim citing a `law`-status Regularity may generalize EXACTLY across the
# member PDK set the law's own cross-PDK replication covers — "across sky130A, gf180mcuD and
# ihp-sg13g2" / "across the three foundry model families tested" / "in every process model we
# tested" all PASS (the law licenses exactly that span, no more). ONE step further — an untested,
# unbounded process/foundry/node claim no member PDK was ever run under ("on any 28nm process", "in
# silicon", "on all CMOS processes", "universally") still FAILS: a law is the union of its replicated
# members, never a license to claim every foundry/node/process. Deliberately narrow + scope-semantic
# (mirrors `_GENERALIZE_ANALOG`'s own discipline): the phrases below name UNBOUNDED process/foundry/
# node language; `_law_bounded_to_members` exempts a claim whose own text is bounded to what was
# actually tested — either it says so explicitly ("tested", the exact word the spec's own passing
# phrases use) or it names the member PDKs themselves (an explicit enumeration IS a bounded citation,
# no "tested" needed).
_LAW_OVERREACH_RE = re.compile(
    r"\bany\s+(?:\d{1,3}\s?nm\s+)?process(?:es)?\b|"
    r"\ball\s+(?:cmos\s+)?process(?:es)?\b|"
    r"\bevery\s+(?:cmos\s+)?process(?:es)?\b|"
    r"\buniversal(?:ly)?\b|"
    r"\bin\s+silicon\b|"
    r"\bany\s+foundry\b|\ball\s+foundr(?:y|ies)\b|"
    r"\b\d{1,3}\s?nm\b",
    re.I)

# A bounding phrase ("tested" / a member-PDK enumeration) only exempts an overreach hit that is IN THE
# SAME CLAUSE — otherwise an OR-gate over the WHOLE text lets bounding language anywhere neutralize an
# explicit, unrelated overreach elsewhere in the SAME sentence (live-confirmed must-fix: "...across
# sky130A, gf180mcuD and ihp-sg13g2 — and therefore in silicon on any 28nm process." was passing
# because the member enumeration in clause 1 "bounded" the unrelated overreach in clause 2; likewise
# "We tested it thoroughly; ... holds on any process at any foundry." and "...tested processes and
# beyond, universally."). Split on the connectives that can pivot a bounded clause into a NEW,
# unbounded one within the SAME sentence: em-dash, semicolon, "therefore", "beyond".
_CLAUSE_BREAK_RE = re.compile(r"[—;]|\btherefore\b|\bbeyond\b", re.I)


def _clause_spans(text: str) -> list[str]:
    """Split `text` into clauses at `_CLAUSE_BREAK_RE`'s connectives, so a bounded clause cannot
    launder an overreach clause that follows it in the same sentence."""
    clauses = []
    start = 0
    for m in _CLAUSE_BREAK_RE.finditer(text):
        clauses.append(text[start:m.start()])
        start = m.end()
    clauses.append(text[start:])
    return clauses


def _law_bounded_to_members(text: str, members: list[str]) -> bool:
    """True if `text` scopes its generalization to what the law actually replicated. Two ways to be
    bounded: the text says so ('tested'/'we tested') or it names the member PDKs by name (a literal
    enumeration of the certified span is itself the bounded citation, no 'tested' needed)."""
    if re.search(r"\btested\b", text or "", re.I):
        return True
    return bool(members) and all(re.search(re.escape(pdk), text or "", re.I) for pdk in members)


def _law_overreach(text: str, members: list[str]) -> str | None:
    """Concern string if `text` generalizes a law citation beyond its certified member set, else None.
    Checked PER CLAUSE (`_clause_spans`): a clause containing an overreach phrase must ITSELF carry
    the bounding language — bounding language in a different clause of the same sentence must not
    neutralize it (the OR-gate-over-whole-text must-fix this closes)."""
    t = text or ""
    if not _LAW_OVERREACH_RE.search(t):
        return None
    for clause in _clause_spans(t):
        if _LAW_OVERREACH_RE.search(clause) and not _law_bounded_to_members(clause, members):
            return (f"a law citation licenses generalization EXACTLY across its certified member set "
                    f"({', '.join(members) or 'none'}) — this claim goes one step further, to an "
                    f"untested/universal process, foundry, or node the law never replicated")
    return None


def _overgeneralizes(text: str, basis: str | None, idealization: str | None = None,
                      law_members: list[str] | None = None) -> str | None:
    """Concern string if a certified claim generalizes a scoped verdict beyond what the engine certified,
    else None. basis=None (an older card with no recorded basis) -> skipped (prior behavior preserved).

    `idealization` (E3-I2 spec Q3-b): when the CITED CARD carries a named idealization (an
    intervention-family card whose variant deck is not the literal physical circuit — e.g. ff_break's
    ideal unity-gain buffer), a claim that asserts the mechanism WITHOUT acknowledging that idealized/
    model scope over-generalizes exactly the way a nominal-corner verdict over-generalizes to "silicon"
    — REUSES this same function/finding rather than a parallel checker (the idealization must not be
    laundered away by omission).

    `law_members` (law-tier I2, spec §5 bullet 2): when the citation resolves to a `law`-status
    Regularity rather than a ClaimCard, pass its member PDK list here — the SAME function/finding
    mechanism, parameterized with the law's own certified span as the "allowed" scope, rather than a
    parallel audit path. None (the default, and always for ordinary ClaimCard citations) skips this
    check entirely. A Regularity carries no `basis` (it spans PDKs/engines, not one engine's single
    run) — spec §5 relaxes refuse-to-generalize ONLY on the process-span axis (`_law_overreach`,
    checked FIRST for a law citation, below), to exactly the member set. Every OTHER axis (silicon/
    PVT/mismatch/yield robustness for an analog-basis member, timing/CDC/spec-match for a functional-
    basis member) still applies to a law citation exactly as it would to citing one of the law's own
    member cards — a cross-PDK model replication certifies no more silicon/corner/yield robustness
    than a single member card does, so `basis is None` must not be read as "skip the whole family"
    for a law (must-fix: it was). `_law_overreach` runs BEFORE the analog/digital families (not
    after, as a card citation's checks are ordered) because its own vocabulary partially OVERLAPS
    theirs ("in silicon", "\\d{1,3}nm") but with a different, member-set-bounded exemption — a law
    citation properly bounded to its member set must resolve through that bounded check, not get
    swept up by the blanket-disallow analog family and lose its "member set" framing; an unbounded
    hit still fails, just with the correctly-attributed reason. A properly-bounded law citation falls
    through to still be checked against the OTHER (non-overlapping) analog/digital vocabulary below."""
    t = text or ""
    is_law = law_members is not None
    if is_law and (concern := _law_overreach(t, law_members)):
        return concern
    if (basis == "physical-nominal" or is_law) and _GENERALIZE_ANALOG.search(t):
        if is_law:
            return ("a cross-PDK law replication does not certify silicon/PVT/mismatch/yield "
                     "robustness — each member card behind the law is itself a nominal "
                     "single-corner verdict")
        return "a nominal single-corner ngspice verdict does not certify silicon/PVT/mismatch/yield robustness"
    if (basis == "functional" or is_law) and _GENERALIZE_DIGITAL.search(t):
        if is_law:
            return ("a cross-PDK law replication does not certify timing/CDC/metastability, a "
                     "match to the real spec, or behavior on un-exercised stimulus — each member "
                     "card behind the law is itself a single functional verdict")
        return ("a functional iverilog verdict (the RTL entails the property) does not certify "
                "timing/CDC/metastability, a match to the real spec, or behavior on un-exercised stimulus")
    if idealization and not _IDEALIZATION_ACK_RE.search(t):
        return (f"the cited card's mechanism is certified only under a named idealization "
                f"({idealization!r}) — the claim must acknowledge the idealized/model scope (e.g. "
                f"'idealized', 'in the model', 'abstraction'), not assert the mechanism as if it held "
                f"for the real, un-idealized circuit")
    return None


# Causal-family requirement (E3-I2 spec Q3-a, the mechanism-axis this whole track exists to move): an
# OBSERVED verdict — however certified its direction/magnitude/invariance — proves only WHAT moved; it
# can never prove WHY (a wrong cause that predicts the same numbers passes every sign test — the
# Type-3 error `mechanism_never_fact` exists to name). The simulator's do-operator (E3-I1: a paired
# baseline/variant run that structurally removes the hypothesized pathway) is the one substrate that
# CAN certify a causal component, and it says so by carrying an intervention id in the card's
# grounds/scope (executor.py's stamping; `scope["intervention"]`, undetachable). So a CERTIFIED claim
# may make a causal assertion ("caused by" / "causes" / "the cause of" / "because" / "due to the ...
# path" / "the culprit is" / "explains why" / "responsible for" / "stems from" / "attributable to")
# ONLY when it cites an intervention-family card — the SAME sentence citing an ordinary observational
# card (however certified) is exactly the causal-laundering risk this check exists to close.
#
# Scope-semantic, paraphrase-tested per the _overgeneralizes precedent (not an exact-token blocklist).
# Deliberately biased toward OVER-detecting causal phrasing: missing a causal claim (a false NEGATIVE
# here, which lets a causal sentence past this gate uncited-to-an-intervention) is worse than an
# occasional false-positive flag on borderline prose — "conservative" means "catch more", not "catch
# less", for exactly the reason a silently-accepted wrong cause is the failure mode E3 exists to close.
# The family below is deliberately wider than the 4 seed phrases first drafted for this check: it also
# covers ACTIVE-voice "causes"/"causing"/"caused" (not just the passive "caused by"), and the
# culprit-is / explains-why / responsible-for / stems-from / attributable-to paraphrase family — each
# a plain restatement of "X causes Y" that a lesson could reach for to say the same thing without
# tripping a token-exact "caused by" blocklist.
_CAUSAL_RE = re.compile(
    r"\b(?:is\s+)?caused\s+by\b|"
    r"\bcaus(?:e|es|ed|ing)\b|"
    r"\bthe\s+cause\s+of\b|"
    r"\b(?:the\s+)?culprit\b|"
    r"\bexplains?\s+why\b|"
    r"\bresponsible\s+for\b|"
    r"\bstems?\s+from\b|"
    r"\battributable\s+to\b|"
    r"\bbecause\b|"
    r"\bdue\s+to\s+the\b[^.]{0,40}\bpath\b",
    re.I,
)


def _is_causal_claim(text: str) -> bool:
    """True if `text` makes (or paraphrases) a causal assertion — see `_CAUSAL_RE`'s module comment
    for why this is deliberately biased toward over-detection."""
    return bool(_CAUSAL_RE.search(text or ""))


def _causal_without_intervention(text: str, card: dict) -> str | None:
    """Concern string if `text` is a causal claim (`_is_causal_claim`) but the cited `card` carries no
    intervention id, else None. `card.get("intervention")` is absent/None for every pre-E3 (and every
    non-intervention post-E3) card — this is a NEW refusal, not a relaxed one: a causal-sounding claim
    citing an ordinary observational card was ALREADY forced to the interpretive tier by
    `mechanism_never_fact`; this check is what makes `audit_citations` enforce that at the lesson
    layer too, for the certified tier specifically."""
    if _is_causal_claim(text) and not card.get("intervention"):
        return (
            "certified claim asserts a CAUSAL mechanism (\"caused by\"/\"causes\"/\"the cause of\"/"
            "\"because\"/\"due to the ... path\"/\"the culprit is\"/\"explains why\"/\"responsible "
            "for\"/\"stems from\"/\"attributable to\"), but the cited card carries no intervention "
            "id — a causal assertion about a pathway is certified only by an intervention-family "
            "claim-card (an executor do-operator paired run, E3-I1), never by an observational "
            "verdict alone (mechanism_never_fact: certifying WHAT moved is not certifying WHY) — "
            "cite an intervention-family card, or move the causal claim to interpretive"
        )
    return None


# Metric-consistency check (HERMES_TEACHING_EVAL_2026-06-29.md finding #1, second half): the
# magnitude-under-shape guard above catches "a scalar asserted under a shape-only kind", but that is
# blind to a DIFFERENT leak the same finding names — a certified claim that names one measurable
# quantity (e.g. "gain") while citing a card whose mechanism.metric is a DIFFERENT quantity entirely
# (e.g. offset). This slips past the shape guard whenever the cited card's kind legitimately certifies
# a scalar (value/elasticity/statistical/corner — see _SHAPE_ONLY_KINDS above, which already covers all
# 7 QuantTest kinds: those 4 are the magnitude-legitimate complement of the 3 shape-only kinds), because
# the kind check alone cannot see that the NUMBER belongs to the wrong metric.
#
# Deliberately SMALL and SCOPE-SEMANTIC (a curated concept table over the metric vocabulary actually
# used by seeds.py, not a token-blocklist): each category groups the `mechanism.metric` field values
# that mean the same physical quantity, plus a regex for how that quantity is named in prose. "noise"
# has no certifying metric in the current corpus on purpose (kTC/thermal noise is explicitly disclaimed
# as uncertified in cds_vo's own narrative) — a certified claim about noise citing ANY card is a real
# mismatch, which is exactly the offset-vs-noise confusion the eval finding calls out by name.
_METRIC_TEXT_RE: dict[str, re.Pattern] = {
    "gain": re.compile(r"\b(?:open-loop gain|closed-loop gain|dc gain|voltage gain|differential gain|"
                        r"av0|adm|acl|gain)\b", re.I),
    "bandwidth": re.compile(r"\b(?:gain-bandwidth|gbw|bandwidth)\b", re.I),
    "phase_margin": re.compile(r"\bphase margin\b", re.I),
    "offset": re.compile(r"\b(?:offset|mismatch|vos|fpn|fixed[- ]pattern noise)\b", re.I),
    "noise": re.compile(r"\bnoise\b", re.I),
    "current": re.compile(r"\b(?:output current|iout)\b", re.I),
    "resistance": re.compile(r"\b(?:input resistance|rin)\b", re.I),
    "delay": re.compile(r"\b(?:propagation delay|tpd)\b", re.I),
}

# mechanism.metric (as stamped onto the graph node / seeds.py) -> its category above. A metric absent
# from this table is UNMAPPED: the check is skipped for it rather than guessing (false-positive
# discipline — an unrecognized metric string is not evidence of a mismatch).
_METRIC_TO_CATEGORY: dict[str, str] = {
    "av0_db": "gain", "adm_db": "gain", "acl_db": "gain",
    "gbw_hz": "bandwidth",
    "pm_deg": "phase_margin",
    "vos_v": "offset", "a_vos": "offset",
    "iout_a": "current",
    "rin_ohm": "resistance",
    "tpd_s": "delay",
}


def _metric_mismatch(text: str, metric: str | None) -> str | None:
    """Concern string if a certified claim's text names a DIFFERENT measurable quantity than the cited
    card's mechanism.metric, else None. Fires only when BOTH sides are confidently nameable: `metric`
    is present and mapped, AND the text names at least one recognized category. A text that names no
    category (generic/qualitative prose) or a metric absent from the table are both left alone — the
    check refuses to guess rather than over-flag."""
    if not metric:
        return None
    card_category = _METRIC_TO_CATEGORY.get(metric)
    if card_category is None:
        return None
    mentioned = {cat for cat, rx in _METRIC_TEXT_RE.items() if rx.search(text or "")}
    if mentioned and card_category not in mentioned:
        return (f"claim is about {'/'.join(sorted(mentioned))}, but cited card certifies "
                f"'{metric}' ({card_category}) — a different measurable quantity")
    return None


# resolve_card(claim_card_id) -> the cited thing's data, or None if it does not exist. Two shapes,
# distinguished by "kind" (law-tier I2; "kind" absent/"card" is the pre-I2 default so every existing
# caller keeps working unchanged):
#   - kind="card" (or absent): {"verdict": str, "quant_kind": str, "metric": str, "basis": str,
#     "intervention": str, "idealization": str, ...} — a ClaimCard, exactly the pre-I2 contract.
#     "quant_kind"/"metric"/"basis" are each optional: when absent (None) the corresponding check is
#     skipped, so older callers / cards without those recorded fields keep the prior, citation-only
#     behavior. "intervention" (E3-I2) is the id an intervention-family card carries in its scope
#     (`scope["intervention"]`, projected as-is — see projection.py/executor.py's scope-stamping);
#     absent/None for every pre-E3 card, which is exactly the backcompat: a causal claim citing such
#     a card is refused (spec Q3-a), not silently passed. "idealization" (also from `scope`) is the
#     named abstraction an idealized intervention (e.g. ff_break) carries; absent/None for a physical
#     intervention (rz_null) or any non-intervention card.
#   - kind="law": {"status": str, "quant_kind": str, "metric": str, "pdks": list[str]} — a
#     `Regularity` node (law-tier I2, docs/superpowers/specs/2026-07-04-law-tier-graph-representation
#     .md §5 bullet 2): a certified claim may cite a law_id (40-hex sha1) INSTEAD of a claim_card id.
#     A law's own `status` gates certification exactly as a card's `verdict` does (status != "law" ->
#     FAIL, naming the status — R5's payoff: a stale/demoted law citation breaks loudly, never
#     silently); its `pdks` member list becomes the audit's allowed generalization span (see
#     `_overgeneralizes`'s `law_members` parameter — reused, not forked). "intervention"/
#     "idealization" are never present for a law (a Regularity is never itself an intervention) —
#     the shared checks below treat their absence as "skip", exactly the existing backcompat
#     semantics. "basis" is likewise never present (a Regularity spans PDKs/engines, has no single
#     basis) but its absence does NOT skip the silicon/PVT/mismatch/yield (analog) or timing/CDC/
#     spec-match (digital) refuse-to-generalize families for a law citation — `kind == "law"` runs
#     both families unconditionally in `_overgeneralizes` (a cross-PDK replication certifies no more
#     robustness than any one of its member cards does); only the process-SPAN axis is relaxed to
#     the member set. `quant_kind`'s scalar-legitimate values ('elasticity'/'value'/'statistical')
#     are ALSO treated as shape-only for a law citation's magnitude-vs-kind check — a law never
#     certifies one cross-PDK scalar (member exponents/values differ by PDK); 'corner' is untouched
#     (out of scope, no live/spec evidence a corner-kind law exists).
CardResolver = Callable[[str], "dict | None"]


def _certified_verdict_ok(card: dict) -> str | None:
    """FAIL reason if the cited thing does not license `certified`, else None. Kind-dispatched
    (never a parallel checker): a ClaimCard is licensed by its own VERIFIED-family `verdict`; a
    Regularity (kind="law") is licensed by its own `status` — a law's citable "verdict" IS its
    status. A non-"law" status (process_scoped/demoted) fails LOUDLY, naming the status (spec §5
    bullet 2 / R5: a stale law citation must break loudly, never silently pass as still-taught)."""
    if card.get("kind") == "law":
        status = card.get("status")
        if status != "law":
            return f"cited regularity is {status} — re-verify before teaching"
        return None
    if card.get("verdict") not in _CERTIFIED_VERDICTS:
        return f"cited card not certified (verdict={card.get('verdict')})"
    return None


def _audit_certified_claim(index: int, claim: "LessonClaim", resolve_card: CardResolver) -> "AuditFinding":
    """The full certified-claim check sequence for ONE claim, kind-dispatched (card vs law) but
    running the SAME shared checks either way — magnitude-vs-kind, metric-consistency, causal-
    without-intervention, refuse-to-generalize (with the law's member set as the allowed span when
    the citation is a law). Extracted so `audit_citations` stays a thin per-claim dispatch loop."""
    if not claim.cites:
        return AuditFinding(index=index, ok=False, reason="certified claim has no citation")
    card = resolve_card(claim.cites)
    if card is None:
        return AuditFinding(index=index, ok=False,
                             reason=f"cited claim-card not found: {claim.cites}")
    kind = card.get("kind", "card")
    cite_label = "law" if kind == "law" else "card"

    verdict_reason = _certified_verdict_ok(card)
    if verdict_reason:
        return AuditFinding(index=index, ok=False, reason=verdict_reason)

    quant_kind = card.get("quant_kind")
    law_scalar_leak = kind == "law" and quant_kind in _LAW_SCALAR_KINDS_STILL_SHAPE_ONLY
    if (quant_kind in _SHAPE_ONLY_KINDS or law_scalar_leak) and _asserts_magnitude(claim.text):
        # citation resolves + is certified, but the claim asserts a scalar magnitude the oracle
        # never certified (the card proved only a direction/invariance; OR — law_scalar_leak — the
        # citation is a LAW, which never certifies a single cross-PDK magnitude even under a
        # scalar-legitimate kind, since the members' fitted scalars differ by PDK). Flag so the
        # lesson moves the number to the interpretive tier (closes the magnitude-under-certified
        # leak the Hermes teaching eval surfaced, and its law-tier extension).
        if law_scalar_leak:
            return AuditFinding(index=index, ok=False, reason=(
                f"certified claim asserts a scalar magnitude, but cited law {claim.cites} "
                f"certifies only the shared SHAPE (+ in-band replication) across its member set — "
                f"the per-member fitted '{quant_kind}' value differs by PDK and is member data, "
                f"never a law-certified magnitude — label the magnitude interpretive or cite the "
                f"specific member claim-card instead"))
        return AuditFinding(index=index, ok=False, reason=(
            f"certified claim asserts a scalar magnitude, but cited {cite_label} "
            f"{claim.cites} certifies only '{quant_kind}' "
            f"(direction/invariance), not an absolute value — label the magnitude interpretive"))

    if (concern := _metric_mismatch(claim.text, card.get("metric"))):
        # citation resolves + is certified + (if checked) shape-consistent, but the claim's prose
        # names a measurable quantity the cited thing does not actually certify (e.g. a "gain" claim
        # citing an offset card) — the semantically-blind half of the Hermes teaching-eval finding
        # the kind-only shape guard above cannot see.
        return AuditFinding(index=index, ok=False,
            reason=f"certified claim/citation metric mismatch: {concern} (cited {cite_label} "
                   f"{claim.cites}) — cite a {cite_label} whose metric matches the claim, or move "
                   f"the claim to interpretive")

    if (concern := _causal_without_intervention(claim.text, card)):
        # E3-I2 spec Q3-a: a certified claim may assert CAUSATION only when the citation is
        # intervention-family (the do-operator substrate, E3-I1) — an un-intervened, purely
        # observational verdict never licenses "caused by"/"because"/etc, however certified its
        # direction/value is (mechanism_never_fact's whole point). A law citation NEVER carries an
        # intervention id (a Regularity is cross-PDK OBSERVATION, never a do-operator run), so this
        # refuses a causal claim citing an observational law exactly like it refuses one citing an
        # observational card — reused as-is, no law-specific branch needed.
        return AuditFinding(index=index, ok=False, reason=concern)

    concern = _overgeneralizes(claim.text, card.get("basis"), card.get("idealization"),
                                card.get("pdks") if kind == "law" else None)
    if concern:
        # citation resolves + is certified, but the prose generalizes the verdict beyond its scope
        # (silicon/PVT/yield for analog; timing/CDC/spec-match for digital; beyond-member-set for a
        # law). Refuse so the lesson keeps the claim in-scope or labels the generalization
        # interpretive (ADR 4.3 / law-tier spec §5 bullet 2).
        return AuditFinding(index=index, ok=False, reason=(
            f"certified claim over-generalizes its scope: {concern} (cited {cite_label} "
            f"{claim.cites}, basis={card.get('basis')}) — keep it within scope or label the "
            f"generalization interpretive"))

    if kind == "law":
        return AuditFinding(index=index, ok=True, reason=(
            f"certified by law {claim.cites} (member set: {', '.join(card.get('pdks') or [])})"))
    return AuditFinding(index=index, ok=True, reason=f"certified by {claim.cites}")


class LessonClaim(BaseModel):
    """One factual statement in a lesson. `tier` is a plain str (not enum-validated) so a malformed
    tier becomes an audit FINDING rather than a parse error — the audit owns that judgement."""

    text: str                       # human-facing prose (Hermes's rendering)
    tier: str                       # "certified" | "interpretive"
    cites: str | None = None        # claim_card_id ({spec_id}:{claim}); required when tier=="certified"


class LessonPlan(BaseModel):
    topology_class: str
    spec_id: str | None = None
    title: str | None = None
    claims: list[LessonClaim] = Field(default_factory=list)


class AuditFinding(BaseModel):
    index: int                      # position of the claim in plan.claims
    ok: bool
    reason: str


class AuditReport(BaseModel):
    passed: bool                    # True iff every finding is ok
    certified_total: int
    certified_ok: int
    interpretive_total: int
    findings: list[AuditFinding]


def audit_citations(plan: LessonPlan, resolve_card: CardResolver) -> AuditReport:
    """Check each claim's tier + citation. Pure and deterministic; `resolve_card` is injected so the
    same function unit-tests with a mock and runs against the graph in production."""
    findings: list[AuditFinding] = []
    certified_total = certified_ok = interpretive_total = 0

    for i, claim in enumerate(plan.claims):
        if claim.tier == "interpretive":
            interpretive_total += 1
            findings.append(AuditFinding(index=i, ok=True, reason="interpretive (labeled; no citation required)"))
        elif claim.tier == "certified":
            certified_total += 1
            finding = _audit_certified_claim(i, claim, resolve_card)
            findings.append(finding)
            if finding.ok:
                certified_ok += 1
        else:
            findings.append(AuditFinding(index=i, ok=False,
                                         reason=f"invalid tier '{claim.tier}' (expected certified|interpretive)"))

    return AuditReport(
        passed=all(f.ok for f in findings),
        certified_total=certified_total, certified_ok=certified_ok,
        interpretive_total=interpretive_total, findings=findings,
    )


# ── Answer-surface citation audit ────────────────────────────────────────────────────────────────
#
# The blind-eval finding this closes: an answering agent cited REAL associative-layer node ids
# (Concepts/Insights) under a "Certified" heading. Every id EXISTED, so an id-existence check
# passes — but the trust TIER was misrepresented: only a ClaimCard (oracle-certified) or a
# law-status Regularity (cross-PDK replication) is certified evidence; a Concept/Insight/Equation
# is associative text knowledge — real and citable, but never oracle-backed. `audit_citations`
# above guards structured LessonPlans only; free-form answer TEXT had no guard. This is the
# answer-surface twin: same brain=evidence discipline, same pure/injected-resolver shape, applied
# to prose. Advisory like the lesson audit — it judges ONLY citation-tier honesty, never content,
# structure, or narrative quality.

# One bracket citation: [cite: ...], [certified: ...], [assoc: ...] — case-insensitive tag;
# several ids may share one bracket (comma/semicolon-separated).
_ANSWER_CITE_RE = re.compile(r"\[\s*(cite|certified|assoc)\s*:\s*([^\]]+)\]", re.IGNORECASE)

# Trust language: prose claiming certification/verification/oracle-backing. Judged per PARAGRAPH
# (blank-line split, with heading merging — see `_answer_paragraphs`) so a trust claim and the
# citations backing it are audited together. NOTE: a "[certified: id]" bracket itself matches (the
# word "certified") — intended: that syntax IS a certification claim, so its paragraph is held to
# the same evidence bar.
_ANSWER_TRUST_RE = re.compile(r"certif|verified|oracle", re.IGNORECASE)

# Negation guard for trust markers: an HONEST boundary answer says "No certified claim-cards exist
# for PLL" / "this is not verified" / "without certified evidence" — those negated occurrences must
# not count as trust language (they are the opposite of a trust claim). A marker occurrence is
# negated when (a) a negation word appears within ~6 words BEFORE it, or (b) the marker's own word
# is morphologically negated ("uncertified"/"unverified"/"non-certified" — the marker regex matches
# INSIDE those words, so the attached prefix must be checked directly). A paragraph whose ONLY
# trust markers are negated is not a trust paragraph; one non-negated marker anywhere still is.
_ANSWER_NEGATION_RE = re.compile(
    r"\b(?:no|not|\w+n't|never|without|absence|lack(?:s|ing)?|nothing|none|neither|nor)\b",
    re.IGNORECASE)
_ANSWER_NEG_PREFIX_RE = re.compile(r"\b(?:un|non-?)$", re.IGNORECASE)

_ANSWER_PARA_SPLIT_RE = re.compile(r"\n\s*\n")

# A markdown heading line ("# ..." .. "###### ..."). A heading is a LABEL for the paragraph that
# follows it, never a standalone claim — see `_answer_paragraphs`.
_ANSWER_HEADING_RE = re.compile(r"^#{1,6}\s")

# The two tiers that license certified/verified/oracle language.
_ANSWER_CERTIFIED_TIERS = {"certified", "certified_law"}

# ── M5 scope-attribution: canonical PDK lexicon ──
# The three process families the executable corpus actually runs, with every spelling the graph
# and answer prose are known to use (live-verified 2026-07-22: ClaimCard scopes say
# "sky130"/"gf180mcuD"/"ihp-sg13g2" while Regularity member lists say "sky130A" — the spelling
# mismatch is REAL and is exactly why canonicalization exists). Case-insensitive, and
# CONSERVATIVE by construction: ONLY these exact aliases are ever recognized as a stated PDK —
# indirect span phrases ("all process nodes", "every foundry", "both PDKs") are deliberately OUT
# of this check's scope (guessing them mechanically would trade false negatives for false
# positives; they remain `_overgeneralizes`/`_law_overreach` territory on the lesson surface).
_PDK_ALIASES: dict[str, str] = {
    "sky130a": "sky130", "sky130": "sky130",
    "gf180mcud": "gf180", "gf180mcu": "gf180", "gf180": "gf180",
    "ihp-sg13g2": "ihp", "sg13g2": "ihp", "ihp": "ihp",
}
# Longest alias first at each alternation site so "sky130A" matches whole, never as "sky130"+"A".
_PDK_TOKEN_RE = re.compile(
    r"\b(sky130a|sky130|gf180mcud|gf180mcu|gf180|ihp-sg13g2|sg13g2|ihp)\b", re.IGNORECASE)


def canonical_pdk(raw: str) -> str:
    """Canonical name for one PDK spelling (case-insensitive `_PDK_ALIASES` lookup):
    sky130A/sky130 -> sky130; gf180mcuD/gf180mcu/gf180 -> gf180; ihp-sg13g2/sg13g2/ihp -> ihp.
    An out-of-lexicon name canonicalizes to its own lowercased form — it stays a KNOWN scope
    that can never collide with the three lexicon families, rather than being silently dropped
    (a card scoped to an unknown PDK still refuses to license lexicon-PDK claims)."""
    token = (raw or "").strip().lower()
    return _PDK_ALIASES.get(token, token)

# Tiers that do NOT license trust language — for rules (a)/(b) an `uncertified_card` (a real
# ClaimCard whose own verdict is outside the certified band) counts exactly like associative: the
# id exists, but nothing about it is oracle-certified evidence for the claim being made.
_ANSWER_NONCERTIFIED_TIERS = {"associative", "uncertified_card"}

# Resolver: cited id -> at least {"labels": [...], "status": str|None, "verdict": str|None,
# "scope_pdks": list[str]|None} when the id resolves to a real graph node, else None. "verdict"
# is the node's own verdict property (ClaimCard) — required so the tier can GATE on it (a
# REFUTED card is not certified evidence); absent/None for every non-ClaimCard label.
# "scope_pdks" (M5 scope-attribution) is the CANONICAL (`canonical_pdk`) PDK scope of the cited
# thing: a ClaimCard's parsed scope-JSON "pdk" as a single-element list (None when the card's
# pdk is null/unparseable — an UNKNOWN scope is never guessed); a law-status Regularity's
# LICENSED member set (its member_summary keys whose verdict is VERIFIED-family, falling back to
# the recorded pdks list only when member_summary is missing/unparseable — pdks records every
# PDK with ANY verdict, agree or not, so it over-licenses and is fallback-only); None for every
# other label. Absent/None keeps the pre-M5 behavior: the scope check skips that citation.
# Injected (like CardResolver) so the same function unit-tests with a fake and runs against
# Neo4j via BrainAgent.audit_answer.
AnswerIdResolver = Callable[[str], "dict | None"]


def _answer_paragraphs(text: str) -> list[str]:
    """Paragraphs for trust-marker scanning: blank-line split, but a block consisting ONLY of
    markdown heading lines is MERGED with the block that follows it. A heading ('### Certified
    Evidence') labels the NEXT paragraph — audited separately it degrades into a spurious
    bare_verified_claim while the actual claim+citation paragraph escapes the trust scan
    entirely (the blind-eval false-positive/false-negative pair this fixes)."""
    merged: list[str] = []
    pending = ""
    for block in _ANSWER_PARA_SPLIT_RE.split(text or ""):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if lines and all(_ANSWER_HEADING_RE.match(ln) for ln in lines):
            pending = f"{pending}\n{block}" if pending else block
            continue
        merged.append(f"{pending}\n{block}" if pending else block)
        pending = ""
    if pending:
        merged.append(pending)
    return merged


def _negated_occurrence(text: str, start: int) -> bool:
    """True when the token starting at `text[start]` is negated: (a) a morphological negation
    prefix fused to it ("uncertified"/"non-verified"), or (b) a negation word within ~6 words
    BEFORE it in the same sentence. The window never crosses a SENTENCE boundary ([.!?] +
    whitespace — the lookahead spares decimals like '0.1 µm'): in 'smaller transistors worsen
    matching, not improve it. This directly contradicts certified evidence.' the 'not' belongs
    to the previous sentence and must not launder the affirmative certification claim that
    follows (a live v1-answer false negative caught in regression). SHARED by the trust-marker
    scan (`_has_affirmative_trust_marker`) and the M5 stated-PDK scan (`_stated_pdks`) — one
    negation semantics, never two parallel ones."""
    if _ANSWER_NEG_PREFIX_RE.search(text[:start]):
        return True     # morphological negation fused to the token
    same_sentence = re.split(r"[.!?](?=\s)", text[:start])[-1]
    window = " ".join(re.findall(r"[\w']+", same_sentence)[-6:])
    return bool(_ANSWER_NEGATION_RE.search(window))


def _has_affirmative_trust_marker(para: str) -> bool:
    """True iff at least one trust-marker occurrence in `para` is NOT negated (see
    `_negated_occurrence`). Negated-only paragraphs ('No certified claim-cards exist for PLL')
    are honest boundary statements and must not be audited as trust claims; a negated marker
    plus a SEPARATE affirmative one still makes the paragraph a trust paragraph."""
    return any(not _negated_occurrence(para, m.start())
               for m in _ANSWER_TRUST_RE.finditer(para))


# Trailing negation predicated ON the PDK token itself ("gf180mcuD remains untested",
# "gf180mcuD is not yet verified") — anchored right after the token so "sky130A but not on
# gf180mcuD" does NOT launder sky130A (the backward window already drops gf180mcuD there).
_PDK_TRAILING_NEG_RE = re.compile(
    r"\s+(?:remains?|is|are|was|were|stays?)\s+(?:still\s+)?"
    r"(?:un(?:tested|verified|certified)|not\s+(?:yet\s+)?(?:tested|verified|certified))",
    re.IGNORECASE)


def _stated_pdks(para: str) -> set[str]:
    """Canonical PDK names the paragraph PROSE affirmatively states (M5 scope-attribution).
    Bracket citations are stripped first (an id inside [certified: ...] is a citation, never a
    scope assertion), then every `_PDK_ALIASES` spelling is collected, minus negated mentions
    ("not verified on gf180mcuD" states nothing) — removed by the SAME negation machinery the
    trust-marker scan uses (`_negated_occurrence`). Lexicon-only and deliberately conservative:
    indirect span phrases ("all process nodes") are out of scope for this mechanical check."""
    prose = _ANSWER_CITE_RE.sub(" ", para or "")
    return {_PDK_ALIASES[m.group(1).lower()]
            for m in _PDK_TOKEN_RE.finditer(prose)
            if not _negated_occurrence(prose, m.start())
            and not _PDK_TRAILING_NEG_RE.match(prose, m.end())}


def _clean_answer_id(raw: str) -> str:
    """One id token: strip whitespace/backticks and any trailing ' — prose' annotation the answerer
    appended inside the bracket (em-dash prose is commentary, never part of an id)."""
    token = raw.split("—", 1)[0]
    return token.strip().strip("`").strip()


def _parse_answer_citations(text: str) -> list[tuple[str, str]]:
    """(cited_id, syntax) pairs in order of appearance; syntax lowercased to cite|certified|assoc."""
    pairs: list[tuple[str, str]] = []
    for m in _ANSWER_CITE_RE.finditer(text or ""):
        syntax = m.group(1).lower()
        for raw in re.split(r"[,;]", m.group(2)):
            cid = _clean_answer_id(raw)
            if cid:
                pairs.append((cid, syntax))
    return pairs


def extract_answer_citation_ids(text: str) -> list[str]:
    """Distinct cited ids in first-appearance order — the agent layer pre-resolves EXACTLY these
    (the pure audit takes a SYNC resolver, mirroring audit_citations' pre-resolve pattern)."""
    seen: dict[str, None] = {}
    for cid, _syntax in _parse_answer_citations(text):
        seen.setdefault(cid)
    return list(seen)


def _answer_tier(node: dict | None) -> str:
    """Trust tier of what a cited id ACTUALLY resolves to (independent of the syntax used):
    ClaimCard whose own verdict is in the certified band (`_CERTIFIED_VERDICTS`) -> certified
    (oracle-backed); ClaimCard with any OTHER/missing verdict -> uncertified_card — a real card,
    but a REFUTED/FLAGGED/... verdict licenses no trust language, so rules (a)/(b) treat it
    exactly like associative (`_ANSWER_NONCERTIFIED_TIERS`), and findings name the verdict;
    Regularity with status=="law" -> certified_law (cross-PDK replication; any other status —
    process_scoped/demoted — does NOT certify, R5's loud-demotion discipline); any other
    resolved node (Concept/Insight/Equation/...) -> associative; nothing -> unresolved."""
    if node is None:
        return "unresolved"
    labels = node.get("labels") or []
    if "ClaimCard" in labels:
        return "certified" if node.get("verdict") in _CERTIFIED_VERDICTS else "uncertified_card"
    if "Regularity" in labels and node.get("status") == "law":
        return "certified_law"
    return "associative"


class AnswerCitation(BaseModel):
    """One bracket-citation occurrence in the answer text. `tier` is what the id RESOLVES to,
    independent of the `syntax` the answerer chose — the gap between the two is the audit's whole
    subject. Plain str fields (not enum-validated), mirroring LessonClaim: a malformed value is the
    audit's judgement to make, never a parse error."""

    id: str
    syntax: str                     # "cite" | "certified" | "assoc" (as written, lowercased)
    tier: str                       # "certified" | "certified_law" | "uncertified_card" | "associative" | "unresolved"
    scope_pdks: list[str] | None = None  # canonical PDK scope the citation certifies (M5); None = unknown/not applicable


class AnswerAuditFinding(BaseModel):
    kind: str                       # "unresolved_citation" | "tier_misrepresentation" | "bare_verified_claim" | "scope_overstatement" | "mixed_certified_paragraph"
    detail: str
    citation_id: str | None = None  # None for paragraph-level findings with no single culprit id
    advisory: bool = False          # advisory findings inform but never fail the audit


class AnswerAuditReport(BaseModel):
    passed: bool                    # True iff no NON-advisory findings
    citations: list[AnswerCitation]
    findings: list[AnswerAuditFinding]


def audit_answer_citations(text: str, resolve_id: AnswerIdResolver) -> AnswerAuditReport:
    """Audit free-form answer TEXT for citation-tier honesty. Pure and deterministic; `resolve_id`
    is injected so the same function unit-tests with a fake and runs against the graph in
    production (BrainAgent.audit_answer). No I/O.

    Finding kinds — citation-tier honesty plus mechanical scope-attribution, nothing else:
      * unresolved_citation    — a cited id resolves to nothing in the graph.
      * tier_misrepresentation — (a) `[certified: id]` syntax on an id whose tier is not
        certified/certified_law — including a REAL ClaimCard whose own verdict is outside the
        certified band (tier `uncertified_card`; the finding names the verdict); OR (b) a
        trust-language paragraph (/certif|verified|oracle/i) whose citations include at least
        one associative/uncertified_card id and ZERO certified-tier ids (the blind-eval leak:
        real Concept/Insight ids under a "Certified" heading pass id-existence checks while
        misrepresenting the tier).
      * bare_verified_claim    — a trust-language paragraph with no citations at all.
      * scope_overstatement    — NON-advisory, paragraph-level (M5 scope-attribution, the
        blind-judge-verified failure class: "(verified on sky130A, gf180mcuD, ihp-sg13g2)"
        attributed to a single-PDK claim-card). stated = canonical PDK tokens affirmatively
        named in the paragraph's prose (`_stated_pdks`: alias lexicon only, citations stripped,
        negated mentions removed by the same machinery as trust markers); allowed = the UNION
        of `scope_pdks` over the paragraph's certified-tier citations that carry scope info.
        Fires ONLY when stated is nonempty AND at least one certified-tier citation in the
        paragraph has known scope_pdks AND stated ⊄ allowed — when every certified citation
        lacks scope info the check is SKIPPED entirely (the audit never guesses a scope), and
        non-certified citations' scopes never license anything. The finding names the missing
        PDKs and the cited ids. LIMITATION (deliberate): stated-PDK extraction recognizes ONLY
        the alias lexicon (sky130A/sky130; gf180mcuD/gf180mcu/gf180; ihp-sg13g2/sg13g2/ihp,
        case-insensitive) — indirect span phrases ("all process nodes", "every foundry") are
        out of scope for this mechanical check.
      * mixed_certified_paragraph — ADVISORY (advisory=True, never fails the audit): a
        trust-language paragraph citing BOTH >=1 certified-tier id AND >=1 associative/
        uncertified_card id. Semantic per-claim attribution inside a mixed paragraph (WHICH
        sentence leans on WHICH citation) is out of scope for this mechanical audit — the
        advisory surfaces the mix for a human/judge instead of silently passing it.

    Trust language is scanned per PARAGRAPH, with two mechanical guards:
      * markdown heading lines (/^#{1,6}\\s/) are merged with the following block
        (`_answer_paragraphs`) — a heading is a label for the next paragraph, never a
        standalone claim;
      * negated trust markers ("no certified claim-cards exist", "not verified", "without/
        absence of/lacks certified", "unverified") do not count (`_has_affirmative_trust_marker`)
        — an honest boundary answer must not be flagged for saying what it does NOT have.

    `passed` is True iff there are no NON-advisory findings.
    """
    citations: list[AnswerCitation] = []
    findings: list[AnswerAuditFinding] = []

    def _describe(c: AnswerCitation, node: dict | None) -> str:
        if c.tier == "uncertified_card":
            return f"{c.id} (uncertified card, verdict={(node or {}).get('verdict')})"
        return f"{c.id} ({c.tier})"

    for para in _answer_paragraphs(text):
        para_citations: list[tuple[AnswerCitation, dict | None]] = []
        for cid, syntax in _parse_answer_citations(para):
            node = resolve_id(cid)
            citation = AnswerCitation(id=cid, syntax=syntax, tier=_answer_tier(node),
                                      scope_pdks=(node or {}).get("scope_pdks"))
            para_citations.append((citation, node))
            citations.append(citation)
            if citation.tier == "unresolved":
                findings.append(AnswerAuditFinding(
                    kind="unresolved_citation", citation_id=cid,
                    detail=f"cited id resolves to nothing in the graph: {cid}"))
            if syntax == "certified" and citation.tier not in _ANSWER_CERTIFIED_TIERS:
                if citation.tier == "uncertified_card":
                    detail = (f"[certified: {cid}] but the cited ClaimCard's own verdict is "
                              f"{(node or {}).get('verdict')} — outside the certified band "
                              f"({'/'.join(sorted(_CERTIFIED_VERDICTS))}); an uncertified card "
                              f"never licenses certified syntax")
                else:
                    detail = (f"[certified: {cid}] but the id's actual tier is {citation.tier} — "
                              f"only a VERIFIED-family ClaimCard or a law-status Regularity is "
                              f"certified evidence")
                findings.append(AnswerAuditFinding(
                    kind="tier_misrepresentation", citation_id=cid, detail=detail))
        # M5 scope-attribution (independent of trust language — "(verified on X, Y, Z)" after a
        # citation is the live failure shape, but a PDK list stated next to a certified citation
        # overstates scope with or without a trust marker in the same paragraph).
        stated = _stated_pdks(para)
        if stated:
            certified_here = [c for c, _n in para_citations
                              if c.tier in _ANSWER_CERTIFIED_TIERS]
            scoped = [c for c in certified_here if c.scope_pdks is not None]
            allowed = {p for c in scoped for p in c.scope_pdks}
            missing = sorted(stated - allowed)
            # Any certified citation with UNKNOWN scope makes the paragraph unjudgeable —
            # the unknown card may itself license the stated PDK; the audit never guesses.
            if scoped and len(scoped) == len(certified_here) and missing:
                cited_desc = "; ".join(
                    f"{c.id} -> {'/'.join(c.scope_pdks) if c.scope_pdks else '(none)'}"
                    for c in scoped)
                findings.append(AnswerAuditFinding(
                    kind="scope_overstatement",
                    detail=(f"paragraph states PDK scope {', '.join(sorted(stated))} but its "
                            f"certified-tier citations cover only "
                            f"{', '.join(sorted(allowed)) or '(none)'} — missing: "
                            f"{', '.join(missing)} (cited: {cited_desc}); keep the stated scope "
                            f"within what the citations certify, or cite certified evidence for "
                            f"the missing PDK(s)")))
        if not _has_affirmative_trust_marker(para):
            continue
        if not para_citations:
            snippet = " ".join(para.split())[:120]
            findings.append(AnswerAuditFinding(
                kind="bare_verified_claim",
                detail=f'trust language ("certif/verified/oracle") with no citation at all: "{snippet}"'))
            continue
        noncert = [(c, n) for c, n in para_citations if c.tier in _ANSWER_NONCERTIFIED_TIERS]
        has_certified = any(c.tier in _ANSWER_CERTIFIED_TIERS for c, _n in para_citations)
        if noncert and not has_certified:
            findings.append(AnswerAuditFinding(
                kind="tier_misrepresentation", citation_id=noncert[0][0].id,
                detail=(f"trust-language paragraph is backed only by non-certified ids "
                        f"({', '.join(_describe(c, n) for c, n in noncert)}) — the ids exist, "
                        f"but nothing cited here is oracle-certified; cite a VERIFIED-family "
                        f"ClaimCard/law-status Regularity or drop the certified framing")))
        elif noncert and has_certified:
            findings.append(AnswerAuditFinding(
                kind="mixed_certified_paragraph", advisory=True,
                detail=(f"trust-language paragraph mixes certified-tier citations with "
                        f"non-certified ones ({', '.join(_describe(c, n) for c, n in noncert)}) "
                        f"— per-claim attribution inside a mixed paragraph is out of scope for "
                        f"the mechanical audit; verify the certified framing covers only the "
                        f"certified-cited claims (advisory, non-fatal)")))

    return AnswerAuditReport(passed=not any(not f.advisory for f in findings),
                             citations=citations, findings=findings)
