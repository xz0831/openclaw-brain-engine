"""Production link resolver for projection LinkRequests (SPEC §13.1).

The dry-run found that PURE top-1 embedding mis-ranks near-synonym distractors: "phase margin" and
"open-loop gain" sit close to "DC Loop Gain" in embedding space, so the correct node — which EXISTS
in the graph — loses to the distractor (av0 -> "DC Loop Gain" @0.921 over "DC Open-Loop Gain" @0.912;
"Phase Margin" absent from pm's top-4 entirely). Neo4j's NO-PHANTOM invariant only blocks UNRESOLVED
targets, not CONFIDENTLY-WRONG ones — so a naive resolver would inject wrong GROUNDS edges.

This hybrid resolves TEXT-FIRST: a curated munged-hint -> canonical-vocabulary expansion filters to
the right KIND of node by `canonical_name CONTAINS`, then embedding ranks the best instance among
those (text precision + embedding ranking). It falls back to pure per-label vector search only when
no text candidate exists, gated by a cosine floor so an unresolved hint forms NO edge.

Resolver signature matches projection.Resolver: `async (label, match_text) -> id | None`.
"""

from __future__ import annotations

import logging
import math

from ..graph.schema import NodeLabel
from ..graph.store import _vector_score_to_cosine
from .projection import _ID_FIELD

logger = logging.getLogger(__name__)

# Curated expansion: the projection's munged hint (lowercased) -> canonical-vocabulary phrases the
# graph actually uses (first phrase is the PRIMARY, used as the embedding-rank query). Keyed on the
# exact output of projection._readable / _metric_text. Unmapped hints fall through to the raw text
# (vector-only) path — less precise, but safe (gated).
_TOPOLOGY_EXPANSIONS: dict[str, list[str]] = {
    "miller ota 2stage nmos in": ["two-stage miller ota", "miller ota", "two-stage op amp"],
    "current mirror simple nmos": ["nmos current mirror", "current mirror"],
    "common source active load nmos": ["active load cs stage", "common-source stage",
                                       "common source amplifier", "active-loaded common source"],
    "common gate nmos": ["cg stage", "common-gate stage", "common gate amplifier", "common-gate"],
    "source follower nmos": ["source follower", "common-drain", "voltage buffer", "ac-coupled source follower"],
    "diff pair resistive nmos": ["basic differential pair", "differential pair", "diff pair",
                                 "resistively-loaded differential pair"],
    # F5 (2026-07-09 graph-to-graph relink, 3,359 new edges): the relink made the graph's row-return
    # order for a bare "cascode" CONTAINS scan unstable, and that phrase alone now matches 167
    # CircuitTopology nodes -- far past the resolver's LIMIT 40 -- so the exact-match candidate could
    # silently fall outside the fetched window depending on which 40-of-167 rows Neo4j happens to
    # return (an order-dependent drift risk independent of embedding ranking; verified live: dropping
    # the bare "cascode" catch-all shrinks the pool to 7, safely under the cap). The netlist
    # (_CASC_BODY in templates.py) is a bare 2-diode self-biased stack -- M1 (bottom, diode-connected)
    # in series with M2 (top, diode-connected), both bias nodes (nb, nc) reused directly by the output
    # branch (M3, M4) -- no resistor, no regulation, no wide-swing sizing. That is Razavi's "poor
    # man's" scheme specifically: the cheapest possible 4-device cascode mirror. It is distinct from
    # the graph's sibling nodes extracted from the same source passage -- "Self-Biased Cascode Current
    # Mirror" (a resistor-augmented VDS-matching scheme per secondary lit. (P.E. Allen's "self-biased
    # cascode current mirror" lecture fig. uses a series R for vDS1=vDS3 matching, "Excellent"
    # accuracy) -- our netlist has no such resistor), "Conventional Stacked..." and "Wide-Swing..."
    # (dedicated low-voltage bias sizing, also absent here) -- and from the bare umbrella "Cascode
    # Current Mirror" node (degree 73, too generic: could mean any of the above variants). "poor man's
    # cascode current mirror" is now the PRIMARY (exact-match) phrase so the specific, faithful node
    # wins deterministically regardless of which other decoys the LIMIT-40 window happens to surface.
    "cascode current mirror nmos": ["poor man's cascode current mirror", "cascode current mirror",
                                    "cascode mirror", "nmos cascode mirror"],
    "ota 5t nmos in": ["active-mirror differential pair", "5t ota", "five-transistor ota",
                       "single-stage ota", "5-transistor ota"],
    "telescopic cascode ota nmos in": ["telescopic cascode ota", "telescopic ota",
                                       "telescopic op amp", "telescopic cascode"],
    "folded cascode ota nmos in": ["nmos input folded-cascode op amp", "folded-cascode op amp",
                                   "folded-cascode operational amplifier", "folded-cascode ota",
                                   "folded-cascode"],
    "regulated cascode nmos": ["regulated cascode", "gain-boosted cascode amplifier",
                               "gain-boosted cascode", "cascode (gain-boosting technique)",
                               "gain-boosted stage"],
    "comparator continuous nmos": ["comparator", "analog comparator", "column comparator",
                                   "per-column comparator"],
    "cds switched cap nmos": ["correlated double-sampling amplifier", "column-level cds amplifier",
                              "correlated double sampling", "cds amplifier"],
    # "global ramp reference" is the defensible exact-match (a single-slope ADC's single global ramp);
    # the DAC-based / multiple-ramp nodes are MORE-SPECIFIC architectures this plain current-into-cap
    # specimen does not realize, so they are removed (a CONTAINS+embedding mis-bind would overclaim).
    "single slope ramp generator": ["global ramp reference", "ramp generator"],
    # NOT "variable-gain amplifier" (VGA = a distinct, often continuously-tuned class); keep to the
    # programmable-gain vocabulary the inverting-feedback PGA actually realizes.
    "column pga inverting nmos": ["programmable-gain amplifier", "programmable gain amplifier",
                                  "column amplifier"],
    # Digital (iverilog) pedagogy specimens (Tier-B). Digital-SPECIFIC phrases: a digital specimen binds
    # to a faithful digital node or to nothing (no GROUNDS/REALIZES edge), never silently to the analog
    # CDS node — same no-wrong-edge discipline the ramp-slope note above established.
    "digital cds nmos": ["digital correlated double sampling", "digital cds"],
    "gray code counter": ["gray counter", "gray code counter", "gray-code counter", "gray code"],
    "ss adc digital backend": ["single-slope adc digital back-end", "single-slope adc backend",
                               "single-slope adc", "digital readout back-end"],
    # S3-inc2a W2: the classic dVBE/R PTAT current core (TOPOLOGY_BACKLOG.md #22/#23) — the smallest
    # bandgap-family member, sky130-only for now (pdks.py's BjtUnavailable guard).
    "ptat ctat core bjt": ["ptat current source", "ptat current generator", "bandgap reference core",
                           "delta vbe circuit"],
}
_METRIC_EXPANSIONS: dict[str, list[str]] = {
    "gbw": ["gain-bandwidth product", "gain bandwidth product"],
    # F5 (2026-07-09 graph-to-graph relink): made "DC Open-Loop Gain" (id dc_open_loop_gain, degree 1,
    # ZERO HAS_PARAMETER edges from any CircuitTopology -- its only edge is DERIVED_FROM "DC Open-Loop
    # Gain Increase") newly resolver-visible (embedded, not retracted). With "dc open-loop gain"
    # primary, the exact-match tie-break picked this thin, structurally-disconnected duplicate over
    # the graph's actual amplifier-open-loop-gain node -- "Open-Loop Gain" (degree 48 / 23 across two
    # duplicate ids, HAS_PARAMETER from "Two-Stage Op Amp" / "Operational Amplifier" / "Differential
    # Cascode Op Amp" / etc -- the node every av0_db-emitting OTA recipe's gain concept actually
    # integrates with in the graph). "open-loop gain" is now primary so the established node wins
    # regardless of which thin duplicates the graph accumulates; "dc open-loop gain" stays as a
    # secondary phrase (still useful if "Open-Loop Gain" itself is ever retracted/renamed).
    "av0": ["open-loop gain", "dc open-loop gain"],
    "pm": ["phase margin"],
    "iout": ["output current"],
    "rin": ["input resistance", "input impedance"],
    # F5 (2026-07-09 graph-to-graph relink): introduced a new thin duplicate "Differential Gain" (id
    # new_differential_gain, degree 4) whose entire edge set is about a DIFFERENT context (cascode
    # internal-node gain: HAS_PARAMETER to/from "Cascode Amplifier Gain Mechanism", "Differential
    # Voltage Gain at Cascode Nodes (A and B)") -- not the resistively-loaded differential pair
    # (diff_pair_resistive_nmos) this metric actually characterizes. With "differential gain" primary,
    # the exact-match tie-break picked this decoy over "Differential Voltage Gain" (id
    # differential_voltage_gain, degree 40; HAS_PARAMETER from "Differential Pair" / "Differential
    # Amplifier" / "Asymmetric Differential Pair" / etc, and already the target of 3 live GROUNDS
    # edges from prior claim-cards). "differential voltage gain" is now primary so the established,
    # already-grounded node wins.
    "adm": ["differential voltage gain", "differential-mode gain", "differential gain"],
    "tpd": ["propagation delay", "comparator delay", "delay"],
    "vo": ["dc output offset voltage", "output offset voltage", "offset voltage", "output voltage"],
    # The OTA5T offset-MC and comparator-FPN claim cards both route through metric "vos_v" -> "vos"
    # (projection._metric_text strips the "_v" unit suffix). "Input-Referred Offset Voltage" is the
    # graph's dedicated node for exactly this quantity (description: "The DC offset voltage of an
    # amplifier referred to its input... modeled as a voltage source Vos in series with the input
    # signal"). NOTE: the graph carries an UNRESOLVED duplicate with the identical canonical_name
    # (id 'param_vos', a thin 1-edge stub vs. this node's 11 in/9 out edges incl. HAS_PARAMETER from
    # a CircuitTopology and VARIABLE_MAPS_TO an Equation) — see the G6 dedup audit. The resolver's
    # exact-match tie-break falls back to Neo4j row order for identical-name ties, which is stable
    # for an unchanging graph (verified: 3x repeat resolves to the same id) and happens to prefer
    # this richer node; a G6 merge would remove the duplicate and this ambiguity outright.
    "vos": ["input-referred offset voltage", "offset voltage", "dc offset voltage"],
    # "a vos" (metric "a_vos", projection._metric_text: no unit suffix -> unchanged) is the OTA5T
    # Pelgrom recipe's fitted log-log SCALING EXPONENT of sigma(Vos) vs. device area (see
    # executor.py's "pelgrom" sweep mode) — a dimensionless exponent, NOT a coefficient with units.
    # The graph has no node for this: no "pelgrom" hit at all, and the closest candidates
    # (threshold_voltage_mismatch_coefficient / A_VTH, current_factor_mismatch_coefficient / A_K)
    # are per-transistor mismatch COEFFICIENTS (units V*um / depends-on-Delta(uC_ox*W/L)), a
    # different physical quantity than a whole-circuit fitted exponent — binding "a vos" to either
    # would be a confidently-wrong GROUNDS edge. Left DELIBERATELY uncurated (NO-PHANTOM stays king);
    # it falls through to the gated vector fallback and is correctly refused (top cos 0.583, margin
    # 0.000, both below floor) — see docs/superpowers/specs/2026-07-03-corpus-growth-automation-
    # design.md §6 (G4).
    # The single-slope ramp's slope (dV/dt = I/Cramp, a PROGRAMMED charging rate) is NOT amplifier
    # "slew rate" (a large-signal output-current LIMIT) nor "gain slope" (dB/decade rolloff) — grounding
    # to those is a confidently-wrong edge (NO-PHANTOM can't catch it). The graph has no faithful
    # ramp-slope Parameter, so these faithful phrases resolve to nothing and the claim simply forms no
    # GROUNDS edge (REALIZES still carries the topology grounding) — no edge beats a wrong edge.
    "slope vps": ["ramp slope", "ramp rate", "ramp dv/dt"],
    "acl": ["closed-loop gain", "closed-loop gain ratio", "voltage gain"],
    # broader metrics, for generality beyond the current specimens:
    "gain": ["open-loop gain", "voltage gain"],
    "rout": ["output resistance"],
    "compliance": ["output voltage compliance", "compliance voltage"],
    # Digital (iverilog) metrics (Tier-B): digital-specific so they bind to a faithful digital node or
    # to nothing — never to an analog parameter.
    "diff lsb": ["digital output code", "differenced code", "output code"],
    "hamming": ["hamming distance"],
    "g2b match": ["gray-to-binary decode", "gray to binary", "binary code"],
    "code": ["digital output code", "adc output code", "output code"],
    "diff match": ["digital code difference", "differenced output code"],
    # S3-inc2a W1: swing/ICMR headroom metrics on the two OTA templates (spec §2).
    "vout swing": ["output voltage swing", "output swing"],
    "icmr lo": ["input common-mode range", "lower input common-mode limit", "icmr"],
    "icmr hi": ["input common-mode range", "upper input common-mode limit", "icmr"],
    # S3-inc2a W2: the PTAT/CTAT core's own metrics.
    "vbe": ["base-emitter voltage", "diode voltage", "vbe"],
    "iptat": ["ptat current", "proportional-to-absolute-temperature current"],
}
_VEC_INDEX = {
    NodeLabel.CIRCUIT_TOPOLOGY: "topology_embedding",
    NodeLabel.PARAMETER: "parameter_embedding",
    NodeLabel.CONCEPT: "concept_embedding",
}


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else -1.0


def _expansions(label: NodeLabel, match_text: str) -> list[str]:
    key = match_text.strip().lower()
    table = _TOPOLOGY_EXPANSIONS if label == NodeLabel.CIRCUIT_TOPOLOGY else _METRIC_EXPANSIONS
    return table.get(key, [])


class GraphResolver:
    """Hybrid text-first / embedding-ranked resolver for projection links. Read-only against the
    live graph; returns an existing node's id or None (None -> the projector forms no edge)."""

    def __init__(self, store, encode_fn=None, *, embed_model: str = "Qwen/Qwen3-Embedding-0.6B",
                 vec_threshold: float = 0.60, margin_min: float = 0.03, text_limit: int = 40):
        self._store = store
        self._embed_model = embed_model
        self._encode = encode_fn or self._default_encode
        self._vec_threshold = vec_threshold
        self._margin_min = margin_min       # top-1 must beat the runner-up by this cosine margin
        self._text_limit = text_limit

    def _default_encode(self, text: str) -> list[float]:
        from ..embedding import encode
        return encode(text, model_name=self._embed_model, is_query=True)

    async def _text_candidates(self, label: NodeLabel, phrases: list[str]) -> list[dict]:
        # F7: this LIMIT truncates the CONTAINS match down to a fixed window before either the
        # exact-match tie-break or the embedding-rank step ever sees a row. Without an ORDER BY,
        # which $lim-of-N rows Neo4j returns is an unspecified, order-dependent accident of internal
        # storage/traversal order -- for any hint whose pool exceeds $lim this made candidate
        # VISIBILITY itself nondeterministic (the mechanism that flipped 3 bindings, fixed in
        # 44a4eb4). ORDER BY relationship degree DESC surfaces established, well-integrated nodes
        # first (matching how 44a4eb4's own verdicts were grounded: prefer the node other topologies
        # already HAS_PARAMETER/GROUNDS-connect to over a thin duplicate) -- `size([(n)--()|1])`
        # mirrors store.py's existing degree-ranking idiom (consolidate's merge-survivor pick,
        # store.py:1210), any relationship type/direction, no APOC needed. canonical_name ASC is a
        # second, fully deterministic tiebreak for same-degree rows, replacing row-order luck
        # entirely. This changes WHICH $lim rows are fetched, never how exact-match/embedding-rank
        # choose among whatever rows arrive -- verified live (F7) that all then-flagged near-cap
        # hints keep today's binding under this order.
        idf = _ID_FIELD[label]
        rows = await self._store.run_read_query(
            f"MATCH (n:{label.value}) "
            "WHERE n.canonical_name IS NOT NULL AND n.embedding IS NOT NULL "
            "  AND NOT coalesce(n.retracted, false) "
            "  AND any(p IN $phrases WHERE toLower(n.canonical_name) CONTAINS p) "
            f"RETURN n.canonical_name AS name, n.{idf} AS id, n.embedding AS embedding "
            "ORDER BY size([(n)--()|1]) DESC, n.canonical_name ASC "
            "LIMIT $lim",
            {"phrases": phrases, "lim": self._text_limit},
        )
        return [r for r in rows if r.get("id") and r.get("embedding")]

    async def _vector_topk(self, label: NodeLabel, qvec: list[float], k: int = 25) -> list[dict]:
        """Top-k existing nodes by vector similarity, retracted-filtered, with TRUE cosine. Over-fetch
        (k=25) before the retracted filter so accumulated retracted nodes don't push the real best
        candidate past a tiny LIMIT (the old top-3 had a hidden recall cliff)."""
        idx = _VEC_INDEX.get(label)
        if not idx or not qvec:
            return []
        idf = _ID_FIELD[label]
        rows = await self._store.run_read_query(
            f"CALL db.index.vector.queryNodes('{idx}', $k, $e) YIELD node, score "
            f"WHERE NOT coalesce(node.retracted, false) "
            f"RETURN node.{idf} AS id, score ORDER BY score DESC",
            {"k": k, "e": qvec},
        )
        return [{"id": r["id"], "cos": _vector_score_to_cosine(r["score"])}
                for r in rows if r.get("id")]

    async def resolve(self, label: NodeLabel, match_text: str) -> str | None:
        """Return the existing node id this link should attach to, or None (no confident match)."""
        phrases = _expansions(label, match_text)
        if phrases:
            cands = await self._text_candidates(label, phrases)
            if cands:
                # 1. exact canonical_name == a curated phrase wins outright (most precise, and beats
                #    a verbose context-specific node that embedding would otherwise mis-prefer, e.g.
                #    "Output Current" over "Output Current (I) in Inverter Delay Model"). Prefer the
                #    earliest (primary) phrase, then the shorter name.
                order = {p: i for i, p in enumerate(phrases)}
                exact = [c for c in cands if (c.get("name") or "").strip().lower() in order]
                if exact:
                    exact.sort(key=lambda c: (order[c["name"].strip().lower()], len(c["name"])))
                    return exact[0]["id"]
                # 2. otherwise embedding-rank among the right-KIND candidates, but require a cosine
                #    floor — a text match whose best embedding support is weak should not slip through
                #    unranked (_cosine returns TRUE cosine on the stored vectors, no conversion needed).
                qvec = self._encode(phrases[0])
                best = max(cands, key=lambda n: _cosine(qvec, n["embedding"]))
                if _cosine(qvec, best["embedding"]) >= self._vec_threshold:
                    return best["id"]
        # 3. fallback: no curated text candidate. This is the path the resolver was BUILT to distrust
        #    — pure vector top-1 mis-ranks near-synonyms (av0 -> "DC Loop Gain" @0.921 over the correct
        #    "DC Open-Loop Gain" @0.912). So WARN (an un-curated hint should get a curated expansion to
        #    be deterministic) and gate hard: require a cosine floor AND a margin over the runner-up,
        #    so a confidently-wrong near-tie forms NO edge (I2). (_vector_topk already returns cosine.)
        logger.warning(
            "resolver: un-curated hint %r (%s) hit the vector fallback; add a curated expansion to "
            "make this link deterministic", match_text, label.value)
        hits = await self._vector_topk(label, self._encode(match_text))
        if hits:
            top_cos = hits[0]["cos"]
            runner_up = hits[1]["cos"] if len(hits) > 1 else -1.0
            if top_cos >= self._vec_threshold and (top_cos - runner_up) >= self._margin_min:
                return hits[0]["id"]
            logger.warning("resolver: vector fallback for %r rejected (top cos %.3f, margin %.3f) -> "
                           "no edge", match_text, top_cos, top_cos - runner_up)
        return None
