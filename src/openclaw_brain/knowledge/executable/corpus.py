"""git-corpus SSOT (SPEC §3) — content-addressed specimen store.

The corpus is the single source of truth: each specimen lives in a content-addressed
directory `specimens/<topology_class>/<spec_id>/` holding the netlist + verdicted
claim-cards + metadata. Neo4j/DuckDB are derived, rebuildable projections of this.

The specimen identity is the NETLIST (spec_id = hash of netlist+tb+role_map+pdk+tool),
NOT the claim-cards — so when a later source mentions the same topology, its claim-cards
MERGE onto the existing specimen (the accretion model; topology_class is the merge key,
content-hash dedups for free — no ER machine). The corpus root is intended to be a git
repo; `commit()` is a thin best-effort wrapper (git is the versioning layer on top).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess

import yaml

from .models import ClaimCard, Specimen

logger = logging.getLogger(__name__)


def _canonical(spec: Specimen) -> str:
    """Deterministic serialization of a specimen's IDENTITY (netlist, not claim-cards)."""
    return "\x00".join([
        spec.netlist,
        spec.testbench,
        json.dumps(spec.role_map, sort_keys=True),
        spec.pdk,
        spec.tool,
    ])


def compute_spec_id(spec: Specimen) -> str:
    return "sha256:" + hashlib.sha256(_canonical(spec).encode()).hexdigest()


def _claim_signature(card: ClaimCard) -> tuple[str, str, str]:
    """The (knob, metric, quant.kind) triple a claim card ASSERTS — its semantic identity, distinct
    from `verdict`/`verdict_note` (which legitimately change on every re-verification of the SAME
    claim) and from `narrative` (prose, may be reworded without changing what's being claimed).
    Mirrors laws.py::ClaimShape's own (topology_class, knob, metric, quant.kind) notion of "what
    makes two claims the same fact" — the established convention elsewhere in this subsystem for
    claim identity that isn't the bare id string."""
    m = card.mechanism
    return (m.knob, m.metric, m.quant.kind)


def _merge_claim_cards(old: list[ClaimCard], new: list[ClaimCard]) -> list[ClaimCard]:
    """Union by claim id — a re-run / later source updates a claim by id and adds new ones.
    Content-hash identity means this never duplicates a specimen; only its claims accrete.

    `id` is the ONLY key this union uses, and nothing upstream (recipe.py's raw-fallback authoring
    path defaults an unset id to a list-POSITION-based string, `f"{topology_class}_claim_{idx}"` —
    see recipe.py::_normalize_claim_card) guarantees it is stable across repeated authoring of the
    same (topology_class, knob, metric) cell. Two DIFFERENT semantic claims can therefore collide on
    the same id (e.g. two authoring calls whose claim-card lists differ in length/order). Silently
    replacing the stored claim in that case would permanently lose a citable fact with zero record of
    what was overwritten (the identity-clobber class) — so a collision is only treated as a
    legitimate update when the (knob, metric, quant.kind) signature is UNCHANGED (the normal
    "re-verify the same claim, verdict/narrative may change" path, e.g. reverify.py's apply flow).
    A signature-changed collision is refused instead: the OLD claim is kept, a loud warning names
    both, and the NEW claim is skipped (never silently applied, never silently dropped from the log).
    """
    by_id = {c.id: c for c in old}
    for c in new:
        prior = by_id.get(c.id)
        if prior is not None and _claim_signature(prior) != _claim_signature(c):
            logger.warning(
                "corpus: claim id %r collision with different content (stored knob/metric/kind=%r, "
                "incoming=%r) — keeping the stored claim, skipping the incoming one.\n"
                "  stored narrative:   %r\n"
                "  incoming narrative: %r",
                c.id, _claim_signature(prior), _claim_signature(c),
                prior.mechanism.narrative, c.mechanism.narrative,
            )
            continue
        by_id[c.id] = c
    return list(by_id.values())


class SpecimenCorpus:
    """Content-addressed specimen store rooted at a (git) directory."""

    def __init__(self, root: str):
        self.root = root

    # ── paths ──
    def _dir(self, topology_class: str, spec_id: str) -> str:
        short = spec_id.split(":", 1)[-1][:16]
        return os.path.join(self.root, "specimens", topology_class, short)

    def spec_dir(self, topology_class: str, spec_id: str) -> str:
        """Public path resolver (same short-hash scheme as `_dir`) — used by `schematic.py` to
        locate a specimen's `meta.yaml`/`cell.spice` without a full `Specimen.model_validate`
        round trip through `claim_cards.yaml` (schematic rendering doesn't need the claim cards)."""
        return self._dir(topology_class, spec_id)

    # ── write ──
    def store(self, spec: Specimen) -> str:
        """Write the specimen; if one with the same spec_id exists, MERGE claim-cards
        onto it (accretion). Returns the specimen directory."""
        spec.spec_id = spec.spec_id or compute_spec_id(spec)
        d = self._dir(spec.topology_class, spec.spec_id)
        if os.path.isfile(os.path.join(d, "meta.yaml")):
            existing = self._load_dir(d)
            spec.claim_cards = _merge_claim_cards(existing.claim_cards, spec.claim_cards)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "cell.spice"), "w") as f:
            f.write(spec.netlist)
        if spec.testbench:
            with open(os.path.join(d, "tb.spice"), "w") as f:
                f.write(spec.testbench)
        # I5: persist the rendered sweep decks so the specimen is self-contained / re-runnable from
        # the corpus alone. These reproduce the canonical data but are NOT part of spec_id.
        if spec.testbenches:
            sweeps_dir = os.path.join(d, "sweeps")
            os.makedirs(sweeps_dir, exist_ok=True)
            for key, deck in spec.testbenches.items():
                with open(os.path.join(sweeps_dir, f"{key}.spice"), "w") as f:
                    f.write(deck)
        with open(os.path.join(d, "claim_cards.yaml"), "w") as f:
            yaml.safe_dump([c.model_dump(mode="json") for c in spec.claim_cards], f, sort_keys=False)
        with open(os.path.join(d, "meta.yaml"), "w") as f:
            yaml.safe_dump({
                "spec_id": spec.spec_id, "topology_class": spec.topology_class,
                "pdk": spec.pdk, "tool": spec.tool, "role_map": spec.role_map,
            }, f, sort_keys=False)
        return d

    # ── read ──
    def _load_dir(self, d: str) -> Specimen:
        with open(os.path.join(d, "meta.yaml")) as f:
            meta = yaml.safe_load(f)
        with open(os.path.join(d, "cell.spice")) as f:
            netlist = f.read()
        tb_path = os.path.join(d, "tb.spice")
        testbench = open(tb_path).read() if os.path.isfile(tb_path) else ""
        sweeps_dir = os.path.join(d, "sweeps")
        testbenches = {}
        if os.path.isdir(sweeps_dir):
            for fn in sorted(os.listdir(sweeps_dir)):
                if fn.endswith(".spice"):
                    testbenches[fn[:-len(".spice")]] = open(os.path.join(sweeps_dir, fn)).read()
        cc_path = os.path.join(d, "claim_cards.yaml")
        cards = yaml.safe_load(open(cc_path)) if os.path.isfile(cc_path) else []
        return Specimen(
            topology_class=meta["topology_class"], netlist=netlist, testbench=testbench,
            testbenches=testbenches,
            pdk=meta.get("pdk", "sky130A"), tool=meta.get("tool", "ngspice-46"),
            role_map=meta.get("role_map", {}) or {},
            claim_cards=[ClaimCard.model_validate(c) for c in (cards or [])],
            spec_id=meta["spec_id"],
        )

    def load(self, topology_class: str, spec_id: str) -> Specimen:
        return self._load_dir(self._dir(topology_class, spec_id))

    def list_class(self, topology_class: str) -> list[str]:
        """spec_ids stored for a topology_class."""
        base = os.path.join(self.root, "specimens", topology_class)
        if not os.path.isdir(base):
            return []
        out = []
        for short in sorted(os.listdir(base)):
            meta = os.path.join(base, short, "meta.yaml")
            if os.path.isfile(meta):
                out.append(yaml.safe_load(open(meta))["spec_id"])
        return out

    def list_classes(self) -> list[str]:
        base = os.path.join(self.root, "specimens")
        return sorted(os.listdir(base)) if os.path.isdir(base) else []

    # ── git (thin, best-effort; the corpus root is the SSOT git repo) ──
    def commit(self, message: str) -> bool:
        try:
            subprocess.run(["git", "-C", self.root, "add", "specimens"], check=True,
                           capture_output=True, timeout=30)
            subprocess.run(["git", "-C", self.root, "commit", "-q", "-m", message], check=True,
                           capture_output=True, timeout=30)
            return True
        except Exception:
            return False
