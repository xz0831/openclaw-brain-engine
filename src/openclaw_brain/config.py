"""Configuration loader for openclaw-brain."""

from __future__ import annotations

import importlib.resources
import os
from dataclasses import dataclass, field
from pathlib import Path

import tomli
import tomlkit

# Explicit override: an absolute file path to the config TOML. Highest priority
# in the implicit (path=None) resolution order used by load_config()/get_schema_path().
CONFIG_ENV_VAR = "OPENCLAW_BRAIN_CONFIG"
EGRESS_ENV_VAR = "OPENCLAW_EGRESS"


def _xdg_root(env_name: str, fallback: Path) -> Path:
    value = os.environ.get(env_name)
    return Path(value).expanduser() if value else fallback


def brain_config_home() -> Path:
    return _xdg_root(
        "OPENCLAW_BRAIN_CONFIG_HOME", Path.home() / ".config" / "openclaw-brain"
    )


def brain_data_home() -> Path:
    return _xdg_root(
        "OPENCLAW_BRAIN_DATA_HOME", Path.home() / ".local" / "share" / "openclaw-brain"
    )


def brain_state_home() -> Path:
    return _xdg_root(
        "OPENCLAW_BRAIN_STATE_HOME", Path.home() / ".local" / "state" / "openclaw-brain"
    )


@dataclass
class Neo4jConfig:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "openclaw-brain-dev"
    database: str = "neo4j"


@dataclass
class EmbeddingConfig:
    model: str = "Qwen/Qwen3-Embedding-0.6B"  # aligned with TOML SSOT (final-hunt H3: 9-day drift)
    dimensions: int = 1024
    # Instruction prefix for query-side encoding (instruction-asymmetric
    # models like Qwen3-Embedding). Empty = symmetric (MiniLM behavior).
    query_instruction: str = ""
    # MRL truncation target (0 = use the model's native dimension).
    truncate_dim: int = 0


@dataclass
class MatcherConfig:
    """Entity-resolution decision thresholds (embedding-first tiers)."""
    # Cosine above which a candidate auto-matches (Tier 1), guard permitting.
    # Calibrated on the ER gold set with Qwen3-Embedding-0.6B (2026-06-10):
    # the worst false positive (Rolling vs Global Shutter) sits at 0.817,
    # so 0.85 keeps Tier-1 zero-FP; the verify band carries the rest.
    t_high: float = 0.85
    # Cosine (or name-sim) above which a candidate enters the verify band (Tier 2).
    t_low: float = 0.60
    # Max LLM verifications per chunk; overflow falls back to the ambiguous path.
    max_verify_per_chunk: int = 8
    # Feature flag: False reverts to the legacy name-similarity decision.
    use_embedding_decision: bool = True


@dataclass
class ConsolidationConfig:
    """Offline retro-dedup of existing graph nodes."""
    # Cosine above which (with name-sim/alias support and no guard) merge is automatic.
    # LEGACY: superseded by the tiered merge_tier path (cosine is a candidate gate, never an
    # auto-merge decision); kept for back-compat and the review-queue floor.
    t_auto: float = 0.88
    # Max nodes merged into one cluster per run (cascade brake).
    cluster_cap: int = 4
    # Cosine ≥ this routes a non-lexical pair to the VERIFY (LLM) tier. Dedicated knob calibrated
    # on the re-ingested distribution — do NOT reuse matcher.t_low (that is the per-chunk match
    # boundary, a different decision). Below this floor a pair is REJECTed (no verify, no merge).
    candidate_floor: float = 0.82
    # Embedding-cohesion chaining guard: every member of an auto-band cluster (size > 2) must be
    # within this cosine of the cluster medoid, else it is split off. Stops A~B~C transitive
    # over-merge of verifier-SAME semantic pairs (the clustering analogue of all-pairs false-merge).
    cohesion_threshold: float = 0.86


# Allowed values for ModelEntry.status. The catalog ACCRETES — every model ever tried stays in
# it, nothing expires, and until 2026-07-25 nothing recorded whether an entry was still a real
# candidate. `status` is that record: it DOCUMENTS a decision already made elsewhere (a bake-off,
# a cost directive, a protocol incompatibility) and never makes one — nothing promotes or demotes
# a model by editing this field, and the anthropic ban keeps its own provider-keyed mechanism in
# llm/provider.py rather than depending on `banned` being written here.
# It is not INERT, though, and the vocabulary is closed precisely because it is not: a non-active
# entry is disqualified from every fallback chain (`get_chain` skips it, the config-write MCP
# tools refuse to persist it), which is why an unrecognized value must fail loudly instead of
# quietly reading as "not active" — see normalize_model_status below.
MODEL_STATUSES = ("active", "deprecated", "banned", "incompatible")


class ConfigError(ValueError):
    """A config file (or an in-memory config object) is not loadable as written.

    Subclasses ValueError so pre-existing `except ValueError` call sites around config
    construction keep working; the distinct type exists so a bad config is greppable and
    distinguishable from an arbitrary value error raised deeper in the stack.
    """


EGRESS_POLICIES = ("any", "local-only")


def normalize_egress(value: object, source: str = "[deployment].egress") -> str:
    """Normalize an egress policy or fail closed on an unknown value."""
    if not isinstance(value, str):
        raise ConfigError(
            f"{source} must be a string, got {type(value).__name__} ({value!r}). "
            f"Allowed: {', '.join(EGRESS_POLICIES)}"
        )
    normalized = value.strip().lower()
    if normalized not in EGRESS_POLICIES:
        raise ConfigError(
            f"{source} has invalid value {value!r}. "
            f"Allowed: {', '.join(EGRESS_POLICIES)}"
        )
    return normalized


@dataclass
class DeploymentConfig:
    """Deployment-wide data-egress policy.

    ``any`` preserves the historical behavior. ``local-only`` permits network access only
    to loopback HTTP(S) endpoints; enforcement lives in :mod:`openclaw_brain.egress`.
    """

    egress: str = "any"

    def __post_init__(self) -> None:
        self.egress = normalize_egress(self.egress)


def normalize_model_status(value: object, entry_name: str = "") -> str:
    """Normalize a catalog entry's `status`, or raise ConfigError naming the offender.

    Why this is a hard failure rather than a shrug-and-default: `status` is a MECHANICAL
    guard — `LLMProvider.get_chain()` drops every non-active fallback — so an unrecognized
    value ("depricated", 5) silently disqualifies that model from every chain it appears in.
    A guard whose whole point is to end folklore must not fail open: a 3-model chain quietly
    collapsing to 1 because of a typo is exactly the invisible drift this field exists to
    stop. Case/whitespace ARE forgiven ("Active", " active ") because those carry no
    ambiguity about intent; anything else — including a non-string — is rejected loudly.
    """
    if not isinstance(value, str):
        raise ConfigError(
            f"model catalog entry {entry_name or '<unnamed>'!r}: status must be a string, got "
            f"{type(value).__name__} ({value!r}). Allowed: {', '.join(MODEL_STATUSES)}"
        )
    normalized = value.strip().lower()
    if normalized not in MODEL_STATUSES:
        raise ConfigError(
            f"model catalog entry {entry_name or '<unnamed>'!r}: unknown status {value!r}. "
            f"Allowed: {', '.join(MODEL_STATUSES)}. (A non-active status disqualifies the model "
            f"from every fallback chain, so an unrecognized value would silently shorten chains.)"
        )
    return normalized


@dataclass
class ModelEntry:
    """A single model in the catalog.

    `status` (see MODEL_STATUSES) marks an entry's standing, so a dead-but-present entry is
    explicit rather than folklore:
      - ``active``       — a real candidate; the only status that may appear in a fallback chain.
      - ``deprecated``   — superseded or evaluated-off (e.g. lost a bake-off). Still constructible
                           on an explicit `get(name)`, but SKIPPED by `get_chain` with a warning.
      - ``banned``       — policy-blocked (the 2026-07-22 anthropic-API cost ban). The mechanical
                           block lives in `LLMProvider.get()` keyed on provider, not here.
      - ``incompatible`` — cannot work against this repo's routing at all (e.g. gpt-5.4: OpenAI
                           Codex OAuth speaks chatgpt.com/backend-api, not api.openai.com).
    The REASON for a non-active status lives as a one-line comment on the entry in
    config/default.toml (the human channel, and the only one populated today); `status_reason` is
    the optional machine-readable counterpart — when set it is named in the provider's warning.
    """
    name: str = ""
    provider: str = ""          # anthropic | openai | google | local
    model_id: str = ""          # Actual model ID for the API
    tier: str = "frontier"      # frontier | fast | local
    endpoint: str = ""          # Only for local models (Ollama, EXO)
    max_tokens: int | None = None  # Output-token bound (needed for EXO reasoning models)
    status: str = "active"      # active | deprecated | banned | incompatible (MODEL_STATUSES)
    status_reason: str = ""     # Optional one-line reason; the on-disk comment is the human channel

    def __post_init__(self) -> None:
        # Validated HERE rather than only in load_config() so every construction path is
        # covered by one rule: TOML load, the add_catalog_model MCP tool, and test/experiment
        # code all go through ModelEntry(...). Normalizing in place keeps the dataclass the
        # single answer to "what is this entry's status" — no caller has to re-normalize.
        self.status = normalize_model_status(self.status, self.name)


@dataclass
class ModelsConfig:
    # Default model names for pipeline stages (references catalog entries by name)
    default_extraction: str = "deepseek-v4-flash"
    default_reasoning: str = "deepseek-v4-flash"
    default_matching: str = "gemini-3.1-flash-lite"
    default_vision: str = ""  # Vision-capable model for figure classification (empty = disabled)
    default_figure_analysis: str = ""  # Frontier VLM for figure analysis (empty = disabled)

    # Model catalog
    catalog: list[ModelEntry] = field(default_factory=list)

    def get_model(self, name: str) -> ModelEntry | None:
        """Look up a model by name from the catalog."""
        for entry in self.catalog:
            if entry.name == name:
                return entry
        return None

    def list_models(self, tier: str | None = None, provider: str | None = None) -> list[ModelEntry]:
        """List available models, optionally filtered."""
        models = list(self.catalog)
        if tier:
            models = [m for m in models if m.tier == tier]
        if provider:
            models = [m for m in models if m.provider == provider]
        return models


@dataclass
class FiguresConfig:
    """Lecture-slide VLM analysis (figures-only HTML ingest — see
    knowledge/extraction/slide_analyzer.py). Deliberately a SEPARATE section from
    ModelsConfig.default_figure_analysis (the PDF-pipeline figure-VLM default): the slide path's
    primary is LOCAL-first by design (2026-07-17 bake-off, 3 real lecture slides, 3/3 correct
    equation LaTeX + circuit topology ID, 0 hallucination — see config/default.toml [figures]
    comment for the full note) rather than a paid frontier default, reflecting a very different
    volume/cost profile: ~4,000 lecture-slide images across the undergrad HTML corpus vs. a PDF's
    dozens of figures. See slide_analyzer.py's circuit-breaker docstring for why a run-of local
    failures aborts the file instead of bulk-falling-back to the paid model below.
    """
    slide_analysis_model: str = "qwen3-vl-32b"
    slide_analysis_fallback: str = "gemini-3.1-pro"
    schematic_mos_style: str = "razavi"  # specimen->SVG MOSFET symbol style: "razavi" (minimal, operator preference 2026-07-20) | "ieee" (detailed NMos/PMos)


@dataclass
class MemoryConfig:
    episodic_retention_days: int = 90
    semantic_max_results: int = 10
    auto_inject_token_budget: int = 4000
    promotion_interval_hours: int = 24


@dataclass
class ReasoningConfig:
    context_token_budget: int = 8000
    chunk_token_budget: int = 6000
    # Raised 8000 -> 16000 on 2026-07-25 (mirrors config/default.toml): observed max body
    # ~22,700 chars ≈ 5.5-6.5k tokens + reasoning tokens (observed 1,256) ≈ 7.8k total →
    # 8000 clipped real outputs; 16000 = ~2x headroom, still bounds runaway.
    output_token_budget: int = 16000
    max_graph_neighbors: int = 6
    graph_hop_depth: int = 2


@dataclass
class AnswerConfig:
    """Answer feedback; gap logging stores the original question in local state."""

    gap_log: bool = True


@dataclass
class KnowledgeConfig:
    reinforcement_threshold: float = 0.85
    bridge_similarity_low: float = 0.6
    bridge_similarity_high: float = 0.85
    confidence_decay_rate: float = 0.1
    confidence_floor: float = 0.3
    insight_review_days: int = 14
    prune_after_days: int = 30


@dataclass
class LearnerConfig:
    """S5 learner model — UNDERSTANDS-edge confidence decay only (knowledge/learner_decay.py).

    Deliberately its own config section, NOT added to KnowledgeConfig's
    confidence_decay_rate/confidence_floor: those decay a Concept's own truth-confidence
    (the truth layer); this decays how well a LEARNER still remembers something. Reusing one
    config section for both would blur a distinction the code itself keeps structurally
    separate (docs/specs/S5_LEARNER_MODEL_DESIGN.md §3.2). Follows the same "config-driven
    decay rate" pattern as [knowledge] without touching it.
    """
    # Half-life, in days, for UNDERSTANDS.confidence to fade halfway back toward the floor
    # since last_assessed (exponential decay — see apply_understands_decay).
    understands_half_life_days: float = 30.0
    # Floor confidence never crossed by decay (a learner's faded-but-once-real understanding
    # never decays all the way to "never assessed" — mirrors KnowledgeConfig.confidence_floor's
    # role for the truth layer, kept as a separate value here on purpose).
    understands_confidence_floor: float = 0.05


@dataclass
class ResilienceConfig:
    """Retry, backoff, and fallback settings for LLM calls."""
    max_retries: int = 1               # Per-model retry count (see request_timeout_s note)
    initial_backoff_s: float = 1.0     # First retry delay
    max_backoff_s: float = 60.0        # Backoff cap
    backoff_multiplier: float = 2.0    # Exponential factor
    jitter: bool = True                # Add random jitter to backoff
    # Fallback chains per pipeline stage (list of model names, first = primary)
    fallback_extraction: list[str] = field(default_factory=list)
    fallback_reasoning: list[str] = field(default_factory=list)
    fallback_matching: list[str] = field(default_factory=list)
    # Per-request timeout in seconds (0 = no timeout). Aligned with config/default.toml's 120.0.
    # Calibration (2026-07-06, measured): deepseek-v4-flash legitimately takes up to ~66s on DENSE
    # extraction chunks (it emits 30-50 entities — more output = more time; measured in the A/B),
    # so a 60s timeout (the earlier over-correction) CUT legitimate long extractions on exactly the
    # densest/most-valuable chunks -> fail -> retry -> fallback. 120s covers deepseek's real dense-
    # chunk latency with margin; the bad-window protection comes from max_retries (1, per model)
    # so a genuinely-stuck primary fails to the fallback chain in 2x120s instead of 4x60s.
    request_timeout_s: float = 120.0
    # OAuth token refresh
    oauth_refresh_enabled: bool = True
    # Checkpoint/resume
    checkpoint_enabled: bool = True
    # Circuit breaker: once a model has racked up `circuit_breaker_threshold` consecutive
    # failures, later chunks skip straight past it to the fallback chain instead of paying its
    # full per-model retry budget again — until `circuit_breaker_cooldown_s` elapses, at which
    # point exactly one half-open trial is allowed. See llm/resilience.py.
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 3
    circuit_breaker_cooldown_s: float = 120.0


@dataclass
class MinerUConfig:
    """MinerU PDF parsing settings."""
    backend: str = "pipeline"  # TOML SSOT default; code-bearing docs override per S1-W6  # hybrid-auto-engine | vlm-auto-engine | pipeline


@dataclass
class StorageConfig:
    """Harness-neutral filesystem locations owned by openclaw-brain."""

    workspace: str = field(default_factory=lambda: str(brain_data_home() / "workspace"))
    state_dir: str = field(default_factory=lambda: str(brain_state_home()))


# Compatibility for callers that imported the old type name. New code and TOML
# use ``storage``; no default path points into the OpenClaw agent home.
OpenClawConfig = StorageConfig


@dataclass
class ExecutableConfig:
    """Executable-circuit substrate (knowledge/executable/): git-corpus SSOT persistence."""
    # Relative paths are resolved against the repo root (same anchor `config/default.toml` is
    # found at — see `_repo_root()`); absolute paths pass through unchanged.
    corpus_dir: str = "corpus"

    @property
    def corpus_path(self) -> Path:
        p = Path(self.corpus_dir).expanduser()
        return p if p.is_absolute() else _repo_root() / p


@dataclass
class BrainConfig:
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    figures: FiguresConfig = field(default_factory=FiguresConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    answer: AnswerConfig = field(default_factory=AnswerConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    learner: LearnerConfig = field(default_factory=LearnerConfig)
    resilience: ResilienceConfig = field(default_factory=ResilienceConfig)
    mineru: MinerUConfig = field(default_factory=MinerUConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    executable: ExecutableConfig = field(default_factory=ExecutableConfig)

    # Provenance: the file this config was actually loaded from, set by load_config().
    # None for a bare `BrainConfig()` literal. save_config() consults it to (a) round-trip
    # a save back to its origin file and (b) detect a read-only bundled-package source, in
    # which case it requires an explicit target rather than silently failing to write.
    # Excluded from repr/eq so it never affects dataclass comparisons or logging.
    _source: Path | None = field(default=None, repr=False, compare=False, init=False)

    @property
    def workspace_path(self) -> Path:
        return Path(self.storage.workspace).expanduser()

    @property
    def state_path(self) -> Path:
        p = Path(self.storage.state_dir).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def openclaw(self) -> StorageConfig:
        """Deprecated source-compatibility alias; storage is not OpenClaw-owned."""
        return self.storage

    @openclaw.setter
    def openclaw(self, value: StorageConfig) -> None:
        self.storage = value


def _repo_root() -> Path:
    """The repo root — the same anchor `config/default.toml` is resolved against."""
    return Path(__file__).parent.parent.parent


def _repo_config_path() -> Path:
    """The repo-tree config/default.toml — present in a checkout, absent from a wheel."""
    return _repo_root() / "config" / "default.toml"


def _repo_schema_path() -> Path:
    """The repo-tree config/neo4j_schema.cypher — present in a checkout, absent from a wheel."""
    return _repo_root() / "config" / "neo4j_schema.cypher"


def _bundled_data_path(filename: str) -> Path:
    """Resolve a file bundled into the wheel as package data (see pyproject.toml
    [tool.hatch.build.targets.wheel.force-include]), copied there from config/.
    """
    ref = importlib.resources.files("openclaw_brain") / "_data" / filename
    return Path(str(ref))


def _bundled_config_path() -> Path:
    return _bundled_data_path("default.toml")


def _bundled_schema_path() -> Path:
    return _bundled_data_path("neo4j_schema.cypher")


#: Untracked secret file for the live Neo4j password (0600). Introduced with the
#: 2026-08-16 credential rotation: `config/default.toml` is git-tracked (with a GitHub
#: remote), so the real credential must never live there.
NEO4J_PASSWORD_FILE = brain_config_home() / "neo4j-password"


def resolve_neo4j_password(configured: str) -> str:
    """Resolve the effective Neo4j password at DRIVER-CONSTRUCTION time.

    Precedence: `$OPENCLAW_NEO4J_PASSWORD` env > Brain config-home password file
    (stripped; empty file ignored) > the configured TOML value.

    Deliberately NOT applied inside `load_config()`: the config object must only ever
    hold the committed placeholder, because `save_config()` round-trips the object back
    into the git-tracked `config/default.toml` — resolving at load time would leak the
    real credential into the tracked file on the next config-mutation MCP call. The two
    driver call sites (`graph/store.py`, `export/obsidian.py`) call this instead.
    """
    env = os.environ.get("OPENCLAW_NEO4J_PASSWORD")
    if env:
        return env
    try:
        text = NEO4J_PASSWORD_FILE.read_text(encoding="utf-8").strip()
        if text:
            return text
    except OSError:
        pass
    return configured


def load_config(path: str | Path | None = None) -> BrainConfig:
    """Load config from a TOML file.

    Resolution order when `path` is not given:
      1. `$OPENCLAW_BRAIN_CONFIG` env var — an explicit file path.
      2. the repo-tree `config/default.toml` (editable/dev checkout).
      3. the bundled package-data copy shipped inside the wheel.

    Raises FileNotFoundError, listing every location checked, if none exist. An
    explicitly-passed `path` that doesn't exist also raises rather than silently
    returning defaults — callers that actually want pure defaults should construct
    `BrainConfig()` directly instead of routing a missing path through this function.
    """
    searched: list[Path] = []
    if path is not None:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"openclaw-brain config not found at explicitly-requested path: {resolved}"
            )
    else:
        resolved = None
        env_value = os.environ.get(CONFIG_ENV_VAR)
        if env_value:
            env_path = Path(env_value).expanduser()
            searched.append(env_path)
            if env_path.exists():
                resolved = env_path
        if resolved is None:
            for candidate in (_repo_config_path(), _bundled_config_path()):
                searched.append(candidate)
                if candidate.exists():
                    resolved = candidate
                    break
        if resolved is None:
            locations = "\n".join(f"  - {p}" for p in searched)
            raise FileNotFoundError(
                "No openclaw-brain config found. Searched:\n"
                f"{locations}\n"
                f"Set ${CONFIG_ENV_VAR} to an explicit config file, run from a repo "
                "checkout containing config/default.toml, or construct BrainConfig() "
                "directly if you want pure defaults."
            )

    with open(resolved, "rb") as f:
        raw = tomli.load(f)

    cfg = BrainConfig()
    if "deployment" in raw:
        cfg.deployment = DeploymentConfig(**raw["deployment"])
    if "neo4j" in raw:
        cfg.neo4j = Neo4jConfig(**raw["neo4j"])
    if "embedding" in raw:
        cfg.embedding = EmbeddingConfig(**raw["embedding"])
    if "matcher" in raw:
        cfg.matcher = MatcherConfig(**raw["matcher"])
    if "consolidation" in raw:
        cfg.consolidation = ConsolidationConfig(**raw["consolidation"])
    if "models" in raw:
        models_raw = raw["models"]
        try:
            catalog_entries = [
                ModelEntry(**entry) for entry in models_raw.pop("catalog", [])
            ]
        except ConfigError as e:
            # Name the FILE too: load_config searches several candidate paths, so "which
            # default.toml did this come from" is not obvious from the entry name alone.
            raise ConfigError(f"{resolved}: {e}") from e
        cfg.models = ModelsConfig(
            **{k: v for k, v in models_raw.items() if k != "catalog"},
            catalog=catalog_entries,
        )
    if "figures" in raw:
        cfg.figures = FiguresConfig(**raw["figures"])
    if "memory" in raw:
        cfg.memory = MemoryConfig(**raw["memory"])
    if "reasoning" in raw:
        cfg.reasoning = ReasoningConfig(**raw["reasoning"])
    if "answer" in raw:
        cfg.answer = AnswerConfig(**raw["answer"])
    if "knowledge" in raw:
        cfg.knowledge = KnowledgeConfig(**raw["knowledge"])
    if "learner" in raw:
        cfg.learner = LearnerConfig(**raw["learner"])
    if "resilience" in raw:
        cfg.resilience = ResilienceConfig(**raw["resilience"])
    if "mineru" in raw:
        cfg.mineru = MinerUConfig(**raw["mineru"])
    if "storage" in raw and "openclaw" in raw:
        raise ConfigError(f"{resolved}: choose [storage], not both [storage] and legacy [openclaw]")
    if "storage" in raw:
        cfg.storage = StorageConfig(**raw["storage"])
    elif "openclaw" in raw:  # read-only migration compatibility for old external configs
        cfg.storage = StorageConfig(**raw["openclaw"])
    if "executable" in raw:
        cfg.executable = ExecutableConfig(**raw["executable"])

    if EGRESS_ENV_VAR in os.environ:
        cfg.deployment.egress = normalize_egress(
            os.environ[EGRESS_ENV_VAR], f"${EGRESS_ENV_VAR}"
        )

    cfg._source = resolved
    return cfg


def get_schema_path() -> Path:
    """Resolve the neo4j_schema.cypher file, mirroring load_config()'s search order:

      1. a `neo4j_schema.cypher` sibling of `$OPENCLAW_BRAIN_CONFIG`, if that env var
         is set and such a file exists next to it.
      2. the repo-tree `config/neo4j_schema.cypher` (editable/dev checkout).
      3. the bundled package-data copy shipped inside the wheel.

    Raises FileNotFoundError, listing every location checked, if none exist.
    """
    searched: list[Path] = []
    env_value = os.environ.get(CONFIG_ENV_VAR)
    if env_value:
        sibling = Path(env_value).expanduser().parent / "neo4j_schema.cypher"
        searched.append(sibling)
        if sibling.exists():
            return sibling
    for candidate in (_repo_schema_path(), _bundled_schema_path()):
        searched.append(candidate)
        if candidate.exists():
            return candidate
    locations = "\n".join(f"  - {p}" for p in searched)
    raise FileNotFoundError(
        "No neo4j_schema.cypher found. Searched:\n"
        f"{locations}\n"
        f"Set ${CONFIG_ENV_VAR} to a config file whose directory also contains "
        "neo4j_schema.cypher, or run from a repo checkout."
    )


def _default_config_path() -> Path:
    """Return the default config file path."""
    return _repo_root() / "config" / "default.toml"


def _ensure_table(doc: tomlkit.TOMLDocument | tomlkit.items.Table, name: str):
    """Get doc[name], creating it as a fresh (comment-free) table if absent."""
    if name not in doc:
        doc[name] = tomlkit.table()
    return doc[name]


def _patch_kv(table, key: str, value: object) -> None:
    """Set table[key] = value only if the key is missing or its current value
    differs from `value`. Leaving an unchanged key untouched is what makes this
    a patch rather than a rewrite: tomlkit only regenerates the trivia (inline
    comments, whitespace) of items it actually assigns, so a same-value key keeps
    its on-disk formatting byte-for-byte, and a changed-value key keeps its trivia
    (e.g. an inline `# rationale` comment) while only its value token changes.
    """
    if key not in table or table[key] != value:
        table[key] = value


def _patch_model_entry_fields(tbl, entry: "ModelEntry") -> None:
    """Patch one existing [[models.catalog]] table's fields in place (name is the
    match key and is never rewritten). `endpoint`/`max_tokens` are optional fields
    that the writer omits entirely when falsy — mirrored here by deleting the key
    rather than writing an empty/zero placeholder, matching pre-patch behavior.

    `status` follows the same optional-field convention with "active" as its default
    sentinel: it is never ADDED to an entry that is active (so adding the field in
    2026-07 does not rewrite 21 untouched catalog entries), and a stale non-active
    value IS removed when the entry goes back to active. An explicit hand-written
    `status = "active"` on disk is left alone — omitting is the writer's default, not
    a rule the file must obey.
    """
    for key in ("provider", "model_id", "tier"):
        _patch_kv(tbl, key, getattr(entry, key))
    if entry.endpoint:
        _patch_kv(tbl, "endpoint", entry.endpoint)
    elif "endpoint" in tbl:
        del tbl["endpoint"]
    if entry.max_tokens:
        _patch_kv(tbl, "max_tokens", entry.max_tokens)
    elif "max_tokens" in tbl:
        del tbl["max_tokens"]
    if entry.status and entry.status != "active":
        _patch_kv(tbl, "status", entry.status)
    elif "status" in tbl and tbl["status"] != "active":
        del tbl["status"]
    if entry.status_reason:
        _patch_kv(tbl, "status_reason", entry.status_reason)
    elif "status_reason" in tbl:
        del tbl["status_reason"]


def _append_new_model_entry(aot, entry: "ModelEntry") -> None:
    """Append a brand-new [[models.catalog]] table (no prior on-disk formatting to
    preserve — it didn't exist). Adds a trailing blank line after its last field to
    match the one-blank-line-between-entries convention used throughout the file.
    """
    tbl = tomlkit.table()
    tbl["name"] = entry.name
    tbl["provider"] = entry.provider
    tbl["model_id"] = entry.model_id

    fields: list[tuple[str, object]] = [("tier", entry.tier)]
    if entry.endpoint:
        fields.append(("endpoint", entry.endpoint))
    if entry.max_tokens:
        fields.append(("max_tokens", entry.max_tokens))
    if entry.status and entry.status != "active":
        fields.append(("status", entry.status))
    if entry.status_reason:
        fields.append(("status_reason", entry.status_reason))

    for i, (key, val) in enumerate(fields):
        item = tomlkit.item(val)
        if i == len(fields) - 1:
            item.trivia.trail = "\n\n"
        tbl[key] = item

    aot.append(tbl)


def _patch_catalog(models_table, catalog: list["ModelEntry"]) -> None:
    """Reconcile [[models.catalog]] against the incoming BrainConfig's catalog list,
    matching entries by `name` (the identity field for this array-of-tables):
      - matched entries: patch only the fields that changed (comments/formatting on
        an entry whose fields are all unchanged survive untouched, e.g. the
        multi-line rationale block above the claude-sonnet-5 entry).
      - doc entries with no counterpart in `catalog` (removed via remove_catalog_model):
        deleted, taking their own leading comment/trivia with them.
      - `catalog` entries with no counterpart on disk (added via add_catalog_model):
        appended fresh at the end, in incoming order.
    Matched entries keep their on-disk position rather than being reordered to match
    `catalog`'s in-memory order — the 4 MCP config tools only ever append or filter
    the catalog list (both order-preserving relative to disk order), so this never
    diverges from list order in practice, and it avoids ever rebuilding — and thus
    losing the comments on — an entry that didn't change.
    """
    if "catalog" not in models_table:
        models_table["catalog"] = tomlkit.aot()
    aot = models_table["catalog"]

    consumed_idx: set[int] = set()
    matched_names: set[str] = set()
    for entry in catalog:
        idx = None
        for i, tbl in enumerate(aot):
            if i in consumed_idx:
                continue
            if tbl.get("name") == entry.name:
                idx = i
                break
        if idx is not None:
            consumed_idx.add(idx)
            matched_names.add(entry.name)
            _patch_model_entry_fields(aot[idx], entry)

    # Remove stale entries (on disk, absent from `catalog`) highest-index-first so
    # earlier deletions don't shift the indices still queued for removal.
    for i in sorted(range(len(aot)), reverse=True):
        if i not in consumed_idx:
            del aot[i]

    for entry in catalog:
        if entry.name not in matched_names:
            _append_new_model_entry(aot, entry)


def save_config(config: BrainConfig, path: str | Path | None = None) -> None:
    """Patch a BrainConfig's changed values into the TOML file at `path`, preserving
    everything else — comments, key ordering, and unknown/hand-added keys — byte-for-
    byte for any value that didn't actually change.

    Uses tomlkit for a comment-preserving round-trip: the existing file (if any) is
    parsed into a style-aware document, only the fields that differ from what's on
    disk are assigned, and the result is dumped back. This replaced an earlier
    from-scratch serializer that regenerated the whole file from the BrainConfig
    dataclass tree on every call — technically correct but destructive, since it
    silently dropped every hand-written comment (model-selection rationale, etc.)
    regardless of how narrow the actual change was. See `_patch_kv`/`_patch_catalog`
    for how the narrow-update behavior is implemented.

    Resolution for the write target when `path` is not given:
      1. `$OPENCLAW_BRAIN_CONFIG`, if set — always wins so an operator can redirect
         writes even for a config whose `_source` is elsewhere.
      2. `config._source` (wherever this config was actually loaded from), UNLESS
         that source is the read-only bundled package-data copy — round-tripping a
         save back to its origin is the expected default.
      3. the bundled package-data copy with no env var override raises a clear error
         instead of silently dropping the save or failing with an opaque permission
         error — there is nowhere writable to save to.
      4. no recorded source at all (e.g. a bare `BrainConfig()` literal) falls back
         to the repo-tree `config/default.toml`, matching the historical default.
    """
    if path is None:
        env_value = os.environ.get(CONFIG_ENV_VAR)
        if env_value:
            path = Path(env_value).expanduser()
        elif config._source is not None and config._source == _bundled_config_path():
            raise RuntimeError(
                "This config was loaded from the read-only bundled package data "
                f"({config._source}) — there is nowhere writable to save it. "
                f"Set ${CONFIG_ENV_VAR} to a writable file path and try again."
            )
        elif config._source is not None:
            path = config._source
        else:
            path = _default_config_path()
    path = Path(path)

    doc = tomlkit.parse(path.read_text()) if path.exists() else tomlkit.document()

    # Preserve a pre-WP1 config byte-for-byte on a no-op save: the implicit default is
    # ``any``, so an existing non-empty document without [deployment] need not gain a table.
    # New documents, explicit tables, and non-default policy changes are all serialized.
    if "deployment" in doc or config.deployment.egress != "any" or not doc:
        deployment = _ensure_table(doc, "deployment")
        _patch_kv(deployment, "egress", config.deployment.egress)

    neo4j = _ensure_table(doc, "neo4j")
    for k in ("uri", "user", "password", "database"):
        _patch_kv(neo4j, k, getattr(config.neo4j, k))

    models = _ensure_table(doc, "models")
    for k in ("default_extraction", "default_reasoning", "default_matching",
              "default_vision", "default_figure_analysis"):
        _patch_kv(models, k, getattr(config.models, k))
    _patch_catalog(models, config.models.catalog)

    figures = _ensure_table(doc, "figures")
    for k in ("slide_analysis_model", "slide_analysis_fallback"):
        _patch_kv(figures, k, getattr(config.figures, k))

    embedding = _ensure_table(doc, "embedding")
    for k in ("model", "dimensions", "query_instruction", "truncate_dim"):
        _patch_kv(embedding, k, getattr(config.embedding, k))

    matcher = _ensure_table(doc, "matcher")
    for k in ("t_high", "t_low", "max_verify_per_chunk", "use_embedding_decision"):
        _patch_kv(matcher, k, getattr(config.matcher, k))

    consolidation = _ensure_table(doc, "consolidation")
    for k in ("t_auto", "cluster_cap", "candidate_floor", "cohesion_threshold"):
        _patch_kv(consolidation, k, getattr(config.consolidation, k))

    memory = _ensure_table(doc, "memory")
    for k in ("episodic_retention_days", "semantic_max_results",
              "auto_inject_token_budget", "promotion_interval_hours"):
        _patch_kv(memory, k, getattr(config.memory, k))

    reasoning = _ensure_table(doc, "reasoning")
    for k in ("context_token_budget", "chunk_token_budget", "output_token_budget",
              "max_graph_neighbors", "graph_hop_depth"):
        _patch_kv(reasoning, k, getattr(config.reasoning, k))

    # The tracked default.toml intentionally has no [answer] table. Its implicit
    # true default must remain a byte-identical no-op when another setting is saved.
    if "answer" in doc or config.answer.gap_log is not True:
        answer = _ensure_table(doc, "answer")
        _patch_kv(answer, "gap_log", config.answer.gap_log)

    knowledge = _ensure_table(doc, "knowledge")
    for k in ("reinforcement_threshold", "bridge_similarity_low", "bridge_similarity_high",
              "confidence_decay_rate", "confidence_floor", "insight_review_days",
              "prune_after_days"):
        _patch_kv(knowledge, k, getattr(config.knowledge, k))

    learner = _ensure_table(doc, "learner")
    for k in ("understands_half_life_days", "understands_confidence_floor"):
        _patch_kv(learner, k, getattr(config.learner, k))

    resilience = _ensure_table(doc, "resilience")
    for k in ("max_retries", "initial_backoff_s", "max_backoff_s", "backoff_multiplier",
              "jitter", "request_timeout_s", "oauth_refresh_enabled", "checkpoint_enabled",
              "circuit_breaker_enabled", "circuit_breaker_threshold", "circuit_breaker_cooldown_s"):
        _patch_kv(resilience, k, getattr(config.resilience, k))
    for k in ("fallback_extraction", "fallback_reasoning", "fallback_matching"):
        _patch_kv(resilience, k, getattr(config.resilience, k))

    mineru = _ensure_table(doc, "mineru")
    _patch_kv(mineru, "backend", config.mineru.backend)

    if "openclaw" in doc:
        del doc["openclaw"]
    storage = _ensure_table(doc, "storage")
    _patch_kv(storage, "workspace", config.storage.workspace)
    _patch_kv(storage, "state_dir", config.storage.state_dir)

    executable = _ensure_table(doc, "executable")
    _patch_kv(executable, "corpus_dir", config.executable.corpus_dir)

    path.write_text(tomlkit.dumps(doc))
