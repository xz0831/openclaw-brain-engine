// ============================================================
// OpenClaw Brain — Neo4j Schema
// Knowledge Graph + Agent Memory
// ============================================================

// --- Uniqueness Constraints ---

CREATE CONSTRAINT concept_id IF NOT EXISTS
FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE;

CREATE CONSTRAINT equation_id IF NOT EXISTS
FOR (e:Equation) REQUIRE e.equation_id IS UNIQUE;

CREATE CONSTRAINT principle_id IF NOT EXISTS
FOR (p:Principle) REQUIRE p.principle_id IS UNIQUE;

CREATE CONSTRAINT topology_id IF NOT EXISTS
FOR (t:CircuitTopology) REQUIRE t.topology_id IS UNIQUE;

CREATE CONSTRAINT parameter_id IF NOT EXISTS
FOR (p:Parameter) REQUIRE p.parameter_id IS UNIQUE;

CREATE CONSTRAINT source_id IF NOT EXISTS
FOR (s:Source) REQUIRE s.source_id IS UNIQUE;

CREATE CONSTRAINT chunk_id IF NOT EXISTS
FOR (c:SourceChunk) REQUIRE c.chunk_id IS UNIQUE;

CREATE CONSTRAINT insight_id IF NOT EXISTS
FOR (i:Insight) REQUIRE i.insight_id IS UNIQUE;

CREATE CONSTRAINT assumption_id IF NOT EXISTS
FOR (a:Assumption) REQUIRE a.assumption_id IS UNIQUE;

// --- Memory Node Constraints ---

CREATE CONSTRAINT memory_id IF NOT EXISTS
FOR (m:Memory) REQUIRE m.memory_id IS UNIQUE;

CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE;

CREATE CONSTRAINT session_id IF NOT EXISTS
FOR (s:Session) REQUIRE s.session_id IS UNIQUE;

CREATE CONSTRAINT skill_run_id IF NOT EXISTS
FOR (r:SkillRun) REQUIRE r.run_id IS UNIQUE;

// --- Design Reasoning Node Constraints ---

CREATE CONSTRAINT hypothesis_id IF NOT EXISTS
FOR (h:Hypothesis) REQUIRE h.hypothesis_id IS UNIQUE;

CREATE CONSTRAINT decision_id IF NOT EXISTS
FOR (d:DesignDecision) REQUIRE d.decision_id IS UNIQUE;

CREATE CONSTRAINT bench_id IF NOT EXISTS
FOR (b:BenchResult) REQUIRE b.bench_id IS UNIQUE;

// --- Law-tier graph representation (docs/superpowers/specs/2026-07-04-law-tier-graph-
// representation.md §3) ---

CREATE CONSTRAINT law_id IF NOT EXISTS
FOR (r:Regularity) REQUIRE r.law_id IS UNIQUE;

// --- Indexes for frequent queries ---

CREATE INDEX concept_name IF NOT EXISTS
FOR (c:Concept) ON (c.canonical_name);

CREATE INDEX concept_domain IF NOT EXISTS
FOR (c:Concept) ON (c.domain);

CREATE INDEX equation_type IF NOT EXISTS
FOR (e:Equation) ON (e.equation_type);

CREATE INDEX parameter_symbol IF NOT EXISTS
FOR (p:Parameter) ON (p.symbol);

CREATE INDEX topology_function IF NOT EXISTS
FOR (t:CircuitTopology) ON (t.function);

CREATE INDEX source_title IF NOT EXISTS
FOR (s:Source) ON (s.title);

CREATE INDEX chunk_source IF NOT EXISTS
FOR (c:SourceChunk) ON (c.source_id);

CREATE INDEX memory_type IF NOT EXISTS
FOR (m:Memory) ON (m.memory_type);

CREATE INDEX memory_created IF NOT EXISTS
FOR (m:Memory) ON (m.created_at);

CREATE INDEX entity_type IF NOT EXISTS
FOR (e:Entity) ON (e.entity_type);

CREATE INDEX session_date IF NOT EXISTS
FOR (s:Session) ON (s.date);

CREATE INDEX skill_run_skill IF NOT EXISTS
FOR (r:SkillRun) ON (r.skill_id);

CREATE INDEX hypothesis_status IF NOT EXISTS
FOR (h:Hypothesis) ON (h.status);

CREATE INDEX decision_status IF NOT EXISTS
FOR (d:DesignDecision) ON (d.status);

CREATE INDEX bench_type IF NOT EXISTS
FOR (b:BenchResult) ON (b.bench_type);

// --- Full-Text Index (description search) ---
CREATE FULLTEXT INDEX knowledge_fulltext IF NOT EXISTS
FOR (c:Concept|CircuitTopology|Parameter|Equation|Principle)
ON EACH [c.canonical_name, c.description];

// --- Vector Indexes (Neo4j 5.x native) ---
// 1024d = Qwen/Qwen3-Embedding-0.6B. All five embeddable labels are indexed
// (historically only Concept was, while embeddings were written to all five).
// On a dimension change run: openclaw-brain backfill-embeddings --force --recreate-indexes
CREATE VECTOR INDEX concept_embedding IF NOT EXISTS
FOR (c:Concept) ON (c.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX equation_embedding IF NOT EXISTS
FOR (e:Equation) ON (e.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX principle_embedding IF NOT EXISTS
FOR (p:Principle) ON (p.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX topology_embedding IF NOT EXISTS
FOR (t:CircuitTopology) ON (t.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX parameter_embedding IF NOT EXISTS
FOR (p:Parameter) ON (p.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}};

// ── Learner model + executable-substrate merge keys (2026-07-12, final-hunt H2 finding:
// these four labels had NO uniqueness constraint; latent until Specimen/ClaimCard growth) ──
CREATE CONSTRAINT learner_id_unique IF NOT EXISTS FOR (n:Learner) REQUIRE n.learner_id IS UNIQUE;
CREATE CONSTRAINT assessment_id_unique IF NOT EXISTS FOR (n:Assessment) REQUIRE n.assessment_id IS UNIQUE;
CREATE CONSTRAINT spec_id_unique IF NOT EXISTS FOR (n:Specimen) REQUIRE n.spec_id IS UNIQUE;
CREATE CONSTRAINT claim_id_unique IF NOT EXISTS FOR (n:ClaimCard) REQUIRE n.claim_id IS UNIQUE;

// ── Symbolic-anchor derivation records (created 2026-07-20 via graph-surgery Cypher;
// registered in the schema layer 2026-07-28 after the unregistered label made
// export_graph/import_graph silently drop the anchor structure — see schema.py) ──
CREATE CONSTRAINT derivation_id_unique IF NOT EXISTS FOR (n:SymbolicDerivation) REQUIRE n.derivation_id IS UNIQUE;
