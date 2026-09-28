"""Unit tests for Evidence Vault hooks in the knowledge pipeline."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openclaw_brain.knowledge.extraction.models import SourceChunkInfo
from openclaw_brain.knowledge.graph.schema import GraphDelta, InsightProposal, NodeLabel
from openclaw_brain.knowledge.pipeline import KnowledgePipeline, _inject_insight_evidence


class FakeGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    async def merge_node(self, **kwargs):
        self.nodes.append(kwargs)
        return kwargs["id_value"]

    async def merge_edge(self, **kwargs):
        self.edges.append(kwargs)


class RecordingVault:
    def __init__(self, sha: str = "abc123"):
        self.sha = sha
        self.texts = []

    def put_text(self, text: str) -> str:
        self.texts.append(text)
        return self.sha


class FailingTextVault:
    def put_text(self, text: str) -> str:
        raise OSError("disk full")


class FailingFileVault:
    def put_file(self, src, kind: str, key: str):
        raise OSError("disk full")

    def put_dir_files(self, src_dir, kind: str, key: str):
        raise OSError("disk full")


def make_pipeline(vault) -> KnowledgePipeline:
    pipeline = KnowledgePipeline.__new__(KnowledgePipeline)
    pipeline._graph = FakeGraph()
    pipeline._vault = vault
    return pipeline


@pytest.mark.asyncio
async def test_register_source_records_model_provenance():
    """Source node is tagged with the extraction/reasoning models it was ingested with."""
    pipeline = make_pipeline(RecordingVault())
    chunked = SimpleNamespace(
        source_id="src_1", title="Paper", author="A. Author", checksum="deadbeef",
    )

    await pipeline._register_source(
        chunked, extraction_model="claude-sonnet-4-6", reasoning_model="claude-sonnet-4-6"
    )

    props = pipeline._graph.nodes[0]["properties"]
    assert props["extraction_model"] == "claude-sonnet-4-6"
    assert props["reasoning_model"] == "claude-sonnet-4-6"
    assert props["source_id"] == "src_1"


@pytest.mark.asyncio
async def test_register_chunk_writes_raw_text_hash():
    pipeline = make_pipeline(RecordingVault("sha_test"))
    chunk = SourceChunkInfo(
        chunk_id="chunk_1",
        source_id="src_1",
        text="full chunk text",
        pages="1-2",
        section_title="Intro",
    )

    await pipeline._register_chunk(chunk)

    props = pipeline._graph.nodes[0]["properties"]
    assert props["raw_text_hash"] == "sha_test"
    assert props["text_preview"] == "full chunk text"
    assert chunk.raw_text_hash == "sha_test"


@pytest.mark.asyncio
async def test_register_chunk_continues_when_vault_write_fails():
    pipeline = make_pipeline(FailingTextVault())
    chunk = SourceChunkInfo(
        chunk_id="chunk_2",
        source_id="src_1",
        text="full chunk text",
    )

    await pipeline._register_chunk(chunk)

    props = pipeline._graph.nodes[0]["properties"]
    assert props["chunk_id"] == "chunk_2"
    assert props["raw_text_hash"] == ""
    assert pipeline._graph.edges[0]["source_label"] == NodeLabel.SOURCE_CHUNK


def test_artifact_vault_failures_do_not_raise(tmp_path):
    pipeline = make_pipeline(FailingFileVault())
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF")

    pipeline._persist_pdf(pdf, "checksum")
    pipeline._persist_mineru_outputs(str(tmp_path), "checksum")


def test_inject_insight_evidence_only_fills_empty_fields():
    chunk = SimpleNamespace(source_id="src_new", chunk_id="chunk_new")
    delta = GraphDelta(
        insights=[
            InsightProposal(statement="needs evidence", related_concept_ids=[]),
            InsightProposal(
                statement="keeps evidence",
                related_concept_ids=[],
                source_id="src_old",
                evidence_chunk_ids=["chunk_old"],
            ),
        ]
    )

    _inject_insight_evidence(delta, chunk)

    assert delta.insights[0].source_id == "src_new"
    assert delta.insights[0].evidence_chunk_ids == ["chunk_new"]
    assert delta.insights[1].source_id == "src_old"
    assert delta.insights[1].evidence_chunk_ids == ["chunk_old"]
