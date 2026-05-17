import sys
import types
from dataclasses import dataclass, field
from typing import NamedTuple


@dataclass
class EmbeddedChunk:
    text: str
    metadata: dict
    embedding: list[float] = field(repr=False)


class VectorSearchResult(NamedTuple):
    chunk: EmbeddedChunk
    score: float
    rank: int


class BM25SearchResult(NamedTuple):
    chunk: EmbeddedChunk
    score: float
    rank: int


embedder_stub = types.ModuleType("app.ingestion.embedder")
embedder_stub.EmbeddedChunk = EmbeddedChunk
embedder_stub.get_query_embedding = lambda query: [0.0, 1.0]

vector_store_stub = types.ModuleType("app.retrieval.vector_store")
vector_store_stub.VectorSearchResult = VectorSearchResult
vector_store_stub.VectorStore = object

bm25_store_stub = types.ModuleType("app.retrieval.bm25_store")
bm25_store_stub.BM25SearchResult = BM25SearchResult
bm25_store_stub.BM25Store = object

sys.modules["app.ingestion.embedder"] = embedder_stub
sys.modules["app.retrieval.vector_store"] = vector_store_stub
sys.modules["app.retrieval.bm25_store"] = bm25_store_stub

from app.retrieval.hybrid import reciprocal_rank_fusion


def make_chunk(chunk_id: str) -> EmbeddedChunk:
    return EmbeddedChunk(
        text=f"chunk {chunk_id}",
        metadata={"chunk_id": chunk_id},
        embedding=[0.0, 1.0],
    )


def test_rrf_prioritizes_results_seen_by_both_retrievers():
    shared = make_chunk("shared")
    vector_only = make_chunk("vector-only")
    bm25_only = make_chunk("bm25-only")

    results = reciprocal_rank_fusion(
        faiss_results=[
            VectorSearchResult(chunk=vector_only, score=0.95, rank=1),
            VectorSearchResult(chunk=shared, score=0.90, rank=2),
        ],
        bm25_results=[
            BM25SearchResult(chunk=bm25_only, score=12.0, rank=1),
            BM25SearchResult(chunk=shared, score=10.5, rank=2),
        ],
        top_n=3,
        rrf_k=60,
    )

    assert [result.chunk.metadata["chunk_id"] for result in results] == [
        "shared",
        "bm25-only",
        "vector-only",
    ]
    assert results[0].sources == {"faiss", "bm25"}
    assert results[0].faiss_rank == 2
    assert results[0].bm25_rank == 2


def test_rrf_limits_results_and_assigns_final_ranks():
    results = reciprocal_rank_fusion(
        faiss_results=[
            VectorSearchResult(chunk=make_chunk("a"), score=0.9, rank=1),
            VectorSearchResult(chunk=make_chunk("b"), score=0.8, rank=2),
            VectorSearchResult(chunk=make_chunk("c"), score=0.7, rank=3),
        ],
        bm25_results=[],
        top_n=2,
        rrf_k=60,
    )

    assert [result.final_rank for result in results] == [1, 2]
    assert [result.chunk.metadata["chunk_id"] for result in results] == ["a", "b"]


def test_rrf_skips_chunks_without_stable_ids(caplog):
    missing_id = EmbeddedChunk(text="missing", metadata={}, embedding=[1.0])

    with caplog.at_level("WARNING"):
        results = reciprocal_rank_fusion(
            faiss_results=[
                VectorSearchResult(chunk=missing_id, score=0.9, rank=1),
            ],
            bm25_results=[],
        )

    assert results == []
    assert "missing 'chunk_id'" in caplog.text


def test_rrf_returns_empty_for_empty_inputs():
    assert reciprocal_rank_fusion([], []) == []
