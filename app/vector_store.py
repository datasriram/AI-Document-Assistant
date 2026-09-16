from __future__ import annotations

import math
from dataclasses import dataclass

from app.chunking import TextChunk


@dataclass(frozen=True)
class VectorRecord:
    id: str
    embedding: list[float]
    chunk: TextChunk


@dataclass(frozen=True)
class SearchResult:
    chunk: TextChunk
    score: float


class InMemoryVectorStore:
    """Simple vector store that keeps embeddings in process memory."""

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}
        self._document_ids: set[str] = set()

    def upsert(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            if not embedding:
                raise ValueError("embedding cannot be empty")
            self._records[chunk.id] = VectorRecord(
                id=chunk.id,
                embedding=embedding,
                chunk=chunk,
            )
            self._document_ids.add(chunk.document_id)

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        scored = [
            SearchResult(chunk=record.chunk, score=cosine_similarity(query_embedding, record.embedding))
            for record in self._records.values()
        ]
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]

    def document_count(self) -> int:
        return len(self._document_ids)

    def chunk_count(self) -> int:
        return len(self._records)

    def is_empty(self) -> bool:
        return not self._records


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimension")

    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)
