from app.chunking import TextChunk
from app.vector_store import InMemoryVectorStore, cosine_similarity


def make_chunk(chunk_id: str, text: str) -> TextChunk:
    return TextChunk(
        id=chunk_id,
        document_id="doc-1",
        filename="demo.pdf",
        page=1,
        chunk_index=0,
        text=text,
    )


def test_cosine_similarity_orders_vectors() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0


def test_vector_store_returns_top_k_matches() -> None:
    store = InMemoryVectorStore()
    store.upsert(
        chunks=[
            make_chunk("a", "about APIs"),
            make_chunk("b", "about cooking"),
            make_chunk("c", "about embeddings"),
        ],
        embeddings=[
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
        ],
    )

    results = store.search([1.0, 0.0], top_k=2)

    assert [result.chunk.id for result in results] == ["a", "c"]
    assert store.document_count() == 1
    assert store.chunk_count() == 3
