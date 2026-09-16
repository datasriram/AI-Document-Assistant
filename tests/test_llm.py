import asyncio

from app.llm import LocalContextAnswerer
from app.prompt import build_rag_prompt
from app.chunking import TextChunk
from app.vector_store import SearchResult


def test_local_context_answerer_uses_chunk_text_not_metadata() -> None:
    chunk = TextChunk(
        id="chunk-1",
        document_id="doc-1",
        filename="demo.pdf",
        page=1,
        chunk_index=0,
        text="FastAPI provides upload and ask endpoints.",
    )
    prompt = build_rag_prompt(
        question="What does FastAPI provide?",
        results=[SearchResult(chunk=chunk, score=0.9)],
    )

    answer = asyncio.run(LocalContextAnswerer().generate("system", prompt))

    assert answer == "FastAPI provides upload and ask endpoints. [Source 1]"
    assert "Document:" not in answer
