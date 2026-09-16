import asyncio
from collections.abc import Sequence
from pathlib import Path

from app.chunking import TextChunk
from app.config import Settings
from app.rag_pipeline import RagService
from app.vector_store import InMemoryVectorStore


class TinyEmbeddingProvider:
    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        text = text.lower()
        return [
            1.0 if "fastapi" in text else 0.0,
            1.0 if "docker" in text else 0.0,
        ]


class CapturingLLM:
    def __init__(self) -> None:
        self.user_prompt = ""

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        assert "Answer using only the retrieved context" in system_prompt
        self.user_prompt = user_prompt
        return "FastAPI provides the API layer. [Source 1]"


def test_rag_service_retrieves_context_and_returns_sources(tmp_path: Path) -> None:
    settings = Settings(upload_dir=tmp_path, top_k=1)
    store = InMemoryVectorStore()
    llm = CapturingLLM()
    service = RagService(
        settings=settings,
        embedding_provider=TinyEmbeddingProvider(),
        llm=llm,
        vector_store=store,
    )

    chunk = TextChunk(
        id="chunk-1",
        document_id="doc-1",
        filename="architecture.pdf",
        page=3,
        chunk_index=0,
        text="FastAPI provides the API layer for upload and question answering.",
    )
    store.upsert([chunk], [[1.0, 0.0]])

    response = asyncio.run(service.ask("What does FastAPI do?"))

    assert "FastAPI" in response.answer
    assert response.sources[0].filename == "architecture.pdf"
    assert response.sources[0].page == 3
    assert "[Source 1]" in llm.user_prompt
