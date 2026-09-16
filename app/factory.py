from __future__ import annotations

from app.config import Settings
from app.embeddings import create_embedding_provider
from app.llm import create_llm
from app.rag_pipeline import RagService


def build_rag_service(settings: Settings) -> RagService:
    embedding_provider = create_embedding_provider(settings)
    llm = create_llm(settings)
    return RagService(
        settings=settings,
        embedding_provider=embedding_provider,
        llm=llm,
    )
