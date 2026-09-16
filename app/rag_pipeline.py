from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from app.chunking import RecursiveTextChunker
from app.config import Settings
from app.embeddings import EmbeddingProvider
from app.llm import LanguageModel
from app.models import AskResponse, Source, UploadResponse
from app.pdf_loader import PDFExtractionError, extract_text_from_pdf
from app.prompt import SYSTEM_PROMPT, build_rag_prompt
from app.vector_store import InMemoryVectorStore


class RagServiceError(RuntimeError):
    """Base error for user-facing RAG pipeline failures."""


class EmptyIndexError(RagServiceError):
    """Raised when asking a question before any document was indexed."""


class RagService:
    def __init__(
        self,
        settings: Settings,
        embedding_provider: EmbeddingProvider,
        llm: LanguageModel,
        vector_store: InMemoryVectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_provider = embedding_provider
        self.llm = llm
        self.vector_store = vector_store or InMemoryVectorStore()
        self.chunker = RecursiveTextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.upload_dir = settings.upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def upload_pdf(self, file_bytes: bytes, filename: str) -> UploadResponse:
        document_id = uuid4().hex
        safe_name = self._safe_filename(filename)
        pdf_path = self.upload_dir / f"{document_id}_{safe_name}"

        pdf_path.write_bytes(file_bytes)

        try:
            pages = extract_text_from_pdf(pdf_path)
            chunks = self.chunker.chunk_pages(
                pages=pages,
                document_id=document_id,
                filename=safe_name,
            )
            if not chunks:
                raise PDFExtractionError("The PDF did not produce any chunks.")

            embeddings = await self.embedding_provider.embed_texts([chunk.text for chunk in chunks])
            self.vector_store.upsert(chunks, embeddings)
        except Exception:
            pdf_path.unlink(missing_ok=True)
            raise

        return UploadResponse(
            document_id=document_id,
            filename=safe_name,
            pages=len(pages),
            chunks=len(chunks),
            message="PDF uploaded and indexed successfully.",
        )

    async def ask(self, question: str, top_k: int | None = None) -> AskResponse:
        if self.vector_store.is_empty():
            raise EmptyIndexError("Upload at least one PDF before asking questions.")

        effective_top_k = top_k or self.settings.top_k
        effective_top_k = max(1, min(10, effective_top_k))

        query_embedding = await self.embedding_provider.embed_query(question)
        results = self.vector_store.search(query_embedding, effective_top_k)
        rag_prompt = build_rag_prompt(question, results)
        answer = await self.llm.generate(SYSTEM_PROMPT, rag_prompt)

        return AskResponse(
            question=question,
            answer=answer,
            sources=[
                Source(
                    document_id=result.chunk.document_id,
                    filename=result.chunk.filename,
                    page=result.chunk.page,
                    chunk_id=result.chunk.id,
                    score=round(result.score, 4),
                    preview=self._preview(result.chunk.text),
                )
                for result in results
            ],
        )

    def stats(self) -> tuple[int, int]:
        return self.vector_store.document_count(), self.vector_store.chunk_count()

    def _safe_filename(self, filename: str) -> str:
        base = Path(filename).name.strip() or "uploaded.pdf"
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
        if not base.lower().endswith(".pdf"):
            base = f"{base}.pdf"
        return base

    def _preview(self, text: str, max_chars: int = 240) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) <= max_chars:
            return compact
        return f"{compact[: max_chars - 3]}..."
