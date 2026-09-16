from __future__ import annotations

from pydantic import BaseModel, Field


class Source(BaseModel):
    document_id: str
    filename: str
    page: int
    chunk_id: str
    score: float = Field(ge=-1.0, le=1.0)
    preview: str


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int
    message: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    chunks_indexed: int
    embedding_provider: str
    llm_provider: str
