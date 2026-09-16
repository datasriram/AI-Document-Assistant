from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

import httpx

from app.config import Settings


class EmbeddingProvider(Protocol):
    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text."""

    async def embed_query(self, text: str) -> list[float]:
        """Return one vector for a user query."""


class ProviderConfigurationError(RuntimeError):
    """Raised when an external provider is selected but not configured."""


class OpenAIEmbeddingProvider:
    def __init__(self, api_key: str | None, base_url: str, model: str) -> None:
        if not api_key:
            raise ProviderConfigurationError("OPENAI_API_KEY is required for OpenAI embeddings.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = {"model": self.model, "input": list(texts)}
        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
            )

        if response.status_code >= 400:
            raise RuntimeError(f"Embedding API error {response.status_code}: {response.text}")

        data = response.json()["data"]
        data.sort(key=lambda item: item["index"])
        return [item["embedding"] for item in data]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_texts([text]))[0]


class HashingEmbeddingProvider:
    """Deterministic local embeddings for tests and no-key development."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.embedding_model,
        )
    if settings.embedding_provider == "local":
        return HashingEmbeddingProvider()

    raise ProviderConfigurationError(
        f"Unsupported EMBEDDING_PROVIDER={settings.embedding_provider!r}. Use 'openai' or 'local'."
    )
