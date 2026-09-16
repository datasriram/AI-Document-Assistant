from __future__ import annotations

import re
from typing import Protocol

import httpx

from app.config import Settings
from app.embeddings import ProviderConfigurationError


class LanguageModel(Protocol):
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate an answer from a system prompt and RAG user prompt."""


class OpenAIChatLLM:
    def __init__(self, api_key: str | None, base_url: str, model: str) -> None:
        if not api_key:
            raise ProviderConfigurationError("OPENAI_API_KEY is required for OpenAI chat generation.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )

        if response.status_code >= 400:
            raise RuntimeError(f"LLM API error {response.status_code}: {response.text}")

        return response.json()["choices"][0]["message"]["content"].strip()


class LocalContextAnswerer:
    """Small extractive answerer for tests and demos when no external API key exists."""

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        del system_prompt
        context = self._extract_between(user_prompt, "Context:", "Question:")
        question = self._extract_after(user_prompt, "Question:").split("\n", maxsplit=1)[0]

        question_terms = set(re.findall(r"[a-zA-Z0-9]+", question.lower()))
        best_sentence = ""
        best_citation = "[Source 1]"
        best_score = 0

        for source_text, citation in self._source_texts(context):
            for sentence in re.split(r"(?<=[.!?])\s+", source_text):
                terms = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
                score = len(question_terms & terms)
                if score > best_score:
                    best_sentence = sentence.strip()
                    best_citation = citation
                    best_score = score

        if not best_sentence:
            return "I do not know based on the retrieved document context."

        return f"{best_sentence} {best_citation}"

    def _source_texts(self, context: str) -> list[tuple[str, str]]:
        source_texts: list[tuple[str, str]] = []

        for block in context.split("\n\n---\n\n"):
            citation_match = re.search(r"\[Source\s+\d+\]", block)
            citation = citation_match.group(0) if citation_match else "[Source 1]"
            text = block.split("Text:", maxsplit=1)[-1].strip()
            if text:
                source_texts.append((text, citation))

        return source_texts

    def _extract_between(self, text: str, start_marker: str, end_marker: str) -> str:
        start = text.find(start_marker)
        end = text.find(end_marker)
        if start == -1 or end == -1 or end <= start:
            return ""
        return text[start + len(start_marker) : end].strip()

    def _extract_after(self, text: str, marker: str) -> str:
        index = text.find(marker)
        if index == -1:
            return ""
        return text[index + len(marker) :].strip()


def create_llm(settings: Settings) -> LanguageModel:
    if settings.llm_provider == "openai":
        return OpenAIChatLLM(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.llm_model,
        )
    if settings.llm_provider == "local":
        return LocalContextAnswerer()

    raise ProviderConfigurationError(
        f"Unsupported LLM_PROVIDER={settings.llm_provider!r}. Use 'openai' or 'local'."
    )
