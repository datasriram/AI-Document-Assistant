from __future__ import annotations

from app.vector_store import SearchResult


SYSTEM_PROMPT = """You are a careful AI Document Assistant.
Answer using only the retrieved context provided by the user.
If the context does not contain the answer, say you do not know from the provided documents.
Always include source citations like [Source 1] for facts you use."""


def build_rag_prompt(question: str, results: list[SearchResult]) -> str:
    context_blocks = []

    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        context_blocks.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"Document: {chunk.filename}",
                    f"Page: {chunk.page}",
                    f"Chunk ID: {chunk.id}",
                    f"Similarity Score: {result.score:.4f}",
                    "Text:",
                    chunk.text,
                ]
            )
        )

    context = "\n\n---\n\n".join(context_blocks)
    return f"""Context:
{context}

Question: {question}

Write a concise answer grounded only in the context above."""
