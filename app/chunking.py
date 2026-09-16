from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha1

from app.pdf_loader import PDFPageText


@dataclass(frozen=True)
class TextChunk:
    id: str
    document_id: str
    filename: str
    page: int
    chunk_index: int
    text: str


class RecursiveTextChunker:
    """Semantic-aware chunker that tries paragraphs, lines, sentences, then words."""

    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 150) -> None:
        if chunk_size < 100:
            raise ValueError("chunk_size must be at least 100 characters")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def chunk_pages(
        self,
        pages: list[PDFPageText],
        document_id: str,
        filename: str,
    ) -> list[TextChunk]:
        chunks: list[TextChunk] = []

        for page in pages:
            cleaned_text = self._clean_text(page.text)
            raw_chunks = self._split_recursively(cleaned_text, separator_index=0)
            overlapped_chunks = self._add_overlap(raw_chunks)

            for text in overlapped_chunks:
                chunk_index = len(chunks)
                stable_suffix = sha1(
                    f"{document_id}:{page.page_number}:{chunk_index}:{text[:80]}".encode()
                ).hexdigest()[:10]
                chunks.append(
                    TextChunk(
                        id=f"{document_id}:p{page.page_number}:c{chunk_index}:{stable_suffix}",
                        document_id=document_id,
                        filename=filename,
                        page=page.page_number,
                        chunk_index=chunk_index,
                        text=text,
                    )
                )

        return chunks

    def _clean_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_recursively(self, text: str, separator_index: int) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        separator = self.separators[separator_index]
        if separator == "":
            return [
                text[start : start + self.chunk_size].strip()
                for start in range(0, len(text), self.chunk_size)
                if text[start : start + self.chunk_size].strip()
            ]

        pieces = text.split(separator)
        chunks: list[str] = []
        current = ""

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue

            candidate = self._join_piece(current, piece, separator)
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current.strip())
                current = ""

            if len(piece) > self.chunk_size:
                chunks.extend(
                    self._split_recursively(piece, min(separator_index + 1, len(self.separators) - 1))
                )
            else:
                current = piece

        if current:
            chunks.append(current.strip())

        return chunks

    def _join_piece(self, current: str, piece: str, separator: str) -> str:
        if not current:
            return piece
        if separator == ". ":
            return f"{current}. {piece}"
        return f"{current}{separator}{piece}"

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        if self.chunk_overlap == 0 or len(chunks) <= 1:
            return chunks

        overlapped: list[str] = []
        previous_tail = ""

        for chunk in chunks:
            combined = f"{previous_tail} {chunk}".strip() if previous_tail else chunk
            overlapped.append(combined)
            previous_tail = self._tail_by_words(chunk)

        return overlapped

    def _tail_by_words(self, text: str) -> str:
        words = text.split()
        tail: list[str] = []
        total_chars = 0

        for word in reversed(words):
            total_chars += len(word) + 1
            if total_chars > self.chunk_overlap:
                break
            tail.append(word)

        return " ".join(reversed(tail))
