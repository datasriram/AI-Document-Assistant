from app.chunking import RecursiveTextChunker
from app.pdf_loader import PDFPageText


def test_recursive_chunker_preserves_metadata_and_overlap() -> None:
    text = (
        "FastAPI is used for the web API. "
        "PyPDF extracts text from uploaded documents. "
        "Embeddings convert chunks into vectors. "
        "Similarity search finds relevant context. "
    ) * 8

    chunker = RecursiveTextChunker(chunk_size=180, chunk_overlap=45)
    chunks = chunker.chunk_pages(
        pages=[PDFPageText(page_number=2, text=text)],
        document_id="doc-1",
        filename="notes.pdf",
    )

    assert len(chunks) > 1
    assert all(chunk.document_id == "doc-1" for chunk in chunks)
    assert all(chunk.filename == "notes.pdf" for chunk in chunks)
    assert all(chunk.page == 2 for chunk in chunks)
    assert chunks[0].text != chunks[1].text
    assert "Similarity search" in chunks[1].text or "Embeddings convert" in chunks[1].text
