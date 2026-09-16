from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models import AskResponse, Source, UploadResponse


class FakeRagService:
    def stats(self) -> tuple[int, int]:
        return (1, 2)

    async def upload_pdf(self, file_bytes: bytes, filename: str) -> UploadResponse:
        assert file_bytes
        return UploadResponse(
            document_id="doc-1",
            filename=filename,
            pages=1,
            chunks=2,
            message="PDF uploaded and indexed successfully.",
        )

    async def ask(self, question: str, top_k: int | None = None) -> AskResponse:
        assert top_k == 2
        return AskResponse(
            question=question,
            answer="The document discusses FastAPI. [Source 1]",
            sources=[
                Source(
                    document_id="doc-1",
                    filename="demo.pdf",
                    page=1,
                    chunk_id="chunk-1",
                    score=0.98,
                    preview="FastAPI is used for the API layer.",
                )
            ],
        )


def test_health_endpoint() -> None:
    app = create_app(Settings(), rag_service=FakeRagService())
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["documents_indexed"] == 1


def test_static_frontend_uses_relative_api_paths() -> None:
    app = create_app(Settings(), rag_service=FakeRagService())
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "AI Document Assistant" in response.text
    assert 'fetch("/health")' in response.text
    assert 'fetch("/upload"' in response.text
    assert 'fetch("/ask"' in response.text
    assert "localhost" not in response.text


def test_upload_rejects_non_pdf() -> None:
    app = create_app(Settings(), rag_service=FakeRagService())
    client = TestClient(app)

    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400


def test_upload_accepts_pdf_filename() -> None:
    app = create_app(Settings(), rag_service=FakeRagService())
    client = TestClient(app)

    response = client.post(
        "/upload",
        files={"file": ("notes.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["chunks"] == 2


def test_ask_endpoint() -> None:
    app = create_app(Settings(), rag_service=FakeRagService())
    client = TestClient(app)

    response = client.post("/ask", json={"question": "What is used?", "top_k": 2})

    assert response.status_code == 200
    assert response.json()["sources"][0]["page"] == 1
