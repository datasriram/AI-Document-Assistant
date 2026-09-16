from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, load_settings
from app.embeddings import ProviderConfigurationError
from app.factory import build_rag_service
from app.models import AskRequest, AskResponse, HealthResponse, UploadResponse
from app.pdf_loader import PDFExtractionError
from app.rag_pipeline import EmptyIndexError, RagService, RagServiceError


STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None, rag_service: RagService | None = None) -> FastAPI:
    settings = settings or load_settings()

    app = FastAPI(title=settings.app_name, version="1.0.0")
    app.state.settings = settings
    app.state.rag_service = rag_service or build_rag_service(settings)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        document_count, chunk_count = app.state.rag_service.stats()
        return HealthResponse(
            status="ok",
            documents_indexed=document_count,
            chunks_indexed=chunk_count,
            embedding_provider=settings.embedding_provider,
            llm_provider=settings.llm_provider,
        )

    @app.post("/upload", response_model=UploadResponse)
    async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
        if not file.filename:
            raise HTTPException(status_code=400, detail="A filename is required.")

        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")

        if not file_bytes.lstrip().startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail="The uploaded file does not look like a PDF.")

        if len(file_bytes) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"PDF is too large. Maximum size is {settings.max_upload_mb} MB.",
            )

        try:
            return await app.state.rag_service.upload_pdf(file_bytes, file.filename)
        except PDFExtractionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ProviderConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    @app.post("/ask", response_model=AskResponse)
    async def ask(request: AskRequest) -> AskResponse:
        try:
            return await app.state.rag_service.ask(
                question=request.question.strip(),
                top_k=request.top_k,
            )
        except EmptyIndexError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ProviderConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RagServiceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Question answering failed: {exc}") from exc

    return app


app = create_app()
