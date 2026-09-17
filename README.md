# AI Document Assistant

A beginner-friendly Retrieval-Augmented Generation (RAG) app built with Python, FastAPI, PyPDF, embeddings, vector similarity search, an external LLM API, and Docker.

The code intentionally keeps each RAG stage visible:

1. PDF upload and validation
2. Text extraction with PyPDF
3. Recursive text chunking with overlap
4. Embedding generation
5. In-memory vector storage
6. Top-k cosine similarity retrieval
7. RAG prompt construction
8. LLM answer generation grounded in retrieved context
9. Source metadata and citations

## Project Structure

```text
app/
  chunking.py       # recursive chunking with overlap
  config.py         # environment-based settings
  embeddings.py     # OpenAI-compatible and local embedding providers
  factory.py        # wires the pipeline together
  llm.py            # OpenAI-compatible and local answer providers
  main.py           # FastAPI routes and static UI
  models.py         # request/response schemas
  pdf_loader.py     # PDF text extraction
  prompt.py         # RAG prompt construction
  rag_pipeline.py   # upload and ask orchestration
  vector_store.py   # cosine similarity vector store
  static/index.html # browser UI
tests/
  test_api.py
  test_chunking.py
  test_pdf_loader.py
  test_rag_pipeline.py
  test_vector_store.py
```

## Local Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

Run the app:

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## External API Configuration

For real LLM answers, configure an OpenAI-compatible API key:

```bash
OPENAI_API_KEY=your_key_here
EMBEDDING_PROVIDER=openai
LLM_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_key_here"
$env:EMBEDDING_PROVIDER="openai"
$env:LLM_PROVIDER="openai"
```

If no API key is set, the app uses deterministic local providers. That mode is useful for tests and demos, but real deployments should use external embedding and LLM services.

## API Endpoints

### `GET /health`

Returns service status and index counts.

### `POST /upload`

Accepts multipart form data with a `file` field containing a PDF.

Example:

```bash
curl -F "file=@sample.pdf" http://127.0.0.1:8000/upload
```

### `POST /ask`

Accepts a question and optional `top_k`.

Example:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What is this document about?\",\"top_k\":4}"
```

## Docker

Build the image:

```bash
docker build -t ai-document-assistant .
```

Run the container:

```bash
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=your_key_here \
  -e EMBEDDING_PROVIDER=openai \
  -e LLM_PROVIDER=openai \
  ai-document-assistant
```

The Docker container starts Uvicorn on `0.0.0.0` and uses Render's `PORT` environment variable when it is available. If `PORT` is not set, it falls back to `8000` for local Docker runs. The container also creates `data/uploads` on startup.

## Deploy on Render

Use Render's Docker web service flow. No database, persistent disk, vector database, LangChain, or authentication is required for this version.

1. Push this repository to GitHub, GitLab, or Bitbucket.
2. In Render, create a new **Web Service** from the repository.
3. Choose **Docker** as the runtime/environment.
4. Keep the Dockerfile path as `./Dockerfile`.
5. Leave the Docker command blank so Render uses the Dockerfile `CMD`.
6. Set the health check path to `/health`.
7. Add the environment variables listed below.
8. Deploy the service.

Recommended Render settings:

```text
Service type: Web Service
Runtime: Docker
Dockerfile path: ./Dockerfile
Docker command: leave empty
Health check path: /health
Auto deploy: optional
```

Required environment variable for real LLM-backed answers:

```text
OPENAI_API_KEY=<set this in Render as a secret value>
```

Optional environment variables:

```text
OPENAI_BASE_URL=https://api.openai.com/v1
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
CHUNK_SIZE=900
CHUNK_OVERLAP=150
TOP_K=4
MAX_UPLOAD_MB=20
UPLOAD_DIR=data/uploads
```

Render provides `PORT` automatically. Do not set it unless you have a specific reason to override Render's default.

The frontend at `/` uses relative API paths such as `/health`, `/upload`, and `/ask`, so it works behind a Render `onrender.com` hostname or a custom domain without code changes.

### Demo Data Limitation

Uploaded PDFs are stored in the container filesystem, and vectors are stored in process memory. This is intentional for the current demo version:

- uploads can disappear on redeploys, restarts, or instance replacement;
- the vector index starts empty after each process restart;
- multiple scaled instances would not share uploaded files or vectors.


## Notes

The vector store is intentionally in memory to keep the core algorithm easy to explain. For larger production use, replace `InMemoryVectorStore` with a persistent vector database such as Postgres/pgvector, Qdrant, Weaviate, or FAISS-backed storage.
