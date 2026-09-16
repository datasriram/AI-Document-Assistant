FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY README.md .

RUN mkdir -p data/uploads

EXPOSE 8000

CMD ["sh", "-c", "mkdir -p \"${UPLOAD_DIR:-data/uploads}\" && uvicorn app.main:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
