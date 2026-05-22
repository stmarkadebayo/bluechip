FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml constraints.txt README.md Makefile ./
COPY app ./app
COPY ui ./ui
COPY data/sample ./data/sample
COPY data/deploy/processed ./data/deploy/processed
COPY eval ./eval
COPY scripts ./scripts
COPY tests ./tests

RUN pip install --no-cache-dir -c constraints.txt -e ".[dev]" \
    && mkdir -p data/processed data/deploy/processed runs/eval runs/traces /var/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; port=os.getenv('PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health', timeout=3).read()"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
