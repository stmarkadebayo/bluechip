FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md Makefile ./
COPY app ./app
COPY ui ./ui
COPY data/sample ./data/sample
COPY eval ./eval
COPY scripts ./scripts
COPY tests ./tests

RUN pip install --no-cache-dir -e ".[dev]" \
    && mkdir -p data/processed runs/eval runs/traces

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).read()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
