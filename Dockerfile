FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --requirement requirements.txt

COPY app/ ./app/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY tools/ ./tools/
COPY tests/ ./tests/
COPY README.md ./

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/db \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
