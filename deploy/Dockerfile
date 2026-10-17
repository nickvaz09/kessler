# Kessler - application image.
#
# Build from the REPO ROOT (the build context must contain backend/ and
# frontend/):
#   docker build -f deploy/Dockerfile -t kessler .
# or via docker compose (see deploy/docker-compose.yml).
#
# The app runs from /app, which mirrors the repo-root layout that
# backend/app/config.py derives its paths from (data/, frontend/).
FROM python:3.13-slim

# libgomp1: OpenMP runtime required by the xgboost wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
# Trained risk-model artifact (small); the sqlite DB and training data are
# excluded by .dockerignore. Seeds fresh /app/data volumes.
COPY data/models ./data/models

# SQLite database location in the no-env default mode; harmless under
# Postgres (KESSLER_DATABASE_URL) where only data/models is read from it.
VOLUME /app/data

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
