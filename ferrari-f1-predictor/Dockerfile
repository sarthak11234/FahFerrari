# -------------------------------------------------------------------
# Multi-Stage Dockerfile for FahFerrari 2027 Intelligence Platform
# -------------------------------------------------------------------

# Stage 1: Build Frontend (Vite + React 18)
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend (Flask + Gunicorn)
FROM python:3.12-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies including gunicorn for production
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy backend source, models, and data artifacts
COPY api/ ./api/
COPY src/ ./src/
COPY data/ ./data/

# Copy compiled frontend from Stage 1 into the location expected by Flask
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

EXPOSE 5000

# Production execution via Gunicorn
CMD exec gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --threads 4 --timeout 120 api.app:app
