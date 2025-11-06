FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build deps and curl for healthchecks
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps early for layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only necessary files
COPY train_and_save.py /app/train_and_save.py
COPY app /app/app
COPY content /app/content

# If a prebuilt model exists in the repo, copy it into the image. This path
# is optional in some workflows; presence is convenient for immediate serving.
COPY app/model /app/app/model

# Create a non-root user for security
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Healthcheck for Docker and render.com
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

# Use Gunicorn with Uvicorn workers for production readiness
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "app.main:app", "--workers", "4", "--log-level", "info"]
