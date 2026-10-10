# Imagen de la aplicación: servidor MLflow, Jupyter, API Flask y comandos del Makefile.
FROM python:3.11-slim

# curl: healthchecks · libgomp1: lo necesita LightGBM
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-pipeline.txt requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt psycopg2-binary==2.9.10

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1
