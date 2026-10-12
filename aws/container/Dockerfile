# Imagen del modelo (BYOC): la misma imagen corre todas las cargas de
# SageMaker del proyecto, así el código y las versiones son idénticos en todas.
#   Processing  → ContainerEntrypoint ["python3", "-m", "src.pipelines.training.<paso>"]
#   Training    → `docker run <imagen> train`
#   Endpoint /
#   Transform   → `docker run <imagen> serve` (HTTP en :8080, /ping + /invocations)
# Contexto de build: la carpeta del proyecto (aws/). La
# construye y publica container/build-push.sh; el tag sale de
# container/image-version.js.
FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/program \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OpenMP: LightGBM lo necesita en tiempo de ejecución.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY container/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Solo los paquetes del modelo (sin Lambdas ni scripts): /opt/program/src/...
WORKDIR /opt/program
COPY src/__init__.py src/__init__.py
COPY src/features/ src/features/
COPY src/models/ src/models/
COPY src/monitoring/ src/monitoring/
COPY src/pipelines/ src/pipelines/
COPY src/serving/ src/serving/
COPY container/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["entrypoint.sh"]
