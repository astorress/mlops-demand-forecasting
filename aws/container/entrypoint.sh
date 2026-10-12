#!/bin/sh
# Contrato de SageMaker: el entrenamiento corre la imagen con `train`; los
# endpoints y Batch Transform, con `serve`. Processing sobrescribe el
# entrypoint.
set -e

case "$1" in
  train)
    exec python3 -m src.models.train
    ;;
  serve)
    exec gunicorn src.serving.serve:app \
      --bind "0.0.0.0:${SAGEMAKER_BIND_TO_PORT:-8080}" \
      --workers "${MODEL_SERVER_WORKERS:-2}" \
      --timeout "${MODEL_SERVER_TIMEOUT:-60}" \
      --access-logfile -
    ;;
  *)
    exec "$@"
    ;;
esac
