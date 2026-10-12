# Airflow con las librerías del pipeline instaladas en la imagen, para que el
# scheduler y el webserver puedan importar src/.
# Variante slim: sin providers de terceros que choquen con las dependencias de ML.
FROM apache/airflow:slim-2.10.5-python3.11

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow

COPY requirements-pipeline.txt /tmp/requirements-pipeline.txt
# Se fija apache-airflow para que pip no lo actualice al resolver dependencias.
RUN pip install --no-cache-dir "apache-airflow[postgres]==2.10.5" -r /tmp/requirements-pipeline.txt
