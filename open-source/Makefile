# =============================================================================
# MLOps Demand Forecasting - Makefile
# Uso: make <target>
# Todo corre dentro de Docker: no hace falta instalar Python en tu máquina.
# =============================================================================

.PHONY: help setup build up down destroy logs status generate-data validate preprocess train hpo \
	evaluate drift pipeline serve-flask stop-flask serve-mlflow test clean

# Variables de entorno en .envs/config.env (si existe): Compose las usa para
# interpolar docker-compose.yml. Se evalúa en el shell de cada receta, así
# también aplica justo después de que `make setup` crea el archivo.
COMPOSE = docker compose $$([ -f .envs/config.env ] && echo --env-file .envs/config.env)
RUN := $(COMPOSE) run --rm app
MLFLOW_MODEL_NAME ?= demand_forecaster
N_TRIALS ?= 30

## help: Muestra esta ayuda
help:
	@echo "MLOps Demand Forecasting - Comandos disponibles:"
	@echo ""
	@grep -E '^## ' Makefile | sed 's/## /  make /'

## setup: Crea .envs/config.env desde el .example (con tu UID y claves nuevas para Airflow) y las carpetas de datos
setup:
	@echo "==> Configurando entorno..."
	@if [ ! -f .envs/config.env ]; then \
		sed -e "s/^AIRFLOW_UID=.*/AIRFLOW_UID=$$(id -u)/" \
			-e "s|^AIRFLOW__CORE__FERNET_KEY=.*|AIRFLOW__CORE__FERNET_KEY=$$(openssl rand -base64 32 | tr '+/' '-_')|" \
			-e "s|^AIRFLOW__WEBSERVER__SECRET_KEY=.*|AIRFLOW__WEBSERVER__SECRET_KEY=$$(openssl rand -hex 32)|" \
			.envs/config.env.example > .envs/config.env; \
		echo "  [OK] .envs/config.env creado desde .envs/config.env.example (con claves de Airflow nuevas)"; \
	else \
		echo "  [SKIP] .envs/config.env ya existe"; \
	fi
	@mkdir -p data/raw data/processed artifacts
	@echo "  [OK] Carpetas data/ y artifacts/ creadas"
	@echo "==> Setup completado. Siguiente paso: make up"

## build: Construye las imágenes Docker (app y Airflow)
build:
	@$(COMPOSE) build

## up: Levanta PostgreSQL, MLflow, Airflow y JupyterLab y espera a que estén healthy
up: setup
	@echo "==> Levantando servicios (la primera vez construye las imágenes, ~3-5 min)..."
	@$(COMPOSE) up -d --build --wait
	@$(MAKE) --no-print-directory status

## down: Detiene los servicios (conserva datos, runs y modelos)
down:
	@$(COMPOSE) --profile serving down
	@echo "  [OK] Servicios detenidos"

## destroy: Detiene los servicios y BORRA los volúmenes (bases de datos y artefactos de MLflow)
destroy:
	@$(COMPOSE) --profile serving down -v
	@echo "  [OK] Servicios y volúmenes eliminados"

## logs: Muestra logs en tiempo real de todos los servicios
logs:
	@$(COMPOSE) logs -f

## generate-data: Genera el dataset sintético en data/raw/demand.csv
generate-data:
	@$(RUN) python data/generate_dataset.py

## validate: Valida el dataset crudo con Great Expectations
validate:
	@$(RUN) python -m src.data.validation data/raw/demand.csv

## preprocess: Feature engineering + split temporal → data/processed/{train,val,test}.parquet
preprocess:
	@$(RUN) python -m src.data.preprocessing

## train: Entrena el modelo con parámetros por defecto (sin HPO) y lo registra en MLflow
train:
	@$(RUN) python -m src.models.train \
		--data-path data/processed/ \
		--experiment-name demand_forecasting \
		--n-estimators 500 \
		--learning-rate 0.05 \
		--num-leaves 63

## hpo: Ejecuta HPO con Optuna (N_TRIALS=30 por defecto)
hpo:
	@$(RUN) python -m src.models.hpo --data-path data/processed/ --n-trials $(N_TRIALS)

## evaluate: Evalúa un run en test y lo compara con el champion (make evaluate RUN_ID=<id>)
evaluate:
	@test -n "$(RUN_ID)" || (echo "Uso: make evaluate RUN_ID=<mlflow run id>" && exit 1)
	@$(RUN) python -m src.models.evaluate --run-id $(RUN_ID) --model-name $(MLFLOW_MODEL_NAME)

## drift: Reporte de drift con Evidently (train vs test) → artifacts/drift_report.html
drift:
	@$(RUN) python -m src.monitoring.drift_detector

## pipeline: Activa y dispara el DAG mlops_full_pipeline en Airflow
pipeline:
	@$(COMPOSE) exec airflow-scheduler airflow dags unpause mlops_full_pipeline
	@$(COMPOSE) exec airflow-scheduler airflow dags trigger mlops_full_pipeline
	@echo "  [OK] DAG disparado. Síguelo en http://localhost:8080 (admin/admin)"

## serve-flask: Levanta la API Flask en http://localhost:8000 (requiere un modelo en Production)
serve-flask:
	@$(COMPOSE) --profile serving up -d --wait api
	@echo "  [OK] API en http://localhost:8000/health"

## stop-flask: Detiene la API Flask
stop-flask:
	@$(COMPOSE) --profile serving stop api

## serve-mlflow: Sirve el modelo en Production con MLflow en http://localhost:8001
serve-mlflow:
	@$(COMPOSE) run --rm -p 8001:8001 app mlflow models serve \
		--model-uri "models:/$(MLFLOW_MODEL_NAME)/Production" \
		--host 0.0.0.0 \
		--port 8001 \
		--env-manager local

## test: Ejecuta los tests
test:
	@$(RUN) pytest tests/ -v --tb=short

## clean: Borra datos generados, reportes y runs locales (no toca MLflow ni Airflow)
clean:
	@rm -rf data/raw/*.csv data/processed/* artifacts/* mlruns/
	@echo "  [OK] Limpieza completada"

## status: Muestra el estado de los servicios y sus URLs
status:
	@echo ""
	@echo "=============================="
	@echo "  MLOps Demand Forecasting"
	@echo "=============================="
	@$(COMPOSE) ps --format "table {{.Name}}\t{{.Status}}"
	@echo ""
	@echo "  - MLflow UI:   http://localhost:5050"
	@echo "  - Airflow UI:  http://localhost:8080  (admin/admin)"
	@echo "  - JupyterLab:  http://localhost:8888"
	@echo "  - Flask API:   http://localhost:8000  (make serve-flask)"
	@echo ""
