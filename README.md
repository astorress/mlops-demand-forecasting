# MLOps Demand Forecasting — Predicción de Demanda Eléctrica

Proyecto MLOps end-to-end que muestra el ciclo de vida completo de un modelo de Machine Learning con herramientas open source: desde la generación y validación de datos hasta el serving y el monitoreo de drift con reentrenamiento automático.

Todo corre en Docker con un solo comando (`make up`): PostgreSQL, MLflow, Airflow y JupyterLab.

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────────┐
│                  MLOps Demand Forecasting Stack                     │
├───────────────┬───────────────┬────────────────┬────────────────────┤
│   Datos       │  Tracking     │  Orquestación  │  Serving &         │
│               │               │                │  Monitoreo         │
│ ┌──────────┐  │ ┌──────────┐  │ ┌────────────┐ │ ┌───────────────┐  │
│ │ Dataset  │  │ │ MLflow   │  │ │ Airflow    │ │ │ Flask API     │  │
│ │ sintético│  │ │ Tracking │  │ │ DAGs       │ │ │ /predict      │  │
│ └──────────┘  │ └──────────┘  │ └────────────┘ │ └───────────────┘  │
│ ┌──────────┐  │ ┌──────────┐  │ ┌────────────┐ │ ┌───────────────┐  │
│ │ Great    │  │ │ MLflow   │  │ │ Pipeline   │ │ │ MLflow Serve  │  │
│ │ Expect.  │  │ │ Registry │  │ │ completo   │ │ │ (alternativa) │  │
│ └──────────┘  │ └──────────┘  │ └────────────┘ │ └───────────────┘  │
│ ┌──────────┐  │               │ ┌────────────┐ │ ┌───────────────┐  │
│ │ LightGBM │  │               │ │ Monitoreo +│ │ │ Evidently AI  │  │
│ │ + Optuna │  │               │ │ Reentreno  │ │ │ Drift Report  │  │
│ └──────────┘  │               │ └────────────┘ │ └───────────────┘  │
└───────────────┴───────────────┴────────────────┴────────────────────┘
                        │                 │
                   PostgreSQL        PostgreSQL
               (MLflow backend)    (Airflow metadata)
               + volumen Docker
               (artefactos MLflow)
```

### Pipeline (DAG `mlops_full_pipeline`)

```
generate_data → validate_data → preprocess_data → run_hpo → train_model → evaluate_model
    → branch_promote ─┬─ promote_model  (el nuevo modelo supera al champion → Production + @champion)
                      └─ skip_promote   (el champion se mantiene)
    → notify_completion
```

El DAG `mlops_monitoring` compara a diario los últimos 7 días contra el training set y, si detecta drift, dispara `mlops_retraining`.

## Stack tecnológico

| Componente | Tecnología | Puerto |
|---|---|---|
| Experiment tracking y registry | MLflow 2.17 | 5050 |
| Orquestación | Apache Airflow 2.10 | 8080 |
| Notebooks | JupyterLab | 8888 |
| Base de datos | PostgreSQL 15 | interno |
| Modelo | LightGBM | — |
| HPO | Optuna | — |
| Validación de datos | Great Expectations 0.18 | — |
| Monitoreo de drift | Evidently AI 0.4 | — |
| API REST | Flask + Pydantic | 8000 |

> MLflow se publica en el **5050** porque en macOS el puerto 5000 lo ocupa AirPlay Receiver. Todos los puertos se pueden cambiar en `.env`.

## Quick start

### Prerrequisitos

- Docker Desktop ≥ 24 (con Docker Compose v2), con al menos 4 CPU y 6 GB de RAM asignados
- Make

No hace falta Python local: todos los comandos corren dentro de contenedores.

### 1. Levantar la infraestructura

```bash
git clone https://github.com/astorress/mlops-demand-forecasting.git
cd mlops-demand-forecasting
make up
```

La primera vez construye las imágenes (~3-5 min) y espera a que todos los servicios estén *healthy*.

| URL | Servicio | Credenciales |
|---|---|---|
| http://localhost:5050 | MLflow UI | — |
| http://localhost:8080 | Airflow UI | admin / admin |
| http://localhost:8888 | JupyterLab | — |

### 2. Correr el pipeline completo

```bash
make pipeline
```

Síguelo en Airflow (http://localhost:8080). Al terminar (~1-2 min) verás el modelo `demand_forecaster` en **Production** en la pestaña *Models* de MLflow.

### 3. Servir el modelo

```bash
make serve-flask

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature_c": 24.5, "hour": 19, "day_of_week": 2, "month": 7, "is_weekend": 0,
       "is_holiday": 0, "demand_lag_24h": 1450, "demand_lag_168h": 1420, "demand_rolling_mean_7d": 1300}'
# {"prediction": 1164.03, "unit": "kWh", ...}
```

Guía completa paso a paso: [docs/quickstart.md](docs/quickstart.md).

## Comandos

| Comando | Qué hace |
|---|---|
| `make up` / `make down` | Levanta / detiene los servicios (`down` conserva datos y modelos) |
| `make destroy` | Detiene y borra los volúmenes (bases de datos y artefactos de MLflow) |
| `make status` / `make logs` | Estado de los contenedores / logs en tiempo real |
| `make generate-data` | Genera el dataset sintético en `data/raw/demand.csv` |
| `make validate` | Valida el dataset con Great Expectations |
| `make preprocess` | Feature engineering + split temporal → `data/processed/` |
| `make train` | Entrena LightGBM y lo registra en MLflow |
| `make hpo` | HPO con Optuna (`N_TRIALS=30` por defecto) |
| `make evaluate RUN_ID=<id>` | Métricas en test + comparación con el champion |
| `make drift` | Reporte de drift de Evidently en `artifacts/` |
| `make pipeline` | Activa y dispara el DAG `mlops_full_pipeline` |
| `make serve-flask` / `make stop-flask` | API Flask en el puerto 8000 |
| `make serve-mlflow` | MLflow Serving del modelo en Production en el puerto 8001 |
| `make test` | Tests con pytest |
| `make clean` | Borra datos generados y reportes |

## Notebooks

Abre JupyterLab (http://localhost:8888) y recorre `notebooks/` en orden. Los runs que crean aparecen en la UI de MLflow.

| Notebook | Tema |
|---|---|
| 00_dataset_exploration | EDA del dataset |
| 01_feature_engineering | Feature engineering y training-serving skew |
| 02_experiment_tracking_mlflow | MLflow tracking |
| 03_hpo_optuna | HPO con Optuna |
| 04_model_registry_mlflow | Model Registry: stages y aliases |
| 05_serving_flask | Serving con Flask |
| 06_serving_mlflow | Serving con MLflow |
| 07_airflow_pipeline_walkthrough | Pipeline en Airflow |
| 08_monitoring_drift | Monitoreo con Evidently |

## Estructura del proyecto

```
mlops-demand-forecasting/
├── data/
│   ├── generate_dataset.py      # Generador de datos sintéticos
│   ├── raw/                     # demand.csv (generado)
│   └── processed/               # train/val/test.parquet (generados)
├── src/
│   ├── data/
│   │   ├── validation.py        # Validación con Great Expectations
│   │   └── preprocessing.py     # Feature engineering y split temporal
│   ├── models/
│   │   ├── train.py             # Entrenamiento LightGBM + MLflow
│   │   ├── evaluate.py          # Métricas y comparación con el champion
│   │   └── hpo.py               # HPO con Optuna
│   ├── serving/
│   │   ├── flask_app.py         # API REST Flask
│   │   ├── mlflow_serve.md      # Guía de MLflow Serving
│   │   └── sagemaker_deploy.py  # Deploy opcional en AWS SageMaker
│   └── monitoring/
│       └── drift_detector.py    # Detección de drift con Evidently
├── airflow/
│   ├── dags/
│   │   ├── dag_full_pipeline.py # Pipeline completo (semanal)
│   │   ├── dag_monitoring.py    # Monitoreo de drift (diario)
│   │   └── dag_retraining.py    # Reentrenamiento disparado por drift
│   └── plugins/
│       └── mlflow_hook.py       # Hook Airflow → MLflow
├── docker/
│   ├── app.Dockerfile           # MLflow server, Jupyter, API y comandos del Makefile
│   ├── airflow.Dockerfile       # Airflow con las librerías del pipeline
│   └── postgres/init-db.sql     # Crea la base de datos de MLflow
├── notebooks/                   # 9 notebooks explicativos
├── tests/                       # Tests de datos, modelos, monitoreo y serving
├── docs/                        # Quickstart y manuales de las UIs
├── docker-compose.yml
├── Makefile
├── requirements-pipeline.txt    # Dependencias del pipeline (Airflow + app)
├── requirements.txt             # + serving, visualización y AWS
└── requirements-dev.txt         # + Jupyter, pytest y linters
```

## Dataset

Dataset sintético de demanda eléctrica horaria (2022-2024):

- **26,136 registros**: 3 años horarios (2024 es bisiesto) menos la primera semana, que se descarta por los lags de 168 h
- **Target**: `demand_kwh`, con tendencia, estacionalidad diaria, semanal y anual, y ruido
- **Features**: temperatura, calendario (hora, día, mes, fin de semana, festivos colombianos), lags de 24 h y 168 h, media móvil de 7 días y encodings cíclicos

Split temporal (sin mezclar el futuro con el pasado): train 2022-2023, validación 1.er semestre de 2024, test 2.º semestre de 2024.

Las features no se escalan: LightGBM es un modelo de árboles y no lo necesita, así que el modelo trabaja en kWh reales y la API recibe los valores tal cual, sin riesgo de *training-serving skew*. Con los parámetros por defecto el modelo obtiene un **MAPE de ~3.9 %** en test.

## Documentación

- [Quickstart](docs/quickstart.md)
- [Manual de la UI de MLflow](docs/manual_mlflow_ui.md)
- [Manual de la UI de Airflow](docs/manual_airflow_ui.md)
- [MLflow Serving](src/serving/mlflow_serve.md)

## Licencia

MIT
