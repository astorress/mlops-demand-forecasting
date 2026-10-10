"""
DAG: mlops_full_pipeline

Pipeline MLOps completo para el modelo de predicción de demanda eléctrica.

Schedule: Lunes a las 2AM (0 2 * * 1)
Orden de tasks:
  generate_data → validate_data → preprocess_data → run_hpo →
  train_model → evaluate_model → branch_promote →
    → promote_model (si el nuevo modelo es mejor)
    → skip_promote (si el champion se mantiene)
  → notify_completion
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

# Agregar el directorio del proyecto al path para importar src/
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/opt/mlops-demo")
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default args del DAG
# ---------------------------------------------------------------------------
default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2024, 1, 1),
}

# Variables de entorno con defaults
DATA_RAW_PATH = os.environ.get("DATA_RAW_PATH", f"{PROJECT_ROOT}/data/raw/demand.csv")
DATA_PROCESSED_PATH = os.environ.get("DATA_PROCESSED_PATH", f"{PROJECT_ROOT}/data/processed/")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "demand_forecasting")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "demand_forecaster")

# ---------------------------------------------------------------------------
# Funciones de cada task
# ---------------------------------------------------------------------------

def task_generate_data(**context):
    """Task 1: Genera el dataset sintético de demanda eléctrica."""
    logger.info("Generando dataset de demanda eléctrica...")

    # Importar y ejecutar el generador
    import importlib.util
    generator_path = Path(PROJECT_ROOT) / "data" / "generate_dataset.py"

    spec = importlib.util.spec_from_file_location("generate_dataset", generator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    df = module.generate_dataset(output_path=DATA_RAW_PATH, seed=42)

    logger.info(f"Dataset generado: {len(df):,} registros en {DATA_RAW_PATH}")
    context["ti"].xcom_push(key="n_records", value=len(df))


def task_validate_data(**context):
    """Task 2: Valida el dataset con Great Expectations. Falla si no pasa."""
    import pandas as pd
    from src.data.validation import validate_dataset

    logger.info(f"Validando dataset: {DATA_RAW_PATH}")
    df = pd.read_csv(DATA_RAW_PATH, parse_dates=["timestamp"])

    result = validate_dataset(df)
    logger.info(result["summary"])

    context["ti"].xcom_push(key="validation_passed", value=result["passed"])
    context["ti"].xcom_push(key="validation_summary", value=result["summary"])

    if not result["passed"]:
        raise ValueError(
            f"Validación de datos FALLIDA. "
            f"Checks fallidos: {result.get('failed_checks', [])}. "
            f"Detalle: {result['summary']}"
        )

    logger.info("Validación exitosa. Procediendo con el pipeline.")


def task_preprocess_data(**context):
    """Task 3: Feature engineering y split temporal."""
    from src.data.preprocessing import run_preprocessing

    logger.info("Ejecutando preprocesamiento...")
    paths = run_preprocessing(
        raw_path=DATA_RAW_PATH,
        output_dir=DATA_PROCESSED_PATH,
    )

    logger.info(f"Datos preprocesados guardados: {paths}")
    context["ti"].xcom_push(key="processed_paths", value=paths)


def task_run_hpo(**context):
    """Task 4: HPO con Optuna (20 trials). Pushea los mejores params."""
    import mlflow
    from src.models.hpo import run_hpo

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    logger.info("Iniciando HPO con Optuna (20 trials)...")
    best_params = run_hpo(
        data_path=DATA_PROCESSED_PATH,
        n_trials=20,
        experiment_name=f"{MLFLOW_EXPERIMENT_NAME}_hpo",
    )

    logger.info(f"Mejores parámetros encontrados: {best_params}")
    context["ti"].xcom_push(key="best_params", value=best_params)


def task_train_model(**context):
    """Task 5: Entrena el modelo con los mejores hiperparámetros del HPO."""
    import mlflow
    from src.models.train import train_model

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Obtener parámetros del HPO via XCom
    best_params = context["ti"].xcom_pull(key="best_params", task_ids="run_hpo")
    logger.info(f"Entrenando con params: {best_params}")

    run_id = train_model(
        params=best_params,
        data_path=DATA_PROCESSED_PATH,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )

    logger.info(f"Modelo entrenado. Run ID: {run_id}")
    context["ti"].xcom_push(key="run_id", value=run_id)


def task_evaluate_model(**context):
    """Task 6: Evalúa el modelo en test set y compara con el champion."""
    import mlflow
    from src.models.evaluate import compare_with_champion, evaluate_model

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    run_id = context["ti"].xcom_pull(key="run_id", task_ids="train_model")
    test_path = DATA_PROCESSED_PATH + "test.parquet"

    logger.info(f"Evaluando modelo {run_id} en test set...")
    metrics = evaluate_model(run_id=run_id, test_path=test_path)
    logger.info(f"Test metrics: {metrics}")

    logger.info("Comparando con el modelo champion...")
    is_better = compare_with_champion(
        run_id=run_id,
        model_name=MLFLOW_MODEL_NAME,
        test_path=test_path,
    )

    logger.info(f"¿Nuevo modelo es mejor? {is_better}")
    context["ti"].xcom_push(key="is_better", value=is_better)
    context["ti"].xcom_push(key="test_metrics", value=metrics)


def task_branch_promote(**context):
    """Task 7: BranchPythonOperator — decide si promover el modelo."""
    is_better = context["ti"].xcom_pull(key="is_better", task_ids="evaluate_model")

    if is_better:
        logger.info("Nuevo modelo es mejor. Promoviendo a Production.")
        return "promote_model"
    else:
        logger.info("Champion se mantiene. Saltando promoción.")
        return "skip_promote"


def task_promote_model(**context):
    """Task 8: Registra el modelo en MLflow Registry y lo promueve a Production."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    run_id = context["ti"].xcom_pull(key="run_id", task_ids="train_model")
    model_uri = f"runs:/{run_id}/model"

    logger.info(f"Registrando modelo en el registry: {MLFLOW_MODEL_NAME}")

    # Registrar en el Model Registry
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=MLFLOW_MODEL_NAME,
    )

    version = model_version.version
    logger.info(f"Modelo registrado como versión {version}")

    # Archivar el modelo anterior en Production (si existe)
    try:
        prod_versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
        for old_version in prod_versions:
            client.transition_model_version_stage(
                name=MLFLOW_MODEL_NAME,
                version=old_version.version,
                stage="Archived",
            )
            logger.info(f"Versión {old_version.version} archivada")
    except Exception as e:
        logger.warning(f"No se pudo archivar versión anterior: {e}")

    # Promover nueva versión a Production
    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=version,
        stage="Production",
    )

    # Asignar alias @champion
    try:
        client.set_registered_model_alias(
            name=MLFLOW_MODEL_NAME,
            alias="champion",
            version=version,
        )
        logger.info(f"Alias @champion asignado a versión {version}")
    except Exception as e:
        logger.warning(f"No se pudo asignar alias (requiere MLflow >= 2.8): {e}")

    logger.info(f"Modelo versión {version} promovido a Production exitosamente")
    context["ti"].xcom_push(key="promoted_version", value=version)


def task_notify_completion(**context):
    """Task 10: Imprime un resumen del pipeline en los logs."""
    ti = context["ti"]
    run_date = context["ds"]

    run_id = ti.xcom_pull(key="run_id", task_ids="train_model")
    is_better = ti.xcom_pull(key="is_better", task_ids="evaluate_model")
    test_metrics = ti.xcom_pull(key="test_metrics", task_ids="evaluate_model")
    promoted_version = ti.xcom_pull(key="promoted_version", task_ids="promote_model")
    n_records = ti.xcom_pull(key="n_records", task_ids="generate_data")

    summary = f"""
{'='*60}
  MLOps Pipeline - Resumen de Ejecución
  Fecha: {run_date}
{'='*60}
  Datos generados: {n_records:,} registros
  MLflow Run ID: {run_id}
  Métricas en test: {test_metrics}
  ¿Nuevo modelo mejor que champion? {is_better}
  {'Versión promovida a Production: ' + str(promoted_version) if promoted_version else 'Champion se mantiene'}
  MLflow UI: {MLFLOW_TRACKING_URI}
{'='*60}
"""
    logger.info(summary)
    print(summary)


# ---------------------------------------------------------------------------
# Definición del DAG
# ---------------------------------------------------------------------------

with DAG(
    dag_id="mlops_full_pipeline",
    default_args=default_args,
    schedule="0 2 * * 1",  # Lunes a las 2AM
    catchup=False,
    max_active_runs=1,  # las ejecuciones comparten data/processed: no deben solaparse
    tags=["mlops", "demand_forecasting", "pipeline"],
    doc_md="""
## mlops_full_pipeline

Pipeline MLOps completo para el modelo de predicción de demanda eléctrica horaria.

### Flujo
```
generate_data → validate_data → preprocess_data → run_hpo →
train_model → evaluate_model → branch_promote →
    ├── promote_model (si nuevo modelo es mejor)
    └── skip_promote  (si champion se mantiene)
→ notify_completion
```

### Schedule
Corre automáticamente cada lunes a las 2AM.
También puede dispararse manualmente desde la UI de Airflow.

### Descripción de tasks
| Task | Descripción |
|------|-------------|
| generate_data | Genera el dataset sintético con 26k registros |
| validate_data | Valida calidad de datos con Great Expectations |
| preprocess_data | Feature engineering + split temporal |
| run_hpo | HPO con Optuna (20 trials) |
| train_model | Entrenamiento LightGBM con mejores params |
| evaluate_model | Métricas en test + comparación con champion |
| branch_promote | Decide si promover (BranchPythonOperator) |
| promote_model | Registra en MLflow Registry + Production |
| skip_promote | No hace nada (champion se mantiene) |
| notify_completion | Imprime resumen del pipeline |

### Variables de entorno necesarias
- `PROJECT_ROOT`: raíz del proyecto
- `MLFLOW_TRACKING_URI`: URL del MLflow server
- `MLFLOW_EXPERIMENT_NAME`: nombre del experimento
- `MLFLOW_MODEL_NAME`: nombre del modelo en el registry
""",
) as dag:

    # Task 1: Generar datos
    generate_data = PythonOperator(
        task_id="generate_data",
        python_callable=task_generate_data,
    )

    # Task 2: Validar datos
    validate_data = PythonOperator(
        task_id="validate_data",
        python_callable=task_validate_data,
    )

    # Task 3: Preprocesar datos
    preprocess_data = PythonOperator(
        task_id="preprocess_data",
        python_callable=task_preprocess_data,
    )

    # Task 4: HPO con Optuna
    run_hpo = PythonOperator(
        task_id="run_hpo",
        python_callable=task_run_hpo,
    )

    # Task 5: Entrenar modelo
    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
    )

    # Task 6: Evaluar modelo
    evaluate_model_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=task_evaluate_model,
    )

    # Task 7: Branch
    branch_promote = BranchPythonOperator(
        task_id="branch_promote",
        python_callable=task_branch_promote,
    )

    # Task 8: Promover modelo
    promote_model = PythonOperator(
        task_id="promote_model",
        python_callable=task_promote_model,
    )

    # Task 9: Skip (champion se mantiene)
    skip_promote = EmptyOperator(task_id="skip_promote")

    # Task 10: Notificación final
    notify_completion = PythonOperator(
        task_id="notify_completion",
        python_callable=task_notify_completion,
        trigger_rule="none_failed_min_one_success",
    )

    # Dependencias
    (
        generate_data
        >> validate_data
        >> preprocess_data
        >> run_hpo
        >> train_model_task
        >> evaluate_model_task
        >> branch_promote
        >> [promote_model, skip_promote]
    )
    [promote_model, skip_promote] >> notify_completion
