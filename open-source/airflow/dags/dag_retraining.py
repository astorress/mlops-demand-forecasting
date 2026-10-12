"""
DAG: mlops_retraining

DAG de reentrenamiento del modelo, disparado por detección de drift.

DIFERENCIAS con dag_full_pipeline:
- schedule=None: solo se dispara externamente (por dag_monitoring o manualmente)
- No regenera el dataset (los datos ya existen)
- Empieza desde preprocess_data (asume que los datos raw están actualizados)
- Recibe configuración de contexto (quién/qué disparó el reentrenamiento)

El reentrenamiento completo (HPO + train + evaluate + promote) garantiza
que si el modelo se degradó por drift, volvemos a un modelo óptimo para
la distribución actual de los datos.
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/opt/mlops-demo")
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

DATA_RAW_PATH = os.environ.get("DATA_RAW_PATH", f"{PROJECT_ROOT}/data/raw/demand.csv")
DATA_PROCESSED_PATH = os.environ.get("DATA_PROCESSED_PATH", f"{PROJECT_ROOT}/data/processed/")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "demand_forecasting_retrain")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "demand_forecaster")

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2024, 1, 1),
}


def task_log_trigger_info(**context):
    """Task 0: Loguea información sobre quién disparó el reentrenamiento."""
    conf = context.get("dag_run").conf or {}
    triggered_by = conf.get("triggered_by", "manual")
    run_date = conf.get("run_date", context["ds"])

    message = f"""
  ====================================================
  REENTRENAMIENTO INICIADO
  Disparado por: {triggered_by}
  Fecha de referencia: {run_date}
  Experiment: {MLFLOW_EXPERIMENT_NAME}
  ====================================================
"""
    logger.info(message)
    print(message)
    context["ti"].xcom_push(key="triggered_by", value=triggered_by)


# Reutilizamos las mismas funciones del pipeline completo
# DIFERENCIA CLAVE: no regeneramos datos, solo reprocesamos y reentrenamos

def task_preprocess_data(**context):
    """Task 1: Feature engineering y split temporal (reutiliza datos existentes)."""
    from src.data.preprocessing import run_preprocessing

    logger.info("Reprocesando datos existentes...")
    paths = run_preprocessing(
        raw_path=DATA_RAW_PATH,
        output_dir=DATA_PROCESSED_PATH,
    )
    logger.info(f"Datos reprocesados: {paths}")
    context["ti"].xcom_push(key="processed_paths", value=paths)


def task_run_hpo(**context):
    """Task 2: HPO con Optuna (20 trials)."""
    import mlflow
    from src.models.hpo import run_hpo

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info("HPO con Optuna (20 trials)...")

    best_params = run_hpo(
        data_path=DATA_PROCESSED_PATH,
        n_trials=20,
        experiment_name=f"{MLFLOW_EXPERIMENT_NAME}_hpo",
    )

    logger.info(f"Mejores params: {best_params}")
    context["ti"].xcom_push(key="best_params", value=best_params)


def task_train_model(**context):
    """Task 3: Entrenamiento con los mejores hiperparámetros."""
    import mlflow
    from src.models.train import train_model

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    best_params = context["ti"].xcom_pull(key="best_params", task_ids="run_hpo")

    run_id = train_model(
        params=best_params,
        data_path=DATA_PROCESSED_PATH,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )

    logger.info(f"Modelo reentrenado. Run ID: {run_id}")
    context["ti"].xcom_push(key="run_id", value=run_id)


def task_evaluate_model(**context):
    """Task 4: Evaluación y comparación con el champion."""
    import mlflow
    from src.models.evaluate import compare_with_champion, evaluate_model

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    run_id = context["ti"].xcom_pull(key="run_id", task_ids="train_model")
    test_path = DATA_PROCESSED_PATH + "test.parquet"

    metrics = evaluate_model(run_id=run_id, test_path=test_path)
    is_better = compare_with_champion(
        run_id=run_id,
        model_name=MLFLOW_MODEL_NAME,
        test_path=test_path,
    )

    context["ti"].xcom_push(key="is_better", value=is_better)
    context["ti"].xcom_push(key="test_metrics", value=metrics)


def task_branch_promote(**context):
    """Task 5: Decide si promover el modelo reentrenado."""
    is_better = context["ti"].xcom_pull(key="is_better", task_ids="evaluate_model")
    return "promote_model" if is_better else "skip_promote"


def task_promote_model(**context):
    """Task 6: Registra en MLflow Registry y promueve a Production."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    run_id = context["ti"].xcom_pull(key="run_id", task_ids="train_model")

    model_version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=MLFLOW_MODEL_NAME,
    )
    version = model_version.version

    # Archivar versión anterior
    try:
        prod_versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
        for old_v in prod_versions:
            client.transition_model_version_stage(
                name=MLFLOW_MODEL_NAME, version=old_v.version, stage="Archived"
            )
    except Exception as e:
        logger.warning(f"No se pudo archivar versión anterior: {e}")

    # Promover nueva versión
    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME, version=version, stage="Production"
    )

    try:
        client.set_registered_model_alias(name=MLFLOW_MODEL_NAME, alias="champion", version=version)
    except Exception:
        pass

    logger.info(f"Modelo v{version} promovido a Production por reentrenamiento post-drift")
    context["ti"].xcom_push(key="promoted_version", value=version)


def task_notify_retraining(**context):
    """Task final: Resumen del reentrenamiento."""
    ti = context["ti"]
    triggered_by = ti.xcom_pull(key="triggered_by", task_ids="log_trigger_info")
    run_id = ti.xcom_pull(key="run_id", task_ids="train_model")
    is_better = ti.xcom_pull(key="is_better", task_ids="evaluate_model")
    metrics = ti.xcom_pull(key="test_metrics", task_ids="evaluate_model")
    promoted = ti.xcom_pull(key="promoted_version", task_ids="promote_model")

    summary = f"""
  ====================================================
  REENTRENAMIENTO COMPLETADO
  Disparado por: {triggered_by}
  Fecha: {context['ds']}
  Run ID: {run_id}
  Métricas test: {metrics}
  Nuevo modelo mejor: {is_better}
  {'Version promovida: ' + str(promoted) if promoted else 'Champion se mantiene'}
  ====================================================
"""
    logger.info(summary)
    print(summary)


with DAG(
    dag_id="mlops_retraining",
    default_args=default_args,
    schedule=None,  # Solo se dispara externamente
    catchup=False,
    max_active_runs=1,  # las ejecuciones comparten data/processed: no deben solaparse
    tags=["mlops", "retraining", "drift"],
    doc_md="""
## mlops_retraining

DAG de reentrenamiento disparado automáticamente por detección de drift.

**Diferencias con `mlops_full_pipeline`:**
- `schedule=None`: nunca corre automáticamente, solo es disparado por otros DAGs o manualmente
- No regenera el dataset raw
- Comienza desde `preprocess_data` (asume que el CSV de datos existe)
- El experimento MLflow es `demand_forecasting_retrain` (separado del pipeline principal)

### Cuándo se dispara
- Automáticamente: cuando `mlops_monitoring` detecta drift (score > 0.2)
- Manualmente: desde la UI de Airflow o vía REST API

### Configuración via `dag_run.conf`
```json
{
    "triggered_by": "drift_detection",
    "run_date": "2024-03-15"
}
```
""",
) as dag:

    log_trigger_info = PythonOperator(
        task_id="log_trigger_info",
        python_callable=task_log_trigger_info,
    )

    preprocess_data = PythonOperator(
        task_id="preprocess_data",
        python_callable=task_preprocess_data,
    )

    run_hpo = PythonOperator(
        task_id="run_hpo",
        python_callable=task_run_hpo,
    )

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
    )

    evaluate_model_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=task_evaluate_model,
    )

    branch_promote = BranchPythonOperator(
        task_id="branch_promote",
        python_callable=task_branch_promote,
    )

    promote_model = PythonOperator(
        task_id="promote_model",
        python_callable=task_promote_model,
    )

    skip_promote = EmptyOperator(task_id="skip_promote")

    notify_retraining = PythonOperator(
        task_id="notify_retraining",
        python_callable=task_notify_retraining,
        trigger_rule="none_failed_min_one_success",
    )

    (
        log_trigger_info
        >> preprocess_data
        >> run_hpo
        >> train_model_task
        >> evaluate_model_task
        >> branch_promote
        >> [promote_model, skip_promote]
    )
    [promote_model, skip_promote] >> notify_retraining
