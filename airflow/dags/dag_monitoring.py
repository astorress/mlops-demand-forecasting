"""
DAG: mlops_monitoring

Monitoreo diario de data drift del modelo de demanda eléctrica.
Compara los últimos 7 días de datos contra el training set (referencia).
Si se detecta drift, dispara automáticamente el DAG de reentrenamiento.

Schedule: Diario a las 8AM (0 8 * * *)
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/opt/mlops-demo")
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

DATA_PROCESSED_PATH = os.environ.get("DATA_PROCESSED_PATH", f"{PROJECT_ROOT}/data/processed/")
DATA_RAW_PATH = os.environ.get("DATA_RAW_PATH", f"{PROJECT_ROOT}/data/raw/demand.csv")
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "0.2"))

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2024, 1, 1),
}


def task_load_recent_data(**context):
    """Task 1: Carga los últimos 7 días de datos del CSV crudo."""
    logger.info("Cargando datos recientes (últimos 7 días)...")

    df = pd.read_csv(DATA_RAW_PATH, parse_dates=["timestamp"])

    # Obtener los últimos 7 días
    cutoff = df["timestamp"].max() - timedelta(days=7)
    recent_df = df[df["timestamp"] > cutoff].copy()

    logger.info(f"Datos recientes: {len(recent_df):,} registros desde {cutoff}")

    # Guardar temporalmente
    recent_path = f"{DATA_PROCESSED_PATH}recent_data.parquet"
    Path(DATA_PROCESSED_PATH).mkdir(parents=True, exist_ok=True)
    recent_df.to_parquet(recent_path, index=False)

    context["ti"].xcom_push(key="recent_path", value=recent_path)
    context["ti"].xcom_push(key="n_recent_records", value=len(recent_df))


def task_run_drift_detection(**context):
    """Task 2: Ejecuta la detección de drift con Evidently AI."""
    from src.monitoring.drift_detector import run_drift_report

    recent_path = context["ti"].xcom_pull(key="recent_path", task_ids="load_recent_data")
    reference_path = f"{DATA_PROCESSED_PATH}train.parquet"

    logger.info(f"Ejecutando detección de drift...")
    logger.info(f"  Referencia: {reference_path}")
    logger.info(f"  Datos actuales: {recent_path}")

    reference_df = pd.read_parquet(reference_path)
    current_df = pd.read_parquet(recent_path)

    run_date = context["ds"].replace("-", "")
    output_path = f"{PROJECT_ROOT}/artifacts/drift_report_{run_date}.html"

    result = run_drift_report(
        reference_df=reference_df,
        current_df=current_df,
        output_path=output_path,
        drift_threshold=DRIFT_THRESHOLD,
    )

    logger.info(f"Resultado drift: {result}")

    context["ti"].xcom_push(key="drift_detected", value=result["drift_detected"])
    context["ti"].xcom_push(key="drift_score", value=result["drift_score"])
    context["ti"].xcom_push(key="drifted_features", value=result.get("drifted_features", []))
    context["ti"].xcom_push(key="drift_report_path", value=output_path)


def task_check_drift_threshold(**context):
    """Task 3: BranchPythonOperator — decide si dispara reentrenamiento."""
    drift_detected = context["ti"].xcom_pull(key="drift_detected", task_ids="run_drift_detection")
    drift_score = context["ti"].xcom_pull(key="drift_score", task_ids="run_drift_detection")
    drifted_features = context["ti"].xcom_pull(key="drifted_features", task_ids="run_drift_detection")

    logger.info(f"Drift score: {drift_score} | Threshold: {DRIFT_THRESHOLD}")
    logger.info(f"Features drifteadas: {drifted_features}")

    if drift_detected:
        logger.warning(f"DRIFT DETECTADO! Score: {drift_score:.3f}. Disparando reentrenamiento.")
        return "trigger_retraining"
    else:
        logger.info(f"Sin drift significativo (score: {drift_score:.3f}). Todo bien.")
        return "log_ok"


def task_log_ok(**context):
    """Task 5: Loguea que no hay drift y todo funciona correctamente."""
    drift_score = context["ti"].xcom_pull(key="drift_score", task_ids="run_drift_detection")
    n_records = context["ti"].xcom_pull(key="n_recent_records", task_ids="load_recent_data")

    message = f"""
  Monitoreo {context['ds']}: Sin drift detectado
  - Registros analizados: {n_records:,}
  - Drift score: {drift_score:.4f}
  - Threshold: {DRIFT_THRESHOLD}
  - Estado: OK - El modelo sigue siendo valido
"""
    logger.info(message)
    print(message)


with DAG(
    dag_id="mlops_monitoring",
    default_args=default_args,
    schedule="0 8 * * *",  # Diario a las 8AM
    catchup=False,
    max_active_runs=1,  # las ejecuciones comparten data/processed: no deben solaparse
    tags=["mlops", "monitoring", "drift"],
    doc_md="""
## mlops_monitoring

Monitoreo diario de data drift.

Compara los últimos 7 días de datos de producción contra el training set.
Si se detecta drift significativo (score > 0.2), dispara automáticamente
el DAG `mlops_retraining`.

### Flujo
```
load_recent_data → run_drift_detection → check_drift_threshold →
    ├── trigger_retraining (si drift > threshold)
    └── log_ok            (si todo está bien)
```

### Schedule
Corre automáticamente todos los días a las 8AM.

### Configuración
- `DRIFT_THRESHOLD`: umbral PSI (default: 0.2)
- Un PSI > 0.2 indica drift severo que requiere reentrenamiento
""",
) as dag:

    load_recent_data = PythonOperator(
        task_id="load_recent_data",
        python_callable=task_load_recent_data,
    )

    run_drift_detection = PythonOperator(
        task_id="run_drift_detection",
        python_callable=task_run_drift_detection,
    )

    check_drift_threshold = BranchPythonOperator(
        task_id="check_drift_threshold",
        python_callable=task_check_drift_threshold,
    )

    # Dispara el DAG de reentrenamiento cuando se detecta drift
    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="mlops_retraining",
        conf={"triggered_by": "drift_detection", "run_date": "{{ ds }}"},
        wait_for_completion=False,  # No esperar a que termine el reentrenamiento
    )

    log_ok = PythonOperator(
        task_id="log_ok",
        python_callable=task_log_ok,
    )

    (
        load_recent_data
        >> run_drift_detection
        >> check_drift_threshold
        >> [trigger_retraining, log_ok]
    )
