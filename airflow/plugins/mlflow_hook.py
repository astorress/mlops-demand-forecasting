"""
Plugin de Airflow: MLflowHook

Hook personalizado que permite a los DAGs de Airflow conectarse
y logear información a un servidor MLflow.

Uso en DAGs:
    from plugins.mlflow_hook import MLflowHook

    hook = MLflowHook(mlflow_conn_id="mlflow_default")
    with hook.get_run(experiment_name="my_experiment") as run:
        hook.log_metric("accuracy", 0.95)
"""

import logging
from typing import Optional

from airflow.hooks.base import BaseHook

logger = logging.getLogger(__name__)


class MLflowHook(BaseHook):
    """
    Hook de Airflow para interactuar con MLflow Tracking Server.

    Permite a las tasks de Airflow loguear parámetros, métricas
    y artefactos a MLflow como parte de su ejecución.

    Configuración en Airflow Admin > Connections:
    - Conn ID: mlflow_default
    - Conn Type: HTTP
    - Host: mlflow-server (o localhost si es local)
    - Port: 5000
    """

    conn_name_attr = "mlflow_conn_id"
    default_conn_name = "mlflow_default"
    conn_type = "http"
    hook_name = "MLflow"

    def __init__(self, mlflow_conn_id: str = "mlflow_default") -> None:
        super().__init__()
        self.mlflow_conn_id = mlflow_conn_id
        self._client = None

    def get_conn(self) -> str:
        """Retorna la URI de tracking de MLflow."""
        try:
            conn = self.get_connection(self.mlflow_conn_id)
            scheme = conn.schema or "http"
            host = conn.host or "localhost"
            port = conn.port or 5000
            return f"{scheme}://{host}:{port}"
        except Exception:
            import os
            return os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")

    def get_mlflow_client(self):
        """Retorna un cliente MlflowClient configurado."""
        import mlflow
        from mlflow.tracking import MlflowClient

        tracking_uri = self.get_conn()
        mlflow.set_tracking_uri(tracking_uri)

        if self._client is None:
            self._client = MlflowClient(tracking_uri=tracking_uri)

        return self._client

    def log_airflow_task_metrics(
        self,
        run_id: str,
        dag_id: str,
        task_id: str,
        execution_date: str,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """
        Loguea metadatos de la ejecución de Airflow como tags en un run de MLflow.

        Útil para trazar la cadena Airflow task → MLflow run para auditoría.

        Args:
            run_id: ID del run MLflow donde loguear.
            dag_id: ID del DAG de Airflow.
            task_id: ID de la task de Airflow.
            execution_date: Fecha de ejecución del DAG.
            duration_seconds: Duración de la task en segundos.
        """
        client = self.get_mlflow_client()

        client.set_tag(run_id, "airflow.dag_id", dag_id)
        client.set_tag(run_id, "airflow.task_id", task_id)
        client.set_tag(run_id, "airflow.execution_date", execution_date)

        if duration_seconds is not None:
            client.log_metric(run_id, "airflow.task_duration_seconds", duration_seconds)

        logger.info(f"Metadatos de Airflow logueados en MLflow run {run_id}")

    def get_experiment_id(self, experiment_name: str) -> str:
        """
        Obtiene o crea un experimento MLflow por nombre.

        Args:
            experiment_name: Nombre del experimento.

        Returns:
            ID del experimento.
        """
        import mlflow

        mlflow.set_tracking_uri(self.get_conn())

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Experimento creado: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Experimento encontrado: {experiment_name} (ID: {experiment_id})")

        return experiment_id

    def get_production_model_info(self, model_name: str) -> Optional[dict]:
        """
        Obtiene información del modelo en stage Production.

        Args:
            model_name: Nombre del modelo en el Registry.

        Returns:
            Dict con info del modelo, o None si no existe.
        """
        client = self.get_mlflow_client()

        try:
            versions = client.get_latest_versions(model_name, stages=["Production"])
            if versions:
                v = versions[0]
                return {
                    "name": v.name,
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "creation_timestamp": v.creation_timestamp,
                }
        except Exception as e:
            logger.warning(f"No se pudo obtener info del modelo {model_name}: {e}")

        return None
